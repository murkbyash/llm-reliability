"""Unified root cause diagnostic engine orchestrating multi-layer analysis."""

import uuid
from pathlib import Path
from typing import Any, TextIO

from llm_reliability.agent.analyzer import AgentAnalyzer
from llm_reliability.agent.models import AgentMetrics
from llm_reliability.grounding.analyzer import GroundingAnalyzer
from llm_reliability.grounding.enums import SupportStatus
from llm_reliability.grounding.models import GroundingMetrics
from llm_reliability.models.diagnosis import (
    Diagnosis,
    Evidence,
    Failure,
    Hypothesis,
    Metric,
)
from llm_reliability.models.enums import (
    EvidenceType,
    FailureCategory,
    Severity,
    SpanKind,
)
from llm_reliability.models.execution import RetrievalStep, RetrievedDocument
from llm_reliability.models.trace import Run, Trace
from llm_reliability.multimodal.analyzer import MultiModalReliabilityAnalyzer
from llm_reliability.normalization.loader import load_trace
from llm_reliability.plugins.manager import PluginManager, get_plugin_manager
from llm_reliability.rag.analyzer import RAGAnalyzer
from llm_reliability.rag.metrics import RetrievalMetrics
from llm_reliability.recommendations.engine import RecommendationEngine


class DiagnosticEngine:
    """Orchestrates RAG, grounding, agent trajectory, and LLM diagnostic evaluations."""

    def __init__(
        self,
        rag_analyzer: RAGAnalyzer | None = None,
        grounding_analyzer: GroundingAnalyzer | None = None,
        agent_analyzer: AgentAnalyzer | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        plugin_manager: PluginManager | None = None,
        multimodal_analyzer: MultiModalReliabilityAnalyzer | None = None,
    ) -> None:
        """Initialize diagnostic engine with optional custom sub-analyzers and plugin manager."""
        self.rag_analyzer = rag_analyzer or RAGAnalyzer()
        self.grounding_analyzer = grounding_analyzer or GroundingAnalyzer()
        self.agent_analyzer = agent_analyzer or AgentAnalyzer()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.plugin_manager = plugin_manager or get_plugin_manager()
        self.multimodal_analyzer = multimodal_analyzer or MultiModalReliabilityAnalyzer()

    def diagnose(
        self,
        source: Trace | Run | str | Path | dict[str, Any] | list[Any] | TextIO,
    ) -> Diagnosis:
        """Diagnose a trace, run, file path, JSON string, stream, or dict.

        Returns a unified Diagnosis object with ranked recommendations.
        """
        if isinstance(source, Run):
            return self.diagnose_run(source)

        if isinstance(source, Trace):
            trace = source
        else:
            trace = load_trace(source)

        return self.diagnose_trace(trace)

    def diagnose_trace(self, trace: Trace) -> Diagnosis:
        """Diagnose an entire multi-run trace, synthesizing findings into a top-level Diagnosis."""
        trace = self.plugin_manager.apply_pre_middlewares(trace)

        if not trace.runs:
            return Diagnosis(
                trace_id=trace.trace_id,
                root_cause=FailureCategory.NONE,
                primary_category=FailureCategory.NONE,
                severity=Severity.INFO,
                summary="Empty trace with no execution runs.",
            )

        if len(trace.runs) == 1:
            return self.diagnose_run(trace.runs[0])

        run_diagnoses = [self.diagnose_run(run) for run in trace.runs]

        all_metrics: list[Metric] = []
        all_evidence: list[Evidence] = []
        all_failures: list[Failure] = []
        all_hypotheses: list[Hypothesis] = []

        for d in run_diagnoses:
            all_metrics.extend(d.metrics)
            all_evidence.extend(d.evidence)
            all_failures.extend(d.failures)
            all_hypotheses.extend(d.hypotheses)

        severity_order = [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]
        highest_severity = Severity.INFO
        for sev in severity_order:
            if any(d.severity == sev for d in run_diagnoses):
                highest_severity = sev
                break

        ranked_hypotheses = self._rank_hypotheses(all_hypotheses)
        primary_category = (
            ranked_hypotheses[0].category if ranked_hypotheses else FailureCategory.NONE
        )

        failed_count = sum(1 for d in run_diagnoses if d.primary_category != FailureCategory.NONE)
        summary = (
            f"Analyzed {len(run_diagnoses)} runs: {failed_count} run(s) with detected issues. "
            f"Primary failure category: {primary_category.value}."
            if failed_count > 0
            else f"Analyzed {len(run_diagnoses)} runs: All runs executed successfully with zero detected reliability failures."
        )

        diag = Diagnosis(
            trace_id=trace.trace_id,
            root_cause=primary_category,
            primary_category=primary_category,
            severity=highest_severity,
            summary=summary,
            failures=all_failures,
            hypotheses=ranked_hypotheses,
            metrics=all_metrics,
            evidence=all_evidence,
        )

        # Generate prioritized recommendations
        diag.recommendations = self.recommendation_engine.generate_recommendations(diag)
        return self.plugin_manager.apply_post_middlewares(diag)

    def diagnose_run(self, run: Run) -> Diagnosis:
        """Perform comprehensive root cause diagnosis on a single execution Run."""
        collected_metrics: list[Metric] = []
        collected_evidence: list[Evidence] = []
        failures: list[Failure] = []
        candidate_hypotheses: list[Hypothesis] = []

        # 1. RAG Retrieval Analysis
        retrieval_steps = run.get_retrievals()
        all_docs: list[RetrievedDocument] = []
        rag_metrics: RetrievalMetrics | None = None

        if retrieval_steps:
            for step in retrieval_steps:
                all_docs.extend(step.documents)
            combined_step = RetrievalStep(
                query=run.input_query or (retrieval_steps[0].query if retrieval_steps else ""),
                documents=all_docs,
            )
            rag_metrics = self.rag_analyzer.analyze_retrieval_step(combined_step)
            collected_metrics.extend(rag_metrics.to_diagnostic_metrics())

            if rag_metrics.is_empty_retrieval:
                ev = Evidence(
                    evidence_type=EvidenceType.RETRIEVAL_SCORE,
                    description="Retriever returned 0 documents.",
                    supporting_data={"retrieved_count": 0},
                )
                collected_evidence.append(ev)
                failures.append(
                    Failure(
                        failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                        category=FailureCategory.RETRIEVAL_FAILURE,
                        severity=Severity.HIGH,
                        title="Empty Retrieval Result",
                        description="Retrieval step returned zero documents for the prompt query.",
                        message="Retrieval step returned zero documents for the prompt query.",
                        evidence=[ev],
                    )
                )
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.RETRIEVAL_FAILURE,
                        title="Retriever Failed to Return Documents",
                        description="The vector search / retriever returned an empty document candidate set.",
                        rationale="The vector search / retriever returned an empty document candidate set.",
                        confidence=0.95,
                        evidence=[ev],
                        contributing_factors=[
                            "Embedding mismatch",
                            "Overly restrictive query filter",
                            "Empty index",
                        ],
                    )
                )
            elif rag_metrics.is_low_relevance:
                ev = Evidence(
                    evidence_type=EvidenceType.RETRIEVAL_SCORE,
                    description=f"Low relevance retrieval: max score {rag_metrics.max_score} below threshold.",
                    supporting_data={
                        "max_score": rag_metrics.max_score,
                        "mean_score": rag_metrics.mean_score,
                    },
                )
                collected_evidence.append(ev)
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.RETRIEVAL_FAILURE,
                        title="Low Relevance of Retrieved Context",
                        description="Retrieved documents exhibited weak semantic relevance scores relative to query.",
                        rationale="Retrieved documents exhibited weak semantic relevance scores relative to query.",
                        confidence=0.80,
                        evidence=[ev],
                        contributing_factors=[
                            "Suboptimal embedding model",
                            "Query phrasing divergence",
                        ],
                    )
                )

            if rag_metrics.is_high_duplicate_ratio:
                ev = Evidence(
                    evidence_type=EvidenceType.TOKEN_DISCREPANCY,
                    description=f"High redundancy in retrieved documents: {rag_metrics.duplicate_count} duplicates found.",
                    supporting_data={
                        "duplicate_count": rag_metrics.duplicate_count,
                        "duplicate_ratio": rag_metrics.duplicate_ratio,
                    },
                )
                collected_evidence.append(ev)
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.CONTEXT_CONSTRUCTION_FAILURE,
                        title="Context Redundancy and Duplication",
                        description="Retrieved documents contained high duplicate overlap, wasting LLM context budget.",
                        rationale="Retrieved documents contained high duplicate overlap, wasting LLM context budget.",
                        confidence=0.75,
                        evidence=[ev],
                        contributing_factors=[
                            "Lack of retrieval deduplication",
                            "Repetitive chunks in knowledge base",
                        ],
                    )
                )

        # 2. Grounding / Answer Support Analysis
        response_text = run.final_response.text if run.final_response else None
        if not response_text:
            llm_calls = run.get_llm_calls()
            if llm_calls and llm_calls[-1].response:
                response_text = llm_calls[-1].response

        grounding_metrics: GroundingMetrics | None = None
        if response_text and (all_docs or retrieval_steps):
            grounding_metrics = self.grounding_analyzer.analyze_response(response_text, all_docs)
            collected_metrics.extend(grounding_metrics.to_diagnostic_metrics())
            grounding_evidence = grounding_metrics.to_evidence()
            collected_evidence.extend(grounding_evidence)

            if grounding_metrics.status == SupportStatus.CONTRADICTED:
                failures.append(
                    Failure(
                        failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                        category=FailureCategory.HALLUCINATION,
                        severity=Severity.HIGH,
                        title="Factual Contradiction in Response",
                        description="The generated response explicitly contradicts facts stated in the retrieved context.",
                        message="The generated response explicitly contradicts facts stated in the retrieved context.",
                        evidence=grounding_evidence,
                    )
                )
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.HALLUCINATION,
                        title="Model Generation Contradicts Retrieved Knowledge",
                        description="Response contains assertions directly conflicting with context evidence.",
                        rationale="Response contains assertions directly conflicting with context evidence.",
                        confidence=0.90,
                        evidence=grounding_evidence,
                        contributing_factors=[
                            "Model parametric memory bias",
                            "Ambiguous prompt instructions",
                        ],
                    )
                )
            elif grounding_metrics.status == SupportStatus.UNSUPPORTED:
                failures.append(
                    Failure(
                        failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                        category=FailureCategory.GROUNDING_FAILURE,
                        severity=Severity.HIGH,
                        title="Ungrounded Response / Hallucination",
                        description="The generated response contains claims not supported by retrieved context documents.",
                        message="The generated response contains claims not supported by retrieved context documents.",
                        evidence=grounding_evidence,
                    )
                )
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.GROUNDING_FAILURE,
                        title="Ungrounded Claims in LLM Output",
                        description="Model fabricated statements and entities unsupported by retrieved context.",
                        rationale="Model fabricated statements and entities unsupported by retrieved context.",
                        confidence=0.85,
                        evidence=grounding_evidence,
                        contributing_factors=[
                            "Insufficient context grounding",
                            "Hallucinated named entities",
                        ],
                    )
                )
            elif grounding_metrics.status == SupportStatus.PARTIALLY_SUPPORTED:
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.GROUNDING_FAILURE,
                        title="Partial Grounding Deficit",
                        description="Some claims in the answer could not be verified against the supplied context.",
                        rationale="Some claims in the answer could not be verified against the supplied context.",
                        confidence=0.65,
                        evidence=grounding_evidence,
                        contributing_factors=["Extraneous detail generation"],
                    )
                )

        # 3. Agent & Tool Failure Analysis
        agent_metrics: AgentMetrics = self.agent_analyzer.analyze_agent_run(run)
        collected_metrics.extend(agent_metrics.to_diagnostic_metrics())
        agent_evidence = agent_metrics.to_evidence()
        collected_evidence.extend(agent_evidence)

        if agent_metrics.has_loop:
            failures.append(
                Failure(
                    failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                    category=FailureCategory.AGENT_LOOP,
                    severity=Severity.CRITICAL
                    if agent_metrics.has_unresolved_error
                    else Severity.HIGH,
                    title="Agent Repetition / Loop Detected",
                    description=f"Agent entered repetitive execution loop: {len(agent_metrics.detected_loops)} patterns found.",
                    message=f"Agent entered repetitive execution loop: {len(agent_metrics.detected_loops)} patterns found.",
                    evidence=agent_evidence,
                )
            )
            candidate_hypotheses.append(
                Hypothesis(
                    category=FailureCategory.AGENT_LOOP,
                    title="Agent Infinite Loop / Repetitive Tool Invocation",
                    description="Agent repeatedly invoked tools in a cycle without advancing toward completion.",
                    rationale="Agent repeatedly invoked tools in a cycle without advancing toward completion.",
                    confidence=0.95,
                    evidence=agent_evidence,
                    contributing_factors=[
                        "Lack of loop termination condition",
                        "Identical observation handling",
                        "Prompt state stagnation",
                    ],
                )
            )

        if agent_metrics.tool_metrics.failed_calls > 0:
            failures.append(
                Failure(
                    failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                    category=FailureCategory.TOOL_ERROR,
                    severity=Severity.HIGH,
                    title="Tool Execution Failure",
                    description=f"{agent_metrics.tool_metrics.failed_calls} tool invocations failed with errors.",
                    message=f"{agent_metrics.tool_metrics.failed_calls} tool invocations failed with errors.",
                    evidence=agent_evidence,
                )
            )
            candidate_hypotheses.append(
                Hypothesis(
                    category=FailureCategory.TOOL_ERROR,
                    title="Tool Invocation or Execution Failure",
                    description="One or more tools returned error responses or threw runtime exceptions.",
                    rationale="One or more tools returned error responses or threw runtime exceptions.",
                    confidence=0.90,
                    evidence=agent_evidence,
                    contributing_factors=[
                        "Downstream service unavailable",
                        "Authentication error",
                        "Malformed payload",
                    ],
                )
            )

        if agent_metrics.tool_metrics.argument_error_count > 0:
            failures.append(
                Failure(
                    failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                    category=FailureCategory.SCHEMA_VIOLATION,
                    severity=Severity.HIGH,
                    title="Tool Argument Schema Violation",
                    description=f"{agent_metrics.tool_metrics.argument_error_count} tool call(s) had unparseable arguments.",
                    message=f"{agent_metrics.tool_metrics.argument_error_count} tool call(s) had unparseable arguments.",
                    evidence=agent_evidence,
                )
            )
            candidate_hypotheses.append(
                Hypothesis(
                    category=FailureCategory.SCHEMA_VIOLATION,
                    title="Malformed Tool Call Arguments",
                    description="Model generated invalid JSON or malformed arguments for tool calls.",
                    rationale="Model generated invalid JSON or malformed arguments for tool calls.",
                    confidence=0.90,
                    evidence=agent_evidence,
                    contributing_factors=[
                        "Weak function calling syntax",
                        "Missing schema constraints in system prompt",
                    ],
                )
            )

        # 4. LLM Call Execution Analysis (Direct Span Errors)
        for span in run.spans:
            if span.kind == SpanKind.LLM and (span.is_error or span.error_message):
                ev = Evidence(
                    evidence_type=EvidenceType.ERROR_LOG,
                    description=f"LLM call span '{span.name}' failed: {span.error_message}",
                    supporting_data={"error": span.error_message},
                    span_id=span.span_id,
                )
                collected_evidence.append(ev)
                failures.append(
                    Failure(
                        failure_id=f"fail-{uuid.uuid4().hex[:8]}",
                        category=FailureCategory.LLM_CALL_FAILURE,
                        severity=Severity.CRITICAL,
                        title="LLM API / Execution Failure",
                        description=span.error_message or "LLM call failed with error status",
                        message=span.error_message or "LLM call failed with error status",
                        span_id=span.span_id,
                        evidence=[ev],
                    )
                )
                candidate_hypotheses.append(
                    Hypothesis(
                        category=FailureCategory.LLM_CALL_FAILURE,
                        title="LLM Generation Call Failed",
                        description="Underlying LLM API or model process encountered a fatal error.",
                        rationale="Underlying LLM API or model process encountered a fatal error.",
                        confidence=0.98,
                        evidence=[ev],
                        contributing_factors=[
                            "API rate limit",
                            "Context length exceeded",
                            "Provider downtime",
                        ],
                    )
                )

        # 4.5. Custom Third-Party Analyzer Plugins
        custom_failures, custom_evidence, custom_metrics = self.plugin_manager.run_analyzers(
            Trace(trace_id=run.trace_id, runs=[run])
        )
        failures.extend(custom_failures)
        collected_evidence.extend(custom_evidence)
        collected_metrics.extend(custom_metrics)
        for cf in custom_failures:
            candidate_hypotheses.append(
                Hypothesis(
                    category=cf.category,
                    title=cf.title or "Plugin Detected Failure",
                    description=cf.description or cf.message,
                    rationale=cf.description or cf.message,
                    confidence=0.85,
                    evidence=cf.evidence,
                )
            )

        # 4.6. Multi-Modal and Structured Output Analysis
        mm_res = self.multimodal_analyzer.analyze_run(run)
        failures.extend(mm_res.failures)
        collected_evidence.extend(mm_res.evidence)
        collected_metrics.extend(mm_res.metrics)
        for mmf in mm_res.failures:
            candidate_hypotheses.append(
                Hypothesis(
                    category=mmf.category,
                    title=mmf.title or "Multi-Modal / Schema Failure",
                    description=mmf.description or mmf.message,
                    rationale=mmf.description or mmf.message,
                    confidence=0.88,
                    evidence=mmf.evidence,
                    contributing_factors=[
                        "Invalid JSON format",
                        "Schema constraint violation",
                        "Unresolved visual media",
                    ],
                )
            )

        # 5. Synthesize Diagnosis

        ranked_hypotheses = self._rank_hypotheses(candidate_hypotheses)

        if ranked_hypotheses:
            primary_category = ranked_hypotheses[0].category
            if any(f.severity == Severity.CRITICAL for f in failures):
                severity = Severity.CRITICAL
            elif any(f.severity == Severity.HIGH for f in failures):
                severity = Severity.HIGH
            elif any(f.severity == Severity.MEDIUM for f in failures):
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            summary = (
                f"Identified {len(failures)} failure(s). Primary root cause: {ranked_hypotheses[0].title} "
                f"(Confidence: {int(ranked_hypotheses[0].confidence * 100)}%)."
            )
        else:
            primary_category = FailureCategory.NONE
            severity = Severity.INFO
            summary = "No reliability failures detected. Run executed within healthy operational thresholds."

        diag = Diagnosis(
            trace_id=run.trace_id,
            run_id=run.run_id,
            root_cause=primary_category,
            primary_category=primary_category,
            severity=severity,
            summary=summary,
            failures=failures,
            hypotheses=ranked_hypotheses,
            metrics=collected_metrics,
            evidence=collected_evidence,
        )

        # Attach prioritized recommendations
        diag.recommendations = self.recommendation_engine.generate_recommendations(diag)
        return self.plugin_manager.apply_post_middlewares(diag)

    def _rank_hypotheses(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Deduplicate, rank, and sort hypotheses by confidence score in descending order."""
        if not hypotheses:
            return []

        unique_map: dict[tuple[FailureCategory, str], Hypothesis] = {}
        for h in hypotheses:
            key: tuple[FailureCategory, str] = (h.category, str(h.title or ""))
            if key not in unique_map or h.confidence > unique_map[key].confidence:
                unique_map[key] = h

        sorted_list = sorted(unique_map.values(), key=lambda x: x.confidence, reverse=True)

        ranked_result: list[Hypothesis] = []
        for i, h in enumerate(sorted_list, start=1):
            ranked_result.append(
                Hypothesis(
                    category=h.category,
                    title=h.title,
                    description=h.description,
                    rationale=h.rationale,
                    confidence=h.confidence,
                    rank=i,
                    evidence=h.evidence,
                    contributing_factors=h.contributing_factors,
                )
            )

        return ranked_result
