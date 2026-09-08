"""Recommendation rules mapping diagnostic metrics, failures, and evidence to actionable remediation steps."""

from typing import Any

from llm_reliability.models.diagnosis import Diagnosis, Recommendation
from llm_reliability.models.enums import EvidenceType, FailureCategory


def generate_rag_recommendations(diagnosis: Diagnosis) -> list[Recommendation]:
    """Generate targeted recommendations for RAG retrieval and context construction issues."""
    recs: list[Recommendation] = []
    metrics_map: dict[str, Any] = {m.name: m for m in diagnosis.metrics}

    # 1. Empty Retrieval / Low Relevance
    if diagnosis.primary_category == FailureCategory.RETRIEVAL_FAILURE or any(
        f.category == FailureCategory.RETRIEVAL_FAILURE for f in diagnosis.failures
    ):
        total_count_metric = metrics_map.get("retrieval_total_count")
        is_empty = (total_count_metric and total_count_metric.value == 0) or any(
            "empty" in (f.title or "").lower() or "0 document" in (f.description or "").lower()
            for f in diagnosis.failures
        )

        if is_empty:
            recs.append(
                Recommendation(
                    title="Expand Query Filters and Verify Knowledge Base Index",
                    description=(
                        "1. Verify that your vector database / search index contains indexed documents for the target collection.\n"
                        "2. Inspect metadata filters in your retriever to ensure they are not overly restrictive.\n"
                        "3. Implement query expansion or HyDE (Hypothetical Document Embeddings) to bridge lexical gaps."
                    ),
                    action_type="RETRIEVER_CONFIG",
                    priority=1,
                    rationale="Retriever returned zero documents for the user query.",
                )
            )

        # Low Relevance
        low_rel_metric = metrics_map.get("retrieval_relevance_ratio")
        if low_rel_metric and low_rel_metric.passed is False:
            recs.append(
                Recommendation(
                    title="Integrate Cross-Encoder Reranker and Upgrade Embedding Model",
                    description=(
                        "1. Add a second-stage cross-encoder reranker (e.g., BAAI/bge-reranker or Cohere Rerank) "
                        "after initial retrieval.\n"
                        "2. Evaluate domain-adapted embedding models with higher retrieval benchmark scores (MTEB).\n"
                        "3. Tune similarity score cutoffs to filter out low-confidence noise chunks."
                    ),
                    action_type="RERANKING",
                    priority=2,
                    rationale="Retrieved candidate documents exhibited low semantic similarity scores relative to query.",
                )
            )

    # 2. Context Duplication & Chunking
    dup_metric = metrics_map.get("retrieval_duplicate_ratio")
    if (
        dup_metric and dup_metric.passed is False
    ) or diagnosis.primary_category == FailureCategory.CONTEXT_CONSTRUCTION_FAILURE:
        recs.append(
            Recommendation(
                title="Enable Pre-Context Document Deduplication and Adjust Chunk Overlap",
                description=(
                    "1. Deduplicate retrieved chunks before context formatting using embedding cosine similarity (>0.85) "
                    "or lexical Jaccard distance.\n"
                    "2. Reduce chunk overlap percentage during document indexing (e.g., from 30% to 10-15%).\n"
                    "3. Implement contextual compression to retain only unique informative sentences."
                ),
                action_type="CHUNKING_CONFIG",
                priority=2,
                rationale="High proportion of duplicate and near-duplicate content detected across retrieved documents.",
            )
        )

    # 3. Context Length Shortfall (when non-empty retrieval fails to meet volume requirement)
    if diagnosis.primary_category != FailureCategory.NONE:
        context_tokens_metric = metrics_map.get("retrieval_context_tokens")
        if context_tokens_metric and context_tokens_metric.passed is False:
            recs.append(
                Recommendation(
                    title="Optimize Top-K Retrieval Parameter and Chunk Size",
                    description=(
                        "1. Adjust top-k retrieval parameter to balance information density against LLM context budget.\n"
                        "2. Increase chunk size (e.g., from 256 to 512/1024 tokens) to preserve contextual completeness."
                    ),
                    action_type="TOP_K_ADJUSTMENT",
                    priority=3,
                    rationale="Retrieved context volume did not meet minimum recommended character/token threshold.",
                )
            )

    return recs


def generate_grounding_recommendations(diagnosis: Diagnosis) -> list[Recommendation]:
    """Generate recommendations for hallucination, ungrounded claims, and contradiction failures."""
    recs: list[Recommendation] = []
    grounding_failures = [
        f
        for f in diagnosis.failures
        if f.category in (FailureCategory.GROUNDING_FAILURE, FailureCategory.HALLUCINATION)
    ]

    if not grounding_failures and diagnosis.primary_category not in (
        FailureCategory.GROUNDING_FAILURE,
        FailureCategory.HALLUCINATION,
    ):
        return recs

    has_contradiction = any(
        e.evidence_type == EvidenceType.CONTRADICTION for e in diagnosis.evidence
    ) or any(f.category == FailureCategory.HALLUCINATION for f in diagnosis.failures)

    if has_contradiction:
        recs.append(
            Recommendation(
                title="Enforce Strict Context Adherence and Lower Model Temperature",
                description=(
                    "1. Set LLM temperature to 0.0 or 0.1 for factual extractive question-answering tasks.\n"
                    "2. Update system prompt with explicit guardrails: 'Answer STRICTLY using only the provided context. "
                    "If the context states X, do not contradict it with prior knowledge.'\n"
                    "3. Add a post-generation verification step or self-correction prompt to catch factual conflicts."
                ),
                action_type="PROMPT_TUNING",
                priority=1,
                rationale="Model generated claims that directly contradict factual statements in the retrieved context.",
            )
        )
    else:
        recs.append(
            Recommendation(
                title="Add Grounding Constraints and Negative Answering Instructions",
                description=(
                    "1. Explicitly instruct the model in system prompt: 'If the provided context does not contain sufficient "
                    "information to answer the question, state that you do not know rather than fabricating details.'\n"
                    "2. Reduce temperature and top_p parameters to discourage creative completion.\n"
                    "3. Include structured few-shot examples demonstrating refusal when context is missing."
                ),
                action_type="PROMPT_TUNING",
                priority=1,
                rationale="Generated response contained unsupported claims and named entities absent from context documents.",
            )
        )

    return recs


def generate_agent_recommendations(diagnosis: Diagnosis) -> list[Recommendation]:
    """Generate recommendations for agent trajectory loops, tool failures, and argument errors."""
    recs: list[Recommendation] = []

    # 1. Agent Loops and Repetitions
    loop_failures = [
        f
        for f in diagnosis.failures
        if f.category in (FailureCategory.AGENT_LOOP, FailureCategory.AGENT_LOOP_FAILURE)
    ]
    if loop_failures or diagnosis.primary_category in (
        FailureCategory.AGENT_LOOP,
        FailureCategory.AGENT_LOOP_FAILURE,
    ):
        recs.append(
            Recommendation(
                title="Implement Agent Loop Guardrails and Trajectory History Tracking",
                description=(
                    "1. Add an execution middleware that detects repeated tool calls with identical arguments and interrupts.\n"
                    "2. Enforce hard maximum step limit (e.g. max 8 steps) with graceful fallback synthesis.\n"
                    "3. Append observation history to prompt with explicit instructions: 'You have already tried X without success, try a different approach.'"
                ),
                action_type="AGENT_GUARDRAIL",
                priority=1,
                rationale="Agent entered an infinite loop or repetitive invocation cycle without advancing state.",
            )
        )

    # 2. Tool Execution Failures
    tool_failures = [
        f
        for f in diagnosis.failures
        if f.category in (FailureCategory.TOOL_ERROR, FailureCategory.TOOL_EXECUTION_FAILURE)
    ]
    if tool_failures or diagnosis.primary_category in (
        FailureCategory.TOOL_ERROR,
        FailureCategory.TOOL_EXECUTION_FAILURE,
    ):
        recs.append(
            Recommendation(
                title="Configure Tool Retry Policy with Exponential Backoff and Error Recovery",
                description=(
                    "1. Wrap external tool APIs with retry policies (exponential backoff with jitter, max 3 attempts).\n"
                    "2. Return informative, structured error messages to the model so it can self-heal (e.g. 'Invalid parameter format').\n"
                    "3. Provide alternative fallback tools for high-failure operations."
                ),
                action_type="TOOL_RETRY_POLICY",
                priority=1,
                rationale="One or more tool executions raised unhandled exceptions or HTTP error codes.",
            )
        )

    # 3. Schema & Argument Violations
    schema_failures = [
        f for f in diagnosis.failures if f.category == FailureCategory.SCHEMA_VIOLATION
    ]
    if schema_failures or diagnosis.primary_category == FailureCategory.SCHEMA_VIOLATION:
        recs.append(
            Recommendation(
                title="Enforce Pydantic / JSON Schema Validation on Tool Calling Definitions",
                description=(
                    "1. Use strict schema definitions (e.g., OpenAI Structured Outputs or Pydantic models) for tool arguments.\n"
                    "2. Provide explicit type annotations, parameter descriptions, and enum constraints in tool descriptions.\n"
                    "3. Use structured JSON mode to prevent malformed string generation."
                ),
                action_type="TOOL_SCHEMA_DEFINITION",
                priority=1,
                rationale="Model produced invalid or unparseable JSON arguments when invoking tools.",
            )
        )

    return recs


def generate_llm_recommendations(diagnosis: Diagnosis) -> list[Recommendation]:
    """Generate recommendations for direct LLM API failures, rate limits, and latency spikes."""
    recs: list[Recommendation] = []

    llm_failures = [f for f in diagnosis.failures if f.category == FailureCategory.LLM_CALL_FAILURE]
    if llm_failures or diagnosis.primary_category == FailureCategory.LLM_CALL_FAILURE:
        recs.append(
            Recommendation(
                title="Configure Provider Fallback Routing and Rate Limit Throttling",
                description=(
                    "1. Implement multi-provider fallback routing (e.g. fallback to alternate model or deployment region).\n"
                    "2. Implement client-side token bucket rate limiting to prevent 429 Too Many Requests.\n"
                    "3. Monitor prompt context length against model context window limits."
                ),
                action_type="PROVIDER_ROUTING",
                priority=1,
                rationale="Underlying LLM API or model process failed during execution.",
            )
        )

    return recs
