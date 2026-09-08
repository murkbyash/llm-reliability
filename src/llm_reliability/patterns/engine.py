"""Failure pattern clustering and structural signature detection engine."""

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.models.diagnosis import Diagnosis
from llm_reliability.models.enums import FailureCategory, Severity
from llm_reliability.models.trace import Run, Trace
from llm_reliability.normalization.loader import load_trace
from llm_reliability.patterns.models import (
    FailureCluster,
    FailureSignature,
    PatternAnalysisReport,
)


class PatternDetectionEngine:
    """Detects recurrent structural failure patterns, error fingerprints, and clusters across trace corpora."""

    def __init__(self, diagnostic_engine: DiagnosticEngine | None = None) -> None:
        """Initialize pattern detection engine with diagnostic orchestrator."""
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()

    def analyze_patterns(
        self,
        source: Sequence[Run | Trace | dict[str, Any] | Any] | Trace,
    ) -> PatternAnalysisReport:
        """Cluster execution traces by structural failure signatures and rank by occurrence frequency.

        Args:
            source: Collection of runs, traces, or raw trace dictionaries.

        Returns:
            PatternAnalysisReport containing ranked failure clusters and summaries.
        """
        runs = self._extract_runs(source)
        if not runs:
            return PatternAnalysisReport(
                total_runs_analyzed=0,
                failed_runs_count=0,
                unique_patterns_count=0,
                clusters=[],
                summary="Pattern analysis completed: No runs provided.",
            )

        run_diagnoses: list[tuple[Run, Diagnosis]] = [
            (run, self.diagnostic_engine.diagnose_run(run)) for run in runs
        ]

        failed_diagnoses = [
            (r, d)
            for r, d in run_diagnoses
            if d.primary_category != FailureCategory.NONE or d.failures
        ]
        failed_count = len(failed_diagnoses)

        if failed_count == 0:
            return PatternAnalysisReport(
                total_runs_analyzed=len(runs),
                failed_runs_count=0,
                unique_patterns_count=0,
                clusters=[],
                summary=f"Pattern analysis completed: All {len(runs)} execution runs succeeded with zero diagnosed failures.",
            )

        # Map fingerprint -> list of (Run, Diagnosis, FailureSignature)
        signature_groups: dict[str, list[tuple[Run, Diagnosis, FailureSignature]]] = defaultdict(
            list
        )

        for run, diag in failed_diagnoses:
            signatures = self.generate_signatures(diag, run)
            for sig in signatures:
                signature_groups[sig.fingerprint].append((run, diag, sig))

        clusters: list[FailureCluster] = []
        cluster_idx = 1

        for fingerprint, items in signature_groups.items():
            primary_sig = items[0][2]
            cluster_runs = [item[0] for item in items]
            cluster_diags = [item[1] for item in items]

            run_ids = [r.run_id for r in cluster_runs if r.run_id]
            trace_ids = [r.trace_id for r in cluster_runs if r.trace_id]

            # Select representative run (prefer run with highest severity or most evidence)
            representative_run = max(
                cluster_runs,
                key=lambda r: len(r.spans),
            )
            representative_diag = next(
                (
                    d
                    for r, d in zip(cluster_runs, cluster_diags, strict=False)
                    if r == representative_run
                ),
                cluster_diags[0],
            )

            # Aggregate contributing factors
            contributing_factors: set[str] = set()
            for diag in cluster_diags:
                for h in diag.hypotheses:
                    contributing_factors.update(h.contributing_factors)

            # Consolidate remediation advice
            remediation_summary = None
            if representative_diag.recommendations:
                remediation_summary = "; ".join(
                    f"{rec.title}" for rec in representative_diag.recommendations[:2]
                )

            occurrence_count = len(items)
            frequency_ratio = round(occurrence_count / failed_count, 4) if failed_count > 0 else 0.0

            clusters.append(
                FailureCluster(
                    cluster_id=f"cluster-{cluster_idx:03d}-{fingerprint[:6]}",
                    signature=primary_sig,
                    occurrence_count=occurrence_count,
                    frequency_ratio=frequency_ratio,
                    representative_run_id=representative_run.run_id,
                    representative_trace_id=representative_run.trace_id,
                    run_ids=run_ids,
                    trace_ids=trace_ids,
                    common_contributing_factors=sorted(list(contributing_factors)),
                    remediation_summary=remediation_summary,
                )
            )
            cluster_idx += 1

        # Sort clusters by occurrence frequency descending
        clusters.sort(key=lambda c: c.occurrence_count, reverse=True)

        summary = (
            f"Pattern Analysis: Discovered {len(clusters)} distinct failure pattern(s) "
            f"across {failed_count} failed run(s) out of {len(runs)} total runs analyzed. "
            f"Top pattern '{clusters[0].signature.pattern_name}' accounted for "
            f"{clusters[0].occurrence_count} failure(s) ({int(clusters[0].frequency_ratio * 100)}%)."
        )

        return PatternAnalysisReport(
            total_runs_analyzed=len(runs),
            failed_runs_count=failed_count,
            unique_patterns_count=len(clusters),
            clusters=clusters,
            summary=summary,
        )

    def generate_signatures(
        self, diagnosis: Diagnosis, run: Run | None = None
    ) -> list[FailureSignature]:
        """Generate deterministic structural signatures for diagnosed failures in an execution run."""
        signatures: list[FailureSignature] = []

        if diagnosis.failures:
            for failure in diagnosis.failures:
                sig = self._create_signature_from_failure(failure, diagnosis)
                signatures.append(sig)
        elif diagnosis.primary_category != FailureCategory.NONE:
            # Generate from primary diagnosis
            category = diagnosis.primary_category
            structural_key = (
                f"{category.value}:primary_diagnosis:{self._normalize_text(diagnosis.summary)}"
            )
            fingerprint = hashlib.sha256(structural_key.encode("utf-8")).hexdigest()[:12]

            signatures.append(
                FailureSignature(
                    fingerprint=fingerprint,
                    category=category,
                    pattern_name=f"{category.value.replace('_', ' ').title()}",
                    structural_key=structural_key,
                    severity=diagnosis.severity,
                    description=diagnosis.summary,
                )
            )

        return signatures

    def _create_signature_from_failure(
        self, failure: Any, diagnosis: Diagnosis
    ) -> FailureSignature:
        """Build normalized structural key and cryptographic fingerprint for a single failure instance."""
        category: FailureCategory = failure.category
        severity: Severity = failure.severity

        # Invariant key based on category, failure title, and normalized description
        normalized_desc = self._normalize_text(failure.description)
        evidence_types = sorted(list({ev.evidence_type.value for ev in failure.evidence}))
        ev_key = ":".join(evidence_types) if evidence_types else "no_evidence"

        structural_key = (
            f"{category.value}:{failure.title.lower().strip()}:{normalized_desc}:{ev_key}"
        )
        fingerprint = hashlib.sha256(structural_key.encode("utf-8")).hexdigest()[:12]

        return FailureSignature(
            fingerprint=fingerprint,
            category=category,
            pattern_name=f"{category.value.replace('_', ' ').title()}: {failure.title}",
            structural_key=structural_key,
            severity=severity,
            description=failure.description,
        )

    def _normalize_text(self, text: str) -> str:
        """Strip UUIDs, timestamps, numbers, and whitespace to produce invariant structural text."""
        # Replace hex UUIDs and hashes
        cleaned = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            "<UUID>",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b[0-9a-f]{16,64}\b", "<HASH>", cleaned, flags=re.IGNORECASE)
        # Replace digits/numbers
        cleaned = re.sub(r"\b\d+\b", "<NUM>", cleaned)
        # Replace variable whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    def _extract_runs(
        self, source: Sequence[Run | Trace | dict[str, Any] | Any] | Trace
    ) -> list[Run]:
        """Normalize various input types into a list of canonical Run models."""
        if isinstance(source, Trace):
            return list(source.runs)

        runs: list[Run] = []
        for item in source:
            if isinstance(item, Run):
                runs.append(item)
            elif isinstance(item, Trace):
                runs.extend(item.runs)
            else:
                trace = load_trace(item)
                runs.extend(trace.runs)
        return runs
