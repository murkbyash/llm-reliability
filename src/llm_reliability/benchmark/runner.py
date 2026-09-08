"""Benchmark runner evaluating diagnostic precision, recall, and accuracy against ground truth datasets."""

from collections import defaultdict

from llm_reliability.benchmark.generator import SyntheticTraceGenerator
from llm_reliability.benchmark.models import (
    BenchmarkReport,
    BenchmarkSample,
    CategoryAccuracyMetrics,
)
from llm_reliability.diagnosis.engine import DiagnosticEngine
from llm_reliability.models.enums import FailureCategory


class BenchmarkRunner:
    """Evaluates DiagnosticEngine accuracy, precision, and recall on ground-truth benchmark datasets."""

    def __init__(self, diagnostic_engine: DiagnosticEngine | None = None) -> None:
        """Initialize benchmark runner."""
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()

    def run(
        self,
        samples: list[BenchmarkSample] | None = None,
        min_accuracy_target: float = 0.90,
    ) -> BenchmarkReport:
        """Execute benchmark evaluation across all test samples and compile accuracy report.

        Args:
            samples: List of BenchmarkSample instances. Defaults to standard synthetic suite.
            min_accuracy_target: Target minimum overall accuracy and precision (default 90%).

        Returns:
            BenchmarkReport with detailed precision, recall, and F1 metrics.
        """
        eval_samples = (
            samples
            if samples is not None
            else SyntheticTraceGenerator().generate_benchmark_suite(samples_per_category=5)
        )
        if not eval_samples:
            return BenchmarkReport(
                total_samples=0,
                overall_accuracy=1.0,
                macro_precision=1.0,
                macro_recall=1.0,
                macro_f1=1.0,
                passed=True,
                summary="Empty benchmark dataset.",
            )

        correct_count = 0
        total = len(eval_samples)

        # Track confusion matrix per category
        tp: dict[FailureCategory, int] = defaultdict(int)
        fp: dict[FailureCategory, int] = defaultdict(int)
        fn: dict[FailureCategory, int] = defaultdict(int)
        category_totals: dict[FailureCategory, int] = defaultdict(int)

        all_categories = set(FailureCategory)

        for sample in eval_samples:
            ground_truth = sample.ground_truth_category
            category_totals[ground_truth] += 1

            diagnosis = self.diagnostic_engine.diagnose(sample.trace)
            predicted = diagnosis.primary_category or diagnosis.root_cause

            if self._is_category_match(predicted, ground_truth):
                correct_count += 1
                tp[ground_truth] += 1
            else:
                fn[ground_truth] += 1
                fp[predicted] += 1

        overall_accuracy = correct_count / total if total > 0 else 0.0

        # Calculate per-category precision, recall, and F1
        category_metrics: dict[str, CategoryAccuracyMetrics] = {}
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        for cat in all_categories:
            cat_tot = category_totals.get(cat, 0)
            if cat_tot == 0 and tp[cat] == 0 and fp[cat] == 0:
                continue

            c_tp = tp[cat]
            c_fp = fp[cat]
            c_fn = fn[cat]

            prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 1.0
            rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 1.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            category_metrics[cat.value] = CategoryAccuracyMetrics(
                category=cat,
                total_samples=cat_tot,
                true_positives=c_tp,
                false_positives=c_fp,
                false_negatives=c_fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
            )

        macro_precision = sum(precisions) / len(precisions) if precisions else 1.0
        macro_recall = sum(recalls) / len(recalls) if recalls else 1.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 1.0

        passed = overall_accuracy >= min_accuracy_target and macro_precision >= min_accuracy_target

        summary = (
            f"Benchmark Evaluation: {'PASSED' if passed else 'FAILED'}. "
            f"Overall Accuracy: {overall_accuracy * 100:.1f}%, "
            f"Macro Precision: {macro_precision * 100:.1f}%, "
            f"Macro Recall: {macro_recall * 100:.1f}%, "
            f"Macro F1: {macro_f1 * 100:.1f}% across {total} samples."
        )

        return BenchmarkReport(
            total_samples=total,
            overall_accuracy=round(overall_accuracy, 4),
            macro_precision=round(macro_precision, 4),
            macro_recall=round(macro_recall, 4),
            macro_f1=round(macro_f1, 4),
            category_metrics=category_metrics,
            passed=passed,
            summary=summary,
        )

    def _is_category_match(self, predicted: FailureCategory, ground_truth: FailureCategory) -> bool:
        """Determine if predicted root cause matches ground truth."""
        if predicted == ground_truth:
            return True
        return False


def run_benchmark(samples_per_category: int = 5) -> BenchmarkReport:
    """Convenience function to generate synthetic dataset and execute diagnostic benchmark."""
    suite = SyntheticTraceGenerator().generate_benchmark_suite(
        samples_per_category=samples_per_category
    )
    runner = BenchmarkRunner()
    return runner.run(suite)
