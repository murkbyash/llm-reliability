"""Data models for benchmark samples, synthetic datasets, and diagnostic accuracy reporting."""

from pydantic import BaseModel, Field

from llm_reliability.models.enums import FailureCategory, Severity
from llm_reliability.models.trace import Trace


class BenchmarkSample(BaseModel):
    """A ground-truth labeled benchmark test sample containing a trace and expected diagnosis."""

    sample_id: str = Field(description="Unique identifier for the benchmark sample")
    trace: Trace = Field(description="Synthetic or real execution trace")
    ground_truth_category: FailureCategory = Field(
        description="Expected primary root cause failure category"
    )
    expected_severity: Severity | None = Field(
        default=None, description="Optional expected severity level"
    )
    description: str = Field(default="", description="Description of the test scenario")


class CategoryAccuracyMetrics(BaseModel):
    """Precision, recall, and F1 accuracy metrics for a specific failure category."""

    category: FailureCategory = Field(description="Evaluated failure category")
    total_samples: int = Field(default=0, ge=0, description="Total ground truth samples")
    true_positives: int = Field(default=0, ge=0, description="Correctly diagnosed occurrences")
    false_positives: int = Field(
        default=0, ge=0, description="Incorrectly attributed to this category"
    )
    false_negatives: int = Field(default=0, ge=0, description="Missed occurrences of this category")
    precision: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Precision score (TP / (TP + FP))"
    )
    recall: float = Field(default=1.0, ge=0.0, le=1.0, description="Recall score (TP / (TP + FN))")
    f1_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Harmonic mean of precision and recall"
    )


class BenchmarkReport(BaseModel):
    """Comprehensive accuracy evaluation report produced by the benchmark runner."""

    total_samples: int = Field(description="Total number of evaluated benchmark samples")
    overall_accuracy: float = Field(
        ge=0.0, le=1.0, description="Proportion of samples with exact matching root cause"
    )
    macro_precision: float = Field(
        ge=0.0, le=1.0, description="Unweighted average precision across categories"
    )
    macro_recall: float = Field(
        ge=0.0, le=1.0, description="Unweighted average recall across categories"
    )
    macro_f1: float = Field(
        ge=0.0, le=1.0, description="Unweighted average F1 score across categories"
    )
    category_metrics: dict[str, CategoryAccuracyMetrics] = Field(
        default_factory=dict, description="Detailed per-category metrics breakdown"
    )
    passed: bool = Field(
        description="True if overall accuracy and precision satisfy benchmark requirements (>=90%)"
    )
    summary: str = Field(description="Human-readable benchmark evaluation summary")
