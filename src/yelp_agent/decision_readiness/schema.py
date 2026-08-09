"""Stable Step 19 contracts consumed later by AgentState and Router."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel

type TaskType = Literal[
    "recommendation_request",
    "business_detail_question",
    "candidate_comparison",
    "feedback_refinement",
    "official_policy_question",
    "review_experience_question",
    "unknown",
]
type RankingUncertaintyReason = Literal[
    "sparse_history",
    "unseen_category",
    "feature_disagreement",
    "small_top_margin",
    "weak_collaborative_support",
    "low_profile_reliability",
]
type InformationGap = Literal[
    "missing_location",
    "missing_budget",
    "missing_party_size",
    "constraint_conflict",
    "ambiguous_reference",
]
type CalibratorKind = Literal["logistic", "isotonic"]
type ConfidenceTarget = Literal[
    "hybrid_v2_b_next_business_top1",
    "unavailable",
]
type ConfidenceUnavailableReason = Literal[
    "not_applicable_for_task",
    "query_aware_labels_unavailable",
    "ranking_signals_unavailable",
    "calibrator_not_loaded",
]


CALIBRATION_FEATURE_NAMES = (
    "top1_blend_score",
    "blend_score_margin",
    "model_score_margin",
    "hybrid_v1_score_margin",
    "model_v1_rank_disagreement",
    "component_disagreement",
    "top1_item_knn_positive_score",
    "top1_item_knn_missing",
    "user_history_length_log",
    "user_profile_reliability",
    "top1_user_category_novelty",
    "top1_route_coverage",
)
ISOTONIC_INPUT_FEATURE = "blend_score_margin"


class RankingSignals(StrictModel):
    """Target-blind task signals available before a recommendation is returned."""

    top1_blend_score: float
    blend_score_margin: float
    model_score_margin: float
    hybrid_v1_score_margin: float
    model_v1_rank_disagreement: float = Field(ge=0, le=1)
    component_disagreement: float = Field(ge=0, le=1)
    top1_item_knn_positive_score: float
    top1_item_knn_missing: float = Field(ge=0, le=1)
    user_history_length_log: float = Field(ge=0)
    user_profile_reliability: float = Field(ge=0, le=1)
    top1_user_category_novelty: float = Field(ge=0, le=1)
    top1_route_coverage: float = Field(ge=0)

    @field_validator("*")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("ranking signals must be finite")
        return value

    def as_feature_dict(self) -> dict[str, float]:
        values = self.model_dump()
        return {name: float(values[name]) for name in CALIBRATION_FEATURE_NAMES}


class UncertaintyThresholds(StrictModel):
    sparse_history_log_max: float = Field(ge=0)
    unseen_category_min: float = Field(ge=0, le=1)
    feature_disagreement_min: float = Field(ge=0, le=1)
    small_top_margin_max: float
    weak_collaborative_support_max: float
    low_profile_reliability_max: float = Field(ge=0, le=1)


class RankingConfidenceEstimate(StrictModel):
    probability_top1_correct: float = Field(ge=0, le=1)
    target: Literal["hybrid_v2_b_next_business_top1"]
    calibrator_kind: CalibratorKind
    calibrator_version: str = Field(min_length=1)
    uncertainty_reasons: list[RankingUncertaintyReason]

    @field_validator("uncertainty_reasons")
    @classmethod
    def validate_unique_reasons(
        cls,
        values: list[RankingUncertaintyReason],
    ) -> list[RankingUncertaintyReason]:
        if len(values) != len(set(values)):
            raise ValueError("ranking uncertainty reasons must be unique")
        return values


class DecisionReadiness(StrictModel):
    """One compact, explainable state snapshot for a future Router."""

    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_type: TaskType
    task_type_reason_code: str = Field(min_length=1)
    information_gaps: list[InformationGap]
    conflict_fields: list[str]
    ranking_confidence: RankingConfidenceEstimate | None = None
    confidence_target: ConfidenceTarget
    confidence_unavailable_reason: ConfidenceUnavailableReason | None = None
    analyzer_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_state(self) -> DecisionReadiness:
        if len(self.information_gaps) != len(set(self.information_gaps)):
            raise ValueError("information gaps must be unique")
        if len(self.conflict_fields) != len(set(self.conflict_fields)):
            raise ValueError("conflict fields must be unique")
        available = self.confidence_target == "hybrid_v2_b_next_business_top1"
        if available != (self.ranking_confidence is not None):
            raise ValueError("confidence target and estimate availability disagree")
        if available == (self.confidence_unavailable_reason is not None):
            raise ValueError("confidence availability reason is inconsistent")
        if bool(self.conflict_fields) != (
            "constraint_conflict" in self.information_gaps
        ):
            raise ValueError("conflict gap must match conflict_fields")
        return self


class ReliabilityBin(StrictModel):
    lower_bound: float = Field(ge=0, le=1)
    upper_bound: float = Field(ge=0, le=1)
    task_count: int = Field(ge=0)
    mean_confidence: float | None = Field(default=None, ge=0, le=1)
    observed_accuracy: float | None = Field(default=None, ge=0, le=1)


class CoverageRiskPoint(StrictModel):
    requested_coverage: float = Field(gt=0, le=1)
    actual_coverage: float = Field(gt=0, le=1)
    selected_task_count: int = Field(ge=1)
    minimum_confidence: float = Field(ge=0, le=1)
    observed_accuracy: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)


class CalibrationMetrics(StrictModel):
    method: CalibratorKind
    task_count: int = Field(ge=1)
    positive_count: int = Field(ge=1)
    positive_rate: float = Field(gt=0, le=1)
    brier_score: float = Field(ge=0, le=1)
    expected_calibration_error: float = Field(ge=0, le=1)
    reliability_binning: Literal["equal_frequency"] = "equal_frequency"
    reliability_bins: list[ReliabilityBin]
    coverage_risk: list[CoverageRiskPoint]


class DecisionReadinessExperimentReport(StrictModel):
    experiment_name: Literal["Step 19 Decision Readiness"] = (
        "Step 19 Decision Readiness"
    )
    source_task_count: int = Field(ge=1)
    fold_counts: dict[str, int]
    selected_calibrator: CalibratorKind
    candidate_metrics: dict[CalibratorKind, CalibrationMetrics]
    base_rate_brier_score: float = Field(ge=0, le=1)
    selected_brier_improvement_vs_base_rate: float
    confidence_target: Literal["hybrid_v2_b_next_business_top1"]
    query_aware_confidence_calibrated: Literal[False] = False
    query_aware_limitation: str = Field(min_length=1)
    test_data_used_for_fit: Literal[False] = False
    test_data_used_for_selection: Literal[False] = False
