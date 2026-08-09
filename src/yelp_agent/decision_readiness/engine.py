"""Deep Step 19 module: one request and optional ranking signals in, one state out."""

from __future__ import annotations

from typing import Literal

from yelp_agent.query.schema import RecommendationRequest

from .calibration import FrozenConfidenceCalibrator
from .request_analysis import classify_task_type, identify_information_gaps
from .schema import ConfidenceUnavailableReason, DecisionReadiness, RankingSignals


class DecisionReadinessAnalyzer:
    """Hide task classification, gap policy, and calibrated ranking risk."""

    version = "1.0.0"

    def __init__(self, calibrator: FrozenConfidenceCalibrator | None = None) -> None:
        self._calibrator = calibrator

    def analyze(
        self,
        request: RecommendationRequest,
        *,
        ranking_source: Literal["hybrid_v2_b", "query_aware", "none"] = "none",
        ranking_signals: RankingSignals | None = None,
    ) -> DecisionReadiness:
        task_type, reason_code = classify_task_type(request)
        gaps, conflicts = identify_information_gaps(request, task_type)
        confidence = None
        unavailable_reason: ConfidenceUnavailableReason | None
        if task_type != "recommendation_request" or ranking_source == "none":
            unavailable_reason = "not_applicable_for_task"
        elif ranking_source == "query_aware":
            # No Query x business relevance labels exist yet. Reusing next-business
            # confidence here would create a plausible-looking but false probability.
            unavailable_reason = "query_aware_labels_unavailable"
        elif self._calibrator is None:
            unavailable_reason = "calibrator_not_loaded"
        elif ranking_signals is None:
            unavailable_reason = "ranking_signals_unavailable"
        else:
            confidence = self._calibrator.estimate(ranking_signals)
            unavailable_reason = None
        return DecisionReadiness(
            request_id=request.request_id,
            task_type=task_type,
            task_type_reason_code=reason_code,
            information_gaps=list(gaps),
            conflict_fields=list(conflicts),
            ranking_confidence=confidence,
            confidence_target=(
                "hybrid_v2_b_next_business_top1"
                if confidence is not None
                else "unavailable"
            ),
            confidence_unavailable_reason=unavailable_reason,
        )
