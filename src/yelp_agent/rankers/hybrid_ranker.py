"""Weighted ranking over category, text, quality, and location features."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

from yelp_agent.features.hybrid import HybridFeatureStore, HybridWeights
from yelp_agent.models import (
    Prediction,
    RecommendationTask,
    ScoreBreakdown,
    StrictModel,
    UserProfile,
)


class HybridTaskScore(StrictModel):
    profile: UserProfile
    score_breakdowns: dict[str, ScoreBreakdown]


class HybridRanker:
    """Produce explainable weighted rankings without reading ground truth."""

    def __init__(
        self,
        feature_store: HybridFeatureStore,
        weights: HybridWeights,
    ) -> None:
        self._feature_store = feature_store
        self._weights = weights
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            HybridTaskScore,
        ] = {}

    @property
    def weights(self) -> HybridWeights:
        return self._weights

    def score(self, task: RecommendationTask) -> HybridTaskScore:
        """Return the merged profile and explainable candidate score details."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        features = self._feature_store.features_for(task)
        breakdowns: dict[str, ScoreBreakdown] = {}
        for business_id in task.candidate_business_ids:
            component = features.business_scores[business_id]
            hybrid_score = (
                self._weights.category * component.category_score
                + self._weights.text * component.text_score
                + self._weights.quality * component.quality_score
                + self._weights.location * component.location_score
            )
            breakdowns[business_id] = ScoreBreakdown(
                business_id=business_id,
                category_score=component.category_score,
                text_score=component.text_score,
                quality_score=component.quality_score,
                location_score=component.location_score,
                hybrid_score=hybrid_score,
            )
        result = HybridTaskScore(
            profile=features.profile,
            score_breakdowns=breakdowns,
        )
        self._cache[cache_key] = result
        return result

    def rank(self, task: RecommendationTask) -> Prediction:
        started_at = perf_counter()
        scored = self.score(task)
        ranking = sorted(
            task.candidate_business_ids,
            key=lambda business_id: (
                -scored.score_breakdowns[business_id].hybrid_score,
                business_id,
            ),
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        return Prediction(
            task_id=task.task_id,
            ranking=ranking,
            latency_ms=latency_ms,
            fallback=False,
            tool_calls=0,
            llm_tokens=None,
            metadata={
                "method": "hybrid",
                "weights": self._weights.as_dict,
                "llm_attempted": False,
            },
        )
