"""Personalized category-affinity recommendation baseline."""

from __future__ import annotations

from time import perf_counter

from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.models import Prediction, RecommendationTask


class CategoryRanker:
    """Rank candidates by the user's point-in-time category preferences."""

    def __init__(self, category_store: TemporalCategoryStore) -> None:
        self._category_store = category_store

    def rank(self, task: RecommendationTask) -> Prediction:
        started_at = perf_counter()
        features = self._category_store.features_for(task)
        ranking = sorted(
            task.candidate_business_ids,
            key=lambda business_id: (
                -features.category_scores[business_id],
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
                "method": "category",
                "history_count": features.profile.history_count,
                "llm_attempted": False,
            },
        )
