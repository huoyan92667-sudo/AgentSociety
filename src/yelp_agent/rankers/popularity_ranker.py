"""Point-in-time business quality and popularity baseline."""

from __future__ import annotations

from time import perf_counter

from yelp_agent.features.quality import TemporalQualityStore
from yelp_agent.models import Prediction, RecommendationTask


class PopularityRanker:
    """Rank candidates by historical quality without personalization."""

    def __init__(self, quality_store: TemporalQualityStore) -> None:
        self._quality_store = quality_store

    def rank(self, task: RecommendationTask) -> Prediction:
        started_at = perf_counter()
        scores = self._quality_store.score_businesses(
            task.candidate_business_ids,
            task.cutoff_time,
        )
        ranking = sorted(
            task.candidate_business_ids,
            key=lambda business_id: (
                -scores[business_id].quality_score,
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
                "method": "popularity",
                "prior_count": self._quality_store.prior_count,
                "llm_attempted": False,
            },
        )
