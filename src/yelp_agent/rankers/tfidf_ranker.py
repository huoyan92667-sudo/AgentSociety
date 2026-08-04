"""Personalized positive/negative TF-IDF text recommendation baseline."""

from __future__ import annotations

from time import perf_counter

from yelp_agent.features.text import TemporalTextStore
from yelp_agent.models import Prediction, RecommendationTask


class TfidfRanker:
    """Rank candidates by point-in-time text preference similarity."""

    def __init__(self, text_store: TemporalTextStore) -> None:
        self._text_store = text_store

    def rank(self, task: RecommendationTask) -> Prediction:
        started_at = perf_counter()
        features = self._text_store.features_for(task)
        ranking = sorted(
            task.candidate_business_ids,
            key=lambda business_id: (
                -features.business_scores[business_id].text_score,
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
                "method": "tfidf",
                "positive_review_count": features.positive_review_count,
                "negative_review_count": features.negative_review_count,
                "llm_attempted": False,
            },
        )
