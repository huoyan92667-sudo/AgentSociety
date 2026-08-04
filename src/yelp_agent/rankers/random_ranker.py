"""Deterministic task-level random recommendation baseline."""

from __future__ import annotations

import hashlib
import random
from time import perf_counter

from yelp_agent.models import Prediction, RecommendationTask


class RandomRanker:
    """Shuffle candidates reproducibly without reading features or truth."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def rank(self, task: RecommendationTask) -> Prediction:
        started_at = perf_counter()
        ranking = list(task.candidate_business_ids)
        digest = hashlib.sha256(
            f"{self._seed}:random-ranker:{task.task_id}".encode("utf-8")
        ).digest()
        random.Random(int.from_bytes(digest, "big")).shuffle(ranking)
        latency_ms = (perf_counter() - started_at) * 1000.0
        return Prediction(
            task_id=task.task_id,
            ranking=ranking,
            latency_ms=latency_ms,
            fallback=False,
            tool_calls=0,
            llm_tokens=None,
            metadata={
                "method": "random",
                "seed": self._seed,
                "llm_attempted": False,
            },
        )
