"""Small interfaces implemented by recommendation adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from yelp_agent.models import Prediction, RecommendationTask


class CandidateScoringRequest(Protocol):
    """The fields needed to score any nonempty candidate collection."""

    task_id: str
    user_id: str
    cutoff_time: datetime
    candidate_business_ids: list[str]


@runtime_checkable
class Ranker(Protocol):
    def rank(self, task: RecommendationTask) -> Prediction:
        """Return a complete ranking for one frozen recommendation task."""
        ...
