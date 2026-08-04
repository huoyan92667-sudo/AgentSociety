"""Small interfaces implemented by recommendation adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from yelp_agent.models import Prediction, RecommendationTask


@runtime_checkable
class Ranker(Protocol):
    def rank(self, task: RecommendationTask) -> Prediction:
        """Return a complete ranking for one frozen recommendation task."""
        ...
