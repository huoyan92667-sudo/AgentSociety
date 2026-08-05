"""Leak-resistant point-in-time business quality features."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from pydantic import Field

from yelp_agent.data.temporal_view import ReviewStatistics, TemporalDataView
from yelp_agent.models import StrictModel, UnitScore


class QualityFeatureError(RuntimeError):
    """Raised when historical quality features cannot be built safely."""


class BusinessQuality(StrictModel):
    business_id: str = Field(min_length=1)
    review_count: int = Field(ge=0)
    mean_rating: float | None = Field(default=None, ge=1, le=5)
    bayesian_rating: float = Field(ge=1, le=5)
    normalized_bayesian_rating: UnitScore
    normalized_popularity: UnitScore
    quality_score: UnitScore


class TemporalQualityStore:
    """Turn time-safe review statistics into quality and popularity scores."""

    def __init__(
        self,
        data_view: TemporalDataView,
        *,
        prior_count: int = 20,
    ) -> None:
        if prior_count <= 0:
            raise ValueError("prior_count must be positive")
        self._data_view = data_view
        self._prior_count = prior_count
        self._calibration_cache: dict[datetime, tuple[float, float]] = {}

    @property
    def prior_count(self) -> int:
        return self._prior_count

    def _calibration(
        self,
        cutoff_time: datetime,
        statistics: ReviewStatistics,
    ) -> tuple[float, float]:
        cached = self._calibration_cache.get(cutoff_time)
        if cached is not None:
            return cached

        global_mean = statistics.global_mean_rating
        if global_mean is None:
            global_mean = 3.5
        popularity_p95 = (
            float(
                np.percentile(
                    np.log1p(statistics.positive_business_counts),
                    95,
                )
            )
            if statistics.positive_business_counts
            else 0.0
        )
        calibration = (global_mean, popularity_p95)
        self._calibration_cache[cutoff_time] = calibration
        return calibration

    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessQuality]:
        """Return point-in-time quality details for requested businesses."""

        if not business_ids:
            raise ValueError("business_ids cannot be empty")
        if len(set(business_ids)) != len(business_ids):
            raise ValueError("business_ids must be unique")

        statistics = self._data_view.review_statistics_before(
            business_ids,
            cutoff_time,
        )
        global_mean, popularity_p95 = self._calibration(
            cutoff_time,
            statistics,
        )
        scores: dict[str, BusinessQuality] = {}
        for business_id in business_ids:
            aggregate = statistics.businesses[business_id]
            count = aggregate.count
            rating_sum = aggregate.star_sum
            mean_rating = aggregate.mean_rating
            bayesian_rating = (
                rating_sum + self._prior_count * global_mean
            ) / (count + self._prior_count)
            normalized_rating = min(
                1.0,
                max(0.0, (bayesian_rating - 1.0) / 4.0),
            )
            normalized_popularity = (
                min(float(np.log1p(count)), popularity_p95)
                / popularity_p95
                if popularity_p95 > 0
                else 0.0
            )
            quality_score = min(
                1.0,
                max(
                    0.0,
                    0.8 * normalized_rating
                    + 0.2 * normalized_popularity,
                ),
            )
            scores[business_id] = BusinessQuality(
                business_id=business_id,
                review_count=count,
                mean_rating=mean_rating,
                bayesian_rating=bayesian_rating,
                normalized_bayesian_rating=normalized_rating,
                normalized_popularity=normalized_popularity,
                quality_score=quality_score,
            )
        return scores
