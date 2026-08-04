"""Leak-resistant point-in-time business quality features."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
from pydantic import Field

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


@dataclass(frozen=True)
class _BusinessHistory:
    dates: tuple[datetime, ...]
    prefix_stars: tuple[float, ...]


def _prefix_sums(values: list[float]) -> tuple[float, ...]:
    prefix = [0.0]
    running = 0.0
    for value in values:
        running += value
        prefix.append(running)
    return tuple(prefix)


class TemporalQualityStore:
    """Score businesses using only reviews strictly earlier than a cutoff."""

    def __init__(
        self,
        reviews_path: str | Path,
        *,
        prior_count: int = 20,
    ) -> None:
        if prior_count <= 0:
            raise ValueError("prior_count must be positive")
        self._prior_count = prior_count
        self._histories, self._global_dates, self._global_prefix_stars = (
            self._load_reviews(Path(reviews_path))
        )
        self._calibration_cache: dict[datetime, tuple[float, float]] = {}

    @property
    def prior_count(self) -> int:
        return self._prior_count

    @staticmethod
    def _load_reviews(
        path: Path,
    ) -> tuple[
        dict[str, _BusinessHistory],
        tuple[datetime, ...],
        tuple[float, ...],
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Review Parquet does not exist: {path}")
        try:
            with duckdb.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT business_id, date, stars
                    FROM read_parquet(?)
                    ORDER BY date, business_id
                    """,
                    [str(path)],
                ).fetchall()
        except duckdb.Error as exc:
            raise QualityFeatureError(
                f"Could not read historical quality reviews from {path}: {exc}"
            ) from exc

        dated_stars: defaultdict[
            str, list[tuple[datetime, float]]
        ] = defaultdict(list)
        global_dates: list[datetime] = []
        global_stars: list[float] = []
        for business_id, date, stars in rows:
            if not business_id or not isinstance(date, datetime):
                raise QualityFeatureError(
                    "Historical quality data contains an invalid business or date"
                )
            rating = float(stars)
            if not 1.0 <= rating <= 5.0:
                raise QualityFeatureError(
                    "Historical quality data contains a rating outside 1 to 5"
                )
            business_key = str(business_id)
            dated_stars[business_key].append((date, rating))
            global_dates.append(date)
            global_stars.append(rating)

        if not rows:
            raise QualityFeatureError("Historical quality data contains no reviews")

        histories = {
            business_id: _BusinessHistory(
                dates=tuple(date for date, _ in values),
                prefix_stars=_prefix_sums(
                    [rating for _, rating in values]
                ),
            )
            for business_id, values in dated_stars.items()
        }
        return (
            histories,
            tuple(global_dates),
            _prefix_sums(global_stars),
        )

    def _calibration(self, cutoff_time: datetime) -> tuple[float, float]:
        cached = self._calibration_cache.get(cutoff_time)
        if cached is not None:
            return cached

        global_count = bisect_left(self._global_dates, cutoff_time)
        global_mean = (
            self._global_prefix_stars[global_count] / global_count
            if global_count
            else 3.5
        )
        positive_counts = [
            count
            for history in self._histories.values()
            if (count := bisect_left(history.dates, cutoff_time)) > 0
        ]
        popularity_p95 = (
            float(np.percentile(np.log1p(positive_counts), 95))
            if positive_counts
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
        """Return point-in-time quality details for the requested businesses."""

        if not business_ids:
            raise ValueError("business_ids cannot be empty")
        if len(set(business_ids)) != len(business_ids):
            raise ValueError("business_ids must be unique")

        global_mean, popularity_p95 = self._calibration(cutoff_time)
        scores: dict[str, BusinessQuality] = {}
        for business_id in business_ids:
            history = self._histories.get(business_id)
            count = (
                bisect_left(history.dates, cutoff_time)
                if history is not None
                else 0
            )
            rating_sum = (
                history.prefix_stars[count]
                if history is not None
                else 0.0
            )
            mean_rating = rating_sum / count if count else None
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
