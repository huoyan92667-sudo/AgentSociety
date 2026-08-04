"""Task-specific user category profiles and candidate affinity scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.models import (
    RecommendationTask,
    StrictModel,
    UnitScore,
    UserProfile,
)


class CategoryFeatureError(RuntimeError):
    """Raised when a task profile cannot be built without leakage."""


class CategoryTaskFeatures(StrictModel):
    profile: UserProfile
    category_scores: dict[str, UnitScore]


@dataclass(frozen=True)
class _HistoryInteraction:
    position: int
    review_id: str
    user_id: str
    business_id: str
    stars: float
    date: datetime


class TemporalCategoryStore:
    """Build category features solely from each frozen task history."""

    def __init__(
        self,
        businesses_path: str | Path,
        interactions_path: str | Path,
        histories_path: str | Path,
        *,
        broad_categories: set[str],
    ) -> None:
        self._broad_categories = {
            category.strip()
            for category in broad_categories
            if category.strip()
        }
        self._business_categories = self._load_business_categories(
            Path(businesses_path)
        )
        self._histories = self._load_histories(
            Path(interactions_path),
            Path(histories_path),
        )
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            CategoryTaskFeatures,
        ] = {}

    @staticmethod
    def _load_business_categories(path: Path) -> dict[str, frozenset[str]]:
        if not path.is_file():
            raise FileNotFoundError(f"Business Parquet does not exist: {path}")
        try:
            rows = pq.read_table(
                path,
                columns=["business_id", "categories"],
            ).to_pylist()
        except (OSError, pa.ArrowException) as exc:
            raise CategoryFeatureError(
                f"Could not read category businesses from {path}: {exc}"
            ) from exc

        businesses: dict[str, frozenset[str]] = {}
        for row in rows:
            business_id = row.get("business_id")
            categories = row.get("categories")
            if (
                not isinstance(business_id, str)
                or not business_id
                or not isinstance(categories, list)
            ):
                raise CategoryFeatureError(
                    "Category business data contains an invalid row"
                )
            if business_id in businesses:
                raise CategoryFeatureError(
                    f"Duplicate business_id in category data: {business_id!r}"
                )
            businesses[business_id] = frozenset(
                str(category).strip()
                for category in categories
                if str(category).strip()
            )
        if not businesses:
            raise CategoryFeatureError("Category business data is empty")
        return businesses

    def _load_histories(
        self,
        interactions_path: Path,
        histories_path: Path,
    ) -> dict[str, tuple[_HistoryInteraction, ...]]:
        for label, path in (
            ("Interaction", interactions_path),
            ("Temporal history", histories_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} Parquet does not exist: {path}")
        try:
            with duckdb.connect() as connection:
                interaction_counts = connection.execute(
                    """
                    SELECT count(*), count(DISTINCT review_id)
                    FROM read_parquet(?)
                    """,
                    [str(interactions_path)],
                ).fetchone()
                history_count = connection.execute(
                    "SELECT count(*) FROM read_parquet(?)",
                    [str(histories_path)],
                ).fetchone()[0]
                rows = connection.execute(
                    """
                    SELECT
                        history.task_id,
                        history.position,
                        interaction.review_id,
                        interaction.user_id,
                        interaction.business_id,
                        interaction.stars,
                        interaction.date
                    FROM read_parquet(?) AS history
                    JOIN read_parquet(?) AS interaction USING (review_id)
                    ORDER BY history.task_id, history.position
                    """,
                    [str(histories_path), str(interactions_path)],
                ).fetchall()
        except duckdb.Error as exc:
            raise CategoryFeatureError(
                f"Could not join frozen category histories: {exc}"
            ) from exc

        if interaction_counts[0] != interaction_counts[1]:
            raise CategoryFeatureError("Interaction review_id values must be unique")
        if len(rows) != history_count:
            raise CategoryFeatureError(
                "A frozen history review is missing from interactions"
            )

        grouped: defaultdict[str, list[_HistoryInteraction]] = defaultdict(list)
        for task_id, position, review_id, user_id, business_id, stars, date in rows:
            if (
                not task_id
                or not review_id
                or not user_id
                or not business_id
                or not isinstance(date, datetime)
            ):
                raise CategoryFeatureError(
                    "Frozen category history contains an invalid row"
                )
            rating = float(stars)
            if not 1.0 <= rating <= 5.0 or not rating.is_integer():
                raise CategoryFeatureError(
                    "Frozen category history contains a non-integer 1-5 rating"
                )
            business_key = str(business_id)
            if business_key not in self._business_categories:
                raise CategoryFeatureError(
                    f"History references unknown business_id {business_key!r}"
                )
            grouped[str(task_id)].append(
                _HistoryInteraction(
                    position=int(position),
                    review_id=str(review_id),
                    user_id=str(user_id),
                    business_id=business_key,
                    stars=rating,
                    date=date,
                )
            )

        result: dict[str, tuple[_HistoryInteraction, ...]] = {}
        for task_id, interactions in grouped.items():
            positions = [interaction.position for interaction in interactions]
            review_ids = [interaction.review_id for interaction in interactions]
            if (
                positions != list(range(1, len(interactions) + 1))
                or len(set(review_ids)) != len(review_ids)
            ):
                raise CategoryFeatureError(
                    f"Frozen history positions are invalid for task {task_id!r}"
                )
            result[task_id] = tuple(interactions)
        return result

    def features_for(self, task: RecommendationTask) -> CategoryTaskFeatures:
        """Return a dynamic profile and category scores for one frozen task."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        history = self._histories.get(task.task_id)
        if not history:
            raise CategoryFeatureError(
                f"No frozen history exists for task {task.task_id!r}"
            )
        for interaction in history:
            if interaction.user_id != task.user_id:
                raise CategoryFeatureError(
                    f"Task user does not match history for {task.task_id!r}"
                )
            if interaction.date >= task.cutoff_time:
                raise CategoryFeatureError(
                    f"History is not strictly before cutoff for {task.task_id!r}"
                )

        category_counts: Counter[str] = Counter()
        category_rating_sums: defaultdict[str, float] = defaultdict(float)
        rating_distribution = {str(stars): 0 for stars in range(1, 6)}
        rating_sum = 0.0
        for interaction in history:
            rating_sum += interaction.stars
            rating_distribution[str(int(interaction.stars))] += 1
            categories = self._business_categories[
                interaction.business_id
            ].difference(self._broad_categories)
            for category in categories:
                category_counts[category] += 1
                category_rating_sums[category] += interaction.stars

        preferences: dict[str, float] = {}
        if category_counts:
            max_category_count = max(category_counts.values())
            for category, count in category_counts.items():
                mean_rating = category_rating_sums[category] / count
                rating_affinity = (mean_rating - 1.0) / 4.0
                frequency = count / max_category_count
                preferences[category] = (
                    0.7 * rating_affinity + 0.3 * frequency
                )
        preferences = dict(sorted(preferences.items()))

        category_scores: dict[str, float] = {}
        for business_id in task.candidate_business_ids:
            categories = self._business_categories.get(business_id)
            if categories is None:
                raise CategoryFeatureError(
                    f"Task references unknown business_id {business_id!r}"
                )
            fine_categories = categories.difference(self._broad_categories)
            category_scores[business_id] = max(
                (preferences.get(category, 0.0) for category in fine_categories),
                default=0.0,
            )

        features = CategoryTaskFeatures(
            profile=UserProfile(
                user_id=task.user_id,
                history_count=len(history),
                average_rating=rating_sum / len(history),
                rating_distribution=rating_distribution,
                preferred_categories=preferences,
                disliked_categories={},
                preferred_city=None,
                location_center=None,
                positive_keywords=[],
                negative_keywords=[],
            ),
            category_scores=category_scores,
        )
        self._cache[cache_key] = features
        return features
