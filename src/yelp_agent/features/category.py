"""Task-specific user category profiles and candidate affinity scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from yelp_agent.data.temporal_view import TemporalDataView, TemporalDataError
from yelp_agent.models import (
    StrictModel,
    UnitScore,
    UserProfile,
)
from yelp_agent.protocols import CandidateScoringRequest


class CategoryFeatureError(RuntimeError):
    """Raised when a task profile cannot be built without leakage."""


class CategoryTaskFeatures(StrictModel):
    profile: UserProfile
    category_scores: dict[str, UnitScore]


class TemporalCategoryStore:
    """Build category features solely from each frozen task history."""

    def __init__(
        self,
        data_view: TemporalDataView,
        *,
        broad_categories: set[str],
    ) -> None:
        self._data_view = data_view
        self._broad_categories = {
            category.strip()
            for category in broad_categories
            if category.strip()
        }
        self._fine_categories_by_business = {
            business.business_id: tuple(
                sorted(
                    set(business.categories).difference(
                        self._broad_categories
                    )
                )
            )
            for business in data_view.businesses()
        }
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            CategoryTaskFeatures,
        ] = {}

    def score_candidates(
        self,
        task: CandidateScoringRequest,
    ) -> dict[str, float]:
        """Return category affinity only, avoiding profile model construction."""

        history = self._data_view.user_history(
            task.user_id,
            task.cutoff_time,
        )
        if not history:
            raise CategoryFeatureError(
                f"No frozen history exists for task {task.task_id!r}"
            )
        counts: Counter[str] = Counter()
        rating_sums: defaultdict[str, float] = defaultdict(float)
        for interaction in history:
            categories = self._fine_categories_by_business[
                interaction.business_id
            ]
            for category in categories:
                counts[category] += 1
                rating_sums[category] += interaction.stars
        preferences: dict[str, float] = {}
        if counts:
            maximum_count = max(counts.values())
            for category, count in counts.items():
                preferences[category] = (
                    0.7 * ((rating_sums[category] / count - 1.0) / 4.0)
                    + 0.3 * (count / maximum_count)
                )
        return {
            business_id: max(
                (
                    preferences.get(category, 0.0)
                    for category in self._fine_categories_by_business[
                        business_id
                    ]
                ),
                default=0.0,
            )
            for business_id in task.candidate_business_ids
        }

    def features_for(
        self,
        task: CandidateScoringRequest,
        *,
        cache: bool = True,
    ) -> CategoryTaskFeatures:
        """Return a dynamic profile and category scores for one frozen task."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key) if cache else None
        if cached is not None:
            return cached

        history = self._data_view.user_history(
            task.user_id,
            task.cutoff_time,
        )
        if not history:
            raise CategoryFeatureError(
                f"No frozen history exists for task {task.task_id!r}"
            )
        category_counts: Counter[str] = Counter()
        category_rating_sums: defaultdict[str, float] = defaultdict(float)
        rating_distribution = {str(stars): 0 for stars in range(1, 6)}
        rating_sum = 0.0
        for interaction in history:
            rating_sum += interaction.stars
            rating_distribution[str(int(interaction.stars))] += 1
            try:
                categories = set(
                    self._data_view.business(
                        interaction.business_id
                    ).categories
                ).difference(self._broad_categories)
            except TemporalDataError as exc:
                raise CategoryFeatureError(str(exc)) from exc
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
            try:
                categories = set(
                    self._data_view.business(business_id).categories
                )
            except TemporalDataError as exc:
                raise CategoryFeatureError(str(exc)) from exc
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
        if cache:
            self._cache[cache_key] = features
        return features
