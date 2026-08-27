"""Unified task features consumed by Hybrid ranking and later Agent tools."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Protocol

from pydantic import Field, model_validator

from yelp_agent.config import HybridConfig
from yelp_agent.features.category import CategoryTaskFeatures
from yelp_agent.features.location import LocationTaskFeatures
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.features.text import TextTaskFeatures
from yelp_agent.models import (
    RecommendationTask,
    StrictModel,
    UnitScore,
    UserProfile,
)


class HybridFeatureError(RuntimeError):
    """Raised when component feature providers disagree on a task."""


class HybridWeights(StrictModel):
    category: UnitScore
    text: UnitScore
    quality: UnitScore
    location: UnitScore

    @model_validator(mode="after")
    def validate_sum(self) -> "HybridWeights":
        if not math.isclose(sum(self.as_dict.values()), 1.0, abs_tol=1e-9):
            raise ValueError("hybrid weights must sum to 1")
        return self

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "category": self.category,
            "text": self.text,
            "quality": self.quality,
            "location": self.location,
        }

    @classmethod
    def from_config(cls, config: HybridConfig) -> "HybridWeights":
        return cls(**config.weights)


class HybridComponentScore(StrictModel):
    business_id: str = Field(min_length=1)
    category_score: UnitScore
    text_score: UnitScore
    quality_score: UnitScore
    location_score: UnitScore


class HybridComponentFeatures(StrictModel):
    profile: UserProfile
    business_scores: dict[str, HybridComponentScore]


class _CategoryProvider(Protocol):
    def features_for(
        self,
        task: RecommendationTask,
    ) -> CategoryTaskFeatures: ...


class _TextProvider(Protocol):
    def features_for(self, task: RecommendationTask) -> TextTaskFeatures: ...


class _QualityProvider(Protocol):
    def score_businesses(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessQuality]: ...


class _LocationProvider(Protocol):
    def features_for(
        self,
        task: RecommendationTask,
    ) -> LocationTaskFeatures: ...


def _require_candidate_keys(
    task: RecommendationTask,
    provider_name: str,
    values: dict[str, object],
) -> None:
    if set(values) != set(task.candidate_business_ids):
        raise HybridFeatureError(
            f"{provider_name} scores do not match candidates for {task.task_id!r}"
        )


class HybridFeatureStore:
    """Aggregate four independently tested feature providers at one seam."""

    def __init__(
        self,
        *,
        category_store: _CategoryProvider,
        text_store: _TextProvider,
        quality_store: _QualityProvider,
        location_store: _LocationProvider,
    ) -> None:
        self._category_store = category_store
        self._text_store = text_store
        self._quality_store = quality_store
        self._location_store = location_store
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            HybridComponentFeatures,
        ] = {}

    def features_for(self, task: RecommendationTask) -> HybridComponentFeatures:
        """Return a merged profile and four scores for every task candidate."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        category = self._category_store.features_for(task)
        text = self._text_store.features_for(task)
        quality = self._quality_store.score_businesses(
            task.candidate_business_ids,
            task.cutoff_time,
        )
        location = self._location_store.features_for(task)
        _require_candidate_keys(task, "category", category.category_scores)
        _require_candidate_keys(task, "text", text.business_scores)
        _require_candidate_keys(task, "quality", quality)
        _require_candidate_keys(task, "location", location.business_scores)
        if category.profile.user_id != task.user_id:
            raise HybridFeatureError(
                f"Category profile user does not match task {task.task_id!r}"
            )

        profile_payload = category.profile.model_dump()
        profile_payload.update(
            {
                "positive_keywords": text.positive_keywords,
                "negative_keywords": text.negative_keywords,
                "location_center": (
                    location.location_center.model_dump()
                    if location.location_center is not None
                    else None
                ),
            }
        )
        profile = UserProfile.model_validate(profile_payload)
        business_scores = {
            business_id: HybridComponentScore(
                business_id=business_id,
                category_score=category.category_scores[business_id],
                text_score=text.business_scores[business_id].text_score,
                quality_score=quality[business_id].quality_score,
                location_score=(
                    location.business_scores[business_id].location_score
                ),
            )
            for business_id in task.candidate_business_ids
        }
        features = HybridComponentFeatures(
            profile=profile,
            business_scores=business_scores,
        )
        self._cache[cache_key] = features
        return features

