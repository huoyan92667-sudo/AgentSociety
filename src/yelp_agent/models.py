"""Stable data contracts shared by the recommendation pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


UnitScore = Annotated[float, Field(ge=0, le=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]
RECOMMENDATION_CANDIDATE_COUNT = 20


def _validate_business_id_permutation(
    values: list[str],
    *,
    field_name: str,
) -> list[str]:
    if len(values) != RECOMMENDATION_CANDIDATE_COUNT:
        raise ValueError(
            f"{field_name} must contain exactly "
            f"{RECOMMENDATION_CANDIDATE_COUNT} businesses"
        )
    if any(not value or value != value.strip() for value in values):
        raise ValueError(
            f"{field_name} IDs must be nonempty and contain no surrounding whitespace"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} IDs must be unique")
    return values


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecommendationTask(StrictModel):
    task_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    candidate_business_ids: list[str]

    @field_validator("candidate_business_ids")
    @classmethod
    def validate_candidates(cls, values: list[str]) -> list[str]:
        return _validate_business_id_permutation(
            values,
            field_name="candidate list",
        )


class GroundTruth(StrictModel):
    task_id: str = Field(min_length=1)
    target_business_id: str = Field(min_length=1)


class LocationCenter(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class UserProfile(StrictModel):
    user_id: str = Field(min_length=1)
    history_count: NonNegativeCount
    average_rating: float = Field(ge=0, le=5)
    rating_distribution: dict[str, NonNegativeCount]
    preferred_categories: dict[str, UnitScore]
    disliked_categories: dict[str, UnitScore]
    preferred_city: str | None = None
    location_center: LocationCenter | None = None
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)

    @field_validator("positive_keywords", "negative_keywords")
    @classmethod
    def validate_keywords(cls, values: list[str]) -> list[str]:
        if len(values) > 10:
            raise ValueError("keyword lists can contain at most 10 items")
        if any(not value or value != value.strip() for value in values):
            raise ValueError("keywords must be nonempty and contain no surrounding whitespace")
        if len(set(values)) != len(values):
            raise ValueError("keywords must be unique")
        return values

    @model_validator(mode="after")
    def validate_rating_distribution(self) -> "UserProfile":
        if set(self.rating_distribution) != {"1", "2", "3", "4", "5"}:
            raise ValueError("rating_distribution must contain keys 1 through 5")
        if sum(self.rating_distribution.values()) != self.history_count:
            raise ValueError("rating_distribution must sum to history_count")
        return self


class ScoreBreakdown(StrictModel):
    business_id: str = Field(min_length=1)
    category_score: UnitScore
    text_score: UnitScore
    quality_score: UnitScore
    location_score: UnitScore
    hybrid_score: UnitScore


class Prediction(StrictModel):
    task_id: str = Field(min_length=1)
    ranking: list[str]
    latency_ms: float = Field(ge=0)
    fallback: bool
    fallback_reason: str | None = None
    tool_calls: NonNegativeCount = 0
    llm_tokens: NonNegativeCount | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ranking")
    @classmethod
    def validate_ranking(cls, values: list[str]) -> list[str]:
        return _validate_business_id_permutation(values, field_name="ranking")

    @model_validator(mode="after")
    def validate_fallback_state(self) -> "Prediction":
        if self.fallback and (
            self.fallback_reason is None or not self.fallback_reason.strip()
        ):
            raise ValueError("fallback predictions must include a fallback_reason")
        if not self.fallback and self.fallback_reason is not None:
            raise ValueError("non-fallback predictions cannot include a fallback_reason")
        return self
