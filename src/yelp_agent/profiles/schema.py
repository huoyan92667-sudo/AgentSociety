"""Stable contracts for task-time user profiles and their evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from yelp_agent.models import LocationCenter, StrictModel

type PreferenceKind = Literal["category", "aspect", "price", "area"]
type PreferenceSource = Literal[
    "rating_category",
    "review_aspect",
    "business_price",
    "business_area",
]


class PreferenceSignal(StrictModel):
    """One aggregated preference with traceable temporal support."""

    kind: PreferenceKind
    value: str = Field(min_length=1)
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=1)
    effective_evidence: float = Field(gt=0)
    first_seen: datetime
    last_confirmed: datetime
    source: PreferenceSource

    @model_validator(mode="after")
    def validate_time_range(self) -> PreferenceSignal:
        if self.last_confirmed < self.first_seen:
            raise ValueError("last_confirmed cannot be earlier than first_seen")
        return self


class ProfileEvidenceSummary(StrictModel):
    category_evidence_count: int = Field(ge=0)
    aspect_evidence_count: int = Field(ge=0)
    price_evidence_count: int = Field(ge=0)
    area_evidence_count: int = Field(ge=0)
    first_interaction: datetime
    last_interaction: datetime


class UserProfileV1(StrictModel):
    """Immutable long-term memory derived strictly before one cutoff."""

    profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    history_length: int = Field(ge=1)
    average_rating: float = Field(ge=1, le=5)
    rating_distribution: dict[str, int]
    category_preferences: list[PreferenceSignal]
    category_dislikes: list[PreferenceSignal]
    aspect_preferences: list[PreferenceSignal]
    aspect_dislikes: list[PreferenceSignal]
    price_preference: PreferenceSignal | None = None
    frequent_areas: list[PreferenceSignal]
    location_center: LocationCenter | None = None
    reliability: float = Field(ge=0, le=1)
    evidence_summary: ProfileEvidenceSummary
    profile_version: Literal["1.0.0"]

    def preference_signals(self) -> tuple[PreferenceSignal, ...]:
        """Return every normalized signal in deterministic storage order."""

        values = (
            *self.category_preferences,
            *self.category_dislikes,
            *self.aspect_preferences,
            *self.aspect_dislikes,
            *(() if self.price_preference is None else (self.price_preference,)),
            *self.frequent_areas,
        )
        return tuple(sorted(values, key=lambda signal: (signal.kind, signal.value)))

    @field_validator(
        "category_preferences",
        "category_dislikes",
        "aspect_preferences",
        "aspect_dislikes",
        "frequent_areas",
    )
    @classmethod
    def validate_unique_signals(
        cls,
        values: list[PreferenceSignal],
    ) -> list[PreferenceSignal]:
        keys = [(signal.kind, signal.value) for signal in values]
        if len(set(keys)) != len(keys):
            raise ValueError("profile preference signals must be unique")
        return values

    @model_validator(mode="after")
    def validate_profile(self) -> UserProfileV1:
        if set(self.rating_distribution) != {"1", "2", "3", "4", "5"}:
            raise ValueError("rating_distribution must contain keys 1 through 5")
        if sum(self.rating_distribution.values()) != self.history_length:
            raise ValueError("rating_distribution must sum to history_length")
        if any(signal.score <= 0 for signal in self.category_preferences):
            raise ValueError("category preferences must have positive scores")
        if any(signal.score >= 0 for signal in self.category_dislikes):
            raise ValueError("category dislikes must have negative scores")
        if any(signal.kind != "aspect" for signal in self.aspect_preferences):
            raise ValueError("aspect preferences must contain aspect signals")
        if any(signal.kind != "aspect" for signal in self.aspect_dislikes):
            raise ValueError("aspect dislikes must contain aspect signals")
        return self


class TaskProfileLink(StrictModel):
    task_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_history_count: int = Field(ge=1)
    fold: int | None = Field(default=None, ge=1)
    sample_weight: float | None = Field(default=None, gt=0, le=1)


PROFILE_SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field("profile_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("cutoff_time", pa.timestamp("us"), nullable=False),
        pa.field("history_length", pa.int64(), nullable=False),
        pa.field("average_rating", pa.float64(), nullable=False),
        *[
            pa.field(f"rating_{stars}_count", pa.int64(), nullable=False)
            for stars in range(1, 6)
        ],
        pa.field("location_latitude", pa.float64()),
        pa.field("location_longitude", pa.float64()),
        pa.field("reliability", pa.float64(), nullable=False),
        pa.field("category_evidence_count", pa.int64(), nullable=False),
        pa.field("aspect_evidence_count", pa.int64(), nullable=False),
        pa.field("price_evidence_count", pa.int64(), nullable=False),
        pa.field("area_evidence_count", pa.int64(), nullable=False),
        pa.field("first_interaction", pa.timestamp("us"), nullable=False),
        pa.field("last_interaction", pa.timestamp("us"), nullable=False),
        pa.field("profile_version", pa.string(), nullable=False),
    ]
)

PREFERENCE_SIGNAL_SCHEMA = pa.schema(
    [
        pa.field("profile_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("cutoff_time", pa.timestamp("us"), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("evidence_count", pa.int64(), nullable=False),
        pa.field("effective_evidence", pa.float64(), nullable=False),
        pa.field("first_seen", pa.timestamp("us"), nullable=False),
        pa.field("last_confirmed", pa.timestamp("us"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

TASK_PROFILE_LINK_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("cutoff_time", pa.timestamp("us"), nullable=False),
        pa.field("profile_id", pa.string(), nullable=False),
        pa.field("expected_history_count", pa.int64(), nullable=False),
        pa.field("fold", pa.int32()),
        pa.field("sample_weight", pa.float64()),
    ]
)
