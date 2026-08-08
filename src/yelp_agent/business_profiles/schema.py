"""Stable contracts for point-in-time business knowledge."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from yelp_agent.features.quality import BusinessQuality
from yelp_agent.models import StrictModel
from yelp_agent.reviews.schema import ASPECT_NAMES, AspectName, AspectSentiment


class BusinessRatingEvent(StrictModel):
    """One compact rating event without review text."""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    stars: float = Field(ge=1, le=5)
    review_time: datetime


class BusinessAspectEvent(StrictModel):
    """One compact, traceable Review Aspect event without source text."""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    aspect: AspectName
    sentiment: AspectSentiment
    confidence: float = Field(ge=0, le=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extractor_version: str = Field(min_length=1)


class BusinessAspectSummary(StrictModel):
    """Aggregated evidence for one aspect of one business at one cutoff."""

    aspect: AspectName
    status: Literal["known", "unknown"]
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    mixed_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    unique_users: int = Field(ge=0)
    effective_evidence: float = Field(ge=0)
    positive_ratio: float | None = Field(default=None, ge=0, le=1)
    negative_ratio: float | None = Field(default=None, ge=0, le=1)
    weighted_positive_ratio: float | None = Field(default=None, ge=0, le=1)
    weighted_negative_ratio: float | None = Field(default=None, ge=0, le=1)
    latest_evidence_time: datetime | None = None
    confidence: float = Field(ge=0, le=1)
    conflict: bool

    @model_validator(mode="after")
    def validate_summary(self) -> BusinessAspectSummary:
        counts = (
            self.positive_count
            + self.negative_count
            + self.neutral_count
            + self.mixed_count
        )
        if counts != self.evidence_count:
            raise ValueError("aspect sentiment counts must sum to evidence_count")
        if self.unique_users > self.evidence_count:
            raise ValueError("unique_users cannot exceed evidence_count")
        if (self.evidence_count == 0) != (self.latest_evidence_time is None):
            raise ValueError("latest evidence must match whether evidence exists")
        ratios = (
            self.positive_ratio,
            self.negative_ratio,
            self.weighted_positive_ratio,
            self.weighted_negative_ratio,
        )
        if self.status == "unknown":
            if any(value is not None for value in ratios):
                raise ValueError("unknown aspects cannot expose directional ratios")
            if self.conflict:
                raise ValueError("unknown aspects cannot be marked as conflicting")
        else:
            if any(value is None for value in ratios):
                raise ValueError("known aspects require directional ratios")
            if not math.isclose(
                float(self.positive_ratio) + float(self.negative_ratio),
                1.0,
                abs_tol=1e-9,
            ):
                raise ValueError("raw directional ratios must sum to one")
            if not math.isclose(
                float(self.weighted_positive_ratio)
                + float(self.weighted_negative_ratio),
                1.0,
                abs_tol=1e-9,
            ):
                raise ValueError("weighted directional ratios must sum to one")
        return self


class BusinessProfileEvidenceSummary(StrictModel):
    rating_count: int = Field(ge=0)
    aspect_evidence_count: int = Field(ge=0)
    aspect_unique_users: int = Field(ge=0)
    known_aspect_count: int = Field(ge=0, le=len(ASPECT_NAMES))
    latest_rating_time: datetime | None = None
    latest_aspect_time: datetime | None = None


class BusinessProfileV1(StrictModel):
    """Immutable shared business knowledge at one exact cutoff."""

    profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_id: str = Field(min_length=1)
    cutoff_time: datetime
    name: str = Field(min_length=1)
    address: str
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    postal_code: str
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    categories: list[str]
    structured_attributes: dict[str, Any]
    quality: BusinessQuality
    aspect_summaries: dict[AspectName, BusinessAspectSummary]
    profile_reliability: float = Field(ge=0, le=1)
    evidence_summary: BusinessProfileEvidenceSummary
    source_scope: Literal["selected_user_interactions"]
    profile_version: Literal["1.0.0"]

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, values: list[str]) -> list[str]:
        if not values or any(not value or value != value.strip() for value in values):
            raise ValueError("categories must contain valid names")
        if len(set(values)) != len(values):
            raise ValueError("categories must be unique")
        return values

    @model_validator(mode="after")
    def validate_profile(self) -> BusinessProfileV1:
        if set(self.aspect_summaries) != set(ASPECT_NAMES):
            raise ValueError("aspect_summaries must contain the frozen taxonomy")
        if any(key != summary.aspect for key, summary in self.aspect_summaries.items()):
            raise ValueError("aspect summary keys must match their aspect names")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("business coordinates must be both present or both absent")
        if self.quality.business_id != self.business_id:
            raise ValueError("quality business_id must match profile business_id")
        return self


BUSINESS_RATING_EVENT_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("stars", pa.float64(), nullable=False),
        pa.field("review_time", pa.timestamp("us"), nullable=False),
    ]
)

BUSINESS_ASPECT_EVENT_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("review_time", pa.timestamp("us"), nullable=False),
        pa.field("aspect", pa.string(), nullable=False),
        pa.field("sentiment", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("source_text_sha256", pa.string(), nullable=False),
        pa.field("extractor_version", pa.string(), nullable=False),
    ]
)

BUSINESS_COVERAGE_SCHEMA = pa.schema(
    [
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("aspect", pa.string(), nullable=False),
        pa.field("evidence_count", pa.int64(), nullable=False),
        pa.field("review_count", pa.int64(), nullable=False),
        pa.field("unique_users", pa.int64(), nullable=False),
        pa.field("first_evidence_time", pa.timestamp("us"), nullable=False),
        pa.field("latest_evidence_time", pa.timestamp("us"), nullable=False),
    ]
)
