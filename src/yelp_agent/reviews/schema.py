"""Stable records shared by Review Aspect extractors and stores."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pyarrow as pa
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

ASPECT_NAMES = (
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
)

type AspectName = Literal[
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "crowded",
    "queue_time",
    "portion_size",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "spiciness",
    "cleanliness",
]
type AspectSentiment = Literal["positive", "negative", "neutral", "mixed"]


class ReviewDocument(StrictModel):
    """One immutable source review passed to an extractor."""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    text: str = Field(min_length=1)


class ReviewAspectRecord(StrictModel):
    """One traceable aspect assertion supported by an exact source span."""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    aspect: AspectName
    sentiment: AspectSentiment
    confidence: float = Field(ge=0, le=1)
    evidence_span: str = Field(min_length=1)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extractor_name: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_offsets(self) -> "ReviewAspectRecord":
        if self.evidence_end <= self.evidence_start:
            raise ValueError("evidence_end must be greater than evidence_start")
        if self.evidence_end - self.evidence_start != len(self.evidence_span):
            raise ValueError("evidence offsets must match evidence_span length")
        return self


REVIEW_ASPECT_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("review_time", pa.timestamp("us"), nullable=False),
        pa.field("aspect", pa.string(), nullable=False),
        pa.field("sentiment", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("evidence_span", pa.string(), nullable=False),
        pa.field("evidence_start", pa.int64(), nullable=False),
        pa.field("evidence_end", pa.int64(), nullable=False),
        pa.field("source_text_sha256", pa.string(), nullable=False),
        pa.field("extractor_name", pa.string(), nullable=False),
        pa.field("extractor_version", pa.string(), nullable=False),
    ]
)
