"""Stable contracts for business-scoped, cutoff-safe Review RAG."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.reviews.schema import AspectName, AspectSentiment
from yelp_agent.semantic_embedding.schema import EmbeddingUsage


REVIEW_SEGMENT_SCHEMA = pa.schema(
    [
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("review_time", pa.timestamp("us"), nullable=False),
        pa.field("stars", pa.float64(), nullable=False),
        pa.field("useful", pa.int64(), nullable=False),
        pa.field("segment_index", pa.int32(), nullable=False),
        pa.field("char_start", pa.int32(), nullable=False),
        pa.field("char_end", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("review_text_sha256", pa.string(), nullable=False),
    ]
)


class ReviewSegment(StrictModel):
    """One immutable passage that still points back to its source Review ID."""

    segment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    stars: float = Field(ge=1, le=5)
    useful: int = Field(ge=0)
    segment_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("segment char_end must exceed char_start")
        return self


class SegmentAspectEvidence(StrictModel):
    aspect: AspectName
    sentiment: AspectSentiment
    confidence: float = Field(ge=0, le=1)
    evidence_span: str = Field(min_length=1)


class ReviewSearchRequest(StrictModel):
    """The complete public input to the Review RAG module."""

    query_text: str = Field(min_length=1, max_length=2000)
    business_ids: list[str] = Field(min_length=1, max_length=10)
    cutoff_time: datetime
    aspects: list[AspectName] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=5)
    usage_scope: str = Field(min_length=1, max_length=200)

    @field_validator("business_ids", "aspects")
    @classmethod
    def validate_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Review RAG scope and aspects must be unique")
        return values


class ReviewEvidenceHit(StrictModel):
    """One deduplicated source Review returned to the Agent and evaluator."""

    rank: int = Field(ge=1)
    segment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    stars: float = Field(ge=1, le=5)
    useful: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=2000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matched_aspects: list[AspectName] = Field(default_factory=list)
    aspect_sentiments: list[AspectSentiment] = Field(default_factory=list)
    aspect_evidence: list[SegmentAspectEvidence] = Field(default_factory=list)
    aspect_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    embedding_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1)


class ReviewSearchResult(StrictModel):
    """Cutoff-safe evidence plus local-model accounting."""

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_ids: list[str] = Field(min_length=1)
    cutoff_time: datetime
    hits: list[ReviewEvidenceHit]
    route_result_counts: dict[str, int]
    eligible_segment_count: int = Field(ge=0)
    embedding_provider: Literal["local"] | None = None
    embedding_model: str | None = None
    embedding_usage: EmbeddingUsage

    @model_validator(mode="after")
    def validate_scope_and_ranks(self) -> Self:
        if len(self.business_ids) != len(set(self.business_ids)):
            raise ValueError("business scope must be unique")
        if any(hit.business_id not in self.business_ids for hit in self.hits):
            raise ValueError("Review RAG returned evidence outside business scope")
        if len({hit.review_id for hit in self.hits}) != len(self.hits):
            raise ValueError("Review RAG hits must be unique by review_id")
        if [hit.rank for hit in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("Review RAG ranks must be contiguous")
        if any(hit.review_time >= self.cutoff_time for hit in self.hits):
            raise ValueError("Review RAG returned evidence at or after cutoff")
        if (self.embedding_provider is None) != (self.embedding_model is None):
            raise ValueError("embedding provider and model must appear together")
        return self


class ReviewRAGManifest(StrictModel):
    schema_version: Literal[1] = 1
    artifact_version: Literal["1.0.0"] = "1.0.0"
    source_reviews_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_review_count: int = Field(ge=1)
    nonempty_review_count: int = Field(ge=1)
    segment_count: int = Field(ge=1)
    business_count: int = Field(ge=1)
    truncated_review_count: int = Field(ge=0)


class ReviewRAGAuditReport(StrictModel):
    schema_version: Literal[1] = 1
    source_review_count: int = Field(ge=1)
    segment_count: int = Field(ge=1)
    distinct_review_count: int = Field(ge=1)
    distinct_business_count: int = Field(ge=1)
    duplicate_segment_ids: int = Field(ge=0)
    invalid_segment_offsets: int = Field(ge=0)
    source_reviews_without_segments: int = Field(ge=0)
    hidden_review_label_count: int = Field(ge=0)
    hidden_review_id_coverage: float = Field(ge=0, le=1)
    hidden_hash_match_rate: float = Field(ge=0, le=1)
    cutoff_violation_count: int = Field(ge=0)
    passed: bool
