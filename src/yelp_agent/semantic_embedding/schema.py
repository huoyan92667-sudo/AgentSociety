"""Stable contracts for Step 25 semantic embedding evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel


type EmbeddingInputType = Literal["query", "document"]


class SemanticDocument(StrictModel):
    """One deterministic text whose vector can be cached by its content hash."""

    source_id: str = Field(min_length=1)
    source_kind: Literal["query", "business_static", "review_chunk"]
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_version: str = Field(min_length=1)


class EmbeddingUsage(StrictModel):
    api_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    estimated_cost_cny: float = Field(ge=0)
    provider_latency_ms: float = Field(ge=0)


class SemanticBusinessMatch(StrictModel):
    business_id: str = Field(min_length=1)
    cutoff_time: datetime
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cosine_similarity: float = Field(ge=-1, le=1)
    normalized_score: float = Field(ge=0, le=1)
    semantic_rank: int = Field(ge=1)


class SemanticMatchResult(StrictModel):
    """Semantic evidence only; it is not the final recommendation ranking."""

    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    provider: Literal["dashscope"] = "dashscope"
    dimension: int = Field(gt=0)
    matches: list[SemanticBusinessMatch] = Field(min_length=1)
    usage: EmbeddingUsage

    @field_validator("matches")
    @classmethod
    def validate_unique_businesses(
        cls,
        values: list[SemanticBusinessMatch],
    ) -> list[SemanticBusinessMatch]:
        ids = [item.business_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic matches must contain unique businesses")
        return values

    @model_validator(mode="after")
    def validate_complete_ranks(self) -> SemanticMatchResult:
        ranks = sorted(item.semantic_rank for item in self.matches)
        if ranks != list(range(1, len(self.matches) + 1)):
            raise ValueError("semantic ranks must be contiguous")
        return self
