"""Stable contracts for Step 26 local Cross-Encoder evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel


class CrossEncoderUsage(StrictModel):
    scorer_calls: int = Field(default=0, ge=0)
    api_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    logical_input_tokens: int = Field(default=0, ge=0)
    cache_saved_tokens: int = Field(default=0, ge=0)
    truncated_pair_count: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    provider_latency_ms: float = Field(default=0, ge=0)


class CrossEncoderUsageEvent(StrictModel):
    usage_scope: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1)
    requested_pair_count: int = Field(ge=1)
    unique_pair_count: int = Field(ge=1)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    logical_input_tokens: int = Field(ge=0)
    scored_input_tokens: int = Field(ge=0)
    cache_saved_tokens: int = Field(ge=0)
    truncated_pair_count: int = Field(ge=0)
    scorer_calls: int = Field(ge=0)
    api_calls: Literal[0] = 0
    latency_ms: float = Field(ge=0)


class CrossEncoderBusinessMatch(StrictModel):
    business_id: str = Field(min_length=1)
    cutoff_time: datetime
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevance_score: float = Field(ge=0, le=1)
    cross_encoder_rank: int = Field(ge=1)


class CrossEncoderMatchResult(StrictModel):
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    provider: Literal["local"] = "local"
    matches: list[CrossEncoderBusinessMatch] = Field(min_length=1)
    usage: CrossEncoderUsage

    @field_validator("matches")
    @classmethod
    def validate_unique_businesses(
        cls, values: list[CrossEncoderBusinessMatch]
    ) -> list[CrossEncoderBusinessMatch]:
        ids = [item.business_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("cross-encoder matches must contain unique businesses")
        return values

    @model_validator(mode="after")
    def validate_contiguous_ranks(self) -> CrossEncoderMatchResult:
        ranks = sorted(item.cross_encoder_rank for item in self.matches)
        if ranks != list(range(1, len(self.matches) + 1)):
            raise ValueError("cross-encoder ranks must be contiguous")
        return self


class CrossEncoderPolicy(StrictModel):
    """Development-frozen policy loaded unchanged for validation."""

    schema_version: Literal[1] = 1
    candidate_limit: int = Field(ge=1, le=100)
    fusion_beta: float = Field(ge=0, le=1)
    display_limit: Literal[5] = 5
    objective: Literal["HR@5"] = "HR@5"
    tie_breakers: list[str] = Field(
        default_factory=lambda: ["MRR", "HR@1", "candidate_limit", "fusion_beta"]
    )
    development_scenario_count: int = Field(ge=0)
    development_metrics: dict[str, float]
