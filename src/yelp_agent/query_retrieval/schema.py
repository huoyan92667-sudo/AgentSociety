"""Stable, label-free contracts for full-catalog Query retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.schema import RecommendationRequest

type QueryRouteName = Literal[
    "query_category",
    "query_embedding",
    "query_aspect",
    "query_location",
]


class QueryRetrievalTask(StrictModel):
    """One current request; labels and future behavior are deliberately absent."""

    request: RecommendationRequest
    usage_scope: str = Field(min_length=1)


class QueryRetrievalCandidate(StrictModel):
    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    fusion_score: float = Field(gt=0)
    route_count: int = Field(ge=1, le=4)
    category_rank: int | None = Field(default=None, ge=1)
    category_score: float | None = Field(default=None, ge=0, le=1)
    embedding_rank: int | None = Field(default=None, ge=1)
    embedding_score: float | None = Field(default=None, ge=0, le=1)
    aspect_rank: int | None = Field(default=None, ge=1)
    aspect_score: float | None = Field(default=None, ge=0, le=1)
    location_rank: int | None = Field(default=None, ge=1)
    location_score: float | None = Field(default=None, ge=0, le=1)
    distance_km: float | None = Field(default=None, ge=0)
    matched_fields: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    source_scope: Literal["selected_user_interactions"]


class ExcludedQueryBusiness(StrictModel):
    business_id: str = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)


class QueryRetrievalUsage(StrictModel):
    embedding_input_tokens: int = Field(default=0, ge=0)
    embedding_logical_tokens: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    provider_calls: int = Field(default=0, ge=0)


class QueryRetrievalResult(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    cutoff_time: datetime
    catalog_size: int = Field(ge=0)
    pre_cutoff_business_count: int = Field(ge=0)
    eligible_business_count: int = Field(ge=0)
    candidates: list[QueryRetrievalCandidate]
    excluded: list[ExcludedQueryBusiness]
    route_result_counts: dict[QueryRouteName, int]
    warnings: list[str]
    usage: QueryRetrievalUsage = Field(default_factory=QueryRetrievalUsage)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> QueryRetrievalResult:
        ids = [item.business_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Query retrieval candidate IDs must be unique")
        if [item.rank for item in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("Query retrieval ranks must be contiguous from one")
        expected_routes = {
            "query_category",
            "query_embedding",
            "query_aspect",
            "query_location",
        }
        if set(self.route_result_counts) != expected_routes:
            raise ValueError("Query retrieval must report all four route counts")
        if self.eligible_business_count > self.pre_cutoff_business_count:
            raise ValueError("eligible count cannot exceed pre-cutoff count")
        if self.pre_cutoff_business_count > self.catalog_size:
            raise ValueError("pre-cutoff count cannot exceed catalog size")
        return self


class DualChannelCandidate(StrictModel):
    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    fusion_score: float = Field(gt=0)
    channel_count: int = Field(ge=1, le=2)
    history_rank: int | None = Field(default=None, ge=1)
    history_score: float | None = Field(default=None, ge=0)
    query_rank: int | None = Field(default=None, ge=1)
    query_score: float | None = Field(default=None, ge=0)


class DualChannelRetrievalResult(StrictModel):
    candidates: list[DualChannelCandidate]
    history_candidate_count: int = Field(ge=0)
    query_candidate_count: int = Field(ge=0)
    overlap_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_candidates(self) -> DualChannelRetrievalResult:
        ids = [item.business_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("dual-channel candidate IDs must be unique")
        if [item.rank for item in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("dual-channel ranks must be contiguous from one")
        return self
