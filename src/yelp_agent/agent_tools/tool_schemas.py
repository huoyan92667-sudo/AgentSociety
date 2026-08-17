"""Input and output contracts for the real Step 23 deterministic tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.learning_to_rank.runtime import HybridV2ScoredCandidate
from yelp_agent.models import StrictModel
from yelp_agent.profiles.schema import UserProfileV1
from yelp_agent.semantic_embedding.schema import (
    EmbeddingUsage,
    SemanticBusinessMatch,
)
from yelp_agent.cross_encoder.schema import (
    CrossEncoderBusinessMatch,
    CrossEncoderUsage,
)
from yelp_agent.review_rag.schema import ReviewSearchResult
from yelp_agent.evidence_aggregation.schema import EvidenceAssessment
from yelp_agent.semantic_ranking import SemanticRankingResult
from yelp_agent.session_memory.schema import EffectiveSessionRequest, RouterMemoryContext


class EmptyToolInput(StrictModel):
    """A tool whose identity and cutoff come exclusively from Agent state."""


class UserProfileOutput(StrictModel):
    profile: UserProfileV1


class SessionMemoryOutput(StrictModel):
    session_id: str = Field(min_length=1)
    turn_index: int = Field(ge=1)
    observations: list[dict[str, Any]]
    memory_context: RouterMemoryContext | None = None
    effective_request: EffectiveSessionRequest | None = None


class BusinessIdsInput(StrictModel):
    business_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("business_ids")
    @classmethod
    def validate_unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("business_ids must be unique")
        if any(not value or value != value.strip() for value in values):
            raise ValueError("business_ids must be nonempty and trimmed")
        return values


class CandidateBusinessIdsInput(BusinessIdsInput):
    business_ids: list[str] = Field(min_length=1, max_length=500)


class BusinessDetail(StrictModel):
    business_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str
    city: str
    state: str
    postal_code: str
    latitude: float | None = None
    longitude: float | None = None
    categories: list[str]
    structured_attributes: dict[str, Any]
    quality_score: float = Field(ge=0, le=1)
    review_count: int = Field(ge=0)


class BusinessDetailsOutput(StrictModel):
    businesses: list[BusinessDetail]


class BusinessProfilesOutput(StrictModel):
    profiles: list[BusinessProfileV1]


class RetrievalCandidateOutput(StrictModel):
    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    fusion_score: float = Field(gt=0)
    route_count: int = Field(ge=1)
    quality_rank: int | None = Field(default=None, ge=1)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    category_rank: int | None = Field(default=None, ge=1)
    category_score: float | None = Field(default=None, ge=0, le=1)
    text_rank: int | None = Field(default=None, ge=1)
    text_score: float | None = Field(default=None, ge=0, le=1)
    location_rank: int | None = Field(default=None, ge=1)
    location_score: float | None = Field(default=None, ge=0, le=1)
    distance_km: float | None = Field(default=None, ge=0)
    item_knn_rank: int | None = Field(default=None, ge=1)
    item_knn_positive_score: float = Field(ge=0)
    item_knn_negative_evidence: float = Field(ge=0)
    item_knn_positive_support_count: int = Field(ge=0)
    item_knn_negative_support_count: int = Field(ge=0)
    item_knn_positive_neighbor_count: int = Field(ge=0)
    item_knn_negative_neighbor_count: int = Field(ge=0)
    item_knn_missing: bool
    history_rank: int | None = Field(default=None, ge=1)
    history_fusion_score: float | None = Field(default=None, ge=0)
    query_rank: int | None = Field(default=None, ge=1)
    query_fusion_score: float | None = Field(default=None, ge=0)
    query_category_rank: int | None = Field(default=None, ge=1)
    query_category_score: float | None = Field(default=None, ge=0, le=1)
    query_embedding_rank: int | None = Field(default=None, ge=1)
    query_embedding_score: float | None = Field(default=None, ge=0, le=1)
    query_aspect_rank: int | None = Field(default=None, ge=1)
    query_aspect_score: float | None = Field(default=None, ge=0, le=1)
    query_location_rank: int | None = Field(default=None, ge=1)
    query_location_score: float | None = Field(default=None, ge=0, le=1)
    query_distance_km: float | None = Field(default=None, ge=0)
    source_channels: list[Literal["history", "query"]] = Field(default_factory=list)


class CandidateRetrievalOutput(StrictModel):
    candidate_business_ids: list[str]
    candidates: list[RetrievalCandidateOutput]
    catalog_size: int = Field(ge=0)
    eligible_candidate_count: int = Field(ge=0)
    excluded_history_businesses: int = Field(ge=0)
    route_result_counts: dict[str, int]
    retrieval_mode: Literal[
        "history_only",
        "dual_channel",
        "history_fallback",
    ] = "history_only"
    history_candidate_count: int = Field(default=0, ge=0)
    query_candidate_count: int = Field(default=0, ge=0)
    overlap_count: int = Field(default=0, ge=0)
    query_pre_cutoff_business_count: int = Field(default=0, ge=0)
    query_eligible_business_count: int = Field(default=0, ge=0)
    query_warnings: list[str] = Field(default_factory=list)
    excluded_session_rejections: int = Field(default=0, ge=0)


class ExcludedBusiness(StrictModel):
    business_id: str = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)


class ConstraintOutput(StrictModel):
    candidate_business_ids: list[str]
    excluded: list[ExcludedBusiness]


class CompareBusinessesInput(BusinessIdsInput):
    business_ids: list[str] = Field(min_length=2, max_length=10)


class SearchBusinessReviewsInput(BusinessIdsInput):
    """A locked business scope; Review RAG always returns at most Top-5."""

    business_ids: list[str] = Field(min_length=1, max_length=10)
    top_k: int = Field(default=5, ge=1, le=5)


class SearchBusinessReviewsOutput(ReviewSearchResult):
    """Named Agent-tool output while preserving the Review RAG contract."""


class AggregateReviewEvidenceInput(BusinessIdsInput):
    business_ids: list[str] = Field(min_length=1, max_length=10)


class AggregateReviewEvidenceOutput(EvidenceAssessment):
    """Named Agent-tool output preserving the aggregation contract."""


class ComparedBusiness(StrictModel):
    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    query_score: float = Field(ge=0, le=1)
    matched_fields: list[str]
    unmatched_fields: list[str]
    unknown_fields: list[str]


class BusinessComparisonOutput(StrictModel):
    ranking: list[str]
    compared: list[ComparedBusiness]
    excluded: list[ExcludedBusiness]


class HybridRankingOutput(StrictModel):
    ranking: list[str] = Field(min_length=1)
    scored_candidates: list[HybridV2ScoredCandidate] = Field(min_length=1)


class EmbeddingMatchOutput(StrictModel):
    """Semantic evidence; final ordering remains a deterministic policy decision."""

    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    matches: list[SemanticBusinessMatch] = Field(min_length=1)
    usage: EmbeddingUsage


class CrossEncoderMatchOutput(StrictModel):
    """Fine-grained evidence; final ordering remains code-enforced fusion."""

    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    matches: list[CrossEncoderBusinessMatch] = Field(min_length=1)
    usage: CrossEncoderUsage


class SemanticRankingOutput(SemanticRankingResult):
    """Named Agent-tool output preserving the Step-30 ranking contract."""


class QueryAwareRankingOutput(StrictModel):
    """Agent-safe projection of the complete Step-33 ranking result."""

    candidate_business_ids: list[str] = Field(min_length=1, max_length=1000)
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "validation"]
    ranking: list[str] = Field(min_length=1, max_length=1000)
    top_10: list[str] = Field(min_length=1, max_length=10)
    displayed_top_5: list[str] = Field(min_length=1, max_length=5)
    coarse_scores: list[dict[str, Any]] = Field(min_length=1)
    semantic_scores: list[dict[str, Any]]
    hard_exclusions: list[dict[str, Any]]
    query_weight: float = Field(ge=0, le=1)
    semantic_candidate_limit: int = Field(ge=5, le=100)
    fallback: bool = False
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0)
    embedding_input_tokens: int = Field(ge=0)
    cross_encoder_input_tokens: int = Field(ge=0)
    logical_input_tokens: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    external_model_calls: Literal[0] = 0

    @field_validator("candidate_business_ids", "ranking", "top_10", "displayed_top_5")
    @classmethod
    def validate_unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError("ranking IDs must be nonempty and unique")
        return values
