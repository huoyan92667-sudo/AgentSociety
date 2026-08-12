"""Target-blind contracts for protected candidate fusion and final ranking."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.schema import RecommendationRequest
from yelp_agent.semantic_ranking import CandidateSemanticScore


type CandidateChannel = Literal["history", "query"]


class CandidatePoolItem(StrictModel):
    business_id: str = Field(min_length=1)
    source_channels: list[CandidateChannel] = Field(min_length=1, max_length=2)
    history_rank: int | None = Field(default=None, ge=1, le=500)
    history_fusion_score: float | None = Field(default=None, gt=0)
    query_rank: int | None = Field(default=None, ge=1, le=500)
    query_fusion_score: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        expected = []
        if self.history_rank is not None:
            expected.append("history")
        if self.query_rank is not None:
            expected.append("query")
        if self.source_channels != expected:
            raise ValueError("candidate channels do not match source ranks")
        if (self.history_rank is None) != (self.history_fusion_score is None):
            raise ValueError("history rank and score must appear together")
        if (self.query_rank is None) != (self.query_fusion_score is None):
            raise ValueError("query rank and score must appear together")
        return self


class HardConstraintExclusion(StrictModel):
    business_id: str = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)


class CandidateEvidence(StrictModel):
    business_id: str = Field(min_length=1)
    source_channels: list[CandidateChannel] = Field(min_length=1, max_length=2)
    history_rank: int | None = Field(default=None, ge=1, le=500)
    query_rank: int | None = Field(default=None, ge=1, le=500)
    lightgbm_rank: int = Field(ge=1)
    lightgbm_model_score: float
    lightgbm_percentile: float = Field(ge=0, le=1)
    embedding_rank: int = Field(ge=1)
    embedding_score: float = Field(ge=0, le=1)
    embedding_percentile: float = Field(ge=0, le=1)
    query_retrieval_percentile: float = Field(ge=0, le=1)


class PreparedQueryAwareCase(StrictModel):
    """Visible evidence frozen before any benchmark label is loaded."""

    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "validation"]
    request: RecommendationRequest
    history_ranking: list[str] = Field(max_length=500)
    query_ranking: list[str] = Field(max_length=500)
    rrf_fusion_ranking: list[str] = Field(max_length=500)
    history_lightgbm_ranking: list[str] = Field(max_length=500)
    union_candidate_count: int = Field(ge=1, le=1000)
    eligible_candidate_count: int = Field(ge=1, le=1000)
    overlap_count: int = Field(ge=0, le=500)
    exclusions: list[HardConstraintExclusion]
    candidates: list[CandidateEvidence] = Field(min_length=1, max_length=1000)
    retrieval_latency_ms: float = Field(ge=0)
    preparation_latency_ms: float = Field(ge=0)
    embedding_input_tokens: int = Field(ge=0)
    embedding_logical_tokens: int = Field(ge=0)
    embedding_cache_hits: int = Field(ge=0)
    embedding_cache_misses: int = Field(ge=0)
    external_model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def validate_prepared_case(self) -> Self:
        candidate_ids = [item.business_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prepared candidates must be unique")
        if len(candidate_ids) != self.eligible_candidate_count:
            raise ValueError("eligible candidate count does not match evidence")
        if self.eligible_candidate_count > self.union_candidate_count:
            raise ValueError("eligible candidate count exceeds union count")
        excluded_ids = {item.business_id for item in self.exclusions}
        if excluded_ids.intersection(candidate_ids):
            raise ValueError("hard-excluded businesses escaped into evidence")
        for ranking in (
            self.history_ranking,
            self.query_ranking,
            self.rrf_fusion_ranking,
            self.history_lightgbm_ranking,
        ):
            if len(ranking) != len(set(ranking)):
                raise ValueError("prepared rankings must contain unique IDs")
        return self


class CoarseCandidateScore(StrictModel):
    business_id: str = Field(min_length=1)
    source_channels: list[CandidateChannel]
    history_rank: int | None = Field(default=None, ge=1)
    query_rank: int | None = Field(default=None, ge=1)
    lightgbm_rank: int = Field(ge=1)
    embedding_rank: int = Field(ge=1)
    query_signal: float = Field(ge=0, le=1)
    coarse_score: float = Field(ge=0, le=1)
    coarse_rank: int = Field(ge=1)
    final_rank: int | None = Field(default=None, ge=1)


class QueryAwareRankingResult(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "validation"]
    ranking: list[str] = Field(min_length=1, max_length=1000)
    top_10: list[str] = Field(min_length=1, max_length=10)
    displayed_top_5: list[str] = Field(min_length=1, max_length=5)
    coarse_scores: list[CoarseCandidateScore] = Field(min_length=1)
    semantic_scores: list[CandidateSemanticScore]
    hard_exclusions: list[HardConstraintExclusion]
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

    @field_validator("ranking", "top_10", "displayed_top_5")
    @classmethod
    def validate_unique_ranking(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError("ranking IDs must be nonempty and unique")
        return values

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.top_10 != self.ranking[: len(self.top_10)]:
            raise ValueError("top_10 must be the final ranking prefix")
        if self.displayed_top_5 != self.ranking[: len(self.displayed_top_5)]:
            raise ValueError("displayed results must be the final ranking prefix")
        score_ids = [item.business_id for item in self.coarse_scores]
        if set(score_ids) != set(self.ranking) or len(score_ids) != len(self.ranking):
            raise ValueError("coarse scores must cover the complete final ranking")
        if any(item.final_rank is None for item in self.coarse_scores):
            raise ValueError("every candidate requires a final rank")
        final_ranks = sorted(int(item.final_rank) for item in self.coarse_scores)
        if final_ranks != list(range(1, len(self.ranking) + 1)):
            raise ValueError("candidate final ranks must be contiguous")
        if self.fallback != (self.fallback_reason is not None):
            raise ValueError("fallback and reason must appear together")
        excluded = {item.business_id for item in self.hard_exclusions}
        if excluded.intersection(self.ranking):
            raise ValueError("hard-excluded businesses escaped into final ranking")
        return self
