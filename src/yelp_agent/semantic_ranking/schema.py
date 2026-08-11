"""Stable contracts for Step 30 structured semantic ranking."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.schema import (
    ConditionField,
    ConditionOperator,
    ConditionSource,
    EnforcementMode,
    RequirementImportance,
)

type SemanticRankingMode = Literal["aggressive", "protected"]
type MatchStatus = Literal["matched", "unmatched", "unknown"]


class RankingIntentCondition(StrictModel):
    field: ConditionField
    operator: ConditionOperator
    value: str | int | float | bool
    importance: RequirementImportance
    enforcement: EnforcementMode
    source: ConditionSource
    confidence: float = Field(ge=0, le=1)
    evidence_span: str = Field(min_length=1)


class RankingIntent(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document: str = Field(min_length=1, max_length=5000)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conditions: list[RankingIntentCondition]
    rankable_condition_count: int = Field(ge=0)
    semantic_model_condition_count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0, le=1)


class ConditionMatch(StrictModel):
    field: ConditionField
    status: MatchStatus
    score: float = Field(ge=0, le=1)
    evidence_confidence: float = Field(ge=0, le=1)


class CandidateSemanticScore(StrictModel):
    business_id: str = Field(min_length=1)
    base_rank: int = Field(ge=1)
    embedding_score: float = Field(ge=0, le=1)
    cross_encoder_score: float = Field(ge=0, le=1)
    structured_score: float = Field(ge=0, le=1)
    evidence_coverage: float = Field(ge=0, le=1)
    semantic_score: float = Field(ge=0, le=1)
    semantic_rank: int = Field(ge=1)
    fused_score: float = Field(ge=0, le=1)
    final_rank: int = Field(ge=1)
    rank_movement: int
    matches: list[ConditionMatch]
    protection_reason_codes: list[str] = Field(default_factory=list)

    @field_validator("protection_reason_codes")
    @classmethod
    def unique_reasons(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("protection reason codes must be unique")
        return values


class SemanticRankingUsage(StrictModel):
    embedding_input_tokens: int = Field(ge=0)
    cross_encoder_input_tokens: int = Field(ge=0)
    logical_input_tokens: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    latency_ms: float = Field(ge=0)

    @property
    def actual_input_tokens(self) -> int:
        return self.embedding_input_tokens + self.cross_encoder_input_tokens


class SemanticRankingResult(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_id: str | None = Field(default=None, min_length=1, max_length=200)
    mode: SemanticRankingMode
    intent: RankingIntent
    base_ranking: list[str] = Field(min_length=1)
    ranking: list[str] = Field(min_length=1)
    candidate_scores: list[CandidateSemanticScore] = Field(default_factory=list)
    effective_alpha: float = Field(ge=0, le=1)
    no_op_reason: str | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    usage: SemanticRankingUsage

    @model_validator(mode="after")
    def validate_complete_permutation(self) -> Self:
        if len(self.base_ranking) != len(set(self.base_ranking)):
            raise ValueError("base ranking must be unique")
        if set(self.ranking) != set(self.base_ranking):
            raise ValueError("semantic ranking must preserve the complete candidate set")
        if len(self.ranking) != len(set(self.ranking)):
            raise ValueError("semantic ranking must be unique")
        score_ids = [item.business_id for item in self.candidate_scores]
        if len(score_ids) != len(set(score_ids)):
            raise ValueError("candidate scores must contain unique businesses")
        if not set(score_ids).issubset(self.base_ranking):
            raise ValueError("candidate scores escaped the base ranking")
        if not self.fallback and self.no_op_reason is None and not self.candidate_scores:
            raise ValueError("successful semantic ranking requires candidate scores")
        if self.fallback != (self.fallback_reason is not None):
            raise ValueError("fallback and fallback_reason must appear together")
        return self
