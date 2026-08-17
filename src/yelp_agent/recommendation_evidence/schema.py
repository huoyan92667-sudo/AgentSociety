"""Public contracts for evidence-aware recommendation and failure diagnosis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel


type EvidenceSourcePreference = Literal[
    "business_attribute",
    "review",
    "official_source",
]
type EvidenceFreshness = Literal[
    "historical_experience_acceptable",
    "current_required",
]
type EvidenceImportance = Literal["mandatory", "strong", "preferred"]
type GapSeverity = Literal["blocking", "optional"]
type ClaimStance = Literal[
    "supports",
    "contradicts",
    "partially_supports",
    "irrelevant",
    "insufficient",
]
type ClaimVerificationStatus = Literal[
    "officially_confirmed",
    "historically_supported",
    "historically_contradicted",
    "conflicting_evidence",
    "insufficient_evidence",
    "current_verification_required",
]
type RequirementStatus = Literal[
    "satisfied",
    "supported",
    "partially_supported",
    "contradicted",
    "unknown",
]
type RootCauseLayer = Literal[
    "query_understanding",
    "memory",
    "reference_resolution",
    "tool_selection",
    "retrieval",
    "hard_constraint",
    "ranking",
    "evidence_retrieval",
    "evidence_judgment",
    "answer_composition",
    "benchmark_label",
    "evaluator",
    "source_data",
]


class OpenRequirement(StrictModel):
    """A user goal preserved even when no frozen Aspect can represent it."""

    requirement_id: str = Field(pattern=r"^requirement_[0-9a-f]{12}$")
    text: str = Field(min_length=1, max_length=500)
    importance: EvidenceImportance = "preferred"
    evidence_span: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_turn_index: int | None = Field(default=None, ge=1)


class EvidenceNeed(StrictModel):
    """One claim the Agent should verify before making a strong recommendation."""

    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{12}$")
    claim: str = Field(min_length=1, max_length=500)
    source_preference: list[EvidenceSourcePreference] = Field(min_length=1)
    freshness: EvidenceFreshness
    importance: EvidenceImportance
    search_queries: list[str] = Field(default_factory=list, max_length=12)
    source_turn_index: int | None = Field(default=None, ge=1)

    @field_validator("source_preference", "search_queries")
    @classmethod
    def unique_nonempty_values(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values]
        if any(not value for value in cleaned):
            raise ValueError("evidence need lists cannot contain blank values")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("evidence need lists must contain unique values")
        return cleaned


class InformationGapAssessment(StrictModel):
    gap: str = Field(min_length=1)
    severity: GapSeverity
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")


class OpenClaimJudgment(StrictModel):
    """A source-bound stance judgment; it cannot create or alter Review text."""

    claim: str = Field(min_length=1, max_length=500)
    stance: ClaimStance
    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    evidence_span: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")


class ClaimJudgeOutput(StrictModel):
    judgments: list[OpenClaimJudgment] = Field(default_factory=list, max_length=25)


class EvidenceExcerpt(StrictModel):
    business_id: str = Field(min_length=1)
    source_type: EvidenceSourcePreference
    stance: ClaimStance
    review_id: str | None = None
    source_field: str | None = None
    source_url: str | None = None
    source_time: datetime | None = None
    accessed_at: datetime | None = None
    evidence_span: str = Field(min_length=1, max_length=2000)
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    local_evidence_path: str | None = None

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        if self.source_type == "review":
            if self.review_id is None or self.source_field is not None:
                raise ValueError("review evidence requires only review_id")
        elif self.source_type == "business_attribute":
            if self.source_field is None or self.review_id is not None:
                raise ValueError("attribute evidence requires only source_field")
        elif self.source_url is None:
            raise ValueError("official-source evidence requires source_url")
        return self


class BusinessClaimVerification(StrictModel):
    business_id: str = Field(min_length=1)
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{12}$")
    claim: str = Field(min_length=1, max_length=500)
    status: ClaimVerificationStatus
    confidence: float = Field(ge=0, le=1)
    supporting_evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    partial_evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    missing_sources: list[EvidenceSourcePreference] = Field(default_factory=list)
    current_policy_verified: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        evidence = (
            self.supporting_evidence
            + self.contradicting_evidence
            + self.partial_evidence
        )
        if any(item.business_id != self.business_id for item in evidence):
            raise ValueError("claim verification evidence escaped business scope")
        if self.current_policy_verified and self.status != "officially_confirmed":
            raise ValueError("only official evidence can verify current policy")
        return self


class BusinessClaimVerificationResult(StrictModel):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_ids: list[str] = Field(min_length=1, max_length=5)
    cutoff_time: datetime
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{12}$")
    claim: str = Field(min_length=1, max_length=500)
    freshness: EvidenceFreshness
    verifications: list[BusinessClaimVerification]
    provider_called: bool = False
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_hit: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if [item.business_id for item in self.verifications] != self.business_ids:
            raise ValueError("claim verifications must preserve business order")
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("claim-judge token usage must appear as a pair")
        return self


class BusinessClaimVerificationRequest(StrictModel):
    business_ids: list[str] = Field(min_length=1, max_length=5)
    cutoff_time: datetime
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{12}$")
    claim: str = Field(min_length=1, max_length=500)
    source_preference: list[EvidenceSourcePreference] = Field(min_length=1)
    freshness: EvidenceFreshness
    search_queries: list[str] = Field(default_factory=list, max_length=12)
    usage_scope: str = Field(min_length=1, max_length=200)

    @field_validator("business_ids", "source_preference", "search_queries")
    @classmethod
    def validate_unique_request_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("claim verification request values must be unique")
        return values


class RecommendationScoreBreakdown(StrictModel):
    history_score: float | None = None
    query_retrieval_score: float | None = None
    category_score: float | None = None
    distance_score: float | None = None
    price_match_score: float | None = None
    lightgbm_score: float | None = None
    embedding_score: float | None = None
    cross_encoder_score: float | None = None
    aspect_score: float | None = None
    review_evidence_score: float | None = None
    final_score: float | None = None


class RequirementMatch(StrictModel):
    requirement: str = Field(min_length=1, max_length=500)
    status: RequirementStatus
    source: str = Field(min_length=1)
    value: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class RecommendationEvidenceCard(StrictModel):
    rank: int = Field(ge=1, le=5)
    business_id: str = Field(min_length=1)
    business_name: str = Field(min_length=1)
    final_score: float | None = None
    matched_requirements: list[RequirementMatch] = Field(default_factory=list)
    supporting_evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    score_breakdown: RecommendationScoreBreakdown
    why_ranked_here: str = Field(min_length=1, max_length=1000)
    warnings: list[str] = Field(default_factory=list)


class EvidenceAwareCandidateScore(StrictModel):
    business_id: str = Field(min_length=1)
    base_rank: int = Field(ge=1)
    evidence_score: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)
    verified_claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    contradicted_claim_count: int = Field(ge=0)


class EvidenceAwareRankingResult(StrictModel):
    ranking: list[str] = Field(min_length=1)
    scored_candidates: list[EvidenceAwareCandidateScore] = Field(min_length=1)
    evidence_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_ranking(self) -> Self:
        if [item.business_id for item in self.scored_candidates] != self.ranking:
            raise ValueError("evidence-aware scores must preserve final ranking order")
        if len(self.ranking) != len(set(self.ranking)):
            raise ValueError("evidence-aware ranking must be unique")
        return self


class ClarificationImpact(StrictModel):
    turn_index: int = Field(ge=2)
    answered_gaps: list[str] = Field(min_length=1)
    before_ranking: list[str] = Field(default_factory=list)
    after_ranking: list[str] = Field(default_factory=list)
    new_search_queries: list[str] = Field(default_factory=list)
    new_evidence_count: int = Field(default=0, ge=0)
    top5_changed: bool
    utility_gain: float | None = None
    low_utility: bool


class AgentFailureRecord(StrictModel):
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_index: int = Field(ge=1)
    user_goal: str = Field(min_length=1, max_length=2000)
    stage: str = Field(min_length=1)
    attempted_action: str = Field(min_length=1)
    attempted_tool: str | None = None
    attempted_arguments_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    available_evidence: list[dict[str, Any]] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    failure_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    root_cause_layer: RootCauseLayer
    impact: str = Field(min_length=1)
    recovery_attempted: bool
    next_recommended_action: str | None = None
    user_visible_message: str = Field(min_length=1)
    human_review_status: Literal["pending", "confirmed", "corrected"] = "pending"
