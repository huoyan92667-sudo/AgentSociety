"""Stable contracts for deterministic cross-review evidence aggregation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.decision_readiness.schema import TaskType
from yelp_agent.models import StrictModel
from yelp_agent.reviews.schema import AspectName
from yelp_agent.review_rag.schema import ReviewSearchResult


type EvidenceStance = Literal["supports", "contradicts", "neutral"]
type EvidenceConsensus = Literal["supports", "contradicts", "mixed", "insufficient"]
type EvidenceConfidenceLevel = Literal["insufficient", "low", "medium", "high"]
type EvidenceResponseMode = Literal["grounded", "uncertain"]
type DesiredPolarity = Literal["positive", "negative"]


class EvidenceAggregationRequest(StrictModel):
    """The complete public input to the aggregation module."""

    query_text: str = Field(min_length=1, max_length=2000)
    task_type: TaskType
    requested_aspects: list[AspectName] = Field(default_factory=list)
    desired_polarity_by_aspect: dict[AspectName, DesiredPolarity] = Field(
        default_factory=dict
    )
    explicit_uncertainty_request: bool = False
    search_result: ReviewSearchResult

    @field_validator("requested_aspects")
    @classmethod
    def validate_unique_aspects(cls, values: list[AspectName]) -> list[AspectName]:
        if len(values) != len(set(values)):
            raise ValueError("requested aspects must be unique")
        return values

    @model_validator(mode="after")
    def validate_polarity_scope(self) -> Self:
        if not set(self.desired_polarity_by_aspect).issubset(self.requested_aspects):
            raise ValueError("desired polarity must belong to a requested aspect")
        return self


class EvidenceAtom(StrictModel):
    """One normalized, traceable assertion used by the aggregator."""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    hit_rank: int = Field(ge=1, le=5)
    aspect: AspectName
    stance: EvidenceStance
    relevance_score: float = Field(ge=0, le=1)
    extraction_confidence: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    condition_tags: list[str] = Field(default_factory=list)
    evidence_span: str = Field(min_length=1, max_length=2000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("condition_tags")
    @classmethod
    def validate_unique_tags(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("condition tags must be unique")
        return values


class ConditionEvidenceGroup(StrictModel):
    condition_tag: str = Field(min_length=1)
    evidence_count: int = Field(ge=1)
    support_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    review_ids: list[str] = Field(min_length=1)


class AspectEvidenceAssessment(StrictModel):
    business_id: str = Field(min_length=1)
    aspect: AspectName
    consensus: EvidenceConsensus
    confidence_score: float = Field(ge=0, le=1)
    confidence_level: EvidenceConfidenceLevel
    response_mode: EvidenceResponseMode
    requires_caveat: bool
    has_conflict: bool
    evidence_count: int = Field(ge=0)
    directional_evidence_count: int = Field(ge=0)
    unique_user_count: int = Field(ge=0)
    support_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    support_mass: float = Field(ge=0)
    contradiction_mass: float = Field(ge=0)
    mean_relevance: float = Field(ge=0, le=1)
    mean_recency: float = Field(ge=0, le=1)
    consistency: float = Field(ge=0, le=1)
    source_diversity: float = Field(ge=0, le=1)
    sample_strength: float = Field(ge=0, le=1)
    mean_extraction_confidence: float = Field(ge=0, le=1)
    latest_evidence_time: datetime | None = None
    condition_groups: list[ConditionEvidenceGroup] = Field(default_factory=list)
    citation_review_ids: list[str] = Field(default_factory=list)
    atoms: list[EvidenceAtom] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts_and_citations(self) -> Self:
        if self.evidence_count != len(self.atoms):
            raise ValueError("evidence count must match normalized atoms")
        if self.directional_evidence_count != self.support_count + self.contradiction_count:
            raise ValueError("directional evidence count is inconsistent")
        if self.evidence_count != self.directional_evidence_count + self.neutral_count:
            raise ValueError("stance counts are inconsistent")
        known = {atom.review_id for atom in self.atoms}
        if not set(self.citation_review_ids).issubset(known):
            raise ValueError("citations must refer to normalized evidence")
        if len(self.citation_review_ids) != len(set(self.citation_review_ids)):
            raise ValueError("citation Review IDs must be unique")
        return self


class BusinessEvidenceAssessment(StrictModel):
    business_id: str = Field(min_length=1)
    aspects: list[AspectEvidenceAssessment]
    overall_response_mode: EvidenceResponseMode
    has_conflict: bool
    confidence_level: EvidenceConfidenceLevel
    latest_evidence_time: datetime | None = None

    @model_validator(mode="after")
    def validate_aspect_businesses(self) -> Self:
        if any(item.business_id != self.business_id for item in self.aspects):
            raise ValueError("aspect assessment escaped its business scope")
        if len({item.aspect for item in self.aspects}) != len(self.aspects):
            raise ValueError("business aspect assessments must be unique")
        return self


class EvidenceAssessment(StrictModel):
    """Deterministic conclusion consumed by the Agent and future LLM composer."""

    schema_version: Literal[1] = 1
    aggregation_version: Literal["1.0.0"] = "1.0.0"
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: str = Field(min_length=1)
    task_type: TaskType
    business_ids: list[str] = Field(min_length=1)
    cutoff_time: datetime
    businesses: list[BusinessEvidenceAssessment]
    is_official_information: Literal[False] = False
    recommend_official_verification: bool

    @model_validator(mode="after")
    def validate_scope_and_cutoff(self) -> Self:
        if len(self.business_ids) != len(set(self.business_ids)):
            raise ValueError("evidence business scope must be unique")
        if [item.business_id for item in self.businesses] != self.business_ids:
            raise ValueError("assessment businesses must preserve request scope")
        for business in self.businesses:
            for aspect in business.aspects:
                if any(atom.review_time >= self.cutoff_time for atom in aspect.atoms):
                    raise ValueError("evidence aggregation crossed the cutoff")
        return self
