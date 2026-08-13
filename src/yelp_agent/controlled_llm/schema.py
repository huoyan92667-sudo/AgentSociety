"""Strict public contracts for the two controlled Step 29 LLM modules."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.agent_evaluation.schema import ResponseClaimTrace
from yelp_agent.decision_readiness import DecisionReadiness, TaskType
from yelp_agent.models import StrictModel
from yelp_agent.query.schema import (
    ConditionField,
    ConditionOperator,
    ConditionValue,
    MissingField,
    RecommendationRequest,
    RequirementImportance,
)

type LLMCapability = Literal[
    "semantic_interpretation",
    "answer_composition",
    "memory_update",
]
type ControlledLLMStatus = Literal[
    "success",
    "skipped",
    "disabled",
    "provider_failure",
    "invalid_output",
]


class ControlledLLMCallTrace(StrictModel):
    """One sanitized logical call; secrets and raw headers never enter traces."""

    call_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability: LLMCapability
    status: ControlledLLMStatus
    model: str | None = None
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_called: bool
    cache_hit: bool = False
    latency_ms: float = Field(ge=0)
    attempt_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_unknown: bool = False
    failure_reason: str | None = None
    provider_request_id: str | None = None
    context_id: str | None = Field(default=None, min_length=1, max_length=200)
    turn_index: int | None = Field(default=None, ge=1)
    created_at: datetime

    @model_validator(mode="after")
    def validate_usage_and_status(self) -> Self:
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input and output token counts must appear together")
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total token count must equal input plus output")
        if self.cache_hit and self.provider_called:
            raise ValueError("a cache hit cannot call the provider")
        if self.status == "success" and self.failure_reason is not None:
            raise ValueError("successful calls cannot have a failure reason")
        if self.status not in {"success", "skipped"} and not self.failure_reason:
            raise ValueError("non-success calls require a failure reason")
        return self


class SemanticConditionSuggestion(StrictModel):
    """A model suggestion that is not executable until code validates it."""

    field: ConditionField
    operator: ConditionOperator
    value: ConditionValue
    importance: RequirementImportance
    evidence_span: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)

    @field_validator("evidence_span")
    @classmethod
    def normalize_span(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("semantic evidence span cannot be blank")
        return stripped

    @model_validator(mode="after")
    def validate_field_value(self) -> Self:
        numeric_fields = {"distance_km", "budget_per_person", "price_level"}
        if self.field in numeric_fields and not isinstance(self.value, (int, float)):
            raise ValueError(f"{self.field} requires a numeric value")
        if self.field == "category" and not isinstance(self.value, str):
            raise ValueError("category requires a string value")
        if self.field not in numeric_fields | {"category"} and not isinstance(
            self.value, bool
        ):
            raise ValueError(f"{self.field} requires a boolean value")
        return self


class SemanticModelOutput(StrictModel):
    """Only JSON shape accepted from the semantic model."""

    task_type: TaskType
    task_type_confidence: float = Field(ge=0, le=1)
    conditions: list[SemanticConditionSuggestion] = Field(
        default_factory=list, max_length=12
    )
    party_size: int | None = Field(default=None, ge=1, le=100)
    missing_fields: list[MissingField] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list, max_length=5)
    overall_confidence: float = Field(ge=0, le=1)

    @field_validator("missing_fields")
    @classmethod
    def unique_missing_fields(cls, values: list[MissingField]) -> list[MissingField]:
        if len(values) != len(set(values)):
            raise ValueError("semantic missing fields must be unique")
        return values


class SemanticEnhancementInput(StrictModel):
    base_request: RecommendationRequest
    base_readiness: DecisionReadiness
    language: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_request_alignment(self) -> Self:
        if self.base_readiness.request_id != self.base_request.request_id:
            raise ValueError("request and readiness must align")
        return self


class SemanticEnhancementResult(StrictModel):
    request: RecommendationRequest
    readiness: DecisionReadiness
    status: ControlledLLMStatus
    trace: ControlledLLMCallTrace
    accepted_signal_count: int = Field(ge=0)
    rejected_signals: list[str] = Field(default_factory=list)


class AnswerEvidenceItem(StrictModel):
    evidence_code: str = Field(pattern=r"^E[1-9][0-9]*$")
    claim: ResponseClaimTrace


class AnswerCompositionInput(StrictModel):
    context_id: str = Field(min_length=1, max_length=200)
    turn_index: int = Field(ge=1)
    query_text: str = Field(min_length=1, max_length=2000)
    language: str = Field(min_length=1)
    task_type: TaskType
    response_kind: Literal["grounded_answer", "uncertain_answer"]
    allowed_business_ids: list[str]
    evidence: list[AnswerEvidenceItem] = Field(min_length=1, max_length=20)
    reported_conflict: bool
    reported_evidence_recency: bool
    recommended_official_verification: bool

    @model_validator(mode="after")
    def validate_evidence_scope(self) -> Self:
        if len(self.allowed_business_ids) != len(set(self.allowed_business_ids)):
            raise ValueError("allowed business IDs must be unique")
        allowed = set(self.allowed_business_ids)
        codes = [item.evidence_code for item in self.evidence]
        if len(codes) != len(set(codes)):
            raise ValueError("evidence codes must be unique")
        if any(
            item.claim.business_id is not None
            and item.claim.business_id not in allowed
            for item in self.evidence
        ):
            raise ValueError("answer evidence escaped the allowed business scope")
        return self


class ComposedSentence(StrictModel):
    text: str = Field(min_length=1, max_length=1000)
    business_id: str | None = None
    evidence_codes: list[str] = Field(min_length=1, max_length=5)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("evidence_codes")
    @classmethod
    def unique_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("a sentence cannot repeat an evidence code")
        return values


class AnswerModelOutput(StrictModel):
    sentences: list[ComposedSentence] = Field(min_length=1, max_length=8)
    contains_uncertainty: bool
    recommends_official_verification: bool


class AnswerCompositionResult(StrictModel):
    claims: list[ResponseClaimTrace]
    status: ControlledLLMStatus
    trace: ControlledLLMCallTrace
    used_deterministic_fallback: bool
