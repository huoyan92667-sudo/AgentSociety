"""Authoritative session-memory contracts for Step 34."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.decision_readiness import DecisionReadiness, InformationGap, TaskType
from yelp_agent.models import StrictModel
from yelp_agent.query.schema import (
    ConditionField,
    ConditionOperator,
    ConditionValue,
    RecommendationRequest,
    RequirementImportance,
)

type MemoryLifetime = Literal["session", "long_term_candidate"]
type MemoryPatchOperation = Literal["add", "replace", "remove"]
type MemoryRequestMode = Literal["patch", "replace", "no_change"]
type RelativePreferenceField = Literal["distance", "price", "noise", "crowding"]
type RelativePreferenceDirection = Literal[
    "closer",
    "farther",
    "lower",
    "higher",
    "quieter",
    "less_crowded",
]
type MemoryExtractionStatus = Literal[
    "success",
    "deterministic_bootstrap",
    "rule_fallback",
    "disabled",
    "provider_failure",
    "invalid_output",
]


class MemoryReferenceMention(StrictModel):
    """A model-visible reference expression, never a trusted business ID."""

    reference_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    expression: str = Field(min_length=1, max_length=200)
    ordinal: int | None = Field(default=None, ge=1, le=100)
    explicit_business_id: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_target_hint(self) -> Self:
        if (self.ordinal is None) == (self.explicit_business_id is None):
            raise ValueError(
                "a reference requires exactly one ordinal or explicit business ID"
            )
        return self


class MemoryConditionPatch(StrictModel):
    """One proposed condition mutation; code still decides whether it is legal."""

    operation: MemoryPatchOperation
    field: ConditionField
    operator: ConditionOperator | None = None
    value: ConditionValue | None = None
    importance: RequirementImportance | None = None
    evidence_span: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    lifetime: MemoryLifetime = "session"

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if self.operation in {"add", "replace"} and any(
            value is None for value in (self.operator, self.value, self.importance)
        ):
            raise ValueError("add and replace patches require a complete condition")
        return self


class ClarificationAnswerProposal(StrictModel):
    """A proposed answer to one previously requested information gap."""

    information_gap: InformationGap
    value: str | int | float | bool = Field(union_mode="left_to_right")
    evidence_span: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)


class RelativePreference(StrictModel):
    """A comparison against prior results without inventing an exact threshold."""

    field: RelativePreferenceField
    direction: RelativePreferenceDirection
    evidence_span: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)
    lifetime: MemoryLifetime = "session"


class MemoryProposal(StrictModel):
    """Strict JSON emitted by an LLM. It is a proposal, not canonical memory."""

    request_mode: MemoryRequestMode = "patch"
    task_type: TaskType
    task_type_confidence: float = Field(ge=0, le=1)
    references: list[MemoryReferenceMention] = Field(default_factory=list, max_length=10)
    rejected_reference_ids: list[str] = Field(default_factory=list, max_length=10)
    condition_patches: list[MemoryConditionPatch] = Field(
        default_factory=list, max_length=20
    )
    clarification_answers: list[ClarificationAnswerProposal] = Field(
        default_factory=list, max_length=5
    )
    relative_preferences: list[RelativePreference] = Field(
        default_factory=list, max_length=8
    )
    party_size: int | None = Field(default=None, ge=1, le=100)
    semantic_summary: str | None = Field(default=None, min_length=1, max_length=1200)
    long_term_candidates: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0, le=1)
    uncertainty_reasons: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("relative_preferences")
    @classmethod
    def deduplicate_relative_preferences(
        cls,
        values: list[RelativePreference],
    ) -> list[RelativePreference]:
        unique: dict[tuple[str, str, str], RelativePreference] = {}
        for item in values:
            unique[(item.field, item.direction, item.evidence_span)] = item
        return list(unique.values())

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        ids = [item.reference_id for item in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("memory reference IDs must be unique")
        if not set(self.rejected_reference_ids).issubset(ids):
            raise ValueError("rejected references must exist in references")
        if len(self.rejected_reference_ids) != len(set(self.rejected_reference_ids)):
            raise ValueError("rejected reference IDs must be unique")
        return self


class ResolvedMemoryReference(StrictModel):
    reference_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    expression: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    resolution_source: Literal["ordinal", "explicit_visible_id"]


class MemoryExtractionTrace(StrictModel):
    """Sanitized usage and fallback record for one memory extraction."""

    status: MemoryExtractionStatus
    extractor: Literal["deepseek", "rule"]
    prompt_version: str = Field(min_length=1)
    provider_called: bool = False
    cache_hit: bool = False
    model: str | None = None
    latency_ms: float = Field(default=0, ge=0)
    attempt_count: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    primary_failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("memory input and output tokens must appear together")
        if self.total_tokens is not None and self.input_tokens is not None:
            if self.total_tokens != self.input_tokens + self.output_tokens:
                raise ValueError("memory total tokens must equal input plus output")
        if self.status in {
            "success",
            "deterministic_bootstrap",
            "rule_fallback",
        } and self.failure_reason:
            raise ValueError("successful memory extraction cannot have failure_reason")
        if self.status not in {
            "success",
            "deterministic_bootstrap",
            "rule_fallback",
        } and not self.failure_reason:
            raise ValueError("failed memory extraction requires failure_reason")
        return self


class PresentedCandidateSet(StrictModel):
    turn_index: int = Field(ge=1)
    business_ids: list[str] = Field(min_length=1, max_length=20)

    @field_validator("business_ids")
    @classmethod
    def unique_businesses(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("presented businesses must be unique")
        return values


class MemoryTurnRecord(StrictModel):
    turn_index: int = Field(ge=1)
    query_text: str = Field(min_length=1, max_length=2000)
    task_type: TaskType
    accepted_changes: list[str] = Field(default_factory=list)
    rejected_changes: list[str] = Field(default_factory=list)
    resolved_references: list[ResolvedMemoryReference] = Field(default_factory=list)
    extraction: MemoryExtractionTrace


class SessionMemory(StrictModel):
    """Canonical, serializable session state. LLM output never replaces it."""

    schema_version: Literal[1] = 1
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    revision: int = Field(default=1, ge=1)
    current_request: RecommendationRequest
    current_task_type: TaskType
    information_gaps: list[InformationGap] = Field(default_factory=list)
    rejected_business_ids: list[str] = Field(default_factory=list)
    last_presented_business_ids: list[str] = Field(default_factory=list)
    presented_candidate_sets: list[PresentedCandidateSet] = Field(default_factory=list)
    current_business_scope: list[str] = Field(default_factory=list)
    business_scope_known: bool = False
    clarification_answers: dict[str, Any] = Field(default_factory=dict)
    relative_preferences: list[RelativePreference] = Field(default_factory=list)
    semantic_summary: str | None = Field(default=None, min_length=1, max_length=1200)
    long_term_candidates: list[str] = Field(default_factory=list)
    recent_turns: list[MemoryTurnRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.current_request.session_id != self.session_id:
            raise ValueError("memory request and session must align")
        if self.current_request.user_id != self.user_id:
            raise ValueError("memory request and user must align")
        if self.current_request.cutoff_time != self.cutoff_time:
            raise ValueError("memory cutoff cannot change within a session")
        for values, label in (
            (self.information_gaps, "information gaps"),
            (self.rejected_business_ids, "rejected businesses"),
            (self.last_presented_business_ids, "last presented businesses"),
            (self.current_business_scope, "business scope"),
            (self.long_term_candidates, "long-term candidates"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"memory {label} must be unique")
        if self.business_scope_known and not set(
            self.last_presented_business_ids
        ).issubset(self.current_business_scope):
            raise ValueError("presented businesses must remain inside known scope")
        return self


class RouterMemoryContext(StrictModel):
    """Compact, authoritative state exposed to tools and the future LLM Router."""

    revision: int = Field(ge=1)
    task_type: TaskType
    hard_constraints: list[dict[str, Any]]
    soft_preferences: list[dict[str, Any]]
    information_gaps: list[InformationGap]
    rejected_business_ids: list[str]
    last_presented_business_ids: list[str]
    current_business_scope: list[str]
    business_scope_known: bool
    clarification_answers: dict[str, Any]
    relative_preferences: list[RelativePreference] = Field(default_factory=list)
    semantic_summary: str | None = None
    recent_turn_summaries: list[str] = Field(default_factory=list)


class MemoryTurnInput(StrictModel):
    query_text: str = Field(min_length=1, max_length=2000)
    language: str = Field(min_length=1)
    current_turn: int = Field(ge=1)
    base_request: RecommendationRequest
    base_readiness: DecisionReadiness
    previous_memory: SessionMemory | None = None
    explicit_referenced_business_ids: list[str] = Field(default_factory=list)


class MemoryTurnResult(StrictModel):
    request: RecommendationRequest
    readiness: DecisionReadiness
    memory: SessionMemory
    proposal: MemoryProposal
    extraction: MemoryExtractionTrace
    accepted_changes: list[str] = Field(default_factory=list)
    rejected_changes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if self.memory.current_request.request_id != self.request.request_id:
            raise ValueError("memory result request must be canonical")
        if self.readiness.request_id != self.request.request_id:
            raise ValueError("memory result readiness must align")
        return self
