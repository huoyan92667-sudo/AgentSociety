"""Frozen public contracts for the Step 34.5 multi-turn benchmark."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.decision_readiness import InformationGap, TaskType
from yelp_agent.models import StrictModel
from yelp_agent.query.schema import (
    ConditionField,
    ConditionOperator,
    ConditionValue,
    RequirementImportance,
)

type BenchmarkSplit = Literal["development", "validation"]
type BenchmarkLanguage = Literal["zh-CN", "en-US"]
type DeltaOperation = Literal["add", "replace", "remove"]
type RelativeField = Literal["distance", "price", "noise", "crowding"]
type RelativeDirection = Literal[
    "closer",
    "farther",
    "lower",
    "higher",
    "quieter",
    "less_crowded",
]
type BehaviorKind = Literal[
    "none",
    "cheaper",
    "closer",
    "quieter",
    "constraint_satisfaction",
]
type TurnFamily = Literal[
    "explicit_condition",
    "relative_preference",
    "clarification_answer",
    "reference_rejection",
    "conflict_resolution",
    "combined_update",
    "no_state_change",
]


class PresentedBusinessSnapshot(StrictModel):
    """Visible facts for one business that was actually presented to the user."""

    business_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    name: str = Field(min_length=1)
    categories: list[str] = Field(default_factory=list)
    price_level: int | None = Field(default=None, ge=1, le=4)
    distance_km: float | None = Field(default=None, ge=0)
    noise_level: Literal["quiet", "average", "loud", "very_loud"] | None = None
    is_chain: bool | None = None

    @field_validator("categories")
    @classmethod
    def unique_categories(cls, values: list[str]) -> list[str]:
        normalized = [item.strip() for item in values if item.strip()]
        return list(dict.fromkeys(normalized))


class ExpectedConditionDelta(StrictModel):
    operation: DeltaOperation
    field: ConditionField
    operator: ConditionOperator | None = None
    value: ConditionValue | None = None
    importance: RequirementImportance | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        payload = (self.operator, self.value, self.importance)
        if self.operation == "remove" and any(item is not None for item in payload):
            raise ValueError("remove deltas identify only the field")
        if self.operation != "remove" and any(item is None for item in payload):
            raise ValueError("add and replace deltas require a complete condition")
        return self


class ExpectedRelativePreference(StrictModel):
    field: RelativeField
    direction: RelativeDirection


class ExpectedMemoryDeltaV2(StrictModel):
    """Every observable state change expected from one released user message."""

    task_type: TaskType
    condition_deltas: list[ExpectedConditionDelta] = Field(default_factory=list)
    relative_preferences: list[ExpectedRelativePreference] = Field(
        default_factory=list
    )
    clarification_answers: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )
    rejected_business_ids: list[str] = Field(default_factory=list)
    resolved_business_ids: list[str] = Field(default_factory=list)
    conflict_resolutions: list[str] = Field(default_factory=list)
    expected_information_gaps: list[InformationGap] = Field(default_factory=list)
    party_size: int | None = Field(default=None, ge=1, le=100)
    no_state_change: bool = False

    @model_validator(mode="after")
    def validate_noop(self) -> Self:
        changed = any(
            (
                self.condition_deltas,
                self.relative_preferences,
                self.clarification_answers,
                self.rejected_business_ids,
                self.resolved_business_ids,
                self.conflict_resolutions,
                self.party_size is not None,
            )
        )
        if self.no_state_change and changed:
            raise ValueError("a no-op turn cannot carry an expected memory change")
        return self


class BehaviorExpectation(StrictModel):
    kind: BehaviorKind = "none"
    baseline_business_id: str | None = None
    baseline_value: float | int | None = None
    acceptable_business_ids: list[str] = Field(default_factory=list)
    excluded_business_ids: list[str] = Field(default_factory=list)
    required_category: str | None = None
    maximum_value: float | int | None = None

    @model_validator(mode="after")
    def validate_behavior(self) -> Self:
        if self.kind in {"cheaper", "closer", "quieter"}:
            if self.baseline_business_id is None or self.baseline_value is None:
                raise ValueError("relative behavior requires a visible baseline")
        if self.kind == "none" and any(
            value is not None
            for value in (
                self.baseline_business_id,
                self.baseline_value,
                self.required_category,
                self.maximum_value,
            )
        ):
            raise ValueError("no-op behavior cannot carry a threshold")
        if set(self.acceptable_business_ids).intersection(self.excluded_business_ids):
            raise ValueError("acceptable and excluded businesses cannot overlap")
        return self


class MemoryBenchmarkInitialSession(StrictModel):
    schema_version: Literal[2] = 2
    session_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: BenchmarkSplit
    language: BenchmarkLanguage
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cutoff_time: datetime
    query_text: str = Field(min_length=1, max_length=2000)
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)
    referenced_business_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("session coordinates must appear together")
        return self


class FrozenPresentation(StrictModel):
    schema_version: Literal[2] = 2
    session_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_index: int = Field(ge=1)
    pipeline_version: str = Field(min_length=1)
    source_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_business_ids: list[str]
    presented_businesses: list[PresentedBusinessSnapshot] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        presented = [item.business_id for item in self.presented_businesses]
        if len(presented) != len(set(presented)):
            raise ValueError("presented businesses must be unique")
        if not set(presented).issubset(self.candidate_business_ids):
            raise ValueError("presented businesses must be in the frozen candidate scope")
        return self


class TurnGenerationSpec(StrictModel):
    """Code-owned meaning that the provider may phrase but cannot relabel."""

    turn_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: BenchmarkSplit
    turn_index: int = Field(ge=2)
    language: BenchmarkLanguage
    family: TurnFamily
    intent_code: str = Field(min_length=1, max_length=100)
    visible_context: dict[str, Any] = Field(default_factory=dict)
    required_meaning: dict[str, Any] = Field(default_factory=dict)
    forbidden_meaning: list[str] = Field(default_factory=list)
    expected_delta: ExpectedMemoryDeltaV2
    behaviors: list[BehaviorExpectation] = Field(default_factory=list)
    trigger_action: str = Field(min_length=1)
    state_updates: dict[str, str | int | float | bool] = Field(default_factory=dict)


class GeneratedUserTurn(StrictModel):
    turn_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_text: str = Field(min_length=3, max_length=1000)

    @field_validator("query_text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())


class GeneratedTurnBatch(StrictModel):
    turns: list[GeneratedUserTurn] = Field(min_length=1, max_length=20)


class SemanticReviewDecision(StrictModel):
    turn_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent_consistent: bool
    extra_constraint_detected: bool
    hidden_value_leaked: bool
    reference_answerable: bool
    naturalness_score: int = Field(ge=1, le=5)
    issue_codes: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return (
            self.intent_consistent
            and not self.extra_constraint_detected
            and not self.hidden_value_leaked
            and self.reference_answerable
            and self.naturalness_score >= 3
        )


class SemanticReviewBatch(StrictModel):
    decisions: list[SemanticReviewDecision] = Field(min_length=1, max_length=20)


class FrozenScriptedTurnV2(StrictModel):
    schema_version: Literal[2] = 2
    turn_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: BenchmarkSplit
    turn_index: int = Field(ge=2)
    language: BenchmarkLanguage
    family: TurnFamily
    intent_code: str
    query_text: str = Field(min_length=3, max_length=1000)
    trigger_action: str
    state_updates: dict[str, str | int | float | bool] = Field(default_factory=dict)
    generator_model: str
    generator_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_model: str
    reviewer_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenTurnGroundTruthV2(StrictModel):
    schema_version: Literal[2] = 2
    turn_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: BenchmarkSplit
    turn_index: int = Field(ge=2)
    family: TurnFamily
    expected_delta: ExpectedMemoryDeltaV2
    behaviors: list[BehaviorExpectation] = Field(default_factory=list)


class PlanningContext(StrictModel):
    """One source scenario plus the frozen result visible after its first turn."""

    initial_session: MemoryBenchmarkInitialSession
    source_category: str = Field(min_length=1)
    source_frame_family: str = Field(min_length=1)
    initial_task_type: TaskType
    initial_information_gaps: list[InformationGap] = Field(default_factory=list)
    initial_conditions: list[ExpectedConditionDelta] = Field(default_factory=list)
    presentation: FrozenPresentation | None = None
    candidate_businesses: list[PresentedBusinessSnapshot] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_presentation(self) -> Self:
        if self.presentation is not None:
            if self.presentation.session_case_id != self.initial_session.session_case_id:
                raise ValueError("planning context and presentation must align")
            candidate_ids = {item.business_id for item in self.candidate_businesses}
            if not set(self.presentation.candidate_business_ids).issubset(candidate_ids):
                raise ValueError("planning context must contain every frozen candidate fact")
        return self


class BenchmarkGenerationPlan(StrictModel):
    initial_sessions: list[MemoryBenchmarkInitialSession]
    presentations: list[FrozenPresentation]
    turn_specs: list[TurnGenerationSpec]

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        session_ids = [item.session_case_id for item in self.initial_sessions]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("planned sessions must be unique")
        if not {item.session_case_id for item in self.turn_specs}.issubset(session_ids):
            raise ValueError("every turn must belong to a planned session")
        turn_ids = [item.turn_case_id for item in self.turn_specs]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("planned turns must be unique")
        return self


class BenchmarkV2Manifest(StrictModel):
    schema_version: Literal[2] = 2
    benchmark_version: str
    session_count: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    split_counts: dict[str, int]
    language_counts: dict[str, int]
    family_counts: dict[str, int]
    generator_model: str
    reviewer_model: str
    source_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: dict[str, str]
    generation_input_tokens: int = Field(ge=0)
    generation_output_tokens: int = Field(ge=0)
    review_input_tokens: int = Field(ge=0)
    review_output_tokens: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    rejected_generation_count: int = Field(ge=0)
