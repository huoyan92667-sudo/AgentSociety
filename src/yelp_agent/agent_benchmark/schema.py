"""Stable visible and hidden contracts for Step 20 Agent scenarios."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pyarrow as pa
from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.query.benchmark import ExpectedRequestCondition


type ScenarioSplit = Literal["development", "validation"]
type ScenarioLanguage = Literal["zh-CN", "en-US"]
type ScenarioCategory = Literal[
    "hard_constraint",
    "profile_conflict",
    "information_gap",
    "business_detail",
    "candidate_comparison",
    "multi_turn_feedback",
    "evidence_uncertainty",
]
type ScenarioTaskType = Literal[
    "recommendation_request",
    "business_detail_question",
    "candidate_comparison",
    "feedback_refinement",
    "official_policy_question",
    "review_experience_question",
]
type InformationGap = Literal[
    "missing_location",
    "missing_budget",
    "missing_party_size",
    "constraint_conflict",
    "ambiguous_reference",
]
type AgentAction = Literal[
    "ask_clarification",
    "retrieve_candidates",
    "apply_hard_constraints",
    "rank_candidates",
    "get_business_details",
    "retrieve_business_reviews",
    "compare_candidates",
    "apply_feedback",
    "check_official_source",
    "return_recommendation",
    "return_grounded_answer",
    "return_uncertain_answer",
    "safe_fallback",
]
type UncertaintyPolicy = Literal[
    "proceed",
    "clarify_before_action",
    "answer_with_caveat",
    "report_conflict",
    "require_official_verification",
    "abstain",
]
type EvidenceSourceType = Literal["review", "business_attribute"]
type EvidenceRelevance = Literal["relevant", "irrelevant", "out_of_scope"]
type EvidenceStance = Literal["supports", "contradicts", "neutral"]
type GeneratorKind = Literal["deterministic", "fake", "openai_compatible"]


class VisibleAgentScenario(StrictModel):
    """The complete scenario input an Agent runner is allowed to receive."""

    schema_version: Literal[1] = 1
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: ScenarioSplit
    language: ScenarioLanguage
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cutoff_time: datetime
    query_text: str = Field(min_length=1, max_length=2000)
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)
    referenced_business_ids: list[str] = Field(default_factory=list)
    generator_kind: GeneratorKind = "deterministic"
    generator_model: str | None = None
    generator_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("query_text")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query text cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_visible_state(self) -> VisibleAgentScenario:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("scenario coordinates must be both present or absent")
        if len(self.referenced_business_ids) != len(
            set(self.referenced_business_ids)
        ):
            raise ValueError("referenced business IDs must be unique")
        generated = self.generator_kind == "openai_compatible"
        metadata = self.generator_model is not None
        if metadata != (self.generator_prompt_sha256 is not None):
            raise ValueError("generator model and prompt hash must appear together")
        if generated != metadata:
            raise ValueError("only provider-generated scenarios carry model metadata")
        return self


class ScriptedUserTurn(StrictModel):
    """A hidden user reply released only after the matching Agent action."""

    turn_index: int = Field(ge=2)
    trigger_action: AgentAction
    query_text: str = Field(min_length=1, max_length=2000)
    expected_task_type: ScenarioTaskType
    expected_information_gaps: list[InformationGap] = Field(default_factory=list)
    added_conditions: list[ExpectedRequestCondition] = Field(default_factory=list)
    state_updates: dict[str, str | int | float | bool] = Field(default_factory=dict)
    rejected_business_ids: list[str] = Field(default_factory=list)
    expected_allowed_actions: list[AgentAction]

    @model_validator(mode="after")
    def validate_turn(self) -> ScriptedUserTurn:
        if len(self.expected_information_gaps) != len(
            set(self.expected_information_gaps)
        ):
            raise ValueError("scripted-turn gaps must be unique")
        if len(self.rejected_business_ids) != len(set(self.rejected_business_ids)):
            raise ValueError("rejected business IDs must be unique")
        if not self.expected_allowed_actions:
            raise ValueError("scripted turns require at least one allowed action")
        return self


class ScenarioGroundTruth(StrictModel):
    """Hidden evaluation state; never accepted by an Agent runner."""

    schema_version: Literal[1] = 1
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_category: ScenarioCategory
    frame_family: str = Field(min_length=1)
    source_query_case_id: str | None = None
    source_task_id: str = Field(min_length=1)
    source_profile_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_type: ScenarioTaskType
    expected_conditions: list[ExpectedRequestCondition]
    expected_information_gaps: list[InformationGap]
    allowed_actions: list[AgentAction]
    required_actions: list[AgentAction]
    forbidden_actions: list[AgentAction]
    business_scope: list[str]
    acceptable_business_ids: list[str]
    scripted_user_turns: list[ScriptedUserTurn]
    uncertainty_policy: UncertaintyPolicy
    current_request_overrides_profile: bool

    @model_validator(mode="after")
    def validate_ground_truth(self) -> ScenarioGroundTruth:
        allowed = set(self.allowed_actions)
        required = set(self.required_actions)
        forbidden = set(self.forbidden_actions)
        if len(allowed) != len(self.allowed_actions):
            raise ValueError("allowed actions must be unique")
        if not required.issubset(allowed):
            raise ValueError("required actions must be allowed")
        if forbidden.intersection(allowed):
            raise ValueError("forbidden actions cannot also be allowed")
        if len(self.business_scope) != len(set(self.business_scope)):
            raise ValueError("business scope must be unique")
        if not set(self.acceptable_business_ids).issubset(self.business_scope):
            raise ValueError("acceptable businesses must be inside business scope")
        if len(self.expected_information_gaps) != len(
            set(self.expected_information_gaps)
        ):
            raise ValueError("expected information gaps must be unique")
        turn_indices = [turn.turn_index for turn in self.scripted_user_turns]
        if turn_indices != list(range(2, len(turn_indices) + 2)):
            raise ValueError("scripted user turns must be contiguous from turn two")
        if self.current_request_overrides_profile != (
            self.scenario_category == "profile_conflict"
        ):
            raise ValueError("profile precedence flag must match profile-conflict scenes")
        return self


class EvidenceLabel(StrictModel):
    """One business-scoped static or review evidence judgment."""

    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_id: str = Field(min_length=1)
    source_type: EvidenceSourceType
    review_id: str | None = None
    source_field: str | None = None
    aspect: str | None = None
    relevance: EvidenceRelevance
    stance: EvidenceStance
    event_time: datetime | None = None
    confidence: float = Field(ge=0, le=1)
    source_text_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_source(self) -> EvidenceLabel:
        if self.source_type == "review":
            if (
                self.review_id is None
                or self.event_time is None
                or self.source_text_sha256 is None
                or self.source_field is not None
            ):
                raise ValueError("review evidence requires review metadata only")
        elif self.source_field is None or any(
            value is not None
            for value in (self.review_id, self.event_time, self.source_text_sha256)
        ):
            raise ValueError("attribute evidence requires only source_field")
        return self


EVIDENCE_LABEL_SCHEMA = pa.schema(
    [
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("source_type", pa.string(), nullable=False),
        pa.field("review_id", pa.string()),
        pa.field("source_field", pa.string()),
        pa.field("aspect", pa.string()),
        pa.field("relevance", pa.string(), nullable=False),
        pa.field("stance", pa.string(), nullable=False),
        pa.field("event_time", pa.timestamp("us")),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("source_text_sha256", pa.string()),
    ]
)
