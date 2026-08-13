"""Versioned run and report contracts for Agent Scenario evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.agent_benchmark.schema import (
    AgentAction,
    EvidenceSourceType,
    InformationGap,
)
from yelp_agent.decision_readiness.schema import TaskType
from yelp_agent.models import StrictModel


type ActionStatus = Literal["completed", "failed", "rejected"]
type ToolKind = Literal[
    "deterministic",
    "semantic",
    "review_rag",
    "official_external",
]
type ResponseKind = Literal[
    "none",
    "clarification",
    "recommendation",
    "grounded_answer",
    "uncertain_answer",
    "fallback",
]
type MetricStatus = Literal["measured", "not_applicable", "unavailable"]


class AgentActionTrace(StrictModel):
    """One high-level action selected by an Agent on a user turn."""

    step_index: int = Field(ge=1)
    action: AgentAction
    status: ActionStatus
    reason_code: str = Field(min_length=1, max_length=100)


class ClarificationQuestionTrace(StrictModel):
    """One user-facing question and the information gaps it is meant to resolve."""

    question_text: str = Field(min_length=1, max_length=1000)
    requested_information_gaps: list[InformationGap] = Field(min_length=1)

    @field_validator("question_text")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("requested_information_gaps")
    @classmethod
    def validate_unique_gaps(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("a clarification question cannot repeat a gap")
        return values


class ToolCallTrace(StrictModel):
    """One observable call emitted by the future Tool Registry executor."""

    call_id: str = Field(min_length=1, max_length=200)
    action_step_index: int = Field(ge=1)
    action: AgentAction
    tool_name: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    tool_kind: ToolKind
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ActionStatus
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    cache_hit: bool = False
    retrieved_evidence: list[RetrievedEvidenceTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_retrieval_order(self) -> ToolCallTrace:
        ranks = [item.rank for item in self.retrieved_evidence]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("retrieved evidence ranks must be contiguous from one")
        return self


class EvidenceReference(StrictModel):
    """A public reference to one review or one structured business field."""

    business_id: str = Field(min_length=1)
    source_type: EvidenceSourceType
    review_id: str | None = None
    source_field: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> EvidenceReference:
        if self.source_type == "review":
            if self.review_id is None or self.source_field is not None:
                raise ValueError("review references require only review_id")
        elif self.source_field is None or self.review_id is not None:
            raise ValueError("attribute references require only source_field")
        return self


class RetrievedEvidenceTrace(StrictModel):
    rank: int = Field(ge=1)
    evidence: EvidenceReference
    relevance_score: float | None = Field(default=None, ge=0, le=1)


class ResponseClaimTrace(StrictModel):
    """One user-visible factual claim and its explicit citations."""

    claim_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    business_id: str | None = None
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)


class AgentTurnTrace(StrictModel):
    """Observable output for one user turn; never contains hidden labels."""

    turn_index: int = Field(ge=1)
    # Runtime parsing can honestly return ``unknown``. Hidden benchmark labels
    # remain restricted to the six supported task types.
    predicted_task_type: TaskType
    detected_information_gaps: list[InformationGap] = Field(default_factory=list)
    actions: list[AgentActionTrace] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    clarification_questions: list[ClarificationQuestionTrace] = Field(
        default_factory=list
    )
    candidate_ranking: list[str] = Field(default_factory=list)
    recommended_business_ids: list[str] = Field(default_factory=list)
    claims: list[ResponseClaimTrace] = Field(default_factory=list)
    response_kind: ResponseKind = "none"
    reported_conflict: bool = False
    reported_evidence_recency: bool = False
    recommended_official_verification: bool = False
    effective_request: dict[str, Any] | None = None

    @field_validator(
        "detected_information_gaps",
        "candidate_ranking",
        "recommended_business_ids",
    )
    @classmethod
    def validate_unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Agent turn lists must contain unique values")
        if any(not value or value != value.strip() for value in values):
            raise ValueError("Agent turn list values must be nonempty and trimmed")
        return values

    @model_validator(mode="after")
    def validate_action_order(self) -> AgentTurnTrace:
        indices = [item.step_index for item in self.actions]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("action steps must be contiguous from one")
        action_by_step = {item.step_index: item.action for item in self.actions}
        call_ids = [item.call_id for item in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique within a turn")
        if any(
            action_by_step.get(item.action_step_index) != item.action
            for item in self.tool_calls
        ):
            raise ValueError("tool calls must reference a matching action step")
        if self.response_kind == "clarification" and not self.clarification_questions:
            raise ValueError("clarification responses must include a question")
        if self.response_kind != "clarification" and self.clarification_questions:
            raise ValueError("questions require a clarification response")
        if not set(self.recommended_business_ids).issubset(self.candidate_ranking):
            raise ValueError(
                "recommended_business_ids must be a subset of candidate_ranking"
            )
        claim_ids = [item.claim_id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique within a turn")
        return self


class AgentScenarioRun(StrictModel):
    """One complete Agent attempt on a visible benchmark scenario."""

    schema_version: Literal[1] = 1
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_version: str = Field(min_length=1)
    turns: list[AgentTurnTrace] = Field(min_length=1)
    fallback: bool = False
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_run(self) -> AgentScenarioRun:
        indices = [item.turn_index for item in self.turns]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("turns must be contiguous from one")
        if self.fallback != (self.fallback_reason is not None):
            raise ValueError("fallback and fallback_reason must appear together")
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input and output tokens must appear together")
        call_ids = [call.call_id for turn in self.turns for call in turn.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique within a run")
        return self


class MetricScore(StrictModel):
    """One frozen metric with an explicit denominator and availability state."""

    status: MetricStatus
    value: float | None = None
    numerator: float
    denominator: float = Field(ge=0)
    unit: Literal["ratio", "milliseconds", "tokens", "usd"]
    reason: str | None = None


class ScenarioEvaluationResult(StrictModel):
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str
    category: str
    metrics: dict[str, float | None]
    violations: list[str] = Field(default_factory=list)


class AgentEvaluationSlice(StrictModel):
    """A split/category view using the exact same metric definitions as overall."""

    scenario_count: int = Field(ge=1)
    metrics: dict[str, MetricScore]


class AgentEvaluationReport(StrictModel):
    schema_version: Literal[1] = 1
    contract_version: str = "1.0.0"
    scenario_count: int = Field(ge=1)
    split_counts: dict[str, int]
    metrics: dict[str, MetricScore]
    by_split: dict[str, AgentEvaluationSlice] = Field(default_factory=dict)
    by_category: dict[str, AgentEvaluationSlice] = Field(default_factory=dict)
    scenario_results: list[ScenarioEvaluationResult]
