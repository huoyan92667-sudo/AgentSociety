"""Public runtime contracts for the controlled Agent harness."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.agent_benchmark.schema import AgentAction
from yelp_agent.agent_evaluation.schema import (
    AgentScenarioRun,
    AgentTurnTrace,
    ClarificationQuestionTrace,
    RetrievedEvidenceTrace,
    ResponseClaimTrace,
    ToolKind,
)
from yelp_agent.decision_readiness import DecisionReadiness
from yelp_agent.models import StrictModel
from yelp_agent.query import RecommendationRequest


type HarnessStatus = Literal["running", "awaiting_user", "completed", "fallback"]
type OutcomeStatus = Literal["completed", "failed"]


class HarnessBudget(StrictModel):
    """Hard execution limits enforced by code rather than by a Router."""

    max_steps: int = Field(default=10, ge=1)
    max_tool_calls: int = Field(default=5, ge=0)
    max_semantic_calls: int = Field(default=1, ge=0)
    max_rag_calls: int = Field(default=2, ge=0)
    max_total_tokens: int = Field(default=12_000, ge=0)
    timeout_ms: float = Field(default=90_000, gt=0)


class UserTurnInput(StrictModel):
    """One visible user message used to start or resume a session."""

    query_text: str = Field(min_length=1, max_length=2000)
    user_latitude: float | None = Field(default=None, ge=-90, le=90)
    user_longitude: float | None = Field(default=None, ge=-180, le=180)
    referenced_business_ids: list[str] = Field(default_factory=list)

    @field_validator("query_text")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query text cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_visible_context(self) -> UserTurnInput:
        if (self.user_latitude is None) != (self.user_longitude is None):
            raise ValueError("turn coordinates must be both present or absent")
        if len(self.referenced_business_ids) != len(
            set(self.referenced_business_ids)
        ):
            raise ValueError("referenced business IDs must be unique")
        return self


class TurnInterpretation(StrictModel):
    """The Step 18 request plus Step 19 readiness snapshot for one turn."""

    request: RecommendationRequest
    readiness: DecisionReadiness
    semantic_calls: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> TurnInterpretation:
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("interpretation token counts must appear together")
        if self.semantic_calls == 0 and any(
            value is not None for value in (self.input_tokens, self.cost_usd)
        ):
            raise ValueError("rule-only interpretation cannot report model usage")
        return self


class AgentDecision(StrictModel):
    """Exactly one action selected by a Router."""

    action: AgentAction
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason_code: str = Field(min_length=1, max_length=100)
    tool_name: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    tool_kind: ToolKind | None = None

    @model_validator(mode="after")
    def validate_tool_identity(self) -> AgentDecision:
        if (self.tool_name is None) != (self.tool_kind is None):
            raise ValueError("tool_name and tool_kind must appear together")
        return self


class AgentObservation(StrictModel):
    """A durable, visible-only result produced by one executed action."""

    observation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_index: int = Field(ge=1)
    step_index: int = Field(ge=1)
    action: AgentAction
    payload: dict[str, Any]


class ToolResultMetadata(StrictModel):
    """Observable accounting and evidence returned with a tool result."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    cache_hit: bool = False
    retrieved_evidence: list[RetrievedEvidenceTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_token_pair(self) -> ToolResultMetadata:
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input and output token counts must appear together")
        return self


class ActionOutcome(StrictModel):
    """Structured executor result; raw provider objects never enter state."""

    status: OutcomeStatus
    observation: dict[str, Any] | None = Field(default=None, min_length=1)
    response_kind: Literal[
        "none",
        "clarification",
        "recommendation",
        "grounded_answer",
        "uncertain_answer",
        "fallback",
    ] = "none"
    candidate_ranking: list[str] = Field(default_factory=list)
    recommended_business_ids: list[str] = Field(default_factory=list)
    clarification_question: ClarificationQuestionTrace | None = None
    claims: list[ResponseClaimTrace] = Field(default_factory=list)
    reported_conflict: bool = False
    reported_evidence_recency: bool = False
    recommended_official_verification: bool = False
    tool_result: ToolResultMetadata | None = None
    business_scope: list[str] | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> ActionOutcome:
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed outcomes require failure_reason")
        if self.status == "completed" and self.failure_reason is not None:
            raise ValueError("completed outcomes cannot include failure_reason")
        if len(self.candidate_ranking) != len(set(self.candidate_ranking)):
            raise ValueError("candidate ranking must contain unique IDs")
        if self.business_scope is not None and len(self.business_scope) != len(
            set(self.business_scope)
        ):
            raise ValueError("business scope must contain unique IDs")
        if not set(self.recommended_business_ids).issubset(self.candidate_ranking):
            raise ValueError("recommendations must be inside the candidate ranking")
        has_question = self.clarification_question is not None
        if has_question != (self.response_kind == "clarification"):
            raise ValueError("clarification response and question must appear together")
        if self.response_kind == "recommendation" and (
            not self.candidate_ranking or not self.recommended_business_ids
        ):
            raise ValueError("recommendation responses require ranking and result IDs")
        return self


class AgentSession(StrictModel):
    """Serializable visible state that can be paused and resumed."""

    schema_version: Literal[1] = 1
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = Field(min_length=1)
    language: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cutoff_time: datetime
    current_turn: int = Field(default=1, ge=1)
    status: HarnessStatus = "running"
    request: RecommendationRequest
    readiness: DecisionReadiness
    available_actions: list[AgentAction] = Field(default_factory=list)
    observations: list[AgentObservation] = Field(default_factory=list)
    action_history: list[AgentDecision] = Field(default_factory=list)
    executed_call_signatures: list[str] = Field(default_factory=list)
    business_scope: list[str] = Field(default_factory=list)
    business_scope_known: bool = False
    turns: list[AgentTurnTrace] = Field(default_factory=list)
    budget: HarnessBudget
    step_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    semantic_call_count: int = Field(default=0, ge=0)
    rag_call_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    # Session totals above are reported to evaluation. These two counters are
    # reset on every user turn and are the only values used for per-turn budget.
    turn_input_tokens: int = Field(default=0, ge=0)
    turn_output_tokens: int = Field(default=0, ge=0)
    token_usage_observed: bool = False
    cost_usd: float = Field(default=0, ge=0)
    started_at_ms: float = Field(ge=0)
    elapsed_ms: float = Field(default=0, ge=0)
    fallback_reason: str | None = None

    @field_validator("executed_call_signatures")
    @classmethod
    def validate_unique_call_signatures(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("executed call signatures must be unique")
        if any(len(value) != 64 for value in values):
            raise ValueError("executed call signatures must be SHA256 hex digests")
        return values

    def to_json(self) -> str:
        """Serialize durable state without re-emitting request computed fields."""

        return self.model_dump_json(exclude_computed_fields=True)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.model_validate_json(payload)


class HarnessResult(StrictModel):
    """A resumable session plus a final Step 21 run when execution terminates."""

    session: AgentSession
    run: AgentScenarioRun | None = None


# AgentState is the conceptual name used by the architecture. AgentSession is
# retained as the persistence-oriented public name used by start()/resume().
AgentState = AgentSession
