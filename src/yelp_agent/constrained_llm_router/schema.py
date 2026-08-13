"""Strict visible-only contracts for the Step 35 constrained Router."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from yelp_agent.agent_benchmark.schema import AgentAction
from yelp_agent.agent_evaluation.schema import ToolKind
from yelp_agent.decision_readiness.schema import TaskType
from yelp_agent.models import StrictModel


class RouterChoice(StrictModel):
    """One complete code-owned decision the model may select."""

    choice_id: str = Field(pattern=r"^choice_[0-9a-f]{12}$")
    action: AgentAction
    task_type: TaskType
    reason_code: str = Field(min_length=1, max_length=100)
    tool_name: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    tool_kind: ToolKind | None = None
    public_summary: str = Field(min_length=1, max_length=500)


class RouterStateSummary(StrictModel):
    task_type: str = Field(min_length=1)
    information_gaps: list[str] = Field(default_factory=list)
    candidate_count: int = Field(ge=0)
    ranked_candidate_count: int = Field(ge=0)
    detailed_business_count: int = Field(ge=0)
    profiled_business_count: int = Field(ge=0)
    referenced_business_count: int = Field(ge=0)
    review_evidence_count: int = Field(ge=0)
    business_scope_known: bool
    hard_constraints_required: bool
    current_turn_tools: list[str] = Field(default_factory=list)
    recent_actions: list[str] = Field(default_factory=list)
    remaining_steps: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)
    remaining_semantic_calls: int = Field(ge=0)
    remaining_rag_calls: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)


class RouterDecisionContext(StrictModel):
    """The complete sanitized input passed to the Router model."""

    schema_version: Literal[1] = 1
    scenario_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_index: int = Field(ge=1)
    language: str = Field(min_length=1)
    latest_user_query: str = Field(min_length=1, max_length=2000)
    effective_request: dict[str, Any]
    state: RouterStateSummary
    choices: list[RouterChoice] = Field(min_length=1, max_length=12)

    @field_validator("choices")
    @classmethod
    def validate_unique_choices(cls, values: list[RouterChoice]) -> list[RouterChoice]:
        ids = [item.choice_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Router choice IDs must be unique")
        return values


class RouterModelOutput(StrictModel):
    """The only JSON shape accepted from DeepSeek."""

    choice_id: str = Field(pattern=r"^choice_[0-9a-f]{12}$")
    confidence: float = Field(ge=0, le=1)
    reason_code: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_numeric_confidence(cls, value: object) -> object:
        """Accept DeepSeek's stable quoted-number variant, then validate bounds."""

        if isinstance(value, str):
            normalized = value.strip()
            try:
                return float(normalized)
            except ValueError:
                return value
        return value


class RouterExperimentSummary(StrictModel):
    schema_version: Literal[1] = 1
    decision_count: int = Field(ge=0)
    model_decision_count: int = Field(ge=0)
    single_choice_bypass_count: int = Field(ge=0)
    rule_fallback_count: int = Field(ge=0)
    task_correction_count: int = Field(ge=0)
    information_gap_correction_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    low_confidence_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    usage_unknown_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
