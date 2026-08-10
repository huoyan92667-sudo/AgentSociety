"""Stable contracts shared by every Step 23 Agent tool."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from yelp_agent.agent_benchmark.schema import AgentAction
from yelp_agent.agent_evaluation.schema import EvidenceReference, ToolKind
from yelp_agent.models import StrictModel


type ToolStatus = Literal[
    "success",
    "no_result",
    "partial",
    "retryable_error",
    "permanent_error",
    "unavailable",
]


class ToolExecutionContext(StrictModel):
    """Visible runtime context; hidden benchmark labels cannot enter tools."""

    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_id: str = Field(min_length=1)
    cutoff_time: datetime
    action: AgentAction
    business_scope: tuple[str, ...] = ()
    business_scope_known: bool = False
    state_snapshot: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(StrictModel):
    """One normalized tool result suitable for state and trace persistence."""

    tool_name: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    cost_usd: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    cache_hit: bool = False
    attempt_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        is_error = self.status in {
            "retryable_error",
            "permanent_error",
            "unavailable",
        }
        if is_error != (self.error_code is not None):
            raise ValueError("only error and unavailable observations need error_code")
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input and output token counts must appear together")
        return self

    @classmethod
    def success(
        cls,
        *,
        tool_name: str,
        data: dict[str, Any],
        confidence: float | None = None,
        warnings: list[str] | None = None,
    ) -> Self:
        return cls(
            tool_name=tool_name,
            status="success",
            data=data,
            confidence=confidence,
            warnings=warnings or [],
        )

    @classmethod
    def error(
        cls,
        *,
        tool_name: str,
        status: Literal["retryable_error", "permanent_error", "unavailable"],
        error_code: str,
        warning: str,
    ) -> Self:
        return cls(
            tool_name=tool_name,
            status=status,
            error_code=error_code,
            warnings=[warning],
        )


class ToolAvailability(StrictModel):
    available: bool
    reason: str | None = None


class ToolDescriptor(StrictModel):
    """Serializable public description returned to Routers and documentation."""

    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    version: str = Field(min_length=1)
    kind: ToolKind
    allowed_actions: list[AgentAction]
    public_summary: str = Field(min_length=1)
    availability: ToolAvailability
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    preconditions: list[str] = Field(default_factory=list)
    resolves_information_gaps: list[str] = Field(default_factory=list)
    resolves_uncertainties: list[str] = Field(default_factory=list)
    cache_scope: Literal["none", "request"]
    empty_result_behavior: Literal["observe_no_result", "error"]
    max_attempts: int = Field(ge=1)
    timeout_ms: float = Field(gt=0)
    estimated_cost_usd: float = Field(ge=0)
    fallback_policy: str = Field(min_length=1)
