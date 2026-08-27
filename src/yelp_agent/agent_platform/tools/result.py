"""工具正文和统一执行流水线之间使用的结果结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from yelp_agent.models import StrictModel

from ..runtime.schema import TokenUsage

type ToolStatus = Literal[
    "success",
    "invalid_arguments",
    "denied",
    "needs_user_input",
    "timeout",
    "error",
]


class ToolBodyResult(StrictModel):
    """具体工具可选返回的丰富结果；简单工具也可以直接返回字典或模型。"""

    value: Any = None
    model_content: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    nested_model_usage: TokenUsage = Field(default_factory=TokenUsage)
    nested_model_calls: int = Field(default=0, ge=0)
    terminal_answer: str | None = Field(default=None, min_length=1)


class ToolResult(StrictModel):
    """流水线固化后的工具结果，生成后不允许原地修改。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: ToolStatus
    value: Any = None
    model_content: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    execution_latency_ms: float = Field(default=0.0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0)
    nested_model_usage: TokenUsage = Field(default_factory=TokenUsage)
    nested_model_calls: int = Field(default=0, ge=0)
    terminal_answer: str | None = Field(default=None, min_length=1)
