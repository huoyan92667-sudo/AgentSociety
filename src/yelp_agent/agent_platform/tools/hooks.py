"""工具执行流水线的五类扩展点。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import model_validator

from yelp_agent.models import StrictModel

from .definition import ToolExecution
from .result import ToolBodyResult, ToolResult


class PreExecuteDecision(StrictModel):
    """执行前处理可以继续、拒绝，或要求用户补充信息。"""

    action: Literal["continue", "deny", "ask_user"] = "continue"
    arguments: dict[str, Any] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> PreExecuteDecision:
        if self.action != "continue" and not self.reason:
            raise ValueError("deny and ask_user decisions require a reason")
        if self.action != "continue" and self.arguments is not None:
            raise ValueError("only continue decisions may replace arguments")
        return self


class PreExecuteHook(Protocol):
    """可补充或规范参数，也可以拒绝当前调用。"""

    def __call__(
        self,
        execution: ToolExecution,
    ) -> Awaitable[PreExecuteDecision] | PreExecuteDecision: ...


class ToolGuard(Protocol):
    """最终检查只能返回拒绝原因，不能推翻之前的拒绝。"""

    def __call__(
        self,
        execution: ToolExecution,
    ) -> Awaitable[str | None] | str | None: ...


type ExecuteNext = Callable[[], Awaitable[ToolBodyResult]]


class ExecuteWrapper(Protocol):
    """包裹真正执行过程，用于缓存、追踪等通用能力。"""

    def __call__(
        self,
        execution: ToolExecution,
        call_next: ExecuteNext,
    ) -> Awaitable[ToolBodyResult] | ToolBodyResult: ...


class PostExecuteHook(Protocol):
    """结果固化前可保存大结果、压缩模型内容或增加警告。"""

    def __call__(
        self,
        execution: ToolExecution,
        result: ToolResult,
    ) -> Awaitable[ToolResult] | ToolResult: ...


class ResultObserver(Protocol):
    """只观察最终结果，用于统计和通知，不能再修改结果。"""

    def __call__(
        self,
        execution: ToolExecution,
        result: ToolResult,
    ) -> Awaitable[None] | None: ...


@dataclass(slots=True)
class ToolPipelineHooks:
    """方便启动程序一次性安装全部流水线扩展。"""

    pre_execute: list[PreExecuteHook] = field(default_factory=list)
    guards: list[ToolGuard] = field(default_factory=list)
    execute_wrappers: list[ExecuteWrapper] = field(default_factory=list)
    post_execute: list[PostExecuteHook] = field(default_factory=list)
    result_observers: list[ResultObserver] = field(default_factory=list)
