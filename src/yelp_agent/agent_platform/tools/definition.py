"""一个工具的稳定定义以及一次执行时的服务器上下文。"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from yelp_agent.models import StrictModel

from ..runtime.schema import ToolCall, ToolSchema
from .result import ToolBodyResult


class ToolExecutionContext(StrictModel):
    """由服务器注入的可信信息，大模型不能自行填写或覆盖。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    request_time: datetime
    current_user_message: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


type ToolHandler = Callable[
    [BaseModel, ToolExecutionContext],
    Awaitable[ToolBodyResult | BaseModel | Any] | ToolBodyResult | BaseModel | Any,
]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """工具作者需要声明的全部信息；日志、计时和异常由流水线负责。"""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    output_model: type[BaseModel] | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 0
    read_only: bool = True
    concurrency_safe: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.name):
            raise ValueError(
                "tool name must contain 2-64 lowercase letters, numbers, or underscores"
            )
        if not self.description.strip():
            raise ValueError("tool description cannot be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("tool timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("tool max_retries cannot be negative")
        if self.max_retries and not self.read_only:
            raise ValueError("only read-only tools may be retried automatically")

    def model_schema(self) -> ToolSchema:
        """生成只提供给大模型看的安全工具说明。"""

        return ToolSchema(
            name=self.name,
            description=self.description.strip(),
            input_schema=self.input_model.model_json_schema(),
        )


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """参数验证成功后形成的流水线内部执行对象。"""

    call: ToolCall
    definition: ToolDefinition
    arguments: BaseModel
    context: ToolExecutionContext
