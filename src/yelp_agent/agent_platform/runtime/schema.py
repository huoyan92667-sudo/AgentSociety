"""通用 Agent 运行时在模型、循环和调用方之间传递的数据结构。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel


class TokenUsage(StrictModel):
    """一次或一轮运行中真实观测到的模型词元消耗。"""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AgentLimits(StrictModel):
    """防止模型无限循环的运行上限，不参与业务语义判断。"""

    max_steps: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=12, ge=0)
    max_total_tokens: int = Field(default=30_000, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)


class AgentTurnInput(StrictModel):
    """外部调用方启动一轮对话时必须提供的信息。"""

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    request_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("user_id", "session_id", "message")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be blank")
        return stripped

    @field_validator("request_time")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("request_time must include timezone information")
        return value


class ToolCall(StrictModel):
    """大模型要求运行时执行的一次工具调用。"""

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalAnswerAction(StrictModel):
    """模型已经可以直接结束本轮。"""

    type: Literal["final_answer"] = "final_answer"
    answer: str = Field(min_length=1)


class AskUserAction(StrictModel):
    """模型认为缺少关键事实，需要等待用户补充。"""

    type: Literal["ask_user"] = "ask_user"
    question: str = Field(min_length=1)


class ToolCallsAction(StrictModel):
    """模型选择一个或多个工具，执行后还要继续思考。"""

    type: Literal["tool_calls"] = "tool_calls"
    calls: list[ToolCall] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_call_ids(self) -> ToolCallsAction:
        call_ids = [item.call_id for item in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool call IDs must be unique within one model response")
        return self


type ModelAction = Annotated[
    FinalAnswerAction | AskUserAction | ToolCallsAction,
    Field(discriminator="type"),
]


class ToolSchema(StrictModel):
    """提供给大模型看的工具名称、用途和参数格式。"""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]


class ModelMessage(StrictModel):
    """与具体模型供应商无关的一条对话消息。"""

    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None

    @model_validator(mode="after")
    def validate_role_shape(self) -> ModelMessage:
        if self.role == "user" and not self.content:
            raise ValueError("user messages require content")
        if self.role == "assistant" and not self.content and not self.tool_calls:
            raise ValueError("assistant messages require content or tool calls")
        if self.role == "tool" and (
            not self.content or not self.tool_call_id or not self.tool_name
        ):
            raise ValueError(
                "tool messages require content, tool_call_id, and tool_name"
            )
        return self


class ContextStats(StrictModel):
    """记录本次模型上下文各部分字符量，用于验证压缩是否生效。"""

    working_memory_chars: int = Field(default=0, ge=0)
    episode_summary_chars: int = Field(default=0, ge=0)
    completed_turn_chars: int = Field(default=0, ge=0)
    current_turn_chars: int = Field(default=0, ge=0)
    included_completed_turns: int = Field(default=0, ge=0)
    included_episodes: int = Field(default=0, ge=0)


class ModelRequest(StrictModel):
    """主循环交给模型适配层的一次完整请求。"""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    system_prompt: str = Field(min_length=1)
    messages: list[ModelMessage] = Field(min_length=1)
    tools: list[ToolSchema] = Field(default_factory=list)
    source_event_seqs: list[int] = Field(default_factory=list)
    context_stats: ContextStats | None = None


class ModelResponse(StrictModel):
    """模型适配层返回的动作和供应商实际消耗。"""

    action: ModelAction
    model: str = Field(min_length=1)
    provider: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0)
    provider_request_id: str | None = None


class TurnUsage(StrictModel):
    """一轮对话结束时返回的统一消耗汇总。"""

    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model_latency_ms: float = Field(default=0.0, ge=0)
    tool_latency_ms: float = Field(default=0.0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AgentTurnResult(StrictModel):
    """调用方最终拿到的一轮结果；完整过程保存在会话事件中。"""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    status: Literal["completed", "awaiting_user", "failed", "limit_reached"]
    answer: str | None = None
    step_count: int = Field(ge=0)
    used_tools: list[str] = Field(default_factory=list)
    usage: TurnUsage
    error_code: str | None = None


class AgentStreamEvent(StrictModel):
    """调用方实时收到的回答片段，以及最后唯一一次完整结果。"""

    type: Literal["answer_delta", "final"]
    delta: str | None = None
    elapsed_ms: float = Field(default=0.0, ge=0)
    result: AgentTurnResult | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> AgentStreamEvent:
        if self.type == "answer_delta":
            if not self.delta or self.result is not None:
                raise ValueError("answer delta requires only nonempty delta text")
        elif self.delta is not None or self.result is None:
            raise ValueError("final stream event requires only the full turn result")
        return self
