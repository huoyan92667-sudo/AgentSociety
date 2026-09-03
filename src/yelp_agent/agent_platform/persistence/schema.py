"""Agent 持久化层公开的数据结构。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel

from ..runtime.schema import TurnUsage

type TurnStatus = Literal[
    "running",
    "completed",
    "awaiting_user",
    "failed",
    "limit_reached",
    "interrupted",
]
type LLMCallStatus = Literal["success", "failure"]


class TurnRecord(StrictModel):
    """为了列表查询和恢复而保存的一轮对话摘要。"""

    turn_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    # 同一会话内严格递增。不能只靠时间判断先后，因为快速连续结束的两轮
    # 在部分系统上可能得到完全相同的时间值。
    turn_index: int = Field(ge=1)
    user_message: str = Field(min_length=1)
    request_time: datetime
    status: TurnStatus
    started_at: datetime
    ended_at: datetime | None = None
    answer: str | None = None
    error_code: str | None = None
    step_count: int = Field(default=0, ge=0)
    used_tools: list[str] = Field(default_factory=list)
    usage: TurnUsage = Field(default_factory=TurnUsage)


class LLMCallRecord(StrictModel):
    """一次真实模型请求的耗时和词元消耗。"""

    llm_call_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    purpose: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str | None = None
    status: LLMCallStatus
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    provider_request_id: str | None = None
    tool_call_id: str | None = None
    error_code: str | None = None
    created_at: datetime


class DomainStateVersion(StrictModel):
    """某一领域在一次更新后的完整状态快照。"""

    state_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    domain: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    version: int = Field(ge=1)
    state: dict[str, Any]
    source_event_id: str | None = None
    created_at: datetime


class ResultArtifactDraft(StrictModel):
    """工具或领域模块准备保存的一份完整结果。"""

    session_id: str = Field(min_length=1)
    turn_id: str | None = None
    kind: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_./-]{1,127}$")
    summary: dict[str, Any] = Field(default_factory=dict)
    content: Any
    expires_at: datetime | None = None


class ResultArtifact(StrictModel):
    """大结果的索引和按需加载后的内容。"""

    result_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str | None = None
    kind: str = Field(min_length=1)
    summary: dict[str, Any]
    content: Any = None
    storage_uri: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    created_at: datetime
    expires_at: datetime | None = None


class RecoveryReport(StrictModel):
    """启动恢复实际标记为中断的轮次。"""

    interrupted_turn_ids: list[str] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.interrupted_turn_ids)


class PersistenceHealth(StrictModel):
    """供启动检查使用的数据库连通性结果。"""

    ok: bool
    database_kind: str = Field(min_length=1)
    checked_at: datetime
    error_code: str | None = None


class DomainStateWrite(StrictModel):
    """保存新状态时需要的内容和可选并发版本检查。"""

    session_id: str = Field(min_length=1)
    domain: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    state: dict[str, Any]
    source_event_id: str | None = None
    expected_previous_version: int | None = Field(default=None, ge=0)

    @field_validator("state")
    @classmethod
    def require_nonempty_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("domain state cannot be empty")
        return value
