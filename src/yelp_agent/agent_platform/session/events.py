"""可持久化的会话和事件结构。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from yelp_agent.models import StrictModel

type SessionStatus = Literal["idle", "active", "awaiting_user", "failed"]
type EventType = Literal[
    "session/created",
    "turn/start",
    "turn/end",
    "step/start",
    "step/end",
    "user/message",
    "model/request",
    "assistant/message",
    "tool/call",
    "tool/result",
]


class SessionRecord(StrictModel):
    """一段长对话的基本信息，内容本身保存在事件列表中。"""

    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    status: SessionStatus
    created_at: datetime
    updated_at: datetime


class OpenedSession(StrictModel):
    """打开会话时同时说明它是否刚刚创建，避免重复写 created 事件。"""

    session: SessionRecord
    created: bool


class SessionEvent(StrictModel):
    """按顺序只追加的一条会话事实。"""

    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    seq: int = Field(ge=1)
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    turn_id: str | None = None
    step_index: int | None = Field(default=None, ge=1)
