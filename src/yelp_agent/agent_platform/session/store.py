"""会话存储的接口；内存和 PostgreSQL 实现都要遵守它。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from .events import EventType, OpenedSession, SessionEvent, SessionRecord, SessionStatus


class SessionStore(Protocol):
    """主循环持久化会话时依赖的最小接口。"""

    async def get_or_create(
        self,
        *,
        session_id: str,
        user_id: str,
        now: datetime,
    ) -> OpenedSession: ...

    async def append_event(
        self,
        *,
        session_id: str,
        event_type: EventType,
        payload: dict[str, Any],
        now: datetime,
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> SessionEvent: ...

    async def list_events(self, session_id: str) -> list[SessionEvent]: ...

    async def set_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        now: datetime,
    ) -> SessionRecord: ...
