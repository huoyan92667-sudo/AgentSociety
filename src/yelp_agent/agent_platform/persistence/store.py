"""PostgreSQL 和内存实现共同遵守的持久化接口。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..memory.store import ConversationMemoryStore
from ..runtime.schema import TurnUsage
from ..session.events import SessionEvent, SessionStatus
from ..session.store import SessionStore
from .schema import (
    DomainStateVersion,
    DomainStateWrite,
    LLMCallRecord,
    PersistenceHealth,
    RecoveryReport,
    ResultArtifact,
    ResultArtifactDraft,
    TurnRecord,
    TurnStatus,
)


class RuntimePersistence(SessionStore, ConversationMemoryStore, Protocol):
    """主循环除了事件追加之外需要的持久化操作。"""

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        request_time: datetime,
        now: datetime,
    ) -> TurnRecord: ...

    async def finish_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        turn_status: TurnStatus,
        session_status: SessionStatus,
        answer: str,
        error_code: str | None,
        step_count: int,
        used_tools: list[str],
        usage: TurnUsage,
        now: datetime,
    ) -> TurnRecord: ...

    async def record_llm_call(self, record: LLMCallRecord) -> None: ...

    async def record_model_response(
        self,
        *,
        record: LLMCallRecord,
        event_payload: dict,
        now: datetime,
    ) -> SessionEvent: ...

    async def recover_interrupted_turns(
        self,
        *,
        now: datetime,
    ) -> RecoveryReport: ...

    async def healthcheck(self) -> PersistenceHealth: ...

    async def list_recent_turns(
        self,
        *,
        session_id: str,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[TurnRecord]: ...

    async def list_recent_turns_after(
        self,
        *,
        session_id: str,
        after_turn_index: int | None,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[TurnRecord]: ...

    async def list_completed_turns_after(
        self,
        *,
        session_id: str,
        after_turn_index: int | None,
        limit: int,
    ) -> list[TurnRecord]: ...

    async def list_turn_events(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> list[SessionEvent]: ...


class DomainStateStore(Protocol):
    """餐饮、旅游等领域保存和读取版本化状态的接口。"""

    async def save_domain_state(
        self,
        value: DomainStateWrite,
        *,
        now: datetime,
    ) -> DomainStateVersion: ...

    async def get_latest_domain_state(
        self,
        *,
        session_id: str,
        domain: str,
    ) -> DomainStateVersion | None: ...


class ResultStore(Protocol):
    """保存大结果并通过编号按需读取的接口。"""

    async def save_result(
        self,
        value: ResultArtifactDraft,
        *,
        now: datetime,
    ) -> ResultArtifact: ...

    async def get_result(
        self,
        *,
        result_id: str,
        user_id: str,
    ) -> ResultArtifact | None: ...


class AgentPersistence(RuntimePersistence, DomainStateStore, ResultStore, Protocol):
    """完整 Agent 框架所需的持久化能力集合。"""
