"""以 PostgreSQL 为正式后端的 Agent 持久化实现。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..memory.models import (
    ConversationEpisode,
    ConversationEpisodeDraft,
    ToolMemoryUpdate,
    WorkingMemory,
)
from ..memory.operations import (
    apply_tool_update,
    rank_episode_matches,
    with_pending_question,
)
from ..results.content import ArtifactContentStore
from ..runtime.schema import TurnUsage
from ..session.events import (
    EventType,
    OpenedSession,
    SessionEvent,
    SessionRecord,
    SessionStatus,
)
from .errors import PersistenceError, SessionBusyError, StateVersionConflictError
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
from .tables import (
    AgentEventRow,
    AgentSessionRow,
    AgentTurnRow,
    ConversationEpisodeRow,
    DomainStateVersionRow,
    LLMCallRow,
    ResultArtifactRow,
    WorkingMemoryRow,
)


class PostgresAgentPersistence:
    """在事务中维护会话事件、轮次、领域状态、结果和模型调用记录。"""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        content_store: ArtifactContentStore | None = None,
        inline_result_max_bytes: int = 256 * 1024,
    ) -> None:
        if inline_result_max_bytes < 1024:
            raise ValueError("inline_result_max_bytes must be at least 1024")
        self._sessions = sessions
        self._content_store = content_store
        self._inline_result_max_bytes = inline_result_max_bytes

    async def get_or_create(
        self,
        *,
        session_id: str,
        user_id: str,
        now: datetime,
    ) -> OpenedSession:
        async with self._sessions() as database:
            existing = await database.get(AgentSessionRow, session_id)
            if existing is not None:
                self._require_user(existing, user_id)
                return OpenedSession(
                    session=self._session_record(existing),
                    created=False,
                )

            row = AgentSessionRow(
                session_id=session_id,
                user_id=user_id,
                status="idle",
                active_turn_id=None,
                last_event_seq=1,
                last_turn_index=0,
                created_at=now,
                updated_at=now,
            )
            database.add(row)
            # 事件表通过外键引用会话。这里先把会话真正写入当前事务，
            # 避免 PostgreSQL 在两条新增记录之间无法推断正确插入顺序。
            await database.flush()
            database.add(
                AgentEventRow(
                    event_id=uuid4().hex,
                    session_id=session_id,
                    turn_id=None,
                    seq=1,
                    event_type="session/created",
                    payload={"user_id": user_id},
                    step_index=None,
                    created_at=now,
                )
            )
            try:
                await database.commit()
            except IntegrityError:
                # 两个请求同时创建同一会话时，失败方重新读取获胜记录。
                await database.rollback()
                existing = await database.get(AgentSessionRow, session_id)
                if existing is None:
                    raise
                self._require_user(existing, user_id)
                return OpenedSession(
                    session=self._session_record(existing),
                    created=False,
                )
            return OpenedSession(
                session=self._session_record(row),
                created=True,
            )

    async def append_event(
        self,
        *,
        session_id: str,
        event_type: EventType,
        payload: dict[str, Any],
        now: datetime,
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> SessionEvent:
        self._canonical_json_bytes(payload)
        async with self._sessions.begin() as database:
            session = await self._lock_session(database, session_id)
            row = self._append_event_row(
                database,
                session=session,
                event_type=event_type,
                payload=payload,
                now=now,
                turn_id=turn_id,
                step_index=step_index,
            )
        return self._event_record(row)

    async def list_events(self, session_id: str) -> list[SessionEvent]:
        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(AgentEventRow)
                    .where(AgentEventRow.session_id == session_id)
                    .order_by(AgentEventRow.seq)
                )
            ).all()
        if not rows:
            async with self._sessions() as database:
                if await database.get(AgentSessionRow, session_id) is None:
                    raise KeyError(f"unknown session: {session_id}")
        return [self._event_record(row) for row in rows]

    async def list_turn_events(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> list[SessionEvent]:
        """只读取一轮模型真正需要的事件，不再扫描整段会话。"""

        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(AgentEventRow)
                    .where(
                        AgentEventRow.session_id == session_id,
                        AgentEventRow.turn_id == turn_id,
                    )
                    .order_by(AgentEventRow.seq)
                )
            ).all()
        return [self._event_record(row) for row in rows]

    async def list_recent_turns(
        self,
        *,
        session_id: str,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[TurnRecord]:
        """读取最近已结束轮次；数据库先限量，再恢复自然对话顺序。"""

        if limit < 1:
            return []
        conditions = [
            AgentTurnRow.session_id == session_id,
            AgentTurnRow.ended_at.is_not(None),
            AgentTurnRow.answer.is_not(None),
        ]
        if exclude_turn_id is not None:
            conditions.append(AgentTurnRow.turn_id != exclude_turn_id)
        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(AgentTurnRow)
                    .where(*conditions)
                    .order_by(AgentTurnRow.turn_index.desc())
                    .limit(limit)
                )
            ).all()
        return [self._turn_record(row) for row in reversed(rows)]

    async def list_recent_turns_after(
        self,
        *,
        session_id: str,
        after_turn_index: int | None,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[TurnRecord]:
        """数据库先按最新轮次限量，再按自然对话顺序返回。"""

        if limit < 1:
            return []
        conditions = [
            AgentTurnRow.session_id == session_id,
            AgentTurnRow.ended_at.is_not(None),
            AgentTurnRow.answer.is_not(None),
        ]
        if after_turn_index is not None:
            conditions.append(AgentTurnRow.turn_index > after_turn_index)
        if exclude_turn_id is not None:
            conditions.append(AgentTurnRow.turn_id != exclude_turn_id)
        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(AgentTurnRow)
                    .where(*conditions)
                    .order_by(AgentTurnRow.turn_index.desc())
                    .limit(limit)
                )
            ).all()
        return [self._turn_record(row) for row in reversed(rows)]

    async def list_completed_turns_after(
        self,
        *,
        session_id: str,
        after_turn_index: int | None,
        limit: int,
    ) -> list[TurnRecord]:
        """按时间正序读取尚未进入片段总结的已结束轮次。"""

        if limit < 1:
            return []
        conditions = [
            AgentTurnRow.session_id == session_id,
            AgentTurnRow.ended_at.is_not(None),
            AgentTurnRow.answer.is_not(None),
        ]
        if after_turn_index is not None:
            conditions.append(AgentTurnRow.turn_index > after_turn_index)
        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(AgentTurnRow)
                    .where(*conditions)
                    .order_by(AgentTurnRow.turn_index)
                    .limit(limit)
                )
            ).all()
        return [self._turn_record(row) for row in rows]

    async def set_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        now: datetime,
    ) -> SessionRecord:
        async with self._sessions.begin() as database:
            row = await self._lock_session(database, session_id)
            row.status = status
            row.updated_at = now
        return self._session_record(row)

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        request_time: datetime,
        now: datetime,
    ) -> TurnRecord:
        async with self._sessions.begin() as database:
            session = await self._lock_session(database, session_id)
            if session.active_turn_id is not None:
                raise SessionBusyError(
                    f"session already has active turn: {session.active_turn_id}"
                )
            session.last_turn_index += 1
            row = AgentTurnRow(
                turn_id=turn_id,
                session_id=session_id,
                turn_index=session.last_turn_index,
                user_message=user_message,
                request_time=request_time,
                status="running",
                started_at=now,
                ended_at=None,
                answer=None,
                error_code=None,
                step_count=0,
                used_tools=[],
                usage_json=TurnUsage().model_dump(mode="json"),
            )
            database.add(row)
            # 后面的 turn/start 和 user/message 事件会引用这个轮次，
            # 所以必须先让 PostgreSQL 看见轮次记录，再追加事件。
            await database.flush()
            session.status = "active"
            session.active_turn_id = turn_id
            session.updated_at = now
            self._append_event_row(
                database,
                session=session,
                event_type="turn/start",
                payload={"request_time": request_time.isoformat()},
                now=now,
                turn_id=turn_id,
            )
            self._append_event_row(
                database,
                session=session,
                event_type="user/message",
                payload={"content": user_message},
                now=now,
                turn_id=turn_id,
            )
        return self._turn_record(row)

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
    ) -> TurnRecord:
        async with self._sessions.begin() as database:
            session = await self._lock_session(database, session_id)
            row = await database.scalar(
                select(AgentTurnRow)
                .where(AgentTurnRow.turn_id == turn_id)
                .with_for_update()
            )
            if row is None or row.session_id != session_id:
                raise KeyError(f"unknown turn: {turn_id}")
            if row.status != "running":
                raise PersistenceError(f"turn is already finalized: {turn_id}")
            row.status = turn_status
            row.ended_at = now
            row.answer = answer
            row.error_code = error_code
            row.step_count = step_count
            row.used_tools = list(used_tools)
            row.usage_json = usage.model_dump(mode="json")
            session.status = session_status
            if session.active_turn_id == turn_id:
                session.active_turn_id = None
            session.updated_at = now
            self._append_event_row(
                database,
                session=session,
                event_type="turn/end",
                payload={
                    "status": turn_status,
                    "error_code": error_code,
                    "step_count": step_count,
                    "used_tools": used_tools,
                    "usage": usage.model_dump(mode="json"),
                },
                now=now,
                turn_id=turn_id,
            )
        return self._turn_record(row)

    async def record_llm_call(self, record: LLMCallRecord) -> None:
        async with self._sessions.begin() as database:
            database.add(self._llm_call_row(record))

    async def record_model_response(
        self,
        *,
        record: LLMCallRecord,
        event_payload: dict,
        now: datetime,
    ) -> SessionEvent:
        """原子保存主模型调用统计和对应的助手消息事件。"""

        self._canonical_json_bytes(event_payload)
        async with self._sessions.begin() as database:
            session = await self._lock_session(database, record.session_id)
            turn = await database.get(AgentTurnRow, record.turn_id)
            if turn is None or turn.status != "running":
                raise KeyError(f"unknown running turn: {record.turn_id}")
            database.add(self._llm_call_row(record))
            event = self._append_event_row(
                database,
                session=session,
                event_type="assistant/message",
                payload=event_payload,
                now=now,
                turn_id=record.turn_id,
                step_index=record.step_index,
            )
        return self._event_record(event)

    async def save_domain_state(
        self,
        value: DomainStateWrite,
        *,
        now: datetime,
    ) -> DomainStateVersion:
        self._canonical_json_bytes(value.state)
        async with self._sessions.begin() as database:
            await self._lock_session(database, value.session_id)
            latest = await database.scalar(
                select(DomainStateVersionRow)
                .where(
                    DomainStateVersionRow.session_id == value.session_id,
                    DomainStateVersionRow.domain == value.domain,
                )
                .order_by(DomainStateVersionRow.version.desc())
                .limit(1)
            )
            previous_version = latest.version if latest is not None else 0
            if (
                value.expected_previous_version is not None
                and value.expected_previous_version != previous_version
            ):
                raise StateVersionConflictError(
                    f"expected version {value.expected_previous_version}, "
                    f"found {previous_version}"
                )
            row = DomainStateVersionRow(
                state_id=uuid4().hex,
                session_id=value.session_id,
                domain=value.domain,
                version=previous_version + 1,
                state_json=value.state,
                source_event_id=value.source_event_id,
                created_at=now,
            )
            database.add(row)
        return self._domain_state_record(row)

    async def get_latest_domain_state(
        self,
        *,
        session_id: str,
        domain: str,
    ) -> DomainStateVersion | None:
        async with self._sessions() as database:
            row = await database.scalar(
                select(DomainStateVersionRow)
                .where(
                    DomainStateVersionRow.session_id == session_id,
                    DomainStateVersionRow.domain == domain,
                )
                .order_by(DomainStateVersionRow.version.desc())
                .limit(1)
            )
        return self._domain_state_record(row) if row is not None else None

    async def get_working_memory(self, session_id: str) -> WorkingMemory | None:
        async with self._sessions() as database:
            row = await database.get(WorkingMemoryRow, session_id)
        return self._working_memory_record(row) if row is not None else None

    async def apply_tool_memory_update(
        self,
        *,
        session_id: str,
        update: ToolMemoryUpdate,
        now: datetime,
    ) -> WorkingMemory:
        async with self._sessions.begin() as database:
            await self._lock_session(database, session_id)
            row = await database.scalar(
                select(WorkingMemoryRow)
                .where(WorkingMemoryRow.session_id == session_id)
                .with_for_update()
            )
            current = self._working_memory_record(row) if row is not None else None
            merged = apply_tool_update(
                current,
                session_id=session_id,
                update=update,
                now=now,
            )
            if row is None:
                database.add(self._working_memory_row(merged))
            else:
                row.version = merged.version
                row.memory_json = merged.model_dump(mode="json")
                row.updated_at = now
        return merged.model_copy(deep=True)

    async def set_pending_question(
        self,
        *,
        session_id: str,
        question: str | None,
        now: datetime,
    ) -> WorkingMemory:
        async with self._sessions.begin() as database:
            await self._lock_session(database, session_id)
            row = await database.scalar(
                select(WorkingMemoryRow)
                .where(WorkingMemoryRow.session_id == session_id)
                .with_for_update()
            )
            current = self._working_memory_record(row) if row is not None else None
            updated = with_pending_question(
                current,
                session_id=session_id,
                question=question,
                now=now,
            )
            if row is None:
                database.add(self._working_memory_row(updated))
            else:
                row.version = updated.version
                row.memory_json = updated.model_dump(mode="json")
                row.updated_at = now
        return updated.model_copy(deep=True)

    async def save_episode(
        self,
        value: ConversationEpisodeDraft,
        *,
        now: datetime,
    ) -> ConversationEpisode:
        stored = ConversationEpisode(
            episode_id=uuid4().hex,
            created_at=now,
            **value.model_dump(),
        )
        async with self._sessions.begin() as database:
            session = await self._lock_session(database, value.session_id)
            self._require_user(session, value.user_id)
            database.add(
                ConversationEpisodeRow(
                    episode_id=stored.episode_id,
                    session_id=stored.session_id,
                    user_id=stored.user_id,
                    topic=stored.topic,
                    summary=stored.summary,
                    details_json={
                        "decisions": stored.decisions,
                        "unresolved_questions": stored.unresolved_questions,
                        "entities": [
                            item.model_dump(mode="json") for item in stored.entities
                        ],
                        "result_ids": stored.result_ids,
                        "source_turn_ids": stored.source_turn_ids,
                        "source_start_turn_index": stored.source_start_turn_index,
                        "source_end_turn_index": stored.source_end_turn_index,
                    },
                    source_started_at=stored.source_started_at,
                    source_ended_at=stored.source_ended_at,
                    created_at=now,
                )
            )
            memory_row = await database.scalar(
                select(WorkingMemoryRow)
                .where(WorkingMemoryRow.session_id == value.session_id)
                .with_for_update()
            )
            current = (
                self._working_memory_record(memory_row)
                if memory_row is not None
                else WorkingMemory(session_id=value.session_id, updated_at=now)
            )
            updated = current.model_copy(
                update={
                    "version": current.version + (1 if memory_row is not None else 0),
                    "summarized_through": value.source_ended_at,
                    "summarized_through_turn_index": value.source_end_turn_index,
                    "updated_at": now,
                },
                deep=True,
            )
            if memory_row is None:
                database.add(self._working_memory_row(updated))
            else:
                memory_row.version = updated.version
                memory_row.memory_json = updated.model_dump(mode="json")
                memory_row.updated_at = now
        return stored.model_copy(deep=True)

    async def list_recent_episodes(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> list[ConversationEpisode]:
        if limit < 1:
            return []
        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(ConversationEpisodeRow)
                    .where(ConversationEpisodeRow.session_id == session_id)
                    .order_by(ConversationEpisodeRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
        return [self._episode_record(row) for row in rows]

    async def search_episodes(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None,
        limit: int,
    ) -> list[ConversationEpisode]:
        """只在最近200条短总结中排序，不扫描原始事件和工具大结果。"""

        conditions = [ConversationEpisodeRow.user_id == user_id]
        if session_id is not None:
            conditions.append(ConversationEpisodeRow.session_id == session_id)
        async with self._sessions() as database:
            rows = (
                await database.scalars(
                    select(ConversationEpisodeRow)
                    .where(*conditions)
                    .order_by(ConversationEpisodeRow.created_at.desc())
                    .limit(200)
                )
            ).all()
        candidates = [self._episode_record(row) for row in rows]
        return rank_episode_matches(candidates, query, limit=limit)

    async def save_result(
        self,
        value: ResultArtifactDraft,
        *,
        now: datetime,
    ) -> ResultArtifact:
        raw = self._canonical_json_bytes(value.content)
        result_id = uuid4().hex
        digest = hashlib.sha256(raw).hexdigest()
        content_json: Any | None = value.content
        storage_uri: str | None = None
        stored_uri_to_clean: str | None = None
        if len(raw) > self._inline_result_max_bytes:
            if self._content_store is None:
                raise PersistenceError("large result requires an ArtifactContentStore")
            stored = await self._content_store.put(
                result_id=result_id,
                content=value.content,
            )
            if stored.sha256 != digest or stored.size_bytes != len(raw):
                await self._content_store.delete(stored.uri)
                raise PersistenceError("artifact content store changed result bytes")
            content_json = None
            storage_uri = stored.uri
            stored_uri_to_clean = stored.uri

        row = ResultArtifactRow(
            result_id=result_id,
            session_id=value.session_id,
            turn_id=value.turn_id,
            kind=value.kind,
            summary_json=value.summary,
            content_json=content_json,
            storage_uri=storage_uri,
            sha256=digest,
            size_bytes=len(raw),
            created_at=now,
            expires_at=value.expires_at,
        )
        try:
            async with self._sessions.begin() as database:
                await self._lock_session(database, value.session_id)
                if value.turn_id is not None:
                    turn = await database.get(AgentTurnRow, value.turn_id)
                    if turn is None or turn.session_id != value.session_id:
                        raise KeyError(f"unknown turn: {value.turn_id}")
                database.add(row)
        except Exception:
            if stored_uri_to_clean is not None and self._content_store is not None:
                await self._content_store.delete(stored_uri_to_clean)
            raise
        return self._result_record(row, content=None)

    async def get_result(
        self,
        *,
        result_id: str,
        user_id: str,
    ) -> ResultArtifact | None:
        async with self._sessions() as database:
            row = await database.scalar(
                select(ResultArtifactRow)
                .join(
                    AgentSessionRow,
                    AgentSessionRow.session_id == ResultArtifactRow.session_id,
                )
                .where(
                    ResultArtifactRow.result_id == result_id,
                    AgentSessionRow.user_id == user_id,
                )
            )
        if row is None:
            return None
        content = row.content_json
        if row.storage_uri is not None:
            if self._content_store is None:
                raise PersistenceError("artifact content store is not configured")
            content = await self._content_store.get(row.storage_uri)
        raw = self._canonical_json_bytes(content)
        if hashlib.sha256(raw).hexdigest() != row.sha256:
            raise PersistenceError("stored result checksum does not match metadata")
        return self._result_record(row, content=content)

    async def recover_interrupted_turns(
        self,
        *,
        now: datetime,
    ) -> RecoveryReport:
        interrupted: list[str] = []
        async with self._sessions.begin() as database:
            turns = (
                await database.scalars(
                    select(AgentTurnRow)
                    .where(AgentTurnRow.status == "running")
                    .with_for_update()
                )
            ).all()
            for turn in turns:
                session = await self._lock_session(database, turn.session_id)
                turn.status = "interrupted"
                turn.ended_at = now
                turn.error_code = "runtime_interrupted"
                session.status = "idle"
                if session.active_turn_id == turn.turn_id:
                    session.active_turn_id = None
                session.updated_at = now
                self._append_event_row(
                    database,
                    session=session,
                    event_type="turn/end",
                    payload={
                        "status": "interrupted",
                        "error_code": "runtime_interrupted",
                        "step_count": turn.step_count,
                        "used_tools": turn.used_tools,
                        "usage": turn.usage_json,
                    },
                    now=now,
                    turn_id=turn.turn_id,
                )
                interrupted.append(turn.turn_id)
        return RecoveryReport(interrupted_turn_ids=interrupted)

    async def healthcheck(self) -> PersistenceHealth:
        checked_at = datetime.now().astimezone()
        try:
            async with self._sessions() as database:
                await database.scalar(select(1))
                bind = database.get_bind()
                kind = bind.dialect.name
        # 健康检查必须把任意驱动或网络错误收敛成可返回的状态。
        except Exception as exc:  # noqa: BLE001
            return PersistenceHealth(
                ok=False,
                database_kind="unknown",
                checked_at=checked_at,
                error_code=type(exc).__name__,
            )
        return PersistenceHealth(
            ok=True,
            database_kind=kind,
            checked_at=checked_at,
        )

    @staticmethod
    async def _lock_session(
        database: AsyncSession,
        session_id: str,
    ) -> AgentSessionRow:
        row = await database.scalar(
            select(AgentSessionRow)
            .where(AgentSessionRow.session_id == session_id)
            .with_for_update()
        )
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        return row

    @staticmethod
    def _append_event_row(
        database: AsyncSession,
        *,
        session: AgentSessionRow,
        event_type: EventType,
        payload: dict[str, Any],
        now: datetime,
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> AgentEventRow:
        session.last_event_seq += 1
        session.updated_at = now
        row = AgentEventRow(
            event_id=uuid4().hex,
            session_id=session.session_id,
            turn_id=turn_id,
            seq=session.last_event_seq,
            event_type=event_type,
            payload=payload,
            step_index=step_index,
            created_at=now,
        )
        database.add(row)
        return row

    @staticmethod
    def _require_user(row: AgentSessionRow, user_id: str) -> None:
        if row.user_id != user_id:
            raise ValueError("session belongs to a different user")

    @staticmethod
    def _llm_call_row(record: LLMCallRecord) -> LLMCallRow:
        return LLMCallRow(
            llm_call_id=record.llm_call_id,
            session_id=record.session_id,
            turn_id=record.turn_id,
            step_index=record.step_index,
            purpose=record.purpose,
            provider=record.provider,
            model=record.model,
            status=record.status,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            latency_ms=record.latency_ms,
            provider_request_id=record.provider_request_id,
            tool_call_id=record.tool_call_id,
            error_code=record.error_code,
            created_at=record.created_at,
        )

    @staticmethod
    def _canonical_json_bytes(value: Any) -> bytes:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("persisted value must be JSON serializable") from exc

    @staticmethod
    def _session_record(row: AgentSessionRow) -> SessionRecord:
        return SessionRecord(
            session_id=row.session_id,
            user_id=row.user_id,
            status=cast(SessionStatus, row.status),
            created_at=PostgresAgentPersistence._aware(row.created_at),
            updated_at=PostgresAgentPersistence._aware(row.updated_at),
        )

    @staticmethod
    def _event_record(row: AgentEventRow) -> SessionEvent:
        return SessionEvent(
            event_id=row.event_id,
            session_id=row.session_id,
            seq=row.seq,
            type=cast(EventType, row.event_type),
            payload=row.payload,
            created_at=PostgresAgentPersistence._aware(row.created_at),
            turn_id=row.turn_id,
            step_index=row.step_index,
        )

    @staticmethod
    def _turn_record(row: AgentTurnRow) -> TurnRecord:
        return TurnRecord(
            turn_id=row.turn_id,
            session_id=row.session_id,
            turn_index=row.turn_index,
            user_message=row.user_message,
            request_time=PostgresAgentPersistence._aware(row.request_time),
            status=cast(TurnStatus, row.status),
            started_at=PostgresAgentPersistence._aware(row.started_at),
            ended_at=(
                PostgresAgentPersistence._aware(row.ended_at)
                if row.ended_at is not None
                else None
            ),
            answer=row.answer,
            error_code=row.error_code,
            step_count=row.step_count,
            used_tools=row.used_tools,
            usage=TurnUsage.model_validate(row.usage_json),
        )

    @staticmethod
    def _domain_state_record(row: DomainStateVersionRow) -> DomainStateVersion:
        return DomainStateVersion(
            state_id=row.state_id,
            session_id=row.session_id,
            domain=row.domain,
            version=row.version,
            state=row.state_json,
            source_event_id=row.source_event_id,
            created_at=PostgresAgentPersistence._aware(row.created_at),
        )

    @staticmethod
    def _working_memory_row(value: WorkingMemory) -> WorkingMemoryRow:
        return WorkingMemoryRow(
            session_id=value.session_id,
            version=value.version,
            memory_json=value.model_dump(mode="json"),
            updated_at=value.updated_at,
        )

    @staticmethod
    def _working_memory_record(row: WorkingMemoryRow) -> WorkingMemory:
        return WorkingMemory.model_validate(row.memory_json).model_copy(
            update={"updated_at": PostgresAgentPersistence._aware(row.updated_at)}
        )

    @staticmethod
    def _episode_record(row: ConversationEpisodeRow) -> ConversationEpisode:
        details = row.details_json
        return ConversationEpisode(
            episode_id=row.episode_id,
            session_id=row.session_id,
            user_id=row.user_id,
            topic=row.topic,
            summary=row.summary,
            decisions=details.get("decisions", []),
            unresolved_questions=details.get("unresolved_questions", []),
            entities=details.get("entities", []),
            result_ids=details.get("result_ids", []),
            source_turn_ids=details.get("source_turn_ids", []),
            source_start_turn_index=details.get("source_start_turn_index"),
            source_end_turn_index=details.get("source_end_turn_index"),
            source_started_at=PostgresAgentPersistence._aware(row.source_started_at),
            source_ended_at=PostgresAgentPersistence._aware(row.source_ended_at),
            created_at=PostgresAgentPersistence._aware(row.created_at),
        )

    @staticmethod
    def _result_record(row: ResultArtifactRow, *, content: Any) -> ResultArtifact:
        return ResultArtifact(
            result_id=row.result_id,
            session_id=row.session_id,
            turn_id=row.turn_id,
            kind=row.kind,
            summary=row.summary_json,
            content=content,
            storage_uri=row.storage_uri,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            created_at=PostgresAgentPersistence._aware(row.created_at),
            expires_at=(
                PostgresAgentPersistence._aware(row.expires_at)
                if row.expires_at is not None
                else None
            ),
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        """SQLite测试驱动会丢失时区；正式 PostgreSQL 返回值保持原时区。"""

        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
