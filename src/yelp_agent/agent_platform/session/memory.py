"""用于第一批骨架和单元测试的内存会话存储。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

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
from ..persistence.errors import SessionBusyError, StateVersionConflictError
from ..persistence.schema import (
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
from ..runtime.schema import TurnUsage
from .events import EventType, OpenedSession, SessionEvent, SessionRecord, SessionStatus


class MemorySessionStore:
    """在进程内保存事件；接口与后续 PostgreSQL 实现保持一致。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._events: dict[str, list[SessionEvent]] = {}
        self._active_turns: dict[str, str] = {}
        self._last_turn_indexes: dict[str, int] = {}
        self._turns: dict[str, TurnRecord] = {}
        self._llm_calls: dict[str, LLMCallRecord] = {}
        self._domain_states: dict[tuple[str, str], list[DomainStateVersion]] = {}
        self._results: dict[str, ResultArtifact] = {}
        self._working_memories: dict[str, WorkingMemory] = {}
        self._episodes: dict[str, ConversationEpisode] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        *,
        session_id: str,
        user_id: str,
        now: datetime,
    ) -> OpenedSession:
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.user_id != user_id:
                    raise ValueError("session belongs to a different user")
                return OpenedSession(
                    session=existing.model_copy(deep=True),
                    created=False,
                )

            session = SessionRecord(
                session_id=session_id,
                user_id=user_id,
                status="idle",
                created_at=now,
                updated_at=now,
            )
            self._sessions[session_id] = session
            self._events[session_id] = []
            self._last_turn_indexes[session_id] = 0
            self._append_event_unlocked(
                session_id=session_id,
                event_type="session/created",
                payload={"user_id": user_id},
                now=now,
            )
            return OpenedSession(
                session=session.model_copy(deep=True),
                created=True,
            )

    async def append_event(
        self,
        *,
        session_id: str,
        event_type: EventType,
        payload: dict[str, object],
        now: datetime,
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> SessionEvent:
        # PostgreSQL 的 JSONB 也只能接收 JSON 数据，所以内存版提前执行相同检查。
        try:
            json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("session event payload must be JSON serializable") from exc

        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            return self._append_event_unlocked(
                session_id=session_id,
                event_type=event_type,
                payload=payload,
                now=now,
                turn_id=turn_id,
                step_index=step_index,
            )

    async def list_events(self, session_id: str) -> list[SessionEvent]:
        async with self._lock:
            if session_id not in self._events:
                raise KeyError(f"unknown session: {session_id}")
            return [item.model_copy(deep=True) for item in self._events[session_id]]

    async def list_turn_events(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> list[SessionEvent]:
        """只读取当前轮事件，避免上下文重新扫描并回放整个会话。"""

        async with self._lock:
            if session_id not in self._events:
                raise KeyError(f"unknown session: {session_id}")
            return [
                item.model_copy(deep=True)
                for item in self._events[session_id]
                if item.turn_id == turn_id
            ]

    async def list_recent_turns(
        self,
        *,
        session_id: str,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[TurnRecord]:
        """返回最近已经结束的轮次，并保持由旧到新的对话顺序。"""

        if limit < 1:
            return []
        async with self._lock:
            values = [
                item
                for item in self._turns.values()
                if item.session_id == session_id
                and item.turn_id != exclude_turn_id
                and item.ended_at is not None
                and item.answer is not None
            ]
            # 字典按 begin_turn 的插入顺序保存；不要依赖测试机器上可能相同的时钟值。
            return [item.model_copy(deep=True) for item in values[-limit:]]

    async def list_recent_turns_after(
        self,
        *,
        session_id: str,
        after_turn_index: int | None,
        limit: int,
        exclude_turn_id: str | None = None,
    ) -> list[TurnRecord]:
        """读取尚未总结的最近轮次，供主模型直接保持短期上下文。"""

        if limit < 1:
            return []
        async with self._lock:
            values = [
                item
                for item in self._turns.values()
                if item.session_id == session_id
                and item.turn_id != exclude_turn_id
                and item.ended_at is not None
                and item.answer is not None
                and (
                    after_turn_index is None
                    or item.turn_index > after_turn_index
                )
            ]
            return [item.model_copy(deep=True) for item in values[-limit:]]

    async def list_completed_turns_after(
        self,
        *,
        session_id: str,
        after_turn_index: int | None,
        limit: int,
    ) -> list[TurnRecord]:
        """为片段总结读取尚未被总结的已完成轮次。"""

        if limit < 1:
            return []
        async with self._lock:
            values = [
                item
                for item in self._turns.values()
                if item.session_id == session_id
                and item.ended_at is not None
                and item.answer is not None
                and (
                    after_turn_index is None
                    or item.turn_index > after_turn_index
                )
            ]
            return [item.model_copy(deep=True) for item in values[:limit]]

    async def set_status(
        self,
        *,
        session_id: str,
        status: SessionStatus,
        now: datetime,
    ) -> SessionRecord:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            updated = self._sessions[session_id].model_copy(
                update={"status": status, "updated_at": now}
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def begin_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        request_time: datetime,
        now: datetime,
    ) -> TurnRecord:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            active = self._active_turns.get(session_id)
            if active is not None:
                raise SessionBusyError(f"session already has active turn: {active}")
            if turn_id in self._turns:
                raise ValueError(f"duplicate turn ID: {turn_id}")
            turn_index = self._last_turn_indexes.get(session_id, 0) + 1
            self._last_turn_indexes[session_id] = turn_index
            turn = TurnRecord(
                turn_id=turn_id,
                session_id=session_id,
                turn_index=turn_index,
                user_message=user_message,
                request_time=request_time,
                status="running",
                started_at=now,
            )
            self._turns[turn_id] = turn
            self._active_turns[session_id] = turn_id
            current = self._sessions[session_id]
            self._sessions[session_id] = current.model_copy(
                update={"status": "active", "updated_at": now}
            )
            self._append_event_unlocked(
                session_id=session_id,
                event_type="turn/start",
                payload={"request_time": request_time.isoformat()},
                now=now,
                turn_id=turn_id,
            )
            self._append_event_unlocked(
                session_id=session_id,
                event_type="user/message",
                payload={"content": user_message},
                now=now,
                turn_id=turn_id,
            )
            return turn.model_copy(deep=True)

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
        async with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None or turn.session_id != session_id:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn.status != "running":
                raise ValueError(f"turn is already finalized: {turn_id}")
            updated = turn.model_copy(
                update={
                    "status": turn_status,
                    "ended_at": now,
                    "answer": answer,
                    "error_code": error_code,
                    "step_count": step_count,
                    "used_tools": list(used_tools),
                    "usage": usage,
                }
            )
            self._turns[turn_id] = updated
            self._active_turns.pop(session_id, None)
            session = self._sessions[session_id]
            self._sessions[session_id] = session.model_copy(
                update={"status": session_status, "updated_at": now}
            )
            self._append_event_unlocked(
                session_id=session_id,
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
            return updated.model_copy(deep=True)

    async def record_llm_call(self, record: LLMCallRecord) -> None:
        async with self._lock:
            if record.llm_call_id in self._llm_calls:
                raise ValueError(f"duplicate LLM call ID: {record.llm_call_id}")
            self._llm_calls[record.llm_call_id] = record.model_copy(deep=True)

    async def record_model_response(
        self,
        *,
        record: LLMCallRecord,
        event_payload: dict,
        now: datetime,
    ) -> SessionEvent:
        """原子保存主模型调用统计和模型可见的助手消息。"""

        self._require_json(event_payload)
        async with self._lock:
            if record.llm_call_id in self._llm_calls:
                raise ValueError(f"duplicate LLM call ID: {record.llm_call_id}")
            turn = self._turns.get(record.turn_id)
            if turn is None or turn.status != "running":
                raise KeyError(f"unknown running turn: {record.turn_id}")
            self._llm_calls[record.llm_call_id] = record.model_copy(deep=True)
            return self._append_event_unlocked(
                session_id=record.session_id,
                event_type="assistant/message",
                payload=event_payload,
                now=now,
                turn_id=record.turn_id,
                step_index=record.step_index,
            )

    async def save_domain_state(
        self,
        value: DomainStateWrite,
        *,
        now: datetime,
    ) -> DomainStateVersion:
        self._require_json(value.state)
        async with self._lock:
            if value.session_id not in self._sessions:
                raise KeyError(f"unknown session: {value.session_id}")
            key = (value.session_id, value.domain)
            versions = self._domain_states.setdefault(key, [])
            previous_version = versions[-1].version if versions else 0
            if (
                value.expected_previous_version is not None
                and value.expected_previous_version != previous_version
            ):
                raise StateVersionConflictError(
                    f"expected version {value.expected_previous_version}, "
                    f"found {previous_version}"
                )
            stored = DomainStateVersion(
                state_id=uuid4().hex,
                session_id=value.session_id,
                domain=value.domain,
                version=previous_version + 1,
                state=value.state,
                source_event_id=value.source_event_id,
                created_at=now,
            )
            versions.append(stored)
            return stored.model_copy(deep=True)

    async def get_latest_domain_state(
        self,
        *,
        session_id: str,
        domain: str,
    ) -> DomainStateVersion | None:
        async with self._lock:
            versions = self._domain_states.get((session_id, domain), [])
            return versions[-1].model_copy(deep=True) if versions else None

    async def get_working_memory(self, session_id: str) -> WorkingMemory | None:
        async with self._lock:
            value = self._working_memories.get(session_id)
            return value.model_copy(deep=True) if value is not None else None

    async def apply_tool_memory_update(
        self,
        *,
        session_id: str,
        update: ToolMemoryUpdate,
        now: datetime,
    ) -> WorkingMemory:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            merged = apply_tool_update(
                self._working_memories.get(session_id),
                session_id=session_id,
                update=update,
                now=now,
            )
            self._working_memories[session_id] = merged
            return merged.model_copy(deep=True)

    async def set_pending_question(
        self,
        *,
        session_id: str,
        question: str | None,
        now: datetime,
    ) -> WorkingMemory:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            updated = with_pending_question(
                self._working_memories.get(session_id),
                session_id=session_id,
                question=question,
                now=now,
            )
            self._working_memories[session_id] = updated
            return updated.model_copy(deep=True)

    async def save_episode(
        self,
        value: ConversationEpisodeDraft,
        *,
        now: datetime,
    ) -> ConversationEpisode:
        async with self._lock:
            if value.session_id not in self._sessions:
                raise KeyError(f"unknown session: {value.session_id}")
            stored = ConversationEpisode(
                episode_id=uuid4().hex,
                created_at=now,
                **value.model_dump(),
            )
            self._episodes[stored.episode_id] = stored
            current = self._working_memories.get(value.session_id)
            current = current or WorkingMemory(
                session_id=value.session_id,
                updated_at=now,
            )
            self._working_memories[value.session_id] = current.model_copy(
                update={
                    "version": current.version + (1 if value.session_id in self._working_memories else 0),
                    "summarized_through": value.source_ended_at,
                    "summarized_through_turn_index": value.source_end_turn_index,
                    "updated_at": now,
                },
                deep=True,
            )
            return stored.model_copy(deep=True)

    async def list_recent_episodes(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> list[ConversationEpisode]:
        if limit < 1:
            return []
        async with self._lock:
            values = [
                item for item in self._episodes.values() if item.session_id == session_id
            ]
            values.sort(key=lambda item: item.created_at, reverse=True)
            return [item.model_copy(deep=True) for item in values[:limit]]

    async def search_episodes(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None,
        limit: int,
    ) -> list[ConversationEpisode]:
        async with self._lock:
            candidates = [
                item
                for item in self._episodes.values()
                if item.user_id == user_id
                and (session_id is None or item.session_id == session_id)
            ]
        return rank_episode_matches(candidates, query, limit=limit)

    async def save_result(
        self,
        value: ResultArtifactDraft,
        *,
        now: datetime,
    ) -> ResultArtifact:
        raw = self._canonical_json_bytes(value.content)
        async with self._lock:
            if value.session_id not in self._sessions:
                raise KeyError(f"unknown session: {value.session_id}")
            if value.turn_id is not None:
                turn = self._turns.get(value.turn_id)
                if turn is None or turn.session_id != value.session_id:
                    raise KeyError(f"unknown turn: {value.turn_id}")
            stored = ResultArtifact(
                result_id=uuid4().hex,
                session_id=value.session_id,
                turn_id=value.turn_id,
                kind=value.kind,
                summary=value.summary,
                content=value.content,
                storage_uri=None,
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
                created_at=now,
                expires_at=value.expires_at,
            )
            self._results[stored.result_id] = stored
            return stored.model_copy(update={"content": None}, deep=True)

    async def get_result(
        self,
        *,
        result_id: str,
        user_id: str,
    ) -> ResultArtifact | None:
        async with self._lock:
            stored = self._results.get(result_id)
            if stored is None:
                return None
            session = self._sessions[stored.session_id]
            if session.user_id != user_id:
                return None
            return stored.model_copy(deep=True)

    async def recover_interrupted_turns(
        self,
        *,
        now: datetime,
    ) -> RecoveryReport:
        interrupted: list[str] = []
        async with self._lock:
            for turn_id, turn in list(self._turns.items()):
                if turn.status != "running":
                    continue
                updated = turn.model_copy(
                    update={
                        "status": "interrupted",
                        "ended_at": now,
                        "error_code": "runtime_interrupted",
                    }
                )
                self._turns[turn_id] = updated
                self._active_turns.pop(turn.session_id, None)
                session = self._sessions[turn.session_id]
                self._sessions[turn.session_id] = session.model_copy(
                    update={"status": "idle", "updated_at": now}
                )
                self._append_event_unlocked(
                    session_id=turn.session_id,
                    event_type="turn/end",
                    payload={
                        "status": "interrupted",
                        "error_code": "runtime_interrupted",
                        "step_count": turn.step_count,
                        "used_tools": turn.used_tools,
                        "usage": turn.usage.model_dump(mode="json"),
                    },
                    now=now,
                    turn_id=turn_id,
                )
                interrupted.append(turn_id)
        return RecoveryReport(interrupted_turn_ids=interrupted)

    async def healthcheck(self) -> PersistenceHealth:
        return PersistenceHealth(
            ok=True,
            database_kind="memory",
            checked_at=datetime.now(UTC),
        )

    async def list_llm_calls(self) -> list[LLMCallRecord]:
        """仅供测试和本地诊断读取模型调用记录。"""

        async with self._lock:
            return [item.model_copy(deep=True) for item in self._llm_calls.values()]

    async def get_turn(self, turn_id: str) -> TurnRecord | None:
        """仅供测试和本地诊断读取轮次摘要。"""

        async with self._lock:
            value = self._turns.get(turn_id)
            return value.model_copy(deep=True) if value is not None else None

    def _append_event_unlocked(
        self,
        *,
        session_id: str,
        event_type: EventType,
        payload: dict[str, Any],
        now: datetime,
        turn_id: str | None = None,
        step_index: int | None = None,
    ) -> SessionEvent:
        self._require_json(payload)
        events = self._events[session_id]
        event = SessionEvent(
            event_id=uuid4().hex,
            session_id=session_id,
            seq=len(events) + 1,
            type=event_type,
            payload=dict(payload),
            created_at=now,
            turn_id=turn_id,
            step_index=step_index,
        )
        events.append(event)
        current = self._sessions[session_id]
        self._sessions[session_id] = current.model_copy(update={"updated_at": now})
        return event.model_copy(deep=True)

    @staticmethod
    def _require_json(value: Any) -> None:
        try:
            json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("persisted value must be JSON serializable") from exc

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
