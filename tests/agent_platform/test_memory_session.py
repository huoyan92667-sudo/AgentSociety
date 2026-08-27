from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from yelp_agent.agent_platform import MemorySessionStore

NOW = datetime(2026, 8, 26, tzinfo=UTC)


def test_session_cannot_be_reopened_by_another_user() -> None:
    store = MemorySessionStore()
    opened = asyncio.run(
        store.get_or_create(session_id="session-1", user_id="user-1", now=NOW)
    )

    assert opened.created is True

    with pytest.raises(ValueError, match="different user"):
        asyncio.run(
            store.get_or_create(
                session_id="session-1",
                user_id="user-2",
                now=NOW,
            )
        )


def test_event_sequence_is_monotonic_and_payload_must_be_json() -> None:
    store = MemorySessionStore()
    asyncio.run(store.get_or_create(session_id="session-1", user_id="user-1", now=NOW))
    first = asyncio.run(
        store.append_event(
            session_id="session-1",
            event_type="user/message",
            payload={"content": "你好"},
            now=NOW,
        )
    )
    second = asyncio.run(
        store.append_event(
            session_id="session-1",
            event_type="assistant/message",
            payload={"action": {"type": "final_answer", "answer": "你好"}},
            now=NOW,
        )
    )

    # 创建会话本身已经写入第1条 session/created 事件。
    assert first.seq == 2
    assert second.seq == 3

    with pytest.raises(ValueError, match="JSON"):
        asyncio.run(
            store.append_event(
                session_id="session-1",
                event_type="turn/end",
                payload={"bad": object()},
                now=NOW,
            )
        )
