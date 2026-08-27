"""真实 PostgreSQL 联调测试；未提供数据库地址时自动跳过。"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import inspect, text

from yelp_agent.agent_platform import (
    AgentDatabase,
    AgentRuntime,
    AgentTurnInput,
    DatabaseSettings,
    DomainStateWrite,
    FinalAnswerAction,
    ModelResponse,
    PostgresAgentPersistence,
    ResultArtifactDraft,
    ScriptedLanguageModel,
    TokenUsage,
)

DATABASE_URL = os.getenv("AGENT_DATABASE_URL", "")


@pytest.mark.skipif(
    not DATABASE_URL.startswith("postgresql+asyncpg://"),
    reason="需要通过 AGENT_DATABASE_URL 指定真实 PostgreSQL",
)
def test_real_postgres_round_trip() -> None:
    """跑一轮 Agent，并验证会话、状态、结果和模型统计都能写入再读出。"""

    async def scenario() -> None:
        database = AgentDatabase(DatabaseSettings.from_environment())
        store = PostgresAgentPersistence(database.sessions)
        suffix = uuid4().hex[:12]
        user_id = f"postgres-smoke-user-{suffix}"
        session_id = f"postgres-smoke-session-{suffix}"
        now = datetime.now(UTC)

        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=FinalAnswerAction(answer="真实 PostgreSQL 写入成功。"),
                    model="persistence-smoke-model",
                    provider="local-test",
                    usage=TokenUsage(input_tokens=11, output_tokens=7),
                    latency_ms=3.5,
                )
            ]
        )
        runtime = AgentRuntime(model=model, session_store=store)

        try:
            turn = await runtime.handle(
                AgentTurnInput(
                    user_id=user_id,
                    session_id=session_id,
                    message="验证真实数据库持久化。",
                    request_time=now,
                )
            )
            state = await store.save_domain_state(
                DomainStateWrite(
                    session_id=session_id,
                    domain="restaurant",
                    state={
                        "hard_constraints": {"category": ["Szechuan"]},
                        "soft_preferences": ["authentic_food"],
                    },
                    expected_previous_version=0,
                ),
                now=now,
            )
            saved_result = await store.save_result(
                ResultArtifactDraft(
                    session_id=session_id,
                    turn_id=turn.turn_id,
                    kind="persistence/smoke_result",
                    summary={"business_count": 1},
                    content={"business_ids": ["smoke-business-1"]},
                ),
                now=now,
            )

            events = await store.list_events(session_id)
            latest_state = await store.get_latest_domain_state(
                session_id=session_id,
                domain="restaurant",
            )
            loaded_result = await store.get_result(
                result_id=saved_result.result_id,
                user_id=user_id,
            )
            health = await store.healthcheck()

            async with database.engine.connect() as connection:
                table_names = await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_table_names()
                )
                revision = (
                    await connection.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar_one()

            expected_tables = {
                "agent_sessions",
                "agent_turns",
                "agent_events",
                "agent_domain_state_versions",
                "agent_result_artifacts",
                "agent_llm_calls",
            }
            assert health.ok is True
            assert health.database_kind == "postgresql"
            assert revision == "0001_agent_persistence"
            assert expected_tables.issubset(set(table_names))
            assert turn.status == "completed"
            assert turn.answer == "真实 PostgreSQL 写入成功。"
            assert latest_state is not None and latest_state.version == 1
            assert loaded_result is not None
            assert loaded_result.content == {"business_ids": ["smoke-business-1"]}
            assert [event.seq for event in events] == list(range(1, len(events) + 1))

            # 这段输出让人工联调时能直接看到真实写入的记录编号和验证结果。
            print(
                json.dumps(
                    {
                        "database": health.database_kind,
                        "migration_revision": revision,
                        "session_id": session_id,
                        "turn_id": turn.turn_id,
                        "event_count": len(events),
                        "event_types": [event.type for event in events],
                        "domain_state_version": state.version,
                        "result_id": saved_result.result_id,
                        "result_loaded": loaded_result.content,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            await database.close()

    asyncio.run(scenario())
