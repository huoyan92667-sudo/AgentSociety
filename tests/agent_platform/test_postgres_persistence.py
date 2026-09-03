from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

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
    ToolBodyResult,
    ToolCall,
    ToolCallsAction,
    ToolDefinition,
)
from yelp_agent.agent_platform.memory import (
    ConversationEpisodeDraft,
    EntityReference,
    ToolMemoryUpdate,
)
from yelp_agent.agent_platform.persistence.errors import StateVersionConflictError
from yelp_agent.agent_platform.persistence.tables import AgentTurnRow, LLMCallRow
from yelp_agent.agent_platform.results import LocalJsonContentStore
from yelp_agent.models import StrictModel

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def _settings(path: Path) -> DatabaseSettings:
    url = f"sqlite+aiosqlite:///{path.resolve().as_posix()}"
    return DatabaseSettings(url=SecretStr(url))


def test_runtime_persists_session_turn_events_and_llm_calls(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = AgentDatabase(_settings(tmp_path / "runtime.db"))
        await database.create_schema_for_tests()
        store = PostgresAgentPersistence(database.sessions)
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=FinalAnswerAction(answer="已经持久化。"),
                    model="fake-model",
                    provider="fake-provider",
                    latency_ms=12.5,
                )
            ]
        )
        runtime = AgentRuntime(model=model, session_store=store)

        result = await runtime.handle(
            AgentTurnInput(
                user_id="user-1",
                session_id="session-1",
                message="保存这轮对话。",
                request_time=NOW,
            )
        )
        events = await store.list_events("session-1")
        health = await store.healthcheck()
        async with database.sessions() as sql:
            turn = await sql.get(AgentTurnRow, result.turn_id)
            llm_calls = (await sql.execute(LLMCallRow.__table__.select())).all()

        assert result.status == "completed"
        assert health.ok is True
        assert health.database_kind == "sqlite"
        assert turn is not None
        assert turn.status == "completed"
        assert turn.answer == "已经持久化。"
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
        assert [event.type for event in events] == [
            "session/created",
            "turn/start",
            "user/message",
            "step/start",
            "model/request",
            "assistant/message",
            "step/end",
            "turn/end",
        ]
        assert len(llm_calls) == 1
        assert llm_calls[0].model == "fake-model"
        assert llm_calls[0].provider == "fake-provider"
        await database.close()

    asyncio.run(scenario())


def test_domain_state_is_versioned_and_rejects_stale_writes(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = AgentDatabase(_settings(tmp_path / "state.db"))
        await database.create_schema_for_tests()
        store = PostgresAgentPersistence(database.sessions)
        await store.get_or_create(
            session_id="session-1",
            user_id="user-1",
            now=NOW,
        )

        first = await store.save_domain_state(
            DomainStateWrite(
                session_id="session-1",
                domain="restaurant",
                state={"hard_constraints": {"category": "Szechuan"}},
                expected_previous_version=0,
            ),
            now=NOW,
        )
        second = await store.save_domain_state(
            DomainStateWrite(
                session_id="session-1",
                domain="restaurant",
                state={"hard_constraints": {"category": "Cantonese"}},
                expected_previous_version=1,
            ),
            now=NOW,
        )
        latest = await store.get_latest_domain_state(
            session_id="session-1",
            domain="restaurant",
        )

        assert first.version == 1
        assert second.version == 2
        assert latest == second
        with pytest.raises(StateVersionConflictError):
            await store.save_domain_state(
                DomainStateWrite(
                    session_id="session-1",
                    domain="restaurant",
                    state={"hard_constraints": {"category": "Japanese"}},
                    expected_previous_version=1,
                ),
                now=NOW,
            )
        await database.close()

    asyncio.run(scenario())


def test_large_result_is_stored_outside_database_and_checked_by_user(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database = AgentDatabase(_settings(tmp_path / "results.db"))
        await database.create_schema_for_tests()
        content_store = LocalJsonContentStore(tmp_path / "artifacts")
        store = PostgresAgentPersistence(
            database.sessions,
            content_store=content_store,
            inline_result_max_bytes=1024,
        )
        await store.get_or_create(
            session_id="session-1",
            user_id="user-1",
            now=NOW,
        )
        content = {"reviews": [f"完整评论-{index}-" + "x" * 80 for index in range(80)]}

        saved = await store.save_result(
            ResultArtifactDraft(
                session_id="session-1",
                kind="review_evidence/full_audit",
                summary={"review_count": 80},
                content=content,
            ),
            now=NOW,
        )
        loaded = await store.get_result(
            result_id=saved.result_id,
            user_id="user-1",
        )
        denied = await store.get_result(
            result_id=saved.result_id,
            user_id="another-user",
        )

        assert saved.content is None
        assert saved.storage_uri is not None
        assert loaded is not None
        assert loaded.content == content
        assert denied is None
        assert (tmp_path / "artifacts" / saved.storage_uri).is_file()
        await database.close()

    asyncio.run(scenario())


class LargeToolInput(StrictModel):
    count: int


def test_tool_post_processing_keeps_large_value_out_of_session_event(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database = AgentDatabase(_settings(tmp_path / "tool-result.db"))
        await database.create_schema_for_tests()
        content_store = LocalJsonContentStore(tmp_path / "tool-artifacts")
        store = PostgresAgentPersistence(
            database.sessions,
            content_store=content_store,
            inline_result_max_bytes=1024,
        )

        def build_large_result(arguments, context):
            return ToolBodyResult(
                value={
                    "reviews": [
                        f"review-{index}-" + "x" * 100
                        for index in range(arguments.count)
                    ]
                },
                model_content="已找到评论，完整内容已保留。",
            )

        tool = ToolDefinition(
            name="search_review_evidence",
            description="查询并保存评论证据。",
            input_model=LargeToolInput,
            handler=build_large_result,
        )
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=ToolCallsAction(
                        calls=[
                            ToolCall(
                                call_id="call-1",
                                tool_name="search_review_evidence",
                                arguments={"count": 80},
                            )
                        ]
                    ),
                    model="fake-model",
                ),
                ModelResponse(
                    action=FinalAnswerAction(answer="评论证据已经查到。"),
                    model="fake-model",
                ),
            ]
        )
        runtime = AgentRuntime(
            model=model,
            session_store=store,
            result_store=store,
            tools=[tool],
            large_tool_result_threshold_bytes=1024,
        )

        result = await runtime.handle(
            AgentTurnInput(
                user_id="user-1",
                session_id="session-1",
                message="查询这些评论。",
                request_time=NOW,
            )
        )
        events = await store.list_events("session-1")
        tool_result = next(event for event in events if event.type == "tool/result")
        stored_result = tool_result.payload["result"]
        result_id = stored_result["artifact_id"]
        loaded = await store.get_result(result_id=result_id, user_id="user-1")

        assert result.status == "completed"
        assert stored_result["value"]["result_id"] == result_id
        assert "reviews" not in stored_result["value"]
        assert loaded is not None
        assert len(loaded.content["reviews"]) == 80
        await database.close()

    asyncio.run(scenario())


def test_recovery_closes_interrupted_turn_and_allows_next_turn(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = AgentDatabase(_settings(tmp_path / "recovery.db"))
        await database.create_schema_for_tests()
        store = PostgresAgentPersistence(database.sessions)
        await store.get_or_create(
            session_id="session-1",
            user_id="user-1",
            now=NOW,
        )
        await store.begin_turn(
            session_id="session-1",
            turn_id="interrupted-turn",
            user_message="程序在这里中断。",
            request_time=NOW,
            now=NOW,
        )

        report = await store.recover_interrupted_turns(now=NOW)
        events = await store.list_events("session-1")
        next_turn = await store.begin_turn(
            session_id="session-1",
            turn_id="next-turn",
            user_message="继续对话。",
            request_time=NOW,
            now=NOW,
        )
        async with database.sessions() as sql:
            interrupted = await sql.get(AgentTurnRow, "interrupted-turn")

        assert report.interrupted_turn_ids == ["interrupted-turn"]
        assert interrupted is not None
        assert interrupted.status == "interrupted"
        assert events[-1].type == "turn/end"
        assert events[-1].payload["error_code"] == "runtime_interrupted"
        assert next_turn.status == "running"
        await database.close()

    asyncio.run(scenario())


def test_postgres_store_persists_working_memory_and_searchable_episode(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database = AgentDatabase(_settings(tmp_path / "conversation-memory.db"))
        await database.create_schema_for_tests()
        store = PostgresAgentPersistence(database.sessions)
        await store.get_or_create(
            session_id="session-1",
            user_id="user-1",
            now=NOW,
        )
        memory = await store.apply_tool_memory_update(
            session_id="session-1",
            update=ToolMemoryUpdate(
                focused_entities=[
                    EntityReference(
                        entity_type="restaurant",
                        entity_id="business-1",
                        display_name="Han Dynasty",
                        source_turn_id="turn-1",
                    )
                ]
            ),
            now=NOW,
        )
        episode = await store.save_episode(
            ConversationEpisodeDraft(
                session_id="session-1",
                user_id="user-1",
                topic="费城川菜",
                summary="用户曾讨论费城的川菜餐厅。",
                entities=memory.focused_entities,
                source_turn_ids=["turn-1"],
                source_started_at=NOW,
                source_ended_at=NOW,
            ),
            now=NOW,
        )

        reopened = await store.get_working_memory("session-1")
        matches = await store.search_episodes(
            user_id="user-1",
            query="费城川菜",
            session_id="session-1",
            limit=3,
        )

        assert reopened is not None
        assert reopened.focused_entities[0].entity_id == "business-1"
        assert reopened.summarized_through == NOW
        assert matches[0].episode_id == episode.episode_id
        await database.close()

    asyncio.run(scenario())
