from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from yelp_agent.agent_platform import (
    AgentRuntime,
    AgentTurnInput,
    FinalAnswerAction,
    MemorySessionStore,
    ModelResponse,
    ScriptedLanguageModel,
    ToolBodyResult,
    ToolCall,
    ToolCallsAction,
    ToolDefinition,
)
from yelp_agent.agent_platform.memory import (
    ConversationEpisodeDraft,
    RankedEntityReference,
    ResultSetReference,
    ToolMemoryUpdate,
    build_conversation_memory_tool,
)
from yelp_agent.agent_platform.runtime.schema import TurnUsage
from yelp_agent.models import StrictModel

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _turn(index: int, *, message: str | None = None) -> AgentTurnInput:
    return AgentTurnInput(
        user_id="user-1",
        session_id="session-1",
        message=message or f"问题{index}",
        request_time=NOW + timedelta(minutes=index),
    )


def test_context_only_replays_recent_completed_turns() -> None:
    async def scenario() -> None:
        store = MemorySessionStore()
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=FinalAnswerAction(answer=f"回答{index}"),
                    model="main-model",
                )
                for index in range(1, 9)
            ]
        )
        runtime = AgentRuntime(model=model, session_store=store)

        for index in range(1, 9):
            await runtime.handle(_turn(index))

        last_messages = [item.content for item in model.requests[-1].messages]
        assert "问题1" not in last_messages
        assert "回答1" not in last_messages
        assert last_messages == [
            "问题2",
            "回答2",
            "问题3",
            "回答3",
            "问题4",
            "回答4",
            "问题5",
            "回答5",
            "问题6",
            "回答6",
            "问题7",
            "回答7",
            "问题8",
        ]
        stats = model.requests[-1].context_stats
        assert stats is not None
        assert stats.included_completed_turns == 6

    asyncio.run(scenario())


class _EmptyArguments(StrictModel):
    pass


def test_tool_result_updates_generic_result_position_memory() -> None:
    async def scenario() -> None:
        def remember_results(arguments, context):
            items = [
                RankedEntityReference(
                    entity_type="restaurant",
                    entity_id=f"business-{index}",
                    display_name=f"餐厅{index}",
                    position=index,
                    source_turn_id=context.turn_id,
                )
                for index in range(1, 4)
            ]
            return ToolBodyResult(
                value={"businesses": [item.model_dump() for item in items]},
                model_content="三家餐厅已返回",
                memory_update=ToolMemoryUpdate(
                    result_sets=[
                        ResultSetReference(
                            result_type="restaurant_recommendation",
                            items=items,
                            source_turn_id=context.turn_id,
                        )
                    ]
                ),
            )

        tool = ToolDefinition(
            name="remember_results",
            description="返回三家测试餐厅。",
            input_model=_EmptyArguments,
            handler=remember_results,
        )
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=ToolCallsAction(
                        calls=[
                            ToolCall(
                                call_id="remember-call",
                                tool_name="remember_results",
                                arguments={},
                            )
                        ]
                    ),
                    model="main-model",
                ),
                ModelResponse(
                    action=FinalAnswerAction(answer="已经推荐三家。"),
                    model="main-model",
                ),
                ModelResponse(
                    action=FinalAnswerAction(answer="第三家是餐厅3。"),
                    model="main-model",
                ),
            ]
        )
        store = MemorySessionStore()
        runtime = AgentRuntime(model=model, session_store=store, tools=[tool])

        await runtime.handle(_turn(1, message="给我三家餐厅"))
        await runtime.handle(_turn(2, message="第三家叫什么？"))

        latest_request = model.requests[-1]
        assert "business-3" in latest_request.system_prompt
        assert "餐厅3" in latest_request.system_prompt
        memory = await store.get_working_memory("session-1")
        assert memory is not None
        assert memory.recent_result_sets[0].items[2].entity_id == "business-3"

    asyncio.run(scenario())


def test_old_turns_are_summarized_and_recent_original_words_remain() -> None:
    async def scenario() -> None:
        store = MemorySessionStore()
        main_model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=FinalAnswerAction(answer=f"回答{index}"),
                    model="main-model",
                )
                for index in range(1, 8)
            ]
        )
        summary_model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=FinalAnswerAction(
                        answer=(
                            '{"topic":"早期讨论","summary":"用户讨论了问题1和问题2。",'
                            '"decisions":["保留结论"],"unresolved_questions":[]}'
                        )
                    ),
                    model="summary-model",
                )
            ]
        )
        runtime = AgentRuntime(
            model=main_model,
            memory_summary_model=summary_model,
            session_store=store,
        )

        results = []
        for index in range(1, 8):
            results.append(await runtime.handle(_turn(index)))

        assert len(summary_model.requests) == 1
        assert "问题1" in (summary_model.requests[0].messages[0].content or "")
        latest = main_model.requests[-1]
        latest_messages = [item.content for item in latest.messages]
        assert "问题1" not in latest_messages
        assert "问题3" in latest_messages
        assert "用户讨论了问题1和问题2" in latest.system_prompt
        assert results[-1].usage.model_calls == 2
        episodes = await store.list_recent_episodes(session_id="session-1", limit=5)
        assert episodes[0].source_turn_ids

    asyncio.run(scenario())


def test_main_model_can_search_old_summary_and_request_original_dialogue() -> None:
    async def scenario() -> None:
        store = MemorySessionStore()
        await store.get_or_create(session_id="session-1", user_id="user-1", now=NOW)
        await store.begin_turn(
            session_id="session-1",
            turn_id="old-turn",
            user_message="以前决定去费城。",
            request_time=NOW,
            now=NOW,
        )
        await store.append_event(
            session_id="session-1",
            event_type="assistant/message",
            payload={
                "action": {"type": "final_answer", "answer": "决定周六去费城。"}
            },
            now=NOW,
            turn_id="old-turn",
            step_index=1,
        )
        await store.finish_turn(
            session_id="session-1",
            turn_id="old-turn",
            turn_status="completed",
            session_status="idle",
            answer="决定周六去费城。",
            error_code=None,
            step_count=1,
            used_tools=[],
            usage=TurnUsage(),
            now=NOW,
        )
        await store.save_episode(
            ConversationEpisodeDraft(
                session_id="session-1",
                user_id="user-1",
                topic="费城行程",
                summary="用户决定周六去费城。",
                source_turn_ids=["old-turn"],
                source_started_at=NOW,
                source_ended_at=NOW,
            ),
            now=NOW,
        )
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=ToolCallsAction(
                        calls=[
                            ToolCall(
                                call_id="memory-call",
                                tool_name="search_conversation_memory",
                                arguments={
                                    "query": "费城",
                                    "include_original_dialogue": True,
                                },
                            )
                        ]
                    ),
                    model="main-model",
                ),
                ModelResponse(
                    action=FinalAnswerAction(answer="你之前决定周六去费城。"),
                    model="main-model",
                ),
            ]
        )
        runtime = AgentRuntime(
            model=model,
            session_store=store,
            tools=[build_conversation_memory_tool(store)],
        )

        result = await runtime.handle(_turn(2, message="我们以前怎么定的？"))

        assert result.answer == "你之前决定周六去费城。"
        tool_message = model.requests[-1].messages[-1]
        assert "以前决定去费城" in (tool_message.content or "")
        assert "决定周六去费城" in (tool_message.content or "")

    asyncio.run(scenario())
