from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import Field

from yelp_agent.agent_platform import (
    AgentRuntime,
    AgentTurnInput,
    FinalAnswerAction,
    MemorySessionStore,
    ModelResponse,
    ScriptedLanguageModel,
    TokenUsage,
    ToolBodyResult,
    ToolCall,
    ToolCallsAction,
    ToolDefinition,
)
from yelp_agent.models import StrictModel

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def _turn(message: str = "怎么和面？") -> AgentTurnInput:
    return AgentTurnInput(
        user_id="user-1",
        session_id="session-1",
        message=message,
        request_time=NOW,
    )


def test_runtime_can_answer_directly_without_calling_tools() -> None:
    store = MemorySessionStore()
    model = ScriptedLanguageModel(
        [
            ModelResponse(
                action=FinalAnswerAction(answer="面粉加水后揉成团。"),
                model="fake-model",
                usage=TokenUsage(input_tokens=20, output_tokens=8),
                latency_ms=12.0,
            )
        ]
    )
    runtime = AgentRuntime(model=model, session_store=store)

    result = asyncio.run(runtime.handle(_turn()))
    events = asyncio.run(store.list_events("session-1"))

    assert result.status == "completed"
    assert result.answer == "面粉加水后揉成团。"
    assert result.step_count == 1
    assert result.used_tools == []
    assert result.usage.model_calls == 1
    assert result.usage.tool_calls == 0
    assert result.usage.total_tokens == 28
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
    request_event = next(event for event in events if event.type == "model/request")
    assert "system_prompt" not in request_event.payload
    assert len(request_event.payload["system_prompt_sha256"]) == 64
    end_event = events[-1]
    assert "answer" not in end_event.payload
    # 最终回答仍然保存在助手消息和轮次摘要中，不影响历史恢复。
    assert result.answer == "面粉加水后揉成团。"


class BusinessInput(StrictModel):
    business_id: str = Field(min_length=1)


class BusinessOutput(StrictModel):
    business_id: str
    name: str


def test_tool_result_is_added_to_history_before_model_continues() -> None:
    seen_contexts = []

    def read_business(arguments, context):
        seen_contexts.append(context)
        return ToolBodyResult(
            value=BusinessOutput(
                business_id=arguments.business_id,
                name="Han Dynasty",
            ),
            model_content="商家名称：Han Dynasty",
            nested_model_usage=TokenUsage(input_tokens=3, output_tokens=2),
        )

    tool = ToolDefinition(
        name="get_business_facts",
        description="读取指定商家的基础事实。",
        input_model=BusinessInput,
        output_model=BusinessOutput,
        handler=read_business,
    )
    model = ScriptedLanguageModel(
        [
            ModelResponse(
                action=ToolCallsAction(
                    calls=[
                        ToolCall(
                            call_id="call-1",
                            tool_name="get_business_facts",
                            arguments={"business_id": "business-1"},
                        )
                    ]
                ),
                model="fake-model",
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            ),
            ModelResponse(
                action=FinalAnswerAction(answer="这家店是 Han Dynasty。"),
                model="fake-model",
                usage=TokenUsage(input_tokens=40, output_tokens=10),
            ),
        ]
    )
    store = MemorySessionStore()
    runtime = AgentRuntime(model=model, session_store=store, tools=[tool])

    result = asyncio.run(runtime.handle(_turn("第一家叫什么？")))
    events = asyncio.run(store.list_events("session-1"))

    assert result.status == "completed"
    assert result.step_count == 2
    assert result.used_tools == ["get_business_facts"]
    assert result.usage.model_calls == 2
    assert result.usage.tool_calls == 1
    assert result.usage.input_tokens == 73
    assert result.usage.output_tokens == 22
    assert seen_contexts[0].user_id == "user-1"
    assert seen_contexts[0].session_id == "session-1"

    second_request = model.requests[1]
    assert [message.role for message in second_request.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert second_request.messages[-1].content == "商家名称：Han Dynasty"
    assert second_request.messages[-1].tool_call_id == "call-1"
    assert [event.type for event in events].count("tool/call") == 1
    assert [event.type for event in events].count("tool/result") == 1


def test_tool_can_stream_without_persisting_each_text_delta() -> None:
    """流式片段只发给调用方，事件日志仍只保留完整最终答案。"""

    class EmptyInput(StrictModel):
        pass

    def stream_answer(arguments, context):
        callback = context.metadata["answer_delta_callback"]
        callback("第一段")
        callback("第二段")
        return ToolBodyResult(
            value={"ok": True},
            model_content="完成",
            terminal_answer="第一段第二段",
        )

    tool = ToolDefinition(
        name="stream_answer",
        description="生成流式回答。",
        input_model=EmptyInput,
        handler=stream_answer,
    )
    model = ScriptedLanguageModel(
        [
            ModelResponse(
                action=ToolCallsAction(
                    calls=[
                        ToolCall(
                            call_id="stream-call",
                            tool_name="stream_answer",
                            arguments={},
                        )
                    ]
                ),
                model="fake-model",
            )
        ]
    )
    store = MemorySessionStore()
    runtime = AgentRuntime(model=model, session_store=store, tools=[tool])
    deltas: list[str] = []

    result = asyncio.run(runtime.handle(_turn("流式回答"), on_answer_delta=deltas.append))
    events = asyncio.run(store.list_events("session-1"))

    assert result.answer == "第一段第二段"
    assert deltas == ["第一段", "第二段"]
    serialized_events = "\n".join(event.model_dump_json() for event in events)
    assert "第一段第二段" in serialized_events
    assert '"第一段"' not in serialized_events
    assert '"第二段"' not in serialized_events


def test_invalid_arguments_do_not_reach_tool_handler() -> None:
    call_count = 0

    def must_not_run(arguments, context):
        nonlocal call_count
        call_count += 1
        raise AssertionError("invalid arguments must be rejected before execution")

    tool = ToolDefinition(
        name="get_business_facts",
        description="读取指定商家的基础事实。",
        input_model=BusinessInput,
        handler=must_not_run,
    )
    model = ScriptedLanguageModel(
        [
            ModelResponse(
                action=ToolCallsAction(
                    calls=[
                        ToolCall(
                            call_id="bad-call",
                            tool_name="get_business_facts",
                            arguments={},
                        )
                    ]
                ),
                model="fake-model",
            ),
            ModelResponse(
                action=FinalAnswerAction(answer="缺少商家编号，暂时无法查询。"),
                model="fake-model",
            ),
        ]
    )
    store = MemorySessionStore()
    runtime = AgentRuntime(model=model, session_store=store, tools=[tool])

    result = asyncio.run(runtime.handle(_turn("查一下这家店。")))

    assert result.status == "completed"
    assert call_count == 0
    tool_message = model.requests[1].messages[-1]
    assert "参数" in (tool_message.content or "")


def test_multiple_user_turns_are_rebuilt_from_the_same_session_log() -> None:
    store = MemorySessionStore()
    model = ScriptedLanguageModel(
        [
            ModelResponse(
                action=FinalAnswerAction(answer="第一轮回答。"),
                model="fake-model",
            ),
            ModelResponse(
                action=FinalAnswerAction(answer="第二轮回答。"),
                model="fake-model",
            ),
        ]
    )
    runtime = AgentRuntime(model=model, session_store=store)

    first = asyncio.run(runtime.handle(_turn("第一轮问题。")))
    second = asyncio.run(runtime.handle(_turn("第二轮问题。")))

    assert first.status == "completed"
    assert second.status == "completed"
    assert [message.content for message in model.requests[1].messages] == [
        "第一轮问题。",
        "第一轮回答。",
        "第二轮问题。",
    ]
