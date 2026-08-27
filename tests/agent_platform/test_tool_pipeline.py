from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import Field

from yelp_agent.agent_platform import (
    PreExecuteDecision,
    ToolBodyResult,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolPipelineHooks,
)
from yelp_agent.agent_platform.tools import ToolPipeline, ToolRegistry
from yelp_agent.models import StrictModel


class SearchInput(StrictModel):
    query: str = Field(min_length=1)


class SearchOutput(StrictModel):
    result: str


def _context(call_id: str = "call-1") -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id="user-1",
        session_id="session-1",
        turn_id="turn-1",
        step_index=1,
        call_id=call_id,
        request_time=datetime(2026, 8, 26, tzinfo=UTC),
    )


def test_pipeline_runs_pre_wrapper_post_and_final_observer_in_order() -> None:
    order: list[str] = []
    observed = []

    def pre(execution):
        order.append("pre")
        return PreExecuteDecision(
            arguments={"query": execution.arguments.query.strip()}
        )

    def guard(execution):
        order.append("guard")

    async def wrapper(execution, call_next):
        order.append("wrapper_before")
        result = await call_next()
        order.append("wrapper_after")
        return result

    def handler(arguments, context):
        order.append("handler")
        return SearchOutput(result=f"找到：{arguments.query}")

    def post(execution, result):
        order.append("post")
        return result.model_copy(update={"model_content": "给模型的精简结果"})

    def observer(execution, result):
        order.append("observer")
        observed.append(result)

    definition = ToolDefinition(
        name="search_reviews",
        description="查询评论证据。",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=handler,
    )
    pipeline = ToolPipeline(
        ToolRegistry([definition]),
        hooks=ToolPipelineHooks(
            pre_execute=[pre],
            guards=[guard],
            execute_wrappers=[wrapper],
            post_execute=[post],
            result_observers=[observer],
        ),
    )

    result = asyncio.run(
        pipeline.execute(
            ToolCall(
                call_id="call-1",
                tool_name="search_reviews",
                arguments={"query": "  安静  "},
            ),
            context=_context(),
        )
    )

    assert result.status == "success"
    assert result.value == {"result": "找到：安静"}
    assert result.model_content == "给模型的精简结果"
    assert order == [
        "pre",
        "guard",
        "wrapper_before",
        "handler",
        "wrapper_after",
        "post",
        "observer",
    ]
    assert observed == [result]


def test_guard_denial_cannot_reach_tool_body_but_still_reaches_post_processing() -> (
    None
):
    handler_calls = 0
    post_statuses = []

    def handler(arguments, context):
        nonlocal handler_calls
        handler_calls += 1
        return {"result": "should not happen"}

    def guard(execution):
        return "这个结果编号不属于当前用户。"

    def post(execution, result):
        post_statuses.append(result.status)
        return result

    definition = ToolDefinition(
        name="search_reviews",
        description="查询评论证据。",
        input_model=SearchInput,
        handler=handler,
    )
    pipeline = ToolPipeline(
        ToolRegistry([definition]),
        hooks=ToolPipelineHooks(guards=[guard], post_execute=[post]),
    )

    result = asyncio.run(
        pipeline.execute(
            ToolCall(
                call_id="call-1",
                tool_name="search_reviews",
                arguments={"query": "服务"},
            ),
            context=_context(),
        )
    )

    assert result.status == "denied"
    assert result.error_code == "guard_denied"
    assert handler_calls == 0
    assert post_statuses == ["denied"]


def test_post_hook_cannot_change_tool_identity() -> None:
    definition = ToolDefinition(
        name="search_reviews",
        description="查询评论证据。",
        input_model=SearchInput,
        handler=lambda arguments, context: ToolBodyResult(value={"ok": True}),
    )

    def invalid_post(execution, result):
        return result.model_copy(update={"tool_name": "another_tool"})

    pipeline = ToolPipeline(
        ToolRegistry([definition]),
        hooks=ToolPipelineHooks(post_execute=[invalid_post]),
    )

    result = asyncio.run(
        pipeline.execute(
            ToolCall(
                call_id="call-1",
                tool_name="search_reviews",
                arguments={"query": "服务"},
            ),
            context=_context(),
        )
    )

    assert result.status == "error"
    assert result.tool_name == "search_reviews"
    assert result.error_code == "post_execute_failed"


def test_unknown_tool_returns_structured_denial() -> None:
    pipeline = ToolPipeline(ToolRegistry())

    result = asyncio.run(
        pipeline.execute(
            ToolCall(
                call_id="call-1",
                tool_name="unknown_tool",
                arguments={},
            ),
            context=_context(),
        )
    )

    assert result.status == "denied"
    assert result.error_code == "tool_not_available"
