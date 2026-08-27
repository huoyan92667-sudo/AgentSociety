from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

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
from yelp_agent.agent_platform.domains.restaurant import build_restaurant_tools
from yelp_agent.agent_platform.persistence.schema import DomainStateVersion
from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.answer_synthesis import RecommendationAnswer
from yelp_agent.recommendation_v2.business_facts import load_business_fact_catalog
from yelp_agent.recommendation_v2.preference_fusion import PreferenceFusionAttempt
from yelp_agent.recommendation_v2.schema import UnifiedRecommendationState
from yelp_agent.recommendation_v2.workflow import RecommendationTurnResult

NOW = datetime(2026, 8, 26, 13, 0, tzinfo=UTC)


class EmptyArguments(StrictModel):
    """测试用空参数；继承项目统一严格模型配置。"""


class FakeWorkflow:
    def __init__(self) -> None:
        self.requests = []
        self.restored = []
        self.closed = False

    def restore_state(self, state) -> None:
        self.restored.append(state)

    def process(self, request, *, on_answer_delta=None) -> RecommendationTurnResult:
        self.requests.append(request)
        state = UnifiedRecommendationState(
            user_id=request.user_id,
            session_id=request.session_id,
            revision=1,
            turn_index=1,
            latest_query_text=request.query_text,
        )
        return RecommendationTurnResult(
            fusion=PreferenceFusionAttempt(
                status="success",
                state=state,
                model="nested-fusion-model",
                latency_ms=10,
                input_tokens=100,
                output_tokens=20,
                model_call_count=1,
            ),
            answer=RecommendationAnswer(
                status="success",
                text="已经按完整原话处理。",
                model="nested-answer-model",
                input_tokens=50,
                output_tokens=10,
                latency_ms=5,
            ),
        )

    def close(self) -> None:
        self.closed = True


class MemoryDomainStateStore:
    def __init__(self) -> None:
        self.value: DomainStateVersion | None = None

    async def get_latest_domain_state(self, *, session_id: str, domain: str):
        return self.value

    async def save_domain_state(self, value, *, now: datetime):
        version = 1 if self.value is None else self.value.version + 1
        self.value = DomainStateVersion(
            state_id=f"state-{version}",
            session_id=value.session_id,
            domain=value.domain,
            version=version,
            state=value.state,
            source_event_id=value.source_event_id,
            created_at=now,
        )
        return self.value


def test_agent_selects_recommendation_and_tool_receives_exact_user_message() -> None:
    async def scenario() -> None:
        workflow = FakeWorkflow()
        states = MemoryDomainStateStore()
        tools = build_restaurant_tools(
            workflow=workflow,  # type: ignore[arg-type]
            business_catalog=load_business_fact_catalog(),
            state_store=states,
        )
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=ToolCallsAction(
                        calls=[
                            ToolCall(
                                call_id="call-recommend",
                                tool_name="recommend_restaurants",
                                arguments={},
                            )
                        ]
                    ),
                    model="main-model",
                    usage=TokenUsage(input_tokens=30, output_tokens=5),
                ),
                ModelResponse(
                    action=FinalAnswerAction(answer="已完成推荐。"),
                    model="main-model",
                    usage=TokenUsage(input_tokens=40, output_tokens=6),
                ),
            ]
        )
        runtime = AgentRuntime(
            model=model,
            session_store=MemorySessionStore(),
            tools=tools.definitions,
        )
        original = "我今晚9点想吃地道川菜，必须在唐人街。"

        result = await runtime.handle(
            AgentTurnInput(
                user_id="user-1",
                session_id="session-1",
                message=original,
                request_time=NOW,
            )
        )

        assert result.status == "completed"
        assert workflow.requests[0].query_text == original
        assert states.value is not None
        assert states.value.state["latest_query_text"] == original
        # 两次主模型 + 融合模型一次 + 最终推荐总结一次。
        assert result.usage.model_calls == 4
        assert result.usage.input_tokens == 220
        assert result.usage.output_tokens == 41
        tool_message = model.requests[1].messages[-1]
        assert tool_message.role == "tool"
        # 工具上下文只保留状态和展示商家，不再复制完整最终回答。
        assert "已经按完整原话处理" not in (tool_message.content or "")
        assert '"presented_businesses":[]' in (tool_message.content or "")

    asyncio.run(scenario())


def test_restaurant_facts_tool_reads_real_catalog() -> None:
    async def scenario() -> None:
        catalog = load_business_fact_catalog()
        business = catalog.all()[0]
        workflow = FakeWorkflow()
        tools = build_restaurant_tools(
            workflow=workflow,  # type: ignore[arg-type]
            business_catalog=catalog,
            state_store=MemoryDomainStateStore(),
        )
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=ToolCallsAction(
                        calls=[
                            ToolCall(
                                call_id="call-facts",
                                tool_name="lookup_business_facts",
                                arguments={"business_ids": [business.business_id]},
                            )
                        ]
                    ),
                    model="main-model",
                ),
                ModelResponse(
                    action=FinalAnswerAction(answer="事实查询完成。"),
                    model="main-model",
                ),
            ]
        )
        runtime = AgentRuntime(
            model=model,
            session_store=MemorySessionStore(),
            tools=tools.definitions,
        )

        result = await runtime.handle(
            AgentTurnInput(
                user_id="user-1",
                session_id="session-1",
                message="查一下这家餐厅。",
                request_time=NOW,
            )
        )

        assert result.status == "completed"
        tool_message = model.requests[1].messages[-1]
        assert business.business_id in (tool_message.content or "")
        assert business.name in (tool_message.content or "")
        tool_payload = json.loads(tool_message.content or "{}")
        parking = tool_payload["businesses"][0]["parking_semantics"]
        assert parking["parking_cost"] == "unknown"
        assert parking["discount_or_reimbursement"] == "unknown"

    asyncio.run(scenario())


def test_terminal_tool_answer_is_not_rewritten_by_main_model() -> None:
    async def scenario() -> None:
        model = ScriptedLanguageModel(
            [
                ModelResponse(
                    action=ToolCallsAction(
                        calls=[
                            ToolCall(
                                call_id="call-terminal",
                                tool_name="complete_recommendation",
                                arguments={},
                            )
                        ]
                    ),
                    model="main-model",
                )
            ]
        )
        tool = ToolDefinition(
            name="complete_recommendation",
            description="返回已经完成证据约束的最终推荐。",
            input_model=EmptyArguments,
            handler=lambda *_: ToolBodyResult(
                value={"top5": ["business-1"]},
                model_content="已完成推荐。",
                terminal_answer="这是不可二次改写的最终推荐。",
                nested_model_calls=1,
            ),
        )
        store = MemorySessionStore()
        runtime = AgentRuntime(model=model, session_store=store, tools=[tool])

        result = await runtime.handle(
            AgentTurnInput(
                user_id="user-1",
                session_id="terminal-session",
                message="给我推荐餐厅。",
                request_time=NOW,
            )
        )

        assert result.answer == "这是不可二次改写的最终推荐。"
        assert result.step_count == 1
        assert len(model.requests) == 1
        assert result.usage.model_calls == 2
        events = await store.list_events("terminal-session")
        final_message = [event for event in events if event.type == "assistant/message"][-1]
        assert final_message.payload["model"] == "tool:complete_recommendation"
        tool_result = next(event for event in events if event.type == "tool/result")
        assert "terminal_answer" not in tool_result.payload["result"]

    asyncio.run(scenario())
