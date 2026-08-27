"""把真实模型、PostgreSQL 和业务工具装配成可直接调用的应用。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from typing import Self

from dotenv import load_dotenv

from yelp_agent.recommendation_v2.business_facts import (
    load_business_fact_catalog,
)
from yelp_agent.recommendation_v2.workflow import build_recommendation_workflow

from .domains.restaurant import RestaurantToolSet, build_restaurant_tools
from .llm import AgentModelSettings, OpenAICompatibleAgentModel
from .persistence import AgentDatabase, DatabaseSettings, PostgresAgentPersistence
from .results import LocalJsonContentStore
from .runtime.runtime import AgentRuntime
from .runtime.schema import (
    AgentLimits,
    AgentStreamEvent,
    AgentTurnInput,
    AgentTurnResult,
)

RESTAURANT_AGENT_PROMPT = """
你是一个通用生活助手，可以回答普通问题，也可以根据用户目标自行选择工具。

餐饮工具使用规则：
1. 用户要求找餐厅、重新推荐或修改上次餐饮条件时，调用 recommend_restaurants。
   这个工具会由服务器直接读取用户当前原话，因此参数必须是空对象，不要重新抄写或改写问题。
2. 用户只追问某家餐厅的地址、评分、价格档位、停车、营业时间或其他基础事实时，
   从历史推荐结果找到对应 business_id，再调用 lookup_business_facts；不要重跑完整推荐。
3. recommend_restaurants 已经完成四路要求融合、硬过滤、评论证据排序和最终推荐总结。
   你必须保留工具给出的前五顺序、商家和重要反面证据，不得自行替换或重新排序。
4. 工具没有给出的事实不能靠常识补充。历史 Yelp 营业时间只能表述为数据记录，不能冒充实时状态。
5. 工具调用后阅读其真实结果，再用自然中文回答用户。相同问题不要重复调用相同工具。
6. 如果问题与餐饮无关且你能够直接回答，就直接回答；不要为了显示能力而调用餐饮工具。
7. 如果完整推荐返回零候选，只能根据 applied_filter_steps 中真正把候选降为零的条件说明原因，
   并询问用户是否愿意修改这一个条件；禁止凭空追加位置、预算、氛围等无关问题。
8. 商家停车字段必须按字面解释：parking_lot 只表示是否记录有专用停车场，不表示免费；
   parking_validated 只表示是否记录有停车验证，不等于报销；空值表示数据未知。
   即使 parking_validated=false，也禁止写成“不能凭小票减免/报销”；停车费用、免费与否、
   减免和报销只要没有独立数据就必须说未知。
""".strip()


class RestaurantAgentApplication:
    """持有整套资源，并提供处理消息和安全关闭两个简单入口。"""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        database: AgentDatabase,
        restaurant_tools: RestaurantToolSet,
    ) -> None:
        self._runtime = runtime
        self._database = database
        self._restaurant_tools = restaurant_tools
        self._closed = False

    async def __aenter__(self) -> Self:
        await self._runtime.recover_interrupted()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def handle(self, value: AgentTurnInput) -> AgentTurnResult:
        if self._closed:
            raise RuntimeError("agent application is closed")
        return await self._runtime.handle(value)

    async def handle_stream(
        self,
        value: AgentTurnInput,
    ) -> AsyncIterator[AgentStreamEvent]:
        """边生成边返回文字；最后再返回一次完整、已持久化的轮次结果。"""

        if self._closed:
            raise RuntimeError("agent application is closed")
        queue: asyncio.Queue[AgentStreamEvent] = asyncio.Queue()
        event_loop = asyncio.get_running_loop()
        started = perf_counter()

        def emit(delta: str) -> None:
            event = AgentStreamEvent(
                type="answer_delta",
                delta=delta,
                elapsed_ms=(perf_counter() - started) * 1000,
            )
            event_loop.call_soon_threadsafe(queue.put_nowait, event)

        async def run() -> None:
            result = await self._runtime.handle(value, on_answer_delta=emit)
            await queue.put(
                AgentStreamEvent(
                    type="final",
                    result=result,
                    elapsed_ms=(perf_counter() - started) * 1000,
                )
            )

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                yield event
                if event.type == "final":
                    break
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._restaurant_tools.close()
        await self._database.close()
        self._closed = True


def build_restaurant_agent_application(
    project_root: str | Path,
    *,
    database_settings: DatabaseSettings | None = None,
    model_settings: AgentModelSettings | None = None,
    limits: AgentLimits | None = None,
) -> RestaurantAgentApplication:
    """从项目配置建立真实第三批 Agent；数据库必须已经完成升级。"""

    load_dotenv()
    root = Path(project_root).resolve()
    database = AgentDatabase(database_settings or DatabaseSettings.from_environment())
    store = PostgresAgentPersistence(
        database.sessions,
        content_store=LocalJsonContentStore(root / "runs" / "agent_platform" / "artifacts"),
    )
    workflow = build_recommendation_workflow(root)
    restaurant_tools = build_restaurant_tools(
        workflow=workflow,
        business_catalog=load_business_fact_catalog(),
        state_store=store,
    )
    runtime = AgentRuntime(
        model=OpenAICompatibleAgentModel(
            model_settings or AgentModelSettings.from_environment()
        ),
        session_store=store,
        result_store=store,
        tools=restaurant_tools.definitions,
        system_prompt=RESTAURANT_AGENT_PROMPT,
        limits=limits
        or AgentLimits(
            max_steps=6,
            max_tool_calls=6,
            max_total_tokens=60_000,
            timeout_seconds=420.0,
        ),
        max_model_tool_result_chars=12_000,
        large_tool_result_threshold_bytes=32 * 1024,
    )
    return RestaurantAgentApplication(
        runtime=runtime,
        database=database,
        restaurant_tools=restaurant_tools,
    )
