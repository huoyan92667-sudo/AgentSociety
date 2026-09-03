"""把真实模型、PostgreSQL 和业务工具装配成可直接调用的应用。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter
from typing import Self

from dotenv import load_dotenv

from yelp_agent.recommendation_v2.business_aspect_profiles import (
    load_business_aspect_profile_catalog,
)
from yelp_agent.recommendation_v2.business_facts import (
    load_business_fact_catalog,
)
from yelp_agent.recommendation_v2.review_evidence import (
    build_review_evidence_capabilities,
)
from yelp_agent.recommendation_v2.workflow import build_recommendation_workflow

from .domains.restaurant import RestaurantToolSet, build_restaurant_tools
from .llm import AgentModelSettings, OpenAICompatibleAgentModel
from .memory import build_conversation_memory_tool
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
2. 用户说“第一家、第三家、这家、刚才那家”时，先从历史工具结果的 presented_businesses
   读取对应 business_id。用户直接说店名而历史中没有编号时，调用 search_restaurant_businesses；
   多个同名候选无法消歧时询问用户，禁止猜商家编号。
3. 用户查询地址、评分、价格档位、停车场、营业时间或真假属性等已有字段时，
   调用 lookup_business_facts。它只回答结构化事实，不能证明“停车方便、服务好、适合约会”等评论观点。
4. 用户询问某家在 food_quality、service、price_value、quiet_environment、crowded、queue_time、
   portion_size、parking、pet_friendly、family_friendly、date_suitable、group_suitable、spiciness、
   cleanliness 中某项的总体表现时，优先调用 lookup_business_aspect_evidence。
5. 用户询问固定14项以外的长尾需求、明确索要具体评论、询问近期某类好评或差评，
   或离线特征工具提示该商家不受支持时，调用 search_business_review_evidence。
   只填写 business_ids 和“需要查证的自然语言意思”，不要自己编造关键词、英文同义句或向量。
6. 一个问题可以需要多个工具。例如“有没有停车场而且停车方便吗”应同时查结构化停车属性和
   评论中的停车便利证据；“服务总体怎样、最近有没有服务慢的差评”可先查离线服务特征，再查近期具体评论。
7. 工具选择和连续调用由你根据整句含义决定，不能靠单个关键词写死。拿到工具结果后再判断是否需要下一项能力。
8. recommend_restaurants 已经完成四路要求融合、硬过滤和评论证据排序，但不会替你写最终答复。
   你要阅读它返回的前五、满足档位和正反证据后再回答；必须保留前五顺序和商家，
   不得自行替换或重新排序，也不能把低充分程度的证据说成确定事实。
9. 工具没有给出的事实不能靠常识补充。历史 Yelp 营业时间只能表述为数据记录，不能冒充实时状态。
   评论证据不足时直接说明不足；不能把“没有召回”写成“没有人这样评价”。
10. 工具调用后阅读其真实结果，再用自然中文回答用户。相同问题不要重复调用相同工具。
11. 如果问题与餐饮无关且你能够直接回答，就直接回答；不要为了显示能力而调用餐饮工具。
12. 如果完整推荐返回零候选，只能根据 applied_filter_steps 中真正把候选降为零的条件说明原因，
   并询问用户是否愿意修改这一个条件；禁止凭空追加位置、预算、氛围等无关问题。
13. 商家停车字段必须按字面解释：parking_lot 只表示是否记录有专用停车场，不表示免费；
   parking_validated 只表示是否记录有停车验证，不等于报销；空值表示数据未知。
   即使 parking_validated=false，也禁止写成“不能凭小票减免/报销”；停车费用、免费与否、
   减免和报销只要没有独立数据就必须说未知。
14. 系统会直接提供最近几轮原始问答、当前工作记忆和少量旧话题摘要。当前工作记忆中的
   result_sets 用于理解“第一家、第三家”等指代；不要猜测位置或编号。
15. 用户追问更早、已经不在当前上下文中的事情时，调用 search_conversation_memory。
   查到的是历史对话，不代表外部事实至今未变；需要当前事实时继续调用相应查询工具。
16. 推荐前五时用不超过1200个中文字符完整说完五家，每家保留最关键的满足点和风险；
   不能因为篇幅在中途停止，也不要在结尾自行改写工具给出的固定顺序。
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

    root = Path(project_root).resolve()
    # 分支工作目录可以复用主项目里的真实数据与配置。显式指定路径，避免
    # python-dotenv 从当前源码文件向上查找时误判到另一个工作目录。
    load_dotenv(root / ".env")
    database = AgentDatabase(database_settings or DatabaseSettings.from_environment())
    store = PostgresAgentPersistence(
        database.sessions,
        content_store=LocalJsonContentStore(root / "runs" / "agent_platform" / "artifacts"),
    )
    business_catalog = load_business_fact_catalog()
    aspect_catalog = load_business_aspect_profile_catalog()
    review_capabilities = build_review_evidence_capabilities(
        profile_catalog=aspect_catalog,
        project_root=root,
    )
    workflow = build_recommendation_workflow(
        root,
        business_catalog=business_catalog,
        aspect_profiles=aspect_catalog,
        review_evidence_ranker=review_capabilities.ranker,
    )
    restaurant_tools = build_restaurant_tools(
        workflow=workflow,
        business_catalog=business_catalog,
        state_store=store,
        aspect_catalog=aspect_catalog,
        direct_review_search=review_capabilities.direct_search,
    )
    model = OpenAICompatibleAgentModel(
        model_settings or AgentModelSettings.from_environment()
    )
    runtime = AgentRuntime(
        model=model,
        session_store=store,
        result_store=store,
        tools=[
            *restaurant_tools.definitions,
            build_conversation_memory_tool(store),
        ],
        system_prompt=RESTAURANT_AGENT_PROMPT,
        memory_summary_model=model,
        limits=limits
        or AgentLimits(
            max_steps=6,
            max_tool_calls=6,
            max_total_tokens=60_000,
            timeout_seconds=420.0,
        ),
        max_model_tool_result_chars=24_000,
        large_tool_result_threshold_bytes=32 * 1024,
    )
    return RestaurantAgentApplication(
        runtime=runtime,
        database=database,
        restaurant_tools=restaurant_tools,
    )
