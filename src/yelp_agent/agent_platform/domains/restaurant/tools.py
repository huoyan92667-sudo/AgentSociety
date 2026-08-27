"""餐饮领域对主模型公开的少量高层工具。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import BusinessFactCatalog
from yelp_agent.recommendation_v2.schema import UnifiedRecommendationState
from yelp_agent.recommendation_v2.tools import (
    BusinessFactsQuery,
    BusinessFactsTool,
)
from yelp_agent.recommendation_v2.workflow import (
    RecommendationInput,
    RecommendationTurnResult,
    RecommendationWorkflow,
)

from ...persistence.schema import DomainStateWrite
from ...persistence.store import DomainStateStore
from ...runtime.schema import TokenUsage
from ...tools.definition import ToolDefinition, ToolExecutionContext
from ...tools.result import ToolBodyResult

_DOMAIN = "restaurant"


class RecommendRestaurantsArguments(StrictModel):
    """用户原问题由运行时注入；模型只负责决定是否需要重新推荐。"""


class LookupRestaurantFactsArguments(StrictModel):
    """读取一到五家已知编号餐厅的可核验基础事实。"""

    business_ids: list[str] = Field(min_length=1, max_length=5)


@dataclass(slots=True)
class RestaurantToolSet:
    """同时持有工具定义和需要在应用退出时关闭的餐饮工作流。"""

    definitions: list[ToolDefinition]
    workflow: RecommendationWorkflow

    def close(self) -> None:
        self.workflow.close()


class _RestaurantRecommendationHandler:
    """恢复数据库状态、运行完整推荐，并把新状态写回数据库。"""

    def __init__(
        self,
        workflow: RecommendationWorkflow,
        state_store: DomainStateStore,
    ) -> None:
        self._workflow = workflow
        self._state_store = state_store
        # 工作流内部包含同一会话的状态和本地模型资源，当前版本串行使用。
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        _: RecommendRestaurantsArguments,
        context: ToolExecutionContext,
    ) -> ToolBodyResult:
        query_text = context.current_user_message
        if query_text is None:
            raise ValueError("recommendation tool requires the current user message")

        async with self._lock:
            latest = await self._state_store.get_latest_domain_state(
                session_id=context.session_id,
                domain=_DOMAIN,
            )
            if latest is not None:
                restored = UnifiedRecommendationState.model_validate(latest.state)
                await asyncio.to_thread(self._workflow.restore_state, restored)

            result = await asyncio.to_thread(
                self._workflow.process,
                RecommendationInput(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    query_text=query_text,
                    request_time=context.request_time,
                ),
                on_answer_delta=_answer_delta_callback(context),
            )
            state = result.fusion.state
            if state is not None:
                await self._state_store.save_domain_state(
                    DomainStateWrite(
                        session_id=context.session_id,
                        domain=_DOMAIN,
                        state=state.model_dump(mode="json"),
                        expected_previous_version=(
                            0 if latest is None else latest.version
                        ),
                    ),
                    now=datetime.now(UTC),
                )

        compact = _compact_recommendation(result)
        context_handoff = _recommendation_context_handoff(compact)
        input_tokens, output_tokens, model_calls = _nested_model_usage(result)
        return ToolBodyResult(
            # 完整过程交给大结果保存机制；主模型只读取下面的精简内容。
            value={
                "summary": compact,
                "full_trace": result.model_dump(mode="json"),
            },
            model_content=json.dumps(
                context_handoff,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            nested_model_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            nested_model_calls=model_calls,
            terminal_answer=(
                str(compact["answer"])
                if compact["status"] == "success" and compact["answer"]
                else None
            ),
        )


def _answer_delta_callback(
    context: ToolExecutionContext,
) -> Callable[[str], None] | None:
    """取出运行时注入的流式出口；它从不进入模型参数或持久化内容。"""

    callback = context.metadata.get("answer_delta_callback")
    return callback if callable(callback) else None


class _BusinessFactsHandler:
    """把统一商家事实读取工具适配成 Agent 工具处理函数。"""

    def __init__(self, catalog: BusinessFactCatalog) -> None:
        self._tool = BusinessFactsTool(catalog)

    def __call__(
        self,
        arguments: LookupRestaurantFactsArguments,
        _: ToolExecutionContext,
    ) -> ToolBodyResult:
        result = self._tool.execute(
            BusinessFactsQuery(business_ids=arguments.business_ids)
        )
        payload = result.model_dump(mode="json")
        for business in payload.get("businesses", []):
            if isinstance(business, dict):
                business["parking_semantics"] = {
                    "availability_fields_only_describe_recorded_parking_options": True,
                    "parking_cost": "unknown",
                    "free_or_paid": "unknown",
                    "discount_or_reimbursement": "unknown",
                    "required_wording": (
                        "parking_validated 只说明数据是否记录停车验证；"
                        "不能据此判断停车费、免费、减免或报销"
                    ),
                }
        return ToolBodyResult(
            value=payload,
            model_content=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )


def build_restaurant_tools(
    *,
    workflow: RecommendationWorkflow,
    business_catalog: BusinessFactCatalog,
    state_store: DomainStateStore,
) -> RestaurantToolSet:
    """只向主模型公开完整推荐和事实查询，不暴露容易被乱序调用的内部步骤。"""

    recommendation = _RestaurantRecommendationHandler(workflow, state_store)
    facts = _BusinessFactsHandler(business_catalog)
    return RestaurantToolSet(
        workflow=workflow,
        definitions=[
            ToolDefinition(
                name="recommend_restaurants",
                description=(
                    "当用户要找餐厅、修改上次餐饮要求或要求重新推荐时使用。"
                    "工具会直接读取用户当前原话，完成长期画像、场景、历史会话和"
                    "当前要求融合，然后严格执行距离计算、硬条件过滤、评论证据排序"
                    "和前五总结。不要为单纯查询某家事实而调用它。调用参数为空。"
                ),
                input_model=RecommendRestaurantsArguments,
                handler=recommendation,
                timeout_seconds=300.0,
                read_only=False,
            ),
            ToolDefinition(
                name="lookup_business_facts",
                description=(
                    "按商家编号查询餐厅基础事实，包括名称、地址、经纬度、类别、"
                    "价格档位、评分、评论数、历史营业时间和已有真假属性。"
                    "适合回答上轮某家能否停车、几点营业、评分多少等追问；"
                    "停车场字段不代表免费，停车验证字段不代表费用报销，空值代表未知。"
                    "一次最多查询五家，不用于重新推荐。"
                ),
                input_model=LookupRestaurantFactsArguments,
                handler=facts,
                timeout_seconds=20.0,
                max_retries=1,
                read_only=True,
                concurrency_safe=True,
            ),
        ],
    )


def _compact_recommendation(result: RecommendationTurnResult) -> dict[str, object]:
    """仅保留主模型继续回答真正需要的前五、证据和失败信息。"""

    state = result.fusion.state
    ranking = result.review_evidence_ranking
    top5: list[dict[str, object]] = []
    if ranking is not None and ranking.status == "success":
        for item in ranking.ranking:
            evidence: list[dict[str, object]] = []
            for preference in item.preference_evidence:
                if preference.positive_evidence:
                    review = preference.positive_evidence[0]
                    evidence.append(
                        {
                            "requirement_id": preference.requirement_id,
                            "direction": "positive",
                            "review_id": review.review_id,
                            "review": review.review_text,
                        }
                    )
                if preference.negative_evidence:
                    review = preference.negative_evidence[0]
                    evidence.append(
                        {
                            "requirement_id": preference.requirement_id,
                            "direction": "negative",
                            "review_id": review.review_id,
                            "review": review.review_text,
                        }
                    )
            top5.append(
                {
                    "position": item.final_rank,
                    "business_id": item.business.business_id,
                    "name": item.business.name,
                    "address": item.business.address,
                    "rating": item.business.rating,
                    "review_count": item.business.review_count,
                    "price_level": item.business.price_level,
                    "straight_line_distance_km": item.distance_km,
                    "categories": item.business.categories,
                    "representative_evidence": evidence[:4],
                }
            )

    answer = result.answer
    filter_steps = (
        []
        if result.hard_filter is None
        else [
            {
                "field": step.field,
                "value": step.value,
                "source": step.source,
                "before_count": step.before_count,
                "after_count": step.after_count,
                "excluded_count": step.excluded_count,
                "unknown_excluded_count": step.unknown_excluded_count,
            }
            for step in result.hard_filter.steps
        ]
    )
    failure_reasons = [
        value
        for value in (
            result.fusion.failure_reason,
            None if ranking is None else ranking.failure_reason,
            None if answer is None else answer.failure_reason,
        )
        if value
    ]
    if (
        result.hard_filter is not None
        and result.hard_filter.candidate_count == 0
        and filter_steps
    ):
        blocking = next(
            (
                step
                for step in reversed(filter_steps)
                if step["after_count"] == 0 and step["before_count"] > 0
            ),
            filter_steps[-1],
        )
        failure_reasons.append(
            "没有餐厅满足全部过滤条件；最后清空候选的是"
            f"{blocking['field']}={blocking['value']}（来源：{blocking['source']}）"
        )
    return {
        "status": (
            "success"
            if answer is not None and answer.status == "success" and top5
            else "incomplete"
        ),
        "state_revision": None if state is None else state.revision,
        "hard_filtered_count": (
            None if result.hard_filter is None else result.hard_filter.candidate_count
        ),
        "applied_filter_steps": filter_steps,
        "answer": None if answer is None else answer.text,
        "top5": top5,
        "performance": {
            "workflow": (
                None
                if result.timing is None
                else result.timing.model_dump(mode="json")
            ),
            "review_retrieval": (
                None
                if ranking is None or ranking.retrieval_metrics is None
                else ranking.retrieval_metrics.model_dump(mode="json")
            ),
            "review_description_ms": (
                None if ranking is None else ranking.description_latency_ms
            ),
            "review_scoring_ms": (
                None if ranking is None else ranking.scoring_latency_ms
            ),
        },
        "failure_reasons": failure_reasons,
    }


def _nested_model_usage(
    result: RecommendationTurnResult,
) -> tuple[int, int, int]:
    """汇总完整推荐内部的需求理解、长尾描述和最终总结模型调用。"""

    calls: list[tuple[int | None, int | None, int]] = [
        (
            result.fusion.input_tokens,
            result.fusion.output_tokens,
            result.fusion.model_call_count,
        )
    ]
    if result.review_evidence_ranking is not None:
        ranking = result.review_evidence_ranking
        calls.append(
            (ranking.input_tokens, ranking.output_tokens, ranking.model_call_count)
        )
    if result.answer is not None and result.answer.model is not None:
        calls.append((result.answer.input_tokens, result.answer.output_tokens, 1))
    return (
        sum(value or 0 for value, _, _ in calls),
        sum(value or 0 for _, value, _ in calls),
        sum(count for _, _, count in calls),
    )


def _recommendation_context_handoff(compact: dict[str, object]) -> dict[str, object]:
    """下一轮只记商家位置和编号；完整证据、回答由结果文件与助手消息保存。"""

    top5 = compact.get("top5")
    businesses = []
    if isinstance(top5, list):
        for item in top5:
            if not isinstance(item, dict):
                continue
            businesses.append(
                {
                    "position": item.get("position"),
                    "business_id": item.get("business_id"),
                    "name": item.get("name"),
                }
            )
    return {
        "status": compact.get("status"),
        "state_revision": compact.get("state_revision"),
        "presented_businesses": businesses,
        "failure_reasons": compact.get("failure_reasons", []),
    }
