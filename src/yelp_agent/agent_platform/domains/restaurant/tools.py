"""餐饮领域对主模型公开的少量高层工具。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_aspect_profiles import (
    BusinessAspectProfileCatalog,
)
from yelp_agent.recommendation_v2.business_facts import BusinessFactCatalog
from yelp_agent.recommendation_v2.review_evidence import DirectReviewEvidenceSearch
from yelp_agent.recommendation_v2.review_evidence.schema import (
    BusinessPreferenceEvidence,
    PreferenceRankingLayer,
    RankedReviewEvidence,
)
from yelp_agent.recommendation_v2.schema import AspectField, UnifiedRecommendationState
from yelp_agent.recommendation_v2.tools import (
    BusinessAspectEvidenceQuery,
    BusinessAspectEvidenceTool,
    BusinessFactsQuery,
    BusinessFactsTool,
    BusinessNameSearchQuery,
    BusinessNameSearchTool,
)
from yelp_agent.recommendation_v2.workflow import (
    RecommendationInput,
    RecommendationTurnResult,
    RecommendationWorkflow,
)

from ...memory.models import (
    DomainStateReference,
    EntityReference,
    RankedEntityReference,
    ResultSetReference,
    ToolMemoryUpdate,
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


class SearchRestaurantBusinessesArguments(StrictModel):
    """主模型从用户原话提取店名；工具只负责寻找真实商家编号。"""

    name: str = Field(min_length=1, max_length=200)
    city: str | None = Field(default=None, min_length=1, max_length=100)
    state: str | None = Field(default=None, min_length=1, max_length=100)


class LookupRestaurantAspectEvidenceArguments(StrictModel):
    """读取500家离线计算好的固定14项特征与代表性评论。"""

    business_ids: list[str] = Field(min_length=1, max_length=5)
    aspect_ids: list[AspectField] = Field(min_length=1, max_length=5)
    evidence_limit_per_group: int = Field(default=2, ge=1, le=3)

    @field_validator("business_ids", "aspect_ids")
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("business IDs and aspect IDs must be unique")
        return values


class SearchRestaurantReviewEvidenceArguments(StrictModel):
    """主模型给出商家编号和要查证的意思，不需要生成检索词或向量。"""

    business_ids: list[str] = Field(min_length=1, max_length=5)
    evidence_queries: list[str] = Field(min_length=1, max_length=3)
    evidence_limit_per_direction: int = Field(default=3, ge=1, le=5)

    @field_validator("business_ids", "evidence_queries")
    @classmethod
    def clean_unique_values(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(item.split()) for item in values]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("review search values must be nonempty and unique")
        return cleaned


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

        saved_state = None
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
                    # 工作流负责产生排序和证据，最终措辞交给外层主模型。
                    synthesize_answer=False,
                ),
                on_answer_delta=_answer_delta_callback(context),
            )
            state = result.fusion.state
            if state is not None:
                saved_state = await self._state_store.save_domain_state(
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
        result_set = _recommendation_result_set(compact, context.turn_id)
        input_tokens, output_tokens, model_calls = _nested_model_usage(result)
        return ToolBodyResult(
            # 正式工具结果在数据库和主模型上下文中完全一致。
            value=compact,
            model_content=json.dumps(
                compact,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            nested_model_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            nested_model_calls=model_calls,
            memory_update=ToolMemoryUpdate(
                result_sets=[] if result_set is None else [result_set],
                domain_state_refs=(
                    []
                    if saved_state is None
                    else [
                        DomainStateReference(
                            domain=_DOMAIN,
                            revision=saved_state.version,
                        )
                    ]
                ),
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
        self._catalog = catalog
        self._tool = BusinessFactsTool(catalog)

    def __call__(
        self,
        arguments: LookupRestaurantFactsArguments,
        context: ToolExecutionContext,
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
            memory_update=ToolMemoryUpdate(
                focused_entities=[
                    _entity_for_business(
                        self._catalog,
                        str(item["business_id"]),
                        source_turn_id=context.turn_id,
                    )
                    for item in payload.get("businesses", [])
                    if isinstance(item, dict) and item.get("business_id")
                ]
            ),
        )


class _BusinessSearchHandler:
    """把名称搜索适配成主模型可调用的只读工具。"""

    def __init__(self, catalog: BusinessFactCatalog) -> None:
        self._catalog = catalog
        self._tool = BusinessNameSearchTool(catalog)

    def __call__(
        self,
        arguments: SearchRestaurantBusinessesArguments,
        context: ToolExecutionContext,
    ) -> ToolBodyResult:
        result = self._tool.execute(
            BusinessNameSearchQuery(
                name=arguments.name,
                city=arguments.city,
                state=arguments.state,
            )
        )
        payload = result.model_dump(mode="json")
        return ToolBodyResult(
            value=payload,
            model_content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            memory_update=ToolMemoryUpdate(
                focused_entities=[
                    _entity_for_business(
                        self._catalog,
                        str(item["business_id"]),
                        source_turn_id=context.turn_id,
                    )
                    for item in payload.get("matches", [])
                    if isinstance(item, dict) and item.get("business_id")
                ]
            ),
        )


class _BusinessAspectEvidenceHandler:
    """读取离线特征；不支持的商家会明确提示主模型改用实时评论工具。"""

    def __init__(
        self,
        catalog: BusinessAspectProfileCatalog,
        business_catalog: BusinessFactCatalog,
    ) -> None:
        self._tool = BusinessAspectEvidenceTool(catalog)
        self._business_catalog = business_catalog

    def __call__(
        self,
        arguments: LookupRestaurantAspectEvidenceArguments,
        context: ToolExecutionContext,
    ) -> ToolBodyResult:
        result = self._tool.execute(
            BusinessAspectEvidenceQuery(
                business_ids=arguments.business_ids,
                aspect_ids=arguments.aspect_ids,
                evidence_limit_per_group=arguments.evidence_limit_per_group,
            )
        )
        payload = _compact_aspect_evidence(result.model_dump(mode="json"))
        return ToolBodyResult(
            value=payload,
            model_content=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            memory_update=ToolMemoryUpdate(
                focused_entities=[
                    _entity_for_business(
                        self._business_catalog,
                        str(item["business_id"]),
                        source_turn_id=context.turn_id,
                    )
                    for item in payload.get("assessments", [])
                    if isinstance(item, dict) and item.get("business_id")
                ]
            ),
        )


class _DirectReviewEvidenceHandler:
    """执行实时评论检索；用户意图和商家范围都由主模型提前给定。"""

    def __init__(
        self,
        search: DirectReviewEvidenceSearch,
        business_catalog: BusinessFactCatalog,
    ) -> None:
        self._search = search
        self._business_catalog = business_catalog
        # 本地向量编码器和向量读取资源按一次调用使用，避免并发互相抢占。
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        arguments: SearchRestaurantReviewEvidenceArguments,
        context: ToolExecutionContext,
    ) -> ToolBodyResult:
        known_ids = [
            item for item in arguments.business_ids if self._business_catalog.contains(item)
        ]
        missing_ids = [
            item for item in arguments.business_ids if item not in known_ids
        ]
        if not known_ids:
            payload = {
                "status": "not_found",
                "missing_business_ids": missing_ids,
                "instruction": "先调用 search_restaurant_businesses 获取真实商家编号",
            }
            return ToolBodyResult(
                value=payload,
                model_content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )

        async with self._lock:
            result = await asyncio.to_thread(
                self._search.search,
                business_ids=known_ids,
                evidence_queries=arguments.evidence_queries,
                user_query_text=context.current_user_message
                or "；".join(arguments.evidence_queries),
                reference_time=context.request_time,
            )
        raw_payload = result.model_dump(mode="json")
        raw_payload["missing_business_ids"] = missing_ids
        payload = _compact_direct_review_evidence(
            raw_payload,
            self._business_catalog,
            limit=arguments.evidence_limit_per_direction,
        )
        return ToolBodyResult(
            value=payload,
            model_content=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            nested_model_usage=TokenUsage(
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
            ),
            nested_model_calls=result.model_call_count,
            memory_update=ToolMemoryUpdate(
                focused_entities=[
                    _entity_for_business(
                        self._business_catalog,
                        business_id,
                        source_turn_id=context.turn_id,
                    )
                    for business_id in known_ids
                ]
            ),
        )


def build_restaurant_tools(
    *,
    workflow: RecommendationWorkflow,
    business_catalog: BusinessFactCatalog,
    state_store: DomainStateStore,
    aspect_catalog: BusinessAspectProfileCatalog | None = None,
    direct_review_search: DirectReviewEvidenceSearch | None = None,
) -> RestaurantToolSet:
    """公开可由主模型自由组合的餐饮能力，内部检索步骤仍保持封装。"""

    recommendation = _RestaurantRecommendationHandler(workflow, state_store)
    facts = _BusinessFactsHandler(business_catalog)
    definitions = [
        ToolDefinition(
            name="recommend_restaurants",
            description=(
                "当用户要找餐厅、修改上次餐饮要求或要求重新推荐时使用。"
                "工具会直接读取用户当前原话，完成长期画像、场景、历史会话和"
                "当前要求融合，然后严格执行距离计算、硬条件过滤、评论证据排序"
                "和前五总结。不要为单纯查询某家事实或评论而调用它。调用参数为空。"
            ),
            input_model=RecommendRestaurantsArguments,
            handler=recommendation,
            timeout_seconds=300.0,
            read_only=False,
        ),
        ToolDefinition(
            name="search_restaurant_businesses",
            description=(
                "用户直接说出餐厅名称、但当前对话里没有其 business_id 时使用。"
                "按名称以及可选城市、州查找真实商家编号；如果返回多个候选，"
                "结合对话消歧，仍无法确定就询问用户。已有上轮商家编号时不要调用。"
            ),
            input_model=SearchRestaurantBusinessesArguments,
            handler=_BusinessSearchHandler(business_catalog),
            timeout_seconds=20.0,
            max_retries=1,
            read_only=True,
            concurrency_safe=True,
        ),
        ToolDefinition(
            name="lookup_business_facts",
            description=(
                "按商家编号查询餐厅基础事实，包括名称、地址、经纬度、类别、"
                "价格档位、评分、评论数、历史营业时间和已有真假属性。"
                "只回答数据字段能够直接确认的事实；停车场字段不代表免费，"
                "停车验证字段不代表费用报销，空值代表未知。一次最多查询五家。"
            ),
            input_model=LookupRestaurantFactsArguments,
            handler=facts,
            timeout_seconds=20.0,
            max_retries=1,
            read_only=True,
            concurrency_safe=True,
        ),
    ]
    if aspect_catalog is not None:
        definitions.append(
            ToolDefinition(
                name="lookup_business_aspect_evidence",
                description=(
                    "按商家编号读取已经离线判断好的14种评论特征、证据充分程度、"
                    "争议程度和代表性原评论。可选特征：food_quality、service、"
                    "price_value、quiet_environment、crowded、queue_time、portion_size、"
                    "parking、pet_friendly、family_friendly、date_suitable、"
                    "group_suitable、spiciness、cleanliness。用户问这些方面的总体表现"
                    "时优先使用；若商家不在离线支持范围，改用实时评论检索。"
                ),
                input_model=LookupRestaurantAspectEvidenceArguments,
                handler=_BusinessAspectEvidenceHandler(
                    aspect_catalog,
                    business_catalog,
                ),
                timeout_seconds=20.0,
                max_retries=1,
                read_only=True,
                concurrency_safe=True,
            )
        )
    if direct_review_search is not None:
        definitions.append(
            ToolDefinition(
                name="search_business_review_evidence",
                description=(
                    "在一到五个已知 business_id 的真实评论中查找与任意问题相关的"
                    "正反和条件性证据。用于固定14项以外的长尾需求、用户明确索要"
                    "具体评论、询问近期某类差评，或离线特征不支持该商家时。"
                    "主模型只填写要查证的自然语言意思，不要生成关键词、同义句或向量。"
                ),
                input_model=SearchRestaurantReviewEvidenceArguments,
                handler=_DirectReviewEvidenceHandler(
                    direct_review_search,
                    business_catalog,
                ),
                timeout_seconds=180.0,
                read_only=True,
                concurrency_safe=False,
            )
        )
    return RestaurantToolSet(
        workflow=workflow,
        definitions=definitions,
    )


def _compact_aspect_evidence(payload: dict[str, object]) -> dict[str, object]:
    """给主模型保留档位、充分程度、争议和少量原评论，不发送整库字段。"""

    compact: list[dict[str, object]] = []
    raw_assessments = payload.get("assessments", [])
    if isinstance(raw_assessments, list):
        for item in raw_assessments:
            if not isinstance(item, dict):
                continue
            score = item.get("score") if isinstance(item.get("score"), dict) else {}
            direction = (
                item.get("direction")
                if isinstance(item.get("direction"), dict)
                else {}
            )
            compact.append(
                {
                    "business_id": item.get("business_id"),
                    "aspect_id": item.get("aspect_id"),
                    "lower_value_means": direction.get("lower_value_means"),
                    "higher_value_means": direction.get("higher_value_means"),
                    "degree_level": score.get("degree_level_name_zh"),
                    "degree_level_meaning": score.get("degree_level_meaning"),
                    "evidence_sufficiency": score.get("evidence_sufficiency_level"),
                    "controversy": score.get("controversy_level"),
                    "usable_for_ranking": score.get("usable_for_ranking"),
                    "unusable_reasons": score.get("unusable_reasons", []),
                    "review_counts": {
                        "business_total": score.get("business_total_review_count"),
                        "retrieved_candidates": score.get("retrieved_candidate_count"),
                        "model_related": score.get("model_related_review_count"),
                        "unique_evidence_users": score.get("unique_evidence_user_count"),
                    },
                    "high_degree_evidence": _compact_review_quotes(
                        item.get("high_degree_evidence")
                    ),
                    "low_degree_evidence": _compact_review_quotes(
                        item.get("low_degree_evidence")
                    ),
                    "middle_degree_evidence": _compact_review_quotes(
                        item.get("middle_degree_evidence"),
                        limit=1,
                    ),
                }
            )
    return {
        "status": payload.get("status"),
        "assessments": compact,
        "unsupported_business_ids": payload.get("unsupported_business_ids", []),
        "instruction": (
            "unsupported_business_ids 中的商家没有离线画像；若仍需评论结论，"
            "调用 search_business_review_evidence"
        ),
    }


def _compact_direct_review_evidence(
    payload: dict[str, object],
    business_catalog: BusinessFactCatalog,
    *,
    limit: int,
) -> dict[str, object]:
    """实时检索只把结论所需的正反原文交给主模型，完整轨迹另行保存。"""

    compact: list[dict[str, object]] = []
    raw_findings = payload.get("findings", [])
    if isinstance(raw_findings, list):
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            assessment = (
                item.get("assessment")
                if isinstance(item.get("assessment"), dict)
                else {}
            )
            business_id = str(item.get("business_id") or "")
            compact.append(
                {
                    "business_id": business_id,
                    "business_name": (
                        business_catalog.get(business_id).name
                        if business_catalog.contains(business_id)
                        else None
                    ),
                    "question": item.get("requirement_text"),
                    "evidence_balance": {
                        "score": assessment.get("evidence_score"),
                        "recalled_review_count": assessment.get(
                            "recalled_review_count"
                        ),
                        "ambiguous_review_count": assessment.get(
                            "ambiguous_review_count"
                        ),
                        "positive_count_reliability": assessment.get(
                            "positive_count_reliability"
                        ),
                        "negative_count_reliability": assessment.get(
                            "negative_count_reliability"
                        ),
                    },
                    "supporting_evidence": _compact_review_quotes(
                        assessment.get("positive_evidence"),
                        limit=limit,
                    ),
                    "contradicting_evidence": _compact_review_quotes(
                        assessment.get("negative_evidence"),
                        limit=limit,
                    ),
                }
            )
    return {
        "status": payload.get("status"),
        "findings": compact,
        "missing_business_ids": payload.get("missing_business_ids", []),
        "failure_reason": payload.get("failure_reason"),
        "instruction": (
            "只能依据返回的真实评论作答；没有证据或正反接近时必须明确说明"
            "证据不足，不能把检索不到写成用户需求不成立"
        ),
    }


def _entity_for_business(
    catalog: BusinessFactCatalog,
    business_id: str,
    *,
    source_turn_id: str,
) -> EntityReference:
    """把真实商家编号和名称写入通用工作记忆，供“第三家”之类追问使用。"""

    business = catalog.get(business_id)
    return EntityReference(
        entity_type="restaurant",
        entity_id=business.business_id,
        display_name=business.name,
        source_turn_id=source_turn_id,
    )


def _recommendation_result_set(
    compact: dict[str, object],
    source_turn_id: str,
) -> ResultSetReference | None:
    """最多记住本轮展示的十家及其顺序，不保存几百个内部候选。"""

    raw_items = compact.get("top5")
    if not isinstance(raw_items, list):
        return None
    items = [
        RankedEntityReference(
            entity_type="restaurant",
            entity_id=str(item["business_id"]),
            display_name=str(item["name"]),
            position=int(item["position"]),
            source_turn_id=source_turn_id,
        )
        for item in raw_items[:10]
        if isinstance(item, dict)
        and item.get("business_id")
        and item.get("name")
        and item.get("position")
    ]
    if not items:
        return None
    return ResultSetReference(
        result_type="restaurant_recommendation",
        items=items,
        source_turn_id=source_turn_id,
    )


def _compact_review_quotes(value: object, *, limit: int = 2) -> list[dict[str, object]]:
    """统一离线和实时评论字段，方便主模型引用同一种证据格式。"""

    if not isinstance(value, list):
        return []
    result: list[dict[str, object]] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "review_id": item.get("review_id"),
                "review_time": item.get("review_time"),
                "stars": item.get("stars"),
                "useful": item.get("useful"),
                "review": item.get("text", item.get("review_text")),
                "matched_part": item.get("matched_segment_text"),
                "model_relevance": item.get("relevance", item.get("model_relevance")),
                "model_strength": item.get("strength", item.get("model_strength")),
            }
        )
    return result


def _compact_recommendation(result: RecommendationTurnResult) -> dict[str, object]:
    """仅保留主模型继续回答真正需要的前五、证据和失败信息。"""

    state = result.fusion.state
    ranking = result.review_evidence_ranking
    top5: list[dict[str, object]] = []
    if ranking is not None and ranking.status == "success":
        for item in ranking.ranking:
            positive_evidence: list[dict[str, object]] = []
            negative_evidence: list[dict[str, object]] = []
            for preference in item.preference_evidence:
                if preference.positive_evidence:
                    review = preference.positive_evidence[0]
                    positive_evidence.append(
                        _recommendation_review(
                            preference.requirement_id,
                            "positive",
                            review,
                        )
                    )
                if preference.negative_evidence:
                    review = preference.negative_evidence[0]
                    negative_evidence.append(
                        _recommendation_review(
                            preference.requirement_id,
                            "negative",
                            review,
                        )
                    )
            evidence_by_requirement = {
                preference.requirement_id: preference
                for preference in item.preference_evidence
            }
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
                    # 满足档位和证据质量属于同一项要求，放在一起可避免把
                    # requirement_id 等内容重复发送两遍。
                    "preferences": [
                        _recommendation_preference(
                            layer,
                            evidence_by_requirement.get(layer.requirement_id),
                        )
                        for layer in item.priority_layers[:5]
                    ],
                    # 正式输出保留两条正面和一条反面证据；主模型据此组织回答。
                    "representative_evidence": [
                        *positive_evidence[:2],
                        *negative_evidence[:1],
                    ],
                }
            )

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
            if ranking is not None and ranking.status == "success" and top5
            else "incomplete"
        ),
        "state_revision": None if state is None else state.revision,
        "hard_filtered_count": (
            None if result.hard_filter is None else result.hard_filter.candidate_count
        ),
        "applied_filter_steps": filter_steps,
        "top5": top5,
        "failure_reasons": failure_reasons,
    }


def _recommendation_preference(
    layer: PreferenceRankingLayer,
    evidence: BusinessPreferenceEvidence | None,
) -> dict[str, object]:
    """把同一要求的排序档位和证据质量合成一条正式输出。"""

    return {
        "priority": layer.priority,
        "requirement_id": layer.requirement_id,
        "requirement": layer.requirement_text,
        "level": layer.satisfaction_level,
        "evidence_source": getattr(evidence, "evidence_source", None),
        "evidence_sufficiency": getattr(
            evidence,
            "evidence_sufficiency_level",
            None,
        ),
        "controversy": getattr(evidence, "controversy_level", None),
        "usable_for_ranking": getattr(evidence, "usable_for_ranking", None),
    }


def _recommendation_review(
    requirement_id: str,
    direction: str,
    review: RankedReviewEvidence,
) -> dict[str, object]:
    """推荐工具返回有上下文的评论摘录，并明确标记是否截短。"""

    text = review.review_text
    max_chars = 450
    matched = review.matched_segment_text
    return {
        "requirement_id": requirement_id,
        "direction": direction,
        "review_id": review.review_id,
        "review_time": review.review_time.isoformat(),
        "stars": review.stars,
        "useful": review.useful,
        # 长评论从命中句附近截取，而不是永远只取开头；这样更短也不会
        # 丢掉真正支持当前偏好的上下文。
        "review_excerpt": _centered_review_excerpt(text, matched, max_chars),
        "review_excerpt_truncated": len(text) > max_chars,
    }


def _centered_review_excerpt(text: str, matched: str, max_chars: int) -> str:
    """优先保留命中句前后的原文；找不到命中句时才退回评论开头。"""

    if len(text) <= max_chars:
        return text
    position = text.find(matched) if matched else -1
    if position < 0:
        return text[:max_chars]
    start = max(0, position - max_chars // 3)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    return text[start:end]


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
