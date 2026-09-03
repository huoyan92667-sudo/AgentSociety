"""隐藏画像读取、画像转换、场景加载和多轮状态的完整推荐入口。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, Self

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.profiles.schema import UserProfileV1
from yelp_agent.profiles.store import UserProfileStore
from yelp_agent.recommendation_v2.answer_synthesis import (
    RecommendationAnswer,
    RecommendationAnswerSynthesizer,
    build_recommendation_answer_synthesizer,
)
from yelp_agent.recommendation_v2.business_aspect_profiles import (
    BusinessAspectProfileCatalog,
    load_business_aspect_profile_catalog,
)
from yelp_agent.recommendation_v2.business_facts import (
    BusinessFactCatalog,
    catalog_local_time,
    load_business_fact_catalog,
)
from yelp_agent.recommendation_v2.category_catalog import load_fixed_category_catalog
from yelp_agent.recommendation_v2.preference_fusion import (
    ConversationHistoryTurn,
    PreferenceFusion,
    PreferenceFusionAttempt,
    PreferenceFusionRequest,
    ProfilePreferenceSet,
    RecommendationSnapshot,
    build_preference_fusion,
)
from yelp_agent.recommendation_v2.review_evidence import (
    ReviewEvidenceRanker,
    ReviewEvidenceRankingResult,
    build_review_evidence_ranker,
)
from yelp_agent.recommendation_v2.schema import (
    AspectField,
    BusinessReference,
    DefaultConstraint,
    GeoPoint,
    RequirementBasis,
    UnifiedRecommendationState,
    merchant_feature_for,
    requirement_unit_for,
)
from yelp_agent.recommendation_v2.soft_ranking import (
    PriorityLayeredRanker,
    SoftRankingAttempt,
)
from yelp_agent.recommendation_v2.tools import (
    GeographicDistanceResult,
    GeographicDistanceTool,
    RatingBaselineRankingTool,
    StructuredHardFilterResult,
    StructuredHardFilterTool,
    UserProfileTool,
)


class ManagedUserProfileReader(Protocol):
    """推荐入口需要的画像读取和资源关闭能力。"""

    def latest(self, user_id: str) -> UserProfileV1:
        """读取最新画像。"""

    def close(self) -> None:
        """关闭底层资源。"""


class RecommendationInput(StrictModel):
    """调用方真正需要提供的全部内容。"""

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1, max_length=4000)
    request_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 独立工作流可直接生成答案；嵌入 Agent 时由外层主模型读取排序证据后回答。
    synthesize_answer: bool = True


class RecommendationWorkflowTiming(StrictModel):
    """把一轮推荐拆到足以定位性能问题的主要阶段。"""

    profile_load_ms: float = Field(default=0, ge=0)
    fusion_ms: float = Field(default=0, ge=0)
    geography_ms: float = Field(default=0, ge=0)
    hard_filter_ms: float = Field(default=0, ge=0)
    review_ranking_ms: float = Field(default=0, ge=0)
    answer_synthesis_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)


class RecommendationTurnResult(StrictModel):
    """保留真实画像、转换结果和最终状态，方便验证完整链路。"""

    raw_profile: UserProfileV1 | None = None
    adapted_profile: ProfilePreferenceSet | None = None
    fusion: PreferenceFusionAttempt
    geography: GeographicDistanceResult | None = None
    hard_filter: StructuredHardFilterResult | None = None
    soft_ranking: SoftRankingAttempt | None = None
    review_evidence_ranking: ReviewEvidenceRankingResult | None = None
    answer: RecommendationAnswer | None = None
    timing: RecommendationWorkflowTiming | None = None


class RecommendationWorkflow:
    """用一个小入口完成画像读取、场景接入和多轮需求融合。"""

    def __init__(
        self,
        *,
        fusion: PreferenceFusion,
        profile_store: ManagedUserProfileReader,
        geography_tool: GeographicDistanceTool | None = None,
        hard_filter_tool: StructuredHardFilterTool | None = None,
        baseline_ranking_tool: RatingBaselineRankingTool | None = None,
        soft_ranker: PriorityLayeredRanker | None = None,
        review_evidence_ranker: ReviewEvidenceRanker | None = None,
        answer_synthesizer: RecommendationAnswerSynthesizer | None = None,
    ) -> None:
        self._fusion = fusion
        self._profile_store = profile_store
        self._profile_tool = UserProfileTool(profile_store)
        self._geography_tool = geography_tool
        self._hard_filter_tool = hard_filter_tool
        self._baseline_ranking_tool = baseline_ranking_tool
        self._soft_ranker = soft_ranker
        self._review_evidence_ranker = review_evidence_ranker
        self._answer_synthesizer = answer_synthesizer
        self._states: dict[tuple[str, str], UnifiedRecommendationState] = {}
        self._history: dict[tuple[str, str], list[ConversationHistoryTurn]] = {}
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """关闭真实画像存储；模型客户端不持有需要关闭的本地资源。"""

        if not self._closed:
            if self._soft_ranker is not None:
                self._soft_ranker.close()
            if self._review_evidence_ranker is not None:
                self._review_evidence_ranker.close()
            self._profile_store.close()
            self._closed = True

    def restore_state(self, state: UnifiedRecommendationState) -> None:
        """把数据库中的最新完整状态放回工作流，支持进程重启后继续多轮对话。"""

        if self._closed:
            raise RuntimeError("recommendation workflow is closed")
        key = (state.user_id, state.session_id)
        current = self._states.get(key)
        if current is None or current.revision <= state.revision:
            self._states[key] = state.model_copy(deep=True)

    def process(
        self,
        request: RecommendationInput,
        *,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> RecommendationTurnResult:
        """处理一轮问题；调用方不需要知道内部四路数据怎么准备。"""

        if self._closed:
            raise RuntimeError("recommendation workflow is closed")
        total_started = perf_counter()
        profile_started = perf_counter()
        key = (request.user_id, request.session_id)
        previous = self._states.get(key)
        raw_profile: UserProfileV1 | None = None
        adapted_profile: ProfilePreferenceSet | None = None
        user_location = None
        if previous is None:
            raw_profile, adapted_profile = self._profile_tool.load(request.user_id)
            user_location = self._profile_tool.location(raw_profile)
        profile_load_ms = (perf_counter() - profile_started) * 1000

        turn_index = 1 if previous is None else previous.turn_index + 1
        fusion_started = perf_counter()
        attempt = self._fusion.fuse(
            PreferenceFusionRequest(
                user_id=request.user_id,
                session_id=request.session_id,
                turn_index=turn_index,
                query_text=request.query_text,
                request_time=request.request_time,
                previous_state=previous,
                conversation_history=list(self._history.get(key, [])),
                # 场景基准不由调用方传；大模型识别场景后，融合模块自动读取。
                profile_preferences=adapted_profile,
                user_location=user_location,
            )
        )
        fusion_ms = (perf_counter() - fusion_started) * 1000
        geography: GeographicDistanceResult | None = None
        hard_filter: StructuredHardFilterResult | None = None
        soft_ranking: SoftRankingAttempt | None = None
        review_evidence_ranking: ReviewEvidenceRankingResult | None = None
        answer: RecommendationAnswer | None = None
        geography_ms = 0.0
        hard_filter_ms = 0.0
        review_ranking_ms = 0.0
        answer_synthesis_ms = 0.0
        if attempt.state is not None:
            state = _ensure_open_time_constraint(
                attempt.state,
                request.request_time,
            )
            attempt = attempt.model_copy(update={"state": state}, deep=True)
            # 先用统一搜索中心计算距离，再把距离和商家事实交给硬过滤。
            if (
                self._geography_tool is not None
                and attempt.state.search_center is not None
            ):
                geography_started = perf_counter()
                geography = self._geography_tool.execute(attempt.state.search_center)
                geography_ms = (perf_counter() - geography_started) * 1000
            if self._hard_filter_tool is not None:
                hard_filter_started = perf_counter()
                hard_filter = self._hard_filter_tool.execute(
                    attempt.state,
                    geography=geography,
                )
                hard_filter_ms = (perf_counter() - hard_filter_started) * 1000
            if hard_filter is not None and self._review_evidence_ranker is not None:
                review_started = perf_counter()
                review_evidence_ranking = self._review_evidence_ranker.rank(
                    state=attempt.state,
                    hard_filter=hard_filter,
                    # 真实推荐只排除当前请求时间之后的评论。用户画像的
                    # 生成时间只描述画像本身，不再限制可检索评论范围。
                    reference_time=request.request_time,
                    prepared_descriptions=attempt.review_search_descriptions,
                )
                review_ranking_ms = (perf_counter() - review_started) * 1000
                if (
                    review_evidence_ranking.status == "success"
                    and review_evidence_ranking.ranking
                    and self._answer_synthesizer is not None
                    and request.synthesize_answer
                ):
                    answer_started = perf_counter()
                    answer = self._answer_synthesizer.synthesize(
                        query_text=request.query_text,
                        state=attempt.state,
                        ranking=review_evidence_ranking,
                        on_delta=on_answer_delta,
                    )
                    answer_synthesis_ms = (perf_counter() - answer_started) * 1000
            elif (
                hard_filter is not None
                and self._baseline_ranking_tool is not None
                and self._soft_ranker is not None
            ):
                # 先按评分从硬筛结果中取前十，再只检索这十家的评论证据。
                baseline = self._baseline_ranking_tool.execute(
                    hard_filter=hard_filter,
                )
                soft_ranking = self._soft_ranker.rank(
                    preferences=attempt.state.soft_preferences,
                    hard_filter=hard_filter,
                    baseline=baseline,
                )
            presented = _presented_businesses(
                turn_index,
                review_evidence_ranking,
            )
            if presented:
                previous_references = {
                    (item.presented_turn_index, item.position): item
                    for item in attempt.state.referenced_businesses
                }
                for item in presented:
                    previous_references[(item.presented_turn_index, item.position)] = (
                        item
                    )
                state = attempt.state.model_copy(
                    update={
                        "referenced_businesses": [
                            previous_references[item]
                            for item in sorted(previous_references)
                        ]
                    },
                    deep=True,
                )
                attempt = attempt.model_copy(update={"state": state}, deep=True)
            self._states[key] = attempt.state
            snapshot = (
                RecommendationSnapshot(
                    state_revision=attempt.state.revision,
                    ordered_business_ids=[item.business_id for item in presented],
                    evidence_review_ids_by_business=(
                        _ranking_review_ids(review_evidence_ranking)
                    ),
                )
                if presented
                else None
            )
            self._history.setdefault(key, []).append(
                ConversationHistoryTurn(
                    turn_index=turn_index,
                    user_message=request.query_text,
                    assistant_message=None if answer is None else answer.text,
                    presented_businesses=presented,
                    recommendation_snapshot=snapshot,
                )
            )
        return RecommendationTurnResult(
            raw_profile=raw_profile,
            adapted_profile=adapted_profile,
            fusion=attempt,
            geography=geography,
            hard_filter=hard_filter,
            soft_ranking=soft_ranking,
            review_evidence_ranking=review_evidence_ranking,
            answer=answer,
            timing=RecommendationWorkflowTiming(
                profile_load_ms=profile_load_ms,
                fusion_ms=fusion_ms,
                geography_ms=geography_ms,
                hard_filter_ms=hard_filter_ms,
                review_ranking_ms=review_ranking_ms,
                answer_synthesis_ms=answer_synthesis_ms,
                total_ms=(perf_counter() - total_started) * 1000,
            ),
        )


def _ensure_open_time_constraint(
    state: UnifiedRecommendationState,
    request_time: datetime,
) -> UnifiedRecommendationState:
    """用户没说到店时间时，用本轮请求时刻补一条可覆盖的营业默认值。"""

    defaults = [item for item in state.default_constraints if item.field != "open_at"]
    if not any(item.field == "open_at" for item in state.hard_constraints):
        local = catalog_local_time(request_time)
        defaults.append(
            DefaultConstraint(
                key="default.open_at.request_time",
                field="open_at",
                operator="equals",
                value=local.isoformat(timespec="minutes"),
                unit=requirement_unit_for("open_at"),
                merchant_feature=merchant_feature_for("open_at"),
                controlling_source="system_default",
                sources=[
                    RequirementBasis(
                        source="system_default",
                        text="用户未指定到店时间，按本轮请求时刻筛选营业商家",
                    )
                ],
            )
        )
    return state.model_copy(
        update={"default_constraints": defaults},
        deep=True,
    )


def _presented_businesses(
    turn_index: int,
    ranking: ReviewEvidenceRankingResult | None,
) -> list[BusinessReference]:
    """把本轮真正展示的 Top5 写成下一轮可查询的紧凑商家快照。"""

    if ranking is None or ranking.status != "success":
        return []
    result: list[BusinessReference] = []
    for item in ranking.ranking:
        aspect_scores: dict[AspectField, float] = {}
        requirement_fields = {
            requirement.requirement_id: (
                None if requirement.preference is None else requirement.preference.field
            )
            for requirement in ranking.requirements
        }
        for evidence in item.preference_evidence:
            field = requirement_fields.get(evidence.requirement_id)
            if field is not None:
                aspect_scores[field] = round(evidence.evidence_score * 100, 4)
        result.append(
            BusinessReference(
                presented_turn_index=turn_index,
                position=item.final_rank,
                business_id=item.business.business_id,
                business_name=item.business.name,
                location=GeoPoint(
                    latitude=item.business.latitude,
                    longitude=item.business.longitude,
                ),
                distance_km=item.distance_km,
                price_level=item.business.price_level,
                categories=list(item.business.categories),
                aspect_scores=aspect_scores,
            )
        )
    return result


def _ranking_review_ids(
    ranking: ReviewEvidenceRankingResult | None,
) -> dict[str, list[str]]:
    """答案由外层主模型生成时，仍从真实排序结果保存本轮代表性证据编号。"""

    if ranking is None or ranking.status != "success":
        return {}
    result: dict[str, list[str]] = {}
    for business in ranking.ranking:
        review_ids: list[str] = []
        for preference in business.preference_evidence:
            review_ids.extend(item.review_id for item in preference.positive_evidence[:1])
            review_ids.extend(item.review_id for item in preference.negative_evidence[:1])
        result[business.business.business_id] = list(dict.fromkeys(review_ids))
    return result


def build_recommendation_workflow(
    project_root: str | Path,
    *,
    business_catalog: BusinessFactCatalog | None = None,
    aspect_profiles: BusinessAspectProfileCatalog | None = None,
    review_evidence_ranker: ReviewEvidenceRanker | None = None,
) -> RecommendationWorkflow:
    """建立推荐入口；Agent 可注入共享评论资源，避免重复加载本地模型。"""

    root = Path(project_root)
    business_catalog = business_catalog or load_business_fact_catalog()
    aspect_profiles = aspect_profiles or load_business_aspect_profile_catalog()
    supported_business_ids = [
        item.business_id for item in aspect_profiles.supported_businesses()
    ]
    profile_store = UserProfileStore(
        root / "data" / "features" / "user_profiles" / "v1"
    )
    return RecommendationWorkflow(
        fusion=build_preference_fusion(business_catalog),
        profile_store=profile_store,
        geography_tool=GeographicDistanceTool(business_catalog),
        hard_filter_tool=StructuredHardFilterTool(
            business_catalog,
            load_fixed_category_catalog(),
            default_candidate_business_ids=supported_business_ids,
        ),
        review_evidence_ranker=(
            review_evidence_ranker
            or build_review_evidence_ranker(
                profile_catalog=aspect_profiles,
                project_root=root,
            )
        ),
        answer_synthesizer=build_recommendation_answer_synthesizer(),
    )
