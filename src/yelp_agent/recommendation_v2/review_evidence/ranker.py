"""在硬筛结果内，用评论证据、商家评分和距离重新排出前五。"""

from __future__ import annotations

import math
import time
from datetime import datetime

from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS, SoftPreference
from yelp_agent.recommendation_v2.tools.hard_filter import (
    FilteredBusiness,
    StructuredHardFilterResult,
)

from .descriptions import PreferenceDescriptionBuilder
from .offline_aspects import OfflineAspectEvidenceResolver
from .retrieval import ReviewEvidenceRetriever
from .schema import (
    BusinessPreferenceEvidence,
    PreferenceRankingLayer,
    PreferenceSearchDescription,
    RankedEvidenceBusiness,
    ReviewEvidenceRankingResult,
    ReviewRetrievalMetrics,
)
from .scoring import EvidenceScoringConfig, aggregate_business_evidence

_FORMULA = (
    "固定14项满足分=0.5+(按用户方向转换后的离线程度-0.5)×证据充分程度，"
    "证据不可用时按0.5；长尾证据分=0.5+0.5×正证据分-0.5×反证据分；"
    "每条偏好换成明确满足、比较满足、一般或证据不足、比较不满足、明确不满足五档，"
    "按priority从前到后逐层比较；全部偏好档位相同后，依次按商家评分、评论数、距离兜底。"
    "原偏好综合分和最终分只保留作新旧对照，不再决定名次"
)


class ReviewEvidenceRanker:
    """新版软排序的单一入口；内部完成描述、检索、证据聚合和最终融合。"""

    def __init__(
        self,
        *,
        description_builder: PreferenceDescriptionBuilder,
        retriever: ReviewEvidenceRetriever,
        offline_aspects: OfflineAspectEvidenceResolver | None = None,
        scoring_config: EvidenceScoringConfig | None = None,
    ) -> None:
        scoring_config = scoring_config or EvidenceScoringConfig()
        if scoring_config.acceptance_threshold != retriever.acceptance_threshold:
            raise ValueError("retrieval and scoring acceptance thresholds must match")
        self._description_builder = description_builder
        self._retriever = retriever
        self._offline_aspects = offline_aspects
        self._scoring_config = scoring_config

    def close(self) -> None:
        self._retriever.close()

    def rank(
        self,
        *,
        state: object,
        hard_filter: StructuredHardFilterResult,
        reference_time: datetime,
        prepared_descriptions: list[PreferenceSearchDescription] | None = None,
    ) -> ReviewEvidenceRankingResult:
        """排序硬筛后的全部商家，不先用评分砍成前十。"""

        # 这里延迟导入和校验，避免为了类型提示在运行时形成模块循环。
        from yelp_agent.recommendation_v2.schema import UnifiedRecommendationState

        if not isinstance(state, UnifiedRecommendationState):
            raise TypeError("state must be a UnifiedRecommendationState")
        started = time.perf_counter()
        description_started = time.perf_counter()
        if prepared_descriptions is None:
            # 保留离线工具和旧调用方兼容入口；真实推荐工作流会把融合模型
            # 同一次生成的检索说法直接传进来，因此在线不再发生第二次调用。
            description_result = self._description_builder.build(
                state.soft_preferences,
                state.open_requirements,
                query_text=state.latest_query_text,
            )
        else:
            from .descriptions import DescriptionBuildResult

            description_result = DescriptionBuildResult(
                descriptions=[
                    item.model_copy(deep=True) for item in prepared_descriptions
                ]
            )
        description_latency_ms = (time.perf_counter() - description_started) * 1000
        call = description_result.call
        if description_result.failure_reason is not None:
            return ReviewEvidenceRankingResult(
                status="description_failure",
                hard_filtered_count=hard_filter.candidate_count,
                requirements=description_result.descriptions,
                recall_threshold=self._retriever.recall_threshold,
                acceptance_threshold=self._retriever.acceptance_threshold,
                direction_margin=self._retriever.direction_margin,
                formula=_FORMULA,
                model_call_count=1 if call is not None else 0,
                input_tokens=call.input_tokens if call is not None else None,
                output_tokens=call.output_tokens if call is not None else None,
                latency_ms=(time.perf_counter() - started) * 1000,
                description_latency_ms=description_latency_ms,
                description_warning=description_result.warning,
                failure_reason=description_result.failure_reason,
            )

        business_ids = hard_filter.candidate_business_ids
        evidence_by_requirement: dict[str, dict[str, BusinessPreferenceEvidence]] = {}
        retrieval_metrics: ReviewRetrievalMetrics | None = None
        scoring_started = time.perf_counter()
        try:
            fixed_requirements = [
                item
                for item in description_result.descriptions
                if item.kind == "fixed_aspect"
            ]
            dynamic_requirements = [
                item
                for item in description_result.descriptions
                if item.kind == "long_tail"
            ]
            if fixed_requirements and self._offline_aspects is not None:
                evidence_by_requirement.update(
                    self._offline_aspects.assess_many(
                        fixed_requirements,
                        business_ids,
                        reference_time=reference_time,
                    )
                )
            elif fixed_requirements:
                # 兼容旧的离线实验构造方式。正式运行时一定会传入离线画像，
                # 因此固定14项不会再走这条动态评论检索路线。
                dynamic_requirements = [*fixed_requirements, *dynamic_requirements]

            if dynamic_requirements:
                retrieval = self._retriever.retrieve_many(
                    dynamic_requirements,
                    business_ids,
                    cutoff_time=reference_time,
                )
                retrieval_metrics = retrieval.metrics
            else:
                retrieval = None
                retrieval_metrics = ReviewRetrievalMetrics()

            for requirement in dynamic_requirements:
                if retrieval is None:  # pragma: no cover - 由上面的分支保证
                    raise RuntimeError("dynamic requirements have no retrieval result")
                recalled = retrieval.by_requirement[requirement.requirement_id]
                evidence_by_requirement[requirement.requirement_id] = {
                    business_id: aggregate_business_evidence(
                        requirement,
                        business_id,
                        recalled.get(business_id, []),
                        reference_time=reference_time,
                        config=self._scoring_config,
                    )
                    for business_id in business_ids
                }
        # 检索边界同时包含本地模型、Qdrant 和原评论表；统一转成可观察失败，
        # 不能让一次外部组件异常破坏已经融合好的会话状态。
        except Exception as exc:  # noqa: BLE001
            return ReviewEvidenceRankingResult(
                status="retrieval_failure",
                hard_filtered_count=hard_filter.candidate_count,
                requirements=description_result.descriptions,
                recall_threshold=self._retriever.recall_threshold,
                acceptance_threshold=self._retriever.acceptance_threshold,
                direction_margin=self._retriever.direction_margin,
                formula=_FORMULA,
                model_call_count=1 if call is not None else 0,
                input_tokens=call.input_tokens if call is not None else None,
                output_tokens=call.output_tokens if call is not None else None,
                latency_ms=(time.perf_counter() - started) * 1000,
                description_latency_ms=description_latency_ms,
                description_warning=description_result.warning,
                retrieval_metrics=retrieval_metrics,
                failure_reason=f"review evidence loading failed: {exc}",
            )

        candidates = [
            self._rank_business(
                candidate,
                state.soft_preferences,
                description_result.descriptions,
                evidence_by_requirement,
            )
            for candidate in hard_filter.candidates
        ]
        ordered = sorted(candidates, key=_priority_layered_sort_key)[:5]
        ranking = [
            item.model_copy(update={"final_rank": index})
            for index, item in enumerate(ordered, start=1)
        ]
        scoring_latency_ms = (time.perf_counter() - scoring_started) * 1000
        return ReviewEvidenceRankingResult(
            status="success",
            hard_filtered_count=hard_filter.candidate_count,
            requirements=description_result.descriptions,
            ranking=ranking,
            recall_threshold=self._retriever.recall_threshold,
            acceptance_threshold=self._retriever.acceptance_threshold,
            direction_margin=self._retriever.direction_margin,
            formula=_FORMULA,
            model_call_count=1 if call is not None else 0,
            input_tokens=call.input_tokens if call is not None else None,
            output_tokens=call.output_tokens if call is not None else None,
            latency_ms=(time.perf_counter() - started) * 1000,
            description_latency_ms=description_latency_ms,
            description_warning=description_result.warning,
            scoring_latency_ms=scoring_latency_ms,
            retrieval_metrics=retrieval_metrics,
        )

    @staticmethod
    def _rank_business(
        candidate: FilteredBusiness,
        preferences: list[SoftPreference],
        review_requirements: list[PreferenceSearchDescription],
        evidence_by_requirement: dict[str, dict[str, BusinessPreferenceEvidence]],
    ) -> RankedEvidenceBusiness:
        business_id = candidate.business.business_id
        weighted_scores: list[tuple[float, float]] = []
        evidence: list[BusinessPreferenceEvidence] = []
        priority_layers: list[PreferenceRankingLayer] = []
        handled_preference_keys: set[str] = set()
        for requirement in review_requirements:
            assessment = evidence_by_requirement[requirement.requirement_id][
                business_id
            ]
            evidence.append(assessment)
            weighted_scores.append(
                (
                    _preference_weight(
                        requirement.priority, requirement.preference_strength
                    ),
                    assessment.evidence_score,
                )
            )
            if requirement.preference is not None:
                handled_preference_keys.add(requirement.preference.key)
            priority_layers.append(_evidence_ranking_layer(requirement, assessment))

        for preference in preferences:
            if (
                preference.key in handled_preference_keys
                or preference.field in ASPECT_FIELDS
            ):
                continue
            score = _structured_preference_score(preference, candidate)
            weighted_scores.append(
                (
                    _preference_weight(
                        preference.priority,
                        preference.preference_strength,
                    ),
                    score,
                )
            )
            priority_layers.append(_structured_ranking_layer(preference, score))

        preference_score = _weighted_mean(weighted_scores, default=0.5)
        rating_score = candidate.business.rating / 5.0
        distance_score = (
            math.exp(-candidate.distance_km / 5.0)
            if candidate.distance_km is not None
            else 0.5
        )
        final_score = 0.6 * preference_score + 0.3 * rating_score + 0.1 * distance_score
        return RankedEvidenceBusiness(
            final_rank=1,
            business=candidate.business,
            distance_km=candidate.distance_km,
            preference_score=preference_score,
            rating_score=rating_score,
            distance_score=distance_score,
            final_score=final_score,
            priority_layers=sorted(
                priority_layers,
                key=lambda item: (item.priority, item.requirement_id),
            ),
            preference_evidence=evidence,
        )


def _priority_layered_sort_key(item: RankedEvidenceBusiness) -> tuple[object, ...]:
    """先逐层比较偏好档位；同档时才回到稳定的商家基础顺序。"""

    ordered_layers = sorted(
        item.priority_layers,
        key=lambda layer: (layer.priority, layer.requirement_id),
    )
    return (
        *(-layer.satisfaction_tier for layer in ordered_layers),
        -item.business.rating,
        -item.business.review_count,
        item.distance_km if item.distance_km is not None else math.inf,
        item.business.business_id,
    )


def _evidence_ranking_layer(
    requirement: PreferenceSearchDescription,
    assessment: BusinessPreferenceEvidence,
) -> PreferenceRankingLayer:
    """把评论证据结果压成稳定档位；证据不足只能进入中间未知档。"""

    level = assessment.satisfaction_level or _score_level(assessment.evidence_score)
    preference = requirement.preference
    return PreferenceRankingLayer(
        priority=requirement.priority,
        requirement_id=requirement.requirement_id,
        requirement_text=requirement.requirement_text,
        field=None if preference is None else preference.field,
        controlling_source=(
            None if preference is None else preference.controlling_source
        ),
        satisfaction_score=assessment.evidence_score,
        satisfaction_level=level,
        satisfaction_tier=_level_tier(level),
    )


def _structured_ranking_layer(
    preference: SoftPreference,
    score: float,
) -> PreferenceRankingLayer:
    """把评分、距离等已知商家事实也放进同一套逐层比较结构。"""

    level = _score_level(score)
    return PreferenceRankingLayer(
        priority=preference.priority,
        requirement_id=preference.key,
        requirement_text=preference.sources[0].text,
        field=preference.field,
        controlling_source=preference.controlling_source,
        satisfaction_score=score,
        satisfaction_level=level,
        satisfaction_tier=_level_tier(level),
    )


def _score_level(score: float) -> str:
    """统一动态评论和结构化事实的五档阈值。"""

    if score >= 0.8:
        return "明确满足"
    if score >= 0.6:
        return "比较满足"
    if score > 0.4:
        return "一般"
    if score > 0.2:
        return "比较不满足"
    return "明确不满足"


def _level_tier(level: str) -> int:
    """证据不足与一般同处中间档，既不冒充满足也不被当成反证。"""

    return {
        "明确满足": 4,
        "比较满足": 3,
        "一般": 2,
        "证据不足": 2,
        "比较不满足": 1,
        "明确不满足": 0,
    }[level]


def _preference_weight(priority: int, strength: int) -> float:
    return strength / 100.0 * 0.75 ** (priority - 1)


def _weighted_mean(values: list[tuple[float, float]], *, default: float) -> float:
    total_weight = sum(weight for weight, _ in values)
    if total_weight <= 0:
        return default
    return sum(weight * score for weight, score in values) / total_weight


def _structured_preference_score(
    preference: SoftPreference,
    candidate: FilteredBusiness,
) -> float:
    """无需评论即可确定的软偏好直接读取商家事实或本轮距离。"""

    business = candidate.business
    field = preference.field
    if field == "distance_km":
        if candidate.distance_km is None:
            return 0.5
        return math.exp(-candidate.distance_km / 5.0)
    if field == "rating":
        score = business.rating / 5.0
        return score if preference.direction == "higher" else 1.0 - score
    if field == "review_count":
        # 评论数没有固定上限，用1000条作为逐渐饱和的可解释尺度。
        score = min(1.0, math.log1p(business.review_count) / math.log1p(1000))
        return score if preference.direction == "higher" else 1.0 - score
    if field == "price_level":
        if business.price_level is None:
            return 0.5
        if preference.direction == "closer_to":
            target = int(preference.target_value)  # type: ignore[arg-type]
            return 1.0 - abs(business.price_level - target) / 3.0
        normalized = (business.price_level - 1) / 3.0
        return normalized if preference.direction == "higher" else 1.0 - normalized
    if field == "category":
        targets = set(preference.target_value)  # type: ignore[arg-type]
        matches = bool(targets.intersection(business.categories))
        return float(matches if preference.direction == "match" else not matches)
    if hasattr(business, field):
        value = getattr(business, field)
        if value is None:
            return 0.5
        target = bool(preference.target_value)
        matches = value is target
        return float(matches if preference.direction == "match" else not matches)
    return 0.5
