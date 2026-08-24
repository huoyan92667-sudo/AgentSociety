"""用多条软偏好的加权短板分重排评分前十候选。"""

from __future__ import annotations

import math
import time

from yelp_agent.recommendation_v2.review_features.definitions import (
    preference_semantic_anchors,
)
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS, SoftPreference
from yelp_agent.recommendation_v2.tools.hard_filter import (
    FilteredBusiness,
    StructuredHardFilterResult,
)

from .evidence_judge import ReviewEvidenceJudge
from .review_store import ReviewVectorStore
from .schema import (
    BaselineRankingResult,
    EvidenceLevel,
    FinalRankedBusiness,
    PreferenceEvidenceAssessment,
    PreferenceRankingPass,
    SoftRankingAttempt,
)

_SATISFACTION_FORMULA = "(正面权重 + 0.5×条件性权重 + 1) / (正反条件权重总和 + 2)"
_COMBINATION_FORMULA = "偏好权重=强度/100×0.75^(优先级-1)；总分为各满足分的加权几何平均"


class WeightedPreferenceRanker:
    """先评分取前十，再让所有偏好共同影响顺序，避免第一条偏好一票定生死。"""

    def __init__(
        self,
        *,
        review_store: ReviewVectorStore,
        evidence_judge: ReviewEvidenceJudge,
        candidate_limit: int = 10,
        reviews_each_side: int = 5,
    ) -> None:
        if candidate_limit < 1 or reviews_each_side < 1:
            raise ValueError("ranking and review limits must be positive")
        self._review_store = review_store
        self._evidence_judge = evidence_judge
        self._candidate_limit = candidate_limit
        self._reviews_each_side = reviews_each_side

    def close(self) -> None:
        """关闭完整评论查询和本地向量模型。"""

        self._review_store.close()

    def rank(
        self,
        *,
        preferences: list[SoftPreference],
        hard_filter: StructuredHardFilterResult,
        baseline: BaselineRankingResult,
    ) -> SoftRankingAttempt:
        """计算每项满足分，再用优先级和强度共同生成最终分。"""

        started_at = time.perf_counter()
        baseline_ids = [item.business_id for item in baseline.ranked_businesses]
        hard_ids = set(hard_filter.candidate_business_ids)
        if len(baseline_ids) != len(hard_ids) or set(baseline_ids) != hard_ids:
            raise ValueError("baseline ranking must cover the hard-filter result")

        scope = baseline_ids[: self._candidate_limit]
        omitted = len(baseline_ids) - len(scope)
        by_business = {
            item.business.business_id: item for item in hard_filter.candidates
        }
        baseline_rank = {
            item.business_id: item.baseline_rank for item in baseline.ranked_businesses
        }
        ordered_preferences = sorted(preferences, key=lambda item: item.priority)
        current_order = list(scope)
        weighted_scores: dict[str, list[tuple[float, float]]] = {
            business_id: [] for business_id in scope
        }
        field_scores: dict[str, dict[str, float]] = {
            business_id: {} for business_id in scope
        }
        field_levels: dict[str, dict[str, EvidenceLevel]] = {
            business_id: {} for business_id in scope
        }
        passes: list[PreferenceRankingPass] = []
        raw_outputs: list[str] = []
        model = None
        calls = 0
        input_tokens = 0
        output_tokens = 0

        for preference in ordered_preferences:
            order_before = list(current_order)
            if preference.field in ASPECT_FIELDS:
                anchors = preference_semantic_anchors(
                    preference.field,
                    preference.direction,
                )
                reviews = self._review_store.retrieve(
                    scope,
                    anchors.satisfying,
                    anchors.contradicting,
                    limit_each_side=self._reviews_each_side,
                )
                judged = self._evidence_judge.judge(preference, reviews)
                calls += 1
                model = judged.call.model or model
                if judged.call.input_tokens is not None:
                    input_tokens += judged.call.input_tokens
                if judged.call.output_tokens is not None:
                    output_tokens += judged.call.output_tokens
                if judged.raw_json is not None:
                    raw_outputs.append(judged.raw_json)
                if judged.failure_reason is not None:
                    status = (
                        "provider_failure"
                        if judged.call.status != "success"
                        else "invalid_output"
                    )
                    return SoftRankingAttempt(
                        status=status,
                        baseline=baseline,
                        evaluated_candidate_count=len(scope),
                        omitted_after_baseline_count=omitted,
                        passes=passes,
                        failure_reason=judged.failure_reason,
                        model=model,
                        model_call_count=calls,
                        input_tokens=input_tokens or None,
                        output_tokens=output_tokens or None,
                        latency_ms=(time.perf_counter() - started_at) * 1000.0,
                        raw_model_outputs=raw_outputs,
                    )
                assessments = judged.assessments
                method = "review_evidence"
                formula = _SATISFACTION_FORMULA + "；" + _COMBINATION_FORMULA
            else:
                assessments = self._structured_assessments(
                    preference,
                    scope,
                    by_business,
                )
                method = "structured_fact"
                formula = _COMBINATION_FORMULA

            assessment_by_id = {item.business_id: item for item in assessments}
            for business_id in scope:
                assessment = assessment_by_id[business_id]
                weighted_scores[business_id].append(
                    (assessment.satisfaction_score, assessment.preference_weight)
                )
                field_scores[business_id][preference.field] = (
                    assessment.satisfaction_score
                )
                field_levels[business_id][preference.field] = assessment.level
            current_order = sorted(
                scope,
                key=lambda business_id: (
                    -self._combined_score(weighted_scores[business_id]),
                    baseline_rank[business_id],
                ),
            )
            passes.append(
                PreferenceRankingPass(
                    preference=preference,
                    method=method,
                    formula=formula,
                    order_before=order_before,
                    order_after=list(current_order),
                    assessments=assessments,
                )
            )

        ranking = [
            FinalRankedBusiness(
                final_rank=rank,
                baseline_rank=baseline_rank[business_id],
                business=by_business[business_id].business,
                distance_km=by_business[business_id].distance_km,
                combined_preference_score=self._combined_score(
                    weighted_scores[business_id]
                ),
                preference_scores=field_scores[business_id],  # type: ignore[arg-type]
                preference_levels=field_levels[business_id],  # type: ignore[arg-type]
            )
            for rank, business_id in enumerate(current_order, 1)
        ]
        return SoftRankingAttempt(
            status="success",
            baseline=baseline,
            evaluated_candidate_count=len(scope),
            omitted_after_baseline_count=omitted,
            passes=passes,
            ranking=ranking,
            model=model,
            model_call_count=calls,
            input_tokens=input_tokens or None,
            output_tokens=output_tokens or None,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            raw_model_outputs=raw_outputs,
        )

    @staticmethod
    def _combined_score(values: list[tuple[float, float]]) -> float:
        """几何平均会惩罚明显短板，但不会让第一条偏好直接一票否决。"""

        if not values:
            return 0.5
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0:
            return 0.5
        log_score = (
            sum(weight * math.log(max(score, 0.05)) for score, weight in values)
            / total_weight
        )
        return math.exp(log_score)

    @staticmethod
    def _structured_assessments(
        preference: SoftPreference,
        scope: list[str],
        by_business: dict[str, FilteredBusiness],
    ) -> list[PreferenceEvidenceAssessment]:
        """评分、价格、距离等已知事实直接换成0到1满足分。"""

        raw_values = {
            business_id: WeightedPreferenceRanker._raw_structured_value(
                preference,
                by_business[business_id],
            )
            for business_id in scope
        }
        numeric_values = [
            float(value)
            for value in raw_values.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        low = min(numeric_values) if numeric_values else None
        high = max(numeric_values) if numeric_values else None
        preference_weight = (preference.preference_strength / 100.0) * (
            0.75 ** (preference.priority - 1)
        )
        output: list[PreferenceEvidenceAssessment] = []
        for business_id in scope:
            value = raw_values[business_id]
            if value is None:
                level: EvidenceLevel = "unknown"
                score = 0.5
                reason = "商家基础事实中没有这个字段的已知值"
            else:
                score = WeightedPreferenceRanker._structured_score(
                    preference,
                    value,
                    low,
                    high,
                )
                if score >= 0.75:
                    level = "clearly_satisfies"
                elif score <= 0.25:
                    level = "clearly_contradicts"
                else:
                    level = "conditionally_satisfies"
                reason = f"商家基础事实值为{value!r}，换算满足分为{score:.3f}"
            output.append(
                PreferenceEvidenceAssessment(
                    business_id=business_id,
                    level=level,
                    satisfaction_score=score,
                    preference_weight=preference_weight,
                    reason=reason,
                    retrieved_review_count=0,
                )
            )
        return output

    @staticmethod
    def _raw_structured_value(
        preference: SoftPreference,
        candidate: FilteredBusiness,
    ) -> object | None:
        business = candidate.business
        if preference.field == "distance_km":
            return candidate.distance_km
        if preference.field == "category":
            targets = set(preference.target_value or [])
            overlap = bool(targets.intersection(business.categories))
            return overlap if preference.direction == "match" else not overlap
        return getattr(business, preference.field, None)

    @staticmethod
    def _structured_score(
        preference: SoftPreference,
        value: object,
        low: float | None,
        high: float | None,
    ) -> float:
        if preference.direction in {"match", "avoid"}:
            if preference.field == "category":
                matched = bool(value)
            else:
                matched = value == preference.target_value
                if preference.direction == "avoid":
                    matched = not matched
            return 1.0 if matched else 0.05
        if preference.direction == "closer_to":
            difference = abs(float(value) - float(preference.target_value))
            return max(0.05, 1.0 - difference / 3.0)
        numeric = float(value)
        if low is None or high is None or math.isclose(low, high):
            return 1.0
        normalized = (numeric - low) / (high - low)
        score = normalized if preference.direction == "higher" else 1.0 - normalized
        return max(0.05, min(1.0, score))


# 保留旧名字，外部入口不需要跟着实现细节改名。
PriorityLayeredRanker = WeightedPreferenceRanker
