"""把500家固定14项离线画像转换成现有排序能够直接使用的证据。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from yelp_agent.recommendation_v2.business_aspect_profiles import (
    BusinessAspectEvidence,
    BusinessAspectProfileCatalog,
    BusinessAspectScore,
)
from yelp_agent.recommendation_v2.schema import AspectField

from .schema import (
    BusinessPreferenceEvidence,
    PreferenceSearchDescription,
    RankedReviewEvidence,
)
from .scoring import count_reliability


class OfflineAspectEvidenceResolver:
    """一次批量读取固定特征分数和代表性评论，并按用户方向解释。"""

    def __init__(
        self,
        catalog: BusinessAspectProfileCatalog,
        *,
        evidence_limit_per_group: int = 5,
        half_life_days: float = 730.0,
    ) -> None:
        if evidence_limit_per_group < 1 or half_life_days <= 0:
            raise ValueError("offline evidence settings must be positive")
        self._catalog = catalog
        self._evidence_limit = evidence_limit_per_group
        self._half_life_days = half_life_days

    def assess_many(
        self,
        requirements: list[PreferenceSearchDescription],
        business_ids: list[str],
        *,
        reference_time: datetime,
    ) -> dict[str, dict[str, BusinessPreferenceEvidence]]:
        """返回“要求编号→商家编号→证据判断”，不执行任何模型调用。"""

        if not requirements or not business_ids:
            return {}
        for requirement in requirements:
            if requirement.kind != "fixed_aspect" or requirement.preference is None:
                raise ValueError("offline profiles only accept fixed review aspects")

        aspect_ids = list(
            dict.fromkeys(
                requirement.preference.field  # type: ignore[union-attr]
                for requirement in requirements
            )
        )
        scores = self._catalog.scores(business_ids, aspect_ids)
        evidence = self._catalog.evidence(
            business_ids,
            aspect_ids,
            limit_per_group=self._evidence_limit,
        )
        score_by_key = {(item.business_id, item.aspect_id): item for item in scores}
        evidence_by_key: dict[
            tuple[str, AspectField, str], list[BusinessAspectEvidence]
        ] = defaultdict(list)
        for item in evidence:
            evidence_by_key[(item.business_id, item.aspect_id, item.evidence_group)].append(item)

        result: dict[str, dict[str, BusinessPreferenceEvidence]] = {}
        for requirement in requirements:
            preference = requirement.preference
            if preference is None:  # pragma: no cover - 前面的校验已经保证
                raise RuntimeError("fixed aspect preference disappeared")
            aspect_id: AspectField = preference.field  # type: ignore[assignment]
            result[requirement.requirement_id] = {
                business_id: self._assessment(
                    requirement,
                    score_by_key[(business_id, aspect_id)],
                    evidence_by_key,
                    reference_time,
                )
                for business_id in business_ids
            }
        return result

    def _assessment(
        self,
        requirement: PreferenceSearchDescription,
        score: BusinessAspectScore,
        evidence_by_key: dict[
            tuple[str, AspectField, str], list[BusinessAspectEvidence]
        ],
        reference_time: datetime,
    ) -> BusinessPreferenceEvidence:
        preference = requirement.preference
        if preference is None:
            raise ValueError("fixed aspect requirement lost its preference")
        wants_higher = preference.direction == "higher"

        # ranking_degree为空表示证据达不到离线程序规定的最低标准。
        # 此时必须按未知0.5处理，不能偷偷改用看似很极端的原始程度。
        if score.usable_for_ranking and score.ranking_degree is not None:
            direction_adjusted = (
                score.ranking_degree if wants_higher else 1.0 - score.ranking_degree
            )
            satisfaction = 0.5 + (
                direction_adjusted - 0.5
            ) * score.evidence_sufficiency
        else:
            direction_adjusted = None
            satisfaction = 0.5

        high = evidence_by_key.get(
            (score.business_id, score.aspect_id, "high_degree"), []
        )
        low = evidence_by_key.get(
            (score.business_id, score.aspect_id, "low_degree"), []
        )
        supporting = high if wants_higher else low
        contradicting = low if wants_higher else high
        positive = [
            self._ranked(item, "positive", reference_time) for item in supporting
        ]
        negative = [
            self._ranked(item, "negative", reference_time) for item in contradicting
        ]
        return BusinessPreferenceEvidence(
            business_id=score.business_id,
            requirement_id=requirement.requirement_id,
            evidence_source="offline_business_profile",
            positive_evidence=positive,
            negative_evidence=negative,
            positive_component=max(0.0, 2.0 * (satisfaction - 0.5)),
            negative_component=max(0.0, 2.0 * (0.5 - satisfaction)),
            positive_count_reliability=count_reliability(len(positive)),
            negative_count_reliability=count_reliability(len(negative)),
            evidence_score=satisfaction,
            recalled_review_count=score.retrieved_candidate_count,
            ambiguous_review_count=0,
            objective_degree=score.degree,
            direction_adjusted_degree=direction_adjusted,
            evidence_sufficiency=score.evidence_sufficiency,
            evidence_sufficiency_level=score.evidence_sufficiency_level,
            controversy=score.controversy,
            controversy_level=score.controversy_level,
            usable_for_ranking=score.usable_for_ranking,
            unusable_reasons=list(score.unusable_reasons),
            satisfaction_level=_satisfaction_level(
                satisfaction,
                usable=score.usable_for_ranking,
                sufficiency=score.evidence_sufficiency,
            ),
        )

    def _ranked(
        self,
        item: BusinessAspectEvidence,
        role: str,
        reference_time: datetime,
    ) -> RankedReviewEvidence:
        age_days = max(
            0.0,
            (reference_time - item.review_time).total_seconds() / 86400.0,
        )
        return RankedReviewEvidence(
            review_id=item.review_id,
            role=role,  # type: ignore[arg-type]
            evidence_source="offline_business_profile",
            review_time=item.review_time,
            stars=item.stars,
            useful=item.useful,
            review_text=item.text,
            matched_segment_text=item.text,
            model_relevance=item.relevance,
            model_strength=item.strength,
            relevance_score=item.relevance / 3.0,
            time_weight=2 ** (-age_days / self._half_life_days),
            # 直接复用离线聚合时已经计算好的相关、时间和useful轻微加权结果。
            evidence_weight=min(1.0, item.evidence_weight),
        )


def _satisfaction_level(score: float, *, usable: bool, sufficiency: float) -> str:
    """给最终回答一个通俗档位，不把精确内部排序分暴露给大模型。"""

    if not usable or sufficiency <= 0.3:
        return "证据不足"
    if score >= 0.8:
        return "明确满足"
    if score >= 0.6:
        return "比较满足"
    if score > 0.4:
        return "一般"
    if score > 0.2:
        return "比较不满足"
    return "明确不满足"
