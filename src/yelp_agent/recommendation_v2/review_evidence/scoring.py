"""用正反相似度、时间和数量计算商家在一条偏好下的证据分。"""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import Field

from yelp_agent.models import StrictModel

from .schema import (
    BusinessPreferenceEvidence,
    PreferenceSearchDescription,
    RankedReviewEvidence,
    ReviewSimilarityCandidate,
)


class EvidenceScoringConfig(StrictModel):
    """当前暂定的证据阈值和时间衰减参数，后续可用测试集校准。"""

    acceptance_threshold: float = Field(default=0.60, ge=-1, lt=1)
    top_each_side: int = Field(default=5, ge=1, le=20)
    half_life_days: float = Field(default=730.0, gt=0)


def aggregate_business_evidence(
    requirement: PreferenceSearchDescription,
    business_id: str,
    candidates: list[ReviewSimilarityCandidate],
    *,
    reference_time: datetime,
    config: EvidenceScoringConfig | None = None,
) -> BusinessPreferenceEvidence:
    """模糊评论不计分；正反各保留最多五条真实评论。"""

    config = config or EvidenceScoringConfig()
    eligible = [item for item in candidates if item.review_time < reference_time]
    positive = sorted(
        (
            _ranked_evidence(item, "positive", reference_time, config)
            for item in eligible
            if item.direction == "positive"
        ),
        key=lambda item: (-item.evidence_weight, item.review_id),
    )[: config.top_each_side]
    negative = sorted(
        (
            _ranked_evidence(item, "negative", reference_time, config)
            for item in eligible
            if item.direction == "negative"
        ),
        key=lambda item: (-item.evidence_weight, item.review_id),
    )[: config.top_each_side]
    positive_reliability = count_reliability(len(positive))
    negative_reliability = count_reliability(len(negative))
    positive_component = _component(positive, positive_reliability)
    negative_component = _component(negative, negative_reliability)

    # 没证据时为0.5；正证据把它推高，反证据把它压低。
    evidence_score = min(
        1.0,
        max(0.0, 0.5 + 0.5 * positive_component - 0.5 * negative_component),
    )
    return BusinessPreferenceEvidence(
        business_id=business_id,
        requirement_id=requirement.requirement_id,
        positive_evidence=positive,
        negative_evidence=negative,
        positive_component=positive_component,
        negative_component=negative_component,
        positive_count_reliability=positive_reliability,
        negative_count_reliability=negative_reliability,
        evidence_score=evidence_score,
        recalled_review_count=len(eligible),
        ambiguous_review_count=sum(
            item.direction == "ambiguous" for item in eligible
        ),
        max_positive_similarity=max(
            (item.positive_similarity for item in eligible),
            default=None,
        ),
        max_negative_similarity=max(
            (item.negative_similarity for item in eligible),
            default=None,
        ),
        max_direction_gap=max(
            (
                abs(item.positive_similarity - item.negative_similarity)
                for item in eligible
            ),
            default=None,
        ),
    )


def count_reliability(count: int) -> float:
    """五条及以上达到满数量可靠度，单条偶然命中的影响会被压低。"""

    if count <= 0:
        return 0.0
    return min(1.0, math.log(count + 1) / math.log(6))


def _ranked_evidence(
    candidate: ReviewSimilarityCandidate,
    role: str,
    reference_time: datetime,
    config: EvidenceScoringConfig,
) -> RankedReviewEvidence:
    similarity = (
        candidate.positive_similarity
        if role == "positive"
        else candidate.negative_similarity
    )
    # 按既定聚合公式直接使用余弦相似度；接受门槛只负责决定能否成为证据，
    # 不再把刚过门槛的真实证据额外压成接近零。
    relevance = min(1.0, max(0.0, similarity))
    age_days = max(
        0.0,
        (reference_time - candidate.review_time).total_seconds() / 86400.0,
    )
    time_weight = 2 ** (-age_days / config.half_life_days)
    return RankedReviewEvidence(
        review_id=candidate.review_id,
        role=role,  # type: ignore[arg-type]
        review_time=candidate.review_time,
        stars=candidate.stars,
        useful=candidate.useful,
        review_text=candidate.review_text,
        matched_segment_text=candidate.matched_segment_text,
        positive_similarity=candidate.positive_similarity,
        negative_similarity=candidate.negative_similarity,
        relevance_score=relevance,
        time_weight=time_weight,
        evidence_weight=relevance * time_weight,
    )


def _component(evidence: list[RankedReviewEvidence], reliability: float) -> float:
    if not evidence:
        return 0.0
    mean_weight = sum(item.evidence_weight for item in evidence) / len(evidence)
    return min(1.0, max(0.0, mean_weight * reliability))
