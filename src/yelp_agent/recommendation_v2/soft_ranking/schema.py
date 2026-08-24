"""评论证据排序对外使用的固定数据结构。"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import BusinessFact
from yelp_agent.recommendation_v2.schema import RequirementField, SoftPreference

type EvidenceLevel = Literal[
    "clearly_satisfies",
    "conditionally_satisfies",
    "unknown",
    "clearly_contradicts",
]
type EvidenceRole = Literal["positive", "negative", "conditional"]
type ReviewVerdict = Literal["positive", "negative", "conditional", "irrelevant"]
type RetrievalSide = Literal["positive", "negative", "both"]
type PreferenceEvaluationMethod = Literal["review_evidence", "structured_fact"]


class BaselineRankedBusiness(StrictModel):
    """按照商家评分产生的基础位置。"""

    business_id: str = Field(min_length=1)
    baseline_rank: int = Field(ge=1)
    # 下面这些字段只为兼容此前真实对照结果，新流程不再填写。
    model_rank: int | None = Field(default=None, ge=1)
    old_hybrid_rank: int | None = Field(default=None, ge=1)
    model_score: float | None = None
    old_hybrid_score: float | None = None
    blend_score: float | None = Field(default=None, ge=0, le=1)


class BaselineRankingResult(StrictModel):
    """硬筛选候选按评分、评论数和距离得到的完整基础顺序。"""

    source: Literal["rating", "legacy_hybrid_v2", "fact_fallback"]
    fallback_reason: str | None = Field(default=None, max_length=500)
    ranked_businesses: list[BaselineRankedBusiness]

    @model_validator(mode="after")
    def validate_complete_order(self) -> Self:
        ids = [item.business_id for item in self.ranked_businesses]
        ranks = [item.baseline_rank for item in self.ranked_businesses]
        if len(ids) != len(set(ids)):
            raise ValueError("baseline ranking business IDs must be unique")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("baseline ranks must be contiguous from one")
        if (self.source == "fact_fallback") != (self.fallback_reason is not None):
            raise ValueError("only fallback baseline ranking has a fallback reason")
        return self


class RetrievedReview(StrictModel):
    """正面或反面向量检索找到的一条完整原评论。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    review_time: str = Field(min_length=1)
    stars: float = Field(ge=1, le=5)
    useful: int = Field(ge=0)
    review_text: str = Field(min_length=1)
    retrieval_side: RetrievalSide
    positive_similarity: float | None = Field(default=None, ge=-1, le=1)
    negative_similarity: float | None = Field(default=None, ge=-1, le=1)
    positive_weight: float = Field(default=0, ge=0, le=1)
    negative_weight: float = Field(default=0, ge=0, le=1)


class SelectedEvidence(StrictModel):
    """大模型判定方向后，程序附回的一条真实评论证据。"""

    role: EvidenceRole
    review_id: str = Field(min_length=1)
    review_time: str = Field(min_length=1)
    stars: float = Field(ge=1, le=5)
    review_text: str = Field(min_length=1)
    evidence_weight: float = Field(ge=0, le=1)
    judgment_reason: str = Field(min_length=1, max_length=300)


class PreferenceEvidenceAssessment(StrictModel):
    """一家餐厅在一条偏好下，由程序根据正反证据计算出的结果。"""

    business_id: str = Field(min_length=1)
    level: EvidenceLevel
    satisfaction_score: float = Field(ge=0, le=1)
    preference_weight: float = Field(ge=0)
    reason: str = Field(min_length=1, max_length=500)
    positive_weight: float = Field(default=0, ge=0)
    negative_weight: float = Field(default=0, ge=0)
    conditional_weight: float = Field(default=0, ge=0)
    accepted_evidence_count: int = Field(default=0, ge=0)
    positive_evidence: list[SelectedEvidence] = Field(
        default_factory=list, max_length=2
    )
    negative_evidence: list[SelectedEvidence] = Field(
        default_factory=list, max_length=2
    )
    conditional_evidence: list[SelectedEvidence] = Field(
        default_factory=list, max_length=2
    )
    retrieved_review_count: int = Field(ge=0)


class PreferenceRankingPass(StrictModel):
    """加入一条偏好后，记录累计加权分和顺序变化。"""

    preference: SoftPreference
    method: PreferenceEvaluationMethod
    formula: str = Field(min_length=1)
    order_before: list[str]
    order_after: list[str]
    assessments: list[PreferenceEvidenceAssessment]

    @model_validator(mode="after")
    def validate_same_scope(self) -> Self:
        before = self.order_before
        after = self.order_after
        assessment_ids = [item.business_id for item in self.assessments]
        if len(before) != len(set(before)) or set(before) != set(after):
            raise ValueError("preference ranking must preserve the candidate scope")
        if set(assessment_ids) != set(before) or len(assessment_ids) != len(before):
            raise ValueError("every ranked business requires one assessment")
        return self


class FinalRankedBusiness(StrictModel):
    """全部软偏好加权后的最终一家餐厅。"""

    final_rank: int = Field(ge=1)
    baseline_rank: int = Field(ge=1)
    business: BusinessFact
    distance_km: float | None = Field(default=None, ge=0)
    combined_preference_score: float = Field(ge=0, le=1)
    preference_scores: dict[RequirementField, float]
    preference_levels: dict[RequirementField, EvidenceLevel]


class SoftRankingAttempt(StrictModel):
    """软排序成功或失败的完整公开结果。"""

    status: Literal["success", "provider_failure", "invalid_output"]
    baseline: BaselineRankingResult | None = None
    evaluated_candidate_count: int = Field(ge=0)
    omitted_after_baseline_count: int = Field(ge=0)
    passes: list[PreferenceRankingPass] = Field(default_factory=list)
    ranking: list[FinalRankedBusiness] = Field(default_factory=list)
    failure_reason: str | None = Field(default=None, max_length=500)
    model: str | None = None
    model_call_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    raw_model_outputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.status == "success":
            if self.baseline is None or self.failure_reason is not None:
                raise ValueError(
                    "successful soft ranking needs a baseline and no failure"
                )
            ranks = [item.final_rank for item in self.ranking]
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("final ranks must be contiguous from one")
            if len(self.ranking) != self.evaluated_candidate_count:
                raise ValueError("final ranking must cover every evaluated candidate")
        elif self.failure_reason is None:
            raise ValueError("failed soft ranking requires a safe failure reason")
        return self
