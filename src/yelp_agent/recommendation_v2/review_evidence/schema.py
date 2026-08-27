"""评论召回、证据聚合和最终排序共同使用的数据结构。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import BusinessFact
from yelp_agent.recommendation_v2.schema import SoftPreference

type EvidenceDirection = Literal["positive", "negative", "ambiguous"]
type RequirementKind = Literal["fixed_aspect", "long_tail"]


class PreferenceSearchDescription(StrictModel):
    """一条软偏好用于查评论的正向和反向说法。"""

    requirement_id: str = Field(min_length=1, max_length=200)
    requirement_text: str = Field(min_length=1, max_length=500)
    kind: RequirementKind
    priority: int = Field(ge=1, le=100)
    preference_strength: int = Field(ge=1, le=100)
    # 兼容已经保存的两条说法，同时允许新提示词生成最多五个互补角度。
    # 实际使用几条由召回对照实验决定，不在数据结构里提前写死成三条。
    # 在线提示词当前固定生成2条；上限保留到5只为复放评测实验，
    # 不能据此让正式流程默认执行5倍查询。
    positive_descriptions: list[str] = Field(min_length=2, max_length=5)
    negative_descriptions: list[str] = Field(min_length=2, max_length=5)
    preference: SoftPreference | None = None

    @field_validator("positive_descriptions", "negative_descriptions")
    @classmethod
    def validate_descriptions(cls, values: list[str]) -> list[str]:
        cleaned = [item.strip() for item in values]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("search descriptions must be nonempty and unique")
        return cleaned

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        if (self.kind == "fixed_aspect") != (self.preference is not None):
            raise ValueError("only fixed aspects carry a structured preference")
        return self


class QdrantSegmentHit(StrictModel):
    """Qdrant 只返回片段事实；向量随后按 point_id 从本地文件读取。"""

    point_id: int = Field(ge=0)
    segment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    stars: float = Field(ge=1, le=5)
    useful: int = Field(ge=0)
    segment_index: int = Field(ge=0)
    segment_text: str = Field(min_length=1)
    review_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # 向量路线返回0～1附近的余弦分，关键词路线返回的BM25原始分可大于1；
    # 两者只会按各自名次融合，绝不会把原始数值直接相加。
    route_similarity: float = Field(ge=0)
    # 两路原始名次用于RRF融合审计。向量或关键词任一路没有命中时为None。
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)

    @field_validator("review_time")
    @classmethod
    def normalize_review_time(cls, value: datetime) -> datetime:
        return _utc_datetime(value)


class ReviewSimilarityCandidate(StrictModel):
    """合并片段后，一条原评论相对正反说法的最高相似度。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    stars: float = Field(ge=1, le=5)
    useful: int = Field(ge=0)
    review_text: str = Field(min_length=1)
    review_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matched_segment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    matched_segment_text: str = Field(min_length=1)
    positive_similarity: float = Field(ge=-1, le=1)
    negative_similarity: float = Field(ge=-1, le=1)
    positive_retrieval_score: float = Field(default=0, ge=0)
    negative_retrieval_score: float = Field(default=0, ge=0)
    positive_dense_match: bool = False
    negative_dense_match: bool = False
    positive_bm25_match: bool = False
    negative_bm25_match: bool = False
    direction: EvidenceDirection

    @field_validator("review_time")
    @classmethod
    def normalize_review_time(cls, value: datetime) -> datetime:
        return _utc_datetime(value)


class RankedReviewEvidence(StrictModel):
    """参与商家分数计算、同时可以原样展示给用户的真实评论。"""

    review_id: str = Field(min_length=1)
    role: Literal["positive", "negative"]
    review_time: datetime
    stars: float = Field(ge=1, le=5)
    review_text: str = Field(min_length=1)
    matched_segment_text: str = Field(min_length=1)
    positive_similarity: float = Field(ge=-1, le=1)
    negative_similarity: float = Field(ge=-1, le=1)
    relevance_score: float = Field(ge=0, le=1)
    time_weight: float = Field(ge=0, le=1)
    evidence_weight: float = Field(ge=0, le=1)

    @field_validator("review_time")
    @classmethod
    def normalize_review_time(cls, value: datetime) -> datetime:
        return _utc_datetime(value)


class BusinessPreferenceEvidence(StrictModel):
    """一家商家在一条用户偏好下的正反证据与中性化分数。"""

    business_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    positive_evidence: list[RankedReviewEvidence] = Field(max_length=5)
    negative_evidence: list[RankedReviewEvidence] = Field(max_length=5)
    positive_component: float = Field(ge=0, le=1)
    negative_component: float = Field(ge=0, le=1)
    positive_count_reliability: float = Field(ge=0, le=1)
    negative_count_reliability: float = Field(ge=0, le=1)
    evidence_score: float = Field(ge=0, le=1)
    recalled_review_count: int = Field(ge=0)
    ambiguous_review_count: int = Field(ge=0)
    max_positive_similarity: float | None = Field(default=None, ge=-1, le=1)
    max_negative_similarity: float | None = Field(default=None, ge=-1, le=1)
    max_direction_gap: float | None = Field(default=None, ge=0, le=2)


class RankedEvidenceBusiness(StrictModel):
    """偏好、评分和距离融合后的最终一家餐厅。"""

    final_rank: int = Field(ge=1)
    business: BusinessFact
    distance_km: float | None = Field(default=None, ge=0)
    preference_score: float = Field(ge=0, le=1)
    rating_score: float = Field(ge=0, le=1)
    distance_score: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)
    preference_evidence: list[BusinessPreferenceEvidence]


class ReviewRetrievalMetrics(StrictModel):
    """记录评论检索每个主要步骤的真实耗时和数据量。"""

    query_vector_count: int = Field(default=0, ge=0)
    query_description_count: int = Field(default=0, ge=0)
    bm25_enabled: bool = False
    embedding_batch_count: int = Field(default=0, ge=0)
    embedding_latency_ms: float = Field(default=0, ge=0)
    dense_route_latency_ms: float = Field(default=0, ge=0)
    bm25_route_latency_ms: float = Field(default=0, ge=0)
    rrf_fusion_latency_ms: float = Field(default=0, ge=0)
    hybrid_search_wall_latency_ms: float = Field(default=0, ge=0)
    dense_segment_hit_count: int = Field(default=0, ge=0)
    bm25_segment_hit_count: int = Field(default=0, ge=0)
    bm25_only_segment_hit_count: int = Field(default=0, ge=0)
    first_pass_search_latency_ms: float = Field(default=0, ge=0)
    middle_pass_search_latency_ms: float = Field(default=0, ge=0)
    final_pass_search_latency_ms: float = Field(default=0, ge=0)
    local_vector_load_latency_ms: float = Field(default=0, ge=0)
    full_review_load_latency_ms: float = Field(default=0, ge=0)
    first_pass_segment_hit_count: int = Field(default=0, ge=0)
    middle_pass_segment_hit_count: int = Field(default=0, ge=0)
    final_pass_segment_hit_count: int = Field(default=0, ge=0)
    middle_pass_business_count: int = Field(default=0, ge=0)
    final_pass_business_count: int = Field(default=0, ge=0)
    loaded_vector_count: int = Field(default=0, ge=0)
    requirement_segment_relation_count: int = Field(default=0, ge=0)
    unique_segment_count: int = Field(default=0, ge=0)
    full_review_count: int = Field(default=0, ge=0)


class ReviewEvidenceRankingResult(StrictModel):
    """新版评论证据排序的完整输出。"""

    status: Literal["success", "description_failure", "retrieval_failure"]
    hard_filtered_count: int = Field(ge=0)
    requirements: list[PreferenceSearchDescription] = Field(default_factory=list)
    ranking: list[RankedEvidenceBusiness] = Field(default_factory=list, max_length=5)
    recall_threshold: float = Field(ge=-1, le=1)
    acceptance_threshold: float = Field(ge=-1, le=1)
    direction_margin: float = Field(ge=0, le=2)
    formula: str = Field(min_length=1)
    model_call_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    description_latency_ms: float = Field(default=0, ge=0)
    description_warning: str | None = Field(default=None, max_length=500)
    scoring_latency_ms: float = Field(default=0, ge=0)
    retrieval_metrics: ReviewRetrievalMetrics | None = None
    failure_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status == "success":
            if self.failure_reason is not None:
                raise ValueError("successful ranking cannot have a failure reason")
            ranks = [item.final_rank for item in self.ranking]
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("final evidence ranks must be contiguous")
        elif self.failure_reason is None:
            raise ValueError("failed ranking requires a failure reason")
        return self


def _utc_datetime(value: datetime) -> datetime:
    """Yelp 原始时间没有时区标记，数据集语义统一按 UTC 处理。"""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
