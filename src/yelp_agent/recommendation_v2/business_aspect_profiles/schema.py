"""离线商家软偏好画像对后续排序暴露的稳定数据结构。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import AspectField

type EvidenceGroup = Literal["high_degree", "low_degree", "middle_degree"]


class AspectDirection(StrictModel):
    """训练时使用的客观刻度；数值方向不能由在线代码重新猜测。"""

    aspect_id: AspectField
    name_zh: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    lower_value_means: str = Field(min_length=1)
    higher_value_means: str = Field(min_length=1)
    strength_scale: dict[str, str]
    special_rules: list[str]

    @field_validator("strength_scale")
    @classmethod
    def validate_complete_scale(cls, value: dict[str, str]) -> dict[str, str]:
        if list(value) != ["0", "1", "2", "3", "4"]:
            raise ValueError("strength scale must contain ordered levels 0 through 4")
        if any(not meaning.strip() for meaning in value.values()):
            raise ValueError("strength scale meanings must be nonempty")
        return value


class SupportedBusiness(StrictModel):
    """第一版允许进入离线软排序的餐厅。"""

    business_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    selection_index: int = Field(ge=1)


class BusinessAspectScore(StrictModel):
    """一家餐厅在一项固定软偏好上的离线聚合结果。"""

    business_id: str = Field(min_length=1)
    aspect_id: AspectField
    degree: float | None = Field(default=None, ge=0, le=1)
    degree_0_to_100: float | None = Field(default=None, ge=0, le=100)
    degree_level_code: str = Field(min_length=1)
    degree_level_name_zh: str = Field(min_length=1)
    degree_level_meaning: str = Field(min_length=1)
    evidence_sufficiency: float = Field(ge=0, le=1)
    evidence_sufficiency_level: str = Field(min_length=1)
    controversy: float | None = Field(default=None, ge=0, le=1)
    controversy_level: str = Field(min_length=1)
    business_total_review_count: int = Field(ge=0)
    retrieved_candidate_count: int = Field(ge=0)
    model_related_review_count: int = Field(ge=0)
    unique_evidence_user_count: int = Field(ge=0)
    strong_evidence_count: int = Field(ge=0)
    unique_strong_user_count: int = Field(ge=0)
    usable_for_ranking: bool
    ranking_degree: float | None = Field(default=None, ge=0, le=1)
    unusable_reasons: list[str]
    effective_sample_size: float = Field(ge=0)
    evidence_weight_sum: float = Field(ge=0)
    high_retrieval_limit_reached: bool
    low_retrieval_limit_reached: bool

    @model_validator(mode="after")
    def validate_ranking_value(self) -> Self:
        if self.usable_for_ranking:
            if self.degree is None or self.ranking_degree != self.degree:
                raise ValueError("usable score must expose its degree for ranking")
            if self.unusable_reasons:
                raise ValueError("usable score cannot contain unusable reasons")
        else:
            if self.ranking_degree is not None:
                raise ValueError("unusable score must not expose a ranking degree")
            if not self.unusable_reasons:
                raise ValueError("unusable score must explain why it cannot rank")
        return self


class BusinessAspectEvidence(StrictModel):
    """最终回答可以引用的一条真实评论证据。"""

    business_id: str = Field(min_length=1)
    aspect_id: AspectField
    evidence_group: EvidenceGroup
    evidence_rank: int = Field(ge=1)
    review_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    review_time: datetime
    stars: float = Field(ge=1, le=5)
    useful: int = Field(ge=0)
    text: str = Field(min_length=1)
    relevance: Literal[1, 2, 3]
    strength: Literal[0, 1, 2, 3, 4]
    evidence_weight: float = Field(gt=0)


class BusinessAspectProfileManifest(StrictModel):
    """记录服务器来源、导入数量和本地文件校验值。"""

    schema_version: Literal[1] = 1
    profile_version: Literal["1.0.0"] = "1.0.0"
    created_at: datetime
    source_directory: str = Field(min_length=1)
    source_sha256: dict[str, str]
    source_model_input_count: int = Field(ge=1)
    source_valid_output_count: int = Field(ge=0)
    source_invalid_output_count: int = Field(ge=0)
    business_count: int = Field(ge=1)
    aspect_count: int = Field(ge=1)
    score_count: int = Field(ge=1)
    usable_score_count: int = Field(ge=0)
    unusable_score_count: int = Field(ge=0)
    representative_review_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    evidence_counts_by_group: dict[str, int]
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_counts_and_hashes(self) -> Self:
        if self.source_valid_output_count + self.source_invalid_output_count != (
            self.source_model_input_count
        ):
            raise ValueError("source valid and invalid counts must equal input count")
        if self.usable_score_count + self.unusable_score_count != self.score_count:
            raise ValueError("usable and unusable counts must equal score count")
        if sum(self.evidence_counts_by_group.values()) != self.evidence_count:
            raise ValueError("evidence group counts must equal evidence count")
        for hashes in (self.source_sha256, self.output_sha256):
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()
            ):
                raise ValueError("manifest hashes must be lowercase sha256 values")
        return self
