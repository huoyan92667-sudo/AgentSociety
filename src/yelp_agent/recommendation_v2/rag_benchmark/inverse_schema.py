"""逆向评论召回评测集的固定数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import AspectField


class InverseGenerationCandidate(StrictModel):
    """交给生成模型的一条真实评论候选；星级不会进入这个结构。"""

    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    review_text: str = Field(min_length=20, max_length=1800)


class InverseGeneratedDraft(StrictModel):
    """GLM从一条真实评论反推出的问题、方向和逐字证据。"""

    seed_review_id: str = Field(min_length=1)
    query_text: str = Field(min_length=6, max_length=300)
    requirement_text: str = Field(min_length=2, max_length=200)
    expected_direction: Literal["positive", "negative"]
    evidence_span: str = Field(min_length=5, max_length=500)

    @field_validator("query_text")
    @classmethod
    def require_chinese_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not any("\u4e00" <= char <= "\u9fff" for char in cleaned):
            raise ValueError("query_text must be natural Chinese")
        return cleaned


class InverseGenerationProposal(StrictModel):
    drafts: list[InverseGeneratedDraft] = Field(min_length=1, max_length=20)


class InverseBenchmarkCase(StrictModel):
    """程序校验后保存的一条逆向评测正确答案。"""

    schema_version: Literal[1] = 1
    case_id: str = Field(min_length=1, max_length=120)
    dataset_kind: Literal["fixed_aspect", "long_tail"]
    aspect: AspectField | None = None
    target_direction: Literal["higher", "lower"] | None = None
    query_text: str = Field(min_length=6, max_length=300)
    requirement_text: str = Field(min_length=2, max_length=200)
    expected_direction: Literal["positive", "negative"]
    seed_review_id: str = Field(min_length=1)
    seed_business_id: str = Field(min_length=1)
    seed_segment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_span: str = Field(min_length=5, max_length=500)
    evidence_context: str = Field(min_length=5, max_length=2000)
    review_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["test"] = "test"
    verification_status: Literal["glm_generated_program_validated"] = (
        "glm_generated_program_validated"
    )
    teacher_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
