"""教师标注输入、输出和候选记录的数据约束。"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import AspectField


class TeacherModelInput(StrictModel):
    """教师和学生真正看到的同一份输入。"""

    aspect_id: AspectField
    definition: str = Field(min_length=1)
    relevance_scale: dict[str, str]
    strength_scale: dict[str, str]
    special_rules: list[str] = Field(min_length=1)
    review_text: str = Field(min_length=1, max_length=900)

    @field_validator("relevance_scale")
    @classmethod
    def validate_relevance_scale(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != {"0", "1", "2", "3"}:
            raise ValueError("relevance_scale must define levels 0 through 3")
        if any(not text.strip() for text in value.values()):
            raise ValueError("relevance scale descriptions cannot be empty")
        return value

    @field_validator("strength_scale")
    @classmethod
    def validate_strength_scale(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != {"0", "1", "2", "3", "4"}:
            raise ValueError("strength_scale must define levels 0 through 4")
        if any(not text.strip() for text in value.values()):
            raise ValueError("strength scale descriptions cannot be empty")
        return value


class TeacherCandidate(BaseModel):
    """候选文件中的追踪信息；只有model_input会发送给模型。"""

    model_config = ConfigDict(extra="allow")

    sample_id: str = Field(pattern=r"^[a-z][a-z0-9_]+_[0-9]{6}$")
    review_id: str = Field(min_length=1)
    business_id: str = Field(min_length=1)
    selection_relevance: Literal[0, 1, 2, 3]
    selection_strength: Literal[0, 1, 2, 3, 4]
    model_input: TeacherModelInput

    @model_validator(mode="after")
    def validate_aspect_matches_sample(self) -> Self:
        prefix = f"{self.model_input.aspect_id}_"
        if not self.sample_id.startswith(prefix):
            raise ValueError("sample_id must start with the model-input aspect")
        return self


class TeacherLabel(StrictModel):
    """教师唯一允许返回的两个判断结果。"""

    relevance: Literal[0, 1, 2, 3]
    strength: Literal[0, 1, 2, 3, 4] | None

    @model_validator(mode="after")
    def validate_strength_presence(self) -> Self:
        if self.relevance == 0 and self.strength is not None:
            raise ValueError("strength must be null when relevance is 0")
        if self.relevance > 0 and self.strength is None:
            raise ValueError("strength is required when relevance is above 0")
        return self


class LabeledTeacherSample(StrictModel):
    """通过校验后保存的正式教师样本。"""

    sample_id: str = Field(min_length=1)
    model_input: TeacherModelInput
    model_output: TeacherLabel
