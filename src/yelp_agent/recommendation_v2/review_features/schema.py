"""评论候选召回的固定输出结构。"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

CANDIDATE_REVIEW_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("review_time", pa.timestamp("us"), nullable=False),
        pa.field("stars", pa.float64(), nullable=False),
        pa.field("useful", pa.int64(), nullable=False),
        pa.field("review_text", pa.string(), nullable=False),
        pa.field("review_text_sha256", pa.string(), nullable=False),
    ]
)


CANDIDATE_ASPECT_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("aspect", pa.string(), nullable=False),
        pa.field("keyword_hit", pa.bool_(), nullable=False),
        pa.field("semantic_hit", pa.bool_(), nullable=False),
        pa.field(
            "matched_terms",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("semantic_score", pa.float32(), nullable=True),
        pa.field(
            "matched_anchor_ids",
            pa.list_(pa.string()),
            nullable=False,
        ),
    ]
)


class KeywordRecallManifest(StrictModel):
    """记录旧词表粗筛的来源、数量和生成文件校验值。"""

    schema_version: Literal[1] = 1
    recall_version: Literal["1.0.0"] = "1.0.0"
    source_paths: dict[str, str]
    source_sha256: dict[str, str]
    definitions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_count: int = Field(ge=1)
    source_review_count: int = Field(ge=1)
    candidate_review_count: int = Field(ge=1)
    candidate_aspect_count: int = Field(ge=1)
    aspect_counts: dict[str, int]
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_counts(self) -> KeywordRecallManifest:
        if sum(self.aspect_counts.values()) != self.candidate_aspect_count:
            raise ValueError("aspect counts must sum to candidate_aspect_count")
        if self.candidate_review_count > self.source_review_count:
            raise ValueError("candidate reviews cannot exceed source reviews")
        return self


class KeywordRecallBuildResult(StrictModel):
    """一次旧词表粗筛的公开结果。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: KeywordRecallManifest


class SemanticRecallManifest(StrictModel):
    """记录按商家、按特征进行意思相近粗筛的来源和数量。"""

    schema_version: Literal[1] = 1
    recall_version: Literal["1.0.0"] = "1.0.0"
    source_paths: dict[str, str]
    source_sha256: dict[str, str]
    definitions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    top_reviews_per_business_aspect: int = Field(ge=1)
    business_count: int = Field(ge=1)
    candidate_aspect_count: int = Field(ge=1)
    aspect_counts: dict[str, int]
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_semantic_counts(self) -> SemanticRecallManifest:
        if sum(self.aspect_counts.values()) != self.candidate_aspect_count:
            raise ValueError("semantic aspect counts must sum to candidate count")
        return self


class SemanticRecallBuildResult(StrictModel):
    """一次意思相近评论粗筛的公开结果。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: SemanticRecallManifest


class CandidateRecallManifest(StrictModel):
    """记录旧词语与意思相近两路合并后的最终候选。"""

    schema_version: Literal[1] = 1
    recall_version: Literal["1.0.0"] = "1.0.0"
    source_paths: dict[str, str]
    source_sha256: dict[str, str]
    candidate_review_count: int = Field(ge=1)
    candidate_aspect_count: int = Field(ge=1)
    route_counts: dict[str, int]
    aspect_counts: dict[str, int]
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_merged_counts(self) -> CandidateRecallManifest:
        if sum(self.route_counts.values()) != self.candidate_aspect_count:
            raise ValueError("route counts must sum to merged candidate count")
        if sum(self.aspect_counts.values()) != self.candidate_aspect_count:
            raise ValueError("aspect counts must sum to merged candidate count")
        return self


class CandidateRecallBuildResult(StrictModel):
    """两路评论候选合并的公开结果。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: CandidateRecallManifest
