"""餐饮评论片段文字向量的固定结构和生成记录。"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.review_rag.schema import REVIEW_SEGMENT_SCHEMA

REVIEW_INDEX_SEGMENT_SCHEMA = pa.schema(
    [
        pa.field("row_index", pa.int64(), nullable=False),
        *REVIEW_SEGMENT_SCHEMA,
    ]
)


class ReviewVectorIndexManifest(StrictModel):
    """记录评论片段与向量的严格一一对应关系。"""

    schema_version: Literal[1] = 1
    index_version: Literal["1.0.0"] = "1.0.0"
    source_paths: dict[str, str]
    source_sha256: dict[str, str]
    model: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    vector_dtype: Literal["float16"] = "float16"
    business_count: int = Field(ge=1)
    review_count: int = Field(ge=1)
    segment_count: int = Field(ge=1)
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_outputs(self) -> ReviewVectorIndexManifest:
        if set(self.output_sha256) != {"segments", "embeddings"}:
            raise ValueError("review vector index must hash segments and embeddings")
        return self


class ReviewVectorIndexBuildResult(StrictModel):
    """一次评论文字向量构建的公开结果。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: ReviewVectorIndexManifest
