"""餐厅专用数据集的稳定结构。"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

# 这里只统计餐厅子集中真实出现过的原始类别；类别的统一名称和上下级关系
# 留给下一步固定类别表处理，避免在筛选餐厅时提前误删稀有菜系。
DINING_CATEGORY_SCHEMA = pa.schema(
    [
        pa.field("category", pa.string(), nullable=False),
        pa.field("business_count", pa.int64(), nullable=False),
        pa.field("business_share", pa.float64(), nullable=False),
        pa.field("is_scope_marker", pa.bool_(), nullable=False),
    ]
)


class DiningCatalogManifest(StrictModel):
    """说明餐厅子集从哪里来、使用了什么规则以及产出了什么。"""

    schema_version: Literal[1] = 1
    catalog_version: Literal["1.0.0"] = "1.0.0"
    selection_category: Literal["Restaurants"] = "Restaurants"
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_business_count: int = Field(ge=1)
    dining_business_count: int = Field(ge=1)
    source_category_count: int = Field(ge=1)
    dining_category_count: int = Field(ge=1)
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_counts_and_outputs(self) -> DiningCatalogManifest:
        if self.dining_business_count > self.source_business_count:
            raise ValueError("dining business count cannot exceed source count")
        if self.dining_category_count > self.source_category_count:
            raise ValueError("dining category count cannot exceed source count")
        if set(self.output_sha256) != {"businesses", "categories"}:
            raise ValueError("manifest must hash businesses and categories")
        if any(
            len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for value in self.output_sha256.values()
        ):
            raise ValueError("output hashes must be lowercase sha256 values")
        return self


class DiningCatalogBuildResult(StrictModel):
    """一次构建的结果，方便命令行、测试和后续流程共同读取。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: DiningCatalogManifest
