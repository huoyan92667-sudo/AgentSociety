"""固定类别表、上下级分组和生成记录的数据结构。"""

from __future__ import annotations

from typing import Literal, Self

import pyarrow as pa
from pydantic import Field, model_validator

from yelp_agent.models import StrictModel

type CategoryKind = Literal[
    "cuisine",
    "dish",
    "venue",
    "dietary",
    "generic",
    "excluded",
]


FIXED_CATEGORY_SCHEMA = pa.schema(
    [
        pa.field("category", pa.string(), nullable=False),
        pa.field("business_count", pa.int64(), nullable=False),
        pa.field("business_share", pa.float64(), nullable=False),
        pa.field("selectable", pa.bool_(), nullable=False),
        pa.field("category_kind", pa.string(), nullable=False),
        pa.field("parent", pa.string(), nullable=False),
        pa.field("parent_is_category", pa.bool_(), nullable=False),
    ]
)


class CategoryGroup(StrictModel):
    """虚拟上级分组；大模型只能选择真实类别，不能选择这些分组。"""

    group_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,60}$")
    label: str = Field(min_length=1, max_length=100)
    parent_group_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{2,60}$",
    )


class FixedCategoryRow(StrictModel):
    """319种真实类别中的一行，没有用户说法或模型生成内容。"""

    category: str = Field(min_length=1, max_length=100)
    business_count: int = Field(ge=1)
    business_share: float = Field(gt=0, le=1)
    selectable: bool
    category_kind: CategoryKind
    parent: str = Field(min_length=1, max_length=100)
    parent_is_category: bool

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.selectable and self.category_kind in {"generic", "excluded"}:
            raise ValueError("generic and excluded categories cannot be selected")
        if not self.selectable and self.category_kind not in {"generic", "excluded"}:
            raise ValueError("real dining categories must be selectable")
        if self.parent_is_category and self.parent == self.category:
            raise ValueError("a category cannot be its own parent")
        return self


class FixedCategoryDocument(StrictModel):
    """完整固定表；既便于人查看，也便于运行时一次读取。"""

    schema_version: Literal[1] = 1
    taxonomy_version: Literal["1.0.1"] = "1.0.1"
    groups: list[CategoryGroup]
    categories: list[FixedCategoryRow]

    @model_validator(mode="after")
    def validate_tree(self) -> Self:
        group_ids = {item.group_id for item in self.groups}
        if len(group_ids) != len(self.groups):
            raise ValueError("category group IDs must be unique")
        category_names = {item.category for item in self.categories}
        if len(category_names) != len(self.categories):
            raise ValueError("fixed categories must be unique")
        for group in self.groups:
            if (
                group.parent_group_id is not None
                and group.parent_group_id not in group_ids
            ):
                raise ValueError("category group parent must exist")
        for row in self.categories:
            if row.parent_is_category:
                if row.parent not in category_names:
                    raise ValueError("category parent must be a real category")
            elif row.parent not in group_ids:
                raise ValueError("category group parent must exist")
        return self


class FixedCategoryManifest(StrictModel):
    """记录固定表输入、数量和文件校验值。"""

    schema_version: Literal[1] = 1
    taxonomy_version: Literal["1.0.1"] = "1.0.1"
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_category_count: int = Field(ge=1)
    selectable_category_count: int = Field(ge=1)
    non_selectable_category_count: int = Field(ge=1)
    output_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if (
            self.selectable_category_count + self.non_selectable_category_count
            != self.source_category_count
        ):
            raise ValueError("fixed category counts must cover the source inventory")
        if set(self.output_sha256) != {"catalog", "categories"}:
            raise ValueError("manifest must hash catalog and categories")
        return self


class FixedCategoryBuildResult(StrictModel):
    """固定类别表的一次构建结果。"""

    status: Literal["written", "skipped"]
    output_root: str = Field(min_length=1)
    manifest: FixedCategoryManifest
