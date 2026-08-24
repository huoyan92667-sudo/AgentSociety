"""固定餐饮类别表：为大模型选择和后续数据库过滤提供同一份事实。"""

from .builder import build_fixed_category_catalog
from .catalog import FixedCategoryCatalog, load_fixed_category_catalog
from .schema import (
    FIXED_CATEGORY_SCHEMA,
    FixedCategoryBuildResult,
    FixedCategoryManifest,
    FixedCategoryRow,
)

__all__ = [
    "FIXED_CATEGORY_SCHEMA",
    "FixedCategoryBuildResult",
    "FixedCategoryCatalog",
    "FixedCategoryManifest",
    "FixedCategoryRow",
    "build_fixed_category_catalog",
    "load_fixed_category_catalog",
]
