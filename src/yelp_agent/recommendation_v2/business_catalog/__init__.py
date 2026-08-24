"""餐饮商家数据集：从旧商家全集派生，不改动原始数据。"""

from .builder import build_dining_catalog
from .schema import (
    DINING_CATEGORY_SCHEMA,
    DiningCatalogBuildResult,
    DiningCatalogManifest,
)

__all__ = [
    "DINING_CATEGORY_SCHEMA",
    "DiningCatalogBuildResult",
    "DiningCatalogManifest",
    "build_dining_catalog",
]
