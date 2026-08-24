"""后续过滤和排序只通过这个入口读取商家基础事实。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from .builder import DEFAULT_OUTPUT_ROOT
from .schema import BUSINESS_FACT_SCHEMA, BusinessFact, PriceBand, PriceBandDocument


class BusinessFactCatalog:
    """隐藏文件校验和行转换，调用方只处理商家编号和明确事实。"""

    def __init__(
        self,
        facts: list[BusinessFact],
        price_bands: PriceBandDocument,
        fact_path: Path,
    ) -> None:
        self._by_id = {item.business_id: item for item in facts}
        self._price_by_level = {
            item.price_level: item for item in price_bands.bands
        }
        self._fact_path = fact_path.resolve()

    @classmethod
    def from_files(
        cls,
        fact_path: str | Path,
        price_band_path: str | Path,
    ) -> BusinessFactCatalog:
        facts_source = Path(fact_path)
        parquet = pq.ParquetFile(facts_source)
        if not parquet.schema_arrow.equals(BUSINESS_FACT_SCHEMA, check_metadata=False):
            raise ValueError("business fact file has an unexpected schema")
        facts = [BusinessFact.model_validate(row) for row in parquet.read().to_pylist()]
        if len({item.business_id for item in facts}) != len(facts):
            raise ValueError("business fact IDs must be unique")
        price_bands = PriceBandDocument.model_validate_json(
            Path(price_band_path).read_text(encoding="utf-8")
        )
        return cls(facts, price_bands, facts_source)

    @property
    def fact_path(self) -> Path:
        """数据库过滤工具可直接读取这一个已经校验过的文件。"""

        return self._fact_path

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> tuple[BusinessFact, ...]:
        """返回稳定排序的全部餐厅事实，供地点计算和过滤工具共同使用。"""

        return tuple(self._by_id[key] for key in sorted(self._by_id))

    def contains(self, business_id: str) -> bool:
        """判断商家编号是否属于当前餐饮数据范围。"""

        return business_id in self._by_id

    def get(self, business_id: str) -> BusinessFact:
        try:
            return self._by_id[business_id]
        except KeyError as exc:
            raise KeyError(f"unknown restaurant business ID: {business_id}") from exc

    def price_band(self, price_level: int) -> PriceBand:
        try:
            return self._price_by_level[price_level]
        except KeyError as exc:
            raise KeyError(f"unknown price level: {price_level}") from exc


@lru_cache(maxsize=1)
def load_business_fact_catalog() -> BusinessFactCatalog:
    """从 recommendation_v2 内部固定位置读取共享商家事实。"""

    return BusinessFactCatalog.from_files(
        DEFAULT_OUTPUT_ROOT / "business_facts.parquet",
        DEFAULT_OUTPUT_ROOT / "price_bands.json",
    )
