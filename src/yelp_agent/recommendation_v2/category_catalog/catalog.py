"""运行时只通过这个小入口读取模型可选类别和上下级关系。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .builder import DEFAULT_OUTPUT_ROOT
from .schema import FixedCategoryDocument, FixedCategoryRow


class FixedCategoryCatalog:
    """隐藏文件位置、排除项和树遍历，调用方只处理真实类别名。"""

    def __init__(self, document: FixedCategoryDocument) -> None:
        self._document = document
        self._by_name = {item.category: item for item in document.categories}
        self._children: dict[str, list[str]] = {}
        for item in document.categories:
            if item.parent_is_category:
                self._children.setdefault(item.parent, []).append(item.category)

    @classmethod
    def from_file(cls, path: str | Path) -> FixedCategoryCatalog:
        document = FixedCategoryDocument.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
        return cls(document)

    def model_options(self) -> tuple[dict[str, str | None], ...]:
        """返回大模型可以选择的真实类别和真实上级，不包含用户说法。"""

        return tuple(
            {
                "category": item.category,
                "parent_category": item.parent if item.parent_is_category else None,
            }
            for item in self._document.categories
            if item.selectable
        )

    def is_selectable(self, category: str) -> bool:
        item = self._by_name.get(category)
        return item is not None and item.selectable

    def get(self, category: str) -> FixedCategoryRow:
        try:
            return self._by_name[category]
        except KeyError as exc:
            raise KeyError(f"unknown dining category: {category}") from exc

    def expand_for_filter(self, category: str) -> tuple[str, ...]:
        """选择上级类别时返回它自身及所有真实下级，供数据库条件使用。"""

        item = self.get(category)
        if not item.selectable:
            raise ValueError(f"category is not selectable: {category}")
        result: list[str] = []
        pending = [category]
        while pending:
            current = pending.pop(0)
            if current in result:
                continue
            result.append(current)
            pending.extend(sorted(self._children.get(current, [])))
        return tuple(result)


@lru_cache(maxsize=1)
def load_fixed_category_catalog() -> FixedCategoryCatalog:
    """从 recommendation_v2 内部的固定位置读取一份共享类别表。"""

    return FixedCategoryCatalog.from_file(DEFAULT_OUTPUT_ROOT / "catalog.json")
