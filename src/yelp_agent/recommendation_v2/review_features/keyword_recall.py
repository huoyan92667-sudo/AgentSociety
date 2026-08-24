"""用旧词表从完整评论中召回可能相关的14种特征。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .definitions import AspectRecallDefinition


def _term_pattern(terms: Iterable[str]) -> str:
    alternatives = sorted(
        (re.escape(term) for term in terms),
        key=len,
        reverse=True,
    )
    return rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)"


class KeywordAspectMatcher:
    """扫描一条完整评论，返回每个命中特征及实际命中的旧词语。"""

    def __init__(self, definitions: Iterable[AspectRecallDefinition]) -> None:
        values = tuple(definitions)
        if not values:
            raise ValueError("definitions cannot be empty")
        alternatives = [
            f"(?P<{item.aspect}>{_term_pattern(item.keyword_terms)})"
            for item in values
        ]
        self._pattern = re.compile("|".join(alternatives), flags=re.IGNORECASE)
        self._aspect_order = {
            item.aspect: index for index, item in enumerate(values)
        }

    def match(self, review_text: str) -> dict[str, list[str]]:
        """只判断可能涉及哪些特征；不根据旧正负词产生方向或分数。"""

        result: dict[str, set[str]] = {}
        for match in self._pattern.finditer(review_text):
            aspect = match.lastgroup
            if aspect is None:
                continue
            result.setdefault(aspect, set()).add(match.group(0).casefold())
        return {
            aspect: sorted(terms)
            for aspect, terms in sorted(
                result.items(), key=lambda item: self._aspect_order[item[0]]
            )
        }
