"""在固定餐饮类别表中先做本地候选召回，避免每轮把整张表发给大模型。"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import Field

from yelp_agent.models import StrictModel

from .catalog import FixedCategoryCatalog


# 这里只保存用户常见叫法与真实 Yelp 类别的对应关系。类别是否合法仍由
# FixedCategoryCatalog 决定，别名不能凭空创建一个数据库不存在的类别。
_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "Steakhouses": ("牛排", "牛排馆", "扒房", "steak", "steakhouse"),
    "Szechuan": ("川菜", "四川菜", "四川馆子", "麻辣川味", "sichuan", "szechuan"),
    "Cantonese": ("粤菜", "广东菜", "港式粤菜", "cantonese"),
    "Chinese": ("中餐", "中式餐厅", "中国菜", "chinese"),
    "Japanese": ("日料", "日本料理", "日本菜", "japanese"),
    "Sushi Bars": ("寿司", "寿司店", "sushi"),
    "Ramen": ("拉面", "日式拉面", "ramen"),
    "Izakaya": ("居酒屋", "izakaya"),
    "Teppanyaki": ("铁板烧", "teppanyaki"),
    "Korean": ("韩餐", "韩国料理", "韩国菜", "korean"),
    "Thai": ("泰餐", "泰国菜", "thai"),
    "Vietnamese": ("越南菜", "越南粉", "vietnamese"),
    "Indian": ("印度菜", "印度餐", "indian"),
    "Italian": ("意餐", "意大利菜", "italian"),
    "French": ("法餐", "法国菜", "french"),
    "Spanish": ("西班牙菜", "西班牙餐", "spanish"),
    "Mexican": ("墨西哥菜", "墨西哥餐", "mexican"),
    "Mediterranean": ("地中海菜", "地中海餐", "mediterranean"),
    "Middle Eastern": ("中东菜", "中东餐", "middle eastern"),
    "American (New)": ("新派美餐", "新美式", "new american"),
    "American (Traditional)": ("传统美餐", "美式餐厅", "traditional american"),
    "Barbeque": ("烧烤", "烤肉", "barbecue", "bbq"),
    "Hot Pot": ("火锅", "涮锅", "hot pot"),
    "Seafood": ("海鲜", "海鲜餐厅", "seafood"),
    "Pizza": ("披萨", "比萨", "pizza"),
    "Burgers": ("汉堡", "汉堡店", "burger"),
    "Sandwiches": ("三明治", "sandwich"),
    "Noodles": ("面馆", "面条", "noodles"),
    "Breakfast & Brunch": ("早餐", "早午餐", "brunch"),
    "Buffets": ("自助餐", "自助", "buffet"),
    "Fast Food": ("快餐", "fast food"),
    "Cafes": ("咖啡馆", "咖啡厅", "cafe"),
    "Coffee & Tea": ("咖啡", "茶饮", "coffee", "tea"),
    "Bars": ("酒吧", "bar", "bars"),
    "Vegetarian": ("素食", "素食餐厅", "vegetarian"),
    "Vegan": ("纯素", "全素", "vegan"),
    "Halal": ("清真", "halal"),
}

_LATIN_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


class CategorySearchCandidate(StrictModel):
    """本地检索命中的一个真实类别；分数只用于程序排序。"""

    category: str = Field(min_length=1)
    parent_category: str | None = None
    score: float

    def model_payload(self) -> dict[str, str | None]:
        """大模型只需要真实类别和上级，不需要理解 BM25 分数。"""

        return {
            "category": self.category,
            "parent_category": self.parent_category,
        }


class CategoryCandidateSearch:
    """对179个可选类别做毫秒级关键词/BM25召回。"""

    def __init__(self, catalog: FixedCategoryCatalog) -> None:
        self._documents: list[tuple[str, str | None, tuple[str, ...], Counter[str]]] = []
        options = catalog.model_options()
        for option in options:
            category = str(option["category"])
            parent = option["parent_category"]
            aliases = _CATEGORY_ALIASES.get(category, ())
            text_parts = (category, *(aliases or ()), *( [str(parent)] if parent else []))
            terms = Counter(
                term
                for part in text_parts
                for term in _terms(part)
            )
            self._documents.append((category, parent, aliases, terms))
        self._document_frequency = Counter(
            term
            for _, _, _, terms in self._documents
            for term in terms
        )
        self._average_length = sum(
            sum(terms.values()) for _, _, _, terms in self._documents
        ) / max(len(self._documents), 1)

    def search(self, query_text: str, *, limit: int = 5) -> list[CategorySearchCandidate]:
        """返回少量相关真实类别；没有类别信号时返回空列表。"""

        if limit < 1:
            raise ValueError("category candidate limit must be positive")
        query_terms = Counter(_terms(query_text))
        if not query_terms:
            return []
        scored: list[CategorySearchCandidate] = []
        for category, parent, aliases, terms in self._documents:
            score = self._bm25(query_terms, terms)
            # 完整别名命中比零散字符重合可靠得多，给它固定加分但仍保留
            # BM25 对多个相关词和上级类别的排序能力。
            if any(_contains_alias(query_text, alias) for alias in aliases):
                score += 12.0
            if _contains_alias(query_text, category):
                score += 12.0
            if score <= 0:
                continue
            scored.append(
                CategorySearchCandidate(
                    category=category,
                    parent_category=parent,
                    score=score,
                )
            )
        return sorted(
            scored,
            key=lambda item: (-item.score, item.category),
        )[:limit]

    def _bm25(self, query: Counter[str], document: Counter[str]) -> float:
        score = 0.0
        document_length = sum(document.values())
        total_documents = len(self._documents)
        k1 = 1.2
        b = 0.75
        for term, query_count in query.items():
            frequency = document.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequency[term]
            inverse_frequency = math.log(
                1 + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(self._average_length, 1e-9)
            )
            score += query_count * inverse_frequency * (
                frequency * (k1 + 1) / denominator
            )
        return score


def _terms(text: str) -> list[str]:
    lowered = text.casefold()
    terms = _LATIN_WORD.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        # 中文没有天然空格，二元和三元片段能让“想吃牛排”与“牛排馆”相遇，
        # 单字只保留在长度为1的叫法中，避免“餐、店”等泛字制造大量误召回。
        if len(run) == 1:
            terms.append(run)
            continue
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            terms.extend(run[index : index + 3] for index in range(len(run) - 2))
    return terms


def _contains_alias(text: str, alias: str) -> bool:
    lowered = text.casefold()
    value = alias.casefold()
    if value.isascii() and value.replace(" ", "").isalnum():
        return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", lowered) is not None
    return value in lowered
