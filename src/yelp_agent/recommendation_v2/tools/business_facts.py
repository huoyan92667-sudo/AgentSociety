"""让大模型按商家编号读取新版统一商家事实。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import (
    BusinessFact,
    BusinessFactCatalog,
)


class BusinessFactsQuery(StrictModel):
    """一次最多读取五家，防止把整个商家目录塞进模型上下文。"""

    business_ids: list[str] = Field(min_length=1, max_length=5)

    @field_validator("business_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("business IDs must be unique")
        if any(not item or item != item.strip() for item in values):
            raise ValueError("business IDs must be nonempty and trimmed")
        return values


class BusinessFactsObservation(StrictModel):
    """返回找到的完整事实，同时明确列出不属于当前餐饮目录的编号。"""

    status: Literal["found", "partial", "not_found"]
    businesses: list[BusinessFact] = Field(default_factory=list, max_length=5)
    missing_business_ids: list[str] = Field(default_factory=list, max_length=5)


class BusinessFactsTool:
    """复用新版商家事实目录，不再返回旧 Agent 的另一套字段。"""

    name = "lookup_business_facts"

    def __init__(self, catalog: BusinessFactCatalog) -> None:
        self._catalog = catalog

    def execute(self, query: BusinessFactsQuery) -> BusinessFactsObservation:
        businesses = [
            self._catalog.get(business_id).model_copy(deep=True)
            for business_id in query.business_ids
            if self._catalog.contains(business_id)
        ]
        missing = [
            business_id
            for business_id in query.business_ids
            if not self._catalog.contains(business_id)
        ]
        status: Literal["found", "partial", "not_found"]
        if not businesses:
            status = "not_found"
        elif missing:
            status = "partial"
        else:
            status = "found"
        return BusinessFactsObservation(
            status=status,
            businesses=businesses,
            missing_business_ids=missing,
        )


class BusinessNameSearchQuery(StrictModel):
    """按用户给出的店名查真实商家编号；地点提示用于消除同名店歧义。"""

    name: str = Field(min_length=1, max_length=200)
    city: str | None = Field(default=None, min_length=1, max_length=100)
    state: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=5)

    @field_validator("name", "city", "state")
    @classmethod
    def trim_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("business search text cannot be blank")
        return cleaned


class BusinessNameMatch(StrictModel):
    """一个名称候选；只返回主模型消歧真正需要的少量事实。"""

    business_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    categories: list[str]
    match_kind: Literal["exact", "contains", "similar"]


class BusinessNameSearchObservation(StrictModel):
    """名称查找结果；多条候选时由主模型结合对话决定或询问用户。"""

    status: Literal["resolved", "ambiguous", "not_found"]
    matches: list[BusinessNameMatch] = Field(default_factory=list, max_length=5)


class BusinessNameSearchTool:
    """在统一商家事实中解析名称，不替主模型决定用户想查询什么。"""

    name = "search_restaurant_businesses"

    def __init__(self, catalog: BusinessFactCatalog) -> None:
        self._catalog = catalog

    def execute(self, query: BusinessNameSearchQuery) -> BusinessNameSearchObservation:
        wanted = _normalize_business_name(query.name)
        if not wanted:
            return BusinessNameSearchObservation(status="not_found")

        candidates: list[tuple[int, float, BusinessFact, str]] = []
        for business in self._catalog.all():
            if query.city and business.city.casefold() != query.city.casefold():
                continue
            if query.state and business.state.casefold() != query.state.casefold():
                continue
            actual = _normalize_business_name(business.name)
            if actual == wanted:
                kind = "exact"
                group = 3
                similarity = 1.0
            elif wanted in actual or actual in wanted:
                kind = "contains"
                group = 2
                similarity = min(len(wanted), len(actual)) / max(len(wanted), len(actual))
            else:
                similarity = SequenceMatcher(None, wanted, actual).ratio()
                if similarity < 0.72:
                    continue
                kind = "similar"
                group = 1
            candidates.append((group, similarity, business, kind))

        # 精确命中存在时，近似名称没有继续进入上下文的价值；但多家同名店
        # 必须全部保留，交给主模型结合城市、地址或用户补充信息消歧。
        best_group = max((item[0] for item in candidates), default=0)
        strongest = [item for item in candidates if item[0] == best_group]
        ordered = sorted(
            strongest,
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2].review_count,
                item[2].business_id,
            ),
        )[: query.limit]
        matches = [
            BusinessNameMatch(
                business_id=business.business_id,
                name=business.name,
                address=business.address,
                city=business.city,
                state=business.state,
                categories=list(business.categories),
                match_kind=kind,  # type: ignore[arg-type]
            )
            for _, _, business, kind in ordered
        ]
        if not matches:
            status: Literal["resolved", "ambiguous", "not_found"] = "not_found"
        elif len(matches) == 1:
            status = "resolved"
        else:
            status = "ambiguous"
        return BusinessNameSearchObservation(status=status, matches=matches)


def _normalize_business_name(value: str) -> str:
    """忽略大小写、空白和标点，但保留各语言中的字母与数字。"""

    return "".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))
