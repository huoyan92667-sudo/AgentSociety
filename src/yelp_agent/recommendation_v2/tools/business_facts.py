"""让大模型按商家编号读取新版统一商家事实。"""

from __future__ import annotations

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
