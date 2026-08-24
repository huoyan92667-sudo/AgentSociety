"""查询历史推荐商家真实快照的通用工具。"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import Field, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.schema import (
    AspectField,
    BusinessReference,
    GeoPoint,
)

type HistoryFactField = Literal[
    "business_id",
    "business_name",
    "location",
    "distance_km",
    "price_level",
    "categories",
    "aspect_scores",
]


class HistoryFactQuery(StrictModel):
    """大模型为了理解用户指代而发起的一次事实查询。"""

    turn_index: int | None = Field(default=None, ge=1)
    position: int | None = Field(default=None, ge=1, le=100)
    business_id: str | None = Field(default=None, min_length=1, max_length=200)
    fields: list[HistoryFactField] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def validate_locator(self) -> Self:
        """位置和商家编号二选一；不写轮次时默认查最近一次展示。"""

        if (self.position is None) == (self.business_id is None):
            raise ValueError("use exactly one of position or business_id")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("history fact fields must be unique")
        return self


class HistoryBusinessFact(StrictModel):
    """工具从历史快照中查到的一份可核验商家事实。"""

    fact_id: str = Field(pattern=r"^history_business\.[a-f0-9]{16}$")
    presented_turn_index: int = Field(ge=1)
    position: int = Field(ge=1, le=100)
    business_id: str = Field(min_length=1, max_length=200)
    business_name: str = Field(min_length=1, max_length=500)
    requested_fields: list[HistoryFactField]
    missing_fields: list[HistoryFactField] = Field(default_factory=list)
    location: GeoPoint | None = None
    distance_km: float | None = Field(default=None, ge=0)
    price_level: int | None = Field(default=None, ge=1, le=4)
    categories: list[str] | None = None
    aspect_scores: dict[AspectField, float] | None = None


class HistoryFactObservation(StrictModel):
    """一次工具执行结果；找不到时也返回结构化原因供大模型继续处理。"""

    status: Literal["found", "not_found"]
    fact: HistoryBusinessFact | None = None
    reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status == "found":
            if self.fact is None or self.reason is not None:
                raise ValueError("found observation requires a fact only")
        elif self.fact is not None or self.reason is None:
            raise ValueError("not-found observation requires a reason only")
        return self


class HistoryBusinessFactTool:
    """按展示轮次、位置或商家编号读取事实，不解释用户语言。"""

    name = "lookup_history_business"

    def execute(
        self,
        query: HistoryFactQuery,
        businesses: list[BusinessReference],
    ) -> HistoryFactObservation:
        """查找最合适的历史商家，并只返回大模型明确请求的字段。"""

        candidates = [
            item
            for item in businesses
            if (query.turn_index is None or item.presented_turn_index == query.turn_index)
            and (query.position is None or item.position == query.position)
            and (query.business_id is None or item.business_id == query.business_id)
        ]
        if not candidates:
            return HistoryFactObservation(
                status="not_found",
                reason="历史展示记录中没有符合轮次和位置或商家编号的记录",
            )
        # 没指定轮次时，指代默认落到最近一次展示，符合多轮对话的自然含义。
        selected = max(
            candidates,
            key=lambda item: (item.presented_turn_index, item.position),
        )
        fact = self._build_fact(query, selected)
        return HistoryFactObservation(status="found", fact=fact)

    @staticmethod
    def _build_fact(
        query: HistoryFactQuery,
        business: BusinessReference,
    ) -> HistoryBusinessFact:
        """从已经保存的商家快照中复制事实，并标明缺失字段。"""

        missing: list[HistoryFactField] = []
        for field_name in query.fields:
            value = getattr(business, field_name)
            if value is None or value == [] or value == {}:
                missing.append(field_name)
        identity = json.dumps(
            {
                "turn": business.presented_turn_index,
                "position": business.position,
                "business_id": business.business_id,
                "fields": sorted(query.fields),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        requested = set(query.fields)
        return HistoryBusinessFact(
            fact_id=f"history_business.{digest}",
            presented_turn_index=business.presented_turn_index,
            position=business.position,
            business_id=business.business_id,
            business_name=business.business_name,
            requested_fields=query.fields,
            missing_fields=missing,
            location=(
                business.location.model_copy(deep=True)
                if "location" in requested and business.location is not None
                else None
            ),
            distance_km=(
                business.distance_km if "distance_km" in requested else None
            ),
            price_level=(
                business.price_level if "price_level" in requested else None
            ),
            categories=(
                list(business.categories) if "categories" in requested else None
            ),
            aspect_scores=(
                dict(business.aspect_scores)
                if "aspect_scores" in requested
                else None
            ),
        )
