"""根据统一搜索中心计算候选餐厅的真实直线距离。"""

from __future__ import annotations

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.business_facts import BusinessFactCatalog
from yelp_agent.recommendation_v2.schema import GeoPoint, SearchCenter


class BusinessDistance(StrictModel):
    """一家餐厅相对本轮搜索中心的距离事实。"""

    business_id: str = Field(min_length=1)
    distance_km: float = Field(ge=0)


class GeographicDistanceResult(StrictModel):
    """地点工具的一次完整输出，可直接交给硬过滤和后续距离排序。"""

    search_center: SearchCenter
    source_business_count: int = Field(ge=0)
    distances: list[BusinessDistance]

    @field_validator("distances")
    @classmethod
    def validate_unique_businesses(
        cls,
        values: list[BusinessDistance],
    ) -> list[BusinessDistance]:
        ids = [item.business_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("geographic distance business IDs must be unique")
        return values

    def distance_by_business_id(self) -> dict[str, float]:
        """转成过滤工具读取方便的商家编号到距离映射。"""

        return {item.business_id: item.distance_km for item in self.distances}


class GeographicDistanceTool:
    """读取商家经纬度，并以统一搜索中心为起点计算球面距离。"""

    name = "calculate_business_distances"

    def __init__(self, catalog: BusinessFactCatalog) -> None:
        self._catalog = catalog

    def execute(
        self,
        search_center: SearchCenter,
        candidate_business_ids: list[str] | None = None,
    ) -> GeographicDistanceResult:
        """计算指定候选；不指定候选时计算全部餐厅。"""

        if candidate_business_ids is None:
            facts = list(self._catalog.all())
        else:
            if len(candidate_business_ids) != len(set(candidate_business_ids)):
                raise ValueError("candidate business IDs must be unique")
            facts = [self._catalog.get(item) for item in candidate_business_ids]

        center = search_center.location
        distances = [
            BusinessDistance(
                business_id=fact.business_id,
                distance_km=center.distance_km_to(
                    GeoPoint(latitude=fact.latitude, longitude=fact.longitude)
                ),
            )
            for fact in facts
        ]
        distances.sort(key=lambda item: item.business_id)
        return GeographicDistanceResult(
            search_center=search_center.model_copy(deep=True),
            source_business_count=len(facts),
            distances=distances,
        )
