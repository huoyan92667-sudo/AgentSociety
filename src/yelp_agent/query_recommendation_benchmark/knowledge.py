"""Adapter from frozen Yelp stores to cutoff-safe Query frame knowledge."""

from __future__ import annotations

import math

from yelp_agent.business_profiles import BusinessKnowledgeStore
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.reviews.schema import AspectName

from .frames import AnchorQueryKnowledge
from .selection import BehaviorAnchor

_QUERY_POSITIVE_ASPECTS: tuple[AspectName, ...] = (
    "food_quality",
    "service",
    "price_value",
    "quiet_environment",
    "parking",
    "pet_friendly",
    "family_friendly",
    "date_suitable",
    "group_suitable",
    "cleanliness",
)


def _haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius = 6371.0088
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(value))


def _price_level(attributes: dict[str, object]) -> int | None:
    raw = attributes.get("RestaurantsPriceRange2")
    try:
        value = int(str(raw).strip().strip("'\""))
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 4 else None


class YelpAnchorKnowledgeReader:
    """Hide static, location, and Aspect lookup behind one frame-planning seam."""

    def __init__(
        self,
        data_view: TemporalDataView,
        business_knowledge: BusinessKnowledgeStore,
        *,
        broad_categories: set[str],
    ) -> None:
        self._data = data_view
        self._knowledge = business_knowledge
        self._broad_categories = {value.casefold() for value in broad_categories}
        if business_knowledge.source_scope != "selected_user_interactions":
            raise ValueError("Benchmark Aspect evidence must use selected users")

    def describe(self, anchor: BehaviorAnchor) -> AnchorQueryKnowledge:
        target = self._data.business(anchor.target_business_id)
        fine_categories = tuple(
            sorted(
                category
                for category in target.categories
                if category.casefold() not in self._broad_categories
            )
        )
        if not fine_categories:
            raise ValueError(
                f"Target has no fine-grained category: {anchor.target_business_id}"
            )
        history = [self._data.business(value) for value in anchor.history_business_ids]
        coordinates = [
            (float(item.latitude), float(item.longitude))
            for item in history
            if item.latitude is not None and item.longitude is not None
        ]
        user_latitude = (
            sum(item[0] for item in coordinates) / len(coordinates)
            if coordinates
            else None
        )
        user_longitude = (
            sum(item[1] for item in coordinates) / len(coordinates)
            if coordinates
            else None
        )
        target_distance = (
            _haversine_km(
                user_latitude,
                user_longitude,
                float(target.latitude),
                float(target.longitude),
            )
            if user_latitude is not None
            and user_longitude is not None
            and target.latitude is not None
            and target.longitude is not None
            else None
        )
        summaries = self._knowledge.get_aspects(
            [anchor.target_business_id],
            list(_QUERY_POSITIVE_ASPECTS),
            anchor.cutoff_time,
        )[anchor.target_business_id]
        positive_aspects = tuple(
            aspect
            for aspect in _QUERY_POSITIVE_ASPECTS
            if summaries[aspect].status == "known"
            and float(summaries[aspect].weighted_positive_ratio) > 0.5
        )
        return AnchorQueryKnowledge(
            business_id=anchor.target_business_id,
            fine_categories=fine_categories,
            is_food_business=bool(
                {category.casefold() for category in target.categories}
                .intersection({"food", "restaurants"})
            ),
            price_level=_price_level(target.attributes_dict()),
            user_latitude=user_latitude,
            user_longitude=user_longitude,
            target_distance_km=target_distance,
            positive_aspects=positive_aspects,
            aspect_source_scope="selected_user_interactions",
        )
