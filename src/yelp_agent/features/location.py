"""Point-in-time user location centers and candidate distance scores."""

from __future__ import annotations

import math
from datetime import datetime
from pydantic import Field

from yelp_agent.data.temporal_view import TemporalDataError, TemporalDataView
from yelp_agent.models import (
    LocationCenter,
    RecommendationTask,
    StrictModel,
    UnitScore,
)


EARTH_RADIUS_KM = 6371.0088


class LocationFeatureError(RuntimeError):
    """Raised when a location feature cannot be built without leakage."""


class LocationBusinessScore(StrictModel):
    business_id: str = Field(min_length=1)
    distance_km: float | None = Field(default=None, ge=0)
    location_score: UnitScore


class LocationTaskFeatures(StrictModel):
    location_center: LocationCenter | None
    history_coordinate_count: int = Field(ge=0)
    business_scores: dict[str, LocationBusinessScore]


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the great-circle distance between two WGS84 coordinates."""

    latitude_a_radians = math.radians(latitude_a)
    latitude_b_radians = math.radians(latitude_b)
    delta_latitude = latitude_b_radians - latitude_a_radians
    delta_longitude = math.radians(longitude_b - longitude_a)
    haversine_value = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(latitude_a_radians)
        * math.cos(latitude_b_radians)
        * math.sin(delta_longitude / 2.0) ** 2
    )
    central_angle = 2.0 * math.asin(
        math.sqrt(min(1.0, max(0.0, haversine_value)))
    )
    return EARTH_RADIUS_KM * central_angle


def _optional_coordinate(
    latitude: object,
    longitude: object,
    *,
    business_id: str,
) -> LocationCenter | None:
    if latitude is None or longitude is None:
        return None
    latitude_value = float(latitude)
    longitude_value = float(longitude)
    if not math.isfinite(latitude_value) or not math.isfinite(longitude_value):
        return None
    if not -90.0 <= latitude_value <= 90.0:
        raise LocationFeatureError(
            f"Business {business_id!r} has latitude outside -90 to 90"
        )
    if not -180.0 <= longitude_value <= 180.0:
        raise LocationFeatureError(
            f"Business {business_id!r} has longitude outside -180 to 180"
        )
    return LocationCenter(
        latitude=latitude_value,
        longitude=longitude_value,
    )


class TemporalLocationStore:
    """Calculate candidate proximity solely from each frozen task history."""

    def __init__(
        self,
        data_view: TemporalDataView,
        *,
        scale_km: float = 10.0,
    ) -> None:
        if not math.isfinite(scale_km) or scale_km <= 0:
            raise ValueError("scale_km must be a positive finite number")
        self._data_view = data_view
        self._scale_km = scale_km
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            LocationTaskFeatures,
        ] = {}

    def features_for(self, task: RecommendationTask) -> LocationTaskFeatures:
        """Return a history center and distance scores for one frozen task."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        history = self._data_view.user_history(
            task.user_id,
            task.cutoff_time,
        )
        if not history:
            raise LocationFeatureError(
                f"No frozen location history exists for task {task.task_id!r}"
            )
        history_coordinates: list[LocationCenter] = []
        for interaction in history:
            try:
                business = self._data_view.business(interaction.business_id)
            except TemporalDataError as exc:
                raise LocationFeatureError(str(exc)) from exc
            coordinate = _optional_coordinate(
                business.latitude,
                business.longitude,
                business_id=business.business_id,
            )
            if coordinate is not None:
                history_coordinates.append(coordinate)

        location_center = (
            LocationCenter(
                latitude=sum(
                    coordinate.latitude for coordinate in history_coordinates
                )
                / len(history_coordinates),
                longitude=sum(
                    coordinate.longitude for coordinate in history_coordinates
                )
                / len(history_coordinates),
            )
            if history_coordinates
            else None
        )

        business_scores: dict[str, LocationBusinessScore] = {}
        for business_id in task.candidate_business_ids:
            try:
                business = self._data_view.business(business_id)
            except TemporalDataError as exc:
                raise LocationFeatureError(str(exc)) from exc
            coordinate = _optional_coordinate(
                business.latitude,
                business.longitude,
                business_id=business.business_id,
            )
            if location_center is None or coordinate is None:
                distance_km = None
                location_score = 0.5
            else:
                distance_km = haversine_km(
                    location_center.latitude,
                    location_center.longitude,
                    coordinate.latitude,
                    coordinate.longitude,
                )
                location_score = math.exp(-distance_km / self._scale_km)
            business_scores[business_id] = LocationBusinessScore(
                business_id=business_id,
                distance_km=distance_km,
                location_score=location_score,
            )

        features = LocationTaskFeatures(
            location_center=location_center,
            history_coordinate_count=len(history_coordinates),
            business_scores=business_scores,
        )
        self._cache[cache_key] = features
        return features
