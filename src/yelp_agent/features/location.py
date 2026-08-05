"""Point-in-time user location centers and candidate distance scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import Field

from yelp_agent.data.temporal_view import TemporalDataError, TemporalDataView
from yelp_agent.models import (
    LocationCenter,
    StrictModel,
    UnitScore,
)
from yelp_agent.protocols import CandidateScoringRequest


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


@dataclass(frozen=True, slots=True)
class CandidateLocationScores:
    """Lightweight aligned proximity values for high-volume retrieval."""

    business_ids: tuple[str, ...]
    location_scores: np.ndarray
    distances_km: np.ndarray


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
        self._coordinates = {
            business.business_id: (business.latitude, business.longitude)
            for business in data_view.businesses()
        }
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            LocationTaskFeatures,
        ] = {}

    def score_candidates(
        self,
        task: CandidateScoringRequest,
    ) -> CandidateLocationScores:
        """Return the same location formula without per-business models."""

        history = self._data_view.user_history(
            task.user_id,
            task.cutoff_time,
        )
        if not history:
            raise LocationFeatureError(
                f"No frozen location history exists for task {task.task_id!r}"
            )
        coordinates: list[tuple[float, float]] = []
        for interaction in history:
            latitude, longitude = self._coordinates[interaction.business_id]
            if latitude is not None and longitude is not None:
                coordinates.append((latitude, longitude))
        candidate_count = len(task.candidate_business_ids)
        scores = np.full(candidate_count, 0.5, dtype=np.float64)
        distances = np.full(candidate_count, np.nan, dtype=np.float64)
        if coordinates:
            center_latitude = sum(item[0] for item in coordinates) / len(
                coordinates
            )
            center_longitude = sum(item[1] for item in coordinates) / len(
                coordinates
            )
            latitudes = np.asarray(
                [
                    (
                        np.nan
                        if self._coordinates[item][0] is None
                        else self._coordinates[item][0]
                    )
                    for item in task.candidate_business_ids
                ],
                dtype=np.float64,
            )
            longitudes = np.asarray(
                [
                    (
                        np.nan
                        if self._coordinates[item][1] is None
                        else self._coordinates[item][1]
                    )
                    for item in task.candidate_business_ids
                ],
                dtype=np.float64,
            )
            valid = np.isfinite(latitudes) & np.isfinite(longitudes)
            center_latitude_radians = math.radians(center_latitude)
            latitudes_radians = np.radians(latitudes[valid])
            delta_latitude = latitudes_radians - center_latitude_radians
            delta_longitude = np.radians(
                longitudes[valid] - center_longitude
            )
            haversine_values = (
                np.sin(delta_latitude / 2.0) ** 2
                + math.cos(center_latitude_radians)
                * np.cos(latitudes_radians)
                * np.sin(delta_longitude / 2.0) ** 2
            )
            valid_distances = EARTH_RADIUS_KM * 2.0 * np.arcsin(
                np.sqrt(np.clip(haversine_values, 0.0, 1.0))
            )
            distances[valid] = valid_distances
            scores[valid] = np.exp(-valid_distances / self._scale_km)
        return CandidateLocationScores(
            business_ids=tuple(task.candidate_business_ids),
            location_scores=scores,
            distances_km=distances,
        )

    def features_for(
        self,
        task: CandidateScoringRequest,
        *,
        cache: bool = True,
    ) -> LocationTaskFeatures:
        """Return a history center and distance scores for one frozen task."""

        cache_key = (
            task.task_id,
            task.user_id,
            task.cutoff_time,
            tuple(task.candidate_business_ids),
        )
        cached = self._cache.get(cache_key) if cache else None
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
        if cache:
            self._cache[cache_key] = features
        return features
