"""Point-in-time user location centers and candidate distance scores."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

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


@dataclass(frozen=True)
class _LocationHistoryInteraction:
    position: int
    review_id: str
    user_id: str
    business_id: str
    date: datetime


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
        businesses_path: str | Path,
        interactions_path: str | Path,
        histories_path: str | Path,
        *,
        scale_km: float = 10.0,
    ) -> None:
        if not math.isfinite(scale_km) or scale_km <= 0:
            raise ValueError("scale_km must be a positive finite number")
        self._scale_km = scale_km
        self._business_coordinates = self._load_business_coordinates(
            Path(businesses_path)
        )
        self._histories = self._load_histories(
            Path(interactions_path),
            Path(histories_path),
        )
        self._cache: dict[
            tuple[str, str, datetime, tuple[str, ...]],
            LocationTaskFeatures,
        ] = {}

    @staticmethod
    def _load_business_coordinates(
        path: Path,
    ) -> dict[str, LocationCenter | None]:
        if not path.is_file():
            raise FileNotFoundError(f"Business Parquet does not exist: {path}")
        try:
            rows = pq.read_table(
                path,
                columns=["business_id", "latitude", "longitude"],
            ).to_pylist()
        except (OSError, pa.ArrowException) as exc:
            raise LocationFeatureError(
                f"Could not read location businesses from {path}: {exc}"
            ) from exc

        coordinates: dict[str, LocationCenter | None] = {}
        for row in rows:
            business_id = row.get("business_id")
            if (
                not isinstance(business_id, str)
                or not business_id
                or business_id in coordinates
            ):
                raise LocationFeatureError(
                    "Location business data contains a missing or duplicate ID"
                )
            coordinates[business_id] = _optional_coordinate(
                row.get("latitude"),
                row.get("longitude"),
                business_id=business_id,
            )
        if not coordinates:
            raise LocationFeatureError("Location business data is empty")
        return coordinates

    def _load_histories(
        self,
        interactions_path: Path,
        histories_path: Path,
    ) -> dict[str, tuple[_LocationHistoryInteraction, ...]]:
        for label, path in (
            ("Interaction", interactions_path),
            ("Temporal history", histories_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} Parquet does not exist: {path}")
        try:
            with duckdb.connect() as connection:
                interaction_counts = connection.execute(
                    """
                    SELECT count(*), count(DISTINCT review_id)
                    FROM read_parquet(?)
                    """,
                    [str(interactions_path)],
                ).fetchone()
                history_count = connection.execute(
                    "SELECT count(*) FROM read_parquet(?)",
                    [str(histories_path)],
                ).fetchone()[0]
                rows = connection.execute(
                    """
                    SELECT
                        history.task_id,
                        history.position,
                        interaction.review_id,
                        interaction.user_id,
                        interaction.business_id,
                        interaction.date
                    FROM read_parquet(?) AS history
                    JOIN read_parquet(?) AS interaction USING (review_id)
                    ORDER BY history.task_id, history.position
                    """,
                    [str(histories_path), str(interactions_path)],
                ).fetchall()
        except duckdb.Error as exc:
            raise LocationFeatureError(
                f"Could not join frozen location histories: {exc}"
            ) from exc

        if interaction_counts[0] != interaction_counts[1]:
            raise LocationFeatureError("Interaction review_id values must be unique")
        if len(rows) != history_count:
            raise LocationFeatureError(
                "A frozen location history review is missing from interactions"
            )

        grouped: defaultdict[
            str, list[_LocationHistoryInteraction]
        ] = defaultdict(list)
        for task_id, position, review_id, user_id, business_id, date in rows:
            if (
                not task_id
                or not review_id
                or not user_id
                or not business_id
                or not isinstance(date, datetime)
            ):
                raise LocationFeatureError(
                    "Frozen location history contains an invalid row"
                )
            business_key = str(business_id)
            if business_key not in self._business_coordinates:
                raise LocationFeatureError(
                    f"History references unknown business_id {business_key!r}"
                )
            grouped[str(task_id)].append(
                _LocationHistoryInteraction(
                    position=int(position),
                    review_id=str(review_id),
                    user_id=str(user_id),
                    business_id=business_key,
                    date=date,
                )
            )

        result: dict[str, tuple[_LocationHistoryInteraction, ...]] = {}
        for task_id, interactions in grouped.items():
            positions = [interaction.position for interaction in interactions]
            review_ids = [interaction.review_id for interaction in interactions]
            if (
                positions != list(range(1, len(interactions) + 1))
                or len(set(review_ids)) != len(review_ids)
            ):
                raise LocationFeatureError(
                    f"Frozen location positions are invalid for task {task_id!r}"
                )
            result[task_id] = tuple(interactions)
        return result

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

        history = self._histories.get(task.task_id)
        if not history:
            raise LocationFeatureError(
                f"No frozen location history exists for task {task.task_id!r}"
            )
        history_coordinates: list[LocationCenter] = []
        for interaction in history:
            if interaction.user_id != task.user_id:
                raise LocationFeatureError(
                    f"Task user does not match location history for {task.task_id!r}"
                )
            if interaction.date >= task.cutoff_time:
                raise LocationFeatureError(
                    f"Location history is not before cutoff for {task.task_id!r}"
                )
            coordinate = self._business_coordinates[interaction.business_id]
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
            if business_id not in self._business_coordinates:
                raise LocationFeatureError(
                    f"Task references unknown business_id {business_id!r}"
                )
            coordinate = self._business_coordinates[business_id]
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
