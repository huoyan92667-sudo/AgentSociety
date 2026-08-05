"""Read-only point-in-time access to recommendation data."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


class TemporalDataError(RuntimeError):
    """Raised when source data cannot support safe point-in-time queries."""


@dataclass(frozen=True, slots=True)
class BusinessRecord:
    """Static business fields that are safe at every recommendation cutoff."""

    business_id: str
    name: str
    address: str
    city: str
    state: str
    postal_code: str
    latitude: float | None
    longitude: float | None
    categories: tuple[str, ...]
    attributes_json: str

    def attributes_dict(self) -> dict[str, Any]:
        """Return a fresh mutable copy without exposing shared state."""

        value = json.loads(self.attributes_json)
        if not isinstance(value, dict):
            raise TemporalDataError(
                f"Business {self.business_id!r} attributes are not an object"
            )
        return value


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """One selected-user interaction, including text needed by profiles."""

    review_id: str
    user_id: str
    business_id: str
    stars: float
    text: str
    date: datetime


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """One lightweight business review used by time-aware aggregates."""

    review_id: str
    business_id: str
    stars: float
    date: datetime


@dataclass(frozen=True, slots=True)
class ReviewAggregate:
    """Count and rating sum before one cutoff."""

    count: int
    star_sum: float

    @property
    def mean_rating(self) -> float | None:
        return None if self.count == 0 else self.star_sum / self.count


@dataclass(frozen=True, slots=True)
class ReviewStatistics:
    """Global calibration inputs and requested business aggregates."""

    global_count: int
    global_star_sum: float
    positive_business_counts: tuple[int, ...]
    businesses: Mapping[str, ReviewAggregate]

    @property
    def global_mean_rating(self) -> float | None:
        return (
            None
            if self.global_count == 0
            else self.global_star_sum / self.global_count
        )


@dataclass(frozen=True, slots=True)
class _ReviewHistory:
    dates: tuple[datetime, ...]
    review_ids: tuple[str, ...]
    stars: tuple[float, ...]
    prefix_stars: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _InteractionHistory:
    dates: tuple[datetime, ...]
    records: tuple[InteractionRecord, ...]


def _prefix_sums(values: Sequence[float]) -> tuple[float, ...]:
    prefix = [0.0]
    running = 0.0
    for value in values:
        running += value
        prefix.append(running)
    return tuple(prefix)


def _optional_coordinate(
    value: object,
    *,
    minimum: float,
    maximum: float,
    label: str,
    business_id: str,
) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if not minimum <= number <= maximum:
        raise TemporalDataError(
            f"Business {business_id!r} has {label} outside "
            f"{minimum:g} to {maximum:g}"
        )
    return number


class TemporalDataView:
    """Load Parquet once and answer only static or cutoff-bounded queries."""

    def __init__(
        self,
        businesses_path: str | Path,
        reviews_path: str | Path,
        interactions_path: str | Path,
    ) -> None:
        self._businesses = self._load_businesses(Path(businesses_path))
        (
            self._review_histories,
            self._global_review_dates,
            self._global_review_prefix_stars,
        ) = self._load_reviews(Path(reviews_path))
        (
            self._user_interactions,
            self._global_interaction_dates,
            self._global_interactions,
        ) = self._load_interactions(Path(interactions_path))

    @staticmethod
    def _load_businesses(path: Path) -> dict[str, BusinessRecord]:
        if not path.is_file():
            raise FileNotFoundError(f"Business Parquet does not exist: {path}")
        try:
            rows = pq.read_table(
                path,
                columns=[
                    "business_id",
                    "name",
                    "address",
                    "city",
                    "state",
                    "postal_code",
                    "latitude",
                    "longitude",
                    "categories",
                    "attributes_json",
                ],
            ).to_pylist()
        except (OSError, pa.ArrowException) as exc:
            raise TemporalDataError(
                f"Could not read temporal business data from {path}: {exc}"
            ) from exc

        businesses: dict[str, BusinessRecord] = {}
        for row in rows:
            business_id = row.get("business_id")
            name = row.get("name")
            categories = row.get("categories")
            attributes_json = row.get("attributes_json")
            if (
                not isinstance(business_id, str)
                or not business_id
                or business_id in businesses
                or not isinstance(name, str)
                or not name
                or not isinstance(categories, list)
                or not isinstance(attributes_json, str)
            ):
                raise TemporalDataError(
                    "Temporal business data contains an invalid row"
                )
            try:
                attributes = json.loads(attributes_json)
            except json.JSONDecodeError as exc:
                raise TemporalDataError(
                    f"Business {business_id!r} has invalid attributes JSON"
                ) from exc
            if not isinstance(attributes, dict):
                raise TemporalDataError(
                    f"Business {business_id!r} attributes must be an object"
                )
            canonical_attributes = json.dumps(
                attributes,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            businesses[business_id] = BusinessRecord(
                business_id=business_id,
                name=name,
                address=str(row.get("address") or ""),
                city=str(row.get("city") or ""),
                state=str(row.get("state") or ""),
                postal_code=str(row.get("postal_code") or ""),
                latitude=_optional_coordinate(
                    row.get("latitude"),
                    minimum=-90.0,
                    maximum=90.0,
                    label="latitude",
                    business_id=business_id,
                ),
                longitude=_optional_coordinate(
                    row.get("longitude"),
                    minimum=-180.0,
                    maximum=180.0,
                    label="longitude",
                    business_id=business_id,
                ),
                categories=tuple(
                    str(category).strip()
                    for category in categories
                    if str(category).strip()
                ),
                attributes_json=canonical_attributes,
            )
        if not businesses:
            raise TemporalDataError("Temporal business data is empty")
        return businesses

    def _load_reviews(
        self,
        path: Path,
    ) -> tuple[
        dict[str, _ReviewHistory],
        tuple[datetime, ...],
        tuple[float, ...],
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Review Parquet does not exist: {path}")
        try:
            with duckdb.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT review_id, business_id, date, stars
                    FROM read_parquet(?)
                    ORDER BY date, review_id
                    """,
                    [str(path)],
                ).fetchall()
        except duckdb.Error as exc:
            raise TemporalDataError(
                f"Could not read temporal reviews from {path}: {exc}"
            ) from exc

        grouped: defaultdict[
            str, list[tuple[datetime, str, float]]
        ] = defaultdict(list)
        global_dates: list[datetime] = []
        global_stars: list[float] = []
        seen_review_ids: set[str] = set()
        for review_id, business_id, date, stars in rows:
            review_key = str(review_id or "")
            business_key = str(business_id or "")
            rating = float(stars)
            if (
                not review_key
                or review_key in seen_review_ids
                or not business_key
                or business_key not in self._businesses
                or not isinstance(date, datetime)
                or not 1.0 <= rating <= 5.0
            ):
                raise TemporalDataError(
                    "Temporal review data contains an invalid row"
                )
            seen_review_ids.add(review_key)
            grouped[business_key].append((date, review_key, rating))
            global_dates.append(date)
            global_stars.append(rating)
        if not rows:
            raise TemporalDataError("Temporal review data is empty")

        histories = {
            business_id: _ReviewHistory(
                dates=tuple(date for date, _, _ in values),
                review_ids=tuple(review_id for _, review_id, _ in values),
                stars=tuple(rating for _, _, rating in values),
                prefix_stars=_prefix_sums(
                    [rating for _, _, rating in values]
                ),
            )
            for business_id, values in grouped.items()
        }
        return (
            histories,
            tuple(global_dates),
            _prefix_sums(global_stars),
        )

    def _load_interactions(
        self,
        path: Path,
    ) -> tuple[
        dict[str, _InteractionHistory],
        tuple[datetime, ...],
        tuple[InteractionRecord, ...],
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Interaction Parquet does not exist: {path}"
            )
        try:
            with duckdb.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT review_id, user_id, business_id, stars, text, date
                    FROM read_parquet(?)
                    ORDER BY date, review_id
                    """,
                    [str(path)],
                ).fetchall()
        except duckdb.Error as exc:
            raise TemporalDataError(
                f"Could not read temporal interactions from {path}: {exc}"
            ) from exc

        by_user: defaultdict[str, list[InteractionRecord]] = defaultdict(list)
        global_records: list[InteractionRecord] = []
        seen_review_ids: set[str] = set()
        for review_id, user_id, business_id, stars, text, date in rows:
            review_key = str(review_id or "")
            user_key = str(user_id or "")
            business_key = str(business_id or "")
            rating = float(stars)
            if (
                not review_key
                or review_key in seen_review_ids
                or not user_key
                or not business_key
                or business_key not in self._businesses
                or not isinstance(text, str)
                or not isinstance(date, datetime)
                or not 1.0 <= rating <= 5.0
                or not rating.is_integer()
            ):
                raise TemporalDataError(
                    "Temporal interaction data contains an invalid row"
                )
            seen_review_ids.add(review_key)
            record = InteractionRecord(
                review_id=review_key,
                user_id=user_key,
                business_id=business_key,
                stars=rating,
                text=text,
                date=date,
            )
            by_user[user_key].append(record)
            global_records.append(record)
        if not global_records:
            raise TemporalDataError("Temporal interaction data is empty")

        user_histories: dict[str, _InteractionHistory] = {}
        for user_id, records in by_user.items():
            ordered = tuple(
                sorted(records, key=lambda item: (item.date, item.review_id))
            )
            user_histories[user_id] = _InteractionHistory(
                dates=tuple(record.date for record in ordered),
                records=ordered,
            )
        ordered_global = tuple(global_records)
        return (
            user_histories,
            tuple(record.date for record in ordered_global),
            ordered_global,
        )

    def business(self, business_id: str) -> BusinessRecord:
        """Return one static business record without snapshot rating fields."""

        try:
            return self._businesses[business_id]
        except KeyError as exc:
            raise TemporalDataError(
                f"Unknown business_id: {business_id!r}"
            ) from exc

    def businesses(self) -> tuple[BusinessRecord, ...]:
        """Return all static businesses in source order."""

        return tuple(self._businesses.values())

    def user_history(
        self,
        user_id: str,
        cutoff_time: datetime,
    ) -> tuple[InteractionRecord, ...]:
        """Return this user's interactions strictly before the cutoff."""

        history = self._user_interactions.get(user_id)
        if history is None:
            return ()
        return history.records[:bisect_left(history.dates, cutoff_time)]

    def reviews_before(
        self,
        business_id: str,
        cutoff_time: datetime,
        limit: int | None = None,
    ) -> tuple[ReviewRecord, ...]:
        """Return chronological lightweight reviews strictly before cutoff."""

        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        self.business(business_id)
        history = self._review_histories.get(business_id)
        if history is None:
            return ()
        stop = bisect_left(history.dates, cutoff_time)
        start = 0 if limit is None else max(0, stop - limit)
        return tuple(
            ReviewRecord(
                review_id=history.review_ids[index],
                business_id=business_id,
                stars=history.stars[index],
                date=history.dates[index],
            )
            for index in range(start, stop)
        )

    def interactions_before(
        self,
        cutoff_time: datetime,
    ) -> tuple[InteractionRecord, ...]:
        """Return all selected-user interactions strictly before cutoff."""

        stop = bisect_left(self._global_interaction_dates, cutoff_time)
        return self._global_interactions[:stop]

    def review_statistics_before(
        self,
        business_ids: Sequence[str],
        cutoff_time: datetime,
    ) -> ReviewStatistics:
        """Return time-safe global and requested per-business quality inputs."""

        if len(set(business_ids)) != len(business_ids):
            raise ValueError("business_ids must be unique")
        for business_id in business_ids:
            self.business(business_id)

        global_count = bisect_left(self._global_review_dates, cutoff_time)
        positive_counts: list[int] = []
        requested: dict[str, ReviewAggregate] = {}
        requested_ids = set(business_ids)
        for business_id, history in self._review_histories.items():
            count = bisect_left(history.dates, cutoff_time)
            if count > 0:
                positive_counts.append(count)
            if business_id in requested_ids:
                requested[business_id] = ReviewAggregate(
                    count=count,
                    star_sum=history.prefix_stars[count],
                )
        for business_id in business_ids:
            requested.setdefault(
                business_id,
                ReviewAggregate(count=0, star_sum=0.0),
            )
        return ReviewStatistics(
            global_count=global_count,
            global_star_sum=self._global_review_prefix_stars[global_count],
            positive_business_counts=tuple(positive_counts),
            businesses=MappingProxyType(requested),
        )
