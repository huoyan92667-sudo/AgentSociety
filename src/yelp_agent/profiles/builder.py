"""Deterministic aggregation behind the small user-profile builder interface."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.config import UserProfileConfig
from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.data.temporal_view import BusinessRecord, InteractionRecord
from yelp_agent.models import LocationCenter
from yelp_agent.profiles.schema import (
    PreferenceSignal,
    ProfileEvidenceSummary,
    UserProfileV1,
)
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA, ReviewAspectRecord


class UserProfileBuildError(RuntimeError):
    """Raised when source data cannot support a safe profile snapshot."""


class UserProfileBuilder:
    """Build one immutable profile through a cutoff-required interface."""

    def __init__(
        self,
        *,
        businesses: dict[str, BusinessRecord],
        interactions_by_user: dict[str, tuple[InteractionRecord, ...]],
        aspect_records_by_user: dict[str, tuple[ReviewAspectRecord, ...]],
        config: UserProfileConfig,
        broad_categories: frozenset[str],
    ) -> None:
        self._businesses = businesses
        self._interactions_by_user = interactions_by_user
        self._interaction_dates = {
            user_id: tuple(item.date for item in rows)
            for user_id, rows in interactions_by_user.items()
        }
        self._aspect_records_by_user = aspect_records_by_user
        self._aspect_dates = {
            user_id: tuple(item.review_time for item in rows)
            for user_id, rows in aspect_records_by_user.items()
        }
        self._config = config
        self._broad_categories = broad_categories

    @classmethod
    def from_records(
        cls,
        *,
        businesses: Sequence[BusinessRecord],
        interactions: Sequence[InteractionRecord],
        aspect_records: Sequence[ReviewAspectRecord],
        config: UserProfileConfig,
        broad_categories: set[str],
    ) -> UserProfileBuilder:
        """Create an indexed builder from validated in-memory records."""

        business_map: dict[str, BusinessRecord] = {}
        for business in businesses:
            if business.business_id in business_map:
                raise UserProfileBuildError("business IDs must be unique")
            business_map[business.business_id] = business
        if not business_map:
            raise UserProfileBuildError("at least one business is required")

        grouped_interactions: defaultdict[str, list[InteractionRecord]] = defaultdict(
            list
        )
        seen_reviews: set[str] = set()
        for interaction in interactions:
            if interaction.review_id in seen_reviews:
                raise UserProfileBuildError("interaction review IDs must be unique")
            if interaction.business_id not in business_map:
                raise UserProfileBuildError(
                    "interaction references an unknown business"
                )
            seen_reviews.add(interaction.review_id)
            grouped_interactions[interaction.user_id].append(interaction)

        grouped_aspects: defaultdict[str, list[ReviewAspectRecord]] = defaultdict(list)
        for record in aspect_records:
            if record.business_id not in business_map:
                raise UserProfileBuildError("aspect references an unknown business")
            grouped_aspects[record.user_id].append(record)

        ordered_interactions = {
            user_id: tuple(sorted(rows, key=lambda row: (row.date, row.review_id)))
            for user_id, rows in grouped_interactions.items()
        }
        ordered_aspects = {
            user_id: tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        row.review_time,
                        row.review_id,
                        row.evidence_start,
                        row.aspect,
                    ),
                )
            )
            for user_id, rows in grouped_aspects.items()
        }
        return cls(
            businesses=business_map,
            interactions_by_user=ordered_interactions,
            aspect_records_by_user=ordered_aspects,
            config=config,
            broad_categories=frozenset(broad_categories),
        )

    @classmethod
    def from_parquet(
        cls,
        *,
        businesses_path: str | Path,
        interactions_path: str | Path,
        aspect_records_path: str | Path,
        config: UserProfileConfig,
        broad_categories: set[str],
    ) -> UserProfileBuilder:
        """Load only selected-user sources; full business reviews are not read."""

        paths_and_schemas = (
            (Path(businesses_path), BUSINESS_SCHEMA),
            (Path(interactions_path), REVIEW_SCHEMA),
            (Path(aspect_records_path), REVIEW_ASPECT_SCHEMA),
        )
        for path, expected_schema in paths_and_schemas:
            if not path.is_file():
                raise FileNotFoundError(f"User-profile source does not exist: {path}")
            try:
                actual_schema = pq.ParquetFile(path).schema_arrow
            except (OSError, pa.ArrowException) as exc:
                raise UserProfileBuildError(
                    f"Could not read user-profile source: {path}"
                ) from exc
            if not actual_schema.equals(expected_schema, check_metadata=False):
                raise UserProfileBuildError(
                    f"User-profile source has an unexpected schema: {path}"
                )

        business_rows = pq.read_table(businesses_path).to_pylist()
        businesses = tuple(
            BusinessRecord(
                business_id=str(row["business_id"]),
                name=str(row["name"]),
                address=str(row["address"] or ""),
                city=str(row["city"] or ""),
                state=str(row["state"] or ""),
                postal_code=str(row["postal_code"] or ""),
                latitude=(None if row["latitude"] is None else float(row["latitude"])),
                longitude=(
                    None if row["longitude"] is None else float(row["longitude"])
                ),
                categories=tuple(str(value) for value in row["categories"]),
                attributes_json=str(row["attributes_json"]),
            )
            for row in business_rows
        )
        interaction_rows = pq.read_table(
            interactions_path,
            columns=["review_id", "user_id", "business_id", "stars", "text", "date"],
        ).to_pylist()
        interactions = tuple(
            InteractionRecord(
                review_id=str(row["review_id"]),
                user_id=str(row["user_id"]),
                business_id=str(row["business_id"]),
                stars=float(row["stars"]),
                text=str(row["text"]),
                date=row["date"],
            )
            for row in interaction_rows
        )
        aspect_records = tuple(
            ReviewAspectRecord.model_validate(row)
            for row in pq.read_table(aspect_records_path).to_pylist()
        )
        return cls.from_records(
            businesses=businesses,
            interactions=interactions,
            aspect_records=aspect_records,
            config=config,
            broad_categories=broad_categories,
        )

    def _time_weight(self, event_time: datetime, cutoff_time: datetime) -> float:
        age_days = (cutoff_time - event_time).total_seconds() / 86400.0
        if age_days <= 0:
            raise UserProfileBuildError("profile evidence must be before cutoff")
        return 0.5 ** (age_days / self._config.half_life_days)

    def _signal(
        self,
        *,
        kind: str,
        value: str,
        observations: Iterable[tuple[float, datetime, float]],
        source: str,
        cutoff_time: datetime,
    ) -> PreferenceSignal:
        rows = tuple(observations)
        weighted_sum = 0.0
        effective_evidence = 0.0
        for direction, event_time, source_confidence in rows:
            weight = self._time_weight(event_time, cutoff_time) * source_confidence
            weighted_sum += direction * weight
            effective_evidence += weight
        score = weighted_sum / effective_evidence
        saturation = 1.0 - math.exp(
            -effective_evidence / self._config.confidence_saturation
        )
        consistency = 0.5 + 0.5 * abs(score)
        return PreferenceSignal.model_validate(
            {
                "kind": kind,
                "value": value,
                "score": max(-1.0, min(1.0, score)),
                "confidence": saturation * consistency,
                "evidence_count": len(rows),
                "effective_evidence": effective_evidence,
                "first_seen": min(row[1] for row in rows),
                "last_confirmed": max(row[1] for row in rows),
                "source": source,
            }
        )

    @staticmethod
    def _price_level(business: BusinessRecord) -> int | None:
        raw = business.attributes_dict().get("RestaurantsPriceRange2")
        if raw is None:
            return None
        normalized = str(raw).strip().strip("'\"")
        if normalized not in {"1", "2", "3", "4"}:
            return None
        return int(normalized)

    def build(self, user_id: str, cutoff_time: datetime) -> UserProfileV1:
        """Build one profile from records strictly before ``cutoff_time``."""

        if not user_id or user_id != user_id.strip():
            raise ValueError("user_id must be nonempty without surrounding whitespace")
        if not isinstance(cutoff_time, datetime):
            raise TypeError("cutoff_time must be a datetime")
        rows = self._interactions_by_user.get(user_id, ())
        dates = self._interaction_dates.get(user_id, ())
        history = rows[: bisect_left(dates, cutoff_time)]
        if not history:
            raise UserProfileBuildError("no interaction history exists before cutoff")

        categories: defaultdict[str, list[tuple[float, datetime, float]]] = defaultdict(
            list
        )
        prices: defaultdict[str, list[tuple[float, datetime, float]]] = defaultdict(
            list
        )
        areas: defaultdict[str, list[tuple[float, datetime, float]]] = defaultdict(list)
        coordinates: list[tuple[float, float, float]] = []
        rating_distribution = {str(stars): 0 for stars in range(1, 6)}
        rating_sum = 0.0
        for interaction in history:
            rating_sum += interaction.stars
            rating_distribution[str(int(interaction.stars))] += 1
            business = self._businesses[interaction.business_id]
            direction = (interaction.stars - 3.0) / 2.0
            for category in sorted(
                set(business.categories).difference(self._broad_categories)
            ):
                categories[category].append((direction, interaction.date, 1.0))
            price_level = self._price_level(business)
            positive_strength = max(0.0, direction)
            if price_level is not None and positive_strength > 0:
                prices[str(price_level)].append(
                    (
                        (price_level - 1.0) / 3.0,
                        interaction.date,
                        positive_strength,
                    )
                )
            if business.postal_code:
                areas[business.postal_code].append((1.0, interaction.date, 1.0))
            if business.latitude is not None and business.longitude is not None:
                coordinate_weight = self._time_weight(interaction.date, cutoff_time)
                coordinates.append(
                    (business.latitude, business.longitude, coordinate_weight)
                )

        category_signals = [
            self._signal(
                kind="category",
                value=category,
                observations=observations,
                source="rating_category",
                cutoff_time=cutoff_time,
            )
            for category, observations in categories.items()
        ]
        category_signals.sort(
            key=lambda signal: (
                -(abs(signal.score) * signal.confidence),
                signal.value,
            )
        )
        positive_categories = [
            signal for signal in category_signals if signal.score > 0
        ][: self._config.max_category_preferences]
        negative_categories = [
            signal for signal in category_signals if signal.score < 0
        ][: self._config.max_category_preferences]

        aspect_rows = self._aspect_records_by_user.get(user_id, ())
        aspect_dates = self._aspect_dates.get(user_id, ())
        aspect_history = aspect_rows[: bisect_left(aspect_dates, cutoff_time)]
        aspect_observations: defaultdict[str, list[tuple[float, datetime, float]]] = (
            defaultdict(list)
        )
        sentiment_directions = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0,
            "mixed": 0.0,
        }
        for record in aspect_history:
            aspect_observations[record.aspect].append(
                (
                    sentiment_directions[record.sentiment],
                    record.review_time,
                    record.confidence,
                )
            )
        aspect_signals = [
            self._signal(
                kind="aspect",
                value=aspect,
                observations=observations,
                source="review_aspect",
                cutoff_time=cutoff_time,
            )
            for aspect, observations in aspect_observations.items()
        ]
        aspect_signals.sort(
            key=lambda signal: (
                -(abs(signal.score) * signal.confidence),
                signal.value,
            )
        )
        positive_aspects = [signal for signal in aspect_signals if signal.score > 0]
        negative_aspects = [signal for signal in aspect_signals if signal.score < 0]

        price_signals = [
            self._signal(
                kind="price",
                value=price_level,
                observations=observations,
                source="business_price",
                cutoff_time=cutoff_time,
            )
            for price_level, observations in prices.items()
        ]
        price_preference = (
            max(
                price_signals,
                key=lambda signal: (
                    signal.effective_evidence,
                    signal.evidence_count,
                    -int(signal.value),
                ),
            )
            if price_signals
            else None
        )

        area_signals = [
            self._signal(
                kind="area",
                value=postal_code,
                observations=observations,
                source="business_area",
                cutoff_time=cutoff_time,
            )
            for postal_code, observations in areas.items()
        ]
        total_area_evidence = sum(signal.effective_evidence for signal in area_signals)
        area_signals = [
            signal.model_copy(
                update={"score": signal.effective_evidence / total_area_evidence}
            )
            for signal in area_signals
        ]
        area_signals.sort(key=lambda signal: (-signal.score, signal.value))
        frequent_areas = area_signals[: self._config.max_area_preferences]

        coordinate_total = sum(row[2] for row in coordinates)
        location_center = (
            LocationCenter(
                latitude=sum(row[0] * row[2] for row in coordinates) / coordinate_total,
                longitude=sum(row[1] * row[2] for row in coordinates)
                / coordinate_total,
            )
            if coordinate_total > 0
            else None
        )

        history_factor = 1.0 - math.exp(
            -len(history) / self._config.reliability_history_saturation
        )
        aspect_factor = 1.0 - math.exp(
            -len(aspect_history) / self._config.reliability_aspect_saturation
        )
        profile_id = hashlib.sha256(
            f"{user_id}\0{cutoff_time.isoformat()}\0{self._config.profile_version}".encode()
        ).hexdigest()
        return UserProfileV1(
            profile_id=profile_id,
            user_id=user_id,
            cutoff_time=cutoff_time,
            history_length=len(history),
            average_rating=rating_sum / len(history),
            rating_distribution=rating_distribution,
            category_preferences=positive_categories,
            category_dislikes=negative_categories,
            aspect_preferences=positive_aspects,
            aspect_dislikes=negative_aspects,
            price_preference=price_preference,
            frequent_areas=frequent_areas,
            location_center=location_center,
            reliability=0.7 * history_factor + 0.3 * aspect_factor,
            evidence_summary=ProfileEvidenceSummary(
                category_evidence_count=sum(len(rows) for rows in categories.values()),
                aspect_evidence_count=len(aspect_history),
                price_evidence_count=sum(len(rows) for rows in prices.values()),
                area_evidence_count=sum(len(rows) for rows in areas.values()),
                first_interaction=history[0].date,
                last_interaction=history[-1].date,
            ),
            profile_version=self._config.profile_version,
        )
