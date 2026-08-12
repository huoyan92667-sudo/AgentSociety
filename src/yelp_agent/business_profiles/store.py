"""Deep module for exact-cutoff business profiles and cached shared knowledge."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.business_profiles.artifacts import BusinessKnowledgeManifest
from yelp_agent.business_profiles.schema import (
    BUSINESS_ASPECT_EVENT_SCHEMA,
    BUSINESS_COVERAGE_SCHEMA,
    BUSINESS_RATING_EVENT_SCHEMA,
    BusinessAspectEvent,
    BusinessAspectSummary,
    BusinessProfileEvidenceSummary,
    BusinessProfileV1,
    BusinessRatingEvent,
)
from yelp_agent.config import BusinessProfileConfig
from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.features.quality import BusinessQuality
from yelp_agent.reviews.schema import ASPECT_NAMES, AspectName


class BusinessKnowledgeError(RuntimeError):
    """Raised when compact business knowledge is invalid or unavailable."""


@dataclass(frozen=True, slots=True)
class _RatingHistory:
    dates: tuple[datetime, ...]
    prefix_stars: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _AspectHistory:
    dates: tuple[datetime, ...]
    events: tuple[BusinessAspectEvent, ...]


def _prefix_sums(values: Sequence[float]) -> tuple[float, ...]:
    result = [0.0]
    running = 0.0
    for value in values:
        running += value
        result.append(running)
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(config: BusinessProfileConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BusinessKnowledgeStore:
    """Return rich profiles through one cutoff-required, batch interface."""

    def __init__(
        self,
        *,
        businesses: dict[str, BusinessRecord],
        rating_histories: dict[str, _RatingHistory],
        global_rating_dates: tuple[datetime, ...],
        global_rating_prefix_stars: tuple[float, ...],
        aspect_histories: dict[tuple[str, AspectName], _AspectHistory],
        config: BusinessProfileConfig,
    ) -> None:
        self._businesses = businesses
        self._business_ids = tuple(sorted(businesses))
        self._rating_histories = rating_histories
        self._global_rating_dates = global_rating_dates
        self._global_rating_prefix_stars = global_rating_prefix_stars
        self._aspect_histories = aspect_histories
        self._config = config
        self._calibration_cache: dict[datetime, tuple[float, float]] = {}
        self._profile_cache: OrderedDict[tuple[str, datetime], BusinessProfileV1] = (
            OrderedDict()
        )

    @property
    def source_scope(self) -> str:
        """Declare the frozen evidence population backing every Aspect summary."""

        return self._config.source_scope

    @classmethod
    def from_records(
        cls,
        *,
        businesses: Sequence[BusinessRecord],
        rating_events: Sequence[BusinessRatingEvent],
        aspect_events: Sequence[BusinessAspectEvent],
        config: BusinessProfileConfig,
    ) -> BusinessKnowledgeStore:
        """Build the same index used by production from validated records."""

        business_map: dict[str, BusinessRecord] = {}
        for business in businesses:
            if business.business_id in business_map:
                raise BusinessKnowledgeError("business IDs must be unique")
            business_map[business.business_id] = business
        if not business_map:
            raise BusinessKnowledgeError("at least one business is required")

        rating_groups: defaultdict[str, list[BusinessRatingEvent]] = defaultdict(list)
        seen_rating_reviews: set[str] = set()
        for event in rating_events:
            if event.review_id in seen_rating_reviews:
                raise BusinessKnowledgeError("rating review IDs must be unique")
            if event.business_id not in business_map:
                raise BusinessKnowledgeError("rating references an unknown business")
            seen_rating_reviews.add(event.review_id)
            rating_groups[event.business_id].append(event)
        rating_histories: dict[str, _RatingHistory] = {}
        for business_id, events in rating_groups.items():
            ordered = sorted(events, key=lambda row: (row.review_time, row.review_id))
            rating_histories[business_id] = _RatingHistory(
                dates=tuple(row.review_time for row in ordered),
                prefix_stars=_prefix_sums(tuple(row.stars for row in ordered)),
            )
        global_ratings = sorted(
            rating_events,
            key=lambda row: (row.review_time, row.review_id),
        )

        aspect_groups: defaultdict[
            tuple[str, AspectName], list[BusinessAspectEvent]
        ] = defaultdict(list)
        for event in aspect_events:
            if event.business_id not in business_map:
                raise BusinessKnowledgeError("aspect references an unknown business")
            aspect_groups[(event.business_id, event.aspect)].append(event)
        aspect_histories: dict[tuple[str, AspectName], _AspectHistory] = {}
        for key, events in aspect_groups.items():
            ordered = sorted(
                events,
                key=lambda row: (row.review_time, row.review_id, row.user_id),
            )
            aspect_histories[key] = _AspectHistory(
                dates=tuple(row.review_time for row in ordered),
                events=tuple(ordered),
            )
        return cls(
            businesses=business_map,
            rating_histories=rating_histories,
            global_rating_dates=tuple(row.review_time for row in global_ratings),
            global_rating_prefix_stars=_prefix_sums(
                tuple(row.stars for row in global_ratings)
            ),
            aspect_histories=aspect_histories,
            config=config,
        )

    @classmethod
    def from_artifacts(
        cls,
        artifact_root: str | Path,
        *,
        config: BusinessProfileConfig,
    ) -> BusinessKnowledgeStore:
        """Load the compact, self-contained production index once."""

        root = Path(artifact_root)
        paths_and_schemas = {
            "businesses": (root / "businesses.parquet", BUSINESS_SCHEMA),
            "ratings": (root / "rating_events.parquet", BUSINESS_RATING_EVENT_SCHEMA),
            "aspects": (root / "aspect_events.parquet", BUSINESS_ASPECT_EVENT_SCHEMA),
            "coverage": (
                root / "business_coverage.parquet",
                BUSINESS_COVERAGE_SCHEMA,
            ),
        }
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Business-knowledge artifact does not exist: {manifest_path}"
            )
        try:
            manifest = BusinessKnowledgeManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise BusinessKnowledgeError(
                f"Business-knowledge manifest is invalid: {manifest_path}"
            ) from exc
        if manifest.configuration_sha256 != _configuration_sha256(config):
            raise BusinessKnowledgeError(
                "Business-knowledge configuration hash does not match"
            )
        for name, (path, schema) in paths_and_schemas.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Business-knowledge artifact does not exist: {path}"
                )
            try:
                actual = pq.ParquetFile(path).schema_arrow
            except (OSError, pa.ArrowException) as exc:
                raise BusinessKnowledgeError(
                    f"Could not read business-knowledge artifact: {path}"
                ) from exc
            if not actual.equals(schema, check_metadata=False):
                raise BusinessKnowledgeError(
                    f"Business-knowledge artifact has an unexpected schema: {path}"
                )
            if manifest.output_sha256.get(name) != _sha256_file(path):
                raise BusinessKnowledgeError(
                    f"Business-knowledge artifact hash does not match: {path}"
                )
        business_rows = pq.read_table(root / "businesses.parquet").to_pylist()
        businesses = tuple(
            BusinessRecord(
                business_id=str(row["business_id"]),
                name=str(row["name"]),
                address=str(row["address"] or ""),
                city=str(row["city"]),
                state=str(row["state"]),
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
        rating_events = tuple(
            BusinessRatingEvent.model_validate(row)
            for row in pq.read_table(root / "rating_events.parquet").to_pylist()
        )
        aspect_events = tuple(
            BusinessAspectEvent.model_validate(row)
            for row in pq.read_table(root / "aspect_events.parquet").to_pylist()
        )
        return cls.from_records(
            businesses=businesses,
            rating_events=rating_events,
            aspect_events=aspect_events,
            config=config,
        )

    def _calibration(self, cutoff_time: datetime) -> tuple[float, float]:
        cached = self._calibration_cache.get(cutoff_time)
        if cached is not None:
            return cached
        global_count = bisect_left(self._global_rating_dates, cutoff_time)
        global_mean = (
            self._global_rating_prefix_stars[global_count] / global_count
            if global_count
            else 3.5
        )
        counts = [
            bisect_left(history.dates, cutoff_time)
            for history in self._rating_histories.values()
        ]
        positive_counts = [count for count in counts if count > 0]
        popularity_p95 = (
            float(np.percentile(np.log1p(positive_counts), 95))
            if positive_counts
            else 0.0
        )
        result = (global_mean, popularity_p95)
        self._calibration_cache[cutoff_time] = result
        return result

    def _quality(self, business_id: str, cutoff_time: datetime) -> BusinessQuality:
        history = self._rating_histories.get(business_id)
        count = 0 if history is None else bisect_left(history.dates, cutoff_time)
        star_sum = 0.0 if history is None else history.prefix_stars[count]
        mean_rating = None if count == 0 else star_sum / count
        global_mean, popularity_p95 = self._calibration(cutoff_time)
        prior = self._config.bayesian_prior_count
        bayesian_rating = (star_sum + prior * global_mean) / (count + prior)
        normalized_rating = min(1.0, max(0.0, (bayesian_rating - 1.0) / 4.0))
        normalized_popularity = (
            min(float(np.log1p(count)), popularity_p95) / popularity_p95
            if popularity_p95 > 0
            else 0.0
        )
        return BusinessQuality(
            business_id=business_id,
            review_count=count,
            mean_rating=mean_rating,
            bayesian_rating=bayesian_rating,
            normalized_bayesian_rating=normalized_rating,
            normalized_popularity=normalized_popularity,
            quality_score=min(
                1.0,
                max(0.0, 0.8 * normalized_rating + 0.2 * normalized_popularity),
            ),
        )

    def _aspect_summary(
        self,
        business_id: str,
        aspect: AspectName,
        cutoff_time: datetime,
    ) -> BusinessAspectSummary:
        history = self._aspect_histories.get((business_id, aspect))
        if history is None:
            events: tuple[BusinessAspectEvent, ...] = ()
        else:
            events = history.events[: bisect_left(history.dates, cutoff_time)]
        counts = {name: 0 for name in ("positive", "negative", "neutral", "mixed")}
        weighted = {"positive": 0.0, "negative": 0.0}
        effective = 0.0
        for event in events:
            counts[event.sentiment] += 1
            age_days = (cutoff_time - event.review_time).total_seconds() / 86400.0
            time_weight = 0.5 ** (age_days / self._config.aspect_half_life_days)
            support = time_weight * event.confidence
            effective += support
            if event.sentiment in weighted:
                weighted[event.sentiment] += support
        unique_users = len({event.user_id for event in events})
        directional_count = counts["positive"] + counts["negative"]
        directional_weight = weighted["positive"] + weighted["negative"]
        directional_users = len(
            {
                event.user_id
                for event in events
                if event.sentiment in {"positive", "negative"}
            }
        )
        known = (
            directional_count >= self._config.minimum_aspect_evidence
            and directional_users >= self._config.minimum_aspect_users
            and directional_weight > 0
        )
        if not known:
            return BusinessAspectSummary(
                aspect=aspect,
                status="unknown",
                positive_count=counts["positive"],
                negative_count=counts["negative"],
                neutral_count=counts["neutral"],
                mixed_count=counts["mixed"],
                evidence_count=len(events),
                unique_users=unique_users,
                effective_evidence=effective,
                latest_evidence_time=(events[-1].review_time if events else None),
                confidence=0.0,
                conflict=False,
            )
        positive_ratio = counts["positive"] / directional_count
        negative_ratio = counts["negative"] / directional_count
        weighted_positive = weighted["positive"] / directional_weight
        weighted_negative = weighted["negative"] / directional_weight
        volume = 1.0 - math.exp(
            -directional_weight / self._config.aspect_confidence_saturation
        )
        diversity = 1.0 - math.exp(
            -directional_users / self._config.aspect_user_saturation
        )
        consistency = max(weighted_positive, weighted_negative)
        confidence = volume * (0.5 + 0.25 * diversity + 0.25 * consistency)
        threshold = self._config.conflict_ratio_threshold
        return BusinessAspectSummary(
            aspect=aspect,
            status="known",
            positive_count=counts["positive"],
            negative_count=counts["negative"],
            neutral_count=counts["neutral"],
            mixed_count=counts["mixed"],
            evidence_count=len(events),
            unique_users=unique_users,
            effective_evidence=effective,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            weighted_positive_ratio=weighted_positive,
            weighted_negative_ratio=weighted_negative,
            latest_evidence_time=events[-1].review_time,
            confidence=min(1.0, max(0.0, confidence)),
            conflict=(
                weighted_positive >= threshold and weighted_negative >= threshold
            ),
        )

    def _build_profile(
        self,
        business_id: str,
        cutoff_time: datetime,
    ) -> BusinessProfileV1:
        business = self._businesses[business_id]
        quality = self._quality(business_id, cutoff_time)
        aspects = {
            aspect: self._aspect_summary(business_id, aspect, cutoff_time)
            for aspect in ASPECT_NAMES
        }
        aspect_count = sum(summary.evidence_count for summary in aspects.values())
        known_count = sum(summary.status == "known" for summary in aspects.values())
        aspect_users: set[str] = set()
        latest_aspect: datetime | None = None
        for aspect in ASPECT_NAMES:
            history = self._aspect_histories.get((business_id, aspect))
            if history is None:
                continue
            events = history.events[: bisect_left(history.dates, cutoff_time)]
            aspect_users.update(event.user_id for event in events)
            if events and (
                latest_aspect is None or events[-1].review_time > latest_aspect
            ):
                latest_aspect = events[-1].review_time
        rating_history = self._rating_histories.get(business_id)
        rating_count = quality.review_count
        latest_rating = (
            None
            if rating_history is None or rating_count == 0
            else rating_history.dates[rating_count - 1]
        )
        rating_factor = 1.0 - math.exp(
            -rating_count / self._config.rating_reliability_saturation
        )
        aspect_factor = 1.0 - math.exp(
            -aspect_count / self._config.aspect_reliability_saturation
        )
        coverage = known_count / len(ASPECT_NAMES)
        profile_id = hashlib.sha256(
            f"{business_id}\0{cutoff_time.isoformat()}\0"
            f"{self._config.profile_version}".encode()
        ).hexdigest()
        return BusinessProfileV1(
            profile_id=profile_id,
            business_id=business_id,
            cutoff_time=cutoff_time,
            name=business.name,
            address=business.address,
            city=business.city,
            state=business.state,
            postal_code=business.postal_code,
            latitude=business.latitude,
            longitude=business.longitude,
            categories=list(business.categories),
            structured_attributes=business.attributes_dict(),
            quality=quality,
            aspect_summaries=aspects,
            profile_reliability=min(
                1.0,
                max(0.0, 0.5 * rating_factor + 0.3 * aspect_factor + 0.2 * coverage),
            ),
            evidence_summary=BusinessProfileEvidenceSummary(
                rating_count=rating_count,
                aspect_evidence_count=aspect_count,
                aspect_unique_users=len(aspect_users),
                known_aspect_count=known_count,
                latest_rating_time=latest_rating,
                latest_aspect_time=latest_aspect,
            ),
            source_scope=self._config.source_scope,
            profile_version=self._config.profile_version,
        )

    def _cached_profile(
        self,
        business_id: str,
        cutoff_time: datetime,
    ) -> BusinessProfileV1:
        key = (business_id, cutoff_time)
        cached = self._profile_cache.pop(key, None)
        if cached is not None:
            self._profile_cache[key] = cached
            return cached
        profile = self._build_profile(business_id, cutoff_time)
        self._profile_cache[key] = profile
        if len(self._profile_cache) > self._config.cache_max_entries:
            self._profile_cache.popitem(last=False)
        return profile

    def get(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessProfileV1]:
        """Return requested profiles only, preserving the caller's ID order."""

        if not business_ids:
            raise ValueError("business_ids cannot be empty")
        if len(set(business_ids)) != len(business_ids):
            raise ValueError("business_ids must be unique")
        if not isinstance(cutoff_time, datetime):
            raise TypeError("cutoff_time must be a datetime")
        unknown = sorted(set(business_ids).difference(self._businesses))
        if unknown:
            raise BusinessKnowledgeError(f"Unknown business IDs: {unknown[:3]}")
        return {
            business_id: self._cached_profile(business_id, cutoff_time)
            for business_id in business_ids
        }

    def get_aspects(
        self,
        business_ids: list[str],
        aspects: list[AspectName],
        cutoff_time: datetime,
    ) -> dict[str, dict[AspectName, BusinessAspectSummary]]:
        """Return only requested cutoff-safe Aspect summaries.

        Query retrieval normally needs one or two aspects across the full business
        catalog.  This narrow batch method avoids constructing quality statistics
        and every frozen aspect for every business while retaining the exact same
        evidence and cutoff semantics as :meth:`get`.
        """

        if not business_ids:
            raise ValueError("business_ids cannot be empty")
        if len(set(business_ids)) != len(business_ids):
            raise ValueError("business_ids must be unique")
        if not aspects:
            raise ValueError("aspects cannot be empty")
        if len(set(aspects)) != len(aspects):
            raise ValueError("aspects must be unique")
        if any(aspect not in ASPECT_NAMES for aspect in aspects):
            raise ValueError("aspects must use the frozen Aspect taxonomy")
        if not isinstance(cutoff_time, datetime):
            raise TypeError("cutoff_time must be a datetime")
        unknown = sorted(set(business_ids).difference(self._businesses))
        if unknown:
            raise BusinessKnowledgeError(f"Unknown business IDs: {unknown[:3]}")
        return {
            business_id: {
                aspect: self._aspect_summary(business_id, aspect, cutoff_time)
                for aspect in aspects
            }
            for business_id in business_ids
        }
