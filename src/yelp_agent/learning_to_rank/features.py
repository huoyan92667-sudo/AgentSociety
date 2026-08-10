"""Stream point-in-time user/business knowledge into Hybrid V2-A features."""

from __future__ import annotations

import hashlib
import math
import os
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from yelp_agent.config import BusinessProfileConfig
from yelp_agent.models import StrictModel, UnitScore
from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.profiles.schema import UserProfileV1

BASE_FEATURES = (
    "retrieval_fusion_score",
    "retrieval_reciprocal_rank",
    "route_coverage",
    "quality_score",
    "category_score",
    "text_score",
    "location_score",
    "distance_km_log",
    "hybrid_v1_score",
    "quality_route_missing",
    "category_route_missing",
    "text_route_missing",
    "location_route_missing",
)
ITEM_KNN_FEATURES = (
    "item_knn_positive_score",
    "item_knn_negative_evidence",
    "item_knn_positive_support_log",
    "item_knn_negative_support_log",
    "item_knn_positive_neighbors_log",
    "item_knn_negative_neighbors_log",
    "item_knn_missing",
)
USER_PROFILE_FEATURES = (
    "user_history_length_log",
    "user_average_rating_normalized",
    "user_profile_reliability",
    "user_category_positive_match",
    "user_category_negative_conflict",
    "user_category_novelty",
    "user_category_profile_missing",
)
BUSINESS_PROFILE_FEATURES = (
    "business_rating_count_log",
    "business_rating_evidence_saturation",
    "business_aspect_evidence_saturation",
    "business_known_aspect_coverage",
    "business_profile_reliability",
)
REVIEW_ASPECT_FEATURES = (
    "aspect_positive_match",
    "aspect_negative_conflict",
    "aspect_match_confidence",
    "aspect_evidence_missing",
)
ALL_FEATURE_NAMES = (
    *BASE_FEATURES,
    *ITEM_KNN_FEATURES,
    *USER_PROFILE_FEATURES,
    *BUSINESS_PROFILE_FEATURES,
    *REVIEW_ASPECT_FEATURES,
)

HYBRID_V2_FEATURE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("cutoff_time", pa.timestamp("us"), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("retrieval_rank", pa.int32(), nullable=False),
        pa.field("label", pa.bool_()),
        pa.field("negative_kind", pa.string()),
        pa.field("sample_weight", pa.float64(), nullable=False),
        pa.field("fold", pa.int32()),
        *(pa.field(name, pa.float64(), nullable=False) for name in ALL_FEATURE_NAMES),
    ]
)


class HybridV2FeatureError(RuntimeError):
    """Raised when profile and retrieval artifacts cannot be joined safely."""


class HybridV1Weights(StrictModel):
    category: UnitScore
    text: UnitScore
    quality: UnitScore
    location: UnitScore

    @model_validator(mode="after")
    def validate_sum(self) -> HybridV1Weights:
        if not math.isclose(
            self.category + self.text + self.quality + self.location,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Hybrid V1 weights must sum to one")
        return self


@dataclass(frozen=True, slots=True)
class HybridV2FeatureSources:
    candidates: Path
    user_profile_root: Path
    business_profile_root: Path


class HybridV2FeatureBuildResult(StrictModel):
    status: Literal["written"] = "written"
    split: Literal["train", "validation", "test"]
    output_path: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    feature_count: int = Field(ge=1)
    feature_names: list[str]


@dataclass(frozen=True, slots=True)
class _Signal:
    score: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _TaskProfile:
    user_id: str
    cutoff_time: datetime
    history_length: int
    average_rating: float
    reliability: float
    sample_weight: float
    fold: int | None
    category_signals: dict[str, _Signal]
    aspect_signals: dict[str, _Signal]


@dataclass(frozen=True, slots=True)
class _AspectHistory:
    dates: tuple[datetime, ...]
    positive_counts: tuple[int, ...]
    negative_counts: tuple[int, ...]
    directional_users: tuple[int, ...]
    positive_weight_bases: tuple[float, ...]
    negative_weight_bases: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _AspectPoint:
    positive_ratio: float
    negative_ratio: float
    confidence: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


class _UserProfileIndex:
    def __init__(self, root: Path, split: str) -> None:
        snapshots_path = _required(
            root / "profile_snapshots.parquet", "User profile snapshots"
        )
        signals_path = _required(
            root / "preference_signals.parquet", "User preference signals"
        )
        task_map_path = _required(root / "task_profile_map.parquet", "Task profile map")
        snapshots = {
            str(row["profile_id"]): row
            for row in pq.read_table(snapshots_path).to_pylist()
        }
        signals: defaultdict[str, dict[str, dict[str, _Signal]]] = defaultdict(
            lambda: {"category": {}, "aspect": {}}
        )
        for row in pq.read_table(
            signals_path,
            columns=["profile_id", "kind", "value", "score", "confidence"],
        ).to_pylist():
            kind = str(row["kind"])
            if kind not in {"category", "aspect"}:
                continue
            profile_id = str(row["profile_id"])
            value = str(row["value"])
            if value in signals[profile_id][kind]:
                raise HybridV2FeatureError(
                    "A user profile contains duplicate normalized signals"
                )
            signals[profile_id][kind][value] = _Signal(
                score=float(row["score"]), confidence=float(row["confidence"])
            )
        self.tasks: dict[str, _TaskProfile] = {}
        for link in pq.read_table(task_map_path).to_pylist():
            if str(link["split"]) != split:
                continue
            task_id = str(link["task_id"])
            profile_id = str(link["profile_id"])
            snapshot = snapshots.get(profile_id)
            if snapshot is None:
                raise HybridV2FeatureError(
                    f"Task {task_id!r} references a missing user profile"
                )
            cutoff = link["cutoff_time"]
            if (
                snapshot["user_id"] != link["user_id"]
                or snapshot["cutoff_time"] != cutoff
                or int(snapshot["history_length"])
                != int(link["expected_history_count"])
            ):
                raise HybridV2FeatureError(
                    f"Task {task_id!r} disagrees with its frozen user profile"
                )
            self.tasks[task_id] = _TaskProfile(
                user_id=str(link["user_id"]),
                cutoff_time=cutoff,
                history_length=int(snapshot["history_length"]),
                average_rating=float(snapshot["average_rating"]),
                reliability=float(snapshot["reliability"]),
                sample_weight=(
                    1.0
                    if link["sample_weight"] is None
                    else float(link["sample_weight"])
                ),
                fold=None if link["fold"] is None else int(link["fold"]),
                category_signals=dict(signals[profile_id]["category"]),
                aspect_signals=dict(signals[profile_id]["aspect"]),
            )
        if not self.tasks:
            raise HybridV2FeatureError(f"No {split!r} user profiles were found")


class _BusinessTemporalIndex:
    _EPOCH = datetime(2000, 1, 1)

    def __init__(self, root: Path, config: BusinessProfileConfig) -> None:
        business_path = _required(root / "businesses.parquet", "Business records")
        rating_path = _required(root / "rating_events.parquet", "Business ratings")
        aspect_path = _required(root / "aspect_events.parquet", "Business aspects")
        self.config = config
        self.categories = {
            str(row["business_id"]): tuple(str(value) for value in row["categories"])
            for row in pq.read_table(
                business_path, columns=["business_id", "categories"]
            ).to_pylist()
        }
        rating_dates: defaultdict[str, list[datetime]] = defaultdict(list)
        for row in pq.read_table(
            rating_path, columns=["business_id", "review_time"]
        ).to_pylist():
            rating_dates[str(row["business_id"])].append(row["review_time"])
        self.rating_dates = {
            business_id: tuple(sorted(dates))
            for business_id, dates in rating_dates.items()
        }

        grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(
            list
        )
        all_aspect_dates: defaultdict[str, list[datetime]] = defaultdict(list)
        for row in pq.read_table(
            aspect_path,
            columns=[
                "business_id",
                "user_id",
                "review_time",
                "aspect",
                "sentiment",
                "confidence",
            ],
        ).to_pylist():
            business_id = str(row["business_id"])
            grouped[(business_id, str(row["aspect"]))].append(row)
            all_aspect_dates[business_id].append(row["review_time"])
        self.all_aspect_dates = {
            business_id: tuple(sorted(dates))
            for business_id, dates in all_aspect_dates.items()
        }
        histories: dict[tuple[str, str], _AspectHistory] = {}
        known_times: defaultdict[str, list[datetime]] = defaultdict(list)
        half_life = float(config.aspect_half_life_days)
        for key, events in grouped.items():
            events.sort(
                key=lambda row: (
                    row["review_time"],
                    str(row["user_id"]),
                    str(row["sentiment"]),
                )
            )
            dates: list[datetime] = []
            positive_counts = [0]
            negative_counts = [0]
            directional_users = [0]
            positive_bases = [0.0]
            negative_bases = [0.0]
            seen_directional_users: set[str] = set()
            known_time: datetime | None = None
            for event in events:
                date = event["review_time"]
                sentiment = str(event["sentiment"])
                dates.append(date)
                positive_counts.append(
                    positive_counts[-1] + int(sentiment == "positive")
                )
                negative_counts.append(
                    negative_counts[-1] + int(sentiment == "negative")
                )
                if sentiment in {"positive", "negative"}:
                    seen_directional_users.add(str(event["user_id"]))
                directional_users.append(len(seen_directional_users))
                age_days = (date - self._EPOCH).total_seconds() / 86400.0
                base = float(event["confidence"]) * 2.0 ** (age_days / half_life)
                positive_bases.append(
                    positive_bases[-1] + (base if sentiment == "positive" else 0.0)
                )
                negative_bases.append(
                    negative_bases[-1] + (base if sentiment == "negative" else 0.0)
                )
                directional_count = positive_counts[-1] + negative_counts[-1]
                if (
                    known_time is None
                    and directional_count >= config.minimum_aspect_evidence
                    and len(seen_directional_users) >= config.minimum_aspect_users
                ):
                    known_time = date
            histories[key] = _AspectHistory(
                dates=tuple(dates),
                positive_counts=tuple(positive_counts),
                negative_counts=tuple(negative_counts),
                directional_users=tuple(directional_users),
                positive_weight_bases=tuple(positive_bases),
                negative_weight_bases=tuple(negative_bases),
            )
            if known_time is not None:
                known_times[key[0]].append(known_time)
        self.aspect_histories = histories
        self.known_aspect_times = {
            business_id: tuple(sorted(times))
            for business_id, times in known_times.items()
        }

    def profile_features(
        self, business_id: str, cutoff: datetime
    ) -> tuple[float, float, float, float, float]:
        if business_id not in self.categories:
            raise HybridV2FeatureError(f"Unknown business ID: {business_id!r}")
        rating_count = bisect_left(self.rating_dates.get(business_id, ()), cutoff)
        aspect_count = bisect_left(self.all_aspect_dates.get(business_id, ()), cutoff)
        known_count = bisect_left(self.known_aspect_times.get(business_id, ()), cutoff)
        rating_factor = 1.0 - math.exp(
            -rating_count / self.config.rating_reliability_saturation
        )
        aspect_factor = 1.0 - math.exp(
            -aspect_count / self.config.aspect_reliability_saturation
        )
        coverage = known_count / 14.0
        reliability = 0.5 * rating_factor + 0.3 * aspect_factor + 0.2 * coverage
        return (
            math.log1p(rating_count),
            rating_factor,
            aspect_factor,
            coverage,
            min(1.0, max(0.0, reliability)),
        )

    def aspect_point(
        self, business_id: str, aspect: str, cutoff: datetime
    ) -> _AspectPoint | None:
        history = self.aspect_histories.get((business_id, aspect))
        if history is None:
            return None
        index = bisect_left(history.dates, cutoff)
        positive_count = history.positive_counts[index]
        negative_count = history.negative_counts[index]
        directional_count = positive_count + negative_count
        users = history.directional_users[index]
        if (
            directional_count < self.config.minimum_aspect_evidence
            or users < self.config.minimum_aspect_users
        ):
            return None
        positive_base = history.positive_weight_bases[index]
        negative_base = history.negative_weight_bases[index]
        base_total = positive_base + negative_base
        if base_total <= 0:
            return None
        positive_ratio = positive_base / base_total
        negative_ratio = negative_base / base_total
        cutoff_days = (cutoff - self._EPOCH).total_seconds() / 86400.0
        directional_weight = base_total * 2.0 ** (
            -cutoff_days / self.config.aspect_half_life_days
        )
        volume = 1.0 - math.exp(
            -directional_weight / self.config.aspect_confidence_saturation
        )
        diversity = 1.0 - math.exp(-users / self.config.aspect_user_saturation)
        consistency = max(positive_ratio, negative_ratio)
        confidence = volume * (0.5 + 0.25 * diversity + 0.25 * consistency)
        return _AspectPoint(
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            confidence=min(1.0, max(0.0, confidence)),
        )


def _category_features(
    profile: _TaskProfile,
    categories: tuple[str, ...],
    broad_categories: set[str],
) -> tuple[float, float, float, float]:
    signals = profile.category_signals
    positive = max(
        (
            signal.score * signal.confidence
            for category in categories
            if (signal := signals.get(category)) is not None and signal.score > 0
        ),
        default=0.0,
    )
    negative = max(
        (
            abs(signal.score) * signal.confidence
            for category in categories
            if (signal := signals.get(category)) is not None and signal.score < 0
        ),
        default=0.0,
    )
    fine_categories = [value for value in categories if value not in broad_categories]
    seen = any(value in signals for value in fine_categories)
    return (
        round(positive, 12),
        round(negative, 12),
        float(bool(fine_categories) and not seen),
        float(not signals),
    )


def _aspect_features(
    profile: _TaskProfile,
    business_id: str,
    business_index: _BusinessTemporalIndex,
) -> tuple[float, float, float, float]:
    signals = profile.aspect_signals
    if not signals:
        return (0.0, 0.0, 0.0, 1.0)
    positive_total = sum(
        signal.score * signal.confidence
        for signal in signals.values()
        if signal.score > 0
    )
    negative_total = sum(
        abs(signal.score) * signal.confidence
        for signal in signals.values()
        if signal.score < 0
    )
    all_total = sum(
        abs(signal.score) * signal.confidence for signal in signals.values()
    )
    positive_match = 0.0
    negative_conflict = 0.0
    confidence_sum = 0.0
    known = 0
    for aspect, signal in signals.items():
        point = business_index.aspect_point(business_id, aspect, profile.cutoff_time)
        if point is None:
            continue
        known += 1
        strength = abs(signal.score) * signal.confidence
        confidence_sum += strength * point.confidence
        if signal.score > 0:
            positive_match += strength * point.confidence * point.positive_ratio
        else:
            negative_conflict += strength * point.confidence * point.negative_ratio
    return (
        0.0 if positive_total <= 0 else positive_match / positive_total,
        0.0 if negative_total <= 0 else negative_conflict / negative_total,
        0.0 if all_total <= 0 else confidence_sum / all_total,
        float(known == 0),
    )


def _optional_score(row: dict[str, object], name: str) -> float:
    value = row[name]
    return 0.0 if value is None else float(value)


def _feature_row(
    row: dict[str, object],
    profile: _TaskProfile,
    business_index: _BusinessTemporalIndex,
    weights: HybridV1Weights,
    broad_categories: set[str],
) -> dict[str, object]:
    business_id = str(row["business_id"])
    categories = business_index.categories.get(business_id)
    if categories is None:
        raise HybridV2FeatureError(f"Unknown business ID: {business_id!r}")
    quality = _optional_score(row, "quality_score")
    category = _optional_score(row, "category_score")
    text = _optional_score(row, "text_score")
    location = _optional_score(row, "location_score")
    category_features = _category_features(profile, categories, broad_categories)
    business_features = business_index.profile_features(
        business_id, profile.cutoff_time
    )
    aspect_features = _aspect_features(profile, business_id, business_index)
    distance = row["distance_km"]
    values: dict[str, float] = {
        "retrieval_fusion_score": float(row["fusion_score"]),
        "retrieval_reciprocal_rank": 1.0 / int(row["rank"]),
        "route_coverage": float(row["route_count"]) / 5.0,
        "quality_score": quality,
        "category_score": category,
        "text_score": text,
        "location_score": location,
        "distance_km_log": 0.0
        if distance is None
        else math.log1p(max(0.0, float(distance))),
        "hybrid_v1_score": (
            weights.category * category
            + weights.text * text
            + weights.quality * quality
            + weights.location * location
        ),
        "quality_route_missing": float(row["quality_rank"] is None),
        "category_route_missing": float(row["category_rank"] is None),
        "text_route_missing": float(row["text_rank"] is None),
        "location_route_missing": float(row["location_rank"] is None),
        "item_knn_positive_score": float(row["item_knn_positive_score"]),
        "item_knn_negative_evidence": float(row["item_knn_negative_evidence"]),
        "item_knn_positive_support_log": math.log1p(
            int(row["item_knn_positive_support_count"])
        ),
        "item_knn_negative_support_log": math.log1p(
            int(row["item_knn_negative_support_count"])
        ),
        "item_knn_positive_neighbors_log": math.log1p(
            int(row["item_knn_positive_neighbor_count"])
        ),
        "item_knn_negative_neighbors_log": math.log1p(
            int(row["item_knn_negative_neighbor_count"])
        ),
        "item_knn_missing": float(bool(row["item_knn_missing"])),
        "user_history_length_log": math.log1p(profile.history_length),
        "user_average_rating_normalized": (profile.average_rating - 1.0) / 4.0,
        "user_profile_reliability": profile.reliability,
        "user_category_positive_match": category_features[0],
        "user_category_negative_conflict": category_features[1],
        "user_category_novelty": category_features[2],
        "user_category_profile_missing": category_features[3],
        "business_rating_count_log": business_features[0],
        "business_rating_evidence_saturation": business_features[1],
        "business_aspect_evidence_saturation": business_features[2],
        "business_known_aspect_coverage": business_features[3],
        "business_profile_reliability": business_features[4],
        "aspect_positive_match": aspect_features[0],
        "aspect_negative_conflict": aspect_features[1],
        "aspect_match_confidence": aspect_features[2],
        "aspect_evidence_missing": aspect_features[3],
    }
    if set(values) != set(ALL_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise HybridV2FeatureError("Hybrid V2 produced invalid numeric features")
    return {
        "task_id": str(row["task_id"]),
        "split": "",
        "user_id": profile.user_id,
        "cutoff_time": profile.cutoff_time,
        "business_id": business_id,
        "retrieval_rank": int(row["rank"]),
        "label": row.get("label"),
        "negative_kind": row.get("negative_kind"),
        "sample_weight": profile.sample_weight,
        "fold": profile.fold,
        **values,
    }


class _FrozenBusinessProfileView:
    """Present stored profiles through the same feature seam as offline events."""

    def __init__(
        self,
        profiles: dict[str, BusinessProfileV1],
        config: BusinessProfileConfig,
    ) -> None:
        self._profiles = profiles
        self.config = config
        self.categories = {
            business_id: tuple(profile.categories)
            for business_id, profile in profiles.items()
        }

    def profile_features(
        self,
        business_id: str,
        cutoff: datetime,
    ) -> tuple[float, float, float, float, float]:
        profile = self._profiles[business_id]
        if profile.cutoff_time != cutoff:
            raise HybridV2FeatureError("business profile cutoff does not match request")
        evidence = profile.evidence_summary
        rating_factor = 1.0 - math.exp(
            -evidence.rating_count / self.config.rating_reliability_saturation
        )
        aspect_factor = 1.0 - math.exp(
            -evidence.aspect_evidence_count
            / self.config.aspect_reliability_saturation
        )
        coverage = evidence.known_aspect_count / 14.0
        return (
            math.log1p(evidence.rating_count),
            rating_factor,
            aspect_factor,
            coverage,
            profile.profile_reliability,
        )

    def aspect_point(
        self,
        business_id: str,
        aspect: str,
        cutoff: datetime,
    ) -> _AspectPoint | None:
        profile = self._profiles[business_id]
        if profile.cutoff_time != cutoff:
            raise HybridV2FeatureError("business profile cutoff does not match request")
        summary = profile.aspect_summaries.get(aspect)  # type: ignore[arg-type]
        if summary is None or summary.status == "unknown":
            return None
        return _AspectPoint(
            positive_ratio=float(summary.weighted_positive_ratio),
            negative_ratio=float(summary.weighted_negative_ratio),
            confidence=summary.confidence,
        )


def build_online_hybrid_v2_features(
    *,
    request_id: str,
    profile: UserProfileV1,
    business_profiles: dict[str, BusinessProfileV1],
    candidates: list[dict[str, object]],
    weights: HybridV1Weights,
    broad_categories: set[str],
    business_profile_config: BusinessProfileConfig,
) -> list[dict[str, object]]:
    """Build the exact frozen feature contract for one live Agent request."""

    if not candidates:
        raise HybridV2FeatureError("online candidates cannot be empty")
    business_ids = [str(row.get("business_id") or "") for row in candidates]
    if any(not value for value in business_ids) or len(set(business_ids)) != len(
        business_ids
    ):
        raise HybridV2FeatureError("online candidate IDs must be nonempty and unique")
    if set(business_ids) != set(business_profiles):
        raise HybridV2FeatureError("business profiles must exactly match candidates")
    if any(item.cutoff_time != profile.cutoff_time for item in business_profiles.values()):
        raise HybridV2FeatureError("online profile cutoffs must match")
    signals = profile.preference_signals()
    task_profile = _TaskProfile(
        user_id=profile.user_id,
        cutoff_time=profile.cutoff_time,
        history_length=profile.history_length,
        average_rating=profile.average_rating,
        reliability=profile.reliability,
        sample_weight=1.0,
        fold=None,
        category_signals={
            signal.value: _Signal(signal.score, signal.confidence)
            for signal in signals
            if signal.kind == "category"
        },
        aspect_signals={
            signal.value: _Signal(signal.score, signal.confidence)
            for signal in signals
            if signal.kind == "aspect"
        },
    )
    business_view = _FrozenBusinessProfileView(
        business_profiles,
        business_profile_config,
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        source = {"task_id": request_id, **candidate}
        rows.append(
            _feature_row(
                source,
                task_profile,
                business_view,  # type: ignore[arg-type]
                weights,
                broad_categories,
            )
        )
    return rows


def build_hybrid_v2_features(
    sources: HybridV2FeatureSources,
    output_path: str | Path,
    *,
    split: Literal["train", "validation", "test"],
    weights: HybridV1Weights,
    broad_categories: set[str],
    business_profile_config: BusinessProfileConfig,
    batch_size: int,
) -> HybridV2FeatureBuildResult:
    """Build compact numeric features without accepting any evaluation labels."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not broad_categories:
        raise ValueError("broad_categories cannot be empty")
    candidates = _required(Path(sources.candidates), "Retrieval candidates")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Hybrid V2 features already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    profiles = _UserProfileIndex(Path(sources.user_profile_root), split)
    businesses = _BusinessTemporalIndex(
        Path(sources.business_profile_root), business_profile_config
    )
    candidate_file = pq.ParquetFile(candidates)
    candidate_names = set(candidate_file.schema_arrow.names)
    required_candidate_names = {
        "task_id",
        "rank",
        "business_id",
        "fusion_score",
        "route_count",
        "quality_rank",
        "quality_score",
        "category_rank",
        "category_score",
        "text_rank",
        "text_score",
        "location_rank",
        "location_score",
        "distance_km",
        "item_knn_positive_score",
        "item_knn_negative_evidence",
        "item_knn_positive_support_count",
        "item_knn_negative_support_count",
        "item_knn_positive_neighbor_count",
        "item_knn_negative_neighbor_count",
        "item_knn_missing",
    }
    if not required_candidate_names.issubset(candidate_names):
        raise HybridV2FeatureError("Candidate artifact has an unexpected schema")
    columns = sorted(required_candidate_names)
    for optional in ("label", "negative_kind"):
        if optional in candidate_names:
            columns.append(optional)
    writer = pq.ParquetWriter(
        partial,
        HYBRID_V2_FEATURE_SCHEMA,
        compression="zstd",
        use_dictionary=["task_id", "split", "user_id", "business_id", "negative_kind"],
    )
    row_count = 0
    task_ids: set[str] = set()
    try:
        for batch in candidate_file.iter_batches(
            batch_size=batch_size, columns=columns
        ):
            output_rows: list[dict[str, object]] = []
            for candidate in batch.to_pylist():
                task_id = str(candidate["task_id"])
                profile = profiles.tasks.get(task_id)
                if profile is None:
                    raise HybridV2FeatureError(
                        f"Candidate task {task_id!r} has no exact {split} profile"
                    )
                result = _feature_row(
                    candidate, profile, businesses, weights, broad_categories
                )
                result["split"] = split
                output_rows.append(result)
                task_ids.add(task_id)
            if output_rows:
                writer.write_table(
                    pa.Table.from_pylist(output_rows, schema=HYBRID_V2_FEATURE_SCHEMA)
                )
                row_count += len(output_rows)
    except Exception:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    if row_count == 0:
        partial.unlink(missing_ok=True)
        raise HybridV2FeatureError("Candidate artifact contains no rows")
    os.replace(partial, output)
    return HybridV2FeatureBuildResult(
        split=split,
        output_path=str(output),
        output_sha256=_sha256(output),
        task_count=len(task_ids),
        row_count=row_count,
        feature_count=len(ALL_FEATURE_NAMES),
        feature_names=list(ALL_FEATURE_NAMES),
    )
