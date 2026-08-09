"""Adapters from frozen Yelp artifacts into one Step 20 in-memory catalog."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from yelp_agent.config import AgentBenchmarkConfig
from yelp_agent.query.benchmark import QueryBenchmarkCase, load_query_benchmark

from .schema import ScenarioSplit


@dataclass(frozen=True, slots=True)
class AgentBenchmarkSources:
    businesses: Path
    rating_events: Path
    aspect_records: Path
    profile_snapshots: Path
    preference_signals: Path
    task_profile_map: Path
    query_benchmark: Path


@dataclass(frozen=True, slots=True)
class UserContextRecord:
    task_id: str
    profile_id: str
    user_id: str
    cutoff_time: datetime
    history_length: int
    reliability: float
    latitude: float | None
    longitude: float | None
    preferred_categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BusinessRecord:
    business_id: str
    name: str
    postal_code: str
    latitude: float | None
    longitude: float | None
    categories: tuple[str, ...]
    attributes: dict[str, object]
    first_review_time: datetime


@dataclass(frozen=True, slots=True)
class ReviewEvidenceRecord:
    review_id: str
    business_id: str
    review_time: datetime
    aspect: str
    sentiment: str
    confidence: float
    source_text_sha256: str


@dataclass(frozen=True, slots=True)
class BenchmarkCatalog:
    user_contexts: dict[ScenarioSplit, tuple[UserContextRecord, ...]]
    businesses: dict[ScenarioSplit, tuple[BusinessRecord, ...]]
    evidence_by_business: dict[str, tuple[ReviewEvidenceRecord, ...]]
    query_cases: tuple[QueryBenchmarkCase, ...]


def stable_partition(
    value: str,
    *,
    seed: int,
    namespace: str,
) -> ScenarioSplit:
    payload = f"{seed}\x1f{namespace}\x1f{value}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 5
    return "validation" if bucket == 0 else "development"


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_user_contexts(
    sources: AgentBenchmarkSources,
    config: AgentBenchmarkConfig,
) -> dict[ScenarioSplit, tuple[UserContextRecord, ...]]:
    snapshots = {
        str(row["profile_id"]): row
        for row in pq.read_table(sources.profile_snapshots).to_pylist()
    }
    candidates: defaultdict[
        ScenarioSplit,
        defaultdict[str, list[dict[str, object]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in pq.read_table(sources.task_profile_map).to_pylist():
        source_split = str(row["split"])
        if source_split == "test":
            continue
        user_id = str(row["user_id"])
        benchmark_split = stable_partition(
            user_id,
            seed=config.random_seed,
            namespace="user",
        )
        if benchmark_split == "development" and source_split != "train":
            continue
        if benchmark_split == "validation" and source_split != "validation":
            continue
        snapshot = snapshots.get(str(row["profile_id"]))
        if snapshot is None:
            continue
        cutoff = row["cutoff_time"]
        if (
            cutoff.year < config.minimum_cutoff_year
            or int(snapshot["history_length"]) < config.minimum_profile_history
            or float(snapshot["reliability"]) < config.minimum_profile_reliability
        ):
            continue
        candidates[benchmark_split][user_id].append(row)

    selected_rows: list[tuple[ScenarioSplit, dict[str, object]]] = []
    for split in ("development", "validation"):
        for user_id, rows in candidates[split].items():
            latest = max(rows, key=lambda row: (row["cutoff_time"], row["task_id"]))
            selected_rows.append((split, latest))
    selected_profile_ids = {str(row["profile_id"]) for _, row in selected_rows}
    preferences: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
    table = pq.read_table(
        sources.preference_signals,
        columns=["profile_id", "kind", "value", "score"],
    )
    for row in table.to_pylist():
        profile_id = str(row["profile_id"])
        if (
            profile_id in selected_profile_ids
            and row["kind"] == "category"
            and float(row["score"]) > 0
        ):
            preferences[profile_id].append((float(row["score"]), str(row["value"])))

    result: defaultdict[ScenarioSplit, list[UserContextRecord]] = defaultdict(list)
    for split, row in selected_rows:
        profile_id = str(row["profile_id"])
        snapshot = snapshots[profile_id]
        preferred = tuple(
            value
            for _, value in sorted(
                preferences.get(profile_id, []),
                key=lambda item: (-item[0], item[1]),
            )[:10]
        )
        if not preferred:
            continue
        result[split].append(
            UserContextRecord(
                task_id=str(row["task_id"]),
                profile_id=profile_id,
                user_id=str(row["user_id"]),
                cutoff_time=row["cutoff_time"],
                history_length=int(snapshot["history_length"]),
                reliability=float(snapshot["reliability"]),
                latitude=(
                    None
                    if snapshot["location_latitude"] is None
                    else float(snapshot["location_latitude"])
                ),
                longitude=(
                    None
                    if snapshot["location_longitude"] is None
                    else float(snapshot["location_longitude"])
                ),
                preferred_categories=preferred,
            )
        )
    for split in result:
        result[split].sort(
            key=lambda item: hashlib.sha256(
                f"{config.random_seed}:{item.user_id}".encode()
            ).hexdigest()
        )
    return {split: tuple(result[split]) for split in ("development", "validation")}


def _load_businesses(
    sources: AgentBenchmarkSources,
    config: AgentBenchmarkConfig,
) -> dict[ScenarioSplit, tuple[BusinessRecord, ...]]:
    connection = duckdb.connect()
    try:
        first_reviews = {
            str(business_id): review_time
            for business_id, review_time in connection.execute(
                """
                SELECT business_id, min(review_time)
                FROM read_parquet(?)
                GROUP BY business_id
                """,
                [str(sources.rating_events)],
            ).fetchall()
        }
    finally:
        connection.close()
    result: defaultdict[ScenarioSplit, list[BusinessRecord]] = defaultdict(list)
    for row in pq.read_table(sources.businesses).to_pylist():
        business_id = str(row["business_id"])
        first_review = first_reviews.get(business_id)
        categories = tuple(str(value) for value in (row["categories"] or []))
        if first_review is None or not categories:
            continue
        try:
            attributes = json.loads(str(row["attributes_json"] or "{}"))
        except json.JSONDecodeError:
            attributes = {}
        split = stable_partition(
            business_id,
            seed=config.random_seed,
            namespace="business",
        )
        result[split].append(
            BusinessRecord(
                business_id=business_id,
                name=str(row["name"]),
                postal_code=str(row["postal_code"] or ""),
                latitude=(None if row["latitude"] is None else float(row["latitude"])),
                longitude=(
                    None if row["longitude"] is None else float(row["longitude"])
                ),
                categories=categories,
                attributes=attributes,
                first_review_time=first_review,
            )
        )
    for split in result:
        result[split].sort(key=lambda item: item.business_id)
    return {split: tuple(result[split]) for split in ("development", "validation")}


def _load_evidence(
    sources: AgentBenchmarkSources,
    config: AgentBenchmarkConfig,
) -> dict[str, tuple[ReviewEvidenceRecord, ...]]:
    grouped: defaultdict[str, list[ReviewEvidenceRecord]] = defaultdict(list)
    table = pq.read_table(
        sources.aspect_records,
        columns=[
            "review_id",
            "business_id",
            "review_time",
            "aspect",
            "sentiment",
            "confidence",
            "source_text_sha256",
        ],
    )
    for row in table.to_pylist():
        confidence = float(row["confidence"])
        if confidence < config.minimum_review_evidence_confidence:
            continue
        business_id = str(row["business_id"])
        grouped[business_id].append(
            ReviewEvidenceRecord(
                review_id=str(row["review_id"]),
                business_id=business_id,
                review_time=row["review_time"],
                aspect=str(row["aspect"]),
                sentiment=str(row["sentiment"]),
                confidence=confidence,
                source_text_sha256=str(row["source_text_sha256"]),
            )
        )
    return {
        business_id: tuple(
            sorted(events, key=lambda event: (event.review_time, event.review_id))
        )
        for business_id, events in grouped.items()
    }


def load_benchmark_catalog(
    sources: AgentBenchmarkSources,
    config: AgentBenchmarkConfig,
) -> BenchmarkCatalog:
    """Read source artifacts once and hide their layout from scenario builders."""

    for path, label in (
        (sources.businesses, "businesses"),
        (sources.rating_events, "rating events"),
        (sources.aspect_records, "aspect records"),
        (sources.profile_snapshots, "profile snapshots"),
        (sources.preference_signals, "preference signals"),
        (sources.task_profile_map, "task profile map"),
        (sources.query_benchmark, "query benchmark"),
    ):
        _required(path, label)
    catalog = BenchmarkCatalog(
        user_contexts=_load_user_contexts(sources, config),
        businesses=_load_businesses(sources, config),
        evidence_by_business=_load_evidence(sources, config),
        query_cases=load_query_benchmark(sources.query_benchmark),
    )
    for split in ("development", "validation"):
        if len(catalog.user_contexts[split]) < (
            config.development_count if split == "development" else config.validation_count
        ):
            raise ValueError(f"not enough {split} user contexts for Step 20")
        if len(catalog.businesses[split]) < config.candidate_scope_size:
            raise ValueError(f"not enough {split} businesses for Step 20")
    return catalog
