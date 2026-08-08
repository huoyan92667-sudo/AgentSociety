"""Build compact business-knowledge indexes without carrying review text."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.business_profiles.schema import (
    BUSINESS_ASPECT_EVENT_SCHEMA,
    BUSINESS_COVERAGE_SCHEMA,
    BUSINESS_RATING_EVENT_SCHEMA,
)
from yelp_agent.config import BusinessProfileConfig
from yelp_agent.data.businesses import BUSINESS_SCHEMA
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel
from yelp_agent.reviews.schema import REVIEW_ASPECT_SCHEMA


class BusinessKnowledgeArtifactError(RuntimeError):
    """Raised when business-knowledge sources or outputs are invalid."""


class BusinessKnowledgeManifest(StrictModel):
    format_version: Literal[1] = 1
    artifact_name: Literal["Business Knowledge V1"] = "Business Knowledge V1"
    profile_version: Literal["1.0.0"]
    source_scope: Literal["selected_user_interactions"]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: dict[str, str]
    businesses: int = Field(ge=1)
    rating_events: int = Field(ge=0)
    aspect_events: int = Field(ge=0)
    aspect_businesses: int = Field(ge=0)
    coverage_rows: int = Field(ge=0)
    review_text_loaded: Literal[False] = False
    evidence_span_loaded: Literal[False] = False
    duplicate_rating_review_ids: Literal[0] = 0
    aspect_source_mismatches: Literal[0] = 0


class BusinessKnowledgeBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    artifact_root: str
    manifest_path: str
    businesses: int = Field(ge=1)
    rating_events: int = Field(ge=0)
    aspect_events: int = Field(ge=0)
    coverage_rows: int = Field(ge=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source(path: Path, schema: pa.Schema) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Business-knowledge source does not exist: {path}")
    try:
        actual = pq.ParquetFile(path).schema_arrow
    except (OSError, pa.ArrowException) as exc:
        raise BusinessKnowledgeArtifactError(f"Could not read source: {path}") from exc
    if not actual.equals(schema, check_metadata=False):
        raise BusinessKnowledgeArtifactError(
            f"Business-knowledge source has an unexpected schema: {path}"
        )


def _result(
    status: Literal["written", "skipped"],
    root: Path,
    manifest: BusinessKnowledgeManifest,
) -> BusinessKnowledgeBuildResult:
    return BusinessKnowledgeBuildResult(
        status=status,
        artifact_root=str(root),
        manifest_path=str(root / "manifest.json"),
        businesses=manifest.businesses,
        rating_events=manifest.rating_events,
        aspect_events=manifest.aspect_events,
        coverage_rows=manifest.coverage_rows,
    )


def _reusable_manifest(
    root: Path,
    *,
    source_sha256: dict[str, str],
    configuration_sha256: str,
) -> BusinessKnowledgeManifest | None:
    schemas = {
        "businesses": BUSINESS_SCHEMA,
        "ratings": BUSINESS_RATING_EVENT_SCHEMA,
        "aspects": BUSINESS_ASPECT_EVENT_SCHEMA,
        "coverage": BUSINESS_COVERAGE_SCHEMA,
    }
    paths = {
        "businesses": root / "businesses.parquet",
        "ratings": root / "rating_events.parquet",
        "aspects": root / "aspect_events.parquet",
        "coverage": root / "business_coverage.parquet",
    }
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or not all(
        path.is_file() for path in paths.values()
    ):
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not {
            "duplicate_rating_review_ids",
            "aspect_source_mismatches",
        }.issubset(payload):
            return None
        manifest = BusinessKnowledgeManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None
    if (
        manifest.source_sha256 != source_sha256
        or manifest.configuration_sha256 != configuration_sha256
        or manifest.output_sha256
        != {name: _sha256_file(path) for name, path in paths.items()}
    ):
        return None
    for name, path in paths.items():
        try:
            actual = pq.ParquetFile(path).schema_arrow
        except (OSError, pa.ArrowException):
            return None
        if not actual.equals(schemas[name], check_metadata=False):
            return None
    return manifest


def _compact_rating_events(reviews_path: Path) -> pa.Table:
    table = pq.read_table(
        reviews_path,
        columns=["review_id", "business_id", "user_id", "stars", "date"],
    ).rename_columns(["review_id", "business_id", "user_id", "stars", "review_time"])
    return table.cast(BUSINESS_RATING_EVENT_SCHEMA).sort_by(
        [
            ("business_id", "ascending"),
            ("review_time", "ascending"),
            ("review_id", "ascending"),
        ]
    )


def _compact_aspect_events(aspect_records_path: Path) -> pa.Table:
    table = pq.read_table(
        aspect_records_path,
        columns=[
            "review_id",
            "business_id",
            "user_id",
            "review_time",
            "aspect",
            "sentiment",
            "confidence",
            "source_text_sha256",
            "extractor_version",
        ],
    )
    return table.cast(BUSINESS_ASPECT_EVENT_SCHEMA).sort_by(
        [
            ("business_id", "ascending"),
            ("aspect", "ascending"),
            ("review_time", "ascending"),
            ("review_id", "ascending"),
        ]
    )


def _coverage_table(aspect_events: pa.Table) -> pa.Table:
    connection = duckdb.connect()
    try:
        connection.register("aspect_events", aspect_events)
        result = connection.execute(
            """
            SELECT
                business_id,
                aspect,
                count(*)::BIGINT AS evidence_count,
                count(DISTINCT review_id)::BIGINT AS review_count,
                count(DISTINCT user_id)::BIGINT AS unique_users,
                min(review_time) AS first_evidence_time,
                max(review_time) AS latest_evidence_time
            FROM aspect_events
            GROUP BY business_id, aspect
            ORDER BY business_id, aspect
            """
        ).to_arrow_table()
    finally:
        connection.close()
    return result.cast(BUSINESS_COVERAGE_SCHEMA)


def _validate_event_lineage(
    rating_events: pa.Table,
    aspect_events: pa.Table,
) -> None:
    connection = duckdb.connect()
    try:
        connection.register("rating_events", rating_events)
        connection.register("aspect_events", aspect_events)
        duplicate_ratings = connection.execute(
            """
            SELECT count(*)
            FROM (
                SELECT review_id
                FROM rating_events
                GROUP BY review_id
                HAVING count(*) <> 1
            )
            """
        ).fetchone()[0]
        if duplicate_ratings:
            raise BusinessKnowledgeArtifactError(
                "Rating event review IDs must be unique"
            )
        source_mismatches = connection.execute(
            """
            SELECT count(*)
            FROM aspect_events AS aspect
            LEFT JOIN rating_events AS rating USING (review_id)
            WHERE rating.review_id IS NULL
               OR aspect.business_id <> rating.business_id
               OR aspect.user_id <> rating.user_id
               OR aspect.review_time <> rating.review_time
            """
        ).fetchone()[0]
        if source_mismatches:
            raise BusinessKnowledgeArtifactError(
                "Aspect evidence does not match its source review"
            )
    finally:
        connection.close()


def build_business_knowledge_artifacts(
    *,
    businesses_path: str | Path,
    reviews_path: str | Path,
    aspect_records_path: str | Path,
    output_root: str | Path,
    config: BusinessProfileConfig,
) -> BusinessKnowledgeBuildResult:
    """Create compact indexes from projected columns only; no ground truth."""

    sources = {
        "businesses": Path(businesses_path),
        "reviews": Path(reviews_path),
        "review_aspects": Path(aspect_records_path),
    }
    for path, schema in (
        (sources["businesses"], BUSINESS_SCHEMA),
        (sources["reviews"], REVIEW_SCHEMA),
        (sources["review_aspects"], REVIEW_ASPECT_SCHEMA),
    ):
        _validate_source(path, schema)
    source_sha256 = {name: _sha256_file(path) for name, path in sources.items()}
    configuration_sha256 = _sha256_json(config.model_dump(mode="json"))
    root = Path(output_root)
    reusable = _reusable_manifest(
        root,
        source_sha256=source_sha256,
        configuration_sha256=configuration_sha256,
    )
    if reusable is not None:
        return _result("skipped", root, reusable)

    businesses = pq.read_table(sources["businesses"]).sort_by(
        [("business_id", "ascending")]
    )
    ratings = _compact_rating_events(sources["reviews"])
    aspects = _compact_aspect_events(sources["review_aspects"])
    business_ids = set(businesses.column("business_id").to_pylist())
    if len(business_ids) != businesses.num_rows:
        raise BusinessKnowledgeArtifactError("business IDs must be unique")
    for label, table in (("rating", ratings), ("aspect", aspects)):
        unknown = set(table.column("business_id").to_pylist()).difference(business_ids)
        if unknown:
            raise BusinessKnowledgeArtifactError(
                f"{label} events reference unknown businesses: {sorted(unknown)[:3]}"
            )
    _validate_event_lineage(ratings, aspects)
    coverage = _coverage_table(aspects)
    root.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "businesses": root / "businesses.parquet",
        "ratings": root / "rating_events.parquet",
        "aspects": root / "aspect_events.parquet",
        "coverage": root / "business_coverage.parquet",
    }
    partial_paths = {
        name: path.with_name(path.name + ".partial")
        for name, path in final_paths.items()
    }
    for path in partial_paths.values():
        path.unlink(missing_ok=True)
    tables = {
        "businesses": businesses.cast(BUSINESS_SCHEMA),
        "ratings": ratings,
        "aspects": aspects,
        "coverage": coverage,
    }
    try:
        for name, table in tables.items():
            pq.write_table(
                table,
                partial_paths[name],
                compression="zstd",
                use_dictionary=True,
            )
        for name, path in final_paths.items():
            os.replace(partial_paths[name], path)
    except Exception:
        for path in partial_paths.values():
            path.unlink(missing_ok=True)
        raise

    manifest = BusinessKnowledgeManifest(
        profile_version=config.profile_version,
        source_scope=config.source_scope,
        source_sha256=source_sha256,
        configuration_sha256=configuration_sha256,
        output_sha256={name: _sha256_file(path) for name, path in final_paths.items()},
        businesses=businesses.num_rows,
        rating_events=ratings.num_rows,
        aspect_events=aspects.num_rows,
        aspect_businesses=len(set(aspects.column("business_id").to_pylist())),
        coverage_rows=coverage.num_rows,
    )
    write_json_artifact(root / "manifest.json", manifest)
    return _result("written", root, manifest)
