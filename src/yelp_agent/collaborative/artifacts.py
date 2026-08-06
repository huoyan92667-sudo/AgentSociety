"""Build reusable, audited event inputs for temporal Item-KNN."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.config import ItemKNNConfig
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel

ITEM_KNN_EVENT_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("stars", pa.float64(), nullable=False),
        pa.field("date", pa.timestamp("us"), nullable=False),
    ]
)


class ItemKNNArtifactError(RuntimeError):
    """Raised when Item-KNN event artifacts are invalid or stale."""


class ItemKNNArtifactManifest(StrictModel):
    format_version: Literal[1] = 1
    artifact_name: Literal["Temporal Item-KNN Events V1"] = (
        "Temporal Item-KNN Events V1"
    )
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    excluded_users_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: ItemKNNConfig
    source_interactions: int = Field(ge=1)
    retained_interactions: int = Field(ge=0)
    reserved_interactions: int = Field(ge=0)
    excluded_user_interactions: int = Field(ge=0)
    positive_interactions: int = Field(ge=0)
    negative_interactions: int = Field(ge=0)
    neutral_interactions: int = Field(ge=0)
    output_sha256: dict[str, str]


class ItemKNNArtifactResult(StrictModel):
    status: Literal["written", "skipped"]
    manifest_path: str
    positive_events_path: str
    negative_events_path: str
    neutral_events_path: str
    retained_interactions: int = Field(ge=0)
    reserved_interactions: int = Field(ge=0)
    excluded_user_interactions: int = Field(ge=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_users_sha256(user_ids: frozenset[str]) -> str:
    payload = "".join(f"{user_id}\n" for user_id in sorted(user_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paths(root: Path) -> dict[str, Path]:
    return {
        "positive_events": root / "positive_events.parquet",
        "negative_events": root / "negative_events.parquet",
        "neutral_events": root / "neutral_events.parquet",
    }


def _result(
    status: Literal["written", "skipped"],
    manifest_path: Path,
    paths: dict[str, Path],
    manifest: ItemKNNArtifactManifest,
) -> ItemKNNArtifactResult:
    return ItemKNNArtifactResult(
        status=status,
        manifest_path=str(manifest_path),
        positive_events_path=str(paths["positive_events"]),
        negative_events_path=str(paths["negative_events"]),
        neutral_events_path=str(paths["neutral_events"]),
        retained_interactions=manifest.retained_interactions,
        reserved_interactions=manifest.reserved_interactions,
        excluded_user_interactions=manifest.excluded_user_interactions,
    )


def _load_manifest(path: Path) -> ItemKNNArtifactManifest:
    try:
        return ItemKNNArtifactManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise ItemKNNArtifactError(
            f"Item-KNN artifact manifest is invalid: {path}"
        ) from exc


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=ITEM_KNN_EVENT_SCHEMA),
            partial,
            compression="zstd",
            use_dictionary=["user_id", "business_id"],
        )
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def build_item_knn_artifacts(
    interactions_path: str | Path,
    output_root: str | Path,
    config: ItemKNNConfig,
    *,
    excluded_user_ids: set[str] | frozenset[str] = frozenset(),
    force: bool = False,
) -> ItemKNNArtifactResult:
    """Freeze graph-eligible events without exposing reserved targets."""

    source = Path(interactions_path)
    if not source.is_file():
        raise FileNotFoundError(f"Interaction Parquet does not exist: {source}")
    excluded = frozenset(excluded_user_ids)
    if any(not user_id for user_id in excluded):
        raise ValueError("excluded_user_ids cannot contain empty IDs")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    paths = _paths(root)
    source_sha256 = _sha256(source)
    users_sha256 = _excluded_users_sha256(excluded)

    if manifest_path.exists() and not force:
        manifest = _load_manifest(manifest_path)
        if (
            manifest.source_sha256 != source_sha256
            or manifest.excluded_users_sha256 != users_sha256
            or manifest.configuration != config
        ):
            raise ItemKNNArtifactError(
                "Item-KNN artifact inputs changed; use force=True to rebuild"
            )
        for name, path in paths.items():
            if not path.is_file() or manifest.output_sha256.get(name) != _sha256(path):
                raise ItemKNNArtifactError(
                    f"Item-KNN artifact output changed or is missing: {name}"
                )
        return _result("skipped", manifest_path, paths, manifest)
    if any(path.exists() for path in paths.values()) and not force:
        raise ItemKNNArtifactError(
            "Item-KNN artifacts are incomplete; use force=True to rebuild"
        )

    try:
        source_rows = pq.read_table(
            source,
            columns=["user_id", "review_id", "business_id", "stars", "date"],
        ).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise ItemKNNArtifactError(
            f"Could not read Item-KNN interactions: {source}"
        ) from exc

    by_user: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_review_ids: set[str] = set()
    excluded_interactions = 0
    for row in source_rows:
        user_id = str(row.get("user_id") or "")
        review_id = str(row.get("review_id") or "")
        business_id = str(row.get("business_id") or "")
        stars = row.get("stars")
        date = row.get("date")
        if (
            not user_id
            or not review_id
            or review_id in seen_review_ids
            or not business_id
            or not isinstance(stars, (int, float))
            or not 1.0 <= float(stars) <= 5.0
            or not isinstance(date, datetime)
        ):
            raise ItemKNNArtifactError(
                "Item-KNN source interactions contain an invalid row"
            )
        seen_review_ids.add(review_id)
        if user_id in excluded:
            excluded_interactions += 1
            continue
        by_user[user_id].append(
            {
                "user_id": user_id,
                "review_id": review_id,
                "business_id": business_id,
                "stars": float(stars),
                "date": date,
            }
        )

    retained: list[dict[str, object]] = []
    reserved_interactions = 0
    reserve = config.reserved_tail_interactions
    for user_id in sorted(by_user):
        history = sorted(
            by_user[user_id],
            key=lambda row: (row["date"], row["review_id"]),
        )
        reserve_count = min(reserve, len(history))
        reserved_interactions += reserve_count
        retained.extend(history if reserve_count == 0 else history[:-reserve_count])
    retained.sort(key=lambda row: (row["date"], row["review_id"]))
    positive = [row for row in retained if float(row["stars"]) >= 4.0]
    negative = [row for row in retained if float(row["stars"]) <= 2.0]
    neutral = [row for row in retained if float(row["stars"]) == 3.0]

    _write_parquet(paths["positive_events"], positive)
    _write_parquet(paths["negative_events"], negative)
    _write_parquet(paths["neutral_events"], neutral)
    output_sha256 = {name: _sha256(path) for name, path in sorted(paths.items())}
    manifest = ItemKNNArtifactManifest(
        source_sha256=source_sha256,
        excluded_users_sha256=users_sha256,
        configuration=config,
        source_interactions=len(source_rows),
        retained_interactions=len(retained),
        reserved_interactions=reserved_interactions,
        excluded_user_interactions=excluded_interactions,
        positive_interactions=len(positive),
        negative_interactions=len(negative),
        neutral_interactions=len(neutral),
        output_sha256=output_sha256,
    )
    write_json_artifact(manifest_path, manifest)
    return _result("written", manifest_path, paths, manifest)
