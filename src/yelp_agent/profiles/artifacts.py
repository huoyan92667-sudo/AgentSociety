"""Atomically materialize task-time user profiles without ground truth access."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.config import UserProfileConfig
from yelp_agent.data.rolling_training import ROLLING_CONTEXT_SCHEMA
from yelp_agent.data.temporal import CONTEXT_SCHEMA
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel
from yelp_agent.profiles.builder import UserProfileBuilder
from yelp_agent.profiles.schema import (
    PREFERENCE_SIGNAL_SCHEMA,
    PROFILE_SNAPSHOT_SCHEMA,
    TASK_PROFILE_LINK_SCHEMA,
    TaskProfileLink,
    UserProfileV1,
)


class UserProfileArtifactError(RuntimeError):
    """Raised when frozen profile inputs or outputs violate invariants."""


class UserProfileManifest(StrictModel):
    format_version: Literal[1] = 1
    artifact_name: Literal["Task-Time User Profiles V1"] = "Task-Time User Profiles V1"
    profile_version: Literal["1.0.0"]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: dict[str, str]
    profile_snapshots: int = Field(ge=1)
    task_links: int = Field(ge=1)
    preference_signals: int = Field(ge=0)
    split_task_counts: dict[str, int]
    preference_kind_counts: dict[str, int]
    users: int = Field(ge=1)
    profiles_without_price: int = Field(ge=0)
    profiles_without_location: int = Field(ge=0)
    mean_reliability: float = Field(ge=0, le=1)
    history_count_mismatches: Literal[0]


class UserProfileBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    profiles_path: str
    preferences_path: str
    task_map_path: str
    manifest_path: str
    profile_snapshots: int = Field(ge=1)
    task_links: int = Field(ge=1)
    preference_signals: int = Field(ge=0)
    split_task_counts: dict[str, int]
    history_count_mismatches: Literal[0]


@dataclass(frozen=True, slots=True)
class _Context:
    task_id: str
    split: Literal["train", "validation", "test"]
    user_id: str
    cutoff_time: datetime
    history_count: int
    fold: int | None
    sample_weight: float | None


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


def _validate_schema(path: Path, expected: pa.Schema) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"User-profile context does not exist: {path}")
    try:
        actual = pq.ParquetFile(path).schema_arrow
    except (OSError, pa.ArrowException) as exc:
        raise UserProfileArtifactError(f"Could not read context: {path}") from exc
    if not actual.equals(expected, check_metadata=False):
        raise UserProfileArtifactError(f"Context has an unexpected schema: {path}")


def _load_contexts(
    rolling_contexts_path: Path,
    evaluation_contexts_path: Path,
) -> tuple[_Context, ...]:
    _validate_schema(rolling_contexts_path, ROLLING_CONTEXT_SCHEMA)
    _validate_schema(evaluation_contexts_path, CONTEXT_SCHEMA)
    contexts: list[_Context] = []
    for row in pq.read_table(rolling_contexts_path).to_pylist():
        if row["split"] != "train":
            raise UserProfileArtifactError("rolling contexts must be train only")
        contexts.append(
            _Context(
                task_id=str(row["task_id"]),
                split="train",
                user_id=str(row["user_id"]),
                cutoff_time=row["cutoff_time"],
                history_count=int(row["history_count"]),
                fold=int(row["fold"]),
                sample_weight=float(row["sample_weight"]),
            )
        )
    for row in pq.read_table(evaluation_contexts_path).to_pylist():
        split = str(row["split"])
        if split not in {"validation", "test"}:
            raise UserProfileArtifactError(
                "evaluation contexts must be validation or test"
            )
        contexts.append(
            _Context(
                task_id=str(row["task_id"]),
                split=split,  # type: ignore[arg-type]
                user_id=str(row["user_id"]),
                cutoff_time=row["cutoff_time"],
                history_count=int(row["history_count"]),
                fold=None,
                sample_weight=None,
            )
        )
    if not contexts:
        raise UserProfileArtifactError("No user-profile contexts were loaded")
    task_ids = [context.task_id for context in contexts]
    if len(set(task_ids)) != len(task_ids):
        raise UserProfileArtifactError("Task IDs must be unique across contexts")
    split_order = {"train": 0, "validation": 1, "test": 2}
    return tuple(
        sorted(
            contexts,
            key=lambda row: (
                split_order[row.split],
                row.user_id,
                row.cutoff_time,
                row.task_id,
            ),
        )
    )


def _profile_row(profile: UserProfileV1) -> dict[str, object]:
    summary = profile.evidence_summary
    location = profile.location_center
    return {
        "profile_id": profile.profile_id,
        "user_id": profile.user_id,
        "cutoff_time": profile.cutoff_time,
        "history_length": profile.history_length,
        "average_rating": profile.average_rating,
        **{
            f"rating_{stars}_count": profile.rating_distribution[str(stars)]
            for stars in range(1, 6)
        },
        "location_latitude": None if location is None else location.latitude,
        "location_longitude": None if location is None else location.longitude,
        "reliability": profile.reliability,
        "category_evidence_count": summary.category_evidence_count,
        "aspect_evidence_count": summary.aspect_evidence_count,
        "price_evidence_count": summary.price_evidence_count,
        "area_evidence_count": summary.area_evidence_count,
        "first_interaction": summary.first_interaction,
        "last_interaction": summary.last_interaction,
        "profile_version": profile.profile_version,
    }


def _preference_rows(profile: UserProfileV1) -> list[dict[str, object]]:
    return [
        {
            "profile_id": profile.profile_id,
            "user_id": profile.user_id,
            "cutoff_time": profile.cutoff_time,
            **signal.model_dump(mode="python"),
        }
        for signal in profile.preference_signals()
    ]


def _link_row(link: TaskProfileLink) -> dict[str, object]:
    return link.model_dump(mode="python")


def _flush(
    writer: pq.ParquetWriter,
    rows: list[dict[str, object]],
    schema: pa.Schema,
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def _result(
    status: Literal["written", "skipped"],
    root: Path,
    manifest: UserProfileManifest,
) -> UserProfileBuildResult:
    return UserProfileBuildResult(
        status=status,
        profiles_path=str(root / "profile_snapshots.parquet"),
        preferences_path=str(root / "preference_signals.parquet"),
        task_map_path=str(root / "task_profile_map.parquet"),
        manifest_path=str(root / "manifest.json"),
        profile_snapshots=manifest.profile_snapshots,
        task_links=manifest.task_links,
        preference_signals=manifest.preference_signals,
        split_task_counts=manifest.split_task_counts,
        history_count_mismatches=manifest.history_count_mismatches,
    )


def _reusable_manifest(
    root: Path,
    *,
    source_sha256: dict[str, str],
    configuration_sha256: str,
) -> UserProfileManifest | None:
    manifest_path = root / "manifest.json"
    output_paths = {
        "profiles": root / "profile_snapshots.parquet",
        "preferences": root / "preference_signals.parquet",
        "task_map": root / "task_profile_map.parquet",
    }
    if not manifest_path.is_file() or not all(
        path.is_file() for path in output_paths.values()
    ):
        return None
    try:
        manifest = UserProfileManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if (
        manifest.source_sha256 != source_sha256
        or manifest.configuration_sha256 != configuration_sha256
        or manifest.output_sha256
        != {name: _sha256_file(path) for name, path in output_paths.items()}
    ):
        return None
    for path, schema in (
        (output_paths["profiles"], PROFILE_SNAPSHOT_SCHEMA),
        (output_paths["preferences"], PREFERENCE_SIGNAL_SCHEMA),
        (output_paths["task_map"], TASK_PROFILE_LINK_SCHEMA),
    ):
        try:
            actual = pq.ParquetFile(path).schema_arrow
        except (OSError, pa.ArrowException):
            return None
        if not actual.equals(schema, check_metadata=False):
            return None
    return manifest


def build_user_profile_artifacts(
    *,
    businesses_path: str | Path,
    interactions_path: str | Path,
    aspect_records_path: str | Path,
    rolling_contexts_path: str | Path,
    evaluation_contexts_path: str | Path,
    output_root: str | Path,
    config: UserProfileConfig,
    broad_categories: set[str],
) -> UserProfileBuildResult:
    """Build all train/validation/test snapshots without reading ground truth."""

    source_paths = {
        "businesses": Path(businesses_path),
        "interactions": Path(interactions_path),
        "review_aspects": Path(aspect_records_path),
        "rolling_contexts": Path(rolling_contexts_path),
        "evaluation_contexts": Path(evaluation_contexts_path),
    }
    source_sha256 = {name: _sha256_file(path) for name, path in source_paths.items()}
    configuration_sha256 = _sha256_json(
        {
            "config": config.model_dump(mode="json"),
            "broad_categories": sorted(broad_categories),
        }
    )
    root = Path(output_root)
    reusable = _reusable_manifest(
        root,
        source_sha256=source_sha256,
        configuration_sha256=configuration_sha256,
    )
    if reusable is not None:
        return _result("skipped", root, reusable)

    contexts = _load_contexts(
        source_paths["rolling_contexts"],
        source_paths["evaluation_contexts"],
    )
    builder = UserProfileBuilder.from_parquet(
        businesses_path=source_paths["businesses"],
        interactions_path=source_paths["interactions"],
        aspect_records_path=source_paths["review_aspects"],
        config=config,
        broad_categories=broad_categories,
    )
    root.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "profiles": root / "profile_snapshots.parquet",
        "preferences": root / "preference_signals.parquet",
        "task_map": root / "task_profile_map.parquet",
    }
    partial_paths = {
        key: path.with_name(path.name + ".partial") for key, path in final_paths.items()
    }
    for path in partial_paths.values():
        path.unlink(missing_ok=True)

    writers = {
        "profiles": pq.ParquetWriter(
            partial_paths["profiles"], PROFILE_SNAPSHOT_SCHEMA, compression="zstd"
        ),
        "preferences": pq.ParquetWriter(
            partial_paths["preferences"],
            PREFERENCE_SIGNAL_SCHEMA,
            compression="zstd",
            use_dictionary=["kind", "value", "source"],
        ),
        "task_map": pq.ParquetWriter(
            partial_paths["task_map"],
            TASK_PROFILE_LINK_SCHEMA,
            compression="zstd",
            use_dictionary=["split", "user_id", "profile_id"],
        ),
    }
    buffers: dict[str, list[dict[str, object]]] = {
        "profiles": [],
        "preferences": [],
        "task_map": [],
    }
    seen_profiles: set[str] = set()
    users: set[str] = set()
    split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    reliabilities: list[float] = []
    profiles_without_price = 0
    profiles_without_location = 0
    preference_count = 0
    try:
        for context in contexts:
            profile = builder.build(context.user_id, context.cutoff_time)
            if profile.history_length != context.history_count:
                raise UserProfileArtifactError(
                    f"History count mismatch for task {context.task_id!r}: "
                    f"expected {context.history_count}, got {profile.history_length}"
                )
            link = TaskProfileLink(
                task_id=context.task_id,
                split=context.split,
                user_id=context.user_id,
                cutoff_time=context.cutoff_time,
                profile_id=profile.profile_id,
                expected_history_count=context.history_count,
                fold=context.fold,
                sample_weight=context.sample_weight,
            )
            buffers["task_map"].append(_link_row(link))
            split_counts[context.split] += 1
            if profile.profile_id not in seen_profiles:
                seen_profiles.add(profile.profile_id)
                users.add(profile.user_id)
                buffers["profiles"].append(_profile_row(profile))
                rows = _preference_rows(profile)
                buffers["preferences"].extend(rows)
                preference_count += len(rows)
                kind_counts.update(row["kind"] for row in rows)
                reliabilities.append(profile.reliability)
                profiles_without_price += profile.price_preference is None
                profiles_without_location += profile.location_center is None
            if len(buffers["task_map"]) >= config.artifact_batch_size:
                _flush(
                    writers["profiles"], buffers["profiles"], PROFILE_SNAPSHOT_SCHEMA
                )
                _flush(
                    writers["preferences"],
                    buffers["preferences"],
                    PREFERENCE_SIGNAL_SCHEMA,
                )
                _flush(
                    writers["task_map"], buffers["task_map"], TASK_PROFILE_LINK_SCHEMA
                )
        _flush(writers["profiles"], buffers["profiles"], PROFILE_SNAPSHOT_SCHEMA)
        _flush(
            writers["preferences"],
            buffers["preferences"],
            PREFERENCE_SIGNAL_SCHEMA,
        )
        _flush(writers["task_map"], buffers["task_map"], TASK_PROFILE_LINK_SCHEMA)
        for writer in writers.values():
            writer.close()
        for key, path in final_paths.items():
            os.replace(partial_paths[key], path)
    except Exception:
        for writer in writers.values():
            with suppress(OSError, pa.ArrowException):
                writer.close()
        for path in partial_paths.values():
            path.unlink(missing_ok=True)
        raise

    manifest = UserProfileManifest(
        profile_version=config.profile_version,
        source_sha256=source_sha256,
        configuration_sha256=configuration_sha256,
        output_sha256={name: _sha256_file(path) for name, path in final_paths.items()},
        profile_snapshots=len(seen_profiles),
        task_links=len(contexts),
        preference_signals=preference_count,
        split_task_counts=dict(sorted(split_counts.items())),
        preference_kind_counts=dict(sorted(kind_counts.items())),
        users=len(users),
        profiles_without_price=profiles_without_price,
        profiles_without_location=profiles_without_location,
        mean_reliability=sum(reliabilities) / len(reliabilities),
        history_count_mismatches=0,
    )
    write_json_artifact(root / "manifest.json", manifest)
    return _result("written", root, manifest)
