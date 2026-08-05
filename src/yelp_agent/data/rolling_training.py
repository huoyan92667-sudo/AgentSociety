"""Build deterministic, leak-resistant rolling temporal train examples."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.config import (
    CrossValidationConfig,
    EvaluationDataUsageConfig,
    RollingTrainingConfig,
)
from yelp_agent.data.temporal import HISTORY_SCHEMA
from yelp_agent.evaluation.data_usage import assign_user_fold
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel


RESERVED_TAIL_INTERACTIONS = 2

ROLLING_CONTEXT_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("cutoff_time", pa.timestamp("us"), nullable=False),
        pa.field("history_count", pa.int64(), nullable=False),
        pa.field("history_max_time", pa.timestamp("us"), nullable=False),
        pa.field("target_position", pa.int64(), nullable=False),
        pa.field("user_interaction_count", pa.int64(), nullable=False),
        pa.field("fold", pa.int32(), nullable=False),
        pa.field("sample_weight", pa.float64(), nullable=False),
    ]
)

ROLLING_GROUND_TRUTH_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("target_review_id", pa.string(), nullable=False),
        pa.field("target_business_id", pa.string(), nullable=False),
    ]
)


class RollingTrainingError(RuntimeError):
    """Raised when rolling examples cannot be built or safely reused."""


class MinimalTrainTaskReference(StrictModel):
    task_id: str = Field(min_length=1)


class FoldTrainingSize(StrictModel):
    user_count: int = Field(ge=0)
    task_count: int = Field(ge=0)


class HistoryCountSummary(StrictModel):
    minimum: int = Field(ge=1)
    median: float = Field(ge=1)
    p95: int = Field(ge=1)
    maximum: int = Field(ge=1)


class RollingTrainingManifest(StrictModel):
    format_version: Literal[1] = 1
    purpose: Literal["rolling_temporal_training"] = "rolling_temporal_training"
    reserved_splits: list[Literal["validation", "test"]]
    reserved_tail_interactions: Literal[2]
    rolling_configuration: RollingTrainingConfig
    cross_validation: CrossValidationConfig
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: dict[str, str]
    output_sha256: dict[str, str]
    source_users: int = Field(ge=1)
    users_with_tasks: int = Field(ge=1)
    users_without_tasks: int = Field(ge=0)
    train_tasks: int = Field(ge=1)
    history_rows: int = Field(ge=1)
    minimal_train_tasks: int = Field(ge=1)
    skipped_cutoff_ties: int = Field(ge=0)
    tasks_per_user: dict[str, int]
    folds: dict[str, FoldTrainingSize]
    history_count: HistoryCountSummary
    minimum_observed_weight: float = Field(gt=0, le=1)
    maximum_observed_weight: float = Field(gt=0, le=1)
    target_history_leaks: Literal[0] = 0
    history_cutoff_violations: Literal[0] = 0
    reserved_review_leaks: Literal[0] = 0
    fold_assignment_violations: Literal[0] = 0
    legacy_candidate_files_read: Literal[False] = False


class RollingTrainingBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    interactions_path: str
    frozen_contexts_path: str
    contexts_path: str
    histories_path: str
    ground_truth_path: str
    minimal_train_tasks_path: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_users: int = Field(ge=1)
    users_with_tasks: int = Field(ge=1)
    users_without_tasks: int = Field(ge=0)
    train_tasks: int = Field(ge=1)
    history_rows: int = Field(ge=1)
    minimal_train_tasks: int = Field(ge=1)
    skipped_cutoff_ties: int = Field(ge=0)
    tasks_per_user: dict[str, int]
    folds: dict[str, FoldTrainingSize]
    history_count: HistoryCountSummary
    output_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Interaction:
    user_id: str
    review_id: str
    business_id: str
    date: datetime


@dataclass(frozen=True, slots=True)
class _FrozenContext:
    split: str
    cutoff_time: datetime
    history_count: int


@dataclass(frozen=True, slots=True)
class _OutputPaths:
    contexts: Path
    histories: Path
    ground_truth: Path
    minimal_train_tasks: Path
    manifest: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.contexts,
            self.histories,
            self.ground_truth,
            self.minimal_train_tasks,
            self.manifest,
        )


def _output_paths(output_root: Path) -> _OutputPaths:
    training = output_root / "training"
    return _OutputPaths(
        contexts=training / "rolling_train_contexts.parquet",
        histories=training / "rolling_train_histories.parquet",
        ground_truth=(
            output_root
            / "ground_truth"
            / "rolling_train_ground_truth.parquet"
        ),
        minimal_train_tasks=training / "minimal_train_task_ids.jsonl",
        manifest=training / "rolling_train_manifest.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(
    config: RollingTrainingConfig,
    cross_validation: CrossValidationConfig,
) -> str:
    payload = {
        "cross_validation": cross_validation.model_dump(mode="json"),
        "reserved_tail_interactions": RESERVED_TAIL_INTERACTIONS,
        "rolling_training": config.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_interactions(path: Path) -> list[_Interaction]:
    if not path.is_file():
        raise FileNotFoundError(f"Interaction Parquet does not exist: {path}")
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT user_id, review_id, business_id, date
                FROM read_parquet(?)
                ORDER BY user_id, date, review_id
                """,
                [str(path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise RollingTrainingError(
            f"Could not read rolling training interactions from {path}: {exc}"
        ) from exc

    interactions: list[_Interaction] = []
    seen_review_ids: set[str] = set()
    for user_id, review_id, business_id, date in rows:
        user_key = str(user_id or "")
        review_key = str(review_id or "")
        business_key = str(business_id or "")
        if (
            not user_key
            or not review_key
            or review_key in seen_review_ids
            or not business_key
            or not isinstance(date, datetime)
        ):
            raise RollingTrainingError(
                "Interaction Parquet contains an invalid or duplicate row"
            )
        seen_review_ids.add(review_key)
        interactions.append(
            _Interaction(
                user_id=user_key,
                review_id=review_key,
                business_id=business_key,
                date=date,
            )
        )
    if not interactions:
        raise RollingTrainingError("Interaction Parquet contains no rows")
    return interactions


def _load_frozen_contexts(
    path: Path,
) -> dict[str, dict[str, _FrozenContext]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen temporal contexts do not exist: {path}"
        )
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT split, user_id, cutoff_time, history_count
                FROM read_parquet(?)
                ORDER BY user_id, split
                """,
                [str(path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise RollingTrainingError(
            f"Could not read frozen temporal contexts from {path}: {exc}"
        ) from exc

    contexts: dict[str, dict[str, _FrozenContext]] = defaultdict(dict)
    for split, user_id, cutoff_time, history_count in rows:
        split_key = str(split or "")
        user_key = str(user_id or "")
        if (
            split_key not in {"validation", "test"}
            or not user_key
            or split_key in contexts[user_key]
            or not isinstance(cutoff_time, datetime)
            or int(history_count) < 1
        ):
            raise RollingTrainingError(
                "Frozen temporal contexts contain an invalid row"
            )
        contexts[user_key][split_key] = _FrozenContext(
            split=split_key,
            cutoff_time=cutoff_time,
            history_count=int(history_count),
        )
    if not contexts or any(
        set(user_contexts) != {"validation", "test"}
        for user_contexts in contexts.values()
    ):
        raise RollingTrainingError(
            "Frozen contexts must contain one validation and one test row "
            "for every user"
        )
    return dict(contexts)


def _select_target_positions(
    interactions: list[_Interaction],
    config: RollingTrainingConfig,
) -> tuple[list[int], int]:
    latest_train_position = len(interactions) - RESERVED_TAIL_INTERACTIONS - 1
    possible = range(
        config.minimum_history_count,
        latest_train_position + 1,
    )
    safe: list[int] = []
    cutoff_ties = 0
    for position in possible:
        if interactions[position - 1].date < interactions[position].date:
            safe.append(position)
        else:
            cutoff_ties += 1
    if not safe:
        return [], cutoff_ties

    spaced: list[int] = []
    for position in safe:
        if (
            not spaced
            or position - spaced[-1] >= config.minimum_target_gap
        ):
            spaced.append(position)
    if spaced[-1] != safe[-1]:
        spaced[-1] = safe[-1]

    limit = config.maximum_tasks_per_user
    if len(spaced) <= limit:
        return spaced, cutoff_ties
    if limit == 1:
        return [spaced[-1]], cutoff_ties
    last_index = len(spaced) - 1
    selected_indices = [
        round(index * last_index / (limit - 1))
        for index in range(limit)
    ]
    selected = [spaced[index] for index in selected_indices]
    if len(set(selected)) != len(selected):
        raise RollingTrainingError(
            "Rolling target selection produced duplicate positions"
        )
    return selected, cutoff_ties


def _sample_weight(
    target_position: int,
    selected_positions: list[int],
    config: RollingTrainingConfig,
) -> float:
    oldest = selected_positions[0]
    newest = selected_positions[-1]
    if oldest == newest:
        return config.maximum_sample_weight
    progress = (target_position - oldest) / (newest - oldest)
    weight = config.minimum_sample_weight + progress * (
        config.maximum_sample_weight - config.minimum_sample_weight
    )
    return round(weight, 12)


def _history_summary(history_counts: list[int]) -> HistoryCountSummary:
    ordered = sorted(history_counts)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return HistoryCountSummary(
        minimum=ordered[0],
        median=float(statistics.median(ordered)),
        p95=ordered[p95_index],
        maximum=ordered[-1],
    )


def _build_rows(
    interactions: list[_Interaction],
    frozen_contexts: dict[str, dict[str, _FrozenContext]],
    config: RollingTrainingConfig,
    policy: EvaluationDataUsageConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[MinimalTrainTaskReference],
    dict[str, object],
]:
    by_user: defaultdict[str, list[_Interaction]] = defaultdict(list)
    for interaction in interactions:
        by_user[interaction.user_id].append(interaction)
    if set(by_user) != set(frozen_contexts):
        raise RollingTrainingError(
            "Interaction users do not match the frozen validation/test users"
        )

    contexts: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    ground_truth: list[dict[str, object]] = []
    minimal: list[MinimalTrainTaskReference] = []
    tasks_per_user: Counter[int] = Counter()
    fold_users: defaultdict[int, set[str]] = defaultdict(set)
    fold_tasks: Counter[int] = Counter()
    history_counts: list[int] = []
    weights: list[float] = []
    skipped_cutoff_ties = 0
    seen_task_ids: set[str] = set()

    for user_id in sorted(by_user):
        records = by_user[user_id]
        if len(records) < RESERVED_TAIL_INTERACTIONS + 1:
            raise RollingTrainingError(
                f"User {user_id!r} has too few interactions"
            )
        pair = frozen_contexts[user_id]
        validation = pair["validation"]
        test = pair["test"]
        if (
            validation.cutoff_time != records[-2].date
            or validation.history_count != len(records) - 2
            or test.cutoff_time != records[-1].date
            or test.history_count != len(records) - 1
        ):
            raise RollingTrainingError(
                f"Frozen validation/test contexts disagree for {user_id!r}"
            )

        selected, user_cutoff_ties = _select_target_positions(records, config)
        skipped_cutoff_ties += user_cutoff_ties
        tasks_per_user[len(selected)] += 1
        if not selected:
            continue
        fold = assign_user_fold(user_id, policy)
        fold_users[fold].add(user_id)
        minimal_task_id: str | None = None
        previous_target_position: int | None = None
        reserved_review_ids = {records[-2].review_id, records[-1].review_id}

        for target_position in selected:
            if (
                previous_target_position is not None
                and target_position - previous_target_position
                < config.minimum_target_gap
            ):
                raise RollingTrainingError(
                    f"Rolling targets are too close for {user_id!r}"
                )
            previous_target_position = target_position
            target = records[target_position]
            history = records[:target_position]
            if not history or history[-1].date >= target.date:
                raise RollingTrainingError(
                    f"Rolling history crosses cutoff for {user_id!r}"
                )
            history_review_ids = [item.review_id for item in history]
            if (
                target.review_id in history_review_ids
                or reserved_review_ids.intersection(history_review_ids)
                or target.review_id in reserved_review_ids
            ):
                raise RollingTrainingError(
                    f"Rolling target or reserved review leaked for {user_id!r}"
                )

            one_based_target_position = target_position + 1
            task_id = (
                f"train:{user_id}:{one_based_target_position:06d}"
            )
            if task_id in seen_task_ids:
                raise RollingTrainingError(
                    f"Duplicate rolling task_id: {task_id!r}"
                )
            seen_task_ids.add(task_id)
            weight = _sample_weight(target_position, selected, config)
            contexts.append(
                {
                    "task_id": task_id,
                    "split": "train",
                    "user_id": user_id,
                    "cutoff_time": target.date,
                    "history_count": len(history),
                    "history_max_time": history[-1].date,
                    "target_position": one_based_target_position,
                    "user_interaction_count": len(records),
                    "fold": fold,
                    "sample_weight": weight,
                }
            )
            histories.extend(
                {
                    "task_id": task_id,
                    "position": position,
                    "review_id": item.review_id,
                }
                for position, item in enumerate(history, start=1)
            )
            ground_truth.append(
                {
                    "task_id": task_id,
                    "target_review_id": target.review_id,
                    "target_business_id": target.business_id,
                }
            )
            fold_tasks[fold] += 1
            history_counts.append(len(history))
            weights.append(weight)
            minimal_task_id = task_id

        if minimal_task_id is None:
            raise RollingTrainingError(
                f"Could not identify a minimal train task for {user_id!r}"
            )
        minimal.append(MinimalTrainTaskReference(task_id=minimal_task_id))

    if not contexts:
        raise RollingTrainingError(
            "No rolling train tasks satisfy the configured history rules"
        )
    fold_sizes = {
        str(fold): FoldTrainingSize(
            user_count=len(fold_users[fold]),
            task_count=fold_tasks[fold],
        )
        for fold in range(1, policy.cross_validation.folds + 1)
    }
    stats: dict[str, object] = {
        "source_users": len(by_user),
        "users_with_tasks": len(minimal),
        "users_without_tasks": len(by_user) - len(minimal),
        "train_tasks": len(contexts),
        "history_rows": len(histories),
        "minimal_train_tasks": len(minimal),
        "skipped_cutoff_ties": skipped_cutoff_ties,
        "tasks_per_user": {
            str(task_count): user_count
            for task_count, user_count in sorted(tasks_per_user.items())
        },
        "folds": fold_sizes,
        "history_count": _history_summary(history_counts),
        "minimum_observed_weight": min(weights),
        "maximum_observed_weight": max(weights),
    }
    return contexts, histories, ground_truth, minimal, stats


def _write_outputs(
    paths: _OutputPaths,
    contexts: list[dict[str, object]],
    histories: list[dict[str, object]],
    ground_truth: list[dict[str, object]],
    minimal: list[MinimalTrainTaskReference],
    manifest_values: dict[str, object],
) -> RollingTrainingManifest:
    for path in paths.all():
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations = {
        "contexts": paths.contexts,
        "histories": paths.histories,
        "ground_truth": paths.ground_truth,
        "minimal_train_tasks": paths.minimal_train_tasks,
        "manifest": paths.manifest,
    }
    partials = {
        name: path.with_name(path.name + ".partial")
        for name, path in destinations.items()
    }
    for partial in partials.values():
        partial.unlink(missing_ok=True)
    try:
        pq.write_table(
            pa.Table.from_pylist(contexts, schema=ROLLING_CONTEXT_SCHEMA),
            partials["contexts"],
            compression="zstd",
            use_dictionary=["split", "user_id"],
        )
        pq.write_table(
            pa.Table.from_pylist(histories, schema=HISTORY_SCHEMA),
            partials["histories"],
            compression="zstd",
            use_dictionary=["task_id"],
        )
        pq.write_table(
            pa.Table.from_pylist(
                ground_truth,
                schema=ROLLING_GROUND_TRUTH_SCHEMA,
            ),
            partials["ground_truth"],
            compression="zstd",
            use_dictionary=["task_id"],
        )
        partials["minimal_train_tasks"].write_text(
            "".join(item.model_dump_json() + "\n" for item in minimal),
            encoding="utf-8",
            newline="\n",
        )
        output_sha256 = {
            name: _sha256_file(partials[name])
            for name in (
                "contexts",
                "histories",
                "ground_truth",
                "minimal_train_tasks",
            )
        }
        manifest = RollingTrainingManifest(
            **manifest_values,
            output_sha256=output_sha256,
        )
        partials["manifest"].write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for name in (
            "contexts",
            "histories",
            "ground_truth",
            "minimal_train_tasks",
            "manifest",
        ):
            os.replace(partials[name], destinations[name])
        return manifest
    except Exception:
        for partial in partials.values():
            partial.unlink(missing_ok=True)
        raise


def _read_manifest(path: Path) -> RollingTrainingManifest:
    try:
        return RollingTrainingManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise RollingTrainingError(
            f"Rolling training manifest is invalid: {path}"
        ) from exc


def _validate_existing_outputs(
    paths: _OutputPaths,
    *,
    source_sha256: dict[str, str],
    config: RollingTrainingConfig,
    policy: EvaluationDataUsageConfig,
) -> RollingTrainingManifest:
    if not all(path.is_file() for path in paths.all()):
        raise RollingTrainingError(
            "Rolling training outputs are incomplete; use force=True to rebuild"
        )
    manifest = _read_manifest(paths.manifest)
    if (
        manifest.source_sha256 != source_sha256
        or manifest.rolling_configuration != config
        or manifest.cross_validation != policy.cross_validation
        or manifest.configuration_sha256
        != _configuration_sha256(config, policy.cross_validation)
    ):
        raise RollingTrainingError(
            "Rolling training inputs or configuration changed; "
            "use force=True to rebuild"
        )
    for name in (
        "contexts",
        "histories",
        "ground_truth",
        "minimal_train_tasks",
    ):
        if _sha256_file(getattr(paths, name)) != manifest.output_sha256.get(name):
            raise RollingTrainingError(
                f"Rolling training output fingerprint changed: {name}"
            )
    expected_schemas = {
        "contexts": ROLLING_CONTEXT_SCHEMA,
        "histories": HISTORY_SCHEMA,
        "ground_truth": ROLLING_GROUND_TRUTH_SCHEMA,
    }
    for name, expected in expected_schemas.items():
        try:
            actual = pq.ParquetFile(getattr(paths, name)).schema_arrow
        except (OSError, pa.ArrowException) as exc:
            raise RollingTrainingError(
                f"Rolling training output is unreadable: {name}"
            ) from exc
        if not actual.equals(expected, check_metadata=False):
            raise RollingTrainingError(
                f"Rolling training output schema changed: {name}"
            )

    try:
        with duckdb.connect() as connection:
            context_stats = connection.execute(
                """
                SELECT
                    count(*),
                    count(DISTINCT task_id),
                    count(DISTINCT user_id),
                    count(*) FILTER (
                        WHERE split != 'train'
                            OR history_count < ?
                            OR history_max_time >= cutoff_time
                            OR fold < 1 OR fold > ?
                            OR sample_weight < ? OR sample_weight > ?
                    )
                FROM read_parquet(?)
                """,
                [
                    config.minimum_history_count,
                    policy.cross_validation.folds,
                    config.minimum_sample_weight,
                    config.maximum_sample_weight,
                    str(paths.contexts),
                ],
            ).fetchone()
            history_mismatches = connection.execute(
                """
                WITH history_stats AS (
                    SELECT
                        task_id,
                        count(*) AS rows,
                        count(DISTINCT review_id) AS reviews,
                        min(position) AS first_position,
                        max(position) AS last_position
                    FROM read_parquet(?)
                    GROUP BY task_id
                )
                SELECT count(*)
                FROM read_parquet(?) AS context
                FULL OUTER JOIN history_stats USING (task_id)
                WHERE context.task_id IS NULL
                    OR history_stats.task_id IS NULL
                    OR rows != history_count
                    OR reviews != history_count
                    OR first_position != 1
                    OR last_position != history_count
                """,
                [str(paths.histories), str(paths.contexts)],
            ).fetchone()[0]
            truth_mismatches = connection.execute(
                """
                WITH truth AS (
                    SELECT task_id, count(*) AS rows
                    FROM read_parquet(?)
                    GROUP BY task_id
                )
                SELECT count(*)
                FROM read_parquet(?) AS context
                FULL OUTER JOIN truth USING (task_id)
                WHERE context.task_id IS NULL
                    OR truth.task_id IS NULL
                    OR rows != 1
                """,
                [str(paths.ground_truth), str(paths.contexts)],
            ).fetchone()[0]
            spacing_violations = connection.execute(
                """
                WITH ordered AS (
                    SELECT
                        user_id,
                        target_position,
                        lag(target_position) OVER (
                            PARTITION BY user_id ORDER BY target_position
                        ) AS previous_position,
                        count(*) OVER (PARTITION BY user_id) AS user_tasks
                    FROM read_parquet(?)
                )
                SELECT count(*)
                FROM ordered
                WHERE user_tasks > ?
                    OR (
                        previous_position IS NOT NULL
                        AND target_position - previous_position < ?
                    )
                """,
                [
                    str(paths.contexts),
                    config.maximum_tasks_per_user,
                    config.minimum_target_gap,
                ],
            ).fetchone()[0]
            newest_task_ids = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT task_id
                    FROM read_parquet(?)
                    QUALIFY row_number() OVER (
                        PARTITION BY user_id ORDER BY target_position DESC
                    ) = 1
                    """,
                    [str(paths.contexts)],
                ).fetchall()
            }
    except duckdb.Error as exc:
        raise RollingTrainingError(
            f"Could not audit existing rolling training outputs: {exc}"
        ) from exc

    minimal_ids: list[str] = []
    try:
        with paths.minimal_train_tasks.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    minimal_ids.append(
                        MinimalTrainTaskReference.model_validate_json(line).task_id
                    )
    except (OSError, ValidationError) as exc:
        raise RollingTrainingError(
            "Minimal rolling train task references are invalid"
        ) from exc
    context_rows, distinct_tasks, distinct_users, invalid_contexts = context_stats
    if (
        context_rows != manifest.train_tasks
        or distinct_tasks != manifest.train_tasks
        or distinct_users != manifest.users_with_tasks
        or invalid_contexts
        or history_mismatches
        or truth_mismatches
        or spacing_violations
        or set(minimal_ids) != newest_task_ids
        or len(minimal_ids) != len(set(minimal_ids))
        or len(minimal_ids) != manifest.minimal_train_tasks
    ):
        raise RollingTrainingError(
            "Existing rolling training outputs are structurally inconsistent; "
            "use force=True to rebuild"
        )
    return manifest


def _result(
    *,
    status: Literal["written", "skipped"],
    interactions_path: Path,
    frozen_contexts_path: Path,
    paths: _OutputPaths,
    manifest: RollingTrainingManifest,
) -> RollingTrainingBuildResult:
    return RollingTrainingBuildResult(
        status=status,
        interactions_path=str(interactions_path),
        frozen_contexts_path=str(frozen_contexts_path),
        contexts_path=str(paths.contexts),
        histories_path=str(paths.histories),
        ground_truth_path=str(paths.ground_truth),
        minimal_train_tasks_path=str(paths.minimal_train_tasks),
        manifest_path=str(paths.manifest),
        manifest_sha256=_sha256_file(paths.manifest),
        source_users=manifest.source_users,
        users_with_tasks=manifest.users_with_tasks,
        users_without_tasks=manifest.users_without_tasks,
        train_tasks=manifest.train_tasks,
        history_rows=manifest.history_rows,
        minimal_train_tasks=manifest.minimal_train_tasks,
        skipped_cutoff_ties=manifest.skipped_cutoff_ties,
        tasks_per_user=manifest.tasks_per_user,
        folds=manifest.folds,
        history_count=manifest.history_count,
        output_sha256=manifest.output_sha256,
    )


def build_rolling_training_tasks(
    interactions_path: str | Path,
    frozen_contexts_path: str | Path,
    output_root: str | Path,
    config: RollingTrainingConfig,
    policy: EvaluationDataUsageConfig,
    *,
    force: bool = False,
) -> RollingTrainingBuildResult:
    """Build rolling train contexts, histories and isolated labels once."""

    interactions_source = Path(interactions_path)
    frozen_contexts_source = Path(frozen_contexts_path)
    if not interactions_source.is_file():
        raise FileNotFoundError(
            f"Interaction Parquet does not exist: {interactions_source}"
        )
    if not frozen_contexts_source.is_file():
        raise FileNotFoundError(
            "Frozen temporal contexts do not exist: "
            f"{frozen_contexts_source}"
        )
    paths = _output_paths(Path(output_root))
    source_sha256 = {
        "frozen_temporal_contexts": _sha256_file(frozen_contexts_source),
        "interactions": _sha256_file(interactions_source),
    }
    if any(path.exists() for path in paths.all()) and not force:
        manifest = _validate_existing_outputs(
            paths,
            source_sha256=source_sha256,
            config=config,
            policy=policy,
        )
        return _result(
            status="skipped",
            interactions_path=interactions_source,
            frozen_contexts_path=frozen_contexts_source,
            paths=paths,
            manifest=manifest,
        )

    interactions = _load_interactions(interactions_source)
    frozen_contexts = _load_frozen_contexts(frozen_contexts_source)
    contexts, histories, ground_truth, minimal, stats = _build_rows(
        interactions,
        frozen_contexts,
        config,
        policy,
    )
    manifest = _write_outputs(
        paths,
        contexts,
        histories,
        ground_truth,
        minimal,
        {
            "reserved_splits": ["validation", "test"],
            "reserved_tail_interactions": RESERVED_TAIL_INTERACTIONS,
            "rolling_configuration": config,
            "cross_validation": policy.cross_validation,
            "configuration_sha256": _configuration_sha256(
                config,
                policy.cross_validation,
            ),
            "source_sha256": source_sha256,
            **stats,
        },
    )
    return _result(
        status="written",
        interactions_path=interactions_source,
        frozen_contexts_path=frozen_contexts_source,
        paths=paths,
        manifest=manifest,
    )


def write_rolling_training_report(
    result: RollingTrainingBuildResult,
    output_path: str | Path,
) -> None:
    """Write the human-facing build summary as a stable JSON artifact."""

    write_json_artifact(output_path, result)
