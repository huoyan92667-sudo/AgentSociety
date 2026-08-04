"""Build leak-resistant validation and test targets from user histories."""

from __future__ import annotations

import json
import os
from itertools import groupby
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.models import StrictModel


CONTEXT_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("cutoff_time", pa.timestamp("us"), nullable=False),
        pa.field("history_count", pa.int64(), nullable=False),
        pa.field("history_max_time", pa.timestamp("us"), nullable=False),
    ]
)

HISTORY_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("position", pa.int64(), nullable=False),
        pa.field("review_id", pa.string(), nullable=False),
    ]
)

GROUND_TRUTH_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("target_business_id", pa.string(), nullable=False),
    ]
)


class TemporalSplitError(RuntimeError):
    """Raised when temporal targets cannot be isolated without leakage."""


class TemporalSplitResult(StrictModel):
    status: Literal["written", "skipped"]
    interactions_path: str
    contexts_path: str
    histories_path: str
    ground_truth_path: str
    users: int = Field(ge=0)
    validation_tasks: int = Field(ge=0)
    test_tasks: int = Field(ge=0)
    history_rows: int = Field(ge=0)
    target_history_leaks: int = Field(ge=0)
    history_cutoff_violations: int = Field(ge=0)
    test_missing_validation: int = Field(ge=0)


def _output_paths(output_root: Path) -> tuple[Path, Path, Path]:
    tasks_directory = output_root / "tasks"
    ground_truth_directory = output_root / "ground_truth"
    return (
        tasks_directory / "temporal_contexts.parquet",
        tasks_directory / "temporal_histories.parquet",
        ground_truth_directory / "ground_truth.parquet",
    )


def _load_interactions(path: Path) -> list[tuple[str, str, str, object]]:
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
        raise TemporalSplitError(
            f"Could not read temporal interactions from {path}: {exc}"
        ) from exc
    return rows


def _validate_existing_outputs(
    paths: tuple[Path, Path, Path],
) -> tuple[int, int, int, int]:
    if not all(path.is_file() for path in paths):
        raise TemporalSplitError(
            "Temporal split outputs are incomplete; use force=True to rebuild all"
        )
    expected_schemas = (CONTEXT_SCHEMA, HISTORY_SCHEMA, GROUND_TRUTH_SCHEMA)
    parquet_files: list[pq.ParquetFile] = []
    for path, expected_schema in zip(paths, expected_schemas, strict=True):
        try:
            parquet_file = pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as exc:
            raise TemporalSplitError(
                f"Existing temporal output is unreadable: {path}"
            ) from exc
        if not parquet_file.schema_arrow.equals(
            expected_schema, check_metadata=False
        ):
            raise TemporalSplitError(
                f"Existing temporal output has an unexpected schema: {path}"
            )
        parquet_files.append(parquet_file)

    contexts_path, histories_path, ground_truth_path = paths
    try:
        with duckdb.connect() as connection:
            context_stats = connection.execute(
                """
                SELECT
                    count(*) AS rows,
                    count(DISTINCT task_id) AS task_ids,
                    count(DISTINCT user_id) AS users,
                    count(*) FILTER (WHERE split = 'validation')
                        AS validation_tasks,
                    count(*) FILTER (WHERE split = 'test') AS test_tasks,
                    count(*) FILTER (
                        WHERE history_max_time >= cutoff_time
                    ) AS cutoff_violations
                FROM read_parquet(?)
                """,
                [str(contexts_path)],
            ).fetchone()
            history_mismatches = connection.execute(
                """
                WITH history_stats AS (
                    SELECT
                        task_id,
                        count(*) AS row_count,
                        count(DISTINCT position) AS position_count,
                        count(DISTINCT review_id) AS review_count,
                        min(position) AS min_position,
                        max(position) AS max_position
                    FROM read_parquet(?)
                    GROUP BY task_id
                )
                SELECT count(*)
                FROM read_parquet(?) AS contexts
                FULL OUTER JOIN history_stats USING (task_id)
                WHERE
                    contexts.task_id IS NULL
                    OR history_stats.task_id IS NULL
                    OR row_count != history_count
                    OR position_count != history_count
                    OR review_count != history_count
                    OR min_position != 1
                    OR max_position != history_count
                """,
                [str(histories_path), str(contexts_path)],
            ).fetchone()[0]
            ground_truth_mismatches = connection.execute(
                """
                WITH truth AS (
                    SELECT task_id, count(*) AS truth_count
                    FROM read_parquet(?)
                    GROUP BY task_id
                )
                SELECT count(*)
                FROM read_parquet(?) AS contexts
                FULL OUTER JOIN truth USING (task_id)
                WHERE
                    contexts.task_id IS NULL
                    OR truth.task_id IS NULL
                    OR truth_count != 1
                """,
                [str(ground_truth_path), str(contexts_path)],
            ).fetchone()[0]
            pair_mismatches = connection.execute(
                """
                WITH validation AS (
                    SELECT user_id, history_count, cutoff_time
                    FROM read_parquet(?)
                    WHERE split = 'validation'
                ), test AS (
                    SELECT user_id, history_count, cutoff_time
                    FROM read_parquet(?)
                    WHERE split = 'test'
                )
                SELECT count(*)
                FROM validation
                FULL OUTER JOIN test USING (user_id)
                WHERE
                    validation.user_id IS NULL
                    OR test.user_id IS NULL
                    OR test.history_count != validation.history_count + 1
                    OR test.cutoff_time <= validation.cutoff_time
                """,
                [str(contexts_path), str(contexts_path)],
            ).fetchone()[0]
            prefix_mismatches = connection.execute(
                """
                WITH validation_context AS (
                    SELECT task_id, user_id, history_count
                    FROM read_parquet(?)
                    WHERE split = 'validation'
                ), test_context AS (
                    SELECT task_id, user_id
                    FROM read_parquet(?)
                    WHERE split = 'test'
                ), validation_history AS (
                    SELECT
                        context.user_id,
                        history.position,
                        history.review_id
                    FROM validation_context AS context
                    JOIN read_parquet(?) AS history USING (task_id)
                ), test_prefix AS (
                    SELECT
                        test_context.user_id,
                        history.position,
                        history.review_id
                    FROM test_context
                    JOIN validation_context USING (user_id)
                    JOIN read_parquet(?) AS history
                        ON history.task_id = test_context.task_id
                        AND history.position
                            <= validation_context.history_count
                )
                SELECT count(*)
                FROM validation_history
                FULL OUTER JOIN test_prefix USING (user_id, position)
                WHERE
                    validation_history.review_id IS NULL
                    OR test_prefix.review_id IS NULL
                    OR validation_history.review_id != test_prefix.review_id
                """,
                [
                    str(contexts_path),
                    str(contexts_path),
                    str(histories_path),
                    str(histories_path),
                ],
            ).fetchone()[0]
    except duckdb.Error as exc:
        raise TemporalSplitError(
            f"Could not validate existing temporal outputs: {exc}"
        ) from exc

    (
        context_rows,
        distinct_task_ids,
        users,
        validation_tasks,
        test_tasks,
        cutoff_violations,
    ) = context_stats
    structure_is_invalid = (
        context_rows != distinct_task_ids
        or validation_tasks != users
        or test_tasks != users
        or context_rows != validation_tasks + test_tasks
        or cutoff_violations
        or history_mismatches
        or ground_truth_mismatches
        or pair_mismatches
        or prefix_mismatches
    )
    if structure_is_invalid:
        raise TemporalSplitError(
            "Existing temporal outputs are inconsistent; "
            "use force=True to rebuild all"
        )
    return (
        users,
        validation_tasks,
        test_tasks,
        parquet_files[1].metadata.num_rows,
    )


def _task_rows(
    rows: list[tuple[str, str, str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    int,
]:
    contexts: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    ground_truth: list[dict[str, object]] = []
    seen_review_ids: set[str] = set()
    user_count = 0

    for user_id, grouped_rows in groupby(rows, key=lambda row: row[0]):
        interactions = list(grouped_rows)
        user_count += 1
        if len(interactions) < 3:
            raise TemporalSplitError(
                f"User {user_id!r} has fewer than 3 interactions"
            )
        for _, review_id, business_id, _ in interactions:
            if not user_id or not review_id or not business_id:
                raise TemporalSplitError(
                    f"User {user_id!r} has an interaction with a missing ID"
                )
            if review_id in seen_review_ids:
                raise TemporalSplitError(
                    f"Duplicate review_id in interactions: {review_id!r}"
                )
            seen_review_ids.add(review_id)

        targets = (
            ("validation", interactions[-2], interactions[:-2]),
            ("test", interactions[-1], interactions[:-1]),
        )
        validation_review_id = str(interactions[-2][1])

        for split, target, history in targets:
            _, target_review_id, target_business_id, cutoff_time = target
            history_max_time = history[-1][3]
            if history_max_time >= cutoff_time:
                raise TemporalSplitError(
                    f"User {user_id!r} has a {split} cutoff tie; "
                    "history must be strictly earlier than cutoff_time"
                )
            history_review_ids = [str(row[1]) for row in history]
            if str(target_review_id) in history_review_ids:
                raise TemporalSplitError(
                    f"Target review leaked into {split} history for {user_id!r}"
                )
            if split == "test" and validation_review_id not in history_review_ids:
                raise TemporalSplitError(
                    f"Test history omits validation behavior for {user_id!r}"
                )

            task_id = f"{split}:{user_id}"
            contexts.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "user_id": user_id,
                    "cutoff_time": cutoff_time,
                    "history_count": len(history),
                    "history_max_time": history_max_time,
                }
            )
            histories.extend(
                {
                    "task_id": task_id,
                    "position": position,
                    "review_id": review_id,
                }
                for position, review_id in enumerate(
                    history_review_ids,
                    start=1,
                )
            )
            ground_truth.append(
                {
                    "task_id": task_id,
                    "target_business_id": target_business_id,
                }
            )

    if not contexts:
        raise TemporalSplitError("Interaction Parquet contains no users")
    return contexts, histories, ground_truth, user_count


def _write_outputs(
    contexts: list[dict[str, object]],
    histories: list[dict[str, object]],
    ground_truth: list[dict[str, object]],
    paths: tuple[Path, Path, Path],
) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    partial_paths = tuple(
        path.with_name(path.name + ".partial") for path in paths
    )
    for partial in partial_paths:
        partial.unlink(missing_ok=True)

    try:
        pq.write_table(
            pa.Table.from_pylist(contexts, schema=CONTEXT_SCHEMA),
            partial_paths[0],
            compression="zstd",
            use_dictionary=["split", "user_id"],
        )
        pq.write_table(
            pa.Table.from_pylist(histories, schema=HISTORY_SCHEMA),
            partial_paths[1],
            compression="zstd",
            use_dictionary=["task_id"],
        )
        pq.write_table(
            pa.Table.from_pylist(ground_truth, schema=GROUND_TRUTH_SCHEMA),
            partial_paths[2],
            compression="zstd",
        )
        for partial, destination in zip(partial_paths, paths, strict=True):
            os.replace(partial, destination)
    except Exception:
        for partial in partial_paths:
            partial.unlink(missing_ok=True)
        raise


def build_temporal_splits(
    interactions_path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> TemporalSplitResult:
    """Create validation/test contexts, histories and isolated ground truth."""

    source = Path(interactions_path)
    root = Path(output_root)
    if not source.is_file():
        raise FileNotFoundError(f"Interaction Parquet does not exist: {source}")
    paths = _output_paths(root)
    existing = [path.exists() for path in paths]
    if any(existing) and not force:
        users, validation_tasks, test_tasks, history_rows = (
            _validate_existing_outputs(paths)
        )
        return TemporalSplitResult(
            status="skipped",
            interactions_path=str(source),
            contexts_path=str(paths[0]),
            histories_path=str(paths[1]),
            ground_truth_path=str(paths[2]),
            users=users,
            validation_tasks=validation_tasks,
            test_tasks=test_tasks,
            history_rows=history_rows,
            target_history_leaks=0,
            history_cutoff_violations=0,
            test_missing_validation=0,
        )

    rows = _load_interactions(source)
    contexts, histories, ground_truth, user_count = _task_rows(rows)
    _write_outputs(contexts, histories, ground_truth, paths)
    validation_tasks = sum(
        context["split"] == "validation" for context in contexts
    )
    test_tasks = sum(context["split"] == "test" for context in contexts)
    return TemporalSplitResult(
        status="written",
        interactions_path=str(source),
        contexts_path=str(paths[0]),
        histories_path=str(paths[1]),
        ground_truth_path=str(paths[2]),
        users=user_count,
        validation_tasks=validation_tasks,
        test_tasks=test_tasks,
        history_rows=len(histories),
        target_history_leaks=0,
        history_cutoff_violations=0,
        test_missing_validation=0,
    )


def write_temporal_split_report(
    result: TemporalSplitResult,
    output_path: str | Path,
) -> None:
    """Persist split counts and temporal leakage audit results."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
