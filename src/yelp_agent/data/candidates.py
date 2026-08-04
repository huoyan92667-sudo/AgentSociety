"""Build frozen twenty-business recommendation candidate tasks."""

from __future__ import annotations

import hashlib
import json
import os
import random
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError

from yelp_agent.config import DataConfig
from yelp_agent.data.temporal import GROUND_TRUTH_SCHEMA
from yelp_agent.models import RecommendationTask, StrictModel


PROVENANCE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("source_bucket", pa.string(), nullable=False),
        pa.field("final_position", pa.int64(), nullable=False),
    ]
)

DROPPED_TASK_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)


class CandidateBuildError(RuntimeError):
    """Raised when valid, leak-free candidate tasks cannot be constructed."""


class CandidateBuildResult(StrictModel):
    status: Literal["written", "skipped"]
    validation_tasks_path: str
    test_tasks_path: str
    ground_truth_path: str
    provenance_path: str
    dropped_tasks_path: str
    input_tasks: int = Field(ge=0)
    retained_tasks: int = Field(ge=0)
    validation_tasks: int = Field(ge=0)
    test_tasks: int = Field(ge=0)
    dropped_unavailable_targets: int = Field(ge=0)
    tasks_using_refill: int = Field(ge=0)
    bucket_counts: dict[str, int]


@dataclass(frozen=True)
class _Business:
    categories: frozenset[str]
    fine_categories: frozenset[str]
    groups: frozenset[str]


@dataclass(frozen=True)
class _TaskSeed:
    task_id: str
    split: str
    user_id: str
    cutoff_time: datetime
    target_business_id: str


class _ReviewIndex:
    def __init__(
        self,
        business_dates: dict[str, list[datetime]],
        business_prefix_stars: dict[str, list[float]],
        global_dates: list[datetime],
        global_prefix_stars: list[float],
    ) -> None:
        self._business_dates = business_dates
        self._business_prefix_stars = business_prefix_stars
        self._global_dates = global_dates
        self._global_prefix_stars = global_prefix_stars

    def available(self, business_id: str, cutoff_time: datetime) -> bool:
        dates = self._business_dates.get(business_id, [])
        return bisect_left(dates, cutoff_time) > 0

    def normalized_bayesian_rating(
        self,
        business_id: str,
        cutoff_time: datetime,
        *,
        prior_count: int = 20,
    ) -> float:
        dates = self._business_dates.get(business_id, [])
        count = bisect_left(dates, cutoff_time)
        if count == 0:
            return 0.0
        business_sum = self._business_prefix_stars[business_id][count]
        global_count = bisect_left(self._global_dates, cutoff_time)
        if global_count:
            global_mean = self._global_prefix_stars[global_count] / global_count
        else:
            global_mean = 3.5
        bayesian_rating = (
            business_sum + prior_count * global_mean
        ) / (count + prior_count)
        return max(0.0, min(1.0, (bayesian_rating - 1.0) / 4.0))


def _paths(
    processed_root: Path,
    task_root: Path,
) -> dict[str, Path]:
    return {
        "businesses": processed_root / "businesses.parquet",
        "reviews": processed_root / "reviews.parquet",
        "interactions": processed_root / "interactions.parquet",
        "contexts": task_root / "tasks" / "temporal_contexts.parquet",
        "histories": task_root / "tasks" / "temporal_histories.parquet",
        "temporal_truth": task_root / "ground_truth" / "ground_truth.parquet",
        "validation_tasks": task_root / "tasks" / "validation_tasks.jsonl",
        "test_tasks": task_root / "tasks" / "test_tasks.jsonl",
        "candidate_truth": (
            task_root / "ground_truth" / "candidate_ground_truth.parquet"
        ),
        "provenance": task_root / "audit" / "candidate_provenance.parquet",
        "dropped_tasks": task_root / "audit" / "dropped_tasks.parquet",
    }


def _require_inputs(paths: dict[str, Path]) -> None:
    for name in (
        "businesses",
        "reviews",
        "interactions",
        "contexts",
        "histories",
        "temporal_truth",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Candidate input {name!r} does not exist: {paths[name]}")


def _load_businesses(
    path: Path,
    config: DataConfig,
) -> tuple[
    dict[str, _Business],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    try:
        rows = pq.read_table(path, columns=["business_id", "categories"]).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise CandidateBuildError(f"Could not read businesses from {path}: {exc}") from exc

    broad = set(config.broad_categories)
    group_categories = {
        group: set(categories)
        for group, categories in config.category_groups.items()
    }
    businesses: dict[str, _Business] = {}
    fine_index: dict[str, set[str]] = defaultdict(set)
    group_index: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        business_id = row.get("business_id")
        raw_categories = row.get("categories")
        if (
            not isinstance(business_id, str)
            or not business_id
            or not isinstance(raw_categories, list)
        ):
            raise CandidateBuildError("Business candidate data contains an invalid row")
        if business_id in businesses:
            raise CandidateBuildError(f"Duplicate candidate business_id: {business_id!r}")
        categories = frozenset(
            str(category).strip()
            for category in raw_categories
            if str(category).strip()
        )
        fine_categories = frozenset(categories.difference(broad))
        groups = frozenset(
            group
            for group, category_set in group_categories.items()
            if categories.intersection(category_set)
        )
        businesses[business_id] = _Business(
            categories=categories,
            fine_categories=fine_categories,
            groups=groups,
        )
        for category in fine_categories:
            fine_index[category].add(business_id)
        for group in groups:
            group_index[group].add(business_id)
    if not businesses:
        raise CandidateBuildError("Business candidate universe is empty")
    return businesses, dict(fine_index), dict(group_index)


def _prefix_sums(values: list[float]) -> list[float]:
    result = [0.0]
    running = 0.0
    for value in values:
        running += value
        result.append(running)
    return result


def _load_review_index(path: Path) -> _ReviewIndex:
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT business_id, date, stars
                FROM read_parquet(?)
                ORDER BY business_id, date
                """,
                [str(path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise CandidateBuildError(f"Could not index point-in-time reviews: {exc}") from exc

    dated_stars: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    global_rows: list[tuple[datetime, float]] = []
    for business_id, date, stars in rows:
        if not business_id or not isinstance(date, datetime):
            raise CandidateBuildError("Review index contains an invalid business or date")
        rating = float(stars)
        if not 1.0 <= rating <= 5.0:
            raise CandidateBuildError("Review index contains a rating outside 1 to 5")
        dated_stars[str(business_id)].append((date, rating))
        global_rows.append((date, rating))

    business_dates: dict[str, list[datetime]] = {}
    business_prefix: dict[str, list[float]] = {}
    for business_id, values in dated_stars.items():
        business_dates[business_id] = [date for date, _ in values]
        business_prefix[business_id] = _prefix_sums(
            [stars for _, stars in values]
        )
    global_rows.sort(key=lambda row: row[0])
    return _ReviewIndex(
        business_dates,
        business_prefix,
        [date for date, _ in global_rows],
        _prefix_sums([stars for _, stars in global_rows]),
    )


def _load_tasks_and_histories(
    paths: dict[str, Path],
) -> tuple[list[_TaskSeed], dict[str, list[tuple[str, float]]]]:
    try:
        with duckdb.connect() as connection:
            task_rows = connection.execute(
                """
                SELECT
                    context.task_id,
                    context.split,
                    context.user_id,
                    context.cutoff_time,
                    truth.target_business_id
                FROM read_parquet(?) AS context
                JOIN read_parquet(?) AS truth USING (task_id)
                ORDER BY context.task_id
                """,
                [str(paths["contexts"]), str(paths["temporal_truth"])],
            ).fetchall()
            history_rows = connection.execute(
                """
                SELECT
                    history.task_id,
                    history.position,
                    interaction.business_id,
                    interaction.stars
                FROM read_parquet(?) AS history
                JOIN read_parquet(?) AS interaction USING (review_id)
                ORDER BY history.task_id, history.position
                """,
                [str(paths["histories"]), str(paths["interactions"])],
            ).fetchall()
    except duckdb.Error as exc:
        raise CandidateBuildError(f"Could not load temporal candidate inputs: {exc}") from exc

    tasks = [
        _TaskSeed(
            task_id=str(row[0]),
            split=str(row[1]),
            user_id=str(row[2]),
            cutoff_time=row[3],
            target_business_id=str(row[4]),
        )
        for row in task_rows
    ]
    histories: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for task_id, _, business_id, stars in history_rows:
        histories[str(task_id)].append((str(business_id), float(stars)))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise CandidateBuildError("Temporal candidate tasks contain duplicate task IDs")
    return tasks, dict(histories)


def _category_preferences(
    history: list[tuple[str, float]],
    businesses: dict[str, _Business],
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    rating_sums: defaultdict[str, float] = defaultdict(float)
    for business_id, stars in history:
        business = businesses.get(business_id)
        if business is None:
            raise CandidateBuildError(
                f"History references unknown business_id {business_id!r}"
            )
        for category in business.fine_categories:
            counts[category] += 1
            rating_sums[category] += stars
    if not counts:
        return {}
    max_count = max(counts.values())
    return {
        category: (
            0.7 * (((rating_sums[category] / count) - 1.0) / 4.0)
            + 0.3 * (count / max_count)
        )
        for category, count in counts.items()
    }


def _stable_order(
    business_ids: set[str] | list[str],
    *,
    seed: int,
    task_id: str,
    bucket: str,
) -> list[str]:
    return sorted(
        business_ids,
        key=lambda business_id: (
            hashlib.sha256(
                f"{seed}:{task_id}:{bucket}:{business_id}".encode("utf-8")
            ).digest(),
            business_id,
        ),
    )


def _preference_order(
    pool: set[str],
    preferences: dict[str, float],
    businesses: dict[str, _Business],
    review_index: _ReviewIndex,
    cutoff_time: datetime,
) -> list[str]:
    scored: list[tuple[float, float, str]] = []
    for business_id in pool:
        preference = max(
            (
                preferences.get(category, 0.0)
                for category in businesses[business_id].fine_categories
            ),
            default=0.0,
        )
        if preference <= 0:
            continue
        quality = review_index.normalized_bayesian_rating(
            business_id,
            cutoff_time,
        )
        combined = 0.6 * preference + 0.4 * quality
        scored.append((combined, quality, business_id))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [business_id for _, _, business_id in scored]


def _build_one_task(
    task: _TaskSeed,
    history: list[tuple[str, float]],
    businesses: dict[str, _Business],
    fine_index: dict[str, set[str]],
    group_index: dict[str, set[str]],
    review_index: _ReviewIndex,
    config: DataConfig,
) -> tuple[RecommendationTask | None, list[tuple[str, str]], str | None]:
    target = businesses.get(task.target_business_id)
    if target is None:
        raise CandidateBuildError(
            f"Task {task.task_id!r} targets unknown business "
            f"{task.target_business_id!r}"
        )
    if not review_index.available(task.target_business_id, task.cutoff_time):
        return None, [], "target_unavailable_before_cutoff"

    history_businesses = {business_id for business_id, _ in history}
    all_business_ids = set(businesses)

    def eligible(pool: set[str]) -> set[str]:
        return {
            business_id
            for business_id in pool
            if business_id != task.target_business_id
            and business_id not in history_businesses
            and review_index.available(business_id, task.cutoff_time)
        }

    same_pool: set[str] = set()
    for category in target.fine_categories:
        same_pool.update(fine_index.get(category, set()))
    same_pool = eligible(same_pool)

    related_pool: set[str] = set()
    for group in target.groups:
        related_pool.update(group_index.get(group, set()))
    related_pool = {
        business_id
        for business_id in eligible(related_pool)
        if not businesses[business_id].fine_categories.intersection(
            target.fine_categories
        )
    }

    preferences = _category_preferences(history, businesses)
    preference_pool: set[str] = set()
    for category in preferences:
        preference_pool.update(fine_index.get(category, set()))
    preference_pool = eligible(preference_pool)
    preference_order = _preference_order(
        preference_pool,
        preferences,
        businesses,
        review_index,
        task.cutoff_time,
    )
    global_pool = eligible(all_business_ids)

    selected: set[str] = set()
    negatives: list[tuple[str, str]] = []
    bucket_specs = (
        (
            "same_fine",
            config.hard_negative_same_category,
            _stable_order(
                same_pool,
                seed=config.random_seed,
                task_id=task.task_id,
                bucket="same_fine",
            ),
        ),
        (
            "related",
            config.hard_negative_related_category,
            _stable_order(
                related_pool,
                seed=config.random_seed,
                task_id=task.task_id,
                bucket="related",
            ),
        ),
        ("preference", config.preference_negative_count, preference_order),
        (
            "random",
            config.random_negative_count,
            _stable_order(
                global_pool,
                seed=config.random_seed,
                task_id=task.task_id,
                bucket="random",
            ),
        ),
    )
    for bucket, count, pool in bucket_specs:
        before = len(negatives)
        for business_id in pool:
            if business_id in selected:
                continue
            selected.add(business_id)
            negatives.append((business_id, bucket))
            if len(negatives) - before >= count:
                break

    required_negatives = config.candidate_count - 1
    if len(negatives) < required_negatives:
        refill_specs = (
            ("refill_same_fine", bucket_specs[0][2]),
            ("refill_related", bucket_specs[1][2]),
            ("refill_preference", preference_order),
            (
                "refill_random",
                _stable_order(
                    global_pool,
                    seed=config.random_seed,
                    task_id=task.task_id,
                    bucket="refill_random",
                ),
            ),
        )
        for bucket, pool in refill_specs:
            for business_id in pool:
                if business_id in selected:
                    continue
                selected.add(business_id)
                negatives.append((business_id, bucket))
                if len(negatives) == required_negatives:
                    break
            if len(negatives) == required_negatives:
                break
    if len(negatives) != required_negatives:
        raise CandidateBuildError(
            f"Task {task.task_id!r} has only {len(negatives)} valid negatives"
        )

    candidate_ids = [task.target_business_id] + [
        business_id for business_id, _ in negatives
    ]
    seed_bytes = hashlib.sha256(
        f"{config.random_seed}{task.task_id}".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(seed_bytes, "big")).shuffle(candidate_ids)
    recommendation_task = RecommendationTask(
        task_id=task.task_id,
        user_id=task.user_id,
        cutoff_time=task.cutoff_time,
        candidate_business_ids=candidate_ids,
    )
    return recommendation_task, negatives, None


def _write_jsonl(tasks: list[RecommendationTask], path: Path) -> None:
    payload = "".join(
        task.model_dump_json() + "\n"
        for task in sorted(tasks, key=lambda task: task.task_id)
    )
    path.write_text(payload, encoding="utf-8")


def _write_outputs(
    paths: dict[str, Path],
    validation_tasks: list[RecommendationTask],
    test_tasks: list[RecommendationTask],
    truth_rows: list[dict[str, str]],
    provenance_rows: list[dict[str, object]],
    dropped_rows: list[dict[str, str]],
) -> None:
    output_names = (
        "validation_tasks",
        "test_tasks",
        "candidate_truth",
        "provenance",
        "dropped_tasks",
    )
    outputs = [paths[name] for name in output_names]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    partials = [path.with_name(path.name + ".partial") for path in outputs]
    for partial in partials:
        partial.unlink(missing_ok=True)
    try:
        _write_jsonl(validation_tasks, partials[0])
        _write_jsonl(test_tasks, partials[1])
        pq.write_table(
            pa.Table.from_pylist(truth_rows, schema=GROUND_TRUTH_SCHEMA),
            partials[2],
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(provenance_rows, schema=PROVENANCE_SCHEMA),
            partials[3],
            compression="zstd",
            use_dictionary=["source_bucket"],
        )
        pq.write_table(
            pa.Table.from_pylist(dropped_rows, schema=DROPPED_TASK_SCHEMA),
            partials[4],
            compression="zstd",
        )
        for partial, destination in zip(partials, outputs, strict=True):
            os.replace(partial, destination)
    except Exception:
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise


def _read_frozen_tasks(path: Path) -> list[RecommendationTask]:
    tasks: list[RecommendationTask] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    tasks.append(RecommendationTask.model_validate_json(line))
                except ValidationError as exc:
                    raise CandidateBuildError(
                        f"Invalid frozen task at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise CandidateBuildError(f"Could not read frozen tasks from {path}") from exc
    return tasks


def _read_exact_parquet(
    path: Path,
    schema: pa.Schema,
) -> list[dict[str, object]]:
    try:
        parquet_file = pq.ParquetFile(path)
        if not parquet_file.schema_arrow.equals(schema, check_metadata=False):
            raise CandidateBuildError(
                f"Frozen candidate output has an unexpected schema: {path}"
            )
        return parquet_file.read().to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise CandidateBuildError(
            f"Could not read frozen candidate output: {path}"
        ) from exc


def _validate_existing_outputs(
    paths: dict[str, Path],
) -> CandidateBuildResult:
    output_names = (
        "validation_tasks",
        "test_tasks",
        "candidate_truth",
        "provenance",
        "dropped_tasks",
    )
    if not all(paths[name].is_file() for name in output_names):
        raise CandidateBuildError(
            "Candidate outputs are incomplete; use force=True to rebuild all"
        )

    validation_tasks = _read_frozen_tasks(paths["validation_tasks"])
    test_tasks = _read_frozen_tasks(paths["test_tasks"])
    all_tasks = validation_tasks + test_tasks
    task_by_id = {task.task_id: task for task in all_tasks}
    if len(task_by_id) != len(all_tasks):
        raise CandidateBuildError("Frozen candidate tasks contain duplicate task IDs")

    contexts = pq.read_table(paths["contexts"]).to_pylist()
    context_by_id = {
        str(row["task_id"]): row
        for row in contexts
    }
    if len(context_by_id) != len(contexts):
        raise CandidateBuildError("Temporal contexts contain duplicate task IDs")
    truth_rows = _read_exact_parquet(
        paths["candidate_truth"],
        GROUND_TRUTH_SCHEMA,
    )
    truth_by_id = {
        str(row["task_id"]): str(row["target_business_id"])
        for row in truth_rows
    }
    if len(truth_by_id) != len(truth_rows):
        raise CandidateBuildError("Candidate ground truth contains duplicate task IDs")
    provenance_rows = _read_exact_parquet(
        paths["provenance"],
        PROVENANCE_SCHEMA,
    )
    dropped_rows = _read_exact_parquet(
        paths["dropped_tasks"],
        DROPPED_TASK_SCHEMA,
    )
    dropped_by_id = {
        str(row["task_id"]): row
        for row in dropped_rows
    }
    if len(dropped_by_id) != len(dropped_rows):
        raise CandidateBuildError("Dropped candidate tasks contain duplicate task IDs")

    retained_ids = set(task_by_id)
    dropped_ids = set(dropped_by_id)
    context_ids = set(context_by_id)
    if (
        retained_ids.intersection(dropped_ids)
        or retained_ids.union(dropped_ids) != context_ids
        or set(truth_by_id) != retained_ids
    ):
        raise CandidateBuildError(
            "Frozen candidate tasks, dropped tasks and ground truth are inconsistent"
        )

    static_business_ids = set(
        pq.read_table(paths["businesses"], columns=["business_id"])
        .column("business_id")
        .to_pylist()
    )
    try:
        with duckdb.connect() as connection:
            history_rows = connection.execute(
                """
                SELECT history.task_id, interaction.business_id
                FROM read_parquet(?) AS history
                JOIN read_parquet(?) AS interaction USING (review_id)
                """,
                [str(paths["histories"]), str(paths["interactions"])],
            ).fetchall()
    except duckdb.Error as exc:
        raise CandidateBuildError(
            f"Could not validate frozen candidate histories: {exc}"
        ) from exc
    history_businesses: dict[str, set[str]] = defaultdict(set)
    for task_id, business_id in history_rows:
        history_businesses[str(task_id)].add(str(business_id))

    provenance_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    bucket_counts: Counter[str] = Counter()
    refill_task_ids: set[str] = set()
    for row in provenance_rows:
        task_id = str(row["task_id"])
        bucket = str(row["source_bucket"])
        provenance_by_task[task_id].append(row)
        bucket_counts[bucket] += 1
        if bucket.startswith("refill_"):
            refill_task_ids.add(task_id)
    if set(provenance_by_task) != retained_ids:
        raise CandidateBuildError(
            "Candidate provenance task IDs do not match frozen tasks"
        )

    validation_ids = {task.task_id for task in validation_tasks}
    test_ids = {task.task_id for task in test_tasks}
    for task_id, task in task_by_id.items():
        context = context_by_id[task_id]
        expected_split = str(context["split"])
        if (
            task.user_id != str(context["user_id"])
            or task.cutoff_time != context["cutoff_time"]
            or (task_id in validation_ids) != (expected_split == "validation")
            or (task_id in test_ids) != (expected_split == "test")
        ):
            raise CandidateBuildError(
                f"Frozen task does not match temporal context: {task_id!r}"
            )

        provenance = sorted(
            provenance_by_task[task_id],
            key=lambda row: int(row["final_position"]),
        )
        positions = [int(row["final_position"]) for row in provenance]
        provenance_ids = [str(row["business_id"]) for row in provenance]
        if (
            positions != list(range(1, 21))
            or provenance_ids != task.candidate_business_ids
            or not set(provenance_ids).issubset(static_business_ids)
        ):
            raise CandidateBuildError(
                f"Frozen task provenance is invalid: {task_id!r}"
            )
        target_rows = [
            row
            for row in provenance
            if str(row["source_bucket"]) == "target"
        ]
        if (
            len(target_rows) != 1
            or str(target_rows[0]["business_id"]) != truth_by_id[task_id]
        ):
            raise CandidateBuildError(
                f"Frozen task target provenance is invalid: {task_id!r}"
            )
        leaked_negatives = {
            str(row["business_id"])
            for row in provenance
            if str(row["source_bucket"]) != "target"
        }.intersection(history_businesses.get(task_id, set()))
        if leaked_negatives:
            raise CandidateBuildError(
                f"Frozen task contains historical negative businesses: {task_id!r}"
            )

    dropped_unavailable = sum(
        row["reason"] == "target_unavailable_before_cutoff"
        for row in dropped_rows
    )
    if dropped_unavailable != len(dropped_rows):
        raise CandidateBuildError("Frozen candidate output has an unknown drop reason")
    return CandidateBuildResult(
        status="skipped",
        validation_tasks_path=str(paths["validation_tasks"]),
        test_tasks_path=str(paths["test_tasks"]),
        ground_truth_path=str(paths["candidate_truth"]),
        provenance_path=str(paths["provenance"]),
        dropped_tasks_path=str(paths["dropped_tasks"]),
        input_tasks=len(contexts),
        retained_tasks=len(all_tasks),
        validation_tasks=len(validation_tasks),
        test_tasks=len(test_tasks),
        dropped_unavailable_targets=dropped_unavailable,
        tasks_using_refill=len(refill_task_ids),
        bucket_counts=dict(sorted(bucket_counts.items())),
    )


def build_candidate_tasks(
    processed_root: str | Path,
    task_root: str | Path,
    config: DataConfig,
    *,
    force: bool = False,
) -> CandidateBuildResult:
    """Build deterministic, time-valid candidate tasks behind one interface."""

    processed = Path(processed_root)
    tasks_root = Path(task_root)
    paths = _paths(processed, tasks_root)
    _require_inputs(paths)
    output_names = (
        "validation_tasks",
        "test_tasks",
        "candidate_truth",
        "provenance",
        "dropped_tasks",
    )
    if any(paths[name].exists() for name in output_names) and not force:
        return _validate_existing_outputs(paths)

    businesses, fine_index, group_index = _load_businesses(
        paths["businesses"],
        config,
    )
    review_index = _load_review_index(paths["reviews"])
    task_seeds, histories = _load_tasks_and_histories(paths)
    validation_tasks: list[RecommendationTask] = []
    test_tasks: list[RecommendationTask] = []
    truth_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, str]] = []
    bucket_counts: Counter[str] = Counter()
    tasks_using_refill = 0

    for task_seed in task_seeds:
        history = histories.get(task_seed.task_id, [])
        task, negatives, dropped_reason = _build_one_task(
            task_seed,
            history,
            businesses,
            fine_index,
            group_index,
            review_index,
            config,
        )
        if task is None:
            dropped_rows.append(
                {
                    "task_id": task_seed.task_id,
                    "split": task_seed.split,
                    "reason": str(dropped_reason),
                }
            )
            continue
        if any(bucket.startswith("refill_") for _, bucket in negatives):
            tasks_using_refill += 1
        source_by_business = {
            task_seed.target_business_id: "target",
            **{business_id: bucket for business_id, bucket in negatives},
        }
        for position, business_id in enumerate(
            task.candidate_business_ids,
            start=1,
        ):
            bucket = source_by_business[business_id]
            bucket_counts[bucket] += 1
            provenance_rows.append(
                {
                    "task_id": task.task_id,
                    "business_id": business_id,
                    "source_bucket": bucket,
                    "final_position": position,
                }
            )
        truth_rows.append(
            {
                "task_id": task.task_id,
                "target_business_id": task_seed.target_business_id,
            }
        )
        if task_seed.split == "validation":
            validation_tasks.append(task)
        elif task_seed.split == "test":
            test_tasks.append(task)
        else:
            raise CandidateBuildError(
                f"Task {task_seed.task_id!r} has unknown split {task_seed.split!r}"
            )

    _write_outputs(
        paths,
        validation_tasks,
        test_tasks,
        truth_rows,
        provenance_rows,
        dropped_rows,
    )
    retained_tasks = len(validation_tasks) + len(test_tasks)
    return CandidateBuildResult(
        status="written",
        validation_tasks_path=str(paths["validation_tasks"]),
        test_tasks_path=str(paths["test_tasks"]),
        ground_truth_path=str(paths["candidate_truth"]),
        provenance_path=str(paths["provenance"]),
        dropped_tasks_path=str(paths["dropped_tasks"]),
        input_tasks=len(task_seeds),
        retained_tasks=retained_tasks,
        validation_tasks=len(validation_tasks),
        test_tasks=len(test_tasks),
        dropped_unavailable_targets=len(dropped_rows),
        tasks_using_refill=tasks_using_refill,
        bucket_counts=dict(sorted(bucket_counts.items())),
    )


def write_candidate_build_report(
    result: CandidateBuildResult,
    output_path: str | Path,
) -> None:
    """Persist candidate counts, refill usage and drop audit as JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
