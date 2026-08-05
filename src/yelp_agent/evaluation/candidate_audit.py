"""Audit the current target-conditioned twenty-candidate benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import DataConfig
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel


class CandidateAuditError(RuntimeError):
    """Raised when benchmark inputs cannot be audited safely."""


class ProtocolProperties(StrictModel):
    target_conditioned_candidate_generation: bool
    target_used_for_same_fine_pool: bool
    target_used_for_related_pool: bool
    target_label_visible_to_ranker: bool
    ground_truth_loaded_only_by_audit_or_evaluation: bool
    evaluates_full_catalog_retrieval: bool


class SplitCandidateAudit(StrictModel):
    task_count: int = Field(ge=0)
    candidate_row_count: int = Field(ge=0)
    bucket_counts: dict[str, int]
    tasks_using_refill: int = Field(ge=0)
    refill_task_rate: float = Field(ge=0, le=1)
    target_position_counts: dict[str, int]
    dropped_task_count: int = Field(ge=0)
    dropped_reason_counts: dict[str, int]


class CandidateFrequencyAudit(StrictModel):
    business_universe_count: int = Field(ge=0)
    distinct_candidate_business_count: int = Field(ge=0)
    candidate_universe_coverage_rate: float = Field(ge=0, le=1)
    distinct_negative_business_count: int = Field(ge=0)
    negative_occurrence_count: int = Field(ge=0)
    median_negative_occurrences: float = Field(ge=0)
    p95_negative_occurrences: float = Field(ge=0)
    p99_negative_occurrences: float = Field(ge=0)
    max_negative_occurrences: int = Field(ge=0)
    top_negative_businesses: list[dict[str, int | str]]


class CandidateAuditReport(StrictModel):
    format_version: int
    benchmark_name: str
    benchmark_type: str
    total_task_count: int = Field(ge=0)
    required_candidate_count: int = Field(ge=1)
    required_negative_count: int = Field(ge=0)
    total_candidate_row_count: int = Field(ge=0)
    total_bucket_counts: dict[str, int]
    tasks_using_refill: int = Field(ge=0)
    refill_task_rate: float = Field(ge=0, le=1)
    target_position_counts: dict[str, int]
    dropped_task_count: int = Field(ge=0)
    dropped_reason_counts: dict[str, int]
    splits: dict[str, SplitCandidateAudit]
    candidate_frequency: CandidateFrequencyAudit
    candidate_configuration: dict[str, object]
    protocol_properties: ProtocolProperties
    invariant_violations: dict[str, int]
    invariant_violation_total: int = Field(ge=0)
    source_sha256: dict[str, str]
    limitations: list[str]


@dataclass(frozen=True)
class _AuditTask:
    task_id: str
    split: str
    cutoff_time: datetime
    candidate_business_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ProvenanceRow:
    task_id: str
    business_id: str
    source_bucket: str
    final_position: int


@dataclass(frozen=True)
class _Taxonomy:
    fine_categories: frozenset[str]
    groups: frozenset[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_cutoff(value: object, *, path: Path, line_number: int) -> datetime:
    if not isinstance(value, str) or not value:
        raise CandidateAuditError(
            f"Task cutoff_time is invalid at {path}:{line_number}"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateAuditError(
            f"Task cutoff_time is invalid at {path}:{line_number}"
        ) from exc


def _load_tasks(
    path: Path,
    split: str,
) -> tuple[list[_AuditTask], Counter[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate task file does not exist: {path}")
    tasks: list[_AuditTask] = []
    violations: Counter[str] = Counter()
    seen_task_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateAuditError(
                    f"Invalid task JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise CandidateAuditError(
                    f"Task row must be an object at {path}:{line_number}"
                )
            task_id = payload.get("task_id")
            candidates = payload.get("candidate_business_ids")
            if not isinstance(task_id, str) or not task_id:
                raise CandidateAuditError(
                    f"Task ID is invalid at {path}:{line_number}"
                )
            if not isinstance(candidates, list) or any(
                not isinstance(candidate, str) or not candidate
                for candidate in candidates
            ):
                raise CandidateAuditError(
                    f"Candidate list is invalid at {path}:{line_number}"
                )
            if task_id in seen_task_ids:
                violations["duplicate_task_id_rows"] += 1
            seen_task_ids.add(task_id)
            if len(candidates) != 20:
                violations["candidate_count_task_violations"] += 1
            if len(set(candidates)) != len(candidates):
                violations["duplicate_candidate_task_violations"] += 1
            expected_prefix = "validation:" if split == "validation" else "test:"
            if not task_id.startswith(expected_prefix):
                violations["split_task_id_prefix_violations"] += 1
            tasks.append(
                _AuditTask(
                    task_id=task_id,
                    split=split,
                    cutoff_time=_parse_cutoff(
                        payload.get("cutoff_time"),
                        path=path,
                        line_number=line_number,
                    ),
                    candidate_business_ids=tuple(candidates),
                )
            )
    if not tasks:
        raise CandidateAuditError(f"Candidate task file is empty: {path}")
    return tasks, violations


def _read_parquet_rows(
    path: Path,
    columns: list[str],
) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate audit input does not exist: {path}")
    try:
        return pq.read_table(path, columns=columns).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise CandidateAuditError(f"Could not read audit input: {path}") from exc


def _load_truth(
    path: Path,
) -> tuple[dict[str, str], int]:
    rows = _read_parquet_rows(path, ["task_id", "target_business_id"])
    truth: dict[str, str] = {}
    duplicate_rows = 0
    for row in rows:
        task_id = row.get("task_id")
        target = row.get("target_business_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(target, str)
            or not target
        ):
            raise CandidateAuditError(
                "Candidate ground truth contains an invalid row"
            )
        if task_id in truth:
            duplicate_rows += 1
            continue
        truth[task_id] = target
    return truth, duplicate_rows


def _load_provenance(path: Path) -> list[_ProvenanceRow]:
    rows = _read_parquet_rows(
        path,
        ["task_id", "business_id", "source_bucket", "final_position"],
    )
    provenance: list[_ProvenanceRow] = []
    for row in rows:
        task_id = row.get("task_id")
        business_id = row.get("business_id")
        source_bucket = row.get("source_bucket")
        final_position = row.get("final_position")
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(business_id, str)
            or not business_id
            or not isinstance(source_bucket, str)
            or not source_bucket
        ):
            raise CandidateAuditError("Candidate provenance contains an invalid row")
        try:
            position = int(final_position)
        except (TypeError, ValueError) as exc:
            raise CandidateAuditError(
                "Candidate provenance contains an invalid final_position"
            ) from exc
        provenance.append(
            _ProvenanceRow(
                task_id=task_id,
                business_id=business_id,
                source_bucket=source_bucket,
                final_position=position,
            )
        )
    return provenance


def _load_taxonomy(
    path: Path,
    config: DataConfig,
) -> dict[str, _Taxonomy]:
    rows = _read_parquet_rows(path, ["business_id", "categories"])
    broad = set(config.broad_categories)
    group_categories = {
        group: set(categories)
        for group, categories in config.category_groups.items()
    }
    businesses: dict[str, _Taxonomy] = {}
    for row in rows:
        business_id = row.get("business_id")
        categories = row.get("categories")
        if (
            not isinstance(business_id, str)
            or not business_id
            or business_id in businesses
            or not isinstance(categories, list)
        ):
            raise CandidateAuditError("Business taxonomy contains an invalid row")
        category_set = {
            str(category).strip()
            for category in categories
            if str(category).strip()
        }
        businesses[business_id] = _Taxonomy(
            fine_categories=frozenset(category_set.difference(broad)),
            groups=frozenset(
                group
                for group, group_values in group_categories.items()
                if category_set.intersection(group_values)
            ),
        )
    if not businesses:
        raise CandidateAuditError("Business taxonomy is empty")
    return businesses


def _load_history_businesses(
    histories_path: Path,
    interactions_path: Path,
    tasks: dict[str, _AuditTask],
) -> tuple[dict[str, set[str]], int]:
    for path in (histories_path, interactions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Candidate audit input does not exist: {path}")
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT history.task_id, interaction.business_id, interaction.date
                FROM read_parquet(?) AS history
                JOIN read_parquet(?) AS interaction USING (review_id)
                ORDER BY history.task_id, history.position
                """,
                [str(histories_path), str(interactions_path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise CandidateAuditError("Could not join temporal histories") from exc
    result: defaultdict[str, set[str]] = defaultdict(set)
    cutoff_violations = 0
    for task_id, business_id, date in rows:
        task_key = str(task_id)
        if task_key not in tasks:
            continue
        result[task_key].add(str(business_id))
        if not isinstance(date, datetime) or date >= tasks[task_key].cutoff_time:
            cutoff_violations += 1
    return dict(result), cutoff_violations


def _load_earliest_review_dates(path: Path) -> dict[str, datetime]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate audit input does not exist: {path}")
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                """
                SELECT business_id, min(date) AS first_review_time
                FROM read_parquet(?)
                GROUP BY business_id
                """,
                [str(path)],
            ).fetchall()
    except duckdb.Error as exc:
        raise CandidateAuditError("Could not aggregate review availability") from exc
    return {
        str(business_id): date
        for business_id, date in rows
        if isinstance(date, datetime)
    }


def _position_counts(rows: list[_ProvenanceRow]) -> dict[str, int]:
    counts = Counter(
        row.final_position for row in rows if row.source_bucket == "target"
    )
    return {str(position): counts[position] for position in range(1, 21)}


def _split_report(
    *,
    split: str,
    tasks: list[_AuditTask],
    provenance: list[_ProvenanceRow],
    dropped_rows: list[dict[str, object]],
) -> SplitCandidateAudit:
    task_ids = {task.task_id for task in tasks if task.split == split}
    split_rows = [row for row in provenance if row.task_id in task_ids]
    refill_ids = {
        row.task_id
        for row in split_rows
        if row.source_bucket.startswith("refill_")
    }
    source_split = "test" if split == "legacy_test" else split
    split_dropped = [
        row for row in dropped_rows if row.get("split") == source_split
    ]
    task_count = len(task_ids)
    return SplitCandidateAudit(
        task_count=task_count,
        candidate_row_count=len(split_rows),
        bucket_counts=dict(
            sorted(Counter(row.source_bucket for row in split_rows).items())
        ),
        tasks_using_refill=len(refill_ids),
        refill_task_rate=0.0 if not task_count else len(refill_ids) / task_count,
        target_position_counts=_position_counts(split_rows),
        dropped_task_count=len(split_dropped),
        dropped_reason_counts=dict(
            sorted(Counter(str(row.get("reason")) for row in split_dropped).items())
        ),
    )


def _candidate_frequency(
    tasks: list[_AuditTask],
    truth: dict[str, str],
    business_universe_count: int,
) -> CandidateFrequencyAudit:
    all_candidates: set[str] = set()
    negative_counts: Counter[str] = Counter()
    for task in tasks:
        target = truth.get(task.task_id)
        all_candidates.update(task.candidate_business_ids)
        negative_counts.update(
            candidate
            for candidate in task.candidate_business_ids
            if candidate != target
        )
    occurrences = np.asarray(list(negative_counts.values()), dtype=float)
    percentiles = (
        np.percentile(occurrences, [50, 95, 99])
        if occurrences.size
        else np.zeros(3)
    )
    top = sorted(
        negative_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]
    return CandidateFrequencyAudit(
        business_universe_count=business_universe_count,
        distinct_candidate_business_count=len(all_candidates),
        candidate_universe_coverage_rate=(
            0.0
            if not business_universe_count
            else len(all_candidates) / business_universe_count
        ),
        distinct_negative_business_count=len(negative_counts),
        negative_occurrence_count=sum(negative_counts.values()),
        median_negative_occurrences=float(percentiles[0]),
        p95_negative_occurrences=float(percentiles[1]),
        p99_negative_occurrences=float(percentiles[2]),
        max_negative_occurrences=max(negative_counts.values(), default=0),
        top_negative_businesses=[
            {"business_id": business_id, "occurrences": count}
            for business_id, count in top
        ],
    )


def audit_20_candidate_benchmark(
    *,
    validation_tasks_path: str | Path,
    test_tasks_path: str | Path,
    ground_truth_path: str | Path,
    provenance_path: str | Path,
    dropped_tasks_path: str | Path,
    businesses_path: str | Path,
    reviews_path: str | Path,
    interactions_path: str | Path,
    histories_path: str | Path,
    config: DataConfig,
) -> CandidateAuditReport:
    """Audit task construction without exposing truth to a ranker or Agent."""

    paths = {
        "validation_tasks": Path(validation_tasks_path),
        "legacy_test_tasks": Path(test_tasks_path),
        "candidate_ground_truth": Path(ground_truth_path),
        "candidate_provenance": Path(provenance_path),
        "dropped_tasks": Path(dropped_tasks_path),
        "businesses": Path(businesses_path),
        "reviews": Path(reviews_path),
        "interactions": Path(interactions_path),
        "temporal_histories": Path(histories_path),
    }
    validation_tasks, validation_violations = _load_tasks(
        paths["validation_tasks"], "validation"
    )
    test_tasks_raw, test_violations = _load_tasks(
        paths["legacy_test_tasks"], "legacy_test"
    )
    tasks = [*validation_tasks, *test_tasks_raw]
    violations = validation_violations + test_violations
    task_by_id: dict[str, _AuditTask] = {}
    for task in tasks:
        if task.task_id in task_by_id:
            violations["cross_split_duplicate_task_id_rows"] += 1
            continue
        task_by_id[task.task_id] = task

    truth, duplicate_truth_rows = _load_truth(paths["candidate_ground_truth"])
    violations["duplicate_ground_truth_rows"] += duplicate_truth_rows
    task_ids = set(task_by_id)
    truth_ids = set(truth)
    violations["missing_ground_truth_task_rows"] += len(task_ids - truth_ids)
    violations["unexpected_ground_truth_task_rows"] += len(truth_ids - task_ids)

    provenance = _load_provenance(paths["candidate_provenance"])
    provenance_by_task: defaultdict[str, list[_ProvenanceRow]] = defaultdict(list)
    for row in provenance:
        provenance_by_task[row.task_id].append(row)
    violations["unexpected_provenance_task_rows"] += sum(
        len(rows)
        for task_id, rows in provenance_by_task.items()
        if task_id not in task_ids
    )

    businesses = _load_taxonomy(paths["businesses"], config)
    history_businesses, history_cutoff_violations = _load_history_businesses(
        paths["temporal_histories"],
        paths["interactions"],
        task_by_id,
    )
    violations["history_cutoff_row_violations"] += history_cutoff_violations
    history_categories = {
        task_id: frozenset(
            category
            for business_id in business_ids
            for category in businesses.get(
                business_id,
                _Taxonomy(frozenset(), frozenset()),
            ).fine_categories
        )
        for task_id, business_ids in history_businesses.items()
    }
    earliest_reviews = _load_earliest_review_dates(paths["reviews"])

    allowed_buckets = {
        "target",
        "same_fine",
        "related",
        "preference",
        "random",
        "refill_same_fine",
        "refill_related",
        "refill_preference",
        "refill_random",
    }
    for task in task_by_id.values():
        target_id = truth.get(task.task_id)
        if target_id is None:
            continue
        if target_id not in task.candidate_business_ids:
            violations["target_missing_task_violations"] += 1
        rows = provenance_by_task.get(task.task_id, [])
        if len(rows) != len(task.candidate_business_ids):
            violations["provenance_count_task_violations"] += 1
        if {row.business_id for row in rows} != set(task.candidate_business_ids):
            violations["provenance_candidate_task_violations"] += 1
        target_rows = [row for row in rows if row.source_bucket == "target"]
        if len(target_rows) != 1 or target_rows[0].business_id != target_id:
            violations["target_bucket_task_violations"] += 1

        target_taxonomy = businesses.get(target_id)
        for row in rows:
            if row.source_bucket not in allowed_buckets:
                violations["unknown_source_bucket_rows"] += 1
            if (
                row.final_position < 1
                or row.final_position > len(task.candidate_business_ids)
                or task.candidate_business_ids[row.final_position - 1]
                != row.business_id
            ):
                violations["provenance_position_row_violations"] += 1
            taxonomy = businesses.get(row.business_id)
            if taxonomy is None:
                violations["unknown_candidate_business_rows"] += 1
                continue
            first_review = earliest_reviews.get(row.business_id)
            if first_review is None or first_review >= task.cutoff_time:
                violations["candidate_unavailable_before_cutoff_rows"] += 1
            if (
                row.business_id != target_id
                and row.business_id in history_businesses.get(task.task_id, set())
            ):
                violations["negative_history_overlap_rows"] += 1
            if target_taxonomy is None:
                continue
            if row.source_bucket in {"same_fine", "refill_same_fine"} and not (
                taxonomy.fine_categories.intersection(
                    target_taxonomy.fine_categories
                )
            ):
                violations["same_fine_bucket_rule_row_violations"] += 1
            if row.source_bucket in {"related", "refill_related"} and (
                taxonomy.fine_categories.intersection(
                    target_taxonomy.fine_categories
                )
                or not taxonomy.groups.intersection(target_taxonomy.groups)
            ):
                violations["related_bucket_rule_row_violations"] += 1
            if row.source_bucket in {"preference", "refill_preference"} and not (
                taxonomy.fine_categories.intersection(
                    history_categories.get(task.task_id, frozenset())
                )
            ):
                violations["preference_bucket_rule_row_violations"] += 1

    invariant_names = (
        "duplicate_task_id_rows",
        "cross_split_duplicate_task_id_rows",
        "candidate_count_task_violations",
        "duplicate_candidate_task_violations",
        "split_task_id_prefix_violations",
        "duplicate_ground_truth_rows",
        "missing_ground_truth_task_rows",
        "unexpected_ground_truth_task_rows",
        "unexpected_provenance_task_rows",
        "target_missing_task_violations",
        "provenance_count_task_violations",
        "provenance_candidate_task_violations",
        "target_bucket_task_violations",
        "provenance_position_row_violations",
        "unknown_source_bucket_rows",
        "unknown_candidate_business_rows",
        "candidate_unavailable_before_cutoff_rows",
        "negative_history_overlap_rows",
        "history_cutoff_row_violations",
        "same_fine_bucket_rule_row_violations",
        "related_bucket_rule_row_violations",
        "preference_bucket_rule_row_violations",
    )
    normalized_violations = {
        name: int(violations[name]) for name in invariant_names
    }

    dropped_rows = _read_parquet_rows(
        paths["dropped_tasks"],
        ["task_id", "split", "reason"],
    )
    all_refill_task_ids = {
        row.task_id
        for row in provenance
        if row.task_id in task_ids and row.source_bucket.startswith("refill_")
    }
    total_tasks = len(task_by_id)
    target_positions = _position_counts(
        [row for row in provenance if row.task_id in task_ids]
    )
    total_bucket_counts = dict(
        sorted(
            Counter(
                row.source_bucket
                for row in provenance
                if row.task_id in task_ids
            ).items()
        )
    )
    dropped_reason_counts = dict(
        sorted(Counter(str(row.get("reason")) for row in dropped_rows).items())
    )
    report = CandidateAuditReport(
        format_version=1,
        benchmark_name="Current 20-Candidate Controlled Reranking",
        benchmark_type="ground-truth-conditioned closed-set reranking",
        total_task_count=total_tasks,
        required_candidate_count=config.candidate_count,
        required_negative_count=config.candidate_count - 1,
        total_candidate_row_count=sum(
            len(task.candidate_business_ids) for task in task_by_id.values()
        ),
        total_bucket_counts=total_bucket_counts,
        tasks_using_refill=len(all_refill_task_ids),
        refill_task_rate=(
            0.0 if not total_tasks else len(all_refill_task_ids) / total_tasks
        ),
        target_position_counts=target_positions,
        dropped_task_count=len(dropped_rows),
        dropped_reason_counts=dropped_reason_counts,
        splits={
            "validation": _split_report(
                split="validation",
                tasks=list(task_by_id.values()),
                provenance=provenance,
                dropped_rows=dropped_rows,
            ),
            "legacy_test": _split_report(
                split="legacy_test",
                tasks=list(task_by_id.values()),
                provenance=provenance,
                dropped_rows=dropped_rows,
            ),
        },
        candidate_frequency=_candidate_frequency(
            list(task_by_id.values()),
            truth,
            len(businesses),
        ),
        candidate_configuration={
            "random_seed": config.random_seed,
            "candidate_count": config.candidate_count,
            "same_fine_requested": config.hard_negative_same_category,
            "related_requested": config.hard_negative_related_category,
            "preference_requested": config.preference_negative_count,
            "random_requested": config.random_negative_count,
            "broad_categories": sorted(config.broad_categories),
            "category_groups": {
                name: sorted(categories)
                for name, categories in sorted(config.category_groups.items())
            },
        },
        protocol_properties=ProtocolProperties(
            target_conditioned_candidate_generation=True,
            target_used_for_same_fine_pool=True,
            target_used_for_related_pool=True,
            target_label_visible_to_ranker=False,
            ground_truth_loaded_only_by_audit_or_evaluation=True,
            evaluates_full_catalog_retrieval=False,
        ),
        invariant_violations=normalized_violations,
        invariant_violation_total=sum(normalized_violations.values()),
        source_sha256={
            name: _sha256_file(path) for name, path in sorted(paths.items())
        },
        limitations=[
            (
                "The target business is known before the nineteen negatives "
                "are constructed."
            ),
            (
                "Target fine categories create the same-fine pool and target "
                "coarse groups create the related pool."
            ),
            (
                "Ranking metrics measure ordering inside a frozen set of "
                "twenty candidates, not retrieval from all Philadelphia "
                "businesses."
            ),
            (
                "Legacy Test V0 has already been observed and is restricted "
                "to historical comparison."
            ),
            (
                "Candidate difficulty and repeated-negative frequency can "
                "influence closed-set ranking metrics."
            ),
            (
                "The fine-category exclusion list contains only Food, "
                "Nightlife, Restaurants, and Shopping; broad labels such as "
                "Bars can still act as fine categories."
            ),
        ],
    )
    return report


def write_candidate_audit(
    report: CandidateAuditReport,
    output_path: str | Path,
) -> None:
    """Atomically write one deterministic machine-readable audit report."""

    write_json_artifact(output_path, report)
