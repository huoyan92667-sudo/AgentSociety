"""Compare frozen four-route and five-route retrieval results."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel


class ItemKNNComparisonError(RuntimeError):
    """Raised when retrieval comparison inputs are not aligned."""


class ItemKNNHistoryBucket(StrictModel):
    task_count: int = Field(ge=1)
    baseline_recall_at: dict[str, float]
    item_knn_fused_recall_at: dict[str, float]
    recall_delta_at: dict[str, float]
    item_knn_route_recall_at: dict[str, float]
    novel_hits_at: dict[str, int]
    damaged_hits_at: dict[str, int]


class ItemKNNRetrievalComparison(StrictModel):
    task_count: int = Field(ge=1)
    population: str = "catalog_eligible_and_not_previously_visited"
    baseline_recall_at: dict[str, float]
    item_knn_fused_recall_at: dict[str, float]
    recall_delta_at: dict[str, float]
    item_knn_route_recall_at: dict[str, float]
    novel_fused_hits_at: dict[str, int]
    damaged_fused_hits_at: dict[str, int]
    item_unique_vs_category_text_at: dict[str, int]
    history_buckets: dict[str, ItemKNNHistoryBucket]


def _read_rows(path: Path, label: str) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        return pq.read_table(path).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise ItemKNNComparisonError(f"Could not read {label}: {path}") from exc


def _rank_hit(value: object, cutoff: int) -> bool:
    return value is not None and int(value) <= cutoff


def _history_bucket(history_count: int) -> str:
    if history_count <= 15:
        return "8-15"
    if history_count <= 30:
        return "16-30"
    if history_count <= 60:
        return "31-60"
    return "61+"


def _bucket_metrics(
    rows: list[dict[str, object]],
    metric_cutoffs: list[int],
) -> ItemKNNHistoryBucket:
    task_count = len(rows)
    baseline_recall: dict[str, float] = {}
    item_fused_recall: dict[str, float] = {}
    item_route_recall: dict[str, float] = {}
    novel: dict[str, int] = {}
    damaged: dict[str, int] = {}
    for cutoff in metric_cutoffs:
        key = str(cutoff)
        baseline_hits = [_rank_hit(row["baseline_rank"], cutoff) for row in rows]
        fused_hits = [_rank_hit(row["item_knn_fused_rank"], cutoff) for row in rows]
        route_hits = [_rank_hit(row["item_knn_rank"], cutoff) for row in rows]
        baseline_recall[key] = sum(baseline_hits) / task_count
        item_fused_recall[key] = sum(fused_hits) / task_count
        item_route_recall[key] = sum(route_hits) / task_count
        novel[key] = sum(
            not baseline_hit and fused_hit
            for baseline_hit, fused_hit in zip(
                baseline_hits,
                fused_hits,
                strict=True,
            )
        )
        damaged[key] = sum(
            baseline_hit and not fused_hit
            for baseline_hit, fused_hit in zip(
                baseline_hits,
                fused_hits,
                strict=True,
            )
        )
    return ItemKNNHistoryBucket(
        task_count=task_count,
        baseline_recall_at=baseline_recall,
        item_knn_fused_recall_at=item_fused_recall,
        recall_delta_at={
            key: item_fused_recall[key] - baseline_recall[key]
            for key in baseline_recall
        },
        item_knn_route_recall_at=item_route_recall,
        novel_hits_at=novel,
        damaged_hits_at=damaged,
    )


def compare_item_knn_retrieval(
    baseline_task_results_path: str | Path,
    item_knn_task_results_path: str | Path,
    contexts_path: str | Path,
    *,
    metric_cutoffs: list[int],
    output_path: str | Path | None = None,
) -> ItemKNNRetrievalComparison:
    """Measure new hits and rank damage after adding the Item-KNN route."""

    if (
        not metric_cutoffs
        or metric_cutoffs != sorted(metric_cutoffs)
        or len(metric_cutoffs) != len(set(metric_cutoffs))
        or any(cutoff <= 0 for cutoff in metric_cutoffs)
    ):
        raise ValueError("metric_cutoffs must be unique, positive and sorted")
    baseline_rows = _read_rows(
        Path(baseline_task_results_path),
        "baseline task results",
    )
    item_rows = _read_rows(
        Path(item_knn_task_results_path),
        "Item-KNN task results",
    )
    contexts = _read_rows(Path(contexts_path), "retrieval contexts")
    baseline = {str(row["task_id"]): row for row in baseline_rows}
    item_knn = {str(row["task_id"]): row for row in item_rows}
    history_counts = {
        str(row["task_id"]): int(row["history_count"]) for row in contexts
    }
    if (
        len(baseline) != len(baseline_rows)
        or len(item_knn) != len(item_rows)
        or set(baseline) != set(item_knn)
        or not set(baseline).issubset(history_counts)
    ):
        raise ItemKNNComparisonError(
            "Baseline, Item-KNN and context task IDs are not aligned"
        )

    rows: list[dict[str, object]] = []
    for task_id in sorted(baseline):
        old = baseline[task_id]
        new = item_knn[task_id]
        if bool(old["catalog_eligible"]) != bool(new["catalog_eligible"]) or bool(
            old["target_in_history"]
        ) != bool(new["target_in_history"]):
            raise ItemKNNComparisonError(
                f"Task {task_id!r} changed evaluation population"
            )
        if not bool(new["catalog_eligible"]) or bool(new["target_in_history"]):
            continue
        rows.append(
            {
                "task_id": task_id,
                "history_count": history_counts[task_id],
                "baseline_rank": old.get("target_rank"),
                "item_knn_fused_rank": new.get("target_rank"),
                "item_knn_rank": new.get("item_knn_rank"),
                "category_rank": new.get("category_rank"),
                "text_rank": new.get("text_rank"),
            }
        )
    if not rows:
        raise ItemKNNComparisonError("No policy-reachable tasks to compare")

    overall = _bucket_metrics(rows, metric_cutoffs)
    unique: dict[str, int] = {}
    for cutoff in metric_cutoffs:
        unique[str(cutoff)] = sum(
            _rank_hit(row["item_knn_rank"], cutoff)
            and not _rank_hit(row["category_rank"], cutoff)
            and not _rank_hit(row["text_rank"], cutoff)
            for row in rows
        )
    bucket_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        label = _history_bucket(int(row["history_count"]))
        bucket_rows.setdefault(label, []).append(row)
    result = ItemKNNRetrievalComparison(
        task_count=len(rows),
        baseline_recall_at=overall.baseline_recall_at,
        item_knn_fused_recall_at=overall.item_knn_fused_recall_at,
        recall_delta_at=overall.recall_delta_at,
        item_knn_route_recall_at=overall.item_knn_route_recall_at,
        novel_fused_hits_at=overall.novel_hits_at,
        damaged_fused_hits_at=overall.damaged_hits_at,
        item_unique_vs_category_text_at=unique,
        history_buckets={
            label: _bucket_metrics(bucket, metric_cutoffs)
            for label, bucket in sorted(bucket_rows.items())
        },
    )
    if output_path is not None:
        write_json_artifact(output_path, result)
    return result
