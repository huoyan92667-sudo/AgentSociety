"""Validation-only selection of the Item-KNN temporal half-life."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.collaborative.item_knn import (
    ItemKNNHistoryEvent,
    ItemKNNRequest,
    TemporalItemKNNStore,
)
from yelp_agent.config import ItemKNNConfig, RetrievalConfig
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.experiments import write_json_artifact
from yelp_agent.models import StrictModel

ITEM_ROUTE_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("route_rank", pa.int32(), nullable=False),
        pa.field("business_id", pa.string(), nullable=False),
        pa.field("route_score", pa.float64(), nullable=False),
        pa.field("negative_evidence", pa.float64(), nullable=False),
        pa.field("positive_support_count", pa.int32(), nullable=False),
        pa.field("negative_support_count", pa.int32(), nullable=False),
    ]
)


class ItemKNNTuningError(RuntimeError):
    """Raised when validation-only Item-KNN tuning cannot be trusted."""


@dataclass(frozen=True, slots=True)
class ItemKNNTuningSources:
    businesses: Path
    reviews: Path
    interactions: Path
    contexts: Path
    ground_truth: Path
    base_route_provenance: Path
    positive_events: Path
    negative_events: Path
    neutral_events: Path


class ItemKNNTuningCandidate(StrictModel):
    label: str
    half_life_days: int | None
    fused_recall_at: dict[str, float]
    item_knn_recall_at: dict[str, float]
    policy_reachable_task_count: int = Field(ge=1)
    route_rows: int = Field(ge=0)
    missing_task_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    route_path: str


class ItemKNNTuningResult(StrictModel):
    status: Literal["written"] = "written"
    selection_split: Literal["validation"] = "validation"
    selection_objective: Literal["fused_recall_at_largest_k_then_smaller_k"] = (
        "fused_recall_at_largest_k_then_smaller_k"
    )
    selected_half_life_days: int | None
    selected_config_path: str
    results_path: str
    candidates: list[ItemKNNTuningCandidate]


class _Task(StrictModel):
    task_id: str
    split: Literal["validation"]
    user_id: str
    cutoff_time: datetime
    history_count: int = Field(ge=1)


def _load_validation_tasks(path: Path) -> list[_Task]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation contexts do not exist: {path}")
    rows = pq.read_table(
        path,
        columns=["task_id", "split", "user_id", "cutoff_time", "history_count"],
    ).to_pylist()
    tasks = [
        _Task.model_validate(row) for row in rows if row.get("split") == "validation"
    ]
    tasks.sort(key=lambda task: (task.cutoff_time, task.task_id))
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ItemKNNTuningError(
            "Validation contexts are empty or contain duplicate task IDs"
        )
    return tasks


def _score_route(
    *,
    tasks: list[_Task],
    data_view: TemporalDataView,
    sources: ItemKNNTuningSources,
    config: ItemKNNConfig,
    retrieval_config: RetrievalConfig,
    output_path: Path,
    progress: Callable[[str], None] | None,
) -> tuple[int, int, float, float]:
    store = TemporalItemKNNStore.from_event_artifacts(
        sources.positive_events,
        sources.negative_events,
        sources.neutral_events,
        config,
    )
    partial = output_path.with_name(output_path.name + ".partial")
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        partial,
        ITEM_ROUTE_SCHEMA,
        compression="zstd",
        use_dictionary=["task_id", "business_id"],
    )
    buffer: list[dict[str, object]] = []
    row_count = 0
    missing_count = 0
    latencies: list[float] = []

    def flush() -> None:
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=ITEM_ROUTE_SCHEMA))
            buffer.clear()

    try:
        for task_index, task in enumerate(tasks, start=1):
            started = perf_counter()
            history = data_view.user_history(task.user_id, task.cutoff_time)
            if len(history) != task.history_count:
                raise ItemKNNTuningError(f"Task {task.task_id!r} history count changed")
            history_ids = {item.business_id for item in history}
            catalog = data_view.review_catalog_before(task.cutoff_time)
            eligible_ids = tuple(
                business_id
                for business_id, review_count in zip(
                    catalog.business_ids,
                    catalog.review_counts,
                    strict=True,
                )
                if review_count > 0
                and (
                    not retrieval_config.exclude_history_businesses
                    or business_id not in history_ids
                )
            )
            if not eligible_ids:
                missing_count += 1
                latencies.append((perf_counter() - started) * 1000.0)
                continue
            result = store.score_candidates(
                ItemKNNRequest(
                    user_id=task.user_id,
                    cutoff_time=task.cutoff_time,
                    candidate_business_ids=eligible_ids,
                    history=tuple(
                        ItemKNNHistoryEvent(
                            business_id=item.business_id,
                            stars=item.stars,
                            date=item.date,
                        )
                        for item in history
                    ),
                )
            )
            if result.missing:
                missing_count += 1
            ranked = sorted(
                (score for score in result.scores if score.positive_score > 0.0),
                key=lambda score: (-score.positive_score, score.business_id),
            )[: retrieval_config.per_route_limit]
            for rank, score in enumerate(ranked, start=1):
                buffer.append(
                    {
                        "task_id": task.task_id,
                        "route_rank": rank,
                        "business_id": score.business_id,
                        "route_score": score.positive_score,
                        "negative_evidence": score.negative_evidence,
                        "positive_support_count": (score.positive_support_count),
                        "negative_support_count": (score.negative_support_count),
                    }
                )
            row_count += len(ranked)
            latencies.append((perf_counter() - started) * 1000.0)
            if len(buffer) >= 25_000:
                flush()
            if progress is not None and (
                task_index == len(tasks) or task_index % 250 == 0
            ):
                progress(
                    f"half_life={config.half_life_days}: "
                    f"{task_index}/{len(tasks)} validation tasks"
                )
        flush()
    except Exception:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(partial, output_path)
    return (
        row_count,
        missing_count,
        float(np.mean(latencies)) if latencies else 0.0,
        float(np.percentile(latencies, 95)) if latencies else 0.0,
    )


def _evaluate_route(
    *,
    sources: ItemKNNTuningSources,
    item_route_path: Path,
    retrieval_config: RetrievalConfig,
) -> tuple[dict[str, float], dict[str, float], int]:
    query = """
        WITH selected_context AS (
            SELECT task_id, user_id, cutoff_time
            FROM read_parquet(?)
            WHERE split = 'validation'
        ),
        truth AS (
            SELECT label.task_id, label.target_business_id
            FROM read_parquet(?) AS label
            JOIN selected_context USING (task_id)
        ),
        base_and_item AS (
            SELECT task_id, business_id, route_rank
            FROM read_parquet(?)
            UNION ALL
            SELECT task_id, business_id, route_rank
            FROM read_parquet(?)
        ),
        fused_scores AS (
            SELECT
                task_id,
                business_id,
                sum(1.0 / (? + route_rank)) AS fusion_score
            FROM base_and_item
            GROUP BY task_id, business_id
        ),
        fused AS (
            SELECT
                task_id,
                business_id,
                row_number() OVER (
                    PARTITION BY task_id
                    ORDER BY fusion_score DESC, business_id ASC
                ) AS fused_rank
            FROM fused_scores
            QUALIFY fused_rank <= ?
        ),
        item_hits AS (
            SELECT item.task_id, min(item.route_rank) AS item_rank
            FROM read_parquet(?) AS item
            JOIN truth
              ON truth.task_id = item.task_id
             AND truth.target_business_id = item.business_id
            GROUP BY item.task_id
        ),
        fused_hits AS (
            SELECT candidate.task_id, min(candidate.fused_rank) AS fused_rank
            FROM fused AS candidate
            JOIN truth
              ON truth.task_id = candidate.task_id
             AND truth.target_business_id = candidate.business_id
            GROUP BY candidate.task_id
        )
        SELECT
            context.task_id,
            EXISTS (
                SELECT 1
                FROM read_parquet(?) AS review
                WHERE review.business_id = truth.target_business_id
                  AND review.date < context.cutoff_time
            ) AS catalog_eligible,
            EXISTS (
                SELECT 1
                FROM read_parquet(?) AS interaction
                WHERE interaction.user_id = context.user_id
                  AND interaction.business_id = truth.target_business_id
                  AND interaction.date < context.cutoff_time
            ) AS target_in_history,
            fused_hits.fused_rank,
            item_hits.item_rank
        FROM selected_context AS context
        JOIN truth USING (task_id)
        LEFT JOIN fused_hits USING (task_id)
        LEFT JOIN item_hits USING (task_id)
        ORDER BY context.task_id
    """
    try:
        with duckdb.connect() as connection:
            rows = connection.execute(
                query,
                [
                    str(sources.contexts),
                    str(sources.ground_truth),
                    str(sources.base_route_provenance),
                    str(item_route_path),
                    retrieval_config.rrf_constant,
                    retrieval_config.candidate_limit,
                    str(item_route_path),
                    str(sources.reviews),
                    str(sources.interactions),
                ],
            ).fetchall()
    except duckdb.Error as exc:
        raise ItemKNNTuningError(
            f"Could not evaluate Item-KNN validation route: {exc}"
        ) from exc
    reachable = [row for row in rows if bool(row[1]) and not bool(row[2])]
    if not reachable:
        raise ItemKNNTuningError("No policy-reachable validation tasks")

    def recalls(rank_index: int) -> dict[str, float]:
        return {
            str(cutoff): sum(
                row[rank_index] is not None and int(row[rank_index]) <= cutoff
                for row in reachable
            )
            / len(reachable)
            for cutoff in retrieval_config.metric_cutoffs
        }

    return recalls(3), recalls(4), len(reachable)


def tune_item_knn(
    sources: ItemKNNTuningSources,
    output_root: str | Path,
    base_config: ItemKNNConfig,
    retrieval_config: RetrievalConfig,
    *,
    half_life_candidates: Sequence[int | None] = (None, 180, 365, 730),
    progress: Callable[[str], None] | None = None,
) -> ItemKNNTuningResult:
    """Freeze the validation-selected half-life without reading test tasks."""

    if not half_life_candidates or len(set(half_life_candidates)) != len(
        half_life_candidates
    ):
        raise ValueError("half_life_candidates must be non-empty and unique")
    allowed = {None, 180, 365, 730}
    if not set(half_life_candidates).issubset(allowed):
        raise ValueError(
            "half_life_candidates must be drawn from null, 180, 365 and 730"
        )
    for path in (
        sources.businesses,
        sources.reviews,
        sources.interactions,
        sources.contexts,
        sources.ground_truth,
        sources.base_route_provenance,
        sources.positive_events,
        sources.negative_events,
        sources.neutral_events,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Item-KNN tuning source is missing: {path}")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = _load_validation_tasks(sources.contexts)
    data_view = TemporalDataView(
        sources.businesses,
        sources.reviews,
        sources.interactions,
    )
    candidates: list[ItemKNNTuningCandidate] = []
    for half_life in half_life_candidates:
        label = "none" if half_life is None else str(half_life)
        route_path = root / f"item_knn_{label}_route.parquet"
        config = base_config.model_copy(update={"half_life_days": half_life})
        rows, missing, mean_latency, p95_latency = _score_route(
            tasks=tasks,
            data_view=data_view,
            sources=sources,
            config=config,
            retrieval_config=retrieval_config,
            output_path=route_path,
            progress=progress,
        )
        fused_recall, item_recall, reachable_count = _evaluate_route(
            sources=sources,
            item_route_path=route_path,
            retrieval_config=retrieval_config,
        )
        candidates.append(
            ItemKNNTuningCandidate(
                label=label,
                half_life_days=half_life,
                fused_recall_at=fused_recall,
                item_knn_recall_at=item_recall,
                policy_reachable_task_count=reachable_count,
                route_rows=rows,
                missing_task_count=missing,
                mean_latency_ms=mean_latency,
                p95_latency_ms=p95_latency,
                route_path=str(route_path),
            )
        )

    descending_cutoffs = tuple(reversed(retrieval_config.metric_cutoffs))

    def selection_key(candidate: ItemKNNTuningCandidate) -> tuple[float, ...]:
        return (
            *(candidate.fused_recall_at[str(cutoff)] for cutoff in descending_cutoffs),
            *(
                candidate.item_knn_recall_at[str(cutoff)]
                for cutoff in descending_cutoffs
            ),
            -candidate.mean_latency_ms,
            -(candidate.half_life_days or 0),
        )

    selected = max(candidates, key=selection_key)
    selected_config = base_config.model_copy(
        update={"half_life_days": selected.half_life_days}
    )
    selected_config_path = root / "selected_config.json"
    results_path = root / "tuning_results.json"
    write_json_artifact(selected_config_path, selected_config)
    result = ItemKNNTuningResult(
        selected_half_life_days=selected.half_life_days,
        selected_config_path=str(selected_config_path),
        results_path=str(results_path),
        candidates=candidates,
    )
    write_json_artifact(results_path, result)
    return result
