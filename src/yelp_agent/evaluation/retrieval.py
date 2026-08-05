"""Evaluate target-blind retrieval only after candidate files are frozen."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.experiments.artifacts import write_json_artifact
from yelp_agent.models import StrictModel


class RetrievalEvaluationError(RuntimeError):
    """Raised when retrieval artifacts violate the evaluation contract."""


class RetrievalMetrics(StrictModel):
    benchmark_name: Literal["Full Retrieval Benchmark V1"] = (
        "Full Retrieval Benchmark V1"
    )
    split: Literal["train", "validation", "test"]
    task_count: int = Field(ge=1)
    catalog_eligible_task_count: int = Field(ge=0)
    policy_reachable_task_count: int = Field(ge=0)
    target_in_history_task_count: int = Field(ge=0)
    catalog_ineligible_task_count: int = Field(ge=0)
    target_not_retrieved_count: int = Field(ge=0)
    candidate_without_prior_review_count: Literal[0] = 0
    candidate_in_user_history_count: Literal[0] = 0
    catalog_eligible_rate: float = Field(ge=0, le=1)
    primary_population: Literal[
        "catalog_eligible_and_not_previously_visited"
    ] = "catalog_eligible_and_not_previously_visited"
    recall_at: dict[str, float]
    all_task_recall_at: dict[str, float]
    route_recall_at: dict[str, dict[str, float]]
    mean_candidate_count: float = Field(ge=0)
    minimum_candidate_count: int = Field(ge=0)
    maximum_candidate_count: int = Field(ge=0)
    mean_catalog_size: float = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


def _required_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _validate_public_candidate_schema(path: Path) -> None:
    try:
        names = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, pa.ArrowException) as exc:
        raise RetrievalEvaluationError(
            f"Could not read retrieval candidates: {path}"
        ) from exc
    forbidden = {"target_business_id", "target_review_id", "ground_truth"}
    if names.intersection(forbidden):
        raise RetrievalEvaluationError(
            "Public retrieval candidates contain a ground-truth field"
        )


def _recall(
    rows: list[dict[str, object]],
    cutoff: int,
    *,
    denominator_filter,
    rank_field: str,
) -> float:
    population = [row for row in rows if denominator_filter(row)]
    if not population:
        return 0.0
    hits = sum(
        row.get(rank_field) is not None
        and int(row[rank_field]) <= cutoff
        for row in population
    )
    return hits / len(population)


def evaluate_full_retrieval(
    *,
    split: Literal["train", "validation", "test"],
    contexts_path: str | Path,
    ground_truth_path: str | Path,
    candidates_path: str | Path,
    task_audit_path: str | Path,
    reviews_path: str | Path,
    interactions_path: str | Path,
    metric_cutoffs: list[int],
    route_provenance_path: str | Path | None = None,
    metrics_output_path: str | Path | None = None,
    task_results_output_path: str | Path | None = None,
) -> RetrievalMetrics:
    """Score frozen candidates; target labels cross the seam only here."""

    contexts = _required_file(contexts_path, "Retrieval contexts")
    truth = _required_file(ground_truth_path, "Retrieval ground truth")
    candidates = _required_file(candidates_path, "Retrieval candidates")
    audit = _required_file(task_audit_path, "Retrieval task audit")
    reviews = _required_file(reviews_path, "Review data")
    interactions = _required_file(interactions_path, "Interaction data")
    provenance = (
        None
        if route_provenance_path is None
        else _required_file(route_provenance_path, "Route provenance")
    )
    if (
        not metric_cutoffs
        or metric_cutoffs != sorted(metric_cutoffs)
        or len(set(metric_cutoffs)) != len(metric_cutoffs)
        or any(cutoff <= 0 for cutoff in metric_cutoffs)
    ):
        raise ValueError("metric_cutoffs must be unique, positive and sorted")
    _validate_public_candidate_schema(candidates)

    provenance_cte = (
        """
        , route_hits AS (
            SELECT
                candidate.task_id,
                min(route_rank) FILTER (WHERE route = 'quality') AS quality_rank,
                min(route_rank) FILTER (WHERE route = 'category') AS category_rank,
                min(route_rank) FILTER (WHERE route = 'text') AS text_rank,
                min(route_rank) FILTER (WHERE route = 'location') AS location_rank
            FROM read_parquet(?) AS candidate
            JOIN truth
              ON truth.task_id = candidate.task_id
             AND truth.target_business_id = candidate.business_id
            GROUP BY candidate.task_id
        )
        """
        if provenance is not None
        else """
        , route_hits AS (
            SELECT
                CAST(NULL AS VARCHAR) AS task_id,
                CAST(NULL AS INTEGER) AS quality_rank,
                CAST(NULL AS INTEGER) AS category_rank,
                CAST(NULL AS INTEGER) AS text_rank,
                CAST(NULL AS INTEGER) AS location_rank
            WHERE false
        )
        """
    )
    query = f"""
        WITH selected_context AS (
            SELECT task_id, user_id, cutoff_time
            FROM read_parquet(?)
            WHERE split = ?
        ),
        truth AS (
            SELECT label.task_id, label.target_business_id
            FROM read_parquet(?) AS label
            JOIN selected_context USING (task_id)
        ),
        fused_hits AS (
            SELECT candidate.task_id, min(rank) AS target_rank
            FROM read_parquet(?) AS candidate
            JOIN truth
              ON truth.task_id = candidate.task_id
             AND truth.target_business_id = candidate.business_id
            GROUP BY candidate.task_id
        )
        {provenance_cte}
        SELECT
            context.task_id,
            context.user_id,
            context.cutoff_time,
            truth.target_business_id,
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
            fused.target_rank,
            routes.quality_rank,
            routes.category_rank,
            routes.text_rank,
            routes.location_rank
        FROM selected_context AS context
        JOIN truth USING (task_id)
        LEFT JOIN fused_hits AS fused USING (task_id)
        LEFT JOIN route_hits AS routes USING (task_id)
        ORDER BY context.task_id
    """
    parameters: list[object] = [
        str(contexts),
        split,
        str(truth),
        str(candidates),
    ]
    if provenance is not None:
        parameters.append(str(provenance))
    parameters.extend([str(reviews), str(interactions)])

    try:
        with duckdb.connect() as connection:
            connection.execute("SET enable_progress_bar = false")
            context_counts = connection.execute(
                """
                SELECT count(*), count(DISTINCT task_id)
                FROM read_parquet(?)
                WHERE split = ?
                """,
                [str(contexts), split],
            ).fetchone()
            truth_counts = connection.execute(
                """
                SELECT count(*), count(DISTINCT label.task_id)
                FROM read_parquet(?) AS label
                JOIN read_parquet(?) AS context USING (task_id)
                WHERE context.split = ?
                """,
                [str(truth), str(contexts), split],
            ).fetchone()
            invalid_candidate_tasks = connection.execute(
                """
                WITH candidate_stats AS (
                    SELECT
                        task_id,
                        count(*) AS rows,
                        count(DISTINCT business_id) AS businesses,
                        count(DISTINCT rank) AS ranks,
                        min(rank) AS first_rank,
                        max(rank) AS last_rank
                    FROM read_parquet(?)
                    GROUP BY task_id
                )
                SELECT count(*)
                FROM read_parquet(?) AS task_audit
                LEFT JOIN candidate_stats USING (task_id)
                WHERE task_audit.split = ?
                  AND (
                    rows IS NULL
                    OR rows != task_audit.candidate_count
                    OR businesses != rows
                    OR ranks != rows
                    OR first_rank != 1
                    OR last_rank != rows
                  )
                """,
                [str(candidates), str(audit), split],
            ).fetchone()[0]
            candidate_artifact_counts = connection.execute(
                """
                SELECT count(*), count(DISTINCT task_id)
                FROM read_parquet(?)
                """,
                [str(candidates)],
            ).fetchone()
            audit_artifact_counts = connection.execute(
                """
                SELECT count(*), count(DISTINCT task_id), sum(candidate_count)
                FROM read_parquet(?)
                WHERE split = ?
                """,
                [str(audit), split],
            ).fetchone()
            temporal_candidate_violations = connection.execute(
                """
                WITH selected_context AS (
                    SELECT task_id, user_id, cutoff_time
                    FROM read_parquet(?)
                    WHERE split = ?
                ),
                business_first_review AS (
                    SELECT business_id, min(date) AS first_review_time
                    FROM read_parquet(?)
                    GROUP BY business_id
                ),
                user_business_first_interaction AS (
                    SELECT
                        user_id,
                        business_id,
                        min(date) AS first_interaction_time
                    FROM read_parquet(?)
                    GROUP BY user_id, business_id
                )
                SELECT
                    count(*) FILTER (
                        WHERE first_review.first_review_time IS NULL
                           OR first_review.first_review_time >= context.cutoff_time
                    ),
                    count(*) FILTER (
                        WHERE first_interaction.first_interaction_time
                              < context.cutoff_time
                    )
                FROM read_parquet(?) AS candidate
                JOIN selected_context AS context USING (task_id)
                LEFT JOIN business_first_review AS first_review
                  ON first_review.business_id = candidate.business_id
                LEFT JOIN user_business_first_interaction AS first_interaction
                  ON first_interaction.user_id = context.user_id
                 AND first_interaction.business_id = candidate.business_id
                """,
                [
                    str(contexts),
                    split,
                    str(reviews),
                    str(interactions),
                    str(candidates),
                ],
            ).fetchone()
            invalid_provenance_groups = 0
            if provenance is not None:
                invalid_provenance_groups = connection.execute(
                    """
                    WITH route_stats AS (
                        SELECT
                            task_id,
                            route,
                            count(*) AS rows,
                            count(DISTINCT business_id) AS businesses,
                            count(DISTINCT route_rank) AS ranks,
                            min(route_rank) AS first_rank,
                            max(route_rank) AS last_rank
                        FROM read_parquet(?)
                        GROUP BY task_id, route
                    )
                    SELECT count(*)
                    FROM route_stats
                    WHERE route NOT IN ('quality', 'category', 'text', 'location')
                       OR rows != businesses
                       OR rows != ranks
                       OR first_rank != 1
                       OR last_rank != rows
                    """,
                    [str(provenance)],
                ).fetchone()[0]
            relation = connection.execute(query, parameters)
            result_table = relation.to_arrow_table()
            audit_rows = connection.execute(
                """
                SELECT candidate_count, catalog_size, latency_ms
                FROM read_parquet(?)
                WHERE split = ?
                ORDER BY task_id
                """,
                [str(audit), split],
            ).fetchall()
    except duckdb.Error as exc:
        raise RetrievalEvaluationError(
            f"Could not evaluate full retrieval artifacts: {exc}"
        ) from exc

    if (
        context_counts[0] == 0
        or context_counts[0] != context_counts[1]
        or truth_counts != context_counts
        or invalid_candidate_tasks
        or candidate_artifact_counts[1] != context_counts[0]
        or audit_artifact_counts[0] != context_counts[0]
        or audit_artifact_counts[1] != context_counts[0]
        or audit_artifact_counts[2] != candidate_artifact_counts[0]
        or temporal_candidate_violations != (0, 0)
        or invalid_provenance_groups
        or result_table.num_rows != context_counts[0]
        or len(audit_rows) != context_counts[0]
    ):
        raise RetrievalEvaluationError(
            "Retrieval task, truth, candidate or audit rows are inconsistent"
        )
    rows = result_table.to_pylist()
    reachable = lambda row: bool(row["catalog_eligible"]) and not bool(
        row["target_in_history"]
    )
    recall_at = {
        str(cutoff): _recall(
            rows,
            cutoff,
            denominator_filter=reachable,
            rank_field="target_rank",
        )
        for cutoff in metric_cutoffs
    }
    all_task_recall_at = {
        str(cutoff): _recall(
            rows,
            cutoff,
            denominator_filter=lambda row: True,
            rank_field="target_rank",
        )
        for cutoff in metric_cutoffs
    }
    route_recall_at = {
        route: {
            str(cutoff): _recall(
                rows,
                cutoff,
                denominator_filter=reachable,
                rank_field=f"{route}_rank",
            )
            for cutoff in metric_cutoffs
        }
        for route in ("quality", "category", "text", "location")
    }
    candidate_counts = [int(row[0]) for row in audit_rows]
    catalog_sizes = [int(row[1]) for row in audit_rows]
    latencies = [float(row[2]) for row in audit_rows]
    eligible_count = sum(bool(row["catalog_eligible"]) for row in rows)
    target_in_history_count = sum(bool(row["target_in_history"]) for row in rows)
    reachable_count = sum(reachable(row) for row in rows)
    largest_cutoff = metric_cutoffs[-1]
    missed = sum(
        reachable(row)
        and (
            row["target_rank"] is None
            or int(row["target_rank"]) > largest_cutoff
        )
        for row in rows
    )
    metrics = RetrievalMetrics(
        split=split,
        task_count=len(rows),
        catalog_eligible_task_count=eligible_count,
        policy_reachable_task_count=reachable_count,
        target_in_history_task_count=target_in_history_count,
        catalog_ineligible_task_count=len(rows) - eligible_count,
        target_not_retrieved_count=missed,
        candidate_without_prior_review_count=0,
        candidate_in_user_history_count=0,
        catalog_eligible_rate=eligible_count / len(rows),
        recall_at=recall_at,
        all_task_recall_at=all_task_recall_at,
        route_recall_at=route_recall_at,
        mean_candidate_count=float(np.mean(candidate_counts)),
        minimum_candidate_count=min(candidate_counts),
        maximum_candidate_count=max(candidate_counts),
        mean_catalog_size=float(np.mean(catalog_sizes)),
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
    )
    for values in (
        metrics.recall_at,
        metrics.all_task_recall_at,
        *metrics.route_recall_at.values(),
    ):
        if any(not math.isfinite(value) for value in values.values()):
            raise RetrievalEvaluationError("A retrieval metric is not finite")

    if task_results_output_path is not None:
        output = Path(task_results_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        partial.unlink(missing_ok=True)
        pq.write_table(result_table, partial, compression="zstd")
        os.replace(partial, output)
    if metrics_output_path is not None:
        write_json_artifact(metrics_output_path, metrics)
    return metrics
