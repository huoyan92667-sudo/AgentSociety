"""Label-isolated end-to-end and retrieval-conditioned ranking evaluation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.learning_to_rank.model import PairwiseLogisticModel
from yelp_agent.models import StrictModel


class HybridV2EvaluationError(RuntimeError):
    """Raised when evaluation inputs violate the frozen ranking contract."""


class HybridV2RankingMetrics(StrictModel):
    model_name: str = Field(min_length=1)
    split: Literal["validation", "test"]
    blend_alpha: float = Field(ge=0, le=1)
    task_count: int = Field(ge=1)
    primary_task_count: int = Field(ge=1)
    catalog_ineligible_task_count: int = Field(ge=0)
    target_in_history_task_count: int = Field(ge=0)
    target_retrieved_count: int = Field(ge=0)
    target_not_retrieved_count: int = Field(ge=0)
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    avg_hr: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    retrieved_target_hr_at_1: float = Field(ge=0, le=1)
    retrieved_target_hr_at_3: float = Field(ge=0, le=1)
    retrieved_target_hr_at_5: float = Field(ge=0, le=1)
    retrieved_target_mrr: float = Field(ge=0, le=1)
    all_task_hr_at_1: float = Field(ge=0, le=1)
    all_task_hr_at_3: float = Field(ge=0, le=1)
    all_task_hr_at_5: float = Field(ge=0, le=1)


class HybridV2RankingWriteResult(StrictModel):
    status: Literal["written"] = "written"
    output_path: str
    task_count: int = Field(ge=1)
    row_count: int = Field(ge=1)


def _required(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _model_expression(model: PairwiseLogisticModel | None) -> str:
    if model is None:
        return "hybrid_v1_score"
    terms = [
        f"({float(coefficient / scale):.17g}) * {_quoted(name)}"
        for name, scale, coefficient in zip(
            model.feature_names,
            model.feature_scale,
            model.coefficients,
            strict=True,
        )
        if coefficient != 0
    ]
    return " + ".join(terms) if terms else "0.0"


def _mean_indicator(rows: list[dict[str, object]], cutoff: int) -> float:
    if not rows:
        return 0.0
    return sum(
        row["target_rank"] is not None and int(row["target_rank"]) <= cutoff
        for row in rows
    ) / len(rows)


def _mrr(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(
        0.0 if row["target_rank"] is None else 1.0 / int(row["target_rank"])
        for row in rows
    ) / len(rows)


def _validate_features(path: Path, model: PairwiseLogisticModel | None) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    names = set(schema.names)
    required = {"task_id", "business_id", "hybrid_v1_score"}
    if model is not None:
        required.update(model.feature_names)
    if not required.issubset(names):
        raise HybridV2EvaluationError(
            "Evaluation feature artifact does not match the frozen model"
        )
    forbidden = {"target_business_id", "target_review_id", "ground_truth"}
    if names.intersection(forbidden):
        raise HybridV2EvaluationError(
            "Evaluation features expose a forbidden ground-truth column"
        )


def evaluate_hybrid_v2_ranking(
    *,
    features_path: str | Path,
    contexts_path: str | Path,
    ground_truth_path: str | Path,
    reviews_path: str | Path,
    interactions_path: str | Path,
    split: Literal["validation", "test"],
    model_name: str,
    model: PairwiseLogisticModel | None,
    blend_alpha: float,
    task_filter_path: str | Path | None = None,
) -> HybridV2RankingMetrics:
    """Evaluate labels only after candidate features and model are frozen."""

    if not 0 <= blend_alpha <= 1:
        raise ValueError("blend_alpha must be between zero and one")
    features = _required(features_path, "Ranking features")
    contexts = _required(contexts_path, "Task contexts")
    truth = _required(ground_truth_path, "Ground truth")
    reviews = _required(reviews_path, "Review events")
    interactions = _required(interactions_path, "User interactions")
    task_filter = (
        None
        if task_filter_path is None
        else _required(task_filter_path, "Evaluation task filter")
    )
    _validate_features(features, model)
    model_score = _model_expression(model)
    alpha = float(blend_alpha)
    candidate_filter = (
        "" if task_filter is None else "JOIN read_parquet(?) AS filter USING (task_id)"
    )
    context_filter = (
        "" if task_filter is None else "JOIN read_parquet(?) AS filter USING (task_id)"
    )
    query = f"""
        WITH candidate AS (
            SELECT
                task_id,
                business_id,
                hybrid_v1_score,
                ({model_score}) AS model_score
            FROM read_parquet(?)
            {candidate_filter}
        ), ranked AS (
            SELECT
                *,
                count(*) OVER (PARTITION BY task_id) AS candidate_count,
                row_number() OVER (
                    PARTITION BY task_id
                    ORDER BY model_score DESC, business_id
                ) AS model_rank,
                row_number() OVER (
                    PARTITION BY task_id
                    ORDER BY hybrid_v1_score DESC, business_id
                ) AS v1_rank
            FROM candidate
        ), blended AS (
            SELECT
                *,
                {alpha:.17g} * CASE
                    WHEN candidate_count = 1 THEN 1.0
                    ELSE (candidate_count - model_rank)::DOUBLE
                         / (candidate_count - 1)
                END
                + {1.0 - alpha:.17g} * CASE
                    WHEN candidate_count = 1 THEN 1.0
                    ELSE (candidate_count - v1_rank)::DOUBLE
                         / (candidate_count - 1)
                END AS blend_score
            FROM ranked
        ), final_ranking AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY task_id
                    ORDER BY blend_score DESC, business_id
                ) AS final_rank
            FROM blended
        ), selected_context AS (
            SELECT task_id, user_id, cutoff_time
            FROM read_parquet(?)
            {context_filter}
            WHERE split = ?
        ), truth AS (
            SELECT label.task_id, label.target_business_id
            FROM read_parquet(?) AS label
            JOIN selected_context USING (task_id)
        ), target_rank AS (
            SELECT ranking.task_id, ranking.final_rank AS target_rank
            FROM final_ranking AS ranking
            JOIN truth
              ON truth.task_id = ranking.task_id
             AND truth.target_business_id = ranking.business_id
        )
        SELECT
            context.task_id,
            EXISTS (
                SELECT 1 FROM read_parquet(?) AS review
                WHERE review.business_id = truth.target_business_id
                  AND review.date < context.cutoff_time
            ) AS catalog_eligible,
            EXISTS (
                SELECT 1 FROM read_parquet(?) AS interaction
                WHERE interaction.user_id = context.user_id
                  AND interaction.business_id = truth.target_business_id
                  AND interaction.date < context.cutoff_time
            ) AS target_in_history,
            target_rank.target_rank
        FROM selected_context AS context
        JOIN truth USING (task_id)
        LEFT JOIN target_rank USING (task_id)
        ORDER BY context.task_id
    """
    try:
        with duckdb.connect() as connection:
            duplicate_candidates = connection.execute(
                """
                SELECT count(*) - count(DISTINCT task_id || chr(0) || business_id)
                FROM read_parquet(?)
                """,
                [str(features)],
            ).fetchone()[0]
            if int(duplicate_candidates) != 0:
                raise HybridV2EvaluationError(
                    "Evaluation features contain duplicate candidates"
                )
            if "label" in pq.ParquetFile(features).schema_arrow.names:
                exposed_labels = connection.execute(
                    "SELECT count(*) FROM read_parquet(?) WHERE label IS NOT NULL",
                    [str(features)],
                ).fetchone()[0]
                if int(exposed_labels) != 0:
                    raise HybridV2EvaluationError(
                        "Evaluation features must not contain training labels"
                    )
            parameters: list[object] = [str(features)]
            if task_filter is not None:
                parameters.append(str(task_filter))
            parameters.append(str(contexts))
            if task_filter is not None:
                parameters.append(str(task_filter))
            parameters.extend([split, str(truth), str(reviews), str(interactions)])
            result = connection.execute(query, parameters).to_arrow_table()
    except duckdb.Error as exc:
        raise HybridV2EvaluationError(
            f"Could not evaluate frozen Hybrid V2 ranking: {exc}"
        ) from exc
    rows = result.to_pylist()
    if not rows:
        raise HybridV2EvaluationError("Evaluation population is empty")
    primary = [
        row
        for row in rows
        if bool(row["catalog_eligible"]) and not bool(row["target_in_history"])
    ]
    if not primary:
        raise HybridV2EvaluationError("Primary evaluation population is empty")
    retrieved = [row for row in primary if row["target_rank"] is not None]
    hr1 = _mean_indicator(primary, 1)
    hr3 = _mean_indicator(primary, 3)
    hr5 = _mean_indicator(primary, 5)
    ndcg5 = sum(
        0.0
        if row["target_rank"] is None or int(row["target_rank"]) > 5
        else 1.0 / math.log2(int(row["target_rank"]) + 1)
        for row in primary
    ) / len(primary)
    return HybridV2RankingMetrics(
        model_name=model_name,
        split=split,
        blend_alpha=alpha,
        task_count=len(rows),
        primary_task_count=len(primary),
        catalog_ineligible_task_count=sum(
            not bool(row["catalog_eligible"]) for row in rows
        ),
        target_in_history_task_count=sum(
            bool(row["target_in_history"]) for row in rows
        ),
        target_retrieved_count=len(retrieved),
        target_not_retrieved_count=len(primary) - len(retrieved),
        hr_at_1=hr1,
        hr_at_3=hr3,
        hr_at_5=hr5,
        avg_hr=(hr1 + hr3 + hr5) / 3.0,
        mrr=_mrr(primary),
        ndcg_at_5=ndcg5,
        retrieved_target_hr_at_1=_mean_indicator(retrieved, 1),
        retrieved_target_hr_at_3=_mean_indicator(retrieved, 3),
        retrieved_target_hr_at_5=_mean_indicator(retrieved, 5),
        retrieved_target_mrr=_mrr(retrieved),
        all_task_hr_at_1=_mean_indicator(rows, 1),
        all_task_hr_at_3=_mean_indicator(rows, 3),
        all_task_hr_at_5=_mean_indicator(rows, 5),
    )


def write_hybrid_v2_rankings(
    *,
    features_path: str | Path,
    output_path: str | Path,
    model: PairwiseLogisticModel,
    blend_alpha: float,
) -> HybridV2RankingWriteResult:
    """Publish target-blind complete rankings for every frozen candidate set."""

    if not 0 <= blend_alpha <= 1:
        raise ValueError("blend_alpha must be between zero and one")
    features = _required(features_path, "Ranking features")
    _validate_features(features, model)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Hybrid V2 rankings already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    model_score = _model_expression(model)
    alpha = float(blend_alpha)
    destination = str(partial.resolve()).replace("'", "''")
    query = f"""
        COPY (
            WITH candidate AS (
                SELECT
                    task_id,
                    business_id,
                    hybrid_v1_score,
                    ({model_score}) AS model_score
                FROM read_parquet(?)
            ), ranked AS (
                SELECT
                    *,
                    count(*) OVER (PARTITION BY task_id) AS candidate_count,
                    row_number() OVER (
                        PARTITION BY task_id
                        ORDER BY model_score DESC, business_id
                    ) AS model_rank,
                    row_number() OVER (
                        PARTITION BY task_id
                        ORDER BY hybrid_v1_score DESC, business_id
                    ) AS v1_rank
                FROM candidate
            ), blended AS (
                SELECT
                    *,
                    {alpha:.17g} * CASE
                        WHEN candidate_count = 1 THEN 1.0
                        ELSE (candidate_count - model_rank)::DOUBLE
                             / (candidate_count - 1)
                    END
                    + {1.0 - alpha:.17g} * CASE
                        WHEN candidate_count = 1 THEN 1.0
                        ELSE (candidate_count - v1_rank)::DOUBLE
                             / (candidate_count - 1)
                    END AS blend_score
                FROM ranked
            ), final AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY task_id
                        ORDER BY blend_score DESC, business_id
                    ) AS rank
                FROM blended
            )
            SELECT
                task_id,
                business_id,
                rank,
                model_rank,
                v1_rank,
                model_score,
                hybrid_v1_score,
                blend_score
            FROM final
            ORDER BY task_id, rank, business_id
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    try:
        with duckdb.connect() as connection:
            connection.execute(query, [str(features)])
            task_count, row_count = connection.execute(
                "SELECT count(DISTINCT task_id), count(*) FROM read_parquet(?)",
                [str(partial)],
            ).fetchone()
    except duckdb.Error as exc:
        partial.unlink(missing_ok=True)
        raise HybridV2EvaluationError(
            f"Could not publish Hybrid V2 rankings: {exc}"
        ) from exc
    os.replace(partial, output)
    return HybridV2RankingWriteResult(
        output_path=str(output),
        task_count=int(task_count),
        row_count=int(row_count),
    )
