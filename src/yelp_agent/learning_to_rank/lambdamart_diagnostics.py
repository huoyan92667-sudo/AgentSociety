"""Segmented and paired Validation diagnostics for Logistic versus LambdaMART."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
from pydantic import Field

from yelp_agent.learning_to_rank.robustness import (
    PairedBootstrapResult,
    paired_user_bootstrap,
)
from yelp_agent.models import StrictModel


@dataclass(frozen=True, slots=True)
class LambdaMARTDiagnosticSources:
    logistic_predictions: Path
    lambdamart_predictions: Path
    validation_contexts: Path
    validation_ground_truth: Path
    reviews: Path
    interactions: Path
    businesses: Path


class SegmentRankingMetrics(StrictModel):
    task_count: int = Field(ge=1)
    target_retrieved_count: int = Field(ge=0)
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    avg_hr: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)


class SegmentRankingComparison(StrictModel):
    dimension: Literal["history_count", "category_familiarity"]
    value: str = Field(min_length=1)
    logistic: SegmentRankingMetrics
    lambdamart: SegmentRankingMetrics
    avg_hr_delta: float
    mrr_delta: float


class LambdaMARTValidationDiagnostics(StrictModel):
    experiment_name: Literal["Hybrid V2-B segmented validation diagnostics"]
    population: Literal["catalog_eligible_and_not_previously_visited"]
    category_definition: Literal["fine_category_overlap_before_cutoff"]
    segments: list[SegmentRankingComparison]
    paired_user_avg_hr: PairedBootstrapResult
    legacy_test_files_read: Literal[False] = False


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _metrics(rows: list[dict[str, object]], rank_name: str) -> SegmentRankingMetrics:
    count = len(rows)
    if count == 0:
        raise ValueError("Ranking segment cannot be empty")

    def hit(cutoff: int) -> float:
        return (
            sum(
                row[rank_name] is not None and int(row[rank_name]) <= cutoff
                for row in rows
            )
            / count
        )

    hr1, hr3, hr5 = hit(1), hit(3), hit(5)
    return SegmentRankingMetrics(
        task_count=count,
        target_retrieved_count=sum(row[rank_name] is not None for row in rows),
        hr_at_1=hr1,
        hr_at_3=hr3,
        hr_at_5=hr5,
        avg_hr=(hr1 + hr3 + hr5) / 3.0,
        mrr=sum(
            0.0 if row[rank_name] is None else 1.0 / int(row[rank_name]) for row in rows
        )
        / count,
    )


def _task_avg_hr(rank: object) -> float:
    if rank is None:
        return 0.0
    value = int(rank)
    return sum(value <= cutoff for cutoff in (1, 3, 5)) / 3.0


def build_lambdamart_validation_diagnostics(
    sources: LambdaMARTDiagnosticSources,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> LambdaMARTValidationDiagnostics:
    """Compare frozen rankings without refitting or reading Legacy Test."""

    paths = {
        "logistic": _required(sources.logistic_predictions, "Logistic predictions"),
        "lambdamart": _required(
            sources.lambdamart_predictions, "LambdaMART predictions"
        ),
        "contexts": _required(sources.validation_contexts, "Validation contexts"),
        "truth": _required(sources.validation_ground_truth, "Validation ground truth"),
        "reviews": _required(sources.reviews, "Reviews"),
        "interactions": _required(sources.interactions, "Interactions"),
        "businesses": _required(sources.businesses, "Businesses"),
    }
    query = """
        WITH selected_context AS (
            SELECT task_id, user_id, cutoff_time, history_count
            FROM read_parquet(?)
            WHERE split = 'validation'
        ), truth AS (
            SELECT label.task_id, label.target_business_id
            FROM read_parquet(?) AS label
            JOIN selected_context USING (task_id)
        ), logistic_target AS (
            SELECT prediction.task_id, prediction.rank AS logistic_rank
            FROM read_parquet(?) AS prediction
            JOIN truth
              ON truth.task_id = prediction.task_id
             AND truth.target_business_id = prediction.business_id
        ), lambdamart_target AS (
            SELECT prediction.task_id, prediction.rank AS lambdamart_rank
            FROM read_parquet(?) AS prediction
            JOIN truth
              ON truth.task_id = prediction.task_id
             AND truth.target_business_id = prediction.business_id
        ), base AS (
            SELECT context.*, truth.target_business_id,
                   logistic_target.logistic_rank,
                   lambdamart_target.lambdamart_rank,
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
                   ) AS target_in_history
            FROM selected_context AS context
            JOIN truth USING (task_id)
            LEFT JOIN logistic_target USING (task_id)
            LEFT JOIN lambdamart_target USING (task_id)
        ), history_category AS (
            SELECT DISTINCT base.task_id, category.value AS category
            FROM base
            JOIN read_parquet(?) AS interaction
              ON interaction.user_id = base.user_id
             AND interaction.date < base.cutoff_time
            JOIN read_parquet(?) AS business
              ON business.business_id = interaction.business_id
            CROSS JOIN UNNEST(business.categories) AS category(value)
            WHERE category.value NOT IN (
                'Restaurants', 'Food', 'Nightlife', 'Shopping'
            )
        ), target_category AS (
            SELECT DISTINCT truth.task_id, category.value AS category
            FROM truth
            JOIN read_parquet(?) AS business
              ON business.business_id = truth.target_business_id
            CROSS JOIN UNNEST(business.categories) AS category(value)
            WHERE category.value NOT IN (
                'Restaurants', 'Food', 'Nightlife', 'Shopping'
            )
        ), category_status AS (
            SELECT target.task_id,
                   count(history.category) > 0 AS category_seen
            FROM target_category AS target
            LEFT JOIN history_category AS history
              ON history.task_id = target.task_id
             AND history.category = target.category
            GROUP BY target.task_id
        )
        SELECT base.task_id, base.user_id, base.history_count,
               base.logistic_rank, base.lambdamart_rank,
               coalesce(category_status.category_seen, false) AS category_seen
        FROM base
        LEFT JOIN category_status USING (task_id)
        WHERE base.catalog_eligible AND NOT base.target_in_history
        ORDER BY base.task_id
    """
    with duckdb.connect() as connection:
        table = connection.execute(
            query,
            [
                str(paths["contexts"]),
                str(paths["truth"]),
                str(paths["logistic"]),
                str(paths["lambdamart"]),
                str(paths["reviews"]),
                str(paths["interactions"]),
                str(paths["interactions"]),
                str(paths["businesses"]),
                str(paths["businesses"]),
            ],
        ).to_arrow_table()
    rows = table.to_pylist()
    if not rows:
        raise ValueError("Validation diagnostic population is empty")

    def history_bucket(count: int) -> str:
        if count <= 15:
            return "8-15"
        if count <= 30:
            return "16-30"
        if count <= 60:
            return "31-60"
        return "61+"

    comparisons: list[SegmentRankingComparison] = []
    segment_specs: list[
        tuple[Literal["history_count", "category_familiarity"], str, object]
    ] = []
    for value in ("8-15", "16-30", "31-60", "61+"):
        segment_specs.append(
            (
                "history_count",
                value,
                lambda row, selected=value: (
                    history_bucket(int(row["history_count"])) == selected
                ),
            )
        )
    for value, seen in (("seen", True), ("unseen", False)):
        segment_specs.append(
            (
                "category_familiarity",
                value,
                lambda row, selected=seen: bool(row["category_seen"]) == selected,
            )
        )
    for dimension, value, predicate in segment_specs:
        selected_rows = [row for row in rows if predicate(row)]
        if not selected_rows:
            continue
        logistic = _metrics(selected_rows, "logistic_rank")
        lambdamart = _metrics(selected_rows, "lambdamart_rank")
        comparisons.append(
            SegmentRankingComparison(
                dimension=dimension,
                value=value,
                logistic=logistic,
                lambdamart=lambdamart,
                avg_hr_delta=lambdamart.avg_hr - logistic.avg_hr,
                mrr_delta=lambdamart.mrr - logistic.mrr,
            )
        )

    by_user: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        by_user.setdefault(str(row["user_id"]), []).append(
            (
                _task_avg_hr(row["logistic_rank"]),
                _task_avg_hr(row["lambdamart_rank"]),
            )
        )
    user_ids = sorted(by_user)
    logistic_user = np.asarray(
        [np.mean([pair[0] for pair in by_user[user]]) for user in user_ids]
    )
    lambdamart_user = np.asarray(
        [np.mean([pair[1] for pair in by_user[user]]) for user in user_ids]
    )
    return LambdaMARTValidationDiagnostics(
        experiment_name="Hybrid V2-B segmented validation diagnostics",
        population="catalog_eligible_and_not_previously_visited",
        category_definition="fine_category_overlap_before_cutoff",
        segments=comparisons,
        paired_user_avg_hr=paired_user_bootstrap(
            logistic_user,
            lambdamart_user,
            samples=bootstrap_samples,
            confidence_level=0.95,
            seed=random_seed,
        ),
    )


def write_lambdamart_validation_diagnostics(
    path: str | Path,
    report: LambdaMARTValidationDiagnostics,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
