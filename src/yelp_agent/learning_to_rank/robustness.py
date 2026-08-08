"""Fixed-configuration user-fold refits and paired validation bootstrap."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.config import EvaluationDataUsageConfig
from yelp_agent.evaluation.data_usage import assign_user_fold
from yelp_agent.experiments import write_json_artifact
from yelp_agent.learning_to_rank.artifacts import load_frozen_hybrid_v2
from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    evaluate_hybrid_v2_ranking,
)
from yelp_agent.learning_to_rank.model import (
    load_pairwise_training_batch,
    train_pairwise_logistic,
)
from yelp_agent.models import StrictModel


class PairedBootstrapResult(StrictModel):
    samples: int = Field(ge=100)
    confidence_level: float = Field(gt=0, lt=1)
    seed: int
    observed_delta: float
    lower_bound: float
    upper_bound: float
    probability_positive: float = Field(ge=0, le=1)


class HybridV2FoldResult(StrictModel):
    fold: int = Field(ge=1)
    training_excluded_fold: int = Field(ge=1)
    hybrid_v1: HybridV2RankingMetrics
    hybrid_v2: HybridV2RankingMetrics
    avg_hr_delta: float
    mrr_delta: float


class HybridV2RobustnessReport(StrictModel):
    experiment_name: Literal["Hybrid V2-A validation robustness audit"]
    method: Literal["fixed_configuration_user_fold_refit"]
    nested_model_selection: Literal[False]
    legacy_test_files_read: Literal[False]
    selected_feature_set: str
    selected_regularization_c: float = Field(gt=0)
    selected_blend_alpha: float = Field(ge=0, le=1)
    folds: list[HybridV2FoldResult]
    fold_avg_hr_mean: float
    fold_avg_hr_standard_deviation: float = Field(ge=0)
    fold_delta_mean: float
    fold_delta_standard_deviation: float = Field(ge=0)
    paired_avg_hr_bootstrap: PairedBootstrapResult


@dataclass(frozen=True, slots=True)
class HybridV2RobustnessSources:
    train_features: Path
    validation_features: Path
    validation_predictions: Path
    validation_contexts: Path
    validation_ground_truth: Path
    reviews: Path
    interactions: Path
    frozen_model_root: Path


def paired_user_bootstrap(
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> PairedBootstrapResult:
    """Resample paired user outcomes, never independent model rows."""

    baseline_values = np.asarray(baseline, dtype=np.float64)
    challenger_values = np.asarray(challenger, dtype=np.float64)
    if (
        baseline_values.ndim != 1
        or challenger_values.shape != baseline_values.shape
        or len(baseline_values) == 0
        or not np.all(np.isfinite(baseline_values))
        or not np.all(np.isfinite(challenger_values))
    ):
        raise ValueError("paired bootstrap inputs must be aligned finite vectors")
    if samples < 100 or not 0 < confidence_level < 1:
        raise ValueError("bootstrap configuration is invalid")
    differences = challenger_values - baseline_values
    generator = np.random.default_rng(seed)
    replicate_means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = generator.integers(0, len(differences), size=len(differences))
        replicate_means[index] = float(np.mean(differences[sampled]))
    tail = (1.0 - confidence_level) / 2.0
    return PairedBootstrapResult(
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
        observed_delta=float(np.mean(differences)),
        lower_bound=float(np.quantile(replicate_means, tail)),
        upper_bound=float(np.quantile(replicate_means, 1.0 - tail)),
        probability_positive=float(np.mean(replicate_means > 0.0)),
    )


def _write_fold_filters(
    contexts_path: Path,
    output_root: Path,
    policy: EvaluationDataUsageConfig,
) -> dict[int, Path]:
    rows = [
        row
        for row in pq.read_table(
            contexts_path,
            columns=["task_id", "split", "user_id"],
        ).to_pylist()
        if row["split"] == "validation"
    ]
    by_fold: dict[int, list[dict[str, str]]] = {
        fold: [] for fold in range(1, policy.cross_validation.folds + 1)
    }
    for row in rows:
        fold = assign_user_fold(str(row["user_id"]), policy)
        by_fold[fold].append({"task_id": str(row["task_id"])})
    output_root.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field("task_id", pa.string(), nullable=False)])
    paths: dict[int, Path] = {}
    for fold, values in by_fold.items():
        if not values:
            raise ValueError(f"Validation fold {fold} is empty")
        path = output_root / f"fold_{fold}_tasks.parquet"
        table = pa.Table.from_pylist(
            sorted(values, key=lambda value: value["task_id"]), schema=schema
        )
        if path.exists():
            if pq.read_table(path).to_pylist() != table.to_pylist():
                raise ValueError(f"Existing validation fold {fold} changed")
        else:
            pq.write_table(table, path, compression="zstd")
        paths[fold] = path
    return paths


def _per_user_avg_hr(
    sources: HybridV2RobustnessSources,
) -> tuple[np.ndarray, np.ndarray]:
    query = """
        WITH selected_context AS (
            SELECT task_id, user_id, cutoff_time
            FROM read_parquet(?) WHERE split = 'validation'
        ), truth AS (
            SELECT label.task_id, label.target_business_id
            FROM read_parquet(?) AS label
            JOIN selected_context USING (task_id)
        ), target_rank AS (
            SELECT prediction.task_id,
                   prediction.v1_rank,
                   prediction.rank AS v2_rank
            FROM read_parquet(?) AS prediction
            JOIN truth
              ON truth.task_id = prediction.task_id
             AND truth.target_business_id = prediction.business_id
        )
        SELECT context.user_id, target_rank.v1_rank, target_rank.v2_rank
        FROM selected_context AS context
        JOIN truth USING (task_id)
        LEFT JOIN target_rank USING (task_id)
        WHERE EXISTS (
            SELECT 1 FROM read_parquet(?) AS review
            WHERE review.business_id = truth.target_business_id
              AND review.date < context.cutoff_time
        )
          AND NOT EXISTS (
            SELECT 1 FROM read_parquet(?) AS interaction
            WHERE interaction.user_id = context.user_id
              AND interaction.business_id = truth.target_business_id
              AND interaction.date < context.cutoff_time
        )
        ORDER BY context.user_id
    """
    with duckdb.connect() as connection:
        rows = connection.execute(
            query,
            [
                str(sources.validation_contexts),
                str(sources.validation_ground_truth),
                str(sources.validation_predictions),
                str(sources.reviews),
                str(sources.interactions),
            ],
        ).fetchall()
    if not rows:
        raise ValueError("Validation robustness population is empty")

    def avg_hr(rank: object) -> float:
        if rank is None:
            return 0.0
        value = int(rank)
        return sum(value <= cutoff for cutoff in (1, 3, 5)) / 3.0

    baseline = np.asarray([avg_hr(row[1]) for row in rows], dtype=np.float64)
    challenger = np.asarray([avg_hr(row[2]) for row in rows], dtype=np.float64)
    return baseline, challenger


def run_hybrid_v2_validation_robustness(
    sources: HybridV2RobustnessSources,
    *,
    fold_output_root: str | Path,
    report_path: str | Path,
    policy: EvaluationDataUsageConfig,
) -> HybridV2RobustnessReport:
    """Audit a frozen configuration without reading or changing Legacy Test."""

    report = Path(report_path)
    if report.is_file():
        return HybridV2RobustnessReport.model_validate_json(
            report.read_text(encoding="utf-8")
        )
    _, manifest = load_frozen_hybrid_v2(sources.frozen_model_root)
    fold_paths = _write_fold_filters(
        sources.validation_contexts, Path(fold_output_root), policy
    )
    fold_results: list[HybridV2FoldResult] = []
    for fold, filter_path in sorted(fold_paths.items()):
        batch = load_pairwise_training_batch(
            sources.train_features,
            feature_names=tuple(manifest.feature_names),
            excluded_fold=fold,
        )
        model = train_pairwise_logistic(
            batch, regularization_c=manifest.selected_regularization_c
        )
        baseline = evaluate_hybrid_v2_ranking(
            features_path=sources.validation_features,
            contexts_path=sources.validation_contexts,
            ground_truth_path=sources.validation_ground_truth,
            reviews_path=sources.reviews,
            interactions_path=sources.interactions,
            split="validation",
            model_name=f"hybrid_v1_fold_{fold}",
            model=None,
            blend_alpha=0.0,
            task_filter_path=filter_path,
        )
        challenger = evaluate_hybrid_v2_ranking(
            features_path=sources.validation_features,
            contexts_path=sources.validation_contexts,
            ground_truth_path=sources.validation_ground_truth,
            reviews_path=sources.reviews,
            interactions_path=sources.interactions,
            split="validation",
            model_name=f"hybrid_v2_fold_{fold}",
            model=model,
            blend_alpha=manifest.selected_blend_alpha,
            task_filter_path=filter_path,
        )
        fold_results.append(
            HybridV2FoldResult(
                fold=fold,
                training_excluded_fold=fold,
                hybrid_v1=baseline,
                hybrid_v2=challenger,
                avg_hr_delta=challenger.avg_hr - baseline.avg_hr,
                mrr_delta=challenger.mrr - baseline.mrr,
            )
        )
    fold_values = [result.hybrid_v2.avg_hr for result in fold_results]
    delta_values = [result.avg_hr_delta for result in fold_results]
    baseline_values, challenger_values = _per_user_avg_hr(sources)
    bootstrap_config = policy.bootstrap
    result = HybridV2RobustnessReport(
        experiment_name="Hybrid V2-A validation robustness audit",
        method="fixed_configuration_user_fold_refit",
        nested_model_selection=False,
        legacy_test_files_read=False,
        selected_feature_set=manifest.selected_feature_set,
        selected_regularization_c=manifest.selected_regularization_c,
        selected_blend_alpha=manifest.selected_blend_alpha,
        folds=fold_results,
        fold_avg_hr_mean=statistics.mean(fold_values),
        fold_avg_hr_standard_deviation=statistics.pstdev(fold_values),
        fold_delta_mean=statistics.mean(delta_values),
        fold_delta_standard_deviation=statistics.pstdev(delta_values),
        paired_avg_hr_bootstrap=paired_user_bootstrap(
            baseline_values,
            challenger_values,
            samples=bootstrap_config.samples,
            confidence_level=bootstrap_config.confidence_level,
            seed=bootstrap_config.seed,
        ),
    )
    write_json_artifact(report, result)
    return result
