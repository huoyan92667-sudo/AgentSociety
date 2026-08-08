"""Validation-only LambdaMART selection on the frozen Top-500 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from yelp_agent.config import HybridV2BConfig, LambdaMARTTrialConfig
from yelp_agent.learning_to_rank.evaluation import HybridV2RankingMetrics
from yelp_agent.learning_to_rank.lambdamart import (
    LambdaMARTModel,
    LambdaMARTParameters,
    load_lambdamart_training_batch,
    train_lambdamart,
)
from yelp_agent.learning_to_rank.scored_ranking import (
    CandidateScoreWriteResult,
    evaluate_candidate_scores,
    write_candidate_scores,
)
from yelp_agent.learning_to_rank.selection import FEATURE_SETS
from yelp_agent.models import StrictModel


class LambdaMARTSelectionTrial(StrictModel):
    stage: str = Field(min_length=1)
    feature_set: str = Field(min_length=1)
    parameters: LambdaMARTParameters
    blend_alpha: float = Field(ge=0, le=1)
    metrics: HybridV2RankingMetrics


@dataclass(frozen=True, slots=True)
class LambdaMARTSelectionSources:
    train_features: Path
    validation_features: Path
    validation_contexts: Path
    validation_ground_truth: Path
    reviews: Path
    interactions: Path


@dataclass(frozen=True, slots=True)
class LambdaMARTModelSelection:
    model: LambdaMARTModel
    selected_feature_set: str
    selected_parameters: LambdaMARTParameters
    selected_blend_alpha: float
    fair_comparison_metrics: HybridV2RankingMetrics
    selected_metrics: HybridV2RankingMetrics
    parameter_trials: tuple[LambdaMARTSelectionTrial, ...]
    blend_trials: tuple[LambdaMARTSelectionTrial, ...]
    ablation_metrics: dict[str, HybridV2RankingMetrics]
    validation_scores: CandidateScoreWriteResult


def selection_key(metrics: HybridV2RankingMetrics) -> tuple[float, float, float]:
    """Apply the frozen AvgHR, MRR, HR@1 comparison order."""

    return (metrics.avg_hr, metrics.mrr, metrics.hr_at_1)


def _parameters(config: LambdaMARTTrialConfig) -> LambdaMARTParameters:
    return LambdaMARTParameters(**config.model_dump(mode="python"))


def _evaluate(
    sources: LambdaMARTSelectionSources,
    *,
    scores_path: Path,
    model_name: str,
    blend_alpha: float,
) -> HybridV2RankingMetrics:
    return evaluate_candidate_scores(
        scores_path=scores_path,
        contexts_path=sources.validation_contexts,
        ground_truth_path=sources.validation_ground_truth,
        reviews_path=sources.reviews,
        interactions_path=sources.interactions,
        split="validation",
        model_name=model_name,
        blend_alpha=blend_alpha,
    )


def _write_scores(
    sources: LambdaMARTSelectionSources,
    *,
    model: LambdaMARTModel,
    path: Path,
    batch_size: int,
) -> CandidateScoreWriteResult:
    path.unlink(missing_ok=True)
    return write_candidate_scores(
        features_path=sources.validation_features,
        output_path=path,
        scorer=model,
        batch_size=batch_size,
    )


def _fit_feature_set(
    sources: LambdaMARTSelectionSources,
    *,
    feature_set: str,
    parameters: LambdaMARTParameters,
    random_seed: int,
) -> LambdaMARTModel:
    feature_names = FEATURE_SETS[feature_set]
    batch = load_lambdamart_training_batch(
        sources.train_features,
        feature_names=feature_names,
    )
    return train_lambdamart(
        batch,
        parameters=parameters,
        random_seed=random_seed,
    )


def select_lambdamart_model(
    sources: LambdaMARTSelectionSources,
    config: HybridV2BConfig,
    *,
    scratch_root: str | Path,
    final_scores_path: str | Path,
) -> LambdaMARTModelSelection:
    """Fit on rolling Train and use only Validation for bounded selection."""

    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    fair_set = config.fair_comparison_feature_set
    fair_batch = load_lambdamart_training_batch(
        sources.train_features,
        feature_names=FEATURE_SETS[fair_set],
    )
    parameter_trials: list[LambdaMARTSelectionTrial] = []
    parameter_models: list[LambdaMARTModel] = []
    trial_score_path = scratch / "parameter_trial_scores.parquet"
    for trial_config in config.parameter_trials:
        parameters = _parameters(trial_config)
        model = train_lambdamart(
            fair_batch,
            parameters=parameters,
            random_seed=config.random_seed,
        )
        _write_scores(
            sources,
            model=model,
            path=trial_score_path,
            batch_size=config.prediction_batch_size,
        )
        metrics = _evaluate(
            sources,
            scores_path=trial_score_path,
            model_name=f"hybrid_v2_b_{parameters.name}_raw",
            blend_alpha=1.0,
        )
        parameter_models.append(model)
        parameter_trials.append(
            LambdaMARTSelectionTrial(
                stage="parameters",
                feature_set=fair_set,
                parameters=parameters,
                blend_alpha=1.0,
                metrics=metrics,
            )
        )
    trial_score_path.unlink(missing_ok=True)
    selected_parameter_index = max(
        range(len(parameter_trials)),
        key=lambda index: (*selection_key(parameter_trials[index].metrics), -index),
    )
    selected_parameters = parameter_trials[selected_parameter_index].parameters
    fair_model = parameter_models[selected_parameter_index]

    fair_score_path = scratch / "fair_feature_scores.parquet"
    _write_scores(
        sources,
        model=fair_model,
        path=fair_score_path,
        batch_size=config.prediction_batch_size,
    )
    blend_trials: list[LambdaMARTSelectionTrial] = []
    for alpha in config.blend_alphas:
        metrics = _evaluate(
            sources,
            scores_path=fair_score_path,
            model_name=f"hybrid_v2_b_{fair_set}_alpha_{alpha:g}",
            blend_alpha=float(alpha),
        )
        blend_trials.append(
            LambdaMARTSelectionTrial(
                stage="fair_feature_blend",
                feature_set=fair_set,
                parameters=selected_parameters,
                blend_alpha=float(alpha),
                metrics=metrics,
            )
        )
    fair_alpha_trial = max(
        blend_trials,
        key=lambda trial: (*selection_key(trial.metrics), -trial.blend_alpha),
    )
    fair_alpha = fair_alpha_trial.blend_alpha
    fair_metrics = fair_alpha_trial.metrics

    ablation_metrics: dict[str, HybridV2RankingMetrics] = {}
    ablation_models: dict[str, LambdaMARTModel] = {}
    for feature_set in config.ablation_feature_sets:
        if feature_set == fair_set:
            model = fair_model
            score_path = fair_score_path
        else:
            model = _fit_feature_set(
                sources,
                feature_set=feature_set,
                parameters=selected_parameters,
                random_seed=config.random_seed,
            )
            score_path = scratch / "ablation_scores.parquet"
            _write_scores(
                sources,
                model=model,
                path=score_path,
                batch_size=config.prediction_batch_size,
            )
        ablation_metrics[feature_set] = _evaluate(
            sources,
            scores_path=score_path,
            model_name=f"hybrid_v2_b_{feature_set}",
            blend_alpha=fair_alpha,
        )
        ablation_models[feature_set] = model
        if feature_set != fair_set:
            score_path.unlink(missing_ok=True)
    selected_feature_set = max(
        ablation_metrics,
        key=lambda name: (
            *selection_key(ablation_metrics[name]),
            int(name == "full"),
            name,
        ),
    )
    selected_model = ablation_models[selected_feature_set]
    selected_alpha = fair_alpha
    selected_metrics = ablation_metrics[selected_feature_set]
    if selected_feature_set != fair_set:
        selected_score_path = scratch / "selected_feature_scores.parquet"
        _write_scores(
            sources,
            model=selected_model,
            path=selected_score_path,
            batch_size=config.prediction_batch_size,
        )
        variant_trials: list[LambdaMARTSelectionTrial] = []
        for alpha in config.blend_alphas:
            metrics = _evaluate(
                sources,
                scores_path=selected_score_path,
                model_name=(f"hybrid_v2_b_{selected_feature_set}_alpha_{alpha:g}"),
                blend_alpha=float(alpha),
            )
            variant_trials.append(
                LambdaMARTSelectionTrial(
                    stage="selected_feature_blend",
                    feature_set=selected_feature_set,
                    parameters=selected_parameters,
                    blend_alpha=float(alpha),
                    metrics=metrics,
                )
            )
        blend_trials.extend(variant_trials)
        selected_alpha_trial = max(
            variant_trials,
            key=lambda trial: (*selection_key(trial.metrics), -trial.blend_alpha),
        )
        selected_alpha = selected_alpha_trial.blend_alpha
        selected_metrics = selected_alpha_trial.metrics
        selected_score_path.unlink(missing_ok=True)

    fair_score_path.unlink(missing_ok=True)
    final_scores = Path(final_scores_path)
    final_score_result = _write_scores(
        sources,
        model=selected_model,
        path=final_scores,
        batch_size=config.prediction_batch_size,
    )
    return LambdaMARTModelSelection(
        model=selected_model,
        selected_feature_set=selected_feature_set,
        selected_parameters=selected_parameters,
        selected_blend_alpha=selected_alpha,
        fair_comparison_metrics=fair_metrics,
        selected_metrics=selected_metrics,
        parameter_trials=tuple(parameter_trials),
        blend_trials=tuple(blend_trials),
        ablation_metrics=ablation_metrics,
        validation_scores=final_score_result,
    )
