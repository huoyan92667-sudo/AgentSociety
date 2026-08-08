"""Validation-only model selection and leave-one-signal-family-out ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from yelp_agent.config import HybridV2Config
from yelp_agent.learning_to_rank.evaluation import (
    HybridV2RankingMetrics,
    evaluate_hybrid_v2_ranking,
)
from yelp_agent.learning_to_rank.features import (
    ALL_FEATURE_NAMES,
    BASE_FEATURES,
    BUSINESS_PROFILE_FEATURES,
    ITEM_KNN_FEATURES,
    REVIEW_ASPECT_FEATURES,
    USER_PROFILE_FEATURES,
)
from yelp_agent.learning_to_rank.model import (
    PairwiseLogisticModel,
    PairwiseTrainingBatch,
    load_pairwise_training_batch,
    train_pairwise_logistic,
)
from yelp_agent.models import StrictModel

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "base_only": BASE_FEATURES,
    "without_item_knn": tuple(
        name for name in ALL_FEATURE_NAMES if name not in ITEM_KNN_FEATURES
    ),
    "without_user_profile": tuple(
        name for name in ALL_FEATURE_NAMES if name not in USER_PROFILE_FEATURES
    ),
    "without_business_profile": tuple(
        name for name in ALL_FEATURE_NAMES if name not in BUSINESS_PROFILE_FEATURES
    ),
    "without_review_aspect": tuple(
        name for name in ALL_FEATURE_NAMES if name not in REVIEW_ASPECT_FEATURES
    ),
    "full": ALL_FEATURE_NAMES,
}


class HybridV2SelectionTrial(StrictModel):
    stage: str = Field(min_length=1)
    feature_set: str = Field(min_length=1)
    regularization_c: float = Field(gt=0)
    blend_alpha: float = Field(ge=0, le=1)
    metrics: HybridV2RankingMetrics


@dataclass(frozen=True, slots=True)
class HybridV2SelectionSources:
    train_features: Path
    validation_features: Path
    validation_contexts: Path
    validation_ground_truth: Path
    reviews: Path
    interactions: Path


@dataclass(frozen=True, slots=True)
class HybridV2ModelSelection:
    model: PairwiseLogisticModel
    selected_feature_set: str
    selected_regularization_c: float
    selected_blend_alpha: float
    baseline_metrics: HybridV2RankingMetrics
    selected_metrics: HybridV2RankingMetrics
    c_trials: tuple[HybridV2SelectionTrial, ...]
    alpha_trials: tuple[HybridV2SelectionTrial, ...]
    ablation_metrics: dict[str, HybridV2RankingMetrics]


def _selection_key(metrics: HybridV2RankingMetrics) -> tuple[float, float, float]:
    return (metrics.avg_hr, metrics.mrr, metrics.hr_at_1)


def _subset_batch(
    batch: PairwiseTrainingBatch,
    feature_names: tuple[str, ...],
) -> PairwiseTrainingBatch:
    positions = [batch.feature_names.index(name) for name in feature_names]
    return PairwiseTrainingBatch(
        feature_names=feature_names,
        positive_features=batch.positive_features[:, positions],
        negative_features=batch.negative_features[:, positions],
        sample_weights=batch.sample_weights,
    )


def _evaluate(
    sources: HybridV2SelectionSources,
    *,
    name: str,
    model: PairwiseLogisticModel | None,
    alpha: float,
) -> HybridV2RankingMetrics:
    return evaluate_hybrid_v2_ranking(
        features_path=sources.validation_features,
        contexts_path=sources.validation_contexts,
        ground_truth_path=sources.validation_ground_truth,
        reviews_path=sources.reviews,
        interactions_path=sources.interactions,
        split="validation",
        model_name=name,
        model=model,
        blend_alpha=alpha,
    )


def select_hybrid_v2_model(
    sources: HybridV2SelectionSources,
    config: HybridV2Config,
) -> HybridV2ModelSelection:
    """Fit on rolling train only; use validation for the frozen small search."""

    full_batch = load_pairwise_training_batch(
        sources.train_features, feature_names=ALL_FEATURE_NAMES
    )
    baseline = _evaluate(sources, name="hybrid_v1", model=None, alpha=0.0)
    c_trials: list[HybridV2SelectionTrial] = []
    c_models: dict[float, PairwiseLogisticModel] = {}
    for value in config.regularization_c_candidates:
        model = train_pairwise_logistic(full_batch, regularization_c=float(value))
        metrics = _evaluate(
            sources,
            name=f"hybrid_v2_full_c_{value:g}",
            model=model,
            alpha=1.0,
        )
        c_models[float(value)] = model
        c_trials.append(
            HybridV2SelectionTrial(
                stage="regularization",
                feature_set="full",
                regularization_c=float(value),
                blend_alpha=1.0,
                metrics=metrics,
            )
        )
    selected_c_trial = max(
        c_trials,
        key=lambda trial: (*_selection_key(trial.metrics), -trial.regularization_c),
    )
    selected_c = selected_c_trial.regularization_c
    selected_model = c_models[selected_c]

    alpha_trials: list[HybridV2SelectionTrial] = []
    for alpha in config.blend_alphas:
        metrics = _evaluate(
            sources,
            name=f"hybrid_v2_full_alpha_{alpha:g}",
            model=selected_model,
            alpha=float(alpha),
        )
        alpha_trials.append(
            HybridV2SelectionTrial(
                stage="conservative_blend",
                feature_set="full",
                regularization_c=selected_c,
                blend_alpha=float(alpha),
                metrics=metrics,
            )
        )
    selected_alpha_trial = max(
        alpha_trials,
        key=lambda trial: (*_selection_key(trial.metrics), -trial.blend_alpha),
    )
    selected_alpha = selected_alpha_trial.blend_alpha
    selected_metrics = selected_alpha_trial.metrics

    ablations: dict[str, HybridV2RankingMetrics] = {}
    ablation_models: dict[str, PairwiseLogisticModel] = {}
    for name, feature_names in FEATURE_SETS.items():
        if name == "full":
            ablations[name] = selected_metrics
            ablation_models[name] = selected_model
            continue
        model = train_pairwise_logistic(
            _subset_batch(full_batch, feature_names),
            regularization_c=selected_c,
        )
        ablations[name] = _evaluate(
            sources,
            name=f"hybrid_v2_{name}",
            model=model,
            alpha=selected_alpha,
        )
        ablation_models[name] = model
    selected_feature_set = max(
        ablations,
        key=lambda name: (
            *_selection_key(ablations[name]),
            int(name == "full"),
            name,
        ),
    )
    selected_model = ablation_models[selected_feature_set]
    if selected_feature_set != "full":
        variant_alpha_trials: list[HybridV2SelectionTrial] = []
        for alpha in config.blend_alphas:
            metrics = _evaluate(
                sources,
                name=f"hybrid_v2_{selected_feature_set}_alpha_{alpha:g}",
                model=selected_model,
                alpha=float(alpha),
            )
            variant_alpha_trials.append(
                HybridV2SelectionTrial(
                    stage="selected_feature_set_blend",
                    feature_set=selected_feature_set,
                    regularization_c=selected_c,
                    blend_alpha=float(alpha),
                    metrics=metrics,
                )
            )
        alpha_trials.extend(variant_alpha_trials)
        selected_alpha_trial = max(
            variant_alpha_trials,
            key=lambda trial: (
                *_selection_key(trial.metrics),
                -trial.blend_alpha,
            ),
        )
        selected_alpha = selected_alpha_trial.blend_alpha
        selected_metrics = selected_alpha_trial.metrics
    return HybridV2ModelSelection(
        model=selected_model,
        selected_feature_set=selected_feature_set,
        selected_regularization_c=selected_c,
        selected_blend_alpha=selected_alpha,
        baseline_metrics=baseline,
        selected_metrics=selected_metrics,
        c_trials=tuple(c_trials),
        alpha_trials=tuple(alpha_trials),
        ablation_metrics=ablations,
    )
