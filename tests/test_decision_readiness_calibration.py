import numpy as np

from yelp_agent.decision_readiness.calibration import (
    CalibrationBatch,
    FrozenConfidenceCalibrator,
    derive_uncertainty_thresholds,
    select_confidence_calibrator,
)
from yelp_agent.decision_readiness.schema import (
    CALIBRATION_FEATURE_NAMES,
    RankingSignals,
)


def _synthetic_batch() -> CalibrationBatch:
    rng = np.random.default_rng(42)
    row_count = 250
    features = rng.normal(size=(row_count, len(CALIBRATION_FEATURE_NAMES)))
    for name in (
        "model_v1_rank_disagreement",
        "component_disagreement",
        "top1_item_knn_missing",
        "user_profile_reliability",
        "top1_user_category_novelty",
    ):
        features[:, CALIBRATION_FEATURE_NAMES.index(name)] = rng.uniform(
            0,
            1,
            row_count,
        )
    features[:, CALIBRATION_FEATURE_NAMES.index("user_history_length_log")] = rng.uniform(
        1,
        5,
        row_count,
    )
    features[:, CALIBRATION_FEATURE_NAMES.index("top1_route_coverage")] = rng.uniform(
        1,
        5,
        row_count,
    )
    margin = features[:, CALIBRATION_FEATURE_NAMES.index("blend_score_margin")]
    reliability = features[
        :, CALIBRATION_FEATURE_NAMES.index("user_profile_reliability")
    ]
    latent = 1.4 * margin + 1.2 * reliability - 0.4
    probabilities = 1.0 / (1.0 + np.exp(-latent))
    labels = (rng.uniform(size=row_count) < probabilities).astype(np.int8)
    return CalibrationBatch(
        task_ids=tuple(f"task-{index:03d}" for index in range(row_count)),
        folds=np.asarray([(index % 5) + 1 for index in range(row_count)]),
        features=features,
        labels=labels,
    )


def test_calibrator_selection_is_deterministic_and_returns_oof_metrics() -> None:
    batch = _synthetic_batch()

    first = select_confidence_calibrator(
        batch,
        random_seed=42,
        maximum_iterations=1000,
        ece_bin_count=10,
        coverage_points=(0.25, 0.5, 1.0),
    )
    second = select_confidence_calibrator(
        batch,
        random_seed=42,
        maximum_iterations=1000,
        ece_bin_count=10,
        coverage_points=(0.25, 0.5, 1.0),
    )

    assert set(first.candidate_metrics) == {"logistic", "isotonic"}
    assert first.selected_kind == second.selected_kind
    assert first.candidate_metrics == second.candidate_metrics
    assert all(len(metric.reliability_bins) == 10 for metric in first.candidate_metrics.values())
    assert all(len(metric.coverage_risk) == 3 for metric in first.candidate_metrics.values())


def test_frozen_calibrator_returns_probability_and_reason_codes() -> None:
    batch = _synthetic_batch()
    selection = select_confidence_calibrator(
        batch,
        random_seed=42,
        maximum_iterations=1000,
        ece_bin_count=5,
        coverage_points=(1.0,),
    )
    thresholds = derive_uncertainty_thresholds(
        batch,
        sparse_history_count_max=8,
        unseen_category_min=0.5,
        feature_disagreement_min=0.5,
        small_top_margin_quantile=0.25,
        weak_collaborative_support_quantile=0.25,
        low_profile_reliability_max=0.4,
    )
    calibrator = FrozenConfidenceCalibrator(
        model=selection.selected_model,
        thresholds=thresholds,
        version="1.0.0",
    )
    signals = RankingSignals(
        top1_blend_score=0.1,
        blend_score_margin=thresholds.small_top_margin_max - 0.1,
        model_score_margin=0.0,
        hybrid_v1_score_margin=0.0,
        model_v1_rank_disagreement=0.8,
        component_disagreement=0.8,
        top1_item_knn_positive_score=0.0,
        top1_item_knn_missing=1.0,
        user_history_length_log=0.5,
        user_profile_reliability=0.2,
        top1_user_category_novelty=0.9,
        top1_route_coverage=1.0,
    )

    estimate = calibrator.estimate(signals)

    assert 0 <= estimate.probability_top1_correct <= 1
    assert estimate.uncertainty_reasons == [
        "sparse_history",
        "unseen_category",
        "feature_disagreement",
        "small_top_margin",
        "weak_collaborative_support",
        "low_profile_reliability",
    ]
