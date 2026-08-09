from pathlib import Path

import joblib

from yelp_agent.decision_readiness.artifacts import (
    FrozenConfidenceManifest,
    load_frozen_confidence_calibrator,
    save_frozen_confidence_calibrator,
)
from yelp_agent.decision_readiness.calibration import (
    FrozenConfidenceCalibrator,
    IsotonicConfidenceModel,
)
from yelp_agent.decision_readiness.schema import (
    CALIBRATION_FEATURE_NAMES,
    CalibrationMetrics,
    CoverageRiskPoint,
    ReliabilityBin,
    UncertaintyThresholds,
)


def _calibrator() -> FrozenConfidenceCalibrator:
    import numpy as np

    return FrozenConfidenceCalibrator(
        model=IsotonicConfidenceModel(
            kind="isotonic",
            feature_names=("blend_score_margin",),
            x_thresholds=np.asarray([0.0, 1.0]),
            y_thresholds=np.asarray([0.1, 0.8]),
        ),
        thresholds=UncertaintyThresholds(
            sparse_history_log_max=2.0,
            unseen_category_min=0.5,
            feature_disagreement_min=0.5,
            small_top_margin_max=0.1,
            weak_collaborative_support_max=0.1,
            low_profile_reliability_max=0.4,
        ),
        version="1.0.0",
    )


def _metrics() -> CalibrationMetrics:
    return CalibrationMetrics(
        method="isotonic",
        task_count=10,
        positive_count=2,
        positive_rate=0.2,
        brier_score=0.1,
        expected_calibration_error=0.05,
        reliability_bins=[
            ReliabilityBin(
                lower_bound=0,
                upper_bound=1,
                task_count=10,
                mean_confidence=0.2,
                observed_accuracy=0.2,
            )
        ],
        coverage_risk=[
            CoverageRiskPoint(
                requested_coverage=1,
                actual_coverage=1,
                selected_task_count=10,
                minimum_confidence=0.1,
                observed_accuracy=0.2,
                risk=0.8,
            )
        ],
    )


def test_frozen_calibrator_detects_model_tampering(tmp_path: Path) -> None:
    calibrator = _calibrator()
    manifest = FrozenConfidenceManifest(
        artifact_name="test",
        model_version="1.0.0",
        selected_calibrator="isotonic",
        feature_names=list(CALIBRATION_FEATURE_NAMES),
        uncertainty_thresholds=calibrator.thresholds,
        candidate_metrics={"isotonic": _metrics()},
        source_sha256={"source": "a" * 64},
        configuration_sha256="b" * 64,
        model_sha256="0" * 64,
        training_task_count=10,
        top1_correct_count=2,
        target_retrieved_count=5,
        fold_counts={"1": 2, "2": 2, "3": 2, "4": 2, "5": 2},
        confidence_target="hybrid_v2_b_next_business_top1",
    )
    root = tmp_path / "frozen"

    save_frozen_confidence_calibrator(root, calibrator, manifest)
    loaded, loaded_manifest = load_frozen_confidence_calibrator(root)

    assert loaded.model.kind == "isotonic"
    assert loaded_manifest.model_sha256 != "0" * 64

    joblib.dump({"tampered": True}, root / "model.joblib")
    try:
        load_frozen_confidence_calibrator(root)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("tampered confidence model should be rejected")
