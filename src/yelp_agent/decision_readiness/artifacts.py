"""Hash-verified frozen confidence-calibrator artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
from pydantic import Field

from yelp_agent.models import StrictModel

from .calibration import FrozenConfidenceCalibrator
from .schema import (
    CALIBRATION_FEATURE_NAMES,
    CalibrationMetrics,
    CalibratorKind,
    UncertaintyThresholds,
)


class FrozenConfidenceManifest(StrictModel):
    artifact_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    selected_calibrator: CalibratorKind
    feature_names: list[str]
    uncertainty_thresholds: UncertaintyThresholds
    candidate_metrics: dict[CalibratorKind, CalibrationMetrics]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_task_count: int = Field(ge=1)
    top1_correct_count: int = Field(ge=1)
    target_retrieved_count: int = Field(ge=1)
    fold_counts: dict[str, int]
    confidence_target: str = Field(min_length=1)
    query_aware_confidence_calibrated: bool = False
    test_data_used_for_fit: bool = False
    test_data_used_for_selection: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_frozen_confidence_calibrator(
    root: str | Path,
    calibrator: FrozenConfidenceCalibrator,
    manifest: FrozenConfidenceManifest,
) -> Path:
    """Publish model and manifest together, never as a mixed partial run."""

    artifact_root = Path(root)
    if artifact_root.exists():
        raise FileExistsError(f"Confidence artifact already exists: {artifact_root}")
    if tuple(manifest.feature_names) != CALIBRATION_FEATURE_NAMES:
        raise ValueError("Confidence manifest feature order is invalid")
    if manifest.selected_calibrator != calibrator.model.kind:
        raise ValueError("Confidence manifest and model kind disagree")
    if manifest.uncertainty_thresholds != calibrator.thresholds:
        raise ValueError("Confidence manifest and runtime thresholds disagree")
    partial = artifact_root.with_name(artifact_root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    model_path = partial / "model.joblib"
    manifest_path = partial / "manifest.json"
    try:
        joblib.dump(calibrator, model_path)
        model_hash = sha256_file(model_path)
        finalized = manifest.model_copy(update={"model_sha256": model_hash})
        manifest_path.write_text(
            finalized.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, artifact_root)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return artifact_root


def load_frozen_confidence_calibrator(
    root: str | Path,
) -> tuple[FrozenConfidenceCalibrator, FrozenConfidenceManifest]:
    artifact_root = Path(root)
    model_path = artifact_root / "model.joblib"
    manifest_path = artifact_root / "manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen confidence artifact is incomplete: {artifact_root}")
    manifest = FrozenConfidenceManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if sha256_file(model_path) != manifest.model_sha256:
        raise ValueError("Frozen confidence model hash does not match manifest")
    calibrator = joblib.load(model_path)
    if not isinstance(calibrator, FrozenConfidenceCalibrator):
        raise TypeError("Frozen confidence artifact has an unexpected model type")
    if tuple(manifest.feature_names) != CALIBRATION_FEATURE_NAMES:
        raise ValueError("Frozen confidence feature order does not match runtime")
    if calibrator.model.kind != manifest.selected_calibrator:
        raise ValueError("Frozen confidence model kind does not match manifest")
    if calibrator.thresholds != manifest.uncertainty_thresholds:
        raise ValueError("Frozen confidence thresholds do not match manifest")
    return calibrator, manifest
