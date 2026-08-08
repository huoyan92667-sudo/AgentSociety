"""Tamper-evident frozen artifacts for the selected Hybrid V2-A ranker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import joblib
from pydantic import Field, ValidationError, field_validator

from yelp_agent.learning_to_rank.model import PairwiseLogisticModel
from yelp_agent.models import StrictModel


class HybridV2ArtifactError(RuntimeError):
    """Raised when a frozen model artifact is incomplete or modified."""


class FrozenHybridV2Manifest(StrictModel):
    format_version: Literal[1] = 1
    artifact_name: Literal["Hybrid V2-A Pairwise Logistic"]
    model_version: Literal["2.0.0-a"]
    feature_version: Literal["1.0.0"]
    selected_regularization_c: float = Field(gt=0)
    selected_blend_alpha: float = Field(ge=0, le=1)
    selected_feature_set: str = Field(default="full", min_length=1)
    feature_names: list[str]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_summary: dict[str, float]
    regularization_trials: list[dict[str, object]] = Field(default_factory=list)
    blend_trials: list[dict[str, object]] = Field(default_factory=list)
    ablation_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)
    coefficient_report: dict[str, float] = Field(default_factory=dict)
    test_data_used_for_training: Literal[False]
    test_data_used_for_selection: Literal[False]

    @field_validator("feature_names")
    @classmethod
    def validate_feature_names(cls, values: list[str]) -> list[str]:
        if (
            not values
            or len(set(values)) != len(values)
            or any(not value for value in values)
        ):
            raise ValueError("feature_names must be nonempty and unique")
        return values

    @field_validator("source_sha256")
    @classmethod
    def validate_source_hashes(cls, values: dict[str, str]) -> dict[str, str]:
        if not values or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in values.values()
        ):
            raise ValueError("source hashes must be lowercase SHA256 values")
        return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frozen_hybrid_v2(
    artifact_root: str | Path,
    model: PairwiseLogisticModel,
    manifest: FrozenHybridV2Manifest,
) -> FrozenHybridV2Manifest:
    """Atomically freeze a model and replace the placeholder model hash."""

    root = Path(artifact_root)
    model_path = root / "model.joblib"
    manifest_path = root / "manifest.json"
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Frozen Hybrid V2 artifact already exists: {root}")
    if tuple(manifest.feature_names) != model.feature_names:
        raise ValueError("Manifest feature order does not match the model")
    if manifest.selected_regularization_c != model.regularization_c:
        raise ValueError("Manifest regularization does not match the model")
    root.mkdir(parents=True, exist_ok=True)
    model_partial = root / "model.joblib.partial"
    manifest_partial = root / "manifest.json.partial"
    model_partial.unlink(missing_ok=True)
    manifest_partial.unlink(missing_ok=True)
    try:
        joblib.dump(model, model_partial, compress=3)
        frozen = manifest.model_copy(update={"model_sha256": _sha256(model_partial)})
        manifest_partial.write_text(
            json.dumps(
                frozen.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(model_partial, model_path)
        os.replace(manifest_partial, manifest_path)
    except Exception:
        model_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
    return frozen


def load_frozen_hybrid_v2(
    artifact_root: str | Path,
) -> tuple[PairwiseLogisticModel, FrozenHybridV2Manifest]:
    """Load only when content hash, type, feature order, and C all agree."""

    root = Path(artifact_root)
    model_path = root / "model.joblib"
    manifest_path = root / "manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen Hybrid V2 artifact is incomplete: {root}")
    try:
        manifest = FrozenHybridV2Manifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise HybridV2ArtifactError("Frozen Hybrid V2 manifest is invalid") from exc
    if _sha256(model_path) != manifest.model_sha256:
        raise HybridV2ArtifactError("Frozen Hybrid V2 model hash does not match")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise HybridV2ArtifactError("Frozen Hybrid V2 model cannot be loaded") from exc
    if not isinstance(model, PairwiseLogisticModel):
        raise HybridV2ArtifactError("Frozen Hybrid V2 object has an invalid type")
    if (
        model.feature_names != tuple(manifest.feature_names)
        or model.regularization_c != manifest.selected_regularization_c
    ):
        raise HybridV2ArtifactError(
            "Frozen Hybrid V2 model disagrees with its manifest"
        )
    return model, manifest
