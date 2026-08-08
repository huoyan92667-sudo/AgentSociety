"""Tamper-evident frozen artifacts for the Hybrid V2-B challenger."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import joblib
from pydantic import Field, ValidationError, field_validator

from yelp_agent.learning_to_rank.lambdamart import (
    LambdaMARTModel,
    LambdaMARTParameters,
)
from yelp_agent.models import StrictModel


class LambdaMARTArtifactError(RuntimeError):
    """Raised when a frozen nonlinear model is incomplete or modified."""


class FrozenLambdaMARTManifest(StrictModel):
    format_version: Literal[1] = 1
    artifact_name: Literal["Hybrid V2-B LambdaMART"]
    model_version: Literal["2.0.0-b"]
    feature_version: Literal["1.0.0"]
    selected_parameters: LambdaMARTParameters
    selected_blend_alpha: float = Field(ge=0, le=1)
    selected_feature_set: str = Field(min_length=1)
    feature_names: list[str]
    source_sha256: dict[str, str]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_summary: dict[str, float]
    fair_comparison_summary: dict[str, float] = Field(default_factory=dict)
    parameter_trials: list[dict[str, object]] = Field(default_factory=list)
    blend_trials: list[dict[str, object]] = Field(default_factory=list)
    ablation_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)
    feature_importance_gain: dict[str, float] = Field(default_factory=dict)
    feature_importance_split: dict[str, float] = Field(default_factory=dict)
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


def save_frozen_lambdamart(
    artifact_root: str | Path,
    model: LambdaMARTModel,
    manifest: FrozenLambdaMARTManifest,
) -> FrozenLambdaMARTManifest:
    """Atomically freeze one selected challenger and its validation contract."""

    root = Path(artifact_root)
    model_path = root / "model.joblib"
    manifest_path = root / "manifest.json"
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Frozen LambdaMART artifact already exists: {root}")
    if tuple(manifest.feature_names) != model.feature_names:
        raise ValueError("Manifest feature order does not match the model")
    if manifest.selected_parameters != model.parameters:
        raise ValueError("Manifest parameters do not match the model")
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


def load_frozen_lambdamart(
    artifact_root: str | Path,
) -> tuple[LambdaMARTModel, FrozenLambdaMARTManifest]:
    """Load only when hash, type, feature order, and parameters all agree."""

    root = Path(artifact_root)
    model_path = root / "model.joblib"
    manifest_path = root / "manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen LambdaMART artifact is incomplete: {root}")
    try:
        manifest = FrozenLambdaMARTManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise LambdaMARTArtifactError("Frozen LambdaMART manifest is invalid") from exc
    if _sha256(model_path) != manifest.model_sha256:
        raise LambdaMARTArtifactError("Frozen LambdaMART model hash does not match")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise LambdaMARTArtifactError(
            "Frozen LambdaMART model cannot be loaded"
        ) from exc
    if not isinstance(model, LambdaMARTModel):
        raise LambdaMARTArtifactError("Frozen LambdaMART object has an invalid type")
    if (
        model.feature_names != tuple(manifest.feature_names)
        or model.parameters != manifest.selected_parameters
    ):
        raise LambdaMARTArtifactError(
            "Frozen LambdaMART model disagrees with its manifest"
        )
    return model, manifest
