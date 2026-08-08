from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yelp_agent.learning_to_rank.artifacts import (
    FrozenHybridV2Manifest,
    HybridV2ArtifactError,
    load_frozen_hybrid_v2,
    save_frozen_hybrid_v2,
)
from yelp_agent.learning_to_rank.model import PairwiseLogisticModel


def test_frozen_model_rejects_content_tampering(tmp_path: Path) -> None:
    model = PairwiseLogisticModel(
        feature_names=("a",),
        feature_scale=np.asarray([1.0]),
        coefficients=np.asarray([2.0]),
        regularization_c=1.0,
    )
    manifest = FrozenHybridV2Manifest(
        format_version=1,
        artifact_name="Hybrid V2-A Pairwise Logistic",
        model_version="2.0.0-a",
        feature_version="1.0.0",
        selected_regularization_c=1.0,
        selected_blend_alpha=0.5,
        feature_names=["a"],
        source_sha256={"train_features": "a" * 64},
        configuration_sha256="b" * 64,
        model_sha256="0" * 64,
        validation_summary={"avg_hr": 0.2},
        test_data_used_for_training=False,
        test_data_used_for_selection=False,
    )
    save_frozen_hybrid_v2(tmp_path, model, manifest)

    loaded, loaded_manifest = load_frozen_hybrid_v2(tmp_path)
    assert loaded.feature_names == ("a",)
    assert loaded_manifest.model_sha256 != "0" * 64

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(HybridV2ArtifactError, match="hash"):
        load_frozen_hybrid_v2(tmp_path)
