from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yelp_agent.learning_to_rank.evaluation import HybridV2RankingMetrics
from yelp_agent.learning_to_rank.lambdamart import (
    LambdaMARTParameters,
    LambdaMARTTrainingBatch,
    load_lambdamart_training_batch,
    train_lambdamart,
)
from yelp_agent.learning_to_rank.lambdamart_artifacts import (
    FrozenLambdaMARTManifest,
    LambdaMARTArtifactError,
    load_frozen_lambdamart,
    save_frozen_lambdamart,
)
from yelp_agent.learning_to_rank.lambdamart_selection import selection_key


def test_lambdamart_learns_groupwise_preference_and_freezes_feature_order() -> None:
    features = np.asarray(
        [
            [0.95, 0.90],
            [0.20, 0.10],
            [0.10, 0.30],
            [0.90, 0.85],
            [0.30, 0.20],
            [0.15, 0.25],
            [0.85, 0.95],
            [0.25, 0.15],
            [0.20, 0.35],
            [0.92, 0.88],
            [0.35, 0.10],
            [0.10, 0.20],
        ],
        dtype=np.float64,
    )
    batch = LambdaMARTTrainingBatch(
        feature_names=("preference_match", "quality"),
        features=features,
        labels=np.asarray([1, 0, 0] * 4, dtype=np.int8),
        group_sizes=np.asarray([3, 3, 3, 3], dtype=np.int32),
        sample_weights=np.ones(len(features), dtype=np.float64),
    )
    parameters = LambdaMARTParameters(
        name="tiny",
        num_leaves=3,
        learning_rate=0.2,
        num_boost_round=20,
        min_child_samples=1,
        reg_lambda=0.0,
    )

    model = train_lambdamart(batch, parameters=parameters, random_seed=42)
    scores = model.score(features, feature_names=batch.feature_names)

    for start in range(0, len(features), 3):
        assert scores[start] > max(scores[start + 1 : start + 3])
    with pytest.raises(ValueError, match="feature order"):
        model.score(features, feature_names=tuple(reversed(batch.feature_names)))


def test_lambdamart_loader_builds_contiguous_weighted_task_groups(tmp_path) -> None:
    path = tmp_path / "train_features.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t2",
                    "business_id": "n2",
                    "label": False,
                    "sample_weight": 1.0,
                    "a": 0.1,
                    "b": 0.8,
                },
                {
                    "task_id": "t1",
                    "business_id": "p1",
                    "label": True,
                    "sample_weight": 0.5,
                    "a": 0.9,
                    "b": 0.2,
                },
                {
                    "task_id": "t2",
                    "business_id": "p2",
                    "label": True,
                    "sample_weight": 1.0,
                    "a": 0.8,
                    "b": 0.3,
                },
                {
                    "task_id": "t1",
                    "business_id": "n1",
                    "label": False,
                    "sample_weight": 0.5,
                    "a": 0.2,
                    "b": 0.7,
                },
            ]
        ),
        path,
    )

    batch = load_lambdamart_training_batch(path, feature_names=("a", "b"))

    np.testing.assert_array_equal(batch.group_sizes, [2, 2])
    np.testing.assert_array_equal(batch.labels, [1, 0, 1, 0])
    np.testing.assert_allclose(batch.sample_weights, [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(batch.features[0], [0.9, 0.2])


def test_frozen_lambdamart_rejects_model_tampering(tmp_path) -> None:
    batch = LambdaMARTTrainingBatch(
        feature_names=("signal",),
        features=np.asarray([[1.0], [0.0], [0.9], [0.1]], dtype=np.float64),
        labels=np.asarray([1, 0, 1, 0], dtype=np.int8),
        group_sizes=np.asarray([2, 2], dtype=np.int32),
        sample_weights=np.ones(4, dtype=np.float64),
    )
    parameters = LambdaMARTParameters(
        name="tiny",
        num_leaves=3,
        learning_rate=0.2,
        num_boost_round=5,
        min_child_samples=1,
        reg_lambda=0.0,
    )
    model = train_lambdamart(batch, parameters=parameters, random_seed=42)
    manifest = FrozenLambdaMARTManifest(
        artifact_name="Hybrid V2-B LambdaMART",
        model_version="2.0.0-b",
        feature_version="1.0.0",
        selected_parameters=parameters,
        selected_blend_alpha=0.5,
        selected_feature_set="full",
        feature_names=["signal"],
        source_sha256={"train_features": "a" * 64},
        configuration_sha256="b" * 64,
        model_sha256="0" * 64,
        validation_summary={"avg_hr": 0.1},
        test_data_used_for_training=False,
        test_data_used_for_selection=False,
    )

    frozen = save_frozen_lambdamart(tmp_path, model, manifest)
    loaded, loaded_manifest = load_frozen_lambdamart(tmp_path)

    assert loaded.feature_names == ("signal",)
    assert loaded_manifest.model_sha256 == frozen.model_sha256
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(LambdaMARTArtifactError, match="hash"):
        load_frozen_lambdamart(tmp_path)


def test_lambdamart_selection_uses_avghr_then_mrr_then_hr1() -> None:
    common = {
        "model_name": "candidate",
        "split": "validation",
        "blend_alpha": 1.0,
        "task_count": 10,
        "primary_task_count": 10,
        "catalog_ineligible_task_count": 0,
        "target_in_history_task_count": 0,
        "target_retrieved_count": 10,
        "target_not_retrieved_count": 0,
        "hr_at_1": 0.1,
        "hr_at_3": 0.2,
        "hr_at_5": 0.3,
        "avg_hr": 0.2,
        "mrr": 0.25,
        "ndcg_at_5": 0.2,
        "retrieved_target_hr_at_1": 0.1,
        "retrieved_target_hr_at_3": 0.2,
        "retrieved_target_hr_at_5": 0.3,
        "retrieved_target_mrr": 0.25,
        "all_task_hr_at_1": 0.1,
        "all_task_hr_at_3": 0.2,
        "all_task_hr_at_5": 0.3,
    }
    baseline = HybridV2RankingMetrics(**common)
    better_mrr = baseline.model_copy(update={"mrr": 0.26})
    better_avg_hr = baseline.model_copy(update={"avg_hr": 0.21, "mrr": 0.1})

    assert selection_key(better_mrr) > selection_key(baseline)
    assert selection_key(better_avg_hr) > selection_key(better_mrr)
