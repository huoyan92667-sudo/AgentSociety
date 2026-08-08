from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yelp_agent.learning_to_rank import (
    PairwiseTrainingBatch,
    load_pairwise_training_batch,
    train_pairwise_logistic,
)


def test_pairwise_logistic_learns_to_put_preferred_business_first() -> None:
    batch = PairwiseTrainingBatch(
        feature_names=("preference_match", "negative_conflict"),
        positive_features=np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.1]], dtype=np.float64
        ),
        negative_features=np.asarray(
            [[0.2, 0.8], [0.3, 0.9], [0.1, 0.7]], dtype=np.float64
        ),
        sample_weights=np.ones(3, dtype=np.float64),
    )

    model = train_pairwise_logistic(batch, regularization_c=1.0)

    scores = model.score(
        np.asarray([[0.85, 0.1], [0.2, 0.85]], dtype=np.float64),
        feature_names=batch.feature_names,
    )
    assert scores[0] > scores[1]
    assert model.feature_names == batch.feature_names


def test_pairwise_model_rejects_reordered_features() -> None:
    batch = PairwiseTrainingBatch(
        feature_names=("a", "b"),
        positive_features=np.asarray([[1.0, 0.0]], dtype=np.float64),
        negative_features=np.asarray([[0.0, 1.0]], dtype=np.float64),
        sample_weights=np.ones(1, dtype=np.float64),
    )
    model = train_pairwise_logistic(batch, regularization_c=1.0)

    with np.testing.assert_raises_regex(ValueError, "feature order"):
        model.score(
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            feature_names=("b", "a"),
        )


def test_pairwise_loader_aligns_each_negative_with_its_task_target(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "t1",
                    "business_id": "p1",
                    "label": True,
                    "sample_weight": 0.5,
                    "a": 1.0,
                    "b": 0.0,
                },
                {
                    "task_id": "t1",
                    "business_id": "n1",
                    "label": False,
                    "sample_weight": 0.5,
                    "a": 0.0,
                    "b": 1.0,
                },
                {
                    "task_id": "t1",
                    "business_id": "n2",
                    "label": False,
                    "sample_weight": 0.5,
                    "a": 0.2,
                    "b": 0.8,
                },
                {
                    "task_id": "t2",
                    "business_id": "p2",
                    "label": True,
                    "sample_weight": 1.0,
                    "a": 0.9,
                    "b": 0.1,
                },
                {
                    "task_id": "t2",
                    "business_id": "n3",
                    "label": False,
                    "sample_weight": 1.0,
                    "a": 0.1,
                    "b": 0.9,
                },
                {
                    "task_id": "t2",
                    "business_id": "n4",
                    "label": False,
                    "sample_weight": 1.0,
                    "a": 0.0,
                    "b": 1.0,
                },
            ]
        ),
        path,
    )

    batch = load_pairwise_training_batch(path, feature_names=("a", "b"))

    assert batch.positive_features.shape == (4, 2)
    np.testing.assert_allclose(batch.sample_weights, [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(batch.positive_features[0], [1.0, 0.0])
    np.testing.assert_allclose(batch.negative_features[1], [0.2, 0.8])
