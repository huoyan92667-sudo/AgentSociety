from __future__ import annotations

import numpy as np

from yelp_agent.learning_to_rank.lambdamart import (
    LambdaMARTParameters,
    LambdaMARTTrainingBatch,
    train_lambdamart,
)
from yelp_agent.learning_to_rank.model import PairwiseLogisticModel
from yelp_agent.learning_to_rank.runtime import (
    FrozenHybridV2Ranker,
    FrozenLambdaMARTRanker,
)


def test_frozen_runtime_returns_complete_deterministic_blended_ranking() -> None:
    model = PairwiseLogisticModel(
        feature_names=("preference",),
        feature_scale=np.asarray([1.0]),
        coefficients=np.asarray([1.0]),
        regularization_c=1.0,
    )
    ranker = FrozenHybridV2Ranker(model=model, blend_alpha=0.5)

    ranking = ranker.rank(
        [
            {"business_id": "b", "preference": 0.9, "hybrid_v1_score": 0.1},
            {"business_id": "a", "preference": 0.1, "hybrid_v1_score": 0.5},
            {"business_id": "c", "preference": 0.5, "hybrid_v1_score": 0.9},
        ]
    )

    assert [candidate.business_id for candidate in ranking] == ["c", "b", "a"]
    assert [candidate.rank for candidate in ranking] == [1, 2, 3]
    assert {candidate.business_id for candidate in ranking} == {"a", "b", "c"}


def test_lambdamart_runtime_uses_the_same_complete_ranking_contract() -> None:
    model = train_lambdamart(
        LambdaMARTTrainingBatch(
            feature_names=("signal",),
            features=np.asarray([[1.0], [0.0], [0.9], [0.1]], dtype=np.float64),
            labels=np.asarray([1, 0, 1, 0], dtype=np.int8),
            group_sizes=np.asarray([2, 2], dtype=np.int32),
            sample_weights=np.ones(4, dtype=np.float64),
        ),
        parameters=LambdaMARTParameters(
            name="tiny",
            num_leaves=3,
            learning_rate=0.2,
            num_boost_round=10,
            min_child_samples=1,
            reg_lambda=0.0,
        ),
        random_seed=42,
    )
    ranker = FrozenLambdaMARTRanker(model=model, blend_alpha=1.0)

    ranking = ranker.rank(
        [
            {"business_id": "bad", "signal": 0.0, "hybrid_v1_score": 0.9},
            {"business_id": "good", "signal": 1.0, "hybrid_v1_score": 0.1},
        ]
    )

    assert [candidate.business_id for candidate in ranking] == ["good", "bad"]
