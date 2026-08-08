from __future__ import annotations

import numpy as np

from yelp_agent.learning_to_rank.model import PairwiseLogisticModel
from yelp_agent.learning_to_rank.runtime import FrozenHybridV2Ranker


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
