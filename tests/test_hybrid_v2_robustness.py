from __future__ import annotations

import numpy as np

from yelp_agent.learning_to_rank.robustness import paired_user_bootstrap


def test_paired_user_bootstrap_is_deterministic_and_keeps_pairing() -> None:
    baseline = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    challenger = np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float64)

    first = paired_user_bootstrap(
        baseline,
        challenger,
        samples=500,
        confidence_level=0.95,
        seed=42,
    )
    second = paired_user_bootstrap(
        baseline,
        challenger,
        samples=500,
        confidence_level=0.95,
        seed=42,
    )

    assert first == second
    assert first.observed_delta == 0.5
    assert first.probability_positive > 0.9
    assert first.lower_bound <= first.observed_delta <= first.upper_bound
