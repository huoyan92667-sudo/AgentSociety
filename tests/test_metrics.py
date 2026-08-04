import math

import pytest

from yelp_agent.evaluation.metrics import compute_single_positive_metrics


@pytest.mark.parametrize(
    ("target_position", "expected"),
    [
        (
            1,
            {
                "hit_at_1": 1.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "reciprocal_rank": 1.0,
                "ndcg_at_5": 1.0,
            },
        ),
        (
            3,
            {
                "hit_at_1": 0.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "reciprocal_rank": 1 / 3,
                "ndcg_at_5": 1 / math.log2(4),
            },
        ),
        (
            6,
            {
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "reciprocal_rank": 1 / 6,
                "ndcg_at_5": 0.0,
            },
        ),
        (
            20,
            {
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "reciprocal_rank": 1 / 20,
                "ndcg_at_5": 0.0,
            },
        ),
    ],
)
def test_single_positive_ranking_metrics(
    target_position: int,
    expected: dict[str, float],
) -> None:
    ranking = [f"business-{index}" for index in range(1, 21)]

    metrics = compute_single_positive_metrics(
        f"business-{target_position}",
        ranking,
    )

    assert metrics.target_rank == target_position
    for field, value in expected.items():
        assert getattr(metrics, field) == pytest.approx(value)
