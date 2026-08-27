from __future__ import annotations

from yelp_agent.recommendation_v2.rag_benchmark.retrieval_experiment import (
    _slice_requirement,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
)


def _requirement() -> PreferenceSearchDescription:
    return PreferenceSearchDescription(
        requirement_id="open.authentic_szechuan.test",
        requirement_text="地道川菜",
        kind="long_tail",
        priority=1,
        preference_strength=100,
        positive_descriptions=[f"positive {index}" for index in range(5)],
        negative_descriptions=[f"negative {index}" for index in range(5)],
    )


def test_slice_requirement_can_measure_one_to_five_descriptions() -> None:
    source = _requirement()

    variants = [_slice_requirement(source, count) for count in range(1, 6)]

    assert [len(item.positive_descriptions) for item in variants] == [1, 2, 3, 4, 5]
    assert [len(item.negative_descriptions) for item in variants] == [1, 2, 3, 4, 5]
    assert source.positive_descriptions == [f"positive {index}" for index in range(5)]
