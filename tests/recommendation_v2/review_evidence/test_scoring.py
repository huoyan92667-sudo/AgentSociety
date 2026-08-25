from datetime import UTC, datetime

import pytest

from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
    ReviewSimilarityCandidate,
)
from yelp_agent.recommendation_v2.review_evidence.scoring import (
    aggregate_business_evidence,
    count_reliability,
)


def _requirement() -> PreferenceSearchDescription:
    return PreferenceSearchDescription(
        requirement_id="open.quiet",
        requirement_text="想要安静",
        kind="long_tail",
        priority=1,
        preference_strength=100,
        positive_descriptions=["quiet room", "easy conversation"],
        negative_descriptions=["very noisy", "need to shout"],
    )


def _candidate(
    review_id: str,
    *,
    positive: float,
    negative: float,
    direction: str,
) -> ReviewSimilarityCandidate:
    text = f"full original review {review_id}"
    return ReviewSimilarityCandidate(
        review_id=review_id,
        business_id="business-1",
        user_id=f"user-{review_id}",
        review_time=datetime(2022, 1, 1, tzinfo=UTC),
        stars=4,
        useful=1,
        review_text=text,
        review_text_sha256="a" * 64,
        matched_segment_id="b" * 64,
        matched_segment_text=text,
        positive_similarity=positive,
        negative_similarity=negative,
        direction=direction,
    )


def test_no_evidence_is_neutral() -> None:
    result = aggregate_business_evidence(
        _requirement(),
        "business-1",
        [],
        reference_time=datetime(2023, 1, 1, tzinfo=UTC),
    )

    assert result.evidence_score == 0.5
    assert result.positive_component == 0
    assert result.negative_component == 0


def test_positive_and_negative_are_scored_separately() -> None:
    result = aggregate_business_evidence(
        _requirement(),
        "business-1",
        [
            _candidate("positive", positive=0.92, negative=0.2, direction="positive"),
            _candidate("negative", positive=0.1, negative=0.9, direction="negative"),
            _candidate("ambiguous", positive=0.75, negative=0.73, direction="ambiguous"),
        ],
        reference_time=datetime(2023, 1, 1, tzinfo=UTC),
    )

    assert len(result.positive_evidence) == 1
    assert len(result.negative_evidence) == 1
    assert result.ambiguous_review_count == 1
    assert 0 <= result.evidence_score <= 1
    assert count_reliability(1) == pytest.approx(
        0.3868528072345416,
    )
    assert count_reliability(5) == 1
