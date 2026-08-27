from __future__ import annotations

from datetime import UTC, datetime

import pytest

from yelp_agent.recommendation_v2.rag_benchmark.benchmark import (
    _compare,
    _split_reviews,
    _validate_annotation_batch,
)
from yelp_agent.recommendation_v2.rag_benchmark.schema import (
    LabeledReview,
    ReviewAnnotationBatch,
    ReviewAnnotationRecord,
    ReviewEvidenceLabel,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    ReviewSimilarityCandidate,
)


def _review(review_id: str, text: str, business_id: str = "b1") -> ReviewAnnotationRecord:
    return ReviewAnnotationRecord(
        review_id=review_id,
        business_id=business_id,
        review_time=datetime(2021, 1, 1, tzinfo=UTC),
        review_text=text,
    )


def _label(
    review_id: str,
    *,
    kind: str,
    grade: str,
    positive: list[str] | None = None,
    negative: list[str] | None = None,
    ambiguous: str | None = None,
) -> ReviewEvidenceLabel:
    return ReviewEvidenceLabel(
        review_id=review_id,
        business_id="b1",
        label=kind,
        evidence_grade=grade,
        positive_spans=positive or [],
        negative_spans=negative or [],
        ambiguous_span=ambiguous,
        condition_text=None,
    )


def test_split_reviews_preserves_every_review_once() -> None:
    reviews = [_review(f"r{index}", "x" * 2400) for index in range(7)]

    batches = _split_reviews(reviews, max_reviews=3, max_chars=5000)

    assert [len(batch) for batch in batches] == [2, 2, 2, 1]
    assert [item.review_id for batch in batches for item in batch] == [
        item.review_id for item in reviews
    ]


def test_annotation_requires_exact_coverage_and_exact_source_span() -> None:
    reviews = [
        _review("r1", "The food tasted authentically Sichuan."),
        _review("r2", "We only discussed the service."),
    ]
    valid = ReviewAnnotationBatch(
        batch_id="batch-1",
        case_id="case-1",
        labels=[
            _label(
                "r1",
                kind="positive",
                grade="direct",
                positive=["authentically Sichuan"],
            ),
            _label("r2", kind="irrelevant", grade="none"),
        ],
    )

    _validate_annotation_batch(
        valid,
        expected_batch_id="batch-1",
        expected_case_id="case-1",
        reviews=reviews,
    )

    invalid = valid.model_copy(deep=True)
    invalid.labels[0].positive_spans = ["rewritten evidence"]
    with pytest.raises(ValueError, match="exact substring"):
        _validate_annotation_batch(
            invalid,
            expected_batch_id="batch-1",
            expected_case_id="case-1",
            reviews=reviews,
        )


def test_ambiguous_grade_is_normalized_but_direction_is_not_changed() -> None:
    label = ReviewEvidenceLabel.model_validate(
        {
            "review_id": "r1",
            "business_id": "b1",
            "label": "ambiguous",
            "evidence_grade": "weak",
            "positive_spans": [],
            "negative_spans": [],
            "ambiguous_span": "mentions Sichuan without judging authenticity",
            "condition_text": None,
        }
    )

    assert label.label == "ambiguous"
    assert label.evidence_grade == "none"


def test_comparison_uses_all_strict_labels_as_recall_denominator() -> None:
    positive_review = _review("r1", "Authentic Sichuan food.")
    negative_review = _review("r2", "Not authentic Sichuan food.")
    irrelevant_review = _review("r3", "Friendly staff.")
    labels = [
        LabeledReview(
            review=positive_review,
            label=_label(
                "r1",
                kind="positive",
                grade="direct",
                positive=["Authentic Sichuan"],
            ),
            batch_id="batch-1",
        ),
        LabeledReview(
            review=negative_review,
            label=_label(
                "r2",
                kind="negative",
                grade="supporting",
                negative=["Not authentic Sichuan"],
            ),
            batch_id="batch-1",
        ),
        LabeledReview(
            review=irrelevant_review,
            label=_label("r3", kind="irrelevant", grade="none"),
            batch_id="batch-1",
        ),
    ]
    retrieved = {
        "b1": [
            _candidate("r1", "positive", positive_review.review_text),
            _candidate("r3", "positive", irrelevant_review.review_text),
        ]
    }

    comparison = _compare(labels, retrieved)

    assert comparison.strict_relevant_count == 2
    assert comparison.direct_relevant_count == 1
    assert comparison.direct_recall == 1.0
    assert comparison.positive_direct_recall == 1.0
    assert comparison.negative_direct_recall is None
    assert comparison.retrieved_strict_relevant_count == 1
    assert comparison.strict_recall == 0.5
    assert comparison.positive_recall == 1.0
    assert comparison.negative_recall == 0.0
    assert comparison.missed_review_ids == ["r2"]
    assert comparison.false_directional_review_ids == ["r3"]


def _candidate(
    review_id: str,
    direction: str,
    text: str,
) -> ReviewSimilarityCandidate:
    return ReviewSimilarityCandidate(
        review_id=review_id,
        business_id="b1",
        user_id="u1",
        review_time=datetime(2021, 1, 1, tzinfo=UTC),
        stars=4,
        useful=0,
        review_text=text,
        review_text_sha256="a" * 64,
        matched_segment_id="b" * 64,
        matched_segment_text=text,
        positive_similarity=0.7,
        negative_similarity=0.2,
        direction=direction,
    )
