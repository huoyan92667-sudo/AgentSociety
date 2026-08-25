from datetime import UTC, datetime

from yelp_agent.recommendation_v2.review_evidence.segmenter import (
    OverlapSegmentConfig,
    segment_review_with_overlap,
)


def _row(text: str) -> dict[str, object]:
    return {
        "review_id": "review-1",
        "user_id": "user-1",
        "business_id": "business-1",
        "stars": 4.0,
        "useful": 2,
        "text": text,
        "date": datetime(2022, 1, 2, tzinfo=UTC),
    }


def test_complete_sentences_overlap_and_tail_is_kept() -> None:
    first = "The first complete sentence has enough words to retain its meaning."
    second = "The second complete sentence supplies the shared context for both chunks."
    third = "The final complete sentence must never disappear from a long review."
    text = f"{first} {second} {third}"

    built = segment_review_with_overlap(
        _row(text),
        OverlapSegmentConfig(max_chars=150, overlap_sentences=1),
    )

    assert len(built.segments) == 2
    assert built.segments[0].text == f"{first} {second}"
    assert built.segments[1].text == f"{second} {third}"
    assert built.segments[-1].char_end == len(text)
    assert built.split_long_sentence is False


def test_overlong_sentence_is_split_without_dropping_its_tail() -> None:
    text = " ".join(f"word{index:03d}" for index in range(80))

    built = segment_review_with_overlap(
        _row(text),
        OverlapSegmentConfig(max_chars=100, overlap_sentences=1),
    )

    assert len(built.segments) > 5
    assert built.split_long_sentence is True
    assert built.segments[0].char_start == 0
    assert built.segments[-1].char_end == len(text)
    assert all(len(item.text) <= 100 for item in built.segments)
