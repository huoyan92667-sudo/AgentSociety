from __future__ import annotations

from yelp_agent.recommendation_v2.review_evidence.qwen_five_class import (
    LabelBatch,
    QwenEvidenceInput,
    QwenFiveClassEvidenceJudge,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
)


class FakeLabelClassifier:
    batch_size = 2

    def __init__(self, labels: list[str]) -> None:
        self._labels = iter(labels)

    def classify(self, prompts) -> LabelBatch:
        labels = tuple(next(self._labels) for _ in prompts)
        return LabelBatch(
            labels=labels,
            input_token_count=10 * len(prompts),
            truncated_candidate_count=0,
            latency_ms=1.0,
        )


def _requirement() -> PreferenceSearchDescription:
    return PreferenceSearchDescription(
        requirement_id="authentic",
        requirement_text="地道正宗",
        kind="long_tail",
        priority=1,
        preference_strength=100,
        positive_descriptions=["authentic Szechuan", "regional flavors"],
        negative_descriptions=["not authentic Szechuan", "Americanized flavors"],
    )


def test_qwen_judge_maps_all_five_allowed_labels_and_batches() -> None:
    candidates = [
        QwenEvidenceInput(
            review_id=f"review-{index}",
            business_id="business-1",
            segment_text=f"segment {index}",
        )
        for index in range(5)
    ]

    result = QwenFiveClassEvidenceJudge(
        FakeLabelClassifier(["A", "B", "C", "D", "E"])
    ).judge(_requirement(), candidates)

    assert [item.label for item in result.judgments] == [
        "irrelevant",
        "positive",
        "negative",
        "mixed",
        "ambiguous",
    ]
    assert result.metrics.candidate_count == 5
    assert result.metrics.batch_count == 3
    assert result.metrics.input_token_count == 50


def test_qwen_judge_rejects_output_outside_five_labels() -> None:
    candidate = QwenEvidenceInput(
        review_id="review-1",
        business_id="business-1",
        segment_text="a review segment",
    )

    try:
        QwenFiveClassEvidenceJudge(FakeLabelClassifier(["Z"])).judge(
            _requirement(), [candidate]
        )
    except RuntimeError as exc:
        assert "unsupported label" in str(exc)
    else:
        raise AssertionError("unsupported Qwen label must fail")
