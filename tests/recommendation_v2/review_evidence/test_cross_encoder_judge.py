from __future__ import annotations

from datetime import UTC, datetime

from yelp_agent.cross_encoder.scorer import ScoredPairBatch
from yelp_agent.recommendation_v2.review_evidence.cross_encoder_judge import (
    CrossEncoderReviewEvidenceJudge,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
    ReviewSimilarityCandidate,
)


class FakePairScorer:
    provider = "fake"
    model = "fake-evidence-model"
    batch_size = 2

    def score(self, query_text: str, documents) -> ScoredPairBatch:
        scores = tuple(self._score(query_text, document) for document in documents)
        return ScoredPairBatch(
            scores=scores,
            input_tokens=10 * len(documents),
            per_pair_input_tokens=tuple(10 for _ in documents),
            truncated_pair_count=0,
            latency_ms=1.0,
        )

    @staticmethod
    def _score(query: str, document: str) -> float:
        if "Does the review passage discuss" in query:
            return 0.1 if document == "general" else 0.9
        if "satisfies the requirement" in query:
            return {
                "authentic": 0.9,
                "not authentic": 0.1,
                "both": 0.8,
                "uncertain": 0.3,
            }[document]
        return {
            "authentic": 0.1,
            "not authentic": 0.9,
            "both": 0.8,
            "uncertain": 0.3,
        }[document]


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


def _candidate(index: int, text: str) -> ReviewSimilarityCandidate:
    digest = f"{index:064x}"
    return ReviewSimilarityCandidate(
        review_id=f"review-{index}",
        business_id="business-1",
        user_id="user-1",
        review_time=datetime(2020, 1, 1, tzinfo=UTC),
        stars=4,
        useful=0,
        review_text=text,
        review_text_sha256=digest,
        matched_segment_id=digest,
        matched_segment_text=text,
        positive_similarity=0.7,
        negative_similarity=0.7,
        direction="ambiguous",
    )


def test_judge_separates_relevance_from_evidence_direction() -> None:
    candidates = [
        _candidate(1, "authentic"),
        _candidate(2, "not authentic"),
        _candidate(3, "both"),
        _candidate(4, "uncertain"),
        _candidate(5, "general"),
    ]

    result = CrossEncoderReviewEvidenceJudge(FakePairScorer()).judge(
        _requirement(), candidates
    )

    assert [item.label for item in result.judgments] == [
        "positive",
        "negative",
        "mixed",
        "ambiguous",
        "irrelevant",
    ]
    assert result.metrics.candidate_count == 5
    assert result.metrics.relevant_candidate_count == 4
    assert result.metrics.scored_pair_count == 13
    assert result.metrics.scorer_call_count == 7


def test_irrelevant_candidate_skips_positive_and_negative_scoring() -> None:
    result = CrossEncoderReviewEvidenceJudge(FakePairScorer()).judge(
        _requirement(), [_candidate(1, "general")]
    )

    judgment = result.judgments[0]
    assert judgment.label == "irrelevant"
    assert judgment.positive_support_score is None
    assert judgment.negative_support_score is None
    assert result.metrics.scored_pair_count == 1
