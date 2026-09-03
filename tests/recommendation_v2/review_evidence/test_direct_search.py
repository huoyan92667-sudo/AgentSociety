from __future__ import annotations

from datetime import UTC, datetime

from yelp_agent.recommendation_v2.review_evidence.descriptions import (
    DescriptionBuildResult,
)
from yelp_agent.recommendation_v2.review_evidence.direct_search import (
    DirectReviewEvidenceSearch,
)
from yelp_agent.recommendation_v2.review_evidence.retrieval import (
    ReviewRetrievalBatch,
)
from yelp_agent.recommendation_v2.review_evidence.schema import (
    PreferenceSearchDescription,
    ReviewRetrievalMetrics,
    ReviewSimilarityCandidate,
)

NOW = datetime(2026, 8, 26, tzinfo=UTC)


class FakeDescriptionBuilder:
    def build(self, preferences, open_requirements, *, query_text=None):
        requirement = open_requirements[0]
        return DescriptionBuildResult(
            descriptions=[
                PreferenceSearchDescription(
                    requirement_id=requirement.key,
                    requirement_text=requirement.text,
                    kind="long_tail",
                    priority=1,
                    preference_strength=75,
                    positive_descriptions=[
                        "Suitable for a quiet business conversation.",
                        "Tables allow private discussion without shouting.",
                    ],
                    negative_descriptions=[
                        "Unsuitable for a business conversation.",
                        "Noise makes a focused discussion difficult.",
                    ],
                )
            ]
        )


class FakeRetriever:
    acceptance_threshold = 0.60
    recall_threshold = 0.55
    direction_margin = 0.05

    def retrieve_many(self, requirements, business_ids, *, cutoff_time=None):
        requirement = requirements[0]
        return ReviewRetrievalBatch(
            by_requirement={
                requirement.requirement_id: {
                    business_ids[0]: [
                        _candidate("review-positive", "positive", 0.84, 0.31),
                        _candidate("review-negative", "negative", 0.30, 0.81),
                    ]
                }
            },
            metrics=ReviewRetrievalMetrics(
                query_vector_count=4,
                bm25_enabled=True,
                full_review_count=2,
            ),
        )


def _candidate(
    review_id: str,
    direction: str,
    positive: float,
    negative: float,
) -> ReviewSimilarityCandidate:
    return ReviewSimilarityCandidate(
        review_id=review_id,
        business_id="business-1",
        user_id=f"user-{review_id}",
        review_time=datetime(2025, 8, 1, tzinfo=UTC),
        stars=5 if direction == "positive" else 2,
        useful=2,
        review_text=(
            "We could discuss work comfortably."
            if direction == "positive"
            else "The room was too loud for a serious discussion."
        ),
        review_text_sha256="a" * 64 if direction == "positive" else "b" * 64,
        matched_segment_id="c" * 64 if direction == "positive" else "d" * 64,
        matched_segment_text="matched text",
        positive_similarity=positive,
        negative_similarity=negative,
        direction=direction,  # type: ignore[arg-type]
    )


def test_direct_search_uses_model_selected_business_and_natural_language_need() -> None:
    search = DirectReviewEvidenceSearch(
        description_builder=FakeDescriptionBuilder(),  # type: ignore[arg-type]
        retriever=FakeRetriever(),  # type: ignore[arg-type]
    )

    result = search.search(
        business_ids=["business-1"],
        evidence_queries=["是否适合安静地进行商务谈判"],
        user_query_text="第三家适合商务谈判吗？",
        reference_time=NOW,
    )

    assert result.status == "success"
    assert result.model_call_count == 0
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.requirement_text == "是否适合安静地进行商务谈判"
    assert [item.review_id for item in finding.assessment.positive_evidence] == [
        "review-positive"
    ]
    assert [item.review_id for item in finding.assessment.negative_evidence] == [
        "review-negative"
    ]
    assert result.retrieval_metrics is not None
    assert result.retrieval_metrics.bm25_enabled is True
