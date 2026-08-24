from __future__ import annotations

from datetime import UTC, datetime

from yelp_agent.agent.llm import LLMCallResult
from yelp_agent.recommendation_v2.business_facts import BusinessFact
from yelp_agent.recommendation_v2.schema import (
    RequirementBasis,
    SoftPreference,
    merchant_feature_for,
)
from yelp_agent.recommendation_v2.soft_ranking import (
    BaselineRankedBusiness,
    BaselineRankingResult,
    BusinessEvidenceProposal,
    EvidenceJudgeProposal,
    EvidenceJudgeResult,
    PreferenceEvidenceAssessment,
    PriorityLayeredRanker,
    RetrievedReview,
    ReviewEvidenceJudge,
    ReviewEvidenceProposal,
)
from yelp_agent.recommendation_v2.tools import (
    FilteredBusiness,
    RatingBaselineRankingTool,
    StructuredHardFilterResult,
)


def _preference(field: str, direction: str, priority: int) -> SoftPreference:
    return SoftPreference.model_validate(
        {
            "key": f"query.{field}",
            "field": field,
            "direction": direction,
            "preference_strength": 100,
            "priority": priority,
            "merchant_feature": merchant_feature_for(field),
            "controlling_source": "current_query",
            "sources": [
                RequirementBasis(
                    source="current_query",
                    text=f"prefer {field}",
                    turn_index=1,
                    preference_strength=100,
                )
            ],
        }
    )


def _candidate(business_id: str, distance: float) -> FilteredBusiness:
    return FilteredBusiness(
        business=BusinessFact(
            business_id=business_id,
            name=f"Restaurant {business_id}",
            address="1 Main St",
            city="Philadelphia",
            state="PA",
            postal_code="19107",
            latitude=39.95,
            longitude=-75.16,
            categories=["Restaurants", "Steakhouses"],
            rating=4.0,
            review_count=100,
        ),
        distance_km=distance,
    )


def test_rating_baseline_uses_reviews_then_distance_for_equal_rating() -> None:
    first = _candidate("b1", 3.0)
    second = _candidate("b2", 1.0)
    third = _candidate("b3", 0.5)
    first.business.review_count = 200
    second.business.review_count = 200
    third.business.review_count = 100
    hard_filter = StructuredHardFilterResult(
        source_business_count=3,
        candidate_count=3,
        candidate_business_ids=["b3", "b1", "b2"],
        candidates=[third, first, second],
        steps=[],
        generated_sql="SELECT * FROM facts",
        sql_parameters=[],
    )

    result = RatingBaselineRankingTool().execute(hard_filter=hard_filter)

    assert result.source == "rating"
    assert [item.business_id for item in result.ranked_businesses] == [
        "b2",
        "b1",
        "b3",
    ]


class FakeReviewStore:
    def retrieve(
        self,
        business_ids,
        satisfying_anchors,
        contradicting_anchors,
        *,
        limit_each_side,
    ):
        del satisfying_anchors, contradicting_anchors, limit_each_side
        return {business_id: [] for business_id in business_ids}

    def close(self) -> None:
        pass


class FakeEvidenceJudge:
    def judge(self, preference, reviews_by_business):
        del preference
        levels = {
            "b1": "conditionally_satisfies",
            "b2": "clearly_satisfies",
            "b3": "clearly_satisfies",
        }
        scores = {"b1": 0.6, "b2": 0.9, "b3": 0.9}
        return EvidenceJudgeResult(
            call=LLMCallResult(
                status="success",
                content='{"assessments":[]}',
                model="fake",
                latency_ms=1,
                attempt_count=1,
                input_tokens=10,
                output_tokens=5,
            ),
            raw_json='{"assessments":[]}',
            assessments=[
                PreferenceEvidenceAssessment(
                    business_id=business_id,
                    level=levels[business_id],
                    satisfaction_score=scores[business_id],
                    preference_weight=1.0,
                    reason="test evidence",
                    retrieved_review_count=len(reviews),
                )
                for business_id, reviews in reviews_by_business.items()
            ],
        )


def test_all_preferences_contribute_without_first_preference_veto() -> None:
    candidates = [_candidate("b1", 0.5), _candidate("b2", 3.0), _candidate("b3", 1.0)]
    hard_filter = StructuredHardFilterResult(
        source_business_count=3,
        candidate_count=3,
        candidate_business_ids=["b1", "b2", "b3"],
        candidates=candidates,
        steps=[],
        generated_sql="SELECT * FROM facts",
        sql_parameters=[],
    )
    baseline = BaselineRankingResult(
        source="legacy_hybrid_v2",
        ranked_businesses=[
            BaselineRankedBusiness(business_id=value, baseline_rank=index)
            for index, value in enumerate(["b1", "b2", "b3"], 1)
        ],
    )
    ranker = PriorityLayeredRanker(
        review_store=FakeReviewStore(),  # type: ignore[arg-type]
        evidence_judge=FakeEvidenceJudge(),  # type: ignore[arg-type]
        candidate_limit=3,
    )

    result = ranker.rank(
        preferences=[
            _preference("quiet_environment", "higher", 1),
            _preference("distance_km", "lower", 2),
        ],
        hard_filter=hard_filter,
        baseline=baseline,
    )

    assert result.status == "success"
    assert result.passes[0].order_after == ["b2", "b3", "b1"]
    # b2虽然第一项安静分高，但距离短板明显，不能仅凭第一项永远排在前面。
    assert result.passes[1].order_after == ["b3", "b1", "b2"]
    assert [item.business.business_id for item in result.ranking] == [
        "b3",
        "b1",
        "b2",
    ]


class FakeGenerator:
    def __init__(self, proposal: EvidenceJudgeProposal) -> None:
        self._proposal = proposal

    def generate(self, messages):
        assert "逐条阅读完整原评论" in messages[0].content
        return LLMCallResult(
            status="success",
            content=self._proposal.model_dump_json(),
            model="fake",
            latency_ms=1,
            attempt_count=1,
        )


def test_evidence_judge_attaches_original_review_text_by_id() -> None:
    review = RetrievedReview(
        review_id="r1",
        business_id="b1",
        review_time=datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
        stars=4,
        useful=2,
        review_text="The dining room was calm and quiet even on Friday night.",
        retrieval_side="positive",
        positive_similarity=0.9,
        negative_similarity=0.4,
        positive_weight=1.0,
    )
    judge = ReviewEvidenceJudge(
        FakeGenerator(
            EvidenceJudgeProposal(
                assessments=[
                    BusinessEvidenceProposal(
                        business_id="b1",
                        reviews=[
                            ReviewEvidenceProposal(
                                review_id="r1",
                                verdict="positive",
                                reason="评论直接说明周五也很安静",
                            )
                        ],
                    )
                ]
            )
        )
    )

    result = judge.judge(
        _preference("quiet_environment", "higher", 1),
        {"b1": [review]},
    )

    assert result.failure_reason is None
    # 只有一条有效评论时按固定规则仍是未知，避免单条好评就判“明确满足”。
    assert result.assessments[0].level == "unknown"
    evidence = result.assessments[0].positive_evidence[0]
    assert evidence.review_id == "r1"
    assert evidence.review_text == review.review_text
