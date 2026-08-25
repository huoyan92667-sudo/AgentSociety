import json
from datetime import UTC, datetime

from yelp_agent.agent.llm import LLMCallResult, LLMMessage
from yelp_agent.recommendation_v2.answer_synthesis import (
    RecommendationAnswerSynthesizer,
)
from yelp_agent.recommendation_v2.business_facts import BusinessFact, WeeklyHours
from yelp_agent.recommendation_v2.review_evidence.schema import (
    BusinessPreferenceEvidence,
    PreferenceSearchDescription,
    RankedEvidenceBusiness,
    RankedReviewEvidence,
    ReviewEvidenceRankingResult,
)
from yelp_agent.recommendation_v2.schema import (
    HardConstraint,
    RequirementBasis,
    UnifiedRecommendationState,
)

NOW = datetime(2026, 8, 25, tzinfo=UTC)


class FakeAnswerGenerator:
    """保存最终总结实际收到的内容，并返回一段自然中文。"""

    def __init__(self) -> None:
        self.messages: list[LLMMessage] = []

    def generate(self, messages: list[LLMMessage]) -> LLMCallResult:
        self.messages = list(messages)
        return LLMCallResult(
            status="success",
            content="首选测试川菜馆：评论直接提到川味足，但也有人觉得偏咸。",
            model="fake-answer-model",
            latency_ms=3,
            attempt_count=1,
            input_tokens=100,
            output_tokens=20,
        )


def _review(review_id: str, role: str, weight: float) -> RankedReviewEvidence:
    return RankedReviewEvidence.model_validate(
        {
            "review_id": review_id,
            "role": role,
            "review_time": NOW,
            "stars": 4,
            "review_text": f"完整真实评论 {review_id}",
            "matched_segment_text": f"命中片段 {review_id}",
            "positive_similarity": 0.9 if role == "positive" else 0.1,
            "negative_similarity": 0.1 if role == "positive" else 0.9,
            "relevance_score": weight,
            "time_weight": 1,
            "evidence_weight": weight,
        }
    )


def _assessment(
    requirement_id: str,
    positives: list[RankedReviewEvidence],
    negatives: list[RankedReviewEvidence],
) -> BusinessPreferenceEvidence:
    return BusinessPreferenceEvidence(
        business_id="business-1",
        requirement_id=requirement_id,
        positive_evidence=positives,
        negative_evidence=negatives,
        positive_component=0.8,
        negative_component=0.2,
        positive_count_reliability=0.7,
        negative_count_reliability=0.4,
        evidence_score=0.75,
        recalled_review_count=len(positives) + len(negatives),
        ambiguous_review_count=0,
        max_positive_similarity=0.9,
        max_negative_similarity=0.9 if negatives else 0.1,
        max_direction_gap=0.8,
    )


def _ranking() -> ReviewEvidenceRankingResult:
    requirements = [
        PreferenceSearchDescription(
            requirement_id="authentic-szechuan",
            requirement_text="地道的川菜",
            kind="long_tail",
            priority=1,
            preference_strength=100,
            positive_descriptions=["authentic Sichuan flavor", "proper mala balance"],
            negative_descriptions=["not authentic Sichuan", "watered down flavor"],
        ),
        PreferenceSearchDescription(
            requirement_id="date-fit",
            requirement_text="适合约会",
            kind="long_tail",
            priority=2,
            preference_strength=100,
            positive_descriptions=["good for a date", "comfortable for couples"],
            negative_descriptions=["bad for a date", "uncomfortable for couples"],
        ),
    ]
    first = _assessment(
        "authentic-szechuan",
        [_review("r1", "positive", 0.95), _review("r2", "positive", 0.8)],
        [_review("r3", "negative", 0.9)],
    )
    second = _assessment(
        "date-fit",
        [_review("r4", "positive", 0.88), _review("r5", "positive", 0.7)],
        [_review("r6", "negative", 0.75)],
    )
    business = BusinessFact(
        business_id="business-1",
        name="测试川菜馆",
        address="100 Test St",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=39.95,
        longitude=-75.16,
        categories=["Szechuan", "Chinese"],
        rating=4.5,
        review_count=200,
        weekly_hours=WeeklyHours(tuesday="11:0-21:30"),
    )
    return ReviewEvidenceRankingResult(
        status="success",
        hard_filtered_count=6,
        requirements=requirements,
        ranking=[
            RankedEvidenceBusiness(
                final_rank=1,
                business=business,
                distance_km=0.4,
                preference_score=0.75,
                rating_score=0.9,
                distance_score=0.95,
                final_score=0.82,
                preference_evidence=[first, second],
            )
        ],
        recall_threshold=0.55,
        acceptance_threshold=0.6,
        direction_margin=0.05,
        formula="test formula",
        model_call_count=1,
        latency_ms=10,
    )


def test_synthesis_sends_only_compact_full_review_evidence_and_returns_free_text() -> None:
    generator = FakeAnswerGenerator()
    synthesizer = RecommendationAnswerSynthesizer(generator)
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="今晚九点去费城唐人街吃地道川菜",
        hard_constraints=[
            HardConstraint(
                key="hard.open_at.test",
                field="open_at",
                operator="equals",
                value="2026-08-25T21:00:00-04:00",
                unit="datetime",
                merchant_feature="weekly_hours",
                controlling_source="current_query",
                sources=[
                    RequirementBasis(
                        source="current_query",
                        text="今晚九点",
                        turn_index=1,
                    )
                ],
            )
        ],
    )

    answer = synthesizer.synthesize(
        query_text=state.latest_query_text,
        state=state,
        ranking=_ranking(),
    )

    assert answer.status == "success"
    assert answer.text == "首选测试川菜馆：评论直接提到川味足，但也有人觉得偏咸。"
    assert len(answer.selected_review_ids_by_business["business-1"]) == 4
    assert "r3" in answer.selected_review_ids_by_business["business-1"]
    payload = json.loads(generator.messages[1].content)
    sent_reviews = payload["top5"][0]["evidence"]
    assert len(sent_reviews) == 4
    assert payload["top5"][0]["straight_line_distance_km"] == 0.4
    assert "distance_km" not in payload["top5"][0]
    assert all(item["full_review"].startswith("完整真实评论") for item in sent_reviews)
    assert payload["current_query"] == state.latest_query_text
    assert "weekly_hours" not in payload["top5"][0]["business"]
    assert payload["top5"][0]["visit_context"] == {
        "local_datetime": "2026-08-25T21:00-04:00",
        "weekday": "周二",
        "recorded_hours_for_visit_day": "11:0-21:30",
        "recorded_open_at_visit_time": True,
        "source": "historical_yelp_weekly_hours",
    }
    assert "不需要返回 JSON" in generator.messages[0].content
    assert "不是步行、驾车或路线距离" in generator.messages[0].content
