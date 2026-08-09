from datetime import datetime

from yelp_agent.decision_readiness import (
    DecisionReadinessAnalyzer,
    RankingConfidenceEstimate,
    RankingSignals,
    classify_task_type,
    identify_information_gaps,
)
from yelp_agent.query.schema import (
    RecommendationRequest,
    RequestCondition,
)


def _request(
    text: str,
    *,
    conditions: list[RequestCondition] | None = None,
    party_size: int | None = None,
    missing_fields: list[str] | None = None,
    references: list[str] | None = None,
) -> RecommendationRequest:
    return RecommendationRequest(
        request_id="a" * 64,
        user_id="user",
        session_id="session",
        cutoff_time=datetime(2024, 1, 1),
        query_text=text,
        intent="recommendation_request",
        conditions=conditions or [],
        party_size=party_size,
        missing_fields=missing_fields or [],
        referenced_business_ids=references or [],
        parser_version="test",
    )


def _condition(
    field: str,
    operator: str,
    value: str | float | bool,
) -> RequestCondition:
    return RequestCondition.model_validate(
        {
            "field": field,
            "operator": operator,
            "value": value,
            "importance": "mandatory",
            "enforcement": "filter",
            "explicit": True,
            "confidence": 1.0,
            "evidence_span": "x",
            "evidence_start": 0,
            "evidence_end": 1,
            "source": "rule",
            "unknown_policy": "exclude",
        }
    )


def test_task_type_priority_distinguishes_six_supported_actions() -> None:
    cases = (
        (_request("推荐一家安静的牛排馆", conditions=[_condition("category", "includes", "Steakhouses")]), "recommendation_request"),
        (_request("第一家停车方便吗？", references=["b1"]), "business_detail_question"),
        (_request("第一家和第二家哪个更适合约会？", references=["b1", "b2"]), "candidate_comparison"),
        (_request("这几家太贵了，换一个", references=["b1"]), "feedback_refinement"),
        (_request("这家目前官方允许带宠物吗？", references=["b1"]), "official_policy_question"),
        (_request("评论里有人说这家很吵吗？", references=["b1"]), "review_experience_question"),
    )

    assert [classify_task_type(request)[0] for request, _ in cases] == [
        expected for _, expected in cases
    ]


def test_information_gaps_include_missing_inputs_conflicts_and_references() -> None:
    request = _request(
        "比较第一家和第二家，想找适合聚会的，必须在5公里内",
        conditions=[
            _condition("category", "includes", "Bars"),
            _condition("category", "excludes", "Bars"),
            _condition("group_suitable", "prefer", True),
        ],
        missing_fields=["user_location"],
        references=["b1"],
    )

    task_type, _ = classify_task_type(request)
    gaps, conflicts = identify_information_gaps(request, task_type)

    assert task_type == "candidate_comparison"
    assert gaps == (
        "ambiguous_reference",
        "constraint_conflict",
        "missing_location",
        "missing_party_size",
    )
    assert conflicts == ("category",)


def test_analyzer_refuses_to_fake_query_aware_confidence() -> None:
    request = _request(
        "推荐一家安静的牛排馆",
        conditions=[_condition("category", "includes", "Steakhouses")],
    )

    readiness = DecisionReadinessAnalyzer().analyze(
        request,
        ranking_source="query_aware",
    )

    assert readiness.task_type == "recommendation_request"
    assert readiness.ranking_confidence is None
    assert readiness.confidence_target == "unavailable"
    assert (
        readiness.confidence_unavailable_reason
        == "query_aware_labels_unavailable"
    )


class _FakeCalibrator:
    def estimate(self, signals: RankingSignals) -> RankingConfidenceEstimate:
        assert signals.blend_score_margin == 0.01
        return RankingConfidenceEstimate(
            probability_top1_correct=0.12,
            target="hybrid_v2_b_next_business_top1",
            calibrator_kind="logistic",
            calibrator_version="test",
            uncertainty_reasons=["small_top_margin"],
        )


def test_analyzer_combines_request_and_hybrid_ranking_state() -> None:
    request = _request(
        "推荐一家适合聚餐的餐厅",
        conditions=[_condition("group_suitable", "prefer", True)],
        missing_fields=["user_location"],
    )
    signals = RankingSignals(
        top1_blend_score=0.2,
        blend_score_margin=0.01,
        model_score_margin=0.02,
        hybrid_v1_score_margin=0.01,
        model_v1_rank_disagreement=0.3,
        component_disagreement=0.4,
        top1_item_knn_positive_score=0.1,
        top1_item_knn_missing=0.0,
        user_history_length_log=3.0,
        user_profile_reliability=0.6,
        top1_user_category_novelty=0.2,
        top1_route_coverage=4.0,
    )

    readiness = DecisionReadinessAnalyzer(_FakeCalibrator()).analyze(
        request,
        ranking_source="hybrid_v2_b",
        ranking_signals=signals,
    )

    assert readiness.task_type == "recommendation_request"
    assert readiness.information_gaps == ["missing_location", "missing_party_size"]
    assert readiness.ranking_confidence is not None
    assert readiness.ranking_confidence.probability_top1_correct == 0.12
    assert readiness.ranking_confidence.uncertainty_reasons == ["small_top_margin"]
    assert readiness.confidence_unavailable_reason is None
