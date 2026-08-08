from datetime import datetime

from yelp_agent.query import (
    ExtractedRequestSignal,
    QueryParseInput,
    RecommendationRequestParser,
    RuleBasedRequestSignalExtractor,
    build_rule_based_request_parser,
)


def test_parser_separates_filterable_constraints_from_soft_preferences() -> None:
    parser = build_rule_based_request_parser()

    request = parser.parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text=(
                "只想吃牛排，不要酒吧，必须在5公里以内，最好安静一点，两个人。"
            ),
            user_latitude=39.9526,
            user_longitude=-75.1652,
        )
    )

    assert request.intent == "recommendation_request"
    assert request.desired_categories == ["Steakhouses"]
    assert request.excluded_categories == ["Bars"]
    assert request.party_size == 2
    assert request.missing_fields == []
    assert {
        (condition.field, condition.operator, condition.value)
        for condition in request.hard_constraints
    } == {
        ("category", "includes", "Steakhouses"),
        ("category", "excludes", "Bars"),
        ("distance_km", "less_than_or_equal", 5.0),
    }
    assert [condition.field for condition in request.soft_preferences] == [
        "quiet_environment"
    ]
    assert request.soft_preferences[0].importance == "preferred"
    assert request.soft_preferences[0].evidence_span == "最好安静一点"
    assert request.evidence_requirements == []


def test_semantic_adapter_cannot_turn_a_subjective_requirement_into_a_filter() -> None:
    class FakeSemanticExtractor:
        version = "fake-semantic-v1"

        def extract(self, value: QueryParseInput):
            start = value.query_text.index("必须安静")
            return (
                ExtractedRequestSignal(
                    field="quiet_environment",
                    operator="prefer",
                    value=True,
                    importance="mandatory",
                    evidence_span="必须安静",
                    evidence_start=start,
                    evidence_end=start + len("必须安静"),
                    confidence=0.91,
                    source="semantic_model",
                ),
            )

    parser = RecommendationRequestParser(FakeSemanticExtractor())

    request = parser.parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="这次求婚的餐厅必须安静",
        )
    )

    assert request.hard_constraints == []
    assert request.soft_preferences == []
    assert len(request.evidence_requirements) == 1
    condition = request.evidence_requirements[0]
    assert condition.field == "quiet_environment"
    assert condition.source == "semantic_model"
    assert condition.unknown_policy == "ask"


def test_unverifiable_budget_and_missing_location_require_clarification() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="想吃日料，人均不超过200，必须在3公里以内。",
        )
    )

    assert request.desired_categories == ["Japanese"]
    assert request.hard_constraints == []
    assert {item.field for item in request.clarification_requirements} == {
        "budget_per_person",
        "distance_km",
    }
    assert set(request.missing_fields) == {
        "budget_precision",
        "user_location",
    }


def test_parser_marks_missing_category_without_inventing_one() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="帮我找一个环境好一点的地方。",
        )
    )

    assert request.conditions == []
    assert request.desired_categories == []
    assert request.missing_fields == ["desired_category"]


def test_optional_semantic_adapter_failure_keeps_the_rule_baseline() -> None:
    class FailingSemanticExtractor:
        version = "failing-semantic-v1"

        def extract(self, value: QueryParseInput):
            raise TimeoutError("model timeout")

    parser = RecommendationRequestParser(
        primary_extractor=RuleBasedRequestSignalExtractor(),
        semantic_extractor=FailingSemanticExtractor(),
    )

    request = parser.parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="想吃牛排，最好安静一点。",
        )
    )

    assert request.desired_categories == ["Steakhouses"]
    assert [item.field for item in request.soft_preferences] == [
        "category",
        "quiet_environment",
    ]
    assert request.parse_warnings == [
        "SEMANTIC_EXTRACTOR_FAILED:failing-semantic-v1:TimeoutError"
    ]
