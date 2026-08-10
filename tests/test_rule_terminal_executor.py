"""Behavior tests for Step 24 deterministic terminal responses."""

from __future__ import annotations

from datetime import datetime

from yelp_agent.agent_harness import (
    AgentDecision,
    AgentObservation,
    AgentSession,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.query import QueryParseInput
from yelp_agent.rule_router import RuleRouter, TerminalActionExecutor


def _state(query_text: str, *, language: str = "zh-CN") -> AgentSession:
    interpretation = RuleBasedRequestInterpreter().interpret(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text=query_text,
        )
    )
    return AgentSession(
        scenario_id="c" * 64,
        split="development",
        language=language,
        agent_version="rule-router-v1",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        request=interpretation.request,
        readiness=interpretation.readiness,
        budget=HarnessBudget(),
        started_at_ms=0,
    )


def test_terminal_executor_turns_a_gap_decision_into_a_localized_question() -> None:
    state = _state("Find a steakhouse within 5 km")
    decision = RuleRouter().choose_action(state)

    outcome = TerminalActionExecutor().execute(state, decision)

    assert outcome.status == "completed"
    assert outcome.response_kind == "clarification"
    assert outcome.clarification_question is not None
    assert outcome.clarification_question.requested_information_gaps == [
        "missing_location"
    ]
    assert "位置" in outcome.clarification_question.question_text


def test_terminal_executor_returns_only_the_current_ranked_candidates() -> None:
    state = _state("Recommend a restaurant", language="en-US").model_copy(
        update={
            "business_scope_known": True,
            "business_scope": ["b1", "b2", "b3"],
            "observations": [
                AgentObservation(
                    observation_id="1" * 64,
                    turn_index=1,
                    step_index=1,
                    action="rank_candidates",
                    payload={
                        "tool_name": "GET_HYBRID_RANKING",
                        "status": "success",
                        "data": {"ranking": ["b2", "b1", "b3"]},
                    },
                )
            ],
        }
    )
    decision = AgentDecision(
        action="return_recommendation",
        arguments={"business_ids": ["b2", "b1"]},
        reason_code="READY_TO_FINALIZE",
    )

    outcome = TerminalActionExecutor().execute(state, decision)

    assert outcome.status == "completed"
    assert outcome.response_kind == "recommendation"
    assert outcome.candidate_ranking == ["b2", "b1", "b3"]
    assert outcome.recommended_business_ids == ["b2", "b1"]


def test_grounded_detail_answer_cites_the_observed_business_field() -> None:
    state = _state("What category is this business?", language="en-US")
    state = state.model_copy(
        update={
            "request": state.request.model_copy(
                update={"referenced_business_ids": ["b1"]}
            ),
            "readiness": state.readiness.model_copy(
                update={"task_type": "business_detail_question"}
            ),
            "observations": [
                AgentObservation(
                    observation_id="2" * 64,
                    turn_index=1,
                    step_index=1,
                    action="get_business_details",
                    payload={
                        "tool_name": "GET_BUSINESS_DETAILS",
                        "status": "success",
                        "data": {
                            "businesses": [
                                {
                                    "business_id": "b1",
                                    "name": "Example Steakhouse",
                                    "categories": ["Steakhouses", "Restaurants"],
                                }
                            ]
                        },
                    },
                )
            ],
        }
    )
    decision = AgentDecision(
        action="return_grounded_answer",
        arguments={"business_ids": ["b1"]},
        reason_code="STRUCTURED_EVIDENCE_SUFFICIENT",
    )

    outcome = TerminalActionExecutor().execute(state, decision)

    assert outcome.status == "completed"
    assert outcome.response_kind == "grounded_answer"
    assert len(outcome.claims) == 1
    assert outcome.claims[0].business_id == "b1"
    assert outcome.claims[0].evidence_refs[0].source_field == "categories"


def test_uncertain_answer_preserves_official_verification_requirement() -> None:
    state = _state("Does this business officially allow pets?", language="en-US")
    decision = AgentDecision(
        action="return_uncertain_answer",
        arguments={
            "reason": "official_verification_required",
            "recommended_official_verification": True,
        },
        reason_code="OFFICIAL_VERIFICATION_REQUIRED",
    )

    outcome = TerminalActionExecutor().execute(state, decision)

    assert outcome.status == "completed"
    assert outcome.response_kind == "uncertain_answer"
    assert outcome.claims == []
    assert outcome.recommended_official_verification is True


def test_grounded_review_profile_reports_aggregate_evidence_and_recency() -> None:
    state = _state(
        "Is this business quiet according to reviews?",
        language="en-US",
    )
    state = state.model_copy(
        update={
            "request": state.request.model_copy(
                update={"referenced_business_ids": ["b1"]}
            ),
            "readiness": state.readiness.model_copy(
                update={"task_type": "review_experience_question"}
            ),
            "observations": [
                AgentObservation(
                    observation_id="3" * 64,
                    turn_index=1,
                    step_index=1,
                    action="get_business_details",
                    payload={
                        "tool_name": "GET_BUSINESS_PROFILE",
                        "status": "success",
                        "data": {
                            "profiles": [
                                {
                                    "business_id": "b1",
                                    "name": "Example Steakhouse",
                                    "aspect_summaries": {
                                        "quiet_environment": {
                                            "status": "known",
                                            "weighted_positive_ratio": 0.8,
                                            "weighted_negative_ratio": 0.2,
                                            "confidence": 0.7,
                                            "conflict": False,
                                            "latest_evidence_time": (
                                                "2021-12-01T00:00:00"
                                            ),
                                        }
                                    },
                                }
                            ]
                        },
                    },
                )
            ],
        }
    )
    decision = AgentDecision(
        action="return_grounded_answer",
        arguments={"business_ids": ["b1"]},
        reason_code="STRUCTURED_EVIDENCE_SUFFICIENT",
    )

    outcome = TerminalActionExecutor().execute(state, decision)

    assert outcome.status == "completed"
    assert outcome.claims[0].evidence_refs[0].source_field == (
        "aspect_summaries.quiet_environment"
    )
    assert "positive" in outcome.claims[0].text
    assert outcome.reported_evidence_recency is True
