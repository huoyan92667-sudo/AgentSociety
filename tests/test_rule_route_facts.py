"""Public-interface tests for the Step 24.1 Router state summary."""

from __future__ import annotations

from datetime import datetime

from yelp_agent.agent_harness import (
    AgentObservation,
    AgentSession,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.decision_readiness import DecisionReadiness
from yelp_agent.query import QueryParseInput
from yelp_agent.rule_router import RouteFacts


def _state(
    *,
    query_text: str = "I only want a steakhouse within 5 km",
    referenced_business_ids: list[str] | None = None,
) -> AgentSession:
    interpretation = RuleBasedRequestInterpreter().interpret(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2022, 1, 1),
            query_text=query_text,
            referenced_business_ids=referenced_business_ids or [],
        )
    )
    return AgentSession(
        scenario_id="a" * 64,
        split="development",
        language="en-US",
        agent_version="rule-router-v1",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        request=interpretation.request,
        readiness=interpretation.readiness,
        budget=HarnessBudget(
            max_steps=10,
            max_tool_calls=5,
            max_semantic_calls=1,
            max_rag_calls=2,
            max_total_tokens=12_000,
            timeout_ms=90_000,
        ),
        started_at_ms=0,
    )


def _tool_observation(
    *,
    observation_id: str,
    step_index: int,
    tool_name: str,
    data: dict[str, object],
    status: str = "success",
    turn_index: int = 1,
    error_code: str | None = None,
) -> AgentObservation:
    return AgentObservation(
        observation_id=observation_id * 64,
        turn_index=turn_index,
        step_index=step_index,
        action={
            "EXPAND_CANDIDATES": "retrieve_candidates",
            "APPLY_CONSTRAINTS": "apply_hard_constraints",
            "GET_HYBRID_RANKING": "rank_candidates",
            "GET_BUSINESS_DETAILS": "get_business_details",
            "GET_BUSINESS_PROFILE": "get_business_details",
            "COMPARE_BUSINESSES": "compare_candidates",
        }[tool_name],
        payload={
            "tool_name": tool_name,
            "status": status,
            "data": data,
            "error_code": error_code,
        },
    )


def test_route_facts_summarize_a_fresh_visible_request() -> None:
    state = _state(referenced_business_ids=["business-1"])

    facts = RouteFacts.from_state(state)

    assert facts.request_id == state.request.request_id
    assert facts.current_turn == 1
    assert facts.task_type == "recommendation_request"
    assert facts.information_gaps == ["missing_location"]
    assert facts.hard_constraints_required is True
    assert facts.referenced_business_ids == ["business-1"]
    assert facts.business_scope_known is False
    assert facts.candidate_ids == []
    assert facts.remaining.steps == 10
    assert facts.remaining.tool_calls == 5
    assert facts.remaining.semantic_calls == 1
    assert facts.remaining.rag_calls == 2
    assert facts.remaining.tokens == 12_000


def test_route_facts_summarize_the_current_tool_chain_without_raw_rows() -> None:
    state = _state().model_copy(
        update={
            "observations": [
                _tool_observation(
                    observation_id="b",
                    step_index=1,
                    tool_name="EXPAND_CANDIDATES",
                    data={
                        "candidate_business_ids": ["business-1", "business-2"],
                        "candidates": [
                            {"business_id": "business-1", "fusion_score": 0.8},
                            {"business_id": "business-2", "fusion_score": 0.7},
                        ],
                    },
                ),
                _tool_observation(
                    observation_id="c",
                    step_index=2,
                    tool_name="APPLY_CONSTRAINTS",
                    data={"candidate_business_ids": ["business-2"], "excluded": []},
                ),
                _tool_observation(
                    observation_id="d",
                    step_index=3,
                    tool_name="GET_HYBRID_RANKING",
                    data={
                        "ranking": ["business-2"],
                        "scored_candidates": [
                            {"business_id": "business-2", "blend_score": 1.0}
                        ],
                    },
                ),
                _tool_observation(
                    observation_id="e",
                    step_index=4,
                    tool_name="GET_BUSINESS_DETAILS",
                    data={"businesses": [{"business_id": "business-2"}]},
                ),
            ],
            "business_scope": ["business-2"],
            "business_scope_known": True,
            "step_count": 4,
            "tool_call_count": 4,
        }
    )

    facts = RouteFacts.from_state(state)

    assert facts.candidate_ids == ["business-2"]
    assert facts.retrieved_candidate_ids == ["business-1", "business-2"]
    assert facts.ranked_business_ids == ["business-2"]
    assert facts.detailed_business_ids == ["business-2"]
    assert facts.candidate_retrieval is not None
    assert facts.candidate_retrieval.status == "success"
    assert facts.constraint_filter is not None
    assert facts.constraint_filter.turn_index == 1
    assert facts.hybrid_ranking is not None
    assert facts.business_details is not None
    assert facts.last_tool is not None
    assert facts.last_tool.tool_name == "GET_BUSINESS_DETAILS"
    assert facts.remaining.steps == 6
    assert facts.remaining.tool_calls == 1
    assert "candidates" not in facts.model_dump_json()


def test_route_facts_reuse_stable_history_but_not_old_filter_or_ranking() -> None:
    state = _state().model_copy(
        update={
            "current_turn": 2,
            "observations": [
                _tool_observation(
                    observation_id="f",
                    step_index=1,
                    tool_name="EXPAND_CANDIDATES",
                    data={"candidate_business_ids": ["business-1"]},
                ),
                _tool_observation(
                    observation_id="1",
                    step_index=2,
                    tool_name="APPLY_CONSTRAINTS",
                    data={"candidate_business_ids": ["business-1"]},
                ),
                _tool_observation(
                    observation_id="2",
                    step_index=3,
                    tool_name="GET_HYBRID_RANKING",
                    data={"ranking": ["business-1"]},
                ),
                _tool_observation(
                    observation_id="3",
                    step_index=4,
                    tool_name="GET_BUSINESS_DETAILS",
                    data={"businesses": [{"business_id": "business-1"}]},
                ),
            ],
            "business_scope": ["business-1"],
            "business_scope_known": True,
        }
    )

    facts = RouteFacts.from_state(state)

    assert facts.candidate_retrieval is not None
    assert facts.candidate_retrieval.turn_index == 1
    assert facts.retrieved_candidate_ids == ["business-1"]
    assert facts.constraint_filter is None
    assert facts.hybrid_ranking is None
    assert facts.ranked_business_ids == []
    assert facts.business_details is not None
    assert facts.detailed_business_ids == ["business-1"]


def test_route_facts_accumulate_reusable_details_and_current_comparison() -> None:
    state = _state(
        query_text="Compare these businesses",
        referenced_business_ids=["business-1", "business-2"],
    ).model_copy(
        update={
            "current_turn": 2,
            "observations": [
                _tool_observation(
                    observation_id="4",
                    step_index=1,
                    turn_index=1,
                    tool_name="GET_BUSINESS_DETAILS",
                    data={"businesses": [{"business_id": "business-1"}]},
                ),
                _tool_observation(
                    observation_id="5",
                    step_index=1,
                    turn_index=2,
                    tool_name="GET_BUSINESS_DETAILS",
                    data={"businesses": [{"business_id": "business-2"}]},
                ),
                _tool_observation(
                    observation_id="6",
                    step_index=2,
                    turn_index=2,
                    tool_name="GET_BUSINESS_PROFILE",
                    data={
                        "profiles": [
                            {"business_id": "business-1"},
                            {"business_id": "business-2"},
                        ]
                    },
                ),
                _tool_observation(
                    observation_id="7",
                    step_index=3,
                    turn_index=2,
                    tool_name="COMPARE_BUSINESSES",
                    data={
                        "ranking": ["business-2", "business-1"],
                        "compared": [
                            {"business_id": "business-2"},
                            {"business_id": "business-1"},
                        ],
                    },
                ),
            ],
            "business_scope": ["business-1", "business-2"],
            "business_scope_known": True,
        }
    )

    facts = RouteFacts.from_state(state)

    assert facts.detailed_business_ids == ["business-1", "business-2"]
    assert facts.profiled_business_ids == ["business-1", "business-2"]
    assert facts.business_profiles is not None
    assert facts.business_profiles.turn_index == 2
    assert facts.comparison is not None
    assert facts.comparison.turn_index == 2
    assert facts.compared_business_ids == ["business-2", "business-1"]
    assert facts.comparison_ranking == ["business-2", "business-1"]


def test_route_facts_distinguish_empty_results_from_tool_errors() -> None:
    malformed = AgentObservation(
        observation_id="8" * 64,
        turn_index=1,
        step_index=1,
        action="retrieve_candidates",
        payload={"unexpected": "legacy observation"},
    )
    state = _state().model_copy(
        update={
            "observations": [
                malformed,
                _tool_observation(
                    observation_id="9",
                    step_index=2,
                    tool_name="EXPAND_CANDIDATES",
                    status="no_result",
                    data={"candidate_business_ids": []},
                ),
                _tool_observation(
                    observation_id="a",
                    step_index=3,
                    tool_name="GET_BUSINESS_DETAILS",
                    status="unavailable",
                    error_code="TOOL_NOT_IMPLEMENTED",
                    data={},
                ),
            ],
            "business_scope": [],
            "business_scope_known": True,
        }
    )

    facts = RouteFacts.from_state(state)

    assert facts.business_scope_known is True
    assert facts.candidate_ids == []
    assert facts.candidate_retrieval is not None
    assert facts.candidate_retrieval.status == "no_result"
    assert facts.business_details is not None
    assert facts.business_details.status == "unavailable"
    assert facts.last_tool is not None
    assert facts.last_tool.error_code == "TOOL_NOT_IMPLEMENTED"


def test_route_facts_expose_conflicts_and_calibrated_ranking_reliability() -> None:
    state = _state()
    readiness_payload = state.readiness.model_dump(mode="json")
    readiness_payload.update(
        {
            "information_gaps": ["constraint_conflict"],
            "conflict_fields": ["category"],
            "ranking_confidence": {
                "probability_top1_correct": 0.37,
                "target": "hybrid_v2_b_next_business_top1",
                "calibrator_kind": "logistic",
                "calibrator_version": "test-v1",
                "uncertainty_reasons": [
                    "feature_disagreement",
                    "small_top_margin",
                ],
            },
            "confidence_target": "hybrid_v2_b_next_business_top1",
            "confidence_unavailable_reason": None,
        }
    )
    state = state.model_copy(
        update={"readiness": DecisionReadiness.model_validate(readiness_payload)}
    )

    facts = RouteFacts.from_state(state)

    assert facts.conflict_fields == ["category"]
    assert facts.ranking_confidence == 0.37
    assert facts.ranking_uncertainty_reasons == [
        "feature_disagreement",
        "small_top_margin",
    ]
