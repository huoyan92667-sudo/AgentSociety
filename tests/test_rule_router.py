"""Behavior tests for the complete Step 24 deterministic Router."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from yelp_agent.agent_harness import (
    AgentObservation,
    AgentSession,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.query import QueryParseInput
from yelp_agent.rule_router import (
    RuleBasedActionPolicy,
    RuleRouter,
    load_rule_router_config,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _state(
    query_text: str,
    *,
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
        scenario_id="b" * 64,
        split="development",
        language="en-US",
        agent_version="rule-router-v1",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        request=interpretation.request,
        readiness=interpretation.readiness,
        budget=HarnessBudget(),
        started_at_ms=0,
    )


def _observation(
    marker: str,
    step: int,
    action: str,
    tool_name: str,
    data: dict[str, object],
) -> AgentObservation:
    return AgentObservation(
        observation_id=marker * 64,
        turn_index=1,
        step_index=step,
        action=action,
        payload={"tool_name": tool_name, "status": "success", "data": data},
    )


def test_review_rag_router_retrieves_then_returns_cited_answer() -> None:
    state = _state(
        "What do reviews say about this business's quiet environment?",
        referenced_business_ids=["b1"],
    ).model_copy(update={"business_scope_known": True, "business_scope": ["b1"]})
    router = RuleRouter(review_rag_enabled=True)
    policy = RuleBasedActionPolicy(review_rag_enabled=True)

    assert policy.allowed_actions(state) == (
        "retrieve_business_reviews",
        "safe_fallback",
    )
    decision = router.choose_action(state)
    assert decision.action == "retrieve_business_reviews"
    assert decision.tool_name == "SEARCH_BUSINESS_REVIEWS"
    assert decision.arguments == {"business_ids": ["b1"], "top_k": 5}

    searched = state.model_copy(
        update={
            "observations": [
                _observation(
                    "e",
                    1,
                    "retrieve_business_reviews",
                    "SEARCH_BUSINESS_REVIEWS",
                    {
                        "hits": [
                            {
                                "business_id": "b1",
                                "review_id": "r1",
                                "matched_aspects": ["quiet_environment"],
                                "aspect_sentiments": ["positive"],
                            }
                        ]
                    },
                )
            ]
        }
    )
    assert policy.allowed_actions(searched) == (
        "return_grounded_answer",
        "safe_fallback",
    )
    assert router.choose_action(searched).action == "return_grounded_answer"


def test_step28_router_searches_aggregates_then_answers() -> None:
    state = _state(
        "What do reviews say about this business's quiet environment?",
        referenced_business_ids=["b1"],
    ).model_copy(update={"business_scope_known": True, "business_scope": ["b1"]})
    router = RuleRouter(
        review_rag_enabled=True,
        evidence_aggregation_enabled=True,
    )
    policy = RuleBasedActionPolicy(
        review_rag_enabled=True,
        evidence_aggregation_enabled=True,
    )
    searched = state.model_copy(
        update={
            "observations": [
                _observation(
                    "e",
                    1,
                    "retrieve_business_reviews",
                    "SEARCH_BUSINESS_REVIEWS",
                    {
                        "hits": [
                            {
                                "business_id": "b1",
                                "review_id": "r1",
                                "matched_aspects": ["quiet_environment"],
                                "aspect_sentiments": ["positive"],
                            }
                        ]
                    },
                )
            ]
        }
    )
    aggregate_decision = router.choose_action(searched)
    assert policy.allowed_actions(searched) == (
        "retrieve_business_reviews",
        "safe_fallback",
    )
    assert aggregate_decision.tool_name == "AGGREGATE_REVIEW_EVIDENCE"
    aggregated = searched.model_copy(
        update={
            "observations": searched.observations
            + [
                _observation(
                    "f",
                    2,
                    "retrieve_business_reviews",
                    "AGGREGATE_REVIEW_EVIDENCE",
                    {
                        "businesses": [
                            {
                                "business_id": "b1",
                                "overall_response_mode": "grounded",
                                "has_conflict": False,
                                "aspects": [
                                    {
                                        "aspect": "quiet_environment",
                                        "evidence_count": 1,
                                    }
                                ],
                            }
                        ]
                    },
                )
            ]
        }
    )
    assert policy.allowed_actions(aggregated) == (
        "return_grounded_answer",
        "safe_fallback",
    )
    assert router.choose_action(aggregated).reason_code == (
        "AGGREGATED_REVIEW_EVIDENCE_SUFFICIENT"
    )


def test_step28_conflict_returns_uncertain_answer() -> None:
    state = _state(
        "Reviews conflict about the quiet environment.",
        referenced_business_ids=["b1"],
    ).model_copy(update={"business_scope_known": True, "business_scope": ["b1"]})
    observations = [
        _observation(
            "e",
            1,
            "retrieve_business_reviews",
            "SEARCH_BUSINESS_REVIEWS",
            {"hits": [{"business_id": "b1", "review_id": "r1"}]},
        ),
        _observation(
            "f",
            2,
            "retrieve_business_reviews",
            "AGGREGATE_REVIEW_EVIDENCE",
            {
                "businesses": [
                    {
                        "business_id": "b1",
                        "overall_response_mode": "uncertain",
                        "has_conflict": True,
                        "aspects": [
                            {"aspect": "quiet_environment", "evidence_count": 2}
                        ],
                    }
                ]
            },
        ),
    ]
    decision = RuleRouter(
        review_rag_enabled=True,
        evidence_aggregation_enabled=True,
    ).choose_action(state.model_copy(update={"observations": observations}))
    assert decision.action == "return_uncertain_answer"
    assert decision.reason_code == "CONFLICTING_REVIEW_EVIDENCE"


def _recommendation_states() -> tuple[
    AgentSession,
    AgentSession,
    AgentSession,
    AgentSession,
    AgentSession,
]:
    fresh = _state("I only want a steakhouse")
    retrieved = fresh.model_copy(
        update={
            "observations": [
                _observation(
                    "1",
                    1,
                    "retrieve_candidates",
                    "EXPAND_CANDIDATES",
                    {"candidate_business_ids": ["b1", "b2", "b3", "b4"]},
                )
            ],
            "business_scope_known": True,
            "business_scope": ["b1", "b2", "b3", "b4"],
        }
    )
    filtered = retrieved.model_copy(
        update={
            "observations": retrieved.observations
            + [
                _observation(
                    "2",
                    2,
                    "apply_hard_constraints",
                    "APPLY_CONSTRAINTS",
                    {"candidate_business_ids": ["b1", "b2", "b3"]},
                )
            ],
            "business_scope": ["b1", "b2", "b3"],
        }
    )
    ranked = filtered.model_copy(
        update={
            "observations": filtered.observations
            + [
                _observation(
                    "3",
                    3,
                    "rank_candidates",
                    "GET_HYBRID_RANKING",
                    {"ranking": ["b2", "b1", "b3"]},
                )
            ]
        }
    )
    detailed = ranked.model_copy(
        update={
            "observations": ranked.observations
            + [
                _observation(
                    "4",
                    4,
                    "get_business_details",
                    "GET_BUSINESS_DETAILS",
                    {
                        "businesses": [
                            {"business_id": "b2"},
                            {"business_id": "b1"},
                            {"business_id": "b3"},
                        ]
                    },
                )
            ]
        }
    )
    return fresh, retrieved, filtered, ranked, detailed


def test_policy_allows_only_clarification_or_fallback_for_a_blocking_gap() -> None:
    state = _state("Find a steakhouse within 5 km")

    allowed = RuleBasedActionPolicy().allowed_actions(state)

    assert allowed == ("ask_clarification", "safe_fallback")


def test_policy_exposes_one_recommendation_stage_at_a_time() -> None:
    policy = RuleBasedActionPolicy(display_limit=3)
    fresh, retrieved, filtered, ranked, detailed = _recommendation_states()

    assert policy.allowed_actions(fresh) == (
        "retrieve_candidates",
        "safe_fallback",
    )
    assert policy.allowed_actions(retrieved) == (
        "apply_hard_constraints",
        "safe_fallback",
    )
    assert policy.allowed_actions(filtered) == ("rank_candidates", "safe_fallback")
    assert policy.allowed_actions(ranked) == (
        "get_business_details",
        "safe_fallback",
    )
    assert policy.allowed_actions(detailed) == (
        "return_recommendation",
        "safe_fallback",
    )


def test_step25_router_calls_embedding_after_hybrid_and_uses_fused_prefix() -> None:
    _, _, filtered, ranked, _ = _recommendation_states()
    router = RuleRouter(
        display_limit=3,
        semantic_enabled=True,
        semantic_candidate_limit=3,
        fusion_alpha=0.6,
    )
    policy = RuleBasedActionPolicy(
        display_limit=3,
        semantic_enabled=True,
        fusion_alpha=0.6,
    )

    assert policy.allowed_actions(ranked) == ("rank_candidates", "safe_fallback")
    decision = router.choose_action(ranked)
    assert decision.tool_name == "COMPUTE_EMBEDDING_MATCH"
    assert decision.arguments == {"business_ids": ["b2", "b1", "b3"]}

    semantic = ranked.model_copy(
        update={
            "observations": ranked.observations
            + [
                _observation(
                    "5",
                    4,
                    "rank_candidates",
                    "COMPUTE_EMBEDDING_MATCH",
                    {
                        "matches": [
                            {"business_id": "b3", "semantic_rank": 1},
                            {"business_id": "b1", "semantic_rank": 2},
                            {"business_id": "b2", "semantic_rank": 3},
                        ]
                    },
                )
            ]
        }
    )
    details = router.choose_action(semantic)

    assert details.tool_name == "GET_BUSINESS_DETAILS"
    assert details.arguments == {"business_ids": ["b3", "b1", "b2"]}


def test_step25_router_default_semantic_scope_is_hybrid_top_30() -> None:
    _, _, _, ranked, _ = _recommendation_states()
    hybrid_ranking = [f"business-{index:02d}" for index in range(40)]
    state = ranked.model_copy(
        update={
            "business_scope": hybrid_ranking,
            "observations": ranked.observations[:-1]
            + [
                _observation(
                    "6",
                    3,
                    "rank_candidates",
                    "GET_HYBRID_RANKING",
                    {"ranking": hybrid_ranking},
                )
            ],
        }
    )

    decision = RuleRouter(semantic_enabled=True).choose_action(state)

    assert decision.tool_name == "COMPUTE_EMBEDDING_MATCH"
    assert decision.arguments == {"business_ids": hybrid_ranking[:30]}


def test_step24_interpreter_recognizes_open_category_recommendations() -> None:
    state = _state("For 6 people near 19103, it must be Lounges.")

    assert state.readiness.task_type == "recommendation_request"
    assert RuleRouter().choose_action(state).action == "retrieve_candidates"


def test_router_asks_for_the_highest_priority_blocking_gap() -> None:
    policy = RuleBasedActionPolicy()
    state = _state("Find a steakhouse within 5 km")
    state = state.model_copy(
        update={"available_actions": policy.allowed_actions(state)}
    )

    decision = RuleRouter().choose_action(state)

    assert decision.action == "ask_clarification"
    assert decision.arguments == {
        "information_gaps": ["missing_location"],
    }
    assert decision.reason_code == "MISSING_LOCATION"
    assert decision.tool_name is None


def test_router_selects_the_complete_recommendation_tool_chain() -> None:
    policy = RuleBasedActionPolicy(display_limit=3)
    router = RuleRouter(display_limit=3)
    states = _recommendation_states()
    decisions = []
    for state in states:
        state = state.model_copy(
            update={"available_actions": policy.allowed_actions(state)}
        )
        decisions.append(router.choose_action(state))

    assert [item.action for item in decisions] == [
        "retrieve_candidates",
        "apply_hard_constraints",
        "rank_candidates",
        "get_business_details",
        "return_recommendation",
    ]
    assert [item.tool_name for item in decisions] == [
        "EXPAND_CANDIDATES",
        "APPLY_CONSTRAINTS",
        "GET_HYBRID_RANKING",
        "GET_BUSINESS_DETAILS",
        None,
    ]
    assert decisions[1].arguments == {
        "business_ids": ["b1", "b2", "b3", "b4"]
    }
    assert decisions[2].arguments == {"business_ids": ["b1", "b2", "b3"]}
    assert decisions[3].arguments == {"business_ids": ["b2", "b1", "b3"]}
    assert decisions[-1].reason_code == "READY_TO_FINALIZE"


def test_business_detail_route_reads_locked_details_then_answers() -> None:
    policy = RuleBasedActionPolicy()
    router = RuleRouter()
    fresh = _state(
        "What is the address of this business?",
        referenced_business_ids=["b1"],
    )
    detailed = fresh.model_copy(
        update={
            "observations": [
                _observation(
                    "5",
                    1,
                    "get_business_details",
                    "GET_BUSINESS_DETAILS",
                    {"businesses": [{"business_id": "b1", "address": "Main St"}]},
                )
            ]
        }
    )

    fresh_allowed = policy.allowed_actions(fresh)
    detailed_allowed = policy.allowed_actions(detailed)
    first = router.choose_action(
        fresh.model_copy(update={"available_actions": fresh_allowed})
    )
    second = router.choose_action(
        detailed.model_copy(update={"available_actions": detailed_allowed})
    )

    assert fresh_allowed == ("get_business_details", "safe_fallback")
    assert first.action == "get_business_details"
    assert first.tool_name == "GET_BUSINESS_DETAILS"
    assert first.arguments == {"business_ids": ["b1"]}
    assert detailed_allowed == ("return_grounded_answer", "safe_fallback")
    assert second.action == "return_grounded_answer"
    assert second.reason_code == "STRUCTURED_EVIDENCE_SUFFICIENT"


def test_group_aspect_comparison_does_not_ask_for_party_size() -> None:
    state = _state(
        "Which is reviewed better for group suitable: A or B?",
        referenced_business_ids=["b1", "b2"],
    )

    assert state.readiness.task_type == "candidate_comparison"
    assert "missing_party_size" not in state.readiness.information_gaps
    assert RuleRouter().choose_action(state).action == "get_business_details"


def test_review_question_uses_known_aspect_profile_without_fake_rag() -> None:
    policy = RuleBasedActionPolicy()
    router = RuleRouter()
    fresh = _state(
        "Is this business quiet according to reviews?",
        referenced_business_ids=["b1"],
    )
    profiled = fresh.model_copy(
        update={
            "observations": [
                _observation(
                    "6",
                    1,
                    "get_business_details",
                    "GET_BUSINESS_PROFILE",
                    {
                        "profiles": [
                            {
                                "business_id": "b1",
                                "aspect_summaries": {
                                    "quiet_environment": {
                                        "status": "known",
                                        "conflict": False,
                                    }
                                },
                            }
                        ]
                    },
                )
            ]
        }
    )

    first = router.choose_action(
        fresh.model_copy(update={"available_actions": policy.allowed_actions(fresh)})
    )
    second = router.choose_action(
        profiled.model_copy(
            update={"available_actions": policy.allowed_actions(profiled)}
        )
    )

    assert first.action == "get_business_details"
    assert first.tool_name == "GET_BUSINESS_PROFILE"
    assert first.arguments == {"business_ids": ["b1"]}
    assert second.action == "return_grounded_answer"
    assert second.reason_code == "STRUCTURED_EVIDENCE_SUFFICIENT"


def test_review_question_abstains_when_structured_evidence_is_unknown() -> None:
    policy = RuleBasedActionPolicy()
    state = _state(
        "Is this business quiet according to reviews?",
        referenced_business_ids=["b1"],
    )
    state = state.model_copy(
        update={
            "observations": [
                _observation(
                    "7",
                    1,
                    "get_business_details",
                    "GET_BUSINESS_PROFILE",
                    {
                        "profiles": [
                            {
                                "business_id": "b1",
                                "aspect_summaries": {
                                    "quiet_environment": {
                                        "status": "unknown",
                                        "conflict": False,
                                    }
                                },
                            }
                        ]
                    },
                )
            ]
        }
    )

    decision = RuleRouter().choose_action(
        state.model_copy(update={"available_actions": policy.allowed_actions(state)})
    )

    assert decision.action == "return_uncertain_answer"
    assert decision.reason_code == "UNSTRUCTURED_EVIDENCE_REQUIRED"


def test_official_policy_route_requires_external_verification() -> None:
    policy = RuleBasedActionPolicy()
    router = RuleRouter()
    fresh = _state(
        "Does this business officially allow pets?",
        referenced_business_ids=["b1"],
    )
    detailed = fresh.model_copy(
        update={
            "observations": [
                _observation(
                    "8",
                    1,
                    "get_business_details",
                    "GET_BUSINESS_DETAILS",
                    {"businesses": [{"business_id": "b1"}]},
                )
            ]
        }
    )

    first = router.choose_action(
        fresh.model_copy(update={"available_actions": policy.allowed_actions(fresh)})
    )
    second = router.choose_action(
        detailed.model_copy(
            update={"available_actions": policy.allowed_actions(detailed)}
        )
    )

    assert first.tool_name == "GET_BUSINESS_DETAILS"
    assert second.action == "return_uncertain_answer"
    assert second.arguments == {
        "reason": "official_verification_required",
        "recommended_official_verification": True,
    }
    assert second.reason_code == "OFFICIAL_VERIFICATION_REQUIRED"


def test_comparison_route_profiles_candidates_then_compares_them() -> None:
    policy = RuleBasedActionPolicy()
    router = RuleRouter()
    fresh = _state(
        "Compare these businesses",
        referenced_business_ids=["b1", "b2"],
    )
    profiled = fresh.model_copy(
        update={
            "observations": [
                _observation(
                    "9",
                    1,
                    "get_business_details",
                    "GET_BUSINESS_PROFILE",
                    {
                        "profiles": [
                            {"business_id": "b1", "aspect_summaries": {}},
                            {"business_id": "b2", "aspect_summaries": {}},
                        ]
                    },
                )
            ]
        }
    )
    compared = profiled.model_copy(
        update={
            "observations": profiled.observations
            + [
                _observation(
                    "a",
                    2,
                    "compare_candidates",
                    "COMPARE_BUSINESSES",
                    {
                        "ranking": ["b2", "b1"],
                        "compared": [
                            {"business_id": "b2"},
                            {"business_id": "b1"},
                        ],
                    },
                )
            ]
        }
    )

    decisions = []
    for state in (fresh, profiled, compared):
        decisions.append(
            router.choose_action(
                state.model_copy(
                    update={"available_actions": policy.allowed_actions(state)}
                )
            )
        )

    assert [item.action for item in decisions] == [
        "get_business_details",
        "compare_candidates",
        "return_grounded_answer",
    ]
    assert [item.tool_name for item in decisions] == [
        "GET_BUSINESS_PROFILE",
        "COMPARE_BUSINESSES",
        None,
    ]
    assert decisions[1].arguments == {"business_ids": ["b1", "b2"]}
    assert decisions[2].arguments == {"business_ids": ["b2", "b1"]}


def test_rule_router_configuration_freezes_the_zero_llm_baseline() -> None:
    config = load_rule_router_config(PROJECT_ROOT / "configs" / "rule_router.yaml")

    assert config.router_version == "1.0.0"
    assert config.agent_version == "step24-rule-agent-v1"
    assert config.display_limit == 3
    assert config.rules_frozen is True
    assert config.llm_enabled is False
    assert config.review_rag_enabled is False
    assert config.official_source_enabled is False
