from __future__ import annotations

import json
from datetime import UTC, datetime

from yelp_agent.agent.llm import LLMCallResult
from yelp_agent.agent_harness import (
    ActionOutcome,
    AgentSession,
    HarnessBudget,
    RuleBasedRequestInterpreter,
)
from yelp_agent.agent_harness.state_transition import (
    apply_routed_task_type,
    record_execution,
)
from yelp_agent.controlled_llm import (
    ControlledJSONCaller,
    ControlledLLMUsageLedger,
    FakeChatGenerator,
    SqliteControlledLLMCache,
)
from yelp_agent.constrained_llm_router import (
    ConstrainedActionPolicy,
    ConstrainedDecisionBuilder,
    ConstrainedLLMRouter,
    ConstrainedLLMRouterConfig,
    build_router_context,
)
from yelp_agent.query import QueryParseInput
from yelp_agent.rule_router import RuleRouter


def _state(
    text: str,
    *,
    referenced: list[str] | None = None,
) -> AgentSession:
    interpreted = RuleBasedRequestInterpreter().interpret(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 1, tzinfo=UTC),
            query_text=text,
            referenced_business_ids=referenced or [],
        )
    )
    return AgentSession(
        scenario_id="a" * 64,
        split="development",
        language="en-US",
        agent_version="test",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2024, 1, 1, tzinfo=UTC),
        request=interpreted.request,
        readiness=interpreted.readiness,
        budget=HarnessBudget(max_semantic_calls=12),
        started_at_ms=0,
    )


def _config(**updates: object) -> ConstrainedLLMRouterConfig:
    base = ConstrainedLLMRouterConfig(
        agent_version="test-router",
        prompt_version="test-router-v1",
        enabled=True,
        temperature=0,
        timeout_seconds=90,
        max_retries=0,
        max_output_tokens=300,
        thinking="disabled",
        response_format_json=True,
        cache_relative_path="cache.sqlite3",
        minimum_confidence=0.6,
        repair_invalid_output_once=True,
        bypass_single_choice=True,
        maximum_choices=10,
        tuning_split="development",
        validation_used_for_selection=False,
    )
    return base.model_copy(update=updates)


def _success(payload: object, *, input_tokens: int = 80, output_tokens: int = 20):
    return LLMCallResult(
        status="success",
        content=json.dumps(payload),
        model="fake-model",
        latency_ms=12,
        attempt_count=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _router(tmp_path, state: AgentSession, results: list[LLMCallResult], **updates):
    fallback = RuleRouter(review_rag_enabled=True)
    builder = ConstrainedDecisionBuilder(fallback, maximum_choices=10)
    generator = FakeChatGenerator(results)
    ledger = ControlledLLMUsageLedger()
    cache = SqliteControlledLLMCache(tmp_path / "router.sqlite3")
    caller = ControlledJSONCaller(
        generator=generator,
        model_name="fake-model",
        cache=cache,
        ledger=ledger,
    )
    router = ConstrainedLLMRouter(
        config=_config(**updates),
        caller=caller,
        builder=builder,
        fallback_router=fallback,
    )
    return router, builder, generator, ledger, cache


def test_single_safe_choice_bypasses_the_provider(tmp_path) -> None:
    state = _state("Find a restaurant for me")
    router, builder, generator, _, cache = _router(tmp_path, state, [])
    try:
        assert len(builder.build(state)) == 1
        decision = router.choose_action(state)
    finally:
        cache.close()

    assert decision.action == "retrieve_candidates"
    assert decision.router_trace is not None
    assert decision.router_trace.router_kind == "single_choice_bypass"
    assert generator.call_count == 0


def test_model_selects_only_a_code_owned_complete_decision(tmp_path) -> None:
    state = _state(
        "Is this place quiet, or should you recommend another one?",
        referenced=["business-1"],
    )
    fallback = RuleRouter(review_rag_enabled=True)
    builder = ConstrainedDecisionBuilder(fallback)
    pairs = builder.build(state)
    retrieval = next(
        (choice, decision)
        for choice, decision in pairs
        if decision.action == "retrieve_candidates"
    )
    router, _, generator, _, cache = _router(
        tmp_path,
        state,
        [
            _success(
                {
                    "choice_id": retrieval[0].choice_id,
                    "confidence": 0.94,
                    "reason_code": "USER_REQUESTS_NEW_RECOMMENDATION",
                }
            )
        ],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()

    assert decision.model_copy(update={"router_trace": None}) == retrieval[1]
    assert decision.arguments == retrieval[1].arguments
    assert decision.router_trace is not None
    assert decision.router_trace.router_kind == "constrained_llm"
    assert decision.router_trace.total_tokens == 100
    assert generator.call_count == 1


def test_invalid_json_gets_one_repair_then_succeeds(tmp_path) -> None:
    state = _state("Tell me about this place or find another", referenced=["b1"])
    builder = ConstrainedDecisionBuilder(RuleRouter(review_rag_enabled=True))
    target_choice, target = builder.build(state)[0]
    router, _, generator, _, cache = _router(
        tmp_path,
        state,
        [
            LLMCallResult(
                status="success",
                content="not json",
                model="fake-model",
                latency_ms=5,
                attempt_count=1,
                input_tokens=50,
                output_tokens=4,
                total_tokens=54,
            ),
            _success(
                {
                    "choice_id": target_choice.choice_id,
                    "confidence": 0.9,
                    "reason_code": "REPAIRED_SELECTION",
                },
                input_tokens=60,
                output_tokens=10,
            ),
        ],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()

    assert decision.model_copy(update={"router_trace": None}) == target
    assert decision.router_trace is not None
    assert decision.router_trace.attempt_count == 2
    assert decision.router_trace.total_tokens == 124
    assert generator.call_count == 2


def test_unknown_choice_repairs_once_then_falls_back_to_rule(tmp_path) -> None:
    state = _state("Tell me about this place or find another", referenced=["b1"])
    invalid = {
        "choice_id": "choice_deadbeefdead",
        "confidence": 0.99,
        "reason_code": "INVENTED_CHOICE",
    }
    router, _, generator, _, cache = _router(
        tmp_path,
        state,
        [_success(invalid), _success(invalid)],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()

    expected = RuleRouter(review_rag_enabled=True).choose_action(state).model_copy(
        update={"routed_task_type": state.readiness.task_type}
    )
    assert decision.model_copy(update={"router_trace": None}) == expected
    assert decision.router_trace is not None
    assert decision.router_trace.router_kind == "rule_fallback"
    assert decision.router_trace.status == "invalid_output"
    assert generator.call_count == 2


def test_low_confidence_does_not_retry_and_falls_back(tmp_path) -> None:
    state = _state("Tell me about this place or find another", referenced=["b1"])
    choice = ConstrainedDecisionBuilder(RuleRouter(review_rag_enabled=True)).build(state)[0][0]
    router, _, generator, _, cache = _router(
        tmp_path,
        state,
        [
            _success(
                {
                    "choice_id": choice.choice_id,
                    "confidence": 0.2,
                    "reason_code": "UNCERTAIN",
                }
            )
        ],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()

    assert decision.router_trace is not None
    assert decision.router_trace.status == "low_confidence"
    assert generator.call_count == 1


def test_provider_failure_does_not_trigger_format_repair(tmp_path) -> None:
    state = _state("Tell me about this place or find another", referenced=["b1"])
    router, _, generator, _, cache = _router(
        tmp_path,
        state,
        [
            LLMCallResult(
                status="failure",
                content=None,
                model="fake-model",
                latency_ms=90_000,
                attempt_count=1,
                failure_reason="timeout",
            )
        ],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()

    assert decision.router_trace is not None
    assert decision.router_trace.router_kind == "rule_fallback"
    assert decision.router_trace.status == "provider_failure"
    assert generator.call_count == 1


def test_disabled_provider_falls_back_without_retry(tmp_path) -> None:
    state = _state("Tell me about this place or find another", referenced=["b1"])
    router, _, generator, _, cache = _router(
        tmp_path,
        state,
        [
            LLMCallResult(
                status="disabled",
                content=None,
                model=None,
                latency_ms=0,
                attempt_count=0,
                failure_reason="llm_disabled",
            )
        ],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()

    assert decision.router_trace is not None
    assert decision.router_trace.status == "disabled"
    assert decision.router_trace.provider_called is False
    assert decision.router_trace.failure_reason == "llm_disabled"
    assert generator.call_count == 1


def test_router_usage_is_counted_by_agent_state_transition(tmp_path) -> None:
    state = _state("Tell me about this place or find another", referenced=["b1"])
    fallback = RuleRouter(review_rag_enabled=True)
    target_choice, target = next(
        pair
        for pair in ConstrainedDecisionBuilder(fallback).build(state)
        if pair[1].action == "retrieve_candidates"
    )
    router, _, _, _, cache = _router(
        tmp_path,
        state,
        [
            _success(
                {
                    "choice_id": target_choice.choice_id,
                    "confidence": 0.95,
                    "reason_code": "NEW_RECOMMENDATION_REQUIRED",
                }
            )
        ],
    )
    try:
        decision = router.choose_action(state)
    finally:
        cache.close()
    assert decision.model_copy(update={"router_trace": None}) == target

    updated = record_execution(
        state=state,
        decision=decision,
        outcome=ActionOutcome(
            status="completed",
            observation={"tool_name": "EXPAND_CANDIDATES", "status": "success"},
            business_scope=["b1", "b2"],
        ),
        action_step_index=1,
    )

    assert updated.input_tokens == 80
    assert updated.output_tokens == 20
    assert updated.semantic_call_count == 1
    assert updated.token_usage_observed is True


def test_context_contains_effective_request_but_no_hidden_labels() -> None:
    state = _state("Compare these places", referenced=["b1", "b2"])
    builder = ConstrainedDecisionBuilder(RuleRouter(review_rag_enabled=True))
    pairs = builder.build(state)
    context = build_router_context(state, [choice for choice, _ in pairs])
    payload = context.model_dump_json()

    assert context.latest_user_query == "Compare these places"
    assert "ground_truth" not in payload.casefold()
    assert "acceptable_business_ids" not in payload
    assert all("arguments" not in choice.model_dump() for choice in context.choices)


def test_policy_and_router_share_the_same_choice_builder() -> None:
    state = _state("Compare these places", referenced=["b1", "b2"])
    builder = ConstrainedDecisionBuilder(RuleRouter(review_rag_enabled=True))
    expected = tuple(
        dict.fromkeys(decision.action for _, decision in builder.build(state))
    )

    assert ConstrainedActionPolicy(builder).allowed_actions(state) == expected


def test_known_scope_choices_never_bind_an_external_business() -> None:
    state = _state("Compare these places", referenced=["b1", "b2"]).model_copy(
        update={"business_scope_known": True, "business_scope": ["b1", "b2"]}
    )
    pairs = ConstrainedDecisionBuilder(RuleRouter(review_rag_enabled=True)).build(state)

    for _, decision in pairs:
        values = decision.arguments.get("business_ids", [])
        if isinstance(values, list):
            assert set(values).issubset({"b1", "b2"})


def test_selected_alternative_task_type_changes_following_agent_state() -> None:
    state = _state(
        "Is this place quiet, or should you recommend another one?",
        referenced=["b1"],
    )
    decision = next(
        item
        for _, item in ConstrainedDecisionBuilder(
            RuleRouter(review_rag_enabled=True)
        ).build(state)
        if item.routed_task_type == "recommendation_request"
    )

    updated = apply_routed_task_type(state, decision)

    assert updated.readiness.task_type == "recommendation_request"
