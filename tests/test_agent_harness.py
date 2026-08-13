from datetime import datetime
from pathlib import Path

from yelp_agent.agent_benchmark import ScriptedUserTurn, VisibleAgentScenario
from yelp_agent.agent_harness import (
    ActionOutcome,
    AgentDecision,
    AgentHarness,
    AgentSession,
    BenchmarkSessionDriver,
    FakeFallbackHandler,
    HarnessBudget,
    RuleBasedRequestInterpreter,
    ToolResultMetadata,
    UserTurnInput,
    load_agent_harness_config,
)
from yelp_agent.agent_evaluation import (
    ClarificationQuestionTrace,
    EvidenceReference,
    RetrievedEvidenceTrace,
)


class _FixedClock:
    def now_ms(self) -> float:
        return 100.0


class _ReturnPolicy:
    def allowed_actions(self, state):
        return ("return_recommendation",)


class _ReturnRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="return_recommendation",
            arguments={},
            reason_code="READY_TO_FINALIZE",
        )


class _ReturnExecutor:
    def execute(self, state, decision):
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-1", "business-2"],
            recommended_business_ids=["business-1"],
        )


def _scenario() -> VisibleAgentScenario:
    return VisibleAgentScenario(
        scenario_id="a" * 64,
        split="development",
        language="zh-CN",
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2021, 1, 1),
        query_text="推荐一家牛排馆",
    )


def test_harness_runs_one_valid_turn_and_emits_step21_contract() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReturnPolicy(),
        router=_ReturnRouter(),
        executor=_ReturnExecutor(),
        budget=HarnessBudget(
            max_steps=10,
            max_tool_calls=5,
            max_semantic_calls=1,
            max_rag_calls=2,
            max_total_tokens=12_000,
            timeout_ms=90_000,
        ),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.session.status == "completed"
    assert result.run is not None
    assert result.run.scenario_id == "a" * 64
    assert result.run.agent_version == "fake-agent-v1"
    assert result.run.fallback is False
    assert result.run.latency_ms == 0.0
    assert len(result.run.turns) == 1
    turn = result.run.turns[0]
    assert turn.predicted_task_type == "recommendation_request"
    assert turn.actions[0].action == "return_recommendation"
    assert turn.actions[0].reason_code == "READY_TO_FINALIZE"
    assert turn.candidate_ranking == ["business-1", "business-2"]
    assert turn.recommended_business_ids == ["business-1"]
    assert turn.response_kind == "recommendation"


class _GapAwarePolicy:
    def allowed_actions(self, state):
        if "missing_location" in state.readiness.information_gaps:
            return ("ask_clarification",)
        return ("return_recommendation",)


class _FirstAllowedRouter:
    def choose_action(self, state):
        return AgentDecision(
            action=state.available_actions[0],
            arguments={},
            reason_code=(
                "MISSING_LOCATION"
                if state.available_actions[0] == "ask_clarification"
                else "READY_TO_FINALIZE"
            ),
        )


class _ClarifyThenReturnExecutor:
    def execute(self, state, decision):
        if decision.action == "ask_clarification":
            return ActionOutcome(
                status="completed",
                response_kind="clarification",
                clarification_question=ClarificationQuestionTrace(
                    question_text="请告诉我你现在的位置。",
                    requested_information_gaps=["missing_location"],
                ),
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-1", "business-2"],
            recommended_business_ids=["business-1"],
        )


def test_harness_pauses_for_clarification_and_resumes_same_session() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_GapAwarePolicy(),
        router=_FirstAllowedRouter(),
        executor=_ClarifyThenReturnExecutor(),
        clock=_FixedClock(),
    )
    scenario = _scenario().model_copy(
        update={"query_text": "Find a steakhouse within 5 km"}
    )

    paused = harness.start(scenario)

    assert paused.session.status == "awaiting_user"
    assert paused.run is None
    assert len(paused.session.turns) == 1
    assert paused.session.turns[0].response_kind == "clarification"

    resumed = harness.resume(
        paused.session,
        UserTurnInput(
            query_text="I am near Philadelphia City Hall.",
            user_latitude=39.9526,
            user_longitude=-75.1652,
        ),
    )

    assert resumed.session.status == "completed"
    assert resumed.run is not None
    assert len(resumed.run.turns) == 2
    assert resumed.run.turns[1].turn_index == 2
    assert resumed.run.turns[1].response_kind == "recommendation"
    assert "missing_location" not in resumed.session.readiness.information_gaps
    assert resumed.session.request.desired_categories == ["Steakhouses"]


def test_resume_resets_per_turn_token_budget_but_keeps_session_total() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_GapAwarePolicy(),
        router=_FirstAllowedRouter(),
        executor=_ClarifyThenReturnExecutor(),
        budget=HarnessBudget(max_total_tokens=100),
        clock=_FixedClock(),
    )
    paused = harness.start(
        _scenario().model_copy(update={"query_text": "Find a steakhouse within 5 km"})
    )
    previous = paused.session.model_copy(
        update={"input_tokens": 90, "turn_input_tokens": 90}
    )

    resumed = harness.resume(
        previous,
        UserTurnInput(
            query_text="I am near Philadelphia City Hall.",
            user_latitude=39.9526,
            user_longitude=-75.1652,
        ),
    )

    assert resumed.run is not None
    assert resumed.run.fallback is False
    assert resumed.session.input_tokens == 90
    assert resumed.session.turn_input_tokens == 0


class _MultiStepPolicy:
    def allowed_actions(self, state):
        return (
            "retrieve_candidates",
            "rank_candidates",
            "return_recommendation",
        )


class _MultiStepRouter:
    def choose_action(self, state):
        actions = (
            "retrieve_candidates",
            "rank_candidates",
            "return_recommendation",
        )
        return AgentDecision(
            action=actions[state.step_count],
            arguments={"stage": state.step_count},
            reason_code=f"STEP_{state.step_count + 1}",
        )


class _MultiStepExecutor:
    def execute(self, state, decision):
        if decision.action == "retrieve_candidates":
            return ActionOutcome(
                status="completed",
                observation={"candidate_business_ids": ["business-1", "business-2"]},
            )
        if decision.action == "rank_candidates":
            return ActionOutcome(
                status="completed",
                observation={"ranking": ["business-1", "business-2"]},
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-1", "business-2"],
            recommended_business_ids=["business-1"],
        )


def test_harness_runs_multiple_state_changing_actions_in_one_turn() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_MultiStepRouter(),
        executor=_MultiStepExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.session.status == "completed"
    assert result.session.step_count == 3
    assert len(result.session.observations) == 2
    assert result.run is not None
    assert [item.action for item in result.run.turns[0].actions] == [
        "retrieve_candidates",
        "rank_candidates",
        "return_recommendation",
    ]


class _DisallowedRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="return_grounded_answer",
            arguments={},
            reason_code="ROUTER_MISTAKE",
        )


class _MustNotExecute:
    def __init__(self):
        self.call_count = 0

    def execute(self, state, decision):
        self.call_count += 1
        raise AssertionError("a rejected action must not reach the executor")


def test_disallowed_router_action_is_rejected_and_safely_falls_back() -> None:
    executor = _MustNotExecute()
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReturnPolicy(),
        router=_DisallowedRouter(),
        executor=executor,
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert executor.call_count == 0
    assert result.session.status == "fallback"
    assert result.run is not None
    assert result.run.fallback is True
    assert result.run.fallback_reason == "disallowed_action:return_grounded_answer"
    assert [item.action for item in result.run.turns[0].actions] == [
        "return_grounded_answer",
        "safe_fallback",
    ]
    assert result.run.turns[0].actions[0].status == "rejected"
    assert result.run.turns[0].response_kind == "fallback"


class _ToolRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="retrieve_candidates",
            arguments={"limit": 100},
            reason_code="CANDIDATES_REQUIRED",
            tool_name="FAKE_CANDIDATE_RETRIEVAL",
            tool_kind="deterministic",
        )


class _ExplodingExecutor:
    def execute(self, state, decision):
        raise RuntimeError("simulated tool failure")


def test_tool_exception_is_traced_and_batch_safe_fallback_is_returned() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_ToolRouter(),
        executor=_ExplodingExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.session.status == "fallback"
    assert result.run is not None
    assert result.run.fallback_reason == "tool_exception:RuntimeError"
    assert result.session.tool_call_count == 1
    assert len(result.run.turns[0].tool_calls) == 1
    tool_call = result.run.turns[0].tool_calls[0]
    assert tool_call.tool_name == "FAKE_CANDIDATE_RETRIEVAL"
    assert tool_call.status == "failed"
    assert tool_call.latency_ms == 0.0


def test_tool_budget_is_checked_before_executor_is_called() -> None:
    executor = _MustNotExecute()
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_ToolRouter(),
        executor=executor,
        budget=HarnessBudget(max_tool_calls=0),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert executor.call_count == 0
    assert result.run is not None
    assert result.run.fallback_reason == "max_tool_calls_exceeded"
    assert result.run.turns[0].actions[0].status == "rejected"
    assert result.session.tool_call_count == 0


class _CountingObservationExecutor:
    def __init__(self):
        self.call_count = 0

    def execute(self, state, decision):
        self.call_count += 1
        return ActionOutcome(
            status="completed",
            observation={"result": "unchanged"},
        )


def test_identical_tool_call_is_blocked_before_second_execution() -> None:
    executor = _CountingObservationExecutor()
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_ToolRouter(),
        executor=executor,
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert executor.call_count == 1
    assert result.run is not None
    assert result.run.fallback_reason == "duplicate_tool_call"
    assert [item.status for item in result.run.turns[0].actions] == [
        "completed",
        "rejected",
        "completed",
    ]
    assert result.session.tool_call_count == 1


def test_max_steps_stops_an_infinite_nonterminal_loop() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_MultiStepRouter(),
        executor=_MultiStepExecutor(),
        budget=HarnessBudget(max_steps=1),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "max_steps_exceeded"
    assert [item.action for item in result.run.turns[0].actions] == [
        "retrieve_candidates",
        "safe_fallback",
    ]


class _NoProgressExecutor:
    def execute(self, state, decision):
        return ActionOutcome(status="completed")


def test_nonterminal_action_without_observation_falls_back() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_ToolRouter(),
        executor=_NoProgressExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "no_state_progress"
    assert result.run.turns[0].actions[-1].action == "safe_fallback"


class _SemanticThenReturnRouter:
    def choose_action(self, state):
        if state.step_count == 0:
            return AgentDecision(
                action="retrieve_candidates",
                arguments={"query": "steak"},
                reason_code="SEMANTIC_RECALL_REQUIRED",
                tool_name="FAKE_SEMANTIC_RECALL",
                tool_kind="semantic",
            )
        return AgentDecision(
            action="return_recommendation",
            arguments={},
            reason_code="READY_TO_FINALIZE",
        )


class _SemanticThenReturnExecutor:
    def execute(self, state, decision):
        if decision.tool_name is not None:
            return ActionOutcome(
                status="completed",
                observation={"candidate_business_ids": ["business-1"]},
                business_scope=["business-1"],
                tool_result=ToolResultMetadata(
                    input_tokens=120,
                    output_tokens=30,
                    cost_usd=0.002,
                ),
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-1"],
            recommended_business_ids=["business-1"],
        )


def test_tool_token_cost_and_scope_are_carried_into_final_run() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_SemanticThenReturnRouter(),
        executor=_SemanticThenReturnExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback is False
    assert result.run.input_tokens == 120
    assert result.run.output_tokens == 30
    assert result.run.cost_usd == 0.002
    assert result.session.semantic_call_count == 1
    assert result.session.business_scope == ["business-1"]
    assert result.run.turns[0].tool_calls[0].input_tokens == 120


class _ReviewPolicy:
    def allowed_actions(self, state):
        return ("retrieve_business_reviews", "return_grounded_answer")


class _UnlockedReviewRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="retrieve_business_reviews",
            arguments={"aspect": "quiet_environment"},
            reason_code="REVIEW_EVIDENCE_REQUIRED",
            tool_name="FAKE_REVIEW_SEARCH",
            tool_kind="review_rag",
        )


def test_review_search_requires_a_locked_business_before_execution() -> None:
    executor = _MustNotExecute()
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReviewPolicy(),
        router=_UnlockedReviewRouter(),
        executor=executor,
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert executor.call_count == 0
    assert result.run is not None
    assert result.run.fallback_reason == "review_business_not_locked"


class _LockedReviewRouter:
    def choose_action(self, state):
        return AgentDecision(
            action="retrieve_business_reviews",
            arguments={
                "business_id": "business-1",
                "aspect": "quiet_environment",
            },
            reason_code="REVIEW_EVIDENCE_REQUIRED",
            tool_name="FAKE_REVIEW_SEARCH",
            tool_kind="review_rag",
        )


class _MismatchedEvidenceExecutor:
    def execute(self, state, decision):
        return ActionOutcome(
            status="completed",
            observation={"review_ids": ["review-2"]},
            tool_result=ToolResultMetadata(
                retrieved_evidence=[
                    RetrievedEvidenceTrace(
                        rank=1,
                        evidence=EvidenceReference(
                            business_id="business-2",
                            source_type="review",
                            review_id="review-2",
                        ),
                        relevance_score=0.9,
                    )
                ]
            ),
        )


def test_review_evidence_from_another_business_is_rejected() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReviewPolicy(),
        router=_LockedReviewRouter(),
        executor=_MismatchedEvidenceExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "evidence_business_mismatch"
    assert result.run.turns[0].tool_calls[0].status == "failed"


class _OutOfScopeExecutor(_SemanticThenReturnExecutor):
    def execute(self, state, decision):
        if decision.tool_name is not None:
            return super().execute(state, decision)
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-2"],
            recommended_business_ids=["business-2"],
        )


def test_final_ranking_cannot_escape_tool_provided_business_scope() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_SemanticThenReturnRouter(),
        executor=_OutOfScopeExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "business_id_out_of_scope"


class _SequenceClock:
    def __init__(self, values):
        self._values = iter(values)
        self._last = 0.0

    def now_ms(self):
        try:
            self._last = float(next(self._values))
        except StopIteration:
            pass
        return self._last


def test_tool_that_exceeds_wall_clock_budget_is_traced_then_falls_back() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_ToolRouter(),
        executor=_CountingObservationExecutor(),
        budget=HarnessBudget(timeout_ms=50),
        clock=_SequenceClock([0, 0, 0, 100, 100, 100]),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "timeout_exceeded"
    assert result.run.turns[0].tool_calls[0].latency_ms == 100.0
    assert result.run.turns[0].tool_calls[0].status == "failed"


def test_token_budget_excess_after_semantic_call_forces_fallback() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_MultiStepPolicy(),
        router=_SemanticThenReturnRouter(),
        executor=_SemanticThenReturnExecutor(),
        budget=HarnessBudget(max_total_tokens=100),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "max_total_tokens_exceeded"
    assert result.run.input_tokens == 120
    assert result.run.output_tokens == 30


def test_unknown_runtime_task_type_is_recorded_honestly() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReturnPolicy(),
        router=_ReturnRouter(),
        executor=_ReturnExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario().model_copy(update={"query_text": "hello"}))

    assert result.run is not None
    assert result.run.turns[0].predicted_task_type == "unknown"


def test_fixed_dependencies_produce_byte_identical_run_json() -> None:
    def build_harness():
        return AgentHarness(
            agent_version="fake-agent-v1",
            interpreter=RuleBasedRequestInterpreter(),
            action_policy=_MultiStepPolicy(),
            router=_MultiStepRouter(),
            executor=_MultiStepExecutor(),
            clock=_FixedClock(),
        )

    first = build_harness().start(_scenario())
    second = build_harness().start(_scenario())

    assert first.run is not None
    assert second.run is not None
    assert first.run.model_dump_json() == second.run.model_dump_json()


def test_checked_in_harness_config_matches_step22_limits() -> None:
    config = load_agent_harness_config(
        Path(__file__).parents[1] / "configs" / "agent_harness.yaml"
    )

    assert config.agent_version == "step22-controlled-harness-v1"
    assert config.budget.max_steps == 10
    assert config.budget.max_tool_calls == 5
    assert config.budget.max_total_tokens == 12_000
    assert config.budget.timeout_ms == 90_000


def test_injected_fallback_runs_once_and_can_return_safe_ranking() -> None:
    fallback = FakeFallbackHandler(["business-1", "business-2"])
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReturnPolicy(),
        router=_DisallowedRouter(),
        executor=_MustNotExecute(),
        fallback_handler=fallback,
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert fallback.call_count == 1
    assert fallback.reasons == ["disallowed_action:return_grounded_answer"]
    assert result.run is not None
    assert result.run.turns[0].candidate_ranking == [
        "business-1",
        "business-2",
    ]


def test_paused_session_can_be_serialized_reloaded_and_resumed() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_GapAwarePolicy(),
        router=_FirstAllowedRouter(),
        executor=_ClarifyThenReturnExecutor(),
        clock=_FixedClock(),
    )
    paused = harness.start(
        _scenario().model_copy(
            update={"query_text": "Find a steakhouse within 5 km"}
        )
    )
    reloaded = AgentSession.from_json(paused.session.to_json())

    resumed = harness.resume(
        reloaded,
        UserTurnInput(
            query_text="I am at City Hall.",
            user_latitude=39.9526,
            user_longitude=-75.1652,
        ),
    )

    assert resumed.run is not None
    assert len(resumed.run.turns) == 2


def test_public_runtime_state_has_no_hidden_ground_truth_fields() -> None:
    forbidden = {
        "ground_truth",
        "target_business_id",
        "acceptable_business_ids",
        "hidden_intent",
        "scripted_user_turns",
    }

    assert forbidden.isdisjoint(AgentSession.model_fields)
    assert forbidden.isdisjoint(AgentDecision.model_fields)
    assert forbidden.isdisjoint(ActionOutcome.model_fields)


class _HardConstraintPolicy:
    def allowed_actions(self, state):
        return (
            "retrieve_candidates",
            "apply_hard_constraints",
            "return_recommendation",
        )


class _HardConstraintRouter:
    def choose_action(self, state):
        actions = (
            "retrieve_candidates",
            "apply_hard_constraints",
            "return_recommendation",
        )
        return AgentDecision(
            action=actions[state.step_count],
            arguments={},
            reason_code=f"HARD_CONSTRAINT_STEP_{state.step_count + 1}",
        )


class _ReintroducingFilteredBusinessExecutor:
    def execute(self, state, decision):
        if decision.action == "retrieve_candidates":
            return ActionOutcome(
                status="completed",
                observation={"candidate_business_ids": ["business-1", "business-2"]},
                business_scope=["business-1", "business-2"],
            )
        if decision.action == "apply_hard_constraints":
            return ActionOutcome(
                status="completed",
                observation={"eligible_business_ids": ["business-1"]},
                business_scope=["business-1"],
            )
        return ActionOutcome(
            status="completed",
            response_kind="recommendation",
            candidate_ranking=["business-2"],
            recommended_business_ids=["business-2"],
        )


def test_business_removed_by_hard_constraints_cannot_be_reintroduced() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_HardConstraintPolicy(),
        router=_HardConstraintRouter(),
        executor=_ReintroducingFilteredBusinessExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback_reason == "business_id_out_of_scope"
    assert result.session.business_scope == ["business-1"]


def test_new_retrieval_may_replace_previous_turn_business_scope() -> None:
    from yelp_agent.agent_harness.validation import outcome_violation

    state = AgentSession.model_validate(
        AgentHarness(
            agent_version="fake-agent-v1",
            interpreter=RuleBasedRequestInterpreter(),
            action_policy=_HardConstraintPolicy(),
            router=_HardConstraintRouter(),
            executor=_ReintroducingFilteredBusinessExecutor(),
            clock=_FixedClock(),
        ).start(_scenario()).session
    ).model_copy(
        update={
            "status": "running",
            "business_scope": ["old-business"],
            "business_scope_known": True,
        }
    )
    decision = AgentDecision(
        action="retrieve_candidates",
        arguments={},
        reason_code="CANDIDATES_REQUIRED",
        tool_name="EXPAND_CANDIDATES",
        tool_kind="deterministic",
    )
    outcome = ActionOutcome(
        status="completed",
        observation={"candidate_business_ids": ["new-business"]},
        business_scope=["new-business"],
    )

    assert outcome_violation(state, decision, outcome) is None


def test_benchmark_driver_releases_hidden_reply_only_after_trigger_action() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_GapAwarePolicy(),
        router=_FirstAllowedRouter(),
        executor=_ClarifyThenReturnExecutor(),
        clock=_FixedClock(),
    )
    scripted_turn = ScriptedUserTurn(
        turn_index=2,
        trigger_action="ask_clarification",
        query_text="I am at Philadelphia City Hall.",
        expected_task_type="recommendation_request",
        expected_information_gaps=[],
        added_conditions=[],
        state_updates={
            "user_latitude": 39.9526,
            "user_longitude": -75.1652,
        },
        rejected_business_ids=[],
        expected_allowed_actions=["return_recommendation"],
    )

    driven = BenchmarkSessionDriver().drive(
        harness,
        _scenario().model_copy(
            update={"query_text": "Find a steakhouse within 5 km"}
        ),
        [scripted_turn],
    )

    assert driven.stop_reason == "completed"
    assert driven.released_turn_indices == [2]
    assert driven.result.run is not None
    assert len(driven.result.run.turns) == 2


def test_benchmark_driver_releases_feedback_after_a_completed_recommendation() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_ReturnPolicy(),
        router=_ReturnRouter(),
        executor=_ReturnExecutor(),
        clock=_FixedClock(),
    )
    scripted_turn = ScriptedUserTurn(
        turn_index=2,
        trigger_action="return_recommendation",
        query_text="That is too far. Please suggest another option.",
        expected_task_type="feedback_refinement",
        expected_information_gaps=[],
        added_conditions=[],
        state_updates={},
        rejected_business_ids=["business-1"],
        expected_allowed_actions=["return_recommendation"],
    )

    driven = BenchmarkSessionDriver().drive(
        harness,
        _scenario(),
        [scripted_turn],
    )

    assert driven.stop_reason == "completed"
    assert driven.released_turn_indices == [2]
    assert driven.result.run is not None
    assert len(driven.result.run.turns) == 2


class _EmptyScopePolicy:
    def allowed_actions(self, state):
        return (
            "retrieve_candidates",
            "apply_hard_constraints",
            "return_uncertain_answer",
        )


class _EmptyScopeRouter:
    def choose_action(self, state):
        actions = (
            "retrieve_candidates",
            "apply_hard_constraints",
            "return_uncertain_answer",
        )
        return AgentDecision(
            action=actions[state.step_count],
            arguments={},
            reason_code=f"EMPTY_SCOPE_STEP_{state.step_count + 1}",
        )


class _EmptyScopeExecutor:
    def execute(self, state, decision):
        if decision.action == "retrieve_candidates":
            return ActionOutcome(
                status="completed",
                observation={"candidate_business_ids": ["business-1"]},
                business_scope=["business-1"],
            )
        if decision.action == "apply_hard_constraints":
            return ActionOutcome(
                status="completed",
                observation={"eligible_business_ids": []},
                business_scope=[],
            )
        return ActionOutcome(
            status="completed",
            response_kind="uncertain_answer",
        )


def test_hard_constraints_may_legitimately_produce_empty_known_scope() -> None:
    harness = AgentHarness(
        agent_version="fake-agent-v1",
        interpreter=RuleBasedRequestInterpreter(),
        action_policy=_EmptyScopePolicy(),
        router=_EmptyScopeRouter(),
        executor=_EmptyScopeExecutor(),
        clock=_FixedClock(),
    )

    result = harness.start(_scenario())

    assert result.run is not None
    assert result.run.fallback is False
    assert result.run.turns[0].response_kind == "uncertain_answer"
    assert result.session.business_scope_known is True
    assert result.session.business_scope == []
