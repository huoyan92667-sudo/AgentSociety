"""Controlled state-action loop for visible Agent scenarios."""

from __future__ import annotations

import time

from yelp_agent.agent_benchmark import VisibleAgentScenario
from yelp_agent.agent_evaluation.schema import AgentScenarioRun
from yelp_agent.query import QueryParseInput

from .fallback import StaticFallbackHandler
from .interfaces import (
    ActionExecutor,
    ActionRouter,
    AllowedActionPolicy,
    Clock,
    FallbackHandler,
    RequestInterpreter,
)
from .schema import (
    ActionOutcome,
    AgentDecision,
    AgentSession,
    HarnessBudget,
    HarnessResult,
    UserTurnInput,
)
from .state_transition import record_decision, record_execution
from .trace_recorder import TurnTraceRecorder
from .validation import decision_rejection, outcome_violation, pre_loop_violation


class SystemClock:
    def now_ms(self) -> float:
        return time.perf_counter() * 1000.0


class AgentHarness:
    """Run a safe state-action loop using only visible scenario inputs."""

    def __init__(
        self,
        *,
        agent_version: str,
        interpreter: RequestInterpreter,
        action_policy: AllowedActionPolicy,
        router: ActionRouter,
        executor: ActionExecutor,
        fallback_handler: FallbackHandler | None = None,
        budget: HarnessBudget | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not agent_version.strip():
            raise ValueError("agent_version cannot be blank")
        self._agent_version = agent_version
        self._interpreter = interpreter
        self._action_policy = action_policy
        self._router = router
        self._executor = executor
        self._fallback_handler = fallback_handler or StaticFallbackHandler()
        self._budget = budget or HarnessBudget()
        self._clock = clock or SystemClock()

    def start(self, scenario: VisibleAgentScenario) -> HarnessResult:
        """Create state from visible data and run until pause or termination."""

        started_at_ms = self._clock.now_ms()
        interpretation = self._interpreter.interpret(
            QueryParseInput(
                user_id=scenario.user_id,
                session_id=scenario.session_id,
                cutoff_time=scenario.cutoff_time,
                query_text=scenario.query_text,
                user_latitude=scenario.user_latitude,
                user_longitude=scenario.user_longitude,
                referenced_business_ids=scenario.referenced_business_ids,
            )
        )
        state = AgentSession(
            scenario_id=scenario.scenario_id,
            split=scenario.split,
            language=scenario.language,
            agent_version=self._agent_version,
            user_id=scenario.user_id,
            session_id=scenario.session_id,
            cutoff_time=scenario.cutoff_time,
            request=interpretation.request,
            readiness=interpretation.readiness,
            memory=interpretation.memory,
            budget=self._budget,
            semantic_call_count=interpretation.semantic_calls,
            memory_provider_call_count=int(
                interpretation.memory_extraction is not None
                and interpretation.memory_extraction.provider_called
            ),
            memory_fallback_count=int(
                interpretation.memory_extraction is not None
                and interpretation.memory_extraction.status == "rule_fallback"
            ),
            input_tokens=interpretation.input_tokens or 0,
            output_tokens=interpretation.output_tokens or 0,
            turn_input_tokens=interpretation.input_tokens or 0,
            turn_output_tokens=interpretation.output_tokens or 0,
            token_usage_observed=interpretation.input_tokens is not None,
            cost_usd=interpretation.cost_usd or 0,
            started_at_ms=started_at_ms,
        )
        return self._run_loop(state)

    def resume(
        self,
        session: AgentSession,
        user_turn: UserTurnInput,
    ) -> HarnessResult:
        """Resume only a session that explicitly paused for user information."""

        if session.status != "awaiting_user":
            raise ValueError("only an awaiting_user session can be resumed")
        return self._continue_with_user_turn(session, user_turn)

    def follow_up(
        self,
        session: AgentSession,
        user_turn: UserTurnInput,
    ) -> HarnessResult:
        """Start a new turn after a completed response in the same session."""

        if session.status != "completed":
            raise ValueError("only a completed session can receive a follow-up")
        return self._continue_with_user_turn(session, user_turn)

    def _continue_with_user_turn(
        self,
        session: AgentSession,
        user_turn: UserTurnInput,
    ) -> HarnessResult:
        interpretation = self._interpreter.interpret(
            QueryParseInput(
                user_id=session.user_id,
                session_id=session.session_id,
                cutoff_time=session.cutoff_time,
                query_text=user_turn.query_text,
                user_latitude=user_turn.user_latitude,
                user_longitude=user_turn.user_longitude,
                referenced_business_ids=user_turn.referenced_business_ids,
            ),
            previous_session=session,
        )
        state = session.model_copy(
            update={
                "current_turn": session.current_turn + 1,
                "status": "running",
                "request": interpretation.request,
                "readiness": interpretation.readiness,
                "memory": interpretation.memory,
                "available_actions": [],
                # Step/tool/model limits are per user turn. Durable traces keep
                # session-wide accounting for evaluation without preventing a
                # legitimate later feedback turn from running.
                "step_count": 0,
                "tool_call_count": 0,
                "semantic_call_count": interpretation.semantic_calls,
                "memory_provider_call_count": session.memory_provider_call_count
                + int(
                    interpretation.memory_extraction is not None
                    and interpretation.memory_extraction.provider_called
                ),
                "memory_fallback_count": session.memory_fallback_count
                + int(
                    interpretation.memory_extraction is not None
                    and interpretation.memory_extraction.status == "rule_fallback"
                ),
                "rag_call_count": 0,
                "input_tokens": session.input_tokens
                + (interpretation.input_tokens or 0),
                "output_tokens": session.output_tokens
                + (interpretation.output_tokens or 0),
                "turn_input_tokens": interpretation.input_tokens or 0,
                "turn_output_tokens": interpretation.output_tokens or 0,
                "token_usage_observed": session.token_usage_observed
                or interpretation.input_tokens is not None,
                "cost_usd": session.cost_usd + (interpretation.cost_usd or 0),
            }
        )
        return self._run_loop(state)

    def _run_loop(self, state: AgentSession) -> HarnessResult:
        recorder = TurnTraceRecorder()
        fallback_reason: str | None = None

        while state.status == "running":
            violation = pre_loop_violation(
                state,
                elapsed_ms=self._elapsed_ms(state),
            )
            if violation is not None:
                fallback_reason = violation
                state, outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                )
                recorder.absorb(outcome)
                break

            allowed, policy_failure = self._allowed_actions(state)
            if policy_failure is not None:
                fallback_reason = policy_failure
                state, outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                )
                recorder.absorb(outcome)
                break
            state = state.model_copy(update={"available_actions": allowed})

            decision, router_failure = self._choose_action(state)
            if router_failure is not None:
                fallback_reason = router_failure
                state, outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                )
                recorder.absorb(outcome)
                break
            assert decision is not None

            rejection = decision_rejection(state, decision, allowed)
            if rejection is not None:
                fallback_reason, reason_code = rejection
                recorder.add_action(
                    decision,
                    status="rejected",
                    reason_code=reason_code,
                )
                state = record_decision(state, decision)
                state, outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                )
                recorder.absorb(outcome)
                break

            if decision.action == "safe_fallback":
                fallback_reason = "router_requested_fallback"
                recorder.add_action(decision, status="completed")
                state = record_decision(state, decision)
                state, outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                    append_action=False,
                )
                recorder.absorb(outcome)
                break

            action_step_index = recorder.next_step_index
            outcome, tool_latency_ms = self._execute(state, decision)
            result_violation = outcome_violation(state, decision, outcome)
            if result_violation is not None:
                outcome = self._failed_copy(outcome, result_violation)
            if self._elapsed_ms(state) >= state.budget.timeout_ms:
                outcome = self._failed_copy(outcome, "timeout_exceeded")

            recorder.add_action(
                decision,
                status="completed" if outcome.status == "completed" else "failed",
            )
            if decision.tool_name is not None:
                recorder.add_tool_call(
                    state=state,
                    decision=decision,
                    outcome=outcome,
                    action_step_index=action_step_index,
                    latency_ms=tool_latency_ms,
                )
            state = record_execution(
                state=state,
                decision=decision,
                outcome=outcome,
                action_step_index=action_step_index,
            )
            recorder.absorb(outcome)

            if (
                state.turn_input_tokens + state.turn_output_tokens
                > state.budget.max_total_tokens
            ):
                fallback_reason = "max_total_tokens_exceeded"
            elif outcome.status == "failed":
                fallback_reason = outcome.failure_reason or "action_failed"
            else:
                fallback_reason = None
            if fallback_reason is not None:
                state, fallback_outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                )
                recorder.absorb(fallback_outcome)
                break

            if outcome.response_kind == "clarification":
                state = state.model_copy(update={"status": "awaiting_user"})
            elif outcome.response_kind != "none":
                state = state.model_copy(update={"status": "completed"})
            elif outcome.observation is None:
                fallback_reason = "no_state_progress"
                state, fallback_outcome = self._perform_fallback(
                    state,
                    recorder,
                    fallback_reason,
                )
                recorder.absorb(fallback_outcome)
                break

        return self._finish(
            state=state,
            recorder=recorder,
            fallback_reason=fallback_reason,
        )

    def _allowed_actions(
        self,
        state: AgentSession,
    ) -> tuple[list[str], str | None]:
        try:
            allowed = list(self._action_policy.allowed_actions(state))
        except Exception as exc:
            return [], f"action_policy_exception:{type(exc).__name__}"
        if not allowed:
            return [], "no_available_actions"
        return allowed, None

    def _choose_action(
        self,
        state: AgentSession,
    ) -> tuple[AgentDecision | None, str | None]:
        try:
            decision = self._router.choose_action(state)
            if not isinstance(decision, AgentDecision):
                decision = AgentDecision.model_validate(decision)
        except Exception as exc:
            return None, f"router_exception:{type(exc).__name__}"
        return decision, None

    def _execute(
        self,
        state: AgentSession,
        decision: AgentDecision,
    ) -> tuple[ActionOutcome, float]:
        started_ms = self._clock.now_ms()
        try:
            outcome = ActionOutcome.model_validate(
                self._executor.execute(state, decision)
            )
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                failure_reason = "tool_timeout"
            else:
                prefix = "tool_exception" if decision.tool_name else "action_exception"
                failure_reason = f"{prefix}:{type(exc).__name__}"
            outcome = ActionOutcome(
                status="failed",
                failure_reason=failure_reason,
            )
        return outcome, max(0.0, self._clock.now_ms() - started_ms)

    def _perform_fallback(
        self,
        state: AgentSession,
        recorder: TurnTraceRecorder,
        reason: str,
        *,
        append_action: bool = True,
    ) -> tuple[AgentSession, ActionOutcome]:
        """Run fallback exactly once and force a valid terminal result."""

        decision = AgentDecision(
            action="safe_fallback",
            arguments={"reason": reason},
            reason_code="SAFE_FALLBACK",
        )
        try:
            outcome = ActionOutcome.model_validate(
                self._fallback_handler.fallback(state, reason)
            )
            if outcome.status != "completed" or outcome.response_kind != "fallback":
                raise ValueError("fallback handler returned a non-fallback outcome")
            if outcome_violation(state, decision, outcome) is not None:
                raise ValueError("fallback handler escaped the current business scope")
        except Exception:
            outcome = StaticFallbackHandler().fallback(state, reason)
        if append_action:
            recorder.add_action(decision, status="completed")
            state = state.model_copy(
                update={"action_history": state.action_history + [decision]}
            )
        return state.model_copy(update={"status": "fallback"}), outcome

    def _finish(
        self,
        *,
        state: AgentSession,
        recorder: TurnTraceRecorder,
        fallback_reason: str | None,
    ) -> HarnessResult:
        elapsed_ms = self._elapsed_ms(state)
        state = state.model_copy(
            update={
                "turns": state.turns + [recorder.build_turn(state)],
                "elapsed_ms": elapsed_ms,
                "fallback_reason": fallback_reason,
            }
        )
        if state.status == "awaiting_user":
            return HarnessResult(session=state, run=None)
        return HarnessResult(
            session=state,
            run=AgentScenarioRun(
                scenario_id=state.scenario_id,
                agent_version=state.agent_version,
                turns=state.turns,
                fallback=state.status == "fallback",
                fallback_reason=state.fallback_reason,
                latency_ms=elapsed_ms,
                input_tokens=(
                    state.input_tokens if state.token_usage_observed else None
                ),
                output_tokens=(
                    state.output_tokens if state.token_usage_observed else None
                ),
                cost_usd=state.cost_usd if state.cost_usd > 0 else None,
            ),
        )

    @staticmethod
    def _failed_copy(outcome: ActionOutcome, reason: str) -> ActionOutcome:
        return ActionOutcome(
            status="failed",
            tool_result=outcome.tool_result,
            model_result=outcome.model_result,
            failure_reason=reason,
        )

    def _elapsed_ms(self, state: AgentSession) -> float:
        return max(0.0, self._clock.now_ms() - state.started_at_ms)
