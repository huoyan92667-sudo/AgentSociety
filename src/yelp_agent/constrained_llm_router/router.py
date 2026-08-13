"""Deep constrained Router: one state in, one validated decision out."""

from __future__ import annotations

from yelp_agent.agent_evaluation.schema import RouterDecisionTrace
from yelp_agent.agent_harness.schema import AgentDecision, AgentState
from yelp_agent.controlled_llm.gateway import ControlledJSONCaller
from yelp_agent.controlled_llm.schema import ControlledLLMCallTrace
from yelp_agent.decision_readiness.schema import TaskType
from yelp_agent.rule_router.router import RuleRouter

from .choices import ConstrainedDecisionBuilder
from .config import ConstrainedLLMRouterConfig
from .context import build_router_context
from .prompt import router_messages
from .schema import RouterChoice, RouterDecisionContext, RouterModelOutput


class ConstrainedLLMRouter:
    """Let a model select only a complete decision already authorized by code."""

    def __init__(
        self,
        *,
        config: ConstrainedLLMRouterConfig,
        caller: ControlledJSONCaller,
        builder: ConstrainedDecisionBuilder,
        fallback_router: RuleRouter,
    ) -> None:
        self._config = config
        self._caller = caller
        self._builder = builder
        self._fallback = fallback_router

    def choose_action(self, state: AgentState) -> AgentDecision:
        pairs = self._builder.build(state)
        choices = [item[0] for item in pairs]
        decisions = {item[0].choice_id: item[1] for item in pairs}
        if len(pairs) == 1 and self._config.bypass_single_choice:
            choice, decision = pairs[0]
            return decision.model_copy(
                update={
                    "router_trace": RouterDecisionTrace(
                        router_kind="single_choice_bypass",
                        status="skipped",
                        allowed_choice_ids=[choice.choice_id],
                        selected_choice_id=choice.choice_id,
                        input_task_type=state.readiness.task_type,
                        selected_task_type=decision.routed_task_type,
                        input_information_gaps=list(
                            state.readiness.information_gaps
                        ),
                        selected_information_gaps=(
                            list(state.readiness.information_gaps)
                            if decision.routed_information_gaps is None
                            else decision.routed_information_gaps
                        ),
                    )
                }
            )

        context = build_router_context(state, choices)
        traces: list[ControlledLLMCallTrace] = []
        output = self._call(state, context, repair=False, traces=traces)
        failure = self._selection_failure(output, decisions)
        if (
            output is None
            and traces
            and traces[-1].status in {"disabled", "provider_failure"}
        ):
            failure = traces[-1].failure_reason or traces[-1].status
        if (
            failure is not None
            and self._config.repair_invalid_output_once
            and (
                failure == "choice_not_allowed"
                or (traces and traces[-1].status == "invalid_output")
            )
        ):
            output = self._call(state, context, repair=True, traces=traces)
            failure = self._selection_failure(output, decisions)
        if failure is None and output is not None:
            decision = decisions[output.choice_id]
            return decision.model_copy(
                update={
                    "router_trace": _trace(
                        router_kind="constrained_llm",
                        status="success",
                        choices=list(decisions),
                        selected=output.choice_id,
                        proposed=output.choice_id,
                        confidence=output.confidence,
                        selected_task_type=decision.routed_task_type,
                        input_task_type=state.readiness.task_type,
                        selected_information_gaps=(
                            list(state.readiness.information_gaps)
                            if decision.routed_information_gaps is None
                            else decision.routed_information_gaps
                        ),
                        input_information_gaps=list(
                            state.readiness.information_gaps
                        ),
                        traces=traces,
                    )
                }
            )
        return self._rule_fallback(
            state,
            pairs=pairs,
            traces=traces,
            output=output,
            failure=failure or "provider_failure",
        )

    def _call(
        self,
        state: AgentState,
        context: RouterDecisionContext,
        *,
        repair: bool,
        traces: list[ControlledLLMCallTrace],
    ) -> RouterModelOutput | None:
        result = self._caller.call(
            capability="router_decision",
            prompt_version=(
                f"{self._config.prompt_version}-repair"
                if repair
                else self._config.prompt_version
            ),
            input_payload={
                "repair": repair,
                "context": context.model_dump(mode="json"),
            },
            messages=router_messages(context, repair=repair),
            output_model=RouterModelOutput,
            context_id=state.scenario_id,
            turn_index=state.current_turn,
        )
        traces.append(result.trace)
        return result.output

    def _selection_failure(
        self,
        output: RouterModelOutput | None,
        decisions: dict[str, AgentDecision],
    ) -> str | None:
        if output is None:
            return "invalid_output"
        if output.choice_id not in decisions:
            return "choice_not_allowed"
        if output.confidence < self._config.minimum_confidence:
            return "low_confidence"
        return None

    def _rule_fallback(
        self,
        state: AgentState,
        *,
        pairs: tuple[tuple[RouterChoice, AgentDecision], ...],
        traces: list[ControlledLLMCallTrace],
        output: RouterModelOutput | None,
        failure: str,
    ) -> AgentDecision:
        fallback = self._fallback.choose_action(state).model_copy(
            update={"routed_task_type": state.readiness.task_type}
        )
        by_key = {
            _decision_identity(decision): choice.choice_id
            for choice, decision in pairs
        }
        selected = by_key.get(_decision_identity(fallback))
        if selected is None:
            choice, fallback = self._builder.build(state)[0]
            selected = choice.choice_id
        status = (
            "low_confidence"
            if failure == "low_confidence"
            else "disabled"
            if traces and traces[-1].status == "disabled"
            else "provider_failure"
            if traces and traces[-1].status == "provider_failure"
            else "invalid_output"
        )
        return fallback.model_copy(
            update={
                "router_trace": _trace(
                    router_kind="rule_fallback",
                    status=status,
                    choices=[choice.choice_id for choice, _ in pairs],
                    selected=selected,
                    proposed=None if output is None else output.choice_id,
                    confidence=None if output is None else output.confidence,
                    selected_task_type=fallback.routed_task_type,
                    input_task_type=state.readiness.task_type,
                    selected_information_gaps=(
                        list(state.readiness.information_gaps)
                        if fallback.routed_information_gaps is None
                        else fallback.routed_information_gaps
                    ),
                    input_information_gaps=list(state.readiness.information_gaps),
                    traces=traces,
                    failure=failure,
                )
            }
        )


def _trace(
    *,
    router_kind: str,
    status: str,
    choices: list[str],
    selected: str,
    proposed: str | None = None,
    confidence: float | None = None,
    selected_task_type: TaskType | None = None,
    input_task_type: TaskType | None = None,
    selected_information_gaps: list[str] | None = None,
    input_information_gaps: list[str] | None = None,
    traces: list[ControlledLLMCallTrace],
    failure: str | None = None,
) -> RouterDecisionTrace:
    known_usage = bool(traces) and all(
        item.input_tokens is not None
        and item.output_tokens is not None
        and item.total_tokens is not None
        for item in traces
        if item.provider_called
    )
    called = [item for item in traces if item.provider_called]
    return RouterDecisionTrace(
        router_kind=router_kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        allowed_choice_ids=choices,
        proposed_choice_id=proposed,
        selected_choice_id=selected,
        selection_confidence=confidence,
        input_task_type=input_task_type,
        selected_task_type=selected_task_type,
        input_information_gaps=input_information_gaps or [],
        selected_information_gaps=selected_information_gaps or [],
        model=next((item.model for item in reversed(traces) if item.model), None),
        provider_called=bool(called),
        cache_hit=bool(traces) and any(item.cache_hit for item in traces),
        latency_ms=sum(item.latency_ms for item in called),
        attempt_count=sum(item.attempt_count for item in traces),
        input_tokens=(
            sum(item.input_tokens or 0 for item in called)
            if called and known_usage
            else None
        ),
        output_tokens=(
            sum(item.output_tokens or 0 for item in called)
            if called and known_usage
            else None
        ),
        total_tokens=(
            sum(item.total_tokens or 0 for item in called)
            if called and known_usage
            else None
        ),
        failure_reason=failure,
        call_ids=[item.call_id for item in traces],
    )


def _decision_identity(decision: AgentDecision) -> tuple[object, ...]:
    return (
        decision.action,
        decision.reason_code,
        decision.tool_name,
        decision.tool_kind,
        decision.routed_task_type,
        tuple(decision.routed_information_gaps or []),
        str(sorted(decision.arguments.items())),
    )
