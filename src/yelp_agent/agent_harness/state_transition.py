"""The only functions that mutate a copied AgentSession state."""

from __future__ import annotations

from .schema import ActionOutcome, AgentDecision, AgentSession
from .trace_recorder import build_observation, call_signature
from yelp_agent.session_memory.reducer import record_memory_observation


def apply_routed_task_type(
    state: AgentSession,
    decision: AgentDecision,
) -> AgentSession:
    """Apply only validated, enum-bounded Router semantic corrections."""

    task_type = decision.routed_task_type
    gaps = decision.routed_information_gaps
    if (
        (task_type is None or task_type == state.readiness.task_type)
        and (gaps is None or gaps == state.readiness.information_gaps)
    ):
        return state
    memory = state.memory
    if memory is not None:
        memory_updates: dict[str, object] = {}
        if task_type is not None:
            memory_updates["current_task_type"] = task_type
        if gaps is not None:
            memory_updates["information_gaps"] = gaps
        memory = memory.model_copy(update=memory_updates)
    readiness_updates: dict[str, object] = {}
    if task_type is not None:
        readiness_updates["task_type"] = task_type
    if gaps is not None:
        readiness_updates["information_gaps"] = gaps
    return state.model_copy(
        update={
            "readiness": state.readiness.model_copy(update=readiness_updates),
            "memory": memory,
        }
    )


def record_decision(state: AgentSession, decision: AgentDecision) -> AgentSession:
    """Record a Router choice that was rejected before execution."""

    accounting = _router_accounting(state, decision)
    return state.model_copy(
        update={
            "action_history": state.action_history + [decision],
            "step_count": state.step_count + 1,
            **accounting,
        }
    )


def record_execution(
    *,
    state: AgentSession,
    decision: AgentDecision,
    outcome: ActionOutcome,
    action_step_index: int,
) -> AgentSession:
    """Apply one validated executor result and its accounting atomically."""

    observations = state.observations
    if outcome.observation is not None:
        observations = observations + [
            build_observation(
                state=state,
                action_step_index=action_step_index,
                decision=decision,
                payload=outcome.observation,
            )
        ]
    metadata = outcome.tool_result or outcome.model_result
    input_tokens = state.input_tokens
    output_tokens = state.output_tokens
    turn_input_tokens = state.turn_input_tokens
    turn_output_tokens = state.turn_output_tokens
    token_usage_observed = state.token_usage_observed
    cost_usd = state.cost_usd
    if metadata is not None and metadata.input_tokens is not None:
        input_tokens += metadata.input_tokens
        output_tokens += metadata.output_tokens or 0
        turn_input_tokens += metadata.input_tokens
        turn_output_tokens += metadata.output_tokens or 0
        token_usage_observed = True
    if metadata is not None and metadata.cost_usd is not None:
        cost_usd += metadata.cost_usd
    router = decision.router_trace
    if router is not None and router.input_tokens is not None:
        input_tokens += router.input_tokens
        output_tokens += router.output_tokens or 0
        turn_input_tokens += router.input_tokens
        turn_output_tokens += router.output_tokens or 0
        token_usage_observed = True
    signature = call_signature(state, decision)
    signatures = state.executed_call_signatures
    if signature is not None:
        signatures = signatures + [signature]
    memory = record_memory_observation(
        state.memory,
        turn_index=state.current_turn,
        business_scope=outcome.business_scope,
        presented_business_ids=(
            outcome.recommended_business_ids
            if outcome.response_kind == "recommendation"
            else None
        ),
    )
    return state.model_copy(
        update={
            "action_history": state.action_history + [decision],
            "observations": observations,
            "executed_call_signatures": signatures,
            "memory": memory,
            "business_scope": (
                state.business_scope
                if outcome.business_scope is None
                else outcome.business_scope
            ),
            "business_scope_known": state.business_scope_known
            or outcome.business_scope is not None,
            "step_count": state.step_count + 1,
            "tool_call_count": state.tool_call_count
            + int(decision.tool_name is not None),
            "semantic_call_count": state.semantic_call_count
            + int(decision.tool_kind == "semantic")
            + int(router is not None and router.provider_called)
            + int(
                outcome.model_result is not None
                and outcome.model_result.provider_called
            ),
            "rag_call_count": state.rag_call_count
            + int(decision.tool_kind == "review_rag"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "turn_input_tokens": turn_input_tokens,
            "turn_output_tokens": turn_output_tokens,
            "token_usage_observed": token_usage_observed,
            "cost_usd": cost_usd,
        }
    )


def _router_accounting(
    state: AgentSession,
    decision: AgentDecision,
) -> dict[str, object]:
    trace = decision.router_trace
    if trace is None:
        return {}
    input_tokens = state.input_tokens
    output_tokens = state.output_tokens
    turn_input_tokens = state.turn_input_tokens
    turn_output_tokens = state.turn_output_tokens
    observed = state.token_usage_observed
    if trace.input_tokens is not None:
        input_tokens += trace.input_tokens
        output_tokens += trace.output_tokens or 0
        turn_input_tokens += trace.input_tokens
        turn_output_tokens += trace.output_tokens or 0
        observed = True
    return {
        "semantic_call_count": state.semantic_call_count
        + int(trace.provider_called),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turn_input_tokens": turn_input_tokens,
        "turn_output_tokens": turn_output_tokens,
        "token_usage_observed": observed,
    }
