"""The only functions that mutate a copied AgentSession state."""

from __future__ import annotations

from .schema import ActionOutcome, AgentDecision, AgentSession
from .trace_recorder import build_observation, call_signature


def record_decision(state: AgentSession, decision: AgentDecision) -> AgentSession:
    """Record a Router choice that was rejected before execution."""

    return state.model_copy(
        update={
            "action_history": state.action_history + [decision],
            "step_count": state.step_count + 1,
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
    if outcome.status == "completed" and outcome.observation is not None:
        observations = observations + [
            build_observation(
                state=state,
                action_step_index=action_step_index,
                decision=decision,
                payload=outcome.observation,
            )
        ]
    metadata = outcome.tool_result
    input_tokens = state.input_tokens
    output_tokens = state.output_tokens
    token_usage_observed = state.token_usage_observed
    cost_usd = state.cost_usd
    if metadata is not None and metadata.input_tokens is not None:
        input_tokens += metadata.input_tokens
        output_tokens += metadata.output_tokens or 0
        token_usage_observed = True
    if metadata is not None and metadata.cost_usd is not None:
        cost_usd += metadata.cost_usd
    signature = call_signature(state, decision)
    signatures = state.executed_call_signatures
    if signature is not None:
        signatures = signatures + [signature]
    return state.model_copy(
        update={
            "action_history": state.action_history + [decision],
            "observations": observations,
            "executed_call_signatures": signatures,
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
            + int(decision.tool_kind == "semantic"),
            "rag_call_count": state.rag_call_count
            + int(decision.tool_kind == "review_rag"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_usage_observed": token_usage_observed,
            "cost_usd": cost_usd,
        }
    )
