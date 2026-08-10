"""Code-enforced action, budget, scope, and evidence safety rules."""

from __future__ import annotations

from collections.abc import Sequence

from .schema import ActionOutcome, AgentDecision, AgentSession
from .trace_recorder import call_signature


_TERMINAL_RESPONSE_BY_ACTION = {
    "ask_clarification": "clarification",
    "return_recommendation": "recommendation",
    "return_grounded_answer": "grounded_answer",
    "return_uncertain_answer": "uncertain_answer",
}


def pre_loop_violation(state: AgentSession, *, elapsed_ms: float) -> str | None:
    if elapsed_ms >= state.budget.timeout_ms:
        return "timeout_exceeded"
    if state.step_count >= state.budget.max_steps:
        return "max_steps_exceeded"
    if state.semantic_call_count > state.budget.max_semantic_calls:
        return "max_semantic_calls_exceeded"
    if state.input_tokens + state.output_tokens > state.budget.max_total_tokens:
        return "max_total_tokens_exceeded"
    return None


def decision_rejection(
    state: AgentSession,
    decision: AgentDecision,
    allowed: Sequence[str],
) -> tuple[str, str] | None:
    if decision.action not in allowed:
        return f"disallowed_action:{decision.action}", "ACTION_NOT_ALLOWED"
    signature = call_signature(state, decision)
    if signature is not None and signature in state.executed_call_signatures:
        return "duplicate_tool_call", "DUPLICATE_TOOL_CALL"
    budget_reason = _decision_budget_violation(state, decision)
    if budget_reason is not None:
        return budget_reason, "BUDGET_EXCEEDED"
    if decision.action == "retrieve_business_reviews" and not (
        decision.arguments.get("business_id")
        or decision.arguments.get("business_ids")
    ):
        return "review_business_not_locked", "BUSINESS_NOT_LOCKED"
    return None


def _decision_budget_violation(
    state: AgentSession,
    decision: AgentDecision,
) -> str | None:
    if decision.tool_name is None:
        return None
    if state.tool_call_count >= state.budget.max_tool_calls:
        return "max_tool_calls_exceeded"
    if (
        decision.tool_kind == "semantic"
        and state.semantic_call_count >= state.budget.max_semantic_calls
    ):
        return "max_semantic_calls_exceeded"
    if (
        decision.tool_kind == "review_rag"
        and state.rag_call_count >= state.budget.max_rag_calls
    ):
        return "max_rag_calls_exceeded"
    if (
        decision.tool_kind == "semantic"
        and state.input_tokens + state.output_tokens
        >= state.budget.max_total_tokens
    ):
        return "max_total_tokens_exceeded"
    return None


def outcome_violation(
    state: AgentSession,
    decision: AgentDecision,
    outcome: ActionOutcome,
) -> str | None:
    expected_response = _TERMINAL_RESPONSE_BY_ACTION.get(decision.action)
    if expected_response is not None and outcome.status == "completed":
        if outcome.response_kind != expected_response:
            return "invalid_response_for_action"
    elif (
        decision.action != "safe_fallback"
        and outcome.status == "completed"
        and outcome.response_kind != "none"
    ):
        return "nonterminal_action_returned_response"
    if decision.tool_name is None and outcome.tool_result is not None:
        return "unexpected_tool_result"
    evidence = (
        [] if outcome.tool_result is None else outcome.tool_result.retrieved_evidence
    )
    requested_ids: set[str] = set()
    business_id = decision.arguments.get("business_id")
    if isinstance(business_id, str):
        requested_ids.add(business_id)
    business_ids = decision.arguments.get("business_ids")
    if isinstance(business_ids, list):
        requested_ids.update(str(item) for item in business_ids)
    if evidence and (
        not requested_ids
        or any(item.evidence.business_id not in requested_ids for item in evidence)
    ):
        return "evidence_business_mismatch"
    current_scope = set(state.business_scope)
    new_scope = (
        None if outcome.business_scope is None else set(outcome.business_scope)
    )
    if decision.action == "apply_hard_constraints" and state.business_scope_known:
        if new_scope is None or not new_scope.issubset(current_scope):
            return "invalid_hard_constraint_scope"
    elif (
        new_scope is not None
        and state.business_scope_known
        and new_scope != current_scope
    ):
        return "unauthorized_business_scope_change"
    scope_known = new_scope is not None or state.business_scope_known
    effective_scope = current_scope if new_scope is None else new_scope
    returned_ids = set(outcome.candidate_ranking + outcome.recommended_business_ids)
    if scope_known and not returned_ids.issubset(effective_scope):
        return "business_id_out_of_scope"
    return None
