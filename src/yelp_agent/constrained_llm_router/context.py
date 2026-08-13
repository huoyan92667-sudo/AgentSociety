"""Create one deterministic, compact and ground-truth-free Router context."""

from __future__ import annotations

from yelp_agent.agent_harness.schema import AgentState
from yelp_agent.rule_router.state_view import RouteFacts
from yelp_agent.session_memory.effective_request import compile_effective_request

from .schema import RouterChoice, RouterDecisionContext, RouterStateSummary


def build_router_context(
    state: AgentState,
    choices: list[RouterChoice],
) -> RouterDecisionContext:
    facts = RouteFacts.from_state(state)
    effective = (
        compile_effective_request(state.memory).model_dump(
            mode="json", exclude_computed_fields=True
        )
        if state.memory is not None
        else {
            "schema_version": 0,
            "request": state.request.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "task_type": state.readiness.task_type,
            "relative_preferences": [],
            "rejected_business_ids": [],
            "clarification_answers": {},
            "latest_user_query": state.request.query_text,
        }
    )
    latest_query = (
        state.memory.recent_turns[-1].query_text
        if state.memory is not None and state.memory.recent_turns
        else state.request.query_text
    )
    current_tools = [
        item.tool_name
        for item in (
            facts.candidate_retrieval,
            facts.constraint_filter,
            facts.hybrid_ranking,
            facts.semantic_match,
            facts.cross_encoder_match,
            facts.semantic_ranking,
            facts.business_details,
            facts.business_profiles,
            facts.review_search,
            facts.evidence_aggregation,
            facts.comparison,
        )
        if item is not None and item.turn_index == state.current_turn
    ]
    return RouterDecisionContext(
        scenario_id=state.scenario_id,
        turn_index=state.current_turn,
        language=state.language,
        latest_user_query=latest_query,
        effective_request=effective,
        state=RouterStateSummary(
            task_type=facts.task_type,
            information_gaps=list(facts.information_gaps),
            candidate_count=len(facts.candidate_ids),
            ranked_candidate_count=len(facts.ranked_business_ids),
            detailed_business_count=len(facts.detailed_business_ids),
            profiled_business_count=len(facts.profiled_business_ids),
            referenced_business_count=len(facts.referenced_business_ids),
            review_evidence_count=facts.review_evidence_count,
            business_scope_known=facts.business_scope_known,
            hard_constraints_required=facts.hard_constraints_required,
            current_turn_tools=current_tools,
            recent_actions=[
                f"{item.action}:{item.tool_name or 'terminal'}:{item.reason_code}"
                for item in state.action_history[-8:]
            ],
            remaining_steps=facts.remaining.steps,
            remaining_tool_calls=facts.remaining.tool_calls,
            remaining_semantic_calls=facts.remaining.semantic_calls,
            remaining_rag_calls=facts.remaining.rag_calls,
            remaining_tokens=facts.remaining.tokens,
        ),
        choices=choices,
    )
