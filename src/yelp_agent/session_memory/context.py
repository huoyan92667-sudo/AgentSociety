"""Deterministic compaction of canonical memory for tools and model prompts."""

from __future__ import annotations

from yelp_agent.query.schema import RequestCondition

from .schema import RouterMemoryContext, SessionMemory


def compact_memory(memory: SessionMemory) -> RouterMemoryContext:
    """Return only current facts; raw observations and hidden labels never enter."""

    request = memory.current_request
    return RouterMemoryContext(
        revision=memory.revision,
        task_type=memory.current_task_type,
        hard_constraints=[_condition_payload(item) for item in request.hard_constraints],
        soft_preferences=[_condition_payload(item) for item in request.soft_preferences],
        information_gaps=list(memory.information_gaps),
        rejected_business_ids=list(memory.rejected_business_ids),
        last_presented_business_ids=list(memory.last_presented_business_ids),
        current_business_scope=list(memory.current_business_scope),
        business_scope_known=memory.business_scope_known,
        clarification_answers=dict(memory.clarification_answers),
        relative_preferences=list(memory.relative_preferences),
        semantic_summary=memory.semantic_summary,
        recent_turn_summaries=[
            f"turn={item.turn_index}; task={item.task_type}; "
            f"accepted={','.join(item.accepted_changes) or 'none'}"
            for item in memory.recent_turns[-4:]
        ],
    )


def _condition_payload(condition: RequestCondition) -> dict[str, object]:
    return {
        "field": condition.field,
        "operator": condition.operator,
        "value": condition.value,
        "importance": condition.importance,
        "enforcement": condition.enforcement,
        "unknown_policy": condition.unknown_policy,
    }
