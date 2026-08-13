from __future__ import annotations

from datetime import datetime

from yelp_agent.query.schema import RecommendationRequest, RequestCondition
from yelp_agent.agent_tools.request_context import (
    query_text_from_tool_context,
    rejected_business_ids_from_tool_context,
    request_from_tool_context,
)
from yelp_agent.agent_tools.schema import ToolExecutionContext
from yelp_agent.session_memory.effective_request import compile_effective_request
from yelp_agent.session_memory.schema import (
    MemoryExtractionTrace,
    MemoryTurnRecord,
    RelativePreference,
    SessionMemory,
)


def _condition(
    field: str,
    operator: str,
    value: str | int | float | bool,
    *,
    evidence: str,
    enforcement: str,
    importance: str,
) -> RequestCondition:
    return RequestCondition(
        field=field,
        operator=operator,
        value=value,
        importance=importance,
        enforcement=enforcement,
        explicit=True,
        confidence=1.0,
        evidence_span=evidence,
        evidence_start=0,
        evidence_end=len(evidence),
        source="semantic_model",
        unknown_policy="exclude" if enforcement == "filter" else "not_applicable",
    )


def _memory() -> SessionMemory:
    request = RecommendationRequest(
        request_id="a" * 64,
        user_id="user-1",
        session_id="session-1",
        cutoff_time=datetime(2022, 1, 1),
        query_text=(
            "I used to want Chinese food. Today only show Burgers under price level 2."
        ),
        intent="feedback_refinement",
        conditions=[
            _condition(
                "category",
                "includes",
                "Burgers",
                evidence="Burgers",
                enforcement="filter",
                importance="mandatory",
            ),
            _condition(
                "price_level",
                "less_than_or_equal",
                2,
                evidence="price level 2",
                enforcement="rank",
                importance="strong",
            ),
        ],
        party_size=2,
        missing_fields=[],
        referenced_business_ids=["shown-1"],
        parser_version="session-memory-v1",
    )
    return SessionMemory(
        session_id=request.session_id,
        user_id=request.user_id,
        cutoff_time=request.cutoff_time,
        revision=3,
        current_request=request,
        current_task_type="feedback_refinement",
        rejected_business_ids=["rejected-1"],
        relative_preferences=[
            RelativePreference(
                field="price",
                direction="lower",
                evidence_span="cheaper than the first one",
                confidence=0.98,
            )
        ],
        relative_preference_references={"price": "shown-1"},
        clarification_answers={"missing_party_size": 2},
        semantic_summary="Old summary that still says Chinese food.",
        recent_turns=[
            MemoryTurnRecord(
                turn_index=3,
                query_text="Make it cheaper than the first one.",
                task_type="feedback_refinement",
                accepted_changes=["relative:price:lower"],
                extraction=MemoryExtractionTrace(
                    status="rule_fallback",
                    extractor="rule",
                    prompt_version="test",
                ),
            )
        ],
    )


def test_compiler_builds_one_complete_request_without_stale_raw_text() -> None:
    effective = compile_effective_request(_memory())

    assert effective.revision == 3
    assert effective.request.request_id == effective.effective_request_id
    assert effective.request.desired_categories == ["Burgers"]
    assert effective.rejected_business_ids == ["rejected-1"]
    assert effective.relative_preferences[0].reference_business_id == "shown-1"
    assert effective.relative_preferences[0].source_turn_index == 3
    assert "Burgers" in effective.request.query_text
    assert "price_level less_than_or_equal 2" in effective.request.query_text
    assert "Prefer a lower price than the referenced result" in effective.request.query_text
    assert "Chinese" not in effective.request.query_text


def test_effective_id_depends_on_active_state_not_stale_summary_text() -> None:
    memory = _memory()
    first = compile_effective_request(memory)
    second = compile_effective_request(
        memory.model_copy(update={"semantic_summary": "A different stale summary."})
    )

    assert first.effective_request_id == second.effective_request_id
    assert first.request.query_text == second.request.query_text


def test_tool_context_prefers_the_effective_request_over_legacy_turn_text() -> None:
    memory = _memory()
    effective = compile_effective_request(memory)
    context = ToolExecutionContext(
        request_id=memory.current_request.request_id,
        user_id=memory.user_id,
        cutoff_time=memory.cutoff_time,
        action="retrieve_candidates",
        state_snapshot={
            "request": memory.current_request.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "effective_request": effective.model_dump(
                mode="json", exclude_computed_fields=True
            ),
        },
    )

    assert request_from_tool_context(context).request_id == effective.effective_request_id
    assert query_text_from_tool_context(context) == effective.request.query_text
    assert rejected_business_ids_from_tool_context(context) == {"rejected-1"}
