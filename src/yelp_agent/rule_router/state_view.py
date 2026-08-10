"""Compact, visible-only facts shared by deterministic and model Routers."""

from __future__ import annotations

from typing import Self

from pydantic import Field

from yelp_agent.agent_harness.schema import AgentObservation, AgentState
from yelp_agent.agent_tools.schema import ToolObservation, ToolStatus
from yelp_agent.decision_readiness.schema import (
    InformationGap,
    RankingUncertaintyReason,
    TaskType,
)
from yelp_agent.models import StrictModel


class RemainingBudgetFacts(StrictModel):
    """Execution capacity still available to the next Router decision."""

    steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    semantic_calls: int = Field(ge=0)
    rag_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)


class RouteToolFact(StrictModel):
    """Small status record for one normalized tool observation."""

    tool_name: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    status: ToolStatus
    turn_index: int = Field(ge=1)
    error_code: str | None = None


class RouteFacts(StrictModel):
    """A small Router-facing view derived only from visible Agent state."""

    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_turn: int = Field(ge=1)
    task_type: TaskType
    information_gaps: list[InformationGap]
    conflict_fields: list[str]
    ranking_confidence: float | None = Field(default=None, ge=0, le=1)
    ranking_uncertainty_reasons: list[RankingUncertaintyReason]
    hard_constraints_required: bool
    referenced_business_ids: list[str]
    business_scope_known: bool
    candidate_ids: list[str]
    retrieved_candidate_ids: list[str]
    ranked_business_ids: list[str]
    detailed_business_ids: list[str]
    profiled_business_ids: list[str]
    compared_business_ids: list[str]
    comparison_ranking: list[str]
    candidate_retrieval: RouteToolFact | None = None
    constraint_filter: RouteToolFact | None = None
    hybrid_ranking: RouteToolFact | None = None
    business_details: RouteToolFact | None = None
    business_profiles: RouteToolFact | None = None
    comparison: RouteToolFact | None = None
    last_tool: RouteToolFact | None = None
    remaining: RemainingBudgetFacts

    @classmethod
    def from_state(cls, state: AgentState) -> Self:
        """Hide request, readiness, scope, and accounting representation details."""

        tools = _normalized_tools(state.observations)
        retrieval = _latest_tool(tools, "EXPAND_CANDIDATES")
        constraint_filter = _latest_tool(
            tools,
            "APPLY_CONSTRAINTS",
            turn_index=state.current_turn,
        )
        ranking = _latest_tool(
            tools,
            "GET_HYBRID_RANKING",
            turn_index=state.current_turn,
        )
        details = _latest_tool(tools, "GET_BUSINESS_DETAILS")
        profiles = _latest_tool(tools, "GET_BUSINESS_PROFILE")
        comparison = _latest_tool(
            tools,
            "COMPARE_BUSINESSES",
            turn_index=state.current_turn,
        )
        last_tool = tools[-1] if tools else None
        return cls(
            request_id=state.request.request_id,
            current_turn=state.current_turn,
            task_type=state.readiness.task_type,
            information_gaps=list(state.readiness.information_gaps),
            conflict_fields=list(state.readiness.conflict_fields),
            ranking_confidence=(
                None
                if state.readiness.ranking_confidence is None
                else state.readiness.ranking_confidence.probability_top1_correct
            ),
            ranking_uncertainty_reasons=(
                []
                if state.readiness.ranking_confidence is None
                else list(state.readiness.ranking_confidence.uncertainty_reasons)
            ),
            hard_constraints_required=bool(state.request.hard_constraints),
            referenced_business_ids=list(state.request.referenced_business_ids),
            business_scope_known=state.business_scope_known,
            candidate_ids=list(state.business_scope),
            retrieved_candidate_ids=_string_list(
                _tool_data(retrieval).get("candidate_business_ids")
            ),
            ranked_business_ids=_string_list(
                _tool_data(ranking).get("ranking")
            ),
            detailed_business_ids=_accumulated_record_ids(
                tools,
                tool_name="GET_BUSINESS_DETAILS",
                data_key="businesses",
            ),
            profiled_business_ids=_accumulated_record_ids(
                tools,
                tool_name="GET_BUSINESS_PROFILE",
                data_key="profiles",
            ),
            compared_business_ids=_record_ids(
                _tool_data(comparison).get("compared")
            ),
            comparison_ranking=_string_list(
                _tool_data(comparison).get("ranking")
            ),
            candidate_retrieval=_route_tool_fact(retrieval),
            constraint_filter=_route_tool_fact(constraint_filter),
            hybrid_ranking=_route_tool_fact(ranking),
            business_details=_route_tool_fact(details),
            business_profiles=_route_tool_fact(profiles),
            comparison=_route_tool_fact(comparison),
            last_tool=_route_tool_fact(last_tool),
            remaining=RemainingBudgetFacts(
                steps=max(0, state.budget.max_steps - state.step_count),
                tool_calls=max(
                    0,
                    state.budget.max_tool_calls - state.tool_call_count,
                ),
                semantic_calls=max(
                    0,
                    state.budget.max_semantic_calls - state.semantic_call_count,
                ),
                rag_calls=max(
                    0,
                    state.budget.max_rag_calls - state.rag_call_count,
                ),
                tokens=max(
                    0,
                    state.budget.max_total_tokens
                    - state.input_tokens
                    - state.output_tokens,
                ),
            ),
        )


type _NormalizedTool = tuple[AgentObservation, ToolObservation]


def _normalized_tools(
    observations: list[AgentObservation],
) -> list[_NormalizedTool]:
    normalized: list[_NormalizedTool] = []
    for observation in observations:
        try:
            payload = ToolObservation.model_validate(observation.payload)
        except ValueError:
            continue
        normalized.append((observation, payload))
    return normalized


def _latest_tool(
    tools: list[_NormalizedTool],
    tool_name: str,
    *,
    turn_index: int | None = None,
) -> _NormalizedTool | None:
    for item in reversed(tools):
        observation, payload = item
        if payload.tool_name != tool_name:
            continue
        if turn_index is not None and observation.turn_index != turn_index:
            continue
        return item
    return None


def _route_tool_fact(tool: _NormalizedTool | None) -> RouteToolFact | None:
    if tool is None:
        return None
    observation, payload = tool
    return RouteToolFact(
        tool_name=payload.tool_name,
        status=payload.status,
        turn_index=observation.turn_index,
        error_code=payload.error_code,
    )


def _tool_data(tool: _NormalizedTool | None) -> dict[str, object]:
    return {} if tool is None else dict(tool[1].data)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _record_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        business_id = item.get("business_id")
        if isinstance(business_id, str) and business_id not in result:
            result.append(business_id)
    return result


def _accumulated_record_ids(
    tools: list[_NormalizedTool],
    *,
    tool_name: str,
    data_key: str,
) -> list[str]:
    result: list[str] = []
    for _, payload in tools:
        if payload.tool_name != tool_name or payload.status not in {
            "success",
            "partial",
        }:
            continue
        for business_id in _record_ids(payload.data.get(data_key)):
            if business_id not in result:
                result.append(business_id)
    return result
