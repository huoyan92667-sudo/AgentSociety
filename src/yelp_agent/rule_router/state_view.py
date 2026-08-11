"""Compact, visible-only facts shared by deterministic and model Routers."""

from __future__ import annotations

from typing import Self

from pydantic import Field

from yelp_agent.agent_harness.schema import AgentObservation, AgentState
from yelp_agent.agent_tools.schema import ToolObservation, ToolStatus
from yelp_agent.cross_encoder import fuse_ranking_and_cross_encoder
from yelp_agent.decision_readiness.schema import (
    InformationGap,
    RankingUncertaintyReason,
    TaskType,
)
from yelp_agent.models import StrictModel
from yelp_agent.reviews.schema import ASPECT_NAMES, AspectName
from yelp_agent.semantic_embedding import fuse_hybrid_and_semantic


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
    requested_aspects: list[AspectName]
    known_aspects_by_business: dict[str, list[AspectName]]
    conflicting_aspects_by_business: dict[str, list[AspectName]]
    structured_evidence_sufficient: bool
    structured_evidence_conflict: bool
    review_evidence_count: int = Field(ge=0)
    review_evidence_business_ids: list[str]
    review_evidence_conflict: bool
    explicit_uncertainty_request: bool
    feedback_applied: bool
    reject_previous_recommendation: bool
    previous_recommended_business_ids: list[str]
    hard_constraints_required: bool
    referenced_business_ids: list[str]
    business_scope_known: bool
    candidate_ids: list[str]
    retrieved_candidate_ids: list[str]
    ranked_business_ids: list[str]
    semantic_ranks_by_business: dict[str, int]
    cross_encoder_ranks_by_business: dict[str, int]
    detailed_business_ids: list[str]
    profiled_business_ids: list[str]
    compared_business_ids: list[str]
    comparison_ranking: list[str]
    candidate_retrieval: RouteToolFact | None = None
    constraint_filter: RouteToolFact | None = None
    hybrid_ranking: RouteToolFact | None = None
    semantic_match: RouteToolFact | None = None
    cross_encoder_match: RouteToolFact | None = None
    business_details: RouteToolFact | None = None
    business_profiles: RouteToolFact | None = None
    comparison: RouteToolFact | None = None
    review_search: RouteToolFact | None = None
    last_tool: RouteToolFact | None = None
    remaining: RemainingBudgetFacts

    def embedding_ranking(self, *, fusion_alpha: float) -> list[str]:
        """Return the frozen Step-25 ranking from visible evidence only."""

        return fuse_hybrid_and_semantic(
            self.ranked_business_ids,
            self.semantic_ranks_by_business,
            alpha=fusion_alpha,
        )

    def final_ranking(
        self,
        *,
        fusion_alpha: float,
        cross_encoder_beta: float = 0.0,
    ) -> list[str]:
        """Apply Step-25 then Step-26 fusion, preserving every unscored tail."""

        return fuse_ranking_and_cross_encoder(
            self.embedding_ranking(fusion_alpha=fusion_alpha),
            self.cross_encoder_ranks_by_business,
            beta=cross_encoder_beta,
        )

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
        semantic_match = _latest_tool(
            tools,
            "COMPUTE_EMBEDDING_MATCH",
            turn_index=state.current_turn,
        )
        cross_encoder_match = _latest_tool(
            tools,
            "COMPUTE_CROSS_ENCODER_MATCH",
            turn_index=state.current_turn,
        )
        details = _latest_tool(tools, "GET_BUSINESS_DETAILS")
        profiles = _latest_tool(tools, "GET_BUSINESS_PROFILE")
        review_search = _latest_tool(
            tools,
            "SEARCH_BUSINESS_REVIEWS",
            turn_index=state.current_turn,
        )
        review_hits = _review_hits(review_search)
        known_aspects, conflicting_aspects = _profile_aspect_facts(tools)
        requested_aspects = [
            condition.field
            for condition in state.request.conditions
            if condition.field in ASPECT_NAMES
        ]
        referenced_ids = list(state.request.referenced_business_ids)
        structured_sufficient = bool(
            requested_aspects
            and referenced_ids
            and all(
                all(
                    aspect in known_aspects.get(business_id, [])
                    for aspect in requested_aspects
                )
                for business_id in referenced_ids
            )
        )
        comparison = _latest_tool(
            tools,
            "COMPARE_BUSINESSES",
            turn_index=state.current_turn,
        )
        previous_recommended = (
            [] if not state.turns else list(state.turns[-1].recommended_business_ids)
        )
        feedback_applied = any(
            observation.turn_index == state.current_turn
            and observation.action == "apply_feedback"
            and observation.payload.get("feedback_applied") is True
            for observation in state.observations
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
            requested_aspects=requested_aspects,
            known_aspects_by_business=known_aspects,
            conflicting_aspects_by_business=conflicting_aspects,
            structured_evidence_sufficient=structured_sufficient,
            structured_evidence_conflict=any(
                aspect in conflicting_aspects.get(business_id, [])
                for business_id in referenced_ids
                for aspect in requested_aspects
            ),
            review_evidence_count=len(review_hits),
            review_evidence_business_ids=list(
                dict.fromkeys(
                    str(item["business_id"])
                    for item in review_hits
                    if isinstance(item.get("business_id"), str)
                )
            ),
            review_evidence_conflict=_review_conflict(review_hits),
            explicit_uncertainty_request=_asks_for_uncertainty(
                state.request.query_text
            ),
            feedback_applied=feedback_applied,
            reject_previous_recommendation=_rejects_previous_recommendation(
                state.request.query_text
            ),
            previous_recommended_business_ids=previous_recommended,
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
            semantic_ranks_by_business=_semantic_ranks(semantic_match),
            cross_encoder_ranks_by_business=_cross_encoder_ranks(
                cross_encoder_match
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
            semantic_match=_route_tool_fact(semantic_match),
            cross_encoder_match=_route_tool_fact(cross_encoder_match),
            business_details=_route_tool_fact(details),
            business_profiles=_route_tool_fact(profiles),
            comparison=_route_tool_fact(comparison),
            review_search=_route_tool_fact(review_search),
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
                    - state.turn_input_tokens
                    - state.turn_output_tokens,
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


def _semantic_ranks(tool: _NormalizedTool | None) -> dict[str, int]:
    rows = _tool_data(tool).get("matches")
    if not isinstance(rows, list):
        return {}
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        business_id = row.get("business_id")
        rank = row.get("semantic_rank")
        if isinstance(business_id, str) and isinstance(rank, int):
            result[business_id] = rank
    return result


def _cross_encoder_ranks(tool: _NormalizedTool | None) -> dict[str, int]:
    rows = _tool_data(tool).get("matches")
    if not isinstance(rows, list):
        return {}
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        business_id = row.get("business_id")
        rank = row.get("cross_encoder_rank")
        if isinstance(business_id, str) and isinstance(rank, int):
            result[business_id] = rank
    return result


def _review_hits(tool: _NormalizedTool | None) -> list[dict[str, object]]:
    rows = _tool_data(tool).get("hits")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _review_conflict(hits: list[dict[str, object]]) -> bool:
    by_business_aspect: dict[tuple[str, str], set[str]] = {}
    for hit in hits:
        business_id = hit.get("business_id")
        aspects = hit.get("matched_aspects")
        sentiments = hit.get("aspect_sentiments")
        if not isinstance(business_id, str) or not isinstance(aspects, list):
            continue
        values = {
            str(value)
            for value in sentiments or []
            if value in {"positive", "negative"}
        }
        for aspect in aspects:
            if isinstance(aspect, str):
                by_business_aspect.setdefault((business_id, aspect), set()).update(values)
    return any(values == {"positive", "negative"} for values in by_business_aspect.values())


def _asks_for_uncertainty(query_text: str) -> bool:
    text = query_text.casefold()
    return any(
        marker in text
        for marker in (
            "only a few",
            "very few",
            "conflicting",
            "mixed reviews",
            "can we be sure",
            "能确定吗",
            "很少评论",
            "说法不一",
            "相互矛盾",
        )
    )


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


def _profile_aspect_facts(
    tools: list[_NormalizedTool],
) -> tuple[dict[str, list[AspectName]], dict[str, list[AspectName]]]:
    known: dict[str, list[AspectName]] = {}
    conflicting: dict[str, list[AspectName]] = {}
    for _, payload in tools:
        if payload.tool_name != "GET_BUSINESS_PROFILE" or payload.status not in {
            "success",
            "partial",
        }:
            continue
        profiles = payload.data.get("profiles")
        if not isinstance(profiles, list):
            continue
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            business_id = profile.get("business_id")
            summaries = profile.get("aspect_summaries")
            if not isinstance(business_id, str) or not isinstance(summaries, dict):
                continue
            known[business_id] = []
            conflicting[business_id] = []
            for aspect in ASPECT_NAMES:
                summary = summaries.get(aspect)
                if not isinstance(summary, dict) or summary.get("status") != "known":
                    continue
                known[business_id].append(aspect)
                if summary.get("conflict") is True:
                    conflicting[business_id].append(aspect)
    return known, conflicting


def _rejects_previous_recommendation(query_text: str) -> bool:
    text = query_text.casefold()
    markers = (
        "too expensive",
        "too far",
        "too noisy",
        "another",
        "something else",
        "replace that",
        "replace it",
        "太贵",
        "太远",
        "太吵",
        "换一家",
        "换一个",
        "不要刚才",
        "重新推荐",
    )
    return any(marker in text for marker in markers)
