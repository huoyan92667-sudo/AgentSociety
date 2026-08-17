"""Visible-only adapter from Query Recommendation cases to the full Agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from yelp_agent.agent_benchmark import VisibleAgentScenario
from yelp_agent.agent_harness import AgentHarness, HarnessResult
from yelp_agent.query_recommendation_benchmark import VisibleQueryRecommendationCase

from .schema import QueryRecommendationAgentPrediction


class AgentStarter(Protocol):
    def start(self, scenario: VisibleAgentScenario) -> HarnessResult: ...


ProgressCallback = Callable[[int, int, VisibleQueryRecommendationCase], None]


class QueryRecommendationAgentRunner:
    """Execute visible Query cases through exactly the same production Harness."""

    def __init__(self, harness: AgentHarness | AgentStarter) -> None:
        self._harness = harness

    def run(
        self,
        cases: Sequence[VisibleQueryRecommendationCase],
        *,
        progress: ProgressCallback | None = None,
    ) -> tuple[QueryRecommendationAgentPrediction, ...]:
        ordered = tuple(sorted(cases, key=lambda item: item.case_id))
        if not ordered:
            raise ValueError("Query Recommendation Agent run requires visible cases")
        if len({item.case_id for item in ordered}) != len(ordered):
            raise ValueError("visible Query cases must be unique")
        predictions = []
        for index, case in enumerate(ordered, start=1):
            result = self._harness.start(as_visible_agent_scenario(case))
            predictions.append(prediction_from_harness_result(case, result))
            if progress is not None:
                progress(index, len(ordered), case)
        return tuple(predictions)


def as_visible_agent_scenario(
    case: VisibleQueryRecommendationCase,
) -> VisibleAgentScenario:
    """Translate fields one-for-one; hidden truth is impossible to pass here."""

    return VisibleAgentScenario(
        scenario_id=case.case_id,
        split=case.split,
        language=case.language,
        user_id=case.user_id,
        session_id=case.session_id,
        cutoff_time=case.cutoff_time,
        query_text=case.query_text,
        user_latitude=case.user_latitude,
        user_longitude=case.user_longitude,
        referenced_business_ids=[],
        generator_kind=case.generator_kind,
        generator_model=case.generator_model,
        generator_prompt_sha256=case.generator_prompt_sha256,
    )


def prediction_from_harness_result(
    case: VisibleQueryRecommendationCase,
    result: HarnessResult,
) -> QueryRecommendationAgentPrediction:
    session = result.session
    turns = session.turns
    if not turns:
        raise ValueError("Agent Harness returned no durable turn trace")
    terminal = turns[-1]
    retrieval_ranking = _latest_retrieval_ranking(session.observations)
    query_aware_result = _latest_query_aware_result(session.observations)
    action_count = sum(len(turn.actions) for turn in turns)
    invalid_action_count = sum(
        action.status == "rejected"
        for turn in turns
        for action in turn.actions
    )
    tool_call_count = sum(len(turn.tool_calls) for turn in turns)
    fallback = session.status == "fallback"
    return QueryRecommendationAgentPrediction(
        case_id=case.case_id,
        split=case.split,
        agent_version=session.agent_version,
        status=session.status,
        response_kind=terminal.response_kind,
        request=session.request,
        retrieval_ranking=retrieval_ranking,
        final_ranking=terminal.candidate_ranking,
        displayed_business_ids=terminal.recommended_business_ids,
        evidence_cards=[],
        query_aware_result=query_aware_result,
        turns=turns,
        fallback=fallback,
        fallback_reason=session.fallback_reason,
        failure_codes=[
            f"ACTION_{action.status.upper()}:{action.action}"
            for turn in turns
            for action in turn.actions
            if action.status in {"failed", "rejected"}
        ],
        action_count=action_count,
        invalid_action_count=invalid_action_count,
        tool_call_count=tool_call_count,
        latency_ms=session.elapsed_ms,
        input_tokens=session.input_tokens if session.token_usage_observed else None,
        output_tokens=(
            session.output_tokens if session.token_usage_observed else None
        ),
        cost_usd=session.cost_usd if session.cost_usd > 0 else None,
    )


def _latest_retrieval_ranking(observations: Sequence[object]) -> list[str]:
    for observation in reversed(observations):
        payload = getattr(observation, "payload", None)
        if not isinstance(payload, dict) or payload.get("tool_name") not in {
            "EXPAND_CANDIDATES",
            "GET_QUERY_AWARE_RANKING",
        }:
            continue
        data = payload.get("data")
        values = data.get("candidate_business_ids") if isinstance(data, dict) else None
        if isinstance(values, list):
            return list(dict.fromkeys(value for value in values if isinstance(value, str)))[:500]
    return []


def _latest_query_aware_result(
    observations: Sequence[object],
) -> dict[str, object] | None:
    for observation in reversed(observations):
        payload = getattr(observation, "payload", None)
        if not isinstance(payload, dict):
            continue
        if payload.get("tool_name") != "GET_QUERY_AWARE_RANKING":
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        return dict(data)
    return None
