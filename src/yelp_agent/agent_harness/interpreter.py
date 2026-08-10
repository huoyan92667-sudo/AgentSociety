"""Adapters from visible user turns to the existing Step 18/19 contracts."""

from __future__ import annotations

import re

from yelp_agent.decision_readiness import (
    DecisionReadinessAnalyzer,
    identify_information_gaps,
)
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query.parser import RecommendationRequestParser

from .schema import AgentSession, TurnInterpretation


class RuleBasedRequestInterpreter:
    """Use the deterministic request parser and readiness analyzer."""

    def __init__(
        self,
        parser: RecommendationRequestParser | None = None,
        analyzer: DecisionReadinessAnalyzer | None = None,
    ) -> None:
        self._parser = parser or build_rule_based_request_parser()
        self._analyzer = analyzer or DecisionReadinessAnalyzer()

    def interpret(
        self,
        value: QueryParseInput,
        *,
        previous_session: AgentSession | None = None,
    ) -> TurnInterpretation:
        current_value = value
        landmark = _known_landmark_coordinates(value.query_text)
        if (
            landmark is not None
            and value.user_latitude is None
            and value.user_longitude is None
        ):
            value = value.model_copy(
                update={
                    "user_latitude": landmark[0],
                    "user_longitude": landmark[1],
                }
            )
        if previous_session is not None:
            previous_location = previous_session.request.location_center
            previous_recommendations = (
                []
                if not previous_session.turns
                else previous_session.turns[-1].recommended_business_ids
            )
            references = list(
                dict.fromkeys(
                    previous_session.request.referenced_business_ids
                    + previous_recommendations
                    + value.referenced_business_ids
                )
            )
            value = value.model_copy(
                update={
                    "query_text": (
                        f"{previous_session.request.query_text}\n{value.query_text}"
                    ),
                    "user_latitude": (
                        value.user_latitude
                        if value.user_latitude is not None
                        else (
                            None
                            if previous_location is None
                            else previous_location.latitude
                        )
                    ),
                    "user_longitude": (
                        value.user_longitude
                        if value.user_longitude is not None
                        else (
                            None
                            if previous_location is None
                            else previous_location.longitude
                        )
                    ),
                    "referenced_business_ids": references,
                }
            )
        request = self._parser.parse(value)
        if previous_session is not None and previous_session.status == "awaiting_user":
            missing = list(request.missing_fields)
            previous_gaps = set(previous_session.readiness.information_gaps)
            if (
                "missing_budget" in previous_gaps
                and _contains_explicit_number(current_value.query_text)
                and "budget_precision" in missing
            ):
                missing.remove("budget_precision")
            request = request.model_copy(update={"missing_fields": missing})
        readiness = self._analyzer.analyze(request, ranking_source="none")
        task_override = _step24_task_override(
            request.query_text,
            referenced_business_ids=request.referenced_business_ids,
            current_task_type=readiness.task_type,
        )
        if task_override is not None:
            gaps, conflicts = identify_information_gaps(request, task_override)
            readiness = readiness.model_copy(
                update={
                    "task_type": task_override,
                    "task_type_reason_code": "step24_extended_task_pattern",
                    "information_gaps": list(gaps),
                    "conflict_fields": list(conflicts),
                }
            )
        if (
            readiness.task_type
            not in {"recommendation_request", "feedback_refinement"}
            and "missing_party_size" in readiness.information_gaps
        ):
            readiness = readiness.model_copy(
                update={
                    "information_gaps": [
                        gap
                        for gap in readiness.information_gaps
                        if gap != "missing_party_size"
                    ]
                }
            )
        return TurnInterpretation(request=request, readiness=readiness)


def _contains_explicit_number(text: str) -> bool:
    return (
        re.search(r"(?:\$|usd|dollars?|元|块)?\s*\d+(?:\.\d+)?", text, re.I)
        is not None
    )


def _known_landmark_coordinates(text: str) -> tuple[float, float] | None:
    lowered = text.casefold()
    if "philadelphia city hall" in lowered or "费城市政厅" in text:
        return (39.9526, -75.1652)
    return None


def _step24_task_override(
    text: str,
    *,
    referenced_business_ids: list[str],
    current_task_type: str,
) -> str | None:
    lowered = text.casefold()
    comparison = (
        len(referenced_business_ids) >= 2
        and (
            re.search(r"which\s+is.{0,40}\bbetter", lowered) is not None
            or re.search(r"哪(?:个|家).{0,40}更", text) is not None
        )
    )
    if comparison and current_task_type in {
        "unknown",
        "recommendation_request",
        "review_experience_question",
    }:
        return "candidate_comparison"
    if current_task_type != "unknown":
        return None
    recommendation_markers = (
        "find a ",
        "find an ",
        "only want",
        "it must be",
        "follow today's request",
        "follow today’s request",
        "只想找",
        "必须是",
        "请按这次要求",
    )
    if any(marker in lowered for marker in recommendation_markers):
        return "recommendation_request"
    return None
