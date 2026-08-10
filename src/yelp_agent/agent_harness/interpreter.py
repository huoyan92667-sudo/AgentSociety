"""Adapters from visible user turns to the existing Step 18/19 contracts."""

from __future__ import annotations

from yelp_agent.decision_readiness import DecisionReadinessAnalyzer
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
        if previous_session is not None:
            previous_location = previous_session.request.location_center
            references = list(
                dict.fromkeys(
                    previous_session.request.referenced_business_ids
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
        readiness = self._analyzer.analyze(request, ranking_source="none")
        return TurnInterpretation(request=request, readiness=readiness)
