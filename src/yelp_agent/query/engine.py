"""Agent-ready static recommendation interface for one current request."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from yelp_agent.models import StrictModel
from yelp_agent.query.parser import RecommendationRequestParser
from yelp_agent.query.ranking import (
    QueryAwareCandidate,
    QueryAwareRankingResult,
    QueryAwareStaticRanker,
)
from yelp_agent.query.schema import QueryParseInput, RecommendationRequest


class QueryAwareRecommendation(StrictModel):
    """Three directly comparable outputs over the exact same candidates."""

    request: RecommendationRequest
    hybrid_v2_ranking: list[str]
    query_only: QueryAwareRankingResult
    hybrid_query: QueryAwareRankingResult


@dataclass(frozen=True, slots=True)
class QueryAwareRecommender:
    """Hide parsing, policy enforcement and both static ranking modes."""

    parser: RecommendationRequestParser
    ranker: QueryAwareStaticRanker

    def recommend(
        self,
        value: QueryParseInput,
        candidates: Sequence[QueryAwareCandidate],
    ) -> QueryAwareRecommendation:
        request = self.parser.parse(value)
        baseline = [
            candidate.business_id
            for candidate in sorted(
                candidates,
                key=lambda candidate: (
                    candidate.hybrid_rank,
                    candidate.business_id,
                ),
            )
        ]
        return QueryAwareRecommendation(
            request=request,
            hybrid_v2_ranking=baseline,
            query_only=self.ranker.rank(request, candidates, mode="query_only"),
            hybrid_query=self.ranker.rank(request, candidates, mode="hybrid_query"),
        )
