from __future__ import annotations

from datetime import datetime

from yelp_agent.agent_tools.adapters.query_aware_ranking import (
    GetQueryAwareRankingTool,
)
from yelp_agent.agent_tools.schema import ToolExecutionContext
from yelp_agent.agent_tools.tool_schemas import EmptyToolInput
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query_aware_ranking import (
    CoarseCandidateScore,
    QueryAwareRankingResult,
)


class FixedQueryAwareRankingService:
    def __init__(self, result: QueryAwareRankingResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def rank(self, **kwargs: object) -> QueryAwareRankingResult:
        self.calls.append(kwargs)
        return self.result


def test_query_aware_tool_exposes_final_ranking_and_session_exclusions() -> None:
    cutoff = datetime(2022, 1, 1)
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=cutoff,
            query_text="Find a steakhouse in Philadelphia",
        )
    )
    result = QueryAwareRankingResult(
        case_id="a" * 64,
        request_id=request.request_id,
        split="development",
        ranking=["b1"],
        top_10=["b1"],
        displayed_top_5=["b1"],
        coarse_scores=[
            CoarseCandidateScore(
                business_id="b1",
                source_channels=["query"],
                query_rank=1,
                lightgbm_rank=1,
                embedding_rank=1,
                query_signal=1.0,
                coarse_score=1.0,
                coarse_rank=1,
                final_rank=1,
            )
        ],
        semantic_scores=[],
        hard_exclusions=[],
        query_weight=0.5,
        semantic_candidate_limit=30,
        latency_ms=2.0,
        embedding_input_tokens=10,
        cross_encoder_input_tokens=5,
        logical_input_tokens=20,
        cache_hits=2,
        cache_misses=0,
    )
    service = FixedQueryAwareRankingService(result)
    context = ToolExecutionContext(
        request_id=request.request_id,
        user_id=request.user_id,
        cutoff_time=cutoff,
        action="retrieve_candidates",
        state_snapshot={
            "scenario_id": "a" * 64,
            "split": "development",
            "request": request.model_dump(mode="json"),
            "memory_context": {"rejected_business_ids": ["b2"]},
        },
    )

    observation = GetQueryAwareRankingTool(service).run(
        EmptyToolInput(),
        context,
    )

    assert observation.status == "success"
    assert observation.data["candidate_business_ids"] == ["b1"]
    assert observation.data["ranking"] == ["b1"]
    assert observation.input_tokens == 15
    assert observation.cache_hit is True
    assert service.calls[0]["case_id"] == "a" * 64
    assert service.calls[0]["rejected_business_ids"] == {"b2"}
