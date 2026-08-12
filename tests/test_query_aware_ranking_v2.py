from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import pytest

from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query_aware_ranking import (
    CandidateEvidence,
    ExternalModelCallError,
    ProtectedCandidatePool,
    QueryAwareRankingPolicy,
    QueryAwareRecommendationEngine,
    metrics_for_rankings,
    score_coarse_candidates,
)
from yelp_agent.query_retrieval import (
    ExcludedQueryBusiness,
    QueryRetrievalCandidate,
    QueryRetrievalResult,
    QueryRetrievalUsage,
)
from yelp_agent.retrieval import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalTaskContext,
)
from yelp_agent.semantic_ranking import SemanticRankingPolicy


CUTOFF = datetime(2024, 1, 1)


def _history_candidate(business_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        business_id=business_id,
        rank=rank,
        fusion_score=1.0 / (60 + rank),
        route_count=1,
        quality_rank=rank,
        quality_score=0.8,
        category_rank=None,
        category_score=0.0,
        text_rank=None,
        text_score=0.0,
        location_rank=None,
        location_score=0.5,
        distance_km=None,
        item_knn_rank=None,
        item_knn_positive_score=0.0,
        item_knn_negative_evidence=0.0,
        item_knn_positive_support_count=0,
        item_knn_negative_support_count=0,
        item_knn_positive_neighbor_count=0,
        item_knn_negative_neighbor_count=0,
        item_knn_missing=True,
    )


def _history(*ids: str) -> RetrievalResult:
    return RetrievalResult(
        task=RetrievalTaskContext(
            task_id="case",
            split="development",
            user_id="user",
            cutoff_time=CUTOFF,
            history_count=3,
        ),
        catalog_size=10,
        eligible_candidate_count=7,
        excluded_history_businesses=3,
        candidates=tuple(
            _history_candidate(business_id, rank)
            for rank, business_id in enumerate(ids, start=1)
        ),
        route_candidates=(),
        route_result_counts={
            "quality": len(ids),
            "category": 0,
            "text": 0,
            "location": 0,
            "item_knn": 0,
        },
        latency_ms=1.0,
    )


def _query(*ids: str, excluded: tuple[str, ...] = ()) -> QueryRetrievalResult:
    return QueryRetrievalResult(
        request_id="a" * 64,
        cutoff_time=CUTOFF,
        catalog_size=10,
        pre_cutoff_business_count=10,
        eligible_business_count=10 - len(excluded),
        candidates=[
            QueryRetrievalCandidate(
                business_id=business_id,
                rank=rank,
                fusion_score=1.0 / (60 + rank),
                route_count=1,
                embedding_rank=rank,
                embedding_score=1.0 - 0.1 * rank,
                source_scope="selected_user_interactions",
            )
            for rank, business_id in enumerate(ids, start=1)
        ],
        excluded=[
            ExcludedQueryBusiness(
                business_id=business_id,
                reason_codes=["HARD_CATEGORY_MISSING:Steakhouses"],
            )
            for business_id in excluded
        ],
        route_result_counts={
            "query_category": 0,
            "query_embedding": len(ids),
            "query_aspect": 0,
            "query_location": 0,
        },
        warnings=[],
        usage=QueryRetrievalUsage(),
        latency_ms=2.0,
    )


def _policy() -> QueryAwareRankingPolicy:
    semantic = SemanticRankingPolicy(
        candidate_limit=5,
        embedding_weight=0.4,
        cross_encoder_weight=0.5,
        structured_weight=0.1,
        minimum_intent_confidence=0.0,
        maximum_fusion_alpha=1.0,
        evidence_floor=0.5,
        maximum_upward_move=5,
        maximum_downward_move=3,
        protected_top_k=1,
        displacement_margin=0.2,
        development_scenario_count=1,
        development_metrics={},
    )
    return QueryAwareRankingPolicy(
        selected_query_weight=0.75,
        query_retrieval_signal_weight=0.8,
        embedding_signal_weight=0.2,
        union_candidate_limit=500,
        coarse_diagnostic_limit=100,
        semantic_candidate_limit=5,
        internal_result_limit=10,
        display_limit=5,
        tie_breakers=["MRR"],
        development_case_count=1,
        development_metrics={},
        semantic_policy=semantic,
    )


def test_protected_pool_keeps_both_channels_until_hard_filter() -> None:
    pool = ProtectedCandidatePool(candidate_limit=500).build(
        _history("history-only", "both"),
        _query("both", "query-only", excluded=("history-only",)),
    )

    assert pool.union_candidate_count == 3
    assert pool.overlap_count == 1
    assert [item.business_id for item in pool.items] == ["both", "query-only"]
    assert pool.items[0].source_channels == ["history", "query"]
    assert pool.items[1].source_channels == ["query"]
    assert [item.business_id for item in pool.exclusions] == ["history-only"]


def test_coarse_fusion_lets_current_query_rescue_a_history_miss() -> None:
    candidates = [
        CandidateEvidence(
            business_id="history-favorite",
            source_channels=["history"],
            history_rank=1,
            lightgbm_rank=1,
            lightgbm_model_score=0.9,
            lightgbm_percentile=1.0,
            embedding_rank=2,
            embedding_score=0.6,
            embedding_percentile=0.0,
            query_retrieval_percentile=0.0,
        ),
        CandidateEvidence(
            business_id="query-favorite",
            source_channels=["query"],
            query_rank=1,
            lightgbm_rank=2,
            lightgbm_model_score=0.1,
            lightgbm_percentile=0.0,
            embedding_rank=1,
            embedding_score=0.9,
            embedding_percentile=1.0,
            query_retrieval_percentile=1.0,
        ),
    ]

    ranking = score_coarse_candidates(
        candidates,
        query_weight=0.75,
        query_retrieval_signal_weight=0.8,
        embedding_signal_weight=0.2,
    )

    assert ranking[0].business_id == "query-favorite"
    assert ranking[0].query_signal == pytest.approx(1.0)
    assert ranking[0].coarse_score > ranking[1].coarse_score


def test_metrics_include_hr_at_10_and_ndcg_at_10() -> None:
    rankings = [
        [f"b{index}" for index in range(1, 12)],
        ["target", "other"],
    ]
    metrics = metrics_for_rankings(
        rankings,
        ["b8", "target"],
        latencies=[10.0, 30.0],
    )

    assert metrics.hr_at_5 == pytest.approx(0.5)
    assert metrics.hr_at_10 == pytest.approx(1.0)
    assert metrics.avg_hr_1_3_5_10 == pytest.approx(0.625)
    assert metrics.ndcg_at_10 > metrics.ndcg_at_5
    assert metrics.mean_latency_ms == pytest.approx(20.0)
    assert metrics.p95_latency_ms == pytest.approx(29.0)


@dataclass
class _FakeRankingService:
    def rank(self, *, candidates, **kwargs):
        del kwargs
        return [
            {
                "business_id": row["business_id"],
                "rank": rank,
                "model_rank": rank,
                "hybrid_v1_rank": rank,
                "model_score": float(len(candidates) - rank),
                "hybrid_v1_score": 0.5,
                "blend_score": 1.0 if len(candidates) == 1 else (len(candidates) - rank) / (len(candidates) - 1),
            }
            for rank, row in enumerate(candidates, start=1)
        ]


class _FakeEmbedding:
    def match(self, *, business_ids, **kwargs):
        del kwargs
        size = len(business_ids)
        return SimpleNamespace(
            matches=[
                SimpleNamespace(
                    business_id=business_id,
                    semantic_rank=rank,
                    normalized_score=1.0 - (rank - 1) / max(1, size),
                )
                for rank, business_id in enumerate(business_ids, start=1)
            ],
            usage=SimpleNamespace(
                input_tokens=0,
                logical_input_tokens=10,
                cache_hits=size,
                cache_misses=0,
                api_calls=0,
            ),
        )


class _FailingCrossEncoder:
    def rerank(self, **kwargs):
        del kwargs
        raise TimeoutError("synthetic timeout")


class _ExternalEmbedding(_FakeEmbedding):
    def match(self, **kwargs):
        result = super().match(**kwargs)
        result.usage.api_calls = 1
        return result


class _ExternalCrossEncoder:
    def rerank(self, *, business_ids, **kwargs):
        del kwargs
        return SimpleNamespace(
            matches=[
                SimpleNamespace(
                    business_id=business_id,
                    relevance_score=1.0 / rank,
                )
                for rank, business_id in enumerate(business_ids, start=1)
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                logical_input_tokens=10,
                cache_hits=0,
                cache_misses=len(business_ids),
                api_calls=1,
            ),
        )


class _Profiles:
    def get(self, business_ids, cutoff_time):
        del business_ids, cutoff_time
        return {}


def test_engine_preserves_complete_union_when_semantic_stage_falls_back() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user",
            session_id="session",
            cutoff_time=CUTOFF,
            query_text="Find a steakhouse for dinner.",
        )
    )
    engine = QueryAwareRecommendationEngine(
        ranking_service=_FakeRankingService(),
        embedding_matcher=_FakeEmbedding(),
        cross_encoder=_FailingCrossEncoder(),
        business_profiles=_Profiles(),
        policy=_policy(),
    )
    history = _history("h1", "both")
    query = _query("q1", "both")

    prepared = engine.prepare(
        case_id="f" * 64,
        split="development",
        request=request,
        history=history,
        query=query,
        rrf_fusion_ranking=["both", "h1", "q1"],
        usage_scope="test",
    )
    result = engine.finalize(prepared, usage_scope="test")

    assert prepared.union_candidate_count == 3
    assert set(result.ranking) == {"h1", "both", "q1"}
    assert result.fallback is True
    assert result.fallback_reason == "SEMANTIC_RANKING_FALLBACK:TimeoutError"
    assert result.displayed_top_5 == result.ranking
    assert sorted(item.final_rank for item in result.coarse_scores) == [1, 2, 3]


def test_preparation_rejects_external_embedding_calls() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user",
            session_id="session",
            cutoff_time=CUTOFF,
            query_text="Find a steakhouse for dinner.",
        )
    )
    engine = QueryAwareRecommendationEngine(
        ranking_service=_FakeRankingService(),
        embedding_matcher=_ExternalEmbedding(),
        cross_encoder=_FailingCrossEncoder(),
        business_profiles=_Profiles(),
        policy=_policy(),
    )

    with pytest.raises(ExternalModelCallError, match="must not call an external model"):
        engine.prepare(
            case_id="e" * 64,
            split="development",
            request=request,
            history=_history("h1"),
            query=_query("q1"),
            rrf_fusion_ranking=["h1", "q1"],
            usage_scope="test",
        )


def test_finalization_rejects_external_cross_encoder_calls() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user",
            session_id="session",
            cutoff_time=CUTOFF,
            query_text="Find a steakhouse for dinner.",
        )
    )
    engine = QueryAwareRecommendationEngine(
        ranking_service=_FakeRankingService(),
        embedding_matcher=_FakeEmbedding(),
        cross_encoder=_ExternalCrossEncoder(),
        business_profiles=_Profiles(),
        policy=_policy(),
    )
    prepared = engine.prepare(
        case_id="d" * 64,
        split="development",
        request=request,
        history=_history("h1"),
        query=_query("q1"),
        rrf_fusion_ranking=["h1", "q1"],
        usage_scope="test",
    )

    with pytest.raises(ExternalModelCallError, match="must not call an external model"):
        engine.finalize(prepared, usage_scope="test")
