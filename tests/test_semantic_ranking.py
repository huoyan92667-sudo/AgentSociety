"""Step 30 intent compilation, scoring, ablation, and safety tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from yelp_agent.business_profiles.schema import BusinessAspectEvent
from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config
from yelp_agent.cross_encoder import (
    CrossEncoderBusinessMatch,
    CrossEncoderMatchResult,
    CrossEncoderUsage,
)
from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.query.schema import RecommendationRequest, RequestCondition
from yelp_agent.semantic_embedding import (
    EmbeddingUsage,
    SemanticBusinessMatch,
    SemanticMatchResult,
)
from yelp_agent.semantic_ranking import (
    RankingIntentCompiler,
    SemanticRankingEngine,
    SemanticRankingPolicy,
    apply_ranking_policy,
    score_candidates,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CUTOFF = datetime(2022, 1, 1)


def _policy(**updates: object) -> SemanticRankingPolicy:
    payload: dict[str, object] = {
        "candidate_limit": 3,
        "embedding_weight": 0.2,
        "cross_encoder_weight": 0.5,
        "structured_weight": 0.3,
        "minimum_intent_confidence": 0.5,
        "maximum_fusion_alpha": 0.6,
        "evidence_floor": 0.5,
        "maximum_upward_move": 1,
        "maximum_downward_move": 1,
        "protected_top_k": 1,
        "displacement_margin": 0.2,
        "development_scenario_count": 0,
        "development_metrics": {},
    }
    payload.update(updates)
    return SemanticRankingPolicy.model_validate(payload)


def _condition(
    field: str,
    span: str,
    *,
    source: str = "semantic_model",
) -> RequestCondition:
    return RequestCondition.model_validate(
        {
            "field": field,
            "operator": "prefer",
            "value": True,
            "importance": "strong",
            "enforcement": "rank",
            "explicit": True,
            "confidence": 0.9,
            "evidence_span": span,
            "evidence_start": 0,
            "evidence_end": len(span),
            "source": source,
            "unknown_policy": "allow_with_warning",
        }
    )


def _request(*conditions: RequestCondition) -> RecommendationRequest:
    return RecommendationRequest(
        request_id="a" * 64,
        user_id="user-1",
        session_id="session-1",
        cutoff_time=CUTOFF,
        query_text="quiet place for a date",
        intent="recommendation_request",
        conditions=list(conditions),
        parser_version="test",
    )


def _business(business_id: str) -> BusinessRecord:
    return BusinessRecord(
        business_id=business_id,
        name=f"Business {business_id}",
        address="1 Test Street",
        city="Philadelphia",
        state="PA",
        postal_code="19103",
        latitude=39.95,
        longitude=-75.16,
        categories=("Restaurants", "Steakhouses"),
        attributes_json='{"RestaurantsPriceRange2":"3"}',
    )


def _store() -> BusinessKnowledgeStore:
    events = []
    for business_id, sentiment in (("a", "negative"), ("b", "positive")):
        for index in range(3):
            events.append(
                BusinessAspectEvent(
                    review_id=f"{business_id}-{index}",
                    business_id=business_id,
                    user_id=f"u-{index}",
                    review_time=datetime(2020, 1, index + 1),
                    aspect="quiet_environment",
                    sentiment=sentiment,  # type: ignore[arg-type]
                    confidence=0.95,
                    source_text_sha256="b" * 64,
                    extractor_version="test",
                )
            )
    return BusinessKnowledgeStore.from_records(
        businesses=(_business("a"), _business("b"), _business("c")),
        rating_events=(),
        aspect_events=events,
        config=load_business_profile_config(CONFIG_DIR),
    )


def test_intent_compiler_preserves_source_confidence_and_llm_signal_count() -> None:
    intent = RankingIntentCompiler().compile(
        _request(
            _condition("quiet_environment", "quiet"),
            _condition("date_suitable", "date", source="rule"),
        )
    )

    assert intent.rankable_condition_count == 2
    assert intent.semantic_model_condition_count == 1
    assert intent.mean_confidence == 0.9
    assert "quiet environment" in intent.document
    assert "source=semantic_model" in intent.document


def test_aggressive_uses_semantic_order_while_protected_bounds_movement() -> None:
    request = _request(_condition("quiet_environment", "quiet"))
    intent = RankingIntentCompiler().compile(request)
    profiles = _store().get(["a", "b", "c"], CUTOFF)
    rows = score_candidates(
        request=request,
        base_ranking=["a", "c", "b"],
        profiles=profiles,
        embedding_scores={"a": 0.1, "b": 0.9, "c": 0.5},
        cross_encoder_scores={"a": 0.1, "b": 0.9, "c": 0.5},
        policy=_policy(),
    )

    aggressive, aggressive_rows, alpha, no_op = apply_ranking_policy(
        base_ranking=["a", "c", "b"],
        rows=rows,
        intent=intent,
        mode="aggressive",
        policy=_policy(),
    )
    protected, protected_rows, protected_alpha, _ = apply_ranking_policy(
        base_ranking=["a", "c", "b"],
        rows=rows,
        intent=intent,
        mode="protected",
        policy=_policy(
            displacement_margin=1.0,
            maximum_upward_move=2,
            maximum_downward_move=2,
            maximum_fusion_alpha=1.0,
            evidence_floor=1.0,
        ),
    )

    assert aggressive == ["b", "c", "a"]
    assert alpha == 1.0
    assert no_op is None
    assert max(abs(item.rank_movement) for item in aggressive_rows) == 2
    assert protected[0] == "a"
    assert protected_alpha == 0.9
    assert max(abs(item.rank_movement) for item in protected_rows) <= 2
    assert any(item.protection_reason_codes for item in protected_rows)


def test_no_rankable_condition_is_byte_stable_no_op() -> None:
    request = _request()
    intent = RankingIntentCompiler().compile(request)
    profiles = _store().get(["a", "b", "c"], CUTOFF)
    rows = score_candidates(
        request=request,
        base_ranking=["a", "b", "c"],
        profiles=profiles,
        embedding_scores={"a": 0.1, "b": 0.9, "c": 0.5},
        cross_encoder_scores={"a": 0.1, "b": 0.9, "c": 0.5},
        policy=_policy(),
    )

    ranking, _, alpha, reason = apply_ranking_policy(
        base_ranking=["a", "b", "c"],
        rows=rows,
        intent=intent,
        mode="aggressive",
        policy=_policy(),
    )

    assert ranking == ["a", "b", "c"]
    assert alpha == 0
    assert reason == "NO_RANKABLE_SEMANTIC_CONDITIONS"


class _Embedding:
    def match(self, *, query_text: str, business_ids: list[str], **_: object) -> SemanticMatchResult:
        assert "Accepted structured requirements" in query_text
        return SemanticMatchResult(
            query_sha256="c" * 64,
            provider="local",
            model="fake-embedding",
            dimension=3,
            matches=[
                SemanticBusinessMatch(
                    business_id=business_id,
                    cutoff_time=CUTOFF,
                    document_sha256="d" * 64,
                    cosine_similarity=0.8 - index * 0.2,
                    normalized_score=0.9 - index * 0.1,
                    semantic_rank=index + 1,
                )
                for index, business_id in enumerate(business_ids)
            ],
            usage=EmbeddingUsage(
                encoder_calls=1,
                api_calls=0,
                input_tokens=10,
                logical_input_tokens=20,
                cache_saved_tokens=10,
                truncated_text_count=0,
                cache_hits=3,
                cache_misses=1,
                estimated_cost_cny=0,
                provider_latency_ms=1,
            ),
        )


class _Cross:
    def rerank(self, *, query_text: str, business_ids: list[str], **_: object) -> CrossEncoderMatchResult:
        assert "Accepted structured requirements" in query_text
        return CrossEncoderMatchResult(
            query_sha256="e" * 64,
            model="fake-cross",
            matches=[
                CrossEncoderBusinessMatch(
                    business_id=business_id,
                    cutoff_time=CUTOFF,
                    document_sha256="f" * 64,
                    relevance_score=0.9 - index * 0.1,
                    cross_encoder_rank=index + 1,
                )
                for index, business_id in enumerate(business_ids)
            ],
            usage=CrossEncoderUsage(
                input_tokens=30,
                logical_input_tokens=40,
                cache_hits=2,
                cache_misses=1,
            ),
        )


def test_engine_accounts_local_tokens_and_preserves_complete_scope() -> None:
    engine = SemanticRankingEngine(
        profiles=_store(),
        embedding_matcher=_Embedding(),  # type: ignore[arg-type]
        cross_encoder_reranker=_Cross(),  # type: ignore[arg-type]
        policy=_policy(),
        mode="aggressive",
        candidate_limit=3,
    )

    result = engine.rank(
        request=_request(_condition("quiet_environment", "quiet")),
        base_ranking=["a", "b", "c"],
        cutoff_time=CUTOFF,
    )

    assert set(result.ranking) == {"a", "b", "c"}
    assert result.usage.actual_input_tokens == 40
    assert result.usage.logical_input_tokens == 60
    assert result.usage.provider_calls == 0
    assert result.fallback is False


class _FailingEmbedding:
    def match(self, **_: object) -> SemanticMatchResult:
        raise TimeoutError("synthetic timeout")


def test_engine_failure_preserves_step26_ranking() -> None:
    engine = SemanticRankingEngine(
        profiles=_store(),
        embedding_matcher=_FailingEmbedding(),  # type: ignore[arg-type]
        cross_encoder_reranker=_Cross(),  # type: ignore[arg-type]
        policy=_policy(),
        mode="protected",
        candidate_limit=3,
    )

    result = engine.rank(
        request=_request(_condition("quiet_environment", "quiet")),
        base_ranking=["a", "b", "c"],
        cutoff_time=CUTOFF,
    )

    assert result.ranking == ["a", "b", "c"]
    assert result.fallback is True
    assert result.fallback_reason == "TimeoutError"
