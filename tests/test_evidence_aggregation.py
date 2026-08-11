from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from yelp_agent.evidence_aggregation import (
    EvidenceAggregationPolicy,
    EvidenceAggregationRequest,
    EvidenceAggregator,
    load_evidence_aggregation_config,
    load_evidence_aggregation_policy,
)
from yelp_agent.agent_tools import PermanentToolError, ToolExecutionContext
from yelp_agent.agent_tools.adapters import AggregateReviewEvidenceTool
from yelp_agent.agent_tools.tool_schemas import AggregateReviewEvidenceInput
from yelp_agent.review_rag import (
    ReviewEvidenceHit,
    ReviewSearchResult,
    SegmentAspectEvidence,
)
from yelp_agent.semantic_embedding import EmbeddingUsage


_CUTOFF = datetime(2022, 1, 1)
_PROJECT_ROOT = Path(__file__).parents[1]


def _usage() -> EmbeddingUsage:
    return EmbeddingUsage(
        encoder_calls=0,
        api_calls=0,
        input_tokens=0,
        logical_input_tokens=0,
        cache_saved_tokens=0,
        truncated_text_count=0,
        cache_hits=0,
        cache_misses=0,
        estimated_cost_cny=0.0,
        provider_latency_ms=0.0,
    )


def _hit(
    *,
    rank: int,
    review_id: str,
    user_id: str,
    sentiment: str,
    days_old: int = 30,
    span: str = "The dining room was quiet on a weekday.",
) -> ReviewEvidenceHit:
    evidence = SegmentAspectEvidence(
        aspect="quiet_environment",
        sentiment=sentiment,  # type: ignore[arg-type]
        confidence=0.95,
        evidence_span=span,
    )
    return ReviewEvidenceHit(
        rank=rank,
        segment_id=f"{rank:064x}",
        review_id=review_id,
        business_id="business-1",
        user_id=user_id,
        review_time=_CUTOFF - timedelta(days=days_old),
        stars=4,
        useful=2,
        text=span,
        text_sha256=f"{rank + 10:064x}",
        matched_aspects=["quiet_environment"],
        aspect_sentiments=[sentiment],  # type: ignore[list-item]
        aspect_evidence=[evidence],
        aspect_rank=rank,
        bm25_rank=rank,
        embedding_rank=rank,
        rrf_score=0.1,
        relevance_score=0.9,
    )


def _request(
    *hits: ReviewEvidenceHit,
    explicit_uncertainty: bool = False,
    task_type: str = "review_experience_question",
    polarity: str = "positive",
) -> EvidenceAggregationRequest:
    result = ReviewSearchResult(
        request_sha256="1" * 64,
        business_ids=["business-1"],
        cutoff_time=_CUTOFF,
        hits=list(hits),
        route_result_counts={"aspect": len(hits), "bm25": len(hits), "embedding": 0},
        eligible_segment_count=len(hits),
        embedding_usage=_usage(),
    )
    return EvidenceAggregationRequest(
        query_text="Is this business quiet enough for a conversation?",
        task_type=task_type,  # type: ignore[arg-type]
        requested_aspects=["quiet_environment"],
        desired_polarity_by_aspect={
            "quiet_environment": polarity,  # type: ignore[dict-item]
        },
        explicit_uncertainty_request=explicit_uncertainty,
        search_result=result,
    )


def _aggregator() -> EvidenceAggregator:
    return EvidenceAggregator(
        EvidenceAggregationPolicy(policy_version="test-policy")
    )


def test_consistent_multi_user_evidence_becomes_grounded() -> None:
    assessment = _aggregator().aggregate(
        _request(
            _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive"),
            _hit(rank=2, review_id="r2", user_id="u2", sentiment="positive"),
            _hit(rank=3, review_id="r3", user_id="u3", sentiment="positive"),
        )
    )
    aspect = assessment.businesses[0].aspects[0]
    assert aspect.consensus == "supports"
    assert aspect.confidence_level == "high"
    assert aspect.response_mode == "grounded"
    assert aspect.unique_user_count == 3
    assert aspect.citation_review_ids == ["r1", "r2", "r3"]
    assert aspect.condition_groups[0].condition_tag == "weekday"


def test_two_sided_directional_evidence_reports_conflict() -> None:
    assessment = _aggregator().aggregate(
        _request(
            _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive"),
            _hit(rank=2, review_id="r2", user_id="u2", sentiment="negative"),
        )
    )
    aspect = assessment.businesses[0].aspects[0]
    assert aspect.consensus == "mixed"
    assert aspect.has_conflict is True
    assert aspect.response_mode == "uncertain"


def test_explicit_uncertainty_keeps_one_review_uncertain() -> None:
    assessment = _aggregator().aggregate(
        _request(
            _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive"),
            explicit_uncertainty=True,
        )
    )
    aspect = assessment.businesses[0].aspects[0]
    assert aspect.confidence_level == "low"
    assert aspect.response_mode == "uncertain"
    assert aspect.requires_caveat is True


def test_negative_requested_polarity_reverses_stance() -> None:
    assessment = _aggregator().aggregate(
        _request(
            _hit(rank=1, review_id="r1", user_id="u1", sentiment="negative"),
            polarity="negative",
        )
    )
    assert assessment.businesses[0].aspects[0].consensus == "supports"


def test_policy_question_never_becomes_official_information() -> None:
    assessment = _aggregator().aggregate(
        _request(
            _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive"),
            task_type="official_policy_question",
        )
    )
    assert assessment.is_official_information is False
    assert assessment.recommend_official_verification is True


def test_aggregation_is_byte_deterministic() -> None:
    request = _request(
        _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive"),
        _hit(rank=2, review_id="r2", user_id="u2", sentiment="positive"),
    )
    first = _aggregator().aggregate(request).model_dump_json()
    second = _aggregator().aggregate(request).model_dump_json()
    assert first == second


def test_agent_tool_consumes_only_the_existing_locked_search_result() -> None:
    search_result = _request(
        _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive")
    ).search_result
    context = ToolExecutionContext(
        request_id="a" * 64,
        user_id="u1",
        cutoff_time=_CUTOFF,
        action="retrieve_business_reviews",
        business_scope=("business-1",),
        business_scope_known=True,
        state_snapshot={
            "turn_index": 1,
            "request": {
                "query_text": "Is the quiet environment good?",
                "conditions": [],
            },
            "readiness": {"task_type": "review_experience_question"},
            "observations": [
                {
                    "turn_index": 1,
                    "payload": {
                        "tool_name": "SEARCH_BUSINESS_REVIEWS",
                        "status": "success",
                        "data": search_result.model_dump(mode="json"),
                    },
                }
            ],
        },
    )
    observation = AggregateReviewEvidenceTool(_aggregator()).run(
        AggregateReviewEvidenceInput(business_ids=["business-1"]),
        context,
    )
    assert observation.status == "success"
    assert observation.data["business_ids"] == ["business-1"]
    assert observation.evidence[0].review_id == "r1"
    assert observation.input_tokens == 0


def test_agent_tool_rejects_scope_different_from_review_search() -> None:
    search_result = _request(
        _hit(rank=1, review_id="r1", user_id="u1", sentiment="positive")
    ).search_result
    context = ToolExecutionContext(
        request_id="a" * 64,
        user_id="u1",
        cutoff_time=_CUTOFF,
        action="retrieve_business_reviews",
        state_snapshot={
            "turn_index": 1,
            "request": {"query_text": "quiet?", "conditions": []},
            "readiness": {"task_type": "review_experience_question"},
            "observations": [
                {
                    "turn_index": 1,
                    "payload": {
                        "tool_name": "SEARCH_BUSINESS_REVIEWS",
                        "status": "success",
                        "data": search_result.model_dump(mode="json"),
                    },
                }
            ],
        },
    )
    try:
        AggregateReviewEvidenceTool(_aggregator()).run(
            AggregateReviewEvidenceInput(business_ids=["business-2"]),
            context,
        )
    except PermanentToolError as exc:
        assert "scope" in str(exc)
    else:
        raise AssertionError("scope mismatch must be rejected")


def test_frozen_policy_is_development_only() -> None:
    config = load_evidence_aggregation_config(
        _PROJECT_ROOT / "configs" / "evidence_aggregator.yaml"
    )
    policy = load_evidence_aggregation_policy(_PROJECT_ROOT, config)
    assert config.agent_version == "step28-evidence-aggregator-v1"
    assert policy.policy_version == "dev-sensitive-conflict-v1"
    assert policy.selection_split == "development"
    assert policy.validation_used_for_tuning is False
    assert policy.conflict_minority_mass_share == 0.15
