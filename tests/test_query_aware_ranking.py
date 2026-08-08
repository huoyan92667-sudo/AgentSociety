from datetime import datetime
from pathlib import Path

from yelp_agent.business_profiles.schema import (
    BusinessAspectEvent,
    BusinessRatingEvent,
)
from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config
from yelp_agent.data.temporal_view import BusinessRecord
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query.adapters import candidate_from_business_profile
from yelp_agent.query.engine import QueryAwareRecommender
from yelp_agent.query.ranking import (
    CandidateAspectEvidence,
    QueryAwareCandidate,
    QueryAwareStaticRanker,
)

PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _candidate(
    business_id: str,
    *,
    hybrid_rank: int,
    categories: tuple[str, ...],
    quiet_positive_ratio: float,
    latitude: float = 39.9526,
    longitude: float = -75.1652,
) -> QueryAwareCandidate:
    return QueryAwareCandidate(
        business_id=business_id,
        hybrid_rank=hybrid_rank,
        categories=categories,
        latitude=latitude,
        longitude=longitude,
        price_level=None,
        aspect_evidence={
            "quiet_environment": CandidateAspectEvidence(
                status="known",
                positive_ratio=quiet_positive_ratio,
                negative_ratio=1.0 - quiet_positive_ratio,
                confidence=1.0,
                evidence_count=5,
            )
        },
    )


def test_query_only_and_hybrid_query_share_query_evidence_but_keep_modes_distinct() -> (
    None
):
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="想吃牛排，最好安静一点。",
        )
    )
    candidates = (
        _candidate(
            "loud-hybrid-favorite",
            hybrid_rank=1,
            categories=("Steakhouses",),
            quiet_positive_ratio=0.1,
        ),
        _candidate(
            "quiet-query-favorite",
            hybrid_rank=2,
            categories=("Steakhouses",),
            quiet_positive_ratio=0.9,
        ),
        _candidate(
            "quiet-wrong-category",
            hybrid_rank=3,
            categories=("Japanese",),
            quiet_positive_ratio=0.9,
        ),
    )
    ranker = QueryAwareStaticRanker(query_rrf_weight=2.0, rrf_constant=60.0)

    query_only = ranker.rank(request, candidates, mode="query_only")
    combined = ranker.rank(request, candidates, mode="hybrid_query")

    assert query_only.ranking[0].business_id == "quiet-query-favorite"
    assert combined.ranking[0].business_id == "quiet-query-favorite"
    assert query_only.ranking[0].query_score > query_only.ranking[-1].query_score
    assert all(item.constraint_status == "eligible" for item in combined.ranking)


def test_filterable_hard_constraints_remove_only_proven_failures() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="只想吃牛排，不要酒吧，必须在5公里以内。",
            user_latitude=39.9526,
            user_longitude=-75.1652,
        )
    )
    candidates = (
        _candidate(
            "eligible",
            hybrid_rank=3,
            categories=("Steakhouses",),
            quiet_positive_ratio=0.5,
        ),
        _candidate(
            "wrong-category",
            hybrid_rank=1,
            categories=("Japanese",),
            quiet_positive_ratio=0.5,
        ),
        _candidate(
            "bar-steak",
            hybrid_rank=2,
            categories=("Steakhouses", "Bars"),
            quiet_positive_ratio=0.5,
        ),
        _candidate(
            "too-far",
            hybrid_rank=4,
            categories=("Steakhouses",),
            quiet_positive_ratio=0.5,
            latitude=40.10,
            longitude=-75.1652,
        ),
    )

    result = QueryAwareStaticRanker().rank(
        request,
        candidates,
        mode="hybrid_query",
    )

    assert [item.business_id for item in result.ranking] == ["eligible"]
    assert {item.business_id for item in result.excluded} == {
        "wrong-category",
        "bar-steak",
        "too-far",
    }
    assert all(item.reason_codes for item in result.excluded)


def test_business_profile_adapter_uses_only_the_point_in_time_profile() -> None:
    business = BusinessRecord(
        business_id="business-1",
        name="Test Steakhouse",
        address="1 Test Street",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=39.95,
        longitude=-75.16,
        categories=("Restaurants", "Steakhouses"),
        attributes_json='{"RestaurantsPriceRange2":"3"}',
    )
    aspects = tuple(
        BusinessAspectEvent(
            review_id=f"aspect-{index}",
            business_id=business.business_id,
            user_id=f"user-{index}",
            review_time=datetime(2020, 1, index),
            aspect="quiet_environment",
            sentiment="positive",
            confidence=0.9,
            source_text_sha256="a" * 64,
            extractor_version="1.2.0",
        )
        for index in range(1, 4)
    )
    store = BusinessKnowledgeStore.from_records(
        businesses=(business,),
        rating_events=(
            BusinessRatingEvent(
                review_id="rating-1",
                business_id=business.business_id,
                user_id="user-1",
                stars=5,
                review_time=datetime(2020, 1, 1),
            ),
        ),
        aspect_events=aspects,
        config=load_business_profile_config(PROJECT_CONFIG_DIR),
    )
    profile = store.get([business.business_id], datetime(2021, 1, 1))[
        business.business_id
    ]

    candidate = candidate_from_business_profile(profile, hybrid_rank=7)

    assert candidate.business_id == "business-1"
    assert candidate.hybrid_rank == 7
    assert candidate.price_level == 3
    assert candidate.categories == ("Restaurants", "Steakhouses")
    assert candidate.aspect_evidence["quiet_environment"].status == "known"
    assert candidate.aspect_evidence["quiet_environment"].positive_ratio == 1.0


def test_agent_ready_recommender_returns_all_three_static_comparisons() -> None:
    candidates = (
        _candidate(
            "history-favorite",
            hybrid_rank=1,
            categories=("Steakhouses",),
            quiet_positive_ratio=0.1,
        ),
        _candidate(
            "current-request-favorite",
            hybrid_rank=2,
            categories=("Steakhouses",),
            quiet_positive_ratio=0.9,
        ),
    )
    recommender = QueryAwareRecommender(
        parser=build_rule_based_request_parser(),
        ranker=QueryAwareStaticRanker(query_rrf_weight=2.0),
    )

    result = recommender.recommend(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="想吃牛排，最好安静一点。",
        ),
        candidates,
    )

    assert result.hybrid_v2_ranking == [
        "history-favorite",
        "current-request-favorite",
    ]
    assert result.query_only.ranking[0].business_id == "current-request-favorite"
    assert result.hybrid_query.ranking[0].business_id == "current-request-favorite"
    assert "target" not in result.model_dump_json()


def test_multiple_required_categories_are_treated_as_allowed_alternatives() -> None:
    request = build_rule_based_request_parser().parse(
        QueryParseInput(
            user_id="user-1",
            session_id="session-1",
            cutoff_time=datetime(2024, 1, 20, 18, 0),
            query_text="只想吃牛排或者日料，不要酒吧。",
        )
    )
    candidates = (
        _candidate(
            "japanese-is-allowed",
            hybrid_rank=1,
            categories=("Japanese",),
            quiet_positive_ratio=0.5,
        ),
        _candidate(
            "unrelated",
            hybrid_rank=2,
            categories=("Italian",),
            quiet_positive_ratio=0.5,
        ),
    )

    result = QueryAwareStaticRanker().rank(
        request,
        candidates,
        mode="hybrid_query",
    )

    assert [item.business_id for item in result.ranking] == ["japanese-is-allowed"]
    assert [item.business_id for item in result.excluded] == ["unrelated"]
