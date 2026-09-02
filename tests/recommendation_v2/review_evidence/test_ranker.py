from datetime import UTC, datetime

from yelp_agent.recommendation_v2.business_facts import BusinessFact
from yelp_agent.recommendation_v2.review_evidence.ranker import ReviewEvidenceRanker
from yelp_agent.recommendation_v2.review_evidence.retrieval import ReviewRetrievalBatch
from yelp_agent.recommendation_v2.review_evidence.schema import (
    BusinessPreferenceEvidence,
    PreferenceSearchDescription,
    ReviewRetrievalMetrics,
)
from yelp_agent.recommendation_v2.schema import (
    RequirementBasis,
    SoftPreference,
    UnifiedRecommendationState,
    merchant_feature_for,
)
from yelp_agent.recommendation_v2.tools.hard_filter import (
    FilteredBusiness,
    StructuredHardFilterResult,
)


class _MustNotBuildDescriptions:
    def build(self, *args, **kwargs):
        raise AssertionError(
            "prepared fusion descriptions must bypass a second model call"
        )


class _EmptyRetriever:
    recall_threshold = 0.55
    acceptance_threshold = 0.60
    direction_margin = 0.05

    def retrieve_many(self, requirements, business_ids, *, cutoff_time=None):
        return ReviewRetrievalBatch(
            by_requirement={},
            metrics=ReviewRetrievalMetrics(),
        )

    def close(self) -> None:
        pass


class _FixedOfflineAssessments:
    """按测试给定的商家和偏好返回离线满足分。"""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores

    def assess_many(self, requirements, business_ids, *, reference_time):
        del reference_time
        return {
            requirement.requirement_id: {
                business_id: _assessment(
                    business_id,
                    requirement.requirement_id,
                    self._scores[(business_id, requirement.preference.field)],
                )
                for business_id in business_ids
            }
            for requirement in requirements
        }


def test_prepared_fusion_descriptions_bypass_description_model() -> None:
    ranker = ReviewEvidenceRanker(
        description_builder=_MustNotBuildDescriptions(),  # type: ignore[arg-type]
        retriever=_EmptyRetriever(),  # type: ignore[arg-type]
    )
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="我想吃牛排",
    )
    hard_filter = StructuredHardFilterResult(
        source_business_count=0,
        candidate_count=0,
        candidate_business_ids=[],
        candidates=[],
        steps=[],
        generated_sql="SELECT 1 WHERE FALSE",
        sql_parameters=[],
    )

    result = ranker.rank(
        state=state,
        hard_filter=hard_filter,
        reference_time=datetime(2026, 8, 26, tzinfo=UTC),
        prepared_descriptions=[],
    )

    assert result.status == "success"
    assert result.model_call_count == 0
    assert result.description_latency_ms >= 0


def test_priority_layers_compare_service_only_after_quiet_tier_is_equal() -> None:
    """更高优先级只比较档位；同档后才允许下一条偏好改变顺序。"""

    quiet = _preference("quiet_environment", priority=1)
    service = _preference("service", priority=2)
    parking = _preference("parking", priority=3)
    requirements = [_requirement(item) for item in [quiet, service, parking]]
    ranker = ReviewEvidenceRanker(
        description_builder=_MustNotBuildDescriptions(),  # type: ignore[arg-type]
        retriever=_EmptyRetriever(),  # type: ignore[arg-type]
        offline_aspects=_FixedOfflineAssessments(  # type: ignore[arg-type]
            {
                # b1和b2的安静分虽然不同，但都属于“比较满足”。
                # 因此第二层服务更好的b2应排在b1前面。
                ("b1", "quiet_environment"): 0.79,
                ("b1", "service"): 0.61,
                ("b1", "parking"): 0.90,
                ("b2", "quiet_environment"): 0.61,
                ("b2", "service"): 0.90,
                ("b2", "parking"): 0.20,
                # b3的服务和停车再好，也不能越过安静档位更高的b1、b2。
                ("b3", "quiet_environment"): 0.50,
                ("b3", "service"): 0.95,
                ("b3", "parking"): 0.95,
            }
        ),
    )
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="安静、服务好、停车方便的牛排店",
        soft_preferences=[quiet, service, parking],
    )
    candidates = [
        FilteredBusiness(business=_business("b1", rating=5.0), distance_km=1.0),
        FilteredBusiness(business=_business("b2", rating=4.0), distance_km=2.0),
        FilteredBusiness(business=_business("b3", rating=5.0), distance_km=0.1),
    ]
    hard_filter = StructuredHardFilterResult(
        source_business_count=3,
        candidate_count=3,
        candidate_business_ids=["b1", "b2", "b3"],
        candidates=candidates,
        steps=[],
        generated_sql="SELECT * FROM facts",
        sql_parameters=[],
    )

    result = ranker.rank(
        state=state,
        hard_filter=hard_filter,
        reference_time=datetime(2026, 9, 2, tzinfo=UTC),
        prepared_descriptions=requirements,
    )

    assert [item.business.business_id for item in result.ranking] == [
        "b2",
        "b1",
        "b3",
    ]
    assert [item.priority_layers[0].satisfaction_tier for item in result.ranking] == [
        3,
        3,
        2,
    ]


def test_equal_preference_tiers_fall_back_to_rating_then_review_count_then_distance() -> (
    None
):
    quiet = _preference("quiet_environment", priority=1)
    requirement = _requirement(quiet)
    ranker = ReviewEvidenceRanker(
        description_builder=_MustNotBuildDescriptions(),  # type: ignore[arg-type]
        retriever=_EmptyRetriever(),  # type: ignore[arg-type]
        offline_aspects=_FixedOfflineAssessments(  # type: ignore[arg-type]
            {
                ("b1", "quiet_environment"): 0.65,
                ("b2", "quiet_environment"): 0.75,
                ("b3", "quiet_environment"): 0.70,
            }
        ),
    )
    state = UnifiedRecommendationState(
        user_id="user-1",
        session_id="session-1",
        revision=1,
        turn_index=1,
        latest_query_text="安静的牛排店",
        soft_preferences=[quiet],
    )
    candidates = [
        FilteredBusiness(
            business=_business("b1", rating=4.5, reviews=100), distance_km=1.0
        ),
        FilteredBusiness(
            business=_business("b2", rating=5.0, reviews=10), distance_km=4.0
        ),
        FilteredBusiness(
            business=_business("b3", rating=4.5, reviews=200), distance_km=3.0
        ),
    ]
    hard_filter = StructuredHardFilterResult(
        source_business_count=3,
        candidate_count=3,
        candidate_business_ids=["b1", "b2", "b3"],
        candidates=candidates,
        steps=[],
        generated_sql="SELECT * FROM facts",
        sql_parameters=[],
    )

    result = ranker.rank(
        state=state,
        hard_filter=hard_filter,
        reference_time=datetime(2026, 9, 2, tzinfo=UTC),
        prepared_descriptions=[requirement],
    )

    assert [item.business.business_id for item in result.ranking] == [
        "b2",
        "b3",
        "b1",
    ]


def _preference(field: str, *, priority: int) -> SoftPreference:
    return SoftPreference.model_validate(
        {
            "key": f"current.{field}",
            "field": field,
            "direction": "higher",
            "preference_strength": 100,
            "priority": priority,
            "merchant_feature": merchant_feature_for(field),
            "controlling_source": "current_query",
            "sources": [
                RequirementBasis(
                    source="current_query",
                    text=f"test {field}",
                    turn_index=1,
                    preference_strength=100,
                )
            ],
        }
    )


def _requirement(preference: SoftPreference) -> PreferenceSearchDescription:
    return PreferenceSearchDescription(
        requirement_id=preference.key,
        requirement_text=preference.sources[0].text,
        kind="fixed_aspect",
        priority=preference.priority,
        preference_strength=preference.preference_strength,
        positive_descriptions=["positive one", "positive two"],
        negative_descriptions=["negative one", "negative two"],
        preference=preference,
    )


def _assessment(
    business_id: str,
    requirement_id: str,
    score: float,
) -> BusinessPreferenceEvidence:
    return BusinessPreferenceEvidence(
        business_id=business_id,
        requirement_id=requirement_id,
        evidence_source="offline_business_profile",
        positive_evidence=[],
        negative_evidence=[],
        positive_component=max(0.0, 2 * (score - 0.5)),
        negative_component=max(0.0, 2 * (0.5 - score)),
        positive_count_reliability=0,
        negative_count_reliability=0,
        evidence_score=score,
        recalled_review_count=10,
        ambiguous_review_count=0,
        evidence_sufficiency=1,
        evidence_sufficiency_level="充分",
        controversy=0,
        controversy_level="低",
        usable_for_ranking=True,
    )


def _business(
    business_id: str,
    *,
    rating: float,
    reviews: int = 100,
) -> BusinessFact:
    return BusinessFact(
        business_id=business_id,
        name=f"Restaurant {business_id}",
        address="1 Main Street",
        city="Philadelphia",
        state="PA",
        postal_code="19107",
        latitude=39.95,
        longitude=-75.16,
        categories=["Restaurants", "Steakhouses"],
        rating=rating,
        review_count=reviews,
    )
