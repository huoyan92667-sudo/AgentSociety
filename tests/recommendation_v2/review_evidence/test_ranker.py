from datetime import UTC, datetime

from yelp_agent.recommendation_v2.review_evidence.ranker import ReviewEvidenceRanker
from yelp_agent.recommendation_v2.review_evidence.retrieval import ReviewRetrievalBatch
from yelp_agent.recommendation_v2.review_evidence.schema import ReviewRetrievalMetrics
from yelp_agent.recommendation_v2.schema import UnifiedRecommendationState
from yelp_agent.recommendation_v2.tools.hard_filter import StructuredHardFilterResult


class _MustNotBuildDescriptions:
    def build(self, *args, **kwargs):
        raise AssertionError("prepared fusion descriptions must bypass a second model call")


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
