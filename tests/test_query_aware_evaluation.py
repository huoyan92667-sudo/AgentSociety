from datetime import datetime

from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query.engine import QueryAwareRecommender
from yelp_agent.query.evaluation import (
    QueryRelevanceJudgment,
    evaluate_query_aware_recommendation,
)
from yelp_agent.query.ranking import (
    CandidateAspectEvidence,
    QueryAwareCandidate,
    QueryAwareStaticRanker,
)


def test_evaluator_compares_three_methods_against_external_relevance_labels() -> None:
    candidates = (
        QueryAwareCandidate(
            business_id="history-favorite",
            hybrid_rank=1,
            categories=("Steakhouses",),
            aspect_evidence={
                "quiet_environment": CandidateAspectEvidence(
                    status="known",
                    positive_ratio=0.1,
                    negative_ratio=0.9,
                    confidence=1.0,
                    evidence_count=5,
                )
            },
        ),
        QueryAwareCandidate(
            business_id="request-favorite",
            hybrid_rank=2,
            categories=("Steakhouses",),
            aspect_evidence={
                "quiet_environment": CandidateAspectEvidence(
                    status="known",
                    positive_ratio=0.9,
                    negative_ratio=0.1,
                    confidence=1.0,
                    evidence_count=5,
                )
            },
        ),
    )
    comparison = QueryAwareRecommender(
        parser=build_rule_based_request_parser(),
        ranker=QueryAwareStaticRanker(query_rrf_weight=2.0),
    ).recommend(
        QueryParseInput(
            user_id="u1",
            session_id="s1",
            cutoff_time=datetime(2024, 1, 1),
            query_text="想吃牛排，最好安静一点。",
        ),
        candidates,
    )
    judgments = (
        QueryRelevanceJudgment(
            request_id=comparison.request.request_id,
            business_id="history-favorite",
            relevance_grade=1,
            hard_constraint_status="satisfied",
            label_source="human",
        ),
        QueryRelevanceJudgment(
            request_id=comparison.request.request_id,
            business_id="request-favorite",
            relevance_grade=3,
            hard_constraint_status="satisfied",
            label_source="human",
        ),
    )

    report = evaluate_query_aware_recommendation(comparison, judgments)

    assert report.methods["query_only"].ndcg_at_5 == 1.0
    assert report.methods["hybrid_query"].ndcg_at_5 == 1.0
    assert (
        report.methods["hybrid_query"].ndcg_at_5 > report.methods["hybrid_v2"].ndcg_at_5
    )
    assert report.methods["hybrid_query"].hard_constraint_satisfaction_at_5 == 1.0
    assert report.label_source_counts == {"human": 2}
