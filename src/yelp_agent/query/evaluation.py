"""External-label evaluation for the three Step 18 static ranking methods."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.query.engine import QueryAwareRecommendation

type QueryMethod = Literal["hybrid_v2", "query_only", "hybrid_query"]


class QueryRelevanceJudgment(StrictModel):
    """One label created outside the ranker and independent of next-business GT."""

    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    business_id: str = Field(min_length=1)
    relevance_grade: int = Field(ge=0, le=3)
    hard_constraint_status: Literal["satisfied", "violated", "unknown"]
    label_source: Literal["human", "llm_assisted"]
    evidence_refs: list[str] = Field(default_factory=list)


class QueryMethodMetrics(StrictModel):
    candidate_count: int = Field(ge=0)
    ndcg_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    hard_constraint_satisfaction_at_5: float = Field(ge=0, le=1)
    hard_constraint_unknown_rate_at_5: float = Field(ge=0, le=1)


class QueryAwareEvaluationReport(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    judgment_count: int = Field(ge=1)
    methods: dict[QueryMethod, QueryMethodMetrics]
    label_source_counts: dict[str, int]
    ground_truth_business_field_used: Literal[False] = False


def _metrics(
    ranking: Sequence[str],
    judgments: Mapping[str, QueryRelevanceJudgment],
) -> QueryMethodMetrics:
    grades = [judgments[business_id].relevance_grade for business_id in ranking]
    top = ranking[:5]
    dcg = sum(
        (2 ** judgments[business_id].relevance_grade - 1) / math.log2(rank + 1)
        for rank, business_id in enumerate(top, start=1)
    )
    ideal_grades = sorted(
        (judgment.relevance_grade for judgment in judgments.values()),
        reverse=True,
    )[:5]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    first_relevant = next(
        (rank for rank, grade in enumerate(grades, start=1) if grade > 0),
        None,
    )
    denominator = max(1, len(top))
    satisfied = sum(
        judgments[business_id].hard_constraint_status == "satisfied"
        for business_id in top
    )
    unknown = sum(
        judgments[business_id].hard_constraint_status == "unknown"
        for business_id in top
    )
    return QueryMethodMetrics(
        candidate_count=len(ranking),
        ndcg_at_5=0.0 if ideal_dcg == 0 else dcg / ideal_dcg,
        mrr=0.0 if first_relevant is None else 1.0 / first_relevant,
        hard_constraint_satisfaction_at_5=satisfied / denominator,
        hard_constraint_unknown_rate_at_5=unknown / denominator,
    )


def evaluate_query_aware_recommendation(
    comparison: QueryAwareRecommendation,
    judgments: Sequence[QueryRelevanceJudgment],
) -> QueryAwareEvaluationReport:
    """Compare outputs only after externally supplied relevance labels exist."""

    if not judgments:
        raise ValueError("query relevance judgments cannot be empty")
    by_business: dict[str, QueryRelevanceJudgment] = {}
    for judgment in judgments:
        if judgment.request_id != comparison.request.request_id:
            raise ValueError("judgment request_id does not match the recommendation")
        if judgment.business_id in by_business:
            raise ValueError("query relevance judgments must be unique")
        by_business[judgment.business_id] = judgment
    expected = set(comparison.hybrid_v2_ranking)
    if set(by_business) != expected:
        raise ValueError("judgments must cover the complete candidate universe")
    rankings: dict[QueryMethod, list[str]] = {
        "hybrid_v2": comparison.hybrid_v2_ranking,
        "query_only": [item.business_id for item in comparison.query_only.ranking],
        "hybrid_query": [item.business_id for item in comparison.hybrid_query.ranking],
    }
    return QueryAwareEvaluationReport(
        request_id=comparison.request.request_id,
        judgment_count=len(judgments),
        methods={
            method: _metrics(ranking, by_business)
            for method, ranking in rankings.items()
        },
        label_source_counts=dict(
            sorted(Counter(item.label_source for item in judgments).items())
        ),
    )
