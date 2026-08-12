"""Development-only coarse fusion selection for Step 33."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.semantic_ranking import SemanticRankingPolicy

from .config import QueryAwareRankingConfig, QueryAwareRankingPolicy
from .evaluation import SinglePositiveRankingMetrics, metrics_for_rankings
from .schema import PreparedQueryAwareCase
from .scoring import score_coarse_candidates


class QueryWeightTrial(StrictModel):
    query_weight: float = Field(ge=0, le=1)
    metrics: SinglePositiveRankingMetrics


class QueryWeightSelection(StrictModel):
    trials: list[QueryWeightTrial] = Field(min_length=1)
    policy: QueryAwareRankingPolicy


def select_query_weight(
    prepared: Sequence[PreparedQueryAwareCase],
    target_by_case: Mapping[str, str],
    config: QueryAwareRankingConfig,
    semantic_policy: SemanticRankingPolicy,
) -> QueryWeightSelection:
    development = [item for item in prepared if item.split == config.tuning_split]
    if not development:
        raise ValueError("development evidence is required for policy selection")
    if not {item.case_id for item in development}.issubset(target_by_case):
        raise ValueError("development labels do not align with prepared evidence")
    trials = []
    for query_weight in config.query_weight_candidates:
        rankings = []
        targets = []
        for item in development:
            scored = score_coarse_candidates(
                item.candidates,
                query_weight=query_weight,
                query_retrieval_signal_weight=config.query_retrieval_signal_weight,
                embedding_signal_weight=config.embedding_signal_weight,
            )
            rankings.append([row.business_id for row in scored])
            targets.append(target_by_case[item.case_id])
        trials.append(
            QueryWeightTrial(
                query_weight=query_weight,
                metrics=metrics_for_rankings(rankings, targets),
            )
        )
    selected = max(
        trials,
        key=lambda item: (
            item.metrics.avg_hr_1_3_5_10,
            item.metrics.mrr,
            item.metrics.hr_at_1,
            item.metrics.ndcg_at_10,
            -item.query_weight,
        ),
    )
    policy = QueryAwareRankingPolicy(
        selected_query_weight=selected.query_weight,
        query_retrieval_signal_weight=config.query_retrieval_signal_weight,
        embedding_signal_weight=config.embedding_signal_weight,
        union_candidate_limit=config.union_candidate_limit,
        coarse_diagnostic_limit=config.coarse_diagnostic_limit,
        semantic_candidate_limit=config.semantic_candidate_limit,
        internal_result_limit=config.internal_result_limit,
        display_limit=config.display_limit,
        tie_breakers=["MRR", "HR@1", "NDCG@10", "lower_query_weight"],
        development_case_count=len(development),
        development_metrics={
            key: float(value)
            for key, value in selected.metrics.model_dump().items()
            if key != "case_count"
        },
        semantic_policy=semantic_policy.model_copy(
            update={"candidate_limit": config.semantic_candidate_limit}
        ),
    )
    return QueryWeightSelection(trials=trials, policy=policy)
