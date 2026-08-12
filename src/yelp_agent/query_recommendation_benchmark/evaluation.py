"""Hidden-label evaluation for the three approved retrieval comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel

from .schema import (
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)

type RetrievalMethod = Literal["history_only", "query_only", "history_query"]
_METHODS: tuple[RetrievalMethod, ...] = (
    "history_only",
    "query_only",
    "history_query",
)


class BenchmarkRetrievalRun(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: RetrievalMethod
    ranking: list[str] = Field(max_length=500)
    latency_ms: float = Field(ge=0)
    embedding_encoded_tokens: int = Field(default=0, ge=0)
    embedding_logical_tokens: int = Field(default=0, ge=0)
    embedding_provider_calls: int = Field(default=0, ge=0)

    @field_validator("ranking")
    @classmethod
    def validate_ranking(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("retrieval ranking IDs must be unique")
        if any(not value for value in values):
            raise ValueError("retrieval ranking IDs must be nonempty")
        return values


class QueryRetrievalMethodMetrics(StrictModel):
    case_count: int = Field(ge=1)
    recall_at_50: float = Field(ge=0, le=1)
    recall_at_100: float = Field(ge=0, le=1)
    recall_at_500: float = Field(ge=0, le=1)
    mrr_at_500: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)


class QueryRecommendationRetrievalReport(StrictModel):
    case_count: int = Field(ge=1)
    overall: dict[RetrievalMethod, QueryRetrievalMethodMetrics]
    by_split: dict[str, dict[RetrievalMethod, QueryRetrievalMethodMetrics]]
    query_rescue_rate: float = Field(ge=0, le=1)
    query_rescue_count: int = Field(ge=0)
    history_miss_count: int = Field(ge=0)
    fusion_loss_rate: float = Field(ge=0, le=1)
    fusion_loss_count: int = Field(ge=0)
    either_channel_hit_count: int = Field(ge=0)
    execution_embedding_encoded_tokens: int = Field(ge=0)
    execution_embedding_logical_tokens: int = Field(ge=0)
    execution_embedding_provider_calls: int = Field(ge=0)
    external_llm_tokens: Literal[0] = 0
    performance_claim_scope: Literal["single_known_positive_retrieval"] = (
        "single_known_positive_retrieval"
    )
    ablation_experiments_included: Literal[False] = False


def _metrics(
    case_ids: list[str],
    target_by_case: dict[str, str],
    run_by_key: dict[tuple[str, str], BenchmarkRetrievalRun],
    method: RetrievalMethod,
) -> QueryRetrievalMethodMetrics:
    ranks = []
    latencies = []
    for case_id in case_ids:
        run = run_by_key[(case_id, method)]
        target = target_by_case[case_id]
        try:
            rank = run.ranking.index(target) + 1
        except ValueError:
            rank = None
        ranks.append(rank)
        latencies.append(run.latency_ms)
    count = len(case_ids)
    return QueryRetrievalMethodMetrics(
        case_count=count,
        recall_at_50=sum(rank is not None and rank <= 50 for rank in ranks) / count,
        recall_at_100=sum(rank is not None and rank <= 100 for rank in ranks) / count,
        recall_at_500=sum(rank is not None and rank <= 500 for rank in ranks) / count,
        mrr_at_500=sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count,
        mean_latency_ms=sum(latencies) / count,
    )


def evaluate_query_recommendation_retrieval(
    visible_cases: Sequence[VisibleQueryRecommendationCase],
    ground_truth: Sequence[QueryRecommendationGroundTruth],
    runs: Sequence[BenchmarkRetrievalRun],
) -> QueryRecommendationRetrievalReport:
    """Load hidden positives only here and compare the three approved methods."""

    case_ids = [item.case_id for item in visible_cases]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("visible cases must be nonempty and unique")
    target_by_case = {item.case_id: item.target_business_id for item in ground_truth}
    if set(target_by_case) != set(case_ids) or len(target_by_case) != len(ground_truth):
        raise ValueError("ground truth must align exactly with visible cases")
    run_by_key = {(item.case_id, item.method): item for item in runs}
    expected = {(case_id, method) for case_id in case_ids for method in _METHODS}
    if set(run_by_key) != expected or len(run_by_key) != len(runs):
        raise ValueError("runs must contain each case and approved method exactly once")
    overall = {
        method: _metrics(case_ids, target_by_case, run_by_key, method)
        for method in _METHODS
    }
    cases_by_split: defaultdict[str, list[str]] = defaultdict(list)
    for item in visible_cases:
        cases_by_split[item.split].append(item.case_id)
    by_split = {
        split: {
            method: _metrics(values, target_by_case, run_by_key, method)
            for method in _METHODS
        }
        for split, values in sorted(cases_by_split.items())
    }
    history_misses = 0
    rescued = 0
    either_hits = 0
    fusion_losses = 0
    for case_id in case_ids:
        target = target_by_case[case_id]
        history_hit = target in run_by_key[(case_id, "history_only")].ranking
        query_hit = target in run_by_key[(case_id, "query_only")].ranking
        fusion_hit = target in run_by_key[(case_id, "history_query")].ranking
        if not history_hit:
            history_misses += 1
            rescued += int(query_hit)
        if history_hit or query_hit:
            either_hits += 1
            fusion_losses += int(not fusion_hit)
    return QueryRecommendationRetrievalReport(
        case_count=len(case_ids),
        overall=overall,
        by_split=by_split,
        query_rescue_rate=(rescued / history_misses if history_misses else 0.0),
        query_rescue_count=rescued,
        history_miss_count=history_misses,
        fusion_loss_rate=(fusion_losses / either_hits if either_hits else 0.0),
        fusion_loss_count=fusion_losses,
        either_channel_hit_count=either_hits,
        execution_embedding_encoded_tokens=sum(
            item.embedding_encoded_tokens for item in runs
        ),
        execution_embedding_logical_tokens=sum(
            item.embedding_logical_tokens for item in runs
        ),
        execution_embedding_provider_calls=sum(
            item.embedding_provider_calls for item in runs
        ),
    )
