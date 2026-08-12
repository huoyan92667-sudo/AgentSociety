"""Hidden-label evaluation for Step 33 retrieval and ranking funnels."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, field_validator

from yelp_agent.models import StrictModel
from yelp_agent.query_recommendation_benchmark import (
    QueryRecommendationGroundTruth,
    VisibleQueryRecommendationCase,
)

from .schema import PreparedQueryAwareCase, QueryAwareRankingResult


type QueryAwareMethod = Literal[
    "history_lightgbm",
    "query_only",
    "rrf_fusion",
    "query_lightgbm_coarse",
    "query_aware_final",
]

METHODS: tuple[QueryAwareMethod, ...] = (
    "history_lightgbm",
    "query_only",
    "rrf_fusion",
    "query_lightgbm_coarse",
    "query_aware_final",
)


class QueryAwareBenchmarkRun(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["development", "validation"]
    method: QueryAwareMethod
    ranking: list[str] = Field(max_length=1000)
    latency_ms: float = Field(ge=0)

    @field_validator("ranking")
    @classmethod
    def validate_ranking(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError("benchmark rankings must contain unique nonempty IDs")
        return values


class SinglePositiveRankingMetrics(StrictModel):
    case_count: int = Field(ge=1)
    hr_at_1: float = Field(ge=0, le=1)
    hr_at_3: float = Field(ge=0, le=1)
    hr_at_5: float = Field(ge=0, le=1)
    hr_at_10: float = Field(ge=0, le=1)
    avg_hr_1_3_5_10: float = Field(ge=0, le=1)
    recall_at_50: float = Field(ge=0, le=1)
    recall_at_100: float = Field(ge=0, le=1)
    recall_at_500: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


class QueryAwareEvaluationReport(StrictModel):
    case_count: int = Field(ge=1)
    overall: dict[QueryAwareMethod, SinglePositiveRankingMetrics]
    by_split: dict[str, dict[QueryAwareMethod, SinglePositiveRankingMetrics]]
    protected_union_recall: float = Field(ge=0, le=1)
    post_filter_union_recall: float = Field(ge=0, le=1)
    target_filter_loss_rate: float = Field(ge=0, le=1)
    query_candidate_preservation_rate: float = Field(ge=0, le=1)
    old_rrf_fusion_loss_rate: float = Field(ge=0, le=1)
    coarse_top_500_loss_rate: float = Field(ge=0, le=1)
    hard_constraint_violation_at_10: float = Field(ge=0, le=1)
    fallback_rate: float = Field(ge=0, le=1)
    embedding_input_tokens: int = Field(ge=0)
    cross_encoder_input_tokens: int = Field(ge=0)
    logical_input_tokens: int = Field(ge=0)
    external_model_calls: Literal[0] = 0
    validation_used_for_selection: Literal[False] = False


def metrics_for_rankings(
    rankings: Sequence[Sequence[str]],
    targets: Sequence[str],
    *,
    latencies: Sequence[float] | None = None,
) -> SinglePositiveRankingMetrics:
    if not rankings or len(rankings) != len(targets):
        raise ValueError("rankings and targets must be nonempty and aligned")
    if latencies is None:
        latencies = [0.0] * len(rankings)
    if len(latencies) != len(rankings):
        raise ValueError("latencies must align with rankings")
    ranks: list[int | None] = []
    for ranking, target in zip(rankings, targets, strict=True):
        try:
            ranks.append(list(ranking).index(target) + 1)
        except ValueError:
            ranks.append(None)
    count = len(ranks)

    def hit(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / count

    def ndcg(k: int) -> float:
        return sum(
            0.0
            if rank is None or rank > k
            else 1.0 / math.log2(rank + 1)
            for rank in ranks
        ) / count

    hr_1, hr_3, hr_5, hr_10 = (hit(value) for value in (1, 3, 5, 10))
    ordered_latencies = sorted(float(value) for value in latencies)
    percentile_index = (len(ordered_latencies) - 1) * 0.95
    lower_index = math.floor(percentile_index)
    upper_index = math.ceil(percentile_index)
    fraction = percentile_index - lower_index
    p95_latency = ordered_latencies[lower_index] + fraction * (
        ordered_latencies[upper_index] - ordered_latencies[lower_index]
    )
    return SinglePositiveRankingMetrics(
        case_count=count,
        hr_at_1=hr_1,
        hr_at_3=hr_3,
        hr_at_5=hr_5,
        hr_at_10=hr_10,
        avg_hr_1_3_5_10=(hr_1 + hr_3 + hr_5 + hr_10) / 4.0,
        recall_at_50=hit(50),
        recall_at_100=hit(100),
        recall_at_500=hit(500),
        mrr=sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count,
        ndcg_at_5=ndcg(5),
        ndcg_at_10=ndcg(10),
        mean_latency_ms=sum(float(value) for value in latencies) / count,
        p95_latency_ms=p95_latency,
    )


def build_benchmark_runs(
    prepared: Sequence[PreparedQueryAwareCase],
    final_results: Sequence[QueryAwareRankingResult],
    coarse_rankings: Mapping[str, Sequence[str]],
) -> tuple[QueryAwareBenchmarkRun, ...]:
    final_by_case = {item.case_id: item for item in final_results}
    if len(final_by_case) != len(final_results):
        raise ValueError("final results contain duplicate cases")
    output: list[QueryAwareBenchmarkRun] = []
    for item in prepared:
        final = final_by_case[item.case_id]
        methods = {
            "history_lightgbm": item.history_lightgbm_ranking,
            "query_only": item.query_ranking,
            "rrf_fusion": item.rrf_fusion_ranking,
            "query_lightgbm_coarse": list(coarse_rankings[item.case_id]),
            "query_aware_final": final.ranking,
        }
        for method in METHODS:
            output.append(
                QueryAwareBenchmarkRun(
                    case_id=item.case_id,
                    split=item.split,
                    method=method,
                    ranking=methods[method],
                    latency_ms=(
                        final.latency_ms
                        if method == "query_aware_final"
                        else item.preparation_latency_ms
                        if method in {"history_lightgbm", "query_lightgbm_coarse"}
                        else item.retrieval_latency_ms
                    ),
                )
            )
    return tuple(output)


def evaluate_query_aware_ranking(
    visible_cases: Sequence[VisibleQueryRecommendationCase],
    ground_truth: Sequence[QueryRecommendationGroundTruth],
    prepared: Sequence[PreparedQueryAwareCase],
    final_results: Sequence[QueryAwareRankingResult],
    runs: Sequence[QueryAwareBenchmarkRun],
) -> QueryAwareEvaluationReport:
    case_ids = [item.case_id for item in visible_cases]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("visible cases must be nonempty and unique")
    target_by_case = {item.case_id: item.target_business_id for item in ground_truth}
    if set(target_by_case) != set(case_ids) or len(target_by_case) != len(ground_truth):
        raise ValueError("ground truth must align exactly with visible cases")
    prepared_by_case = {item.case_id: item for item in prepared}
    final_by_case = {item.case_id: item for item in final_results}
    if set(prepared_by_case) != set(case_ids) or set(final_by_case) != set(case_ids):
        raise ValueError("prepared and final artifacts must align with visible cases")
    run_by_key = {(item.case_id, item.method): item for item in runs}
    expected = {(case_id, method) for case_id in case_ids for method in METHODS}
    if set(run_by_key) != expected or len(run_by_key) != len(runs):
        raise ValueError("runs must contain every case and method exactly once")

    def evaluate_subset(values: list[str]) -> dict[QueryAwareMethod, SinglePositiveRankingMetrics]:
        return {
            method: metrics_for_rankings(
                [run_by_key[(case_id, method)].ranking for case_id in values],
                [target_by_case[case_id] for case_id in values],
                latencies=[run_by_key[(case_id, method)].latency_ms for case_id in values],
            )
            for method in METHODS
        }

    cases_by_split: defaultdict[str, list[str]] = defaultdict(list)
    for item in visible_cases:
        cases_by_split[item.split].append(item.case_id)
    overall = evaluate_subset(case_ids)
    by_split = {
        split: evaluate_subset(values)
        for split, values in sorted(cases_by_split.items())
    }

    union_hits = 0
    post_filter_hits = 0
    filter_losses = 0
    query_hits = 0
    query_preserved = 0
    old_rrf_losses = 0
    coarse_losses = 0
    hard_violations = 0
    for case_id in case_ids:
        target = target_by_case[case_id]
        item = prepared_by_case[case_id]
        final = final_by_case[case_id]
        history_or_query = target in set(item.history_ranking).union(item.query_ranking)
        eligible = target in {row.business_id for row in item.candidates}
        query_hit = target in item.query_ranking
        union_hits += int(history_or_query)
        post_filter_hits += int(eligible)
        filter_losses += int(history_or_query and not eligible)
        query_hits += int(query_hit)
        query_preserved += int(query_hit and eligible)
        old_rrf_losses += int(history_or_query and target not in item.rrf_fusion_ranking)
        coarse_losses += int(
            eligible
            and target
            not in run_by_key[(case_id, "query_lightgbm_coarse")].ranking[:500]
        )
        excluded = {row.business_id for row in final.hard_exclusions}
        hard_violations += int(bool(excluded.intersection(final.ranking[:10])))
    count = len(case_ids)
    return QueryAwareEvaluationReport(
        case_count=count,
        overall=overall,
        by_split=by_split,
        protected_union_recall=union_hits / count,
        post_filter_union_recall=post_filter_hits / count,
        target_filter_loss_rate=(filter_losses / union_hits if union_hits else 0.0),
        query_candidate_preservation_rate=(
            query_preserved / query_hits if query_hits else 0.0
        ),
        old_rrf_fusion_loss_rate=(old_rrf_losses / union_hits if union_hits else 0.0),
        coarse_top_500_loss_rate=(coarse_losses / post_filter_hits if post_filter_hits else 0.0),
        hard_constraint_violation_at_10=hard_violations / count,
        fallback_rate=sum(item.fallback for item in final_results) / count,
        embedding_input_tokens=sum(item.embedding_input_tokens for item in final_results),
        cross_encoder_input_tokens=sum(
            item.cross_encoder_input_tokens for item in final_results
        ),
        logical_input_tokens=sum(item.logical_input_tokens for item in final_results),
    )
