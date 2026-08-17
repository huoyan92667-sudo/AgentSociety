"""Step 33 protected Query-aware retrieval and final ranking."""

from .candidate_pool import (
    ProtectedCandidatePool,
    ProtectedCandidatePoolResult,
    hybrid_candidate_rows,
)
from .config import (
    QueryAwareRankingConfig,
    QueryAwareRankingPolicy,
    load_query_aware_ranking_config,
    load_query_aware_ranking_policy,
    write_query_aware_ranking_policy,
)
from .engine import ExternalModelCallError, QueryAwareRecommendationEngine
from .evaluation import (
    METHODS,
    QueryAwareBenchmarkRun,
    QueryAwareEvaluationReport,
    SinglePositiveRankingMetrics,
    build_benchmark_runs,
    evaluate_query_aware_ranking,
    metrics_for_rankings,
)
from .schema import (
    CandidateEvidence,
    CandidatePoolItem,
    CoarseCandidateScore,
    HardConstraintExclusion,
    PreparedQueryAwareCase,
    QueryAwareRankingResult,
)
from .runtime import (
    OnlineQueryAwareRankingRuntime,
    QueryAwareFinalizationRuntime,
    QueryAwarePreparationRuntime,
    QueryAwareRankingSources,
)
from .scoring import apply_coarse_policy, rank_percentile, score_coarse_candidates
from .tuning import QueryWeightSelection, QueryWeightTrial, select_query_weight

__all__ = [
    "CandidateEvidence",
    "CandidatePoolItem",
    "CoarseCandidateScore",
    "ExternalModelCallError",
    "HardConstraintExclusion",
    "METHODS",
    "OnlineQueryAwareRankingRuntime",
    "PreparedQueryAwareCase",
    "ProtectedCandidatePool",
    "ProtectedCandidatePoolResult",
    "QueryAwareRankingConfig",
    "QueryAwareBenchmarkRun",
    "QueryAwareEvaluationReport",
    "QueryAwareRankingPolicy",
    "QueryAwareRankingResult",
    "QueryAwareRankingSources",
    "QueryAwareRecommendationEngine",
    "QueryAwarePreparationRuntime",
    "QueryAwareFinalizationRuntime",
    "QueryWeightSelection",
    "QueryWeightTrial",
    "SinglePositiveRankingMetrics",
    "apply_coarse_policy",
    "build_benchmark_runs",
    "evaluate_query_aware_ranking",
    "hybrid_candidate_rows",
    "load_query_aware_ranking_config",
    "load_query_aware_ranking_policy",
    "metrics_for_rankings",
    "rank_percentile",
    "score_coarse_candidates",
    "select_query_weight",
    "write_query_aware_ranking_policy",
]
