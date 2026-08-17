"""Production assembly of the frozen Step 24 Rule Agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from yelp_agent.agent_harness import AgentHarness, load_agent_harness_config
from yelp_agent.agent_tools import (
    HybridV2FallbackHandler,
    OnlineHybridV2RankingService,
    RankingCascadeFallbackHandler,
    build_step23_tool_registry,
    load_agent_tool_runtime_config,
)
from yelp_agent.business_profiles import BusinessKnowledgeStore
from yelp_agent.collaborative import TemporalItemKNNStore
from yelp_agent.config import (
    load_business_profile_config,
    load_config,
    load_item_knn_config,
    load_retrieval_config,
    load_review_aspect_settings,
)
from yelp_agent.controlled_llm import (
    ControlledLLMRuntime,
    build_controlled_llm_runtime,
    load_controlled_llm_config,
)
from yelp_agent.cross_encoder import (
    CachedCrossEncoderReranker,
    LocalQwenCrossEncoder,
    SqliteCrossEncoderCache,
    load_cross_encoder_config,
    load_cross_encoder_policy,
    load_local_cross_encoder_environment,
)
from yelp_agent.evidence_aggregation import (
    EvidenceAggregator,
    load_evidence_aggregation_config,
    load_evidence_aggregation_policy,
)
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.learning_to_rank import FrozenLambdaMARTRanker
from yelp_agent.learning_to_rank.features import HybridV1Weights
from yelp_agent.profiles.store import UserProfileStore
from yelp_agent.ranking.assembly import HybridSourcePaths, build_frozen_hybrid_runtime
from yelp_agent.retrieval import MultiRouteRetriever
from yelp_agent.query_retrieval import (
    DualChannelFusion,
    QueryCandidateRetriever,
    load_query_retrieval_config,
)
from yelp_agent.query_aware_ranking import (
    OnlineQueryAwareRankingRuntime,
    QueryAwareRankingSources,
    load_query_aware_ranking_config,
    load_query_aware_ranking_policy,
)
from yelp_agent.review_rag import (
    ReviewRAGStore,
    ReviewRetriever,
    load_review_rag_config,
    load_review_rag_policy,
)
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    DashScopeEmbeddingEncoder,
    LocalQwenEmbeddingEncoder,
    SemanticEmbeddingMatcher,
    SqliteEmbeddingCache,
    load_dashscope_embedding_environment,
    load_local_embedding_environment,
    load_semantic_embedding_config,
)
from yelp_agent.semantic_ranking import (
    SemanticRankingEngine,
    load_semantic_ranking_config,
    load_semantic_ranking_policy,
)
from yelp_agent.session_memory.config import load_session_memory_config
from yelp_agent.session_memory.runtime import (
    SessionMemoryRuntime,
    build_session_memory_runtime,
)

from .config import load_rule_router_config
from .factory import build_rule_agent
from .router import RuleRouter

if TYPE_CHECKING:
    from yelp_agent.constrained_llm_router import ConstrainedLLMRouterRuntime


@dataclass(frozen=True, slots=True)
class RuleAgentSourcePaths:
    """Name every frozen input used by the real Rule Agent runtime."""

    project_root: Path
    config_dir: Path
    businesses: Path
    reviews: Path
    interactions: Path
    histories: Path
    tfidf_artifact: Path
    tfidf_manifest: Path
    hybrid_weights: Path
    user_profile_root: Path
    business_profile_root: Path
    hybrid_v2_root: Path
    item_knn_positive: Path
    item_knn_negative: Path
    item_knn_neutral: Path
    benchmark_root: Path
    rule_router_config: Path
    agent_harness_config: Path
    agent_tools_config: Path
    review_aspects: Path
    review_rag_root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> RuleAgentSourcePaths:
        root = Path(project_root).resolve()
        config_dir = root / "configs"
        return cls(
            project_root=root,
            config_dir=config_dir,
            businesses=root / "data" / "processed" / "businesses.parquet",
            reviews=root / "data" / "processed" / "reviews.parquet",
            interactions=root / "data" / "processed" / "interactions.parquet",
            histories=(
                root / "data" / "task_dataset" / "tasks" / "temporal_histories.parquet"
            ),
            tfidf_artifact=root / "data" / "features" / "tfidf_vectorizer.joblib",
            tfidf_manifest=root / "data" / "features" / "tfidf_manifest.json",
            hybrid_weights=root / "runs" / "hybrid" / "hybrid_weights.json",
            user_profile_root=root / "data" / "features" / "user_profiles" / "v1",
            business_profile_root=(
                root / "data" / "features" / "business_profiles" / "v1"
            ),
            hybrid_v2_root=root / "runs" / "hybrid_v2_b" / "frozen",
            item_knn_positive=(
                root / "data" / "features" / "item_knn" / "positive_events.parquet"
            ),
            item_knn_negative=(
                root / "data" / "features" / "item_knn" / "negative_events.parquet"
            ),
            item_knn_neutral=(
                root / "data" / "features" / "item_knn" / "neutral_events.parquet"
            ),
            benchmark_root=root / "benchmarks" / "agent_scenarios_v1",
            rule_router_config=config_dir / "rule_router.yaml",
            agent_harness_config=config_dir / "agent_harness.yaml",
            agent_tools_config=config_dir / "agent_tools.yaml",
            review_aspects=(
                root
                / "data"
                / "features"
                / "review_aspects"
                / "development"
                / "aspect_records.parquet"
            ),
            review_rag_root=root / "data" / "features" / "review_rag" / "v1",
        )

    def required_files(self) -> tuple[Path, ...]:
        """Return the complete fail-fast input contract for Step 24."""

        return (
            self.businesses,
            self.reviews,
            self.interactions,
            self.histories,
            self.tfidf_artifact,
            self.tfidf_manifest,
            self.hybrid_weights,
            self.user_profile_root / "manifest.json",
            self.user_profile_root / "profile_snapshots.parquet",
            self.user_profile_root / "preference_signals.parquet",
            self.user_profile_root / "task_profile_map.parquet",
            self.business_profile_root / "manifest.json",
            self.business_profile_root / "businesses.parquet",
            self.business_profile_root / "rating_events.parquet",
            self.business_profile_root / "aspect_events.parquet",
            self.business_profile_root / "business_coverage.parquet",
            self.hybrid_v2_root / "manifest.json",
            self.hybrid_v2_root / "model.joblib",
            self.item_knn_positive,
            self.item_knn_negative,
            self.item_knn_neutral,
            self.benchmark_root / "visible" / "scenarios.jsonl",
            self.benchmark_root / "hidden" / "ground_truth.jsonl",
            self.benchmark_root / "hidden" / "evidence_labels.parquet",
            self.rule_router_config,
            self.agent_harness_config,
            self.agent_tools_config,
            self.config_dir / "retrieval.yaml",
            self.config_dir / "item_knn.yaml",
            self.config_dir / "business_profiles.yaml",
            self.config_dir / "query_retrieval.yaml",
        )

    def validate(self) -> None:
        missing = [path for path in self.required_files() if not path.is_file()]
        if missing:
            joined = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Rule Agent inputs are incomplete:\n{joined}")


class RuleAgentRuntime:
    """Own the heavy, read-only data stores behind one reusable Harness."""

    def __init__(
        self,
        *,
        sources: RuleAgentSourcePaths,
        harness: AgentHarness,
        user_profiles: UserProfileStore,
        embedding_encoder: object | None = None,
        cross_encoder: object | None = None,
        review_embedding_encoder: object | None = None,
        review_store: object | None = None,
        controlled_llm: ControlledLLMRuntime | None = None,
        semantic_ranking: SemanticRankingEngine | None = None,
        query_retrieval: QueryCandidateRetriever | None = None,
        session_memory: SessionMemoryRuntime | None = None,
        constrained_router: ConstrainedLLMRouterRuntime | None = None,
        query_aware_ranking: OnlineQueryAwareRankingRuntime | None = None,
    ) -> None:
        self.sources = sources
        self.harness = harness
        self._user_profiles = user_profiles
        self._embedding_encoder = embedding_encoder
        self._cross_encoder = cross_encoder
        self._review_embedding_encoder = review_embedding_encoder
        self._review_store = review_store
        self.controlled_llm = controlled_llm
        self.semantic_ranking = semantic_ranking
        self.query_retrieval = query_retrieval
        self.session_memory = session_memory
        self.constrained_router = constrained_router
        self.query_aware_ranking = query_aware_ranking
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._user_profiles.close()
            close = getattr(self._embedding_encoder, "close", None)
            if callable(close):
                close()
            close = getattr(self._cross_encoder, "close", None)
            if callable(close):
                close()
            close = getattr(self._review_embedding_encoder, "close", None)
            if callable(close):
                close()
            close = getattr(self._review_store, "close", None)
            if callable(close):
                close()
            close = getattr(self.controlled_llm, "close", None)
            if callable(close):
                close()
            close = getattr(self.session_memory, "close", None)
            if callable(close):
                close()
            close = getattr(self.constrained_router, "close", None)
            if callable(close):
                close()
            close = getattr(self.query_aware_ranking, "close", None)
            if callable(close):
                close()
            self._closed = True


def build_real_rule_agent_runtime(
    sources: RuleAgentSourcePaths,
    *,
    embedding_config_path: str | Path | None = None,
    embedding_environment: Mapping[str, str] | None = None,
    cross_encoder_config_path: str | Path | None = None,
    cross_encoder_environment: Mapping[str, str] | None = None,
    review_rag_config_path: str | Path | None = None,
    review_embedding_environment: Mapping[str, str] | None = None,
    evidence_aggregation_config_path: str | Path | None = None,
    controlled_llm_config_path: str | Path | None = None,
    controlled_llm_environment: Mapping[str, str] | None = None,
    session_memory_config_path: str | Path | None = None,
    session_memory_environment: Mapping[str, str] | None = None,
    semantic_ranking_config_path: str | Path | None = None,
    constrained_router_config_path: str | Path | None = None,
    constrained_router_environment: Mapping[str, str] | None = None,
    agent_harness_config_path: str | Path | None = None,
    query_retrieval_mode: Literal["config", "history_only"] = "config",
    query_aware_ranking_config_path: str | Path | None = None,
    query_aware_ranking_policy_path: str | Path | None = None,
) -> RuleAgentRuntime:
    """Load all frozen artifacts once and assemble the production Rule Agent."""

    sources.validate()
    app_config = load_config(sources.config_dir)
    retrieval_config = load_retrieval_config(sources.config_dir)
    item_knn_config = load_item_knn_config(sources.config_dir)
    business_config = load_business_profile_config(sources.config_dir)
    rule_config = load_rule_router_config(sources.rule_router_config)
    harness_config = load_agent_harness_config(
        agent_harness_config_path or sources.agent_harness_config
    )
    tool_config = load_agent_tool_runtime_config(sources.agent_tools_config)
    query_retrieval_config = load_query_retrieval_config(
        sources.config_dir / "query_retrieval.yaml"
    )

    frozen_hybrid = build_frozen_hybrid_runtime(
        app_config,
        HybridSourcePaths(
            businesses=sources.businesses,
            reviews=sources.reviews,
            interactions=sources.interactions,
            histories=sources.histories,
            tfidf_artifact=sources.tfidf_artifact,
            tfidf_manifest=sources.tfidf_manifest,
            config_dir=sources.config_dir,
        ),
        sources.hybrid_weights,
    )
    data_view = frozen_hybrid.assembly.data_view
    item_knn = TemporalItemKNNStore.from_event_artifacts(
        sources.item_knn_positive,
        sources.item_knn_negative,
        sources.item_knn_neutral,
        item_knn_config,
    )
    retriever = MultiRouteRetriever(
        data_view,
        category_store=TemporalCategoryStore(
            data_view,
            broad_categories=set(app_config.data.broad_categories),
        ),
        text_store=TemporalTextStore(
            data_view,
            sources.tfidf_artifact,
            sources.tfidf_manifest,
        ),
        quality_store=frozen_hybrid.assembly.quality_store,
        location_store=frozen_hybrid.assembly.location_store,
        config=retrieval_config,
        item_knn_store=item_knn,
    )
    user_profiles = UserProfileStore(sources.user_profile_root)
    encoder = None
    cross_encoder = None
    review_encoder = None
    review_store = None
    controlled_llm = None
    session_memory = None
    semantic_ranking = None
    constrained_router = None
    query_aware_ranking = None
    query_aware_policy = None
    try:
        business_profiles = BusinessKnowledgeStore.from_artifacts(
            sources.business_profile_root,
            config=business_config,
        )
        ranker = FrozenLambdaMARTRanker.from_artifacts(sources.hybrid_v2_root)
        ranking_service = OnlineHybridV2RankingService(
            user_profiles=user_profiles,
            business_profiles=business_profiles,
            ranker=ranker,
            weights=HybridV1Weights.model_validate(
                frozen_hybrid.weights.as_dict
            ),
            broad_categories=set(app_config.data.broad_categories),
            business_profile_config=business_config,
        )
        embedding_match = None
        semantic_config = None
        if embedding_config_path is not None:
            semantic_config = load_semantic_embedding_config(embedding_config_path)
            if semantic_config.provider == "dashscope":
                environment = load_dashscope_embedding_environment(
                    embedding_environment
                )
                if environment.enabled:
                    encoder = DashScopeEmbeddingEncoder.from_environment(
                        semantic_config,
                        environment,
                    )
            else:
                local_environment = load_local_embedding_environment(
                    embedding_environment
                )
                if local_environment.enabled:
                    encoder = LocalQwenEmbeddingEncoder.from_environment(
                        semantic_config,
                        local_environment,
                    )
            if encoder is None:
                raise ValueError(
                    f"{semantic_config.provider} embedding environment is not configured"
                )
            cache = SqliteEmbeddingCache(
                sources.project_root / semantic_config.cache_relative_path
            )
            embedding_match = SemanticEmbeddingMatcher(
                businesses=data_view,
                gateway=CachedEmbeddingGateway(
                    encoder=encoder,
                    cache=cache,
                    config=semantic_config,
                ),
                config=semantic_config,
            )
        cross_reranker = None
        cross_config = None
        cross_policy = None
        if cross_encoder_config_path is not None:
            if embedding_match is None or semantic_config is None:
                raise ValueError(
                    "Step 26 Cross-Encoder requires the Step 25 embedding runtime"
                )
            cross_config = load_cross_encoder_config(cross_encoder_config_path)
            cross_policy = load_cross_encoder_policy(
                sources.project_root, cross_config
            )
            cross_environment = load_local_cross_encoder_environment(
                cross_encoder_environment
            )
            if not cross_environment.enabled:
                raise ValueError("local Cross-Encoder environment is not configured")
            cross_encoder = LocalQwenCrossEncoder.from_environment(
                cross_config, cross_environment
            )
            cross_reranker = CachedCrossEncoderReranker(
                businesses=data_view,
                scorer=cross_encoder,
                cache=SqliteCrossEncoderCache(
                    sources.project_root / cross_config.cache_relative_path
                ),
                config=cross_config,
            )
        review_search = None
        review_config = None
        evidence_aggregator = None
        evidence_config = None
        if review_rag_config_path is not None:
            review_config = load_review_rag_config(review_rag_config_path)
            required_review_files = (
                sources.review_rag_root / "review_segments.parquet",
                sources.review_rag_root / "manifest.json",
                sources.review_aspects,
                sources.project_root / review_config.policy_relative_path,
            )
            missing_review = [path for path in required_review_files if not path.is_file()]
            if missing_review:
                raise FileNotFoundError(
                    "Review RAG inputs are incomplete:\n"
                    + "\n".join(f"- {path}" for path in missing_review)
                )
            review_environment = load_local_embedding_environment(
                review_embedding_environment or embedding_environment
            )
            if not review_environment.enabled:
                raise ValueError("local Review RAG embedding environment is not configured")
            review_semantic_config = review_config.semantic_config()
            review_encoder = LocalQwenEmbeddingEncoder.from_environment(
                review_semantic_config,
                review_environment,
            )
            review_store = ReviewRAGStore(
                sources.review_rag_root / "review_segments.parquet",
                sources.review_aspects,
                review_config,
            )
            _, review_vocabulary = load_review_aspect_settings(sources.config_dir)
            review_search = ReviewRetriever(
                store=review_store,
                config=review_config,
                policy=load_review_rag_policy(sources.project_root, review_config),
                vocabulary=review_vocabulary,
                embedding_gateway=CachedEmbeddingGateway(
                    encoder=review_encoder,
                    cache=SqliteEmbeddingCache(
                        sources.project_root
                        / review_config.embedding_cache_relative_path
                    ),
                    config=review_semantic_config,
                ),
            )
        if evidence_aggregation_config_path is not None:
            if review_search is None:
                raise ValueError(
                    "Step 28 evidence aggregation requires the Step 27 Review RAG runtime"
                )
            evidence_config = load_evidence_aggregation_config(
                evidence_aggregation_config_path
            )
            evidence_aggregator = EvidenceAggregator(
                load_evidence_aggregation_policy(
                    sources.project_root,
                    evidence_config,
                )
            )
        controlled_config = None
        if controlled_llm_config_path is not None:
            controlled_config = load_controlled_llm_config(
                controlled_llm_config_path
            )
            controlled_llm = build_controlled_llm_runtime(
                project_root=sources.project_root,
                config=controlled_config,
                environment=controlled_llm_environment,
            )
        semantic_ranking_config = None
        if semantic_ranking_config_path is not None:
            if embedding_match is None or cross_reranker is None:
                raise ValueError(
                    "Step 30 semantic ranking requires Step 25 and Step 26 runtimes"
                )
            semantic_ranking_config = load_semantic_ranking_config(
                semantic_ranking_config_path
            )
            if semantic_ranking_config.enabled:
                semantic_ranking_policy = load_semantic_ranking_policy(
                    sources.project_root,
                    semantic_ranking_config,
                )
                semantic_ranking = SemanticRankingEngine(
                    profiles=business_profiles,
                    embedding_matcher=embedding_match,
                    cross_encoder_reranker=cross_reranker,
                    policy=semantic_ranking_policy,
                    mode=semantic_ranking_config.mode,
                    candidate_limit=semantic_ranking_config.candidate_limit,
                )
        query_retrieval = None
        dual_channel_fusion = None
        if (
            query_retrieval_mode == "config"
            and query_retrieval_config.enabled
        ):
            query_retrieval = QueryCandidateRetriever(
                catalog=data_view,
                profiles=business_profiles,
                embedding_matcher=embedding_match,
                config=query_retrieval_config,
            )
            dual_channel_fusion = DualChannelFusion(query_retrieval_config)
        query_aware_config = None
        if query_aware_ranking_config_path is not None:
            query_aware_config = load_query_aware_ranking_config(
                query_aware_ranking_config_path
            )
            query_aware_policy = load_query_aware_ranking_policy(
                query_aware_ranking_policy_path
                or sources.project_root / query_aware_config.policy_relative_path
            )
            query_aware_embedding_environment = load_local_embedding_environment(
                embedding_environment
            )
            if not query_aware_embedding_environment.enabled:
                raise ValueError(
                    "Query-aware ranking local embedding environment is not configured"
                )
            query_aware_cross_environment = (
                load_local_cross_encoder_environment(cross_encoder_environment)
            )
            if not query_aware_cross_environment.enabled:
                raise ValueError(
                    "Query-aware ranking local Cross-Encoder environment is not configured"
                )
            query_aware_ranking = OnlineQueryAwareRankingRuntime.from_sources(
                QueryAwareRankingSources(
                    source_root=sources.project_root,
                    project_root=sources.project_root,
                    config_root=sources.project_root,
                ),
                config=query_aware_config,
                policy=query_aware_policy,
                embedding_environment=query_aware_embedding_environment,
                cross_encoder_environment=query_aware_cross_environment,
            )
        if session_memory_config_path is not None:
            memory_config = load_session_memory_config(session_memory_config_path)
            session_memory = build_session_memory_runtime(
                project_root=sources.project_root,
                config=memory_config,
                environment=(
                    session_memory_environment or controlled_llm_environment
                ),
            )
        registry = build_step23_tool_registry(
            user_profiles=user_profiles,
            business_profiles=business_profiles,
            retriever=retriever,
            history_reader=data_view,
            hybrid_ranking=ranking_service,
            embedding_match=embedding_match,
            cross_encoder_reranker=cross_reranker,
            review_search=review_search,
            evidence_aggregator=evidence_aggregator,
            semantic_ranking=semantic_ranking,
            query_aware_ranking=query_aware_ranking,
            query_retriever=query_retrieval,
            dual_channel_fusion=dual_channel_fusion,
            embedding_alpha=(
                semantic_config.fusion_alpha if semantic_config is not None else 0.0
            ),
            cross_encoder_beta=(
                cross_policy.fusion_beta if cross_policy is not None else 0.0
            ),
            runtime_config=tool_config,
        )
        fallback_handler = (
            RankingCascadeFallbackHandler(
                ranking_service,
                display_limit=cross_policy.display_limit,
                embedding_alpha=semantic_config.fusion_alpha,
                cross_encoder_beta=cross_policy.fusion_beta,
            )
            if cross_reranker is not None
            and cross_policy is not None
            and semantic_config is not None
            else HybridV2FallbackHandler(ranking_service)
        )
        base_budget = harness_config.budget
        if embedding_match is not None and semantic_config is not None:
            base_budget = base_budget.model_copy(
                update={
                    "max_total_tokens": max(
                        base_budget.max_total_tokens,
                        semantic_config.max_total_tokens_per_turn,
                    )
                }
            )
        if cross_reranker is not None and cross_config is not None:
            base_budget = base_budget.model_copy(
                update={
                    "max_tool_calls": max(base_budget.max_tool_calls, 6),
                    "max_semantic_calls": max(base_budget.max_semantic_calls, 2),
                    "max_total_tokens": max(
                        base_budget.max_total_tokens,
                        cross_config.max_total_tokens_per_turn,
                    ),
                }
            )
        if controlled_llm is not None:
            base_budget = base_budget.model_copy(
                update={
                    "max_semantic_calls": max(base_budget.max_semantic_calls, 4),
                }
            )
        if session_memory is not None:
            base_budget = base_budget.model_copy(
                update={
                    "max_semantic_calls": max(base_budget.max_semantic_calls, 4),
                }
            )
        if semantic_ranking is not None:
            base_budget = base_budget.model_copy(
                update={
                    "max_tool_calls": max(base_budget.max_tool_calls, 7),
                    "max_semantic_calls": max(base_budget.max_semantic_calls, 3),
                }
            )
        if query_aware_ranking is not None:
            base_budget = base_budget.model_copy(
                update={
                    "max_tool_calls": max(base_budget.max_tool_calls, 3),
                    "max_semantic_calls": max(base_budget.max_semantic_calls, 1),
                    "max_steps": max(base_budget.max_steps, 5),
                }
            )
        display_limit = (
            query_aware_policy.display_limit
            if query_aware_policy is not None
            else cross_policy.display_limit
            if cross_policy is not None
            else rule_config.display_limit
        )
        if constrained_router_config_path is not None:
            from yelp_agent.constrained_llm_router import (
                build_constrained_llm_router_runtime,
                load_constrained_llm_router_config,
            )

            constrained_config = load_constrained_llm_router_config(
                constrained_router_config_path
            )
            fallback_router = RuleRouter(
                display_limit=display_limit,
                semantic_enabled=embedding_match is not None,
                semantic_candidate_limit=(
                    semantic_config.candidate_limit
                    if semantic_config is not None
                    else 30
                ),
                fusion_alpha=(
                    semantic_config.fusion_alpha
                    if embedding_match is not None and semantic_config is not None
                    else 0.0
                ),
                cross_encoder_enabled=cross_reranker is not None,
                cross_encoder_candidate_limit=(
                    cross_policy.candidate_limit if cross_policy is not None else 20
                ),
                cross_encoder_beta=(
                    cross_policy.fusion_beta if cross_policy is not None else 0.0
                ),
                review_rag_enabled=review_search is not None,
                evidence_aggregation_enabled=evidence_aggregator is not None,
                semantic_ranking_enabled=semantic_ranking is not None,
                query_aware_enabled=query_aware_ranking is not None,
            )
            constrained_router = build_constrained_llm_router_runtime(
                project_root=sources.project_root,
                config=constrained_config,
                fallback_router=fallback_router,
                environment=(
                    constrained_router_environment or controlled_llm_environment
                ),
            )
            base_budget = base_budget.model_copy(
                update={
                    "max_semantic_calls": max(base_budget.max_semantic_calls, 12),
                    "max_steps": max(base_budget.max_steps, 12),
                }
            )
        harness = build_rule_agent(
            registry=registry,
            fallback_handler=fallback_handler,
            budget=base_budget,
            display_limit=display_limit,
            semantic_enabled=embedding_match is not None,
            semantic_candidate_limit=(
                semantic_config.candidate_limit
                if semantic_config is not None
                else 30
            ),
            fusion_alpha=(
                semantic_config.fusion_alpha
                if embedding_match is not None and semantic_config is not None
                else 0.0
            ),
            cross_encoder_enabled=cross_reranker is not None,
            cross_encoder_candidate_limit=(
                cross_policy.candidate_limit if cross_policy is not None else 20
            ),
            cross_encoder_beta=(
                cross_policy.fusion_beta if cross_policy is not None else 0.0
            ),
            review_rag_enabled=review_search is not None,
            evidence_aggregation_enabled=evidence_aggregator is not None,
            semantic_ranking_enabled=semantic_ranking is not None,
            query_aware_enabled=query_aware_ranking is not None,
            semantic_enhancer=(
                None
                if controlled_llm is None
                else controlled_llm.semantic_enhancer
            ),
            session_memory_manager=(
                None if session_memory is None else session_memory.manager
            ),
            answer_composer=(
                None
                if controlled_llm is None
                else controlled_llm.answer_composer
            ),
            answer_evidence_limit=(
                12
                if controlled_config is None
                else controlled_config.answer.maximum_evidence_items
            ),
            action_policy=(
                None
                if constrained_router is None
                else constrained_router.action_policy
            ),
            router=(
                None if constrained_router is None else constrained_router.router
            ),
            agent_version=(
                constrained_config.agent_version
                if constrained_router is not None
                else query_aware_config.agent_version
                if query_aware_ranking is not None
                and query_aware_config is not None
                else memory_config.memory_version
                if session_memory is not None
                else query_retrieval_config.agent_version
                if query_retrieval is not None
                else semantic_ranking_config.agent_version
                if semantic_ranking is not None
                and semantic_ranking_config is not None
                else controlled_config.agent_version
                if controlled_llm is not None and controlled_config is not None
                else evidence_config.agent_version
                if evidence_aggregator is not None and evidence_config is not None
                else review_config.agent_version
                if review_search is not None and review_config is not None
                else cross_config.agent_version
                if cross_reranker is not None and cross_config is not None
                else semantic_config.agent_version
                if embedding_match is not None and semantic_config is not None
                else rule_config.agent_version
            ),
        )
    except Exception:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
        close = getattr(cross_encoder, "close", None)
        if callable(close):
            close()
        close = getattr(review_encoder, "close", None)
        if callable(close):
            close()
        close = getattr(review_store, "close", None)
        if callable(close):
            close()
        close = getattr(controlled_llm, "close", None)
        if callable(close):
            close()
        close = getattr(session_memory, "close", None)
        if callable(close):
            close()
        close = getattr(constrained_router, "close", None)
        if callable(close):
            close()
        close = getattr(query_aware_ranking, "close", None)
        if callable(close):
            close()
        user_profiles.close()
        raise
    return RuleAgentRuntime(
        sources=sources,
        harness=harness,
        user_profiles=user_profiles,
        embedding_encoder=encoder,
        cross_encoder=cross_encoder,
        review_embedding_encoder=review_encoder,
        review_store=review_store,
        controlled_llm=controlled_llm,
        semantic_ranking=semantic_ranking,
        query_retrieval=query_retrieval,
        session_memory=session_memory,
        constrained_router=constrained_router,
        query_aware_ranking=query_aware_ranking,
    )
