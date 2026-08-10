"""Production assembly of the frozen Step 24 Rule Agent."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path

from yelp_agent.agent_harness import AgentHarness, load_agent_harness_config
from yelp_agent.agent_tools import (
    HybridV2FallbackHandler,
    OnlineHybridV2RankingService,
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
)
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.learning_to_rank import FrozenLambdaMARTRanker
from yelp_agent.learning_to_rank.features import HybridV1Weights
from yelp_agent.profiles.store import UserProfileStore
from yelp_agent.ranking.assembly import HybridSourcePaths, build_frozen_hybrid_runtime
from yelp_agent.retrieval import MultiRouteRetriever
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    DashScopeEmbeddingEncoder,
    SemanticEmbeddingMatcher,
    SqliteEmbeddingCache,
    load_dashscope_embedding_environment,
    load_semantic_embedding_config,
)

from .config import load_rule_router_config
from .factory import build_rule_agent


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
    ) -> None:
        self.sources = sources
        self.harness = harness
        self._user_profiles = user_profiles
        self._closed = False

    def __enter__(self) -> RuleAgentRuntime:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._user_profiles.close()
            self._closed = True


def build_real_rule_agent_runtime(
    sources: RuleAgentSourcePaths,
    *,
    embedding_config_path: str | Path | None = None,
    embedding_environment: Mapping[str, str] | None = None,
) -> RuleAgentRuntime:
    """Load all frozen artifacts once and assemble the production Rule Agent."""

    sources.validate()
    app_config = load_config(sources.config_dir)
    retrieval_config = load_retrieval_config(sources.config_dir)
    item_knn_config = load_item_knn_config(sources.config_dir)
    business_config = load_business_profile_config(sources.config_dir)
    rule_config = load_rule_router_config(sources.rule_router_config)
    harness_config = load_agent_harness_config(sources.agent_harness_config)
    tool_config = load_agent_tool_runtime_config(sources.agent_tools_config)

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
            environment = load_dashscope_embedding_environment(
                embedding_environment
            )
            if environment.enabled:
                encoder = DashScopeEmbeddingEncoder.from_environment(
                    semantic_config,
                    environment,
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
        registry = build_step23_tool_registry(
            user_profiles=user_profiles,
            business_profiles=business_profiles,
            retriever=retriever,
            history_reader=data_view,
            hybrid_ranking=ranking_service,
            embedding_match=embedding_match,
            runtime_config=tool_config,
        )
        harness = build_rule_agent(
            registry=registry,
            fallback_handler=HybridV2FallbackHandler(ranking_service),
            budget=(
                harness_config.budget.model_copy(
                    update={
                        "max_total_tokens": max(
                            harness_config.budget.max_total_tokens,
                            semantic_config.max_total_tokens_per_turn,
                        )
                    }
                )
                if embedding_match is not None and semantic_config is not None
                else harness_config.budget
            ),
            display_limit=rule_config.display_limit,
            agent_version=(
                semantic_config.agent_version
                if embedding_match is not None and semantic_config is not None
                else rule_config.agent_version
            ),
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
        )
    except Exception:
        user_profiles.close()
        raise
    return RuleAgentRuntime(
        sources=sources,
        harness=harness,
        user_profiles=user_profiles,
    )
