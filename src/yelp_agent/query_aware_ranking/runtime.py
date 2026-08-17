"""Real local-model assembly for the Step 33 benchmark and Agent integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from yelp_agent.agent_tools import OnlineHybridV2RankingService
from yelp_agent.business_profiles import BusinessKnowledgeStore
from yelp_agent.collaborative import TemporalItemKNNStore
from yelp_agent.config import (
    load_business_profile_config,
    load_config,
    load_item_knn_config,
    load_retrieval_config,
)
from yelp_agent.cross_encoder import (
    CachedCrossEncoderReranker,
    LocalCrossEncoderEnvironment,
    LocalQwenCrossEncoder,
    SqliteCrossEncoderCache,
    load_cross_encoder_config,
)
from yelp_agent.features.category import TemporalCategoryStore
from yelp_agent.features.text import TemporalTextStore
from yelp_agent.learning_to_rank import FrozenLambdaMARTRanker
from yelp_agent.learning_to_rank.features import HybridV1Weights
from yelp_agent.profiles.store import UserProfileStore
from yelp_agent.query import (
    QueryParseInput,
    RecommendationRequest,
    build_rule_based_request_parser,
)
from yelp_agent.query_recommendation_benchmark import VisibleQueryRecommendationCase
from yelp_agent.query_retrieval import (
    DualChannelFusion,
    QueryCandidateRetriever,
    QueryRetrievalTask,
    load_query_retrieval_config,
)
from yelp_agent.ranking.assembly import HybridSourcePaths, build_frozen_hybrid_runtime
from yelp_agent.retrieval import MultiRouteRetriever, RetrievalTaskContext
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
    SemanticEmbeddingMatcher,
    SqliteEmbeddingCache,
    load_semantic_embedding_config,
)

from .config import QueryAwareRankingConfig, QueryAwareRankingPolicy
from .engine import QueryAwareRecommendationEngine
from .schema import PreparedQueryAwareCase, QueryAwareRankingResult


@dataclass(frozen=True, slots=True)
class QueryAwareRankingSources:
    source_root: Path
    project_root: Path
    config_root: Path

    @property
    def data_root(self) -> Path:
        return self.source_root / "data"

    def common_required_files(self) -> tuple[Path, ...]:
        data = self.data_root
        return (
            data / "processed" / "businesses.parquet",
            data / "processed" / "reviews.parquet",
            data / "processed" / "interactions.parquet",
            data / "task_dataset" / "tasks" / "temporal_histories.parquet",
            data / "features" / "tfidf_vectorizer.joblib",
            data / "features" / "tfidf_manifest.json",
            self.source_root / "runs" / "hybrid" / "hybrid_weights.json",
            data / "features" / "item_knn" / "positive_events.parquet",
            data / "features" / "item_knn" / "negative_events.parquet",
            data / "features" / "item_knn" / "neutral_events.parquet",
            data / "features" / "user_profiles" / "v1" / "manifest.json",
            data / "features" / "business_profiles" / "v1" / "manifest.json",
            self.source_root / "runs" / "hybrid_v2_b" / "frozen" / "model.joblib",
            self.config_root / "configs" / "query_retrieval.yaml",
            self.config_root / "configs" / "embedding.yaml",
            self.config_root / "configs" / "cross_encoder.yaml",
            self.config_root / "configs" / "query_aware_ranking.yaml",
        )

    def validate(self) -> None:
        missing = [path for path in self.common_required_files() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Step 33 inputs are incomplete:\n"
                + "\n".join(f"- {path}" for path in missing)
            )


class _UnavailableDependency:
    def __getattr__(self, name: str) -> object:
        raise RuntimeError(f"dependency is unavailable in this runtime stage: {name}")


class QueryAwarePreparationRuntime:
    """Own all heavy preparation stores once and never receive target labels."""

    def __init__(
        self,
        *,
        data_view: object,
        history_retriever: MultiRouteRetriever,
        query_retriever: QueryCandidateRetriever,
        dual_fusion: DualChannelFusion,
        engine: QueryAwareRecommendationEngine,
        encoder: LocalQwenEmbeddingEncoder,
        user_profiles: UserProfileStore,
    ) -> None:
        self._data_view = data_view
        self._history = history_retriever
        self._query = query_retriever
        self._dual = dual_fusion
        self._engine = engine
        self._encoder = encoder
        self._user_profiles = user_profiles
        self._parser = build_rule_based_request_parser()
        self._closed = False

    @classmethod
    def from_sources(
        cls,
        sources: QueryAwareRankingSources,
        *,
        config: QueryAwareRankingConfig,
        provisional_policy: QueryAwareRankingPolicy,
        embedding_environment: LocalEmbeddingEnvironment,
    ) -> Self:
        sources.validate()
        data = sources.data_root
        source_configs = sources.source_root / "configs"
        app_config = load_config(source_configs)
        frozen = build_frozen_hybrid_runtime(
            app_config,
            HybridSourcePaths(
                businesses=data / "processed" / "businesses.parquet",
                reviews=data / "processed" / "reviews.parquet",
                interactions=data / "processed" / "interactions.parquet",
                histories=data
                / "task_dataset"
                / "tasks"
                / "temporal_histories.parquet",
                tfidf_artifact=data / "features" / "tfidf_vectorizer.joblib",
                tfidf_manifest=data / "features" / "tfidf_manifest.json",
                config_dir=source_configs,
            ),
            sources.source_root / "runs" / "hybrid" / "hybrid_weights.json",
        )
        data_view = frozen.assembly.data_view
        item_knn = TemporalItemKNNStore.from_event_artifacts(
            data / "features" / "item_knn" / "positive_events.parquet",
            data / "features" / "item_knn" / "negative_events.parquet",
            data / "features" / "item_knn" / "neutral_events.parquet",
            load_item_knn_config(source_configs),
        )
        history = MultiRouteRetriever(
            data_view,
            category_store=TemporalCategoryStore(
                data_view,
                broad_categories=set(app_config.data.broad_categories),
            ),
            text_store=TemporalTextStore(
                data_view,
                data / "features" / "tfidf_vectorizer.joblib",
                data / "features" / "tfidf_manifest.json",
            ),
            quality_store=frozen.assembly.quality_store,
            location_store=frozen.assembly.location_store,
            config=load_retrieval_config(source_configs),
            item_knn_store=item_knn,
        )
        business_config = load_business_profile_config(source_configs)
        business_profiles = BusinessKnowledgeStore.from_artifacts(
            data / "features" / "business_profiles" / "v1",
            config=business_config,
        )
        user_profiles = UserProfileStore(
            data / "features" / "user_profiles" / "v1"
        )
        ranking_service = OnlineHybridV2RankingService(
            user_profiles=user_profiles,
            business_profiles=business_profiles,
            ranker=FrozenLambdaMARTRanker.from_artifacts(
                sources.source_root / "runs" / "hybrid_v2_b" / "frozen"
            ),
            weights=HybridV1Weights.model_validate(frozen.weights.as_dict),
            broad_categories=set(app_config.data.broad_categories),
            business_profile_config=business_config,
        )
        embedding_config = load_semantic_embedding_config(
            sources.config_root / "configs" / "embedding.yaml"
        )
        if embedding_config.provider != "local":
            raise ValueError("Step 33 formal experiment requires local embedding")
        encoder = LocalQwenEmbeddingEncoder.from_environment(
            embedding_config,
            embedding_environment,
        )
        embedding = SemanticEmbeddingMatcher(
            businesses=data_view,
            gateway=CachedEmbeddingGateway(
                encoder=encoder,
                cache=SqliteEmbeddingCache(
                    sources.project_root / embedding_config.cache_relative_path
                ),
                config=embedding_config,
            ),
            config=embedding_config,
        )
        query_config = load_query_retrieval_config(
            sources.config_root / "configs" / "query_retrieval.yaml"
        )
        query = QueryCandidateRetriever(
            catalog=data_view,
            profiles=business_profiles,
            embedding_matcher=embedding,
            config=query_config,
        )
        engine = QueryAwareRecommendationEngine(
            ranking_service=ranking_service,
            embedding_matcher=embedding,
            cross_encoder=_UnavailableDependency(),  # type: ignore[arg-type]
            business_profiles=business_profiles,
            policy=provisional_policy,
        )
        return cls(
            data_view=data_view,
            history_retriever=history,
            query_retriever=query,
            dual_fusion=DualChannelFusion(query_config),
            engine=engine,
            encoder=encoder,
            user_profiles=user_profiles,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._encoder.close()
            self._user_profiles.close()
            self._closed = True

    def prepare_case(self, case: VisibleQueryRecommendationCase) -> PreparedQueryAwareCase:
        request = self._parser.parse(
            QueryParseInput(
                user_id=case.user_id,
                session_id=case.session_id,
                cutoff_time=case.cutoff_time,
                query_text=case.query_text,
                user_latitude=case.user_latitude,
                user_longitude=case.user_longitude,
            )
        )
        return self.prepare_request(
            case_id=case.case_id,
            split=case.split,
            request=request,
            usage_scope=f"step33:{case.case_id}",
        )

    def prepare_request(
        self,
        *,
        case_id: str,
        split: Literal["development", "validation"],
        request: RecommendationRequest,
        usage_scope: str,
        rejected_business_ids: set[str] | frozenset[str] = frozenset(),
    ) -> PreparedQueryAwareCase:
        """Prepare one live Agent request through the frozen Step-33 pipeline."""

        parsed_request = RecommendationRequest.model_validate(request)
        history_rows = self._data_view.user_history(
            parsed_request.user_id,
            parsed_request.cutoff_time,
        )
        history = self._history.retrieve(
            RetrievalTaskContext(
                task_id=case_id,
                split=split,
                user_id=parsed_request.user_id,
                cutoff_time=parsed_request.cutoff_time,
                history_count=len(history_rows),
            )
        )
        query = self._query.retrieve(
            QueryRetrievalTask(
                request=parsed_request,
                usage_scope=usage_scope,
            )
        )
        dual = self._dual.fuse(history, query)
        return self._engine.prepare(
            case_id=case_id,
            split=split,
            request=parsed_request,
            history=history,
            query=query,
            rrf_fusion_ranking=[item.business_id for item in dual.candidates],
            usage_scope=usage_scope,
            rejected_business_ids=rejected_business_ids,
        )


class QueryAwareFinalizationRuntime:
    """Load only the local Cross-Encoder and cutoff-safe profile store."""

    def __init__(
        self,
        *,
        engine: QueryAwareRecommendationEngine,
        cross_encoder: LocalQwenCrossEncoder,
    ) -> None:
        self._engine = engine
        self._cross_encoder = cross_encoder
        self._closed = False

    @classmethod
    def from_sources(
        cls,
        sources: QueryAwareRankingSources,
        *,
        policy: QueryAwareRankingPolicy,
        cross_encoder_environment: LocalCrossEncoderEnvironment,
    ) -> Self:
        sources.validate()
        source_configs = sources.source_root / "configs"
        business_profiles = BusinessKnowledgeStore.from_artifacts(
            sources.data_root / "features" / "business_profiles" / "v1",
            config=load_business_profile_config(source_configs),
        )
        cross_config = load_cross_encoder_config(
            sources.config_root / "configs" / "cross_encoder.yaml"
        )
        cross_model = LocalQwenCrossEncoder.from_environment(
            cross_config,
            cross_encoder_environment,
        )
        # Cross-Encoder documents are static, so the lightweight temporal data view
        # is the only additional reader required during finalization.
        app_config = load_config(source_configs)
        frozen = build_frozen_hybrid_runtime(
            app_config,
            HybridSourcePaths(
                businesses=sources.data_root / "processed" / "businesses.parquet",
                reviews=sources.data_root / "processed" / "reviews.parquet",
                interactions=sources.data_root / "processed" / "interactions.parquet",
                histories=sources.data_root
                / "task_dataset"
                / "tasks"
                / "temporal_histories.parquet",
                tfidf_artifact=sources.data_root
                / "features"
                / "tfidf_vectorizer.joblib",
                tfidf_manifest=sources.data_root / "features" / "tfidf_manifest.json",
                config_dir=source_configs,
            ),
            sources.source_root / "runs" / "hybrid" / "hybrid_weights.json",
        )
        cross = CachedCrossEncoderReranker(
            businesses=frozen.assembly.data_view,
            scorer=cross_model,
            cache=SqliteCrossEncoderCache(
                sources.project_root / cross_config.cache_relative_path
            ),
            config=cross_config,
        )
        unavailable = _UnavailableDependency()
        engine = QueryAwareRecommendationEngine(
            ranking_service=unavailable,  # type: ignore[arg-type]
            embedding_matcher=unavailable,  # type: ignore[arg-type]
            cross_encoder=cross,
            business_profiles=business_profiles,
            policy=policy,
        )
        return cls(engine=engine, cross_encoder=cross_model)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._cross_encoder.close()
            self._closed = True

    def finalize_case(
        self,
        prepared: PreparedQueryAwareCase,
    ) -> QueryAwareRankingResult:
        return self._engine.finalize(
            prepared,
            usage_scope=f"step33:{prepared.case_id}",
        )


class OnlineQueryAwareRankingRuntime:
    """One Agent-facing seam owning the complete frozen Step-33 runtime."""

    def __init__(
        self,
        *,
        preparation: QueryAwarePreparationRuntime,
        finalization: QueryAwareFinalizationRuntime,
    ) -> None:
        self._preparation = preparation
        self._finalization = finalization
        self._closed = False

    @classmethod
    def from_sources(
        cls,
        sources: QueryAwareRankingSources,
        *,
        config: QueryAwareRankingConfig,
        policy: QueryAwareRankingPolicy,
        embedding_environment: LocalEmbeddingEnvironment,
        cross_encoder_environment: LocalCrossEncoderEnvironment,
    ) -> Self:
        preparation = QueryAwarePreparationRuntime.from_sources(
            sources,
            config=config,
            provisional_policy=policy,
            embedding_environment=embedding_environment,
        )
        try:
            finalization = QueryAwareFinalizationRuntime.from_sources(
                sources,
                policy=policy,
                cross_encoder_environment=cross_encoder_environment,
            )
        except Exception:
            preparation.close()
            raise
        return cls(preparation=preparation, finalization=finalization)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._finalization.close()
            self._preparation.close()
            self._closed = True

    def rank(
        self,
        *,
        request: RecommendationRequest,
        case_id: str,
        split: Literal["development", "validation"],
        usage_scope: str,
        rejected_business_ids: set[str] | frozenset[str] = frozenset(),
    ) -> QueryAwareRankingResult:
        prepared = self._preparation.prepare_request(
            case_id=case_id,
            split=split,
            request=request,
            usage_scope=usage_scope,
            rejected_business_ids=rejected_business_ids,
        )
        return self._finalization.finalize_case(prepared)
