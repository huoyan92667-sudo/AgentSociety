"""Target-blind runtime for the three approved retrieval comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

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
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query_retrieval import (
    DualChannelFusion,
    QueryCandidateRetriever,
    QueryRetrievalTask,
    load_query_retrieval_config,
)
from yelp_agent.ranking import HybridSourcePaths, build_frozen_hybrid_runtime
from yelp_agent.retrieval import MultiRouteRetriever, RetrievalTaskContext
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
    SemanticEmbeddingMatcher,
    SqliteEmbeddingCache,
    load_semantic_embedding_config,
)

from .evaluation import BenchmarkRetrievalRun
from .schema import VisibleQueryRecommendationCase


@dataclass(frozen=True, slots=True)
class QueryRecommendationRetrievalSources:
    project_root: Path
    config_root: Path

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    def required_files(self) -> tuple[Path, ...]:
        root = self.project_root
        return (
            self.data_root / "processed" / "businesses.parquet",
            self.data_root / "processed" / "reviews.parquet",
            self.data_root / "processed" / "interactions.parquet",
            self.data_root / "task_dataset" / "tasks" / "temporal_histories.parquet",
            self.data_root / "features" / "tfidf_vectorizer.joblib",
            self.data_root / "features" / "tfidf_manifest.json",
            root / "runs" / "hybrid" / "hybrid_weights.json",
            self.data_root / "features" / "item_knn" / "positive_events.parquet",
            self.data_root / "features" / "item_knn" / "negative_events.parquet",
            self.data_root / "features" / "item_knn" / "neutral_events.parquet",
            self.data_root / "features" / "business_profiles" / "v1" / "manifest.json",
            self.config_root / "configs" / "query_retrieval.yaml",
            self.config_root / "configs" / "embedding.yaml",
        )

    def validate(self) -> None:
        missing = [path for path in self.required_files() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Query recommendation retrieval inputs are incomplete:\n"
                + "\n".join(f"- {path}" for path in missing)
            )


class QueryRecommendationRetrievalRuntime:
    """Own heavy stores once and expose one label-free case operation."""

    def __init__(
        self,
        *,
        history_retriever: MultiRouteRetriever,
        query_retriever: QueryCandidateRetriever,
        fusion: DualChannelFusion,
        data_view: object,
        encoder: LocalQwenEmbeddingEncoder,
    ) -> None:
        self._history = history_retriever
        self._query = query_retriever
        self._fusion = fusion
        self._data_view = data_view
        self._encoder = encoder
        self._parser = build_rule_based_request_parser()
        self._closed = False

    @classmethod
    def from_sources(
        cls,
        sources: QueryRecommendationRetrievalSources,
        *,
        embedding_environment: LocalEmbeddingEnvironment,
    ) -> QueryRecommendationRetrievalRuntime:
        sources.validate()
        project = sources.project_root
        data = sources.data_root
        source_configs = project / "configs"
        app_config = load_config(source_configs)
        frozen = build_frozen_hybrid_runtime(
            app_config,
            HybridSourcePaths(
                businesses=data / "processed" / "businesses.parquet",
                reviews=data / "processed" / "reviews.parquet",
                interactions=data / "processed" / "interactions.parquet",
                histories=(
                    data / "task_dataset" / "tasks" / "temporal_histories.parquet"
                ),
                tfidf_artifact=data / "features" / "tfidf_vectorizer.joblib",
                tfidf_manifest=data / "features" / "tfidf_manifest.json",
                config_dir=source_configs,
            ),
            project / "runs" / "hybrid" / "hybrid_weights.json",
        )
        data_view = frozen.assembly.data_view
        item_knn = TemporalItemKNNStore.from_event_artifacts(
            data / "features" / "item_knn" / "positive_events.parquet",
            data / "features" / "item_knn" / "negative_events.parquet",
            data / "features" / "item_knn" / "neutral_events.parquet",
            load_item_knn_config(source_configs),
        )
        history_retriever = MultiRouteRetriever(
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
        profiles = BusinessKnowledgeStore.from_artifacts(
            data / "features" / "business_profiles" / "v1",
            config=load_business_profile_config(source_configs),
        )
        embedding_config = load_semantic_embedding_config(
            sources.config_root / "configs" / "embedding.yaml"
        )
        if embedding_config.provider != "local":
            raise ValueError("formal benchmark requires the local embedding provider")
        encoder = LocalQwenEmbeddingEncoder.from_environment(
            embedding_config,
            embedding_environment,
        )
        matcher = SemanticEmbeddingMatcher(
            businesses=data_view,
            gateway=CachedEmbeddingGateway(
                encoder=encoder,
                cache=SqliteEmbeddingCache(
                    project / embedding_config.cache_relative_path
                ),
                config=embedding_config,
            ),
            config=embedding_config,
        )
        query_config = load_query_retrieval_config(
            sources.config_root / "configs" / "query_retrieval.yaml"
        )
        return cls(
            history_retriever=history_retriever,
            query_retriever=QueryCandidateRetriever(
                catalog=data_view,
                profiles=profiles,
                embedding_matcher=matcher,
                config=query_config,
            ),
            fusion=DualChannelFusion(query_config),
            data_view=data_view,
            encoder=encoder,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._encoder.close()
            self._closed = True

    def run_case(
        self,
        case: VisibleQueryRecommendationCase,
    ) -> tuple[BenchmarkRetrievalRun, BenchmarkRetrievalRun, BenchmarkRetrievalRun]:
        """Run visible inputs only; this method has no target-label parameter."""

        history_count = len(
            self._data_view.user_history(case.user_id, case.cutoff_time)
        )
        history = self._history.retrieve(
            RetrievalTaskContext(
                task_id=case.case_id,
                split=case.split,
                user_id=case.user_id,
                cutoff_time=case.cutoff_time,
                history_count=history_count,
            )
        )
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
        query = self._query.retrieve(
            QueryRetrievalTask(
                request=request,
                usage_scope=f"query-recommendation:{case.case_id}",
            )
        )
        dual = self._fusion.fuse(history, query)
        return (
            BenchmarkRetrievalRun(
                case_id=case.case_id,
                method="history_only",
                ranking=[item.business_id for item in history.candidates],
                latency_ms=history.latency_ms,
            ),
            BenchmarkRetrievalRun(
                case_id=case.case_id,
                method="query_only",
                ranking=[item.business_id for item in query.candidates],
                latency_ms=query.latency_ms,
                embedding_encoded_tokens=query.usage.embedding_input_tokens,
                embedding_logical_tokens=query.usage.embedding_logical_tokens,
                embedding_provider_calls=query.usage.provider_calls,
            ),
            BenchmarkRetrievalRun(
                case_id=case.case_id,
                method="history_query",
                ranking=[item.business_id for item in dual.candidates],
                latency_ms=dual.latency_ms,
            ),
        )
