"""建立新版评论证据排序需要的 Qdrant、本地向量模型和长尾描述模型。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.recommendation_v2.business_aspect_profiles import (
    BusinessAspectProfileCatalog,
    load_business_aspect_profile_catalog,
)
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
)
from yelp_agent.semantic_embedding.config import load_local_embedding_environment

from .descriptions import PreferenceDescriptionBuilder
from .direct_search import DirectReviewEvidenceSearch
from .full_reviews import FullReviewStore
from .offline_aspects import OfflineAspectEvidenceResolver
from .qdrant_store import QdrantReviewSegmentStore
from .ranker import ReviewEvidenceRanker
from .retrieval import ReviewEvidenceRetriever
from .scoring import EvidenceScoringConfig
from .segment_vectors import ReviewSegmentVectorStore

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class ReviewEvidenceCapabilities:
    """推荐排序和按商家查评论共享同一套模型、向量和Qdrant资源。"""

    ranker: ReviewEvidenceRanker
    direct_search: DirectReviewEvidenceSearch


class _LazyReviewEvidenceRetriever:
    """只有出现固定14项以外的自由要求时，才启动向量模型和Qdrant。"""

    recall_threshold = 0.55
    acceptance_threshold = 0.60
    direction_margin = 0.05

    def __init__(self, factory: Callable[[], ReviewEvidenceRetriever]) -> None:
        self._factory = factory
        self._retriever: ReviewEvidenceRetriever | None = None

    def retrieve_many(self, *args: Any, **kwargs: Any):
        if self._retriever is None:
            self._retriever = self._factory()
        return self._retriever.retrieve_many(*args, **kwargs)

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()


def _local_embedding_environment() -> LocalEmbeddingEnvironment:
    """读取本地向量模型；开发机无环境变量时沿用当前已下载模型。"""

    configured = load_local_embedding_environment()
    if configured.enabled:
        return configured
    model_path = Path(r"D:\models\Qwen3-Embedding-0.6B")
    python_executable = Path(r"D:\anaconda3\python.exe")
    if not model_path.is_dir() or not python_executable.is_file():
        raise RuntimeError(
            "local review embedding model is not configured; set "
            "LOCAL_EMBEDDING_MODEL_PATH and LOCAL_EMBEDDING_PYTHON"
        )
    return LocalEmbeddingEnvironment(
        model_path=model_path,
        python_executable=python_executable,
        device="cuda",
    )


def build_review_evidence_ranker(
    *,
    qdrant_url: str | None = None,
    profile_catalog: BusinessAspectProfileCatalog | None = None,
    project_root: str | Path | None = None,
) -> ReviewEvidenceRanker:
    """建立真实运行时；固定14种不会调用大模型，只有长尾要求才会调用。"""

    return build_review_evidence_capabilities(
        qdrant_url=qdrant_url,
        profile_catalog=profile_catalog,
        project_root=project_root,
    ).ranker


def build_review_evidence_capabilities(
    *,
    qdrant_url: str | None = None,
    profile_catalog: BusinessAspectProfileCatalog | None = None,
    project_root: str | Path | None = None,
) -> ReviewEvidenceCapabilities:
    """建立共享评论能力，避免推荐和单店查询各加载一份本地向量模型。"""

    generator = OpenAICompatibleLLM.from_environment(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=120,
            max_retries=2,
            max_tokens=2500,
            response_format_json=True,
            thinking="disabled",
        )
    )
    retriever = _LazyReviewEvidenceRetriever(
        lambda: _build_dynamic_review_retriever(
            qdrant_url=qdrant_url,
            project_root=project_root,
        )
    )
    description_builder = PreferenceDescriptionBuilder(generator)
    scoring_config = EvidenceScoringConfig(
        acceptance_threshold=0.60,
        top_each_side=5,
        half_life_days=730,
    )
    ranker = ReviewEvidenceRanker(
        description_builder=description_builder,
        retriever=retriever,
        offline_aspects=OfflineAspectEvidenceResolver(
            profile_catalog or load_business_aspect_profile_catalog()
        ),
        scoring_config=scoring_config,
    )
    return ReviewEvidenceCapabilities(
        ranker=ranker,
        direct_search=DirectReviewEvidenceSearch(
            description_builder=description_builder,
            retriever=retriever,  # type: ignore[arg-type]
            scoring_config=scoring_config,
        ),
    )


def _build_dynamic_review_retriever(
    *,
    qdrant_url: str | None,
    project_root: str | Path | None,
) -> ReviewEvidenceRetriever:
    """建立昂贵的自由评论检索环境；固定14项不会调用这里。"""

    root = _PROJECT_ROOT if project_root is None else Path(project_root)
    rag_config = load_review_rag_config(root / "configs" / "review_rag.yaml")
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": 16}),
        _local_embedding_environment(),
    )
    store = QdrantReviewSegmentStore.from_url(
        qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    index_root = (
        root
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "review_evidence"
        / "v1"
        / "index"
    )
    return ReviewEvidenceRetriever(
        store=store,
        encoder=encoder,
        segment_vectors=ReviewSegmentVectorStore(index_root / "segment_embeddings.npy"),
        full_reviews=FullReviewStore(root / "data" / "processed" / "reviews.parquet"),
        recall_threshold=0.55,
        acceptance_threshold=0.60,
        direction_margin=0.05,
        recall_each_side=15,
        initial_segment_group_size=15,
        middle_segment_group_size=30,
        final_segment_group_size=60,
        minimum_clear_evidence=5,
        search_concurrency=4,
        enable_bm25=True,
        rrf_k=60,
    )
