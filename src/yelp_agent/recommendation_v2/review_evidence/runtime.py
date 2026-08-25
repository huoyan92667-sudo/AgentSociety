"""建立新版评论证据排序需要的 Qdrant、本地向量模型和长尾描述模型。"""

from __future__ import annotations

import os
from pathlib import Path

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
)
from yelp_agent.semantic_embedding.config import load_local_embedding_environment

from .descriptions import PreferenceDescriptionBuilder
from .full_reviews import FullReviewStore
from .qdrant_store import QdrantReviewSegmentStore
from .ranker import ReviewEvidenceRanker
from .retrieval import ReviewEvidenceRetriever
from .scoring import EvidenceScoringConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_REVIEW_RAG_CONFIG = _PROJECT_ROOT / "configs" / "review_rag.yaml"


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
) -> ReviewEvidenceRanker:
    """建立真实运行时；固定14种不会调用大模型，只有长尾要求才会调用。"""

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
    rag_config = load_review_rag_config(_REVIEW_RAG_CONFIG)
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": 16}),
        _local_embedding_environment(),
    )
    store = QdrantReviewSegmentStore.from_url(
        qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    retriever = ReviewEvidenceRetriever(
        store=store,
        encoder=encoder,
        full_reviews=FullReviewStore(),
        recall_threshold=0.55,
        acceptance_threshold=0.60,
        direction_margin=0.05,
        recall_each_side=15,
        segment_group_size=60,
    )
    return ReviewEvidenceRanker(
        description_builder=PreferenceDescriptionBuilder(generator),
        retriever=retriever,
        scoring_config=EvidenceScoringConfig(
            acceptance_threshold=0.60,
            top_each_side=5,
            half_life_days=730,
        ),
    )
