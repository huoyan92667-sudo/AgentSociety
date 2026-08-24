"""建立评论向量检索、大模型逐条判断和加权排序运行时。"""

from __future__ import annotations

from pathlib import Path

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
)
from yelp_agent.semantic_embedding.config import load_local_embedding_environment

from .evidence_judge import ReviewEvidenceJudge
from .ranker import WeightedPreferenceRanker
from .review_store import ReviewVectorStore

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_REVIEW_RAG_CONFIG = _PROJECT_ROOT / "configs" / "review_rag.yaml"


def _local_embedding_environment() -> LocalEmbeddingEnvironment:
    """优先读环境变量；当前开发机未配置时使用已经建索引用的同一套本地模型。"""

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


def build_priority_layered_ranker() -> WeightedPreferenceRanker:
    """使用项目 DeepSeek 和已经保存的全部评论向量建立新版软排序器。"""

    generator = OpenAICompatibleLLM.from_environment(
        AgentConfig(
            enabled=True,
            temperature=0.0,
            timeout_seconds=120,
            max_retries=2,
            max_tokens=8000,
            response_format_json=True,
            thinking="disabled",
        )
    )
    rag_config = load_review_rag_config(_REVIEW_RAG_CONFIG)
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": 16}),
        _local_embedding_environment(),
    )
    return WeightedPreferenceRanker(
        review_store=ReviewVectorStore(
            encoder,
            similarity_threshold=0.55,
            full_weight_similarity=0.82,
        ),
        evidence_judge=ReviewEvidenceJudge(generator),
        candidate_limit=10,
        reviews_each_side=5,
    )
