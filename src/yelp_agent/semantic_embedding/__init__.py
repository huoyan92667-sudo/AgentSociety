"""Step 25 query-to-business semantic matching."""

from .cache import CachedEmbedding, SqliteEmbeddingCache
from .config import (
    DashScopeEmbeddingEnvironment,
    SemanticEmbeddingConfig,
    load_dashscope_embedding_environment,
    load_semantic_embedding_config,
)
from .documents import build_business_document, build_query_document
from .diagnostics import (
    SemanticMethodCandidate,
    SemanticMethodComparison,
    compare_tfidf_and_embedding,
)
from .encoder import (
    DashScopeEmbeddingEncoder,
    EmbeddingEncoder,
    EmbeddingProviderError,
    EncodedBatch,
)
from .fusion import fuse_hybrid_and_semantic
from .matcher import CachedEmbeddingGateway, SemanticEmbeddingMatcher
from .schema import (
    EmbeddingUsage,
    SemanticBusinessMatch,
    SemanticDocument,
    SemanticMatchResult,
)

__all__ = [
    "CachedEmbedding",
    "CachedEmbeddingGateway",
    "DashScopeEmbeddingEncoder",
    "DashScopeEmbeddingEnvironment",
    "EmbeddingEncoder",
    "EmbeddingProviderError",
    "EmbeddingUsage",
    "EncodedBatch",
    "SemanticBusinessMatch",
    "SemanticDocument",
    "SemanticEmbeddingConfig",
    "SemanticEmbeddingMatcher",
    "SemanticMatchResult",
    "SemanticMethodCandidate",
    "SemanticMethodComparison",
    "SqliteEmbeddingCache",
    "build_business_document",
    "build_query_document",
    "compare_tfidf_and_embedding",
    "fuse_hybrid_and_semantic",
    "load_dashscope_embedding_environment",
    "load_semantic_embedding_config",
]
