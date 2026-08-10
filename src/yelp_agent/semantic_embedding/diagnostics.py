"""Label-free TF-IDF versus Embedding diagnostics for Step 25."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from yelp_agent.models import StrictModel

from .schema import SemanticMatchResult


class SemanticMethodCandidate(StrictModel):
    business_id: str = Field(min_length=1)
    tfidf_cosine: float = Field(ge=0, le=1)
    tfidf_rank: int = Field(ge=1)
    embedding_cosine: float = Field(ge=-1, le=1)
    embedding_rank: int = Field(ge=1)


class SemanticMethodComparison(StrictModel):
    """A diagnostic comparison, not an accuracy claim without relevance labels."""

    candidates: list[SemanticMethodCandidate] = Field(min_length=1)
    top_5_overlap: float = Field(ge=0, le=1)
    spearman_rank_correlation: float = Field(ge=-1, le=1)
    has_external_relevance_labels: bool = False


def compare_tfidf_and_embedding(
    *,
    query_text: str,
    business_documents: Mapping[str, str],
    embedding_result: SemanticMatchResult,
) -> SemanticMethodComparison:
    """Compare lexical and semantic ordering over exactly the same documents."""

    ids = sorted(business_documents)
    if not ids or set(ids) != {item.business_id for item in embedding_result.matches}:
        raise ValueError("TF-IDF documents and embedding matches must align")
    texts = [business_documents[business_id] for business_id in ids]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([*texts, query_text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    tfidf_order = sorted(
        range(len(ids)),
        key=lambda index: (-float(scores[index]), ids[index]),
    )
    tfidf_rank = {index: rank for rank, index in enumerate(tfidf_order, start=1)}
    embedding_by_id = {item.business_id: item for item in embedding_result.matches}
    rows = [
        SemanticMethodCandidate(
            business_id=business_id,
            tfidf_cosine=float(scores[index]),
            tfidf_rank=tfidf_rank[index],
            embedding_cosine=embedding_by_id[business_id].cosine_similarity,
            embedding_rank=embedding_by_id[business_id].semantic_rank,
        )
        for index, business_id in enumerate(ids)
    ]
    limit = min(5, len(ids))
    lexical_top = {
        row.business_id for row in rows if row.tfidf_rank <= limit
    }
    semantic_top = {
        row.business_id for row in rows if row.embedding_rank <= limit
    }
    lexical_ranks = np.asarray([row.tfidf_rank for row in rows], dtype=float)
    semantic_ranks = np.asarray([row.embedding_rank for row in rows], dtype=float)
    if len(rows) == 1:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(lexical_ranks, semantic_ranks)[0, 1])
    return SemanticMethodComparison(
        candidates=sorted(rows, key=lambda row: row.embedding_rank),
        top_5_overlap=len(lexical_top & semantic_top) / limit,
        spearman_rank_correlation=float(np.clip(correlation, -1, 1)),
    )
