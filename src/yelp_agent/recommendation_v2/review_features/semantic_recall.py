"""利用完整评论向量，为每家餐厅的14种特征补充意思相近候选。"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.recommendation_v2.review_index import (
    REVIEW_INDEX_SEGMENT_SCHEMA,
    ReviewVectorIndexManifest,
)
from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS
from yelp_agent.semantic_embedding.encoder import EmbeddingEncoder

from .definitions import AspectRecallDefinition, build_aspect_recall_definitions
from .schema import (
    CANDIDATE_ASPECT_SCHEMA,
    SemanticRecallBuildResult,
    SemanticRecallManifest,
)

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INDEX_ROOT = _RECOMMENDATION_ROOT / "data" / "review_index" / "v1"
DEFAULT_CONFIG_ROOT = _PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = _RECOMMENDATION_ROOT / "data" / "review_features" / "v1"


class SemanticRecallBuildError(RuntimeError):
    """评论向量与片段错位，或无法形成语义候选时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_anchors(
    encoder: EmbeddingEncoder,
    definitions: tuple[AspectRecallDefinition, ...],
) -> tuple[np.ndarray, list[str], list[list[int]]]:
    """把14种特征两端的自然语言说明转换成查询向量。"""

    texts: list[str] = []
    anchor_ids: list[str] = []
    aspect_anchor_indices: list[list[int]] = []
    for definition in definitions:
        indices: list[int] = []
        for index, text in enumerate(definition.semantic_anchors):
            indices.append(len(texts))
            texts.append(text)
            anchor_ids.append(f"{definition.aspect}:{index}")
        aspect_anchor_indices.append(indices)

    vectors: list[np.ndarray] = []
    for offset in range(0, len(texts), encoder.batch_size):
        encoded = encoder.encode(
            texts[offset : offset + encoder.batch_size],
            input_type="query",
        )
        vectors.extend(encoded.vectors)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if matrix.shape != (len(texts), encoder.dimension) or np.any(norms <= 0):
        raise SemanticRecallBuildError("semantic anchor vectors are invalid")
    return matrix / norms, anchor_ids, aspect_anchor_indices


def _aspect_scores(
    segment_vectors: np.ndarray,
    anchor_vectors: np.ndarray,
    aspect_anchor_indices: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    similarities = segment_vectors @ anchor_vectors.T
    scores = np.empty((len(segment_vectors), len(aspect_anchor_indices)), dtype=np.float32)
    anchors = np.empty((len(segment_vectors), len(aspect_anchor_indices)), dtype=np.int16)
    for aspect_index, indices in enumerate(aspect_anchor_indices):
        values = similarities[:, indices]
        local_anchor = np.argmax(values, axis=1)
        scores[:, aspect_index] = values[np.arange(len(values)), local_anchor]
        anchors[:, aspect_index] = np.asarray(indices, dtype=np.int16)[local_anchor]
    return scores, anchors


def _push_review(
    heaps: dict[tuple[str, str], list[tuple[float, str, int]]],
    *,
    business_id: str,
    review_id: str,
    scores: np.ndarray,
    anchors: np.ndarray,
    top_k: int,
) -> None:
    for aspect_index, aspect in enumerate(ASPECT_FIELDS):
        heap = heaps.setdefault((business_id, aspect), [])
        value = (float(scores[aspect_index]), review_id, int(anchors[aspect_index]))
        if len(heap) < top_k:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heapreplace(heap, value)


def _semantic_rows(
    segments_path: Path,
    embeddings_path: Path,
    *,
    anchor_vectors: np.ndarray,
    anchor_ids: list[str],
    aspect_anchor_indices: list[list[int]],
    top_k: int,
) -> list[dict[str, object]]:
    """每条评论先合并自身片段，再为每家商户每种特征保留最相近评论。"""

    vectors = np.load(embeddings_path, mmap_mode="r")
    segment_count = pq.ParquetFile(segments_path).metadata.num_rows
    if vectors.shape[0] != segment_count:
        raise SemanticRecallBuildError("review segments and vectors have different rows")

    heaps: dict[tuple[str, str], list[tuple[float, str, int]]] = {}
    closed_reviews: set[str] = set()
    current_review: str | None = None
    current_business: str | None = None
    current_scores: np.ndarray | None = None
    current_anchors: np.ndarray | None = None

    for batch in pq.ParquetFile(segments_path).iter_batches(
        batch_size=4096,
        columns=["row_index", "review_id", "business_id"],
    ):
        row_start = int(batch.column(0)[0].as_py())
        row_end = row_start + batch.num_rows
        values = np.asarray(vectors[row_start:row_end], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, np.finfo(np.float32).eps)
        batch_scores, batch_anchors = _aspect_scores(
            values,
            anchor_vectors,
            aspect_anchor_indices,
        )
        review_ids = [str(value) for value in batch.column(1).to_pylist()]
        business_ids = [str(value) for value in batch.column(2).to_pylist()]
        for index, (review_id, business_id) in enumerate(
            zip(review_ids, business_ids, strict=True)
        ):
            if current_review is None:
                if review_id in closed_reviews:
                    raise SemanticRecallBuildError("review segments are not contiguous")
                current_review = review_id
                current_business = business_id
                current_scores = batch_scores[index].copy()
                current_anchors = batch_anchors[index].copy()
                continue
            if review_id == current_review:
                assert current_scores is not None and current_anchors is not None
                better = batch_scores[index] > current_scores
                current_scores[better] = batch_scores[index][better]
                current_anchors[better] = batch_anchors[index][better]
                continue
            assert current_business is not None
            assert current_scores is not None and current_anchors is not None
            _push_review(
                heaps,
                business_id=current_business,
                review_id=current_review,
                scores=current_scores,
                anchors=current_anchors,
                top_k=top_k,
            )
            closed_reviews.add(current_review)
            if review_id in closed_reviews:
                raise SemanticRecallBuildError("review segments are not contiguous")
            current_review = review_id
            current_business = business_id
            current_scores = batch_scores[index].copy()
            current_anchors = batch_anchors[index].copy()

    if current_review is not None:
        assert current_business is not None
        assert current_scores is not None and current_anchors is not None
        _push_review(
            heaps,
            business_id=current_business,
            review_id=current_review,
            scores=current_scores,
            anchors=current_anchors,
            top_k=top_k,
        )
    del vectors

    rows: list[dict[str, object]] = []
    for (business_id, aspect), heap in sorted(heaps.items()):
        for score, review_id, anchor_index in sorted(heap, reverse=True):
            rows.append(
                {
                    "review_id": review_id,
                    "business_id": business_id,
                    "aspect": aspect,
                    "keyword_hit": False,
                    "semantic_hit": True,
                    "matched_terms": [],
                    "semantic_score": score,
                    "matched_anchor_ids": [anchor_ids[anchor_index]],
                }
            )
    return rows


def _existing_manifest(
    output_root: Path,
    *,
    source_hashes: dict[str, str],
    definitions_hash: str,
    model: str,
    dimension: int,
    top_k: int,
) -> SemanticRecallManifest | None:
    manifest_path = output_root / "semantic_manifest.json"
    candidate_path = output_root / "semantic_candidate_aspects.parquet"
    if not manifest_path.is_file() or not candidate_path.is_file():
        return None
    try:
        manifest = SemanticRecallManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if (
        manifest.source_sha256 != source_hashes
        or manifest.definitions_sha256 != definitions_hash
        or manifest.model != model
        or manifest.dimension != dimension
        or manifest.top_reviews_per_business_aspect != top_k
        or manifest.output_sha256 != {"semantic_candidates": _sha256_file(candidate_path)}
    ):
        return None
    return manifest


def build_semantic_review_candidates(
    encoder: EmbeddingEncoder,
    index_root: str | Path = DEFAULT_INDEX_ROOT,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    top_reviews_per_business_aspect: int = 20,
    force: bool = False,
) -> SemanticRecallBuildResult:
    """为每家商户每种特征保留意思最接近的完整评论编号。"""

    if top_reviews_per_business_aspect < 1:
        raise ValueError("top_reviews_per_business_aspect must be positive")
    index_path = Path(index_root)
    destination = Path(output_root)
    segments_path = index_path / "review_segments.parquet"
    embeddings_path = index_path / "segment_embeddings.npy"
    index_manifest_path = index_path / "manifest.json"
    for path in (segments_path, embeddings_path, index_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"review index file does not exist: {path}")
    actual_schema = pq.ParquetFile(segments_path).schema_arrow
    if not actual_schema.equals(REVIEW_INDEX_SEGMENT_SCHEMA, check_metadata=False):
        raise SemanticRecallBuildError("review index segment schema is invalid")
    index_manifest = ReviewVectorIndexManifest.model_validate_json(
        index_manifest_path.read_text(encoding="utf-8")
    )
    if index_manifest.model != encoder.model or index_manifest.dimension != encoder.dimension:
        raise SemanticRecallBuildError("semantic encoder does not match review vectors")

    _, vocabulary = load_review_aspect_settings(config_root)
    definitions = build_aspect_recall_definitions(vocabulary)
    definitions_payload = [item.model_dump(mode="json") for item in definitions]
    definitions_hash = _sha256_json(definitions_payload)
    source_hashes = {
        "review_index_manifest": _sha256_file(index_manifest_path),
        "review_segments": _sha256_file(segments_path),
        "segment_embeddings": _sha256_file(embeddings_path),
    }
    if not force:
        existing = _existing_manifest(
            destination,
            source_hashes=source_hashes,
            definitions_hash=definitions_hash,
            model=encoder.model,
            dimension=encoder.dimension,
            top_k=top_reviews_per_business_aspect,
        )
        if existing is not None:
            return SemanticRecallBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    anchor_vectors, anchor_ids, aspect_anchor_indices = _encode_anchors(
        encoder,
        definitions,
    )
    rows = _semantic_rows(
        segments_path,
        embeddings_path,
        anchor_vectors=anchor_vectors,
        anchor_ids=anchor_ids,
        aspect_anchor_indices=aspect_anchor_indices,
        top_k=top_reviews_per_business_aspect,
    )
    if not rows:
        raise SemanticRecallBuildError("semantic recall produced no candidates")
    destination.mkdir(parents=True, exist_ok=True)
    candidate_path = destination / "semantic_candidate_aspects.parquet"
    partial_candidate = destination / "semantic_candidate_aspects.parquet.partial"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATE_ASPECT_SCHEMA),
        partial_candidate,
        compression="zstd",
    )
    aspect_counts = Counter(str(row["aspect"]) for row in rows)
    manifest = SemanticRecallManifest(
        source_paths={
            "review_index_manifest": str(index_manifest_path.resolve()),
            "review_segments": str(segments_path.resolve()),
            "segment_embeddings": str(embeddings_path.resolve()),
        },
        source_sha256=source_hashes,
        definitions_sha256=definitions_hash,
        model=encoder.model,
        dimension=encoder.dimension,
        top_reviews_per_business_aspect=top_reviews_per_business_aspect,
        business_count=index_manifest.business_count,
        candidate_aspect_count=len(rows),
        aspect_counts=dict(sorted(aspect_counts.items())),
        output_sha256={"semantic_candidates": _sha256_file(partial_candidate)},
    )
    partial_manifest = destination / "semantic_manifest.json.partial"
    manifest_path = destination / "semantic_manifest.json"
    partial_manifest.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial_candidate, candidate_path)
    os.replace(partial_manifest, manifest_path)
    return SemanticRecallBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )
