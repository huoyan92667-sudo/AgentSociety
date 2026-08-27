"""Qdrant 评论片段导入和候选商家范围内的分组检索。"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
from pydantic import Field
from qdrant_client import QdrantClient, models

from yelp_agent.models import StrictModel

from .schema import QdrantSegmentHit

DEFAULT_COLLECTION_NAME = "recommendation_v2_review_segments_v3_hybrid"
DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"
BM25_MODEL_NAME = "qdrant/bm25"
BM25_AVERAGE_LENGTH = 91.09041914634108
BM25_K = 1.5
BM25_B = 0.75
RRF_K = 60


class QdrantImportManifest(StrictModel):
    """记录 Qdrant 中的数据与本地片段、向量文件严格对应。"""

    schema_version: Literal[2] = 2
    collection_name: str = Field(min_length=1)
    segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension: int = Field(ge=1)
    point_count: int = Field(ge=1)
    dense_vector_name: str = DENSE_VECTOR_NAME
    bm25_vector_name: str = BM25_VECTOR_NAME
    bm25_model_name: str = BM25_MODEL_NAME
    bm25_average_length: float = Field(gt=0)


@dataclass(frozen=True, slots=True)
class QdrantHybridSearchResult:
    """一次批量混合检索的融合结果和两路真实耗时。"""

    results: list[dict[str, list[QdrantSegmentHit]]]
    wall_latency_ms: float
    dense_latency_ms: float
    bm25_latency_ms: float
    fusion_latency_ms: float
    dense_hit_count: int
    bm25_hit_count: int
    bm25_only_hit_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QdrantReviewSegmentStore:
    """把 Qdrant 细节封装起来，排序器只关心“限定商家后找片段”。"""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._client = client
        self.collection_name = collection_name

    @classmethod
    def from_url(
        cls,
        url: str = "http://localhost:6333",
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        prefer_grpc: bool = False,
    ) -> QdrantReviewSegmentStore:
        return cls(
            QdrantClient(
                url=url,
                grpc_port=6334,
                prefer_grpc=prefer_grpc,
                timeout=120,
            ),
            collection_name=collection_name,
        )

    def close(self) -> None:
        self._client.close()

    def import_index(
        self,
        index_root: str | Path,
        *,
        recreate: bool = False,
        batch_size: int = 256,
    ) -> QdrantImportManifest:
        """把本地片段和向量导入 Qdrant；固定行号使重复导入不会产生重复点。"""

        root = Path(index_root)
        segments_path = root / "review_segments.parquet"
        embeddings_path = root / "segment_embeddings.npy"
        if not segments_path.is_file() or not embeddings_path.is_file():
            raise FileNotFoundError("review evidence vector index is incomplete")
        vectors = np.load(embeddings_path, mmap_mode="r")
        point_count, dimension = vectors.shape
        segment_count = pq.ParquetFile(segments_path).metadata.num_rows
        if segment_count != point_count:
            raise ValueError("segment rows and embedding rows do not match")

        hashes = {
            "segments": _sha256_file(segments_path),
            "embeddings": _sha256_file(embeddings_path),
        }
        # 新混合集合使用独立清单，旧纯向量集合和清单继续保留，导入失败时
        # 正式流程仍可切回旧集合，不需要破坏已有数据。
        manifest_path = root.parent / "qdrant_hybrid_import_manifest.json"
        existing_manifest = self._load_manifest(manifest_path)
        if self._client.collection_exists(self.collection_name):
            if recreate:
                self._client.delete_collection(self.collection_name)
            elif (
                existing_manifest is not None
                and existing_manifest.segments_sha256 == hashes["segments"]
                and existing_manifest.embeddings_sha256 == hashes["embeddings"]
                and existing_manifest.point_count == point_count
                and self._collection_point_count() == point_count
            ):
                return existing_manifest
            else:
                raise RuntimeError(
                    "Qdrant collection exists but does not match this index; "
                    "use recreate=True after confirming the target collection"
                )

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                BM25_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(on_disk=True),
                )
            },
            # 批量导入期间先不反复重建搜索索引，全部点写完后再统一开启。
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

        parquet = pq.ParquetFile(segments_path)
        uploaded = 0
        for batch in parquet.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            ids = [int(row["row_index"]) for row in rows]
            start = ids[0]
            expected_ids = list(range(start, start + len(ids)))
            if ids != expected_ids:
                raise ValueError("segment row indices must be contiguous")
            points = [
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: np.asarray(
                            vectors[point_id], dtype=np.float32
                        ).tolist(),
                        BM25_VECTOR_NAME: _bm25_document(str(row["text"])),
                    },
                    payload=self._payload(row),
                )
                for point_id, row in zip(ids, rows, strict=True)
            ]
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            uploaded += len(ids)
            if uploaded % 10000 < batch_size:
                print(f"qdrant_uploaded={uploaded}/{point_count}", flush=True)
        if uploaded != point_count or self._collection_point_count() != point_count:
            raise RuntimeError("Qdrant import stopped before every segment was stored")

        for field_name, field_schema in (
            ("business_id", models.PayloadSchemaType.KEYWORD),
            ("review_id", models.PayloadSchemaType.KEYWORD),
            ("review_timestamp", models.PayloadSchemaType.INTEGER),
        ):
            self._client.create_payload_index(
                self.collection_name,
                field_name,
                field_schema=field_schema,
                wait=True,
            )
        self._client.update_collection(
            self.collection_name,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )

        manifest = QdrantImportManifest(
            collection_name=self.collection_name,
            segments_sha256=hashes["segments"],
            embeddings_sha256=hashes["embeddings"],
            dimension=dimension,
            point_count=point_count,
            bm25_average_length=BM25_AVERAGE_LENGTH,
        )
        partial = manifest_path.with_name(manifest_path.name + ".partial")
        partial.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(partial, manifest_path)
        return manifest

    def search_grouped(
        self,
        query_vector: np.ndarray,
        business_ids: list[str],
        *,
        score_threshold: float,
        cutoff_time: datetime | None = None,
        group_size: int = 60,
        business_batch_size: int = 100,
    ) -> dict[str, list[QdrantSegmentHit]]:
        """Qdrant 先按商家过滤，再保证每家都有各自的召回结果。"""

        unique_ids = list(dict.fromkeys(business_ids))
        if len(unique_ids) != len(business_ids):
            raise ValueError("business IDs must be unique")
        if not unique_ids:
            return {}
        vector = np.asarray(query_vector, dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        result: dict[str, list[QdrantSegmentHit]] = {item: [] for item in unique_ids}
        for offset in range(0, len(unique_ids), business_batch_size):
            batch_ids = unique_ids[offset : offset + business_batch_size]
            conditions: list[models.Condition] = [
                models.FieldCondition(
                    key="business_id",
                    match=models.MatchAny(any=batch_ids),
                )
            ]
            if cutoff_time is not None:
                conditions.append(
                    models.FieldCondition(
                        key="review_timestamp",
                        range=models.Range(lt=_unix_timestamp(cutoff_time)),
                    )
                )
            groups = self._client.query_points_groups(
                collection_name=self.collection_name,
                query=vector.tolist(),
                using=DENSE_VECTOR_NAME,
                query_filter=models.Filter(must=conditions),
                group_by="business_id",
                limit=len(batch_ids),
                group_size=group_size,
                with_payload=True,
                # 完整向量在本机的 .npy 文件中按 point_id 读取。
                with_vectors=False,
                score_threshold=score_threshold,
            )
            for group in groups.groups:
                business_id = str(group.id)
                result[business_id] = [
                    self._hit(item).model_copy(update={"dense_rank": rank})
                    for rank, item in enumerate(group.hits, 1)
                ]
        return result

    def search_keyword_grouped(
        self,
        query_text: str,
        business_ids: list[str],
        *,
        cutoff_time: datetime | None = None,
        group_size: int = 60,
        business_batch_size: int = 100,
    ) -> dict[str, list[QdrantSegmentHit]]:
        """用Qdrant内置BM25在同一硬筛商家范围内按关键词找片段。"""

        unique_ids = list(dict.fromkeys(business_ids))
        if len(unique_ids) != len(business_ids):
            raise ValueError("business IDs must be unique")
        if not unique_ids:
            return {}
        cleaned = query_text.strip()
        if not cleaned:
            raise ValueError("BM25 query text must be nonempty")
        result: dict[str, list[QdrantSegmentHit]] = {item: [] for item in unique_ids}
        for offset in range(0, len(unique_ids), business_batch_size):
            batch_ids = unique_ids[offset : offset + business_batch_size]
            conditions: list[models.Condition] = [
                models.FieldCondition(
                    key="business_id",
                    match=models.MatchAny(any=batch_ids),
                )
            ]
            if cutoff_time is not None:
                conditions.append(
                    models.FieldCondition(
                        key="review_timestamp",
                        range=models.Range(lt=_unix_timestamp(cutoff_time)),
                    )
                )
            groups = self._client.query_points_groups(
                collection_name=self.collection_name,
                query=_bm25_document(cleaned),
                using=BM25_VECTOR_NAME,
                query_filter=models.Filter(must=conditions),
                group_by="business_id",
                limit=len(batch_ids),
                group_size=group_size,
                with_payload=True,
                with_vectors=False,
            )
            for group in groups.groups:
                business_id = str(group.id)
                result[business_id] = [
                    self._hit(item).model_copy(update={"bm25_rank": rank})
                    for rank, item in enumerate(group.hits, 1)
                ]
        return result

    def search_grouped_many(
        self,
        query_vectors: list[np.ndarray],
        business_ids: list[str],
        *,
        score_threshold: float,
        cutoff_time: datetime | None = None,
        group_size: int = 25,
        max_concurrency: int = 4,
    ) -> list[dict[str, list[QdrantSegmentHit]]]:
        """有限并行执行多种评论说法，返回顺序与查询向量完全一致。"""

        if not query_vectors:
            return []
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")

        def search(vector: np.ndarray) -> dict[str, list[QdrantSegmentHit]]:
            return self.search_grouped(
                vector,
                business_ids,
                score_threshold=score_threshold,
                cutoff_time=cutoff_time,
                group_size=group_size,
            )

        # Qdrant 客户端内部使用连接池；限制并行数可以缩短串行等待，
        # 又不会让十几条查询同时把本地服务压满。
        workers = min(max_concurrency, len(query_vectors))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(search, query_vectors))

    def search_keyword_grouped_many(
        self,
        query_texts: list[str],
        business_ids: list[str],
        *,
        cutoff_time: datetime | None = None,
        group_size: int = 25,
        max_concurrency: int = 4,
    ) -> list[dict[str, list[QdrantSegmentHit]]]:
        """有限并行执行多种关键词说法，返回顺序与输入文字完全一致。"""

        if not query_texts:
            return []
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")

        def search(text: str) -> dict[str, list[QdrantSegmentHit]]:
            return self.search_keyword_grouped(
                text,
                business_ids,
                cutoff_time=cutoff_time,
                group_size=group_size,
            )

        workers = min(max_concurrency, len(query_texts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(search, query_texts))

    def search_hybrid_grouped_many(
        self,
        query_texts: list[str],
        query_vectors: list[np.ndarray],
        business_ids: list[str],
        *,
        score_threshold: float,
        cutoff_time: datetime | None = None,
        group_size: int = 25,
        max_concurrency: int = 4,
        rrf_k: int = RRF_K,
    ) -> QdrantHybridSearchResult:
        """并行执行向量和BM25两路查询，再按每家商户的名次做RRF融合。"""

        if len(query_texts) != len(query_vectors):
            raise ValueError("query texts and vectors must have the same length")
        if rrf_k < 1:
            raise ValueError("RRF k must be positive")
        wall_started = perf_counter()

        def dense_route() -> tuple[list[dict[str, list[QdrantSegmentHit]]], float]:
            started = perf_counter()
            values = self.search_grouped_many(
                query_vectors,
                business_ids,
                score_threshold=score_threshold,
                cutoff_time=cutoff_time,
                group_size=group_size,
                max_concurrency=max_concurrency,
            )
            return values, (perf_counter() - started) * 1000

        def bm25_route() -> tuple[list[dict[str, list[QdrantSegmentHit]]], float]:
            started = perf_counter()
            values = self.search_keyword_grouped_many(
                query_texts,
                business_ids,
                cutoff_time=cutoff_time,
                group_size=group_size,
                max_concurrency=max_concurrency,
            )
            return values, (perf_counter() - started) * 1000

        # 两条路线彼此独立，同时查询能够避免把混合检索简单变成双倍串行等待。
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(dense_route)
            bm25_future = executor.submit(bm25_route)
            dense_results, dense_latency_ms = dense_future.result()
            bm25_results, bm25_latency_ms = bm25_future.result()

        fusion_started = perf_counter()
        fused = _fuse_grouped_results(
            dense_results,
            bm25_results,
            business_ids,
            group_size=group_size,
            rrf_k=rrf_k,
        )
        fusion_latency_ms = (perf_counter() - fusion_started) * 1000
        dense_ids = _segment_ids(dense_results)
        bm25_ids = _segment_ids(bm25_results)
        return QdrantHybridSearchResult(
            results=fused,
            wall_latency_ms=(perf_counter() - wall_started) * 1000,
            dense_latency_ms=dense_latency_ms,
            bm25_latency_ms=bm25_latency_ms,
            fusion_latency_ms=fusion_latency_ms,
            dense_hit_count=_grouped_hit_count(dense_results),
            bm25_hit_count=_grouped_hit_count(bm25_results),
            bm25_only_hit_count=len(bm25_ids - dense_ids),
        )

    def _collection_point_count(self) -> int:
        return int(self._client.get_collection(self.collection_name).points_count or 0)

    @staticmethod
    def _payload(row: dict[str, object]) -> dict[str, object]:
        review_time = row["review_time"]
        return {
            "segment_id": str(row["segment_id"]),
            "review_id": str(row["review_id"]),
            "business_id": str(row["business_id"]),
            "user_id": str(row["user_id"]),
            "review_time": review_time.isoformat(),  # type: ignore[union-attr]
            "review_timestamp": _unix_timestamp(review_time),  # type: ignore[arg-type]
            "stars": float(row["stars"]),
            "useful": int(row["useful"]),
            "segment_index": int(row["segment_index"]),
            "segment_text": str(row["text"]),
            "review_text_sha256": str(row["review_text_sha256"]),
        }

    @staticmethod
    def _hit(point: models.ScoredPoint) -> QdrantSegmentHit:
        payload = point.payload or {}
        return QdrantSegmentHit(
            point_id=int(point.id),
            segment_id=str(payload["segment_id"]),
            review_id=str(payload["review_id"]),
            business_id=str(payload["business_id"]),
            user_id=str(payload["user_id"]),
            review_time=str(payload["review_time"]),
            stars=float(payload["stars"]),
            useful=int(payload["useful"]),
            segment_index=int(payload["segment_index"]),
            segment_text=str(payload["segment_text"]),
            review_text_sha256=str(payload["review_text_sha256"]),
            route_similarity=float(point.score),
        )

    @staticmethod
    def _load_manifest(path: Path) -> QdrantImportManifest | None:
        if not path.is_file():
            return None
        try:
            return QdrantImportManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None


def _unix_timestamp(value: datetime) -> int:
    """Yelp 时间没有时区；统一按 UTC 解释，避免受开发机时区影响。"""

    if value.tzinfo is None:
        return calendar.timegm(value.utctimetuple())
    return int(value.timestamp())


def _bm25_document(text: str) -> models.Document:
    """统一导入和查询的BM25分词配置，并保留not/no等否定词。"""

    options = models.Bm25Config(
        k=BM25_K,
        b=BM25_B,
        avg_len=BM25_AVERAGE_LENGTH,
        language="english",
        # 默认英文停用词会删除not，正反证据检索不能接受这种信息损失。
        stopwords=models.StopwordsSet(languages=[], custom=[]),
        stemmer=models.SnowballParams(type="snowball", language="english"),
    )
    return models.Document(
        text=text,
        model=BM25_MODEL_NAME,
        # qdrant-client 1.x 的gRPC转换器只接受普通映射；REST与gRPC
        # 都能读取同一份序列化结果，因此导入和在线查询保持完全一致。
        options=options.model_dump(mode="json", exclude_none=True),
    )


def _fuse_grouped_results(
    dense_results: list[dict[str, list[QdrantSegmentHit]]],
    bm25_results: list[dict[str, list[QdrantSegmentHit]]],
    business_ids: list[str],
    *,
    group_size: int,
    rrf_k: int,
) -> list[dict[str, list[QdrantSegmentHit]]]:
    """两路原始分数不可直接相加，按各自名次融合后再保留相同数量。"""

    if len(dense_results) != len(bm25_results):
        raise ValueError("dense and BM25 result batches must align")
    output: list[dict[str, list[QdrantSegmentHit]]] = []
    for dense_grouped, bm25_grouped in zip(
        dense_results, bm25_results, strict=True
    ):
        per_query: dict[str, list[QdrantSegmentHit]] = {}
        for business_id in business_ids:
            rows: dict[str, tuple[QdrantSegmentHit, float]] = {}
            for route, hits in (
                ("dense", dense_grouped.get(business_id, [])),
                ("bm25", bm25_grouped.get(business_id, [])),
            ):
                for rank, hit in enumerate(hits, 1):
                    previous = rows.get(hit.segment_id)
                    score = 1.0 / (rrf_k + rank)
                    if previous is None:
                        merged = hit.model_copy(
                            update={
                                "dense_rank": rank if route == "dense" else None,
                                "bm25_rank": rank if route == "bm25" else None,
                                "route_similarity": score,
                            }
                        )
                        rows[hit.segment_id] = (merged, score)
                        continue
                    merged, previous_score = previous
                    update = {
                        "dense_rank": (
                            rank if route == "dense" else merged.dense_rank
                        ),
                        "bm25_rank": (
                            rank if route == "bm25" else merged.bm25_rank
                        ),
                        "route_similarity": previous_score + score,
                    }
                    rows[hit.segment_id] = (
                        merged.model_copy(update=update),
                        previous_score + score,
                    )
            ordered = sorted(
                rows.values(),
                key=lambda item: (
                    -item[1],
                    -item[0].useful,
                    -item[0].review_time.timestamp(),
                    item[0].segment_id,
                ),
            )
            per_query[business_id] = [item[0] for item in ordered[:group_size]]
        output.append(per_query)
    return output


def _grouped_hit_count(
    results: list[dict[str, list[QdrantSegmentHit]]],
) -> int:
    return sum(len(hits) for grouped in results for hits in grouped.values())


def _segment_ids(
    results: list[dict[str, list[QdrantSegmentHit]]],
) -> set[str]:
    return {
        hit.segment_id
        for grouped in results
        for hits in grouped.values()
        for hit in hits
    }
