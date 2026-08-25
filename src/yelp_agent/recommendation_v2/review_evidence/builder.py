"""生成新版重叠评论片段，并复用现有本地模型建立文字向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_v2.review_index import (
    ReviewVectorIndexBuildResult,
    ReviewVectorIndexManifest,
)
from yelp_agent.recommendation_v2.review_index.schema import (
    REVIEW_INDEX_SEGMENT_SCHEMA,
)
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.review_rag.schema import REVIEW_SEGMENT_SCHEMA
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
)
from yelp_agent.semantic_embedding.encoder import EmbeddingEncoder

from .segmenter import OverlapSegmentConfig, segment_review_with_overlap

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REVIEW_SOURCE = _PROJECT_ROOT / "data" / "processed" / "reviews.parquet"
DEFAULT_BUSINESS_FACT_SOURCE = (
    _RECOMMENDATION_ROOT / "data" / "business_facts" / "v1" / "business_facts.parquet"
)
DEFAULT_EVIDENCE_ROOT = _RECOMMENDATION_ROOT / "data" / "review_evidence" / "v1"
DEFAULT_REUSE_INDEX_ROOT = _RECOMMENDATION_ROOT / "data" / "review_index" / "v1"
DEFAULT_REVIEW_RAG_CONFIG = _PROJECT_ROOT / "configs" / "review_rag.yaml"


class ReviewSegmentSourceManifest(StrictModel):
    """记录新版片段是如何从原评论生成的。"""

    schema_version: Literal[1] = 1
    segmentation_version: Literal["2.0.0"] = "2.0.0"
    source_paths: dict[str, str]
    source_sha256: dict[str, str]
    max_chars: int = Field(ge=100)
    overlap_sentences: int = Field(ge=0)
    business_count: int = Field(ge=1)
    review_count: int = Field(ge=1)
    segment_count: int = Field(ge=1)
    long_sentence_review_count: int = Field(ge=0)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReviewSegmentSourceBuildResult(StrictModel):
    """评论重切分阶段的公开结果。"""

    status: Literal["written", "skipped"]
    segment_path: str = Field(min_length=1)
    manifest: ReviewSegmentSourceManifest


class ReviewEvidenceIndexBuildResult(StrictModel):
    """评论重切分和向量生成两个阶段的合并结果。"""

    segments: ReviewSegmentSourceBuildResult
    vectors: ReviewVectorIndexBuildResult


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_business_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"business facts do not exist: {path}")
    values = pq.read_table(path, columns=["business_id"]).column(0).to_pylist()
    business_ids = [str(value) for value in values]
    if not business_ids or len(business_ids) != len(set(business_ids)):
        raise ValueError("business fact IDs must be nonempty and unique")
    return business_ids


def _existing_segment_manifest(
    manifest_path: Path,
    segment_path: Path,
    *,
    source_hashes: dict[str, str],
    config: OverlapSegmentConfig,
) -> ReviewSegmentSourceManifest | None:
    if not manifest_path.is_file() or not segment_path.is_file():
        return None
    try:
        manifest = ReviewSegmentSourceManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if (
        manifest.source_sha256 != source_hashes
        or manifest.max_chars != config.max_chars
        or manifest.overlap_sentences != config.overlap_sentences
        or manifest.output_sha256 != _sha256_file(segment_path)
    ):
        return None
    if not pq.ParquetFile(segment_path).schema_arrow.equals(
        REVIEW_SEGMENT_SCHEMA,
        check_metadata=False,
    ):
        return None
    return manifest


def build_review_segment_source(
    review_source: str | Path = DEFAULT_REVIEW_SOURCE,
    business_fact_source: str | Path = DEFAULT_BUSINESS_FACT_SOURCE,
    output_root: str | Path = DEFAULT_EVIDENCE_ROOT,
    *,
    config: OverlapSegmentConfig | None = None,
    force: bool = False,
    source_batch_size: int = 2000,
    write_batch_size: int = 5000,
) -> ReviewSegmentSourceBuildResult:
    """只切餐饮商家的评论；逐批写入，避免把全部评论放进内存。"""

    config = config or OverlapSegmentConfig()
    reviews = Path(review_source)
    business_facts = Path(business_fact_source)
    if not reviews.is_file():
        raise FileNotFoundError(f"review source does not exist: {reviews}")
    destination = Path(output_root) / "segments"
    destination.mkdir(parents=True, exist_ok=True)
    segment_path = destination / "review_segments.parquet"
    manifest_path = destination / "manifest.json"
    source_hashes = {
        "reviews": _sha256_file(reviews),
        "business_facts": _sha256_file(business_facts),
    }
    if not force:
        existing = _existing_segment_manifest(
            manifest_path,
            segment_path,
            source_hashes=source_hashes,
            config=config,
        )
        if existing is not None:
            return ReviewSegmentSourceBuildResult(
                status="skipped",
                segment_path=str(segment_path.resolve()),
                manifest=existing,
            )

    business_ids = _load_business_ids(business_facts)
    allowed = pa.array(business_ids, type=pa.string())
    partial = segment_path.with_name(segment_path.name + ".partial")
    partial.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    pending: list[dict[str, object]] = []
    review_count = 0
    segment_count = 0
    long_sentence_review_count = 0

    def flush() -> None:
        nonlocal writer, pending
        if not pending:
            return
        table = pa.Table.from_pylist(pending, schema=REVIEW_SEGMENT_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(
                partial,
                REVIEW_SEGMENT_SCHEMA,
                compression="zstd",
            )
        writer.write_table(table)
        pending = []

    try:
        parquet = pq.ParquetFile(reviews)
        columns = [
            "review_id",
            "user_id",
            "business_id",
            "stars",
            "useful",
            "text",
            "date",
        ]
        for batch in parquet.iter_batches(batch_size=source_batch_size, columns=columns):
            business_column = batch.column(batch.schema.get_field_index("business_id"))
            mask = pc.is_in(business_column, value_set=allowed)
            scoped = pa.Table.from_batches([batch]).filter(mask)
            for row in scoped.to_pylist():
                built = segment_review_with_overlap(row, config)
                if not built.segments:
                    continue
                review_count += 1
                if built.split_long_sentence:
                    long_sentence_review_count += 1
                values = [item.model_dump() for item in built.segments]
                pending.extend(values)
                segment_count += len(values)
                if len(pending) >= write_batch_size:
                    flush()
            if review_count and review_count % 10000 < source_batch_size:
                print(
                    f"segmented_reviews={review_count} segments={segment_count}",
                    flush=True,
                )
        flush()
        if writer is None or segment_count == 0:
            raise RuntimeError("restaurant scope produced no comment segments")
        writer.close()
        writer = None
        os.replace(partial, segment_path)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise

    manifest = ReviewSegmentSourceManifest(
        source_paths={
            "reviews": str(reviews.resolve()),
            "business_facts": str(business_facts.resolve()),
        },
        source_sha256=source_hashes,
        max_chars=config.max_chars,
        overlap_sentences=config.overlap_sentences,
        business_count=len(business_ids),
        review_count=review_count,
        segment_count=segment_count,
        long_sentence_review_count=long_sentence_review_count,
        output_sha256=_sha256_file(segment_path),
    )
    partial_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    partial_manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial_manifest, manifest_path)
    return ReviewSegmentSourceBuildResult(
        status="written",
        segment_path=str(segment_path.resolve()),
        manifest=manifest,
    )


def build_review_evidence_index(
    encoder: EmbeddingEncoder,
    *,
    review_source: str | Path = DEFAULT_REVIEW_SOURCE,
    business_fact_source: str | Path = DEFAULT_BUSINESS_FACT_SOURCE,
    output_root: str | Path = DEFAULT_EVIDENCE_ROOT,
    config: OverlapSegmentConfig | None = None,
    force: bool = False,
    reuse_index_root: str | Path | None = DEFAULT_REUSE_INDEX_ROOT,
) -> ReviewEvidenceIndexBuildResult:
    """完成重切分和向量生成；后续 Qdrant 导入直接读取这里的结果。"""

    config = config or OverlapSegmentConfig()
    segments = build_review_segment_source(
        review_source,
        business_fact_source,
        output_root,
        config=config,
        force=force,
    )
    vectors = _build_vector_index_with_reuse(
        encoder=encoder,
        segment_source=Path(segments.segment_path),
        business_fact_source=Path(business_fact_source),
        output_root=Path(output_root) / "index",
        segment_manifest=segments.manifest,
        reuse_index_root=(
            Path(reuse_index_root) if reuse_index_root is not None else None
        ),
        force=force,
    )
    return ReviewEvidenceIndexBuildResult(segments=segments, vectors=vectors)


def _build_vector_index_with_reuse(
    *,
    encoder: EmbeddingEncoder,
    segment_source: Path,
    business_fact_source: Path,
    output_root: Path,
    segment_manifest: ReviewSegmentSourceManifest,
    reuse_index_root: Path | None,
    force: bool,
) -> ReviewVectorIndexBuildResult:
    """复用文字完全相同的旧向量，只编码重叠切分产生的新片段。"""

    output_root.mkdir(parents=True, exist_ok=True)
    indexed_segments = output_root / "review_segments.parquet"
    embeddings = output_root / "segment_embeddings.npy"
    manifest_path = output_root / "manifest.json"
    source_hashes = {
        "review_segments": _sha256_file(segment_source),
        "business_facts": _sha256_file(business_fact_source),
    }
    if not force and manifest_path.is_file() and indexed_segments.is_file() and embeddings.is_file():
        try:
            existing = ReviewVectorIndexManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            existing = None
        if (
            existing is not None
            and existing.source_sha256 == source_hashes
            and existing.model == encoder.model
            and existing.dimension == encoder.dimension
            and existing.output_sha256
            == {
                "segments": _sha256_file(indexed_segments),
                "embeddings": _sha256_file(embeddings),
            }
        ):
            return ReviewVectorIndexBuildResult(
                status="skipped",
                output_root=str(output_root.resolve()),
                manifest=existing,
            )

    if force or not indexed_segments.is_file():
        _add_stable_row_indices(segment_source, indexed_segments)
    segment_count = pq.ParquetFile(indexed_segments).metadata.num_rows
    matrix_valid = False
    if not force and embeddings.is_file():
        matrix = np.load(embeddings, mmap_mode="r")
        matrix_valid = (
            matrix.shape == (segment_count, encoder.dimension)
            and matrix.dtype == np.float16
        )
        del matrix
    if not matrix_valid:
        _write_embeddings_reusing_old(
            encoder=encoder,
            indexed_segments=indexed_segments,
            output_path=embeddings,
            reuse_index_root=reuse_index_root,
            segment_count=segment_count,
        )

    manifest = ReviewVectorIndexManifest(
        source_paths={
            "review_segments": str(segment_source.resolve()),
            "business_facts": str(business_fact_source.resolve()),
        },
        source_sha256=source_hashes,
        model=encoder.model,
        dimension=encoder.dimension,
        business_count=segment_manifest.business_count,
        review_count=segment_manifest.review_count,
        segment_count=segment_count,
        output_sha256={
            "segments": _sha256_file(indexed_segments),
            "embeddings": _sha256_file(embeddings),
        },
    )
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    partial.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, manifest_path)
    return ReviewVectorIndexBuildResult(
        status="written",
        output_root=str(output_root.resolve()),
        manifest=manifest,
    )


def _add_stable_row_indices(source: Path, output: Path) -> None:
    """片段文件顺序固定，因此顺序行号可以直接作为 Qdrant 点编号。"""

    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    next_row = 0
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=5000):
            table = pa.Table.from_batches([batch])
            row_indices = pa.array(
                range(next_row, next_row + table.num_rows),
                type=pa.int64(),
            )
            indexed = table.add_column(0, "row_index", row_indices).cast(
                REVIEW_INDEX_SEGMENT_SCHEMA
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    partial,
                    REVIEW_INDEX_SEGMENT_SCHEMA,
                    compression="zstd",
                )
            writer.write_table(indexed)
            next_row += table.num_rows
        if writer is None:
            raise RuntimeError("cannot index an empty segment source")
        writer.close()
        writer = None
        os.replace(partial, output)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise


def _write_embeddings_reusing_old(
    *,
    encoder: EmbeddingEncoder,
    indexed_segments: Path,
    output_path: Path,
    reuse_index_root: Path | None,
    segment_count: int,
) -> None:
    """按文字校验值复用向量；重复正文的向量数学上应完全相同。"""

    old_rows: dict[str, int] = {}
    old_vectors: np.ndarray | None = None
    if reuse_index_root is not None:
        old_segments = reuse_index_root / "review_segments.parquet"
        old_embeddings = reuse_index_root / "segment_embeddings.npy"
        old_manifest_path = reuse_index_root / "manifest.json"
        if old_segments.is_file() and old_embeddings.is_file() and old_manifest_path.is_file():
            try:
                old_manifest = ReviewVectorIndexManifest.model_validate_json(
                    old_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                old_manifest = None
            if (
                old_manifest is not None
                and old_manifest.model == encoder.model
                and old_manifest.dimension == encoder.dimension
            ):
                for batch in pq.ParquetFile(old_segments).iter_batches(
                    batch_size=10000,
                    columns=["row_index", "text_sha256"],
                ):
                    for row_index, text_hash in zip(
                        batch.column(0).to_pylist(),
                        batch.column(1).to_pylist(),
                    ):
                        old_rows.setdefault(str(text_hash), int(row_index))
                old_vectors = np.load(old_embeddings, mmap_mode="r")

    partial = output_path.with_name("segment_embeddings.reuse.partial.npy")
    progress_path = output_path.with_name("vector_reuse_progress.json")
    progress_identity = {
        "segment_size": indexed_segments.stat().st_size,
        "segment_count": segment_count,
        "model": encoder.model,
        "dimension": encoder.dimension,
    }
    progress: dict[str, object] | None = None
    if partial.is_file() and progress_path.is_file():
        try:
            loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and all(
            loaded.get(key) == value for key, value in progress_identity.items()
        ):
            progress = loaded
    if progress is None:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        matrix = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype="<f2",
            shape=(segment_count, encoder.dimension),
        )
        next_row = 0
        reused = 0
        encoded_count = 0
    else:
        matrix = np.load(partial, mmap_mode="r+")
        if matrix.shape != (segment_count, encoder.dimension):
            raise RuntimeError("resumable vector matrix has an invalid shape")
        next_row = int(progress.get("next_row", 0))
        reused = int(progress.get("reused", 0))
        encoded_count = int(progress.get("newly_encoded", 0))
    for batch in pq.ParquetFile(indexed_segments).iter_batches(
        batch_size=2048,
        columns=["row_index", "text_sha256", "text"],
    ):
        rows = [int(value) for value in batch.column(0).to_pylist()]
        batch_end = rows[-1] + 1
        if batch_end <= next_row:
            continue
        hashes = [str(value) for value in batch.column(1).to_pylist()]
        texts = [str(value) for value in batch.column(2).to_pylist()]
        missing: list[tuple[int, str]] = []
        for row_index, text_hash, text in zip(rows, hashes, texts):
            old_row = old_rows.get(text_hash)
            if old_vectors is not None and old_row is not None:
                matrix[row_index] = old_vectors[old_row]
                reused += 1
            else:
                missing.append((row_index, text))

        ordered = sorted(missing, key=lambda item: len(item[1]))
        for offset in range(0, len(ordered), encoder.batch_size):
            group = ordered[offset : offset + encoder.batch_size]
            encoded = encoder.encode(
                [text for _, text in group],
                input_type="document",
            )
            values = np.asarray(encoded.vectors, dtype=np.float32)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            matrix[[row for row, _ in group]] = values.astype("<f2")
            encoded_count += len(group)
        next_row = batch_end
        matrix.flush()
        progress_payload = {
            **progress_identity,
            "next_row": next_row,
            "reused": reused,
            "newly_encoded": encoded_count,
        }
        # Windows 上安全软件可能短暂占用已有目标文件，直接覆写这份小型进度记录
        # 比频繁 os.replace 更稳定；若中途损坏，下次只会放弃断点并重新生成。
        progress_path.write_text(
            json.dumps(progress_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if (reused + encoded_count) % 10000 < batch.num_rows:
            print(
                f"vectors_ready={reused + encoded_count}/{segment_count} "
                f"reused={reused} newly_encoded={encoded_count}",
                flush=True,
            )
    matrix.flush()
    del matrix
    del old_vectors
    if reused + encoded_count != segment_count:
        partial.unlink(missing_ok=True)
        raise RuntimeError("vector reuse stopped before every segment was written")
    os.replace(partial, output_path)
    progress_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重新切分餐饮评论并生成本地向量")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    rag_config = load_review_rag_config(DEFAULT_REVIEW_RAG_CONFIG)
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": args.batch_size}),
        LocalEmbeddingEnvironment(
            model_path=args.model_path,
            python_executable=args.python_executable,
            device=args.device,
        ),
    )
    try:
        result = build_review_evidence_index(
            encoder,
            config=OverlapSegmentConfig(
                max_chars=args.max_chars,
                overlap_sentences=args.overlap_sentences,
            ),
            force=args.force,
        )
    finally:
        encoder.close()
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
