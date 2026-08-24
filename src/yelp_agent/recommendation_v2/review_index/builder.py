"""筛出餐饮评论片段，并为每个片段保存可复用的本地文字向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.recommendation_v2.business_facts import BUSINESS_FACT_SCHEMA
from yelp_agent.review_rag.config import load_review_rag_config
from yelp_agent.review_rag.schema import REVIEW_SEGMENT_SCHEMA
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
)
from yelp_agent.semantic_embedding.encoder import EmbeddingEncoder

from .schema import (
    REVIEW_INDEX_SEGMENT_SCHEMA,
    ReviewVectorIndexBuildResult,
    ReviewVectorIndexManifest,
)

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEGMENT_SOURCE = (
    _PROJECT_ROOT / "data" / "features" / "review_rag" / "v1" / "review_segments.parquet"
)
DEFAULT_BUSINESS_FACT_SOURCE = (
    _RECOMMENDATION_ROOT / "data" / "business_facts" / "v1" / "business_facts.parquet"
)
DEFAULT_REVIEW_RAG_CONFIG = _PROJECT_ROOT / "configs" / "review_rag.yaml"
DEFAULT_OUTPUT_ROOT = _RECOMMENDATION_ROOT / "data" / "review_index" / "v1"


class ReviewVectorIndexBuildError(RuntimeError):
    """评论片段、餐饮范围或向量输出不完整时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parquet(path: Path, expected: pa.Schema, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = pq.ParquetFile(path).schema_arrow
    if not actual.equals(expected, check_metadata=False):
        raise ReviewVectorIndexBuildError(f"{label} has an unexpected schema")


def _business_ids(path: Path) -> list[str]:
    values = pq.read_table(path, columns=["business_id"]).column(0).to_pylist()
    result = [str(value) for value in values]
    if not result or len(result) != len(set(result)):
        raise ReviewVectorIndexBuildError("business IDs must be nonempty and unique")
    return result


def _build_scoped_segments(
    source: Path,
    business_ids: list[str],
    output: Path,
    *,
    batch_size: int,
) -> tuple[int, int]:
    """把旧评论片段限制到2447家餐厅，并增加稳定的向量行号。"""

    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    business_values = pa.array(business_ids, type=pa.string())
    writer: pq.ParquetWriter | None = None
    row_index = 0
    review_ids: set[str] = set()
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size):
            business_column = batch.column(batch.schema.get_field_index("business_id"))
            mask = pc.is_in(business_column, value_set=business_values)
            scoped = pa.Table.from_batches([batch]).filter(mask)
            if scoped.num_rows == 0:
                continue
            indices = pa.array(
                range(row_index, row_index + scoped.num_rows),
                type=pa.int64(),
            )
            indexed = scoped.add_column(0, "row_index", indices).cast(
                REVIEW_INDEX_SEGMENT_SCHEMA
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    partial,
                    REVIEW_INDEX_SEGMENT_SCHEMA,
                    compression="zstd",
                )
            writer.write_table(indexed)
            row_index += scoped.num_rows
            review_ids.update(str(value) for value in scoped.column("review_id").to_pylist())
        if writer is None or row_index == 0:
            raise ReviewVectorIndexBuildError("restaurant scope contains no review segments")
        writer.close()
        writer = None
        os.replace(partial, output)
    except Exception:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        raise
    return row_index, len(review_ids)


def _progress_payload(
    *,
    segment_sha256: str,
    model: str,
    dimension: int,
    segment_count: int,
    next_row: int,
) -> dict[str, object]:
    return {
        "segment_sha256": segment_sha256,
        "model": model,
        "dimension": dimension,
        "segment_count": segment_count,
        "next_row": next_row,
    }


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _resume_row(
    progress_path: Path,
    partial_embeddings: Path,
    *,
    expected: dict[str, object],
) -> int:
    if not progress_path.is_file() or not partial_embeddings.is_file():
        return 0
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    for key, value in expected.items():
        if payload.get(key) != value:
            return 0
    next_row = payload.get("next_row")
    if not isinstance(next_row, int) or not 0 <= next_row <= int(expected["segment_count"]):
        return 0
    return next_row


def _normalized_float16(vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ReviewVectorIndexBuildError("embedding model returned an invalid vector")
    return (values / norms).astype("<f2")


def _build_embeddings(
    segments_path: Path,
    output_path: Path,
    progress_path: Path,
    *,
    encoder: EmbeddingEncoder,
    segment_count: int,
) -> None:
    """按评论片段行号生成可恢复的 float16 向量矩阵。"""

    partial = output_path.with_name("segment_embeddings.partial.npy")
    segment_hash = _sha256_file(segments_path)
    expected = {
        "segment_sha256": segment_hash,
        "model": encoder.model,
        "dimension": encoder.dimension,
        "segment_count": segment_count,
    }
    next_row = _resume_row(
        progress_path,
        partial,
        expected=expected,
    )
    if next_row == 0:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        matrix = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype="<f2",
            shape=(segment_count, encoder.dimension),
        )
    else:
        matrix = np.load(partial, mmap_mode="r+")
        if matrix.shape != (segment_count, encoder.dimension) or matrix.dtype != np.float16:
            del matrix
            partial.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            return _build_embeddings(
                segments_path,
                output_path,
                progress_path,
                encoder=encoder,
                segment_count=segment_count,
            )

    started = time.monotonic()
    last_report = started
    processed = next_row
    parquet = pq.ParquetFile(segments_path)
    for batch in parquet.iter_batches(batch_size=2048, columns=["row_index", "text"]):
        batch_start = int(batch.column(0)[0].as_py())
        batch_end = batch_start + batch.num_rows
        if batch_end <= next_row:
            continue
        offset = max(0, next_row - batch_start)
        texts = [str(value) for value in batch.column(1).slice(offset).to_pylist()]
        write_start = batch_start + offset

        # 同一批里先按文字长度靠拢，避免很短的片段陪着最长片段一起补空并做无用计算。
        # 向量写回原始行号，因此这种提速不会改变评论片段与向量之间的对应关系。
        ordered = sorted(enumerate(texts), key=lambda item: len(item[1]))
        for text_offset in range(0, len(ordered), encoder.batch_size):
            group = ordered[text_offset : text_offset + encoder.batch_size]
            values = [text for _, text in group]
            encoded = encoder.encode(values, input_type="document")
            target_rows = np.asarray(
                [write_start + original_offset for original_offset, _ in group],
                dtype=np.int64,
            )
            matrix[target_rows] = _normalized_float16(encoded.vectors)

        # 只有整个2048行数据块都写完，才记录可恢复位置；中途退出最多重做这一小块。
        processed = batch_end
        now = time.monotonic()
        matrix.flush()
        _write_progress(
            progress_path,
            _progress_payload(next_row=processed, **expected),
        )
        if processed % 2048 == 0 or now - last_report >= 30:
            elapsed = max(0.001, now - started)
            rate = max(0.0, (processed - next_row) / elapsed)
            print(
                f"embedded={processed}/{segment_count} rate={rate:.1f}/s",
                flush=True,
            )
            last_report = now
    if processed != segment_count:
        del matrix
        raise ReviewVectorIndexBuildError(
            f"embedding rows stopped at {processed}, expected {segment_count}"
        )
    matrix.flush()
    del matrix
    os.replace(partial, output_path)
    progress_path.unlink(missing_ok=True)


def _existing_manifest(
    output_root: Path,
    *,
    source_hashes: dict[str, str],
    model: str,
    dimension: int,
) -> ReviewVectorIndexManifest | None:
    manifest_path = output_root / "manifest.json"
    paths = {
        "segments": output_root / "review_segments.parquet",
        "embeddings": output_root / "segment_embeddings.npy",
    }
    if not manifest_path.is_file() or not all(path.is_file() for path in paths.values()):
        return None
    try:
        manifest = ReviewVectorIndexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if (
        manifest.source_sha256 != source_hashes
        or manifest.model != model
        or manifest.dimension != dimension
        or manifest.output_sha256
        != {name: _sha256_file(path) for name, path in paths.items()}
    ):
        return None
    return manifest


def build_review_vector_index(
    encoder: EmbeddingEncoder,
    segment_source: str | Path = DEFAULT_SEGMENT_SOURCE,
    business_fact_source: str | Path = DEFAULT_BUSINESS_FACT_SOURCE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
    segment_batch_size: int = 5000,
) -> ReviewVectorIndexBuildResult:
    """生成一份同时服务14种特征粗筛和以后任意评论检索的向量底座。"""

    segment_path = Path(segment_source)
    business_path = Path(business_fact_source)
    destination = Path(output_root)
    _validate_parquet(segment_path, REVIEW_SEGMENT_SCHEMA, "review segment source")
    _validate_parquet(business_path, BUSINESS_FACT_SCHEMA, "business fact source")
    source_hashes = {
        "review_segments": _sha256_file(segment_path),
        "business_facts": _sha256_file(business_path),
    }
    if not force:
        existing = _existing_manifest(
            destination,
            source_hashes=source_hashes,
            model=encoder.model,
            dimension=encoder.dimension,
        )
        if existing is not None:
            return ReviewVectorIndexBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    destination.mkdir(parents=True, exist_ok=True)
    scoped_segments = destination / "review_segments.parquet"
    embeddings = destination / "segment_embeddings.npy"
    progress = destination / "embedding_progress.json"
    business_ids = _business_ids(business_path)

    if force or not scoped_segments.is_file():
        segment_count, review_count = _build_scoped_segments(
            segment_path,
            business_ids,
            scoped_segments,
            batch_size=segment_batch_size,
        )
    else:
        _validate_parquet(
            scoped_segments,
            REVIEW_INDEX_SEGMENT_SCHEMA,
            "scoped review segments",
        )
        segment_count = pq.ParquetFile(scoped_segments).metadata.num_rows
        review_ids: set[str] = set()
        for batch in pq.ParquetFile(scoped_segments).iter_batches(
            batch_size=segment_batch_size,
            columns=["review_id"],
        ):
            review_ids.update(str(value) for value in batch.column(0).to_pylist())
        review_count = len(review_ids)

    if force:
        embeddings.unlink(missing_ok=True)
    if not embeddings.is_file():
        _build_embeddings(
            scoped_segments,
            embeddings,
            progress,
            encoder=encoder,
            segment_count=segment_count,
        )
    matrix = np.load(embeddings, mmap_mode="r")
    if matrix.shape != (segment_count, encoder.dimension) or matrix.dtype != np.float16:
        raise ReviewVectorIndexBuildError("saved embedding matrix has an invalid shape")
    del matrix

    output_hashes = {
        "segments": _sha256_file(scoped_segments),
        "embeddings": _sha256_file(embeddings),
    }
    manifest = ReviewVectorIndexManifest(
        source_paths={
            "review_segments": str(segment_path.resolve()),
            "business_facts": str(business_path.resolve()),
        },
        source_sha256=source_hashes,
        model=encoder.model,
        dimension=encoder.dimension,
        business_count=len(business_ids),
        review_count=review_count,
        segment_count=segment_count,
        output_sha256=output_hashes,
    )
    manifest_path = destination / "manifest.json"
    partial_manifest = destination / "manifest.json.partial"
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
    os.replace(partial_manifest, manifest_path)
    return ReviewVectorIndexBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成全部餐饮评论片段的本地文字向量")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        choices=range(1, 65),
        default=16,
        metavar="1..64",
        help="本地模型一次处理的评论片段数；显存足够时可适当调大",
    )
    parser.add_argument("--segment-source", type=Path, default=DEFAULT_SEGMENT_SOURCE)
    parser.add_argument(
        "--business-fact-source",
        type=Path,
        default=DEFAULT_BUSINESS_FACT_SOURCE,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    rag_config = load_review_rag_config(DEFAULT_REVIEW_RAG_CONFIG)
    semantic_config = rag_config.semantic_config().model_copy(
        update={"batch_size": args.embedding_batch_size}
    )
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        semantic_config,
        LocalEmbeddingEnvironment(
            model_path=args.model_path,
            python_executable=args.python_executable,
            device=args.device,
        ),
    )
    try:
        result = build_review_vector_index(
            encoder,
            args.segment_source,
            args.business_fact_source,
            args.output_root,
            force=args.force,
        )
    finally:
        encoder.close()
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
