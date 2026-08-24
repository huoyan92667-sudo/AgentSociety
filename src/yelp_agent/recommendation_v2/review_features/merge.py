"""合并旧词语和意思相近两路候选，并保存完整原始评论。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.recommendation_v2.schema import ASPECT_FIELDS

from .schema import (
    CANDIDATE_ASPECT_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    CandidateRecallBuildResult,
    CandidateRecallManifest,
)

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REVIEW_SOURCE = _PROJECT_ROOT / "data" / "processed" / "reviews.parquet"
DEFAULT_OUTPUT_ROOT = _RECOMMENDATION_ROOT / "data" / "review_features" / "v1"


class CandidateRecallBuildError(RuntimeError):
    """两路候选存在重复键、错位商家或输出结构不合法时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _existing_manifest(
    output_root: Path,
    *,
    source_hashes: dict[str, str],
) -> CandidateRecallManifest | None:
    manifest_path = output_root / "manifest.json"
    paths = {
        "candidate_reviews": output_root / "candidate_reviews.parquet",
        "candidate_aspects": output_root / "candidate_aspects.parquet",
        "recall_audit_samples": output_root / "recall_audit_samples.json",
    }
    if not manifest_path.is_file() or not all(path.is_file() for path in paths.values()):
        return None
    try:
        manifest = CandidateRecallManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if (
        manifest.source_sha256 != source_hashes
        or manifest.output_sha256
        != {name: _sha256_file(path) for name, path in paths.items()}
    ):
        return None
    return manifest


def _check_unique_keys(connection: duckdb.DuckDBPyConnection, path: Path) -> None:
    duplicate_count = connection.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT review_id, business_id, aspect, count(*) AS copies
            FROM read_parquet('{_sql_path(path)}')
            GROUP BY review_id, business_id, aspect
            HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if int(duplicate_count) != 0:
        raise CandidateRecallBuildError(f"candidate source contains duplicate keys: {path}")


def _write_merged_aspects(
    connection: duckdb.DuckDBPyConnection,
    keyword_path: Path,
    semantic_path: Path,
    output: Path,
) -> None:
    raw_output = output.with_name(output.name + ".duckdb")
    raw_output.unlink(missing_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT
                coalesce(k.review_id, s.review_id)::VARCHAR AS review_id,
                coalesce(k.business_id, s.business_id)::VARCHAR AS business_id,
                coalesce(k.aspect, s.aspect)::VARCHAR AS aspect,
                (k.review_id IS NOT NULL)::BOOLEAN AS keyword_hit,
                (s.review_id IS NOT NULL)::BOOLEAN AS semantic_hit,
                coalesce(k.matched_terms, []::VARCHAR[]) AS matched_terms,
                s.semantic_score::FLOAT AS semantic_score,
                coalesce(s.matched_anchor_ids, []::VARCHAR[]) AS matched_anchor_ids
            FROM read_parquet('{_sql_path(keyword_path)}') k
            FULL OUTER JOIN read_parquet('{_sql_path(semantic_path)}') s
            USING (review_id, business_id, aspect)
            ORDER BY business_id, review_id, aspect
        ) TO '{_sql_path(raw_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    _cast_parquet(raw_output, output, CANDIDATE_ASPECT_SCHEMA)
    raw_output.unlink(missing_ok=True)


def _write_complete_reviews(
    connection: duckdb.DuckDBPyConnection,
    review_source: Path,
    aspects_path: Path,
    output: Path,
) -> None:
    raw_output = output.with_name(output.name + ".duckdb")
    raw_output.unlink(missing_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT
                r.review_id::VARCHAR AS review_id,
                r.business_id::VARCHAR AS business_id,
                r.user_id::VARCHAR AS user_id,
                r.date::TIMESTAMP AS review_time,
                r.stars::DOUBLE AS stars,
                r.useful::BIGINT AS useful,
                r.text::VARCHAR AS review_text,
                sha256(r.text)::VARCHAR AS review_text_sha256
            FROM read_parquet('{_sql_path(review_source)}') r
            INNER JOIN (
                SELECT DISTINCT review_id, business_id
                FROM read_parquet('{_sql_path(aspects_path)}')
            ) c USING (review_id, business_id)
            ORDER BY r.business_id, r.review_id
        ) TO '{_sql_path(raw_output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    _cast_parquet(raw_output, output, CANDIDATE_REVIEW_SCHEMA)
    raw_output.unlink(missing_ok=True)


def _cast_parquet(source: Path, output: Path, schema: pa.Schema) -> None:
    """把 DuckDB 临时结果流式改写成项目声明的严格非空结构。"""

    writer = pq.ParquetWriter(output, schema, compression="zstd")
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=10_000):
            writer.write_table(pa.Table.from_batches([batch]).cast(schema))
    finally:
        writer.close()


def _audit_samples(
    connection: duckdb.DuckDBPyConnection,
    aspects_path: Path,
    reviews_path: Path,
    *,
    per_group: int = 3,
) -> dict[str, object]:
    result: dict[str, object] = {}
    route_queries = {
        "keyword_only": (
            "a.keyword_hit AND NOT a.semantic_hit",
            "a.semantic_score DESC NULLS LAST, a.review_id",
        ),
        "semantic_only": (
            "a.semantic_hit AND NOT a.keyword_hit",
            "a.semantic_score DESC NULLS LAST, a.review_id",
        ),
        # 不能只看最高分样例。尾部样例专门暴露“每店保留前几条”带来的弱候选，
        # 为下一步相关性细判断和候选数量调整提供真实依据。
        "semantic_only_weak_tail": (
            "a.semantic_hit AND NOT a.keyword_hit",
            "a.semantic_score ASC NULLS LAST, a.review_id",
        ),
        "both": (
            "a.keyword_hit AND a.semantic_hit",
            "a.semantic_score DESC NULLS LAST, a.review_id",
        ),
    }
    for aspect in ASPECT_FIELDS:
        groups: dict[str, object] = {}
        for route, (condition, order_by) in route_queries.items():
            rows = connection.execute(
                f"""
                SELECT
                    a.review_id,
                    a.business_id,
                    a.matched_terms,
                    a.semantic_score,
                    a.matched_anchor_ids,
                    r.review_text
                FROM read_parquet('{_sql_path(aspects_path)}') a
                JOIN read_parquet('{_sql_path(reviews_path)}') r
                USING (review_id, business_id)
                WHERE a.aspect = ? AND {condition}
                ORDER BY {order_by}
                LIMIT ?
                """,
                [aspect, per_group],
            ).fetchall()
            groups[route] = [
                {
                    "review_id": row[0],
                    "business_id": row[1],
                    "matched_terms": row[2],
                    "semantic_score": row[3],
                    "matched_anchor_ids": row[4],
                    "review_text": row[5],
                }
                for row in rows
            ]
        result[aspect] = groups
    return {
        "meaning": (
            "只用于检查两路粗筛是否找到相关完整评论，不是最终正负判断；"
            "semantic_only_weak_tail 专门展示最弱候选，不能把它当成优质证据"
        ),
        "samples_per_aspect_route": per_group,
        "aspects": result,
    }


def merge_review_candidates(
    review_source: str | Path = DEFAULT_REVIEW_SOURCE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
) -> CandidateRecallBuildResult:
    """按评论和特征去重两路候选，并为候选编号保存一份完整原始评论。"""

    review_path = Path(review_source)
    destination = Path(output_root)
    keyword_path = destination / "keyword_candidate_aspects.parquet"
    semantic_path = destination / "semantic_candidate_aspects.parquet"
    keyword_manifest = destination / "keyword_manifest.json"
    semantic_manifest = destination / "semantic_manifest.json"
    for path in (
        review_path,
        keyword_path,
        semantic_path,
        keyword_manifest,
        semantic_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"candidate merge source does not exist: {path}")
    source_paths = {
        "reviews": review_path,
        "keyword_candidates": keyword_path,
        "semantic_candidates": semantic_path,
        "keyword_manifest": keyword_manifest,
        "semantic_manifest": semantic_manifest,
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    if not force:
        existing = _existing_manifest(destination, source_hashes=source_hashes)
        if existing is not None:
            return CandidateRecallBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    destination.mkdir(parents=True, exist_ok=True)
    aspect_output = destination / "candidate_aspects.parquet"
    review_output = destination / "candidate_reviews.parquet"
    audit_output = destination / "recall_audit_samples.json"
    partial_aspects = destination / "candidate_aspects.parquet.partial"
    partial_reviews = destination / "candidate_reviews.parquet.partial"
    partial_audit = destination / "recall_audit_samples.json.partial"
    partial_manifest = destination / "manifest.json.partial"
    partials = (partial_aspects, partial_reviews, partial_audit, partial_manifest)
    for path in partials:
        path.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        _check_unique_keys(connection, keyword_path)
        _check_unique_keys(connection, semantic_path)
        _write_merged_aspects(connection, keyword_path, semantic_path, partial_aspects)
        _write_complete_reviews(connection, review_path, partial_aspects, partial_reviews)
        actual_aspect_schema = pq.ParquetFile(partial_aspects).schema_arrow
        actual_review_schema = pq.ParquetFile(partial_reviews).schema_arrow
        if not actual_aspect_schema.equals(CANDIDATE_ASPECT_SCHEMA, check_metadata=False):
            raise CandidateRecallBuildError(
                "merged candidate aspect schema is invalid: "
                f"actual={actual_aspect_schema}, expected={CANDIDATE_ASPECT_SCHEMA}"
            )
        if not actual_review_schema.equals(CANDIDATE_REVIEW_SCHEMA, check_metadata=False):
            raise CandidateRecallBuildError("merged candidate review schema is invalid")

        candidate_review_count = pq.ParquetFile(partial_reviews).metadata.num_rows
        candidate_aspect_count = pq.ParquetFile(partial_aspects).metadata.num_rows
        route_rows = connection.execute(
            f"""
            SELECT
                CASE
                    WHEN keyword_hit AND semantic_hit THEN 'both'
                    WHEN keyword_hit THEN 'keyword_only'
                    ELSE 'semantic_only'
                END AS route,
                count(*)
            FROM read_parquet('{_sql_path(partial_aspects)}')
            GROUP BY route
            ORDER BY route
            """
        ).fetchall()
        aspect_rows = connection.execute(
            f"""
            SELECT aspect, count(*)
            FROM read_parquet('{_sql_path(partial_aspects)}')
            GROUP BY aspect
            ORDER BY aspect
            """
        ).fetchall()
        audit = _audit_samples(connection, partial_aspects, partial_reviews)
        partial_audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            "candidate_reviews": _sha256_file(partial_reviews),
            "candidate_aspects": _sha256_file(partial_aspects),
            "recall_audit_samples": _sha256_file(partial_audit),
        }
        manifest = CandidateRecallManifest(
            source_paths={name: str(path.resolve()) for name, path in source_paths.items()},
            source_sha256=source_hashes,
            candidate_review_count=candidate_review_count,
            candidate_aspect_count=candidate_aspect_count,
            route_counts={str(row[0]): int(row[1]) for row in route_rows},
            aspect_counts={str(row[0]): int(row[1]) for row in aspect_rows},
            output_sha256=output_hashes,
        )
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
        os.replace(partial_aspects, aspect_output)
        os.replace(partial_reviews, review_output)
        os.replace(partial_audit, audit_output)
        os.replace(partial_manifest, destination / "manifest.json")
    except Exception:
        for path in partials:
            path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    return CandidateRecallBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )
