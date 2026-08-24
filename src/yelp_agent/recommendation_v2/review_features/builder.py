"""为2447家餐厅构建旧词表评论候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import ValidationError

from yelp_agent.config import load_review_aspect_settings
from yelp_agent.data.reviews import REVIEW_SCHEMA
from yelp_agent.recommendation_v2.business_facts import BUSINESS_FACT_SCHEMA

from .definitions import AspectRecallDefinition, build_aspect_recall_definitions
from .keyword_recall import KeywordAspectMatcher
from .schema import (
    CANDIDATE_ASPECT_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    KeywordRecallBuildResult,
    KeywordRecallManifest,
)

_RECOMMENDATION_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REVIEW_SOURCE = _PROJECT_ROOT / "data" / "processed" / "reviews.parquet"
DEFAULT_BUSINESS_FACT_SOURCE = (
    _RECOMMENDATION_ROOT / "data" / "business_facts" / "v1" / "business_facts.parquet"
)
DEFAULT_CONFIG_ROOT = _PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = _RECOMMENDATION_ROOT / "data" / "review_features" / "v1"


class KeywordRecallBuildError(RuntimeError):
    """评论来源、餐饮范围或旧词表无法形成合法候选时抛出。"""


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


def _validate_parquet(path: Path, expected: pa.Schema, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = pq.ParquetFile(path).schema_arrow
    if not actual.equals(expected, check_metadata=False):
        raise KeywordRecallBuildError(f"{label} has an unexpected schema")


def _business_ids(path: Path) -> list[str]:
    values = pq.read_table(path, columns=["business_id"]).column(0).to_pylist()
    result = [str(value) for value in values]
    if not result or len(result) != len(set(result)):
        raise KeywordRecallBuildError("business IDs must be nonempty and unique")
    return result


def _definitions_payload(
    definitions: tuple[AspectRecallDefinition, ...],
) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in definitions]


def _existing_manifest(
    output_root: Path,
    *,
    source_hashes: dict[str, str],
    definitions_hash: str,
) -> KeywordRecallManifest | None:
    manifest_path = output_root / "keyword_manifest.json"
    paths = {
        "keyword_candidate_reviews": output_root / "keyword_candidate_reviews.parquet",
        "keyword_candidate_aspects": output_root / "keyword_candidate_aspects.parquet",
        "aspect_recall_definitions": output_root / "aspect_recall_definitions.json",
    }
    if not manifest_path.is_file() or not all(path.is_file() for path in paths.values()):
        return None
    try:
        manifest = KeywordRecallManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    if (
        manifest.source_sha256 != source_hashes
        or manifest.definitions_sha256 != definitions_hash
        or manifest.output_sha256
        != {name: _sha256_file(path) for name, path in paths.items()}
    ):
        return None
    return manifest


def build_keyword_review_candidates(
    review_source: str | Path = DEFAULT_REVIEW_SOURCE,
    business_fact_source: str | Path = DEFAULT_BUSINESS_FACT_SOURCE,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
    batch_size: int = 5000,
) -> KeywordRecallBuildResult:
    """用旧词表扫描餐饮评论，保存完整评论和评论对应的候选特征。"""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    review_path = Path(review_source)
    business_path = Path(business_fact_source)
    destination = Path(output_root)
    _validate_parquet(review_path, REVIEW_SCHEMA, "review source")
    _validate_parquet(business_path, BUSINESS_FACT_SCHEMA, "business fact source")

    _, vocabulary = load_review_aspect_settings(config_root)
    definitions = build_aspect_recall_definitions(vocabulary)
    definitions_payload = _definitions_payload(definitions)
    definitions_hash = _sha256_json(definitions_payload)
    source_hashes = {
        "reviews": _sha256_file(review_path),
        "business_facts": _sha256_file(business_path),
        "old_vocabulary": _sha256_file(Path(config_root) / "review_aspect_vocabulary.yaml"),
    }
    if not force:
        existing = _existing_manifest(
            destination,
            source_hashes=source_hashes,
            definitions_hash=definitions_hash,
        )
        if existing is not None:
            return KeywordRecallBuildResult(
                status="skipped",
                output_root=str(destination.resolve()),
                manifest=existing,
            )

    matcher = KeywordAspectMatcher(definitions)
    business_ids = _business_ids(business_path)
    business_values = pa.array(business_ids, type=pa.string())
    destination.mkdir(parents=True, exist_ok=True)
    review_output = destination / "keyword_candidate_reviews.parquet"
    aspect_output = destination / "keyword_candidate_aspects.parquet"
    definitions_output = destination / "aspect_recall_definitions.json"
    manifest_output = destination / "keyword_manifest.json"
    partial_review = destination / "keyword_candidate_reviews.parquet.partial"
    partial_aspect = destination / "keyword_candidate_aspects.parquet.partial"
    partial_definitions = destination / "aspect_recall_definitions.json.partial"
    partial_manifest = destination / "keyword_manifest.json.partial"
    partials = (
        partial_review,
        partial_aspect,
        partial_definitions,
        partial_manifest,
    )
    for path in partials:
        path.unlink(missing_ok=True)

    source_review_count = 0
    candidate_review_count = 0
    candidate_aspect_count = 0
    aspect_counts: Counter[str] = Counter()
    review_writer: pq.ParquetWriter | None = None
    aspect_writer: pq.ParquetWriter | None = None
    try:
        parquet = pq.ParquetFile(review_path)
        columns = [
            "review_id",
            "user_id",
            "business_id",
            "stars",
            "useful",
            "text",
            "date",
        ]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            business_column = batch.column(batch.schema.get_field_index("business_id"))
            mask = pc.is_in(business_column, value_set=business_values)
            scoped = pa.Table.from_batches([batch]).filter(mask)
            rows = scoped.to_pylist()
            source_review_count += len(rows)
            review_rows: list[dict[str, object]] = []
            aspect_rows: list[dict[str, object]] = []
            for row in rows:
                text = str(row["text"])
                matches = matcher.match(text)
                if not matches:
                    continue
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                review_rows.append(
                    {
                        "review_id": row["review_id"],
                        "business_id": row["business_id"],
                        "user_id": row["user_id"],
                        "review_time": row["date"],
                        "stars": row["stars"],
                        "useful": row["useful"],
                        "review_text": text,
                        "review_text_sha256": text_hash,
                    }
                )
                for aspect, terms in matches.items():
                    aspect_rows.append(
                        {
                            "review_id": row["review_id"],
                            "business_id": row["business_id"],
                            "aspect": aspect,
                            "keyword_hit": True,
                            "semantic_hit": False,
                            "matched_terms": terms,
                            "semantic_score": None,
                            "matched_anchor_ids": [],
                        }
                    )
                    aspect_counts[aspect] += 1
            if review_rows:
                review_table = pa.Table.from_pylist(
                    review_rows,
                    schema=CANDIDATE_REVIEW_SCHEMA,
                )
                aspect_table = pa.Table.from_pylist(
                    aspect_rows,
                    schema=CANDIDATE_ASPECT_SCHEMA,
                )
                if review_writer is None:
                    review_writer = pq.ParquetWriter(
                        partial_review,
                        CANDIDATE_REVIEW_SCHEMA,
                        compression="zstd",
                    )
                    aspect_writer = pq.ParquetWriter(
                        partial_aspect,
                        CANDIDATE_ASPECT_SCHEMA,
                        compression="zstd",
                    )
                review_writer.write_table(review_table)
                assert aspect_writer is not None
                aspect_writer.write_table(aspect_table)
                candidate_review_count += len(review_rows)
                candidate_aspect_count += len(aspect_rows)
        if review_writer is None or aspect_writer is None:
            raise KeywordRecallBuildError("old vocabulary produced no candidates")
        review_writer.close()
        review_writer = None
        aspect_writer.close()
        aspect_writer = None
        partial_definitions.write_text(
            json.dumps(definitions_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            "keyword_candidate_reviews": _sha256_file(partial_review),
            "keyword_candidate_aspects": _sha256_file(partial_aspect),
            "aspect_recall_definitions": _sha256_file(partial_definitions),
        }
        manifest = KeywordRecallManifest(
            source_paths={
                "reviews": str(review_path.resolve()),
                "business_facts": str(business_path.resolve()),
                "old_vocabulary": str(
                    (Path(config_root) / "review_aspect_vocabulary.yaml").resolve()
                ),
            },
            source_sha256=source_hashes,
            definitions_sha256=definitions_hash,
            business_count=len(business_ids),
            source_review_count=source_review_count,
            candidate_review_count=candidate_review_count,
            candidate_aspect_count=candidate_aspect_count,
            aspect_counts=dict(sorted(aspect_counts.items())),
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
        os.replace(partial_review, review_output)
        os.replace(partial_aspect, aspect_output)
        os.replace(partial_definitions, definitions_output)
        os.replace(partial_manifest, manifest_output)
    except Exception:
        if review_writer is not None:
            review_writer.close()
        if aspect_writer is not None:
            aspect_writer.close()
        for path in partials:
            path.unlink(missing_ok=True)
        raise

    return KeywordRecallBuildResult(
        status="written",
        output_root=str(destination.resolve()),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用旧词表粗筛餐饮评论")
    parser.add_argument("review_source", nargs="?", type=Path, default=DEFAULT_REVIEW_SOURCE)
    parser.add_argument(
        "business_fact_source",
        nargs="?",
        type=Path,
        default=DEFAULT_BUSINESS_FACT_SOURCE,
    )
    parser.add_argument("output_root", nargs="?", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_keyword_review_candidates(
        args.review_source,
        args.business_fact_source,
        args.config_root,
        args.output_root,
        force=args.force,
        batch_size=args.batch_size,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
