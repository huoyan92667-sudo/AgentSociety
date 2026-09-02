"""为菜系多样化500家商户准备微调模型判断前的分批数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__:
    from . import prepare as base
else:
    import prepare as base


DEFAULT_SOURCE_PROJECT = Path(r"C:\Users\29072\PycharmProjects\AgentSociety")
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_EMBEDDING_MODEL = Path(r"D:\models\Qwen3-Embedding-0.6B")
DEFAULT_EMBEDDING_PYTHON = Path(r"D:\anaconda3\python.exe")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output_dir() -> Path:
    return (
        _project_root()
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "review_evidence"
        / "v1"
        / "offline_profiles"
        / "restaurants_diverse_500_v1"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--source-project", type=Path, default=DEFAULT_SOURCE_PROJECT)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-python", type=Path, default=DEFAULT_EMBEDDING_PYTHON
    )
    parser.add_argument("--shard-business-count", type=int, default=100)
    parser.add_argument("--per-side-limit", type=int, default=15)
    parser.add_argument("--route-segment-limit", type=int, default=30)
    parser.add_argument("--search-concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    for name in (
        "shard_business_count",
        "per_side_limit",
        "route_segment_limit",
        "search_concurrency",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_side_candidates(
    high: list[dict[str, Any]], low: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_review: dict[str, dict[str, Any]] = {}
    for side, candidates in (("high", high), ("low", low)):
        for candidate in candidates:
            review_id = candidate["fact"].review_id
            entry = by_review.setdefault(
                review_id,
                {"sides": [], "side_candidates": {}, "segments": {}},
            )
            entry["sides"].append(side)
            entry["side_candidates"][side] = candidate
            for segment_id, segment in candidate["segments"].items():
                existing = entry["segments"].get(segment_id)
                if existing is None:
                    entry["segments"][segment_id] = {
                        **segment,
                        "routes": set(segment["routes"]),
                    }
                else:
                    existing["support_score"] += segment["support_score"]
                    existing["routes"].update(segment["routes"])
    return by_review


def _materialize_shard(
    *,
    shard_dir: Path,
    businesses: list[dict[str, Any]],
    definitions: dict[str, Any],
    selected: dict[tuple[str, str, str], list[dict[str, Any]]],
    full_review_store: Any,
    per_side_limit: int,
) -> dict[str, Any]:
    all_review_ids = {
        candidate["fact"].review_id
        for candidates in selected.values()
        for candidate in candidates
    }
    full_texts = full_review_store.get_many(sorted(all_review_ids))
    records: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    business_order = {
        item["business_id"]: item["selection_index"] for item in businesses
    }
    aspect_order = {
        item["id"]: index for index, item in enumerate(definitions["aspects"])
    }
    for business in businesses:
        business_id = business["business_id"]
        counts[business_id] = {}
        for definition in definitions["aspects"]:
            aspect_id = definition["id"]
            high = selected.get((business_id, aspect_id, "high"), [])
            low = selected.get((business_id, aspect_id, "low"), [])
            merged = _merge_side_candidates(high, low)
            counts[business_id][aspect_id] = {
                "high_candidate_count": len(high),
                "low_candidate_count": len(low),
                "unique_candidate_count": len(merged),
                "high_limit_reached": len(high) >= per_side_limit,
                "low_limit_reached": len(low) >= per_side_limit,
            }
            for review_id, item in merged.items():
                fact = next(iter(item["side_candidates"].values()))["fact"]
                segments = [
                    {**segment, "routes": sorted(segment["routes"])}
                    for segment in item["segments"].values()
                ]
                model_text = base._model_review_text(full_texts[review_id], segments)
                sample_id = hashlib.sha256(
                    f"{business_id}\0{aspect_id}\0{review_id}".encode()
                ).hexdigest()
                records.append(
                    {
                        "schema_version": "2.0",
                        "sample_id": sample_id,
                        "business_id": business_id,
                        "aspect_id": aspect_id,
                        "review_id": review_id,
                        "user_id": fact.user_id,
                        "review_time": fact.review_time.isoformat(),
                        "stars": fact.stars,
                        "useful": fact.useful,
                        "model_review_text": model_text,
                        "retrieved_sides": sorted(item["sides"]),
                        "input_char_count": len(model_text),
                    }
                )
    records.sort(
        key=lambda item: (
            business_order[item["business_id"]],
            aspect_order[item["aspect_id"]],
            item["review_id"],
        )
    )
    review_rows = [
        {"review_id": review_id, "full_review_text": full_texts[review_id]}
        for review_id in sorted(full_texts)
    ]
    input_path = shard_dir / "model_inputs.jsonl"
    reviews_path = shard_dir / "reviews.jsonl"
    selection_path = shard_dir / "selection.json"
    _write_jsonl(input_path, records)
    _write_jsonl(reviews_path, review_rows)
    _write_json(selection_path, {"businesses": businesses})
    return {
        "business_count": len(businesses),
        "candidate_relation_count": len(records),
        "unique_review_count": len(review_rows),
        "counts_by_business_and_aspect": counts,
        "files": {
            "model_inputs": input_path.name,
            "reviews": reviews_path.name,
            "selection": selection_path.name,
        },
        "sha256": {
            "model_inputs": _sha256(input_path),
            "reviews": _sha256(reviews_path),
            "selection": _sha256(selection_path),
        },
    }


def prepare(args: argparse.Namespace) -> Path:
    from yelp_agent.recommendation_v2.review_evidence.full_reviews import (
        FullReviewStore,
    )
    from yelp_agent.recommendation_v2.review_evidence.qdrant_store import (
        QdrantReviewSegmentStore,
    )

    started = perf_counter()
    output_dir = args.output_dir.resolve()
    selection_path = output_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "selection.json is missing; run select_diverse_businesses.py first"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    businesses = selection.get("businesses", [])
    if len(businesses) != 500 or len(
        {item["business_id"] for item in businesses}
    ) != len(businesses):
        raise ValueError("selection must contain exactly 500 unique businesses")

    definitions, _contract = base._load_and_validate_definitions()
    definition_sha256 = hashlib.sha256(base._definition_path().read_bytes()).hexdigest()
    dense_descriptors = base._dense_query_descriptors(definitions)
    keyword_descriptors = base._keyword_query_descriptors(definitions)
    vectors, vector_manifest = base._load_or_encode_query_vectors(
        output_dir=output_dir,
        descriptors=dense_descriptors,
        definition_sha256=definition_sha256,
        embedding_model=args.embedding_model,
        embedding_python=args.embedding_python,
    )
    shutil.copy2(base._definition_path(), output_dir / "fixed_aspects.v1.json")
    shutil.copy2(base._teacher_template_path(), output_dir / "model_contract.v1.json")

    reviews_path = args.source_project / "data" / "processed" / "reviews.parquet"
    store = QdrantReviewSegmentStore.from_url(args.qdrant_url)
    full_review_store = FullReviewStore(reviews_path)
    shard_root = output_dir / "input_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_manifests: list[dict[str, Any]] = []
    total_search_ms = 0.0
    try:
        for start in range(0, len(businesses), args.shard_business_count):
            shard_number = start // args.shard_business_count + 1
            shard_businesses = businesses[start : start + args.shard_business_count]
            shard_name = f"shard_{shard_number:04d}"
            shard_dir = shard_root / shard_name
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_started = perf_counter()
            selected, search_metrics = base._search_and_select(
                store=store,
                business_ids=[item["business_id"] for item in shard_businesses],
                dense_descriptors=dense_descriptors,
                dense_vectors=vectors,
                keyword_descriptors=keyword_descriptors,
                route_segment_limit=args.route_segment_limit,
                per_side_limit=args.per_side_limit,
                search_concurrency=args.search_concurrency,
            )
            total_search_ms += search_metrics["search_wall_latency_ms"]
            shard_manifest = _materialize_shard(
                shard_dir=shard_dir,
                businesses=shard_businesses,
                definitions=definitions,
                selected=selected,
                full_review_store=full_review_store,
                per_side_limit=args.per_side_limit,
            )
            shard_manifest.update(
                {
                    "schema_version": "2.0",
                    "shard_name": shard_name,
                    "search_metrics": search_metrics,
                    "wall_latency_ms": (perf_counter() - shard_started) * 1000,
                }
            )
            _write_json(shard_dir / "shard_manifest.json", shard_manifest)
            shard_manifests.append(shard_manifest)
            print(
                f"[SHARD {shard_number}/5] businesses={len(shard_businesses)} "
                f"relations={shard_manifest['candidate_relation_count']} "
                f"reviews={shard_manifest['unique_review_count']} "
                f"ms={shard_manifest['wall_latency_ms']:.1f}"
            )
    finally:
        store.close()
        full_review_store.close()

    server_script = Path(__file__).with_name("server_judge.py")
    server_readme = Path(__file__).with_name("SERVER_README_DIVERSE_500.md")
    if not server_script.is_file() or not server_readme.is_file():
        raise FileNotFoundError("500-business server bundle is incomplete")
    shutil.copy2(server_script, output_dir / "run_server_judge.py")
    shutil.copy2(server_readme, output_dir / "SERVER_README.md")
    reference_time = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "2.0",
        "created_at": reference_time,
        "reference_time": reference_time,
        "purpose": "diverse_500_restaurants_fixed_14_aspect_offline_profiles",
        "business_count": len(businesses),
        "aspect_count": len(definitions["aspects"]),
        "shard_business_count": args.shard_business_count,
        "shard_count": len(shard_manifests),
        "candidate_relation_count": sum(
            item["candidate_relation_count"] for item in shard_manifests
        ),
        "unique_reviews_within_shards_sum": sum(
            item["unique_review_count"] for item in shard_manifests
        ),
        "input_policy": {
            "model_contract": "服务器根据固定合同和model_review_text还原训练时消息",
            "full_review_storage": "每个分片按review_id只保存一次完整评论",
            "per_side_limit": args.per_side_limit,
            "route_segment_limit": args.route_segment_limit,
            "expansion": "disabled for fixed 14 aspects",
        },
        "query_vector_manifest": vector_manifest,
        "qdrant_url": args.qdrant_url,
        "timing_ms": {
            "search_total": total_search_ms,
            "total": (perf_counter() - started) * 1000,
        },
        "shards": [
            {
                "name": item["shard_name"],
                "business_count": item["business_count"],
                "candidate_relation_count": item["candidate_relation_count"],
                "unique_review_count": item["unique_review_count"],
                "wall_latency_ms": item["wall_latency_ms"],
            }
            for item in shard_manifests
        ],
        "files": {
            "selection": "selection.json",
            "fixed_aspects": "fixed_aspects.v1.json",
            "model_contract": "model_contract.v1.json",
            "input_shards": "input_shards",
            "server_runner": "run_server_judge.py",
            "server_readme": "SERVER_README.md",
        },
    }
    manifest_path = output_dir / "prepare_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        f"[PREPARE] businesses=500 relations={manifest['candidate_relation_count']} "
        f"shards={manifest['shard_count']} total_ms={manifest['timing_ms']['total']:.1f}"
    )
    print(f"[PREPARE] output={output_dir}")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
