"""在本地为10家牛排馆准备固定14项软偏好的服务器输入。

这个阶段不加载评论判断模型。它只做四件事：选择商家、读取固定高低端
查找定义、在Qdrant中执行一次关键词与向量混合召回、生成与训练时完全
一致的模型消息。产出目录可以整体复制到4090服务器继续运行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

DEFAULT_SOURCE_PROJECT = Path(r"C:\Users\29072\PycharmProjects\AgentSociety")
DEFAULT_EMBEDDING_MODEL = Path(r"D:\models\Qwen3-Embedding-0.6B")
DEFAULT_EMBEDDING_PYTHON = Path(r"D:\anaconda3\python.exe")
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
RRF_K = 60


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output() -> Path:
    return (
        _project_root()
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "review_evidence"
        / "v1"
        / "offline_profiles"
        / "steakhouses_top10_v1"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output())
    parser.add_argument("--source-project", type=Path, default=DEFAULT_SOURCE_PROJECT)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-python", type=Path, default=DEFAULT_EMBEDDING_PYTHON
    )
    parser.add_argument("--business-count", type=int, default=10)
    parser.add_argument("--minimum-reviews", type=int, default=100)
    parser.add_argument("--per-side-limit", type=int, default=15)
    parser.add_argument(
        "--route-segment-limit",
        type=int,
        default=30,
        help="每条关键词或向量查询先取多少片段，再按评论编号合并",
    )
    parser.add_argument("--search-concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    for name in (
        "business_count",
        "minimum_reviews",
        "per_side_limit",
        "route_segment_limit",
        "search_concurrency",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')}必须大于0")
    if args.route_segment_limit < args.per_side_limit:
        parser.error("--route-segment-limit不能小于--per-side-limit")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _definition_path() -> Path:
    return Path(__file__).with_name("fixed_aspects.v1.json")


def _teacher_template_path() -> Path:
    return (
        Path(__file__).parents[1] / "training_data" / "teacher_input_templates.v1.json"
    )


def _load_and_validate_definitions() -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = _read_json(_definition_path())
    template = _read_json(_teacher_template_path())
    aspect_ids = [item["id"] for item in definitions.get("aspects", [])]
    template_ids = [item["id"] for item in template.get("aspects", [])]
    if len(aspect_ids) != 14 or len(set(aspect_ids)) != 14:
        raise ValueError(
            "fixed aspect retrieval definition must contain 14 unique aspects"
        )
    if aspect_ids != template_ids:
        raise ValueError(
            "retrieval definitions and model templates must use the same order"
        )
    required = {
        "high_keywords",
        "low_keywords",
        "high_semantic_queries",
        "low_semantic_queries",
    }
    for item in definitions["aspects"]:
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"{item['id']} is missing retrieval fields: {sorted(missing)}"
            )
        for key in required:
            values = item[key]
            minimum = 2 if key.endswith("semantic_queries") else 1
            if len(values) < minimum or any(not str(value).strip() for value in values):
                raise ValueError(f"{item['id']}.{key} is incomplete")
            normalized = [str(value).strip().casefold() for value in values]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{item['id']}.{key} contains duplicates")
    return definitions, template


def _select_steakhouses(
    *, facts_path: Path, reviews_path: Path, count: int, minimum_reviews: int
) -> list[dict[str, Any]]:
    """按可复现规则选出Yelp评分靠前且评论量足够的牛排类别商家。"""

    import duckdb

    if not facts_path.is_file() or not reviews_path.is_file():
        raise FileNotFoundError("business facts or full review parquet is missing")
    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            """
            WITH review_counts AS (
                SELECT business_id, count(*) AS actual_review_count
                FROM read_parquet(?)
                GROUP BY business_id
            )
            SELECT
                facts.business_id,
                facts.name,
                facts.rating,
                facts.review_count,
                counts.actual_review_count,
                facts.price_level,
                facts.city,
                facts.state,
                facts.categories
            FROM read_parquet(?) AS facts
            JOIN review_counts AS counts USING (business_id)
            WHERE list_contains(facts.categories, 'Steakhouses')
              AND counts.actual_review_count >= ?
            ORDER BY facts.rating DESC, facts.review_count DESC, facts.business_id
            LIMIT ?
            """,
            [str(reviews_path), str(facts_path), minimum_reviews, count],
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != count:
        raise RuntimeError(f"only selected {len(rows)} businesses, expected {count}")
    return [
        {
            "rank_by_selection_rule": rank,
            "business_id": str(row[0]),
            "name": str(row[1]),
            "rating": float(row[2]),
            "catalog_review_count": int(row[3]),
            "actual_review_count": int(row[4]),
            "price_level": None if row[5] is None else int(row[5]),
            "city": str(row[6]),
            "state": str(row[7]),
            "categories": [str(value) for value in row[8]],
        }
        for rank, row in enumerate(rows, 1)
    ]


def _dense_query_descriptors(
    definitions: dict[str, Any],
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for aspect in definitions["aspects"]:
        for side in ("high", "low"):
            for index, text in enumerate(aspect[f"{side}_semantic_queries"], 1):
                descriptors.append(
                    {
                        "aspect_id": aspect["id"],
                        "side": side,
                        "query_index": index,
                        "text": text,
                    }
                )
    return descriptors


def _keyword_query_descriptors(
    definitions: dict[str, Any],
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for aspect in definitions["aspects"]:
        for side in ("high", "low"):
            terms = aspect[f"{side}_keywords"]
            descriptors.append(
                {
                    "aspect_id": aspect["id"],
                    "side": side,
                    "terms": terms,
                    # 一项一端只发一次BM25请求，避免把词表长度变成请求倍数。
                    "text": " ".join(terms),
                }
            )
    return descriptors


def _load_or_encode_query_vectors(
    *,
    output_dir: Path,
    descriptors: list[dict[str, Any]],
    definition_sha256: str,
    embedding_model: Path,
    embedding_python: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    vector_path = output_dir / "fixed_query_vectors.npy"
    manifest_path = output_dir / "fixed_query_vectors_manifest.json"
    texts = [item["text"] for item in descriptors]
    if vector_path.is_file() and manifest_path.is_file():
        manifest = _read_json(manifest_path)
        vectors = np.load(vector_path)
        if (
            manifest.get("definition_sha256") == definition_sha256
            and manifest.get("texts") == texts
            and vectors.shape[0] == len(texts)
        ):
            return np.asarray(vectors, dtype=np.float32), {
                **manifest,
                "cache_hit": True,
                "wall_latency_ms": 0.0,
            }

    from yelp_agent.review_rag.config import load_review_rag_config
    from yelp_agent.semantic_embedding import (
        LocalEmbeddingEnvironment,
        LocalQwenEmbeddingEncoder,
    )

    started = perf_counter()
    rag_config = load_review_rag_config(_project_root() / "configs" / "review_rag.yaml")
    encoder = LocalQwenEmbeddingEncoder.from_environment(
        rag_config.semantic_config().model_copy(update={"batch_size": 16}),
        LocalEmbeddingEnvironment(
            model_path=embedding_model,
            python_executable=embedding_python,
            device="cuda",
        ),
    )
    vector_batches: list[np.ndarray] = []
    model_reported_latency_ms = 0.0
    model_input_tokens = 0
    try:
        # 固定定义共有56条向量说法，而本地编码器单批上限通常是16。
        # 保持同一个模型进程常驻，分批编码后再按原顺序拼回去。
        for start in range(0, len(texts), encoder.batch_size):
            encoded = encoder.encode(
                texts[start : start + encoder.batch_size], input_type="query"
            )
            vector_batches.append(np.asarray(encoded.vectors, dtype=np.float32))
            model_reported_latency_ms += float(encoded.latency_ms)
            model_input_tokens += int(encoded.input_tokens)
    finally:
        encoder.close()
    vectors = np.concatenate(vector_batches, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    np.save(vector_path, vectors)
    manifest = {
        "schema_version": "1.0",
        "definition_sha256": definition_sha256,
        "texts": texts,
        "vector_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "embedding_model": str(embedding_model),
        "cache_hit": False,
        "model_reported_latency_ms": model_reported_latency_ms,
        "model_input_tokens": model_input_tokens,
        "wall_latency_ms": (perf_counter() - started) * 1000,
    }
    _write_json(manifest_path, manifest)
    return vectors, manifest


def _add_hit(
    destination: dict[str, dict[str, Any]],
    hit: Any,
    *,
    route: str,
    query_index: int,
    rank: int,
    contributes_to_rrf: bool,
) -> None:
    item = destination.get(hit.review_id)
    if item is None:
        item = {
            "fact": hit,
            "rrf_score": 0.0,
            "dense_best_score": None,
            "bm25_best_score": None,
            "dense_match": False,
            "bm25_match": False,
            "segments": {},
            "route_hits": [],
        }
        destination[hit.review_id] = item
    # Qdrant返回的是评论片段。一条长评论可能命中多个片段，但它在同一次
    # 查询中只能贡献一个名次，否则长评论会仅因切得更多而获得额外优势。
    contribution = 1.0 / (RRF_K + rank) if contributes_to_rrf else 0.0
    item["rrf_score"] += contribution
    score_key = "dense_best_score" if route == "dense" else "bm25_best_score"
    previous = item[score_key]
    item[score_key] = (
        float(hit.route_similarity)
        if previous is None
        else max(float(previous), float(hit.route_similarity))
    )
    item[f"{route}_match"] = True
    item["route_hits"].append(
        {
            "route": route,
            "query_index": query_index,
            "rank": rank,
            "contributes_to_rrf": contributes_to_rrf,
            "route_score": float(hit.route_similarity),
            "segment_id": hit.segment_id,
        }
    )
    segment = item["segments"].get(hit.segment_id)
    if segment is None:
        segment = {
            "segment_id": hit.segment_id,
            "segment_index": hit.segment_index,
            "text": hit.segment_text,
            "support_score": 0.0,
            "routes": set(),
        }
        item["segments"][hit.segment_id] = segment
    segment["support_score"] += contribution
    segment["routes"].add(route)


def _search_and_select(
    *,
    store: Any,
    business_ids: list[str],
    dense_descriptors: list[dict[str, Any]],
    dense_vectors: np.ndarray,
    keyword_descriptors: list[dict[str, Any]],
    route_segment_limit: int,
    per_side_limit: int,
    search_concurrency: int,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    """两条路线各查一次，按商家、特征、高低端合并后截断。"""

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            store.search_grouped_many,
            [vector for vector in dense_vectors],
            business_ids,
            score_threshold=0.0,
            cutoff_time=None,
            group_size=route_segment_limit,
            max_concurrency=search_concurrency,
        )
        keyword_future = executor.submit(
            store.search_keyword_grouped_many,
            [item["text"] for item in keyword_descriptors],
            business_ids,
            cutoff_time=None,
            group_size=route_segment_limit,
            # qdrant-client 的本地稀疏查询预处理器共享可变批缓存，多线程偶发
            # "dictionary changed size"。单请求本身很快，串行更可靠。
            max_concurrency=1,
        )
        dense_results = dense_future.result()
        keyword_results = keyword_future.result()
    search_ms = (perf_counter() - started) * 1000

    buckets: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    dense_hit_count = 0
    for query_index, (descriptor, grouped) in enumerate(
        zip(dense_descriptors, dense_results, strict=True)
    ):
        for business_id, hits in grouped.items():
            dense_hit_count += len(hits)
            bucket = buckets[(business_id, descriptor["aspect_id"], descriptor["side"])]
            seen_reviews: set[str] = set()
            review_rank = 0
            for segment_rank, hit in enumerate(hits, 1):
                contributes = hit.review_id not in seen_reviews
                if contributes:
                    seen_reviews.add(hit.review_id)
                    review_rank += 1
                _add_hit(
                    bucket,
                    hit,
                    route="dense",
                    query_index=query_index,
                    rank=review_rank if contributes else segment_rank,
                    contributes_to_rrf=contributes,
                )

    keyword_hit_count = 0
    for query_index, (descriptor, grouped) in enumerate(
        zip(keyword_descriptors, keyword_results, strict=True)
    ):
        for business_id, hits in grouped.items():
            keyword_hit_count += len(hits)
            bucket = buckets[(business_id, descriptor["aspect_id"], descriptor["side"])]
            seen_reviews = set()
            review_rank = 0
            for segment_rank, hit in enumerate(hits, 1):
                contributes = hit.review_id not in seen_reviews
                if contributes:
                    seen_reviews.add(hit.review_id)
                    review_rank += 1
                _add_hit(
                    bucket,
                    hit,
                    route="bm25",
                    query_index=query_index,
                    rank=review_rank if contributes else segment_rank,
                    contributes_to_rrf=contributes,
                )

    selected: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key, by_review in buckets.items():
        selected[key] = sorted(
            by_review.values(),
            key=lambda item: (
                -item["rrf_score"],
                -float(item["dense_best_score"] or 0.0),
                -int(item["fact"].useful),
                item["fact"].review_id,
            ),
        )[:per_side_limit]
    return selected, {
        "search_wall_latency_ms": search_ms,
        "dense_query_count": len(dense_descriptors),
        "keyword_query_count": len(keyword_descriptors),
        "dense_segment_hit_count": dense_hit_count,
        "keyword_segment_hit_count": keyword_hit_count,
        "expansion_pass_count": 0,
    }


def _model_review_text(full_text: str, segments: list[dict[str, Any]]) -> str:
    """短评论给完整原文；长评论拼接最高命中的完整上下文片段。"""

    if len(full_text) <= 900:
        return full_text
    ordered = sorted(
        segments,
        key=lambda item: (-float(item["support_score"]), item["segment_index"]),
    )
    pieces: list[str] = []
    current_length = 0
    for item in ordered:
        text = " ".join(str(item["text"]).split()).strip()
        if not text or text in pieces:
            continue
        separator = 5 if pieces else 0
        available = 900 - current_length - separator
        if available <= 0:
            break
        pieces.append(text[:available])
        current_length += min(len(text), available) + separator
    return "\n\n...\n\n".join(pieces)[:900]


def _messages(
    *, template: dict[str, Any], aspect: dict[str, Any], review_text: str
) -> list[dict[str, str]]:
    model_input = {
        "aspect_id": aspect["id"],
        "definition": aspect["definition"],
        "relevance_scale": template["common_relevance_scale"],
        "strength_scale": aspect["strength_scale"],
        "special_rules": aspect["special_rules"],
        "review_text": review_text,
    }
    return [
        {"role": "system", "content": template["system_prompt"]},
        {
            "role": "user",
            "content": json.dumps(
                model_input, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def _materialize_inputs(
    *,
    output_dir: Path,
    businesses: list[dict[str, Any]],
    definitions: dict[str, Any],
    template: dict[str, Any],
    selected: dict[tuple[str, str, str], list[dict[str, Any]]],
    full_review_store: Any,
    per_side_limit: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    aspect_templates = {item["id"]: item for item in template["aspects"]}
    all_ids = {
        item["fact"].review_id for values in selected.values() for item in values
    }
    full_texts = full_review_store.get_many(sorted(all_ids))
    records: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for business in businesses:
        business_id = business["business_id"]
        counts[business_id] = {}
        for definition in definitions["aspects"]:
            aspect_id = definition["id"]
            high = selected.get((business_id, aspect_id, "high"), [])
            low = selected.get((business_id, aspect_id, "low"), [])
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

            counts[business_id][aspect_id] = {
                "high_candidate_count": len(high),
                "low_candidate_count": len(low),
                "unique_candidate_count": len(by_review),
                "high_limit_reached": len(high) >= per_side_limit,
                "low_limit_reached": len(low) >= per_side_limit,
            }
            for review_id, merged in sorted(by_review.items()):
                side_candidates = merged["side_candidates"]
                fact = next(iter(side_candidates.values()))["fact"]
                segments = [
                    {**segment, "routes": sorted(segment["routes"])}
                    for segment in merged["segments"].values()
                ]
                model_text = _model_review_text(full_texts[review_id], segments)
                sample_id = _sha256_bytes(
                    f"{business_id}\0{aspect_id}\0{review_id}".encode()
                )
                audit: dict[str, Any] = {}
                for side in ("high", "low"):
                    candidate = side_candidates.get(side)
                    audit[side] = (
                        None
                        if candidate is None
                        else {
                            "rrf_score": candidate["rrf_score"],
                            "dense_best_score": candidate["dense_best_score"],
                            "bm25_best_score": candidate["bm25_best_score"],
                            "dense_match": candidate["dense_match"],
                            "bm25_match": candidate["bm25_match"],
                            "route_hits": candidate["route_hits"],
                        }
                    )
                records.append(
                    {
                        "schema_version": "1.0",
                        "sample_id": sample_id,
                        "business_id": business_id,
                        "business_name": business["name"],
                        "business_rating": business["rating"],
                        "business_total_review_count": business["actual_review_count"],
                        "aspect_id": aspect_id,
                        "aspect_name_zh": aspect_templates[aspect_id]["name_zh"],
                        "review_id": review_id,
                        "user_id": fact.user_id,
                        "review_time": fact.review_time.isoformat(),
                        "stars": fact.stars,
                        "useful": fact.useful,
                        "full_review_text": full_texts[review_id],
                        "model_review_text": model_text,
                        "matched_segments": sorted(
                            segments,
                            key=lambda item: (
                                -float(item["support_score"]),
                                item["segment_index"],
                            ),
                        )[:5],
                        "retrieved_sides": sorted(merged["sides"]),
                        "retrieval_audit": audit,
                        # 服务器直接使用messages，不会重新解释或改写模型输入。
                        "messages": _messages(
                            template=template,
                            aspect=aspect_templates[aspect_id],
                            review_text=model_text,
                        ),
                        "input_char_count": len(model_text),
                    }
                )
    records.sort(
        key=lambda item: (
            next(
                row["rank_by_selection_rule"]
                for row in businesses
                if row["business_id"] == item["business_id"]
            ),
            [row["id"] for row in definitions["aspects"]].index(item["aspect_id"]),
            item["review_id"],
        )
    )
    path = output_dir / "model_inputs.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return path, records, counts


def _copy_server_bundle(output_dir: Path) -> None:
    source_script = Path(__file__).with_name("server_judge.py")
    source_readme = Path(__file__).with_name("SERVER_README.md")
    if not source_script.is_file() or not source_readme.is_file():
        raise FileNotFoundError("server judge bundle source is incomplete")
    shutil.copy2(source_script, output_dir / "run_server_judge.py")
    shutil.copy2(source_readme, output_dir / "SERVER_README.md")


def prepare(args: argparse.Namespace) -> Path:
    from yelp_agent.recommendation_v2.review_evidence.full_reviews import (
        FullReviewStore,
    )
    from yelp_agent.recommendation_v2.review_evidence.qdrant_store import (
        QdrantReviewSegmentStore,
    )

    total_started = perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions, template = _load_and_validate_definitions()
    definition_bytes = _definition_path().read_bytes()
    definition_sha256 = _sha256_bytes(definition_bytes)
    facts_path = (
        _project_root()
        / "src"
        / "yelp_agent"
        / "recommendation_v2"
        / "data"
        / "business_facts"
        / "v1"
        / "business_facts.parquet"
    )
    reviews_path = args.source_project / "data" / "processed" / "reviews.parquet"

    selection_started = perf_counter()
    businesses = _select_steakhouses(
        facts_path=facts_path,
        reviews_path=reviews_path,
        count=args.business_count,
        minimum_reviews=args.minimum_reviews,
    )
    selection_ms = (perf_counter() - selection_started) * 1000
    selection = {
        "schema_version": "1.0",
        "selection_rule": {
            "required_yelp_category": "Steakhouses",
            "minimum_actual_review_count": args.minimum_reviews,
            "order": ["rating desc", "catalog review_count desc", "business_id"],
            "limit": args.business_count,
        },
        "businesses": businesses,
    }
    _write_json(output_dir / "selection.json", selection)
    shutil.copy2(_definition_path(), output_dir / "fixed_aspects.v1.json")
    shutil.copy2(_teacher_template_path(), output_dir / "model_contract.v1.json")

    dense_descriptors = _dense_query_descriptors(definitions)
    keyword_descriptors = _keyword_query_descriptors(definitions)
    vectors, vector_manifest = _load_or_encode_query_vectors(
        output_dir=output_dir,
        descriptors=dense_descriptors,
        definition_sha256=definition_sha256,
        embedding_model=args.embedding_model,
        embedding_python=args.embedding_python,
    )

    store = QdrantReviewSegmentStore.from_url(args.qdrant_url)
    try:
        selected, search_metrics = _search_and_select(
            store=store,
            business_ids=[item["business_id"] for item in businesses],
            dense_descriptors=dense_descriptors,
            dense_vectors=vectors,
            keyword_descriptors=keyword_descriptors,
            route_segment_limit=args.route_segment_limit,
            per_side_limit=args.per_side_limit,
            search_concurrency=args.search_concurrency,
        )
    finally:
        store.close()

    materialize_started = perf_counter()
    full_review_store = FullReviewStore(reviews_path)
    try:
        input_path, records, counts = _materialize_inputs(
            output_dir=output_dir,
            businesses=businesses,
            definitions=definitions,
            template=template,
            selected=selected,
            full_review_store=full_review_store,
            per_side_limit=args.per_side_limit,
        )
    finally:
        full_review_store.close()
    materialize_ms = (perf_counter() - materialize_started) * 1000
    _copy_server_bundle(output_dir)

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "steakhouses_top10_fixed_14_aspect_offline_profiles",
        "reference_time": datetime.now(UTC).isoformat(),
        "business_count": len(businesses),
        "aspect_count": len(definitions["aspects"]),
        "candidate_relation_count": len(records),
        "unique_review_count": len({item["review_id"] for item in records}),
        "retrieval_policy": {
            "per_side_limit": args.per_side_limit,
            "route_segment_limit": args.route_segment_limit,
            "routes": ["fixed_keyword_bm25", "fixed_semantic_embedding"],
            "fusion": f"review-level RRF, k={RRF_K}",
            "expansion": "disabled for fixed 14 aspects",
            "direction_decision": "deferred entirely to fine-tuned Qwen",
        },
        "qdrant_url": args.qdrant_url,
        "query_vector_manifest": vector_manifest,
        "counts_by_business_and_aspect": counts,
        "timing_ms": {
            "business_selection": selection_ms,
            "query_vector_preparation": vector_manifest["wall_latency_ms"],
            **search_metrics,
            "input_materialization": materialize_ms,
            "total": (perf_counter() - total_started) * 1000,
        },
        "files": {
            "selection": "selection.json",
            "fixed_definitions": "fixed_aspects.v1.json",
            "model_contract": "model_contract.v1.json",
            "model_inputs": input_path.name,
            "query_vectors": "fixed_query_vectors.npy",
            "server_runner": "run_server_judge.py",
            "server_readme": "SERVER_README.md",
        },
        "sha256": {
            "fixed_definitions": definition_sha256,
            "model_inputs": _sha256_file(input_path),
        },
    }
    manifest_path = output_dir / "prepare_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[PREPARE] businesses={len(businesses)} aspects=14")
    print(
        f"[PREPARE] relations={len(records)} unique_reviews="
        f"{manifest['unique_review_count']}"
    )
    print(f"[PREPARE] search_ms={search_metrics['search_wall_latency_ms']:.1f}")
    print(f"[PREPARE] total_ms={manifest['timing_ms']['total']:.1f}")
    print(f"[PREPARE] output={output_dir}")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
