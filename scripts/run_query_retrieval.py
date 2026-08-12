"""Run one label-free full-catalog Query Recall request.

This command deliberately does not load a recommendation target or Query
Recommendation Benchmark.  Aspect evidence comes from the frozen
``selected_user_interactions`` business-knowledge artifact only.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from yelp_agent.business_profiles import BusinessKnowledgeStore
from yelp_agent.config import load_business_profile_config
from yelp_agent.data.temporal_view import TemporalDataView
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser
from yelp_agent.query_retrieval import (
    QueryCandidateRetriever,
    QueryRetrievalTask,
    load_query_retrieval_config,
)
from yelp_agent.semantic_embedding import (
    CachedEmbeddingGateway,
    DashScopeEmbeddingEncoder,
    LocalEmbeddingEnvironment,
    LocalQwenEmbeddingEncoder,
    SemanticEmbeddingMatcher,
    SqliteEmbeddingCache,
    load_dashscope_embedding_environment,
    load_local_embedding_environment,
    load_semantic_embedding_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", default="query-recall-user")
    parser.add_argument("--session-id", default="query-recall-session")
    parser.add_argument("--cutoff", default="2022-01-01T00:00:00")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="project containing data/processed and frozen business profiles",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path.cwd(),
        help="project containing configs/query_retrieval.yaml",
    )
    parser.add_argument(
        "--embedding-config",
        type=Path,
        help="enable Query Embedding with this Step 25 config",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="override the persistent embedding cache directory",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/query_retrieval/single_query.json"),
    )
    return parser


def _parse_cutoff(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--cutoff must be ISO-8601") from exc
    if parsed.tzinfo is not None:
        raise argparse.ArgumentTypeError("--cutoff must be timezone-naive")
    return parsed


def _build_embedding_matcher(
    *,
    args: argparse.Namespace,
    source_root: Path,
    data_view: TemporalDataView,
) -> tuple[SemanticEmbeddingMatcher | None, object | None]:
    if args.embedding_config is None:
        return None, None
    config = load_semantic_embedding_config(args.embedding_config)
    if config.provider == "local":
        detected = load_local_embedding_environment(os.environ)
        environment = LocalEmbeddingEnvironment(
            model_path=args.model_path or detected.model_path,
            python_executable=args.model_python or detected.python_executable,
            device=args.device or detected.device,
        )
        encoder = LocalQwenEmbeddingEncoder.from_environment(config, environment)
    else:
        encoder = DashScopeEmbeddingEncoder.from_environment(
            config,
            load_dashscope_embedding_environment(os.environ),
        )
    cache_root = args.cache_root or source_root / config.cache_relative_path
    matcher = SemanticEmbeddingMatcher(
        businesses=data_view,
        gateway=CachedEmbeddingGateway(
            encoder=encoder,
            cache=SqliteEmbeddingCache(cache_root),
            config=config,
        ),
        config=config,
    )
    return matcher, encoder


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.latitude is None) != (args.longitude is None):
        raise SystemExit("--latitude and --longitude must be supplied together")
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    try:
        cutoff = _parse_cutoff(args.cutoff)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    source_root = args.source_root.resolve()
    config_root = args.config_root.resolve()
    data_view = TemporalDataView(
        source_root / "data" / "processed" / "businesses.parquet",
        source_root / "data" / "processed" / "reviews.parquet",
        source_root / "data" / "processed" / "interactions.parquet",
    )
    profiles = BusinessKnowledgeStore.from_artifacts(
        source_root / "data" / "features" / "business_profiles" / "v1",
        config=load_business_profile_config(source_root / "configs"),
    )
    parser = build_rule_based_request_parser()
    request = parser.parse(
        QueryParseInput(
            user_id=args.user_id,
            session_id=args.session_id,
            cutoff_time=cutoff,
            query_text=args.query,
            user_latitude=args.latitude,
            user_longitude=args.longitude,
        )
    )

    encoder = None
    try:
        embedding_matcher, encoder = _build_embedding_matcher(
            args=args,
            source_root=source_root,
            data_view=data_view,
        )
        result = QueryCandidateRetriever(
            catalog=data_view,
            profiles=profiles,
            embedding_matcher=embedding_matcher,
            config=load_query_retrieval_config(
                config_root / "configs" / "query_retrieval.yaml"
            ),
        ).retrieve(
            QueryRetrievalTask(
                request=request,
                usage_scope=f"query-recall:{request.request_id}",
            )
        )
    finally:
        if encoder is not None and hasattr(encoder, "close"):
            encoder.close()

    top_rows = []
    for item in result.candidates[: args.top_k]:
        business = data_view.business(item.business_id)
        top_rows.append(
            {
                **item.model_dump(mode="json"),
                "name": business.name,
                "categories": list(business.categories),
                "address": business.address,
            }
        )
    payload: dict[str, object] = {
        "evaluation_status": "label_free_query_recall_run",
        "benchmark_loaded": False,
        "ground_truth_loaded": False,
        "performance_claim_allowed": False,
        "aspect_source_scope": profiles.source_scope,
        "aspect_scope_note": (
            "Aspect evidence is restricted to comments from selected user "
            "interactions; missing evidence remains unknown."
        ),
        "request": request.model_dump(mode="json"),
        "retrieval": result.model_dump(mode="json"),
        "top_candidates": top_rows,
    }
    _write_report(args.output, payload)
    summary = {
        "request_id": request.request_id,
        "candidate_count": len(result.candidates),
        "eligible_business_count": result.eligible_business_count,
        "route_result_counts": result.route_result_counts,
        "warnings": result.warnings,
        "usage": result.usage.model_dump(),
        "latency_ms": result.latency_ms,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
