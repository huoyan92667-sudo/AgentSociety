"""Run History, Query, and History+Query retrieval on the frozen benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from yelp_agent.query_recommendation_benchmark import (
    BenchmarkRetrievalRun,
    QueryRecommendationGroundTruth,
    QueryRecommendationRetrievalRuntime,
    QueryRecommendationRetrievalSources,
    VisibleQueryRecommendationCase,
    evaluate_query_recommendation_retrieval,
)
from yelp_agent.semantic_embedding import (
    LocalEmbeddingEnvironment,
    load_local_embedding_environment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/query_recommendation_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/query_recommendation_v1/retrieval"),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--limit", type=int)
    return parser


def _load_jsonl(path: Path, model):
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(path)


def _write_jsonl(path: Path, values: Sequence[BenchmarkRetrievalRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        "".join(item.model_dump_json() + "\n" for item in values),
        encoding="utf-8",
        newline="\n",
    )
    partial.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    source_root = args.source_root.resolve()
    config_root = args.config_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    load_dotenv(source_root / ".env", override=False)

    # The runtime receives visible cases only. Hidden labels are deliberately
    # loaded after every retrieval result has been frozen on disk.
    visible = _load_jsonl(
        benchmark_root / "visible" / "cases.jsonl",
        VisibleQueryRecommendationCase,
    )
    if args.limit is not None:
        visible = visible[: args.limit]
    detected = load_local_embedding_environment(os.environ)
    embedding_environment = LocalEmbeddingEnvironment(
        model_path=args.model_path or detected.model_path,
        python_executable=args.model_python or detected.python_executable,
        device=args.device or detected.device,
    )
    if not embedding_environment.enabled:
        raise SystemExit(
            "Set LOCAL_EMBEDDING_MODEL_PATH or pass --model-path for the local encoder"
        )

    cache_root = output_root / "case_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    all_runs: list[BenchmarkRetrievalRun] = []
    sources = QueryRecommendationRetrievalSources(
        project_root=source_root,
        config_root=config_root,
    )
    with QueryRecommendationRetrievalRuntime.from_sources(
        sources,
        embedding_environment=embedding_environment,
    ) as runtime:
        for index, case in enumerate(visible, start=1):
            cache_path = cache_root / f"{case.case_id}.jsonl"
            if cache_path.is_file():
                case_runs = _load_jsonl(cache_path, BenchmarkRetrievalRun)
            else:
                case_runs = runtime.run_case(case)
                _write_jsonl(cache_path, case_runs)
            if {item.method for item in case_runs} != {
                "history_only",
                "query_only",
                "history_query",
            }:
                raise ValueError(f"incomplete retrieval cache for {case.case_id}")
            all_runs.extend(case_runs)
            if index == 1 or index % 10 == 0 or index == len(visible):
                print(
                    json.dumps(
                        {"stage": "retrieval", "completed": index, "total": len(visible)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    _write_jsonl(output_root / "retrieval_runs.jsonl", all_runs)
    all_truth = _load_jsonl(
        benchmark_root / "hidden" / "ground_truth.jsonl",
        QueryRecommendationGroundTruth,
    )
    selected_ids = {item.case_id for item in visible}
    truth = tuple(item for item in all_truth if item.case_id in selected_ids)
    report = evaluate_query_recommendation_retrieval(visible, truth, all_runs)
    _write_json(output_root / "metrics.json", report)
    print(
        json.dumps(
            {
                "stage": "complete",
                "case_count": len(visible),
                "methods": list(report.overall),
                "recall_at_500": {
                    method: metrics.recall_at_500
                    for method, metrics in report.overall.items()
                },
                "query_rescue_rate": report.query_rescue_rate,
                "fusion_loss_rate": report.fusion_loss_rate,
                "embedding_encoded_tokens": (
                    report.execution_embedding_encoded_tokens
                ),
                "embedding_logical_tokens": (
                    report.execution_embedding_logical_tokens
                ),
                "embedding_provider_calls": (
                    report.execution_embedding_provider_calls
                ),
                "external_llm_tokens": report.external_llm_tokens,
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
