"""Run one real-data, label-free Step 18 static recommendation example."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb

from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.config import (
    load_business_profile_config,
    load_query_aware_config,
)
from yelp_agent.query import (
    QueryAwareRecommender,
    QueryAwareStaticRanker,
    QueryParseInput,
    build_rule_based_request_parser,
    candidate_from_business_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--query",
        default="想吃牛排，最好安静一点，适合约会。",
    )
    parser.add_argument("--task-id")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/features/hybrid_v2_b/validation_predictions.parquet"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("data/task_dataset/tasks/temporal_contexts.parquet"),
    )
    parser.add_argument(
        "--business-knowledge",
        type=Path,
        default=Path("data/features/business_profiles/v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/query_aware_v1/real_data_demo.json"),
    )
    return parser.parse_args()


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _select_task(
    connection: duckdb.DuckDBPyConnection,
    *,
    predictions: Path,
    contexts: Path,
    businesses: Path,
    desired_categories: list[str],
    explicit_task_id: str | None,
) -> str:
    if explicit_task_id:
        count = connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE task_id = ?",
            [str(predictions), explicit_task_id],
        ).fetchone()[0]
        if count == 0:
            raise ValueError(f"Unknown validation task_id: {explicit_task_id}")
        return explicit_task_id
    if desired_categories:
        row = connection.execute(
            """
            SELECT p.task_id
            FROM read_parquet(?) p
            JOIN read_parquet(?) b USING (business_id)
            JOIN read_parquet(?) c USING (task_id)
            WHERE c.split = 'validation'
              AND list_has_any(b.categories, ?)
            GROUP BY p.task_id, c.cutoff_time
            HAVING count(*) >= 5
            ORDER BY c.cutoff_time DESC, p.task_id
            LIMIT 1
            """,
            [str(predictions), str(businesses), str(contexts), desired_categories],
        ).fetchone()
    else:
        row = connection.execute(
            """
            SELECT p.task_id
            FROM read_parquet(?) p
            JOIN read_parquet(?) c USING (task_id)
            WHERE c.split = 'validation'
            GROUP BY p.task_id, c.cutoff_time
            ORDER BY c.cutoff_time DESC, p.task_id
            LIMIT 1
            """,
            [str(predictions), str(contexts)],
        ).fetchone()
    if row is None:
        raise RuntimeError("No validation task can demonstrate this query")
    return str(row[0])


def _top_rows(
    result, names: dict[str, str], *, limit: int = 10
) -> list[dict[str, object]]:
    return [
        {
            "business_id": row.business_id,
            "name": names[row.business_id],
            "rank": row.rank,
            "hybrid_rank": row.hybrid_rank,
            "query_rank": row.query_rank,
            "query_score": row.query_score,
            "fusion_score": row.fusion_score,
            "constraint_status": row.constraint_status,
            "matched_fields": row.matched_fields,
            "unmatched_fields": row.unmatched_fields,
            "unknown_fields": row.unknown_fields,
        }
        for row in result.ranking[:limit]
    ]


def main() -> None:
    args = parse_args()
    predictions = _required(args.predictions, "Validation predictions")
    contexts = _required(args.contexts, "Temporal contexts")
    business_rows = _required(
        args.business_knowledge / "businesses.parquet",
        "Business knowledge records",
    )
    query_config = load_query_aware_config(args.config_dir)
    parser = build_rule_based_request_parser()
    if parser.version != query_config.rule_parser_version:
        raise RuntimeError("Query parser version disagrees with frozen configuration")

    provisional = parser.parse(
        QueryParseInput(
            user_id="task-selected-after-parse",
            session_id="step18-real-data-demo",
            cutoff_time=datetime(2024, 1, 1),
            query_text=args.query,
        )
    )
    connection = duckdb.connect()
    try:
        task_id = _select_task(
            connection,
            predictions=predictions,
            contexts=contexts,
            businesses=business_rows,
            desired_categories=provisional.desired_categories,
            explicit_task_id=args.task_id,
        )
        context_row = connection.execute(
            """
            SELECT user_id, cutoff_time
            FROM read_parquet(?)
            WHERE task_id = ? AND split = 'validation'
            """,
            [str(contexts), task_id],
        ).fetchone()
        if context_row is None:
            raise RuntimeError("Selected task has no validation context")
        prediction_rows = connection.execute(
            """
            SELECT business_id, rank
            FROM read_parquet(?)
            WHERE task_id = ?
            ORDER BY rank, business_id
            """,
            [str(predictions), task_id],
        ).fetchall()
    finally:
        connection.close()

    user_id, cutoff_time = str(context_row[0]), context_row[1]
    business_ids = [str(row[0]) for row in prediction_rows]
    store = BusinessKnowledgeStore.from_artifacts(
        args.business_knowledge,
        config=load_business_profile_config(args.config_dir),
    )
    profiles = store.get(business_ids, cutoff_time)
    candidates = tuple(
        candidate_from_business_profile(
            profiles[str(business_id)],
            hybrid_rank=int(rank),
        )
        for business_id, rank in prediction_rows
    )
    recommender = QueryAwareRecommender(
        parser=parser,
        ranker=QueryAwareStaticRanker(
            query_rrf_weight=query_config.query_rrf_weight,
            rrf_constant=query_config.rrf_constant,
        ),
    )
    comparison = recommender.recommend(
        QueryParseInput(
            user_id=user_id,
            session_id="step18-real-data-demo",
            cutoff_time=cutoff_time,
            query_text=args.query,
        ),
        candidates,
    )
    names = {business_id: profile.name for business_id, profile in profiles.items()}
    payload = {
        "evaluation_status": "label_free_real_data_demo",
        "performance_claim_allowed": False,
        "ground_truth_loaded": False,
        "legacy_test_loaded": False,
        "task_id": task_id,
        "request": comparison.request.model_dump(mode="json"),
        "candidate_count": len(candidates),
        "excluded_candidate_count": len(comparison.hybrid_query.excluded),
        "hybrid_v2_top_10": [
            {
                "business_id": business_id,
                "name": names[business_id],
                "rank": rank,
            }
            for rank, business_id in enumerate(
                comparison.hybrid_v2_ranking[:10],
                start=1,
            )
        ],
        "query_only_top_10": _top_rows(comparison.query_only, names),
        "hybrid_query_top_10": _top_rows(comparison.hybrid_query, names),
        "warnings": comparison.hybrid_query.warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
