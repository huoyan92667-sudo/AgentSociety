from scripts.compare_rule_agent_reranking import build_report


def _run(ranking: list[str], *, semantic: bool = False) -> dict[str, object]:
    calls = []
    if semantic:
        calls.append(
            {
                "tool_name": "COMPUTE_EMBEDDING_MATCH",
                "status": "completed",
                "tool_kind": "semantic",
                "input_tokens": 12,
            }
        )
    return {
        "scenario_id": "a" * 64,
        "fallback": False,
        "turns": [{"candidate_ranking": ranking, "tool_calls": calls}],
    }


def test_comparison_reports_semantic_rank_improvement() -> None:
    report = build_report(
        baseline_runs={"a" * 64: _run(["a", "b", "c"])},
        semantic_runs={"a" * 64: _run(["b", "a", "c"], semantic=True)},
        baseline_metrics={"metrics": {"hr_at_1": {"value": 0.0}}},
        semantic_metrics={"metrics": {"hr_at_1": {"value": 1.0}}},
        visible={
            "a" * 64: {
                "split": "development",
                "query_text": "find a business",
            }
        },
        truth={"a" * 64: {"acceptable_business_ids": ["b"]}},
    )

    ranking = report["ranking_metrics"]
    assert ranking["semantic_call_count"] == 1
    assert ranking["improved_count"] == 1
    assert ranking["hybrid_hit_at_1"] == 0.0
    assert ranking["semantic_hit_at_1"] == 1.0
    assert report["metric_deltas_step25_minus_step24"]["hr_at_1"]["delta"] == 1.0
