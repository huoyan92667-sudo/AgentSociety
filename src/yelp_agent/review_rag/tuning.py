"""Development-only policy selection for Review RAG V1."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from yelp_agent.agent_benchmark import (
    EvidenceLabel,
    ScenarioGroundTruth,
    VisibleAgentScenario,
    load_evidence_labels,
    load_scenario_ground_truth,
    load_visible_scenarios,
)
from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.query import QueryParseInput
from yelp_agent.reviews.schema import ASPECT_NAMES

from .config import ReviewRAGPolicy
from .query import infer_review_aspects
from .retriever import ReviewRetriever
from .schema import ReviewSearchRequest, ReviewSearchResult


@dataclass(frozen=True, slots=True)
class ReviewRAGTuningResult:
    selected_policy: ReviewRAGPolicy
    report_path: Path
    traces_path: Path
    policy_path: Path


def tune_review_rag_policy(
    *,
    benchmark_root: str | Path,
    output_root: str | Path,
    policy_path: str | Path,
    retriever_factory: Callable[[ReviewRAGPolicy], ReviewRetriever],
    policies: Sequence[ReviewRAGPolicy] | None = None,
    progress: Callable[[int, int, VisibleAgentScenario], None] | None = None,
) -> ReviewRAGTuningResult:
    """Select solely on Development; Validation rows are never loaded into scoring."""

    root = Path(benchmark_root)
    visible = tuple(
        item
        for item in load_visible_scenarios(root / "visible" / "scenarios.jsonl")
        if item.split == "development"
    )
    visible_ids = {item.scenario_id for item in visible}
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(root / "hidden" / "ground_truth.jsonl")
        if item.scenario_id in visible_ids
        and "retrieve_business_reviews" in item.required_actions
    }
    labels = [
        item
        for item in load_evidence_labels(root / "hidden" / "evidence_labels.parquet")
        if item.scenario_id in truth
    ]
    scenarios = tuple(item for item in visible if item.scenario_id in truth)
    if len(scenarios) != 104 or any(item.split != "development" for item in scenarios):
        raise ValueError("Step 27 tuning requires the frozen 104 Development RAG scenes")
    candidates = tuple(policies or default_policy_candidates())
    if not candidates or any(item.selection_split != "development" for item in candidates):
        raise ValueError("Review RAG policy candidates must be Development-only")
    relevant_by_scenario: dict[str, set[str]] = {}
    for label in labels:
        if label.source_type == "review" and label.relevance == "relevant":
            relevant_by_scenario.setdefault(label.scenario_id, set()).add(
                str(label.review_id)
            )
    interpreter = RuleBasedRequestInterpreter()
    traces: list[dict[str, object]] = []
    metrics_by_policy: dict[str, dict[str, float | int]] = {}
    for policy_index, policy in enumerate(candidates):
        retriever = retriever_factory(policy)
        rows: list[dict[str, object]] = []
        actual_tokens = 0
        cache_misses = 0
        for index, scenario in enumerate(scenarios, 1):
            result = retriever.search(_request(scenario, interpreter))
            actual_tokens += result.embedding_usage.input_tokens
            cache_misses += result.embedding_usage.cache_misses
            row = _score_result(
                scenario,
                truth[scenario.scenario_id],
                relevant_by_scenario[scenario.scenario_id],
                result,
            )
            row["policy_version"] = policy.policy_version
            rows.append(row)
            traces.append(row)
            if progress is not None and policy_index == 0:
                progress(index, len(scenarios), scenario)
        metrics = _aggregate(rows)
        metrics["actual_local_input_tokens"] = actual_tokens
        metrics["embedding_cache_misses"] = cache_misses
        metrics_by_policy[policy.policy_version] = metrics
    selected = max(
        candidates,
        key=lambda item: (
            float(metrics_by_policy[item.policy_version]["avg_review_recall"]),
            float(metrics_by_policy[item.policy_version]["precision_at_5"]),
            float(metrics_by_policy[item.policy_version]["recall_at_1"]),
            -item.embedding_weight,
            item.policy_version,
        ),
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    traces_path = _write_jsonl(output / "development_retrieval_runs.jsonl", traces)
    report = {
        "schema_version": 1,
        "selection_split": "development",
        "scenario_count": len(scenarios),
        "validation_rows_loaded_for_scoring": 0,
        "silver_label_warning": (
            "Review relevance comes from Step 13 high-confidence rule extraction; "
            "it is not human gold relevance."
        ),
        "objective": "AvgReviewRecall; tie Precision@5, Recall@1, lower embedding weight",
        "selected_policy": selected.model_dump(mode="json"),
        "candidates": [
            {
                "policy": item.model_dump(mode="json"),
                "metrics": metrics_by_policy[item.policy_version],
            }
            for item in candidates
        ],
        "external_api_calls": 0,
        "billed_tokens": 0,
        "cost_cny": 0.0,
    }
    report_path = output / "development_tuning.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    frozen_path = Path(policy_path)
    frozen_path.write_text(
        json.dumps(selected.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ReviewRAGTuningResult(
        selected_policy=selected,
        report_path=report_path,
        traces_path=traces_path,
        policy_path=frozen_path,
    )


def default_policy_candidates() -> tuple[ReviewRAGPolicy, ...]:
    values = (
        ("dev-bm25-v1", 60, 0.0, 1.0, 0.0),
        ("dev-aspect-bm25-v1", 60, 1.0, 1.0, 0.0),
        ("dev-balanced-v1", 60, 1.0, 1.0, 1.0),
        ("dev-embedding-heavy-v1", 60, 0.75, 0.75, 1.5),
        ("dev-aspect-heavy-v1", 60, 1.5, 0.75, 1.0),
        ("dev-balanced-small-k-v1", 20, 1.0, 1.0, 1.0),
    )
    return tuple(
        ReviewRAGPolicy(
            policy_version=name,
            rrf_k=rrf_k,
            aspect_weight=aspect,
            bm25_weight=bm25,
            embedding_weight=embedding,
        )
        for name, rrf_k, aspect, bm25, embedding in values
    )


def _request(
    scenario: VisibleAgentScenario,
    interpreter: RuleBasedRequestInterpreter,
) -> ReviewSearchRequest:
    interpreted = interpreter.interpret(
        QueryParseInput(
            user_id=scenario.user_id,
            session_id=scenario.session_id,
            cutoff_time=scenario.cutoff_time,
            query_text=scenario.query_text,
            user_latitude=scenario.user_latitude,
            user_longitude=scenario.user_longitude,
            referenced_business_ids=scenario.referenced_business_ids,
        )
    )
    aspects = [
        condition.field
        for condition in interpreted.request.conditions
        if condition.field in ASPECT_NAMES
    ]
    for aspect in infer_review_aspects(scenario.query_text):
        if aspect not in aspects:
            aspects.append(aspect)
    return ReviewSearchRequest(
        query_text=scenario.query_text,
        business_ids=scenario.referenced_business_ids,
        cutoff_time=scenario.cutoff_time,
        aspects=aspects,
        top_k=5,
        usage_scope=scenario.scenario_id,
    )


def _score_result(
    scenario: VisibleAgentScenario,
    truth: ScenarioGroundTruth,
    relevant: set[str],
    result: ReviewSearchResult,
) -> dict[str, object]:
    hit_ids = [item.review_id for item in result.hits]
    values: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "split": scenario.split,
        "category": truth.scenario_category,
        "retrieved_review_ids": hit_ids,
        "relevant_review_count": len(relevant),
        "business_scope_isolation": float(
            all(item.business_id in truth.business_scope for item in result.hits)
        ),
        "input_tokens": result.embedding_usage.input_tokens,
        "cache_misses": result.embedding_usage.cache_misses,
    }
    for cutoff in (1, 3, 5):
        prefix = hit_ids[:cutoff]
        found = len(set(prefix) & relevant)
        values[f"recall_at_{cutoff}"] = found / len(relevant)
        values[f"precision_at_{cutoff}"] = found / len(prefix) if prefix else 0.0
    values["avg_review_recall"] = sum(
        float(values[f"recall_at_{cutoff}"]) for cutoff in (1, 3, 5)
    ) / 3.0
    return values


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    names = [
        *(f"recall_at_{cutoff}" for cutoff in (1, 3, 5)),
        *(f"precision_at_{cutoff}" for cutoff in (1, 3, 5)),
        "avg_review_recall",
        "business_scope_isolation",
    ]
    return {
        "scenario_count": len(rows),
        **{
            name: sum(float(row[name]) for row in rows) / len(rows)
            for name in names
        },
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path
