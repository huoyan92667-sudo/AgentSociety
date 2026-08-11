"""Development-only policy selection for deterministic evidence aggregation."""

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
from yelp_agent.review_rag import ReviewRetriever
from yelp_agent.review_rag.tuning import _request as _review_search_request

from .aggregator import EvidenceAggregator
from .config import EvidenceAggregationPolicy
from .query import aggregation_query_facts
from .schema import EvidenceAggregationRequest, EvidenceAssessment


@dataclass(frozen=True, slots=True)
class EvidenceAggregationTuningResult:
    selected_policy: EvidenceAggregationPolicy
    report_path: Path
    traces_path: Path
    policy_path: Path


@dataclass(frozen=True, slots=True)
class _Case:
    scenario: VisibleAgentScenario
    truth: ScenarioGroundTruth
    labels: tuple[EvidenceLabel, ...]
    request: EvidenceAggregationRequest


def tune_evidence_aggregation_policy(
    *,
    benchmark_root: str | Path,
    output_root: str | Path,
    policy_path: str | Path,
    retriever: ReviewRetriever,
    policies: Sequence[EvidenceAggregationPolicy] | None = None,
    progress: Callable[[int, int, VisibleAgentScenario], None] | None = None,
) -> EvidenceAggregationTuningResult:
    """Tune only deterministic thresholds on the frozen Development split."""

    cases = _load_cases(
        benchmark_root=benchmark_root,
        split="development",
        retriever=retriever,
        progress=progress,
    )
    if len(cases) != 104:
        raise ValueError("Step 28 tuning requires the frozen 104 Development RAG scenes")
    candidates = tuple(policies or default_policy_candidates())
    if not candidates or any(item.selection_split != "development" for item in candidates):
        raise ValueError("evidence policies must be Development-only")
    traces: list[dict[str, object]] = []
    metrics_by_policy: dict[str, dict[str, float | int]] = {}
    for policy in candidates:
        aggregator = EvidenceAggregator(policy)
        rows = []
        for case in cases:
            assessment = aggregator.aggregate(case.request)
            row = _score(case, assessment)
            row["policy_version"] = policy.policy_version
            rows.append(row)
            traces.append(row)
        metrics_by_policy[policy.policy_version] = _aggregate(rows)
    selected = max(
        candidates,
        key=lambda item: (
            float(metrics_by_policy[item.policy_version]["selection_objective"]),
            -float(
                metrics_by_policy[item.policy_version][
                    "uncertain_case_grounded_false_positive_rate"
                ]
            ),
            float(metrics_by_policy[item.policy_version]["citation_precision"]),
            item.policy_version,
        ),
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    traces_path = _write_jsonl(output / "development_aggregation_runs.jsonl", traces)
    report = {
        "schema_version": 1,
        "selection_split": "development",
        "scenario_count": len(cases),
        "validation_rows_loaded_for_scoring": 0,
        "objective": (
            "0.4 response-policy accuracy + 0.3 conflict accuracy + "
            "0.2 stance accuracy + 0.1 citation precision; "
            "tie lower uncertain-case grounded false positives"
        ),
        "silver_label_warning": (
            "Evidence stance and relevance are Step 13 traceable silver labels, "
            "not human gold judgments."
        ),
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
    frozen = Path(policy_path)
    frozen.write_text(
        json.dumps(selected.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return EvidenceAggregationTuningResult(
        selected_policy=selected,
        report_path=report_path,
        traces_path=traces_path,
        policy_path=frozen,
    )


def default_policy_candidates() -> tuple[EvidenceAggregationPolicy, ...]:
    return (
        EvidenceAggregationPolicy(
            policy_version="dev-balanced-evidence-v1",
        ),
        EvidenceAggregationPolicy(
            policy_version="dev-sensitive-conflict-v1",
            conflict_minority_mass_share=0.15,
        ),
        EvidenceAggregationPolicy(
            policy_version="dev-lenient-conflict-v1",
            conflict_minority_mass_share=0.35,
        ),
        EvidenceAggregationPolicy(
            policy_version="dev-recent-evidence-v1",
            recency_half_life_days=365,
        ),
        EvidenceAggregationPolicy(
            policy_version="dev-long-history-v1",
            recency_half_life_days=1460,
        ),
        EvidenceAggregationPolicy(
            policy_version="dev-high-precision-atoms-v1",
            minimum_relevance=0.15,
            minimum_extraction_confidence=0.85,
        ),
        EvidenceAggregationPolicy(
            policy_version="dev-multi-user-grounding-v1",
            grounded_minimum_evidence=2,
            grounded_minimum_unique_users=2,
        ),
    )


def _load_cases(
    *,
    benchmark_root: str | Path,
    split: str,
    retriever: ReviewRetriever,
    progress: Callable[[int, int, VisibleAgentScenario], None] | None = None,
) -> tuple[_Case, ...]:
    root = Path(benchmark_root)
    visible = tuple(
        item
        for item in load_visible_scenarios(root / "visible" / "scenarios.jsonl")
        if item.split == split
    )
    visible_ids = {item.scenario_id for item in visible}
    truth = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(root / "hidden" / "ground_truth.jsonl")
        if item.scenario_id in visible_ids
        and "retrieve_business_reviews" in item.required_actions
    }
    labels_by_scenario: dict[str, list[EvidenceLabel]] = {}
    for label in load_evidence_labels(root / "hidden" / "evidence_labels.parquet"):
        if label.scenario_id in truth and label.source_type == "review":
            labels_by_scenario.setdefault(label.scenario_id, []).append(label)
    scenarios = tuple(item for item in visible if item.scenario_id in truth)
    interpreter = RuleBasedRequestInterpreter()
    cases: list[_Case] = []
    for index, scenario in enumerate(scenarios, 1):
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
        aspects, polarity, explicit = aggregation_query_facts(
            scenario.query_text,
            interpreted.request.conditions,
        )
        search = retriever.search(_review_search_request(scenario, interpreter))
        cases.append(
            _Case(
                scenario=scenario,
                truth=truth[scenario.scenario_id],
                labels=tuple(labels_by_scenario.get(scenario.scenario_id, [])),
                request=EvidenceAggregationRequest(
                    query_text=scenario.query_text,
                    task_type=interpreted.readiness.task_type,
                    requested_aspects=aspects,
                    desired_polarity_by_aspect=polarity,  # type: ignore[arg-type]
                    explicit_uncertainty_request=explicit,
                    search_result=search,
                ),
            )
        )
        if progress is not None:
            progress(index, len(scenarios), scenario)
    return tuple(cases)


def _score(case: _Case, assessment: EvidenceAssessment) -> dict[str, object]:
    relevant = {
        str(label.review_id): label.stance
        for label in case.labels
        if label.relevance == "relevant" and label.review_id is not None
    }
    aspects = [
        aspect for business in assessment.businesses for aspect in business.aspects
    ]
    predicted_stance = {
        atom.review_id: atom.stance for aspect in aspects for atom in aspect.atoms
    }
    overlapping = sorted(set(predicted_stance) & set(relevant))
    stance_accuracy = (
        sum(predicted_stance[key] == relevant[key] for key in overlapping)
        / len(overlapping)
        if overlapping
        else 0.0
    )
    citations = {
        review_id for aspect in aspects for review_id in aspect.citation_review_ids
    }
    citation_precision = (
        len(citations & set(relevant)) / len(citations) if citations else 0.0
    )
    citation_recall = (
        len(citations & set(relevant)) / len(relevant) if relevant else 0.0
    )
    predicted_conflict = any(item.has_conflict for item in assessment.businesses)
    expected_conflict = case.truth.uncertainty_policy == "report_conflict"
    predicted_mode = (
        "grounded"
        if assessment.businesses
        and all(item.overall_response_mode == "grounded" for item in assessment.businesses)
        else "uncertain"
    )
    expected_mode = (
        "uncertain"
        if case.truth.uncertainty_policy
        in {"answer_with_caveat", "report_conflict", "require_official_verification", "abstain"}
        else "grounded"
    )
    response_correct = predicted_mode == expected_mode
    conflict_correct = predicted_conflict == expected_conflict
    objective = (
        0.4 * float(response_correct)
        + 0.3 * float(conflict_correct)
        + 0.2 * stance_accuracy
        + 0.1 * citation_precision
    )
    return {
        "scenario_id": case.scenario.scenario_id,
        "split": case.scenario.split,
        "category": case.truth.scenario_category,
        "expected_response_mode": expected_mode,
        "predicted_response_mode": predicted_mode,
        "expected_conflict": expected_conflict,
        "predicted_conflict": predicted_conflict,
        "response_policy_correct": response_correct,
        "conflict_correct": conflict_correct,
        "stance_accuracy": stance_accuracy,
        "stance_overlap_count": len(overlapping),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_count": len(citations),
        "relevant_label_count": len(relevant),
        "selection_objective": objective,
        "uncertain_case_grounded_false_positive": (
            expected_mode == "uncertain" and predicted_mode == "grounded"
        ),
    }


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate empty evidence evaluation rows")
    mean_names = (
        "response_policy_correct",
        "conflict_correct",
        "stance_accuracy",
        "citation_precision",
        "citation_recall",
        "selection_objective",
    )
    uncertain = [row for row in rows if row["expected_response_mode"] == "uncertain"]
    return {
        "scenario_count": len(rows),
        **{
            name: sum(float(row[name]) for row in rows) / len(rows)
            for name in mean_names
        },
        "uncertain_scenario_count": len(uncertain),
        "uncertain_case_grounded_false_positive_rate": (
            sum(bool(row["uncertain_case_grounded_false_positive"]) for row in uncertain)
            / len(uncertain)
            if uncertain
            else 0.0
        ),
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path
