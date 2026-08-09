"""Step 19 readiness baseline measured on the hidden Step 20 answer key."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import Field

from yelp_agent.decision_readiness import DecisionReadinessAnalyzer
from yelp_agent.models import StrictModel
from yelp_agent.query import QueryParseInput, build_rule_based_request_parser

from .artifacts import load_scenario_ground_truth, load_visible_scenarios


class DecisionReadinessFailure(StrictModel):
    scenario_id: str
    split: str
    category: str
    expected_task_type: str
    actual_task_type: str
    expected_gaps: list[str]
    actual_gaps: list[str]


class DecisionReadinessBenchmarkReport(StrictModel):
    analyzer_version: str
    parser_version: str
    scenario_count: int = Field(ge=1)
    split_counts: dict[str, int]
    task_type_accuracy: float = Field(ge=0, le=1)
    information_gap_exact_match_rate: float = Field(ge=0, le=1)
    information_gap_precision: float = Field(ge=0, le=1)
    information_gap_recall: float = Field(ge=0, le=1)
    information_gap_f1: float = Field(ge=0, le=1)
    query_aware_confidence_refusal_rate: float = Field(ge=0, le=1)
    category_task_type_accuracy: dict[str, float]
    failures: list[DecisionReadinessFailure]


def evaluate_decision_readiness_benchmark(
    visible_path: str | Path,
    ground_truth_path: str | Path,
) -> DecisionReadinessBenchmarkReport:
    """Evaluate observable Step 19 state without exposing answers to the analyzer."""

    visible = load_visible_scenarios(visible_path)
    hidden = {
        item.scenario_id: item
        for item in load_scenario_ground_truth(ground_truth_path)
    }
    if set(item.scenario_id for item in visible) != set(hidden):
        raise ValueError("visible scenarios and hidden answer key do not align")
    parser = build_rule_based_request_parser()
    analyzer = DecisionReadinessAnalyzer()
    task_matches = 0
    gap_exact = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    query_expected = 0
    query_refused = 0
    category_total: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    failures: list[DecisionReadinessFailure] = []
    for scenario in visible:
        truth = hidden[scenario.scenario_id]
        split_counts[scenario.split] += 1
        parse_input = QueryParseInput(
            user_id=scenario.user_id,
            session_id=scenario.session_id,
            cutoff_time=scenario.cutoff_time,
            query_text=scenario.query_text,
            user_latitude=scenario.user_latitude,
            user_longitude=scenario.user_longitude,
            referenced_business_ids=scenario.referenced_business_ids,
        )
        request = parser.parse(parse_input)
        ranking_source = (
            "query_aware"
            if truth.task_type == "recommendation_request"
            else "none"
        )
        readiness = analyzer.analyze(request, ranking_source=ranking_source)
        task_match = readiness.task_type == truth.task_type
        task_matches += int(task_match)
        category_total[truth.scenario_category] += 1
        category_correct[truth.scenario_category] += int(task_match)
        expected_gaps = set(truth.expected_information_gaps)
        actual_gaps = set(readiness.information_gaps)
        intersection = expected_gaps.intersection(actual_gaps)
        true_positive += len(intersection)
        false_positive += len(actual_gaps.difference(expected_gaps))
        false_negative += len(expected_gaps.difference(actual_gaps))
        gaps_match = expected_gaps == actual_gaps
        gap_exact += int(gaps_match)
        if truth.task_type == "recommendation_request":
            query_expected += 1
            query_refused += int(
                readiness.confidence_unavailable_reason
                == "query_aware_labels_unavailable"
            )
        if not task_match or not gaps_match:
            failures.append(
                DecisionReadinessFailure(
                    scenario_id=scenario.scenario_id,
                    split=scenario.split,
                    category=truth.scenario_category,
                    expected_task_type=truth.task_type,
                    actual_task_type=readiness.task_type,
                    expected_gaps=sorted(expected_gaps),
                    actual_gaps=sorted(actual_gaps),
                )
            )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 1.0
    recall = true_positive / recall_denominator if recall_denominator else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    count = len(visible)
    return DecisionReadinessBenchmarkReport(
        analyzer_version=analyzer.version,
        parser_version=parser.version,
        scenario_count=count,
        split_counts=dict(sorted(split_counts.items())),
        task_type_accuracy=task_matches / count,
        information_gap_exact_match_rate=gap_exact / count,
        information_gap_precision=precision,
        information_gap_recall=recall,
        information_gap_f1=f1,
        query_aware_confidence_refusal_rate=(
            query_refused / query_expected if query_expected else 1.0
        ),
        category_task_type_accuracy={
            category: category_correct[category] / total
            for category, total in sorted(category_total.items())
        },
        failures=failures,
    )


def write_decision_readiness_benchmark_report(
    report: DecisionReadinessBenchmarkReport,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        partial.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return output
