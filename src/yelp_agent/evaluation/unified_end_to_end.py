"""Transparent end-to-end metrics for Query Recommendation Agent runs.

The evaluator never invents missing labels.  Metrics that require an Evidence
Benchmark or human judgment are explicitly unavailable until those labels are
provided.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator, model_validator

from yelp_agent.models import StrictModel
from yelp_agent.recommendation_evidence.schema import RecommendationEvidenceCard


class EvidenceJudgmentLabels(StrictModel):
    support_correct: list[bool] = Field(default_factory=list)
    citation_correct: list[bool] = Field(default_factory=list)
    contradiction_expected: bool | None = None
    contradiction_detected: bool | None = None
    insufficiency_expected: bool | None = None
    insufficiency_detected: bool | None = None
    factual_claim_supported: list[bool] = Field(default_factory=list)
    freshness_compliant: list[bool] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pairs(self) -> "EvidenceJudgmentLabels":
        if (self.contradiction_expected is None) != (
            self.contradiction_detected is None
        ):
            raise ValueError("contradiction label and prediction must appear together")
        if (self.insufficiency_expected is None) != (
            self.insufficiency_detected is None
        ):
            raise ValueError("insufficiency label and prediction must appear together")
        return self


class EndToEndCaseResult(StrictModel):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_business_id: str = Field(min_length=1)
    retrieval_ranking: list[str] = Field(max_length=500)
    final_ranking: list[str]
    hard_constraint_satisfied_business_ids: list[str] | None = None
    query_compliant_business_ids: list[str] | None = None
    rejected_business_ids: list[str] = Field(default_factory=list)
    evidence_cards: list[RecommendationEvidenceCard] = Field(default_factory=list, max_length=5)
    evidence_labels: EvidenceJudgmentLabels | None = None
    agent_status: Literal["completed", "awaiting_user", "fallback"] = "completed"
    response_kind: Literal[
        "none",
        "clarification",
        "recommendation",
        "grounded_answer",
        "uncertain_answer",
        "fallback",
    ] = "recommendation"
    fallback: bool = False
    reported_evidence_recency: bool | None = None
    clarification_utility_gain: float | None = None
    tool_selection_correct: bool | None = None
    clarification_utility: float | None = None
    recovery_success: bool | None = None
    action_count: int = Field(default=0, ge=0)
    invalid_action_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @field_validator(
        "retrieval_ranking",
        "final_ranking",
        "rejected_business_ids",
    )
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("ranking and rejection IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_case(self) -> "EndToEndCaseResult":
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input and output tokens must appear together")
        if self.invalid_action_count > self.action_count:
            raise ValueError("invalid action count cannot exceed all actions")
        if self.evidence_cards and [item.business_id for item in self.evidence_cards] != self.final_ranking[: len(self.evidence_cards)]:
            raise ValueError("evidence cards must align with the final ranking prefix")
        return self


class UnifiedMetric(StrictModel):
    status: Literal["measured", "not_applicable", "unavailable"]
    value: float | None = None
    numerator: float = 0
    denominator: float = Field(default=0, ge=0)
    reason: str | None = None


class UnifiedEndToEndReport(StrictModel):
    schema_version: Literal[1] = 1
    benchmark_id: Literal["query_recommendation_v1"] = "query_recommendation_v1"
    case_count: int = Field(ge=1)
    metrics: dict[str, UnifiedMetric]


def evaluate_end_to_end(
    cases: Sequence[EndToEndCaseResult],
) -> UnifiedEndToEndReport:
    if not cases:
        raise ValueError("end-to-end evaluation requires at least one case")
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("end-to-end cases must be unique")
    metrics: dict[str, UnifiedMetric] = {}
    ranks = [_rank(item.final_ranking, item.target_business_id) for item in cases]
    retrieval_ranks = [_rank(item.retrieval_ranking, item.target_business_id) for item in cases]
    for cutoff in (50, 100, 500):
        metrics[f"Recall@{cutoff}"] = _ratio(
            sum(rank is not None and rank <= cutoff for rank in retrieval_ranks),
            len(cases),
        )
    for cutoff in (1, 3, 5, 10):
        metrics[f"HR@{cutoff}"] = _ratio(
            sum(rank is not None and rank <= cutoff for rank in ranks),
            len(cases),
        )
    metrics["MRR"] = _mean([0.0 if rank is None else 1.0 / rank for rank in ranks])
    metrics["NDCG@10"] = _mean([
        0.0 if rank is None or rank > 10 else 1.0 / math.log2(rank + 1)
        for rank in ranks
    ])
    metrics["RecommendationCompletionRate"] = _ratio(
        sum(
            item.agent_status == "completed"
            and item.response_kind == "recommendation"
            and bool(item.final_ranking)
            for item in cases
        ),
        len(cases),
    )
    metrics["ValidOutputRate"] = _ratio(
        sum(
            item.response_kind == "recommendation"
            and bool(item.final_ranking)
            and len(item.final_ranking) == len(set(item.final_ranking))
            for item in cases
        ),
        len(cases),
    )
    metrics["FallbackRate"] = _ratio(sum(item.fallback for item in cases), len(cases))
    metrics["ClarificationRate"] = _ratio(
        sum(item.agent_status == "awaiting_user" for item in cases),
        len(cases),
    )
    for name, attribute in (
        ("HardConstraintSatisfaction", "hard_constraint_satisfied_business_ids"),
        ("QueryCompliance", "query_compliant_business_ids"),
    ):
        for cutoff in (1, 5):
            scores = []
            for item in cases:
                allowed = getattr(item, attribute)
                if allowed is None:
                    continue
                shown = item.final_ranking[:cutoff]
                if shown:
                    scores.append(sum(value in set(allowed) for value in shown) / len(shown))
            metrics[f"{name}@{cutoff}"] = _mean_or_unavailable(scores, f"{attribute}_labels_missing")
    exclusion = [
        float(not set(item.final_ranking[:5]).intersection(item.rejected_business_ids))
        for item in cases if item.rejected_business_ids
    ]
    metrics["RejectedBusinessExclusionRate"] = _mean_or_na(exclusion)
    for cutoff in (1, 5):
        coverage = []
        for item in cases:
            cards = item.evidence_cards[:cutoff]
            if not cards:
                continue
            coverage.append(sum(bool(card.supporting_evidence or card.contradicting_evidence) for card in cards) / len(cards))
        metrics[f"EvidenceCoverage@{cutoff}"] = _mean_or_unavailable(coverage, "recommendation_evidence_cards_missing")
    labels = [item.evidence_labels for item in cases if item.evidence_labels is not None]
    metrics["EvidenceSupportPrecision"] = _booleans(labels, "support_correct")
    metrics["CitationCorrectness"] = _booleans(labels, "citation_correct")
    metrics["UnsupportedClaimRate"] = _booleans(labels, "factual_claim_supported", invert=True)
    metrics["EvidenceFreshnessCompliance"] = _booleans(labels, "freshness_compliant")
    freshness_reporting = [
        float(item.reported_evidence_recency)
        for item in cases
        if item.reported_evidence_recency is not None
    ]
    metrics["SourceFreshnessReporting"] = _mean_or_unavailable(
        freshness_reporting,
        "no_recommendation_review_evidence_to_report",
    )
    metrics["ContradictionDetection"] = _binary_detection(labels, "contradiction_expected", "contradiction_detected")
    metrics["InsufficientEvidenceDetection"] = _binary_detection(labels, "insufficiency_expected", "insufficiency_detected")
    gains = [item.clarification_utility_gain for item in cases if item.clarification_utility_gain is not None]
    metrics["PostClarificationUtilityGain"] = _mean_or_na([float(value) for value in gains])
    tool_selection = [float(item.tool_selection_correct) for item in cases if item.tool_selection_correct is not None]
    metrics["ToolSelectionAccuracy"] = _mean_or_unavailable(tool_selection, "agent_process_labels_missing")
    clarification = [float(item.clarification_utility) for item in cases if item.clarification_utility is not None]
    metrics["ClarificationUtility"] = _mean_or_na(clarification)
    recovery = [float(item.recovery_success) for item in cases if item.recovery_success is not None]
    metrics["RecoverySuccessRate"] = _mean_or_na(recovery)
    total_actions = sum(item.action_count for item in cases)
    metrics["InvalidActionRate"] = (
        _ratio(sum(item.invalid_action_count for item in cases), total_actions)
        if total_actions
        else UnifiedMetric(status="not_applicable", reason="no_agent_actions")
    )
    metrics["MeanToolCalls"] = _mean([float(item.tool_call_count) for item in cases])
    latencies = [item.latency_ms for item in cases]
    metrics["MeanLatencyMs"] = _mean(latencies)
    metrics["P95LatencyMs"] = UnifiedMetric(
        status="measured",
        value=_percentile(latencies, 0.95),
        numerator=_percentile(latencies, 0.95),
        denominator=1,
    )
    token_totals = [item.input_tokens + item.output_tokens for item in cases if item.input_tokens is not None and item.output_tokens is not None]
    metrics["MeanTokens"] = _mean_or_unavailable(token_totals, "token_usage_not_reported")
    costs = [item.cost_usd for item in cases if item.cost_usd is not None]
    metrics["MeanCostUsd"] = _mean_or_unavailable([float(value) for value in costs], "cost_not_reported")
    return UnifiedEndToEndReport(case_count=len(cases), metrics=metrics)


def _rank(ranking: Sequence[str], target: str) -> int | None:
    try:
        return list(ranking).index(target) + 1
    except ValueError:
        return None


def _ratio(numerator: float, denominator: float) -> UnifiedMetric:
    return UnifiedMetric(status="measured", value=numerator / denominator, numerator=numerator, denominator=denominator)


def _mean(values: Sequence[float]) -> UnifiedMetric:
    return _ratio(sum(values), len(values))


def _mean_or_na(values: Sequence[float]) -> UnifiedMetric:
    return _mean(values) if values else UnifiedMetric(status="not_applicable", reason="no_applicable_cases")


def _mean_or_unavailable(values: Sequence[float], reason: str) -> UnifiedMetric:
    return _mean(values) if values else UnifiedMetric(status="unavailable", reason=reason)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _booleans(labels: Sequence[EvidenceJudgmentLabels], field: str, *, invert: bool = False) -> UnifiedMetric:
    values = [value for label in labels for value in getattr(label, field)]
    if not values:
        return UnifiedMetric(status="unavailable", reason="evidence_judgment_labels_missing")
    hits = sum((not value) if invert else value for value in values)
    return _ratio(hits, len(values))


def _binary_detection(labels: Sequence[EvidenceJudgmentLabels], expected: str, detected: str) -> UnifiedMetric:
    pairs = [(getattr(item, expected), getattr(item, detected)) for item in labels if getattr(item, expected) is not None]
    positives = [(truth, prediction) for truth, prediction in pairs if truth]
    if not positives:
        return UnifiedMetric(status="unavailable", reason="positive_evidence_labels_missing")
    return _ratio(sum(prediction is True for _, prediction in positives), len(positives))
