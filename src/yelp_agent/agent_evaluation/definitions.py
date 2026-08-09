"""Human-readable, hashable metric catalog frozen by Step 21."""

from __future__ import annotations

from typing import Literal

from yelp_agent.models import StrictModel


type MetricGroup = Literal[
    "understanding",
    "ranking",
    "constraints",
    "clarification",
    "routing",
    "evidence",
    "cost",
]
type MetricSource = Literal["agent_scenario", "full_retrieval"]
type MetricAggregation = Literal[
    "micro",
    "macro",
    "mean",
    "percentile",
    "cost_per_success",
]


class MetricDefinition(StrictModel):
    name: str
    group: MetricGroup
    source: MetricSource
    aggregation: MetricAggregation
    unit: Literal["ratio", "milliseconds", "tokens", "usd"]
    higher_is_better: bool
    applicability: str
    formula: str


def _metric(
    name: str,
    group: MetricGroup,
    *,
    aggregation: MetricAggregation = "macro",
    unit: str = "ratio",
    higher: bool = True,
    source: MetricSource = "agent_scenario",
    applicability: str,
    formula: str,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        group=group,
        source=source,
        aggregation=aggregation,
        unit=unit,
        higher_is_better=higher,
        applicability=applicability,
        formula=formula,
    )


_METRICS = (
    _metric(
        "task_type_accuracy", "understanding", aggregation="micro",
        applicability="all observed user turns",
        formula="correct predicted task types / evaluated turns",
    ),
    *(
        _metric(
            f"hr_at_{cutoff}", "ranking",
            applicability="Agent recommendation scenes with acceptable-set labels",
            formula=f"mean[any acceptable business in candidate_ranking[:{cutoff}]]",
        )
        for cutoff in (1, 3, 5)
    ),
    _metric(
        "mrr", "ranking",
        applicability="Agent recommendation scenes with acceptable-set labels",
        formula="mean reciprocal rank of first acceptable business",
    ),
    _metric(
        "ndcg_at_5", "ranking",
        applicability="Agent recommendation scenes with acceptable-set labels",
        formula="mean binary-relevance NDCG@5 over acceptable businesses",
    ),
    *(
        _metric(
            f"recall_at_{cutoff}", "ranking", source="full_retrieval",
            applicability="Step 11 Full Retrieval tasks with held-out target labels",
            formula=f"targets retrieved in top {cutoff} / all held-out targets",
        )
        for cutoff in (50, 100, 500)
    ),
    _metric(
        "hard_constraint_satisfaction", "constraints",
        applicability="scenes with hard-filter labels",
        formula="mean fraction of displayed recommendations in acceptable set; empty=0",
    ),
    _metric(
        "valid_candidate_rate", "constraints",
        applicability="runs emitting a candidate ranking",
        formula="ranked business IDs inside hidden business scope / ranked IDs",
    ),
    _metric(
        "empty_result_rate", "constraints", higher=False,
        applicability="recommendation requests",
        formula="recommendation responses with no displayed business / requests",
    ),
    _metric(
        "missing_field_detection_precision", "clarification", aggregation="micro",
        applicability="turns with predicted or expected information gaps",
        formula="correctly detected gaps / all detected gaps",
    ),
    _metric(
        "missing_field_detection_recall", "clarification", aggregation="micro",
        applicability="turns with expected information gaps",
        formula="correctly detected gaps / all expected gaps",
    ),
    _metric(
        "unnecessary_question_rate", "clarification", aggregation="micro", higher=False,
        applicability="emitted clarification questions",
        formula="questions requesting non-hidden gaps / all questions",
    ),
    _metric(
        "question_answerability_rate", "clarification", aggregation="micro",
        applicability="emitted clarification questions",
        formula="questions covered by a releasable scripted user turn / all questions",
    ),
    _metric(
        "post_clarification_utility_gain", "clarification",
        applicability="clarified scenes with acceptable labels and pre/post rankings",
        formula="NDCG@5 after clarification - NDCG@5 before clarification",
    ),
    _metric(
        "average_questions_before_finalize", "clarification", aggregation="mean", higher=False,
        applicability="runs reaching a terminal action",
        formula="mean clarification questions before first terminal action",
    ),
    _metric(
        "action_accuracy", "routing", aggregation="micro",
        applicability="hidden required initial actions",
        formula="completed required actions / required actions",
    ),
    _metric(
        "tool_selection_accuracy", "routing", aggregation="micro",
        applicability="tool calls",
        formula="tool calls supporting an allowed action / all tool calls",
    ),
    _metric(
        "invalid_action_rate", "routing", aggregation="micro", higher=False,
        applicability="selected actions",
        formula="forbidden or non-allowed actions / selected actions",
    ),
    _metric(
        "repeated_tool_call_rate", "routing", aggregation="micro", higher=False,
        applicability="tool calls",
        formula="duplicate tool-name plus argument-hash calls / all tool calls",
    ),
    _metric(
        "direct_return_precision", "routing", aggregation="micro",
        applicability="non-fallback terminal return actions",
        formula="returns with no unresolved gap and completed prerequisites / returns",
    ),
    _metric(
        "unnecessary_semantic_tool_rate", "routing", aggregation="micro", higher=False,
        applicability="semantic tool calls",
        formula="semantic calls not supporting a required action / semantic calls",
    ),
    _metric(
        "unnecessary_rag_call_rate", "routing", aggregation="micro", higher=False,
        applicability="Review RAG calls",
        formula="Review RAG calls when review retrieval is not required / RAG calls",
    ),
    _metric(
        "fallback_rate", "routing", higher=False,
        applicability="all Agent scenario runs",
        formula="safe fallback runs / runs",
    ),
    _metric(
        "business_scope_isolation_rate", "evidence", aggregation="micro",
        applicability="retrieved evidence or required review retrieval",
        formula="retrieved evidence from hidden business scope / retrieved evidence",
    ),
    *(
        _metric(
            f"review_retrieval_recall_at_{cutoff}", "evidence",
            applicability="scenes with relevant Review-ID labels",
            formula=f"unique relevant reviews in top {cutoff} / all relevant reviews",
        )
        for cutoff in (1, 3, 5)
    ),
    *(
        _metric(
            f"evidence_precision_at_{cutoff}", "evidence",
            applicability="scenes with evidence labels",
            formula=f"relevant evidence in returned top {cutoff} / returned evidence",
        )
        for cutoff in (1, 3, 5)
    ),
    _metric(
        "grounded_answer_rate", "evidence",
        applicability="grounded or uncertain answer responses",
        formula="answers whose factual claims all cite relevant evidence / answers",
    ),
    _metric(
        "unsupported_claim_rate", "evidence", aggregation="micro", higher=False,
        applicability="user-visible factual claims",
        formula="claims without any relevant citation / factual claims",
    ),
    _metric(
        "citation_correctness", "evidence", aggregation="micro",
        applicability="explicit evidence citations",
        formula="citations matching relevant hidden labels / citations",
    ),
    _metric(
        "evidence_recency_reporting_rate", "evidence",
        applicability="scenes with relevant review evidence",
        formula="responses reporting evidence recency / applicable scenes",
    ),
    _metric(
        "conflict_detection_accuracy", "evidence",
        applicability="evidence-uncertainty scenes or scenes with evidence labels",
        formula="reported-conflict flag equals hidden conflict policy",
    ),
    _metric(
        "official_policy_caution_accuracy", "evidence",
        applicability="official-policy questions",
        formula="responses recommending official verification / policy questions",
    ),
    _metric(
        "mean_latency_ms", "cost", aggregation="mean", unit="milliseconds", higher=False,
        applicability="all runs", formula="mean end-to-end run latency",
    ),
    _metric(
        "p50_latency_ms", "cost", aggregation="percentile", unit="milliseconds", higher=False,
        applicability="all runs", formula="linear-interpolated latency percentile 0.50",
    ),
    _metric(
        "p95_latency_ms", "cost", aggregation="percentile", unit="milliseconds", higher=False,
        applicability="all runs", formula="linear-interpolated latency percentile 0.95",
    ),
    _metric(
        "mean_tokens", "cost", aggregation="mean", unit="tokens", higher=False,
        applicability="runs with provider token usage", formula="mean input plus output tokens",
    ),
    _metric(
        "p95_tokens", "cost", aggregation="percentile", unit="tokens", higher=False,
        applicability="runs with provider token usage", formula="linear-interpolated total-token percentile 0.95",
    ),
    _metric(
        "cost_per_successful_recommendation", "cost", aggregation="cost_per_success", unit="usd", higher=False,
        applicability="recommendation runs with complete cost observations",
        formula="total recommendation-run USD cost / successful recommendations",
    ),
    _metric(
        "cache_hit_rate", "cost", aggregation="micro",
        applicability="tool calls", formula="cached tool calls / tool calls",
    ),
)


def metric_definitions() -> tuple[MetricDefinition, ...]:
    """Return the immutable V1 catalog in stable name order."""

    return tuple(sorted(_METRICS, key=lambda item: item.name))
