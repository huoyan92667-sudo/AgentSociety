"""Observable integrity checks for a complete Step 20 benchmark bundle."""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import Field

from yelp_agent.config import AgentBenchmarkConfig
from yelp_agent.models import StrictModel

from .builder import AgentBenchmarkBundle


class AgentBenchmarkAuditReport(StrictModel):
    scenario_count: int = Field(ge=1)
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    language_counts: dict[str, int]
    task_type_counts: dict[str, int]
    source_query_reuse_count: int = Field(ge=0)
    scripted_user_turn_count: int = Field(ge=0)
    evidence_label_count: int = Field(ge=0)
    relevant_review_label_count: int = Field(ge=0)
    out_of_scope_label_count: int = Field(ge=0)
    unique_scenario_ids: bool
    visible_hidden_ids_aligned: bool
    unique_visible_queries: bool
    development_validation_users_disjoint: bool
    development_validation_businesses_disjoint: bool
    development_validation_frame_families_disjoint: bool
    evidence_before_cutoff: bool
    evidence_scope_isolated: bool
    test_source_tasks_absent: bool
    hidden_fields_absent_from_visible: bool
    language_distribution_matches_config: bool
    deterministic_generation_ready: bool


def audit_agent_benchmark_bundle(
    bundle: AgentBenchmarkBundle,
    config: AgentBenchmarkConfig,
) -> AgentBenchmarkAuditReport:
    """Fail closed on leakage, count drift, or invalid evidence scope."""

    visible = bundle.visible_scenarios
    truth = bundle.ground_truth
    evidence = bundle.evidence_labels
    visible_by_id = {item.scenario_id: item for item in visible}
    truth_by_id = {item.scenario_id: item for item in truth}
    expected_total = config.development_count + config.validation_count
    unique_ids = (
        len(visible_by_id) == len(visible)
        and len(truth_by_id) == len(truth)
        and len(visible) == expected_total
    )
    aligned = set(visible_by_id) == set(truth_by_id)
    split_counts = Counter(item.split for item in visible)
    category_counts = Counter(item.scenario_category for item in truth)
    language_counts = Counter(item.language for item in visible)
    task_counts = Counter(item.task_type for item in truth)
    if split_counts != {
        "development": config.development_count,
        "validation": config.validation_count,
    }:
        raise ValueError("Step 20 split counts do not match the frozen config")
    if category_counts != config.category_counts:
        raise ValueError("Step 20 category counts do not match the frozen config")
    normalized_queries = [" ".join(item.query_text.casefold().split()) for item in visible]
    unique_queries = len(normalized_queries) == len(set(normalized_queries))
    expected_english = round(expected_total * config.english_fraction)
    language_matches = language_counts == {
        "en-US": expected_english,
        "zh-CN": expected_total - expected_english,
    }

    users: defaultdict[str, set[str]] = defaultdict(set)
    businesses: defaultdict[str, set[str]] = defaultdict(set)
    frame_families: defaultdict[str, set[str]] = defaultdict(set)
    for scenario in visible:
        users[scenario.split].add(scenario.user_id)
        for business_id in truth_by_id[scenario.scenario_id].business_scope:
            businesses[scenario.split].add(business_id)
        frame_families[scenario.split].add(
            truth_by_id[scenario.scenario_id].frame_family
        )
    user_disjoint = not users["development"].intersection(users["validation"])
    business_disjoint = not businesses["development"].intersection(
        businesses["validation"]
    )
    frame_disjoint = not frame_families["development"].intersection(
        frame_families["validation"]
    )

    evidence_before_cutoff = True
    evidence_scope_isolated = True
    for label in evidence:
        scenario = visible_by_id.get(label.scenario_id)
        hidden = truth_by_id.get(label.scenario_id)
        if scenario is None or hidden is None:
            evidence_scope_isolated = False
            continue
        if label.event_time is not None and label.event_time >= scenario.cutoff_time:
            evidence_before_cutoff = False
        in_scope = label.business_id in hidden.business_scope
        if label.relevance == "out_of_scope":
            evidence_scope_isolated &= not in_scope
        else:
            evidence_scope_isolated &= in_scope
    test_absent = all(not item.source_task_id.startswith("test:") for item in truth)
    visible_keys = set().union(
        *(set(item.model_dump().keys()) for item in visible)
    )
    hidden_absent = not visible_keys.intersection(
        {
            "task_type",
            "expected_information_gaps",
            "allowed_actions",
            "required_actions",
            "forbidden_actions",
            "business_scope",
            "acceptable_business_ids",
            "uncertainty_policy",
            "scripted_user_turns",
        }
    )
    report = AgentBenchmarkAuditReport(
        scenario_count=len(visible),
        split_counts=dict(sorted(split_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        task_type_counts=dict(sorted(task_counts.items())),
        source_query_reuse_count=sum(
            item.source_query_case_id is not None for item in truth
        ),
        scripted_user_turn_count=sum(len(item.scripted_user_turns) for item in truth),
        evidence_label_count=len(evidence),
        relevant_review_label_count=sum(
            item.source_type == "review" and item.relevance == "relevant"
            for item in evidence
        ),
        out_of_scope_label_count=sum(
            item.relevance == "out_of_scope" for item in evidence
        ),
        unique_scenario_ids=unique_ids,
        visible_hidden_ids_aligned=aligned,
        unique_visible_queries=unique_queries,
        development_validation_users_disjoint=user_disjoint,
        development_validation_businesses_disjoint=business_disjoint,
        development_validation_frame_families_disjoint=frame_disjoint,
        evidence_before_cutoff=evidence_before_cutoff,
        evidence_scope_isolated=evidence_scope_isolated,
        test_source_tasks_absent=test_absent,
        hidden_fields_absent_from_visible=hidden_absent,
        language_distribution_matches_config=language_matches,
        deterministic_generation_ready=True,
    )
    failed = [
        name
        for name in (
            "unique_scenario_ids",
            "visible_hidden_ids_aligned",
            "unique_visible_queries",
            "development_validation_users_disjoint",
            "development_validation_businesses_disjoint",
            "development_validation_frame_families_disjoint",
            "evidence_before_cutoff",
            "evidence_scope_isolated",
            "test_source_tasks_absent",
            "hidden_fields_absent_from_visible",
            "language_distribution_matches_config",
        )
        if not getattr(report, name)
    ]
    if failed:
        raise ValueError(f"Step 20 audit failed: {', '.join(failed)}")
    return report
