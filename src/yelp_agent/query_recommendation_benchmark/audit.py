"""Leakage and target-support audit for the frozen Query benchmark."""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import Field

from yelp_agent.models import StrictModel
from yelp_agent.reviews.schema import ASPECT_NAMES

from .frames import AnchorKnowledgeReader, PlannedQueryRecommendationCase
from .schema import QueryRecommendationBenchmarkBundle


class QueryRecommendationAuditReport(StrictModel):
    schema_version: Literal[1] = 1
    case_count: int = Field(ge=1)
    passed: bool
    violation_count: int = Field(ge=0)
    violations: list[str]
    split_counts: dict[str, int]
    target_review_text_loaded: Literal[False] = False
    test_source_tasks_used: Literal[False] = False
    hidden_labels_visible_to_agent: Literal[False] = False
    llm_determined_hidden_labels: Literal[False] = False
    aspect_source_scope: Literal["selected_user_interactions"] = (
        "selected_user_interactions"
    )


def audit_query_recommendation_benchmark(
    bundle: QueryRecommendationBenchmarkBundle,
    drafts: Sequence[PlannedQueryRecommendationCase],
    reader: AnchorKnowledgeReader,
) -> QueryRecommendationAuditReport:
    """Prove real-positive, cutoff, support, split, and visibility invariants."""

    visible = {item.case_id: item for item in bundle.visible_cases}
    truth = {item.case_id: item for item in bundle.ground_truth}
    frames = {item.case_id: item for item in bundle.frames}
    violations: list[str] = []
    split_counts: dict[str, int] = {}
    for draft in drafts:
        case_id = draft.frame.case_id
        case = visible.get(case_id)
        target = truth.get(case_id)
        frame = frames.get(case_id)
        if case is None or target is None or frame is None:
            violations.append(f"{case_id}:ARTIFACT_ALIGNMENT_FAILURE")
            continue
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
        anchor = draft.anchor
        knowledge = reader.describe(anchor)
        if anchor.source_split == "train" and case.split != "development":
            violations.append(f"{case_id}:SOURCE_SPLIT_MISMATCH")
        if anchor.source_split == "validation" and case.split != "validation":
            violations.append(f"{case_id}:SOURCE_SPLIT_MISMATCH")
        if anchor.source_split not in {"train", "validation"}:
            violations.append(f"{case_id}:TEST_SOURCE_TASK_USED")
        if anchor.target_stars < 4:
            violations.append(f"{case_id}:TARGET_RATING_BELOW_FOUR")
        if anchor.target_time != anchor.cutoff_time:
            violations.append(f"{case_id}:TARGET_TIME_NOT_CUTOFF")
        if anchor.target_pre_cutoff_review_count < 1:
            violations.append(f"{case_id}:TARGET_NOT_PREEXISTING")
        if anchor.target_business_id in set(anchor.history_business_ids):
            violations.append(f"{case_id}:TARGET_IN_USER_HISTORY")
        if target.target_business_id != anchor.target_business_id:
            violations.append(f"{case_id}:HIDDEN_TARGET_MISMATCH")
        normalized_query = case.query_text.casefold()
        if anchor.target_business_id.casefold() in normalized_query:
            violations.append(f"{case_id}:TARGET_ID_IN_QUERY")
        if anchor.target_review_id.casefold() in normalized_query:
            violations.append(f"{case_id}:TARGET_REVIEW_ID_IN_QUERY")
        if knowledge.business_id != anchor.target_business_id:
            violations.append(f"{case_id}:KNOWLEDGE_BUSINESS_MISMATCH")
        if knowledge.aspect_source_scope != "selected_user_interactions":
            violations.append(f"{case_id}:ASPECT_SCOPE_MISMATCH")
        for condition in frame.conditions:
            if condition.field == "category":
                if str(condition.value) not in knowledge.fine_categories:
                    violations.append(f"{case_id}:UNSUPPORTED_CATEGORY")
            elif condition.field == "distance_km":
                if (
                    knowledge.target_distance_km is None
                    or knowledge.target_distance_km > float(condition.value)
                ):
                    violations.append(f"{case_id}:UNSUPPORTED_DISTANCE")
            elif condition.field == "price_level":
                if knowledge.price_level is None:
                    violations.append(f"{case_id}:UNSUPPORTED_PRICE")
                elif condition.operator == "less_than_or_equal" and (
                    knowledge.price_level > int(condition.value)
                ):
                    violations.append(f"{case_id}:UNSUPPORTED_PRICE")
            elif condition.field in ASPECT_NAMES and (
                condition.field not in knowledge.positive_aspects
            ):
                violations.append(
                    f"{case_id}:UNSUPPORTED_ASPECT:{condition.field}"
                )
    users_by_split: dict[str, set[str]] = {}
    for item in bundle.visible_cases:
        users_by_split.setdefault(item.split, set()).add(item.user_id)
    if users_by_split.get("development", set()).intersection(
        users_by_split.get("validation", set())
    ):
        violations.append("GLOBAL:USER_SPLIT_LEAKAGE")
    return QueryRecommendationAuditReport(
        case_count=len(bundle.visible_cases),
        passed=not violations,
        violation_count=len(violations),
        violations=violations,
        split_counts=dict(sorted(split_counts.items())),
    )
