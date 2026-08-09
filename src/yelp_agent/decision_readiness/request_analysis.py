"""Deterministic request-state analysis before any Agent action is chosen."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from yelp_agent.query.schema import RecommendationRequest, RequestCondition

from .schema import InformationGap, TaskType


_FEEDBACK_PATTERNS = (
    re.compile(r"(?:太贵|太远|太吵|换一(?:个|家|批)|换个|不要刚才|重新推荐)"),
    re.compile(
        r"(?:too expensive|too far|too noisy|show me another|something else|"
        r"replace (?:that|it)|new recommendations?)",
        re.IGNORECASE,
    ),
)
_COMPARISON_PATTERNS = (
    re.compile(r"(?:哪(?:个|家).{0,8}更|比较|对比|区别)"),
    re.compile(r"(?:which (?:one|place).{0,12}better|compare|versus|\bvs\.?\b)", re.I),
)
_REVIEW_PATTERNS = (
    re.compile(r"(?:评论|评价里|有人说|食客说|大家说)"),
    re.compile(r"(?:reviews?|reviewers?|people say|customers? say)", re.I),
)
_OFFICIAL_PATTERNS = (
    re.compile(r"(?:现在|目前|官方|政策|今天).{0,16}(?:允许|营业|开放|能不能|可不可以|规定)"),
    re.compile(
        r"(?:official|policy|currently|right now|open today|allows? .* now)",
        re.I,
    ),
)
_REFERENCE_PATTERNS = (
    re.compile(r"(?:这家|那家|第一家|第二家|前一家|上一个|刚才的)"),
    re.compile(r"(?:this place|that place|the first one|the second one|previous one)", re.I),
)
_DETAIL_QUESTION_PATTERNS = (
    re.compile(r"(?:吗|么|如何|怎么样|方便|能不能|有没有|是否)"),
    re.compile(r"(?:does|is|can|what|how|where|has|have)\b", re.I),
)
_RECOMMENDATION_PATTERNS = (
    re.compile(r"(?:推荐|找一?家|想吃|去哪吃|帮我找|有没有.*(?:餐厅|店|馆))"),
    re.compile(r"(?:recommend|suggest|looking for|find me|where (?:can|should) .* eat)", re.I),
)
_GROUP_PATTERNS = (
    re.compile(r"(?:聚餐|聚会|团建|一群人|多人)"),
    re.compile(r"(?:group|team dinner|party of|gathering)", re.I),
)


def _matches(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def classify_task_type(request: RecommendationRequest) -> tuple[TaskType, str]:
    """Classify one turn with explicit priority and an auditable reason code."""

    text = request.query_text
    if _matches(_FEEDBACK_PATTERNS, text):
        return "feedback_refinement", "feedback_language"
    if _matches(_COMPARISON_PATTERNS, text):
        return "candidate_comparison", "comparison_language"
    if _matches(_REVIEW_PATTERNS, text):
        return "review_experience_question", "review_evidence_language"
    if _matches(_OFFICIAL_PATTERNS, text):
        return "official_policy_question", "current_policy_language"
    has_reference = bool(request.referenced_business_ids) or _matches(
        _REFERENCE_PATTERNS,
        text,
    )
    if has_reference and _matches(_DETAIL_QUESTION_PATTERNS, text):
        return "business_detail_question", "referenced_business_question"
    if (
        request.intent == "recommendation_request"
        and (request.conditions or _matches(_RECOMMENDATION_PATTERNS, text))
    ):
        return "recommendation_request", "recommendation_request_contract"
    return "unknown", "no_supported_task_signal"


def _condition_conflicts(
    conditions: Iterable[RequestCondition],
) -> tuple[str, ...]:
    by_field_value: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    numeric_bounds: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for condition in conditions:
        key = (condition.field, str(condition.value).casefold())
        by_field_value[key].add(condition.operator)
        if condition.operator in {"less_than_or_equal", "greater_than_or_equal"}:
            try:
                numeric_bounds[condition.field][condition.operator] = float(
                    condition.value
                )
            except (TypeError, ValueError):
                continue
    conflicts: set[str] = set()
    for (field, _), operators in by_field_value.items():
        if {"includes", "excludes"}.issubset(operators) or {
            "prefer",
            "avoid",
        }.issubset(operators):
            conflicts.add(field)
    for field, bounds in numeric_bounds.items():
        lower = bounds.get("greater_than_or_equal")
        upper = bounds.get("less_than_or_equal")
        if lower is not None and upper is not None and lower > upper:
            conflicts.add(field)
    return tuple(sorted(conflicts))


def identify_information_gaps(
    request: RecommendationRequest,
    task_type: TaskType,
) -> tuple[tuple[InformationGap, ...], tuple[str, ...]]:
    """Convert parser state and reference requirements into Router-ready gaps."""

    gaps: set[InformationGap] = set()
    if "user_location" in request.missing_fields:
        gaps.add("missing_location")
    if "budget_precision" in request.missing_fields:
        gaps.add("missing_budget")
    group_request = any(
        condition.field == "group_suitable" for condition in request.conditions
    ) or _matches(_GROUP_PATTERNS, request.query_text)
    if group_request and request.party_size is None:
        gaps.add("missing_party_size")

    conflicts = _condition_conflicts(request.conditions)
    if conflicts:
        gaps.add("constraint_conflict")

    reference_count = len(request.referenced_business_ids)
    if task_type in {
        "business_detail_question",
        "feedback_refinement",
        "official_policy_question",
        "review_experience_question",
    } and reference_count < 1:
        gaps.add("ambiguous_reference")
    if task_type == "candidate_comparison" and reference_count < 2:
        gaps.add("ambiguous_reference")
    if "ambiguous_requirement" in request.missing_fields:
        gaps.add("ambiguous_reference")
    return tuple(sorted(gaps)), conflicts
