"""Visible-query helpers shared by the Agent Adapter and offline evaluator."""

from __future__ import annotations

from collections.abc import Sequence

from yelp_agent.query.schema import RequestCondition
from yelp_agent.reviews.schema import ASPECT_NAMES, AspectName
from yelp_agent.review_rag import infer_review_aspects


def aggregation_query_facts(
    query_text: str,
    conditions: Sequence[RequestCondition],
) -> tuple[list[AspectName], dict[AspectName, str], bool]:
    aspects: list[AspectName] = []
    polarity: dict[AspectName, str] = {}
    for condition in conditions:
        if condition.field not in ASPECT_NAMES:
            continue
        aspect = condition.field
        if aspect not in aspects:
            aspects.append(aspect)
        negative = condition.operator in {"avoid", "excludes"} or (
            condition.operator == "equals" and condition.value is False
        )
        polarity[aspect] = "negative" if negative else "positive"
    for aspect in infer_review_aspects(query_text):
        if aspect not in aspects:
            aspects.append(aspect)
        polarity.setdefault(aspect, "positive")
    return aspects, polarity, asks_for_uncertainty(query_text)


def asks_for_uncertainty(query_text: str) -> bool:
    text = query_text.casefold()
    return any(
        marker in text
        for marker in (
            "only a few",
            "very few",
            "conflicting",
            "mixed reviews",
            "can we be sure",
            "能确定吗",
            "很少评论",
            "说法不一",
            "相互矛盾",
            "新旧评论",
        )
    )
