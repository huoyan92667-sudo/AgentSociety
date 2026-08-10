"""Stable clarification priority and reason codes."""

from __future__ import annotations

from yelp_agent.decision_readiness.schema import InformationGap


GAP_PRIORITY: tuple[InformationGap, ...] = (
    "constraint_conflict",
    "ambiguous_reference",
    "missing_location",
    "missing_budget",
    "missing_party_size",
)

GAP_REASON_CODES: dict[InformationGap, str] = {
    "constraint_conflict": "CONSTRAINT_CONFLICT",
    "ambiguous_reference": "AMBIGUOUS_REFERENCE",
    "missing_location": "MISSING_LOCATION",
    "missing_budget": "MISSING_BUDGET",
    "missing_party_size": "MISSING_PARTY_SIZE",
}


def highest_priority_gap(gaps: list[InformationGap]) -> InformationGap:
    """Return one stable blocking gap without inventing new information."""

    available = set(gaps)
    for gap in GAP_PRIORITY:
        if gap in available:
            return gap
    raise ValueError("at least one supported information gap is required")
