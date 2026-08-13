"""Context-derived behavior labels; no LLM may invent these thresholds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .schema import BehaviorExpectation, PresentedBusinessSnapshot


def derive_relative_behavior(
    kind: Literal["cheaper", "closer", "quieter"],
    reference: PresentedBusinessSnapshot,
    candidates: Sequence[PresentedBusinessSnapshot],
) -> BehaviorExpectation:
    """Derive acceptable businesses by comparing frozen facts with the shown item."""

    if kind == "cheaper":
        baseline = reference.price_level
        values = {item.business_id: item.price_level for item in candidates}
    elif kind == "closer":
        baseline = reference.distance_km
        values = {item.business_id: item.distance_km for item in candidates}
    else:
        baseline = _noise_score(reference.noise_level)
        values = {
            item.business_id: _noise_score(item.noise_level) for item in candidates
        }
    if baseline is None:
        raise ValueError(f"presented business lacks a {kind} baseline")
    acceptable = [
        item.business_id
        for item in candidates
        if item.business_id != reference.business_id
        and values[item.business_id] is not None
        and float(values[item.business_id]) < float(baseline)
    ]
    if not acceptable:
        raise ValueError(f"candidate scope contains no business that is {kind}")
    return BehaviorExpectation(
        kind=kind,
        baseline_business_id=reference.business_id,
        baseline_value=baseline,
        acceptable_business_ids=acceptable,
    )


def _noise_score(value: str | None) -> int | None:
    return {
        "quiet": 0,
        "average": 1,
        "loud": 2,
        "very_loud": 3,
    }.get(value)  # type: ignore[arg-type]
