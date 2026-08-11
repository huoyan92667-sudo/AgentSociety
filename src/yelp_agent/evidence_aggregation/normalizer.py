"""Convert Review RAG hits into deduplicated query-relative evidence atoms."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import math

from .config import EvidenceAggregationPolicy
from .schema import EvidenceAggregationRequest, EvidenceAtom, EvidenceStance


_CONDITION_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("indoor", ("indoor", "inside", "室内", "店内")),
    ("outdoor", ("outdoor", "outside", "patio", "露台", "户外", "室外")),
    ("weekday", ("weekday", "monday", "tuesday", "wednesday", "thursday", "工作日")),
    ("weekend", ("weekend", "friday", "saturday", "sunday", "周末", "周五", "周六", "周日")),
    ("lunch", ("lunch", "noon", "午餐", "中午")),
    ("dinner", ("dinner", "evening", "night", "晚餐", "晚上", "夜间")),
)


def normalize_evidence(
    request: EvidenceAggregationRequest,
    policy: EvidenceAggregationPolicy,
) -> tuple[EvidenceAtom, ...]:
    """Return one strongest atom per Review and Aspect, in deterministic order."""

    requested = set(request.requested_aspects)
    candidates: dict[tuple[str, str], EvidenceAtom] = {}
    for hit in request.search_result.hits:
        if hit.relevance_score < policy.minimum_relevance:
            continue
        for evidence in hit.aspect_evidence:
            if requested and evidence.aspect not in requested:
                continue
            if evidence.confidence < policy.minimum_extraction_confidence:
                continue
            recency = _recency_score(
                review_time=hit.review_time,
                cutoff_time=request.search_result.cutoff_time,
                half_life_days=policy.recency_half_life_days,
            )
            stance = _stance(
                evidence.sentiment,
                polarity=request.desired_polarity_by_aspect.get(
                    evidence.aspect, "positive"
                ),
            )
            atom = EvidenceAtom(
                review_id=hit.review_id,
                business_id=hit.business_id,
                user_id=hit.user_id,
                review_time=hit.review_time,
                hit_rank=hit.rank,
                aspect=evidence.aspect,
                stance=stance,
                relevance_score=hit.relevance_score,
                extraction_confidence=evidence.confidence,
                recency_score=recency,
                weight=max(
                    0.0,
                    min(1.0, hit.relevance_score * evidence.confidence * recency),
                ),
                condition_tags=_condition_tags(
                    (evidence.evidence_span, hit.text)
                ),
                evidence_span=evidence.evidence_span[:2000],
                text_sha256=hit.text_sha256,
            )
            key = (atom.review_id, atom.aspect)
            current = candidates.get(key)
            if current is None or _atom_order(atom) < _atom_order(current):
                candidates[key] = atom
    return tuple(sorted(candidates.values(), key=_atom_order))


def _atom_order(atom: EvidenceAtom) -> tuple[float, int, str, str]:
    return (-atom.weight, atom.hit_rank, atom.review_id, atom.aspect)


def _recency_score(
    *,
    review_time: datetime,
    cutoff_time: datetime,
    half_life_days: int,
) -> float:
    age_days = max(0.0, (cutoff_time - review_time).total_seconds() / 86_400.0)
    return float(math.pow(2.0, -age_days / half_life_days))


def _stance(sentiment: str, *, polarity: str) -> EvidenceStance:
    if sentiment not in {"positive", "negative"}:
        return "neutral"
    supports = sentiment == polarity
    return "supports" if supports else "contradicts"


def _condition_tags(texts: Iterable[str]) -> list[str]:
    text = " ".join(texts).casefold()
    return [
        name
        for name, markers in _CONDITION_MARKERS
        if any(marker in text for marker in markers)
    ]
