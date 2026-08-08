"""Adapters from frozen point-in-time knowledge into Query-aware candidates."""

from __future__ import annotations

from yelp_agent.business_profiles.schema import BusinessProfileV1
from yelp_agent.query.ranking import CandidateAspectEvidence, QueryAwareCandidate


def _price_level(attributes: dict[str, object]) -> int | None:
    raw = attributes.get("RestaurantsPriceRange2")
    if raw is None:
        return None
    try:
        value = int(str(raw).strip().strip("'\""))
    except ValueError:
        return None
    return value if 1 <= value <= 4 else None


def candidate_from_business_profile(
    profile: BusinessProfileV1,
    *,
    hybrid_rank: int,
) -> QueryAwareCandidate:
    """Preserve the cutoff-frozen profile without reading source reviews."""

    aspects = {
        aspect: CandidateAspectEvidence(
            status=summary.status,
            positive_ratio=summary.weighted_positive_ratio,
            negative_ratio=summary.weighted_negative_ratio,
            confidence=summary.confidence,
            evidence_count=summary.evidence_count,
            conflict=summary.conflict,
            latest_evidence_time=summary.latest_evidence_time,
        )
        for aspect, summary in profile.aspect_summaries.items()
    }
    return QueryAwareCandidate(
        business_id=profile.business_id,
        hybrid_rank=hybrid_rank,
        categories=tuple(profile.categories),
        latitude=profile.latitude,
        longitude=profile.longitude,
        price_level=_price_level(profile.structured_attributes),
        aspect_evidence=aspects,
    )
