"""Compile accepted Step-29 semantics into a deterministic ranking document."""

from __future__ import annotations

import hashlib

from yelp_agent.query.schema import RecommendationRequest, RequestCondition

from .schema import RankingIntent, RankingIntentCondition


_FIELD_LABELS = {
    "category": "business category",
    "distance_km": "maximum distance in kilometres",
    "budget_per_person": "budget per person",
    "price_level": "price level",
    "quiet_environment": "quiet environment",
    "crowded": "crowding",
    "queue_time": "queue time",
    "parking": "parking",
    "pet_friendly": "pet friendly",
    "family_friendly": "family friendly",
    "date_suitable": "date suitable",
    "group_suitable": "group suitable",
    "spiciness": "spiciness",
    "cleanliness": "cleanliness",
    "food_quality": "food quality",
    "service": "service",
    "price_value": "value for money",
}


def _condition_text(condition: RequestCondition) -> str:
    return (
        f"{condition.importance} {_FIELD_LABELS[condition.field]} "
        f"{condition.operator} {condition.value}; source={condition.source}; "
        f"confidence={condition.confidence:.3f}"
    )


class RankingIntentCompiler:
    """One stable interface from a canonical request to ranking intent."""

    version = "step30-ranking-intent-v1"

    def compile(self, request: RecommendationRequest) -> RankingIntent:
        accepted = [
            condition
            for condition in request.conditions
            if condition.enforcement != "clarify"
        ]
        rankable = [
            condition
            for condition in accepted
            if condition.enforcement in {"rank", "evidence"}
        ]
        condition_rows = [
            RankingIntentCondition(
                field=condition.field,
                operator=condition.operator,
                value=condition.value,
                importance=condition.importance,
                enforcement=condition.enforcement,
                source=condition.source,
                confidence=condition.confidence,
                evidence_span=condition.evidence_span,
            )
            for condition in accepted
        ]
        lines = [
            f"Original request: {request.query_text}",
            f"Task: {request.intent}",
        ]
        if request.party_size is not None:
            lines.append(f"Party size: {request.party_size}")
        if accepted:
            lines.append("Accepted structured requirements:")
            lines.extend(f"- {_condition_text(item)}" for item in accepted)
        else:
            lines.append("Accepted structured requirements: none")
        document = "\n".join(lines)
        confidence = (
            sum(condition.confidence for condition in rankable) / len(rankable)
            if rankable
            else 0.0
        )
        return RankingIntent(
            request_id=request.request_id,
            document=document,
            document_sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
            conditions=condition_rows,
            rankable_condition_count=len(rankable),
            semantic_model_condition_count=sum(
                condition.source == "semantic_model" for condition in rankable
            ),
            mean_confidence=confidence,
        )
