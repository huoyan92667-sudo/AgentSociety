"""Agent tool Adapter for deterministic cross-review evidence aggregation."""

from __future__ import annotations

from typing import Protocol

from yelp_agent.agent_evaluation.schema import EvidenceReference
from yelp_agent.evidence_aggregation import (
    EvidenceAggregationRequest,
    EvidenceAssessment,
    aggregation_query_facts,
)
from yelp_agent.query.schema import RequestCondition
from yelp_agent.review_rag import ReviewSearchResult

from ..errors import PermanentToolError
from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import (
    AggregateReviewEvidenceInput,
    AggregateReviewEvidenceOutput,
)


class ReviewEvidenceAggregator(Protocol):
    def aggregate(self, request: EvidenceAggregationRequest) -> EvidenceAssessment: ...


class AggregateReviewEvidenceTool:
    """Consume an existing Review search observation; never query a wider scope."""

    definition = ToolDefinition(
        name="AGGREGATE_REVIEW_EVIDENCE",
        version="1.0.0",
        kind="review_rag",
        allowed_actions=("retrieve_business_reviews",),
        input_model=AggregateReviewEvidenceInput,
        output_model=AggregateReviewEvidenceOutput,
        public_summary=(
            "Aggregate supporting, contradicting, sparse, and time-sensitive Review evidence."
        ),
        preconditions=(
            "SEARCH_BUSINESS_REVIEWS already completed in the current turn",
            "aggregation scope exactly matches the locked Review search scope",
        ),
        resolves_uncertainties=(
            "conflicting_review_evidence",
            "sparse_review_evidence",
        ),
        cache_scope="request",
        max_attempts=1,
        timeout_ms=5_000,
        fallback_policy="return_uncertain_answer",
    )

    def __init__(self, aggregator: ReviewEvidenceAggregator) -> None:
        self._aggregator = aggregator

    def run(
        self,
        arguments: AggregateReviewEvidenceInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        search_result = _latest_review_search(context)
        if search_result.business_ids != arguments.business_ids:
            raise PermanentToolError("aggregation scope must match Review search scope")
        request_payload = context.state_snapshot.get("request")
        readiness = context.state_snapshot.get("readiness")
        if not isinstance(request_payload, dict) or not isinstance(readiness, dict):
            raise PermanentToolError("visible request or readiness state is missing")
        query_text = request_payload.get("query_text")
        conditions = request_payload.get("conditions")
        task_type = readiness.get("task_type")
        if not isinstance(query_text, str) or not isinstance(task_type, str):
            raise PermanentToolError("query text or task type is missing")
        parsed_conditions = [
            RequestCondition.model_validate(value)
            for value in conditions or []
            if isinstance(value, dict)
        ]
        aspects, polarity, explicit_uncertainty = aggregation_query_facts(
            query_text,
            parsed_conditions,
        )
        assessment = self._aggregator.aggregate(
            EvidenceAggregationRequest(
                query_text=query_text,
                task_type=task_type,  # type: ignore[arg-type]
                requested_aspects=aspects,  # type: ignore[arg-type]
                desired_polarity_by_aspect=polarity,  # type: ignore[arg-type]
                explicit_uncertainty_request=explicit_uncertainty,
                search_result=search_result,
            )
        )
        aspect_rows = [
            aspect
            for business in assessment.businesses
            for aspect in business.aspects
        ]
        citation_ids = {
            (aspect.business_id, review_id)
            for aspect in aspect_rows
            for review_id in aspect.citation_review_ids
        }
        evidence = [
            EvidenceReference(
                business_id=business_id,
                source_type="review",
                review_id=review_id,
            )
            for business_id, review_id in sorted(citation_ids)
        ]
        status = "success" if any(row.evidence_count for row in aspect_rows) else "no_result"
        warnings: list[str] = []
        if any(row.has_conflict for row in aspect_rows):
            warnings.append("supporting and contradicting Review evidence coexist")
        if any(row.confidence_level in {"insufficient", "low"} for row in aspect_rows):
            warnings.append("Review evidence is sparse or low confidence")
        if assessment.recommend_official_verification:
            warnings.append("historical Reviews are not official current policy")
        return ToolObservation(
            tool_name=self.definition.name,
            status=status,
            data=assessment.model_dump(mode="json"),
            confidence=(
                min(row.confidence_score for row in aspect_rows)
                if aspect_rows
                else None
            ),
            evidence=evidence,
            input_tokens=0,
            output_tokens=0,
            warnings=warnings,
        )


def _latest_review_search(context: ToolExecutionContext) -> ReviewSearchResult:
    observations = context.state_snapshot.get("observations")
    if not isinstance(observations, list):
        raise PermanentToolError("Review search observation is missing")
    current_turn = context.state_snapshot.get("turn_index")
    for item in reversed(observations):
        if not isinstance(item, dict) or item.get("turn_index") != current_turn:
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("tool_name") != "SEARCH_BUSINESS_REVIEWS":
            continue
        if payload.get("status") not in {"success", "partial", "no_result"}:
            continue
        data = payload.get("data")
        if isinstance(data, dict):
            return ReviewSearchResult.model_validate(data)
    raise PermanentToolError("Review search must complete before aggregation")
