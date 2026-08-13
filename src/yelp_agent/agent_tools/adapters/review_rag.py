"""Agent tool Adapter for business-scoped Review RAG."""

from __future__ import annotations

from typing import Protocol

from yelp_agent.agent_evaluation.schema import EvidenceReference
from yelp_agent.reviews.schema import ASPECT_NAMES
from yelp_agent.review_rag import (
    ReviewSearchRequest,
    ReviewSearchResult,
    infer_review_aspects,
)
from yelp_agent.semantic_embedding import EmbeddingProviderError

from ..errors import PermanentToolError, RetryableToolError
from ..registry import ToolDefinition
from ..request_context import (
    query_text_from_tool_context,
    request_conditions_from_tool_context,
)
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import (
    SearchBusinessReviewsInput,
    SearchBusinessReviewsOutput,
)


class BusinessReviewSearch(Protocol):
    def search(self, request: ReviewSearchRequest) -> ReviewSearchResult: ...


class SearchBusinessReviewsTool:
    """Retrieve raw Review IDs without allowing the model to change scope."""

    definition = ToolDefinition(
        name="SEARCH_BUSINESS_REVIEWS",
        version="1.0.0",
        kind="review_rag",
        allowed_actions=("retrieve_business_reviews",),
        input_model=SearchBusinessReviewsInput,
        output_model=SearchBusinessReviewsOutput,
        public_summary=(
            "Retrieve Top-5 cutoff-safe Review passages for explicitly locked businesses."
        ),
        preconditions=(
            "business IDs are explicitly referenced and inside current scope",
            "Review timestamps must be strictly earlier than cutoff",
        ),
        resolves_uncertainties=("unstructured_review_evidence",),
        cache_scope="request",
        max_attempts=1,
        timeout_ms=90_000,
        fallback_policy="return_uncertain_answer",
    )

    def __init__(self, search: BusinessReviewSearch) -> None:
        self._search = search

    def run(
        self,
        arguments: SearchBusinessReviewsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        try:
            query_text = query_text_from_tool_context(context)
            conditions = request_conditions_from_tool_context(context)
        except (TypeError, ValueError) as exc:
            raise PermanentToolError(str(exc)) from None
        aspects: list[str] = []
        for condition in conditions:
            if condition.field in ASPECT_NAMES and condition.field not in aspects:
                aspects.append(str(condition.field))
        for aspect in infer_review_aspects(query_text):
            if aspect not in aspects:
                aspects.append(aspect)
        try:
            result = self._search.search(
                ReviewSearchRequest(
                    query_text=query_text,
                    business_ids=arguments.business_ids,
                    cutoff_time=context.cutoff_time,
                    aspects=aspects,  # type: ignore[arg-type]
                    top_k=arguments.top_k,
                    usage_scope=context.request_id,
                )
            )
        except EmbeddingProviderError as exc:
            if exc.retryable:
                raise RetryableToolError(exc.reason) from None
            raise PermanentToolError(exc.reason) from None
        return ToolObservation(
            tool_name=self.definition.name,
            status="success" if result.hits else "no_result",
            data=result.model_dump(mode="json"),
            confidence=(
                max(hit.relevance_score for hit in result.hits)
                if result.hits
                else None
            ),
            evidence=[
                EvidenceReference(
                    business_id=hit.business_id,
                    source_type="review",
                    review_id=hit.review_id,
                )
                for hit in result.hits
            ],
            input_tokens=result.embedding_usage.input_tokens,
            output_tokens=0,
            cache_hit=(
                result.embedding_usage.cache_misses == 0
                and result.embedding_usage.cache_hits > 0
            ),
            warnings=(
                ["no Review evidence existed inside scope before cutoff"]
                if not result.hits
                else ["local embedding inference; no external API calls"]
                if result.embedding_usage.encoder_calls > 0
                else []
            ),
        )
