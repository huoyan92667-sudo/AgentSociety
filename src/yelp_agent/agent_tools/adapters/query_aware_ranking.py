"""Compound Agent tool exposing the complete frozen Step-33 ranker."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from yelp_agent.query import RecommendationRequest

from ..registry import ToolDefinition
from ..request_context import (
    rejected_business_ids_from_tool_context,
    request_from_tool_context,
)
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import EmptyToolInput, QueryAwareRankingOutput


class QueryAwareRankingService(Protocol):
    def rank(
        self,
        *,
        request: RecommendationRequest,
        case_id: str,
        split: Literal["development", "validation"],
        usage_scope: str,
        rejected_business_ids: set[str] | frozenset[str],
    ) -> Any: ...


class GetQueryAwareRankingTool:
    """Replace candidate retrieval and every legacy ranking step with Step 33."""

    definition = ToolDefinition(
        name="GET_QUERY_AWARE_RANKING",
        version="1.0.0",
        kind="semantic",
        allowed_actions=("retrieve_candidates",),
        input_model=EmptyToolInput,
        output_model=QueryAwareRankingOutput,
        public_summary=(
            "Run protected History/Query retrieval, hard filtering, frozen "
            "LambdaMART coarse ranking and local Qwen3 reranking in one call."
        ),
        preconditions=("current structured recommendation request exists",),
        resolves_uncertainties=(
            "candidate_set_missing",
            "current_query_ranking_effect",
            "semantic_ranking_effect",
        ),
        cache_scope="request",
        max_attempts=1,
        timeout_ms=180_000,
        fallback_policy="step33_internal_coarse_ranking",
    )

    def __init__(self, service: QueryAwareRankingService) -> None:
        self._service = service

    def run(
        self,
        arguments: EmptyToolInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del arguments
        try:
            request = request_from_tool_context(context)
        except (TypeError, ValueError):
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="REQUEST_MISSING",
                warning="canonical request is missing from visible state",
            )
        raw_split = str(context.state_snapshot.get("split") or "development")
        if raw_split not in {"development", "validation"}:
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="INVALID_SPLIT",
                warning="query-aware ranking requires a benchmark-compatible split",
            )
        split: Literal["development", "validation"] = raw_split  # type: ignore[assignment]
        case_id = str(
            context.state_snapshot.get("scenario_id") or context.request_id
        )
        result = self._service.rank(
            request=request,
            case_id=case_id,
            split=split,
            usage_scope=f"agent-step33:{case_id}",
            rejected_business_ids=rejected_business_ids_from_tool_context(context),
        )
        payload = result.model_dump(mode="json")
        payload["candidate_business_ids"] = list(result.ranking)
        validated = QueryAwareRankingOutput.model_validate(payload)
        warnings = []
        if result.fallback:
            warnings.append(result.fallback_reason or "STEP33_INTERNAL_FALLBACK")
        return ToolObservation(
            tool_name=self.definition.name,
            status="partial" if result.fallback else "success",
            data=validated.model_dump(mode="json"),
            confidence=1.0,
            warnings=warnings,
            input_tokens=(
                result.embedding_input_tokens
                + result.cross_encoder_input_tokens
            ),
            output_tokens=0,
            latency_ms=result.latency_ms,
            cache_hit=result.cache_misses == 0,
        )
