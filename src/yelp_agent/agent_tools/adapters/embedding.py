"""Agent tool Adapter for Step 25 semantic matching evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from yelp_agent.semantic_embedding import EmbeddingProviderError, SemanticMatchResult

from ..errors import PermanentToolError, RetryableToolError
from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import CandidateBusinessIdsInput, EmbeddingMatchOutput


class SemanticMatchService(Protocol):
    def match(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
    ) -> SemanticMatchResult: ...


class ComputeEmbeddingMatchTool:
    """Return query-candidate semantic evidence without choosing final results."""

    definition = ToolDefinition(
        name="COMPUTE_EMBEDDING_MATCH",
        version="1.0.0",
        kind="semantic",
        allowed_actions=("rank_candidates",),
        input_model=CandidateBusinessIdsInput,
        output_model=EmbeddingMatchOutput,
        public_summary=(
            "Compute cutoff-safe query-to-business semantic similarity evidence."
        ),
        preconditions=(
            "GET_HYBRID_RANKING observation exists",
            "business IDs form a prefix of the current Hybrid ranking",
            "business IDs are inside current scope",
        ),
        resolves_uncertainties=("current_query_semantic_match",),
        cache_scope="none",
        max_attempts=1,
        timeout_ms=90_000,
        fallback_policy="safe_hybrid_v2",
    )

    def __init__(self, service: SemanticMatchService) -> None:
        self._service = service

    def run(
        self,
        arguments: CandidateBusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        query_text = self._query_text(context)
        hybrid_ranking = self._hybrid_ranking(context)
        if arguments.business_ids != hybrid_ranking[: len(arguments.business_ids)]:
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="HYBRID_PREFIX_REQUIRED",
                warning="semantic candidates must be the current Hybrid prefix",
            )
        try:
            result = self._service.match(
                query_text=query_text,
                business_ids=arguments.business_ids,
                cutoff_time=context.cutoff_time,
            )
        except EmbeddingProviderError as exc:
            if exc.retryable:
                raise RetryableToolError(exc.reason) from None
            raise PermanentToolError(exc.reason) from None
        return ToolObservation(
            tool_name=self.definition.name,
            status="success",
            data=result.model_dump(),
            input_tokens=result.usage.input_tokens,
            output_tokens=0,
            cache_hit=result.usage.cache_misses == 0,
            warnings=(
                []
                if result.usage.api_calls > 0
                else ["all embedding vectors served from local cache"]
            ),
        )

    @staticmethod
    def _query_text(context: ToolExecutionContext) -> str:
        request = context.state_snapshot.get("request")
        query_text = request.get("query_text") if isinstance(request, dict) else None
        if not isinstance(query_text, str) or not query_text.strip():
            raise PermanentToolError("visible request query text is missing")
        return query_text

    @staticmethod
    def _hybrid_ranking(context: ToolExecutionContext) -> list[str]:
        observations = context.state_snapshot.get("observations")
        if not isinstance(observations, list):
            return []
        for observation in reversed(observations):
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload")
            if not isinstance(payload, dict) or payload.get("tool_name") != (
                "GET_HYBRID_RANKING"
            ):
                continue
            data = payload.get("data")
            ranking = data.get("ranking") if isinstance(data, dict) else None
            if isinstance(ranking, list):
                return [value for value in ranking if isinstance(value, str)]
        return []
