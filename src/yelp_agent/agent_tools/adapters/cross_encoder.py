"""Agent tool Adapter for Step 26 Cross-Encoder evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from yelp_agent.cross_encoder import (
    CrossEncoderMatchResult,
    CrossEncoderProviderError,
)
from yelp_agent.semantic_embedding import fuse_hybrid_and_semantic

from ..errors import PermanentToolError, RetryableToolError
from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import CandidateBusinessIdsInput, CrossEncoderMatchOutput


class CrossEncoderRerankingService(Protocol):
    def rerank(
        self,
        *,
        query_text: str,
        business_ids: Sequence[str],
        cutoff_time: datetime,
        usage_scope: str | None = None,
    ) -> CrossEncoderMatchResult: ...


class ComputeCrossEncoderMatchTool:
    definition = ToolDefinition(
        name="COMPUTE_CROSS_ENCODER_MATCH",
        version="1.0.0",
        kind="semantic",
        allowed_actions=("rank_candidates",),
        input_model=CandidateBusinessIdsInput,
        output_model=CrossEncoderMatchOutput,
        public_summary="Score the Step-25 ranking prefix with a local Cross-Encoder.",
        preconditions=(
            "GET_HYBRID_RANKING observation exists",
            "COMPUTE_EMBEDDING_MATCH observation exists",
            "business IDs form a prefix of the Step-25 fused ranking",
        ),
        resolves_uncertainties=("fine_grained_current_query_match",),
        cache_scope="none",
        max_attempts=1,
        timeout_ms=90_000,
        fallback_policy="safe_step25_then_hybrid_v2",
    )

    def __init__(
        self,
        service: CrossEncoderRerankingService,
        *,
        embedding_alpha: float,
    ) -> None:
        if not 0 <= embedding_alpha <= 1:
            raise ValueError("embedding alpha must be between zero and one")
        self._service = service
        self._embedding_alpha = embedding_alpha

    def run(
        self,
        arguments: CandidateBusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        expected_ranking = self._step25_ranking(context)
        if arguments.business_ids != expected_ranking[: len(arguments.business_ids)]:
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="STEP25_PREFIX_REQUIRED",
                warning="Cross-Encoder candidates must be the Step-25 ranking prefix",
            )
        try:
            result = self._service.rerank(
                query_text=self._query_text(context),
                business_ids=arguments.business_ids,
                cutoff_time=context.cutoff_time,
                usage_scope=context.request_id,
            )
        except CrossEncoderProviderError as exc:
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
                ["all Cross-Encoder scores served from local cache"]
                if result.usage.cache_misses == 0
                else ["Cross-Encoder scores computed by the local model"]
            ),
        )

    @staticmethod
    def _query_text(context: ToolExecutionContext) -> str:
        request = context.state_snapshot.get("request")
        query_text = request.get("query_text") if isinstance(request, dict) else None
        if not isinstance(query_text, str) or not query_text.strip():
            raise PermanentToolError("visible request query text is missing")
        return query_text

    def _step25_ranking(self, context: ToolExecutionContext) -> list[str]:
        observations = context.state_snapshot.get("observations")
        if not isinstance(observations, list):
            return []
        hybrid: list[str] = []
        semantic_ranks: dict[str, int] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload")
            if not isinstance(payload, dict):
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            if payload.get("tool_name") == "GET_HYBRID_RANKING":
                raw = data.get("ranking")
                if isinstance(raw, list):
                    hybrid = [value for value in raw if isinstance(value, str)]
            elif payload.get("tool_name") == "COMPUTE_EMBEDDING_MATCH":
                matches = data.get("matches")
                if isinstance(matches, list):
                    semantic_ranks = {
                        str(row["business_id"]): int(row["semantic_rank"])
                        for row in matches
                        if isinstance(row, dict)
                        and isinstance(row.get("business_id"), str)
                        and isinstance(row.get("semantic_rank"), int)
                    }
        if not hybrid:
            return []
        try:
            return fuse_hybrid_and_semantic(
                hybrid, semantic_ranks, alpha=self._embedding_alpha
            )
        except ValueError as exc:
            raise PermanentToolError(f"invalid Step-25 ranking evidence: {exc}") from None
