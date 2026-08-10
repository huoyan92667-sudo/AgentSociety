"""Deterministic, request-aware comparison for a small scoped set."""

from __future__ import annotations

from yelp_agent.query.ranking import QueryAwareStaticRanker

from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import BusinessComparisonOutput, CompareBusinessesInput
from .constraints import ConstraintCandidateReader, request_from_context


class CompareBusinessesTool:
    definition = ToolDefinition(
        name="COMPARE_BUSINESSES",
        version="1.0.0",
        kind="deterministic",
        allowed_actions=("compare_candidates",),
        input_model=CompareBusinessesInput,
        output_model=BusinessComparisonOutput,
        public_summary=(
            "Compare two to ten scoped businesses using current-request rules."
        ),
        preconditions=(
            "structured request exists",
            "two to ten business IDs are inside current scope",
        ),
        resolves_uncertainties=("candidate_tradeoff",),
    )

    def __init__(self, reader: ConstraintCandidateReader) -> None:
        self._reader = reader
        self._ranker = QueryAwareStaticRanker()

    def run(
        self,
        arguments: CompareBusinessesInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        request = request_from_context(context)
        candidates = self._reader.get_candidates(
            arguments.business_ids,
            context.cutoff_time,
        )
        result = self._ranker.rank(request, candidates, mode="query_only")
        compared = [
            {
                "business_id": item.business_id,
                "rank": item.rank,
                "query_score": item.query_score,
                "matched_fields": item.matched_fields,
                "unmatched_fields": item.unmatched_fields,
                "unknown_fields": item.unknown_fields,
            }
            for item in result.ranking
        ]
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={
                "ranking": [item.business_id for item in result.ranking],
                "compared": compared,
                "excluded": [item.model_dump() for item in result.excluded],
            },
            confidence=(
                0.0
                if not compared
                else sum(1.0 - 0.5 * bool(item["unknown_fields"]) for item in compared)
                / len(compared)
            ),
            warnings=result.warnings,
        )
