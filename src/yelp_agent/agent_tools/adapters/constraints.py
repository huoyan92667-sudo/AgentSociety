"""Hard-constraint adapter shared with offline Query-aware ranking."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from yelp_agent.business_profiles.store import BusinessKnowledgeStore
from yelp_agent.query import RecommendationRequest, candidate_from_business_profile
from yelp_agent.query.ranking import (
    QueryAwareCandidate,
    hard_constraint_failures,
)

from ..registry import ToolDefinition
from ..request_context import request_from_tool_context
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import CandidateBusinessIdsInput, ConstraintOutput


class ConstraintCandidateReader(Protocol):
    def get_candidates(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> Sequence[QueryAwareCandidate]: ...


class BusinessProfileCandidateReader:
    """Convert exact-cutoff business profiles into constraint candidates."""

    def __init__(self, store: BusinessKnowledgeStore) -> None:
        self._store = store

    def get_candidates(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> list[QueryAwareCandidate]:
        profiles = self._store.get(business_ids, cutoff_time)
        return [
            candidate_from_business_profile(
                profiles[business_id],
                hybrid_rank=rank,
            )
            for rank, business_id in enumerate(business_ids, start=1)
        ]


class ApplyConstraintsTool:
    definition = ToolDefinition(
        name="APPLY_CONSTRAINTS",
        version="1.0.0",
        kind="deterministic",
        allowed_actions=("apply_hard_constraints",),
        input_model=CandidateBusinessIdsInput,
        output_model=ConstraintOutput,
        public_summary=(
            "Apply current-request hard constraints without expanding candidate scope."
        ),
        preconditions=(
            "structured request exists",
            "business IDs are inside current scope",
        ),
        resolves_uncertainties=("hard_constraint_eligibility",),
    )

    def __init__(self, reader: ConstraintCandidateReader) -> None:
        self._reader = reader

    def run(
        self,
        arguments: CandidateBusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        try:
            request = request_from_tool_context(context)
        except (TypeError, ValueError):
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="REQUEST_STATE_MISSING",
                warning="current structured request is absent from Agent state",
            )
        candidates = self._reader.get_candidates(
            arguments.business_ids,
            context.cutoff_time,
        )
        if [item.business_id for item in candidates] != arguments.business_ids:
            return ToolObservation.error(
                tool_name=self.definition.name,
                status="permanent_error",
                error_code="MALFORMED_CANDIDATE_DATA",
                warning="constraint candidate order does not match requested IDs",
            )
        eligible: list[str] = []
        excluded: list[dict[str, object]] = []
        for candidate in candidates:
            failures = hard_constraint_failures(request, candidate)
            if failures:
                excluded.append(
                    {
                        "business_id": candidate.business_id,
                        "reason_codes": failures,
                    }
                )
            else:
                eligible.append(candidate.business_id)
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"candidate_business_ids": eligible, "excluded": excluded},
            confidence=1.0,
        )


def request_from_context(context: ToolExecutionContext) -> RecommendationRequest:
    """Rebuild only canonical request fields from a visible state snapshot."""
    return request_from_tool_context(context)
