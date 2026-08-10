"""Point-in-time business detail tools."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from yelp_agent.business_profiles.schema import BusinessProfileV1

from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import (
    BusinessDetailsOutput,
    BusinessIdsInput,
    BusinessProfilesOutput,
)


class BusinessProfileReader(Protocol):
    def get(
        self,
        business_ids: list[str],
        cutoff_time: datetime,
    ) -> dict[str, BusinessProfileV1]: ...


class GetBusinessDetailsTool:
    definition = ToolDefinition(
        name="GET_BUSINESS_DETAILS",
        version="1.0.0",
        kind="deterministic",
        allowed_actions=("get_business_details", "compare_candidates"),
        input_model=BusinessIdsInput,
        output_model=BusinessDetailsOutput,
        public_summary=(
            "Read static fields and cutoff-safe quality for scoped businesses."
        ),
        preconditions=("business IDs are inside current scope",),
        resolves_uncertainties=("business_static_details",),
        cache_scope="request",
    )

    def __init__(self, store: BusinessProfileReader) -> None:
        self._store = store

    def run(
        self,
        arguments: BusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        profiles = self._store.get(arguments.business_ids, context.cutoff_time)
        rows = []
        for business_id in arguments.business_ids:
            profile = profiles[business_id]
            rows.append(
                {
                    "business_id": business_id,
                    "name": profile.name,
                    "address": profile.address,
                    "city": profile.city,
                    "state": profile.state,
                    "postal_code": profile.postal_code,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                    "categories": profile.categories,
                    "structured_attributes": profile.structured_attributes,
                    "quality_score": profile.quality.quality_score,
                    "review_count": profile.quality.review_count,
                }
            )
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"businesses": rows},
            confidence=min(profile.profile_reliability for profile in profiles.values()),
        )


class GetBusinessProfileTool:
    definition = ToolDefinition(
        name="GET_BUSINESS_PROFILE",
        version="1.0.0",
        kind="deterministic",
        allowed_actions=(
            "get_business_details",
            "rank_candidates",
            "compare_candidates",
        ),
        input_model=BusinessIdsInput,
        output_model=BusinessProfilesOutput,
        public_summary=(
            "Read cutoff-safe quality and aggregated aspect evidence for businesses."
        ),
        preconditions=("business IDs are inside current scope",),
        resolves_uncertainties=("business_profile_evidence",),
        cache_scope="request",
    )

    def __init__(self, store: BusinessProfileReader) -> None:
        self._store = store

    def run(
        self,
        arguments: BusinessIdsInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        profiles = self._store.get(arguments.business_ids, context.cutoff_time)
        ordered = [profiles[business_id] for business_id in arguments.business_ids]
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"profiles": [profile.model_dump() for profile in ordered]},
            confidence=min(profile.profile_reliability for profile in ordered),
        )
