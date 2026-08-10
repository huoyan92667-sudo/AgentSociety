"""User-profile adapter using exact session identity and cutoff."""

from __future__ import annotations

from typing import Protocol

from yelp_agent.profiles.schema import UserProfileV1

from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import EmptyToolInput, UserProfileOutput


class UserProfileReader(Protocol):
    def get(self, user_id: str, cutoff_time: object) -> UserProfileV1: ...


class GetUserProfileTool:
    definition = ToolDefinition(
        name="GET_USER_PROFILE",
        version="1.0.0",
        kind="deterministic",
        allowed_actions=(
            "retrieve_candidates",
            "rank_candidates",
            "compare_candidates",
            "apply_feedback",
        ),
        input_model=EmptyToolInput,
        output_model=UserProfileOutput,
        public_summary=(
            "Load the current user's frozen profile at the session cutoff."
        ),
        preconditions=("exact user_id and cutoff_time exist",),
        resolves_uncertainties=("user_preference_context",),
        cache_scope="request",
    )

    def __init__(self, store: UserProfileReader) -> None:
        self._store = store

    def run(
        self,
        arguments: EmptyToolInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del arguments
        profile = self._store.get(context.user_id, context.cutoff_time)
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={"profile": profile.model_dump()},
            confidence=profile.reliability,
        )
