"""Read-only access to the current session's already-visible observations."""

from __future__ import annotations

from ..registry import ToolDefinition
from ..schema import ToolExecutionContext, ToolObservation
from ..tool_schemas import EmptyToolInput, SessionMemoryOutput


class GetSessionMemoryTool:
    definition = ToolDefinition(
        name="GET_SESSION_MEMORY",
        version="1.0.0",
        kind="deterministic",
        allowed_actions=("apply_feedback", "rank_candidates", "compare_candidates"),
        input_model=EmptyToolInput,
        output_model=SessionMemoryOutput,
        public_summary="Read observations already produced in this visible session.",
        preconditions=("visible session state exists",),
        resolves_uncertainties=("conversation_context",),
    )

    def run(
        self,
        arguments: EmptyToolInput,
        context: ToolExecutionContext,
    ) -> ToolObservation:
        del arguments
        snapshot = context.state_snapshot
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={
                "session_id": str(snapshot.get("session_id") or "unknown-session"),
                "turn_index": int(snapshot.get("turn_index") or 1),
                "observations": list(snapshot.get("observations") or []),
            },
            confidence=1.0,
        )
