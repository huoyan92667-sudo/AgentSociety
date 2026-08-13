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
        memory_context = snapshot.get("memory_context")
        effective_request = snapshot.get("effective_request")
        return ToolObservation.success(
            tool_name=self.definition.name,
            data={
                "session_id": str(snapshot.get("session_id") or "unknown-session"),
                "turn_index": int(snapshot.get("turn_index") or 1),
                # Canonical memory replaces the old unbounded observation dump.
                # Legacy sessions without Step 34 keep the previous behavior.
                "observations": (
                    []
                    if isinstance(memory_context, dict)
                    else list(snapshot.get("observations") or [])
                ),
                "memory_context": (
                    memory_context if isinstance(memory_context, dict) else None
                ),
                "effective_request": (
                    effective_request
                    if isinstance(effective_request, dict)
                    else None
                ),
            },
            confidence=1.0,
        )
