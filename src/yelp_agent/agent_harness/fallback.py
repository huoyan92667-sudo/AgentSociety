"""Safe, provider-independent terminal fallback behavior."""

from __future__ import annotations

from .schema import ActionOutcome, AgentSession


class StaticFallbackHandler:
    """Return a valid empty fallback until Step 23 wires Hybrid V2."""

    def fallback(self, state: AgentSession, reason: str) -> ActionOutcome:
        del state, reason
        return ActionOutcome(status="completed", response_kind="fallback")
