"""Deterministic adapters for testing the harness without Yelp data or APIs."""

from __future__ import annotations

from collections.abc import Sequence

from yelp_agent.agent_benchmark.schema import AgentAction

from .schema import ActionOutcome, AgentDecision, AgentSession


class FakeClock:
    """Return a deterministic arithmetic sequence of monotonic timestamps."""

    def __init__(self, *, start_ms: float = 0.0, step_ms: float = 0.0) -> None:
        self._next_ms = start_ms
        self._step_ms = step_ms

    def now_ms(self) -> float:
        value = self._next_ms
        self._next_ms += self._step_ms
        return value


class ScriptedActionPolicy:
    """Expose the same allowed actions on every harness step."""

    def __init__(self, actions: Sequence[AgentAction]) -> None:
        if not actions:
            raise ValueError("scripted action policy requires at least one action")
        self._actions = tuple(actions)

    def allowed_actions(self, state: AgentSession) -> tuple[AgentAction, ...]:
        del state
        return self._actions


class ScriptedRouter:
    """Select decisions by the session's total Router step count."""

    def __init__(self, decisions: Sequence[AgentDecision]) -> None:
        if not decisions:
            raise ValueError("scripted Router requires at least one decision")
        self._decisions = tuple(decisions)

    def choose_action(self, state: AgentSession) -> AgentDecision:
        try:
            return self._decisions[state.step_count]
        except IndexError as exc:
            raise RuntimeError("scripted Router exhausted") from exc


class ScriptedExecutor:
    """Return or raise scripted results in execution order."""

    def __init__(self, outcomes: Sequence[ActionOutcome | Exception]) -> None:
        if not outcomes:
            raise ValueError("scripted executor requires at least one outcome")
        self._outcomes = tuple(outcomes)
        self.call_count = 0

    def execute(
        self,
        state: AgentSession,
        decision: AgentDecision,
    ) -> ActionOutcome:
        del state, decision
        try:
            outcome = self._outcomes[self.call_count]
        except IndexError as exc:
            raise RuntimeError("scripted executor exhausted") from exc
        self.call_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeFallbackHandler:
    """Return a deterministic ranking and expose how often fallback ran."""

    def __init__(self, ranking: Sequence[str] = ()) -> None:
        self._ranking = list(ranking)
        self.call_count = 0
        self.reasons: list[str] = []

    def fallback(self, state: AgentSession, reason: str) -> ActionOutcome:
        del state
        self.call_count += 1
        self.reasons.append(reason)
        return ActionOutcome(
            status="completed",
            response_kind="fallback",
            candidate_ranking=self._ranking,
            recommended_business_ids=self._ranking[:1],
        )
