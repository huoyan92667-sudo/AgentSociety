"""Small dependency interfaces used by the Agent harness."""

from __future__ import annotations

from typing import Protocol, Sequence

from yelp_agent.query.schema import QueryParseInput

from .schema import (
    ActionOutcome,
    AgentDecision,
    AgentSession,
    TurnInterpretation,
)


class Clock(Protocol):
    def now_ms(self) -> float: ...


class RequestInterpreter(Protocol):
    def interpret(
        self,
        value: QueryParseInput,
        *,
        previous_session: AgentSession | None = None,
    ) -> TurnInterpretation: ...


class AllowedActionPolicy(Protocol):
    def allowed_actions(self, state: AgentSession) -> Sequence[str]: ...


class ActionRouter(Protocol):
    def choose_action(self, state: AgentSession) -> AgentDecision: ...


class ActionExecutor(Protocol):
    def execute(
        self,
        state: AgentSession,
        decision: AgentDecision,
    ) -> ActionOutcome: ...


class FallbackHandler(Protocol):
    def fallback(self, state: AgentSession, reason: str) -> ActionOutcome: ...
