"""Adapter connecting canonical memory to the existing request-interpreter seam."""

from __future__ import annotations

from yelp_agent.agent_harness.interpreter import RuleBasedRequestInterpreter
from yelp_agent.agent_harness.schema import AgentSession, TurnInterpretation
from yelp_agent.query import QueryParseInput

from .manager import SessionMemoryManager
from .schema import MemoryTurnInput


class MemoryAwareRequestInterpreter:
    """Interpret only the new utterance, then patch canonical session memory."""

    def __init__(
        self,
        manager: SessionMemoryManager,
        *,
        baseline: RuleBasedRequestInterpreter | None = None,
    ) -> None:
        self._manager = manager
        self._baseline = baseline or RuleBasedRequestInterpreter()

    def interpret(
        self,
        value: QueryParseInput,
        *,
        previous_session: AgentSession | None = None,
    ) -> TurnInterpretation:
        # Deliberately do not concatenate the previous raw query. The manager
        # applies an explicit RequestPatch to canonical state instead.
        baseline = self._baseline.interpret(value, previous_session=None)
        current_turn = 1 if previous_session is None else previous_session.current_turn + 1
        result = self._manager.update(
            MemoryTurnInput(
                query_text=value.query_text,
                language=(
                    _infer_language(value.query_text)
                    if previous_session is None
                    else previous_session.language
                ),
                current_turn=current_turn,
                base_request=baseline.request,
                base_readiness=baseline.readiness,
                previous_memory=(
                    None if previous_session is None else previous_session.memory
                ),
                explicit_referenced_business_ids=value.referenced_business_ids,
            )
        )
        trace = result.extraction
        return TurnInterpretation(
            request=result.request,
            readiness=result.readiness,
            semantic_calls=int(trace.provider_called),
            input_tokens=trace.input_tokens if trace.provider_called else None,
            output_tokens=trace.output_tokens if trace.provider_called else None,
            cost_usd=None,
            memory=result.memory,
            memory_extraction=trace,
        )


def _infer_language(text: str) -> str:
    return "zh-CN" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en-US"
