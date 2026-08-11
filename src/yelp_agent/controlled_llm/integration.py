"""Adapters connecting the two Step 29 modules to the existing Agent seams."""

from __future__ import annotations

from yelp_agent.agent_harness.interpreter import RuleBasedRequestInterpreter
from yelp_agent.agent_harness.schema import AgentSession, TurnInterpretation
from yelp_agent.query import QueryParseInput
from yelp_agent.query.schema import RecommendationRequest

from .schema import SemanticEnhancementInput
from .semantic import ControlledSemanticEnhancer


class ControlledRequestInterpreter:
    """Rule-first interpretation with an optional, failure-safe semantic pass."""

    def __init__(
        self,
        enhancer: ControlledSemanticEnhancer,
        *,
        baseline: RuleBasedRequestInterpreter | None = None,
    ) -> None:
        self._enhancer = enhancer
        self._baseline = baseline or RuleBasedRequestInterpreter()

    def interpret(
        self,
        value: QueryParseInput,
        *,
        previous_session: AgentSession | None = None,
    ) -> TurnInterpretation:
        baseline = self._baseline.interpret(
            value,
            previous_session=previous_session,
        )
        result = self._enhancer.enhance(
            SemanticEnhancementInput(
                base_request=baseline.request,
                base_readiness=baseline.readiness,
                language=(
                    previous_session.language
                    if previous_session is not None
                    else _infer_language(baseline.request.query_text)
                ),
            )
        )
        trace = result.trace
        usage_known = trace.provider_called and trace.input_tokens is not None
        return TurnInterpretation(
            request=result.request,
            readiness=result.readiness,
            semantic_calls=int(trace.provider_called),
            input_tokens=trace.input_tokens if usage_known else None,
            output_tokens=trace.output_tokens if usage_known else None,
            cost_usd=None,
        )


class ControlledParserAdapter:
    """Expose the controlled interpreter to the frozen Query Parser benchmark."""

    def __init__(self, interpreter: ControlledRequestInterpreter) -> None:
        self._interpreter = interpreter
        self.version = "rule-first+step29-controlled-semantic"

    def parse(self, value: QueryParseInput) -> RecommendationRequest:
        return self._interpreter.interpret(value).request


def _infer_language(text: str) -> str:
    return "zh-CN" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en-US"
