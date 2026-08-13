"""Prompt rendering for a choice-only Router model."""

from __future__ import annotations

import json

from yelp_agent.agent.llm import LLMMessage

from .schema import RouterDecisionContext

_SYSTEM = """You are a constrained next-action Router for a Yelp recommendation Agent.
Choose exactly one choice_id from the supplied code-owned choices. You are not answering
the user and must not invent an action, tool, argument, business, fact, or threshold.
Prioritize the latest user message together with the complete effective session request.
Ask a clarification only when the missing information truly blocks a reliable action.
Prefer evidence-gathering before factual answers, and never bypass explicit constraints.
Return JSON only with exactly: choice_id, confidence, reason_code. reason_code must be a
short UPPER_SNAKE_CASE label and must not contain private reasoning."""


def router_messages(
    context: RouterDecisionContext,
    *,
    repair: bool = False,
) -> list[LLMMessage]:
    suffix = (
        "\nYour previous response was invalid or selected an unavailable choice. "
        "Return one valid choice_id copied exactly from choices."
        if repair
        else ""
    )
    payload = context.model_dump(mode="json")
    return [
        LLMMessage(role="system", content=_SYSTEM + suffix),
        LLMMessage(
            role="user",
            content=json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    ]
