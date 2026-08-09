"""Optional language-only rewriting seam for frozen Step 20 semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from yelp_agent.agent.llm import LLMCallResult, LLMMessage

from .schema import ScenarioLanguage


@dataclass(frozen=True, slots=True)
class RewriteResult:
    text: str
    generator_kind: str
    generator_model: str | None = None
    prompt_sha256: str | None = None


class ScenarioRewriter(Protocol):
    """Vary phrasing without receiving hidden labels or Yelp evidence."""

    def rewrite(
        self,
        *,
        scenario_id: str,
        language: ScenarioLanguage,
        visible_query: str,
    ) -> RewriteResult: ...


class _LanguageModel(Protocol):
    def generate(self, messages: tuple[LLMMessage, ...]) -> LLMCallResult: ...


class DeterministicScenarioRewriter:
    def rewrite(
        self,
        *,
        scenario_id: str,
        language: ScenarioLanguage,
        visible_query: str,
    ) -> RewriteResult:
        del scenario_id, language
        return RewriteResult(text=visible_query, generator_kind="deterministic")


class FakeScenarioRewriter:
    """Test adapter proving rewriting can vary without provider access."""

    def rewrite(
        self,
        *,
        scenario_id: str,
        language: ScenarioLanguage,
        visible_query: str,
    ) -> RewriteResult:
        marker = "请注意，" if language == "zh-CN" else "Please note: "
        del scenario_id
        return RewriteResult(
            text=f"{marker}{visible_query}",
            generator_kind="fake",
        )


class OpenAICompatibleScenarioRewriter:
    """Explicitly injected provider adapter that never receives hidden answers."""

    def __init__(self, llm: _LanguageModel, *, model: str) -> None:
        if not model:
            raise ValueError("rewriter model cannot be empty")
        self._llm = llm
        self._model = model

    def rewrite(
        self,
        *,
        scenario_id: str,
        language: ScenarioLanguage,
        visible_query: str,
    ) -> RewriteResult:
        del scenario_id
        system = (
            "Rewrite only the visible user query in the requested language. "
            "Preserve every requirement, negation, reference, and uncertainty. "
            "Do not add facts, businesses, answers, labels, actions, or evidence. "
            "Return JSON only as {\"text\": \"...\"}."
        )
        user = json.dumps(
            {"language": language, "visible_query": visible_query},
            ensure_ascii=False,
            sort_keys=True,
        )
        prompt_hash = hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()
        result = self._llm.generate(
            (
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            )
        )
        if result.status != "success" or result.content is None:
            raise RuntimeError(
                "scenario rewrite failed: "
                f"{result.failure_reason or result.status}"
            )
        try:
            payload = json.loads(result.content)
            text = str(payload["text"]).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("scenario rewriter returned invalid JSON") from exc
        if not text or len(text) > 2000:
            raise ValueError("scenario rewriter returned an invalid text length")
        return RewriteResult(
            text=text,
            generator_kind="openai_compatible",
            generator_model=self._model,
            prompt_sha256=prompt_hash,
        )
