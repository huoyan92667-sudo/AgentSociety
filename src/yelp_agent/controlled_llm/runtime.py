"""Assembly and lifetime ownership for controlled OpenAI-compatible calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig, load_llm_environment

from .answer import GroundedAnswerComposer
from .cache import SqliteControlledLLMCache
from .config import ControlledLLMConfig
from .gateway import ControlledJSONCaller
from .ledger import ControlledLLMUsageLedger
from .semantic import ControlledSemanticEnhancer


@dataclass(slots=True)
class ControlledLLMRuntime:
    semantic_enhancer: ControlledSemanticEnhancer
    answer_composer: GroundedAnswerComposer
    ledger: ControlledLLMUsageLedger
    cache: SqliteControlledLLMCache

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def build_controlled_llm_runtime(
    *,
    project_root: str | Path,
    config: ControlledLLMConfig,
    environment: Mapping[str, str] | None = None,
) -> ControlledLLMRuntime:
    env = load_llm_environment(environment)
    cache = SqliteControlledLLMCache(Path(project_root) / config.cache_relative_path)
    ledger = ControlledLLMUsageLedger()
    semantic_client = OpenAICompatibleLLM(
        AgentConfig(
            enabled=config.semantic.enabled,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            max_tokens=config.semantic.max_output_tokens,
            response_format_json=config.response_format_json,
            thinking=config.thinking,
        ),
        env,
    )
    answer_client = OpenAICompatibleLLM(
        AgentConfig(
            enabled=config.answer.enabled,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            max_tokens=config.answer.max_output_tokens,
            response_format_json=config.response_format_json,
            thinking=config.thinking,
        ),
        env,
    )
    semantic_caller = ControlledJSONCaller(
        generator=semantic_client,
        model_name=env.model,
        cache=cache,
        ledger=ledger,
    )
    answer_caller = ControlledJSONCaller(
        generator=answer_client,
        model_name=env.model,
        cache=cache,
        ledger=ledger,
    )
    return ControlledLLMRuntime(
        semantic_enhancer=ControlledSemanticEnhancer(
            config=config.semantic,
            caller=semantic_caller,
        ),
        answer_composer=GroundedAnswerComposer(
            config=config.answer,
            caller=answer_caller,
        ),
        ledger=ledger,
        cache=cache,
    )
