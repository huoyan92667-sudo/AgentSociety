"""Assembly and resource ownership for the Step 34 memory module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from yelp_agent.agent.llm import OpenAICompatibleLLM
from yelp_agent.config import AgentConfig, load_llm_environment
from yelp_agent.controlled_llm.cache import SqliteControlledLLMCache
from yelp_agent.controlled_llm.gateway import ControlledJSONCaller
from yelp_agent.controlled_llm.ledger import ControlledLLMUsageLedger

from .config import SessionMemoryConfig
from .extractor import DeepSeekMemoryExtractor
from .manager import SessionMemoryManager


@dataclass(slots=True)
class SessionMemoryRuntime:
    manager: SessionMemoryManager
    ledger: ControlledLLMUsageLedger
    cache: SqliteControlledLLMCache

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def build_session_memory_runtime(
    *,
    project_root: str | Path,
    config: SessionMemoryConfig,
    environment: Mapping[str, str] | None = None,
) -> SessionMemoryRuntime:
    env = load_llm_environment(environment)
    cache = SqliteControlledLLMCache(Path(project_root) / config.cache_relative_path)
    ledger = ControlledLLMUsageLedger()
    generator = OpenAICompatibleLLM(
        AgentConfig(
            enabled=config.enabled,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            max_tokens=config.max_output_tokens,
            response_format_json=config.response_format_json,
            thinking=config.thinking,
        ),
        env,
    )
    caller = ControlledJSONCaller(
        generator=generator,
        model_name=env.model,
        cache=cache,
        ledger=ledger,
    )
    manager = SessionMemoryManager(
        config=config,
        primary_extractor=DeepSeekMemoryExtractor(caller=caller, config=config),
    )
    return SessionMemoryRuntime(manager=manager, ledger=ledger, cache=cache)
