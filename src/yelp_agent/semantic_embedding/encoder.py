"""DashScope OpenAI-compatible embedding Adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from .config import DashScopeEmbeddingEnvironment, SemanticEmbeddingConfig
from .schema import EmbeddingInputType


class EmbeddingProviderError(RuntimeError):
    """Secret-free provider failure with an explicit retry disposition."""

    def __init__(self, reason: str, *, retryable: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    vectors: tuple[np.ndarray, ...]
    model: str
    input_tokens: int
    request_id: str | None
    latency_ms: float
    per_text_input_tokens: tuple[int, ...] | None = None
    truncated_text_count: int = 0

    def __post_init__(self) -> None:
        if self.per_text_input_tokens is None:
            return
        if len(self.per_text_input_tokens) != len(self.vectors):
            raise ValueError("per-text token counts must match encoded vectors")
        if any(value < 0 for value in self.per_text_input_tokens):
            raise ValueError("per-text token counts cannot be negative")
        if sum(self.per_text_input_tokens) != self.input_tokens:
            raise ValueError("per-text token counts must sum to input_tokens")


class EmbeddingEncoder(Protocol):
    provider: str
    model: str
    dimension: int
    batch_size: int

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EncodedBatch: ...


class DashScopeEmbeddingEncoder:
    """Hide SDK arguments, provider errors, and response validation."""

    provider = "dashscope"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        config: SemanticEmbeddingConfig,
    ) -> None:
        self._client = client
        self.model = model
        self.dimension = config.dimension
        self.batch_size = config.batch_size
        self._timeout_seconds = config.timeout_seconds
        self._instruction = config.query_instruction
        self._output_type = config.output_type

    @classmethod
    def from_environment(
        cls,
        config: SemanticEmbeddingConfig,
        environment: DashScopeEmbeddingEnvironment,
    ) -> DashScopeEmbeddingEncoder:
        if not environment.enabled or environment.api_key is None:
            raise ValueError("DashScope embedding environment is not configured")
        return cls(
            client=OpenAI(
                api_key=environment.api_key.get_secret_value(),
                base_url=environment.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            ),
            model=str(environment.model),
            config=config,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EncodedBatch:
        values = [text.strip() for text in texts]
        if not values or any(not text for text in values):
            raise ValueError("embedding inputs must be nonempty text")
        if len(values) > self.batch_size:
            raise ValueError("embedding batch exceeds configured provider limit")
        extra_body: dict[str, object] = {
            "text_type": input_type,
            "output_type": self._output_type,
        }
        if input_type == "query":
            extra_body["instruct"] = self._instruction
        started = perf_counter()
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=values,
                dimensions=self.dimension,
                encoding_format="float",
                extra_body=extra_body,
                timeout=self._timeout_seconds,
            )
        except OpenAIError as exc:
            raise self._translate_error(exc) from None
        latency_ms = (perf_counter() - started) * 1000.0
        rows = sorted(response.data, key=lambda item: int(item.index))
        if len(rows) != len(values) or [int(item.index) for item in rows] != list(
            range(len(values))
        ):
            raise EmbeddingProviderError("malformed_response", retryable=False)
        vectors: list[np.ndarray] = []
        for item in rows:
            vector = np.asarray(item.embedding, dtype=np.float32)
            if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
                raise EmbeddingProviderError("invalid_vector", retryable=False)
            vectors.append(vector)
        usage = getattr(response, "usage", None)
        input_tokens = int(
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "total_tokens", None)
            or 0
        )
        request_id = getattr(response, "_request_id", None)
        if request_id is None:
            request_id = getattr(response, "id", None)
        return EncodedBatch(
            vectors=tuple(vectors),
            model=str(getattr(response, "model", self.model)),
            input_tokens=input_tokens,
            request_id=request_id if isinstance(request_id, str) else None,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _translate_error(exc: OpenAIError) -> EmbeddingProviderError:
        if isinstance(exc, APITimeoutError):
            return EmbeddingProviderError("timeout", retryable=True)
        if isinstance(exc, APIConnectionError):
            return EmbeddingProviderError("api_connection", retryable=True)
        if isinstance(exc, RateLimitError):
            return EmbeddingProviderError("rate_limit", retryable=True)
        if isinstance(exc, AuthenticationError):
            return EmbeddingProviderError("authentication_error", retryable=False)
        if isinstance(exc, PermissionDeniedError):
            return EmbeddingProviderError("permission_error", retryable=False)
        if isinstance(exc, BadRequestError):
            return EmbeddingProviderError("bad_request", retryable=False)
        if isinstance(exc, APIStatusError):
            retryable = exc.status_code in {408, 409, 429} or exc.status_code >= 500
            return EmbeddingProviderError(
                "server_error" if retryable else "api_error",
                retryable=retryable,
            )
        return EmbeddingProviderError("api_error", retryable=False)
