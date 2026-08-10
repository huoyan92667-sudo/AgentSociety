"""Local Qwen3 Adapter backed by one persistent isolated Python worker."""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np

from .config import LocalEmbeddingEnvironment, SemanticEmbeddingConfig
from .encoder import EncodedBatch, EmbeddingProviderError
from .schema import EmbeddingInputType


@dataclass(frozen=True, slots=True)
class LocalTokenCount:
    per_text_tokens: tuple[int, ...]
    truncated_text_count: int

    @property
    def total_tokens(self) -> int:
        return sum(self.per_text_tokens)


class LocalQwenEmbeddingEncoder:
    provider = "local"

    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        model: str,
        dimension: int,
        config: SemanticEmbeddingConfig,
    ) -> None:
        self._process = process
        self.model = model
        self.dimension = dimension
        self.batch_size = config.batch_size
        self._instruction = config.query_instruction
        self._lock = Lock()
        self._request_index = 0

    @classmethod
    def from_environment(
        cls,
        config: SemanticEmbeddingConfig,
        environment: LocalEmbeddingEnvironment,
    ) -> LocalQwenEmbeddingEncoder:
        if not environment.enabled or environment.model_path is None:
            raise ValueError("local embedding environment is not configured")
        python_executable = environment.python_executable
        worker_path = Path(__file__).with_name("local_worker.py")
        process = subprocess.Popen(
            [
                str(python_executable),
                str(worker_path),
                "--model-path",
                str(environment.model_path),
                "--device",
                environment.device,
                "--max-sequence-length",
                str(config.max_sequence_length),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if not ready_line:
            detail = "worker_start_failed"
            if process.stderr is not None:
                detail = process.stderr.read()[-500:] or detail
            process.kill()
            raise EmbeddingProviderError(detail, retryable=False)
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError:
            process.kill()
            raise EmbeddingProviderError("malformed_worker_ready", retryable=False) from None
        if ready.get("status") != "ready":
            process.kill()
            raise EmbeddingProviderError("worker_start_failed", retryable=False)
        dimension = int(ready.get("dimension", 0))
        if dimension != config.dimension:
            process.kill()
            raise EmbeddingProviderError("local_dimension_mismatch", retryable=False)
        return cls(
            process=process,
            model=f"local:{ready.get('model', environment.model_path.name)}",
            dimension=dimension,
            config=config,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EncodedBatch:
        values = [_sanitize_text(text) for text in texts]
        if not values or any(not text for text in values):
            raise ValueError("embedding inputs must be nonempty text")
        if len(values) > self.batch_size:
            raise ValueError("embedding batch exceeds configured local limit")
        prepared = (
            [f"Instruct: {self._instruction}\nQuery:{text}" for text in values]
            if input_type == "query"
            else values
        )
        response, wall_latency_ms = self._send(
            {"operation": "encode", "texts": prepared}
        )
        request_id = str(response["request_id"])
        rows = int(response["rows"])
        dimension = int(response["dimension"])
        raw = base64.b64decode(str(response["vectors_b64"]), validate=True)
        array = np.frombuffer(raw, dtype="<f4").copy()
        if rows != len(values) or dimension != self.dimension:
            raise EmbeddingProviderError("invalid_local_vector_shape", retryable=False)
        array = array.reshape(rows, dimension)
        if not np.all(np.isfinite(array)):
            raise EmbeddingProviderError("invalid_local_vector", retryable=False)
        per_text_tokens = tuple(int(value) for value in response["per_text_input_tokens"])
        return EncodedBatch(
            vectors=tuple(array[index] for index in range(rows)),
            model=self.model,
            input_tokens=sum(per_text_tokens),
            request_id=request_id,
            latency_ms=wall_latency_ms,
            per_text_input_tokens=per_text_tokens,
            truncated_text_count=int(response["truncated_text_count"]),
        )

    def count_tokens(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType,
    ) -> LocalTokenCount:
        values = [_sanitize_text(text) for text in texts]
        if not values or any(not text for text in values):
            raise ValueError("token count inputs must be nonempty text")
        prepared = (
            [f"Instruct: {self._instruction}\nQuery:{text}" for text in values]
            if input_type == "query"
            else values
        )
        response, _ = self._send(
            {"operation": "count_tokens", "texts": prepared}
        )
        return LocalTokenCount(
            per_text_tokens=tuple(
                int(value) for value in response["per_text_input_tokens"]
            ),
            truncated_text_count=int(response["truncated_text_count"]),
        )

    def _send(self, payload: dict[str, object]) -> tuple[dict[str, object], float]:
        with self._lock:
            if self._process.poll() is not None:
                raise EmbeddingProviderError("local_worker_stopped", retryable=False)
            self._request_index += 1
            request_id = f"local-{self._request_index}"
            request = {"request_id": request_id, **payload}
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            started = perf_counter()
            self._process.stdin.write(
                json.dumps(request, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            wall_latency_ms = (perf_counter() - started) * 1000.0
        if not line:
            raise EmbeddingProviderError("local_worker_stopped", retryable=False)
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            raise EmbeddingProviderError("malformed_local_response", retryable=False) from None
        if response.get("status") != "success":
            error_type = str(response.get("error_type") or "local_inference_error")
            error_message = str(response.get("error_message") or "").strip()
            raise EmbeddingProviderError(
                f"{error_type}: {error_message}" if error_message else error_type,
                retryable=False,
            )
        return response, wall_latency_ms

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


def _sanitize_text(text: str) -> str:
    """Replace isolated legacy-encoding surrogates that tokenizers reject."""
    return text.strip().encode("utf-8", errors="replace").decode("utf-8")
