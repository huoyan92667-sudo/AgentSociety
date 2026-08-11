"""Local Qwen3 Cross-Encoder Adapter backed by one persistent worker."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from time import perf_counter

from .config import CrossEncoderConfig, LocalCrossEncoderEnvironment
from .scorer import CrossEncoderProviderError, ScoredPairBatch


class LocalQwenCrossEncoder:
    provider = "local"

    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        model: str,
        config: CrossEncoderConfig,
    ) -> None:
        self._process = process
        self.model = model
        self.batch_size = config.batch_size
        self._instruction = config.instruction
        self._lock = Lock()
        self._request_index = 0

    @classmethod
    def from_environment(
        cls,
        config: CrossEncoderConfig,
        environment: LocalCrossEncoderEnvironment,
    ) -> LocalQwenCrossEncoder:
        if not environment.enabled or environment.model_path is None:
            raise ValueError("local Cross-Encoder environment is not configured")
        worker_path = Path(__file__).with_name("local_worker.py")
        process = subprocess.Popen(
            [
                str(environment.python_executable),
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
            raise CrossEncoderProviderError(detail, retryable=False)
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError:
            process.kill()
            raise CrossEncoderProviderError("malformed_worker_ready", retryable=False) from None
        if ready.get("status") != "ready":
            process.kill()
            raise CrossEncoderProviderError("worker_start_failed", retryable=False)
        return cls(
            process=process,
            model=f"local:{ready.get('model', environment.model_path.name)}",
            config=config,
        )

    def score(
        self, query_text: str, documents: Sequence[str]
    ) -> ScoredPairBatch:
        query = _sanitize_text(query_text)
        values = [_sanitize_text(document) for document in documents]
        if not query or not values or any(not value for value in values):
            raise ValueError("Cross-Encoder query and documents must be nonempty")
        if len(values) > self.batch_size:
            raise ValueError("Cross-Encoder batch exceeds configured local limit")
        response, wall_latency_ms = self._send(
            {
                "instruction": self._instruction,
                "query": query,
                "documents": values,
            }
        )
        scores = tuple(float(value) for value in response["scores"])
        token_counts = tuple(
            int(value) for value in response["per_pair_input_tokens"]
        )
        if len(scores) != len(values) or len(token_counts) != len(values):
            raise CrossEncoderProviderError("invalid_local_score_shape", retryable=False)
        if any(not 0 <= value <= 1 for value in scores):
            raise CrossEncoderProviderError("invalid_local_score", retryable=False)
        return ScoredPairBatch(
            scores=scores,
            input_tokens=sum(token_counts),
            per_pair_input_tokens=token_counts,
            truncated_pair_count=int(response["truncated_pair_count"]),
            latency_ms=wall_latency_ms,
        )

    def _send(self, payload: dict[str, object]) -> tuple[dict[str, object], float]:
        with self._lock:
            if self._process.poll() is not None:
                raise CrossEncoderProviderError("local_worker_stopped", retryable=False)
            self._request_index += 1
            request_id = f"cross-local-{self._request_index}"
            request = {"request_id": request_id, **payload}
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            started = perf_counter()
            self._process.stdin.write(
                json.dumps(request, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            latency_ms = (perf_counter() - started) * 1000.0
        if not line:
            raise CrossEncoderProviderError("local_worker_stopped", retryable=False)
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            raise CrossEncoderProviderError("malformed_local_response", retryable=False) from None
        if response.get("status") != "success":
            error_type = str(response.get("error_type") or "local_inference_error")
            message = str(response.get("error_message") or "").strip()
            reason = f"{error_type}: {message}" if message else error_type
            raise CrossEncoderProviderError(reason, retryable=False)
        return response, latency_ms

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
    return text.strip().encode("utf-8", errors="replace").decode("utf-8")
