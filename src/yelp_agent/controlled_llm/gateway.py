"""One deep JSON-call module hiding provider, cache, parsing, and tracing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from yelp_agent.agent.llm import LLMCallResult, LLMMessage

from .cache import SqliteControlledLLMCache
from .ledger import ControlledLLMUsageLedger
from .schema import ControlledLLMCallTrace, LLMCapability

OutputT = TypeVar("OutputT", bound=BaseModel)


class ChatGenerator(Protocol):
    def generate(self, messages: list[LLMMessage]) -> LLMCallResult: ...


@dataclass(frozen=True, slots=True)
class JSONCallResult[OutputT]:
    output: OutputT | None
    trace: ControlledLLMCallTrace


class ControlledJSONCaller:
    """Return one validated model or a structured failure through one interface."""

    def __init__(
        self,
        *,
        generator: ChatGenerator,
        model_name: str | None,
        cache: SqliteControlledLLMCache,
        ledger: ControlledLLMUsageLedger,
    ) -> None:
        self._generator = generator
        self._model_name = model_name
        self._cache = cache
        self._ledger = ledger

    def call(
        self,
        *,
        capability: LLMCapability,
        prompt_version: str,
        input_payload: dict[str, object],
        messages: list[LLMMessage],
        output_model: type[OutputT],
        normalize_payload: Callable[[object], object] | None = None,
        context_id: str | None = None,
        turn_index: int | None = None,
    ) -> JSONCallResult[OutputT]:
        input_json = _canonical_json(input_payload)
        prompt_json = _canonical_json([message.model_dump() for message in messages])
        input_hash = _sha256(input_json)
        prompt_hash = _sha256(prompt_json)
        model_name = self._model_name or "disabled"
        cache_key = _sha256(
            _canonical_json(
                {
                    "capability": capability,
                    "model": model_name,
                    "prompt_version": prompt_version,
                    "prompt_sha256": prompt_hash,
                }
            )
        )
        call_id = _sha256(
            _canonical_json(
                {
                    "cache_key": cache_key,
                    "input_sha256": input_hash,
                }
            )
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            try:
                output = output_model.model_validate_json(cached)
            except ValidationError:
                output = None
            if output is not None:
                trace = ControlledLLMCallTrace(
                    call_id=call_id,
                    capability=capability,
                    status="success",
                    model=self._model_name,
                    prompt_version=prompt_version,
                    input_sha256=input_hash,
                    prompt_sha256=prompt_hash,
                    provider_called=False,
                    cache_hit=True,
                    latency_ms=0.0,
                    attempt_count=0,
                    context_id=context_id,
                    turn_index=turn_index,
                    created_at=datetime.now(UTC),
                )
                self._ledger.record(trace)
                return JSONCallResult(output=output, trace=trace)

        result = self._generator.generate(messages)
        if result.status != "success" or result.content is None:
            status = "disabled" if result.status == "disabled" else "provider_failure"
            trace = self._trace_from_result(
                call_id=call_id,
                capability=capability,
                status=status,
                prompt_version=prompt_version,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                result=result,
                failure_reason=result.failure_reason or "provider_failure",
                context_id=context_id,
                turn_index=turn_index,
            )
            self._ledger.record(trace)
            return JSONCallResult(output=None, trace=trace)
        try:
            raw_payload = json.loads(_strip_code_fence(result.content))
            if normalize_payload is not None:
                raw_payload = normalize_payload(raw_payload)
            output = output_model.model_validate(raw_payload)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0] if exc.errors() else {}
            location = ".".join(str(item) for item in first.get("loc", ())) or "root"
            error_type = str(first.get("type") or "validation_error")
            reason = f"schema_validation_failed:{location}:{error_type}"[:200]
            trace = self._trace_from_result(
                call_id=call_id,
                capability=capability,
                status="invalid_output",
                prompt_version=prompt_version,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                result=result,
                failure_reason=reason,
                context_id=context_id,
                turn_index=turn_index,
            )
            self._ledger.record(trace)
            return JSONCallResult(output=None, trace=trace)
        except (ValueError, json.JSONDecodeError):
            trace = self._trace_from_result(
                call_id=call_id,
                capability=capability,
                status="invalid_output",
                prompt_version=prompt_version,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                result=result,
                failure_reason="schema_validation_failed:root:invalid_json",
                context_id=context_id,
                turn_index=turn_index,
            )
            self._ledger.record(trace)
            return JSONCallResult(output=None, trace=trace)
        created_at = datetime.now(UTC)
        self._cache.put(
            cache_key=cache_key,
            capability=capability,
            model=model_name,
            prompt_version=prompt_version,
            payload_json=output.model_dump_json(),
            created_at=created_at.isoformat(),
        )
        trace = self._trace_from_result(
            call_id=call_id,
            capability=capability,
            status="success",
            prompt_version=prompt_version,
            input_hash=input_hash,
            prompt_hash=prompt_hash,
            result=result,
            failure_reason=None,
            created_at=created_at,
            context_id=context_id,
            turn_index=turn_index,
        )
        self._ledger.record(trace)
        return JSONCallResult(output=output, trace=trace)

    def skipped_trace(
        self,
        *,
        capability: LLMCapability,
        prompt_version: str,
        input_payload: dict[str, object],
        context_id: str | None = None,
        turn_index: int | None = None,
    ) -> ControlledLLMCallTrace:
        input_hash = _sha256(_canonical_json(input_payload))
        trace = ControlledLLMCallTrace(
            call_id=_sha256(f"skipped:{capability}:{prompt_version}:{input_hash}"),
            capability=capability,
            status="skipped",
            model=self._model_name,
            prompt_version=prompt_version,
            input_sha256=input_hash,
            prompt_sha256=_sha256("skipped"),
            provider_called=False,
            cache_hit=False,
            latency_ms=0.0,
            attempt_count=0,
            context_id=context_id,
            turn_index=turn_index,
            created_at=datetime.now(UTC),
        )
        self._ledger.record(trace)
        return trace

    def reclassify(self, trace: ControlledLLMCallTrace) -> None:
        self._ledger.replace(trace)

    def _trace_from_result(
        self,
        *,
        call_id: str,
        capability: LLMCapability,
        status: str,
        prompt_version: str,
        input_hash: str,
        prompt_hash: str,
        result: LLMCallResult,
        failure_reason: str | None,
        created_at: datetime | None = None,
        context_id: str | None = None,
        turn_index: int | None = None,
    ) -> ControlledLLMCallTrace:
        provider_request_id = next(
            (
                item.provider_request_id
                for item in reversed(result.attempts)
                if item.provider_request_id
            ),
            None,
        )
        return ControlledLLMCallTrace(
            call_id=call_id,
            capability=capability,
            status=status,  # type: ignore[arg-type]
            model=result.model or self._model_name,
            prompt_version=prompt_version,
            input_sha256=input_hash,
            prompt_sha256=prompt_hash,
            provider_called=result.attempt_count > 0,
            cache_hit=False,
            latency_ms=result.latency_ms,
            attempt_count=result.attempt_count,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            usage_unknown=(
                result.attempt_count > 0 and result.total_tokens is None
            ),
            failure_reason=failure_reason,
            provider_request_id=provider_request_id,
            context_id=context_id,
            turn_index=turn_index,
            created_at=created_at or datetime.now(UTC),
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_code_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text
