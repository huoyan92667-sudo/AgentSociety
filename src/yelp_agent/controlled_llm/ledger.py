"""In-memory, secret-free accounting for Step 29 model calls."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ControlledLLMCallTrace


class ControlledLLMUsageLedger:
    def __init__(self) -> None:
        self._traces: list[ControlledLLMCallTrace] = []

    @property
    def traces(self) -> tuple[ControlledLLMCallTrace, ...]:
        return tuple(self._traces)

    def record(self, trace: ControlledLLMCallTrace) -> None:
        self._traces.append(trace)

    def replace(self, trace: ControlledLLMCallTrace) -> None:
        """Reclassify a call after higher-level policy validation."""

        for index in range(len(self._traces) - 1, -1, -1):
            if self._traces[index].call_id == trace.call_id:
                self._traces[index] = trace
                return
        raise KeyError(f"controlled LLM call is not in the ledger: {trace.call_id}")

    def summary(self) -> dict[str, object]:
        provider = [item for item in self._traces if item.provider_called]
        known = [item for item in provider if item.total_tokens is not None]
        unknown = [item for item in provider if item.usage_unknown]
        by_capability: dict[str, dict[str, object]] = {}
        for capability in ("semantic_interpretation", "answer_composition"):
            rows = [item for item in self._traces if item.capability == capability]
            called = [item for item in rows if item.provider_called]
            by_capability[capability] = {
                "logical_call_count": len(rows),
                "provider_call_count": len(called),
                "cache_hit_count": sum(item.cache_hit for item in rows),
                "success_count": sum(item.status == "success" for item in rows),
                "failure_count": sum(
                    item.status in {"provider_failure", "invalid_output"}
                    for item in rows
                ),
                "input_tokens": sum(item.input_tokens or 0 for item in called),
                "output_tokens": sum(item.output_tokens or 0 for item in called),
                "total_tokens": sum(item.total_tokens or 0 for item in called),
                "usage_unknown_count": sum(item.usage_unknown for item in called),
                "mean_latency_ms": (
                    sum(item.latency_ms for item in called) / len(called)
                    if called
                    else 0.0
                ),
            }
        return {
            "schema_version": 1,
            "logical_call_count": len(self._traces),
            "provider_call_count": len(provider),
            "cache_hit_count": sum(item.cache_hit for item in self._traces),
            "success_count": sum(item.status == "success" for item in self._traces),
            "failure_count": sum(
                item.status in {"provider_failure", "invalid_output"}
                for item in self._traces
            ),
            "input_tokens": sum(item.input_tokens or 0 for item in known),
            "output_tokens": sum(item.output_tokens or 0 for item in known),
            "total_tokens": sum(item.total_tokens or 0 for item in known),
            "usage_unknown_count": len(unknown),
            "by_capability": by_capability,
        }

    def write(self, root: str | Path) -> tuple[Path, Path]:
        output = Path(root)
        output.mkdir(parents=True, exist_ok=True)
        traces_path = output / "llm_calls.jsonl"
        summary_path = output / "llm_usage.json"
        traces_path.write_text(
            "".join(
                item.model_dump_json() + "\n"
                for item in self._traces
            ),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return traces_path, summary_path
