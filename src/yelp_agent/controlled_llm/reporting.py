"""Publish Step 29 usage accounting without altering frozen evaluator data."""

from __future__ import annotations

import json
from pathlib import Path


def augment_runtime_metrics(
    path: str | Path,
    usage: dict[str, object],
) -> Path:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("runtime metrics must contain a JSON object")
    payload["llm_call_count"] = usage.get("provider_call_count", 0)
    payload["llm_input_tokens"] = usage.get("input_tokens", 0)
    payload["llm_output_tokens"] = usage.get("output_tokens", 0)
    payload["llm_total_tokens"] = usage.get("total_tokens", 0)
    payload["llm_usage_unknown_count"] = usage.get("usage_unknown_count", 0)
    payload["llm_cache_hit_count"] = usage.get("cache_hit_count", 0)
    payload["llm_failure_count"] = usage.get("failure_count", 0)
    payload["llm_by_capability"] = usage.get("by_capability", {})
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
