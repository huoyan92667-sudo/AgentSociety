"""Sum observed provider usage across all persisted Step 29 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/controlled_llm_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/controlled_llm_v1/cumulative_usage.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    runs_root = args.runs_root if args.runs_root.is_absolute() else root / args.runs_root
    output = args.output if args.output.is_absolute() else root / args.output
    rows: list[dict[str, object]] = []
    totals = {
        "provider_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_unknown_count": 0,
        "failure_count": 0,
    }
    for path in sorted(runs_root.rglob("llm_usage.json")):
        if output == path:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("accounting_kind") == "merged_view_not_additional_provider_usage":
            continue
        row = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            **{
                key: int(payload.get(key, 0) or 0)
                for key in totals
            },
        }
        rows.append(row)
        for key in totals:
            totals[key] += int(row[key])
    result = {
        "schema_version": 1,
        "accounting_scope": "persisted_step29_provider_runs",
        "cache_hits_are_not_counted_as_provider_tokens": True,
        "known_usage_is_a_lower_bound_when_usage_unknown_count_is_nonzero": True,
        "totals": totals,
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
