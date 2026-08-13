"""One idempotent migration for location clarification state in early V2 bundles."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from yelp_agent.session_memory_benchmark.schema import (
    BenchmarkV2Manifest,
    FrozenScriptedTurnV2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    turns_path = root / "visible" / "scripted_turns.jsonl"
    manifest_path = root / "manifest.json"
    turns = [
        FrozenScriptedTurnV2.model_validate_json(line)
        for line in turns_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    migrated = [
        item.model_copy(
            update={
                "state_updates": {
                    "user_latitude": 39.9526,
                    "user_longitude": -75.1652,
                }
            }
        )
        if item.intent_code == "answer_missing_location"
        else item
        for item in turns
    ]
    _atomic_write(
        turns_path,
        "".join(
            item.model_dump_json() + "\n"
            for item in sorted(migrated, key=lambda value: value.turn_case_id)
        ),
    )
    manifest = BenchmarkV2Manifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    hashes = dict(manifest.output_sha256)
    hashes["visible_scripted_turns"] = _sha256_file(turns_path)
    _atomic_write(
        manifest_path,
        manifest.model_copy(update={"output_sha256": hashes}).model_dump_json(indent=2)
        + "\n",
    )
    print(f"migrated_location_turns={sum(bool(item.state_updates) for item in migrated)}")


def _atomic_write(path: Path, content: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(content, encoding="utf-8", newline="\n")
    partial.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
