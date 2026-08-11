"""Freeze the Development-selected Step 30 policy before Validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from yelp_agent.semantic_ranking import (
    SemanticRankingPolicy,
    write_semantic_ranking_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tuning",
        type=Path,
        default=Path("runs/semantic_ranking_v1/development_tuning.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/semantic_ranking_policy.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/semantic_ranking_v1/frozen_policy_manifest.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    tuning_path = _resolve(root, args.tuning)
    payload = json.loads(tuning_path.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "development" or payload.get(
        "validation_used_for_selection"
    ) is not False:
        raise ValueError("Step 30 policy must be selected on Development only")
    selected = payload.get("selected_protected")
    policy_payload = selected.get("policy") if isinstance(selected, dict) else None
    if not isinstance(policy_payload, dict):
        raise ValueError("tuning output has no selected protected policy")
    policy = SemanticRankingPolicy.model_validate(policy_payload)
    output = write_semantic_ranking_policy(policy, _resolve(root, args.output))
    manifest = {
        "schema_version": 1,
        "policy_status": "frozen_before_validation",
        "selection_split": "development",
        "validation_used_for_selection": False,
        "policy_sha256": _sha256(output),
        "tuning_sha256": _sha256(tuning_path),
        "source_sha256": {
            path.as_posix(): _sha256(root / path)
            for path in (
                Path("src/yelp_agent/semantic_ranking/intent_compiler.py"),
                Path("src/yelp_agent/semantic_ranking/scoring.py"),
                Path("src/yelp_agent/semantic_ranking/policy.py"),
                Path("src/yelp_agent/semantic_ranking/engine.py"),
            )
        },
    }
    manifest_path = _resolve(root, args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"policy={output}")
    print(f"policy_sha256={manifest['policy_sha256']}")
    print(f"manifest={manifest_path}")


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.casefold() in {".json", ".md", ".py", ".yaml", ".yml"}:
        payload = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
