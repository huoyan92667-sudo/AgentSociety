"""Freeze the Step 29 Development-selected policy before Validation is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from yelp_agent.controlled_llm import load_controlled_llm_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--semantic-tuning",
        type=Path,
        default=Path("runs/controlled_llm_v1/semantic_tuning.json"),
    )
    parser.add_argument(
        "--development-agent-root",
        type=Path,
        default=Path("runs/controlled_llm_v1/agent/development"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/controlled_llm_policy.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = root / "configs" / "controlled_llm.yaml"
    config = load_controlled_llm_config(config_path)
    tuning_path = _resolve(root, args.semantic_tuning)
    agent_root = _resolve(root, args.development_agent_root)
    metrics_path = agent_root / "metrics.json"
    runtime_path = agent_root / "runtime_metrics.json"
    for required in (tuning_path, metrics_path, runtime_path):
        if not required.is_file():
            raise FileNotFoundError(f"freeze input does not exist: {required}")
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    selected = tuning.get("selected")
    if not isinstance(selected, dict):
        raise TypeError("semantic tuning output has no selected policy")
    selected_threshold = selected.get("minimum_signal_confidence")
    if selected_threshold != config.semantic.minimum_signal_confidence:
        raise ValueError("configured semantic threshold does not match Development tuning")
    agent_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runtime_metrics = json.loads(runtime_path.read_text(encoding="utf-8"))
    reported_metric_names = (
        "action_accuracy",
        "tool_selection_accuracy",
        "direct_return_precision",
        "missing_field_detection_precision",
        "missing_field_detection_recall",
        "grounded_answer_rate",
        "citation_correctness",
        "unsupported_claim_rate",
        "conflict_detection_accuracy",
        "official_policy_caution_accuracy",
        "hr_at_5",
        "mrr",
        "fallback_rate",
        "mean_latency_ms",
    )
    all_metrics = agent_metrics.get("metrics", {})
    source_files = (
        config_path,
        root / "src" / "yelp_agent" / "controlled_llm" / "semantic.py",
        root / "src" / "yelp_agent" / "controlled_llm" / "answer.py",
        root / "src" / "yelp_agent" / "controlled_llm" / "schema.py",
    )
    payload = {
        "schema_version": 1,
        "policy_status": "frozen_before_validation",
        "selection_split": "development",
        "validation_used_for_selection": False,
        "agent_version": config.agent_version,
        "semantic_prompt_version": config.semantic.prompt_version,
        "answer_prompt_version": config.answer.prompt_version,
        "selected_semantic_threshold": selected_threshold,
        "semantic_tuning": tuning,
        "development_agent_summary": {
            "scenario_count": agent_metrics.get("scenario_count"),
            "metrics": {
                name: all_metrics.get(name)
                for name in reported_metric_names
            }
            if isinstance(all_metrics, dict)
            else {},
            "runtime_metrics": runtime_metrics,
        },
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in source_files
        },
    }
    output = _resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"policy_sha256={_sha256(output)}")


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.casefold() in {".json", ".md", ".py", ".yaml", ".yml"}:
        payload = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
