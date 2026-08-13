from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.agent_harness import RuleBasedRequestInterpreter
from yelp_agent.query import QueryParseInput
from yelp_agent.session_memory.config import load_session_memory_config
from yelp_agent.session_memory.reducer import record_memory_observation
from yelp_agent.session_memory.runtime import build_session_memory_runtime
from yelp_agent.session_memory.schema import MemoryTurnInput


def _turn(text: str, *, turn: int, previous=None) -> MemoryTurnInput:
    raw = QueryParseInput(
        user_id="step34-smoke-user",
        session_id="step34-smoke-session",
        cutoff_time=datetime(2022, 1, 1),
        query_text=text,
    )
    base = RuleBasedRequestInterpreter().interpret(raw)
    return MemoryTurnInput(
        query_text=text,
        language="zh-CN",
        current_turn=turn,
        base_request=base.request,
        base_readiness=base.readiness,
        previous_memory=previous,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise FileNotFoundError(f"environment file does not exist: {args.env_file}")
        load_dotenv(args.env_file, override=False)
    root = args.project_root.resolve()
    config = load_session_memory_config(root / "configs" / "session_memory.yaml")
    with build_session_memory_runtime(project_root=root, config=config) as runtime:
        initial = runtime.manager.update(
            _turn("帮我找一家适合约会的牛排馆，人均不要超过80美元。", turn=1)
        )
        memory = record_memory_observation(
            initial.memory,
            turn_index=1,
            business_scope=["business-A", "business-B", "business-C"],
            presented_business_ids=["business-A", "business-B", "business-C"],
        )
        assert memory is not None
        followup = runtime.manager.update(
            _turn(
                "第一家太贵了，换一家近一点的，但安静这个要求要保留。",
                turn=2,
                previous=memory,
            )
        )
        output = {
            "proposal": followup.proposal.model_dump(mode="json"),
            "accepted_changes": followup.accepted_changes,
            "rejected_changes": followup.rejected_changes,
            "memory": {
                "revision": followup.memory.revision,
                "task_type": followup.memory.current_task_type,
                "rejected_business_ids": followup.memory.rejected_business_ids,
                "referenced_business_ids": followup.request.referenced_business_ids,
                "conditions": [
                    {
                        "field": item.field,
                        "operator": item.operator,
                        "value": item.value,
                        "importance": item.importance,
                    }
                    for item in followup.request.conditions
                ],
                "relative_preferences": [
                    item.model_dump(mode="json")
                    for item in followup.memory.relative_preferences
                ],
                "summary": followup.memory.semantic_summary,
            },
            "extraction": followup.extraction.model_dump(mode="json"),
            "usage": runtime.ledger.summary(),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
