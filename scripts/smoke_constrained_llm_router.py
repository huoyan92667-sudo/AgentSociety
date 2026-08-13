"""One secret-free real-provider smoke call for the Step 35 Router."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from yelp_agent.agent_harness import AgentSession, HarnessBudget, RuleBasedRequestInterpreter
from yelp_agent.constrained_llm_router import (
    build_constrained_llm_router_runtime,
    load_constrained_llm_router_config,
)
from yelp_agent.query import QueryParseInput
from yelp_agent.rule_router import RuleRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/constrained_llm_router.yaml"))
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--query",
        default="第一家有点贵。请继续给我推荐，但要更便宜一些。",
    )
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    root = args.project_root.resolve()
    interpreted = RuleBasedRequestInterpreter().interpret(
        QueryParseInput(
            user_id="router-smoke-user",
            session_id="router-smoke-session",
            cutoff_time=datetime(2022, 1, 1, tzinfo=UTC),
            query_text=args.query,
            referenced_business_ids=["visible-business-1"],
        )
    )
    state = AgentSession(
        scenario_id="5" * 64,
        split="development",
        language="zh-CN",
        agent_version="step35-smoke",
        user_id="router-smoke-user",
        session_id="router-smoke-session",
        cutoff_time=datetime(2022, 1, 1, tzinfo=UTC),
        request=interpreted.request,
        readiness=interpreted.readiness,
        budget=HarnessBudget(max_semantic_calls=12, max_total_tokens=100_000),
        started_at_ms=0,
    )
    config_path = args.config if args.config.is_absolute() else root / args.config
    with build_constrained_llm_router_runtime(
        project_root=root,
        config=load_constrained_llm_router_config(config_path),
        fallback_router=RuleRouter(review_rag_enabled=True),
    ) as runtime:
        decision = runtime.router.choose_action(state)
        print(
            json.dumps(
                {
                    "action": decision.action,
                    "reason_code": decision.reason_code,
                    "tool_name": decision.tool_name,
                    "arguments": decision.arguments,
                    "router_trace": (
                        None
                        if decision.router_trace is None
                        else decision.router_trace.model_dump(mode="json")
                    ),
                    "usage": runtime.ledger.summary(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
