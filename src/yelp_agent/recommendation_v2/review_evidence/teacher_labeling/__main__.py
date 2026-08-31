"""教师标注模块的命令行入口。"""

from __future__ import annotations

import sys

from .audit import main as audit_main
from .candidate_sampler import main as candidates_main
from .reuse_labels import main as reuse_main
from .teacher_runner import main as run_main


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"candidates", "reuse", "run", "audit"}:
        raise SystemExit(
            "usage: python -m yelp_agent.recommendation_v2.review_evidence."
            "teacher_labeling {candidates|reuse|run|audit} [options]"
        )
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "candidates":
        candidates_main()
    elif command == "reuse":
        reuse_main()
    elif command == "run":
        run_main()
    else:
        audit_main()


if __name__ == "__main__":
    main()
