"""用全量硬筛评论检验评论召回，不用预选候选冒充正确答案。"""

from .benchmark import (
    SingleCaseBenchmarkConfig,
    SingleCaseBenchmarkReport,
    run_single_case_benchmark,
)

__all__ = [
    "SingleCaseBenchmarkConfig",
    "SingleCaseBenchmarkReport",
    "run_single_case_benchmark",
]
