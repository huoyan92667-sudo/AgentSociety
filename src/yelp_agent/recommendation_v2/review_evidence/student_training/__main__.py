"""学生训练数据模块的命令行入口。"""

from __future__ import annotations

import sys

from .dataset_builder import main


if __name__ == "__main__":
    main(sys.argv[1:])
