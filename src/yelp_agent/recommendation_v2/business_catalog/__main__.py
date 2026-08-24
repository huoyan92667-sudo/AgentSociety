"""允许直接运行餐厅子集构建命令，不触发重复模块加载警告。"""

from .builder import main

if __name__ == "__main__":
    main()
