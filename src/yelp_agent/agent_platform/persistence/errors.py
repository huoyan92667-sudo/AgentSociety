"""持久化层对上层暴露的可判断错误。"""


class PersistenceError(RuntimeError):
    """持久化操作失败的共同父类。"""


class SessionBusyError(PersistenceError):
    """同一会话已有一轮正在执行，拒绝并发覆盖状态。"""


class StateVersionConflictError(PersistenceError):
    """保存领域状态时，调用方看到的旧版本已经过期。"""
