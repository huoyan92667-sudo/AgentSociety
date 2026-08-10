# 第 23 步：通用工具注册表与真实 Yelp 工具

## 1. 这一步解决了什么

第 22 步只有 Agent 的运行骨架，本步给它安装了一套统一、受控、可记录的工具。Router 不直接操作 Pandas、Parquet、模型或 Yelp 数据，只需要产生一个结构化决定：

```text
AgentDecision
  -> RegistryActionExecutor
  -> AgentToolRegistry.execute(tool_name, arguments, visible_context)
  -> 参数、动作、候选范围和可用性校验
  -> 执行工具（必要时在同一次逻辑调用内重试）
  -> ToolObservation
  -> ActionOutcome
  -> AgentState 与 Step 21 Trace
```

外部调用方只需要学习三个注册表接口：

```python
registry.describe(tool_name)
registry.list_tools()
registry.execute(tool_name, arguments, context)
```

工具的输入和输出均由 Pydantic 校验。原始异常、DuckDB 连接、模型对象和认证信息不会进入 AgentState。

## 2. 当前可用的工具

| 工具 | 对应 Agent 动作 | 作用 |
|---|---|---|
| `GET_SESSION_MEMORY` | `apply_feedback` 等 | 读取当前会话中已经产生的可见观察，不读取隐藏脚本 |
| `GET_USER_PROFILE` | 召回、排序、比较 | 按 `user_id + cutoff_time` 精确读取冻结用户画像 |
| `EXPAND_CANDIDATES` | `retrieve_candidates` | 从全量可用商家通过质量、类别、文本、位置和 Item-KNN 五路召回候选 |
| `APPLY_CONSTRAINTS` | `apply_hard_constraints` | 使用与离线 Query 排序相同的规则删除确定违反硬条件的商家 |
| `GET_HYBRID_RANKING` | `rank_candidates` | 为当前候选构造冻结的 Hybrid V2 特征并输出完整排序 |
| `GET_BUSINESS_DETAILS` | `get_business_details`、比较 | 读取静态字段和截止时间前重新计算的质量信息 |
| `GET_BUSINESS_PROFILE` | 详情、排序、比较 | 读取商家质量与聚合 Aspect 证据，不返回原始评论文本 |
| `COMPARE_BUSINESSES` | `compare_candidates` | 用当前请求条件确定性比较 2～10 个候选 |

`ASK_CLARIFICATION` 和 `FINALIZE` 没有放入工具目录，因为它们是高层 Agent 动作，不应虚增工具调用次数。

## 3. 已登记但尚不可用的未来工具

| 工具 | 状态 | 实现步骤 |
|---|---|---|
| `COMPUTE_EMBEDDING_MATCH` | `unavailable` | 第 25 步 |
| `COMPUTE_CROSS_ENCODER_MATCH` | `unavailable` | 第 26 步 |
| `SEARCH_BUSINESS_REVIEWS` | `unavailable` | 第 27 步 |
| `AGGREGATE_REVIEW_EVIDENCE` | `unavailable` | 第 28 步 |
| `ASSESS_LLM_SEMANTICS` | `unavailable` | 第 29 步 |

这些名称已经进入稳定目录，未来可替换为真实 Adapter；现在调用会明确返回 `TOOL_NOT_IMPLEMENTED`，不会伪造结果。本步不调用 DeepSeek、Embedding 或 Cross-Encoder。

## 4. 一个工具如何被约束

每个 `ToolDefinition` 明确保存：

- 名称、版本、工具类别和允许对应的 Agent 动作；
- 输入、输出 JSON Schema；
- 执行前提；
- 能解决的信息缺口或不确定性；
- 是否允许缓存、空结果如何解释；
- 最大尝试次数、单次超时上限和预计成本；
- 工具不可用原因与安全回退策略；
- 可展示给 Router 或文档的简短说明。

注册表还会执行两项代码级保护：

1. 详情、比较、排序等工具只能访问当前 `business_scope` 内的商家。
2. 工具上下文只由第 22 步的可见 `AgentSession` 构造，不存在 Ground Truth、正确商家或隐藏用户回复字段。

## 5. 没查到结果、临时失败和永久失败的区别

`ToolObservation.status` 有六种值：

| 状态 | 含义 | 后续处理 |
|---|---|---|
| `success` | 得到完整可用结果 | 写入状态并继续 |
| `partial` | 得到部分可用结果 | 带警告写入状态，Router 决定下一步 |
| `no_result` | 工具正常完成，但数据中确实没有结果 | 不原地重试，允许换工具、放宽条件或追问 |
| `retryable_error` | 超时或临时依赖故障 | 只在同一次注册表调用内按配置重试 |
| `permanent_error` | 参数、范围、数据结构或确定性依赖错误 | 不重试，进入安全回退 |
| `unavailable` | 工具尚未实现或未启用 | 不重试，记录明确原因并回退 |

第 22 步禁止 Router 用完全相同的参数再次调用同一工具，防止死循环。第 23 步的内部重试不会与它冲突：一次 Router 决策只产生一条逻辑工具调用和一条 Tool Trace；注册表可以在这条调用内部完成最多两次物理尝试，并在 `attempt_count` 中记录。`no_result` 不是故障，因此不会浪费重试。

失败工具的完整 `status/error_code/warnings/attempt_count` 也会写入 AgentState，之后可以统计错误类型或让未来 Router 选择替代工具。

## 6. 在线 Hybrid V2 如何复用已有模型

`OnlineHybridV2RankingService` 完成四件事：

1. 按当前用户和截止时间读取精确用户画像；
2. 只为当前候选读取同一截止时间的商家画像；
3. 将召回工具保存的候选子分数、用户画像和商家画像组装成训练时的 `ALL_FEATURE_NAMES`；
4. 交给已经冻结的 Hybrid V2/LambdaMART Ranker，并返回完整、唯一、确定性的排列。

在线特征组装复用了 `learning_to_rank/features.py` 中的同一套类别、Aspect、可靠性、Hybrid V1 和缺失值公式。它不重新拟合模型，也不读取 validation/test 标签。

## 7. 安全回退

`HybridV2FallbackHandler` 按以下顺序处理失败：

1. 已经存在合法 Hybrid V2 排名时直接复用；
2. 已经召回但尚未排序时，只针对当前候选范围重新计算 Hybrid V2；
3. 若排序依赖也失败，按确定性的召回融合顺序返回；
4. 如果尚无候选，返回合法的空回退。

硬约束工具缩小过的 `business_scope` 始终优先，回退不能把已过滤商家重新加入。

## 8. 配置和主要文件

```text
configs/agent_tools.yaml

src/yelp_agent/agent_tools/
├── schema.py          # ToolObservation、Descriptor、ExecutionContext
├── registry.py        # 注册、发现、校验、缓存、重试、错误归一化
├── executor.py        # 第 22 步 Harness 适配层
├── assembly.py        # 在线 Hybrid V2 特征连接与排序
├── fallback.py        # Hybrid V2 安全回退
├── catalog.py         # 8 个可用工具和 5 个未来工具
├── config.py          # 有界运行配置
├── errors.py          # 可重试/永久错误类型
├── tool_schemas.py    # 真实工具输入输出 Schema
└── adapters/          # 画像、召回、约束、排序、商家和比较 Adapter
```

硬约束公共规则位于 `src/yelp_agent/query/ranking.py`；在线特征公共组装位于 `src/yelp_agent/learning_to_rank/features.py`。离线和 Agent 路径因此不会各维护一套规则。

## 9. 验证结果和本步边界

测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

第 23 步完成时结果为 `356 passed`。测试覆盖工具契约、范围隔离、真实点时商家数据、候选召回、硬约束、在线 Hybrid V2、缓存、内部重试、结构化失败、Fallback 和多工具 Harness 集成链。

本步仍没有正式决策大脑。第 24 步才会实现 Rule Router，让 Agent 根据 `task_type`、`information_gaps`、已有观察和错误记录动态选择这些工具。
