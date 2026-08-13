# 第34步：统一会话记忆与上下文压缩

## 1. 本步解决的问题

第34步把旧的“上一轮问题和本轮问题直接拼接后重新解析”替换为真正的会话记忆更新：

```text
新用户消息
  → DeepSeek 输出 MemoryProposal
  → 代码解析“第一家”等引用
  → 校验商家范围、证据片段、硬约束和数值
  → MemoryReducer 更新正式 SessionMemory
  → 压缩成 RouterMemoryContext
  → 重新执行 Query 召回和排序
```

核心原则是：**模型理解人话，代码核对事实并记账。** `MemoryProposal` 只是修改建议，不能直接覆盖正式记忆。

## 2. 三层记忆结构

### 2.1 只读长期行为画像

原有 `UserProfileV1` 继续作为从 Yelp 历史行为计算出来的只读长期画像，由 Hybrid V2 / LightGBM 排序使用。第34步不把“今天想吃牛排”写进长期画像。可写长期记忆仍留到第37步。

### 2.2 权威 SessionMemory

`SessionMemory` 保存：

- 当前合并后的结构化请求；
- 当前任务类型与信息缺口；
- 已拒绝商家；
- 最近一次展示的 Top-5；
- 当前合法候选范围；
- 追问答案；
- “更近、便宜一点、安静一点”等相对偏好；
- 最近8轮结构化变更记录；
- 可能值得长期保存、但尚未写入长期画像的候选偏好。

### 2.3 RouterMemoryContext

给工具和未来第35步 LLM Router 的不是全部 observations，而是压缩后的权威字段。旧 `GET_SESSION_MEMORY` 在新会话中不再返回不断增长的原始 observation 列表。

## 3. API 调用策略

- 首轮请求直接用现有结构化解析器建立初始记忆，不重复调用“记忆更新模型”。
- 第二轮及以后，每条新用户消息最多调用一次 DeepSeek。
- 温度为0，`thinking=disabled`，90秒超时，最多重试2次。
- 使用 JSON 模式和完整 Pydantic JSON Schema。
- 只有通过 Schema、引用范围和证据校验的结果才能进入正式记忆。
- 缓存键同时包含模型、Prompt 和可见输入哈希，不会跨用户错误复用。
- 缺少配置、超时、非法 JSON 或不合法字段时保留旧记忆并使用高精度 Rule fallback。

## 4. 安全校验

### 4.1 引用解析

模型只允许输出 `ordinal=1`，不能根据“第一家”自己编 business ID。代码只从上次真正展示的 Top-5 中解析 ID。越界序号和不可见 ID 会变成 `ambiguous_reference`。

### 4.2 硬约束保护

模型不能静默删除或降级已有硬约束。只有用户消息中出现明确的“取消、放宽、不再、remove、relax”等证据时，删除建议才会被接受。

### 4.3 数值防编造

“近一点”不能被改写成“5公里以内”，“便宜一点”也不能变成任意金额。精确数值必须原样出现在 `evidence_span`；相对语言进入 `relative_preferences`。

### 4.4 拒绝商家保护

已拒绝商家会进入 `rejected_business_ids`。新一轮 Query + history 召回会主动排除这些 ID，避免“第一家太贵”之后同一家又被召回。

### 4.5 时间与标签隔离

Session 的 `cutoff_time` 固定不可修改。Memory Manager、引用解析器和 Reducer 均不接收 Ground Truth；隐藏脚本只由 Benchmark evaluator 读取。

## 5. 真实 DeepSeek 冒烟

测试对话：

1. “帮我找一家适合约会的牛排馆，人均不要超过80美元。”
2. “第一家太贵了，换一家近一点的，但安静这个要求要保留。”

最终真实调用结果：

- 模型：`deepseek-v4-flash`；
- Provider 调用：1次（首轮不调用记忆更新模型）；
- 延迟：2821.05 ms；
- 输入 Token：2518；
- 输出 Token：364；
- 总 Token：2882；
- 状态：success；
- `第一家` 被代码解析为 `business-A`；
- `business-A` 被加入拒绝集合；
- 原有“牛排馆、人均80美元、适合约会”得到保留；
- “近一点”保存为 `distance=closer`，没有伪造公里数；
- “安静”继续保留为当前 Session 偏好。

本步实现与修正过程中共执行8次真实 Provider 调用，累计消耗14,726 Token。前几次调用专门暴露并修复了“相对距离被编成固定公里数”、引用编号格式不稳定和过度压缩 Schema 导致输出不稳定等问题；最终配置对应的是上面单次2,882 Token的成功结果。

开发过程中另外验证了非法引用编号、非法字段和 API disabled 场景，均安全回退且不会清空原记忆。

## 6. 250个多轮脚本的无API基线

现有500个 Agent 场景中，130个场景包含隐藏后续回复，共250个后续用户回合。完整 Rule fallback 基线结果：

| 指标 | 结果 |
|---|---:|
| 多轮脚本数 | 250 |
| 任务类型准确率 | 45.60% |
| 信息缺口完全匹配 | 98.00% |
| 新增条件召回率 | 52.00% |
| 拒绝商家召回率 | 100.00% |
| 引用范围合法率 | 100.00% |
| Rule fallback rate | 100.00% |

这组数据的作用是证明纯规则的下限和短板。任务类型及新增条件只有约一半，说明第34步采用 LLM-first 语义抽取是必要的。

没有直接执行250回合真实 API 全量实验：按最后一次冒烟粗略估算约需 `250 × 2882 ≈ 720,500` Token，并产生约12分钟的纯 API 等待时间。这会明显消耗用户额度。完整运行入口已经提供，用户明确决定消耗额度后可执行：

```powershell
python scripts/run_step34_memory_benchmark.py --mode api --env-file <你的.env路径>
```

如果需要把记忆接入完整的 Query 召回、Hybrid/LightGBM、Embedding、Cross-Encoder、Review RAG 和回答链路，可使用 `scripts/run_step34_agent_benchmark.py`。该脚本支持 `--limit` 先做小样本，并分别保存 Memory LLM 与回答 LLM 的用量。

## 7. 主要代码

- `src/yelp_agent/session_memory/schema.py`：Proposal、正式记忆和压缩上下文 Schema。
- `extractor.py`：DeepSeek 与规则回退 Adapter。
- `resolver.py`：引用解析。
- `reducer.py`：校验、合并和权威状态更新。
- `manager.py`：对外唯一的 `update()` Interface。
- `integration.py`：接入 Agent Harness。
- `evaluation.py`：多轮记忆评测。
- `configs/session_memory.yaml`：调用与预算配置。
- `scripts/run_step34_agent_benchmark.py`：完整 Agent 多轮运行入口。

## 8. 与后续步骤的关系

- 第35步 LLM Router 读取 `RouterMemoryContext`，决定下一步动作；它不能直接改正式记忆。
- 第35步应把 `relative_preferences` 交给 Query-aware ranking，使“更近、便宜一点”等相对要求直接改变排序。
- 第37步才允许将用户明确确认的长期偏好写入可写长期记忆；第34步只产生 `long_term_candidates`。
