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

## 7. 250回合 DeepSeek 全量实验

在用户明确授权消耗 API 额度后，使用冻结的 `step34-memory-proposal-v1` Prompt、`deepseek-v4-flash`、`thinking=disabled` 和安全 Reducer，对全部250个后续用户回合执行了一次正式实验。运行过程中没有根据 validation 标签修改 Prompt、Schema 或规则。

### 7.1 总体结果

| 指标 | Rule baseline | DeepSeek + 安全 Reducer | 差值 |
|---|---:|---:|---:|
| 多轮脚本数 | 250 | 250 | 0 |
| 任务类型准确率 | 45.60% | 95.60% | +50.00pp |
| 信息缺口完全匹配 | 98.00% | 54.80% | -43.20pp |
| 新增条件召回率 | 52.00% | 52.00% | 0.00pp |
| 拒绝商家召回率 | 100.00% | 99.60% | -0.40pp |
| 引用范围合法率 | 100.00% | 100.00% | 0.00pp |
| Rule fallback rate | 100.00% | 8.00% | -92.00pp |

DeepSeek 显著改善了任务类型识别，说明它确实能够理解“继续推荐、修改条件、追问详情”等多轮语义。引用范围合法率仍为100%，说明 LLM-first 没有绕过业务范围校验。

### 7.2 Development 与 Validation

| Split | 回合数 | 任务类型准确率 | 信息缺口完全匹配 | 新增条件召回率 | 拒绝商家召回率 | 引用合法率 | Rule fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 200 | 95.00% | 54.00% | 52.00% | 100.00% | 100.00% | 9.00% |
| validation | 50 | 98.00% | 58.00% | 52.00% | 98.00% | 100.00% | 4.00% |

Validation 只运行并报告一次，没有根据其结果反向调参。

### 7.3 调用量与可靠性

- 逻辑调用：250次；真实 Provider 调用：250次；缓存命中：0次。
- 成功结构化输出：230次；非法结构化输出：20次；成功率92.00%。
- 20次失败全部安全转入 Rule fallback，没有中断批处理，也没有清空旧记忆。
- 失败原因：14次引用字段校验失败、5次条件值校验失败、1次条件操作符校验失败。
- 输入 Token：674,294；输出 Token：55,233；合计729,527 Token。
- 平均 Provider 延迟：1,753.29 ms；端到端全量运行耗时约448.2秒。

### 7.4 当前指标口径限制

`新增条件召回率` 没有随 DeepSeek 提升，不能简单解释为模型没有理解新增要求。现有隐藏脚本会把“便宜一点”标成 `price_level <= 2`，把“近一点”标成 `distance_km <= 3`；但这些精确数字并没有出现在用户原话中。第34步的安全策略会把它们保存为 `price=lower`、`distance=closer`，并拒绝模型凭空生成2或3。旧评测器只比较精确 `conditions`，没有给正确的 `relative_preferences` 计分，因此这个指标低估了安全记忆更新能力。

`信息缺口完全匹配` 的下降也表明现有隐藏脚本、任务类型定义与新的多轮记忆语义没有完全对齐。后续应在不查看 validation 答案调规则的前提下，为 development 增加：相对偏好召回率、数值防编造率、任务类型与信息缺口联合一致性，并重新审计隐藏 `state_updates` 是否向评测器提供了 Agent 实际不可见的信息。

全量运行命令为：

```powershell
python scripts/run_step34_memory_benchmark.py --mode api --env-file <你的.env路径>
```

如果需要把记忆接入完整的 Query 召回、Hybrid/LightGBM、Embedding、Cross-Encoder、Review RAG 和回答链路，可使用 `scripts/run_step34_agent_benchmark.py`。该脚本支持 `--limit` 先做小样本，并分别保存 Memory LLM 与回答 LLM 的用量。

原始逐回合文件保存在本地 `runs/session_memory_v1/api_full/`。`runs/` 按仓库规则不提交，避免提交大体积运行日志和 Provider 请求标识；可复现的汇总结果保存在本文档中。

## 8. 主要代码

- `src/yelp_agent/session_memory/schema.py`：Proposal、正式记忆和压缩上下文 Schema。
- `extractor.py`：DeepSeek 与规则回退 Adapter。
- `resolver.py`：引用解析。
- `reducer.py`：校验、合并和权威状态更新。
- `manager.py`：对外唯一的 `update()` Interface。
- `integration.py`：接入 Agent Harness。
- `evaluation.py`：多轮记忆评测。
- `configs/session_memory.yaml`：调用与预算配置。
- `scripts/run_step34_agent_benchmark.py`：完整 Agent 多轮运行入口。

## 9. 与后续步骤的关系

- 第35步 LLM Router 读取 `RouterMemoryContext`，决定下一步动作；它不能直接改正式记忆。
- 第35步应把 `relative_preferences` 交给 Query-aware ranking，使“更近、便宜一点”等相对要求直接改变排序。
- 第37步才允许将用户明确确认的长期偏好写入可写长期记忆；第34步只产生 `long_term_candidates`。
