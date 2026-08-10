# 第 24 步：Rule Agent V1（完整实现与实验结果）

## 1. 这一步做成了什么

第 24 步给第 22 步的 Agent Harness 和第 23 步的真实工具安装了第一版确定性“大脑”。它不调用 DeepSeek，而是根据当前状态每次只选择一个合法动作。

推荐请求的主流程是：

```text
读取结构化请求
  -> 信息不足时追问并暂停
  -> 五路召回候选
  -> 执行硬约束
  -> Hybrid V2 排序
  -> 读取 Top-3 商家详情
  -> 返回推荐
```

详情、评论体验、官网政策和商家比较问题走各自的受控分支。尚未安装 Review RAG 或官网实时核验时，Agent 会明确保守回答，不会伪造评论或实时政策。

## 2. 完成的子模块

### 24.1 RouteFacts

`state_view.py` 把复杂的 `AgentSession` 整理为 Router 真正需要的小型事实：任务类型、信息缺口、当前候选范围、是否已过滤、是否已排序、是否读过详情、证据是否足够以及本轮剩余预算。

### 24.2 AllowedActionPolicy

`policy.py` 根据当前事实只开放本轮合法动作。例如缺位置时只能追问或回退；已经召回但尚未过滤时只能过滤或回退。

### 24.3 追问与恢复

`clarification.py` 固定了信息缺口优先级和原因码。Harness 支持：

- 追问后暂停；
- 用户回答后从同一 Session 恢复；
- 用户收到推荐后继续说“太远了，换一家”；
- 隐藏测试回复只有在触发动作真实发生后才释放。

预算、工具次数和语义调用次数按用户轮次限制，完整 trace 仍按 Session 累积。

### 24.4 推荐 Router

`router.py` 实现召回、硬约束、排序、详情和返回推荐的阶段决策。返回的商家 ID 只能来自当前有效排名，Router 不能自己编造 ID。

### 24.5 详情、体验、政策、比较和反馈

- 商家详情：读取冻结的静态详情后回答；
- 评论体验：先读取商家 Aspect 聚合画像，证据不足时保守回答；
- 官网政策：当前没有实时官网工具，因此要求用户进行官方确认；
- 商家比较：读取多个商家画像并调用确定性比较工具；
- 反馈修改：保留 Session 上下文，显式“换一家”会排除上一轮 Top-1，然后在当前候选范围重新排序。

临时反馈不会写入长期用户画像。

### 24.6 Terminal Executor

`terminal_executor.py` 负责四类最终输出和反馈状态更新：追问、推荐、有依据回答、保守回答。事实性 claim 必须附带结构化 `EvidenceReference`。

### 24.7 真实运行时装配

`runtime.py` 一次性加载并连接：

- Yelp 时点数据；
- TF-IDF 与四类 Hybrid V1 特征；
- Item-KNN；
- 用户画像 V1；
- 商家画像 V1；
- 冻结的 LambdaMART Hybrid V2；
- 第 23 步工具注册表；
- Hybrid V2 安全回退；
- Rule Router 与 Agent Harness。

所有路径都由 `RuleAgentSourcePaths` 显式声明并在启动时校验。Agent、Router 和工具构造函数均不接收隐藏答案路径。

### 24.8 500 场景 Benchmark

规则只使用 400 条 development 暴露问题；修正明确的任务识别与会话恢复错误后冻结。随后只运行一次 100 条 validation，不根据 validation 隐藏标签调规则。最后把两个冻结运行结果离线合并为 500 场景总报告。

## 3. 最终结果

| 指标 | Development（400） | Validation（100） | 总计（500） |
|---|---:|---:|---:|
| Task type accuracy | 97.75% | 97.92% | 97.78% |
| Action accuracy | 60.28% | 59.44% | 60.11% |
| Tool selection accuracy | 65.39% | 65.69% | 65.45% |
| Evaluator invalid action rate | 23.96% | 23.05% | 23.78% |
| Repeated tool call rate | 0.00% | 0.00% | 0.00% |
| Direct return precision | 49.81% | 47.01% | 49.25% |
| Missing-field precision | 68.63% | 88.89% | 71.67% |
| Missing-field recall | 53.03% | 57.14% | 53.75% |
| Unnecessary question rate | 31.11% | 11.11% | 27.78% |
| Question answerability rate | 68.89% | 88.89% | 72.22% |
| Fallback rate | 0.00% | 1.00% | 0.20% |
| Valid candidate rate | 0.98% | 0.65% | 0.92% |
| Empty result rate | 3.26% | 0.00% | 2.61% |
| Mean latency | 1,949 ms | 2,249 ms | 2,009 ms |
| P95 latency | 6,139 ms | 6,379 ms | 6,202 ms |

运行规模：

- 500 个场景；
- 721 个用户轮次；
- 2,435 个动作，平均每场景 4.87 步；
- 1,534 次工具调用，平均每场景 3.068 次；
- LLM 调用 0 次，费用 0 美元。

development 有 35 个、validation 有 7 个场景没有按隐藏脚本完整推进。它们主要集中在预算表达、冲突表达和团体人数的信息缺口判断。

## 4. 两种“非法动作”不要混淆

Harness 的安全校验和 Benchmark 的 `invalid_action_rate` 不是同一个概念。

- Harness 非法动作：Router 选择了当前状态不允许的动作、越界商家、重复工具或错误证据。这些会被代码拒绝。
- Benchmark invalid action：动作在程序上合法，但不在该测试题隐藏标签希望的动作集合内。例如当前没有 Review RAG，Agent 读取商家聚合画像并保守回答，而题目希望执行评论检索。

因此 23.78% 不表示 23.78% 的动作绕过了安全验证，而是说明当前能力和 Benchmark 期望仍有明显差距。

## 5. 结果暴露出的真实短板

### 查询语义还没有真正进入主排序

`valid_candidate_rate=0.92%` 很低。当前 Hybrid V2 主要根据历史偏好和时点质量排序；“今天想吃什么、环境怎样、预算多少”等即时请求还没有 Embedding 或 Cross-Encoder 分数。Rule Router 会正确调用排序工具，但工具本身还不够理解当前请求。

### 规则解析器覆盖有限

开放类别、不同预算说法、复杂冲突和中文人数表达仍可能漏识别或产生多余追问。这是后续低置信度第二层语义模型要解决的问题，不应继续无限堆正则。

### Review 与官网能力尚未安装

当前只有商家 Aspect 聚合画像，没有原子评论 RAG；也没有官网实时核验。因此评论体验题的引用召回率和实时政策题的完整动作率不会高。

### 时点计算仍较慢

完整 development 约 13 分钟，validation 约 4 分钟。每个用户和 cutoff 都不同，必须重算时点特征。后续可以增加不会跨 cutoff 泄漏的特征缓存。

## 6. 如何复现

```powershell
.\.venv\Scripts\python.exe scripts/run_rule_agent_benchmark.py `
  --split development `
  --output-root runs/rule_agent_v1/development

.\.venv\Scripts\python.exe scripts/run_rule_agent_benchmark.py `
  --split validation `
  --output-root runs/rule_agent_v1/validation

.\.venv\Scripts\python.exe scripts/merge_rule_agent_benchmark.py
```

主要产物位于 `runs/rule_agent_v1/`：

- `scenario_runs.jsonl`：500 场景完整结构化轨迹；
- `metrics.json`：第 21 步评测器指标；
- `runtime_metrics.json`：步骤、工具、轮次和 LLM 成本；
- `driver_results.jsonl`：隐藏脚本是否按触发器释放；
- `failures.jsonl`：回退或未完整驱动案例；
- `summary.md`：人类可读摘要。

## 7. 下一步连接点

第 24 步不是最终 Agent，而是后续模型的可解释下限和安全回退：

- 第 25 步：Embedding 查询—商家语义匹配；
- 第 26 步：Cross-Encoder 小候选精排；
- 第 27 步：Review RAG；
- 第 28 步：评论证据聚合与冲突判断；
- 第 29 步：低置信度时调用 DeepSeek 做第二层语义解析；
- 后续：Constrained LLM Router、Cost-aware LLM Router 与 Rule Router 公平对比。
