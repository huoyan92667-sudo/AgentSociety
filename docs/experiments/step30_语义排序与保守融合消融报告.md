# 第 30 步：让大模型语义真正改变排序，并比较激进融合与保守融合

## 1. 这一阶段解决了什么问题

第 29 步已经能够用受控 DeepSeek 语义解析器理解用户请求，例如识别“安静”“适合约会”“不要排队太久”等条件。但是这些新理解主要用于任务判断、追问和回答，并没有稳定地改变最终商家顺序。

第 30 步把这条链路补完整：

```text
用户原始问题
  → 第 29 步受控语义解析
  → 生成经过代码校验的 RecommendationRequest
  → 编译成统一的排序意图文档
  → 本地 Qwen3-Embedding-0.6B 语义匹配
  → 本地 Qwen3-Reranker-0.6B 精排
  → 读取 cutoff 之前的商家 Aspect 画像
  → 激进融合或保守融合
  → 返回新的 Top-5
```

硬约束仍然在语义重排之前由确定性工具执行。已经因为距离、类别或价格等硬条件被过滤掉的商家，语义排序不能重新放回来。第 30 步只允许重排当前有效候选的前 20 名，第 21 名及以后保持原顺序。

## 2. 代码结构

核心模块位于 `src/yelp_agent/semantic_ranking/`：

- `intent_compiler.py`：把第 29 步已经接受的结构化条件编译成稳定的排序意图；
- `scoring.py`：合并 Embedding、Cross-Encoder 和结构化 Aspect 证据；
- `policy.py`：实现激进替换和带排名保护的保守融合；
- `engine.py`：统一调度、记录诊断，并在异常时回退到第 26 步原排名；
- `schema.py`：定义意图、候选分数、移动位置、保护原因和 token 统计；
- `config.py`：读取并校验冻结策略。

Agent 新增 `APPLY_SEMANTIC_RANKING` 工具。Rule Router 的推荐路径现在是：

```text
召回 → 硬约束 → Hybrid V2 → Embedding → Cross-Encoder
     → Step30 语义排序 → 商家详情 → 返回 Top-5
```

当请求没有可参与排序的软条件时，工具直接返回第 26 步原排名，不加载本地模型、不改变候选顺序。模型异常、候选不完整或输出越界时，也完整回退到第 26 步排名。

## 3. 两种融合方式

### 3.1 LLM Aggressive：不做保守保护

语义分由三部分组成：

```text
semantic_score
  = 0.4 × embedding_score
  + 0.5 × cross_encoder_score
  + 0.1 × structured_aspect_score
```

激进模式直接用这个分数替换原 Top-20 的顺序。它最容易让当前请求改变推荐，也最容易破坏已经较可靠的 Hybrid、协同信号和历史偏好排序。

### 3.2 LLM Protected：保守融合

保守模式先把原排名和语义排名转成排名百分位，再融合：

```text
effective_alpha
  = maximum_alpha × intent_confidence × evidence_factor

final_score
  = (1 - effective_alpha) × base_rank_percentile
  + effective_alpha × semantic_rank_percentile
```

随后代码强制执行三层保护：

1. 单个候选最多上升 5 位、最多下降 3 位；
2. 原 Top-3 默认受保护；
3. 新候选只有同时具备足够语义分差、结构化匹配证据和证据覆盖率，才可以挤掉原 Top-3。

评论证据缺失不等于负面，只会降低 `evidence_factor`，让系统更信任原排序。

## 4. 参数选择和数据隔离

融合参数只在 Development 中调节。可用于调参的样本必须同时满足：

- 场景属于 Development；
- 是推荐任务；
- 有合法的目标商家和基础排名；
- 有至少一个可以进入排序的语义条件；
- 有完整的 Step30 诊断记录。

最终共有 95 个 Development 场景参与保守策略搜索。目标先比较 HR@5，并列时依次比较 MRR、HR@1、目标伤害率和平均移动位数。

冻结参数如下：

| 参数 | 冻结值 |
|---|---:|
| candidate limit | 20 |
| Embedding 权重 | 0.4 |
| Cross-Encoder 权重 | 0.5 |
| 结构化 Aspect 权重 | 0.1 |
| 最大融合 alpha | 1.0 |
| 证据下限 | 0.5 |
| 最大上升 | 5 位 |
| 最大下降 | 3 位 |
| 受保护头部 | Top-3 |
| 挑战 Top-3 所需语义分差 | 0.2 |

冻结文件为 `configs/semantic_ranking_policy.json`，SHA256 为：

```text
93f26eee022d99ee7a56113c6c6d72978eb05430814349da129c78cf9fc18def
```

Validation 100 条只在参数冻结后运行一次，没有用于回头选择参数。

## 5. 正式三组消融实验

按照最终决定，本报告只比较三组完整实验：

1. **Step29**：大模型能理解语义，但语义不进入最终排序；
2. **LLM Aggressive**：第 29 步语义进入排序，直接替换 Top-20；
3. **LLM Protected**：第 29 步语义进入排序，但增加置信度、证据和移动范围保护。

三组均覆盖同一批 500 个 Agent 场景，其中 Development 400 条、Validation 100 条。HR/MRR/NDCG 的分母是其中 230 个带目标商家的推荐场景，不是所有 500 个混合任务。

### 5.1 全部 500 场景结果

| 指标 | Step29 | LLM Aggressive | LLM Protected |
|---|---:|---:|---:|
| Action Accuracy | 79.00% | 79.00% | 79.00% |
| Tool Selection Accuracy | 78.10% | 81.33% | 81.28% |
| Direct Return Precision | 75.81% | 76.10% | 76.14% |
| HR@1 | **5.65%** | 4.78% | 5.22% |
| HR@3 | **10.87%** | 10.43% | **10.87%** |
| HR@5 | 14.78% | 14.78% | **15.22%** |
| MRR | **10.27%** | 9.59% | 10.12% |
| NDCG@5 | 4.27% | 4.11% | **4.28%** |
| Fallback Rate | **0.40%** | 0.60% | **0.40%** |
| 平均延迟 | 3076 ms | 2620 ms | **2477 ms** |
| P95 延迟 | 9236 ms | 7721 ms | **7447 ms** |

### 5.2 冻结后的 Validation 100 场景

| 指标 | Step29 | LLM Aggressive | LLM Protected |
|---|---:|---:|---:|
| Action Accuracy | 78.89% | 78.89% | 78.89% |
| Tool Selection Accuracy | 78.97% | 82.11% | **82.24%** |
| HR@1 | 4.35% | 4.35% | 4.35% |
| HR@3 | 10.87% | 10.87% | 10.87% |
| HR@5 | 17.39% | 17.39% | 17.39% |
| MRR | **9.80%** | 9.53% | 9.53% |
| NDCG@5 | **4.92%** | 4.85% | 4.85% |
| Fallback Rate | 1.00% | 1.00% | 1.00% |

Validation 的推荐样本量较小，因此三组 HR@5 相同并不表示候选顺序完全相同；它只表示目标商家是否进入前 5 的二值结果相同。这里不根据 Validation 再修改保护参数。

## 6. 排名究竟改变了多少

在 500 个场景中，280 次执行产生 Step30 排序诊断；其中 219 次同时具备可比较的基础排名和目标商家。

| 排序诊断 | LLM Aggressive | LLM Protected |
|---|---:|---:|
| 排名发生变化 | 59.29% | 59.29% |
| 没有可排序语义而原样返回 | 40.71% | 40.71% |
| 目标商家排名改善 | 1.83% | **2.28%** |
| 目标商家排名受损 | 2.74% | **2.28%** |
| Top-3 保护被触发 | 0.00% | 46.07% |
| 候选平均绝对移动 | 5.26 位 | **2.26 位** |
| Step30 排序工具回退 | 0.00% | 0.00% |

结论很清楚：

- 激进融合确实让当前语义强烈影响排序，但没有提高总体 HR@5，同时降低 HR@1、MRR 和 NDCG@5；
- 保守融合将平均扰动减少约 57%，目标受损率从 2.74% 降到 2.28%；
- 保守融合的总体 HR@5 比 Step29 提高 0.43 个百分点，但 HR@1 和 MRR 仍略低；
- 冻结 Validation 上 HR@5 没有提升，因此现在只能说保守融合更安全、总体结果有小幅正向信号，不能声称它已经稳定显著优于 Step29。

## 7. Token 和调用成本口径

必须区分两种完全不同的 token。

### 7.1 本地模型 token

Embedding 和 Cross-Encoder 都从本地磁盘加载，不调用 API，不计费。两组完整回放记录的逻辑输入量约为：

| 组别 | 本地模型逻辑输入 token | 本次实际重新计算 token |
|---|---:|---:|
| LLM Aggressive | 2,397,705 | 285,130 |
| LLM Protected | 2,397,248 | 5,084 |

“逻辑输入”表示如果完全不使用缓存需要处理多少 token；“实际重新计算”受本地缓存影响。保守组大量命中激进组已经写好的模型缓存，所以实际重新计算量更低。这些数字都不是 DashScope 或 DeepSeek 账单。

### 7.2 DeepSeek API token

Step29 的已缓存正式 500 场景账本记录了 198,698 个已知 provider token。第 30 步重放时，成功语义主要命中旧缓存；其余 provider 请求失败且没有返回 usage，因此第 30 步新增的**可确认计费 token 为 0**，但不能把失败请求记成“绝对没有消耗”。账本将这 185/186 次未返回 usage 的 provider 尝试明确标为 `usage_unknown`。

因此报告不会用本地 tokenizer 数量冒充 API 消耗，也不会对 provider 未返回的 token 做猜测。

## 8. 失败处理与数据安全

- Ground Truth 只进入离线调参脚本和评测器，不进入 Router、AgentState 或排序工具；
- Business Profile 按每个任务的 `cutoff_time` 读取；
- 排序输出必须是当前候选全集的唯一排列；
- 无语义条件时精确保留 Step26 顺序；
- 模型异常、候选缺失、重复 ID 或范围越界时精确回退 Step26；
- 每个任务记录基础顺序、新顺序、每项分数、排名移动、保护原因、缓存命中、token 和回退原因；
- API Key、认证 Header 和完整敏感请求不写入 trace。

## 9. 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\run_step30_agent_benchmark.py `
  --output-root runs\semantic_ranking_v1\llm_aggressive\development `
  --split development `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda `
  --semantic-ranking-config configs\semantic_ranking_aggressive.yaml

.\.venv\Scripts\python.exe scripts\run_step30_agent_benchmark.py `
  --output-root runs\semantic_ranking_v1\llm_protected\validation `
  --split validation `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --cross-encoder-model-path D:\model\Qwen3-Reranker-0.6B `
  --model-python D:\anaconda3\python.exe --device cuda `
  --semantic-ranking-config configs\semantic_ranking.yaml

.\.venv\Scripts\python.exe scripts\compare_step30_ablations.py `
  --variant step29=runs\controlled_llm_v1\agent\full `
  --variant llm_aggressive=runs\semantic_ranking_v1\llm_aggressive\full `
  --variant llm_protected=runs\semantic_ranking_v1\llm_protected\full `
  --output-root runs\semantic_ranking_v1\full_comparison
```

## 10. 当前结论和下一步边界

第 30 步证明了两件事：第一，受控 LLM 语义已经不只是写在 JSON 里，而是真正进入本地模型并改变推荐顺序；第二，直接相信语义模型会破坏原排序，排名保护是必要的。

当前提升仍然很小，主要瓶颈包括：召回阶段约一半任务没有把目标商家放入候选、Aspect 证据覆盖有限、500 场景中真正带可评测排序目标的样本较少，以及离线下一次交互并不总能代表用户当前自然语言意图。后续应优先改善召回和构建更直接的 Query–Business 相关性标注，再考虑学习融合权重；不能只靠继续增加 LLM 自由度来解决。
