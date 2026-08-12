# 第 31 步：Query 召回完整实现与真实数据冒烟报告

## 1. 本步骤解决什么问题

以前的五路召回主要回答：

> 根据这个用户过去喜欢什么，哪些商家可能适合他？

这一召回仍然有用，但无法充分回答当前请求。例如用户今天说：

> 我只想找市政厅 5 公里内、安静、适合求婚的牛排馆。

本步骤增加独立的 Query 召回通道，回答：

> 根据用户这一次明确说出的需求，哪些商家可能相关？

随后用第二层 RRF 将“历史召回”和“Query 召回”合并。旧召回没有删除，也没有被 Query 召回替代。

本步骤只完成召回系统和无标签冒烟测试，**没有构建或读取 Query Recommendation Benchmark，也没有使用正确商家标签调参**。

## 2. 数据边界

Query 召回搜索的商家范围是 Philadelphia 的 3973 家有效商家，但每家商家必须在当前任务 `cutoff_time` 之前至少出现过一条评论。

不同信息使用不同来源：

- 商家名称、类别、属性、经纬度：静态商家表。
- 商家是否在当时已经存在：`cutoff_time` 之前的评论目录。
- Query Embedding：商家名称、类别和静态属性组成的静态文档。
- Aspect 证据：只使用选中 5000 名用户评论所构建的 `selected_user_interactions` 商家知识产物。

本步骤没有把 Yelp 全量评论转换成完整 Aspect 索引，也没有读取 5000 名用户之外的评论来补 Aspect。Aspect 没有证据时保存为 `unknown`，不会把“没看见证据”当成“不满足”。

当前冻结的商家知识产物包含：

- 3973 家商家；
- 617718 条评分事件；
- 99073 条 Aspect 事件；
- 3519 家至少有一条 Aspect 事件的商家；
- `aspect_source_mismatches = 0`。

## 3. 四路 Query 召回

### 3.1 Query Category

根据结构化请求中的目标类别召回，例如 `Steakhouses`。只有真正命中目标类别的商家进入这一路，不会用无关商家凑满 Top-500。

### 3.2 Query Embedding

将完整结构化需求编译成一段稳定文本，再使用本地 `Qwen3-Embedding-0.6B` 与商家静态文档计算相似度。商家向量写入 SQLite 缓存；以后新 Query 通常只需计算一个 Query 向量。

该路不调用 DashScope 或 DeepSeek。运行记录同时区分：

- `embedding_input_tokens`：本次真正送入本地模型重新计算的 Token；
- `embedding_logical_tokens`：如果没有缓存，本次所有文本合计对应的 Token；
- `cache_hits/cache_misses`：缓存命中情况；
- `provider_calls`：外部 API 调用次数，本地模型应为 0。

### 3.3 Query Aspect

将 `quiet_environment`、`date_suitable`、`parking` 等当前需求与商家 Aspect 聚合证据匹配。

规则是：

- 已知且方向与 Query 一致：进入 Aspect 召回路；
- 已知但方向相反：不从 Aspect 路召回，但仍可通过类别、Embedding、位置或历史通道进入；
- 未知：保持中性，不进入 Aspect 路，也不因软偏好被删除；
- 如果用户把 Aspect 明确设为硬条件，则已知冲突会被删除；未知按照结构化请求的 `unknown_policy` 处理。

为了避免全目录重复构造完整商家画像，`BusinessKnowledgeStore.get_aspects()` 只读取本次 Query 涉及的一两个 Aspect。它和完整画像使用同一个冻结事件索引及 `< cutoff_time` 规则。

### 3.4 Query Location

如果当前请求带用户位置，则按当前位置与商家的 Haversine 距离召回，并用：

`location_score = exp(-distance_km / 10)`

计算位置分数。硬距离条件会在位置召回之前执行，因此 5 公里以外的商家不能靠其他 Query 路重新进入。

## 4. 硬条件先过滤，软偏好只影响召回

在四路召回之前，代码先执行可可靠验证的硬条件：

- 必须/禁止的类别；
- 最大距离；
- Yelp 静态价格档位；
- 有足够证据的硬 Aspect 条件。

如果硬条件所需字段未知，则按 `unknown_policy` 决定排除、追问或带警告保留。`budget_per_person` 无法从 Yelp 静态价格档位精确证明，因此不会伪造精确人均价格。

软偏好不会直接删除商家。例如“最好安静”不会删除 Aspect 未知的商家，它只让有正向安静证据的商家进入 Aspect 召回路。

## 5. 两层融合

Query 内部使用加权 RRF 合并四路名次：

`QueryScore(b) = Σ route_weight / (60 + route_rank)`

当前四路权重都为 1.0。这是透明的未调参基线，不是用 Benchmark 选出的最优权重。

Agent 的 `EXPAND_CANDIDATES` 工具随后再融合：

- 旧的历史五路召回；
- 新的当前 Query 四路召回。

两通道都召回的商家会得到两次 RRF 支持；只被其中一路发现的商家也会保留。每条候选保存历史名次、Query 名次、四个 Query 子路名次和来源通道，方便后续分析。

如果 Query 召回出现异常，工具会记录 `QUERY_RETRIEVAL_FALLBACK:<ExceptionType>`，并按原始顺序返回完整历史召回结果，一个任务失败不会中断整批任务。

## 6. 真实数据冒烟结果

测试 Query：

> I only want a quiet romantic steakhouse within 5 km for a proposal dinner

位置：Philadelphia City Hall，cutoff：2022-01-01。

| 项目 | 结果 |
|---|---:|
| 全部商家 | 3973 |
| cutoff 前已存在 | 3973 |
| 5 公里硬过滤后 | 2898 |
| Query Category | 34 |
| Query Embedding | 500 |
| Query Aspect | 293 |
| Query Location | 500 |
| 最终 Query 候选 | 500 |
| 召回核心耗时 | 1199 ms |
| 本地模型实际输入 Token | 140 |
| 缓存命中 | 2898 |
| 缓存未命中 | 1（本次 Query） |
| 外部 Provider 调用 | 0 |

本次查询中：

- `quiet_environment` 有 2562 家证据未知；
- `date_suitable` 有 2770 家证据未知；
- 未知商家没有被伪装成负样本；
- 模型首次载入使整条命令墙钟时间约 57 秒，模型常驻后的召回核心约 1.2 秒；
- 不启用 Embedding 时，类别、Aspect、位置三路核心约 53 ms。

Top-5 冒烟结果依次为：

1. Del Frisco's Double Eagle Steakhouse；
2. Alpen Rose；
3. Butcher and Singer；
4. Malbec Argentine Steakhouse；
5. The Capital Grille。

这些结果只证明流程和数据边界正确，**不能证明推荐准确**。下一步 Query Recommendation Benchmark 才会提供 Query 条件下的相关商家标签，并评估 Recall@50/100/500、硬约束违规率和各召回路增益。

## 7. 运行方法

不启用本地 Embedding：

```powershell
python scripts/run_query_retrieval.py `
  --source-root C:\path\to\AgentSociety `
  --config-root . `
  --query "I want a quiet steakhouse within 5 km" `
  --latitude 39.9526 `
  --longitude -75.1652
```

启用本地 Embedding：

```powershell
python scripts/run_query_retrieval.py `
  --source-root C:\path\to\AgentSociety `
  --config-root . `
  --embedding-config configs\embedding.yaml `
  --model-path D:\models\Qwen3-Embedding-0.6B `
  --model-python D:\anaconda3\python.exe `
  --device cuda `
  --query "I want a quiet romantic steakhouse within 5 km" `
  --latitude 39.9526 `
  --longitude -75.1652
```

输出 JSON 会明确写入：解析后的请求、四路结果数、Top-500 全部候选、每条候选的路由来源、过滤原因、警告、缓存与 Token 使用量。报告固定声明 `benchmark_loaded=false` 和 `ground_truth_loaded=false`。

## 8. 本步骤不做什么

- 不构建 Query Recommendation Benchmark；
- 不生成 Query 的正确商家标签；
- 不用标签调 RRF 权重；
- 不把全量 Yelp 评论做成 Aspect 索引；
- 不声称 Recall、HR 或 NDCG 有提升；
- 不删除原有历史五路召回；
- 不把临时 Query 写入用户长期画像。
