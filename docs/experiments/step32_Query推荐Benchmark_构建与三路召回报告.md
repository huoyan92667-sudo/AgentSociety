# 第 32 步：Query Recommendation Benchmark 构建与三路召回报告

## 1. 本步解决了什么问题

此前的推荐任务只有“用户历史 → 下一家真实商家”，可以衡量个性化历史召回，
却不能回答一个更贴近真实 Agent 的问题：

> 用户临时提出“想吃什么、距离多远、价格如何、环境怎样”以后，当前 Query
> 是否真的改变了候选商家？

本步构建了 500 条半合成 Query Recommendation Benchmark，并只比较三种必要方法：

1. `History Only`：只使用用户过去的行为；
2. `Query Only`：只使用当前自然语言需求；
3. `History + Query`：对两路 Top-500 使用等权 RRF 融合。

本步没有继续增加权重、路由或模型消融实验。

## 2. 500 条题目是怎么来的

### 2.1 正确答案是真实行为，不是大模型编的

候选来源是原有时间任务中的 22,920 个训练/验证时点。代码按 seed 42
确定性选择 500 个任务，并要求：

- 目标评论评分至少 4 星；
- 目标商家不在用户此前历史中；
- 目标商家在 cutoff 之前至少已有一条其他评论，证明当时已经存在；
- development 与 validation 用户完全隔离；
- 不使用 test 任务。

最终划分为：

- development：400 条；
- validation：100 条；
- 中文：300 条；
- 英文：200 条。

这里的正确答案只有一个“已知正例”：用户后来真实访问并给出至少 4 星的商家。
它不表示其他未访问商家一定不相关。

### 2.2 Query 是怎样生成的

代码先从 cutoff 时刻之前可证明的信息构造隐藏的结构化条件，再让
`deepseek-v4-flash` 只负责把条件改写为自然问句。七类题型数量如下：

| 题型 | 数量 |
|---|---:|
| 类别 | 80 |
| 类别 + 距离 | 80 |
| 类别 + 价格 | 60 |
| 类别 + 一个 Aspect | 100 |
| 类别 + 两个 Aspect | 80 |
| 类别 + 距离 + Aspect | 60 |
| 场景/人数需求 | 40 |

DeepSeek 不接收用户 ID、目标商家 ID、目标评论 ID、评论正文、评分、商家名称、
地址或精确经纬度。目标 ID 只在模型调用结束以后由本地代码写入隐藏答案。

生成后还执行了：

- 独立条件忠实度审计；
- 全局 Query 去重；
- 类别碎片检查，例如不把 Yelp 的 `Mags` 切分碎片作为独立需求；
- 类别与 Aspect 兼容性检查，例如禁止“服装店 + food_quality”；
- target 支持、时间、历史隔离、用户 split 和身份泄漏审计。

最终泄漏审计结果为 500/500 通过，违规数为 0。

## 3. 数据隔离

Agent 可见文件：

```text
benchmarks/query_recommendation_v1/visible/cases.jsonl
```

隐藏文件：

```text
benchmarks/query_recommendation_v1/hidden/ground_truth.jsonl
benchmarks/query_recommendation_v1/hidden/structured_frames.jsonl
```

召回运行时只加载 visible cases。三路排名全部写完以后，独立评测器才读取
`ground_truth.jsonl`。召回 Runtime 的公开接口没有 target 参数。

Aspect 证据仍严格限制在 `selected_user_interactions`，也就是此前选中的
5,000 名用户评论，没有扩展到全 Yelp 评论。

## 4. 正式 500 条结果

### 4.1 总体结果

| 方法 | Recall@50 | Recall@100 | Recall@500 | MRR@500 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| History Only | 18.2% | 27.2% | 57.8% | 0.0269 | 1728.8 ms |
| Query Only | 37.6% | 52.2% | **87.8%** | 0.0772 | **312.3 ms** |
| History + Query | **41.6%** | **54.0%** | 86.8% | **0.0851** | 2041.1 ms |

### 4.2 Validation 结果

| 方法 | Recall@50 | Recall@100 | Recall@500 | MRR@500 |
|---|---:|---:|---:|---:|
| History Only | 15.0% | 25.0% | 52.0% | 0.0218 |
| Query Only | **44.0%** | 56.0% | **94.0%** | 0.0841 |
| History + Query | 40.0% | **60.0%** | 87.0% | **0.0932** |

### 4.3 怎么理解这些结果

在 History Only 没召回目标的 211 条任务中，Query Only 救回了 184 条，
Query Rescue Rate 为 **87.20%**。这说明前期历史召回并没有失去作用，但当前
Query 是覆盖用户临时需求盲区的关键新通道。

等权融合并非全面优于 Query Only：

- Top-50、Top-100 和 MRR 更高，说明历史偏好能帮助把一部分目标提前；
- Recall@500 从 87.8% 降到 86.8%，说明少量 Query 命中的目标被历史候选挤出；
- 两个通道至少一个命中目标的 473 条任务中，有 39 条被融合结果丢失，
  Fusion Loss Rate 为 **8.25%**。

因此当前结论不是“融合一定更好”，而是：

> Query 通道显著提高候选覆盖；历史通道对前排排序有帮助；当前等权 RRF
> 需要在未来通过有保护的融合策略减少 Top-500 覆盖损失。

## 5. Token 与运行成本

Benchmark 生成与审计的最终有效调用逻辑合计：

- 输入 Token：139,103；
- 输出 Token：68,846；
- 合计：207,949。

把前期被质量门拒绝、格式失败和语义修复后作废、但实际已经调用的缓存结果也算入，
DeepSeek 累计实际 Token 为 **253,074**。

正式 500 条召回使用本地 `Qwen3-Embedding-0.6B`：

- 新编码 Token：36,456；
- 外部 Embedding API 调用：0；
- 外部 LLM Token：0。

日志中的 262,199,597 logical tokens 包含每条任务重复读取的缓存商家文档 Token，
不是新增推理量，也不产生 API 费用。

## 6. 产物与复现命令

核心产物：

```text
benchmarks/query_recommendation_v1/manifest.json
benchmarks/query_recommendation_v1/audit/leakage_audit.json
benchmarks/query_recommendation_v1/audit/generation_report.json
runs/query_recommendation_v1/retrieval/retrieval_runs.jsonl
runs/query_recommendation_v1/retrieval/metrics.json
```

生成 Benchmark：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python scripts/build_query_recommendation_benchmark.py `
  --source-root C:\Users\29072\PycharmProjects\AgentSociety `
  --config-root . `
  --allow-real-api
```

运行三路召回：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python scripts/run_query_recommendation_benchmark.py `
  --source-root C:\Users\29072\PycharmProjects\AgentSociety `
  --config-root . `
  --model-path D:\models\Qwen3-Embedding-0.6B `
  --model-python D:\anaconda3\python.exe `
  --device cuda
```

## 7. 当前限制与下一步

本 Benchmark 评测的是“单个真实已知正例能否被召回”，不能证明 Top-500 里的
每一家都真正满足用户，也不能把未访问商家当作负例。Query 是基于真实目标的
可证明属性半合成生成，因此适合比较召回通道，但仍不等同于真实线上自由提问。

下一步不需要马上增加大量消融实验。更有价值的是：

1. 先保留本结果作为冻结基线；
2. 让 Query 条件真正进入后续硬约束与精排；
3. 设计不会牺牲 Query Top-500 覆盖的保守融合；
4. 之后再对一小部分 Query 做人工相关性多标注，补充单正例评测的局限。
