# 第 33 步：Query 感知候选保护与学习排序

## 1. 本步解决的问题

第 32 步已经证明，当前 Query 能把 History 召回漏掉的大量真实正例找回来，但旧的等权 RRF 会在 Top-500 截断时重新丢失一部分 Query 候选。第 33 步不再把九条底层路线直接堆成一个巨大候选池，而是只接收两条已经完成内部融合的通道：

- History Top-500：历史偏好、质量、类别、TF-IDF、位置和 Item-KNN 的最终召回；
- Query Top-500：类别、当前 Query Embedding、Aspect 和位置的最终召回。

两路结果先做受保护并集，去重后最多 1000 家。在模型排序之前，不再用 RRF 截成 500 家。完整流程是：

```text
History Top-500 ─┐
                 ├─ 受保护并集（最多 1000）
Query Top-500 ───┘
        ↓
确定性硬约束过滤
        ↓
冻结 Hybrid V2-B LambdaMART 分数
        +
当前 Query 召回与 Embedding 分数
        ↓
Query-aware 粗排
        ↓
Top-30 本地 Qwen3-Reranker 精排
        ↓
内部 Top-10 / 展示 Top-5
```

这里没有重新训练 LightGBM。历史个性化部分继续使用第 17 步冻结的 `without_review_aspect` LambdaMART；第 33 步只在 Development 上选择“历史模型分数”和“当前 Query 分数”的融合权重。

## 2. 数据与隔离规则

正式实验继续使用第 32 步冻结的 500 条 Query Recommendation Benchmark：

- Development：400 条，只允许用于选择 Query 融合权重；
- Validation：100 条，只在策略冻结后运行和报告；
- 每条题目只有一个后来真实访问且至少给出 4 星的已知正例；
- Query 是根据 cutoff 前可证明条件生成的半合成自然语言，不是真实搜索日志；
- 运行时只读取 `visible/cases.jsonl`；ground truth 只由调权器和独立评测器读取；
- 排序引擎、候选池和本地模型接口都不接受 target 参数或 ground-truth 路径。

静态商家文本可以进入 Embedding 与 Cross-Encoder。用户行为、评分、质量、Aspect 证据和商家可用性均受 `cutoff_time` 约束。硬约束排除发生在模型排序之前，被排除商家不能被语义模型重新放回。

## 3. Development 调权

只搜索三个预先写入配置的 Query 权重，目标为：

```text
AvgHR = (HR@1 + HR@3 + HR@5 + HR@10) / 4
```

并列时依次比较 MRR、HR@1、NDCG@10，最后选择更低的 Query 权重。

| Query 权重 | HR@1 | HR@3 | HR@5 | HR@10 | AvgHR | MRR | NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.75% | 13.25% | 16.75% | 22.50% | 14.81% | 12.06% | 13.78% |
| **0.50** | **7.25%** | 12.75% | **17.25%** | **22.75%** | **15.00%** | **12.69%** | **14.10%** |
| 0.75 | 7.00% | 11.25% | 16.00% | 20.75% | 13.75% | 11.98% | 13.00% |

冻结结果为：历史 LambdaMART 百分位占 0.50，当前 Query 信号占 0.50。Query 信号内部由 Query 召回百分位 0.80 和全并集 Embedding 百分位 0.20 组成。

冻结策略文件：`configs/query_aware_ranking_policy.json`

SHA256：

```text
10cf3282e3db035d385b50948406858bcab39397cf7d2af6657787e96f5d649e
```

## 4. 正式 500 条结果

### 4.1 总体结果

| 方法 | HR@1 | HR@3 | HR@5 | HR@10 | AvgHR | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| History + 冻结 LightGBM | 0.80% | 2.20% | 3.60% | 6.80% | 3.35% | 3.11% | 3.20% |
| Query Only | 2.80% | 7.80% | 10.40% | 15.60% | 9.15% | 7.72% | 8.46% |
| 旧 History + Query RRF | 4.00% | 7.40% | 11.00% | 16.60% | 9.75% | 8.51% | 9.27% |
| LightGBM + Query 粗排 | 6.80% | **12.60%** | **16.80%** | 22.20% | 14.60% | 12.25% | 13.64% |
| **最终 Query-aware 精排** | **7.40%** | 12.20% | **16.80%** | **22.80%** | **14.80%** | **12.62%** | **14.08%** |

最终方法相对旧 RRF：

- HR@1：4.00% → 7.40%，提高 3.40 个百分点；
- HR@3：7.40% → 12.20%，提高 4.80 个百分点；
- HR@5：11.00% → 16.80%，提高 5.80 个百分点；
- HR@10：16.60% → 22.80%，提高 6.20 个百分点；
- AvgHR：9.75% → 14.80%，提高 5.05 个百分点。

### 4.2 冻结 Validation 结果

| 方法 | HR@1 | HR@3 | HR@5 | HR@10 | AvgHR | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| History + 冻结 LightGBM | 1.00% | 2.00% | 5.00% | 9.00% | 4.25% | 3.23% | 4.05% |
| Query Only | 2.00% | 9.00% | 14.00% | 19.00% | 11.00% | 8.41% | 9.64% |
| 旧 History + Query RRF | 4.00% | 9.00% | 13.00% | 18.00% | 11.00% | 9.32% | 10.37% |
| LightGBM + Query 粗排 | 5.00% | **12.00%** | 15.00% | 20.00% | 13.00% | 10.50% | 11.80% |
| **最终 Query-aware 精排** | **6.00%** | 11.00% | **16.00%** | **21.00%** | **13.50%** | **11.06%** | **12.49%** |

Validation 没有参与权重选择。最终精排相对粗排提高 HR@1、HR@5、HR@10、MRR 和 NDCG@10，但 HR@3 从 12% 降到 11%。因此结论应当是“本地精排总体产生小幅正向收益”，而不是“所有指标都提高”。

## 5. 候选漏斗

500 条任务的候选流转如下：

| 检查项 | 结果 |
|---|---:|
| History 或 Query 至少一路召回 target | 473/500，94.60% |
| 硬约束后 target 仍在受保护并集 | 473/500，94.60% |
| Query Top-500 target 保留率 | 439/439，100% |
| 旧 RRF Top-500 丢失 | 39/473，8.25% |
| 新粗排 Top-500 丢失 | 38/473，8.03% |
| 最终 Top-10 硬约束违规 | 0/500 |

平均每个任务的原始并集是 859.49 家，硬约束后为 823.93 家，两路平均重叠 138.60 家。122 个任务产生了硬约束排除记录，共排除 17,779 个“任务 × 商家”组合；没有正确答案被硬约束误删。

受保护并集把排序前的候选上限从旧 RRF 的 86.8% 提高到 94.6%。但是模型排序后的 Top-500 仍只有 87.0% Recall，说明“候选是否存在”和“模型能否把它排到前面”是两件不同的事。第 33 步解决了过早截断，不代表排序问题已经解决。

## 6. Cross-Encoder 的实际贡献

相对 Query-aware 粗排，最终本地精排在 500 条任务上：

- target 排名改善 37 条；
- target 排名受损 34 条；
- target 排名不变 402 条；
- 27 条任务的 target 根本没有进入受保护并集，因此无法被排序模型救回。

在冻结 Validation 中，target 排名改善 9 条、受损 3 条、不变 84 条。精排只处理粗排 Top-30，并继续使用第 30 步的保守移动规则：最多上升 5 位、最多下降 3 位、保护原 Top-3。它的主要价值是调整前排，而不是重新搜索整个目录。

## 7. 延迟、Token 与费用

最终完整方法在 500 条上的平均任务延迟为 4264.84 ms，P95 为 7581.49 ms。该数字包含候选召回、特征准备和精排，但不包含每次进程启动时的一次性模型加载时间；Embedding 商家文档已命中本地缓存，因此它不是冷缓存结果。

本步严格使用本地模型：

| 项目 | 数量 |
|---|---:|
| 外部模型/API 调用 | 0 |
| 外部计费 Token | 0 |
| 本次新计算 Embedding Token | 0 |
| 本次新计算 Cross-Encoder Token | 5,757,979 |
| 第 33 步增量逻辑输入 Token | 96,451,680 |
| 加上第 31/32 步 Query 召回后的全链路逻辑输入 Token | 358,651,277 |

“逻辑输入 Token”表示如果完全没有缓存，本次请求需要读取的文本总量；它不是 API 账单。代码增加了强制安全门：正式 Step 33 一旦发现 Embedding 或 Cross-Encoder 产生外部 API 调用，会直接拒绝运行。

## 8. 工程实现

核心模块位于 `src/yelp_agent/query_aware_ranking/`：

- `candidate_pool.py`：保护两路最终 Top-500，并在排序前执行硬排除；
- `scoring.py`：融合冻结 LambdaMART 百分位和当前 Query 信号；
- `engine.py`：统一准备、Top-30 语义精排、完整排列校验与失败回退；
- `runtime.py`：组装真实 Yelp 时点数据、本地 Embedding、本地 Reranker 和冻结模型；
- `tuning.py`：只在 Development 上选权重；
- `evaluation.py`：计算 HR@1/3/5/10、AvgHR、MRR、NDCG、召回漏斗、延迟和安全指标；
- `schema.py`：冻结 target-blind 中间产物和最终结果合同；
- `config.py`：配置与冻结策略校验。

运行脚本支持 `prepare|tune|finalize|evaluate|all`，并为每个场景保存原子缓存。重跑时只有缺失场景才启动全量数据和本地模型，已完成任务直接校验并复用。序列化统一排除 Pydantic 自动计算字段，修复了严格重载时的兼容问题。

主要实验产物位于：

```text
runs/query_aware_ranking_v1/prepared_cases.jsonl
runs/query_aware_ranking_v1/development_selection.json
runs/query_aware_ranking_v1/final_results.jsonl
runs/query_aware_ranking_v1/benchmark_runs.jsonl
runs/query_aware_ranking_v1/metrics.json
```

`metrics.json` SHA256：

```text
f895cfb159fc19d341ef4cfbf27186b160054ef42da3c2acdbf355b996677944
```

## 9. 复现命令

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python scripts/run_step33_query_aware_ranking.py `
  --source-root C:\Users\29072\PycharmProjects\AgentSociety `
  --project-root . `
  --config-root . `
  --benchmark-root benchmarks\query_recommendation_v1 `
  --output-root runs\query_aware_ranking_v1 `
  --embedding-model-path D:\models\Qwen3-Embedding-0.6B `
  --embedding-python D:\anaconda3\python.exe `
  --embedding-device cuda `
  --cross-model-path D:\model\Qwen3-Reranker-0.6B `
  --cross-python D:\anaconda3\python.exe `
  --cross-device cuda `
  --stage all
```

## 10. 当前边界

本实验仍是“单个已知正例”评测，不能把未访问商家视为真实负例。Query 是基于正确商家的 cutoff 前可证明属性构造的半合成问句，因此适合验证 Query 是否进入召回与排序，但不能替代真实线上搜索、点击、预订和满意度日志。

第 33 步已经证明：历史推荐不是废弃信号，当前 Query 也不能只做提示词；二者需要在候选保护之后共同进入排序。下一阶段应在这个冻结排序入口之上增加会话记忆、临时反馈和动态工具选择，而不是回到仅靠一次 LLM Top-N 重排的旧 Agent V0。
