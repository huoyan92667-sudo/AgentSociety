# 第 31–33 步：实验文档与 GitHub 同步清单

> 本文用于记录第 31、32、33 步的实验报告位置、主要代码、数据产物和 GitHub 同步状态。
> 它不是新的算法步骤，只是项目文档索引和版本核对记录。

## 1. 当前结论

第 31、32、33 步都已经有独立的 Markdown 实验报告：

| 步骤 | 实验报告 | 本地提交 | GitHub 状态 |
|---|---|---|---|
| 31 | `docs/experiments/step31_Query召回_完整实现与真实数据冒烟报告.md` | `390848f` | 已上传到 `codex/step-31-query-recall` |
| 32 | `docs/experiments/step32_Query推荐Benchmark_构建与三路召回报告.md` | `b5ecf3c` | 已上传到 `codex/step-32-query-recommendation-benchmark` |
| 33 | `docs/experiments/step33_Query感知候选保护与学习排序报告.md` | `f883ace` | 当前只有本地提交，尚未上传远程分支 |

远程查询时间：2026-08-13。远程仓库中可确认的分支为：

- `main`：`4ab0a41`
- `codex/step-31-query-recall`：`390848f`
- `codex/step-32-query-recommendation-benchmark`：`b5ecf3c`

远程暂时没有 `codex/step-33-query-aware-ranking`。

## 2. 第 31 步：Query 召回

报告：

`docs/experiments/step31_Query召回_完整实现与真实数据冒烟报告.md`

主要代码：

- `src/yelp_agent/query_retrieval/`
- `src/yelp_agent/agent_tools/adapters/retrieval.py`
- `scripts/run_query_retrieval.py`
- `configs/query_retrieval.yaml`
- `tests/test_query_retrieval.py`

主要内容：

- 将用户当前 Query 与历史个性化召回分开；
- 支持类别、语义、Aspect 和位置等 Query 相关通道；
- 通过内部融合生成 Query 候选；
- 保留统一工具接口和召回审计；
- 使用真实 Yelp 数据进行冒烟和召回评估。

Git 提交：

`390848f feat: add dual-channel query recall`

## 3. 第 32 步：Query Recommendation Benchmark

报告：

`docs/experiments/step32_Query推荐Benchmark_构建与三路召回报告.md`

主要代码：

- `src/yelp_agent/query_recommendation_benchmark/`
- `scripts/build_query_recommendation_benchmark.py`
- `scripts/run_query_recommendation_benchmark.py`
- `configs/query_recommendation_benchmark.yaml`
- `tests/test_query_recommendation_benchmark.py`

主要数据：

- `benchmarks/query_recommendation_v1/visible/cases.jsonl`
- `benchmarks/query_recommendation_v1/hidden/ground_truth.jsonl`
- `benchmarks/query_recommendation_v1/hidden/structured_frames.jsonl`
- `benchmarks/query_recommendation_v1/audit/`

主要内容：

- 构造 500 条 Query Recommendation 场景；
- 将用户可见 Query 与隐藏正确商家分开；
- 对历史、Query 和融合召回分别评估；
- 增加数据生成和泄漏审计；
- 为后续 Query 感知排序提供固定评测入口。

Git 提交：

`b5ecf3c feat: add query recommendation benchmark`

## 4. 第 33 步：Query 感知候选保护与学习排序

报告：

`docs/experiments/step33_Query感知候选保护与学习排序报告.md`

主要代码：

- `src/yelp_agent/query_aware_ranking/`
- `scripts/run_step33_query_aware_ranking.py`
- `configs/query_aware_ranking.yaml`
- `configs/query_aware_ranking_policy.json`
- `tests/test_query_aware_ranking_v2.py`

主要内容：

- 对 History Top-500 和 Query Top-500 做候选保护并集；
- 在排序前去重，避免 Query 候选被过早截断；
- 继续复用第 17 步冻结的 LightGBM 历史排序模型；
- 在 Development 上选择 History 与 Query 分数的融合权重；
- 对 Validation/Test 只加载冻结策略，不重新调参；
- 输出候选召回、最终排序和消融评估。

Git 提交：

`f883ace feat: add query-aware protected ranking`

## 5. 当前工作区的额外状态

当前工作区还有一处未提交的代码变化：

`src/yelp_agent/features/hybrid.py`

它只是文件末尾的空白换行变化，不属于第 31–33 步的核心实现，也没有被计入上述三个提交。后续提交第 34 步前应先确认是否保留或清理这处无意义变更。

## 6. 下一步版本动作

第 33 步如果需要上传到 GitHub，应创建或切换到：

`codex/step-33-query-aware-ranking`

然后提交第 33 步代码和本清单文档，再推送到同名远程分支。当前没有自动推送，避免在用户未确认时改变远程仓库状态。
