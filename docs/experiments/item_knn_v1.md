# 时点安全 Item-KNN 与 Full Retrieval V2

## 实验范围

本实验对应路线 V2 的第 12 步，只增加时点安全 Item-KNN。它不实现 Review Aspect、用户画像 V1、Hybrid V2 或 Agent，也不调用任何 LLM。

Item-KNN 为原有质量、类别、TF-IDF 文本和位置四路召回补充协同行为信号：如果许多用户共同喜欢两家商家，这两家商家获得正向相似关系；共同给出 1–2 星只形成独立负向证据，不在本步骤手工扣减正向分数。

## 数据和时间安全

原始 `interactions.parquet` 有 139,402 条交互和 5,000 名用户。建图前执行以下规则：

1. 每名用户最后两条行为作为 validation/test 保留目标，从图事件中移除。
2. 4–5 星写入正向事件，1–2 星写入负向事件，3 星写入中性状态更新事件。
3. 同一用户多次评价同一家商家时，每个任务只使用 `cutoff_time` 前最新一次评分。
4. 图只推进到严格早于任务 cutoff 的事件；追加 cutoff 后数据不会改变旧任务结果。
5. 当前项目没有独立 Final Blind Holdout 用户，因此本次排除用户数为 0；实现保留了按 `user_id` 排除整组用户的接口。

冻结后的训练安全事件：

| 事件 | 数量 |
|---|---:|
| 原始交互 | 139,402 |
| 移除的最后两条目标 | 10,000 |
| 可建图交互 | 129,402 |
| 正向事件 | 92,549 |
| 负向事件 | 15,332 |
| 中性更新事件 | 21,521 |

图产物 manifest 保存源文件、排除用户集合、配置和三个事件文件的 SHA256。图构建和候选生成阶段不接受 ground-truth 路径。

## 特征和相似度

正向图和负向图分别使用：

\[
sim(i,j)=
\frac{cooc(i,j)}{\sqrt{pop(i)pop(j)}}
\times
\frac{n_{ij}}{n_{ij}+\beta}
\]

其中 `cooc` 是带时间权重的共同反馈，`pop` 抑制热门商家偏差，`n_ij` 是共同支持用户数。本步骤固定 `beta=10`，只在 validation 比较时间半衰期。

每个候选保存：

- `item_knn_positive_score`
- `item_knn_negative_evidence`
- 正/负支持用户数
- 正/负历史邻居数
- 缺失标记

第五条召回路线只按正向分数排序。负向证据保留给后续 Hybrid V2 学习，不参与本步骤的 RRF 扣分。

## Validation 半衰期选择

主要目标是五路融合 Recall@500；并列时依次比较 Recall@100、Recall@50、Item-KNN 单路 Recall 和延迟。

| 半衰期 | 融合 R@50 | 融合 R@100 | 融合 R@500 | Item-KNN R@500 |
|---|---:|---:|---:|---:|
| 不衰减 | 13.43% | 20.34% | 50.31% | 51.75% |
| 180 天 | 13.54% | 20.93% | 50.96% | 52.62% |
| **365 天** | **13.91%** | 21.25% | **51.19%** | **53.41%** |
| 730 天 | 13.78% | **21.36%** | 51.00% | 53.24% |

按主目标选择并冻结 365 天。选择结果保存在 `data/features/item_knn/tuning/selected_config.json`，测试任务没有参与选择。

## Full Retrieval V1 与 V2

主要人群仍是 4,645 个“目标商家在 cutoff 前存在，且目标不在用户既有历史”的 validation 任务。

| 指标 | 四路 V1 | 五路 V2 | 绝对提升 |
|---|---:|---:|---:|
| Recall@50 | 8.01% | **13.91%** | +5.90 pp |
| Recall@100 | 12.98% | **21.25%** | +8.27 pp |
| Recall@500 | 42.02% | **51.19%** | +9.17 pp |

全部 5,000 个任务上的 V2 Recall@50/100/500 为 12.92%、19.74% 和 47.56%。候选中未来/不存在商家为 0，用户历史商家为 0。

Item-KNN 单路 Recall@50/100/500 为 15.03%、22.93% 和 53.41%。单路 Recall 高于融合 Recall 不代表应该删除其他路线：RRF 保留了多信号共识，但也会让部分 Item-KNN 候选被其他路线挤出。后续 Hybrid V2 将学习更合适的非等权组合。

## 新增信息与排名伤害

| K | V2 新增命中 | V1 命中但 V2 丢失 | 净增加 | Item-KNN 找到且 Category/Text 均未找到 |
|---|---:|---:|---:|---:|
| 50 | 343 | 69 | 274 | 625 |
| 100 | 465 | 81 | 384 | 887 |
| 500 | 546 | 120 | 426 | 1,214 |

结果说明 Item-KNN 确实提供了类别和文本之外的新增信息，同时也存在可测量的 RRF 排名伤害。第 16 步 Hybrid V2 会使用独立正负分数、支持量和缺失标记学习组合，而不是继续假设五路等权最好。

## 历史长度分桶

| 历史长度 | 任务数 | V1 R@500 | V2 R@500 | 提升 |
|---|---:|---:|---:|---:|
| 8–15 | 2,420 | 43.39% | 52.56% | +9.17 pp |
| 16–30 | 1,221 | 42.34% | 51.52% | +9.17 pp |
| 31–60 | 651 | 38.10% | 48.69% | +10.60 pp |
| 61+ | 353 | 38.81% | 45.33% | +6.52 pp |

四个分桶均为正增益，满足“至少在部分用户分桶中提供 Category/Text 之外新增信息”的阶段门。

## 规模与性能

- rolling train：17,920 个任务，8,949,299 行候选。
- validation：5,000 个任务，2,498,631 行候选。
- validation 五路来源：12,465,228 行。
- V2 单任务平均召回延迟：59.77 ms；P95：130.52 ms。
- V1 平均/P95 为 46.26/65.80 ms，因此协同路线带来了可见但可控的离线开销。
- 四种半衰期 validation 调参约 7.7 分钟；train + validation V2 构建约 20.6 分钟，均为本地单进程观测值。

## 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\build_item_knn.py
.\.venv\Scripts\python.exe scripts\tune_item_knn.py
.\.venv\Scripts\python.exe scripts\build_full_retrieval_v2.py
.\.venv\Scripts\python.exe scripts\evaluate_full_retrieval.py `
  --candidates data\task_dataset\full_retrieval_v2_item_knn\validation_candidates.parquet `
  --task-audit data\task_dataset\full_retrieval_v2_item_knn\validation_task_audit.parquet `
  --route-provenance data\task_dataset\full_retrieval_v2_item_knn\validation_route_provenance.parquet `
  --metrics runs\item_knn_v1\validation_metrics.json `
  --task-results runs\item_knn_v1\validation_task_results.parquet `
  --benchmark-name "Full Retrieval Benchmark V2 + Item-KNN"
.\.venv\Scripts\python.exe scripts\compare_item_knn_retrieval.py
```

## 结论边界

这些结果来自 validation，用于开发和选择 Item-KNN 时间衰减，不能表述成完全未观察盲测结果。本步骤没有使用当前自然语言 Query、Review Aspect、用户画像、商家画像、LTR 或 Agent。它证明的是：时点安全协同行为能够显著扩大真实全库召回，而不是证明完整 Agent 已经完成。
