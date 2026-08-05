# Full Retrieval Benchmark V1：真实多路候选召回

## 这一步解决什么问题

旧实验是 **Controlled Reranking Benchmark**：正确商家已经被放进固定的 20 个候选中，模型只负责重新排序。它仍然保留，用于和 V0 历史结果比较。

本实验是新增的 **Full Retrieval Benchmark V1**：系统只获得用户、截止时间和截止时间前的历史，从 Philadelphia 当时已经存在的全部合格商家中自行寻找候选。候选生成模块的接口不接受 ground truth 路径，也没有 `target_business_id` 字段。

```text
时点可用的 Philadelphia 商家
→ 质量 / 类别 / TF-IDF / 位置四路召回
→ 等权 Reciprocal Rank Fusion
→ 冻结 Top-500
→ 独立评测器读取正确答案
```

这一步不调用 LLM，不产生 token 消耗，也不删除或覆盖旧的 20 候选数据。

## 商家目录和排除规则

对每一个任务独立建立时点目录：

1. 商家必须至少有一条严格早于 `cutoff_time` 的评论，才能视为当时已经存在。
2. 评分、评论数量和质量只使用严格早于 `cutoff_time` 的评论重新计算。
3. 默认排除用户在 `cutoff_time` 前已经访问过的商家。
4. 不会为了凑满 500 家而加入未来商家，也不会强制插入正确答案。因此早期训练任务的候选数可以少于 500。

真实 validation 审计确认：候选中“截止时间前没有评论”的记录为 0，候选中“用户历史已访问”的记录为 0。

## 四条召回路线

| 路线 | 只使用的信息 | 排序含义 |
|---|---|---|
| `quality` | 截止时间前的评分和评论数 | 贝叶斯收缩质量分从高到低 |
| `category` | 用户历史评分和商家静态类别 | 与用户历史类别偏好从匹配到不匹配 |
| `text` | 训练安全 TF-IDF、历史正负评论和商家静态文本 | 正面相似、负面不相似的商家优先 |
| `location` | 用户历史商家与候选商家的坐标 | 距离用户历史活动中心从近到远 |

每条路线最多提供 500 家。融合使用固定的等权 RRF：

```text
fusion_score(business) = Σ 1 / (60 + route_rank)
```

商家在某条路线越靠前，贡献越大；同时被多条路线找到，会累加多份贡献。融合分相同时使用 `business_id` 升序，保证确定性。第一版没有根据 validation 标签调路线权重。

## 评测人群为什么有三种计数

validation 原始任务共 5,000 个：

- 4,971 个目标商家在截止时间前已经进入商家目录；
- 29 个目标商家当时还没有其他历史评论，任何召回器都无法知道它存在；
- 326 个任务的目标商家已经出现在该用户历史中，但本 benchmark 的策略会排除已访问商家；
- 因此主要 Recall 的可达评测人群为 `4,971 - 326 = 4,645` 个任务。

报告同时保存两套指标：

- `recall_at`：只在 4,645 个“目录可用且未访问过”的任务上统计，是主要召回指标；
- `all_task_recall_at`：在全部 5,000 个任务上统计，把目录不可用和已访问目标也视为失败，反映端到端上限。

## 真实 validation 结果

### 四路融合

| 指标 | 主要可达人群 | 全部 5,000 任务 |
|---|---:|---:|
| Recall@50 | 8.01% | 7.44% |
| Recall@100 | 12.98% | 12.06% |
| Recall@500 | 42.02% | 39.04% |

主要人群中，1,952 个任务在融合 Top-500 找到了正确商家，2,693 个任务没有找到。

### 单路 Recall（主要可达人群）

| 路线 | Recall@50 | Recall@100 | Recall@500 |
|---|---:|---:|---:|
| 质量 | 6.14% | 11.24% | 36.84% |
| 类别 | 4.24% | 8.12% | 29.92% |
| TF-IDF 文本 | 4.69% | 8.01% | 28.59% |
| 位置 | 4.61% | 9.11% | 32.79% |
| 四路融合 | **8.01%** | **12.98%** | **42.02%** |

融合在三个 K 值上都高于任意单路，说明不同路线确实找回了互补商家。不过 Recall@500 仍只有 42.02%，召回仍是当前系统的主要瓶颈；第 12 步将加入时点安全 Item-KNN，测量协同过滤带来的增量。

### 规模和运行时间

- rolling train：17,920 个任务，8,949,299 个融合候选行；
- validation：5,000 个任务，2,498,631 个融合候选行；
- validation route provenance：9,993,111 行；
- validation 平均候选数：499.73；
- validation 平均时点商家目录：3,279.89；
- 单任务平均召回延迟：46.26 ms；
- 单任务 P95 召回延迟：65.80 ms。

延迟是当前离线、单进程 Python 实现的观测值，不代表部署后的线上延迟。

## 产物与复现

生成候选：

```powershell
.\.venv\Scripts\python.exe scripts\build_full_retrieval_benchmark.py
```

评测 validation：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_full_retrieval.py
```

本地大文件保存在：

```text
data/task_dataset/full_retrieval_v1/
runs/full_retrieval_v1/
```

核心产物包括候选、任务级审计、validation 四路来源和 manifest。标签只出现在 `runs/full_retrieval_v1/validation_task_results.parquet`，不会写回公开候选文件。manifest SHA256 为：

```text
ce136e27b0c02f58565f7dd24d1aad4036916023bb1440cefa109ecbd3ed0a15
```

## 结果边界

这些结果来自 validation，用于后续开发诊断。根据已经确定的项目政策，当前没有额外 Final Blind Holdout，因此它不能被表述为“完全未观察测试集上的最终泛化结果”。Legacy Test V0 仍只用于历史对照，不能用于调整召回路线。
