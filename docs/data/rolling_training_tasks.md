# Rolling Temporal Training 数据说明

## 这批数据解决什么问题

原有数据只为每个用户保留倒数第二条 validation 和最后一条 Legacy Test。它适合离线评测，但没有充分利用更早的交互来训练 Learning-to-Rank 模型。

Rolling temporal training 会在用户更早的时间线上构造多个“历史 → 下一次选择”样本，同时继续封存最后两条行为：

```text
较早历史 → rolling train targets → validation → Legacy Test
```

本步骤只构造时间样本，不构造候选商家，也没有读取或修改旧的 target-conditioned 候选文件。真正的多路候选召回属于第 11 步。

## 当前固定规则

规则来自 `configs/training.yaml`：

- 每个训练任务至少包含 8 条历史。
- 相邻训练目标至少间隔 2 次交互。
- 每个用户最多保留 6 个训练任务。
- 超过上限时，在用户早期、中期和近期之间确定性地等距选择，并保留最新的安全训练时间点。
- 同一用户的全部训练任务使用同一个确定性五折编号。
- 样本权重按用户生命周期位置从 0.5 线性增加到 1.0。
- 倒数第二条 validation 和最后一条 Legacy Test 永远不能成为训练目标或训练历史。
- 每个任务的历史最大时间必须严格小于 `cutoff_time`。

每个用户最新的安全 rolling task 还会进入 `minimal_train_task_ids.jsonl`，作为每用户最多一个任务的快速 smoke-test 子集。它不是另一份训练标签，也不会重复存储历史。

## 文件及标签隔离

```text
data/task_dataset/
├── training/
│   ├── rolling_train_contexts.parquet
│   ├── rolling_train_histories.parquet
│   ├── minimal_train_task_ids.jsonl
│   └── rolling_train_manifest.json
└── ground_truth/
    └── rolling_train_ground_truth.parquet
```

`rolling_train_contexts.parquet` 保存用户、cutoff、历史长度、序列位置、fold 和样本权重。`rolling_train_histories.parquet` 只保存每个任务可见的历史 review ID。两个公开文件都不包含目标商家的 ID 或目标 review ID。

目标 review ID 和目标商家 ID 只保存在隔离的 `rolling_train_ground_truth.parquet`，供后续候选构造和模型训练模块显式加载，不能交给 Ranker、Agent 或 LLM 上下文。

## 五折使用方式

例如评测 fold 3 时：

1. 使用 fold 1、2、4、5 用户的 rolling tasks 训练模型。
2. 使用 fold 3 用户的 validation tasks 评测。
3. fold 3 用户的 rolling target 标签不能进入这次模型训练。
4. 推理时仍可以使用 fold 3 用户在 cutoff 前的普通历史来构造画像，这是推荐场景允许的输入。

## 真实 Yelp 数据构造结果

2026-08-05 使用当前冻结的 5,000 名用户完成全量构造：

| 项目 | 数量 |
|---|---:|
| 输入用户 | 5,000 |
| 产生训练任务的用户 | 4,435 |
| 历史不足、没有训练任务的用户 | 565 |
| Rolling train tasks | 17,920 |
| History rows | 386,059 |
| Minimal smoke-test tasks | 4,435 |
| 跳过的 cutoff 同时刻目标 | 0 |
| 历史长度中位数 | 14 |
| 历史长度 P95 | 62 |
| 最大历史长度 | 773 |

每用户任务数分布：

| 任务数 | 用户数 |
|---:|---:|
| 0 | 565 |
| 1 | 848 |
| 2 | 541 |
| 3 | 433 |
| 4 | 339 |
| 5 | 309 |
| 6 | 1,965 |

五折任务数为 3,522、3,438、3,658、3,602 和 3,700。每位用户只属于其中一折。

Manifest SHA-256：

```text
4ece6873663d1352211fb3d80ff431ad59821984ddb30a39259580ea11e3ac0c
```

全量审计结果均为 0：同任务 target-history 泄漏、validation/test 保留行为泄漏、cutoff 违规、用户跨 fold、每用户任务数超限。

## 与旧 benchmark 的关系

本步骤没有删除或覆盖：

- 现有 validation/test 时间切分；
- 现有 validation/test 20 候选任务；
- 现有 ground truth；
- 旧候选构造代码；
- Hybrid V1 和 Agent V0 的历史结果。

因此这批 rolling 数据是为 Hybrid V2 新增的训练原料，不会改变 V0 benchmark。
