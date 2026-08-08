# 第 16 步：Hybrid V2-A 成对 Logistic 学习排序

## 1. 这一步解决什么问题

前面的 Full Retrieval V2 负责从有效商家中召回最多 500 家候选。第 16
步不重新召回商家，而是学习如何把这些候选重新排序。

Hybrid V1 使用固定人工权重。Hybrid V2-A 使用训练数据学习一个线性排序器，
再与 Hybrid V1 做保守融合。此步骤不调用 LLM；后续 Agent 只把该模型当作稳定的
基础排序工具。

## 2. 数据边界

- rolling train：17,920 个时点任务。
- validation：5,000 个开发任务，只用于参数和模型版本选择。
- Legacy Test：5,000 个以前已经观察过的历史测试任务，只做冻结后的历史比较，
  不是新的 Final Blind Holdout。
- Full Retrieval 候选不包含 ground truth 字段，召回器也不接受 ground-truth 路径。
- 所有用户、商家和 Aspect 特征只使用严格早于 `cutoff_time` 的证据。
- Legacy Test 不参与训练、特征公式、阈值、`C`、融合比例或特征版本选择。

训练任务中只有 9,514 个目标商家出现在冻结的召回候选中。其余 8,406 个任务
被记录为召回失败，不把正确商家强行塞回候选。每个可训练任务包含 1 个目标和
20 个不重复负样本：

- 10 个融合排名靠前的困难负样本；
- 5 个 Category/Text 更接近目标的画像相似负样本；
- 5 个由 `SHA256(seed + task_id + business_id)` 决定的随机负样本。

最终形成 199,794 行训练候选，其中包含 190,280 组正负比较。成对训练再加入
反向差值后，Logistic 分类器看到 380,560 条对称训练记录。

## 3. 模型与特征

模型为 L2 正则化 Pairwise Logistic。对同一个任务的目标商家和负样本计算：

```text
pair_difference = target_features - negative_features
```

模型学习“目标应该排在负样本前面”。预测时直接计算每家候选的线性分数并按
分数降序排序。特征尺度只在 train 的成对差值上拟合。

完整特征表共有 36 项，分为五组：

1. 基础召回与 Hybrid V1：Category、Text、Quality、Location、RRF、route 覆盖、
   缺失标记和 Hybrid V1 分数。
2. Item-KNN：正向/负向协同证据、支持次数、邻居数量和缺失标记。
3. 用户画像：历史长度、平均评分、画像可靠度、类别正向匹配、类别负向冲突、
   类别新颖度和缺失标记。
4. 商家整体画像：评分证据量、Aspect 证据量、已知 Aspect 覆盖率和整体可靠度。
5. Review Aspect：用户正向偏好与商家正面证据匹配、用户负向偏好与商家负面
   证据冲突、匹配置信度和缺失标记。

Review Aspect 仍来自选定 5,000 名用户的评论原子，`source_scope` 为
`selected_user_interactions`，并不代表全部 617,718 条商家评论的 Aspect。

## 4. Validation 选模规则

模型选择顺序固定为：

1. Full 特征只比较 `C = 0.01, 0.1, 1.0, 10.0`。
2. 使用 `AvgHR = (HR@1 + HR@3 + HR@5) / 3` 选 `C`，依次用 MRR、HR@1
   破除并列。
3. 比较 Full、base-only，以及分别去掉 Item-KNN、用户画像、商家整体画像、
   Review Aspect 的版本。
4. 对 validation 最优特征版本搜索
   `alpha = 0, 0.25, 0.5, 0.75, 1.0`。
5. 最终分数采用两个排序的任务内百分位保守融合；`alpha=0` 为纯 Hybrid V1，
   `alpha=1` 为纯 Logistic。

最终选择：

```text
feature_set = without_business_profile
C = 1.0
alpha = 0.5
```

这里的 `without_business_profile` 只移除商家评分证据量、Aspect 覆盖率和整体
可靠度等 5 个整体特征，仍然保留商家 Review Aspect 匹配特征。

## 5. Validation 结果

主评测人群为 4,645 个“目标在时点目录中有效、且用户此前没有访问目标商家”的
任务。召回找到了其中 2,378 个目标；未召回目标按 0 分计入端到端指标。

| 方法 | HR@1 | HR@3 | HR@5 | AvgHR | MRR |
|---|---:|---:|---:|---:|---:|
| Hybrid V1 | 0.002799 | 0.005597 | 0.009257 | 0.005884 | 0.010919 |
| Hybrid V2-A | 0.002153 | 0.007320 | 0.010549 | 0.006674 | 0.011498 |

Hybrid V2-A 的 validation AvgHR 相对 Hybrid V1 提升约 13.4%，MRR 提升约
5.3%。但是 HR@1 下降，提升主要来自 HR@3 和 HR@5。

## 6. 消融结果

以下版本使用相同 `C=1.0` 和当时冻结的保守融合设置进行比较：

| 特征版本 | AvgHR | MRR | 结论 |
|---|---:|---:|---|
| without business profile | 0.006674 | 0.011498 | validation 最优，最终冻结 |
| without user profile | 0.006602 | 0.011378 | 用户画像当前增益不稳定 |
| full | 0.006387 | 0.011061 | 商家整体可靠度特征产生噪声 |
| without Item-KNN | 0.006172 | 0.010644 | Item-KNN 提供了有效增量 |
| without Review Aspect | 0.005382 | 0.010402 | Aspect 对 AvgHR 有明显贡献 |
| base only | 0.004449 | 0.008989 | 只有基础特征明显不足 |

线性系数只能用于诊断相关性，不能解释成因果关系。困难负样本采样和特征共线性
都会影响系数正负，因此项目报告以端到端指标和消融为主要证据。

## 7. Validation 稳健性审计

冻结 `without_business_profile`、`C=1.0` 和 `alpha=0.5` 后，按用户 SHA256
稳定分成 5 组。每次排除一个 fold 的 rolling train 任务重新拟合相同配置，并只在
对应 validation fold 评测。该审计不重新搜索参数，也不读取 Legacy Test 文件。

5 个 fold 的 AvgHR 差值分别为：

```text
-0.002905, +0.000368, -0.003090, +0.004400, +0.006723
```

- Hybrid V2-A fold AvgHR 均值：0.006991；标准差：0.002966。
- V2-A 相对 V1 的 fold 差值均值：+0.001100；标准差：0.003915。
- 用户级 2,000 次配对 Bootstrap 的整体 AvgHR 差值：+0.000789。
- Bootstrap 95% 区间：`[-0.001868, 0.003518]`。
- Bootstrap 差值为正的比例：69.85%。

置信区间跨过 0，且两个 fold 为负。因此当前证据只能说明“存在提升信号”，不能
声称 Hybrid V2-A 已经统计稳定地优于 Hybrid V1。这里采用的是固定配置 fold
refit，不是每个 fold 内重新选参的 nested cross-validation，这一点也保存在报告中。

## 8. Legacy Test 历史比较

模型和融合比例先在 validation 冻结，之后才构建 Legacy Test 候选和特征。
主评测人群为 4,597 个任务，召回命中其中 2,249 个目标。

| 方法 | HR@1 | HR@3 | HR@5 | AvgHR | MRR |
|---|---:|---:|---:|---:|---:|
| Hybrid V1 | 0.003481 | 0.005438 | 0.008484 | 0.005801 | 0.010680 |
| Hybrid V2-A | 0.002175 | 0.008266 | 0.012399 | 0.007614 | 0.011801 |

Hybrid V2-A 的 Legacy Test AvgHR 相对提升约 31.3%，MRR 相对提升约 10.5%。
HR@3 和 HR@5 提升明显，但 HR@1 下降约 37.5%。因此当前模型适合作为 Top-K
候选排序基础，还不能声称已经改善“唯一第一名”的精度。

## 9. 冻结产物和复现

核心代码位于 `src/yelp_agent/learning_to_rank/`：

- `sampling.py`：ground-truth 隔离的训练负样本选择；
- `features.py`：36 维时点安全特征；
- `model.py`：Pairwise Logistic；
- `selection.py`：validation 选参和消融；
- `evaluation.py`：端到端与已召回条件下评测；
- `artifacts.py`：模型、特征顺序和 SHA256 校验；
- `runtime.py`：未来 Agent 可直接调用的冻结单任务排序接口；
- `robustness.py`：固定配置的用户 fold refit 与配对 Bootstrap；
- `experiment.py`：validation-only 冻结流程；
- `legacy_test.py`：冻结后的历史 Test 比较。

运行命令：

```powershell
.venv\Scripts\python.exe scripts\train_hybrid_v2.py
.venv\Scripts\python.exe scripts\evaluate_hybrid_v2_robustness.py
.venv\Scripts\python.exe scripts\evaluate_hybrid_v2_legacy_test.py
```

本地生成但不提交 Git 的主要产物：

```text
data/features/hybrid_v2_a/training_selection.parquet
data/features/hybrid_v2_a/train_features.parquet
data/features/hybrid_v2_a/validation_features.parquet
data/features/hybrid_v2_a/validation_predictions.parquet
data/features/hybrid_v2_a/test_features.parquet
data/features/hybrid_v2_a/test_predictions.parquet
runs/hybrid_v2_a/frozen/model.joblib
runs/hybrid_v2_a/frozen/manifest.json
runs/hybrid_v2_a/validation_report.json
runs/hybrid_v2_a/validation_robustness_report.json
runs/hybrid_v2_a/legacy_test_report.json
```

冻结 manifest 明确保存模型版本、特征顺序、`C`、融合比例、特征版本、所有输入
SHA256、系数、validation 结果以及 `test_data_used_for_training=false` 和
`test_data_used_for_selection=false`。模型文件内容被修改后加载器会拒绝运行。

## 10. 当前限制

- Full Retrieval 的 Recall@500 约为一半；目标没有被召回时，排序模型无法补救。
- HR@1 下降，说明成对 Logistic 和当前负样本目标更偏向把目标推进 Top-5，而不是
  稳定选出唯一第一名。
- validation 的用户级 Bootstrap 置信区间跨过 0，5 个用户 fold 中有 2 个下降，
  当前提升还不具备统计稳定性。
- 用户画像和商家整体可靠度的增益还不稳定，需要分用户历史长度、Aspect 覆盖率和
  召回来源继续诊断。
- 商家 Aspect 仍是 5,000 名选定用户的评论样本，后续可冻结规则后扩展到完整商家
  评论，并重新做同一套消融。
- Legacy Test 已经被历史实验观察过，只能作为回归参考，不能包装成新的盲测成绩。
