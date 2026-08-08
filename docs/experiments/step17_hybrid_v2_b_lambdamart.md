# 第 17 步：Hybrid V2-B LightGBM LambdaMART 实验报告

## 1. 实验目标

本步骤只回答一个问题：

> 在完全相同的候选商家和特征语义下，非线性的 LightGBM LambdaMART 相对第 16 步 Pairwise Logistic 能提高多少？

本步骤不修改召回机制。Train、Validation 和 Legacy Test 继续使用 Full Retrieval V2 + Item-KNN 已经冻结的最终 Top-500 候选。因此三种排序器的 Recall@500 完全相同；本步骤的增益只来自正确商家已经进入候选以后，在 500 家内部获得了更好的名次。

本步骤不调用 LLM，也不读取自然语言 Query。

## 2. 公平比较边界

Hybrid V2-A Logistic 与 Hybrid V2-B LambdaMART 共享：

- 相同的 17,920 个 Rolling Train 任务；
- 相同的 9,514 个目标已进入召回候选的可训练任务；
- 每个训练任务相同的 1 个目标和 20 个冻结负样本；
- 相同的 Validation 与 Legacy Test 任务；
- 相同的 cutoff 时间边界；
- 相同的最终 Top-500 候选；
- 相同的指标定义和 `business_id` 确定性并列规则；
- 相同的 Hybrid V1 分数与 36 维特征池。

第一组公平对比固定使用第 16 步 Logistic 最终选择的 `without_business_profile` 31 维特征，只更换模型。之后 LambdaMART 才使用与 Logistic 相同的特征消融预算。

Legacy Test 在 Validation 模型冻结后才读取，不参与参数、特征集合或融合比例选择。该 Test 已经被项目观察过，因此只称为历史比较，不能称为全新盲测。

## 3. 模型训练方式

每个训练任务是一个 LightGBM ranking group：

```text
一个用户时点任务
├── 1 个目标商家，label = 1
└── 20 个冻结负样本，label = 0
```

目标函数使用 `lambdarank`。与 Logistic 学习一组全局线性系数不同，LambdaMART 使用多棵决策树学习非线性条件和特征交互，例如“Item-KNN 很强且召回名次靠前”“类别匹配强但距离过远”等组合关系。

有界参数搜索只包含四组配置：

| 参数名 | 叶子数 | 学习率 | 树数 | 叶子最少样本 | L2 | Validation AvgHR | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 15 | 0.05 | 150 | 50 | 1.0 | 0.017653 | 0.022885 |
| balanced | 31 | 0.05 | 250 | 50 | 1.0 | 0.017653 | 0.022427 |
| **slow** | **31** | **0.03** | **400** | **20** | **1.0** | **0.018371** | 0.023411 |
| wide | 63 | 0.05 | 250 | 50 | 1.0 | 0.017582 | **0.023553** |

主目标仍然是 AvgHR，并列时依次比较 MRR、HR@1，因此选择 `slow`，而不是只看 MRR 更高的 `wide`。

## 4. Validation 公平对比

政策内 Validation 任务为 4,645 个，其中目标进入 Top-500 的任务为 2,378 个。召回结果在模型间保持不变。

| 方法 | HR@1 | HR@3 | HR@5 | AvgHR | MRR |
|---|---:|---:|---:|---:|---:|
| Hybrid V1 | 0.002799 | 0.005597 | 0.009257 | 0.005884 | 0.010919 |
| Hybrid V2-A Logistic | 0.002153 | 0.007320 | 0.010549 | 0.006674 | 0.011498 |
| LambdaMART，相同 31 维特征 | 0.007750 | 0.018515 | **0.028848** | 0.018371 | 0.023411 |
| **LambdaMART，Validation 最终选择** | **0.009042** | **0.019591** | 0.028418 | **0.019017** | **0.024468** |

在完全相同 31 维特征下，LambdaMART 相对 Logistic：

- AvgHR 绝对提升 `0.011697`；
- AvgHR 相对提升 `175.27%`；
- AvgHR 达到 Logistic 的 `2.75` 倍。

经过同预算特征消融后，最终 LambdaMART 相对 Logistic：

- AvgHR 从 `0.006674` 提高到 `0.019017`；
- 绝对提升 `0.012343`；
- 相对提升 `184.95%`；
- AvgHR 达到 Logistic 的 `2.85` 倍；
- MRR 相对提升 `112.80%`；
- HR@1 从 10 个任务命中提高到 42 个；
- HR@3 从 34 个任务命中提高到 91 个；
- HR@5 从 49 个任务命中提高到 132 个。

## 5. 保守融合结果

LambdaMART 与 Hybrid V1 使用排名百分位融合：

```text
final = alpha × LambdaMART + (1-alpha) × Hybrid V1
```

| alpha | 最终选择特征下的 Validation AvgHR |
|---:|---:|
| 0.00 | 0.005884 |
| 0.25 | 0.008970 |
| 0.50 | 0.011769 |
| 0.75 | 0.015285 |
| **1.00** | **0.019017** |

Validation 选择 `alpha=1.0`。这表示当前 LambdaMART 已经能直接利用 Hybrid V1 分数及其组成特征，继续强制混合原始 Hybrid V1 反而降低效果。

## 6. 特征消融

所有消融使用同一个已选 LambdaMART 参数和相同 Validation 指标口径。

| 特征集合 | AvgHR | MRR |
|---|---:|---:|
| base_only | 0.014783 | 0.020095 |
| without_item_knn | 0.016649 | 0.022288 |
| without_user_profile | 0.017940 | 0.024182 |
| without_business_profile | 0.018371 | 0.023411 |
| **without_review_aspect** | **0.019017** | **0.024468** |
| full | 0.018658 | 0.024299 |

最终选择 `without_review_aspect`。这里的含义是：当前四个任务级用户—商家 Review Aspect 匹配特征没有在 Validation 上提供稳定增益，不能解释成评论 Aspect 永远无用。Business Profile 中的商家历史 Aspect 证据饱和度仍然保留，并不等于删除所有评论知识。

Item-KNN 被移除后 AvgHR 明显下降，说明协同行为信号仍然重要。用户画像被移除后整体也下降，但 MRR 接近完整模型，说明用户画像的贡献需要在后续 Query-aware 场景继续细分。

## 7. Validation 分桶结果

### 7.1 用户历史长度

| 历史长度 | 任务数 | Logistic AvgHR | LambdaMART AvgHR | 绝对提升 |
|---|---:|---:|---:|---:|
| 8–15 | 2,420 | 0.006749 | 0.018457 | +0.011708 |
| 16–30 | 1,221 | 0.009282 | 0.022932 | +0.013650 |
| 31–60 | 651 | 0.004096 | 0.020481 | +0.016385 |
| 61+ | 353 | 0.001889 | 0.006610 | +0.004721 |

四个分桶都提升。历史超过 61 条的用户仍然最难，可能与兴趣范围更广和长期兴趣漂移有关。

### 7.2 已见与未见细分类别

| 类别关系 | 任务数 | Logistic AvgHR | LambdaMART AvgHR | 绝对提升 |
|---|---:|---:|---:|---:|
| 已见细分类别 | 3,968 | 0.007476 | 0.021253 | +0.013777 |
| 未见细分类别 | 677 | 0.001969 | 0.005908 | +0.003939 |

未见类别同样提高，但绝对表现仍然较低，而且 LambdaMART 在该桶的 HR@1 仍然为 0。历史排序模型对“用户突然想尝试新类别”的能力有限，后续仍需要当前 Query 和探索机制。

## 8. Validation 配对 Bootstrap

使用 4,645 个政策内用户任务，对 Logistic 与最终 LambdaMART 的单任务 AvgHR 做 2,000 次配对用户 Bootstrap：

| 项目 | 结果 |
|---|---:|
| 观察到的 AvgHR 差值 | 0.012343 |
| 95% 下界 | 0.008683 |
| 95% 上界 | 0.016218 |
| 重采样差值为正的比例 | 100% |

该区间没有经过 0，说明在当前 Validation 用户上增益稳定为正。由于同一个 Validation 同时参与参数和特征选择，这个区间仍可能带有选择乐观偏差，不能代替真正的新用户盲测。

## 9. Legacy Test 历史比较

模型、参数、特征集合和 `alpha=1.0` 全部在读取 Test 前冻结。

| 方法 | HR@1 | HR@3 | HR@5 | AvgHR | MRR |
|---|---:|---:|---:|---:|---:|
| Hybrid V1 | 0.003481 | 0.005438 | 0.008484 | 0.005801 | 0.010680 |
| Hybrid V2-A Logistic | 0.002175 | 0.008266 | 0.012399 | 0.007614 | 0.011801 |
| **Hybrid V2-B LambdaMART** | **0.006961** | **0.016533** | **0.024581** | **0.016025** | **0.021719** |

LambdaMART 相对 Logistic：

- AvgHR 绝对提升 `0.008411`；
- AvgHR 相对提升 `110.48%`；
- AvgHR 达到 Logistic 的 `2.10` 倍；
- MRR 相对提升 `84.03%`；
- HR@1 从 10 个任务命中提高到 32 个；
- HR@3 从 38 个任务命中提高到 76 个；
- HR@5 从 57 个任务命中提高到 113 个。

Test 的提升小于 Validation，说明 Validation 选型确实存在一定乐观性；但全部主排序指标仍明显高于 Logistic，因此非线性模型的优势在这份历史 Test 上保留下来。

## 10. 特征重要性

按 LightGBM gain 排名前列的特征包括：

1. `retrieval_reciprocal_rank`
2. `item_knn_positive_score`
3. `hybrid_v1_score`
4. `quality_score`
5. `category_score`
6. `text_score`
7. `location_route_missing`
8. `business_aspect_evidence_saturation`
9. `retrieval_fusion_score`
10. `business_rating_count_log`

这说明模型大量使用召回名次、Item-KNN 和 Hybrid V1 基础信号，也会利用商家历史证据可靠性。Feature importance 只能说明模型分裂和增益中使用了哪些特征，不能当作“这个特征导致用户选择”的因果解释。

## 11. 工程产物

核心代码：

```text
src/yelp_agent/learning_to_rank/lambdamart.py
src/yelp_agent/learning_to_rank/lambdamart_selection.py
src/yelp_agent/learning_to_rank/lambdamart_experiment.py
src/yelp_agent/learning_to_rank/lambdamart_artifacts.py
src/yelp_agent/learning_to_rank/lambdamart_diagnostics.py
src/yelp_agent/learning_to_rank/lambdamart_legacy.py
src/yelp_agent/learning_to_rank/scored_ranking.py
```

本地实验产物：

```text
runs/hybrid_v2_b/frozen/model.joblib
runs/hybrid_v2_b/frozen/manifest.json
runs/hybrid_v2_b/validation_report.json
runs/hybrid_v2_b/validation_diagnostics.json
runs/hybrid_v2_b/legacy_test_report.json
runs/hybrid_v2_b/feature_importance.csv
data/features/hybrid_v2_b/validation_scores.parquet
data/features/hybrid_v2_b/validation_predictions.parquet
data/features/hybrid_v2_b/test_scores.parquet
data/features/hybrid_v2_b/test_predictions.parquet
```

冻结模型约 627 KB。真实 500 候选任务的本地单次运行时排序观察值约 8.6 ms，且与批量 Parquet 排名完全一致。该延迟只代表当前机器上的单次观察，不是严格性能基准。

## 12. 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\train_hybrid_v2_b.py
.\.venv\Scripts\python.exe scripts\evaluate_hybrid_v2_b_validation.py
.\.venv\Scripts\python.exe scripts\evaluate_hybrid_v2_b_legacy_test.py
```

## 13. 结论与边界

第 17 步得出的结论是：

> 在当前已经召回的 Top-500 候选内部，非线性的 LambdaMART 明显优于 Pairwise Logistic；相同 31 维特征下 Validation AvgHR 提升 175%，最终模型在 Legacy Test 上相对 Logistic 提升约 110%。

该结论不能扩展为“完整推荐系统已经解决”：

- Recall@500 没有变化，Legacy Test 仍有 2,348 个政策内任务没有召回目标；
- 当前 ground truth 是下一条评论商家，不完全等于用户会满意的推荐；
- Yelp 没有当前自然语言 Query；
- Test 是已观察的历史测试集，不是全新盲测；
- 训练每个任务只使用 20 个负样本，而推理面对最多 500 个候选，后续仍可研究更强的列表采样；
- Review Aspect 的当前匹配特征未提供增益，需要在 Query-aware 或评论 RAG 场景重新验证，而不是直接删除原始评论知识。

当前 Validation 冠军为 Hybrid V2-B LambdaMART，后续静态 Query-aware 排序和 Agent 工具可以通过统一运行时接口加载该冻结模型；召回瓶颈继续按照独立诊断文档保留，暂不在本步骤修改。
