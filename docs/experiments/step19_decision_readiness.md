# 第 19 步：判断任务类型、信息缺口与排名可靠性

## 1. 这一层解决什么问题

第 18 步把自然语言转换成了结构化的 `RecommendationRequest`。第 19 步不再重新推荐商家，而是给未来 Agent 准备一张“行动前检查表”：

```text
用户请求 + 当前排名信号
        ↓
DecisionReadinessAnalyzer
        ↓
任务类型 + 缺失信息 + 排名置信度 + 不确定原因
```

Agent Router 以后可以根据这张检查表决定：直接推荐、先追问、比较候选、查商家资料、查评论证据或安全回退。置信度只是 Router 的一个输入，不会单独决定是否调用 LLM。

统一输出类型是 `DecisionReadiness`。主要字段包括：

- `task_type`：用户现在要做什么；
- `information_gaps`：继续执行前还缺什么；
- `ranking_confidence`：当前 Hybrid V2-B 第一名命中下一家真实商家的估计概率；
- `uncertainty_reasons`：为什么这次排名可能不稳；
- `confidence_unavailable_reason`：为什么当前不能诚实地给出概率。

## 2. 任务类型怎么判断

当前是可审计的规则基线，支持：

| 类型 | 通俗含义 | 例子 |
|---|---|---|
| `recommendation_request` | 想找一家店 | “推荐一家安静的牛排馆” |
| `business_detail_question` | 问某家店的具体情况 | “第一家停车方便吗？” |
| `candidate_comparison` | 比较两家或多家店 | “第一家和第二家哪家适合约会？” |
| `feedback_refinement` | 对上一轮结果不满意，要求调整 | “太贵了，换一家” |
| `official_policy_question` | 查询当前营业或官方政策 | “这家现在允许带宠物吗？” |
| `review_experience_question` | 想从评论中了解真实体验 | “评论里有人说这家很吵吗？” |

规则采用固定优先级，避免一句话同时命中多类时结果漂移。无法确认时返回内部安全值 `unknown`，不强行猜测。第 20 步才会扩展覆盖六类任务的 Agent 场景 Benchmark；目前 500 条 Query 全是推荐请求，不能用它们证明另外五类的真实泛化能力。

## 3. 信息缺口怎么判断

输出五种缺口：

- `missing_location`：要求距离，但不知道用户在哪里；
- `missing_budget`：用户说“便宜一点”等模糊预算，无法执行确定的价格限制；
- `missing_party_size`：说要聚餐，却没说几个人；
- `constraint_conflict`：同一要求相互冲突，例如既必须包含 Bars 又必须排除 Bars；
- `ambiguous_reference`：说“第一家”“这家”，但上下文没有足够的商家 ID。

这里复用第 18 步解析器已经产出的条件和 `missing_fields`，不再维护另一套自然语言解析逻辑。冲突检测支持同一类别的包含/排除、偏好/避开，以及数值上下界互相矛盾。

## 4. 排名置信度怎么训练

### 4.1 预测目标

当前唯一诚实可训练的目标是：

```text
P(Hybrid V2-B 的 Top-1 等于用户下一家真实商家 | 排名与画像信号)
```

训练数据来自 5000 个 Validation 任务。每个任务只有一个二分类答案：第一名正确记为 1，否则记为 0。按用户稳定分成 5 折，使用 Out-of-Fold 预测比较校准器；Test 没有参与拟合、选择或阈值设定。

校准输入只使用答案揭晓前可见的 12 个信号，例如：

- 第一名总分以及第一、第二名的分差；
- LambdaMART 与 Hybrid V1 的排名分歧；
- 类别、文本、质量、距离、协同特征之间的分歧；
- Item-KNN 支持强度；
- 用户历史长度和画像可靠度；
- 第一名类别对用户是否陌生；
- 第一名被多少条召回路线支持。

`target_business_id` 不会写入校准特征文件，运行时的 `DecisionReadinessAnalyzer` 也不接收 Ground Truth。

### 4.2 校准器选择

比较了 Logistic Calibration 和 Isotonic Regression。选择规则是先看 Brier Score，越小越好；若相同再看 ECE。可靠性图使用等频分箱，使每个箱约有相同数量的任务，避免极低概率任务全部挤进一个 0–10% 大箱。

| 结果 | Logistic | Isotonic |
|---|---:|---:|
| 任务数 | 5000 | 5000 |
| Top-1 命中数 | 42 | 42 |
| Top-1 命中率 | 0.84% | 0.84% |
| Brier Score | **0.00825952** | 0.00833247 |
| ECE（10 个等频箱） | **0.00331868** | 0.00465667 |

最终冻结 Logistic。只输出固定总体命中率 0.84% 的朴素模型，其 Brier Score 是 0.00832944；Logistic 相对它只改善约 **0.84%**。这表示当前信号有一点区分能力，但远没有达到“置信度很可靠”的程度。

Coverage-Risk 也说明了同一问题：只保留模型认为最可靠的前 25% 任务，实际 Top-1 命中率为 1.44%；全部任务是 0.84%。排序有一定筛选作用，但绝对正确率仍很低，因此该概率只能作为 Router 的保守风险信号。

## 5. 排名为什么不确定

系统会同时给出可读的原因码：

| 原因码 | 含义 |
|---|---|
| `sparse_history` | 用户历史太少 |
| `unseen_category` | 第一名属于用户几乎没接触过的类别 |
| `feature_disagreement` | 类别、文本、质量、协同等信号意见不一致 |
| `small_top_margin` | 第一名和第二名差距太小 |
| `weak_collaborative_support` | Item-KNN 没有支持或支持很弱 |
| `low_profile_reliability` | 用户画像本身可靠度低 |

阈值只从 Validation 校准数据冻结，运行时不随单个请求临时改变。

## 6. Query-aware 置信度为什么暂时不给

现有 500 条 Query 的“标准答案”是结构化解析标签，例如应识别出什么类别、距离和预算；它们没有 `Query × 商家` 的相关性答案。因此当前不能验证“针对这句话推荐的第一家店是否正确”。

当 `ranking_source="query_aware"` 时，分析器明确返回：

```text
confidence_target = unavailable
confidence_unavailable_reason = query_aware_labels_unavailable
```

这不是程序没做完，而是防止把“下一家商家预测概率”冒充成“当前 Query 的推荐正确率”。未来取得 Query-aware 商家相关性标签后，可以在相同接口下增加第二个校准器。

## 7. 代码与产物

核心接口：

- `src/yelp_agent/decision_readiness/engine.py`：统一分析入口；
- `src/yelp_agent/decision_readiness/request_analysis.py`：任务类型和缺口；
- `src/yelp_agent/decision_readiness/dataset.py`：构造无目标泄漏的任务级校准数据；
- `src/yelp_agent/decision_readiness/calibration.py`：交叉验证、校准指标和不确定原因；
- `src/yelp_agent/decision_readiness/artifacts.py`：冻结模型、哈希校验和加载；
- `src/yelp_agent/decision_readiness/experiment.py`：完整实验流程；
- `configs/decision_readiness.yaml`：固定配置；
- `scripts/train_decision_readiness.py`：薄 CLI。

本地生成但不提交 Git 的可重复产物：

```text
data/features/decision_readiness/v1/calibration_examples.parquet
runs/decision_readiness_v1/frozen/model.joblib
runs/decision_readiness_v1/frozen/manifest.json
runs/decision_readiness_v1/evaluation_report.json
runs/decision_readiness_v1/reliability_diagram.svg
runs/decision_readiness_v1/coverage_risk.csv
```

复现命令：

```powershell
.\.venv\Scripts\python.exe scripts\train_decision_readiness.py
```

重复运行会校验输入、配置和模型哈希，避免把不同实验的文件混在一起。

## 8. 本步边界

- 本步没有调用 LLM；任务分类先提供透明、稳定的规则基线。
- 本步没有实现 Agent Router；它只准备 Router 需要的状态，Router 属于后续步骤。
- 当前置信度仅适用于 Hybrid V2-B 下一商家 Top-1，不适用于 Query-aware 排名。
- 42 个正例偏少，不能把校准器包装成高可靠模型；后续应在召回和排序改善、正例增多后重新冻结。
- 六种任务类型目前通过单元测试验证逻辑，覆盖真实多轮场景的 Benchmark 在第 20 步完成。
