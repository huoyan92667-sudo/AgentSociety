# Validation、用户交叉验证与 Legacy Test 使用规则

## 这份规则是干什么的

这份规则用来防止我们在开发模型时无意中“看着考试答案改模型”。

可以把数据想成两套题：

- **validation 是练习题**：可以用来比较模型、调整参数和发现问题。
- **Legacy Test V0 是已经看过答案的旧试卷**：只能在模型完全定下来以后做历史对比，不能再指导修改。

项目没有额外的 Final Blind Holdout，也不会假装 Legacy Test 仍然是一套从未见过的试卷。

## 两类数据分别能做什么

| 数据 | 可以做什么 | 不可以做什么 |
|---|---|---|
| validation | 选特征、选模型、调权重、调阈值、选 Prompt、设计 Agent 路由 | 冒充最终盲测成绩 |
| Legacy Test V0 | 在方案全部冻结后，与 V0 做一次历史对比 | 根据结果继续调模型、挑参数或换 Prompt |

以下开发决策只能使用 validation 和用户级交叉验证：

- Category、TF-IDF、协同过滤等特征是否保留。
- Logistic 与 LightGBM 选择。
- 正则化、时间衰减和相似度收缩参数。
- Hybrid 置信度阈值。
- Agent 在什么情况下调用哪种工具。
- LLM、Embedding 或 Cross-Encoder 是否值得调用。
- Rank-Delta 最多移动多少名。
- Prompt、工具预算和停止条件。

## 为什么要把用户分成 5 组

只在一批用户上测试一次，模型可能刚好运气好。我们把 validation 用户稳定地分成 5 组：

```text
第1次：用第2～5组开发，用第1组验证
第2次：用第1、3～5组开发，用第2组验证
……
第5次：用第1～4组开发，用第5组验证
```

最后报告 5 次结果的平均值和波动，而不是只挑最好的一次。

当前 4971 个 validation 用户的实际分组为：

| 组 | 用户数 | 当前任务数 |
|---:|---:|---:|
| 1 | 985 | 985 |
| 2 | 971 | 971 |
| 3 | 1042 | 1042 |
| 4 | 971 | 971 |
| 5 | 1002 | 1002 |

当前每位用户只有一个 validation 任务，所以用户数等于任务数。以后加入 rolling temporal tasks 后，一位用户会拥有多个训练任务，但这些任务仍然必须全部留在同一组。

## 用户是怎么稳定分组的

分组不是每次随机抽，而是根据用户 ID 和 seed 计算 SHA256：

```text
fold = SHA256("42:" + user_id) % 5 + 1
```

这样有三个好处：

1. 同一个用户永远进入同一组。
2. 输入文件换顺序不会改变分组。
3. 不同电脑上运行也会得到相同分组。

分组汇总不会保存用户 ID 或 task ID，只保存每组数量和全部分配结果的 SHA256。当前匿名汇总位于 `current_validation_user_fold_summary.json`。

## 什么是 Bootstrap 置信区间

假设 Hybrid V2 的 AvgHR 比 Hybrid V1 高 0.01，只看这个数字不能确定是真提升还是抽样运气。

Bootstrap 会把“用户”当作抽样单位，有放回地重复抽取用户 2000 次，每次重新计算两个模型的差值。最后观察大多数抽样结果落在哪个范围内。

- 如果绝大多数抽样都显示 V2 更好，提升比较可信。
- 如果结果一会儿正、一会儿负，说明提升不稳定。

必须抽“用户”，不能直接抽“任务”。因为未来同一用户会有多个时间任务，这些任务不是互相独立的人。

默认设置：

- 抽样次数：2000。
- 置信水平：95%。
- seed：42。
- 相同 replicate 编号会得到相同的用户抽样结果。

## 模型选择看哪些指标

在当前 20 候选受控重排任务中：

1. 首先比较 `AvgHR`。
2. AvgHR 相同时比较 `MRR`。
3. 仍然相同时比较 `HR@1`。

不能只展示总体成绩，还需要报告：

- 5 组平均值。
- 5 组标准差。
- 新旧模型的用户配对差值。
- 用户级 Bootstrap 95% 置信区间。

以后进入 Full Retrieval Benchmark 时，主指标会改为 Recall@50/100/500；不同评测层不能共用一个指标名称来解释全部能力。

## 为什么还要分人群看成绩

两个模型总体分数相同，不代表它们解决的是同一种问题。因此后续报告至少按以下情况拆分：

- 用户历史少、中等或丰富。
- target 类别以前是否出现过。
- 候选生成是否发生回填。
- Hybrid 第一名和第二名的分差大不大。
- 类别、文本、质量和位置特征是否互相矛盾。
- 商家位置是否缺失。
- 候选商家是否非常热门。
- 商家是否被频繁选作负样本。

这些分组只能用于解释模型，不允许在看到 Legacy Test 分组结果后继续调模型。

## 一个新版本应该按什么顺序开发

```text
只使用 validation
        ↓
完成 5 组用户交叉验证
        ↓
完成用户级 Bootstrap 和分层报告
        ↓
冻结代码、特征、参数和阈值
        ↓
可选：运行 Legacy Test 做历史对比
        ↓
无论结果好坏，都不能根据 Legacy Test 回头调当前版本
```

如果看过 Legacy Test 后决定继续修改，就必须产生一个新的版本，并明确记录“修改发生在查看 Legacy Test 之后”；它不能再被描述成盲测改进。

## 代码如何阻止误用

项目配置位于 `configs/evaluation_data_usage.yaml`，启动时会进行严格校验：

- 开发数据只能是 `validation`。
- Legacy Test 状态固定为 `previously_observed`。
- Legacy Test 用途固定为 `historical_comparison_only`。
- `strict_blind_holdout` 必须为 `false`。
- 分组数量必须至少为 2。
- Bootstrap 必须按用户抽样。

`assign_development_task_folds` 只接受 `validation:*` 任务；传入任何 `test:*` 任务都会抛出 `DataUsageViolation`。现有 Hybrid 权重调优器也已经独立检查并拒绝 test 任务。

## 如何重新生成匿名分组汇总

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_validation_user_folds.py
```

该命令只读取 validation 任务，不读取 ground truth，也不读取 Legacy Test。相同用户集合、seed 和分组数量会生成完全一致的汇总文件。

## 从哪个版本开始执行

V0 和当前 Hybrid V1 是这份规则建立前产生的历史基线，因此保留原始结果，但不把它们重新包装成用户级交叉验证结果。

从 Hybrid V1 诊断和 Hybrid V2 开发开始，所有模型选择都必须遵守本规则。下一步 Hybrid V1 诊断只读取 validation，不使用 Legacy Test 来决定后续特征和模型。
