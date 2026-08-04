# 当前 20 候选推荐评测协议与审计结论

## 一句话结论

当前任务可以可靠衡量“给定 20 个候选后，模型能否把目标商家排到前面”，但不能衡量“模型能否从 Philadelphia 全量商家中召回目标”。

它的正式名称是：

> **Current 20-Candidate Controlled Reranking Benchmark**  
> 当前 20 候选受控重排评测

从数据构造角度，它属于 `ground-truth-conditioned closed-set reranking`：候选生成阶段先知道目标商家，再利用目标类别构造部分负样本。Ranker 和 Agent 看不到 target 标签，因此这不是模型输入泄漏；但候选集合的难度受到真值影响，所以结果不能解释成全量检索能力。

机器可读的完整结果位于 `current_20_candidate_audit.json`。

## 审计使用的数据

本次审计读取现有冻结任务和 provenance，没有重新生成候选、训练模型或调用 LLM：

- validation：4971 个保留任务。
- Legacy Test V0：4969 个保留任务。
- 合计：9940 个任务、198800 个候选位置。
- 原始时间切分共有 10000 个任务，其中 60 个任务因为 target 在 cutoff 前没有其他评论而被丢弃：validation 29 个，test 31 个。
- 有效商家集合：3973 家。

## 候选是如何构造的

每个任务包含 1 个 target 和 19 个负样本。配置要求依次尝试四个桶，并排除 target、用户历史商家、cutoff 前不可用商家和已被前面桶选中的商家。

| 来源桶 | 每任务计划数量 | 构造依据 | 全量实际数量 |
|---|---:|---|---:|
| target | 1 | 用户下一次真实交互 | 9940 |
| same_fine | 8 | 与 target 共享配置定义的“细粒度类别” | 79266 |
| related | 5 | 不共享 target 细粒度类别，但属于相同粗粒度组 | 46444 |
| preference | 3 | 命中用户历史类别偏好，并按时点偏好与质量排序 | 29820 |
| random | 3 | 从其余时点有效商家中确定性抽样 | 29820 |
| refill_same_fine | 缺口回填 | 原桶不足后再次从 same-fine 池补充 | 3256 |
| refill_related | 缺口回填 | same-fine 仍不足时从 related 池补充 | 254 |

初始 `same_fine` 比计划总数少 254 个，初始 `related` 比计划总数少 3256 个；这些缺口分别由 `refill_related` 和 `refill_same_fine` 补齐。因此最终每个保留任务仍然严格拥有 19 个负样本。

共有 703 个任务发生回填，占 7.07%。validation 为 333 个（6.70%），Legacy Test 为 370 个（7.45%）。

## “细粒度类别”的一个重要限制

当前配置只把以下四个标签排除在细粒度类别之外：

- `Food`
- `Nightlife`
- `Restaurants`
- `Shopping`

因此，`Bars`、`Beauty & Spas`、`Arts & Entertainment` 等相对宽泛的标签仍可能被当作细粒度类别。例如两个商家都含有 `Bars` 时，即使一个是 Cocktail Bars、另一个是 Dive Bars，也可能进入 `same_fine` 池。

这不违反当前代码协议，所以审计不会把它计为错误；但它会影响“困难负样本”的真实含义，后续构建新版候选协议时应扩展宽泛类别表或采用层级化 taxonomy。

## 合规检查结果

本次对 22 类不变量进行了全量检查，违反数量全部为 0，主要包括：

- task ID 在 split 内和 split 间均不重复。
- 每个任务恰好 20 个唯一候选。
- 每个任务恰好存在一条 ground truth，target 位于候选集合中。
- provenance 的商家、最终位置和任务候选顺序完全一致。
- target、same-fine、related、preference 等来源标签符合当前配置规则。
- 所有候选均属于有效商家集合，并且至少有一条严格早于 cutoff 的评论。
- 所有负样本均不在该任务的用户历史商家中。
- 所有历史行为时间严格早于任务 cutoff。
- Ranker 和 Agent 使用的公开任务文件不包含 `target_business_id`。

这里的“合规”表示任务符合当前已定义的协议，不表示当前协议已经等价于真实线上推荐。

## target 初始位置是否有明显偏置

候选在生成后使用任务级 SHA256 随机种子打乱。9940 个任务中，target 在 20 个位置上的理论期望约为每个位置 497 次；实际最少为第 11 位的 457 次，最多为第 4 位的 536 次。

没有发现 target 固定出现在靠前或靠后位置的结构性问题。模型不能仅凭原始候选位置猜测 target。

## 负样本重复频率

3973 家有效商家都至少在某个任务中出现过，候选商家集合覆盖率为 100%。但不同商家被选为负样本的频率差异较大：

- 负样本出现次数中位数：39。
- P95：84。
- P99：216。
- 单个商家最高出现次数：1904。

这意味着少数商家被大量任务重复使用。闭集指标可能部分受到这些高频负样本的质量、类别和热度特征影响。后续诊断 Hybrid 时，应按负样本频率和商家流行度分层报告结果，而不是只看一个总体 AvgHR。

## 当前指标可以和不可以说明什么

可以说明：

- 在已经给定 20 个候选的条件下，模型对目标商家的相对排序能力。
- Category、TF-IDF、Quality、Location、Hybrid 和 Agent 在相同候选难度下的可控比较。
- LLM 重排是否破坏或改善 Hybrid 的已有顺序。

不可以说明：

- 能否从 3973 家 Philadelphia 商家中找到用户下一次访问的商家。
- 真实检索阶段的 Recall@50、Recall@100 或 Recall@500。
- 面对不包含 target 的候选集合时，系统是否能主动扩展候选。
- 在线点击率、转化率或长期用户满意度。
- Agent 是否具备自主工具选择、澄清和成本控制能力。

因此当前 HR@K、MRR 和 NDCG@5 必须带上 `Controlled Reranking` 标签，不能直接写成“端到端推荐准确率”。

## validation 与 Legacy Test 使用规则

- validation 可以用于当前 V1 的历史复现；从下一阶段起，模型选择还必须结合 user-level cross-validation。
- Legacy Test V0 已经被查看，状态为 `previously_observed`。
- Legacy Test 只能用于历史对比，不得用于选择特征、模型、权重、阈值、Prompt 或 Agent 路由策略。
- 项目没有严格未见的 Final Blind Holdout，最终报告必须如实说明。

## 如何重新生成审计报告

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\audit_current_20_candidates.py
```

脚本只读取冻结任务、provenance、商家、评论和历史文件，输出 `docs/evaluation/current_20_candidate_audit.json`。相同输入会生成完全一致的 JSON 字节。

## 后续评测结构

当前 20 候选任务继续保留，作为受控重排基准。后续另行建设两个互不混淆的评测层：

1. **Full Retrieval Benchmark**：从 cutoff 时点全部有效商家中多路召回，报告 Recall@50/100/500。
2. **Agent Scenario Benchmark**：测试硬约束、上下文冲突、信息不足、澄清、工具路由和成本控制。

三个基准分别回答“排得好不好”“找不找得到”“Agent 是否做了有价值的决策”，不再用单一 HR@K 混合解释。
