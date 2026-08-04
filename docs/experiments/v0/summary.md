# V0 实验冻结说明

## 冻结目的

本目录冻结的是项目第一版 Hybrid+LLM 重排实验。它是后续 Hybrid V2、工具路由 Agent 和成本优化的历史基线，不再作为开发阶段的调参依据。

本次冻结只整理已有产物，没有重新运行模型、调用 API、修改 Yelp 数据或更改既有实验结果。

## V0 系统做了什么

V0 先使用 Hybrid 对固定的 20 个候选商家排序，再固定取前 8 名交给 DeepSeek。LLM 必须返回这 8 个商家的完整唯一排列，原 Hybrid 第 9 至 20 名保持原顺序。每个任务固定执行 4 次本地工具读取，LLM 不负责决定是否调用工具、调用哪项工具或何时停止。

因此，V0 更准确的名称是“带工具上下文的固定 LLM 重排器”，还不是能够自主观察、选择动作和控制成本的 Agent。

## 20 任务真实 API 结果

| 指标 | Hybrid | Hybrid+Agent | Agent - Hybrid |
|---|---:|---:|---:|
| HR@1 | 0.150000 | 0.000000 | -0.150000 |
| HR@3 | 0.400000 | 0.200000 | -0.200000 |
| HR@5 | 0.500000 | 0.500000 | 0.000000 |
| AvgHR | 0.350000 | 0.233333 | -0.116667 |
| MRR | 0.328836 | 0.210562 | -0.118274 |
| NDCG@5 | 0.337707 | 0.246651 | -0.091056 |

目标商家名次变化：2 个任务改善，11 个不变，7 个变差。LLM 的 20 次请求全部成功，20 次均改变了 Top-8，没有发生回退，但总体排序效果低于 Hybrid。

## 成本和运行表现

- Agent 平均延迟：100,895.94 ms；P95 延迟：158,292.35 ms。
- LLM 平均延迟：100,882.38 ms；P95 延迟：158,282.18 ms。
- 总输入 Token：110,951。
- 总输出 Token：241,898。
- 总 Token：352,849，平均每任务 17,642.45。
- 平均工具调用：每任务 4 次。
- LLM Failure Rate：0；Fallback Rate：0。

这些结果说明：V0 的主要问题不是 API 稳定性，而是调用成本高、延迟长，并且强制重排会破坏原本较好的 Hybrid 排名。

## 评测边界

当前 20 候选集合包含一个目标商家和基于目标类别等信息构造的 19 个负样本。这是 **ground-truth-conditioned closed-set reranking（基于真值构造候选的封闭集重排）**，不能等同于从 Philadelphia 全量商家中召回目标的真实线上推荐。

此外，当前 test 结果已经被人工查看，正式标记为 `Legacy Test V0`：

- `legacy_test_status: previously_observed`
- `strict_final_blind_holdout: false`
- 后续模型选择只使用 validation 和 user-level cross-validation。
- Legacy Test 只做历史对比，不用于选择特征、权重、阈值或路由策略。

这意味着项目不会宣称拥有严格未见的最终盲测集，后续报告必须分别展示 validation、用户级交叉验证和 Legacy Test 结果。

## 可复现性与已知缺口

V0 运行时没有把 Git commit 和原始命令行写入产物，因此无法证明实验执行时的精确源码提交。冻结审计在提交 `95a807408569c0418ab6d48733f7474454fd68e7` 上完成；该提交是运行后的工程清理版本，不能冒充运行时 revision。

根据当前 CLI 默认值、20 个预测与 test 文件前 20 个任务完全一致这一事实，以及实验输出目录，可以重建出最可能的运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\run_agent.py --limit 20 --output-dir runs\agent\pilot-deepseek-20 --force
```

但原始 `argv` 没有随实验保存，所以该命令在 `manifest.json` 中标记为 `reconstructed`，而不是 `recorded_at_run_time`。

冻结文件不会保存 API Key、认证头、完整评论、ground truth 或任务 ID 明文。任务集合只保存排序后 ID 列表的 SHA256。

## 下一阶段为什么要改变

后续路线不应继续扩大 Prompt 或让 LLM 无条件重排，而应先诊断并升级 Hybrid，再让 Agent 根据不确定性选择是否查询协同信息、语义模型、约束工具或澄清用户。最终排名还需要采用保守 Rank-Delta：证据不足不改变 Hybrid，高置信 Top-1 可锁定，并限制候选最大移动距离。
