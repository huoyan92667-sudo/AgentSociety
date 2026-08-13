# Context-grounded Agent Evaluator V3

## 为什么需要 V3

Session Memory Benchmark V2 的问题文本由一份冻结的 Agent 展示结果生成。例如，问题中的“第一家”原本指冻结列表的第一名。

真实端到端实验会重新运行召回与排序，因此被评 Agent 实际展示的第一名可能已经变化。V2 仍使用冻结商家 ID 判分，会同时产生：

- 假阴性：Agent 按实际第一家正确处理，却因不满足旧第一家的条件而被判错；
- 假阳性：Agent 不满足实际第一家的相对要求，却碰巧满足旧第一家的条件而被判对。

V3 保留冻结的用户意图，但把序号指代重新绑定到被评 Agent 最近一次真正展示的列表。例如：

```text
冻结第一家：Friday Saturday Sunday，价格等级 4
实际第一家：Southgate，价格等级 2
用户要求：比第一家便宜

V2：价格等级 1、2、3 都算满足
V3：只有价格等级 1 才算满足
```

如果当前轮根本没有返回候选或推荐，V3 不会把该题移出分母，而是使用会话里最近一次真实候选范围计算可行答案，并记录为 `no_recommendation`。

## 当前正确答案是怎样生成的

V3 的正确答案不是唯一商家，而是当前候选范围中满足“本轮新增、且能够从 Yelp 字段客观验证的条件”的商家集合：

- “比第一家便宜”：`price_level < 实际第一家的 price_level`；
- “比第一家近”：`distance_km < 实际第一家的 distance_km`；
- “比第一家安静”：Yelp `NoiseLevel` 的有序等级更低；
- “价格等级不超过 2”：`price_level <= 2`；
- “距离不超过 5 km”：`distance_km <= 5`；
- “不要第 N 家”：最终结果不能包含实际展示列表中的第 N 家。

同一轮新增多个条件时，`Joint-delta Compliance` 使用这些答案集合的交集。

## 必须保留的限制

V3 当前的答案标签范围是 `current_turn_delta_on_actual_agent_context`，不是完整会话推荐正确率。它尚未把初始请求和所有历史轮次累积条件全部求交。例如：

```text
初始请求：今天只想吃 Burgers
后续请求：比第一家便宜
```

当前 V3 可以可靠判断“是否更便宜”，但价格更低且不是 Burgers 的商家仍可能进入原子答案集合。因此报告会固定写出：

```text
full_session_constraint_compliance_measured = false
```

在完成 Session 全量条件合并评测之前，不能把 V3 的 Compliance 宣称为“最终推荐准确率”。

## Case Explorer 是什么

Case Explorer 把每一轮评测所需的信息放在同一条记录里：

```text
用户画像
→ 完整会话文本
→ 上一轮实际展示商家
→ Agent 选择的动作与工具
→ 当前候选 Top-20
→ 最终 Top-5
→ 动态正确答案集合与示例
→ 判错类别
```

输出文件：

- `case_explorer_v3.md`：错误索引，适合快速浏览；
- `case_explorer_v3.jsonl`：全部已释放轮次的完整机器可读记录；
- `case_explorer_v3_failures.jsonl`：只保留未命中 Top-1 的记录；
- `behavior_cases_v3.jsonl`：逐轮评分明细；
- `metrics_v3.json`：汇总指标。

常见判错类别：

- `no_recommendation`：该轮存在可行答案，但 Agent 没返回推荐；
- `acceptable_in_top5_not_top1`：Top-5 中有满足项，但第一名不满足；
- `acceptable_below_top5`：满足项已在候选池中，但排名低于第 5；
- `acceptable_absent_from_ranking`：当前候选范围中没有可验证的满足项；
- `not_scored`：Yelp 数据无法客观验证，例如精确“人均 35 美元”。

## Rule Agent 离线重评分结果

本次重评分没有重新运行模型或调用 API：

- 实际上下文与冻结上下文一致：52 轮；
- 需要重绑定：133 轮；
- 不涉及序号指代：230 轮；
- Joint-delta Compliance@1 / @5：20.31% / 29.89%；
- 有可行答案但未返回推荐：79 轮；
- 正确答案集合数量中位数：112。

答案集合偏大正是“尚未合并完整 Session 条件”的直接证据，不能误读为 Yelp 中真的有 112 家同样合适的餐厅。
