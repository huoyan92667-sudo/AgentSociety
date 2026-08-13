# 完整 Session 最终请求：EffectiveSessionRequest

## 目标

这一阶段解决“Agent 已经记住用户后续要求，但不同工具读取的需求不一致”的问题。

此前普通条件保存在 `memory.current_request`，相对偏好、拒绝商家和追问答案则分别保存在其他 Memory 字段。Query 召回、Embedding、Cross-Encoder 与硬约束工具主要读取旧的 `state.request`，因此可能看不到“更便宜”“离第一家更近”等 Session 更新。

现在统一使用一个编译接口：

```python
compile_effective_request(memory) -> EffectiveSessionRequest
```

调用方不需要自己拼接历史对话或遍历 Memory 字段。

## 最终请求包含什么

`EffectiveSessionRequest` 包含：

- 当前仍然生效的全部结构化条件；
- 当前任务类型；
- 人数和用户位置；
- 已解析的相对偏好；
- 相对偏好对应的参考商家 ID；
- 当前 Session 已拒绝的商家；
- 已接受的追问答案；
- 最新一轮用户原文；
- 根据有效状态重新生成的结构化语义文档；
- 基于有效内容计算的确定性 `effective_request_id`。

## 为什么不直接拼接全部聊天记录

把所有历史原文继续拼接会保留已经失效的条件。例如：

```text
第一轮：我想吃中餐
第二轮：今天改成汉堡，不要中餐
```

原文拼接后，Embedding 仍会同时看到“中餐”和“汉堡”。新编译器只根据当前结构化状态生成文档；如果中餐条件已经被替换，它不会再次出现在最终请求里。

生成文档示例：

```text
Active recommendation request reconstructed from accepted session state.
Task: feedback_refinement.
Requirement: category includes Burgers (mandatory; filter).
Requirement: price_level less_than_or_equal 2 (strong; rank).
Party size: 2.
Relative preference: Prefer a lower price than the referenced result.
Exclude 1 user-rejected businesses by their validated IDs.
```

## 合并规则

### patch

在当前有效请求上新增、替换或删除指定字段。未涉及的条件继续保留。

### replace

用户明确开始一个新需求。普通条件以新请求为准，同时清空旧的 Session 临时状态：

- 相对偏好；
- 相对偏好参考商家；
- 被拒绝商家；
- 旧追问答案。

长期画像候选不会在本阶段自动删除。

### no_change

用户只是询问事实，不修改推荐目标。当前轮规则解析结果不会偷偷进入已有推荐条件。

## 接入位置

以下下游现在优先读取同一份 `EffectiveSessionRequest`：

- Query 双通道召回；
- 硬约束过滤；
- Embedding 语义匹配；
- Cross-Encoder 精排；
- Step 30 结构化语义排序；
- `GET_SESSION_MEMORY`；
- Agent 工具调用上下文；
- Agent Turn Trace；
- 新生成的 Case Explorer。

没有 Session Memory 的旧运行仍回退到原 `RecommendationRequest`，保持兼容。

## 安全与确定性

- 最终请求只来自代码验证后保存的 Memory，不直接信任 LLM JSON；
- 相对偏好的商家 ID 只能来自代码解析后的可见引用；
- 被替换或删除的条件不会进入语义文档；
- 相同有效状态生成相同 `effective_request_id`；
- 修改无关的旧 `semantic_summary` 不会改变最终请求；
- Ground Truth 不进入编译器或工具上下文。

## 本阶段仍未解决什么

本阶段完成的是“把完整需求放进同一个可执行对象”，不是最终排序优化。

尚未完成：

- 根据参考商家的真实价格、距离和噪声，把相对偏好转成候选级分数；
- 将 `price=lower`、`distance=closer` 等信号作为强排序特征；
- 对完整 Session 条件计算真正的 Full-query Compliance；
- 修复 Rule Router 把“重新推荐”误判成详情问题；
- 重新运行正式 500 场景端到端实验。

因此，本阶段为下一步“让完整请求真正改变召回、过滤和排序”提供唯一输入接口，但不提前宣称推荐效果已经提升。
