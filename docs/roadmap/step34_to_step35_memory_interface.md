# 第34步到第35步的固定接口

第35步 Router 只读取 `RouterMemoryContext`，不读取全量 observations，也不能直接修改 `SessionMemory`。

必须遵守：

1. `hard_constraints` 是权威字段，LLM Router 不能覆盖。

2. `rejected_business_ids` 必须在召回和排序前排除。

3. `last_presented_business_ids` 只用于解析本轮“第一家、第二家”。

4. `relative_preferences` 是当前 Session 的排序信号，不写入 `UserProfileV1`。

5. `semantic_summary` 是辅助文本，不得覆盖结构化字段。

6. `business_scope_known=true` 时，所有商家工具调用必须限制在 `current_business_scope`。

7. 第35步只能选择动作；新的用户消息仍必须通过第34步 `SessionMemoryManager.update()` 更新记忆。
