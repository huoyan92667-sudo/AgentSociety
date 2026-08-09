# Agentic Recommendation 路线 V2（第 11 步后）

## 状态

- 第 1–19 步已经完成，历史实现、实验产物和 Git 提交不倒改。
- 当前下一开发步骤是第 20 步：扩展 Agent 场景 Benchmark。
- 后续开发一次只执行一个编号步骤。
- validation 用于开发和选择配置；冻结前不使用 test 调参。
- 未经用户明确确认，不调用真实 LLM。

## 路线更新

第 12 步以后按以下主线推进：

1. 时点安全 Item-KNN，补充协同行为召回和特征。
2. Review Aspect、只读用户画像 V1、商家画像 V1。
3. Pairwise Logistic 与 LambdaMART 两版 Hybrid V2。
4. `RecommendationRequest`、Query-aware 静态推荐和置信度。
5. Agent 场景评测、固定评测契约和通用受控 Agent Harness。
6. Rule Router、结构化工具注册表和安全回退。
7. Embedding、Cross-Encoder、Business-scoped Review RAG 和证据聚合。
8. 受控 LLM 语义工具、保守融合与 Rank Protection。
9. 受约束 LLM Router、Cost-aware Learned Router 和 Session Memory。
10. 冻结后的完整消融、最终评测和项目展示。

## 固定架构原则

- 推荐主干保持确定性；LLM 不生成全库候选，也不直接接管全量排序。
- Agent 使用通用 `AgentState + Tool Registry + Router + Validator` 循环，不为每种用户问题手写独立工作流。
- 用户画像是只读长期记忆 V1；商家画像属于共享 Business Knowledge。
- Review 同时支持离线画像/排序特征和在线 Business-scoped RAG，两条能力链相互独立。
- 硬约束不能被语义分数覆盖；任何工具失败都能安全回退 Hybrid V2。
- 对用户展示结构化决策轨迹、证据、日期和不确定性，不展示模型内部 Chain of Thought。

## 第 12 步特别约束

- 4–5 星是正反馈，3 星是中性，1–2 星是负反馈证据。
- 正反馈图和负反馈证据分开建模，输出独立特征，不提前手工相减。
- 每个任务只能使用严格早于 `cutoff_time` 的行为。
- validation/test target 不进入协同图；如果存在 Final Holdout 用户，其全部行为也不进入图。
- validation 只比较无衰减、180、365、730 天四种设置。
- Item-KNN 既要作为独立基线，也要作为 Full Retrieval 的新增召回来源。
- cutoff 后追加数据不能改变旧任务结果。

## 评测边界

- Controlled Reranking 与 Full Retrieval 必须分开报告。
- Item-KNN 不要求对所有用户都有提升，但必须报告它相对 Category/Text 提供的新增召回信息。
- 当前仓库没有单独的 Final Blind Holdout 数据产物；实现保留排除用户接口，最终评测政策在第 35 步前统一。

## 第 18–19 步进展与边界

- 已建立请求 Schema、硬约束/软偏好/查证/澄清政策和规则基线。
- 已为本地模型、Embedding 或 OpenAI-compatible 语义解析保留 `RequestSignalExtractor` seam。
- 已实现 Hybrid V2、Query-only、Hybrid+Query 的统一静态接口和外部相关性评测契约。
- 已冻结 500 条 Query Parser Benchmark；它有结构化解析标签，但没有 Query × 商家相关性标签，不能宣称 Query-aware 排名提升。
- 已实现 `DecisionReadinessAnalyzer`，统一输出任务类型、信息缺口、Hybrid V2-B Top-1 校准置信度和排名不确定原因。
- 已在 5000 个 Validation 任务上按用户做 5 折校准，只使用答案揭晓前可见的特征；Test 未参与拟合或选择。
- Query-aware 置信度会明确返回不可用，不会拿下一商家预测概率冒充 Query 相关性概率。
- 第 19 步的实际结果、限制和复现方法记录在 `docs/experiments/step19_decision_readiness.md`。
