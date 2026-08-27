# Yelp Query-aware 推荐 Agent 学习资源

## Knowledge

- [`MultiRouteRetriever`](src/yelp_agent/retrieval/multi_route.py)
  History 五路召回与内部 RRF 的实际实现。用于核对各路信号、Top-500 和时间边界。
- [`QueryCandidateRetriever`](src/yelp_agent/query_retrieval/engine.py)
  Query 四路召回、硬条件过滤和 Query 内部 RRF 的主要实现。
- [`ProtectedCandidatePool`](src/yelp_agent/query_aware_ranking/candidate_pool.py)
  History Top-500 与 Query Top-500 受保护并集，以及拒绝商家排除逻辑。
- [`QueryAwareRecommendationEngine`](src/yelp_agent/query_aware_ranking/engine.py)
  从并集到 LambdaMART、Embedding、粗排、Top-30 和最终排序的权威数据流。
- [`query_aware_ranking_policy.json`](configs/query_aware_ranking_policy.json)
  当前冻结权重和精排保护规则。用于核对 0.5 Query 权重、Top-30 和移动限制。
- [`step17_hybrid_v2_b_lambdamart.md`](docs/experiments/step17_hybrid_v2_b_lambdamart.md)
  LambdaMART 的训练方式、特征消融和最终 32 维特征选择依据。
- [`step36_Query-aware排序工具接入Agent与500条在线实验报告.md`](docs/experiments/step36_Query-aware排序工具接入Agent与500条在线实验报告.md)
  完整 Agent 接入、500 条实验、指标口径和当前边界。

## Wisdom (Communities)

- 当前优先通过本项目的模拟技术面试检验表达；尚未选择外部社区。

## Gaps

- 尚缺真实线上搜索、点击、预订和满意度日志，当前单正例 Benchmark 不能替代真实用户反馈。
