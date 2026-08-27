# Mission: 清晰讲解 Yelp 推荐 Agent

## Why
能够在算法或 AI Agent 技术面试中，用准确、自然、容易理解的语言讲清自己实现的 Yelp Query-aware 推荐 Agent，并经得住对召回、排序、评测和安全边界的追问。

## Success looks like
- 能在 3～5 分钟内独立复述从用户 Query 到 Top-5 的完整数据流
- 能准确解释 History 五路召回、Query 四路召回及受保护并集的设计理由
- 能区分召回、LambdaMART 粗排、Embedding 和 Cross-Encoder 精排各自解决的问题
- 能主动说明单正例 Benchmark、时间截断和未标注相关商家等实验边界

## Constraints
- 优先使用人话、具体例子和输入输出关系，避免堆砌技术名词
- 所有解释以当前仓库代码和冻结实验策略为准
- 学习过程以面试问答和即时纠错为主要反馈方式

## Out of scope
- 暂不扩展到线上部署、前端产品设计或与当前面试无关的通用推荐系统理论
