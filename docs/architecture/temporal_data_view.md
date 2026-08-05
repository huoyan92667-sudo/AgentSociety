# TemporalDataView：统一时点数据读取

## 它解决什么问题

推荐任务只能使用 `cutoff_time` 之前发生的数据。过去 Category、TF-IDF、
Location、Quality 和 Agent Tools 分别读取 Parquet、连接历史并检查时间，规则容易
重复，也容易在新增模型时漏掉检查。

`TemporalDataView` 现在是这些模块共同使用的时间门卫：Parquet 在装配阶段加载一次，
之后所有动态查询都执行严格的 `date < cutoff_time`。

## 对外接口

- `business(business_id)`：读取名称、类别、属性和坐标等静态字段。
- `businesses()`：读取全部静态商家；不包含 Yelp 快照 `stars/review_count`。
- `user_history(user_id, cutoff_time)`：读取用户在时点前的交互。
- `reviews_before(business_id, cutoff_time, limit)`：读取商家在时点前的轻量评论。
- `interactions_before(cutoff_time)`：读取全局选中用户在时点前的交互切片。
- `review_statistics_before(business_ids, cutoff_time)`：读取质量计算所需的时点聚合。

模块不提供无时间限制的 `all_reviews()`。

## 实现方式

初始化时分别建立：

- `business_id -> BusinessRecord` 静态索引；
- `user_id -> 按 (date, review_id) 排序的交互` 索引；
- `business_id -> 评论时间、星级及前缀和` 索引；
- 全局评论和交互时间索引。

查询时使用二分查找定位 cutoff，不为每个任务复制完整 DataFrame。返回值使用冻结的
dataclass、tuple 或只读 mapping，避免调用方修改共享数据。

## 当前接入范围

Hybrid 装配只创建一个 `TemporalDataView`，并共享给 Category、TF-IDF、Quality、
Location 和 Agent Tools。TF-IDF 的训练语料构造仍读取冻结的 validation histories，
因为那是离线训练产物的生成过程，不是在线时点查询。

`temporal_histories.parquet` 暂时继续保留并参与 Hybrid V1 权重指纹。真实 validation
审计已经证明，根据 `user_id + cutoff_time` 动态重建的 263,804 条历史与冻结历史逐条
一致。

## 安全限制

- Ground truth 不进入该模块。
- 商家快照 `stars/review_count` 不进入 `BusinessRecord`。
- cutoff 当时及之后的数据都不可见。
- Legacy Test 不用于实现选择或参数调整。
