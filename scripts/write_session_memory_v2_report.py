"""Write the tracked Step 34.5 report from ignored experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _delta(new: float, old: float) -> str:
    return f"{(new - old) * 100:+.2f} pp"


def _comparison_table(
    rule: dict[str, float],
    deepseek: dict[str, float],
    metrics: tuple[tuple[str, str], ...],
) -> list[str]:
    rows = ["| 指标 | Rule | DeepSeek | 差值 |", "|---|---:|---:|---:|"]
    for label, key in metrics:
        rows.append(
            f"| {label} | {_pct(rule[key])} | {_pct(deepseek[key])} "
            f"| {_delta(deepseek[key], rule[key])} |"
        )
    return rows


def _build_report(root: Path) -> str:
    run_root = root / "runs" / "session_memory_v2"
    benchmark_root = root / "benchmarks" / "session_memory_v2"
    manifest = _read(benchmark_root / "manifest.json")
    audit = _read(benchmark_root / "audit_report.json")
    rule = _read(run_root / "rule" / "metrics.json")
    deepseek = _read(run_root / "deepseek" / "metrics.json")
    deepseek_usage = _read(run_root / "deepseek" / "llm" / "llm_usage.json")
    agent_rule = _read(run_root / "agent_rule_final" / "metrics.json")
    agent_deepseek = _read(run_root / "agent_deepseek_cache_final" / "metrics.json")
    agent_usage = _read(
        run_root / "agent_deepseek_cache_final" / "memory_llm" / "llm_usage.json"
    )

    generation_tokens = sum(
        manifest[key]
        for key in (
            "generation_input_tokens",
            "generation_output_tokens",
            "review_input_tokens",
            "review_output_tokens",
        )
    )
    # The first complete run predates the final normalized-output reruns. Its ledger
    # is retained in the experiment log rather than the current cache snapshot.
    first_memory_run_tokens = 1_310_084
    normalization_rerun_tokens = 312_691 + 331_248
    known_api_tokens = (
        generation_tokens + first_memory_run_tokens + normalization_rerun_tokens
    )

    overall_rule = rule["overall"]
    overall_deepseek = deepseek["overall"]
    validation_rule = rule["by_split"]["validation"]
    validation_deepseek = deepseek["by_split"]["validation"]

    memory_metrics = (
        ("任务类型准确率", "task_type_accuracy"),
        ("任务目标准确率", "task_goal_accuracy"),
        ("条件字段+操作召回率", "semantic_condition_field_operation_recall"),
        ("条件核心语义召回率", "semantic_condition_core_recall"),
        ("相对偏好 Precision", "semantic_relative_precision"),
        ("相对偏好 Recall", "semantic_relative_recall"),
        ("条件真正写入 Memory 的召回率", "memory_condition_application_recall"),
        ("相对偏好真正写入 Memory 的召回率", "memory_relative_application_recall"),
        ("信息缺口完全匹配率", "information_gap_exact_match"),
        ("指代解析召回率", "reference_resolution_recall"),
        ("拒绝商家召回率", "rejected_business_recall"),
        ("不应修改状态准确率", "no_state_change_accuracy"),
        ("冻结候选行为 Compliance@1", "behavior_compliance_at_1"),
        ("冻结候选行为 Compliance@5", "behavior_compliance_at_5"),
        ("数字幻觉率", "numeric_hallucination_rate"),
    )
    agent_metrics = (
        ("Fallback 会话率", "fallback_session_rate"),
        ("脚本轮次释放率", "released_turn_rate"),
        ("触发动作准确率", "trigger_accuracy"),
        ("有效推荐率", "valid_recommendation_rate"),
        ("拒绝商家排除率", "rejected_business_exclusion_rate"),
        ("真实 Agent Compliance@1", "behavior_compliance_at_1"),
        ("真实 Agent Compliance@5", "behavior_compliance_at_5"),
    )

    lines = [
        "# 第 34.5 步：Session Memory Benchmark V2 与真实 Agent 评测报告",
        "",
        "## 1. 为什么重做 Benchmark",
        "",
        "旧评测把“更便宜”强行标成 `price_level <= 2`，把“更近”强行标成 "
        "`distance_km <= 3`。用户没有说出这些数字，却会因为没有预测同一个数字而被扣分。V2 "
        "把相对要求标成 `price=lower`、`distance=closer`、`noise=quieter`；只有用户明确说出数字时，才评测数字。",
        "",
        "标准答案由代码根据冻结的首轮 Top-5、候选价格、距离和噪声属性动态计算。DeepSeek "
        "负责把已知意图改写成自然问句，并由另一次独立调用审核自然度和标签一致性；DeepSeek 不决定 Ground Truth。",
        "",
        "## 2. 数据规模与泄漏审计",
        "",
        f"- 会话：{manifest['session_count']}；后续轮次：{manifest['turn_count']}。",
        f"- Development / validation：{manifest['split_counts']['development']} / "
        f"{manifest['split_counts']['validation']}。",
        f"- 中文 / 英文：{manifest['language_counts']['zh-CN']} / "
        f"{manifest['language_counts']['en-US']}。",
        "- 场景：明确条件 200、相对偏好 110、拒绝/指代 70、组合修改 50、"
        "不应改变状态 30、澄清回答 20、冲突解决 20。",
        f"- 审计通过：`{audit['passed']}`；完全重复 {audit['exact_duplicate_count']}；"
        f"跨 split 重复 {audit['cross_split_duplicate_count']}；近重复 {audit['near_duplicate_count']}；"
        f"商家 ID 泄漏 {audit['leaked_business_id_count']}；相对请求数字泄漏 "
        f"{audit['relative_numeric_leak_count']}；非法会话序列 {audit['invalid_session_sequence_count']}。",
        "",
        "可见问题、冻结展示列表与隐藏标签分文件保存。Memory Manager、Router 和 Agent "
        "运行时无法读取隐藏标签，只有 Benchmark Evaluator 可以读取。",
        "",
        "## 3. 三层评测口径",
        "",
        "1. **语义层**：本轮 `MemoryProposal` 是否正确理解任务、增删改条件、相对偏好、"
        "澄清答案和指代。",
        "2. **Memory 层**：Reducer 执行后，正确内容是否真正进入 canonical state，而不只是模型输出里出现。",
        "3. **行为层**：把更新后的状态应用到冻结候选，或交给完整 Agent 工具链，检查 Top-1/Top-5 "
        "是否满足新要求、拒绝商家是否被排除。",
        "",
        "## 4. Rule 与 DeepSeek Memory 离线对比（500 轮）",
        "",
        *_comparison_table(overall_rule, overall_deepseek, memory_metrics),
        "",
        "DeepSeek 明显理解了规则难以覆盖的自然语言：相对偏好 Recall 从 0% 到 100%，"
        "任务目标准确率从 50.0% 到 90.6%，冻结候选 Compliance@1 从 68.6% 到 96.3%。",
        "",
        "但这不是全胜。信息缺口完全匹配率从 95.6% 降到 69.4%，拒绝商家召回率也低于 Rule；"
        "说明模型会过度解释，且“不要第 N 家”的指代—拒绝链路仍不稳定。严格条件五元组 Recall "
        "只有 18.1%，而条件核心语义 Recall 为 75.2%，说明多数时候意思理解对了，但 operator/importance "
        "等标准化字段没有对齐。",
        "",
        "### 冻结 validation（100 轮）",
        "",
        *_comparison_table(validation_rule, validation_deepseek, memory_metrics),
        "",
        "Validation 只用于最终报告，没有再用它调整规则或阈值。",
        "",
        "## 5. 完整 Agent Harness 端到端对比（200 会话 / 500 轮）",
        "",
        "完整链路包含 Query 召回、硬约束、LightGBM/Hybrid V2、本地 Qwen Embedding、"
        "本地 Qwen Reranker、语义排序、Top-5 展示和 Session Memory。",
        "",
        *_comparison_table(agent_rule, agent_deepseek, agent_metrics),
        "",
        f"平均会话延迟：Rule {agent_rule['mean_session_latency_ms']:.0f} ms；"
        f"DeepSeek cache-only {agent_deepseek['mean_session_latency_ms']:.0f} ms。",
        "",
        f"端到端 DeepSeek 组使用了 {agent_usage['cache_hit_count']} 条已经审核的 Memory Proposal，"
        f"另有 {agent_usage['logical_call_count'] - agent_usage['cache_hit_count']} 条 cache miss "
        "按 Rule 安全回退；本次回放新增 provider call 为 "
        f"{agent_usage['provider_call_count']}、新增 API Token 为 {agent_usage['total_tokens']}。"
        "因此这一列准确名称是 **DeepSeek Memory cache-only**，不是伪装成 500 轮全量在线调用。",
        "",
        "结论必须如实报告：DeepSeek 把 Fallback 从 45.0% 降到 16.5%，轮次释放率从 83.0% "
        "提高到 92.6%，拒绝商家排除率从 76.1% 提高到 89.1%；但真实 Agent Compliance@1 "
        "从 18.1% 降到 15.9%，Compliance@5 从 26.3% 降到 23.2%。它让 Agent 更能继续执行和"
        "保持会话，却没有让最终商家排序更准。这证明 Memory 的语义指标不能代替端到端推荐指标。",
        "",
        "## 6. 本步修复的运行时问题",
        "",
        "- 多轮重新召回曾被 Harness 误判为 `unauthorized_business_scope_change`。现在只有 "
        "`retrieve_candidates` 可以合法替换范围；硬约束仍只能缩小范围，其他工具仍不能换范围。",
        "- 本地 Embedding/Reranker 工作量曾与外部 LLM Token 共用 12k 上限，造成大量 "
        "`max_total_tokens_exceeded`。V2 保持 Top-30/Top-20 和序列长度限制，但将本地工作预算"
        "提高到 100k；外部 API 仍由独立 LLM 配置限制。",
        "- 发给 DeepSeek 的 Memory Context 已移除真实用户/商家 ID、候选列表和自由文本语义摘要。"
        "只发送结构化条件、信息缺口、相对偏好、结果序号和数量；显式 ID 使用一次性别名后在本地恢复。",
        "- Replay Runner 支持 awaiting-user 快照和每 10 个会话 checkpoint，长跑失败不再丢掉全部结果。",
        "",
        "## 7. API Token 账单",
        "",
        f"- Benchmark 生成和独立审核：{manifest['provider_call_count']} 次调用，"
        f"输入 {manifest['generation_input_tokens'] + manifest['review_input_tokens']:,}，"
        f"输出 {manifest['generation_output_tokens'] + manifest['review_output_tokens']:,}，"
        f"共 {generation_tokens:,} tokens。",
        f"- 首次 500 条 Memory 全量实验：共 {first_memory_run_tokens:,} tokens。",
        f"- 两轮非法输出规范化复跑：新增 {normalization_rerun_tokens:,} tokens。",
        "- 完整 Agent cache-only 回放：0 次新 provider call，0 个新 API Token。",
        f"- 本步可核对的累计 API 消耗：至少 **{known_api_tokens:,} tokens**。",
        "",
        "最早一轮因问题重复而废弃的 500 条生成/审核，旧脚本没有在写入结果前保存 Token 台账。"
        "其精确消耗无法恢复，因此没有猜测或混入累计值；“至少”正是因为这一笔未知。"
        "Agent 指标中的 `input_tokens` 是本地模型工作量估计，不是外部 API Token。",
        "",
        "## 8. 下一步改进",
        "",
        "1. 统一标签、Proposal 和 Reducer 的 operator/importance canonicalization，同时保留严格匹配和核心语义匹配。",
        "2. 为“不要第 N 家”建立统一的 reference + rejection contract，供 Parser、Resolver、Reducer、Router 共用。",
        "3. 把相对偏好、拒绝列表和结构化 Delta 直接变成 Query 召回、硬约束与 LightGBM/语义排序特征，"
        "而不是只存在 Memory 中。",
        "4. 将 Agent 的 provider Token、本地模型 work 和工具调用预算拆成三个独立字段。",
        "5. 再比较 Rule、DeepSeek、Constrained LLM 和 Cost-aware LLM；同时报告 HR@1/3/5/10 "
        "与 Query-conditioned Compliance@1/5，不能只看动作是否正确。",
        "",
        "## 9. 复现入口",
        "",
        "- `scripts/build_session_memory_benchmark_v2.py`：生成、审核并冻结 Benchmark。",
        "- `scripts/run_session_memory_benchmark_v2.py`：运行 Rule/DeepSeek Memory 三层离线评测。",
        "- `scripts/run_session_memory_agent_replay_v2.py`：运行真实 Agent 多轮回放。",
        "- `configs/rule_router_memory_v2.yaml`：本步专用 Top-5 Router 配置，不修改 Step 24 冻结的 Top-3 基线。",
        "",
        "大体积 trace 位于 `runs/session_memory_v2/`，按仓库规则不提交；冻结 Benchmark、审计、"
        "Manifest、代码、测试和本报告提交 Git。",
        "",
        "### 当前最终运行台账",
        "",
        f"- DeepSeek 离线最终运行：logical calls={deepseek_usage['logical_call_count']}，"
        f"cache hits={deepseek_usage['cache_hit_count']}，provider calls={deepseek_usage['provider_call_count']}，"
        f"本轮新增 tokens={deepseek_usage['total_tokens']}。",
        f"- Agent DeepSeek cache-only：logical calls={agent_usage['logical_call_count']}，"
        f"cache hits={agent_usage['cache_hit_count']}，provider calls={agent_usage['provider_call_count']}。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (
        root
        / "docs"
        / "experiments"
        / "step34_5_session_memory_benchmark_v2_and_real_agent_report.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_build_report(root), encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
