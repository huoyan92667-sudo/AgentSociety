from __future__ import annotations

import json
import importlib.util
from pathlib import Path


# 评测程序刻意独立于推荐系统运行，避免为了评测基础模型而加载数据库、
# 大模型接口和旧版数据校验依赖。测试也按同样方式直接加载这个文件。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EVALUATOR_PATH = (
    _PROJECT_ROOT
    / "src"
    / "yelp_agent"
    / "recommendation_v2"
    / "review_evidence"
    / "student_training"
    / "evaluate_qwen3_baseline.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen3_baseline_evaluator", _EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EVALUATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVALUATOR)

INVALID = _EVALUATOR.INVALID
_default_paths = _EVALUATOR._default_paths
confusion_matrix = _EVALUATOR.confusion_matrix
load_jsonl = _EVALUATOR.load_jsonl
macro_f1 = _EVALUATOR.macro_f1
normalize_prediction_for_scoring = _EVALUATOR.normalize_prediction_for_scoring
strict_parse_prediction = _EVALUATOR.strict_parse_prediction


def test_strict_prediction_parser_matches_server_evaluator_contract() -> None:
    parsed, json_valid, schema_valid, violation = strict_parse_prediction(
        '{"relevance":3,"strength":2}'
    )
    assert parsed == {"relevance": 3, "strength": 2}
    assert json_valid is True
    assert schema_valid is True
    assert violation is False

    _, json_valid, schema_valid, violation = strict_parse_prediction(
        '```json\n{"relevance":3,"strength":2}\n```'
    )
    assert json_valid is False
    assert schema_valid is False
    assert violation is False

    _, json_valid, schema_valid, violation = strict_parse_prediction(
        '{"relevance":0,"strength":2}'
    )
    assert json_valid is True
    assert schema_valid is False
    assert violation is True


def test_metrics_count_invalid_predictions_as_errors() -> None:
    gold = [0, 1, 2, 3]
    predicted = [0, 2, INVALID, 3]
    macro, per_class = macro_f1(gold, predicted, [0, 1, 2, 3])
    matrix = confusion_matrix(gold, predicted, [0, 1, 2, 3])

    assert 0.0 < macro < 1.0
    assert per_class["1"]["recall"] == 0.0
    assert matrix["2"][INVALID] == 1


def test_numeric_strings_fail_schema_but_count_as_correct_values() -> None:
    parsed, json_valid, schema_valid, _ = strict_parse_prediction(
        '{"relevance":"3","strength":"0"}'
    )

    assert json_valid is True
    assert schema_valid is False
    assert normalize_prediction_for_scoring(parsed) == (3, 0)


def test_unparseable_values_remain_metric_errors() -> None:
    parsed, _, schema_valid, _ = strict_parse_prediction(
        '{"relevance":"high","strength":"low"}'
    )

    assert schema_valid is False
    assert normalize_prediction_for_scoring(parsed) == (INVALID, INVALID)


def test_default_data_paths_point_to_the_committed_student_dataset() -> None:
    base_model, data_dir, output_dir = _default_paths()
    assert base_model == Path(r"D:\model\Qwen3-4B-Instruct-2507")
    assert (data_dir / "validation.jsonl").is_file()
    assert (data_dir / "test.jsonl").is_file()
    assert output_dir.name == "original_model"


def test_load_jsonl_accepts_debug_limit(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    sample = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "review"},
            {"role": "assistant", "content": json.dumps({"relevance": 3, "strength": 4})},
        ]
    }
    path.write_text(
        json.dumps(sample) + "\n" + json.dumps(sample) + "\n",
        encoding="utf-8",
    )
    assert len(load_jsonl(path, max_samples=1)) == 1
