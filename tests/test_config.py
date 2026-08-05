from pathlib import Path
import shutil

import pytest
import yaml
from pydantic import ValidationError

from yelp_agent.config import (
    build_resolved_configuration,
    configuration_fingerprint,
    load_config,
    load_llm_environment,
    load_retrieval_config,
    load_rolling_training_config,
)


PROJECT_CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_default_project_configuration_loads_mvp_defaults() -> None:
    config = load_config(PROJECT_CONFIG_DIR)

    assert config.data.city == "Philadelphia"
    assert config.data.candidate_count == 20
    assert config.data.random_seed == 42
    assert config.hybrid.weights == {
        "category": 0.4,
        "text": 0.3,
        "quality": 0.2,
        "location": 0.1,
    }
    assert config.agent.timeout_seconds == 90
    assert config.tfidf.ngram_range == (1, 2)
    assert config.tfidf.min_df == 2
    assert config.tfidf.max_features == 50_000
    assert config.tfidf.keyword_count == 10
    assert config.evaluation_data_usage.development_split == "validation"
    assert config.evaluation_data_usage.legacy_test.name == "Legacy Test V0"
    assert config.evaluation_data_usage.strict_blind_holdout is False
    assert config.evaluation_data_usage.cross_validation.folds == 5

    training = load_rolling_training_config(PROJECT_CONFIG_DIR)
    assert training.minimum_history_count == 8
    assert training.maximum_tasks_per_user == 6
    assert training.minimum_target_gap == 2
    assert training.minimum_sample_weight == 0.5
    assert training.maximum_sample_weight == 1.0

    retrieval = load_retrieval_config(PROJECT_CONFIG_DIR)
    assert retrieval.candidate_limit == 500
    assert retrieval.metric_cutoffs == [50, 100, 500]


def test_candidate_buckets_must_describe_one_target_and_nineteen_negatives(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    data_path = config_dir / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["random_negative_count"] = 4
    data_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="negative candidate buckets must sum to 19"):
        load_config(config_dir)


def test_candidate_count_is_fixed_at_twenty_for_the_mvp(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    data_path = config_dir / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["candidate_count"] = 21
    data_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="candidate_count must be exactly 20"):
        load_config(config_dir)


def test_hybrid_weights_must_be_non_negative_and_sum_to_one(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    hybrid_path = config_dir / "hybrid.yaml"
    hybrid = yaml.safe_load(hybrid_path.read_text(encoding="utf-8"))
    hybrid["category_weight"] = 0.5
    hybrid_path.write_text(
        yaml.safe_dump(hybrid, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="hybrid weights must sum to 1"):
        load_config(config_dir)


def test_hybrid_weights_cannot_be_negative(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    hybrid_path = config_dir / "hybrid.yaml"
    hybrid = yaml.safe_load(hybrid_path.read_text(encoding="utf-8"))
    hybrid["category_weight"] = -0.1
    hybrid["location_weight"] = 0.6
    hybrid_path.write_text(
        yaml.safe_dump(hybrid, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="hybrid weights cannot be negative"):
        load_config(config_dir)


def test_unknown_yaml_field_is_rejected_as_a_probable_typo(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    data_path = config_dir / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["random_sead"] = 42
    data_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(config_dir)


def test_fixed_agent_protocol_is_not_a_fake_yaml_knob(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    agent_path = config_dir / "agent.yaml"
    agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    agent["top_k_to_rerank"] = 8
    agent_path.write_text(
        yaml.safe_dump(agent, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(config_dir)


def test_agent_timeout_must_be_positive(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    agent_path = config_dir / "agent.yaml"
    agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    agent["timeout_seconds"] = 0
    agent_path.write_text(
        yaml.safe_dump(agent, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(config_dir)


def test_hybrid_tuning_step_must_divide_one_exactly(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    hybrid_path = config_dir / "hybrid.yaml"
    hybrid = yaml.safe_load(hybrid_path.read_text(encoding="utf-8"))
    hybrid["tuning_step"] = 0.3
    hybrid_path.write_text(
        yaml.safe_dump(hybrid, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must divide 1.0 exactly"):
        load_config(config_dir)


def test_category_groups_must_keep_the_five_benchmark_groups(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    data_path = config_dir / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data["category_groups"].pop("entertainment")
    data_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="category_groups must define"):
        load_config(config_dir)


def test_legacy_test_cannot_be_mislabelled_as_a_blind_holdout(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    policy_path = config_dir / "evaluation_data_usage.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    policy["strict_blind_holdout"] = True
    policy_path.write_text(
        yaml.safe_dump(policy, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Input should be False"):
        load_config(config_dir)


def test_llm_environment_keeps_api_key_secret() -> None:
    settings = load_llm_environment(
        {
            "OPENAI_API_KEY": "top-secret-value",
            "OPENAI_BASE_URL": "https://example.test/v1",
            "OPENAI_MODEL": "example-model",
        }
    )

    assert settings.llm_enabled is True
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "top-secret-value"
    assert settings.base_url == "https://example.test/v1"
    assert settings.model == "example-model"
    assert "top-secret-value" not in repr(settings)
    assert "top-secret-value" not in settings.model_dump_json()


def test_resolved_configuration_is_stable_and_contains_no_environment_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-resolved-config")
    config = load_config(PROJECT_CONFIG_DIR)

    first = build_resolved_configuration(config)
    second = build_resolved_configuration(config.model_copy(deep=True))

    assert first == second
    assert first.fingerprint == configuration_fingerprint(config)
    assert len(first.fingerprint) == 64
    assert "must-not-enter-resolved-config" not in first.model_dump_json()


def test_configuration_fingerprint_changes_with_effective_value() -> None:
    config = load_config(PROJECT_CONFIG_DIR)
    changed = config.model_copy(
        update={
            "agent": config.agent.model_copy(
                update={"timeout_seconds": 89},
            )
        }
    )

    assert configuration_fingerprint(config) != configuration_fingerprint(changed)


def test_rolling_training_weight_range_is_validated(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    training_path = config_dir / "training.yaml"
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    training["minimum_sample_weight"] = 1.1
    training_path.write_text(
        yaml.safe_dump(training, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        load_rolling_training_config(config_dir)


def test_retrieval_metrics_cannot_exceed_frozen_candidate_limit(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    retrieval_path = config_dir / "retrieval.yaml"
    retrieval = yaml.safe_load(retrieval_path.read_text(encoding="utf-8"))
    retrieval["metric_cutoffs"] = [50, 501]
    retrieval_path.write_text(
        yaml.safe_dump(retrieval, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="cannot exceed candidate_limit"):
        load_retrieval_config(config_dir)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"OPENAI_API_KEY": "key-without-model"},
        {"OPENAI_MODEL": "model-without-key"},
    ],
)
def test_missing_key_or_model_enables_safe_no_llm_mode(
    environment: dict[str, str],
) -> None:
    settings = load_llm_environment(environment)

    assert settings.llm_enabled is False
