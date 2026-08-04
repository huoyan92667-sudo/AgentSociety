from pathlib import Path
import shutil

import pytest
import yaml
from pydantic import ValidationError

from yelp_agent.config import load_config, load_llm_environment


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
    assert config.agent.top_k_to_rerank == 8
    assert config.evaluation.hit_cutoffs == (1, 3, 5)
    assert config.tfidf.ngram_range == (1, 2)
    assert config.tfidf.min_df == 2
    assert config.tfidf.max_features == 50_000
    assert config.tfidf.keyword_count == 10


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


def test_agent_top_k_cannot_exceed_candidate_count(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(PROJECT_CONFIG_DIR, config_dir)
    agent_path = config_dir / "agent.yaml"
    agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    agent["top_k_to_rerank"] = 21
    agent_path.write_text(
        yaml.safe_dump(agent, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="less than or equal to 20"):
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
