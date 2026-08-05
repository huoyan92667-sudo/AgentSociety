"""Record and verify the effective, secret-free configuration of a run."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from yelp_agent.config import (
    AppConfig,
    ResolvedConfiguration,
    build_resolved_configuration,
    load_resolved_configuration,
)
from yelp_agent.experiments.artifacts import (
    ArtifactWriteResult,
    write_json_artifact,
)


class ConfigurationArtifactError(RuntimeError):
    """Raised when a run is paired with missing or different configuration."""


def require_matching_configuration(
    path: str | Path,
    config: AppConfig,
) -> ResolvedConfiguration:
    """Require one existing snapshot to exactly match effective settings."""

    resolved = Path(path)
    try:
        recorded = load_resolved_configuration(resolved)
    except (FileNotFoundError, OSError, ValidationError, ValueError) as exc:
        raise ConfigurationArtifactError(
            f"resolved configuration is missing or invalid: {resolved}"
        ) from exc
    expected = build_resolved_configuration(config)
    if recorded != expected:
        raise ConfigurationArtifactError(
            "existing run was produced with a different configuration; "
            "use force=True to rebuild it"
        )
    return recorded


def write_resolved_configuration(
    path: str | Path,
    config: AppConfig,
    *,
    force: bool = False,
) -> ArtifactWriteResult:
    """Write a deterministic snapshot, refusing silent configuration drift."""

    resolved = Path(path)
    expected = build_resolved_configuration(config)
    if resolved.is_file():
        try:
            recorded = load_resolved_configuration(resolved)
        except (OSError, ValidationError, ValueError) as exc:
            if not force:
                raise ConfigurationArtifactError(
                    f"existing resolved configuration is invalid: {resolved}"
                ) from exc
        else:
            if recorded != expected and not force:
                raise ConfigurationArtifactError(
                    "refusing to overwrite a different resolved configuration; "
                    "use force=True to rebuild the run"
                )
    return write_json_artifact(resolved, expected)
