"""Explicit error classes understood by the tool registry."""


class RetryableToolError(RuntimeError):
    """A transient dependency failure that may be retried in the same call."""


class PermanentToolError(RuntimeError):
    """A deterministic failure that retrying unchanged input cannot fix."""
