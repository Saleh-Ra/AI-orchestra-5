"""Tests for version exposure and format."""

from airllm_lab import __version__
from airllm_lab.shared.version import __version__ as shared_version


def test_version_is_exposed() -> None:
    """The package re-exports the shared version constant."""
    assert __version__ == shared_version


def test_version_has_three_numeric_parts() -> None:
    """Version follows MAJOR.MINOR.PATCH with numeric parts."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
