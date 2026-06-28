"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def sample_prompt() -> str:
    """A small fixed prompt reused across tests."""
    return "Explain virtual memory in one sentence."
