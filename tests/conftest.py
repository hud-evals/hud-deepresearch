"""Test fixtures for deepresearch tests."""

import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `env` resolves.
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    """Fail fast if EXA_API_KEY is not set."""
    if not os.getenv("EXA_API_KEY"):
        pytest.exit(
            "EXA_API_KEY environment variable is required to run tests", returncode=1
        )


@pytest.fixture(autouse=True)
def reset_state():
    """Reset env.state before each test so counts/answers don't bleed across."""
    from env import state

    state.reset()
    yield
    state.reset()
