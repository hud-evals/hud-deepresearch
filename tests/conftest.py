"""Test fixtures for deepresearch API tests."""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Add project root to sys.path so `backend.server` resolves.
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()


def pytest_configure(config):
    """Fail fast if EXA_API_KEY is not set."""
    if not os.getenv("EXA_API_KEY"):
        pytest.exit(
            "EXA_API_KEY environment variable is required to run tests", returncode=1
        )


@pytest.fixture
def client():
    """TestClient for the DeepResearch FastAPI app, reset to fresh state each test."""
    from backend.server import app

    c = TestClient(app)
    c.post("/setup")
    return c
