"""Test fixtures for the deepresearch env."""

import sys
from pathlib import Path

# Make `env` and `tasks` importable.
ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
