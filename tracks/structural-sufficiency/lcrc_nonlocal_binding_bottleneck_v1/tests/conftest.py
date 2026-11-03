"""Make ``code/`` importable for the round's tests and pin deterministic seeds."""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def nb():
    """Import the round's modules (heavy frozen imports happen once)."""
    import nb_common
    import nb_cv
    import nb_data
    import nb_features
    import nb_structural

    return {
        "common": nb_common,
        "cv": nb_cv,
        "data": nb_data,
        "features": nb_features,
        "structural": nb_structural,
    }
