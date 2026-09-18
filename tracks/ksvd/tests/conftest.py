"""Shared pytest configuration for the ksvd track.

Most tests are fast static/unit checks. A handful of modules load the frozen
ZINC records / checkpoints or run toy training loops and dominate wall time.
They are marked ``slow`` automatically so a fast *local* subset can be selected
without editing every module:

    uv run pytest -q -m "not slow" tracks/ksvd/tests   # fast local subset
    uv run pytest -q tracks/ksvd/tests                 # full local suite

IMPORTANT: ``slow`` only classifies *known expensive* local tests. It is NOT a
fresh-clone-safety guarantee: many tests that are not marked ``slow`` still need
``data/``, ``tracks/*/results/`` or local checkpoints and will fail on a fresh
clone. GitHub CI therefore does **not** use ``-m "not slow"``; it runs an
explicit, verified fresh-clone-safe allowlist (see
``.github/workflows/research-integrity.yml``).

The ``slow`` marker is registered in ``pyproject.toml``.
"""

from __future__ import annotations

import pytest

# Modules whose collection/fixtures load data or checkpoints or run training.
# Keep this list short; it exists to protect iteration speed and CI.
SLOW_MODULES = {
    "test_canonical_late_readout_adaptation.py",
    "test_compact_v4_identity_capacity_control.py",
    "test_fsar_r2_ar0.py",
    "test_fsar_r2_ar0_edge.py",
    "test_parameter_allocation_representation_leverage.py",
    "test_sbci_fit_generalization_triage.py",
    "test_sbci_fit_matched_generalization_frontier.py",
    "test_stagewise_representation_collision_audit.py",
    "test_triadic_relation_binding_witness.py",
}


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.slow
    for item in items:
        if item.path.name in SLOW_MODULES:
            item.add_marker(marker)
