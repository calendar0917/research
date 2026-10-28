"""CPU gates G0..G12 for CJP-SSL v1 (all marked ``slow``: they load ZINC).

Run explicitly::

    uv run pytest -q tracks/structural-sufficiency/zinc_cjp_ssl_v1/tests

They are not in the default ``testpaths`` and are not in the CI allowlist: they
need ``data/ZINC``, the frozen P3 objects and the historical SSL checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import cjp_gates as G  # noqa: E402

pytestmark = pytest.mark.slow


def _check(result):
    assert result["pass"], result["detail"]
    return result


def test_g0_frozen_trainer_lineage():
    _check(G.gate_sup_trainer_lineage())


def test_g1_label_isolation():
    _check(G.gate_g1())


def test_g2_masking_off_equals_frozen_cache():
    _check(G.gate_g2())


def test_g3_masked_rebuild_matches_reference():
    _check(G.gate_g3())


def test_g4_hidden_truth_swap():
    _check(G.gate_g4())


def test_g5_both_directed_masks():
    _check(G.gate_g5())


def test_g6_no_mask_token_as_type():
    _check(G.gate_g6())


def test_g7_no_mask_forward_equals_lcrc():
    _check(G.gate_g7())


def test_g8_invariance():
    _check(G.gate_g8())


def test_g9_marg_factorisation_and_shared_init():
    _check(G.gate_g9())


def test_g10_gradient_isolation():
    _check(G.gate_g10())


def test_g11_localcontent_gradients():
    _check(G.gate_g11())


def test_g12_export_and_deploy_compatibility():
    # depends on the tiny pretraining export written by G1
    _check(G.gate_g1())
    _check(G.gate_g12())
