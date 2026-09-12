"""Tests for the internal generalization / training-stochasticity audit.

Static and unit tests only: they never train a model, never evaluate official
valid as a decision, and never load official test.  The tests verify the
pre-registered seed factorization, the deterministic 7200/800/2000 split, the
information firewall, the analysis primitives and the factorial formulas.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_internal_generalization_stochasticity_audit as A
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as postv4


# ---------------------------------------------------------------------------
# architecture / split
# ---------------------------------------------------------------------------


def test_exact_82115_params():
    model = shead.build_smallhead(0)
    assert A.shead._n_params(model) == 82115
    assert A.shead._n_params(model.head) == 4135
    assert int(model.unified_graph_width) == 302


def test_split_matches_frozen_manifest():
    roles = A._role_array()
    counts = {label: int((roles == label).sum()) for label in A.SPLIT_ROLES}
    assert counts == {"optimization_train": 7200, "checkpoint_selection": 800, "internal_probe": 2000}
    frozen = A._read_json(A.FROZEN_MANIFEST)
    got = {label: np.flatnonzero(roles == label).astype(int).tolist() for label in A.SPLIT_ROLES}
    assert got["optimization_train"] == frozen["roles"]["adapter_fit"]
    assert got["checkpoint_selection"] == frozen["roles"]["adapter_selection"]
    assert got["internal_probe"] == frozen["roles"]["train_probe"]


def test_split_is_target_independent_and_deterministic():
    roles_a = A._role_array()
    roles_b = A._role_array()
    assert (roles_a == roles_b).all()


# ---------------------------------------------------------------------------
# seed factorization
# ---------------------------------------------------------------------------


def test_I_controls_only_initialization():
    m0a = shead.build_smallhead(0)
    m0b = shead.build_smallhead(0)
    m1 = shead.build_smallhead(1)
    h0a = shead._state_hash(m0a.state_dict())
    h0b = shead._state_hash(m0b.state_dict())
    h1 = shead._state_hash(m1.state_dict())
    assert h0a == h0b
    assert h0a != h1


def test_I0_and_I1_match_historical_init_fingerprints():
    for init_seed in (0, 1):
        hist = A._read_json(
            A.TRACK_ROOT
            / "results/compact_v4_smallhead_e2e"
            / f"initialization_match_seed{init_seed}.json"
        )
        model = shead.build_smallhead(init_seed)
        state = model.state_dict()
        shared = {k: state[k] for k in hist["shared_names"] if k in state}
        assert shead._state_hash(shared) == hist["shared_state_sha256"]


def test_T_order_fingerprint_separable_and_deterministic():
    assert A._trajectory_order_fingerprint(0, 256) == A._trajectory_order_fingerprint(0, 256)
    assert A._trajectory_order_fingerprint(1, 256) == A._trajectory_order_fingerprint(1, 256)
    assert A._trajectory_order_fingerprint(0, 256) != A._trajectory_order_fingerprint(1, 256)


def test_T_order_fingerprint_matches_random_sampler_stream():
    n = 64
    expected = torch.randperm(n, generator=torch.Generator().manual_seed(0 + A.TRAIN_SHUFFLE_SEED_OFFSET))
    digest = hashlib.sha256(",".join(str(int(i)) for i in expected.tolist()).encode()).hexdigest()
    assert digest == A._trajectory_order_fingerprint(0, n)


# ---------------------------------------------------------------------------
# analysis primitives
# ---------------------------------------------------------------------------


def test_argmin_earliest_tie_breaks_to_first():
    assert A._argmin_earliest([0.5, 0.2, 0.2, 0.3]) == 1
    assert A._argmin_earliest([0.1, 0.2, 0.3]) == 0


def test_moving_average_centered_five_epoch():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    out = A._moving_average(values, 5)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[-1]) and np.isnan(out[-2])
    assert out[2] == pytest.approx(3.0)
    assert out[4] == pytest.approx(5.0)


def test_pearson_spearman_monotone():
    x = [1.0, 2.0, 3.0, 4.0]
    assert A._pearson(x, x) == pytest.approx(1.0)
    assert A._spearman(x, x) == pytest.approx(1.0)
    assert A._spearman(x, [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_paired_bootstrap_detects_shift_and_is_deterministic():
    rng = np.random.default_rng(0)
    err_a = rng.normal(0.20, 0.05, 2000)
    err_b = rng.normal(0.20, 0.05, 2000)
    same = A.paired_bootstrap(err_a, err_b, B=400)
    assert same["ci_excludes_zero"] is False
    assert A.paired_bootstrap(err_a, err_b, B=400) == same
    shifted = A.paired_bootstrap(err_a + 0.05, err_b, B=400)
    assert shifted["mae_difference"] == pytest.approx(0.05, abs=0.01)
    assert shifted["ci_excludes_zero"] is True


def test_factorial_effects_formulas():
    # pure initialization effect: P depends only on I
    P = {"I0T0": 0.10, "I0T1": 0.10, "I1T0": 0.13, "I1T1": 0.13}
    e = A._factorial_effects(P)
    assert e["E_I"] == pytest.approx(0.03)
    assert e["E_T"] == pytest.approx(0.0)
    assert e["E_IT"] == pytest.approx(0.0)
    # pure trajectory effect
    P = {"I0T0": 0.10, "I0T1": 0.14, "I1T0": 0.10, "I1T1": 0.14}
    e = A._factorial_effects(P)
    assert e["E_I"] == pytest.approx(0.0)
    assert e["E_T"] == pytest.approx(0.04)
    # pure interaction
    P = {"I0T0": 0.10, "I0T1": 0.14, "I1T0": 0.14, "I1T1": 0.10}
    e = A._factorial_effects(P)
    assert e["E_IT"] == pytest.approx(-0.04)


# ---------------------------------------------------------------------------
# registered protocol
# ---------------------------------------------------------------------------


def test_registered_thresholds_and_lock():
    assert A.S_RAW_MATERIAL == 0.004
    assert A.SELECTION_REGRET_MATERIAL == 0.002
    assert A.OVERFIT_TRAIN_AFTER == 0.005
    assert A.OVERFIT_PROBE_AFTER == 0.003
    assert A.FACTOR_EFFECT == 0.003
    assert A.FACTOR_DOMINANCE == 0.0015
    assert A.BOOTSTRAP_B == 2000
    assert set(A.STAGE1_RUNS) == {"I0T0", "I1T1"}
    assert set(A.STAGE2_RUNS) == {"I0T1", "I1T0"}
    assert A.PROTOCOL["max_epochs"] == 240
    assert A.PROTOCOL["patience"] == 40
    assert A.PROTOCOL["optimizer"] == "Adam"


def test_protocol_lock_declares_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RESULTS_DIR", tmp_path)
    payload = A.stage_protocol()
    assert payload["official_valid_used"] is False
    assert payload["official_test_loaded"] is False
    assert payload["locks"]["no_ema_swa"] is True
    assert payload["locks"]["no_regularization_search"] is True


# ---------------------------------------------------------------------------
# information firewall
# ---------------------------------------------------------------------------


def test_firewall_blocks_valid_extraction():
    original = postv4._extract_v4_records
    try:
        A._install_firewall()
        with pytest.raises(A.InformationFirewallError):
            postv4._extract_v4_records()
    finally:
        postv4._extract_v4_records = original


def test_training_loop_never_references_probe():
    source = inspect.getsource(A.train_run)
    assert 'data["probe"]' not in source
    assert 'data["train"]' in source
    assert "optimizer.step" in source
    # the only evaluation set inside training is the 800 selection set
    assert 'data["select"]' in source


def test_probe_resume_guard_present():
    source = inspect.getsource(A.train_run)
    assert "cannot resume training a run after its internal probe was evaluated" in source
    assert "offline_trajectory_" in source


def test_official_valid_and_test_absent_from_module_source():
    source = inspect.getsource(A)
    assert ("v4_records_" + "valid") not in source
    assert ("_load_zinc(ZINC_ROOT, " + '"test")') not in source
