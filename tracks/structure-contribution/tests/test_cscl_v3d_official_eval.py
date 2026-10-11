"""cscl-v3d frozen official-evaluation tests.

Covers the pure/frozen pieces of ``eval_cscl_v3d_official.py`` (protocol
``configs/cscl_v3d_frozen_official_eval.yaml``):

  * feature-comparison helper: exact on structural fields, tolerant (bounded)
    on float fields, and strict failure propagation;
  * independent S-contribution reconstruction from a SAVED npz (additivity,
    gauge offset identities, per-molecule shift cancellation);
  * pre-frozen bootstrap helper (determinism, seed, percentile CI shape);
  * pre-frozen perturbation semantics for the moment-usage diagnostics;
  * pre-frozen judgment band + O-F equivalence logic on synthetic metrics
    (no official numbers needed).

Heavy end-to-end stages (freeze/replay/extract/score/analyze) are marked
``slow``; they need the committed caches/checkpoints and are exercised by the
actual evaluation run, not by the fast suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for _p in (str(_CODE_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ev():
    import eval_cscl_v3d_official as ev

    return ev


def _bf():
    import strong_backbone_bucket_features as bf

    return bf


# ---------------------------------------------------------------------------
# _features_equal
# ---------------------------------------------------------------------------


def _tiny_bucket_feature(n_units: int = 3, n_rel: int = 2, n_atoms: int = 9, seed: int = 0):
    bf = _bf()
    rng = np.random.default_rng(seed)
    return bf.MolBucketFeatures(
        row=0, y=0.0, n_atoms=n_atoms, n_units=n_units, n_pairs=n_atoms * (n_atoms - 1) // 2,
        unit_sigs=["sig|3|abc"] * n_units, unit_kinds=[0] * n_units,
        unit_atoms=[[k] for k in range(n_units)],
        unit_desc=rng.random((n_units, 40)).astype(np.float32),
        unit_sum_e=rng.random((n_units, 144)).astype(np.float32),
        unit_sum_e2=rng.random((n_units, 144)).astype(np.float32),
        unit_sum_pv=rng.random((n_units, 48)).astype(np.float32),
        unit_sum_pv2=rng.random((n_units, 48)).astype(np.float32),
        unit_n_pv=rng.random(n_units).astype(np.float32) + 0.1,
        rel_lo=np.arange(n_rel, dtype=np.int64), rel_hi=np.arange(1, n_rel + 1, dtype=np.int64),
        rel_sum_pv=rng.random((n_rel, 48)).astype(np.float32),
        rel_sum_pv2=rng.random((n_rel, 48)).astype(np.float32),
        rel_count=rng.random(n_rel).astype(np.float32) + 0.1,
        rel_bucket_counts=rng.random((n_rel, 5)).astype(np.float32),
        rel_bond_hist=rng.random((n_rel, 4)).astype(np.float32),
        rel_mean_relation=rng.random((n_rel, 15)).astype(np.float32),
        ctx_structure=rng.random(30).astype(np.float32),
        topology=rng.random(25).astype(np.float32),
        unit_bucket_counts=rng.random((n_units, 5)).astype(np.float32),
        unit_bucket_sum_pv=rng.random((n_units, 5, 48)).astype(np.float32),
        unit_bucket_sum_pv2=rng.random((n_units, 5, 48)).astype(np.float32),
        rel_bucket_sum_pv=rng.random((n_rel, 5, 48)).astype(np.float32),
        rel_bucket_sum_pv2=rng.random((n_rel, 5, 48)).astype(np.float32),
        pair_ref_first=rng.random((5, 48)).astype(np.float32),
        pair_ref_second=rng.random((5, 48)).astype(np.float32),
        pair_ref_logcount=rng.random(5).astype(np.float32),
    )


def test_features_equal_bit_exact_for_identical_records():
    ev = _ev()
    a = _tiny_bucket_feature()
    exact, worst, field = ev._features_equal(a, _tiny_bucket_feature())
    assert exact and worst == 0.0 and field == ""


def test_features_equal_tolerates_small_float_drift_but_reports_it():
    ev = _ev()
    a = _tiny_bucket_feature()
    b = _tiny_bucket_feature()
    b.pair_ref_first = b.pair_ref_first + np.float32(1e-6)
    exact, worst, field = ev._features_equal(a, b)
    assert not exact
    assert 0.0 < worst <= 5e-6 and field == "pair_ref_first"


def test_features_equal_flags_structural_mismatch_as_infinite():
    ev = _ev()
    a = _tiny_bucket_feature()
    b = _tiny_bucket_feature()
    b.unit_sigs = ["different|sig"] * b.n_units
    exact, worst, field = ev._features_equal(a, b)
    assert not exact and worst == np.inf and field == "unit_sigs"


def test_features_equal_ignores_only_the_label_field():
    ev = _ev()
    a = _tiny_bucket_feature()
    b = _tiny_bucket_feature()
    b.y = -7.0  # audit-only; extraction passes ys=0 and must not consume this
    exact, _, _ = ev._features_equal(a, b)
    assert exact


# ---------------------------------------------------------------------------
# _verify_contrib_file: independent reconstruction from the SAVED npz
# ---------------------------------------------------------------------------


def _write_synthetic_contrib(path, n_mols=4, n_units=10, n_rel=6, seed=3, atol=1e-6):
    """Build a self-consistent contributions npz from per-unit/per-rel terms."""
    ev = _ev()
    rng = np.random.default_rng(seed)
    unit_mol = np.sort(rng.integers(0, n_mols, size=n_units))
    rel_mol = np.sort(rng.integers(0, n_mols, size=n_rel))
    unit_type = rng.integers(0, 5, size=n_units)
    alpha_raw = rng.normal(0, 0.3, n_units)
    delta_raw = rng.normal(0, 0.5, n_units)
    offsets = np.arange(5, dtype=np.float64) * 0.1
    alpha = alpha_raw + offsets[unit_type]
    delta = delta_raw - offsets[unit_type]
    gamma = rng.normal(0, 0.2, n_rel)
    g_ctx = rng.normal(0, 0.1, n_mols)
    g_topo = rng.normal(0, 0.1, n_mols)
    bias = 0.37
    usum = np.zeros(n_mols); np.add.at(usum, unit_mol, alpha + delta)
    rsum = np.zeros(n_mols); np.add.at(rsum, rel_mol, gamma)
    pred = bias + usum + rsum + g_ctx + g_topo
    atom_flat = np.arange(0, 2 * n_units, dtype=np.int64)   # 2 atoms per unit
    atom_ptr = np.arange(0, 2 * (n_units + 1), 2, dtype=np.int64)
    np.savez_compressed(
        path, arm="S_bucket", protocol=ev.PROTOCOL_ID, split="synthetic",
        bias=bias, rows=np.arange(n_mols), pred=pred,
        unit_mol=unit_mol, unit_type=unit_type, unit_offset=offsets[unit_type],
        alpha=alpha, delta=delta, alpha_raw=alpha_raw, delta_raw=delta_raw,
        unit_atom_flat=atom_flat, unit_atom_ptr=atom_ptr,
        rel_mol=rel_mol, gamma=gamma, g_ctx=g_ctx, g_topo=g_topo,
        g_ctx_per_feature=g_ctx[:, None], g_topo_per_feature=g_topo[:, None],
    )
    return pred


def test_verify_contrib_file_passes_on_consistent_export(tmp_path):
    ev = _ev()
    path = tmp_path / "syn_S_contribs.npz"
    _write_synthetic_contrib(path)
    res = ev._verify_contrib_file(path, atol=1e-6)
    assert res["ok"], res
    assert res["additivity_max_abs_residual_canonical"] <= 1e-6
    assert res["alpha_offset_identity_max_abs"] <= 1e-6
    assert res["unit_sum_shift_cancellation_max_abs"] <= 1e-6
    assert res["atom_membership_strictly_increasing"]


def test_verify_contrib_file_detects_corrupted_pred(tmp_path):
    ev = _ev()
    path = tmp_path / "bad_S_contribs.npz"
    pred = _write_synthetic_contrib(path)
    z = dict(np.load(path, allow_pickle=False))
    z["pred"] = pred + 1e-3   # break strict additivity
    np.savez_compressed(path, **z)
    res = ev._verify_contrib_file(path, atol=1e-5)
    assert not res["ok"]
    assert res["additivity_max_abs_residual_canonical"] > 1e-5


def test_verify_contrib_file_detects_broken_gauge_identity(tmp_path):
    ev = _ev()
    path = tmp_path / "bad_gauge_S_contribs.npz"
    _write_synthetic_contrib(path)
    z = dict(np.load(path, allow_pickle=False))
    z["alpha"] = z["alpha"] + 0.01   # alpha no longer equals alpha_raw + offset
    np.savez_compressed(path, **z)
    res = ev._verify_contrib_file(path, atol=1e-5)
    assert not res["ok"]
    assert res["alpha_offset_identity_max_abs"] > 1e-5


# ---------------------------------------------------------------------------
# pre-frozen bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_and_percentile_shaped():
    ev = _ev()
    d = np.linspace(-1.0, 1.0, 100)
    a = ev._bootstrap_ci(d)
    b = ev._bootstrap_ci(d)
    assert a == b                                  # seed 0 determinism
    assert a["resamples"] == ev.BOOTSTRAP_N == 5000 and a["seed"] == 0
    lo, hi = a["ci95"]
    assert lo <= a["mean_diff"] <= hi
    assert abs(a["mean_diff"]) < 1e-12             # symmetric d -> mean ~ 0


def test_prefrozen_constants_match_the_protocol_yaml():
    ev = _ev()
    assert ev.ATOL_SUM == 1e-5
    assert ev.TOL_BAND == 0.02
    assert ev.EQUIV_REGION == (-0.01, 0.01)
    assert ev.BOOTSTRAP_N == 5000 and ev.BOOTSTRAP_SEED == 0
    assert ev.RECORDED_DEV == {"F": 0.11641, "O_bucket": 0.11610, "S_bucket": 0.12694}


# ---------------------------------------------------------------------------
# pre-frozen perturbation semantics (moment-usage diagnostics)
# ---------------------------------------------------------------------------


def test_perturbations_zero_exactly_their_target_blocks():
    ev = _ev()
    base = _tiny_bucket_feature()
    for kind, fields in (
        ("zero_unit_moments", ("unit_bucket_sum_pv", "unit_bucket_sum_pv2")),
        ("zero_unit_all", ("unit_bucket_counts", "unit_bucket_sum_pv", "unit_bucket_sum_pv2")),
        ("zero_rel_moments", ("rel_bucket_sum_pv", "rel_bucket_sum_pv2")),
        ("zero_rel_all", ("rel_bucket_counts", "rel_bucket_sum_pv", "rel_bucket_sum_pv2")),
    ):
        (pert,) = ev._perturbed_features([base], kind)
        for f in fields:
            assert np.all(getattr(pert, f) == 0.0), (kind, f)
        # everything else untouched
        assert np.array_equal(pert.unit_sum_e, base.unit_sum_e)
        assert np.array_equal(pert.rel_mean_relation, base.rel_mean_relation)
        assert pert.unit_sigs == base.unit_sigs


def test_per_bucket_perturbation_targets_one_bucket_only():
    ev = _ev()
    base = _tiny_bucket_feature()
    (pert,) = ev._perturbed_features([base], "zero_rel_bucket_moments", bucket=2)
    assert np.all(pert.rel_bucket_sum_pv[:, 2, :] == 0.0)
    assert np.all(pert.rel_bucket_sum_pv2[:, 2, :] == 0.0)
    for b in (0, 1, 3, 4):
        assert np.array_equal(pert.rel_bucket_sum_pv[:, b, :], base.rel_bucket_sum_pv[:, b, :])
    # counts are never touched by the *_moments variants
    assert np.array_equal(pert.rel_bucket_counts, base.rel_bucket_counts)


# ---------------------------------------------------------------------------
# pre-frozen judgment band (synthetic metrics, no official numbers)
# ---------------------------------------------------------------------------


def _write_metrics(out_dir, split, so, sf, of):
    out_dir.mkdir(parents=True, exist_ok=True)
    def _mk(diff, ci):
        return {"mean_diff": diff, "ci95": ci}
    (out_dir / f"{split}_metrics.json").write_text(json.dumps({
        "paired_differences": {
            "S_bucket_minus_O_bucket": _mk(*so),
            "S_bucket_minus_F": _mk(*sf),
            "O_bucket_minus_F": _mk(*of),
        }
    }))


def test_judgment_strong_support_band(tmp_path, monkeypatch):
    ev = _ev()
    monkeypatch.setattr(ev, "OUT", tmp_path)
    _write_metrics(tmp_path, "valid", (+0.011, [0.005, 0.017]), (+0.010, [0.004, 0.016]), (-0.001, [-0.005, 0.004]))
    _write_metrics(tmp_path, "test", (+0.012, [0.007, 0.018]), (+0.011, [0.005, 0.017]), (-0.001, [-0.006, 0.004]))
    res = ev.stage_judgment(log=lambda *a, **k: None)
    assert res["band"] == "strong_support" and res["direction_consistent"]


def test_judgment_limited_support_band(tmp_path, monkeypatch):
    ev = _ev()
    monkeypatch.setattr(ev, "OUT", tmp_path)
    _write_metrics(tmp_path, "valid", (+0.012, [0.004, 0.020]), (+0.011, [0.003, 0.019]), (0.0, [-0.004, 0.004]))
    _write_metrics(tmp_path, "test", (+0.015, [0.005, 0.026]), (+0.014, [0.004, 0.025]), (0.0, [-0.005, 0.005]))
    res = ev.stage_judgment(log=lambda *a, **k: None)
    assert res["band"] == "limited_support"      # point estimates <= 0.02, CI upper > 0.02


def test_judgment_no_support_band(tmp_path, monkeypatch):
    ev = _ev()
    monkeypatch.setattr(ev, "OUT", tmp_path)
    _write_metrics(tmp_path, "valid", (+0.011, [0.005, 0.017]), (+0.010, [0.004, 0.016]), (0.0, [-0.004, 0.004]))
    _write_metrics(tmp_path, "test", (+0.035, [0.025, 0.045]), (+0.034, [0.024, 0.044]), (0.0, [-0.005, 0.005]))
    res = ev.stage_judgment(log=lambda *a, **k: None)
    assert res["band"] == "no_support"


def test_judgment_direction_flip_blocks_strong_band(tmp_path, monkeypatch):
    ev = _ev()
    monkeypatch.setattr(ev, "OUT", tmp_path)
    _write_metrics(tmp_path, "valid", (+0.010, [0.005, 0.015]), (+0.010, [0.005, 0.015]), (0.0, [-0.004, 0.004]))
    _write_metrics(tmp_path, "test", (-0.005, [-0.012, 0.002]), (+0.010, [0.005, 0.015]), (0.0, [-0.005, 0.005]))
    res = ev.stage_judgment(log=lambda *a, **k: None)
    assert not res["direction_consistent"] and res["band"] != "strong_support"


def test_o_f_equivalence_requires_full_ci_inside_region(tmp_path, monkeypatch):
    ev = _ev()
    monkeypatch.setattr(ev, "OUT", tmp_path)
    _write_metrics(tmp_path, "valid", (+0.01, [0.0, 0.02]), (+0.01, [0.0, 0.02]), (-0.004, [-0.009, 0.000]))
    _write_metrics(tmp_path, "test", (+0.01, [0.0, 0.02]), (+0.01, [0.0, 0.02]), (-0.004, [-0.009, 0.001]))
    res = ev.stage_judgment(log=lambda *a, **k: None)
    assert res["o_vs_f_ci_within_region"] and "equivalent" in res["o_vs_f_wording"]

    _write_metrics(tmp_path, "valid", (+0.01, [0.0, 0.02]), (+0.01, [0.0, 0.02]), (-0.004, [-0.012, 0.004]))
    _write_metrics(tmp_path, "test", (+0.01, [0.0, 0.02]), (+0.01, [0.0, 0.02]), (-0.004, [-0.012, 0.004]))
    res = ev.stage_judgment(log=lambda *a, **k: None)
    assert not res["o_vs_f_ci_within_region"]
    assert "NOT established" in res["o_vs_f_wording"]
