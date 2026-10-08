"""Focused tests for zinc_cssd_g_component_attribution_v1.

Only the NEW pure-numpy risks of this round: the component synthesis
identities (error sign convention, triangle gap >= 0, exact e_g = e_ell + e_s),
the k-group contribution add-back, the coarse structure groupings (priority
partition, bond semantics, node terciles), the SMILES element heuristic used
for the atom-mapping cross-check, and the group bootstrap (determinism, shared
resampling, DICT-RAW direction).  No data, no checkpoints, no GPU, no torch at
module import; the frozen-checkpoint restore/replay/export verification lives
in the runner's ``export`` stage.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_g_component_attribution_v1 as m,
)


# ---- component_stats ---------------------------------------------------------


def test_component_stats_recovers_mae_and_identities():
    rng = np.random.default_rng(0)
    ell_hat = rng.normal(size=500)
    s_hat = rng.normal(size=500)
    ell = rng.normal(size=500)
    s = rng.normal(size=500)
    h = ell_hat + s_hat
    g = ell + s
    stats = m.component_stats(ell_hat - ell, s_hat - s, h - g)
    assert stats["identity_gap_e_g_minus_e_ell_minus_e_s"] < 1e-12
    assert stats["n"] == 500
    # sign convention: MAE of the pure-offset error equals the offset size
    one = np.ones(50)
    st = m.component_stats(one, -one, np.zeros(50))
    assert st["MAE_ell"] == pytest.approx(1.0)
    assert st["MAE_s"] == pytest.approx(1.0)
    assert st["MAE_g"] == pytest.approx(0.0)
    assert st["opposite_sign_fraction"] == pytest.approx(1.0)
    assert st["triangle_gap"] == pytest.approx(2.0)  # 1+1-0 per row
    assert st["bias_ell"] == pytest.approx(1.0)
    assert st["bias_s"] == pytest.approx(-1.0)
    assert st["abs_s_share"] == pytest.approx(0.5)


def test_component_stats_triangle_gap_nonnegative():
    rng = np.random.default_rng(1)
    for _ in range(5):
        e_ell, e_s = rng.normal(size=200), rng.normal(size=200)
        stats = m.component_stats(e_ell, e_s, e_ell + e_s)
        assert stats["triangle_gap"] >= -1e-12


# ---- k-group contributions ---------------------------------------------------


def test_k_group_contributions_add_back_exactly():
    rng = np.random.default_rng(2)
    k = rng.choice([0, -1, -2, -5], size=1000, p=[0.9, 0.06, 0.03, 0.01])
    e_ell, e_s, e_g = rng.normal(size=1000), rng.normal(size=1000), rng.normal(size=1000)
    rows, resid = m.k_group_component_rows(k, e_ell, e_s, e_g, 1000)
    assert resid["groups_cover_all_rows"]
    for comp, key in (("s", "C_s"), ("ell", "C_ell"), ("g", "C_g")):
        total = float(np.abs({"s": e_s, "ell": e_ell, "g": e_g}[comp]).mean())
        assert abs(sum(r[key] for r in rows) - total) < 1e-12
        assert abs(resid[f"addback_C_{comp}_residual"]) < 1e-12
    assert sum(r["n"] for r in rows) == 1000


# ---- coarse structure groupings ---------------------------------------------


def test_atom_group_priority_partition():
    # halogen beats S/P beats N/O beats C-only
    assert m.atom_group_of(np.array([0, 1, 2])) == "N_or_O"          # C, O, N
    assert m.atom_group_of(np.array([0, 5])) == "S_or_P"             # C + S
    assert m.atom_group_of(np.array([0, 5, 2])) == "S_or_P"          # S beats N
    assert m.atom_group_of(np.array([0, 6, 5, 2])) == "halogen"      # Cl beats all
    assert m.atom_group_of(np.array([0, 4])) == "C_only"             # plain + bracket C
    assert m.atom_group_of(np.array([0, 3])) == "halogen"            # F


def test_bond_group_semantics():
    assert m.bond_group_of(np.array([1, 1, 1])) == "single_only"
    assert m.bond_group_of(np.array([1, 2, 1])) == "double_or_aromatic"
    assert m.bond_group_of(np.array([1, 2, 3])) == "triple"
    with pytest.raises(RuntimeError):
        m.bond_group_of(np.array([1, 7]))


def test_node_tercile_rule_covers_all_rows():
    rng = np.random.default_rng(3)
    nc = rng.integers(5, 60, size=999)  # non-divisible size on purpose
    bounds = m.node_tercile_bounds(nc)
    grp = m.node_tercile_group(nc, bounds)
    assert set(np.unique(grp)) <= {0, 1, 2}
    assert (grp[nc <= bounds[0]] == 0).all()
    assert (grp[(nc > bounds[0]) & (nc <= bounds[1])] == 1).all()
    assert (grp[nc > bounds[1]] == 2).all()


# ---- SMILES element heuristic (cross-check support) --------------------------


def test_smiles_elements_counts_common_forms():
    assert m.smiles_elements("CCO") == {"C": 2, "O": 1}
    assert m.smiles_elements("c1ccncc1") == {"C": 5, "N": 1}          # aromatic fold
    assert m.smiles_elements("ClCCBr") == {"C": 2, "Cl": 1, "Br": 1}  # two-letter
    assert m.smiles_elements("C[C@@H](N)C(=O)O") == {"C": 3, "H": 1, "N": 1, "O": 2}
    assert m.smiles_elements("[NH3+]CC(=O)[O-]") == {"N": 1, "H": 1, "C": 2, "O": 2}
    assert m.smiles_elements("C1CCCCC1")["C"] == 6                    # ring digits


# ---- group bootstrap ----------------------------------------------------------


def _make_errs(groups: np.ndarray, delta: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(scale=0.1, size=groups.size)) + 0.05
    raw = {f"RAW_s{s}": base + rng.normal(scale=0.001, size=groups.size) for s in (0, 1)}
    d = {f"DICT_s{s}": base + delta + rng.normal(scale=0.001, size=groups.size) for s in (0, 1)}
    return {**raw, **d}


def test_group_bootstrap_deterministic_and_sign_correct():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(50), 8)
    errs = _make_errs(groups, delta=+0.02)  # DICT worse -> positive delta
    b1 = m.paired_group_bootstrap(errs, groups, n_draws=200, seed=7)
    b2 = m.paired_group_bootstrap(errs, groups, n_draws=200, seed=7)
    assert b1 == b2
    assert b1["ci95"][0] > 0
    # flip the sign of the effect -> negative direction
    errs_flip = _make_errs(groups, delta=-0.02)
    b3 = m.paired_group_bootstrap(errs_flip, groups, n_draws=200, seed=7)
    assert b3["ci95"][1] < 0
    assert all(d > 0 for d in b1["directions_by_seed"])
    assert all(d < 0 for d in b3["directions_by_seed"])


def test_group_bootstrap_remaps_sparse_codes():
    # subgroup-restricted code arrays carry gaps; remapping must reproduce
    # exactly the result of the equivalent dense relabeling
    rng = np.random.default_rng(6)
    e = rng.normal(size=30)
    sparse = np.array([0, 0, 5, 5, 9, 9] * 5)
    dense = np.searchsorted(np.unique(sparse), sparse)
    errs = {f"RAW_s{s}": e for s in (0, 1)}
    errs.update({f"DICT_s{s}": e + 0.01 for s in (0, 1)})
    a = m.paired_group_bootstrap(errs, sparse, n_draws=100, seed=3)
    b = m.paired_group_bootstrap(errs, dense, n_draws=100, seed=3)
    assert a == b


def test_subgroup_contrast_bootstrap():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(40), 5)
    e = np.abs(rng.normal(scale=0.1, size=groups.size)) + 0.05
    mask = np.zeros(groups.size, bool)
    mask[groups < 10] = True
    e[mask] += 0.05
    out = m.subgroup_contrast_bootstrap(e, mask, groups, n_draws=200, seed=11)
    assert out["n_subgroup"] == 50 and out["n_complement"] == 150
    assert out["point"] > 0
