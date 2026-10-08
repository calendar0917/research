"""Focused tests for zinc_cssd_graph_information_sufficiency_v1.

Only the NEW risks of this round: the pure witness math (hand-computed values,
zero-denominator / all-zero-support / duplicate-patch edge cases), the real-vs-
sham distinguishability and the sham's marginal-preservation guarantees, node-
relabel / batch-concatenation invariance, the frozen head/parameter contracts,
the label-isolation structure of the witness path and the absence of official
valid/test data access.  No data, no checkpoints, no GPU — the frozen-checkpoint
restore/replay verification lives in the runner's ``restore-checks`` stage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_graph_information_sufficiency_v1 as m,
)

torch.set_num_threads(2)


# ---- choose3 -----------------------------------------------------------------


def test_choose3_matches_binomial():
    xs = np.arange(0, 12)
    expected = np.array([0, 0, 0, 1, 4, 10, 20, 35, 56, 84, 120, 165])
    assert np.array_equal(m.choose3(xs), expected)


# ---- hand-computed witness on a synthetic molecule ----------------------------

# synthetic molecule: 4 roots/atoms, linear chain 0-1-2-3 with radius-2 patches
#   C(0)={0,1,2}, C(1)={0,1,2,3}, C(2)={0,1,2,3}, C(3)={1,2,3}
# supports: root0 uses atom {0}, root1 uses {1}, root2 uses {2}, root3 uses {3}
OCC_NODE = np.array([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3])
OCC_ROOT = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
B = np.zeros((4, 3), dtype=np.int64)
B[0, 0] = 1
B[1, 1] = 1
B[2, 2] = 1
B[3, 0] = 1  # atom channel 0 shared by roots 0 and 3 (never co-covering one atom)


def _witness_reference(b, occ_node, occ_root, n_roots):
    n_x = np.bincount(occ_node, minlength=n_roots)
    nk = np.zeros((n_roots, b.shape[1]), dtype=np.int64)
    np.add.at(nk, occ_node, b[occ_root])
    num = m.choose3(nk).sum(axis=0).astype(np.float64)
    den = float(m.choose3(n_x).sum())
    return num / (den + m.WITNESS_EPS), den, nk


def test_witness_hand_computed():
    W, diag = m.witness_from_supports(B, OCC_NODE, OCC_ROOT, 4)
    # |C| = [3,4,4,3] -> denominator = 1 + 4 + 4 + 1 = 10
    assert diag["denominator"] == 10
    W_ref, den, nk = _witness_reference(B, OCC_NODE, OCC_ROOT, 4)
    assert den == 10
    assert np.allclose(W, W_ref)
    # channel 0 is used by roots 0 and 3; the atoms they share with others
    # never see both: hand check of n[x,0]: x0: root0 -> 1; x1: roots 0? no
    # (root0 covers 0,1,2; root1 covers 0..3; root2 covers 0..3; root3 covers
    # 1,2,3).  n[.,0] = [1 (root0), 1 (root0), 1 (root0), 1 (root3)] -> all
    # choose3(n)=0 -> W_0 = 0.
    assert W[0] == 0.0
    # channel 1 only root1: n[.,1] = [1,1,1,1] -> W_1 = 0
    assert W[1] == 0.0
    # duplicate channel-0 support on roots 1 and 3 instead -> they co-cover
    # atoms 1,2,3 with multiplicity 2 < 3 -> still 0; make a case with a real
    # triple: three roots supporting the same channel covering one atom.
    b3 = np.zeros((4, 1), dtype=np.int64)
    b3[0, 0] = b3[1, 0] = b3[2, 0] = 1  # roots 0,1,2 all cover atoms 0,1,2
    W3, _ = m.witness_from_supports(b3, OCC_NODE, OCC_ROOT, 4)
    # n[x] = [3,3,3,1] -> numerator = 1+1+1 = 3, denominator = 10
    assert np.isclose(W3[0], 3.0 / (10.0 + m.WITNESS_EPS))


# ---- edge cases ---------------------------------------------------------------


def test_witness_zero_denominator_returns_zero():
    b = np.ones((2, 5), dtype=np.int64)
    # 2 roots, every atom covered by at most 2 roots -> no triple coverage
    occ_node = np.array([0, 1, 0, 1])
    occ_root = np.array([0, 0, 1, 1])
    W, diag = m.witness_from_supports(b, occ_node, occ_root, 2)
    assert np.all(W == 0.0)
    assert diag["denominator"] == 0.0


def test_witness_all_zero_supports():
    b = np.zeros((4, 3), dtype=np.int64)
    W, diag = m.witness_from_supports(b, OCC_NODE, OCC_ROOT, 4)
    assert np.all(W == 0.0)
    assert diag["denominator"] > 0  # incidence still there; numerator zero


def test_witness_duplicate_patches_behave_symmetrically():
    # every root has the same support -> W_k = 1 for used channels, 0 else
    b = np.tile(np.array([1, 0, 1]), (4, 1)).astype(np.int64)
    W, diag = m.witness_from_supports(b, OCC_NODE, OCC_ROOT, 4)
    assert np.allclose(W[[0, 2]], 1.0)
    assert np.allclose(W[[1]], 0.0)
    # shuffling identical rows changes nothing: the sham is degenerate here
    b_shuf, fixed = m.sham_supports(b, gid=3)
    W_shuf, _ = m.witness_from_supports(b_shuf, OCC_NODE, OCC_ROOT, 4)
    assert np.allclose(W, W_shuf)


def test_witness_rejects_bad_shapes_and_indices():
    with pytest.raises(RuntimeError):
        m.witness_from_supports(np.zeros((3, 2)), OCC_NODE, OCC_ROOT, 4)
    with pytest.raises(RuntimeError):
        m.witness_from_supports(np.zeros((4, 2)), OCC_NODE, OCC_ROOT[:-1], 4)
    with pytest.raises(RuntimeError):
        m.witness_from_supports(np.zeros((4, 2)), OCC_NODE, OCC_ROOT + 10, 4)


# ---- sham: marginals preserved, binding destroyed, deterministic --------------


def test_sham_preserves_molecule_marginals_and_deterministic():
    rng = np.random.default_rng(0)
    b = (rng.random((9, 7)) < 0.3).astype(np.int64)
    b1, fixed1 = m.sham_supports(b, gid=42)
    b2, fixed2 = m.sham_supports(b, gid=42)
    assert np.array_equal(b1, b2) and fixed1 == fixed2  # fixed seed by gid
    assert np.array_equal(np.sort(b1, axis=0), np.sort(b, axis=0))  # row multiset
    assert np.array_equal(b1.sum(axis=0), b.sum(axis=0))  # per-atom-k usage counts
    assert b1.shape == b.shape


def test_real_and_sham_distinguishable_on_asymmetric_binding():
    # roots 0,1,2 all cover atom 0 and all support channel 0; roots 3..8 cover
    # other atoms and do NOT support channel 0.  The real binding puts channel-0
    # supports on the triple that shares atom 0 -> W_0 > 0; the within-molecule
    # permutation moves some channel-0 rows onto roots 3..8, breaking the triple
    # for at least one draw -> W_0 drops.  (gid chosen so the permutation is
    # non-degenerate; asserted below.)
    n = 9
    occ_node, occ_root = [], []
    for v in range(3):
        occ_node.append(0)
        occ_root.append(v)
    for v in range(3, n):
        occ_node.append(v)  # atom v covered only by root v
        occ_root.append(v)
    occ_node = np.array(occ_node)
    occ_root = np.array(occ_root)
    b = np.zeros((n, 2), dtype=np.int64)
    b[:3, 0] = 1
    W_real, _ = m.witness_from_supports(b, occ_node, occ_root, n)
    # denominator = C(3,3) + 6*C(1,3) = 1
    assert np.isclose(W_real[0], 1.0)
    # find a gid whose sham actually moves a channel-0 row (record the search,
    # do not tune to labels — this is a synthetic mechanism demo)
    moved = None
    for gid in range(1000):
        b_shuf, _fixed = m.sham_supports(b, gid)
        if not np.array_equal(b_shuf[:3, 0], np.sort(b[:3, 0])) or not b_shuf[:3, 0].all():
            moved = gid
            break
    assert moved is not None, "synthetic sham never moved the binding rows"
    b_shuf, _ = m.sham_supports(b, moved)
    W_shuf, _ = m.witness_from_supports(b_shuf, occ_node, occ_root, n)
    assert W_shuf[0] < W_real[0]  # binding broken -> witness drops
    # marginals still preserved
    assert np.array_equal(b_shuf.sum(axis=0), b.sum(axis=0))


# ---- invariances ---------------------------------------------------------------


def test_relabel_invariance_of_witness():
    rng = np.random.default_rng(5)
    n = 6
    phi = rng.standard_normal((n, 65))
    atom = rng.integers(0, 28, size=n)
    occ_node = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5])
    occ_root = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    occ_shell = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    rem = m.remap_molecule(phi, atom, occ_node, occ_root, occ_shell, perm)
    # every node-addressed field is remapped consistently: new index of old v
    # is perm[v]; new_phi[perm[v]] = phi[v]; occurrences move with their nodes
    assert np.array_equal(rem["occ_node"], perm[occ_node])
    assert np.array_equal(rem["occ_root"], perm[occ_root])
    assert np.array_equal(rem["phi"], phi[inv])
    assert np.array_equal(rem["atom"], atom[inv])
    # witness-level invariance with the support rows relabelled consistently:
    # alpha follows the phi rows (alpha_new = alpha[inv]), so the relabelled
    # witness must be identical
    b = (rng.random((n, 4)) < 0.4).astype(np.int64)
    W1, d1 = m.witness_from_supports(b, occ_node, occ_root, n)
    W2, d2 = m.witness_from_supports(b[inv], rem["occ_node"], rem["occ_root"], n)
    assert np.allclose(W1, W2, atol=0)
    assert d1["denominator"] == d2["denominator"]


def test_batch_concatenation_and_order_invariance():
    """Per-molecule witness is unchanged when molecules are concatenated with
    row offsets (the batching situation), in any order."""
    rng = np.random.default_rng(11)
    mols = []
    for _ in range(3):
        n = int(rng.integers(4, 8))
        phi = rng.standard_normal((n, 65))
        occ_node = np.concatenate([np.arange(n), np.arange(n - 1)])
        occ_root = np.concatenate([np.arange(n), np.arange(n - 1)])
        b = (rng.random((n, 5)) < 0.4).astype(np.int64)
        mols.append((phi, occ_node.copy(), occ_root.copy(), b))
    singles = [m.witness_from_supports(b, on, orr, len(b))[0] for (phi, on, orr, b) in mols]
    for order in [(0, 1, 2), (2, 0, 1)]:
        offset = 0
        all_b, all_node, all_root = [], [], []
        for j in order:
            phi, on, orr, b = mols[j]
            all_b.append(b)
            all_node.append(on + offset)
            all_root.append(orr + offset)
            offset += len(b)
        B = np.concatenate(all_b, axis=0)
        NODE = np.concatenate(all_node)
        ROOT = np.concatenate(all_root)
        for pos, j in enumerate(order):
            lo = sum(len(mols[k][3]) for k in order[:pos])
            hi = lo + len(mols[j][3])
            W_single = singles[j]
            # recompute the part from the concatenated arrays restricted to the
            # molecule's own occurrences (incidence never crosses molecules)
            sel = (NODE >= lo) & (NODE < hi) & (ROOT >= lo) & (ROOT < hi)
            W_part2, _ = m.witness_from_supports(
                B[lo:hi], NODE[sel] - lo, ROOT[sel] - lo, hi - lo
            )
            assert np.allclose(W_part2, W_single, atol=0)


# ---- frozen head / parameter contracts -----------------------------------------


def test_head_parameter_counts_frozen():
    assert m.head_param_count(814) == 10_791
    assert m.head_param_count(846) == 11_207
    head = m.build_head(846)
    assert isinstance(head, torch.nn.Sequential)
    assert head[0].in_features == 846 and head[0].out_features == 13
    assert head[-1].in_features == 13 and head[-1].out_features == 1


def test_head_init_reproducible_and_shared_construction():
    a = m.build_head(846)
    b = m.build_head(846)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


# ---- grouped folds / selection splits ------------------------------------------


def test_grouped_folds_are_group_disjoint_and_deterministic():
    smiles = np.array(["A", "A", "B", "C", "C", "D", "E", "E"])
    f1 = m.grouped_folds(smiles, n_folds=3, seed=7)
    f2 = m.grouped_folds(smiles, n_folds=3, seed=7)
    assert np.array_equal(f1, f2)
    for g in set(smiles.tolist()):
        assert len(set(f1[smiles == g].tolist())) == 1  # group never straddles folds
    assert len(set(f1.tolist())) == 3


def test_selection_split_is_grouped_and_roughly_20pct():
    rng = np.random.default_rng(3)
    groups = np.array([f"g{i // 3}" for i in range(300)])
    rows = np.arange(300)
    sel, fit = m._selection_split(rows, groups, seed=1)
    sel_groups = set(groups[sel].tolist())
    fit_groups = set(groups[fit].tolist())
    assert not (sel_groups & fit_groups)  # grouped carve-out
    assert abs(len(sel) / 300 - 0.2) < 0.05


def test_group_paired_bootstrap_shared_picks_and_reasonable_ci():
    rng = np.random.default_rng(0)
    n = 400
    groups = np.array([f"g{i // 4}" for i in range(n)])
    diff = rng.standard_normal((n, 2)) * 0.01 + 0.002
    out = m.group_paired_bootstrap(diff, groups, n_boot=200, seed=5)
    assert out["n_groups"] == 100
    # point estimate matches the plain mean; the bootstrap mean is close to it
    assert abs(out["point_per_series"][0] - diff[:, 0].mean()) < 1e-12
    assert abs(out["bootstrap_mean"] - diff.mean()) < 0.005
    assert out["ci95_low"] <= diff.mean() <= out["ci95_high"]


# ---- label isolation / official data hygiene (static structural checks) --------


def test_witness_path_has_no_label_inputs():
    """The witness functions' signatures cannot receive targets."""
    import inspect

    for fn in (m.molecule_witness, m.witness_from_supports, m.sham_supports):
        params = inspect.signature(fn).parameters
        assert not ({"y", "g", "target", "targets", "label", "labels"} & set(params))


def test_module_never_accesses_official_valid_or_test_loaders():
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
            lowered = node.attr.lower()
            if ("valid" in lowered or "test" in lowered) and "loaded" not in lowered and "test_access" not in lowered:
                forbidden.append(node.attr)
    assert not forbidden, f"official valid/test access attempted: {sorted(set(forbidden))}"


def test_frozen_thresholds_match_the_preregistration():
    assert m.GO_DG_R == 0.0010
    assert m.GO_DG_SHAM == 0.0005
    assert m.DECISION_SCALE == 0.003
    assert m.MECHANISM_GATE_L1 == 0.1
    assert m.MECHANISM_GATE_FRACTION == 0.95
    assert m.TOP_SHARE_GATE == 0.50
    assert m.SHUFFLE_SEED_BASE == 20261021
    assert m.SEEDS == (0, 1)
    assert m.ARMS == ("R_only", "R_W_shuffled", "R_W_real")
    assert m.R_DIM == 814
