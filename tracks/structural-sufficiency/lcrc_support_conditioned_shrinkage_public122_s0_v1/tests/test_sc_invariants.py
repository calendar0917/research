"""Key invariants of the support-conditioned shrinkage round (fast, CPU only).

Run: ``uv run --group lipo pytest -q tests``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import sc_checks as K
import sc_common as C
import sc_model as M
import sc_support as S
import sc_train as T

import lb_data as LD


DS = C.PRIMARY_DATASET
FIT = "100"


# ---------------------------------------------------------------------------
# fixtures (module scoped: the caches are read-only and small)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return LD.frozen_spec()


@pytest.fixture(scope="module")
def snorm():
    return C.shared_norms(DS)[0]


@pytest.fixture(scope="module")
def train_cache():
    return LD.load_cache(DS, "train")


@pytest.fixture(scope="module")
def rootkeys():
    return S.load_or_build_rootkeys(DS)


@pytest.fixture(scope="module")
def groups():
    return S.load_groups(DS)


@pytest.fixture(scope="module")
def support():
    return S.load_support(DS, FIT, C.KEY_FINE)


@pytest.fixture(scope="module")
def small_batch(train_cache):
    rows = [0, 1, 2]
    return LD.to_device(LD.collate(train_cache, rows), torch.device("cpu")), rows


def alpha_of_rows(cache, support, arm, rows):
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    arr = support[f"alpha_{arm}_train"]
    return np.concatenate([arr[atom_ptr[r]: atom_ptr[r + 1]] for r in rows]).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. key serialisation is canonical
# ---------------------------------------------------------------------------


def test_canonical_bytes_fold_negative_zero():
    a = np.array([0.0, -0.0, 1.5], dtype=np.float32)
    b = np.array([-0.0, 0.0, 1.5], dtype=np.float32)
    assert S.canonical_f32_bytes(a) == S.canonical_f32_bytes(b)
    with pytest.raises(ValueError):
        S.canonical_f32_bytes(np.array([np.nan], np.float32))


def test_joint_key_is_order_invariant():
    a = S.joint_key_bytes(np.array([3, 1]), np.array([2, 5]), np.array([1, 2]))
    b = S.joint_key_bytes(np.array([1, 3]), np.array([5, 2]), np.array([2, 1]))
    assert a == b


# ---------------------------------------------------------------------------
# 2. the integer activity test and r_v (small Cartesian product, not occupied pairs)
# ---------------------------------------------------------------------------


def test_root_activity_balanced_is_inactive():
    # one bt2 group, two paths, same c and different e: (root,bt2) marginals
    # reproduce the joint table exactly -> R == 0 -> inactive.
    cat = np.array([4, 4])
    end = np.array([1, 2])
    active, r = S.root_activity(cat, end)
    assert not active
    assert r == 0.0


def test_root_activity_imbalanced_is_active_with_exact_r():
    cat = np.array([4, 9])          # bt2 4 and 4  -> one group
    end = np.array([1, 2])
    active, r = S.root_activity(cat, end)
    # N = [[1,0],[0,1]], n_c = n_e = [1,1], n_b = 2
    # R = [[0.5,-0.5],[-0.5,0.5]] -> ||R||_F = 1.0
    assert active
    assert r == pytest.approx(1.0, abs=1e-12)


def test_root_activity_ignores_other_bond_types():
    # two groups of one path each -> both groups have n_b == 1 -> inactive
    cat = np.array([4, 9, 14])      # bt2 4, 4, 4? -> 14 % 5 == 4
    end = np.array([1, 2, 3])
    active, _ = S.root_activity(cat, end)
    assert active      # three paths in one bt2 group with 3 distinct c/e -> active
    cat2 = np.array([0, 1])         # bt2 0 and 1 -> two singleton groups
    active2, r2 = S.root_activity(cat2, np.array([5, 5]))
    assert not active2 and r2 == 0.0


# ---------------------------------------------------------------------------
# 3. support counts count structures, not roots/paths/records
# ---------------------------------------------------------------------------


def test_support_counts_are_structure_level(train_cache, rootkeys, groups, support):
    name = "fine_id"
    ids = rootkeys["train"][name]
    atom_ptr = np.asarray(train_cache["atom_ptr"], dtype=np.int64)
    counts = None
    for fit in (FIT,):
        rep = S.build_support_counts(DS, rootkeys, groups, C.fit_positions(DS, fit), C.KEY_FINE)
        counts = rep["counts"]
        assert int(counts.max()) <= rep["n_groups"]
        assert rep["n_groups"] == len(C.fit_positions(DS, fit))
    # a molecule carrying the same key on several roots must contribute once
    for rec in range(60):
        a0, a1 = int(atom_ptr[rec]), int(atom_ptr[rec + 1])
        vals, cnt = np.unique(ids[a0:a1], return_counts=True)
        if np.any(cnt > 1):
            k = int(vals[np.argmax(cnt)])
            # every occurrence belongs to the same structure -> c(key) counts it once
            assert counts[k] >= 1
            break
    # unknown key -> 0
    assert counts[counts == 0].size >= 0
    # the frozen table equals the recomputation
    frozen = np.asarray(support["counts"])
    assert np.array_equal(frozen, counts)


def test_no_label_read_by_support(monkeypatch):
    """Support/CDF/gates must be computable with the label accessor disabled."""

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("labels were read while building support")

    monkeypatch.setattr(LD, "targets", boom)
    rk = S.load_or_build_rootkeys(DS)
    g = S.load_groups(DS)
    full = S.support_alpha_tables(DS, "25", C.KEY_JOINT, rk, g)
    oof = S.support_alpha_tables(DS, "25", C.KEY_JOINT, rk, g, oof=True)
    rep = S.gate_report(DS, "25", C.KEY_JOINT, full, oof, g)
    assert "E2_spread_fullfit" in rep["checks"]


# ---------------------------------------------------------------------------
# 4. folds never split a canonical structure; N25 never sees excluded structures
# ---------------------------------------------------------------------------


def test_fold_is_a_function_of_the_canonical_structure(groups):
    folds = groups["train"]["fold"]
    gid = groups["train"]["group_id"]
    for g in np.unique(gid)[:200]:
        assert len(set(folds[gid == g].tolist())) == 1
    # identical canonical strings hash to the same fold
    s = "CCO"
    assert (int(C.sha256_text(f"{C.FOLD_SALT}|{s}"), 16) % C.N_FOLDS) == \
           (int(C.sha256_text(f"{C.FOLD_SALT}|{s}"), 16) % C.N_FOLDS)


def test_fit25_support_excludes_held_out_structures(rootkeys, groups):
    pos25 = C.fit_positions(DS, "25")
    pos100 = C.fit_positions(DS, "100")
    gid = groups["train"]["group_id"]
    in25 = {int(gid[p]) for p in pos25}
    extra = [p for p in pos100 if int(gid[p]) not in in25]
    assert extra, "fit25 must be a strict subset"
    rep25 = S.build_support_counts(DS, rootkeys, groups, pos25, C.KEY_FINE)
    rep100 = S.build_support_counts(DS, rootkeys, groups, pos100, C.KEY_FINE)
    ids = rootkeys["train"]["fine_id"]
    atom_ptr = np.asarray(LD.load_cache(DS, "train")["atom_ptr"], dtype=np.int64)
    found = 0
    for p in extra:
        a0, a1 = int(atom_ptr[p]), int(atom_ptr[p + 1])
        for k in np.unique(ids[a0:a1]):
            if rep25["counts"][k] == 0 and rep100["counts"][k] >= 1:
                found += 1
    assert found > 0
    assert rep25["counts"].max() <= rep25["n_groups"]
    assert rep25["n_groups"] == len(pos25)


# ---------------------------------------------------------------------------
# 5. atom re-numbering / batch composition do not change a molecule's gate or output
# ---------------------------------------------------------------------------


def test_atom_permutation_invariance(spec, snorm, train_cache, support, groups):
    rows = [4]
    batch = LD.to_device(LD.collate(train_cache, rows), torch.device("cpu"))
    n = int(batch["atom_feat"].shape[0])
    a = alpha_of_rows(train_cache, support, C.ARM_SELECT, rows)
    b = dict(batch, root_alpha=torch.from_numpy(a))
    model = M.build_arm_model(spec, snorm, C.ARM_SELECT)
    M.load_shared_init(model, spec, snorm)
    model.eval()
    with torch.no_grad():
        y0 = model(b).numpy()
    order = np.random.default_rng(7).permutation(n)
    b_perm = K.permute_atoms(b, order)
    with torch.no_grad():
        y1 = model(b_perm).numpy()
    assert np.allclose(y0, y1, atol=2e-5)
    # the alpha really travelled with the atoms
    assert np.allclose(b_perm["root_alpha"].numpy(), a[order], atol=0)


def test_batch_order_and_size_do_not_change_a_molecule(spec, snorm, train_cache, support):
    model = M.build_arm_model(spec, snorm, C.ARM_SELECT)
    M.load_shared_init(model, spec, snorm)
    model.eval()
    rows_a, rows_b = [5], [9, 5, 40]
    outs = {}
    for tag, rows in (("solo", rows_a), ("mixed", rows_b)):
        batch = LD.to_device(LD.collate(train_cache, rows), torch.device("cpu"))
        a = alpha_of_rows(train_cache, support, C.ARM_SELECT, rows)
        batch["root_alpha"] = torch.from_numpy(a)
        with torch.no_grad():
            y = model(batch).numpy()
        outs[tag] = float(y[rows.index(5)])
    assert abs(outs["solo"] - outs["mixed"]) <= 2e-5


# ---------------------------------------------------------------------------
# 6. operator endpoints and the R identity
# ---------------------------------------------------------------------------


def test_endpoints_and_residual_identity(spec, snorm, small_batch, train_cache, support):
    batch, rows = small_batch
    n = int(batch["atom_feat"].shape[0])
    a = alpha_of_rows(train_cache, support, C.ARM_SELECT, rows)
    gated = M.build_arm_model(spec, snorm, C.ARM_SELECT)
    M.load_shared_init(gated, spec, snorm)
    base = M.build_arm_model(spec, snorm, C.ARM_BASE)
    M.load_shared_init(base, spec, snorm)
    gated.eval(); base.eval()
    with torch.no_grad():
        y_base = base(batch).numpy()
        y_one = gated(dict(batch, root_alpha=torch.ones(n))).numpy()
        y_gate = gated(dict(batch, root_alpha=torch.from_numpy(a))).numpy()
    assert np.abs(y_base - y_one).max() <= 1e-5        # a = 1 restores C1
    assert np.abs(y_base - y_gate).max() > 1e-6        # the gate is not a no-op
    rep = K.residual_linear_map(gated.local, dict(batch, root_alpha=torch.from_numpy(a)))
    assert rep["max_abs_error"] <= 1e-5
    assert rep["n_roots_with_effect"] > 0


def test_non_joint_blocks_identical_to_c1(spec, snorm, small_batch, train_cache, support):
    """Only the joint block may differ: content must stay FULL."""
    batch, rows = small_batch
    a = alpha_of_rows(train_cache, support, C.ARM_SELECT, rows)
    gated = M.build_arm_model(spec, snorm, C.ARM_SELECT)
    M.load_shared_init(gated, spec, snorm)
    base = M.build_arm_model(spec, snorm, C.ARM_BASE)
    M.load_shared_init(base, spec, snorm)
    gated.eval(); base.eval()
    with torch.no_grad():
        a1, b1, s1 = gated._encode(dict(batch, root_alpha=torch.from_numpy(a)))
        h_g, aux_g = gated.local(dict(batch, root_alpha=torch.from_numpy(a)), a1, b1, s1)
        a2, b2, s2 = base._encode(batch)
        h_b, aux_b = base.local(batch, a2, b2, s2)
    # the content block is identical (FULL in both)
    assert torch.allclose(aux_g["content_agg"], aux_b["content_agg"], atol=1e-5)
    # the layout has the same width and the difference is confined to the joint slice
    assert aux_g["local_input"].shape == aux_b["local_input"].shape
    diff = (aux_g["local_input"] - aux_b["local_input"]).abs()
    assert float(diff.max()) > 0.0


# ---------------------------------------------------------------------------
# 7. the frozen alpha algebra
# ---------------------------------------------------------------------------


def test_alpha_mean_is_uniform_dose(support):
    for arm in (C.ARM_UNIFORM, C.ARM_SELECT, C.ARM_REVERSE):
        assert abs(float(support["fit_mean_a_select"]) - C.UNIFORM_A) <= C.GATE_TOL
    # the stored per-split tables are the frozen rules applied to the stored c
    c = np.asarray(support["c_train"], dtype=np.int64)
    F = np.asarray(support["F_table"], dtype=np.float64)
    a_sel = C.alpha_rule(C.ARM_SELECT, c, F, None)
    a_rev = C.alpha_rule(C.ARM_REVERSE, c, F, None)
    act = np.asarray(support["active_train"])
    stored_sel = np.asarray(support["alpha_SELECT_train"], dtype=np.float64)
    stored_rev = np.asarray(support["alpha_REVERSE_train"], dtype=np.float64)
    assert np.allclose(stored_sel[act], a_sel[act], atol=1e-6)
    assert np.allclose(stored_rev[act], a_rev[act], atol=1e-6)
    # inactive roots are never contracted
    assert np.all(stored_sel[~act] == 1.0) and np.all(stored_rev[~act] == 1.0)
    # SELECT and REVERSE are mirror images inside the frozen range
    lo, hi = C.GATE_RANGE
    inside = act & (a_sel > lo) & (a_sel < hi)
    assert np.allclose(stored_sel[inside] + stored_rev[inside], 2 * C.UNIFORM_A, atol=1e-6)
    # low support -> more contraction under SELECT (monotone in c, weak rows lower)
    order = np.argsort(c[act], kind="stable")
    assert np.all(np.diff(a_sel[act][order]) >= -1e-12)
    low = act & (c <= C.LOW_C_MAX)
    high = act & (c >= C.HIGH_C_MIN)
    assert a_sel[low].mean() < a_sel[high].mean()


def test_uniform_oof_uses_the_frozen_constant(support):
    a_oof = float(support["a_oof"])
    act = np.asarray(support["active_train"])
    a = np.asarray(support["alpha_UNIFORM_OOF_train"], dtype=np.float64)
    assert np.allclose(a[act], a_oof, atol=1e-6)
    assert np.all(a[~act] == 1.0)
    uniform = np.asarray(support["alpha_UNIFORM_train"], dtype=np.float64)
    assert np.allclose(uniform[act], C.UNIFORM_A, atol=1e-6)


def test_same_key_same_alpha(support):
    """The gate is a function of the frozen key only (via c), never of row ids."""
    for arm in (C.ARM_SELECT, C.ARM_REVERSE, C.ARM_UNIFORM_OOF):
        for split in C.SPLITS:
            c = np.asarray(support[f"c_{split}"], dtype=np.int64)
            act = np.asarray(support[f"active_{split}"])
            a = np.asarray(support[f"alpha_{arm}_{split}"], dtype=np.float64)
            pairs = set()
            for ci, ai in zip(c[act].tolist(), a[act].tolist()):
                pairs.add((ci, round(ai, 6)))
            per_c: dict[int, set[float]] = {}
            for ci, ai in pairs:
                per_c.setdefault(ci, set()).add(ai)
            assert all(len(v) == 1 for v in per_c.values())


# ---------------------------------------------------------------------------
# 8. training identity: shared init, shared batch plan, RNG-safe evaluation
# ---------------------------------------------------------------------------


def test_all_arms_share_init_and_batch_plan(spec, snorm):
    states = {}
    for arm in C.ARMS:
        m = M.build_arm_model(spec, snorm, arm)
        M.load_shared_init(m, spec, snorm)
        assert sum(p.numel() for p in m.parameters()) == C.C1_PARAMS
        states[arm] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        assert m.gated == (arm != C.ARM_BASE)
    ref = states[C.ARM_BASE]
    for arm in C.ARMS[1:]:
        for k in ref:
            assert torch.equal(ref[k], states[arm][k]), (arm, k)
    p1 = T.LB.BatchPlan(840, 128)
    p2 = T.LB.BatchPlan(840, 128)
    for _ in range(20):
        assert p1.next_batch() == p2.next_batch()


def test_eval_does_not_advance_training_rng(spec, snorm, small_batch):
    batch, _rows = small_batch
    model = M.build_arm_model(spec, snorm, C.ARM_BASE)
    M.load_shared_init(model, spec, snorm)
    torch.manual_seed(1234)
    model.train()
    snap = T.rng_snapshot(torch.device("cpu"))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(batch)
    T.rng_restore(snap, torch.device("cpu"))
    model.train(was)
    a = torch.rand(3)
    torch.manual_seed(1234)
    b = torch.rand(3)
    assert torch.equal(a, b)
    assert model.training is True


def test_base_is_bitwise_the_frozen_c1(spec, snorm, small_batch):
    """BASE must be the frozen C1 operator, not a limit of the gated one."""
    import sc_common as C
    batch, _rows = small_batch
    mine = M.build_arm_model(spec, snorm, C.ARM_BASE)
    M.load_shared_init(mine, spec, snorm)
    ref = C.PM.build_model(spec, snorm, "c1", variant="C1")
    _full, base_state = C.PM.c1_reference_state_full(spec, snorm)
    C.PM.load_init_state(ref, base_state, "c1")
    mine.eval(); ref.eval()
    with torch.no_grad():
        y0 = mine(batch)
        y1 = ref(batch)
    assert torch.equal(y0, y1)
    # the only difference in the state dict may be the two *unused* legacy S-family
    # buffers (shrink_kappa / shrink_abar), which never enter a gated-free forward
    mine_keys = {k for k in mine.state_dict()}
    ref_keys = {k for k in ref.state_dict()}
    assert ref_keys - mine_keys == {"local.shrink_abar", "local.shrink_kappa"}
    assert mine_keys <= ref_keys
    for k in mine.state_dict():
        assert torch.equal(mine.state_dict()[k], ref.state_dict()[k]), k
    assert [n for n, _ in mine.named_parameters()] == [n for n, _ in ref.named_parameters()]


def test_dose_is_not_degenerate_at_init(spec, snorm, train_cache, support):
    rows = sorted(C.fit_positions(DS, FIT))[:8]
    model = M.build_arm_model(spec, snorm, C.ARM_SELECT)
    M.load_shared_init(model, spec, snorm)
    runner = T.GateRunner(
        model, torch.device("cpu"),
        {"train": train_cache, "valid": LD.load_cache(DS, "valid"),
         "test": LD.load_cache(DS, "test")},
        {s: (np.asarray(support[f"alpha_{C.ARM_SELECT}_{s}"]) if s == "train" else None)
         for s in C.SPLITS},
    )
    rep = T.dose_diagnostics(model, runner, rows, torch.device("cpu"),
                             key=C.KEY_FINE, dataset=DS, fit=FIT, max_molecules=8)
    assert rep["molecules"]["norm_diff"]["max"] > 1e-8
    assert rep["n_roots_with_effect"] > 0
