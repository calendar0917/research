"""High-risk tests for the LCRC-A dissection implementation.

Model-level tests are self-contained (synthetic batch, no data / no GPU).
Data-level tests read the vendored ChemBench bytes and are marked ``slow``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK / "code"))

import dissect_common as C  # noqa: E402
import dissect_model as M  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic batch
# ---------------------------------------------------------------------------


def _norm():
    g = np.linspace(0.0, 1.0, C.GLOBAL_DIM, dtype=np.float32)
    s = np.linspace(0.0, 1.0, C.ATOM_SCALAR_DIM, dtype=np.float32)
    return {
        "global_mean": g,
        "global_scale": np.ones(C.GLOBAL_DIM, dtype=np.float32) + 0.5,
        "scalar_mean": s,
        "scalar_scale": np.ones(C.ATOM_SCALAR_DIM, dtype=np.float32) + 0.25,
        "source": "synthetic",
        "valid_or_test_used": False,
    }


def make_batch(seed: int = 0, n_paths_per_root: int = 3) -> dict[str, torch.Tensor]:
    """Two molecules: a 4-atom chain (with a multi-path root) and a lone atom."""
    rng = np.random.RandomState(seed)
    atom_types = np.asarray([2, 3, 7, 2], dtype=np.int64)  # C, C_aromatic, N, C
    n_atoms = 4
    bond_index = np.asarray([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    atom_batch = np.asarray([0, 0, 0, 1], dtype=np.int64)
    # S1 incidence: (root, neighbour) for every directed incidence
    nbr_atom = np.asarray([0, 1, 1, 1, 2, 3], dtype=np.int64)
    nbr_code = np.asarray([3 * 5 + 1, 2 * 5 + 1, 7 * 5 + 2, 2 * 5 + 1, 3 * 5 + 2, 2 * 5 + 1], dtype=np.int64)
    # S2 records: root 1 has n_paths_per_root occurrences
    path_atom = np.repeat(np.asarray([1], dtype=np.int64), n_paths_per_root)
    # path_cat % 5 == path_end63 % 5 (zero-based bt2), with a repeated bt2 so
    # that the B5 (root, bt2) grouping is non-degenerate for n_paths_per_root>=2
    cats = np.asarray([80, 85, 192, 81, 87])  # bt2 = [0, 0, 2, 1, 2]
    ends = np.asarray([10, 35, 12, 6, 37])    # bt2 = [0, 0, 2, 1, 2]
    path_cat = cats[:n_paths_per_root].astype(np.int64)
    path_end63 = ends[:n_paths_per_root].astype(np.int64)
    path_end = np.asarray([2, 3, 0, 2, 3])[:n_paths_per_root].astype(np.int64)
    path_closure = np.asarray([1, 0, 0, 1, 0])[:n_paths_per_root].astype(np.int64)
    return {
        "n_graphs": 2,
        "atom_feat": torch.as_tensor(rng.randn(n_atoms, C.ATOM_FEAT_DIM).astype(np.float32)),
        "atom_type": torch.as_tensor(atom_types),
        "atom_scalar": torch.as_tensor(rng.randn(n_atoms, C.ATOM_SCALAR_DIM).astype(np.float32)),
        "atom_batch": torch.as_tensor(atom_batch),
        "atom_off": torch.as_tensor([0, 3], dtype=torch.int64),
        "bond_feat": torch.as_tensor(rng.randn(3, C.BOND_FEAT_DIM).astype(np.float32)),
        "bond_type": torch.as_tensor([1, 2, 1], dtype=torch.int64),
        "bond_index": torch.as_tensor(bond_index),
        "bond_batch": torch.as_tensor([0, 0, 0], dtype=torch.int64),
        "nbr_atom": torch.as_tensor(nbr_atom),
        "nbr_code": torch.as_tensor(nbr_code),
        "path_atom": torch.as_tensor(path_atom),
        "path_cat": torch.as_tensor(path_cat),
        "path_end63": torch.as_tensor(path_end63),
        "path_end": torch.as_tensor(path_end),
        "path_closure": torch.as_tensor(path_closure),
        "path_graph": torch.as_tensor(np.zeros(n_paths_per_root, dtype=np.int64)),
        "global": torch.as_tensor(rng.randn(2, C.GLOBAL_DIM).astype(np.float32)),
        "atom_counts": torch.as_tensor([3, 1], dtype=torch.int64),
        "bond_counts": torch.as_tensor([3, 0], dtype=torch.int64),
    }


@pytest.fixture(scope="module")
def spec():
    import dissect_data as D

    return D.frozen_spec()


@pytest.fixture(scope="module")
def state(spec):
    return M.build_full_state(spec, _norm(), 0)


# ---------------------------------------------------------------------------
# 1. parameter counts and column mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", C.VARIANT_ORDER)
def test_parameter_counts_and_shared_init(spec, state, variant):
    model = M.DissectLCRC(spec, _norm(), variant)
    model.load_state_dict(M.remap_state(variant, state), strict=True)
    assert M.parameter_count(model) == C.EXPECTED_PARAMS[variant]
    shared = M.shared_state_equal(variant, state, model.state_dict())
    assert shared["all_equal"], shared["checks"]


def test_removed_modules_are_absent(spec, state):
    expected = {
        "A1": {"local", "f", "atom_encoder", "bond_encoder"},
        "A5": {"rho"},
        "B0": {"local.path_emb", "local.end_emb", "local.path_u", "local.path_v", "local.joint_proj",
               "local.close_emb", "local.path_gate"},
        "B1": {"local.path_u", "local.path_v", "local.joint_proj"},
        "B2": {"local.path_gate"},
        "B3": {"local.path_u", "local.path_v", "local.joint_proj", "local.path_gate"},
        "B6": {"local.close_emb"},
        "C0": {"local.nbr_emb"},
    }
    for variant, absent in expected.items():
        model = M.DissectLCRC(spec, _norm(), variant)
        names = {n for n, _ in model.named_modules()}
        assert not (absent & names), (variant, absent & names)


def test_local_column_layout_is_frozen():
    cols, rho = C.variant_cols("A0")
    assert list(cols) == list(range(529))
    assert list(rho) == list(range(171))
    assert len(C.variant_cols("B0")[0]) == 329
    assert len(C.variant_cols("B1")[0]) == 497
    assert len(C.variant_cols("B2")[0]) == 433
    assert len(C.variant_cols("B3")[0]) == 401
    assert len(C.variant_cols("B6")[0]) == 520
    assert len(C.variant_cols("C0")[0]) == 497
    assert len(C.variant_cols("C1")[0]) == 433
    assert len(C.variant_cols("C2")[0]) == 520
    assert len(C.variant_cols("C3")[1]) == 107
    assert len(C.variant_cols("A1")[1]) == 42
    assert len(C.variant_cols("A2")[1]) == 129


# ---------------------------------------------------------------------------
# 2. FULL equivalence with the frozen upstream model
# ---------------------------------------------------------------------------


def test_full_init_matches_frozen_upstream(spec):
    import mb_model as MM

    torch.manual_seed(0)
    old = MM.PREP_MODEL.LCRC(spec, _norm(), MM.PREP_MODEL.DEFAULT_CONFIG)
    old_state = old.state_dict()
    new_state = M.build_full_state(spec, _norm(), 0)
    assert set(old_state) == set(new_state)
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key]), key


def test_full_eval_and_train_equivalence(spec, state):
    import mb_model as MM

    batch = make_batch()
    old = MM.PREP_MODEL.LCRC(spec, _norm(), MM.PREP_MODEL.DEFAULT_CONFIG)
    MM.DropoutAtLocal(old, 0.10)
    old.load_state_dict({k: v.clone() for k, v in state.items()})
    new = M.DissectLCRC(spec, _norm(), "A0")
    new.load_state_dict({k: v.clone() for k, v in state.items()})

    old.eval()
    new.eval()
    with torch.no_grad():
        yo, ao = old(batch, return_aux=True)
        yn, an = new(batch, return_aux=True)
    assert torch.equal(yo, yn)
    for key in ("contrib", "rho", "contrib_atom", "mean_h", "mean_h2", "h", "a", "b"):
        assert torch.equal(ao[key], an[key]), key

    # train mode with the same RNG consumes the same dropout mask
    old.train()
    new.train()
    target = torch.as_tensor(np.linspace(-1, 1, 2, dtype=np.float32))
    torch.manual_seed(4321)
    yo, ao = old(batch, return_aux=True)
    torch.manual_seed(4321)
    yn, an = new(batch, return_aux=True)
    assert torch.equal(ao["h"], an["h"])
    a_enc, b_enc, scal = new._encode(batch)
    h_pre, _ = new.local(batch, a_enc, b_enc, scal)
    assert not torch.equal(h_pre, an["h"]), "dropout did not change h"
    assert torch.equal(new.f(an["h"]).view(-1), an["contrib_atom"])
    loss_o = torch.nn.functional.mse_loss(yo, target)
    loss_n = torch.nn.functional.mse_loss(yn, target)
    assert torch.equal(loss_o, loss_n)
    loss_o.backward()
    loss_n.backward()
    for name, p in old.named_parameters():
        other = dict(new.named_parameters())[name]
        assert torch.equal(p.grad, other.grad), name


def test_dropout_called_once_per_forward(spec, state):
    model = M.build_variant(spec, _norm(), "A0", state, torch.device("cpu"))
    calls = {"n": 0}
    model.drop.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    batch = make_batch()
    with torch.no_grad():
        model(batch)
    assert calls["n"] == 1
    # G_ONLY has no h and therefore no dropout module
    g_only = M.build_variant(spec, _norm(), "A1", state, torch.device("cpu"))
    assert not hasattr(g_only, "drop")


# ---------------------------------------------------------------------------
# 3. marginal algebra (B4 / B5)
# ---------------------------------------------------------------------------


def _ref_joint(u, w, groups, W, b):
    out = np.zeros((len(groups), u.shape[1]))
    for g, idx in enumerate(groups):
        n = len(idx)
        if n:
            out[g] = W @ (n * u[idx].mean(0) * w[idx].mean(0)) + n * b
    return out


def test_marginal_joint_float64_cases():
    rs = np.random.RandomState(7)
    rank, groups = 3, 4
    u = rs.randn(5, rank)
    w = rs.randn(5, rank)
    W = rs.randn(rank, rank)
    b = rs.randn(rank)
    key = torch.as_tensor([1, 2, 2, 3, 3], dtype=torch.int64)
    got = M.marginal_joint_agg(
        torch.as_tensor(u, dtype=torch.float64),
        torch.as_tensor(w, dtype=torch.float64),
        key,
        groups,
        torch.as_tensor(W, dtype=torch.float64),
        torch.as_tensor(b, dtype=torch.float64),
    ).numpy()
    ref = _ref_joint(u, w, [np.asarray([], int), np.asarray([0]), np.asarray([1, 2]), np.asarray([3, 4])], W, b)
    assert np.allclose(got, ref, atol=1e-12)
    assert np.allclose(got[0], 0.0)
    assert np.allclose(got[1], W @ (u[0] * w[0]) + b)
    # duplicates collapse to the FULL sum
    dup = M.marginal_joint_agg(
        torch.as_tensor(np.repeat(u[[0]], 3, axis=0), dtype=torch.float64),
        torch.as_tensor(np.repeat(w[[0]], 3, axis=0), dtype=torch.float64),
        torch.zeros(3, dtype=torch.int64),
        1,
        torch.as_tensor(W, dtype=torch.float64),
        torch.as_tensor(b, dtype=torch.float64),
    ).numpy()
    assert np.allclose(dup[0], 3 * (W @ (u[0] * w[0]) + b))
    # distinct pairings differ from the FULL sum
    assert not np.allclose(got[3], sum((u[i] * w[i]) @ W.T + b for i in (3, 4)))


def test_marginal_content_float64_cases():
    rs = np.random.RandomState(11)
    t = rs.rand(5)
    z = rs.randn(5, 2)
    key = torch.as_tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    got = M.marginal_content_agg(
        torch.as_tensor(t, dtype=torch.float64).unsqueeze(1),
        torch.as_tensor(z, dtype=torch.float64),
        key,
        3,
    ).numpy()
    assert np.allclose(got[2], 0.0)
    assert np.allclose(got[0], 2 * t[0:2].mean() * z[0:2].mean(0))
    assert not np.allclose(got[1], (t[2:5, None] * z[2:5]).sum(0))
    single = M.marginal_content_agg(
        torch.as_tensor(t[[3]], dtype=torch.float64).unsqueeze(1),
        torch.as_tensor(z[[3]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.int64),
        1,
    ).numpy()
    assert np.allclose(single[0], t[3] * z[3])


def test_bt2_group_splits_pairings():
    u = np.asarray([[1.0, 0.0], [3.0, 0.0]])
    w = np.asarray([[2.0, 0.0], [5.0, 0.0]])
    W = np.eye(2)
    b = np.zeros(2)
    separate = _ref_joint(u, w, [np.asarray([0]), np.asarray([1])], W, b).sum(0)
    collapsed = _ref_joint(u, w, [np.asarray([0, 1])], W, b)[0]
    assert not np.allclose(separate, collapsed)
    assert np.allclose(separate, ((u * w) @ W.T + b).sum(0))


def test_b4_b5_differ_from_full_on_multipath_root(spec, state):
    batch = make_batch(n_paths_per_root=3)
    joint = slice(64, 96)
    outs = {}
    for variant in ("A0", "B4", "B5"):
        model = M.build_variant(spec, _norm(), variant, state, torch.device("cpu")).eval()
        # path_gate is initialised to 1, which makes the content marginal equal
        # to the FULL sum for n=1 and for constant gates; use a non-constant
        # deterministic gate probe to exercise the content control.
        with torch.no_grad():
            model.local.path_gate.weight.copy_(torch.linspace(-1.0, 1.0, C.N_PATH_CODES).unsqueeze(1))
            _, aux = model(batch, return_aux=True)
        outs[variant] = aux
    assert not torch.allclose(outs["A0"]["path_agg"][:, joint], outs["B4"]["path_agg"][:, joint])
    assert not torch.allclose(outs["A0"]["content_agg"], outs["B4"]["content_agg"])
    assert not torch.allclose(outs["A0"]["content_agg"], outs["B5"]["content_agg"])
    # duplicates with weight 1 are exactly the FULL sum (documented degeneracy)
    with torch.no_grad():
        straight = M.build_variant(spec, _norm(), "B4", state, torch.device("cpu")).eval()
        full = M.build_variant(spec, _norm(), "A0", state, torch.device("cpu")).eval()
        _, a4 = straight(batch, return_aux=True)
        _, a0 = full(batch, return_aux=True)
    assert torch.allclose(a0["content_agg"], a4["content_agg"], atol=1e-7)
    # with a single occurrence per root the marginal control must equal FULL
    batch1 = make_batch(n_paths_per_root=1)
    for variant in ("B4", "B5"):
        m = M.build_variant(spec, _norm(), variant, state, torch.device("cpu")).eval()
        f = M.build_variant(spec, _norm(), "A0", state, torch.device("cpu")).eval()
        with torch.no_grad():
            _, a1 = m(batch1, return_aux=True)
            _, a0 = f(batch1, return_aux=True)
        assert torch.allclose(a0["path_agg"][:, joint], a1["path_agg"][:, joint], atol=1e-6)
        assert torch.allclose(a0["content_agg"], a1["content_agg"], atol=1e-6)


def test_marginal_branches_receive_gradient(spec, state):
    batch = make_batch(n_paths_per_root=3)
    for variant in ("B4", "B5"):
        model = M.build_variant(spec, _norm(), variant, state, torch.device("cpu"))
        model.train()
        model(batch).sum().backward()
        grads = dict(model.named_parameters())
        for key in ("local.path_u.weight", "local.path_v.weight", "local.joint_proj.weight", "local.path_gate.weight"):
            assert grads[key].grad is not None and grads[key].grad.abs().max() > 0, (variant, key)


# ---------------------------------------------------------------------------
# 4. selection / denormalisation
# ---------------------------------------------------------------------------


def test_best_epoch_tie_keeps_earlier():
    from dissect_train import BestEpochTracker

    tracker = BestEpochTracker()
    for epoch, value in [(1, 0.9), (2, 0.8), (3, 0.8), (4, 0.85)]:
        tracker.consider(epoch, {"rmse": value})
    assert tracker.best_epoch == 2
    assert tracker.best_primary == 0.8


def test_denormalisation_identity():
    from dissect_train import raw_metrics

    y = np.asarray([1.0, 2.0, 3.5])
    tn = {"mean": 2.0, "scale": 1.5}
    assert abs(raw_metrics((y - 2.0) / 1.5, y, tn)["rmse"]) < 1e-12


# ---------------------------------------------------------------------------
# 5. data bytes (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_public_split_bytes_and_labels():
    import dissect_source as S

    for repeat in C.REPEATS:
        src = S.load_lipop(repeat, labels="train_valid")
        assert src["sizes"] == list(C.SPLIT_SIZES)
        assert src["split_file_sha256"] == C.SPLIT_FILE_SHA256[repeat]
        assert tuple(C.row_sha256(src["rows"][k]) for k in ("train", "valid", "test")) == C.SPLIT_ROWS_SHA256[repeat]
        assert "test" not in src["y"]


@pytest.mark.slow
def test_train_only_normalisation_and_cache_alignment():
    import dissect_data as D

    view = D.build_view(C.MAIN_REPEAT)
    assert list(np.asarray(view.train_cache["rows"]).tolist()) == view.rows["train"]
    assert view.train_norm["valid_or_test_used"] is False
    assert view.target_norm["source"] == "train_split_only"
    assert not view.train_cache["meta"]["labels_present"]
    assert view.audit()["test_structure_cache_built"] is False
