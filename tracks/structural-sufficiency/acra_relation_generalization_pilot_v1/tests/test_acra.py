"""Engineering gates for the ACRA relation-generalization pilot.

The tests exercise the real cache / model path.  No Test label or prediction is
loaded anywhere.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import acra_common as C
import acra_cache as K
import acra_data as D
import acra_model as M
import acra_split as S
import acra_support_gate as SG

PD = C.PD


@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def raw(cfg) -> dict:
    return D.load_raw(cfg)


@pytest.fixture(scope="module")
def pack(cfg, raw) -> dict:
    return K.build_pack(cfg, raw, rows=list(range(24)), log=lambda _m: None)


@pytest.fixture(scope="module")
def dims():
    return C.official_feature_dims()


def model_for(cfg, pack, dims, arm):
    model = M.ACRAModel(cfg, dims[0], dims[1], arm)
    model.load_pack(pack)
    model.eval()
    return model


def direct_relation(model, batch, out, categories=None, gamma=None):
    import torch

    if int(batch["n_pairs"]) == 0:
        return torch.zeros(int(batch["n_molecules"]), device=model.beta.device)
    qe, ke = out["rel_qe"], out["rel_ke"]
    pe, pf = batch["pair_e"], batch["pair_f"]
    dots = 0.5 * ((qe[pe] * ke[pf]).sum(-1) + (qe[pf] * ke[pe]).sum(-1))
    cat = (out["rel_categories"] if categories is None
           else torch.as_tensor(categories, dtype=torch.long, device=qe.device))
    pv = dots / math.sqrt(model.rel_rank)
    if hasattr(model, "rel_bias"):
        pv = pv + model.rel_bias.weight.squeeze(-1)[cat]
    if gamma is not None:
        pv = pv + (torch.as_tensor(gamma, dtype=qe.dtype, device=qe.device)[cat]
                   - 1.0) * dots / math.sqrt(model.rel_rank)
    total = torch.zeros(int(batch["n_molecules"]), device=qe.device).index_add_(
        0, batch["pair_mol"], pv)
    return torch.sigmoid(model.gate_raw) * total / batch["n_bonds"].float().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Gates 1-5: relation algebra and coupling
# ---------------------------------------------------------------------------


def test_relation_algebra_exact(cfg, pack, dims):
    """Direct old pair loop == moment + histogram; counts sum to m(m-1)/2."""
    import torch

    model = model_for(cfg, pack, dims, "ACRA_REL_REF")
    source = D.BatchSource(pack, "cpu")
    batch = source.make([0, 1, 2, 5, 9])
    with torch.no_grad():
        out = model.forward_batch(batch)
    direct = direct_relation(model, batch, out)
    assert torch.allclose(direct, out["e_rel"], atol=1e-6, rtol=0)
    assert torch.equal(out["e_rel_moment"] + out["e_rel_histogram"], out["e_rel"])
    m = batch["n_bonds"].long()
    expected = m * (m - 1) // 2
    assert torch.equal(out["relation_counts"].sum(1), expected)
    # exact decomposition of the batch relation sum
    counts = out["relation_counts"]
    assert torch.allclose(counts.sum(1).float(), expected.float())


def test_moment_and_histogram_parts(cfg, pack, dims):
    import torch

    full = model_for(cfg, pack, dims, "ACRA_REL_REF")
    moment = model_for(cfg, pack, dims, "ACRA_MOMENT_ONLY")
    hist = model_for(cfg, pack, dims, "ACRA_HIST_ONLY")
    # copy the shared relation state where the modules exist
    full_state = full.state_dict()
    moment_state = {k: v for k, v in full_state.items() if k in moment.state_dict()
                    and k != "combo"}
    hist_state = {k: v for k, v in full_state.items() if k in hist.state_dict()}
    moment.load_state_dict(moment_state, strict=False)
    hist.load_state_dict(hist_state, strict=False)
    source = D.BatchSource(pack, "cpu")
    batch = source.make([0, 1, 2, 5])
    with torch.no_grad():
        a = full.forward_batch(batch)
        b = moment.forward_batch(batch)
        c = hist.forward_batch(batch)
    assert torch.allclose(a["e_rel_moment"], b["e_rel"], atol=1e-6, rtol=0)
    assert torch.allclose(a["e_rel_histogram"], c["e_rel"], atol=1e-6, rtol=0)


def test_gamma_one_is_exactly_old_relation(cfg, pack, dims):
    import torch

    old = model_for(cfg, pack, dims, "ACRA_REL_REF")
    new = model_for(cfg, pack, dims, "ACRA_TOPO_COUPLED")
    state = dict(old.state_dict())
    state["gamma_theta"] = torch.zeros(new.n_rel_categories)
    new.load_state_dict(state, strict=True)
    assert torch.equal(new.gamma(), torch.ones(new.n_rel_categories))
    source = D.BatchSource(pack, "cpu")
    batch = source.make([0, 1, 2, 5])
    with torch.no_grad():
        a = old.forward_batch(batch)
        b = new.forward_batch(batch)
    assert torch.allclose(a["e_rel"], b["e_rel"], atol=1e-7, rtol=0)
    assert torch.allclose(a["yhat_std"], b["yhat_std"], atol=1e-7, rtol=0)


def test_gamma_couples_categories_to_edge_pairs(cfg, pack, dims):
    import torch

    model = model_for(cfg, pack, dims, "ACRA_TOPO_COUPLED")
    with torch.no_grad():
        model.gamma_theta.copy_(torch.linspace(-1.0, 1.0, model.n_rel_categories))
    source = D.BatchSource(pack, "cpu")
    batch = source.make([0, 1, 2, 5, 9])
    with torch.no_grad():
        out = model.forward_batch(batch)
    cat = out["rel_categories"].cpu().numpy()
    if np.unique(cat).size < 2:
        pytest.skip("sample has only one relation category")
    rng = np.random.default_rng(3)
    perm = rng.permutation(cat)
    base = direct_relation(model, batch, out)
    changed = direct_relation(model, batch, out, categories=perm,
                              gamma=model.gamma().detach().cpu().numpy())
    assert float((base - changed).abs().max()) > 1e-6
    # old REL is invariant to the same category permutation
    old = model_for(cfg, pack, dims, "ACRA_REL_REF")
    with torch.no_grad():
        o = old.forward_batch(batch)
    old_base = direct_relation(old, batch, o)
    old_changed = direct_relation(old, batch, o, categories=perm)
    assert float((old_base - old_changed).abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# Gates 6-7: invariance and batch equivalence
# ---------------------------------------------------------------------------


def test_atom_renumbering_invariance(cfg, raw, dims):
    row = int(raw["train_idx"][3])
    mol = D.raw_molecule(raw, row)
    rng = np.random.default_rng(7)
    permuted = mol.permute(rng.permutation(mol.n_atoms))
    colors = PD.ColorSpace.from_raw(raw)
    pack_a = K.pack_from_molecules(cfg, [mol], ["a"], colors, "x", [row], log=lambda _m: None)
    pack_b = K.pack_from_molecules(cfg, [permuted], ["b"], colors, "x", [row],
                                   log=lambda _m: None)
    for arm in M.ARMS:
        a, b = model_for(cfg, pack_a, dims, arm), model_for(cfg, pack_b, dims, arm)
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            ra = M.predict_rows(a, D.BatchSource(pack_a, "cpu"), [0], "cpu")
            rb = M.predict_rows(b, D.BatchSource(pack_b, "cpu"), [0], "cpu")
        for key in ("yhat_std", "e_atom", "e_bond", "e_rel"):
            assert np.allclose(ra[key], rb[key], atol=1e-6, rtol=0), (arm, key)


def test_batch_equals_single(cfg, pack, dims):
    source = D.BatchSource(pack, "cpu")
    rows = [1, 5, 9, 13]
    for arm in M.ARMS:
        model = model_for(cfg, pack, dims, arm)
        batched = M.predict_rows(model, source, rows, "cpu", batch_size=128)
        for i, row in enumerate(rows):
            one = M.predict_rows(model, source, [row], "cpu", batch_size=1)
            for key in ("yhat_std", "e_rel", "e_bond"):
                assert np.allclose(batched[key][i], one[key][0], atol=1e-6), (arm, key)


# ---------------------------------------------------------------------------
# Gate 8: support gate leave-one-molecule-out
# ---------------------------------------------------------------------------


def test_support_gate_loo(cfg, pack):
    payload = S.load(cfg)
    present = set(int(x) for x in pack["row_ids"].tolist())
    train = [r for r in payload["train_sub_rows"] if int(r) in present]
    dev = [r for r in payload["dev_rows"] if int(r) in present]
    sg = cfg["composition"]["support_gate"]
    built = SG.build(pack, train, dev, floor=sg["rho_floor"], span=sg["rho_span"],
                     offset=sg["support_offset"])
    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    combo = np.asarray(pack["ev_combo"], np.int64)
    support = built["support_by_event"]
    # manual distinct-molecule counts from the train rows only
    counts = np.zeros(int(np.asarray(pack["combo_s"]).size), np.int64)
    for row in train:
        ids = np.unique(combo[ev_ptr[row]:ev_ptr[row + 1]])
        counts[ids] += 1
    for row in train:
        ids = combo[ev_ptr[row]:ev_ptr[row + 1]]
        assert np.all(support[ev_ptr[row]:ev_ptr[row + 1]] == counts[ids] - 1)
    for row in dev:
        ids = combo[ev_ptr[row]:ev_ptr[row + 1]]
        assert np.all(support[ev_ptr[row]:ev_ptr[row + 1]] == counts[ids])
    # a repeated combination inside one molecule still subtracts exactly one
    repeated = None
    for row in train:
        ids = combo[ev_ptr[row]:ev_ptr[row + 1]]
        if ids.size > 1 and np.unique(ids, return_counts=True)[1].max() > 1:
            repeated = row
            break
    if repeated is not None:
        ids = combo[ev_ptr[repeated]:ev_ptr[repeated + 1]]
        s = support[ev_ptr[repeated]:ev_ptr[repeated + 1]]
        for cid in np.unique(ids):
            assert np.unique(s[ids == cid]).size == 1
            assert int(s[ids == cid][0]) == int(counts[cid] - 1)


# ---------------------------------------------------------------------------
# Gates 9-10: strict label / split isolation
# ---------------------------------------------------------------------------


def test_cache_is_label_free_and_split_ignores_labels(cfg, raw):
    rows = list(range(12))
    poisoned = dict(raw)
    poisoned["labels"] = np.full_like(np.asarray(raw["labels"]), np.nan)
    a = K.build_pack(cfg, raw, rows=rows, log=lambda _m: None)
    b = K.build_pack(cfg, poisoned, rows=rows, log=lambda _m: None)
    for key in a:
        if key == "meta_json":
            continue
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    cache_src = (CODE_DIR / "acra_cache.py").read_text(encoding="utf-8")
    split_src = (CODE_DIR / "acra_split.py").read_text(encoding="utf-8")
    train_src = (CODE_DIR / "acra_train.py").read_text(encoding="utf-8")
    assert 'raw["labels"]' not in cache_src
    assert "labels" not in split_src.replace("labels_read", "").replace("No label", "")
    assert 'splits["test"]' not in train_src and "'test'" not in train_src
    # scaler is fitted on Train-sub only
    labels = np.asarray(raw["labels"], np.float64)
    payload = S.load(cfg)
    train_sub = np.asarray(payload["train_sub_rows"], np.int64)
    base = C.target_scaler(labels, train_sub)
    poisoned_labels = labels.copy()
    poisoned_labels[np.asarray(raw["valid_idx"], np.int64)] = np.nan
    poisoned_labels[np.asarray(raw["test_idx"], np.int64)] = np.inf
    assert C.target_scaler(poisoned_labels, train_sub) == base


def test_frozen_split_is_label_free_and_disjoint(cfg):
    payload = S.load(cfg)
    assert payload["label_free"] is True and payload["labels_read"] is False
    tr = set(int(r) for r in payload["train_sub_rows"])
    dv = set(int(r) for r in payload["dev_rows"])
    assert not tr & dv
    scaffold = payload["scaffold_by_row"]
    assert {scaffold[str(r)] for r in tr} & {scaffold[str(r)] for r in dv} == set()


# ---------------------------------------------------------------------------
# Gates 11-13: live branches, finiteness, determinism
# ---------------------------------------------------------------------------


def test_no_dead_trainable_branch(cfg, pack, dims):
    import torch

    source = D.BatchSource(pack, "cpu")
    for arm in M.ARMS:
        model = model_for(cfg, pack, dims, arm)
        model.train()
        y = torch.sin(torch.arange(8, dtype=torch.float32))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(4):
            batch = source.make(list(range(8)))
            out = model.forward_batch(batch)
            terms = model.loss(out, batch, y)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            assert torch.isfinite(terms["loss"])
            opt.step()
        batch = source.make(list(range(8)))
        out = model.forward_batch(batch)
        terms = model.loss(out, batch, y)
        model.zero_grad(set_to_none=True)
        terms["loss"].backward()
        dead = [name for name, p in model.named_parameters()
                if p.requires_grad and (p.grad is None
                                        or not torch.isfinite(p.grad).all()
                                        or float(p.grad.abs().sum()) == 0.0)]
        assert not dead, (arm, dead)
        for key in ("yhat_std", "e_atom", "e_bond", "e_rel", "r", "interaction"):
            assert torch.isfinite(out[key]).all(), (arm, key)


def test_shared_initialisation_across_arms(cfg, dims):
    models = {arm: M.ACRAModel(cfg, dims[0], dims[1], arm) for arm in M.ARMS}
    shared = ("frame", "proj_s", "proj_c", "mlp_s", "mlp_c", "bond_emb",
              "flag_bridge", "flag_ring", "scalar_s", "scalar_c", "q", "k",
              "atom_head", "beta")
    hashes = {arm: C.module_hashes(m, list(shared)) for arm, m in models.items()}
    base = hashes[M.ARMS[0]]
    for arm in M.ARMS:
        assert hashes[arm] == base, arm


def test_fixed_seed_reproducibility(cfg, pack, dims):
    import torch

    C.set_deterministic(0)
    a = M.ACRAModel(cfg, dims[0], dims[1], "ACRA_REL_REF")
    C.set_deterministic(0)
    b = M.ACRAModel(cfg, dims[0], dims[1], "ACRA_REL_REF")
    assert C.state_hash(a.state_dict()) == C.state_hash(b.state_dict())
    a.load_pack(pack)
    b.load_pack(pack)
    a.eval()
    b.eval()
    source = D.BatchSource(pack, "cpu")
    with torch.no_grad():
        ra = a.forward_batch(source.make([0, 1, 2]))
        rb = b.forward_batch(source.make([0, 1, 2]))
    assert torch.equal(ra["yhat_std"], rb["yhat_std"])


def test_local_objects_and_ring_events(cfg, raw):
    row = int(raw["train_idx"][0])
    mol = D.raw_molecule(raw, row)
    adj = D.adjacency(mol)
    geo = D.bond_geometry(mol, adj)
    assert len(geo["is_bridge"]) == int(mol.bond_u.size)
    # every real bond has two directed events with the correct roots
    for e in range(int(mol.bond_u.size)):
        for direction in (0, 1):
            (root_s, s_atoms), (root_c, c_atoms) = D.event_balls(mol, adj, e, direction)
            u, v = int(mol.bond_u[e]), int(mol.bond_v[e])
            assert root_s == (u if direction == 0 else v)
            assert root_c == (v if direction == 0 else u)
            assert s_atoms and c_atoms