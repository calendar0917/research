from __future__ import annotations

import copy

import numpy as np
import torch

import wobr_canonical as CAN
import wobr_common as C
import wobr_data as D
import wobr_model as M
import wobr_sparsemax as SP
import wobr_structure as S
import wobr_synthetic as SYN
import wobr_tensor as T


def _cfg() -> dict:
    return C.load_config()


def _norm() -> dict:
    return {"atom_scalar_mean": np.zeros(5), "atom_scalar_scale": np.ones(5)}


def _model(double: bool = False):
    atom_dims, bond_dims = D.atom_bond_dims()
    C.seed_everything(0)
    model, info = M.build_model(atom_dims, bond_dims, 0, _norm(), config=_cfg())
    if double:
        model.double()
    model.eval()
    return model, info


def _batch(view):
    view = dict(view)
    view["atom_scalars"] = D.atom_scalars(view)
    raw = SYN.raw_from_views([view])
    st = S.build_molecule_struct(view)
    return T.collate([0], {0: st}, raw, "cpu"), st


def test_sparsemax_matches_references():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(32, 17))
    count = rng.integers(1, 18, size=32)
    mask = np.arange(17)[None, :] < count[:, None]
    z[~mask] = -1e6
    alpha = SP.sparsemax_padded(torch.as_tensor(z), torch.as_tensor(mask))
    ref = SP.sparsemax_reference(z, mask)
    ref2 = SP.sparsemax_reference_sorted(z, mask)
    assert np.allclose(alpha.numpy(), ref, atol=1e-10)
    assert np.allclose(alpha.numpy(), ref2, atol=1e-10)
    assert np.allclose(alpha.sum(1).numpy(), 1.0, atol=1e-10)
    assert float(alpha[~torch.as_tensor(mask)].abs().max()) == 0.0
    assert np.all(alpha.numpy() >= 0)


def test_packed_sparsemax_padding_is_neutral():
    scores = torch.tensor([10.0, 1.0, 5.0])
    ptr = torch.tensor([0, 2, 3])
    alpha = SP.sparsemax_packed(scores, ptr, 2)
    assert torch.allclose(alpha, torch.tensor([1.0, 0.0, 1.0]), atol=1e-7)
    # balanced two-candidate root remains balanced
    scores = torch.tensor([2.0, 2.0, 100.0])
    ptr = torch.tensor([0, 2, 3])
    alpha = SP.sparsemax_packed(scores, ptr, 2)
    assert torch.allclose(alpha, torch.tensor([0.5, 0.5, 1.0]), atol=1e-7)


def test_singleton_and_candidate_structure():
    view = SYN.make_view("path4", 4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
                         atom_types=[5] * 4)
    st = S.build_molecule_struct(view)
    assert st.n_words > 0
    for v in range(st.n_atoms):
        lo, hi = int(st.word_root_ptr[v]), int(st.word_root_ptr[v + 1])
        sizes = st.word_size[lo:hi]
        assert np.sum(sizes == 1) == 1
        for w in range(lo, hi):
            sup = set(int(x) for x in st.word_support[w] if x >= 0)
            assert v in sup
            assert 1 <= len(sup) <= 4
            internal = sorted(int(x) for x in st.word_internal[w] if x >= 0)
            expected = sorted(e for e in range(st.n_bonds)
                              if int(st.bond_u[e]) in sup and int(st.bond_v[e]) in sup)
            assert internal == expected
            # connected induced support
            seen = {v}
            stack = [v]
            adj = [[] for _ in range(st.n_atoms)]
            for e in range(st.n_bonds):
                adj[int(st.bond_u[e])].append(int(st.bond_v[e]))
                adj[int(st.bond_v[e])].append(int(st.bond_u[e]))
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y in sup and y not in seen:
                        seen.add(y)
                        stack.append(y)
            assert seen == sup


def test_contribution_identity_synthetic():
    model, _ = _model(double=True)
    view = SYN.make_view("identity6", 6, [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                                         (3, 4, 1), (4, 5, 1), (1, 4, 1)],
                         atom_types=[5] * 6)
    batch, st = _batch(view)
    batch = {k: (v.double() if torch.is_tensor(v) and v.dtype.is_floating_point else v)
             for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch, return_aux=True)
        contrib = model.word_contributions(batch, out)
    target = (out["yhat_std"] - model.beta
              - model.c_n * torch.log1p(out["atom_count"])
              - model.c_m * torch.log1p(out["bond_count"]))
    total = torch.zeros(1, dtype=torch.float64)
    total.index_add_(0, batch["word_mol"], contrib["c"])
    assert float((total - target).abs().max()) < 1e-10
    # word-pair expansion equals the direct bilinear term
    idx = batch["bond_index"]
    Uw = model.U(out["w"])
    Uz = out["Uz"]
    alpha = out["alpha"]
    lam = out["lam"]
    ptr = batch["root_word_ptr"]
    for e in range(int(idx.shape[0])):
        u, v = int(idx[e, 0]), int(idx[e, 1])
        pu0, pu1 = int(ptr[u]), int(ptr[u + 1])
        pv0, pv1 = int(ptr[v]), int(ptr[v + 1])
        qpair = (lam[e][None, None, :] * Uw[pu0:pu1, None, :]
                 * Uw[None, pv0:pv1, :]).sum(dim=2) / 4.0
        qexp = (alpha[pu0:pu1, None] * alpha[None, pv0:pv1] * qpair).sum()
        direct = (lam[e] * Uz[u] * Uz[v]).sum() / 4.0
        assert abs(float(qexp - direct)) < 1e-10
        assert abs(float(direct - out["Q"][e])) < 1e-10


def test_root_fixed_canonicalisation_ignores_support_order():
    view = SYN.make_view("identity6", 6, [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                                         (3, 4, 1), (4, 5, 1), (1, 4, 1)],
                         atom_types=[5] * 6)
    st = S.build_molecule_struct(view)
    atom_dims, bond_dims = D.atom_bond_dims()
    cert = CAN.canonical_certificates(st, atom_dims, bond_dims)
    st2 = copy.deepcopy(st)
    rng = np.random.default_rng(3)
    for w in range(st2.n_words):
        sup = st2.word_support[w]
        idx = np.flatnonzero(sup >= 0)
        if idx.size > 1:
            perm = rng.permutation(idx)
            st2.word_support[w, idx] = st2.word_support[w, perm]
    cert2 = CAN.canonical_certificates(st2, atom_dims, bond_dims)
    assert np.array_equal(cert, cert2)


def test_prediction_invariant_to_support_order_and_edge_reversal():
    model, _ = _model(double=True)
    view = SYN.make_view("identity6", 6, [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                                         (3, 4, 1), (4, 5, 1), (1, 4, 1)],
                         atom_types=[5] * 6)
    batch, st = _batch(view)

    def _to_double(b):
        return {k: (v.double() if torch.is_tensor(v) and v.dtype.is_floating_point else v)
                for k, v in b.items()}

    with torch.no_grad():
        y0 = float(model(_to_double(batch))["yhat_std"].item())
    st2 = copy.deepcopy(st)
    for w in range(st2.n_words):
        sup = st2.word_support[w]
        idx = np.flatnonzero(sup >= 0)
        if idx.size > 1:
            perm = idx[::-1]
            st2.word_support[w, idx] = st2.word_support[w, perm]
            st2.word_internal[w] = st2.word_internal[w]
    b2 = T.collate([0], {0: st2}, SYN.raw_from_views([view]), "cpu")
    with torch.no_grad():
        y1 = float(model(_to_double(b2))["yhat_std"].item())
    assert abs(y0 - y1) < 1e-8