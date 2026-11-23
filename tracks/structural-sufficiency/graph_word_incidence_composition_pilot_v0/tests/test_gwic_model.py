from __future__ import annotations

import copy

import numpy as np
import torch

import gwic_data as D
import gwic_model as M
import gwic_structure as S
import gwic_synthetic as SYN
import gwic_tensor as T


def _single_batch(view, norm):
    view = dict(view)
    view["atom_scalars"] = D.atom_scalars(view)
    raw = SYN.raw_from_views([view])
    st = S.build_molecule_struct(view)
    return T.collate([0], {0: st}, raw, "cpu")


def _reference_coverage(batch, alpha):
    n = int(batch["n_atoms"])
    root = batch["word_root"].cpu().numpy()
    support = batch["word_support"].cpu().numpy()
    a = alpha.detach().cpu().numpy()
    C = np.zeros((n, n), np.float64)
    for w in range(root.size):
        r = int(root[w])
        for t in support[w].tolist():
            if t >= 0:
                C[r, int(t)] += a[w]
    return C


def test_alpha_normalisation_softmax_and_coverage_reference(model_word):
    model, _ = model_word
    view = SYN.make_view("micro", 6, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1)])
    batch = _single_batch(view, None)
    out = model(batch, arm="WORD_REAL", return_aux=True)
    alpha = out["alpha"]
    ptr = batch["root_word_ptr"].cpu().numpy()
    roots = int(batch["n_atoms"])
    sums = torch.zeros(roots).index_add_(0, batch["word_root"], alpha)
    for r in range(roots):
        lo, hi = int(ptr[r]), int(ptr[r + 1])
        if hi > lo:
            assert abs(float(sums[r]) - 1.0) < 1e-5
            assert torch.all(alpha[lo:hi] >= 0)
    ref = _reference_coverage(batch, alpha)
    got = out["word"]["coverage"].detach().cpu().numpy()
    assert np.max(np.abs(ref - got)) < 1e-5


def test_overlap_is_atom_incidence_not_embedding_similarity(model_word):
    model, _ = model_word
    view = SYN.make_view("micro2", 6, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1)])
    batch = _single_batch(view, None)
    out = model(batch, arm="WORD_REAL", return_aux=True)
    C = _reference_coverage(batch, out["alpha"])
    idx = batch["bond_index"].cpu().numpy()
    geom = out["geom"].detach().cpu().numpy()
    for e, (u, v) in enumerate(idx.tolist()):
        assert abs(geom[e, 0] - float(np.sum(C[u] * C[v]))) < 1e-5
        assert abs(geom[e, 1] - float(C[u, v] + C[v, u])) < 1e-5
        assert abs(geom[e, 2] - float(abs(C[u, v] - C[v, u]))) < 1e-5
    # the geometric overlap is not the cosine of two word embeddings
    w = out["word"]["w"].detach()
    cos = torch.nn.functional.cosine_similarity(w[: min(8, w.shape[0])], w[1: min(8, w.shape[0]) + 1], dim=1)
    if cos.numel() and geom.size:
        k = min(int(cos.numel()), int(geom.shape[0]))
        assert not np.allclose(cos[:k].cpu().numpy(), geom[:k, 0])


def test_q_endpoint_swap_is_invariant(model_word):
    model, _ = model_word
    view = SYN.make_view("micro3", 5, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1)])
    batch = _single_batch(view, None)
    out = model(batch, arm="WORD_REAL", return_aux=True)
    batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    bi = batch2["bond_index"].clone()
    bi[:, 0], bi[:, 1] = bi[:, 1].clone(), bi[:, 0].clone()
    batch2["bond_index"] = bi
    out2 = model(batch2, arm="WORD_REAL", return_aux=True)
    assert torch.allclose(out["q"], out2["q"], atol=1e-6, rtol=1e-5)
    # q is the swap-invariant object required by the protocol; the aggregate
    # readout is not asserted here because float summation order may differ.


def test_same_checkpoint_geometry_zeroing(model_word):
    model, _ = model_word
    view = SYN.make_view("micro4", 6, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (0, 5, 1)])
    batch = _single_batch(view, None)
    model.eval()
    out_real = model(batch, arm="WORD_REAL", return_aux=True)
    out_zero = model(batch, arm="WORD_REAL", use_geometry=False, return_aux=True)
    zeros = torch.zeros_like(out_real["geom"])
    out_override_zero = model(batch, arm="WORD_REAL", geometry_override=zeros, return_aux=True)
    out_override_real = model(batch, arm="WORD_REAL", geometry_override=out_real["geom"], return_aux=True)
    assert torch.allclose(out_zero["yhat_std"], out_override_zero["yhat_std"], atol=1e-7, rtol=1e-5)
    assert torch.allclose(out_real["yhat_std"], out_override_real["yhat_std"], atol=1e-7, rtol=1e-5)
    assert float((out_real["yhat_std"] - out_zero["yhat_std"]).abs().max()) > 0.0


def test_b_c_identical_init_and_parameter_count(dims, norm):
    cfg = {"dims": {"atom": 48, "bond": 24, "sentence": 96, "word": 64, "q": 64,
                    "path_rank": 24, "path_end": 16, "path_close": 8, "nbr": 16,
                    "backbone_hidden": 192, "word_hidden": 160, "score_hidden": 96,
                    "q_hidden": 128, "readout_hidden": 192, "readout_mid": 96,
                    "dropout": 0.1}}
    m1, i1 = M.build_model(dims[0], dims[1], 0, norm, True, cfg)
    m2, i2 = M.build_model(dims[0], dims[1], 0, norm, True, cfg)
    assert i1["parameters"] == i2["parameters"]
    assert i1["init_hashes"] == i2["init_hashes"]
    assert list(m1.state_dict().keys()) == list(m2.state_dict().keys())
    for k in m1.state_dict():
        assert torch.equal(m1.state_dict()[k], m2.state_dict()[k])


def test_finite_nonzero_gradients(dims, norm):
    cfg = {"dims": {"dropout": 0.0}}
    model, _ = M.build_model(dims[0], dims[1], 3, norm, True, cfg)
    view = SYN.make_view("grad", 7, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)])
    batch = _single_batch(view, norm)
    target = torch.zeros(1)
    model.train()
    out = model(batch, arm="WORD_REAL")
    loss = torch.mean(out["yhat_std"] ** 2) + 1e-3 * out["yhat_std"].sum()
    # add a dense target so the readout receives a non-degenerate gradient
    loss = torch.mean((out["yhat_std"] - target) ** 2)
    model.zero_grad(set_to_none=True)
    loss.backward()
    modules = {
        "atom_emb": model.atom_emb,
        "bond_raw_emb": model.bond_raw_emb,
        "bond_encoder": model.bond_encoder,
        "path_joint": model.path_joint,
        "backbone_mlp": model.backbone_mlp,
        "word_joint": model.word_joint,
        "word_mlp": model.word_mlp,
        "score_mlp": model.score_mlp,
        "q_mlp": model.q_mlp,
        "readout_mlp": model.readout_mlp,
    }
    for name, module in modules.items():
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                total += float(p.grad.pow(2).sum())
        assert total > 0.0, f"zero gradient in {name}"


def test_single_vs_batch_prediction(model_word, bundle):
    model, _ = model_word
    model.eval()
    rows = np.union1d(bundle.fit_rows, bundle.eval_rows)[:3]
    st = S.build_all(bundle.raw, rows)
    batch = T.collate(rows.tolist(), st, bundle.raw, "cpu")
    with torch.no_grad():
        full = model(batch, arm="WORD_REAL")["yhat_std"].cpu().numpy()
        single = []
        for r in rows.tolist():
            b = T.collate([r], st, bundle.raw, "cpu")
            single.append(float(model(b, arm="WORD_REAL")["yhat_std"].cpu().numpy()[0]))
    assert np.max(np.abs(full - np.asarray(single))) < 1e-5


def test_renumbering_edge_order_and_candidate_order(model_word, bundle):
    model, _ = model_word
    model.eval()
    row = int(bundle.fit_rows[0])
    view = D.molecule_view(bundle.raw, row)
    view["atom_scalars"] = D.atom_scalars(view)
    batch0 = _single_batch(view, None)
    with torch.no_grad():
        y0 = model(batch0, arm="WORD_REAL")["yhat_std"].cpu().numpy()
    rng = np.random.default_rng(7)
    n = int(view["n_atoms"])
    perm = rng.permutation(n)
    m = int(view["n_bonds"])
    border = rng.permutation(m)
    view_p = SYN.permute_view(view, perm, border)
    batch_p = _single_batch(view_p, None)
    with torch.no_grad():
        yp = model(batch_p, arm="WORD_REAL")["yhat_std"].cpu().numpy()
    assert np.max(np.abs(y0 - yp)) < 1e-5

    # candidate reordering within each root
    st = S.build_molecule_struct(view)
    st2 = copy.deepcopy(st)
    for r in range(st.n_atoms):
        lo, hi = int(st.word_root_ptr[r]), int(st.word_root_ptr[r + 1])
        if hi - lo > 1:
            idx = np.arange(lo, hi)[::-1]
            st2.word_support[lo:hi] = st.word_support[idx]
            st2.word_internal[lo:hi] = st.word_internal[idx]
            st2.word_scalars[lo:hi] = st.word_scalars[idx]
    raw = SYN.raw_from_views([view])
    batch2 = T.collate([0], {0: st2}, raw, "cpu")
    with torch.no_grad():
        y2 = model(batch2, arm="WORD_REAL")["yhat_std"].cpu().numpy()
    assert np.max(np.abs(y0 - y2)) < 1e-5
