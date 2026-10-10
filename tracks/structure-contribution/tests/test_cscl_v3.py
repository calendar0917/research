"""cscl-v3 correctness tests (Phase C).

Covers: strict additivity of the S head, symmetry/invariance of the
contribution readout, shared-information alignment between S and O,
identifiability machinery, frozen-feature invariants, leakage guards, and a
small known-truth synthetic check.  Heavy tests are marked ``slow``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for _p in (str(_CODE_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.slow


def _drv():
    import run_cscl_v3 as drv

    return drv


def _sba():
    import strong_backbone_attribution as sba

    return sba


@pytest.fixture(scope="module")
def ns():
    drv = _drv()
    return drv.prepare_all(log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, ns):
    """Tiny S+O training run on CPU (few epochs, engineering check only)."""
    import torch

    drv = _drv()
    out = tmp_path_factory.mktemp("v3_train")
    device = torch.device("cpu")
    r_s = drv.train_arm("S", ns, device, out, epochs=4, log=lambda *a, **k: None)
    r_o = drv.train_arm("O", ns, device, out, epochs=4, log=lambda *a, **k: None)
    return drv, out, r_s, r_o


# ---------------------------------------------------------------------------
# static guards
# ---------------------------------------------------------------------------


def test_seed_zero_only():
    text = Path(_drv().__file__).read_text()
    assert "choices=[0]" in text


def test_no_official_split_or_frozen_pred_as_target():
    import ast

    text = Path(_drv().__file__).read_text()
    stripped = ast.unparse(ast.parse(text))
    for forbidden in ('split="valid"', 'split="test"', "encoded_valid", 'split="official_valid"'):
        assert forbidden not in stripped, forbidden
    # the frozen encoder is never rebuilt/replayed inside the v3 driver; the
    # committed F predictions are read ONLY for paired comparison in analyze
    assert "predict_raw" not in stripped
    assert "build_scale_model" not in stripped
    # F predictions must not enter the model input path: the only allowed use
    # is arms["F"] inside analyze (comparison), never collate/loss
    fn_analyze = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.FunctionDef) and n.name in ("collate", "train_arm", "export_contributions", "synthetic_task"):
            assert "DEV_PREDS_NPZ" not in ast.unparse(n), n.name


def test_protocol_yaml_frozen_bands():
    text = (
        REPO_ROOT
        / "tracks/structure-contribution/configs/strong_backbone_contribution_v3.yaml"
    ).read_text()
    for key in ("cscl-v3-strong-backbone-contribution", "0.11641", "best-monitor single checkpoint", "seed 1"):
        assert key in text


# ---------------------------------------------------------------------------
# collate + additivity + symmetry
# ---------------------------------------------------------------------------


def test_collate_deterministic_and_type_shared(ns):
    drv = _drv()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:40])
    b1 = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    b2 = drv.collate(ns.features, list(reversed(picks)), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    # per-molecule rows identical regardless of batch composition
    n0 = ns.features[picks[0]].n_units
    assert np.array_equal(b1.unit_feat[:n0].numpy(), b2.unit_feat[-n0:].numpy())
    assert np.array_equal(b1.unit_type[:n0].numpy(), b2.unit_type[-n0:].numpy())
    # same signature in two molecules -> same type id (cross-molecule sharing)
    sig0 = ns.features[picks[0]].unit_sigs[0]
    for fi in picks[1:]:
        if sig0 in ns.features[fi].unit_sigs:
            k = ns.features[fi].unit_sigs.index(sig0)
            assert b1.unit_type[0].item() == ns.vocab.to_id(sig0, ns.features[fi].unit_kinds[k], len(ns.features[fi].unit_atoms[k]))
            break


def test_S_prediction_is_strict_sum_and_invariant_to_row_permutation(ns, trained):
    import torch

    drv, out, r_s, r_o = trained
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:24])
    batch = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    model = drv.SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(torch.load(out / "S_s0_best_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        res = model(batch)
    pred = res["pred"].numpy()
    usum = np.zeros(batch.n_mols)
    np.add.at(usum, batch.unit_mol.numpy(), res["alpha"].numpy() + res["delta"].numpy())
    rsum = np.zeros(batch.n_mols)
    if res["gamma"].numel():
        np.add.at(rsum, batch.rel_mol.numpy(), res["gamma"].numpy())
    recon = float(res["bias"].item()) + usum + rsum + res["g_ctx"].numpy() + res["g_topo"].numpy()
    assert np.abs(recon - pred).max() < 1e-5

    # row permutation (units and relations reordered) leaves prediction and
    # per-unit/per-relation contributions unchanged (matched by row)
    perm_u = torch.randperm(batch.unit_feat.shape[0], generator=torch.Generator().manual_seed(3))
    perm_r = torch.randperm(batch.rel_feat.shape[0], generator=torch.Generator().manual_seed(4))
    bp = drv.V3Batch(
        batch.unit_mol[perm_u], batch.unit_type[perm_u], batch.unit_feat[perm_u], batch.unit_ctx[perm_u],
        batch.rel_mol[perm_r], batch.rel_feat[perm_r], batch.graph_feat,
    )
    with torch.no_grad():
        resp = model(bp)
    assert np.abs(resp["pred"].numpy() - pred).max() < 1e-6
    assert np.abs(resp["alpha"].numpy()[perm_u.argsort()] - res["alpha"].numpy()).max() < 1e-6
    if res["gamma"].numel():
        assert np.abs(resp["gamma"].numpy()[perm_r.argsort()] - res["gamma"].numpy()).max() < 1e-6


def test_relation_endpoint_symmetry(ns):
    """Relation rows are built ONLY from symmetric functions of the two unit
    representations plus endpoint-symmetric tail quantities: rebuilding a row
    from (z_hi, z_lo) instead of (z_lo, z_hi) must give the identical row, so
    the contribution cannot depend on endpoint order."""
    import torch

    drv = _drv()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:16])
    batch = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    rel_mol = batch.rel_mol.numpy()
    # recover per-relation unit indices by re-walking the features (collate order)
    zd = drv.ZDIM
    checked = 0
    cursor_u = 0
    cursor_r = 0
    for fi in picks:
        m = ns.features[fi]
        z = batch.unit_feat[cursor_u : cursor_u + m.n_units]
        for j in range(len(m.rel_lo)):
            lo, hi = int(m.rel_lo[j]), int(m.rel_hi[j])
            zlo, zhi = z[lo], z[hi]
            tail = batch.rel_feat[cursor_r + j, 3 * zd :]
            row_ab = torch.cat([zlo + zhi, torch.abs(zlo - zhi), zlo * zhi, tail])
            row_ba = torch.cat([zhi + zlo, torch.abs(zhi - zlo), zhi * zlo, tail])
            assert torch.equal(row_ab, row_ba)
            assert torch.allclose(row_ab, batch.rel_feat[cursor_r + j], atol=1e-6)
            checked += 1
        cursor_u += m.n_units
        cursor_r += len(m.rel_lo)
    assert checked == len(rel_mol) and checked > 100


def test_global_bank_sum_and_per_feature_export(ns):
    import torch

    drv = _drv()
    torch.manual_seed(1)
    model = drv.SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    x = torch.randn(7, sba_module().CTX_STRUCTURE)
    g, pf = model._bank(x, model.tau_ctx, model.w_ctx)
    assert torch.abs(pf.sum(1) - g).max() < 1e-6


def sba_module():
    return _sba()


def test_O_is_opaque_no_contribution_channels(ns, trained):
    import torch

    drv, out, r_s, r_o = trained
    model = drv.OpaqueHead(ns.vocab.n_total)
    model.load_state_dict(torch.load(out / "O_s0_best_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:8])
    batch = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        res = model(batch)
    assert set(res.keys()) == {"pred"}


def test_S_and_O_share_relation_inputs(ns, trained):
    """S and O read the same collated relation/unit rows (information parity at
    the input level; the type channel differs by design and is disclosed)."""
    import torch

    drv, out, r_s, r_o = trained
    s_state = torch.load(out / "S_s0_best_state.pt", map_location="cpu", weights_only=True)
    o_state = torch.load(out / "O_s0_best_state.pt", map_location="cpu", weights_only=True)
    assert s_state["gamma.0.weight"].shape == o_state["rel_enc.0.weight"].shape
    assert s_state["delta.0.weight"].shape[1] == o_state["unit_enc.0.weight"].shape[1] - 16


def test_identifiability_penalty_zero_at_constant_zero(ns):
    import torch

    drv = _drv()
    torch.manual_seed(0)
    model = drv.SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:8])
    batch = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        out = model(batch)
    # per-type centered delta + zero-mean gamma -> penalty ~ 0
    d = out["delta"].clone()
    sums = torch.zeros(ns.vocab.n_total)
    counts = torch.zeros(ns.vocab.n_total)
    sums.index_add_(0, batch.unit_type, d)
    counts.index_add_(0, batch.unit_type, torch.ones_like(d))
    means = sums / torch.clamp(counts, min=1.0)
    d = d - means[batch.unit_type]
    out["delta"] = d
    out["gamma"] = out["gamma"] - out["gamma"].mean()
    assert float(model.identifiability_penalty(out, batch.unit_type)) < 1e-12
    out["delta"] = out["delta"] + 1.0
    assert float(model.identifiability_penalty(out, batch.unit_type)) > 0.5


# ---------------------------------------------------------------------------
# paired bootstrap statistic (erratum-consistent)
# ---------------------------------------------------------------------------


def test_paired_bootstrap_signed():
    drv = _drv()
    rng = np.random.default_rng(0)
    e_a = rng.random(400)
    e_b = rng.random(400) + 0.1
    res = drv.paired_bootstrap(e_a, e_b, resamples=300, seed=0)
    point = float((e_a - e_b).mean())
    assert res["ci95"][0] <= point <= res["ci95"][1]
    assert res["mean_diff"] == point
    assert res["frac_a_better"] == float(((e_a - e_b) < 0).mean())


# ---------------------------------------------------------------------------
# extraction-level invariants (need the frozen backbone; heavy)
# ---------------------------------------------------------------------------


def test_extraction_batch_invariance_and_cache_consistency():
    """The same molecules extracted inside different batch groupings give the
    same aggregates (float tolerance), and match the cached features exactly
    when the grouping matches."""
    import torch

    sba = _sba()
    drv = _drv()
    blob = sba.load_features(log=lambda *a, **k: None)
    assert blob is not None, "run the extraction first (checks mode)"
    features = blob["features"]
    model, prep, blob10 = sba.load_frozen_backbone(log=lambda *a, **k: None)
    from run_prediction_feasibility_v2 import load_train_rows, apply_prep_fit7200
    import cscl_features as cf

    train = load_train_rows(log=lambda *a, **k: None)
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    apply_prep_fit7200(train, prep, blob10)
    dataset = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    rows = [100, 101, 102, 103]
    ys = [float(dataset[i].y.reshape(-1)[0]) for i in rows]
    mols_raw = sba.raw_ringchain([dataset[i] for i in rows], [smiles[i] for i in rows])
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    # grouping A: all four in one batch (same as extraction)
    batch = p1.env_collate([train[i] for i in rows])
    batch.y = torch.zeros(len(rows))
    feats_a, _ = sba.extract_batch(model, batch, mols_raw, ys, "cpu")
    # grouping B: two batches of two
    feats_b: list = []
    for sub in ([0, 1], [2, 3]):
        b2 = p1.env_collate([train[rows[i]] for i in sub])
        b2.y = torch.zeros(2)
        fb, _ = sba.extract_batch(model, b2, [mols_raw[i] for i in sub], [ys[i] for i in sub], "cpu")
        feats_b.extend(fb)
    for fa, fb in zip(feats_a, feats_b):
        for key in ("unit_sum_e", "unit_sum_e2", "unit_sum_pv", "rel_sum_pv", "rel_mean_relation", "ctx_structure", "topology"):
            d = np.abs(getattr(fa, key) - getattr(fb, key)).max()
            scale = max(np.abs(getattr(fa, key)).max(), 1.0)
            assert d / scale < 1e-4, (key, d / scale)
    # extracting with the ORIGINAL grouping (first 128 rows in one batch)
    # must reproduce the cached aggregates exactly (deterministic code path)
    rows_full = list(range(128))
    ys_full = [float(dataset[i].y.reshape(-1)[0]) for i in rows_full]
    mols_full = sba.raw_ringchain([dataset[i] for i in rows_full], [smiles[i] for i in rows_full])
    batch_full = p1.env_collate([train[i] for i in rows_full])
    batch_full.y = torch.zeros(len(rows_full))
    feats_full, _ = sba.extract_batch(model, batch_full, mols_full, ys_full, "cpu")
    cached = {m.row: m for m in features}
    for fi in (100, 101, 102, 103):
        fa, fc = feats_full[fi], cached[fi]
        assert np.array_equal(fa.unit_sum_e, fc.unit_sum_e)
        assert np.array_equal(fa.rel_sum_pv, fc.rel_sum_pv)
        assert np.array_equal(fa.rel_mean_relation, fc.rel_mean_relation)


def test_atom_relabel_invariance_of_prediction_and_partition():
    """Permuting the atom order of a molecule must not change the unit
    signature multiset or (per type) the unit-level aggregates of the frozen
    features."""
    import torch as th

    sba = _sba()
    model, prep, blob10 = sba.load_frozen_backbone(log=lambda *a, **k: None)
    from run_prediction_feasibility_v2 import load_train_rows, apply_prep_fit7200
    import cscl_features as cf
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1, e2e_dictenv_v0 as v0
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph

    train = load_train_rows(log=lambda *a, **k: None)
    apply_prep_fit7200(train, prep, blob10)
    dataset = cf.load_official_train()
    row = 5
    data = dataset[row]
    n = int(data.x.shape[0])
    perm = np.random.default_rng(7).permutation(n)
    mapping = th.as_tensor(np.argsort(perm), dtype=th.long)  # old index -> new index

    # relabeled molecule (graph-level fields unchanged by relabeling)
    row2 = train[row]
    saved = {k: getattr(row2, k).clone() for k in ("dict_phi", "dict_atom", "anchor", "patch_cont")}
    row2.dict_phi = row2.dict_phi[th.as_tensor(perm)].clone()
    row2.dict_atom = row2.dict_atom[th.as_tensor(perm)].clone()
    row2.anchor = row2.anchor[th.as_tensor(perm)].clone()
    row2.patch_cont = row2.patch_cont[th.as_tensor(perm)].clone()  # per-root sem block
    row2.x = data.x[th.as_tensor(perm)].clone()
    ei, ea = data.edge_index.clone(), data.edge_attr.clone()
    row2.edge_index = mapping[ei]
    row2.edge_attr = ea
    graph, _nt, edge_types = _data_to_graph(row2)
    inc = v0.env_incidence(graph, edge_types)
    for key, value in inc.items():
        setattr(row2, f"env_{key}", value)
    # pairs: all unordered pairs, canonical (new_lo < new_hi), lexicographic;
    # relation rows follow the corresponding original pair
    pairs_old = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
    pos_old = {(int(i), int(j)): k for k, (i, j) in enumerate(pairs_old)}
    new_pairs = np.stack([mapping[th.as_tensor(pairs_old[:, 0])].numpy(),
                          mapping[th.as_tensor(pairs_old[:, 1])].numpy()], axis=1)
    canon = np.sort(new_pairs, axis=1)
    order = np.lexsort((canon[:, 1], canon[:, 0]))
    row2.pair_index = th.as_tensor(canon[order].T.copy())
    rows_old = [pos_old[(int(min(new_pairs[k])), int(max(new_pairs[k])))] for k in order]
    row2.pair_relation = row2.pair_relation[th.as_tensor(rows_old)].clone()
    row2.pair_bucket = row2.pair_bucket[th.as_tensor(rows_old)].clone()

    batch = p1.env_collate([row2])
    batch.y = th.zeros(1)
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    mu = sba.raw_ringchain([row2], [smiles[row]])
    feats, aud = sba.extract_batch(model, batch, mu, [float(data.y.reshape(-1)[0])], "cpu")
    f = feats[0]
    fc = {m.row: m for m in _cached()}[row]
    assert sorted(f.unit_sigs) == sorted(fc.unit_sigs), "partition signature multiset must be relabel-invariant"
    tid_f = f.type_ids(_vocab())
    tid_c = fc.type_ids(_vocab())
    assert set(tid_f.tolist()) == set(tid_c.tolist())
    for t in set(tid_f.tolist()):
        m_f, m_c = tid_f == t, tid_c == t
        assert np.abs(f.unit_sum_e[m_f].sum(0) - fc.unit_sum_e[m_c].sum(0)).max() < 1e-3
        assert np.abs(f.unit_sum_e2[m_f].sum(0) - fc.unit_sum_e2[m_c].sum(0)).max() < 1e-2
    assert aud["unary_first_recon"] < 2e-4
    # restore the shared row object (this test runs on a private copy)
    for key, value in saved.items():
        setattr(row2, key, value)


_CACHED = None


def _cached():
    global _CACHED
    if _CACHED is None:
        blob = _sba().load_features(log=lambda *a, **k: None)
        assert blob is not None
        _CACHED = blob["features"]
    return _CACHED


_VOCAB = None


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        import cscl_features as cf

        ns_idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
        _VOCAB, _, _, _ = _sba().fit_unit_vocab(_cached(), ns_idx["fit_inner"])
    return _VOCAB


# ---------------------------------------------------------------------------
# synthetic known-truth (mechanism evidence; small fast version)
# ---------------------------------------------------------------------------


def test_synthetic_known_truth_small(ns, tmp_path):
    """Small fast version: subsample fit/monitor/dev, 60 epochs, verify
    direction recovery and true-vs-confuser interaction separation."""
    import torch

    drv = _drv()
    import types as _types

    sub = _types.SimpleNamespace(**vars(ns))
    rng = np.random.default_rng(0)
    sub.idx = {
        "fit_inner": ns.idx["fit_inner"][:400],
        "monitor": ns.idx["monitor"][:100],
        "dev": ns.idx["dev"][:150],
    }
    res = drv.synthetic_task(sub, tmp_path, torch.device("cpu"), epochs=60, log=lambda *a, **k: None)
    assert res["alpha_sign_recovery_top32"] >= 0.6, res["alpha_sign_recovery_top32"]
    assert res["alpha_spearman_top32"] >= 0.4
    a = res["gamma_checks"]["pair_A"]
    assert a["sign_recovered"], a
    assert res["gamma_checks"]["pair_D_confuser"]["gamma_hat_mean"] is not None
    # confuser (no true interaction) must stay small relative to pair A
    abs_d = abs(res["gamma_checks"]["pair_D_confuser"]["gamma_hat_mean"])
    abs_a = abs(a["gamma_hat_mean"])
    assert abs_d <= max(0.5 * abs_a, 0.15), (abs_d, abs_a)
