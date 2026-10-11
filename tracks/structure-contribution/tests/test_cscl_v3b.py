"""cscl-v3b correctness tests.

Covers: the single-variable training intervention (only the delta penalty is
removed; the gamma penalty is numerically identical to v3), architecture /
input / split identity with v3, the corrected synthetic attachment slot,
fit-only gauge offsets (rare types + UNK buckets), the frozen verification
gates (alpha/delta shift cancellation, prediction invariance, within-type
variance preservation, per-type fit means zero), strict-additivity export,
batch-composition invariance, save/load roundtrip, and official-split guards.

Heavy tests are marked ``slow`` (they load the frozen-feature cache and run
short CPU training).
"""

from __future__ import annotations

import ast
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
    import run_cscl_v3b as drv

    return drv


def _v3():
    import run_cscl_v3 as v3

    return v3


def _cg():
    import contribution_gauge as cg

    return cg


@pytest.fixture(scope="module")
def ns():
    return _drv().prepare_all(log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def gauged_setup(tmp_path_factory, ns):
    """Short CPU S-gauge training + full fit-only offsets + gates + export."""
    import torch

    drv = _drv()
    cg = _cg()
    out = tmp_path_factory.mktemp("v3b_train")
    device = torch.device("cpu")
    res = drv.train_gauge(ns, device, out, epochs=4, log=lambda *a, **k: None)
    # offsets from the FULL fit_inner (protocol discipline)
    fit_pos = drv._index_by_row(ns.features, ns.idx["fit_inner"])
    model = _v3().SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(torch.load(out / "S_gauge_s0_best_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    offsets, meta = cg.compute_fit_offsets(
        model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device="cpu", log=lambda *a, **k: None,
    )
    g = cg.GaugedHead(model, offsets).eval()
    return drv, cg, out, res, model, g, offsets, meta


# ---------------------------------------------------------------------------
# static guards
# ---------------------------------------------------------------------------


def test_no_official_split_access():
    import ast as _ast

    for fname in ("run_cscl_v3b.py", "contribution_gauge.py"):
        text = (_CODE_DIR / fname).read_text()
        stripped = _ast.unparse(_ast.parse(text))
        for forbidden in ('split="valid"', 'split="test"', "encoded_valid", "official_valid(", "official_test("):
            assert forbidden not in stripped, (fname, forbidden)


def test_collate_prepare_split_reused_from_v3_not_redefined():
    drv = _drv()
    v3 = _v3()
    assert drv.collate is v3.collate
    assert drv.prepare_all is v3.prepare_all
    assert drv._index_by_row is v3._index_by_row
    assert drv._batches is v3._batches
    assert drv.SBContribHead is v3.SBContribHead
    # and the v3b module does not redefine them
    tree = ast.parse((_CODE_DIR / "run_cscl_v3b.py").read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in ("collate", "prepare_all", "_index_by_row", "_batches", "SBContribHead"):
        assert name not in defined, name


def test_training_objective_changes_only_the_delta_penalty():
    """train_gauge must use L1 + PEN_COEF * gamma penalty only, and must not
    reference the delta penalty; hyper/optimizer constants come from v3."""
    tree = ast.parse((_CODE_DIR / "run_cscl_v3b.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_gauge")
    src = ast.unparse(fn)
    assert "gamma_identifiability_penalty(out)" in src
    assert "delta_identifiability_penalty" not in src
    assert "l1_loss" in src
    # split functions exist and are wired to the protocol constants
    assert "PEN_COEF * gamma_identifiability_penalty(out)" in src
    top = ast.unparse(tree)
    assert "ATTACH_IDX = cf.D_HIST + 4" in top


def test_split_indices_identical_to_v3_definition(ns):
    import cscl_features as cf

    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    for k in ("fit_inner", "monitor", "dev", "fit_all"):
        assert np.array_equal(ns.idx[k], idx[k])
    assert len(idx["fit_inner"]) == 7200 and len(idx["monitor"]) == 800 and len(idx["dev"]) == 2000


def test_head_architecture_identical_to_v3_checkpoint(ns):
    import torch

    drv = _drv()
    v3 = _v3()
    model = v3.SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    n_params = int(sum(p.numel() for p in model.parameters()))
    assert n_params == 118168  # v3 formal run param count
    state = torch.load(drv.V3_S_STATE, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)  # same module names/shapes
    g = _cg().GaugedHead(model, np.zeros(ns.vocab.n_total))
    assert int(sum(p.numel() for p in g.head.parameters())) == n_params


# ---------------------------------------------------------------------------
# penalty split: only delta removed; gamma numerically identical to v3
# ---------------------------------------------------------------------------


def test_penalty_split_sums_to_v3_penalty(ns):
    import torch

    drv = _drv()
    v3 = _v3()
    torch.manual_seed(0)
    model = v3.SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:24])
    batch = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        out = model(batch)
    full = model.identifiability_penalty(out, batch.unit_type)
    d = drv.delta_identifiability_penalty(out, batch.unit_type, ns.vocab.n_total)
    g = drv.gamma_identifiability_penalty(out)
    assert torch.equal(full, d + g)
    # gamma part alone: identical to the v3 penalty evaluated with delta == 0
    out0 = dict(out)
    out0["delta"] = torch.zeros_like(out["delta"])
    full_zero_delta = model.identifiability_penalty(out0, batch.unit_type)
    assert torch.allclose(full_zero_delta, g, atol=0.0)


def test_gamma_penalty_matches_v3_on_trained_outputs(gauged_setup):
    """On real trained-model outputs, the split gamma penalty equals the v3
    combined penalty evaluated with a zero delta (i.e. the gamma term is
    numerically unchanged from v3)."""
    import torch

    drv, cg, out, res, model, g, offsets, meta = gauged_setup
    v3 = _v3()
    ns = _NS["ns"]
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        out = model(b)
    g_v3b = drv.gamma_identifiability_penalty(out)
    out0 = dict(out)
    out0["delta"] = torch.zeros_like(out["delta"])
    g_v3 = v3.SBContribHead.identifiability_penalty(model, out0, b.unit_type)
    assert torch.allclose(g_v3b, g_v3, atol=0.0)


# helper to share the module ns with late tests (fixture scoping)
_NS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _remember_ns(ns):
    _NS["ns"] = ns
    _NS["features"] = ns.features
    yield


# ---------------------------------------------------------------------------
# corrected synthetic attachment slot
# ---------------------------------------------------------------------------


def test_attach_descriptor_slot_is_boundary_attachment_count():
    """The synthetic truth must read the unit boundary attachment count at
    desc[:, D_HIST+4] = 36 -- NOT the atom-histogram bin at index 4 (the v3
    bug)."""
    import cscl_features as cf
    import cscl_units as cu

    drv = _drv()
    assert drv.ATTACH_IDX == cf.D_HIST + 4 == 36
    dataset = cf.load_official_train()
    max_diff_36, n_checked = 0.0, 0
    for row in (3, 40, 123):
        data = dataset[row]
        mu = cf.molecule_units(
            row, data.x.numpy(), data.edge_index.numpy(), data.edge_attr.numpy()
        )
        part = cu.build_partition(
            [int(v) for v in data.x.reshape(-1)],
            [(int(data.edge_index[0, k]), int(data.edge_index[1, k])) for k in range(data.edge_index.shape[1])],
            [int(data.edge_attr.reshape(-1)[k]) for k in range(data.edge_index.shape[1])],
        )
        attach = [0] * part.n_units
        for (a, b), bts in part.inter_relations().items():
            attach[a] += len(bts)
            attach[b] += len(bts)
        got = drv.attach_counts(mu)
        assert np.array_equal(got, np.asarray(attach, dtype=np.float64))
        # and the raw column really is a count-like integer block, distinct
        # from atom-histogram bin 4 in general
        col4 = mu.desc[:, 4]
        if len(attach) > 1:
            assert not np.array_equal(col4, np.asarray(attach, dtype=np.float32)) or col4.abs().sum() == 0
        max_diff_36 = max(max_diff_36, float(np.abs(got - mu.desc[:, drv.ATTACH_IDX]).max()))
        n_checked += 1
    assert n_checked == 3 and max_diff_36 == 0.0


# ---------------------------------------------------------------------------
# fit-only offsets: rare types, UNK buckets, fallback
# ---------------------------------------------------------------------------


def test_fit_offsets_full_fit_and_fallback_rules(gauged_setup, ns):
    import torch

    drv, cg, out, res, model, g, offsets, meta = gauged_setup
    assert meta["n_fit_units"] == sum(m.n_units for fi in drv._index_by_row(ns.features, ns.idx["fit_inner"]) for m in [ns.features[fi]])
    assert meta["fallback_rule"] == "m_t = 0.0 for type buckets with zero fit_inner instances"
    # every type with fit instances has a finite offset; zero-count types are 0
    fit_types = np.concatenate([m.type_ids(ns.vocab) for m in (
        ns.features[fi] for fi in drv._index_by_row(ns.features, ns.idx["fit_inner"])
    )])
    counts = np.bincount(fit_types, minlength=ns.vocab.n_total)
    assert np.isfinite(offsets).all()
    assert np.all(offsets[counts == 0] == 0.0)
    assert meta["n_types_zero_fit_instances"] == int((counts == 0).sum())
    # at least one non-known bucket has fit instances (UNK machinery active)
    unk_ids = [i for i in range(ns.vocab.n_total) if i >= ns.vocab.n_known]
    assert any(counts[i] > 0 for i in unk_ids)
    # offsets hash bookkeeping present (provenance)
    for key in ("fit_unit_type_hash", "fit_rows_hash", "counts_hash", "offsets_hash"):
        assert isinstance(meta[key], str) and len(meta[key]) == 16


# ---------------------------------------------------------------------------
# frozen verification gates (the protocol yaml list)
# ---------------------------------------------------------------------------


def test_gauge_gates_on_all_splits(ns, gauged_setup):
    import torch

    drv, cg, out, res, model, g, offsets, meta = gauged_setup
    device = torch.device("cpu")
    batches = {}
    for name, rows in (("fit", ns.idx["fit_inner"]), ("monitor", ns.idx["monitor"]), ("dev", ns.idx["dev"])):
        pos = drv._index_by_row(ns.features, rows)
        batches[name] = drv.collate(ns.features, pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    checks = cg.verify_gauge_invariance(model, g, batches, log=lambda *a, **k: None)
    assert checks["all_gates_ok"], checks
    for split in ("fit", "monitor", "dev"):
        assert checks[split]["max_abs_pred_diff"] <= 1e-5
        assert checks[split]["max_abs_alpha_delta_identity_diff"] <= 1e-5
        assert checks[split]["max_abs_gamma_diff"] == 0.0
        assert checks[split]["max_abs_g_ctx_diff"] == 0.0
        assert checks[split]["max_abs_g_topo_diff"] == 0.0
        assert checks[split]["max_abs_within_type_std_gap"] <= 1e-6
    assert checks["fit_canonical_delta_per_type_mean_max_abs"] <= 1e-6
    for split in ("fit", "monitor", "dev"):
        assert checks[split]["max_abs_within_type_std_gap"] <= 1e-6
        assert checks[split]["max_abs_same_type_pairwise_diff_gap"] <= 1e-6
    # canonical alpha = alpha_raw + offset, per type (shared scalar preserved)
    b = batches["dev"]
    with torch.no_grad():
        gg = g(b)
    assert float(((gg["alpha"] - gg["alpha_raw"]) - g.offset[b.unit_type]).abs().max()) <= 1e-6


def test_gauged_export_is_strictly_additive_raw_and_canonical(ns, gauged_setup, tmp_path):
    drv, cg, out, res, model, g, offsets, meta = gauged_setup
    import torch

    dev_pos = drv._index_by_row(ns.features, ns.idx["dev"][:120])
    info = drv.export_gauged_contributions(
        g, ns, dev_pos, torch.device("cpu"), tmp_path / "contribs.npz"
    )
    assert info["additivity_max_abs_residual_canonical"] < 1e-5
    assert info["additivity_max_abs_residual_raw"] < 1e-5
    z = np.load(tmp_path / "contribs.npz")
    # canonical and raw unit sums coincide per molecule (shift cancels)
    usum_c = np.zeros(z["pred"].shape[0])
    np.add.at(usum_c, z["unit_mol"], z["alpha"] + z["delta"])
    usum_r = np.zeros_like(usum_c)
    np.add.at(usum_r, z["unit_mol"], z["alpha_raw"] + z["delta_raw"])
    assert np.abs(usum_c - usum_r).max() < 1e-5


def test_batch_composition_and_row_permutation_invariance(ns, gauged_setup):
    import torch

    drv, cg, out, res, model, g, offsets, meta = gauged_setup
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    full = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        r_full = g(full)
        half_a = g(drv.collate(ns.features, picks[:32], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean))
        half_b = g(drv.collate(ns.features, picks[32:], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean))
    assert float((torch.cat([half_a["pred"], half_b["pred"]]) - r_full["pred"]).abs().max()) < 1e-5
    assert float((torch.cat([half_a["delta"], half_b["delta"]]) - r_full["delta"]).abs().max()) < 1e-5
    # unit/relation row permutation inside a batch
    perm_u = torch.randperm(full.unit_feat.shape[0], generator=torch.Generator().manual_seed(3))
    perm_r = torch.randperm(full.rel_feat.shape[0], generator=torch.Generator().manual_seed(4))
    bp = _v3().V3Batch(
        full.unit_mol[perm_u], full.unit_type[perm_u], full.unit_feat[perm_u], full.unit_ctx[perm_u],
        full.rel_mol[perm_r], full.rel_feat[perm_r], full.graph_feat,
    )
    with torch.no_grad():
        r_perm = g(bp)
    assert float((r_perm["pred"] - r_full["pred"]).abs().max()) < 1e-6
    assert float((r_perm["delta"][perm_u.argsort()] - r_full["delta"]).abs().max()) < 1e-6


def test_gauge_save_load_roundtrip(ns, gauged_setup, tmp_path):
    import torch

    drv, cg, out, res, model, g, offsets, meta = gauged_setup
    path = tmp_path / "gauge.pt"
    cg.save_gauge(path, offsets, meta, model.state_dict())
    g2, meta2 = cg.load_gauge(path, ns.vocab.n_total, ns.tau_ctx, ns.tau_topo, device="cpu")
    assert meta2["n_fit_units"] == meta["n_fit_units"]
    offsets2 = np.asarray(torch.load(path, map_location="cpu", weights_only=False)["offsets"])
    assert np.array_equal(offsets2, offsets)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:48])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        p1 = g(b)["pred"].numpy()
        p2 = g2(b)["pred"].numpy()
        d1 = g(b)["delta"].numpy()
        d2 = g2(b)["delta"].numpy()
    assert np.abs(p1 - p2).max() == 0.0
    assert np.abs(d1 - d2).max() == 0.0


# ---------------------------------------------------------------------------
# corrected synthetic known-truth (small fast version)
# ---------------------------------------------------------------------------


def test_synthetic_gauge_small(ns, tmp_path):
    import torch
    import types as _types

    drv = _drv()
    sub = _types.SimpleNamespace(**vars(ns))
    sub.idx = {
        "fit_inner": ns.idx["fit_inner"][:400],
        "monitor": ns.idx["monitor"][:100],
        "dev": ns.idx["dev"][:150],
    }
    res = drv.synthetic_task_gauge(sub, tmp_path, torch.device("cpu"), epochs=30, log=lambda *a, **k: None)
    # prediction invariance under the gauge on the synthetic task
    assert res["gauge_max_abs_pred_diff_synthetic"] <= 1e-5
    assert res["dev_mae_syn_raw"] == pytest.approx(res["dev_mae_syn_canonical"], abs=1e-7)
    # mechanism: direction recovery must survive the corrected truth
    assert res["delta_recovery_pearson_raw"] >= 0.5, res["delta_recovery_pearson_raw"]
    assert res["alpha_sign_recovery_top32"] >= 0.5
    a = res["gamma_checks"]["pair_A"]
    assert a["sign_recovered"], a
    abs_d = abs(res["gamma_checks"]["pair_D_confuser"]["gamma_hat_mean"])
    abs_a = abs(a["gamma_hat_mean"])
    assert abs_d <= max(0.5 * abs_a, 0.15), (abs_d, abs_a)
