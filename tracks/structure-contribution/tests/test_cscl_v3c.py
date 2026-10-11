"""cscl-v3c correctness tests.

Covers (task brief sect. 6.1): identical inputs except the type-embedding
channel, bit-exact shared init + untrained prediction equality, learnability
and exclusivity of the type channel, alpha shared per-type scalar, untouched
gamma / global banks / bias, v3b-identical objective (gamma penalty only, no
delta penalty, no embedding regulariser), strict additivity, lossless
canonicalization (prediction/within-type variance/fit per-type mean zero),
UNK fallback, relabel / endpoint-swap / batch-composition invariance,
checkpoint roundtrip, official-split guards, raw-y-only supervision.

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
    import run_cscl_v3c as drv

    return drv


def _v3():
    import run_cscl_v3 as v3

    return v3


def _v3b():
    import run_cscl_v3b as v3b

    return v3b


def _cg():
    import contribution_gauge as cg

    return cg


@pytest.fixture(scope="module")
def ns():
    return _drv().prepare_all(log=lambda *a, **k: None)


@pytest.fixture(scope="module")
def trained_setup(tmp_path_factory, ns):
    """Short CPU training of the type-conditioned head + full gauge pipeline."""
    import torch

    drv = _drv()
    cg = _cg()
    out = tmp_path_factory.mktemp("v3c_train")
    device = torch.device("cpu")
    res = drv.train_typecond(ns, device, out, epochs=4, log=lambda *a, **k: None)
    fit_pos = drv._index_by_row(ns.features, ns.idx["fit_inner"])
    model = drv.TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(torch.load(out / "S_typecond_s0_best_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    offsets, meta = cg.compute_fit_offsets(
        model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device="cpu", log=lambda *a, **k: None,
    )
    g = cg.GaugedHead(model, offsets).eval()
    return drv, cg, out, res, model, g, offsets, meta


_NS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _remember_ns(ns):
    _NS["ns"] = ns
    yield


# ---------------------------------------------------------------------------
# static guards: official splits, component reuse, channel exclusivity
# ---------------------------------------------------------------------------


def test_no_official_split_access():
    for fname in ("run_cscl_v3c.py", "contribution_gauge.py"):
        text = (_CODE_DIR / fname).read_text()
        stripped = ast.unparse(ast.parse(text))
        for forbidden in ('split="valid"', 'split="test"', "encoded_valid", "official_valid(", "official_test("):
            assert forbidden not in stripped, (fname, forbidden)


def test_reuses_v3_v3b_components_not_redefined():
    drv, v3, v3b = _drv(), _v3(), _v3b()
    assert drv.collate is v3.collate
    assert drv.prepare_all is v3.prepare_all
    assert drv._index_by_row is v3._index_by_row
    assert drv._batches is v3._batches
    assert drv.SBContribHead is v3.SBContribHead
    assert drv.gamma_identifiability_penalty is v3b.gamma_identifiability_penalty
    assert drv.subgroup_table_v3b is v3b.subgroup_table_v3b
    assert drv.attach_counts is v3b.attach_counts
    tree = ast.parse((_CODE_DIR / "run_cscl_v3c.py").read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in ("collate", "prepare_all", "_index_by_row", "_batches", "SBContribHead",
                 "gamma_identifiability_penalty", "compute_fit_offsets", "verify_gauge_invariance"):
        assert name not in defined, name


def test_only_delta_reads_type_embedding():
    """AST: type_emb is read ONLY inside unit_terms; no fusion MLP after the
    additive sum; gamma/global never touch types."""
    drv = _drv()
    tree = ast.parse((_CODE_DIR / "run_cscl_v3c.py").read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TypeCondContribHead")
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        uses = any(
            isinstance(node, ast.Attribute) and node.attr == "type_emb"
            for node in ast.walk(fn)
        )
        if fn.name == "unit_terms" or fn.name == "__init__":
            assert uses or fn.name == "__init__"
        else:
            assert not uses, f"{fn.name} must not read type_emb"
    # no post-sum fusion: forward composes exactly bias + unit_sum + rel_sum + banks
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    fwd_src = ast.unparse(fwd)
    assert "unit_sum" in fwd_src and "rel_sum" in fwd_src
    for bad in ("fusion", "readout", "head_mlp", "nn.Sequential"):
        assert bad not in fwd_src
    # gamma input rows built in v3 collate (1273-D, no type id)
    ns = _NS["ns"]
    model = drv.TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    assert model.gamma[0].weight.shape[1] == _v3().RELIN == 1273
    assert model.alpha.weight.shape == (ns.vocab.n_total, 1)
    assert model.type_emb.weight.shape == (ns.vocab.n_total, 16)
    assert model.delta[0].weight.shape[1] == 384 + 40 + 16


def test_param_budget_and_added_capacity(ns):
    drv = _drv()
    ref, model, report = drv.build_head_pair(ns, log=lambda *a, **k: None)
    n_ref = sum(p.numel() for p in ref.parameters())
    n_new = sum(p.numel() for p in model.parameters())
    assert n_ref == 118168  # v3/v3b formal param count
    assert n_new == n_ref + ns.vocab.n_total * 16 + 16 * 64
    assert report["n_params_added"] == n_new - n_ref == 12176 + 1024
    assert report["added_breakdown"] == {
        "delta_first_layer_new_columns": 1024,
        "type_embedding_table": ns.vocab.n_total * 16,
    }


# ---------------------------------------------------------------------------
# init discipline: shared bit-exact + untrained prediction equality
# ---------------------------------------------------------------------------


def test_shared_init_bit_exact_and_new_columns_zero(ns):
    import torch

    drv = _drv()
    ref, model, report = drv.build_head_pair(ns, log=lambda *a, **k: None)
    assert report["shared_init_bit_exact"]
    # independent deterministic embedding stream, reproducible
    gen = torch.Generator().manual_seed(drv.TYPE_EMB_SEED)
    assert torch.equal(model.type_emb.weight, torch.randn(ns.vocab.n_total, 16, generator=gen) * 0.1)
    # the trained v3b checkpoint is not involved anywhere in init
    assert report["init_rule"]["trained_v3b_checkpoint_used"] is False
    zin = 384 + 40
    assert torch.equal(model.delta[0].weight[:, :zin], ref.delta[0].weight)
    assert float(model.delta[0].weight[:, zin:].abs().sum()) == 0.0
    # every OTHER shared parameter must be bit-exact under its own name
    for name, p in ref.named_parameters():
        if name == "delta.0.weight":
            continue
        assert torch.equal(p, dict(model.named_parameters())[name]), name


def test_untrained_prediction_equality_with_v3b_head(ns):
    """Zero-init of the new columns must make the untrained v3c head predict
    identically to the untrained v3b head (device float noise tolerated;
    exact equality expected on CPU)."""
    import torch

    drv = _drv()
    ref, model, _ = drv.build_head_pair(ns, log=lambda *a, **k: None)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        r_ref, r_new = ref(b), model(b)
    d_pred = float((r_new["pred"] - r_ref["pred"]).abs().max())
    d_delta = float((r_new["delta"] - r_ref["delta"]).abs().max())
    assert d_pred <= 1e-6 and d_delta <= 1e-6
    # alpha/gamma/global/bias must be bit-identical (same ops, same params)
    assert torch.equal(r_new["alpha"], r_ref["alpha"])
    assert torch.equal(r_new["gamma"], r_ref["gamma"])
    assert torch.equal(r_new["g_ctx"], r_ref["g_ctx"])
    assert torch.equal(r_new["g_topo"], r_ref["g_topo"])


# ---------------------------------------------------------------------------
# the type channel is learnable and effective
# ---------------------------------------------------------------------------


def test_type_embedding_channel_is_learnable(ns):
    """Zero-init => the embedding gets zero grad at step 0 while the new
    weight columns get nonzero grad; after one optimizer step the embedding
    grad becomes nonzero AND perturbing it changes delta."""
    import torch

    drv = _drv()
    _ref, model, _ = drv.build_head_pair(ns, log=lambda *a, **k: None)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    target = torch.zeros(b.n_mols)

    def step():
        out = model(b)
        loss = torch.nn.functional.l1_loss(out["pred"], target)
        model.zero_grad(set_to_none=True)
        loss.backward()

    zin = 384 + 40
    step()
    new_cols_grad = model.delta[0].weight.grad[:, zin:]
    emb_grad0 = model.type_emb.weight.grad
    assert float(new_cols_grad.abs().sum()) > 0, "new weight columns must receive gradient"
    assert emb_grad0 is not None and float(emb_grad0.abs().sum()) == 0.0, (
        "with zero-initialized new columns the embedding gradient is zero at step 0"
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    opt.step()
    with torch.no_grad():
        assert float(model.delta[0].weight[:, zin:].abs().sum()) > 0
    step()
    emb_grad1 = model.type_emb.weight.grad
    assert emb_grad1 is not None and float(emb_grad1.abs().sum()) > 0, (
        "after the new columns move, the embedding must become learnable"
    )
    # trained-step channel effectiveness: perturbing the embedding rows that
    # actually occur in the batch must change delta
    with torch.no_grad():
        model.eval()
        d_before = model(b)["delta"].clone()
        present_types = torch.unique(b.unit_type)
        model.type_emb.weight[present_types] += 1.0
        d_after = model(b)["delta"]
    assert float((d_after - d_before).abs().max()) > 0, "embedding must influence delta"


def test_alpha_is_shared_per_type_scalar(ns):
    import torch

    drv = _drv()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    model = drv.TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.eval()
    with torch.no_grad():
        alpha = model(b)["alpha"]
    tab = model.alpha.weight.squeeze(-1)
    ut = b.unit_type
    assert torch.equal(alpha, tab[ut])  # pure type lookup, no unit-level input


# ---------------------------------------------------------------------------
# objective guards: v3b-identical penalty split, raw-y only supervision
# ---------------------------------------------------------------------------


def test_training_objective_is_v3b_verbatim():
    tree = ast.parse((_CODE_DIR / "run_cscl_v3c.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_typecond")
    src = ast.unparse(fn)
    assert "PEN_COEF * gamma_identifiability_penalty(out)" in src
    assert "delta_identifiability_penalty" not in src
    assert "l1_loss" in src
    assert "type_emb" not in src  # no embedding regulariser in the training loop
    # constants inherited from v3
    drv = _drv()
    assert (drv.LR, drv.WEIGHT_DECAY, drv.GRAD_CLIP, drv.BATCH_SIZE, drv.EPOCHS, drv.PEN_COEF) == (
        1.0e-3, 1.0e-5, 5.0, 128, 240, 1.0
    )
    assert drv.TYPE_EMB_DIM == 16 and drv.TYPE_EMB_SEED == 0 and drv.MIN_PER_HALF == 10


def test_no_full_y_predictions_as_inputs_or_labels():
    """The frozen Full-Y prediction files may only appear in analyze-compare
    paths (reference arms), never in training or synthetic supervision."""
    tree = ast.parse((_CODE_DIR / "run_cscl_v3c.py").read_text())
    for fname in ("train_typecond", "synthetic_task_typecond"):
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fname)
        src = ast.unparse(fn)
        # the model's own soup/eval npz exports are fine; reading any saved
        # prediction file (Full-Y or historical heads) is not
        assert "DEV_PREDS" not in src and "MON_PREDS" not in src
        assert "np.load" not in src and "soup_state.pt" not in src
    # synthetic truth built from hash-based alphas + attach counts + gamma pairs
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "synthetic_task_typecond")
    assert "alpha_star" in ast.unparse(fn)


def test_seed_zero_only(ns):
    drv = _drv()
    with pytest.raises(AssertionError):
        drv.train_typecond(ns, __import__("torch").device("cpu"), Path("/tmp/never"), seed=1, epochs=1,
                           log=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# heavy: trained pipeline, gauge gates, additivity, invariance, roundtrip
# ---------------------------------------------------------------------------


def test_gamma_penalty_matches_v3b_on_trained_outputs(trained_setup):
    import torch

    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    v3 = _v3()
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        out_new = model(b)
    g_v3c = drv.gamma_identifiability_penalty(out_new)
    out0 = dict(out_new)
    out0["delta"] = torch.zeros_like(out_new["delta"])
    g_v3 = v3.SBContribHead.identifiability_penalty(model, out0, b.unit_type)
    assert torch.allclose(g_v3c, g_v3, atol=0.0)


def test_gauge_gates_all_splits(trained_setup):
    import torch

    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
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
        assert checks[split]["max_abs_same_type_pairwise_diff_gap"] <= 1e-6
    assert checks["fit_canonical_delta_per_type_mean_max_abs"] <= 1e-6
    b = batches["dev"]
    with torch.no_grad():
        gg = g(b)
    assert float(((gg["alpha"] - gg["alpha_raw"]) - g.offset[b.unit_type]).abs().max()) <= 1e-6


def test_fit_offsets_rules_and_unk_fallback(trained_setup):
    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    fit_pos = drv._index_by_row(ns.features, ns.idx["fit_inner"])
    fit_types = np.concatenate([ns.features[fi].type_ids(ns.vocab) for fi in fit_pos])
    counts = np.bincount(fit_types, minlength=ns.vocab.n_total)
    assert meta["fallback_rule"] == "m_t = 0.0 for type buckets with zero fit_inner instances"
    assert np.isfinite(offsets).all()
    assert np.all(offsets[counts == 0] == 0.0)
    assert meta["n_types_zero_fit_instances"] == int((counts == 0).sum())
    # UNK machinery active and embedded inside the fit-only vocab (761 ids)
    unk_ids = list(range(ns.vocab.n_known, ns.vocab.n_total))
    assert any(counts[i] > 0 for i in unk_ids)
    assert model.type_emb.weight.shape[0] == ns.vocab.n_total  # UNK rows embedded from fit vocab
    # dev-only signatures must never create new embedding rows: to_id stays in range
    for m in (ns.features[fi] for fi in drv._index_by_row(ns.features, ns.idx["dev"][:50])):
        for k, sig in enumerate(m.unit_sigs):
            tid = ns.vocab.to_id(sig, m.unit_kinds[k], len(m.unit_atoms[k]))
            assert 0 <= tid < ns.vocab.n_total


def test_strict_additivity_export_raw_and_canonical(trained_setup, tmp_path):
    import torch

    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    dev_pos = drv._index_by_row(ns.features, ns.idx["dev"][:120])
    info = drv.export_gauged_contributions(g, ns, dev_pos, torch.device("cpu"), tmp_path / "contribs.npz")
    assert info["additivity_max_abs_residual_canonical"] < 1e-5
    assert info["additivity_max_abs_residual_raw"] < 1e-5
    assert info["global_bank_sum_residual"] < 1e-6
    z = np.load(tmp_path / "contribs.npz")
    usum_c = np.zeros(z["pred"].shape[0])
    np.add.at(usum_c, z["unit_mol"], z["alpha"] + z["delta"])
    usum_r = np.zeros_like(usum_c)
    np.add.at(usum_r, z["unit_mol"], z["alpha_raw"] + z["delta_raw"])
    assert np.abs(usum_c - usum_r).max() < 1e-5
    # per-unit offset exported and consistent with the canonical identities
    assert np.abs((z["alpha"] - z["alpha_raw"]) - z["unit_offset"]).max() < 1e-6
    assert np.abs((z["delta_raw"] - z["delta"]) - z["unit_offset"]).max() < 1e-6
    # atom membership roundtrip: every atom exactly one unit, counts conserve
    rows = z["rows"]
    natoms = sum(ns.features[drv._index_by_row(ns.features, rows)[i]].n_atoms for i in range(len(rows)))
    assert len(z["unit_atom_flat"]) == natoms
    assert z["unit_atom_ptr"][0] == 0 and z["unit_atom_ptr"][-1] == natoms


def test_batch_composition_and_permutation_invariance(trained_setup):
    import torch

    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:64])
    full = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        r_full = g(full)
        half_a = g(drv.collate(ns.features, picks[:32], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean))
        half_b = g(drv.collate(ns.features, picks[32:], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean))
    assert float((torch.cat([half_a["pred"], half_b["pred"]]) - r_full["pred"]).abs().max()) < 1e-5
    assert float((torch.cat([half_a["delta"], half_b["delta"]]) - r_full["delta"]).abs().max()) < 1e-5
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
    # the RAW (ungauged) head must be invariant too (relabeled units)
    with torch.no_grad():
        p_full = model(full)["pred"]
        p_perm = model(bp)["pred"]
    assert float((p_perm - p_full).abs().max()) < 1e-6


def test_gauge_save_load_roundtrip_with_v3c_head(trained_setup, tmp_path):
    import torch

    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    path = tmp_path / "gauge.pt"
    cg.save_gauge(path, offsets, meta, model.state_dict(), protocol_id=drv.PROTOCOL_ID)
    g2, meta2 = cg.load_gauge(path, ns.vocab.n_total, ns.tau_ctx, ns.tau_topo, device="cpu",
                              head_builder=drv.TypeCondContribHead)
    assert meta2["n_fit_units"] == meta["n_fit_units"]
    offsets2 = np.asarray(torch.load(path, map_location="cpu", weights_only=False)["offsets"])
    assert np.array_equal(offsets2, offsets)
    assert torch.load(path, map_location="cpu", weights_only=False)["protocol"] == drv.PROTOCOL_ID
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:48])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        assert np.abs(g(b)["pred"].numpy() - g2(b)["pred"].numpy()).max() == 0.0
        assert np.abs(g(b)["delta"].numpy() - g2(b)["delta"].numpy()).max() == 0.0


def test_checkpoint_state_roundtrip(trained_setup, tmp_path):
    import torch

    drv, cg, out, res, model, g, offsets, meta = trained_setup
    ns = _NS["ns"]
    state = torch.load(out / "S_typecond_s0_best_state.pt", map_location="cpu", weights_only=True)
    model2 = drv.TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model2.load_state_dict(state, strict=True)
    picks = drv._index_by_row(ns.features, ns.idx["dev"][:48])
    b = drv.collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    model.eval()
    model2.eval()
    with torch.no_grad():
        assert torch.equal(model(b)["pred"], model2(b)["pred"])


# ---------------------------------------------------------------------------
# heavy: corrected synthetic known-truth (small fast version)
# ---------------------------------------------------------------------------


def test_synthetic_typecond_small(ns, tmp_path):
    import torch
    import types as _types

    drv = _drv()
    sub = _types.SimpleNamespace(**vars(ns))
    sub.idx = {
        "fit_inner": ns.idx["fit_inner"][:400],
        "monitor": ns.idx["monitor"][:100],
        "dev": ns.idx["dev"][:150],
    }
    res = drv.synthetic_task_typecond(sub, tmp_path, torch.device("cpu"), epochs=30, log=lambda *a, **k: None)
    # prediction invariance under the gauge on the synthetic task
    assert res["gauge_max_abs_pred_diff_synthetic"] <= 1e-5
    assert res["dev_mae_syn_raw"] == pytest.approx(res["dev_mae_syn_canonical"], abs=1e-7)
    # mechanism invariants (budget-independent): the gauge must place the
    # type-conditional mean into alpha and restore delta's contextual signal.
    # Absolute recovery levels depend on the (short) training budget and are
    # judged only in the formal 240-epoch run vs the v3b same-protocol numbers.
    assert res["delta_recovery_pearson_canonical"] >= res["delta_recovery_pearson_raw"], (
        res["delta_recovery_pearson_raw"], res["delta_recovery_pearson_canonical"]
    )
    assert res["delta_recovery_pearson_canonical"] >= 0.6
    assert abs(res["delta_hat_vs_alpha_star_confusion_canonical"]) <= 0.2
    assert abs(res["delta_hat_vs_alpha_star_confusion_canonical"]) <= abs(res["delta_hat_vs_alpha_star_confusion_raw"])
    assert res["alpha_sign_recovery_top32"] >= 0.5
    a = res["gamma_checks"]["pair_A"]
    assert a["sign_recovered"], a
    abs_d = abs(res["gamma_checks"]["pair_D_confuser"]["gamma_hat_mean"])
    abs_a = abs(a["gamma_hat_mean"])
    assert abs_d <= max(0.5 * abs_a, 0.15), (abs_d, abs_a)
