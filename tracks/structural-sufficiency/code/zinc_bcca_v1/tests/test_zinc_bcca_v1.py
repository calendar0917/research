"""CPU tests for the zinc bond-conditioned context attention round.

These tests need the frozen SSL encoder / standardiser but no ZINC labels and no
official test data.  They are outside the root ``testpaths``; run explicitly:

    uv run pytest -q tracks/structural-sufficiency/code/zinc_bcca_v1/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPO = CODE.parents[3]
for p in (CODE, REPO / "tracks/structure-contribution/code",
          REPO / "tracks/structural-sufficiency/code/phase2b"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bcca_common as C  # noqa: E402
import bcca_models as BM  # noqa: E402
import bcca_runner as R  # noqa: E402
import phase2a_runner as R2A  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

MOLS = [o for _, o in R2A._synthetic_molecules()]
NORM = None
SSL0 = None


def _norm():
    global NORM
    if NORM is None:
        NORM = C.load_norm()
    return NORM


def _ssl0():
    global SSL0
    if SSL0 is None:
        SSL0, _ = C.ssl_state(0)
    return SSL0


def _build(arm: str, seed: int = 0):
    _, model, info = BM.build_variant(arm, seed=seed, norm=_norm(), ssl_state=_ssl0())
    model.eval()
    return model, info


def _batch(objs):
    return sro.batch_objects(list(objs))


# ---------------------------------------------------------------------------
# frozen configuration
# ---------------------------------------------------------------------------


def test_frozen_config():
    cfg = C.load_config()
    assert cfg["protocol_id"] == "zinc-bond-conditioned-context-attention-v1"
    assert cfg["round"] == "zinc_bond_conditioned_context_attention_v1"
    assert cfg["kind"] == "attention_feasibility_and_mechanism_screen"
    assert cfg["data"]["train_rows"] == 10000
    assert cfg["data"]["valid_rows"] == 1000
    assert cfg["data"]["official_test_rows_accessed"] == 0
    assert cfg["recipe"]["epochs"] == 360
    assert cfg["recipe"]["updates"] == 28440
    assert cfg["recipe"]["batch_size"] == 128
    assert cfg["recipe"]["shuffle_offset"] == 101
    assert cfg["recipe"]["early_stopping"] is False
    assert cfg["architecture"]["new_parameters"] == 6144
    assert cfg["architecture"]["total_parameters"] == 419649
    assert cfg["thresholds"]["performance_mean_gain"] == 0.004
    assert cfg["thresholds"]["conditioning_mean_gain"] == 0.002
    assert cfg["budget"]["gpu_hours_cap"] == 4.0
    assert cfg["budget"]["planned_formal_runs"] == 6
    assert cfg["label_provenance"]["test"]["accessed"] is False
    assert set(cfg["arms"]) == {"P3", "F", "G", "C"}
    for arm in ("F", "G", "C"):
        assert cfg["arms"][arm]["total_parameters"] == 419649
        assert cfg["arms"][arm]["seeds"] == [0, 1]


def test_protocol_declares_three_context_identities():
    cfg = C.load_config()
    m = cfg["architecture"]["context_mapping"]
    assert "sum_j A_ej" in m["C"]
    assert "p_j" in m["G"]
    assert "b + U SiLU(T LN(b))" in m["F"]


# ---------------------------------------------------------------------------
# parameters / init identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["P3", "F", "G", "C"])
def test_parameter_counts_and_audits(arm):
    model, info = _build(arm)
    audit = BM.parameter_audit(model)
    assert audit["actual_parameters"] == C.EXPECTED_PARAMETERS[arm]
    assert audit["parameter_exact"]
    assert audit["reader_in_dim"] == 321
    assert len(model.reader.lins) == 3
    assert BM.module_audit(model)["pass"]
    assert info["shared_matches_base"]
    assert info["global_rng_untouched"]


def test_C_G_bitwise_identity_and_shared_K_T():
    c, ci = _build("C")
    g, gi = _build("G")
    f, fi = _build("F")
    cs, gs = c.state_dict(), g.state_dict()
    assert set(cs) == set(gs)
    for k in cs:
        assert torch.equal(cs[k], gs[k]), k
    assert torch.equal(c.attn.k.weight, g.attn.k.weight)
    assert torch.equal(f.ffn.t.weight, c.attn.k.weight)
    assert float(c.attn.q.weight.abs().sum()) == 0.0
    assert float(g.attn.q.weight.abs().sum()) == 0.0
    assert float(f.ffn.u.weight.abs().sum()) == 0.0
    assert ci["shared_state_sha256"] == gi["shared_state_sha256"]
    assert fi["shared_state_sha256"] == ci["shared_state_sha256"]


def test_k_init_range_matches_linear_default():
    c, _ = _build("C")
    w = c.attn.k.weight.detach()
    bound = 1.0 / np.sqrt(96.0)
    assert float(w.abs().max()) <= bound + 1e-7
    assert float(w.std()) > 0.0


# ---------------------------------------------------------------------------
# zero-init forward identity
# ---------------------------------------------------------------------------


def test_zero_init_forward_matches_p3():
    batch = _batch(MOLS)
    p3, _ = _build("P3")
    outs = {}
    with torch.no_grad():
        base_pred = p3(batch)
        base_feats = p3.forward_features(batch)
        for arm in ("C", "G", "F"):
            model, _ = _build(arm)
            pred = model(batch)
            feats = model.forward_features(batch)
            outs[arm] = {
                "pred": float((pred - base_pred).abs().max()),
                "W": float((feats["W"] - base_feats["W"]).abs().max()),
                "V": float((feats["V"] - base_feats["V"]).abs().max()),
                "W_rel": float((feats["W"] - base_feats["W"]).abs().max()
                               / base_feats["W"].abs().max()),
                "V_rel": float((feats["V"] - base_feats["V"]).abs().max()
                               / base_feats["V"].abs().max()),
                "n_b": float((feats["n_b"] - base_feats["n_b"]).abs().max()),
            }
    for arm, v in outs.items():
        assert v["pred"] <= 5e-5, (arm, v)
        assert v["W_rel"] <= 1e-5, (arm, v)
        assert v["V_rel"] <= 1e-5, (arm, v)
        assert v["n_b"] == 0.0, (arm, v)


def test_F_keeps_p3_w_and_only_changes_composer_input():
    batch = _batch(MOLS)
    p3, _ = _build("P3")
    f, _ = _build("F")
    with torch.no_grad():
        fp = p3.forward_features(batch)
        ff = f.forward_features(batch)
    assert torch.equal(fp["w"], ff["w"])
    assert torch.equal(fp["W"], ff["W"])
    assert torch.equal(fp["V"], ff["V"])
    assert float((ff["b_compose"] - ff["b"]).abs().max()) == 0.0  # U = 0
    assert torch.equal(ff["b"], fp["b"])


# ---------------------------------------------------------------------------
# toy algebra
# ---------------------------------------------------------------------------


def test_toy_moment_conservation_and_allocation():
    from bcca_runner import _toy_attention, _toy_system, _nonlinear_readout

    attn = _toy_attention(seed=3)
    b, w, batch, n_b = _toy_system()
    with torch.no_grad():
        Wc, Vc, aux_c = attn(b, w, batch, n_b, mode="C", return_attention=True)
        Wg, Vg, aux_g = attn(b, w, batch, n_b, mode="G", return_attention=True)
    cons = BM.moment_conservation(Wc, Vc, Wg, Vg, batch)
    assert cons["W_max_relative_error"] <= 1e-10
    assert cons["V_max_relative_error"] <= 1e-10
    safe = n_b.clamp_min(1.0).unsqueeze(1)
    assert bool(((Vc / safe) >= (Wc / safe) ** 2 - 1e-10).all())
    A, p = aux_c["A"], aux_g["p"]
    row, col = aux_c["row"], aux_c["col"]
    per_bond = torch.zeros(int(n_b.numel()), dtype=torch.float64)
    per_bond.index_add_(0, row, (A - p[col]).abs())
    assert float(per_bond.max()) > 0.1
    disp_c = BM.content_dispersion(Wc, Vc, n_b, batch)
    disp_g = BM.content_dispersion(Wg, Vg, n_b, batch)
    assert disp_c["W_content_dispersion"] > 0.0
    assert disp_g["W_content_dispersion"] <= 1e-12
    # linear total moment is conserved; a nonlinear readout separates the maps
    assert abs(float(Wc.sum() - Wg.sum())) <= 1e-9
    assert abs(_nonlinear_readout(Wc, Vc) - _nonlinear_readout(Wg, Vg)) > 0.0


def test_pair_index_has_no_cross_molecule_pairs():
    batch = _batch(MOLS)
    idx = BM.pair_index(batch)
    row, col = idx
    assert torch.equal(batch["bond_batch"][row], batch["junc_batch"][col])
    jcount = torch.bincount(batch["junc_batch"], minlength=batch["n_graphs"])
    expected = int(sum(jcount[batch["bond_batch"]]))
    assert int(row.numel()) == expected
    # each physical bond appears once per junction of its molecule
    assert int(torch.bincount(row, minlength=batch["bond_feat"].shape[0]).min()) == int(jcount.min())


def test_empty_junction_and_single_bond_are_safe():
    tiny = sro.build_mol_objects(
        -1, np.asarray([0, 1])[:, None], np.asarray([[0, 1], [1, 0]]), np.asarray([0, 0])
    )
    batch = _batch([tiny])
    for arm in ("C", "G", "F"):
        model, _ = _build(arm)
        with torch.no_grad():
            pred, feats = model(batch, return_features=True, return_attention=(arm != "F"))
        assert torch.isfinite(pred).all()
        assert float(feats["W"].abs().sum()) == 0.0
        assert float(feats["V"].abs().sum()) == 0.0
        if arm != "F":
            assert int(feats["attn"]["A"].numel()) == 0


# ---------------------------------------------------------------------------
# symmetry / isolation
# ---------------------------------------------------------------------------


def test_atom_bond_junction_and_batch_symmetry():
    batch = _batch(MOLS)
    perm_mols = [MOLS[2], MOLS[0], MOLS[1]]
    perm_batch = _batch(perm_mols)
    rng = np.random.default_rng(0)
    for arm in ("C", "G", "F"):
        model, _ = _build(arm)
        with torch.no_grad():
            p0 = model(batch)
            p1 = model(perm_batch)
            # 5e-5: float32 scatter-add reordering; the frozen P3 arm shows the
            # same order of magnitude on these tiny synthetic molecules.
            assert float((p0[[2, 0, 1]] - p1).abs().max()) <= 5e-5
            for obj in MOLS:
                rel = R2A._reindex_objects(obj, rng.permutation(obj.n_atoms))
                d = abs(float(model(_batch([obj]))[0]) - float(model(_batch([rel]))[0]))
                assert d <= 5e-5
                bp = R._permute_bonds(obj, rng.permutation(obj.bond_index.shape[0]))
                d = abs(float(model(_batch([obj]))[0]) - float(model(_batch([bp]))[0]))
                assert d <= 5e-5
                jp = R._permute_junctions(obj, rng.permutation(obj.junc_index.shape[0]))
                d = abs(float(model(_batch([obj]))[0]) - float(model(_batch([jp]))[0]))
                assert d <= 5e-5


def test_cross_molecule_isolation():
    a, b = MOLS[0], MOLS[1]
    two = _batch([a, b])
    pert = sro.MolObjects(**{**a.__dict__})
    af = pert.atom_feat.copy()
    af[:, sro.ATOM_CATEGORIES: sro.ATOM_CATEGORIES + sro.ATOM_DEGREE_SLOTS] += 1.0
    pert.atom_feat = af
    two_p = _batch([pert, b])
    nb = _batch([a])["bond_feat"].shape[0]
    for arm in ("C", "G", "F"):
        model, _ = _build(arm)
        with torch.no_grad():
            W0 = model.forward_features(two)["W"]
            W1 = model.forward_features(two_p)["W"]
            p0 = model(two)
            p1 = model(two_p)
        assert float((W0[nb:] - W1[nb:]).abs().max()) == 0.0
        assert float(abs(p0[1] - p1[1])) == 0.0
        assert float((W0[:nb] - W1[:nb]).abs().max()) > 0.0


def test_n_e_counts_physical_bonds_once():
    batch = _batch(MOLS)
    n_b = BM.bond_junction_counts(batch)
    base = BM.BondContextBJOC(src_config(), np.zeros(sro.GLOBAL_DIM), np.ones(sro.GLOBAL_DIM), "P3")
    with torch.no_grad():
        w = torch.zeros((batch["junc_feat"].shape[0], 96))
        _, _, n_ref = base.group_to_bonds(w, batch, "null")
    assert torch.equal(n_b, n_ref)
    assert float(n_b.sum()) == 2.0 * int(batch["junc_index"].shape[0])


def src_config():
    import structural_relation_composer_v1 as src

    return src.BJOCConfig()


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


def test_multistep_gradient_expectations():
    batch = _batch(MOLS)
    target = torch.linspace(-1.0, 1.0, int(batch["n_graphs"]))
    for arm in ("C", "G", "F"):
        model, _ = _build(arm)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        names = [n for n, _ in model.named_parameters() if n.startswith(("attn.", "ffn."))]
        rows = []
        for step in range(4):
            opt.zero_grad()
            loss = F.l1_loss(model(batch), target)
            loss.backward()
            rows.append({n: (float(p.grad.norm()) if p.grad is not None else None)
                         for n, p in model.named_parameters() if n in names})
            opt.step()
        if arm in ("C", "G"):
            assert rows[0]["attn.q.weight"] > 0.0
            assert rows[0]["attn.k.weight"] == 0.0
            assert rows[1]["attn.k.weight"] > 0.0
        else:
            assert rows[0]["ffn.u.weight"] > 0.0
            assert rows[0]["ffn.t.weight"] == 0.0
            assert rows[1]["ffn.t.weight"] > 0.0
        assert all(v > 0.0 for v in rows[-1].values())


# ---------------------------------------------------------------------------
# attention statistics / diagnostics
# ---------------------------------------------------------------------------


def test_attention_diagnostics_finite_and_dispersion():
    batch = _batch(MOLS)
    c, _ = _build("C")
    g, _ = _build("G")
    with torch.no_grad():
        _, fc = c(batch, return_features=True, return_attention=True)
        _, fg = g(batch, return_features=True, return_attention=True)
    sc = BM.attention_diagnostics(fc["attn"], batch)
    sg = BM.attention_diagnostics(fg["attn"], batch)
    for s in (sc, sg):
        assert s["D_molecule_mean"] is not None and np.isfinite(s["D_molecule_mean"])
        assert np.isfinite(s["entropy_norm_mean"])
        assert 0.0 <= s["entropy_norm_mean"] <= 1.0 + 1e-6
        assert 0.0 < s["max_weight_mean"] <= 1.0
    dc = BM.content_dispersion(fc["W"], fc["V"], fc["n_b"], batch)
    dg = BM.content_dispersion(fg["W"], fg["V"], fg["n_b"], batch)
    assert np.isfinite(dc["W_content_dispersion"])
    assert dg["W_content_dispersion"] <= 1e-6


def test_mechanism_prediction_names_and_freeze_paths_distinct():
    for name in ("C", "G"):
        for mode in ("C", "G", "uniform"):
            p = C.mech_pred_path(name, 0, mode)
            assert p.name == f"mech_pred_{name}_s0_{mode}.npz"
    assert C.valid_pred_path("C", 1).name == "valid_pred_C_s1.npz"
    assert C.train_path("G", 0).name == "train_G_s0.json"


# ---------------------------------------------------------------------------
# frozen-loop parity (small, P3 mode)
# ---------------------------------------------------------------------------


def test_frozen_loop_parity_on_synthetic_slice():
    import bjoc_perfmech_long_v1 as bp

    synth = [o for _, o in R2A._synthetic_molecules()]
    y = np.linspace(-1.0, 1.0, len(synth))
    ref = bp.train_supervised(
        arm="P3", seed=0, recipe_key="R3", init="ssl", train_objs=synth,
        rows=list(range(len(synth))), y_fit=y, norm=_norm(), device=torch.device("cpu"),
        ssl_state=_ssl0(), model_kind="base", smoke_epochs=2,
    )
    mine = R._train_loop(
        arm="P3", seed=0, train_objs=synth, rows=list(range(len(synth))), y_fit=y,
        norm=_norm(), device=torch.device("cpu"), smoke_epochs=2, save_soup=False, diag_rows=None,
    )
    max_state = max(float((ref["soup_state"][k] - mine["soup_state"][k]).abs().max())
                    for k in ref["soup_state"])
    assert max_state <= 1e-4
    assert ref["batch_order"]["full_hash"] == mine["batch_order"]["full_hash"]
    assert ref["updates"] == mine["updates"]
    assert abs(ref["soup_fit_mae"] - mine["soup_fit_mae"]) <= 1e-5


def test_runner_has_no_official_test_path():
    src = (CODE / "bcca_runner.py").read_text(encoding="utf-8")
    scan = R._label_isolation_scan(src)
    assert scan["pass"], scan
    for fn in ("cmd_train", "cmd_predict_valid", "cmd_mechanism", "cmd_smoke"):
        assert scan[fn]["label_loader_calls"] == []
        assert scan[fn]["split_calls"] == []


def test_protocol_document_matches_frozen_config():
    """The frozen prose and the frozen YAML must not drift."""
    cfg = C.load_config()
    text = C.PROTOCOL_PATH.read_text(encoding="utf-8")
    plain = text.replace("*", "").replace("`", "").lower()
    assert cfg["protocol_id"] in text
    assert cfg["status"] == "FROZEN_PRE_GPU"
    assert "TBD" not in text and "TODO" not in text
    assert "not phase 2c" in plain
    assert "official test" in plain
    # counts and thresholds, formatted as they appear in the prose
    assert "413,505" in text and "419,649" in text and "6,144" in text
    assert "28,440" in text and "360" in text
    assert f"{cfg['thresholds']['performance_mean_gain']:.3f}" == "0.004"
    assert f"{cfg['thresholds']['conditioning_mean_gain']:.3f}" == "0.002"
    assert f"{cfg['budget']['gpu_hours_cap']:.1f}" == "4.0"
    assert str(cfg["recipe"]["batch_size"]) in text
    assert str(cfg["recipe"]["shuffle_offset"]) in text
    assert "20261019" in text  # bootstrap seed
    for seed_mae in cfg["audit_expectations"]["historical_valid_mae"].values():
        assert f"{float(seed_mae):.17f}" in text or f"{float(seed_mae)}" in text
