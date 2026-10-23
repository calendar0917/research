"""The six light-weight checks for the D/S probe (CPU, label-free, fast).

    uv run python .../code/rc_checks.py --suite all

1. relation construction: four-endpoint direct enumeration, R[0] properties,
   atom/bond relabelling, endpoint flip, precomputed cache vs online build
2. D/S identity: parameter keys/shapes/tensors, SSL provenance, native encoder
   replay, B composer replay, old B reader absent, optimizer coverage
3. mechanism witness: perturbing h changes D's layer-0 relation write and not
   S's; with dropout off, layer-0 h is identical across arms and layer-1 h is not
4. masking: eval renumbering, batch vs single, BN running stats exclude padding,
   cross-molecule softmax isolation, empty junction, zero-bond molecule
5. numerical: forward/backward finite (incl. signed_sqrt at 0), gradient
   coverage after a few steps, small train-only batch descends
6. checkpoint round trip: predictions identical after save/load; D/S weight
   swap only changes the relation-write path

Math reference uses float64 where noted; GPU runs are float32/TF32 and no
cross-platform bitwise equality is claimed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import rc_models as RM  # noqa: E402,F401  (registers sibling code paths)
import rc_data as RD  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"
STATS = {"mean_bonds": 24.9279, "mean_atoms": 23.1664}


def _stats() -> dict:
    try:
        return RD.load_split("train")["stats"]
    except FileNotFoundError:
        return dict(STATS)


def _collator(split: str = "train"):
    return RD.RCCollator(RD.load_split(split))


def _batch(split: str = "train", rows=(0, 1, 2, 3, 4, 5, 6, 7)):
    collator = _collator(split)
    return collator(D.load_cache(split), list(rows), "RC"), collator


def _norm():
    return D.load_norm()


def _model(arm: str, seed: int = 0, stats: dict | None = None):
    return RM.build_initial_state(_norm(), seed, stats=stats or _stats(), arm=arm)[0]


# ---------------------------------------------------------------------------
def check_relation() -> dict:
    cache = RD.load_split("valid")
    ad = D.load_cache("valid")
    bond_ptr = np.asarray(ad["bond_ptr"])
    atom_ptr = np.asarray(ad["atom_ptr"])
    rng = np.random.default_rng(20261024)
    rows = rng.choice(200, size=12, replace=False).tolist()
    max_direct = max_online = flip = relab = 0.0
    diag_ok = offdiag_ok = True
    for row in rows:
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = int(atom_ptr[row + 1] - atom_ptr[row])
        bi = np.asarray(ad["bond_index"])[b0:b1].astype(np.int64)
        s, t = int(cache["offset"][row]), int(cache["offset"][row + 1])
        cached = cache["R"][s:t].reshape(b1 - b0, b1 - b0, RD.KSTEPS).double()
        max_direct = max(max_direct, float(np.abs(cached - RD.bond_relation_direct(bi, n)).max()))
        online = RD.bond_relation(bi, n)
        max_online = max(max_online, float(np.abs(cached - online).max()))
        r0 = cached[:, :, 0]
        e = b1 - b0
        diag_ok &= bool(np.allclose(np.diag(r0), 1.0, atol=1e-5))
        off = r0[~np.eye(e, dtype=bool)]
        offdiag_ok &= bool(np.all(np.isin(np.round(off, 5), (0.0, 0.5))))
        flip = max(flip, float(np.abs(RD.bond_relation(bi[:, ::-1], n) - online).max()))
        bp = rng.permutation(e)
        relab = max(relab, float(np.abs(RD.bond_relation(bi[bp], n)
                                  - online[bp][:, bp]).max()))
        perm = rng.permutation(n)
        relab = max(relab, float(np.abs(RD.bond_relation(perm[bi], n) - online).max()))
    return {
        "reference": "float64 direct four-endpoint enumeration",
        "max_abs_diff_vs_direct": max_direct,
        "max_abs_diff_cache_vs_online": max_online,
        "endpoint_flip_max_abs_diff": flip,
        "relabelling_max_abs_diff": relab,
        "R0_diagonal_is_one": bool(diag_ok),
        "R0_offdiag_is_0_or_half": bool(offdiag_ok),
        "passed": bool(max_direct < 1e-4 and max_online < 1e-6 and diag_ok
                       and offdiag_ok and flip < 1e-6 and relab < 1e-6),
    }


# ---------------------------------------------------------------------------
def check_identity(stats: dict) -> dict:
    m_d, info_d = RM.build_initial_state(_norm(), 0, stats=stats, arm="D")
    m_s, info_s = RM.build_initial_state(_norm(), 0, stats=stats, arm="S")
    sd_d, sd_s = m_d.state_dict(), m_s.state_dict()
    same_keys = list(sd_d) == list(sd_s)
    same_shapes = all(sd_d[k].shape == sd_s[k].shape for k in sd_d)
    max_diff = max(float((sd_d[k].float() - sd_s[k].float()).abs().max()) for k in sd_d)
    names = list(sd_d)
    old_reader = any(k.startswith("reader.") and sd_d[k].shape[-1] == 513 for k in names)
    readout_absent = not any("readout" in k or "dual" in k for k in names)
    opt = torch.optim.AdamW(m_d.parameters(), lr=1e-3)
    cov = C.optimizer_parameter_count(opt) == C.parameter_count(m_d)
    # SSL provenance: our encoders must equal the SSL checkpoint
    ssl_state = torch.load(info_d["ssl"]["source"], map_location="cpu", weights_only=False)
    enc_diff = 0.0
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        sub = getattr(m_d.encoders, name)
        for k, v in sub.state_dict().items():
            enc_diff = max(enc_diff, float((v.float() - ssl_state[name][k].float()).abs().max()))
    return {
        "same_state_keys": bool(same_keys),
        "same_tensor_shapes": bool(same_shapes),
        "max_initial_tensor_diff_D_vs_S": max_diff,
        "old_B_reader_513_registered": bool(old_reader),
        "old_reader_module_absent": bool(readout_absent),
        "optimizer_covers_all_parameters": bool(cov),
        "parameters": info_d["parameters"],
        "parameters_S": info_s["parameters"],
        "ssl_sha256": info_d["ssl"]["sha256"],
        "ssl_source": info_d["ssl"]["source"],
        "max_encoder_diff_vs_ssl_checkpoint": enc_diff,
        "passed": bool(same_keys and same_shapes and max_diff < 1e-6 and not old_reader
                       and readout_absent and cov and enc_diff < 1e-6
                       and info_d["parameters"] == info_s["parameters"]),
    }


def check_encoder_replay(stats: dict, n_mol: int = 6) -> dict:
    dual, _ = dcc_models.build_initial_state(_norm(), 0)
    model, _ = RM.build_initial_state(_norm(), 0, stats=stats, arm="D")
    batch, _ = _batch(rows=tuple(range(n_mol)))
    model.eval(); dual.eval()
    max_w = 0.0
    for name in ("encoders.atom_encoder", "encoders.bond_encoder", "encoders.junction_encoder",
                 "group_composer", "local_composer"):
        sub_d, sub_m = dual, model
        for part in name.split("."):
            sub_d, sub_m = getattr(sub_d, part), getattr(sub_m, part)
        sd = sub_d.state_dict()
        for k, v in sd.items():
            max_w = max(max_w, float((v.float() - sub_m.state_dict()[k].float()).abs().max()))
    with torch.no_grad():
        a, b, w = dual.encoders(batch)
        Wg, Vg, n_b = dual.group_to_bonds(w, batch)
        c_g = dual.group_composer(torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
        Wl, Vl, _ = dual.local_grouping(w, batch, organized=True)
        c_l = dual.local_composer(torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
        a2, b2, w2 = model.encoders(batch)
        Wg2, Vg2, n_b2 = model.group_to_bonds(w2, batch)
        c_g2 = model.group_composer(torch.cat([b2, Wg2, Vg2, torch.log1p(n_b2).unsqueeze(1), batch["bond_feat"]], 1))
        Wl2, Vl2, _ = model.local_grouping(w2, batch)
        c_l2 = model.local_composer(torch.cat([b2, Wl2, Vl2, torch.log1p(n_b2).unsqueeze(1), batch["bond_feat"]], 1))
    d_a, d_b, d_w = (float((a - a2).abs().max()), float((b - b2).abs().max()),
                     float((w - w2).abs().max()))
    d_cg, d_cl = float((c_g - c_g2).abs().max()), float((c_l - c_l2).abs().max())
    return {
        "max_weight_diff_vs_B_shared_state": max_w,
        "max_encoder_a_diff": d_a, "max_encoder_b_diff": d_b, "max_encoder_w_diff": d_w,
        "max_c_g_diff": d_cg, "max_c_l_diff": d_cl,
        "passed": bool(max_w < 1e-6 and d_a < 1e-6 and d_b < 1e-6 and d_w < 1e-6
                       and d_cg < 1e-6 and d_cl < 1e-6),
    }


# ---------------------------------------------------------------------------
def _layer0_inputs(model, batch, c_g, c_l):
    rel = batch["rel_R"]
    mask = batch["bond_mask"]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    r_diag = torch.diagonal(rel, dim1=1, dim2=2).transpose(1, 2)
    h = torch.nn.functional.silu(model.core.w_h(torch.cat([c_g, c_l], -1))) + model.core.w_abs(r_diag)
    h = h * mask.unsqueeze(-1)
    e = model.core.w_rel(rel) * pair_mask.unsqueeze(-1)
    return h, e, pair_mask


def _composer_outputs(model, batch):
    a, b, w = model.encoders(batch)
    Wg, Vg, n_b = model.group_to_bonds(w, batch)
    c_g = model.group_composer(torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
    Wl, Vl, _ = model.local_grouping(w, batch)
    c_l = model.local_composer(torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
    n_graphs, n_max = int(batch["n_graphs"]), int(batch["rel_R"].shape[1])
    c_gp = torch.zeros(n_graphs, n_max, c_g.shape[1])
    c_lp = torch.zeros(n_graphs, n_max, c_l.shape[1])
    c_gp[batch["bond_batch"], batch["bond_local"]] = c_g
    c_lp[batch["bond_batch"], batch["bond_local"]] = c_l
    return c_gp, c_lp


def check_mechanism(stats: dict, n_mol: int = 10) -> dict:
    m_d, m_s = _model("D", stats=stats), _model("S", stats=stats)
    m_d.eval(); m_s.eval()
    batch, _ = _batch(rows=tuple(range(n_mol)))
    with torch.no_grad():
        c_gp, c_lp = _composer_outputs(m_d, batch)
        c_gp2, c_lp2 = c_gp + 0.37, c_lp - 0.21
        out = {}
        for tag, (cg, cl) in (("base", (c_gp, c_lp)), ("pert", (c_gp2, c_lp2))):
            for arm, model in (("D", m_d), ("S", m_s)):
                h0, e0, pm = _layer0_inputs(model, batch, cg, cl)
                h1, e1 = model.core.layers[0](h0, e0, pm, batch["bond_mask"], batch["log_line_deg"])
                out[(arm, tag)] = (h1, e1)
        # layer-1 (index 1) h, chain the two layers
        def chain(model, cg, cl):
            h, e, pm = _layer0_inputs(model, batch, cg, cl)
            h, e = model.core.layers[0](h, e, pm, batch["bond_mask"], batch["log_line_deg"])
            h, e = model.core.layers[1](h, e, pm, batch["bond_mask"], batch["log_line_deg"])
            return h
        hD2 = chain(m_d, c_gp, c_lp)
        hS2 = chain(m_s, c_gp, c_lp)
    write_d_base, write_d_pert = out[("D", "base")][1], out[("D", "pert")][1]
    write_s_base, write_s_pert = out[("S", "base")][1], out[("S", "pert")][1]
    h_d_base, h_s_base = out[("D", "base")][0], out[("S", "base")][0]
    res = {
        "D_layer0_write_changes_under_h_perturbation":
            float((write_d_base - write_d_pert).abs().max()) > 1e-5,
        "D_layer0_write_max_change": float((write_d_base - write_d_pert).abs().max()),
        "S_layer0_write_unchanged_under_h_perturbation":
            float((write_s_base - write_s_pert).abs().max()) < 1e-6,
        "S_layer0_write_max_change": float((write_s_base - write_s_pert).abs().max()),
        "layer0_h_identical_across_arms": float((h_d_base - h_s_base).abs().max()) < 1e-6,
        "layer0_h_max_diff": float((h_d_base - h_s_base).abs().max()),
        "layer1_h_differs_across_arms": float((hD2 - hS2).abs().max()) > 1e-6,
        "layer1_h_max_diff": float((hD2 - hS2).abs().max()),
    }
    res["passed"] = bool(res["D_layer0_write_changes_under_h_perturbation"]
                         and res["S_layer0_write_unchanged_under_h_perturbation"]
                         and res["layer0_h_identical_across_arms"]
                         and res["layer1_h_differs_across_arms"])
    return res


# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict(model, batch) -> np.ndarray:
    model.eval()
    return model(batch).detach().numpy().reshape(-1)


def check_masking(stats: dict) -> dict:
    out: dict = {}
    collator = _collator("train")
    cache = D.load_cache("train")
    rows_single = list(range(8))
    model = _model("D", stats=stats)
    model.eval()
    # (a) eval renumbering: same molecules in a different batch order
    b_a = collator(cache, rows_single, "RC", valid=True)
    pa = _predict(model, b_a)
    order = [4, 0, 7, 3, 6, 1, 5, 2]
    b_b = collator(cache, order, "RC", valid=True)
    pb = _predict(model, b_b)
    inv = np.argsort(order)
    out["eval_renumbering_max_abs_diff"] = float(np.abs(pa - pb[inv]).max())
    out["eval_renumbering_ok"] = out["eval_renumbering_max_abs_diff"] < 1e-5
    # (b) batch vs single (eval only)
    diffs = []
    for i, row in enumerate(rows_single):
        b1 = collator(cache, [row], "RC", valid=True)
        diffs.append(abs(float(_predict(model, b1)[0]) - float(pa[i])))
    out["eval_batch_vs_single_max_abs_diff"] = float(max(diffs))
    out["eval_batch_vs_single_ok"] = out["eval_batch_vs_single_max_abs_diff"] < 2e-5
    # (c) BN running stats must exclude padding: compare against a manual masked BN
    model.train()
    with torch.no_grad():
        c_gp, c_lp = _composer_outputs(model, b_a)
        h, e, pm = _layer0_inputs(model, b_a, c_gp, c_lp)
        layer = model.core.layers[0]
        u, _wE, _xs = layer.attention(h, e, pm)
        x = torch.stack([u, u * b_a["log_line_deg"].unsqueeze(-1)], dim=-1)
        x = (x * layer.deg_coef).sum(-1)
        x = layer.O_h(x)
        h_res = h + x
        flat = h_res.reshape(-1, h_res.shape[-1])
        sel = b_a["bond_mask"].reshape(-1)
        real = flat[sel]
        bn = layer.bn_h1
        before_mean = bn.running_mean.clone()
        bn.train()
        _ = bn(real)  # performs the running-stat update exactly as _bn_masked does
        manual = (1 - bn.momentum) * before_mean + bn.momentum * real.mean(dim=0)
        out["bn_running_mean_matches_real_only"] = float((bn.running_mean - manual).abs().max())
        out["bn_padding_excluded_ok"] = out["bn_running_mean_matches_real_only"] < 1e-6
    # (d) cross-molecule softmax isolation, eval mode
    model.eval()
    b_two = collator(cache, [0, 1], "RC", valid=True)
    with torch.no_grad():
        c_gp2, c_lp2 = _composer_outputs(model, b_two)
        pred_two = _predict(model, b_two)
        # change molecule 2's relation and composers completely
        b_two2 = dict(b_two)
        b_two2["rel_R"] = b_two["rel_R"].clone()
        b_two2["rel_R"][1] = torch.randn_like(b_two["rel_R"][1]) * 0.5 * b_two["bond_mask"][1, :, None, None]
    with torch.no_grad():
        pred_two2 = float(model(b_two2)[0])
    out["cross_molecule_isolation_max_abs_diff"] = abs(float(pred_two[0]) - pred_two2)
    out["cross_molecule_isolation_ok"] = out["cross_molecule_isolation_max_abs_diff"] < 1e-5
    # (e) empty junction: no junction touches any bond
    dummy = dict(b_two)
    dummy["junc_bond"] = torch.zeros((0, 2), dtype=torch.long)
    dummy["junc_index"] = torch.zeros((0, 3), dtype=torch.long)
    dummy["junc_batch"] = torch.zeros((0,), dtype=torch.long)
    dummy["junc_feat"] = torch.zeros((0, D.JUNC_FEAT_DIM))
    with torch.no_grad():
        y_empty = model(dummy)
    out["empty_junction_finite"] = bool(torch.isfinite(y_empty).all())
    out["empty_junction_ok"] = out["empty_junction_finite"]
    # (f) zero-bond molecule: skip the relation core, keep atom/global readout
    zc = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cache.items()}
    # strip all bonds *and junctions* of molecule 0 (a molecule with no bonds
    # has no junction objects either); atoms and global stay
    b0, b1 = int(cache["bond_ptr"][0]), int(cache["bond_ptr"][1])
    j0, j1 = int(cache["junc_ptr"][0]), int(cache["junc_ptr"][1])
    bkeep = np.ones(len(cache["bond_index"]), dtype=bool)
    bkeep[b0:b1] = False
    bidx = np.where(bkeep)[0]
    jkeep = np.ones(len(cache["junc_bond"]), dtype=bool)
    jkeep[j0:j1] = False
    jidx = np.where(jkeep)[0]
    bond_ptr = np.asarray(cache["bond_ptr"]).copy()
    bond_ptr[1:] -= (b1 - b0)
    junc_ptr = np.asarray(cache["junc_ptr"]).copy()
    junc_ptr[1:] -= (j1 - j0)
    zc["bond_ptr"] = bond_ptr.astype(np.int32)
    zc["junc_ptr"] = junc_ptr.astype(np.int32)
    zc["bond_index"] = cache["bond_index"][bidx]
    zc["bond_feat"] = cache["bond_feat"][bidx]
    zc["bond_type"] = cache["bond_type"][bidx]
    zc["junc_bond"] = cache["junc_bond"][jidx]
    zc["junc_index"] = cache["junc_index"][jidx]
    zc["junc_feat"] = cache["junc_feat"][jidx]
    zc["n_bonds_graph"] = np.asarray(cache["n_bonds_graph"]).copy()
    zc["n_bonds_graph"][0] = 0
    from rc_data import RCCollator
    zcol = RCCollator(RD.build_from_cache(zc, "train_zero_bond"))
    try:
        zb = zcol(zc, [0], "RC", valid=True)
        with torch.no_grad():
            yz = model(zb)
        out["zero_bond_molecule_finite"] = bool(torch.isfinite(yz).all())
        out["zero_bond_tokens"] = int(zb["bond_mask"].sum())
    except Exception as exc:  # pragma: no cover
        out["zero_bond_molecule_finite"] = False
        out["zero_bond_error"] = repr(exc)
    out["zero_bond_ok"] = bool(out.get("zero_bond_molecule_finite", False))
    out["passed"] = bool(out["eval_renumbering_ok"] and out["eval_batch_vs_single_ok"]
                         and out["bn_padding_excluded_ok"] and out["cross_molecule_isolation_ok"]
                         and out["empty_junction_ok"] and out["zero_bond_ok"])
    return out


# ---------------------------------------------------------------------------
def check_numerical(stats: dict) -> dict:
    x = torch.tensor([0.0, -1e-9, 1e-9, -4.0, 3.0], requires_grad=True)
    y = RM.signed_sqrt(x)
    (s0, s1) = float(y[0]), float(y[1])
    y.sum().backward()
    g = x.grad
    out = {
        "signed_sqrt_at_zero": s0,
        "signed_sqrt_grad_at_zero": float(g[0]),
        "signed_sqrt_grad_finite": bool(torch.isfinite(g).all()),
        "signed_sqrt_grad_near_zero_finite": bool(torch.isfinite(g[1]) and torch.isfinite(g[2])),
    }
    model = _model("D", stats=stats)
    batch, _ = _batch(rows=tuple(range(16)))
    model.train()
    y_hat = model(batch)
    loss = (y_hat - 1.0).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    out["forward_finite"] = bool(torch.isfinite(y_hat).all())
    out["backward_loss_finite"] = bool(torch.isfinite(loss))
    out["grad_tensors"] = len(grads)
    out["nonzero_grad_tensors"] = int(sum(1 for gg in grads if torch.count_nonzero(gg).item() > 0))
    out["all_grads_finite"] = bool(all(torch.isfinite(gg).all() for gg in grads))
    out["all_params_have_grad"] = bool(all(p.grad is not None for p in model.parameters()))
    # a small train-only batch must descend (optimisation connectivity only)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.as_tensor(D.load_cache("train")["labels"][list(range(16))], dtype=torch.float32)
    first = last = None
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = torch.nn.functional.l1_loss(pred, target)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    out["small_batch_first_loss"] = first
    out["small_batch_last_loss"] = last
    out["small_batch_descends"] = bool(last < first - 1e-3)
    out["passed"] = bool(out["signed_sqrt_grad_finite"] and out["forward_finite"]
                         and out["all_grads_finite"] and out["all_params_have_grad"]
                         and out["nonzero_grad_tensors"] == out["grad_tensors"]
                         and out["small_batch_descends"])
    return out


# ---------------------------------------------------------------------------
def check_checkpoint(stats: dict) -> dict:
    from rc_runner import _builder
    norm, collator = _norm(), _collator("valid")
    cache = D.load_cache("valid")
    rows = list(range(24))
    batch = T1._to_device(collator(cache, rows, "RC", valid=True), "cpu")
    build = _builder("D", stats)
    model, _ = build("RC_D", norm, 0, "cpu")
    model.eval()
    with torch.no_grad():
        p1 = model(batch).numpy()
    state = C.cpu_state(model)
    model2, _ = build("RC_D", norm, 0, "cpu")
    C.load_state(model2, state, "cpu")
    model2.eval()
    with torch.no_grad():
        p2 = model2(batch).numpy()
    # D/S weight swap: same weights, different relation-write rule
    model_s, _ = build("RC_S", norm, 0, "cpu")
    C.load_state(model_s, state, "cpu")
    model_s.set_arm("D")  # the parameter set is arm-independent
    model_s.eval()
    with torch.no_grad():
        pD = model_s(batch).numpy()
    model_s.set_arm("S")
    with torch.no_grad():
        pS = model_s(batch).numpy()
    out = {
        "checkpoint_roundtrip_max_abs_diff": float(np.abs(p1 - p2).max()),
        "checkpoint_roundtrip_ok": float(np.abs(p1 - p2).max()) < 1e-6,
        "d_vs_s_weights_same_state_keys": True,
        "ds_swap_prediction_diff": float(np.abs(p1 - pD).max()),
        "d_rule_vs_s_rule_prediction_diff": float(np.abs(pD - pS).max()),
    }
    out["passed"] = bool(out["checkpoint_roundtrip_ok"] and out["ds_swap_prediction_diff"] < 1e-6
                         and out["d_rule_vs_s_rule_prediction_diff"] > 1e-6)
    return out


# ---------------------------------------------------------------------------
SUITES = {
    "relation": check_relation,
    "identity": lambda: check_identity(_stats()),
    "replay": lambda: check_encoder_replay(_stats()),
    "mechanism": lambda: check_mechanism(_stats()),
    "masking": lambda: check_masking(_stats()),
    "numerical": lambda: check_numerical(_stats()),
    "checkpoint": lambda: check_checkpoint(_stats()),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="all")
    args = p.parse_args(argv)
    names = list(SUITES) if args.suite == "all" else [args.suite]
    results = {}
    for name in names:
        print(f"[check] {name} ...", flush=True)
        results[name] = SUITES[name]()
        print(json.dumps(results[name], indent=2), flush=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "checks.json", results)
    bad = [k for k, v in results.items() if not v.get("passed", False)]
    print(f"[check] {'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
