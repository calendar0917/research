"""Erratum gates for zinc-global-operator-search-v1 (broadcast bug in MixA).

Only the checks that decide whether the erratum and the two new arms are sound.
Every check runs the *production* code path: ``MixA.mix`` / ``MixA._attend``
inside the real ``Backbone``.  No restatement of the operator is tested in
isolation, and no label from official test is read.

E0  the mode switch adds/removes/reorders no parameter: four modes built from the
    same seed have identical state_dict keys/shapes/hashes, and the copied
    canonical initial state reproduces the historical ho_A_s0 init hash
E1  float64: legacy attention == plain dot-product attention; ``l2`` ==
    softmax(-||q-k||^2/(2 sqrt(d)) + bias); the two differ materially
E2  q=0, two keys with different norms, zero bias: legacy weights are equal,
    ``l2`` weights are not (and match the closed form)
E3  the exact decomposition legacy = mass * normalised(gated) weights, and GN
    keeps the attention weights of legacy while removing the total gain
E4  float64 gradient of the final loss for the new modes (module level and full
    model level, autograd vs central differences)
E5  frozen ho_A_s0 checkpoint: legacy vs pure-dot predictions on real training
    structures, and legacy reproduces the recorded valid predictions
E6  one forward for train and eval: dropout-free train/eval identity and
    ``ad_train.evaluate`` == direct forward, for the new modes
E7  the 9,000/1,000 split is the same split the historical ho_A_s0 used

Run: python gos_errata_checks.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import gos_data as GD  # noqa: E402
import gos_models as GM  # noqa: E402
import gos_runner as GR  # noqa: E402

OUT = _EXP / "analysis" / "errata_checks.json"
HISTORICAL_INIT_SHA = "4b8c59f8eb54f6a0d2d63422fd92dfe9644466e8bbddcc4925316c0aa936ffd9"
RES: dict[str, object] = {}


def gate(name, ok, **info):
    RES[name] = {"ok": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(info)[:500], flush=True)
    if not ok:
        raise SystemExit(f"gate failed: {name}")


def softmax64(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def fd_grad(loss_fn, param, idx, eps=1e-6):
    """Central difference of ``loss_fn`` w.r.t. ``param[idx]`` (in place, fp64)."""
    with torch.no_grad():
        orig = param.data[idx].item()
        param.data[idx] = orig + eps
        lp = float(loss_fn())
        param.data[idx] = orig - eps
        lm = float(loss_fn())
        param.data[idx] = orig
    return (lp - lm) / (2.0 * eps)


def main() -> int:
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rel_train, rel_valid, scale, mean_nb = GR.prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    norm = D.load_norm()
    RES["mean_train_n_bonds"] = mean_nb
    RES["mix_modes"] = list(GM.MIX_MODES)

    # E0 canonical init ---------------------------------------------------
    states, hashes, counts, keysets = {}, {}, {}, {}
    for mode in GM.MIX_MODES:
        m, info = GM.build_model("A", norm, 0, torch.device("cpu"),
                                 mean_n_bonds=mean_nb, mix_mode=mode)
        states[mode] = C.cpu_state(m)
        hashes[mode] = info["initial_state_sha256"]
        counts[mode] = int(info["parameters"])
        keysets[mode] = {k: tuple(v.shape) for k, v in states[mode].items()}
    same_keys = len({repr(keysets[m]) for m in GM.MIX_MODES}) == 1
    same_hash = len(set(hashes.values())) == 1
    # explicit "copy the canonical initial state, only switch the arithmetic"
    canon = states["legacy"]
    worst_copy = 0.0
    for mode in GM.MIX_MODES:
        m, _ = GM.build_model("A", norm, 0, torch.device("cpu"), mean_n_bonds=mean_nb,
                              mix_mode=mode)
        m.load_state_dict(canon)
        worst_copy = max(worst_copy, max(float((C.cpu_state(m)[k] - canon[k]).abs().max())
                                         for k in canon))
        # a second construction of the same builder (trainer vs holdout path)
        m2, info2 = GR.build_arm_model("A", 0, torch.device("cpu"), mix=mode,
                                       mean_n_bonds=mean_nb, norm=norm)
        worst_copy = max(worst_copy, max(float((C.cpu_state(m2)[k] - canon[k]).abs().max())
                                         for k in canon))
    gate("E0_same_init_across_modes",
         same_keys and same_hash and worst_copy == 0.0 and len(set(counts.values())) == 1,
         parameters=counts, init_sha=hashes["legacy"],
         historical_ho_A_s0_init_sha=HISTORICAL_INIT_SHA,
         matches_historical=bool(hashes["legacy"] == HISTORICAL_INIT_SHA),
         max_abs_copy_error=worst_copy,
         state_keys=int(len(canon)))
    gate("E0_init_reproduces_historical", hashes["legacy"] == HISTORICAL_INIT_SHA,
         got=hashes["legacy"], expected=HISTORICAL_INIT_SHA)

    # E1 float64 broadcast semantics on the real operator -----------------
    d, heads, rel_dim = 8, 2, 6
    B, N = 2, 6
    g = torch.Generator().manual_seed(11)
    # ONE canonical parameter draw, copied into every mode: the same design as the
    # two training arms, so cross-mode statements isolate the arithmetic only.
    base_mod = GM.MixA(d=d, heads=heads, rel=rel_dim, mode="legacy").double().eval()
    with torch.no_grad():
        for p in base_mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.5)
        base_mod.norm.alpha.zero_()
        base_mod.norm.beta.fill_(1.0)
    canon_mod = C.cpu_state(base_mod)
    mats = {}
    for mode in GM.MIX_MODES:
        m = GM.MixA(d=d, heads=heads, rel=rel_dim, mode=mode).double().eval()
        m.load_state_dict(canon_mod)
        mats[mode] = m
    h = torch.randn(B, N, d, generator=g, dtype=torch.float64)
    rel = torch.randn(B, N, N, rel_dim, generator=g, dtype=torch.float64)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 4:] = False                       # padding on the second molecule
    out = {mode: mats[mode].mix(h, rel, mask) for mode in GM.MIX_MODES}
    # ``gnlog`` returns the *gated* weights, so the score references below apply to
    # the score-carrying modes only.
    ATT_MODES = ("legacy", "dot", "l2", "gn")

    ref = {}
    for mode in ATT_MODES:
        q, k, v = out[mode]["q"], out[mode]["k"], out[mode]["v"]
        bias = mats[mode].beta(rel).permute(0, 3, 1, 2)
        sq = math.sqrt(mats[mode].hd)
        dot_logit = torch.einsum("bhid,bhjd->bhij", q, k) / sq + bias
        l2_logit = (-(q.unsqueeze(3) - k.unsqueeze(2)).pow(2).sum(-1) / (2.0 * sq)
                    + bias)           # up to a per-query constant 0.5||q||^2/sqrt(d)
        vmask = mask[:, None, None, :]
        ref[mode] = {
            "dot_attn": softmax64(dot_logit.masked_fill(~vmask, -1e30)),
            "l2_attn": softmax64(l2_logit.masked_fill(~vmask, -1e30)),
        }
    e1 = {}
    for mode in ATT_MODES:
        a = out[mode]["attn"]
        e1[mode] = {
            "vs_dot": float((a - ref[mode]["dot_attn"]).abs().max()),
            "vs_l2": float((a - ref[mode]["l2_attn"]).abs().max()),
        }
    gate("E1_legacy_equals_dot_attention", e1["legacy"]["vs_dot"] < 1e-12,
         max_abs_diff=e1["legacy"]["vs_dot"], **_scale(e1["legacy"]))
    gate("E1_l2_equals_key_energy_attention", e1["l2"]["vs_l2"] < 1e-12,
         max_abs_diff=e1["l2"]["vs_l2"], **_scale(e1["l2"]))
    gate("E1_legacy_is_not_the_l2_operator", e1["legacy"]["vs_l2"] > 1e-3,
         legacy_vs_l2_max_abs_diff=e1["legacy"]["vs_l2"],
         l2_vs_legacy_max_abs_diff=e1["l2"]["vs_dot"])
    RES["E1_attention_reference_diffs"] = e1
    # per-row constant does not survive the softmax (numerical statement)
    with torch.no_grad():
        row_const = torch.randn(B, heads, N, 1, generator=g, dtype=torch.float64) * 7.0
        base_logit = out["legacy"]["logits"]
        a2 = torch.softmax(base_logit + row_const.expand(-1, -1, -1, N), dim=-1)
        a_shift = out["legacy"]["attn"]
        gate("E1_row_constant_cancels", float((a2 - a_shift).abs().max()) < 1e-12,
             note="adding any per-query constant to the legacy logits returns the same weights",
             max_abs_diff=float((a2 - a_shift).abs().max()))

    # E2 q=0, two keys with different norms -------------------------------
    d2, hd2 = 4, 4
    q0 = torch.zeros(1, 1, 2, hd2, dtype=torch.float64)                 # [B,H,q,hd]
    k0 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                       [3.0, 0.0, 0.0, 0.0]], dtype=torch.float64).reshape(1, 1, 2, hd2)
    v0 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64).reshape(1, 1, 2, hd2)
    rel0 = torch.zeros(1, 2, 2, rel_dim, dtype=torch.float64)
    mask0 = torch.ones(1, 2, dtype=torch.bool)
    e2 = {}
    for mode in ("legacy", "dot", "l2"):
        m = GM.MixA(d=d2, heads=1, rel=rel_dim, mode=mode).double().eval()
        with torch.no_grad():
            m.beta.weight.zero_(); m.beta.bias.zero_()
            m.gamma.weight.zero_(); m.gamma.bias.zero_()
            a = m._attend(q0, k0, v0, rel0, mask0)["attn"]
        e2[mode] = [float(x) for x in a.reshape(-1)]
    expected_l2 = softmax64(-k0.pow(2).sum(-1).unsqueeze(2) / (2.0 * math.sqrt(hd2)))
    e2["expected_l2"] = [float(x) for x in expected_l2.reshape(-1)]
    legacy_equal = max(abs(x - 0.5) for x in e2["legacy"]) < 1e-12
    l2_unequal = min(e2["l2"][0] - e2["l2"][1], e2["l2"][2] - e2["l2"][3]) > 0.05
    l2_matches = max(abs(a - b) for a, b in zip(e2["l2"], e2["expected_l2"])) < 1e-12
    gate("E2_zero_query_key_norms",
         legacy_equal and l2_unequal and l2_matches
         and all(abs(x - 0.5) < 1e-12 for x in e2["dot"]),
         note="rows are (query0 key0, query0 key1, query1 key0, query1 key1)",
         legacy_weights=e2["legacy"], l2_weights=e2["l2"],
         closed_form=e2["expected_l2"], dot_weights=e2["dot"])

    # E3 decomposition legacy = mass * normalised weights -----------------
    a_leg, a_gn = out["legacy"]["attn"], out["gn"]["attn"]
    mass_leg, mass_gn = out["legacy"]["mass"], out["gn"]["mass"]
    ctx_leg = out["legacy"]["ctx"] / mass_leg
    ctx_gn = out["gn"]["ctx"]
    scale_ctx = max(1e-30, float(out["legacy"]["ctx"].abs().max()))
    d_attn = float((a_leg - a_gn).abs().max())
    d_ctx = float((ctx_leg - ctx_gn).abs().max()) / scale_ctx
    gate("E3_gate_normalisation_decomposition", d_attn < 1e-15 and d_ctx < 1e-12,
         note="same parameters in every mode; legacy = mass * normalised weights",
         attn_legacy_vs_gn_max_abs=d_attn,
         relative_ctx_diff=d_ctx,
         mass_range=[float(mass_leg.min()), float(mass_leg.max())])
    # E3b gnlog == gn in exact arithmetic (fp64, real operator) -----------
    rel_ctx_gn = max(1e-30, float(out["gn"]["ctx"].abs().max()))
    d_gn_log = float((out["gn"]["ctx"] - out["gnlog"]["ctx"]).abs().max()) / rel_ctx_gn
    # effective (normalised, gated) key weights of gn vs the softmax of gnlog
    eff_gn = out["gn"]["attn"] * out["gn"]["gate"] / out["gn"]["mass"]
    d_attn_gn_log = float((eff_gn - out["gnlog"]["attn"]).abs().max())
    gate("E3b_gnlog_equals_gn", d_gn_log < 1e-12 and d_attn_gn_log < 1e-12,
         relative_ctx_diff=d_gn_log, effective_weight_max_abs_diff=d_attn_gn_log)

    # E3c why gnlog exists: the division form is ill-conditioned when the gate
    # suppresses every key (this is what killed the first GN run at epoch 284).
    def _grad_probe(mode, bias):
        m = GM.MixA(d=d, heads=heads, rel=rel_dim, mode=mode).double()
        with torch.no_grad():
            for p_ in m.parameters():
                p_.copy_(torch.randn(p_.shape, generator=torch.Generator().manual_seed(3),
                                     dtype=torch.float64) * 0.4)
            m.norm.alpha.zero_(); m.norm.beta.fill_(1.0)
            m.gamma.weight.zero_(); m.gamma.bias.fill_(bias)
        hh = torch.randn(1, 6, d, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        rr = torch.randn(1, 6, 6, rel_dim, generator=torch.Generator().manual_seed(5),
                         dtype=torch.float64)
        mm = torch.ones(1, 6, dtype=torch.bool)
        yv = torch.randn(1, 6, d, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
        res_m = m.mix(hh, rr, mm)
        loss = F.l1_loss(m(hh, rr, mm), yv)
        loss.backward()
        g = float(m.gamma.bias.grad.abs().max())
        return {"max_abs_gamma_grad": g,
                "ctx_finite": bool(torch.isfinite(res_m["ctx"]).all()),
                "loss_finite": bool(torch.isfinite(loss))}

    probes = {}
    for bias in (-2.0, -25.0, -45.0):
        probes[f"gamma_bias_{bias}"] = {mode: _grad_probe(mode, bias) for mode in ("gn", "gnlog")}
    ok_stable = all(v["gnlog"]["ctx_finite"] and v["gnlog"]["loss_finite"]
                    and math.isfinite(v["gnlog"]["max_abs_gamma_grad"])
                    for v in probes.values())
    def _bad(v):
        return (not v["ctx_finite"]) or (not v["loss_finite"]) or \
            (not math.isfinite(v["max_abs_gamma_grad"]))

    gn_bad = any(_bad(v["gn"]) for v in probes.values())
    gnlog_ok = all(not _bad(v["gnlog"]) for v in probes.values())
    gate("E3c_gnlog_is_stable_where_gn_breaks",
         gn_bad and gnlog_ok,
         note=("same parameters/inputs; 'gn' is the definition as written "
               "(w/sum(w)), 'gnlog' the log-domain rewrite"),
         probes=probes)

    RES["E3_gate_mass_stats"] = {
        "min": float(mass_leg.min()), "max": float(mass_leg.max()),
        "std_over_queries": float(mass_leg.std()),
        "frac_outside_0p8_1p2": float(((mass_leg < 0.8) | (mass_leg > 1.2)).double().mean()),
    }

    # E4 gradients of the new operators -----------------------------------
    gm = torch.Generator().manual_seed(5)
    h4 = torch.randn(1, 5, d, generator=gm, dtype=torch.float64)
    rel4 = torch.randn(1, 5, 5, rel_dim, generator=gm, dtype=torch.float64)
    mask4 = torch.tensor([[True, True, True, True, False]])
    y4 = torch.randn(1, 5, d, generator=gm, dtype=torch.float64)
    e4 = {}
    for mode in ("legacy", "l2", "gn", "gnlog"):
        m = GM.MixA(d=d, heads=heads, rel=rel_dim, mode=mode).double()
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=gm, dtype=torch.float64) * 0.4)
            m.norm.alpha.zero_(); m.norm.beta.fill_(1.0)

        def loss_fn():
            return F.l1_loss(m(h4, rel4, mask4), y4)

        m.zero_grad(set_to_none=True)
        loss_fn().backward()
        worst = 0.0
        probes = []
        layer = m
        named = [("gamma.weight", layer.gamma.weight, (0, 1)),
                 ("beta.weight", layer.beta.weight, (1, 2)),
                 ("qkv.weight", layer.qkv.weight, (2, 3)),
                 ("out.weight", layer.out.weight, (1, 0)),
                 ("norm.alpha", layer.norm.alpha, (0,))]
        for name, param, idx in named:
            g_auto = float(param.grad[idx])
            g_fd = fd_grad(loss_fn, param, idx)
            rel = abs(g_auto - g_fd) / max(1e-12, abs(g_auto), abs(g_fd))
            worst = max(worst, rel)
            probes.append({"param": name, "idx": list(idx),
                           "autograd": g_auto, "fd": g_fd, "rel_err": rel})
        e4[mode] = {"worst_rel_err": worst, "probes": probes}
        gate(f"E4_module_gradient_{mode}", worst < 1e-6, worst_rel_err=worst,
             n_probes=len(probes))
    # full model: final L1 loss on real rows, float64
    rows = np.arange(16, dtype=np.int64)
    y = torch.tensor(np.asarray(train_cache["labels"], dtype=np.float64)[rows])
    for mode in ("l2", "gn"):
        model, _ = GR.build_arm_model("A", 2, torch.device("cpu"), mix=mode,
                                      mean_n_bonds=mean_nb, norm=norm)
        model.eval().double()
        b = collate(train_cache, rows, "GOS", valid=True)
        b64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in b.items()}
        af = b64["atom_feat"].clone().requires_grad_(True)
        b2 = dict(b64); b2["atom_feat"] = af
        loss = F.l1_loss(model(b2).view(-1), y)
        loss.backward()
        ap = np.asarray(train_cache["atom_ptr"])
        counts = np.diff(ap)[rows]
        off = int(np.concatenate([[0], np.cumsum(counts)[:-1]])[2])
        j = off + 1
        grad = float(af.grad[j, 3])
        # atom_feat is an input, not a parameter: perturb the tensor directly
        with torch.no_grad():
            eps = 1e-6
            fp = b64["atom_feat"].clone(); fp[j, 3] += eps
            fm = b64["atom_feat"].clone(); fm[j, 3] -= eps
            bp = dict(b64); bp["atom_feat"] = fp
            bm = dict(b64); bm["atom_feat"] = fm
            lp = float(F.l1_loss(model(bp).view(-1), y))
            lm = float(F.l1_loss(model(bm).view(-1), y))
        fd = (lp - lm) / (2 * eps)
        rel = abs(grad - fd) / max(1e-12, abs(grad), abs(fd))
        e4[f"model_{mode}"] = {"autograd": grad, "fd": fd, "rel_err": rel}
        gate(f"E4_model_gradient_{mode}", rel < 1e-4, autograd=grad, fd=fd, rel_err=rel)
    RES["E4_gradients"] = e4

    # E5 frozen ho_A_s0 checkpoint: legacy vs pure dot --------------------
    ck_path = GR.RUNS / "ho_A_s0" / "best.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    row64 = np.arange(64, dtype=np.int64)
    b = collate(train_cache, row64, "GOS", valid=True)
    preds = {}
    attn0 = {}
    for mode in ("legacy", "dot", "l2", "gn", "gnlog"):
        m, _ = GR.build_arm_model("A", 0, torch.device("cpu"), mix=mode,
                                  mean_n_bonds=mean_nb, norm=norm)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        preds[mode] = m(b).detach().numpy()
        layer0 = {}
        hooks = [m.layers[0].register_forward_hook(
            lambda mod, inp, outp, store=layer0: store.update(
                {"attn": mod.mix(inp[0], inp[1], inp[2])["attn"].detach()}))]
        m(b)
        for hk in hooks:
            hk.remove()
        attn0[mode] = layer0["attn"]
    d_legacy_dot = float(np.abs(preds["legacy"] - preds["dot"]).max())
    pred_scale = float(np.abs(preds["legacy"]).max())
    d_attn_legacy_dot = float((attn0["legacy"] - attn0["dot"]).abs().max())
    gate("E5_frozen_ckpt_legacy_equals_dot",
         d_legacy_dot / max(pred_scale, 1e-12) < 1e-5 and d_attn_legacy_dot < 1e-6,
         n_rows=int(len(row64)),
         max_abs_pred_diff=d_legacy_dot, pred_scale=pred_scale,
         max_abs_attn_diff_layer0=d_attn_legacy_dot)
    RES["E5_operator_swap_magnitude"] = {
        "legacy_vs_l2_pred_max_abs": float(np.abs(preds["legacy"] - preds["l2"]).max()),
        "legacy_vs_gn_pred_max_abs": float(np.abs(preds["legacy"] - preds["gn"]).max()),
        "legacy_vs_gnlog_pred_max_abs": float(np.abs(preds["legacy"] - preds["gnlog"]).max()),
        "gn_vs_gnlog_pred_max_abs": float(np.abs(preds["gn"] - preds["gnlog"]).max()),
        "note": "inference-time swap only, NOT a result; the trained arms are separate runs",
    }
    # legacy path reproduces the historical recorded valid predictions
    rec = np.load(GR.RUNS / "ho_A_s0" / "predictions" / "valid_best.npz")
    m, _ = GR.build_arm_model("A", 0, torch.device("cpu"), mix="legacy",
                              mean_n_bonds=mean_nb, norm=norm)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    rows_valid = np.arange(int(valid_cache["labels"].shape[0]), dtype=np.int64)
    p_legacy, mae = T1.evaluate(m, valid_cache, rows_valid, candidate="GOS",
                                device=torch.device("cpu"), batch_size=256, collate_fn=collate)
    d_rec = float(np.abs(p_legacy - np.asarray(rec["pred"]).reshape(-1)).max())
    # the recorded npz was produced on the A100 in fp32; a CPU recompute is not
    # bit-identical, so this is a float-noise tolerance, not an equality test.
    gate("E5_legacy_reproduces_recorded_valid",
         d_rec < 1e-5 and abs(mae - float(ck["valid_mae"])) < 1e-8,
         max_abs_pred_diff_vs_recorded=d_rec,
         valid_mae_recomputed=float(mae), valid_mae_stored=float(ck["valid_mae"]))
    RES["E5_checkpoint"] = {
        "path": str(ck_path.relative_to(C.REPO)), "epoch": int(ck.get("epoch", -1)),
        "valid_mae": float(ck["valid_mae"]),
        "sha256": C.sha256_file(ck_path),
    }

    # E5b the trained gate mass is not a constant (motivation for GN) -----
    row256 = np.arange(256, dtype=np.int64)
    b256 = collate(valid_cache, row256, "GOS", valid=True)
    m_leg, _ = GR.build_arm_model("A", 0, torch.device("cpu"), mix="legacy",
                                 mean_n_bonds=mean_nb, norm=norm)
    m_leg.load_state_dict(ck["state_dict"])
    m_leg.eval()
    masses = {}
    hooks = []
    for li, layer in enumerate(m_leg.layers):
        hooks.append(layer.register_forward_hook(
            lambda mod, inp, outp, store=masses, key=f"layer{li}":
            store.__setitem__(key, mod.mix(inp[0], inp[1], inp[2])["mass"].detach())))
    m_leg(b256)
    for hk in hooks:
        hk.remove()
    mass_stats = {}
    for key, mval in masses.items():
        v = mval.reshape(-1)
        mass_stats[key] = {"mean": float(v.mean()), "std": float(v.std()),
                           "min": float(v.min()), "max": float(v.max()),
                           "frac_lt_0p9_or_gt_1p1": float(((v < 0.9) | (v > 1.1)).double().mean())}
    finite = all(math.isfinite(s[k]) for s in mass_stats.values()
                 for k in ("mean", "std", "min", "max"))
    std_ok = all(s["std"] > 1e-4 for s in mass_stats.values())
    gate("E5b_trained_gate_mass_varies", finite and std_ok,
         note="total gain of the trained legacy operator is not a constant per query",
         n_rows=int(len(row256)), per_layer=mass_stats)

    # E6 one forward for train and eval ----------------------------------
    for mode in ("l2", "gn", "gnlog"):
        m, _ = GR.build_arm_model("A", 3, torch.device("cpu"), mix=mode,
                                  mean_n_bonds=mean_nb, norm=norm)
        m.load_state_dict(ck["state_dict"])
        m.train()
        p_train = m(b).detach().clone()
        m.eval()
        p_eval = m(b).detach().clone()
        p_t1, _ = T1.evaluate(m, train_cache, row64, candidate="GOS",
                              device=torch.device("cpu"), batch_size=64, collate_fn=collate)
        d_train_eval = float((p_train - p_eval).abs().max())
        d_eval_engine = float(np.abs(p_eval.numpy() - p_t1).max())
        gate(f"E6_single_forward_{mode}",
             d_train_eval == 0.0 and d_eval_engine < 1e-6,
             train_vs_eval_max_abs=d_train_eval,
             direct_vs_evaluate_max_abs=d_eval_engine)

    # E7 same internal split as the historical holdout run ----------------
    keep, hold = GR.holdout_split(1000)
    recorded = C.read_json(_EXP / "analysis" / "holdout_split_1000.json")
    gate("E7_same_internal_split",
         C.sha256_array(hold) == recorded["holdout_rows_sha256"] and len(keep) == 9000,
         holdout_sha=C.sha256_array(hold), recorded=recorded["holdout_rows_sha256"],
         n_keep=int(len(keep)), n_hold=int(len(hold)))

    # E8 the two new arms: the recorded numbers come from the trained operator --
    arm_specs = {"L2G": ("l2g_A_s0", "l2"), "GN": ("gn_A_s0", "gnlog")}
    _, hold_rows = GR.holdout_split(1000)
    e8 = {}
    for label, (run, mix) in arm_specs.items():
        art = C.read_json(GR.RUNS / run / "artifacts.json")
        ck_arm = torch.load(GR.RUNS / run / "best.pt", map_location="cpu", weights_only=False)
        rec = C.read_json(_EXP / "analysis" / f"holdout_A_{mix}_s0.json")
        out = {}
        for mode in {mix, "legacy"}:
            m, _ = GR.build_arm_model("A", 0, torch.device("cpu"), mix=mode,
                                      mean_n_bonds=mean_nb, norm=norm)
            m.load_state_dict(ck_arm["state_dict"])
            _, v_mae = T1.evaluate(m, valid_cache, rows_valid, candidate="GOS",
                                   device=torch.device("cpu"), batch_size=256,
                                   collate_fn=collate)
            _, h_mae = T1.evaluate(m, train_cache, hold_rows, candidate="GOS",
                                   device=torch.device("cpu"), batch_size=256,
                                   collate_fn=collate)
            out[mode] = {"valid": float(v_mae), "holdout": float(h_mae)}
        e8[label] = {
            "run": run, "trained_mix": mix, "epoch": int(ck_arm.get("epoch", -1)),
            "recorded_valid": float(art["best_valid"]["valid_mae"]),
            "recorded_holdout": float(rec["holdout_mae"]),
            "recomputed": out,
        }
        ok = (abs(out[mix]["valid"] - float(art["best_valid"]["valid_mae"])) < 1e-6
              and abs(out[mix]["holdout"] - float(rec["holdout_mae"])) < 1e-6)
        gate(f"E8_recomputed_{label}", ok,
             trained_mix_valid=out[mix]["valid"], recorded_valid=out[mix] and
             float(art["best_valid"]["valid_mae"]),
             trained_mix_holdout=out[mix]["holdout"],
             recorded_holdout=float(rec["holdout_mae"]),
             legacy_path_holdout=out["legacy"]["holdout"],
             note="same checkpoint re-read with the trained arithmetic vs the legacy one")
    RES["E8_arm_recompute"] = e8

    OUT.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT, RES)
    print("wrote", OUT)
    return 0


def _scale(d):
    return {"max_abs_diff_vs_dot": d["vs_dot"], "max_abs_diff_vs_l2": d["vs_l2"]}


if __name__ == "__main__":
    raise SystemExit(main())
