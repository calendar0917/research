"""Zero-GPU failure diagnosis for SCM v1 at the frozen revision (1730990).

No training, no test split, no new ranks / distances / normalisations /
initialisations, no linear probe.  Forward-only measurements on the two frozen
best checkpoints plus one algebraic identity of the frozen implementation.

The correction is split exactly (``delta = bias + S + A``) and each part is pushed
through the *production* injection point (the hook on ``local_composer``), then
measured at three levels:

1. raw per-bond magnitude      ``rms(delta) / rms(c_loc)``
2. pooled channel magnitude    ``dzl = sum(delta)``, ``dzl2 = sum(2c*delta + delta^2)``
3. prediction displacement     exact prediction change at fixed trained weights

Displacements are reported both as ``pred(x) - pred(0)`` (marginal, from the
correction-off operating point) and as ``pred(all) - pred(all - x)`` (drop from
the real operating point).  The reader Jacobian d pred / d z is exact, so the
first-order response splits into its ``zl`` and ``zl2`` parts and their
cancellation is measured rather than assumed.

The "correction off" prediction is a readout counterfactual at fixed trained
weights.  It is NOT an estimate of what retraining without the branch would give,
and is never interpreted that way here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import scm_common as S
import scm_relations as R
import scm_models as M

import ad_data as D
import ad_train as T1

ARMS = {"S-only": "s_only", "S+A": "s_plus_a"}
FULL = ("bias", "S", "A")
VARIANTS = FULL + ("all", "all_minus_bias", "all_minus_S", "all_minus_A")
WIDTH, GLOBAL_DIM = 96, 33
ZL = slice(3 * WIDTH + GLOBAL_DIM, 3 * WIDTH + GLOBAL_DIM + WIDTH)
ZL2 = slice(3 * WIDTH + GLOBAL_DIM + WIDTH, 3 * WIDTH + GLOBAL_DIM + 2 * WIDTH)


# ---------------------------------------------------------------------------
def load_arm(arm: str):
    norm = D.load_norm()
    model, _info = M.build_initial_state(norm, 0, use_cov=(arm == "S+A"))
    ckpt = torch.load(S.RUNS / f"{ARMS[arm]}_s0" / "best.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def blocks(model, cap, batch):
    """Per-bond delta split into bias / S / A, straight from the frozen Wout."""
    w = model.scm.out.weight.detach()
    b = model.scm.out.bias.detach()
    bb, bl = batch["bond_batch"], batch["bond_local"]
    s1, a1 = cap["S1"][bb, bl], cap["A1"][bb, bl]
    s2, a2 = cap["S2"][bb, bl], cap["A2"][bb, bl]
    return {"bias": b.unsqueeze(0).expand(s1.shape[0], -1),
            "S": s1 @ w[:, 0:8].T + s2 @ w[:, 16:24].T,
            "A": a1 @ w[:, 8:16].T + a2 @ w[:, 24:32].T}


def variant_delta(parts, name):
    if name == "all":
        return parts["bias"] + parts["S"] + parts["A"]
    if name == "all_minus_bias":
        return parts["S"] + parts["A"]
    if name == "all_minus_S":
        return parts["bias"] + parts["A"]
    if name == "all_minus_A":
        return parts["bias"] + parts["S"]
    return parts[name]


def pooled(c_loc, delta, batch, n_graphs):
    dzl = torch.zeros((n_graphs, c_loc.shape[1]), dtype=c_loc.dtype)
    dzl2 = torch.zeros_like(dzl)
    dzl.index_add_(0, batch["bond_batch"], delta)
    dzl2.index_add_(0, batch["bond_batch"], 2.0 * c_loc * delta + delta * delta)
    return dzl, dzl2


def coherent(per_bond, batch, n_graphs, width_scale=None):
    """L1-then-L2 pooled amplitude: what the sum would be with no cancellation."""
    out = torch.zeros((n_graphs, per_bond.shape[1]), dtype=per_bond.dtype)
    out.index_add_(0, batch["bond_batch"], per_bond.abs())
    return out.norm(dim=1)


def forward_capture_z(model, batch, delta):
    """Real forward with an injected correction; also returns the reader input z."""
    cache: dict = {}

    def hook(_m, _i, out):
        return out + delta

    handles = [model.dcc.local_composer.register_forward_hook(hook),
               model.dcc.reader.register_forward_pre_hook(
                   lambda _m, inp: cache.__setitem__("z", inp[0]))]
    try:
        with torch.no_grad():
            pred = model.dcc(batch)
    finally:
        for h in handles:
            h.remove()
    return pred, cache["z"]


def jacobian_at_reader(reader, z):
    """Exact d pred / d z at the trained weights (the reader is z's only consumer)."""
    leaf = z.detach().clone().requires_grad_(True)
    out = reader(leaf).view(-1)
    return out.detach(), torch.autograd.grad(out.sum(), leaf)[0].detach()


# ---------------------------------------------------------------------------
def structure_strata(store, cache, collate, n_mol):
    rows = np.arange(n_mol)
    keys = ("n_bonds", "n1_mean", "n2_mean", "n1_zero_frac", "n2_zero_frac",
            "n1_cv", "n2_cv")
    out = {k: np.zeros(n_mol) for k in keys}
    for start in range(0, n_mol, 256):
        chunk = rows[start:start + 256]
        batch = collate(cache, chunk, "P3N", valid=True)
        n1 = batch["mu1"].sum(-1).double()
        n2 = batch["mu2"].sum(-1).double()
        mask = batch["bond_mask"]
        for i, row in enumerate(chunk):
            m = mask[i]
            a, b = n1[i][m], n2[i][m]
            out["n_bonds"][row] = float(m.sum())
            out["n1_mean"][row] = float(a.mean())
            out["n2_mean"][row] = float(b.mean())
            out["n1_zero_frac"][row] = float((a == 0).double().mean())
            out["n2_zero_frac"][row] = float((b == 0).double().mean())
            out["n1_cv"][row] = float(a.std(unbiased=False) / a.mean().clamp(min=1e-9))
            out["n2_cv"][row] = float(b.std(unbiased=False) / b.mean().clamp(min=1e-9))
    return out


def strata_masks(st):
    n1z, n2z = st["n1_zero_frac"], st["n2_zero_frac"]
    cv, nb = st["n1_cv"] + st["n2_cv"], st["n_bonds"]
    return {
        "all": np.ones(len(nb), dtype=bool),
        "zero_relation_any_band": (n1z > 0) | (n2z > 0),
        "regular_relation_lowest_quartile_dispersion": cv <= np.quantile(cv, 0.25),
        "heterogeneous_relation_top_quartile": cv >= np.quantile(cv, 0.75),
        "low_bond_count_quartile": nb <= np.quantile(nb, 0.25),
        "high_bond_count_quartile": nb >= np.quantile(nb, 0.75),
    }


def first_order_r2(exact, first):
    e, f = np.asarray(exact, dtype=np.float64), np.asarray(first, dtype=np.float64)
    ss = ((e - e.mean()) ** 2).sum()
    return float(1 - ((e - f) ** 2).sum() / max(ss, 1e-30))


def cos_rows(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    den = np.clip(na * nb, 1e-30, None)
    return (a * b).sum(1) / den


def q(a):
    a = np.asarray(a, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std()),
            "p05": float(np.quantile(a, 0.05)), "p50": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)), "min": float(a.min()),
            "max": float(a.max())}


def diag_affine_r2(c, d):
    """Per-dimension affine fit delta_d ~ a_d*c_d + b_d, R^2 per dimension.

    Two parameters per dimension, no hyper-parameter, no cross-dimension mixing:
    this is the part of the correction that a per-dimension rescale of c_loc
    could already represent.
    """
    c, d = c.double(), d.double()
    cc = c - c.mean(0)
    dd = d - d.mean(0)
    cov = (cc * dd).mean(0)
    var_c = (cc ** 2).mean(0).clamp(min=1e-30)
    var_d = (dd ** 2).mean(0).clamp(min=1e-30)
    r2 = (cov ** 2) / (var_c * var_d)
    slope = cov / var_c
    return r2.clamp(min=0.0), slope


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="failure_mechanism.json")
    p.add_argument("--csv", default="failure_per_molecule.csv")
    args = p.parse_args(argv)

    cache = D.load_cache("valid")
    y = np.asarray(cache["labels"], dtype=np.float64)
    n_mol = int(cache["n_molecules"])
    store = R.load_split("valid")
    collate = R.make_split_collate({"valid": store}, T1.collate)
    st = structure_strata(store, cache, collate, n_mol)
    masks = strata_masks(st)
    rows = np.arange(n_mol)

    result = {
        "protocol_id": S.TRACK_ID,
        "revision": "1730990",
        "official_test_loaded": False,
        "no_training": True,
        "no_new_hyperparameters": True,
        "counterfactual_disclaimer":
            "pred(delta=0) is a readout counterfactual at fixed trained weights. "
            "It is not an estimate of what retraining without the branch would give.",
        "launch_gate_status": {
            "G5_registered_as_gate": True,
            "G5_pass": False,
            "frozen_decision_rule": "any gate failure -> STOP_BEFORE_GPU",
            "code_exemption": "scm_data_feasibility.py:304 excludes G5 from "
                              "'hard_failed_gates'; the exemption is not present in "
                              "the pre-registered text and was written after the "
                              "failure was observed and before the GPU run.",
            "gpu_round_under_protocol_deviation": True,
            "consequence": "the GPU round did not satisfy its own frozen launch "
                           "conditions; 'hard_failed_gates: []' must not be reported "
                           "as a full pass.  The deviation can only weaken a positive "
                           "claim, and this round is negative, so the direction of "
                           "the conclusion is unaffected.",
        },
        "algebraic_identity": {},
        "strata_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "structure": {k: q(v) for k, v in st.items()},
        "arms": {},
    }

    # ---- (1) identity: sum_e S[e,t] == sum_f (n_f - mean(n)) v_f ------------
    model0, _ck0 = load_arm("S-only")
    with torch.no_grad():
        batch0 = collate(cache, np.arange(96), "P3N", valid=True)
        _pred, cap0 = model0.forward_with_capture(batch0)
        v0 = model0.scm.v(cap0["c_loc"].float()).double()
        bb, bl = batch0["bond_batch"], batch0["bond_local"]
        scales = model0.scm.scale.double()
        for tag, key, n_full, sc in (
                ("t1", "S1", batch0["mu1"].sum(-1).double(), scales[0]),
                ("t2", "S2", batch0["mu2"].sum(-1).double(), scales[1])):
            n_t = n_full[bb, bl]
            s = cap0[key][bb, bl].double()
            sums = torch.zeros(int(batch0["n_graphs"]), s.shape[1], dtype=s.dtype)
            sums.index_add_(0, bb, s)
            abs_sums = torch.zeros_like(sums).index_add_(0, bb, s.abs())
            rhs = torch.zeros_like(sums)
            disp = []
            for g in range(int(batch0["n_graphs"])):
                m = bb == g
                rhs[g] = ((n_t[m] - n_t[m].mean()).unsqueeze(1) * v0[m]).sum(0) / sc
                disp.append(float(n_t[m].std(unbiased=False)))
            result["algebraic_identity"][tag] = {
                "mean_norm_sum_S": float(sums.norm(dim=1).mean()),
                "mean_norm_rhs": float(rhs.norm(dim=1).mean()),
                "max_abs_error": float((sums - rhs).abs().max()),
                "mean_norm_sum_abs_S": float(abs_sums.norm(dim=1).mean()),
                "cancel_share": float(sums.norm(dim=1).mean()
                                      / abs_sums.norm(dim=1).mean()),
                "corr_dispersion_vs_sum_norm": float(
                    np.corrcoef(disp, sums.norm(dim=1).numpy())[0, 1]),
                "n_molecules": int(batch0["n_graphs"]),
            }

    # ---- (2) per-arm measurement -------------------------------------------
    raw: dict[str, np.ndarray] = {}
    raw_zl, raw_zl2, raw_dzl, raw_dzl2 = {}, {}, {}, {}
    for arm in ARMS:
        model, ckpt = load_arm(arm)
        keys = (["pred_none", "pred_all", "zl_norm", "zl2_norm", "zl_l1"]
                + [f"pred_{v}" for v in VARIANTS if v != "all"]
                + [f"first_{v}" for v in VARIANTS]
                + [f"first_zl_{v}" for v in VARIANTS]
                + [f"first_zl2_{v}" for v in VARIANTS]
                + [f"dzl_{v}" for v in FULL + ("all",)]
        + [f"dzl_coh_{v}" for v in FULL + ("all",)]
        + [f"dzl2_coh_{v}" for v in FULL + ("all",)]
                + [f"dzl2_{v}" for v in FULL + ("all",)])
        per = {k: np.zeros(n_mol) for k in keys}
        raw_zl.setdefault(arm, np.zeros((n_mol, WIDTH)))
        raw_zl2.setdefault(arm, np.zeros((n_mol, WIDTH)))
        raw_dzl.setdefault(arm, {v: np.zeros((n_mol, WIDTH)) for v in FULL + ("all",)})
        raw_dzl2.setdefault(arm, {v: np.zeros((n_mol, WIDTH)) for v in FULL + ("all",)})
        pooled_c, pooled_d = [], []
        for start in range(0, n_mol, 128):
            chunk = rows[start:start + 128]
            batch = collate(cache, chunk, "P3N", valid=True)
            pred_full, cap = model.forward_with_capture(batch)
            c_loc = cap["c_loc"]
            parts = blocks(model, cap, batch)
            assert float((parts["bias"] + parts["S"] + parts["A"]
                          - cap["delta"]).abs().max()) < 1e-4, "block split mismatch"
            n_g = int(batch["n_graphs"])
            pred0, z = forward_capture_z(model, batch, torch.zeros_like(c_loc))
            pred_z, jac = jacobian_at_reader(model.dcc.reader, z)
            assert z.shape[1] == 513, z.shape
            assert float((pred_z - pred0).abs().max()) < 1e-5, "reader path mismatch"
            pred_hook = forward_capture_z(model, batch, cap["delta"])[0]
            assert float((pred_hook - pred_full).abs().max()) < 1e-5, \
                "injection path must equal the production SCM hook"
            j_zl, j_zl2 = jac[:, ZL], jac[:, ZL2]
            per["pred_none"][chunk] = pred0.numpy()
            per["pred_all"][chunk] = pred_full.numpy()
            per["zl_norm"][chunk] = z[:, ZL].norm(dim=1).numpy()
            per["zl2_norm"][chunk] = z[:, ZL2].norm(dim=1).numpy()
            per["zl_l1"][chunk] = z[:, ZL].abs().sum(1).numpy()
            raw_zl[arm][chunk] = z[:, ZL].numpy()
            raw_zl2[arm][chunk] = z[:, ZL2].numpy()
            for v in VARIANTS:
                d = variant_delta(parts, v)
                if v == "all":
                    pred = pred_full
                else:
                    pred = forward_capture_z(model, batch, d)[0]
                    per[f"pred_{v}"][chunk] = pred.numpy()
                dzl, dzl2 = pooled(c_loc, d, batch, n_g)
                per[f"first_{v}"][chunk] = ((j_zl * dzl).sum(1)
                                            + (j_zl2 * dzl2).sum(1)).numpy()
                per[f"first_zl_{v}"][chunk] = (j_zl * dzl).sum(1).numpy()
                per[f"first_zl2_{v}"][chunk] = (j_zl2 * dzl2).sum(1).numpy()
                if v in FULL + ("all",):
                    per[f"dzl_coh_{v}"][chunk] = coherent(d, batch, n_g).numpy()
                    per[f"dzl2_coh_{v}"][chunk] = coherent(
                        torch.cat([2.0 * c_loc * d, d * d], 1), batch, n_g,
                        width_scale=None).numpy()
                    per[f"dzl_{v}"][chunk] = dzl.norm(dim=1).numpy()
                    per[f"dzl2_{v}"][chunk] = dzl2.norm(dim=1).numpy()
                    raw_dzl[arm][v][chunk] = dzl.numpy()
                    raw_dzl2[arm][v][chunk] = dzl2.numpy()
            pooled_c.append(c_loc.detach())
            pooled_d.append(variant_delta(parts, "all").detach())

        C, Dm = torch.cat(pooled_c), torch.cat(pooled_d)
        r2_diag, slope_diag = diag_affine_r2(C, Dm)

        # ---- prediction displacements
        d_marg = {v: per[f"pred_{v}"] - per["pred_none"] for v in FULL}
        d_drop = {"bias": per["pred_all"] - per["pred_all_minus_bias"],
                  "S": per["pred_all"] - per["pred_all_minus_S"],
                  "A": per["pred_all"] - per["pred_all_minus_A"]}
        d_all = per["pred_all"] - per["pred_none"]
        mae = {v: float(np.abs(per[f"pred_{v}"] - y).mean())
               for v in ("none",) + tuple(VARIANTS)}
        mae["all"] = float(np.abs(per["pred_all"] - y).mean())
        resid_off = y - per["pred_none"]
        oracle = float(np.abs(per["pred_none"] + np.median(resid_off) - y).mean())

        # ---- pooling and cancellation
        def canc(v):
            num = np.abs(per[f"first_zl_{v}"] + per[f"first_zl2_{v}"])
            a = np.abs(per[f"first_zl_{v}"]).mean()
            bb_ = np.abs(per[f"first_zl2_{v}"]).mean()
            if a + bb_ <= 0:
                return None
            return {"median_abs_sum_over_sum_abs": float(np.median(num) / (a + bb_)),
                    "mean_abs_zl_term": float(a), "mean_abs_zl2_term": float(bb_)}

        readout = {
            "mae": mae,
            "mae_all_minus_none": mae["all"] - mae["none"],
            "oracle_single_constant_mae": oracle,
            "oracle_constant_share_of_gain": float(
                (mae["none"] - oracle) / max(mae["none"] - mae["all"], 1e-12)),
            "frac_molecules_improved_vs_off": float(
                (np.abs(y - per["pred_all"]) < np.abs(y - per["pred_none"])).mean()),
            "displacement_marginal_from_off": {v: q(d_marg[v]) for v in FULL},
            "displacement_drop_from_full": {v: q(d_drop[v]) for v in FULL},
            "displacement_all": q(d_all),
            "signal_to_constant_all": float(np.std(d_all) / max(abs(d_all.mean()), 1e-12)),
            "corr_dpred_all_vs_residual": float(np.corrcoef(d_all, resid_off)[0, 1]),
            "slope_dpred_on_residual": float(np.polyfit(resid_off, d_all, 1)[0]),
            "r2_dpred_on_residual": float(np.corrcoef(d_all, resid_off)[0, 1] ** 2),
            "cancellation": {v: canc(v) for v in FULL + ("all",)},
            "first_order_r2_vs_exact": {
                v: first_order_r2(d_marg[v] if v in FULL else d_all,
                                 per[f"first_{v}"])
                for v in FULL + ("all",)},
            "interaction_index_all": float(
                np.median(np.abs(d_all - d_marg["bias"] - d_marg["S"] - d_marg["A"]))
                / max(np.median(np.abs(d_all)), 1e-12)),
        }
        pooled_summary = {
            "dzl_over_zl_median": {v: float(np.median(per[f"dzl_{v}"] / per["zl_norm"]))
                                   for v in FULL + ("all",)},
            "dzl2_over_zl2_median": {v: float(np.median(per[f"dzl2_{v}"] / per["zl2_norm"]))
                                     for v in FULL + ("all",)},
            "dzl_norm": {v: q(per[f"dzl_{v}"]) for v in FULL + ("all",)},
            "dzl2_norm": {v: q(per[f"dzl2_{v}"]) for v in FULL + ("all",)},
            "dzl_coherent": {v: q(per[f"dzl_coh_{v}"]) for v in FULL + ("all",)},
            "dzl2_coherent": {v: q(per[f"dzl2_coh_{v}"]) for v in FULL + ("all",)},
            "dzl_survive_share_median": {
                v: float(np.median(per[f"dzl_{v}"] / per[f"dzl_coh_{v}"].clip(1e-9)))
                for v in FULL + ("all",)},
            "dzl2_survive_share_median": {
                v: float(np.median(per[f"dzl2_{v}"] / per[f"dzl2_coh_{v}"].clip(1e-9)))
                for v in FULL + ("all",)},
            "zl_norm": q(per["zl_norm"]),
            "zl2_norm": q(per["zl2_norm"]),
        }
        affine = {
            "r2_diag_per_dimension": {"mean": float(r2_diag.mean()),
                                      "p50": float(r2_diag.median()),
                                      "p05": float(r2_diag.min())},
            "share_dims_r2_above_0p5": float((r2_diag > 0.5).double().mean()),
            "slope_diag_abs_median": float(slope_diag.abs().median()),
            "note": "delta_d ~ a_d*c_d + b_d per dimension, 2 parameters, no "
                    "hyper-parameter.  This is the part of the correction that a "
                    "per-dimension rescale of the local channel already expresses.",
        }
        # collinearity of the pooled shift with the channel it shifts
        zl_m, zl2_m = raw_zl[arm], raw_zl2[arm]
        collinear = {}
        for v in FULL + ("all",):
            dzl_m, dzl2_m = raw_dzl[arm][v], raw_dzl2[arm][v]
            collinear[v] = {
                "median_cos_dzl_vs_zl": float(np.median(cos_rows(dzl_m, zl_m))),
                "median_r2_collinear_with_zl": float(np.median(cos_rows(dzl_m, zl_m) ** 2)),
                "median_cos_dzl2_vs_zl2": float(np.median(cos_rows(dzl2_m, zl2_m))),
                "median_r2_collinear_with_zl2": float(np.median(cos_rows(dzl2_m, zl2_m) ** 2)),
            }
        strata = {}
        for name, m in masks.items():
            strata[name] = {
                "n": int(m.sum()),
                "median_abs_dpred_all": float(np.median(np.abs(d_all[m]))),
                "mean_dpred_all": float(d_all[m].mean()),
                "sd_dpred_all": float(d_all[m].std()),
                "mae_all_minus_none": float(
                    np.abs(per["pred_all"][m] - y[m]).mean()
                    - np.abs(per["pred_none"][m] - y[m]).mean()),
                "frac_improved": float(
                    (np.abs(y[m] - per["pred_all"][m])
                     < np.abs(y[m] - per["pred_none"][m])).mean()),
                "dzl_over_zl_median": float(np.median(
                    per["dzl_all"][m] / per["zl_norm"][m])),
                "mae_none": float(np.abs(per["pred_none"][m] - y[m]).mean()),
                "mae_all": float(np.abs(per["pred_all"][m] - y[m]).mean()),

            }
        result["arms"][arm] = {
            "summary": {
                "checkpoint": str((S.RUNS / f"{ARMS[arm]}_s0" / "best.pt")
                                  .relative_to(S._REPO)),
                "epoch": ckpt.get("epoch"),
                "readout": readout,
                "pooled": pooled_summary,
                "affine": affine,
                "collinearity": collinear,
                "strata": strata,
            },
            "per_molecule": {k: v.tolist() for k, v in per.items()
                             if k.startswith(("pred_", "dzl", "first_zl", "zl"))},
        }
        for k, v in per.items():
            raw[f"{arm}:{k}"] = v

    np.savez_compressed(S.ANALYSIS / "_failure_raw.npz", y=y,
                        **{f"st:{k}": v for k, v in st.items()}, **raw)
    (S.ANALYSIS / args.out).write_text(json.dumps(
        {**{k: v for k, v in result.items() if k != "arms"},
         "arms": {a: d["summary"] for a, d in result["arms"].items()}},
        indent=1, default=float))

    cols = (["row_id", "y", "n_bonds", "n1_cv", "n2_cv", "n1_zero_frac", "n2_zero_frac"]
            + [f"{tag}_{c}" for tag in ("sonly", "splusa")
               for c in ("pred_none", "pred_all", "dpred_all", "dzl_all",
                         "dzl2_all", "first_zl_all", "first_zl2_all", "zl_norm")])
    with open(S.ANALYSIS / args.csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for i in range(n_mol):
            vals = [str(i), f"{y[i]:.6f}", f"{st['n_bonds'][i]:.0f}",
                    f"{st['n1_cv'][i]:.4f}", f"{st['n2_cv'][i]:.4f}",
                    f"{st['n1_zero_frac'][i]:.4f}", f"{st['n2_zero_frac'][i]:.4f}"]
            for arm in ARMS:
                d = result["arms"][arm]["per_molecule"]
                vals += [f"{d['pred_none'][i]:.6f}", f"{d['pred_all'][i]:.6f}",
                         f"{d['pred_all'][i] - d['pred_none'][i]:.6f}",
                         f"{d['dzl_all'][i]:.5f}", f"{d['dzl2_all'][i]:.5f}",
                         f"{d['first_zl_all'][i]:.6f}",
                         f"{d['first_zl2_all'][i]:.6f}", f"{d['zl_norm'][i]:.5f}"]
            fh.write(",".join(vals) + "\n")

    print(json.dumps({a: d["summary"] for a, d in result["arms"].items()},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
