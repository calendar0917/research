#!/usr/bin/env python
"""Task 2: minimal robust conditional-increment test on ProbeFit only.

Primary estimator: LAD / median regression (QuantileRegressor, quantile=0.5,
HiGHS).  All standardization, alpha selection, and correction clipping are fit
inside the training folds.  AuditEval labels are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import followup_common as F  # noqa: E402


def folds(n, cfg):
    ocfg = cfg.get("oof", cfg)
    return list(KFold(n_splits=int(ocfg["outer_folds"]), shuffle=True,
                      random_state=int(ocfg["outer_seed"])).split(np.arange(n)))


def run_pipeline(X, y, cfg, base_seed, outer):
    pred = np.zeros(len(y))
    recs = []
    for fi, (tr, va) in enumerate(outer):
        p, info = F.fit_predict_outer(
            X[tr], y[tr], X[va], cfg, base_seed=int(base_seed) + 100 * fi
        )
        pred[va] = p
        recs.append({
            "fold": int(fi),
            "n_train": int(len(tr)),
            "n_eval": int(len(va)),
            "alpha": info["alpha"],
            "alpha_scores_mae": info["alpha_selection"]["alpha_scores_mae"],
            "n_iter": info["fit"]["n_iter"],
            "solver": info["fit"]["solver"],
            "finite": info["fit"]["finite"],
            "warnings": info["fit"]["warnings"],
            "clip": info["clip"],
            "clip_fraction": info["clip_fraction"],
            "correction": info["correction"],
            "correction_unclipped": info["correction_unclipped"],
            "coef_norm": info["coef_norm"],
        })
    return pred, recs


def summarize(y, raw, pred_c, pred_j, folds_rec, rows, cfg, strata, label):
    # pred_c / pred_j are residual corrections; the actual prediction is raw + correction.
    pc = np.asarray(raw) + np.asarray(pred_c)
    pj = np.asarray(raw) + np.asarray(pred_j)
    gain_c = np.abs(y - pc) - np.abs(y - pj)
    gain_raw = np.abs(y - raw) - np.abs(y - pj)
    out = {
        "label": label,
        "n": int(len(y)),
        "raw_mae": F.mae(y, raw),
        "control_mae": F.mae(y, pc),
        "cj_mae": F.mae(y, pj),
        "raw_rmse": F.rmse(y, raw),
        "control_rmse": F.rmse(y, pc),
        "cj_rmse": F.rmse(y, pj),
        "gain_control": F.boot_ci(gain_c, cfg["bootstrap_n"], cfg["bootstrap_seed"]),
        "gain_raw": F.boot_ci(gain_raw, cfg["bootstrap_n"], cfg["bootstrap_seed"]),
        "fold_gains": [
            {
                "fold": r["fold"],
                "n_eval": r["n_eval"],
                "raw_mae": F.mae(y[fold_eval], raw[fold_eval]),
                "control_mae": F.mae(y[fold_eval], pc[fold_eval]),
                "cj_mae": F.mae(y[fold_eval], pj[fold_eval]),
                "gain_control": F.mae(y[fold_eval], pc[fold_eval]) - F.mae(y[fold_eval], pj[fold_eval]),
                "gain_raw": F.mae(y[fold_eval], raw[fold_eval]) - F.mae(y[fold_eval], pj[fold_eval]),
                "alpha_control": folds_rec["control"][r["fold"]]["alpha"],
                "alpha_cj": folds_rec["cj"][r["fold"]]["alpha"],
                "n_iter_control": folds_rec["control"][r["fold"]]["n_iter"],
                "n_iter_cj": folds_rec["cj"][r["fold"]]["n_iter"],
            }
            for r in folds_rec["control"]
            for fold_eval in [np.asarray(folds_rec["eval"][r["fold"]], dtype=np.int64)]
        ],
        "fold_details": {"control": folds_rec["control"], "cj": folds_rec["cj"]},
        "top_gain_control": F.top_gain_rows(y, pc, pj, rows, k=10),
        "top_gain_raw": F.top_gain_rows(y, raw, pj, rows, k=10),
        "top_error_cj": F.top_error_rows(y, pj, rows, k=10),
        "top_error_raw": F.top_error_rows(y, raw, rows, k=10),
    }
    # Extreme-row contribution, reported without dropping them from the main MAE.
    top2 = np.argsort(np.abs(y - raw))[::-1][:2]
    keep = np.ones(len(y), dtype=bool)
    keep[top2] = False
    out["top2"] = {
        "indices": [int(i) for i in top2],
        "rows": [int(rows[int(i)]) for i in top2],
        "raw_residuals": [float((y - raw)[int(i)]) for i in top2],
        "raw_mae_all": F.mae(y, raw),
        "raw_mae_without_top2": F.mae(y[keep], raw[keep]),
        "control_mae_all": F.mae(y, pc),
        "control_mae_without_top2": F.mae(y[keep], pc[keep]),
        "cj_mae_all": F.mae(y, pj),
        "cj_mae_without_top2": F.mae(y[keep], pj[keep]),
        "top2_abs_error_control": float(np.sum(np.abs(y - pc)[top2])),
        "top2_abs_error_cj": float(np.sum(np.abs(y - pj)[top2])),
        "top2_abs_error_raw": float(np.sum(np.abs(y - raw)[top2])),
        "top2_share_control": float(np.sum(np.abs(y - pc)[top2]) / np.sum(np.abs(y - pc))),
        "top2_share_cj": float(np.sum(np.abs(y - pj)[top2]) / np.sum(np.abs(y - pj))),
    }
    # strata: labels only for explanation; support/size/ring are structural.
    out["strata"] = {}
    for sname, sval in strata.items():
        out["strata"][sname] = {}
        for level in sorted(set(np.asarray(sval).tolist())):
            m = np.asarray(sval) == level
            if not m.any():
                continue
            out["strata"][sname][str(level)] = {
                "n": int(m.sum()),
                "raw_mae": F.mae(y[m], raw[m]),
                "control_mae": F.mae(y[m], pc[m]),
                "cj_mae": F.mae(y[m], pj[m]),
                "gain_control": F.boot_ci(
                    np.abs(y[m] - pc[m]) - np.abs(y[m] - pj[m]),
                    n_boot=2000, seed=cfg["bootstrap_seed"]),
                "gain_raw": F.boot_ci(
                    np.abs(y[m] - raw[m]) - np.abs(y[m] - pj[m]),
                    n_boot=2000, seed=cfg["bootstrap_seed"]),
            }
    out["all_samples_included_in_main_mae"] = True
    return out


def real_pair(ctx, cfg, kind, strata, control_pred=None, control_recs=None):
    y = ctx["y"]
    raw = ctx["raw"]
    g = ctx["g"]
    jp = ctx["jperp_raw"] if kind == "J_raw" else ctx["jperp_latent"]
    Xc = g
    Xj = np.concatenate([g, jp], axis=1)
    outer = folds(len(y), cfg)
    eval_idx = [np.asarray(va, dtype=np.int64) for _, va in outer]
    if control_pred is None:
        control_pred, control_recs = run_pipeline(Xc, y - raw, cfg, cfg["outer_seed"], outer)
    cj_pred, cj_recs = run_pipeline(Xj, y - raw, cfg, cfg["outer_seed"] + 10000, outer)
    rec = summarize(y, raw, control_pred, cj_pred,
                    {"control": control_recs, "cj": cj_recs, "eval": eval_idx},
                    ctx["probe"], cfg, strata, f"real:{kind}")
    rec["kind"] = kind
    rec["control_prediction_sha256"] = F.sha_array(control_pred)
    rec["cj_prediction_sha256"] = F.sha_array(cj_pred)
    return rec, control_pred, control_recs


def null_pair(ctx, cfg, kind, strata, control_pred, control_recs, n_perm):
    y = ctx["y"]
    raw = ctx["raw"]
    g = ctx["g"]
    ocfg = cfg["oof"]
    jp = ctx["jperp_raw"] if kind == "J_raw" else ctx["jperp_latent"]
    outer = folds(len(y), ocfg)
    eval_idx = [np.asarray(va, dtype=np.int64) for _, va in outer]
    gains_c, gains_r, records = [], [], []
    for p in range(int(n_perm)):
        rng = np.random.default_rng(int(cfg["null"]["seed"]) + 1000 * p
                                    + (17 if kind == "J_raw" else 29))
        perm = rng.permutation(len(y))
        Xj = np.concatenate([g, jp[perm]], axis=1)
        cj_pred, cj_recs = run_pipeline(Xj, y - raw, ocfg, ocfg["outer_seed"] + 20000 + 100 * p, outer)
        pc = raw + control_pred
        pj = raw + cj_pred
        gc = np.abs(y - pc) - np.abs(y - pj)
        gr = np.abs(y - raw) - np.abs(y - pj)
        gains_c.append(float(gc.mean()))
        gains_r.append(float(gr.mean()))
        records.append({"perm": p, "gain_control": float(gc.mean()), "gain_raw": float(gr.mean()),
                        "cj_mae": F.mae(y, pj)})
    return {
        "kind": kind,
        "n_permutations": int(n_perm),
        "gain_control": {
            "values": gains_c,
            "mean": float(np.mean(gains_c)),
            "ci95": [float(np.percentile(gains_c, 2.5)), float(np.percentile(gains_c, 97.5))],
            "max": float(np.max(gains_c)),
        },
        "gain_raw": {
            "values": gains_r,
            "mean": float(np.mean(gains_r)),
            "ci95": [float(np.percentile(gains_r, 2.5)), float(np.percentile(gains_r, 97.5))],
            "max": float(np.max(gains_r)),
        },
        "records": records,
    }


def build_positive_target(ctx, kind, target, rep, cfg):
    y = ctx["y"]
    raw = ctx["raw"]
    g = ctx["g"]
    jp = ctx["jperp_raw"] if kind == "J_raw" else ctx["jperp_latent"]
    r = y - raw
    rng = np.random.default_rng(
        int(cfg["positive_control"]["signal_seed"])
        + 100000 * int(ctx["seed"]) + (101 if kind == "J_raw" else 211)
        + 1000 * int(rep) + int(round(10000 * target))
    )
    w = rng.normal(size=jp.shape[1])
    s = jp @ w
    A = np.column_stack([np.ones(len(g)), g])
    coef = np.linalg.lstsq(A, s, rcond=None)[0]
    s = s - A @ coef
    s = (s - s.mean()) / max(s.std(), 1e-12)
    noise = rng.standard_t(int(cfg["positive_control"]["noise_df"]), size=len(r))
    noise = noise * float(cfg["positive_control"]["noise_scale"])
    t = r + noise

    def oracle_gain(beta):
        return float(np.mean(np.abs(t + beta * s)) - np.mean(np.abs(t)))

    cap = float(cfg["positive_control"]["beta_cap"])
    # The oracle curve is monotone for these small signals.  Scan first so a
    # non-monotone finite sample does not silently return a bad root.
    grid = np.linspace(0.0, cap, 2000)
    vals = np.asarray([oracle_gain(b) for b in grid])
    hit = np.where(vals >= float(target))[0]
    if hit.size:
        lo = grid[max(hit[0] - 1, 0)]
        hi = grid[int(hit[0])]
        try:
            beta = float(brentq(lambda b: oracle_gain(b) - float(target), lo, hi,
                                maxiter=int(cfg["positive_control"]["beta_bisection_steps"])))
        except Exception:
            beta = float(hi)
    else:
        beta = cap
    y_sim = raw + t + beta * s
    return {
        "y_sim": y_sim,
        "beta": beta,
        "oracle_gain": oracle_gain(beta),
        "signal_std": float(s.std()),
        "noise_std": float(noise.std()),
        "r_sim": t + beta * s,
    }


def positive_pair(ctx, cfg, kind, target, rep, strata):
    y0 = ctx["y"]
    raw = ctx["raw"]
    g = ctx["g"]
    ocfg = cfg["oof"]
    jp = ctx["jperp_raw"] if kind == "J_raw" else ctx["jperp_latent"]
    sim = build_positive_target(ctx, kind, target, rep, cfg)
    y = sim["y_sim"]
    outer = folds(len(y), ocfg)
    eval_idx = [np.asarray(va, dtype=np.int64) for _, va in outer]
    control_pred, control_recs = run_pipeline(g, y - raw, ocfg,
                                              ocfg["outer_seed"] + 30000 + 1000 * rep,
                                              outer)
    cj_pred, cj_recs = run_pipeline(np.concatenate([g, jp], axis=1), y - raw, ocfg,
                                    ocfg["outer_seed"] + 40000 + 1000 * rep,
                                    outer)
    rec = summarize(y, raw, control_pred, cj_pred,
                    {"control": control_recs, "cj": cj_recs, "eval": eval_idx},
                    ctx["probe"], ocfg, strata, f"positive:{kind}:{target}:{rep}")
    rec.update({
        "seed": int(ctx["seed"]),
        "kind": kind,
        "target_oracle_gain": float(target),
        "replicate": int(rep),
        "beta": sim["beta"],
        "oracle_gain": sim["oracle_gain"],
        "estimated_gain_control": rec["gain_control"]["mean"],
        "estimated_gain_raw": rec["gain_raw"]["mean"],
        "detected": bool(rec["gain_control"]["mean"] > 0 and rec["gain_raw"]["mean"] > 0),
        "magnitude_ratio": (float(rec["gain_control"]["mean"] / sim["oracle_gain"])
                            if sim["oracle_gain"] > 0 else None),
    })
    # Do not retain the full synthetic per-fold label vectors in JSON.
    return rec


def decision_from_results(real, null, positive, cfg):
    # Positive-control power: fraction of replicates detected at the larger
    # target.  The task asks whether the diagnostic can identify ~0.003-0.005.
    pc = cfg["positive_control"]
    target_big = max(pc["target_oracle_gains"])
    det = [r["detected"] for r in positive if abs(r["oracle_gain"] - target_big) < 1e-9]
    det_rate = float(np.mean(det)) if det else 0.0
    power_ok = det_rate >= float(cfg["decision"]["underpowered_if_positive_control_detection_below"])
    det_by_target = {}
    for t in pc["target_oracle_gains"]:
        vals = [r["detected"] for r in positive if abs(r["oracle_gain"] - float(t)) < 1e-9]
        det_by_target[str(t)] = float(np.mean(vals)) if vals else 0.0
    real_ok = {}
    real_details = {}
    for key, rec in real.items():
        n = null[key]
        p_control = float((1 + np.sum(np.asarray(n["gain_control"]["values"]) >= rec["gain_control"]["mean"]))
                          / (len(n["gain_control"]["values"]) + 1))
        p_raw = float((1 + np.sum(np.asarray(n["gain_raw"]["values"]) >= rec["gain_raw"]["mean"]))
                      / (len(n["gain_raw"]["values"]) + 1))
        real_ok[key] = bool(
            rec["gain_control"]["mean"] > 0
            and rec["gain_raw"]["mean"] > 0
            and rec["gain_control"]["mean"] > n["gain_control"]["ci95"][1]
            and rec["gain_raw"]["mean"] > n["gain_raw"]["ci95"][1]
            and rec["gain_control"]["mean"] >= float(cfg["decision"]["meaningful_oof_gain"])
        )
        real_details[key] = {
            "gain_control": rec["gain_control"]["mean"],
            "gain_raw": rec["gain_raw"]["mean"],
            "null95_control": n["gain_control"]["ci95"][1],
            "null95_raw": n["gain_raw"]["ci95"][1],
            "permutation_p_control": p_control,
            "permutation_p_raw": p_raw,
        }
    if not power_ok:
        cls = "AUDIT_UNDERPOWERED_OR_INVALID"
    elif any(real_ok.values()):
        cls = "PROPOSE_CONFIRMATION_DESIGN"
    else:
        cls = "NO_REASON_TO_REOPEN_J_FAMILY"
    return {"class": cls, "positive_control_detection_rate_at_largest_target": det_rate,
            "positive_control_power_ok": bool(power_ok),
            "real_candidate": real_ok,
            "real_details": real_details,
            "positive_control_detection_rates_by_target": det_by_target,
            "positive_control_target_largest": float(target_big)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(_TRACK / "analysis" / "followup" / "oof.json"))
    args = p.parse_args(argv)
    cfg = F.load_config()
    ocfg = cfg["oof"]
    seeds = [0, 1]
    kinds = ["J_raw", "J_latent"]
    out = {
        "protocol": cfg["protocol"],
        "task": "robust_conditional_increment_probe_oof",
        "config": cfg,
        "scope": {
            "rows": "ProbeFit 1000 only",
            "audit_eval_labels_read": False,
            "official_valid_test_read": False,
            "main_mae_includes_all_probe_rows": True,
        },
        "seeds": {},
        "null": {},
        "positive_control": [],
    }
    strata_cache = None
    for seed in seeds:
        ctx = F.load_probe_context(seed, cfg)
        if strata_cache is None:
            strata_cache = F.strata_for_probe(ctx, ocfg)
        strata = strata_cache
        y = ctx["y"]
        raw = ctx["raw"]
        g = ctx["g"]
        outer = folds(len(y), ocfg)
        eval_idx = [np.asarray(va, dtype=np.int64) for _, va in outer]
        control_pred, control_recs = run_pipeline(g, y - raw, ocfg, ocfg["outer_seed"], outer)
        rec_seed = {
            "seed": seed,
            "n_probe": int(len(y)),
            "raw_mae": F.mae(y, raw),
            "control": summarize(y, raw, control_pred, control_pred,
                                 {"control": control_recs, "cj": control_recs, "eval": eval_idx},
                                 ctx["probe"], ocfg, strata, "control_only"),
            "blocks": {},
            "label_guard": {
                "access_blocks": int(len(ctx["guard"].log)),
                "audit_labels_accessed": False,
            },
        }
        # The control summary above uses pred_c == pred_j, so its gains are zero;
        # keep it only as a calibration reference.
        for kind in kinds:
            rec, cj_pred, cj_recs = real_pair(ctx, ocfg, kind, strata,
                                              control_pred=control_pred,
                                              control_recs=control_recs)
            rec_seed["blocks"][kind] = rec
            out["null"][f"{seed}:{kind}"] = null_pair(
                ctx, cfg, kind, strata, control_pred, control_recs,
                int(cfg["null"]["permutations"]))
        out["seeds"][str(seed)] = rec_seed
        print(f"seed {seed}: raw {F.mae(y, raw):.6f}; "
              f"control {F.mae(y, raw + control_pred):.6f}")

    # Positive control: both seeds, both blocks, target sizes, replicates.
    for seed in seeds:
        ctx = F.load_probe_context(seed, cfg)
        if strata_cache is None:
            strata_cache = F.strata_for_probe(ctx, ocfg)
        for kind in kinds:
            for target in cfg["positive_control"]["target_oracle_gains"]:
                for rep in range(int(cfg["positive_control"]["replicates"])):
                    out["positive_control"].append(
                        positive_pair(ctx, cfg, kind, float(target), int(rep), strata_cache)
                    )
    # Summarize positive-control detection rates by seed/kind/target.
    pc_summary = {}
    for r in out["positive_control"]:
        key = f"{r['seed']}:{r['kind']}:target={r['target_oracle_gain']}"
        pc_summary.setdefault(key, {"n": 0, "detected": 0, "oracle_gain": [],
                                    "estimated_gain_control": [], "estimated_gain_raw": [],
                                    "magnitude_ratio": []})
        s = pc_summary[key]
        s["n"] += 1
        s["detected"] += int(r["detected"])
        for k in ("oracle_gain", "estimated_gain_control", "estimated_gain_raw", "magnitude_ratio"):
            if r[k] is not None:
                s[k].append(float(r[k]))
    for s in pc_summary.values():
        s["detection_rate"] = float(s["detected"] / s["n"])
        for k in ("oracle_gain", "estimated_gain_control", "estimated_gain_raw", "magnitude_ratio"):
            s[k] = {"mean": float(np.mean(s[k])), "values": s[k]} if s[k] else None
    out["positive_control_summary"] = pc_summary
    out["decision"] = decision_from_results(
        {f"{seed}:{kind}": out["seeds"][str(seed)]["blocks"][kind]
         for seed in seeds for kind in kinds},
        out["null"],
        out["positive_control"],
        cfg,
    )
    out["label_guard_global"] = {
        "audit_eval_labels_read": False,
        "official_valid_test_read": False,
        "note": "No AuditEval label accessor was created; only ProbeFit rows were allowed.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(F.jsonable(out), indent=2))
    print(json.dumps(out["decision"], indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
