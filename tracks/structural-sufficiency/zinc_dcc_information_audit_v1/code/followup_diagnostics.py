#!/usr/bin/env python
"""Task 1: audit the old Ridge/Huber diagnostic on ProbeFit only.

No AuditEval labels are read.  Old AuditEval metrics are copied from the frozen
B4 JSON as background, not re-scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import followup_common as F  # noqa: E402


def standardize_global(X, y):
    xmu, xsd = F.std_fit(X)
    ymu, ysd = F.std_fit(y)
    return F.std_apply(X, xmu, xsd), F.std_apply(y, ymu, ysd), xmu, xsd, ymu, ysd


def ridge_oof(X, y, alpha, seed, n_folds, global_std=True):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if global_std:
        Xs, ys, *_ = standardize_global(X, y)
    kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    pred = np.zeros_like(y)
    for tr, va in kf.split(X):
        if global_std:
            Xtr, ytr, Xva = Xs[tr], ys[tr], Xs[va]
        else:
            xmu, xsd = F.std_fit(X[tr])
            ymu, ysd = F.std_fit(y[tr])
            Xtr = F.std_apply(X[tr], xmu, xsd)
            ytr = F.std_apply(y[tr], ymu, ysd)
            Xva = F.std_apply(X[va], xmu, xsd)
        m = Ridge(alpha=float(alpha)).fit(Xtr, ytr)
        p = m.predict(Xva)
        if not global_std:
            p = p * ysd + ymu
        pred[va] = p
    if global_std:
        # global standardization was fitted on all ProbeFit; invert it
        _, _, _, _, ymu, ysd = standardize_global(X, y)
        pred = pred * ysd + ymu
    return pred


def choose_alpha(X, y, alphas, seed, n_folds=5):
    rows = {}
    for a in alphas:
        p = ridge_oof(X, y, a, seed, n_folds, global_std=True)
        rows[float(a)] = {
            "mse": float(np.mean((y - p) ** 2)),
            "mae": F.mae(y, p),
            "rmse": F.rmse(y, p),
            "median_ae": F.medae(y, p),
        }
    by_mse = min(rows, key=lambda a: (rows[a]["mse"], -a))
    by_mae = min(rows, key=lambda a: (rows[a]["mae"], -a))
    return rows, float(by_mse), float(by_mae)


def fit_full(X, y, alpha, drop_rows=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.ones(len(y), dtype=bool)
    if drop_rows is not None:
        for r in np.atleast_1d(drop_rows):
            keep[int(r)] = False
    xmu, xsd = F.std_fit(X[keep])
    ymu, ysd = F.std_fit(y[keep])
    Xs = F.std_apply(X[keep], xmu, xsd)
    ys = F.std_apply(y[keep], ymu, ysd)
    m = Ridge(alpha=float(alpha)).fit(Xs, ys)
    pred = (F.std_apply(X, xmu, xsd) @ m.coef_.T + m.intercept_) * ysd + ymu
    return m, pred, {"xmu": xmu, "xsd": xsd, "ymu": ymu, "ysd": ysd}


def head_inputs(ctx, arm):
    if arm == "Z":
        return ctx["z"]
    if arm == "ZJ":
        return np.concatenate([ctx["z"], ctx["jraw"], ctx["jlat"]], axis=1)
    if arm == "ZN":
        return np.concatenate([ctx["z"], ctx["jhat_raw"], ctx["jhat_latent"]], axis=1)
    raise ValueError(arm)


def load_saved_head(ctx, arm):
    path = _TRACK / "analysis" / "b" / f"s{ctx['seed']}" / f"head_{arm}.npz"
    rec = np.load(path)
    X = standardize_global(head_inputs(ctx, arm), ctx["y"] - ctx["raw"])[0]
    pred = (X @ rec["coef"].T + rec["intercept"]) * rec["ysd"] + rec["ymu"]
    return path, rec, pred


def huber_full(ctx, arm, max_iter):
    X = head_inputs(ctx, arm)
    y = ctx["y"] - ctx["raw"]
    xmu, xsd = F.std_fit(X)
    ymu, ysd = F.std_fit(y)
    Xs = F.std_apply(X, xmu, xsd)
    ys = F.std_apply(y, ymu, ysd)
    cfg = {"huber_epsilon": 1.35, "huber_max_iter": int(max_iter), "huber_tol": 1e-6}
    m, info = F.huber_fit(Xs, ys, 1e-4, cfg)
    pred = (Xs @ m.coef_ + m.intercept_) * ysd + ymu
    return pred, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(_TRACK / "analysis" / "followup" / "diagnostics.json"))
    args = p.parse_args(argv)
    cfg = F.load_config()
    rc = cfg["ridge_diagnostics"]
    seeds = [0, 1]
    arms = ("Z", "ZJ", "ZN")
    out = {
        "protocol": cfg["protocol"],
        "task": "old_audit_recheck_probe_only",
        "note": ("ProbeFit labels only. AuditEval labels are forbidden in this process; "
                 "old AuditEval MAE values are copied from the frozen B4 metrics JSON."),
        "config": rc,
        "seeds": {},
    }
    for seed in seeds:
        ctx = F.load_probe_context(seed, cfg)
        y = ctx["y"]
        raw = ctx["raw"]
        resid = y - raw
        rec = {"seed": seed, "n_probe": int(len(y)), "raw_mae": F.mae(y, raw),
               "resid": {"mean": float(resid.mean()), "std": float(resid.std()),
                         "min": float(resid.min()), "max": float(resid.max())},
               "arms": {}, "old_b4_audit_metrics": {}}
        old = json.loads((_TRACK / "analysis" / "b" / f"s{seed}" / "metrics.json").read_text())
        freeze = json.loads((_TRACK / "analysis" / "b" / f"s{seed}" / "freeze.json").read_text())
        rec["old_b4_audit_metrics"] = {
            "arms": {k: {"mae": v["mae"], "rmse": v["rmse"], "median_ae": v["median_ae"]}
                     for k, v in old["arms"].items()},
            "gains": old["gains"],
            "label_access": old["label_access"],
            "note": "frozen B4 values; not recomputed here",
        }

        top2 = np.argsort(np.abs(resid))[::-1][:2]
        rec["top2"] = {
            "indices": [int(i) for i in top2],
            "rows": [int(ctx["probe"][i]) for i in top2],
            "residuals": [float(resid[i]) for i in top2],
            "sse_share": float(np.sum(resid[top2] ** 2) / np.sum(resid ** 2)),
            "mae_share": float(np.sum(np.abs(resid[top2])) / np.sum(np.abs(resid))),
            "resid_std_all": float(resid.std()),
            "resid_std_without_top2": float(np.delete(resid, top2).std()),
        }

        for arm in arms:
            X = head_inputs(ctx, arm)
            path, saved, saved_pred = load_saved_head(ctx, arm)
            saved_file = np.load(_TRACK / "analysis" / "b" / f"s{seed}" / f"pred_{arm}_probe.npy")
            parity = float(np.max(np.abs(saved_pred + raw - saved_file)))
            orig_rows, orig_mse, orig_mae = choose_alpha(
                X, resid, rc["alphas_original"], rc["primary_fold_seed"], rc["folds"])
            ext_rows, ext_mse, ext_mae = choose_alpha(
                X, resid, rc["alphas_extended"], rc["primary_fold_seed"], rc["folds"])
            b3_alpha = float(saved["alpha"])
            # full-fit predictions at the frozen B3 alpha
            _, full_pred, _ = fit_full(X, resid, b3_alpha)
            # stability of alpha selection across fold seeds
            stab = {}
            for fs in rc["stability_fold_seeds"]:
                rows, bm, ba = choose_alpha(X, resid, rc["alphas_original"], int(fs), rc["folds"])
                stab[str(fs)] = {"mse_alpha": bm, "mae_alpha": ba,
                                 "oof": rows}
            alpha_vals = [v["mse_alpha"] for v in stab.values()]
            alpha_mae_vals = [v["mae_alpha"] for v in stab.values()]
            # top-2 influence, full fit at b3 alpha
            m_all, pred_all, _ = fit_full(X, resid, b3_alpha)
            m_no, pred_no, _ = fit_full(X, resid, b3_alpha, drop_rows=top2)
            # high-dimensional conditioning
            Xs, _, *_ = standardize_global(X, resid)
            sv = np.linalg.svd(Xs, compute_uv=False)
            rec["arms"][arm] = {
                "input_dim": int(X.shape[1]),
                "feature_to_n_ratio": float(X.shape[1] / X.shape[0]),
                "saved_head_sha256": F.sha_array(np.asarray(saved["coef"])),
                "saved_probe_prediction_parity_max_abs": parity,
                "b3_alpha": b3_alpha,
                "b3_alpha_is_boundary": bool(
                    np.isclose(b3_alpha, min(rc["alphas_original"]))
                    or np.isclose(b3_alpha, max(rc["alphas_original"]))
                ),
                "b3_cv_mse": freeze.get("heads", {}).get(arm, {}).get("alpha", {}).get("cv_mse"),
                "original_grid": orig_rows,
                "original_grid_selected_by_mse": orig_mse,
                "original_grid_selected_by_mae": orig_mae,
                "extended_grid": ext_rows,
                "extended_grid_selected_by_mse": ext_mse,
                "extended_grid_selected_by_mae": ext_mae,
                "extended_grid_boundary_selected": bool(
                    any(np.isclose(v, min(rc["alphas_extended"])) or np.isclose(v, max(rc["alphas_extended"]))
                        for v in (ext_mse, ext_mae))
                ),
                "stability": stab,
                "stability_mse_alpha_fraction": {
                    str(a): float(np.mean(np.isclose(alpha_vals, a)))
                    for a in sorted(set(alpha_vals))
                },
                "stability_mae_alpha_fraction": {
                    str(a): float(np.mean(np.isclose(alpha_mae_vals, a)))
                    for a in sorted(set(alpha_mae_vals))
                },
                "full_fit_probe_mae": F.mae(y, raw + full_pred),
                "full_fit_probe_rmse": F.rmse(y, raw + full_pred),
                "full_fit_prediction_sha256": F.sha_array(full_pred),
                "drop_top2": {
                    "mae_all": F.mae(y, raw + pred_no),
                    "mae_without_top2_samples": F.mae(np.delete(y, top2), np.delete(raw + pred_no, top2)),
                    "top2_mae": F.mae(y[top2], (raw + pred_no)[top2]),
                    "max_prediction_diff": float(np.max(np.abs(pred_no - pred_all))),
                    "corr_prediction": float(np.corrcoef(pred_no, pred_all)[0, 1]),
                },
                "highdim": {
                    "n": int(X.shape[0]), "p": int(X.shape[1]),
                    "singular_max": float(sv[0]), "singular_min": float(sv[-1]),
                    "condition_number": float(sv[0] / max(sv[-1], 1e-300)),
                    "effective_rank": float(np.exp(-np.sum((sv / sv.sum()) * np.log(np.maximum(sv / sv.sum(), 1e-300))))),
                    "coef_l2": float(np.linalg.norm(m_all.coef_)),
                },
            }
            # Huber: old max_iter vs converged; same global standardization as B3.
            for tag, max_iter in (("max_iter_300", 300), ("max_iter_10000", 10000)):
                hp, hi = huber_full(ctx, arm, max_iter)
                rec["arms"][arm][f"huber_{tag}"] = {
                    "fit": hi,
                    "probe_mae": F.mae(y, raw + hp),
                    "probe_rmse": F.rmse(y, raw + hp),
                    "correction": {
                        "mean": float(hp.mean()), "std": float(hp.std()),
                        "min": float(hp.min()), "max": float(hp.max()),
                    },
                }
            # Compare old 300-iteration fit to converged fit.
            h300, _ = huber_full(ctx, arm, 300)
            h10k, _ = huber_full(ctx, arm, 10000)
            rec["arms"][arm]["huber_300_vs_10000"] = {
                "max_prediction_diff": float(np.max(np.abs(h300 - h10k))),
                "mae_diff": float(F.mae(y, raw + h300) - F.mae(y, raw + h10k)),
                "corr": float(np.corrcoef(h300, h10k)[0, 1]),
            }

        # Phase-1e cycle-penalty explanation for the two ProbeFit anomalies.
        y_map = {int(r): float(v) for r, v in zip(ctx["probe"], y)}
        try:
            rec["cycle_penalty"] = F.cycle_penalty_for_rows([2210, 3776], y_map)
        except Exception as exc:  # diagnostic only; do not hide a real failure
            rec["cycle_penalty"] = {"error": repr(exc)}
        rec["label_guard"] = {
            "allowed_rows": int(len(ctx["probe"])),
            "access_blocks": int(len(ctx["guard"].log)),
            "audit_labels_accessed": False,
            "forbidden_rows": [int(x) for x in ctx["audit"][:5]],
            "official_valid_test_loaded": False,
        }
        out["seeds"][str(seed)] = rec
        print(f"seed {seed}: raw MAE {rec['raw_mae']:.6f}; "
              f"old B4 raw MAE {old['arms']['raw']['mae']:.6f}")

    out["label_guard_global"] = {
        "audit_eval_labels_read": False,
        "official_valid_test_read": False,
        "note": "All new fits use ProbeFit labels only; old AuditEval MAE is from frozen B4 JSON.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(F.jsonable(out), indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
