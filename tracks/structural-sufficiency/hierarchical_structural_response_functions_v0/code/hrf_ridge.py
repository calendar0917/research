"""Five convex structural-response arms: FP64 ridge path, ledger, exports."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import hrf_common as C
import hrf_design as D

ARMS = ("L_BASE", "F_BASE", "L_REAL", "F_NULL", "F_REAL")
SENSITIVITY = ("F_NULL_SELF",)


def _log(msg: str) -> None:
    print(f"[ridge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def solve_path(X: np.ndarray, y_std: np.ndarray, train: np.ndarray,
               lambdas: Sequence[float]) -> dict[str, Any]:
    """Dual ridge path from one Gram eigendecomposition (FP64)."""
    Xt = np.asarray(X[train], np.float64)
    mu = Xt.mean(axis=0)
    Xc = Xt - mu
    n = int(Xc.shape[0])
    ymean = float(y_std[train].mean())
    zc = np.asarray(y_std[train], np.float64) - ymean
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)
    proj = V.T @ zc
    lambdas = np.asarray(lambdas, np.float64)
    B = np.zeros((X.shape[1], lambdas.size), np.float64)
    alphas = np.zeros((n, lambdas.size), np.float64)
    residual = np.zeros(lambdas.size, np.float64)
    objective = np.zeros(lambdas.size, np.float64)
    znorm = float(np.linalg.norm(zc))
    for i, lam in enumerate(lambdas):
        denom = w + n * float(lam)
        coef = proj / denom
        alpha = V @ coef
        alphas[:, i] = alpha
        B[:, i] = Xc.T @ alpha
        r = V @ ((denom * coef) - proj)
        residual[i] = float(np.linalg.norm(r) / max(znorm, 1e-300))
        objective[i] = float(np.mean((G @ alpha - zc) ** 2) +
                             float(lam) * float(B[:, i] @ B[:, i]))
    return {"mu": mu, "ymean": ymean, "G": G, "w": w, "V": V,
            "lambdas": lambdas, "beta": B, "alpha": alphas,
            "residual": residual, "objective": objective, "n_train": n}


def predict_all(X: np.ndarray, mu: np.ndarray, zmean: float, beta: np.ndarray,
                y_mean: float, y_scale: float, chunk: int = 512) -> np.ndarray:
    out = np.empty((X.shape[0], beta.shape[1]), np.float64)
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        zhat = zmean + (X[start:stop] - mu) @ beta
        out[start:stop] = y_mean + y_scale * zhat
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", type=str, default=str(C.DATA_DIR / "hsrf_stage0.npz"))
    ap.add_argument("--out", type=str, default=str(C.RESULTS_DIR / "ridge"))
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    cfg = C.load_config()
    lambdas = [float(x) for x in cfg["ridge"]["lambdas"]]
    stage = D.Stage0(args.stage0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("predictions", "states", "ledger", "curves"):
        (out_dir / sub).mkdir(exist_ok=True)
    arms = [a for a in ARMS if (not args.only or a in args.only.split(","))]
    arms += [a for a in SENSITIVITY if a in args.only.split(",")]
    path_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    for arm in arms:
        t0 = time.time()
        design = stage.design(arm)
        X = design["X"]
        groups = design["groups"]
        _log(f"{arm}: design {X.shape}, groups {len(groups)}")
        sol = solve_path(X, stage.y_std, stage.train, lambdas)
        preds = predict_all(X, sol["mu"], sol["ymean"], sol["beta"],
                            stage.scaler["mean"], stage.scaler["std"])
        metrics = []
        for i, lam in enumerate(sol["lambdas"]):
            row = {"arm": arm, "lambda": float(lam), "n_features": int(X.shape[1]),
                   "params": int(X.shape[1] + 1),
                   "objective": float(sol["objective"][i]),
                   "normal_residual": float(sol["residual"][i])}
            for split in ("train", "valid", "test"):
                rows = stage.rows[split]
                row[f"{split}_rmse"] = C.rmse(stage.y[rows], preds[rows, i])
                row[f"{split}_mae"] = C.mae(stage.y[rows], preds[rows, i])
            metrics.append(row)
            path_rows.append(row)
        # min Valid RMSE; ties -> stronger lambda
        best_i = 0
        for i in range(1, len(metrics)):
            d = metrics[i]["valid_rmse"] - metrics[best_i]["valid_rmse"]
            if d < -1e-12 or (abs(d) <= 1e-12 and sol["lambdas"][i] > sol["lambdas"][best_i]):
                best_i = i
        best = metrics[best_i]
        best["selected_lambda"] = float(sol["lambdas"][best_i])
        best["selected_index"] = int(best_i)
        best["arm"] = arm
        best["wall_seconds"] = float(time.time() - t0)
        main_rows.append(best)
        _log(f"{arm}: best lambda={best['selected_lambda']} "
             f"valid={best['valid_rmse']:.6f} test={best['test_rmse']:.6f}")

        # predictions
        for split in ("train", "valid", "test"):
            rows = stage.rows[split]
            p = preds[rows, best_i]
            np.savez_compressed(out_dir / "predictions" / f"{arm}_{split}.npz",
                                row_id=rows, y_true=stage.y[rows], pred=p,
                                split=np.asarray([split]),
                                arm=np.asarray([arm]))
            C.write_csv(out_dir / "predictions" / f"{arm}_{split}.csv", [
                {"row_id": int(r), "split": split, "y_true": float(y),
                 "pred": float(q)}
                for r, y, q in zip(rows, stage.y[rows], p)
            ])

        # states / coefficients
        beta_best = sol["beta"][:, best_i]
        np.savez_compressed(
            out_dir / "states" / f"{arm}.npz",
            beta_all=sol["beta"].astype(np.float64),
            lambdas=sol["lambdas"], selected_index=np.asarray([best_i]),
            beta_selected=beta_best, mu=sol["mu"], ymean=np.asarray([sol["ymean"]]),
            residual=sol["residual"], objective=sol["objective"],
            n_features=np.asarray([X.shape[1]]),
        )
        coeff_rows = D.coefficient_table(arm, beta_best, groups,
                                         stage.base_knot, stage.rel_knot)
        C.write_csv(out_dir / "states" / f"{arm}_coefficients.csv", coeff_rows)

        # exact ledger
        led = D.ledger(X, beta_best, groups, sol["mu"], stage.scaler["mean"],
                       stage.scaler["std"], float(sol["ymean"]))
        err = led["reconstructed"] - preds[:, best_i]
        max_err = float(np.max(np.abs(err)))
        _log(f"{arm}: ledger max |reconstruction - prediction| = {max_err:.3e}")
        np.savez_compressed(
            out_dir / "ledger" / f"{arm}.npz",
            group_values=led["group_values"], intercept=np.asarray([led["intercept"]]),
            main_effect_sum=led["main_effect_sum"],
            relation_effect_sum=led["relation_effect_sum"],
            reconstructed=led["reconstructed"],
            prediction=preds[:, best_i],
            reconstruction_error=err,
            row_id=stage.z["row_ids"], arm=np.asarray([arm]),
            best_lambda=np.asarray([sol["lambdas"][best_i]]),
        )
        group_meta = []
        mean_abs = np.mean(np.abs(led["group_values"]), axis=0)
        for gi, g in enumerate(groups):
            group_meta.append({
                "group_index": int(gi), "group_id": g["group_id"], "block": g["block"],
                "kind": g["kind"], "raw_col": int(g["raw_col"]),
                "name": g["name"], "meta": g.get("meta", {}),
                "mean_abs_contribution": float(mean_abs[gi]),
            })
        C.write_json(out_dir / "ledger" / f"{arm}_groups.json", {
            "arm": arm, "n_groups": len(groups), "intercept": led["intercept"],
            "max_reconstruction_error": max_err,
            "groups": group_meta,
        })

        # response curves: top 20 main and top 20 relation by mean |contribution|
        main_idx = [i for i, g in enumerate(groups) if g["block"] == "base"]
        rel_idx = [i for i, g in enumerate(groups) if g["block"].startswith("relation")]
        top_main = sorted(main_idx, key=lambda i: -mean_abs[i])[:20]
        top_rel = sorted(rel_idx, key=lambda i: -mean_abs[i])[:20]
        curves = D.function_curves(stage, arm, beta_best, groups,
                                   top_main + top_rel, led["group_values"])
        C.write_json(out_dir / "curves" / f"{arm}.json", {
            "arm": arm, "selected_lambda": float(sol["lambdas"][best_i]),
            "top_main_groups": [int(i) for i in top_main],
            "top_relation_groups": [int(i) for i in top_rel],
            "curves": curves,
        })

    C.write_csv(out_dir / "regularization_path.csv", path_rows)
    C.write_csv(out_dir / "main_table.csv", main_rows)
    C.write_json(out_dir / "run_meta.json", {
        "round": cfg["round_id"], "commit": C.git_commit(), "created_at": C.now_utc(),
        "stage0": str(stage.path), "stage0_sha256": C.sha256_file(stage.path),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "arms": arms, "lambdas": lambdas,
    })


if __name__ == "__main__":
    main()