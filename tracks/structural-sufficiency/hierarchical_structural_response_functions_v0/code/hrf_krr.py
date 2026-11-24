"""Independent RBF kernel-ridge references on the F_BASE / F_REAL matrices."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

import hrf_common as C
import hrf_design as D


def _log(msg: str) -> None:
    print(f"[krr {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", default=str(C.DATA_DIR / "hsrf_stage0.npz"))
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "krr"))
    args = ap.parse_args()
    cfg = C.load_config()
    stage = D.Stage0(args.stage0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "states").mkdir(exist_ok=True)
    lam_grid = [float(x) for x in cfg["krr"]["lambdas"]]
    mults = [float(x) for x in cfg["krr"]["bandwidth_multipliers"]]
    train = stage.train
    y = stage.y
    rows = {}
    path_rows = []
    best_rows = []
    for arm in ("BASE", "REAL"):
        design = stage.design("F_BASE" if arm == "BASE" else "F_REAL")
        X = np.asarray(design["X"], np.float64)
        # Train-only per-column standardisation, as specified for the RBF reference.
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X = (X - mu) / sd
        Xt = X[train]
        t0 = time.time()
        d = pairwise_distances(Xt, Xt, metric="euclidean")
        iu = np.triu_indices(train.size, 1)
        vals = d[iu]
        nz = vals[vals > 1e-12]
        median = float(np.median(nz)) if nz.size else 1.0
        d_valid = pairwise_distances(X[stage.rows["valid"]], Xt, metric="euclidean")
        d_test = pairwise_distances(X[stage.rows["test"]], Xt, metric="euclidean")
        _log(f"{arm}: D={X.shape[1]} median nonzero distance={median:.6f}")
        arm_best = None
        for mult in mults:
            sigma = median * mult
            def kernel(dd: np.ndarray) -> np.ndarray:
                return np.exp(-(dd * dd) / (2.0 * sigma * sigma))
            K = kernel(d)
            rowmean = K.mean(axis=1)
            colmean = K.mean(axis=0)
            total = float(K.mean())
            Kc = K - rowmean[:, None] - colmean[None, :] + total
            yc = stage.y_std[train] - stage.y_std[train].mean()
            w, V = np.linalg.eigh(Kc)
            w = np.clip(w, 0.0, None)
            proj = V.T @ yc
            n = train.size
            kv = kernel(d_valid)
            kt = kernel(d_test)
            kv_c = kv - kv.mean(axis=1)[:, None] - colmean[None, :] + total
            kt_c = kt - kt.mean(axis=1)[:, None] - colmean[None, :] + total
            ktr_c = Kc
            for lam in lam_grid:
                alpha = V @ (proj / (w + n * float(lam)))
                metrics = {"arm": arm, "kernel": "rbf", "bandwidth_multiplier": mult,
                           "sigma": sigma, "lambda": lam, "n_features": int(X.shape[1])}
                preds = {}
                for split, kk in (("train", ktr_c), ("valid", kv_c), ("test", kt_c)):
                    idx = stage.rows[split]
                    z = (kk @ alpha) + stage.y_std[train].mean()
                    p = float(stage.scaler["mean"]) + float(stage.scaler["std"]) * z
                    preds[split] = p
                    metrics[f"{split}_rmse"] = C.rmse(y[idx], p)
                    metrics[f"{split}_mae"] = C.mae(y[idx], p)
                path_rows.append(metrics)
                cur = (metrics["valid_rmse"], lam, mult)
                if arm_best is None:
                    arm_best = (cur, metrics, preds, alpha, sigma)
                else:
                    b = arm_best[0]
                    better = (cur[0] < b[0] - 1e-12 or
                              (abs(cur[0] - b[0]) <= 1e-12 and cur[1] > b[1]) or
                              (abs(cur[0] - b[0]) <= 1e-12 and cur[1] == b[1] and cur[2] > b[2]))
                    if better:
                        arm_best = (cur, metrics, preds, alpha, sigma)
            _log(f"{arm}: mult={mult} best-so-far valid={arm_best[1]['valid_rmse']:.6f}")
        (_cur, best, preds, alpha, sigma) = arm_best
        best = dict(best)
        best["selected"] = True
        best["wall_seconds"] = float(time.time() - t0)
        best_rows.append(best)
        for split in ("train", "valid", "test"):
            idx = stage.rows[split]
            np.savez_compressed(out / "predictions" / f"KRR_{arm}_{split}.npz",
                                row_id=idx, y_true=y[idx], pred=preds[split],
                                arm=np.asarray([f"KRR_{arm}"]), split=np.asarray([split]))
            C.write_csv(out / "predictions" / f"KRR_{arm}_{split}.csv", [
                {"row_id": int(r), "split": split, "y_true": float(yy), "pred": float(pp)}
                for r, yy, pp in zip(idx, y[idx], preds[split])
            ])
        np.savez_compressed(out / "states" / f"KRR_{arm}.npz",
                            alpha=alpha, sigma=np.asarray([sigma]),
                            bandwidth_multiplier=np.asarray([best["bandwidth_multiplier"]]),
                            lambda_=np.asarray([best["lambda"]]),
                            n_features=np.asarray([X.shape[1]]),
                            feature_mean=mu, feature_std=sd,
                            y_mean=np.asarray([float(stage.scaler["mean"])]),
                            y_scale=np.asarray([float(stage.scaler["std"])]))
        _log(f"{arm}: selected sigma={sigma:.6f} lambda={best['lambda']} "
             f"valid={best['valid_rmse']:.6f} test={best['test_rmse']:.6f}")
    C.write_csv(out / "regularization_path.csv", path_rows)
    C.write_csv(out / "main_table.csv", best_rows)
    C.write_json(out / "run_meta.json", {
        "round": cfg["round_id"], "commit": C.git_commit(), "created_at": C.now_utc(),
        "stage0": str(stage.path), "stage0_sha256": C.sha256_file(stage.path),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "bandwidth_multipliers": mults, "lambdas": lam_grid,
        "regularization_convention": "mean((y-f)^2) + lambda*||f||_H^2 => (Kc + n*lambda*I) alpha = yc",
    })


if __name__ == "__main__":
    main()