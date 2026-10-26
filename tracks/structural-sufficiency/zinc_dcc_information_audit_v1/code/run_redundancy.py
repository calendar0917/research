#!/usr/bin/env python
"""A4: label-free conditional redundancy of J_raw / J_latent given the full
513-d DCC-L reader input z and the 33-d global structural descriptor.

Fits ridge (main) and a fixed small MLP (sensitivity) on Fit8k, reports
out-of-sample R^2 on ProbeFit (and AuditEval as a label-free second check),
residual spectra, per-dimension results and the pre-registered STOP gate.

No labels are read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_common as AC  # noqa: E402
import joint_stats as JS  # noqa: E402

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
N_FOLDS = 5
CV_SEED = 20261026
MLP_SEED = 20261026
MLP_HIDDEN = 128
MLP_EPOCHS = 300
MLP_PATIENCE = 20
MLP_BATCH = 256
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_HOLDOUT = 1000
CACHE_DIR = AC.ANALYSIS / "cache"


def j_raw_names():
    names = []
    for r in JS.RELATIONS:
        for a, b in JS.UNORDERED_PAIRS:
            names.append(f"{r}_freq_{a}_{b}")
        names.append(f"{r}_log1p_count")
        names.append(f"{r}_empty")
    return names


J_RAW_NAMES = j_raw_names()
J_LATENT_NAMES = [f"T_{r}" for r in JS.RELATIONS]


def valid_dims(y: np.ndarray, *, abs_thr=1e-6, rel_thr=1e-3,
               zero_thr=0.99, const_thr=1e-3) -> tuple[np.ndarray, dict]:
    std = np.asarray(y, dtype=np.float64).std(axis=0)
    zero_fraction = (np.abs(np.asarray(y, dtype=np.float64)) < 1e-12).mean(axis=0)
    med = float(np.median(std)) if len(std) else 0.0
    near_constant = (zero_fraction >= zero_thr) | (std <= const_thr)
    mask = (std >= abs_thr) & (std >= rel_thr * max(med, abs_thr)) & (~near_constant)
    return mask, {
        "std": std.tolist(),
        "zero_fraction": zero_fraction.tolist(),
        "median_std": med,
        "valid_mask": mask.tolist(),
        "near_constant_indices": np.where(near_constant)[0].tolist(),
        "n_valid": int(mask.sum()),
        "n_total": int(len(mask)),
        "invalid_indices": np.where(~mask)[0].tolist(),
    }


def standardize_fit(x: np.ndarray):
    mu = np.asarray(x, dtype=np.float64).mean(axis=0)
    sd = np.asarray(x, dtype=np.float64).std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def apply_std(x, mu, sd):
    return (np.asarray(x, dtype=np.float64) - mu) / sd


def fit_ridge_cv(X, Y, alphas=ALPHAS, n_folds=N_FOLDS, seed=CV_SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = {}
    for alpha in alphas:
        mses = []
        for tr, va in kf.split(X):
            model = Ridge(alpha=float(alpha))
            model.fit(X[tr], Y[tr])
            pred = model.predict(X[va])
            mses.append(float(np.mean((pred - Y[va]) ** 2)))
        scores[float(alpha)] = float(np.mean(mses))
    best_alpha = min(scores, key=scores.get)
    model = Ridge(alpha=float(best_alpha))
    model.fit(X, Y)
    return model, {"best_alpha": float(best_alpha), "cv_mse": scores}


def block_metrics(Y_true, Y_pred, *, names, rows=None, top_k=5):
    """Y_true/Y_pred are standardized on the fit set; metrics on the eval set."""
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    n, d = Y_true.shape
    resid = Y_true - Y_pred
    sse_dim = np.sum(resid ** 2, axis=0)
    mean_true = Y_true.mean(axis=0)
    sst_dim = np.sum((Y_true - mean_true) ** 2, axis=0)
    r2_dim = np.where(sst_dim > 1e-12, 1.0 - sse_dim / np.maximum(sst_dim, 1e-12), 0.0)
    # baseline: predict the fit mean (0 in these standardized coordinates)
    sst_fitmean_dim = np.sum(Y_true ** 2, axis=0)
    r2_dim_fitmean = np.where(sst_fitmean_dim > 1e-12,
                              1.0 - sse_dim / np.maximum(sst_fitmean_dim, 1e-12), 0.0)
    sse = float(sse_dim.sum())
    sst = float(sst_dim.sum())
    r2_block = 1.0 - sse / max(sst, 1e-12)
    r2_block_fitmean = 1.0 - sse / max(float(sst_fitmean_dim.sum()), 1e-12)
    cov = np.cov(resid, rowvar=False) if d > 1 else np.array([[float(np.var(resid))]])
    ev = np.clip(np.linalg.eigvalsh((cov + cov.T) / 2.0)[::-1], 0.0, None)
    total = float(ev.sum())
    pr = float(total * total / float(np.sum(ev * ev))) if total > 0 else 0.0
    absr = np.abs(resid)
    top = []
    if rows is not None and n > 0:
        order = np.argsort(np.sum(resid ** 2, axis=1))[::-1][:top_k]
        for i in order:
            top.append({
                "row": int(rows[i]),
                "l2_std": float(np.sqrt(np.sum(resid[i] ** 2))),
                "residual": {names[j]: float(resid[i, j]) for j in range(d)},
                "target_std": {names[j]: float(Y_true[i, j]) for j in range(d)},
                "pred_std": {names[j]: float(Y_pred[i, j]) for j in range(d)},
            })
    return {
        "n": int(n),
        "dim": int(d),
        "r2_block": float(r2_block),
        "r2_block_fitmean": float(r2_block_fitmean),
        "residual_variance_share": float(max(0.0, 1.0 - r2_block)),
        "rmse_std": float(np.sqrt(np.mean(resid ** 2))),
        "mae_std": float(np.mean(absr)),
        "median_ae_std": float(np.median(absr)),
        "p90_ae_std": float(np.percentile(absr, 90)),
        "r2_per_dim": {names[j]: float(r2_dim[j]) for j in range(d)},
        "r2_per_dim_fitmean": {names[j]: float(r2_dim_fitmean[j]) for j in range(d)},
        "r2_per_dim_min": float(np.min(r2_dim)) if d else None,
        "r2_per_dim_median": float(np.median(r2_dim)) if d else None,
        "residual_cov_pr": pr,
        "residual_cov_top1": float(ev[0] / total) if total > 0 else 0.0,
        "residual_cov_top5": float(ev[:5].sum() / total) if total > 0 else 0.0,
        "residual_cov_eigenvalues_top20": ev[:20].tolist(),
        "top_residual_rows": top,
    }


def train_mlp(Xtr, Ytr, Xva, Yva, Xte, *, seed=MLP_SEED, epochs=MLP_EPOCHS,
              patience=MLP_PATIENCE, hidden=MLP_HIDDEN, batch=MLP_BATCH,
              lr=MLP_LR, wd=MLP_WD):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, Ytr.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)
    Yva_t = torch.as_tensor(Yva, dtype=torch.float32)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32)
    best = None
    best_state = None
    best_epoch = -1
    bad = 0
    gen = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr_t), generator=gen)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = float(F.mse_loss(model(Xva_t), Yva_t))
        if best is None or va < best:
            best = va
            best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).numpy()
    return pred, {"best_epoch": int(best_epoch), "holdout_mse": float(best), "epochs_run": int(ep),
                  "hidden": int(hidden), "patience": int(patience)}


def evaluate_target(Xfit, Xprobe, Xaudit, Yfit, Yprobe, Yaudit, *, names,
                    rows_probe=None, rows_audit=None, do_mlp=True):
    mask, vinfo = valid_dims(Yfit)
    Yfit_v = Yfit[:, mask]
    Yprobe_v = Yprobe[:, mask]
    Yaudit_v = Yaudit[:, mask]
    xmu, xsd = standardize_fit(Xfit)
    ymu, ysd = standardize_fit(Yfit_v)
    Xf = apply_std(Xfit, xmu, xsd)
    Xp = apply_std(Xprobe, xmu, xsd)
    Xa = apply_std(Xaudit, xmu, xsd)
    Yf = apply_std(Yfit_v, ymu, ysd)
    Yp = apply_std(Yprobe_v, ymu, ysd)
    Ya = apply_std(Yaudit_v, ymu, ysd)
    model, cv = fit_ridge_cv(Xf, Yf)
    pred_p = model.predict(Xp)
    pred_a = model.predict(Xa)
    out = {
        "valid_dims": {k: v for k, v in vinfo.items() if k != "std"},
        "valid_dim_names": [names[i] for i in np.where(mask)[0]],
        "ridge": {
            "alpha": cv,
            "probe": block_metrics(Yp, pred_p, names=[names[i] for i in np.where(mask)[0]], rows=rows_probe),
            "audit": block_metrics(Ya, pred_a, names=[names[i] for i in np.where(mask)[0]], rows=rows_audit),
        },
    }
    if do_mlp:
        # fixed internal holdout from Fit, label-free
        rng = np.random.default_rng(MLP_SEED)
        ho = rng.choice(len(Xf), size=min(MLP_HOLDOUT, len(Xf)), replace=False)
        tr = np.setdiff1d(np.arange(len(Xf)), ho)
        pred_mlp, minfo = train_mlp(Xf[tr], Yf[tr], Xf[ho], Yf[ho], Xp)
        out["mlp"] = {
            "info": minfo,
            "probe": block_metrics(Yp, pred_mlp, names=[names[i] for i in np.where(mask)[0]], rows=rows_probe),
        }
    return out


def main() -> int:
    t0 = time.time()
    AC.install_load_guard()
    cfg = AC.load_audit_config()
    split = AC.read_json(AC.CONFIGS / "split_fit8k_probe_audit.json")
    fit = np.asarray(split["rows"]["fit"], dtype=np.int64)
    probe = np.asarray(split["rows"]["probe"], dtype=np.int64)
    audit = np.asarray(split["rows"]["audit"], dtype=np.int64)
    cache = AC.load_train_cache()
    g = np.asarray(cache["global_feat"], dtype=np.float64)
    j_raw = np.load(AC.ANALYSIS / "j_raw.npy").astype(np.float64)
    j_raw_null = np.load(AC.ANALYSIS / "j_raw_null.npy").astype(np.float64)
    z = {name: np.load(CACHE_DIR / f"z_{name}.npy").astype(np.float64) for name in ("D0", "D1")}
    jl = {name: np.load(AC.ANALYSIS / f"j_latent_{name}.npy").astype(np.float64) for name in ("D0", "D1")}
    jln = {name: np.load(AC.ANALYSIS / f"j_latent_null_{name}.npy").astype(np.float64) for name in ("D0", "D1")}

    features = {
        "z": {name: z[name] for name in z},
        "g": {name: g for name in z},
        "z_no_g": {name: np.concatenate([z[name][:, :288], z[name][:, 321:]], axis=1) for name in z},
    }
    results = {"protocol": cfg["protocol_id"], "targets": {}, "gate": {}}

    # Main: J_raw and J_latent vs z / g / z_no_g, per seed.
    for target_name, target in (("J_raw", j_raw), ("J_raw_null", j_raw_null)):
        results["targets"][target_name] = {}
        for feat_name in ("z", "g", "z_no_g"):
            for seed_name in ("D0", "D1"):
                key = f"{feat_name}_{seed_name}"
                print(f"[A4] {target_name} {key}", flush=True)
                results["targets"][target_name][key] = evaluate_target(
                    features[feat_name][seed_name][fit], features[feat_name][seed_name][probe],
                    features[feat_name][seed_name][audit],
                    target[fit], target[probe], target[audit],
                    names=J_RAW_NAMES, rows_probe=probe, rows_audit=audit,
                    do_mlp=(feat_name == "z" and seed_name in ("D0", "D1")),
                )
    for seed_name in ("D0", "D1"):
        for target_name, target in ((f"J_latent_{seed_name}", jl[seed_name]),
                                    (f"J_latent_null_{seed_name}", jln[seed_name])):
            results["targets"][target_name] = {}
            for feat_name in ("z", "g", "z_no_g"):
                key = f"{feat_name}_{seed_name}"
                print(f"[A4] {target_name} {key}", flush=True)
                results["targets"][target_name][key] = evaluate_target(
                    features[feat_name][seed_name][fit], features[feat_name][seed_name][probe],
                    features[feat_name][seed_name][audit],
                    target[fit], target[probe], target[audit],
                    names=J_LATENT_NAMES, rows_probe=probe, rows_audit=audit,
                    do_mlp=(feat_name == "z"),
                )

    # Gate: both primary blocks, both seeds, z -> J block R2 >= 0.98.
    gate_rows = []
    gate_specs = [("J_raw", "D0"), ("J_raw", "D1"),
                  ("J_latent_D0", "D0"), ("J_latent_D1", "D1")]
    for target_name, seed_name in gate_specs:
        key = f"z_{seed_name}"
        block = results["targets"][target_name][key]["ridge"]["probe"]
        gate_rows.append({
            "target": target_name, "feature": "z", "seed": seed_name,
            "block_r2_probe": block["r2_block"],
            "residual_variance_share": block["residual_variance_share"],
            "residual_cov_pr": block["residual_cov_pr"],
            "residual_cov_top1": block["residual_cov_top1"],
        })
    primary = gate_rows
    stop = all(r["block_r2_probe"] >= 0.98 and r["residual_variance_share"] <= 0.02
               for r in primary)
    results["gate"] = {
        "rule": "STOP_BEFORE_NEW_GPU iff every primary block (J_raw, J_latent) at every frozen seed has z->J block R2 >= 0.98 and residual variance share <= 0.02",
        "rows": gate_rows,
        "primary_rows": primary,
        "stop_before_new_gpu": bool(stop),
        "note": "engineering screening line, not a Bayes sufficiency theorem; low-variance dimensions are excluded before fitting/scoring",
    }
    results["runtime_seconds"] = float(time.time() - t0)
    AC.write_json(AC.ANALYSIS / "redundancy.json", AC.jsonable(results))
    print(json.dumps({
        "gate_stop": stop,
        "rows": [{k: r[k] for k in ("target", "seed", "block_r2_probe", "residual_variance_share", "residual_cov_pr")} for r in primary],
        "runtime_seconds": results["runtime_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
