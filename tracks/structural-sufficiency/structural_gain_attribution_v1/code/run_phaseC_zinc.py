"""Phase C: the four minimal ZINC transfer arms.

``ZX0`` (XGB, Z), ``ZX1`` (XGB, Z+Z-CF), ``ZN0`` (MLP, Z), ``ZN1`` (MLP, Z+Z-CF).

XGBoost recipe = Phase A's frozen recipe with ``objective='reg:absoluteerror'``
(nothing else changes).  MLP recipe = Phase B's frozen architecture and training
recipe with an L1 loss on the Train-standardised target (evaluated after inverting
the standardisation).  No fifth model, no tuning, no extra seed.

Outputs (``--out-dir``): ``<arm>.json``, ``<arm>_pred.npz``, and for the MLP arms a
``<arm>_curve.csv`` + ``<arm>_best.pt``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import sga_common as K
from run_phaseB_mlp import BATCH, CLIP, CLIP_FEATURE, DROPOUT, EPOCHS, HIDDEN, LR, WEIGHT_DECAY
from run_phaseA_xgb import XGB_PARAMS

ARMS = {
    "ZX0": ("xgb", []),
    "ZX1": ("xgb", ["C", "F"]),
    "ZN0": ("mlp", []),
    "ZN1": ("mlp", ["C", "F"]),
}
SIZE_COLUMNS = (0, 1)  # n_atoms, n_bonds: the only continuous columns of Z


def load_design(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def csr(d: dict[str, np.ndarray], prefix: str) -> sp.csr_matrix:
    indices = np.asarray(d[f"{prefix}_indices"], dtype=np.int64)
    return sp.csr_matrix(
        (np.asarray(d[f"{prefix}_data"], dtype=np.float32), indices,
         np.asarray(d[f"{prefix}_indptr"], dtype=np.int64)),
        shape=(int(d["rows_train"].size + d["rows_valid"].size + d["rows_test"].size),
               int(indices.max()) + 1),
    )


def dense(mat: sp.csr_matrix) -> np.ndarray:
    return np.ascontiguousarray(mat.todense(), dtype=np.float32)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))))


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr

    rho = spearmanr(y, pred).statistic if y.size > 2 else float("nan")
    return {
        "n": int(y.size), "mae": mae(y, pred),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "spearman": float(rho),
        "median_ae": float(np.median(np.abs(y - pred))),
    }


# ---------------------------------------------------------------------------
# XGB arms
# ---------------------------------------------------------------------------


def run_xgb(arm: str, extra: list[str], d: dict[str, np.ndarray], out_dir: Path,
            created: str, t0: float) -> None:
    import xgboost as xgb

    z = csr(d, "Z")
    blocks = [csr(d, name) for name in extra]
    x = np.concatenate([dense(z)] + [dense(b) for b in blocks], axis=1)
    width = x.shape[1]
    offsets = {"Z": (0, z.shape[1])}
    col = z.shape[1]
    for name, b in zip(extra, blocks):
        offsets[name] = (col, b.shape[1])
        col += b.shape[1]
    y = np.asarray(d["y"], dtype=np.float64)
    tr, va, te = (np.asarray(d[f"rows_{k}"], dtype=np.int64) for k in ("train", "valid", "test"))
    params = {**XGB_PARAMS, "objective": "reg:absoluteerror"}
    model = xgb.XGBRegressor(**params)
    t_fit = time.perf_counter()
    model.fit(x[tr], y[tr])
    fit_seconds = time.perf_counter() - t_fit
    per_split, preds = {}, {}
    for name, rows in (("train", tr), ("valid", va), ("test", te)):
        p = model.predict(x[rows])
        preds[name] = np.asarray(p, dtype=np.float64)
        per_split[name] = regression_metrics(y[rows], preds[name])
        if name == "test":
            per_split[name]["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST"
    del x
    usage = _feature_usage(model.get_booster(), offsets, width)
    model_path = out_dir / f"{arm}_model.json"
    model.get_booster().save_model(model_path)
    payload = {
        "round_id": K.ROUND_ID, "phase": "C: ZINC transfer (XGB)", "arm": arm,
        "extra_blocks": extra, "created_utc": created,
        "recipe": {**params, "eval_metric": "mae", "dense": True},
        "n_columns": int(width), "fit_seconds": float(fit_seconds),
        "metrics": per_split, "feature_usage": usage,
        "model_dump": {"path": str(model_path), "size_bytes": int(model_path.stat().st_size)},
        "git": K.git_state(), "environment": K.environment(),
        "seconds": float(time.perf_counter() - t0),
    }
    (out_dir / f"{arm}.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                    default=K._json_default) + "\n")
    np.savez_compressed(out_dir / f"{arm}_pred.npz", train=preds["train"], valid=preds["valid"],
                        test=preds["test"], train_rows=tr, valid_rows=va, test_rows=te, y=y)
    print(f"[{arm}] valid MAE={per_split['valid']['mae']:.4f} test MAE={per_split['test']['mae']:.4f} "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)


def _feature_usage(booster, offsets: dict[str, tuple[int, int]], width: int) -> dict[str, Any]:
    gain = booster.get_score(importance_type="total_gain")
    gains = {int(k[1:]): float(v) for k, v in gain.items()}
    total = sum(gains.values())
    z_lo, z_size = offsets["Z"]
    struct_cols = {c: g for c, g in gains.items() if c >= z_size}
    out = {
        "total_gain": float(total),
        "Z_block_gain": float(sum(g for c, g in gains.items() if c < z_size)),
        "structure_block_gain": float(sum(struct_cols.values())),
        "structure_block_gain_share": float(sum(struct_cols.values()) / total) if total else None,
        "n_columns_total": int(width), "n_columns_Z": int(z_size),
        "n_used_features_total": int(len(gains)),
        "n_used_features_structure": int(len(struct_cols)),
        "blocks": {},
    }
    for name, (lo, size) in offsets.items():
        cols = {c: g for c, g in gains.items() if lo <= c < lo + size}
        out["blocks"][name] = {"offset": int(lo), "size": int(size), "n_used": int(len(cols)),
                               "gain": float(sum(cols.values()))}
    return out


# ---------------------------------------------------------------------------
# MLP arms
# ---------------------------------------------------------------------------


def run_mlp(arm: str, extra: list[str], d: dict[str, np.ndarray], out_dir: Path,
            created: str, t0: float, epochs: int) -> None:
    import torch
    import torch.nn.functional as Fn

    from run_phaseB_mlp import make_model, n_parameters, shared_param_hash

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = dense(csr(d, "Z"))
    blocks = [dense(csr(d, name)) for name in extra]
    xh = z
    xq = np.concatenate(blocks, axis=1) if blocks else None
    y = np.asarray(d["y"], dtype=np.float64)
    tr, va, te = (np.asarray(d[f"rows_{k}"], dtype=np.int64) for k in ("train", "valid", "test"))

    # continuous size columns: Train-only standardisation; count blocks keep the fixed scaling
    size_mu = xh[tr][:, list(SIZE_COLUMNS)].mean(axis=0)
    size_sd = xh[tr][:, list(SIZE_COLUMNS)].std(axis=0)
    size_sd = np.where(size_sd > 0, size_sd, 1.0)
    xh[:, list(SIZE_COLUMNS)] = (xh[:, list(SIZE_COLUMNS)] - size_mu) / size_sd
    np.clip(xh, -CLIP_FEATURE, CLIP_FEATURE, out=xh)

    y_mu = float(y[tr].mean())
    y_sd = float(y[tr].std())
    y_std = (y - y_mu) / y_sd

    model, shared_hash = make_model(xh.shape[1], 0 if xq is None else xq.shape[1])
    params = n_parameters(model)
    model.to(device)
    print(f"[{arm}] design H{xh.shape} Q{None if xq is None else xq.shape} device={device} "
          f"init {shared_hash[:12]} params {params['total']}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_train = tr.size
    rng = np.random.default_rng(K.SEED)
    orders = [rng.permutation(n_train) for _ in range(epochs)]
    xh_t = torch.from_numpy(xh[tr]).to(device)
    xq_t = torch.from_numpy(xq[tr]).to(device) if xq is not None else None
    y_t = torch.from_numpy(y_std[tr].astype(np.float32)).to(device)
    curve: list[dict[str, Any]] = []
    best = {"epoch": -1, "valid_mae": np.inf}
    best_state = None
    best_preds: dict[str, np.ndarray] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = orders[epoch - 1]
        for start in range(0, n_train, BATCH):
            sel = order[start: start + BATCH]
            opt.zero_grad(set_to_none=True)
            logit = model(xh_t[sel], xq_t[sel]) if xq_t is not None else model(xh_t[sel])
            loss = Fn.l1_loss(logit, y_t[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        row: dict[str, Any] = {"epoch": epoch}
        preds = {}
        for name, rows in (("train", tr), ("valid", va), ("test", te)):
            p_std = _predict(model, xh, xq, rows, device)
            p = p_std * y_sd + y_mu
            preds[name] = p
            row[f"{name}_mae"] = mae(y[rows], p)
        curve.append(row)
        if row["valid_mae"] < best["valid_mae"]:
            best = {"epoch": epoch, "valid_mae": row["valid_mae"]}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = {k: v.copy() for k, v in preds.items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{arm}] epoch {epoch:02d} train={row['train_mae']:.4f} "
                  f"valid={row['valid_mae']:.4f} test={row['test_mae']:.4f}", flush=True)

    ckpt = out_dir / f"{arm}_best.pt"
    torch.save({"arm": arm, "epoch": best["epoch"], "state_dict": best_state, "params": params,
                "shared_init_hash": shared_hash, "target_standardisation": {"mean": y_mu, "std": y_sd}}, ckpt)
    with (out_dir / f"{arm}_curve.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    per_split = {}
    for name, rows in (("train", tr), ("valid", va), ("test", te)):
        per_split[name] = regression_metrics(y[rows], best_preds[name])
        if name == "test":
            per_split[name]["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST"
    payload = {
        "round_id": K.ROUND_ID, "phase": "C: ZINC transfer (MLP)", "arm": arm,
        "extra_blocks": extra, "created_utc": created,
        "architecture": "Linear(128)->ReLU->Dropout(0.1)->Linear(128)->ReLU->Dropout(0.1)->Linear(1); "
                        "first layer = W_H*Z + W_Q*Q + b for ZN1",
        "recipe": {"epochs": epochs, "batch_size": BATCH, "optimizer": "AdamW", "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "grad_clip": CLIP, "dropout": DROPOUT,
                   "loss": "L1 on the Train-standardised target",
                   "scheduler": None, "early_stopping": None,
                   "checkpoint": "best Valid MAE (original units), ties to the earlier epoch"},
        "target_standardisation": {"mean": y_mu, "std": y_sd, "fit_on": "Train only"},
        "size_column_standardisation": {"mean": size_mu.tolist(), "std": size_sd.tolist(),
                                        "columns": ["n_atoms", "n_bonds"], "fit_on": "Train only"},
        "seed": K.SEED, "device": device, "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": params, "shared_init_hash": shared_hash,
        "best_epoch": int(best["epoch"]), "metrics": per_split, "curve": curve,
        "checkpoint": {"path": str(ckpt), "size_bytes": int(ckpt.stat().st_size)},
        "git": K.git_state(), "environment": K.environment(),
        "seconds": float(time.perf_counter() - t0),
    }
    (out_dir / f"{arm}.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                    default=K._json_default) + "\n")
    np.savez_compressed(out_dir / f"{arm}_pred.npz", train=best_preds["train"], valid=best_preds["valid"],
                        test=best_preds["test"], train_rows=tr, valid_rows=va, test_rows=te, y=y)
    print(f"[{arm}] best epoch {best['epoch']} valid MAE={per_split['valid']['mae']:.4f} "
          f"test MAE={per_split['test']['mae']:.4f} in {time.perf_counter() - t0:.1f}s", flush=True)


def _predict(model, xh: np.ndarray, xq, rows: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    import torch

    model.eval()
    out = np.empty(rows.size, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, batch):
            sel = rows[start: start + batch]
            x = torch.from_numpy(xh[sel]).to(device)
            pred = model(x, torch.from_numpy(xq[sel]).to(device)) if xq is not None else model(x)
            out[start: start + sel.size] = pred.detach().cpu().numpy().astype(np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--design", default=str(K.CACHE / "zinc_design.npz"))
    ap.add_argument("--out-dir", default=str(K.RUNS / "phaseC"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    d = load_design(Path(args.design))
    kind, extra = ARMS[args.arm]
    if kind == "xgb":
        run_xgb(args.arm, extra, d, out_dir, created, t0)
    else:
        run_mlp(args.arm, extra, d, out_dir, created, t0, args.epochs)


if __name__ == "__main__":
    main()
