"""Reconstructors and reconstruction metrics for the audit.

Two complexity levels, both fitted on official-train atoms only:

* :class:`RidgeFit` -- closed-form ridge on standardised features;
* :func:`mlp_fit` -- a 2-hidden-layer SiLU MLP with early stopping.

``λ`` / early-stop epoch are chosen on a **train-internal monitor split of
molecules** (never on valid).  Valid is used exactly once per fit, for scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import pb_common as P

LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# ---------------------------------------------------------------------------
# standardisation
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    raw_scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, floor_rel: float = 1e-6) -> "Standardizer":
        x64 = x.astype(np.float64)
        mean = x64.mean(0)
        raw = x64.std(0)
        floor = max(floor_rel, 1e-12) * max(float(np.median(raw)), 1e-12)
        scale = np.maximum(raw, floor)
        return cls(mean=mean, scale=scale, raw_scale=raw)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float64) - self.mean) / self.scale).astype(np.float32)

    def to_json(self) -> dict:
        return {"n": int(len(self.mean)), "mean_sha": P.sha256_array(self.mean.astype(np.float32)),
                "scale_sha": P.sha256_array(self.scale.astype(np.float32)),
                "min_scale": float(self.scale.min()), "max_scale": float(self.scale.max())}


# ---------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------

class RidgeFit:
    def __init__(self, lam: float):
        self.lam = float(lam)
        self.W: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, chunk: int = 16384) -> "RidgeFit":
        d = X.shape[1]
        A = np.zeros((d, d), dtype=np.float64)
        B = np.zeros((d, Y.shape[1]), dtype=np.float64)
        for s in range(0, len(X), chunk):
            xc = X[s:s + chunk].astype(np.float64)
            A += xc.T @ xc
            B += xc.T @ Y[s:s + chunk].astype(np.float64)
        A.flat[:: d + 1] += self.lam
        self.W = np.linalg.solve(A, B)
        return self

    def predict(self, X: np.ndarray, *, chunk: int = 65536) -> np.ndarray:
        out = np.empty((len(X), self.W.shape[1]), dtype=np.float64)
        for s in range(0, len(X), chunk):
            out[s:s + chunk] = X[s:s + chunk].astype(np.float64) @ self.W
        return out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp_fit(X: np.ndarray, Y: np.ndarray, X_mon: np.ndarray, Y_mon: np.ndarray,
            *, seed: int = P.MLP_SEED, max_epochs: int = 60, patience: int = 8,
            batch: int = 4096, lr: float = 1e-3, wd: float = 1e-5,
            hidden=(256, 128), log=None) -> tuple[_MLP, dict]:
    torch.manual_seed(int(seed))
    model = _MLP(X.shape[1], Y.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
    xm = torch.from_numpy(np.ascontiguousarray(X_mon, dtype=np.float32))
    ym = torch.from_numpy(np.ascontiguousarray(Y_mon, dtype=np.float32))
    n = len(x)
    best = {"sse": float("inf"), "epoch": -1, "state": None}
    bad = 0
    g = torch.Generator().manual_seed(int(seed) + 17)
    hist = []
    for ep in range(int(max_epochs)):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad(set_to_none=True)
            pred = model(x[idx])
            loss = ((pred - y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            sse = float(((model(xm) - ym) ** 2).sum())
        hist.append({"epoch": ep, "train_mse": tot / n, "monitor_sse": sse})
        if sse < best["sse"] * (1.0 - 1e-6):
            best = {"sse": sse, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    info = {"best_epoch": int(best["epoch"]), "monitor_sse": float(best["sse"]),
            "epochs_run": len(hist), "hidden": list(hidden), "seed": int(seed),
            "curve": hist}
    return model, info


@torch.no_grad()
def mlp_predict(model: _MLP, X: np.ndarray, *, chunk: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(X), chunk):
        out.append(model(torch.from_numpy(np.ascontiguousarray(X[s:s + chunk],
                                                              dtype=np.float32))).numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def effective_rank(cov: np.ndarray) -> dict:
    ev = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
    ev = np.clip(ev, 0.0, None)
    tot = float(ev.sum())
    if tot <= 0:
        return {"entropy_rank": 0.0, "participation_ratio": 0.0, "top1_share": 0.0,
                "top1_eig": 0.0, "trace": tot}
    p = ev / tot
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    return {
        "entropy_rank": float(np.exp(ent)),
        "participation_ratio": float(tot ** 2 / float((ev ** 2).sum())),
        "top1_share": float(ev[-1] / tot),
        "top1_eig": float(ev[-1]),
        "trace": tot,
        "n_dims": int(len(ev)),
    }


def block_metrics(pred: np.ndarray, true: np.ndarray, alldeg: np.ndarray,
                  blocks: Mapping[str, tuple[int, int]], train_std: np.ndarray) -> dict:
    """Per-channel and per-block R² on already standardised targets.

    ``pred``/``true`` are in train-standardised units (train mean removed).
    ``alldeg`` marks near-constant channels; they are reported but excluded from
    the block aggregate.
    """
    sse = ((pred - true) ** 2)
    sst = true ** 2
    per_ch = 1.0 - sse.sum(0) / np.maximum(sst.sum(0), 1e-12)
    out = {}
    for name, (a, b) in blocks.items():
        keep = ~alldeg[a:b]
        n_keep = int(keep.sum())
        if n_keep == 0:
            out[name] = {"r2": None, "n_channels": 0, "note": "all channels degenerate"}
            continue
        sse_b = float(sse[:, a:b][:, keep].sum())
        sst_b = float(sst[:, a:b][:, keep].sum())
        resid = (pred[:, a:b] - true[:, a:b])[:, keep]
        cov = np.cov(resid.T) if resid.shape[1] > 1 else np.atleast_2d(resid.var(0))
        out[name] = {
            "r2": float(1.0 - sse_b / max(sst_b, 1e-12)),
            "standardised_rmse": float(np.sqrt(sse_b / max(resid.size, 1))),
            "n_channels": int(b - a),
            "n_nondegenerate": n_keep,
            "n_degenerate": int((alldeg[a:b]).sum()),
            "per_channel_r2": [float(v) for v in per_ch[a:b]],
            "residual_rank": effective_rank(cov),
            "target_std_train": [float(v) for v in train_std[a:b]],
        }
    return out


def degenerate_channels(train_std: np.ndarray, rel_floor: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """Channels with (near-)zero train variance: ``(degenerate, near_constant)``."""
    med = float(np.median(train_std))
    degenerate = train_std < 1e-6
    near = (train_std < rel_floor * med) & ~degenerate
    return degenerate, near


def fit_predict_ridge(Xtr, Ytr, Xmon, Ymon, Xva, *, lambdas=LAMBDA_GRID) -> tuple[np.ndarray, dict]:
    """Ridge with λ chosen on the train-internal monitor (aggregate monitor R²)."""
    sst_mon = float((Ymon ** 2).sum())
    rows = []
    best = None
    for lam in lambdas:
        fit = RidgeFit(lam).fit(Xtr, Ytr)
        p = fit.predict(Xmon)
        sse = float(((p - Ymon) ** 2).sum())
        r2 = 1.0 - sse / max(sst_mon, 1e-12)
        rows.append({"lambda": float(lam), "monitor_r2": float(r2)})
        if best is None or r2 > best[1]:
            best = (float(lam), float(r2))
    fit = RidgeFit(best[0]).fit(Xtr, Ytr)
    return fit.predict(Xva), {"lambda": best[0], "monitor_r2": best[1], "grid": rows}


__all__ = ["LAMBDA_GRID", "Standardizer", "RidgeFit", "mlp_fit", "mlp_predict",
           "effective_rank", "block_metrics", "degenerate_channels", "fit_predict_ridge"]
