"""Shared correction heads (G / S) on a frozen base model's out-of-sample errors.

CPU only. The base model is never loaded and never part of a gradient graph:
the residual records are `r = y - f_base` with `f_base` produced by a model that
did not see the molecule.

Prediction form for both heads:

    m_F = 1[ molecule has >= 1 atom whose k2 support in the base fit is 0 ]
    f_corrected = f_base + m_F * h(context, support, f_base)

* ``G`` sees only coarse context (64 dims: f_base, size, ring rank, atom
  composition, unseen count/fraction, radius-1 parent support of the unseen
  atoms, the fixed 33-dim global descriptor).
* ``S`` additionally sees a shared encoding of the flagged local environment:
  ``psi([x_v, log1p n1(v), log1p n2(v)]) -> 16`` pooled over atoms with
  ``n2 = 0`` (sum and mean), plus unseen count and total atom count.

No trainable table is indexed by an exact ``k2`` id; ``k2`` is only used to look
up support.

Stability choices, fixed before the formal run (and recorded in the protocol
amendment): Adam 2e-3 with per-step cosine to 1e-4, 1500 full-batch steps, C_dev
corrected MAE evaluated every 50 steps and used to keep the best checkpoint.
C_dev is the declared selection split; Q never enters training or selection.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import ad_common as C  # noqa: E402

STEPS = 1500
LR = 2e-3
MIN_LR = 1e-4
EVAL_EVERY = 50
WD_GRID = (0.0, 1e-3)
INITS = (0, 1, 2)
CLIP = 1.0


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


class HeadG(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(Z.G_READ_IN, Z.G_READ_HIDDEN), nn.SiLU(), nn.Linear(Z.G_READ_HIDDEN, 1)
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, ctx, aux, atoms, mask):
        return self.readout(ctx).view(-1)


class HeadS(nn.Module):
    def __init__(self):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(Z.ATOM_IN_DIM, Z.PSI_HIDDEN), nn.SiLU(), nn.Linear(Z.PSI_HIDDEN, Z.POOL_DIM)
        )
        self.readout = nn.Sequential(
            nn.Linear(Z.S_READ_IN, Z.S_READ_HIDDEN), nn.SiLU(), nn.Linear(Z.S_READ_HIDDEN, 1)
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, ctx, aux, atoms, mask):
        # atoms [N, U, 110], mask [N, U]; padding never enters the pooling
        z = self.psi(atoms) * mask.unsqueeze(-1)
        cnt = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled_sum = z.sum(1)
        pooled_mean = pooled_sum / cnt
        return self.readout(torch.cat([ctx, pooled_sum, pooled_mean, aux], dim=1)).view(-1)


def forward_head(model, f):
    return model(f.ctx_t, f.aux_t, f.atoms_t, f.mask_t).view(-1)


def param_counts():
    g, s = HeadG(), HeadS()
    return {"G": sum(p.numel() for p in g.parameters()),
            "S": sum(p.numel() for p in s.parameters())}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


class Features:
    """Per-molecule coarse context + padded unseen-atom block for one base fit."""

    def __init__(self, cache, rows, f_base, sup2, atom_k1, atom_k2, atom_ptr,
                 atom_offset: int = 0):
        n = len(rows)
        n_n1 = np.zeros(n)
        n_n2 = np.zeros(n)
        frac = np.zeros(n)
        min_n1 = np.zeros(n)
        mean_n1 = np.zeros(n)
        min_n2 = np.zeros(n)
        mean_n2 = np.zeros(n)
        atom_blocks = []
        atom_feat_all = np.asarray(cache["atom_feat"])
        for i, m in enumerate(np.asarray(rows, dtype=np.int64).tolist()):
            s0, e0 = int(atom_ptr[m]), int(atom_ptr[m + 1])
            s, e = s0 + int(atom_offset), e0 + int(atom_offset)
            k2 = atom_k2[s:e]
            n2 = sup2[k2].astype(np.float64)
            n1 = sup2[atom_k1[s:e]].astype(np.float64)
            u = n2 == 0
            n_n1[i], n_n2[i] = n1.sum(), n2.sum()
            frac[i] = u.mean()
            if u.any():
                l1, l2 = np.log1p(n1[u]), np.log1p(n2[u])
                min_n1[i], mean_n1[i] = l1.min(), l1.mean()
                min_n2[i], mean_n2[i] = l2.min(), l2.mean()
                atom_blocks.append(
                    np.concatenate([atom_feat_all[s0:e0][u].astype(np.float64),
                                    l1.reshape(-1, 1), l2.reshape(-1, 1)], axis=1))
            else:
                atom_blocks.append(np.zeros((0, Z.ATOM_IN_DIM)))
        n_unseen = np.asarray([b.shape[0] for b in atom_blocks], dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int64)
        coarse = Z.coarse_features(cache)
        self.ctx = Z.molecule_context(
            coarse, rows, f_base, n_unseen, frac, min_n1, mean_n1, min_n2, mean_n2)
        self.aux = np.stack([np.log1p(n_unseen), np.log1p(coarse["n_atoms"][self.rows])], axis=1)
        self.gate = (n_unseen > 0).astype(np.float64)
        self.n_unseen = n_unseen
        self.support_stats = {
            "n_unseen": n_unseen, "unseen_frac": frac,
            "min_n1": min_n1, "mean_n1": mean_n1,
            "min_n2": min_n2, "mean_n2": mean_n2}
        umax = max(1, int(n_unseen.max()) if n else 1)
        atoms = np.zeros((n, umax, Z.ATOM_IN_DIM))
        mask = np.zeros((n, umax))
        for i, b in enumerate(atom_blocks):
            k = b.shape[0]
            if k:
                atoms[i, :k] = b
                mask[i, :k] = 1.0
        self.atoms = atoms
        self.mask = mask


def attach_tensors(f: Features) -> Features:
    f.ctx_t = torch.as_tensor(np.asarray(f.ctx), dtype=torch.float32)
    f.aux_t = torch.as_tensor(np.asarray(f.aux), dtype=torch.float32)
    f.atoms_t = torch.as_tensor(np.asarray(f.atoms), dtype=torch.float32)
    f.mask_t = torch.as_tensor(np.asarray(f.mask), dtype=torch.float32)
    return f


def _scale(std: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Near-constant columns must not be amplified: use scale 1 instead of 1/eps."""
    sd = np.asarray(std, dtype=np.float64).copy()
    sd[~np.isfinite(sd)] = 1.0
    sd[sd < floor] = 1.0
    return sd


class Standardiser:
    """Mean/sigma fitted on C_fit only (molecules: gated rows, atoms: unseen atoms)."""

    def __init__(self, fit: Features):
        g = fit.gate > 0
        self.ctx_mu = fit.ctx[g].mean(0)
        self.ctx_sd = _scale(fit.ctx[g].std(0))
        self.aux_mu = fit.aux[g].mean(0)
        self.aux_sd = _scale(fit.aux[g].std(0))
        m = fit.mask > 0
        at = fit.atoms[m]
        self.at_mu = at.mean(0)
        self.at_sd = _scale(at.std(0))

    def transform(self, src: Features) -> Features:
        o = object.__new__(Features)
        o.rows = src.rows
        o.gate = src.gate
        o.n_unseen = src.n_unseen
        o.mask = src.mask
        o.support_stats = src.support_stats
        o.ctx = (src.ctx - self.ctx_mu) / self.ctx_sd
        o.aux = (src.aux - self.aux_mu) / self.aux_sd
        o.atoms = (src.atoms - self.at_mu) / self.at_sd
        return attach_tensors(o)

    def state(self):
        return {"ctx_mean": self.ctx_mu, "ctx_scale": self.ctx_sd,
                "aux_mean": self.aux_mu, "aux_scale": self.aux_sd,
                "atom_mean": self.at_mu, "atom_scale": self.at_sd}


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------


def _lr(step: int, steps: int) -> float:
    if steps <= 0:
        return LR
    frac = min(1.0, step / steps)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * frac))


def train_head(kind: str, fit: Features, r: np.ndarray, base: np.ndarray,
               wd: float, init: int, dev: Features | None = None,
               dev_base: np.ndarray | None = None, dev_y: np.ndarray | None = None,
               steps: int = STEPS, eval_every: int = EVAL_EVERY):
    torch.manual_seed(10_000 + int(init))
    model = HeadS() if kind == "S" else HeadG()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=float(wd))
    keep = torch.as_tensor(fit.gate > 0)
    y = torch.as_tensor(base + r, dtype=torch.float32)
    b = torch.as_tensor(base, dtype=torch.float32)
    curve = []
    best = {"dev_mae": float("inf"), "step": -1, "state": None}
    dev_keep = None
    if dev is not None:
        # selection metric: C_dev MAE after dropping the 5% of C_dev molecules
        # with the largest *base* absolute error (a function of base and y only,
        # never of the correction). Pre-registered as amendment A2.
        ae = np.abs(dev_base - dev_y)
        k = max(1, int(round(0.05 * len(ae))))
        dev_keep = np.ones(len(ae), dtype=bool)
        dev_keep[np.argsort(-ae)[:k]] = False
    for step in range(steps + 1):
        lr = _lr(step, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        h = forward_head(model, fit)
        corrected = b + keep * h
        loss = F.l1_loss(corrected[keep], y[keep])
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                entry = {"step": step, "fit_l1": float(loss), "lr": lr}
                if dev is not None:
                    hd = forward_head(model, dev)
                    dev_pred = dev_base + dev.gate * hd.numpy().astype(np.float64)
                    dev_mae = float(np.abs(dev_pred - dev_y).mean())
                    dev_trim = float(np.abs(dev_pred - dev_y)[dev_keep].mean())
                    entry["dev_mae"] = dev_mae
                    entry["dev_mae_trim5"] = dev_trim
                    if dev_trim < best["dev_mae"]:
                        best = {"dev_mae": dev_trim, "step": step, "dev_mae_full": dev_mae,
                                "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                curve.append(entry)
        if step == steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, curve, (best["step"] if dev is not None else steps)


@torch.no_grad()
def predict_head(model, f: Features) -> np.ndarray:
    model.eval()
    h = forward_head(model, f).numpy().astype(np.float64)
    return h * f.gate


def metrics(y, base, corr) -> dict:
    ae_b = np.abs(base - y)
    ae_c = np.abs(corr - y)
    d = ae_c - ae_b
    n = len(y)
    k = max(1, int(round(0.01 * n)))
    keep = np.ones(n, dtype=bool)
    keep[np.argsort(-ae_b)[:k]] = False
    dc = corr - base
    return {
        "n": int(n),
        "mae_base": float(ae_b.mean()),
        "mae_corrected": float(ae_c.mean()),
        "delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "frac_improved": float((d < 0).mean()),
        "mean_abs_correction": float(np.abs(dc).mean()),
        "mean_signed_correction": float(dc.mean()),
        "corr_signed_with_residual": float(np.corrcoef(dc, y - base)[0, 1])
        if np.std(dc) > 1e-12 else 0.0,
        "trimmed_mae_base": float(ae_b[keep].mean()),
        "trimmed_mae_corrected": float(ae_c[keep].mean()),
        "trimmed_delta": float(d[keep].mean()),
    }


def boot(y, base, corr, clusters=None, reps=4000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    d = np.abs(corr - y) - np.abs(base - y)
    n = len(y)
    mol = np.asarray([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    out = {"molecule_ci": [float(np.percentile(mol, 2.5)), float(np.percentile(mol, 97.5))],
           "delta_mean": float(d.mean())}
    if clusters is not None:
        uniq = np.unique(clusters)
        idx_by = {u: np.flatnonzero(clusters == u) for u in uniq}
        kk = np.empty(reps)
        for i in range(reps):
            pick = uniq[rng.integers(0, len(uniq), len(uniq))]
            kk[i] = d[np.concatenate([idx_by[u] for u in pick])].mean()
        out["key_cluster_ci"] = [float(np.percentile(kk, 2.5)), float(np.percentile(kk, 97.5))]
        out["n_key_clusters"] = int(len(uniq))
    return out


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------


def key_cluster_ids(rows, target_keys, mol2):
    """Cluster id = the family target key carried by the molecule. Target keys
    inside a family are molecule-disjoint, so this is a partition."""
    kset = {int(k): i for i, k in enumerate(sorted(np.asarray(target_keys).tolist()))}
    out = np.full(len(rows), -1, dtype=np.int64)
    for i, m in enumerate(np.asarray(rows, dtype=np.int64).tolist()):
        hits = [kset[int(k)] for k in mol2[m] if int(k) in kset]
        if hits:
            out[i] = min(hits)
    return out


def build_and_train(seed: int, data: dict, tkeys: dict, mol2, steps: int = STEPS,
                    shuffle_atoms: bool = False, verbose: bool = True) -> dict:
    """Train G and S for one base seed; select wd on C_dev; evaluate on Q."""
    fit, dev = data["f_cfit"], data["f_cdev"]
    r = data["y_cfit"] - data["b_cfit"]
    keep = fit.gate > 0
    if shuffle_atoms:
        def terc(x):
            q = np.quantile(x, [1 / 3, 2 / 3])
            return np.digitize(x, q)
        s1 = terc(fit.aux[:, 1])
        s2 = terc(fit.ctx[:, 0])
        s3 = terc(fit.ctx[:, 7])  # unseen fraction
        rng = np.random.default_rng(4242)
        perm = np.arange(len(s1))
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    ix = np.flatnonzero((s1 == a) & (s2 == b) & (s3 == c))
                    if len(ix) > 1:
                        perm[ix] = ix[rng.permutation(len(ix))]
        fit.atoms = fit.atoms[perm]
        fit.mask = fit.mask[perm]
        attach_tensors(fit)
    rec = {"arms": {}, "wd_selection": {}}
    trained = {}
    for kind in ("G", "S"):
        for wd in WD_GRID:
            for init in INITS:
                key = f"{kind}_wd{wd}_i{init}"
                model, curve, best_step = train_head(
                    kind, fit, r, data["b_cfit"], wd, init,
                    dev=dev, dev_base=data["b_cdev"], dev_y=data["y_cdev"], steps=steps)
                cdev = data["b_cdev"] + predict_head(model, dev)
                cfit = data["b_cfit"] + predict_head(model, fit)
                trained[key] = {
                    "model": model, "kind": kind, "wd": float(wd), "init": int(init),
                    "best_step": int(best_step),
                    "dev_mae": float(np.abs(cdev - data["y_cdev"]).mean()),
                    "dev_mae_trim5": float(np.sort(np.abs(cdev - data["y_cdev"]))[
                        :max(1, int(round(0.95 * len(cdev))))].mean()),
                    "fit_mae": float(np.abs(cfit[keep] - data["y_cfit"][keep]).mean()),
                    "curve": curve,
                    "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                }
                if verbose:
                    print(f"[s{seed} {key}] step={best_step} C_dev={trained[key]['dev_mae']:.6f} "
                          f"C_fit={trained[key]['fit_mae']:.6f}", flush=True)
    for kind in ("G", "S"):
        stats = {wd: float(np.mean([trained[f"{kind}_wd{wd}_i{i}"]["dev_mae_trim5"] for i in INITS]))
                 for wd in WD_GRID}
        rec["wd_selection"][kind] = {"cdev_mean_by_wd": {str(k): v for k, v in stats.items()},
                                     "selected_wd": min(stats, key=stats.get)}
    for kind in ("G", "S"):
        wd = rec["wd_selection"][kind]["selected_wd"]
        entries = []
        for init in INITS:
            t = trained[f"{kind}_wd{wd}_i{init}"]
            entry = {"init": init, "wd": float(wd), "steps_used": t["best_step"],
                     "dev_mae": t["dev_mae"], "dev_mae_trim5": t["dev_mae_trim5"],
                     "fit_mae_cfit_gated": t["fit_mae"],
                     "curve": t["curve"], "q": {}}
            for qname in data["f_q"]:
                fq = data["f_q"][qname]
                base, yy = data["b_q"][qname], data["y_q"][qname]
                corr = base + predict_head(t["model"], fq)
                m = metrics(yy, base, corr)
                gmask = fq.gate > 0
                if qname in ("Q_bg", "M"):
                    m["identity_max_abs_diff_unsupported"] = float(
                        np.max(np.abs(corr[~gmask] - base[~gmask])) if (~gmask).any() else 0.0)
                    m["n_gated"] = int(gmask.sum())
                    m["base_mae_gated_subset"] = float(np.abs(base[gmask] - yy[gmask]).mean())
                    m["mae_gated_subset"] = float(np.abs(corr[gmask] - yy[gmask]).mean())
                    m["delta_gated_subset"] = float(
                        (np.abs(corr[gmask] - yy[gmask]) - np.abs(base[gmask] - yy[gmask])).mean())
                    m["per_molecule"] = {"row": fq.rows.tolist(), "y": yy.tolist(),
                                         "base": base.tolist(), "corrected": corr.tolist(),
                                         "gated": gmask.astype(int).tolist(),
                                         "support": {k: np.asarray(v).tolist()
                                                     for k, v in fq.support_stats.items()}}
                elif tkeys.get(qname) is not None:
                    cl = key_cluster_ids(fq.rows, tkeys[qname], mol2)
                    m["boot"] = boot(yy, base, corr, clusters=cl)
                    m["per_molecule"] = {"row": fq.rows.tolist(),
                                         "y": yy.tolist(), "base": base.tolist(),
                                         "corrected": corr.tolist(),
                                         "cluster": cl.tolist(),
                                         "support": {k: np.asarray(v).tolist()
                                                     for k, v in fq.support_stats.items()}}
                entry["q"][qname] = m
            entries.append(entry)
        rec["arms"][kind] = entries
    rec["base_metrics"] = {}
    for qname in data["f_q"]:
        base, yy = data["b_q"][qname], data["y_q"][qname]
        fq = data["f_q"][qname]
        gmask = fq.gate > 0
        rec["base_metrics"][qname] = {
            "n": int(len(yy)), "mae": float(np.abs(base - yy).mean()),
            "n_gated": int(gmask.sum()),
            "mae_gated_subset": float(np.abs(base[gmask] - yy[gmask]).mean()) if gmask.any() else None,
            "mean_residual": float((yy - base).mean()),
            "mean_abs_residual": float(np.abs(yy - base).mean()),
            "boot_mae": boot(yy, base, base)["molecule_ci"],
            "per_molecule": {"row": fq.rows.tolist(), "y": yy.tolist(),
                             "base": base.tolist(),
                             "support": {k: np.asarray(v).tolist()
                                         for k, v in fq.support_stats.items()}},
        }
    rec["shuffled_atoms"] = bool(shuffle_atoms)
    return rec, trained


# ---------------------------------------------------------------------------
# per-base-seed feature assembly from the frozen prediction file
# ---------------------------------------------------------------------------


def run_base_seed(seed: int, man: dict, cache, atom_k1, atom_k2, atom_ptr) -> dict:
    roles = {k: np.asarray(v, dtype=np.int64) for k, v in man["roles"].items()}
    pf = Z.PREDS / f"base_F_s{seed}.npz"
    z = np.load(pf)
    rows_all = np.asarray(z["rows"])
    pred_all = np.asarray(z["pred"], dtype=np.float64)
    y_all = np.asarray(z["y"], dtype=np.float64)
    fit_rows = np.asarray(z["fit_rows"])
    sup2 = np.asarray(z["key_support_fit"]).astype(np.float64)
    base_ckpt = str(np.asarray(z["base_ckpt_sha256"]).item())
    idx = {int(r): i for i, r in enumerate(rows_all.tolist())}

    def raw(role_rows):
        rr = np.asarray(role_rows, dtype=np.int64)
        sel = np.asarray([idx[int(r)] for r in rr])
        return Features(cache, rr, pred_all[sel], sup2, atom_k1, atom_k2, atom_ptr), \
            pred_all[sel], y_all[sel]

    fit_raw, b_fit, y_fit = raw(roles["C_fit"])
    dev_raw, b_dev, y_dev = raw(roles["C_dev"])
    q_raw = {k: raw(roles[k]) for k in ("Q_A", "Q_B", "Q_bg")}
    std = Standardiser(fit_raw)
    return {
        "seed": int(seed),
        "base_ckpt_sha256": base_ckpt,
        "fit_rows": fit_rows,
        "support_hash": Z.support_hash(sup2),
        "f_cfit": std.transform(fit_raw), "b_cfit": b_fit, "y_cfit": y_fit,
        "f_cdev": std.transform(dev_raw), "b_cdev": b_dev, "y_cdev": y_dev,
        "f_q": {k: std.transform(v[0]) for k, v in q_raw.items()},
        "b_q": {k: v[1] for k, v in q_raw.items()},
        "y_q": {k: v[2] for k, v in q_raw.items()},
        "standardiser": std.state(),
    }
