"""Post-hoc relation-content probe for LCRC-v1 (no training, no test access).

The trained R/N relation head collapsed to an almost constant correction in the
formal runs.  This script asks whether that is because the relation content
carries no residual signal, or because the end-to-end head failed to use it:

1. extract the shared relation readout features on train/valid;
2. fit a fresh ridge linear head on 8,000 train molecules to the residual
   ``y - (contrib + rho)`` and evaluate on 2,000 held-out train molecules and
   on official valid;
3. permute the relation features as a control;
4. report the between-molecule std of the category content means and the
   within-molecule pair-content std (and its 1/sqrt(n) standard error).

This is a diagnostic on frozen features, not a new deployable model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import lc_common as L
import lc_data as D
import lc_models as M


def _model(run_dir: Path):
    info = L.read_json(run_dir / "train.json")
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    model = M.load_model_from_state(info["arm"], D.load_norm(), state, "cpu")
    return info, model


def _extract(model, cache, rows, *, device="cpu", batch_size=256):
    model.eval()
    base, pred, delta, hidden, content, count, exists = [], [], [], [], [], [], []
    zs, pgs, pcs = [], [], []
    with torch.no_grad(), L.C.isolated_rng():
        for start in range(0, len(rows), batch_size):
            chunk = np.asarray(rows[start:start + batch_size], dtype=np.int64)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in D.collate(cache, chunk).items()}
            y, aux = model(batch, return_aux=True)
            base.append((aux["contrib"] + aux["rho"]).cpu().numpy())
            pred.append(y.cpu().numpy())
            delta.append(aux["delta"].cpu().numpy())
            M = aux["M"]
            n = aux["n"]
            ex = aux["exists"]
            cat_ids = torch.arange(model.rel.n_cat, device=M.device)
            cat_feat = model.rel.cat_emb(cat_ids).unsqueeze(0).expand(M.shape[0], -1, -1)
            feat = torch.cat([M, torch.log1p(n).unsqueeze(-1), ex.unsqueeze(-1), cat_feat], dim=-1)
            hidden.append(model.rel.mlp(feat).cpu().numpy())
            content.append(M.cpu().numpy())
            count.append(n.cpu().numpy())
            exists.append(ex.cpu().numpy())
            zs.append(aux["z"].cpu().numpy())
            pgs.append(batch["pair_graph"].cpu().numpy())
            pcs.append(batch["pair_cat"].cpu().numpy())
    return {
        "base": np.concatenate(base, 0),
        "pred": np.concatenate(pred, 0),
        "delta": np.concatenate(delta, 0),
        "hidden": np.concatenate(hidden, 0),
        "M": np.concatenate(content, 0),
        "n": np.concatenate(count, 0),
        "exists": np.concatenate(exists, 0),
        "z": np.concatenate(zs, 0),
        "pair_graph": np.concatenate(pgs, 0),
        "pair_cat": np.concatenate(pcs, 0),
    }


def _ridge_fit(X, y, alpha=1.0):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    b = y.mean()
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - b))
    return {"mu": mu, "sd": sd, "b": float(b), "W": W}


def _ridge_pred(model, X):
    Z = (X - model["mu"]) / model["sd"]
    return Z @ model["W"] + model["b"]


def _mae(p, y):
    return float(np.mean(np.abs(np.asarray(p, np.float64) - np.asarray(y, np.float64))))


def _variance_diag(extract):
    M = extract["M"]  # [N,6,32]
    z = extract["z"]
    pg = extract["pair_graph"]
    pc = extract["pair_cat"]
    out = {}
    between = M.std(axis=0).mean(axis=1)  # per category
    within = np.zeros(M.shape[1])
    within_sem = np.zeros(M.shape[1])
    for c in range(M.shape[1]):
        vals = []
        sems = []
        for g in range(len(M)):
            sel = (pg == g) & (pc == c)
            if sel.sum() >= 3:
                zz = z[sel]
                vals.append(float(zz.std(axis=0).mean()))
                sems.append(float(zz.std(axis=0).mean() / np.sqrt(sel.sum())))
        within[c] = float(np.mean(vals)) if vals else 0.0
        within_sem[c] = float(np.mean(sems)) if sems else 0.0
    out["between_molecule_std_M"] = between.tolist()
    out["within_molecule_pair_std_z"] = within.tolist()
    out["within_molecule_sem_of_mean"] = within_sem.tolist()
    out["between_over_sem"] = (between / np.maximum(within_sem, 1e-12)).tolist()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args(argv)
    train = D.load_cache("train")
    valid = D.load_cache("valid")
    y_train = np.asarray(train["labels"], np.float64)
    y_valid = np.asarray(valid["labels"], np.float64)
    n_fit = 8000
    rows_train = np.arange(len(y_train))
    rows_valid = np.arange(len(y_valid))
    results = []
    for run in args.runs:
        run_dir = L.RUNS / run
        info, model = _model(run_dir)
        if info["arm"] not in ("R", "N"):
            continue
        ext_tr = _extract(model, train, rows_train)
        ext_va = _extract(model, valid, rows_valid)
        X_tr = ext_tr["hidden"].reshape(len(y_train), -1)
        X_va = ext_va["hidden"].reshape(len(y_valid), -1)
        Xr_tr = np.concatenate([ext_tr["M"].reshape(len(y_train), -1),
                                np.log1p(ext_tr["n"]), ext_tr["exists"]], axis=1)
        Xr_va = np.concatenate([ext_va["M"].reshape(len(y_valid), -1),
                                np.log1p(ext_va["n"]), ext_va["exists"]], axis=1)
        resid_tr = y_train - ext_tr["base"]
        fit_idx = np.arange(n_fit)
        hold_idx = np.arange(n_fit, len(y_train))
        fit = _ridge_fit(X_tr[fit_idx], resid_tr[fit_idx])
        corr_hold = _ridge_pred(fit, X_tr[hold_idx])
        corr_valid = _ridge_pred(fit, X_va)
        base_hold = ext_tr["base"][hold_idx]
        base_valid = ext_va["base"]
        # permutation control: shuffle the fit rows
        rng = np.random.default_rng(20261108)
        perm = rng.permutation(n_fit)
        fit_perm = _ridge_fit(X_tr[fit_idx][perm], resid_tr[fit_idx])
        corr_perm = _ridge_pred(fit_perm, X_va)
        fit_raw = _ridge_fit(Xr_tr[fit_idx], resid_tr[fit_idx])
        corr_raw_hold = _ridge_pred(fit_raw, Xr_tr[hold_idx])
        corr_raw_valid = _ridge_pred(fit_raw, Xr_va)
        fit_raw_perm = _ridge_fit(Xr_tr[fit_idx][perm], resid_tr[fit_idx])
        corr_raw_perm = _ridge_pred(fit_raw_perm, Xr_va)
        out = {
            "run": run,
            "arm": info["arm"],
            "seed": int(info["seed"]),
            "best_valid": _mae(ext_va["pred"], y_valid),
            "base_valid": _mae(base_valid, y_valid),
            "model_delta_valid": _mae(base_valid + ext_va["delta"], y_valid),
            "ridge_valid": _mae(base_valid + corr_valid, y_valid),
            "ridge_holdout_train": _mae(base_hold + corr_hold, y_train[hold_idx]),
            "base_holdout_train": _mae(base_hold, y_train[hold_idx]),
            "ridge_permutation_valid": _mae(base_valid + corr_perm, y_valid),
            "ridge_improvement_valid": _mae(base_valid, y_valid) - _mae(base_valid + corr_valid, y_valid),
            "ridge_raw_valid": _mae(base_valid + corr_raw_valid, y_valid),
            "ridge_raw_holdout_train": _mae(base_hold + corr_raw_hold, y_train[hold_idx]),
            "ridge_raw_permutation_valid": _mae(base_valid + corr_raw_perm, y_valid),
            "ridge_raw_improvement_valid": _mae(base_valid, y_valid) - _mae(base_valid + corr_raw_valid, y_valid),
            "ridge_raw_head_norm": float(np.linalg.norm(fit_raw["W"])),
            "delta_between_min_max": float(ext_va["delta"].max() - ext_va["delta"].min()),
            "delta_mean_abs": float(np.mean(np.abs(ext_va["delta"]))),
            "ridge_head_norm": float(np.linalg.norm(fit["W"])),
            "variance": _variance_diag(ext_va),
        }
        results.append(out)
        L.log(f"{run}: base {out['base_valid']:.6f} model {out['best_valid']:.6f} "
              f"ridge {out['ridge_valid']:.6f} raw {out['ridge_raw_valid']:.6f} "
              f"perm {out['ridge_permutation_valid']:.6f}")
    L.write_json(L.ANALYSIS / "relation_probe.json", {"rows": results, "created": L.now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
