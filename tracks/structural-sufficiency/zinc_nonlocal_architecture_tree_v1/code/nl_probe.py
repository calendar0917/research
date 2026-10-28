"""Label-free redundancy probe for the new channel (Phase-0 requirement).

Question: is the per-molecule new contribution (SCE bridge term / GEF field
term) already a linear function of the reader's existing pooled features
[mean h, mean h^2, global33, log n]?  A high held-out R^2 means the new channel
is a re-parameterisation of global statistics, not new structure information.
No labels are read.  Can be re-run on a trained checkpoint.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import nl_bridge
import nl_common as N
import nl_data as ND
import nl_models as NM


def collect(head: str, mode: str, lam: float, ckpt: str | None, n: int = 2048,
            device: str = "cpu"):
    norm = N.load_norm()
    cache = N.train_cache()
    bridge = nl_bridge.load("train")
    rows_all = np.arange(0, len(cache["labels"]), max(1, len(cache["labels"]) // n))[:n]
    model, info = NM.build_model(head, norm, 0, device, mode=mode, lam=lam)
    if ckpt:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["state_dict"])
        model.to(device)
    model.eval()
    feats, terms, sizes = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows_all), 128):
            rows = rows_all[start:start + 128]
            batch = ND.to_device(ND.collate(cache, bridge, rows, with_bridge=(head == "sce")), device)
            pred, aux = model(batch, return_aux=True)
            h = aux["h"]
            b = batch["atom_batch"]
            cnt = h.new_zeros(len(rows)).index_add_(0, b, torch.ones_like(b, dtype=h.dtype))
            mh = h.new_zeros((len(rows), h.shape[1])).index_add_(0, b, h) / cnt.unsqueeze(1)
            mh2 = h.new_zeros((len(rows), h.shape[1])).index_add_(0, b, h * h) / cnt.unsqueeze(1)
            g = (batch["global"] - model.trunk.global_mean) / model.trunk.global_scale
            x = torch.cat([mh, mh2, g, torch.log1p(cnt).unsqueeze(1)], dim=1)
            feats.append(x.cpu().numpy())
            terms.append(aux["term"].cpu().numpy())
            sizes.append(cnt.cpu().numpy())
    X = np.concatenate(feats).astype(np.float64)
    t = np.concatenate(terms).astype(np.float64)
    n_atoms = np.concatenate(sizes).astype(np.float64)
    return {"X": X, "term": t, "n_atoms": n_atoms, "info": info}


def ridge_r2(X, y, train_n=1024, alpha=1e-2):
    mu, sd = X[:train_n].mean(0), X[:train_n].std(0) + 1e-8
    Xs = (X - mu) / sd
    Xtr = np.concatenate([Xs[:train_n], np.ones((train_n, 1))], 1)
    Xte = np.concatenate([Xs[train_n:], np.ones((len(X) - train_n, 1))], 1)
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ y[:train_n])
    pred = Xte @ w
    sse = ((pred - y[train_n:]) ** 2).sum()
    sst = ((y[train_n:] - y[train_n:].mean()) ** 2).sum()
    return float(1 - sse / max(sst, 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--head", choices=["sce", "gef"], required=True)
    ap.add_argument("--mode", default="real")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lam = N.LAMBDA_GEF
    d = collect(args.head, args.mode, lam, args.checkpoint)
    X, t = d["X"], d["term"]
    r2 = ridge_r2(X, t)
    rng = np.random.default_rng(0)
    r2_perm = ridge_r2(X, rng.permutation(t))
    out = {
        "head": args.head, "mode": args.mode, "checkpoint": args.checkpoint,
        "labels_read": False, "n_molecules": int(len(t)),
        "r2_term_from_existing_pooled_features": r2,
        "r2_permutation_control": r2_perm,
        "term_std": float(t.std()), "term_mean": float(t.mean()),
        "corr_term_atom_count": float(np.corrcoef(t, d["n_atoms"])[0, 1]),
        "parameters": d["info"]["parameters"],
        "redundant_warning": bool(r2 > 0.90),
    }
    path = args.out or str(N.ANALYSIS / f"probe_{args.head}_{args.mode}.json")
    N.L.write_json(path, out)
    print(out)


if __name__ == "__main__":
    main()
