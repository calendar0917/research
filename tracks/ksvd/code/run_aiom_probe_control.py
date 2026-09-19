#!/usr/bin/env python
"""Probe-adequacy control for the AIOM representation audit.

The capacity probe (Test F) is only informative about the *lift* if the probe
pipeline itself is capable of reaching low MAE on a representation that is known
to carry more information.  This control runs the *same* fixed probe
(linear and ``Linear(d,64)->SiLU->Linear(64,1)``, identical protocol/seed) on:

* the typed 3-round WL subtree histogram (a strong ordering-invariant
  structural descriptor),
* the AIOM T=8 vector,
* the concatenation of both.

No dictionary, no new architecture candidate; this is a diagnostic only.
Official ZINC test is never loaded; ``y`` is used only to score the probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_aiom_representation_audit import (  # noqa: E402
    RESULTS_DIR,
    apply_standardizer,
    fit_standardizer,
    load_mols,
    repo_loader,
    train_probe,
    wl_fingerprint,
    T_MAIN,
)


def wl_matrix(mols_all, rounds=3):
    hists = [wl_fingerprint(m, rounds=rounds) for m in mols_all]
    keys = sorted({k for h in hists for k in h}, key=repr)
    index = {k: i for i, k in enumerate(keys)}
    W = np.zeros((len(hists), len(keys)), dtype=np.float64)
    for r, h in enumerate(hists):
        for k, v in h.items():
            W[r, index[k]] = v
    return W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    train_mols = load_mols(args.data_root, "train")
    valid_mols = load_mols(args.data_root, "valid")
    load_zinc, _ = repo_loader()
    train_ds = load_zinc(args.data_root, "train")
    valid_ds = load_zinc(args.data_root, "val")
    y_train = np.array([float(train_ds[i].y.reshape(-1)[0]) for i in range(len(train_mols))])
    y_valid = np.array([float(valid_ds[i].y.reshape(-1)[0]) for i in range(len(valid_mols))])

    d = np.load(args.out / "aiom_features_T8.npz")
    phi_train = d["phi_train"]
    phi_valid = d["phi_valid"]
    dim = phi_train.shape[1] // (T_MAIN[-1] + 1)
    aiom_train = phi_train[:, : (T_MAIN[-1] + 1) * dim]
    aiom_valid = phi_valid[:, : (T_MAIN[-1] + 1) * dim]

    print("building WL histograms ...", flush=True)
    W_all = wl_matrix(train_mols + valid_mols, rounds=3)
    wl_train = W_all[: len(train_mols)]
    wl_valid = W_all[len(train_mols):]
    print(f"WL dim={wl_train.shape[1]}", flush=True)

    feature_sets = {
        "wl_subtree": (wl_train, wl_valid),
        "aiom_T8": (aiom_train, aiom_valid),
        "aiom_T8_plus_wl": (np.concatenate([aiom_train, wl_train], axis=1),
                            np.concatenate([aiom_valid, wl_valid], axis=1)),
    }
    rows = []
    for name, (Ftr, Fva) in feature_sets.items():
        mu, sd = fit_standardizer(Ftr)
        Ztr = apply_standardizer(Ftr, mu, sd)
        Zva = apply_standardizer(Fva, mu, sd)
        lin = train_probe(Ztr, y_train, Zva, y_valid, hidden=None, seed=0)
        mlp = train_probe(Ztr, y_train, Zva, y_valid, hidden=64, seed=0)
        for probe, res in (("linear", lin), ("mlp", mlp)):
            row = {"features": name, "probe": probe, "dim": int(Ftr.shape[1]), **res}
            rows.append(row)
            print(f"[control] {name:18s} {probe:6s} dim={Ftr.shape[1]:5d} "
                  f"train={res['train_mae']:.4f} valid={res['valid_mae']:.4f}", flush=True)
    (args.out / "probe_control_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
