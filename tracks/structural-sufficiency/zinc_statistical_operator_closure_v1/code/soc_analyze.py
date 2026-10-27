"""SOC-v1 screen analysis: paired Monitor comparison against frozen DCC-L.

Reads only train-cache labels (Monitor rows), the frozen L checkpoints'
predictions, and the SOC screen predictions.  Official test never loaded.

Usage
-----
uv run python .../soc_analyze.py --runs <dir-with-screen_fit8k_R_s0/s1> \
    --out analysis/screen_R.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_data as D  # noqa: E402
import soc_common as SC  # noqa: E402

DCB = _SUFF / "zinc_dccl_bottleneck_v1"
PER_MOL = DCB / "analysis" / "per_molecule.csv"
BOOT = 10000
SEED = 20261003


def load_pred(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {k: (np.asarray(v) if isinstance(v, (list, tuple)) else v) for k, v in d.items()}


def bootstrap_delta(ae_a: np.ndarray, ae_b: np.ndarray, *, reps: int = BOOT,
                    seed: int = SEED) -> dict:
    """Paired bootstrap of mean(ae_a - ae_b); negative = a better."""
    d = np.asarray(ae_a, dtype=np.float64) - np.asarray(ae_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(reps, n))
    means = d[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "delta_mae": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "p_delta_lt_0": float((means < 0).mean()),
        "win_fraction": float((d < 0).mean()),
        "n": int(n),
        "reps": int(reps),
        "seed": int(seed),
    }


def group_table(rows: np.ndarray, soc_ae: np.ndarray, l_ae: np.ndarray,
                groups: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, g in groups.items():
        g = np.asarray(g)
        vals = []
        for label in np.unique(g):
            m = g == label
            if int(m.sum()) == 0:
                continue
            vals.append({
                "group": str(label),
                "n": int(m.sum()),
                "soc_mae": float(soc_ae[m].mean()),
                "l_mae": float(l_ae[m].mean()),
                "delta": float((soc_ae[m] - l_ae[m]).mean()),
                "win_fraction": float((soc_ae[m] < l_ae[m]).mean()),
            })
        out[name] = vals
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default=str(SC.ANALYSIS / "screen_R.json"))
    p.add_argument("--arm", default="R")
    args = p.parse_args(argv)
    runs = Path(args.runs)

    cache = D.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    rows = SC.split_rows()["monitor"]
    yr = y[rows]

    soc_pred, soc_rows = {}, None
    for s in (0, 1):
        d = load_pred(runs / f"screen_fit8k_{args.arm}_s{s}" / "predictions_best.pt")
        soc_pred[s] = np.asarray(d["monitor_pred"], dtype=np.float64)
        r = np.asarray(d["monitor_rows"])
        if soc_rows is None:
            soc_rows = r
        elif not np.array_equal(r, soc_rows):
            raise RuntimeError("monitor row order differs between SOC seeds")
    if not np.array_equal(soc_rows, rows):
        raise RuntimeError("SOC monitor rows do not equal the frozen split")

    l_pred = {}
    for s in (0, 1):
        d = load_pred(DCB / "runs" / f"fit8k_s{s}" / "predictions_best_monitor.pt")
        l_pred[s] = np.asarray(d["monitor_pred"], dtype=np.float64)
        if not np.array_equal(np.asarray(d["monitor_rows"]), rows):
            raise RuntimeError("frozen L monitor rows do not equal the frozen split")

    soc_ae = {s: np.abs(soc_pred[s] - yr) for s in (0, 1)}
    l_ae = {s: np.abs(l_pred[s] - yr) for s in (0, 1)}
    soc_mean = np.mean([soc_ae[s] for s in (0, 1)], axis=0)
    l_mean = np.mean([l_ae[s] for s in (0, 1)], axis=0)

    per_seed = {}
    for s in (0, 1):
        per_seed[str(s)] = {
            "soc_mae": float(soc_ae[s].mean()),
            "l_mae": float(l_ae[s].mean()),
            "delta_soc_minus_l": float(soc_ae[s].mean() - l_ae[s].mean()),
            "bootstrap": bootstrap_delta(soc_ae[s], l_ae[s]),
        }
    seed_mean = {
        "soc_mae": float(soc_mean.mean()),
        "l_mae": float(l_mean.mean()),
        "delta_soc_minus_l": float(soc_mean.mean() - l_mean.mean()),
        "bootstrap": bootstrap_delta(soc_mean, l_mean),
    }

    import pandas as pd
    # Monitor structural features computed from the train cache itself.  The
    # DCB per_molecule.csv is the official-valid split and must not be used here.
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    n_atoms = ap[rows + 1] - ap[rows]
    n_bonds = bp[rows + 1] - bp[rows]
    n_rings = n_bonds - n_atoms + 1
    groups = {
        "n_atoms_quartile": pd.qcut(n_atoms, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str),
        "n_bonds_quartile": pd.qcut(n_bonds, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str),
        "n_rings_bin": pd.cut(n_rings, bins=[-0.5, 0.5, 1.5, 2.5, 100],
                              labels=["0", "1", "2", "3+"]).astype(str),
        "y_quartile": pd.qcut(yr, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str),
        "y_tail_le_-3": np.where(yr <= -3.0, "tail", "ordinary"),
    }
    tables = group_table(rows, soc_mean, l_mean, groups)

    # largest per-row contributions to the seed-mean delta
    diff = soc_mean - l_mean
    order = np.argsort(diff)
    top_neg = [{"row_id": int(i), "delta": float(diff[i]), "y": float(yr[i]),
                "soc_ae": float(soc_mean[i]), "l_ae": float(l_mean[i])}
               for i in order[:10]]
    top_pos = [{"row_id": int(i), "delta": float(diff[i]), "y": float(yr[i]),
                "soc_ae": float(soc_mean[i]), "l_ae": float(l_mean[i])}
               for i in order[::-1][:10]]

    cfg = SC.load_config()
    gate = cfg["screen_gate"]
    soc_mean_mae = seed_mean["soc_mae"]
    l_anchor = float(np.mean(list(SC.FIT8K_MONITOR_ANCHOR.values())))
    paired_ok = (per_seed["0"]["soc_mae"] < gate["paired_max_monitor"]["seed0"]
                 and per_seed["1"]["soc_mae"] < gate["paired_max_monitor"]["seed1"])
    passed = bool(soc_mean_mae <= gate["primary_seed_mean_monitor_max"] and paired_ok)
    verdict = "SCREEN_PASS" if passed else "SCREEN_FAIL"

    result = {
        "round": SC.ROUND_ID,
        "arm": args.arm,
        "split": "BJOC fit8k train / Monitor 1k",
        "n_monitor": int(len(rows)),
        "rows_sha256": SC.sha256_rows(rows),
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "anchors": {"l_seed0": SC.FIT8K_MONITOR_ANCHOR[0],
                    "l_seed1": SC.FIT8K_MONITOR_ANCHOR[1], "l_mean": l_anchor},
        "per_seed": per_seed,
        "seed_mean": seed_mean,
        "gate": {"seed_mean_max": gate["primary_seed_mean_monitor_max"],
                 "paired_max": gate["paired_max_monitor"],
                 "paired_ok": bool(paired_ok),
                 "passed": passed},
        "verdict": verdict,
        "groups": tables,
        "top_negative_delta": top_neg,
        "top_positive_delta": top_pos,
    }
    SC.write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("arm", "per_seed", "seed_mean", "gate", "verdict")},
                     indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
