"""Zero-GPU coverage projection: how fast does exact radius-2 environment coverage
grow with the number of labelled molecules, and what MAE change does that imply?

Uses only official train/valid structure and *train* labels; no test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import dca_common as C
from dca_part2 import support_tiers

BJOC = C.REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"


def main() -> int:
    train = C.load_cache("train")
    valid = C.load_cache("valid")
    y = np.asarray(valid["labels"], dtype=np.float64)
    pm = pd.read_csv(C.ANALYSIS / "per_molecule_features.csv").sort_values("row_id")
    ae = pm.ae_mean.to_numpy()
    tg = C.molecule_graphs(train)
    vg = C.molecule_graphs(valid)

    split = json.loads((BJOC / "split_a.json").read_text())
    fit4k = json.loads((BJOC / "fit4k_subset.json").read_text())["rows"]
    fit8k = split["fit_rows"]
    monitor = split["monitor_rows"]
    locked = split["locked_rows"]

    fits = {"fit4k": fit4k, "fit8k": fit8k, "train10k": list(range(len(tg)))}
    lg = [tg[i] for i in locked]
    mg = [tg[i] for i in monitor]
    y_locked = np.asarray([float(train["labels"][i]) for i in locked], dtype=np.float64)

    out = {"locked_n": len(locked), "monitor_n": len(monitor),
           "locked_y_mean": float(y_locked.mean()), "locked_y_sd": float(y_locked.std()),
           "locked_n_extreme_y_le_-3": int((y_locked <= -3).sum()),
           "official_valid_y_mean": float(y.mean()), "official_valid_y_sd": float(y.std()),
           "official_valid_n_extreme": int((y <= -3).sum())}

    # atom-level tier shares for each fit size, on the locked-dev and official-valid sets
    per_fit = {}
    for name, rows in fits.items():
        fitset = [tg[i] for i in rows]
        tier_locked = support_tiers(lg, fitset)
        tier_valid = support_tiers(vg, fitset)
        per_fit[name] = {
            "fit_molecules": len(rows),
            "locked_atom_tier_share": {
                "unseen": float(tier_locked["unseen_frac"].mean()),
                "low4": float(tier_locked["low4_frac"].mean()),
                "supported": float(tier_locked["supported_frac"].mean()),
            },
            "locked_molecule_any_unseen_frac": float((tier_locked["unseen_frac"] > 0).mean()),
            "locked_molecule_all_supported_frac": float((tier_locked["supported_frac"] >= 0.999).mean()),
            "valid_atom_tier_share": {
                "unseen": float(tier_valid["unseen_frac"].mean()),
                "low4": float(tier_valid["low4_frac"].mean()),
                "supported": float(tier_valid["supported_frac"].mean()),
            },
            "valid_molecule_any_unseen_frac": float((tier_valid["unseen_frac"] > 0).mean()),
            "valid_molecule_all_supported_frac": float((tier_valid["supported_frac"] >= 0.999).mean()),
        }
        per_fit[name]["_tier_valid"] = tier_valid
        per_fit[name]["_tier_locked"] = tier_locked
    out["coverage"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                       for k, v in per_fit.items()}

    # tier-conditional DCC-L error on official valid (fit = train10k)
    tv = per_fit["train10k"]["_tier_valid"]
    tiers = {
        "all_supported": tv["supported_frac"] >= 0.999,
        "mixed_partial": (tv["supported_frac"] < 0.999) & (tv["unseen_frac"] == 0),
        "any_unseen": tv["unseen_frac"] > 0,
    }
    tier_err = {}
    for k, m in tiers.items():
        tier_err[k] = {"n": int(m.sum()), "mae": float(ae[m].mean()),
                       "ae_share": float(ae[m].sum() / ae.sum())}
    out["valid_tier_error_at_10k"] = tier_err

    # projected MAE change 4k -> 8k if per-tier error stayed fixed (official-valid basis)
    proj = {}
    for arm in ("fit4k", "fit8k"):
        t = per_fit[arm]["_tier_valid"]
        m_all = t["supported_frac"] >= 0.999
        m_mix = (t["supported_frac"] < 0.999) & (t["unseen_frac"] == 0)
        m_un = t["unseen_frac"] > 0
        share = np.asarray([m_all.mean(), m_mix.mean(), m_un.mean()])
        errs = np.asarray([tier_err["all_supported"]["mae"], tier_err["mixed_partial"]["mae"],
                           tier_err["any_unseen"]["mae"]])
        proj[arm] = {"shares": share.tolist(), "fixed_tier_mae": float((share * errs).sum())}
    proj["projected_delta_8k_minus_4k_fixed_tier_error"] = (
        proj["fit8k"]["fixed_tier_mae"] - proj["fit4k"]["fixed_tier_mae"])
    out["coverage_projection"] = proj

    # same on locked-dev (structure only; labels of locked are *not* read here)
    tl = {a: per_fit[a]["_tier_locked"] for a in ("fit4k", "fit8k", "train10k")}
    out["locked_coverage"] = {
        a: {"any_unseen_frac": float((tl[a]["unseen_frac"] > 0).mean()),
            "all_supported_frac": float((tl[a]["supported_frac"] >= 0.999).mean()),
            "atom_unseen_share": float(tl[a]["unseen_frac"].mean())}
        for a in tl
    }
    C.write_json(C.ANALYSIS / "coverage_projection.json", out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
