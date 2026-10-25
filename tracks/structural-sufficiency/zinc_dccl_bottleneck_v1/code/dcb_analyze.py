"""Phase B analysis: Fit4k vs Fit8k locked/monitor/official-valid readout,
split by exact radius-2 environment support relative to the arm's own Fit set.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

import dca_common as C
from dca_part2 import support_tiers
from dcb_train import split_rows

CHAMPION_VALID = 0.075990316172  # historical 10k DCC-L B-R0, official-valid seed mean


def load_run(fit: str, seed: int) -> dict:
    run = C.ROUND / "runs" / f"{fit}_s{seed}"
    info = json.loads((run / "train.json").read_text())
    preds = {}
    for tag in ("best_monitor", "final"):
        d = torch.load(run / f"predictions_{tag}.pt", map_location="cpu", weights_only=False)
        preds[tag] = {k: np.asarray(v) for k, v in d.items() if isinstance(v, np.ndarray)}
    return {"info": info, "preds": preds, "run": run}


def boot_delta(a, b, n_boot=4000, seed=20261025):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = (a - b)[idx].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> int:
    rows = split_rows()
    train = C.load_cache("train")
    valid = C.load_cache("valid")
    tg = C.molecule_graphs(train)
    lock_rows = rows["locked"]
    mon_rows = rows["monitor"]
    y_lock = np.asarray(train["labels"], dtype=np.float64)[lock_rows]
    y_mon = np.asarray(train["labels"], dtype=np.float64)[mon_rows]
    y_val = np.asarray(valid["labels"], dtype=np.float64)

    locked_graphs = [tg[i] for i in lock_rows]
    fit_tiers = {}
    for fit in ("fit4k", "fit8k"):
        fit_tiers[fit] = support_tiers(locked_graphs, [tg[i] for i in rows[fit]])

    out = {"champion_official_valid_10k_seed_mean": CHAMPION_VALID,
           "locked_in_fit4k": int(np.isin(lock_rows, rows["fit4k"]).sum()),
           "monitor_in_fit4k": int(np.isin(mon_rows, rows["fit4k"]).sum()),
           "locked_label_mean": float(y_lock.mean()), "locked_label_sd": float(y_lock.std()),
           "locked_n_tail": int((y_lock <= -3).sum())}

    runs = {}
    for fit in ("fit4k", "fit8k"):
        for seed in (0, 1):
            run_dir = C.ROUND / "runs" / f"{fit}_s{seed}"
            if (run_dir / "train.json").exists():
                runs[(fit, seed)] = load_run(fit, seed)
    out["available_runs"] = [f"{f}_s{s}" for (f, s) in runs]

    per_run = {}
    for (fit, seed), r in runs.items():
        info = r["info"]
        entry = {
            "fit": fit, "seed": seed,
            "parameters": info["parameters"],
            "steps_per_epoch": info["steps_per_epoch"],
            "epochs_run": info["epochs_run"],
            "total_updates": info["total_updates"],
            "best_monitor": info["best_monitor"],
            "tail10_median_monitor_mae": info["tail10_median_monitor_mae"],
            "tail10_median_online_train_mae": info["tail10_median_online_train_mae"],
            "gpu_hours": info["gpu_hours"],
            "norm_source": info["norm"]["source"],
            "reads": {},
        }
        for tag in ("best_monitor", "final"):
            p = r["preds"][tag]["locked_pred"]
            e = p - y_lock
            entry["reads"][tag] = {
                "locked_mae": float(np.abs(e).mean()),
                "locked_mae_top10_removed": float(np.sort(np.abs(e))[::-1][10:].mean()),
                "locked_median_ae": float(np.median(np.abs(e))),
                "locked_p90_ae": float(np.quantile(np.abs(e), 0.9)),
                "locked_signed": float(e.mean()),
                "monitor_mae": float(np.abs(r["preds"][tag]["monitor_pred"] - y_mon).mean()),
                "official_valid_mae": float(
                    np.abs(r["preds"][tag]["official_valid_pred"] - y_val).mean()),
            }
        per_run[f"{fit}_s{seed}"] = entry
    out["per_run"] = per_run

    # ------------------- support-conditional locked readout (per arm, own Fit set)
    tiers = {}
    for fit in ("fit4k", "fit8k"):
        t = fit_tiers[fit]
        tiers[fit] = {
            "all_supported": t["supported_frac"] >= 0.999,
            "mixed_no_unseen": (t["supported_frac"] < 0.999) & (t["unseen_frac"] == 0),
            "any_unseen": t["unseen_frac"] > 0,
            "atom_unseen_share": float(t["unseen_frac"].mean()),
            "atom_low4_share": float(t["low4_frac"].mean()),
            "molecule_any_unseen_frac": float((t["unseen_frac"] > 0).mean()),
        }
    out["locked_tiers"] = {k: {kk: vv for kk, vv in v.items()
                               if not isinstance(vv, np.ndarray)} for k, v in tiers.items()}

    rows_out = []
    for key, entry in per_run.items():
        fit, seed = entry["fit"], entry["seed"]
        e = np.abs(runs[(fit, seed)]["preds"]["best_monitor"]["locked_pred"] - y_lock)
        t = tiers[fit]
        rec = {"run": key, "locked_mae": float(e.mean())}
        for name in ("all_supported", "mixed_no_unseen", "any_unseen"):
            m = t[name]
            rec[name] = {"n": int(m.sum()), "mae": float(e[m].mean()),
                         "ae_share": float(e[m].sum() / e.sum())}
        rows_out.append(rec)
    out["locked_by_support_tier"] = rows_out

    # ------------------- paired Fit8k - Fit4k (same seed, same locked rows)
    contrasts = {}
    for seed in (0, 1):
        if ("fit4k", seed) in runs and ("fit8k", seed) in runs:
            a = np.abs(runs[("fit8k", seed)]["preds"]["best_monitor"]["locked_pred"] - y_lock)
            b = np.abs(runs[("fit4k", seed)]["preds"]["best_monitor"]["locked_pred"] - y_lock)
            af = np.abs(runs[("fit8k", seed)]["preds"]["final"]["locked_pred"] - y_lock)
            bf = np.abs(runs[("fit4k", seed)]["preds"]["final"]["locked_pred"] - y_lock)
            lo, hi = boot_delta(a, b)
            t4 = tiers["fit4k"]
            newly = t4["any_unseen"] & ~tiers["fit8k"]["any_unseen"]
            always = t4["any_unseen"] & tiers["fit8k"]["any_unseen"]
            seen4 = ~t4["any_unseen"]
            contrasts[f"seed{seed}"] = {
                "locked_delta_best_monitor": float(a.mean() - b.mean()),
                "ci95": [lo, hi],
                "locked_delta_final": float(af.mean() - bf.mean()),
                "frac_molecules_improved": float((a < b).mean()),
                "delta_median_ae": float(np.median(a) - np.median(b)),
                "newly_supported": {"n": int(newly.sum()),
                                    "delta": float(a[newly].mean() - b[newly].mean())},
                "still_unseen": {"n": int(always.sum()),
                                 "delta": float(a[always].mean() - b[always].mean())},
                "seen_at_4k": {"n": int(seen4.sum()),
                               "delta": float(a[seen4].mean() - b[seen4].mean())},
                "monitor_delta": float(per_run[f"fit8k_s{seed}"]["reads"]["best_monitor"]["monitor_mae"]
                                       - per_run[f"fit4k_s{seed}"]["reads"]["best_monitor"]["monitor_mae"]),
                "official_valid_delta": float(
                    per_run[f"fit8k_s{seed}"]["reads"]["best_monitor"]["official_valid_mae"]
                    - per_run[f"fit4k_s{seed}"]["reads"]["best_monitor"]["official_valid_mae"]),
            }
    out["contrast_fit8k_minus_fit4k"] = contrasts

    # ------------------- coverage projection vs measured
    proj = json.loads((C.ANALYSIS / "coverage_projection.json").read_text())
    out["coverage_projection"] = proj["coverage_projection"]
    out["coverage_locked"] = proj["locked_coverage"]
    C.write_json(C.ANALYSIS / "phase_b.json", out)
    print(json.dumps({k: out[k] for k in
                      ("per_run", "locked_by_support_tier", "contrast_fit8k_minus_fit4k")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
