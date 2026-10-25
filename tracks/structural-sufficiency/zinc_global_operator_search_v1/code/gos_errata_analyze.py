"""Erratum addendum analysis: L2G / GN against the historical ho_A_s0 run.

Reads only train/valid-derived artifacts that already exist locally (curves and
the train-internal holdout predictions).  Official test is not touched.

Outputs ``analysis/errata_results.json`` (tracked evidence) and prints the tables
used in ``ERRATUM.md``.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import gos_runner as GR  # noqa: E402

RUNS = _EXP / "runs"
ANALYSIS = _EXP / "analysis"
ARMS = {"A": "ho_A_s0", "L2G": "l2g_A_s0", "GN": "gn_A_s0"}
WINDOWS = (100, 200, 300, 400, 500, 600)
#: the literal `w/sum(w)` GN run that stopped at epoch 284 with a non-finite
#: gradient; its wall time is part of the round's budget.
FAILED_GN_GPU_HOURS = (datetime(2026, 10, 25, 8, 18, 0) -
                       datetime(2026, 10, 25, 7, 53, 39)).total_seconds() / 3600.0


def curve(name: str) -> dict[int, dict]:
    out = {}
    with open(RUNS / name / "curve.csv") as fh:
        for r in csv.DictReader(fh):
            out[int(r["epoch"])] = {"valid": float(r["valid_mae"]),
                                    "train": float(r["online_train_mae"]),
                                    "seconds": float(r["seconds"])}
    return out


def run_summary(name: str) -> dict:
    info = C.read_json(RUNS / name / "train.json")
    art = C.read_json(RUNS / name / "artifacts.json")
    cv = curve(name)
    eps = sorted(cv)
    best_ep = min(eps, key=lambda e: cv[e]["valid"])
    return {
        "run": name,
        "mix_mode": art.get("mix_mode") or "legacy(pre-erratum code)",
        "parameters": int(info["parameter_count"]),
        "epochs": len(eps),
        "best_valid": float(cv[best_ep]["valid"]),
        "best_epoch": int(best_ep),
        "final_valid": float(cv[eps[-1]]["valid"]),
        "w20_valid": float(np.mean([cv[e]["valid"] for e in eps[-20:]])),
        "train_mae_eval_mode": float(info["fixed_objects"]["best"]["train_mae_eval_mode"]),
        "train_online_at_best": float(cv[best_ep]["train"]),
        "wall_seconds": float(info["wall_seconds"]),
        "gpu_hours": float(info.get("gpu_hours") or info["wall_seconds"] / 3600.0),
        "init_sha256": info["model_info"]["initial_state_sha256"],
        "best_pt": f"{art['run_dir']}/best.pt",
        "best_pt_sha256": art["sha256"]["best_pt"],
    }


def holdout_pred(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RUNS / name / "predictions" / "holdout.npz")
    return np.asarray(d["row"]).astype(np.int64), np.asarray(d["pred"]).astype(np.float64)


def boot_ci(diff: np.ndarray, n_boot: int = 20000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    _, hold = GR.holdout_split(1000)
    y = np.asarray(D.load_cache("train")["labels"], dtype=np.float64)[hold]

    res: dict = {"arms": {}, "matched_epochs": {}, "holdout": {}, "groups": {},
                 "budget": {}, "windows": list(WINDOWS)}
    curves = {}
    for label, name in ARMS.items():
        res["arms"][label] = run_summary(name)
        curves[label] = curve(name)

    # matched-epoch table -------------------------------------------------
    for ep in WINDOWS:
        row = {label: (float(curves[label][ep]["valid"]) if ep in curves[label] else None)
               for label in ARMS}
        for label in ARMS:
            if label != "A":
                row[f"delta_{label}_minus_A"] = (
                    None if row[label] is None or row["A"] is None else row[label] - row["A"])
        res["matched_epochs"][str(ep)] = row

    # sustained windows: 10-epoch mean valid MAE ending at each window ----
    res["sustained_windows"] = {}
    for ep in WINDOWS:
        row = {}
        for label in ARMS:
            vals = [curves[label][e]["valid"] for e in range(ep - 9, ep + 1)
                    if e in curves[label]]
            row[label] = float(np.mean(vals)) if len(vals) == 10 else None
        for label in ARMS:
            if label != "A" and row[label] is not None and row["A"] is not None:
                row[f"delta_{label}_minus_A"] = row[label] - row["A"]
        res["sustained_windows"][str(ep)] = row

    # holdout (evaluated once, after checkpoint selection) ----------------
    ae = {}
    for label, name in ARMS.items():
        rows, pred = holdout_pred(name)
        if not np.array_equal(rows, hold):
            raise RuntimeError(f"{name}: holdout rows differ from the registered split")
        ae[label] = np.abs(pred - y)
    base = ae["A"]
    order = np.argsort(base, kind="stable")
    tert = np.empty(len(base), dtype=np.int64)
    tert[order[: len(base) // 3]] = 0
    tert[order[len(base) // 3: 2 * len(base) // 3]] = 1
    tert[order[2 * len(base) // 3:]] = 2
    top10 = np.zeros(len(base), dtype=bool)
    top10[order[int(0.9 * len(base)):]] = True

    for label in ARMS:
        entry = {
            "holdout_mae": float(ae[label].mean()),
            "median_ae": float(np.median(ae[label])),
            "p90_ae": float(np.percentile(ae[label], 90)),
            "n": int(len(base)),
        }
        if label != "A":
            diff = ae[label] - base
            entry.update({
                "delta_vs_A": float(diff.mean()),
                "delta_ci95": boot_ci(diff),
                "win_rate_vs_A": float((diff < 0).mean()),
                "delta_drop_top10_by_A": float(diff[~top10].mean()),
                "delta_tertiles": {
                    f"q{i}": {"n": int((tert == i).sum()),
                              "A": float(base[tert == i].mean()),
                              "arm": float(ae[label][tert == i].mean()),
                              "delta": float(diff[tert == i].mean())}
                    for i in range(3)
                },
                "delta_top10pct_by_A": float(diff[top10].mean()),
                "n_top10pct": int(top10.sum()),
            })
        res["holdout"][label] = entry

    res["groups"] = {
        "definition": "tertiles of the frozen ho_A_s0 per-molecule AE on the internal holdout",
        "thresholds": [float(base[order[len(base) // 3]]),
                       float(base[order[2 * len(base) // 3]])],
        "top10pct_threshold": float(base[order[int(0.9 * len(base))]]),
        "n_top10pct": int(top10.sum()),
        "median_ae_A": float(np.median(base)),
    }

    # budget -------------------------------------------------------------
    res["budget"] = {
        "gpu_hours_arms": {label: res["arms"][label]["gpu_hours"] for label in ARMS},
        "gpu_hours_new_arms": sum(res["arms"][l]["gpu_hours"] for l in ARMS if l != "A"),
        "gpu_hours_failed_gn_literal": FAILED_GN_GPU_HOURS,
        "gpu_hours_total_spent": (sum(res["arms"][l]["gpu_hours"] for l in ARMS
                                       if l != "A") + FAILED_GN_GPU_HOURS),
        "cap_gpu_hours": 2.0,
        "note": "process backend, one A100 per run; wall seconds recorded by the trainer",
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "errata_results.json", res)

    # console tables -----------------------------------------------------
    print("\n== runs ==")
    for label in ARMS:
        a = res["arms"][label]
        print(f"{label:4s} {a['run']:12s} {a['mix_mode']:7s} params={a['parameters']} "
              f"best={a['best_valid']:.6f}@{a['best_epoch']} W20={a['w20_valid']:.6f} "
              f"final={a['final_valid']:.6f} train@best={a['train_mae_eval_mode']:.6f} "
              f"gpu_h={a['gpu_hours']:.3f}")
    print("\n== matched epochs (valid MAE) ==")
    labels = list(ARMS)
    print(f"{'ep':>4s} " + " ".join(f"{l:>9s}" for l in labels)
          + " " + " ".join(f"{'d' + l:>9s}" for l in labels if l != "A"))
    for ep in WINDOWS:
        r = res["matched_epochs"][str(ep)]
        def f(v):
            return "       --" if v is None else f"{v:9.6f}"
        print(f"{ep:4d} " + " ".join(f(r[l]) for l in labels) + " "
              + " ".join(f(r[f'delta_{l}_minus_A']) for l in labels if l != "A"))
    print("\n== sustained 10-epoch mean valid MAE ending at the window ==")
    print(f"{'ep':>4s} " + " ".join(f"{l:>9s}" for l in labels)
          + " " + " ".join(f"{'d' + l:>9s}" for l in labels if l != "A"))
    for ep in WINDOWS:
        r = res["sustained_windows"][str(ep)]
        def f(v):
            return "       --" if v is None else f"{v:9.6f}"
        print(f"{ep:4d} " + " ".join(f(r[l]) for l in labels) + " "
              + " ".join(f(r[f'delta_{l}_minus_A']) for l in labels if l != "A"))

    print("\n== internal holdout (1,000 rows, one pass after selection) ==")
    for label in ARMS:
        h = res["holdout"][label]
        extra = ""
        if label != "A":
            extra = (f" delta={h['delta_vs_A']:+.6f} CI[{h['delta_ci95'][0]:+.6f},"
                     f"{h['delta_ci95'][1]:+.6f}] win={h['win_rate_vs_A']:.3f}")
        print(f"{label:4s} mae={h['holdout_mae']:.6f} median={h['median_ae']:.6f}{extra}")
    print("\n== fixed baseline (A) AE groups ==")
    for label in [l for l in ARMS if l != "A"]:
        h = res["holdout"][label]
        t = h["delta_tertiles"]
        print(f"{label:4s} q0={t['q0']['delta']:+.6f} q1={t['q1']['delta']:+.6f} "
              f"q2={t['q2']['delta']:+.6f} top10%={h['delta_top10pct_by_A']:+.6f} "
              f"drop-top10={h['delta_drop_top10_by_A']:+.6f}")
    print(f"\ngpu_h arms = {res['budget']['gpu_hours_new_arms']:.3f}  +failed gn "
          f"{FAILED_GN_GPU_HOURS:.3f} = {res['budget']['gpu_hours_total_spent']:.3f} / 2.0 cap")
    print("wrote", ANALYSIS / "errata_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
