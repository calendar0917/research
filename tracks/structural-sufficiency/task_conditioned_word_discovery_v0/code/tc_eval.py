"""Final evaluation for the executed neural arms.

Only reads committed/pulled manifests and per-molecule predictions.  The
bootstrap resamples whole Bemis-Murcko scaffold clusters and uses the same draw
indices for both arms of a pair.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

import tc_common as C

REF = {"D0": 0.547679, "D1": 0.563802}


def load_run(run_id: str) -> dict[str, Any]:
    path = C.RESULTS_DIR / run_id
    manifest = C.read_json(path / "manifest.json")
    with np.load(path / "predictions_best_dev.npz", allow_pickle=False) as z:
        rows = np.asarray(z["rows"], np.int64)
        y = np.asarray(z["y_true"], np.float64)
        p = np.asarray(z["y_pred"], np.float64)
    return {"run_id": run_id, "manifest": manifest, "rows": rows, "y": y, "pred": p,
            "path": path}


def paired_bootstrap(a: Mapping[str, Any], b: Mapping[str, Any], scaffold_by_row: Mapping[str, str],
                     draws: int, seed: int) -> dict[str, Any]:
    if not np.array_equal(a["rows"], b["rows"]):
        raise ValueError("prediction rows are not aligned")
    rows = np.asarray(a["rows"], np.int64)
    y = np.asarray(a["y"], np.float64)
    pa = np.asarray(a["pred"], np.float64)
    pb = np.asarray(b["pred"], np.float64)
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows.tolist()):
        groups.setdefault(scaffold_by_row[str(int(row))], []).append(i)
    keys = sorted(groups)
    sizes = [len(groups[k]) for k in keys]
    rng = np.random.default_rng(int(seed))
    out = np.zeros((int(draws), 4), np.float64)
    for d in range(int(draws)):
        take = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([np.asarray(groups[keys[int(t)]], np.int64) for t in take])
        yy, aa, bb = y[idx], pa[idx], pb[idx]
        rmse_a = float(np.sqrt(np.mean((aa - yy) ** 2)))
        rmse_b = float(np.sqrt(np.mean((bb - yy) ** 2)))
        mae_a = float(np.mean(np.abs(aa - yy)))
        mae_b = float(np.mean(np.abs(bb - yy)))
        out[d] = [rmse_a, rmse_b, mae_a, mae_b]
    def ci(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    gain_rmse = out[:, 1] - out[:, 0]   # positive = a better
    gain_mae = out[:, 3] - out[:, 2]
    point_a = float(np.sqrt(np.mean((pa - y) ** 2)))
    point_b = float(np.sqrt(np.mean((pb - y) ** 2)))
    return {
        "arm_a": a["run_id"], "arm_b": b["run_id"],
        "n_molecules": int(rows.size), "n_scaffolds": int(len(keys)),
        "bootstrap_draws": int(draws), "seed": int(seed),
        "rmse_a": point_a, "rmse_b": point_b,
        "mae_a": float(np.mean(np.abs(pa - y))),
        "mae_b": float(np.mean(np.abs(pb - y))),
        "gain_rmse_a_minus_b": float(point_b - point_a),
        "gain_rmse_ci95": ci(gain_rmse),
        "gain_mae_a_minus_b": float(np.mean(np.abs(pb - y)) - np.mean(np.abs(pa - y))),
        "gain_mae_ci95": ci(gain_mae),
        "a_better_fraction": float(np.mean(np.abs(pa - y) < np.abs(pb - y))),
        "draws": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="D0")
    ap.add_argument("--reference", type=float, default=REF["D0"])
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1703)
    args = ap.parse_args()
    runs = {r: load_run(r) for r in args.runs}
    arms = {v["manifest"]["arm"]: v for v in runs.values()}
    table = []
    for run_id, r in runs.items():
        m = r["manifest"]
        table.append({
            "run_id": run_id, "arm": m["arm"], "split": m["split"], "seed": m["seed"],
            "fit_rmse": m["best"]["fit_rmse"], "dev_rmse": m["best"]["dev_rmse"],
            "dev_mae": m["best"]["dev_mae"], "best_epoch": m["best"]["epoch"],
            "epochs_run": m["epochs_run"], "params": m["params"]["total"],
            "wall_seconds": m["wall_seconds"], "gpu_hours": m["gpu_hours"],
            "commit": m["git_commit"], "K": m["K"],
            "gap_to_ref": float(m["best"]["dev_rmse"] - args.reference),
        })
    C.write_csv(C.ANALYSIS_DIR / "main_table.csv", table)

    scaffold = C.scaffold_map(args.split)
    boot = {}
    for a, b in (("D_TASK", "D_FREQ"), ("BASE", "D_FREQ"), ("D_TASK", "BASE")):
        if a in arms and b in arms:
            boot[f"{a}_vs_{b}"] = paired_bootstrap(arms[a], arms[b], scaffold,
                                                   args.draws, args.seed)
    np.savez_compressed(C.ANALYSIS_DIR / "bootstrap_d0_s0.npz",
                        **{f"{k}_{name}": v for k, b in boot.items()
                           for name, v in (("draws", b["draws"]),)})
    boot_summary = {k: {kk: vv for kk, vv in v.items() if kk != "draws"}
                    for k, v in boot.items()}
    C.write_json(C.ANALYSIS_DIR / "bootstrap_d0_s0.json", boot_summary)

    # frozen progression gates
    base = arms.get("BASE", {}).get("manifest", {}).get("best", {}).get("dev_rmse")
    task = arms.get("D_TASK", {}).get("manifest", {}).get("best", {}).get("dev_rmse")
    freq = arms.get("D_FREQ", {}).get("manifest", {}).get("best", {}).get("dev_rmse")
    gate = {}
    if task is not None and freq is not None:
        gate["task_gain_vs_freq"] = float(freq - task)
        gate["condition_a_task_beats_freq_0.015"] = bool(freq - task >= 0.015)
    if task is not None and base is not None:
        gate["task_gain_vs_base"] = float(base - task)
        gate["task_within_0.025_of_ref"] = bool(task - args.reference <= 0.025)
        gate["condition_b_base_0.03_and_ref_0.025"] = bool(
            base - task >= 0.03 and task - args.reference <= 0.025)
    gate["stage_2_triggered"] = bool(gate.get("condition_a_task_beats_freq_0.015", False)
                                     or gate.get("condition_b_base_0.03_and_ref_0.025", False))
    gate["stage_2_executed"] = False
    gate["stage_3_executed"] = False
    gate["stopped_by"] = "frozen progression rule after D0/s0" if not gate["stage_2_triggered"] else \
        "stage 2 not run"

    # vocabulary overlap / support stability (loose second-split check)
    v0 = C.read_json(C.ANALYSIS_DIR / "vocab_D0_task.json")
    v1 = C.read_json(C.ANALYSIS_DIR / "vocab_D1_task.json")
    k0 = [str(e["key"]) for e in v0]
    k1 = [str(e["key"]) for e in v1]
    s0, s1 = set(k0), set(k1)
    overlap = sorted(s0 & s1)
    stability = {
        "D0_task_size": len(k0), "D1_task_size": len(k1),
        "intersection": len(overlap),
        "jaccard": len(overlap) / max(1, len(s0 | s1)),
        "overlap_coefficient": len(overlap) / max(1, min(len(s0), len(s1))),
        "D0_size_gran": dict(Counter((e["size"], e["gran"]) for e in v0)),
        "D1_size_gran": dict(Counter((e["size"], e["gran"]) for e in v1)),
        "D0_min_scaffold_df": min(e["scaffold_df"] for e in v0),
        "D0_median_scaffold_df": float(np.median([e["scaffold_df"] for e in v0])),
        "D1_min_scaffold_df": min(e["scaffold_df"] for e in v1),
        "D1_median_scaffold_df": float(np.median([e["scaffold_df"] for e in v1])),
        "note": "D0 and D1 Fit sets overlap; intersection is not independent replication",
    }

    # verdict rule, applied exactly as frozen
    task_better = (task is not None and freq is not None and task < freq)
    close_to_ref = False
    if task is not None:
        close_to_ref = bool(abs(task - args.reference) <= 0.025
                            and abs(freq - args.reference) <= 0.025)
    if task_better and close_to_ref:
        verdict = "VOCABULARY_PROMISING"
    elif task_better:
        verdict = "DISCOVERY_ONLY"
    elif close_to_ref:
        verdict = "COMPRESSION_ONLY"
    else:
        verdict = "NO_SUPPORTED_DISCOVERY"
    evaluation = {
        "protocol_id": C.load_config()["protocol_id"],
        "split": args.split, "reference_r2_ref": args.reference,
        "main_table": table, "progression_gates": gate,
        "bootstrap": boot_summary, "stability": stability,
        "verdict": verdict,
        "verdict_inputs": {
            "task_better_than_freq": bool(task_better),
            "task_and_freq_close_to_ref_0.025": bool(close_to_ref),
            "task_dev_rmse": task, "freq_dev_rmse": freq, "base_dev_rmse": base,
        },
        "timing": {k: {"wall_seconds": v["manifest"]["wall_seconds"],
                       "gpu_hours": v["manifest"]["gpu_hours"]} for k, v in runs.items()},
        "note": "round-resource judgment only; not a claim that all task-conditioned vocabularies are unlearnable",
        "frozen_utc": C.now_utc(),
    }
    C.write_json(C.ANALYSIS_DIR / "evaluation.json", evaluation)
    print(C.jsonable({"verdict": verdict, "gates": gate,
                      "main": table, "eval": stability}))


if __name__ == "__main__":
    main()