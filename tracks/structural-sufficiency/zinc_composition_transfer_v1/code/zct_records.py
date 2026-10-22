"""Assemble the committed record files for the round.

Writes (into ``cache/artifacts``): ``valid_per_molecule.csv``,
``run_records.json`` and ``budget.json``.  Reads only frozen run artifacts plus
the ``rr jobs --json`` table committed as ``analysis/rr_jobs.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

RUNS = _TASK / "runs"
ANALYSIS = _TASK / "analysis"
OUT = _TASK / "cache" / "artifacts"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=float, default=8.0)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    core = ["E_s0", "E_s1", "R_s0", "R_s1", "S_s0", "S_s1", "R_s0_k04", "S_s0_k04", "R_s1_k04"]
    records = {"round": "zinc-composition-transfer-v1", "runs": {}, "support_runs": {}}
    cols = {}
    for name in core:
        d = RUNS / name
        if not (d / "train.json").exists():
            continue
        info = C.read_json(d / "train.json")
        rec = {
            "arm": info["arm"], "seed": info["seed"], "kappa": info["kappa"],
            "parameters": info["parameter_count"],
            "ssl_encoder_sha256": info["model_info"]["sha256"],
            "init_state_sha256": info["init_state_sha256"],
            "git_commit_reported": info.get("git_commit"),
            "device": info.get("device"), "torch": (info.get("environment") or {}).get("torch"),
            "wall_seconds": info.get("wall_seconds"), "gpu_hours": info.get("gpu_hours"),
            "best": info["best_valid"], "fixed_objects": info["fixed_objects"],
            "checkpoints": {
                w: {"sha256": C.sha256_file(d / f"{w}.pt")}
                for w in ("best", "final", "fork") if (d / f"{w}.pt").exists()
            },
            "epoch1_batch_union_sha256": info["curve"][0]["batch_union_sha256"],
            "epoch600_batch_union_sha256": info["curve"][-1]["batch_union_sha256"],
            "train_label_sha256": info["data"]["train_label_sha256"],
            "valid_label_sha256": info["data"]["valid_label_sha256"],
        }
        for w in ("best", "final"):
            f = OUT / f"export_{name}_{w}.npz"
            if f.exists():
                cols[f"{name}_{w}"] = np.asarray(np.load(f)["pred"], dtype=np.float64)
        records["runs"][name] = rec
    for fit in ("plus", "minus"):
        for seed in (0, 1):
            d = RUNS / f"sup_{fit}_s{seed}"
            if not (d / "train.json").exists():
                continue
            info = C.read_json(d / "train.json")
            records["support_runs"][f"{fit}_s{seed}"] = {
                "fit": fit, "seed": seed, "rows": info["rows"],
                "steps_per_epoch": info["steps_per_epoch"], "updates": info["updates"],
                "best_monitor": info["best_monitor"],
                "final_monitor": info["curve"][-1]["monitor_mae"],
                "init_state_sha256": info["init_state_sha256"],
                "manifest_sha256": info["manifest_sha256"],
                "gpu_hours": info.get("gpu_hours"), "wall_seconds": info.get("wall_seconds"),
                "checkpoints": {w: {"sha256": C.sha256_file(d / f"{w}.pt")}
                                for w in ("best_monitor", "final") if (d / f"{w}.pt").exists()},
            }
    C.write_json(OUT / "run_records.json", records)

    # per-molecule valid predictions (best and final checkpoint of every core run)
    import ad_data as D
    y = np.asarray(D.load_cache("valid")["labels"], dtype=np.float64)
    src = _SSUF / "zinc_restricted_adaptation_v1" / "analysis" / "reference_preds_valid.csv"
    with open(src, newline="", encoding="utf-8") as fh:
        ref = list(csv.DictReader(fh))
    names = {(0, 1): "seen_ordinary", (0, 0): "seen_tail",
             (1, 1): "unseen_ordinary", (1, 0): "unseen_tail"}
    of_row = {str(i): names[(int(r["signature_unseen"]), int(r["ordinary_y_gt_minus3"]))]
              for i, r in enumerate(ref)}
    counts = {}
    for v in of_row.values():
        counts[v] = counts.get(v, 0) + 1
    g = {"of_row": of_row, "sizes": counts,
         "source": str(src.relative_to(_TASK.parents[1])),
         "source_sha256": C.sha256_file(src)}
    C.write_json(OUT / "group_frozen.json", {"sizes": counts, "source": g["source"],
                                             "source_sha256": g["source_sha256"]})
    path = OUT / "valid_per_molecule.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["row", "y"] + sorted(cols)
        if g:
            head += ["group"]
        w.writerow(head)
        for i in range(len(y)):
            row = [i, f"{y[i]:.6f}"] + [f"{cols[c][i]:.6f}" for c in sorted(cols)]
            if g:
                row.append(g["of_row"][str(i)])
            w.writerow(row)

    # budget from the rr job table
    jobs = C.read_json(ANALYSIS / "rr_jobs.json")["jobs"]
    rows, total = [], 0.0
    for r in reversed(jobs):
        if not r["experiment"].startswith("zct"):
            continue
        start = datetime.fromisoformat(r["started_at"])
        end = datetime.fromisoformat(r["completed_at"])
        hours = (end - start).total_seconds() / 3600.0
        gpus = 0 if r["pool"] == "res-local" else 1
        total += hours * gpus
        rows.append({"experiment": r["experiment"], "run_id": r["run_id"], "pool": r["pool"],
                     "pool_kind": "cpu" if gpus == 0 else "gpu", "state": r["state"],
                     "wall_hours": round(hours, 4), "gpus": gpus,
                     "gpu_hours": round(hours * gpus, 4), "commit": r["commit"],
                     "exit_code": r.get("exit_code")})
    trainer_total = sum(v["gpu_hours"] or 0.0 for v in records["runs"].values()) + \
        sum(v["gpu_hours"] or 0.0 for v in records["support_runs"].values())
    budget = {
        "cap_gpu_hours": args.cap,
        "gpu_hours_job_wall": round(total, 4),
        "gpu_hours_trainer_measured": round(trainer_total, 4),
        "remaining_gpu_hours": round(args.cap - total, 4),
        "wall_clock_start": "2026-10-23T00:25:27+08:00",
        "wall_clock_end": rows[-1]["run_id"][-8:] if rows else None,
        "note": ("job wall includes process start-up and every failed/cancelled run; "
                 "two jobs shared one GPU for ~8 minutes, so the wall sum is a mild "
                 "over-estimate of exclusive device time"),
        "jobs": rows,
    }
    C.write_json(OUT / "budget.json", budget)
    C.log({"gpu_hours_job_wall": total, "trainer_measured": trainer_total,
           "remaining": args.cap - total})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
