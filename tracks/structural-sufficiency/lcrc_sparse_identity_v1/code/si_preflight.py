"""Performance pre-check: project the 12 formal runs from measured epoch times.

Consumes the timed smoke records written by ``si_train.py --smoke --tag costprobe``
(2 epochs on the FULL Train split, i.e. no subsetting) and projects the GPU cost of
the 12 formal 100-epoch runs.  Writes ``analysis/preflight.json`` with the measured
times, the projection and the explicit budget decision.

The probe is intentionally minimal: it measures the real training loop (including
per-epoch Validation) and nothing else.  A round whose projection exceeded the hard
cap would stop here rather than shrink the plan.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import si_common as C


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="costprobe")
    ap.parse_args(argv)
    probes: dict[str, Any] = {}
    for path in sorted((C.RUNS / "smoke").glob("*")):
        run_json = path / "run.json"
        if not run_json.exists():
            continue
        record = C.read_json(run_json)
        if record.get("tag") != "costprobe":
            continue
        arm = str(record["arm"])
        curve = record["curve"]
        probes[arm] = {
            "unit": C.track_rel(path),
            "epochs_measured": int(record["epochs_run"]),
            "n_train": int(record["n_train"]),
            "subset": int(record["n_train"]) != 32901,
            "train_seconds_per_epoch": [float(c["train_seconds"]) for c in curve],
            "valid_seconds_per_epoch": [float(c["valid_seconds"]) for c in curve],
            "mean_epoch_seconds": float(record["timing"]["mean_epoch_seconds"]),
            "train_wall_seconds": float(record["timing"]["train_wall_seconds"]),
            "gpu_hours_measured": float(record["timing"]["gpu_hours_estimate"]),
            "gpu_peak_allocated_bytes": int(record["timing"]["gpu_peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(record["timing"]["gpu_peak_reserved_bytes"]),
            "process_wall_seconds": float(record["timing"]["process_wall_seconds"]),
            "hostname": record["env"]["hostname"],
            "git_commit": record["git_commit"],
            "torch": record["env"]["torch"],
            "device": record["env"].get("cuda_device_names"),
        }
    if not probes:
        raise SystemExit("no cost-probe records found")
    # conservative per-arm projection: slowest measured epoch time of the probe * 100
    projected: dict[str, Any] = {}
    for arm in C.NEURAL:
        if arm not in probes:
            raise SystemExit(f"no cost probe for {arm}; run it before the formal sweep")
        entry = probes[arm]
        per_epoch = max(entry["train_seconds_per_epoch"]) + max(entry["valid_seconds_per_epoch"])
        hours = per_epoch * C.EPOCHS / 3600.0
        projected[arm] = {
            "conservative_epoch_seconds": float(per_epoch),
            "projected_hours_per_run": float(hours),
            "n_runs": len(C.SEEDS),
            "projected_hours_total": float(hours * len(C.SEEDS)),
        }
    total = float(sum(v["projected_hours_total"] for v in projected.values()))
    measured = float(sum(v["gpu_hours_measured"] for v in probes.values()))
    out = {
        "schema": "lcrc-sparse-identity-preflight-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "purpose": "confirm the 12 formal runs fit the GPU budget before starting them",
        "probes": probes,
        "projection_rule": "max measured train epoch time + max measured Validation time in the 2-epoch probe, "
        "times 100 epochs, times 3 seeds",
        "projection": projected,
        "projected_total_gpu_hours": total,
        "probe_gpu_hours": measured,
        "projected_total_including_probe": total + measured,
        "gpu_soft_cap_hours": float(C.GPU_SOFT_CAP_HOURS),
        "gpu_hard_cap_hours": float(C.GPU_HARD_CAP_HOURS),
        "decision": "PROCEED" if total + measured <= C.GPU_HARD_CAP_HOURS else "STOP_PROJECTED_OVER_HARD_CAP",
        "within_soft_cap": bool(total + measured <= C.GPU_SOFT_CAP_HOURS),
        "within_hard_cap": bool(total + measured <= C.GPU_HARD_CAP_HOURS),
        "plan_unchanged": {
            "arms": list(C.NEURAL),
            "seeds": list(C.SEEDS),
            "epochs": int(C.EPOCHS),
            "dataset": C.DATASET,
        },
        "note": "one formal training process per GPU at a time; res-gpu0 and res-gpu1 used in parallel",
    }
    C.write_json(C.ANALYSIS / "preflight.json", out)
    C.log(
        f"projected {total:.3f} GPU-h for 12 formal runs (+{measured:.3f} probe) vs soft cap "
        f"{C.GPU_SOFT_CAP_HOURS} / hard cap {C.GPU_HARD_CAP_HOURS} -> {out['decision']}"
    )
    return 0 if out["within_hard_cap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
