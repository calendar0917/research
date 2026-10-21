"""Combine L/C full-epoch timing and record the pre-formal budget decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ANALYSIS = _HERE.parent / "analysis"
sys.path.insert(0, str(_HERE))

import ad_common as C


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timing-spent-hours", type=float, required=True,
                   help="actual rr job occupancy for both timing jobs, summed")
    p.add_argument("--cap-hours", type=float, default=0.8594111111111111)
    p.add_argument("--soft-target-hours", type=float, default=0.80)
    args = p.parse_args()
    arms = {}
    for arm in ("L", "C"):
        path = ANALYSIS / f"timing_{arm}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run the real 3-epoch timing first")
        arms[arm] = json.loads(path.read_text(encoding="utf-8"))
    proj_two = sum(float(arms[a]["projected_600ep_gpu_hours"]) for a in arms)
    planned = args.timing_spent_hours + proj_two
    hard_ok = planned <= args.cap_hours
    soft_ok = planned <= args.soft_target_hours
    out = {
        "protocol_id": "zinc-dual-channel-composer-v1",
        "timing_measurement": "full official train 10000 for every measured epoch, full valid-1000 every epoch; no 1024-row projection",
        "arms": arms,
        "projected_formal_two_arm_gpu_hours": proj_two,
        "timing_spent_gpu_hours_job_occupancy": args.timing_spent_hours,
        "planned_total_gpu_hours": planned,
        "soft_target_gpu_hours": args.soft_target_hours,
        "hard_cap_gpu_hours": args.cap_hours,
        "within_soft_target": bool(soft_ok),
        "within_hard_cap": bool(hard_ok),
        "decision": "OPEN_FORMAL_TWO_ARM" if hard_ok and soft_ok else "STOP_BUDGET",
        "note": "If not within the soft target and hard cap, do not launch formal runs; do not shorten 600 epochs or run only one arm.",
    }
    C.write_json(ANALYSIS / "timing.json", out)
    print(json.dumps(out, indent=2))
    return 0 if out["decision"] == "OPEN_FORMAL_TWO_ARM" else 2


if __name__ == "__main__":
    raise SystemExit(main())
