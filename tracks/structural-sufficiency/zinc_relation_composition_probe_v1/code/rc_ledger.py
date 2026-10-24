"""Build analysis/gpu_ledger.json for the relation-composition probe round."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # structural-sufficiency
OUT = ROOT / "zinc_relation_composition_probe_v1/analysis/gpu_ledger.json"
ROUND_START = "2026-10-23T22:05:00Z"  # session start (branch created 22:15:36Z)


def parse(ts):
    if not ts:
        return None
    return datetime.fromisoformat(ts)


def hours(a, b):
    if not (a and b):
        return None
    return round((b - a).total_seconds() / 3600.0, 4)


raw = json.loads(subprocess.run(["rr", "--json", "jobs"], capture_output=True, text=True, check=True).stdout)
jobs = [j for j in raw["jobs"] if str(j.get("experiment", "")).startswith("rc-")]
entries = []
for j in jobs:
    st, en = parse(j["started_at"] or j["created_at"]), parse(j["completed_at"])
    entries.append({
        "experiment": j["experiment"],
        "run_id": j["run_id"],
        "host": j["host"],
        "backend": j["backend"],
        "pool": j["pool"],
        "device": j.get("cuda_visible_devices"),
        "commit": j.get("commit"),
        "state": j["state"],
        "exit_code": j.get("exit_code"),
        "started_at": j["started_at"],
        "completed_at": j["completed_at"],
        "elapsed_hours": hours(st, en) if en else None,
        "gpu_count": 1 if j["experiment"].startswith(("rc-D-s0", "rc-S-s0", "rc-timing-D")) else 0,
        "note": "",
    })
notes = {
    "rc-hygiene": "environment diagnosis on the remote checkout (untracked-file collision with the incoming commit); CPU only",
    "rc-hygiene2": "environment diagnosis, CPU only",
    "rc-hygiene3": "moved colliding untracked files aside, CPU only",
    "rc-hygiene4": "moved the remaining colliding untracked files aside, CPU only",
    "rc-timing-D": "cost measurement: full-split epoch timing + label-free R-cache build",
}
for e in entries:
    e["note"] = notes.get(e["experiment"], "")
    if e["experiment"] in ("rc-D-s0", "rc-S-s0"):
        e["note"] = "formal run: declared R0 schedule, seed 0, arm " + e["experiment"][3]

formal = [e for e in entries if e["experiment"] in ("rc-D-s0", "rc-S-s0")]
gpu_hours = sum((e["elapsed_hours"] or 0.0) for e in entries if e["gpu_count"])
ledger = {
    "round": "zinc-relation-composition-probe-v1",
    "host": "res (a100-2, 2x A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124)",
    "round_started_estimate": ROUND_START,
    "round_started_note": "session start; branch probe/relation-composition-v1 created 22:15:36Z",
    "budget_cap": {"wall_clock_hours": 3.0, "gpu_hours": 6.0},
    "measured_cost": {"seconds_per_epoch_train_arm": 6.68, "projected_600_epoch_gpu_hours": 1.1133494252959888},
    "jobs": entries,
    "formal_runs": [e["experiment"] for e in formal],
    "gpu_hours_from_job_records": round(gpu_hours, 4),
    "gpu_hours_training_only": round(sum((e["elapsed_hours"] or 0.0) for e in formal), 4),
    "dev_phase": "local CPU only (model, relation cache, checks, analysis, report); no GPU spent",
    "compliance": [
        "2 A100s, one run per card, both formal runs on the same committed revision 7464c0d30441",
        "official ZINC test never loaded or scored in this round",
        "no recipe/width/depth grid: exactly two arms (D/S) at one declared configuration",
    ],
}
OUT.write_text(json.dumps(ledger, indent=1))
print(json.dumps({k: v for k, v in ledger.items() if k != "jobs"}, indent=1))
print("formal elapsed (h):", [e["elapsed_hours"] for e in formal])
