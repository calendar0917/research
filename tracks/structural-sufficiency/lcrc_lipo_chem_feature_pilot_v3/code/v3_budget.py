"""Pre-run GPU budget for the v3 pilot.

Anchor: the two frozen v2 Fold-1 seed-0 R3 runs measured on ``res`` /
``res-gpu0`` (A100-SXM4-40GB, torch 2.5.1+cu124).  The chemistry augmentation
only widens the first ``rho`` layer (768 extra parameters), so its cost is
within run-to-run noise of the anchor.  A 10% overhead is added and the whole
pilot is capped at 0.30 A100 GPU-hours, including the conditional third run.
"""

from __future__ import annotations

from typing import Any

import v3_common as C
import v3_registry as R


def _anchor_seconds(arm: str) -> dict[str, Any]:
    ledger = C.read_json(C.V2_ANALYSIS / "gpu_ledger_v2.json")
    key = C.HISTORICAL[arm]["key"]
    for entry in ledger.get("entries", []):
        if entry.get("key") == key:
            return {
                "key": key,
                "measured_process_seconds": float(entry["measured_process_seconds"]),
                "historical_charged_gpu_hours": float(entry["charged_gpu_hours"]),
                "cuda_device_name": entry.get("cuda_device_name"),
            }
    raise FileNotFoundError(f"v2 ledger has no entry for {key}")


def build_budget() -> dict[str, Any]:
    anchors = {arm: _anchor_seconds(arm) for arm in ("A", "C")}
    startup = 8.0
    per_run = []
    for row in R.planned_registry()["runs"]:
        anchor = anchors[row["base_arm"]]
        seconds = float(anchor["measured_process_seconds"]) + startup
        per_run.append(
            {
                **row,
                "anchor_key": anchor["key"],
                "anchor_measured_process_seconds": anchor["measured_process_seconds"],
                "estimated_process_seconds": seconds,
                "estimated_gpu_hours": seconds / 3600.0,
            }
        )
    raw = sum(x["estimated_gpu_hours"] for x in per_run)
    overhead = 0.10
    total = raw * (1.0 + overhead)
    unconditional = sum(x["estimated_gpu_hours"] for x in per_run if not x["conditional"])
    conditional = sum(x["estimated_gpu_hours"] for x in per_run if x["conditional"])
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "ESTIMATE_ONLY_NOT_EXECUTED",
        "anchor": {
            "source": "v2 gpu_ledger_v2.json Fold-1 seed-0 R3 entries on res/res-gpu0",
            "A": anchors["A"],
            "C": anchors["C"],
            "startup_allowance_seconds_per_run": startup,
            "chemistry_parameter_delta": {"A_Chem12": 768, "A_Chem13": 832, "C_Chem12": 768},
            "assumption": "wider rho input is within run-to-run noise; no per-update cost multiplier",
        },
        "per_run": per_run,
        "raw_estimated_gpu_hours": raw,
        "overhead_fraction": overhead,
        "estimated_total_gpu_hours_all_three": total,
        "estimated_unconditional_gpu_hours": unconditional * (1.0 + overhead),
        "estimated_conditional_gpu_hours": conditional * (1.0 + overhead),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "fits_within_hard_cap": bool(total <= C.GPU_HARD_CAP_HOURS),
        "margin_gpu_hours": C.GPU_HARD_CAP_HOURS - total,
        "historical_v1_v2_hours_reference_only": C.HISTORICAL_V1_V2_HOURS_REFERENCE_ONLY,
        "n_runs": len(per_run),
        "includes_conditional_third_run": True,
    }
    if not payload["fits_within_hard_cap"]:
        raise SystemExit(
            f"v3 budget estimate {total:.4f} h exceeds the {C.GPU_HARD_CAP_HOURS} h cap; do not authorize GPU"
        )
    C.write_json(C.ANALYSIS / "budget_v3.json", payload)
    return payload


__all__ = ["build_budget"]
