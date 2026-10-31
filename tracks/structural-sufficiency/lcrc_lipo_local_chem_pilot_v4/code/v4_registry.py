"""Frozen run registry, GPU budget estimate and GPU-hour ledger for v4.

Exactly one unconditional GPU run (P1, ``A_AtomChem4``) and one conditional
contrast (P2, ``A_BroadcastChem4``).  P2 may only start after P1 passes the
pre-registered gate.  No other seed, fold, recipe, architecture, feature
variant or full-Fit retrain is registered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import v4_common as C


def _row(key: str, arm: str, *, conditional: bool) -> dict[str, Any]:
    return {
        "key": key,
        "kind": "train",
        "arm": arm,
        "feature_variant": C.ARM_VARIANT[arm],
        "fold": int(C.FOLD),
        "recipe": C.RECIPE,
        "seed": int(C.SEED),
        "variant": "scratch",
        "conditional": bool(conditional),
        "comparator": C.HISTORICAL["key"],
        "host": "res",
        "pool": "res-gpu0",
        "description": C.ARM_DESCRIPTION[arm],
    }


def planned_registry() -> dict[str, Any]:
    runs = [
        _row("F1_A_AtomChem4_R3_s0", "A_AtomChem4", conditional=False),
        _row("F1_A_BroadcastChem4_R3_s0", "A_BroadcastChem4", conditional=True),
    ]
    keys = [r["key"] for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate run key in registry")
    return {
        "protocol_id": C.PROTOCOL_ID,
        "round_id": C.ROUND_ID,
        "runs": runs,
        "n_runs": len(runs),
        "n_unconditional": sum(1 for r in runs if not r["conditional"]),
        "n_conditional": sum(1 for r in runs if r["conditional"]),
        "n_gpu_jobs": len(runs),
        "allowed_extra_runs": 0,
        "forbidden": [
            "new seeds",
            "second fold",
            "SSL",
            "additional atom features or feature combinations",
            "architecture changes",
            "full-Fit retrain",
            "Monitor / Reserved evaluation",
            "official valid/test labels",
        ],
        "comparator": C.HISTORICAL["key"],
        "conditional_rule": {
            "arm": "A_BroadcastChem4",
            "run_key": "F1_A_BroadcastChem4_R3_s0",
            "rmse_gain_min": C.MATERIALITY_RMSE,
            "mae_must_not_worsen": True,
            "tail_fraction": C.TAIL_FRACTION,
            "tail_min_retained_fraction": C.TAIL_MIN_RETAINED_FRACTION,
            "must_not_be_tail_dependent": True,
            "evaluated_against": C.HISTORICAL["key"],
        },
        "verdict_rules": {
            "LOCAL_CHEM_ALIGNMENT_PROMISING": (
                "P1 gate passed and (P2 delta_rmse < materiality or "
                "P1 delta_rmse - P2 delta_rmse >= alignment_gap_material)"
            ),
            "CHEMICAL_FEATURE_INFORMATION_PROMISING": (
                "P1 gate passed and P2 also reaches materiality and the P1-P2 gap is below alignment_gap_material"
            ),
            "NO_CLEAR_LOCAL_CHEM_GAIN": "P1 delta_rmse < materiality (P2 is not run)",
            "INCONCLUSIVE": "P1 delta_rmse >= materiality but MAE worsens or the gain is tail-dependent",
            "materiality_rmse": C.MATERIALITY_RMSE,
            "alignment_gap_material": C.ALIGNMENT_GAP_MATERIAL,
        },
    }


def run_by_key(key: str) -> dict[str, Any]:
    for row in planned_registry()["runs"]:
        if row["key"] == key:
            return row
    raise KeyError(f"unknown v4 run key {key!r}")


def unconditional_keys() -> list[str]:
    return [r["key"] for r in planned_registry()["runs"] if not r["conditional"]]


def conditional_key() -> str:
    rows = [r for r in planned_registry()["runs"] if r["conditional"]]
    if len(rows) != 1:
        raise ValueError("registry must contain exactly one conditional run")
    return rows[0]["key"]


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def _anchor() -> dict[str, Any]:
    ledger = C.read_json(C.V2_ANALYSIS / "gpu_ledger_v2.json")
    key = C.HISTORICAL["key"]
    for entry in ledger.get("entries", []):
        if entry.get("key") == key:
            return {
                "key": key,
                "measured_process_seconds": float(entry["measured_process_seconds"]),
                "historical_charged_gpu_hours": float(entry["charged_gpu_hours"]),
                "cuda_device_name": entry.get("cuda_device_name"),
                "git_commit": entry.get("git_commit"),
            }
    raise FileNotFoundError(f"v2 ledger has no entry for {key}")


def build_budget() -> dict[str, Any]:
    anchor = _anchor()
    startup = 8.0
    # P2 is the same graph/parameter count/recipe; P1 adds 512 parameters on
    # the first atom-encoder layer only, which is within timing noise.
    seconds = float(anchor["measured_process_seconds"]) + startup
    per_run = []
    for row in planned_registry()["runs"]:
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
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "ESTIMATE_ONLY_NOT_EXECUTED",
        "anchor": {
            "source": "v2 gpu_ledger_v2.json F1_A_sel_s0 (Fold 1, seed 0, R3, arm A) on res/res-gpu0",
            **anchor,
            "startup_allowance_seconds_per_run": startup,
            "parameter_delta_per_arm": C.EXPECTED_PARAMETER_DELTA,
            "assumption": "512 extra parameters in the first atom-encoder layer are within run-to-run timing noise; no per-update multiplier",
        },
        "per_run": per_run,
        "raw_estimated_gpu_hours": raw,
        "overhead_fraction": overhead,
        "estimated_total_gpu_hours_all_runs": total,
        "estimated_unconditional_gpu_hours": per_run[0]["estimated_gpu_hours"] * (1.0 + overhead),
        "estimated_conditional_gpu_hours": per_run[1]["estimated_gpu_hours"] * (1.0 + overhead),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "fits_within_hard_cap": bool(total <= C.GPU_HARD_CAP_HOURS),
        "margin_gpu_hours": C.GPU_HARD_CAP_HOURS - total,
        "historical_v1_v2_v3_hours_reference_only": C.HISTORICAL_V1_V2_V3_HOURS_REFERENCE_ONLY,
        "n_runs": len(per_run),
        "includes_conditional_second_run": True,
    }
    if not payload["fits_within_hard_cap"]:
        raise SystemExit(f"v4 budget estimate {total:.4f} h exceeds the {C.GPU_HARD_CAP_HOURS} h cap; do not authorize GPU")
    C.write_json(C.ANALYSIS / "budget_v4.json", payload)
    return payload


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def _rr_wall(run_dir: Path) -> float | None:
    meta = run_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("wall_seconds", "duration_seconds", "elapsed_seconds"):
        if payload.get(key) is not None:
            try:
                return float(payload[key])
            except Exception:
                pass
    return None


def rebuild_ledger() -> dict[str, Any]:
    entries = []
    total = 0.0
    for row in planned_registry()["runs"]:
        key = row["key"]
        run_dir = C.RUNS / key
        train_path = run_dir / "train.json"
        if not train_path.exists():
            continue
        payload = C.read_json(train_path)
        measured = float(payload.get("wall_seconds", 0.0)) + 8.0
        rr_wall = _rr_wall(run_dir)
        charged = max(measured, rr_wall) if rr_wall is not None else measured
        hours = charged / 3600.0
        total += hours
        entries.append(
            {
                "key": key,
                "arm": row["arm"],
                "fold": row["fold"],
                "seed": row["seed"],
                "recipe": row["recipe"],
                "conditional": row["conditional"],
                "measured_process_seconds": float(payload.get("wall_seconds", 0.0)),
                "rr_wall_seconds": rr_wall,
                "charged_seconds": charged,
                "charged_gpu_hours": hours,
                "cuda_device_name": payload.get("cuda_device_name"),
                "torch": payload.get("torch"),
                "git_commit": payload.get("git_commit"),
                "soup_file_sha256": payload.get("soup_file_sha256"),
                "predictions_sha256": payload.get("predictions_sha256"),
                "file": C.track_rel(train_path),
            }
        )
    if not entries:
        status = "NO_GPU_RUNS"
    elif total <= C.GPU_HARD_CAP_HOURS:
        status = "WITHIN_CAP"
    else:
        status = "CAP_EXCEEDED"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "historical_v1_v2_v3_hours_reference_only": C.HISTORICAL_V1_V2_V3_HOURS_REFERENCE_ONLY,
        "n_jobs_charged": len(entries),
        "n_unconditional_executed": sum(1 for e in entries if not e["conditional"]),
        "n_conditional_executed": sum(1 for e in entries if e["conditional"]),
        "total_gpu_hours": total,
        "remaining_gpu_hours": C.GPU_HARD_CAP_HOURS - total,
        "entries": entries,
        "status": status,
    }
    C.write_json(C.ANALYSIS / "gpu_ledger_v4.json", payload)
    if total > C.GPU_HARD_CAP_HOURS:
        raise SystemExit(f"v4 GPU ledger exceeds the {C.GPU_HARD_CAP_HOURS} h cap: {total:.4f} h")
    return payload


__all__ = [
    "planned_registry",
    "run_by_key",
    "unconditional_keys",
    "conditional_key",
    "build_budget",
    "rebuild_ledger",
]
