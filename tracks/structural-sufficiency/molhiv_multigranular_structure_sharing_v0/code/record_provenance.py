"""Assemble the round's provenance / resource record (no model is trained here).

Writes ``analysis/provenance.json``: code revision, environment, input and
artefact checksums, per-phase timings, and the declared-vs-used budget table.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import msds_common as C

BUDGET_MINUTES = {"enumeration_cpu": 60.0, "linear_arms": 45.0, "xgboost": 30.0, "gpu_hours": 4.0}


def sha(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": C.rel(path), "sha256": h.hexdigest(), "size_bytes": path.stat().st_size}


def inputs() -> dict[str, Any]:
    raw = C.OGB_MOLHIV
    out: dict[str, Any] = {"ogb_raw": {}, "ogb_split": {}, "historical": {}}
    for p in sorted((raw / "raw").glob("*")):
        out["ogb_raw"][p.name] = sha(p)
    for p in sorted((raw / "split" / "scaffold").glob("*")):
        out["ogb_split"][p.name] = sha(p)
    out["ogb_split"]["mapping/mol.csv.gz"] = sha(raw / "mapping" / "mol.csv.gz")
    for p in (C.HIST_BASELINE_MANIFEST, C.HIST_BASELINE_FEATURES,
              C.HIST_BASELINE_DIR / "ogbg-molhiv_valid_predictions.npz",
              C.HIST_BASELINE_DIR / "ogbg-molhiv_test_predictions.npz"):
        out["historical"][C.rel(p)] = sha(p)
    return out


def artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {"cache": {}, "analysis": {}}
    for p in sorted(C.CACHE.glob("*.npz")):
        out["cache"][p.name] = sha(p)
    for p in sorted(C.ANALYSIS.glob("*.json")):
        if p.name == "provenance.json":
            continue
        out["analysis"][p.name] = sha(p)
    for p in sorted((C.ANALYSIS / "phaseE").glob("*")):
        out["analysis"][f"phaseE/{p.name}"] = sha(p)
    for p in sorted((C.ANALYSIS / "predictions").glob("*.npz")):
        out["analysis"][f"predictions/{p.name}"] = sha(p)
    for p in sorted(C.ANALYSIS.glob("*.csv")):
        out["analysis"][p.name] = sha(p)
    return out


def timings() -> dict[str, Any]:
    def seconds(name: str) -> Any:
        path = C.ANALYSIS / name
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        for key in ("seconds_total", "seconds"):
            if key in payload:
                return float(payload[key])
        return None

    phase_e = {}
    total_gpu = 0.0
    for p in sorted((C.ANALYSIS / "phaseE").glob("*_seed0.json")):
        payload = json.loads(p.read_text())
        phase_e[payload["arm"]] = {
            "seconds_total": float(payload["seconds_total"]),
            "epochs_run": len(payload["history"]),
            "selected_epoch": payload["best_valid"]["epoch"],
            "trainable_parameters": payload["trainable_parameters"],
            "initial_backbone_param_sha256": payload["initial_backbone_param_sha256"],
            "device": payload["provenance"]["device"],
            "gpu_name": payload["provenance"]["gpu_name"],
            "hostname": payload["provenance"]["hostname"],
            "provenance_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        total_gpu += float(payload["seconds_total"])
    return {
        "phases": {
            "phaseA_enumeration_and_vocab": seconds("phaseA_run.json"),
            "vocabulary_freeze": seconds("vocabulary.json"),
            "phaseB_linear_arms": seconds("phaseB_linear.json"),
            "phaseD_xgboost": seconds("phaseD_xgb.json"),
            "phaseC_analyses": seconds("phaseC_meta.json"),
            "canonical_partition_check": (
                json.loads((C.ANALYSIS / "canonical_partition_check.json").read_text())["seconds"]
                if (C.ANALYSIS / "canonical_partition_check.json").exists() else None
            ),
        },
        "phaseE": phase_e,
        "gpu_seconds_total": total_gpu,
        "gpu_hours_total": total_gpu / 3600.0,
    }


def main() -> None:
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t = timings()
    hours = {
        "enumeration_cpu_minutes": (t["phases"]["phaseA_enumeration_and_vocab"] or 0) / 60.0
        + (t["phases"]["vocabulary_freeze"] or 0) / 60.0,
        "linear_arms_minutes": (t["phases"]["phaseB_linear_arms"] or 0) / 60.0,
        "xgboost_minutes": (t["phases"]["phaseD_xgboost"] or 0) / 60.0,
        "gpu_hours": t["gpu_hours_total"],
    }
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "phase": "provenance and resource record",
        "created_utc": created,
        "git": C.git_state(),
        "environment_local": C.environment(),
        "environment_remote": {
            p.name: json.loads(p.read_text())["provenance"]
            for p in sorted((C.ANALYSIS / "phaseE").glob("*_seed0.json"))
        },
        "inputs": inputs(),
        "artifacts": artifacts(),
        "timings": t,
        "budget": {
            "declared": BUDGET_MINUTES,
            "used": hours,
            "within_budget": {
                "enumeration_cpu": hours["enumeration_cpu_minutes"] <= BUDGET_MINUTES["enumeration_cpu"],
                "linear_arms": hours["linear_arms_minutes"] <= BUDGET_MINUTES["linear_arms"],
                "xgboost": hours["xgboost_minutes"] <= BUDGET_MINUTES["xgboost"],
                "gpu_hours": hours["gpu_hours"] <= BUDGET_MINUTES["gpu_hours"],
            },
            "neural_training_units": {"declared_max": 3, "used": 2,
                                      "detail": "G-M and G-STRUCT-U, one seed (0) each, 30 epochs"},
            "notes": [
                "measured: 4.23M instances enumerated in ~73 s with 10 local workers",
                "failed/smoke GPU attempts (2 aborted runs + 1 timing run) consumed < 0.1 A100-hour and "
                "trained no reported model",
                "Phase E ran on a single A100-40GB; the two arms ran in parallel on different nodes",
            ],
        },
        "host": {
            "local": {"node": platform.node(), "python": sys.version.split()[0], "cpus": C.environment()["cpu_count"]},
        },
    }
    C.write_json(C.ANALYSIS / "provenance.json", payload)
    print(json.dumps({"used": hours, "ok": payload["budget"]["within_budget"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
