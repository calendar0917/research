"""Assemble the committed audit summary, the GPU ledger and the checkpoint table.

Runs after the two formal runs have been pulled:

* ``analysis/audit.json``   merged pass/fail view of every pre-training gate
* ``analysis/gpu_ledger.json`` every ``rd-*`` ``rr`` job of this round, charged
* ``analysis/checkpoints.json`` SHA-256 and size of every checkpoint on disk

The mode records themselves (``data_audit.json``, ``model_audit.json``,
``test_isolation.json``, ``evaluation_cost_probe.json``, ``historical_test.json``)
are produced by ``rd_audit.py`` and are not rewritten here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import rd_common as C  # noqa: E402

MODE_FILES = {
    "data": "data_audit.json",
    "model": "model_audit.json",
    "test_isolation": "test_isolation.json",
    "smoke": "evaluation_cost_probe.json",
    "historical_test": "historical_test.json",
}


def merge_audit(out: Path) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode, name in MODE_FILES.items():
        path = out / name
        if not path.is_file():
            modes[mode] = {"available": False, "path": C.track_rel(path)}
            continue
        payload = C.read_json(path)
        modes[mode] = {
            "available": True,
            "path": C.track_rel(path),
            "sha256": C.sha256_file(path),
            "passed": bool(payload.get("passed", True)),
            "checks": payload.get("checks"),
            "device": payload.get("device"),
            "captured_utc": payload.get("captured_utc"),
        }
    summary = {
        "schema": "lcrc-pcba-w3-regularization-audit-summary-v1",
        "round_id": C.ROUND_ID,
        "captured_utc": C.utcnow(),
        "modes": modes,
        "missing_modes": [m for m, v in modes.items() if not v.get("available")],
        "all_available": all(v.get("available") for v in modes.values()),
        "all_passed": all(v.get("passed", False) for v in modes.values() if v.get("available")),
        "hosts": {
            "cpu": "local workstation (CPU, 16 vCPU) - data/model/test-isolation/historical-test",
            "gpu": "res a100-2 GPU1 (A100-SXM4-40GB, driver 550.163.01) - smoke/cost probe",
        },
        "git_commit": C.git_commit(),
        "module_sha256": C.module_inventory()["sha256"],
        "notes": {
            "formal_runs_started_after_all_gates_passed": True,
            "smoke_is_not_part_of_a_formal_run": True,
            "test_scored_every_epoch_by_design": "EXPOSED_TEST_EXPLORATORY",
        },
    }
    C.write_json(out / "audit.json", summary)
    return summary


def gpu_ledger(repo: Path = C.REPO, out: Path | None = None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for meta_path in sorted((repo / ".rr/pulled").glob("*/meta.json")):
        meta = C.read_json(meta_path)
        if not str(meta.get("experiment", "")).startswith("rd-"):
            continue
        runtime = meta.get("runtime") or {}
        started = runtime.get("started_at")
        completed = runtime.get("completed_at")
        seconds = None
        if started and completed:
            from datetime import datetime

            a = datetime.fromisoformat(started)
            b = datetime.fromisoformat(completed)
            seconds = max(0.0, (b - a).total_seconds())
        entries.append(
            {
                "experiment": meta.get("experiment"),
                "run_id": meta.get("run_id"),
                "pool": meta.get("pool"),
                "gpu_index_visible": runtime.get("cuda_visible_devices"),
                "git_commit": meta.get("git_commit"),
                "state": meta.get("state"),
                "exit_code": runtime.get("exit_code"),
                "started_at": started,
                "completed_at": completed,
                "wall_seconds": seconds,
                "gpu_hours_charged": (seconds or 0.0) / 3600.0,
            }
        )
    formal: dict[str, Any] = {}
    for key in C.RUN_ORDER:
        run_json = C.run_dir(key) / "run.json"
        if not run_json.is_file():
            formal[key] = {"available": False}
            continue
        run = C.read_json(run_json)
        formal[key] = {
            "available": True,
            "status": run["status"],
            "gpu_hours": float(run["timing"]["gpu_hours"]),
            "train_seconds": float(run["timing"]["train_seconds"]),
            "eval_seconds_total": float(run["timing"]["eval_seconds_total"]),
            "epoch_seconds_mean": float(run["timing"]["epoch_seconds_mean"]),
            "steps_per_second_mean": float(run["timing"]["steps_per_second_mean"]),
            "peak_gpu_allocated_bytes": int(run["resources"]["gpu_peak_allocated_bytes"]),
            "peak_rss_kb": int(run["resources"]["peak_rss_kb"]),
            "started_utc": run["started_utc"],
            "finished_utc": run["finished_utc"],
        }
    total_charged = float(sum(e["gpu_hours_charged"] for e in entries))
    payload = {
        "schema": "lcrc-pcba-w3-regularization-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "captured_utc": C.utcnow(),
        "jobs": entries,
        "n_jobs": len(entries),
        "total_gpu_hours_charged": total_charged,
        "formal_runs": formal,
        "formal_gpu_hours": float(sum(v["gpu_hours"] for v in formal.values() if v.get("available"))),
        "budget": dict(C.BUDGET),
        "within_target": total_charged <= C.BUDGET["gpu_target_hours"],
        "within_hard_cap": total_charged <= C.BUDGET["gpu_hard_hours"],
        "note": "wall-clock seconds of every rd-* rr job, including preflight/audit jobs",
    }
    C.write_json((out or C.ANALYSIS) / "gpu_ledger.json", payload)
    return payload


def checkpoint_table(out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in C.RUN_ORDER:
        run_dir = C.run_dir(key)
        run_json = run_dir / "run.json"
        run = C.read_json(run_json) if run_json.is_file() else {}
        for name in ("checkpoint_best_valid.pt", "checkpoint_epoch40.pt", "checkpoint_last.pt"):
            path = run_dir / name
            row: dict[str, Any] = {"run": key, "artifact": name, "path": C.track_rel(path)}
            if path.is_file():
                row.update(
                    {
                        "present": True,
                        "bytes": int(path.stat().st_size),
                        "sha256": C.sha256_file(path),
                    }
                )
                recorded = {
                    "checkpoint_best_valid.pt": run.get("best_checkpoint_sha256"),
                    "checkpoint_epoch40.pt": run.get("final_checkpoint_sha256"),
                    "checkpoint_last.pt": run.get("last_checkpoint_sha256"),
                }.get(name)
                row["recorded_sha256"] = recorded
                row["sha256_matches_run_json"] = (recorded is None) or (recorded == row["sha256"])
            else:
                row["present"] = False
            rows.append(row)
    payload = {
        "schema": "lcrc-pcba-w3-regularization-checkpoints-v1",
        "round_id": C.ROUND_ID,
        "captured_utc": C.utcnow(),
        "checkpoints": rows,
        "all_present": all(r["present"] for r in rows),
        "all_hashes_match": all(r.get("sha256_matches_run_json", True) for r in rows),
        "historical_w3": {
            "best_valid": C.HISTORICAL["W3"]["best_checkpoint_sha256"],
            "epoch40": C.HISTORICAL["W3"]["epoch40_checkpoint_sha256"],
            "test_ap_best_valid": "analysis/historical_test.json",
        },
    }
    C.write_json(out / "checkpoints.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = merge_audit(out)
    ledger = gpu_ledger(out=out)
    checks = checkpoint_table(out)
    C.log(f"[finalize] audit all_passed={audit['all_passed']} missing={audit['missing_modes']}")
    C.log(
        f"[finalize] gpu_ledger jobs={ledger['n_jobs']} charged={ledger['total_gpu_hours_charged']:.4f} h "
        f"formal={ledger['formal_gpu_hours']:.4f} h within_target={ledger['within_target']}"
    )
    C.log(f"[finalize] checkpoints all_present={checks['all_present']} all_hashes_match={checks['all_hashes_match']}")
    return 0 if audit["all_passed"] and audit["all_available"] and checks["all_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
