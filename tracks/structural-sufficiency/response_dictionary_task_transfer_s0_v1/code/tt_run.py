"""Formal CPU run: evaluate the frozen structure catalogue and record provenance."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tt_common as TC  # noqa: E402
import tt_catalogue as CAT  # noqa: E402
import tt_evaluate as EV  # noqa: E402


def _artifact_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(TC.ANALYSIS_DIR.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(TC.ROUND_DIR))] = TC.sha256_file(path)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-structure", action="store_true",
                        help="build the label-free structure catalogue if it is missing")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    status = "completed"
    error = None
    result: dict[str, Any] | None = None
    structure_built = False
    try:
        if args.build_structure and not (TC.STRUCTURE_DIR / "structure_freeze.json").is_file():
            CAT.build_structure()
            structure_built = True
        result = EV.evaluate()
    except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        wall = time.perf_counter() - t0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        manifest = {
            "round_id": TC.load_config()["round_id"],
            "status": status,
            "error": error,
            "wall_seconds": wall,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "max_rss_kb": usage.ru_maxrss,
            "gpu": "none; no GPU was requested",
        }
        TC.write_json(TC.ROUND_DIR / "run_manifest.json", manifest)
        raise

    wall = time.perf_counter() - t0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    structure_freeze = json.loads(
        (TC.STRUCTURE_DIR / "structure_freeze.json").read_text(encoding="utf-8")
    )
    artifacts = _artifact_hashes()
    report = TC.ROUND_DIR / "REPORT.md"
    if report.is_file():
        artifacts[str(report.relative_to(TC.ROUND_DIR))] = TC.sha256_file(report)
    manifest = {
        "round_id": TC.load_config()["round_id"],
        "status": status,
        "started_utc": datetime.fromtimestamp(time.time() - wall, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "max_rss_kb": usage.ru_maxrss,
        "gpu": "none; no GPU was requested",
        "gpu_hours_a100": 0.0,
        "host": platform.node(),
        "python": platform.python_version(),
        "code_commit": TC.RC.git_commit(),
        "protocol_hash": TC.protocol_hash(),
        "config_sha256": TC.sha256_file(TC.CONFIG_PATH),
        "protocol_sha256": TC.sha256_file(TC.PROTOCOL_PATH),
        "source_hashes": TC.source_hashes(),
        "structure_freeze": {
            "path": str((TC.STRUCTURE_DIR / "structure_freeze.json").relative_to(TC.ROUND_DIR)),
            "structure_hash": structure_freeze["structure_hash"],
            "config_sha256": structure_freeze["config_sha256"],
            "protocol_sha256": structure_freeze["protocol_sha256"],
            "n_pairs": structure_freeze["n_pairs"],
            "n_candidates": structure_freeze["n_candidates"],
            "n_components": structure_freeze["n_components"],
            "n_E_T": structure_freeze["n_E_T"],
            "n_E_HH_T": structure_freeze["n_E_HH_T"],
            "n_E_STRICT": structure_freeze["n_E_STRICT"],
            "structure_built_in_this_run": bool(structure_built),
        },
        "inputs": EV.input_hashes(),
        "frozen_full_checks": EV.frozen_full_metrics(TC.load_payload(), TC.prediction_arrays())[1],
        "commands": [
            "uv run python tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/code/tt_catalogue.py",
            "uv run python tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/code/tt_evaluate.py",
            "uv run python tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/code/tt_run.py",
        ],
        "artifacts": artifacts,
        "counts": {
            "n_pairs": result["n_pairs"],
            "n_E_T": sum(int(p["e_t"]) for p in EV.load_catalogue()[0]),
            "n_E_HH_T": sum(int(p["e_hh_t"]) for p in EV.load_catalogue()[0]),
            "n_E_STRICT": sum(int(p["e_strict_full"]) for p in EV.load_catalogue()[0]),
        },
        "test_policy": "DEVELOPMENT_TEST",
        "test_use": "evaluation only; Test selected no checkpoint, dictionary, alpha, subset or reference",
        "exceptions": [],
        "note": ("CPU-only frozen evaluation. No neural network was trained or re-run. "
                 "The structure catalogue was frozen before labels/predictions were connected."),
    }
    TC.write_json(TC.ROUND_DIR / "run_manifest.json", manifest)
    print(json.dumps({
        "status": status,
        "wall_seconds": wall,
        "cpu_seconds": manifest["cpu_seconds"],
        "max_rss_kb": manifest["max_rss_kb"],
        "structure_hash": structure_freeze["structure_hash"],
        "pairs": result["n_pairs"],
        "E_T": manifest["counts"]["n_E_T"],
        "E_HH_T": manifest["counts"]["n_E_HH_T"],
        "E_STRICT": manifest["counts"]["n_E_STRICT"],
        "frozen_full_all_pass": manifest["frozen_full_checks"]["all_pass_1e-5"],
        "artifacts": len(artifacts),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
