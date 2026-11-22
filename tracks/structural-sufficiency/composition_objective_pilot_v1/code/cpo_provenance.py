"""Collect execution provenance for the three formal runs.

Uses the local `rr` CLI for scheduler/process metadata and hashes the pulled
deliverables.  Read-only with respect to the runs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cpo_common as C

EXPERIMENTS = {
    "F_GRAPH": "CPO_F_GRAPH",
    "B_GRAPH": "CPO_B_GRAPH",
    "C_GRAPH": "CPO_C_GRAPH",
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    cfg = C.load_config()
    prov_dir = C.results_root() / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    repo = C.REPO_ROOT
    out: dict[str, Any] = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "local_repo": str(repo),
        "local_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                       check=False, text=True, capture_output=True).stdout.strip(),
        "local_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                           check=False, text=True, capture_output=True).stdout.strip()),
        "config_path": str(C.ROUND_DIR / "configs" / "cpo_v1.json"),
        "config_sha256": _sha256(C.ROUND_DIR / "configs" / "cpo_v1.json"),
        "protocol_sha256": _sha256(C.ROUND_DIR / "PROTOCOL.md"),
        "cache_path_used": None,
        "cache_content_sha256": None,
        "rr": {},
        "runs": {},
        "gpu_hours_total_new_training": 0.0,
        "budget_gpu_hours_max": float(cfg["budget"]["gpu_hours_max"]),
        "test_policy": "Test labels and Test predictions were never read in this round",
    }
    pack = C.load_pack(cfg, "lipo")
    out["cache_path_used"] = str(pack["cache_path"])
    out["cache_content_sha256"] = pack["cache_sha256"]
    for arm, exp in EXPERIMENTS.items():
        run_id = cfg["arms"][arm]["run_id"]
        run_dir = C.run_dir(run_id)
        man = C.read_json(run_dir / "manifest.json")
        status_raw = subprocess.run(
            ["rr", "status", exp, "--json"], cwd=repo, check=False,
            text=True, capture_output=True, timeout=120,
        ).stdout
        try:
            status = json.loads(status_raw)
        except Exception:
            status = {"raw": status_raw}
        status_path = prov_dir / f"rr_status_{exp}.json"
        C.write_json(status_path, status)
        log_path = prov_dir / f"rr_stdout_{exp}.log"
        log = subprocess.run(
            ["rr", "logs", exp], cwd=repo, check=False,
            text=True, capture_output=True, timeout=120,
        ).stdout
        log_path.write_text(log)
        out["rr"][exp] = {
            "experiment": exp,
            "rr_run_id": status.get("run_id") if isinstance(status, dict) else None,
            "state": status.get("state") if isinstance(status, dict) else None,
            "exit_code": status.get("exit_code") if isinstance(status, dict) else None,
            "pool": status.get("pool") if isinstance(status, dict) else None,
            "cuda_visible_devices": status.get("cuda_visible_devices") if isinstance(status, dict) else None,
            "git_commit": status.get("commit") if isinstance(status, dict) else None,
            "git_dirty": status.get("dirty") if isinstance(status, dict) else None,
            "started_at": status.get("started_at") if isinstance(status, dict) else None,
            "completed_at": status.get("completed_at") if isinstance(status, dict) else None,
            "command": status.get("meta", {}).get("command") if isinstance(status, dict) else None,
            "result_paths": status.get("meta", {}).get("result_paths") if isinstance(status, dict) else None,
            "status_json": str(status_path),
            "stdout_log": str(log_path),
        }
        files = {}
        for name in ("config.json", "manifest.json", "curves.csv", "train.log",
                     "mechanism_diagnostics.json", "predictions_best.npz",
                     "predictions_final.npz", "state_best.pt", "state_final.pt"):
            p = run_dir / name
            files[name] = {"sha256": _sha256(p), "bytes": _size(p)}
        out["runs"][arm] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "arm_family": man["family"],
            "model_arm": man["arm"],
            "main_loss": man["main_loss"],
            "best": man["best"],
            "epochs": man["epochs"],
            "seed": man["seed"],
            "wall_seconds": man["wall_seconds"],
            "gpu_hours": man["gpu_hours"],
            "signature": man["signature"],
            "scaler": man["scaler"],
            "init_hashes": man["init_hashes"],
            "parameter_ledger": man["params"],
            "environment": man["environment"],
            "argv": man["argv"],
            "files": files,
        }
        out["gpu_hours_total_new_training"] += float(man["gpu_hours"])
    C.write_json(C.results_root() / "provenance.json", out)
    print(f"wrote {C.results_root() / 'provenance.json'}")
    print(f"GPU-hours total: {out['gpu_hours_total_new_training']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())