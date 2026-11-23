"""Copy the authoritative rr run metadata into each result directory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import sr_common as C


def main() -> int:
    for run_dir in sorted(C.RESULTS_DIR.iterdir()) if C.RESULTS_DIR.is_dir() else []:
        if not run_dir.is_dir():
            continue
        cp = subprocess.run(["rr", "status", run_dir.name, "--json"],
                            cwd=C.REPO_ROOT, text=True, capture_output=True)
        if cp.returncode != 0:
            continue
        try:
            info = json.loads(cp.stdout)
        except json.JSONDecodeError:
            continue
        payload = {k: info.get(k) for k in (
            "run_id", "experiment", "host", "pool", "state", "commit", "branch",
            "dirty", "created_at", "started_at", "completed_at", "exit_code",
            "cuda_visible_devices", "gpu_names", "driver_version",
            "torch_version", "torch_cuda", "failure_reason")}
        (run_dir / "rr_provenance.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(run_dir.name, payload.get("state"), payload.get("commit"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())