"""Run the full CPU-only post-hoc pipeline in a fixed order."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "ph_integrity.py",
    "ph_p0.py",
    "ph_molhiv.py",
    "ph_local_edit.py",
    "ph_aux.py",
    "ph_manifest.py",
]


def main() -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    records = []
    for name in SCRIPTS:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(ROUND / "code" / name)],
            cwd=ROUND,
            env=env,
            capture_output=False,
            text=True,
        )
        records.append(
            {
                "script": name,
                "seconds": float(time.perf_counter() - t0),
                "returncode": int(proc.returncode),
            }
        )
        if proc.returncode != 0:
            break
    total = float(time.perf_counter() - started)
    payload = {
        "schema": "lcrc-posthoc-runtime-v1",
        "round_id": "lcrc-posthoc-diagnosis-v1",
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "gpu_used": False,
        "total_seconds": total,
        "scripts": records,
    }
    (ROUND / "analysis").mkdir(parents=True, exist_ok=True)
    (ROUND / "analysis/runtime.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
