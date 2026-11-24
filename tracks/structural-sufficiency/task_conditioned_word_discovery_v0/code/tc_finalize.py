"""Post-run provenance manifest (frozen pre-run manifest is left untouched)."""

from __future__ import annotations

import json
from pathlib import Path

import tc_common as C


def sha(path: Path) -> str:
    return C.sha256_file(path)


def main() -> None:
    evaluation = C.read_json(C.ANALYSIS_DIR / "evaluation.json")
    runs = {}
    total_wall = 0.0
    total_gpu = 0.0
    for run_dir in sorted((C.RESULTS_DIR).iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
            continue
        m = C.read_json(run_dir / "manifest.json")
        total_wall += float(m["wall_seconds"])
        total_gpu += float(m["gpu_hours"])
        runs[run_dir.name] = {
            "run_id": m["run_id"], "arm": m["arm"], "split": m["split"],
            "seed": m["seed"], "commit": m["git_commit"],
            "best": m["best"], "wall_seconds": m["wall_seconds"],
            "gpu_hours": m["gpu_hours"], "params": m["params"],
            "manifest_sha256": sha(run_dir / "manifest.json"),
            "curves_sha256": sha(run_dir / "curves.csv"),
            "predictions_dev_sha256": sha(run_dir / "predictions_best_dev.csv"),
            "state_best_sha256": sha(run_dir / "state_best.pt"),
        }
    payload = {
        "schema": "tc-postrun-manifest/v1",
        "protocol_id": C.load_config()["protocol_id"],
        "frozen_utc": C.now_utc(),
        "runs": runs,
        "totals": {"wall_seconds": total_wall, "gpu_hours": total_gpu,
                   "n_runs": len(runs)},
        "verdict": evaluation["verdict"],
        "checks": C.read_json(C.ANALYSIS_DIR / "checks.json"),
        "config_sha256": sha(C.CONFIG_PATH),
        "code_sha256": {p.name: sha(p) for p in sorted(C.CODE_DIR.glob("*.py"))},
        "analysis_sha256": {p.name: sha(p) for p in sorted(C.ANALYSIS_DIR.glob("*"))
                            if p.is_file() and p.name != "postrun_manifest.json"},
        "note": "hashes cover the exact artefacts used by REPORT.md",
    }
    C.write_json(C.ANALYSIS_DIR / "postrun_manifest.json", payload)
    print(json.dumps({"runs": len(runs), "totals": payload["totals"],
                      "verdict": payload["verdict"]}, indent=1))


if __name__ == "__main__":
    main()