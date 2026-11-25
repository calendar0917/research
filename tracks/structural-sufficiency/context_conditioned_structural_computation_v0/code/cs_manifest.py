"""Result manifest: every artefact of this round with its real path and SHA256.

Large checkpoints stay on the compute host (the repository convention); the
manifest records their absolute path there plus the local mirror when it exists,
so nothing is claimed without a hash.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import cs_common as C

C.setup_paths()

REMOTE_ROOT = "/home/hxy/cy/research"


def digest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": C.track_rel(path),
        "bytes": int(path.stat().st_size),
        "sha256": C.sha256_file(path),
    }


def build() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for dataset in C.DATASETS:
        for arm in C.ARM_ORDER:
            out_dir = C.RUNS_DIR / f"{dataset.replace('ogbg-', '')}_{arm}_seed{C.SEED}"
            if not out_dir.is_dir():
                continue
            entry: dict[str, Any] = {"local_dir": C.track_rel(out_dir)}
            run_json = out_dir / "run.json"
            if run_json.is_file():
                run = C.read_json(run_json)
                entry.update(
                    {
                        "status": run.get("status"),
                        "best_epoch": run.get("best_epoch"),
                        "best_valid_rmse_raw": run.get("best_valid_rmse_raw"),
                        "gpu_hours": run.get("gpu_hours"),
                        "checkpoint_best_sha256": run.get("checkpoint_best_sha256"),
                        "checkpoint_final_sha256": run.get("checkpoint_final_sha256"),
                        "curve_sha256": run.get("curve_sha256"),
                    }
                )
            entry["files"] = {
                name: digest(out_dir / name)
                for name in (
                    "run.json",
                    "train_curve.csv",
                    "best_checkpoint_eval.json",
                    "gate_history.json",
                    "training_order.json",
                    "predictions_best.npz",
                    "predictions_final.npz",
                    "predictions_last20.npz",
                )
            }
            entry["checkpoints_on_host"] = {
                name: {
                    "remote_path": f"{REMOTE_ROOT}/{C.track_rel(out_dir)}/{name}",
                    "local_mirror": digest(out_dir / name),
                }
                for name in ("checkpoint_best.pt", "checkpoint_final.pt")
            }
            runs[f"{dataset}:{arm}"] = entry
    analysis = {
        name: digest(C.ANALYSIS_DIR / name)
        for name in (
            "checks.json",
            "preflight_timing.json",
            "decomposition_trace.json",
            "mechanism_diagnostics.json",
            "structure_rules.json",
            "explanation_cases.json",
            "main_table.csv",
            "paired_comparisons.json",
            "strata.json",
            "verdicts.json",
        )
    }
    decomposition = {}
    root = C.RESULTS_DIR / "decomposition"
    if root.is_dir():
        for path in sorted(root.glob("*.npz")):
            decomposition[path.name] = digest(path)
    return {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "runs": runs,
        "analysis": analysis,
        "decomposition": decomposition,
        "note": (
            "checkpoints stay on the host; the local mirror is recorded when pulled. "
            "Every listed sha256 is computed from the real file bytes at manifest time."
        ),
    }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.RESULTS_DIR / "manifest.json"))
    args = parser.parse_args(argv)
    C.write_json(Path(args.out), build())
    C.log(f"manifest written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
