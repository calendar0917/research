"""Freeze the per-run artefacts of one stage of ``zinc_adaptive_arch_search_v1``.

Writes ``analysis/freeze_<stage>.json``: for every completed run of the stage
the parameter count, GPU hours, selection metrics, checkpoint and prediction
hashes, the training-record hash, and the hashes of the frozen config / CPU
gates / reference run.  ``aas_analyze.py`` refuses to analyse a stage whose
config, gates or run artefacts changed after this file was written.

Usage:
    python .../aas_freeze.py --stage a --tags A1_s0,A0_s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aas_common as AC  # noqa: E402

#: frozen matched baseline (BOSI-B, official 10k/1k, seed 0).  Lives in the
#: BOSI-JM round; only its record + per-molecule prediction file are read.
REFERENCE_ROOT = AC.SUFF / "zinc_bosi_joint_moment_v1" / "runs"


def run_record(tag: str) -> dict:
    d = AC.RUNS / tag
    rec = AC.read_json(d / "train.json")
    preds = {}
    for ckpt in ("best", "final"):
        blob = np.load(d / f"valid_{ckpt}.npz")
        pred = np.asarray(blob["pred"], dtype=np.float64)
        preds[ckpt] = {
            "valid_mae": float(np.mean(np.abs(pred - np.asarray(
                AC.load_cache("valid")["labels"], dtype=np.float64)))),
            "prediction_sha256": AC.sha256_array(pred),
            "npz_sha256": AC.sha256_file(d / f"valid_{ckpt}.npz"),
            "checkpoint_sha256": AC.sha256_file(d / f"{ckpt}.pt"),
            "train_mae_eval_mode": rec["reads"][ckpt]["train_mae_eval_mode"],
        }
    return {
        "arm": rec["arm"],
        "seed": int(rec["seed"]),
        "parameters": int(rec["parameters"]),
        "gpu_hours": float(rec["gpu_hours"]),
        "total_updates": int(rec["total_updates"]),
        "best": rec["best"],
        "W20": rec["W20"],
        "T100": rec["T100"],
        "final_valid": rec["reads"]["final"]["valid_mae"],
        "initial_state_sha256": rec["initial"]["initial_state_sha256"],
        "checkpoints": preds,
        "train_json_sha256": AC.sha256_file(d / "train.json"),
        "run_dir": AC.rel_or_abs(d),
    }


def reference_record() -> dict:
    d = REFERENCE_ROOT / "B_s0"
    rec = AC.read_json(d / "train.json")
    blob = np.load(d / "valid_best.npz")
    pred = np.asarray(blob["pred"], dtype=np.float64)
    y = np.asarray(AC.load_cache("valid")["labels"], dtype=np.float64)
    return {
        "arm": "B", "seed": 0, "parameters": int(rec["parameters"]),
        "gpu_hours": float(rec["gpu_hours"]), "total_updates": int(rec["total_updates"]),
        "best_valid": float(np.mean(np.abs(pred - y))),
        "recorded_best_valid": rec["best"]["valid_mae"],
        "best_epoch": rec["best"]["epoch"], "W20": rec["W20"], "T100": rec["T100"],
        "final_valid": rec["reads"]["final"]["valid_mae"],
        "train_at_best": rec["reads"]["best"]["train_mae_eval_mode"],
        "train_at_final": rec["reads"]["final"]["train_mae_eval_mode"],
        "prediction_sha256": AC.sha256_array(pred),
        "train_json_sha256": AC.sha256_file(d / "train.json"),
        "run_dir": AC.rel_or_abs(d),
        "role": "matched frozen baseline: same protocol/seed/recipe, different body",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    runs = {}
    for tag in tags:
        if not (AC.RUNS / tag / "train.json").exists():
            raise SystemExit(f"missing train.json for {tag}")
        runs[tag] = run_record(tag)
    payload = {
        "round": AC.ROUND_ID,
        "stage": args.stage,
        "written": AC.now(),
        "config_sha256": AC.sha256_file(AC.CONFIG),
        "cpu_gates_sha256": AC.sha256_file(AC.ANALYSIS / "cpu_checks.json"),
        "plan_sha256": AC.sha256_file(AC.ROUND / "PLAN.md"),
        "runs": runs,
        "reference": reference_record(),
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    out = Path(args.out) if args.out else AC.ANALYSIS / f"freeze_{args.stage}.json"
    AC.write_json(out, payload)
    for tag, r in runs.items():
        AC.log(f"[aas-freeze {args.stage}] {tag}: best={r['best']['valid_mae']:.6f} "
               f"@ep{r['best']['epoch']} W20={r['W20']:.6f} T100={r['T100']:.6f} "
               f"final={r['final_valid']:.6f} params={r['parameters']} "
               f"gpu_h={r['gpu_hours']:.3f}")
    AC.log(f"[aas-freeze {args.stage}] -> {AC.rel_or_abs(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
