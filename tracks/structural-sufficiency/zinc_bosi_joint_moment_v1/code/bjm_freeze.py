"""Freeze step: verify each finished run, then hash its checkpoints and valid reads.

Run this **before** looking at any comparison.  For every requested run it

* re-reads ``train.json`` and the recorded recipe / parameter count,
* rebuilds the model from the checkpoint on CPU and recomputes the full official
  valid MAE, refusing to continue if it disagrees with ``train.json`` by more
  than 1e-6 (numerical tolerance, not result selection),
* records checkpoint / prediction sha256 and the GPU-hour ledger.

``analysis/freeze.json`` is written once; a second call only re-verifies.

Usage:
    python .../bjm_freeze.py --runs B_s0 JM_s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bjm_common as BC  # noqa: E402
import bjm_models as BJM  # noqa: E402

TOL = 1.0e-6


def verify_run(tag: str, norm, vcache, vrel, y_valid) -> dict:
    d = BC.RUNS / tag
    info = BC.read_json(d / "train.json")
    arm = info["arm"]
    seed = int(info["seed"])
    model, _ = BJM.build_model(norm, arm, seed, "cpu")
    out = {
        "run_dir": BC.rel_or_abs(d),
        "arm": arm,
        "seed": seed,
        "parameters": info["parameters"],
        "recipe": info["recipe"],
        "total_updates": info["total_updates"],
        "best": info["best"],
        "W20": info["W20"],
        "T100": info["T100"],
        "gpu_hours": info["gpu_hours"],
        "initial_state_sha256": info["initial"]["initial_state_sha256"],
        "initial_common_state_sha256": info["initial"]["initial_common_state_sha256"],
        "first_batch_rows_sha256": info["first_batch_rows_sha256"],
        "data": info["data"],
        "checkpoints": {},
        "official_test_loaded": False,
    }
    for ckpt, expected in (("best", info["reads"]["best"]["valid_mae"]),
                           ("final", info["reads"]["final"]["valid_mae"])):
        path = d / "final.pt" if ckpt == "final" else d / "best.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v for k, v in blob["state_dict"].items()})
        pred = BC.predict(model, vcache, vrel, np.arange(len(y_valid)),
                          device="cpu", batch_size=256)
        recomputed = BC.mae(pred, y_valid)
        npz = d / f"valid_{ckpt}.npz"
        stored = np.load(npz)
        stored_pred = np.asarray(stored["pred"], dtype=np.float64)
        out["checkpoints"][ckpt] = {
            "checkpoint": BC.rel_or_abs(path),
            "checkpoint_sha256": BC.sha256_file(path),
            "epoch": int(blob.get("epoch", -1)),
            "valid_mae_train_json": float(expected),
            "valid_mae_recomputed_cpu": float(recomputed),
            "abs_diff": float(abs(recomputed - expected)),
            "prediction_file": BC.rel_or_abs(npz),
            "prediction_sha256": BC.sha256_file(npz),
            "stored_pred_max_abs_diff_vs_recomputed": float(
                np.abs(stored_pred - pred.astype(np.float64)).max()),
            "train_mae_eval_mode": info["reads"][ckpt]["train_mae_eval_mode"],
            "verified": bool(abs(recomputed - expected) <= TOL),
        }
        if not out["checkpoints"][ckpt]["verified"]:
            raise RuntimeError(f"{tag}/{ckpt}: recomputed valid MAE {recomputed} != "
                               f"train.json {expected}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    out_path = Path(args.out) if args.out else BC.ANALYSIS / "freeze.json"
    norm = BC.load_norm()
    vcache = BC.load_cache("valid")
    vrel = BC.load_rel("valid")
    y_valid = np.asarray(vcache["labels"], dtype=np.float64)

    runs = {}
    for tag in args.runs:
        runs[tag] = verify_run(tag, norm, vcache, vrel, y_valid)
        BC.log(f"[bjm-freeze] {tag}: arm={runs[tag]['arm']} seed={runs[tag]['seed']} "
               f"params={runs[tag]['parameters']} "
               f"best={runs[tag]['best']['valid_mae']:.6f}@{runs[tag]['best']['epoch']} "
               f"ckpt={runs[tag]['checkpoints']['best']['checkpoint_sha256'][:8]}")

    payload = {
        "round": BC.ROUND_ID,
        "written": BC.now(),
        "config": BC.rel_or_abs(BC.CONFIG),
        "config_sha256": BC.sha256_file(BC.CONFIG),
        "cpu_gates_sha256": BC.sha256_file(BC.ANALYSIS / "cpu_checks.json"),
        "official_valid_loaded": True,
        "official_valid_role": "development evaluation and checkpoint selection",
        "official_test_loaded": False,
        "gpu_hours_total": float(sum(r["gpu_hours"] for r in runs.values())),
        "runs": runs,
    }
    if out_path.exists() and not args.force:
        prev = BC.read_json(out_path)
        same = (prev.get("runs") == payload["runs"])
        BC.log(f"[bjm-freeze] {BC.rel_or_abs(out_path)} exists; "
               f"{'identical' if same else 'DIFFERENT'} — refusing to rewrite")
        return 0 if same else 2
    BC.write_json(out_path, payload)
    BC.log(f"[bjm-freeze] wrote {BC.rel_or_abs(out_path)} "
           f"(gpu_h={payload['gpu_hours_total']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
