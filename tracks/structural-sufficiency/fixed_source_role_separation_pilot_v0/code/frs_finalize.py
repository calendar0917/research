"""Repair/complete one FSR run after a mechanism-only crash (no retraining).

Recomputes the light mechanism diagnostics from the saved best checkpoint and
rewrites ``run.json`` from the artifacts already on disk.  Used once for the
RSSC arm, whose 12-epoch training finished and then crashed inside the
mechanism diagnostic (numpy/torch bug, since fixed).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import frs_common as FC  # noqa: E402
import frs_train as T  # noqa: E402
import frs_mechanism as MECH  # noqa: E402
import torch  # noqa: E402
import ds_common as C  # noqa: E402
import frs_model as FM  # noqa: E402

FIELDS = T.CURVE_FIELDS


def read_curve(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--started-utc", default="")
    ap.add_argument("--finished-utc", default="")
    ap.add_argument("--mechanism-rows", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out_dir)
    device = torch.device(args.device)
    rows = read_curve(out / "train_curve.csv")
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 completed epochs, found {len(rows)}")
    best_eval = FC.read_json(out / "best_checkpoint_eval.json")
    best_payload = torch.load(out / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    last_payload = torch.load(out / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    model_audit = dict(best_payload["model_audit"])
    task = C.SD.load_task()
    plan, plan_audit = C.plan_for_seed(task, FC.SEED)
    model = FM.FRSModel(task.spec, task.norm, C.N_TASKS, dict(C.BC.config_for_scale(2)), arm=args.arm)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    mech = MECH.run(model=model, task=task, device=device, arm=args.arm,
                    best_checkpoint=out / "checkpoint_best.pt", model_audit=model_audit,
                    n_rows=int(args.mechanism_rows))
    mech["best_epoch"] = int(best_eval["epoch"])
    mech["completed"] = True
    mech["repair_note"] = "mechanism computed by frs_finalize.py from the saved best checkpoint"
    FC.write_json(out / "mechanism.json", mech)
    valid = [(int(r["epoch"]), float(r["valid_ap"]), float(r["valid_bce"])) for r in rows]
    best_epoch, best_ap = max(valid, key=lambda x: (x[1], -x[0]))[0:2]
    last3 = valid[-3:]
    cumulative = float(rows[-1]["cumulative_seconds"])
    mech_seconds = 0.0
    gpu_seconds = cumulative + float(best_eval["seconds"]) + mech_seconds
    endpoints = {k: rows[-1].get(k) for k in ("valid_ap", "valid_bce")}
    run = {
        "schema": FC.SCHEMA_RUN,
        "round_id": FC.ROUND_ID,
        "protocol_id": FC.PROTOCOL_ID,
        "arm": args.arm,
        "base_computation": "FIXED1",
        "role_branch": "low-rank correction uses z + m (same input as fc1)" if args.arm == "TIED"
        else "low-rank correction uses z - m (role-separated input)",
        "seed": FC.SEED,
        "horizon": 12,
        "status": "COMPLETED",
        "stop_reason": "COMPLETED",
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "parameter_audit": model_audit["parameters"],
        "rank": int(FC.RANK),
        "epochs_run": int(len(rows)),
        "global_steps_run": int(rows[-1]["global_step"]),
        "eval_protocol": T.evaluation_protocol(12),
        "task_counts": C.SD.task_counts(task),
        "model_audit": model_audit,
        "plan_identity": plan_audit,
        "batch_plan": best_payload["batch_plan"],
        "best_epoch": int(best_epoch),
        "best_valid_ap": float(best_ap),
        "best_valid_bce": float(best_eval["valid"]["bce"]),
        "horizon_endpoint": {
            "epoch": int(len(rows)),
            "valid_ap": float(endpoints["valid_ap"]),
            "valid_bce": float(endpoints["valid_bce"]),
            "train_ap": (None if rows[-1].get("train_ap") in (None, "") else float(rows[-1]["train_ap"])),
            "train_bce": (None if rows[-1].get("train_bce") in (None, "") else float(rows[-1]["train_bce"])),
        },
        "last3_mean_valid_ap": float(np.mean([v[1] for v in last3])),
        "last3_epochs": [v[0] for v in last3],
        "best_checkpoint_eval": best_eval,
        "best_checkpoint": FC.C.track_rel(out / "checkpoint_best.pt"),
        "best_checkpoint_sha256": FC.sha256_file(out / "checkpoint_best.pt"),
        "last_checkpoint": FC.C.track_rel(out / "checkpoint_last.pt"),
        "last_checkpoint_sha256": FC.sha256_file(out / "checkpoint_last.pt"),
        "curve": FC.C.track_rel(out / "train_curve.csv"),
        "curve_sha256": FC.sha256_file(out / "train_curve.csv"),
        "per_task_valid": FC.C.track_rel(out / "per_task_valid.csv"),
        "per_task_valid_sha256": FC.sha256_file(out / "per_task_valid.csv"),
        "structural_usage": FC.C.track_rel(out / "structural_usage.json"),
        "mechanism": FC.C.track_rel(out / "mechanism.json"),
        "recipe": C.recipe("FIXED1", FC.SEED, 12),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "tf32": {
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "grad_clip": float(C.RECIPE["gradient_clip"]),
        "gpu_seconds": float(gpu_seconds),
        "gpu_hours": float(gpu_seconds) / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in rows)),
        "eval_seconds": float(sum(float(r.get("epoch_seconds") or 0.0) - float(r.get("epoch_train_seconds") or 0.0) for r in rows)),
        "environment": model_audit.get("environment") or C.environment(),
        "host_facts": C.host_facts(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "started_utc": args.started_utc or None,
        "finished_utc": args.finished_utc or C.utcnow(),
        "test_role": FC.TEST_ROLE,
        "test_used_for_selection": False,
        "test_scored": False,
        "inheritance": {
            "source_round": FC.HIST["source_round"],
            "FIXED1_best_valid_ap_25": FC.HIST["best_valid_ap_25"],
            "W2_best_valid_ap_40": FC.HIST["w2_best_valid_ap_40"],
            "historical_same_horizon_reference": "results/runs/FIXED1_seed0 (25 epochs, real curve)",
        },
        "repair_note": (
            "run.json reconstructed by frs_finalize.py: the 12 training epochs, the curve, "
            "best/last checkpoints and best-checkpoint evaluation were produced by the "
            "original run; it crashed only inside the post-training mechanism diagnostic."
        ),
    }
    FC.write_json(out / "run.json", run)
    FC.log(f"finalized {args.arm}: best_epoch {best_epoch} best_valid {best_ap:.6f} "
           f"gpu_seconds {gpu_seconds:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
