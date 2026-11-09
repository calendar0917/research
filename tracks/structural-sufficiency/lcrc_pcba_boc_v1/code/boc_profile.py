"""Real-batch GPU profiling before the formal BOC runs.

Measures the *same* code path as training: inherited order plan, collate,
``PinnedStager``, ``bs_train.train_step`` (AdamW + clip + masked BCE), and the
BOC forward.  The W2 reference is measured with the historical collate.

The output includes a projected 40-epoch GPU-hour estimate.  Nothing here
trains a formal model or reads an official Test label.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

import boc_common as C  # noqa: E402
import boc_model as M  # noqa: E402
import boc_structures as S  # noqa: E402
import bs_model as BSM  # noqa: E402
import bs_train as BT  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _batch(arm: str, task: Any, struct: Mapping[str, Any], rows: Sequence[int]):
    if arm == "W2":
        return C.BC.PC.collate(task.cache, rows)
    return S.collate_boc(task, struct, rows)


def _build(arm: str, task: Any, device: torch.device):
    if arm == "W2":
        return BSM.build_model("W2", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    return M.build_model(arm, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)


def profile_arm(
    arm: str,
    task: Any,
    struct: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    model, audit = _build(arm, task, device)
    optimizer, _ = BSM.make_optimizer(model)
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    batches = [np.asarray(b, dtype=np.int64) for b in plan[0]["batches"]]
    take = min(int(warmup) + int(steps), len(batches))
    collate_times: list[float] = []
    step_times: list[float] = []
    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for i in range(take):
        rows = batches[i]
        t0 = time.perf_counter()
        batch = _batch(arm, task, struct, rows.tolist())
        collate_times.append(float(time.perf_counter() - t0))
        batch = stager.batch(batch)
        y = stager.labels(task.labels[rows])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        out = BT.train_step(model, optimizer, batch, y, global_step=i, clip=clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = float(time.perf_counter() - t1)
        losses.append(float(out["loss"]))
        if i >= int(warmup):
            step_times.append(dt)
    stager.release()
    # Validation inference cost per row on a fixed 256-row slice.
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)[:256]
    t0 = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        batch = _batch(arm, task, struct, valid_rows.tolist())
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    valid_seconds = float(time.perf_counter() - t0)
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    arr = np.asarray(step_times, dtype=np.float64)
    step_mean = float(arr.mean()) if arr.size else float("nan")
    epoch_train = float(step_mean * C.STEPS_PER_EPOCH)
    val_per_row = float(valid_seconds / 256.0)
    epoch_valid = float(val_per_row * C.N_VALID)
    projected = float((epoch_train + epoch_valid) * C.TOTAL_EPOCHS / 3600.0)
    return {
        "arm": arm,
        "parameters": int(audit.get("total_parameters", C.parameter_count(model))),
        "warmup_steps": int(warmup),
        "measured_steps": int(len(step_times)),
        "collate_ms_mean": float(np.mean(collate_times) * 1e3) if collate_times else float("nan"),
        "collate_ms_p90": float(np.percentile(collate_times, 90) * 1e3) if collate_times else float("nan"),
        "step_ms_mean": float(step_mean * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
        "steps_per_second": float(1.0 / step_mean) if step_mean > 0 else float("nan"),
        "loss_first": float(losses[warmup]) if len(losses) > warmup else None,
        "loss_last": float(losses[-1]) if losses else None,
        "valid_ms_per_256_rows": float(valid_seconds * 1e3),
        "projected_epoch_train_seconds": epoch_train,
        "projected_epoch_valid_seconds": epoch_valid,
        "projected_40_epoch_gpu_hours": projected,
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "model_audit_summary": {
            "base_parameters": audit.get("base_parameters"),
            "new_parameters": audit.get("new_parameters"),
            "parameter_inventory": audit.get("parameter_inventory"),
        },
        "batch_shape": {
            "atoms": int(batch["atom_feat"].shape[0]),
            "bonds": int(batch["bond_feat"].shape[0]),
            "n_nb": int(batch.get("boc_n_nb", 0)) if isinstance(batch.get("boc_n_nb", 0), int) else None,
            "n_pairs": int(batch.get("boc_n_pairs", 0)) if isinstance(batch.get("boc_n_pairs", 0), int) else None,
            "n_rings": int(batch.get("boc_n_rings", 0)) if isinstance(batch.get("boc_n_rings", 0), int) else None,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", default=["FULL", "BOND", "W2"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--struct-cache", default=str(C.STRUCT_CACHE_DIR))
    parser.add_argument("--out", default=str(C.ANALYSIS / "profile.json"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    task = C.BC.SD.load_task()
    struct = S.load_cache(args.struct_cache)
    if int(struct["meta"]["n_rows"]) != C.N_MOLECULES:
        raise RuntimeError("profile requires the full structure cache")
    plan = C.BC.SD.make_plan40(task, seed=C.SEED)
    if plan[0]["global_plan_sha256"] != C.W2["global_plan_sha256"]:
        raise RuntimeError("profile order plan does not match the frozen W2 plan")
    try:
        arms = {arm: profile_arm(arm, task, struct, plan, device, warmup=args.warmup, steps=args.steps) for arm in args.arms}
        out = {
            "schema": C.SCHEMA_PROFILE,
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "git_commit": C.git_commit(),
            "git_state": C.git_state(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "environment": C.environment(),
            "host_facts": C.host_facts(),
            "gpu_occupancy": C.gpu_occupancy(),
            "structure_cache": S.cache_identity(struct),
            "order_plan_sha256": plan[0]["global_plan_sha256"],
            "arms": arms,
            "projection_note": (
                "epoch_train = measured step mean * 2738; epoch_valid = measured inference "
                "cost per row * 43,793; total = 40 * (train + valid) / 3600.  No test rows."
            ),
        }
        C.write_json(args.out, out)
        print(json.dumps(out, indent=1, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            Path(args.out),
            {
                "schema": C.SCHEMA_PROFILE,
                "status": "FAILED",
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
