"""Timing preflight: measured train step + measured full three-split evaluation.

This is an engineering probe.  It runs a fixed small number of real training
steps on real batches (same collate/stager/model/optimiser/loss code path as the
formal trainer) and then a full eval-mode pass over Train, Valid and Test, so
the projected epoch cost includes the every-epoch evaluation requirement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch  # noqa: E402

import jt_common as C  # noqa: E402
import jt_models as M  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()
M.prepare_imports()

import cr_common as CC  # noqa: E402
import cr_data as D  # noqa: E402
import cr_train as T  # noqa: E402
import ds_common as DC  # noqa: E402
import ds_train as DT  # noqa: E402
import lr_data as LD  # noqa: E402
import pc_model as PCM  # noqa: E402


def probe_lipo(arm: str, device: torch.device, steps: int) -> dict[str, Any]:
    bundle = D.build_bundle("ogbg-mollipo")
    support = S.load_support(C.ANALYSIS / "support/lipo_support.npz")
    model, info = M.build_lipo_model(bundle, support, arm)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(T.RECIPE["peak_lr"]),
        betas=tuple(T.RECIPE["adam_betas"]),
        eps=float(T.RECIPE["adam_eps"]),
        weight_decay=float(T.RECIPE["weight_decay"]),
    )
    rows = np.arange(int(bundle.caches["train"]["n_molecules"]), dtype=np.int64)
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(rows)
    model.train()
    times = []
    for i in range(int(steps)):
        chunk = order[i * 128 : (i + 1) * 128]
        batch = bundle.collate(bundle.caches["train"], chunk)
        batch["y"] = torch.as_tensor(bundle.y_std["train"][chunk], dtype=torch.float32)
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), batch["y"].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(T.RECIPE["gradient_clip"]))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
    model.eval()
    t0 = time.perf_counter()
    metrics = {}
    with torch.no_grad():
        for split in C.SPLITS:
            out = T.evaluate(model, bundle, split, device)
            metrics[split] = {"rmse_raw": float(out["rmse_raw"]), "seconds": float(out["seconds"])}
    eval_seconds = float(time.perf_counter() - t0)
    step_mean = float(np.mean(times))
    steps_per_epoch = int(np.ceil(rows.size / 128))
    projected = steps_per_epoch * step_mean + eval_seconds
    return {
        "arm": arm,
        "family": "lipo",
        "parameters": int(info["parameters"]),
        "steps_measured": int(steps),
        "step_ms_mean": step_mean * 1e3,
        "eval_seconds_three_splits": eval_seconds,
        "eval_metrics": metrics,
        "steps_per_epoch": steps_per_epoch,
        "projected_epoch_seconds": projected,
        "epochs": int(C.LIPO_RECIPE["epochs"]),
        "projected_gpu_hours": projected * int(C.LIPO_RECIPE["epochs"]) / 3600.0,
        "device": str(device),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }


def probe_pcba(arms: list[str], device: torch.device, steps: int) -> list[dict[str, Any]]:
    task = LD.load_task()
    support = S.load_support(C.ANALYSIS / "support/pcba_support.npz")
    plan, plan_audit = DC.plan_for_seed(task, C.SEED)
    test_info = DC.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    test_labels = np.asarray(test_info["labels"], dtype=np.float32)
    out = []
    for arm in arms:
        model, info = M.build_pcba_model(task, support, arm)
        model = model.to(device)
        optimizer, _ = DT.build_optimizer(model)
        stager = DT.PinnedStager(device, depth=2)
        batches = plan[0]["batches"]
        model.train()
        times = []
        for rows in batches[: int(steps)]:
            t0 = time.perf_counter()
            batch = stager.batch(PCM.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            DT.train_step(model, optimizer, batch, y, global_step=0, clip=float(DC.RECIPE["gradient_clip"]), audit_new=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(float(time.perf_counter() - t0))
        stager.release()
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            reps = {
                "train": DC.evaluate_split(model, task, device, "train", per_task=False),
                "valid": DC.evaluate_split(model, task, device, "valid", per_task=False),
                "test": DC.evaluate_split(model, task, device, "test", y=test_labels, per_task=False),
            }
        eval_seconds = float(time.perf_counter() - t0)
        step_mean = float(np.mean(times))
        projected = int(DC.STEPS_PER_EPOCH) * step_mean + eval_seconds
        out.append(
            {
                "arm": arm,
                "family": "pcba",
                "parameters": int(info["parameters"]),
                "steps_measured": int(steps),
                "step_ms_mean": step_mean * 1e3,
                "eval_seconds_three_splits": eval_seconds,
                "eval_metrics": {s: {"ap": float(reps[s]["ap"]), "bce": float(reps[s]["bce"]), "seconds": float(reps[s]["seconds"])} for s in reps},
                "steps_per_epoch": int(DC.STEPS_PER_EPOCH),
                "projected_epoch_seconds": projected,
                "epochs": int(C.PCBA_RECIPE["horizon"]),
                "projected_gpu_hours": projected * int(C.PCBA_RECIPE["horizon"]) / 3600.0,
                "device": str(device),
                "plan_audit": plan_audit,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("lipo", "pcba"))
    parser.add_argument("--arms", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(C.ANALYSIS / "throughput_probe.json"))
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arms = [a for a in args.arms.split(",") if a] or (list(C.LIPO_ARMS) if args.family == "lipo" else list(C.PCBA_ARMS))
    result: dict[str, Any] = {
        "schema": "lcrc-support-joint-transfer-throughput-probe-v1",
        "round_id": C.ROUND_ID,
        "family": args.family,
        "device": str(device),
        "environment": C.environment(),
        "measured_utc": C.utcnow(),
    }
    if args.family == "lipo":
        result["units"] = [probe_lipo(arm, device, args.steps) for arm in arms]
    else:
        result["units"] = probe_pcba(arms, device, args.steps)
    result["total_projected_gpu_hours"] = float(sum(u["projected_gpu_hours"] for u in result["units"]))
    C.write_json(args.out, result)
    print(C.json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
