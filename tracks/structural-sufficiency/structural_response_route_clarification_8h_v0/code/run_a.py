#!/usr/bin/env python
"""Phase A: four from-scratch FGCR arms on official Lipo Train3360 -> Valid420.

    A_IS  H_IND  ETA_STAR      A_IJ  H_IND  ETA_FULL
    A_FS  H_FULL ETA_STAR      A_FJ  H_FULL ETA_FULL

Identical parameter layout, identical random initial ``state_dict``, identical
batch plan and identical per-epoch molecule order.  No historical checkpoint is
fine-tuned and no early stopping is applied.

    uv run python .../code/run_a.py --arms A_IS,A_IJ --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sr_common as C
import sr_context as SX
import sr_train as ST

ARMS = {
    "A_IS": ("IND", "STAR"),
    "A_IJ": ("IND", "FULL"),
    "A_FS": ("FULL", "STAR"),
    "A_FJ": ("FULL", "FULL"),
}
EPOCHS = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="A_IS,A_IJ,A_FS,A_FJ")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-root", default=str(C.RESULTS_DIR / "A"))
    parser.add_argument("--limit-batches", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    C.seed_everything(C.TRAIN_SEED)

    log_path = Path(args.out_root) / "phaseA.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    env = C.environment_snapshot()
    log(f"phase A start device={device} arms={args.arms} env={json.dumps(C.jsonable(env))}")

    t0 = time.perf_counter()
    ctx = SX.build_lipo(include_valid=True)
    log(f"lipoo context plan={json.dumps(C.jsonable(ctx.plan_stats))} "
        f"plan_seconds={time.perf_counter() - t0:.1f}")
    log(f"data audit={json.dumps(C.jsonable({k: v for k, v in ctx.audit.items() if k != 'official_split_sizes'}))[:600]}")

    train_rows = np.asarray(ctx.split_rows["train"], dtype=np.int64)
    valid_rows = np.asarray(ctx.split_rows["valid"], dtype=np.int64)
    order = SX.batch_order(train_rows.size, args.epochs)
    init_state = SX.fgcr_init_state(ctx)
    init_hash = C.sha256_array(np.concatenate(
        [v.detach().cpu().numpy().reshape(-1).astype(np.float64)
         for _, v in sorted(init_state.items())]))
    log(f"shared init state hash={init_hash}")

    summaries = {}
    for arm_id in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm_id not in ARMS:
            raise SystemExit(f"unknown arm {arm_id}")
        h_mode, eta_mode = ARMS[arm_id]
        out_dir = Path(args.out_root) / arm_id
        if (out_dir / "summary.json").is_file():
            log(f"{arm_id}: already complete, skipping")
            summaries[arm_id] = C.read_json(out_dir / "summary.json")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"{arm_id}: H={h_mode} eta={eta_mode} start")

        eval_curve = []
        best = {"rmse": float("inf"), "epoch": -1, "state": None,
                "train": None, "valid": None}
        arm_holder = {}

        # build once so the Arm's extra dict can carry the evaluation label vector
        arm = SX.make_fgcr_arm(ctx, h_mode=h_mode, eta_mode=eta_mode,
                               init_state=init_state, device=device,
                               meta={"arm_id": arm_id})
        arm.extra["eval_labels_raw"] = ctx.bundle.labels[valid_rows].astype(np.float64)
        arm_holder["arm"] = arm

        # the frozen loop selects on the eval rows; A also needs full Train eval
        result = _train_with_train_eval(
            arm=arm, ctx=ctx, train_rows=train_rows, valid_rows=valid_rows,
            order=order, epochs=args.epochs, out_dir=out_dir, log=log,
            limit_batches=args.limit_batches)
        summary = dict(result["summary"])
        summary.update({
            "arm_id": arm_id, "h_mode": h_mode, "eta_mode": eta_mode,
            "init_hash": init_hash, "device": device,
            "gpu_name": (torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"),
            "peak_vram_gb": (float(torch.cuda.max_memory_allocated() / 1024 ** 3)
                             if device.startswith("cuda") else 0.0),
            "environment": env, "commit": C.git_commit(), "dirty": C.git_dirty(),
            "finished_at": C.utc_stamp(),
            "plan_stats": ctx.plan_stats,
        })
        C.write_json(out_dir / "summary.json", summary)
        np.savez_compressed(out_dir / "predictions.npz",
                            **{k: np.asarray(v) for k, v in result["predictions"].items()})
        C.write_json(out_dir / "curves.json", {"curves": result["curves"]})
        summaries[arm_id] = summary
        log(f"{arm_id}: done best_epoch={summary['best_epoch']} "
            f"valid={summary['best_valid_rmse']:.6f} seconds={summary['seconds']:.0f}")
        C.ledger_append({
            "phase": "A", "unit": arm_id, "device": device,
            "gpu_hours": summary["seconds"] / 3600.0,
            "seconds": summary["seconds"], "started_at": env.get("started_at"),
            "finished_at": C.utc_stamp(), "peak_vram_gb": summary["peak_vram_gb"],
            "status": "completed",
        })

    C.write_json(Path(args.out_root) / f"summary_{int(time.time())}.json", summaries)
    log("phase A finished")
    handle.close()
    return 0


def _train_with_train_eval(*, arm, ctx, train_rows, valid_rows, order, epochs,
                           out_dir, log, limit_batches: int = 0):
    """The frozen loop plus a full Train/Valid evaluation every epoch."""
    recipe = arm.recipe
    model = arm.model.to(arm.device)
    batch_size = int(recipe["batch_molecules"])
    eval_batch = int(recipe["eval_batch_molecules"])
    steps_per_epoch = (train_rows.size + batch_size - 1) // batch_size
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["peak_lr"]),
        betas=tuple(float(b) for b in recipe.get("betas", (0.9, 0.999))),
        eps=float(recipe.get("eps", 1e-8)),
        weight_decay=float(recipe["weight_decay"]))
    scaler = ctx.scaler
    grad_clip = float(recipe["grad_clip"])
    y_train = ctx.bundle.labels[train_rows].astype(np.float64)
    y_valid = ctx.bundle.labels[valid_rows].astype(np.float64)

    curves = []
    best = {"rmse": float("inf"), "epoch": -1, "state": None,
            "train": None, "valid": None}
    final = {"state": None, "train": None, "valid": None}
    step = 0
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        t_epoch = time.perf_counter()
        model.train()
        rows_epoch = train_rows[order[epoch - 1]]
        losses = []
        for start in range(0, rows_epoch.size, batch_size):
            chunk = rows_epoch[start:start + batch_size]
            for group in optimizer.param_groups:
                group["lr"] = float(arm.lr_at(step, steps_per_epoch, int(epochs)))
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = arm.base_loss(chunk)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_std = arm.predict(train_rows)
        valid_std = arm.predict(valid_rows)
        train_raw = train_std * float(scaler["std"]) + float(scaler["mean"])
        valid_raw = valid_std * float(scaler["std"]) + float(scaler["mean"])
        rmse_train = C.rmse(train_raw, y_train)
        rmse_valid = C.rmse(valid_raw, y_valid)
        row = {
            "epoch": epoch, "family": arm.family,
            "h_mode": arm.extra["h_mode"], "eta_mode": arm.extra["eta_mode"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)),
            "train_rmse": rmse_train, "train_mae": C.mae(train_raw, y_train),
            "valid_rmse": rmse_valid, "valid_mae": C.mae(valid_raw, y_valid),
            "train_rows_hash": C.sha256_array(rows_epoch),
            "epoch_seconds": time.perf_counter() - t_epoch,
            "peak_vram_gb": (float(torch.cuda.max_memory_allocated() / 1024 ** 3)
                             if str(arm.device).startswith("cuda") else 0.0),
        }
        curves.append(row)
        if rmse_valid < best["rmse"]:
            best = {"rmse": rmse_valid, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "train": (train_raw.copy(), train_std.copy()),
                    "valid": (valid_raw.copy(), valid_std.copy())}
        final = {"state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                 "train": (train_raw.copy(), train_std.copy()),
                 "valid": (valid_raw.copy(), valid_std.copy())}
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            log(f"[{arm.extra['meta']['arm_id']}] epoch {epoch}/{epochs} loss {row['train_loss']:.5f} "
                f"train {rmse_train:.6f} valid {rmse_valid:.6f} ({row['epoch_seconds']:.1f}s)")

    seconds = time.perf_counter() - t0
    last20 = curves[-20:]
    summary = {
        "epochs": int(epochs), "steps_per_epoch": steps_per_epoch,
        "seconds": seconds, "best_epoch": int(best["epoch"]),
        "best_valid_rmse": float(best["rmse"]),
        "best_train_rmse": C.rmse(best["train"][0], y_train),
        "best_train_mae": C.mae(best["train"][0], y_train),
        "best_valid_mae": C.mae(best["valid"][0], y_valid),
        "final_valid_rmse": float(curves[-1]["valid_rmse"]),
        "final_valid_mae": float(curves[-1]["valid_mae"]),
        "final_train_rmse": C.rmse(final["train"][0], y_train),
        "last20_valid_rmse": float(np.mean([r["valid_rmse"] for r in last20])),
        "last20_valid_mae": float(np.mean([r["valid_mae"] for r in last20])),
        "last20_train_rmse": float(np.mean([r["train_rmse"] for r in last20])),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "recipe": dict(recipe), "lambda": C.LAMBDA,
    }
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "summary": C.jsonable(summary)}, out_dir / "best.pt")
    torch.save({"state_dict": final["state"], "epoch": int(epochs),
                "summary": C.jsonable(summary)}, out_dir / "final.pt")
    predictions = {
        "train_row_ids": train_rows, "valid_row_ids": valid_rows,
        "train_y_true": y_train, "valid_y_true": y_valid,
        "best_train_y_pred": best["train"][0], "best_valid_y_pred": best["valid"][0],
        "final_train_y_pred": final["train"][0], "final_valid_y_pred": final["valid"][0],
        "best_epoch": np.int64(best["epoch"]),
        "last20_valid_rmse": np.float64(summary["last20_valid_rmse"]),
        "last20_valid_mae": np.float64(summary["last20_valid_mae"]),
    }
    return {"curves": curves, "summary": summary, "predictions": predictions}


if __name__ == "__main__":
    raise SystemExit(main())
