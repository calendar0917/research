"""Train one GSRO v0 arm (FULL or LOCAL2) on the official mollipo split.

    python code/gsro_train.py --arm full --epochs 120

Every epoch evaluates Train/Valid/Test in eval mode.  The checkpoint with the
lowest Valid RMSE is kept.  Test is DEVELOPMENT_TEST: it is reported and saved
but never used for any choice.

Artifacts (``runs/<arm>/``): curve.csv, best.pt, predictions.npz, run.json.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import gsro_common as C
import gsro_data as D
from gsro_model import GSROModel


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(C.REPO_ROOT))
    except ValueError:
        return str(path)


def lr_at(epoch: int, epochs: int, warmup: int, peak: float, floor: float) -> float:
    if warmup > 0 and epoch < warmup:
        return peak * float(epoch + 1) / float(warmup)
    span = max(1, epochs - warmup)
    t = float(epoch - warmup) / float(span)
    t = min(max(t, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * t))


def evaluate(
    model: GSROModel,
    batches: list[dict[str, Any]],
    *,
    arm: str | None = None,
) -> dict[str, Any]:
    model.eval()
    row_ids = []
    preds = []
    targets = []
    with torch.no_grad():
        for batch in batches:
            out = model(batch, arm=arm)
            row_ids.append(out["row_ids"].detach().cpu().numpy())
            preds.append(out["pred"].detach().cpu().double().numpy())
            targets.append(batch["y"].detach().cpu().double().numpy())
    row_ids = np.concatenate(row_ids)
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    err = pred - target
    order = np.argsort(row_ids)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "n": int(row_ids.size),
        "row_ids": row_ids[order],
        "pred": pred[order],
        "target": target[order],
    }


def train_epoch(
    model: GSROModel,
    batches: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float,
    generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    order = torch.randperm(len(batches), generator=generator).tolist()
    total = 0.0
    seen = 0
    last_grad_norm = 0.0
    for index in order:
        batch = batches[index]
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = nn.functional.mse_loss(out["pred_std"], batch["y_std"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        last_grad_norm = float(grad_norm)
        optimizer.step()
        total += float(loss.detach()) * int(batch["y_std"].numel())
        seen += int(batch["y_std"].numel())
    return {"train_loss_std": total / max(seen, 1), "grad_norm": last_grad_norm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("full", "local2"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--allow-any-epochs",
        action="store_true",
        help="local smoke tests only; formal runs use a frozen candidate horizon",
    )
    args = parser.parse_args()

    config = C.load_config()
    epochs = int(config["epochs_candidates"][0]) if args.epochs is None else int(args.epochs)
    if epochs not in [int(e) for e in config["epochs_candidates"]] and not args.allow_any_epochs:
        raise SystemExit(f"epochs must be one of {config['epochs_candidates']}")
    seed = int(config["seed"])
    C.set_seed(seed)
    device = torch.device(args.device)

    out_dir = Path(args.out_dir) if args.out_dir else C.runs_dir() / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = D.load_prepared(config)
    model = GSROModel(prep, config, args.arm).to(device)
    initial_sha = model.initial_state_sha()
    param_count = model.parameter_count()

    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["eval_batch_size"])
    train_batches = D.size_sorted_batches(prep.splits["train"], batch_size, device)
    eval_batches = {
        "train": D.size_sorted_batches(prep.splits["train"], eval_batch_size, device),
        "valid": D.size_sorted_batches(prep.splits["valid"], eval_batch_size, device),
        "test": D.size_sorted_batches(prep.splits["test"], eval_batch_size, device),
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr_peak"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed + 991)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    curve_path = out_dir / "curve.csv"
    fieldnames = [
        "epoch", "lr", "train_loss_std", "grad_norm",
        "train_rmse", "train_mae", "valid_rmse", "valid_mae",
        "test_rmse", "test_mae", "epoch_seconds",
    ]
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    best = {"valid_rmse": float("inf"), "epoch": -1}
    start = time.time()
    epoch_rows: list[dict[str, float]] = []
    for epoch in range(epochs):
        epoch_start = time.time()
        lr = lr_at(epoch, epochs, int(config["warmup_epochs"]), float(config["lr_peak"]), float(config["lr_min"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_stats = train_epoch(
            model, train_batches, optimizer,
            grad_clip=float(config["grad_clip"]), generator=generator,
        )
        metrics = {name: evaluate(model, eval_batches[name]) for name in ("train", "valid", "test")}
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss_std": train_stats["train_loss_std"],
            "grad_norm": train_stats["grad_norm"],
            "train_rmse": metrics["train"]["rmse"],
            "train_mae": metrics["train"]["mae"],
            "valid_rmse": metrics["valid"]["rmse"],
            "valid_mae": metrics["valid"]["mae"],
            "test_rmse": metrics["test"]["rmse"],
            "test_mae": metrics["test"]["mae"],
            "epoch_seconds": elapsed,
        }
        epoch_rows.append(row)
        with open(curve_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        if row["valid_rmse"] < best["valid_rmse"]:
            best = {
                "valid_rmse": row["valid_rmse"],
                "valid_mae": row["valid_mae"],
                "train_rmse": row["train_rmse"],
                "test_rmse": row["test_rmse"],
                "test_mae": row["test_mae"],
                "epoch": epoch,
            }
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arm": args.arm,
                    "epoch": epoch,
                    "epochs": epochs,
                    "best": best,
                    "config": config,
                    "data_identity": prep.identity(),
                    "param_count": param_count,
                    "initial_state_sha256": initial_sha,
                },
                out_dir / "best.pt",
            )
        print(
            f"GSRO_PROGRESS arm={args.arm} epoch={epoch + 1}/{epochs} lr={lr:.3e} "
            f"train={row['train_rmse']:.5f} valid={row['valid_rmse']:.5f} "
            f"test={row['test_rmse']:.5f} sec={elapsed:.2f} best_epoch={best['epoch']}",
            flush=True,
        )

    wall = time.time() - start
    # Final predictions of the best-Valid checkpoint on all three splits.
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final = {name: evaluate(model, eval_batches[name]) for name in ("train", "valid", "test")}
    payload: dict[str, np.ndarray] = {}
    for name in ("train", "valid", "test"):
        if not np.array_equal(final[name]["row_ids"], prep.splits[name].row_ids):
            raise RuntimeError(f"{name}: prediction row ids are not aligned with the split")
        payload[f"{name}_row_ids"] = prep.splits[name].row_ids
        payload[f"{name}_y"] = prep.splits[name].y
        payload[f"{name}_pred"] = final[name]["pred"]
        payload[f"{name}_n_atoms"] = prep.splits[name].n_atoms
    np.savez_compressed(out_dir / "predictions.npz", **payload)

    peak_mem = (
        float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    )
    run = {
        "round": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": args.arm,
        "config": config,
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "data": prep.identity(),
        "data_cache_sha256": prep.payload_sha256,
        "environment": C.environment(),
        "param_count": param_count,
        "initial_state_sha256": initial_sha,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": best["epoch"],
        "best_valid_rmse": best["valid_rmse"],
        "best": best,
        "final_metrics_best_checkpoint": {
            name: {
                "rmse": final[name]["rmse"],
                "mae": final[name]["mae"],
                "n": final[name]["n"],
            }
            for name in ("train", "valid", "test")
        },
        "wall_seconds": wall,
        "peak_gpu_bytes": peak_mem,
        "curve": epoch_rows,
        "predictions_path": _relative(out_dir / "predictions.npz"),
        "checkpoint_path": _relative(out_dir / "best.pt"),
    }
    C.write_json(out_dir / "run.json", run)
    print(
        f"GSRO_DONE arm={args.arm} best_epoch={best['epoch']} "
        f"valid={final['valid']['rmse']:.5f} test={final['test']['rmse']:.5f} "
        f"wall_seconds={wall:.1f} peak_gpu_mb={peak_mem / 2**20:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
