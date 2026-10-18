"""Run one Phase 1a training trajectory.

Examples
--------
GPU (one trajectory per rr job):
    python tracks/structural-sufficiency/code/phase1a/run_phase1a.py \
        --model S --n small --seed 0 --device cuda \
        --out-dir tracks/structural-sufficiency/runs/phase1a

CPU smoke:
    python .../run_phase1a.py --model X --n small --seed 0 --device cpu --epochs 2 --patience 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phase1a_data import LabelVault, PatchCache, load_frozen_split, load_raw_graphs  # noqa: E402
from phase1a_train import TrainConfig, train_one  # noqa: E402

SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
RAW = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
LABELS = REPO / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json"
CSV = REPO / "data/lipophilicity/Lipophilicity.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["S", "X", "G"], required=True)
    ap.add_argument("--n", choices=["small", "2352"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(REPO / "tracks/structural-sufficiency/runs/phase1a"))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--raw-graph", default=str(RAW))
    ap.add_argument("--labels", default=str(LABELS), help="Fit/Monitor-only label JSON")
    ap.add_argument("--csv", default=str(CSV), help="local CSV fallback only")
    # Smoke overrides; the frozen GPU protocol uses the TrainConfig defaults.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg_kwargs = {}
    if args.epochs is not None:
        cfg_kwargs["epochs"] = args.epochs
    if args.patience is not None:
        cfg_kwargs["patience"] = args.patience
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    cfg = TrainConfig(**cfg_kwargs)

    split = load_frozen_split(args.split)
    train_rows = split.training_rows(args.n)
    monitor_rows = split.monitor
    reserved_rows = split.reserved
    if set(train_rows) & set(monitor_rows):
        raise SystemExit("train/monitor overlap")
    if set(train_rows) & set(reserved_rows) or set(monitor_rows) & set(reserved_rows):
        raise SystemExit("reserved row leak")

    raws = load_raw_graphs(args.raw_graph)
    label_source = Path(args.labels)
    if not label_source.exists():
        label_source = Path(args.csv)
    vault = LabelVault(label_source, fit_rows=train_rows, monitor_rows=monitor_rows, reserved_rows=reserved_rows)
    vault.load()
    train_y = vault.labels(train_rows, "train")
    monitor_y = vault.labels(monitor_rows, "monitor")
    vault.assert_no_reserved(train_rows)
    vault.assert_no_reserved(monitor_rows)

    patch_cache = PatchCache.build(raws, list(train_rows) + list(monitor_rows), radius=cfg.radius)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out_dir = Path(args.out_dir)
    payload = train_one(
        model_name=args.model,
        seed=args.seed,
        n_label=args.n if args.n == "small" else int(args.n),
        train_rows=train_rows,
        monitor_rows=monitor_rows,
        train_y_raw=train_y,
        monitor_y_raw=monitor_y,
        patch_cache=patch_cache,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        reserved_rows=reserved_rows,
    )
    print(
        json.dumps(
            {
                "model": payload["model"],
                "n": payload["n_label"],
                "seed": payload["seed"],
                "n_train": payload["n_train"],
                "params": payload["parameters"],
                "best_epoch": payload["best_epoch"],
                "monitor_mae": payload["best_metrics"]["monitor"]["mae"],
                "monitor_rmse": payload["best_metrics"]["monitor"]["rmse"],
                "train_mae": payload["best_metrics"]["train"]["mae"],
                "train_seconds": payload["train_seconds"],
                "gpu_seconds": payload["gpu_seconds"],
                "peak_gpu_mb": payload["peak_gpu_memory_mb_allocated"],
                "label_source": str(label_source),
                "label_audit": vault.audit(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
