#!/usr/bin/env python
"""Entry point for the frozen GRIT ZINC reference preparation.

Modes
-----
timing : official config, official scheduler (max_epoch=2000), one warmup
         train epoch + three timed train epochs, then a no-grad forward over
         1000 fixed *train* graphs.  It never constructs official valid/test.
train  : the same official config with train+valid loaders and best-valid
         checkpointing, intended for a later 2000-epoch literature-recipe
         reproduction.  It requires an explicit --confirm-long-run and is not
         started by this preparation task.

The model, loss, optimizer, scheduler and official train/eval epoch functions
are upstream code.  This file only orchestrates them and records timing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import reference_common as rc

rc.setup_import_paths()

import torch  # noqa: E402
from torch_geometric.graphgym.model_builder import create_model  # noqa: E402
from torch_geometric.graphgym.utils.comp_budget import params_count  # noqa: E402
from torch_geometric import seed_everything  # noqa: E402

import grit  # noqa: E402,F401  (must precede set_cfg: registers configs/modules)
from grit.logger import create_logger  # noqa: E402
from grit.train.custom_train import custom_train  # noqa: E402

import reference_checks as checks  # noqa: E402


def _base_dataset(loader):
    ds = loader.dataset
    return getattr(ds, "dataset", ds)


def _dataset_stats(dataset, limit: int | None = None) -> dict:
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    ys = []
    atom_types = set()
    bond_types = set()
    nodes = []
    for i in range(n):
        d = dataset[i]
        ys.append(float(d.y.view(-1)[0]))
        atom_types.update(int(v) for v in torch.unique(d.x[:, 0]).tolist())
        bond_types.update(int(v) for v in torch.unique(d.edge_attr).tolist())
        nodes.append(int(d.num_nodes))
    y = torch.tensor(ys, dtype=torch.float64)
    return {
        "graphs": n,
        "atom_types": sorted(atom_types),
        "bond_types": sorted(bond_types),
        "num_nodes_min": min(nodes),
        "num_nodes_max": max(nodes),
        "num_nodes_mean": sum(nodes) / len(nodes),
        "label_mean": float(y.mean()),
        "label_std_population": float(y.std(unbiased=False)),
        "label_std_sample": float(y.std(unbiased=True)) if n > 1 else None,
        "label_min": float(y.min()),
        "label_max": float(y.max()),
    }


def _timing_eval_train(model, dataset, device: str, n: int = 1000,
                       batch_size: int = 32) -> dict:
    """No-grad forward over fixed train rows; no loss/metric is computed."""
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    n = min(int(n), len(dataset))
    loader = DataLoader(Subset(dataset, range(n)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    seen = 0
    batches = 0
    checksum = 0.0
    finite = True
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device(device))
            pred, _ = model(batch)
            finite = finite and bool(torch.isfinite(pred).all())
            checksum += float(pred.detach().double().sum().cpu())
            seen += int(pred.size(0))
            batches += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = {}
    if torch.cuda.is_available():
        peak = {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
        }
    return {
        "purpose": "timing proxy for one 1000-row eval pass; NOT official valid",
        "split": "first 1000 rows of official ZINC train",
        "graphs": seen,
        "batches": batches,
        "batch_size": batch_size,
        "seconds": elapsed,
        "graphs_per_s": seen / elapsed if elapsed > 0 else None,
        "pred_finite": finite,
        "pred_sum_checksum": checksum,
        "score_computed": False,
        "labels_used_for_score": False,
        **peak,
    }


def _functional_probe(model, dataset) -> dict:
    data0 = dataset[0].clone()
    out = {
        "params": checks.parameter_check(model),
        "rrwp": checks.rrwp_check(data0),
        "participation": checks.participation_check(model, data0),
        "forward_backward": checks.forward_backward_check(model, data0),
        "renumbering": checks.renumbering_check(model, dataset, count=5),
    }
    out["all_pass"] = all(bool(v.get("pass")) for k, v in out.items()
                          if isinstance(v, dict) and "pass" in v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["timing", "train", "smoke"], default="timing")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--splits", default=None,
                   help="train or train,val; test is always disabled")
    p.add_argument("--stop-after", type=int, default=None,
                   help="stop after N full epochs without changing max_epoch")
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="stop before an epoch that would exceed this loop wall time")
    p.add_argument("--max-train-graphs", type=int, default=None,
                   help="local smoke cap; 0 means the full official train split")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--timing-out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--confirm-long-run", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "timing":
        splits = args.splits or "train"
        stop_after = 4 if args.stop_after is None else args.stop_after
        max_wall = 420.0 if args.max_wall_seconds is None else args.max_wall_seconds
        max_train = 0 if args.max_train_graphs is None else args.max_train_graphs
        enable_ckpt = False
        ckpt_best = False
        timing = True
    elif args.mode == "smoke":
        splits = args.splits or "train"
        stop_after = 1 if args.stop_after is None else args.stop_after
        max_wall = 120.0 if args.max_wall_seconds is None else args.max_wall_seconds
        max_train = 64 if args.max_train_graphs is None else args.max_train_graphs
        enable_ckpt = False
        ckpt_best = False
        timing = True
    else:
        splits = args.splits or "train,val"
        stop_after = 0 if args.stop_after is None else args.stop_after
        max_wall = 0.0 if args.max_wall_seconds is None else args.max_wall_seconds
        max_train = 0 if args.max_train_graphs is None else args.max_train_graphs
        enable_ckpt = True
        ckpt_best = True
        timing = False

    if "test" in splits:
        raise SystemExit("official test split is disabled in this bundle")
    if args.mode == "train" and not args.confirm_long_run:
        print("Refusing to start the 2000-epoch run without --confirm-long-run.")
        print("This task only prepares and times the reference; it does not start it.")
        return 2
    if args.mode == "train" and splits != "train,val":
        raise SystemExit("literature-recipe mode requires --splits train,val")

    job_start_unix = time.time()
    job_start_perf = time.perf_counter()

    cfg = rc.load_reference_config(
        seed=args.seed,
        device=args.device,
        splits=splits,
        timing=timing,
        stop_after_epoch=stop_after,
        max_wall_seconds=max_wall,
        max_train_graphs=max_train,
        enable_ckpt=enable_ckpt,
        ckpt_best=ckpt_best,
        out_dir=args.out_dir,
        run_name=args.run_name,
        functional_probe=not args.skip_probe,
    )

    run_dir = Path(cfg.run_dir)
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"run dir exists: {run_dir} (use --overwrite)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reference] mode={args.mode} splits={splits} seed={args.seed} "
          f"max_epoch={cfg.optim.max_epoch} device={args.device}")
    print(f"[reference] upstream={rc.UPSTREAM_COMMIT} config={rc.DEFAULT_CONFIG}")
    print(f"[reference] run_dir={run_dir}")

    seed_everything(cfg.seed)

    data_start = time.perf_counter()
    loaders = rc.make_loaders()
    data_build_s = time.perf_counter() - data_start
    loggers = create_logger()
    model_start = time.perf_counter()
    model = create_model()
    model_build_s = time.perf_counter() - model_start
    optimizer, scheduler = rc.make_optimizer_and_scheduler(model)
    cfg.params = params_count(model)

    dataset = _base_dataset(loaders[0])
    data_stats = _dataset_stats(dataset)
    print(f"[reference] loaders={len(loaders)} train={len(dataset)} "
          f"steps={len(loaders[0])} params={cfg.params}")

    probe = {}
    if not args.skip_probe and cfg.reference.functional_probe:
        probe_start = time.perf_counter()
        probe = _functional_probe(model, dataset)
        probe["seconds"] = time.perf_counter() - probe_start
        print(f"[reference] functional probe all_pass={probe.get('all_pass')}")
        model.train()

    setup_s = time.perf_counter() - job_start_perf
    train_start_unix = time.time()
    custom_train(loggers, loaders, model, optimizer, scheduler)
    train_end_unix = time.time()

    timing_path = run_dir / "timing.json"
    if timing and not timing_path.exists():
        raise SystemExit(f"training loop did not write {timing_path}")
    loop_timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    eval_proxy = {}
    if args.mode in ("timing", "smoke"):
        eval_proxy = _timing_eval_train(model, dataset, args.device,
                                        n=min(1000, len(dataset)),
                                        batch_size=cfg.train.batch_size)

    job_end_unix = time.time()
    job_wall_s = time.perf_counter() - job_start_perf
    epoch_total = sum(float(r["epoch_total_s"]) for r in loop_timing.get("records", []))
    gpu_compute_s = epoch_total + float(eval_proxy.get("seconds", 0.0))

    env = rc.collect_env()
    payload = {
        "schema": "zinc_grit_reference_timing_v1",
        "status": "ok",
        "mode": args.mode,
        "seed": int(args.seed),
        "upstream": {
            "url": "https://github.com/LiamMa/GRIT",
            "commit": rc.UPSTREAM_COMMIT,
            "official_config": "configs/GRIT/zinc-GRIT-RRWP.yaml",
            "official_config_sha256": rc.sha256_file(rc.DEFAULT_CONFIG),
            "patch": "patches/0001-reference-no-test-compat.patch",
            "patch_sha256": rc.sha256_file(rc.PATCH),
        },
        "environment": env,
        "config": {
            "path": str(rc.DEFAULT_CONFIG),
            "sha256": rc.sha256_file(rc.DEFAULT_CONFIG),
            "frozen_values": {
                "layers": cfg.gt.layers,
                "n_heads": cfg.gt.n_heads,
                "dim_hidden": cfg.gt.dim_hidden,
                "attn_dropout": cfg.gt.attn_dropout,
                "update_e": cfg.gt.update_e,
                "deg_scaler": cfg.gt.attn.deg_scaler,
                "full_attn": cfg.gt.attn.full_attn,
                "edge_enhance": cfg.gt.attn.edge_enhance,
                "O_e": cfg.gt.attn.O_e,
                "norm_e": cfg.gt.attn.norm_e,
                "rrwp_ksteps": cfg.posenc_RRWP.ksteps,
                "rrwp_add_identity": cfg.posenc_RRWP.add_identity,
                "batch_size": cfg.train.batch_size,
                "max_epoch": cfg.optim.max_epoch,
                "num_warmup_epochs": cfg.optim.num_warmup_epochs,
                "scheduler": cfg.optim.scheduler,
                "min_lr": cfg.optim.min_lr,
                "optimizer": cfg.optim.optimizer,
                "base_lr": cfg.optim.base_lr,
                "weight_decay": cfg.optim.weight_decay,
                "loss": cfg.model.loss_fun,
                "pooling": cfg.model.graph_pooling,
                "test_enabled": False,
            },
        },
        "data": {
            "source": "official PyG ZINC subset=True",
            "split_loaded": splits,
            "official_test_constructed": False,
            "official_valid_constructed": "val" in splits,
            "train_stats": data_stats,
        },
        "model": {
            "params_total": int(cfg.params),
            "params_expected_paper_zinc": 473473,
            "params_difference": int(cfg.params) - 473473,
        },
        "job": {
            "start_unix": job_start_unix,
            "end_unix": job_end_unix,
            "wall_s": job_wall_s,
            "setup_s": setup_s,
            "data_build_s": data_build_s,
            "model_build_s": model_build_s,
            "train_start_unix": train_start_unix,
            "train_end_unix": train_end_unix,
            "gpu_occupancy_hours_process_wall": job_wall_s / 3600.0,
            "gpu_compute_hours_epochs_plus_eval": gpu_compute_s / 3600.0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "functional_probe": probe,
        "loop_timing": loop_timing,
        "eval_proxy_1000_train": eval_proxy,
        "notes": [
            "No official ZINC test split was constructed or read.",
            "No official-valid score is computed in timing mode.",
            "The 1000-row eval is a timing proxy on train rows, not validation.",
            "The scheduler is the official 2000-epoch cosine/warmup schedule; "
            "timing mode stops early instead of shrinking max_epoch.",
            "Paper 0.059 +/- 0.002 is an official test MAE and is not a local target.",
        ],
    }

    out_path = args.timing_out
    if out_path is None:
        out_path = run_dir / "timing_summary.json"
    rc.write_json(Path(out_path), payload)
    print(f"[reference] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
