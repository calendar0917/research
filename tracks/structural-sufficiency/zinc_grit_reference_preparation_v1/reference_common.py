"""Shared plumbing for the frozen GRIT ZINC reference preparation.

This module deliberately does not implement a model or a training loop.  It
loads the pinned upstream GRIT source, applies the audited compatibility patch,
loads the official ZINC config, and exposes the official GraphGym/GRIT
constructors.  The model mathematics and the optimizer/scheduler maths remain
upstream code.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[2]
UPSTREAM = BUNDLE / "upstream" / "GRIT"
RUNTIME = BUNDLE / "runtime" / "GRIT"
COMPAT = BUNDLE / "compat"
DEPS = BUNDLE / ".deps"
PATCH = BUNDLE / "patches" / "0001-reference-no-test-compat.patch"
DEFAULT_CONFIG = BUNDLE / "configs" / "zinc-GRIT-RRWP.yaml"
UPSTREAM_COMMIT = (BUNDLE / "upstream" / "COMMIT").read_text().strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def _runtime_stamp() -> dict[str, str]:
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree_sha256": sha256_tree(UPSTREAM),
        "patch_sha256": sha256_file(PATCH),
    }


def prepare_runtime(force: bool = False) -> Path:
    """Create a patched runtime copy from pristine upstream + audited patch."""
    stamp_path = BUNDLE / "runtime" / "STAMP.json"
    stamp = _runtime_stamp()
    if not force and RUNTIME.exists() and stamp_path.exists():
        try:
            if json.loads(stamp_path.read_text()) == stamp:
                return RUNTIME
        except Exception:
            pass

    if RUNTIME.exists():
        if BUNDLE / "runtime" not in RUNTIME.parents:
            raise RuntimeError(f"refusing to remove unexpected path {RUNTIME}")
        shutil.rmtree(RUNTIME)
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        UPSTREAM, RUNTIME,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    subprocess.run(
        ["patch", "-p1", "-d", str(RUNTIME), "-i", str(PATCH)],
        check=True,
        capture_output=True,
        text=True,
    )
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    return RUNTIME


def setup_import_paths() -> None:
    """Put the compat shims, optional deps, and patched runtime first."""
    prepare_runtime()
    for path in (RUNTIME, DEPS, COMPAT):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO)] + args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def collect_env() -> dict[str, Any]:
    import torch
    import torch_geometric

    driver = None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except Exception:
        pass
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())]
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "nvidia_driver": driver,
        "torch_geometric": torch_geometric.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "research_git_commit": _git_value(["rev-parse", "HEAD"]),
        "research_git_dirty": bool(_git_value(["status", "--porcelain"])),
        "upstream_commit": UPSTREAM_COMMIT,
        "official_config_sha256": sha256_file(DEFAULT_CONFIG),
        "patch_sha256": sha256_file(PATCH),
        "compat_torch_scatter_sha256": sha256_file(COMPAT / "torch_scatter.py"),
        "compat_torch_sparse_sha256": sha256_file(COMPAT / "torch_sparse.py"),
    }


def load_reference_config(
    *,
    seed: int = 41,
    device: str = "cuda:0",
    splits: str = "train",
    timing: bool = False,
    stop_after_epoch: int = 0,
    max_wall_seconds: float = 0.0,
    max_train_graphs: int = 0,
    enable_ckpt: bool = False,
    ckpt_best: bool = False,
    out_dir: Path | None = None,
    run_name: str | None = None,
    functional_probe: bool = True,
):
    """Load the official config and add only runner/reference switches."""
    import grit  # noqa: F401  (registers custom configs before set_cfg)
    from yacs.config import CfgNode as CN
    from torch_geometric.graphgym.config import cfg, set_cfg

    if "test" in splits:
        raise ValueError("official test split is disabled for this reference bundle")
    if splits not in ("train", "train,val"):
        raise ValueError("splits must be 'train' or 'train,val'")

    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(DEFAULT_CONFIG))
    cfg.cfg_file = str(DEFAULT_CONFIG)
    cfg.work_dir = str(REPO)
    cfg.dataset.dir = str(REPO / "data")
    cfg.dataset.pe_transform_on_the_fly = False
    cfg.dataset.split_mode = "standard"
    cfg.seed = int(seed)
    cfg.run_id = int(seed)
    cfg.accelerator = device
    cfg.device = device
    cfg.num_workers = 0
    cfg.train.auto_resume = False
    cfg.train.eval_period = 1
    cfg.train.enable_ckpt = bool(enable_ckpt)
    cfg.train.ckpt_best = bool(ckpt_best)
    cfg.train.ckpt_clean = True
    cfg.tensorboard_each_run = False
    cfg.wandb.use = False
    cfg.mlflow.use = False
    cfg.reference = CN({
        "splits": splits,
        "timing": bool(timing),
        "stop_after_epoch": int(stop_after_epoch),
        "max_wall_seconds": float(max_wall_seconds),
        "max_train_graphs": int(max_train_graphs),
        "functional_probe": bool(functional_probe),
    })

    # These are the frozen literature-recipe values.  The runner must not
    # silently shorten the schedule; timing mode stops early without changing
    # max_epoch, so the cosine/warmup schedule is the 2000-epoch one.
    expected = {
        "model.type": "GritTransformer",
        "model.loss_fun": "l1",
        "model.graph_pooling": "add",
        "gt.layer_type": "GritTransformer",
        "gt.layers": 10,
        "gt.n_heads": 8,
        "gt.dim_hidden": 64,
        "gt.dropout": 0.0,
        "gt.attn_dropout": 0.2,
        "gt.update_e": True,
        "gt.attn.full_attn": True,
        "gt.attn.edge_enhance": True,
        "gt.attn.O_e": True,
        "gt.attn.norm_e": True,
        "gt.attn.fwl": False,
        "gnn.dim_inner": 64,
        "gnn.batchnorm": True,
        "gnn.dropout": 0.0,
        "optim.optimizer": "adamW",
        "optim.base_lr": 1e-3,
        "optim.weight_decay": 1e-5,
        "optim.max_epoch": 2000,
        "optim.num_warmup_epochs": 50,
        "optim.scheduler": "cosine_with_warmup",
        "optim.min_lr": 1e-6,
        "train.batch_size": 32,
        "train.mode": "custom",
        "dataset.format": "PyG-ZINC",
        "dataset.name": "subset",
        "dataset.node_encoder_num_types": 21,
        "dataset.edge_encoder_num_types": 4,
        "posenc_RRWP.enable": True,
        "posenc_RRWP.ksteps": 21,
        "posenc_RRWP.add_identity": True,
    }
    mismatches = {}
    for key, want in expected.items():
        got = cfg
        for part in key.split("."):
            got = got[part]
        if got != want:
            mismatches[key] = {"got": got, "expected": want}
    if mismatches:
        raise ValueError(f"frozen config mismatch: {mismatches}")

    out_dir = Path(out_dir) if out_dir is not None else BUNDLE / "runs"
    run_name = run_name or f"{splits.replace(',', '+')}-seed{seed}"
    cfg.out_dir = str(out_dir)
    cfg.run_dir = str(out_dir / run_name)
    return cfg


def make_loaders():
    """Create official GraphGym loaders with the split-count compatibility fix.

    PyG 2.6's GraphGym ``set_dataset_info`` counts any key containing the
    substring ``val``; the RRWP node attribute is literally named ``rrwp_val``,
    so a train-only dataset is misclassified as two splits.  The fix below is a
    narrow runner-side monkeypatch; it does not touch the model or loop.
    """
    import torch_geometric.graphgym.loader as gg_loader
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.config import cfg

    if not getattr(gg_loader, "_grit_reference_split_fix", False):
        original = gg_loader.set_dataset_info

        def fixed_set_dataset_info(dataset):
            original(dataset)
            names = getattr(dataset, "split_names", None)
            if names:
                gg_loader.cfg.share.num_splits = len(names)

        gg_loader.set_dataset_info = fixed_set_dataset_info
        gg_loader._grit_reference_split_fix = True

    loaders = create_loader()
    names = cfg.reference.splits.split(",")
    if len(loaders) != len(names):
        raise RuntimeError(
            f"expected {len(names)} loaders for {cfg.reference.splits}, got {len(loaders)}")
    if len(loaders) > 2:
        raise RuntimeError("official test loader was constructed")
    return loaders


def make_optimizer_and_scheduler(model):
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.graphgym.optim import (
        OptimizerConfig, create_optimizer, create_scheduler,
    )
    from grit.optimizer.extra_optimizers import ExtendedSchedulerConfig

    optimizer = create_optimizer(
        model.parameters(),
        OptimizerConfig(
            optimizer=cfg.optim.optimizer,
            base_lr=cfg.optim.base_lr,
            weight_decay=cfg.optim.weight_decay,
            momentum=cfg.optim.momentum,
        ),
    )
    scheduler = create_scheduler(
        optimizer,
        ExtendedSchedulerConfig(
            scheduler=cfg.optim.scheduler,
            steps=cfg.optim.steps,
            lr_decay=cfg.optim.lr_decay,
            max_epoch=cfg.optim.max_epoch,
            reduce_factor=cfg.optim.reduce_factor,
            schedule_patience=cfg.optim.schedule_patience,
            min_lr=cfg.optim.min_lr,
            num_warmup_epochs=cfg.optim.num_warmup_epochs,
            train_mode=cfg.train.mode,
            eval_period=cfg.train.eval_period,
            num_cycles=cfg.optim.num_cycles,
            min_lr_mode=cfg.optim.min_lr_mode,
        ),
    )
    return optimizer, scheduler


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def now_unix() -> float:
    return time.time()
