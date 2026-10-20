"""ZSRT training / evaluation under zinc-perf-protocol-v1.

Backbone: the audited GRIT GritTransformer (RRWP relation input + full
attention), constructed from the vendored pristine upstream runtime. The model
is borrowed, not claimed as novel; this runner only supplies data, recipe,
per-epoch validation, best-valid checkpointing and provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import zpp_common as C
import zpp_data as D

SOUP_LAST = 5
SNAPSHOT_EVERY = 20
SNAPSHOT_ROWS = 2000

REF_CONFIG = C.REF_BUNDLE / "configs" / "zinc-GRIT-RRWP.yaml"


def load_variant(name: str) -> dict:
    doc = C.load_yaml(C.ZSRT_YAML)
    if name not in doc["variants"]:
        raise KeyError(f"unknown variant {name!r}; known {sorted(doc['variants'])}")
    variant = dict(doc["base"])
    variant.update(doc["variants"][name])
    variant["name"] = name
    return variant


def build_cfg(variant: dict, device: str):
    C.ensure_grit_runtime()
    import grit  # noqa: F401  (registers modules)
    from torch_geometric.graphgym.config import cfg, set_cfg

    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(REF_CONFIG))
    cfg.dataset.dir = str(C.REPO / "data")
    cfg.device = str(device)
    cfg.accelerator = str(device)
    cfg.num_workers = 0
    cfg.train.batch_size = int(variant["batch_size"])
    cfg.train.enable_ckpt = False
    cfg.train.ckpt_best = False
    cfg.train.ckpt_clean = True
    cfg.train.eval_period = 1
    cfg.tensorboard_each_run = False
    cfg.wandb.use = False
    cfg.mlflow.use = False
    cfg.optim.max_epoch = int(variant["max_epochs"])
    cfg.optim.num_warmup_epochs = int(variant["warmup_epochs"])
    cfg.optim.base_lr = float(variant["base_lr"])
    cfg.optim.min_lr = float(variant["min_lr"])
    cfg.optim.weight_decay = float(variant["weight_decay"])
    cfg.optim.optimizer = "adamW"
    cfg.optim.scheduler = "cosine_with_warmup"
    cfg.gt.layers = int(variant["layers"])
    cfg.gt.n_heads = int(variant["n_heads"])
    cfg.gt.dim_hidden = int(variant["dim_hidden"])
    cfg.gnn.dim_inner = int(variant["dim_hidden"])
    cfg.gt.dropout = float(variant["dropout"])
    cfg.gt.attn_dropout = float(variant["attn_dropout"])
    cfg.gt.layer_norm = bool(variant["layer_norm"])
    cfg.gt.batch_norm = bool(variant["batch_norm"])
    cfg.gt.update_e = bool(variant["update_e"])
    cfg.gt.attn.full_attn = bool(variant["full_attn"])
    cfg.gt.attn.edge_enhance = bool(variant["edge_enhance"])
    cfg.gt.attn.O_e = bool(variant["O_e"])
    cfg.gt.attn.norm_e = bool(variant["norm_e"])
    cfg.gt.attn.deg_scaler = bool(variant["deg_scaler"])
    cfg.posenc_RRWP.enable = bool(variant["rrwp"])
    cfg.posenc_RRWP.ksteps = int(variant["ksteps"])
    cfg.model.graph_pooling = str(variant["graph_pooling"])
    cfg.model.loss_fun = str(variant["loss"])
    return cfg


def build_model(variant: dict, seed: int, device: str):
    cfg = build_cfg(variant, device)
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.comp_budget import params_count
    from torch_geometric import seed_everything

    seed_everything(int(seed))
    model = create_model()
    n_params = int(sum(p.numel() for p in model.parameters()))
    if n_params != int(params_count(model)):
        raise RuntimeError("parameter count mismatch")
    return model, cfg, n_params


def run_dir(variant: dict, seed: int) -> Path:
    return C.RUNS / f"zsrt_{variant['name']}_s{int(seed)}"


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state(model, state, device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


@torch.no_grad()
def predict(model, data, device, *, batch_size: int = 256, split: str = "valid") -> np.ndarray:
    from torch_geometric.data import Batch

    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(data), batch_size):
            chunk = data[start:min(start + batch_size, len(data))]
            batch = Batch.from_data_list(chunk).to(device)
            batch.split = split
            pred, _ = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def evaluate(model, data, labels, device, *, batch_size: int = 256, split: str = "valid"):
    pred = predict(model, data, device, batch_size=batch_size, split=split)
    return pred, C.mae(pred, labels)


def _batch_iter(data, order, batch_size):
    from torch_geometric.data import Batch

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield Batch.from_data_list([data[i] for i in idx])


def _curve_entry(epoch, online, lr, seconds, snapshot, valid_mae):
    return {
        "epoch": int(epoch),
        "online_train_mae": float(online),
        "valid_mae": float(valid_mae),
        "snapshot_train_mae": (None if snapshot is None else float(snapshot)),
        "lr": float(lr),
        "seconds": float(seconds),
    }


def write_curve_csv(path: Path, curve) -> None:
    keys = []
    for entry in curve:
        for k in entry:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for entry in curve:
            w.writerow(entry)


def train(variant_name: str, seed: int, device: str, *, force: bool = False,
          max_epochs: int | None = None) -> dict:
    variant = load_variant(variant_name)
    if max_epochs is not None:
        variant["max_epochs"] = int(max_epochs)
        variant["smoke_override"] = True
    out = run_dir(variant, seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        C.log(f"[zsrt {variant_name} s{seed}] already complete")
        return C.read_json(out / "train.json")

    if not C.DATA_AUDIT.exists():
        raise RuntimeError("run `zpp_data.py audit` before training")
    audit = C.read_json(C.DATA_AUDIT)

    train_data = D.build_data("train", ksteps=variant["ksteps"], rrwp=variant["rrwp"])
    valid_data = D.build_data("valid", ksteps=variant["ksteps"], rrwp=variant["rrwp"])
    y_train = D.split_labels(train_data)
    y_valid = D.split_labels(valid_data)
    if len(train_data) != 10000 or len(valid_data) != 1000:
        raise RuntimeError("wrong split sizes")

    model, cfg, n_params = build_model(variant, seed, device)
    model.to(torch.device(device))
    init_hash = _state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(variant["base_lr"]),
        weight_decay=float(variant["weight_decay"]),
        betas=(0.9, 0.999), eps=1e-8,
    )
    epochs = int(variant["max_epochs"])
    steps_per_epoch = math.ceil(len(train_data) / int(variant["batch_size"]))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(variant["warmup_epochs"]) * steps_per_epoch
    gen = torch.Generator().manual_seed(int(seed) * 1000 + 7)

    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    order_hasher = hashlib.sha256()
    snap_rows = list(range(min(SNAPSHOT_ROWS, len(train_data))))
    snap_data = [train_data[i] for i in snap_rows]
    snap_y = y_train[snap_rows]
    step = 0
    limit_reached = False
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_data), generator=gen).tolist()
        order_hasher.update(np.ascontiguousarray(np.asarray(order, dtype=np.int64)).tobytes())
        total_err, n_mol = 0.0, 0
        last_lr = float(variant["base_lr"])
        t_ep = time.perf_counter()
        for batch in _batch_iter(train_data, order, int(variant["batch_size"])):
            batch = batch.to(device)
            batch.split = "train"
            pred, true = model(batch)
            target = true.view(-1)
            loss = F.l1_loss(pred.view(-1), target)
            step += 1
            last_lr = C.cosine_with_warmup_lr(
                step, total_steps, warmup_steps,
                float(variant["base_lr"]), float(variant["min_lr"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(variant["clip_grad_norm"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += int(pred.size(0))
        online = total_err / max(n_mol, 1)
        state = _cpu_state(model)
        states.append((epoch, state))
        _, valid_mae = evaluate(model, valid_data, y_valid, device, split="valid")
        snapshot = None
        if epoch % SNAPSHOT_EVERY == 0 or epoch == 1 or epoch == epochs:
            _, snapshot = evaluate(model, snap_data, snap_y, device, split="train")
        curve.append(_curve_entry(epoch, online, last_lr, time.perf_counter() - t_ep, snapshot, valid_mae))
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = state
            torch.save({"state_dict": state, "epoch": int(epoch), "valid_mae": float(valid_mae)},
                       out / "best.pt")
        C.log(f"[zsrt {variant_name} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} "
              f"online={online:.6f} snap={'-' if snapshot is None else f'{snapshot:.6f}'} "
              f"lr={last_lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} {curve[-1]['seconds']:.1f}s")

    wall = time.perf_counter() - started
    torch.save(
        {
            "model": _cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": epochs,
            "best": best,
            "curve": curve,
            "step": step,
        },
        out / "final.pt",
    )
    (out / "last5").mkdir(exist_ok=True)
    for e, st in list(states):
        torch.save({"state_dict": st, "epoch": int(e)}, out / "last5" / f"epoch_{int(e):03d}.pt")
    members = [int(e) for e, _ in states]
    soup = {k: torch.stack([st[k].float() for _, st in states]).mean(0) for k in states[0][1]}
    torch.save({"state_dict": soup, "members": members}, out / "soup.pt")

    fixed = {}
    eval_model, _, _ = build_model(variant, seed, device)
    eval_model.to(torch.device(device))
    _load_state(eval_model, best_state, device)
    bp_valid, bv = evaluate(eval_model, valid_data, y_valid, device, split="valid")
    bp_train, bt = evaluate(eval_model, train_data, y_train, device, split="train")
    fixed["best"] = {"valid_mae": bv, "train_mae_eval_mode": bt, "epoch": best["epoch"]}

    _load_state(eval_model, _cpu_state(model), device)
    fp_valid, fv = evaluate(eval_model, valid_data, y_valid, device, split="valid")
    fp_train, ft = evaluate(eval_model, train_data, y_train, device, split="train")
    fixed["final"] = {"valid_mae": fv, "train_mae_eval_mode": ft, "epoch": epochs}

    _load_state(eval_model, soup, device)
    sp_valid, sv = evaluate(eval_model, valid_data, y_valid, device, split="valid")
    sp_train, st = evaluate(eval_model, train_data, y_train, device, split="train")
    fixed["soup"] = {"valid_mae": sv, "train_mae_eval_mode": st, "members": members}

    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name, pred in (("best", bp_valid), ("final", fp_valid), ("soup", sp_valid)):
        np.savez(pred_dir / f"valid_{name}.npz", row=np.arange(len(pred)), pred=pred)

    info = {
        "protocol": C.PROTOCOL_ID,
        "model": "ZSRT (GRIT-style RRWP + full attention; borrowed backbone)",
        "variant": variant,
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": len(train_data),
            "n_valid": len(valid_data),
            "official_test_loaded": False,
            "data_audit_sha256": C.sha256_file(C.DATA_AUDIT),
            "train_cache_sha256": C.sha256_file(
                C.CACHE / (f"zinc_subset_train_rrwp{variant['ksteps']}.pt" if variant["rrwp"]
                           else "zinc_subset_train_norrwp.pt")),
            "valid_cache_sha256": C.sha256_file(
                C.CACHE / (f"zinc_subset_valid_rrwp{variant['ksteps']}.pt" if variant["rrwp"]
                           else "zinc_subset_valid_norrwp.pt")),
            "train_label_sha256": C.sha256_array(y_train),
            "valid_label_sha256": C.sha256_array(y_valid),
        },
        "recipe": {
            "epochs": epochs,
            "batch_size": int(variant["batch_size"]),
            "steps_per_epoch": steps_per_epoch,
            "updates": total_steps,
            "warmup_steps": warmup_steps,
            "optimizer": "AdamW",
            "base_lr": float(variant["base_lr"]),
            "min_lr": float(variant["min_lr"]),
            "weight_decay": float(variant["weight_decay"]),
            "grad_clip": float(variant["clip_grad_norm"]),
            "schedule": "linear warmup + per-step cosine",
            "loss": "raw-y L1",
            "dropout": float(variant["dropout"]),
            "attn_dropout": float(variant["attn_dropout"]),
        },
        "parameter_count": int(n_params),
        "parameter_cap": 500000,
        "model_init_sha256": init_hash,
        "best_valid": best,
        "fixed_objects": fixed,
        "curve": curve,
        "batch_order_sha256": order_hasher.hexdigest(),
        "wall_seconds": float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "device": str(device),
        "smoke_override": bool(variant.get("smoke_override", False)),
        "notes": [
            "online_train_mae is a within-epoch pre-update running mean",
            "snapshot_train_mae and train_mae_eval_mode are eval-mode fixed-model measurements",
            "official test is never constructed by this runner",
        ],
    }
    C.write_json(out / "train.json", info)
    write_curve_csv(out / "curve.csv", curve)
    C.log(f"[zsrt {variant_name} s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
          f"final={fv:.6f} soup={sv:.6f} params={n_params} wall={wall:.1f}s")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "eval"])
    p.add_argument("--config", default="A_main")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--which", default="best", choices=["best", "final", "soup"])
    p.add_argument("--split", default="valid", choices=["valid"])
    args = p.parse_args(argv)
    if args.command == "train":
        train(args.config, args.seed, args.device, force=args.force, max_epochs=args.max_epochs)
        return 0
    variant = load_variant(args.config)
    out = run_dir(variant, args.seed)
    info = C.read_json(out / "train.json")
    device = args.device
    data = D.build_data(args.split, ksteps=variant["ksteps"], rrwp=variant["rrwp"])
    y = D.split_labels(data)
    model, _, _ = build_model(variant, args.seed, device)
    model.to(torch.device(device))
    ckpt = torch.load(out / f"{args.which}.pt", map_location="cpu", weights_only=False)
    _load_state(model, ckpt["state_dict"], device)
    pred, value = evaluate(model, data, y, device, split=args.split)
    C.log(f"[zsrt {args.config} s{args.seed}] {args.which} {args.split} MAE={value:.6f}")
    C.write_json(out / f"eval_{args.which}_{args.split}.json",
                 {"valid_mae": value, "which": args.which, "split": args.split,
                  "epoch": ckpt.get("epoch"), "n": len(data)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
