"""Recipe R0, training loops, evaluation, checkpointing and run bookkeeping.

``orders`` are the shared per-epoch molecule permutations: every arm of a given
seed consumes exactly the same permutation sequence (``default_rng(20261121+seed)``),
so a BIND/MEAN or REL/SELF difference is never a batch-order difference.
No Test loader is ever built here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import wp_common as C
import wp_data as D

ORDER_SEED_BASE = 20261121


@dataclass
class Recipe:
    epochs: int = 240
    batch: int = 128
    eval_batch: int = 512
    peak_lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-4
    min_lr: float = 1e-5
    clip: float = 5.0
    seed: int = 0
    recipe_id: str = "R0"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def lr_at(step: int, warmup_steps: int, total_steps: int, recipe: Recipe) -> float:
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return recipe.warmup_start_lr + (recipe.peak_lr - recipe.warmup_start_lr) * frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return recipe.min_lr + 0.5 * (recipe.peak_lr - recipe.min_lr) * (1.0 + np.cos(np.pi * progress))


def epoch_orders(n_train: int, epochs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(ORDER_SEED_BASE + seed)
    return [rng.permutation(n_train) for _ in range(epochs)]


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def param_signature(model) -> dict[str, list[int]]:
    return {name: list(p.shape) for name, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def build_word_model(spec, tmpl: Mapping[str, Any], seed: int, device: str, hidden: int = 64):
    import torch

    import wp_model as M

    dims = C.official_dims()
    model = M.PortBindModel(spec, tmpl, dims, hidden=hidden)
    M.init_by_name(model, 1000 + seed)
    return model.to(device)


def build_gine_model(raw: Mapping[str, Any], seed: int, device: str, *, raw_features: bool = False):
    import torch

    import sq_gine as SG
    from torch_geometric.nn import GINEConv
    import torch.nn as nn

    dims = C.official_dims()
    counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(9)}
    counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(3)})
    counts["degree"] = 17
    counts["ring"] = 14
    h, layers = 128, 5

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_embeddings = nn.ModuleList([nn.Embedding(int(counts[f"atom_{j}"]), h)
                                                  for j in range(9)])
            self.bond_embeddings = nn.ModuleList([nn.Embedding(int(counts[f"bond_{j}"]), h)
                                                  for j in range(3)])
            self.use_extra = not raw_features
            if self.use_extra:
                self.degree_embedding = nn.Embedding(int(counts["degree"]), h)
                self.ring_embedding = nn.Embedding(int(counts["ring"]), h)
            self.convs = nn.ModuleList([
                GINEConv(nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h)),
                         train_eps=True) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(layers)])
            self.drop = nn.Dropout(0.10)
            self.head = nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU(), nn.Dropout(0.10),
                                      nn.Linear(h, 1))

        def forward(self, batch):
            x = self.atom_embeddings[0](batch.x_cat[:, 0])
            for j in range(1, 9):
                x = x + self.atom_embeddings[j](batch.x_cat[:, j])
            if self.use_extra:
                x = x + self.degree_embedding(batch.deg) + self.ring_embedding(batch.ring)
            edge_attr = self.bond_embeddings[0](batch.edge_cat[:, 0])
            for j in range(1, 3):
                edge_attr = edge_attr + self.bond_embeddings[j](batch.edge_cat[:, j])
            for conv, norm in zip(self.convs, self.norms):
                residual = x
                x = norm(conv(x, batch.edge_index, edge_attr))
                x = torch.nn.functional.silu(x)
                x = self.drop(x) + residual
            from torch_geometric.nn import global_add_pool, global_mean_pool

            summed = global_add_pool(x, batch.batch)
            meaned = global_mean_pool(x, batch.batch)
            n_atoms = batch.n_atoms.view(-1, 1).to(x.dtype)
            return self.head(torch.cat([summed, meaned, torch.log1p(n_atoms)], dim=1)).view(-1)

    model = Net()
    torch.manual_seed(9000 + seed)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -1 / module.in_features ** 0.5,
                                 1 / module.in_features ** 0.5)
    return model.to(device)


def gine_data_lists(raw: Mapping[str, Any], scaler: Mapping[str, float]):
    import sq_data as SD
    import sq_gine as SG

    dims = C.official_dims()
    counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(9)}
    counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(3)})
    counts["degree"] = 17
    counts["ring"] = 14
    return {split: SG.build_data_list(SD.packed_split(raw, split), counts, scaler)
            for split in ("train", "valid")}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_word(model, data: D.SplitData, device: str, scaler: Mapping[str, float],
                  batch_size: int = 512) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    preds, targets, rows = [], [], []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(start + batch_size, len(data)))
            batch = D.to_device(D.pack_batch(data, idx), device)
            out = model(batch)
            preds.append(out["pred"].detach().cpu().numpy())
            targets.append(batch["y"].detach().cpu().numpy())
            rows.append(batch["row_ids"].detach().cpu().numpy())
    return {"pred_std": np.concatenate(preds), "target_std": np.concatenate(targets),
            "row_ids": np.concatenate(rows)}


def evaluate_gine(model, data_list, device: str) -> dict[str, np.ndarray]:
    import torch
    from torch_geometric.loader import DataLoader

    model.eval()
    loader = DataLoader(data_list, batch_size=512, shuffle=False)
    preds, targets, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).detach().cpu().numpy().reshape(-1))
            targets.append(batch.y.detach().cpu().numpy().reshape(-1))
            rows.append(batch.row_id.detach().cpu().numpy().reshape(-1))
    return {"pred_std": np.concatenate(preds), "target_std": np.concatenate(targets),
            "row_ids": np.concatenate(rows)}


def metric_block(res: Mapping[str, np.ndarray], scaler: Mapping[str, float]) -> dict[str, Any]:
    pred = res["pred_std"] * scaler["std"] + scaler["mean"]
    target = res["target_std"] * scaler["std"] + scaler["mean"]
    return {"pred_raw": pred, "target_raw": target, "row_ids": res["row_ids"],
            "rmse_raw": C.rmse(pred, target), "mae_raw": C.mae(pred, target)}


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    return C.results_root() / run_id


def save_checkpoint(path: Path, *, model, optimizer, epoch: int, global_step: int,
                    best: Mapping[str, Any], curves, scaler, rng_numpy, rng_torch,
                    recipe: Recipe, extra: Mapping[str, Any] | None = None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best": dict(best),
        "curves": list(curves),
        "scaler": dict(scaler),
        "rng_numpy": rng_numpy,
        "rng_torch": rng_torch,
        "recipe": recipe.as_dict(),
        "extra": dict(extra or {}),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, *, model, optimizer=None, restore_rng: bool = True):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        if payload.get("rng_numpy") is not None:
            np.random.set_state(payload["rng_numpy"])
        if payload.get("rng_torch") is not None:
            torch.set_rng_state(payload["rng_torch"])
    return payload


def _summary(curves, best, best_preds, final_preds, epochs) -> dict[str, Any]:
    def last(split, what="rmse_raw"):
        values = [row[f"{split}_{what}"] for row in curves if f"{split}_{what}" in row]
        return float(np.mean(values[-20:])) if values else float("nan")

    row = curves[best["epoch"] - 1]
    return {
        "best": {"epoch": int(best["epoch"]), "valid_rmse_raw": float(best["valid"]),
                 "train_rmse_raw": row.get("train_rmse_raw"),
                 "valid_mae_raw": row.get("valid_mae_raw"),
                 "train_mae_raw": row.get("train_mae_raw")},
        "final": {"epoch": int(epochs), "train_rmse_raw": curves[-1].get("train_rmse_raw"),
                  "valid_rmse_raw": curves[-1].get("valid_rmse_raw"),
                  "train_mae_raw": curves[-1].get("train_mae_raw"),
                  "valid_mae_raw": curves[-1].get("valid_mae_raw")},
        "last20": {f"mean_last20_{split}_{what}": last(split, what)
                   for split in ("train", "valid") for what in ("rmse_raw", "mae_raw")},
        "predictions_best": best_preds,
        "predictions_final": final_preds,
    }


def train(*, model, kind: str, splits: Mapping[str, Any], scaler: Mapping[str, float],
          recipe: Recipe, device: str, run_id: str, deadline: float | None = None,
          resume: bool = False, timing: tuple[int, int] | None = None,
          log_every: int = 10) -> dict[str, Any]:
    """One training run.  ``kind`` is ``word`` or ``gine``."""
    import torch

    out_dir = run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "word":
        train_data: D.SplitData = splits["train"]
        n_train = len(train_data)
    else:
        train_data = splits["train"]
        n_train = len(train_data)
    steps_per_epoch = int(np.ceil(n_train / recipe.batch))
    total_steps = recipe.epochs * steps_per_epoch
    warmup_steps = recipe.warmup_epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.peak_lr,
                                  weight_decay=recipe.weight_decay)
    curves: list[dict[str, Any]] = []
    best = {"valid": float("inf"), "epoch": 0}
    best_preds: dict[str, Any] = {}
    final_preds: dict[str, Any] = {}
    start_epoch, global_step = 0, 0
    resumed_from = None
    if resume and (out_dir / "state_last.pt").is_file():
        payload = load_checkpoint(out_dir / "state_last.pt", model=model, optimizer=optimizer)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best = payload["best"]
        curves = list(payload["curves"])
        resumed_from = start_epoch
        print(f"[{run_id}] resumed from epoch {start_epoch}", flush=True)
    orders = epoch_orders(n_train, recipe.epochs, recipe.seed)
    if kind == "gine":
        from torch_geometric.data import Batch

        train_list = train_data
    t0 = time.perf_counter()
    batch_seconds: list[float] = []
    partial = False
    loss = None
    for epoch in range(start_epoch + 1, recipe.epochs + 1):
        model.train()
        order = orders[epoch - 1]
        ep_t0 = time.perf_counter()
        for step in range(steps_per_epoch):
            idx = order[step * recipe.batch:(step + 1) * recipe.batch]
            if len(idx) == 0:
                continue
            optimizer.param_groups[0]["lr"] = lr_at(global_step, warmup_steps, total_steps, recipe)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            if kind == "word":
                raw = D.pack_batch(train_data, idx)
                batch = D.to_device(raw, device)
                y = np.asarray(raw["y"], dtype=np.float32)
                target = torch.as_tensor((y - scaler["mean"]) / scaler["std"], device=device)
                pred = model(batch)["pred"]
            else:
                batch = Batch.from_data_list([train_list[int(i)] for i in idx]).to(device)
                target = batch.y.view(-1)
                pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            optimizer.step()
        batch_seconds.append(time.perf_counter() - ep_t0)
        row: dict[str, Any] = {"epoch": epoch, "train_step_seconds": batch_seconds[-1],
                               "lr": float(optimizer.param_groups[0]["lr"]),
                               "train_loss": float(loss.detach().cpu()) if loss is not None else None}
        if timing is not None:
            row["seconds"] = time.perf_counter() - t0
            curves.append(row)
            print(f"[timing] {run_id} epoch {epoch}/{recipe.epochs} "
                  f"{row['train_step_seconds']:.3f}s", flush=True)
            if deadline is not None and time.time() > deadline:
                partial = True
                break
            continue
        metrics = {}
        for split in ("train", "valid"):
            if kind == "word":
                res = evaluate_word(model, splits[split], device, scaler, recipe.eval_batch)
            else:
                res = evaluate_gine(model, splits[split], device)
            metrics[split] = metric_block(res, scaler)
            row[f"{split}_rmse_raw"] = metrics[split]["rmse_raw"]
            row[f"{split}_mae_raw"] = metrics[split]["mae_raw"]
        row["seconds"] = time.perf_counter() - t0
        curves.append(row)
        if row["valid_rmse_raw"] < best["valid"]:
            best = {"valid": float(row["valid_rmse_raw"]), "epoch": epoch}
            best_preds = copy.deepcopy(metrics)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "valid_rmse_raw": float(row["valid_rmse_raw"])}, out_dir / "state_best.pt")
        final_preds = metrics
        save_checkpoint(out_dir / "state_last.pt", model=model, optimizer=optimizer, epoch=epoch,
                        global_step=global_step, best=best, curves=curves, scaler=scaler,
                        rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                        recipe=recipe)
        if epoch % log_every == 0 or epoch == recipe.epochs:
            print(f"[{run_id}] {epoch}/{recipe.epochs} train={row['train_rmse_raw']:.5f} "
                  f"valid={row['valid_rmse_raw']:.5f} ({row['seconds']:.1f}s)", flush=True)
        if deadline is not None and time.time() > deadline:
            partial = True
            print(f"[{run_id}] deadline reached at epoch {epoch}; saved PARTIAL", flush=True)
            break
    wall = time.perf_counter() - t0
    result: dict[str, Any] = {"curves": curves, "wall_seconds": wall,
                              "steps_per_epoch": steps_per_epoch, "global_step": global_step,
                              "n_parameters": int(sum(p.numel() for p in model.parameters())),
                              "partial": partial, "resumed_from_epoch": resumed_from,
                              "batch_seconds": batch_seconds,
                              "epoch_seconds_including_eval": [c.get("seconds") for c in curves]}
    if timing is not None:
        timed = batch_seconds[timing[0]:]
        result["timing"] = {"warm_epochs": int(timing[0]), "timed_epochs": int(timing[1]),
                            "mean_train_step_seconds": float(np.mean(timed)),
                            "max_train_step_seconds": float(np.max(timed)),
                            "median_train_step_seconds": float(np.median(timed)),
                            "unit": "s/epoch"}
        return result
    torch.save({"model": model.state_dict(), "epoch": len(curves)}, out_dir / "state_final.pt")
    result["final_state_written"] = True
    result.update(_summary(curves, best, best_preds, final_preds, recipe.epochs))
    np.savez(out_dir / "preds_best.npz", train_row=best_preds["train"]["row_ids"],
             train_pred=best_preds["train"]["pred_raw"], valid_row=best_preds["valid"]["row_ids"],
             valid_pred=best_preds["valid"]["pred_raw"])
    np.savez(out_dir / "preds_final.npz", train_row=final_preds["train"]["row_ids"],
             train_pred=final_preds["train"]["pred_raw"], valid_row=final_preds["valid"]["row_ids"],
             valid_pred=final_preds["valid"]["pred_raw"])
    return result


def write_run(run_id: str, payload: Mapping[str, Any]) -> Path:
    path = run_dir(run_id) / "run.json"
    return C.write_json(path, C.jsonable(dict(payload)))
