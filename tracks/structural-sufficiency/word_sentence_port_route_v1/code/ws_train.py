"""Unified training recipe, evaluation and run bookkeeping.

Every arm uses the same recipe R0: batch 128 molecules, MSE on the standardised
whole-molecule target, AdamW with 5-epoch linear warmup (1e-4 -> peak 1e-3) and
per-optimizer-step cosine decay to 1e-5, weight decay 1e-5, grad clip 5, FP32,
TF32 off, no AMP/EMA/SWA/pretraining.  Train and Valid are evaluated every
epoch; **no Test loader is ever built here**.
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ws_common as C

CACHE_VERSION = "ws-word-route-v1"


# ---------------------------------------------------------------------------
# prepared artefacts
# ---------------------------------------------------------------------------


def cache_path(dataset: str) -> Path:
    return C.data_dir() / f"{dataset}_words_{CACHE_VERSION}.npz"


def freeze_path(dataset: str) -> Path:
    """Host-local freeze record written by ``prepare`` (git-ignored)."""
    return C.data_dir() / f"{dataset}_words_{CACHE_VERSION}_freeze.json"


def record_path(dataset: str) -> Path:
    """Committed copy of the frozen structure record (written only with --write-record)."""
    return C.analysis_dir() / f"structure_freeze_{dataset.replace('ogbg-', '')}.json"


def content_sha256(payload: Mapping[str, Any]) -> str:
    """Machine-independent identity of the prepared arrays (zip timestamps excluded)."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(k for k in payload if isinstance(payload[k], np.ndarray)):
        array = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def record_path_for(dataset: str) -> Path:
    return record_path(dataset)


def load_prepared(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = cache_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"prepared cache missing: {path}")
    freeze = C.read_json(freeze_path(dataset))
    record_path = record_path_for(dataset)
    if record_path.is_file():
        record = C.read_json(record_path)
        if record.get("cache_sha256") != freeze.get("cache_sha256"):
            raise ValueError("host freeze disagrees with the committed structure record")
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["file_sha256"] = C.sha256_file(path)
    payload["cache_sha256"] = content_sha256(payload)
    if payload["cache_sha256"] != freeze["cache_sha256"]:
        raise ValueError(f"prepared content hash mismatch: {payload['cache_sha256']} "
                         f"!= {freeze['cache_sha256']}")
    return payload, freeze


def split_data(payload: Mapping[str, Any], partition: str, split: str, *,
               with_hop: bool = False):
    import ws_model as M

    prefix = f"{partition}/{split}/"
    arrays = {k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)}
    data = M.SplitData(
        name=split,
        row_ids=np.asarray(payload[f"row_ids/{split}"]),
        labels=np.asarray(payload[f"labels/{split}"], dtype=np.float64),
        graph_n_atoms=np.asarray(payload[f"base/{split}/graph_n_atoms"]),
        graph_n_bonds=np.asarray(payload[f"base/{split}/graph_n_bonds"]),
        atom_feat=np.asarray(payload[f"base/{split}/atom_feat"], dtype=np.int64),
        bond_feat=np.asarray(payload[f"base/{split}/bond_feat"], dtype=np.int64),
        arrays=arrays,
    )
    if with_hop:
        data.hop = M.build_hop_index(arrays["graph_word_ptr"], arrays["inc_ptr"],
                                     arrays["inc_other"])
    return data


# ---------------------------------------------------------------------------
# recipe and schedule
# ---------------------------------------------------------------------------


@dataclass
class Recipe:
    epochs: int
    batch: int = 128
    peak_lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-4
    min_lr: float = 1e-5
    clip: float = 5.0
    dataloader_seed: int = 20261120

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def lr_at(step: int, warmup_steps: int, total_steps: int, recipe: Recipe) -> float:
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return recipe.warmup_start_lr + (recipe.peak_lr - recipe.warmup_start_lr) * frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return recipe.min_lr + 0.5 * (recipe.peak_lr - recipe.min_lr) * (1.0 + np.cos(np.pi * progress))


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def body_hash(model, prefix_skip: Sequence[str] = ()) -> str:
    """Hash of the shared sentence body only (excludes the content encoder)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("encoder."):
            continue
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def make_model(spec, payload: Mapping[str, Any], *, seed: int, device: str):
    import torch

    import ws_model as M

    dims = C.official_dims()
    t = {k[len("tmpl/"):]: v for k, v in payload.items() if k.startswith("tmpl/")}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if spec.encoder == "gine_ref":
        model = M.GINERefModel(M.gine_categories(dims))
    else:
        model = M.WordSentenceModel(spec, t, dims, int(dims["bond"][0]))
    return model.to(device)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_word(model, data, device, scaler, *, extra: bool = False, hop: bool = False,
                  batch_size: int = 512):
    import torch

    import ws_model as M

    model.eval()
    preds, targets, rows = [], [], []
    with torch.no_grad():
        for start in range(0, len(data.row_ids), batch_size):
            idx = np.arange(start, min(start + batch_size, len(data.row_ids)))
            batch = M.to_device(M.pack_batch(data, idx, attrpair=extra, hop=hop), device)
            out = model(batch)
            preds.append(out["pred"].detach().cpu().numpy())
            targets.append((np.asarray(batch["y"]) - scaler["mean"]) / scaler["std"])
            rows.append(np.asarray(batch["row_ids"]))
    return {"pred_std": np.concatenate(preds), "target_std": np.concatenate(targets),
            "row_ids": np.concatenate(rows)}


def evaluate_gine(model, loader, device):
    import torch

    model.eval()
    preds, targets, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            targets.append(batch.y.detach().cpu().numpy().reshape(-1))
            rows.append(batch.row_id.detach().cpu().numpy().reshape(-1))
    return {"pred_std": np.concatenate(preds), "target_std": np.concatenate(targets),
            "row_ids": np.concatenate(rows)}


# ---------------------------------------------------------------------------
# GINE reference plumbing
# ---------------------------------------------------------------------------


def build_gine_splits(raw: Mapping[str, Any], scaler: Mapping[str, float], recipe: Recipe,
                      seed: int):
    import sq_data as SD
    import sq_gine as SG
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader

    dims = C.official_dims()
    counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(9)}
    counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(3)})
    counts["degree"] = 17
    counts["ring"] = 14
    data = {split: SG.build_data_list(SD.packed_split(raw, split), counts, scaler)
            for split in ("train", "valid")}
    n = len(data["train"])
    rng = np.random.default_rng(recipe.dataloader_seed + seed)
    order = rng.permutation(n)
    batches = [Batch.from_data_list([data["train"][int(i)] for i in order[start:start + recipe.batch]])
               for start in range(0, n, recipe.batch)]
    return {"train_batches": batches,
            "eval_train": DataLoader(data["train"], batch_size=512, shuffle=False),
            "eval_valid": DataLoader(data["valid"], batch_size=512, shuffle=False)}


# ---------------------------------------------------------------------------
# training loops
# ---------------------------------------------------------------------------


def _metric_block(res: Mapping[str, np.ndarray], scaler: Mapping[str, float]) -> dict[str, Any]:
    pred_raw = res["pred_std"] * scaler["std"] + scaler["mean"]
    target_raw = res["target_std"] * scaler["std"] + scaler["mean"]
    return {
        "pred_raw": pred_raw,
        "target_raw": target_raw,
        "row_ids": res["row_ids"],
        "rmse_raw": C.rmse(pred_raw, target_raw),
        "mae_raw": C.mae(pred_raw, target_raw),
    }


def train_word(*, model, splits: Mapping[str, Any], scaler: Mapping[str, float], recipe: Recipe,
               seed: int, device: str, extra: bool = False, hop: bool = False,
               timing: tuple[int, int] | None = None, log_prefix: str = "") -> dict[str, Any]:
    import torch

    import ws_model as M

    train_data = splits["train"]
    n_train = len(train_data.row_ids)
    rng = np.random.default_rng(recipe.dataloader_seed + seed)
    orders = [rng.permutation(n_train) for _ in range(recipe.epochs)]
    steps_per_epoch = max(1, int(np.ceil(n_train / recipe.batch)))
    total_steps = recipe.epochs * steps_per_epoch
    warmup_steps = recipe.warmup_epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.peak_lr,
                                  weight_decay=recipe.weight_decay)
    curves: list[dict[str, Any]] = []
    best = {"valid": float("inf"), "epoch": None}
    best_state = None
    best_preds: dict[str, Any] = {}
    final_preds: dict[str, Any] = {}
    global_step = 0
    t0 = time.perf_counter()
    for epoch in range(1, recipe.epochs + 1):
        model.train()
        step_t0 = time.perf_counter()
        for step in range(steps_per_epoch):
            idx = orders[epoch - 1][step * recipe.batch:(step + 1) * recipe.batch]
            if len(idx) == 0:
                continue
            optimizer.param_groups[0]["lr"] = lr_at(global_step, warmup_steps, total_steps, recipe)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            raw = M.pack_batch(train_data, idx, attrpair=extra, hop=hop)
            batch = M.to_device(raw, device)
            out = model(batch)
            target = torch.as_tensor(
                (np.asarray(raw["y"], dtype=np.float32) - scaler["mean"]) / scaler["std"],
                device=device)
            loss = torch.nn.functional.mse_loss(out["pred"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            optimizer.step()
        row: dict[str, Any] = {"epoch": epoch,
                               "train_step_seconds": time.perf_counter() - step_t0,
                               "lr": float(optimizer.param_groups[0]["lr"]),
                               "train_loss": float(loss.detach().cpu()) if steps_per_epoch else None}
        if timing is not None:
            row["seconds"] = time.perf_counter() - t0
            curves.append(row)
            print(f"[timing] {log_prefix} epoch {epoch}/{recipe.epochs} "
                  f"{row['train_step_seconds']:.3f}s", flush=True)
            continue
        metrics = {}
        for split in ("train", "valid"):
            res = evaluate_word(model, splits[split], device, scaler, extra=extra, hop=hop)
            metrics[split] = _metric_block(res, scaler)
            row[f"{split}_rmse_raw"] = metrics[split]["rmse_raw"]
            row[f"{split}_mae_raw"] = metrics[split]["mae_raw"]
        row["seconds"] = time.perf_counter() - t0
        curves.append(row)
        if row["valid_rmse_raw"] < best["valid"]:
            best = {"valid": float(row["valid_rmse_raw"]), "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            best_preds = copy.deepcopy(metrics)
        final_preds = metrics
        if epoch % 10 == 0 or epoch == recipe.epochs:
            print(f"[{log_prefix}] {epoch}/{recipe.epochs} "
                  f"train={row['train_rmse_raw']:.5f} valid={row['valid_rmse_raw']:.5f} "
                  f"({row['seconds']:.1f}s)", flush=True)
    result: dict[str, Any] = {"curves": curves, "seconds": time.perf_counter() - t0,
                              "steps_per_epoch": steps_per_epoch, "global_step": global_step,
                              "n_parameters": int(sum(p.numel() for p in model.parameters())),
                              "first_epoch_row_ids": orders[0][:recipe.batch].tolist()}
    if timing is not None:
        timed = [row["train_step_seconds"] for row in curves[timing[0]:]]
        result["timing"] = {"warm_epochs": timing[0], "timed_epochs": timing[1],
                            "mean_train_step_seconds": float(np.mean(timed)),
                            "max_train_step_seconds": float(np.max(timed))}
        return result
    if best_state is not None:
        model.load_state_dict(best_state)
    result.update(_summarise(curves, best, best_preds, final_preds, recipe.epochs))
    return result


def train_gine(*, model, gine: Mapping[str, Any], scaler: Mapping[str, float], recipe: Recipe,
               timing: tuple[int, int] | None = None, log_prefix: str = "") -> dict[str, Any]:
    import torch

    batches = gine["train_batches"]
    steps_per_epoch = len(batches)
    total_steps = recipe.epochs * steps_per_epoch
    warmup_steps = recipe.warmup_epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.peak_lr,
                                  weight_decay=recipe.weight_decay)
    device = next(model.parameters()).device
    curves: list[dict[str, Any]] = []
    best = {"valid": float("inf"), "epoch": None}
    best_state = None
    best_preds: dict[str, Any] = {}
    final_preds: dict[str, Any] = {}
    global_step = 0
    t0 = time.perf_counter()
    for epoch in range(1, recipe.epochs + 1):
        model.train()
        step_t0 = time.perf_counter()
        for step, batch in enumerate(batches):
            batch = batch.to(device)
            optimizer.param_groups[0]["lr"] = lr_at(global_step, warmup_steps, total_steps, recipe)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = torch.nn.functional.mse_loss(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            optimizer.step()
        row: dict[str, Any] = {"epoch": epoch,
                               "train_step_seconds": time.perf_counter() - step_t0,
                               "lr": float(optimizer.param_groups[0]["lr"]),
                               "train_loss": float(loss.detach().cpu())}
        if timing is not None:
            row["seconds"] = time.perf_counter() - t0
            curves.append(row)
            print(f"[timing] {log_prefix} epoch {epoch}/{recipe.epochs} "
                  f"{row['train_step_seconds']:.3f}s", flush=True)
            continue
        metrics = {}
        for split in ("train", "valid"):
            res = evaluate_gine(model, gine[f"eval_{split}"], device)
            metrics[split] = _metric_block(res, scaler)
            row[f"{split}_rmse_raw"] = metrics[split]["rmse_raw"]
            row[f"{split}_mae_raw"] = metrics[split]["mae_raw"]
        row["seconds"] = time.perf_counter() - t0
        curves.append(row)
        if row["valid_rmse_raw"] < best["valid"]:
            best = {"valid": float(row["valid_rmse_raw"]), "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            best_preds = copy.deepcopy(metrics)
        final_preds = metrics
        if epoch % 10 == 0 or epoch == recipe.epochs:
            print(f"[{log_prefix}] {epoch}/{recipe.epochs} "
                  f"train={row['train_rmse_raw']:.5f} valid={row['valid_rmse_raw']:.5f} "
                  f"({row['seconds']:.1f}s)", flush=True)
    result: dict[str, Any] = {"curves": curves, "seconds": time.perf_counter() - t0,
                              "steps_per_epoch": steps_per_epoch, "global_step": global_step,
                              "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    if timing is not None:
        timed = [row["train_step_seconds"] for row in curves[timing[0]:]]
        result["timing"] = {"warm_epochs": timing[0], "timed_epochs": timing[1],
                            "mean_train_step_seconds": float(np.mean(timed)),
                            "max_train_step_seconds": float(np.max(timed))}
        return result
    if best_state is not None:
        model.load_state_dict(best_state)
    result.update(_summarise(curves, best, best_preds, final_preds, recipe.epochs))
    return result


def _summarise(curves, best, best_preds, final_preds, epochs) -> dict[str, Any]:
    def _last(split, what="rmse_raw"):
        values = [row[f"{split}_{what}"] for row in curves if f"{split}_{what}" in row]
        return float(np.mean(values[-20:])) if values else float("nan")

    best_epoch = best["epoch"]
    row = curves[best_epoch - 1]
    return {
        "best": {"epoch": best_epoch, "valid_rmse_raw": best["valid"],
                 "train_rmse_raw": row.get("train_rmse_raw"),
                 "train_mae_raw": row.get("train_mae_raw"),
                 "valid_mae_raw": row.get("valid_mae_raw")},
        "final": {"epoch": epochs, "train_rmse_raw": curves[-1].get("train_rmse_raw"),
                  "valid_rmse_raw": curves[-1].get("valid_rmse_raw"),
                  "train_mae_raw": curves[-1].get("train_mae_raw"),
                  "valid_mae_raw": curves[-1].get("valid_mae_raw")},
        "last20": {f"mean_last20_{split}_{what}": _last(split, what)
                   for split in ("train", "valid") for what in ("rmse_raw", "mae_raw")},
        "predictions_best": best_preds,
        "predictions_final": final_preds,
    }


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    return C.results_root() / run_id


def write_run(run_id: str, payload: Mapping[str, Any]) -> Path:
    return C.write_json(run_dir(run_id) / "run.json", payload)


def allocate_run_id(dataset: str, arm: str, seed: int, tag: str = "") -> str:
    parts = [dataset, arm, f"s{seed}"]
    if tag:
        parts.append(tag)
    return "_".join(parts)
