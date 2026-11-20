"""Fixed training recipe, per-epoch full Train/Val/Test evaluation, artefacts.

The recipe is frozen by ``configs/rd_small_s0.json``.  Nothing here selects an
architecture, a learning rate or a dataset from results; the only decision is
"best Val epoch", which is written out together with every epoch's numbers.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rd_common as RC
import rd_fragment as RF
import rd_model as RM


# ---------------------------------------------------------------------------
# dataset tables
# ---------------------------------------------------------------------------


@dataclass
class Split:
    row_id: np.ndarray
    label_raw: np.ndarray
    label_std: np.ndarray
    mol_ptr: np.ndarray      # per molecule: [start, end) into view arrays
    view_s: np.ndarray
    view_c: np.ndarray
    fallback: np.ndarray     # fragment id for molecules with no cut, else -1


@dataclass
class Dataset:
    name: str
    task: str
    splits: dict[str, Split]
    frag_data: list
    dims: Mapping[str, Any]
    scaler: Mapping[str, float]
    cache_sha256: str
    meta: dict[str, Any] = field(default_factory=dict)


def build_dataset(name: str, task: str) -> Dataset:
    payload = RC.fragment_cache(name)
    scaler = payload["scaler"]
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    splits: dict[str, Split] = {}
    for split in ("train", "valid", "test"):
        labels = np.asarray(payload[f"{split}_mol_label_raw"], dtype=np.float64)
        splits[split] = Split(
            row_id=np.asarray(payload[f"{split}_mol_row_id"], dtype=np.int64),
            label_raw=labels,
            label_std=(labels - mean) / std,
            mol_ptr=np.asarray(payload[f"{split}_mol_view_ptr"], dtype=np.int64),
            view_s=np.asarray(payload[f"{split}_view_s"], dtype=np.int64),
            view_c=np.asarray(payload[f"{split}_view_c"], dtype=np.int64),
            fallback=np.asarray(payload[f"{split}_mol_fallback"], dtype=np.int64),
        )
    frag_data = RM.fragment_data_list(payload)
    meta = {split: payload[f"{split}_meta"] for split in ("train", "valid", "test")}
    return Dataset(
        name=name,
        task=task,
        splits=splits,
        frag_data=frag_data,
        dims=RM.ogb_category_dims(),
        scaler={"mean": mean, "std": std},
        cache_sha256=str(payload["cache_sha256"]),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def fragment_batch(frag_data: Sequence, ids: Sequence[int]):
    from torch_geometric.data import Batch

    return Batch.from_data_list([frag_data[int(i)] for i in ids])


def molecule_scores(
    model: RM.ResponseModel,
    dataset: Dataset,
    split: Split,
    mol_indices: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (F(G) standardised, per-molecule view mean squared error, extras).

    ``extras`` carries the raw per-view scores, their molecule ids and per
    molecule fallback flags so that diagnostics never re-run the encoder.
    """
    unique_ids: list[int] = []
    unique_pos: dict[int, int] = {}
    view_local_s: list[int] = []
    view_local_c: list[int] = []
    view_mol_local: list[int] = []
    fallback_local = np.full(len(mol_indices), -1, dtype=np.int64)
    for j, mol in enumerate(np.asarray(mol_indices, dtype=np.int64).tolist()):
        a, b = int(split.mol_ptr[mol]), int(split.mol_ptr[mol + 1])
        if a == b:
            frag = int(split.fallback[mol])
            if frag < 0:
                raise ValueError(f"molecule {mol} has neither views nor a fallback fragment")
            fallback_local[j] = len(unique_ids)
            if frag not in unique_pos:
                unique_pos[frag] = len(unique_ids)
                unique_ids.append(frag)
            fallback_local[j] = unique_pos[frag]
            continue
        for v in range(a, b):
            for sid in (int(split.view_s[v]), int(split.view_c[v])):
                if sid not in unique_pos:
                    unique_pos[sid] = len(unique_ids)
                    unique_ids.append(sid)
            view_local_s.append(unique_pos[int(split.view_s[v])])
            view_local_c.append(unique_pos[int(split.view_c[v])])
            view_mol_local.append(j)
    batch = fragment_batch(dataset.frag_data, unique_ids).to(device)
    z = model.encoder(batch)
    if view_local_s:
        z_s = z[torch.as_tensor(view_local_s, device=device)]
        z_c = z[torch.as_tensor(view_local_c, device=device)]
        a_head, b_head, q, k, t = model.terms(z_s, z_c)
        scores = b_head + a_head + t
    else:
        scores = z.new_zeros(0)
        a_head = b_head = q = k = t = scores
    fb_mask = fallback_local >= 0
    fb_scores = torch.zeros(len(mol_indices), device=device, dtype=z.dtype)
    if bool(fb_mask.any()):
        idx = torch.as_tensor(fallback_local[fb_mask], device=device, dtype=torch.long)
        fb_scores[fb_mask] = model.fallback_scores(z[idx])
    mol = np.asarray(mol_indices, dtype=np.int64)
    s_np = scores.detach().cpu().numpy().astype(np.float64)
    view_mol_local_np = np.asarray(view_mol_local, dtype=np.int64)
    view_mol_global = mol[view_mol_local_np] if view_mol_local_np.size else np.zeros(0, dtype=np.int64)
    view_mean = np.zeros(len(mol_indices), dtype=np.float64)
    if s_np.size:
        counts = np.bincount(view_mol_local_np, minlength=len(mol_indices)).astype(np.float64)
        sums = np.bincount(view_mol_local_np, weights=s_np, minlength=len(mol_indices))
        has = counts > 0
        view_mean[has] = sums[has] / counts[has]
    fb_np = fb_scores.detach().cpu().numpy().astype(np.float64)
    view_mean[fb_mask] = fb_np[fb_mask]
    extras = {
        "view_score_std": s_np,
        "view_mol_row": split.row_id[view_mol_global] if view_mol_global.size else np.zeros(0, dtype=np.int64),
        "a": a_head.detach().cpu().numpy().astype(np.float64),
        "b": b_head.detach().cpu().numpy().astype(np.float64),
        "q": q.detach().cpu().numpy().astype(np.float64),
        "k": k.detach().cpu().numpy().astype(np.float64),
        "t": t.detach().cpu().numpy().astype(np.float64),
        "view_s_frag": np.asarray([unique_ids[i] for i in view_local_s], dtype=np.int64),
        "view_c_frag": np.asarray([unique_ids[i] for i in view_local_c], dtype=np.int64),
        "n_views_per_mol": np.bincount(view_mol_local_np, minlength=len(mol_indices)) if s_np.size else np.zeros(len(mol_indices), dtype=np.int64),
        "fallback": fb_mask,
    }
    return view_mean, extras


def views_of(
    model: RM.ResponseModel,
    dataset: Dataset,
    split: Split,
    mol_indices: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """Cached per-view quantities for the frozen model (dictionary stage)."""
    unique_ids: list[int] = []
    unique_pos: dict[int, int] = {}
    view_s_id: list[int] = []
    view_c_id: list[int] = []
    view_mol: list[int] = []
    fallback_ids: list[int] = []
    fallback_mol: list[int] = []
    for j, mol in enumerate(np.asarray(mol_indices, dtype=np.int64).tolist()):
        a, b = int(split.mol_ptr[mol]), int(split.mol_ptr[mol + 1])
        if a == b:
            fallback_ids.append(int(split.fallback[mol]))
            fallback_mol.append(j)
            continue
        for v in range(a, b):
            view_s_id.append(int(split.view_s[v]))
            view_c_id.append(int(split.view_c[v]))
            view_mol.append(j)
    need = set(view_s_id) | set(view_c_id) | set(fallback_ids)
    ordered = sorted(need)
    pos = {frag: i for i, frag in enumerate(ordered)}
    with torch.no_grad():
        batch = fragment_batch(dataset.frag_data, ordered).to(device)
        z = model.encoder(batch).cpu().numpy().astype(np.float64)
        if view_s_id:
            z_s = torch.from_numpy(z[[pos[f] for f in view_s_id]]).to(device)
            z_c = torch.from_numpy(z[[pos[f] for f in view_c_id]]).to(device)
            a_head, b_head, q, k, t = model.terms(z_s, z_c)
            scores = (b_head + a_head + t).cpu().numpy().astype(np.float64)
            a_np = a_head.cpu().numpy().astype(np.float64)
            b_np = b_head.cpu().numpy().astype(np.float64)
            q_np = q.cpu().numpy().astype(np.float64)
            k_np = k.cpu().numpy().astype(np.float64)
            t_np = t.cpu().numpy().astype(np.float64)
        else:
            scores = np.zeros(0)
            a_np = b_np = q_np = k_np = t_np = scores
        fb_score = np.zeros(len(fallback_ids))
        if fallback_ids:
            fz = torch.from_numpy(z[[pos[f] for f in fallback_ids]]).to(device)
            fb_score = model.fallback_scores(fz).cpu().numpy().astype(np.float64)
    return {
        "frag_ids": np.asarray(ordered, dtype=np.int64),
        "z": z,
        "view_s_frag": np.asarray(view_s_id, dtype=np.int64),
        "view_c_frag": np.asarray(view_c_id, dtype=np.int64),
        "view_mol_local": np.asarray(view_mol, dtype=np.int64),
        "score": scores,
        "a": a_np,
        "b": b_np,
        "q": q_np,
        "k": k_np,
        "t": t_np,
        "fallback_frag": np.asarray(fallback_ids, dtype=np.int64),
        "fallback_mol_local": np.asarray(fallback_mol, dtype=np.int64),
        "fallback_score": fb_score,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def lr_schedule(config: Mapping[str, Any], steps_per_epoch: int, epochs: int):
    warmup_steps = int(config["warmup_epochs"]) * steps_per_epoch
    total_steps = int(epochs) * steps_per_epoch
    peak = float(config["peak_lr"])
    floor = float(config["min_lr"])
    start = 1e-4

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            frac = step / max(1, warmup_steps)
            return start + (peak - start) * frac
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = float(min(1.0, max(0.0, progress)))
        return floor + 0.5 * (peak - floor) * (1.0 + np.cos(np.pi * progress))

    return lr_at


def train_order(dataset: Dataset, epochs: int, seed: int) -> np.ndarray:
    """Frozen per-epoch permutations of the official Train rows (NumPy RNG)."""
    rng = np.random.default_rng(seed)
    n = len(dataset.splits["train"].row_id)
    return np.stack([rng.permutation(n) for _ in range(epochs)]).astype(np.int64)


def train_arm(
    *,
    dataset: Dataset,
    arm: str,
    config: Mapping[str, Any],
    device: str,
    init_state: Mapping[str, torch.Tensor],
    order: np.ndarray,
    epochs: int,
    log=print,
    eval_batch_size: int = 256,
) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    model_config = RM.ModelConfig(
        hidden=int(config["hidden"]),
        latent=int(config["latent"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        mode=arm,
    )
    model = RM.build_model(dataset.dims, model_config, init_state).to(device)
    model_hash = RM.state_hash(model.state_dict())
    if model_hash != RM.state_hash(init_state):
        raise RuntimeError("model did not load the canonical init state exactly")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["peak_lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(float(b) for b in config["betas"]),
        eps=float(config["eps"]),
    )
    train_split = dataset.splits["train"]
    n_train = len(train_split.row_id)
    batch_size = int(config["batch_size"])
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    lr_at = lr_schedule(config, steps_per_epoch, epochs)
    grad_clip = float(config["grad_clip"])

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse_raw": float("inf"), "epoch": -1}
    best_state: dict[str, torch.Tensor] | None = None
    best_pred: dict[str, Any] = {}
    final_pred: dict[str, Any] = {}
    global_step = 0
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        perm = order[epoch - 1]
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            mol_idx = perm[start:start + batch_size]
            if mol_idx.size == 0:
                continue
            optimizer.param_groups[0]["lr"] = lr_at(global_step)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = molecule_loss(model, dataset, train_split, mol_idx, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "arm": arm,
            "train_loss": epoch_loss / max(1, n_batches),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for split_name in ("train", "valid", "test"):
            pred = evaluate_split(model, dataset, split_name, device, eval_batch_size)
            predictions[split_name] = pred
            row[f"{split_name}_rmse_raw"] = RC.rmse(pred["pred_raw"], pred["target_raw"])
            row[f"{split_name}_mae_raw"] = RC.mae(pred["pred_raw"], pred["target_raw"])
        row["seconds"] = time.perf_counter() - t0
        curves.append(row)
        if row["valid_rmse_raw"] < best["valid_rmse_raw"]:
            best = {"valid_rmse_raw": float(row["valid_rmse_raw"]), "epoch": int(epoch)}
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            best_pred = copy.deepcopy(predictions)
        final_pred = copy.deepcopy(predictions)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            log(
                f"[{dataset.task}:{arm}] epoch {epoch}/{epochs} "
                f"train {row['train_rmse_raw']:.4f} valid {row['valid_rmse_raw']:.4f} "
                f"test {row['test_rmse_raw']:.4f} ({row['seconds']:.1f}s)"
            )
    seconds = time.perf_counter() - t0
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    return {
        "arm": arm,
        "curves": curves,
        "best": best,
        "best_state": best_state,
        "final_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "best_predictions": best_pred,
        "final_predictions": final_pred,
        "seconds": seconds,
        "steps_per_epoch": steps_per_epoch,
        "global_step": global_step,
        "n_parameters": RM.n_parameters(model),
        "init_state_hash": RM.state_hash(init_state),
        "loaded_state_hash": model_hash,
        "eval_batch_size": eval_batch_size,
    }


def molecule_loss(model, dataset: Dataset, split: Split, mol_idx: np.ndarray, device: str):
    """Per-molecule mean of view squared errors, then mean over molecules."""
    unique_ids: list[int] = []
    unique_pos: dict[int, int] = {}
    view_local_s: list[int] = []
    view_local_c: list[int] = []
    view_mol_local: list[int] = []
    fb_local: list[tuple[int, int]] = []
    for j, mol in enumerate(np.asarray(mol_idx, dtype=np.int64).tolist()):
        a, b = int(split.mol_ptr[mol]), int(split.mol_ptr[mol + 1])
        if a == b:
            frag = int(split.fallback[mol])
            if frag not in unique_pos:
                unique_pos[frag] = len(unique_ids)
                unique_ids.append(frag)
            fb_local.append((j, unique_pos[frag]))
            continue
        for v in range(a, b):
            for sid in (int(split.view_s[v]), int(split.view_c[v])):
                if sid not in unique_pos:
                    unique_pos[sid] = len(unique_ids)
                    unique_ids.append(sid)
            view_local_s.append(unique_pos[int(split.view_s[v])])
            view_local_c.append(unique_pos[int(split.view_c[v])])
            view_mol_local.append(j)
    import torch.nn.functional as Fn

    batch = fragment_batch(dataset.frag_data, unique_ids).to(device)
    z = model.encoder(batch)
    target = torch.as_tensor(split.label_std[np.asarray(mol_idx, dtype=np.int64)], dtype=torch.float32, device=device)
    sq = z.new_zeros(len(mol_idx))
    if view_local_s:
        z_s = z[torch.as_tensor(view_local_s, device=device)]
        z_c = z[torch.as_tensor(view_local_c, device=device)]
        scores = model.view_scores(z_s, z_c)
        view_mol = torch.as_tensor(view_mol_local, device=device, dtype=torch.long)
        per_view = (scores - target[view_mol]) ** 2
        sums = torch.zeros(len(mol_idx), device=device, dtype=per_view.dtype).index_add(0, view_mol, per_view)
        counts = torch.zeros(len(mol_idx), device=device, dtype=per_view.dtype).index_add(
            0, view_mol, torch.ones_like(per_view)
        )
        sq = sq + sums / counts.clamp(min=1.0)
    if fb_local:
        rows = torch.as_tensor([r for r, _ in fb_local], device=device, dtype=torch.long)
        cols = torch.as_tensor([c for _, c in fb_local], device=device, dtype=torch.long)
        fb_scores = model.fallback_scores(z[cols])
        sq = sq.index_copy(0, rows, (fb_scores - target[rows]) ** 2)
    return sq.mean()


_EVAL_BATCH_STATE: dict[str, int] = {"effective": 0, "reductions": 0}


def evaluate_split(model, dataset: Dataset, split_name: str, device: str, batch_size: int) -> dict[str, np.ndarray]:
    split = dataset.splits[split_name]
    n = len(split.row_id)
    model.eval()
    preds = np.zeros(n)
    effective = int(batch_size)
    _EVAL_BATCH_STATE.setdefault("effective", effective)
    if _EVAL_BATCH_STATE["effective"]:
        effective = min(effective, int(_EVAL_BATCH_STATE["effective"]))
    with torch.no_grad():
        start = 0
        while start < n:
            idx = np.arange(start, min(n, start + effective))
            try:
                scores, _ = molecule_scores(model, dataset, split, idx, device)
            except RuntimeError as exc:  # CUDA OOM: only the evaluation batch changes
                if "out of memory" not in str(exc).lower() or effective <= 64:
                    raise
                effective //= 2
                _EVAL_BATCH_STATE["effective"] = effective
                _EVAL_BATCH_STATE["reductions"] += 1
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"[eval] OOM: evaluation batch reduced to {effective} molecules", flush=True)
                continue
            preds[idx] = scores
            start += idx.size
    mean = float(dataset.scaler["mean"])
    std = float(dataset.scaler["std"])
    return {
        "pred_std": preds,
        "pred_raw": preds * std + mean,
        "target_raw": split.label_raw,
        "row_ids": split.row_id,
    }
