"""Training loops: WORD_PREFIX, the branch arms, and the GINE_RAW reference.

Rules enforced here (they are part of the round contract, not knobs):

* seed 0 only; python/numpy/torch all seeded, TF32 recorded once.
* every epoch evaluates **Train, Valid and Test** in ``eval()``/``no_grad()``;
  Train RMSE is always recomputed, never the online training loss.
* all reported numbers are back in raw units (one de-standardisation).
* the branch horizon is shared by all arms; no arm-specific early stopping.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Any, Mapping

import numpy as np

import swf_common as C
import swf_model as M

FLOAT_KEYS = frozenset({
    "win_node_mask", "win_edge_mask", "win_adj", "n_atoms", "n_bonds", "word_nframes",
    "row_weight", "sent_ncovers", "rel_mask_u", "rel_mask_v", "rel_e", "rel_c", "rel_c_mask",
})
INT_KEYS = frozenset({
    "atom_code", "bond_code", "win_tmpl", "win_atom_idx", "win_bond_idx", "frame_win",
    "frame_word", "word_tmpl", "word_mol", "frame_mol", "sent_n", "sent_mol", "row_left",
    "row_right", "row_word_u", "row_word_v", "row_sent", "row_rel_uv", "row_rel_vu",
    "row_win_left", "row_win_right", "row_mol", "rel_tmpl_u", "rel_tmpl_v",
})
BRANCH_MODES = {
    "FULL": "full",
    "BAG_CONT": "bag_cont",
    "EARLY_POOL": "early",
    "FLAT_CONTEXT": "flat",
}


def epoch_orders(n: int, epochs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return [rng.permutation(n) for _ in range(int(epochs))]


def to_tensors(pack: Mapping[str, Any], device: str, dtype) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for key, value in pack.items():
        if key in FLOAT_KEYS:
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device)
        elif key in INT_KEYS:
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.long, device=device)
        elif key == "mols":
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.long, device=device)
        else:
            out[key] = value
    return out


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, start: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return start + (peak - start) * frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict(model, cache, mol_ids: np.ndarray, mode: str, device: str, batch: int) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros(len(mol_ids), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(mol_ids), batch):
            chunk = mol_ids[start:start + batch]
            pack = cache.pack(chunk, sentences=(mode != "prefix"))
            b = to_tensors(pack, device, torch.float32)
            y = model(b, mode)
            out[start:start + len(chunk)] = y.double().cpu().numpy()
    return out


def raw_metrics(pred_std: np.ndarray, labels: np.ndarray, scaler: Mapping[str, float]) -> dict[str, float]:
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return {
        "rmse": C.rmse(pred_raw, labels),
        "mae": C.mae(pred_raw, labels),
    }


def predict_raw(pred_std: np.ndarray, scaler: Mapping[str, float]) -> np.ndarray:
    return pred_std * float(scaler["std"]) + float(scaler["mean"])


def _evaluate_all(model, cache, mode, device, batch) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    preds: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for split in C.SPLITS:
        rows = cache.split_rows[split]
        p = predict(model, cache, rows, mode, device, batch)
        preds[split] = p
        metrics[split] = raw_metrics(p, cache.labels[rows], cache.scaler)
    return preds, metrics


def save_predictions(path, cache, preds: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        rows = cache.split_rows[split]
        payload[f"{split}_row_id"] = np.asarray(cache.row_ids[rows], np.int64)
        payload[f"{split}_label"] = np.asarray(cache.labels[rows], np.float64)
        payload[f"{split}_pred"] = np.asarray(predict_raw(preds[split], cache.scaler), np.float64)
    np.savez(path, **payload)
    return C.sha256_file(path)


def _curve_row(run_id, stage_epoch, parent_prefix_epoch, lr, metrics, online_loss, epoch_seconds, cumulative_seconds) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "stage_epoch": int(stage_epoch),
        "parent_prefix_epoch": int(parent_prefix_epoch),
        "lr": float(lr),
    }
    for split in C.SPLITS:
        row[f"{split}_rmse"] = float(metrics[split]["rmse"])
        row[f"{split}_mae"] = float(metrics[split]["mae"])
    row["online_loss"] = float(online_loss) if online_loss is not None else ""
    row["epoch_seconds"] = float(epoch_seconds)
    row["cumulative_seconds"] = float(cumulative_seconds)
    return row


# ---------------------------------------------------------------------------
# WORD_PREFIX (stage A)
# ---------------------------------------------------------------------------


def train_prefix(
    *,
    cfg: Mapping[str, Any],
    cache,
    device: str,
    out_dir,
    horizon: int,
    run_id: str,
    ledger: C.Ledger,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    import torch

    rec = cfg["recipe"]
    pref = rec["prefix"]
    batch = int(rec["batch"])
    eval_batch = int(rec["eval_batch"])
    C.set_deterministic(int(rec["seed"]))
    tf32 = C.enable_tf32(bool(rec["tf32"]))

    model = M.build_model(cfg).to(device)
    support = M.apply_train_support(model, cache.stats)
    M.bind_constants(model, cache.stats)
    init_hash = state_hash(model)

    prefix_names = model.prefix_parameter_names()
    params = [p for name, p in model.named_parameters() if name in prefix_names]
    optimizer = torch.optim.AdamW(params, lr=float(pref["peak_lr"]), weight_decay=float(rec["weight_decay"]))

    train_rows = cache.split_rows["train"]
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch))
    orders = epoch_orders(len(train_rows), horizon, int(rec["order_rng_seed"]))
    warmup_steps = int(pref["warmup_epochs"]) * steps_per_epoch
    total_steps = horizon * steps_per_epoch

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse": float("inf"), "epoch": None}
    es_best = {"valid_rmse": float("inf"), "epoch": 0}
    best_state = None
    best_preds = None
    final_preds = None
    stopped_at = horizon
    global_step = 0
    t_start = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(1, horizon + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        order = orders[epoch - 1]
        t_epoch = time.perf_counter()
        for start in range(0, len(train_rows), batch):
            idx = order[start:start + batch]
            chunk = train_rows[idx]
            pack = cache.pack(chunk, sentences=False)
            b = to_tensors(pack, device, torch.float32)
            optimizer.param_groups[0]["lr"] = lr_at(
                global_step, warmup_steps, total_steps,
                float(pref["peak_lr"]), float(pref["warmup_start_lr"]), float(pref["min_lr"]),
            )
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            y = model(b, "prefix")
            target = torch.as_tensor(
                (cache.labels[chunk] - cache.scaler["mean"]) / cache.scaler["std"],
                dtype=torch.float32, device=device,
            )
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(rec["clip"]))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(chunk)
            n_seen += len(chunk)
        preds, metrics = _evaluate_all(model, cache, "prefix", device, eval_batch)
        epoch_seconds = time.perf_counter() - t_epoch
        epoch_times.append(epoch_seconds)
        cumulative = time.perf_counter() - t_start
        curves.append(_curve_row(
            run_id, epoch, 0, float(optimizer.param_groups[0]["lr"]), metrics,
            loss_sum / max(1, n_seen), epoch_seconds, cumulative,
        ))
        print(
            f"[{run_id}] epoch {epoch}/{horizon} "
            f"train={metrics['train']['rmse']:.4f} valid={metrics['valid']['rmse']:.4f} "
            f"test={metrics['test']['rmse']:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"{epoch_seconds:.2f}s",
            flush=True,
        )
        if metrics["valid"]["rmse"] < best["valid_rmse"]:
            best = {"valid_rmse": metrics["valid"]["rmse"], "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            best_preds = preds
        if metrics["valid"]["rmse"] < es_best["valid_rmse"] - float(pref["min_delta"]):
            es_best = {"valid_rmse": metrics["valid"]["rmse"], "epoch": epoch}
        final_preds = preds
        if deadline_epoch is not None and epoch >= deadline_epoch:
            stopped_at = epoch
            print(f"[{run_id}] stopping at epoch {epoch} (deadline)", flush=True)
            break
        if epoch >= int(pref["early_stop_min_epoch"]):
            if epoch - int(es_best["epoch"]) >= int(pref["patience"]):
                stopped_at = epoch
                print(f"[{run_id}] early stop at epoch {epoch} (es best {es_best['epoch']})", flush=True)
                break
    wall = time.perf_counter() - t_start

    if best_state is None:
        raise RuntimeError("no best prefix state was recorded")
    run = out_dir
    torch.save(model.state_dict(), run / "state_final.pt")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), run / "prefix_best.pt")
    torch.save(best_state, run / "state_best.pt")
    best_pred_sha = save_predictions(run / "predictions_best.npz", cache, best_preds)
    final_pred_sha = save_predictions(run / "predictions_final.npz", cache, final_preds)
    C.write_csv(run / "curves.csv", curves)
    steady = sorted(epoch_times[2:5]) if len(epoch_times) >= 5 else sorted(epoch_times[-3:])
    median_steady = float(np.median(steady)) if steady else float("nan")
    result = {
        "run_id": run_id,
        "arm": "WORD_PREFIX",
        "horizon": int(horizon),
        "stopped_at": int(stopped_at),
        "best_epoch": int(best["epoch"]),
        "best_valid_rmse": float(best["valid_rmse"]),
        "best": {
            "train_rmse": curves[best["epoch"] - 1]["train_rmse"],
            "valid_rmse": curves[best["epoch"] - 1]["valid_rmse"],
            "test_rmse": curves[best["epoch"] - 1]["test_rmse"],
            "train_mae": curves[best["epoch"] - 1]["train_mae"],
            "valid_mae": curves[best["epoch"] - 1]["valid_mae"],
            "test_mae": curves[best["epoch"] - 1]["test_mae"],
        },
        "final": {
            "epoch": int(curves[-1]["stage_epoch"]),
            "train_rmse": curves[-1]["train_rmse"],
            "valid_rmse": curves[-1]["valid_rmse"],
            "test_rmse": curves[-1]["test_rmse"],
        },
        "steady_epoch_seconds_median": median_steady,
        "epoch_seconds": epoch_times,
        "wall_seconds": float(wall),
        "steps_per_epoch": int(steps_per_epoch),
        "global_step": int(global_step),
        "parameters": model.parameter_ledger(),
        "init_state_hash": init_hash,
        "train_support": support,
        "tf32": tf32,
        "predictions_best_sha256": best_pred_sha,
        "predictions_final_sha256": final_pred_sha,
        "state_hashes": {
            "prefix_best": C.sha256_file(run / "prefix_best.pt"),
            "state_best": C.sha256_file(run / "state_best.pt"),
            "state_final": C.sha256_file(run / "state_final.pt"),
        },
        "max_vram_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0,
    }
    return result


# ---------------------------------------------------------------------------
# branch arms (stage B / C)
# ---------------------------------------------------------------------------


def build_branch_init(*, cfg, cache, device: str, prefix_state, out_dir) -> dict[str, Any]:
    """Deterministic shared branch initialisation (identical for every arm)."""
    import torch

    C.set_deterministic(int(cfg["recipe"]["seed"]))
    model = M.build_model(cfg)
    support = M.apply_train_support(model, cache.stats)
    M.bind_constants(model, cache.stats)
    loaded = M.load_prefix_into(model, prefix_state)
    state = model.state_dict()
    torch.save(state, out_dir / "branch_init.pt")
    return {
        "support": support,
        "prefix_keys_loaded": loaded,
        "sha256": C.sha256_file(out_dir / "branch_init.pt"),
        "state_hash": state_hash(model),
    }


def train_branch(
    *,
    cfg: Mapping[str, Any],
    cache,
    device: str,
    out_dir,
    arm: str,
    epochs: int,
    run_id: str,
    ledger: C.Ledger,
    prefix_record: Mapping[str, Any],
    parent_prefix_epoch: int,
    init_ckpt,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    import torch

    if arm not in BRANCH_MODES:
        raise ValueError(f"unknown arm {arm}")
    mode = BRANCH_MODES[arm]
    rec = cfg["recipe"]
    br = rec["branch"]
    batch = int(rec["batch"])
    eval_batch = int(rec["eval_batch"])
    C.set_deterministic(int(rec["seed"]))
    tf32 = C.enable_tf32(bool(rec["tf32"]))

    C.set_deterministic(int(rec["seed"]))
    model = M.build_model(cfg)
    M.apply_train_support(model, cache.stats)
    M.bind_constants(model, cache.stats)
    model.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
    model = model.to(device)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(br["peak_lr"]), weight_decay=float(rec["weight_decay"]))
    train_rows = cache.split_rows["train"]
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch))
    orders = epoch_orders(len(train_rows), epochs, int(rec["order_rng_seed"]))
    warmup_steps = int(br["warmup_epochs"]) * steps_per_epoch
    total_steps = max(1, epochs * steps_per_epoch)

    curves: list[dict[str, Any]] = []
    # epoch 0: the initial (prefix) model, included in the best-Valid candidates
    init_preds, init_metrics = _evaluate_all(model, cache, mode, device, eval_batch)
    curves.append(_curve_row(run_id, 0, parent_prefix_epoch, 0.0, init_metrics, None, 0.0, 0.0))
    print(
        f"[{run_id}] epoch 0 (init) train={init_metrics['train']['rmse']:.4f} "
        f"valid={init_metrics['valid']['rmse']:.4f} test={init_metrics['test']['rmse']:.4f}",
        flush=True,
    )
    best = {"valid_rmse": init_metrics["valid"]["rmse"], "epoch": 0}
    best_state = copy.deepcopy(model.state_dict())
    best_preds = init_preds
    final_preds = init_preds
    stopped_at = 0
    global_step = 0
    t_start = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        order = orders[epoch - 1]
        t_epoch = time.perf_counter()
        for start in range(0, len(train_rows), batch):
            idx = order[start:start + batch]
            chunk = train_rows[idx]
            pack = cache.pack(chunk, sentences=True)
            b = to_tensors(pack, device, torch.float32)
            optimizer.param_groups[0]["lr"] = lr_at(
                global_step, warmup_steps, total_steps,
                float(br["peak_lr"]), float(br["warmup_start_lr"]), float(br["min_lr"]),
            )
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            y = model(b, mode)
            target = torch.as_tensor(
                (cache.labels[chunk] - cache.scaler["mean"]) / cache.scaler["std"],
                dtype=torch.float32, device=device,
            )
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(rec["clip"]))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(chunk)
            n_seen += len(chunk)
        preds, metrics = _evaluate_all(model, cache, mode, device, eval_batch)
        epoch_seconds = time.perf_counter() - t_epoch
        epoch_times.append(epoch_seconds)
        curves.append(_curve_row(
            run_id, epoch, parent_prefix_epoch, float(optimizer.param_groups[0]["lr"]),
            metrics, loss_sum / max(1, n_seen), epoch_seconds, time.perf_counter() - t_start,
        ))
        print(
            f"[{run_id}] epoch {epoch}/{epochs} "
            f"train={metrics['train']['rmse']:.4f} valid={metrics['valid']['rmse']:.4f} "
            f"test={metrics['test']['rmse']:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"{epoch_seconds:.2f}s",
            flush=True,
        )
        if metrics["valid"]["rmse"] < best["valid_rmse"]:
            best = {"valid_rmse": metrics["valid"]["rmse"], "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            best_preds = preds
        final_preds = preds
        stopped_at = epoch
        if deadline_epoch is not None and epoch >= deadline_epoch:
            print(f"[{run_id}] stopping at epoch {epoch} (deadline)", flush=True)
            break
    wall = time.perf_counter() - t_start

    run = out_dir
    torch.save(model.state_dict(), run / "state_final.pt")
    model.load_state_dict(best_state)
    torch.save(best_state, run / "state_best.pt")
    best_pred_sha = save_predictions(run / "predictions_best.npz", cache, best_preds)
    final_pred_sha = save_predictions(run / "predictions_final.npz", cache, final_preds)
    C.write_csv(run / "curves.csv", curves)
    steady = sorted(epoch_times[2:5]) if len(epoch_times) >= 5 else sorted(epoch_times[-3:])
    median_steady = float(np.median(steady)) if steady else float("nan")
    return {
        "run_id": run_id,
        "arm": arm,
        "mode": mode,
        "epochs": int(epochs),
        "stopped_at": int(stopped_at),
        "parent_prefix_epoch": int(parent_prefix_epoch),
        "parent_prefix_best_valid_rmse": float(prefix_record["best_valid_rmse"]),
        "best_epoch": int(best["epoch"]),
        "best_valid_rmse": float(best["valid_rmse"]),
        "best": {k: curves[best["epoch"]][k] for k in (
            "train_rmse", "valid_rmse", "test_rmse", "train_mae", "valid_mae", "test_mae")},
        "final": {
            "epoch": int(curves[-1]["stage_epoch"]),
            "train_rmse": curves[-1]["train_rmse"],
            "valid_rmse": curves[-1]["valid_rmse"],
            "test_rmse": curves[-1]["test_rmse"],
            "train_mae": curves[-1]["train_mae"],
            "valid_mae": curves[-1]["valid_mae"],
            "test_mae": curves[-1]["test_mae"],
        },
        "steady_epoch_seconds_median": median_steady,
        "epoch_seconds": epoch_times,
        "wall_seconds": float(wall),
        "steps_per_epoch": int(steps_per_epoch),
        "global_step": int(global_step),
        "parameters": model.parameter_ledger(),
        "tf32": tf32,
        "predictions_best_sha256": best_pred_sha,
        "predictions_final_sha256": final_pred_sha,
        "state_hashes": {
            "state_best": C.sha256_file(run / "state_best.pt"),
            "state_final": C.sha256_file(run / "state_final.pt"),
        },
        "max_vram_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0,
    }


# ---------------------------------------------------------------------------
# GINE_RAW_REF (external performance reference; own weights, own recipe)
# ---------------------------------------------------------------------------


def _gine_splits(cfg: Mapping[str, Any]):
    import sq_data as SD
    import sq_gine as SG
    import wp_common as WPC
    import swf_data as D

    raw = D.load_raw_cache(cfg)
    train_idx = np.asarray(raw["train_idx"], np.int64)
    scaler = C.train_target_scaler(np.asarray(raw["labels"], np.float64)[train_idx])
    dims = WPC.official_dims()
    counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(C.ATOM_COLUMNS)}
    counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(C.BOND_COLUMNS)})
    counts["degree"] = 17
    counts["ring"] = 14
    data = {split: SG.build_data_list(SD.packed_split(raw, split), counts, scaler) for split in C.SPLITS}
    return data, scaler, counts


def _gine_eval(model, dataset, device: str, batch: int, scaler: Mapping[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    import torch
    from torch_geometric.data import Batch

    model.eval()
    preds = np.zeros(len(dataset), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(dataset), batch):
            chunk = dataset[start:start + batch]
            b = Batch.from_data_list(chunk).to(device)
            out = model(b).detach().cpu().numpy().reshape(-1)
            preds[start:start + len(chunk)] = out
    target = np.asarray([float(d.y.reshape(-1)[0]) for d in dataset], np.float64)
    raw_pred = preds * float(scaler["std"]) + float(scaler["mean"])
    raw_true = target * float(scaler["std"]) + float(scaler["mean"])
    return preds, {"rmse": C.rmse(raw_pred, raw_true), "mae": C.mae(raw_pred, raw_true)}


def train_gine_ref(
    *,
    cfg: Mapping[str, Any],
    device: str,
    out_dir,
    run_id: str,
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    import torch
    from torch_geometric.data import Batch

    import wp_train as WT

    gref = cfg["gine_ref"]
    rec = cfg["recipe"]
    C.set_deterministic(int(rec["seed"]))
    tf32 = C.enable_tf32(bool(rec["tf32"]))
    data, scaler, counts = _gine_splits(cfg)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = WT.build_gine_model({}, seed=0, device=device, raw_features=bool(gref["raw_features"]))
    init_hash = state_hash(model)

    batch = int(gref["batch"])
    eval_batch = int(gref["eval_batch"])
    epochs = int(gref["max_epochs"])
    train_list = data["train"]
    steps_per_epoch = max(1, math.ceil(len(train_list) / batch))
    orders = epoch_orders(len(train_list), epochs, int(rec["order_rng_seed"]))
    warmup_steps = int(gref["warmup_epochs"]) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(gref["peak_lr"]), weight_decay=float(gref["weight_decay"])
    )

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse": float("inf"), "epoch": None}
    es_best = {"valid_rmse": float("inf"), "epoch": 0}
    best_state = None
    best_preds = None
    final_preds = None
    global_step = 0
    stopped_at = 0
    t_start = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = orders[epoch - 1]
        loss_sum = 0.0
        n_seen = 0
        t_epoch = time.perf_counter()
        for start in range(0, len(train_list), batch):
            idx = order[start:start + batch]
            b = Batch.from_data_list([train_list[int(i)] for i in idx]).to(device)
            optimizer.param_groups[0]["lr"] = lr_at(
                global_step, warmup_steps, total_steps,
                float(gref["peak_lr"]), float(gref["peak_lr"]) / 10.0, float(gref["min_lr"]),
            )
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            out = model(b)
            loss = torch.nn.functional.mse_loss(out, b.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gref["clip"]))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(b.num_graphs)
            n_seen += int(b.num_graphs)
        preds: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, float]] = {}
        for split in C.SPLITS:
            p, m = _gine_eval(model, data[split], device, eval_batch, scaler)
            preds[split] = p
            metrics[split] = m
        epoch_seconds = time.perf_counter() - t_epoch
        epoch_times.append(epoch_seconds)
        curves.append(_curve_row(
            run_id, epoch, 0, float(optimizer.param_groups[0]["lr"]), metrics,
            loss_sum / max(1, n_seen), epoch_seconds, time.perf_counter() - t_start,
        ))
        print(
            f"[{run_id}] epoch {epoch}/{epochs} train={metrics['train']['rmse']:.4f} "
            f"valid={metrics['valid']['rmse']:.4f} test={metrics['test']['rmse']:.4f} "
            f"{epoch_seconds:.2f}s",
            flush=True,
        )
        if metrics["valid"]["rmse"] < best["valid_rmse"]:
            best = {"valid_rmse": metrics["valid"]["rmse"], "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            best_preds = preds
        if metrics["valid"]["rmse"] < es_best["valid_rmse"] - float(gref["min_delta"]):
            es_best = {"valid_rmse": metrics["valid"]["rmse"], "epoch": epoch}
        final_preds = preds
        stopped_at = epoch
        if deadline_epoch is not None and epoch >= deadline_epoch:
            print(f"[{run_id}] stopping at epoch {epoch} (deadline)", flush=True)
            break
        if epoch >= int(gref["early_stop_min_epoch"]):
            if epoch - int(es_best["epoch"]) >= int(gref["patience"]):
                print(f"[{run_id}] early stop at epoch {epoch} (es best {es_best['epoch']})", flush=True)
                break
    wall = time.perf_counter() - t_start

    run = out_dir
    torch.save(model.state_dict(), run / "state_final.pt")
    model.load_state_dict(best_state)
    torch.save(best_state, run / "state_best.pt")

    def _save(path, pred_std):
        payload: dict[str, np.ndarray] = {}
        for split in C.SPLITS:
            rows = np.asarray([int(d.row_id.reshape(-1)[0]) for d in data[split]], np.int64)
            truth = np.asarray([float(d.y.reshape(-1)[0]) for d in data[split]], np.float64)
            payload[f"{split}_row_id"] = rows
            payload[f"{split}_label"] = truth * float(scaler["std"]) + float(scaler["mean"])
            payload[f"{split}_pred"] = predict_raw(pred_std[split], scaler)
        np.savez(path, **payload)
        return C.sha256_file(path)

    best_sha = _save(run / "predictions_best.npz", best_preds)
    final_sha = _save(run / "predictions_final.npz", final_preds)
    C.write_csv(run / "curves.csv", curves)
    steady = sorted(epoch_times[2:5]) if len(epoch_times) >= 5 else sorted(epoch_times[-3:])
    n_params = int(sum(p.numel() for p in model.parameters()))
    return {
        "run_id": run_id,
        "arm": "GINE_RAW_REF",
        "epochs": int(epochs),
        "stopped_at": int(stopped_at),
        "best_epoch": int(best["epoch"]),
        "best_valid_rmse": float(best["valid_rmse"]),
        "best": {k: curves[best["epoch"] - 1][k] for k in (
            "train_rmse", "valid_rmse", "test_rmse", "train_mae", "valid_mae", "test_mae")},
        "final": {
            "epoch": int(curves[-1]["stage_epoch"]),
            "train_rmse": curves[-1]["train_rmse"],
            "valid_rmse": curves[-1]["valid_rmse"],
            "test_rmse": curves[-1]["test_rmse"],
        },
        "steady_epoch_seconds_median": float(np.median(steady)) if steady else float("nan"),
        "wall_seconds": float(wall),
        "steps_per_epoch": int(steps_per_epoch),
        "global_step": int(global_step),
        "n_parameters": n_params,
        "category_counts": counts,
        "scaler": dict(scaler),
        "init_state_hash": init_hash,
        "tf32": tf32,
        "predictions_best_sha256": best_sha,
        "predictions_final_sha256": final_sha,
        "raw_features": bool(gref["raw_features"]),
        "factory": str(cfg.get("gine_factory_source")),
        "max_vram_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0,
    }
