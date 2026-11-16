"""Performance reference L: the repository-native C1 / ``NO_BOND_MEAN`` model.

C1 is the frozen LCRC-A local-contribution model with the 96-d bond-mean block
removed (``VARIANT_DEFS["C1"]`` in
``tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py``).
It is reused read-only here: the ``lipo_v1`` FeatureSpec, the molecular-graph
builder, the cache/collate layer and the dissection model definition are
imported, never modified.  No supervised checkpoint and no SSL weights are
loaded: the model is trained from scratch on **this round's** official OGB
indices with the frozen R3 recipe.

Frozen R3 (from ``dissect_common.RECIPE``): batch 128, 360 epochs, AdamW peak
3e-4 -> min 1e-5 cosine per optimizer step, wd 1e-3, dropout 0.10 on the atom
level ``h``, grad clip 5, standardised MSE, no soup / EMA / early stopping and
no warmup.

The reference may use richer structural inputs than the query model, so it is a
*performance* reference only; no difference is attributed to a single operator.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import sq_common as C
from sq_common import mae, rmse

PREP_DIR = C.REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = C.REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"

_PATHS_READY = False


def _prepare_imports() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(PREP_CODE), str(DISSECT_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def _modules():
    _prepare_imports()
    import adapters as PREP_ADAPTERS
    import dissect_common as DC
    import dissect_model as DM
    import feature_spec as FS
    import lcrc_data as PREP_DATA

    return {
        "adapters": PREP_ADAPTERS,
        "dissect_common": DC,
        "dissect_model": DM,
        "feature_spec": FS,
        "lcrc_data": PREP_DATA,
    }


def spec_path():
    return PREP_DIR / "analysis/feature_spec_lipo.json"


def load_spec():
    mods = _modules()
    return mods["feature_spec"].load_spec(spec_path())

def spec_hash() -> str:
    return C.sha256_file(spec_path())


def build_caches(payload: Mapping[str, Any], *, dataset: str) -> dict[str, Any]:
    """Build the ``lipo_v1`` split caches from the frozen raw attributes."""
    mods = _modules()
    spec = load_spec()
    lipo_raw = _lipo_raw(dataset)
    graphs = []
    t0 = time.perf_counter()
    for row in range(len(lipo_raw["smiles"])):
        raw_row = _lipo_row(lipo_raw, row)
        graphs.append(mods["adapters"].lipo_row_to_graph(raw_row, row, spec))
    build_seconds = time.perf_counter() - t0
    caches = {}
    for split in ("train", "valid", "test"):
        indices = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
        caches[split] = mods["lcrc_data"].build_cache(
            [graphs[int(i)] for i in indices], spec, split=split
        )
    norm = mods["lcrc_data"].fit_norm(caches["train"], spec, source="official-train")
    return {
        "spec": spec,
        "caches": caches,
        "norm": norm,
        "build_seconds": build_seconds,
        "n_graphs": len(graphs),
    }


def _lipo_raw(dataset: str):
    import sq_data as D

    return D.load_lipo_raw(dataset)


def _lipo_row(lipo_raw, row: int):
    import sq_data as D

    return D.lipo_raw_row(lipo_raw, row)


def _state_hash(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _attach_target(batch, chunk, y_std: np.ndarray, torch):
    batch["y"] = torch.as_tensor(np.asarray(y_std, dtype=np.float32)[chunk], dtype=torch.float32)
    return batch


def _evaluate(model, cache, collate, rows: np.ndarray, device, scaler, mods, y_std: np.ndarray) -> dict[str, Any]:
    import torch

    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    order: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start : start + 256]
            batch = collate(cache, chunk)
            _attach_target(batch, chunk, y_std, torch)
            for key, value in list(batch.items()):
                if hasattr(value, "to"):
                    batch[key] = value.to(device)
            out = model(batch)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            targets.append(np.asarray(batch["y"].detach().cpu().numpy(), dtype=np.float64).reshape(-1))
            order.append(np.asarray(chunk, dtype=np.int64))
    pred_std = np.concatenate(preds)
    target_std = np.concatenate(targets)
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    return {
        "pred_raw": pred_std * std + mean,
        "target_raw": target_std * std + mean,
        "rows": np.concatenate(order),
    }


def run_lcrc(
    *,
    payload: Mapping[str, Any],
    dataset: str,
    targets_raw: Mapping[str, np.ndarray],
    scaler: Mapping[str, float],
    device: str,
    epochs: int | None = None,
    max_train: int | None = None,
) -> dict[str, Any]:
    import torch

    mods = _modules()
    recipe = mods["dissect_common"].RECIPE
    epochs = int(recipe["epochs"] if epochs is None else epochs)
    bundle = build_caches(payload, dataset=dataset)
    spec = bundle["spec"]
    caches = bundle["caches"]
    norm = bundle["norm"]
    collate = mods["lcrc_data"].collate

    torch.manual_seed(C.TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.TRAIN_SEED)
    model = mods["dissect_model"].DissectLCRC(spec, norm, "C1").to(device)
    init_hash = _state_hash(model)
    n_parameters = int(sum(p.numel() for p in model.parameters()))

    y_std = {
        split: (np.asarray(targets_raw[split], dtype=np.float64) - float(scaler["mean"])) / float(scaler["std"])
        for split in ("train", "valid", "test")
    }
    train_rows = np.arange(int(caches["train"]["n_molecules"]), dtype=np.int64)
    eval_rows = {
        split: np.arange(int(caches[split]["n_molecules"]), dtype=np.int64)
        for split in ("train", "valid", "test")
    }
    if max_train is not None:
        train_rows = train_rows[: int(max_train)]
        eval_rows["train"] = eval_rows["train"][: int(max_train)]
        eval_rows["valid"] = eval_rows["valid"][: min(64, eval_rows["valid"].size)]
        eval_rows["test"] = eval_rows["test"][: min(64, eval_rows["test"].size)]

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = int(np.ceil(train_rows.size / batch_size))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["peak_lr"]),
        betas=tuple(recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    total_steps = epochs * steps_per_epoch
    peak = float(recipe["peak_lr"])
    min_lr = float(recipe["min_lr"])
    rng = np.random.default_rng(C.TRAIN_SEED)
    curves: list[dict[str, Any]] = []
    best = {"valid_rmse_raw": float("inf"), "epoch": None}
    best_predictions: dict[str, Any] = {}
    final_predictions: dict[str, Any] = {}
    best_state = None
    global_step = 0
    first_batch_rows: list[int] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        for start in range(0, order.size, batch_size):
            chunk = order[start : start + batch_size]
            if not first_batch_rows:
                first_batch_rows = [int(v) for v in chunk.tolist()]
            batch = collate(caches["train"], chunk)
            _attach_target(batch, chunk, y_std["train"], torch)
            for key, value in list(batch.items()):
                if hasattr(value, "to"):
                    batch[key] = value.to(device)
            lr = min_lr + 0.5 * (peak - min_lr) * (1.0 + np.cos(np.pi * global_step / max(1, total_steps)))
            optimizer.param_groups[0]["lr"] = float(lr)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            target = batch["y"].view(-1)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"]))
            optimizer.step()
        metrics: dict[str, Any] = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"])}
        predictions: dict[str, Any] = {}
        for split, rows in eval_rows.items():
            out = _evaluate(
                model, caches[split], collate, rows, device, scaler, mods, y_std[split]
            )
            predictions[split] = out
            metrics[f"{split}_rmse_raw"] = rmse(out["pred_raw"], out["target_raw"])
            metrics[f"{split}_mae_raw"] = mae(out["pred_raw"], out["target_raw"])
        metrics["seconds"] = time.perf_counter() - t0
        curves.append(metrics)
        if metrics["valid_rmse_raw"] < best["valid_rmse_raw"]:
            best = {"valid_rmse_raw": metrics["valid_rmse_raw"], "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_predictions = predictions
        final_predictions = predictions
    seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "method": "L",
        "variant": "C1/NO_BOND_MEAN",
        "spec_hash": spec_hash(),
        "norm_fingerprint": mods["lcrc_data"].norm_fingerprint(norm),
        "init_state_hash": init_hash,
        "n_parameters": n_parameters,
        "recipe": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in recipe.items()
            if isinstance(value, (int, float, str, tuple))
        },
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "first_batch_rows": first_batch_rows,
        "curves": curves,
        "selected": curves[best["epoch"] - 1],
        "final": curves[-1],
        "last20": {
            f"mean_last20_{split}_rmse_raw": float(
                np.mean([row[f"{split}_rmse_raw"] for row in curves[-20:]])
            )
            for split in ("train", "valid", "test")
        },
        "selected_predictions": best_predictions,
        "final_predictions": final_predictions,
        "seconds": seconds,
        "cache_build_seconds": bundle["build_seconds"],
        "n_graphs": bundle["n_graphs"],
        "status": "complete",
    }
