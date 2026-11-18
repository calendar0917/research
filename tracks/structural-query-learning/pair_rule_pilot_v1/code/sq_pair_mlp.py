"""Two-hidden-layer MLP readout on exactly the REAL FULL input.

Structure (frozen): ``D -> 128 -> 64 -> 1``, SiLU after each hidden layer,
dropout 0.10, output is the standardised regression scalar.  Training reuses the
repository's ``Rf`` schedule (``gr_common.lr_at("Rf", step, 3360, 180)``) and the
same per-epoch ``np.random.default_rng(0).permutation`` batch order as C1.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_pair_common as C
import sq_pair_fista as F


def _torch():
    import torch

    return torch


def build_mlp(input_dim: int, dropout: float = C.DROPOUT):
    torch = _torch()
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(int(input_dim), 128),
        nn.SiLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(128, 64),
        nn.SiLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(64, 1),
    )


def set_training_seed(seed: int = C.TRAIN_SEED) -> dict[str, Any]:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    return {"seed": int(seed), "torch_initial_seed": int(torch.initial_seed())}


def model_state_hash(model) -> str:
    torch = _torch()
    digest = __import__("hashlib").sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _mlp_signature(system: F.AdditiveSystem, epochs: int) -> str:
    return C.sha256_json(
        {
            "round_id": C.ROUND_ID,
            "arm": "MLP",
            "epochs": int(epochs),
            "input": "REAL_FULL",
            "matrix_hashes": system.matrix_hashes,
            "scales": C.sha256_array(system.scales),
            "scaler": system.scaler,
            "batch_size": C.BATCH_SIZE,
            "eval_batch_size": C.EVAL_BATCH_SIZE,
            "dropout": C.DROPOUT,
            "recipe": C.C1_RECIPE,
            "weight_decay": C.WEIGHT_DECAY,
            "code": C.code_hashes(),
            "protocol": C.protocol_hash(),
        }
    )


def _save_predictions(path: Path, data: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    F.save_predictions(path, data)


def _predict(model, x: np.ndarray, device, batch_size: int = C.EVAL_BATCH_SIZE) -> np.ndarray:
    torch = _torch()
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            block = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(model(block).detach().cpu().numpy().astype(np.float64).reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def _scores(pred_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, float]:
    return F.metrics(pred_raw, y_raw)


def fit_mlp(
    *,
    prep_dir: Path,
    out_dir: Path,
    device: str,
    epochs: int = C.EPOCHS,
    resume: bool = True,
    ledger=None,
) -> dict[str, Any]:
    torch = _torch()
    out = C.ensure_dir(out_dir)
    system = F.load_system(prep_dir, "REAL")
    n_train = system.n_train
    signature = _mlp_signature(system, epochs)
    status_path = out / "status.json"
    if resume and status_path.is_file():
        try:
            existing = C.read_json(status_path)
        except Exception:  # noqa: BLE001
            existing = {}
        if existing.get("state") == "completed" and existing.get("signature") == signature:
            existing["resumed"] = True
            return existing

    init = set_training_seed(C.TRAIN_SEED)
    model = build_mlp(system.n_columns).to(device)
    init["state_hash_before_training"] = model_state_hash(model)
    init["n_parameters"] = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=tuple(C.ADAM_BETAS),
        eps=float(C.ADAM_EPS),
        weight_decay=float(C.WEIGHT_DECAY),
    )
    spe = int(np.ceil(n_train / C.BATCH_SIZE))
    total_steps = int(epochs) * spe
    rng = np.random.default_rng(C.TRAIN_SEED)
    state: dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "curves": [],
        "best": {"valid_rmse_raw": float("inf"), "epoch": None},
        "train_seconds": 0.0,
        "eval_seconds": 0.0,
    }
    best_state = None
    resume_path = out / "resume_state.pt"
    if resume and resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("signature") == signature:
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            state = saved["state"]
            best_state = saved.get("best_state")
            rng.bit_generator.state = saved["np_rng_state"]
            torch.set_rng_state(saved["torch_rng_state"])
            if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        else:
            print("[resume] MLP: signature mismatch; restarting from scratch", flush=True)
            init = set_training_seed(C.TRAIN_SEED)
            model = build_mlp(system.n_columns).to(device)
            init["state_hash_before_training"] = model_state_hash(model)
            init["n_parameters"] = parameter_count(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                betas=tuple(C.ADAM_BETAS),
                eps=float(C.ADAM_EPS),
                weight_decay=float(C.WEIGHT_DECAY),
            )
            best_state = None
    if state["epoch"] == 0 and best_state is None:
        init = {**init, "state_hash_before_training": model_state_hash(model), "n_parameters": parameter_count(model)}

    x = {"train": system.x_train, "valid": system.x_valid, "test": system.x_test}
    row_ids = {
        split: np.asarray(C.read_npz(Path(prep_dir) / "matrices" / f"{split}.npz")["row_id"], dtype=np.int64)
        for split in C.SPLITS
    }
    y_raw = system.y_raw
    mean, std = float(system.scaler["mean"]), float(system.scaler["std"])
    y_std = {split: (y_raw[split] - mean) / std for split in C.SPLITS}
    train_rows = np.arange(n_train, dtype=np.int64)
    t0 = time.perf_counter()

    def evaluate_all() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
        preds: dict[str, dict[str, np.ndarray]] = {}
        scores: dict[str, dict[str, float]] = {}
        for split in C.SPLITS:
            pred_std = _predict(model, x[split], device)
            pred_raw = pred_std * std + mean
            preds[split] = F.prediction_payload(row_ids[split], y_raw[split], pred_raw)
            scores[split] = _scores(pred_raw, y_raw[split])
        return preds, scores

    for epoch in range(int(state["epoch"]) + 1, int(epochs) + 1):
        t_epoch = time.perf_counter()
        model.train()
        order = rng.permutation(train_rows)
        t_train = time.perf_counter()
        for start in range(0, order.size, C.BATCH_SIZE):
            chunk = order[start : start + C.BATCH_SIZE]
            lr = float(
                __import__("gr_common").lr_at(C.C1_RECIPE, int(state["global_step"]), n_train, int(epochs))
            )
            optimizer.param_groups[0]["lr"] = lr
            state["global_step"] += 1
            xb = torch.as_tensor(system.x_train[chunk], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_std["train"][chunk], dtype=torch.float32, device=device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb).view(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.GRAD_CLIP))
            optimizer.step()
        state["train_seconds"] += time.perf_counter() - t_train
        t_eval = time.perf_counter()
        preds, scores = evaluate_all()
        state["eval_seconds"] += time.perf_counter() - t_eval
        row = {
            "epoch": int(epoch),
            "global_step": int(state["global_step"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{
                f"{split}_{name}": score
                for split in C.SPLITS
                for name, score in scores[split].items()
                if name != "n"
            },
            "seconds": time.perf_counter() - t_epoch,
            "train_seconds": state["train_seconds"],
        }
        state["curves"].append(row)
        if scores["valid"]["rmse_raw"] < float(state["best"]["valid_rmse_raw"]):
            state["best"] = {
                "valid_rmse_raw": float(scores["valid"]["rmse_raw"]),
                "epoch": int(epoch),
                "scores": scores,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_predictions(out / "best_predictions.npz", preds)
            torch.save(
                {"model_state": best_state, "epoch": int(epoch), "signature": signature},
                out / "checkpoint_best_val.pt",
            )
        state["epoch"] = int(epoch)
        torch.save(
            {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "state": state,
                "best_state": best_state,
                "signature": signature,
                "np_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            resume_path,
        )
        if epoch % 5 == 0 or epoch == int(epochs):
            _write_curve_csv(out / "curve.csv", state["curves"])
            torch.save(
                {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": int(epoch), "signature": signature},
                out / "checkpoint_final.pt",
            )
        if ledger is not None and (epoch % 20 == 0 or epoch == int(epochs)):
            ledger.record("MLP_progress", epoch, epoch=int(epoch), valid_rmse=row["valid_rmse_raw"])

    final_preds, final_scores = evaluate_all()
    _save_predictions(out / "epoch180_predictions.npz", final_preds)
    _write_curve_csv(out / "curve.csv", state["curves"])
    last20 = {
        f"{key}": float(np.mean([row[key] for row in state["curves"][-20:]]))
        for key in ("train_rmse_raw", "valid_rmse_raw", "test_rmse_raw", "train_mae_raw", "valid_mae_raw", "test_mae_raw")
    }
    summary = {
        "state": "completed",
        "signature": signature,
        "arm": "MLP",
        "input": "REAL_FULL",
        "epochs": int(epochs),
        "n_train": int(n_train),
        "n_columns": int(system.n_columns),
        "input_dim": int(system.n_columns),
        "n_parameters": parameter_count(model),
        "init": init,
        "architecture": "D -> 128 -> 64 -> 1; SiLU; dropout 0.10",
        "optimizer": {
            "name": "AdamW",
            "betas": list(C.ADAM_BETAS),
            "eps": float(C.ADAM_EPS),
            "weight_decay": float(C.WEIGHT_DECAY),
            "recipe": C.C1_RECIPE,
            "peak_lr": 1e-3,
            "warmup_epochs": 5,
            "min_lr": 1e-5,
            "batch_size": C.BATCH_SIZE,
            "steps_per_epoch": spe,
            "total_steps": total_steps,
            "grad_clip": C.GRAD_CLIP,
        },
        "best": state["best"],
        "epoch180": {"scores": final_scores, "epoch": int(epochs)},
        "last20": last20,
        "train_seconds": float(state["train_seconds"]),
        "eval_seconds": float(state["eval_seconds"]),
        "fit_seconds": time.perf_counter() - t0,
        "device": str(device),
        "matrix_hashes": system.matrix_hashes,
        "columns": system.column_names,
        "environment": C.environment(),
        "run_provenance": C.run_provenance(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_json(status_path, summary)
    return summary


def _write_curve_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "epoch",
        "global_step",
        "lr",
        "train_rmse_raw",
        "valid_rmse_raw",
        "test_rmse_raw",
        "train_mae_raw",
        "valid_mae_raw",
        "test_mae_raw",
        "seconds",
        "train_seconds",
    ]
    C.write_csv(path, curves, columns=columns)


# ---------------------------------------------------------------------------
# smoke (never part of a formal trajectory)
# ---------------------------------------------------------------------------


def smoke_mlp(
    *,
    prep_dir: Path,
    device: str,
    n_rows: int = 16,
    seed: int = C.TRAIN_SEED,
) -> dict[str, Any]:
    torch = _torch()
    system = F.load_system(prep_dir, "REAL")
    init = set_training_seed(seed)
    model = build_mlp(system.n_columns).to(device)
    model.train()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(system.n_train, size=min(int(n_rows), system.n_train), replace=False))
    x = torch.as_tensor(system.x_train[idx], dtype=torch.float32, device=device)
    y = torch.as_tensor(
        (system.y_raw["train"][idx] - float(system.scaler["mean"])) / float(system.scaler["std"]),
        dtype=torch.float32,
        device=device,
    ).view(-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    out = model(x).view(-1)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.GRAD_CLIP)))
    optimizer.step()
    finite_output = bool(torch.isfinite(out).all())
    finite_loss = bool(torch.isfinite(loss))
    finite_grad = bool(np.isfinite(grad_norm))
    return {
        "stage": "mlp_smoke",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_rows": int(idx.size),
        "input_dim": int(system.n_columns),
        "n_parameters": parameter_count(model),
        "init": init,
        "loss": float(loss.detach().cpu()),
        "grad_norm_before_clip": grad_norm,
        "finite_output": finite_output,
        "finite_loss": finite_loss,
        "finite_grad": finite_grad,
        "pass": bool(finite_output and finite_loss and finite_grad),
    }
