"""Formal training of ``C1_JOINT`` and ``C1_ADD`` on official Lipo.

The recipe, data view, target normaliser, canonical C1 initial state and every
epoch full-split evaluation are inherited from
``lcrc_readout_context_pilot_small_v1/code/cr_train.py``.  Only the S2 joint
channel is replaced by the Train-only support table.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch  # noqa: E402

import jt_common as C  # noqa: E402
import jt_models as M  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()
M.prepare_imports()

import cr_common as CC  # noqa: E402
import cr_data as D  # noqa: E402
import cr_models as CM  # noqa: E402
import cr_train as T  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "update",
    "lr",
    "train_loss_std",
    "train_loss_std_mean_of_batches",
    "train_rmse_raw",
    "valid_rmse_raw",
    "test_rmse_raw",
    "train_mae_raw",
    "valid_mae_raw",
    "test_mae_raw",
    "train_seconds",
    "eval_seconds",
    "elapsed_s",
]


class Watchdog:
    def __init__(self, hard_seconds: float):
        self.stop = threading.Event()
        self._timer = threading.Timer(float(hard_seconds), self.stop.set)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        self._timer.cancel()


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def _checkpoint(path: Path, model, *, arm: str, epoch: int, metrics: Mapping[str, Any], bundle, support, info) -> None:
    payload = {
        "schema": "lcrc-support-joint-transfer-checkpoint-v1",
        "round_id": C.ROUND_ID,
        "family": "lipo",
        "arm": arm,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "metrics": dict(metrics),
        "target_scaler": dict(bundle.scaler),
        "support": {
            "n_omega": int(support.n_omega),
            "rank_X": int(support.rank),
            "omega_code_sha256": C.sha256_array(np.asarray(support.omega_code, dtype=np.int64)),
            "P_sha256": C.sha256_array(np.asarray(support.P, dtype=np.float64)),
        },
        "model_info": info,
        "dataset": bundle.dataset,
        "split_sizes": bundle.split_sizes,
        "seed": C.SEED,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def run(arm: str, out_dir: Path, *, epochs: int, device: torch.device, hard_seconds: float) -> dict[str, Any]:
    if arm not in C.LIPO_ARMS:
        raise KeyError(arm)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    started_utc = C.utcnow()

    bundle = D.build_bundle("ogbg-mollipo")
    support = S.load_support(C.ANALYSIS / "support/lipo_support.npz")
    if bundle.dataset != "ogbg-mollipo":
        raise RuntimeError("bundle dataset drifted")
    if support.n_omega != 108:
        raise RuntimeError(f"unexpected Lipo Omega size {support.n_omega}")

    model, info = M.build_lipo_model(bundle, support, arm)
    model = model.to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    j_key = next((k for k in init_state if k.endswith("J_table")), None)
    t_key = next((k for k in init_state if k.endswith("Theta")), None)
    if j_key is not None:
        table0 = init_state[j_key].to(torch.float64).numpy()
    elif t_key is not None:
        table0 = (torch.as_tensor(support.U, dtype=torch.float64) @ init_state[t_key].to(torch.float64)).numpy()
    else:
        raise RuntimeError("no support table in the initial state")
    CC.atomic_write_json(
        out_dir / "init.json",
        {
            "arm": arm,
            "model_info": info,
            "table_rms": float(np.sqrt(np.mean(np.square(table0)))),
            "state_sha256": C.sha256_state(init_state),
            "support": S.omega_identity(support),
            "environment": C.environment(),
        },
    )

    torch_device = device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(T.RECIPE["peak_lr"]),
        betas=tuple(T.RECIPE["adam_betas"]),
        eps=float(T.RECIPE["adam_eps"]),
        weight_decay=float(T.RECIPE["weight_decay"]),
    )
    train_rows = np.arange(int(bundle.caches["train"]["n_molecules"]), dtype=np.int64)
    batch_size = int(T.RECIPE["batch_size"])
    steps_per_epoch = int(np.ceil(train_rows.size / batch_size))
    total_steps = int(epochs) * steps_per_epoch

    CC.set_global_seeds(C.SEED)
    rng = np.random.default_rng(C.SEED)
    rng_hash_at_start = CC.rng_state_hash()

    curves: list[dict[str, Any]] = []
    preds: dict[str, list[np.ndarray]] = {split: [] for split in C.SPLITS}
    batch_order_hashes: list[str] = []
    best = {"valid_rmse_raw": math.inf, "epoch": None, "metrics": None}
    best_state = None
    final_state = None
    global_step = 0
    train_seconds_total = 0.0
    eval_seconds_total = 0.0
    watchdog = Watchdog(hard_seconds)
    stop_reason = "COMPLETED"

    # epoch-0 diagnostic (never selectable)
    model.eval()
    init_metrics = {}
    with torch.no_grad():
        for split in C.SPLITS:
            out = T.evaluate(model, bundle, split, torch_device)
            init_metrics[split] = {"rmse_raw": float(out["rmse_raw"]), "mae_raw": float(out["mae_raw"])}
    CC.atomic_write_json(out_dir / "init_metrics.json", init_metrics)

    for epoch in range(1, int(epochs) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        t_epoch = time.perf_counter()
        model.train()
        order = rng.permutation(train_rows)
        batch_order_hashes.append(CC.sha256_array(order))
        t_train = time.perf_counter()
        loss_sum = 0.0
        loss_weight = 0
        batch_losses: list[float] = []
        for start in range(0, order.size, batch_size):
            chunk = order[start : start + batch_size]
            batch = bundle.collate(bundle.caches["train"], chunk)
            batch["y"] = torch.as_tensor(bundle.y_std["train"][chunk], dtype=torch.float32)
            batch = _to_device(batch, torch_device)
            lr = float(T.RECIPE["min_lr"]) + 0.5 * (float(T.RECIPE["peak_lr"]) - float(T.RECIPE["min_lr"])) * (
                1.0 + np.cos(np.pi * global_step / max(1, total_steps))
            )
            optimizer.param_groups[0]["lr"] = float(lr)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch["y"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(T.RECIPE["gradient_clip"]))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            loss_sum += float(loss.detach().cpu()) * int(chunk.size)
            loss_weight += int(chunk.size)
        train_seconds = time.perf_counter() - t_train
        train_seconds_total += train_seconds

        t_eval = time.perf_counter()
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "update": int(global_step),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss_std": float(loss_sum / max(1, loss_weight)),
            "train_loss_std_mean_of_batches": float(np.mean(batch_losses)),
        }
        for split in C.SPLITS:
            out = T.evaluate(model, bundle, split, torch_device)
            preds[split].append(out["pred_raw"].astype(np.float32))
            row[f"{split}_rmse_raw"] = float(out["rmse_raw"])
            row[f"{split}_mae_raw"] = float(out["mae_raw"])
        eval_seconds = time.perf_counter() - t_eval
        eval_seconds_total += eval_seconds
        row["train_seconds"] = float(train_seconds)
        row["eval_seconds"] = float(eval_seconds)
        row["elapsed_s"] = float(time.perf_counter() - t_epoch)
        curves.append(row)

        if float(row["valid_rmse_raw"]) < float(best["valid_rmse_raw"]):
            best = {"valid_rmse_raw": float(row["valid_rmse_raw"]), "epoch": int(epoch), "metrics": dict(row)}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        C.write_csv(out_dir / "curve.csv", curves, columns=CURVE_FIELDS)
        print(
            f"[{arm}] ep {epoch}/{epochs} loss {row['train_loss_std']:.6f} "
            f"validRMSE {row['valid_rmse_raw']:.6f} testRMSE {row['test_rmse_raw']:.6f} "
            f"train {train_seconds:.1f}s eval {eval_seconds:.1f}s {'*best*' if best['epoch'] == epoch else ''}",
            flush=True,
        )

    watchdog.cancel()
    if best_state is None:
        raise RuntimeError("no selectable epoch completed")

    best_epoch = int(best["epoch"])
    best_metrics = best["metrics"]
    _checkpoint(out_dir / "checkpoint_best_val.pt", _StateModel(best_state), arm=arm, epoch=best_epoch, metrics=best_metrics, bundle=bundle, support=support, info=info)
    _checkpoint(out_dir / "checkpoint_final.pt", _StateModel(final_state), arm=arm, epoch=len(curves), metrics=curves[-1], bundle=bundle, support=support, info=info)

    # per-epoch prediction matrices (rows in ascending official row id)
    for split in C.SPLITS:
        order, row_ids = bundle.row_ids_sorted(split)
        matrix = np.stack(preds[split], axis=0).astype(np.float32)
        CC.atomic_write_npz(
            out_dir / f"pred_matrix_{split}.npz",
            {
                "pred_raw": matrix,
                "row_id": row_ids,
                "y_raw": bundle.targets_raw[split][order].astype(np.float32),
                "epoch": np.arange(1, matrix.shape[0] + 1),
            },
        )

    # best / final predictions
    model.load_state_dict({k: v.to(torch_device) for k, v in best_state.items()}, strict=True)
    best_pred = {split: T.evaluate(model, bundle, split, torch_device) for split in C.SPLITS}
    t_infer = time.perf_counter()
    _ = {split: T.evaluate(model, bundle, split, torch_device) for split in C.SPLITS}
    inference_seconds = time.perf_counter() - t_infer
    model.load_state_dict({k: v.to(torch_device) for k, v in final_state.items()}, strict=True)
    final_pred = {split: T.evaluate(model, bundle, split, torch_device) for split in C.SPLITS}
    for tag, payload in (("best", best_pred), ("final", final_pred)):
        arrays = {}
        for split in C.SPLITS:
            out = payload[split]
            arrays[f"{split}_pred_raw"] = out["pred_raw"].astype(np.float64)
            arrays[f"{split}_pred_std"] = out["pred_std"].astype(np.float64)
            arrays[f"{split}_y_raw"] = out["y_raw"].astype(np.float64)
            arrays[f"{split}_row_id"] = out["row_ids"]
        CC.atomic_write_npz(out_dir / f"{tag}_predictions.npz", arrays)

    last20 = {
        f"mean_last20_{split}_{metric}_raw": float(np.mean([r[f"{split}_{metric}_raw"] for r in curves[-20:]]))
        for split in C.SPLITS
        for metric in ("rmse", "mae")
    }
    summary = {
        "round_id": C.ROUND_ID,
        "family": "lipo",
        "dataset": bundle.dataset,
        "arm": arm,
        "status": "complete" if stop_reason == "COMPLETED" else stop_reason,
        "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in T.RECIPE.items()},
        "epochs": int(epochs),
        "epochs_completed": len(curves),
        "steps_per_epoch": steps_per_epoch,
        "total_updates": global_step,
        "n_parameters": int(info["parameters"]),
        "joint_module_parameters": int(info["joint_module_parameters"]),
        "train_supported_joint_parameters": int(info["train_supported_joint_parameters"]),
        "support": S.omega_identity(support),
        "support_summary": support.summary(),
        "init_metrics": init_metrics,
        "curves": curves,
        "selected": {
            "epoch": best_epoch,
            **{k: best_metrics[k] for k in best_metrics if k.endswith("_rmse_raw") or k.endswith("_mae_raw")},
            "lr": best_metrics["lr"],
            "update": best_metrics["update"],
        },
        "final": {
            "epoch": len(curves),
            **{k: curves[-1][k] for k in curves[-1] if k.endswith("_rmse_raw") or k.endswith("_mae_raw")},
        },
        "last20": last20,
        "batch_order_sha256": CC.sha256_text("".join(batch_order_hashes)),
        "batch_order_hashes_first3": batch_order_hashes[:3],
        "rng_hash_at_start": rng_hash_at_start,
        "train_seconds": train_seconds_total,
        "eval_seconds_total": eval_seconds_total,
        "inference_seconds_full_three_splits": inference_seconds,
        "checkpoint_hashes": {
            "checkpoint_best_val.pt": C.sha256_file(out_dir / "checkpoint_best_val.pt"),
            "checkpoint_final.pt": C.sha256_file(out_dir / "checkpoint_final.pt"),
        },
        "environment": C.environment(),
        "device": str(torch_device),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "wall_seconds": float(time.perf_counter() - t_start),
        "stop_reason": stop_reason,
    }
    CC.atomic_write_json(out_dir / "trajectory.json", summary)
    CC.atomic_write_json(
        out_dir / "status.json",
        {
            "state": "completed" if stop_reason == "COMPLETED" else stop_reason,
            "arm": arm,
            "dataset": bundle.dataset,
            "epochs": len(curves),
            "selected_epoch": best_epoch,
            "valid_rmse_raw": float(best["valid_rmse_raw"]),
            "test_rmse_raw": float(best_metrics["test_rmse_raw"]),
            "finished_utc": summary["finished_utc"],
        },
    )
    return summary


class _StateModel(torch.nn.Module):
    """Tiny holder so ``_checkpoint`` can read a state dict uniformly."""

    def __init__(self, state: Mapping[str, torch.Tensor]):
        super().__init__()
        self._state = state

    def state_dict(self, *args, **kwargs):  # noqa: D401 - torch API
        return self._state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=C.LIPO_ARMS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=int(C.LIPO_RECIPE["epochs"]))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hard-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        summary = run(args.arm, Path(args.out_dir), epochs=args.epochs, device=device, hard_seconds=args.hard_seconds)
        print(f"COMPLETED {args.arm} selected={summary['selected']['epoch']} valid={summary['selected']['valid_rmse_raw']:.9f} test={summary['selected']['test_rmse_raw']:.9f}")
    except Exception:
        traceback.print_exc()
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        C.write_json(out / "run.failed.json", {"status": "FAILED", "traceback": traceback.format_exc(), "utc": C.utcnow()})
        raise


if __name__ == "__main__":
    main()
