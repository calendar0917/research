"""Formal neural training entry point for one arm / split / seed.

Recipe is the frozen R2 recipe (AdamW 3e-4 -> 1e-5 cosine, wd 1e-4, clip 5,
batch 64, 5 warmup epochs, 240 max epochs, min 80, patience 45, best-Dev
checkpoint).  Official Valid/Test labels are never read.
"""

from __future__ import annotations

import argparse
import math
import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import tc_common as C
import tc_data as D
import tc_neural as N

CODE_FILES = ("tc_common.py", "tc_vocab.py", "tc_data.py", "tc_neural.py", "tc_train.py")


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT))
    except ValueError:
        return str(path)


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float,
          minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak) * (step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


def _predict(model: N.TCWordModel, data: D.SplitData, indices: Sequence[int],
             arm: str, device: Any, batch_size: int, scaler: Mapping[str, float]
             ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    idx = np.asarray(indices, np.int64)
    pred_std = np.zeros(idx.size, np.float64)
    with torch.no_grad():
        for start in range(0, idx.size, int(batch_size)):
            part = idx[start:start + int(batch_size)]
            batch = D.collate(data, part.tolist(), arm, device)
            out = model(batch, arm=arm)
            pred_std[start:start + part.size] = out.double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def _write_predictions(path: Path, rows: np.ndarray, y: np.ndarray,
                       pred: np.ndarray) -> None:
    np.savez_compressed(path.with_suffix(".npz"), rows=rows, y_true=y, y_pred=pred,
                        residual=y - pred)
    C.write_csv(path.with_suffix(".csv"),
                [{"row_id": int(r), "y_true": float(t), "y_pred": float(p),
                  "residual": float(t - p)}
                 for r, t, p in zip(rows.tolist(), y.tolist(), pred.tolist())])


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    C.assert_source_pins()
    cfg = C.load_config()
    recipe = dict(cfg["neural"]["recipe"])
    dims = dict(cfg["neural"]["dims"])
    if args.arm not in N.ARMS:
        raise ValueError(f"unknown arm {args.arm}")
    C.seed_everything(int(args.seed))
    commit = C.git_commit()
    t_start_utc = C.now_utc()
    out_dir = Path(args.out) if args.out else (C.RESULTS_DIR / str(args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    atom_dims, bond_dims = C.atom_bond_dims()
    t_prep = time.time()
    data = D.build_split_data(args.split, atom_dims, bond_dims)
    prep_seconds = time.time() - t_prep
    K = int(data.n_task)
    if int(data.n_freq) != K:
        raise ValueError("frequency vocabulary size does not match the task vocabulary")
    fit_idx = data.fit_index[:int(args.max_fit)] if args.max_fit else data.fit_index
    eval_idx = data.eval_index[:int(args.max_eval)] if args.max_eval else data.eval_index
    log(f"run={args.run_id} arm={args.arm} split={args.split} seed={args.seed} "
        f"device={args.device} host={socket.gethostname()}")
    log(f"prepared data in {prep_seconds:.1f}s rows={data.rows.size} "
        f"fit={fit_idx.size} dev={eval_idx.size} K={K} union={data.hashes['n_union_words']} "
        f"atoms={data.hashes['n_atoms_total']} bonds={data.hashes['n_bonds_total']}")
    log(f"vocab task={data.hashes['task_vocab_sha256'][:16]} "
        f"freq={data.hashes['freq_vocab_sha256'][:16]}")

    model = N.TCWordModel(atom_dims, bond_dims, K, dims)
    init_hashes = model.init_parameters(int(args.seed))
    zeroed = model.zero_unseen(data.atom_ok, data.bond_ok)
    ledger = model.parameter_ledger()
    model = model.to(args.device)
    log(f"params={ledger}")
    log(f"zeroed_unseen={zeroed}")
    log(f"init={init_hashes}")

    epochs = int(args.epochs or recipe["epochs"])
    batch_mol = int(recipe["batch_molecules"])
    n_fit = int(fit_idx.size)
    steps_per_epoch = int(math.ceil(n_fit / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    rng = np.random.default_rng(int(args.seed) + 101)
    patience = int(args.patience if args.patience is not None else recipe["patience"])
    min_epochs = int(args.min_epochs if args.min_epochs is not None else recipe["min_epochs"])
    eval_batch = int(recipe["eval_batch"])

    curves: list[dict[str, Any]] = []
    best = {"dev_rmse": float("inf"), "dev_mae": float("inf"),
            "fit_rmse": float("inf"), "epoch": -1}
    step = 0
    peak_vram = 0
    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_fit)
        losses: list[float] = []
        grad_norms: list[float] = []
        for start in range(0, n_fit, batch_mol):
            idx = fit_idx[order[start:start + batch_mol]]
            batch = D.collate(data, idx.tolist(), args.arm, args.device)
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model(batch, arm=args.arm)
            target = torch.as_tensor(data.target_std[idx], device=args.device,
                                     dtype=torch.float32)
            loss = torch.mean((out - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(recipe["gradient_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
            step += 1
            if torch.cuda.is_available() and args.device != "cpu":
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            if step == 1:
                log(f"step1 loss={losses[-1]:.4f} grad={grad_norms[-1]:.3f} rows={idx.size}")
        t_eval = time.time()
        pred_fit, _ = _predict(model, data, fit_idx, args.arm, args.device,
                               eval_batch, data.scaler)
        pred_dev, _ = _predict(model, data, eval_idx, args.arm, args.device,
                               eval_batch, data.scaler)
        eval_seconds = time.time() - t_eval
        fit_rmse = float(C.rmse(data.target_raw[fit_idx], pred_fit))
        dev_rmse = float(C.rmse(data.target_raw[eval_idx], pred_dev))
        dev_mae = float(C.mae(data.target_raw[eval_idx], pred_dev))
        row = {
            "epoch": int(epoch), "step": int(step),
            "lr": lr_at(step - 1, warmup_steps, total_steps,
                        float(recipe["peak_lr"]), float(recipe["min_lr"])),
            "train_loss": float(np.mean(losses)),
            "grad_norm_mean": float(np.mean(grad_norms)),
            "fit_rmse": fit_rmse, "dev_rmse": dev_rmse, "dev_mae": dev_mae,
            "epoch_seconds": float(time.time() - t_epoch),
            "eval_seconds": float(eval_seconds),
        }
        curves.append(row)
        log(f"epoch={epoch} loss={row['train_loss']:.5f} fit={fit_rmse:.6f} "
            f"dev={dev_rmse:.6f} mae={dev_mae:.6f} lr={row['lr']:.2e} "
            f"epoch_s={row['epoch_seconds']:.1f}")
        improved = dev_rmse < best["dev_rmse"] - 1e-12
        if improved:
            best = {"dev_rmse": dev_rmse, "dev_mae": dev_mae,
                    "fit_rmse": fit_rmse, "epoch": int(epoch)}
            torch.save({"model": model.state_dict(), "epoch": int(epoch),
                        "best": dict(best), "arm": args.arm, "seed": int(args.seed),
                        "split": args.split, "K": K}, out_dir / "state_best.pt")
            _write_predictions(out_dir / "predictions_best_fit", data.rows[fit_idx],
                               data.target_raw[fit_idx], pred_fit)
            _write_predictions(out_dir / "predictions_best_dev", data.rows[eval_idx],
                               data.target_raw[eval_idx], pred_dev)
        if epoch >= min_epochs and (epoch - best["epoch"]) >= patience:
            log(f"early stop at epoch {epoch} (best {best['epoch']})")
            break
    wall_seconds = time.time() - t_train_start
    model.eval()
    pred_fit, _ = _predict(model, data, fit_idx, args.arm, args.device, eval_batch,
                           data.scaler)
    pred_dev, _ = _predict(model, data, eval_idx, args.arm, args.device, eval_batch,
                           data.scaler)
    torch.save({"model": model.state_dict(), "epoch": len(curves),
                "best": dict(best), "arm": args.arm, "seed": int(args.seed),
                "split": args.split, "K": K}, out_dir / "state_final.pt")
    C.write_csv(out_dir / "curves.csv", curves)
    gpu_hours = wall_seconds / 3600.0 if (args.device != "cpu" and torch.cuda.is_available()) else 0.0
    manifest = {
        "protocol_id": cfg["protocol_id"], "version": cfg["version"],
        "run_id": args.run_id, "arm": args.arm, "split": args.split,
        "seed": int(args.seed), "smoke": bool(args.smoke),
        "K": int(K), "n_task_words": int(data.n_task), "n_freq_words": int(data.n_freq),
        "n_union_words": int(data.hashes["n_union_words"]),
        "data_hashes": data.hashes,
        "params": ledger,
        "init_hashes": init_hashes,
        "zeroed_unseen": zeroed,
        "recipe": recipe, "dims": dims,
        "sentence_input_dim": int(model.sentence_in_dim),
        "pair_input_dim": int(model.pair_in_dim),
        "readout_input_dim": int(model.readout_in_dim),
        "best": dict(best),
        "epochs_run": int(len(curves)), "epochs_max": int(epochs),
        "wall_seconds": float(wall_seconds),
        "process_seconds": float(time.time() - t_process),
        "prep_seconds": float(prep_seconds),
        "gpu_hours": float(gpu_hours),
        "peak_vram_bytes": int(peak_vram),
        "device": str(args.device), "gpu_name": _gpu_name(),
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda or "none",
        "git_commit": commit, "code_sha256": C.code_sha(CODE_FILES),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "started_utc": t_start_utc, "finished_utc": C.now_utc(),
        "fit_rows_sha256": data.hashes["fit_rows_sha256"],
        "dev_rows_sha256": data.hashes["eval_rows_sha256"],
        "predictions_best_fit": _rel(out_dir / "predictions_best_fit.npz"),
        "predictions_best_dev": _rel(out_dir / "predictions_best_dev.npz"),
        "curves": _rel(out_dir / "curves.csv"),
        "checkpoint": _rel(out_dir / "state_best.pt"),
    }
    C.write_json(out_dir / "manifest.json", manifest)
    log(f"done best={best} wall={wall_seconds:.1f}s gpu_h={gpu_hours:.4f}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True, choices=N.ARMS)
    ap.add_argument("--split", required=True, choices=("D0", "D1"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=None)
    ap.add_argument("--max-fit", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()