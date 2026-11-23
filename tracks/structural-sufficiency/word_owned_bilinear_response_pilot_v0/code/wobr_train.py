"""Formal single-seed training entry point for WOBR-v0.

One model, seed 0, official Train/Valid/Test split.  The loss uses only
official Train labels.  Valid selects the best checkpoint.  Test is evaluated
and logged every epoch as ``DEVELOPMENT_TEST`` and never enters gradients,
standardization, checkpoint selection or early stopping.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import wobr_common as C
import wobr_data as D
import wobr_model as M
import wobr_structure as S
import wobr_tensor as T


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak) * (step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT))
    except ValueError:
        return str(path)


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def _norm(bundle: D.DataBundle) -> dict[str, np.ndarray]:
    return D.normalisers(bundle)


def _predict_raw(model: M.WobrModel, rows: Sequence[int],
                 structs: Mapping[int, S.MoleculeStruct], raw: Mapping[str, Any],
                 device: Any, batch_size: int,
                 scaler: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    pred_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = T.collate(part.tolist(), structs, raw, device)
            out = model(batch)
            pred_std[start:start + part.size] = out["yhat_std"].double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def _metrics(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {"rmse": C.rmse(pred, labels), "mae": C.mae(pred, labels)}


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    C.seed_everything(int(args.seed))
    cfg = C.load_config()
    recipe = dict(cfg["recipe"])
    out_dir = Path(args.out) if args.out else (C.RESULTS_DIR / str(args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    if args.device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    bundle = D.load_bundle()
    train_rows = np.asarray(bundle.train_idx, np.int64)
    valid_rows = np.asarray(bundle.valid_idx, np.int64)
    test_rows = np.asarray(bundle.test_idx, np.int64)
    if args.max_fit:
        train_rows = train_rows[:int(args.max_fit)]
    if args.max_eval:
        valid_rows = valid_rows[:int(args.max_eval)]
        test_rows = test_rows[:int(args.max_eval)]
    all_rows = np.union1d(np.union1d(train_rows, valid_rows), test_rows)
    y_std = bundle.labels_std()

    log(f"run={args.run_id} seed={args.seed} device={args.device} host={socket.gethostname()}")
    log(f"data raw={bundle.hashes['raw_cache']['sha256'][:12]} "
        f"train={train_rows.size} valid={valid_rows.size} test={test_rows.size} "
        f"scaler={bundle.scaler}")

    t_prep = time.time()
    structs = S.build_all(bundle.raw, all_rows)
    stats = S.structure_stats(structs, all_rows)
    prep_seconds = time.time() - t_prep
    log(f"prepared structures in {prep_seconds:.1f}s: words={stats['n_words']} "
        f"roots_without_words={stats['roots_without_words']} "
        f"max_words_per_root={stats['words_per_root_max']}")

    atom_ok, bond_ok = D.category_support(bundle.raw, np.asarray(bundle.train_idx),
                                          bundle.atom_dims, bundle.bond_dims)
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, int(args.seed),
                                _norm(bundle), config=cfg)
    zeroed = model.zero_unseen(atom_ok, bond_ok, bundle.atom_dims, bundle.bond_dims)
    fingerprint_before = info["init"]["state_fingerprint"]
    fingerprint_after = C.state_fingerprint(model.state_dict())
    self_fingerprint = C.module_hashes({"model": model})["model"]
    ledger = info["parameters"]
    model = model.to(args.device)
    log(f"params={ledger['total']}")
    log(f"zeroed_unseen={zeroed} unseen={bundle.unseen}")
    log(f"init_fingerprint={fingerprint_before} after_support={fingerprint_after}")

    epochs = int(args.epochs or recipe["epochs"])
    batch_mol = int(recipe["batch_molecules"])
    n_train = int(train_rows.size)
    steps_per_epoch = int(math.ceil(n_train / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    rng = np.random.default_rng(int(args.seed) + 101)
    patience = int(args.patience if args.patience is not None else recipe["patience"])
    min_epochs = int(args.min_epochs if args.min_epochs is not None else recipe["min_epochs"])
    eval_batch = int(recipe["eval_batch"])

    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse": float("inf"), "valid_mae": float("inf"),
            "train_rmse": float("inf"), "epoch": -1}
    step = 0
    peak_vram = 0
    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_train)
        losses: list[float] = []
        grad_norms: list[float] = []
        for start in range(0, n_train, batch_mol):
            rows = train_rows[order[start:start + batch_mol]]
            batch = T.collate(rows.tolist(), structs, bundle.raw, args.device)
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model(batch)
            target = torch.as_tensor(y_std[rows], device=args.device, dtype=torch.float32)
            loss = torch.mean((out["yhat_std"] - target) ** 2)
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
                log(f"step1 loss={losses[-1]:.4f} grad={grad_norms[-1]:.3f} rows={rows.size}")

        t_eval = time.time()
        pred_train, _ = _predict_raw(model, train_rows, structs, bundle.raw,
                                     args.device, eval_batch, bundle.scaler)
        pred_valid, _ = _predict_raw(model, valid_rows, structs, bundle.raw,
                                     args.device, eval_batch, bundle.scaler)
        pred_test, _ = _predict_raw(model, test_rows, structs, bundle.raw,
                                    args.device, eval_batch, bundle.scaler)
        eval_seconds = time.time() - t_eval
        row: dict[str, Any] = {
            "run_id": args.run_id, "seed": int(args.seed), "epoch": int(epoch),
            "step": int(step),
            "lr": lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]),
                        float(recipe["min_lr"])),
            "epoch_seconds": float(time.time() - t_epoch),
            "eval_seconds": float(eval_seconds),
            "train_loss": float(np.mean(losses)),
            "train_grad_norm": float(np.mean(grad_norms)),
        }
        for split, pred, rows in (("train", pred_train, train_rows),
                                  ("valid", pred_valid, valid_rows),
                                  ("test", pred_test, test_rows)):
            m = _metrics(pred, bundle.labels[rows])
            row[f"{split}_rmse"] = m["rmse"]
            row[f"{split}_mae"] = m["mae"]
        row["valid_minus_train_rmse"] = row["valid_rmse"] - row["train_rmse"]
        curves.append(row)
        C.write_csv(out_dir / "curves.csv", curves)

        better = row["valid_rmse"] < float(best["valid_rmse"]) - 1e-12
        if better:
            best = {"valid_rmse": row["valid_rmse"], "valid_mae": row["valid_mae"],
                    "train_rmse": row["train_rmse"], "epoch": int(epoch)}
            torch.save({
                "state": model.state_dict(), "epoch": int(epoch), "row": row,
                "scaler": bundle.scaler, "ledger": ledger, "config": cfg,
                "init": info["init"], "init_fingerprint_after_support": fingerprint_after,
                "zeroed": zeroed, "data_hashes": bundle.hashes,
                "seed": int(args.seed),
            }, out_dir / "best.pt")
        log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} "
            f"train={row['train_rmse']:.4f} valid={row['valid_rmse']:.4f} "
            f"test={row['test_rmse']:.4f} loss={row['train_loss']:.4f} "
            f"grad={row['train_grad_norm']:.2f} {row['epoch_seconds']:.1f}s")
        if epoch >= min_epochs and (epoch - int(best["epoch"])) >= patience:
            log(f"early stop at epoch {epoch}: best epoch {best['epoch']} "
                f"(patience={patience}, min_epochs={min_epochs})")
            break

    train_seconds = time.time() - t_train_start
    torch.save({
        "state": model.state_dict(), "epoch": int(curves[-1]["epoch"]),
        "row": curves[-1], "scaler": bundle.scaler, "ledger": ledger,
        "config": cfg, "init": info["init"],
        "init_fingerprint_after_support": fingerprint_after,
        "zeroed": zeroed, "data_hashes": bundle.hashes, "seed": int(args.seed),
    }, out_dir / "last.pt")

    best_state = torch.load(out_dir / "best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_state["state"])
    split_data: dict[str, dict[str, Any]] = {}
    pred_parts: dict[str, np.ndarray] = {}
    label_parts: dict[str, np.ndarray] = {}
    std_parts: dict[str, np.ndarray] = {}
    for split, rows in (("train", train_rows), ("valid", valid_rows),
                        ("test", test_rows)):
        rows = np.asarray(rows, np.int64)
        pred_raw, pred_std = _predict_raw(model, rows, structs, bundle.raw,
                                          args.device, eval_batch, bundle.scaler)
        split_data[split] = _metrics(pred_raw, bundle.labels[rows])
        pred_parts[split] = pred_raw
        std_parts[split] = pred_std
        label_parts[split] = bundle.labels[rows].astype(np.float64)
    metrics_rows = []
    for split, split_rows in (("train", train_rows), ("valid", valid_rows),
                              ("test", test_rows)):
        metrics_rows.append({
            "split": split,
            "metric_label": "DEVELOPMENT_TEST" if split == "test" else split.upper(),
            "n": int(np.asarray(split_rows).size),
            "rmse": float(split_data[split]["rmse"]),
            "mae": float(split_data[split]["mae"]),
            "best_epoch": int(best["epoch"]),
        })
    C.write_csv(out_dir / "metrics.csv", metrics_rows)

    all_rows_out = np.concatenate([train_rows, valid_rows, test_rows])
    np.savez(out_dir / "predictions_best.npz",
             row_id=all_rows_out.astype(np.int64),
             split=np.concatenate([np.full(len(train_rows), "train"),
                                   np.full(len(valid_rows), "valid"),
                                   np.full(len(test_rows), "test")]),
             label_raw=np.concatenate([label_parts["train"], label_parts["valid"],
                                       label_parts["test"]]),
             pred_raw=np.concatenate([pred_parts["train"], pred_parts["valid"],
                                      pred_parts["test"]]),
             pred_std=np.concatenate([std_parts["train"], std_parts["valid"],
                                      std_parts["test"]]))

    wall_seconds = time.time() - t_process
    manifest: dict[str, Any] = {
        "run_id": args.run_id, "seed": int(args.seed),
        "smoke": bool(args.max_fit or args.max_eval),
        "test_policy": "DEVELOPMENT_TEST",
        "started_utc": C.now_utc(), "host": socket.gethostname(),
        "git_commit": C.git_commit(), "device": str(args.device),
        "gpu_name": _gpu_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "python": platform.python_version(),
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha([
            "wobr_common.py", "wobr_data.py", "wobr_structure.py", "wobr_sparsemax.py",
            "wobr_tensor.py", "wobr_model.py", "wobr_train.py",
        ]),
        "data_hashes": bundle.hashes, "scaler": bundle.scaler,
        "n_train": int(train_rows.size), "n_valid": int(valid_rows.size),
        "n_test": int(test_rows.size),
        "train_rows_sha256": C.array_sha256(np.asarray(train_rows)),
        "valid_rows_sha256": C.array_sha256(np.asarray(valid_rows)),
        "test_rows_sha256": C.array_sha256(np.asarray(test_rows)),
        "params": ledger, "init": info["init"],
        "init_fingerprint_after_support": fingerprint_after,
        "module_fingerprint_after_support": self_fingerprint,
        "zeroed_categories": zeroed, "unseen": bundle.unseen,
        "structure_stats": stats,
        "prep_seconds": float(prep_seconds), "train_seconds": float(train_seconds),
        "wall_seconds": float(wall_seconds),
        "gpu_hours": float(train_seconds / 3600.0),
        "gpu_hours_total_process": float(wall_seconds / 3600.0),
        "peak_vram_bytes": int(peak_vram),
        "epochs_max": int(epochs), "epochs_run": int(curves[-1]["epoch"]),
        "steps_per_epoch": int(steps_per_epoch),
        "best": best, "best_epoch": int(best["epoch"]),
        "best_valid_rmse": float(best["valid_rmse"]),
        "best_valid_mae": float(best["valid_mae"]),
        "train_rmse_at_best": float(split_data["train"]["rmse"]),
        "train_mae_at_best": float(split_data["train"]["mae"]),
        "valid_rmse": float(split_data["valid"]["rmse"]),
        "valid_mae": float(split_data["valid"]["mae"]),
        "test_rmse": float(split_data["test"]["rmse"]),
        "test_mae": float(split_data["test"]["mae"]),
        "test_metric_label": "DEVELOPMENT_TEST",
        "recipe": recipe,
        "early_stop": {"patience": patience, "min_epochs": min_epochs},
        "checkpoint": _rel(out_dir / "best.pt"),
        "metrics": _rel(out_dir / "metrics.csv"),
        "curves": _rel(out_dir / "curves.csv"),
        "predictions": _rel(out_dir / "predictions_best.npz"),
    }
    C.write_json(out_dir / "run.json", manifest)
    log(f"done valid_rmse={split_data['valid']['rmse']:.4f} "
        f"test_rmse={split_data['test']['rmse']:.4f} best_epoch={best['epoch']} "
        f"train_seconds={train_seconds:.1f} wall_seconds={wall_seconds:.1f}")
    log_handle.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-fit", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = f"wobr_s{args.seed}"
    if args.out is None:
        args.out = str(C.RESULTS_DIR / args.run_id)
    manifest = train(args)
    print(json.dumps(C.jsonable(manifest), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()