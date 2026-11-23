"""Formal training entry point for R2-SENTENCE-REPLICATION-v1.

One model (`SENTENCE_BASE`, imported unchanged from GWIC-v0), one split (D0 or
D1), one seed per process.  Fit rows are the only gradient source; the held-out
side of the chosen split selects the best checkpoint.  Official Valid/Test
labels are never read.

The training loop, optimiser, LR schedule, early-stopping rule and artefact
schema are the GWIC-v0 ones; ``r2_checks.py`` (check D5) verifies bit-for-bit
equality against the frozen ``gwic_train.py`` on a 2-epoch CPU reference run.
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

import r2_common as RC
import r2_data as RD
import gwic_model as M
import gwic_structure as S
import gwic_tensor as T

ARM = "SENTENCE_BASE"
CODE_FILES = ("r2_common.py", "r2_data.py", "r2_train.py")


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak) * (step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(RC.REPO_ROOT))
    except ValueError:
        return str(path)


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def _predict_raw(model: M.GwicModel, rows: Sequence[int],
                 structs: Mapping[int, S.MoleculeStruct], raw: Mapping[str, Any],
                 device: Any, batch_size: int, scaler: Mapping[str, float]
                 ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    pred_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = T.collate(part.tolist(), structs, raw, device)
            out = model(batch, arm=ARM)
            pred_std[start:start + part.size] = out["yhat_std"].double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    if args.arm != ARM:
        raise ValueError(f"this round only runs {ARM}, got {args.arm}")
    RC.assert_frozen_sources()
    cfg = RC.frozen_config()
    recipe = dict(cfg["recipe"])
    spec = dict(cfg["arms"][ARM])
    if bool(spec["use_words"]) or bool(spec["use_geometry"]):
        raise ValueError("frozen SENTENCE_BASE spec changed")
    t_start_utc = RC.now_utc()
    commit = RC.git_commit()

    RC.seed_everything(int(args.seed))
    out_dir = Path(args.out) if args.out else (RC.RESULTS_DIR / str(args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    bundle = RD.load_bundle(args.split, args.atom_scalar_mode)
    atom_dims, bond_dims = RD.GD.atom_bond_dims()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    eval_rows = np.asarray(bundle.eval_rows, np.int64)
    if args.max_fit:
        fit_rows = fit_rows[:int(args.max_fit)]
    if args.max_eval:
        eval_rows = eval_rows[:int(args.max_eval)]
    all_rows = np.union1d(fit_rows, eval_rows)
    y_std = bundle.labels_std()
    scaler = bundle.scaler

    log(f"run={args.run_id} arm={ARM} split={args.split} seed={args.seed} "
        f"device={args.device} host={socket.gethostname()}")
    log(f"data raw={bundle.hashes['raw_cache']['sha256'][:12]} "
        f"split={bundle.hashes['split']['sha256'][:12]} "
        f"fit={fit_rows.size} dev={eval_rows.size} scaler={scaler} "
        f"atom_scalar_mode={args.atom_scalar_mode}")

    t_prep = time.time()
    structs = S.build_all(bundle.raw, all_rows)
    stats = S.structure_stats(structs)
    prep_seconds = time.time() - t_prep
    log(f"prepared structures in {prep_seconds:.1f}s: atoms={stats['n_atoms']} "
        f"words={stats['n_words']} roots_without_words={stats['roots_without_words']}")

    atom_ok, bond_ok = RD.GD.category_support(bundle.raw, fit_rows, atom_dims, bond_dims)
    unseen_dev = RD.GD.unseen_category_counts(bundle.raw, eval_rows, atom_ok, bond_ok)
    model, info = M.build_model(atom_dims, bond_dims, int(args.seed), RD.norm_of(bundle),
                                use_words=False, config=cfg)
    zeroed = model.zero_unseen(atom_ok, bond_ok, atom_dims, bond_dims)
    ledger = info["parameters"]
    if int(ledger["total"]) != RC.PARAMS_TOTAL:
        raise ValueError(f"parameter count changed: {ledger['total']} != {RC.PARAMS_TOTAL}")
    model = model.to(args.device)
    log(f"params={ledger}")
    log(f"zeroed_unseen={zeroed} unseen_dev={unseen_dev}")
    log(f"init={json.dumps(info['init_hashes'], sort_keys=True)}")

    epochs = int(args.epochs or recipe["epochs"])
    batch_mol = int(recipe["batch_molecules"])
    n_fit = int(fit_rows.size)
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
    best = {"dev_rmse": float("inf"), "dev_mae": float("inf"), "fit_rmse": float("inf"),
            "epoch": -1}
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
            rows = fit_rows[order[start:start + batch_mol]]
            batch = T.collate(rows.tolist(), structs, bundle.raw, args.device)
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model(batch, arm=ARM)
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
        pred_fit, _ = _predict_raw(model, fit_rows, structs, bundle.raw, args.device,
                                   eval_batch, scaler)
        pred_dev, _ = _predict_raw(model, eval_rows, structs, bundle.raw, args.device,
                                   eval_batch, scaler)
        eval_seconds = time.time() - t_eval
        row: dict[str, Any] = {
            "run_id": args.run_id, "arm": ARM, "split": args.split,
            "seed": int(args.seed), "epoch": int(epoch), "step": int(step),
            "lr": lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]),
                        float(recipe["min_lr"])),
            "epoch_seconds": float(time.time() - t_epoch),
            "eval_seconds": float(eval_seconds),
            "train_loss": float(np.mean(losses)),
            "train_grad_norm": float(np.mean(grad_norms)),
            "fit_rmse": RC.rmse(pred_fit, bundle.labels[fit_rows]),
            "fit_mae": RC.mae(pred_fit, bundle.labels[fit_rows]),
            "dev_rmse": RC.rmse(pred_dev, bundle.labels[eval_rows]),
            "dev_mae": RC.mae(pred_dev, bundle.labels[eval_rows]),
        }
        row["generalization_gap_rmse"] = row["dev_rmse"] - row["fit_rmse"]
        curves.append(row)
        RC.write_csv(out_dir / "curves.csv", curves)
        better = row["dev_rmse"] < float(best["dev_rmse"]) - 1e-12
        if better:
            best = {"dev_rmse": row["dev_rmse"], "dev_mae": row["dev_mae"],
                    "fit_rmse": row["fit_rmse"], "epoch": int(epoch)}
            torch.save({
                "state": model.state_dict(), "epoch": int(epoch), "row": row,
                "scaler": scaler, "ledger": ledger, "arm": ARM, "use_words": False,
                "spec": spec, "init_hashes": info["init_hashes"], "zeroed": zeroed,
                "data_hashes": bundle.hashes, "config": cfg, "split": bundle.split,
                "atom_scalar_mode": args.atom_scalar_mode,
            }, out_dir / "state_best.pt")
        log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} fit={row['fit_rmse']:.4f} "
            f"dev={row['dev_rmse']:.4f} loss={row['train_loss']:.4f} "
            f"grad={row['train_grad_norm']:.2f} {row['epoch_seconds']:.1f}s")
        if epoch >= min_epochs and (epoch - int(best["epoch"])) >= patience:
            log(f"early stop at epoch {epoch}: best epoch {best['epoch']} "
                f"(patience={patience}, min_epochs={min_epochs})")
            break
    train_seconds = time.time() - t_train_start
    torch.save({
        "state": model.state_dict(), "epoch": int(curves[-1]["epoch"]),
        "row": curves[-1], "scaler": scaler, "ledger": ledger, "arm": ARM,
        "use_words": False, "spec": spec, "init_hashes": info["init_hashes"],
        "zeroed": zeroed, "data_hashes": bundle.hashes, "config": cfg,
        "split": bundle.split, "atom_scalar_mode": args.atom_scalar_mode,
    }, out_dir / "state_final.pt")

    best_state = torch.load(out_dir / "state_best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_state["state"])
    pred_fit, pred_fit_std = _predict_raw(model, fit_rows, structs, bundle.raw,
                                          args.device, eval_batch, scaler)
    pred_dev, pred_dev_std = _predict_raw(model, eval_rows, structs, bundle.raw,
                                          args.device, eval_batch, scaler)
    for tag, rows, pred, pred_std in (("fit", fit_rows, pred_fit, pred_fit_std),
                                      ("dev", eval_rows, pred_dev, pred_dev_std)):
        records = []
        for i, r in enumerate(rows.tolist()):
            st = structs[int(r)]
            y = float(bundle.labels[int(r)])
            records.append({
                "row_id": int(r),
                "smiles": str(np.asarray(bundle.raw["smiles"])[int(r)]),
                "label_raw": y, "pred_raw": float(pred[i]),
                "pred_std": float(pred_std[i]),
                "abs_error": float(abs(pred[i] - y)),
                "sq_error": float((pred[i] - y) ** 2),
                "n_atoms": st.n_atoms, "n_bonds": st.n_bonds, "n_words": st.n_words,
            })
        RC.write_csv(out_dir / f"predictions_best_{tag}.csv", records)
        np.savez(out_dir / f"predictions_best_{tag}.npz",
                 row_id=np.asarray([x["row_id"] for x in records], np.int64),
                 pred=np.asarray([x["pred_raw"] for x in records], np.float64),
                 pred_std=np.asarray([x["pred_std"] for x in records], np.float64),
                 label=np.asarray([x["label_raw"] for x in records], np.float64))

    fit_rmse = RC.rmse(pred_fit, bundle.labels[fit_rows])
    fit_mae = RC.mae(pred_fit, bundle.labels[fit_rows])
    dev_rmse = RC.rmse(pred_dev, bundle.labels[eval_rows])
    dev_mae = RC.mae(pred_dev, bundle.labels[eval_rows])
    wall_seconds = time.time() - t_process
    manifest: dict[str, Any] = {
        "run_id": args.run_id, "arm": ARM, "split_name": args.split,
        "seed": int(args.seed),
        "smoke": bool(args.max_fit or args.max_eval),
        "started_utc": t_start_utc, "finished_utc": RC.now_utc(),
        "host": socket.gethostname(), "git_commit": commit,
        "device": str(args.device), "gpu_name": _gpu_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "driver": _driver_version(),
        "config_sha256": RC.GWIC_CONFIG_SHA256,
        "r2_config_sha256": RC.sha256_file(RC.CONFIG_PATH),
        "frozen_gwic_code_sha256": RC.frozen_gwic_code_sha(),
        "code_sha256": RC.code_sha(list(CODE_FILES)),
        "data_hashes": bundle.hashes,
        "split": bundle.split,
        "atom_scalar_mode": args.atom_scalar_mode,
        "normalization": {
            "target": "fit-only",
            "graph_scalars": "fit-only",
            "atom_scalars": ("fit-only" if args.atom_scalar_mode == "fit"
                             else "fit union dev (GWIC-v0 implementation, audit mode)"),
            "category_support": "fit-only",
        },
        "scaler": scaler, "n_fit": int(fit_rows.size), "n_dev": int(eval_rows.size),
        "fit_rows_sha256": RC.array_sha256(fit_rows),
        "dev_rows_sha256": RC.array_sha256(eval_rows),
        "fit_atom_scalars_sha256": RC.array_sha256(bundle.atom_scalar_mean,
                                                   bundle.atom_scalar_scale),
        "graph_norm_sha256": RC.array_sha256(bundle.graph_mean, bundle.graph_scale),
        "params": ledger, "init_hashes": info["init_hashes"],
        "zeroed_categories": zeroed, "unseen_dev_positions": unseen_dev,
        "structure_stats": stats,
        "prep_seconds": float(prep_seconds), "train_seconds": float(train_seconds),
        "wall_seconds": float(wall_seconds),
        "gpu_hours": float(train_seconds / 3600.0),
        "gpu_hours_total_process": float(wall_seconds / 3600.0),
        "peak_vram_bytes": int(peak_vram),
        "epochs_max": int(epochs), "epochs_run": int(curves[-1]["epoch"]),
        "steps_per_epoch": int(steps_per_epoch),
        "best": best, "best_epoch": int(best["epoch"]),
        "best_dev_rmse": float(best["dev_rmse"]), "best_dev_mae": float(best["dev_mae"]),
        "train_rmse_at_best": float(fit_rmse), "train_mae_at_best": float(fit_mae),
        "dev_rmse": float(dev_rmse), "dev_mae": float(dev_mae),
        "recipe": recipe, "early_stop": {"patience": patience, "min_epochs": min_epochs},
        "checkpoint": _rel(out_dir / "state_best.pt"),
        "curves": _rel(out_dir / "curves.csv"),
        "predictions_dev": _rel(out_dir / "predictions_best_dev.csv"),
        "predictions_fit": _rel(out_dir / "predictions_best_fit.csv"),
    }
    RC.write_json(out_dir / "manifest.json", manifest)
    log(f"done dev_rmse={dev_rmse:.4f} dev_mae={dev_mae:.4f} best_epoch={best['epoch']} "
        f"train_seconds={train_seconds:.1f} wall_seconds={wall_seconds:.1f}")
    log_handle.close()
    return manifest


def _driver_version() -> str:
    try:
        out = __import__("subprocess").check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()
        return out[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["D0", "D1"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", default=ARM, choices=[ARM])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atom-scalar-mode", default="fit",
                        choices=list(RD.ATOM_SCALAR_MODES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-fit", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = f"{args.split}_S{args.seed}"
    if args.out is None:
        args.out = str(RC.RESULTS_DIR / args.run_id)
    manifest = train(args)
    print(json.dumps(RC.jsonable(manifest), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()