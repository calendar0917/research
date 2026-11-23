"""Formal single-seed training entry point for RICE-v0.

One arm per process.  D0 train-sub (2856 molecules) is the only fit set; D0 Dev
(504 molecules) is evaluated every epoch and selects the best checkpoint.
Official Valid / Test labels are never read.
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

import rice_common as C
import rice_data as D
import rice_model as M
import rice_structure as S
import rice_tensor as T


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


def collate(rows: Sequence[int], structs: Mapping[int, S.MoleculeStruct],
            raw: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return T.collate(rows, structs, raw, device)


def predict_raw(model: M.RiceModel, rows: Sequence[int],
                structs: Mapping[int, S.MoleculeStruct], raw: Mapping[str, Any],
                arm: str, device: Any, batch_size: int,
                scaler: Mapping[str, float], **forward_kwargs: Any
                ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    pred_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = collate(part.tolist(), structs, raw, device)
            out = model(batch, arm=arm, **forward_kwargs)
            pred_std[start:start + part.size] = out["yhat_std"].double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    C.seed_everything(int(args.seed))
    cfg = C.load_config()
    recipe = dict(cfg["recipe"])
    if args.arm not in cfg["arms"]:
        raise KeyError(f"unknown arm {args.arm}")
    out_dir = Path(args.out) if args.out else (C.RESULTS_DIR / str(args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    bundle = D.load_bundle()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    eval_rows = np.asarray(bundle.eval_rows, np.int64)
    if args.max_fit:
        fit_rows = fit_rows[:int(args.max_fit)]
    if args.max_eval:
        eval_rows = eval_rows[:int(args.max_eval)]
    all_rows = np.union1d(fit_rows, eval_rows)
    y_std = bundle.labels_std()

    log(f"run={args.run_id} arm={args.arm} seed={args.seed} device={args.device} "
        f"host={socket.gethostname()}")
    log(f"data raw={bundle.hashes['raw_cache']['sha256'][:12]} "
        f"split={bundle.hashes['split']['split_sha256'][:12]} "
        f"fit={fit_rows.size} dev={eval_rows.size} scaler={bundle.scaler}")

    t_prep = time.time()
    structs = S.build_all(bundle, all_rows)
    stats = S.structure_stats(structs)
    prep_seconds = time.time() - t_prep
    log(f"prepared structures in {prep_seconds:.1f}s words={stats['words_by_size']} "
        f"roots_with_size={stats['roots_with_size']}")

    atom_ok, bond_ok = D.category_support(bundle, fit_rows)
    unseen_dev = D.unseen_category_counts(bundle, eval_rows, atom_ok, bond_ok)
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, int(args.seed),
                                D.normalisers(bundle), config=cfg)
    zeroed = model.zero_unseen(atom_ok, bond_ok)
    ledger = info["parameters"]
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
    first_batch_grads: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_fit)
        losses: list[float] = []
        grad_norms: list[float] = []
        for start in range(0, n_fit, batch_mol):
            rows = fit_rows[order[start:start + batch_mol]]
            batch = collate(rows.tolist(), structs, bundle.raw, args.device)
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model(batch, arm=args.arm)
            target = torch.as_tensor(y_std[rows], device=args.device, dtype=torch.float32)
            loss = torch.mean((out["yhat_std"] - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                first_batch_grads = _grad_report(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(recipe["gradient_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
            step += 1
            if torch.cuda.is_available() and args.device != "cpu":
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            if step == 1:
                log(f"step1 loss={losses[-1]:.4f} grad={grad_norms[-1]:.3f} rows={rows.size} "
                    f"words={batch['n_words']}")
                log(f"grads={json.dumps(first_batch_grads, sort_keys=True)}")
        t_eval = time.time()
        pred_fit, _ = predict_raw(model, fit_rows, structs, bundle.raw, args.arm,
                                  args.device, eval_batch, bundle.scaler)
        pred_dev, _ = predict_raw(model, eval_rows, structs, bundle.raw, args.arm,
                                  args.device, eval_batch, bundle.scaler)
        eval_seconds = time.time() - t_eval
        row: dict[str, Any] = {
            "run_id": args.run_id, "arm": args.arm, "seed": int(args.seed),
            "epoch": int(epoch), "step": int(step),
            "lr": lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]),
                        float(recipe["min_lr"])),
            "epoch_seconds": float(time.time() - t_epoch),
            "eval_seconds": float(eval_seconds),
            "train_loss": float(np.mean(losses)),
            "train_grad_norm": float(np.mean(grad_norms)),
            "fit_rmse": C.rmse(pred_fit, bundle.labels[fit_rows]),
            "fit_mae": C.mae(pred_fit, bundle.labels[fit_rows]),
            "dev_rmse": C.rmse(pred_dev, bundle.labels[eval_rows]),
            "dev_mae": C.mae(pred_dev, bundle.labels[eval_rows]),
        }
        row["generalization_gap_rmse"] = row["dev_rmse"] - row["fit_rmse"]
        curves.append(row)
        C.write_csv(out_dir / "curves.csv", curves)
        better = row["dev_rmse"] < float(best["dev_rmse"]) - 1e-12
        if better:
            best = {"dev_rmse": row["dev_rmse"], "dev_mae": row["dev_mae"],
                    "fit_rmse": row["fit_rmse"], "epoch": int(epoch)}
            torch.save({
                "state": model.state_dict(), "epoch": int(epoch), "row": row,
                "scaler": bundle.scaler, "ledger": ledger, "arm": args.arm,
                "spec": dict(cfg["arms"][args.arm]),
                "init_hashes": info["init_hashes"], "zeroed": zeroed,
                "data_hashes": bundle.hashes, "config": cfg,
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
        "row": curves[-1], "scaler": bundle.scaler, "ledger": ledger,
        "arm": args.arm, "spec": dict(cfg["arms"][args.arm]),
        "init_hashes": info["init_hashes"], "zeroed": zeroed,
        "data_hashes": bundle.hashes, "config": cfg,
    }, out_dir / "state_final.pt")

    best_state = torch.load(out_dir / "state_best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_state["state"])
    pred_fit, pred_fit_std = predict_raw(model, fit_rows, structs, bundle.raw, args.arm,
                                         args.device, eval_batch, bundle.scaler)
    pred_dev, pred_dev_std = predict_raw(model, eval_rows, structs, bundle.raw, args.arm,
                                         args.device, eval_batch, bundle.scaler)
    for tag, rows, pred, pred_std in (("fit", fit_rows, pred_fit, pred_fit_std),
                                      ("dev", eval_rows, pred_dev, pred_dev_std)):
        records = []
        for i, r in enumerate(rows.tolist()):
            st = structs[int(r)]
            y = float(bundle.labels[int(r)])
            records.append({
                "row_id": int(r), "smiles": str(np.asarray(bundle.raw["smiles"])[int(r)]),
                "label_raw": y, "pred_raw": float(pred[i]), "pred_std": float(pred_std[i]),
                "abs_error": float(abs(pred[i] - y)),
                "sq_error": float((pred[i] - y) ** 2),
                "n_atoms": st.n_atoms, "n_bonds": st.n_bonds, "n_words": st.n_words,
            })
        C.write_csv(out_dir / f"predictions_best_{tag}.csv", records)
        np.savez(out_dir / f"predictions_best_{tag}.npz",
                 row_id=np.asarray([x["row_id"] for x in records], np.int64),
                 pred=np.asarray([x["pred_raw"] for x in records], np.float64),
                 pred_std=np.asarray([x["pred_std"] for x in records], np.float64),
                 label=np.asarray([x["label_raw"] for x in records], np.float64))

    fit_rmse = C.rmse(pred_fit, bundle.labels[fit_rows])
    fit_mae = C.mae(pred_fit, bundle.labels[fit_rows])
    dev_rmse = C.rmse(pred_dev, bundle.labels[eval_rows])
    dev_mae = C.mae(pred_dev, bundle.labels[eval_rows])
    wall_seconds = time.time() - t_process
    manifest: dict[str, Any] = {
        "run_id": args.run_id, "arm": args.arm, "seed": int(args.seed),
        "smoke": bool(args.max_fit or args.max_eval),
        "started_utc": C.now_utc(), "host": socket.gethostname(),
        "git_commit": C.git_commit(), "git_dirty": C.git_dirty(),
        "device": str(args.device), "gpu_name": _gpu_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha(["rice_common.py", "rice_data.py", "rice_structure.py",
                                   "rice_energy.py", "rice_tensor.py", "rice_model.py",
                                   "rice_train.py"]),
        "data_hashes": bundle.hashes, "scaler": bundle.scaler,
        "n_fit": int(fit_rows.size), "n_dev": int(eval_rows.size),
        "fit_rows_sha256": C.array_sha256(fit_rows),
        "dev_rows_sha256": C.array_sha256(eval_rows),
        "params": ledger, "init_hashes": info["init_hashes"],
        "zeroed_categories": zeroed, "unseen_dev_positions": unseen_dev,
        "structure_stats": stats,
        "first_batch_grads": first_batch_grads,
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
        "recipe": recipe,
        "early_stop": {"patience": patience, "min_epochs": min_epochs},
        "checkpoint": _rel(out_dir / "state_best.pt"),
        "curves": _rel(out_dir / "curves.csv"),
        "predictions_dev": _rel(out_dir / "predictions_best_dev.csv"),
        "predictions_fit": _rel(out_dir / "predictions_best_fit.csv"),
    }
    C.write_json(out_dir / "manifest.json", manifest)
    log(f"done dev_rmse={dev_rmse:.4f} dev_mae={dev_mae:.4f} best_epoch={best['epoch']} "
        f"train_seconds={train_seconds:.1f} wall_seconds={wall_seconds:.1f}")
    log_handle.close()
    return manifest


def _grad_report(model: M.RiceModel) -> dict[str, Any]:
    def norm(params: Sequence[torch.nn.Parameter]) -> float:
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return math.sqrt(total)

    return {
        "atom_emb": norm(list(model.atom_emb.parameters())),
        "bond_emb": norm(list(model.bond_emb.parameters())),
        "atom_mlp": norm(list(model.atom_mlp.parameters())),
        "bond_mlp": norm(list(model.bond_mlp.parameters())),
        "order_norm": norm(list(model.order_norm.parameters())),
        "sentence_mlp": norm(list(model.sentence_mlp.parameters())),
        "readout_mlp": norm(list(model.readout_mlp.parameters())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(M.ARMS))
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
        args.run_id = f"{args.arm}_S{args.seed}"
    if args.out is None:
        args.out = str(C.RESULTS_DIR / args.run_id)
    manifest = train(args)
    print(json.dumps(C.jsonable(manifest), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()