"""Formal single-seed training entry point for BBPA-v0.

One arm per process.  D0 train-sub (2856) is the only fit set; D0 Dev (504) is
evaluated every epoch and selects the best checkpoint.  Official Valid / Test
labels are never read.
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

import bbpa_common as C
import bbpa_data as D
import bbpa_model as M
import bbpa_structure as S
import bbpa_tensor as T


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


def _predict_raw(model: M.BBPAModel, rows: Sequence[int],
                 mols: Mapping[int, T.MoleculeTensors], raw: Mapping[str, Any],
                 mode: str, device: Any, batch_size: int,
                 scaler: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    pred_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = T.collate(part.tolist(), mols, raw, device)
            out = model.forward_batch(batch, mode)
            pred_std[start:start + part.size] = out["yhat_std"].double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def _row_structures(decs: Sequence[S.Decomp], rows: Sequence[int]) -> dict[int, dict[str, Any]]:
    out = {}
    for row, d in zip(np.asarray(rows, np.int64).tolist(), decs):
        out[int(row)] = {
            "n_blocks": int(len(d.blocks)),
            "n_bridges": int(len(d.bridges)),
            "block_tree_diameter": int(max([S.tree_diameter(d, c) for c in d.components] or [0])),
            "n_atoms": int(d.n_atoms),
            "n_bonds": int(d.n_bonds),
        }
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    C.seed_everything(int(args.seed))
    cfg = C.load_config()
    recipe = dict(cfg["recipe"])
    arms = cfg["arms"]
    if args.arm not in arms:
        raise KeyError(f"unknown arm {args.arm}")
    spec = dict(arms[args.arm])
    mode = str(spec["mode"])
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
    atom_dims, bond_dims = D.atom_bond_dims()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    eval_rows = np.asarray(bundle.eval_rows, np.int64)
    if args.max_fit:
        fit_rows = fit_rows[:int(args.max_fit)]
    if args.max_eval:
        eval_rows = eval_rows[:int(args.max_eval)]
    all_rows = np.union1d(fit_rows, eval_rows)
    y_std = bundle.labels_std()

    log(f"run={args.run_id} arm={args.arm} mode={mode} seed={args.seed} "
        f"device={args.device} host={socket.gethostname()}")
    log(f"data raw={bundle.hashes['raw_cache']['sha256'][:12]} "
        f"split={bundle.hashes['split']['sha256'][:12]} "
        f"fit={fit_rows.size} dev={eval_rows.size} scaler={bundle.scaler}")

    # label-free structure tensors (CPU precompute, moved per batch)
    t_prep = time.time()
    decs = S.build_all(bundle.raw, all_rows)
    mols = {int(r): T.build_molecule_tensors(d, int(r))
            for r, d in zip(all_rows.tolist(), decs)}
    structures = _row_structures(decs, all_rows)
    stats = S.structure_stats(decs, all_rows.size)
    prep_seconds = time.time() - t_prep
    log(f"prepared structures in {prep_seconds:.1f}s: "
        f"bridge_coverage={stats['bridge_coverage_fraction']:.3f} "
        f"diameter>=3={stats['fraction_with_diameter_ge_3']:.3f}")

    atom_ok, bond_ok = D.category_support(bundle.raw, fit_rows, atom_dims, bond_dims)
    unseen_dev = D.unseen_category_counts(bundle.raw, eval_rows, atom_ok, bond_ok)
    model, init_hashes = M.build_model(atom_dims, bond_dims, int(args.seed), cfg)
    zeroed = model.zero_unseen(atom_ok, bond_ok)
    init_hashes_after_zero = model.shared_hashes()
    ledger = M.parameter_ledger(model)
    model = model.to(args.device)
    log(f"params={ledger}")
    log(f"zeroed_unseen={zeroed} unseen_dev={unseen_dev}")
    log(f"shared_init={json.dumps(init_hashes_after_zero, sort_keys=True)}")

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
            batch = T.collate(rows.tolist(), mols, bundle.raw, args.device)
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model.forward_batch(batch, mode)
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
                log(f"step1 loss={losses[-1]:.4f} grad={grad_norms[-1]:.3f} "
                    f"rows={rows.size}")
        eval_seconds = 0.0
        t_eval = time.time()
        pred_fit, _ = _predict_raw(model, fit_rows, mols, bundle.raw, mode, args.device,
                                   eval_batch, bundle.scaler)
        pred_dev, _ = _predict_raw(model, eval_rows, mols, bundle.raw, mode, args.device,
                                   eval_batch, bundle.scaler)
        eval_seconds = time.time() - t_eval
        row: dict[str, Any] = {
            "run_id": args.run_id, "arm": args.arm, "mode": mode, "seed": int(args.seed),
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
                "scaler": bundle.scaler, "ledger": ledger,
                "init_hashes": init_hashes_after_zero, "arm": args.arm, "mode": mode,
                "spec": spec, "zeroed": zeroed,
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
        "init_hashes": init_hashes_after_zero, "arm": args.arm, "mode": mode,
        "spec": spec, "zeroed": zeroed, "data_hashes": bundle.hashes, "config": cfg,
    }, out_dir / "state_final.pt")

    # best-Dev predictions
    best_state = torch.load(out_dir / "state_best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_state["state"])
    pred_fit, pred_fit_std = _predict_raw(model, fit_rows, mols, bundle.raw, mode,
                                          args.device, eval_batch, bundle.scaler)
    pred_dev, pred_dev_std = _predict_raw(model, eval_rows, mols, bundle.raw, mode,
                                          args.device, eval_batch, bundle.scaler)
    for tag, rows, pred, pred_std in (("fit", fit_rows, pred_fit, pred_fit_std),
                                      ("dev", eval_rows, pred_dev, pred_dev_std)):
        records = []
        for i, r in enumerate(rows.tolist()):
            st = structures[int(r)]
            y = float(bundle.labels[int(r)])
            records.append({
                "row_id": int(r), "smiles": str(np.asarray(bundle.raw["smiles"])[int(r)]),
                "label_raw": y, "pred_raw": float(pred[i]), "pred_std": float(pred_std[i]),
                "abs_error": float(abs(pred[i] - y)),
                "sq_error": float((pred[i] - y) ** 2),
                "n_blocks": st["n_blocks"], "n_bridges": st["n_bridges"],
                "block_tree_diameter": st["block_tree_diameter"],
                "n_atoms": st["n_atoms"], "n_bonds": st["n_bonds"],
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
        "run_id": args.run_id, "arm": args.arm, "mode": mode, "seed": int(args.seed),
        "smoke": bool(args.max_fit or args.max_eval),
        "started_utc": C.now_utc(), "host": socket.gethostname(),
        "git_commit": C.git_commit(), "device": str(args.device),
        "gpu_name": _gpu_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "python": platform.python_version(),
        "config_sha256": C.sha256_json(cfg), "code_sha256": C.code_sha([
            "bbpa_common.py", "bbpa_data.py", "bbpa_structure.py", "bbpa_tensor.py",
            "bbpa_model.py", "bbpa_train.py",
        ]),
        "data_hashes": bundle.hashes, "scaler": bundle.scaler,
        "n_fit": int(fit_rows.size), "n_dev": int(eval_rows.size),
        "fit_rows_sha256": C.array_sha256(fit_rows),
        "dev_rows_sha256": C.array_sha256(eval_rows),
        "params": ledger, "shared_init": init_hashes_after_zero,
        "shared_init_before_zero": init_hashes,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["BLOCK_BAG", "LOCAL_PORT", "TREE_PORT"])
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