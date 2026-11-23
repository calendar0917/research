"""Formal training entry point for the route-resolution sprint.

One arm, one split, one seed per process.  Fit rows are the only gradient
source; the held-out side of the chosen split is evaluated every epoch and
selects the earliest best-RMSE checkpoint.  Official Valid/Test labels are never
read.

Recipes are frozen in ``configs/sprint.json``:

* E/F: RICE-v0 recipe (AdamW 3e-4 -> 1e-5, wd 1e-4, batch 64, warmup 5,
  cosine 240, min 80, patience 45, clip 5, dropout 0.1);
* G:   AdamW 1e-3 -> 1e-5, wd 1e-5, batch 64, warmup 5, cosine 240, min 80,
  patience 45, clip 5, trunk dropout 0, readout dropout 0.1.
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

import sr_common as C
import sr_data as D
import sr_fresponse as FR
import sr_gnn as G
import sr_model as M
import sr_struct as ST
import sr_tensor as T

EF_ARMS = set(M.ARMS)
G_ARMS = set(G.MODEL_ARMS)
CODE_FILES = ("sr_common.py", "sr_data.py", "sr_struct.py", "sr_fresponse.py",
              "sr_tensor.py", "sr_model.py", "sr_gnn.py", "sr_train.py")


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float,
          minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak) * (step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def family_of(arm: str) -> str:
    if arm in EF_ARMS:
        return arm[0]
    if arm in G_ARMS:
        return "G"
    raise KeyError(arm)


def collate(arm: str, rows: Sequence[int], structs: Mapping[int, Any],
            raw: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if arm in EF_ARMS:
        return T.collate_ef(rows, structs, raw, device)
    return G.collate(rows, structs, G.mode_of(arm), device)


def predict_raw(model: Any, arm: str, rows: Sequence[int],
                structs: Mapping[int, Any], raw: Mapping[str, Any], device: Any,
                batch_size: int, scaler: Mapping[str, float]
                ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = np.asarray(rows, np.int64)
    pred_std = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = collate(arm, part.tolist(), structs, raw, device)
            out = model(batch)
            pred_std[start:start + part.size] = out["yhat_std"].double().cpu().numpy()
    pred_raw = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return pred_raw, pred_std


def _grad_report(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            out[name.split(".")[0]] = out.get(name.split(".")[0], 0.0) + \
                float(param.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in out.items()}


def _seed_state(seed: int) -> dict[str, Any]:
    state = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    C.seed_everything(int(args.seed))
    C.assert_pins()
    cfg = C.load_config()
    arm = str(args.arm)
    family = family_of(arm)
    recipe = dict(cfg["recipe_gnn"] if family == "G" else cfg["recipe_ef"])
    if args.epochs:
        recipe["epochs"] = int(args.epochs)
    out_dir = Path(args.out) if args.out else (C.RESULTS_DIR / str(args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    bundle = D.load_bundle(args.split)
    atom_dims, bond_dims = D.feature_dims()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    eval_rows = np.asarray(bundle.eval_rows, np.int64)
    if args.max_fit:
        fit_rows = fit_rows[:int(args.max_fit)]
    if args.max_eval:
        eval_rows = eval_rows[:int(args.max_eval)]
    all_rows = np.union1d(fit_rows, eval_rows)
    y_std = bundle.labels_std()
    scaler = bundle.scaler

    log(f"run={args.run_id} arm={arm} split={args.split} seed={args.seed} "
        f"device={args.device} host={socket.gethostname()}")
    log(f"data split={D.split_identity(bundle)['identity_sha256'][:12]} "
        f"fit={fit_rows.size} dev={eval_rows.size} scaler={scaler}")

    t_prep = time.time()
    structs = ST.build_structures(bundle, all_rows.tolist())
    stats = ST.structure_stats(structs)
    registry_info: dict[str, Any] = {}
    registry = None
    if family in ("E", "F"):
        registry = ST.build_registry([structs[int(r)] for r in all_rows.tolist()],
                                     atom_dims, bond_dims)
        registry_info = ST.registry_summary(registry)
        C.write_json(out_dir / "registry.json", registry_info)
    else:
        for row in all_rows.tolist():
            G.prepare_ranges(structs[int(row)])
        registry_info = {"ego2_nodes": int(sum(structs[int(r)].ego2["atom"].size
                                               for r in all_rows.tolist())),
                         "ego3_nodes": int(sum(structs[int(r)].ego3["atom"].size
                                               for r in all_rows.tolist()))}
    prep_seconds = time.time() - t_prep
    log(f"prepared structures in {prep_seconds:.1f}s words={stats['words_by_size']} "
        f"registry={registry_info}")

    atom_ok, bond_ok = D.support(bundle)
    unseen_dev = D.unseen(bundle, eval_rows.tolist(), atom_ok, bond_ok)
    if family in ("E", "F"):
        model, info = M.build_family_models(cfg, atom_dims, bond_dims, int(args.seed),
                                            D.norm_of(bundle), arm)
        if family == "F":
            model.set_registry(FR.registry_tensors(registry, args.device))
    else:
        model, info = G.build_model(arm, atom_dims, bond_dims, int(args.seed),
                                    D.norm_of(bundle), cfg)
    zeroed = model.zero_unseen(atom_ok, bond_ok)
    ledger = info["parameters"]
    model = model.to(args.device)
    if family == "F":
        model.set_registry(FR.registry_tensors(registry, args.device))
    expected = cfg["expected_params"].get(arm)
    if expected is not None and int(ledger["total"]) != int(expected):
        raise ValueError(f"parameter count changed: {ledger['total']} != {expected}")
    log(f"params={ledger}")
    log(f"zeroed_unseen={zeroed} unseen_dev={unseen_dev}")
    log(f"init={json.dumps(info['init_hashes'], sort_keys=True)}")

    epochs = int(recipe["epochs"])
    batch_mol = int(recipe["batch_molecules"])
    n_fit = int(fit_rows.size)
    steps_per_epoch = int(math.ceil(n_fit / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    rng = np.random.default_rng(int(args.seed) + 101)
    patience = int(recipe["patience"])
    min_epochs = int(recipe["min_epochs"])
    eval_batch = int(recipe["eval_batch"])

    curves: list[dict[str, Any]] = []
    best = {"dev_rmse": float("inf"), "dev_mae": float("inf"),
            "fit_rmse": float("inf"), "epoch": -1}
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
            batch = collate(arm, rows.tolist(), structs, bundle.raw, args.device)
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model(batch)
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
                log(f"step1 loss={losses[-1]:.4f} grad={grad_norms[-1]:.3f} "
                    f"rows={rows.size} atoms={batch['n_atoms']}")
                log(f"grads={json.dumps(first_batch_grads, sort_keys=True)}")
        t_eval = time.time()
        pred_fit, _ = predict_raw(model, arm, fit_rows, structs, bundle.raw,
                                  args.device, eval_batch, scaler)
        pred_dev, _ = predict_raw(model, arm, eval_rows, structs, bundle.raw,
                                  args.device, eval_batch, scaler)
        eval_seconds = time.time() - t_eval
        row: dict[str, Any] = {
            "run_id": args.run_id, "arm": arm, "split": args.split,
            "seed": int(args.seed), "epoch": int(epoch), "step": int(step),
            "lr": lr_at(step - 1, warmup_steps, total_steps,
                        float(recipe["peak_lr"]), float(recipe["min_lr"])),
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
                "scaler": scaler, "ledger": ledger, "arm": arm, "family": family,
                "spec": recipe, "init_hashes": info["init_hashes"],
                "zeroed": zeroed, "data": D.split_identity(bundle),
                "config": cfg,
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
        "row": curves[-1], "scaler": scaler, "ledger": ledger, "arm": arm,
        "family": family, "spec": recipe, "init_hashes": info["init_hashes"],
        "zeroed": zeroed, "data": D.split_identity(bundle), "config": cfg,
        "optimizer": opt.state_dict(), "optimizer_steps": int(step),
        "rng": _seed_state(int(args.seed)),
    }, out_dir / "state_final.pt")

    best_state = torch.load(out_dir / "state_best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_state["state"])
    pred_fit, pred_fit_std = predict_raw(model, arm, fit_rows, structs, bundle.raw,
                                        args.device, eval_batch, scaler)
    pred_dev, pred_dev_std = predict_raw(model, arm, eval_rows, structs, bundle.raw,
                                        args.device, eval_batch, scaler)
    smiles = np.asarray(bundle.raw["smiles"])
    for tag, rows, pred, pred_std in (("fit", fit_rows, pred_fit, pred_fit_std),
                                      ("dev", eval_rows, pred_dev, pred_dev_std)):
        records = []
        for i, r in enumerate(rows.tolist()):
            st = structs[int(r)]
            y = float(bundle.labels[int(r)])
            records.append({
                "row_id": int(r), "smiles": str(smiles[int(r)]),
                "label_raw": y, "pred_raw": float(pred[i]),
                "pred_std": float(pred_std[i]),
                "abs_error": float(abs(pred[i] - y)),
                "sq_error": float((pred[i] - y) ** 2),
                "n_atoms": st.n_atoms, "n_bonds": st.n_bonds,
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
        "run_id": args.run_id, "arm": arm, "family": family,
        "split": args.split, "seed": int(args.seed),
        "smoke": bool(args.max_fit or args.max_eval),
        "started_utc": C.now_utc(), "host": socket.gethostname(),
        "git_commit": C.git_commit(), "git_dirty": C.git_dirty(),
        "device": str(args.device), "gpu_name": _gpu_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha(CODE_FILES),
        "data": D.split_identity(bundle), "scaler": scaler,
        "n_fit": int(fit_rows.size), "n_dev": int(eval_rows.size),
        "fit_rows_sha256": C.array_sha256(fit_rows),
        "dev_rows_sha256": C.array_sha256(eval_rows),
        "params": ledger, "init_hashes": info["init_hashes"],
        "zeroed_categories": zeroed, "unseen_dev_positions": unseen_dev,
        "structure_stats": stats, "registry": registry_info,
        "first_batch_grads": first_batch_grads,
        "prep_seconds": float(prep_seconds), "train_seconds": float(train_seconds),
        "wall_seconds": float(wall_seconds),
        "gpu_hours": float(train_seconds / 3600.0),
        "gpu_hours_total_process": float(wall_seconds / 3600.0),
        "peak_vram_bytes": int(peak_vram),
        "epochs_max": int(epochs), "epochs_run": int(curves[-1]["epoch"]),
        "steps_per_epoch": int(steps_per_epoch), "optimizer_steps": int(step),
        "best": best, "best_epoch": int(best["epoch"]),
        "best_dev_rmse": float(best["dev_rmse"]),
        "best_dev_mae": float(best["dev_mae"]),
        "train_rmse_at_best": float(fit_rmse),
        "train_mae_at_best": float(fit_mae),
        "dev_rmse": float(dev_rmse), "dev_mae": float(dev_mae),
        "recipe": recipe,
        "checkpoint": C.rel(out_dir / "state_best.pt"),
        "curves": C.rel(out_dir / "curves.csv"),
        "predictions_dev": C.rel(out_dir / "predictions_best_dev.npz"),
        "predictions_fit": C.rel(out_dir / "predictions_best_fit.npz"),
    }
    C.write_json(out_dir / "manifest.json", manifest)
    log(f"done dev_rmse={dev_rmse:.4f} dev_mae={dev_mae:.4f} "
        f"best_epoch={best['epoch']} train_seconds={train_seconds:.1f} "
        f"wall_seconds={wall_seconds:.1f}")
    log_handle.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--split", default="D0", choices=["D0", "D1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-fit", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = f"{args.arm}_{args.split}_S{args.seed}"
    if args.out is None:
        args.out = str(C.RESULTS_DIR / args.run_id)
    manifest = train(args)
    print(json.dumps(C.jsonable(manifest), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()