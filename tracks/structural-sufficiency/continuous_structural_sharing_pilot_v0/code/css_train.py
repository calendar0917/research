"""Formal single-unit training for CSS-v0.

Four units exist and nothing else: DIRECT_100, SHARE_100, DIRECT_50,
SHARE_50.  Each unit trains the same frozen R2/SENTENCE_BASE trunk with the
same 32-dimensional branch; only the precomputed branch input differs
(``z_v`` for DIRECT, ``q_v`` for SHARE).

Every epoch evaluates the full official Train / Valid / Test split in original
label units.  The 50% arms additionally separate the 1680 Fit molecules from the
1680 held-out Train molecules.  The best checkpoint is chosen by official Valid
RMSE only; Test is ``DEVELOPMENT_TEST`` and never selects or modifies anything.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import css_common as C
import css_fit as F
import css_model as M
import ses_structure as SS
import ses_tensor as ST
import ses_data as SD

TRAIN_FILES = ("css_common.py", "css_data.py", "css_descriptors.py", "css_fit.py",
               "css_model.py", "css_train.py")


def load_prep(prep_dir: Path, size: int) -> dict[str, Any]:
    prep_dir = Path(prep_dir)
    manifest = C.read_json(prep_dir / "prep_manifest.json")
    if manifest.get("format") != "css-v0-prep-v1":
        raise ValueError("prep artifact format mismatch")
    if manifest["topology"]["sources"][str(C.RAW_CACHE.relative_to(C.REPO_ROOT))] != C.RAW_CACHE_SHA256:
        raise ValueError("prep raw-cache mismatch")
    with (prep_dir / "structs.pkl").open("rb") as handle:
        struct_payload = pickle.load(handle)
    if struct_payload.get("format") != "css-v0-full-structs-v1":
        raise ValueError("struct pickle format mismatch")
    with (prep_dir / f"fitstats_{size}.pkl").open("rb") as handle:
        fitstats = pickle.load(handle)
    if fitstats.get("format") != "css-v0-fitstats-v1" or int(fitstats["size"]) != int(size):
        raise ValueError("fitstats artifact mismatch")
    fits = np.load(prep_dir / f"fit_{size}.npz")
    plan = np.load(prep_dir / f"plan_{size}.npy")
    if plan.shape[1] != int(fitstats["fit_rows"].size):
        raise ValueError("batch plan width mismatch")
    return {"manifest": manifest, "structs": struct_payload["structs"],
            "fitstats": fitstats, "fits": fits, "plan": plan,
            "prep_dir": prep_dir, "size": int(size)}


def row_atom_ptr(structs: Mapping[int, SS.Struct], n_mol: int) -> np.ndarray:
    counts = np.asarray([structs[r].n_atoms for r in range(n_mol)], np.int64)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _norm_from_fitstats(fitstats: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "atom_scalar_mean": np.asarray(fitstats["atom_scalar_mean"], np.float64),
        "atom_scalar_scale": np.asarray(fitstats["atom_scalar_scale"], np.float64),
        "graph_mean": np.asarray(fitstats["graph_mean"], np.float64),
        "graph_scale": np.asarray(fitstats["graph_scale"], np.float64),
    }


@torch.no_grad()
def predict_raw(model: Any, rows: np.ndarray, structs: Mapping[int, SS.Struct],
                table: np.ndarray, ptr: np.ndarray, device: Any,
                batch_molecules: int, scaler: Mapping[str, float]) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows, np.int64)
    out = np.zeros(rows.size, np.float64)
    for start in range(0, rows.size, int(batch_molecules)):
        part = rows[start:start + int(batch_molecules)]
        batch = ST.collate([structs[int(r)] for r in part.tolist()], device)
        batch["branch"] = M.branch_gather(part, structs, table, ptr, device)
        pred = model(batch, arm="SENTENCE_BASE")["yhat_std"]
        out[start:start + part.size] = pred.detach().double().cpu().numpy()
    return out * float(scaler["std"]) + float(scaler["mean"])


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    C.seed_everything(int(args.seed))
    cfg = C.load_config()
    recipe = dict(cfg["recipe"])
    prep = load_prep(Path(args.prep_dir), int(args.size))
    fitstats = prep["fitstats"]
    structs = prep["structs"]
    fits = prep["fits"]
    plan = prep["plan"]
    size = int(args.size)
    arm = str(args.arm)
    if arm not in C.ARMS:
        raise KeyError(arm)
    if size not in C.SIZES:
        raise KeyError(size)
    out_dir = Path(args.out) if args.out else (C.RESULTS_DIR / str(args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    bundle = SD.load_bundle()
    labels = np.asarray(bundle.labels, np.float64)
    atom_dims, bond_dims = SD.atom_bond_dims()
    fit_rows = np.sort(np.asarray(fitstats["fit_rows"], np.int64))
    heldout_rows = np.sort(np.asarray(fitstats["heldout_rows"], np.int64))
    valid_rows = np.sort(np.asarray(fitstats["valid_rows"], np.int64))
    test_rows = np.sort(np.asarray(fitstats["test_rows"], np.int64))
    full_train = np.sort(np.union1d(fit_rows, heldout_rows))
    scaler = {"mean": float(fitstats["target_scaler"]["mean"]),
              "std": float(fitstats["target_scaler"]["std"])}
    smoke = bool(getattr(args, "smoke", False))
    expected_fit = 3360 if size == 100 else 1680
    if int(fit_rows.size) != expected_fit and not smoke:
        raise ValueError(f"unexpected fit size {fit_rows.size}")
    n_mol = max(int(k) for k in structs) + 1
    ptr = row_atom_ptr(structs, n_mol)
    table = np.asarray(fits[f"branch_{arm}"], np.float32)
    if table.shape != (int(ptr[-1]), 64):
        raise ValueError(f"branch table shape {table.shape}")
    if not np.isfinite(table).all():
        raise ValueError("non-finite branch input")
    table_sha = C.array_sha256(table)
    expected = prep["manifest"]["sizes"][str(size)]["branch_table_sha256"][arm]
    if table_sha != expected:
        raise ValueError("branch table hash mismatch")

    model, info = M.build_css_model(atom_dims, bond_dims, int(args.seed),
                                    _norm_from_fitstats(fitstats), cfg)
    zeroed = model.zero_unseen(fitstats["atom_ok"], fitstats["bond_ok"], atom_dims, bond_dims)
    ledger = info["parameters"]
    if int(ledger["total"]) != int(C.FROZEN_PREDICTOR_PARAMS) + info["branch_parameters"]:
        raise ValueError("parameter ledger mismatch")
    model = model.to(args.device)

    epochs = int(args.epochs)
    if epochs > int(plan.shape[0]):
        raise ValueError(f"horizon {epochs} exceeds prepared plan {plan.shape[0]}")
    batch_mol = int(recipe["batch_molecules"])
    n_fit = int(fit_rows.size)
    steps_per_epoch = int(math.ceil(n_fit / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    eval_batch = int(recipe["eval_batch_molecules"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    y_std = (labels - scaler["mean"]) / scaler["std"]

    log(f"run={args.run_id} arm={arm} size={size} seed={args.seed} "
        f"device={args.device} host={socket.gethostname()}")
    log(f"fit={n_fit} heldout={heldout_rows.size} valid={valid_rows.size} "
        f"test={test_rows.size} scaler={scaler} epochs={epochs}")
    log(f"params={ledger} branch={table.shape} sha={table_sha[:16]}")
    log(f"plan_sha={C.array_sha256(plan)} init={json.dumps(info['init_hashes'], sort_keys=True)}")
    log(f"zeroed_unseen={zeroed}")

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse": float("inf"), "epoch": -1, "valid_mae": float("inf")}
    step = 0
    peak_vram = 0
    branch_grad_step1 = float("nan")
    trunk_grad_step1 = float("nan")
    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = plan[epoch - 1]
        losses: list[float] = []
        grad_norms: list[float] = []
        for start in range(0, n_fit, batch_mol):
            rows = fit_rows[order[start:start + batch_mol].astype(np.int64)]
            batch = ST.collate([structs[int(r)] for r in rows.tolist()], args.device)
            batch["branch"] = M.branch_gather(rows, structs, table, ptr, args.device)
            lr = C.lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                         float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out = model(batch, arm="SENTENCE_BASE")
            target = torch.as_tensor(y_std[rows], device=args.device, dtype=torch.float32)
            loss = torch.mean((out["yhat_std"] - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                branch_grad_step1 = float(model.branch_proj.weight.grad.detach().norm())
                trunk_grad_step1 = float(model.atom_emb[0].weight.grad.detach().norm())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(recipe["gradient_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
            step += 1
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
        t_eval = time.time()
        pred_fit = predict_raw(model, fit_rows, structs, table, ptr,
                               args.device, eval_batch, scaler)
        pred_heldout = (predict_raw(model, heldout_rows, structs, table, ptr,
                                    args.device, eval_batch, scaler)
                        if heldout_rows.size else np.zeros(0))
        pred_full = (predict_raw(model, full_train, structs, table, ptr,
                                 args.device, eval_batch, scaler)
                     if heldout_rows.size else pred_fit)
        pred_valid = predict_raw(model, valid_rows, structs, table, ptr,
                                 args.device, eval_batch, scaler)
        pred_test = predict_raw(model, test_rows, structs, table, ptr,
                                args.device, eval_batch, scaler)
        eval_seconds = time.time() - t_eval
        row: dict[str, Any] = {
            "run_id": str(args.run_id), "arm": arm, "size": size, "seed": int(args.seed),
            "epoch": int(epoch), "step": int(step),
            "lr": C.lr_at(step - 1, warmup_steps, total_steps,
                          float(recipe["peak_lr"]), float(recipe["min_lr"])),
            "epoch_seconds": float(time.time() - t_epoch),
            "eval_seconds": float(eval_seconds),
            "train_loss": float(np.mean(losses)),
            "train_grad_norm": float(np.mean(grad_norms)),
            "fit_rmse": C.rmse(pred_fit, labels[fit_rows]),
            "fit_mae": C.mae(pred_fit, labels[fit_rows]),
            "train_full_rmse": C.rmse(pred_full, labels[full_train]),
            "train_full_mae": C.mae(pred_full, labels[full_train]),
            "heldout_rmse": (C.rmse(pred_heldout, labels[heldout_rows])
                             if heldout_rows.size else float("nan")),
            "heldout_mae": (C.mae(pred_heldout, labels[heldout_rows])
                            if heldout_rows.size else float("nan")),
            "valid_rmse": C.rmse(pred_valid, labels[valid_rows]),
            "valid_mae": C.mae(pred_valid, labels[valid_rows]),
            "test_rmse": C.rmse(pred_test, labels[test_rows]),
            "test_mae": C.mae(pred_test, labels[test_rows]),
        }
        row["generalization_gap_rmse"] = row["valid_rmse"] - row["fit_rmse"]
        row["heldout_gap_rmse"] = row["heldout_rmse"] - row["fit_rmse"] if heldout_rows.size else float("nan")
        curves.append(row)
        C.write_csv(out_dir / "curves.csv", curves)
        better = row["valid_rmse"] < float(best["valid_rmse"]) - 1e-12
        if better:
            best = {"valid_rmse": row["valid_rmse"], "valid_mae": row["valid_mae"],
                    "epoch": int(epoch), "test_rmse": row["test_rmse"],
                    "fit_rmse": row["fit_rmse"]}
            torch.save({
                "state": model.state_dict(), "epoch": int(epoch), "row": row,
                "scaler": scaler, "ledger": ledger, "arm": arm, "size": size,
                "init_hashes": info["init_hashes"], "zeroed": zeroed,
                "data_hashes": bundle.hashes, "prep_hashes": {
                    "branch_table": table_sha,
                    "batch_plan": C.array_sha256(plan),
                    "config": C.sha256_json(cfg)},
                "fit_rows_sha256": C.array_sha256(fit_rows),
                "heldout_rows_sha256": C.array_sha256(heldout_rows),
                "valid_rows_sha256": C.array_sha256(valid_rows),
                "test_rows_sha256": C.array_sha256(test_rows),
                "config": cfg,
            }, out_dir / "state_best.pt")
        log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} fit={row['fit_rmse']:.4f} "
            f"held={row['heldout_rmse']:.4f} valid={row['valid_rmse']:.4f} "
            f"test={row['test_rmse']:.4f} loss={row['train_loss']:.4f} "
            f"grad={row['train_grad_norm']:.2f} {row['epoch_seconds']:.1f}s")
    train_seconds = time.time() - t_train_start

    best_state = torch.load(out_dir / "state_best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_state["state"])
    predictions: dict[str, np.ndarray] = {}
    for tag, rows in (("fit", fit_rows), ("heldout", heldout_rows),
                      ("train_full", full_train), ("valid", valid_rows),
                      ("test", test_rows)):
        if rows.size == 0:
            continue
        pred = predict_raw(model, rows, structs, table, ptr, args.device,
                           eval_batch, scaler)
        predictions[tag] = pred
        np.savez(out_dir / f"predictions_best_{tag}.npz",
                 row_id=np.asarray(rows, np.int64), pred=pred,
                 label=labels[rows])
        records = [{"row_id": int(r), "smiles": str(np.asarray(bundle.raw["smiles"])[int(r)]),
                    "label_raw": float(labels[int(r)]), "pred_raw": float(pred[i]),
                    "abs_error": float(abs(pred[i] - labels[int(r)])),
                    "sq_error": float((pred[i] - labels[int(r)]) ** 2),
                    "n_atoms": int(structs[int(r)].n_atoms)}
                   for i, r in enumerate(np.asarray(rows).tolist())]
        C.write_csv(out_dir / f"predictions_best_{tag}.csv", records)

    wall_seconds = time.time() - t_process
    metrics = {
        "best_epoch": int(best["epoch"]),
        "fit_rmse": C.rmse(predictions["fit"], labels[fit_rows]),
        "fit_mae": C.mae(predictions["fit"], labels[fit_rows]),
        "train_full_rmse": C.rmse(predictions["train_full"], labels[full_train]),
        "train_full_mae": C.mae(predictions["train_full"], labels[full_train]),
        "heldout_rmse": (C.rmse(predictions["heldout"], labels[heldout_rows])
                         if heldout_rows.size else None),
        "heldout_mae": (C.mae(predictions["heldout"], labels[heldout_rows])
                        if heldout_rows.size else None),
        "valid_rmse": C.rmse(predictions["valid"], labels[valid_rows]),
        "valid_mae": C.mae(predictions["valid"], labels[valid_rows]),
        "test_rmse": C.rmse(predictions["test"], labels[test_rows]),
        "test_mae": C.mae(predictions["test"], labels[test_rows]),
    }
    manifest: dict[str, Any] = {
        "format": "css-v0-run-v1",
        "smoke": smoke,
        "run_id": str(args.run_id), "arm": arm, "size": size, "seed": int(args.seed),
        "started_utc": C.now_utc(), "host": socket.gethostname(),
        "git_commit": C.git_commit(), "device": str(args.device),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                     and str(args.device).startswith("cuda") else "cpu"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha(TRAIN_FILES),
        "prep": {
            "n_roots": int(prep["manifest"]["n_roots"]),
            "prep_code_sha256": prep["manifest"]["code_sha256"],
            "branch_table_sha256": table_sha,
            "batch_plan_sha256": C.array_sha256(plan),
            "batch_plan_epoch0_sha256": C.array_sha256(plan[0]),
            "descriptor_sha256": prep["manifest"]["descriptor_sha256"],
        },
        "data_hashes": bundle.hashes,
        "official_rows": prep["manifest"]["official_rows"],
        "target_scaler": scaler,
        "n_fit": int(fit_rows.size), "n_heldout": int(heldout_rows.size),
        "n_valid": int(valid_rows.size), "n_test": int(test_rows.size),
        "fit_rows_sha256": C.array_sha256(fit_rows),
        "heldout_rows_sha256": C.array_sha256(heldout_rows),
        "valid_rows_sha256": C.array_sha256(valid_rows),
        "test_rows_sha256": C.array_sha256(test_rows),
        "params": ledger,
        "branch_parameters": int(info["branch_parameters"]),
        "backbone_in_dim": int(info["backbone_in_dim"]),
        "init_hashes": info["init_hashes"],
        "zeroed_categories": zeroed,
        "recipe": recipe,
        "epochs": int(epochs), "epochs_run": int(curves[-1]["epoch"]),
        "steps_per_epoch": int(steps_per_epoch), "total_steps": int(total_steps),
        "gradient_probe_step1": {"branch_proj": branch_grad_step1,
                                 "atom_emb": trunk_grad_step1},
        "metrics": metrics,
        "best": best,
        "train_seconds": float(train_seconds),
        "wall_seconds": float(wall_seconds),
        "gpu_hours": float(wall_seconds / 3600.0),
        "peak_vram_bytes": int(peak_vram),
        "curves": str(out_dir / "curves.csv"),
        "checkpoint": str(out_dir / "state_best.pt"),
        "predictions": {tag: str(out_dir / f"predictions_best_{tag}.npz")
                        for tag in predictions},
        "test_policy": "DEVELOPMENT_TEST: never used for selection or config changes",
    }
    C.write_json(out_dir / "manifest.json", manifest)
    log(f"done valid={metrics['valid_rmse']:.4f} test={metrics['test_rmse']:.4f} "
        f"fit={metrics['fit_rmse']:.4f} best_epoch={best['epoch']} "
        f"train={train_seconds:.1f}s wall={wall_seconds:.1f}s")
    log_handle.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(C.ARMS))
    parser.add_argument("--size", required=True, type=int, choices=list(C.SIZES))
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prep-dir", default=str(C.PREP_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = f"{args.arm}_{args.size}_S{args.seed}_E{args.epochs}"
    if args.out is None:
        args.out = str(C.RESULTS_DIR / args.run_id)
    manifest = train(args)
    print(json.dumps(C.jsonable(manifest["metrics"]), sort_keys=True))


if __name__ == "__main__":
    main()
