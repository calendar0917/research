"""Seed-0 training for the ACRA-v1 arms.

Only whole-molecule labels supervise the composed prediction:
``L = L_task + 0.001 * L_site + 0.01 * L_rel``.  The official Test split is
never loaded here; evaluation is Train/Valid only, model selection is Valid
whole-molecule RMSE at the best epoch.  One launch trains one run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K
import acra_model as M


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


class RunData:
    def __init__(self, cfg: Mapping[str, Any], log=print) -> None:
        self.cfg = dict(cfg)
        self.raw = D.load_raw(cfg)
        self.pack = K.ensure_pack(cfg, log=log)
        self.splits = D.official_splits(self.raw)
        self.labels = np.asarray(self.raw["labels"], np.float64)
        self.row_ids = np.asarray(self.raw["row_ids"], np.int64)
        self.scaler = C.target_scaler(self.labels, self.splits["train"])
        self.y_std = (self.labels - float(self.scaler["mean"])) / float(self.scaler["std"])
        self.atom_dims, self.bond_dims = C.official_feature_dims()

    def raw_metrics(self, pred_std: np.ndarray, rows: np.ndarray) -> dict[str, float]:
        pred = pred_std * float(self.scaler["std"]) + float(self.scaler["mean"])
        truth = self.labels[rows]
        return {"rmse": C.rmse(pred, truth), "mae": C.mae(pred, truth)}


def build_model(cfg: Mapping[str, Any], data: RunData, arm: str):
    model = M.ACRAModel(cfg, data.atom_dims, data.bond_dims, arm)
    model.load_pack(data.pack)
    atom_support, bond_support = M.category_support(
        data.raw, data.splits["train"], data.atom_dims, data.bond_dims)
    zeroed = M.apply_support(model, atom_support, bond_support)
    hashes = C.module_hashes(model, ["frame", "proj_s", "proj_c", "mlp_s", "mlp_c",
                                     "bond_emb", "flag_bridge", "flag_ring",
                                     "scalar_s", "scalar_c", "atom_head", "beta",
                                     "q", "k", "site_head", "rel_q", "rel_k",
                                     "rel_bias", "gate_raw"])
    return model, zeroed, hashes


def evaluate(model: M.ACRAModel, source: D.BatchSource, data: RunData,
             split: str, batch_size: int = 128) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    res = M.predict_rows(model, source, data.splits[split], model.device, batch_size)
    metrics = data.raw_metrics(res["yhat_std"], data.splits[split])
    metrics["term_abs_atom"] = float(np.mean(np.abs(res["e_atom"])))
    metrics["term_abs_bond"] = float(np.mean(np.abs(res["e_bond"])))
    metrics["term_abs_rel"] = float(np.mean(np.abs(res["e_rel"])))
    metrics["yhat_std_mean"] = float(np.mean(res["yhat_std"]))
    metrics["yhat_std_std"] = float(np.std(res["yhat_std"]))
    return res, metrics


def save_predictions(path: Path, data: RunData, rows: np.ndarray,
                     res: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, np.ndarray] = {
        "row_id": data.row_ids[rows],
        "label": data.labels[rows],
        "pred_std": np.asarray(res["yhat_std"], np.float64),
        "pred": np.asarray(res["yhat_std"], np.float64) * float(data.scaler["std"])
        + float(data.scaler["mean"]),
        "e_atom_std": np.asarray(res["e_atom"], np.float64),
        "e_bond_std": np.asarray(res["e_bond"], np.float64),
        "e_rel_std": np.asarray(res["e_rel"], np.float64),
        "beta_std": np.asarray(res["beta"], np.float64),
    }
    np.savez(path, **payload)
    return C.sha256_file(path)


def train(args: argparse.Namespace) -> int:
    import torch

    C.set_deterministic(int(args.seed))
    cfg = dict(C.load_config())
    if args.epochs is None:
        horizon = C.analysis_dir() / "horizon.json"
        if horizon.is_file():
            cfg["recipe"]["epochs"] = int(C.read_json(horizon)["epochs"])
            cfg["recipe"]["horizon_source"] = "analysis/horizon.json"
        else:
            cfg["recipe"]["epochs"] = int(cfg["recipe"]["epochs_max"])
            cfg["recipe"]["horizon_source"] = "config epochs_max"
    else:
        cfg["recipe"]["epochs"] = int(args.epochs)
        cfg["recipe"]["horizon_source"] = "cli --epochs"

    run_id = args.run_id
    rdir = C.run_dir(run_id)
    log_path = rdir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"run={run_id} arm={args.arm} device={args.device} seed={args.seed} "
        f"epochs={cfg['recipe']['epochs']} ({cfg['recipe']['horizon_source']})")
    data = RunData(cfg, log=log)
    log(f"  scaler mean={data.scaler['mean']:.6f} std={data.scaler['std']:.6f} "
        f"n_train={data.splits['train'].size} n_valid={data.splits['valid'].size}")
    log(f"  pack sha256={data.pack['cache_sha256']} entries={data.pack['ent_pat'].size} "
        f"patterns={data.pack['pat_k'].size} events={data.pack['ev_key_s'].size}")
    train_rows = data.splits["train"]
    if args.train_subset and int(args.train_subset) < train_rows.size:
        train_rows = train_rows[:int(args.train_subset)]
        log(f"  SMOKE train subset: {train_rows.size} molecules (not a formal run)")

    model, zeroed, hashes = build_model(cfg, data, args.arm)
    model = model.to(args.device)
    source = D.BatchSource(data.pack, model.device)
    ledger = model.parameter_ledger()
    log(f"  params={ledger}")
    log(f"  init_hashes={json.dumps(hashes, sort_keys=True)}")
    log(f"  zeroed_category_rows={zeroed}")
    log(f"  interaction={model.use_interaction} relation={model.use_relation}")

    recipe = cfg["recipe"]
    batch_molecules = int(recipe["batch_molecules"])
    n_train = int(train_rows.size)
    steps_per_epoch = int(math.ceil(n_train / batch_molecules))
    total_steps = int(recipe["epochs"]) * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    rng = np.random.default_rng(int(args.seed) + 101)
    curves: list[dict[str, Any]] = []
    best: dict[str, Any] = {"valid_rmse": float("inf"), "epoch": -1}
    step = 0
    t_start = time.time()
    gate_value = float("nan")
    for epoch in range(1, int(recipe["epochs"]) + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_train)
        losses: list[dict[str, float]] = []
        for start in range(0, n_train, batch_molecules):
            take = order[start:start + batch_molecules]
            rows = train_rows[take]
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            batch = source.make(rows)
            y_std = torch.as_tensor(data.y_std[rows], device=model.device, dtype=torch.float32)
            out = model.forward_batch(batch)
            terms = model.loss(out, batch, y_std)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(recipe["clip_norm"]))
            opt.step()
            step += 1
            losses.append({k: float(v.detach()) for k, v in terms.items()} | {
                "grad_norm": float(grad_norm),
                "n_events": float(batch["n_events"]),
            })
            if step == 1:
                log(f"  step1 loss={losses[-1]['loss']:.4f} task={losses[-1]['task']:.4f} "
                    f"site={losses[-1]['site']:.6f} rel={losses[-1]['rel']:.6f} "
                    f"grad={losses[-1]['grad_norm']:.3f} batch_atoms={batch['n_atoms']} "
                    f"batch_events={batch['n_events']} batch_pairs={batch['n_pairs']} "
                    f"batch_entries={batch['n_entries']}")

        if args.train_subset:
            # metric over the subset actually trained on, for smoke logging only
            train_res = M.predict_rows(model, source, train_rows, model.device,
                                       int(recipe["eval_batch"]))
            train_metrics = data.raw_metrics(train_res["yhat_std"], train_rows)
            train_metrics["term_abs_atom"] = float(np.mean(np.abs(train_res["e_atom"])))
            train_metrics["term_abs_bond"] = float(np.mean(np.abs(train_res["e_bond"])))
            train_metrics["term_abs_rel"] = float(np.mean(np.abs(train_res["e_rel"])))
        else:
            train_res, train_metrics = evaluate(model, source, data, "train",
                                                batch_size=int(recipe["eval_batch"]))
        valid_res, valid_metrics = evaluate(model, source, data, "valid",
                                            batch_size=int(recipe["eval_batch"]))
        if model.use_relation:
            gate_value = float(torch.sigmoid(model.gate_raw).detach().cpu())
        row: dict[str, Any] = {
            "run_id": run_id, "arm": args.arm, "epoch": epoch,
            "step": step,
            "lr": lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]),
                        float(recipe["min_lr"])),
            "epoch_seconds": time.time() - t_epoch,
            "gate": gate_value,
        }
        for key in ("loss", "task", "site", "rel", "grad_norm"):
            row[f"train_{key}"] = float(np.mean([x[key] for x in losses]))
        for split, metrics in (("train", train_metrics), ("valid", valid_metrics)):
            for key, value in metrics.items():
                row[f"{split}_{key}"] = float(value)
        row["generalization_gap_rmse"] = row["valid_rmse"] - row["train_rmse"]
        curves.append(row)
        C.write_csv(rdir / "curves.csv", curves)

        better = row["valid_rmse"] < float(best["valid_rmse"]) - 1e-12
        if better:
            best = {"valid_rmse": row["valid_rmse"], "valid_mae": row["valid_mae"],
                    "epoch": epoch}
            torch.save({
                "state": model.state_dict(),
                "epoch": epoch,
                "row": row,
                "scaler": data.scaler,
                "ledger": ledger,
                "init_hashes": hashes,
                "arm": args.arm,
                "config": cfg,
                "pack_sha256": data.pack["cache_sha256"],
            }, rdir / "state_best.pt")
            save_predictions(rdir / "predictions_best_train.npz", data,
                             train_rows if args.train_subset else data.splits["train"],
                             M.predict_rows(model, source, train_rows, model.device,
                                            int(recipe["eval_batch"])))
            save_predictions(rdir / "predictions_best.npz", data, data.splits["valid"],
                             valid_res)
        log(f"epoch {epoch:3d}/{recipe['epochs']} lr={row['lr']:.2e} "
            f"train={row['train_rmse']:.4f} valid={row['valid_rmse']:.4f} "
            f"loss={row['train_loss']:.4f} site={row['train_site']:.5f} "
            f"rel={row['train_rel']:.6f} gate={gate_value:.3f} "
            f"terms[atom={row['train_term_abs_atom']:.3f},bond={row['train_term_abs_bond']:.3f},"
            f"rel={row['train_term_abs_rel']:.4f}] {row['epoch_seconds']:.1f}s")

    torch.save({
        "state": model.state_dict(), "epoch": int(curves[-1]["epoch"]),
        "row": curves[-1], "scaler": data.scaler, "ledger": ledger,
        "init_hashes": hashes, "arm": args.arm, "config": cfg,
        "pack_sha256": data.pack["cache_sha256"],
    }, rdir / "state_final.pt")
    final_train = M.predict_rows(model, source, train_rows, model.device,
                                 int(recipe["eval_batch"]))
    save_predictions(rdir / "predictions_final_train.npz", data, train_rows, final_train)
    final_valid = M.predict_rows(model, source, data.splits["valid"], model.device,
                                 int(recipe["eval_batch"]))
    save_predictions(rdir / "predictions_final.npz", data, data.splits["valid"], final_valid)

    wall = time.time() - t_start
    manifest = {
        "run_id": run_id,
        "arm": args.arm,
        "dataset": cfg["dataset"],
        "protocol_id": cfg["protocol_id"],
        "seed": int(args.seed),
        "epochs": int(curves[-1]["epoch"]),
        "epochs_planned": int(recipe["epochs"]),
        "horizon_source": recipe["horizon_source"],
        "steps_per_epoch": steps_per_epoch,
        "batch_molecules": batch_molecules,
        "train_subset": int(args.train_subset) if args.train_subset else None,
        "n_train": int(n_train),
        "n_valid": int(data.splits["valid"].size),
        "test_labels_read": False,
        "selection": "best valid whole-molecule RMSE",
        "best": best,
        "final": curves[-1],
        "scaler": data.scaler,
        "params": ledger,
        "init_hashes": hashes,
        "zeroed_categories": zeroed,
        "cache_sha256": data.pack["cache_sha256"],
        "cache_path": str(data.pack["cache_path"]),
        "raw_cache_sha256": str(data.raw.get("cache_sha256", "")),
        "cache_meta": json.loads(str(data.pack["meta_json"])),
        "composition": cfg["composition"],
        "recipe": {k: v for k, v in recipe.items() if k != "epochs"},
        "wall_seconds": wall,
        "gpu_hours": wall / 3600.0,
        "max_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "environment": C.environment(),
        "provenance": C.code_provenance(),
        "parent_provenance": C.parent_provenance(),
        "argv": list(getattr(args, "argv", [])),
    }
    C.write_json(rdir / "manifest.json", manifest)
    C.write_json(rdir / "config.json", cfg)
    log(f"done wall={wall / 60:.2f}min best_valid_rmse={best['valid_rmse']:.6f}@{best['epoch']} "
        f"gpu_hours={manifest['gpu_hours']:.4f}")
    log_handle.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True, choices=list(M.ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-subset", type=int, default=0)
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else []
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())