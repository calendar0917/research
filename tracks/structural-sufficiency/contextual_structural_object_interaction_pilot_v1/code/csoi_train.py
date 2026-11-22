"""Seed-0 training for one CSOI arm (Train/Valid only).

Differences from the parent round's trainer, all of them required by this
brief: **no Test evaluation at all** (no Test prediction, no Test metric, no
Test-based selection), and the model is the CSOI family.  Everything else --
optimiser, schedule, batch composition, auxiliary CUTRESP supervision, target
scaler, pair sampler, checkpoint rule -- is the parent's frozen recipe.

Epochs are frozen to the common horizon in ``analysis/timing.json``; the timing
smoke (warmup epochs only) is what writes that file, and it never looks at a
label metric.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D
import csoi_model as M


# ---------------------------------------------------------------------------
# data view
# ---------------------------------------------------------------------------


class RunData:
    def __init__(self, cfg: Mapping[str, Any], dataset: str, log=print) -> None:
        self.cfg = dict(cfg)
        self.dataset = dataset
        self.pack = D.load_pack(cfg, dataset)
        self.labels = np.asarray(self.pack["labels"], np.float64)
        self.row_ids = np.asarray(self.pack["row_ids"], np.int64)
        self.splits = {
            "train": np.asarray(self.pack["train_idx"], np.int64),
            "valid": np.asarray(self.pack["valid_idx"], np.int64),
        }
        self.scaler = C.target_scaler(self.labels, self.splits["train"])
        self.y_std = (self.labels - self.scaler["mean"]) / self.scaler["std"]
        self.atom_dims, self.bond_dims = C.official_feature_dims()
        log(f"  scaler mean={self.scaler['mean']:.6f} std={self.scaler['std']:.6f} "
            f"n_train={self.splits['train'].size} n_valid={self.splits['valid'].size}")

    def raw_cache(self) -> dict[str, Any]:
        return C.PD.load_raw_cache(self.cfg, self.dataset)


# ---------------------------------------------------------------------------
# evaluation (Train + Valid only)
# ---------------------------------------------------------------------------


def predict_all(model: M.CSOIModel, data: RunData, diag: dict[str, Any] | None = None):
    import torch

    model.eval()
    out: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        frames = model.frame_embeddings()
        for split, rows in data.splits.items():
            pred = np.zeros(len(rows), np.float64)
            cut_sq = 0.0
            cut_n = 0
            var_parts: list[np.ndarray] = []
            for start in range(0, len(rows), 32):
                part = rows[start:start + 32]
                idx = torch.as_tensor(part.astype(np.int64), device=model.device)
                res = model.molecule_cuts(idx, frames, diag=diag if split == "valid" else None)
                pred[start:start + len(part)] = res["f_bar"].detach().double().cpu().numpy()
                f = res["f_cut"]
                owner = res["owner"]
                y = torch.as_tensor(data.y_std[part], device=model.device, dtype=torch.float32)
                per = torch.zeros(len(part), device=model.device).index_add_(
                    0, owner, f) / res["counts"].float().clamp(min=1.0)
                var = torch.zeros(len(part), device=model.device).index_add_(
                    0, owner, (f - per[owner]) ** 2) / res["counts"].float().clamp(min=1.0)
                var_parts.append(var.detach().cpu().numpy())
                cut_sq += float(((f - y[owner]) ** 2).sum())
                cut_n += int(f.numel())
            out[split] = pred
            s: dict[str, float] = {}
            if cut_n:
                sigma = float(data.scaler["std"])
                s["cut_var_std"] = float(np.mean(np.concatenate(var_parts)))
                s["cut_rmse_std"] = float(math.sqrt(cut_sq / cut_n))
                s["cut_rmse"] = float(math.sqrt(cut_sq / cut_n) * sigma)
                s["cut_n"] = float(cut_n)
            stats[split] = s
    return out, stats


def raw_metrics(pred_std: np.ndarray, labels: np.ndarray,
                scaler: Mapping[str, float]) -> dict[str, float]:
    pred = np.asarray(pred_std, np.float64) * float(scaler["std"]) + float(scaler["mean"])
    return {"rmse": C.rmse(pred, labels), "mae": C.mae(pred, labels)}


def save_predictions(path: Path, data: RunData, preds: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, np.ndarray] = {}
    for split, rows in data.splits.items():
        p = np.asarray(preds[split], np.float64)
        payload[f"{split}_row_id"] = data.row_ids[rows]
        payload[f"{split}_label"] = data.labels[rows]
        payload[f"{split}_pred_std"] = p
        payload[f"{split}_pred"] = p * float(data.scaler["std"]) + float(data.scaler["mean"])
    np.savez(path, **payload)
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# model build
# ---------------------------------------------------------------------------


def build_model(cfg: Mapping[str, Any], data: RunData, arm: str):
    model = M.CSOIModel(cfg, data.atom_dims, data.bond_dims, arm)
    model.load_pack(data.pack)
    support = C.category_support(data.raw_cache(), data.splits["train"],
                                 data.atom_dims, data.bond_dims)
    zeroed = C.apply_atom_bond_support(model, support[0], support[1])
    return model, zeroed


SHARED_MODULES = ("phi", "token_proj", "bucket_emb", "mult_proj", "token_ln",
                  "cut_proj", "null_cut", "null_ctx", "add_s", "add_c", "bond_bias", "beta")


def grad_diagnostics(model: M.CSOIModel, data: RunData, store, cfg: Mapping[str, Any],
                     sampler) -> dict[str, Any]:
    """One real batch: per-group gradient norms and one-step loss values."""
    import torch

    recipe = cfg["recipe"]
    model.train()
    rows = data.splits["train"][: int(recipe["batch_molecules"])]
    pair_idx, aux_mols = sampler.sample(int(recipe["batch_pairs"]))
    all_rows = np.concatenate([rows, aux_mols[0], aux_mols[1]])
    idx = torch.as_tensor(all_rows.astype(np.int64), device=model.device)
    res = model.molecule_forward(idx)
    n_main = len(rows)
    y = torch.as_tensor(data.y_std[all_rows], device=model.device, dtype=torch.float32)
    owner = res["owner"]
    sq = (res["f_cut"] - y[owner]) ** 2
    per_mol = torch.zeros(len(all_rows), device=model.device).index_add_(0, owner, sq)
    per_mol = per_mol / res["counts"].float().clamp(min=1.0)
    l_main = per_mol[:n_main].mean()
    pairs_t = store.tensor_pairs(pair_idx, model.device)
    delta = model.pair_cut_delta(pairs_t)
    target = torch.as_tensor(store.delta_std_of(pair_idx), device=model.device,
                             dtype=torch.float32)
    l_aux = torch.mean((delta - target) ** 2) / float(store.tau2)
    loss = l_main + float(recipe["aux_weight"]) * l_aux
    model.zero_grad(set_to_none=True)
    loss.backward()
    groups = model.module_groups()
    named = dict(model.named_parameters())
    out: dict[str, Any] = {
        "loss_main": float(l_main.detach()), "loss_aux": float(l_aux.detach()),
        "loss_total": float(loss.detach()),
        "finite": bool(np.isfinite(float(loss.detach()))),
        "groups": {},
        "cut_var_std": float(torch.zeros(1).item()),
    }
    for group, prefixes in groups.items():
        total = 0.0
        nonzero = 0
        n = 0
        for name, p in named.items():
            if not any(name == pref or name.startswith(pref + ".") for pref in prefixes):
                continue
            n += 1
            if p.grad is None:
                continue
            g = float(p.grad.detach().norm())
            total += g * g
            nonzero += int(g > 0)
        out["groups"][group] = {"n_params": n, "grad_norm": math.sqrt(total),
                                "params_with_nonzero_grad": nonzero}
    model.zero_grad(set_to_none=True)
    return out


def group_grad_norms(model) -> dict[str, float]:
    """Per-group gradient norm of the current ``.grad`` fields (cheap snapshot)."""
    named = dict(model.named_parameters())
    out: dict[str, float] = {}
    for group, prefixes in model.module_groups().items():
        total = 0.0
        for name, p in named.items():
            if not any(name == pref or name.startswith(pref + ".") for pref in prefixes):
                continue
            if p.grad is None:
                continue
            total += float(p.grad.detach().norm()) ** 2
        out[group] = math.sqrt(total)
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> int:
    import torch

    C.set_deterministic(int(args.seed))
    cfg = dict(C.load_config())
    if args.overrides:
        _patch(cfg, json.loads(args.overrides))
    run_id = args.run_id
    rdir = C.run_dir(run_id)
    log_handle = (rdir / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    device = args.device
    arm = args.arm
    log(f"run={run_id} arm={arm} device={device} epochs={args.epochs} seed={args.seed}")
    data = RunData(cfg, args.dataset, log=log)
    model, zeroed = build_model(cfg, data, arm)
    model = model.to(device)
    ledger = model.parameter_ledger()
    hashes = C.module_hashes(model, list(SHARED_MODULES) + ["q_proj", "k_proj", "v_proj",
                                                           "r_mlp", "out", "rel_mlp"])
    log(f"  params={ledger['total']} ledger={ledger}")
    log(f"  init_hashes={hashes}")
    log(f"  zeroed_category_rows={zeroed}")

    store = C.PPAIR.PairStore(cfg, data.pack, data.labels, None,
                              data.splits["train"], data.row_ids, log=log)
    log(f"  pair store: {store.summary()}")
    sampler = store.sampler()

    recipe = cfg["recipe"]
    n_train = int(data.splits["train"].size)
    steps_per_epoch = int(math.ceil(n_train / int(recipe["batch_molecules"])))
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    epochs_target = int(args.epochs)
    if epochs_target <= 0:
        timing_path = C.analysis_dir() / "timing.json"
        if not timing_path.is_file():
            raise FileNotFoundError("freeze the common horizon first (analysis/timing.json)")
        epochs_target = int(C.read_json(timing_path)["horizon_epochs"])
    total_steps = epochs_target * steps_per_epoch
    log(f"  horizon E={epochs_target} steps_per_epoch={steps_per_epoch}")

    epoch_rng = np.random.default_rng(int(args.seed) + 17)
    orders = [epoch_rng.permutation(n_train) for _ in range(400)]

    curves: list[dict[str, Any]] = []
    best = {"valid": float("inf"), "epoch": -1}
    step = 0
    t_resume = time.time()
    for epoch in range(1, epochs_target + 1):
        t_epoch = time.time()
        model.train()
        order = orders[(epoch - 1) % len(orders)]
        main_losses: list[float] = []
        aux_losses: list[float] = []
        grad_snapshot: dict[str, float] = {}
        for start in range(0, n_train, int(recipe["batch_molecules"])):
            batch_rows = data.splits["train"][order[start:start + int(recipe["batch_molecules"])]]
            lr = C.lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                         float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            pair_idx, aux_mols = sampler.sample(int(recipe["batch_pairs"]))
            all_rows = np.concatenate([batch_rows, aux_mols[0], aux_mols[1]])
            rows_t = torch.as_tensor(all_rows.astype(np.int64), device=device)
            res = model.molecule_forward(rows_t)
            n_main = len(batch_rows)
            y = torch.as_tensor(data.y_std[all_rows], device=device, dtype=torch.float32)
            f_cut = res["f_cut"]
            owner = res["owner"]
            sq = (f_cut - y[owner]) ** 2
            per_mol = torch.zeros(len(all_rows), device=device).index_add_(0, owner, sq)
            per_mol = per_mol / res["counts"].float().clamp(min=1.0)
            l_main = per_mol[:n_main].mean()
            pairs_t = store.tensor_pairs(pair_idx, device)
            delta = model.pair_cut_delta(pairs_t)
            pair_label = torch.as_tensor(store.delta_std_of(pair_idx), device=device,
                                         dtype=torch.float32)
            l_aux = torch.mean((delta - pair_label) ** 2) / float(store.tau2)
            loss = l_main + float(recipe["aux_weight"]) * l_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["clip_norm"]))
            if start == 0:
                grad_snapshot = group_grad_norms(model)
            opt.step()
            step += 1
            main_losses.append(float(l_main.detach()))
            aux_losses.append(float(l_aux.detach()))
            if int(getattr(args, "max_steps", 0)) and step >= int(args.max_steps):
                break
        if int(getattr(args, "max_steps", 0)) and step >= int(args.max_steps):
            log(f"  smoke limit reached: {step} steps")

        preds, cut_stats = predict_all(model, data)
        metrics = {s: raw_metrics(preds[s], data.labels[data.splits[s]], data.scaler)
                   for s in C.SPLITS if s in data.splits}
        row: dict[str, Any] = {
            "run_id": run_id, "epoch": epoch,
            "lr": C.lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]),
                          float(recipe["min_lr"])),
            "train_loss": float(np.mean(main_losses)),
            "aux_loss": float(np.mean(aux_losses)),
            "epoch_seconds": time.time() - t_epoch,
        }
        for gname, gval in grad_snapshot.items():
            row[f"grad_{gname}"] = gval
        for s in ("train", "valid"):
            row[f"{s}_rmse"] = metrics[s]["rmse"]
            row[f"{s}_mae"] = metrics[s]["mae"]
            for key, value in cut_stats.get(s, {}).items():
                row[f"{s}_{key}"] = value
        curves.append(row)
        C.write_csv(rdir / "curves.csv", curves)
        if row["valid_rmse"] < best["valid"] - 1e-12:
            best = {"valid": row["valid_rmse"], "epoch": epoch}
            torch.save({"state": model.state_dict(), "epoch": epoch, "row": row,
                        "scaler": data.scaler, "ledger": ledger, "hashes": hashes,
                        "arm": arm}, rdir / "state_best.pt")
            save_predictions(rdir / "predictions_best.npz", data, preds)
        log(f"epoch {epoch:3d}/{epochs_target} lr={row['lr']:.2e} "
            f"train={row['train_rmse']:.4f} valid={row['valid_rmse']:.4f} "
            f"tr_cut={row.get('train_cut_rmse', float('nan')):.4f} "
            f"cutvar={row.get('train_cut_var_std', float('nan')):.4f} "
            f"aux={row['aux_loss']:.4f} {row['epoch_seconds']:.1f}s")

    torch.save({"state": model.state_dict(), "epoch": int(curves[-1]["epoch"]),
                "row": curves[-1], "scaler": data.scaler, "ledger": ledger,
                "hashes": hashes, "arm": arm}, rdir / "state_final.pt")
    preds_final, _ = predict_all(model, data)
    save_predictions(rdir / "predictions_final.npz", data, preds_final)
    wall = time.time() - t_resume

    diag = None
    manifest = {
        "run_id": run_id, "arm": arm, "dataset": args.dataset,
        "epochs": int(curves[-1]["epoch"]), "epochs_planned": epochs_target,
        "steps_per_epoch": steps_per_epoch,
        "batch_molecules": int(recipe["batch_molecules"]),
        "batch_pairs": int(recipe["batch_pairs"]),
        "supervision": "CUTRESP", "aux_weight": float(recipe["aux_weight"]),
        "seed": int(args.seed), "signature": C.check_signature(),
        "cache_sha256": data.pack["cache_sha256"], "cache_path": data.pack["cache_path"],
        "raw_cache_sha256": data.raw_cache()["cache_sha256"],
        "scaler": data.scaler, "n_train": n_train,
        "n_valid": int(data.splits["valid"].size),
        "params": ledger, "init_hashes": hashes, "zeroed_categories": zeroed,
        "best": best, "final": curves[-1],
        "wall_seconds": wall, "gpu_hours": wall / 3600.0,
        "max_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "environment": C.environment(),
        "pair_store": store.summary(),
        "overrides": json.loads(args.overrides) if args.overrides else None,
        "argv": list(getattr(args, "argv", [])),
        "test_policy": "no test labels read in this round",
        "max_steps": int(getattr(args, "max_steps", 0)),
    }
    C.write_json(rdir / "manifest.json", manifest)
    C.write_json(rdir / "config.json", {"cfg": cfg, "arm": arm})
    log(f"done wall={wall / 60:.2f}min best_valid={best['valid']:.6f}@{best['epoch']}")
    log_handle.close()
    return 0


def _patch(base: dict, patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _patch(base[key], value)
        else:
            base[key] = value


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True, choices=list(M.CSOI_ARMS))
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke only: stop after N optimisation steps (recorded in the manifest)")
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else []
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
