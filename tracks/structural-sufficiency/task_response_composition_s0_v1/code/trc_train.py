"""Seed-0 training for the R0 response-composition candidate family.

One launch trains a full run.  Epochs 1..5 are warmup and are horizon
independent; after epoch 5 the common horizon ``E`` is fixed from the observed
epoch-3..5 median epoch time (including the three-split evaluation) unless a
decided horizon is already recorded in ``analysis/timing.json``.  The cosine
schedule is then built for that ``E``.

Every epoch evaluates the complete Train, Valid and Test splits in eval mode and
reports original-label-unit RMSE/MAE.  Model selection is Valid only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import trc_common as C
import trc_model as M
import trc_prepare as P
import trc_pairs as PR


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def choose_horizon(epoch_seconds: Sequence[float], cfg: Mapping[str, Any]) -> dict[str, Any]:
    steady = sorted(float(x) for x in epoch_seconds)
    median = float(np.median(steady)) if steady else float("nan")
    out: dict[str, Any] = {
        "observed_epoch_seconds": [float(x) for x in steady],
        "median_epoch_seconds": median,
        "threshold_minutes": 50.0,
        "safety_factor": 1.25,
    }
    for cand in (int(cfg["recipe"]["epochs_default"]), int(cfg["recipe"]["epochs_fallback"])):
        est_min = median * cand * 1.25 / 60.0
        out[f"estimate_{cand}_minutes"] = est_min
        if est_min <= 50.0:
            out["epochs"] = cand
            out["rule"] = f"{cand} epochs x 1.25 <= 50 min"
            return out
    out["epochs"] = int(cfg["recipe"]["epochs_fallback"])
    out["rule"] = "fallback 160 epochs kept; later units reduced by real cost"
    return out


# ---------------------------------------------------------------------------
# dataset view
# ---------------------------------------------------------------------------


def ensure_cache(cfg: Mapping[str, Any], dataset: str, log=print) -> None:
    path = P.cache_path(cfg, dataset)
    if path.is_file():
        return
    log(f"  prepared cache missing, building {path}")
    pack = P.build_dataset(cfg, dataset, log=log)
    digest = P.save_pack(path, pack)
    log(f"  wrote {path} sha256={digest}")


class RunData:
    def __init__(self, cfg: Mapping[str, Any], dataset: str, train_fraction: float | None,
                 log=print) -> None:
        self.cfg = dict(cfg)
        self.dataset = dataset
        ensure_cache(cfg, dataset, log=log)
        self.pack = P.load_pack(cfg, dataset)
        self.raw = None
        self.n_mol = int(self.pack["n_molecules"])
        self.labels = np.asarray(self.pack["labels"], np.float64)
        self.row_ids = np.asarray(self.pack["row_ids"], np.int64)
        self.splits = {
            "train": np.asarray(self.pack["train_idx"], np.int64),
            "valid": np.asarray(self.pack["valid_idx"], np.int64),
            "test": np.asarray(self.pack["test_idx"], np.int64),
        }
        self.full_train = self.splits["train"].copy()
        self.train_fraction = train_fraction
        if train_fraction is not None:
            self.splits["train"] = PR.subset_rows(self.row_ids, self.full_train, train_fraction)
            log(f"  train subset {train_fraction:.2f}: {self.splits['train'].size} molecules")
        self.scaler = C.target_scaler(self.labels, self.splits["train"])
        self.y_std = (self.labels - self.scaler["mean"]) / self.scaler["std"]
        self.atom_dims, self.bond_dims = C.official_feature_dims()

    def raw_cache(self) -> dict[str, Any]:
        if self.raw is None:
            self.raw = P.D.load_raw_cache(self.cfg, self.dataset)
        return self.raw


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict_all(model: M.ResponseModel, data: "RunData", erase_distance: bool = False):
    import torch

    model.eval()
    phi = None
    with torch.no_grad():
        phi = None if model.arm == "GLOBAL" else model.frame_phi()
        preds: dict[str, np.ndarray] = {}
        stats: dict[str, dict[str, float]] = {}
        for split, rows in data.splits.items():
            out = np.zeros(len(rows), np.float64)
            cut_sq = 0.0
            cut_n = 0
            var_list: list[np.ndarray] = []
            for start in range(0, len(rows), 128):
                part = rows[start:start + 128]
                idx = torch.as_tensor(part.astype(np.int64), device=model.device)
                if model.arm == "GLOBAL":
                    res = {"f_bar": model.global_scores(idx, phi), "f_cut": None}
                else:
                    res = model.molecule_cuts(idx, phi, erase_distance=erase_distance)
                out[start:start + len(part)] = res["f_bar"].detach().double().cpu().numpy()
                if res.get("f_cut") is None:
                    continue
                f = res["f_cut"]
                owner = res["owner"]
                y = torch.as_tensor(data.y_std[part], device=model.device, dtype=torch.float32)
                per = torch.zeros(len(part), device=model.device).index_add_(
                    0, owner, f) / res["counts"].float().clamp(min=1.0)
                var = torch.zeros(len(part), device=model.device).index_add_(
                    0, owner, (f - per[owner]) ** 2) / res["counts"].float().clamp(min=1.0)
                var_list.append(var.detach().cpu().numpy())
                cut_sq += float(((f - y[owner]) ** 2).sum())
                cut_n += int(f.numel())
            preds[split] = out
            s: dict[str, float] = {}
            if cut_n:
                var_all = np.concatenate(var_list)
                sigma = float(data.scaler["std"])
                s["cut_var_mean"] = float(np.mean(var_all))
                s["cut_rmse_std"] = float(math.sqrt(cut_sq / cut_n))
                s["cut_rmse"] = float(math.sqrt(cut_sq / cut_n) * sigma)
                s["cut_disp"] = float(math.sqrt(np.mean(var_all)) * sigma)
                s["cut_n"] = float(cut_n)
                s["parses_mean"] = float(cut_n / max(1, len(rows)))
            stats[split] = s
    return preds, stats


def raw_metrics(pred_std: np.ndarray, labels: np.ndarray, scaler: Mapping[str, float]) -> dict[str, float]:
    pred = pred_std * float(scaler["std"]) + float(scaler["mean"])
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
# training
# ---------------------------------------------------------------------------


def build_model(cfg: Mapping[str, Any], data: RunData, arm: str):
    import torch

    model = M.ResponseModel(cfg, data.atom_dims, data.bond_dims, arm)
    model.load_pack(data.pack)
    atom_support, bond_support = M.category_support(
        data.raw_cache(), data.splits["train"], data.atom_dims, data.bond_dims)
    zeroed = M.apply_atom_bond_support(model, atom_support, bond_support)
    return model, zeroed


def train(args: argparse.Namespace) -> int:
    import torch

    C.set_deterministic(int(args.seed))
    cfg = dict(C.load_config())
    if args.overrides:
        _patch(cfg, json.loads(args.overrides))
    cfg.setdefault("cache_stem", args.cache_stem)
    run_id = args.run_id
    rdir = C.run_dir(run_id)
    log_path = rdir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    device = args.device
    log(f"run={run_id} arm={args.arm} supervision={args.supervision} device={device} "
        f"fraction={args.train_fraction}")
    data = RunData(cfg, args.dataset, args.train_fraction, log=log)
    log(f"  scaler mean={data.scaler['mean']:.6f} std={data.scaler['std']:.6f} "
        f"n_train={data.splits['train'].size}")

    model, zeroed = build_model(cfg, data, args.arm)
    model = model.to(device)
    ledger = model.parameter_ledger()
    hashes = M.module_hashes(model, ["phi", "enc", "a", "q", "r", "b", "head", "a_add",
                                     "global_head", "null_cut", "beta", "h_empty"])
    log(f"  params={ledger['total']} ledger={ledger}")
    log(f"  init_hashes={hashes}")
    log(f"  zeroed_category_rows={zeroed}")

    store = PR.PairStore(cfg, data.pack, data.labels, None,
                         data.splits["train"], data.row_ids, log=log)
    log(f"  pair store: {store.summary()}")

    recipe = cfg["recipe"]
    n_train = int(data.splits["train"].size)
    steps_per_epoch = int(math.ceil(n_train / int(recipe["batch_molecules"])))
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))

    epochs_target = int(args.epochs)
    if epochs_target <= 0:
        epochs_target = 0  # decided after warmup
    total_steps = (epochs_target or int(recipe["epochs_default"])) * steps_per_epoch
    timing = C.read_json(C.analysis_dir() / "timing.json") if (C.analysis_dir() / "timing.json").is_file() else {}
    if int(args.epochs) <= 0 and "horizon_epochs" in timing:
        epochs_target = int(timing["horizon_epochs"])
        total_steps = epochs_target * steps_per_epoch
        log(f"  horizon from timing.json: E={epochs_target}")

    erase = bool(args.erase_distance)
    lam_var = float(args.lambda_var)
    epoch_rng = np.random.default_rng(int(args.seed) + 17)
    orders = [epoch_rng.permutation(n_train) for _ in range(400)]
    sampler = store.sampler()

    curves: list[dict[str, Any]] = []
    best = {"valid": float("inf"), "epoch": -1}
    step = 0
    t_resume = time.time()
    epoch_times: list[float] = []
    stopped_reason = ""
    for epoch in range(1, int(epochs_target or 400) + 1):
        t_epoch = time.time()
        model.train()
        order = orders[(epoch - 1) % len(orders)]
        main_losses: list[float] = []
        aux_losses: list[float] = []
        for start in range(0, n_train, int(recipe["batch_molecules"])):
            batch_rows = data.splits["train"][order[start:start + int(recipe["batch_molecules"])]]
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]), float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            idx = torch.as_tensor(batch_rows.astype(np.int64), device=device)
            need_aux = args.supervision != "NONE"
            n_main = len(batch_rows)
            if need_aux:
                pair_idx, aux_mols = sampler.sample(int(recipe["batch_pairs"]))
                all_rows = np.concatenate([batch_rows, aux_mols[0], aux_mols[1]])
                rows_t = torch.as_tensor(all_rows.astype(np.int64), device=device)
                res = model.molecule_forward(rows_t, erase_distance=erase)
                n_pair = len(pair_idx)
            else:
                all_rows = batch_rows
                res = model.molecule_forward(idx, erase_distance=erase)
                n_pair = 0
            y = torch.as_tensor(data.y_std[all_rows], device=device, dtype=torch.float32)
            if model.arm == "GLOBAL":
                l_main = torch.mean((res["f_bar"][:n_main] - y[:n_main]) ** 2)
            else:
                f_cut = res["f_cut"]
                owner = res["owner"]
                sq = (f_cut - y[owner]) ** 2
                per_mol = torch.zeros(len(all_rows), device=device).index_add_(0, owner, sq)
                per_mol = per_mol / res["counts"].float().clamp(min=1.0)
                l_main = per_mol[:n_main].mean()
                if lam_var > 0:
                    var_c = torch.zeros(len(all_rows), device=device).index_add_(
                        0, owner, (f_cut - per_mol[owner]) ** 2)
                    var_c = var_c / res["counts"].float().clamp(min=1.0)
                    l_main = l_main + lam_var * var_c[:n_main].mean()
            loss = l_main
            l_aux = torch.zeros((), device=device)
            if need_aux:
                pair_label = torch.as_tensor(store.delta_std_of(pair_idx), device=device, dtype=torch.float32)
                if args.supervision == "CUTRESP":
                    pairs_t = store.tensor_pairs(pair_idx, device)
                    delta = model.pair_cut_delta(pairs_t)
                else:
                    fb = res["f_bar"]
                    delta = fb[n_main:n_main + n_pair] - fb[n_main + n_pair:n_main + 2 * n_pair]
                l_aux = torch.mean((delta - pair_label) ** 2) / float(store.tau2)
                loss = l_main + float(recipe["aux_weight"]) * l_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["clip_norm"]))
            opt.step()
            step += 1
            main_losses.append(float(l_main.detach()))
            aux_losses.append(float(l_aux.detach()) if need_aux else 0.0)

        # horizon decision after warmup (epochs 3..5 steady state)
        if int(epochs_target) == 0 and epoch == int(recipe["warmup_epochs"]):
            recent = epoch_times[-3:] if len(epoch_times) >= 3 else epoch_times
            decision = choose_horizon(recent, cfg)
            epochs_target = int(decision["epochs"])
            total_steps = epochs_target * steps_per_epoch
            decision["run_id"] = run_id
            decision["step"] = "first warmup run in this budget"
            existing = C.read_json(C.analysis_dir() / "timing.json") if (C.analysis_dir() / "timing.json").is_file() else {}
            existing.setdefault("history", []).append(decision)
            existing["horizon_epochs"] = epochs_target
            existing["decided_by"] = run_id
            C.write_json(C.analysis_dir() / "timing.json", existing)
            log(f"  HORIZON E={epochs_target} ({decision['rule']}, "
                f"median_epoch_s={decision['median_epoch_seconds']:.2f})")

        train_seconds = time.time() - t_epoch
        preds, cut_stats = predict_all(model, data, erase_distance=erase)
        metrics = {s: raw_metrics(preds[s], data.labels[data.splits[s]], data.scaler)
                   for s in C.SPLITS}
        row: dict[str, Any] = {
            "run_id": run_id, "epoch": epoch,
            "lr": lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]), float(recipe["min_lr"])),
            "train_loss": float(np.mean(main_losses)),
            "aux_loss": float(np.mean(aux_losses)),
            "epoch_seconds": time.time() - t_epoch,
            "train_seconds": train_seconds,
        }
        for s in C.SPLITS:
            row[f"{s}_rmse"] = metrics[s]["rmse"]
            row[f"{s}_mae"] = metrics[s]["mae"]
        for split in C.SPLITS:
            for key, value in cut_stats[split].items():
                row[f"{split}_{key}"] = value
        row["train_cut_rmse_mol"] = row.get("train_cut_rmse", float("nan"))
        curves.append(row)
        C.write_csv(rdir / "curves.csv", curves)
        better = row["valid_rmse"] < best["valid"] - 1e-12
        if better:
            best = {"valid": row["valid_rmse"], "epoch": epoch}
            torch.save({"state": model.state_dict(), "epoch": epoch, "row": row,
                        "scaler": data.scaler, "ledger": ledger, "hashes": hashes},
                       rdir / "state_best.pt")
            save_predictions(rdir / "predictions_best.npz", data, preds)
        log(f"epoch {epoch:3d}/{epochs_target or '?'} lr={row['lr']:.2e} "
            f"train={row['train_rmse']:.4f} valid={row['valid_rmse']:.4f} "
            f"test={row['test_rmse']:.4f} tr_cut={row.get('train_cut_rmse', float('nan')):.4f} "
            f"cutvar={row.get('train_cut_var_mean', float('nan')):.4f} "
            f"aux={row['aux_loss']:.4f} {row['epoch_seconds']:.1f}s")
        epoch_times.append(row["epoch_seconds"])
        if int(epochs_target) > 0 and epoch >= int(epochs_target):
            break
        if args.max_wall_seconds and time.time() - t_resume > args.max_wall_seconds:
            stopped_reason = "max_wall_seconds"
            log(f"STOP {stopped_reason} at epoch {epoch}")
            break
    else:
        stopped_reason = "epoch_cap"
        log(f"STOP {stopped_reason}")

    torch.save({"state": model.state_dict(), "epoch": int(curves[-1]["epoch"]),
                "row": curves[-1], "scaler": data.scaler, "ledger": ledger, "hashes": hashes},
               rdir / "state_final.pt")
    preds, _fin = predict_all(model, data, erase_distance=erase)
    save_predictions(rdir / "predictions_final.npz", data, preds)
    wall = time.time() - t_resume
    manifest = {
        "run_id": run_id, "arm": args.arm, "supervision": args.supervision,
        "dataset": args.dataset, "train_fraction": args.train_fraction,
        "epochs": int(curves[-1]["epoch"]), "epochs_planned": int(epochs_target) if epochs_target else None,
        "steps_per_epoch": steps_per_epoch, "batch_molecules": int(recipe["batch_molecules"]),
        "batch_pairs": int(recipe["batch_pairs"]),
        "erase_distance": erase, "lambda_var": lam_var,
        "seed": int(args.seed), "commit": C.git_commit(),
        "cache_sha256": data.pack["cache_sha256"], "cache_path": data.pack["cache_path"],
        "raw_cache_sha256": data.raw_cache()["cache_sha256"],
        "scaler": data.scaler, "n_train": int(data.splits["train"].size),
        "params": ledger, "init_hashes": hashes, "zeroed_categories": zeroed,
        "best": best, "final": curves[-1], "stopped_reason": stopped_reason,
        "wall_seconds": wall, "gpu_hours": wall / 3600.0,
        "max_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "environment": C.environment(),
        "pair_store": store.summary(),
        "overrides": json.loads(args.overrides) if args.overrides else None,
        "command": " ".join(["python", "-m", "trc_train", *[]]),
        "argv": list(getattr(args, "argv", [])),
    }
    C.write_json(rdir / "manifest.json", manifest)
    C.write_json(rdir / "config.json", {"cfg": cfg, "arm": args.arm,
                                        "supervision": args.supervision,
                                        "erase_distance": erase, "lambda_var": lam_var})
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
    ap.add_argument("--arm", required=True, choices=list(C.ARMS))
    ap.add_argument("--supervision", default="NONE", choices=list(C.SUPERVISIONS))
    ap.add_argument("--dataset", default="lipo", choices=["lipo", "esol"])
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-fraction", type=float, default=None)
    ap.add_argument("--erase-distance", action="store_true")
    ap.add_argument("--lambda-var", type=float, default=0.0)
    ap.add_argument("--max-wall-seconds", type=float, default=0.0)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--cache-stem", default="trc_cache")
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else []
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
