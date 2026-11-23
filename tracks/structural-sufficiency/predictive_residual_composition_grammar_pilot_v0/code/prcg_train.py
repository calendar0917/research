"""D0 seed-0 training for the three frozen PRCG-v0 arms.

    uv run python .../code/prcg_train.py --device cuda
    uv run python .../code/prcg_train.py --arms PRED_GRAMMAR --epochs 3 --tag smoke

The three arms differ *only* in (grammar, port mode); every other byte of the
model, the initialisation and the optimisation schedule is identical, and the
init hashes are asserted equal before training.  One seed, D0 only, no search.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import prcg_common as C
import prcg_data as D
import prcg_discover as PD
import prcg_graph as G
import prcg_model as M
import prcg_tensor as T


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["FREQ_GRAMMAR", "PRED_GRAMMAR", "PRED_PORT_MEAN"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--tag", default="", help="suffix for smoke outputs")
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow-nondeterministic", action="store_true",
                    help="CPU index_add reductions are otherwise forced deterministic")
    return ap.parse_args(argv)


def _make_deterministic(threads: int) -> bool:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)
    return True


def _lr_at(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    import math

    if step < warmup:
        return peak * float(step + 1) / float(max(1, warmup))
    if total <= warmup:
        return peak
    frac = float(step - warmup) / float(max(1, total - warmup))
    frac = min(1.0, max(0.0, frac))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * frac))


def _batches(rows: np.ndarray, batch: int, generator) -> list[np.ndarray]:
    import torch

    order = torch.randperm(int(rows.size), generator=generator).numpy()
    shuffled = np.asarray(rows, np.int64)[order]
    return [shuffled[i:i + batch] for i in range(0, shuffled.size, batch)]


def _predict(model: M.PrcgModel, rows: np.ndarray, cache: Mapping[str, np.ndarray],
             index: Mapping[int, int], bundle: D.DataBundle, mode: str, device: str,
             batch: int) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, rows.size, batch):
            chunk = rows[start:start + batch]
            bt = T.collate(chunk.tolist(), cache, index, bundle.graph_mean,
                           bundle.graph_scale, device=device)
            out.append(model(bt, mode=mode)["yhat_std"].detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _to_label(z: np.ndarray, bundle: D.DataBundle) -> np.ndarray:
    return z * bundle.scaler["std"] + bundle.scaler["mean"]


def run_arm(arm: str, seed: int, init_seed: int, config: Mapping[str, Any], bundle: D.DataBundle,
            molecules: Mapping[int, G.Molecule], grammars: Mapping[str, G.Grammar],
            caches: Mapping[str, Mapping[str, np.ndarray]], index: Mapping[int, int],
            rows_fit: np.ndarray, rows_eval: np.ndarray, device: str, epochs_override,
            out_dir: Path, tag: str, deterministic: bool = True) -> dict[str, Any]:
    import torch

    spec = dict(config["arms"][arm])
    grammar_name = str(spec["grammar"])
    port_mode = str(spec["port"])
    recipe = dict(config["recipe"])
    epochs = int(epochs_override if epochs_override is not None else recipe["epochs"])
    batch = int(recipe["batch_molecules"])
    warmup_epochs = int(recipe["warmup_epochs"])

    C.seed_everything(seed)
    cache = caches[grammar_name]
    atom_dims, bond_dims = D.atom_bond_dims()
    model, ledger = M.build_model(atom_dims, bond_dims, init_seed, config)
    model.to(device)
    init_hashes = dict(ledger["init_hashes"])

    y_fit = bundle.labels[rows_fit]
    y_eval = bundle.labels[rows_eval]
    std_fit = (y_fit - bundle.scaler["mean"]) / bundle.scaler["std"]
    std_eval = (y_eval - bundle.scaler["mean"]) / bundle.scaler["std"]

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "norm" in name) else decay).append(p)
    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": float(recipe["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=float(recipe["peak_lr"]), betas=(0.9, 0.999), eps=1e-8)

    n_batches = int(np.ceil(rows_fit.size / batch))
    total_steps = n_batches * epochs
    warmup_steps = n_batches * warmup_epochs
    generator = torch.Generator().manual_seed(int(seed) * 100003 + 17)

    history: list[dict[str, Any]] = []
    best = {"epoch": -1, "dev_rmse": float("inf"), "state": None, "dev_pred": None,
            "fit_rmse": float("inf"), "dev_mae": float("inf")}
    stale = 0
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for chunk in _batches(rows_fit, batch, generator):
            lr = _lr_at(step, total_steps, warmup_steps, float(recipe["peak_lr"]),
                        float(recipe["min_lr"]))
            for group in optim.param_groups:
                group["lr"] = lr
            bt = T.collate(chunk.tolist(), cache, index, bundle.graph_mean,
                           bundle.graph_scale, device=device)
            target = torch.as_tensor(
                (bundle.labels[chunk] - bundle.scaler["mean"]) / bundle.scaler["std"],
                dtype=torch.float32, device=device)
            optim.zero_grad(set_to_none=True)
            yhat = model(bt, mode=port_mode)["yhat_std"]
            loss = torch.mean((yhat - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} ({arm})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"]))
            optim.step()
            losses.append(float(loss.detach()))
            step += 1
        train_loss = float(np.mean(losses))
        pred_eval = _predict(model, rows_eval, cache, index, bundle, port_mode, device,
                             int(recipe["eval_batch"]))
        dev_rmse = C.rmse(_to_label(pred_eval, bundle), y_eval)
        dev_mae = C.mae(_to_label(pred_eval, bundle), y_eval)
        pred_fit = _predict(model, rows_fit, cache, index, bundle, port_mode, device,
                            int(recipe["eval_batch"]))
        fit_rmse = C.rmse(_to_label(pred_fit, bundle), y_fit)
        history.append({"epoch": epoch, "lr": lr, "train_loss_std": train_loss,
                        "train_rmse": fit_rmse, "dev_rmse": dev_rmse, "dev_mae": dev_mae,
                        "seconds": time.time() - t0})
        if dev_rmse < best["dev_rmse"] - 1e-9:
            best = {"epoch": epoch, "dev_rmse": dev_rmse, "dev_mae": dev_mae,
                    "fit_rmse": fit_rmse,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "dev_pred": pred_eval.copy()}
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(recipe["min_epochs"]) and stale >= int(recipe["patience"]):
            break
    wall = time.time() - t0

    # final sanity: reload the best state and re-predict
    model.load_state_dict(best["state"])
    model.eval()
    pred_eval = _predict(model, rows_eval, cache, index, bundle, port_mode, device,
                         int(recipe["eval_batch"]))
    pred_fit = _predict(model, rows_fit, cache, index, bundle, port_mode, device,
                        int(recipe["eval_batch"]))
    dev_rmse = C.rmse(_to_label(pred_eval, bundle), y_eval)
    dev_mae = C.mae(_to_label(pred_eval, bundle), y_eval)
    if abs(dev_rmse - best["dev_rmse"]) > 1e-6:
        raise RuntimeError("checkpoint reload does not reproduce the best Dev RMSE")

    # non-degeneracy of the trained word states
    dev_words = []
    with torch.no_grad():
        for start in range(0, rows_eval.size, int(recipe["eval_batch"])):
            chunk = rows_eval[start:start + int(recipe["eval_batch"])]
            bt = T.collate(chunk.tolist(), cache, index, bundle.graph_mean,
                           bundle.graph_scale, device=device)
            dev_words.append(model(bt, mode=port_mode)["z"].detach().cpu().numpy())
    z_dev = np.concatenate(dev_words, axis=0)
    z_centred = z_dev - z_dev.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(z_centred, compute_uv=False)
    participation = float((sv ** 2).sum() ** 2 / max(1e-30, (sv ** 4).sum()))

    suffix = f"_{tag}" if tag else ""
    stem = f"{arm}_seed{seed}{suffix}"
    torch.save({"state_dict": best["state"], "arm": arm, "seed": int(seed),
                "grammar": grammar_name, "port": port_mode, "epoch": best["epoch"]},
               out_dir / f"{stem}.pt")
    result = {
        "protocol_id": config["protocol_id"],
        "arm": arm, "grammar": grammar_name, "port": port_mode, "seed": int(seed),
        "init_seed": int(init_seed), "device": device, "deterministic": deterministic,
        "torch_threads": int(torch.get_num_threads()),
        "split": {"fit": int(rows_fit.size), "eval": int(rows_eval.size),
                  "sha256": bundle.hashes["split"]["sha256"]},
        "hashes": bundle.hashes,
        "code_hashes": C.code_hashes(),
        "code_sha": C.code_sha(sorted(C.CODE_DIR.glob("prcg_*.py"))),
        "git_commit": C.git_commit(),
        "config": config,
        "parameters": ledger["parameters"],
        "init_hashes": init_hashes,
        "grammar_rules": len(grammars[grammar_name].rules),
        "best_epoch": int(best["epoch"]),
        "epochs_run": len(history),
        "dev_rmse": dev_rmse,
        "dev_mae": dev_mae,
        "fit_rmse": C.rmse(_to_label(pred_fit, bundle), y_fit),
        "fit_mae": C.mae(_to_label(pred_fit, bundle), y_fit),
        "dev_pred": _to_label(pred_eval, bundle).tolist(),   # label units
        "fit_pred": _to_label(pred_fit, bundle).tolist(),     # label units
        "dev_rows": rows_eval.tolist(),
        "fit_rows": rows_fit.tolist(),
        "dev_y": y_eval.tolist(),
        "wall_seconds": wall,
        "wall_hours": wall / 3600.0,
        "epoch_seconds_median": float(np.median([h["seconds"] for h in history[1:]]))
        if len(history) > 1 else float(wall),
        "history": history,
        "z_effective_rank_participation": participation,
        "z_variance_mean": float(z_dev.var(axis=0).mean()),
        "z_rows": int(z_dev.shape[0]),
        "finished_utc": C.now_utc(),
    }
    path = C.write_json(out_dir / f"{stem}.json", result)
    print(json.dumps({"arm": arm, "best_epoch": result["best_epoch"],
                      "dev_rmse": round(dev_rmse, 4), "dev_mae": round(dev_mae, 4),
                      "fit_rmse": round(result["fit_rmse"], 4),
                      "epochs": len(history), "wall_s": round(wall, 1),
                      "z_participation": round(participation, 3),
                      "written": str(path)}, indent=1))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    config = C.load_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = D.load_bundle()
    rows_fit, rows_eval = bundle.fit_rows, bundle.eval_rows
    all_rows = np.concatenate([rows_fit, rows_eval])
    molecules = D.build_molecules(bundle.raw, all_rows)
    index = {int(r): i for i, r in enumerate(all_rows.tolist())}
    grammars = {name: PD.load_grammar(C.GRAMMAR_DIR / f"{name}_grammar.json")
                for name in ("predictive", "frequency")}
    caches = {name: T.featurize(molecules, all_rows.tolist(), grammar)
              for name, grammar in grammars.items()}
    deterministic = False
    if not args.allow_nondeterministic:
        deterministic = _make_deterministic(int(args.threads))
    results = {}
    for arm in args.arms:
        if arm not in config["arms"]:
            raise SystemExit(f"unknown arm {arm}")
        results[arm] = run_arm(arm, int(args.seed), int(args.init_seed), config, bundle,
                              molecules, grammars, caches, index, rows_fit, rows_eval,
                              args.device, args.epochs, out_dir, args.tag, deterministic)
    hashes = {arm: r["init_hashes"] for arm, r in results.items()}
    unique = {json.dumps(v, sort_keys=True) for v in hashes.values()}
    if len(unique) != 1:
        raise RuntimeError("arms were not initialised identically")
    if len(results) > 1:
        print(f"identical initialisation verified across {len(results)} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())