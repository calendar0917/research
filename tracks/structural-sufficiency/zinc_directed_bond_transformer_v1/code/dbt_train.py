"""Supervised trainer for the directed-bond Transformer (DBT), seed 0.

One trajectory, one recipe, fixed horizon declared before launch.  Valid is
evaluated every epoch on all 1,000 valid molecules and the best checkpoint is
selected by strict ``<`` (ties keep the earlier epoch).  Test is never
constructed here.

Resume: ``last.pt`` carries state_dict / optimizer / RNG / epoch / global_step
/ best / curve and is rewritten every ``--save-every`` epochs, so a preempted
run continues the *same* schedule (the cosine horizon is frozen at
``total_updates`` and is never reset).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_models as BM  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUNS = _EXP / "runs"

#: The single declared recipe for this round (see DESIGN.md §6).  Nothing here
#: is tuned on valid or test; the horizon is fixed before launch.
RECIPE: dict[str, Any] = {
    "optimizer": "adamw",
    "lr": 3.0e-4,
    "min_lr": 3.0e-6,
    "weight_decay": 1.0e-4,
    "warmup_updates": 1000,
    "epochs": 1500,
    "batch_size": 64,
    "valid_batch_size": 256,
    "eval_train_batch_size": 256,
    "dropout": 0.05,
    "grad_clip": 5.0,
    "loss": "raw-y L1",
    "seed": 0,
}


CONFIG = _EXP / "configs" / "dbt_v1.json"


def validate_config() -> dict:
    """The declared JSON config and this module's recipe must not drift."""
    cfg = C.read_json(CONFIG)
    rec = cfg["recipe"]
    keys = ("optimizer", "lr", "min_lr", "weight_decay", "warmup_updates", "epochs",
            "batch_size", "loss", "grad_clip", "dropout")
    bad = {k: (rec[k], RECIPE[k]) for k in keys if rec[k] != RECIPE[k]}
    if bad:
        raise RuntimeError(f"configs/dbt_v1.json disagrees with RECIPE: {bad}")
    return cfg


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def steps_per_epoch(n_rows: int, batch_size: int) -> int:
    return int(math.ceil(n_rows / float(batch_size)))


@torch.no_grad()
def evaluate(model: torch.nn.Module, cache, rel, rows: np.ndarray, *, device,
             batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = _to_device(BD.collate(cache, rel, chunk, valid=True), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    return pred, C.mae(pred, y)


def _param_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    total = C.parameter_count(model)
    opt = C.optimizer_parameter_count(optimizer)
    requires = sum(1 for p in model.parameters() if p.requires_grad)
    nparams = sum(1 for p in model.parameters())
    return {
        "parameters": int(total),
        "optimizer_parameters": int(opt),
        "parameters_in_optimizer": bool(total == opt),
        "parameter_tensors": int(nparams),
        "requires_grad_tensors": int(requires),
        "all_parameters_require_grad": bool(requires == nparams),
        "module_breakdown": {
            name: int(sum(p.numel() for p in mod.parameters()))
            for name, mod in model.named_children()
        },
    }


def _lr_at(step: int, total: int, recipe: Mapping[str, Any]) -> float:
    return C.cosine_with_warmup_lr(
        step, total, int(recipe["warmup_updates"]),
        float(recipe["lr"]), float(recipe["min_lr"]),
    )


def build(seed: int, device, *, dropout: float) -> tuple[torch.nn.Module, dict]:
    norm = D.load_norm()
    scales = C.read_json(_EXP / "analysis" / "graph_scales.json")
    model, info = BM.build_model(norm, seed, device, graph_scales=scales, dropout=dropout)
    return model, info


def train(seed: int, device, *, epochs: int | None = None, tag: str | None = None,
          force: bool = False, resume: bool = False, save_every: int = 25,
          recipe_override: Mapping[str, Any] | None = None) -> dict:
    cfg = validate_config()
    recipe = dict(RECIPE)
    if recipe_override:
        recipe.update(recipe_override)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    out = RUNS / (f"{tag}_" if tag else "") / f"dbt_s{int(seed)}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force and not resume:
        C.log(f"[dbt s{seed}] already complete in {out}; use --force to rerun")
        return C.read_json(out / "train.json")

    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    train_rel = BD.load_relation_split("train")
    valid_rel = BD.load_relation_split("valid")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    model, model_info = build(seed, device, dropout=float(recipe["dropout"]))
    params = C.parameter_count(model)
    if params > 500_000:
        raise RuntimeError(f"{params} > 500k predictor parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    audit = _param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"] or not audit["all_parameters_require_grad"]:
        raise RuntimeError(f"optimizer/grad coverage failure: {audit}")

    spe = steps_per_epoch(len(rows_train), int(recipe["batch_size"]))
    total_updates = spe * n_epochs
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    global_step = 0
    start_epoch = 1

    if resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in state["state_dict"].items()})
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["gen_state"])
        curve = list(state["curve"])
        best = dict(state["best"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"]) + 1
        if int(state["recipe"]["epochs"]) != n_epochs:
            raise RuntimeError(
                "resume horizon mismatch: last.pt was written for "
                f"{state['recipe']['epochs']} epochs, asked for {n_epochs}"
            )
        C.log(f"[dbt s{seed}] resumed at epoch {state['epoch']} step {global_step} "
              f"best={best['valid_mae']:.6f}@{best['epoch']}")

    started = time.perf_counter()
    lr = _lr_at(max(global_step, 1), total_updates, recipe)
    predicted_valid: list[float] = []

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), int(recipe["batch_size"])):
            rows = order[start:start + int(recipe["batch_size"])]
            batch = _to_device(BD.collate(train_cache, train_rel, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch} step {global_step + 1}")
            global_step += 1
            lr = _lr_at(global_step, total_updates, recipe)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        pred_valid, valid_mae = evaluate(
            model, valid_cache, valid_rel, rows_valid, device=device,
            batch_size=int(recipe["valid_batch_size"]),
        )
        sec = time.perf_counter() - t_ep
        entry = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "online_train_mae": float(online),
            "valid_mae": float(valid_mae),
            "lr": float(lr),
            "seconds": float(sec),
            "steps": int(spe),
        }
        curve.append(entry)
        predicted_valid.append(float(valid_mae))
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(epoch),
                 "valid_mae": float(valid_mae), "candidate": "DBT", "seed": int(seed),
                 "parameter_count": int(params), "recipe": recipe},
                out / "best.pt",
            )
        C.log(
            f"[dbt s{seed}] ep={epoch:04d}/{n_epochs} valid={valid_mae:.6f} "
            f"online={online:.6f} lr={lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} "
            f"{sec:.2f}s"
        )
        if epoch % int(save_every) == 0 or epoch == n_epochs:
            C.atomic_torch_save(
                {
                    "state_dict": C.cpu_state(model),
                    "optimizer": optimizer.state_dict(),
                    "gen_state": gen.get_state(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "best": best,
                    "curve": curve,
                    "candidate": "DBT",
                    "seed": int(seed),
                    "recipe": {**recipe, "epochs": int(n_epochs),
                               "steps_per_epoch": int(spe),
                               "updates": int(total_updates)},
                },
                out / "last.pt",
            )

    wall = time.perf_counter() - started
    if (out / "last.pt").exists():
        C.atomic_torch_save(torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                            out / "final.pt")

    # fixed-object evaluation of the selected best checkpoint
    best_model, _ = build(seed, device, dropout=float(recipe["dropout"]))
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    C.load_state(best_model, state["state_dict"], device)
    pred_valid, best_valid = evaluate(
        best_model, valid_cache, valid_rel, rows_valid, device=device,
        batch_size=int(recipe["valid_batch_size"]),
    )
    _, best_train = evaluate(
        best_model, train_cache, train_rel, rows_train, device=device,
        batch_size=int(recipe["eval_train_batch_size"]),
    )
    _, final_valid = evaluate(
        model, valid_cache, valid_rel, rows_valid, device=device,
        batch_size=int(recipe["valid_batch_size"]),
    )
    _, final_train = evaluate(
        model, train_cache, train_rel, rows_train, device=device,
        batch_size=int(recipe["eval_train_batch_size"]),
    )
    preds = out / "predictions"
    preds.mkdir(exist_ok=True)
    np.savez(preds / "valid_best.npz", row=rows_valid, pred=pred_valid)

    info = {
        "round": "zinc-directed-bond-transformer-v1",
        "candidate": "DBT",
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_label_sha256": C.sha256_array(y_train),
            "valid_label_sha256": C.sha256_array(np.asarray(valid_cache["labels"], np.float64)),
            "path_cache": {s: {"meta": d["meta"], "sha256": C.sha256_file(BD.cache_path(s))}
                           for s, d in (("train", train_rel), ("valid", valid_rel))},
        },
        "model_info": model_info,
        "declared_config": str(CONFIG.relative_to(C.REPO)),
        "declared_total_parameters": int(cfg["model"]["total_parameters"]),
        "parameter_audit": audit,
        "recipe": {
            **recipe,
            "epochs": int(n_epochs),
            "steps_per_epoch": int(spe),
            "updates": int(total_updates),
            "actual_global_steps": int(global_step),
            "valid_every_epoch": True,
            "selection": "argmin(valid_mae), strict <, ties keep the earlier epoch",
        },
        "best_valid": best,
        "fixed_objects": {
            "best": {"valid_mae": float(best_valid),
                     "train_mae_eval_mode": float(best_train),
                     "epoch": int(best["epoch"])},
            "final": {"valid_mae": float(final_valid),
                      "train_mae_eval_mode": float(final_train),
                      "epoch": int(n_epochs)},
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "device": str(device),
        "smoke_override": bool(epochs is not None),
        "notes": [
            "online_train_mae is a within-epoch pre-update running mean",
            "train_mae_eval_mode is a full-10k eval-mode measurement of the fixed checkpoint",
            "the cosine horizon is total_updates = steps_per_epoch * epochs and is never reset",
        ],
    }
    C.write_json(out / "train.json", info)
    C.write_curve_csv(out / "curve.csv", curve)
    C.log(f"[dbt s{seed}] done best={best_valid:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s params={params}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None,
                   help="timing/smoke override; recorded as smoke_override")
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=25)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    train(args.seed, device, epochs=args.epochs, tag=args.tag, force=args.force,
          resume=args.resume, save_every=args.save_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
