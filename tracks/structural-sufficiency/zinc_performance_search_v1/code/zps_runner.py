"""Runner for zinc-performance-search-v1 (arms A = P3, B = L).

This round is a performance search on the official PyG ZINC subset
(10k/1k/1k, raw-y MAE).  The runner only supplies:

* the arm registry (A -> ``ad2_models.P3NullModel``; B -> the ``L`` arm of
  ``dcc_models.DualChannel``), both reused verbatim from their owning rounds;
* the declared recipe table R0-R3 (``ad_train.train`` gained additive optional
  ``min_lr`` / ``loss`` / ``ema`` arguments; defaults reproduce history);
* a per-run ``search.json`` that records the declared reason, the complete
  horizon and the projected cost *before* the run starts;
* an optional single-trajectory EMA checkpoint policy (P3/L only; no BN).

Official test is never read here.  Every run is a single model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_AD2 = _SS / "zinc_recipe_trunk_v2" / "code"
_DCC = _SS / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_AD2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import dcc_models as DM  # noqa: E402
import dcc_runner as DR  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
ROUND_ID = "zinc-performance-search-v1"

# Declared before the first supervised seed-0 run of this round.  Each entry is
# the *complete* recipe: 600 epochs, 79 steps/epoch, 47,400 updates, 500-step
# warm-up, per-step cosine to min_lr, clip 5.0, batch 128, full valid every
# epoch, strict best-valid selection.
RECIPES: dict[str, dict] = {
    "R0": dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
               warmup_steps=500, epochs=600, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=256, loss="l1"),
    "R1": dict(optimizer="adamw", lr=5.0e-4, min_lr=5.0e-6, weight_decay=1.0e-4,
               warmup_steps=500, epochs=600, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=256, loss="l1"),
    "R2": dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-3,
               warmup_steps=500, epochs=600, dropout=0.05, grad_clip=5.0,
               batch_size=128, valid_batch_size=256, loss="l1"),
    # R0L: R0 values with a 1200-epoch full cosine (94,800 updates).  Declared
    # as the round's one targeted extra config (horizon lever on the leader).
    "R0L": dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
                warmup_steps=500, epochs=1200, dropout=0.0, grad_clip=5.0,
                batch_size=128, valid_batch_size=256, loss="l1"),
    "R3": dict(optimizer="adamw", lr=5.0e-4, min_lr=5.0e-6, weight_decay=1.0e-4,
               warmup_steps=500, epochs=600, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=256, loss="smooth_l1",
               smooth_l1_beta=0.02),
}

ARMS = ("A", "B")


def run_dir(arm: str, recipe: str, seed: int, *, ema: bool = False,
            tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    suffix = "_ema" if ema else ""
    return RUNS / f"{prefix}{arm}_{recipe}{suffix}_s{int(seed)}"


def _candidates(arm: str):
    if arm == "A":
        return "P3N", A2.build_model, None
    if arm == "B":
        return "DCC_L", DR.make_builder("L"), DR.collate_p3n
    raise ValueError(f"unknown arm {arm!r}")


def _projected_hours(recipe: dict, epochs: int | None = None, *,
                     seconds_per_epoch: float = 2.6) -> dict:
    """Only used for the *pre-run declaration*; replaced by measured values."""
    ep = int(recipe["epochs"] if epochs is None else epochs)
    updates = math.ceil(10000 / int(recipe["batch_size"])) * ep
    return {
        "declared_epochs": ep,
        "declared_updates": int(updates),
        "assumed_seconds_per_epoch": float(seconds_per_epoch),
        "projected_gpu_hours_with_15pct": float(1.15 * ep * seconds_per_epoch / 3600.0),
    }


def train(arm: str, recipe_name: str, seed: int, device, *, ema: bool = False,
          epochs: int | None = None, tag: str | None = None, force: bool = False,
          reason: str | None = None, seconds_per_epoch: float = 2.6) -> dict:
    if recipe_name not in RECIPES:
        raise KeyError(f"unknown recipe {recipe_name!r}")
    recipe = dict(RECIPES[recipe_name])
    out = run_dir(arm, recipe_name, seed, ema=ema, tag=tag)
    out.mkdir(parents=True, exist_ok=True)
    declaration = {
        "round": ROUND_ID,
        "arm": arm,
        "recipe": recipe_name,
        "recipe_values": recipe,
        "seed": int(seed),
        "ema": bool(ema),
        "reason": reason,
        "declared_before_start": _projected_hours(recipe, epochs,
                                                  seconds_per_epoch=seconds_per_epoch),
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    C.write_json(out / "search.json", declaration)
    candidate, builder, collate_fn = _candidates(arm)
    ema_cfg = ({"decay": 0.999, "start_frac": 0.25} if ema else None)
    info = T1.train(
        candidate,
        seed,
        device,
        epochs=epochs,
        tag=tag,
        force=force,
        recipe_override=recipe,
        min_lr=recipe["min_lr"],
        loss=recipe["loss"],
        smooth_l1_beta=float(recipe.get("smooth_l1_beta", 0.02)),
        ema=ema_cfg,
        model_builder=builder,
        out_dir=out,
        collate_fn=collate_fn,
    )
    # ad_train writes train.json with the *merged* recipe; stamp the round and
    # the pre-run declaration so the record is self-contained.
    info["round"] = ROUND_ID
    info["search_declaration"] = declaration
    C.write_json(out / "train.json", info)
    best = info["best_valid"]
    msg = (f"[{ROUND_ID} {arm}-{recipe_name}{'-ema' if ema else ''} s{seed}] "
           f"online_best={best['valid_mae']:.6f}@{best['epoch']}")
    if info.get("best_valid_ema"):
        eb = info["best_valid_ema"]
        msg += f" ema_best={eb['valid_mae']:.6f}@{eb['epoch']}"
    C.log(msg + f" params={info['parameter_count']} wall={info['wall_seconds']:.1f}s")
    return info


def timing(arm: str, recipe_name: str, seed: int, device, *, epochs: int = 3,
           ema: bool = False) -> dict:
    t0 = time.perf_counter()
    info = train(arm, recipe_name, seed, device, ema=ema, epochs=epochs,
                 tag="timing", force=True, reason="timing only; not a result")
    total = time.perf_counter() - t0
    curve = info["curve"]
    sec = [float(e["seconds"]) for e in curve]
    mean_epoch = float(np.mean(sec))
    result = {
        "arm": arm,
        "recipe": recipe_name,
        "seed": int(seed),
        "epochs_measured": len(sec),
        "epoch_seconds_including_full_valid": sec,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": float(np.mean(sec[1:])) if len(sec) > 1 else mean_epoch,
        "steps_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "parameters": int(info["parameter_count"]),
        "projected_600ep_gpu_hours_with_15pct": 1.15 * 600 * mean_epoch / 3600.0,
        "run_wall_seconds": total,
        "gradient_audit": info["gradient_audit"],
        "official_test_loaded": False,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / f"timing_{arm}_{recipe_name}{'_ema' if ema else ''}.json", result)
    C.log(result)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing"])
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--recipe", default="R0", choices=sorted(RECIPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default=None)
    p.add_argument("--seconds-per-epoch", type=float, default=2.6)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "timing":
        timing(args.arm, args.recipe, args.seed, device, epochs=args.epochs or 3,
               ema=args.ema)
    else:
        train(args.arm, args.recipe, args.seed, device, ema=args.ema,
              epochs=args.epochs, tag=args.tag, force=args.force, reason=args.reason,
              seconds_per_epoch=args.seconds_per_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
