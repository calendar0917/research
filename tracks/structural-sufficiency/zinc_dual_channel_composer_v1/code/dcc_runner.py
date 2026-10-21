"""Runner for the P3 dual-channel composer (L / C, seed 0).

Everything in the supervised loop is the existing ``ad_train.train`` used by
zinc_recipe_trunk_v2: recipe B, batch 128, 600 epochs, 500-step warmup,
per-step cosine 1e-3 -> 1e-5, wd 1e-5, clip 5, raw-y L1, full valid every
epoch, strict best-valid selection.  This module only supplies the model
builder, the P3N collate wrapper and artifact hashing.  Official test is never
read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models as M  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
RECIPE = "B"


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{arm}_s{int(seed)}"


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def collate_p3n(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
    """Use the exact P3N batch layout; candidate is only a label here."""
    return T1.collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)


def make_builder(arm: str):
    organized = arm == "L"
    if arm not in ("L", "C"):
        raise ValueError(f"unknown arm {arm!r}")

    def builder(candidate, norm, seed, device):
        return M.build_model(candidate, norm, seed, device, organized_local=organized)

    return builder


def check_cache() -> None:
    missing = [p for p in (C.CACHE / "ad_train.pt", C.CACHE / "ad_valid.pt") if not p.exists()]
    if missing:
        raise FileNotFoundError("missing train/valid cache: " + ", ".join(str(p) for p in missing))


def write_artifacts(out: Path, info: dict) -> dict:
    best = out / "best.pt"
    final = out / "final.pt"
    pred = out / "predictions" / "valid_best.npz"
    art = {
        "run_dir": str(out.relative_to(C.REPO)),
        "candidate": info.get("candidate"),
        "arm": info.get("model_info", {}).get("arm"),
        "organized_local": info.get("model_info", {}).get("organized_local"),
        "seed": info.get("seed"),
        "recipe_variant": info.get("recipe_variant"),
        "parameter_count": info.get("parameter_count"),
        "parameters_breakdown": info.get("model_info", {}).get("parameters_breakdown"),
        "initial_state_sha256": info.get("model_info", {}).get("initial_state_sha256"),
        "p3_reference_state_sha256": info.get("model_info", {}).get("p3_reference_state_sha256"),
        "best_valid": info.get("best_valid"),
        "fixed_objects": info.get("fixed_objects"),
        "gpu_hours_run_level": info.get("gpu_hours"),
        "wall_seconds": info.get("wall_seconds"),
        "sha256": {
            "best_pt": _sha(best),
            "final_pt": _sha(final),
            "valid_best_npz": _sha(pred),
            "train_json": _sha(out / "train.json"),
            "curve_csv": _sha(out / "curve.csv"),
        },
    }
    C.write_json(out / "artifacts.json", art)
    return art


def train(arm: str, seed: int, device, *, recipe: str = RECIPE,
          epochs: int | None = None, tag: str | None = None, force: bool = False) -> dict:
    check_cache()
    out = run_dir(arm, seed, tag)
    info = T1.train(
        f"DCC_{arm}",
        seed,
        device,
        epochs=epochs,
        tag=tag,
        force=force,
        recipe_variant=recipe,
        model_builder=make_builder(arm),
        out_dir=out,
        collate_fn=collate_p3n,
    )
    art = write_artifacts(out, info)
    C.log(
        f"[dcc-{arm}] seed={seed} best={info['best_valid']['valid_mae']:.9f}"
        f"@{info['best_valid']['epoch']} final={info['fixed_objects']['final']['valid_mae']:.9f} "
        f"params={info['parameter_count']} wall={info['wall_seconds']:.1f}s "
        f"init={info['model_info']['initial_state_sha256'][:16]}"
    )
    return info


def timing(arm: str, seed: int, device, *, epochs: int = 3, recipe: str = RECIPE) -> dict:
    """Measure full-10k epochs plus full valid; never use a 1024-row projection."""
    check_cache()
    t0 = time.perf_counter()
    info = train(arm, seed, device, recipe=recipe, epochs=epochs, tag="timing", force=True)
    total_seconds = time.perf_counter() - t0
    curve = info["curve"]
    epoch_seconds = [float(e["seconds"]) for e in curve]
    sum_epoch = float(np.sum(epoch_seconds))
    mean_epoch = float(np.mean(epoch_seconds))
    steady_epoch = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else mean_epoch
    final_eval_seconds = float(info["wall_seconds"]) - sum_epoch
    startup_seconds = float(total_seconds) - float(info["wall_seconds"])
    process_reserve_seconds = 30.0
    projected_train = 600.0 * mean_epoch
    projected_total = 1.15 * projected_train + startup_seconds + final_eval_seconds + process_reserve_seconds
    out = {
        "protocol_id": "zinc-dual-channel-composer-v1",
        "arm": arm,
        "seed": int(seed),
        "recipe": recipe,
        "timing_rows": "full official train 10000; no 1024-row substitution",
        "epochs_measured": int(len(epoch_seconds)),
        "steps_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "updates_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "epoch_seconds_including_full_valid": epoch_seconds,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady_epoch,
        "sum_epoch_seconds": sum_epoch,
        "startup_seconds": startup_seconds,
        "final_best_final_eval_seconds": final_eval_seconds,
        "process_reserve_seconds": process_reserve_seconds,
        "projected_600ep_train_seconds": projected_train,
        "projected_600ep_total_seconds_with_15pct_and_overheads": projected_total,
        "projected_600ep_gpu_hours": projected_total / 3600.0,
        "run_wall_seconds": float(total_seconds),
        "train_json_wall_seconds": float(info["wall_seconds"]),
        "gpu_hours_run_level": float(info["gpu_hours"]),
        "parameters": int(info["parameter_count"]),
        "parameter_audit": info["parameter_audit"],
        "gradient_audit": info["gradient_audit"],
        "best_valid_at_timing_horizon": info["best_valid"],
        "initial_state_sha256": info["model_info"]["initial_state_sha256"],
        "official_test_loaded": False,
        "note": "Projection uses 600 * mean full-epoch seconds * 1.15 plus the measured one-time startup/final-eval and a 30 s process reserve; job occupancy is recorded separately from rr status.",
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / f"timing_{arm}.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing"])
    p.add_argument("--arm", required=True, choices=["L", "C"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--recipe", default=RECIPE, choices=["A", "B", "C"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "timing":
        timing(args.arm, args.seed, device, epochs=args.epochs or 3, recipe=args.recipe)
    else:
        train(args.arm, args.seed, device, recipe=args.recipe, epochs=args.epochs,
              tag=args.tag, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
