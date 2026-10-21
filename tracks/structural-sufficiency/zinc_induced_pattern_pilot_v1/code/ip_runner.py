"""Runner for the rooted induced-path pilot (J / Q).

Everything in the supervised loop is the existing ``ad_train.train`` used by
zinc_recipe_trunk_v2; this module only supplies the model builder and the
path-aware collate function.  Recipe B is the pre-declared fixed recipe.
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
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_train as T1  # noqa: E402
import ip_data  # noqa: E402
import ip_models  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
RECIPE = "B"


def run_dir(candidate: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{candidate}_s{int(seed)}"


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def write_artifacts(out: Path, info: dict) -> dict:
    best = out / "best.pt"
    final = out / "final.pt"
    pred = out / "predictions" / "valid_best.npz"
    art = {
        "run_dir": str(out.relative_to(C.REPO)),
        "candidate": info.get("candidate"),
        "seed": info.get("seed"),
        "recipe_variant": info.get("recipe_variant"),
        "parameter_count": info.get("parameter_count"),
        "parameters_breakdown": info.get("model_info", {}).get("parameters_breakdown"),
        "best_valid": info.get("best_valid"),
        "fixed_objects": info.get("fixed_objects"),
        "gpu_hours": info.get("gpu_hours"),
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


def check_cache() -> None:
    missing = [ip_data.cache_path(s) for s in ("train", "valid") if not ip_data.cache_path(s).exists()]
    if missing:
        raise FileNotFoundError(
            "missing path cache: " + ", ".join(str(p) for p in missing) +
            "; run `python ip_data.py prep` on this host first"
        )


def train(candidate: str, seed: int, device, *, recipe: str = RECIPE,
          epochs: int | None = None, tag: str | None = None, force: bool = False) -> dict:
    check_cache()
    out = run_dir(candidate, seed, tag)
    info = T1.train(
        candidate,
        seed,
        device,
        epochs=epochs,
        tag=tag,
        force=force,
        recipe_variant=recipe,
        model_builder=ip_models.build_model,
        out_dir=out,
        collate_fn=ip_data.collate_paths,
    )
    art = write_artifacts(out, info)
    C.log(f"[ip-run] {candidate} s{seed} best={info['best_valid']['valid_mae']:.6f}"
          f"@{info['best_valid']['epoch']} params={info['parameter_count']} "
          f"gpu_h={info['gpu_hours']:.4f} best_sha={art['sha256']['best_pt'][:12]}")
    return info


def timing(candidates, device, *, epochs: int = 2, recipe: str = RECIPE) -> dict:
    check_cache()
    out = {}
    for cand in candidates:
        t0 = time.perf_counter()
        info = train(cand, 0, device, recipe=recipe, epochs=epochs, tag="timing", force=True)
        wall = time.perf_counter() - t0
        per_epoch = float(np.mean([e["seconds"] for e in info["curve"]]))
        out[cand] = {
            "timed_seconds": wall,
            "mean_epoch_seconds_including_valid": per_epoch,
            "projected_gpu_hours_600": per_epoch * 600 / 3600.0,
            "projected_gpu_hours_600_with_10pct": per_epoch * 600 / 3600.0 * 1.10,
            "parameters": info["parameter_count"],
            "parameters_breakdown": info["model_info"].get("parameters_breakdown"),
            "optimizer_parameters": info["parameter_audit"]["optimizer_parameters"],
            "all_parameters_require_grad": info["parameter_audit"]["all_parameters_require_grad"],
            "gradient_audit": info["gradient_audit"],
            "gpu_hours_2ep": info["gpu_hours"],
        }
        C.log(f"[ip-timing] {cand}: {per_epoch:.3f}s/epoch, projected {out[cand]['projected_gpu_hours_600']:.4f} GPU-h")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "timing.json", out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing"])
    p.add_argument("--candidate", default="J")
    p.add_argument("--candidates", default="J,Q")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--recipe", default=RECIPE, choices=["A", "B", "C"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "timing":
        timing([x for x in args.candidates.split(",") if x], device,
               epochs=args.epochs or 2, recipe=args.recipe)
    else:
        train(args.candidate, args.seed, device, recipe=args.recipe, epochs=args.epochs,
              tag=args.tag, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
