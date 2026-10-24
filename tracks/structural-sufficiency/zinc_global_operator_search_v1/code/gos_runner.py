"""Runner for zinc-global-operator-search-v1 (routes C0 / A / K).

Everything in the supervised loop is the existing ``ad_train.train``: recipe B
(AdamW, lr 1e-3 -> 1e-5, wd 1e-5, 600 epochs, 500-step warm-up, clip 5, raw-y
L1, batch 128, full valid every epoch, strict best-valid selection).  This
module only supplies the model builder, the relation collate and artifact
hashing.  Official test is never constructed here.
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
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import gos_data as GD  # noqa: E402
import gos_models as GM  # noqa: E402

RUNS = _EXP / "runs"
ANALYSIS = _EXP / "analysis"
RECIPE = "B"


def run_dir(route: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{route}_s{int(seed)}"


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def prepare() -> tuple[dict, dict, dict, float]:
    """Load train/valid caches, relation caches and the train-only scale."""
    for p in (C.CACHE / "ad_train.pt", C.CACHE / "ad_valid.pt"):
        if not p.exists():
            raise FileNotFoundError(f"missing train/valid cache {p}")
    rel_train = GD.load_rel("train")
    rel_valid = GD.load_rel("valid")
    sp = ANALYSIS / "rel_scale.json"
    if sp.exists():
        scale = C.read_json(sp)
    else:
        scale = GD.fit_scale(rel_train)
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        C.write_json(sp, scale)
    mean_nb = float(np.asarray(rel_train["n_bonds_graph"]).astype(np.float64).mean())
    return rel_train, rel_valid, scale, mean_nb


def make_builder(route: str, mean_n_bonds: float, dropout: float = 0.0):
    def builder(candidate, norm, seed, device):
        return GM.build_model(route, norm, seed, device, mean_n_bonds=mean_n_bonds,
                              dropout=dropout)

    return builder


def write_artifacts(out: Path, info: dict) -> dict:
    best = out / "best.pt"
    final = out / "final.pt"
    pred = out / "predictions" / "valid_best.npz"
    art = {
        "run_dir": str(out.relative_to(C.REPO)),
        "candidate": info.get("candidate"),
        "route": info.get("model_info", {}).get("route"),
        "seed": info.get("seed"),
        "recipe_variant": info.get("recipe_variant"),
        "parameter_count": info.get("parameter_count"),
        "parameters_breakdown": info.get("model_info", {}).get("parameters_breakdown"),
        "initial_state_sha256": info.get("model_info", {}).get("initial_state_sha256"),
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


def train(route: str, seed: int, device, *, recipe: str = RECIPE,
          epochs: int | None = None, tag: str | None = None, force: bool = False,
          lr: float | None = None, weight_decay: float | None = None,
          dropout: float | None = None) -> dict:
    rel_train, rel_valid, scale, mean_nb = prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    out = run_dir(route, seed, tag)
    info = T1.train(
        f"GOS_{route}",
        seed,
        device,
        epochs=epochs,
        tag=tag,
        force=force,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        recipe_variant=recipe,
        model_builder=make_builder(route, mean_nb, float(dropout or 0.0)),
        out_dir=out,
        collate_fn=collate,
    )
    art = write_artifacts(out, info)
    bo = info["fixed_objects"]["best"]
    C.log(
        f"[gos-{route}] seed={seed} best={info['best_valid']['valid_mae']:.9f}"
        f"@{info['best_valid']['epoch']} final={info['fixed_objects']['final']['valid_mae']:.9f} "
        f"params={info['parameter_count']} wall={info['wall_seconds']:.1f}s "
        f"init={info['model_info']['initial_state_sha256'][:16]} sha={art['sha256']['best_pt']}"
    )
    return info


def timing(route: str, seed: int, device, *, epochs: int = 3, recipe: str = RECIPE) -> dict:
    t0 = time.perf_counter()
    info = train(route, seed, device, recipe=recipe, epochs=epochs, tag="timing", force=True)
    total_seconds = time.perf_counter() - t0
    curve = info["curve"]
    epoch_seconds = [float(e["seconds"]) for e in curve]
    sum_epoch = float(np.sum(epoch_seconds))
    mean_epoch = float(np.mean(epoch_seconds))
    steady = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else mean_epoch
    final_eval_seconds = float(info["wall_seconds"]) - sum_epoch
    startup = float(total_seconds) - float(info["wall_seconds"])
    projected = 600.0 * mean_epoch
    total_proj = 1.15 * projected + startup + final_eval_seconds + 30.0
    out = {
        "protocol_id": "zinc-global-operator-search-v1",
        "route": route,
        "seed": int(seed),
        "recipe": recipe,
        "rows": "full official train 10000, full valid 1000",
        "epochs_measured": int(len(epoch_seconds)),
        "steps_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "epoch_seconds": epoch_seconds,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady,
        "startup_seconds": startup,
        "final_eval_seconds": final_eval_seconds,
        "projected_600ep_seconds_with_overheads": total_proj,
        "projected_600ep_gpu_hours": total_proj / 3600.0,
        "parameters": int(info["parameter_count"]),
        "parameter_audit": info["parameter_audit"],
        "gradient_audit": info["gradient_audit"],
        "official_test_loaded": False,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / f"timing_{route}.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing", "prep"])
    p.add_argument("--route", default="A", choices=["C0", "A", "K"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--recipe", default=RECIPE, choices=["A", "B", "C"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command == "prep":
        rel_train, rel_valid, scale, mean_nb = prepare()
        C.log(f"prep ok: train pairs={int(rel_train['ptr'][-1])} valid pairs={int(rel_valid['ptr'][-1])} "
              f"mean_n_bonds={mean_nb:.6f}")
        return 0
    device = torch.device(args.device)
    if args.command == "timing":
        timing(args.route, args.seed, device, epochs=args.epochs or 3, recipe=args.recipe)
    else:
        train(args.route, args.seed, device, recipe=args.recipe, epochs=args.epochs,
              tag=args.tag, force=args.force, lr=args.lr,
              weight_decay=args.weight_decay, dropout=args.dropout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
