"""Trainer / scorer for zinc-statistical-program-search-v1.

Reuses the frozen architecture-discovery training loop, data cache, split,
normalisation source and best-valid checkpoint rule.  Only the model body, the
batch view and the declared horizon differ.

Commands
--------
prep                build the static Q record view + Fit-only scaling constants
timing              short timed run(s) to project the full-horizon cost
train               one candidate/seed for the declared 600-epoch cosine
score               re-evaluate a saved checkpoint and export valid predictions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_AD = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_AD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T  # noqa: E402
import sps_data as SD  # noqa: E402
import sps_models as M  # noqa: E402

ROUND = "zinc-statistical-program-search-v1"
SPS = _HERE.parent
RUNS = SPS / "runs"

#: Declared recipe: the mature DCC-R0 recipe, unchanged.
RECIPE = dict(
    optimizer="adamw",
    lr=1.0e-3,
    min_lr=1.0e-5,
    weight_decay=1.0e-5,
    warmup_steps=500,
    epochs=600,
    dropout=0.0,
    grad_clip=5.0,
    batch_size=128,
    valid_batch_size=256,
)

SSL = {"0": C.P3_SSL_S0, "1": C.P3_SSL_S1}


def _ssl_path(seed: int) -> Path:
    return SSL["0"] if int(seed) == 0 else SSL["1"]


def _builder(candidate: str, seed: int, dropout: float = 0.0):
    fit = SD.load_fit()

    def build(cand, norm, s, device):
        model, info = M.build_model(cand, fit, seed, device,
                                    ssl_path=_ssl_path(seed), dropout=dropout)
        info["ssl_sha256"] = C.sha256_file(_ssl_path(seed))
        info["fit_stats_source"] = "analysis/fit_stats.json (train structures only)"
        return model, info

    return build


def run_dir(candidate: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{candidate}_s{int(seed)}"


def train(candidate: str, seed: int, device, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False,
          recipe: dict | None = None, out: Path | None = None,
          subset: int | None = None) -> dict:
    rec = dict(RECIPE)
    if recipe:
        rec.update(recipe)
    out_dir = run_dir(candidate, seed, tag) if out is None else Path(out)
    collate = SD.collate_factory(candidate, "train")
    info = T.train(
        candidate, seed, device,
        epochs=None if epochs is None else int(epochs),
        tag=tag, force=force,
        model_builder=_builder(candidate, seed, float(rec["dropout"])),
        collate_fn=collate,
        recipe_override=rec,
        out_dir=out_dir,
        train_rows=None if subset is None else __import__("numpy").arange(int(subset)),
    )
    # The reused trainer stamps the old round id; record the actual round.
    if info.get("round") != ROUND:
        info["round"] = ROUND
        C.write_json(out_dir / "train.json", info)
    return info


@torch.no_grad()
def score(candidate: str, run: Path, *, checkpoint: str = "best.pt",
          device: str = "cpu", split: str = "valid") -> dict:
    """Evaluate a saved checkpoint from scratch (survives a cancelled run)."""
    run = Path(run)
    ckpt = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
    seed = int(ckpt.get("seed", 0))
    model, info = M.build_model(candidate, SD.load_fit(), seed, device,
                                ssl_path=_ssl_path(seed), dropout=0.0)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    ad = D.load_cache(split)
    collate = SD.collate_factory(candidate, split)
    rows = np.arange(int(ad["n_molecules"]), dtype=np.int64)
    preds = []
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        batch = T._to_device(collate(ad, chunk, candidate, valid=True), device)
        preds.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds)
    y = np.asarray(ad["labels"], dtype=np.float64)
    mae = C.mae(pred, y)
    out = {
        "round": ROUND,
        "candidate": candidate,
        "run": str(run),
        "checkpoint": checkpoint,
        "epoch": int(ckpt.get("epoch", -1)),
        "stored_valid_mae": float(ckpt.get("valid_mae", float("nan"))),
        "recomputed_mae": float(mae),
        "split": split,
        "n": int(len(y)),
        "parameters": int(info["parameters"]),
        "official_test_loaded": False,
    }
    pd = run / "predictions"
    pd.mkdir(exist_ok=True)
    np.savez(pd / f"{split}_{Path(checkpoint).stem}.npz", row=rows, pred=pred, y=y)
    C.write_json(pd / f"{split}_{Path(checkpoint).stem}.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "timing", "train", "score"])
    p.add_argument("--candidate", default="QF")
    p.add_argument("--candidates", default="QF,FC")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--split", default="valid", choices=["valid"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--subset", type=int, default=None,
                   help="smoke only: train on the first N molecules")
    args = p.parse_args(argv)
    device = torch.device(args.device)

    if args.command == "prep":
        t0 = time.perf_counter()
        fit = SD.build_all(force=args.force)
        C.log(f"[prep] {json.dumps({'fit_mean_atoms': fit['fit_mean_atoms'], 'fit_mean_bonds': fit['fit_mean_bonds'], 'group_count': fit['fit_group_count'], 'q_view': fit['q_view']}, indent=2)}")
        C.log(f"[prep] done {time.perf_counter() - t0:.1f}s")
        return 0

    if args.command == "timing":
        out = {}
        for cand in [x for x in args.candidates.split(",") if x]:
            t0 = time.perf_counter()
            info = train(cand, args.seed, device, epochs=args.epochs or 2, tag="timing",
                         force=True, subset=args.subset)
            per_epoch = float(np.mean([e["seconds"] for e in info["curve"]]))
            out[cand] = {
                "wall_seconds_2ep": float(time.perf_counter() - t0),
                "mean_epoch_seconds": per_epoch,
                "projected_gpu_hours_600": per_epoch * RECIPE["epochs"] / 3600.0,
                "parameters": info["parameter_count"],
                "gradient_audit": info["gradient_audit"],
                "best_valid_two_epoch": info["best_valid"]["valid_mae"],
                "updates_per_epoch": info["recipe"]["steps_per_epoch"],
            }
            C.log(f"[timing] {cand}: {out[cand]}")
        C.write_json(SPS / "analysis" / "timing.json", out)
        return 0

    if args.command == "train":
        info = train(args.candidate, args.seed, device, epochs=args.epochs,
                     tag=args.tag, force=args.force, subset=args.subset)
        C.log(json.dumps({"best_valid": info["best_valid"], "parameters": info["parameter_count"]}, indent=2))
        return 0

    if args.command == "score":
        if args.run is None:
            raise SystemExit("--run is required for score")
        score(args.candidate, Path(args.run), checkpoint=args.checkpoint,
              device=args.device, split=args.split)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
