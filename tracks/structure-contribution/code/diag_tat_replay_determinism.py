"""Diagnostic: is the frozen supervised loop bitwise-reproducible on this machine?

Runs the frozen ``bp.train_supervised`` loop twice and the line-equivalent
``tat.train_task_adaptive(aux_every=None)`` loop twice on the same frozen
split-D Fit rows / objects / y, then compares the last-5 soups bitwise.

This is an environment probe, not a research artifact.  It writes no files and
touches no labels beyond the Fit raw y already used by training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_task_adaptive_ssl_v1 as tat  # noqa: E402
import run_bjoc_task_adaptive_ssl_v1 as drv  # noqa: E402


def _max_abs_diff(a: dict, b: dict) -> float:
    worst = 0.0
    for k in a:
        worst = max(worst, float((a[k] - b[k]).abs().max()) if a[k].numel() else 0.0)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    out = drv.OUT
    split = json.loads((out / "split_D.json").read_text(encoding="utf-8"))
    rows = [int(i) for i in split["fit_rows"][: args.n_rows]]
    objs = torch.load(out / "objects_fit_D.pt", map_location="cpu", weights_only=False)[: args.n_rows]
    norm = json.loads((out / "prep_fit_D.json").read_text(encoding="utf-8"))
    y = bp.load_y_rows(drv.ENCODED_TRAIN, rows)
    device = torch.device("cpu")

    soups: dict[str, dict] = {}
    for tag, fn in (
        ("bp", lambda arm: bp.train_supervised(
            arm=arm, seed=0, recipe_key="R3", init="scratch", train_objs=objs, rows=rows,
            y_fit=y, norm=norm, device=device, smoke_epochs=args.epochs,
        )),
        ("tat", lambda arm: tat.train_task_adaptive(
            arm=arm, seed=0, recipe_key="R3", init="scratch", train_objs=objs, rows=rows,
            y_fit=y, norm=norm, device=device, aux_every=None, smoke_epochs=args.epochs,
        )),
    ):
        for rep in (1, 2):
            payload = fn(f"{tag}-rep{rep}")
            soups[f"{tag}{rep}"] = payload["soup_state"]

    names = ["bp1", "bp2", "tat1", "tat2"]
    hashes = {n: tat.state_sha256(soups[n]) for n in names}
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a}_vs_{b}"] = {
                "same_sha256": hashes[a] == hashes[b],
                "max_abs_diff": _max_abs_diff(soups[a], soups[b]),
            }
    print(json.dumps({"hashes": hashes, "pairs": pairs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
