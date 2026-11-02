"""Sharded sweep driver for the frozen run matrix.

One ``rr`` job runs a contiguous shard of pre-registered cells; every cell is a
fresh child process (clean CUDA context, per-run stdout) and writes the same
artefacts as a manual ``dissect_train.py`` call.  A cell whose ``train.json``
already exists with ``status == "completed"`` is skipped, so a shard can be
resumed after a crash without re-buying compute.

Usage (always submitted through ``rr run``):

    rr run res <exp> --pool res-gpu0 --gpus 1 --result <track>/runs -- \
        uv run python <track>/code/dissect_sweep.py --stage A --shard 0 --shards 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
sys.path.insert(0, str(HERE))

import dissect_common as C  # noqa: E402
import dissect_train as T  # noqa: E402


def cells_for(stage: str, matrix: dict) -> list[tuple[int, str, int]]:
    if stage == "cross":
        rule = matrix["cross_split_rule"]
        variants = list(rule["variants"])
        compact = rule["compact_variant"]
        if compact is None:
            raise SystemExit("cross stage needs a committed compact_selection.json")
        if compact not in variants:
            variants.append(compact)
        seeds = list(C.SEEDS)
        out = [(r, v, s) for r in rule["repeats"] for v in variants for s in seeds]
        # identical (repeat, variant, seed) cells are trained once
        return sorted(set(out))
    return [(C.MAIN_REPEAT, v, s) for v in C.VARIANT_STAGE[stage] for s in C.SEEDS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C", "cross"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit("bad shard")

    matrix = C.read_json(C.CONFIGS / "run_matrix.json")
    cells = cells_for(args.stage, matrix)
    for repeat, variant, seed in cells:
        ok, why = T.matrix_allows(repeat, variant)
        if not ok:
            raise SystemExit(f"frozen matrix violation: {(repeat, variant, seed)}: {why}")
    mine = [c for i, c in enumerate(cells) if i % args.shards == args.shard]
    C.log(f"[sweep {args.stage} shard {args.shard}/{args.shards}] {len(mine)}/{len(cells)} cells")

    started = time.time()
    results: list[dict] = []
    failures = 0
    for i, (repeat, variant, seed) in enumerate(mine, 1):
        run_dir = TRACK / "runs" / C.run_name(repeat, variant, seed)
        record = run_dir / "train.json"
        if record.exists() and not args.force:
            try:
                state = json.loads(record.read_text())
            except json.JSONDecodeError:
                state = {}
            if state.get("status") == "completed" and state.get("device", "").startswith("cuda"):
                C.log(f"[sweep] {i}/{len(mine)} skip completed {run_dir.name}")
                results.append({"run": run_dir.name, "skipped": True})
                continue
        t0 = time.time()
        cmd = [
            sys.executable,
            str(HERE / "dissect_train.py"),
            "--variant", variant,
            "--seed", str(seed),
            "--repeat", str(repeat),
            "--device", args.device,
        ]
        C.log(f"[sweep] {i}/{len(mine)} start {run_dir.name} :: {' '.join(cmd[2:])}")
        proc = subprocess.run(cmd, cwd=str(C.REPO))
        wall = time.time() - t0
        results.append({"run": run_dir.name, "returncode": proc.returncode, "wall_seconds": wall})
        if proc.returncode != 0:
            failures += 1
            C.log(f"[sweep] FAILED {run_dir.name} rc={proc.returncode} after {wall:.1f}s")

    out_dir = TRACK / "runs" / "_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": args.stage,
        "shard": args.shard,
        "shards": args.shards,
        "device": args.device,
        "n_cells": len(mine),
        "n_failures": failures,
        "wall_seconds": time.time() - started,
        "results": results,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(C.REPO), capture_output=True, text=True
        ).stdout.strip(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / f"{args.stage}_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    C.log(f"[sweep] done failures={failures} wall={summary['wall_seconds']:.0f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
