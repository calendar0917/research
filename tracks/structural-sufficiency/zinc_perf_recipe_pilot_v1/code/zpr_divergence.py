"""ZPR-v1 C4 resolution: is the 600-epoch divergence from the historical LCRC-A
curve a *trainer* difference or CUDA execution nondeterminism?

    python zpr_divergence.py --device cuda:0 [--epochs 12]

Runs the **frozen** ``lc_train.train`` twice in the same process (identical code,
identical seed, identical data order, separate model objects) and compares each
replay against the historical ``runs/lcrcA_s0/curve.csv`` and against the L1
control arm ``G0``.  If the frozen trainer cannot reproduce itself either, the
600-epoch curve equality criterion is not achievable on this execution regime
and the achievable equivalence criterion must be stated instead.

Writes ``runs/divergence/divergence.json`` (pulled to ``analysis/divergence.json``).
No new scientific arm is trained: this is a replay/equivalence control only,
and it never touches test.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import zpr_common as Z
import zpr_train as T

L, D, M = Z.L, Z.D, Z.M
HIST = Z.LCRC_A_S0_CURVE


def _curve(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _diffs(curve_a: list[dict], curve_b: list[dict], n: int, key: str) -> list[float]:
    return [abs(float(curve_a[i][key]) - float(curve_b[i][key])) for i in range(n)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ZPR-v1 frozen-trainer replay divergence control")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args(argv)
    out_dir = Z.RUNS / "divergence"
    Z.ensure_dir(out_dir)

    hist = _curve(HIST)
    n = int(args.epochs)
    if len(hist) < n:
        raise RuntimeError("historical curve is shorter than the replay horizon")

    runs = {}
    for tag in ("frozen_replay_a", "frozen_replay_b"):
        t0 = time.perf_counter()
        payload = T.train("G0", 0, args.device, epochs=n, out_dir=out_dir / tag, force=True)
        runs[tag] = {"payload": payload, "curve": payload["curve"],
                     "wall_seconds": time.perf_counter() - t0}

    g0 = _curve(Z.RUNS / "G0_s0" / "curve.csv")

    def compare(a: str, b_curve: list[dict]) -> dict:
        dv = _diffs(runs[a]["curve"], b_curve, n, "valid_mae")
        do = _diffs(runs[a]["curve"], b_curve, n, "online_train_mae")
        return {"valid": dv, "online": do}

    cmp = {
        "frozen_a_vs_history": compare("frozen_replay_a", hist),
        "frozen_b_vs_history": compare("frozen_replay_b", hist),
        "frozen_a_vs_frozen_b": compare("frozen_replay_a", runs["frozen_replay_b"]["curve"]),
        "frozen_a_vs_G0": compare("frozen_replay_a", g0),
        "G0_vs_history": {
            "valid": _diffs(g0, hist, n, "valid_mae"),
            "online": _diffs(g0, hist, n, "online_train_mae"),
        },
    }

    def summarise(c: dict) -> dict:
        v, o = c["valid"], c["online"]
        first = next((i + 1 for i, d in enumerate(v) if d > 1e-5), None)
        return {
            "max_valid_diff": max(v),
            "max_valid_diff_epoch": int(np.argmax(v)) + 1,
            "mean_valid_diff": float(np.mean(v)),
            "median_valid_diff": float(np.median(v)),
            "max_online_diff": max(o),
            "first_epoch_valid_diff_gt_1e-5": first,
            "per_epoch_valid_diff": [float(x) for x in v],
        }

    summary = {k: summarise(v) for k, v in cmp.items()}
    same_basin_online = summary["frozen_a_vs_G0"]["max_online_diff"]
    payload = {
        "round": Z.ROUND_ID,
        "purpose": "C4 equivalence control: frozen trainer vs itself vs the historical curve",
        "device": args.device,
        "environment": Z.L.env_summary(),
        "git_commit": Z.L.C.git_commit(),
        "epochs": n,
        "historical_curve": Z.track_rel(HIST),
        "comparisons": summary,
        "frozen_replay_initial_state": {
            "a": runs["frozen_replay_a"]["payload"]["model_info"]["initial_state_sha256"],
            "b": runs["frozen_replay_b"]["payload"]["model_info"]["initial_state_sha256"],
            "G0": Z.read_json(Z.RUNS / "G0_s0" / "train.json")["model_info"]["initial_state_sha256"],
        },
        "first_batch_rows_equal": (
            runs["frozen_replay_a"]["payload"]["first_batch_rows"]
            == runs["frozen_replay_b"]["payload"]["first_batch_rows"]
            == Z.read_json(Z.RUNS / "G0_s0" / "train.json")["first_batch_rows"]),
        "frozen_replay_a_wall_seconds": runs["frozen_replay_a"]["wall_seconds"],
        "frozen_replay_b_wall_seconds": runs["frozen_replay_b"]["wall_seconds"],
        "conclusion_rule": {
            "frozen_trainer_is_deterministic": "max(frozen_a vs frozen_b) < 1e-5 over the horizon",
            "divergence_is_hardware_nondeterminism": "the frozen trainer's own two replays diverge "
                                                     "from each other and from history by the same "
                                                     "magnitude as the L1 control arm",
            "same_basin": "late-epoch online-train MAE agrees to ~1e-4 while the valid curve keeps a "
                          "persistent offset (chaotic amplification of a 1e-6 perturbation)",
        },
        "created": Z.now(),
    }
    # written under the gitignored runs/ tree so a remote run never collides
    # with the tracked analysis copy at deploy time
    Z.write_json(out_dir / "divergence.json", payload)
    for k, v in summary.items():
        Z.log(f"[divergence] {k}: max_valid={v['max_valid_diff']:.3e} "
              f"(ep {v['max_valid_diff_epoch']}), first>1e-5 at ep "
              f"{v['first_epoch_valid_diff_gt_1e-5']}, max_online={v['max_online_diff']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
