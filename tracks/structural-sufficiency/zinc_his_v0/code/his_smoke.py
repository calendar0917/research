"""2-epoch GPU smoke + execution-equivalence gate for HIS-v0.

    python his_smoke.py --device cuda:0

Runs, in one process and on one GPU:

  1. the **frozen** LCRC-A trainer (``lc_train.train``) for 2 epochs,
  2. the new runner with ``lambda = 0`` (REPLAY),
  3. the new runner with the frozen ``lambda`` (HIS),

then writes ``runs/smoke/smoke.json`` with

  * frozen-vs-REPLAY equivalence (valid MAE, online MAE, lr, bitwise state_dict),
  * the HIS penalty being non-zero, finite, and confined to the affected
    parameters,
  * a wall-clock projection of the two 600-epoch arms.

Nothing here writes to a Git-tracked path, so the remote checkout stays clean.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import his_common as HIS
import his_train as HT

L, D, M, T = HIS.L, HIS.D, HIS.M, HIS.T

SMOKE_DIR = HIS.RUNS / "smoke"


def _state_dict(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _bitwise_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HIS-v0 2-epoch GPU smoke")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)
    L.ensure_dir(SMOKE_DIR)

    t0 = time.perf_counter()
    frozen = T.train("A", 0, args.device, epochs=args.epochs,
                     out_dir=HIS.RUNS / "smoke_frozen_s0", force=True)
    replay = HT.train("REPLAY", 0, args.device, epochs=args.epochs, force=True)
    his = HT.train("HIS", 0, args.device, epochs=args.epochs, force=True)

    eq = {
        "same_initial_state": (frozen["model_info"]["initial_state_sha256"]
                               == replay["model_info"]["initial_state_sha256"]
                               == his["model_info"]["initial_state_sha256"]),
        "same_first_batch_rows": (frozen["first_batch_rows"] == replay["first_batch_rows"]
                                  == his["first_batch_rows"]),
        "same_warmup_steps": frozen["warmup_steps"] == replay["warmup_steps"] == his["warmup_steps"],
        "same_updates": frozen["total_updates"] == replay["total_updates"] == his["total_updates"],
        "frozen_vs_replay_state_dict_bitwise": _bitwise_equal(
            _state_dict(HIS.RUNS / "smoke_frozen_s0" / "final.pt"),
            _state_dict(HIS.RUNS / "2ep_REPLAY_s0" / "final.pt")),
        "replay_vs_his_state_dict_differ": not _bitwise_equal(
            _state_dict(HIS.RUNS / "2ep_REPLAY_s0" / "final.pt"),
            _state_dict(HIS.RUNS / "2ep_HIS_s0" / "final.pt")),
        "same_parameters": (frozen["model_info"]["parameters"] == replay["model_info"]["parameters"]
                            == his["model_info"]["parameters"] == 237519),
    }
    epochs_cmp = []
    for i in range(args.epochs):
        a, b, c = frozen["curve"][i], replay["curve"][i], his["curve"][i]
        epochs_cmp.append({
            "epoch": i + 1,
            "frozen_valid": float(a["valid_mae"]),
            "replay_valid": float(b["valid_mae"]),
            "his_valid": float(c["valid_mae"]),
            "frozen_minus_replay_valid": float(a["valid_mae"] - b["valid_mae"]),
            "frozen_minus_replay_online": float(a["online_train_mae"] - b["online_train_mae"]),
            "lr_frozen": float(a["lr"]), "lr_replay": float(b["lr"]), "lr_his": float(c["lr"]),
            "h3_ratio_frozen": None, "h3_ratio_replay": float(b["h3_ratio"]),
            "h3_ratio_his": float(c["h3_ratio"]),
            "penalty_his": float(c["penalty"]),
            "seconds_frozen": float(a["seconds"]), "seconds_replay": float(b["seconds"]),
            "seconds_his": float(c["seconds"]),
        })
    worst_valid_gap = max(abs(e["frozen_minus_replay_valid"]) for e in epochs_cmp)
    worst_online_gap = max(abs(e["frozen_minus_replay_online"]) for e in epochs_cmp)
    eq["frozen_vs_replay_max_abs_valid_gap"] = float(worst_valid_gap)
    eq["frozen_vs_replay_max_abs_online_gap"] = float(worst_online_gap)
    eq["frozen_vs_replay_equal_within_1e-6"] = bool(worst_valid_gap < 1e-6
                                                    and worst_online_gap < 1e-6)
    eq["lr_trajectory_matches"] = all(e["lr_frozen"] == e["lr_replay"] == e["lr_his"]
                                      for e in epochs_cmp)

    regulariser = {
        "lambda": float(his["lambda"]),
        "lambda_check": his["lambda_check"],
        "penalty_first_step": his["penalty_first_step"],
        "penalty_last_step": his["penalty_last_step"],
        "penalty_min": his["penalty_min"],
        "penalty_max": his["penalty_max"],
        "penalty_finite_and_positive": bool(
            his["penalty_first_step"] is not None
            and np.isfinite(his["penalty_first_step"]) and his["penalty_first_step"] > 0),
        "penalty_decreased": bool(his["penalty_last_step"] is not None
                                  and his["penalty_last_step"] < his["penalty_first_step"]),
        "replay_has_no_penalty": his["lambda"] > 0 and replay["lambda"] == 0.0,
        "gradient_audit_his": his["gradient_audit_first_batch"],
        "gradient_audit_replay": replay["gradient_audit_first_batch"],
    }

    steady = float(np.mean([e["seconds_replay"] for e in epochs_cmp[1:]] or
                           [epochs_cmp[-1]["seconds_replay"]]))
    steady_his = float(np.mean([e["seconds_his"] for e in epochs_cmp[1:]] or
                               [epochs_cmp[-1]["seconds_his"]]))
    projection = {
        "replay_steady_seconds_per_epoch": steady,
        "his_steady_seconds_per_epoch": steady_his,
        "projected_gpu_hours_replay": (600 * steady * 1.15 + 60.0) / 3600.0,
        "projected_gpu_hours_his": (600 * steady_his * 1.15 + 60.0) / 3600.0,
        "projected_parallel_wall_hours": max((600 * steady * 1.15 + 60.0) / 3600.0,
                                             (600 * steady_his * 1.15 + 60.0) / 3600.0),
        "budget_hard_gpu_hours": 2.0,
        "smoke_total_seconds": float(time.perf_counter() - t0),
    }
    projection["projected_total_gpu_hours"] = (projection["projected_gpu_hours_replay"]
                                               + projection["projected_gpu_hours_his"])
    projection["within_budget"] = bool(projection["projected_total_gpu_hours"] <= 2.0)

    out = {
        "round": HIS.ROUND_ID,
        "created": HIS.now(),
        "git_commit": L.C.git_commit(),
        "device": args.device,
        "environment": L.env_summary(),
        "epochs": int(args.epochs),
        "equivalence": eq,
        "per_epoch": epochs_cmp,
        "regulariser": regulariser,
        "projection": projection,
        "official_test_loaded": False,
    }
    HIS.write_json(SMOKE_DIR / "smoke.json", out)

    print("\n== HIS-v0 GPU smoke ==")
    for k, v in eq.items():
        print(f"  {k}: {v}")
    print(f"  penalty first/last: {regulariser['penalty_first_step']} -> "
          f"{regulariser['penalty_last_step']}")
    print(f"  h3 ratio (replay, his) after {args.epochs} ep: "
          f"{epochs_cmp[-1]['h3_ratio_replay']:.4f} / {epochs_cmp[-1]['h3_ratio_his']:.4f}")
    print(f"  projected total: {projection['projected_total_gpu_hours']:.3f} GPU-h "
          f"(budget 2.0) within={projection['within_budget']}")
    ok = (eq["same_initial_state"] and eq["same_first_batch_rows"]
          and eq["frozen_vs_replay_equal_within_1e-6"]
          and eq["frozen_vs_replay_state_dict_bitwise"] and eq["lr_trajectory_matches"]
          and regulariser["penalty_finite_and_positive"] and regulariser["penalty_decreased"]
          and projection["within_budget"])
    print(f"  SMOKE PASS: {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
