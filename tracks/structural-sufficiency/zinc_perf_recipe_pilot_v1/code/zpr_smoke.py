"""ZPR-v1 2-epoch GPU smoke + execution-equivalence gate.

    python zpr_smoke.py --device cuda:0

Runs, in one process on one GPU:

  1. the **frozen** LCRC-A trainer (``lc_train.train``, 2 epochs) as reference,
  2. ``G0`` (L1 + passive EMA, 2 epochs),
  3. ``G1`` (SmoothL1(beta=0.05) + passive EMA, 2 epochs),

then writes ``analysis/smoke.json`` with

  * frozen-vs-G0 equivalence (per-epoch valid/online MAE, parameter drift),
  * G1 actually diverging from G0 (the loss switch is live),
  * the SmoothL1 first-batch loss matching its independent reference,
  * the EMA branch being initialised at ``ceil(0.25 * updates)``, evaluated,
    and *not* equal to either the initialisation or the online model,
  * a wall-clock projection of the two 600-epoch arms against the 1.5 A100-h cap.

Nothing here writes to a Git-tracked path except ``analysis/smoke.json``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import zpr_common as Z
import zpr_train as T

L, D, M = Z.L, Z.D, Z.M
SMOKE_DIR = Z.RUNS / "smoke"


def _state(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _max_drift(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).abs().max()) for k in a if a[k].is_floating_point())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ZPR-v1 2-epoch GPU smoke")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)
    Z.ensure_dir(SMOKE_DIR)

    t0 = time.perf_counter()
    frozen = Z.T.train("A", 0, args.device, epochs=args.epochs,
                       out_dir=SMOKE_DIR / "frozen_s0", force=True)
    t_frozen = time.perf_counter() - t0
    g0 = T.train("G0", 0, args.device, epochs=args.epochs, out_dir=SMOKE_DIR / "2ep_G0_s0", force=True)
    g1 = T.train("G1", 0, args.device, epochs=args.epochs, out_dir=SMOKE_DIR / "2ep_G1_s0", force=True)
    wall = time.perf_counter() - t0

    frozen_curve = frozen["curve"]
    g0_curve = g0["curve"]
    g1_curve = g1["curve"]
    epochs_cmp = []
    for i in range(args.epochs):
        a, b, c = frozen_curve[i], g0_curve[i], g1_curve[i]
        epochs_cmp.append({
            "epoch": i + 1,
            "frozen_valid": float(a["valid_mae"]),
            "g0_valid": float(b["valid_mae"]),
            "g1_valid": float(c["valid_mae"]),
            "abs_diff_valid_frozen_vs_g0": abs(float(a["valid_mae"]) - float(b["valid_mae"])),
            "abs_diff_online_frozen_vs_g0": abs(float(a["online_train_mae"]) - float(b["online_train_mae"])),
            "g0_ema_valid": b["ema_valid_mae"],
            "g1_ema_valid": c["ema_valid_mae"],
            "lr_frozen": float(a["lr"]),
            "lr_g0": float(b["lr"]),
            "lr_g1": float(c["lr"]),
        })
    state_frozen = _state(SMOKE_DIR / "frozen_s0" / "final.pt")
    state_g0 = _state(SMOKE_DIR / "2ep_G0_s0" / "final.pt")
    state_g1 = _state(SMOKE_DIR / "2ep_G1_s0" / "final.pt")
    drift_frozen_g0 = _max_drift(state_frozen, state_g0)
    drift_g0_g1 = _max_drift(state_g0, state_g1)

    ema0 = g0["ema"]["audit"]
    ema1 = g1["ema"]["audit"]
    ema_first = {arm: next((c["ema_valid_mae"] for c in run["curve"]
                            if c["ema_valid_mae"] is not None), None)
                 for arm, run in (("G0", g0), ("G1", g1))}
    online_first = {arm: run["curve"][0]["valid_mae"] for arm, run in (("G0", g0), ("G1", g1))}
    ema_or_feature = {
        "start_step_expected": int(T.ema_start_step(79 * args.epochs)),
        "start_step": {"G0": g0["ema"]["start_step"], "G1": g1["ema"]["start_step"]},
        "init_epoch": {"G0": ema0.get("init_epoch"), "G1": ema1.get("init_epoch")},
        "init_equals_online_params": {
            "G0": ema0.get("init_equals_online_params"),
            "G1": ema1.get("init_equals_online_params"),
        },
        "rng_invariance": {"G0": ema0.get("rng_invariance"), "G1": ema1.get("rng_invariance")},
        "first_ema_valid": ema_first,
        "first_online_valid": online_first,
        "ema_differs_from_online": {
            arm: (ema_first[arm] is not None
                  and abs(float(ema_first[arm]) - float(online_first[arm])) > 1e-9)
            for arm in ("G0", "G1")
        },
        "ema_best": {"G0": g0["ema"]["best"], "G1": g1["ema"]["best"]},
        "ema_evaluated_from_epoch": {
            arm: next((c["epoch"] for c in run["curve"] if c["ema_valid_mae"] is not None), None)
            for arm, run in (("G0", g0), ("G1", g1))
        },
    }
    loss_ref = {"G0": g0["loss_reference_first_batch"], "G1": g1["loss_reference_first_batch"]}

    # cost projection: steady epoch time (2nd epoch) x 600 x 2 arms, + 10% overhead
    steady = {}
    for arm, run in (("G0", g0), ("G1", g1)):
        secs = [c["seconds"] for c in run["curve"]]
        steady[arm] = float(np.mean(secs[1:])) if len(secs) > 1 else float(secs[0])
    projected = sum(600.0 * s for s in steady.values()) * 1.10 / 3600.0 + 0.02
    cost = {
        "steady_seconds_per_epoch": steady,
        "projected_two_arm_gpu_hours": float(projected),
        "hard_cap_gpu_hours": 1.5,
        "smoke_wall_seconds": float(wall),
        "smoke_gpu_hours": float(wall / 3600.0),
        "frozen_reference_wall_seconds": float(t_frozen),
        "within_cap": bool(projected <= 1.5),
        "note": "both arms run concurrently on separate GPUs, so the wall clock is "
                "max(arm) not sum(arm); the sum is the conservative GPU-hour projection",
    }

    gates = {
        "same_initial_state": (
            frozen["model_info"]["initial_state_sha256"]
            == g0["model_info"]["initial_state_sha256"]
            == g1["model_info"]["initial_state_sha256"]
            == Z.FROZEN["lcrc_a_s0_initial_state_sha256"]),
        "same_first_batch_rows": frozen["first_batch_rows"] == g0["first_batch_rows"]
                                 == g1["first_batch_rows"],
        "same_warmup_steps": frozen["warmup_steps"] == g0["warmup_steps"] == g1["warmup_steps"],
        "same_total_updates": frozen["total_updates"] == g0["total_updates"] == g1["total_updates"],
        "lr_sequences_equal": [round(e["lr_frozen"], 12) for e in epochs_cmp]
                              == [round(e["lr_g0"], 12) for e in epochs_cmp]
                              == [round(e["lr_g1"], 12) for e in epochs_cmp],
        "frozen_vs_g0_valid_within_tolerance": max(e["abs_diff_valid_frozen_vs_g0"]
                                                  for e in epochs_cmp) < 1e-5,
        "frozen_vs_g0_online_within_tolerance": max(e["abs_diff_online_frozen_vs_g0"]
                                                    for e in epochs_cmp) < 1e-5,
        "frozen_vs_g0_param_drift_within_tolerance": drift_frozen_g0 < 1e-3,
        "g1_differs_from_g0": drift_g0_g1 > 0.0,
        "smooth_l1_loss_matches_reference": (
            loss_ref["G1"]["abs_diff"] < 1e-6 and loss_ref["G0"]["abs_diff"] < 1e-6),
        "ema_start_step_correct": (
            g0["ema"]["start_step"] == g1["ema"]["start_step"] == ema_or_feature["start_step_expected"]),
        "ema_init_equals_online": ema_or_feature["init_equals_online_params"] == {"G0": True, "G1": True},
        "ema_evaluated_and_distinct": all(ema_or_feature["ema_differs_from_online"].values()),
        "ema_rng_invariance": all(
            v.get("torch_rng_state_unchanged") and v.get("data_order_gen_state_unchanged")
            for v in ema_or_feature["rng_invariance"].values()),
        "parameters_237519": (frozen["model_info"]["parameters"] == g0["model_info"]["parameters"]
                              == g1["model_info"]["parameters"] == Z.FROZEN["params"]),
        "cost_within_cap": cost["within_cap"],
        "official_test_never_loaded": (frozen["data"]["official_test_loaded"] is False
                                       and g0["data"]["official_test_loaded"] is False
                                       and g1["data"]["official_test_loaded"] is False),
    }
    payload = {
        "round": Z.ROUND_ID,
        "device": args.device,
        "environment": Z.L.env_summary(),
        "git_commit": Z.L.C.git_commit(),
        "epochs": int(args.epochs),
        "epochs_comparison": epochs_cmp,
        "param_drift": {"frozen_vs_g0": drift_frozen_g0, "g0_vs_g1": drift_g0_g1},
        "loss_reference": loss_ref,
        "ema": ema_or_feature,
        "cost": cost,
        "gates": gates,
        "all_pass": bool(all(gates.values())),
        "created": Z.now(),
    }
    Z.write_json(Z.ANALYSIS / "smoke.json", payload)
    Z.log(f"[zpr smoke] gates={gates}")
    Z.log(f"[zpr smoke] projected two-arm cost {projected:.3f} A100-h")
    return 0 if payload["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
