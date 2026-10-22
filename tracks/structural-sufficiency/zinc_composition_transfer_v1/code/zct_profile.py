"""Measured per-epoch cost of the E / R / S loops, before the formal schedule is frozen.

Times two full epochs of each arm through the real trainer (same collate, model,
optimiser and evaluation code), measures the fixed overheads (valid evaluation,
full-train eval-mode evaluation, checkpoint save), and projects 600-epoch costs
with the declared 15% margin.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code"),
           str(_SSUF / "zinc_recipe_trunk_v2" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import zct_train as ZT  # noqa: E402

ANALYSIS = _TASK / "analysis"
RUNS = _TASK / "runs"


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--kappa", type=float, default=1.0)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    out_dir = RUNS / "_profile"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    result = {"device": str(device), "epochs_per_arm": args.epochs, "kappa_profiled": args.kappa,
              "environment": C.environment()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- E arm with the shared prefix fork written after epoch 1 -------------
    tr = ZT.Trainer("E", 0, device, out_dir / "E", kappa=0.0)
    _, t_e_build = timed(ZT.build_model, D.load_norm(), 0, device)
    t0 = time.perf_counter()
    tr.run(fork_out_epoch=1, epochs=args.epochs)
    wall_e = time.perf_counter() - t0
    ep_e = [e["seconds"] for e in tr.curve]
    result["E"] = {"epoch_seconds": ep_e, "mean_epoch_seconds": float(np.mean(ep_e)),
                   "model_build_seconds": t_e_build, "wall_seconds": wall_e,
                   "parameters": int(tr.model_info["parameters"])}

    # ---- fixed overhead measurements ---------------------------------------
    model = tr.model
    _, t_valid = timed(T1.evaluate, model, tr.valid_cache, tr.rows_valid, candidate="P3N",
                       device=device, batch_size=256)
    _, t_train_eval = timed(T1.evaluate, model, tr.train_cache, tr.rows_train, candidate="P3N",
                            device=device, batch_size=256)
    state = C.cpu_state(model)
    _, t_save = timed(C.atomic_torch_save, {"state_dict": state}, out_dir / "save_probe.pt")
    result["overheads"] = {"valid_eval_1000_seconds": t_valid,
                           "train_eval_10000_seconds": t_train_eval,
                           "checkpoint_save_seconds": t_save}

    fork = out_dir / "E" / "fork.pt"

    # ---- R and S from the same fork ---------------------------------------
    part = ZT.SP.load_partitions(0)
    payload = {
        "r_packed": part["r_packed"], "s_packed": part["s_packed"],
        "batch_rows": part["batch_rows"], "swap_off": part["swap_off"],
        "swap_i": part["swap_i"], "swap_j": part["swap_j"],
        "epoch_orders": part["summary"]["epoch_order_sha256"],
    }
    for arm in ("R", "S"):
        tr2 = ZT.Trainer(arm, 0, device, out_dir / arm, kappa=args.kappa, partition=payload)
        t0 = time.perf_counter()
        tr2.run(fork_in=fork, epochs=args.epochs + 1)
        wall = time.perf_counter() - t0
        ep = [e["seconds"] for e in tr2.curve if e["epoch"] > 1]
        result[arm] = {"epoch_seconds": ep, "mean_epoch_seconds": float(np.mean(ep)),
                       "wall_seconds": wall, "gamma_vs_E": float(np.mean(ep) / np.mean(ep_e))}

    # ---- projections -------------------------------------------------------
    N_EPOCHS = 600
    e_ep = result["E"]["mean_epoch_seconds"]
    proj = {
        "E_600ep_hours_incl_valid": e_ep * N_EPOCHS / 3600.0,
        "R_600ep_hours_incl_valid": result["R"]["mean_epoch_seconds"] * N_EPOCHS / 3600.0,
        "S_600ep_hours_incl_valid": result["S"]["mean_epoch_seconds"] * N_EPOCHS / 3600.0,
        "prefix_50ep_hours": e_ep * 50 / 3600.0,
        "eval_train_100_epochs_hours": t_train_eval * 100 / 3600.0,
        "save_60_checkpoints_hours": t_save * 60 / 3600.0,
    }
    n_sup_rows = 7549
    proj["support_run_600ep_hours"] = (e_ep * n_sup_rows / 10000.0) * N_EPOCHS / 3600.0
    proj["margin_factor"] = 1.15
    proj["with_margin"] = {k: (v * 1.15) for k, v in proj.items() if isinstance(v, float)
                           and k.endswith("hours")}
    core = (proj["E_600ep_hours_incl_valid"] + 2 * proj["R_600ep_hours_incl_valid"])
    plan = {
        "seed0_E_plus_R_plus_S_hours": core,
        "seed1_E_plus_R_plus_S_hours": core,
        "support_4_runs_hours": 4 * proj["support_run_600ep_hours"],
        "diagnostics_4_epochs_R_and_S_hours": (t_train_eval * 0 + 0.02),
    }
    plan["declared_total_hours"] = core * 2 + plan["support_4_runs_hours"] + 0.35
    plan["with_15pct_margin"] = plan["declared_total_hours"] * 1.15
    plan["cap_hours"] = 8.0
    plan["fits_cap"] = bool(plan["with_15pct_margin"] < 8.0)
    result["projection"] = proj
    result["plan"] = plan
    if device.type == "cuda":
        result["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
    C.write_json(ANALYSIS / "profiling.json", result)
    C.log(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
