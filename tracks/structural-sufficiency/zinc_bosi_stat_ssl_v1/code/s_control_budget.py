"""Freeze the compute-matched control budget of arm C.

The control gets the *same GPU time* that the SSL initialisation consumed,
expressed in supervised updates of this seed:

    N_s = round( pretraining wall seconds of seed s
                 / measured supervised seconds per update of seed s )

The denominator is measured on the BOSI-SSL run of the same seed (same regime,
same recipe, same device) and the numerator on the pretraining job, so both
sides are wall-clock seconds on the A100 that actually ran them.  The rule is
mechanical: it cannot be tuned after looking at Monitor or Locked.

Writes ``analysis/control_budget.json``.  Reads no labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402


def main(argv=None) -> int:
    seeds = [int(s) for s in (argv or sys.argv[1:])] or [0, 1]
    out = {"rule": "N_s = round(pretrain_wall_seconds_s / seconds_per_update_s)",
           "measured_on": "the A100 that ran the job (same execution regime)",
           "note": ("the control starts from the historical BOSI-B best_monitor "
                    "checkpoint that the previous rounds deploy, and continues the "
                    "frozen batch-order stream at the terminal learning rate"),
           "seeds": {}}
    for seed in seeds:
        pre_path = SC.RUNS / f"S_pre{seed}_full" / "pretrain.json"
        ssl_path = SC.RUNS / f"S_s{seed}" / "train.json"
        if not pre_path.exists() or not ssl_path.exists():
            raise SystemExit(f"missing artefacts for seed {seed}: {pre_path} / {ssl_path}")
        pre = json.loads(pre_path.read_text())
        ssl = json.loads(ssl_path.read_text())
        if SC.sha256_file(SC.RUNS / f"S_pre{seed}_full" / SC.pretrain_state_name(seed)) \
                != pre["state_sha256"]:
            raise SystemExit("pretraining state changed")
        sec_per_update = float(ssl["wall_seconds"]) / max(int(ssl["updates_run"]), 1)
        n_extra = int(round(float(pre["wall_seconds"]) / sec_per_update))
        out["seeds"][str(seed)] = {
            "pretrain_wall_seconds": float(pre["wall_seconds"]),
            "pretrain_gpu_hours": float(pre["gpu_hours"]),
            "pretrain_updates": int(pre["updates"]),
            "ssl_supervised_wall_seconds": float(ssl["wall_seconds"]),
            "ssl_supervised_updates": int(ssl["updates_run"]),
            "seconds_per_update": sec_per_update,
            "extra_updates": n_extra,
            "projected_control_gpu_hours": float(n_extra * sec_per_update / 3600.0),
            "control_init": "historical BOSI-B best_monitor (hash-verified)",
        }
        SC.log(f"[s-budget] seed {seed}: pretrain {pre['wall_seconds']:.0f}s / "
               f"{sec_per_update * 1000:.1f}ms per update -> {n_extra} extra updates "
               f"({out['seeds'][str(seed)]['projected_control_gpu_hours']:.3f} GPU-h)")
    total = sum(v["projected_control_gpu_hours"] for v in out["seeds"].values())
    out["projected_control_gpu_hours_total"] = float(total)
    out["frozen_utc"] = SC.now()
    SC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    SC.write_json(SC.ANALYSIS / "control_budget.json", out)
    SC.log(f"[s-budget] wrote control_budget.json, projected {total:.3f} A100-h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
