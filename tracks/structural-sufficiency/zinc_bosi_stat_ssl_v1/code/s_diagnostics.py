"""Descriptive diagnostics for the report (no part of the decision rule).

Everything here uses Fit8k and the internal Monitor split only; Locked is not
touched.  The values are explicitly labelled descriptive: the pre-registered
seed-0 Monitor gate and the frozen Locked rule already decided the round.

Questions answered:

1. How good is the pretrained initialisation *before any supervised step*
   (Monitor MAE of the SSL-initialised model vs the historical initial model)?
2. Does the SSL run fit Fit8k better or worse than the historical run at its
   selected checkpoint?
3. How do the two Monitor trajectories compare (minimum, tail median, epoch)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402
import s_pretext as SP  # noqa: E402
import s_train as ST  # noqa: E402
import ad_common as C  # noqa: E402


def _pred(model, cache, rel, rows, y):
    p = SC.BC.predict(model, cache, rel, rows, device="cpu",
                      batch_size=int(SC.BC.RECIPE["eval_batch_size"]))
    return SC.BC.mae(p, y)


def main(argv=None) -> int:
    seeds = [int(s) for s in (argv or sys.argv[1:])] or [0]
    cache = SC.load_train_cache()
    rel = SC.load_relation_cache()
    rows = SC.split_rows()
    y_mon = SC.labels(cache, rows["monitor"])
    y_fit = SC.labels(cache, rows["fit8k"])
    norm = SC.fit_only_norm(cache)
    out = {"kind": "descriptive", "part_of_decision": False,
           "uses": ["fit8k", "monitor"], "locked_used": False, "seeds": {}}

    for seed in seeds:
        entry = {}
        # 1. initial-point quality, zero supervised steps
        model, _ = SP.build_backbone(norm, seed, "cpu")
        entry["monitor_mae_at_historical_init"] = _pred(model, cache, rel,
                                                        rows["monitor"], y_mon)
        pre = json.loads((SC.RUNS / f"S_pre{seed}_full" / "pretrain.json").read_text())
        state = SC.RUNS / f"S_pre{seed}_full" / SC.pretrain_state_name(seed)
        if SC.sha256_file(state) != pre["state_sha256"]:
            raise SystemExit("pretraining state changed")
        model_ssl, _ = SP.build_backbone(norm, seed, "cpu")
        loaded = ST.load_ssl_export(model_ssl, seed, SC.RUNS / f"S_pre{seed}_full")
        entry["monitor_mae_at_ssl_init"] = _pred(model_ssl, cache, rel,
                                                 rows["monitor"], y_mon)
        entry["monitor_mae_at_ssl_init_delta"] = (entry["monitor_mae_at_ssl_init"]
                                                 - entry["monitor_mae_at_historical_init"])
        entry["monitor_mae_at_ssl_init_fit"] = _pred(model_ssl, cache, rel,
                                                     rows["fit8k"], y_fit)
        entry["ssl_init_modules"] = loaded["modules_changed"]

        # 2/3. trajectory comparison of the historical run and the SSL run
        hist = json.loads((SC.V0 / "runs" / f"B_s{seed}" / "train.json").read_text())
        ssl_path = SC.RUNS / f"S_s{seed}" / "train.json"
        entry["historical"] = _trajectory(hist)
        if ssl_path.exists():
            entry["ssl"] = _trajectory(json.loads(ssl_path.read_text()))
            entry["fit8k_mae_at_best_monitor"] = {}
            for tag, path in (("historical", SC.historical_b_checkpoint(seed, "best_monitor")),
                              ("ssl", SC.RUNS / f"S_s{seed}" / "best_monitor.pt")):
                m, _ = SP.build_backbone(norm, seed, "cpu")
                blob = torch.load(path, map_location="cpu", weights_only=False)
                C.load_state(m, blob["state_dict"], "cpu")
                entry["fit8k_mae_at_best_monitor"][tag] = _pred(m, cache, rel,
                                                                rows["fit8k"], y_fit)
        out["seeds"][str(seed)] = entry
        SC.log(f"[s-diag] seed {seed}: monitor at init hist "
               f"{entry['monitor_mae_at_historical_init']:.6f} -> ssl "
               f"{entry['monitor_mae_at_ssl_init']:.6f} "
               f"({entry['monitor_mae_at_ssl_init_delta']:+.6f})")
        if "fit8k_mae_at_best_monitor" in entry:
            f = entry["fit8k_mae_at_best_monitor"]
            SC.log(f"[s-diag] fit8k at best_monitor: historical {f['historical']:.6f} "
                   f"vs ssl {f['ssl']:.6f}")

    SC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    SC.write_json(SC.ANALYSIS / "diagnostics.json", out)
    return 0


def _trajectory(info: dict) -> dict:
    curve = info["curve"]
    mon = np.asarray([e["monitor_mae"] for e in curve], dtype=np.float64)
    online = np.asarray([e["online_train_mae"] for e in curve], dtype=np.float64)
    k = np.argmin(mon)
    return {
        "checkpoint_file": info.get("init", {}).get("init", "historical"),
        "epochs": int(len(curve)),
        "updates_run": int(info.get("updates_run", 0)),
        "best_monitor_mae": float(mon[k]),
        "best_monitor_epoch": int(curve[int(k)]["epoch"]),
        "best_monitor_step": int(curve[int(k)]["step"]),
        "tail100_median_monitor_mae": float(np.median(mon[-100:])),
        "tail100_median_online_train_mae": float(np.median(online[-100:])),
        "monitor_at_epoch_200": float(mon[199]) if len(mon) >= 200 else None,
        "monitor_at_epoch_400": float(mon[399]) if len(mon) >= 400 else None,
        "wall_seconds": float(info.get("wall_seconds", float("nan"))),
        "gpu_hours": float(info.get("gpu_hours", float("nan"))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
