"""Freeze every BOSI checkpoint / configuration / Monitor prediction hash.

Must run before ``bosi_locked.py``.  Writing is refused if a Locked read already
exists, so the order (freeze -> single Locked read) cannot be reversed silently.

Usage:
    python .../bosi_freeze.py --runs R_s0 B_s0 [R_s1 B_s1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args(argv)
    import json

    if (BC.ANALYSIS / "locked_read.json").exists():
        raise SystemExit("refusing to freeze after a Locked read already exists")

    cache = BC.load_train_cache()
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    rows = BC.split_rows()
    y_monitor = BC.labels(cache, rows["monitor"])
    norm = BC.fit_only_norm(cache, rows["fit8k"])
    results = {"frozen_utc": BC.now(), "round": BC.ROUND_ID,
               "config_sha256": BC.sha256_file(BC.CONFIGS / "bosi_v0.json"),
               "protocol_sha256": BC.sha256_file(BC.ROUND / "PROTOCOL.md"),
               "design_sha256": BC.sha256_file(BC.ROUND / "DESIGN.md"),
               "data_cache_sha256": BC.sha256_file(BC.DATA_CACHE),
               "relation_cache_sha256": BD.relation_content_hash(rel),
               "split_hashes": rows["hashes"],
               "runs": {}}
    for tag in args.runs:
        out = BC.RUNS / tag
        info = json.loads((out / "train.json").read_text())
        entry = {
            "run": tag,
            "arm": info["arm"],
            "seed": info["seed"],
            "parameters": info["parameters"],
            "initial_state_sha256": info["model_info"]["initial_state_sha256"],
            "ssl_sha256": info["model_info"]["ssl"]["sha256"],
            "first_batch_rows": info["first_batch_rows"],
            "grad_norms": info["grad_norms"],
            "wall_seconds": info["wall_seconds"],
            "gpu_hours": info["gpu_hours"],
            "epochs_run": info["epochs_run"],
            "best_monitor": info["best_monitor"],
            "checkpoints": {},
        }
        for ckpt, state_tag in (("best_monitor.pt", "best_monitor"), ("final.pt", "final")):
            path = out / ckpt
            blob = torch.load(path, map_location="cpu", weights_only=False)
            model, _ = BM.build_model(norm, "R", info["seed"], "cpu")
            model.arm = info["arm"]
            C.load_state(model, blob["state_dict"], "cpu")
            pred = BC.predict(model, cache, rel, rows["monitor"], device="cpu",
                              batch_size=int(BC.RECIPE["eval_batch_size"]))
            mae = BC.mae(pred, y_monitor)
            recorded = info["monitor_reads"][state_tag]["monitor_mae"]
            entry["checkpoints"][state_tag] = {
                "file": str(path.relative_to(BC.ROUND)),
                "sha256": BC.sha256_file(path),
                "step": blob["step"],
                "epoch": blob["epoch"],
                "monitor_mae_recomputed": mae,
                "monitor_mae_recorded": recorded,
                "monitor_recompute_abs_diff": abs(mae - recorded),
                "monitor_pred_sha256": C.sha256_array(pred.astype(np.float32)),
            }
            if abs(mae - recorded) > 1e-9:
                raise SystemExit(f"{tag}/{state_tag}: Monitor recompute mismatch "
                                 f"{mae} vs {recorded}")
        results["runs"][tag] = entry

    # same-seed R/B must share the public initial state and the batch order
    checks = {}
    for seed in sorted({v["seed"] for v in results["runs"].values()}):
        rs = [v for v in results["runs"].values() if v["seed"] == seed]
        if len(rs) == 2:
            checks[f"seed{seed}"] = {
                "same_initial_state": rs[0]["initial_state_sha256"] == rs[1]["initial_state_sha256"],
                "same_first_batch_rows": rs[0]["first_batch_rows"] == rs[1]["first_batch_rows"],
                "same_parameters": rs[0]["parameters"] == rs[1]["parameters"],
            }
            if not all(checks[f"seed{seed}"].values()):
                raise SystemExit(f"seed {seed}: R/B do not share the public initial state")
    results["pairing_checks"] = checks
    results["locked_read"] = False
    BC.write_json(BC.ANALYSIS / "freeze.json", results)
    BC.log(f"[bosi-freeze] frozen {len(results['runs'])} runs -> analysis/freeze.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
