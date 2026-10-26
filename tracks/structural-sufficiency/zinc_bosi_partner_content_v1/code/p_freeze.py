"""Freeze the partner-content null runs before the single Locked read.

Writes ``analysis/freeze.json``.  Refuses to run if a Locked read already exists,
so the order (freeze -> one Locked read) cannot be reversed silently.

What is frozen here:

* every hash of the reused, already-frozen evidence (BOSI v0 freeze / Locked
  read / summary, the v0 relation cache, the v0 B/R checkpoints, the v0 Locked
  prediction files, the DCC-L prediction files) -- read-only, hash-verified;
* P's checkpoints, Monitor **and Fit** reads (Locked is never touched here);
* the pairing checks against the historical BOSI-B run of the same seed
  (initial state, first batch rows, parameters).

Usage:
    python .../p_freeze.py --runs P_s0 P_s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402

REUSED_ARMS = ("B_s0", "B_s1", "R_s0", "R_s1")


def _v0path(text):
    """Resolve a path recorded by the v0 round (relative to that round's root)."""
    p = Path(text)
    return p if p.is_absolute() else (BC.V0 / p)



def _verify_reused(cfg: dict) -> dict:
    r = cfg["reused_frozen_artifacts"]
    checks = {
        "v0_freeze": (Path(r["v0_freeze"]), r["v0_freeze_sha256"], "file"),
        "v0_locked_read": (Path(r["v0_locked_read"]), r["v0_locked_read_sha256"], "file"),
        "v0_summary": (Path(r["v0_summary"]), r["v0_summary_sha256"], "file"),
    }
    out = {}
    for name, (path, expected, kind) in checks.items():
        actual = BC.sha256_file(path)
        if actual != expected:
            raise SystemExit(f"reused artefact changed: {name} {actual} != {expected}")
        out[name] = {"path": BC.rel_or_abs(path), "sha256": actual}

    v0 = json.loads(Path(r["v0_freeze"]).read_text())
    cache = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu",
                       weights_only=False)
    rel_hash = BD.relation_content_hash(cache)
    if rel_hash != v0["relation_cache_sha256"]:
        raise SystemExit("relation cache content hash does not match the v0 freeze")
    out["relation_cache"] = {"path": BC.rel_or_abs(BC.CACHE / "bosi_rel_train.pt"),
                             "content_sha256": rel_hash}

    for tag in REUSED_ARMS:
        entry = v0["runs"][tag]
        for ckpt, meta in entry["checkpoints"].items():
            path = _v0path(meta["file"])
            actual = BC.sha256_file(path)
            if actual != meta["sha256"]:
                raise SystemExit(f"reused checkpoint changed: {tag}/{ckpt}")
        out[f"checkpoint/{tag}"] = {
            ckpt: {"file": str(Path(meta["file"])),
                   "sha256": meta["sha256"]}
            for ckpt, meta in entry["checkpoints"].items()}

    v0l = json.loads(Path(r["v0_locked_read"]).read_text())
    for tag in REUSED_ARMS:
        for ckpt, meta in v0l["predictions"][tag].items():
            path = _v0path(meta["file"])
            if BC.sha256_file(path) != meta["sha256"]:
                raise SystemExit(f"reused locked prediction changed: {tag}/{ckpt}")
        out[f"locked_pred/{tag}"] = {
            ckpt: {"file": str(Path(meta["file"])),
                   "sha256": meta["sha256"], "locked_mae": meta["locked_mae"]}
            for ckpt, meta in v0l["predictions"][tag].items()}

    v0s = json.loads(Path(r["v0_summary"]).read_text())
    for seed in ("0", "1"):
        meta = v0s["historical_dcc"][seed]
        path = _v0path(meta["file"])
        if BC.sha256_file(path) != meta["sha256"]:
            raise SystemExit(f"DCC-L prediction changed: seed {seed}")
        out[f"dccl/{seed}"] = {"file": str(path), "sha256": meta["sha256"],
                               "locked_mae": meta["locked_mae"]}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args(argv)

    if (BC.ANALYSIS / "locked_read.json").exists():
        raise SystemExit("refusing to freeze after a Locked read already exists")

    cfg = json.loads((BC.CONFIGS / "bosi_p_v1.json").read_text())
    cache = BC.load_train_cache()
    rows = BC.split_rows()
    norm = BC.fit_only_norm(cache, rows["fit8k"])
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    y_monitor = BC.labels(cache, rows["monitor"])
    y_fit = BC.labels(cache, rows["fit8k"])

    frozen = {
        "frozen_utc": BC.now(),
        "round": BC.ROUND_ID,
        "config_sha256": BC.sha256_file(BC.CONFIGS / "bosi_p_v1.json"),
        "protocol_sha256": BC.sha256_file(BC.ROUND / "PROTOCOL.md"),
        "design_sha256": BC.sha256_file(BC.ROUND / "DESIGN.md"),
        "data_cache_sha256": BC.sha256_file(BC.DATA_CACHE),
        "split_hashes": rows["hashes"],
        "reused": _verify_reused(cfg),
        "runs": {},
    }

    def reads(model, out, info, seed, arm):
        entry = {"run": out.name, "arm": arm, "seed": int(seed),
                 "parameters": int(BC.EXPECTED_PARAMETERS),
                 "initial_state_sha256": info["model_info"]["initial_state_sha256"],
                 "ssl_sha256": info["model_info"]["ssl"]["sha256"],
                 "first_batch_rows": info["first_batch_rows"],
                 "wall_seconds": info["wall_seconds"], "gpu_hours": info["gpu_hours"],
                 "epochs_run": info["epochs_run"], "best_monitor": info["best_monitor"],
                 "checkpoints": {}}
        for ckpt, tag in (("best_monitor.pt", "best_monitor"), ("final.pt", "final")):
            path = out / ckpt
            blob = torch.load(path, map_location="cpu", weights_only=False)
            C.load_state(model, blob["state_dict"], "cpu")
            mon = BC.predict(model, cache, rel, rows["monitor"], device="cpu",
                             batch_size=int(BC.RECIPE["eval_batch_size"]))
            fit = BC.predict(model, cache, rel, rows["fit8k"], device="cpu",
                             batch_size=int(BC.RECIPE["eval_batch_size"]))
            mae_m, mae_f = BC.mae(mon, y_monitor), BC.mae(fit, y_fit)
            rec = info["monitor_reads"][tag]["monitor_mae"]
            if abs(mae_m - rec) > 1e-6:
                raise SystemExit(f"{out.name}/{tag}: Monitor recompute {mae_m} != {rec}")
            entry["checkpoints"][tag] = {
                "file": BC.rel_or_abs(path), "sha256": BC.sha256_file(path),
                "step": blob["step"], "epoch": blob["epoch"],
                "monitor_mae_recomputed": mae_m, "monitor_mae_recorded": rec,
                "monitor_recompute_abs_diff": abs(mae_m - rec),
                "monitor_pred_sha256": C.sha256_array(mon.astype(np.float32)),
                "fit_mae": mae_f,
            }
        return entry

    for tag in args.runs:
        out = BC.RUNS / tag
        info = json.loads((out / "train.json").read_text())
        model, minfo = BM.build_model(norm, info["arm"], info["seed"], "cpu")
        model.arm = info["arm"]
        frozen["runs"][tag] = reads(model, out, info, info["seed"], info["arm"])

    # the reused B/R runs need their Fit read too (frozen weights, Fit rows only)
    v0 = json.loads(Path(cfg["reused_frozen_artifacts"]["v0_freeze"]).read_text())
    for tag in REUSED_ARMS:
        v0run = v0["runs"][tag]
        model, _ = BM.build_model(norm, v0run["arm"], v0run["seed"], "cpu")
        model.arm = v0run["arm"]
        entry = {"run": tag, "arm": v0run["arm"], "seed": int(v0run["seed"]),
                 "parameters": v0run["parameters"],
                 "initial_state_sha256": v0run["initial_state_sha256"],
                 "first_batch_rows": v0run["first_batch_rows"],
                 "best_monitor": v0run["best_monitor"],
                 "source": f"zinc_bosi_architecture_v0/runs/{tag}", "checkpoints": {}}
        for ckpt, meta in v0run["checkpoints"].items():
            path = _v0path(meta["file"])
            blob = torch.load(path, map_location="cpu", weights_only=False)
            C.load_state(model, blob["state_dict"], "cpu")
            fit = BC.predict(model, cache, rel, rows["fit8k"], device="cpu",
                             batch_size=int(BC.RECIPE["eval_batch_size"]))
            entry["checkpoints"][ckpt] = {
                "file": str(Path(meta["file"])), "sha256": meta["sha256"],
                "step": meta["step"], "epoch": meta["epoch"],
                "monitor_mae_recomputed": meta["monitor_mae_recomputed"],
                "fit_mae": BC.mae(fit, y_fit), "reused": True}
        frozen["runs"][tag] = entry

    checks = {}
    for seed in sorted({v["seed"] for v in frozen["runs"].values()}):
        p_run = frozen["runs"].get(f"P_s{seed}")
        b_run = frozen["runs"].get(f"B_s{seed}")
        if p_run is None or b_run is None:
            continue
        checks[f"seed{seed}"] = {
            "same_initial_state": p_run["initial_state_sha256"] == b_run["initial_state_sha256"],
            "same_first_batch_rows": p_run["first_batch_rows"] == b_run["first_batch_rows"],
            "same_parameters": p_run["parameters"] == b_run["parameters"],
            "initial_state_sha256": p_run["initial_state_sha256"],
        }
        if not all(v for k, v in checks[f"seed{seed}"].items() if k != "initial_state_sha256"):
            raise SystemExit(f"seed {seed}: P does not share B's public initial state")
    frozen["pairing_checks"] = checks
    frozen["gpu_hours_total"] = float(sum(
        v.get("gpu_hours", 0.0) for k, v in frozen["runs"].items() if k.startswith("P_")))
    frozen["locked_read"] = False
    BC.write_json(BC.ANALYSIS / "freeze.json", frozen)
    BC.log(f"[p-freeze] frozen {len(frozen['runs'])} runs -> analysis/freeze.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
