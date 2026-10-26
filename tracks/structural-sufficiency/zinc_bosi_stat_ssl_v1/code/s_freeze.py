"""Freeze every candidate checkpoint and Monitor/Fit read before the single
Locked read.  Refuses to run if a Locked read already exists, so the order
(freeze -> one Locked read) cannot be reversed silently.

Nothing here touches Locked.  The GPU ledger is written here as well, because
the freeze is the last moment at which the budget can still stop the round.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402
import ad_common as C  # noqa: E402

BUDGET_CAP_GPU_HOURS = 4.0


def _v0path(text):
    p = Path(text)
    return p if p.is_absolute() else (SC.V0 / p)


def _verify_reused() -> dict:
    out = {}
    freeze = json.loads((SC.V0 / "analysis" / "freeze.json").read_text())
    locked = json.loads((SC.V0 / "analysis" / "locked_read.json").read_text())
    out["v0_freeze_sha256"] = SC.sha256_file(SC.V0 / "analysis" / "freeze.json")
    out["v0_locked_read_sha256"] = SC.sha256_file(SC.V0 / "analysis" / "locked_read.json")
    if SC.sha256_file(SC.DATA_CACHE) != SC.DATA_CACHE_SHA256:
        raise SystemExit("official-train cache changed")
    if SC.sha256_file(SC.REL_CACHE) != freeze["relation_cache_sha256"] and \
            SC.BD.relation_content_hash(SC.load_relation_cache()) != SC.REL_CACHE_CONTENT_SHA256:
        raise SystemExit("relation cache changed")
    out["data_cache_sha256"] = SC.DATA_CACHE_SHA256
    out["relation_cache_content_sha256"] = SC.REL_CACHE_CONTENT_SHA256
    out["checkpoint"] = {}
    out["locked_pred"] = {}
    for seed in (0, 1):
        for ckpt in ("best_monitor", "final"):
            p = SC.historical_b_checkpoint(seed, ckpt)
            exp = freeze["runs"][f"B_s{seed}"]["checkpoints"][ckpt]["sha256"]
            if SC.sha256_file(p) != exp:
                raise SystemExit(f"reused BOSI-B checkpoint changed: s{seed}/{ckpt}")
            out["checkpoint"][f"B_s{seed}/{ckpt}"] = {"file": SC.rel_or_abs(p) if
                                                      str(p).startswith(str(SC.ROUND))
                                                      else str(p), "sha256": exp}
            lp = SC.historical_b_locked_pred(seed, ckpt)
            exp_l = locked["predictions"][f"B_s{seed}"][ckpt]["sha256"]
            if SC.sha256_file(lp) != exp_l:
                raise SystemExit(f"reused BOSI-B locked prediction changed: s{seed}/{ckpt}")
            out["locked_pred"][f"B_s{seed}/{ckpt}"] = {
                "file": str(lp), "sha256": exp_l,
                "locked_mae": locked["predictions"][f"B_s{seed}"][ckpt]["locked_mae"]}
        if freeze["runs"][f"B_s{seed}"]["initial_state_sha256"] != \
                SC.HISTORICAL_INIT_SHA256[seed]:
            raise SystemExit(f"historical BOSI-B init hash changed for seed {seed}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pretrain", nargs="*", default=["S_pre0_full", "S_pre1_full"])
    p.add_argument("--pilot", default="S_pre0_pilot")
    args = p.parse_args(argv)

    if (SC.ANALYSIS / "locked_read.json").exists():
        raise SystemExit("refusing to freeze after a Locked read already exists")

    reused = _verify_reused()
    cache = SC.load_train_cache()
    rows = SC.split_rows()
    y_monitor = SC.labels(cache, rows["monitor"])
    y_fit = SC.labels(cache, rows["fit8k"])

    frozen = {
        "frozen_utc": SC.now(),
        "round": SC.ROUND_ID,
        "config_sha256": SC.sha256_file(SC.CONFIGS / "bosi_stat_ssl_v1.json"),
        "protocol_sha256": SC.sha256_file(SC.ROUND / "PROTOCOL.md"),
        "data_cache_sha256": SC.DATA_CACHE_SHA256,
        "relation_cache_sha256": SC.REL_CACHE_CONTENT_SHA256,
        "split_hashes": rows["hashes"],
        "reused": reused,
        "pilot_decision": None,
        "pretrain": {},
        "runs": {},
        "pairing_checks": {},
        "gpu_ledger": {},
        "locked_read": False,
    }

    pilot_path = SC.RUNS / args.pilot / "pretrain.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        frozen["pilot_decision"] = {
            "mode": pilot["mode"], "updates": pilot["updates"],
            "wall_seconds": pilot["wall_seconds"],
            "gpu_hours": pilot["gpu_hours"],
            "probe_delta_ce": pilot["probe"]["delta_ce"],
            "per_head_final_ce": {
                "atom": pilot["curve"][-1]["ce_atom"],
                "bond": pilot["curve"][-1]["ce_bond"]},
            "state_sha256": pilot["state_sha256"],
        }

    for tag in args.pretrain:
        path = SC.RUNS / tag / "pretrain.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text())
        state = SC.RUNS / tag / SC.pretrain_state_name(info["seed"])
        if SC.sha256_file(state) != info["state_sha256"]:
            raise SystemExit(f"pretraining state changed: {tag}")
        frozen["pretrain"][tag] = {
            "seed": info["seed"], "mode": info["mode"], "updates": info["updates"],
            "epochs_run": info["epochs_run"],
            "export_sha256": info["export_sha256"],
            "state_file": SC.rel_or_abs(state),
            "state_sha256": info["state_sha256"],
            "backbone_initial_state_sha256": info["model"]["backbone_initial_state_sha256"],
            "backbone_final_state_sha256": info.get("backbone_final_state_sha256"),
            "wall_seconds": info["wall_seconds"],
            "gpu_hours": info["gpu_hours"],
            "final_ce_atom": info["curve"][-1]["ce_atom"],
            "final_ce_bond": info["curve"][-1]["ce_bond"],
            "probe": info["probe"],
            "recipe": info["recipe"],
            "frozen_modules": info["model"]["frozen"],
            "labels_read": info["labels_read"],
        }

    for tag in args.runs:
        run_dir = SC.RUNS / tag
        info = json.loads((run_dir / "train.json").read_text())
        seed = int(info["seed"])
        ckpts = {}
        for ckpt in ("best_monitor", "final"):
            path = run_dir / f"{ckpt}.pt"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            ckpts[ckpt] = {"file": SC.rel_or_abs(path), "sha256": SC.sha256_file(path),
                           "step": int(blob.get("step", -1)),
                           "monitor_mae": float(blob.get("monitor_mae", float("nan")))}
        hist = SC.historical_b_run(seed)
        pairing = {
            "same_initial_state": None,
            "same_first_batch_rows": info["first_batch_rows"] == hist["first_batch_rows"],
            "same_parameters": info["parameters"] == SC.BC.EXPECTED_PARAMETERS,
            "same_seed": int(info["seed"]) == int(hist["seed"]),
            "initial_state_sha256": info["model_info"]["initial_state_sha256"],
            "historical_initial_state_sha256": hist["initial_state_sha256"],
        }
        pairing["same_initial_state"] = (pairing["initial_state_sha256"]
                                        == pairing["historical_initial_state_sha256"])
        frozen["runs"][tag] = {
            "arm": "B", "seed": seed, "init": info["init"],
            "total_updates": info["total_updates"],
            "extra_updates": info["extra_updates"],
            "updates_run": info["updates_run"],
            "lr_rule": info["lr_rule"],
            "epochs_run": info["epochs_run"],
            "parameters": info["parameters"],
            "initial_state_sha256": info["model_info"]["initial_state_sha256"],
            "first_batch_rows": info["first_batch_rows"],
            "best_monitor": info["best_monitor"],
            "monitor_reads": info["monitor_reads"],
            "fit_mae_at_best_monitor": _fit_mae(info, run_dir, cache, rows),
            "checkpoints": ckpts,
            "module_sha256_at_start": info["module_sha256_at_start"],
            "gpu_hours": info["gpu_hours"],
            "wall_seconds": info["wall_seconds"],
            "device": info["device"],
            "locked_read": False,
        }
        frozen["pairing_checks"][f"seed{seed}"] = pairing
        # the SSL arms must additionally expose the pretrained-init provenance
        if info["init"]["init"] == "ssl":
            frozen["runs"][tag]["ssl_init"] = info["init"]

    total = 0.0
    ledger = []
    entries = list(frozen["pretrain"].items())
    for tag, entry in entries:
        ledger.append({"job": tag, "gpu_hours": entry["gpu_hours"],
                       "wall_seconds": entry["wall_seconds"]})
        total += entry["gpu_hours"]
    if frozen["pilot_decision"]:
        ledger.append({"job": args.pilot,
                       "gpu_hours": frozen["pilot_decision"]["gpu_hours"],
                       "wall_seconds": frozen["pilot_decision"]["wall_seconds"]})
        total += frozen["pilot_decision"]["gpu_hours"]
    for tag, entry in frozen["runs"].items():
        ledger.append({"job": tag, "gpu_hours": entry["gpu_hours"],
                       "wall_seconds": entry["wall_seconds"]})
        total += entry["gpu_hours"]
    frozen["gpu_ledger"] = {
        "jobs": ledger,
        "total_a100_gpu_hours": float(total),
        "cap_gpu_hours": BUDGET_CAP_GPU_HOURS,
        "within_budget": bool(total <= BUDGET_CAP_GPU_HOURS),
        "counting_rule": "1 GPU per job, wall-clock hours per job, failed jobs included",
    }
    if total > BUDGET_CAP_GPU_HOURS:
        raise SystemExit(f"GPU budget exceeded: {total:.3f} > {BUDGET_CAP_GPU_HOURS}")

    SC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    SC.write_json(SC.ANALYSIS / "freeze.json", frozen)
    SC.log(f"[s-freeze] wrote freeze.json: {len(frozen['runs'])} runs, "
           f"{len(frozen['pretrain'])} pretraining jobs, "
           f"{total:.3f} A100-h of {BUDGET_CAP_GPU_HOURS}")
    return 0


def _fit_mae(info: dict, run_dir: Path, cache, rows) -> float | None:
    """Fit MAE of the best-monitor checkpoint (descriptive; Locked is untouched)."""
    ckpt = run_dir / "best_monitor.pt"
    if not ckpt.exists():
        return None
    import s_pretext as SP

    model, _ = SP.build_backbone(SC.fit_only_norm(cache), int(info["seed"]), "cpu")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    C.load_state(model, blob["state_dict"], "cpu")
    pred = SC.BC.predict(model, cache, SC.load_relation_cache(), rows["fit8k"],
                         device="cpu", batch_size=int(SC.BC.RECIPE["eval_batch_size"]))
    return SC.BC.mae(pred, SC.labels(cache, rows["fit8k"]))


if __name__ == "__main__":
    raise SystemExit(main())
