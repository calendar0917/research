"""Write the durable training ledger (no Locked access).

The run directories are git-ignored.  This script copies the compact,
auditable facts — recipe, Monitor curve, hashes, GPU accounting and rr
provenance — into a single JSON file under ``analysis/`` so the training record
survives a fresh clone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ar_common as C  # noqa: E402

TAGS = ("A_s0", "D_s0", "A_s1", "D_s1")


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def _metas() -> dict:
    """Collect rr meta.json records by experiment name."""
    out: dict = {}
    roots = [C.REPO / ".rr" / "pulled"]
    roots += [C.RUNS, C.RUNS_SMOKE, C.ROUND / "runs_smoke"]
    for root in roots:
        if not root.exists():
            continue
        pattern = "*/meta.json" if root.name == "pulled" else "**/meta.json"
        for p in root.glob(pattern):
            try:
                m = C.read_json(p)
            except Exception:
                continue
            if not isinstance(m, dict) or "run_id" not in m:
                continue
            key = m.get("experiment", "unknown")
            out.setdefault(key, []).append({
                "run_id": m["run_id"],
                "host": m.get("host"),
                "pool": m.get("pool"),
                "git_commit": m.get("git_commit"),
                "command": m.get("command"),
                "state": m.get("state"),
                "runtime": m.get("runtime", {}),
                "meta_path": str(p.relative_to(C.REPO)),
            })
    return out


def _smoke_records(metas: dict | None = None) -> list[dict]:
    out = []
    if not C.RUNS_SMOKE.exists():
        return out
    name_map = {
        "A_s0_128": "ar-smoke-A", "D_s0_128": "ar-smoke-D",
        "A_s0_315": "ar-smoke-A5", "D_s0_315": "ar-smoke-D5",
    }
    for p in sorted(C.RUNS_SMOKE.glob("*/smoke.json")):
        rec = C.read_json(p)
        meta = p.parent / "meta.json"
        if meta.exists():
            prov = C.read_json(meta)
        else:
            candidates = (metas or {}).get(name_map.get(p.parent.name, ""), [])
            prov = candidates[0] if candidates else {}
        out.append({
            "tag": p.parent.name,
            "arm": rec.get("arm"),
            "seed": rec.get("seed"),
            "updates": rec.get("updates"),
            "epochs": rec.get("epochs"),
            "epoch_seconds": rec.get("epoch_seconds"),
            "projected_gpu_hours_per_run": rec.get("projected_gpu_hours_per_run"),
            "gpu_hours": rec.get("gpu_hours"),
            "wall_seconds": rec.get("wall_seconds"),
            "peak_memory_mib": rec.get("peak_memory_mib"),
            "device_name": rec.get("device_name"),
            "host": rec.get("host"),
            "git_commit": rec.get("git_commit"),
            "config_sha256": rec.get("config_sha256"),
            "official_test_loaded": rec.get("official_test_loaded"),
            "official_valid_loaded": rec.get("official_valid_loaded"),
            "locked_read": rec.get("locked_read"),
            "rr": {
                "run_id": prov.get("run_id"),
                "pool": prov.get("pool"),
                "state": prov.get("state"),
                "runtime": prov.get("runtime", {}),
            },
        })
    return out


def main() -> int:
    meta = _metas()
    formal = {}
    total = 0.0
    norm_hashes = set()
    shared = {}
    for tag in TAGS:
        d = C.RUNS / tag
        train = C.read_json(d / "train.json")
        total += float(train["gpu_hours"])
        norm_hashes.add(train["norm_hash"])
        shared.setdefault(int(train["seed"]), set()).add(
            train["model_info"]["shared_trunk_phi_state_sha256"])
        exp_name = f"ar-{tag.replace('_', '-')}"
        prov = meta.get(exp_name, [])
        formal[tag] = {
            "arm": train["arm"],
            "seed": train["seed"],
            "parameters": train["parameters"],
            "parameter_breakdown": train["model_info"]["parameters_breakdown_"
                                    + train["arm"]],
            "model_info": train["model_info"],
            "recipe": train["recipe"],
            "total_updates_run": train["total_updates_run"],
            "epochs_run": train["epochs_run"],
            "n_train_rows": train["n_train_rows"],
            "norm_hash": train["norm_hash"],
            "norm_source": train["norm"]["source"],
            "split_hashes": train["split_hashes"],
            "best_monitor": train["best_monitor"],
            "eval_reads": {
                k: {kk: vv for kk, vv in v.items() if not kk.endswith("_pred")}
                for k, v in train["eval_reads"].items()
            },
            "tail10_median_monitor_mae": train["tail10_median_monitor_mae"],
            "tail10_median_online_train_mae": train["tail10_median_online_train_mae"],
            "wall_seconds": train["wall_seconds"],
            "gpu_hours": train["gpu_hours"],
            "peak_memory_mib": train["peak_memory_mib"],
            "host": train["host"],
            "device_name": train["device_name"],
            "torch": train["torch"],
            "cuda": train["cuda"],
            "python": train["python"],
            "git_commit": train["git_commit"],
            "config_sha256": train["config_sha256"],
            "official_test_loaded": train["official_test_loaded"],
            "official_valid_loaded": train["official_valid_loaded"],
            "locked_read": train["locked_read"],
            "curve": train["curve"],
            "artifact_sha256": {
                name: _sha(d / name)
                for name in ("best_monitor.pt", "final.pt",
                             "predictions_best_monitor.pt", "predictions_final.pt",
                             "train.json", "curve.csv")
            },
            "rr": prov,
        }
    smoke = _smoke_records(meta)
    smoke_total = float(sum(float(s.get("gpu_hours") or 0.0) for s in smoke))
    payload = {
        "protocol_id": C.load_config()["protocol_id"],
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "config_sha256": _sha(C.CONFIG),
        "protocol_md_sha256": _sha(C.ROUND / "PROTOCOL.md"),
        "formal_runs": formal,
        "smoke_runs": smoke,
        "protocol_consistency": {
            "all_updates_47400": all(v["total_updates_run"] == 47400 for v in formal.values()),
            "all_epochs_753": all(v["epochs_run"] == 753 for v in formal.values()),
            "one_norm_hash": sorted(norm_hashes),
            "shared_init_identical_within_seed": {str(k): len(v) == 1 for k, v in shared.items()},
            "all_A_params_453361": all(v["parameters"] == 453361
                                       for v in formal.values() if v["arm"] == "A"),
            "all_D_params_461521": all(v["parameters"] == 461521
                                       for v in formal.values() if v["arm"] == "D"),
            "no_locked_read_in_training": all(not v["locked_read"] for v in formal.values()),
            "no_official_valid_loaded": all(not v["official_valid_loaded"]
                                            for v in formal.values()),
            "no_official_test_loaded": all(not v["official_test_loaded"]
                                           for v in formal.values()),
        },
        "budget": {
            "hard_cap_gpu_hours": float(C.load_config()["budget"]["gpu_hours_hard_cap"]),
            "formal_gpu_hours": total,
            "smoke_gpu_hours": smoke_total,
            "total_gpu_hours": total + smoke_total,
            "within_cap": bool(total + smoke_total
                               <= float(C.load_config()["budget"]["gpu_hours_hard_cap"])),
            "accounting": "sum of per-run wall_seconds / 3600 from train.json and smoke.json; includes all four smoke jobs and four formal jobs",
        },
        "historical_reference": {
            "model": "DCC-L fit8k best_monitor (not retrained)",
            "runs": {
                f"C_s{s}": {
                    "train_json": str(C.DCCL_RUNS / f"fit8k_s{s}" / "train.json"),
                    "checkpoint_sha256": _sha(C.DCCL_RUNS / f"fit8k_s{s}" / "best_monitor.pt"),
                    "predictions_sha256": _sha(C.DCCL_RUNS / f"fit8k_s{s}" / "predictions_best_monitor.pt"),
                    "monitor_mae": float(C.read_json(
                        C.DCCL_RUNS / f"fit8k_s{s}" / "train.json")["eval_reads"]["best_monitor"]["monitor_mae"]),
                    "locked_mae": float(C.read_json(
                        C.DCCL_RUNS / f"fit8k_s{s}" / "train.json")["eval_reads"]["best_monitor"]["locked_mae"]),
                }
                for s in (0, 1)
            },
        },
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    C.write_json(C.ANALYSIS / "training_ledger.json", payload)
    C.log(json.dumps(payload["budget"], indent=2))
    C.log(f"[ledger] wrote {C.ANALYSIS / 'training_ledger.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
