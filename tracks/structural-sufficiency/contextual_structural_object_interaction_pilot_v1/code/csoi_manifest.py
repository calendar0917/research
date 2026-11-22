"""Build ``results/run_manifest.json``: one provenance record for the whole round.

Collects the frozen inputs (config, protocol, cache identity, code hashes), every
run that produced a number in the report (trained arm, reused arm, smoke, checks),
and the *execution regime* of each run as recorded by ``rr`` (host, node, driver,
torch, commit, exit code).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D

ROUND = C.ROUND_DIR
CODE = ROUND / "code"
TRAINED = {"B_CSOI_ADD": "CSOI_ADD_LIPO_R0", "C_CSOI_GLOBAL": "CSOI_GLOBAL_LIPO_R0",
           "D_CSOI_REL": "CSOI_REL_LIPO_R0"}
REUSED = {
    "A_TRC_FACTOR_CUTRESP": C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0",
    "E_GINE_RAW_REF": C.PARENT_ROUND / "results" / "GINE_RAW_LIPO_R0",
}


def code_hashes() -> dict[str, str]:
    return {p.name: C.sha256_file(p) for p in sorted(CODE.glob("*.py"))}


def rr_meta(experiment: str) -> dict[str, Any] | None:
    pattern = str(C.REPO_ROOT / ".rr" / "pulled" / f"{experiment}-*" / "meta.json")
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    meta = C.read_json(hits[-1])
    rt = meta.get("runtime") or {}
    return {
        "experiment": meta.get("experiment"), "run_id": meta.get("run_id"),
        "host": meta.get("host"), "pool": meta.get("pool"),
        "backend": meta.get("backend"), "node": rt.get("node"),
        "gpu_names": (rt.get("gpus") or {}).get("names"),
        "cuda_visible_devices": rt.get("cuda_visible_devices"),
        "driver": rt.get("driver_version"), "torch": rt.get("torch_version"),
        "torch_cuda": rt.get("torch_cuda"), "python": rt.get("python_version"),
        "commit": meta.get("git_commit"), "branch": meta.get("git_branch"),
        "dirty": meta.get("git_dirty"), "state": meta.get("state"),
        "exit_code": rt.get("exit_code"), "started": rt.get("started_at"),
        "completed": rt.get("completed_at"), "command": meta.get("command"),
    }


def _artifacts() -> dict[str, str]:
    out = {}
    for path in sorted(list(C.analysis_dir().glob("*.json")) + list(C.results_root().glob("*.json"))
                       + list(C.results_root().glob("*.csv"))):
        out[str(path.relative_to(ROUND))] = C.sha256_file(path)
    return out


def _synthetic_summary() -> dict[str, Any] | None:
    path = C.analysis_dir() / "synthetic_flip.json"
    if not path.is_file():
        return None
    d = C.read_json(path)
    return {"path": "analysis/synthetic_flip.json", "verdict": d.get("verdict"),
            "n_examples": d.get("n_examples"), "steps": d.get("steps"),
            "device": d.get("device"),
            "rows": [{k: r[k] for k in ("arm", "rule", "normalized_rmse", "corr")}
                     for r in d.get("rows", [])]}


def _compression_summary() -> dict[str, Any] | None:
    path = C.results_root() / "compression_diagnostics.json"
    if not path.is_file():
        return None
    d = C.read_json(path)
    return {"path": "results/compression_diagnostics.json",
            "n_pairs": d.get("n_pairs_scored"),
            "designs": d.get("designs"),
            "explained_variance_ratio_top8": d.get("explained_variance_ratio_top8")}


def _structure_summary() -> dict[str, Any]:
    d = C.read_json(C.analysis_dir() / "structure_stats.json")
    tok = C.read_json(C.analysis_dir() / "token_stats.json")
    return {"objects": d["objects"]["n_objects_by_size"],
            "object_bucket_share": d["objects"]["object_distance_bucket_share"],
            "fragment_tokens": d["fragments"]["fragment_tokens"],
            "parses_per_molecule": d["molecules"]["parses_per_molecule"],
            "attention_entries_total": d["molecules"]["attention_entries_total"],
            "attention_entries_by_split": d["molecules"]["attention_entries_by_split"],
            "token_guard": tok["guard"],
            "files": ["analysis/structure_stats.json", "analysis/token_stats.json"]}


def run_entry(run_id: str, arm_key: str) -> dict[str, Any]:
    rdir = C.run_dir(run_id)
    man = C.read_json(rdir / "manifest.json") if (rdir / "manifest.json").is_file() else {}
    entry = {
        "arm_key": arm_key, "run_id": run_id, "kind": "trained",
        "dir": str(rdir.relative_to(C.REPO_ROOT)),
        "arm": man.get("arm"), "epochs": man.get("epochs"),
        "best": man.get("best"), "final": man.get("final"),
        "params": man.get("params"), "scaler": man.get("scaler"),
        "wall_seconds": man.get("wall_seconds"), "gpu_hours": man.get("gpu_hours"),
        "max_vram_bytes": man.get("max_vram_bytes"),
        "commit": (man.get("signature") or {}).get("commit"),
        "cache_sha256": man.get("cache_sha256"),
        "content_sha256": (C.read_json(rdir / "config.json").get("cfg", {})
                           .get("expected_content_sha256", {}).get("lipo")
                           if (rdir / "config.json").is_file() else None),
        "test_policy": man.get("test_policy"),
        "zeroed_categories": man.get("zeroed_categories"),
        "max_steps": man.get("max_steps"),
        "supervision": man.get("supervision"), "aux_weight": man.get("aux_weight"),
        "seed": man.get("seed"),
        "environment": man.get("environment"),
        "rr": rr_meta(run_id),
    }
    return entry


def reused_entry(path: Path, arm_key: str) -> dict[str, Any]:
    man = C.read_json(path / "manifest.json")
    return {"arm_key": arm_key, "run_id": path.name, "kind": "reuse",
            "dir": str(path.relative_to(C.REPO_ROOT)),
            "arm": man.get("arm"), "best": man.get("best"),
            "final": man.get("final"), "params": man.get("params"),
            "gpu_hours": man.get("gpu_hours"), "commit": man.get("commit"),
            "note": man.get("note"), "recipe": man.get("recipe"),
            "environment": man.get("environment")}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.load_pack(cfg, cfg["dataset_key"])
    runs = [run_entry(rid, key) for key, rid in TRAINED.items()
            if (C.run_dir(rid) / "manifest.json").is_file()]
    reused = [reused_entry(p, k) for k, p in REUSED.items() if (p / "manifest.json").is_file()]
    gpu_hours = float(sum((r.get("gpu_hours") or 0.0) for r in runs))
    timing = C.read_json(C.analysis_dir() / "timing.json")
    checks = C.read_json(C.analysis_dir() / "checks.json")
    out = {
        "protocol": cfg["protocol_id"],
        "round": str(ROUND.relative_to(C.REPO_ROOT)),
        "generated_utc": C.now_utc(),
        "commit": C.git_commit(),
        "config_sha256": C.sha256_file(ROUND / "configs" / "csoi_v1.json"),
        "protocol_sha256": C.sha256_file(ROUND / "PROTOCOL.md"),
        "code_hashes": code_hashes(),
        "parent_round": str(C.PARENT_ROUND.relative_to(C.REPO_ROOT)),
        "parent_modules": C.parent_module_provenance(),
        "inputs": {
            "dataset": cfg["dataset"], "dataset_key": cfg["dataset_key"],
            "cache_content_sha256": pack["content_sha256"],
            "cache_file_sha256_this_checkout": pack["cache_sha256"],
            "identity_note": cfg["identity_note"],
            "n_molecules": int(pack["n_molecules"]),
            "n_fragments": int(pack["n_fragments"]),
            "n_parses": int(pack["n_parses"]),
            "splits": {"train": int(np.asarray(pack["train_idx"]).size),
                       "valid": int(np.asarray(pack["valid_idx"]).size),
                       "test_kept_unused": int(np.asarray(pack["test_idx"]).size)},
            "test_labels_read": False,
        },
        "engineering_gate": {"ok": checks["ok"],
                            "checks": {k: bool(v.get("ok"))
                                       for k, v in checks["checks"].items()},
                            "path": "analysis/checks.json"},
        "horizon": {"epochs": timing["horizon_epochs"],
                    "epoch_seconds_measured": timing["epoch_seconds"],
                    "reduced_from_cap": timing["horizon_reduced"],
                    "source": "analysis/timing.json (remote smoke on res)"},
        "budget": {"gpu_hours_max": float(cfg["budget"]["gpu_hours_max"]),
                   "gpu_hours_used": gpu_hours,
                   "gpu_hours_used_fraction": gpu_hours / float(cfg["budget"]["gpu_hours_max"]),
                   "max_parallel_gpus": int(cfg["budget"]["max_parallel_gpus"]),
                   "wall_hours_limit": float(cfg["budget"]["wall_hours"])},
        "runs": runs,
        "reused_runs": reused,
        "smoke_runs": [rr_meta(x) for x in
                       ("csoi-timing-smoke2", "csoi-synthetic-flip2", "csoi-clean-worktree")
                       if rr_meta(x)],
        "execution_regime": {
            "trained_on": "res (a100-2, process backend, res-gpu0/res-gpu1)",
            "local_cpu_work": ["cache build", "13-gate check battery", "corpus and token "
                               "statistics", "all analysis and diagnostics"],
            "no_manual_ssh": True, "launcher": "rr deploy + rr run",
        },
        "analysis_artifacts": _artifacts(),
        "synthetic_flip": _synthetic_summary(),
        "compression_diagnostics": _compression_summary(),
        "structure_record": _structure_summary(),
        "seed": int(cfg["recipe"]["seed"]),
        "seeds_planned": list(cfg["budget"]["seeds"]),
        "test_policy": "no official Test labels were read at any point in this round",
    }
    C.write_json(C.results_root() / "run_manifest.json", out)
    print(json.dumps({"gpu_hours_used": gpu_hours, "runs": [r["run_id"] for r in runs],
                      "reused": [r["run_id"] for r in reused]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
