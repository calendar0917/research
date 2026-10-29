"""GPU / wall-clock ledger for CJP-SSL v1 (writes ``analysis/gpu_ledger.json``).

Every formal run counts its own wall time as GPU·h (conservative: a run that
shares the host with the other GPU is still charged in full).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cjp_common as C

SUP = {"A_replay_s0", "A_replay_s1", "A_scratch_s0", "A_scratch_s1",
       "A_margssl_s0", "A_margssl_s1", "A_jointssl_s0", "A_jointssl_s1"}
PRE = {"pretrain_marg_s0", "pretrain_marg_s1", "pretrain_joint_s0", "pretrain_joint_s1"}


def collect() -> dict:
    entries = []
    for name in sorted(SUP | PRE):
        d = C.RUNS / name
        if name in SUP:
            path = d / "train.json"
            if not path.exists():
                continue
            p = C.read_json(path)
            entries.append({
                "run": name, "kind": "supervised",
                "arm": p.get("cjp", {}).get("mode", p.get("arm")),
                "seed": p["seed"],
                "wall_seconds": float(p["wall_seconds"]),
                "gpu_hours": float(p["gpu_hours_estimate"]),
                "epochs": p["epochs"], "updates": p["total_updates"],
                "best_valid_mae": p["best"]["valid_mae"],
                "best_epoch": p["best"]["epoch"],
                "checkpoint_sha256": p["checkpoint_sha256"],
                "git_commit": p.get("git_commit"),
                "environment": p.get("environment", {}),
                "device_regime": "res / A100-SXM4-40GB / torch "
                                 + str(p.get("environment", {}).get("torch")),
            })
        else:
            path = d / "pretrain.json"
            if not path.exists():
                continue
            p = C.read_json(path)
            entries.append({
                "run": name, "kind": "pretraining", "arm": p["arm"],
                "seed": p["seed"],
                "wall_seconds": float(p["timing"]["wall_seconds"]),
                "gpu_hours": float(p["timing"]["gpu_hours"]),
                "updates": p["updates_done"],
                "local_state_sha256": p["local_state"]["state_sha256"],
                "local_state_file_sha256": p["local_state"]["sha256_file"],
                "pretext_head_sha256": p["pretext_head"]["sha256_file"],
                "git_commit": p.get("git_commit"),
                "environment": p.get("environment", {}),
                "device_regime": "res / A100-SXM4-40GB / torch "
                                 + str(p.get("environment", {}).get("torch")),
            })
    total_gpu = float(sum(e["gpu_hours"] for e in entries))
    total_wall = float(sum(e["wall_seconds"] for e in entries))
    return {
        "round": C.ROUND_ID,
        "gpu_hard_limit_hours": 16.0,
        "gpu_soft_limit_hours": 10.0,
        "wall_clock_hard_limit_hours": 8.0,
        "entries": entries,
        "n_entries": len(entries),
        "total_gpu_hours": total_gpu,
        "total_serial_wall_seconds": total_wall,
        "within_soft_limit": bool(total_gpu <= 10.0),
        "within_hard_limit": bool(total_gpu <= 16.0),
        "official_test_loaded": False,
        "git_commit": C.git_commit(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CJP-SSL GPU ledger")
    args = parser.parse_args(argv)
    payload = collect()
    C.write_json(C.ANALYSIS / "gpu_ledger.json", payload)
    C.log(f"entries={payload['n_entries']} total_gpu_hours={payload['total_gpu_hours']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
