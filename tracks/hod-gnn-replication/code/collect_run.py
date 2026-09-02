"""
collect_run.py — 单次运行结果收集和 registry 写入

解析训练日志，提取关键指标，写入 results/registry.jsonl。

用法:
    python collect_run.py --method gps --dataset zinc --protocol paper-seeds-provisional \\
        --seed 0 --config-hash abc123 --split-hash def456 \\
        --log-dir /path/to/logs --code-repo https://github.com/rampasek/GraphGPS \\
        --code-commit 2801570 --best-epoch 42 --valid-metric 0.071 --test-metric 0.069 \\
        --peak-memory 12345 --train-seconds 3600 --status success
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "results" / "registry.jsonl"


def init_registry():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.touch()


def record_entry(entry):
    init_registry()
    entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"Recorded: {entry['method']}/{entry['dataset']}/{entry['protocol']}/seed={entry['seed']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["gps", "graphvit", "full", "random", "policy_learn"])
    parser.add_argument("--dataset", required=True, choices=["zinc-12k", "moltox21", "molbace", "molhiv", "peptides-func", "peptides-struct"])
    parser.add_argument("--protocol", required=True, choices=["paper-seeds-provisional", "official-seeds"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--code-repo", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--split-hash", required=True)
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--best-epoch", type=int, default=None)
    parser.add_argument("--valid-metric", type=float, default=None)
    parser.add_argument("--test-metric", type=float, required=True)
    parser.add_argument("--peak-memory-mib", type=int, default=None)
    parser.add_argument("--train-seconds", type=int, default=None)
    parser.add_argument("--status", default="success", choices=["success", "oom", "runtime_error", "data_error", "unavailable", "implementation_drift"])
    parser.add_argument("--failure-reason", default=None)
    parser.add_argument("--patch-id", default=None)
    args = parser.parse_args()

    entry = {
        "method": args.method,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "seed": args.seed,
        "sampler_seed": args.sampler_seed,
        "code_repo": args.code_repo,
        "code_commit": args.code_commit,
        "config_hash": args.config_hash,
        "split_hash": args.split_hash,
        "metric_name": args.metric_name,
        "best_epoch": args.best_epoch,
        "valid_metric": args.valid_metric,
        "test_metric": args.test_metric,
        "peak_memory_mib": args.peak_memory_mib,
        "train_seconds": args.train_seconds,
        "status": args.status,
        "failure_reason": args.failure_reason,
        "patch_id": args.patch_id,
    }

    entry = {k: v for k, v in entry.items() if v is not None}
    record_entry(entry)


if __name__ == "__main__":
    main()