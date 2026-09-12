"""
collect_gps_results.py — 从 GraphGPS 训练日志提取结果并写入 registry.jsonl

GraphGPS 每 epoch 输出:
    train: {'epoch': N, ..., 'auc': X}
    val:   {...}
    test:  {...}
    > Epoch N: took ... | Best so far: epoch M ...

用法:
    python collect_gps_results.py --log results/paper-seeds/gps-molhiv-seed0.log \
        --method gps --dataset molhiv --protocol paper-seeds-provisional \
        --seed 0 --code-repo https://github.com/rampasek/GraphGPS \
        --code-commit 28015707cbab7f8ad72bed0ee872d068ea59c94b \
        --split-hash feda8af43ac4b55b656ba2e02727d03560e898f3edad160a615797f6ad321179
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "results" / "registry.jsonl"

# 数据集 -> (split_hash, metric 名)
DATASET_META = {
    "molhiv": {"split_hash": "feda8af43ac4b55b656ba2e02727d03560e898f3edad160a615797f6ad321179", "metric": "auc"},
    "moltox21": {"split_hash": "cc8e66d463fa3923b7a8dd769bdd88dcd8c6f88530af0c78cd098381719b8875", "metric": "auc"},
    "molbace": {"split_hash": "b0bfcf9052e25ed1c2365caa7df668bfdca662ee476d66f57ff60c18c7efdd32", "metric": "auc"},
    "zinc-12k": {"split_hash": None, "metric": "mae"},
}


def parse_log(log_path):
    """解析 GraphGPS 日志，返回 (val_best_epoch_info, per_epoch_list)。"""
    epochs = []
    with open(log_path) as f:
        lines = f.readlines()

    cur = {}
    for line in lines:
        m = re.search(r"^(train|val|test):\s*(\{.*\})", line)
        if m:
            split, d = m.group(1), eval(m.group(2))
            cur[split] = d
            if split == "test":
                epochs.append(cur)
                cur = {}
    return epochs


def pick_best(epochs, metric, mode="min"):
    """按验证集最优选 epoch。mode='min' (MAE) 或 'max' (AUC)。"""
    best = None
    best_val = None
    for e in epochs:
        val = e["val"].get(metric)
        if val is None:
            continue
        if best_val is None or (val < best_val if mode == "min" else val > best_val):
            best_val = val
            best = e
    return best, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--code-repo", default="https://github.com/rampasek/GraphGPS")
    parser.add_argument("--code-commit", default="28015707cbab7f8ad72bed0ee872d068ea59c94b")
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--peak-memory-mib", type=int, default=None)
    parser.add_argument("--train-seconds", type=int, default=None)
    parser.add_argument("--status", default="success")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    epochs = parse_log(args.log)
    if not epochs:
        print(f"ERROR: no epochs parsed from {args.log}")
        return

    meta = DATASET_META.get(args.dataset)
    if meta is None:
        print(f"ERROR: unknown dataset {args.dataset}")
        return

    metric = meta["metric"]
    mode = "min" if metric == "mae" else "max"
    best, best_val = pick_best(epochs, metric, mode)
    if best is None:
        print(f"ERROR: no val {metric} found in log")
        return

    config_hash = args.config_hash or hashlib.sha256(
        f"{args.method}:{args.dataset}:{args.seed}".encode()
    ).hexdigest()[:16]

    test_metric = best["test"].get(metric)
    best_epoch = best["test"].get("epoch") or best["val"].get("epoch")
    entry = {
        "method": args.method,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "seed": args.seed,
        "code_repo": args.code_repo,
        "code_commit": args.code_commit,
        "config_hash": config_hash,
        "split_hash": meta["split_hash"],
        "metric_name": "roc_auc" if metric == "auc" else "mae",
        "best_epoch": best_epoch,
        "valid_metric": round(best_val, 6),
        "test_metric": round(test_metric, 6) if test_metric is not None else None,
        "peak_memory_mib": args.peak_memory_mib,
        "train_seconds": args.train_seconds,
        "status": args.status,
        "failure_reason": None,
        "patch_id": "PL-001" if args.dataset in ("molhiv", "moltox21", "molbace") else None,
        "epochs_total": len(epochs),
    }
    entry = {k: v for k, v in entry.items() if v is not None}

    print(json.dumps(entry, indent=2, sort_keys=True))

    if not args.dry_run:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        print(f"Appended to {REGISTRY_PATH}")


if __name__ == "__main__":
    main()