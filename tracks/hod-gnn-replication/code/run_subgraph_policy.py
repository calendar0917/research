"""
run_subgraph_policy.py — Full/Random/Policy-Learn 基线运行入口

通过官方 policy-learn 代码库运行实验。
三种策略共用同一代码库，通过 selection_type 切换:
  - all (Full)
  - random (Random)
  - gumbel (Policy-Learn)

用法:
    # Full
    python run_subgraph_policy.py --policy full --dataset zinc --protocol paper-seeds-provisional --seeds 1 2 3 4

    # Random
    python run_subgraph_policy.py --policy random --dataset zinc --protocol paper-seeds-provisional --seeds 1 2 3 4

    # Policy-Learn
    python run_subgraph_policy.py --policy policy_learn --dataset zinc --protocol paper-seeds-provisional --seeds 1 2 3 4

    # smoke test
    python run_subgraph_policy.py --policy policy_learn --dataset zinc --smoke

要求:
    - 已克隆 beabevi/policy-learn 到 code/vendor/policy-learn/
    - 已安装依赖 (docker 或 conda)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "code" / "vendor"
POLICY_DIR = VENDOR_DIR / "policy-learn"


POLICY_TYPE_MAP = {
    "full": "all",
    "random": "random",
    "policy_learn": "gumbel",
}

DATASET_YAML_MAP = {
    "zinc": "zinc",
    "molhiv": "molhiv",
    "moltox21": "moltox21",
    "molbace": "molbace",
}


def compute_config_hash(policy, dataset, seeds):
    content = f"policy_learn:{policy}:{dataset}:{json.dumps(sorted(seeds))}".encode()
    return hashlib.sha256(content).hexdigest()[:16]


def run_policy(policy, dataset, seeds, protocol, smoke=False):
    selection_type = POLICY_TYPE_MAP.get(policy)
    if selection_type is None:
        print(f"ERROR: Unknown policy: {policy}")
        return

    dataset_key = DATASET_YAML_MAP.get(dataset)
    if dataset_key is None:
        print(f"ERROR: No config for {dataset} in policy-learn official repo")
        print(f"  Try running from HyMN codebase (Southern et al. 2025) instead")
        return

    config_hash = compute_config_hash(policy, dataset, seeds)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Policy-Learn ({policy}): {dataset}, protocol={protocol}, seed={seed}")
        print(f"Config: {dataset_key}-{selection_type}.yaml")
        print(f"Config hash: {config_hash}")

        cmd = [
            "python", "train.py",
            f"dataset={dataset_key}",
            f"selection_type={selection_type}",
            f"seed={seed}",
        ]

        if smoke:
            cmd.append("epochs=2")

        print(f"Command: {' '.join(cmd)}")
        print(f"CWD: {POLICY_DIR}")

        try:
            result = subprocess.run(cmd, cwd=POLICY_DIR, capture_output=True, text=True, timeout=7200)
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            if result.returncode != 0:
                print(f"STDERR (last 500): {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except FileNotFoundError:
            print("ERROR: policy-learn not found. Clone it first:")
            print(f"  git clone https://github.com/beabevi/policy-learn.git {POLICY_DIR}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["full", "random", "policy_learn"])
    parser.add_argument("--dataset", required=True, choices=["zinc", "molhiv", "moltox21", "molbace"])
    parser.add_argument("--protocol", default="paper-seeds-provisional", choices=["paper-seeds-provisional", "official-seeds"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not POLICY_DIR.exists():
        print(f"policy-learn not found at {POLICY_DIR}")
        print("Clone it first:")
        print(f"  git clone https://github.com/beabevi/policy-learn.git {POLICY_DIR}")
        sys.exit(1)

    run_policy(args.policy, args.dataset, args.seeds, args.protocol, args.smoke)


if __name__ == "__main__":
    main()