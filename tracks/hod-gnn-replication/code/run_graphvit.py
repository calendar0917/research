"""
run_graphvit.py — GraphViT 基线运行入口

通过官方 Graph-ViT-MLPMixer 代码库运行实验。

用法:
    # paper-seeds-provisional
    python run_graphvit.py --dataset zinc --protocol paper-seeds-provisional --seeds 0 1 2 3

    # official-seeds
    python run_graphvit.py --dataset zinc --protocol official-seeds --seeds 0 1 2 3

    # smoke test
    python run_graphvit.py --dataset zinc --smoke

要求:
    - 已克隆 XiaoxinHe/Graph-ViT-MLPMixer 到 code/vendor/GraphViT/
    - 已安装环境 graph_mlpmixer (conda)
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
VIT_DIR = VENDOR_DIR / "GraphViT"


DATASET_MODULES = {
    "zinc": "train.zinc",
    "molhiv": "train.molhiv",
    "moltox21": "train.moltox21",
    "peptides-func": "train.peptides_func",
    "peptides-struct": "train.peptides_struct",
}


def compute_config_hash(dataset, seeds):
    content = f"graphvit:{dataset}:{json.dumps(sorted(seeds))}".encode()
    return hashlib.sha256(content).hexdigest()[:16]


def run_graphvit(dataset, seeds, protocol, smoke=False):
    module = DATASET_MODULES.get(dataset)
    if module is None:
        print(f"ERROR: No dataset module for {dataset}")
        return

    config_hash = compute_config_hash(dataset, seeds)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"GraphViT: {dataset}, protocol={protocol}, seed={seed}")
        print(f"Config hash: {config_hash}")

        cmd = ["python", "-m", module, f"cfg.seed={seed}"]

        if smoke:
            cmd.append("cfg.train.epochs=2")

        print(f"Command: {' '.join(cmd)}")
        print(f"CWD: {VIT_DIR}")

        try:
            result = subprocess.run(cmd, cwd=VIT_DIR, capture_output=True, text=True, timeout=7200)
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            if result.returncode != 0:
                print(f"STDERR (last 500): {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except FileNotFoundError:
            print("ERROR: GraphViT not found. Clone it first:")
            print(f"  git clone https://github.com/XiaoxinHe/Graph-ViT-MLPMixer.git {VIT_DIR}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["zinc", "molhiv", "moltox21", "peptides-func", "peptides-struct"])
    parser.add_argument("--protocol", default="paper-seeds-provisional", choices=["paper-seeds-provisional", "official-seeds"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not VIT_DIR.exists():
        print(f"GraphViT not found at {VIT_DIR}")
        print("Clone it first:")
        print(f"  git clone https://github.com/XiaoxinHe/Graph-ViT-MLPMixer.git {VIT_DIR}")
        sys.exit(1)

    run_graphvit(args.dataset, args.seeds, args.protocol, args.smoke)


if __name__ == "__main__":
    main()