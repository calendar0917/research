"""
run_gps.py — GPS 基线运行入口

通过官方 GraphGPS 代码库运行实验。
支持 paper-seeds-provisional 和 official-seeds 两种协议。

用法:
    # paper-seeds-provisional: ZINC
    python run_gps.py --dataset zinc --protocol paper-seeds-provisional --seeds 0 1 2 3

    # official-seeds: ZINC (use seeds 0-9 per GraphGPS convention)
    python run_gps.py --dataset zinc --protocol official-seeds --seeds 0 1 2 3 4 5 6 7 8 9

    # smoke test: ZINC, 10 batches
    python run_gps.py --dataset zinc --smoke --smoke-batches 10

要求:
    - 已克隆 rampasek/GraphGPS 到 code/vendor/GraphGPS/
    - 已安装环境 graphgps (conda)
    - conda activate graphgps 后运行
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
GPS_DIR = VENDOR_DIR / "GraphGPS"


DATASET_CONFIGS = {
    "zinc": "configs/GPS/zinc-GPS+RWSE.yaml",
    "molhiv": "configs/GPS/ogbg-molhiv-GPS+RWSE.yaml",
    "moltox21": None,  # 不存在于官方仓库
    "peptides-func": "configs/GPS/peptides-func-GPS.yaml",
    "peptides-struct": "configs/GPS/peptides-struct-GPS.yaml",
}


def compute_config_hash(cfg_path, seeds):
    content = f"{cfg_path}:{json.dumps(sorted(seeds))}".encode()
    return hashlib.sha256(content).hexdigest()[:16]


def run_gps(dataset, seeds, protocol, smoke=False, smoke_batches=10):
    cfg_relative = DATASET_CONFIGS.get(dataset)
    if cfg_relative is None:
        print(f"ERROR: No config for {dataset} in GraphGPS official repo")
        print(f"  Try running from HyMN codebase (Southern et al. 2025) instead")
        return

    cfg_path = GPS_DIR / cfg_relative
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}")
        return

    config_hash = compute_config_hash(cfg_relative, seeds)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"GPS: {dataset}, protocol={protocol}, seed={seed}")
        print(f"Config: {cfg_relative}")
        print(f"Config hash: {config_hash}")

        cmd = ["python", "main.py", "--cfg", str(cfg_relative), f"seed={seed}", "wandb.use=False"]

        if smoke:
            cmd.extend(["--smoke", "--smoke_batches", str(smoke_batches)])

        print(f"Command: {' '.join(cmd)}")
        print(f"CWD: {GPS_DIR}")

        try:
            result = subprocess.run(cmd, cwd=GPS_DIR, capture_output=True, text=True, timeout=3600)
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            if result.returncode != 0:
                print(f"STDERR (last 500): {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except FileNotFoundError:
            print("ERROR: GraphGPS not found. Clone it first:")
            print(f"  git clone https://github.com/rampasek/GraphGPS.git {GPS_DIR}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["zinc", "molhiv", "moltox21", "peptides-func", "peptides-struct"])
    parser.add_argument("--protocol", default="paper-seeds-provisional", choices=["paper-seeds-provisional", "official-seeds"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=10)
    args = parser.parse_args()

    if not GPS_DIR.exists():
        print(f"GraphGPS not found at {GPS_DIR}")
        print("Clone it first:")
        print(f"  git clone https://github.com/rampasek/GraphGPS.git {GPS_DIR}")
        sys.exit(1)

    run_gps(args.dataset, args.seeds, args.protocol, args.smoke, args.smoke_batches)


if __name__ == "__main__":
    main()