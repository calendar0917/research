#!/usr/bin/env python3
"""本地 CPU 冒烟：用 shim（networkx 版 graph_tool + wandb stub）跑官方 GSN 管线。

目标：在当前机器（无 conda/graph-tool、无 GPU）确认 2020 官方代码在现代栈
（python 3.12 / torch 2.5.1 / PyG 2.6.1 / numpy 2.1.3 / networkx 3.4.2）下
完整可跑：数据加载 -> 结构计数 -> 编码 -> 模型 -> 训练循环。

只跑 IMDBBINARY（1000 个小图）。两种模式：
  --mode pipeline  完整 README 配置（GSN-e, k=5, 局部）会命中官方仓自带的
                    作者预计算 processed/local/complete_graph_5.pt；
                    --force-counts 时先删除缓存，用 shim 重算（验证计数路径）
  --mode counts     GSN-v (k=4, global) 走 shim 计数，验证 vertex 计数 + orbit 路径

结果只做"管线 OK / 数值范围 sanity"判断，不是科研结果。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
COMPAT = TRACK / "code" / "compat"
sys.path.insert(0, str(COMPAT / "wandb_stub"))
sys.path.insert(0, str(COMPAT / "graph_tool_py"))
sys.path.insert(0, str(TRACK / "code"))

from run_official import build_cli, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pipeline", "counts"], default="pipeline")
    ap.add_argument("--config", default=None,
                    help="覆盖模式默认配置（pipeline=IMDBBINARY--gsn-e, counts=IMDBBINARY--gsn-v）")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--force-counts", action="store_true",
                    help="删除缓存用 shim 重算（pipeline 模式）")
    args = ap.parse_args()

    cfg_name = args.config or ("IMDBBINARY--gsn-e" if args.mode == "pipeline"
                               else "IMDBBINARY--gsn-v")
    cfg = load_config(cfg_name)
    name = cfg["dataset_name"]
    vendor = TRACK / "code" / "vendor" / "graph-substructure-networks"
    processed = vendor / "datasets" / "social" / name / "processed"

    if args.force_counts and processed.exists():
        shutil.rmtree(processed)
        print(f"[smoke] purged {processed}")
        shutil.rmtree(vendor / "datasets" / "social" / name / "10fold_idx", ignore_errors=True)

    cli = build_cli(cfg_name, 0, [args.fold], vendor / "datasets")
    # 冒烟覆盖：短训练
    cli = [c for c in cli if c not in ("300", "--num_epochs")]  # 去 num_epochs 对
    cli += ["--num_epochs", str(args.epochs), "--num_iters", str(args.iters),
            "--GPU", "False", "--num_threads", "1"]

    print(f"[smoke] {cfg_name} mode={args.mode} epochs={args.epochs} iters={args.iters}")
    print("[smoke]", " ".join(cli))

    import subprocess
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([str(COMPAT / "wandb_stub"), str(COMPAT / "graph_tool_py")])
    proc = subprocess.run(
        [sys.executable, "main.py", *cli], cwd=vendor, capture_output=True, text=True, env=env)
    print(proc.stdout[-6000:])
    if proc.stderr:
        print("--- stderr ---")
        print(proc.stderr[-2000:])
    ok = proc.returncode == 0 and "Best test mean" in proc.stdout
    print(f"[smoke] {'OK' if ok else 'FAIL'} rc={proc.returncode}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
