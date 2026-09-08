#!/usr/bin/env python3
"""GSN 官方复现 runner（服务器/本地通用）。

对给定 {数据集}--{变体} 配置：
  1. 数据布局检查（setup_data）
  2. 由 configs/social_paper.json 组装官方 main.py 的 CLI（官方代码零改动）
  3. subprocess 运行（cwd = vendor/graph-substructure-networks）
  4. 解析 stdout 的论文指标: "Best test mean: X +/- Y"（10 折均值最好的 epoch 的 10 折 mean±std）
  5. 结果 JSON 落 tracks/gsn-replication/results/

用法:
  uv run --with wandb python code/run_official.py --config IMDBBINARY--gsn-e --seed 0
  uv run python code/run_official.py --config COLLAB--gsn-v --folds 0,1,2 --seed 0 --dry-run

服务器上：conda 环境见 env/，无 wandb 时不需要 --with wandb（wandb=False 路径）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
VENDOR = TRACK / "code" / "vendor"
OFFICIAL = VENDOR / "graph-substructure-networks"
CONFIGS = json.loads((TRACK / "configs" / "social_paper.json").read_text(encoding="utf-8"))
RESULTS = TRACK / "results"

PAT_BEST = re.compile(r"Best test mean: ([0-9.]+) \+/- ([0-9.]+)")
PAT_PARAMS = re.compile(r"Total number of parameters is: (\d+)")


def load_config(cfg_name: str) -> dict:
    if cfg_name not in CONFIGS:
        raise SystemExit(f"unknown config {cfg_name}; available: "
                         + ", ".join(k for k in CONFIGS if "--" in k))
    return CONFIGS[cfg_name]


def build_cli(cfg_name: str, seed: int, folds: list[int], datasets_root: Path) -> list[str]:
    cfg = load_config(cfg_name)
    name = cfg["dataset_name"]
    base: list[str] = [str(x) for x in CONFIGS["_base_cli"]]
    # 根目录替换
    argv = []
    for i, tok in enumerate(base):
        if tok == "DATASETS_ROOT":
            argv.append(str(datasets_root))
        else:
            argv.append(tok)
    # 数据集级
    argv += ["--seed", str(seed), "--dataset_name", name,
             "--fold_idx", ",".join(str(f) for f in folds)]
    if name in CONFIGS["_decay_steps"]:
        argv += ["--decay_steps", str(CONFIGS["_decay_steps"][name])]
    if name in CONFIGS["_lr"]:
        argv += ["--lr", CONFIGS["_lr"][name]]
    # 配置级
    for k, v in cfg["overrides"].items():
        argv += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    for tok in cfg.get("extra_cli", []):
        argv.append(str(tok))
    return argv


def run(args: argparse.Namespace, seed: int | None = None) -> dict:
    cfg = load_config(args.config)
    name = cfg["dataset_name"]
    seed = args.seed if seed is None else seed
    folds = args.folds
    datasets_root = OFFICIAL / "datasets"
    cli = build_cli(args.config, seed, folds, datasets_root)

    out_path = RESULTS / f"{args.config}__seed{seed}.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} 已存在（--force 重跑）")
        return {}

    if args.purge_processed and (OFFICIAL / "datasets" / "social" / name /
                                 "processed").exists():
        import shutil
        shutil.rmtree(OFFICIAL / "datasets" / "social" / name / "processed")
        print(f"[info] purged processed/ for {name}")

    if args.dry_run:
        print(" ".join(cli))
        return {}

    env = dict(os.environ)
    if args.pythonpath_extra:
        extras = [str(Path(p).resolve()) for p in str(args.pythonpath_extra).split(":")]
        env["PYTHONPATH"] = os.pathsep.join(extras + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    print(f"[info] running: {' '.join(cli[:12])} ... (cwd={OFFICIAL})")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "main.py", *cli],
        cwd=OFFICIAL, env=env, capture_output=True, text=True, timeout=args.timeout,
    )
    elapsed = time.time() - t0
    out = proc.stdout
    err = proc.stderr
    print(out[-4000:])
    if err:
        print("--- stderr tail ---")
        print(err[-2000:])

    metrics = {}
    m = PAT_BEST.search(out)
    if m:
        metrics["best_test_mean"] = float(m.group(1))
        metrics["best_test_std"] = float(m.group(2))
    mp = PAT_PARAMS.search(out)
    if mp:
        metrics["num_params"] = int(mp.group(1))
    if proc.returncode != 0:
        metrics["returncode"] = proc.returncode
        if not metrics:
            raise SystemExit(f"run failed rc={proc.returncode}; see stderr above")

    config_sha = hashlib.sha256(
        json.dumps({"cfg": cfg, "seed": seed, "folds": folds}, sort_keys=True).encode()
    ).hexdigest()[:12]
    result = {
        "protocol_id": "gsn-official-social-v1",
        "track": "gsn-replication",
        "config": args.config,
        "dataset": name,
        "variant": cfg["variant"],
        "seed": seed,
        "folds": folds,
        "in_paper": cfg["in_paper"],
        "expected_paper": cfg.get("expected_paper"),
        "metric_semantics": "10-fold mean of test acc at epoch with best mean across folds",
        "official_repo_head": "6cce24a2c0f59c183c388f3016d33502f63e8175",
        "config_sha256": config_sha,
        "python": platform.python_version(),
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics,
        "cmd": cli,
    }

    if metrics.get("best_test_mean") is not None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n[ok] {args.config} seed={seed}: "
              f"best_test_mean={metrics['best_test_mean']:.4f} "
              f"± {metrics['best_test_std']:.4f}  -> {out_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="configs/social_paper.json 里的键，如 IMDBBINARY--gsn-e")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--purge-processed", action="store_true",
                    help="运行前删除该数据集 processed/ 缓存（graph-tool 重算）")
    ap.add_argument("--force", action="store_true", help="忽略已存在的结果 JSON")
    ap.add_argument("--dry-run", action="store_true", help="只打印 CLI 不运行")
    ap.add_argument("--timeout", type=int, default=3600 * 12)
    ap.add_argument("--pythonpath-extra", type=Path, default=None,
                    help="冒烟脚本用：前置 shim 目录（graph_tool/wandb）")
    args = ap.parse_args()

    # 数据布局（只校验/补齐）
    sys.path.insert(0, str(TRACK / "code"))
    import setup_data
    setup_data.ensure_layout(purge_processed=args.purge_processed)

    for seed in args.seeds:
        run(args, seed=seed)


if __name__ == "__main__":
    main()
