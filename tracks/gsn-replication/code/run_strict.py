#!/usr/bin/env python3
"""GSN 严格协议 runner（10 seeds × 10×10 CV，val 选 epoch，test 无泄漏）。

背景
----
官方评估（gsn-official-social-v1）是**乐观协议**：官方 main.py 在没有验证集时用
`argmax(test_accs)` 选 epoch（main.py:333），即在测试集上做模型选择。这属于测试
泄漏，数字会偏高。

本脚本实现**严格协议 gsn-strict-social-v1**：
  - 划分：RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=seed)，
    与 tracks/wl-subtree-kernel 同 API/同顺序（同种子→同划分，可横比）；
    REDDIT-BINARY 无官方 10fold_idx，也用随机划分 → 全数据集协议统一。
  - 每折：train（90%）内再分层切出 val（10%），写入官方代码约定的
    10fold_idx/{train,test,val}_idx-{fold+1}.txt → 官方 `--split given` 自动
    读取 val 并用 val 选 epoch（main.py:333,398）。
  - 每折只报告 val-best epoch 的 test acc（官方 "Best test mean" 输出）。
  - 超参固定为论文 Table 5（方法给定实现；严格化的对象是"评估选择"，不是重做
    超参搜索——见 notes/README.md）。

用法
----
  python code/run_strict.py --config IMDBBINARY--gsn-e          # 全 10 seeds × 10×10
  python code/run_strict.py --config IMDBBINARY--gsn-e --seeds 0 --n-repeats 1  # 冒烟
  python code/run_strict.py --config IMDBBINARY--gsn-e --resume # 跳过已完成的 fold

输出
----
  results/strict/<config>/seed<SEED>_r<REP>f<FOLD>.json   每 fold 明细
  results/strict/<config>/summary.json                    per-seed + across-seed 聚合
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

TRACK = Path(__file__).resolve().parents[1]
VENDOR = TRACK / "code" / "vendor"
OFFICIAL = VENDOR / "graph-substructure-networks"
DATA_ROOT = TRACK / "data"          # gitignored（data/**）
RESULTS = TRACK / "results" / "strict"
CONFIGS = json.loads((TRACK / "configs" / "social_paper.json").read_text(encoding="utf-8"))

DEFAULT_SEEDS = [0, 41, 123, 1024, 2026, 777, 3407, 999, 111, 888]
PAT_BEST = re.compile(r"Best test mean: ([0-9.]+) \+/- ([0-9.]+)")

sys.path.insert(0, str(TRACK / "code"))
from run_official import build_cli  # noqa: E402   (复用 CLI 组装)


def load_labels(name: str, path: Path) -> np.ndarray:
    """powerful-gnns 格式 txt -> 图标签（顺序即图索引）。"""
    lines = (path / f"{name}.txt").read_text().splitlines()
    n_graphs = int(lines[0])
    labels, i = [], 1
    for _ in range(n_graphs):
        hdr = lines[i].split()
        labels.append(int(hdr[1]))
        i += 1 + int(hdr[0])
    assert len(labels) == n_graphs, f"{name}: {len(labels)} != {n_graphs}"
    return np.array(labels, dtype=np.int64)


def val_tag(seed: int, rep: int, fold: int) -> int:
    return seed * 100000 + rep * 1000 + fold


def write_split_files(data_dir: Path, labels: np.ndarray, seed: int, rep: int,
                      fold: int, n_splits: int, n_repeats: int) -> tuple[np.ndarray, np.ndarray]:
    sklearn = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=seed)
    tr_all, te = list(sklearn.split(np.zeros(len(labels)), labels))[rep * n_splits + fold]
    # 严格协议：val 从 train 分层切出 10%
    tr, va = train_test_split(
        tr_all, test_size=0.1, stratify=labels[tr_all],
        random_state=val_tag(seed, rep, fold))
    idx_dir = data_dir / "10fold_idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    k = fold + 1
    for tag, idx in (("train", tr), ("test", te), ("val", va)):
        (idx_dir / f"{tag}_idx-{k}.txt").write_text(
            "\n".join(str(int(i)) for i in np.sort(idx)) + "\n")
    return tr, te


def set_flag(cli: list[str], flag: str, value: str) -> list[str]:
    """删除已有的 --flag value 对，追加新值（argparse 后者优先）。"""
    out = []
    skip = False
    for tok in cli:
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = True
            continue
        out.append(tok)
    return out + [flag, value]


def run_fold(cfg_name: str, seed: int, rep: int, fold: int, extra_env: str | None,
             args: dict) -> dict:
    """单个 (seed, repeat, fold)：写划分 → 调官方 main.py → 解析 test acc@val-best。"""
    cfg = CONFIGS[cfg_name]
    name = cfg["dataset_name"]
    data_dir = DATA_ROOT / "social" / name

    labels = load_labels(name, data_dir)
    write_split_files(data_dir, labels, seed, rep, fold,
                      args["n_splits"], args["n_repeats"])

    cli = build_cli(cfg_name, seed, [fold], DATA_ROOT)
    # strict 覆盖：删掉 REDDIT 配置里的 --split random / --split_seed，强制 given
    out = []
    skip_next = False
    for tok in cli:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--split", "--split_seed"):
            skip_next = True
            continue
        out.append(tok)
    cli = out + ["--split", "given",
                 "--results_folder", f"strict-s{seed}-r{rep}-f{fold}",
                 "--checkpoint_file", "ckpt"]
    if args.get("num_epochs"):
        cli = set_flag(cli, "--num_epochs", str(args["num_epochs"]))
    if args.get("num_iters"):
        cli = set_flag(cli, "--num_iters", str(args["num_iters"]))
    if args.get("cpu"):
        cli = set_flag(cli, "--GPU", "False")
    if args.get("device_idx") is not None:
        cli = set_flag(cli, "--device_idx", str(args["device_idx"]))

    env = dict(__import__("os").environ)
    if extra_env:
        env["PYTHONPATH"] = extra_env
    proc = subprocess.run([sys.executable, "main.py", *cli], cwd=OFFICIAL,
                          capture_output=True, text=True, env=env,
                          timeout=args.get("timeout", 3600 * 6))
    m = PAT_BEST.search(proc.stdout)
    if not m or proc.returncode != 0:
        return {"config": cfg_name, "seed": seed, "rep": rep, "fold": fold,
                "test_acc": None, "rc": proc.returncode,
                "stderr_tail": proc.stderr[-800:], "cmd": cli}
    # 清理每 epoch 的 checkpoint（磁盘；--keep-checkpoints 保留）
    if not args.get("keep_checkpoints"):
        import shutil
        shutil.rmtree(data_dir / "results" / f"strict-s{seed}-r{rep}-f{fold}",
                      ignore_errors=True)
    return {"config": cfg_name, "seed": seed, "rep": rep, "fold": fold,
            "test_acc": float(m.group(1)),
            "elapsed_s": None, "cmd": cli}


def setup_data_dir(config_name: str, purge_processed: bool) -> None:
    """data/social/<NAME>/：txt 复用 vendor；processed 从 vendor 复制（或重算）。"""
    import shutil
    cfg = CONFIGS[config_name]
    name = cfg["dataset_name"]
    src = OFFICIAL / "datasets" / "social" / name
    dst = DATA_ROOT / "social" / name
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / f"{name}.txt").exists():
        shutil.copy2(src / f"{name}.txt", dst / f"{name}.txt")
    if purge_processed and (dst / "processed").exists():
        shutil.rmtree(dst / "processed")
    if not (dst / "processed").exists() and (src / "processed").exists():
        shutil.copytree(src / "processed", dst / "processed")
        print(f"[setup] copied processed/ from vendor for {name}")
    else:
        print(f"[setup] {name}: processed/ 将按需由官方代码生成（graph-tool）")
    # 作者缓存是 PyG1.4 pickle：在 data/ 副本里迁移，不动 vendor
    if (dst / "processed").exists():
        from setup_data import migrate_legacy_processed
        migrate_legacy_processed(dst / "processed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--n-splits", type=int, default=10, help="折数（默认 10）")
    ap.add_argument("--n-repeats", type=int, default=10, help="重复次数（默认 10=十次十折）")
    ap.add_argument("--num-epochs", type=int, default=None, help="覆盖官方 300（冒烟用）")
    ap.add_argument("--num-iters", type=int, default=None, help="覆盖官方 50（冒烟用）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行度（多 GPU 服务器建议 = GPU 数；默认串行）")
    ap.add_argument("--device-idx", type=int, default=0)
    ap.add_argument("--devices", type=int, default=1,
                    help="workers 模式下 GPU 轮转数量（--workers 4 --devices 2 → 每 GPU 2 进程）")
    ap.add_argument("--cpu", action="store_true", help="本机无 GPU 时冒烟用（追加 --GPU False）")
    ap.add_argument("--purge-processed", action="store_true")
    ap.add_argument("--resume", action="store_true", help="跳过已完成的 fold")
    ap.add_argument("--force", action="store_true", help="重跑全部")
    ap.add_argument("--timeout", type=int, default=3600 * 6)
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="保留每 fold 的 checkpoint（默认清理，防磁盘爆炸）")
    ap.add_argument("--pythonpath", type=str, default=None,
                    help="额外 PYTHONPATH（本地冒烟 shim 用；服务器不需要）")
    args = ap.parse_args()

    setup_data_dir(args.config, args.purge_processed)

    if args.pythonpath:
        args.pythonpath = ":".join(
            str(Path(p).resolve()) for p in args.pythonpath.split(":"))
        print(f"[strict] pythonpath: {args.pythonpath}")

    out_dir = RESULTS / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(s, r, f) for s in args.seeds for r in range(args.n_repeats)
             for f in range(args.n_splits)]
    print(f"[strict] {args.config}: {len(args.seeds)} seeds x {args.n_repeats}x"
          f"{args.n_splits} = {len(tasks)} folds")

    run_kwargs = {"n_splits": args.n_splits, "n_repeats": args.n_repeats,
                  "num_epochs": args.num_epochs, "num_iters": args.num_iters,
                  "timeout": args.timeout, "device_idx": args.device_idx,
                  "cpu": args.cpu, "keep_checkpoints": args.keep_checkpoints}

    def need(t):
        s, r, f = t
        p = out_dir / f"seed{s}_r{r}f{f}.json"
        return not (p.exists() and not args.force)

    todo = [t for t in tasks if need(t)]
    skipped = len(tasks) - len(todo)
    if skipped:
        print(f"[strict] 跳过 {skipped} 个已完成 fold（--force 重跑）")

    t_start = time.time()
    done = []
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for i, t in enumerate(todo):
                kw = dict(run_kwargs, device_idx=i % args.devices)
                futs[ex.submit(run_fold, args.config, *t, args.pythonpath, kw)] = t
            for i, fut in enumerate(futs):
                res = fut.result()
                done.append(res)
                if res.get("test_acc") is not None:
                    print(f"  [{i+1}/{len(todo)}] s{res['seed']} r{res['rep']} "
                          f"f{res['fold']}: {res['test_acc']:.4f}")
                else:
                    print(f"  [{i+1}/{len(todo)}] s{res['seed']} r{res['rep']} "
                          f"f{res['fold']}: FAIL rc={res.get('rc')}")
                    print("     stderr:", (res.get("stderr_tail") or "")[-500:])
                    print("     cmd tail:", " ".join(res.get("cmd", [])[-15:]))
    else:
        for i, t in enumerate(todo):
            res = run_fold(args.config, *t, args.pythonpath, run_kwargs)
            done.append(res)
            if res.get("test_acc") is not None:
                print(f"  [{i+1}/{len(todo)}] s{res['seed']} r{res['rep']} "
                      f"f{res['fold']}: {res['test_acc']:.4f}")
            else:
                print(f"  [{i+1}/{len(todo)}] s{res['seed']} r{res['rep']} "
                      f"f{res['fold']}: FAIL rc={res.get('rc')}")
                print("     stderr:", (res.get("stderr_tail") or "")[-500:])
                print("     cmd tail:", " ".join(res.get("cmd", [])[-15:]))

    for res in done:
        if res.get("test_acc") is not None:
            (out_dir / f"seed{res['seed']}_r{res['rep']}f{res['fold']}.json").write_text(
                json.dumps(res, indent=2, default=str))

    # ---- 聚合 ---------------------------------------------------------------- #
    accs = [json.loads(p.read_text()) for p in out_dir.glob("seed*_r*f*.json")]
    accs = [a for a in accs if a.get("test_acc") is not None]
    if not accs:
        print("[strict] 没有任何成功 fold；请检查日志")
        sys.exit(1)
    per_seed = []
    for s in sorted({a["seed"] for a in accs}):
        v = np.array([a["test_acc"] for a in accs if a["seed"] == s])
        per_seed.append({"seed": s, "n_folds": len(v),
                         "mean": float(v.mean()), "std": float(v.std())})
    allv = np.array([a["test_acc"] for a in accs])
    means = np.array([p["mean"] for p in per_seed])
    summary = {
        "protocol_id": "gsn-strict-social-v1",
        "config": args.config,
        "dataset": CONFIGS[args.config]["dataset_name"],
        "n_splits": args.n_splits, "n_repeats": args.n_repeats,
        "seeds": args.seeds,
        "n_folds_total": int(len(allv)),
        "fold_level": {"mean": float(allv.mean()), "std": float(allv.std())},
        "seed_level": {"mean_of_means": float(means.mean()),
                       "std_of_means": float(means.std()),
                       "mean_of_fold_std": float(np.mean([p["std"] for p in per_seed]))},
        "per_seed": per_seed,
        "elapsed_total_s": round(time.time() - t_start, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[strict] {args.config}: {summary['fold_level']['mean']*100:.2f}±"
          f"{summary['fold_level']['std']*100:.2f} (fold-level, n={len(allv)})")
    print(f"[strict] seed-level: {summary['seed_level']['mean_of_means']*100:.2f}±"
          f"{summary['seed_level']['std_of_means']*100:.2f}")
    print(f"[strict] saved -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
