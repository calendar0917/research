#!/usr/bin/env python3
"""GSN 官方复现：数据布局准备与校验。

目标布局（官方代码硬编码约定）:
    vendor/graph-substructure-networks/datasets/social/<NAME>/
        <NAME>.txt                 powerful-gnns raw（IMDBBINARY 用官方仓自带）
        10fold_idx/train_idx-{1..10}.txt, test_idx-{1..10}.txt

来源: vendor/powerful-gnns/dataset/<NAME>/（本仓已 clone: 9a2ce8a）
校验: IMDBBINARY 与官方仓 datasets/social/IMDBBINARY 逐字节一致(md5)。

用法:
    uv run python code/setup_data.py                # 校验 + 补齐
    uv run python code/setup_data.py --purge-processed
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
VENDOR = TRACK / "code" / "vendor"
OFFICIAL = VENDOR / "graph-substructure-networks"
POWERFUL = VENDOR / "powerful-gnns"

DATASETS = ["IMDBBINARY", "IMDBMULTI", "COLLAB", "REDDITBINARY"]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def ensure_layout(purge_processed: bool = False) -> None:
    """校验/补齐 4 个数据集到官方仓 datasets/social/ 布局。"""
    if not OFFICIAL.exists():
        raise SystemExit(f"官方仓缺失: {OFFICIAL} （git-ignored vendor，需重新 clone）")

    for name in DATASETS:
        dst = OFFICIAL / "datasets" / "social" / name
        src = POWERFUL / "dataset" / name
        if name == "IMDBBINARY" and (dst / "IMDBBINARY.txt").exists():
            a, b = md5(dst / "IMDBBINARY.txt"), md5(src / "IMDBBINARY.txt")
            assert a == b, f"IMDBBINARY.txt mismatch {a} vs {b}"
            print(f"[ok] {name}: official raw == powerful-gnns raw ({a[:12]}…)")
            continue
        if not src.exists():
            raise SystemExit(f"missing source {src} （先解压 powerful-gnns dataset.zip）")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f"{name}.txt", dst / f"{name}.txt")
        if (src / "10fold_idx").exists():
            shutil.copytree(src / "10fold_idx", dst / "10fold_idx", dirs_exist_ok=True)
            n = len(list((dst / "10fold_idx").glob("train_idx-*.txt")))
            assert n >= 10, f"{name} 10fold_idx 不完整: {n} 个 train 文件"
        else:
            print(f"[warn] {name}: powerful-gnns 无 10fold_idx（→ 使用 --split random 协议）")
        print(f"[ok] {name}: {src} -> {dst}")

    if purge_processed and (OFFICIAL / "datasets/social/IMDBBINARY/processed").exists():
        shutil.rmtree(OFFICIAL / "datasets/social/IMDBBINARY/processed")
        print("[warn] 已删除作者预计算 processed；首次运行将用 graph-tool 重算")

    # sanity: 各数据集 (train+test) 索引数量 = 图数（REDDITBINARY 无 10fold_idx，用 random split，跳过）
    for name in DATASETS:
        d = OFFICIAL / "datasets" / "social" / name
        n_graphs = int((d / f"{name}.txt").read_text().splitlines()[0])
        f1 = d / "10fold_idx/train_idx-1.txt"
        if not f1.exists():
            print(f"[ok] {name}: {n_graphs} graphs（无 10fold_idx → --split random 协议）")
            continue
        n_tr = len(f1.read_text().split())
        n_te = len((d / "10fold_idx/test_idx-1.txt").read_text().split())
        assert n_tr + n_te == n_graphs, f"{name}: {n_tr}+{n_te} != {n_graphs}"
        print(f"[ok] {name}: {n_graphs} graphs, fold1 train+test = {n_tr}+{n_te}")

    print("data layout ready.")


def migrate_legacy_processed(proc_dir: Path | None = None) -> None:
    """PyG 1.4 pickle -> PyG 2.x Data：作者预计算缓存（若未迁移且未 purge）。

    官方仓/数据副本里的 processed/*.pt 用 PyG1.4 保存；PyG 2.6 直接 torch.load
    会抛 'older version of PyG'。这里把每个旧 Data 转换为新 Data（仅复制 tensor
    属性），并原址回写。已迁移（新 Data 含 _store）的文件自动跳过。
    """
    import torch
    from torch_geometric.data import Data

    proc = proc_dir or (OFFICIAL / "datasets" / "social" / "IMDBBINARY" / "processed")
    if not proc.exists():
        print("[migrate] no processed dir to migrate")
        return
    for pt in sorted(proc.rglob("*.pt")):
        print(f"[migrate] {pt}")
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        graphs, rest = obj[0], obj[1:]
        if len(graphs) and "_store" in graphs[0].__dict__:
            print("[migrate]   already PyG2 format, skip")
            continue
        new = []
        for g in graphs:
            d = Data()
            for k, v in g.__dict__.items():
                if isinstance(v, torch.Tensor):
                    setattr(d, k, v)
            if not hasattr(d, "y") and hasattr(g, "label"):
                import torch as _t
                setattr(d, "y", _t.tensor([int(g.label)]))
            new.append(d)
        torch.save((new, *rest), pt)
    print("[migrate] done")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--purge-processed", action="store_true",
                    help="删除官方仓 IMDBBINARY/processed 以便服务器用 graph-tool 重算")
    ap.add_argument("--migrate-processed", action="store_true",
                    help="作者预计算缓存从 PyG1.4 pickle 迁移到 PyG2.x Data 对象")
    args = ap.parse_args()
    ensure_layout(purge_processed=args.purge_processed)
    if args.migrate_processed:
        migrate_legacy_processed()


if __name__ == "__main__":
    main()
