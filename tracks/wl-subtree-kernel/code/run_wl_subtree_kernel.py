#!/usr/bin/env python3
"""WL-Subtree-Kernel 实验（v2：修复 TU bioinformatic 节点标签 + 四档协议）。

v2 相对 v1 的变更（2026-09-08+）
--------------------------------
* 节点标签不再由隐式全局 _DS_NAME 决定，显式 `pyg_to_grakel(data, dataset_name)`：
  - REDDIT*  → constant 标签（全 0）
  - BIO（MUTAG/PROTEINS/DD/NCI1/ENZYMES）→ TUDataset 原始 categorical node labels
    （`TUDataset(..., use_node_attr=False)`，data.x 为 one-hot，argmax 还原类别）
  - IMDB-* / COLLAB 等 → degree
* 超参数搜索范围：h ∈ {1..6}、C ∈ {0.001..1000}（7 档对数网格）。
  注意：GIN 论文未公开其 C grid 的确切取值，本网格只是避免最优解撞边界，
  不是 GIN 原始 C grid。
* nested 内层默认 cv5（`--inner-strategy cv5`）；holdout 保留为可选项，
  `--inner-val-frac` 默认 1/9（相对 outer-train 90%），即 overall 80/10/10
  （修正 v1 的 72/18/10）。
* 新增 paperlike 协议（`--protocol paperlike`）：non-nested CV model selection，
  同一组 folds 先按 mean CV acc 选 (h*, C*)，再在该组 folds 上报告 theta* 的
  mean±std —— 有 selection bias，标为 "paper-like approximation"，不是 exact GIN。
* JSON metadata 统一记录 node_label_mode / normalize / kernel / base_kernel。

来源与保真（v1 沿用）
---------------------
原始实现（旧 keyan 仓 phase2_structure_classification，备份于
~/life/archive/科研/keyan.zip）: src/models.py::WLKernelClassifier +
experiments/archive/run_step2.py（Method D）。本文件保持原语义：
grakel WeisfeilerLehman(n_iter=3, normalize=True)（base 为 VertexHistogram）+
sklearn SVC(kernel='precomputed', C=10)。等价优化：全量核矩阵一次计算 + 子矩阵切片
（normalize=True 为余弦归一化，diagonal 与图集无关 → 与逐 fold fit_transform 逐位一致，
MUTAG 上验证 max|Δ|=0.0）。

四档协议
--------
  --protocol strict        固定 n_iter=3 / C=10（先验固定参数），外层 CV 纯评估。
                           metadata protocol = "strict-fixed-wl"；不要用于声称复现 GIN。
  --protocol nested        嵌套严格（推荐主结果）：外层训练折内
                           cv5（默认）或 holdout(1/9) 选 (h, C)，
                           选定后在整个 outer-train 重训，外层留出折只报告一次。
  --protocol optimistic    per-fold optimistic selection：留出折同时参与选参与报告
                           （max_theta Accuracy(fold, theta) 再对 folds 平均）。
                           严重乐观；仅作 selection-bias 诊断，不是严格评估。
  --protocol paperlike     non-nested CV 选参：theta* = argmax_theta mean_cv(theta)，
                           然后报告 theta* 在同一批 folds 上的 mean±std。
                           模拟“同一组 10-fold 既选参又报结果”；selection bias 存在。

用法
----
  uv run --group wl-kernel python code/run_wl_subtree_kernel.py --protocol nested
  uv run --group wl-kernel python code/run_wl_subtree_kernel.py \
      --datasets MUTAG PROTEINS DD NCI1 ENZYMES --protocol nested \
      --inner-strategy cv5 --grid-iters 1 2 3 4 5 6 \
      --grid-C 0.001 0.01 0.1 1 10 100 1000 \
      --seeds 0 42 123 1024 2026 777 3407 999 111 888 --n-splits 10 --n-repeats 10 \
      --force

输出默认写到 results/v2-categorical/（v1 的 results/ 旧结果保留不动）：
  {dataset}__seed{seed}.json / {dataset}_summary.json                strict
  {dataset}__wlnest_seed{seed}.json / {dataset}__wlnest_summary.json nested
  {dataset}__wlpaper_seed{seed}.json / {dataset}__wlpaper_summary.json paperlike
  {dataset}__wlopt_seed{seed}.json / {dataset}__wlopt_summary.json    optimistic
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np

# ---- numpy>=2 兼容 shim：grakel 0.1.10 的 random_walk 在 import 时执行
#      `from numpy import ComplexWarning`（numpy 2 中已移至 numpy.exceptions）。
#      必须在任何 grakel import 之前生效。
if not hasattr(np, "ComplexWarning"):
    np.ComplexWarning = np.exceptions.ComplexWarning

import torch  # noqa: E402
from grakel import Graph  # noqa: E402
from grakel.kernels import WeisfeilerLehman  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from torch_geometric.datasets import TUDataset  # noqa: E402

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
DEFAULT_DATASETS = ["REDDIT-BINARY", "COLLAB"]
DEFAULT_SEEDS = [0, 42, 123, 1024, 2026, 777, 3407, 999, 111, 888]
# GIN 对 WL 明确使用 h ∈ {1..6}
DEFAULT_GRID_ITERS = [1, 2, 3, 4, 5, 6]
# GIN 论文未公开 C grid；这里用对数网格避免最优解撞边界（不是 GIN 原始 grid）
DEFAULT_GRID_C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
REPO_ROOT = Path(__file__).resolve().parents[3]

# TU bioinformatics 数据集：必须使用原始 categorical node labels
BIO_DATASETS = {"MUTAG", "PROTEINS", "DD", "NCI1", "ENZYMES"}
# 期望节点标签类别数（来自 TU 原始 node_labels 文件；DD 只有原始值范围，见 notes）
EXPECTED_LABEL_CARDINALITY = {"MUTAG": 7, "PROTEINS": 3, "NCI1": 37, "ENZYMES": 3}

KERNEL_NAME = "WL-subtree"
BASE_KERNEL = "VertexHistogram"  # grakel WeisfeilerLehman 默认 base_graph_kernel
# holdout 内层验证占比：相对 outer-train 90% 取 1/9 → overall 80% train / 10% val / 10% test
DEFAULT_INNER_VAL_FRAC_HOLDOUT = 1.0 / 9.0

_KS = None  # {n_iter: 全量核矩阵}，fork 后由子进程继承
_YA = None  # 标签数组
_SPLITS = None  # paperlike：当前 seed 的 folds 列表 [(tr, te), ...]


# --------------------------------------------------------------------------- #
# 数据 / 节点标签
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """all random seeds（与原版 utils.set_seed 相同）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tu_dataset(name: str, data_root: Path) -> TUDataset:
    """PyG TUDataset：显式 use_node_attr=False。

    data.x 只包含 TU 原始 categorical node labels（one-hot），
    不把额外 continuous node attributes 混进节点标签。
    """
    root = Path(data_root) / name
    return TUDataset(root=str(root), name=name, use_node_attr=False)


def get_node_label_mode(dataset_name: str) -> str:
    """节点标签模式：constant（REDDIT*）/ categorical（bio）/ degree（其余）。"""
    if dataset_name.startswith("REDDIT"):
        return "constant"
    if dataset_name in BIO_DATASETS:
        return "categorical"
    return "degree"


def categorical_node_labels_from_x(x: torch.Tensor) -> dict[int, int]:
    """从 TUDataset 的 data.x 还原 categorical node labels。

    已对 5 个 bio 数据集核对：use_node_attr=False 时 data.x 为 one-hot
    （行和为 1、二值）。若不是 one-hot（例如只有 raw 整数列），则直接取列值；
    若两者都不是（例如连续特征混入），拒绝盲猜 argmax 并报错。
    """
    if x is None:
        raise ValueError("data.x is None: bio dataset has no categorical node labels available")
    x = x.detach().cpu()
    if x.dim() == 1:
        vals = x.long()
    else:
        row_sum = x.sum(dim=1)
        is_binary = bool(torch.all((x == 0) | (x == 1)))
        is_onehot = bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)) and is_binary
        if is_onehot:
            vals = x.argmax(dim=1)
        else:
            raise ValueError(
                "data.x is neither one-hot categorical labels nor a 1-D label vector; "
                "refusing to argmax blindly (check TUDataset contents / use_node_attr)."
                f" x.shape={tuple(x.shape)}, row sums range="
                f"[{row_sum.min().item():.3f},{row_sum.max().item():.3f}]"
            )
    return {i: int(vals[i]) for i in range(x.shape[0])}


def pyg_to_grakel(data, dataset_name: str) -> Graph:
    """PyG Data -> grakel Graph：无向边补双向（重复 tuple 无害，见 _check_edges_sanity）。

    节点标签由 get_node_label_mode 决定（显式传入 dataset_name，不依赖全局）：
    - REDDIT*：constant 0
    - bio：原始 categorical labels（data.x one-hot -> argmax）
    - IMDB/COLLAB 等：degree
    """
    edges = data.edge_index.t().tolist()
    undirected_edges = edges + [[v, u] for u, v in edges]
    edge_tuples = [(u, v) for u, v in undirected_edges]

    mode = get_node_label_mode(dataset_name)
    if mode == "constant":
        node_labels = {i: 0 for i in range(int(data.num_nodes))}
    elif mode == "categorical":
        if data.x is None or int(data.x.shape[0]) != int(data.num_nodes):
            raise ValueError(
                f"{dataset_name}: data.x missing or node-count mismatch "
                f"(x.shape[0]={None if data.x is None else int(data.x.shape[0])}, "
                f"num_nodes={int(data.num_nodes)})"
            )
        node_labels = categorical_node_labels_from_x(data.x)
    else:  # degree
        degrees = data.edge_index[0].bincount(minlength=data.num_nodes).int().tolist()
        node_labels = {i: d for i, d in enumerate(degrees)}
    return Graph(edge_tuples, node_labels=node_labels)


def build_kernel_matrix(graphs: list, n_iter: int) -> np.ndarray:
    """全量 WL 子树核矩阵（一次计算；子矩阵切片等价于原版逐 fold fit_transform）。

    WeisfeilerLehman 默认 base_graph_kernel 即 VertexHistogram（显式写进 metadata）。
    """
    kernel = WeisfeilerLehman(n_iter=n_iter, normalize=True)
    return np.asarray(kernel.fit_transform(graphs))


def _edges_variant(
    data,
    variant: str,
) -> list[tuple[int, int]]:
    """duplicate-edge sanity check 用：A=当前 edges+reversed；B=set 去重后对称展开。"""
    if variant == "A":
        edges = data.edge_index.t().tolist()
        undirected_edges = edges + [[v, u] for u, v in edges]
        return [(u, v) for u, v in undirected_edges]
    und = set(tuple(sorted((int(u), int(v)))) for u, v in data.edge_index.t().tolist())
    out: list[tuple[int, int]] = []
    for u, v in sorted(und):
        out.append((u, v))
        out.append((v, u))
    return out


def check_duplicate_edges_effect(
    ds: TUDataset, dataset_name: str, n_graphs: int = 8, iters: tuple[int, ...] = (1, 3, 6)
) -> dict:
    """A（当前 edges+reversed，重复 tuple）vs B（唯一无向边）的 WL 核矩阵最大差。

    PyG TUDataset.process 已对 edge_index 做 coalesce（双向各一条），
    当前代码再补 reversed 会产生重复 tuple；grakel 导入时 `nested_dict_add` 覆盖写，
    重复边不影响邻接。以核矩阵数值验证：|Δ|<1e-12 则保留现状。
    """
    graphs_a, graphs_b = [], []
    for i in range(min(n_graphs, len(ds))):
        d = ds[i]
        nl = {j: 0 for j in range(int(d.num_nodes))}
        if get_node_label_mode(dataset_name) == "categorical":
            nl = categorical_node_labels_from_x(d.x)
        graphs_a.append(Graph(_edges_variant(d, "A"), node_labels=nl))
        graphs_b.append(Graph(_edges_variant(d, "B"), node_labels=nl))
    max_diff = 0.0
    for it in iters:
        ka = np.asarray(WeisfeilerLehman(n_iter=it, normalize=True).fit_transform(graphs_a))
        kb = np.asarray(WeisfeilerLehman(n_iter=it, normalize=True).fit_transform(graphs_b))
        max_diff = max(max_diff, float(np.max(np.abs(ka - kb))))
    return {
        "dataset": dataset_name,
        "n_graphs_checked": min(n_graphs, len(ds)),
        "iters": list(iters),
        "max_abs_kernel_diff": max_diff,
        "verdict": "no effect" if max_diff < 1e-12 else "DIFFERS - needs unique-edge fix",
        "action": "keep current edges+reversed (duplicates harmless in grakel)"
        if max_diff < 1e-12
        else "switch to unique undirected edges and rerun",
    }


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def kernel_metadata(dataset_name: str) -> dict:
    """统一 kernel / 节点标签 metadata（所有 protocol 共用）。"""
    return {
        "kernel": KERNEL_NAME,
        "base_kernel": BASE_KERNEL,
        "normalize": True,
        "node_label_mode": get_node_label_mode(dataset_name),
        "node_label_note": (
            "categorical labels from TUDataset(use_node_attr=False) one-hot x, argmax"
            if get_node_label_mode(dataset_name) == "categorical"
            else ("constant 0 (no-attribute social graph)"
                  if get_node_label_mode(dataset_name) == "constant"
                  else "degree (edge_index[0].bincount)")
        ),
    }


# --------------------------------------------------------------------------- #
# strict-fixed：固定参数（n_iter=3, C=10），外层 CV 纯评估
# --------------------------------------------------------------------------- #
def run_seed(
    dataset_name: str,
    K: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
    n_iter: int,
    C: float,
) -> dict:
    """单个种子：RepeatedStratifiedKFold(n_splits, n_repeats) 固定参数评估。"""
    set_seed(seed)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    n_graphs = len(labels)

    fold_acc: list[float] = []
    fold_f1: list[float] = []
    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(
        rskf.split(np.zeros(n_graphs), labels)
    ):
        set_seed(seed + fold_idx)

        K_train = K[np.ix_(train_idx, train_idx)]
        K_test = K[np.ix_(test_idx, train_idx)]

        svm = SVC(kernel="precomputed", C=C)
        svm.fit(K_train, labels[train_idx])
        preds = svm.predict(K_test)

        fold_acc.append(float(accuracy_score(labels[test_idx], preds)))
        fold_f1.append(float(f1_score(labels[test_idx], preds, average="macro")))

        if (fold_idx + 1) % 10 == 0 or fold_idx + 1 == n_splits * n_repeats:
            print(
                f"    [{dataset_name} seed={seed}] fold {fold_idx + 1}"
                f"/{n_splits * n_repeats}  acc={fold_acc[-1]:.4f}  "
                f"(elapsed {time.time() - t0:.0f}s)"
            )

    acc = np.array(fold_acc)
    f1 = np.array(fold_f1)
    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "protocol": "strict-fixed-wl",
        "note": "fixed a-priori params (h=3, C=10); outer CV is pure evaluation; "
                "NOT an approximation of GIN's tuned-WL protocol",
        "wl": {"n_iter": int(n_iter), **kernel_metadata(dataset_name)},
        "svc": {"C": float(C), "kernel": "precomputed"},
        "fold_acc": fold_acc,
        "fold_f1": fold_f1,
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std()),
        "mean_f1": float(f1.mean()),
        "std_f1": float(f1.std()),
        "elapsed_s": float(time.time() - t0),
    }


# --------------------------------------------------------------------------- #
# nested：训练折内选参，外层留出折只报告一次
#   --inner-strategy cv5      默认：训练折内 5 折轮换（推荐）
#   --inner-strategy holdout  可选项：单次 1/9 切分 → overall 80/10/10
# --------------------------------------------------------------------------- #
def _select_params_inner(
    Ki: np.ndarray,
    y_tr: np.ndarray,
    C: float,
    inner_split: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """单个内层切分上评估 (n_iter 已定, C)：返回 (val_acc, val_f1)。"""
    in_tr, in_va = inner_split
    K_fit = Ki[np.ix_(in_tr, in_tr)]
    K_val = Ki[np.ix_(in_va, in_tr)]
    svm = SVC(kernel="precomputed", C=C)
    svm.fit(K_fit, y_tr[in_tr])
    preds = svm.predict(K_val)
    return (
        float(accuracy_score(y_tr[in_va], preds)),
        float(f1_score(y_tr[in_va], preds, average="macro")),
    )


def _eval_fold_nested(task: dict) -> dict:
    """单个外层 fold：训练折内选参 → 全训练折重训 → 外层留出折报告一次。"""
    ks, y = _KS, _YA
    tr, te = task["tr"], task["te"]
    seed, fold_idx = task["seed"], task["fold_idx"]
    set_seed(seed + fold_idx)

    y_tr = y[tr]
    inner_seed = 1000 * seed + fold_idx
    if task["strategy"] == "cv5":
        inner = StratifiedKFold(
            n_splits=task["inner_splits"], shuffle=True, random_state=inner_seed
        )
        inner_splits = list(inner.split(np.zeros(len(tr)), y_tr))
    else:  # holdout: outer-train (90%) 内取 1/9 → overall 80/10/10
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=task["inner_val_frac"], random_state=inner_seed
        )
        inner_splits = list(inner.split(np.zeros(len(tr)), y_tr))

    best: dict | None = None
    for n_iter in task["iters"]:
        Ki_tr_full = ks[n_iter][np.ix_(tr, tr)]
        for C in task["Cs"]:
            scores = [_select_params_inner(Ki_tr_full, y_tr, C, sp) for sp in inner_splits]
            m_acc = float(np.mean([s[0] for s in scores]))
            m_f1 = float(np.mean([s[1] for s in scores]))
            if best is None or (m_acc, m_f1) > (best["inner_acc"], best["inner_f1"]):
                best = {
                    "n_iter": int(n_iter), "C": float(C),
                    "inner_acc": m_acc, "inner_f1": m_f1,
                }

    K_tr = ks[best["n_iter"]][np.ix_(tr, tr)]
    K_te = ks[best["n_iter"]][np.ix_(te, tr)]
    svm = SVC(kernel="precomputed", C=best["C"])
    svm.fit(K_tr, y_tr)
    preds = svm.predict(K_te)
    return {
        "fold": fold_idx,
        "n_iter": best["n_iter"],
        "C": best["C"],
        "inner_val_acc": best["inner_acc"],
        "inner_val_f1": best["inner_f1"],
        "acc": float(accuracy_score(y[te], preds)),
        "f1": float(f1_score(y[te], preds, average="macro")),
    }


def run_seed_nested(
    dataset_name: str,
    ks: dict[int, np.ndarray],
    labels: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
    iters: list[int],
    Cs: list[float],
    strategy: str,
    inner_splits: int,
    inner_val_frac: float,
    workers: int,
) -> dict:
    """单个种子：10×10 CV；每折训练折内选参，外层留出折报告一次（嵌套严格）。"""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    n_graphs = len(labels)
    tasks = [
        {
            "seed": seed, "fold_idx": fi, "tr": tr, "te": te,
            "iters": iters, "Cs": Cs, "strategy": strategy,
            "inner_splits": inner_splits, "inner_val_frac": inner_val_frac,
        }
        for fi, (tr, te) in enumerate(rskf.split(np.zeros(n_graphs), labels))
    ]

    results: list[dict] = []
    t0 = time.time()
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            for res in pool.map(_eval_fold_nested, tasks):
                results.append(res)
                if len(results) % 10 == 0:
                    print(f"    [seed={seed}] fold {len(results)}/{len(tasks)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    else:
        for task in tasks:
            results.append(_eval_fold_nested(task))
            if len(results) % 10 == 0:
                print(f"    [seed={seed}] fold {len(results)}/{len(tasks)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: r["fold"])
    acc = np.array([r["acc"] for r in results])
    f1 = np.array([r["f1"] for r in results])
    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "protocol": f"nested-{strategy}-wl",
        "note": "inner selection on outer-train only (cv5 by default; holdout 1/9 -> overall 80/10/10); "
                "outer hold-out fold reported exactly once; recommended rigorous protocol",
        "inner_strategy": strategy,
        "inner_val_frac": float(inner_val_frac),
        "grid": {"n_iter": list(iters), "C": list(Cs)},
        "grid_origin": "h in {1..6} as in GIN (w.r.t. WL); C grid not disclosed by GIN - "
                       "log grid 0.001..1000 to avoid boundary hits, NOT GIN's original grid",
        "wl": kernel_metadata(dataset_name),
        "fold_acc": [float(a) for a in acc],
        "fold_f1": [float(a) for a in f1],
        "selected_n_iter": [r["n_iter"] for r in results],
        "selected_C": [r["C"] for r in results],
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std()),
        "mean_f1": float(f1.mean()),
        "std_f1": float(f1.std()),
        "elapsed_s": float(time.time() - t0),
    }


# --------------------------------------------------------------------------- #
# paperlike：non-nested CV model selection（同一组 folds 选参并报告）
# --------------------------------------------------------------------------- #
def _eval_theta_paperlike(task: dict) -> dict:
    """一个 theta=(h, C)：在该 seed 的所有 folds 上计算 acc/f1 列表。"""
    ks, y, splits = _KS, _YA, _SPLITS
    n_iter, C = task["n_iter"], task["C"]
    ki = ks[n_iter]
    accs: list[float] = []
    f1s: list[float] = []
    for tr, te in splits:
        k_tr = ki[np.ix_(tr, tr)]
        k_te = ki[np.ix_(te, tr)]
        svm = SVC(kernel="precomputed", C=C)
        svm.fit(k_tr, y[tr])
        preds = svm.predict(k_te)
        accs.append(float(accuracy_score(y[te], preds)))
        f1s.append(float(f1_score(y[te], preds, average="macro")))
    return {"n_iter": int(n_iter), "C": float(C), "accs": accs, "f1s": f1s}


def run_seed_paperlike(
    dataset_name: str,
    ks: dict[int, np.ndarray],
    labels: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
    iters: list[int],
    Cs: list[float],
    workers: int,
) -> dict:
    """单个种子：同一批 folds 上 theta* = argmax mean_cv(theta)，再报告 theta* 的 fold acc。

    non-nested：参数选择与结果报告使用同一组 folds → 有 selection bias。
    标为 paper-like approximation（GIN 未公开其具体选择实现）。
    """
    global _SPLITS
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    n_graphs = len(labels)
    _SPLITS = list(rskf.split(np.zeros(n_graphs), labels))

    tasks = [
        {"n_iter": it, "C": c}
        for it in iters
        for c in Cs
    ]
    t0 = time.time()
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            theta_results = list(pool.map(_eval_theta_paperlike, tasks))
    else:
        theta_results = [_eval_theta_paperlike(t) for t in tasks]

    # theta* = argmax over (h, C) of mean CV accuracy (tie-break: macro-F1, first candidate wins)
    best = None
    for r in theta_results:
        m_acc = float(np.mean(r["accs"]))
        m_f1 = float(np.mean(r["f1s"]))
        if best is None or (m_acc, m_f1) > best:
            best = (m_acc, m_f1)
            best_theta = r
    acc = np.array(best_theta["accs"])
    f1 = np.array(best_theta["f1s"])
    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "protocol": "paperlike-globalcv-wl",
        "note": "non-nested CV model selection: theta* = argmax_(h,C) mean_cv(theta) on the SAME folds "
                "used for reporting; selection bias present; paper-llike approximation, NOT exact GIN "
                "protocol (GIN did not disclose its WL selection/C-grid implementation)",
        "grid": {"n_iter": list(iters), "C": list(Cs)},
        "grid_origin": "h in {1..6} as in GIN; C log grid 0.001..1000 (GIN's C grid not disclosed)",
        "wl": kernel_metadata(dataset_name),
        "selected_theta": {"n_iter": best_theta["n_iter"], "C": best_theta["C"]},
        "selected_cv_mean_acc": float(np.mean(acc)),
        "selected_cv_mean_f1": float(np.mean(f1)),
        "fold_acc": [float(a) for a in acc],
        "fold_f1": [float(a) for a in f1],
        "selected_n_iter": [best_theta["n_iter"]],
        "selected_C": [best_theta["C"]],
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std()),
        "mean_f1": float(f1.mean()),
        "std_f1": float(f1.std()),
        "elapsed_s": float(time.time() - t0),
    }


# --------------------------------------------------------------------------- #
# optimistic：per-fold 乐观选择（留出折同时参与选参与报告）
# --------------------------------------------------------------------------- #
def _eval_fold(task: dict) -> dict:
    """单 fold：在留出折（验证集）上对 (n_iter, C) 网格选最优，报告所选参数分数。

    fork 子进程执行，_KS/_YA 继承；tie-break: (acc, f1) 字典序，先候选先得。
    """
    ks, y = _KS, _YA
    tr, va = task["tr"], task["va"]
    set_seed(task["seed"] + task["fold_idx"])

    best: dict | None = None
    for n_iter in task["iters"]:
        ki = ks[n_iter]
        k_tr = ki[np.ix_(tr, tr)]
        k_va = ki[np.ix_(va, tr)]
        for C in task["Cs"]:
            svm = SVC(kernel="precomputed", C=C)
            svm.fit(k_tr, y[tr])
            preds = svm.predict(k_va)
            acc = float(accuracy_score(y[va], preds))
            f1v = float(f1_score(y[va], preds, average="macro"))
            if best is None or (acc, f1v) > (best["acc"], best["f1"]):
                best = {"n_iter": int(n_iter), "C": float(C), "acc": acc, "f1": f1v}
    return {"fold": task["fold_idx"], **best}


def run_seed_optimistic(
    dataset_name: str,
    ks: dict[int, np.ndarray],
    labels: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
    iters: list[int],
    Cs: list[float],
    workers: int,
) -> dict:
    """单个种子：10×10 CV；每折留出折选参并报告（乐观，非严格评估）。"""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    n_graphs = len(labels)
    tasks = [
        {"seed": seed, "fold_idx": fi, "tr": tr, "va": va, "iters": iters, "Cs": Cs}
        for fi, (tr, va) in enumerate(rskf.split(np.zeros(n_graphs), labels))
    ]

    results: list[dict] = []
    t0 = time.time()
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            for res in pool.map(_eval_fold, tasks):
                results.append(res)
                if len(results) % 10 == 0:
                    print(f"    [seed={seed}] fold {len(results)}/{len(tasks)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    else:
        for task in tasks:
            results.append(_eval_fold(task))
            if len(results) % 10 == 0:
                print(f"    [seed={seed}] fold {len(results)}/{len(tasks)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: r["fold"])
    acc = np.array([r["acc"] for r in results])
    f1 = np.array([r["f1"] for r in results])
    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "protocol": "paper-optimistic-wl",
        "note": "per-fold optimistic selection: the held-out fold participates in BOTH parameter "
                "selection and result reporting (max_theta Accuracy(fold, theta) averaged over folds); "
                "strong optimistic bias; diagnostics only, NOT a rigorous protocol, NOT exact GIN",
        "grid": {"n_iter": list(iters), "C": list(Cs)},
        "grid_origin": "h in {1..6} as in GIN; C log grid 0.001..1000 (GIN's C grid not disclosed)",
        "wl": kernel_metadata(dataset_name),
        "fold_acc": [float(a) for a in acc],
        "fold_f1": [float(a) for a in f1],
        "selected_n_iter": [r["n_iter"] for r in results],
        "selected_C": [r["C"] for r in results],
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std()),
        "mean_f1": float(f1.mean()),
        "std_f1": float(f1.std()),
        "elapsed_s": float(time.time() - t0),
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summarize(results: list[dict]) -> dict:
    """跨种子汇总。

    统计口径（与 GIN Table 1 的 fold std 不同，勿直接比较 ±）：
    - across_seed_mean_acc / across_seed_std_acc（= std of seed-level means）：
      每个 seed 的 repeated-CV mean accuracy 在 seeds 之间的 mean ± std。
    - all_outer_fold_mean_acc / all_outer_fold_std_acc：
      把所有 seed 的所有 outer fold accuracy 直接合并的 mean ± std。
    GIN 报告的是 10-fold validation mean ± fold std，因此只能比较 central accuracy。
    """
    seeds_mean_acc = [r["mean_acc"] for r in results]
    seeds_mean_f1 = [r["mean_f1"] for r in results]
    all_folds_acc = [a for r in results for a in r["fold_acc"]]
    all_folds_f1 = [a for r in results for a in r["fold_f1"]]
    std_of_means = float(np.std(seeds_mean_acc))
    return {
        "dataset": results[0]["dataset"],
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        # 主统计：跨种子 mean of repeated-CV means
        "across_seed_mean_acc": float(np.mean(seeds_mean_acc)),
        "across_seed_std_acc": std_of_means,
        "across_seed_std_of_mean_acc": std_of_means,  # 同 across_seed_std_acc，语义更明确
        "across_seed_mean_f1": float(np.mean(seeds_mean_f1)),
        "across_seed_std_f1": float(np.std(seeds_mean_f1)),
        # 全部 outer fold 直接合并
        "all_outer_fold_mean_acc": float(np.mean(all_folds_acc)),
        "all_outer_fold_std_acc": float(np.std(all_folds_acc)),
        "all_outer_fold_mean_f1": float(np.mean(all_folds_f1)),
        "all_outer_fold_std_f1": float(np.std(all_folds_f1)),
        "n_outer_folds_total": len(all_folds_acc),
        "per_seed": results,
    }


def param_stats(seed_results: list[dict], grid_iters: list[int], grid_Cs: list[float]) -> dict:
    """选择参数频次 + 边界命中统计。"""
    from collections import Counter

    ni_sel = sum((r["selected_n_iter"] for r in seed_results), [])
    c_sel = sum((r["selected_C"] for r in seed_results), [])
    ni_counts = dict(Counter(ni_sel))
    c_counts = dict(Counter([float(c) for c in c_sel]))
    return {
        "n_iter": ni_counts,
        "C": c_counts,
        "grid": {"n_iter": list(grid_iters), "C": list(grid_Cs)},
        "grid_boundary_hits": {
            "selected_n_iter == max(grid)": int(sum(1 for x in ni_sel if x == max(grid_iters))),
            "selected_C == min(grid)": int(sum(1 for x in c_sel if float(x) == float(min(grid_Cs)))),
            "selected_C == max(grid)": int(sum(1 for x in c_sel if float(x) == float(max(grid_Cs)))),
            "n_total_selections": len(ni_sel),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="TUDataset 名字（默认 %(default)s）")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="每个数据集的 CV 种子（默认 10 个）")
    ap.add_argument("--n-splits", type=int, default=10, help="折数（默认 10）")
    ap.add_argument("--n-repeats", type=int, default=10, help="重复次数（默认 10）")
    ap.add_argument("--wl-iter", type=int, default=3, help="WL 迭代轮数（仅 strict，默认 3）")
    ap.add_argument("--wl-C", type=float, default=10.0, help="SVC C（仅 strict，默认 10.0）")
    ap.add_argument("--protocol", choices=["strict", "nested", "optimistic", "paperlike"],
                    default="strict",
                    help="strict: 固定参数 (h=3,C=10) 报告留出折（先验固定基线）；"
                         "nested: 训练折内 cv5(默认) 选参、留出折只报告一次（推荐）；"
                         "optimistic: 留出折=验证集选参并报告（乐观上界/诊断）；"
                         "paperlike: 同一组 folds 上 non-nested CV 选参再报告（近似 GIN 风格，有偏差）")
    ap.add_argument("--grid-iters", type=int, nargs="+", default=DEFAULT_GRID_ITERS,
                    help="选参网格候选 n_iter（默认 1 2 3 4 5 6，与 GIN 的 h∈{1..6} 一致）")
    ap.add_argument("--grid-C", type=float, nargs="+", default=DEFAULT_GRID_C,
                    help="选参网格候选 C（默认 0.001 0.01 0.1 1 10 100 1000；"
                         "注意 GIN 未公开其 C grid，这不是 GIN 原始网格）")
    ap.add_argument("--inner-strategy", choices=["holdout", "cv5"], default="cv5",
                    help="nested 内层选参（默认 cv5）：cv5=训练折内 5 折轮换（推荐）；"
                         "holdout=outer-train 内单次 1/9 验证 → overall 80/10/10")
    ap.add_argument("--inner-val-frac", type=float, default=DEFAULT_INNER_VAL_FRAC_HOLDOUT,
                    help="holdout 内层验证占比，相对 outer-train(90%)；默认 1/9≈0.1111 "
                         "→ overall 80% train / 10% val / 10% test（实现 8:1:1 原意）")
    ap.add_argument("--inner-splits", type=int, default=5,
                    help="cv5 内层折数（仅 nested + cv5）")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="并行 fold 进程数（nested / optimistic / paperlike）")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "TUD",
                    help="TUDataset 根目录（默认 <repo>/data/TUD）")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "v2-categorical",
                    help="结果目录（默认 <track>/results/v2-categorical；v1 旧结果保留在原 results/）")
    ap.add_argument("--force", action="store_true", help="重跑已存在的 (数据集, 种子)")
    args = ap.parse_args()

    use_grid = args.protocol in ("nested", "optimistic", "paperlike")
    suffix = {"nested": "wlnest", "optimistic": "wlopt", "paperlike": "wlpaper"}.get(args.protocol)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"data root : {args.data_root}")
    print(f"output    : {out_dir}")
    print(f"protocol  : {args.protocol}")
    print(f"datasets  : {args.datasets}")
    print(f"seeds     : {args.seeds}  ({len(args.seeds)} seeds x {args.n_splits}x{args.n_repeats} CV)")

    for name in args.datasets:
        print(f"\n{'=' * 70}\nDataset: {name}\n{'=' * 70}")

        # ---- 数据加载（use_node_attr=False：data.x = TU categorical labels only） ---- #
        ds = load_tu_dataset(name, args.data_root)
        y = np.array([d.y.item() for d in ds], dtype=np.int64)
        n_graphs, n_classes = len(ds), len(np.unique(y))
        avg_n = np.mean([int(d.num_nodes) for d in ds])
        mode = get_node_label_mode(name)
        print(f"  {n_graphs} graphs, {n_classes} classes, avg nodes {avg_n:.1f}")
        if mode == "categorical":
            x_first = ds[0].x
            print(f"  data.x shape (first graph): {tuple(x_first.shape)} "
                  f"(use_node_attr=False; TU categorical labels only)", flush=True)
            # 全库标签核验
            lab_all = []
            for d in ds:
                if d.x is None or int(d.x.shape[0]) != int(d.num_nodes):
                    raise ValueError(f"{name}: data.x missing or node-count mismatch")
                lab_all.append(categorical_node_labels_from_x(d.x))
            card = len({lab for nl in lab_all for lab in nl.values()})
            print(f"  observed node-label cardinality: {card}  "
                  f"(expected: {EXPECTED_LABEL_CARDINALITY.get(name, 'n/a - check raw file')})",
                  flush=True)
            # one-hot 检查输出（已在 categorical_node_labels_from_x 内部断言）
        elif mode == "constant":
            print("  node labels: constant 0 (REDDIT*: no node attributes)", flush=True)
        else:
            print("  node labels: degree (edge_index[0].bincount)", flush=True)

        # ---- duplicate-edge sanity check ---- #
        edge_check = check_duplicate_edges_effect(ds, name)
        print(f"  edge-duplicate check: max|K_A-K_B|={edge_check['max_abs_kernel_diff']:.2e} "
              f"-> {edge_check['verdict']} ({edge_check['action']})", flush=True)

        # ---- 核矩阵 ---- #
        if use_grid:
            global _KS, _YA
            _YA = y
            _KS = {}
            for it in args.grid_iters:
                t0 = time.time()
                graphs = [pyg_to_grakel(d, name) for d in ds]
                _KS[it] = build_kernel_matrix(graphs, n_iter=it)
                print(f"  WL kernel matrix n_iter={it} ({n_graphs}x{n_graphs})"
                      f" computed in {time.time() - t0:.1f}s", flush=True)
        else:
            t0 = time.time()
            graphs = [pyg_to_grakel(d, name) for d in ds]
            K = build_kernel_matrix(graphs, n_iter=args.wl_iter)
            print(f"  WL kernel matrix ({n_graphs}x{n_graphs}) computed in {time.time() - t0:.1f}s",
                  flush=True)

        # ---- 每个种子 ---- #
        seed_results: list[dict] = []
        for seed in args.seeds:
            out_name = f"{name}__{suffix}_seed{seed}.json" if suffix else f"{name}__seed{seed}.json"
            out_path = out_dir / out_name
            if out_path.exists() and not args.force:
                with open(out_path, encoding="utf-8") as f:
                    r = json.load(f)
                seed_results.append(r)
                print(f"  seed={seed}: cached {r['mean_acc']:.4f}+-{r['std_acc']:.4f}")
                continue

            print(f"  seed={seed}: running {args.n_splits}x{args.n_repeats} CV ...")
            t0 = time.time()
            if args.protocol == "optimistic":
                r = run_seed_optimistic(
                    name, _KS, y, seed, args.n_splits, args.n_repeats,
                    args.grid_iters, args.grid_C, args.workers,
                )
            elif args.protocol == "nested":
                r = run_seed_nested(
                    name, _KS, y, seed, args.n_splits, args.n_repeats,
                    args.grid_iters, args.grid_C, args.inner_strategy,
                    args.inner_splits, args.inner_val_frac, args.workers,
                )
            elif args.protocol == "paperlike":
                r = run_seed_paperlike(
                    name, _KS, y, seed, args.n_splits, args.n_repeats,
                    args.grid_iters, args.grid_C, args.workers,
                )
            else:
                r = run_seed(name, K, y, seed, args.n_splits, args.n_repeats,
                             args.wl_iter, args.wl_C)
            print(f"    done acc={r['mean_acc']:.4f}+-{r['std_acc']:.4f} "
                  f"f1={r['mean_f1']:.4f}+-{r['std_f1']:.4f} ({r['elapsed_s']:.0f}s)")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, default=str)
            seed_results.append(r)

        # ---- 汇总 ---- #
        summary = summarize(seed_results)
        if suffix:
            summary["param_stats"] = param_stats(seed_results, args.grid_iters, args.grid_C)
            sum_path = out_dir / f"{name}__{suffix}_summary.json"
        else:
            sum_path = out_dir / f"{name}_summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n  Summary ({name}, {summary['n_seeds']} seeds):")
        print(f"  {'seed':>6}  {'acc':>12}  {'f1':>12}")
        for r in seed_results:
            print(f"  {r['seed']:>6}  {r['mean_acc']:.4f}+-{r['std_acc']:.4f}"
                  f"  {r['mean_f1']:.4f}+-{r['std_f1']:.4f}")
        print(f"  across seeds: acc {summary['across_seed_mean_acc']:.4f}"
              f"+-{summary['across_seed_std_acc']:.4f}  "
              f"f1 {summary['across_seed_mean_f1']:.4f}"
              f"+-{summary['across_seed_std_f1']:.4f}")
        print(f"  all outer folds pooled: acc {summary['all_outer_fold_mean_acc']:.4f}"
              f"+-{summary['all_outer_fold_std_acc']:.4f}")
        if suffix:
            print(f"  param stats: {summary['param_stats']}")
        print(f"  saved -> {sum_path}")


if __name__ == "__main__":
    main()
