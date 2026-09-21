#!/usr/bin/env python3
"""Beam8 连续重叠 patch + 共享 KSVD 流水线 —— 单文件可运行版。

本文件把 ``ksvd`` 轨道的**导师 50 节点真实子图**主线
（2026-08-05/06 冻结协议）的全部代码合并成单个自包含模块，方便独立审阅与运行：

    原始邻接矩阵
        -> GLOBAL_WL 稳定排序（重编号不变的节点结构类）
        -> Beam8 边缘候选覆盖采样（连续重叠：相邻两块硬共享 o 个节点，
           保留子集 beam = 8）
        -> rooted-canonical 槽位排序（与节点 ID 无关的局部邻接向量）
        -> 覆盖档位（BASE / EDGE95 / FAIR95 ...，同一条链的前缀截断）
        -> root-candidate 分组 3 折划分
        -> 共享 KSVD 字典学习（K=24 原子、稀疏度 T=3、T_min=1、
           25 轮更新、每折仅用训练集居中）
        -> 按重叠权重融合拼回完整 50 节点图

数据：``data/subgraphs_50_20_10000_batch_0.pkl``（SHA-256
``936134e7876740d5efc478f6bdf9663e31440ab506ecee507b00f8e32a7f05cb``），
10000 张连通、简单、无向的 50 节点图，共享 2805 个源节点 ID。
原始 pickle 只读不改写；首次运行时构建紧凑 NPZ 缓存。

运行（默认即冻结 pilot 配置）：

    python beam8_mentor_pipeline_20260809.py              # 30 图冒烟测试
    python beam8_mentor_pipeline_20260809.py --pilot-size 500   # 复现报告
    python beam8_mentor_pipeline_20260809.py --geometry s8_o2 --checkpoint BASE

函数体与 ``tracks/ksvd/code/*`` 逐字一致，仅删去未使用的非 beam8 辅助逻辑
（walk 采样器、MIL 池化、下游 readout、各 CLI main）。源节点 ID 一律不进入
patch 向量或字典，只用于划分记账与暴露审计。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ===========================================================================
# A. 图基础工具（来自 code/from_scratch_unplanted_representation.py）
#    上下三角边枚举、邻接->上三角向量、简单图校验、连通性判断。
# ===========================================================================

def upper_triangle_edges(n_nodes: int) -> tuple[tuple[int, int], ...]:
    return tuple((left, right) for left in range(n_nodes) for right in range(left + 1, n_nodes))


def adjacency_to_upper_vector(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    return np.asarray(
        [adjacency[left, right] for left, right in upper_triangle_edges(adjacency.shape[0])],
        dtype=np.float64,
    )


def validate_simple_adjacency(adjacency: np.ndarray) -> None:
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("adjacency diagonal must be zero")
    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError("adjacency must be binary")


def is_connected(adjacency: np.ndarray) -> bool:
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return False
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if value not in seen:
                seen.add(value)
                stack.append(value)
    return len(seen) == n_nodes


# ===========================================================================
# B. 导师子图数据契约（code/data_mentor_subgraphs.py）
#    把 10000 张 networkx.Graph 审计后转成紧凑 NPZ 缓存；含密度分层
#    与 root-aware 分层 pilot 抽样。源 ID 只用于记账，不进特征。
# ===========================================================================

EXPECTED_N_NODES = 50
DEFAULT_SOURCE_NAME = "subgraphs_50_20_10000_batch_0.pkl"
SCHEMA_VERSION = 1

AVERAGE_DEGREE_BIN_EDGES = (5.0, 10.0, 15.0, 25.0)
STRATUM_NAMES = ("lt5", "d5_10", "d10_15", "d15_25", "ge25")

DEFAULT_PILOT_SIZE = 500
DEFAULT_PILOT_SEED = 20260806

_REQUIRED_CACHE_KEYS = {
    "adjacency",
    "global_node_ids",
    "vocab",
    "roots",
    "edge_counts",
    "avg_degrees",
    "density_percent",
    "metadata",
}


class MentorSubgraphBundle:
    """不可变的子图库缓存视图：邻接、全局节点 ID、词典、根、边数、度等。"""

    __slots__ = (
        "adjacency",
        "global_node_ids",
        "vocab",
        "roots",
        "edge_counts",
        "avg_degrees",
        "density_percent",
        "metadata",
    )

    def __init__(
        self,
        adjacency: np.ndarray,
        global_node_ids: np.ndarray,
        vocab: np.ndarray,
        roots: np.ndarray,
        edge_counts: np.ndarray,
        avg_degrees: np.ndarray,
        density_percent: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.adjacency = adjacency
        self.global_node_ids = global_node_ids
        self.vocab = vocab
        self.roots = roots
        self.edge_counts = edge_counts
        self.avg_degrees = avg_degrees
        self.density_percent = density_percent
        self.metadata = metadata

    @property
    def n_graphs(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[1])


class PilotSelection:
    """确定性、root-aware 的分层 pilot 索引选择结果。"""

    __slots__ = (
        "indices",
        "strata",
        "quotas",
        "selected_counts",
        "n_pilot",
        "seed",
        "n_distinct_roots",
    )

    def __init__(
        self,
        indices: np.ndarray,
        strata: np.ndarray,
        quotas: tuple[int, ...],
        selected_counts: tuple[int, ...],
        n_pilot: int,
        seed: int,
        n_distinct_roots: int,
    ) -> None:
        self.indices = indices
        self.strata = strata
        self.quotas = quotas
        self.selected_counts = selected_counts
        self.n_pilot = n_pilot
        self.seed = seed
        self.n_distinct_roots = n_distinct_roots

    @property
    def size(self) -> int:
        return int(self.indices.size)

    def summary(self) -> dict[str, Any]:
        stratum_counts = np.bincount(
            self.strata, minlength=len(STRATUM_NAMES)
        ).tolist()
        return {
            "n_pilot": self.n_pilot,
            "seed": self.seed,
            "size": self.size,
            "stratum_counts": dict(zip(STRATUM_NAMES, stratum_counts)),
            "quotas": dict(zip(STRATUM_NAMES, self.quotas)),
            "selected_counts": dict(zip(STRATUM_NAMES, self.selected_counts)),
            "n_distinct_roots": self.n_distinct_roots,
        }


def _repo_root() -> Path:
    """向上逐级查找包含数据文件的仓库根目录（保证脚本可移到子目录运行）。"""
    if __file__ == "<string>":
        return Path(".").resolve()
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / DEFAULT_SOURCE_NAME).exists():
            return candidate
    return Path(".").resolve()


def default_source_path() -> Path:
    return _repo_root() / "data" / DEFAULT_SOURCE_NAME


def default_cache_path(source: str | Path | None = None) -> Path:
    return Path(source or default_source_path()).with_suffix(".npz")


def sha256_of(path: str | Path) -> str:
    """分块流式计算文件 SHA-256 十六进制摘要。"""
    digest = sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def density_stratum(average_degrees: np.ndarray) -> np.ndarray:
    """把平均度映射到冻结的五个真实数据密度分层。"""
    values = np.asarray(average_degrees, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("average_degrees must be a 1-D array")
    edges = np.asarray(AVERAGE_DEGREE_BIN_EDGES, dtype=np.float64)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def edge_density_percent(
    edge_counts: np.ndarray, n_nodes: int = EXPECTED_N_NODES
) -> np.ndarray:
    counts = np.asarray(edge_counts)
    if counts.ndim != 1:
        raise ValueError("edge_counts must be a 1-D array")
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    return 100.0 * 2.0 * counts.astype(np.float64) / (n_nodes * (n_nodes - 1))


def edge_avg_degree(
    edge_counts: np.ndarray, n_nodes: int = EXPECTED_N_NODES
) -> np.ndarray:
    counts = np.asarray(edge_counts)
    if counts.ndim != 1:
        raise ValueError("edge_counts must be a 1-D array")
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    return 2.0 * counts.astype(np.float64) / n_nodes


def _normalize_node_id(raw: object) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"node id must be str, got {type(raw).__name__}: {raw!r}")
    normalized = str(raw)
    if not normalized:
        raise ValueError("node id must be non-empty")
    if normalized != normalized.strip():
        raise ValueError(f"node id must not contain surrounding whitespace: {normalized!r}")
    return normalized


def _normalize_edge_id(data: Mapping[str, Any]) -> str:
    if not isinstance(data, Mapping):
        raise TypeError(f"edge data must be a mapping, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError("edge is missing the 'id' attribute")
    return _normalize_node_id(data["id"])


def _pair_key(a: int, b: int, vocab_size: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return lo * vocab_size + hi


def _validate_graph_structure(graph: Any, index: int) -> None:
    import networkx as nx

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"graph[{index}] is not a networkx.Graph (got {type(graph).__name__})")
    if nx.is_directed(graph):
        raise ValueError(f"graph[{index}] is directed; expected undirected")
    if graph.is_multigraph():
        raise ValueError(f"graph[{index}] contains a parallel edge; expected simple graph")
    if graph.number_of_nodes() != EXPECTED_N_NODES:
        raise ValueError(
            f"graph[{index}] has {graph.number_of_nodes()} nodes; expected {EXPECTED_N_NODES}"
        )
    if not nx.is_connected(graph):
        raise ValueError(f"graph[{index}] is disconnected")


def bundle_from_graphs(
    graphs: Sequence[Any],
    *,
    source_path: str | Path | None = None,
    limit: int | None = None,
) -> MentorSubgraphBundle:
    """审计一列图并构建紧凑内存 bundle（任一违规即抛异常）。

    完整校验：无向/连通/简单/恰 50 节点；节点 ID 归一化且图内唯一；
    跨图共享节点对邻接不冲突；共享边 ID 与端点一一对应。
    """
    import networkx as nx  # noqa: F401  （读取 nx.Graph 对象所需）

    graphs = list(graphs)
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        graphs = graphs[:limit]
    if not graphs:
        raise ValueError("graph list is empty")
    n_graphs = len(graphs)

    node_id_rows: list[list[str]] = []
    edge_count_list: list[int] = []
    root_list: list[str] = []
    all_ids: set[str] = set()
    edge_count_min = edge_count_max = 0
    for gi, graph in enumerate(graphs):
        _validate_graph_structure(graph, gi)
        node_ids = [_normalize_node_id(n) for n in graph.nodes()]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"graph[{gi}] contains duplicate node ids")
        node_id_rows.append(node_ids)
        all_ids.update(node_ids)
        root_list.append(node_ids[0])
        pair_seen: set[tuple[str, str]] = set()
        for u, v, data in graph.edges(data=True):
            uu = _normalize_node_id(u)
            vv = _normalize_node_id(v)
            if uu == vv:
                raise ValueError(f"graph[{gi}] contains a self-loop on {uu!r}")
            _normalize_edge_id(data)
            key = (uu, vv) if uu <= vv else (vv, uu)
            if key in pair_seen:
                raise ValueError(f"graph[{gi}] contains a parallel edge {key!r}")
            pair_seen.add(key)
        edge_count = len(pair_seen)
        edge_count_list.append(edge_count)
        if gi == 0 or edge_count < edge_count_min:
            edge_count_min = edge_count
        if edge_count > edge_count_max:
            edge_count_max = edge_count

    vocab_size = len(all_ids)
    if vocab_size > np.iinfo(np.int32).max:
        raise ValueError(f"vocabulary too large for int32 ids: {vocab_size}")
    vocab_list = sorted(all_ids)
    vocab = np.asarray(vocab_list, dtype=f"U{max(len(s) for s in vocab_list)}")
    id_to_index = {node_id: i for i, node_id in enumerate(vocab_list)}

    adjacency = np.zeros((n_graphs, EXPECTED_N_NODES, EXPECTED_N_NODES), dtype=np.uint8)
    global_node_ids = np.zeros((n_graphs, EXPECTED_N_NODES), dtype=np.int32)
    roots = np.zeros(n_graphs, dtype=np.int32)
    edge_keys_chunks: list[np.ndarray] = []
    nonedge_keys_chunks: list[np.ndarray] = []
    eid_chunks: list[np.ndarray] = []
    degree_min = degree_max = 0
    for gi, graph in enumerate(graphs):
        node_ids = node_id_rows[gi]
        row = np.asarray([id_to_index[node_id] for node_id in node_ids], dtype=np.int32)
        global_node_ids[gi] = row
        roots[gi] = id_to_index[root_list[gi]]
        position = {node_id: p for p, node_id in enumerate(node_ids)}
        edge_keys: list[int] = []
        eids: list[str] = []
        for u, v, data in graph.edges(data=True):
            uu = _normalize_node_id(u)
            vv = _normalize_node_id(v)
            try:
                iu, iv = position[uu], position[vv]
            except KeyError:
                raise ValueError(f"graph[{gi}] edge endpoint {uu!r}/{vv!r} not in its node set")
            adjacency[gi, iu, iv] = 1
            adjacency[gi, iv, iu] = 1
            a, b = int(row[iu]), int(row[iv])
            edge_keys.append(_pair_key(a, b, vocab_size))
            eids.append(_normalize_edge_id(data))
        edge_keys_chunks.append(np.asarray(edge_keys, dtype=np.int64))
        eid_chunks.append(np.asarray(eids, dtype="U"))
        left_pos, right_pos = np.triu_indices(EXPECTED_N_NODES, k=1)
        left = row[left_pos].astype(np.int64)
        right = row[right_pos].astype(np.int64)
        all_pair_keys = np.minimum(left, right) * vocab_size + np.maximum(left, right)
        nonedge_keys_chunks.append(
            np.setdiff1d(all_pair_keys, np.unique(edge_keys_chunks[-1]))
        )
        degrees = adjacency[gi].sum(axis=0)
        if gi == 0:
            degree_min = int(degrees.min())
            degree_max = int(degrees.max())
        else:
            degree_min = min(degree_min, int(degrees.min()))
            degree_max = max(degree_max, int(degrees.max()))

    edge_keys_all = np.unique(np.concatenate(edge_keys_chunks))
    nonedge_keys_all = np.unique(np.concatenate(nonedge_keys_chunks))
    conflicting_pairs = np.intersect1d(edge_keys_all, nonedge_keys_all)
    if conflicting_pairs.size:
        raise ValueError(
            f"{conflicting_pairs.size} node pairs appear as an edge in one "
            "graph and a non-edge in another"
        )

    eid_codes = np.concatenate(eid_chunks)
    all_pair_keys = np.concatenate(edge_keys_chunks)
    by_eid = np.lexsort((all_pair_keys, eid_codes))
    n_eid_endpoint_conflicts = int(
        np.count_nonzero(
            (eid_codes[by_eid][1:] == eid_codes[by_eid][:-1])
            & (all_pair_keys[by_eid][1:] != all_pair_keys[by_eid][:-1])
        )
    )
    if n_eid_endpoint_conflicts:
        raise ValueError(f"{n_eid_endpoint_conflicts} edge ids map to conflicting endpoints")
    by_pair = np.lexsort((eid_codes, all_pair_keys))
    n_pair_eid_conflicts = int(
        np.count_nonzero(
            (all_pair_keys[by_pair][1:] == all_pair_keys[by_pair][:-1])
            & (eid_codes[by_pair][1:] != eid_codes[by_pair][:-1])
        )
    )
    if n_pair_eid_conflicts:
        raise ValueError(f"{n_pair_eid_conflicts} node pairs map to conflicting edge ids")

    edge_counts = np.asarray(edge_count_list, dtype=np.int32)
    density = edge_density_percent(edge_counts)
    average_degrees = edge_avg_degree(edge_counts)
    strata = density_stratum(average_degrees)
    stratum_counts = np.bincount(strata, minlength=len(STRATUM_NAMES)).tolist()

    source = Path(source_path).resolve() if source_path is not None else None
    if source is not None and not source.exists():
        raise FileNotFoundError(f"source pkl not found: {source}")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder": Path(__file__).name,
        "source_path": str(source) if source is not None else None,
        "source_size_bytes": source.stat().st_size if source is not None else None,
        "source_sha256": sha256_of(source) if source is not None else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "limit": None if limit is None else int(limit),
        "average_degree_bins": list(AVERAGE_DEGREE_BIN_EDGES),
        "n_graphs": n_graphs,
        "n_nodes_per_graph": EXPECTED_N_NODES,
        "n_edges_total": int(edge_counts.sum()),
        "n_unique_pairs": int(edge_keys_all.size),
        "n_unique_edge_ids": int(np.unique(eid_codes).size),
        "n_unique_node_ids": vocab_size,
        "n_unique_roots": len(set(root_list)),
        "edge_count_min": edge_count_min,
        "edge_count_max": edge_count_max,
        "degree_min": degree_min,
        "degree_max": degree_max,
        "density_percent_min": float(density.min()),
        "density_percent_max": float(density.max()),
        "stratum_counts": dict(zip(STRATUM_NAMES, stratum_counts)),
        "validation": {
            "n_nodes_per_graph": EXPECTED_N_NODES,
            "undirected": True,
            "connected": True,
            "simple": True,
            "binary": True,
            "node_ids_normalized": True,
            "n_pair_adjacency_conflicts": 0,
            "n_edge_id_endpoint_conflicts": 0,
            "n_pair_edge_id_conflicts": 0,
        },
    }
    return MentorSubgraphBundle(
        adjacency=adjacency,
        global_node_ids=global_node_ids,
        vocab=vocab,
        roots=roots,
        edge_counts=edge_counts,
        avg_degrees=average_degrees,
        density_percent=density,
        metadata=metadata,
    )


def save_bundle(bundle: MentorSubgraphBundle, cache_path: str | Path) -> None:
    """把 bundle 写成紧凑 NPZ 缓存（原子替换，绝不写 pickle）。"""
    cache_path = Path(cache_path)
    if cache_path.suffix.lower() == ".pkl":
        raise ValueError(f"refusing to write a cache over a .pkl path: {cache_path}")
    payload = {
        "adjacency": bundle.adjacency,
        "global_node_ids": bundle.global_node_ids,
        "vocab": bundle.vocab,
        "roots": bundle.roots,
        "edge_counts": bundle.edge_counts,
        "avg_degrees": bundle.avg_degrees,
        "density_percent": bundle.density_percent,
        "metadata": np.array(json.dumps(bundle.metadata, indent=2)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, cache_path)


def load_graph_list(source_pkl: str | Path) -> list[Any]:
    import networkx as nx  # noqa: F401  （反序列化 nx.Graph 对象所需）

    source_pkl = Path(source_pkl)
    if not source_pkl.exists():
        raise FileNotFoundError(f"source pkl not found: {source_pkl}")
    with open(source_pkl, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"source pkl must contain a list, got {type(payload).__name__}")
    return payload


def load_cache(
    cache_path: str | Path,
    *,
    limit: int | None = None,
    verify_checksum: bool = True,
) -> MentorSubgraphBundle:
    """加载 NPZ 缓存（allow_pickle=False）并重新审计结构与源校验和。"""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as loaded:
        missing = _REQUIRED_CACHE_KEYS.difference(loaded.files)
        if missing:
            raise ValueError(f"cache is missing {sorted(missing)}")
        adjacency = np.array(loaded["adjacency"], copy=True)
        global_node_ids = np.array(loaded["global_node_ids"], copy=True)
        vocab = np.array(loaded["vocab"], copy=True)
        roots = np.array(loaded["roots"], copy=True)
        edge_counts = np.array(loaded["edge_counts"], copy=True)
        avg_degrees = np.array(loaded["avg_degrees"], copy=True)
        density_percent = np.array(loaded["density_percent"], copy=True)
        meta_raw = loaded["metadata"]
        meta_text = str(meta_raw) if meta_raw.ndim == 0 else str(meta_raw[0])
    try:
        metadata: dict[str, Any] = json.loads(meta_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cache metadata is not valid JSON: {exc}")

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"cache schema_version {metadata.get('schema_version')!r} != expected "
            f"{SCHEMA_VERSION}; rebuild with force=True"
        )
    if adjacency.ndim != 3 or adjacency.shape[1:] != (EXPECTED_N_NODES, EXPECTED_N_NODES):
        raise ValueError(f"invalid adjacency shape {adjacency.shape}")
    n_graphs = adjacency.shape[0]
    if adjacency.dtype != np.uint8:
        raise ValueError(f"adjacency must be uint8, got {adjacency.dtype}")
    if global_node_ids.shape != (n_graphs, EXPECTED_N_NODES):
        raise ValueError(f"global_node_ids shape {global_node_ids.shape} inconsistent with adjacency")
    if global_node_ids.dtype != np.int32:
        raise ValueError(f"global_node_ids must be int32, got {global_node_ids.dtype}")
    if vocab.ndim != 1 or vocab.dtype.kind != "U":
        raise ValueError(f"vocab must be a 1-D unicode array, got shape {vocab.shape} dtype {vocab.dtype}")
    if roots.shape != (n_graphs,):
        raise ValueError(f"invalid roots shape {roots.shape}")
    if edge_counts.shape != (n_graphs,):
        raise ValueError(f"invalid edge_counts shape {edge_counts.shape}")
    if avg_degrees.shape != (n_graphs,):
        raise ValueError(f"invalid avg_degrees shape {avg_degrees.shape}")
    if density_percent.shape != (n_graphs,):
        raise ValueError(f"invalid density_percent shape {density_percent.shape}")
    if not np.array_equal(adjacency, adjacency.transpose(0, 2, 1)):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diagonal(adjacency, axis1=1, axis2=2) != 0):
        raise ValueError("adjacency diagonal must be zero")
    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError("adjacency must be binary")
    vocab_size = len(vocab)
    if not np.all((global_node_ids >= 0) & (global_node_ids < vocab_size)):
        raise ValueError("global_node_ids out of vocabulary range")
    if not np.all((roots >= 0) & (roots < vocab_size)):
        raise ValueError("roots out of vocabulary range")
    if not np.array_equal(global_node_ids[:, 0], roots):
        raise ValueError("roots must equal global_node_ids[:, 0]")
    expected_edges = adjacency.sum(axis=(1, 2), dtype=np.int64) // 2
    if not np.array_equal(expected_edges, edge_counts):
        raise ValueError("edge_counts inconsistent with adjacency row sums")
    if not np.array_equal(avg_degrees, edge_avg_degree(edge_counts)):
        raise ValueError("avg_degrees inconsistent with edge_counts")
    if not np.array_equal(density_percent, edge_density_percent(edge_counts)):
        raise ValueError("density_percent inconsistent with edge_counts")

    metadata = dict(metadata)
    source = metadata.get("source_path")
    if verify_checksum and source and metadata.get("source_sha256"):
        if Path(source).exists():
            actual = sha256_of(source)
            if actual != metadata["source_sha256"]:
                raise ValueError("source pkl checksum mismatch: the cache is stale, rebuild with force=True")
            metadata["checksum_verified"] = True
        else:
            metadata["checksum_verified"] = "source_missing"
    else:
        metadata["checksum_verified"] = "source_missing"

    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        limit = min(limit, n_graphs)
        if limit < n_graphs:
            metadata["cache_n_graphs"] = n_graphs
            metadata["limit_applied"] = limit
            metadata["n_graphs"] = limit
            adjacency = adjacency[:limit]
            global_node_ids = global_node_ids[:limit]
            roots = roots[:limit]
            edge_counts = edge_counts[:limit]
            avg_degrees = avg_degrees[:limit]
            density_percent = density_percent[:limit]

    return MentorSubgraphBundle(
        adjacency=adjacency,
        global_node_ids=global_node_ids,
        vocab=vocab,
        roots=roots,
        edge_counts=edge_counts,
        avg_degrees=avg_degrees,
        density_percent=density_percent,
        metadata=metadata,
    )


def load_bundle(
    *,
    source_pkl: str | Path | None = None,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    force: bool = False,
) -> MentorSubgraphBundle:
    """加载缓存；缓存缺失/过期时从源 pkl 构建并保存。"""
    source_pkl = Path(source_pkl) if source_pkl is not None else default_source_path()
    cache_path = Path(cache_path) if cache_path is not None else default_cache_path(source_pkl)
    if cache_path.resolve() == source_pkl.resolve():
        raise ValueError("cache path must differ from the source pkl (the pkl is never overwritten)")
    if cache_path.exists() and not force:
        return load_cache(cache_path, limit=limit)
    graphs = load_graph_list(source_pkl)
    bundle = bundle_from_graphs(graphs, source_path=source_pkl, limit=limit)
    save_bundle(bundle, cache_path)
    return bundle


def _balanced_quotas(counts: np.ndarray, n_pilot: int) -> np.ndarray:
    """按分层均衡分配额度，短缺部分确定性地重新分配。"""
    counts = np.asarray(counts, dtype=np.int64)
    remaining = min(int(n_pilot), int(counts.sum()))
    quotas = np.zeros_like(counts)
    while remaining > 0:
        active = np.flatnonzero(quotas < counts)
        if active.size == 0:
            break
        share, extra = divmod(remaining, int(active.size))
        proposed = np.full(active.size, share, dtype=np.int64)
        proposed[:extra] += 1
        capacity = counts[active] - quotas[active]
        added = np.minimum(proposed, capacity)
        quotas[active] += added
        used = int(added.sum())
        if used == 0:
            break
        remaining -= used
    return quotas


def select_pilot_indices(
    bundle: MentorSubgraphBundle,
    n_pilot: int = DEFAULT_PILOT_SIZE,
    seed: int = DEFAULT_PILOT_SEED,
) -> PilotSelection:
    """root-aware 分层 pilot 抽样（固定 seed 下完全确定）。

    各分层额度均衡（某层过小则确定性再分配）；层内先取根未被选过的图，
    全局稀有根优先、平局用 seed 扰动打散，再补足同顺序的重复根。
    """
    n_pilot = int(n_pilot)
    if n_pilot < 0:
        raise ValueError("n_pilot must be >= 0")
    seed = int(seed)
    roots = bundle.roots
    strata = density_stratum(bundle.avg_degrees)
    n_graphs = bundle.n_graphs
    n_pilot = min(n_pilot, n_graphs)
    counts = np.bincount(strata, minlength=len(STRATUM_NAMES))
    quotas = _balanced_quotas(counts, n_pilot)
    root_freq = np.bincount(roots.astype(np.int64), minlength=len(bundle.vocab))
    rng = np.random.default_rng(seed)

    selected: list[int] = []
    selected_counts: list[int] = []
    for s in range(len(STRATUM_NAMES)):
        quota = int(quotas[s])
        indices = np.flatnonzero(strata == s)
        if quota == 0 or indices.size == 0:
            selected_counts.append(0)
            continue
        randomized_tie = rng.permutation(indices.size)
        order = np.lexsort(
            (indices, randomized_tie, root_freq[roots[indices].astype(np.int64)])
        )
        ordered_indices = indices[order]
        chosen: list[int] = []
        seen_roots: set[int] = set()
        for raw_index in ordered_indices:
            if len(chosen) >= quota:
                break
            graph_index = int(raw_index)
            root = int(roots[graph_index])
            if root not in seen_roots:
                seen_roots.add(root)
                chosen.append(graph_index)
        chosen_set = set(chosen)
        for raw_index in ordered_indices:
            if len(chosen) >= quota:
                break
            graph_index = int(raw_index)
            if graph_index not in chosen_set:
                chosen.append(graph_index)
                chosen_set.add(graph_index)
        selected.extend(chosen)
        selected_counts.append(len(chosen))

    indices = np.sort(np.asarray(selected, dtype=np.int64))
    n_distinct_roots = int(np.unique(roots[indices]).size)
    return PilotSelection(
        indices=indices,
        strata=strata,
        quotas=tuple(int(q) for q in quotas),
        selected_counts=tuple(selected_counts),
        n_pilot=n_pilot,
        seed=seed,
        n_distinct_roots=n_distinct_roots,
    )


# ===========================================================================
# C. GLOBAL-WL 稳定排序（code/global_stable_ids.py）
#    计算与节点编号无关的结构签名（度/三角/k-core/距离直方图/闭游走），
#    再做 WL 迭代精化得到结构类；automorphism 等价类内身份不可识别，
#    只保证"类"与"重排序后的邻接"在重编号下不变。
# ===========================================================================

Digest = bytes


@dataclass(frozen=True)
class StableNodeIDs:
    """稳定的结构类加上保守的唯一 ID 诊断信息。

    canonical_ids 仅在 unique_mask 为真的节点上精确；非单例类内部的
    具体节点身份不可识别，只用输入顺序做不透明占位（这正是"重编号稳定
    但 automorphism 等价类内不可识别"的含义）。
    """

    method: str
    canonical_ids: tuple[int, ...]
    class_ids: tuple[int, ...]
    unique_mask: tuple[bool, ...]
    order: tuple[int, ...]
    class_sizes: tuple[int, ...]
    class_keys: tuple[tuple, ...]

    @property
    def singleton_fraction(self) -> float:
        return float(np.mean(self.unique_mask)) if self.unique_mask else 0.0

    @property
    def fully_singleton(self) -> bool:
        return all(self.unique_mask)

    @property
    def largest_class(self) -> int:
        return max(self.class_sizes, default=0)

    @property
    def ambiguous_class_count(self) -> int:
        return sum(size > 1 for size in self.class_sizes)


def _integer_core_numbers(adjacency: np.ndarray) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    maximum_degree = int(np.max(np.sum(adjacency, axis=1), initial=0))
    core = np.zeros(n_nodes, dtype=np.int64)
    for k in range(1, maximum_degree + 1):
        alive = np.ones(n_nodes, dtype=bool)
        while True:
            alive_indices = np.flatnonzero(alive)
            if alive_indices.size == 0:
                break
            degrees = np.sum(adjacency[np.ix_(alive_indices, alive_indices)], axis=1)
            remove = alive_indices[degrees < k]
            if remove.size == 0:
                break
            alive[remove] = False
        core[alive] = k
    return core


def _distance_histograms(adjacency: np.ndarray) -> tuple[tuple[int, ...], ...]:
    n_nodes = adjacency.shape[0]
    output = []
    for root in range(n_nodes):
        distances = np.full(n_nodes, n_nodes + 1, dtype=np.int64)
        distances[root] = 0
        queue = [root]
        for node in queue:
            for raw_neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(raw_neighbor)
                if distances[neighbor] > distances[node] + 1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        if np.any(distances > n_nodes):
            raise ValueError("stable IDs require a connected graph")
        histogram = np.bincount(distances, minlength=n_nodes).astype(np.int64)
        last = int(np.max(distances))
        output.append(tuple(int(value) for value in histogram[1 : last + 1]))
    return tuple(output)


def structural_signatures(adjacency: np.ndarray) -> tuple[tuple, ...]:
    """返回只依赖结构、与数值节点 ID 无关的整型初始签名。

    每个节点用 (度, 三角数, k-core, 到各节点的距离直方图, 闭游走数)
    五元组作初始颜色，为后续 WL 精化提供可复现的起点。
    """
    values = np.asarray(adjacency, dtype=np.int8)
    validate_simple_adjacency(values)
    degrees = np.sum(values, axis=1).astype(np.int64)
    triangles = np.diag(
        values.astype(np.int64) @ values.astype(np.int64) @ values.astype(np.int64)
    ) // 2
    cores = _integer_core_numbers(values)
    distances = _distance_histograms(values)
    power = np.eye(values.shape[0], dtype=np.int64)
    matrix = values.astype(np.int64)
    closed_walks: list[np.ndarray] = []
    for exponent in range(1, 6):
        power = power @ matrix
        if exponent >= 2:
            closed_walks.append(np.diag(power).copy())
    return tuple(
        (
            int(degrees[node]),
            int(triangles[node]),
            int(cores[node]),
            distances[node],
            tuple(int(walk[node]) for walk in closed_walks),
        )
        for node in range(values.shape[0])
    )


def _digest_parts(parts: Sequence[bytes]) -> Digest:
    digest = sha256()
    for part in parts:
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def _initial_digest(signature: tuple, *, rooted: bool) -> Digest:
    return _digest_parts(
        (b"stable-node-id-v1", repr(signature).encode("ascii"), bytes((int(rooted),)))
    )


def _wl_digest_refinement(
    adjacency: np.ndarray,
    signatures: Sequence[tuple],
    *,
    root: int | None,
    rounds: int | None = None,
) -> tuple[tuple[Digest, ...], tuple[tuple[int, ...], ...]]:
    n_nodes = adjacency.shape[0]
    count = n_nodes if rounds is None else int(rounds)
    if count < 1:
        raise ValueError("round count must be positive")
    digests = tuple(
        _initial_digest(signature, rooted=(root == node))
        for node, signature in enumerate(signatures)
    )
    history = []
    for _iteration in range(count):
        history.append(tuple(sorted(Counter(digests).values())))
        updated = []
        for node in range(n_nodes):
            neighbors = sorted(digests[int(value)] for value in np.flatnonzero(adjacency[node]))
            updated.append(_digest_parts((b"wl", digests[node], *neighbors)))
        digests = tuple(updated)
    history.append(tuple(sorted(Counter(digests).values())))
    return digests, tuple(history)


def _rooted_fingerprint(
    adjacency: np.ndarray,
    signatures: Sequence[tuple],
    root: int,
) -> tuple:
    digests, history = _wl_digest_refinement(adjacency, signatures, root=root)
    cells: dict[Digest, list[int]] = {}
    for node, digest in enumerate(digests):
        cells.setdefault(digest, []).append(node)
    ordered_digests = tuple(sorted(cells))
    cell_sizes = tuple(len(cells[digest]) for digest in ordered_digests)
    quotient = []
    for left_digest in ordered_digests:
        left = np.asarray(cells[left_digest], dtype=np.int64)
        for right_digest in ordered_digests:
            right = np.asarray(cells[right_digest], dtype=np.int64)
            quotient.append(int(np.sum(adjacency[np.ix_(left, right)])))
    return (
        digests[root],
        history,
        ordered_digests,
        cell_sizes,
        tuple(quotient),
    )


def _build_result(method: str, keys: Sequence[tuple]) -> StableNodeIDs:
    node_keys = tuple(keys)
    unique_keys = tuple(sorted(set(node_keys)))
    key_to_class = {key: index for index, key in enumerate(unique_keys)}
    class_ids = tuple(key_to_class[key] for key in node_keys)
    members = {
        class_id: tuple(node for node, value in enumerate(class_ids) if value == class_id)
        for class_id in range(len(unique_keys))
    }
    class_sizes = tuple(len(members[index]) for index in range(len(unique_keys)))
    unique_mask = tuple(class_sizes[class_ids[node]] == 1 for node in range(len(node_keys)))
    order = tuple(
        node
        for class_id in range(len(unique_keys))
        for node in members[class_id]
    )
    canonical_ids_list = [0] * len(node_keys)
    for canonical_id, node in enumerate(order):
        canonical_ids_list[node] = canonical_id
    return StableNodeIDs(
        method=method,
        canonical_ids=tuple(canonical_ids_list),
        class_ids=class_ids,
        unique_mask=unique_mask,
        order=order,
        class_sizes=class_sizes,
        class_keys=unique_keys,
    )


def compute_global_wl_ids(adjacency: np.ndarray) -> StableNodeIDs:
    values = np.asarray(adjacency, dtype=np.int8)
    signatures = structural_signatures(values)
    digests, _history = _wl_digest_refinement(values, signatures, root=None)
    keys = tuple((digest,) for digest in digests)
    return _build_result("global_wl", keys)


def reorder_by_stable_ids(
    adjacency: np.ndarray, ids: StableNodeIDs
) -> np.ndarray:
    values = np.asarray(adjacency, dtype=np.int8)
    if values.shape != (len(ids.order), len(ids.order)):
        raise ValueError("stable ID order is incompatible with adjacency")
    order = np.asarray(ids.order, dtype=np.int64)
    return values[np.ix_(order, order)].copy()


# ===========================================================================
# D. 连续重叠 patch 覆盖（code/overlap_cover.py 的 beam8 相关子集）
#    PatchCover 数据结构、预算、边目标桥接链采样、覆盖审计。
#    连续覆盖 = 相邻 patch 恰好共享 target_overlap 个节点。
# ===========================================================================

@dataclass(frozen=True)
class OrderedPatch:
    node_ids: tuple[int, ...]
    center: int
    adjacency: np.ndarray


@dataclass(frozen=True)
class PatchTransition:
    left_to_right_slots: tuple[tuple[int, int], ...]

    @property
    def overlap_size(self) -> int:
        return len(self.left_to_right_slots)


@dataclass(frozen=True)
class PatchCover:
    method: str
    patches: tuple[OrderedPatch, ...]
    transitions: tuple[PatchTransition, ...]
    segment_ids: tuple[int, ...]
    target_edges: tuple[tuple[int, int] | None, ...]
    bridge_lengths: tuple[int, ...]


def patch_budget(
    adjacency: np.ndarray,
    *,
    patch_size: int,
    target_overlap: int,
    edge_capacity_multiplier: float = 1.5,
) -> int:
    """按节点与边容量下界挑选匹配的 patch 预算。

    node_bound 保证滑动窗口能盖住全部节点；edge_bound 保证块内节点对
    容量能容纳全部边，edge_capacity_multiplier=1.5 留余量。
    """
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if not 1 <= target_overlap < patch_size <= n_nodes:
        raise ValueError("expected 1 <= target_overlap < patch_size <= n_nodes")
    stride = patch_size - target_overlap
    node_bound = 1 + int(np.ceil(max(n_nodes - patch_size, 0) / stride))
    edge_count = int(adjacency.sum() // 2)
    pair_capacity = patch_size * (patch_size - 1) // 2
    edge_bound = int(np.ceil(edge_count / pair_capacity))
    return max(node_bound, int(np.ceil(edge_capacity_multiplier * edge_bound)))


def _make_patch(adjacency: np.ndarray, node_ids: Sequence[int], center: int) -> OrderedPatch:
    nodes = tuple(int(node) for node in node_ids)
    if len(nodes) != len(set(nodes)):
        raise ValueError("patch nodes must be distinct")
    if center not in nodes:
        raise ValueError("patch center must belong to the patch")
    induced = adjacency[np.ix_(nodes, nodes)].astype(np.int8, copy=True)
    return OrderedPatch(node_ids=nodes, center=int(center), adjacency=induced)


def _transition(left: OrderedPatch, right: OrderedPatch) -> PatchTransition:
    right_slots = {node: slot for slot, node in enumerate(right.node_ids)}
    pairs = tuple(
        (left_slot, right_slots[node])
        for left_slot, node in enumerate(left.node_ids)
        if node in right_slots
    )
    return PatchTransition(left_to_right_slots=pairs)


def _make_cover(
    method: str,
    patches: Sequence[OrderedPatch],
    segment_ids: Sequence[int] | None = None,
    target_edges: Sequence[tuple[int, int] | None] | None = None,
    bridge_lengths: Sequence[int] | None = None,
) -> PatchCover:
    patches = tuple(patches)
    if not patches:
        raise ValueError("a patch cover cannot be empty")
    if segment_ids is None:
        segments = tuple(0 for _ in patches)
    else:
        segments = tuple(int(value) for value in segment_ids)
        if len(segments) != len(patches):
            raise ValueError("segment_ids must match the patch count")
        if segments[0] != 0 or any(
            right not in (left, left + 1) for left, right in zip(segments, segments[1:])
        ):
            raise ValueError("segment_ids must start at zero and increase contiguously")
    if target_edges is None:
        targets = tuple(None for _ in patches)
    else:
        targets = tuple(
            None if edge is None else tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in target_edges
        )
        if len(targets) != len(patches):
            raise ValueError("target_edges must match the patch count")
    if bridge_lengths is None:
        bridges = tuple(0 for _ in patches)
    else:
        bridges = tuple(int(value) for value in bridge_lengths)
        if len(bridges) != len(patches) or any(value < 0 for value in bridges):
            raise ValueError("bridge_lengths must be non-negative and match patches")
    transitions = tuple(_transition(left, right) for left, right in zip(patches, patches[1:]))
    return PatchCover(
        method=method,
        patches=patches,
        transitions=transitions,
        segment_ids=segments,
        target_edges=targets,
        bridge_lengths=bridges,
    )


def _distances_to_set(adjacency: np.ndarray, targets: set[int]) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    distance = np.full(n_nodes, n_nodes + 1, dtype=np.int64)
    frontier = list(targets)
    for node in frontier:
        distance[node] = 0
    cursor = 0
    while cursor < len(frontier):
        node = frontier[cursor]
        cursor += 1
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if distance[value] > distance[node] + 1:
                distance[value] = distance[node] + 1
                frontier.append(value)
    return distance


def _candidate_score(
    adjacency: np.ndarray,
    candidate: int,
    current_nodes: Sequence[int],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    new_pairs = sum(
        tuple(sorted((candidate, node))) not in observed_pairs
        for node in current_nodes
        if node != candidate
    )
    new_edges = sum(
        adjacency[candidate, node] != 0
        and tuple(sorted((candidate, node))) not in covered_edges
        for node in current_nodes
        if node != candidate
    )
    boundary_degree = int(np.count_nonzero(adjacency[candidate]))
    return (int(new_pairs), int(new_edges), int(candidate not in covered_nodes), boundary_degree)


def _select_best(
    candidates: Iterable[int],
    score,
    rng: np.random.Generator,
) -> int:
    candidates = tuple(sorted(set(int(node) for node in candidates)))
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    scored = [(score(node), node) for node in candidates]
    best = max(item[0] for item in scored)
    choices = [node for value, node in scored if value == best]
    return int(rng.choice(choices))


def _observed_pairs_from_nodes(nodes: Sequence[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(nodes, 2)}


def _frontier_expand(
    adjacency: np.ndarray,
    seed_nodes: Sequence[int],
    rng: np.random.Generator,
    *,
    patch_size: int,
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
    forbidden: set[int] | None = None,
) -> list[int]:
    nodes = list(dict.fromkeys(int(node) for node in seed_nodes))
    forbidden = set() if forbidden is None else set(forbidden)
    n_nodes = adjacency.shape[0]
    while len(nodes) < patch_size:
        node_set = set(nodes)
        candidates = {
            int(neighbor)
            for node in nodes
            for neighbor in np.flatnonzero(adjacency[node])
            if int(neighbor) not in node_set and int(neighbor) not in forbidden
        }
        if not candidates:
            distances = _distances_to_set(adjacency, node_set)
            eligible = [
                node for node in range(n_nodes) if node not in node_set and node not in forbidden
            ]
            if not eligible:
                raise RuntimeError("no eligible frontier node remains")
            minimum = min(int(distances[node]) for node in eligible)
            candidates = {node for node in eligible if distances[node] == minimum}
        selected = _select_best(
            candidates,
            lambda candidate: _candidate_score(
                adjacency,
                candidate,
                nodes,
                observed_pairs,
                covered_nodes,
                covered_edges,
            ),
            rng,
        )
        nodes.append(selected)
    return nodes


def _true_edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    return {
        (left, right)
        for left in range(adjacency.shape[0])
        for right in range(left + 1, adjacency.shape[0])
        if adjacency[left, right] != 0
    }


def _register_nodes(
    adjacency: np.ndarray,
    nodes: Sequence[int],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> None:
    pairs = _observed_pairs_from_nodes(nodes)
    observed_pairs.update(pairs)
    covered_nodes.update(nodes)
    covered_edges.update(pair for pair in pairs if adjacency[pair[0], pair[1]] != 0)


def _uncovered_edge_deficits(
    n_nodes: int,
    true_edges: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
) -> np.ndarray:
    deficits = np.zeros(n_nodes, dtype=np.int64)
    for left, right in true_edges - covered_edges:
        deficits[left] += 1
        deficits[right] += 1
    return deficits


def _ordered_uncovered_edges(
    adjacency: np.ndarray,
    true_edges: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    deficits = _uncovered_edge_deficits(adjacency.shape[0], true_edges, covered_edges)
    buckets: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for edge in true_edges - covered_edges:
        left, right = edge
        key = (
            int(deficits[left] + deficits[right]),
            int(left not in covered_nodes) + int(right not in covered_nodes),
            int(min(deficits[left], deficits[right])),
        )
        buckets.setdefault(key, []).append(edge)
    ordered: list[tuple[int, int]] = []
    for key in sorted(buckets, reverse=True):
        values = sorted(buckets[key])
        permutation = rng.permutation(len(values))
        ordered.extend(values[int(index)] for index in permutation)
    return ordered


def _shortest_path_avoiding(
    adjacency: np.ndarray,
    start: int,
    target: int,
    blocked: set[int],
) -> list[int] | None:
    if start == target:
        return [int(start)]
    parent = {int(start): -1}
    queue = [int(start)]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for raw_neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(raw_neighbor)
            if neighbor in blocked or neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == target:
                path = [neighbor]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    return None


def _choose_target_bundle(
    adjacency: np.ndarray,
    previous_nodes: Sequence[int],
    true_edges: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
    *,
    new_slot_count: int,
) -> tuple[tuple[int, int], int, list[int], int]:
    previous_set = set(previous_nodes)
    for edge in _ordered_uncovered_edges(
        adjacency, true_edges, covered_edges, covered_nodes, rng
    ):
        if edge[0] in previous_set or edge[1] in previous_set:
            continue
        feasible: list[tuple[int, int, list[int], int]] = []
        for orientation in (edge, (edge[1], edge[0])):
            target, partner = orientation
            for anchor in previous_nodes:
                blocked = previous_set - {int(anchor)}
                path = _shortest_path_avoiding(adjacency, int(anchor), int(target), blocked)
                if path is None:
                    continue
                bundle = list(dict.fromkeys(path[1:] + [int(partner)]))
                if len(bundle) <= new_slot_count:
                    feasible.append((len(bundle), int(anchor), bundle, len(path) - 1))
        if feasible:
            shortest = min(item[0] for item in feasible)
            choices = [item for item in feasible if item[0] == shortest]
            selected = choices[int(rng.integers(len(choices)))]
            _, anchor, bundle, bridge_length = selected
            return edge, anchor, bundle, bridge_length
    raise RuntimeError("no feasible uncovered target edge fits the next patch")


def _select_retained_for_bundle(
    adjacency: np.ndarray,
    previous_nodes: Sequence[int],
    anchor: int,
    bundle: Sequence[int],
    rng: np.random.Generator,
    *,
    target_overlap: int,
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> list[int]:
    retained = [int(anchor)]
    remaining = set(int(node) for node in previous_nodes) - {int(anchor)}
    while len(retained) < target_overlap:
        connected = {
            node
            for node in remaining
            if any(adjacency[node, kept] != 0 for kept in retained)
        }
        candidates = connected or remaining
        selected = _select_best(
            candidates,
            lambda node: _candidate_score(
                adjacency,
                node,
                retained + list(bundle),
                observed_pairs,
                covered_nodes,
                covered_edges,
            ),
            rng,
        )
        retained.append(selected)
        remaining.remove(selected)
    return retained


def _target_seed_patch(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    patch_size: int,
    true_edges: set[tuple[int, int]],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> tuple[OrderedPatch, tuple[int, int]]:
    ordered_targets = _ordered_uncovered_edges(
        adjacency, true_edges, covered_edges, covered_nodes, rng
    )
    if not ordered_targets:
        ordered_targets = sorted(true_edges)
    target = ordered_targets[0]
    nodes = _frontier_expand(
        adjacency,
        list(target),
        rng,
        patch_size=patch_size,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    return _make_patch(adjacency, nodes, target[0]), target


def _target_bridge_next_patch(
    adjacency: np.ndarray,
    previous: OrderedPatch,
    rng: np.random.Generator,
    *,
    patch_size: int,
    target_overlap: int,
    true_edges: set[tuple[int, int]],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> tuple[OrderedPatch, tuple[int, int], int]:
    target, anchor, bundle, bridge_length = _choose_target_bundle(
        adjacency,
        previous.node_ids,
        true_edges,
        covered_edges,
        covered_nodes,
        rng,
        new_slot_count=patch_size - target_overlap,
    )
    retained = _select_retained_for_bundle(
        adjacency,
        previous.node_ids,
        anchor,
        bundle,
        rng,
        target_overlap=target_overlap,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    forbidden = set(previous.node_ids) - set(retained)
    nodes = _frontier_expand(
        adjacency,
        retained + bundle,
        rng,
        patch_size=patch_size,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
        forbidden=forbidden,
    )
    return _make_patch(adjacency, nodes, target[0]), target, bridge_length


def sample_edge_target_bridge_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
) -> PatchCover:
    """一条连续链，目标边朝全局未覆盖边推进（边目标桥接）。

    首块从最高优先级的未覆盖边扩散出 patch；之后每块通过"目标未覆盖边
    + 保留锚点 + 最短桥接束"延续链，块间恰好共享 target_overlap 个节点。
    """
    validate_simple_adjacency(adjacency)
    if not is_connected(adjacency):
        raise ValueError("target bridge cover requires a connected graph")
    if not 0 < target_overlap < patch_size:
        raise ValueError("target_overlap must be between zero and patch_size")
    true_edges = _true_edge_set(adjacency)
    observed_pairs: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    covered_edges: set[tuple[int, int]] = set()
    patches: list[OrderedPatch] = []
    targets: list[tuple[int, int]] = []
    bridges: list[int] = []

    first, target = _target_seed_patch(
        adjacency,
        rng,
        patch_size=patch_size,
        true_edges=true_edges,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    patches.append(first)
    targets.append(target)
    bridges.append(0)
    _register_nodes(adjacency, first.node_ids, observed_pairs, covered_nodes, covered_edges)
    while len(patches) < n_patches:
        patch, target, bridge_length = _target_bridge_next_patch(
            adjacency,
            patches[-1],
            rng,
            patch_size=patch_size,
            target_overlap=target_overlap,
            true_edges=true_edges,
            observed_pairs=observed_pairs,
            covered_nodes=covered_nodes,
            covered_edges=covered_edges,
        )
        patches.append(patch)
        targets.append(target)
        bridges.append(bridge_length)
        _register_nodes(adjacency, patch.node_ids, observed_pairs, covered_nodes, covered_edges)
    return _make_cover(
        "edge_target_bridge",
        patches,
        target_edges=targets,
        bridge_lengths=bridges,
    )


def audit_cover(adjacency: np.ndarray, cover: PatchCover) -> dict[str, Any]:
    """测量一个有序覆盖的连续性、覆盖率与精确拼接指标。"""
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    true_edges = {
        (left, right)
        for left in range(n_nodes)
        for right in range(left + 1, n_nodes)
        if adjacency[left, right] != 0
    }
    all_pairs = set(combinations(range(n_nodes), 2))
    observed: dict[tuple[int, int], list[int]] = {}
    covered_nodes: set[int] = set()
    for patch in cover.patches:
        covered_nodes.update(patch.node_ids)
        for left_slot, right_slot in combinations(range(len(patch.node_ids)), 2):
            pair = tuple(sorted((patch.node_ids[left_slot], patch.node_ids[right_slot])))
            observed.setdefault(pair, []).append(int(patch.adjacency[left_slot, right_slot]))

    observed_pairs = set(observed)
    observed_edges = true_edges & observed_pairs
    conflicts = sum(len(set(values)) > 1 for values in observed.values())
    observed_correct = sum(
        int(round(float(np.mean(values)))) == int(adjacency[pair[0], pair[1]])
        for pair, values in observed.items()
    )
    predicted_edges = {
        pair for pair, values in observed.items() if float(np.mean(values)) >= 0.5
    }
    full_correct = len(
        (predicted_edges & true_edges) | ((all_pairs - predicted_edges) & (all_pairs - true_edges))
    )

    consecutive_overlap = []
    consecutive_jaccard = []
    center_distances = []
    new_nodes = []
    new_pairs = []
    running_nodes: set[int] = set()
    running_pairs: set[tuple[int, int]] = set()
    patch_connected = []
    persistent_slot_matches = 0
    persistent_slot_trials = 0
    for index, patch in enumerate(cover.patches):
        node_set = set(patch.node_ids)
        patch_pairs = _observed_pairs_from_nodes(patch.node_ids)
        new_nodes.append(len(node_set - running_nodes))
        new_pairs.append(len(patch_pairs - running_pairs))
        running_nodes.update(node_set)
        running_pairs.update(patch_pairs)
        patch_connected.append(is_connected(patch.adjacency))
        if index > 0 and cover.segment_ids[index] == cover.segment_ids[index - 1]:
            previous = set(cover.patches[index - 1].node_ids)
            overlap = len(previous & node_set)
            consecutive_overlap.append(overlap)
            consecutive_jaccard.append(overlap / len(previous | node_set))
            distances = _distances_to_set(adjacency, {cover.patches[index - 1].center})
            center_distances.append(int(distances[patch.center]))
            for left_slot, right_slot in cover.transitions[index - 1].left_to_right_slots:
                persistent_slot_trials += 1
                persistent_slot_matches += int(left_slot == right_slot)

    nonconsecutive_overlap = []
    nonconsecutive_jaccard = []
    for left in range(len(cover.patches)):
        for right in range(left + 1, len(cover.patches)):
            if right == left + 1 and cover.segment_ids[left] == cover.segment_ids[right]:
                continue
            left_nodes = set(cover.patches[left].node_ids)
            right_nodes = set(cover.patches[right].node_ids)
            overlap = len(left_nodes & right_nodes)
            nonconsecutive_overlap.append(overlap)
            nonconsecutive_jaccard.append(overlap / len(left_nodes | right_nodes))

    multiplicities = np.asarray(
        [len(observed[edge]) for edge in observed_edges], dtype=np.float64
    )
    mean_multiplicity = float(np.mean(multiplicities)) if multiplicities.size else 0.0
    cv_multiplicity = (
        float(np.std(multiplicities) / mean_multiplicity) if mean_multiplicity > 0 else 0.0
    )

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "method": cover.method,
        "patch_count": len(cover.patches),
        "patch_size": len(cover.patches[0].node_ids),
        "node_coverage": len(covered_nodes) / n_nodes,
        "true_edge_coverage": len(observed_edges) / max(len(true_edges), 1),
        "node_pair_coverage": len(observed_pairs) / len(all_pairs),
        "observed_pair_consistency": 1.0 - conflicts / max(len(observed_pairs), 1),
        "observed_pair_accuracy": observed_correct / max(len(observed_pairs), 1),
        "full_adjacency_accuracy": full_correct / len(all_pairs),
        "patch_connected_rate": float(np.mean(patch_connected)),
        "segment_count": int(max(cover.segment_ids) + 1),
        "continuous_transition_fraction": float(
            np.mean(
                [
                    cover.segment_ids[index] == cover.segment_ids[index - 1]
                    for index in range(1, len(cover.patches))
                ]
            )
        )
        if len(cover.patches) > 1
        else 1.0,
        "continuous_shared_slot_persistence_rate": float(
            persistent_slot_matches / persistent_slot_trials
        )
        if persistent_slot_trials
        else 1.0,
        "target_edge_hit_rate": float(
            np.mean(
                [
                    edge is None or set(edge).issubset(patch.node_ids)
                    for patch, edge in zip(cover.patches, cover.target_edges)
                ]
            )
        ),
        "bridge_length_mean": float(np.mean(cover.bridge_lengths)),
        "bridge_length_maximum": int(max(cover.bridge_lengths)),
        "edge_observation_multiplicity_mean": mean_multiplicity,
        "edge_observation_multiplicity_cv": cv_multiplicity,
        "consecutive_overlap_mean": mean(consecutive_overlap),
        "consecutive_jaccard_mean": mean(consecutive_jaccard),
        "nonconsecutive_overlap_mean": mean(nonconsecutive_overlap),
        "nonconsecutive_jaccard_mean": mean(nonconsecutive_jaccard),
        "consecutive_nonconsecutive_jaccard_gap": mean(consecutive_jaccard)
        - mean(nonconsecutive_jaccard),
        "within_segment_overlap_mean": mean(consecutive_overlap),
        "within_segment_jaccard_mean": mean(consecutive_jaccard),
        "within_segment_nonlocal_jaccard_gap": mean(consecutive_jaccard)
        - mean(nonconsecutive_jaccard),
        "consecutive_center_distance_mean": mean(center_distances),
        "new_nodes_per_patch_mean": mean(new_nodes),
        "new_pairs_per_patch_mean": mean(new_pairs),
        "transition_overlap_exact": [transition.overlap_size for transition in cover.transitions],
    }


def cover_vectors(cover: PatchCover) -> np.ndarray:
    """返回有序诱导邻接的上三角向量，供后续 KSVD 阶段使用。"""
    return np.stack([adjacency_to_upper_vector(patch.adjacency) for patch in cover.patches])


# ===========================================================================
# E. Beam8 边缘候选覆盖采样（code/marginal_candidate_cover.py）
#    对上一块的每个 o 组合枚举保留子集，评估其"向外继续覆盖"的潜力，
#    只保留前 beam=8 个候选做贪心补全，再按新增边/新增节点对/新增节点
#    排序选最优。这就是"Beam8 连续重叠采样"。
# ===========================================================================

def _pairs(nodes: Sequence[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(nodes, 2)}


def _candidate_score_marginal(
    adjacency: np.ndarray,
    nodes: Sequence[int],
    observed_pairs: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
) -> tuple[int, int, int, int]:
    pairs = _pairs(nodes)
    edges = {pair for pair in pairs if adjacency[pair[0], pair[1]] != 0}
    return (
        len(edges - covered_edges),
        len(pairs - observed_pairs),
        len(set(nodes) - covered_nodes),
        len(edges),
    )


def _retained_potential(
    adjacency: np.ndarray,
    retained: Sequence[int],
    previous_set: set[int],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
) -> tuple[int, int, int]:
    outside = set(range(adjacency.shape[0])) - previous_set
    boundary_uncovered = 0
    boundary_edges = 0
    unseen_neighbors = set()
    for node in retained:
        for raw_neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(raw_neighbor)
            if neighbor not in outside:
                continue
            boundary_edges += 1
            pair = tuple(sorted((int(node), neighbor)))
            boundary_uncovered += int(pair not in covered_edges)
            if neighbor not in covered_nodes:
                unseen_neighbors.add(neighbor)
    return boundary_uncovered, len(unseen_neighbors), boundary_edges


def _greedy_fill(
    adjacency: np.ndarray,
    retained: Sequence[int],
    previous_set: set[int],
    observed_pairs: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
    *,
    patch_size: int,
) -> tuple[int, ...] | None:
    nodes = list(retained)
    forbidden = previous_set - set(retained)
    while len(nodes) < patch_size:
        node_set = set(nodes)
        frontier = sorted(
            {
                int(neighbor)
                for node in nodes
                for neighbor in np.flatnonzero(adjacency[node])
                if int(neighbor) not in node_set and int(neighbor) not in forbidden
            }
        )
        if not frontier:
            return None
        scored = []
        for candidate in frontier:
            new_edge_increment = 0
            new_pair_increment = 0
            for node in nodes:
                pair = tuple(sorted((candidate, int(node))))
                new_pair_increment += int(pair not in observed_pairs)
                new_edge_increment += int(
                    adjacency[candidate, node] != 0 and pair not in covered_edges
                )
            deficit = sum(
                tuple(sorted((candidate, int(neighbor)))) not in covered_edges
                for neighbor in np.flatnonzero(adjacency[candidate])
            )
            score = (
                new_edge_increment,
                int(deficit),
                new_pair_increment,
                int(candidate not in covered_nodes),
                int(np.count_nonzero(adjacency[candidate])),
            )
            scored.append((score, candidate))
        best = max(score for score, _candidate in scored)
        choices = [candidate for score, candidate in scored if score == best]
        nodes.append(int(choices[int(rng.integers(len(choices)))]))
    return tuple(nodes)


def sample_marginal_candidate_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
    retained_beam: int = 32,
    candidate_restarts: int = 2,
    allow_partial: bool = False,
) -> PatchCover:
    """保留子集 beam 搜索的一步边际最大化近似覆盖。

    对上一块的每个 o 组合保留子集：先按"边界未覆盖边 / 新邻居 / 边界边"
    潜力排序取前 retained_beam 个，再每个用贪心补全到整块（可重启多次），
    最后按新增边 / 新增对 / 新增节点选最优。这就是"Beam8"。
    """
    validate_simple_adjacency(adjacency)
    if retained_beam < 1 or candidate_restarts < 1:
        raise ValueError("beam and restarts must be positive")
    first_cover = sample_edge_target_bridge_cover(
        adjacency,
        rng,
        n_patches=1,
        patch_size=patch_size,
        target_overlap=target_overlap,
    )
    patches = [first_cover.patches[0]]
    # 已观测节点对、已覆盖边、已覆盖节点三个记账集合，用于计算边际收益。
    observed_pairs = _pairs(patches[0].node_ids)
    covered_edges = {
        pair for pair in observed_pairs if adjacency[pair[0], pair[1]] != 0
    }
    covered_nodes = set(patches[0].node_ids)

    def register(nodes: Sequence[int]) -> None:
        """新块被采纳后，更新全部记账集合。"""
        pairs = _pairs(nodes)
        observed_pairs.update(pairs)
        covered_edges.update(pair for pair in pairs if adjacency[pair[0], pair[1]] != 0)
        covered_nodes.update(nodes)

    while len(patches) < n_patches:
        previous = patches[-1]
        previous_set = set(previous.node_ids)
        # 1) 枚举上一块的所有 o 组合保留子集，筛掉不连通的，按"向外潜力"排序
        retained_candidates = []
        for retained in combinations(previous.node_ids, target_overlap):
            induced = adjacency[np.ix_(retained, retained)]
            if not is_connected(induced):
                continue
            potential = _retained_potential(
                adjacency,
                retained,
                previous_set,
                covered_edges,
                covered_nodes,
            )
            retained_candidates.append((potential, retained))
        # 2) 只保留潜力最高的 beam 个保留子集
        retained_candidates.sort(key=lambda item: item[0], reverse=True)
        retained_candidates = retained_candidates[:retained_beam]
        # 3) 对每个保留子集做贪心补全到整块（可重启多次，增加多样性）
        completed = []
        for _potential, retained in retained_candidates:
            for _restart in range(candidate_restarts):
                candidate = _greedy_fill(
                    adjacency,
                    retained,
                    previous_set,
                    observed_pairs,
                    covered_edges,
                    covered_nodes,
                    rng,
                    patch_size=patch_size,
                )
                if candidate is not None:
                    completed.append(candidate)
        if not completed:
            if allow_partial:
                break
            raise RuntimeError("marginal candidate beam produced no connected next patch")
        # 4) 按 (新增边, 新增对, 新增节点) 字典序选最优候选，采纳并记账
        scores = [
            _candidate_score_marginal(
                adjacency, candidate, observed_pairs, covered_edges, covered_nodes
            )
            for candidate in completed
        ]
        best = max(scores)
        choices = [
            candidate for score, candidate in zip(scores, completed) if score == best
        ]
        selected = choices[int(rng.integers(len(choices)))]
        new_nodes = [node for node in selected if node not in previous_set]
        center = new_nodes[0] if new_nodes else selected[0]
        patches.append(_make_patch(adjacency, selected, center))
        register(selected)
    return _make_cover("marginal_candidate_beam", patches)


# ===========================================================================
# F. Rooted-canonical 槽位排序（code/canonical_slots.py）
#    把每个 patch 的局部邻接重排成与 ID 无关的规范顺序：根节点固定在第
#    一位，其余按精确图同构 canonical 搜索求字典序最小的上三角码，得到
#    可复现、可跨图比较的局部邻接向量。
# ===========================================================================

@dataclass(frozen=True)
class CanonicalOrderResult:
    node_ids: tuple[int, ...]
    adjacency_code: tuple[int, ...]
    search_leaves: int
    optimal_leaf_count: int
    symmetry_pruned: bool

    @property
    def ambiguous(self) -> bool:
        return self.optimal_leaf_count > 1 or self.symmetry_pruned


def _upper_code(adjacency: np.ndarray, order: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(adjacency[order[left], order[right]])
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )


def _refine(
    adjacency: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    current = partition
    while True:
        refined: list[tuple[int, ...]] = []
        for cell in current:
            buckets: dict[tuple[int, ...], list[int]] = {}
            for vertex in cell:
                signature = tuple(
                    int(np.sum(adjacency[vertex, np.asarray(block, dtype=np.int64)]))
                    for block in current
                )
                buckets.setdefault(signature, []).append(vertex)
            for signature in sorted(buckets):
                refined.append(tuple(sorted(buckets[signature])))
        updated = tuple(refined)
        if updated == current:
            return current
        current = updated


def _swap_is_automorphism(adjacency: np.ndarray, left: int, right: int) -> bool:
    if left == right:
        return True
    for vertex in range(adjacency.shape[0]):
        if vertex in (left, right):
            continue
        if adjacency[left, vertex] != adjacency[right, vertex]:
            return False
    return True


def _branch_representatives(adjacency: np.ndarray, cell: tuple[int, ...]) -> tuple[int, ...]:
    representatives: list[int] = []
    for vertex in cell:
        if any(_swap_is_automorphism(adjacency, vertex, other) for other in representatives):
            continue
        representatives.append(vertex)
    return tuple(representatives)


def exact_canonical_order(
    patch_adjacency: np.ndarray,
    node_ids: Sequence[int],
    *,
    color_cells: Sequence[Sequence[int]] | None = None,
) -> CanonicalOrderResult:
    """返回精确的着色图 canonical 邻接序。

    颜色细化（1-WL 式）后对未定单元做分支搜索，找字典序最小的上三角
    邻接码；返回的是"邻接码"这个不变量，而不是 automorphic 节点映射
    的具体选择。
    """
    values = np.asarray(patch_adjacency, dtype=np.int8)
    nodes = tuple(int(node) for node in node_ids)
    n_nodes = len(nodes)
    if values.shape != (n_nodes, n_nodes):
        raise ValueError("patch_adjacency and node_ids are incompatible")
    if not np.array_equal(values, values.T) or np.any(np.diag(values) != 0):
        raise ValueError("expected a simple undirected patch adjacency")
    local = {node: index for index, node in enumerate(nodes)}
    if color_cells is None:
        partition = (tuple(range(n_nodes)),)
    else:
        seen: set[int] = set()
        converted = []
        for raw_cell in color_cells:
            cell = tuple(local[int(node)] for node in raw_cell)
            if not cell or any(vertex in seen for vertex in cell):
                raise ValueError("color_cells must be a nonempty ordered partition")
            seen.update(cell)
            converted.append(cell)
        if seen != set(range(n_nodes)):
            raise ValueError("color_cells must contain every patch node exactly once")
        partition = tuple(converted)

    leaves = 0
    best_code: tuple[int, ...] | None = None
    best_orders: list[tuple[int, ...]] = []
    symmetry_pruned = False

    def search(current: tuple[tuple[int, ...], ...]) -> None:
        nonlocal leaves, best_code, best_orders, symmetry_pruned
        current = _refine(values, current)
        target_index = next(
            (index for index, cell in enumerate(current) if len(cell) > 1),
            None,
        )
        if target_index is None:
            leaves += 1
            order = tuple(cell[0] for cell in current)
            code = _upper_code(values, order)
            if best_code is None or code < best_code:
                best_code = code
                best_orders = [order]
            elif code == best_code:
                best_orders.append(order)
            return
        cell = current[target_index]
        representatives = _branch_representatives(values, cell)
        symmetry_pruned = symmetry_pruned or len(representatives) < len(cell)
        for vertex in representatives:
            remainder = tuple(item for item in cell if item != vertex)
            individualized = (
                current[:target_index]
                + ((vertex,), remainder)
                + current[target_index + 1 :]
            )
            search(individualized)

    search(partition)
    if best_code is None or not best_orders:
        raise RuntimeError("canonical search produced no leaf")
    selected = min(best_orders, key=lambda order: tuple(nodes[index] for index in order))
    return CanonicalOrderResult(
        node_ids=tuple(nodes[index] for index in selected),
        adjacency_code=best_code,
        search_leaves=leaves,
        optimal_leaf_count=len(best_orders),
        symmetry_pruned=symmetry_pruned,
    )


def _distances(adjacency: np.ndarray, root: int) -> np.ndarray:
    distance = np.full(adjacency.shape[0], adjacency.shape[0] + 1, dtype=np.int64)
    distance[root] = 0
    queue = [root]
    for node in queue:
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if distance[value] > distance[node] + 1:
                distance[value] = distance[node] + 1
                queue.append(value)
    return distance


def signature_order(
    patch_adjacency: np.ndarray, node_ids: Sequence[int], center: int
) -> tuple[tuple[int, ...], bool]:
    values = np.asarray(patch_adjacency, dtype=np.int8)
    nodes = tuple(int(node) for node in node_ids)
    local = {node: index for index, node in enumerate(nodes)}
    center_local = local[int(center)]
    degrees = np.sum(values, axis=1).astype(np.int64)
    triangles = np.diag(values @ values @ values).astype(np.int64) // 2
    distances = _distances(values, center_local)
    colors = [int(value) for value in degrees]
    for _iteration in range(3):
        signatures = [
            (
                colors[index],
                tuple(sorted(colors[int(neighbor)] for neighbor in np.flatnonzero(values[index]))),
            )
            for index in range(len(nodes))
        ]
        vocabulary = {signature: rank for rank, signature in enumerate(sorted(set(signatures)))}
        colors = [vocabulary[signature] for signature in signatures]
    structural = {
        node: (
            int(node != center),
            int(distances[index]),
            -int(degrees[index]),
            -int(triangles[index]),
            int(colors[index]),
        )
        for index, node in enumerate(nodes)
    }
    tied = len(set(structural.values())) < len(nodes)
    return tuple(sorted(nodes, key=lambda node: structural[node] + (node,))), tied


def reorder_cover_structurally(
    adjacency: np.ndarray,
    cover: PatchCover,
    mode: str,
) -> tuple[PatchCover, dict[str, float | int]]:
    if mode not in {"id", "signature", "canonical", "rooted_canonical", "overlap_canonical"}:
        raise ValueError(f"unknown structural slot mode: {mode}")
    ordered_patches = []
    tie_count = 0
    ambiguity_count = 0
    leaf_counts = []
    previous_order: tuple[int, ...] | None = None
    for patch_index, patch in enumerate(cover.patches):
        if mode == "id":
            order = tuple(sorted(patch.node_ids))
        elif mode == "signature":
            order, tied = signature_order(patch.adjacency, patch.node_ids, patch.center)
            tie_count += int(tied)
        else:
            if mode == "canonical":
                colors = None
            elif mode == "rooted_canonical" or patch_index == 0:
                rest = tuple(node for node in patch.node_ids if node != patch.center)
                colors = ((patch.center,), rest) if rest else ((patch.center,),)
            else:
                assert previous_order is not None
                shared = [node for node in previous_order if node in set(patch.node_ids)]
                remainder = [node for node in patch.node_ids if node not in set(shared)]
                colors_list: list[tuple[int, ...]] = [(node,) for node in shared]
                if patch.center in remainder:
                    colors_list.append((patch.center,))
                    remainder.remove(patch.center)
                if remainder:
                    colors_list.append(tuple(remainder))
                colors = tuple(colors_list)
            result = exact_canonical_order(
                patch.adjacency,
                patch.node_ids,
                color_cells=colors,
            )
            order = result.node_ids
            ambiguity_count += int(result.ambiguous)
            leaf_counts.append(result.search_leaves)
        ordered_patches.append(_make_patch(adjacency, order, patch.center))
        previous_order = order
    reordered = _make_cover(
        mode,
        ordered_patches,
        cover.segment_ids,
        cover.target_edges,
        cover.bridge_lengths,
    )
    count = len(cover.patches)
    return reordered, {
        "patch_count": count,
        "signature_tie_rate": tie_count / max(count, 1),
        "canonical_ambiguity_rate": ambiguity_count / max(count, 1),
        "mean_search_leaves": float(np.mean(leaf_counts)) if leaf_counts else 0.0,
        "max_search_leaves": max(leaf_counts, default=0),
    }


# ===========================================================================
# G. 覆盖档位与率代理（code/beam8_edge_residual.py + beam8_coverage_operating_point.py）
#    沿同一条采样链逐前缀测量边/对/节点覆盖、残差结构与比特代理；
#    select_operating_checkpoints 从中选出 BASE / EDGE90/95/99/100 / FAIR95。
#    K=24 / T=3 的码率与字典标量也在此估算（乐观代理，不恢复图编码主张）。
# ===========================================================================

EPS = 1e-12
ACTIVATION_THRESHOLD = 1e-10


def log2_combination(n: int, k: int) -> float:
    """返回 log2(C(n, k))，避免构造超大整数组合数。"""
    if not 0 <= k <= n:
        raise ValueError("expected 0 <= k <= n")
    return float(
        (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
        / math.log(2.0)
    )


def log2_permutation(n: int, k: int) -> float:
    if not 0 <= k <= n:
        raise ValueError("expected 0 <= k <= n")
    return float((math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(2.0))


def true_edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    values = np.asarray(adjacency)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("adjacency must be square")
    return {
        (left, right)
        for left in range(values.shape[0])
        for right in range(left + 1, values.shape[0])
        if values[left, right] != 0
    }


def identity_proxy_bits(
    *,
    n_nodes: int,
    patch_size: int,
    overlap: int,
    patch_count: int,
) -> dict[str, int]:
    if not 0 <= patch_count:
        raise ValueError("patch_count must be nonnegative")
    if not 0 < overlap < patch_size <= n_nodes:
        raise ValueError("expected 0 < overlap < patch_size <= n_nodes")
    if patch_count == 0:
        return {"ordered_identity_bits": 0, "canonical_set_identity_bits": 0}
    new_count = patch_size - overlap
    outside_count = n_nodes - patch_size
    if new_count > outside_count:
        raise ValueError("not enough outside nodes for exact-overlap transition")
    retained = log2_combination(patch_size, overlap)
    ordered = log2_permutation(n_nodes, patch_size) + (patch_count - 1) * (
        retained + log2_permutation(outside_count, new_count)
    )
    canonical = (
        log2_combination(n_nodes, patch_size)
        + math.log2(patch_size)
        + (patch_count - 1)
        * (retained + log2_combination(outside_count, new_count) + math.log2(new_count))
    )
    return {
        "ordered_identity_bits": int(math.ceil(ordered)),
        "canonical_set_identity_bits": int(math.ceil(canonical)),
    }


def residual_subset_bits(unobserved_pair_count: int, residual_edge_count: int) -> int:
    if not 0 <= residual_edge_count <= unobserved_pair_count:
        raise ValueError("residual edges must be a subset of unobserved pairs")
    cardinality_bits = math.ceil(math.log2(unobserved_pair_count + 1))
    subset_bits = math.ceil(
        log2_combination(unobserved_pair_count, residual_edge_count)
    )
    return int(cardinality_bits + subset_bits)


def direct_enumerative_bits(pair_count: int, edge_count: int) -> int:
    if not 0 <= edge_count <= pair_count:
        raise ValueError("edge_count must lie inside pair universe")
    return int(
        math.ceil(math.log2(pair_count + 1))
        + math.ceil(log2_combination(pair_count, edge_count))
    )


CHECKPOINTS = ("BASE", "EDGE90", "EDGE95", "FAIR95", "EDGE99", "EDGE100")


def _bridge_edges(adjacency: np.ndarray) -> set[tuple[int, int]]:
    """用确定性 Tarjan 遍历返回图的所有桥边。"""
    values = np.asarray(adjacency, dtype=np.int8)
    n_nodes = values.shape[0]
    discovery = [-1] * n_nodes
    low = [-1] * n_nodes
    parent = [-1] * n_nodes
    timer = 0
    bridges: set[tuple[int, int]] = set()

    def visit(node: int) -> None:
        nonlocal timer
        discovery[node] = low[node] = timer
        timer += 1
        for raw_neighbor in np.flatnonzero(values[node]):
            neighbor = int(raw_neighbor)
            if discovery[neighbor] < 0:
                parent[neighbor] = node
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(n_nodes):
        if discovery[node] < 0:
            visit(node)
    return bridges


def edge_categories(
    adjacency: np.ndarray,
    *,
    block_groups: Sequence[int] | None = None,
) -> dict[str, set[tuple[int, int]]]:
    values = np.asarray(adjacency, dtype=np.int8)
    true_edges = true_edge_set(values)
    degrees = np.sum(values, axis=1).astype(np.int64)
    q25 = float(np.quantile(degrees, 0.25))
    q75 = float(np.quantile(degrees, 0.75))
    categories = {
        "low_degree_incident": {
            edge
            for edge in true_edges
            if min(int(degrees[edge[0]]), int(degrees[edge[1]])) <= q25
        },
        "high_degree_incident": {
            edge
            for edge in true_edges
            if max(int(degrees[edge[0]]), int(degrees[edge[1]])) >= q75
        },
        "zero_common_neighbor": {
            edge
            for edge in true_edges
            if int(np.dot(values[edge[0]], values[edge[1]])) == 0
        },
        "bridge": _bridge_edges(values),
    }
    if block_groups is not None:
        groups = tuple(int(group) for group in block_groups)
        if len(groups) != values.shape[0]:
            raise ValueError("block_groups must align with adjacency")
        categories["cross_block"] = {
            edge for edge in true_edges if groups[edge[0]] != groups[edge[1]]
        }
    return categories


def _category_metrics(
    categories: dict[str, set[tuple[int, int]]],
    covered_edges: set[tuple[int, int]],
    residual_edges: set[tuple[int, int]],
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for name, edges in categories.items():
        covered = len(edges & covered_edges)
        residual = len(edges & residual_edges)
        metrics[f"{name}_edge_count"] = len(edges)
        metrics[f"{name}_edge_recall"] = covered / len(edges) if edges else None
        metrics[f"{name}_residual_share"] = (
            residual / len(residual_edges) if residual_edges else 0.0
        )
    return metrics


def prefix_coverage_trajectory(
    adjacency: np.ndarray,
    cover: PatchCover,
    *,
    patch_size: int,
    overlap: int,
    maximum_patches: int,
    block_groups: Sequence[int] | None = None,
    n_atoms: int = 24,
    sparsity: int = 3,
    coefficient_bits: int = 8,
) -> list[dict[str, Any]]:
    """沿一条 Beam8 链的每个前缀测量覆盖、公平性、残差结构与率。

    每个前缀都计算边/对/节点覆盖、节点入射边召回、残差边数、边多重度，
    以及 KSVD 码率与字典标量的乐观代理（bits 只作相对成本比较）。
    """
    values = np.asarray(adjacency, dtype=np.int8)
    n_nodes = values.shape[0]
    pair_count = n_nodes * (n_nodes - 1) // 2
    true_edges = true_edge_set(values)
    degrees = np.sum(values, axis=1).astype(np.int64)
    nonisolated = np.flatnonzero(degrees > 0)
    categories = edge_categories(values, block_groups=block_groups)
    observed_pairs: set[tuple[int, int]] = set()
    covered_edges: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    edge_multiplicity: dict[tuple[int, int], int] = {}
    marginal_new_edges: list[int] = []
    atom_index_bits = int(math.ceil(math.log2(n_atoms)))
    code_bits_per_patch = sparsity * (atom_index_bits + coefficient_bits)
    framing_bits = int(math.ceil(math.log2(maximum_patches + 1)))
    direct_bits = direct_enumerative_bits(pair_count, len(true_edges))
    rows: list[dict[str, Any]] = []

    for patch_count in range(len(cover.patches) + 1):
        if patch_count:
            patch = cover.patches[patch_count - 1]
            patch_pairs = _pairs(patch.node_ids)
            patch_edges = true_edges & patch_pairs
            marginal_new_edges.append(len(patch_edges - covered_edges))
            observed_pairs.update(patch_pairs)
            covered_edges.update(patch_edges)
            covered_nodes.update(int(node) for node in patch.node_ids)
            for edge in patch_edges:
                edge_multiplicity[edge] = edge_multiplicity.get(edge, 0) + 1

        residual_edges = true_edges - covered_edges
        covered_incident = np.zeros(n_nodes, dtype=np.int64)
        residual_incident = np.zeros(n_nodes, dtype=np.int64)
        for left, right in covered_edges:
            covered_incident[left] += 1
            covered_incident[right] += 1
        for left, right in residual_edges:
            residual_incident[left] += 1
            residual_incident[right] += 1
        recalls = (
            covered_incident[nonisolated] / degrees[nonisolated]
            if nonisolated.size
            else np.ones(0, dtype=np.float64)
        )
        multiplicities = np.asarray(tuple(edge_multiplicity.values()), dtype=np.float64)
        identity = identity_proxy_bits(
            n_nodes=n_nodes,
            patch_size=patch_size,
            overlap=overlap,
            patch_count=patch_count,
        )
        unobserved_count = pair_count - len(observed_pairs)
        residual_bits = residual_subset_bits(unobserved_count, len(residual_edges))
        ksvd_bits = patch_count * code_bits_per_patch
        canonical_bits = (
            framing_bits + identity["canonical_set_identity_bits"] + ksvd_bits + residual_bits
        )
        row = {
            "patch_count": patch_count,
            "raw_pair_slots": patch_count * patch_size * (patch_size - 1) // 2,
            "unique_observed_pair_count": len(observed_pairs),
            "covered_edge_count": len(covered_edges),
            "residual_edge_count": len(residual_edges),
            "node_coverage": len(covered_nodes) / max(n_nodes, 1),
            "pair_coverage": len(observed_pairs) / max(pair_count, 1),
            "edge_coverage": len(covered_edges) / max(len(true_edges), 1),
            "raw_zero_fill_rmse": math.sqrt(len(residual_edges) / max(pair_count, 1)),
            "covered_edge_density": len(covered_edges) / max(len(observed_pairs), 1),
            "last_marginal_new_edges": marginal_new_edges[-1] if marginal_new_edges else 0,
            "mean_marginal_new_edges": float(np.mean(marginal_new_edges))
            if marginal_new_edges
            else 0.0,
            "node_incident_recall_mean": float(np.mean(recalls)) if recalls.size else 1.0,
            "node_incident_recall_p10": float(
                np.quantile(recalls, 0.10, method="linear")
            )
            if recalls.size
            else 1.0,
            "node_incident_recall_minimum": float(np.min(recalls)) if recalls.size else 1.0,
            "fully_covered_node_fraction": float(np.mean(recalls == 1.0))
            if recalls.size
            else 1.0,
            "nodes_with_residual_fraction": float(np.mean(residual_incident > 0)),
            "maximum_residual_incident_count": int(np.max(residual_incident)),
            "edge_multiplicity_mean": float(np.mean(multiplicities))
            if multiplicities.size
            else 0.0,
            "edge_multiplicity_cv": float(
                np.std(multiplicities) / max(np.mean(multiplicities), 1e-12)
            )
            if multiplicities.size
            else 0.0,
            **_category_metrics(categories, covered_edges, residual_edges),
            **identity,
            "prefix_length_bits": framing_bits,
            "ksvd_code_proxy_bits": ksvd_bits,
            "residual_subset_bits": residual_bits,
            "canonical_hybrid_proxy_bits": canonical_bits,
            "direct_bitset_bits": pair_count,
            "direct_enumerative_exact_bits": direct_bits,
            "dictionary_scalars": patch_size * (patch_size - 1) // 2 * n_atoms,
            "code_scalars": patch_count * sparsity,
        }
        rows.append(row)
    return rows


def select_operating_checkpoints(
    trajectory: Sequence[dict[str, Any]],
    *,
    base_patch_count: int,
) -> dict[str, dict[str, Any] | None]:
    by_count = {int(row["patch_count"]): dict(row) for row in trajectory}
    selected: dict[str, dict[str, Any] | None] = {
        "BASE": by_count.get(int(base_patch_count))
    }
    for threshold in (0.90, 0.95, 0.99, 1.0):
        label = f"EDGE{int(round(100 * threshold))}"
        selected[label] = next(
            (
                dict(row)
                for row in trajectory
                if float(row["edge_coverage"]) + 1e-12 >= threshold
            ),
            None,
        )
    selected["FAIR95"] = next(
        (
            dict(row)
            for row in trajectory
            if float(row["edge_coverage"]) + 1e-12 >= 0.95
            and float(row["node_incident_recall_p10"]) + 1e-12 >= 0.90
            and float(row["node_coverage"]) + 1e-12 >= 1.0
        ),
        None,
    )
    return selected


# ===========================================================================
# H. 分组折与源暴露审计（code/mentor_grouped_splits.py）
#    以密度分层平衡、root-candidate 为不可分割组做 3 折划分，保证同一根
#    不跨折；随机参考折用于对照。源节点/边暴露只作转导式审计，不作归纳证明。
# ===========================================================================

@dataclass(frozen=True)
class MentorFold:
    view: str
    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _as_int_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return result


def balanced_group_folds(
    graph_indices: Sequence[int] | np.ndarray,
    strata: Sequence[int] | np.ndarray,
    groups: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 20260807,
    view: str = "grouped",
) -> tuple[MentorFold, ...]:
    """把不可分割的组分配到密度均衡的确定性测试折。

    组按"最大 / 密度最偏 / seed 扰动 / 组号"排序后贪心放置，目标为最小化
    各折图数与各密度层计数的平方偏差；前 n_splits 个组分别打底一折，
    避免出现空测试折。组完整性审计失败即抛错。
    """
    indices = _as_int_vector(graph_indices, "graph_indices")
    strata_array = _as_int_vector(strata, "strata")
    groups_array = _as_int_vector(groups, "groups")
    if not (indices.size == strata_array.size == groups_array.size):
        raise ValueError("graph_indices, strata, and groups must have equal length")
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("graph_indices must be nonempty and unique")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if np.any((strata_array < 0) | (strata_array >= len(STRATUM_NAMES))):
        raise ValueError("strata contain an out-of-range value")

    unique_groups = np.unique(groups_array)
    if unique_groups.size < n_splits:
        raise ValueError("fewer groups than folds")
    rng = np.random.default_rng(int(seed))
    tie_order = rng.permutation(unique_groups.size)
    tie_rank = {
        int(unique_groups[position]): int(rank)
        for rank, position in enumerate(tie_order)
    }

    records: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for raw_group in unique_groups:
        group = int(raw_group)
        positions = np.flatnonzero(groups_array == group)
        histogram = np.bincount(
            strata_array[positions], minlength=len(STRATUM_NAMES)
        ).astype(np.int64)
        concentration = float(np.max(histogram) / positions.size)
        records.append((group, positions, histogram, concentration))
    records.sort(
        key=lambda item: (
            -int(item[1].size),
            -item[3],
            tie_rank[item[0]],
            item[0],
        )
    )

    target_graphs = indices.size / float(n_splits)
    total_strata = np.bincount(
        strata_array, minlength=len(STRATUM_NAMES)
    ).astype(np.float64)
    target_strata = total_strata / float(n_splits)
    graph_scale = max(target_graphs, 1.0)
    stratum_scale = np.maximum(target_strata, 1.0)
    fold_graphs = np.zeros(n_splits, dtype=np.float64)
    fold_strata = np.zeros((n_splits, len(STRATUM_NAMES)), dtype=np.float64)
    fold_positions: list[list[int]] = [[] for _ in range(n_splits)]

    def objective(candidate_fold: int, size: int, histogram: np.ndarray) -> float:
        counts = fold_graphs.copy()
        density = fold_strata.copy()
        counts[candidate_fold] += size
        density[candidate_fold] += histogram
        graph_error = np.sum(((counts - target_graphs) / graph_scale) ** 2)
        density_error = np.sum(
            ((density - target_strata[None, :]) / stratum_scale[None, :]) ** 2
        )
        return float(graph_error + density_error)

    for assignment_index, (_group, positions, histogram, _concentration) in enumerate(records):
        candidates = (
            (assignment_index,)
            if assignment_index < n_splits
            else tuple(range(n_splits))
        )
        selected_fold = min(
            candidates,
            key=lambda fold: (
                objective(int(fold), int(positions.size), histogram),
                fold_graphs[int(fold)],
                int(fold),
            ),
        )
        fold_positions[int(selected_fold)].extend(int(value) for value in positions)
        fold_graphs[int(selected_fold)] += positions.size
        fold_strata[int(selected_fold)] += histogram

    folds = []
    all_indices = set(int(value) for value in indices)
    for fold_index, positions in enumerate(fold_positions):
        test = np.sort(indices[np.asarray(positions, dtype=np.int64)])
        train = np.asarray(sorted(all_indices - set(int(value) for value in test)), dtype=np.int64)
        if train.size == 0 or test.size == 0:
            raise RuntimeError("balanced assignment produced an empty train/test fold")
        folds.append(
            MentorFold(
                view=str(view),
                fold_index=fold_index,
                train_indices=train,
                test_indices=test,
            )
        )
    partition_audit = audit_fold_partition(
        folds,
        indices,
        groups_array_by_index=dict(zip(indices, groups_array)),
        require_group_integrity=True,
    )
    if not partition_audit["passed"]:
        raise RuntimeError("balanced assignment failed its partition/group audit")
    return tuple(folds)


def make_mentor_folds(
    graph_indices: Sequence[int] | np.ndarray,
    strata: Sequence[int] | np.ndarray,
    roots: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 20260807,
) -> dict[str, tuple[MentorFold, ...]]:
    """构建随机参考与 root-candidate 分组两个划分视角。"""
    indices = _as_int_vector(graph_indices, "graph_indices")
    strata_array = _as_int_vector(strata, "strata")
    roots_array = _as_int_vector(roots, "roots")
    random_groups = np.arange(indices.size, dtype=np.int64)
    return {
        "random_reference": balanced_group_folds(
            indices,
            strata_array,
            random_groups,
            n_splits=n_splits,
            seed=seed,
            view="random_reference",
        ),
        "root_candidate_grouped": balanced_group_folds(
            indices,
            strata_array,
            roots_array,
            n_splits=n_splits,
            seed=seed,
            view="root_candidate_grouped",
        ),
    }


def audit_fold_partition(
    folds: Sequence[MentorFold],
    graph_indices: Sequence[int] | np.ndarray,
    *,
    groups_array_by_index: dict[int, int] | None = None,
    require_group_integrity: bool = False,
) -> dict[str, object]:
    indices = _as_int_vector(graph_indices, "graph_indices")
    expected = set(int(value) for value in indices)
    test_memberships: list[int] = []
    rows = []
    for fold in folds:
        train = set(int(value) for value in fold.train_indices)
        test = set(int(value) for value in fold.test_indices)
        rows.append(
            {
                "fold_index": int(fold.fold_index),
                "train_graph_count": len(train),
                "test_graph_count": len(test),
                "train_test_intersection_count": len(train & test),
                "partition_matches": train | test == expected,
            }
        )
        test_memberships.extend(test)
    membership_counts = {index: test_memberships.count(index) for index in expected}
    unexpected_test_indices = set(test_memberships) - expected
    group_leakage = 0
    if groups_array_by_index is not None:
        for fold in folds:
            train_groups = {
                int(groups_array_by_index[int(index)]) for index in fold.train_indices
            }
            test_groups = {
                int(groups_array_by_index[int(index)]) for index in fold.test_indices
            }
            group_leakage += len(train_groups & test_groups)
    checks = {
        "fold_count_at_least_two": len(folds) >= 2,
        "each_fold_is_partition": all(
            row["partition_matches"]
            and row["train_test_intersection_count"] == 0
            and row["train_graph_count"] > 0
            and row["test_graph_count"] > 0
            for row in rows
        ),
        "each_graph_tested_once": not unexpected_test_indices
        and all(value == 1 for value in membership_counts.values()),
        "group_integrity": (not require_group_integrity) or group_leakage == 0,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "group_leakage_count": int(group_leakage),
        "folds": rows,
    }


# ===========================================================================
# I. KSVD 稀疏字典学习（code/ksvd.py 核心子集）
#    交替执行：稀疏编码（OMP，每列非零数 <= T 且至少 T_min）与逐原子
#    整列 SVD 更新；原子始终单位归一化。K=24 原子在训练折上跨图共享训练。
# ===========================================================================

def _omp(D: np.ndarray, y: np.ndarray, T: int) -> np.ndarray:
    """正交匹配追踪（OMP）求解 y ≈ D x 的稀疏系数。

    D 形状 (n, k)，y 为单列 (n,)，返回系数 x (k,)。贪心逐次挑选与当前
    残差相关度最大的原子，用最小二乘重估已选原子的系数，最多选 T 个。
    """
    n, k = D.shape
    x = np.zeros(k, dtype=np.float64)
    residual = y.copy()
    support: list[int] = []
    for _ in range(min(T, k)):
        corr = D.T @ residual
        for j in support:
            corr[j] = 0.0
        j = int(np.argmax(np.abs(corr)))
        if abs(corr[j]) < 1e-12:
            break
        support.append(j)
        Ds = D[:, support]
        coef, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)
        residual = y - Ds @ coef
        x = np.zeros(k, dtype=np.float64)
        for c, idx in zip(coef, support):
            x[idx] = c
        if np.linalg.norm(residual) < 1e-8:
            break
    return x


def ksvd(
    Y: np.ndarray,
    n_atoms: int = 12,
    T: int = 3,
    n_iter: int = 10,
    seed: int = 0,
    T_min: int = 2,
    initial_dictionary: np.ndarray | None = None,
    coherence_step: float = 0.0,
    anchor_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    KSVD 风格稀疏字典学习（交替更新）。

    Y: (特征数, 样本数)。返回 D (特征数, 原子数)、X (原子数, 样本数)
    以及训练信息。每轮交替：(1) OMP 稀疏编码（每列非零数 <= T 且至少
    T_min 个）；(2) 逐原子用残差矩阵的秩 1 SVD 更新。原子始终单位范数。
    提供 initial_dictionary 时用确定性初始字典做 warm start，便于把
    FINAL 与 INIT 的差距归因到"学习"而非"初始化"。
    """
    rng = np.random.default_rng(seed)
    n, N = Y.shape
    if initial_dictionary is None:
        n_atoms = min(n_atoms, max(n, 2), max(N, 2))
    else:
        n_atoms = int(n_atoms)
        if n_atoms < 1:
            raise ValueError("n_atoms must be positive")
    T = max(T, T_min)
    T = min(T, n_atoms)
    if not np.isfinite(coherence_step) or coherence_step < 0.0:
        raise ValueError("coherence_step must be finite and non-negative")
    if not np.isfinite(anchor_strength) or not 0.0 <= anchor_strength < 1.0:
        raise ValueError("anchor_strength must be finite and in [0, 1)")

    if initial_dictionary is None:
        idx = rng.choice(N, size=n_atoms, replace=N < n_atoms)
        D = Y[:, idx].astype(np.float64).copy()
        D += 1e-3 * rng.standard_normal(D.shape)
        initialization = "random_training_columns"
    else:
        supplied = np.asarray(initial_dictionary, dtype=np.float64)
        if supplied.shape != (n, n_atoms):
            raise ValueError(
                f"initial_dictionary must have shape {(n, n_atoms)}, got {supplied.shape}"
            )
        if not np.all(np.isfinite(supplied)):
            raise ValueError("initial_dictionary contains non-finite values")
        D = supplied.copy()
        initialization = "provided"
    for j in range(n_atoms):
        nj = np.linalg.norm(D[:, j])
        if nj < 1e-12:
            D[:, j] = rng.standard_normal(n)
            nj = np.linalg.norm(D[:, j])
        D[:, j] /= nj
    anchor_dictionary = D.copy() if anchor_strength > 0.0 else None

    X = np.zeros((n_atoms, N), dtype=np.float64)
    errs: list[float] = []

    for _it in range(n_iter):
        # ---- 稀疏编码阶段：用当前字典对每一列做 OMP，稀疏度 <= T ----
        for i in range(N):
            X[:, i] = _omp(D, Y[:, i], T)
            # 若编码过稀疏（非零数 < T_min），强制按相关度取前 T_min 个原子
            if np.count_nonzero(np.abs(X[:, i]) > 1e-10) < T_min:
                corr = np.abs(D.T @ Y[:, i])
                top = np.argsort(-corr)[:T_min]
                Ds = D[:, top]
                coef, _, _, _ = np.linalg.lstsq(Ds, Y[:, i], rcond=None)
                X[:, i] = 0.0
                for c, j in zip(coef, top):
                    X[j, i] = c

        # ---- 字典更新阶段：逐原子对残差矩阵做秩 1 SVD 更新 ----
        for j in range(n_atoms):
            omega = np.where(np.abs(X[j, :]) > 1e-10)[0]
            if omega.size == 0:
                # 死原子：用随机训练列重新初始化，避免字典坍缩
                i = int(rng.integers(0, N))
                D[:, j] = Y[:, i] + 1e-3 * rng.standard_normal(n)
                nj = np.linalg.norm(D[:, j])
                if nj > 1e-12:
                    D[:, j] /= nj
                continue
            # E = 去掉第 j 个原子贡献后的残差（只在使用该原子的列上）
            E = Y[:, omega] - D @ X[:, omega] + np.outer(D[:, j], X[j, omega])
            try:
                U, S, Vt = np.linalg.svd(E, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            # 用最大奇异值对应的左右奇异向量更新原子及其系数
            D[:, j] = U[:, 0]
            X[j, omega] = S[0] * Vt[0, :]

        # 原子单位归一化，系数按范数吸收，保证 D 各列 |·|=1
        for j in range(n_atoms):
            nj = np.linalg.norm(D[:, j])
            if nj > 1e-12:
                D[:, j] /= nj
                X[j, :] *= nj

        if anchor_strength > 0.0:
            assert anchor_dictionary is not None
            for j in range(n_atoms):
                anchor = anchor_dictionary[:, j]
                if float(D[:, j] @ anchor) < 0.0:
                    anchor = -anchor
                D[:, j] = (1.0 - anchor_strength) * D[:, j] + anchor_strength * anchor
                nj = np.linalg.norm(D[:, j])
                if nj < 1e-12:
                    raise RuntimeError("anchor update produced a degenerate atom")
                D[:, j] /= nj

        if coherence_step > 0.0 and n_atoms > 1:
            gram_offdiag = D.T @ D
            np.fill_diagonal(gram_offdiag, 0.0)
            D = D - coherence_step * (D @ gram_offdiag)
            for j in range(n_atoms):
                nj = np.linalg.norm(D[:, j])
                if nj < 1e-12:
                    raise RuntimeError("incoherence update produced a degenerate atom")
                D[:, j] /= nj

        if anchor_strength > 0.0 or coherence_step > 0.0:
            for i in range(N):
                X[:, i] = _omp(D, Y[:, i], T)
                if np.count_nonzero(np.abs(X[:, i]) > 1e-10) < T_min:
                    corr = np.abs(D.T @ Y[:, i])
                    top = np.argsort(-corr)[:T_min]
                    Ds = D[:, top]
                    coef, _, _, _ = np.linalg.lstsq(Ds, Y[:, i], rcond=None)
                    X[:, i] = 0.0
                    for c, atom in zip(coef, top):
                        X[atom, i] = c

        R = Y - D @ X
        errs.append(float(np.linalg.norm(R, "fro") / max(np.linalg.norm(Y, "fro"), 1e-12)))

    if n_iter == 0:
        for i in range(N):
            X[:, i] = _omp(D, Y[:, i], T)
        R = Y - D @ X
        errs.append(float(np.linalg.norm(R, "fro") / max(np.linalg.norm(Y, "fro"), 1e-12)))

    atoms_used = int(np.sum(np.any(np.abs(X) > 1e-10, axis=1)))
    nnz_per = np.array([np.count_nonzero(np.abs(X[:, i]) > 1e-10) for i in range(N)])
    gram_absolute = np.abs(D.T @ D)
    offdiagonal_mask = ~np.eye(n_atoms, dtype=bool)
    offdiagonal = gram_absolute[offdiagonal_mask]
    info = {
        "recon_rel": errs[-1] if errs else 1.0,
        "recon_curve": errs,
        "atoms_used": atoms_used,
        "mean_nnz": float(nnz_per.mean()) if N else 0.0,
        "n_atoms": n_atoms,
        "T": T,
        "n_iter": n_iter,
        "initialization": initialization,
        "coherence_step": float(coherence_step),
        "anchor_strength": float(anchor_strength),
        "mean_absolute_anchor_cosine": (
            float(np.mean(np.abs(np.sum(D * anchor_dictionary, axis=0))))
            if anchor_dictionary is not None
            else None
        ),
        "minimum_absolute_anchor_cosine": (
            float(np.min(np.abs(np.sum(D * anchor_dictionary, axis=0))))
            if anchor_dictionary is not None
            else None
        ),
        "mean_absolute_offdiagonal_coherence": (
            float(offdiagonal.mean()) if offdiagonal.size else 0.0
        ),
        "maximum_absolute_offdiagonal_coherence": (
            float(offdiagonal.max()) if offdiagonal.size else 0.0
        ),
    }
    return D, X, info


# ===========================================================================
# J. 字典初始化与健康指标（code/from_scratch_unplanted_dictionary.py）
#    INIT = 确定性最远点（maximin）初始化，从训练列里选多样且可复现的
#    K 个原子；FINAL 从同一 INIT 出发做 25 轮 KSVD 更新，以便把"学习字典
#    的价值"与"初始化的质量"分开。dictionary_metrics 报告死原子/有效原子/
#    激活占比/相干性等健康指标。
# ===========================================================================

def deterministic_maximin_initialization(
    centered_train: np.ndarray,
    n_atoms: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """确定性最远点初始化：从训练列里选 K 个多样且可复现的原子。

    第一个原子取范数最大的训练列（按字典序破平）；之后每次选与已选集合
    最大绝对内积最小（即最"新颖"）的一列。全程无随机性，保证 INIT 与
    FINAL 的对比可复现。
    """
    values = np.asarray(centered_train, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < n_atoms:
        raise ValueError("centered_train must contain at least n_atoms columns")
    norms = np.linalg.norm(values, axis=0)
    valid = np.flatnonzero(norms > EPS)
    if valid.size < n_atoms:
        raise ValueError("not enough nonzero centered training columns")
    normalized = values[:, valid] / norms[valid][None, :]

    def lex_key(position: int) -> tuple[float, ...]:
        return tuple(float(value) for value in normalized[:, position])

    maximum_norm = float(np.max(norms[valid]))
    first_candidates = [
        position
        for position, source in enumerate(valid)
        if abs(float(norms[source]) - maximum_norm) <= 1e-12
    ]
    first = min(first_candidates, key=lambda position: (lex_key(position), int(valid[position])))
    selected_positions = [int(first)]
    selection_scores: list[float | None] = [None]

    while len(selected_positions) < n_atoms:
        selected = normalized[:, selected_positions]
        novelty = 1.0 - np.max(np.abs(selected.T @ normalized), axis=0)
        novelty[selected_positions] = -np.inf
        best_score = float(np.max(novelty))
        candidates = [
            position
            for position in range(normalized.shape[1])
            if np.isfinite(novelty[position])
            and abs(float(novelty[position]) - best_score) <= 1e-12
        ]
        best = min(candidates, key=lambda position: (lex_key(position), int(valid[position])))
        selected_positions.append(int(best))
        selection_scores.append(best_score)

    selected_indices = [int(valid[position]) for position in selected_positions]
    dictionary = normalized[:, selected_positions].copy()
    return dictionary, {
        "name": "deterministic_maximin",
        "selected_training_indices": selected_indices,
        "selected_raw_norms": [float(norms[index]) for index in selected_indices],
        "selection_novelty_scores": selection_scores,
        "selected_unique_column_count": int(
            np.unique(values[:, selected_indices].T, axis=0).shape[0]
        ),
    }


def dictionary_metrics(
    values: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    *,
    activation_threshold: float = ACTIVATION_THRESHOLD,
    dead_frequency: float = 0.005,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    dictionary = np.asarray(dictionary, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    if values.ndim != 2 or dictionary.ndim != 2 or codes.ndim != 2:
        raise ValueError("values, dictionary, and codes must be matrices")
    if dictionary.shape[0] != values.shape[0] or codes.shape != (
        dictionary.shape[1],
        values.shape[1],
    ):
        raise ValueError("dictionary/code shapes are incompatible with values")

    residual = values - dictionary @ codes
    residual_frobenius = float(np.linalg.norm(residual, "fro"))
    signal_frobenius = float(np.linalg.norm(values, "fro"))
    relative = residual_frobenius / max(signal_frobenius, EPS)
    nmse = float(np.sum(residual**2) / max(float(np.sum(values**2)), EPS))

    active = np.abs(codes) > activation_threshold
    activation_counts = np.sum(active, axis=1).astype(np.int64)
    frequencies = activation_counts / max(codes.shape[1], 1)
    total_activations = int(np.sum(activation_counts))
    if total_activations > 0:
        shares = activation_counts.astype(np.float64) / total_activations
        positive = shares > 0
        entropy = float(-np.sum(shares[positive] * np.log(shares[positive])))
        effective = float(np.exp(entropy))
        maximum_share = float(np.max(shares))
    else:
        entropy = 0.0
        effective = 0.0
        maximum_share = 0.0

    gram = np.abs(dictionary.T @ dictionary)
    np.fill_diagonal(gram, 0.0)
    maximum_coherence = float(np.max(gram)) if dictionary.shape[1] > 1 else 0.0
    nonzeros_per_patch = np.sum(active, axis=0)
    return {
        "patch_count": int(values.shape[1]),
        "relative_reconstruction_error": float(relative),
        "nmse": nmse,
        "mean_nonzeros_per_patch": float(np.mean(nonzeros_per_patch)),
        "minimum_nonzeros_per_patch": int(np.min(nonzeros_per_patch)),
        "maximum_nonzeros_per_patch": int(np.max(nonzeros_per_patch)),
        "activation_counts": activation_counts.tolist(),
        "activation_frequencies": frequencies.tolist(),
        "activation_entropy": entropy,
        "effective_atom_count": effective,
        "dead_atom_count": int(np.count_nonzero(frequencies < dead_frequency)),
        "nondead_atom_count": int(np.count_nonzero(frequencies >= dead_frequency)),
        "dead_frequency_threshold": float(dead_frequency),
        "maximum_absolute_offdiagonal_coherence": maximum_coherence,
        "maximum_activation_share": maximum_share,
    }


def encode_with_minimum_sparsity(
    values: np.ndarray,
    dictionary: np.ndarray,
    *,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
) -> np.ndarray:
    """用给定字典对留出列做 OMP 编码，并显式保证每列至少 T_min 个非零。

    以 n_iter=0 调用 ksvd 只做编码不更新字典；随后若有列的非零数不足
    T_min，则按相关度排序强制补足到 T_min 个原子。
    """
    values = np.asarray(values, dtype=np.float64)
    dictionary = np.asarray(dictionary, dtype=np.float64)
    if minimum_sparsity < 0 or minimum_sparsity > sparsity:
        raise ValueError("minimum_sparsity must lie in [0, sparsity]")
    _, codes, _ = ksvd(
        values,
        n_atoms=dictionary.shape[1],
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=0,
        seed=0,
        initial_dictionary=dictionary,
        coherence_step=0.0,
        anchor_strength=0.0,
    )
    if minimum_sparsity:
        nnz = np.sum(np.abs(codes) > ACTIVATION_THRESHOLD, axis=0)
        for column in np.flatnonzero(nnz < minimum_sparsity):
            correlations = np.abs(dictionary.T @ values[:, column])
            support = np.argsort(-correlations, kind="stable")[:minimum_sparsity]
            coefficients, _, _, _ = np.linalg.lstsq(
                dictionary[:, support], values[:, column], rcond=None
            )
            codes[:, column] = 0.0
            codes[support, column] = coefficients
    return codes


# ===========================================================================
# K. 拼图与评估（code/overlap_stitching.py）
#    把每块局部邻接预测映射回全局节点对，重叠节点对按出现权重取加权平均
#    融合成整图预测；报告块内相对误差、观测对/全图的 RMSE / F1 等。
# ===========================================================================

@dataclass(frozen=True)
class CoverExample:
    graph_index: int
    family: str
    target_degree: int
    adjacency: np.ndarray
    cover: PatchCover
    patch_vectors: np.ndarray


def make_cover_example(
    graph_index: int,
    family: str,
    target_degree: int,
    adjacency: np.ndarray,
    cover: PatchCover,
) -> CoverExample:
    validate_simple_adjacency(adjacency)
    vectors = cover_vectors(cover).astype(np.float64, copy=False)
    return CoverExample(
        graph_index=int(graph_index),
        family=str(family),
        target_degree=int(target_degree),
        adjacency=np.asarray(adjacency, dtype=np.int8),
        cover=cover,
        patch_vectors=vectors,
    )


def stack_cover_examples(examples: Sequence[CoverExample]) -> np.ndarray:
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples cannot be empty")
    dimension = examples[0].patch_vectors.shape[1]
    if any(
        example.patch_vectors.ndim != 2
        or example.patch_vectors.shape[1] != dimension
        for example in examples
    ):
        raise ValueError("all examples must have equal-dimensional patch matrices")
    return np.concatenate([example.patch_vectors for example in examples], axis=0).T


def fit_pca_basis(centered_train: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(centered_train, dtype=np.float64)
    if values.ndim != 2 or not 1 <= rank <= min(values.shape):
        raise ValueError("invalid centered train matrix or PCA rank")
    left, _singular, _right = np.linalg.svd(values, full_matrices=False)
    return left[:, :rank]


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("binary metrics require equal vectors")
    true_positive = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    false_positive = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    false_negative = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def stitch_patch_predictions(
    example: CoverExample,
    predicted_patch_vectors: np.ndarray,
    *,
    threshold: float = 0.5,
    local_pair_weights: np.ndarray | None = None,
    exact_residual_edges: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """把局部 patch 预测映射回全局节点对，做加权融合并评估整图拼接。

    - 同一全局节点对被多个 patch 覆盖时，按出现权重取加权平均作为最终
      预测（权重默认全 1，即等权平均）。
    - exact_residual_edges 非空时把"未被任何 patch 覆盖的真实边"单独给
      定为 1.0，用于把"采样覆盖缺失"从"patch 压缩误差"中分离出来。
    - 返回块内相对误差、观测对与整图（1225 对）的 RMSE/准确率/精度/
      召回/F1 等。
    """
    predicted = np.asarray(predicted_patch_vectors, dtype=np.float64)
    truth_patches = np.asarray(example.patch_vectors, dtype=np.float64)
    if predicted.shape != truth_patches.shape:
        raise ValueError("predicted patches must match the cover patch matrix")
    patch_size = len(example.cover.patches[0].node_ids)
    local_edges = upper_triangle_edges(patch_size)
    if predicted.shape[1] != len(local_edges):
        raise ValueError("patch vector dimension does not match patch_size")
    if local_pair_weights is None:
        pair_weights = np.ones(len(local_edges), dtype=np.float64)
    else:
        pair_weights = np.asarray(local_pair_weights, dtype=np.float64)
        if (
            pair_weights.shape != (len(local_edges),)
            or not np.all(np.isfinite(pair_weights))
            or np.any(pair_weights <= 0.0)
        ):
            raise ValueError("local_pair_weights must be finite positive slot weights")

    occurrence_predictions: dict[tuple[int, int], list[float]] = {}
    occurrence_weights: dict[tuple[int, int], list[float]] = {}
    # 把每块局部的上三角预测按"节点对"归集到全局：同一对可能出现多次
    for patch, vector in zip(example.cover.patches, predicted):
        for edge_index, (value, (left_slot, right_slot)) in enumerate(
            zip(vector, local_edges)
        ):
            pair = tuple(sorted((patch.node_ids[left_slot], patch.node_ids[right_slot])))
            occurrence_predictions.setdefault(pair, []).append(float(value))
            occurrence_weights.setdefault(pair, []).append(float(pair_weights[edge_index]))

    n_nodes = example.adjacency.shape[0]
    all_pairs = list(combinations(range(n_nodes), 2))
    truth = np.asarray(
        [example.adjacency[left, right] for left, right in all_pairs],
        dtype=np.float64,
    )
    pair_to_position = {pair: index for index, pair in enumerate(all_pairs)}
    full_prediction = np.zeros(len(all_pairs), dtype=np.float64)
    observed_positions = []
    repeated_stds = []
    repeated_ranges = []
    # 权重融合：同一节点对的多次预测按出现权重取加权平均
    for pair, values in occurrence_predictions.items():
        position = pair_to_position[pair]
        full_prediction[position] = float(
            np.average(values, weights=occurrence_weights[pair])
        )
        observed_positions.append(position)
        if len(values) > 1:
            repeated_stds.append(float(np.std(values, ddof=0)))
            repeated_ranges.append(float(np.max(values) - np.min(values)))
    # 残差校正：未被任何 patch 覆盖的真实边单独置 1，分离"采样缺失"与"压缩误差"
    if exact_residual_edges is not None:
        observed_pairs = set(occurrence_predictions)
        for raw_pair in exact_residual_edges:
            pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
            if pair in observed_pairs:
                raise ValueError("residual edges must be unobserved by all patches")
            if pair not in pair_to_position:
                raise ValueError("residual edge endpoint is outside the graph")
            if example.adjacency[pair[0], pair[1]] == 0:
                raise ValueError("exact residual sidecar may contain only true edges")
            full_prediction[pair_to_position[pair]] = 1.0

    # 观测对 = 至少被一个 patch 覆盖的节点对；整图 = 全部 1225 对
    observed_positions_array = np.asarray(sorted(observed_positions), dtype=np.int64)
    observed_truth = truth[observed_positions_array]
    observed_prediction = full_prediction[observed_positions_array]

    patch_residual = predicted - truth_patches
    patch_relative = float(
        np.linalg.norm(patch_residual, "fro") / max(np.linalg.norm(truth_patches, "fro"), EPS)
    )
    observed_binary = (observed_prediction >= threshold).astype(np.int8)
    full_binary = (full_prediction >= threshold).astype(np.int8)
    observed_metrics = _binary_metrics(observed_truth.astype(np.int8), observed_binary)
    full_metrics = _binary_metrics(truth.astype(np.int8), full_binary)
    return {
        "patch_relative_error": patch_relative,
        "observed_pair_count": int(observed_positions_array.size),
        "observed_pair_rmse": float(
            np.sqrt(np.mean((observed_prediction - observed_truth) ** 2))
        ),
        "observed_pair_accuracy": float(np.mean(observed_binary == observed_truth)),
        "observed_edge_precision": observed_metrics["precision"],
        "observed_edge_recall": observed_metrics["recall"],
        "observed_edge_f1": observed_metrics["f1"],
        "full_adjacency_rmse": float(
            np.sqrt(np.mean((full_prediction - truth) ** 2))
        ),
        "full_adjacency_accuracy": float(np.mean(full_binary == truth)),
        "full_edge_precision": full_metrics["precision"],
        "full_edge_recall": full_metrics["recall"],
        "full_edge_f1": full_metrics["f1"],
        "repeated_pair_count": int(len(repeated_stds)),
        "repeated_pair_disagreement_std_mean": float(np.mean(repeated_stds))
        if repeated_stds
        else 0.0,
        "repeated_pair_disagreement_range_mean": float(np.mean(repeated_ranges))
        if repeated_ranges
        else 0.0,
        "prediction_minimum": float(np.min(predicted)),
        "prediction_maximum": float(np.max(predicted)),
    }


def graph_balanced_stitch_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    rows = tuple(rows)
    if not rows:
        raise ValueError("stitch rows cannot be empty")
    metrics = (
        "patch_relative_error",
        "observed_pair_rmse",
        "observed_pair_accuracy",
        "observed_edge_precision",
        "observed_edge_recall",
        "observed_edge_f1",
        "full_adjacency_rmse",
        "full_adjacency_accuracy",
        "full_edge_precision",
        "full_edge_recall",
        "full_edge_f1",
        "repeated_pair_count",
        "repeated_pair_disagreement_std_mean",
        "repeated_pair_disagreement_range_mean",
    )
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in metrics
    }


# ===========================================================================
# L. 冻结几何与协议参数
#    GEOMETRIES: s8_o2 / s10_o3 / s12_o4（patch 尺寸 / 目标重叠）
#    N_ATOMS=24, SPARSITY=3, T_MIN=1：共享 KSVD 字典 24 原子，每块最多
#    用 3 个原子的稀疏线性组合，最小 1 个。这是"小字典效果好"的核心配置。
# ===========================================================================

GEOMETRIES = {"s8_o2": (8, 2), "s10_o3": (10, 3), "s12_o4": (12, 4)}

# 冻结的 2026-08-06 分组 KSVD 协议参数。
N_ATOMS = 24
SPARSITY = 3
T_MIN = 1
KSVD_SEED = 0
PCA_RANK = 3
RANDOM_DICTIONARY_SEED_BASE = 970301
DECOMPOSITION_TOLERANCE = 1e-10

CHECKPOINTS_PROTOCOL = ("BASE", "FAIR95")
VIEWS = ("random_reference", "root_candidate_grouped")
STAGES = ("RAW", "PCA3", "RANDOM", "INIT", "FINAL")
CORRECTIONS = ("uncorrected", "residual_corrected")
STITCH_METRICS = (
    "patch_relative_error",
    "observed_pair_rmse",
    "observed_pair_accuracy",
    "observed_edge_precision",
    "observed_edge_recall",
    "observed_edge_f1",
    "full_adjacency_rmse",
    "full_adjacency_accuracy",
    "full_edge_precision",
    "full_edge_recall",
    "full_edge_f1",
    "repeated_pair_count",
    "repeated_pair_disagreement_std_mean",
    "repeated_pair_disagreement_range_mean",
)


# ===========================================================================
# M. 单图准备 + 每折重构驱动
#    （code/run_mentor_subgraphs_grouped_ksvd_followup.py 核心子集）
#    每个分支（geometry×checkpoint）把采样链截到目标 patch 数、做
#    rooted-canonical 排序、生成 CoverExample；每折只读训练折拟合字典，
#    比较 RAW / PCA3 / RANDOM / INIT / FINAL 五种阶段。
# ===========================================================================

@dataclass(frozen=True)
class PreparedGraph:
    """某张 pilot 图在某分支（geometry×checkpoint）重新生成的 rooted-canonical 前缀。"""

    source_index: int
    pilot_position: int
    density_stratum: str
    average_degree: float
    geometry: str
    checkpoint: str
    patch_count: int
    ids: StableNodeIDs
    example: CoverExample


def _prefix_cover(cover: PatchCover, patch_count: int, method: str) -> PatchCover:
    if not 1 <= patch_count <= len(cover.patches):
        raise ValueError("prefix patch count is outside cover")
    return _make_cover(
        method,
        cover.patches[:patch_count],
        cover.segment_ids[:patch_count],
        cover.target_edges[:patch_count],
        cover.bridge_lengths[:patch_count],
    )


def _residual_edges(example: CoverExample) -> set[tuple[int, int]]:
    observed = {
        tuple(sorted(pair))
        for patch in example.cover.patches
        for pair in combinations(patch.node_ids, 2)
    }
    return {
        (left, right)
        for left in range(example.adjacency.shape[0])
        for right in range(left + 1, example.adjacency.shape[0])
        if example.adjacency[left, right] and (left, right) not in observed
    }


def _evaluate(
    examples: Sequence[CoverExample],
    reconstructed: np.ndarray | None,
    *,
    residual_corrected: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """拼接 patch 预测，返回逐图行 + 图均衡汇总。

    reconstructed 为 None 时用原始 patch 向量（RAW 恒等上界）；否则把
    重构矩阵按各图 patch 数切片，逐图调用 stitch_patch_predictions。
    """
    rows = []
    cursor = 0
    for example in examples:
        patch_count = example.patch_vectors.shape[0]
        predictions = (
            example.patch_vectors
            if reconstructed is None
            else reconstructed[:, cursor : cursor + patch_count].T
        )
        metrics = stitch_patch_predictions(
            example,
            predictions,
            exact_residual_edges=(_residual_edges(example) if residual_corrected else None),
        )
        rows.append({"source_index": example.graph_index, **metrics})
        cursor += patch_count
    if reconstructed is not None and cursor != reconstructed.shape[1]:
        raise ValueError("reconstruction columns do not match examples")
    return rows, graph_balanced_stitch_summary(rows)


def _compact_health(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "patch_count",
        "relative_reconstruction_error",
        "mean_nonzeros_per_patch",
        "minimum_nonzeros_per_patch",
        "maximum_nonzeros_per_patch",
        "activation_entropy",
        "effective_atom_count",
        "dead_atom_count",
        "nondead_atom_count",
        "dead_frequency_threshold",
        "maximum_absolute_offdiagonal_coherence",
        "maximum_activation_share",
    )
    return {key: metrics[key] for key in keys}


def _random_dictionary_seed(geometry: str, checkpoint: str, fold_index: int) -> int:
    """随机字典的确定性分支/折偏移：970301 + 分支序*100 + 折号。

    用冻结的 GEOMETRIES 全序保证子集运行与全量运行使用相同 seed。
    """
    geometry_index = tuple(GEOMETRIES).index(geometry)
    checkpoint_index = CHECKPOINTS_PROTOCOL.index(checkpoint)
    branch_index = geometry_index * len(CHECKPOINTS_PROTOCOL) + checkpoint_index
    return RANDOM_DICTIONARY_SEED_BASE + branch_index * 100 + int(fold_index)


def _run_fold(
    train_examples: Sequence[CoverExample],
    test_examples: Sequence[CoverExample],
    *,
    view: str,
    fold_index: int,
    n_atoms: int = N_ATOMS,
    sparsity: int = SPARSITY,
    iterations: int = 25,
    random_seed: int,
) -> dict[str, Any]:
    """只读训练折做一折重构，返回各阶段汇总、字典健康与训练信息。

    五种阶段对比：
    - RAW    : 原始 patch 向量直接拼回（覆盖/拼接上界）
    - PCA3   : 训练集 3 维主成分的稠密重构基线
    - RANDOM : 随机选取训练列作为字典，不更新
    - INIT   : deterministic maximin 初始字典，不更新
    - FINAL  : 同一 INIT 出发做 25 轮 KSVD 更新
    全部编码都强制 T_min=1，测试集只做编码不参与字典拟合。
    """
    raw_train = stack_cover_examples(train_examples)
    raw_test = stack_cover_examples(test_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    centered_test = raw_test - train_mean

    reconstructed: dict[str, np.ndarray | None] = {"RAW": None}

    pca_basis = fit_pca_basis(centered_train, PCA_RANK)
    reconstructed["PCA3"] = pca_basis @ (pca_basis.T @ centered_test) + train_mean

    rng = np.random.default_rng(int(random_seed))
    column_norms = np.linalg.norm(centered_train, axis=0)
    nonzero_columns = np.flatnonzero(column_norms > EPS)
    if nonzero_columns.size < n_atoms:
        raise ValueError(
            f"not enough nonzero centered train columns ({nonzero_columns.size} < {n_atoms})"
        )
    chosen_columns = rng.choice(nonzero_columns, size=n_atoms, replace=False)
    random_dictionary = centered_train[:, chosen_columns].copy()
    random_dictionary = random_dictionary / np.linalg.norm(random_dictionary, axis=0)[None, :]
    random_train_codes = encode_with_minimum_sparsity(
        centered_train, random_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    random_codes = encode_with_minimum_sparsity(
        centered_test, random_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    reconstructed["RANDOM"] = random_dictionary @ random_codes + train_mean

    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    init_train_codes = encode_with_minimum_sparsity(
        centered_train, initial_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    init_codes = encode_with_minimum_sparsity(
        centered_test, initial_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    reconstructed["INIT"] = initial_dictionary @ init_codes + train_mean

    final_dictionary, _training_codes, training_info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=T_MIN,
        n_iter=iterations,
        seed=KSVD_SEED,
        initial_dictionary=initial_dictionary,
    )
    final_train_codes = encode_with_minimum_sparsity(
        centered_train, final_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    final_codes = encode_with_minimum_sparsity(
        centered_test, final_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    reconstructed["FINAL"] = final_dictionary @ final_codes + train_mean

    # 对五种阶段分别做 uncorrected 与 residual_corrected 两种评估；
    # 只保留 RAW / FINAL 的逐图行，其余只存汇总以控制 JSON 体积。
    stages: dict[str, Any] = {}
    keep_rows = {"RAW": True, "FINAL": True}
    for stage in STAGES:
        stages[stage] = {}
        for correction in CORRECTIONS:
            rows, summary = _evaluate(
                test_examples,
                reconstructed[stage],
                residual_corrected=(correction == "residual_corrected"),
            )
            stages[stage][correction] = {"summary": summary}
            if keep_rows.get(stage):
                stages[stage][correction]["rows"] = rows

    stages["RANDOM"]["dictionary_health"] = {
        "train": _compact_health(
            dictionary_metrics(centered_train, random_dictionary, random_train_codes)
        ),
    }
    stages["RANDOM"]["random_seed"] = int(random_seed)
    stages["INIT"]["dictionary_health"] = {
        "train": _compact_health(
            dictionary_metrics(centered_train, initial_dictionary, init_train_codes)
        ),
    }
    stages["INIT"]["initialization"] = {
        "name": initialization["name"],
        "selection_novelty_scores": initialization["selection_novelty_scores"],
    }
    stages["FINAL"]["dictionary_health"] = {
        "train": _compact_health(
            dictionary_metrics(centered_train, final_dictionary, final_train_codes)
        ),
        "test": _compact_health(
            dictionary_metrics(centered_test, final_dictionary, final_codes)
        ),
    }
    stages["FINAL"]["training"] = {
        "training_iterations": len(training_info.get("recon_curve", [])),
        "final_train_recon_rel": training_info.get("recon_rel"),
        "mean_train_nnz": training_info.get("mean_nnz"),
    }

    return {
        "view": view,
        "fold_index": int(fold_index),
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_patch_count": int(raw_train.shape[1]),
        "test_patch_count": int(raw_test.shape[1]),
        "stages": stages,
    }


# ===========================================================================
# N. 命令行入口
#    加载数据 -> 抽样 pilot -> 逐图稳定排序+Beam8 采样+canonical 排序
#    -> 分组折 -> 每折训练共享 KSVD -> 拼图汇总。
# ===========================================================================

def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _weighted_stage_mean(
    folds: Sequence[dict[str, Any]], stage: str, correction: str, key: str
) -> float | None:
    folds = [fold for fold in folds if fold.get("skipped") is not True]
    if not folds:
        return None
    values = np.asarray(
        [float(fold["stages"][stage][correction]["summary"][key]) for fold in folds],
        dtype=np.float64,
    )
    weights = np.asarray([float(fold["test_graph_count"]) for fold in folds], dtype=np.float64)
    return float(np.average(values, weights=weights))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Beam8 continuous-overlap + shared KSVD reconstruction on the "
        "mentor 50-node real subgraph bank (single-file pipeline)."
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--pilot-seed", type=int, default=DEFAULT_PILOT_SEED)
    parser.add_argument("--cover-seed", type=int, default=970201)
    parser.add_argument("--maximum-patches", type=int, default=60)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--geometry", type=str, choices=tuple(GEOMETRIES), default="s8_o2")
    parser.add_argument("--checkpoint", type=str, choices=tuple(CHECKPOINTS_PROTOCOL), default="BASE")
    parser.add_argument("--view", type=str, choices=tuple(VIEWS), default="root_candidate_grouped")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260807)
    args = parser.parse_args()

    source_path = Path(args.source) if args.source else default_source_path()
    cache_path = Path(args.cache) if args.cache else default_cache_path(source_path)
    bundle = load_bundle(source_pkl=source_path, cache_path=cache_path, force=args.force_cache)
    selection = select_pilot_indices(bundle, n_pilot=args.pilot_size, seed=args.pilot_seed)
    pilot_indices = selection.indices
    all_strata = density_stratum(bundle.avg_degrees)
    strata_values = all_strata[pilot_indices]
    root_values = bundle.roots[pilot_indices]
    geometry = args.geometry
    checkpoint = args.checkpoint
    patch_size, overlap = GEOMETRIES[geometry]
    geometry_index = tuple(GEOMETRIES).index(geometry)

    print(f"source_sha256={bundle.metadata.get('source_sha256')}")
    print(f"pilot={selection.size} roots={selection.n_distinct_roots} strata={selection.summary()['selected_counts']}")
    print(f"geometry={geometry} (s={patch_size}, o={overlap}) checkpoint={checkpoint} view={args.view}")

    prepared_by_index: dict[int, PreparedGraph] = {}
    coverage_missing: list[dict[str, Any]] = []
    # 逐图执行：稳定排序 -> Beam8 采样 -> 覆盖档位 -> rooted-canonical 前缀
    for position, raw_source_index in enumerate(pilot_indices):
        source_index = int(raw_source_index)
        source_adjacency = bundle.adjacency[source_index].astype(np.int8, copy=False)
        ids = compute_global_wl_ids(source_adjacency)
        stable_adjacency = reorder_by_stable_ids(source_adjacency, ids)
        stratum = STRATUM_NAMES[int(all_strata[source_index])]
        base_budget = patch_budget(
            stable_adjacency,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=args.multiplier,
        )
        # 采样 seed 由 (cover_seed, 图索引, 几何序) 派生，保证确定可复现
        sampler_seed = int(
            np.random.SeedSequence([args.cover_seed, source_index, geometry_index])
            .generate_state(1, dtype=np.uint32)[0]
        )
        try:
            stable_cover = sample_marginal_candidate_cover(
                stable_adjacency,
                np.random.default_rng(sampler_seed),
                n_patches=args.maximum_patches,
                patch_size=patch_size,
                target_overlap=overlap,
                retained_beam=args.retained_beam,
                candidate_restarts=args.candidate_restarts,
                allow_partial=True,
            )
        except Exception as exc:  # noqa: BLE001 - 按图记录失败，不中断整批
            coverage_missing.append(
                {
                    "source_index": source_index,
                    "geometry": geometry,
                    "checkpoint": checkpoint,
                    "reason": f"sampling failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        # 沿同一条采样链计算各前缀覆盖指标，选中目标档位的 patch 数
        trajectory = prefix_coverage_trajectory(
            stable_adjacency,
            stable_cover,
            patch_size=patch_size,
            overlap=overlap,
            maximum_patches=args.maximum_patches,
        )
        selected = select_operating_checkpoints(trajectory, base_patch_count=base_budget)
        row = selected[checkpoint]
        if row is None:
            # 目标档位在链上不可达（采样不足/覆盖不够），保留失败记录
            coverage_missing.append(
                {
                    "source_index": source_index,
                    "geometry": geometry,
                    "checkpoint": checkpoint,
                    "reason": "checkpoint unreachable",
                }
            )
            continue
        # 截取到目标 patch 数，做 rooted-canonical 排序，生成可编码的 patch 矩阵
        prefix = _prefix_cover(stable_cover, int(row["patch_count"]), f"{geometry}_{checkpoint}")
        ordered, _diagnostics = reorder_cover_structurally(
            stable_adjacency, prefix, "rooted_canonical"
        )
        example = make_cover_example(
            source_index,
            stratum,
            int(round(float(bundle.avg_degrees[source_index]))),
            stable_adjacency,
            ordered,
        )
        prepared_by_index[source_index] = PreparedGraph(
            source_index=source_index,
            pilot_position=position,
            density_stratum=stratum,
            average_degree=float(bundle.avg_degrees[source_index]),
            geometry=geometry,
            checkpoint=checkpoint,
            patch_count=int(row["patch_count"]),
            ids=ids,
            example=example,
        )
        if (position + 1) % 25 == 0 or position + 1 == len(pilot_indices):
            print(f"prepared={position + 1}/{len(pilot_indices)} missing={len(coverage_missing)}", flush=True)

    folds = make_mentor_folds(
        pilot_indices,
        strata_values,
        root_values,
        n_splits=args.n_splits,
        seed=args.split_seed,
    )[args.view]

    # 逐折：只把本折训练图喂给 KSVD 拟合，测试图仅编码/评估
    branch_folds: list[dict[str, Any]] = []
    for fold in folds:
        train_examples = [
            prepared_by_index[int(index)].example
            for index in fold.train_indices
            if int(index) in prepared_by_index
        ]
        test_examples = [
            prepared_by_index[int(index)].example
            for index in fold.test_indices
            if int(index) in prepared_by_index
        ]
        if not train_examples or not test_examples:
            # 该折无可用图（采样失败被过滤），跳过并在汇总中标记
            branch_folds.append(
                {
                    "view": args.view,
                    "fold_index": int(fold.fold_index),
                    "skipped": True,
                    "train_graph_count": len(train_examples),
                    "test_graph_count": len(test_examples),
                }
            )
            continue
        random_seed = _random_dictionary_seed(geometry, checkpoint, fold.fold_index)
        branch_folds.append(
            _run_fold(
                train_examples,
                test_examples,
                view=args.view,
                fold_index=fold.fold_index,
                iterations=args.iterations,
                random_seed=random_seed,
            )
        )
        print(
            f"fold={fold.fold_index} train={len(train_examples)} test={len(test_examples)} "
            f"patches(train/test)={sum(e.patch_vectors.shape[0] for e in train_examples)}/"
            f"{sum(e.patch_vectors.shape[0] for e in test_examples)}",
            flush=True,
        )

    print()
    print("## Reconstruction (test-count weighted, per fold then aggregated)")
    header = f"{'stage':8s} | " + " | ".join(f"fold{f['fold_index']}" for f in branch_folds if not f.get("skipped")) + " | weighted"
    print(header)
    for stage in STAGES:
        per_fold = [
            _fmt(branch_folds[i]["stages"][stage]["uncorrected"]["summary"]["patch_relative_error"])
            for i, fold in enumerate(branch_folds)
            if not fold.get("skipped")
        ]
        per_fold_corr = [
            _fmt(branch_folds[i]["stages"][stage]["residual_corrected"]["summary"]["patch_relative_error"])
            for i, fold in enumerate(branch_folds)
            if not fold.get("skipped")
        ]
        weighted = _weighted_stage_mean(branch_folds, stage, "uncorrected", "patch_relative_error")
        weighted_corr = _weighted_stage_mean(branch_folds, stage, "residual_corrected", "patch_relative_error")
        print(
            f"{stage:8s} | uncorr: {' | '.join(per_fold)} | {_fmt(weighted)}"
        )
        print(
            f"{'':8s} | corr : {' | '.join(per_fold_corr)} | {_fmt(weighted_corr)}"
        )

    print()
    print("## Full-graph stitch (weighted)")
    for metric in ("full_adjacency_rmse", "full_edge_recall", "observed_edge_f1"):
        values = [
            _weighted_stage_mean(branch_folds, stage, "uncorrected", metric)
            for stage in STAGES
        ]
        print(f"{metric:24s} " + " ".join(f"{s}={_fmt(v)}" for s, v in zip(STAGES, values)))

    print()
    print("## Dictionary health (FINAL, train / test)")
    for fold in branch_folds:
        if fold.get("skipped"):
            continue
        health = fold["stages"]["FINAL"]["dictionary_health"]
        for split, metrics in health.items():
            print(
                f"fold{fold['fold_index']} {split}: "
                f"nondead={metrics['nondead_atom_count']}/{N_ATOMS} "
                f"effective={_fmt(metrics['effective_atom_count'], 2)} "
                f"max_share={_fmt(metrics['maximum_activation_share'])} "
                f"coherence={_fmt(metrics['maximum_absolute_offdiagonal_coherence'])} "
                f"rel_err={_fmt(metrics['relative_reconstruction_error'])}"
            )

    init_err = _weighted_stage_mean(branch_folds, "INIT", "uncorrected", "patch_relative_error")
    final_err = _weighted_stage_mean(branch_folds, "FINAL", "uncorrected", "patch_relative_error")
    if init_err and final_err:
        print()
        print(f"FINAL vs INIT patch-error reduction = {(init_err - final_err) / init_err:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
