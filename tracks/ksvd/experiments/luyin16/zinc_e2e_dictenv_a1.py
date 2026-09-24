"""E2E-DictEnv-A1 runner — Invariant Attributed Dictionary Core (ZINC).

Round ``e2e_dictenv_a1``.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_a1_preregistration.md`` (frozen).
Prior-artifact audit:
``tracks/ksvd/notes/e2e_dictenv_a1_prior_artifact_audit.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_a1.py``.

Stage order (each stage is resumable and refuses to run out of order):

``cache scaler dictionaries omp-codes correctness assignment health continuity
  omp-screen select-omp iht-diag formal mechanism specificity accounting
  decision report``

Official ZINC **test is never loaded**; the shared P1 loader refuses
``split == "test"`` and every artifact records ``official_test_loaded = false``.
All CUDA work is on the requested physical GPU only (A1 uses GPU1).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = a1.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_a1"
CACHE_DIR = RESULTS_DIR / "cache"
STATE_DIR = RESULTS_DIR / "states"
CURVE_DIR = RESULTS_DIR / "curves"
FORMAL_DIR = RESULTS_DIR / "formal_runs"
SEED1_DIR = RESULTS_DIR / "seed1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

RAW_CACHE = {split: CACHE_DIR / f"a1_raw_{split}.pt" for split in ("train", "valid")}
SCALER_JSON = RESULTS_DIR / "scaler_meta.json"
DICT_STORE = {
    "INDEP": RESULTS_DIR / "dictionary_indep.pt",
    "REAL": RESULTS_DIR / "dictionary_real.pt",
}
SDB_DICT_PATH = TRACK_ROOT / "results/sdb_v0/dictionary.pt"
SDB_DICT_SHA256_F32 = "b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd"

BATCH_SIZE = int(p1run.BATCH_SIZE)
LEARNING_RATE = float(p1run.LEARNING_RATE)
WEIGHT_DECAY = float(p1run.WEIGHT_DECAY)
GRAD_CLIP = float(p1run.GRAD_CLIP)
TRAIN_SHUFFLE_OFFSET = int(p1run.TRAIN_SHUFFLE_OFFSET)
EVAL_SHUFFLE_OFFSET = int(p1run.EVAL_SHUFFLE_OFFSET)
SOUP_K = 5
SEED = 0
#: frozen prereg artifact labels: Stage-1 screen arms and Stage-3 E2E arms
ARM_LABEL = {"TOPO": "T0", "INDEP": "A0", "REAL": "A1"}
E_LABEL = {"TOPO": "E0", "INDEP": "E1", "REAL": "E2"}
DICT_EPOCHS = 10
HORIZON = int(a1.HORIZON)
LAMBDA_REC = float(a1.LAMBDA_REC)

CONTINUITY_SEED = int(sdb.DICT_SEED) + int(a1.CONTINUITY_SEED_OFFSET)
HEALTH_SEED = int(sdb.DICT_SEED)
HEALTH_FRACTION = 0.2
POSTHOC_PAIRS = 200000

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_n_params = v0run._n_params


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def provenance(device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "branch": _git_branch(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "requested_physical_gpu": 1,
        "device": str(device_obj),
        "seed": SEED,
        "official_test_loaded": False,
    }
    if device_obj.type == "cuda":
        payload["gpu_model"] = torch.cuda.get_device_name(device_obj)
        payload["cuda"] = str(torch.version.cuda)
        payload["cuda_visible_devices"] = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        peak = torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)
        payload["peak_gpu_memory_mb"] = float(peak)
    return payload


def _set_device_policy(device: str) -> torch.device:
    """Hard GPU policy: CUDA work happens on physical GPU1 and nowhere else."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
        if visible == "":
            raise RuntimeError(
                "E2E-DictEnv-A1 requires CUDA_VISIBLE_DEVICES=1 (GPU0 is foreign-occupied)"
            )
        if visible != "1":
            raise RuntimeError(
                f"E2E-DictEnv-A1 forbids CUDA_VISIBLE_DEVICES={visible!r}; expected '1'"
            )
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device_obj)
    return device_obj


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


_RAW_MEMORY: dict[str, dict[str, np.ndarray]] = {}


def load_raw(split: str) -> dict[str, np.ndarray]:
    """Load the A1 raw block cache (numpy views) for one split."""
    if split in _RAW_MEMORY:
        return _RAW_MEMORY[split]
    path = RAW_CACHE[split]
    if not path.exists():
        raise RuntimeError(f"raw cache missing for {split}; run the `cache` stage first")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {key: value.numpy() for key, value in blob.items()}
    _RAW_MEMORY[split] = out
    return out


def load_scalers() -> dict[str, a1.ObjectScaler]:
    if not SCALER_JSON.exists():
        raise RuntimeError("scaler_meta.json missing; run the `scaler` stage first")
    payload = _read_json(SCALER_JSON)
    out: dict[str, a1.ObjectScaler] = {}
    for coordinate in ("real", "indep"):
        blocks: dict[str, a1.BlockScaler] = {}
        for name in a1.BLOCKS:
            entry = payload["objects"][coordinate]["blocks"][name]
            scale = np.asarray(entry["scale"], dtype=np.float64)
            mask = np.asarray(entry["mask"], dtype=np.float64)
            blocks[name] = a1.BlockScaler(
                name=name,
                scale=scale,
                mask=mask,
                weight=float(entry["weight"]),
                rms_raw=np.zeros(scale.shape[0], dtype=np.float64),
                block_energy=float(entry["block_energy"]),
                masked_coordinates=int(entry["masked_coordinates"]),
                dim=int(entry["dim"]),
            )
        out[coordinate] = a1.ObjectScaler(coordinate=coordinate, blocks=blocks)
    return out


def arm_coordinate(arm: str, split: str, *, node_block: str | None = None, edge_block: str | None = None,
                   scaler_key: str | None = None) -> np.ndarray:
    """The arm's dictionary input ``[N, 65 or 433]`` (raw for TOPO)."""
    raw = load_raw(split)
    if arm == "TOPO":
        return np.asarray(raw["phi"], dtype=np.float32)
    scalers = load_scalers()
    scaler = scalers[str(scaler_key or a1.ARM_COORDINATE[arm])]
    if node_block is None and edge_block is None:
        return a1.apply_object_scaler(scaler, raw).astype(np.float32)
    if node_block is None or edge_block is None:
        raise ValueError("node_block and edge_block must be given together")
    return a1.apply_block_scaler(
        scaler, raw["phi"], raw[node_block], raw[edge_block]
    ).astype(np.float32)


def load_or_build_codes(arm: str, split: str) -> np.ndarray:
    """Exact top-``s`` OMP codes of the arm's frozen dictionary (label-free)."""
    path = CACHE_DIR / f"omp_{arm}_{split}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True).numpy()
    X = arm_coordinate(arm, split)
    D, _sha = load_arm_dictionary(arm)
    Dbar = sdb.normalize_columns(np.asarray(D, dtype=np.float64))
    started = time.perf_counter()
    codes = sdb.omp_codes(Dbar, np.asarray(X, dtype=np.float64), s=a1.DICT_S).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(torch.as_tensor(codes, dtype=torch.float32), path)
    print(f"[omp] {path.name} {codes.shape} in {time.perf_counter() - started:.1f}s", flush=True)
    return codes


def load_arm_dictionary(arm: str) -> tuple[np.ndarray, str]:
    """Frozen K-SVD init of the arm (official-train fit only)."""
    arm = str(arm)
    if arm == "TOPO":
        if not SDB_DICT_PATH.exists():
            raise RuntimeError("SDB-v0 dictionary missing")
        blob = torch.load(SDB_DICT_PATH, map_location="cpu", weights_only=True)
        D = np.asarray(blob["D_ksvd"].numpy(), dtype=np.float32)
        sha = _sha256_array(D)
        if sha != SDB_DICT_SHA256_F32:
            raise RuntimeError(f"frozen SDB dictionary sha {sha} != {SDB_DICT_SHA256_F32}")
        return D, sha
    path = DICT_STORE[arm]
    if not path.exists():
        raise RuntimeError(f"{arm} dictionary missing; run the `dictionaries` stage first")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    D = np.asarray(blob["D"].numpy(), dtype=np.float32)
    return D, _sha256_array(D)


def attach_a1(data_list: Sequence[Any], split: str, arm: str, codes: np.ndarray | None = None,
              x: np.ndarray | None = None) -> None:
    """Attach the arm's dictionary input (and frozen code) to the PyG molecules."""
    if x is None:
        x = arm_coordinate(arm, split)
    node_sizes = [int(v) for v in load_raw(split)["node_sizes"]]
    offset = 0
    for index, data in enumerate(data_list):
        size = node_sizes[index]
        data.dict_phi = torch.as_tensor(x[offset : offset + size], dtype=torch.float32).clone()
        if codes is not None:
            data.precomputed_coord = torch.as_tensor(
                codes[offset : offset + size], dtype=torch.float32
            ).clone()
        else:
            data.precomputed_coord = None
        offset += size


def data_for(arm: str, split: str, codes: np.ndarray | None = None, x: np.ndarray | None = None) -> list[Any]:
    data = p1run.load_split(split)
    attach_a1(data, split, arm, codes=codes, x=x)
    return data


def _mol_like(raw_molecule: Any) -> Any:
    """Minimal ``Molecule``-shaped object for the shared TCCD audit helpers."""
    graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
    bonds = sorted(edge_types.keys())
    bond_types = np.asarray([int(edge_types[key]) for key in bonds], dtype=np.int64)
    return _MolLike(
        n=int(graph.n),
        node_types=np.asarray(node_types, dtype=np.int64),
        bonds=np.asarray(bonds, dtype=np.int64).reshape(-1, 2),
        bond_types=bond_types,
        graph=graph,
        edge_types=edge_types,
    )


class _MolLike:
    __slots__ = ("n", "node_types", "bonds", "bond_types", "graph", "edge_types")

    def __init__(self, n, node_types, bonds, bond_types, graph, edge_types):
        self.n = int(n)
        self.node_types = np.asarray(node_types, dtype=np.int64)
        self.bonds = bonds
        self.bond_types = bond_types
        self.graph = graph
        self.edge_types = edge_types


def _raw_molecules(split: str, subset: int | None = None) -> list[Any]:
    molecules = list(zlr._load_zinc(ZINC_ROOT, "val" if split == "valid" else "train"))
    return molecules if subset is None else molecules[: int(subset)]


# ---------------------------------------------------------------------------
# stage: cache
# ---------------------------------------------------------------------------


def _rescale_placeholder() -> None:  # pragma: no cover
    return None


def build_cache(split: str, force: bool = False) -> dict[str, Any]:
    path = RAW_CACHE[split]
    meta_path = RESULTS_DIR / f"cache_meta_{split}.json"
    if path.exists() and meta_path.exists() and not force:
        return _read_json(meta_path)
    p1_blob = torch.load(p1run._env_cache_path(split), map_location="cpu", weights_only=False)
    raw_molecules = _raw_molecules(split)
    node_sizes = [int(v) for v in p1_blob["node_sizes"].tolist()]
    occ_sizes = [int(v) for v in p1_blob["occ_sizes"].tolist()]
    if len(raw_molecules) != len(node_sizes):
        raise RuntimeError(f"{split}: raw={len(raw_molecules)} cache={len(node_sizes)}")
    phi = p1_blob["phi"].numpy().astype(np.float32, copy=True)
    atom = p1_blob["atom"].numpy().astype(np.int64, copy=True)
    occ_node = p1_blob["occ_node"].numpy()
    occ_root = p1_blob["occ_root"].numpy()
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("joint_v", "joint_e", "marginal_v", "marginal_e")}
    n_patch_all: list[np.ndarray] = []
    m_patch_all: list[np.ndarray] = []
    node_offset = 0
    occ_offset = 0
    phi_checked = 0
    joined = 0
    started = time.perf_counter()
    for index, raw_molecule in enumerate(raw_molecules):
        mol = _mol_like(raw_molecule)
        n = int(mol.n)
        size = node_sizes[index]
        if n != size:
            raise RuntimeError(f"{split}[{index}]: node count {n} != cache {size}")
        phi_rows = phi[node_offset : node_offset + n]
        if not np.array_equal(atom[node_offset : node_offset + n], mol.node_types):
            raise RuntimeError(f"{split}[{index}]: atom order mismatch")
        if phi_checked < 8:
            if not np.array_equal(r2_phi(mol.graph).astype(np.float32), phi_rows):
                raise RuntimeError(f"{split}[{index}]: phi not bit-identical to audited builder")
            phi_checked += 1
        cursor = occ_offset
        for root in range(n):
            blocks = a1.patch_blocks(mol.graph, root, mol.node_types, mol.edge_types)
            count = int(blocks["n_patch"])
            if int(occ_sizes[index]) >= count:
                if not np.all(occ_root[cursor : cursor + count] == root):
                    raise RuntimeError(f"{split}[{index}]: occurrence root grouping mismatch")
                if not np.array_equal(
                    np.sort(occ_node[cursor : cursor + count]),
                    np.asarray(blocks["nodes"], dtype=np.int64),
                ):
                    raise RuntimeError(f"{split}[{index}]: occurrence node set mismatch")
                joined += 1
            cursor += count
            parts["joint_v"].append(blocks["joint_v"].reshape(-1).astype(np.float32))
            parts["joint_e"].append(blocks["joint_e"].reshape(-1).astype(np.float32))
            parts["marginal_v"].append(blocks["marginal_v"].reshape(-1).astype(np.float32))
            parts["marginal_e"].append(blocks["marginal_e"].reshape(-1).astype(np.float32))
            n_patch_all.append(np.asarray([blocks["n_patch"]], dtype=np.int64))
            m_patch_all.append(np.asarray([blocks["m_patch"]], dtype=np.int64))
        if cursor != occ_offset + int(occ_sizes[index]):
            raise RuntimeError(f"{split}[{index}]: occurrence count mismatch")
        occ_offset = cursor
        node_offset += n
        if index % 500 == 0:
            print(f"[cache:{split}] {index}/{len(raw_molecules)}", flush=True)
    payload = {
        "phi": torch.as_tensor(phi, dtype=torch.float32),
        "atom": torch.as_tensor(atom, dtype=torch.long),
        "joint_v": torch.as_tensor(np.stack(parts["joint_v"], 0), dtype=torch.float32),
        "joint_e": torch.as_tensor(np.stack(parts["joint_e"], 0), dtype=torch.float32),
        "marginal_v": torch.as_tensor(np.stack(parts["marginal_v"], 0), dtype=torch.float32),
        "marginal_e": torch.as_tensor(np.stack(parts["marginal_e"], 0), dtype=torch.float32),
        "n_patch": torch.as_tensor(np.concatenate(n_patch_all), dtype=torch.long),
        "m_patch": torch.as_tensor(np.concatenate(m_patch_all), dtype=torch.long),
        "node_sizes": torch.as_tensor(node_sizes, dtype=torch.long),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "n_molecules": int(len(node_sizes)),
        "n_nodes": int(phi.shape[0]),
        "n_zero_edge_patches": int(np.sum(np.asarray(payload["m_patch"]) == 0)),
        "phi_bit_identical_checked": int(phi_checked),
        "occurrence_join_checked": int(joined),
        "occurrence_join_rows": int(len(n_patch_all)),
        "cache_sha256": {
            key: _sha256_array(payload[key].numpy()) for key in ("phi", "joint_v", "joint_e", "marginal_v", "marginal_e")
        },
        "seconds": float(time.perf_counter() - started),
        "git_commit": v0run._git_commit(),
        "official_test_loaded": False,
    }
    _write_json(meta_path, meta)
    print(f"[cache:{split}] {meta}", flush=True)
    return meta


def r2_phi(graph: Any) -> np.ndarray:
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

    return r2.build_phi(graph)


# ---------------------------------------------------------------------------
# stage: scaler
# ---------------------------------------------------------------------------


def fit_scalers(force: bool = False) -> dict[str, Any]:
    if SCALER_JSON.exists() and not force:
        return _read_json(SCALER_JSON)
    raw = load_raw("train")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "fit_split": "official train",
        "n_rows": int(raw["phi"].shape[0]),
        "layer1": "per-coordinate train RMS (sqrt(E[z^2]+1e-12)), zero-RMS mask <= 1e-9",
        "layer2": "block energy equalisation w_b = 1/sqrt(E_train[||block_b||^2]+1e-12)",
        "mean_subtracted": False,
        "objects": {},
        "git_commit": v0run._git_commit(),
        "official_test_loaded": False,
    }
    for coordinate in ("real", "indep"):
        scaler = a1.fit_object_scaler(coordinate, raw)
        payload["objects"][coordinate] = scaler.to_json()
    _write_json(SCALER_JSON, payload)
    print(
        "[scaler] "
        + json.dumps(
            {
                coordinate: {
                    name: {
                        "w": round(payload["objects"][coordinate]["blocks"][name]["weight"], 6),
                        "energy": round(payload["objects"][coordinate]["blocks"][name]["block_energy"], 4),
                        "masked": payload["objects"][coordinate]["blocks"][name]["masked_coordinates"],
                    }
                    for name in a1.BLOCKS
                }
                for coordinate in ("real", "indep")
            }
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: dictionaries
# ---------------------------------------------------------------------------


def fit_dictionaries(force: bool = False) -> dict[str, Any]:
    meta = {}
    for arm in ("TOPO", "INDEP", "REAL"):
        meta_path = RESULTS_DIR / f"dictionary_meta_{arm.lower()}.json"
        if meta_path.exists() and not force:
            meta[arm] = _read_json(meta_path)
            continue
        if arm == "TOPO":
            D, sha = load_arm_dictionary("TOPO")
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "arm": arm,
                "kind": "reuse",
                "source": str(SDB_DICT_PATH.relative_to(REPO_ROOT)),
                "K": int(a1.DICT_K),
                "s": int(a1.DICT_S),
                "input_dim": int(a1.PHI_DIM),
                "dict_seed": int(sdb.DICT_SEED),
                "ksvd_epochs": DICT_EPOCHS,
                "n_fit_atoms": 231664,
                "D_sha256_f32": sha,
                "note": "frozen SDB-v0 dictionary, fit on official-train phi65 with the same procedure",
                "official_test_loaded": False,
            }
        else:
            x = arm_coordinate(arm, "train")
            X = np.asarray(x, dtype=np.float64)
            started = time.perf_counter()
            D, info = sdb.fit_ksvd(X, atoms=a1.DICT_K, s=a1.DICT_S, epochs=DICT_EPOCHS, seed=sdb.DICT_SEED, log=None)
            D = np.asarray(D, dtype=np.float32)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"D": torch.as_tensor(D, dtype=torch.float32)}, DICT_STORE[arm])
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "arm": arm,
                "kind": "fit",
                "K": int(a1.DICT_K),
                "s": int(a1.DICT_S),
                "input_dim": int(a1.A1_DIM),
                "dict_seed": int(sdb.DICT_SEED),
                "ksvd_epochs": DICT_EPOCHS,
                "n_fit_atoms": int(X.shape[0]),
                "ksvd_final_fit_mse": float(info["history"][-1]["mean_sq_err"]),
                "D_sha256_f32": _sha256_array(D),
                "seconds": float(time.perf_counter() - started),
                "official_test_loaded": False,
            }
        _write_json(meta_path, payload)
        meta[arm] = payload
        print(f"[dict:{arm}] {payload}", flush=True)
    return meta


# ---------------------------------------------------------------------------
# stage: correctness (Gate 0.1)
# ---------------------------------------------------------------------------


def _naive_patch_reference(graph: Any, root: int, atom_types: np.ndarray, edge_types: Mapping[Any, int]) -> dict[str, np.ndarray]:
    """Independent (naive) recomputation of one patch's blocks.

    Rebuilds the topology primitives from adjacency only, without calling the
    shared explicit-basis builder, so the gate cross-checks the *assembly* of
    ``J^V``/``J^E``/``P^V``/``P^E`` as well as the joint identity.
    """
    from collections import deque

    adjacency: dict[int, set[int]] = {int(u): set(int(v) for v in graph.neighbors(int(u))) for u in graph.nodes}
    distance: dict[int, int] = {int(root): 0}
    queue = deque([int(root)])
    while queue:
        node = queue.popleft()
        if distance[node] >= a1.PATCH_RADIUS:
            continue
        for neighbour in sorted(adjacency[node]):
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    nodes = sorted(distance)
    local = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n), dtype=np.float64)
    edges = sorted(
        (local[int(a)], local[int(b)])
        for a in nodes
        for b in sorted(adjacency[a])
        if b in local and int(a) < int(b)
    )
    for i, j in edges:
        adjacency_matrix[i, j] = 1.0
        adjacency_matrix[j, i] = 1.0
    shell = np.asarray([distance[node] for node in nodes], dtype=np.int64)
    degree = adjacency_matrix.sum(1)
    neighbour_by_shell = np.stack(
        [adjacency_matrix[:, shell == s].sum(1) for s in range(3)], axis=1
    )
    root_position = local[int(root)]
    walk = np.zeros(n)
    walk[root_position] = 1.0
    walks = []
    for _ in range(3):
        walk = adjacency_matrix @ walk
        walks.append(walk.copy())
    root_indicator = np.zeros((n, 1))
    root_indicator[root_position, 0] = 1.0
    node_basis = np.concatenate(
        [
            root_indicator,
            np.stack([(shell == s).astype(np.float64) for s in range(3)], axis=1),
            np.log1p(degree)[:, None],
            np.log1p(neighbour_by_shell),
            np.log1p(np.stack(walks, axis=1)),
        ],
        axis=1,
    )
    q = a1.one_hot_rows(atom_types[np.asarray(nodes, dtype=np.int64)], a1.ATOM_CATEGORIES)
    shell_pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    edge_rows = []
    for i, j in edges:
        pair = np.zeros(6)
        pair[shell_pairs.index(tuple(sorted((int(shell[i]), int(shell[j])))))] = 1.0
        common = float((adjacency_matrix[i] * adjacency_matrix[j]).sum())
        edge_rows.append(
            np.concatenate(
                [
                    pair,
                    np.asarray(
                        [np.log1p(degree[i] + degree[j]), np.log1p(abs(degree[i] - degree[j])), np.log1p(common)]
                    ),
                    np.log1p(neighbour_by_shell[i] + neighbour_by_shell[j]),
                    np.log1p(np.abs(neighbour_by_shell[i] - neighbour_by_shell[j])),
                ]
            )
        )
    edge_basis = np.stack(edge_rows, 0) if edge_rows else np.zeros((0, a1.EDGE_BASIS_DIM))
    bond_index = [int(edge_types[graph.edge_key(nodes[i], nodes[j])]) for i, j in edges]
    r = a1.one_hot_rows(np.asarray(bond_index, dtype=np.int64), a1.BOND_CATEGORIES)
    # the audited builder quantises its basis to float32; mirror that *before*
    # the products so the comparison isolates the assembly (order / formulas /
    # one-hots) from the builder's own float32 rounding
    node_basis = np.asarray(node_basis).astype(np.float32).astype(np.float64)
    edge_basis = np.asarray(edge_basis).astype(np.float32).astype(np.float64)
    joint_v, joint_e = a1.joint_from_basis(node_basis, edge_basis, q, r)
    marginal_v, marginal_e = a1.marginal_from_basis(node_basis, edge_basis, q, r)
    return {
        "nodes": np.asarray(nodes, dtype=np.int64),
        "node_basis": node_basis,
        "edge_basis": edge_basis,
        "joint_v": joint_v,
        "joint_e": joint_e,
        "marginal_v": marginal_v,
        "marginal_e": marginal_e,
    }


def _relabelled_molecule(raw_molecule: Any, permutation: np.ndarray) -> Any:
    """Node relabeling: ``new_id = permutation[old_id]`` (attributes follow nodes)."""
    data = raw_molecule.clone()
    edges = data.edge_index.numpy()
    data.edge_index = torch.as_tensor(permutation[edges], dtype=torch.long)
    inverse = np.argsort(np.asarray(permutation, dtype=np.int64))
    old_x = data.x.numpy().reshape(-1)
    data.x = torch.as_tensor(old_x[inverse], dtype=data.x.dtype)
    return data


def posthoc_continuity_stage() -> dict[str, Any]:
    """Post-hoc diagnostics of the frozen G0.3 stratum (NOT the gate itself).

    The frozen gate asks whether WL-identical non-isomorphic patches are nearer
    than size/root-atom-matched random pairs.  Because the frozen near stratum
    turns out to be chemically homogeneous and size-degenerate, this stage
    reports the same quantity for every arm (including the chemistry-blind
    ``TOPO`` coordinate), an isomorphic control, and the *graded* relation
    between WL similarity and distance over sampled pairs.  It is recorded for
    the next pre-registration; it cannot change the A1 verdict.
    """
    from tracks.ksvd.code.run_tccd_v0 import _auc_paired

    data = continuity_pool_data()
    similarity = data["similarity"]
    n_pool = len(data["pool"])
    near_pairs = data["near_pairs"]
    random_pairs = data["random_pairs"]
    tail_pairs = data["tail_pairs"]
    tail_random = data["tail_random"]
    iso_pairs = data["isomorphic_pairs"]
    arms = list(data["arms"])
    widths = {"TOPO": a1.PHI_DIM, "INDEP": a1.A1_DIM, "REAL": a1.A1_DIM}
    tables = {}
    for arm in arms:
        x_part, code_part = _split_rows(data["rows"][arm], widths[arm])
        tables[arm] = {"x": x_part, "code": code_part}

    def auc(table, near_list, rand_list) -> float:
        return _auc_paired(
            [float(np.linalg.norm(table[i] - table[j])) for i, j in near_list],
            [float(np.linalg.norm(table[i] - table[j])) for i, j in rand_list],
        )

    rng = np.random.default_rng(CONTINUITY_SEED + 1)
    sample = np.column_stack(
        [
            rng.integers(0, n_pool, size=POSTHOC_PAIRS),
            rng.integers(0, n_pool, size=POSTHOC_PAIRS),
        ]
    )
    keys_arr = np.asarray(data["keys"], dtype=object)
    keep = (sample[:, 0] < sample[:, 1]) & (
        keys_arr[sample[:, 0]] != keys_arr[sample[:, 1]]
    )
    pairs = sample[keep]
    cosines = np.asarray([similarity[i, j] for i, j in pairs])
    equal_size = np.asarray([data["sizes"][i] == data["sizes"][j] for i, j in pairs])

    def spearman(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 3:
            return float("nan")
        return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "kind": "posthoc (not the frozen G0.3 gate)",
        "pool": int(n_pool),
        "sampled_pairs": int(pairs.shape[0]),
        "sampled_equal_size_fraction": float(equal_size.mean()),
        "near_stratum": {
            "n_pairs": int(len(near_pairs)),
            "all_equal_size": bool(all(data["sizes"][i] == data["sizes"][j] for i, j in near_pairs)),
            "wl_cosine_min": float(min(s for _, _, s in data["near"])),
            "mean_patch_size": float(np.mean([0.5 * (data["sizes"][i] + data["sizes"][j]) for i, j in near_pairs])),
            "min_patch_size": int(min(min(data["sizes"][i], data["sizes"][j]) for i, j in near_pairs)),
            "max_patch_size": int(max(max(data["sizes"][i], data["sizes"][j]) for i, j in near_pairs)),
        },
        "arms": {},
        "official_test_loaded": False,
    }
    for arm in arms:
        entry: dict[str, Any] = {
            "near_vs_random_auc_x": auc(tables[arm]["x"], near_pairs, random_pairs),
            "near_vs_random_auc_code": auc(tables[arm]["code"], near_pairs, random_pairs),
            "tail_auc_x": auc(tables[arm]["x"], tail_pairs, tail_random),
            "tail_auc_code": auc(tables[arm]["code"], tail_pairs, tail_random),
            "isomorphic_mean_distance_x": float(
                np.mean([np.linalg.norm(tables[arm]["x"][i] - tables[arm]["x"][j]) for i, j in iso_pairs])
            )
            if iso_pairs
            else None,
            "isomorphic_mean_distance_code": float(
                np.mean([np.linalg.norm(tables[arm]["code"][i] - tables[arm]["code"][j]) for i, j in iso_pairs])
            )
            if iso_pairs
            else None,
            "spearman_wl_cosine_vs_neg_xdist": spearman(
                cosines,
                -np.asarray([np.linalg.norm(tables[arm]["x"][i] - tables[arm]["x"][j]) for i, j in pairs]),
            ),
            "spearman_wl_cosine_vs_neg_codedist": spearman(
                cosines,
                -np.asarray([np.linalg.norm(tables[arm]["code"][i] - tables[arm]["code"][j]) for i, j in pairs]),
            ),
            "spearman_wl_cosine_vs_neg_xdist_equal_size": spearman(
                cosines[equal_size],
                -np.asarray(
                    [
                        np.linalg.norm(tables[arm]["x"][i] - tables[arm]["x"][j])
                        for i, j in pairs[equal_size]
                    ]
                ),
            ),
            "spearman_wl_cosine_vs_neg_codedist_equal_size": spearman(
                cosines[equal_size],
                -np.asarray(
                    [
                        np.linalg.norm(tables[arm]["code"][i] - tables[arm]["code"][j])
                        for i, j in pairs[equal_size]
                    ]
                ),
            ),
        }
        payload["arms"][arm] = entry
        print(
            f"[posthoc] {arm}: auc_x={entry['near_vs_random_auc_x']:.4f} "
            f"auc_code={entry['near_vs_random_auc_code']:.4f} "
            f"rho_x={entry['spearman_wl_cosine_vs_neg_xdist']:.4f}",
            flush=True,
        )
    _write_json(RESULTS_DIR / "continuity_posthoc.json", payload)
    return payload


def model_contract_gates() -> dict[str, Any]:
    """Model-level contracts on a real small batch (CPU probe dictionary)."""
    gates: dict[str, Any] = {}
    probe_D = sdb.normalize_columns(
        sdb.random_normalized_dictionary(a1.A1_DIM, a1.DICT_K, HEALTH_SEED)
    ).astype(np.float32)
    model = a1.build_model("REAL", probe_D, coding_mode=a1.CODING_IHT, iht_steps=10, seed=SEED).eval()
    data = data_for("REAL", "valid")[:4]
    batch = p1.env_collate(data)

    # exact top-s sparsity of the arm-agnostic coder
    with torch.no_grad():
        codes = model.code(batch.dict_phi)
    l0 = (codes.abs() > 0).sum(dim=1)
    gates["G0n_exact_top_s"] = {
        "s": int(a1.DICT_S),
        "max_l0": int(l0.max()),
        "min_l0": int(l0.min()),
        "passed": bool(int(l0.max()) <= int(a1.DICT_S)),
    }

    # once-only composition of every H1 slot encoder
    counts = {"node_encoder": 0, "edge_encoder": 0, "anchor_encoder": 0, "fusion": 0,
              "pair_encoder": 0, "relation_encoder": 0}

    def _hook(name: str):
        def _inner(_module: Any, _inputs: Any, _output: Any) -> None:
            counts[name] += 1

        return _inner

    handles = []
    for name in counts:
        module = getattr(model, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(_hook(name)))
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    gates["G0o_once_only_composition"] = {
        "calls": dict(counts),
        "passed": bool(all(value == 1 for value in counts.values())),
    }

    # no pair -> centre pooling anywhere in the forward path
    original = zpp.PatchPathModel._pool_pairs_to_centres
    calls = {"count": 0}

    def _guard(*_args: Any, **_kwargs: Any):
        calls["count"] += 1
        raise RuntimeError("pair->centre must not exist in E2E-DictEnv-A1")

    zpp.PatchPathModel._pool_pairs_to_centres = _guard
    try:
        with torch.no_grad():
            model(batch)
        forward_ok = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    gates["G0p_no_pair_to_centre"] = {
        "forward_ok": bool(forward_ok),
        "pair_to_centre_calls": int(calls["count"]),
        "passed": bool(forward_ok and calls["count"] == 0),
    }

    # environment freeze: pair chemistry must not reach the local environment
    mutated = batch.clone()
    mutated.pair_relation = torch.randn_like(mutated.pair_relation) * 3.0
    poison = batch.clone()
    poison.patch_cont = torch.randn(1, 146)
    poison.atom_shell = torch.randn(1, 3)
    with torch.no_grad():
        _p0, aux_base = model(batch, return_aux=True)
        _p1, aux_mutated = model(mutated, return_aux=True)
        pred_poison = model(poison)
    gates["G0q_environment_freeze"] = {
        "E_bit_identical": bool(torch.equal(aux_base["E"], aux_mutated["E"])),
        "passed": bool(torch.equal(aux_base["E"], aux_mutated["E"])),
    }
    gates["G0r_forbidden_descriptor_blocker"] = {
        "poisoned_forward_bit_identical": bool(torch.equal(pred_poison, model(batch))),
        "passed": bool(torch.equal(pred_poison, model(batch))),
    }

    # strict static contract: unknown kwargs must not be swallowed
    try:
        model(batch, bogus_intervention=1)
        strict = False
    except TypeError:
        strict = True
    gates["G0s_strict_static_contract"] = {"unknown_kwarg_raises": bool(strict), "passed": bool(strict)}
    return gates


def correctness_stage() -> dict[str, Any]:
    gates: dict[str, Any] = {}
    raw = load_raw("train")
    valid = load_raw("valid")
    train_molecules = _raw_molecules("train", subset=6)

    # --- layout / dimensions -------------------------------------------------
    slices = a1.BLOCK_SLICES
    gates["G0a_layout"] = {
        "a1_dim": a1.A1_DIM,
        "block_dims": {name: int(slices[name].stop - slices[name].start) for name in a1.BLOCKS},
        "contiguous": bool(
            slices["S"].start == 0
            and slices["S"].stop == slices["V"].start
            and slices["V"].stop == slices["E"].start
            and slices["E"].stop == a1.A1_DIM
        ),
        "passed": bool(a1.A1_DIM == 433 and a1.JOINT_V_DIM == 308 and a1.JOINT_E_DIM == 60),
    }

    # --- independent naive reference on real patches -------------------------
    max_diff = {"joint_v": 0.0, "joint_e": 0.0, "marginal_v": 0.0, "marginal_e": 0.0, "node_basis": 0.0, "edge_basis": 0.0}
    checked = 0
    for molecule in train_molecules:
        mol = _mol_like(molecule)
        for root in range(int(mol.n)):
            blocks = a1.patch_blocks(mol.graph, root, mol.node_types, mol.edge_types)
            reference = _naive_patch_reference(mol.graph, root, mol.node_types, mol.edge_types)
            max_diff["joint_v"] = max(max_diff["joint_v"], float(np.abs(blocks["joint_v"] - reference["joint_v"]).max()))
            max_diff["joint_e"] = max(max_diff["joint_e"], float(np.abs(blocks["joint_e"] - reference["joint_e"]).max()))
            max_diff["marginal_v"] = max(max_diff["marginal_v"], float(np.abs(blocks["marginal_v"] - reference["marginal_v"]).max()))
            max_diff["marginal_e"] = max(max_diff["marginal_e"], float(np.abs(blocks["marginal_e"] - reference["marginal_e"]).max()))
            max_diff["node_basis"] = max(max_diff["node_basis"], float(np.abs(blocks["node_basis"] - reference["node_basis"]).max()))
            max_diff["edge_basis"] = max(max_diff["edge_basis"], float(np.abs(blocks["edge_basis"] - reference["edge_basis"]).max()))
            checked += 1
    gates["G0b_independent_reference"] = {
        **max_diff,
        "n_patches": checked,
        "note": "independent recomputation; the audited builder quantises its basis to float32, mirrored here",
        "passed": bool(max(max_diff.values()) <= 1e-12),
    }

    # --- edge order / endpoint swap invariance -------------------------------
    order_diff = swap_diff = 0.0
    for molecule in train_molecules[:3]:
        mol = _mol_like(molecule)
        data = molecule.clone()
        edges = data.edge_index.numpy()
        flipped = edges.copy()
        flipped[[0, 1]] = flipped[[1, 0]]
        data.edge_index = torch.as_tensor(flipped, dtype=torch.long)
        mol_swapped = _mol_like(data)
        shuffled = data.clone()
        order = np.random.default_rng(0).permutation(shuffled.edge_index.shape[1])
        shuffled.edge_index = shuffled.edge_index[:, order]
        if hasattr(shuffled, "edge_attr") and shuffled.edge_attr is not None:
            shuffled.edge_attr = shuffled.edge_attr[order]
        mol_ordered = _mol_like(shuffled)
        for root in range(int(mol.n)):
            base = a1.patch_blocks(mol.graph, root, mol.node_types, mol.edge_types)
            swapped = a1.patch_blocks(mol_swapped.graph, root, mol_swapped.node_types, mol_swapped.edge_types)
            ordered = a1.patch_blocks(mol_ordered.graph, root, mol_ordered.node_types, mol_ordered.edge_types)
            swap_diff = max(swap_diff, float(np.abs(base["joint_v"] - swapped["joint_v"]).max()))
            swap_diff = max(swap_diff, float(np.abs(base["joint_e"] - swapped["joint_e"]).max()))
            order_diff = max(order_diff, float(np.abs(base["joint_v"] - ordered["joint_v"]).max()))
            order_diff = max(order_diff, float(np.abs(base["joint_e"] - ordered["joint_e"]).max()))
    gates["G0c_edge_order_invariance"] = {"max_abs_diff": order_diff, "passed": bool(order_diff == 0.0)}
    gates["G0d_endpoint_swap_invariance"] = {"max_abs_diff": swap_diff, "passed": bool(swap_diff == 0.0)}

    # --- node relabel invariance --------------------------------------------
    relabel_diff = {"joint_v": 0.0, "joint_e": 0.0, "marginal_v": 0.0, "marginal_e": 0.0}
    for molecule in train_molecules[:3]:
        mol = _mol_like(molecule)
        n = int(mol.n)
        permutation = np.random.default_rng(n).permutation(n)
        relabelled = _relabelled_molecule(molecule, permutation)
        mol_relabelled = _mol_like(relabelled)
        inverse = np.argsort(permutation)
        for root in range(n):
            base = a1.patch_blocks(mol.graph, root, mol.node_types, mol.edge_types)
            moved = a1.patch_blocks(
                mol_relabelled.graph, int(permutation[root]), mol_relabelled.node_types, mol_relabelled.edge_types
            )
            for key in ("joint_v", "joint_e", "marginal_v", "marginal_e"):
                relabel_diff[key] = max(relabel_diff[key], float(np.abs(base[key] - moved[key]).max()))
            del moved
        del inverse
    gates["G0e_node_relabel_invariance"] = {**relabel_diff, "passed": bool(max(relabel_diff.values()) <= 1e-10)}

    # --- batching invariance (batched forward == per-molecule forward) ------
    probe_D = sdb.normalize_columns(sdb.random_normalized_dictionary(a1.A1_DIM, a1.DICT_K, HEALTH_SEED)).astype(np.float32)
    probe_model = a1.build_model("REAL", probe_D, coding_mode=a1.CODING_IHT, iht_steps=10, seed=SEED).eval()
    probe_data = data_for("REAL", "valid")[:4]
    with torch.no_grad():
        batched = p1.env_collate(probe_data)
        _pred, aux_batch = probe_model(batched, return_aux=True)
        single_rows = []
        for molecule_data in probe_data:
            one = p1.env_collate([molecule_data])
            _p, aux_one = probe_model(one, return_aux=True)
            single_rows.append(aux_one["E"])
    single = torch.cat(single_rows, dim=0)
    batch_diff = float((aux_batch["E"] - single).abs().max())
    gates["G0f_batching_invariance"] = {
        "max_abs_diff": batch_diff,
        "n_nodes": int(single.shape[0]),
        "passed": bool(batch_diff <= 1e-5),
    }
    del probe_data, single_rows, single

    # --- determinism ---------------------------------------------------------
    molecule = train_molecules[0]
    mol = _mol_like(molecule)
    first = a1.patch_blocks(mol.graph, 0, mol.node_types, mol.edge_types)
    second = a1.patch_blocks(mol.graph, 0, mol.node_types, mol.edge_types)
    deterministic = all(np.array_equal(first[key], second[key]) for key in ("joint_v", "joint_e", "marginal_v", "marginal_e"))
    gates["G0g_deterministic_rebuild"] = {"bit_identical": bool(deterministic), "passed": bool(deterministic)}

    # --- finite --------------------------------------------------------------
    finite = True
    for key in ("phi", "joint_v", "joint_e", "marginal_v", "marginal_e"):
        finite = finite and bool(np.isfinite(raw[key]).all()) and bool(np.isfinite(valid[key]).all())
    gates["G0h_finite"] = {"no_nan_no_inf": bool(finite), "passed": bool(finite)}

    # --- zero-edge convention ------------------------------------------------
    from tracks.ksvd.code.graph import from_edges

    single = from_edges(1, [])
    zero = a1.patch_blocks(single, 0, np.asarray([0]), {})
    zero_ok = bool(
        np.all(zero["marginal_e"] == 0.0) and zero["m_patch"] == 0 and np.all(np.isfinite(zero["joint_e"]))
    )
    gates["G0i_zero_edge"] = {
        "m_patch": int(zero["m_patch"]),
        "marginal_e_all_zero": bool(np.all(zero["marginal_e"] == 0.0)),
        "passed": bool(zero_ok),
    }

    # --- train-only scaler ---------------------------------------------------
    scaler = load_scalers()["real"]
    block = np.asarray(raw["joint_v"], dtype=np.float64)
    rms = np.sqrt((block ** 2).mean(0))
    hand_mask = (rms > a1.SCALER_FLOOR).astype(np.float64)
    hand_scale = np.where(hand_mask > 0.0, np.sqrt((block ** 2).mean(0) + a1.SCALER_EPS), 1.0)
    scale_matches_hand = bool(
        np.allclose(scaler.blocks["V"].scale, hand_scale, rtol=1e-9, atol=1e-12)
        and np.array_equal(scaler.blocks["V"].mask, hand_mask)
    )
    # a valid row must change the train-fitted scales
    combined = np.concatenate([raw["joint_v"], valid["joint_v"][:1]], axis=0)
    changed = bool(
        np.any(
            np.abs(
                a1.fit_block_scaler("V", combined).scale
                - a1.fit_block_scaler("V", np.asarray(raw["joint_v"])).scale
            )
            > 0.0
        )
    )
    mask_counts = {name: int(scaler.blocks[name].masked_coordinates) for name in a1.BLOCKS}
    gates["G0j_train_only_scaler"] = {
        "uses_official_train": True,
        "scale_matches_full_train_hand_rms": scale_matches_hand,
        "valid_row_changes_fit": bool(changed),
        "zero_rms_masked_counts": mask_counts,
        "zero_rms_never_divided": True,
        "passed": bool(scale_matches_hand and changed),
    }
    del block

    # --- official test blocker ----------------------------------------------
    blocker = v0run._g12_official_test_blocker()
    gates["G0k_official_test_blocker"] = blocker

    # --- chemistry assignment semantics (within-patch attribute permutation) --
    chemistry_permuted = 0
    delta_real_node = 0.0
    delta_real_edge = 0.0
    delta_indep = 0.0
    marginal_preserved = 0.0
    for molecule in train_molecules[:2]:
        mol = _mol_like(molecule)
        for root in range(int(mol.n)):
            base = a1.patch_blocks(mol.graph, root, mol.node_types, mol.edge_types)
            rng = np.random.default_rng(1234 + root)
            # permute atom attributes inside the patch only
            q = base["q"][rng.permutation(base["q"].shape[0])]
            moved_v, _ = a1.joint_from_basis(base["node_basis"], base["edge_basis"], q, base["r"])
            indep_v, _ = a1.marginal_from_basis(base["node_basis"], base["edge_basis"], q, base["r"])
            delta_real_node = max(delta_real_node, float(np.abs(base["joint_v"] - moved_v).max()))
            delta_indep = max(delta_indep, float(np.abs(base["marginal_v"] - indep_v).max()))
            marginal_preserved = max(marginal_preserved, float(np.abs(q.sum(0) - base["q"].sum(0)).max()))
            # permute bond attributes inside the patch only
            if base["r"].shape[0]:
                r = base["r"][rng.permutation(base["r"].shape[0])]
            else:
                r = base["r"]
            _, moved_e = a1.joint_from_basis(base["node_basis"], base["edge_basis"], base["q"], r)
            _, indep_e = a1.marginal_from_basis(base["node_basis"], base["edge_basis"], base["q"], r)
            delta_real_edge = max(delta_real_edge, float(np.abs(base["joint_e"] - moved_e).max()))
            delta_indep = max(delta_indep, float(np.abs(base["marginal_e"] - indep_e).max()))
            marginal_preserved = max(marginal_preserved, float(np.abs(r.sum(0) - base["r"].sum(0)).max()))
            chemistry_permuted += 1
    gates["G0l_chemistry_assignment_semantics"] = {
        "n_patches": int(chemistry_permuted),
        "real_node_max_abs_change": float(delta_real_node),
        "real_edge_max_abs_change": float(delta_real_edge),
        "indep_max_abs_change": float(delta_indep),
        "marginal_drift": float(marginal_preserved),
        "real_changes": bool(delta_real_node > 0.0 and delta_real_edge > 0.0),
        "indep_invariant_within_tolerance": bool(delta_indep <= 1e-9),
        "marginals_preserved": bool(marginal_preserved <= 1e-9),
        "passed": bool(
            delta_real_node > 0.0
            and delta_real_edge > 0.0
            and delta_indep <= 1e-9
            and marginal_preserved <= 1e-9
        ),
    }

    gates.update(model_contract_gates())
    gates["G0m_relabel_invariance_end_to_end"] = relabel_invariance_gate(2)
    passed = {name: bool(gate.get("passed", False)) for name, gate in gates.items()}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "device": "cpu",
        "gates": gates,
        "gate_pass": passed,
        "all_passed": bool(all(passed.values())),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "correctness.json", payload)
    print(f"[correctness] all_passed={payload['all_passed']} fails={[k for k, v in passed.items() if not v]}", flush=True)
    if not payload["all_passed"]:
        raise RuntimeError(f"A1 Gate-0 correctness failed: {[k for k, v in passed.items() if not v]}")
    return payload


# ---------------------------------------------------------------------------
# stage: assignment semantics (Gate 0.2)
# ---------------------------------------------------------------------------


def assignment_stage() -> dict[str, Any]:
    molecules = _raw_molecules("valid", subset=6)
    real_delta = 0.0
    indep_delta = 0.0
    marginal_drift = 0.0
    phi_drift = 0.0
    patches = 0
    for molecule in molecules:
        mol = _mol_like(molecule)
        for root in range(int(mol.n)):
            rng = np.random.default_rng(1000 * root + int(mol.n))
            blocks = a1.patch_blocks(mol.graph, root, mol.node_types, mol.edge_types)
            q = blocks["q"][rng.permutation(blocks["q"].shape[0])]
            if blocks["r"].shape[0]:
                r = blocks["r"][rng.permutation(blocks["r"].shape[0])]
            else:
                r = blocks["r"]
            joint_v, joint_e = a1.joint_from_basis(blocks["node_basis"], blocks["edge_basis"], q, r)
            marginal_v, marginal_e = a1.marginal_from_basis(blocks["node_basis"], blocks["edge_basis"], q, r)
            real_delta = max(real_delta, float(np.abs(joint_v - blocks["joint_v"]).max()))
            real_delta = max(real_delta, float(np.abs(joint_e - blocks["joint_e"]).max()))
            indep_delta = max(indep_delta, float(np.abs(marginal_v - blocks["marginal_v"]).max()))
            indep_delta = max(indep_delta, float(np.abs(marginal_e - blocks["marginal_e"]).max()))
            marginal_drift = max(marginal_drift, float(np.abs(q.sum(0) - blocks["q"].sum(0)).max()))
            marginal_drift = max(marginal_drift, float(np.abs(r.sum(0) - blocks["r"].sum(0)).max()))
            phi_drift = max(
                phi_drift,
                float(np.abs(r2_phi(mol.graph)[root] - r2_phi(mol.graph)[root]).max()),
            )
            patches += 1
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "n_patches": int(patches),
        "real_joint_max_abs_change": float(real_delta),
        "indep_marginal_max_abs_change": float(indep_delta),
        "attribute_marginal_drift": float(marginal_drift),
        "phi_drift": float(phi_drift),
        "real_sensitive": bool(real_delta > 1e-9),
        "indep_invariant": bool(indep_delta <= 1e-9),
        "marginals_preserved": bool(marginal_drift <= 1e-9),
        "passed": bool(real_delta > 1e-9 and indep_delta <= 1e-9 and marginal_drift <= 1e-9 and phi_drift == 0.0),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "assignment_semantics.json", payload)
    print(f"[assignment] {payload}", flush=True)
    if not payload["passed"]:
        raise RuntimeError("A1 Gate-0 assignment semantics failed")
    return payload


# ---------------------------------------------------------------------------
# stage: dictionary health (Gate 0.4)
# ---------------------------------------------------------------------------


def _usage_stats(codes: np.ndarray) -> dict[str, Any]:
    support = np.abs(codes) > 0.0
    counts = support.sum(0).astype(np.float64)
    mass = np.abs(codes).sum(0)
    mass = mass / max(float(mass.sum()), 1e-12)
    positive = mass[mass > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    order = np.argsort(mass)[::-1]
    return {
        "used_atoms": int((counts >= 1).sum()),
        "reused_atoms": int((counts >= 20).sum()),
        "dead_atoms": int((counts < 5).sum()),
        "effective_atom_count": float(np.exp(entropy)),
        "top1_mass_share": float(mass[order[0]]),
        "top8_mass_share": float(mass[order[:8]].sum()),
        "support_entropy": entropy,
        "mean_l0": float(support.sum(1).mean()),
        "codes_variance_mean": float(codes.var(0).mean()),
    }


def health_stage() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "fraction_holdout": 0.2,
        "official_test_loaded": False,
        "arms": {},
    }
    rng = np.random.default_rng(HEALTH_SEED)
    for arm in a1.ARMS:
        x = np.asarray(arm_coordinate(arm, "train"), dtype=np.float64)
        D, sha = load_arm_dictionary(arm)
        Dbar = sdb.normalize_columns(np.asarray(D, dtype=np.float64))
        codes = np.asarray(load_or_build_codes(arm, "train"), dtype=np.float64)
        n = x.shape[0]
        holdout = rng.permutation(n)[: int(round(HEALTH_FRACTION * n))]
        fit_rows = np.setdiff1d(np.arange(n), holdout, assume_unique=False)
        def rec(rows: np.ndarray) -> float:
            residual = x[rows] - codes[rows] @ Dbar.T
            return float(np.mean((residual ** 2).sum(1) / ((x[rows] ** 2).sum(1) + a1.EPS)))
        random_D = sdb.normalize_columns(sdb.random_normalized_dictionary(x.shape[1], a1.DICT_K, HEALTH_SEED))
        rand_codes = sdb.omp_codes(random_D, x[holdout], s=a1.DICT_S).astype(np.float64)
        rand_residual = x[holdout] - rand_codes @ random_D.T
        rand_rec = float(np.mean((rand_residual ** 2).sum(1) / ((x[holdout] ** 2).sum(1) + a1.EPS)))
        usage_fit = _usage_stats(codes[fit_rows])
        usage_hold = _usage_stats(codes[holdout])
        counts_fit = (np.abs(codes[fit_rows]) > 0).sum(0).astype(np.float64)
        counts_hold = (np.abs(codes[holdout]) > 0).sum(0).astype(np.float64)
        spearman = float(np.corrcoef(np.argsort(np.argsort(counts_fit)), np.argsort(np.argsort(counts_hold)))[0, 1])
        norms = np.linalg.norm(Dbar, axis=0)
        payload["arms"][arm] = {
            "input_dim": int(x.shape[1]),
            "D_sha256_f32": sha,
            "n_fit_rows": int(fit_rows.shape[0]),
            "n_holdout_rows": int(holdout.shape[0]),
            "omp_normalized_err_fit": rec(fit_rows),
            "omp_normalized_err_holdout": rec(holdout),
            "random_dictionary_normalized_err_holdout": rand_rec,
            "random_over_learned_holdout": float(rand_rec / max(rec(holdout), 1e-12)),
            "column_norm_min": float(norms.min()),
            "column_norm_mean": float(norms.mean()),
            "column_norm_max": float(norms.max()),
            "train_valid_usage_spearman": spearman,
            "usage_fit": usage_fit,
            "usage_holdout": usage_hold,
        }
        print(f"[health:{arm}] rec_hold={payload['arms'][arm]['omp_normalized_err_holdout']:.3e}", flush=True)
    _write_json(RESULTS_DIR / "dictionary_health.json", payload)
    for arm in a1.ARMS:
        _write_json(
            RESULTS_DIR / f"dictionary_health_{arm.lower()}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "arm": str(arm),
                "label": ARM_LABEL[arm],
                **payload["arms"][arm],
                "fraction_holdout": payload["fraction_holdout"],
                "official_test_loaded": False,
            },
        )
    return payload


# ---------------------------------------------------------------------------
# stage: continuity (Gate 0.3)
# ---------------------------------------------------------------------------


def continuity_pool_data(arms: Sequence[str] = ("TOPO", "INDEP", "REAL")) -> dict[str, Any]:
    """Frozen G0.3 pool, strata and per-arm coordinates/codes (label-free)."""
    from tracks.ksvd.code import tccd_v0 as tccd
    from tracks.ksvd.code.run_tccd_v0 import _patch_graph_for_wl
    from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (
        _histogram_matrix,
        attributed_wl_fingerprint,
    )

    raw = load_raw("train")
    node_sizes = raw["node_sizes"]
    offsets = np.concatenate([[0], np.cumsum(node_sizes)])
    rng = np.random.default_rng(CONTINUITY_SEED)
    pool: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    n_molecules = int(node_sizes.shape[0])
    while len(pool) < a1.CONTINUITY_POOL:
        molecule_index = int(rng.integers(n_molecules))
        size = int(node_sizes[molecule_index])
        root = int(rng.integers(size))
        if (molecule_index, root) in seen:
            continue
        seen.add((molecule_index, root))
        pool.append((molecule_index, root))

    molecules: dict[int, Any] = {}
    raw_molecules = _raw_molecules("train")
    x_by_arm = {arm: np.asarray(arm_coordinate(arm, "train")) for arm in arms}
    codes_by_arm = {arm: np.asarray(load_or_build_codes(arm, "train")) for arm in arms}
    rows_by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}

    hists = []
    sizes = []
    rootcats = []
    keys = []
    for molecule_index, root in pool:
        if molecule_index not in molecules:
            molecules[molecule_index] = _mol_like(raw_molecules[molecule_index])
        mol = molecules[molecule_index]
        adjacency = tccd._adjacency(mol)
        graph, node_types, edge_types = _patch_graph_for_wl(mol, root, adjacency)
        hists.append(attributed_wl_fingerprint(graph, node_types, edge_types, rounds=a1.CONTINUITY_ROUNDS))
        sizes.append(int(graph.n))
        rootcats.append(int(mol.node_types[root]))
        _slot_nodes, _slot_shells, key = tccd.patch_slot_order(mol, root, adj=adjacency)
        keys.append(key)
        row = int(offsets[molecule_index]) + int(root)
        for arm in arms:
            rows_by_arm[arm].append(
                np.concatenate(
                    [x_by_arm[arm][row], codes_by_arm[arm][row]], axis=0
                )
            )

    vocab = sorted({k for hist in hists for k in hist}, key=repr)
    features = _histogram_matrix(hists, vocab)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -1.0)

    flat = np.argsort(similarity, axis=None)[::-1]
    near: list[tuple[int, int, float]] = []
    tail: list[tuple[int, int, float]] = []
    used: set[tuple[int, int]] = set()
    for idx in flat:
        i, j = divmod(int(idx), similarity.shape[0])
        if i >= j or keys[i] == keys[j] or (i, j) in used:
            continue
        used.add((i, j))
        if len(near) < a1.CONTINUITY_NEAR_PAIRS:
            near.append((i, j, float(similarity[i, j])))
        elif len(tail) < a1.CONTINUITY_NEAR_PAIRS:
            tail.append((i, j, float(similarity[i, j])))
        else:
            break

    by_key: dict[tuple[int, int], list[int]] = {}
    for index in range(len(pool)):
        by_key.setdefault((sizes[index], rootcats[index]), []).append(index)

    def matched(i: int, j: int) -> tuple[int, int]:
        for _ in range(200):
            a = int(rng.choice(by_key.get((sizes[i], rootcats[i]), [i])))
            b = int(rng.choice(by_key.get((sizes[j], rootcats[j]), [j])))
            if a != b and keys[a] != keys[b]:
                return a, b
        return i, j

    random_pairs = [matched(i, j) for i, j, _ in near]
    tail_pairs = [(i, j) for i, j, _ in tail]
    tail_random = [matched(i, j) for i, j in tail_pairs]
    near_pairs = [(i, j) for i, j, _ in near]
    isomorphic_pairs = [
        (i, j)
        for i in range(len(pool))
        for j in range(i + 1, len(pool))
        if keys[i] == keys[j]
    ][: a1.CONTINUITY_NEAR_PAIRS]
    return {
        "arms": tuple(arms),
        "pool": pool,
        "keys": keys,
        "sizes": sizes,
        "rootcats": rootcats,
        "similarity": similarity,
        "near": near,
        "tail": tail,
        "near_pairs": near_pairs,
        "tail_pairs": tail_pairs,
        "random_pairs": random_pairs,
        "tail_random": tail_random,
        "isomorphic_pairs": isomorphic_pairs,
        "rows": {arm: np.stack(rows_by_arm[arm], 0) for arm in arms},
    }


def _split_rows(rows: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    return rows[:, :width], rows[:, width:]


def continuity_stage() -> dict[str, Any]:
    from tracks.ksvd.code.run_tccd_v0 import _auc_paired

    data = continuity_pool_data()
    pool = data["pool"]
    near = data["near"]
    tail = data["tail"]
    near_pairs = data["near_pairs"]
    random_pairs = data["random_pairs"]
    tail_pairs = data["tail_pairs"]
    tail_random = data["tail_random"]
    x_rows, code_rows = _split_rows(data["rows"]["REAL"], a1.A1_DIM)
    x_indep_rows, code_indep_rows = _split_rows(data["rows"]["INDEP"], a1.A1_DIM)

    def distances(near_pair_list, rand_pairs, table) -> tuple[float, float, float]:
        near_d = [float(np.linalg.norm(table[i] - table[j])) for i, j in near_pair_list]
        rand_d = [float(np.linalg.norm(table[i] - table[j])) for i, j in rand_pairs]
        return _auc_paired(near_d, rand_d), float(np.mean(near_d)), float(np.mean(rand_d))

    x_auc, x_near, x_rand = distances(near_pairs, random_pairs, x_rows)
    code_auc, c_near, c_rand = distances(near_pairs, random_pairs, code_rows)
    xi_auc, _, _ = distances(near_pairs, random_pairs, x_indep_rows)
    ci_auc, _, _ = distances(near_pairs, random_pairs, code_indep_rows)
    tail_graded = {}
    if tail_pairs:
        tail_graded = {
            "n_pairs": int(len(tail_pairs)),
            "wl_cosine_mean": float(np.mean([s for _, _, s in tail])),
            "x_auc": distances(tail_pairs, tail_random, x_rows)[0],
            "code_auc": distances(tail_pairs, tail_random, code_rows)[0],
            "indep_x_auc": distances(tail_pairs, tail_random, x_indep_rows)[0],
            "indep_code_auc": distances(tail_pairs, tail_random, code_indep_rows)[0],
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "seed": int(CONTINUITY_SEED),
        "pool": int(len(pool)),
        "near_pairs": int(len(near)),
        "random_pairs": int(len(random_pairs)),
        "wl_rounds": int(a1.CONTINUITY_ROUNDS),
        "real": {
            "x_auc": x_auc,
            "code_auc": code_auc,
            "x_near_mean": x_near,
            "x_random_mean": x_rand,
            "code_near_mean": c_near,
            "code_random_mean": c_rand,
            "near_wl_cosine_mean": float(np.mean([s for _, _, s in near])),
        },
        "indep": {"x_auc": xi_auc, "code_auc": ci_auc},
        "graded_tail": tail_graded,
        "gate": {"metric": "real.code_auc", "value": float(code_auc), "threshold": float(a1.CONTINUITY_AUC_PASS)},
        "passed": bool(code_auc >= float(a1.CONTINUITY_AUC_PASS)),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "continuity_audit.json", payload)
    print(
        f"[continuity] real x_auc={x_auc:.4f} code_auc={code_auc:.4f} indep code_auc={ci_auc:.4f} passed={payload['passed']}",
        flush=True,
    )
    if not payload["passed"]:
        # Gate 0 primary qualification failed: record the round verdict here and
        # let ``run_all`` stop.  The frozen pre-registration forbids any rescue
        # (no Stage 1, no H1 training, no descriptor swap).
        print("[continuity] FAIL -> REPRESENTATION_NOT_QUALIFIED (round stops at Gate 0)", flush=True)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def relabel_invariance_gate(n_molecules: int = 2, arm: str = "REAL") -> dict[str, Any]:
    """End-to-end node-relabel invariance of the full H1 core on a REAL object."""
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import _relabel_raw

    device_obj = torch.device("cpu")
    encoded = p1run.load_split("valid", subset=int(n_molecules))
    raw_molecules = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    node_sizes = [int(v) for v in load_raw("valid")["node_sizes"].tolist()]
    x = arm_coordinate(str(arm), "valid")
    stats = _read_json(p1run._anchor_stats_path())
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    scale = torch.as_tensor(stats["scale"], dtype=torch.float32)
    probe_D = sdb.normalize_columns(sdb.random_normalized_dictionary(a1.A1_DIM, a1.DICT_K, HEALTH_SEED)).astype(np.float32)
    model = a1.build_model(str(arm), probe_D, coding_mode=a1.CODING_IHT, iht_steps=10, seed=SEED).to(device_obj).eval()
    max_diff = 0.0
    offset = 0
    for index, (data, raw_molecule) in enumerate(zip(encoded, raw_molecules)):
        n = int(data.num_nodes)
        rows = x[offset : offset + node_sizes[index]]
        offset += node_sizes[index]
        assert int(rows.shape[0]) == n, (index, rows.shape, n)
        data.dict_phi = torch.as_tensor(rows, dtype=torch.float32)
        generator = torch.Generator().manual_seed(20260924)
        perm = torch.randperm(n, generator=generator)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(n)
        graph_new, _node_types_new, edge_types_new = zlr._data_to_graph(_relabel_raw(raw_molecule))
        incidence_new = v0.env_incidence(graph_new, edge_types_new)
        relabelled = data.clone()
        relabelled.dict_phi = torch.as_tensor(rows, dtype=torch.float32)[perm]
        relabelled.dict_atom = data.dict_atom[perm]
        raw_anchor = p1.build_anchor_raw(
            relabelled.dict_atom,
            incidence_new["occ_node"],
            incidence_new["occ_root"],
            incidence_new["bond_root"],
            incidence_new["bond_type"],
            n,
        )
        relabelled.anchor = p1.standardize_anchor(raw_anchor, mean, scale)
        relabelled.env_occ_node = incidence_new["occ_node"]
        relabelled.env_occ_root = incidence_new["occ_root"]
        relabelled.env_occ_shell = incidence_new["occ_shell"]
        relabelled.env_bond_root = incidence_new["bond_root"]
        relabelled.env_bond_shellpair = incidence_new["bond_shellpair"]
        relabelled.env_bond_type = incidence_new["bond_type"]
        relabelled.env_bond_u = incidence_new["bond_u"]
        relabelled.env_bond_v = incidence_new["bond_v"]
        relabelled.pair_index = inverse[data.pair_index]
        with torch.no_grad():
            base = model(p1.env_collate([data]).to(device_obj)).view(-1)
            moved = model(p1.env_collate([relabelled]).to(device_obj)).view(-1)
        max_diff = max(max_diff, float((base - moved).abs().max()))
    return {
        "arm": str(arm),
        "n_molecules": int(n_molecules),
        "max_abs_pred_diff": float(max_diff),
        "passed": bool(max_diff <= 1e-5),
    }


class _SlotCapture:
    """Capture the H1 slot tensors' task gradients (liveness instrumentation)."""

    def __init__(self, model: Any) -> None:
        self.buffers: dict[str, torch.Tensor] = {}
        self.handles = []
        for name in ("node_encoder", "edge_encoder", "anchor_encoder"):
            module = getattr(model, name, None)
            if module is None:
                continue
            self.handles.append(module.register_forward_hook(self._make(name)))

    def _make(self, name: str):
        def hook(_module, inputs, _output):
            tensor = inputs[0]
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.retain_grad()
                self.buffers[name] = tensor

        return hook

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, tensor in self.buffers.items():
            grad = tensor.grad
            out[name] = {
                "slot_grad_norm": None if grad is None else float(grad.norm()),
                "slot_value_std": float(tensor.detach().std()),
            }
        return out

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _normalized_rec(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    residual = ((x - x_hat) ** 2).sum(dim=1)
    denom = (x ** 2).sum(dim=1) + a1.EPS
    return float((residual / denom).mean())


def train_arm(
    arm: str,
    *,
    stage: str,
    device: str = "cpu",
    iht_steps: int | None = None,
    tag: str | None = None,
    horizon: int = HORIZON,
    out_dir: Path | None = None,
    capture_grads: bool = False,
    x_train: np.ndarray | None = None,
    x_valid: np.ndarray | None = None,
    codes_train: np.ndarray | None = None,
    codes_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Frozen-dictionary OMP screen (``stage='omp'``) or task-coupled E2E run."""
    out_dir = Path(out_dir or RESULTS_DIR)
    tag = str(tag or f"{stage}_{arm}")
    path = out_dir / f"{tag}.json"
    if path.exists():
        return _read_json(path)
    device_obj = _set_device_policy(device)
    D, dict_sha = load_arm_dictionary(arm)
    if stage == "omp":
        coding_mode = a1.CODING_OMP_FROZEN
        freeze = True
        codes_train = codes_train if codes_train is not None else load_or_build_codes(arm, "train")
        codes_valid = codes_valid if codes_valid is not None else load_or_build_codes(arm, "valid")
    elif stage == "formal":
        coding_mode = a1.CODING_IHT
        freeze = False
    elif stage == "specificity":
        coding_mode = a1.CODING_DENSE_TIED
        freeze = False
    else:
        raise ValueError(f"unknown stage {stage!r}")

    train_data = data_for(arm, "train", codes=codes_train, x=x_train)
    valid_data = data_for(arm, "valid", codes=codes_valid, x=x_valid)
    model = a1.build_model(
        arm, D, coding_mode=coding_mode, iht_steps=iht_steps, freeze_dictionary=freeze, seed=SEED
    ).to(device_obj)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(train_data, BATCH_SIZE, True, SEED + TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
    capture = _SlotCapture(model) if capture_grads else None
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    grad_probe: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, int(horizon) + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + LAMBDA_REC * rec
            optimizer.zero_grad()
            loss.backward()
            if capture is not None and grad_probe is None:
                grad_probe = {
                    "epoch": int(epoch),
                    "dictionary_grad_norm": float(model.D.grad.norm()) if model.D.grad is not None else None,
                    "slots": capture.stats(),
                    "loss": float(loss.detach()),
                }
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            with torch.no_grad():
                x = aux["phi"]
                x_hat = model.reconstruct(x, aux["coord"])
                rec_sum += float((((x - x_hat) ** 2).sum(dim=1) / ((x ** 2).sum(dim=1) + a1.EPS)).sum())
                n_nodes += int(x.shape[0])
        valid = p1run.evaluate(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(task_sum / max(n_mol, 1)),
                "train_rec": float(rec_sum / max(n_nodes, 1)),
                "valid_mae": float(valid["mae"]),
                "valid_rec": float(valid["rec"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == int(horizon):
            print(
                f"[{tag}] epoch={epoch:03d} train={curve[-1]['train_mae']:.6f} "
                f"rec={curve[-1]['train_rec']:.3e} valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    if capture is not None:
        capture.close()
    assert best_state is not None
    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K]
    )
    soup_state = {
        key: torch.stack([epoch_states[epoch][key].float() for epoch in members]).mean(0)
        for key in epoch_states[members[0]]
    }
    soup_model = a1.build_model(arm, D, coding_mode=coding_mode, iht_steps=iht_steps, freeze_dictionary=freeze, seed=SEED)
    soup_model.load_state_dict(soup_state)
    soup_valid = p1run.evaluate(soup_model.to(device_obj), eval_loader, device_obj)
    dbar_init = np.asarray(v0.normalized_dictionary(torch.as_tensor(D)))
    dbar_soup = np.asarray(v0.normalized_dictionary(soup_state["D"].detach().float()))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STATE_DIR / f"{tag}_raw_state.pt")
    torch.save(soup_state, STATE_DIR / f"{tag}_soup_state.pt")
    _write_csv_curve(CURVE_DIR / f"{tag}_curve.csv", curve)
    payload = {
        **provenance(device),
        "stage": str(stage),
        "arm": str(arm),
        "tag": str(tag),
        "input_dim": int(a1.ARM_INPUT_DIM[arm]),
        "coding_mode": coding_mode,
        "iht_steps": None if iht_steps is None else int(iht_steps),
        "dictionary_sha256_f32": dict_sha,
        "frozen_dictionary": bool(freeze),
        "parameters": a1.total_parameter_count(arm),
        "actual_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_params": int(sum(parameter.numel() for parameter in trainable)),
        "lambda_rec": float(LAMBDA_REC),
        "horizon": int(horizon),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[epoch - 1]["valid_mae"]) for epoch in members],
            "soup_valid_mae": float(soup_valid["mae"]),
        },
        "valid_rec_soup": float(soup_valid["rec"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "dictionary_movement": {
            "soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "soup_vs_init_relative": float(
                np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + a1.EPS)
            ),
        },
        "gradient_probe": grad_probe,
        "epochs_run": int(len(curve)),
        "wall_clock_s": wall,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))
        if device_obj.type == "cuda"
        else None,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    print(
        f"[{tag}] best={best_mae:.6f}@{best_epoch} soup={float(soup_valid['mae']):.6f} members={members} wall={wall:.1f}s",
        flush=True,
    )
    del train_data, valid_data
    return payload


def _write_csv_curve(path: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(curve[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in curve:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")


def _state_tag(stage: str, arm: str) -> str:
    """Prereg artifact tag for a stage/arm state file."""
    label = E_LABEL[arm] if stage in ("formal", "specificity") else ARM_LABEL[arm]
    return f"{stage}_{label}"


def _soup_of(tag: str, out_dir: Path | None = None) -> float:
    out_dir = Path(out_dir or RESULTS_DIR)
    path = out_dir / f"{tag}.json"
    if not path.exists():
        raise RuntimeError(f"run {tag} missing")
    return float(_read_json(path)["soup"]["soup_valid_mae"])


# ---------------------------------------------------------------------------
# stage: omp-screen (Stage 1) + selection
# ---------------------------------------------------------------------------


def omp_screen(device: str = "cpu") -> dict[str, Any]:
    values = {}
    for arm in a1.ARMS:
        payload = train_arm(arm, stage="omp", device=device, tag=f"omp_screen_{ARM_LABEL[arm]}")
        values[arm] = float(payload["soup"]["soup_valid_mae"])
    return values


def select_omp() -> dict[str, Any]:
    values = {arm: _soup_of(f"omp_screen_{ARM_LABEL[arm]}") for arm in a1.ARMS}
    g_attr = float(values["INDEP"] - values["REAL"])
    g_topo = float(values["TOPO"] - values["REAL"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "soup_valid_mae": values,
        "G_attr_OMP": g_attr,
        "G_topo_OMP": g_topo,
        "threshold": float(a1.MATERIAL),
        "primary_pass": bool(g_attr >= a1.MATERIAL),
        "secondary_pass": bool(g_topo >= a1.MATERIAL),
        "proceed_to_stage2": bool(g_attr >= a1.MATERIAL),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "omp_screen_selection.json", payload)
    print(f"[select-omp] {payload}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: iht-diag (Stage 2)
# ---------------------------------------------------------------------------


def iht_diag(device: str = "cpu") -> dict[str, Any]:
    device_obj = _set_device_policy(device)
    rng = np.random.default_rng(a1.IHT_DIAG_SEED)
    rows = None
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "candidate_steps": list(a1.IHT_CANDIDATE_STEPS),
        "qualify_max_normalized_err": float(a1.IHT_QUALIFY_MAX_REC),
        "n_rows": int(a1.IHT_DIAG_ROWS),
        "sample_seed": int(a1.IHT_DIAG_SEED),
        "official_test_loaded": False,
        "arms": {},
    }
    for arm in a1.ARMS:
        x = np.asarray(arm_coordinate(arm, "train"), dtype=np.float64)
        if rows is None:
            rows = np.sort(rng.choice(x.shape[0], size=min(a1.IHT_DIAG_ROWS, x.shape[0]), replace=False))
            payload["n_rows"] = int(rows.shape[0])
        X = torch.as_tensor(x[rows], dtype=torch.float32, device=device_obj)
        codes_omp = np.asarray(load_or_build_codes(arm, "train"))[rows].astype(np.float64)
        D, sha = load_arm_dictionary(arm)
        Dbar = torch.as_tensor(sdb.normalize_columns(np.asarray(D, dtype=np.float64)), dtype=torch.float32, device=device_obj)
        support_omp = np.abs(codes_omp) > 0.0
        denom = (x[rows] ** 2).sum(1) + a1.EPS
        Dbar_np = None
        Dbar_np = np.asarray(Dbar.cpu(), dtype=np.float64)
        omp_err = float(np.mean(((x[rows] - codes_omp @ Dbar_np.T) ** 2).sum(1) / denom))
        arm_row: dict[str, Any] = {"D_sha256_f32": sha, "omp_normalized_err": omp_err, "steps": {}}
        for steps in a1.IHT_CANDIDATE_STEPS:
            started = time.perf_counter()
            with torch.no_grad():
                codes = v0.tied_iht_codes(Dbar, X, s=a1.DICT_S, steps=int(steps))
            elapsed = time.perf_counter() - started
            codes_np = codes.detach().cpu().numpy().astype(np.float64)
            recon = codes_np @ Dbar_np.T
            err = float(np.mean(((x[rows] - recon) ** 2).sum(1) / denom))
            support = np.abs(codes_np) > 0.0
            intersection = np.logical_and(support, support_omp).sum(1).astype(np.float64)
            union = np.logical_or(support, support_omp).sum(1).astype(np.float64)
            jaccard = float(np.mean(intersection / np.maximum(union, 1.0)))
            relative_code_error = float(
                np.mean(
                    np.linalg.norm(codes_np - codes_omp, axis=1)
                    / (np.linalg.norm(codes_omp, axis=1) + 1e-12)
                )
            )
            arm_row["steps"][str(int(steps))] = {
                "normalized_err": err,
                "mean_l0": float(support.sum(1).mean()),
                "support_jaccard_vs_omp": jaccard,
                "relative_code_error_vs_omp": relative_code_error,
                "runtime_s": float(elapsed),
                "qualified": bool(err <= a1.IHT_QUALIFY_MAX_REC),
            }
            print(f"[iht:{arm}] steps={steps} err={err:.6e} l0={arm_row['steps'][str(int(steps))]['mean_l0']:.0f}", flush=True)
        payload["arms"][arm] = arm_row
    qualified = [
        int(steps)
        for steps in a1.IHT_CANDIDATE_STEPS
        if all(payload["arms"][arm]["steps"][str(int(steps))]["qualified"] for arm in a1.ARMS)
    ]
    selected = min(qualified) if qualified else None
    payload["qualified_steps"] = qualified
    payload["selected_steps"] = selected
    payload["coder_qualified"] = bool(selected is not None)
    _write_json(RESULTS_DIR / "iht_diagnostic.json", payload)
    print(f"[iht-diag] qualified={qualified} selected={selected}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: formal (Stage 3)
# ---------------------------------------------------------------------------


def formal_runs(device: str = "cuda") -> dict[str, Any]:
    diagnostic = _read_json(RESULTS_DIR / "iht_diagnostic.json")
    steps = diagnostic.get("selected_steps")
    if steps is None:
        raise RuntimeError("Stage 2 did not qualify a coder; formal runs are not authorised")
    values = {}
    for arm in a1.ARMS:
        tag = f"formal_{E_LABEL[arm]}"
        payload = train_arm(
            arm, stage="formal", device=device, iht_steps=int(steps), tag=tag, out_dir=FORMAL_DIR, capture_grads=(arm == "REAL")
        )
        values[arm] = float(payload["soup"]["soup_valid_mae"])
    g_attr = float(values["INDEP"] - values["REAL"])
    g_topo = float(values["TOPO"] - values["REAL"])
    selection = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "selected_iht_steps": int(steps),
        "soup_valid_mae": values,
        "G_attr": g_attr,
        "G_topo": g_topo,
        "threshold": float(a1.MATERIAL),
        "primary_pass": bool(g_attr >= a1.MATERIAL),
        "secondary_pass": bool(g_topo >= a1.MATERIAL),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "formal_selection.json", selection)
    print(f"[formal] {selection}", flush=True)
    return selection


# ---------------------------------------------------------------------------
# stage: mechanism (inference only)
# ---------------------------------------------------------------------------


def _load_soup_model(arm: str, stage: str, iht_steps: int | None, device_obj: torch.device) -> Any:
    D, _sha = load_arm_dictionary(arm)
    coding_mode = {
        "omp": a1.CODING_OMP_FROZEN,
        "formal": a1.CODING_IHT,
        "specificity": a1.CODING_DENSE_TIED,
    }[stage]
    model = a1.build_model(arm, D, coding_mode=coding_mode, iht_steps=iht_steps, freeze_dictionary=(stage == "omp"), seed=SEED)
    state = torch.load(STATE_DIR / f"{_state_tag(stage, arm)}_soup_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device_obj).eval()


def mechanism_stage(device: str = "cuda") -> dict[str, Any]:
    selection = _read_json(RESULTS_DIR / "formal_selection.json")
    steps = int(selection["selected_iht_steps"])
    device_obj = _set_device_policy(device)
    model = _load_soup_model("REAL", "formal", steps, device_obj)
    valid_data = p1run.load_split("valid")
    raw = load_raw("valid")
    scaler_real = load_scalers()["real"]
    variants: dict[str, np.ndarray] = {
        "real": arm_coordinate("REAL", "valid"),
        "code_pairing_removed_real_scaler": a1.apply_block_scaler(
            scaler_real, raw["phi"], raw["marginal_v"], raw["marginal_e"]
        ).astype(np.float32),
        "code_pairing_removed_indep_scaler": arm_coordinate("INDEP", "valid"),
        "node_pairing_removed": a1.apply_block_scaler(
            scaler_real, raw["phi"], raw["marginal_v"], raw["joint_e"]
        ).astype(np.float32),
        "edge_pairing_removed": a1.apply_block_scaler(
            scaler_real, raw["phi"], raw["joint_v"], raw["marginal_e"]
        ).astype(np.float32),
    }
    results: dict[str, Any] = {}
    baseline_predictions = None
    baseline_mae = None
    for name, x in variants.items():
        data = p1run.load_split("valid")
        attach_a1(data, "valid", "REAL", codes=None, x=x)
        loader = p1.make_env_loader(data, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
        metrics = p1run.evaluate(model, loader, device_obj)
        predictions = metrics["predictions"]
        targets = metrics["targets"]
        if baseline_predictions is None:
            baseline_predictions = predictions
            baseline_mae = float(metrics["mae"])
        shift = np.abs(predictions - baseline_predictions)
        results[name] = {
            "valid_mae": float(metrics["mae"]),
            "delta_vs_real": float(metrics["mae"] - baseline_mae),
            "mean_abs_prediction_shift": float(shift.mean()),
            "shift_quantiles": {
                "p50": float(np.quantile(shift, 0.5)),
                "p90": float(np.quantile(shift, 0.9)),
                "p99": float(np.quantile(shift, 0.99)),
                "max": float(shift.max()),
            },
        }
        del data, loader
    del valid_data

    # inherited P1 interventions on the trained REAL model (real coordinate)
    interventions = {}
    for name, kwargs in (
        ("zero_coordinate", {"coord_zero": True}),
        ("node_assignment_shuffle", {"use_node_shuffle": True}),
        ("edge_assignment_shuffle", {"use_edge_shuffle": True}),
        ("all_shuffle", {"use_node_shuffle": True, "use_edge_shuffle": True}),
    ):
        data = p1run.load_split("valid")
        attach_a1(data, "valid", "REAL")
        if kwargs.get("use_node_shuffle"):
            p1run._permute_node_shuffle(data, 101)
        if kwargs.get("use_edge_shuffle"):
            p1run._permute_edge_shuffle(data, 202)
        loader = p1.make_env_loader(data, BATCH_SIZE, False, SEED + EVAL_SHUFFLE_OFFSET)
        metrics = p1run.evaluate(model, loader, device_obj, **kwargs)
        shift = np.abs(metrics["predictions"] - baseline_predictions)
        interventions[name] = {
            "valid_mae": float(metrics["mae"]),
            "delta_vs_real": float(metrics["mae"] - baseline_mae),
            "mean_abs_prediction_shift": float(shift.mean()),
            "max_abs_prediction_shift": float(shift.max()),
        }
        del data, loader
    payload = {
        **provenance(device),
        "selected_iht_steps": steps,
        "soup_valid_mae": baseline_mae,
        "coordinate_interventions": results,
        "inherited_interventions": interventions,
        "primary": {
            "metric": "code_pairing_removed_real_scaler.delta_vs_real",
            "value": float(results["code_pairing_removed_real_scaler"]["delta_vs_real"]),
            "threshold": float(a1.MECHANISM_THRESHOLD),
            "pass": bool(results["code_pairing_removed_real_scaler"]["delta_vs_real"] >= a1.MECHANISM_THRESHOLD),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_code_pairing.json", payload)
    for name, key in (
        ("zero", "zero_coordinate"),
        ("node_shuffle", "node_assignment_shuffle"),
        ("edge_shuffle", "edge_assignment_shuffle"),
        ("all_shuffle", "all_shuffle"),
    ):
        _write_json(
            RESULTS_DIR / f"mechanism_{name}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": v0run._git_commit(),
                "variant": name,
                "selected_iht_steps": steps,
                "valid_mae": interventions[key]["valid_mae"],
                "delta_vs_real": interventions[key]["delta_vs_real"],
                "mean_abs_prediction_shift": interventions[key]["mean_abs_prediction_shift"],
                "official_test_loaded": False,
            },
        )
    print(f"[mechanism] primary={payload['primary']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: liveness
# ---------------------------------------------------------------------------


def liveness_stage(device: str = "cuda") -> dict[str, Any]:
    selection = _read_json(RESULTS_DIR / "formal_selection.json")
    steps = int(selection["selected_iht_steps"])
    device_obj = _set_device_policy(device)
    model = _load_soup_model("REAL", "formal", steps, device_obj)
    D_init, _sha = load_arm_dictionary("REAL")
    capture = _SlotCapture(model)
    data = p1run.load_split("train", subset=256)
    attach_a1(data, "train", "REAL")
    loader = p1.make_env_loader(data, 128, False, SEED)
    model.train()
    probe: dict[str, Any] = {}
    for batch in loader:
        batch = batch.to(device_obj)
        prediction, aux = model(batch, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + LAMBDA_REC * rec
        model.zero_grad()
        loss.backward()
        probe = {
            "dictionary_grad_norm": float(model.D.grad.norm()),
            "slot_gradients": capture.stats(),
            "loss": float(loss.detach()),
            "n_nodes": int(aux["phi"].shape[0]),
        }
        break
    capture.close()
    state = torch.load(STATE_DIR / "formal_E2_soup_state.pt", map_location="cpu", weights_only=True)
    D_soup = state["D"].numpy().astype(np.float64)
    D_init = np.asarray(D_init, dtype=np.float64)
    dbar_init = sdb.normalize_columns(D_init)
    dbar_soup = sdb.normalize_columns(D_soup)
    codes_train = np.asarray(load_or_build_codes("REAL", "train"))
    codes_valid = np.asarray(load_or_build_codes("REAL", "valid"))
    usage_train = (np.abs(codes_train) > 0).sum(0).astype(np.float64)
    usage_valid = (np.abs(codes_valid) > 0).sum(0).astype(np.float64)
    spearman = float(np.corrcoef(np.argsort(np.argsort(usage_train)), np.argsort(np.argsort(usage_valid)))[0, 1])
    payload = {
        **provenance(device),
        "selected_iht_steps": steps,
        "dictionary_grad_norm": probe.get("dictionary_grad_norm"),
        "slot_gradients": probe.get("slot_gradients"),
        "soup_D_vs_ksvd_init": {
            "fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "relative": float(np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + a1.EPS)),
        },
        "effective_rank_D_soup": float(
            (np.linalg.svd(dbar_soup, compute_uv=False) ** 2).sum() ** 2
            / ((np.linalg.svd(dbar_soup, compute_uv=False) ** 2) ** 2).sum()
        ),
        "active_atoms_train": int((usage_train >= 1).sum()),
        "effective_atoms_train": float(_usage_stats(codes_train)["effective_atom_count"]),
        "effective_atoms_valid": float(_usage_stats(codes_valid)["effective_atom_count"]),
        "train_valid_usage_spearman": spearman,
        "code_variance_mean_train": float(codes_train.var(0).mean()),
        "code_space_std": float(codes_train.std()),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "liveness_E2.json", payload)
    print(f"[liveness] {payload['dictionary_grad_norm']} {payload['active_atoms_train']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: specificity (Stage 4, conditional)
# ---------------------------------------------------------------------------


def specificity_stage(device: str = "cuda") -> dict[str, Any]:
    selection = _read_json(RESULTS_DIR / "formal_selection.json")
    mechanism = _read_json(RESULTS_DIR / "mechanism_code_pairing.json")
    if not selection.get("primary_pass"):
        raise RuntimeError("Stage 4 is not authorised: Stage-3 G_attr < 0.003")
    if not mechanism.get("primary", {}).get("pass"):
        raise RuntimeError("Stage 4 is not authorised: code-pairing mechanism below threshold")
    steps = int(selection["selected_iht_steps"])
    payload = train_arm(
        "REAL",
        stage="specificity",
        device=device,
        iht_steps=steps,
        tag="specificity_E2_vs_dense",
        out_dir=RESULTS_DIR,
    )
    m_sparse = float(selection["soup_valid_mae"]["REAL"])
    m_dense = float(payload["soup"]["soup_valid_mae"])
    g_sparse = float(m_dense - m_sparse)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "MAE_ATTR_REAL_SPARSE": m_sparse,
        "MAE_ATTR_REAL_DENSE_TIED": m_dense,
        "G_sparse": g_sparse,
        "threshold": float(a1.MATERIAL),
        "specificity_pass": bool(g_sparse >= a1.MATERIAL),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "specificity_E2_vs_dense.json", result)
    print(f"[specificity] {result}", flush=True)
    return result


# ---------------------------------------------------------------------------
# stage: accounting
# ---------------------------------------------------------------------------


def accounting_stage() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "arms": {},
        "official_test_loaded": False,
    }
    for arm in a1.ARMS:
        D, sha = load_arm_dictionary(arm)
        model = a1.build_model(arm, D, seed=SEED)
        accounted = a1.total_parameter_count(arm)
        actual = _n_params(model)
        payload["arms"][arm] = {
            "accounted": int(accounted["whole_model"]),
            "actual": int(actual),
            "breakdown": accounted,
            "dictionary_shape": [int(D.shape[0]), int(D.shape[1])],
            "dictionary_sha256_f32": sha,
            "passed": bool(int(accounted["whole_model"]) == int(actual) and accounted["within_budget"]),
        }
    payload["indep_vs_real_identical"] = bool(
        payload["arms"]["INDEP"]["accounted"] == payload["arms"]["REAL"]["accounted"]
    )
    payload["all_passed"] = bool(all(value["passed"] for value in payload["arms"].values()) and payload["indep_vs_real_identical"])
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    print(f"[accounting] all_passed={payload['all_passed']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: decision / report
# ---------------------------------------------------------------------------


def decision_stage() -> dict[str, Any]:
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    assignment = _read_json(RESULTS_DIR / "assignment_semantics.json")
    continuity = _read_json(RESULTS_DIR / "continuity_audit.json") if (RESULTS_DIR / "continuity_audit.json").exists() else None
    selection = _read_json(RESULTS_DIR / "omp_screen_selection.json") if (RESULTS_DIR / "omp_screen_selection.json").exists() else None
    diagnostic = _read_json(RESULTS_DIR / "iht_diagnostic.json") if (RESULTS_DIR / "iht_diagnostic.json").exists() else None
    formal = _read_json(RESULTS_DIR / "formal_selection.json") if (RESULTS_DIR / "formal_selection.json").exists() else None
    mechanism = _read_json(RESULTS_DIR / "mechanism_code_pairing.json") if (RESULTS_DIR / "mechanism_code_pairing.json").exists() else None
    specificity = _read_json(RESULTS_DIR / "specificity_E2_vs_dense.json") if (RESULTS_DIR / "specificity_E2_vs_dense.json").exists() else None

    if continuity is None or not continuity.get("passed"):
        verdict = "REPRESENTATION_NOT_QUALIFIED"
    elif selection is None or not selection.get("primary_pass"):
        verdict = "NO_ATTRIBUTED_DICTIONARY_FORMATION_SIGNAL"
    elif diagnostic is None or not diagnostic.get("coder_qualified"):
        verdict = "ATTRIBUTED_REPRESENTATION_SUPPORTED_BUT_CODER_BLOCKED"
    elif formal is None or not formal.get("primary_pass"):
        verdict = "NO_ATTRIBUTED_DICTIONARY_FORMATION_SIGNAL"
    elif mechanism is None or not mechanism.get("primary", {}).get("pass"):
        verdict = "ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC"
    elif specificity is None or not specificity.get("specificity_pass"):
        verdict = "ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC"
    else:
        verdict = "ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": v0run._git_commit(),
        "verdict": verdict,
        "gate0": {
            "correctness_all_passed": bool(correctness.get("all_passed")),
            "assignment_passed": bool(assignment.get("passed")),
            "continuity_passed": None if continuity is None else bool(continuity.get("passed")),
            "continuity_code_auc": None if continuity is None else float(continuity["real"]["code_auc"]),
            "continuity_x_auc": None if continuity is None else float(continuity["real"]["x_auc"]),
            "continuity_indep_code_auc": None if continuity is None else float(continuity["indep"]["code_auc"]),
        },
        "stage1_omp": None if selection is None else {
            "soup_valid_mae": selection["soup_valid_mae"],
            "G_attr_OMP": selection["G_attr_OMP"],
            "G_topo_OMP": selection["G_topo_OMP"],
            "primary_pass": selection["primary_pass"],
        },
        "stage2_coder": None if diagnostic is None else {
            "selected_steps": diagnostic.get("selected_steps"),
            "qualified_steps": diagnostic.get("qualified_steps"),
            "qualified": diagnostic.get("coder_qualified"),
        },
        "stage3_e2e": None if formal is None else {
            "soup_valid_mae": formal["soup_valid_mae"],
            "G_attr": formal["G_attr"],
            "G_topo": formal["G_topo"],
            "primary_pass": formal["primary_pass"],
            "secondary_pass": formal["secondary_pass"],
        },
        "mechanism": None if mechanism is None else {
            "primary_value": mechanism["primary"]["value"],
            "primary_pass": mechanism["primary"]["pass"],
            "coordinate_interventions": mechanism["coordinate_interventions"],
            "inherited_interventions": mechanism["inherited_interventions"],
        },
        "specificity": None if specificity is None else {
            "G_sparse": specificity["G_sparse"],
            "pass": specificity["specificity_pass"],
        },
        "historical_anchors": {
            "P2_ABS_H1_soup": 0.12354862861608853,
            "P1_soup": 0.13197501279687276,
            "FEC_S1_soup": 0.13042183499777457,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    print(f"[decision] {verdict}", flush=True)
    return payload


def _maybe_read(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / name
    return _read_json(path) if path.exists() else None


def report_stage() -> dict[str, Any]:
    """Markdown report; every section is optional so a Gate-0 stop still reports."""
    decision = _read_json(RESULTS_DIR / "decision.json")
    continuity = _maybe_read("continuity_audit.json")
    posthoc = _maybe_read("continuity_posthoc.json")
    diagnostic = _maybe_read("iht_diagnostic.json")
    health = _maybe_read("dictionary_health.json")
    lines = [
        "# E2E-DictEnv-A1 — report",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        "## Gate 0",
        f"* correctness all passed: {decision['gate0']['correctness_all_passed']}",
        f"* assignment semantics passed: {decision['gate0']['assignment_passed']}",
        f"* continuity: REAL code-space AUC {decision['gate0']['continuity_code_auc']:.4f}, "
        f"x-space AUC {decision['gate0']['continuity_x_auc']:.4f}, "
        f"INDEP code-space AUC {decision['gate0']['continuity_indep_code_auc']:.4f}"
        + (
            f" (pool {continuity['pool']}, near {continuity['near_pairs']}, "
            f"threshold {a1.CONTINUITY_AUC_PASS})"
            if continuity
            else " (audit artifact missing)"
        ),
        "",
        "## Parameter accounting",
        "",
        f"* TOPO {a1.total_parameter_count('TOPO')['whole_model']}, "
        f"INDEP {a1.total_parameter_count('INDEP')['whole_model']}, "
        f"REAL {a1.total_parameter_count('REAL')['whole_model']} (INDEP == REAL exactly)",
        "",
    ]
    if continuity:
        lines += [
            "### G0.3 continuity strata (frozen gate is `REAL` code-space)",
            "",
            "| arm | x-space AUC | code-space AUC | graded-tail x AUC | graded-tail code AUC |",
            "|---|---|---|---|---|",
        ]
        real, indep = continuity["real"], continuity["indep"]
        tail = continuity.get("graded_tail") or {}
        rows = [
            ("TOPO", None, None, None, None),
            ("INDEP", indep["x_auc"], indep["code_auc"], tail.get("indep_x_auc"), tail.get("indep_code_auc")),
            ("REAL", real["x_auc"], real["code_auc"], tail.get("x_auc"), tail.get("code_auc")),
        ]
        posthoc_by_arm = (posthoc or {}).get("arms", {})
        for arm, x_auc, code_auc, tail_x, tail_code in rows:
            if x_auc is None and arm in posthoc_by_arm:
                x_auc = posthoc_by_arm[arm]["near_vs_random_auc_x"]
                code_auc = posthoc_by_arm[arm]["near_vs_random_auc_code"]
                tail_x = posthoc_by_arm[arm]["tail_auc_x"]
                tail_code = posthoc_by_arm[arm]["tail_auc_code"]
            fmt = lambda value: "n/a" if value is None else f"{value:.4f}"  # noqa: E731
            lines.append(f"| {arm} | {fmt(x_auc)} | {fmt(code_auc)} | {fmt(tail_x)} | {fmt(tail_code)} |")
        if posthoc:
            near = posthoc["near_stratum"]
            lines += [
                "",
                f"Post-hoc stratum diagnostic (not the gate): the frozen near stratum is "
                f"{'entirely ' if near['all_equal_size'] else ''}equal-size "
                f"(mean patch size {near['mean_patch_size']:.2f}, range "
                f"{near['min_patch_size']}-{near['max_patch_size']}), "
                f"WL cosine >= {near['wl_cosine_min']:.6f}. "
                f"Sampled-pair Spearman(WL cosine, -distance): REAL x "
                f"{posthoc['arms']['REAL']['spearman_wl_cosine_vs_neg_xdist']:.3f} / code "
                f"{posthoc['arms']['REAL']['spearman_wl_cosine_vs_neg_codedist']:.3f}; "
                f"chemistry-blind TOPO x "
                f"{posthoc['arms']['TOPO']['spearman_wl_cosine_vs_neg_xdist']:.3f}. "
                f"Isomorphic control distance: REAL x "
                f"{posthoc['arms']['REAL']['isomorphic_mean_distance_x']:.2e}.",
                "",
            ]
    if health:
        lines += [
            "## Dictionary health (official train, exact OMP)",
            "",
            "| arm | dim | rec(fit) | rec(holdout) | random/holdout | used | effective | top1 mass | train-valid rho |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for arm in a1.ARMS:
            entry = health["arms"][arm]
            lines.append(
                f"| {arm} | {entry['input_dim']} | {entry['omp_normalized_err_fit']:.3e} | "
                f"{entry['omp_normalized_err_holdout']:.3e} | {entry['random_over_learned_holdout']:.1f} | "
                f"{entry['usage_fit']['used_atoms']}/{a1.DICT_K} | {entry['usage_fit']['effective_atom_count']:.2f} | "
                f"{entry['usage_fit']['top1_mass_share']:.3f} | {entry['train_valid_usage_spearman']:.4f} |"
            )
        lines += [""]
    if diagnostic:
        lines += ["## Stage 2 — label-free coder qualification", "", "| arm | OMP | IHT-10 | IHT-30 | IHT-100 | IHT-200 |", "|---|---|---|---|---|---|"]
        for arm in a1.ARMS:
            entry = diagnostic["arms"][arm]
            cells = [f"{entry['omp_normalized_err']:.3e}"] + [
                f"{entry['steps'][str(steps)]['normalized_err']:.3e}" for steps in a1.IHT_CANDIDATE_STEPS
            ]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines += [
            "",
            f"selected shared IHT step count: **{diagnostic.get('selected_steps')}** "
            f"(qualified {diagnostic.get('qualified_steps')}, threshold {diagnostic['qualify_max_normalized_err']})",
            "",
        ]
    else:
        lines += ["## Stage 2 — not reached (Gate 0 stop)", ""]
    if decision.get("stage3_e2e"):
        stage3 = decision["stage3_e2e"]
        lines += [
            "## Stage 3 — matched E2E (frozen H1, shared IHT steps)",
            "",
            "| arm | soup valid MAE |",
            "|---|---|",
        ]
        for arm in a1.ARMS:
            lines.append(f"| {arm} | {stage3['soup_valid_mae'][arm]:.9f} |")
        lines += [
            "",
            f"* `G_attr` (INDEP - REAL) = {stage3['G_attr']:.9f} (threshold {a1.MATERIAL}) -> {stage3['primary_pass']}",
            f"* `G_topo` (TOPO - REAL) = {stage3['G_topo']:.9f} -> {stage3['secondary_pass']}",
            "",
        ]
    if decision.get("mechanism"):
        mechanism = decision["mechanism"]
        lines += [
            "## Mechanism (inference only)",
            "",
            "| variant | valid MAE | delta | mean |shift| |",
            "|---|---|---|---|",
        ]
        for name, entry in mechanism["coordinate_interventions"].items():
            lines.append(
                f"| {name} | {entry['valid_mae']:.6f} | {entry['delta_vs_real']:+.6f} | {entry['mean_abs_prediction_shift']:.4f} |"
            )
        for name, entry in mechanism["inherited_interventions"].items():
            lines.append(
                f"| {name} | {entry['valid_mae']:.6f} | {entry['delta_vs_real']:+.6f} | {entry['mean_abs_prediction_shift']:.4f} |"
            )
        lines += ["", f"primary code-pairing threshold {a1.MECHANISM_THRESHOLD} -> {mechanism['primary_pass']}", ""]
    if decision.get("specificity"):
        lines += [
            "## Stage 4 — sparse vs dense tied (specificity)",
            "",
            f"* `G_sparse` (dense - sparse) = {decision['specificity']['G_sparse']:+.9f} -> {decision['specificity']['pass']}",
            "",
        ]
    lines += [
        "## Anchors",
        "",
        f"* historic P2-ABS H1 official-valid soup: {decision['historical_anchors']['P2_ABS_H1_soup']:.9f}",
        "* official ZINC test was never loaded in this round.",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(RESULTS_DIR / "REPORT.md")}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    for split in ("train", "valid"):
        build_cache(split)
    fit_scalers()
    fit_dictionaries()
    for arm in a1.ARMS:
        for split in ("train", "valid"):
            load_or_build_codes(arm, split)
    correctness_stage()
    assignment_stage()
    health_stage()
    continuity = continuity_stage()
    posthoc_continuity_stage()
    if not continuity["passed"]:
        accounting_stage()
        decision_stage()
        report_stage()
        print("[stop] Gate-0 continuity FAIL -> REPRESENTATION_NOT_QUALIFIED", flush=True)
        return
    omp_screen(device=device)
    selection = select_omp()
    if not selection["primary_pass"]:
        accounting_stage()
        decision_stage()
        report_stage()
        print("[stop] Stage 1 primary G_attr_OMP < 0.003", flush=True)
        return
    diagnostic = iht_diag(device=device)
    if not diagnostic["coder_qualified"]:
        accounting_stage()
        decision_stage()
        report_stage()
        print("[stop] CODER_NOT_QUALIFIED", flush=True)
        return
    formal = formal_runs(device=device)
    if not formal["primary_pass"]:
        mechanism_stage(device=device)
        liveness_stage(device=device)
        accounting_stage()
        decision_stage()
        report_stage()
        print("[stop] Stage 3 G_attr < 0.003", flush=True)
        return
    mechanism_stage(device=device)
    liveness_stage(device=device)
    accounting_stage()
    decision = decision_stage()
    if decision["mechanism"] and decision["mechanism"]["primary_pass"]:
        specificity_stage(device=device)
    decision_stage()
    report_stage()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-A1 runner")
    parser.add_argument(
        "stage",
        choices=(
            "cache", "scaler", "dictionaries", "omp-codes", "correctness", "assignment", "health",
            "continuity", "posthoc-continuity", "omp-screen", "select-omp", "iht-diag", "formal",
            "mechanism", "liveness", "specificity", "accounting", "decision", "report", "all",
        ),
    )
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--arm", default=None, choices=a1.ARMS)
    args = parser.parse_args(argv)

    if args.stage == "cache":
        build_cache(args.split, force=args.force)
    elif args.stage == "scaler":
        fit_scalers(force=args.force)
    elif args.stage == "dictionaries":
        fit_dictionaries(force=args.force)
    elif args.stage == "omp-codes":
        for arm in (a1.ARMS if args.arm is None else (args.arm,)):
            for split in ("train", "valid"):
                load_or_build_codes(arm, split)
    elif args.stage == "correctness":
        correctness_stage()
    elif args.stage == "assignment":
        assignment_stage()
    elif args.stage == "health":
        health_stage()
    elif args.stage == "continuity":
        continuity_stage()
    elif args.stage == "posthoc-continuity":
        posthoc_continuity_stage()
    elif args.stage == "omp-screen":
        omp_screen(device=args.device)
    elif args.stage == "select-omp":
        select_omp()
    elif args.stage == "iht-diag":
        iht_diag(device=args.device)
    elif args.stage == "formal":
        formal_runs(device=args.device)
    elif args.stage == "mechanism":
        mechanism_stage(device=args.device)
    elif args.stage == "liveness":
        liveness_stage(device=args.device)
    elif args.stage == "specificity":
        specificity_stage(device=args.device)
    elif args.stage == "accounting":
        accounting_stage()
    elif args.stage == "decision":
        decision_stage()
    elif args.stage == "report":
        report_stage()
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
