"""E2E-DictEnv-RoleCorr-v1 stage runner (ZINC, CPU).

Implements ``tracks/ksvd/notes/e2e_dictenv_rolecorr_v1_preregistration.md``:

1. build the frozen 536-D role<->attribute correspondence cache for the
   official train / valid splits (topology + attributes only; never ``y``);
2. fit the train-only node/edge block scaler;
3. run the zero-training construction audit (invariance, permutation
   non-triviality, zero groups, label-free, non-zero fractions);
4. fit the unlabeled structural residual dictionary and the correspondence
   dictionary (existing K-SVD pipeline, frozen budget);
5. model correctness gates (coordinate layout, freezing, shared readout init,
   zero-``alpha_C`` purity, official-test blocker);
6. smoke trainability;
7. train arms A (``TOPO``) and B (``CORR``) with the frozen Sem108 protocol;
8. frozen inference probes (``alpha_C`` zero, correspondence shuffle);
9. conditionally train controls C (``CORR-SHUF``) and D (``CORR-PCA``);
10. analysis: report + decision.

The official ZINC **test** split is never instantiated; the frozen P1 loader
refuses ``split == "test"`` and the control plane blocks test access for
non-terminal modes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_v1 as rc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as a1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = rc.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_rolecorr_v1"
CACHE_DIR = RESULTS_DIR / "cache"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_rolecorr_v1_preregistration.md"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
COMMON_SUBSPACE_PATH = (
    TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
)

THREADS = 8
SEED = 0
TRAIN_EPOCHS = 320
SMOKE_EPOCHS = 8
SMOKE_TRAIN_SUBSET = 1024
SMOKE_VALID_SUBSET = 512
BATCH_SIZE = int(p2run.BATCH_SIZE)
SHUFFLE_SEEDS = rc.SHUFFLE_SEEDS
CONTROL_SHUFFLE_SEED = int(SHUFFLE_SEEDS[0])
RELABEL_TOL = 1.0e-9
STRUCT_DICT_NAME = "struct"
CORR_DICT_NAME = "corr"
CORR_DICT_SHUF_NAME = "corr_shuffled"

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_git_commit = v0run._git_commit


def configure(*, epochs: int | None = None, threads: int | None = None) -> dict[str, Any]:
    """Fix the two knobs the runner exposes (epoch budget / torch threads).

    Sparsity, K, K-SVD budget, scaler, PCA rank, optimiser, soup rule and the
    screening gate are frozen in the pre-registration and are **not**
    configurable.
    """
    global TRAIN_EPOCHS, THREADS
    if epochs is not None:
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        TRAIN_EPOCHS = int(epochs)
    if threads is not None:
        if int(threads) <= 0:
            raise ValueError("threads must be positive")
        THREADS = int(threads)
    return {"epochs": int(TRAIN_EPOCHS), "threads": int(THREADS)}


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CACHE_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# shared data helpers
# ---------------------------------------------------------------------------


def _env_blob(split: str) -> dict[str, Any]:
    path = p1run._env_cache_path(split)
    if not path.exists():
        raise RuntimeError(f"environment cache missing for {split}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_subspace() -> cssd.CommonSubspace:
    payload = _read_json(COMMON_SUBSPACE_PATH)
    entry = payload["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(entry["components"], dtype=np.float64),
        rms=np.asarray(entry["rms"], dtype=np.float64),
        kind=str(entry["kind"]),
    )


def train_phi() -> np.ndarray:
    return _env_blob("train")["phi"].numpy().astype(np.float64)


def train_residual() -> np.ndarray:
    subspace = load_subspace()
    X = train_phi()
    c = X @ subspace.components
    return X - c @ subspace.components.T


def sdb_dictionary() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return np.asarray(dictionary, dtype=np.float32)


def _raw_corr_path(split: str, seed: int | None) -> Path:
    if seed is None:
        return CACHE_DIR / f"corr_raw_{split}.pt"
    return CACHE_DIR / f"corr_shuf_{split}_{int(seed)}.pt"


_RAW_MEMORY: dict[str, dict[str, np.ndarray]] = {}
_SCALED_MEMORY: dict[tuple[str, int | None], np.ndarray] = {}


def load_raw_corr(split: str, seed: int | None = None) -> dict[str, np.ndarray]:
    key = f"{split}:{seed}"
    if key in _RAW_MEMORY:
        return _RAW_MEMORY[key]
    path = _raw_corr_path(split, seed)
    if not path.exists():
        raise RuntimeError(f"correspondence cache missing: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {name: blob[name].numpy() for name in ("node", "edge", "group_sizes", "node_sizes")}
    _RAW_MEMORY[key] = out
    return out


def load_scaler() -> rc.CorrScaler:
    path = RESULTS_DIR / "standardizers.json"
    if not path.exists():
        raise RuntimeError("standardizers.json missing; run the `scaler` stage first")
    return rc.CorrScaler.from_json(_read_json(path))


def scaled_corr(split: str, seed: int | None = None) -> np.ndarray:
    key = (split, seed)
    if key in _SCALED_MEMORY:
        return _SCALED_MEMORY[key]
    raw = load_raw_corr(split, seed)
    scaled = rc.apply_corr_scaler(load_scaler(), raw["node"], raw["edge"]).astype(np.float32)
    _SCALED_MEMORY[key] = scaled
    return scaled


def attach_corr(data_list: Sequence[Any], split: str, seed: int | None = None) -> None:
    """Attach the scaled correspondence vector per patch to a (prefix) split subset."""
    values = scaled_corr(split, seed)
    node_sizes = [int(v) for v in _env_blob(split)["node_sizes"].tolist()]
    if len(data_list) > len(node_sizes):
        raise RuntimeError(
            f"attach_corr got {len(data_list)} molecules but {split} has {len(node_sizes)}"
        )
    expected = int(sum(node_sizes[: len(data_list)]))
    if expected > int(values.shape[0]):
        raise RuntimeError(
            f"attach_corr needs {expected} rows but {split} cache has {values.shape[0]}"
        )
    offset = 0
    for index, data in enumerate(data_list):
        size = node_sizes[index]
        data.corr_vec = torch.as_tensor(
            values[offset : offset + size], dtype=torch.float32
        ).clone()
        offset += size
    if offset != expected:
        raise RuntimeError(f"attach_corr consumed {offset} rows != {expected}")


# ---------------------------------------------------------------------------
# stage: cache
# ---------------------------------------------------------------------------


def build_corr_cache(split: str, force: bool = False) -> dict[str, Any]:
    """Build the raw 536-D object cache for one split (topology + attributes)."""
    path = _raw_corr_path(split, None)
    meta_path = RESULTS_DIR / f"cache_meta_{split}.json"
    if path.exists() and meta_path.exists() and not force:
        return _read_json(meta_path)
    _ensure_dirs()
    blob = _env_blob(split)
    node_sizes = [int(v) for v in blob["node_sizes"].tolist()]
    occ_sizes = [int(v) for v in blob["occ_sizes"].tolist()]
    occ_root = blob["occ_root"].numpy()
    raw_molecules = list(zlr._load_zinc(ZINC_ROOT, "val" if split == "valid" else "train"))
    if len(raw_molecules) != len(node_sizes):
        raise RuntimeError(f"{split}: raw={len(raw_molecules)} cache={len(node_sizes)}")
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    occ_offset = 0
    started = time.perf_counter()
    for index, raw in enumerate(raw_molecules):
        graph, node_types, edge_types = zlr._data_to_graph(raw)
        n = int(graph.n)
        if n != node_sizes[index] or n != int(raw.num_nodes):
            raise RuntimeError(f"{split}[{index}]: patch-count mismatch")
        occ_size = int(occ_sizes[index])
        roots = occ_root[occ_offset : occ_offset + occ_size]
        if (
            roots.size == 0
            or int(roots[0]) != 0
            or int(roots[-1]) != n - 1
            or np.any(np.diff(roots) < 0)
            or np.unique(roots).size != n
        ):
            raise RuntimeError(f"{split}[{index}]: occurrence root grouping mismatch")
        occ_offset += occ_size
        for root in range(n):
            blocks = rc.corr_blocks(a1.patch_blocks(graph, root, node_types, edge_types))
            node_parts.append(blocks.node.astype(np.float32))
            edge_parts.append(blocks.edge.astype(np.float32))
            group_parts.append(blocks.group_size_vector())
        if index % 1000 == 0:
            print(f"[corr-cache:{split}] {index}/{len(raw_molecules)}", flush=True)
    payload = {
        "node": torch.as_tensor(np.stack(node_parts, axis=0), dtype=torch.float32),
        "edge": torch.as_tensor(np.stack(edge_parts, axis=0), dtype=torch.float32),
        "group_sizes": torch.as_tensor(np.stack(group_parts, axis=0), dtype=torch.int64),
        "node_sizes": torch.as_tensor(node_sizes, dtype=torch.long),
    }
    torch.save(payload, path)
    report = rc.corr_block_report(payload["node"].numpy(), payload["edge"].numpy())
    report.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "split": split,
            "n_molecules": int(len(node_sizes)),
            "n_patches": int(payload["node"].shape[0]),
            "group_size_histogram": _group_size_histogram(payload["group_sizes"].numpy()),
            "node_sha256": _sha256_array(payload["node"].numpy()),
            "edge_sha256": _sha256_array(payload["edge"].numpy()),
            "seconds": float(time.perf_counter() - started),
            "git_commit": _git_commit(),
            "official_test_loaded": False,
        }
    )
    _write_json(meta_path, report)
    print(f"[corr-cache:{split}] n_patches={report['n_patches']}", flush=True)
    return report


def _group_size_histogram(group_sizes: np.ndarray) -> dict[str, Any]:
    labels = [f"shell{s}" for s in rc.NODE_SHELLS] + [
        f"pair{p[0]}{p[1]}" for p in rc.EDGE_SHELL_PAIRS
    ]
    out: dict[str, Any] = {}
    for column, label in enumerate(labels):
        values = np.asarray(group_sizes[:, column], dtype=np.int64)
        out[label] = {
            "n_patches": int(values.size),
            "zero": int(np.sum(values == 0)),
            "one": int(np.sum(values == 1)),
            "ge2": int(np.sum(values >= 2)),
            "fraction_lt2": float(np.mean(values < 2)),
            "max": int(values.max()) if values.size else 0,
        }
    return out


def build_shuffled_cache(split: str, seed: int, force: bool = False) -> dict[str, Any]:
    """Build the within-group attribute-permuted object cache (frozen shuffle)."""
    path = _raw_corr_path(split, int(seed))
    meta_path = RESULTS_DIR / f"cache_meta_{split}_shuf{int(seed)}.json"
    if path.exists() and meta_path.exists() and not force:
        return _read_json(meta_path)
    _ensure_dirs()
    blob = _env_blob(split)
    node_sizes = [int(v) for v in blob["node_sizes"].tolist()]
    raw_molecules = list(zlr._load_zinc(ZINC_ROOT, "val" if split == "valid" else "train"))
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    started = time.perf_counter()
    for index, raw in enumerate(raw_molecules):
        graph, node_types, edge_types = zlr._data_to_graph(raw)
        n = int(graph.n)
        if n != node_sizes[index]:
            raise RuntimeError(f"{split}[{index}]: patch-count mismatch")
        for root in range(n):
            blocks = rc.corr_blocks(
                a1.patch_blocks(graph, root, node_types, edge_types),
                permute_seed=int(seed),
            )
            node_parts.append(blocks.node.astype(np.float32))
            edge_parts.append(blocks.edge.astype(np.float32))
            group_parts.append(blocks.group_size_vector())
    payload = {
        "node": torch.as_tensor(np.stack(node_parts, axis=0), dtype=torch.float32),
        "edge": torch.as_tensor(np.stack(edge_parts, axis=0), dtype=torch.float32),
        "group_sizes": torch.as_tensor(np.stack(group_parts, axis=0), dtype=torch.int64),
        "node_sizes": torch.as_tensor(node_sizes, dtype=torch.long),
    }
    torch.save(payload, path)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "shuffle_seed": int(seed),
        "n_patches": int(payload["node"].shape[0]),
        "node_sha256": _sha256_array(payload["node"].numpy()),
        "edge_sha256": _sha256_array(payload["edge"].numpy()),
        "seconds": float(time.perf_counter() - started),
        "note": "within-group attribute permutation; role/attribute multisets preserved",
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    _write_json(meta_path, report)
    print(f"[corr-cache-shuf:{split}:{seed}] n={report['n_patches']}", flush=True)
    return report


# ---------------------------------------------------------------------------
# stage: scaler
# ---------------------------------------------------------------------------


def stage_scaler(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "standardizers.json"
    if path.exists() and not force:
        return _read_json(path)
    raw = load_raw_corr("train")
    scaler = rc.fit_corr_scaler(raw["node"], raw["edge"])
    payload = scaler.to_json()
    payload["protocol_version"] = PROTOCOL_VERSION
    payload["git_commit"] = _git_commit()
    payload["report"] = rc.corr_block_report(raw["node"], raw["edge"])
    _write_json(path, payload)
    print(
        "[scaler] node w={:.6f} masked={} edge w={:.6f} masked={}".format(
            scaler.node.weight,
            scaler.node.masked_coordinates,
            scaler.edge.weight,
            scaler.edge.masked_coordinates,
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: audit
# ---------------------------------------------------------------------------


def _label_free_ast_check() -> dict[str, Any]:
    forbidden = {"y", "target", "targets", "label", "labels"}
    findings: list[str] = []
    for path, guarded in (
        (Path(rc.__file__), ("corr_blocks", "corr_vector", "fit_corr_scaler", "fit_corr_dictionary")),
        (Path(__file__), ("build_corr_cache", "build_shuffled_cache")),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in guarded:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Name) and inner.id in forbidden:
                        findings.append(f"{path.name}:{node.name}:{inner.id}")
    return {
        "guarded_functions": sorted(
            {"corr_blocks", "corr_vector", "fit_corr_scaler", "fit_corr_dictionary"}
            | {"build_corr_cache", "build_shuffled_cache"}
        ),
        "forbidden_identifiers": sorted(forbidden),
        "findings": findings,
        "passed": not findings,
    }


def _swapped_graph(edges: Sequence[tuple[int, int]], n: int) -> Any:
    """A graph with every edge inserted in the reversed endpoint order.

    ``patch_blocks`` canonicalises every lookup through ``graph.edge_key``, so
    this checks that the object depends only on the undirected pair and not on
    the stored endpoint / insertion order.
    """
    from tracks.ksvd.code.graph import from_edges

    return from_edges(int(n), [(int(b), int(a)) for a, b in edges])


def _hand_reference(patch: Mapping[str, Any]) -> np.ndarray:
    node_basis = np.asarray(patch["node_basis"], dtype=np.float64)
    edge_basis = np.asarray(patch["edge_basis"], dtype=np.float64)
    q = np.asarray(patch["q"], dtype=np.float64)
    r = np.asarray(patch["r"], dtype=np.float64)
    shells = node_basis[:, 1:4].argmax(axis=1)
    pair_ids = edge_basis[:, :6].argmax(axis=1)
    out = np.zeros(rc.CORR_DIM, dtype=np.float64)
    for position, shell in enumerate(rc.NODE_SHELLS):
        mask = shells == shell
        if int(mask.sum()) >= 2:
            role = node_basis[mask][:, 4:11]
            attribute = q[mask]
            block = (role - role.mean(0, keepdims=True)).T @ (
                attribute - attribute.mean(0, keepdims=True)
            )
            low = position * rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES
            out[low : low + rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES] = block.reshape(-1)
    for position, pair in enumerate(rc.EDGE_SHELL_PAIRS):
        mask = pair_ids == rc.ALL_SHELL_PAIRS.index(pair)
        if int(mask.sum()) >= 2:
            role = edge_basis[mask][:, 6:15]
            attribute = r[mask]
            block = (role - role.mean(0, keepdims=True)).T @ (
                attribute - attribute.mean(0, keepdims=True)
            )
            low = rc.NODE_BLOCK_DIM + position * rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES
            out[low : low + rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES] = block.reshape(-1)
    return out


def stage_audit(force: bool = False, n_molecules: int = 24) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "audit.json"
    if path.exists() and not force:
        return _read_json(path)
    rc.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    stats: dict[str, Any] = {
        "n_molecules": int(n_molecules),
        "hand_reference_patches": 0,
        "hand_reference_max_abs_error": 0.0,
        "relabel_patches": 0,
        "relabel_max_abs_error": 0.0,
        "endpoint_swap_patches": 0,
        "endpoint_swap_max_abs_error": 0.0,
        "permutation_patches": 0,
        "permutation_max_delta": 0.0,
        "permutation_nonzero_change": 0,
        "zero_group_examples": 0,
    }
    raw_molecules = list(zlr._load_zinc(ZINC_ROOT, "train"))[: int(n_molecules)]
    rng = np.random.default_rng(rc.DICT_SEED)
    for index, raw in enumerate(raw_molecules):
        graph, node_types, edge_types = zlr._data_to_graph(raw)
        edges = sorted((int(a), int(b)) for a, b in graph.edges())
        n = int(graph.n)
        permutation = np.asarray(rng.permutation(n), dtype=np.int64)
        relabelled = a1run._relabelled_molecule(raw, permutation)
        moved_graph, moved_atoms, moved_edges = zlr._data_to_graph(relabelled)
        swapped_graph = _swapped_graph(edges, n)
        swapped_edges = {
            swapped_graph.edge_key(int(a), int(b)): int(edge_types[graph.edge_key(int(a), int(b))])
            for a, b in edges
        }
        for root in range(n):
            base_patch = a1.patch_blocks(graph, root, node_types, edge_types)
            base = rc.corr_blocks(base_patch)
            reference = _hand_reference(base_patch)
            stats["hand_reference_patches"] += 1
            stats["hand_reference_max_abs_error"] = max(
                stats["hand_reference_max_abs_error"], float(np.abs(base.vector - reference).max())
            )
            moved = rc.corr_blocks(
                a1.patch_blocks(moved_graph, int(permutation[root]), moved_atoms, moved_edges)
            )
            stats["relabel_patches"] += 1
            stats["relabel_max_abs_error"] = max(
                stats["relabel_max_abs_error"], float(np.abs(base.vector - moved.vector).max())
            )
            swapped = rc.corr_blocks(
                a1.patch_blocks(swapped_graph, root, node_types, swapped_edges)
            )
            stats["endpoint_swap_patches"] += 1
            stats["endpoint_swap_max_abs_error"] = max(
                stats["endpoint_swap_max_abs_error"],
                float(np.abs(base.vector - swapped.vector).max()),
            )
            shuffled = rc.corr_blocks(base_patch, permute_seed=int(SHUFFLE_SEEDS[0]))
            delta = float(np.abs(base.vector - shuffled.vector).max())
            stats["permutation_patches"] += 1
            stats["permutation_max_delta"] = max(stats["permutation_max_delta"], delta)
            if delta > rc.PERMUTE_DELTA_FLOOR:
                stats["permutation_nonzero_change"] += 1
            if not np.array_equal(base.group_size_vector(), shuffled.group_size_vector()):
                raise RuntimeError("attribute permutation changed the group sizes")
            for position, shell in enumerate(rc.NODE_SHELLS):
                if base.node_group_sizes.get(shell, 0) < 2:
                    low = position * rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES
                    high = low + rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES
                    if float(np.abs(base.node[low:high]).max()) != 0.0:
                        raise RuntimeError("singleton/empty node group is not zero")
                    stats["zero_group_examples"] += 1
            for position, pair in enumerate(rc.EDGE_SHELL_PAIRS):
                if base.edge_group_sizes.get(pair, 0) < 2:
                    low = position * rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES
                    high = low + rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES
                    if float(np.abs(base.edge[low:high]).max()) != 0.0:
                        raise RuntimeError("singleton/empty edge group is not zero")
                    stats["zero_group_examples"] += 1

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "construction": stats,
        "cache_reports": {
            split: _read_json(RESULTS_DIR / f"cache_meta_{split}.json")
            for split in ("train", "valid")
        },
        "label_free": _label_free_ast_check(),
    }
    payload["passed"] = bool(
        stats["hand_reference_patches"] > 0
        and stats["hand_reference_max_abs_error"] <= RELABEL_TOL
        and stats["relabel_max_abs_error"] <= RELABEL_TOL
        and stats["endpoint_swap_max_abs_error"] == 0.0
        and stats["permutation_nonzero_change"] > 0
        and stats["zero_group_examples"] > 0
        and payload["label_free"]["passed"]
    )
    rc.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        "[audit] hand_err={:.2e} relabel_err={:.2e} perm_changed={} zero_groups={} passed={}".format(
            stats["hand_reference_max_abs_error"],
            stats["relabel_max_abs_error"],
            stats["permutation_nonzero_change"],
            stats["zero_group_examples"],
            payload["passed"],
        ),
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(f"construction audit failed: {stats}")
    return payload


# ---------------------------------------------------------------------------
# stage: dictionaries (unlabeled, frozen budget)
# ---------------------------------------------------------------------------


def stage_dictionaries(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    meta: dict[str, Any] = {}
    struct_path = RESULTS_DIR / f"dictionary_{STRUCT_DICT_NAME}.pt"
    struct_meta = RESULTS_DIR / f"dictionary_{STRUCT_DICT_NAME}.json"
    if struct_path.exists() and struct_meta.exists() and not force:
        meta[STRUCT_DICT_NAME] = _read_json(struct_meta)
    else:
        residual = train_residual()
        started = time.perf_counter()
        D, info = rc.fit_struct_dictionary(residual, log=print)
        torch.save({"D": torch.as_tensor(D, dtype=torch.float32)}, struct_path)
        meta[STRUCT_DICT_NAME] = {
            "name": STRUCT_DICT_NAME,
            "shape": list(D.shape),
            "atoms": rc.K_ATOMS,
            "sparsity": rc.SPARSITY,
            "ksvd_epochs": rc.DICT_EPOCHS,
            "dict_seed": rc.DICT_SEED,
            "n_fit_rows": int(residual.shape[0]),
            "input": "train residual r = (I - UU^T) phi",
            "ksvd_final_fit_mse": float(info["history"][-1]["mean_sq_err"]),
            "sha256_f32": _sha256_array(D),
            "seconds": float(time.perf_counter() - started),
            "official_test_loaded": False,
        }
        _write_json(struct_meta, meta[STRUCT_DICT_NAME])
        print(f"[dict:struct] mse={meta[STRUCT_DICT_NAME]['ksvd_final_fit_mse']:.6f}", flush=True)

    corr_path = RESULTS_DIR / f"dictionary_{CORR_DICT_NAME}.pt"
    corr_meta = RESULTS_DIR / f"dictionary_{CORR_DICT_NAME}.json"
    if corr_path.exists() and corr_meta.exists() and not force:
        meta[CORR_DICT_NAME] = _read_json(corr_meta)
    else:
        scaled = scaled_corr("train")
        started = time.perf_counter()
        D, info = rc.fit_corr_dictionary(scaled, log=print)
        torch.save({"D": torch.as_tensor(D, dtype=torch.float32)}, corr_path)
        meta[CORR_DICT_NAME] = {
            "name": CORR_DICT_NAME,
            "shape": list(D.shape),
            "atoms": rc.K_ATOMS,
            "sparsity": rc.SPARSITY,
            "ksvd_epochs": rc.DICT_EPOCHS,
            "dict_seed": rc.DICT_SEED,
            "n_fit_rows": int(scaled.shape[0]),
            "input": "train scaled C (real correspondence)",
            "ksvd_final_fit_mse": float(info["history"][-1]["mean_sq_err"]),
            "sha256_f32": _sha256_array(D),
            "seconds": float(time.perf_counter() - started),
            "official_test_loaded": False,
        }
        _write_json(corr_meta, meta[CORR_DICT_NAME])
        print(f"[dict:corr] mse={meta[CORR_DICT_NAME]['ksvd_final_fit_mse']:.6f}", flush=True)

    _write_json(
        RESULTS_DIR / "dictionaries.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            **meta,
        },
    )
    return meta


def load_dictionary(name: str) -> np.ndarray:
    path = RESULTS_DIR / f"dictionary_{name}.pt"
    if not path.exists():
        raise RuntimeError(f"dictionary {name} missing; run the `dictionaries` stage first")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return np.asarray(blob["D"].numpy(), dtype=np.float32)


# ---------------------------------------------------------------------------
# stage: correctness
# ---------------------------------------------------------------------------


def _readout_reference_state() -> dict[str, torch.Tensor]:
    """The shared readout initialisation, created exactly once from the TOPO arm."""
    path = RESULTS_DIR / "readout_reference_state.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    model = rc.build_rolecorr_model(
        arm=rc.ARM_TOPO, dictionary=sdb_dictionary(), seed=SEED, subspace=load_subspace()
    )
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key not in ("D", "D_corr")
    }
    torch.save(state, path)
    return state


def model_factory(
    arm: str,
    *,
    d_struct: np.ndarray,
    d_corr: np.ndarray | None = None,
    corr_mode: str = rc.CORR_MODE_SPARSE,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    reference_state: Mapping[str, torch.Tensor],
):
    """A ``cssd.train_cssd``-compatible factory closure for one arm."""

    def factory(dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace):
        if arm == rc.ARM_TOPO:
            return rc.build_rolecorr_model(
                arm=arm,
                dictionary=dictionary,
                seed=seed,
                subspace=subspace,
                freeze_dictionary=True,
                reference_state=reference_state,
            )
        return rc.build_rolecorr_model(
            arm=arm,
            dictionary=d_struct,
            dictionary_corr=d_corr,
            corr_mode=corr_mode,
            seed=seed,
            subspace=subspace,
            freeze_dictionary=True,
            reference_state=reference_state,
            pca_mean=pca_mean,
            pca_components=pca_components,
        )

    return factory


def _zero_struct_purity(
    model: Any, phi: torch.Tensor, corr: torch.Tensor | None = None
) -> dict[str, Any]:
    """Verify that zeroing ``alpha_S`` touches only the structural coordinates."""
    coord_full = model.code(phi, corr)
    span = (
        slice(rc.COMMON_DIM, coord_full.shape[1])
        if model.arm == rc.ARM_TOPO
        else rc.STRUCT_SLICE
    )
    model.inference_zero_struct = True
    try:
        coord_zero = model.code(phi, corr)
    finally:
        model.inference_zero_struct = False
    keep_full = torch.cat([coord_full[:, : span.start], coord_full[:, span.stop :]], dim=1)
    keep_zero = torch.cat([coord_zero[:, : span.start], coord_zero[:, span.stop :]], dim=1)
    return {
        "span": [int(span.start), int(span.stop)],
        "zeroed_slice_is_zero": bool(
            torch.equal(coord_zero[:, span], torch.zeros_like(coord_zero[:, span]))
        ),
        "other_slices_bit_identical": bool(torch.equal(keep_full, keep_zero)),
        "common_untouched": bool(
            torch.equal(coord_full[:, : span.start], coord_zero[:, : span.start])
        ),
    }


def _single_batch(split: str = "valid", subset: int = 8) -> Any:
    data = p1run.load_split(split, subset=subset)
    attach_corr(data, split)
    loader = p1.make_env_loader(data, subset, False, 0)
    return next(iter(loader))


def stage_correctness(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "correctness.json"
    if path.exists() and not force:
        return _read_json(path)
    rc.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = load_subspace()
    d_struct = load_dictionary(STRUCT_DICT_NAME)
    d_corr = load_dictionary(CORR_DICT_NAME)
    d_base = sdb_dictionary()
    reference = _readout_reference_state()
    arm_a = rc.build_rolecorr_model(
        arm=rc.ARM_TOPO,
        dictionary=d_base,
        seed=SEED,
        subspace=subspace,
        freeze_dictionary=True,
        reference_state=reference,
    )
    arm_b = rc.build_rolecorr_model(
        arm=rc.ARM_CORR,
        dictionary=d_struct,
        dictionary_corr=d_corr,
        seed=SEED,
        subspace=subspace,
        freeze_dictionary=True,
        reference_state=reference,
    )
    parent = sem.build_sem108_model(d_base, SEED, subspace)
    # the frozen dropout (p=0.05 in relation/pair/global encoders) must be off
    # for the forward-equivalence gate to be a deterministic architecture check
    arm_a.eval()
    arm_b.eval()
    parent.eval()
    state_a = arm_a.state_dict()
    state_b = arm_b.state_dict()
    state_parent = parent.state_dict()
    shared_keys = [
        key
        for key in state_a
        if key in state_b
        and key not in ("D", "D_corr")
        and tuple(state_a[key].shape) == tuple(state_b[key].shape)
    ]
    init_mismatch = [key for key in shared_keys if not torch.equal(state_a[key], state_b[key])]
    init_identical = not init_mismatch
    parent_mismatch = [
        key
        for key in state_a
        if key not in ("D", "D_corr")
        and (key not in state_parent or not torch.equal(state_a[key], state_parent[key]))
    ]
    parent_identical = not parent_mismatch
    batch = _single_batch()
    with torch.no_grad():
        coord_b = arm_b.code(batch.dict_phi, batch.corr_vec)
        coord_a = arm_a.code(batch.dict_phi)
        common_expected = (
            batch.dict_phi.double()
            @ torch.as_tensor(subspace.components, dtype=torch.float64)
        ).float() / arm_b.common_rms
        common_error = float((coord_b[:, :1] - common_expected).abs().max())
        l0_struct = int((coord_b[:, rc.STRUCT_SLICE] != 0).sum(dim=1).max())
        l0_corr = int((coord_b[:, rc.CORR_SLICE] != 0).sum(dim=1).max())
        purity = rc.zero_corr_purity(arm_b, batch.dict_phi, batch.corr_vec)
        zero_struct_a = _zero_struct_purity(arm_a, batch.dict_phi)
        zero_struct_b = _zero_struct_purity(arm_b, batch.dict_phi, batch.corr_vec)
        perturbed = batch.clone()
        perturbed.corr_vec = batch.corr_vec + 0.5 * torch.randn_like(batch.corr_vec)
        pred_base = arm_b(batch, mask=cm.C6_MASK)
        pred_perturbed = arm_b(perturbed, mask=cm.C6_MASK)
        corr_sensitive = float((pred_base - pred_perturbed).abs().max())
        pred_a = arm_a(batch, mask=cm.C6_MASK)
        pred_parent = parent(batch, mask=cm.C6_MASK)
        topo_forward_error = float((pred_a - pred_parent).abs().max())
        coord_detached = arm_b.code(batch.dict_phi, batch.corr_vec)
        rec = arm_b.reconstruction_loss(batch.dict_phi, coord_detached)
        rec_grad_inert = bool(not rec.requires_grad)
        struct_sparsity_ok = bool(l0_struct <= rc.SPARSITY)
        corr_sparsity_ok = bool(l0_corr <= rc.SPARSITY)
    frozen = bool(
        not arm_a.D.requires_grad
        and not arm_b.D.requires_grad
        and (arm_b.D_corr is None or not arm_b.D_corr.requires_grad)
    )
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "coordinate_dim": int(coord_b.shape[1]),
        "coordinate_slices": rc.coordinate_slices(),
        "common_reconstruction_max_abs_error": float(common_error),
        "alpha_struct_max_l0": int(l0_struct),
        "alpha_corr_max_l0": int(l0_corr),
        "arm_a_coordinate_dim": int(coord_a.shape[1]),
        "struct_sparsity_ok": struct_sparsity_ok,
        "corr_sparsity_ok": corr_sparsity_ok,
        "zero_corr_purity": purity,
        "zero_struct_purity_arm_a": zero_struct_a,
        "zero_struct_purity_arm_b": zero_struct_b,
        "corr_perturbation_max_pred_shift": float(corr_sensitive),
        "corr_perturbation_sensitive": bool(corr_sensitive > 1e-6),
        "topo_arm_vs_sem108_forward_max_abs_error": float(topo_forward_error),
        "topo_arm_matches_sem108": bool(topo_forward_error == 0.0),
        "readout_init": {
            "shared_keys": int(len(shared_keys)),
            "bit_identical": bool(init_identical),
            "mismatch_keys": init_mismatch,
            "topo_matches_parent_init": bool(parent_identical),
            "parent_mismatch_keys": parent_mismatch,
            "trainable_parameters_B": int(
                sum(p.numel() for p in arm_b.parameters() if p.requires_grad)
            ),
            "trainable_parameters_A": int(
                sum(p.numel() for p in arm_a.parameters() if p.requires_grad)
            ),
        },
        "dictionaries_frozen": frozen,
        "reconstruction_term_gradient_inert": bool(rec_grad_inert),
        "coordinate_payload_arm_a": rc.coordinate_payload(arm_a),
        "coordinate_payload_arm_b": rc.coordinate_payload(arm_b),
    }
    payload["all_passed"] = bool(
        payload["coordinate_dim"] == rc.COORD_DIM
        and payload["arm_a_coordinate_dim"] == rc.COORD_DIM
        and payload["common_reconstruction_max_abs_error"] <= 1e-5
        and struct_sparsity_ok
        and corr_sparsity_ok
        and purity["head_columns_bit_identical"]
        and purity["corr_columns_zero"]
        and zero_struct_a["zeroed_slice_is_zero"]
        and zero_struct_a["other_slices_bit_identical"]
        and zero_struct_b["zeroed_slice_is_zero"]
        and zero_struct_b["other_slices_bit_identical"]
        and payload["corr_perturbation_sensitive"]
        and payload["topo_arm_matches_sem108"]
        and init_identical
        and parent_identical
        and frozen
        and rec_grad_inert
        and payload["readout_init"]["trainable_parameters_B"]
        == payload["readout_init"]["trainable_parameters_A"]
    )
    rc.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        f"[correctness] coord={payload['coordinate_dim']} l0=({l0_struct},{l0_corr}) "
        f"init={init_identical} topo_forward={topo_forward_error:.1e} "
        f"zero_corr={purity['corr_columns_zero']} passed={payload['all_passed']}",
        flush=True,
    )
    if not payload["all_passed"]:
        raise RuntimeError(f"correctness gates failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------


def _gradient_probe(
    arm: str,
    *,
    d_struct: np.ndarray,
    d_corr: np.ndarray | None = None,
    reference_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    train_data = p1run.load_split("train", subset=64)
    attach_corr(train_data, "train")
    loader = p1.make_env_loader(train_data, 32, False, 0)
    batch = next(iter(loader))
    model = model_factory(
        arm, d_struct=d_struct, d_corr=d_corr, reference_state=reference_state
    )(sdb_dictionary(), SEED, load_subspace())
    model.train()
    prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    rec_term = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(
        cm.H1_LAMBDA
    ) * rec_term
    model.zero_grad(set_to_none=True)
    loss.backward()
    named = dict(model.named_parameters())
    reader_keys = sorted(name for name in named if name.startswith("reader."))
    probe_keys = ["W_A_S", "W_E_S", "fusion.0.weight"] + reader_keys[:1]
    grads: dict[str, float | None] = {}
    for name in probe_keys:
        parameter = named.get(name)
        grads[name] = (
            None
            if parameter is None or parameter.grad is None
            else float(parameter.grad.norm())
        )
    grads["D"] = None if model.D.grad is None else float(model.D.grad.norm())
    if arm != rc.ARM_TOPO:
        grads["D_corr"] = (
            None
            if model.D_corr is None or model.D_corr.grad is None
            else float(model.D_corr.grad.norm())
        )
    finite = bool(torch.isfinite(loss).item() and torch.isfinite(prediction).all().item())
    return {
        "finite": finite,
        "loss": float(loss.detach()),
        "probe_keys": probe_keys,
        "reader_keys": reader_keys,
        "grad_norms": grads,
        "readout_grads_nonzero": all(
            grads[name] is not None and float(grads[name]) > 0.0 for name in probe_keys
        ),
        "dictionary_grads_absent": grads["D"] is None,
    }


def stage_smoke(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "smoke.json"
    if path.exists() and not force:
        return _read_json(path)
    rc.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    reference = _readout_reference_state()
    d_struct = load_dictionary(STRUCT_DICT_NAME)
    d_corr = load_dictionary(CORR_DICT_NAME)
    train_data = p1run.load_split("train", subset=SMOKE_TRAIN_SUBSET)
    valid_data = p1run.load_split("valid", subset=SMOKE_VALID_SUBSET)
    attach_corr(train_data, "train")
    attach_corr(valid_data, "valid")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "epochs": SMOKE_EPOCHS,
        "train_subset": SMOKE_TRAIN_SUBSET,
        "valid_subset": SMOKE_VALID_SUBSET,
        "arms": {},
    }
    for arm in (rc.ARM_TOPO, rc.ARM_CORR):
        factory = model_factory(
            arm,
            d_struct=d_struct,
            d_corr=None if arm == rc.ARM_TOPO else d_corr,
            reference_state=reference,
        )
        result = cssd.train_cssd(
            tag=f"RC-smoke-{arm}",
            epochs=SMOKE_EPOCHS,
            threads=THREADS,
            out_dir=CHECKPOINT_DIR,
            subspace=load_subspace(),
            train_data=train_data,
            valid_data=valid_data,
            seed=SEED,
            model_factory=factory,
            save_states=False,
            log=True,
        )
        probe = _gradient_probe(
            arm,
            d_struct=d_struct,
            d_corr=None if arm == rc.ARM_TOPO else d_corr,
            reference_state=reference,
        )
        payload["arms"][arm] = {
            "best_valid_mae": float(result["best_valid_mae"]),
            "epochs_run": int(result["epochs_run"]),
            "wall_clock_s": float(result["wall_clock_s"]),
            **probe,
        }
    payload["passed"] = bool(
        all(
            entry["finite"]
            and entry["readout_grads_nonzero"]
            and entry["dictionary_grads_absent"]
            and math.isfinite(entry["best_valid_mae"])
            for entry in payload["arms"].values()
        )
    )
    rc.official_test_blocker(payload)
    _write_json(path, payload)
    print(f"[smoke] passed={payload['passed']}", flush=True)
    if not payload["passed"]:
        raise RuntimeError(f"smoke failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------


def _train_arm(
    arm: str,
    *,
    d_struct: np.ndarray,
    d_corr: np.ndarray | None,
    corr_mode: str = rc.CORR_MODE_SPARSE,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    train_corr_seed: int | None = None,
    valid_corr_seed: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    epochs = int(TRAIN_EPOCHS) if epochs is None else int(epochs)
    reference = _readout_reference_state()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    attach_corr(train_data, "train", seed=train_corr_seed)
    attach_corr(valid_data, "valid", seed=valid_corr_seed)
    factory = model_factory(
        arm,
        d_struct=d_struct,
        d_corr=d_corr,
        corr_mode=corr_mode,
        pca_mean=pca_mean,
        pca_components=pca_components,
        reference_state=reference,
    )
    tag = f"RC-{arm}-seed0"
    started = time.perf_counter()
    result = cssd.train_cssd(
        tag=tag,
        epochs=int(epochs),
        threads=THREADS,
        out_dir=CHECKPOINT_DIR,
        subspace=load_subspace(),
        train_data=train_data,
        valid_data=valid_data,
        seed=SEED,
        model_factory=factory,
        save_states=True,
        log=True,
    )
    result.update(
        {
            "git_commit": _git_commit(),
            "device": "cpu",
            "threads": THREADS,
            "seed": SEED,
            "arm": arm,
            "train_corr_seed": train_corr_seed,
            "valid_corr_seed": valid_corr_seed,
            "official_test_loaded": False,
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    rc.official_test_blocker(result)
    _write_json(RESULTS_DIR / f"run_{arm}.json", result)
    _write_csv(RESULTS_DIR / f"curve_{arm}.csv", result["curve"])
    _write_json(
        RESULTS_DIR / f"soup_{arm}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "arm": arm,
            "tag": tag,
            "members": result["soup"]["members"],
            "member_valid_mae": result["soup"]["member_valid_mae"],
            "soup_valid_mae": result["soup"]["soup_valid_mae"],
            "best_valid_mae": result["best_valid_mae"],
            "best_epoch": result["best_epoch"],
            "soup_state_sha256": result.get("soup_state_sha256"),
            "best_state_sha256": result.get("best_state_sha256"),
        },
    )
    print(
        f"[train:{arm}] best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    if (
        (RESULTS_DIR / "run_CORR.json").exists()
        and (RESULTS_DIR / "run_TOPO.json").exists()
        and not force
    ):
        print("[train] cache hit", flush=True)
        return {
            arm: _read_json(RESULTS_DIR / f"run_{arm}.json")
            for arm in (rc.ARM_TOPO, rc.ARM_CORR)
        }
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not correctness.get("all_passed") or not smoke.get("passed"):
        raise RuntimeError("correctness/smoke gates not passed; refusing to train")
    train_phi()
    d_struct = load_dictionary(STRUCT_DICT_NAME)
    d_corr = load_dictionary(CORR_DICT_NAME)
    outputs: dict[str, Any] = {}
    outputs[rc.ARM_TOPO] = _train_arm(rc.ARM_TOPO, d_struct=d_struct, d_corr=None)
    outputs[rc.ARM_CORR] = _train_arm(rc.ARM_CORR, d_struct=d_struct, d_corr=d_corr)
    return outputs


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------


def _soup_model(
    arm: str,
    *,
    d_struct: np.ndarray,
    d_corr: np.ndarray | None = None,
    corr_mode: str = rc.CORR_MODE_SPARSE,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    tag: str | None = None,
) -> Any:
    reference = _readout_reference_state()
    factory = model_factory(
        arm,
        d_struct=d_struct,
        d_corr=d_corr,
        corr_mode=corr_mode,
        pca_mean=pca_mean,
        pca_components=pca_components,
        reference_state=reference,
    )
    model = factory(sdb_dictionary(), SEED, load_subspace())
    path = CHECKPOINT_DIR / f"{tag or f'RC-{arm}-seed0'}_soup_state.pt"
    if not path.exists():
        raise RuntimeError(f"soup checkpoint missing: {path}")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def evaluate_predictions(model: Any, loader: Any, mask: Any) -> dict[str, Any]:
    rc.cpu_only_guard(torch.device("cpu"))
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch, mask=mask)
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "n_molecules": int(target.size),
        "predictions": pred,
        "targets": target,
    }


def collect_codes(model: Any, loader: Any) -> dict[str, np.ndarray]:
    struct: list[np.ndarray] = []
    corr: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            struct.append(model.structural_codes(batch.dict_phi).cpu().numpy())
            if model.arm != rc.ARM_TOPO:
                corr.append(model.corr_codes(batch.corr_vec).cpu().numpy())
    out = {"struct": np.concatenate(struct, axis=0)}
    out["corr"] = np.concatenate(corr, axis=0) if corr else np.zeros((0, rc.K_ATOMS))
    return out


# ---------------------------------------------------------------------------
# stage: interventions
# ---------------------------------------------------------------------------


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "interventions.json"
    if path.exists() and not force:
        return _read_json(path)
    rc.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    d_struct = load_dictionary(STRUCT_DICT_NAME)
    d_corr = load_dictionary(CORR_DICT_NAME)
    run_b = _read_json(RESULTS_DIR / "run_CORR.json")
    if int(run_b.get("epochs_run", 0)) < TRAIN_EPOCHS:
        raise RuntimeError("the CORR formal run is incomplete; refusing the frozen probes")
    model_a = _soup_model(rc.ARM_TOPO, d_struct=d_struct)
    model_b = _soup_model(rc.ARM_CORR, d_struct=d_struct, d_corr=d_corr)
    for seed in SHUFFLE_SEEDS:
        build_shuffled_cache("valid", int(seed))
    valid_data = p1run.load_split("valid")
    attach_corr(valid_data, "valid")
    loader = p1.make_env_loader(
        valid_data, BATCH_SIZE, False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK

    base_a = evaluate_predictions(model_a, loader, mask)
    base_b = evaluate_predictions(model_b, loader, mask)
    model_a.inference_zero_struct = True
    zero_a = evaluate_predictions(model_a, loader, mask)
    model_a.inference_zero_struct = False
    model_b.inference_zero_corr = True
    zero_corr = evaluate_predictions(model_b, loader, mask)
    model_b.inference_zero_corr = False
    model_b.inference_zero_struct = True
    zero_struct = evaluate_predictions(model_b, loader, mask)
    model_b.inference_zero_struct = False

    shuffle_rows: list[dict[str, Any]] = []
    for seed in SHUFFLE_SEEDS:
        attach_corr(valid_data, "valid", seed=int(seed))
        row = evaluate_predictions(model_b, loader, mask)
        shuffle_rows.append(
            {
                "seed": int(seed),
                "mae": float(row["mae"]),
                "delta_vs_M_B": float(row["mae"] - base_b["mae"]),
            }
        )
    attach_corr(valid_data, "valid")

    codes = collect_codes(model_b, loader)
    usage = {
        "struct": rc.code_usage(codes["struct"], rc.K_ATOMS),
        "corr": rc.code_usage(codes["corr"], rc.K_ATOMS),
    }
    m_a = float(base_a["mae"])
    m_b = float(base_b["mae"])
    g_c0 = float(zero_corr["mae"] - m_b)
    g_s0 = float(zero_struct["mae"] - m_b)
    deltas = [row["delta_vs_M_B"] for row in shuffle_rows]
    g_cshuf = float(np.mean(deltas))
    pair_delta = base_b["predictions"] - base_a["predictions"]
    abs_a = np.abs(base_a["targets"] - base_a["predictions"])
    abs_b = np.abs(base_b["targets"] - base_b["predictions"])
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "M_A": m_a,
        "M_B": m_b,
        "M_A0": float(zero_a["mae"]),
        "M_C0": float(zero_corr["mae"]),
        "M_S0": float(zero_struct["mae"]),
        "G_A0": float(zero_a["mae"] - m_a),
        "G_C0": g_c0,
        "G_S0": g_s0,
        "corr_shuffle_rows": shuffle_rows,
        "G_Cshuf": g_cshuf,
        "G_Cshuf_min": float(np.min(deltas)),
        "G_Cshuf_max": float(np.max(deltas)),
        "code_usage": usage,
        "per_molecule": {
            "mean_abs_error_A": float(abs_a.mean()),
            "mean_abs_error_B": float(abs_b.mean()),
            "mean_delta_B_minus_A": float(pair_delta.mean()),
            "median_delta_B_minus_A": float(np.median(pair_delta)),
            "fraction_molecules_B_better": float(np.mean(abs_b < abs_a)),
            "mean_abs_error_delta": float((abs_b - abs_a).mean()),
            "median_abs_error_delta": float(np.median(abs_b - abs_a)),
            "n_molecules": int(abs_a.size),
        },
    }
    rc.official_test_blocker(payload)
    _write_json(path, payload)
    np.savez(
        RESULTS_DIR / "per_molecule_errors.npz",
        target=base_a["targets"],
        pred_A=base_a["predictions"],
        pred_B=base_b["predictions"],
    )
    print(
        f"[interventions] M_A={m_a:.6f} M_B={m_b:.6f} G_C0={g_c0:+.6f} "
        f"G_Cshuf={g_cshuf:+.6f} G_S0={g_s0:+.6f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: controls (only after the primary gate fires)
# ---------------------------------------------------------------------------


def stage_control_objects(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    meta_path = RESULTS_DIR / "controls_objects.json"
    if meta_path.exists() and not force:
        return _read_json(meta_path)
    build_shuffled_cache("train", CONTROL_SHUFFLE_SEED)
    build_shuffled_cache("valid", CONTROL_SHUFFLE_SEED)
    scaled_shuf = scaled_corr("train", seed=CONTROL_SHUFFLE_SEED)
    started = time.perf_counter()
    D_shuf, info_shuf = rc.fit_corr_dictionary(scaled_shuf, log=print)
    torch.save(
        {"D": torch.as_tensor(D_shuf, dtype=torch.float32)},
        RESULTS_DIR / f"dictionary_{CORR_DICT_SHUF_NAME}.pt",
    )
    scaled_real = scaled_corr("train")
    started_pca = time.perf_counter()
    pca = rc.fit_pca16(scaled_real, rank=rc.K_ATOMS)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "shuffled_dictionary": {
            "name": CORR_DICT_SHUF_NAME,
            "control_shuffle_seed": int(CONTROL_SHUFFLE_SEED),
            "shape": list(D_shuf.shape),
            "ksvd_final_fit_mse": float(info_shuf["history"][-1]["mean_sq_err"]),
            "sha256_f32": _sha256_array(D_shuf),
            "seconds": float(time.perf_counter() - started),
        },
        "pca16": {
            "shape": list(pca.components.shape),
            "seconds": float(time.perf_counter() - started_pca),
            **rc.pca16_report(pca, scaled_real),
        },
    }
    torch.save(
        {"mean": pca.mean, "components": pca.components}, RESULTS_DIR / "pca16.pt"
    )
    _write_json(RESULTS_DIR / "pca16.json", pca.to_json())
    _write_json(meta_path, payload)
    print(
        f"[controls-objects] shuffled_dict_mse={info_shuf['history'][-1]['mean_sq_err']:.6f}",
        flush=True,
    )
    return payload


def load_pca16() -> Any:
    return rc.PCA16.from_json(_read_json(RESULTS_DIR / "pca16.json"))


def stage_control_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    if (
        (RESULTS_DIR / f"run_{rc.ARM_CORR_SHUF}.json").exists()
        and (RESULTS_DIR / f"run_{rc.ARM_CORR_PCA}.json").exists()
        and not force
    ):
        return {
            arm: _read_json(RESULTS_DIR / f"run_{arm}.json")
            for arm in (rc.ARM_CORR_SHUF, rc.ARM_CORR_PCA)
        }
    d_struct = load_dictionary(STRUCT_DICT_NAME)
    d_shuf = load_dictionary(CORR_DICT_SHUF_NAME)
    pca = load_pca16()
    outputs: dict[str, Any] = {}
    outputs[rc.ARM_CORR_SHUF] = _train_arm(
        rc.ARM_CORR_SHUF,
        d_struct=d_struct,
        d_corr=d_shuf,
        train_corr_seed=CONTROL_SHUFFLE_SEED,
        valid_corr_seed=CONTROL_SHUFFLE_SEED,
    )
    outputs[rc.ARM_CORR_PCA] = _train_arm(
        rc.ARM_CORR_PCA,
        d_struct=d_struct,
        d_corr=None,
        corr_mode=rc.CORR_MODE_PCA,
        pca_mean=pca.mean,
        pca_components=pca.components,
        train_corr_seed=None,
        valid_corr_seed=None,
    )
    return outputs


def stage_control_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "control_interventions.json"
    if path.exists() and not force:
        return _read_json(path)
    d_struct = load_dictionary(STRUCT_DICT_NAME)
    d_shuf = load_dictionary(CORR_DICT_SHUF_NAME)
    pca = load_pca16()
    model_c = _soup_model(rc.ARM_CORR_SHUF, d_struct=d_struct, d_corr=d_shuf)
    model_d = _soup_model(
        rc.ARM_CORR_PCA,
        d_struct=d_struct,
        d_corr=None,
        corr_mode=rc.CORR_MODE_PCA,
        pca_mean=pca.mean,
        pca_components=pca.components,
    )
    valid_real = p1run.load_split("valid")
    attach_corr(valid_real, "valid")
    loader_real = p1.make_env_loader(
        valid_real, BATCH_SIZE, False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    valid_shuf = p1run.load_split("valid")
    attach_corr(valid_shuf, "valid", seed=CONTROL_SHUFFLE_SEED)
    loader_shuf = p1.make_env_loader(
        valid_shuf, BATCH_SIZE, False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    m_c = evaluate_predictions(model_c, loader_shuf, mask)["mae"]
    m_d = evaluate_predictions(model_d, loader_real, mask)["mae"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "M_C": float(m_c),
        "M_D": float(m_d),
        "control_shuffle_seed": int(CONTROL_SHUFFLE_SEED),
        "note": "C trains and evaluates on the seed-101 within-group permuted object; D uses the train-fitted PCA16 code of the real object",
    }
    rc.official_test_blocker(payload)
    _write_json(path, payload)
    print(f"[control-interventions] M_C={m_c:.6f} M_D={m_d:.6f}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------


def _per_molecule_summary() -> dict[str, Any]:
    path = RESULTS_DIR / "per_molecule_errors.npz"
    if not path.exists():
        return {}
    blob = np.load(path)
    target, pred_a, pred_b = blob["target"], blob["pred_A"], blob["pred_B"]
    abs_a = np.abs(target - pred_a)
    abs_b = np.abs(target - pred_b)
    return {
        "mean_abs_error_A": float(abs_a.mean()),
        "mean_abs_error_B": float(abs_b.mean()),
        "fraction_molecules_B_better": float(np.mean(abs_b < abs_a)),
        "mean_abs_error_delta_B_minus_A": float((abs_b - abs_a).mean()),
        "median_abs_error_delta_B_minus_A": float(np.median(abs_b - abs_a)),
        "quantiles_delta": {
            f"p{int(q * 100)}": float(np.quantile(abs_b - abs_a, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
    }


def stage_analysis(controls: bool = False) -> dict[str, Any]:
    """Frozen analysis: numbers, answers, verdict, report, decision."""
    _ensure_dirs()
    run_a = _read_json(RESULTS_DIR / "run_TOPO.json")
    run_b = _read_json(RESULTS_DIR / "run_CORR.json")
    interventions = _read_json(RESULTS_DIR / "interventions.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    audit = _read_json(RESULTS_DIR / "audit.json")
    dictionaries = _read_json(RESULTS_DIR / "dictionaries.json")
    m_a = float(run_a["soup"]["soup_valid_mae"])
    m_b = float(run_b["soup"]["soup_valid_mae"])
    relative = (m_a - m_b) / m_a
    gate = bool(relative >= rc.SCREEN_RELATIVE_GATE)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "seed": SEED,
        "epochs": int(run_b["epoch_budget"]),
        "M_A_soup": m_a,
        "M_B_soup": m_b,
        "M_A_best": float(run_a["best_valid_mae"]),
        "M_B_best": float(run_b["best_valid_mae"]),
        "M_A_best_epoch": int(run_a["best_epoch"]),
        "M_B_best_epoch": int(run_b["best_epoch"]),
        "relative_improvement": float(relative),
        "screen_gate": float(rc.SCREEN_RELATIVE_GATE),
        "screen_gate_fired": gate,
        "mechanism_gate_directional": float(rc.GATE_MECHANISM_DIRECTIONAL),
        "mechanism_gate_clear": float(rc.GATE_MECHANISM_CLEAR),
        "interventions": {
            key: interventions[key]
            for key in (
                "M_A0",
                "M_C0",
                "M_S0",
                "G_A0",
                "G_C0",
                "G_S0",
                "G_Cshuf",
                "G_Cshuf_min",
                "G_Cshuf_max",
                "code_usage",
                "per_molecule",
                "corr_shuffle_rows",
            )
        },
        "per_molecule_summary": _per_molecule_summary(),
        "correctness_all_passed": bool(correctness["all_passed"]),
        "audit_passed": bool(audit["passed"]),
        "dictionaries": dictionaries,
        "dictionary_shapes": {
            "D_SDB_arm_A": list(run_a.get("dictionary_shape", [])) or None,
            "D_S_arm_B": list(dictionaries[STRUCT_DICT_NAME]["shape"]),
            "D_C_arm_B": list(dictionaries[CORR_DICT_NAME]["shape"]),
        },
        "soup": {
            "TOPO": dict(run_a["soup"]),
            "CORR": dict(run_b["soup"]),
        },
        "curves": {
            "TOPO": _curve_summary(run_a["curve"]),
            "CORR": _curve_summary(run_b["curve"]),
        },
    }
    control_path = RESULTS_DIR / "control_interventions.json"
    if controls or control_path.exists():
        control_interventions = _read_json(control_path)
        control_objects = _read_json(RESULTS_DIR / "controls_objects.json")
        m_c = float(control_interventions["M_C"])
        m_d = float(control_interventions["M_D"])
        g_c = float(m_c - m_b)
        g_pca = float(m_d - m_b)
        payload["controls"] = {
            "M_C": m_c,
            "M_D": m_d,
            "G_C_vs_B": g_c,
            "G_PCA_vs_B": g_pca,
            "control_shuffle_seed": int(CONTROL_SHUFFLE_SEED),
            "objects": control_objects,
        }
        payload["answers"] = {
            "correspondence_worth_keeping": _mechanism_supported(
                payload, rc.GATE_MECHANISM_DIRECTIONAL
            ),
            "correspondence_clear": _mechanism_supported(payload, rc.GATE_MECHANISM_CLEAR),
            "pca16_better": bool(m_d < m_b),
            "sparse_dictionary_advantage": bool(g_pca <= 0.0),
        }
        if not gate:
            verdict = rc.VERDICTS["no_gain"]
        elif not payload["answers"]["correspondence_worth_keeping"]:
            verdict = rc.VERDICTS["corr_not_used"]
        elif payload["answers"]["sparse_dictionary_advantage"]:
            verdict = rc.VERDICTS["supported_sparse"]
        else:
            verdict = rc.VERDICTS["supported_not_sparse"]
    else:
        verdict = rc.VERDICTS["proceed"] if gate else rc.VERDICTS["no_gain"]
    payload["verdict"] = verdict
    rc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "summary.json", payload)
    _write_report(payload, run_a, run_b)
    _write_decision(payload)
    print(
        f"[analysis] M_A={m_a:.6f} M_B={m_b:.6f} rel={relative:+.4%} gate={gate} "
        f"G_C0={float(interventions['G_C0']):+.6f} verdict={verdict}",
        flush=True,
    )
    return payload


def _curve_summary(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = np.asarray([float(row["valid_mae"]) for row in curve], dtype=np.float64)
    train = np.asarray([float(row["train_mae"]) for row in curve], dtype=np.float64)
    return {
        "epochs": int(valid.size),
        "valid_first": float(valid[0]),
        "valid_last": float(valid[-1]),
        "valid_min": float(valid.min()),
        "valid_min_epoch": int(valid.argmin()) + 1,
        "train_last": float(train[-1]),
        "valid_last20_mean": float(valid[-20:].mean()),
    }


def _write_report(summary: Mapping[str, Any], run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> None:
    m_a, m_b = float(summary["M_A_soup"]), float(summary["M_B_soup"])
    lines = [
        "# E2E-DictEnv-RoleCorr-v1 - report\n\n",
        f"- protocol: `{PROTOCOL_VERSION}` (study `zinc-context-gap`, seed 0, CPU)\n",
        f"- git commit: `{summary['git_commit']}`\n",
        "- official test loaded: `false` (never instantiated)\n",
        f"- verdict: **{summary['verdict']}**\n\n",
        "## Primary result (official valid, Top-5 soup)\n\n",
        "| arm | coordinate | best valid MAE | soup valid MAE |\n",
        "|---|---|---:|---:|\n",
        f"| A `TOPO` | frozen SDB K32/s8 + common1 (33) | {float(run_a['best_valid_mae']):.6f} | {m_a:.6f} |\n",
        f"| B `CORR` | D_S16 + D_C16 + common1 (33) | {float(run_b['best_valid_mae']):.6f} | {m_b:.6f} |\n",
        f"\n`relative improvement = (M_A - M_B) / M_A = {float(summary['relative_improvement']):+.4%}` ",
        f"(frozen screening gate `>= {float(summary['screen_gate']):.0%}`, fired: `{summary['screen_gate_fired']}`).\n\n",
        "## Curves\n\n| arm | first | last | min | min epoch | last-20 mean |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for arm in ("TOPO", "CORR"):
        row = summary["curves"][arm]
        lines.append(
            f"| {arm} | {row['valid_first']:.6f} | {row['valid_last']:.6f} | "
            f"{row['valid_min']:.6f} | {row['valid_min_epoch']} | {row['valid_last20_mean']:.6f} |\n"
        )
    inter = summary["interventions"]
    lines += [
        "\n## Frozen inference probes (soup states)\n\n",
        f"- `M_A` = {m_a:.6f}, `M_B` = {m_b:.6f}\n",
        f"- `M_A0` (A, alpha32 -> 0) = {float(inter['M_A0']):.6f} (G = {float(inter['G_A0']):+.6f})\n",
        f"- `M_C0` (B, alpha_C -> 0) = {float(inter['M_C0']):.6f} (G_C0 = {float(inter['G_C0']):+.6f})\n",
        f"- `M_S0` (B, alpha_S -> 0) = {float(inter['M_S0']):.6f} (G_S0 = {float(inter['G_S0']):+.6f}, diagnostic)\n",
        f"- `M_Cshuf` (B, within-group attribute permutation, 5 seeds) mean G = "
        f"{float(inter['G_Cshuf']):+.6f} (range {float(inter['G_Cshuf_min']):+.6f} .. "
        f"{float(inter['G_Cshuf_max']):+.6f})\n\n",
        "## Per-molecule paired difference (official valid)\n\n",
    ]
    per = summary.get("per_molecule_summary") or {}
    if per:
        lines += [
            f"- mean |err| A = {per['mean_abs_error_A']:.6f}, B = {per['mean_abs_error_B']:.6f}\n",
            f"- fraction of molecules where B is better = {per['fraction_molecules_B_better']:.4f}\n",
            f"- mean / median (|err_B| - |err_A|) = {per['mean_abs_error_delta_B_minus_A']:+.6f} / "
            f"{per['median_abs_error_delta_B_minus_A']:+.6f}\n",
        ]
    if "controls" in summary:
        ctl = summary["controls"]
        lines += [
            "\n## Controls (seed 0, identical protocol)\n\n",
            f"- `M_C` (CORR-SHUF, shuffled correspondence, refit D_C) = {ctl['M_C']:.6f} "
            f"(G vs B = {ctl['G_C_vs_B']:+.6f})\n",
            f"- `M_D` (CORR-PCA, PCA16 code of the real C) = {ctl['M_D']:.6f} "
            f"(G vs B = {ctl['G_PCA_vs_B']:+.6f})\n\n",
            "### Frozen answers\n\n",
            f"1. correspondence worth keeping: **{summary['answers']['correspondence_worth_keeping']}** "
            f"(clear: {summary['answers']['correspondence_clear']})\n",
            f"2. sparse dictionary advantage over PCA16: "
            f"**{summary['answers']['sparse_dictionary_advantage']}** "
            f"(PCA16 better: {summary['answers']['pca16_better']})\n",
        ]
    usage = inter["code_usage"]
    lines += [
        "\n## Object and code usage\n\n",
        f"- `alpha_S` active atoms {usage['struct']['active_atoms']}/{usage['struct']['atoms']}, "
        f"effective {usage['struct']['effective_atoms']:.2f}, top1 share {usage['struct']['top1_share']:.4f}\n",
        f"- `alpha_C` active atoms {usage['corr']['active_atoms']}/{usage['corr']['atoms']}, "
        f"effective {usage['corr']['effective_atoms']:.2f}, top1 share {usage['corr']['top1_share']:.4f}\n",
        "\n## Scope notes\n\n",
        "- Baseline A is a **frozen-dictionary** re-run of the Sem108 route, not a historical\n",
        "  end-to-end number; A and B share the readout initialisation bit-for-bit.\n",
        "- Single seed; this is a screening round, not a significance claim.\n",
        "- Controls C/D were only trained if the 2% screening gate fired; otherwise they are absent.\n",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def _mechanism_supported(summary: Mapping[str, Any], threshold: float) -> bool:
    inter = summary["interventions"]
    return bool(
        float(summary["controls"]["G_C_vs_B"]) >= threshold
        and float(inter["G_C0"]) >= threshold
        and float(inter["G_Cshuf"]) >= threshold
    )


def _write_decision(summary: Mapping[str, Any]) -> None:
    verdict = str(summary["verdict"])
    if verdict == rc.VERDICTS["no_gain"]:
        body = (
            "The 2% relative screening gate did not fire: the role/attribute correspondence "
            "dictionary coordinate gives no material task gain over the frozen pure-topology "
            "coordinate under the identical frozen-dictionary protocol. Per the pre-registration "
            "the round stops here; controls C/D are not trained. No width / horizon / seed / "
            "regularisation rescue is authorised; a future step needs a new pre-registration. "
            "This does not falsify every correspondence object, and does not replace the "
            "historical end-to-end Sem108 number."
        )
    elif verdict == rc.VERDICTS["proceed"]:
        body = (
            "The screening gate fired. Per the pre-registration the frozen controls C (shuffled "
            "correspondence) and D (PCA16 of the real object) are authorised in a second "
            "invocation of this runner (`--stage controls`); no hyper-parameter or seed change."
        )
    elif verdict == rc.VERDICTS["supported_sparse"]:
        body = (
            "The screening gate fired and the candidate beats both the shuffled-correspondence "
            "control and the PCA16 control under the identical protocol: the correspondence is "
            "load-bearing and the sparse dictionary shows an advantage over a dense PCA code. "
            "Single seed; a multi-seed confirmation is the next step (new pre-registration)."
        )
    elif verdict == rc.VERDICTS["supported_not_sparse"]:
        body = (
            "The screening gate fired and the correspondence is load-bearing versus the shuffled "
            "control, but the PCA16 control is at least as good as the sparse dictionary: report "
            "`object valuable, sparse-dictionary advantage not supported`. No sparse-dictionary "
            "claim may be made from this round."
        )
    elif verdict == rc.VERDICTS["corr_not_used"]:
        body = (
            "The screening gate fired numerically but the correspondence probes do not show the "
            "information is used (zeroing / shuffling alpha_C does not materially hurt): the gain "
            "is not attributable to the correspondence coordinate."
        )
    else:
        body = f"The round is `{verdict}`; repair the blocking condition and re-run under this pre-registration."
    lines = [
        "# Decision - E2E-DictEnv-RoleCorr-v1\n\n",
        f"- verdict: **{verdict}**\n",
        f"- `M_A (TOPO) = {float(summary['M_A_soup']):.9f}`, "
        f"`M_B (CORR) = {float(summary['M_B_soup']):.9f}`, "
        f"`relative = {float(summary['relative_improvement']):+.4%}`\n\n",
        body,
        "\n\n## Forbidden without a new pre-registration\n\n",
        "- seed 1 of either arm;\n",
        "- any K / sparsity / K-SVD-epoch / scaler / PCA-rank / width / horizon change;\n",
        "- end-to-end (unfrozen-dictionary) fine-tuning of either arm;\n",
        "- touching the official ZINC test split;\n",
        "- re-using these numbers as a replacement for the historical Sem108 end-to-end result.\n",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# stage: orchestration
# ---------------------------------------------------------------------------

PRIMARY_STAGES = ("cache", "scaler", "audit", "dictionaries", "correctness", "smoke", "train", "interventions", "analysis")
CONTROL_STAGES = ("control_objects", "control_train", "control_interventions", "analysis")


def run_primary(stages: Sequence[str]) -> dict[str, Any]:
    _ensure_dirs()
    print(f"[runner] commit={_git_commit()} protocol={PROTOCOL_VERSION}", flush=True)
    for split in ("train", "valid"):
        build_corr_cache(split)
    results: dict[str, Any] = {}
    if "scaler" in stages:
        results["scaler"] = stage_scaler()
    if "audit" in stages:
        results["audit"] = stage_audit()
    if "dictionaries" in stages:
        results["dictionaries"] = stage_dictionaries()
    if "correctness" in stages:
        results["correctness"] = stage_correctness()
    if "smoke" in stages:
        results["smoke"] = stage_smoke()
    if "train" in stages:
        results["train"] = stage_train()
    if "interventions" in stages:
        results["interventions"] = stage_interventions()
    if "analysis" in stages:
        results["analysis"] = stage_analysis()
    return results


def run_controls(stages: Sequence[str]) -> dict[str, Any]:
    _ensure_dirs()
    summary = _read_json(RESULTS_DIR / "summary.json")
    if str(summary.get("verdict")) != rc.VERDICTS["proceed"]:
        raise RuntimeError(
            f"controls are only authorised after a `{rc.VERDICTS['proceed']}` verdict, "
            f"found `{summary.get('verdict')}`"
        )
    results: dict[str, Any] = {}
    if "control_objects" in stages:
        results["control_objects"] = stage_control_objects()
    if "control_train" in stages:
        results["control_train"] = stage_control_train()
    if "control_interventions" in stages:
        results["control_interventions"] = stage_control_interventions()
    if "analysis" in stages:
        results["analysis"] = stage_analysis(controls=True)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-RoleCorr-v1 stage runner")
    parser.add_argument("--stage", required=True, choices=("primary", "controls", "all"))
    parser.add_argument("--only", default=None, help="comma-separated stage subset override")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    only = [part.strip() for part in args.only.split(",") if part.strip()] if args.only else None
    if args.stage in ("primary", "all"):
        stages = list(PRIMARY_STAGES)
        if only:
            stages = [stage for stage in stages if stage in set(only)]
        run_primary(stages)
    if args.stage in ("controls", "all"):
        stages = list(CONTROL_STAGES)
        if only:
            stages = [stage for stage in stages if stage in set(only)]
        run_controls(stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
