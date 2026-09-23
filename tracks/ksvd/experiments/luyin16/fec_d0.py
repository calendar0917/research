"""FEC-D0 — Within-Coarse-Role Residual Structural Dictionary Audit.

Pre-registration: ``tracks/ksvd/notes/fec_d0_preregistration.md`` (frozen).
Prior-artifact audit: ``tracks/ksvd/notes/fec_d0_prior_artifact_audit.md``.

Object: a **root-conditioned occurrence-level pure-topology residual structural
subrole** dictionary.

For every rooted radius-2 patch ``i`` of a ZINC molecule and every occurrence
``v`` in that patch:

* ``phi_iv``  = audited FSAR explicit rooted node basis ``b^V_iv in R^11``
                (``fsar_v2._explicit_basis_for_patch``; chemistry-free);
* ``s_iv``    = ``shell(i, v) in {0,1,2}`` (BFS distance from the patch root);
* delete the exact shell coordinates (``[0]`` and ``[1:4]``);
* residualize per shell with FIT-only ``mu_s`` / ``sigma_s`` -> ``phi^perp``;
* fit ONE shared ``K=16`` / ``s=4`` dictionary with the correctness-tested
  SDB/TCCD K-SVD + OMP primitives; compare against a matched random dictionary
  and a FIT-only PCA-4 reference;
* test reuse (atoms across molecules / roots / heldout) and
  marked-topology subrole semantics with a label-free probe.

This module contains only definitions so every property demanded by the
pre-registration is unit-testable without data. The runner is
``zinc_fec_d0.py``; focused CPU tests are ``tracks/ksvd/tests/test_fec_d0.py``.

``official_valid_loaded = false``; ``official_test_loaded = false``;
``targets_loaded = false``. No property training, no FEC-S1 retrain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import typed_patch_tokenizer as tpt

PROTOCOL_VERSION = "fec_d0"

# --- frozen constants (pre-registered; no sweep) ---------------------------
NODE_BASIS_DIM = int(v2.NODE_BASIS_DIM)  # 11
SHELL_ONEHOT_SLICE = slice(1, 4)  # exact shell one-hot
ROOT_INDICATOR_INDEX = 0  # exactly 1[shell == 0]
N_SHELLS = 3

#: coordinates removed before residualization = {0} u {1,2,3}
DELETED_COORDINATES = (ROOT_INDICATOR_INDEX, 1, 2, 3)
KEPT_COORDINATES = tuple(
    i for i in range(NODE_BASIS_DIM) if i not in DELETED_COORDINATES
)  # 7 coordinates: 4,5,6,7,8,9,10
D_RES = len(KEPT_COORDINATES)  # 7

K_ATOMS = 16
SPARSITY = 4
KSVD_EPOCHS = 10
PATCH_RADIUS = 2

SPLIT_SEED = 20260922
N_TOTAL = 10000
N_HOLDOUT = 2000
N_FIT = N_TOTAL - N_HOLDOUT  # 8000

DICT_SEED = 20260922
PCA_RANK = 4
SIGMA_FLOOR = 1.0e-6
NOISE_SIGMA = 1.0e-4
NOISE_SEED = 20260924

PROBE_C = 1.0
PROBE_MAX_ITER = 200
CLASS_MIN_FIT = 20
CLASS_MIN_HOLDOUT = 5

#: frozen decision gates
REC_GATE = 0.80
ACTIVE_FIT_GATE = 12
ACTIVE_HOLDOUT_GATE = 12
ACTIVE_JACCARD_GATE = 0.75
MOLECULE_COVER_GATE = 20
MOLECULE_COVER_ATOMS = 12
USAGE_SPEARMAN_GATE = 0.70
ORACLE_GAIN_GATE = 0.03
DICT_VS_RANDOM_GATE = 0.03
DICT_RETAIN_GATE = 0.70
SHELL_WORSE_SLACK = 0.02

VERDICTS = {
    "qualified": "FEC_D0_RESIDUAL_STRUCTURAL_DICTIONARY_QUALIFIED",
    "basis_unavailable": "FEC_D0_ROOTED_BASIS_UNAVAILABLE",
    "correctness": "FEC_D0_CORRECTNESS_FAILURE",
    "no_structure": "FEC_D0_NO_LEARNABLE_RESIDUAL_STRUCTURE",
    "not_reusable": "FEC_D0_DICTIONARY_NOT_REUSABLE",
    "no_signal": "FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL",
    "loses_info": "FEC_D0_DICTIONARY_LOSES_SUBROLE_INFORMATION",
}


# ---------------------------------------------------------------------------
# json helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


# ---------------------------------------------------------------------------
# target guard
# ---------------------------------------------------------------------------


class TargetAccessError(RuntimeError):
    """Raised when any code path attempts to read a target value."""


def _forbidden_target(*_args: Any, **_kwargs: Any) -> Any:
    raise TargetAccessError("FEC-D0 is label-free: target access is forbidden")


class LabelFreeZinc:
    """Wraps a PyG ZINC dataset and poisons ``.y`` and any ``val``/``test`` read.

    Iteration yields the raw ``Data`` objects of the requested split with their
    ``y`` attribute replaced by a descriptor that raises on any use.
    """

    def __init__(self, split: str, dataset: Any) -> None:
        if split != "train":
            raise RuntimeError(
                f"FEC-D0 forbids non-train splits; requested {split!r}"
            )
        self.split = str(split)
        self._dataset = dataset
        self.accessed = {"val": False, "test": False, "targets": False}

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __iter__(self):
        for data in self._dataset:
            yield _PoisonedData(data)

    def __getitem__(self, index: int):
        return _PoisonedData(self._dataset[index])


class _PoisonedData:
    """A ``Data``-like proxy whose ``y`` raises and whose tensors are readable."""

    def __init__(self, data: Any) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        if name == "y":
            return _forbidden_target()
        return getattr(object.__getattribute__(self, "_data"), name)


def assert_label_free(split: str) -> None:
    if split != "train":
        raise RuntimeError(f"FEC-D0 forbids split {split!r}; official valid/test never loaded")


# ---------------------------------------------------------------------------
# occurrence extraction (basis + shell + marked topology)
# ---------------------------------------------------------------------------


@dataclass
class OccurrenceBatch:
    """Flat occurrence table for one molecule."""

    basis: np.ndarray  # [n_occ, 11]
    shell: np.ndarray  # [n_occ]
    molecule_index: int
    root_index: np.ndarray  # [n_occ] occurrence's patch root (local node id)
    marked_key: list[bytes] = field(default_factory=list)  # [n_occ]

    def __len__(self) -> int:
        return int(self.basis.shape[0])


def _untyped_marked_incidence(
    graph: Any, center: int, marked: int, radius: int
) -> tpt.ColoredIncidence:
    """Untyped marked-node rooted incidence graph: colors carry no chemistry.

    Node color: ``("node", is_root, is_marked, distance_from_root)``.
    Edge color: ``("edge",)``.
    """
    distances = tpt.ego_distances(graph, int(center), int(radius))
    original_nodes = tuple(sorted(distances))
    node_to_local = {node: i for i, node in enumerate(original_nodes)}
    induced = graph.induced(set(original_nodes))
    local_edges = tuple(
        (node_to_local[int(left)], node_to_local[int(right)])
        for left, right in sorted(induced.edges())
    )
    n_nodes = len(original_nodes)
    n_edges = len(local_edges)
    adjacency: dict[int, list[int]] = {v: [] for v in range(n_nodes + n_edges)}
    color_of_vertex: dict[int, tuple[Any, ...]] = {}
    root_local = node_to_local[int(center)]
    marked_local = node_to_local[int(marked)]
    for local, node in enumerate(original_nodes):
        color_of_vertex[local] = (
            "node",
            int(local == root_local),
            int(local == marked_local),
            int(distances[node]),
        )
    for edge_local, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_local
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
        color_of_vertex[edge_vertex] = ("edge",)
    key_to_vertices: dict[tuple[Any, ...], list[int]] = {}
    for vertex in range(n_nodes + n_edges):
        key_to_vertices.setdefault(color_of_vertex[vertex], []).append(vertex)
    cell_keys = tuple(sorted(key_to_vertices, key=repr))
    cell_vertices = tuple(tuple(sorted(key_to_vertices[key])) for key in cell_keys)
    color_keys = tuple(color_of_vertex[v] for v in range(n_nodes + n_edges))
    return tpt.ColoredIncidence(
        number_of_vertices=n_nodes + n_edges,
        adjacency=adjacency,
        cell_keys=cell_keys,
        cell_vertices=cell_vertices,
        color_keys=color_keys,
        n_nodes=n_nodes,
        n_edges=n_edges,
        root_local=root_local,
        original_nodes=original_nodes,
    )


def marked_topology_key(
    graph: Any, center: int, marked: int, radius: int = PATCH_RADIUS
) -> bytes:
    """Exact untyped marked-node rooted topology class (complete invariant)."""
    incidence = _untyped_marked_incidence(graph, int(center), int(marked), int(radius))
    return tpt.corrected_canonical_key(incidence)


def extract_molecule(
    graph: Any,
    molecule_index: int,
    *,
    with_marked: bool = True,
    radius: int = PATCH_RADIUS,
) -> OccurrenceBatch:
    """Extract every occurrence ``(root i, node v)`` of one molecule.

    ``graph`` is an untyped graph (adjacency only); node/bond chemistry is
    never consulted here.
    """
    centers = sorted(int(node) for node in graph.nodes)
    basis_rows: list[np.ndarray] = []
    shells: list[int] = []
    roots: list[int] = []
    marked: list[bytes] = []
    for center in centers:
        nodes, index, node_basis, _edge_basis, _edges = v2._explicit_basis_for_patch(
            graph, int(center), int(radius)
        )
        for local, node in enumerate(nodes):
            basis_rows.append(node_basis[local])
            shells.append(int(node_basis[local, 1:4].argmax()))
            roots.append(int(center))
            if with_marked:
                marked.append(marked_topology_key(graph, int(center), int(node), radius))
    return OccurrenceBatch(
        basis=np.stack(basis_rows, axis=0).astype(np.float64) if basis_rows else np.zeros((0, NODE_BASIS_DIM)),
        shell=np.asarray(shells, dtype=np.int64),
        molecule_index=int(molecule_index),
        root_index=np.asarray(roots, dtype=np.int64),
        marked_key=marked,
    )


def occurrence_fingerprint(batches: Sequence[OccurrenceBatch]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(batches)).encode())
    for batch in batches:
        digest.update(f"|{batch.molecule_index}:{len(batch)}".encode())
        digest.update(batch.basis.tobytes())
        digest.update(batch.shell.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def internal_split(n_total: int = N_TOTAL, n_holdout: int = N_HOLDOUT, seed: int = SPLIT_SEED):
    """Canonical repo train-internal split (SPLIT_SEED permutation convention)."""
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(int(n_total))
    holdout = np.sort(perm[: int(n_holdout)].astype(np.int64))
    holdout_set = set(int(i) for i in holdout)
    fit = np.asarray(
        sorted(i for i in range(int(n_total)) if i not in holdout_set), dtype=np.int64
    )
    return fit, holdout


def split_fingerprint(indices: Sequence[int], batches: Mapping[int, OccurrenceBatch]) -> str:
    digest = hashlib.sha256()
    digest.update(f"n={len(indices)}".encode())
    for index in indices:
        batch = batches[int(index)]
        digest.update(f"|{int(index)}:{len(batch)}".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# residualization
# ---------------------------------------------------------------------------


@dataclass
class ShellStats:
    mean: np.ndarray  # [3, D_RES]
    std: np.ndarray  # [3, D_RES]
    counts: np.ndarray  # [3]


def fit_shell_stats(basis: np.ndarray, shell: np.ndarray) -> ShellStats:
    """FIT-only per-shell mean/std of the shell-coordinate-deleted basis."""
    reduced = np.asarray(basis, dtype=np.float64)[:, KEPT_COORDINATES]
    mean = np.zeros((N_SHELLS, D_RES), dtype=np.float64)
    std = np.zeros((N_SHELLS, D_RES), dtype=np.float64)
    counts = np.zeros(N_SHELLS, dtype=np.int64)
    for s in range(N_SHELLS):
        mask = shell == s
        counts[s] = int(mask.sum())
        if counts[s] == 0:
            std[s] = 1.0
            continue
        block = reduced[mask]
        mean[s] = block.mean(axis=0)
        std[s] = block.std(axis=0)
    return ShellStats(mean=mean, std=std, counts=counts)


def residualize(basis: np.ndarray, shell: np.ndarray, stats: ShellStats) -> np.ndarray:
    """Apply FIT statistics: ``(phi~ - mu_s) / max(sigma_s, floor)``."""
    reduced = np.asarray(basis, dtype=np.float64)[:, KEPT_COORDINATES]
    out = np.empty_like(reduced)
    for s in range(N_SHELLS):
        mask = shell == s
        if not bool(mask.any()):
            continue
        denom = np.maximum(stats.std[s], SIGMA_FLOOR)
        out[mask] = (reduced[mask] - stats.mean[s][None, :]) / denom[None, :]
    return out


# ---------------------------------------------------------------------------
# dictionaries (reuse SDB / TCCD primitives; no re-implementation)
# ---------------------------------------------------------------------------


def normalize_columns(D: np.ndarray) -> np.ndarray:
    return T.normalize_columns(np.asarray(D, dtype=np.float64))


def random_dictionary(features: int = D_RES, atoms: int = K_ATOMS, seed: int = DICT_SEED) -> np.ndarray:
    return T.random_normalized_dictionary(int(features), int(atoms), int(seed))


def omp_codes(D: np.ndarray, X: np.ndarray, s: int = SPARSITY) -> np.ndarray:
    return T.omp_codes(np.asarray(D, dtype=np.float64), np.asarray(X, dtype=np.float64), s=int(s))


def fit_dictionary(
    X: np.ndarray, atoms: int = K_ATOMS, s: int = SPARSITY, epochs: int = KSVD_EPOCHS,
    seed: int = DICT_SEED, log: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    D0 = random_dictionary(int(X.shape[1]), int(atoms), int(seed))
    return T.ksvd_fit(
        np.asarray(X, dtype=np.float64),
        D0,
        s=int(s),
        epochs=int(epochs),
        seed=int(seed),
        log=(log if log is not None else (lambda *_: None)),
    )


def fit_pca(X: np.ndarray, rank: int = PCA_RANK):
    return sdb.fit_pca_rank(np.asarray(X, dtype=np.float64), rank=int(rank))


def relative_reconstruction_error(
    D: np.ndarray, X: np.ndarray, C: np.ndarray | None = None, s: int = SPARSITY
) -> float:
    return float(
        T.relative_reconstruction_error(
            np.asarray(D, dtype=np.float64),
            np.asarray(X, dtype=np.float64),
            C=None if C is None else np.asarray(C, dtype=np.float64),
            s=int(s),
        )
    )


# ---------------------------------------------------------------------------
# reuse statistics
# ---------------------------------------------------------------------------


def code_support_stats(
    codes: np.ndarray, molecule_index: np.ndarray, root_index: np.ndarray, shell: np.ndarray
) -> dict[str, Any]:
    """Per-atom support / molecule / root / shell statistics for one split."""
    codes = np.asarray(codes)
    n_occ, atoms = codes.shape
    support = np.abs(codes) > 0
    stats: dict[str, Any] = {}
    active: list[int] = []
    per_atom: list[dict[str, Any]] = []
    shell_counts = np.zeros((atoms, N_SHELLS), dtype=np.int64)
    for k in range(atoms):
        mask = support[:, k]
        count = int(mask.sum())
        molecules = int(np.unique(molecule_index[mask]).size) if count else 0
        roots = int(np.unique(root_index[mask] * 1000003 + molecule_index[mask]).size) if count else 0
        if count:
            for s in range(N_SHELLS):
                shell_counts[k, s] = int((mask & (shell == s)).sum())
        mags = np.abs(codes[mask, k])
        per_atom.append(
            {
                "atom": k,
                "support_count": count,
                "support_frequency": count / max(n_occ, 1),
                "distinct_molecules": molecules,
                "distinct_roots": roots,
                "shell_distribution": shell_counts[k].tolist(),
                "coefficient_abs_mean": float(mags.mean()) if count else 0.0,
                "coefficient_abs_max": float(mags.max()) if count else 0.0,
            }
        )
        if count:
            active.append(k)
    stats["per_atom"] = per_atom
    stats["active_atoms"] = active
    stats["n_active"] = len(active)
    stats["n_occurrences"] = int(n_occ)
    # max single-molecule contribution per atom
    per_atom_mol: list[float] = []
    for k in range(atoms):
        mask = support[:, k]
        if not bool(mask.any()):
            per_atom_mol.append(0.0)
            continue
        contrib: dict[int, float] = {}
        for idx in np.flatnonzero(mask):
            m = int(molecule_index[idx])
            contrib[m] = contrib.get(m, 0.0) + float(abs(codes[idx, k]))
        total = sum(contrib.values())
        per_atom_mol.append(max(contrib.values()) / total if total > 0 else 0.0)
    stats["max_single_molecule_contribution"] = per_atom_mol
    return stats


def usage_distribution(stats: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["support_count"] for row in stats["per_atom"]], dtype=np.float64
    )


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n)


def usage_summary(stats: Mapping[str, Any]) -> dict[str, Any]:
    usage = usage_distribution(stats)
    total = usage.sum()
    p = usage / total if total > 0 else np.zeros_like(usage)
    order = np.sort(p)[::-1]
    return {
        "total_mass": float(total),
        "entropy_nats": _entropy(p),
        "effective_atom_count": float(np.exp(_entropy(p))),
        "top1_mass": float(order[0]) if order.size else 0.0,
        "top4_mass": float(order[:4].sum()) if order.size >= 4 else float(order.sum()),
        "gini": _gini(usage),
        "n_active": int(stats["n_active"]),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def fit_probe(
    X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=PROBE_C, max_iter=PROBE_MAX_ITER, multi_class="auto", n_jobs=None
    )
    model.fit(np.asarray(X_fit, dtype=np.float64), np.asarray(y_fit))
    return model.predict(np.asarray(X_eval, dtype=np.float64))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro"))
