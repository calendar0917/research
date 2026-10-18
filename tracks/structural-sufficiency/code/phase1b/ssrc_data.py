"""Phase 1b data: canonical molecules, frozen splits, rooted-radius-2 patch
cache and flat batching.

Scope discipline
----------------
* ZINC: only the official PyG ``ZINC(subset=True, split="train")`` 10,000 rows
  are read.  Official valid/test are never instantiated.  The internal split
  reuses the repository-frozen TCCD-v0 permutation
  ``np.random.default_rng(20260922).permutation(10000)``: first 8,000 Fit,
  the remaining 2,000 split *in permutation order* into 1,000 Monitor and
  1,000 Reserved.
* Lipophilicity: reuses the Phase 1a frozen 2352/504/504 scaffold split and
  the Fit/Monitor-only label artifact.  Reserved labels are never loaded.
* The cache contains structures only (no labels).  Labels live in a separate
  vault restricted to Fit/Monitor.

The patch cache stores one exact rooted induced radius-2 patch per atom
(Phase 1a convention: root first, remaining nodes by (distance, atom index);
node feature = raw atom one-hot + root flag + distance one-hot).  Patches are
flattened in molecule/root order; batching is vectorised with
``repeat_interleave`` + per-patch starts, never a per-batch Python loop.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE1A_CODE = Path(__file__).resolve().parents[1] / "phase1a"
if str(PHASE1A_CODE) not in sys.path:
    sys.path.insert(0, str(PHASE1A_CODE))

# Reuse the audit-frozen Lipophilicity raw-graph reader and feature convention.
from phase1a_data import (  # noqa: E402
    ATOM_BASE_DIM as LIPO_ATOM_DIM,
    BOND_FEAT_DIM as LIPO_BOND_DIM,
    atom_base_features,
    bond_features,
    load_raw_graphs,
    physical_bonds,
)

RADIUS = 2
ZINC_ATOM_DIM = 21  # derived from official train 10k (max atom type 20)
ZINC_BOND_DIM = 4
ZINC_SPLIT_SEED = 20260922
ZINC_N_TOTAL = 10_000
ZINC_N_FIT = 8_000
ZINC_N_MONITOR = 1_000
ZINC_N_RESERVED = 1_000

LIPO_RAW = (
    REPO_ROOT
    / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
)
LIPO_SPLIT = REPO_ROOT / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
LIPO_LABELS = REPO_ROOT / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

CACHE_DIR = REPO_ROOT / "tracks/structural-sufficiency/results/phase1b/cache"


# ---------------------------------------------------------------------------
# canonical molecule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolGraph:
    """Undirected attributed molecule with physical bonds only."""

    atom_feat: np.ndarray  # [N, A] float32 one-hot / raw feature block
    bond_u: np.ndarray  # [M] int64, u < v, sorted
    bond_v: np.ndarray  # [M] int64
    bond_feat: np.ndarray  # [M, B] float32
    smiles: str = ""

    @property
    def n_atoms(self) -> int:
        return int(self.atom_feat.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.bond_u.shape[0])


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_sha256(rows: Sequence[int]) -> str:
    """Phase 1a row-index hash convention: comma-separated decimal string."""
    return hashlib.sha256(",".join(str(int(r)) for r in rows).encode("utf-8")).hexdigest()


def row_sha256_bytes(rows: Sequence[int]) -> str:
    arr = np.ascontiguousarray(np.asarray(rows, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# ZINC loaders
# ---------------------------------------------------------------------------


def load_zinc_train(root: Path | str = ZINC_ROOT) -> tuple[list[MolGraph], np.ndarray, dict[str, Any]]:
    """Load the official PyG ZINC subset train split.

    Returns structures, the official train target vector, and provenance.
    The target is PyG's ``logP_SA_cycle_normalized`` (already normalized by the
    benchmark authors); no additional standardization is applied anywhere.
    """
    from torch_geometric.datasets import ZINC

    root = Path(root)
    ds = ZINC(root=str(root), subset=True, split="train")
    if len(ds) != ZINC_N_TOTAL:
        raise ValueError(f"unexpected ZINC train size {len(ds)}")
    mols: list[MolGraph] = []
    ys: list[float] = []
    for data in ds:
        atom_type = data.x.reshape(-1).long().numpy()
        if atom_type.size == 0 or atom_type.min() < 0 or atom_type.max() >= ZINC_ATOM_DIM:
            raise ValueError("unexpected ZINC atom type range")
        ei = data.edge_index.numpy()
        ea = data.edge_attr.reshape(-1).long().numpy()
        if ei.shape[1] != ea.shape[0]:
            raise ValueError("edge_index / edge_attr length mismatch")
        bond_map: dict[tuple[int, int], int] = {}
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u == v:
                raise ValueError("self-loop in ZINC graph")
            key = (min(u, v), max(u, v))
            val = int(ea[k])
            if key in bond_map and bond_map[key] != val:
                raise ValueError(f"inconsistent directed ZINC bond {key}")
            bond_map[key] = val
        keys = sorted(bond_map)
        bu = np.asarray([k[0] for k in keys], dtype=np.int64)
        bv = np.asarray([k[1] for k in keys], dtype=np.int64)
        bf = np.zeros((len(keys), ZINC_BOND_DIM), dtype=np.float32)
        if keys:
            bf[np.arange(len(keys)), [bond_map[k] for k in keys]] = 1.0
        af = np.zeros((atom_type.size, ZINC_ATOM_DIM), dtype=np.float32)
        af[np.arange(atom_type.size), atom_type] = 1.0
        mols.append(MolGraph(af, bu, bv, bf))
        ys.append(float(data.y.reshape(-1)[0]))

    raw_files = [root / "raw/train.pickle", root / "raw/train.index", root / "subset/processed/train.pt"]
    provenance = {
        "dataset": "ZINC subset=True official train",
        "n_rows": len(mols),
        "target": "PyG logP_SA_cycle_normalized (benchmark-normalized; no extra standardization)",
        "atom_vocab": ZINC_ATOM_DIM,
        "bond_vocab": ZINC_BOND_DIM,
        "raw_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in raw_files if p.exists()},
    }
    return mols, np.asarray(ys, dtype=np.float32), provenance


def zinc_internal_split(
    n_total: int = ZINC_N_TOTAL,
    seed: int = ZINC_SPLIT_SEED,
) -> dict[str, list[int]]:
    """Exact TCCD-v0 internal split, with the dev half split in permutation order."""
    if n_total != ZINC_N_TOTAL:
        raise ValueError("the frozen ZINC split is defined for 10,000 rows")
    perm = np.random.default_rng(int(seed)).permutation(int(n_total))
    fit = perm[:ZINC_N_FIT]
    dev = perm[ZINC_N_FIT:]
    monitor = dev[:ZINC_N_MONITOR]
    reserved = dev[ZINC_N_MONITOR:]
    return {
        "fit": [int(i) for i in fit],
        "monitor": [int(i) for i in monitor],
        "reserved": [int(i) for i in reserved],
    }


def write_zinc_split(path: Path, split: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    payload = {
        "protocol_id": "ssrc-phase1b-zinc-v1",
        "scope": {"labels_read": False, "structures_only": True},
        "rule": (
            "np.random.default_rng(20260922).permutation(10000); "
            "fit=perm[:8000]; dev=perm[8000:]; monitor=dev[:1000]; "
            "reserved=dev[1000:] (dev order preserved)"
        ),
        "rows": {k: [int(i) for i in v] for k, v in split.items()},
        "sha256": {k: row_sha256(v) for k, v in split.items()},
        "counts": {k: len(v) for k, v in split.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ---------------------------------------------------------------------------
# Lipophilicity loaders / frozen split
# ---------------------------------------------------------------------------


def load_lipo_mols(path: Path | str = LIPO_RAW) -> list[MolGraph]:
    raws = load_raw_graphs(path)
    mols: list[MolGraph] = []
    for raw in raws:
        af = atom_base_features(raw)
        bonds = physical_bonds(raw)
        bf = bond_features(bonds)
        bu = np.asarray([b[0] for b in bonds], dtype=np.int64)
        bv = np.asarray([b[1] for b in bonds], dtype=np.int64)
        mols.append(MolGraph(af, bu, bv, bf, smiles=raw.smiles))
    return mols


def load_lipo_split(path: Path | str = LIPO_SPLIT) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("protocol_id") != "ssrc-phase1a-lipo-v1":
        raise ValueError("unexpected Lipophilicity split protocol")
    rows = payload["rows"]
    split = {
        "fit": [int(i) for i in rows["fit"]],
        "monitor": [int(i) for i in rows["monitor"]],
        "reserved": [int(i) for i in rows["reserved"]],
    }
    if len(split["fit"]) != 2352 or len(split["monitor"]) != 504 or len(split["reserved"]) != 504:
        raise ValueError("unexpected Lipophilicity split sizes")
    if set(split["fit"]) & set(split["monitor"]) or set(split["fit"]) & set(split["reserved"]) or set(split["monitor"]) & set(split["reserved"]):
        raise ValueError("Lipophilicity split overlap")
    return {
        "split": split,
        "sha256": {
            "split_file": sha256_file(path),
            "fit": row_sha256(split["fit"]),
            "monitor": row_sha256(split["monitor"]),
            "reserved": row_sha256(split["reserved"]),
        },
        "frozen_protocol_id": payload["protocol_id"],
    }


# ---------------------------------------------------------------------------
# label vault
# ---------------------------------------------------------------------------


class LabelVault:
    """Fit/Monitor-only labels; Reserved and official valid/test are absent."""

    def __init__(self, source_path: Path | str, fit_rows: Iterable[int], monitor_rows: Iterable[int], reserved_rows: Iterable[int]):
        self.source_path = Path(source_path)
        self.fit = frozenset(int(r) for r in fit_rows)
        self.monitor = frozenset(int(r) for r in monitor_rows)
        self.reserved = frozenset(int(r) for r in reserved_rows)
        if self.fit & self.monitor or self.fit & self.reserved or self.monitor & self.reserved:
            raise ValueError("label-vault row sets overlap")
        self.allowed = self.fit | self.monitor
        self._labels: dict[int, float] = {}
        self._loaded = False

    def load(self) -> None:
        payload = json.loads(self.source_path.read_text())
        scope = payload.get("scope", {})
        if scope.get("reserved_labels_present") is not False:
            raise PermissionError("label artifact does not certify Reserved exclusion")
        if scope.get("official_valid_test_labels_present") is not False:
            raise PermissionError("label artifact does not certify official valid/test exclusion")
        raw = {int(k): float(v) for k, v in payload["labels"].items()}
        if set(raw) & self.reserved:
            raise PermissionError("Reserved labels present in artifact")
        self._labels = {k: v for k, v in raw.items() if k in self.allowed}
        del raw
        missing = self.allowed - set(self._labels)
        if missing:
            raise ValueError(f"missing allowed labels: {sorted(missing)[:5]} ...")
        if set(self._labels) & self.reserved:
            raise PermissionError("Reserved labels retained")
        self._loaded = True

    def labels(self, rows: Sequence[int], purpose: str) -> torch.Tensor:
        if not self._loaded:
            self.load()
        if purpose not in {"train", "monitor"}:
            raise PermissionError(f"label purpose {purpose!r} is not allowed")
        allowed = self.fit if purpose == "train" else self.monitor
        rows = [int(r) for r in rows]
        if not set(rows) <= allowed:
            bad = sorted(set(rows) - allowed)
            raise PermissionError(f"{purpose} requested non-allowed rows: {bad[:5]}")
        if set(rows) & self.reserved:
            raise PermissionError("Reserved labels requested")
        return torch.tensor([self._labels[r] for r in rows], dtype=torch.float32)

    def audit(self) -> dict[str, Any]:
        return {
            "fit_rows": len(self.fit),
            "monitor_rows": len(self.monitor),
            "reserved_rows": len(self.reserved),
            "allowed_rows": len(self.allowed),
            "labels_retained": len(self._labels),
            "reserved_labels_retained": len(set(self._labels) & self.reserved),
            "official_valid_test_labels_retained": 0,
            "source_kind": "json",
            "source_sha256": sha256_file(self.source_path),
        }


def write_labels_artifact(
    path: Path,
    labels: np.ndarray,
    fit_rows: Sequence[int],
    monitor_rows: Sequence[int],
    reserved_rows: Sequence[int],
    extra_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = [int(r) for r in fit_rows] + [int(r) for r in monitor_rows]
    if set(allowed) & set(int(r) for r in reserved_rows):
        raise ValueError("allowed rows overlap Reserved")
    payload = {
        "scope": {
            "fit_rows": len(fit_rows),
            "monitor_rows": len(monitor_rows),
            "reserved_rows": len(reserved_rows),
            "reserved_labels_present": False,
            "official_valid_test_labels_present": False,
            "labels_retained": len(allowed),
            **(dict(extra_scope) if extra_scope else {}),
        },
        "labels": {str(int(r)): float(labels[int(r)]) for r in allowed},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload["scope"]


# ---------------------------------------------------------------------------
# patch extraction
# ---------------------------------------------------------------------------


def extract_patch_arrays(mol: MolGraph, radius: int = RADIUS) -> dict[str, Any]:
    """Exact rooted induced radius-r patches, one per atom, in atom order."""
    n = mol.n_atoms
    atom_dim = mol.atom_feat.shape[1]
    bond_dim = mol.bond_feat.shape[1]
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k in range(mol.n_bonds):
        u, v = int(mol.bond_u[k]), int(mol.bond_v[k])
        adj[u].append((v, k))
        adj[v].append((u, k))

    node_feats: list[np.ndarray] = []
    edge_src: list[np.ndarray] = []
    edge_dst: list[np.ndarray] = []
    edge_attrs: list[np.ndarray] = []
    n_nodes: list[int] = []
    n_edges: list[int] = []
    roots: list[int] = []

    for root in range(n):
        dist = [-1] * n
        dist[root] = 0
        queue = deque([root])
        while queue:
            v = queue.popleft()
            if dist[v] == radius:
                continue
            for u, _k in adj[v]:
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        nodes = [root] + sorted(
            (v for v in range(n) if 0 < dist[v] <= radius),
            key=lambda v: (dist[v], v),
        )
        local = {v: i for i, v in enumerate(nodes)}
        nf = np.zeros((len(nodes), atom_dim + 4), dtype=np.float32)
        nf[:, :atom_dim] = mol.atom_feat[nodes]
        nf[0, atom_dim] = 1.0  # root flag
        for i, v in enumerate(nodes):
            nf[i, atom_dim + 1 + dist[v]] = 1.0  # distance one-hot 0..radius
        es: list[int] = []
        ed: list[int] = []
        ea: list[np.ndarray] = []
        for k in range(mol.n_bonds):
            u, v = int(mol.bond_u[k]), int(mol.bond_v[k])
            if u in local and v in local:
                iu, iv = local[u], local[v]
                es.extend([iu, iv])
                ed.extend([iv, iu])
                ea.extend([mol.bond_feat[k], mol.bond_feat[k]])
        node_feats.append(nf)
        n_nodes.append(len(nodes))
        n_edges.append(len(es))
        edge_src.append(np.asarray(es, dtype=np.int64))
        edge_dst.append(np.asarray(ed, dtype=np.int64))
        edge_attrs.append(np.stack(ea, axis=0) if ea else np.zeros((0, bond_dim), dtype=np.float32))
        roots.append(root)

    return {
        "node_feat": node_feats,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_attr": edge_attrs,
        "n_nodes": np.asarray(n_nodes, dtype=np.int64),
        "n_edges": np.asarray(n_edges, dtype=np.int64),
        "roots": np.asarray(roots, dtype=np.int64),
        "atom_dim": atom_dim,
        "bond_dim": bond_dim,
    }


def build_cache(mols: Sequence[MolGraph], radius: int = RADIUS) -> dict[str, torch.Tensor | dict[str, Any]]:
    """Build a compact flat patch/root cache for a list of molecules."""
    atom_dim = int(mols[0].atom_feat.shape[1])
    bond_dim = int(mols[0].bond_feat.shape[1])
    node_dim = atom_dim + 4

    patch_node_feat: list[np.ndarray] = []
    patch_edge_src: list[np.ndarray] = []
    patch_edge_dst: list[np.ndarray] = []
    patch_edge_attr: list[np.ndarray] = []
    patch_n_nodes: list[int] = []
    patch_n_edges: list[int] = []
    patch_mol: list[int] = []
    patch_root: list[int] = []
    mol_patch_start: list[int] = [0]

    atom_feat_flat: list[np.ndarray] = []
    atom_start: list[int] = [0]
    root_edge_src: list[np.ndarray] = []
    root_edge_dst: list[np.ndarray] = []
    root_edge_attr: list[np.ndarray] = []
    root_edge_start: list[int] = [0]
    root_n_edges: list[int] = []
    mol_n_atoms: list[int] = []

    for mol_id, mol in enumerate(mols):
        p = extract_patch_arrays(mol, radius=radius)
        for i in range(len(p["n_nodes"])):
            patch_node_feat.append(p["node_feat"][i])
            patch_edge_src.append(p["edge_src"][i])
            patch_edge_dst.append(p["edge_dst"][i])
            patch_edge_attr.append(p["edge_attr"][i])
            patch_n_nodes.append(int(p["n_nodes"][i]))
            patch_n_edges.append(int(p["n_edges"][i]))
            patch_mol.append(mol_id)
            patch_root.append(int(p["roots"][i]))
        mol_patch_start.append(len(patch_node_feat))

        atom_feat_flat.append(mol.atom_feat)
        atom_start.append(atom_start[-1] + mol.n_atoms)
        mol_n_atoms.append(mol.n_atoms)

        # root graph: both directed records for every physical bond
        src: list[int] = []
        dst: list[int] = []
        attr: list[np.ndarray] = []
        for k in range(mol.n_bonds):
            u, v = int(mol.bond_u[k]), int(mol.bond_v[k])
            src.extend([u, v])
            dst.extend([v, u])
            attr.extend([mol.bond_feat[k], mol.bond_feat[k]])
        root_edge_src.append(np.asarray(src, dtype=np.int64))
        root_edge_dst.append(np.asarray(dst, dtype=np.int64))
        root_edge_attr.append(np.stack(attr, axis=0) if attr else np.zeros((0, bond_dim), dtype=np.float32))
        root_n_edges.append(len(src))
        root_edge_start.append(root_edge_start[-1] + len(src))

    patch_n_nodes_arr = np.asarray(patch_n_nodes, dtype=np.int64)
    patch_n_edges_arr = np.asarray(patch_n_edges, dtype=np.int64)
    patch_node_start = np.concatenate([[0], np.cumsum(patch_n_nodes_arr)]).astype(np.int64)
    patch_edge_start = np.concatenate([[0], np.cumsum(patch_n_edges_arr)]).astype(np.int64)

    cache: dict[str, torch.Tensor | dict[str, Any]] = {
        "patch_node_feat_flat": torch.as_tensor(np.concatenate(patch_node_feat, axis=0), dtype=torch.float16),
        "patch_edge_src": torch.as_tensor(np.concatenate(patch_edge_src, axis=0), dtype=torch.int32),
        "patch_edge_dst": torch.as_tensor(np.concatenate(patch_edge_dst, axis=0), dtype=torch.int32),
        "patch_edge_attr": torch.as_tensor(np.concatenate(patch_edge_attr, axis=0), dtype=torch.float16),
        "patch_n_nodes": torch.as_tensor(patch_n_nodes_arr, dtype=torch.int64),
        "patch_n_edges": torch.as_tensor(patch_n_edges_arr, dtype=torch.int64),
        "patch_node_start": torch.as_tensor(patch_node_start, dtype=torch.int64),
        "patch_edge_start": torch.as_tensor(patch_edge_start, dtype=torch.int64),
        "patch_mol": torch.as_tensor(patch_mol, dtype=torch.int64),
        "patch_root": torch.as_tensor(patch_root, dtype=torch.int64),
        "mol_patch_start": torch.as_tensor(mol_patch_start, dtype=torch.int64),
        "atom_feat_flat": torch.as_tensor(np.concatenate(atom_feat_flat, axis=0), dtype=torch.float16),
        "atom_start": torch.as_tensor(atom_start, dtype=torch.int64),
        "mol_n_atoms": torch.as_tensor(mol_n_atoms, dtype=torch.int64),
        "root_edge_src_flat": torch.as_tensor(np.concatenate(root_edge_src, axis=0), dtype=torch.int32),
        "root_edge_dst_flat": torch.as_tensor(np.concatenate(root_edge_dst, axis=0), dtype=torch.int32),
        "root_edge_attr_flat": torch.as_tensor(np.concatenate(root_edge_attr, axis=0), dtype=torch.float16),
        "root_edge_start": torch.as_tensor(root_edge_start, dtype=torch.int64),
        "root_n_edges": torch.as_tensor(root_n_edges, dtype=torch.int64),
        "meta": {
            "radius": int(radius),
            "n_molecules": len(mols),
            "n_patches": int(len(patch_n_nodes)),
            "n_patch_nodes": int(patch_node_start[-1]),
            "n_patch_edges": int(patch_edge_start[-1]),
            "n_root_edges": int(root_edge_start[-1]),
            "atom_dim": atom_dim,
            "bond_dim": bond_dim,
            "node_dim": node_dim,
            "max_patch_nodes": int(patch_n_nodes_arr.max()) if len(patch_n_nodes_arr) else 0,
            "max_atoms": int(max(mol_n_atoms)) if mol_n_atoms else 0,
            "max_bonds": int(max(r // 2 for r in root_n_edges)) if root_n_edges else 0,
        },
    }
    return cache


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def gather_batch(
    cache: Mapping[str, Any],
    mol_indices: Sequence[int],
    y: torch.Tensor | None = None,
    include_size: bool = False,
    include_edge_mol: bool = False,
) -> dict[str, Any]:
    """Vectorised batch gather.

    Patch tokens and root-graph nodes are in the same order (one patch / one
    root atom per atom), so the token tensor can be used directly as the root
    graph node feature.

    ``include_size`` adds ``mol_size_feat`` (Phase 1c only).  The feature is
    read from ``cache['mol_size_feat']`` when present (a frozen sidecar built by
    Phase 1c prep), otherwise it is derived from the structure counts as
    ``[n_atoms/10, n_bonds/10, n_bonds - n_atoms + 1]`` for connected molecular
    graphs.  ``include_edge_mol`` adds ``root_edge_mol`` (molecule id per
    directed edge) and ``root_edge_unique`` (the canonical undirected copy) for
    the marginal-correspondence arm; no endpoint indices are added.
    """
    sel_mols = torch.as_tensor([int(i) for i in mol_indices], dtype=torch.long)
    B = int(sel_mols.numel())
    mol_n_atoms = cache["mol_n_atoms"][sel_mols].long()  # one patch per atom
    P = int(mol_n_atoms.sum())

    # selected patch ids in batch order
    patch_starts = cache["mol_patch_start"][sel_mols]
    p_off = torch.cumsum(mol_n_atoms, dim=0) - mol_n_atoms
    p_local = torch.arange(P) - torch.repeat_interleave(p_off, mol_n_atoms)
    sel_patch = torch.repeat_interleave(patch_starts, mol_n_atoms) + p_local

    # patch nodes
    counts = cache["patch_n_nodes"][sel_patch].long()
    starts = cache["patch_node_start"][sel_patch]
    node_offsets = torch.cumsum(counts, dim=0) - counts
    total_nodes = int(counts.sum())
    n_local = torch.arange(total_nodes) - torch.repeat_interleave(node_offsets, counts)
    flat_node_ids = torch.repeat_interleave(starts, counts) + n_local
    patch_node_feat = cache["patch_node_feat_flat"][flat_node_ids].float()
    patch_batch = torch.repeat_interleave(torch.arange(P, dtype=torch.long), counts)
    patch_to_mol = torch.repeat_interleave(torch.arange(B, dtype=torch.long), mol_n_atoms)

    # patch edges
    ecounts = cache["patch_n_edges"][sel_patch].long()
    estarts = cache["patch_edge_start"][sel_patch]
    edge_offsets = torch.cumsum(ecounts, dim=0) - ecounts
    total_edges = int(ecounts.sum())
    e_local = torch.arange(total_edges) - torch.repeat_interleave(edge_offsets, ecounts)
    flat_edge_ids = torch.repeat_interleave(estarts, ecounts) + e_local
    patch_edge_src = cache["patch_edge_src"][flat_edge_ids].long() + torch.repeat_interleave(node_offsets, ecounts)
    patch_edge_dst = cache["patch_edge_dst"][flat_edge_ids].long() + torch.repeat_interleave(node_offsets, ecounts)
    patch_edge_index = torch.stack([patch_edge_src, patch_edge_dst], dim=0)
    patch_edge_attr = cache["patch_edge_attr"][flat_edge_ids].float()

    # root graph edges (both directions)
    rcounts = cache["root_n_edges"][sel_mols].long()
    rstarts = cache["root_edge_start"][sel_mols]
    redge_offsets = torch.cumsum(rcounts, dim=0) - rcounts
    total_redges = int(rcounts.sum())
    re_local = torch.arange(total_redges) - torch.repeat_interleave(redge_offsets, rcounts)
    rflat = torch.repeat_interleave(rstarts, rcounts) + re_local
    root_edge_src = cache["root_edge_src_flat"][rflat].long() + torch.repeat_interleave(p_off, rcounts)
    root_edge_dst = cache["root_edge_dst_flat"][rflat].long() + torch.repeat_interleave(p_off, rcounts)
    root_edge_index = torch.stack([root_edge_src, root_edge_dst], dim=0)
    root_edge_attr = cache["root_edge_attr_flat"][rflat].float()
    root_batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long), mol_n_atoms)
    # raw atom features for the full-graph G_ref model
    astarts = cache["atom_start"][sel_mols]
    total_atoms = int(mol_n_atoms.sum())
    a_local = torch.arange(total_atoms) - torch.repeat_interleave(p_off, mol_n_atoms)
    aflat = torch.repeat_interleave(astarts, mol_n_atoms) + a_local
    atom_feat = cache["atom_feat_flat"][aflat].float()

    batch: dict[str, Any] = {
        # patch-only
        "patch_node_feat": patch_node_feat,
        "patch_edge_index": patch_edge_index,
        "patch_edge_attr": patch_edge_attr,
        "patch_batch": patch_batch,
        "patch_to_mol": patch_to_mol,
        "n_patches": P,
        # root graph
        "root_edge_index": root_edge_index,
        "root_edge_attr": root_edge_attr,
        "root_batch": root_batch,
        "atom_feat": atom_feat,
        # provenance / bookkeeping
        "mol_index": sel_mols,
        "n_molecules": B,
    }
    if include_size:
        if "mol_size_feat" in cache:
            batch["mol_size_feat"] = cache["mol_size_feat"][sel_mols].float()
        else:
            n_atoms = mol_n_atoms.float()
            n_bonds = (cache["root_n_edges"][sel_mols].float() / 2.0)
            rings = (n_bonds - n_atoms + 1.0).clamp(min=0.0)
            batch["mol_size_feat"] = torch.stack([n_atoms / 10.0, n_bonds / 10.0, rings], dim=-1)
    if include_edge_mol:
        batch["root_edge_mol"] = torch.repeat_interleave(torch.arange(B, dtype=torch.long), rcounts)
        batch["root_edge_unique"] = root_edge_src < root_edge_dst
    if y is not None:
        batch["y"] = y
    return batch


# ---------------------------------------------------------------------------
# cache IO
# ---------------------------------------------------------------------------


def save_cache(cache: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), path)


def load_cache(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "meta" not in obj:
        raise ValueError(f"bad cache {path}")
    return obj


def cache_manifest(path: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    obj = load_cache(path)
    manifest = {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "sha256": sha256_file(path),
        "meta": dict(obj["meta"]),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
