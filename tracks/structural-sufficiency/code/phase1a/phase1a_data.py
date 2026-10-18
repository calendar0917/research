"""Phase 1a data pipeline: label-free raw graphs -> exact E_2 patches.

The only structural input to S/X is the multiset of rooted radius-2 induced
patches.  Patches are represented as local node/edge tensors plus a root flag
and a distance-to-root one-hot; no original atom index, no patch-to-patch
incidence and no full-graph edge_index is passed to S/X.

The raw graph artifact is the committed label-free BJOC Lipophilicity
conversion (``dataset/lipo_raw_v1.json.gz``); RDKit is not needed on the GPU
host.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# frozen raw vocabulary (matches the committed label-free raw graph artifact)
# ---------------------------------------------------------------------------

N_ATOM_CLASSES = 15
N_ATOM_SLOTS = 17          # 15 real + UNK + MASK (MASK never appears in raw data)
N_CHARGE_SLOTS = 7         # -2,-1,0,+1,+2, UNK, MASK
N_H_SLOTS = 7              # 0..4, UNK, MASK
N_CHIRAL_SLOTS = 6         # UNSPECIFIED,CW,CCW,OTHER,UNK,MASK
N_BOND_CLASSES = 4
N_BOND_SLOTS = 6           # 4 real + UNK + MASK
N_STEREO_SLOTS = 6         # NONE,E,Z,OTHER,UNK,MASK

ATOM_BASE_DIM = N_ATOM_SLOTS + N_CHARGE_SLOTS + N_H_SLOTS + N_CHIRAL_SLOTS  # 37
PATCH_NODE_DIM = ATOM_BASE_DIM + 1 + 3  # + root flag + distance one-hot (0,1,2)
BOND_FEAT_DIM = N_BOND_SLOTS + N_STEREO_SLOTS  # 12
RADIUS = 2

# ---------------------------------------------------------------------------
# raw molecule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawMol:
    mol_index: int
    smiles: str
    atom_class: np.ndarray
    atom_charge: np.ndarray
    atom_h: np.ndarray
    atom_chirality: np.ndarray
    edge_index: np.ndarray   # [2, 2E] directed, both directions
    edge_class: np.ndarray   # [2E]
    edge_stereo: np.ndarray  # [2E]
    n_components: int

    @property
    def n_atoms(self) -> int:
        return int(self.atom_class.shape[0])


def raw_from_record(rec: Mapping[str, Any]) -> RawMol:
    return RawMol(
        mol_index=int(rec["mol_index"]),
        smiles=str(rec["smiles"]),
        atom_class=np.asarray(rec["atom_class"], dtype=np.int64),
        atom_charge=np.asarray(rec["atom_charge"], dtype=np.int64),
        atom_h=np.asarray(rec["atom_h"], dtype=np.int64),
        atom_chirality=np.asarray(rec["atom_chirality"], dtype=np.int64),
        edge_index=np.asarray(rec["edge_index"], dtype=np.int64).reshape(2, -1),
        edge_class=np.asarray(rec["edge_class"], dtype=np.int64),
        edge_stereo=np.asarray(rec["edge_stereo"], dtype=np.int64),
        n_components=int(rec["n_components"]),
    )


def load_raw_graphs(path: Path | str) -> list[RawMol]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != "bjoc-lipo-rawgraph-v1":
        raise ValueError(f"unexpected raw graph schema {payload.get('schema')!r}")
    return [raw_from_record(r) for r in payload["rows"]]


def physical_bonds(raw: RawMol) -> list[tuple[int, int, int, int]]:
    """Deduplicate directed records -> sorted (u, v, class, stereo)."""
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    ei = raw.edge_index
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            raise ValueError("self-loop in raw graph")
        key = (min(u, v), max(u, v))
        val = (int(raw.edge_class[k]), int(raw.edge_stereo[k]))
        if key in seen and seen[key] != val:
            raise ValueError(f"inconsistent directed bond records for {key}")
        seen[key] = val
    return [(u, v, cls, st) for (u, v), (cls, st) in sorted(seen.items())]


# ---------------------------------------------------------------------------
# local node/edge features (raw attributes only; no global ring/distance stats)
# ---------------------------------------------------------------------------


def atom_base_features(raw: RawMol) -> np.ndarray:
    n = raw.n_atoms
    out = np.zeros((n, ATOM_BASE_DIM), dtype=np.float32)
    out[np.arange(n), raw.atom_class] = 1.0
    out[np.arange(n), N_ATOM_SLOTS + raw.atom_charge] = 1.0
    out[np.arange(n), N_ATOM_SLOTS + N_CHARGE_SLOTS + raw.atom_h] = 1.0
    out[np.arange(n), N_ATOM_SLOTS + N_CHARGE_SLOTS + N_H_SLOTS + raw.atom_chirality] = 1.0
    return out


def bond_features(bonds: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.zeros((len(bonds), BOND_FEAT_DIM), dtype=np.float32)
    for k, (_u, _v, cls, st) in enumerate(bonds):
        out[k, cls] = 1.0
        out[k, N_BOND_SLOTS + st] = 1.0
    return out


# ---------------------------------------------------------------------------
# rooted-radius-2 patches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    root: int
    nodes: tuple[int, ...]
    dists: tuple[int, ...]
    node_feat: np.ndarray       # [n_nodes, PATCH_NODE_DIM]
    edge_index: np.ndarray      # [2, 2E] local directed
    edge_attr: np.ndarray       # [2E, BOND_FEAT_DIM]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def extract_patches(raw: RawMol, radius: int = RADIUS) -> list[Patch]:
    """Exact induced rooted balls, one per atom, in root-atom order.

    The patch carries only its own local structure; the root is marked and the
    graph distance from the root is one-hot.  The original atom indices are
    retained internally for relabelling tests but are never fed to a model.
    """
    n = raw.n_atoms
    bonds = physical_bonds(raw)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (u, v, _cls, _st) in enumerate(bonds):
        adj[u].append((v, k))
        adj[v].append((u, k))
    base = atom_base_features(raw)
    bfeat = bond_features(bonds)
    patches: list[Patch] = []
    for root in range(n):
        dist = [-1] * n
        dist[root] = 0
        queue = [root]
        for v in queue:
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
        node_feat = np.zeros((len(nodes), PATCH_NODE_DIM), dtype=np.float32)
        node_feat[:, :ATOM_BASE_DIM] = base[nodes]
        node_feat[0, ATOM_BASE_DIM] = 1.0
        for i, v in enumerate(nodes):
            node_feat[i, ATOM_BASE_DIM + 1 + dist[v]] = 1.0
        ei: list[tuple[int, int]] = []
        ea: list[np.ndarray] = []
        for k, (u, v, _cls, _st) in enumerate(bonds):
            if u in local and v in local:
                iu, iv = local[u], local[v]
                ei.extend([(iu, iv), (iv, iu)])
                ea.extend([bfeat[k], bfeat[k]])
        patches.append(
            Patch(
                root=root,
                nodes=tuple(nodes),
                dists=tuple(int(dist[v]) for v in nodes),
                node_feat=node_feat,
                edge_index=np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64),
                edge_attr=np.stack(ea, axis=0) if ea else np.zeros((0, BOND_FEAT_DIM), dtype=np.float32),
            )
        )
    return patches


# ---------------------------------------------------------------------------
# collation
# ---------------------------------------------------------------------------


def collate_patch_lists(
    patch_lists: Sequence[Sequence[Patch]],
    y: torch.Tensor,
    mol_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Collate per-molecule patch lists into one S/X batch.

    ``patch_pos`` is only a storage index for the dense token tensor; it is not
    used as a positional feature.
    """
    node_feats: list[np.ndarray] = []
    eis: list[np.ndarray] = []
    eas: list[np.ndarray] = []
    patch_batch: list[np.ndarray] = []
    patch_to_mol: list[int] = []
    patch_pos: list[int] = []
    mol_n_patches: list[int] = []
    node_off = 0
    for bi, patches in enumerate(patch_lists):
        mol_n_patches.append(len(patches))
        for pi, p in enumerate(patches):
            pid = len(patch_to_mol)
            node_feats.append(p.node_feat)
            if p.n_edges:
                eis.append(p.edge_index + node_off)
                eas.append(p.edge_attr)
            patch_batch.append(np.full(p.n_nodes, pid, dtype=np.int64))
            patch_to_mol.append(bi)
            patch_pos.append(pi)
            node_off += p.n_nodes
    n_patches = len(patch_to_mol)
    max_patches = max(mol_n_patches) if mol_n_patches else 1
    mask = torch.zeros((len(patch_lists), max_patches), dtype=torch.bool)
    for bi, npatch in enumerate(mol_n_patches):
        mask[bi, :npatch] = True
    return {
        "patch_node_feat": torch.as_tensor(np.concatenate(node_feats, axis=0), dtype=torch.float32),
        "patch_edge_index": torch.as_tensor(np.concatenate(eis, axis=1), dtype=torch.long)
        if eis
        else torch.zeros((2, 0), dtype=torch.long),
        "patch_edge_attr": torch.as_tensor(np.concatenate(eas, axis=0), dtype=torch.float32)
        if eas
        else torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32),
        "patch_batch": torch.as_tensor(np.concatenate(patch_batch, axis=0), dtype=torch.long),
        "patch_to_mol": torch.as_tensor(patch_to_mol, dtype=torch.long),
        "patch_pos": torch.as_tensor(patch_pos, dtype=torch.long),
        "patch_mask": mask,
        "mol_n_patches": torch.as_tensor(mol_n_patches, dtype=torch.long),
        "n_patches": n_patches,
        "y": y,
        "mol_index": mol_index,
    }


def collate_graph_batch(
    raws: Sequence[RawMol],
    y: torch.Tensor,
    mol_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Collate full molecular graphs for the G reference model."""
    node_feats: list[np.ndarray] = []
    eis: list[np.ndarray] = []
    eas: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    node_off = 0
    for bi, raw in enumerate(raws):
        feats = atom_base_features(raw)
        bonds = physical_bonds(raw)
        bfeat = bond_features(bonds)
        node_feats.append(feats)
        batch.append(np.full(raw.n_atoms, bi, dtype=np.int64))
        if bonds:
            ei = []
            ea = []
            for k, (u, v, _cls, _st) in enumerate(bonds):
                ei.extend([(u + node_off, v + node_off), (v + node_off, u + node_off)])
                ea.extend([bfeat[k], bfeat[k]])
            eis.append(np.asarray(ei, dtype=np.int64).T)
            eas.append(np.stack(ea, axis=0))
        node_off += raw.n_atoms
    return {
        "node_feat": torch.as_tensor(np.concatenate(node_feats, axis=0), dtype=torch.float32),
        "edge_index": torch.as_tensor(np.concatenate(eis, axis=1), dtype=torch.long)
        if eis
        else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.as_tensor(np.concatenate(eas, axis=0), dtype=torch.float32)
        if eas
        else torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32),
        "batch": torch.as_tensor(np.concatenate(batch, axis=0), dtype=torch.long),
        "y": y,
        "mol_index": mol_index,
    }


# ---------------------------------------------------------------------------
# exact patch keys (for collision / invariance gates only; never model input)
# ---------------------------------------------------------------------------


def patch_key(patch: Patch) -> bytes:
    """Exact rooted-patch key using the frozen Phase 0 pynauty convention."""
    import sys

    phase0_code = Path(__file__).resolve().parents[2] / "results/phase0/code"
    if str(phase0_code) not in sys.path:
        sys.path.insert(0, str(phase0_code))
    import ssrc_toy as toy  # noqa: WPS433

    node_colors = []
    for i in range(patch.n_nodes):
        base = tuple(int(x) for x in np.flatnonzero(patch.node_feat[i, :ATOM_BASE_DIM]))
        root = int(patch.node_feat[i, ATOM_BASE_DIM])
        dist = int(np.argmax(patch.node_feat[i, ATOM_BASE_DIM + 1 :]))
        node_colors.append(base + (root, dist))
    edge_colors = [
        tuple(int(x) for x in np.flatnonzero(patch.edge_attr[k]))
        for k in range(patch.n_edges)
    ]
    # The patch stores both directions; Phase 0 E_2 uses one undirected edge per
    # physical bond, so keep only u < v.  Both directions carry identical colours.
    edges = [
        (int(patch.edge_index[0, k]), int(patch.edge_index[1, k]), edge_colors[k])
        for k in range(patch.n_edges)
        if int(patch.edge_index[0, k]) < int(patch.edge_index[1, k])
    ]
    return toy.pynauty_key_from_colors(node_colors, edges)


def patch_multiset_key(patches: Sequence[Patch]) -> tuple[bytes, ...]:
    return tuple(sorted(patch_key(p) for p in patches))


# ---------------------------------------------------------------------------
# label vault: only Fit/Monitor labels are retained
# ---------------------------------------------------------------------------


class LabelVault:
    """Load labels from a Fit/Monitor-only artifact, or stream the CSV.

    On the GPU host we use the committed ``labels_fit_monitor.json`` artifact
    (2,856 labels: Fit + Monitor).  The artifact has no internal Reserved rows
    and no official valid/test rows.  The CSV fallback is for local tests only;
    it streams the file and keeps only allowed rows.
    """

    def __init__(
        self,
        source_path: Path | str,
        fit_rows: Iterable[int],
        monitor_rows: Iterable[int],
        reserved_rows: Iterable[int],
    ) -> None:
        self.source_path = Path(source_path)
        self.fit = frozenset(int(r) for r in fit_rows)
        self.monitor = frozenset(int(r) for r in monitor_rows)
        self.reserved = frozenset(int(r) for r in reserved_rows)
        if self.fit & self.monitor or self.fit & self.reserved or self.monitor & self.reserved:
            raise ValueError("label-vault row sets overlap")
        self.allowed = self.fit | self.monitor
        self._labels: dict[int, float] = {}
        self._loaded = False
        self._source_kind = "json" if self.source_path.suffix == ".json" else "csv"

    def load(self) -> None:
        if self._source_kind == "json":
            payload = json.loads(self.source_path.read_text())
            scope = payload.get("scope", {})
            if scope.get("reserved_labels_present") is not False:
                raise PermissionError("label artifact does not certify Reserved exclusion")
            if scope.get("official_valid_test_labels_present") is not False:
                raise PermissionError("label artifact does not certify official valid/test exclusion")
            raw = {int(k): float(v) for k, v in payload["labels"].items()}
            if set(raw) & self.reserved:
                raise PermissionError("Reserved labels present in artifact")
            # Keep only the rows this run may use.  Other Fit labels are
            # allowed-domain data but are irrelevant for an n=253 run.
            self._labels = {k: v for k, v in raw.items() if k in self.allowed}
            del raw
        else:
            with open(self.source_path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                exp_index = header.index("exp")
                for i, row in enumerate(reader):
                    if i in self.allowed:
                        self._labels[i] = float(row[exp_index])
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

    def assert_no_reserved(self, rows: Iterable[int]) -> None:
        bad = sorted(set(int(r) for r in rows) & self.reserved)
        if bad:
            raise PermissionError(f"Reserved rows in Phase 1a batch: {bad[:5]}")

    def audit(self) -> dict[str, Any]:
        return {
            "fit_rows": len(self.fit),
            "monitor_rows": len(self.monitor),
            "reserved_rows": len(self.reserved),
            "allowed_rows": len(self.allowed),
            "labels_retained": len(self._labels),
            "reserved_labels_retained": len(set(self._labels) & self.reserved),
            "official_valid_test_labels_retained": 0,
            "source_kind": self._source_kind,
            "source_sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
        }


# ---------------------------------------------------------------------------
# split container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSplit:
    path: Path
    payload: dict[str, Any]

    @property
    def fit(self) -> list[int]:
        return [int(r) for r in self.payload["rows"]["fit"]]

    @property
    def monitor(self) -> list[int]:
        return [int(r) for r in self.payload["rows"]["monitor"]]

    @property
    def reserved(self) -> list[int]:
        return [int(r) for r in self.payload["rows"]["reserved"]]

    @property
    def small(self) -> list[int]:
        return [int(r) for r in self.payload["small"]["rows"]]

    @property
    def small_n(self) -> int:
        return len(self.small)

    def training_rows(self, n: int | str) -> list[int]:
        if n == "small":
            return self.small
        if int(n) == len(self.fit):
            return self.fit
        raise ValueError(f"unknown n={n!r}")


def load_frozen_split(path: Path | str) -> FrozenSplit:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("protocol_id") != "ssrc-phase1a-lipo-v1":
        raise ValueError("wrong split protocol")
    if payload.get("scope", {}).get("labels_read") is not False:
        raise ValueError("split was not frozen label-blind")
    split = FrozenSplit(path=path, payload=payload)
    if len(split.fit) != 2352 or len(split.monitor) != 504 or len(split.reserved) != 504:
        raise ValueError("unexpected split sizes")
    all_rows = split.fit + split.monitor + split.reserved
    if len(set(all_rows)) != len(all_rows):
        raise ValueError("split rows are not unique")
    if set(split.fit) & set(split.monitor) or set(split.fit) & set(split.reserved) or set(split.monitor) & set(split.reserved):
        raise ValueError("split overlap")
    if not set(split.small) <= set(split.fit):
        raise ValueError("small subset not in Fit")
    return split


# ---------------------------------------------------------------------------
# dataset caches
# ---------------------------------------------------------------------------


@dataclass
class PatchCache:
    raws: list[RawMol]
    patches: dict[int, list[Patch]]
    build_seconds: float
    n_patches: int

    @classmethod
    def build(cls, raws: list[RawMol], rows: Iterable[int], radius: int = RADIUS) -> "PatchCache":
        t0 = time.perf_counter()
        patches: dict[int, list[Patch]] = {}
        total = 0
        for row in rows:
            p = extract_patches(raws[int(row)], radius=radius)
            patches[int(row)] = p
            total += len(p)
        return cls(raws=raws, patches=patches, build_seconds=time.perf_counter() - t0, n_patches=total)

    def patch_lists(self, rows: Sequence[int]) -> list[list[Patch]]:
        return [self.patches[int(r)] for r in rows]


def scaffold_groups_from_split(split: FrozenSplit) -> dict[int, str]:
    """Map every Fit/Monitor/Reserved row to its frozen scaffold hash.

    The split file stores scaffold-group hashes only at the group level, not a
    per-row map.  For cluster bootstrap we reconstruct scaffold groups from the
    committed label-free row manifest (structure-only).
    """

    manifest = json.loads(
        (Path(__file__).resolve().parents[4]
         / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/row_manifest.json").read_text()
    )
    hashes = manifest["scaffold_sha256"]
    return {i: str(h) for i, h in enumerate(hashes)}
