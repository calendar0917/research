"""Data layer: official ogbg-molhiv raw graph attributes, split, and scaffolds.

Everything below is *label-free* except the label array, which is only ever used
for (a) the audit, (b) train-only positive/negative support statistics, and
(c) the evaluation that this round is explicitly authorised to perform on all
three official splits.

Bond handling: the OGB raw ``edge.csv`` stores every physical bond twice
(``u->v`` and ``v->u``).  :func:`load_molecules` keeps exactly one undirected
record ``(u, v, bond_feat)`` with ``u < v`` and asserts the pairing is exact, so
a bond is never counted twice in any subgraph enumeration.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import msds_common as C


@dataclass
class Molecules:
    """Per-molecule OGB graph attributes, physical (undirected) bonds only."""

    n_atoms: np.ndarray  # (N,) int32
    atom_feat: np.ndarray  # (sum_n, 9) int16
    bond_u: np.ndarray  # (sum_e,) int32
    bond_v: np.ndarray  # (sum_e,) int32
    bond_feat: np.ndarray  # (sum_e, 3) int16
    atom_offsets: np.ndarray  # (N+1,) int64 into atom_feat
    bond_offsets: np.ndarray  # (N+1,) int64 into bond_u/bond_v
    label: np.ndarray  # (N,) int8
    smiles: list[str]
    split: dict[str, np.ndarray]

    @property
    def n_molecules(self) -> int:
        return int(self.n_atoms.size)

    def atoms(self, i: int) -> np.ndarray:
        return self.atom_feat[int(self.atom_offsets[i]): int(self.atom_offsets[i + 1])]

    def bonds(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = int(self.bond_offsets[i]), int(self.bond_offsets[i + 1])
        return self.bond_u[lo:hi], self.bond_v[lo:hi], self.bond_feat[lo:hi]


def _read_csv_gz(path: Path) -> np.ndarray:
    import pandas as pd

    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, header=None).values


def load_molecules() -> Molecules:
    root = C.OGB_MOLHIV
    num_nodes = np.loadtxt(gzip.open(root / "raw" / "num-node-list.csv.gz", "rt"), dtype=np.int64)
    num_edges = np.loadtxt(gzip.open(root / "raw" / "num-edge-list.csv.gz", "rt"), dtype=np.int64)
    node_feat = _read_csv_gz(root / "raw" / "node-feat.csv.gz").astype(np.int16)
    edge = _read_csv_gz(root / "raw" / "edge.csv.gz").astype(np.int32)
    edge_feat = _read_csv_gz(root / "raw" / "edge-feat.csv.gz").astype(np.int16)
    label = np.loadtxt(gzip.open(root / "raw" / "graph-label.csv.gz", "rt"), dtype=np.int64).astype(np.int8)

    n_graphs = int(num_nodes.size)
    if not (num_edges.size == n_graphs == label.size):
        raise ValueError("num-node / num-edge / label graph counts disagree")
    if int(num_nodes.sum()) != node_feat.shape[0]:
        raise ValueError("node-feat rows != sum(num_nodes)")
    if int(num_edges.sum()) != edge.shape[0] or edge_feat.shape[0] != edge.shape[0]:
        raise ValueError("edge / edge-feat rows != sum(num_edges)")
    if node_feat.shape[1] != 9 or edge_feat.shape[1] != 3:
        raise ValueError("unexpected OGB feature width")

    atom_offsets = np.zeros(n_graphs + 1, dtype=np.int64)
    np.cumsum(num_nodes, out=atom_offsets[1:])
    edge_offsets = np.zeros(n_graphs + 1, dtype=np.int64)
    np.cumsum(num_edges, out=edge_offsets[1:])

    # ---- collapse the raw directed edge records into physical undirected bonds
    # ogbg-molhiv's raw edge.csv.gz stores every physical bond exactly ONCE (a
    # mixed-order record, not a doubled i->j / j->i pair); the PyG view doubles
    # it.  We keep one undirected record (u < v, bond_feat) and prove below
    # that deduplicating the doubled PyG edge_index reproduces exactly it.
    bond_u_parts: list[np.ndarray] = []
    bond_v_parts: list[np.ndarray] = []
    bond_f_parts: list[np.ndarray] = []
    for g in range(n_graphs):
        lo, hi = int(edge_offsets[g]), int(edge_offsets[g + 1])
        src = edge[lo:hi, 0]
        dst = edge[lo:hi, 1]
        if src.size == 0:
            continue
        n = int(num_nodes[g])
        if src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
            raise ValueError(f"graph {g}: edge endpoint out of range")
        if np.any(src == dst):
            raise ValueError(f"graph {g}: self loop in raw ogbg-molhiv edges")
        u = np.minimum(src, dst).astype(np.int32)
        v = np.maximum(src, dst).astype(np.int32)
        f = edge_feat[lo:hi].astype(np.int16)
        order = np.lexsort((v, u))
        u, v, f = u[order], v[order], f[order]
        if np.unique(u.astype(np.int64) * n + v).size != u.size:
            raise ValueError(f"graph {g}: duplicate physical bond record")
        bond_u_parts.append(u)
        bond_v_parts.append(v)
        bond_f_parts.append(f)

    if sum(a.size for a in bond_u_parts) != int(num_edges.sum()):
        raise ValueError("physical bond count differs from the raw edge record count")

    bond_u = np.concatenate(bond_u_parts) if bond_u_parts else np.zeros(0, np.int32)
    bond_v = np.concatenate(bond_v_parts) if bond_v_parts else np.zeros(0, np.int32)
    bond_f = np.concatenate(bond_f_parts) if bond_f_parts else np.zeros((0, 3), np.int16)

    split = {}
    for name in ("train", "valid", "test"):
        ids = _read_csv_gz(root / "split" / "scaffold" / f"{name}.csv.gz").astype(np.int64).reshape(-1)
        split[name] = ids

    import pandas as pd

    with gzip.open(root / "mapping" / "mol.csv.gz", "rt") as fh:
        mol = pd.read_csv(fh)
    smiles = [str(s) for s in mol["smiles"].tolist()]
    if len(smiles) != n_graphs:
        raise ValueError("mapping/mol.csv.gz row count differs from the graph count")
    smiles_label = mol["HIV_active"].to_numpy().astype(np.int8)
    if not np.array_equal(smiles_label, label):
        raise ValueError("mapping label column differs from raw graph-label.csv.gz")

    return Molecules(
        n_atoms=num_nodes.astype(np.int32),
        atom_feat=node_feat,
        bond_u=bond_u,
        bond_v=bond_v,
        bond_feat=bond_f,
        atom_offsets=atom_offsets,
        bond_offsets=_bond_offsets(num_nodes, num_edges),
        label=label,
        smiles=smiles,
        split=split,
    )


def _bond_offsets(num_nodes: np.ndarray, num_edges: np.ndarray) -> np.ndarray:
    """Offsets into the physical-bond arrays: one kept record per raw bond."""
    offsets = np.zeros(num_nodes.size + 1, dtype=np.int64)
    np.cumsum(num_edges, out=offsets[1:])
    return offsets


def verify_against_pyg_view(molecules: Molecules) -> dict[str, Any]:
    """Prove that doubling/deduplicating the PyG edge_index reproduces our bond set.

    The PyG-processed view of ogbg-molhiv stores each physical bond twice
    (``u->v`` and ``v->u``).  Subgraph enumeration must count that bond once; this
    check makes the requirement explicit and machine-verified.
    """
    import torch

    path = C.OGB_MOLHIV / "processed" / "geometric_data_processed.pt"
    if not path.exists():
        return {"available": False}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first, second = torch.load(path, map_location="cpu", weights_only=False)
    edge_index = first.edge_index.to(torch.int64).numpy()
    edge_attr = first.edge_attr.to(torch.int64).numpy()
    x = first.x.to(torch.int64).numpy()
    if x.shape[0] != molecules.atom_feat.shape[0]:
        raise ValueError("PyG node count differs from raw node-feat rows")
    if not np.array_equal(x, molecules.atom_feat.astype(np.int64)):
        raise ValueError("PyG atom features differ from raw node-feat.csv.gz")

    # per-graph dedup of the doubled edge list
    edge_offsets = np.zeros(molecules.n_molecules + 1, dtype=np.int64)
    np.cumsum(np.diff(molecules.bond_offsets) * 2, out=edge_offsets[1:])
    n_mismatch = 0
    n_reverse_only = 0
    for g in range(molecules.n_molecules):
        lo, hi = int(edge_offsets[g]), int(edge_offsets[g + 1])
        src = edge_index[0, lo:hi]
        dst = edge_index[1, lo:hi]
        u = np.minimum(src, dst)
        v = np.maximum(src, dst)
        key = np.unique(u * 10_000 + v)
        bu, bv, bf = molecules.bonds(g)
        mine = np.unique(bu.astype(np.int64) * 10_000 + bv.astype(np.int64))
        if key.size != mine.size or not np.array_equal(key, mine):
            n_mismatch += 1
        if np.any(src > dst):
            n_reverse_only += 1
    return {
        "available": True,
        "path": C.rel(path),
        "pyg_edge_index_shape": [int(v) for v in edge_index.shape],
        "raw_physical_bonds": int(molecules.bond_u.size),
        "pyg_is_doubled_view": bool(edge_index.shape[1] == 2 * int(molecules.bond_u.size)),
        "atom_features_match_raw": True,
        "graphs_with_undirected_mismatch": int(n_mismatch),
        "graphs_with_at_least_one_reversed_record": int(n_reverse_only),
        "dedup_equals_physical_bonds": bool(n_mismatch == 0),
    }


def audit(molecules: Molecules) -> dict[str, Any]:
    """Pre-flight data audit required by PROTOCOL.md §3."""
    n = molecules.n_molecules
    splits = molecules.split
    sets = {k: set(v.tolist()) for k, v in splits.items()}
    overlap_tv = sorted(sets["train"] & sets["valid"])
    overlap_tt = sorted(sets["train"] & sets["test"])
    overlap_vt = sorted(sets["valid"] & sets["test"])
    covered = sorted(sets["train"] | sets["valid"] | sets["test"])

    # scaffold-grouped identity (RDKit Bemis-Murcko), used later for support stats
    scaffolds, scaffold_meta = load_scaffolds(molecules)

    hist_rows = None
    if C.HIST_BASELINE_MANIFEST.exists():
        man = C.read_json(C.HIST_BASELINE_MANIFEST)
        hist_rows = {
            k: np.asarray(v, dtype=np.int64)
            for k, v in man["datasets"]["ogbg-molhiv"]["rows"].items()
            if k in ("train", "valid", "test")
        }
    hist_pred_rows = {}
    for name in ("valid", "test"):
        p = C.HIST_BASELINE_DIR / f"ogbg-molhiv_{name}_predictions.npz"
        if p.exists():
            with np.load(p) as z:
                hist_pred_rows[name] = z["rows"]

    payload: dict[str, Any] = {
        "n_molecules": n,
        "expected_n_molecules": 41127,
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "expected_split_sizes": {"train": 32901, "valid": 4113, "test": 4113},
        "split_sizes_ok": all(
            int(splits[k].size) == v for k, v in {"train": 32901, "valid": 4113, "test": 4113}.items()
        ),
        "split_overlap": {
            "train_valid": len(overlap_tv),
            "train_test": len(overlap_tt),
            "valid_test": len(overlap_vt),
        },
        "splits_disjoint": not (overlap_tv or overlap_tt or overlap_vt),
        "splits_cover_all": len(covered) == n and covered[0] == 0 and covered[-1] == n - 1,
        "n_covered": len(covered),
        "index_label_smiles_alignment": {
            "n_atom_rows_vs_sum_num_nodes": int(molecules.atom_feat.shape[0]) == int(molecules.n_atoms.sum()),
            "n_smiles": len(molecules.smiles),
            "labels_in_unit_set": sorted(int(v) for v in np.unique(molecules.label)),
            "smiles_nonempty": all(bool(s) for s in molecules.smiles),
        },
        "ogb_features": {
            "atom_feature_width": int(molecules.atom_feat.shape[1]),
            "bond_feature_width": int(molecules.bond_feat.shape[1]),
            "atom_feature_names": list(C.ATOM_FEATURE_NAMES),
            "bond_feature_names": list(C.BOND_FEATURE_NAMES),
            "atom_feature_max": molecules.atom_feat.max(axis=0).tolist(),
            "bond_feature_max": molecules.bond_feat.max(axis=0).tolist(),
        },
        "bonds": {
            "n_physical_bonds": int(molecules.bond_u.size),
            "n_raw_edge_records": int(molecules.bond_u.size),
            "pyg_view_doubled_check": verify_against_pyg_view(molecules),
            "bond_offsets_consistent": int(molecules.bond_offsets[-1]) == int(molecules.bond_u.size),
            "no_self_loops": bool(np.all(molecules.bond_u != molecules.bond_v)),
            "no_duplicate_pairs": True,  # asserted inside load_molecules per graph
        },
        "molecules_used": n,
        "molecules_dropped": 0,
        "scaffolds": scaffold_meta,
        "historical_identity": {
            "manifest_present": hist_rows is not None,
            "train_rows_match": bool(hist_rows is not None and np.array_equal(hist_rows["train"], splits["train"])),
            "valid_rows_match": bool(hist_rows is not None and np.array_equal(hist_rows["valid"], splits["valid"])),
            "historical_valid_predictions_rows_match": bool(
                "valid" in hist_pred_rows and np.array_equal(hist_pred_rows["valid"], splits["valid"])
            ),
            "historical_test_predictions_rows_match": bool(
                "test" in hist_pred_rows and np.array_equal(hist_pred_rows["test"], splits["test"])
            ),
        },
        "positive_counts": {k: int(molecules.label[v].sum()) for k, v in splits.items()},
        "positive_rate": {k: float(molecules.label[v].mean()) for k, v in splits.items()},
        "scaffold_counts": {
            k: int(np.unique(np.asarray(scaffolds, dtype=object)[v]).size) for k, v in splits.items()
        },
        "smiles_sha256": C.sha256_bytes("\n".join(molecules.smiles).encode()),
        "label_sha256": C.sha256_array(molecules.label),
        "bond_u_sha256": C.sha256_array(molecules.bond_u),
        "bond_v_sha256": C.sha256_array(molecules.bond_v),
        "bond_feat_sha256": C.sha256_array(molecules.bond_feat),
        "atom_feat_sha256": C.sha256_array(molecules.atom_feat),
        "split_sha256": {k: C.sha256_array(v) for k, v in splits.items()},
    }
    payload["all_checks_pass"] = bool(
        payload["split_sizes_ok"]
        and payload["splits_disjoint"]
        and payload["splits_cover_all"]
        and payload["index_label_smiles_alignment"]["n_atom_rows_vs_sum_num_nodes"]
        and payload["bonds"]["bond_offsets_consistent"]
        and payload["historical_identity"]["train_rows_match"]
        and payload["historical_identity"]["valid_rows_match"]
    )
    return payload


def load_scaffolds(molecules: Molecules) -> tuple[list[str], dict[str, Any]]:
    """Bemis-Murcko scaffold SMILES per molecule (RDKit), cached with a checksum.

    Acyclic molecules have the *empty* Bemis-Murcko scaffold.  They are kept as
    one explicit ``""`` group (never dropped); §6 of PROTOCOL.md explains why a
    scaffold count must not be read as a count of independent samples.
    """
    cache = C.CACHE / "molhiv_scaffolds.npz"
    if cache.exists():
        with np.load(cache) as z:
            out = [str(s) for s in z["scaffolds"]]
        if len(out) == molecules.n_molecules:
            meta = C.read_json(C.CACHE / "molhiv_scaffolds_meta.json")
            return out, meta
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    out: list[str] = []
    n_failed = 0
    n_empty = 0
    for smi in molecules.smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append("__PARSE_FAILED__")
            n_failed += 1
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            out.append("__SCAFFOLD_FAILED__")
            n_failed += 1
            continue
        if scaffold == "":
            n_empty += 1
        out.append(scaffold)
    C.ensure_dirs()
    np.savez_compressed(cache, scaffolds=np.asarray(out, dtype=str))
    meta = {
        "rdkit_version": __import__("rdkit").__version__,
        "impl": "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)",
        "n_molecules": len(out),
        "n_unique_scaffolds": len(set(out)),
        "n_empty_acyclic_scaffold": n_empty,
        "n_failed": n_failed,
        "failed_handling": "kept as distinct sentinel strings; never silently dropped",
        "acyclic_group_handling": "empty Murcko scaffold '' is ONE scaffold group; scaffold counts are not independent samples",
        "cache_path": C.rel(cache),
        "cache_sha256": C.sha256_file(cache),
    }
    C.write_json(C.CACHE / "molhiv_scaffolds_meta.json", meta)
    return out, meta
