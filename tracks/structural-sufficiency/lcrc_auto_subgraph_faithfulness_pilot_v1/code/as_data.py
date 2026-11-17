"""Split caches, per-molecule reconstruction and the occurrence -> physical-bond map.

Why this module exists
----------------------
The frozen S1/S2 caches store occurrence *codes* (``nbr_code``, ``path_cat``,
``path_end63``, ``path_closure``) but **not** the ``(root, middle, end)`` triple
or the physical bond ids behind each occurrence.  The gating experiment in
:mod:`as_gate` needs exactly that missing bookkeeping, so it is rebuilt here from
the frozen ``molecular_graph.enumerate_s1`` / ``enumerate_s2`` deterministic
order and then *verified item by item against the cache*.

A single unverifiable occurrence aborts the dataset (module gate
``OCCURRENCE_MAPPING``); the round never guesses a correspondence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import as_common as C
import molecular_graph as MG

C.install_import_paths()

import lcrc_data as PREP_DATA  # noqa: E402  (frozen read-only reuse)


# ---------------------------------------------------------------------------
# per-molecule views of the frozen split cache
# ---------------------------------------------------------------------------

_MOL_KEYS = (
    "atom_feat",
    "atom_scalar",
    "atom_type",
    "bond_feat",
    "bond_type",
    "bond_index",
    "nbr_code",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
)


def slice_molecule(cache: Mapping[str, Any], row: int) -> dict[str, Any]:
    """One molecule's cached arrays (no re-derivation, no relabelling)."""
    row = int(row)
    a0, a1 = int(cache["atom_ptr"][row]), int(cache["atom_ptr"][row + 1])
    b0, b1 = int(cache["bond_ptr"][row]), int(cache["bond_ptr"][row + 1])
    n0, n1 = int(cache["nbr_ptr"][a0]), int(cache["nbr_ptr"][a1])
    p0, p1 = int(cache["path_ptr"][a0]), int(cache["path_ptr"][a1])
    out: dict[str, Any] = {
        "n_atoms": a1 - a0,
        "n_bonds": b1 - b0,
        "atom_feat": np.asarray(cache["atom_feat"][a0:a1], dtype=np.float32),
        "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1], dtype=np.float32),
        "atom_type": np.asarray(cache["atom_type"][a0:a1], dtype=np.int64),
        "bond_feat": np.asarray(cache["bond_feat"][b0:b1], dtype=np.float32),
        "bond_type": np.asarray(cache["bond_type"][b0:b1], dtype=np.int64),
        "bond_index": np.asarray(cache["bond_index"][b0:b1], dtype=np.int64).reshape(-1, 2),
        "nbr_code": np.asarray(cache["nbr_code"][n0:n1], dtype=np.int64),
        "nbr_ptr_local": (
            np.asarray(cache["nbr_ptr"][a0 : a1 + 1], dtype=np.int64) - int(cache["nbr_ptr"][a0])
        ),
        "path_cat": np.asarray(cache["path_cat"][p0:p1], dtype=np.int64),
        "path_end63": np.asarray(cache["path_end63"][p0:p1], dtype=np.int64),
        "path_end": np.asarray(cache["path_end"][p0:p1], dtype=np.int64),
        "path_closure": np.asarray(cache["path_closure"][p0:p1], dtype=np.int64),
        "path_ptr_local": (
            np.asarray(cache["path_ptr"][a0 : a1 + 1], dtype=np.int64) - int(cache["path_ptr"][a0])
        ),
        "global": np.asarray(cache["global"][row], dtype=np.float32),
        "smiles": str(cache["smiles"][row]),
        "molecule_id": str(cache["molecule_ids"][row]),
    }
    return out


def analysis_from_mol(mol: Mapping[str, Any]) -> tuple[MG.MolecularGraph, MG.GraphAnalysis]:
    """Rebuild the canonical molecular graph from cached arrays and analyse it."""
    graph = MG.build_molecular_graph(
        atom_type=np.asarray(mol["atom_type"], dtype=np.int64),
        bond_index=np.asarray(mol["bond_index"], dtype=np.int64),
        bond_type=np.asarray(mol["bond_type"], dtype=np.int64),
        atom_feat=np.asarray(mol["atom_feat"], dtype=np.float32),
        bond_feat=np.asarray(mol["bond_feat"], dtype=np.float32),
        molecule_id=str(mol.get("molecule_id", "")),
        smiles=str(mol.get("smiles", "")),
    )
    analysis = MG.analyse_graph(graph)
    return graph, analysis


# ---------------------------------------------------------------------------
# occurrence map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OccMap:
    """Every S1/S2 occurrence with its root atom and its physical bond ids."""

    n_atoms: int
    n_bonds: int
    nbr_ptr: np.ndarray
    nbr_code: np.ndarray
    nbr_atom: np.ndarray
    nbr_bond: np.ndarray
    path_ptr: np.ndarray
    path_root: np.ndarray
    path_middle: np.ndarray
    path_end: np.ndarray
    path_bond1: np.ndarray
    path_bond2: np.ndarray
    path_cat: np.ndarray
    path_end63: np.ndarray
    path_closure: np.ndarray

    @property
    def n_nbr(self) -> int:
        return int(self.nbr_code.size)

    @property
    def n_path(self) -> int:
        return int(self.path_cat.size)


def build_occ_map(atom_type: np.ndarray, analysis: MG.GraphAnalysis, n_bond_types: int) -> OccMap:
    """Rebuild ``enumerate_s1`` / ``enumerate_s2`` in their *deterministic* order.

    The third element of every ``analysis.adj`` entry is the physical bond id, so
    each S1 neighbour code and each ordered non-backtracking 2-path occurrence is
    traced back to real bonds.  This is the only place where that correspondence
    is created.
    """
    at = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    nb = int(n_bond_types)
    adj = analysis.adj
    n = int(at.size)

    nbr_counts = np.asarray([len(row) for row in adj], dtype=np.int64)
    nbr_ptr = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int64)
    nbr_code: list[int] = []
    nbr_atom: list[int] = []
    nbr_bond: list[int] = []
    for root, row in enumerate(adj):
        for v, bt, e in row:
            nbr_code.append(int(at[int(v)]) * nb + (int(bt) - 1))
            nbr_atom.append(int(root))
            nbr_bond.append(int(e))

    path_counts = np.zeros(n, dtype=np.int64)
    path_root: list[int] = []
    path_middle: list[int] = []
    path_end: list[int] = []
    path_b1: list[int] = []
    path_b2: list[int] = []
    path_cat: list[int] = []
    path_end63: list[int] = []
    path_closure: list[int] = []
    for root, row in enumerate(adj):
        before = len(path_root)
        for middle, bt1, e1 in row:
            type_middle = int(at[int(middle)])
            for end, bt2, e2 in adj[int(middle)]:
                end = int(end)
                if end == root:
                    continue
                path_root.append(int(root))
                path_middle.append(int(middle))
                path_end.append(end)
                path_b1.append(int(e1))
                path_b2.append(int(e2))
                path_cat.append(type_middle * nb * nb + (int(bt1) - 1) * nb + (int(bt2) - 1))
                path_end63.append(int(at[end]) * nb + (int(bt2) - 1))
                path_closure.append(1 if int(analysis.dist[root, end]) == 1 else 0)
        path_counts[root] = len(path_root) - before
    path_ptr = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int64)

    def _arr(values: Sequence[int], dtype=np.int64) -> np.ndarray:
        return np.asarray(values, dtype=dtype) if values else np.zeros(0, dtype=dtype)

    return OccMap(
        n_atoms=n,
        n_bonds=int(analysis.n_bonds),
        nbr_ptr=nbr_ptr,
        nbr_code=_arr(nbr_code),
        nbr_atom=_arr(nbr_atom),
        nbr_bond=_arr(nbr_bond),
        path_ptr=path_ptr,
        path_root=_arr(path_root),
        path_middle=_arr(path_middle),
        path_end=_arr(path_end),
        path_bond1=_arr(path_b1),
        path_bond2=_arr(path_b2),
        path_cat=_arr(path_cat),
        path_end63=_arr(path_end63),
        path_closure=_arr(path_closure),
    )


def verify_occurrences(occ: OccMap, mol: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Item-by-item agreement between the rebuilt map and the frozen cache."""
    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    _check("nbr_ptr", np.array_equal(occ.nbr_ptr, np.asarray(mol["nbr_ptr_local"])),
           {"rebuilt": int(occ.n_nbr), "cache": int(mol["nbr_code"].size)})
    _check("nbr_code", np.array_equal(occ.nbr_code, np.asarray(mol["nbr_code"], dtype=np.int64)))
    _check("path_ptr", np.array_equal(occ.path_ptr, np.asarray(mol["path_ptr_local"])),
           {"rebuilt": int(occ.n_path), "cache": int(mol["path_cat"].size)})
    _check("path_cat", np.array_equal(occ.path_cat, np.asarray(mol["path_cat"], dtype=np.int64)))
    _check("path_end63", np.array_equal(occ.path_end63, np.asarray(mol["path_end63"], dtype=np.int64)))
    _check("path_end", np.array_equal(occ.path_end, np.asarray(mol["path_end"], dtype=np.int64)))
    _check("path_closure", np.array_equal(occ.path_closure, np.asarray(mol["path_closure"], dtype=np.int64)))
    # Structural invariants of the rebuilt map itself.
    if occ.n_nbr:
        root_of = np.repeat(np.arange(occ.n_atoms), np.diff(occ.nbr_ptr))
        _check("nbr_root_consistent", np.array_equal(root_of, occ.nbr_atom))
        u = np.asarray(mol["bond_index"])[:, 0]
        v = np.asarray(mol["bond_index"])[:, 1]
        endpoints_ok = (
            ((u[occ.nbr_bond] == occ.nbr_atom) | (v[occ.nbr_bond] == occ.nbr_atom)).all()
        )
        _check("nbr_bond_incident_to_root", bool(endpoints_ok))
    if occ.n_path:
        root_of = np.repeat(np.arange(occ.n_atoms), np.diff(occ.path_ptr))
        _check("path_root_consistent", np.array_equal(root_of, occ.path_root))
        u = np.asarray(mol["bond_index"])[:, 0]
        v = np.asarray(mol["bond_index"])[:, 1]
        e1 = occ.path_bond1
        e2 = occ.path_bond2
        mid = occ.path_middle
        _check("path_bond1_incident_root_middle",
               bool((((u[e1] == occ.path_root) & (v[e1] == mid)) | ((v[e1] == occ.path_root) & (u[e1] == mid))).all()))
        _check("path_bond2_incident_middle_end",
               bool((((u[e2] == mid) & (v[e2] == occ.path_end)) | ((v[e2] == mid) & (u[e2] == occ.path_end))).all()))
        _check("path_no_immediate_backtrack", bool((occ.path_end != occ.path_root).all()))
        _check("path_bonds_distinct", bool((e1 != e2).all()), {"violations": int((e1 == e2).sum())})
        # closure == 1 exactly for the triangle case (root and end are bonded)
        bonded = set(zip(u.tolist(), v.tolist()))
        expected_closure = np.asarray(
            [1 if (min(int(a), int(b)), max(int(a), int(b))) in bonded else 0
             for a, b in zip(occ.path_root.tolist(), occ.path_end.tolist())],
            dtype=np.int64,
        )
        _check("path_closure_matches_endpoints", np.array_equal(expected_closure, occ.path_closure),
               {"n_closure": int(occ.path_closure.sum())})
    failed = [c for c in checks if not c["ok"]]
    report = {
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": [c["check"] for c in failed],
        "checks": checks,
        "n_s1_occurrences": int(occ.n_nbr),
        "n_s2_occurrences": int(occ.n_path),
    }
    if strict and failed:
        raise ValueError(f"occurrence mapping does not reproduce the cache: {report['failed']}")
    return report


# ---------------------------------------------------------------------------
# split-level occurrence arrays + gated collate
# ---------------------------------------------------------------------------


@dataclass
class SplitOcc:
    """Split-indexed occurrence bookkeeping (cache row order, not row ids)."""

    dataset: str
    split: str
    n_molecules: int
    nbr_bond: np.ndarray        # per S1 occurrence: local physical bond id
    path_bond1: np.ndarray      # per S2 occurrence: local physical bond id 1
    path_bond2: np.ndarray      # per S2 occurrence: local physical bond id 2
    path_middle: np.ndarray     # per S2 occurrence: local middle atom
    bond_counts: np.ndarray
    atom_counts: np.ndarray
    n_s2_total: np.ndarray      # per molecule: number of S2 occurrences
    n_bond_types: int
    sha256: str


def build_split_occ(cache: Mapping[str, Any], spec, dataset: str, split: str, *,
                    max_molecules: int | None = None, strict: bool = True) -> SplitOcc:
    n = int(cache["n_molecules"]) if max_molecules is None else int(max_molecules)
    nbr_parts: list[np.ndarray] = []
    p1_parts: list[np.ndarray] = []
    p2_parts: list[np.ndarray] = []
    pm_parts: list[np.ndarray] = []
    n_s2: list[int] = []
    failures: list[dict[str, Any]] = []
    for row in range(n):
        mol = slice_molecule(cache, row)
        _graph, analysis = analysis_from_mol(mol)
        if not np.array_equal(np.stack([analysis.bond_u, analysis.bond_v], axis=1), mol["bond_index"]):
            raise ValueError(f"{dataset}/{split}/{row}: bond order is not the cached canonical order")
        occ = build_occ_map(mol["atom_type"], analysis, int(spec.n_bond_types))
        report = verify_occurrences(occ, mol, strict=False)
        if report["n_failed"]:
            failures.append({"row": row, "failed": report["failed"]})
            if strict:
                raise ValueError(f"{dataset}/{split}/{row}: occurrence mapping failed {report['failed']}")
        nbr_parts.append(occ.nbr_bond)
        p1_parts.append(occ.path_bond1)
        p2_parts.append(occ.path_bond2)
        pm_parts.append(occ.path_middle)
        n_s2.append(int(occ.n_path))
    nbr_bond = np.concatenate(nbr_parts) if nbr_parts else np.zeros(0, dtype=np.int64)
    p1 = np.concatenate(p1_parts) if p1_parts else np.zeros(0, dtype=np.int64)
    p2 = np.concatenate(p2_parts) if p2_parts else np.zeros(0, dtype=np.int64)
    pm = np.concatenate(pm_parts) if pm_parts else np.zeros(0, dtype=np.int64)
    atom_counts = np.diff(np.asarray(cache["atom_ptr"][: n + 1], dtype=np.int64))
    bond_counts = np.diff(np.asarray(cache["bond_ptr"][: n + 1], dtype=np.int64))
    n_s2_total = np.asarray(n_s2, dtype=np.int64)
    digest = C.sha256_json(
        {
            "dataset": dataset,
            "split": split,
            "n": n,
            "nbr_bond": C.sha256_array(nbr_bond),
            "path_bond1": C.sha256_array(p1),
            "path_bond2": C.sha256_array(p2),
            "path_middle": C.sha256_array(pm),
            "n_s2": C.sha256_array(n_s2_total),
            "occurrence_failures": failures,
        }
    )
    return SplitOcc(
        dataset=dataset,
        split=split,
        n_molecules=n,
        nbr_bond=np.asarray(nbr_bond, dtype=np.int64),
        path_bond1=np.asarray(p1, dtype=np.int64),
        path_bond2=np.asarray(p2, dtype=np.int64),
        path_middle=np.asarray(pm, dtype=np.int64),
        bond_counts=np.asarray(bond_counts, dtype=np.int64),
        atom_counts=np.asarray(atom_counts, dtype=np.int64),
        n_s2_total=n_s2_total,
        n_bond_types=int(spec.n_bond_types),
        sha256=digest,
    )


def _range_index(ptr: np.ndarray, rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Same index algebra as ``lcrc_data.collate`` (kept in one place, verified)."""
    rows = np.asarray(rows, dtype=np.int64)
    counts = ptr[rows + 1] - ptr[rows]
    if counts.size == 0 or int(counts.sum()) == 0:
        return np.zeros(0, dtype=np.int64), counts
    idx = np.concatenate([np.arange(int(s), int(s) + int(c), dtype=np.int64)
                          for s, c in zip(ptr[rows], counts)])
    return idx, counts


@dataclass
class GatedBatch:
    """A frozen collate result plus per-bond / per-occurrence graph bookkeeping.

    ``gate_matrix`` inputs are ``(n_graphs, max_bonds)`` tensors of physical-bond
    gates; the S1 side reads the gate of the bond that produced each neighbour
    code and the S2 side reads the product of the two path bonds' gates.
    """

    batch: Mapping[str, torch.Tensor]
    n_graphs: int
    bond_jump: torch.Tensor      # (n_bond_rows,) row index into the gate matrix
    nbr_jump: torch.Tensor       # (n_nbr,) row index into the gate matrix
    nbr_bond: torch.Tensor       # (n_nbr,) local physical bond id
    path_jump: torch.Tensor      # (n_path,) row index into the gate matrix
    path_bond1: torch.Tensor
    path_bond2: torch.Tensor
    bond_local: torch.Tensor     # (n_bond_rows,) local physical bond id
    bond_counts: np.ndarray
    max_bonds: int
    atom_jump: torch.Tensor      # (n_atoms,) graph index
    occurrence_check: Mapping[str, Any]
    _moved: dict[str, "GatedBatch"] = field(default_factory=dict, repr=False, compare=False)

    def to(self, device) -> "GatedBatch":
        """A device-resident copy of the whole batch (cached per device).

        ``lcrc_data.collate`` returns CPU tensors; the frozen model may live on a
        GPU.  Moving here keeps every call site free of device bookkeeping and is
        a no-op when the batch is already on the requested device.
        """
        key = str(device)
        if key in self._moved:
            return self._moved[key]
        dev = torch.device(device)
        if dev.type == "cpu":
            self._moved[key] = self
            return self
        batch: dict[str, Any] = {}
        for name, value in self.batch.items():
            batch[name] = value.to(dev) if torch.is_tensor(value) else value
        moved = GatedBatch(
            batch=batch,
            n_graphs=self.n_graphs,
            bond_jump=self.bond_jump.to(dev),
            nbr_jump=self.nbr_jump.to(dev),
            nbr_bond=self.nbr_bond.to(dev),
            path_jump=self.path_jump.to(dev),
            path_bond1=self.path_bond1.to(dev),
            path_bond2=self.path_bond2.to(dev),
            bond_local=self.bond_local.to(dev),
            bond_counts=self.bond_counts,
            max_bonds=self.max_bonds,
            atom_jump=self.atom_jump.to(dev),
            occurrence_check=self.occurrence_check,
        )
        self._moved[key] = moved
        return moved

    def jump_from_graph(self, graph_index: np.ndarray) -> torch.Tensor:
        jumps = np.concatenate([[0], np.cumsum(self.bond_counts)]).astype(np.int64)
        return torch.from_numpy(jumps[graph_index])

    def gate_ones(self, n_graphs: int | None = None, *, device=None, dtype=torch.float32,
                  requires_grad: bool = False) -> torch.Tensor:
        n = int(self.n_graphs if n_graphs is None else n_graphs)
        return torch.ones((n, max(self.max_bonds, 1)), device=device, dtype=dtype, requires_grad=requires_grad)

    def gate_matrix(self, per_graph: Sequence[np.ndarray], *, device=None,
                    dtype=torch.float32, requires_grad: bool = False) -> torch.Tensor:
        """Pack per-graph bond-gate vectors into one padded matrix."""
        if len(per_graph) != self.n_graphs:
            raise ValueError(f"expected {self.n_graphs} gate vectors, got {len(per_graph)}")
        width = max(self.max_bonds, 1)
        out = np.zeros((self.n_graphs, width), dtype=np.float64)
        for g, vec in enumerate(per_graph):
            vec = np.asarray(vec, dtype=np.float64).reshape(-1)
            if vec.size != int(self.bond_counts[g]):
                raise ValueError(f"graph {g}: {vec.size} gates for {int(self.bond_counts[g])} bonds")
            out[g, : vec.size] = vec
        return torch.tensor(out, device=device, dtype=dtype, requires_grad=requires_grad)

    def s1_gate(self, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(bond gate aligned with bond rows, neighbour gate aligned with S1 codes)."""
        return M[self.bond_jump, self.bond_local], M[self.nbr_jump, self.nbr_bond]

    def s2_weight(self, M: torch.Tensor) -> torch.Tensor:
        return M[self.path_jump, self.path_bond1] * M[self.path_jump, self.path_bond2]


# Implementation guards, not research rules.  The frozen split replays legitimately
# batch a few hundred molecules at once (~4k atoms / 4k bonds); these caps sit far
# above that while still refusing a pathological allocation instead of OOM-ing.
MAX_BATCH_BONDS = 500_000
MAX_BATCH_ATOMS = 500_000


def collate_gated(cache: Mapping[str, Any], rows: Sequence[int], occ: SplitOcc,
                  *, collate_fn=None) -> GatedBatch:
    """``lcrc_data.collate`` plus verified S1/S2 occurrence bookkeeping."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    batch = (collate_fn or PREP_DATA.collate)(cache, rows)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)

    atom_idx, _atom_counts = _range_index(atom_ptr, rows)
    nbr_idx, nbr_counts = _range_index(nbr_ptr, atom_idx)
    path_idx, path_counts = _range_index(path_ptr, atom_idx)
    bond_idx, bond_counts = _range_index(np.asarray(cache["bond_ptr"], dtype=np.int64), rows)

    n_atoms_batch = int(batch["atom_feat"].shape[0])
    n_bonds_batch = int(batch["bond_index"].shape[0])
    if n_atoms_batch > MAX_BATCH_ATOMS or n_bonds_batch > MAX_BATCH_BONDS:
        raise RuntimeError(
            f"refusing a gated batch of {n_atoms_batch} atoms / {n_bonds_batch} bonds "
            f"(guards {MAX_BATCH_ATOMS}/{MAX_BATCH_BONDS}); this is an implementation "
            "memory guard, not a research rule -- inspect the molecule before raising it"
        )

    expected_nbr = int(batch["nbr_code"].numel())
    expected_path = int(batch["path_cat"].numel())
    if nbr_idx.size != expected_nbr or path_idx.size != expected_path:
        raise RuntimeError(
            f"occurrence index algebra disagrees with the frozen collate "
            f"(nbr {nbr_idx.size} vs {expected_nbr}, path {path_idx.size} vs {expected_path})"
        )
    nbr_bond = occ.nbr_bond[nbr_idx] if nbr_idx.size else np.zeros(0, dtype=np.int64)
    p1 = occ.path_bond1[path_idx] if path_idx.size else np.zeros(0, dtype=np.int64)
    p2 = occ.path_bond2[path_idx] if path_idx.size else np.zeros(0, dtype=np.int64)
    p_mid = occ.path_middle[path_idx] if path_idx.size else np.zeros(0, dtype=np.int64)

    atom_batch = np.asarray(batch["atom_batch"], dtype=np.int64)
    nbr_atom = np.asarray(batch["nbr_atom"], dtype=np.int64)
    path_atom = np.asarray(batch["path_atom"], dtype=np.int64)
    path_graph = atom_batch[path_atom] if path_atom.size else np.zeros(0, dtype=np.int64)
    nbr_graph = atom_batch[nbr_atom] if nbr_atom.size else np.zeros(0, dtype=np.int64)
    bond_graph = np.asarray(batch["bond_batch"], dtype=np.int64)

    # Cross-check the rebuilt bookkeeping against the frozen collate tensors.
    # All coordinates below are the *collated batch* coordinates that the frozen
    # model consumes, so the check validates exactly what the gates touch.
    checks: dict[str, Any] = {}
    bond_offsets = (
        np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64)
        if bond_counts.size else np.zeros(0, dtype=np.int64)
    )
    n_bond_types = int(occ.n_bond_types)
    atom_off = np.asarray(batch["atom_off"], dtype=np.int64)
    cache_atom_types = np.asarray(cache["atom_type"], dtype=np.int64)
    batch_bond_index = np.asarray(batch["bond_index"], dtype=np.int64)
    bond_type_rows = np.asarray(batch["bond_type"], dtype=np.int64)
    nbr_atom_types = cache_atom_types[atom_idx]  # batch atom -> frozen atom type
    if nbr_idx.size:
        nbr_root = nbr_atom
        nbr_row = bond_offsets[nbr_graph] + nbr_bond
        endpoint_a = batch_bond_index[nbr_row, 0]
        endpoint_b = batch_bond_index[nbr_row, 1]
        checks["nbr_bond_incident_to_root"] = bool(
            np.all((endpoint_a == nbr_root) | (endpoint_b == nbr_root))
        )
        other = np.where(endpoint_a == nbr_root, endpoint_b, endpoint_a)
        rebuilt_code = nbr_atom_types[other] * n_bond_types + (bond_type_rows[nbr_row] - 1)
        checks["nbr_code_rebuilt_from_bond"] = bool(
            np.array_equal(rebuilt_code, np.asarray(batch["nbr_code"], dtype=np.int64))
        )
        checks["nbr_code_matches_cache"] = bool(
            np.array_equal(np.asarray(batch["nbr_code"], dtype=np.int64),
                           np.asarray(cache["nbr_code"], dtype=np.int64)[nbr_idx])
        )
        checks["nbr_root_order_matches_adjacency"] = bool(
            np.array_equal(nbr_root, np.repeat(np.arange(atom_idx.size), nbr_counts))
        )
    if path_idx.size:
        p1_row = bond_offsets[path_graph] + p1
        p2_row = bond_offsets[path_graph] + p2
        root = path_atom
        e1a, e1b = batch_bond_index[p1_row, 0], batch_bond_index[p1_row, 1]
        e2a, e2b = batch_bond_index[p2_row, 0], batch_bond_index[p2_row, 1]
        mid = np.where(e1a == root, e1b, e1a)
        end = np.where(e2a == mid, e2b, e2a)
        checks["path_chain_root_middle_end"] = bool(
            np.all((e1a == root) | (e1b == root))
            and np.all((e2a == mid) | (e2b == mid))
            and np.all(mid != root)
            and np.all(end != root)
        )
        checks["path_end_matches_cache"] = bool(
            np.array_equal(end, np.asarray(batch["path_end"], dtype=np.int64))
        )
        rebuilt_cat = (nbr_atom_types[mid] * n_bond_types * n_bond_types
                       + (bond_type_rows[p1_row] - 1) * n_bond_types
                       + (bond_type_rows[p2_row] - 1))
        checks["path_cat_rebuilt_from_bonds"] = bool(
            np.array_equal(rebuilt_cat, np.asarray(batch["path_cat"], dtype=np.int64))
        )
        checks["path_end63_rebuilt_from_bonds"] = bool(
            np.array_equal(nbr_atom_types[end] * n_bond_types + (bond_type_rows[p2_row] - 1),
                           np.asarray(batch["path_end63"], dtype=np.int64))
        )
        checks["path_cat_matches_cache"] = bool(
            np.array_equal(np.asarray(batch["path_cat"], dtype=np.int64),
                           np.asarray(cache["path_cat"], dtype=np.int64)[path_idx])
        )
        checks["path_closure_matches_cache"] = bool(
            np.array_equal(np.asarray(batch["path_closure"], dtype=np.int64),
                           np.asarray(cache["path_closure"], dtype=np.int64)[path_idx])
        )
        checks["path_middle_matches_rebuilt_chain"] = bool(
            np.array_equal(mid, atom_off[path_graph] + p_mid)
        )
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"gated collate cross-checks failed: {failed}")

    # local bond id per collated bond row
    bond_local = (
        np.concatenate([np.arange(int(c), dtype=np.int64) for c in bond_counts])
        if bond_counts.size
        else np.zeros(0, dtype=np.int64)
    )
    return GatedBatch(
        batch=batch,
        n_graphs=int(batch["n_graphs"]),
        bond_jump=torch.from_numpy(bond_graph.astype(np.int64)),
        nbr_jump=torch.from_numpy(nbr_graph.astype(np.int64)),
        nbr_bond=torch.from_numpy(nbr_bond.astype(np.int64)),
        path_jump=torch.from_numpy(path_graph.astype(np.int64)),
        path_bond1=torch.from_numpy(p1.astype(np.int64)),
        path_bond2=torch.from_numpy(p2.astype(np.int64)),
        bond_local=torch.from_numpy(bond_local.astype(np.int64)),
        bond_counts=np.asarray(bond_counts, dtype=np.int64),
        max_bonds=int(np.max(bond_counts)) if bond_counts.size else 0,
        atom_jump=torch.from_numpy(atom_batch.astype(np.int64)),
        occurrence_check={"checks": checks, "n_failed": 0, "nbr_idx_size": int(nbr_idx.size),
                          "path_idx_size": int(path_idx.size)},
    )


__all__ = [
    "slice_molecule",
    "analysis_from_mol",
    "OccMap",
    "build_occ_map",
    "verify_occurrences",
    "SplitOcc",
    "build_split_occ",
    "GatedBatch",
    "collate_gated",
    "_MOL_KEYS",
]


def molecule_cache(mol: Mapping[str, Any], occ: "OccMap", *,
                   drop_paths: np.ndarray | None = None) -> tuple[dict[str, Any], "SplitOcc"]:
    """A one-molecule cache dict (and its ``SplitOcc``) from analysed arrays.

    Used by the correctness tests: an atom permutation or an explicit removal of
    individual S2 occurrences is expressed as a *different molecule*, then fed
    through exactly the same frozen collate + gated forward as the real data.
    """
    n_atoms = int(occ.n_atoms)
    n_bonds = int(occ.n_bonds)
    keep = np.ones(occ.n_path, dtype=bool) if drop_paths is None else ~np.asarray(drop_paths, dtype=bool)
    if keep.size != occ.n_path:
        raise ValueError(f"drop_paths has {keep.size} entries for {occ.n_path} occurrences")
    p_cat = occ.path_cat[keep]
    p_end63 = occ.path_end63[keep]
    p_end = occ.path_end[keep]
    p_close = occ.path_closure[keep]
    p_root = occ.path_root[keep]
    counts = np.bincount(p_root, minlength=n_atoms).astype(np.int64)
    path_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    glob = np.asarray(mol["global"], dtype=np.float32).reshape(1, -1)
    cache = {
        "atom_ptr": np.asarray([0, n_atoms], dtype=np.int64),
        "bond_ptr": np.asarray([0, n_bonds], dtype=np.int64),
        "nbr_ptr": np.asarray(occ.nbr_ptr, dtype=np.int64),
        "path_ptr": path_ptr,
        "atom_feat": np.asarray(mol["atom_feat"], dtype=np.float32),
        "atom_type": np.asarray(mol["atom_type"], dtype=np.int64),
        "atom_scalar": np.asarray(mol["atom_scalar"], dtype=np.float32),
        "bond_feat": np.asarray(mol["bond_feat"], dtype=np.float32),
        "bond_type": np.asarray(mol["bond_type"], dtype=np.int64),
        "bond_index": np.asarray(mol["bond_index"], dtype=np.int64).reshape(-1, 2),
        "nbr_code": np.asarray(occ.nbr_code, dtype=np.int32),
        "path_cat": np.asarray(p_cat, dtype=np.int32),
        "path_end63": np.asarray(p_end63, dtype=np.int32),
        "path_end": np.asarray(p_end, dtype=np.int64),
        "path_closure": np.asarray(p_close, dtype=np.int8),
        "global": glob,
        "n_molecules": 1,
        "smiles": [str(mol.get("smiles", ""))],
        "molecule_ids": [str(mol.get("molecule_id", "0"))],
    }
    split_occ = SplitOcc(
        dataset="synthetic",
        split="single",
        n_molecules=1,
        nbr_bond=np.asarray(occ.nbr_bond, dtype=np.int64),
        path_bond1=np.asarray(occ.path_bond1[keep], dtype=np.int64),
        path_bond2=np.asarray(occ.path_bond2[keep], dtype=np.int64),
        path_middle=np.asarray(occ.path_middle[keep], dtype=np.int64),
        bond_counts=np.asarray([n_bonds], dtype=np.int64),
        atom_counts=np.asarray([n_atoms], dtype=np.int64),
        n_s2_total=np.asarray([int(keep.sum())], dtype=np.int64),
        n_bond_types=5,
        sha256="",
    )
    return cache, split_occ


__all__.append("molecule_cache")


def replicate_molecule(cache: Mapping[str, Any], row: int, n_copies: int,
                       split_occ: "SplitOcc") -> tuple[dict[str, Any], "SplitOcc"]:
    """The same cached molecule repeated ``n_copies`` times as distinct batch rows.

    The frozen ``lcrc_data.collate`` rejects a duplicated row index, and the
    intervention stage needs every mask variant of one molecule in a single
    batched forward.  Replication keeps the frozen collate (and therefore the
    exact graph-isolation behaviour that was verified for it) untouched: each
    replica is an independent graph with its own gate row.
    """
    mol = slice_molecule(cache, int(row))
    n = int(n_copies)
    if n < 1:
        raise ValueError("n_copies must be >= 1")
    n_atoms, n_bonds = int(mol["n_atoms"]), int(mol["n_bonds"])
    n_nbr = int(mol["nbr_code"].size)
    n_path = int(mol["path_cat"].size)

    def _tile(arr, dtype):
        arr = np.asarray(arr, dtype=dtype)
        return np.tile(arr, (n,) + (1,) * (arr.ndim - 1))

    a_ptr = np.arange(n + 1, dtype=np.int64) * n_atoms
    b_ptr = np.arange(n + 1, dtype=np.int64) * n_bonds
    nbr_ptr = np.concatenate([[0], (mol["nbr_ptr_local"][1:][None, :]
                                    + np.arange(n, dtype=np.int64)[:, None] * n_nbr).ravel()]).astype(np.int64)
    path_ptr = np.concatenate([[0], (mol["path_ptr_local"][1:][None, :]
                                     + np.arange(n, dtype=np.int64)[:, None] * n_path).ravel()]).astype(np.int64)
    rep = {
        "atom_ptr": a_ptr,
        "bond_ptr": b_ptr,
        "nbr_ptr": nbr_ptr,
        "path_ptr": path_ptr,
        "atom_feat": _tile(mol["atom_feat"], np.float32),
        "atom_type": _tile(mol["atom_type"], np.int64),
        "atom_scalar": _tile(mol["atom_scalar"], np.float32),
        "bond_feat": _tile(mol["bond_feat"], np.float32),
        "bond_type": _tile(mol["bond_type"], np.int64),
        "bond_index": _tile(mol["bond_index"], np.int64),
        "nbr_code": np.tile(np.asarray(mol["nbr_code"], dtype=np.int32), n),
        "path_cat": np.tile(np.asarray(mol["path_cat"], dtype=np.int32), n),
        "path_end63": np.tile(np.asarray(mol["path_end63"], dtype=np.int32), n),
        "path_end": np.tile(np.asarray(mol["path_end"], dtype=np.int64), n),
        "path_closure": np.tile(np.asarray(mol["path_closure"], dtype=np.int8), n),
        "global": np.tile(np.asarray(mol["global"], dtype=np.float32).reshape(1, -1), (n, 1)),
        "n_molecules": n,
        "smiles": [mol["smiles"]] * n,
        "molecule_ids": [mol["molecule_id"]] * n,
    }
    # SplitOcc per split-local row: keep only this molecule's occurrence block.
    a0, a1 = int(cache["atom_ptr"][row]), int(cache["atom_ptr"][row + 1])
    n0, n1 = int(cache["nbr_ptr"][a0]), int(cache["nbr_ptr"][a1])
    p0, p1 = int(cache["path_ptr"][a0]), int(cache["path_ptr"][a1])
    occ_rep = SplitOcc(
        dataset=split_occ.dataset,
        split=split_occ.split,
        n_molecules=n,
        nbr_bond=np.tile(np.asarray(split_occ.nbr_bond[n0:n1], dtype=np.int64), n),
        path_bond1=np.tile(np.asarray(split_occ.path_bond1[p0:p1], dtype=np.int64), n),
        path_bond2=np.tile(np.asarray(split_occ.path_bond2[p0:p1], dtype=np.int64), n),
        path_middle=np.tile(np.asarray(split_occ.path_middle[p0:p1], dtype=np.int64), n),
        bond_counts=np.full(n, n_bonds, dtype=np.int64),
        atom_counts=np.full(n, n_atoms, dtype=np.int64),
        n_s2_total=np.full(n, n_path, dtype=np.int64),
        n_bond_types=int(split_occ.n_bond_types),
        sha256=split_occ.sha256,
    )
    return rep, occ_rep


__all__.append("replicate_molecule")
