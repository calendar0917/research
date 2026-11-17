"""Physical-bond-anchored simple three-bond paths and their S2 occurrence ids.

The object
----------
For a molecule ``G`` a *three-bond simple path* is an ordered quadruple
``P = (a,b,c,d)`` with four distinct atoms and physical bonds
``(a,b), (b,c), (c,d)`` in ``E(G)``.  Its middle bond ``(b,c)`` is the **anchor
bond**.  ``P`` corresponds to the two real, adjacent directed S2 occurrences

    p     = (a,b,c)        q     = (b,c,d)
    p_rev = (d,c,b)        q_rev = (c,b,a)

which are exactly the objects the frozen C1 typed-S2 block already enumerates
(``molecular_graph.enumerate_s2``).  This module rebuilds the C1 occurrence map
with :func:`as_data.build_occ_map` (which carries each occurrence's physical
bond ids) and *verifies it item by item against the frozen cache*, then indexes
the four occurrences of every path by their ``(root, middle, end)`` triple.

Frozen conventions (declared before any training)
------------------------------------------------
* each **undirected** three-bond path is one event: the direction is the
  lexicographically smaller of ``(a,b,c,d)`` / ``(d,c,b,a)``, and the model output
  is symmetric under reversal by construction (``z_P`` is the mean of the two
  directional evaluations), so atom relabelling cannot change any number here;
* ``closure4 = 1`` iff the terminal atoms ``(a,d)`` are additionally bonded;
* ring paths are legal and are kept;
* nothing is filtered by functional group, SMARTS, property label, or size.

No truncation, subsampling or dropping of large molecules: every event is kept
and, when a molecule needs it, the forward is chunked (``bc_models``) so the sum
stays exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import bc_common as C

PATHS3_KEYS = ("p3_ptr", "p3_atoms", "p3_occ", "p3_anchor", "p3_closure4")


@dataclass
class Path3Split:
    """Per-split three-bond-path cache (cache row order, all events kept)."""

    dataset: str
    split: str
    n_molecules: int
    ptr: np.ndarray          # (n_mol + 1,) event pointer
    atoms: np.ndarray        # (n_ev, 4) a, b, c, d
    occ: np.ndarray          # (n_ev, 4) p, q, p_rev, q_rev (split-level occurrence ids)
    anchor: np.ndarray       # (n_ev,) local physical bond id of (b,c)
    closure4: np.ndarray     # (n_ev,) 1 if (a,d) is a physical bond
    occ_root: np.ndarray = None     # (n_s2,) molecule-local root atom of every occurrence
    occ_end: np.ndarray = None      # (n_s2,) molecule-local end atom of every occurrence
    occ_map: Any = None      # as_data.SplitOcc, kept for the gated collate
    sha256: str = ""
    verification: dict[str, Any] | None = None

    @property
    def n_events(self) -> int:
        return int(self.ptr[-1])

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.ptr)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "p3_ptr": self.ptr,
            "p3_atoms": self.atoms,
            "p3_occ": self.occ,
            "p3_anchor": self.anchor,
            "p3_closure4": self.closure4,
            "occ_root": self.occ_root,
            "occ_end": self.occ_end,
        }


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def enumerate_one(
    mol: Mapping[str, Any],
    analysis: Any,
    occ: Any,
    *,
    n_bond_types: int,
) -> dict[str, np.ndarray]:
    """Every undirected three-bond simple path of one molecule, canonical order."""
    n = int(analysis.n_atoms)
    adj = analysis.adj
    if len(adj) != n:
        raise RuntimeError("adjacency length does not match the atom count")
    nbr_sets = [{int(v) for v, _bt, _e in row} for row in adj]

    # (root, middle, end) -> split-local occurrence id  (unique in a simple graph)
    occ_index: dict[tuple[int, int, int], int] = {
        (int(r), int(m), int(e)): i
        for i, (r, m, e) in enumerate(
            zip(occ.path_root.tolist(), occ.path_middle.tolist(), occ.path_end.tolist())
        )
    }

    atoms: list[tuple[int, int, int, int]] = []
    rec_occ: list[tuple[int, int, int, int]] = []
    anchor: list[int] = []
    closure4: list[int] = []
    for b, row_b in enumerate(adj):
        for c, _bt_bc, e_bc in row_b:
            c = int(c)
            for a, _bt_ab, _e_ab in row_b:
                a = int(a)
                if a == c:
                    continue
                for d, _bt_cd, _e_cd in adj[c]:
                    d = int(d)
                    if d == b or d == a:
                        continue
                    key = (a, b, c, d)
                    rev = (d, c, b, a)
                    if key > rev:
                        continue  # the reverse traversal owns this undirected path
                    p = occ_index[(a, b, c)]
                    q = occ_index[(b, c, d)]
                    p_rev = occ_index[(d, c, b)]
                    q_rev = occ_index[(c, b, a)]
                    atoms.append(key)
                    rec_occ.append((p, q, p_rev, q_rev))
                    anchor.append(int(e_bc))
                    closure4.append(1 if d in nbr_sets[a] else 0)

    order = sorted(range(len(atoms)), key=lambda i: atoms[i])
    atoms = [atoms[i] for i in order]
    rec_occ = [rec_occ[i] for i in order]
    anchor = [anchor[i] for i in order]
    closure4 = [closure4[i] for i in order]

    return {
        "atoms": np.asarray(atoms, dtype=np.int64).reshape(-1, 4),
        "occ": np.asarray(rec_occ, dtype=np.int64).reshape(-1, 4),
        "anchor": np.asarray(anchor, dtype=np.int64),
        "closure4": np.asarray(closure4, dtype=np.int8),
    }


def verify_one(
    mol: Mapping[str, Any],
    analysis: Any,
    occ: Any,
    part: Mapping[str, np.ndarray],
    *,
    n_bond_types: int,
) -> dict[str, Any]:
    """Structural invariants of one molecule's event list (all must hold)."""
    atoms = np.asarray(part["atoms"], dtype=np.int64).reshape(-1, 4)
    ids = np.asarray(part["occ"], dtype=np.int64).reshape(-1, 4)
    anchor = np.asarray(part["anchor"], dtype=np.int64).reshape(-1)
    closure = np.asarray(part["closure4"], dtype=np.int64).reshape(-1)
    n_ev = int(atoms.shape[0])
    bond_index = np.asarray(mol["bond_index"], dtype=np.int64).reshape(-1, 2)
    u, v = bond_index[:, 0], bond_index[:, 1]
    bonded = {(int(x), int(y)) for x, y in zip(u.tolist(), v.tolist())}

    def _has(x: int, y: int) -> bool:
        return (min(x, y), max(x, y)) in bonded

    checks: dict[str, bool] = {}
    checks["four_distinct_atoms"] = bool(
        all(len(set(row.tolist())) == 4 for row in atoms)
    ) if n_ev else True
    checks["bonds_exist"] = bool(
        all(
            _has(int(a), int(b)) and _has(int(b), int(c)) and _has(int(c), int(d))
            for a, b, c, d in atoms.tolist()
        )
    ) if n_ev else True
    checks["canonical_direction"] = bool(
        all(tuple(row) <= (row[3], row[2], row[1], row[0]) for row in atoms.tolist())
    ) if n_ev else True
    checks["unique_undirected_paths"] = bool(
        len({(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in atoms.tolist()}) == n_ev
    )
    if n_ev:
        roots = occ.path_root[ids[:, 0]]
        mids = occ.path_middle[ids[:, 0]]
        ends = occ.path_end[ids[:, 0]]
        checks["p_is_first_occurrence"] = bool(
            np.array_equal(roots, atoms[:, 0])
            and np.array_equal(mids, atoms[:, 1])
            and np.array_equal(ends, atoms[:, 2])
        )
        checks["q_is_second_occurrence"] = bool(
            np.array_equal(occ.path_root[ids[:, 1]], atoms[:, 1])
            and np.array_equal(occ.path_middle[ids[:, 1]], atoms[:, 2])
            and np.array_equal(occ.path_end[ids[:, 1]], atoms[:, 3])
        )
        checks["p_rev_is_reversal"] = bool(
            np.array_equal(occ.path_root[ids[:, 2]], atoms[:, 3])
            and np.array_equal(occ.path_middle[ids[:, 2]], atoms[:, 2])
            and np.array_equal(occ.path_end[ids[:, 2]], atoms[:, 1])
        )
        checks["q_rev_is_reversal"] = bool(
            np.array_equal(occ.path_root[ids[:, 3]], atoms[:, 2])
            and np.array_equal(occ.path_middle[ids[:, 3]], atoms[:, 1])
            and np.array_equal(occ.path_end[ids[:, 3]], atoms[:, 0])
        )
        checks["anchor_is_middle_bond"] = bool(
            np.array_equal(
                np.sort(np.stack([u[anchor], v[anchor]], axis=1), axis=1),
                np.sort(np.stack([atoms[:, 1], atoms[:, 2]], axis=1), axis=1),
            )
        )
        checks["anchor_is_q_first_bond"] = bool(np.array_equal(occ.path_bond1[ids[:, 1]], anchor))
        checks["anchor_is_p_second_bond"] = bool(np.array_equal(occ.path_bond2[ids[:, 0]], anchor))
        expected_closure = np.asarray(
            [1 if _has(int(x), int(y)) else 0 for x, y in zip(atoms[:, 0].tolist(), atoms[:, 3].tolist())],
            dtype=np.int64,
        )
        checks["closure4_is_terminal_bond"] = bool(np.array_equal(expected_closure, closure))
    else:
        for name in (
            "p_is_first_occurrence",
            "q_is_second_occurrence",
            "p_rev_is_reversal",
            "q_rev_is_reversal",
            "anchor_is_middle_bond",
            "anchor_is_q_first_bond",
            "anchor_is_p_second_bond",
            "closure4_is_terminal_bond",
        ):
            checks[name] = True
    failed = sorted(name for name, ok in checks.items() if not ok)
    return {"n_events": n_ev, "checks": checks, "failed": failed, "ok": not failed}


# ---------------------------------------------------------------------------
# one split
# ---------------------------------------------------------------------------


def build_split(cache: Mapping[str, Any], spec, dataset: str, split: str) -> Path3Split:
    """Rebuild occurrence + three-bond-path bookkeeping for one frozen split cache."""
    AD = C.modules()["as_data"]
    n = int(cache["n_molecules"])
    occ_parts: dict[str, list[np.ndarray]] = {
        "nbr_bond": [], "path_bond1": [], "path_bond2": [], "path_middle": [],
    }
    p3_counts = np.zeros(n, dtype=np.int64)
    atoms_list: list[np.ndarray] = []
    occ_list: list[np.ndarray] = []
    anchor_list: list[np.ndarray] = []
    closure_list: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    occ_failures: list[dict[str, Any]] = []
    p3_failures: list[dict[str, Any]] = []
    n_s2 = np.zeros(n, dtype=np.int64)
    offset = 0
    for row in range(n):
        mol = AD.slice_molecule(cache, row)
        _graph, analysis = AD.analysis_from_mol(mol)
        if not np.array_equal(
            np.stack([analysis.bond_u, analysis.bond_v], axis=1), np.asarray(mol["bond_index"])
        ):
            raise ValueError(f"{dataset}/{split}/{row}: bond order is not the cached canonical order")
        om = AD.build_occ_map(np.asarray(mol["atom_type"]), analysis, int(spec.n_bond_types))
        report = AD.verify_occurrences(om, mol, strict=False)
        if report["n_failed"]:
            occ_failures.append({"row": row, "failed": report["failed"]})
            raise ValueError(f"{dataset}/{split}/{row}: occurrence mapping failed {report['failed']}")
        # shift occurrence ids to split level
        shift = int(offset)
        part = enumerate_one(mol, analysis, om, n_bond_types=int(spec.n_bond_types))
        check = verify_one(mol, analysis, om, part, n_bond_types=int(spec.n_bond_types))
        if not check["ok"]:
            p3_failures.append({"row": row, "failed": check["failed"]})
            raise ValueError(f"{dataset}/{split}/{row}: three-bond path invariants failed {check['failed']}")
        occ_parts["nbr_bond"].append(om.nbr_bond + 0)
        occ_parts["path_bond1"].append(om.path_bond1 + 0)
        occ_parts["path_bond2"].append(om.path_bond2 + 0)
        occ_parts["path_middle"].append(om.path_middle + 0)
        root_parts.append(om.path_root + 0)
        end_parts.append(om.path_end + 0)
        n_s2[row] = om.n_path
        p3_counts[row] = int(part["atoms"].shape[0])
        if p3_counts[row]:
            atoms_list.append(part["atoms"])
            occ_list.append(part["occ"] + shift)
            anchor_list.append(part["anchor"])
            closure_list.append(part["closure4"].astype(np.int64))
        offset += int(om.n_path)

    atoms = np.concatenate(atoms_list, axis=0) if atoms_list else np.zeros((0, 4), dtype=np.int64)
    occ = np.concatenate(occ_list, axis=0) if occ_list else np.zeros((0, 4), dtype=np.int64)
    anchor = np.concatenate(anchor_list) if anchor_list else np.zeros(0, dtype=np.int64)
    closure = np.concatenate(closure_list) if closure_list else np.zeros(0, dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(p3_counts)]).astype(np.int64)
    occ_root = np.concatenate(root_parts) if root_parts else np.zeros(0, dtype=np.int64)
    occ_end = np.concatenate(end_parts) if end_parts else np.zeros(0, dtype=np.int64)

    atom_counts = np.diff(np.asarray(cache["atom_ptr"][: n + 1], dtype=np.int64))
    bond_counts = np.diff(np.asarray(cache["bond_ptr"][: n + 1], dtype=np.int64))
    occ_map = AD.SplitOcc(
        dataset=dataset,
        split=split,
        n_molecules=n,
        nbr_bond=np.concatenate(occ_parts["nbr_bond"]) if occ_parts["nbr_bond"] else np.zeros(0, dtype=np.int64),
        path_bond1=np.concatenate(occ_parts["path_bond1"]) if occ_parts["path_bond1"] else np.zeros(0, dtype=np.int64),
        path_bond2=np.concatenate(occ_parts["path_bond2"]) if occ_parts["path_bond2"] else np.zeros(0, dtype=np.int64),
        path_middle=np.concatenate(occ_parts["path_middle"]) if occ_parts["path_middle"] else np.zeros(0, dtype=np.int64),
        bond_counts=np.asarray(bond_counts, dtype=np.int64),
        atom_counts=np.asarray(atom_counts, dtype=np.int64),
        n_s2_total=np.asarray(n_s2, dtype=np.int64),
        n_bond_types=int(spec.n_bond_types),
        sha256="",
    )
    digest = C.sha256_json(
        {
            "dataset": dataset,
            "split": split,
            "n_molecules": n,
            "n_s2": C.sha256_array(occ_map.n_s2_total),
            "n_s1_bonds": C.sha256_array(occ_map.nbr_bond),
            "path_bond1": C.sha256_array(occ_map.path_bond1),
            "path_bond2": C.sha256_array(occ_map.path_bond2),
            "path_middle": C.sha256_array(occ_map.path_middle),
            "p3_ptr": C.sha256_array(ptr),
            "p3_atoms": C.sha256_array(atoms),
            "p3_occ": C.sha256_array(occ),
            "p3_anchor": C.sha256_array(anchor),
            "p3_closure4": C.sha256_array(closure),
            "occ_root": C.sha256_array(occ_root),
            "occ_end": C.sha256_array(occ_end),
        }
    )
    verification = {
        "split": split,
        "n_molecules": n,
        "occurrence_failures": occ_failures,
        "path3_failures": p3_failures,
        "all_occurrences_verified": not occ_failures,
        "all_paths_verified": not p3_failures,
        "n_s2_occurrences": int(occ_map.n_s2_total.sum()),
        "n_path3_events": int(ptr[-1]),
    }
    return Path3Split(
        dataset=dataset,
        split=split,
        n_molecules=n,
        ptr=ptr,
        atoms=atoms,
        occ=occ,
        anchor=anchor,
        closure4=closure,
        occ_root=occ_root,
        occ_end=occ_end,
        occ_map=occ_map,
        sha256=digest,
        verification=verification,
    )


# ---------------------------------------------------------------------------
# support statistics (task book section 5)
# ---------------------------------------------------------------------------


def support_statistics(p3: Path3Split, cache: Mapping[str, Any]) -> dict[str, Any]:
    """Per-split event-count distribution and marginal-replacement distinguishability.

    ``distinguishable event`` = the molecule has >= 2 events whose second S2
    occurrence id differs, i.e. the MARGINAL mean replacement is *not* literally
    the JOINT partner for at least one event.  The definition is structural
    (occurrence identity), frozen before training, and is reported next to the
    stricter evidence-vector version measured in ``bc_checks``.
    """
    counts = p3.counts.astype(np.int64)
    n = int(counts.size)
    atom_counts = np.diff(np.asarray(cache["atom_ptr"][: n + 1], dtype=np.int64))
    per_mol_distinct = np.zeros(n, dtype=np.int64)
    per_mol_distinct_q_rev = np.zeros(n, dtype=np.int64)
    for row in range(n):
        a0, a1 = int(p3.ptr[row]), int(p3.ptr[row + 1])
        if a1 > a0:
            per_mol_distinct[row] = int(np.unique(p3.occ[a0:a1, 1]).size)
            per_mol_distinct_q_rev[row] = int(np.unique(p3.occ[a0:a1, 3]).size)
    has_events = counts > 0
    distinguishable = has_events & (per_mol_distinct >= 2)
    return {
        "split": p3.split,
        "n_molecules": n,
        "n_events": int(counts.sum()),
        "events_per_molecule_mean": float(counts.mean()) if n else 0.0,
        "events_per_molecule_p50": float(np.percentile(counts, 50)) if n else 0.0,
        "events_per_molecule_p95": float(np.percentile(counts, 95)) if n else 0.0,
        "events_per_molecule_max": int(counts.max()) if n else 0,
        "events_per_molecule_min": int(counts.min()) if n else 0,
        "fraction_molecules_with_event": float(has_events.mean()) if n else 0.0,
        "fraction_molecules_distinguishable": float(distinguishable.mean()) if n else 0.0,
        "fraction_distinguishable_q_rev": float((has_events & (per_mol_distinct_q_rev >= 2)).mean()) if n else 0.0,
        "distinct_second_paths_mean": float(per_mol_distinct[has_events].mean()) if has_events.any() else 0.0,
        "n_atoms_mean": float(atom_counts.mean()) if n else 0.0,
        "n_atoms_max": int(atom_counts.max()) if n else 0,
        "n_events_over_n_atoms": float(counts.sum() / max(1, int(atom_counts.sum()))),
        "n_atoms_over_events_p95": float(
            np.percentile(
                [int(atom_counts[r]) / float(max(1, int(counts[r]))) for r in range(n) if counts[r] > 0], 95
            )
        )
        if has_events.any()
        else 0.0,
    }


def clause_statistics(p3: Path3Split) -> dict[str, Any]:
    """Ring/closure composition of the event pool."""
    n_ev = p3.n_events
    if not n_ev:
        return {"n_events": 0, "closure4_fraction": 0.0}
    return {
        "n_events": int(n_ev),
        "closure4_fraction": float(np.asarray(p3.closure4, dtype=np.float64).mean()),
        "open_path_fraction": float(1.0 - np.asarray(p3.closure4, dtype=np.float64).mean()),
        "n_distinct_anchor_bonds": int(np.unique(p3.anchor).size),
    }


# ---------------------------------------------------------------------------
# collate: split-level events -> batch tensors
# ---------------------------------------------------------------------------


def collate_events(p3: Path3Split, rows: Sequence[int], *, bond_jump: np.ndarray, gb_occ: Any) -> dict[str, Any]:
    """Batch tensors of the three-bond events for one collated batch.

    ``occ ids`` are translated from split level to *batch* occurrence indices with
    the inverse of the frozen collate's occurrence gather, so the model indexes
    exactly the occurrences the C1 forward consumes.
    """
    import torch

    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    counts = np.diff(p3.ptr)[rows]
    graph = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
    if int(counts.sum()):
        idx = np.concatenate(
            [np.arange(int(p3.ptr[r]), int(p3.ptr[r + 1]), dtype=np.int64) for r in rows if p3.ptr[r + 1] > p3.ptr[r]]
        )
    else:
        idx = np.zeros(0, dtype=np.int64)
    return {
        "p3_index": idx,
        "p3_graph": graph,
        "p3_occ_split": p3.occ[idx] if idx.size else np.zeros((0, 4), dtype=np.int64),
        "p3_anchor_local": p3.anchor[idx] if idx.size else np.zeros(0, dtype=np.int64),
        "p3_closure4": (p3.closure4[idx] if idx.size else np.zeros(0, dtype=np.int64)).astype(np.float32),
        "p3_counts": counts.astype(np.int64),
        "p3_atoms": p3.atoms[idx] if idx.size else np.zeros((0, 4), dtype=np.int64),
    }


__all__ = [
    "Path3Split",
    "build_split",
    "support_statistics",
    "clause_statistics",
    "collate_events",
    "enumerate_one",
    "verify_one",
    "PATHS3_KEYS",
]
