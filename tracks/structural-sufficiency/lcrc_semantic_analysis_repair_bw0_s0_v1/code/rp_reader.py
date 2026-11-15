"""Corrected per-molecule occurrence reader + document-frequency oracle.

The predecessor bug
-------------------
In a concatenated LCRC cache the pointer arrays have **two different levels**:

* ``atom_ptr`` / ``bond_ptr`` are ``molecule -> atom/bond`` boundaries
  (length ``n_molecules + 1``);
* ``path_ptr`` / ``nbr_ptr`` are ``atom -> occurrence`` boundaries
  (length ``n_atoms + 1``).

The old ``se_data.code_support`` / ``se_audit.support_counts`` / ``se_mechanism``
/ ``se_explicit.build_matrix`` indexed the **atom-level** ``path_ptr`` /
``nbr_ptr`` with a **molecule** index (``path_ptr[m:m+2]``).  For the Lipo Train
split that reads only the first ``n_molecules`` atoms' worth of occurrences
(``path_ptr[3360] = 10238`` instead of the true ``277312``), so the support and
explicit-statistics coordinate were computed on ~3.7% of the data.

This module is the single correct entry point.  Every reader asserts the two
level identities and is cross-checked against the independent
``lcrc_data.build_molecule`` oracle and against ``lcrc_data.collate``.

Per molecule ``m``::

    a0, a1 = atom_ptr[m], atom_ptr[m+1]
    p0, p1 = path_ptr[a0], path_ptr[a1]
    n0, n1 = nbr_ptr[a0],  nbr_ptr[a1]

    path_cat/path_end63/path_closure = [p0:p1]
    nbr_code                         = [n0:n1]
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

import rp_common as RP

SPLITS = RP.SPLITS


# ---------------------------------------------------------------------------
# layout invariants
# ---------------------------------------------------------------------------


def check_layout(cache: Mapping[str, Any], *, name: str = "") -> dict[str, Any]:
    """Assert the two pointer levels and the total-length identities."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    n_mol = int(cache["n_molecules"])
    n_atoms = int(atom_ptr[-1])
    errs: list[str] = []
    if atom_ptr.shape != (n_mol + 1,):
        errs.append(f"atom_ptr {atom_ptr.shape} != ({n_mol + 1},)")
    if bond_ptr.shape != (n_mol + 1,):
        errs.append(f"bond_ptr {bond_ptr.shape} != ({n_mol + 1},)")
    if path_ptr.shape != (n_atoms + 1,):
        errs.append(f"path_ptr {path_ptr.shape} != ({n_atoms + 1},) [atom-level]")
    if nbr_ptr.shape != (n_atoms + 1,):
        errs.append(f"nbr_ptr {nbr_ptr.shape} != ({n_atoms + 1},) [atom-level]")
    path_cat = np.asarray(cache["path_cat"])
    e63 = np.asarray(cache["path_end63"])
    clo = np.asarray(cache["path_closure"])
    nbr = np.asarray(cache["nbr_code"])
    if not (path_cat.shape == e63.shape == clo.shape == (int(path_ptr[-1]),)):
        errs.append("path occurrence array lengths disagree with path_ptr[-1]")
    if nbr.shape != (int(nbr_ptr[-1]),):
        errs.append("nbr_code length != nbr_ptr[-1]")
    # per-molecule sums reproduce the totals
    tot_path = 0
    tot_s1 = 0
    for m in range(n_mol):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        tot_path += int(path_ptr[a1] - path_ptr[a0])
        tot_s1 += int(nbr_ptr[a1] - nbr_ptr[a0])
        bonds = int(bond_ptr[m + 1] - bond_ptr[m])
        if int(nbr_ptr[a1] - nbr_ptr[a0]) != 2 * bonds:
            errs.append(f"molecule {m}: S1 count != 2*bonds")
            break
    if tot_path != int(path_ptr[-1]):
        errs.append(f"sum molecule paths {tot_path} != path_ptr[-1] {int(path_ptr[-1])}")
    if tot_s1 != int(nbr_ptr[-1]):
        errs.append(f"sum molecule S1 {tot_s1} != nbr_ptr[-1] {int(nbr_ptr[-1])}")
    if np.any(atom_ptr[1:] < atom_ptr[:-1]) or np.any(path_ptr[1:] < path_ptr[:-1]) or np.any(nbr_ptr[1:] < nbr_ptr[:-1]):
        errs.append("a pointer is not monotone")
    if errs:
        raise ValueError(f"cache layout invalid ({name}): {errs}")
    return {
        "n_molecules": n_mol,
        "n_atoms": n_atoms,
        "n_paths": int(path_ptr[-1]),
        "n_s1": int(nbr_ptr[-1]),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# the single correct reader
# ---------------------------------------------------------------------------


class MoleculeReader:
    """Correct molecule-level view over a concatenated cache."""

    def __init__(self, cache: Mapping[str, Any], *, name: str = ""):
        check_layout(cache, name=name)
        self.cache = cache
        self.n_molecules = int(cache["n_molecules"])
        self.atom_ptr = np.asarray(cache["atom_ptr"])
        self.bond_ptr = np.asarray(cache["bond_ptr"])
        self.path_ptr = np.asarray(cache["path_ptr"])
        self.nbr_ptr = np.asarray(cache["nbr_ptr"])
        self.path_cat = np.asarray(cache["path_cat"])
        self.path_end63 = np.asarray(cache["path_end63"])
        self.path_closure = np.asarray(cache["path_closure"])
        self.nbr_code = np.asarray(cache["nbr_code"])

    # -- ranges -----------------------------------------------------------
    def ranges(self, m: int) -> dict[str, int]:
        m = int(m)
        a0, a1 = int(self.atom_ptr[m]), int(self.atom_ptr[m + 1])
        p0, p1 = int(self.path_ptr[a0]), int(self.path_ptr[a1])
        n0, n1 = int(self.nbr_ptr[a0]), int(self.nbr_ptr[a1])
        return {"atom0": a0, "atom1": a1, "path0": p0, "path1": p1, "s1_0": n0, "s1_1": n1}

    # -- occurrence arrays ------------------------------------------------
    def path_codes(self, m: int) -> np.ndarray:
        r = self.ranges(m)
        return self.path_cat[r["path0"]:r["path1"]]

    def end_codes(self, m: int) -> np.ndarray:
        r = self.ranges(m)
        return self.path_end63[r["path0"]:r["path1"]]

    def closures(self, m: int) -> np.ndarray:
        r = self.ranges(m)
        return self.path_closure[r["path0"]:r["path1"]]

    def nbr_codes(self, m: int) -> np.ndarray:
        r = self.ranges(m)
        return self.nbr_code[r["s1_0"]:r["s1_1"]]

    def joint_pairs(self, m: int) -> list[tuple[int, int]]:
        r = self.ranges(m)
        pc = self.path_cat[r["path0"]:r["path1"]]
        e = self.path_end63[r["path0"]:r["path1"]]
        return [(int(a), int(b)) for a, b in zip(pc.tolist(), e.tolist())]

    # -- molecule-level document-frequency sets --------------------------
    def sets(self, m: int) -> dict[str, Any]:
        """Each code contributes **once** per molecule (document frequency)."""
        r = self.ranges(m)
        pc = self.path_cat[r["path0"]:r["path1"]].tolist()
        e = self.path_end63[r["path0"]:r["path1"]].tolist()
        nc = self.nbr_code[r["s1_0"]:r["s1_1"]].tolist()
        return {
            "path": set(pc),
            "end": set(e),
            "nbr": set(nc),
            "joint": set(zip(pc, e)),
            "n_path": len(pc),
            "n_s1": len(nc),
            "has_no_path": len(pc) == 0,
        }


# ---------------------------------------------------------------------------
# document frequency (Train document frequency by default)
# ---------------------------------------------------------------------------


def document_frequency(cache: Mapping[str, Any], *, name: str = "") -> dict[str, Any]:
    """Per-molecule document frequency of path/nbr/end/joint codes in one cache."""
    reader = MoleculeReader(cache, name=name)
    df_path: Counter = Counter()
    df_end: Counter = Counter()
    df_nbr: Counter = Counter()
    df_joint: Counter = Counter()
    per_mol: list[dict[str, Any]] = []
    for m in range(reader.n_molecules):
        s = reader.sets(m)
        df_path.update(s["path"])
        df_end.update(s["end"])
        df_nbr.update(s["nbr"])
        df_joint.update(s["joint"])
        per_mol.append(s)
    return {
        "df_path": df_path,
        "df_end": df_end,
        "df_nbr": df_nbr,
        "df_joint": df_joint,
        "per_mol": per_mol,
        "n_molecules": reader.n_molecules,
        "n_paths": int(reader.path_ptr[-1]),
        "n_s1": int(reader.nbr_ptr[-1]),
        "layout": {"n_molecules": reader.n_molecules, "n_atoms": int(reader.atom_ptr[-1])},
    }


# ---------------------------------------------------------------------------
# support of a set of molecules against a reference df (with Fit-LOO)
# ---------------------------------------------------------------------------


def support_stats(
    cache: Mapping[str, Any],
    df: Mapping[str, Any],
    *,
    loo_positions: Any = None,
    split: str = "",
) -> dict[str, Any]:
    """Per-split support statistics against a reference df.

    ``loo_positions`` is the set of molecule indices in *this* cache that built
    ``df``; each of them has its own contribution removed (``support = df - 1``).
    Molecules outside the reference never subtract.  ``"all"`` applies LOO to
    every molecule (when the cache *is* the reference).
    """
    reader = MoleculeReader(cache, name=split)
    if loo_positions == "all":
        loo_set: set[int] = set(range(reader.n_molecules))
    elif loo_positions is None:
        loo_set = set()
    else:
        loo_set = {int(x) for x in loo_positions}
    df_path = df["df_path"]
    df_end = df["df_end"]
    df_nbr = df["df_nbr"]
    df_joint = df["df_joint"]

    def _empty() -> dict[str, int]:
        return {"0": 0, "1-5": 0, "6-20": 0, ">20": 0}

    buckets = {"path": _empty(), "end": _empty(), "nbr": _empty(), "joint": _empty()}
    distinct_path: set[int] = set()
    distinct_end: set[int] = set()
    distinct_nbr: set[int] = set()
    distinct_joint: set[tuple[int, int]] = set()
    n_path_tok = n_end_tok = n_nbr_tok = n_joint_tok = 0
    r_joint = np.zeros(reader.n_molecules, dtype=np.float64)
    has_no_path = 0

    def sup(counter: Counter, code, own: bool) -> int:
        v = int(counter.get(code, 0))
        return v - 1 if (own and v > 0) else v

    for m in range(reader.n_molecules):
        own = m in loo_set
        s = reader.sets(m)
        if s["has_no_path"]:
            has_no_path += 1
        distinct_path |= s["path"]
        distinct_end |= s["end"]
        distinct_nbr |= s["nbr"]
        distinct_joint |= s["joint"]
        # token-weighted df buckets for every bank
        for code in reader.path_codes(m).tolist():
            n_path_tok += 1
            v = sup(df_path, code, own and code in s["path"])
            buckets["path"][_bucket(v)] += 1
        for code in reader.end_codes(m).tolist():
            n_end_tok += 1
            v = sup(df_end, code, own and code in s["end"])
            buckets["end"][_bucket(v)] += 1
        for code in reader.nbr_codes(m).tolist():
            n_nbr_tok += 1
            v = sup(df_nbr, code, own and code in s["nbr"])
            buckets["nbr"][_bucket(v)] += 1
        jp = reader.joint_pairs(m)
        low = 0
        for pair in jp:
            n_joint_tok += 1
            v = sup(df_joint, pair, own and pair in s["joint"])
            buckets["joint"][_bucket(v)] += 1
            if v <= RP.LOW_SUPPORT_DF:
                low += 1
        r_joint[m] = (low / len(jp)) if jp else 0.0

    def _frac(b: Mapping[str, int]) -> dict[str, float]:
        t = max(sum(b.values()), 1)
        return {k: v / t for k, v in b.items()}

    return {
        "split": split,
        "n_molecules": reader.n_molecules,
        "has_no_path": int(has_no_path),
        "distinct_codes": {
            "path": len(distinct_path),
            "end": len(distinct_end),
            "nbr": len(distinct_nbr),
            "joint": len(distinct_joint),
        },
        "occurrence_counts": {
            "path": int(n_path_tok),
            "end": int(n_end_tok),
            "nbr": int(n_nbr_tok),
            "joint": int(n_joint_tok),
        },
        "df_buckets_frac": {bank: _frac(b) for bank, b in buckets.items()},
        "r_joint": r_joint,
        "df_reference_size": {
            "path": len(df_path),
            "end": len(df_end),
            "nbr": len(df_nbr),
            "joint": len(df_joint),
        },
        "n_loo_molecules": len(loo_set),
    }


def document_frequency_subset(
    cache: Mapping[str, Any], positions, *, name: str = ""
) -> dict[str, Any]:
    """Document frequency built only from the given molecule positions."""
    reader = MoleculeReader(cache, name=name)
    df_path: Counter = Counter()
    df_end: Counter = Counter()
    df_nbr: Counter = Counter()
    df_joint: Counter = Counter()
    per_mol: dict[int, Any] = {}
    for m in (int(x) for x in positions):
        s = reader.sets(m)
        df_path.update(s["path"])
        df_end.update(s["end"])
        df_nbr.update(s["nbr"])
        df_joint.update(s["joint"])
        per_mol[m] = s
    return {
        "df_path": df_path,
        "df_end": df_end,
        "df_nbr": df_nbr,
        "df_joint": df_joint,
        "per_mol": per_mol,
        "n_molecules": len(per_mol),
        "n_paths": int(sum(v["n_path"] for v in per_mol.values())),
        "n_s1": int(sum(v["n_s1"] for v in per_mol.values())),
        "layout": {"n_molecules": len(per_mol), "atoms_reference": int(reader.atom_ptr[-1])},
    }


def _bucket(df: int) -> str:
    if df <= 0:
        return "0"
    if df <= 5:
        return "1-5"
    if df <= 20:
        return "6-20"
    return ">20"


# ---------------------------------------------------------------------------
# the independent single-molecule oracle (never reuses ``MoleculeReader``)
# ---------------------------------------------------------------------------


def oracle_from_molecule(mol: Mapping[str, Any]) -> dict[str, Any]:
    """Counts straight from a ``lcrc_data.build_molecule`` record (per-atom pointers).

    The per-molecule record has its own ``nbr_ptr`` / ``path_ptr`` that are
    ``atom -> occurrence`` boundaries *within the molecule*; that is the
    independent source of truth.
    """
    nbr_ptr = np.asarray(mol["nbr_ptr"])
    path_ptr = np.asarray(mol["path_ptr"])
    nbr_code = np.asarray(mol["nbr_code"])
    path_cat = np.asarray(mol["path_cat"])
    e63 = np.asarray(mol["path_end63"])
    n_s1 = int(nbr_ptr[-1])
    n_path = int(path_ptr[-1])
    return {
        "n_atoms": int(mol["n_atoms"]),
        "n_bonds": int(mol["n_bonds"]),
        "n_s1": n_s1,
        "n_path": n_path,
        "nbr_counter": dict(Counter(int(x) for x in nbr_code.tolist())),
        "path_counter": dict(Counter(int(x) for x in path_cat.tolist())),
        "end_counter": dict(Counter(int(x) for x in e63.tolist())),
        "joint_counter": {f"{a},{b}": int(v) for (a, b), v in Counter(zip(path_cat.tolist(), e63.tolist())).items()},
    }


__all__ = [name for name in dir() if not name.startswith("_")]
