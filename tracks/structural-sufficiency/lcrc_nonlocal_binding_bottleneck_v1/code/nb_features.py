"""The two representations compared in this round.

``REAL``
    ``P[type_pair, distance_bin]`` -- the count of unordered **heavy-atom** pairs
    of the specified element class at the specified shortest-path distance bin,
    restricted to ``distance >= 5`` and to pairs inside one connected component,
    divided by ``max(1, total_number_of_pairs_with_distance_>=5)``.

``NULL``
    The independence (outer-product) null of the *same* molecule:
    ``NULL[p, k] = row_margin[p] * col_margin[k] / N`` computed on the raw counts
    and divided by the same ``max(1, total)``.  It preserves the type-pair
    margins, the untyped distance-bin margins and the total pair count exactly;
    the only difference from ``REAL`` is the pairing of element types with
    distance bins.

Both are 126-d = 21 unordered type pairs x 6 distance bins.  The 42-d
``lipo_v1`` global block (``g``) and the 2048-bit ECFP4 fingerprint come from
the frozen production path and the frozen baseline definition respectively.

Everything here is label-free: no target, split or performance quantity is read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import nb_common as C

#: Frozen ECFP4/Morgan generator, once (radius 2, 2048 bits, no chirality).
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_ECFP4_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=C.MORGAN_RADIUS, fpSize=C.MORGAN_NBITS, includeChirality=False
)

#: Numerical tolerance for margin conservation in raw atom-pair units.
MARGIN_TOL = 1e-8

#: Partial-sanitisation mask used only for the three invalid-valence metal
#: complexes in the official Train split (Al valence 7, Ge valence 5).  Valence
#: perception is the failing operation; everything the fingerprint needs
#: (aromaticity, ring info, conjugation) is still perceived.
_PARTIAL_SANITIZE_OPS = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES


# ---------------------------------------------------------------------------
# element classification
# ---------------------------------------------------------------------------


def element_class(symbol: str) -> int:
    """Index into :data:`nb_common.ELEMENT_CLASSES` (``X`` = F/Cl/Br/I)."""
    s = str(symbol)
    if s == "C":
        return 0
    if s == "N":
        return 1
    if s == "O":
        return 2
    if s == "S":
        return 3
    if s in C.HALOGENS:
        return 4
    return 5


def type_pair_index(left: int, right: int) -> int:
    """Index of the unordered class pair ``{left, right}`` in ``TYPE_PAIRS``."""
    i, j = (int(left), int(right)) if int(left) <= int(right) else (int(right), int(left))
    if not (0 <= i <= j < len(C.ELEMENT_CLASSES)):
        raise ValueError(f"class pair ({left}, {right}) outside 0..5")
    return i * 6 - i * (i - 1) // 2 + (j - i)


def distance_bin(distance: int) -> int:
    """Bin index for a finite shortest-path distance ``>= 5`` (else ``-1``)."""
    d = int(distance)
    if d < C.MIN_PAIR_DISTANCE:
        return -1
    return min(d - C.MIN_PAIR_DISTANCE, C.N_DIST_BINS - 1)


# ---------------------------------------------------------------------------
# 2D heavy-atom graph and shortest-path distances
# ---------------------------------------------------------------------------


def heavy_graph(smiles: str) -> dict[str, Any]:
    """Connectivity-only parse -> heavy-atom adjacency + all-pairs BFS distances.

    Distances are the number of bonds on a shortest path inside the 2D molecular
    graph.  Hydrogen atoms (atomic number 1) are **not** graph nodes.  Unreachable
    pairs keep distance ``-1`` and are never counted.
    """
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    n_all = int(mol.GetNumAtoms())
    atomic_num = np.asarray([int(a.GetAtomicNum()) for a in mol.GetAtoms()], dtype=np.int64)
    keep = np.flatnonzero(atomic_num > 1)
    remap = -np.ones(n_all, dtype=np.int64)
    remap[keep] = np.arange(keep.size, dtype=np.int64)
    classes = np.asarray([element_class(mol.GetAtomWithIdx(int(i)).GetSymbol()) for i in keep], dtype=np.int64)
    symbols = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in keep]
    n = int(keep.size)
    adj: list[list[int]] = [[] for _ in range(n)]
    bonds = 0
    for bond in mol.GetBonds():
        u = int(remap[bond.GetBeginAtomIdx()])
        v = int(remap[bond.GetEndAtomIdx()])
        if u < 0 or v < 0:
            continue  # a bond to an explicit hydrogen is not a heavy-heavy pair
        adj[u].append(v)
        adj[v].append(u)
        bonds += 1
    dist = np.full((n, n), -1, dtype=np.int64)
    for root in range(n):
        d = dist[root]
        d[root] = 0
        frontier = [root]
        level = 0
        while frontier:
            level += 1
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if d[v] < 0:
                        d[v] = level
                        nxt.append(v)
            frontier = nxt
    n_components = 0
    seen = np.zeros(n, dtype=bool)
    for root in range(n):
        if seen[root]:
            continue
        n_components += 1
        stack = [root]
        seen[root] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return {
        "n_heavy_atoms": n,
        "n_heavy_bonds": bonds,
        "classes": classes,
        "symbols": symbols,
        "distance": dist,
        "n_components": n_components,
        "n_hydrogen_nodes_removed": int(n_all - n),
    }


# ---------------------------------------------------------------------------
# REAL / NULL pair statistics
# ---------------------------------------------------------------------------


def pair_counts(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Raw ``R[p, k]`` counts plus the margins used by the NULL."""
    dist = np.asarray(graph["distance"], dtype=np.int64)
    classes = np.asarray(graph["classes"], dtype=np.int64)
    n = int(classes.size)
    counts = np.zeros((C.N_TYPE_PAIRS, C.N_DIST_BINS), dtype=np.int64)
    pair_distances: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = int(dist[i, j])
            k = distance_bin(d) if d >= 0 else -1
            if k < 0:
                continue
            counts[type_pair_index(int(classes[i]), int(classes[j])), k] += 1
            pair_distances.append(d)
    row_margin = counts.sum(axis=1)
    col_margin = counts.sum(axis=0)
    total = int(counts.sum())
    return {
        "counts": counts,
        "row_margin": row_margin,
        "col_margin": col_margin,
        "total": total,
        "pair_distances": np.asarray(pair_distances, dtype=np.int64),
    }


def null_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    """Independence null of the raw counts (exact margins, no exchangeability)."""
    row = np.asarray(counts["row_margin"], dtype=np.float64)
    col = np.asarray(counts["col_margin"], dtype=np.float64)
    total = float(counts["total"])
    if total <= 0:
        null = np.zeros((C.N_TYPE_PAIRS, C.N_DIST_BINS), dtype=np.float64)
    else:
        null = np.outer(row, col) / total
    return {
        "null": null,
        "row_margin_recovered": null.sum(axis=1),
        "col_margin_recovered": null.sum(axis=0),
        "total_recovered": float(null.sum()),
    }


def pair_feature_molecule(smiles: str) -> dict[str, Any]:
    """Full label-free pair representation for one molecule (REAL + NULL)."""
    graph = heavy_graph(smiles)
    counts = pair_counts(graph)
    null = null_counts(counts)
    scale = float(max(1, int(counts["total"])))
    real_vec = counts["counts"].astype(np.float64).reshape(-1) / scale
    null_vec = np.asarray(null["null"], dtype=np.float64).reshape(-1) / scale
    return {
        "smiles": str(smiles),
        "n_heavy_atoms": int(graph["n_heavy_atoms"]),
        "n_heavy_bonds": int(graph["n_heavy_bonds"]),
        "n_components_heavy": int(graph["n_components"]),
        "n_hydrogen_nodes_removed": int(graph["n_hydrogen_nodes_removed"]),
        "counts": counts["counts"],
        "row_margin": np.asarray(counts["row_margin"], dtype=np.int64),
        "col_margin": np.asarray(counts["col_margin"], dtype=np.int64),
        "total": int(counts["total"]),
        "pair_distances": counts["pair_distances"],
        "real": real_vec,
        "null": null_vec,
        "scale": scale,
        "distance": np.asarray(graph["distance"], dtype=np.int64),
        "classes": np.asarray(graph["classes"], dtype=np.int64),
        "symbols": list(graph["symbols"]),
    }


# ---------------------------------------------------------------------------
# robust RDKit parsing ladder (documented, deterministic)
# ---------------------------------------------------------------------------


def parse_robust(smiles: str) -> tuple[Any, str]:
    """Strict parse, then documented partial sanitisation for valence failures.

    Step 1: ``Chem.MolFromSmiles`` (full sanitisation).
    Step 2: parse without sanitisation, then ``SANITIZE_ALL ^ SANITIZE_PROPERTIES``
            plus ``FastFindRings``.  Only valence perception is skipped, which is
            the operation the published invalid-valence metal complexes in the
            official Train split fail.  No row is ever dropped.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None:
        return mol, "strict"
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r} at all")
    Chem.SanitizeMol(mol, sanitizeOps=_PARTIAL_SANITIZE_OPS)
    Chem.FastFindRings(mol)
    return mol, "partial_sanitize"


def ecfp4_bits(smiles: str) -> tuple[np.ndarray, str]:
    """Frozen ECFP4 block: Morgan radius 2, 2048 bits, binary, no chirality."""
    mol, mode = parse_robust(smiles)
    fp = _ECFP4_GENERATOR.GetFingerprint(mol)
    bits = np.zeros(C.MORGAN_NBITS, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, bits)
    return bits, mode


def canonical_smiles(smiles: str) -> tuple[str, str]:
    mol, mode = parse_robust(smiles)
    try:
        return str(Chem.MolToSmiles(mol)), mode
    except Exception:  # pragma: no cover - never observed on the frozen data
        return str(smiles), f"{mode}_smiles_fallback"


def scaffold_key(smiles: str) -> dict[str, Any]:
    """Bemis-Murcko scaffold, with the pre-registered fallback ladder.

    ``scaffold`` (Murcko) -> ``canonical_smiles`` when the scaffold is empty or
    cannot be computed -> the raw source SMILES when even that fails.
    """
    mol, mode = parse_robust(smiles)
    can, can_mode = canonical_smiles(smiles)
    try:
        scaffold = str(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        scaffold_mode = f"murcko:{mode}"
    except Exception:
        scaffold = ""
        scaffold_mode = f"murcko_failed:{mode}"
    if scaffold:
        return {"group": scaffold, "kind": "murcko", "mode": scaffold_mode, "canonical_smiles": can,
                "canonical_mode": can_mode}
    return {
        "group": can if can else str(smiles),
        "kind": "canonical_smiles_fallback",
        "mode": scaffold_mode,
        "canonical_smiles": can,
        "canonical_mode": can_mode,
    }


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------


def build_feature_matrices(smiles: Sequence[str], *, label: str = "") -> dict[str, Any]:
    """REAL / NULL / ECFP4 blocks for a list of SMILES (label-free)."""
    n = len(smiles)
    real = np.zeros((n, C.N_PAIR_FEATURES), dtype=np.float64)
    null = np.zeros((n, C.N_PAIR_FEATURES), dtype=np.float64)
    counts = np.zeros((n, C.N_TYPE_PAIRS, C.N_DIST_BINS), dtype=np.int32)
    row_margin = np.zeros((n, C.N_TYPE_PAIRS), dtype=np.int32)
    col_margin = np.zeros((n, C.N_DIST_BINS), dtype=np.int32)
    totals = np.zeros(n, dtype=np.int32)
    n_heavy = np.zeros(n, dtype=np.int32)
    n_comp = np.zeros(n, dtype=np.int32)
    n_h_removed = np.zeros(n, dtype=np.int32)
    ecfp = np.zeros((n, C.MORGAN_NBITS), dtype=np.uint8)
    parse_modes: list[str] = []
    for i, s in enumerate(smiles):
        rec = pair_feature_molecule(s)
        real[i] = rec["real"].astype(np.float64)
        null[i] = rec["null"].astype(np.float64)
        counts[i] = rec["counts"].astype(np.int32)
        row_margin[i] = rec["row_margin"]
        col_margin[i] = rec["col_margin"]
        totals[i] = rec["total"]
        n_heavy[i] = rec["n_heavy_atoms"]
        n_comp[i] = rec["n_components_heavy"]
        n_h_removed[i] = rec["n_hydrogen_nodes_removed"]
        bits, mode = ecfp4_bits(s)
        ecfp[i] = bits
        parse_modes.append(mode)
        if (i + 1) % 5000 == 0:
            C.log(f"[features {label}] {i + 1}/{n}")
    return {
        "smiles": np.asarray(list(smiles), dtype=object),
        "real": real,
        "null": null,
        "counts": counts,
        "row_margin": row_margin,
        "col_margin": col_margin,
        "totals": totals,
        "n_heavy_atoms": n_heavy,
        "n_components_heavy": n_comp,
        "n_hydrogen_nodes_removed": n_h_removed,
        "ecfp4": ecfp,
        "parse_modes": np.asarray(parse_modes, dtype=object),
    }


def margin_residuals(data: Mapping[str, Any], *, tol: float = MARGIN_TOL) -> dict[str, Any]:
    """Prove REAL and NULL share the exact type-pair / distance-bin margins.

    The comparison is done in raw-count units (feature x ``max(1, total)``), so a
    residual of ``1e-8`` means "a ten-millionth of one atom pair".  Floating-point
    conservation is asserted within :data:`MARGIN_TOL` and the integer
    reconstruction of the margins is asserted exactly.
    """
    real = np.asarray(data["real"], dtype=np.float64)
    null = np.asarray(data["null"], dtype=np.float64)
    totals = np.asarray(data["totals"], dtype=np.float64)
    scale = np.maximum(1.0, totals)
    real3 = real.reshape(-1, C.N_TYPE_PAIRS, C.N_DIST_BINS) * scale.reshape(-1, 1, 1)
    null3 = null.reshape(-1, C.N_TYPE_PAIRS, C.N_DIST_BINS) * scale.reshape(-1, 1, 1)
    row_real, row_null = real3.sum(axis=2), null3.sum(axis=2)
    col_real, col_null = real3.sum(axis=1), null3.sum(axis=1)
    tot_real, tot_null = real3.sum(axis=(1, 2)), null3.sum(axis=(1, 2))
    nonzero = totals > 0
    row_res = float(np.abs(row_real - row_null).max()) if totals.size else 0.0
    col_res = float(np.abs(col_real - col_null).max()) if totals.size else 0.0
    tot_res = float(np.abs(tot_real - tot_null).max()) if totals.size else 0.0
    return {
        "n_molecules": int(totals.size),
        "n_molecules_with_pairs": int(nonzero.sum()),
        "n_molecules_without_pairs": int((~nonzero).sum()),
        "row_margin_max_abs_residual": row_res,
        "col_margin_max_abs_residual": col_res,
        "total_max_abs_residual": tot_res,
        "conservation_tolerance": float(tol),
        "conservation_units": "atom pairs (feature x max(1,total))",
        "row_margin_conserved": bool(row_res <= tol),
        "col_margin_conserved": bool(col_res <= tol),
        "total_conserved": bool(tot_res <= tol),
        "row_margin_exact": bool(row_res == 0.0),
        "col_margin_exact": bool(col_res == 0.0),
        "total_exact": bool(tot_res == 0.0),
        "zero_pair_rows_all_zero": bool(
            np.all(real[~nonzero] == 0.0) and np.all(null[~nonzero] == 0.0)
        ),
        "raw_row_margin_matches_counts": bool(
            np.array_equal(np.rint(row_real).astype(np.int64), np.asarray(data["row_margin"], dtype=np.int64))
        ),
        "raw_col_margin_matches_counts": bool(
            np.array_equal(np.rint(col_real).astype(np.int64), np.asarray(data["col_margin"], dtype=np.int64))
        ),
        "raw_total_matches_counts": bool(
            np.array_equal(np.rint(tot_real).astype(np.int64), np.asarray(data["totals"], dtype=np.int64))
        ),
    }


__all__ = [
    "element_class",
    "type_pair_index",
    "distance_bin",
    "heavy_graph",
    "pair_counts",
    "null_counts",
    "pair_feature_molecule",
    "parse_robust",
    "ecfp4_bits",
    "canonical_smiles",
    "scaffold_key",
    "build_feature_matrices",
    "margin_residuals",
]
