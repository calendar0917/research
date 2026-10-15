"""cscl-bjoc-signature-transfer-v1 — CPU, label-free signature-transfer analysis.

Reusable primitives for the preregistered round:

* frozen Phase-0 attribution records -> molecule-capped signature importance table
* within-molecule |IG| percentile (average-rank min-max)
* deterministic top-k selection, random/shuffle control rankings
* per-Junction natural-replacement responses (identical arithmetic to Phase 0)
* RDKit dataset-graph reconstruction (canonical SMILES / Murcko scaffold)
* bootstrap / sign-test / Spearman statistics
* the exact three-way decision rule

No training, no label access, no official-test / Locked access anywhere here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_source_attribution_feasibility_v1 as bsaf  # noqa: E402
import bjoc_source_ownership_v1 as own  # noqa: E402
import bjoc_structure_explanation_audit_v1 as v1  # noqa: E402
import cscl_smiles_graph as csg  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-signature-transfer-v1"
TRACK = REPO_ROOT / "tracks/structure-contribution"
OUT = TRACK / "results" / "bjoc_signature_transfer_v1"
CONFIG_PATH = TRACK / "configs" / "bjoc_signature_transfer_v1.yaml"

DATA = TRACK / "results" / "bjoc_task_adaptive_ssl_v1"
F10K = TRACK / "results" / "bjoc_full10k_pretraining_transfer_v1"
REC = TRACK / "results" / "bjoc_recipe_tuning_v1"
BSAF_DIR = TRACK / "results" / "bjoc_source_attribution_feasibility_v1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

# ---- frozen protocol constants -------------------------------------------
DISCOVERY_FLOOR_MOLECULES = 20
INTERNAL_CONTROL_FLOOR_MOLECULES = 10
UNKNOWN_SCORE_PRIMARY = 0.5
TOP_K_PRIMARY = 4
TOP_K_SENSITIVITY = (1, 2, 4, 8)
RANDOM_DRAWS = 16
RANDOM_RNG_BASE = 123_400_000
RANDOM_RNG_STRIDE_SEED = 100_000
SHUFFLE_PERMUTATIONS = 16
SHUFFLE_RNG_BASE = 987_000_000
SHUFFLE_RNG_STRIDE_SEED = 10_000
ALPHA = 1.0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
ORACLE_FRACTION_MIN = 0.25
REPLAY_TOLERANCE = 1.0e-9
METHODS = ("SIG", "SIG_MIN", "RING", "DEGREE", "RING_DEGREE", "DEV", "RARE", "ORACLE")
CONTROL_METHODS = ("RANDOM", "SHUFFLE")
SIG_LABELS = {
    "TRANSFERABLE_SOURCE_SIGNATURE_SIGNAL",
    "CONTEXT_DEPENDENT_OR_PARTIAL_TRANSFER",
    "NO_SUPPORTED_SIGNATURE_TRANSFER",
}


# ---------------------------------------------------------------------------
# small io / hashing
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# signatures
# ---------------------------------------------------------------------------


def sig_key(sig: Any) -> tuple:
    """JSON signature -> canonical hashable key."""
    return (int(sig[0]), tuple(int(x) for x in sig[1]), tuple(int(x) for x in sig[2]), int(sig[3]), int(sig[4]))


def sig_json(key: tuple) -> list:
    return [int(key[0]), [int(x) for x in key[1]], [int(x) for x in key[2]], int(key[3]), int(key[4])]


def sig_join(key: tuple) -> str:
    """Flat canonical string key for a signature tuple."""
    return "|".join(str(int(x)) for x in [key[0], *key[1], *key[2], key[3], key[4]])


def signature_words(key: tuple, atom_words: Sequence[str], bond_words: Sequence[str]) -> dict[str, Any]:
    return {
        "center": atom_words[int(key[0])],
        "endpoints": sorted(atom_words[int(x)] for x in key[1]),
        "bond_orders": sorted(bond_words[int(b)] for b in key[2]),
        "center_in_ring": int(key[3]),
        "n_ring_bonds": int(key[4]),
    }


# ---------------------------------------------------------------------------
# discovery table (molecule-capped)
# ---------------------------------------------------------------------------


def average_rank_percentile(values: Sequence[float]) -> np.ndarray:
    """Average-rank min-max percentile in [0, 1]; n == 1 -> 0.5 (ties averaged)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    sx = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (float(i + j) / 2.0) + 1.0
        i = j + 1
    return (ranks - 1.0) / float(n - 1)


def build_discovery_table(records: Sequence[Mapping[str, Any]], floor: int = DISCOVERY_FLOOR_MOLECULES) -> dict[str, Any]:
    """Molecule-capped class table from frozen Phase-0 ``records_c``.

    Returns candidate classes, exclusion counts, rarity counts and coverage.
    """
    by_ms: dict[tuple[int, int], list[tuple[tuple, float]]] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)
    seen_junction: set[tuple[int, int]] = set()
    for rec in records:
        key = sig_key(rec["signature"])
        by_ms[(int(rec["seed"]), int(rec["row"]))].append((key, float(rec["ig_abs"])))
        jk = (int(rec["row"]), int(rec["j"]))
        if jk not in seen_junction:
            seen_junction.add(jk)
            counts[key] += 1

    molseed_class: dict[tuple[int, int], dict[tuple, float]] = {}
    for (seed, row), items in by_ms.items():
        pcts = average_rank_percentile([v for _k, v in items])
        per_class: dict[tuple, list[float]] = defaultdict(list)
        for (key, _v), p in zip(items, pcts):
            per_class[key].append(float(p))
        molseed_class[(seed, row)] = {k: float(np.mean(v)) for k, v in per_class.items()}

    # molecule -> class -> [per-seed values]
    mol_class: dict[int, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (seed, row), cls in molseed_class.items():
        for key, val in cls.items():
            mol_class[int(row)][key].append(float(val))

    # class -> molecule -> per-seed values
    class_rows: dict[tuple, dict[int, list[float]]] = defaultdict(dict)
    for row, per_class in mol_class.items():
        for key, seed_vals in per_class.items():
            class_rows[key][int(row)] = seed_vals

    classes: list[dict[str, Any]] = []
    for key, per_mol in class_rows.items():
        vals = [float(np.mean(v)) for v in per_mol.values()]
        n_mol = len(vals)
        seeds_present = [len(v) for v in per_mol.values()]
        classes.append(
            {
                "signature": sig_json(key),
                "support_molecules": int(n_mol),
                "support_molecule_seed_units": int(sum(seeds_present)),
                "T": float(np.mean(vals)),
                "T_std_over_molecules": float(np.std(vals, ddof=0)),
            }
        )
    classes.sort(key=lambda c: (-c["support_molecules"], -c["T"], json.dumps(c["signature"])))
    candidates = sorted([c for c in classes if c["support_molecules"] >= int(floor)], key=lambda c: (-c["T"], -c["support_molecules"], json.dumps(c["signature"])))

    total = int(sum(counts.values()))
    vocab = len(counts)
    rarity = {
        "formula": "-log2((count + 0.5) / (N + 0.5*(V+1)))",
        "N": total,
        "V": int(vocab),
        "counts": {sig_join(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: json.dumps(sig_json(kv[0])))},
    }
    n_junctions = len(seen_junction)
    n_molecules = len({row for _seed, row in by_ms})
    cand_cov = float(sum(counts[sig_key(c["signature"])] for c in candidates) / max(n_junctions, 1))
    return {
        "protocol": PROTOCOL_ID,
        "floor_molecules": int(floor),
        "n_molecules": int(n_molecules),
        "n_seeds": len({s for s, _r in by_ms}),
        "n_junctions_unique": int(n_junctions),
        "n_classes": int(len(classes)),
        "n_candidates": int(len(candidates)),
        "candidate_molecule_junction_coverage": cand_cov,
        "classes": classes,
        "candidates": candidates,
        "candidate_keys": [sig_json(c["signature"]) for c in candidates],
        "rarity": rarity,
        "percentile_rule": "average-rank minmax (rank-1)/(n_j-1); n_j==1 -> 0.5",
        "molecule_cap_rule": "per (seed, molecule, class) mean percentile; table = mean over molecules of mean over seeds",
    }


def table_candidate_index(table: Mapping[str, Any]) -> dict[tuple, int]:
    return {sig_key(s): i for i, s in enumerate(table["candidate_keys"])}


def table_candidate_values(table: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(c["T"]) for c in table["candidates"]], dtype=np.float64)


def rarity_score(key: tuple, rarity: Mapping[str, Any]) -> float:
    cnt = rarity["counts"].get(sig_join(key), 0)
    n = float(rarity["N"])
    v = float(rarity["V"])
    return float(-math.log2((cnt + 0.5) / (n + 0.5 * (v + 1.0))))


def junction_arrays(obj: sro.MolObjects) -> dict[str, np.ndarray]:
    """Per-Junction signature code / degree / ring / endpoint-ring context.

    Codes are indices into the *global* deterministic signature order of
    ``signature_corpus_counts``; the caller maps them through the table.
    """
    n_j = int(obj.junc_index.shape[0])
    sigs: list[tuple] = []
    deg = np.zeros(n_j, dtype=np.int64)
    in_ring = np.zeros(n_j, dtype=np.int64)
    ring_bonds = np.zeros(n_j, dtype=np.int64)
    endpoint_ring_count = np.zeros(n_j, dtype=np.int64)
    ring_flag = sro.ATOM_CATEGORIES + sro.ATOM_DEGREE_SLOTS
    for j in range(n_j):
        sig = own.junction_signature(obj, j)
        sigs.append(sig)
        deg[j] = own.center_degree(obj, j)
        in_ring[j] = int(sig[3])
        ring_bonds[j] = int(sig[4])
        i, c, k = (int(v) for v in obj.junc_index[j])
        endpoint_ring_count[j] = int(round(float(obj.atom_feat[i, ring_flag]))) + int(round(float(obj.atom_feat[k, ring_flag])))
    return {
        "signatures": sigs,
        "degree": deg,
        "in_ring": in_ring,
        "n_ring_bonds": ring_bonds,
        "endpoint_ring_count": endpoint_ring_count,
    }


def signature_scores(signatures: Sequence[tuple], table: Mapping[str, Any], unknown: float | str = "primary") -> np.ndarray:
    """Per-Junction table score; unknown / below-floor classes use the frozen fallback."""
    idx = table_candidate_index(table)
    cand = table_candidate_values(table)
    if unknown == "primary":
        fallback = float(UNKNOWN_SCORE_PRIMARY)
    elif unknown == "min":
        fallback = float(cand.min()) if cand.size else float(UNKNOWN_SCORE_PRIMARY)
    else:
        fallback = float(unknown)
    out = np.full(len(signatures), fallback, dtype=np.float64)
    for j, sig in enumerate(signatures):
        c = idx.get(sig)
        if c is not None:
            out[j] = cand[c]
    return out


# ---------------------------------------------------------------------------
# deterministic selection / controls
# ---------------------------------------------------------------------------


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    k_eff = min(int(k), int(arr.size))
    return np.argsort(-arr, kind="stable")[:k_eff].astype(np.int64)


def random_draw_sets(n_j: int, k: int, seed: int, draws: int = RANDOM_DRAWS) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    k_eff = min(int(k), int(n_j))
    if k_eff <= 0:
        return np.zeros((0, 0), dtype=np.int64)
    return np.stack([rng.choice(n_j, size=k_eff, replace=False) for _ in range(int(draws))]).astype(np.int64)


def shuffle_top_sets(candidate_scores: np.ndarray, codes: np.ndarray, permutation: np.ndarray, k: int) -> np.ndarray:
    mapped = np.full(codes.shape, np.nan, dtype=np.float64)
    known = codes >= 0
    if known.any():
        mapped[known] = candidate_scores[permutation][codes[known]]
    mapped[~known] = UNKNOWN_SCORE_PRIMARY
    return top_k_indices(np.nan_to_num(mapped, nan=UNKNOWN_SCORE_PRIMARY), k)


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------


def all_junction_responses(model: src.BJOC, pack: Mapping[str, Any], w_ref: torch.Tensor, alpha: float = ALPHA) -> np.ndarray:
    """delta_j for every Junction j; identical arithmetic to Phase-0 _subset_response."""
    w = pack["w"]
    n_j = int(w.shape[0])
    out = np.empty(n_j, dtype=np.float64)
    with torch.no_grad():
        base = float(bsaf.predict_null(model, pack).item())
        for j in range(n_j):
            w2 = w.clone()
            w2[j] = w[j] + float(alpha) * (w_ref[j % w_ref.shape[0]] - w[j])
            out[j] = float(bsaf.predict_null(model, pack, w=w2).item()) - base
    return out


def nearest_reference(row_stats: Mapping[str, Any], reference: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen nearest-reference rule (L1 on atom/bond/junction counts, tie smaller row)."""
    return min(
        reference,
        key=lambda r: (
            abs(int(r["n_atoms"]) - int(row_stats["n_atoms"]))
            + abs(int(r["n_bonds"]) - int(row_stats["n_bonds"]))
            + abs(int(r["n_junctions"]) - int(row_stats["n_junctions"])),
            int(r["row"]),
        ),
    )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def average_ranks(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    order = np.argsort(arr, kind="stable")
    sx = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (float(i + j) / 2.0) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size < 2 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return float("nan")
    rx = average_ranks(xa)
    ry = average_ranks(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size < 2:
        return float("nan")
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    return float((xc * yc).sum() / denom) if denom > 0 else float("nan")


def bootstrap_mean(values: Sequence[float], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_lower": float("nan"), "ci95_upper": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(arr, size=(int(n), arr.size), replace=True).mean(1)
    return {
        "mean": float(arr.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "n": int(arr.size),
    }


def sign_test_one_sided(deltas: Sequence[float]) -> dict[str, float]:
    d = np.asarray(deltas, dtype=np.float64).reshape(-1)
    d = d[~np.isnan(d)]
    n = int(d.size)
    k = int((d > 0).sum())
    p = float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0**n)) if n else float("nan")
    return {"n_positive": k, "n": n, "p_one_sided": p}


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------


def decide_from_gates(gates: Mapping[str, bool]) -> str:
    required = ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
    values = {k: bool(gates[k]) for k in required}
    if values["M0"] and all(values[k] for k in required[1:]):
        return "TRANSFERABLE_SOURCE_SIGNATURE_SIGNAL"
    if not values["M0"] or not values["G1"]:
        return "NO_SUPPORTED_SIGNATURE_TRANSFER"
    return "CONTEXT_DEPENDENT_OR_PARTIAL_TRANSFER"


# ---------------------------------------------------------------------------
# RDKit dataset-graph reconstruction
# ---------------------------------------------------------------------------


_ZINC_BOND_ORDER = {"SINGLE": "SINGLE", "DOUBLE": "DOUBLE", "TRIPLE": "TRIPLE"}


def zinc_words() -> tuple[list[str], list[str]]:
    return csg._zinc_dicts()


def build_rdkit_mol(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> tuple[Any, str]:
    """Label-free RDKit reconstruction of one PyG-ZINC dataset graph.

    Returns ``(mol, sanitize_tier)`` where tier is ``"full"`` or ``"topology"``.
    The committed ZINC atom dictionary cannot express uncharged N/O/S H counts;
    molecule identity is therefore a cache-level graph identity.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    awords, bwords = zinc_words()
    rw = Chem.RWMol()
    for t in np.asarray(x).reshape(-1):
        el, h, q = csg.parse_atom_word(awords[int(t)])
        a = Chem.Atom(el)
        a.SetFormalCharge(int(q))
        a.SetNumExplicitHs(int(h))
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    ei = np.asarray(edge_index)
    ea = np.asarray(edge_attr).reshape(-1)
    bond_types = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
    }
    seen: set[tuple[int, int]] = set()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
        word = bwords[int(ea[k])]
        if word == "NONE":
            continue
        rw.AddBond(u, v, bond_types[word])
    mol = rw.GetMol()
    tier = "full"
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        tier = "topology"
    return mol, tier


def canonical_smiles_and_scaffold(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> tuple[str, str, str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol, tier = build_rdkit_mol(x, edge_index, edge_attr)
    smi = Chem.MolToSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return smi, scaffold, tier


__all__ = [
    "PROTOCOL_ID",
    "OUT",
    "CONFIG_PATH",
    "DATA",
    "F10K",
    "REC",
    "BSAF_DIR",
    "ZINC_ROOT",
    "DISCOVERY_FLOOR_MOLECULES",
    "INTERNAL_CONTROL_FLOOR_MOLECULES",
    "UNKNOWN_SCORE_PRIMARY",
    "TOP_K_PRIMARY",
    "TOP_K_SENSITIVITY",
    "RANDOM_DRAWS",
    "RANDOM_RNG_BASE",
    "SHUFFLE_PERMUTATIONS",
    "SHUFFLE_RNG_BASE",
    "ALPHA",
    "METHODS",
    "CONTROL_METHODS",
    "SIG_LABELS",
    "read_json",
    "write_json",
    "file_sha256",
    "sig_key",
    "sig_json",
    "signature_words",
    "average_rank_percentile",
    "average_ranks",
    "build_discovery_table",
    "table_candidate_index",
    "table_candidate_values",
    "rarity_score",
    "junction_arrays",
    "signature_scores",
    "top_k_indices",
    "random_draw_sets",
    "shuffle_top_sets",
    "all_junction_responses",
    "nearest_reference",
    "spearman",
    "pearson",
    "bootstrap_mean",
    "sign_test_one_sided",
    "decide_from_gates",
    "zinc_words",
    "build_rdkit_mol",
    "canonical_smiles_and_scaffold",
]
