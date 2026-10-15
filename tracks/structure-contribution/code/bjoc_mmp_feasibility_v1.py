"""bjoc-mmp-feasibility-v1: single-cut MMP feasibility + target audit on ZINC-12k.

Research question (direction C, first feasibility round)
-------------------------------------------------------
For a genuine local replacement ``R -> R'`` at a shared context ``C``, can the
property delta ``dy(C, R -> R') = y(C+R') - y(C+R)`` be reused across different
contexts?  This module answers only the *feasibility* part: how many clean
single-cut matched molecular pairs (MMPs) exist in the official ZINC-12k
*training* split, how they distribute over molecules / contexts / Murcko
scaffolds, and whether the observed deltas go beyond the known ZINC target
formula.

Hard data discipline (encoded, not just documented)
---------------------------------------------------
* Only the PyG ``ZINC(subset=True, split="train")`` rows (10,000) and the
  committed canonical-SMILES table for those rows are read.
* ``official valid/test``, the internal Monitor/Locked labels, ZINC-Full and
  any external labels are never loaded by this module.
* Phase 0A (structure + MMP construction + coverage) must be run, hashed and
  frozen *before* :func:`load_train_labels` is ever called.  The manifest
  written by ``--phase0a`` records the SHA-256 of every structure artifact and
  of this module; ``--phase0b`` refuses to run unless the manifest matches the
  current artifacts.
* The MMP construction code never receives ``y``; ``load_train_labels`` is a
  separate entry point used only by Phase 0B.

Frozen structural rules (see PROTOCOL.md; all fixed before any label read)
--------------------------------------------------------------------------
* exactly one cut, at an acyclic SINGLE bond (no ring cuts, no multiple cuts);
* each side of the cut is a connected fragment carrying exactly one dummy atom;
* both molecules carrying the replacement really exist in official train;
* direction ``R -> R'`` is fixed by lexicographic order of the canonical
  fragment keys, never by the label or by row order;
* primary regime ``P0``: context ``n_heavy(C) >= 5`` and replacement size
  ``1 <= n_heavy(R), n_heavy(R') <= 10``;
* sensitivity regimes: ``P0s`` (stricter sizes) and ``P1`` (context >= 3, no
  fragment cap, pair table deterministically capped per context for storage).

Everything label-free is computed in Phase 0A.  Phase 0B only *reads* the
frozen Phase 0A pair tables, loads the 10,000 official-train ``y`` values and
runs the delta statistics / simple baselines / target-formula audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK_ROOT = REPO_ROOT / "tracks/structure-contribution"
RESULTS_DIR = TRACK_ROOT / "results/bjoc_mmp_feasibility_v1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
SMILES_TABLE = (
    REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
)
SMILES_TABLE_PROVENANCE = SMILES_TABLE.with_name("train_canonical_smiles_provenance.json")
SASCORER_DIR = REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit/upstream"
FSCORES = SASCORER_DIR / "fpscores.pkl.gz"
SASCORER_SRC = SASCORER_DIR / "sascorer_gvae.py"

PROTOCOL_ID = "bjoc-mmp-feasibility-v1"
FROZEN_DATE = "2026-10-15"

# ---------------------------------------------------------------------------
# frozen structural extraction constants
# ---------------------------------------------------------------------------

#: official-train rows whose committed canonical SMILES is NOT the constitution
#: of the PyG graph (topology-only upstream matching picked another
#: constitutional isomer).  Established label-free in `--verify-data` and fixed
#: before any label read; these rows never enter the MMP analysis.
EXCLUDED_ROWS: tuple[int, ...] = (941, 2210, 6071, 9766)

#: primary local-edit regime
P0_MIN_CORE_HEAVY = 5
P0_MAX_FRAGMENT_HEAVY = 10
#: stricter size sensitivity
P0S_MIN_CORE_HEAVY = 8
P0S_MAX_FRAGMENT_HEAVY = 6
#: permissive regime: no fragment-size cap, materialised with a per-context cap
P1_MIN_CORE_HEAVY = 3
P1_MAX_FRAGMENT_HEAVY: int | None = None
P1_PER_CORE_PAIR_CAP = 200

#: deterministic ordering key for capped pair selection (label-free)
SUBSAMPLE_SALT = "bjoc-mmp-feasibility-v1|subsample"

# ---------------------------------------------------------------------------
# frozen coverage thresholds (pre-label; see PROTOCOL.md section 7)
# ---------------------------------------------------------------------------

COVERAGE_THRESHOLDS: dict[str, Any] = {
    "min_distinct_molecules": 1000,
    "min_mmp_pairs": 3000,
    "min_distinct_scaffolds": 300,
    "min_rules_support_ge2_contexts": 50,
    "min_rules_support_ge2_scaffolds": 20,
}

#: target-formula constants (community JTNN-era values; the GVAE generation
#: normalises over the full ZINC-250k with ddof=0).  The dataset constants from
#: the earlier long-cycle audit are reported alongside but are NOT recomputed
#: here (that would require reading ZINC-Full, which is out of bounds).
FORMULA_CONSTANTS_COMMUNITY: dict[str, float] = {
    "logP_mean": 2.4570953396190123,
    "logP_std": 1.4498737504491803,
    "SA_mean": -3.0536628609296634,
    "SA_std": 0.8315096218718826,
    "cycle_mean": -0.048569687328769765,
    "cycle_std": 0.2859726330705808,
}
#: long-cycle-audit recomputation over the full 249,456 molecules
#: (recorded artifact, not recomputed here)
FORMULA_CONSTANTS_DATASET: dict[str, float] = {
    "logP_mean": 2.4571235825155546,
    "logP_std": 1.4343304662156684,
    "SA_mean": -3.053303378915154,
    "SA_std": 0.8349004334000893,
    "cycle_mean": -0.049082804181899815,
    "cycle_std": 0.2902886264277354,
}

ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any, digest_size: int = 8) -> bytes:
    payload = "\x00".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).digest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[write] {path}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


# ---------------------------------------------------------------------------
# official-train data loading (labels and structures strictly separated)
# ---------------------------------------------------------------------------


def _load_pyg_train_tuple() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    path = ZINC_ROOT / "subset/processed/train.pt"
    _require(path.exists(), f"missing processed official-train file: {path}")
    obj = torch.load(path, weights_only=False, map_location="cpu")
    _require(isinstance(obj, tuple) and len(obj) == 3, "unexpected PyG processed layout")
    data_dict, slice_dict, _cls = obj
    for key in ("x", "edge_index", "edge_attr", "y"):
        _require(key in data_dict, f"missing PyG field {key!r}")
    return data_dict, slice_dict


def load_train_structures() -> list[dict[str, Any]]:
    """Official-train graphs only; ``y`` is deleted before any graph is built.

    Returns a list of dicts with ``x`` (int64 [n]), ``edge_index`` (int64
    [2, e]), ``edge_attr`` (int64 [e]) per row, in official subset_index order.
    """
    data_dict, slice_dict = _load_pyg_train_tuple()
    y = data_dict.pop("y")  # not part of Phase 0A; drop immediately
    del y
    xs = np.asarray(slice_dict["x"])
    eis = np.asarray(slice_dict["edge_index"])
    n_rows = len(xs) - 1
    _require(n_rows == 10000, f"expected 10000 official-train rows, got {n_rows}")
    structures: list[dict[str, Any]] = []
    for row in range(n_rows):
        x = np.asarray(data_dict["x"][xs[row] : xs[row + 1]]).reshape(-1).astype(np.int64)
        ei = np.asarray(data_dict["edge_index"][:, eis[row] : eis[row + 1]]).astype(np.int64)
        ea = np.asarray(data_dict["edge_attr"][eis[row] : eis[row + 1]]).reshape(-1).astype(np.int64)
        _require(ei.shape[0] == 2, "edge_index must be [2, e]")
        structures.append({"row": row, "x": x, "edge_index": ei, "edge_attr": ea})
    return structures


def load_train_labels() -> np.ndarray:
    """Official-train ``y`` only.  Must not be called from any Phase 0A path."""
    data_dict, slice_dict = _load_pyg_train_tuple()
    y = np.asarray(data_dict["y"], dtype=np.float64).reshape(-1)
    _require(y.shape == (10000,), f"label shape {y.shape}")
    return y


def load_smiles_table() -> list[str]:
    with np.load(SMILES_TABLE, allow_pickle=True) as handle:
        table = np.asarray(handle["smiles"], dtype=object)
    _require(table.shape == (10000,), f"SMILES table shape {table.shape}")
    return [str(s) for s in table.tolist()]


# ---------------------------------------------------------------------------
# RDKit reconstruction of the PyG graph (committed ZINC dictionaries)
# ---------------------------------------------------------------------------


def _zinc_words() -> tuple[list[str], list[str]]:
    sys.path.insert(0, str(TRACK_ROOT / "code"))
    import cscl_smiles_graph as csg

    return csg._zinc_dicts()


def build_rdkit_mol_from_graph(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> Chem.Mol:
    """RDKit molecule of one PyG ZINC graph.

    Implicit hydrogens are re-derived by the RDKit valence model; the committed
    ZINC atom dictionary only encodes element + explicit H + charge and cannot
    represent e.g. neutral ``[nH]``.  This reconstruction is used **only** for
    the label-free SMILES/PyG alignment verification and for composition
    features; all MMP chemistry uses the committed canonical SMILES.
    """
    atom_words, bond_words = _zinc_words()
    sys.path.insert(0, str(TRACK_ROOT / "code"))
    import cscl_smiles_graph as csg

    bond_types = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
    }
    rw = Chem.RWMol()
    for t in np.asarray(x).reshape(-1):
        element, hcount, charge = csg.parse_atom_word(atom_words[int(t)])
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(charge))
        atom.SetNumExplicitHs(int(hcount))
        atom.SetNoImplicit(False)
        rw.AddAtom(atom)
    seen: set[tuple[int, int]] = set()
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
        word = bond_words[int(edge_attr[k])]
        if word == "NONE":
            continue
        _require(word in bond_types, f"unsupported bond word {word!r}")
        rw.AddBond(u, v, bond_types[word])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def constitution(mol: Chem.Mol) -> str:
    """Stereochemistry-free canonical SMILES (constitution)."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


# ---------------------------------------------------------------------------
# Phase 0A-0: data integrity (label-free)
# ---------------------------------------------------------------------------


def verify_data_integrity(
    *,
    run_tiered_sample: int = 2000,
) -> dict[str, Any]:
    """SMILES <-> PyG graph alignment, parse/duplicate/count chemistry checks."""
    structures = load_train_structures()
    smiles = load_smiles_table()
    _require(len(structures) == len(smiles) == 10000, "row count mismatch")

    parse_failures: list[int] = []
    constitution_mismatches: list[dict[str, Any]] = []
    duplicate_groups: dict[str, list[int]] = defaultdict(list)
    n_atoms_mismatch = 0
    n_bonds_mismatch = 0
    for entry in structures:
        row = entry["row"]
        mol = Chem.MolFromSmiles(smiles[row])
        if mol is None:
            parse_failures.append(row)
            continue
        text = constitution(mol)
        duplicate_groups[text].append(row)
        graph_mol = build_rdkit_mol_from_graph(entry["x"], entry["edge_index"], entry["edge_attr"])
        graph_text = constitution(graph_mol)
        if graph_text != text:
            constitution_mismatches.append(
                {
                    "row": row,
                    "smiles_table": smiles[row],
                    "smiles_table_constitution": text,
                    "graph_constitution": graph_text,
                    "n_atoms_table": mol.GetNumAtoms(),
                    "n_atoms_graph": graph_mol.GetNumAtoms(),
                    "n_bonds_table": mol.GetNumBonds(),
                    "n_bonds_graph": graph_mol.GetNumBonds(),
                }
            )
        if mol.GetNumAtoms() != graph_mol.GetNumAtoms():
            n_atoms_mismatch += 1
        if mol.GetNumBonds() != graph_mol.GetNumBonds():
            n_bonds_mismatch += 1

    duplicate_smiles_groups = [
        {"rows": rows, "smiles": smiles[rows[0]]}
        for rows in duplicate_groups.values()
        if len(rows) > 1
    ]

    # atom / bond vocabulary audit (chemistry mapping)
    element_counter: Counter[str] = Counter()
    bond_counter: Counter[str] = Counter()
    stereo_rows = 0
    unassigned_stereo_rows = 0
    double_bond_stereo_rows = 0
    for row, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        element_counter.update(a.GetSymbol() for a in mol.GetAtoms())
        bond_counter.update(str(b.GetBondType()) for b in mol.GetBonds())
        if Chem.MolToSmiles(mol, isomericSmiles=True) != Chem.MolToSmiles(mol, isomericSmiles=False):
            stereo_rows += 1
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        if any(centre[1] == "?" for centre in centres):
            unassigned_stereo_rows += 1
        for bond in mol.GetBonds():
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                double_bond_stereo_rows += 1
                break

    x_counter: Counter[int] = Counter()
    edge_counter: Counter[int] = Counter()
    for entry in structures:
        x_counter.update(entry["x"].tolist())
        edge_counter.update(entry["edge_attr"].tolist())

    # independent tiered checker (hand-written parser + VF2) on a deterministic sample
    sys.path.insert(0, str(TRACK_ROOT / "code"))
    import cscl_smiles_graph as csg

    sample_rows = sorted(
        range(10000),
        key=lambda i: stable_hash("tiered", i),
    )[:run_tiered_sample]
    tier_counter: Counter[str] = Counter()
    kekulization_failed = 0
    tiered_failures: list[int] = []
    for row in sample_rows:
        entry = structures[row]
        try:
            res = csg.check_alignment(
                smiles[row],
                np.asarray(entry["x"]),
                np.asarray(entry["edge_index"]),
                np.asarray(entry["edge_attr"]),
            )
        except csg.SmilesParseError:
            tier_counter["parse_error"] += 1
            tiered_failures.append(row)
            continue
        if res.get("tier1_isomorphic"):
            tier_counter["tier1"] += 1
        elif res.get("tier2_isomorphic"):
            tier_counter["tier2"] += 1
        elif res.get("tier3_isomorphic"):
            tier_counter["tier3"] += 1
        else:
            tier_counter["none"] += 1
            tiered_failures.append(row)
        if res.get("kekulization_failed"):
            kekulization_failed += 1

    try:
        import rdkit

        rdkit_version = rdkit.__version__
    except Exception:  # pragma: no cover
        rdkit_version = "unknown"

    mismatch_rows = sorted({d["row"] for d in constitution_mismatches})
    excluded = sorted(set(EXCLUDED_ROWS))
    ok = (
        not parse_failures
        and mismatch_rows == excluded
        and n_atoms_mismatch == 0
        and n_bonds_mismatch == 0
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": "data-integrity",
        "frozen_date": FROZEN_DATE,
        "rows": 10000,
        "smiles_table": str(SMILES_TABLE.relative_to(REPO_ROOT)),
        "smiles_table_sha256": sha256_file(SMILES_TABLE),
        "smiles_table_provenance_sha256": sha256_file(SMILES_TABLE_PROVENANCE),
        "rdkit_version": rdkit_version,
        "rdkit_parse_failures": parse_failures,
        "constitution_mismatches": constitution_mismatches,
        "excluded_rows": excluded,
        "exclusion_reason": (
            "committed canonical SMILES constitution differs from the PyG graph; "
            "upstream topology-only pynauty matching cannot distinguish "
            "constitutional isomers with identical connectivity"
        ),
        "remaining_rows": 10000 - len(excluded),
        "atom_count_mismatch_rows": n_atoms_mismatch,
        "bond_count_mismatch_rows": n_bonds_mismatch,
        "duplicate_constitution_groups": duplicate_smiles_groups,
        "n_duplicate_constitution_groups": len(duplicate_smiles_groups),
        "n_unique_constitutions": len(duplicate_groups),
        "smiles_element_counts": dict(sorted(element_counter.items())),
        "smiles_bond_type_counts": dict(sorted(bond_counter.items())),
        "stereo_audit": {
            "rows_with_stereo_descriptors": stereo_rows,
            "rows_with_unassigned_stereo_centres": unassigned_stereo_rows,
            "rows_with_double_bond_stereo": double_bond_stereo_rows,
            "note": (
                "fragments and pairs are constitution-level (stereo stripped); "
                "pairs whose two molecules share a constitution are skipped because "
                "a stereo-only difference is not a structural edit"
            ),
        },
        "pyg_atom_type_counts": {str(k): v for k, v in sorted(x_counter.items())},
        "pyg_bond_type_counts": {str(k): v for k, v in sorted(edge_counter.items())},
        "tiered_check": {
            "sample_rows": len(sample_rows),
            "selection": "sha256('tiered|<row>') ascending",
            "tiers": dict(tier_counter),
            "kekulization_failed": kekulization_failed,
            "failure_rows": sorted(tiered_failures),
            "note": (
                "independent hand-written SMILES parser + kekulization + VF2 "
                "attributed isomorphism (cscl_smiles_graph.check_alignment); "
                "no RDKit in the original checker"
            ),
        },
        "verdict": "PASS_WITH_FROZEN_EXCLUSIONS" if ok else "FAIL",
        "verdict_note": (
            "9,996/10,000 rows verified at constitution level with RDKit "
            "reconstructions of the PyG graphs; 4 rows excluded (fixed before "
            "any label read)."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 0A: single-cut MMP construction
# ---------------------------------------------------------------------------


def cut_bond_indices(mol: Chem.Mol) -> list[int]:
    """Acyclic SINGLE bonds only (the frozen cut rule)."""
    out: list[int] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        out.append(bond.GetIdx())
    return out


def fragment_cut(mol: Chem.Mol, bond_idx: int) -> tuple[Chem.Mol, Chem.Mol]:
    """One single-cut fragmentation; both pieces carry exactly one dummy."""
    frag = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True, dummyLabels=[(1, 1)])
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    _require(len(pieces) == 2, f"single cut produced {len(pieces)} pieces")
    for piece in pieces:
        n_dummy = sum(1 for atom in piece.GetAtoms() if atom.GetAtomicNum() == 0)
        _require(n_dummy == 1, f"fragment has {n_dummy} dummy atoms")
    return pieces[0], pieces[1]


def canonical_fragment_smiles(fragment: Chem.Mol) -> str:
    return Chem.MolToSmiles(fragment, isomericSmiles=False)


def fragment_heavy_atoms(fragment_smiles: str) -> int:
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return -1
    return mol.GetNumAtoms() - 1


def attachment_environment(core_smiles: str) -> tuple:
    """Canonical local environment of the core-side attachment atom.

    ``(element, aromatic, charge, n_core_neighbours, total_H, sorted neighbours)``
    where each neighbour is ``(bond_type, element, aromatic, charge)``.  The
    dummy of the core fragment marks the attachment atom.
    """
    mol = Chem.MolFromSmiles(core_smiles)
    dummy = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    attach = dummy.GetNeighbors()[0]
    neighbours = []
    for nb in attach.GetNeighbors():
        if nb.GetAtomicNum() == 0:
            continue
        bond = mol.GetBondBetweenAtoms(attach.GetIdx(), nb.GetIdx())
        neighbours.append(
            (str(bond.GetBondType()), nb.GetSymbol(), bool(nb.GetIsAromatic()), int(nb.GetFormalCharge()))
        )
    return (
        attach.GetSymbol(),
        bool(attach.GetIsAromatic()),
        int(attach.GetFormalCharge()),
        len(neighbours),
        int(attach.GetTotalNumHs()),
        tuple(sorted(neighbours)),
    )


def environment_text(env: tuple) -> str:
    element, aromatic, charge, n_neighbours, n_h, neighbours = env
    charge_text = f"{charge:+d}" if charge else "0"
    return (
        f"{element}{'ar' if aromatic else 'al'}|q{charge_text}|deg{n_neighbours}|H{n_h}|"
        + ",".join(f"{b}-{e}{'ar' if ar else ''}{q:+d}" if q else f"{b}-{e}{'ar' if ar else ''}" for b, e, ar, q in neighbours)
    )


@dataclass(frozen=True)
class Incidence:
    row: int
    core: str
    chain: str
    core_heavy: int
    chain_heavy: int
    n_cut_realizations: int = 1


def extract_incidences(
    structures: Sequence[dict[str, Any]],
    smiles: Sequence[str],
    excluded: Iterable[int] = EXCLUDED_ROWS,
) -> tuple[dict[str, list[Incidence]], dict[str, Any]]:
    """All single-cut incidences (core, chain) for the verified rows.

    ``n_cut_realizations`` counts how many distinct cut bonds produce the same
    ``(row, core, chain)``; symmetric molecules therefore contribute a
    multiplicity (recorded for the pre-dedup accounting).
    """
    excluded_set = set(excluded)
    counter: dict[tuple[str, str, int], int] = defaultdict(int)
    inc_meta: dict[tuple[str, str, int], tuple[int, int]] = {}
    n_cuts = 0
    n_parse_fail = 0
    for entry in structures:
        row = entry["row"]
        if row in excluded_set:
            continue
        mol = Chem.MolFromSmiles(smiles[row])
        if mol is None:
            n_parse_fail += 1
            continue
        for bond_idx in cut_bond_indices(mol):
            n_cuts += 1
            first, second = fragment_cut(mol, bond_idx)
            first_key = canonical_fragment_smiles(first)
            second_key = canonical_fragment_smiles(second)
            first_heavy = first.GetNumAtoms() - 1
            second_heavy = second.GetNumAtoms() - 1
            counter[(first_key, second_key, row)] += 1
            inc_meta[(first_key, second_key, row)] = (first_heavy, second_heavy)
            counter[(second_key, first_key, row)] += 1
            inc_meta[(second_key, first_key, row)] = (second_heavy, first_heavy)
    core_members: dict[str, list[Incidence]] = defaultdict(list)
    for (core, chain, row), count in counter.items():
        core_heavy, chain_heavy = inc_meta[(core, chain, row)]
        core_members[core].append(
            Incidence(
                row=row,
                core=core,
                chain=chain,
                core_heavy=core_heavy,
                chain_heavy=chain_heavy,
                n_cut_realizations=count,
            )
        )
    audit = {"n_cuts": n_cuts, "n_parse_fail": n_parse_fail, "n_cores": len(core_members)}
    return dict(core_members), audit


@dataclass(frozen=True)
class MMPPair:
    row_r: int
    row_rp: int
    core: str
    r: str
    rp: str
    core_heavy: int
    r_heavy: int
    rp_heavy: int
    env: tuple
    scaffold_r: str
    scaffold_rp: str
    pre_dedup_variants: int


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    min_core_heavy: int
    max_fragment_heavy: int | None
    per_core_pair_cap: int | None


REGIMES: dict[str, RegimeSpec] = {
    "P0": RegimeSpec("P0", P0_MIN_CORE_HEAVY, P0_MAX_FRAGMENT_HEAVY, None),
    "P0s": RegimeSpec("P0s", P0S_MIN_CORE_HEAVY, P0S_MAX_FRAGMENT_HEAVY, None),
    "P1": RegimeSpec("P1", P1_MIN_CORE_HEAVY, P1_MAX_FRAGMENT_HEAVY, P1_PER_CORE_PAIR_CAP),
}


def _eligibility(inc: Incidence, spec: RegimeSpec) -> bool:
    if inc.core_heavy < spec.min_core_heavy:
        return False
    if spec.max_fragment_heavy is not None and inc.chain_heavy > spec.max_fragment_heavy:
        return False
    return inc.chain_heavy >= 1


def build_regime(
    core_members: dict[str, list[Incidence]],
    spec: RegimeSpec,
    *,
    same_constitution: dict[str, str],
    scaffolds: dict[int, str],
) -> dict[str, Any]:
    """Enumerate a regime exactly (support) and materialise its pair table.

    * ``pairs``: materialised rows (all rows for uncapped regimes, capped
      deterministically per context for capped ones);
    * support statistics are computed on the **uncapped** incidence structure
      (rules with support >= 2 can only be formed by chains appearing in >= 2
      eligible contexts, so candidate enumeration is exact and cheap).
    """
    core_entries: dict[str, dict[str, dict[int, int]]] = {}
    chain_multiplicity: Counter[str] = Counter()
    for core, members in core_members.items():
        by_chain: dict[str, dict[int, int]] = defaultdict(dict)
        for inc in members:
            if _eligibility(inc, spec):
                by_chain[inc.chain][inc.row] = inc.n_cut_realizations
        if len(by_chain) < 2:
            continue
        frozen = {chain: dict(sorted(rows.items())) for chain, rows in by_chain.items()}
        core_entries[core] = frozen
        for chain in frozen:
            chain_multiplicity[chain] += 1

    # analytic total pair count (distinct chains, distinct molecules) and
    # pre-dedup count (sum of symmetric cut-bond realizations)
    total_incidences = 0
    total_cut_realizations = 0
    n_skipped_same_constitution = 0
    for core, by_chain in core_entries.items():
        chains = sorted(by_chain)
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                rows_i, rows_j = by_chain[chains[i]], by_chain[chains[j]]
                for ra, mul_a in rows_i.items():
                    for rb, mul_b in rows_j.items():
                        if ra == rb:
                            continue
                        if same_constitution[ra] == same_constitution[rb]:
                            n_skipped_same_constitution += 1
                            continue
                        total_incidences += 1
                        total_cut_realizations += mul_a * mul_b

    # support>=2 rules (exact): both chains must have multiplicity >= 2
    candidate_chains = {chain for chain, mult in chain_multiplicity.items() if mult >= 2}
    rule_support_strict: dict[tuple[str, str], set[str]] = defaultdict(set)
    rule_support_env: dict[tuple[str, str, tuple], set[str]] = defaultdict(set)
    for core, by_chain in core_entries.items():
        cand = sorted(chain for chain in by_chain if chain in candidate_chains)
        if len(cand) < 2:
            continue
        env = attachment_environment(core)
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                rule_support_strict[(cand[i], cand[j])].add(core)
                rule_support_env[(cand[i], cand[j], env)].add(core)

    def support_summary(rule_map: dict) -> dict[str, Any]:
        values = np.asarray([len(v) for v in rule_map.values()], dtype=np.int64)
        n_repeated = int((values >= 2).sum()) if len(values) else 0
        n_repeated_inc = int(values[values >= 2].sum()) if len(values) else 0
        n_distinct = total_incidences - int(sum((len(v) - 1) for v in rule_map.values()))
        return {
            "support_values": values,
            "n_repeated_rules": n_repeated,
            "n_repeated_incidences": n_repeated_inc,
            "n_distinct_rules": n_distinct,
        }

    strict = support_summary(rule_support_strict)
    env_summary = support_summary(rule_support_env)

    # materialised pair table (no cap for P0/P0s; deterministic per-context cap for P1)
    heavy_cache: dict[str, int] = {}

    def heavy_of(fragment_smiles: str) -> int:
        if fragment_smiles not in heavy_cache:
            mol = Chem.MolFromSmiles(fragment_smiles)
            heavy_cache[fragment_smiles] = -1 if mol is None else mol.GetNumAtoms() - 1
        return heavy_cache[fragment_smiles]

    pairs: list[MMPPair] = []
    n_capped_cores = 0
    n_uncapped_pairs = 0
    cap = spec.per_core_pair_cap
    for core in sorted(core_entries):
        by_chain = core_entries[core]
        env = attachment_environment(core)
        raw: list[tuple[int, int, str, str, int]] = []
        chains = sorted(by_chain)
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                left, right = chains[i], chains[j]
                for ra, mul_a in by_chain[left].items():
                    for rb, mul_b in by_chain[right].items():
                        if ra == rb:
                            continue
                        if same_constitution[ra] == same_constitution[rb]:
                            continue
                        raw.append((ra, rb, left, right, mul_a * mul_b))
        n_uncapped_pairs += len(raw)
        if cap is not None and len(raw) > cap:
            n_capped_cores += 1
            raw.sort(key=lambda item: stable_hash(SUBSAMPLE_SALT, core, item[0], item[1], item[2], item[3]))
            raw = raw[:cap]
        variants: Counter[tuple[int, int, str, str]] = Counter()
        for ra, rb, left, right, multiplicity in raw:
            variants[(ra, rb, left, right)] += multiplicity
        for (ra, rb, left, right), count in sorted(variants.items()):
            pairs.append(
                MMPPair(
                    row_r=ra,
                    row_rp=rb,
                    core=core,
                    r=left,
                    rp=right,
                    core_heavy=heavy_of(core),
                    r_heavy=heavy_of(left),
                    rp_heavy=heavy_of(right),
                    env=env,
                    scaffold_r=scaffolds[ra],
                    scaffold_rp=scaffolds[rb],
                    pre_dedup_variants=count,
                )
            )
    _require(
        cap is not None or n_uncapped_pairs == total_incidences,
        f"analytic/materialised pair count mismatch in {spec.name}: {n_uncapped_pairs} != {total_incidences}",
    )

    # multi-cut granularity: how many distinct contexts express the same unordered molecule pair
    contexts_per_molecule_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    for pair in pairs:
        key = (min(pair.row_r, pair.row_rp), max(pair.row_r, pair.row_rp))
        contexts_per_molecule_pair[key].add(pair.core)

    return {
        "spec": spec,
        "pairs": pairs,
        "total_incidences": total_incidences,
        "total_cut_realizations": total_cut_realizations,
        "n_skipped_same_constitution": n_skipped_same_constitution,
        "n_uncapped_pairs": n_uncapped_pairs,
        "n_capped_cores": n_capped_cores,
        "strict": strict,
        "env": env_summary,
        "contexts_per_molecule_pair": contexts_per_molecule_pair,
    }


def rr_text(pair: MMPPair) -> str:
    return f"{pair.r}>>{pair.rp}"


def _coarse_class(pair: MMPPair) -> tuple:
    """Exploratory coarse class: attachment environment + element-count delta."""
    counts_r = _element_counts(pair.r)
    counts_rp = _element_counts(pair.rp)
    delta = tuple(
        sorted(
            (element, counts_rp.get(element, 0) - counts_r.get(element, 0))
            for element in set(counts_r) | set(counts_rp)
            if counts_rp.get(element, 0) - counts_r.get(element, 0) != 0
        )
    )
    return (pair.env, delta)


_element_count_cache: dict[str, Counter[str]] = {}


def _element_counts(fragment_smiles: str) -> Counter[str]:
    cached = _element_count_cache.get(fragment_smiles)
    if cached is None:
        mol = Chem.MolFromSmiles(fragment_smiles)
        cached = Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)
        _element_count_cache[fragment_smiles] = cached
    return cached


def compute_scaffolds(smiles: Sequence[str], excluded: Iterable[int] = EXCLUDED_ROWS) -> dict[int, str]:
    excluded_set = set(excluded)
    scaffolds: dict[int, str] = {}
    for row, smi in enumerate(smiles):
        if row in excluded_set:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffolds[row] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    return scaffolds


def regime_coverage(
    regime: dict[str, Any],
    spec: RegimeSpec,
    *,
    core_members: dict[str, list[Incidence]],
    scaffolds: dict[int, str],
) -> dict[str, Any]:
    pairs: list[MMPPair] = regime["pairs"]
    rows_used = {pair.row_r for pair in pairs} | {pair.row_rp for pair in pairs}
    molecule_pairs = {(min(p.row_r, p.row_rp), max(p.row_r, p.row_rp)) for p in pairs}
    scaffold_used = {scaffolds[row] for row in rows_used}
    cores_used = {pair.core for pair in pairs}
    strict_rules = {rr_text(pair) for pair in pairs}
    env_rules = {(pair.r, pair.rp, pair.env) for pair in pairs}

    contexts_hist = Counter(len(v) for v in regime["contexts_per_molecule_pair"].values())

    def support_block(summary: dict[str, Any]) -> dict[str, Any]:
        values: np.ndarray = summary["support_values"]
        return {
            "n_distinct_rules_exact": summary["n_distinct_rules"],
            "n_repeated_rules_exact": summary["n_repeated_rules"],
            "n_repeated_incidences_exact": summary["n_repeated_incidences"],
            "ge2": int((values >= 2).sum()) if len(values) else 0,
            "ge3": int((values >= 3).sum()) if len(values) else 0,
            "ge5": int((values >= 5).sum()) if len(values) else 0,
            "max": int(values.max()) if len(values) else 0,
        }

    strict_support = support_block(regime["strict"])
    env_support = support_block(regime["env"])

    # rule-level scaffold reuse: a rule must appear in at least two distinct
    # contexts (cores) to count as reused at all; scaffold support is measured
    # as the number of distinct scaffold pairs / scaffold values actually
    # observed for that rule
    rule_scaffold_pair_groups: dict[str, set[str]] = defaultdict(set)
    rule_scaffold_values: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        key = rr_text(pair)
        rule_scaffold_pair_groups[key].add("|".join(sorted([pair.scaffold_r, pair.scaffold_rp])))
        rule_scaffold_values[key].add(pair.scaffold_r)
        rule_scaffold_values[key].add(pair.scaffold_rp)
    scaffold_spread = Counter(min(len(v), 10) for v in rule_scaffold_values.values())
    n_rules_spanning_ge = {
        "ge2_contexts": strict_support["ge2"],
        "ge2_scaffold_pairs": sum(1 for v in rule_scaffold_pair_groups.values() if len(v) >= 2),
        "ge3_scaffold_pairs": sum(1 for v in rule_scaffold_pair_groups.values() if len(v) >= 3),
        "ge5_scaffold_pairs": sum(1 for v in rule_scaffold_pair_groups.values() if len(v) >= 5),
        "ge2_touched_scaffolds": sum(1 for v in rule_scaffold_values.values() if len(v) >= 2),
        "ge3_touched_scaffolds": sum(1 for v in rule_scaffold_values.values() if len(v) >= 3),
        "ge5_touched_scaffolds": sum(1 for v in rule_scaffold_values.values() if len(v) >= 5),
        "definition": (
            "contexts = distinct cores of a strict (R,R') rule; scaffold pairs/values "
            "are counted on the materialised table for that rule; touched-scaffold "
            "counts are descriptive (a single occurrence already touches its two "
            "scaffolds) and are NOT reuse evidence"
        ),
    }

    coarse: dict[tuple, set[str]] = defaultdict(set)
    for pair in pairs:
        coarse[_coarse_class(pair)].add(pair.core)
    coarse_support = np.asarray([len(v) for v in coarse.values()], dtype=np.int64)

    participation = Counter()
    for pair in pairs:
        participation[pair.row_r] += 1
        participation[pair.row_rp] += 1
    participation_values = np.asarray(list(participation.values()), dtype=np.int64) if participation else np.zeros(0)

    context_size = Counter(pair.core_heavy for pair in pairs)
    fragment_size = Counter(pair.r_heavy for pair in pairs) + Counter(pair.rp_heavy for pair in pairs)
    same_scaffold = sum(1 for p in pairs if p.scaffold_r == p.scaffold_rp)
    pre_dedup = sum(p.pre_dedup_variants for p in pairs)

    return {
        "regime": spec.name,
        "min_core_heavy": spec.min_core_heavy,
        "max_fragment_heavy": spec.max_fragment_heavy,
        "per_core_pair_cap": spec.per_core_pair_cap,
        "n_pairs": len(pairs),
        "n_pairs_pre_dedup_variants": pre_dedup,
        "n_cut_realizations_uncapped": regime["total_cut_realizations"],
        "n_skipped_same_constitution": regime["n_skipped_same_constitution"],
        "n_molecule_pairs": len(molecule_pairs),
        "n_distinct_molecules": len(rows_used),
        "n_distinct_cores": len(cores_used),
        "n_distinct_scaffolds": len(scaffold_used),
        "n_strict_rules": len(strict_rules),
        "n_env_rules": len(env_rules),
        "n_uncapped_pair_incidences": regime["total_incidences"],
        "n_materialised_pair_incidences": len(pairs),
        "n_capped_cores": regime["n_capped_cores"],
        "rule_support_contexts_strict": strict_support,
        "rule_support_contexts_env": env_support,
        "support_note": (
            "support statistics are exact on the uncapped incidence structure; "
            "scaffold-span and coarse-class statistics use the materialised table"
        ),
        "n_distinct_rules_exact": strict_support["n_distinct_rules_exact"],
        "n_repeated_rules_exact": strict_support["n_repeated_rules_exact"],
        "rule_scaffold_span": {f"n_scaffolds_{k}": v for k, v in sorted(scaffold_spread.items())},
        "rules_spanning": n_rules_spanning_ge,
        "same_murcko_scaffold_pairs": same_scaffold,
        "cross_murcko_scaffold_pairs": len(pairs) - same_scaffold,
        "contexts_per_molecule_pair_hist": {str(k): v for k, v in sorted(contexts_hist.items())},
        "coarse_class": {
            "n_classes": len(coarse),
            "support_ge2": int((coarse_support >= 2).sum()) if len(coarse_support) else 0,
            "support_ge3": int((coarse_support >= 3).sum()) if len(coarse_support) else 0,
            "support_ge5": int((coarse_support >= 5).sum()) if len(coarse_support) else 0,
            "max_support": int(coarse_support.max()) if len(coarse_support) else 0,
            "class_definition": "(attachment_environment, element_count_delta)",
        },
        "molecule_participation": {
            "n": len(rows_used),
            "mean": float(participation_values.mean()) if len(participation_values) else 0.0,
            "median": float(np.median(participation_values)) if len(participation_values) else 0.0,
            "p90": float(np.percentile(participation_values, 90)) if len(participation_values) else 0.0,
            "max": int(participation_values.max()) if len(participation_values) else 0,
        },
        "context_heavy_hist": {str(k): v for k, v in sorted(context_size.items())},
        "fragment_heavy_hist": {str(k): v for k, v in sorted(fragment_size.items())},
    }


def pairs_to_columns(pairs: Sequence[MMPPair]) -> dict[str, list[Any]]:
    return {
        "row_r": [p.row_r for p in pairs],
        "row_rp": [p.row_rp for p in pairs],
        "core": [p.core for p in pairs],
        "r": [p.r for p in pairs],
        "rp": [p.rp for p in pairs],
        "core_heavy": [p.core_heavy for p in pairs],
        "r_heavy": [p.r_heavy for p in pairs],
        "rp_heavy": [p.rp_heavy for p in pairs],
        "env": [environment_text(p.env) for p in pairs],
        "scaffold_r": [p.scaffold_r for p in pairs],
        "scaffold_rp": [p.scaffold_rp for p in pairs],
        "pre_dedup_variants": [p.pre_dedup_variants for p in pairs],
    }


def write_parquet(path: Path, columns: dict[str, list[Any]]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({key: pa.array(value) for key, value in columns.items()})
    pq.write_table(table, path, compression="zstd")
    print(f"[write] {path} ({table.num_rows} rows)")
    return sha256_file(path)


def read_parquet(path: Path) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {name: table.column(name).to_pylist() for name in table.column_names}


# ---------------------------------------------------------------------------
# Phase 0A driver
# ---------------------------------------------------------------------------


def run_phase0a(*, skip_integrity: bool = False) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not skip_integrity:
        integrity = verify_data_integrity()
        write_json(RESULTS_DIR / "data_integrity.json", integrity)
        _require(
            integrity["verdict"] != "FAIL",
            f"data integrity failed: {json.dumps(integrity, ensure_ascii=False)[:2000]}",
        )

    structures = load_train_structures()
    smiles = load_smiles_table()
    scaffolds = compute_scaffolds(smiles)
    core_members, cut_audit = extract_incidences(structures, smiles)
    same_constitution = {row: constitution(Chem.MolFromSmiles(smiles[row])) for row in scaffolds}

    coverage: dict[str, Any] = {}
    pair_files: dict[str, Path] = {}
    for name, spec in REGIMES.items():
        regime = build_regime(core_members, spec, same_constitution=same_constitution, scaffolds=scaffolds)
        coverage[name] = regime_coverage(
            regime, spec, core_members=core_members, scaffolds=scaffolds
        )
        path = RESULTS_DIR / f"phase0a_pairs_{name}.parquet"
        sha = write_parquet(path, pairs_to_columns(regime["pairs"]))
        pair_files[name] = path
        coverage[name]["pair_table_sha256"] = sha
        coverage[name]["pair_table_rows"] = len(regime["pairs"])

    # a-priori thresholds -> coverage verdict (fixed before any label read)
    primary = coverage["P0"]
    checks = {
        "min_distinct_molecules": primary["n_distinct_molecules"] >= COVERAGE_THRESHOLDS["min_distinct_molecules"],
        "min_mmp_pairs": primary["n_pairs"] >= COVERAGE_THRESHOLDS["min_mmp_pairs"],
        "min_distinct_scaffolds": primary["n_distinct_scaffolds"] >= COVERAGE_THRESHOLDS["min_distinct_scaffolds"],
        "min_rules_support_ge2_contexts": primary["rule_support_contexts_strict"]["ge2"]
        >= COVERAGE_THRESHOLDS["min_rules_support_ge2_contexts"],
        "min_rules_support_ge2_scaffolds": primary["rules_spanning"]["ge2_scaffold_pairs"]
        >= COVERAGE_THRESHOLDS["min_rules_support_ge2_scaffolds"],
    }
    coverage_verdict = "MMP_COVERAGE_SUFFICIENT" if all(checks.values()) else "MMP_COVERAGE_INSUFFICIENT"
    coverage["verdict"] = coverage_verdict
    coverage["thresholds"] = dict(COVERAGE_THRESHOLDS)
    coverage["threshold_checks"] = checks
    coverage["cut_audit"] = cut_audit
    coverage["excluded_rows"] = list(EXCLUDED_ROWS)

    pair_hashes = {name: sha256_file(path) for name, path in pair_files.items()}
    code_hash = sha256_file(Path(__file__))
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "stage": "phase0a-freeze",
        "frozen_date": FROZEN_DATE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels_read": False,
        "frozen_regimes": {
            name: {
                "min_core_heavy": spec.min_core_heavy,
                "max_fragment_heavy": spec.max_fragment_heavy,
                "per_core_pair_cap": spec.per_core_pair_cap,
            }
            for name, spec in REGIMES.items()
        },
        "coverage_thresholds": dict(COVERAGE_THRESHOLDS),
        "excluded_rows": list(EXCLUDED_ROWS),
        "pair_table_sha256": pair_hashes,
        "smiles_table_sha256": sha256_file(SMILES_TABLE),
        "pyg_processed_train_sha256": sha256_file(ZINC_ROOT / "subset/processed/train.pt"),
        "module_sha256": code_hash,
        "row_set": sorted(scaffolds.keys()),
    }
    write_json(RESULTS_DIR / "phase0a_coverage.json", coverage)
    write_json(RESULTS_DIR / "phase0a_manifest.json", manifest)
    print(f"[phase0a] done in {time.time() - started:.1f}s; verdict={coverage_verdict}")
    return {"coverage": coverage, "manifest": manifest}


# ---------------------------------------------------------------------------
# Phase 0B: property deltas, simple baselines, target audit
# ---------------------------------------------------------------------------


def _mean_abs(values: Sequence[float]) -> float:
    return float(np.mean(np.abs(values))) if len(values) else 0.0


def summarize_delta(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    nonzero = delta[delta != 0]
    return {
        "n": int(delta.size),
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "std": float(delta.std(ddof=0)),
        "min": float(delta.min()),
        "max": float(delta.max()),
        "abs_mean": _mean_abs(delta),
        "abs_median": float(np.median(np.abs(delta))),
        "quantiles": {
            "p01": float(np.percentile(delta, 1)),
            "p05": float(np.percentile(delta, 5)),
            "p25": float(np.percentile(delta, 25)),
            "p50": float(np.percentile(delta, 50)),
            "p75": float(np.percentile(delta, 75)),
            "p95": float(np.percentile(delta, 95)),
            "p99": float(np.percentile(delta, 99)),
        },
        "positive_fraction": float((delta > 0).mean()),
        "negative_fraction": float((delta < 0).mean()),
        "zero_fraction": float((delta == 0).mean()),
        "nonzero_positive_fraction": float((nonzero > 0).mean()) if nonzero.size else 0.0,
    }


def _sascorer_module():
    sys.path.insert(0, str(SASCORER_DIR))
    import sascorer_gvae  # noqa: E402

    return sascorer_gvae


def target_components(smiles: Sequence[str], rows: Sequence[int]) -> dict[int, dict[str, float]]:
    """Read-only recomputation of the two-member target terms (per row)."""
    import networkx as nx
    from rdkit.Chem import rdmolops

    sascorer = _sascorer_module()
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        mol = Chem.MolFromSmiles(smiles[row])
        logp = float(Descriptors.MolLogP(mol))
        sa = float(sascorer.calculateScore(mol))
        graph = nx.Graph(rdmolops.GetAdjacencyMatrix(mol))
        cycles = nx.cycle_basis(graph)
        cycle_length = 0 if not cycles else max(len(c) for c in cycles)
        cycle_penalty = 0.0 if cycle_length <= 6 else float(cycle_length - 6)
        out[row] = {"logP": logp, "sa": sa, "cycle": cycle_penalty}
    return out


def formula_y(components: dict[str, float], constants: dict[str, float]) -> float:
    return (
        (components["logP"] - constants["logP_mean"]) / constants["logP_std"]
        + (-components["sa"] - constants["SA_mean"]) / constants["SA_std"]
        + (-components["cycle"] - constants["cycle_mean"]) / constants["cycle_std"]
    )


def formula_delta(
    comp_a: dict[str, float], comp_b: dict[str, float], constants: dict[str, float]
) -> dict[str, float]:
    d_logp = comp_b["logP"] - comp_a["logP"]
    d_sa = -(comp_b["sa"] - comp_a["sa"])
    d_cycle = -(comp_b["cycle"] - comp_a["cycle"])
    return {
        "d_logP_raw": d_logp,
        "d_SAneg_raw": d_sa,
        "d_cyclepen_raw": d_cycle,
        "d_logP_scaled": d_logp / constants["logP_std"],
        "d_SAneg_scaled": d_sa / constants["SA_std"],
        "d_cyclepen_scaled": d_cycle / constants["cycle_std"],
        "d_formula": d_logp / constants["logP_std"]
        + d_sa / constants["SA_std"]
        + d_cycle / constants["cycle_std"],
    }


def composition_cache(smiles: Sequence[str], rows: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
    """Label-free composition descriptors for the composition-only baseline."""
    from rdkit.Chem import rdMolDescriptors

    cache: dict[int, dict[str, np.ndarray]] = {}
    for row in sorted(set(rows)):
        mol = Chem.MolFromSmiles(smiles[row])
        counts = np.zeros(len(ELEMENTS), dtype=np.float64)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ELEMENTS:
                counts[ELEMENTS.index(atom.GetSymbol())] += 1
        cache[row] = {
            "elements": counts,
            "ring_count": np.array([float(mol.GetRingInfo().NumRings())]),
            "aromatic_rings": np.array([float(rdMolDescriptors.CalcNumAromaticRings(mol))]),
            "charge": np.array([float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))]),
            "hbd": np.array([float(rdMolDescriptors.CalcNumHBD(mol))]),
            "hba": np.array([float(rdMolDescriptors.CalcNumHBA(mol))]),
            "rotatable": np.array([float(rdMolDescriptors.CalcNumRotatableBonds(mol))]),
        }
    return cache


def _pair_matrix(
    pairs: dict[str, list[Any]],
    composition_cache: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    names = (
        [f"d_{element}" for element in ELEMENTS]
        + ["d_rings", "d_aromatic_rings", "d_charge", "d_hbd", "d_hba", "d_rotatable", "d_heavy"]
    )
    rows = len(pairs["row_r"])
    matrix = np.zeros((rows, len(names)), dtype=np.float64)
    for i in range(rows):
        ra, rb = int(pairs["row_r"][i]), int(pairs["row_rp"][i])
        a, b = composition_cache[ra], composition_cache[rb]
        matrix[i, 0 : len(ELEMENTS)] = b["elements"] - a["elements"]
        offset = len(ELEMENTS)
        matrix[i, offset + 0] = (b["ring_count"] - a["ring_count"])[0]
        matrix[i, offset + 1] = (b["aromatic_rings"] - a["aromatic_rings"])[0]
        matrix[i, offset + 2] = (b["charge"] - a["charge"])[0]
        matrix[i, offset + 3] = (b["hbd"] - a["hbd"])[0]
        matrix[i, offset + 4] = (b["hba"] - a["hba"])[0]
        matrix[i, offset + 5] = (b["rotatable"] - a["rotatable"])[0]
        matrix[i, offset + 6] = float(int(pairs["rp_heavy"][i]) - int(pairs["r_heavy"][i]))
    return matrix, names


def _ols_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design_train = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    coefficients, *_ = np.linalg.lstsq(design_train, train_y, rcond=None)
    design_test = np.hstack([np.ones((test_x.shape[0], 1)), test_x])
    return design_test @ coefficients


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total <= 0:
        return 0.0
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - residual / total


def _groups_by_scaffold_pair(pairs: dict[str, list[Any]]) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    for sa, sb in zip(pairs["scaffold_r"], pairs["scaffold_rp"]):
        groups.append("|".join(sorted([sa, sb])))
    return groups, sorted(set(groups))


def leave_scaffold_out_evaluation(
    pairs: dict[str, list[Any]],
    delta_y: np.ndarray,
    descriptors: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Grouped (scaffold-pair) cross-validation of the frozen baselines."""
    groups, unique_groups = _groups_by_scaffold_pair(pairs)
    group_index = {group: i for i, group in enumerate(unique_groups)}
    assignments = np.asarray([group_index[g] for g in groups], dtype=np.int64)
    rng = np.random.default_rng(20261015)
    fold_of_group = rng.permutation(len(unique_groups)) % 5
    folds = fold_of_group[assignments]

    matrix, feature_names = _pair_matrix(pairs, descriptors)
    rules = [f"{r}>>{rp}" for r, rp in zip(pairs["r"], pairs["rp"])]

    predictions = {
        "zero": np.zeros_like(delta_y),
        "global_train_mean": np.zeros_like(delta_y),
        # primary unknown-rule fallback is 0 (null edit effect), frozen in PROTOCOL
        "rule_mean_fallback_zero": np.zeros_like(delta_y),
        # sensitivity: unseen rules fall back to the training global mean
        "rule_mean_fallback_global": np.zeros_like(delta_y),
        "composition_ols": np.zeros_like(delta_y),
    }
    unknown_rule_count = 0
    for fold in range(5):
        test = folds == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        predictions["global_train_mean"][test] = delta_y[train].mean()
        per_rule: dict[str, list[float]] = defaultdict(list)
        for rule, value in zip(np.asarray(rules)[train], delta_y[train]):
            per_rule[rule].append(float(value))
        rule_mean = {rule: float(np.mean(values)) for rule, values in per_rule.items()}
        for i in np.where(test)[0]:
            rule = rules[i]
            if rule in rule_mean:
                predictions["rule_mean_fallback_zero"][i] = rule_mean[rule]
                predictions["rule_mean_fallback_global"][i] = rule_mean[rule]
            else:
                unknown_rule_count += 1
                predictions["rule_mean_fallback_zero"][i] = 0.0
                predictions["rule_mean_fallback_global"][i] = float(delta_y[train].mean())
        predictions["composition_ols"][test] = _ols_predict(matrix[train], delta_y[train], matrix[test])

    out: dict[str, Any] = {
        "n_folds": 5,
        "group_unit": "murcko_scaffold_pair (unordered concatenation)",
        "n_groups": len(unique_groups),
        "n_pairs": int(len(delta_y)),
        "feature_names": feature_names,
        "unknown_rule_predictions": unknown_rule_count,
        "baselines": {
            name: {
                "mae": float(np.abs(delta_y - pred).mean()),
                "rmse": float(np.sqrt(np.mean((delta_y - pred) ** 2))),
                "r2": _r2(delta_y, pred),
                "mean_prediction": float(pred.mean()),
            }
            for name, pred in predictions.items()
        },
    }
    return out


def run_phase0b() -> dict[str, Any]:
    manifest = read_json(RESULTS_DIR / "phase0a_manifest.json")
    _require(manifest.get("labels_read") is False, "manifest is not a pre-label freeze")
    _require(sha256_file(Path(__file__)) == manifest["module_sha256"], "module changed after Phase 0A freeze")
    for name, digest in manifest["pair_table_sha256"].items():
        path = RESULTS_DIR / f"phase0a_pairs_{name}.parquet"
        _require(sha256_file(path) == digest, f"pair table {name} changed after freeze")
    _require(sha256_file(SMILES_TABLE) == manifest["smiles_table_sha256"], "SMILES table changed after freeze")

    smiles = load_smiles_table()
    descriptors = composition_cache(smiles, list(range(10000)))
    y = load_train_labels()
    components = target_components(smiles, sorted(manifest["row_set"]))

    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "stage": "phase0b",
        "label_source": "PyG ZINC(subset=True, split='train') y (official-train only)",
        "y_summary": summarize_delta(y),
        "target_reconstruction_per_molecule": {},
        "regimes": {},
    }

    # per-molecule formula reconstruction quality (bounds the achievable audit)
    for label, constants in (
        ("community", FORMULA_CONSTANTS_COMMUNITY),
        ("dataset", FORMULA_CONSTANTS_DATASET),
    ):
        residual = np.asarray(
            [y[row] - formula_y(components[row], constants) for row in manifest["row_set"]],
            dtype=np.float64,
        )
        result["target_reconstruction_per_molecule"][label] = {
            "constants": constants,
            "n": int(residual.size),
            "abs_mean": _mean_abs(residual),
            "abs_median": float(np.median(np.abs(residual))),
            "abs_p95": float(np.percentile(np.abs(residual), 95)),
            "abs_max": float(np.abs(residual).max()),
            "std": float(residual.std(ddof=0)),
        }

    for name in ("P0", "P1", "P0s"):
        pairs = read_parquet(RESULTS_DIR / f"phase0a_pairs_{name}.parquet")
        row_r = np.asarray(pairs["row_r"], dtype=np.int64)
        row_rp = np.asarray(pairs["row_rp"], dtype=np.int64)
        delta_y = y[row_rp] - y[row_r]

        entry: dict[str, Any] = {
            "n_pairs": int(delta_y.size),
            "delta_y": summarize_delta(delta_y),
        }

        # formula decomposition for both constant sets
        for label, constants in (
            ("community", FORMULA_CONSTANTS_COMMUNITY),
            ("dataset", FORMULA_CONSTANTS_DATASET),
        ):
            d_logp = np.zeros(delta_y.size)
            d_sa = np.zeros(delta_y.size)
            d_cycle = np.zeros(delta_y.size)
            for i in range(delta_y.size):
                comp = formula_delta(components[int(row_r[i])], components[int(row_rp[i])], constants)
                d_logp[i] = comp["d_logP_scaled"]
                d_sa[i] = comp["d_SAneg_scaled"]
                d_cycle[i] = comp["d_cyclepen_scaled"]
            d_formula = d_logp + d_sa + d_cycle
            residual = delta_y - d_formula
            design = np.column_stack([d_logp, d_sa, d_cycle])
            design_with_const = np.column_stack([np.ones_like(d_logp), design])
            coefficients, *_ = np.linalg.lstsq(design_with_const, delta_y, rcond=None)
            entry["target_formula"] = entry.get("target_formula", {})
            entry["target_formula"][label] = {
                "constants": constants,
                "component_scaled_delta": {
                    "logP": summarize_delta(d_logp),
                    "SAneg": summarize_delta(d_sa),
                    "cyclepen": summarize_delta(d_cycle),
                },
                "delta_formula": summarize_delta(d_formula),
                "residual_delta_y_minus_formula": summarize_delta(residual),
                "r2_formula_vs_delta_y": _r2(delta_y, d_formula),
                "corr_formula_vs_delta_y": float(np.corrcoef(d_formula, delta_y)[0, 1]),
                "ols_delta_y_on_components": {
                    "intercept": float(coefficients[0]),
                    "coef_logP": float(coefficients[1]),
                    "coef_SAneg": float(coefficients[2]),
                    "coef_cyclepen": float(coefficients[3]),
                    "r2": _r2(delta_y, design_with_const @ coefficients),
                },
                "mean_abs_component": {
                    "logP": float(np.abs(d_logp).mean()),
                    "SAneg": float(np.abs(d_sa).mean()),
                    "cyclepen": float(np.abs(d_cycle).mean()),
                },
            }

        # simple baselines with scaffold-pair grouped CV
        entry["baselines"] = leave_scaffold_out_evaluation(pairs, delta_y, descriptors)

        # coarse-class context analysis (exploratory)
        class_stats: dict[str, Any] = {}
        class_values: dict[tuple, list[float]] = defaultdict(list)
        class_cores: dict[tuple, set[str]] = defaultdict(set)
        for i in range(delta_y.size):
            key = (pairs["env"][i], _exploratory_class_delta(pairs["r"][i], pairs["rp"][i]))
            class_values[key].append(float(delta_y[i]))
            class_cores[key].add(pairs["core"][i])
        supports = {key: len(cores) for key, cores in class_cores.items()}
        repeated = [key for key, support in supports.items() if support >= 2]
        per_class = []
        for key in repeated:
            values = np.asarray(class_values[key], dtype=np.float64)
            per_class.append(
                {
                    "support_contexts": supports[key],
                    "n_occurrences": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "positive_fraction": float((values > 0).mean()),
                    "abs_mean": _mean_abs(values),
                }
            )
        class_stats["n_classes"] = len(supports)
        class_stats["n_repeated_classes"] = len(repeated)
        class_stats["repeated_class_summary"] = {
            "n": len(per_class),
            "mean_of_means": float(np.mean([c["mean"] for c in per_class])) if per_class else 0.0,
            "mean_within_std": float(np.mean([c["std"] for c in per_class])) if per_class else 0.0,
            "fraction_sign_consistent": float(
                np.mean([max(c["positive_fraction"], 1 - c["positive_fraction"]) for c in per_class])
            )
            if per_class
            else 0.0,
        }
        entry["coarse_class_context"] = class_stats
        result["regimes"][name] = entry

    write_json(RESULTS_DIR / "phase0b_analysis.json", result)
    return result


def _exploratory_class_delta(r: str, rp: str) -> tuple:
    counts_r = _element_counts(r)
    counts_rp = _element_counts(rp)
    return tuple(
        sorted(
            (element, counts_rp.get(element, 0) - counts_r.get(element, 0))
            for element in set(counts_r) | set(counts_rp)
            if counts_rp.get(element, 0) - counts_r.get(element, 0) != 0
        )
    )


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def decide(coverage: dict[str, Any], analysis: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    primary = coverage["P0"]
    checks = coverage["threshold_checks"]
    coverage_verdict = coverage["verdict"]

    p0 = analysis["regimes"]["P0"]
    formula = p0["target_formula"]["community"]
    recon = analysis["target_reconstruction_per_molecule"]["community"]
    formula_r2 = formula["r2_formula_vs_delta_y"]
    median_residual = formula["residual_delta_y_minus_formula"]["abs_median"]
    composition_r2 = p0["baselines"]["baselines"]["composition_ols"]["r2"]
    zero_r2 = p0["baselines"]["baselines"]["zero"]["r2"]
    formula_noise_floor = max(0.01, 2.0 * recon["abs_median"])

    if formula_r2 >= 0.90 and median_residual <= formula_noise_floor and composition_r2 >= 0.60:
        mechanism = "TARGET_FORMULA_DOMINATED"
    elif formula_r2 >= 0.90 and median_residual <= formula_noise_floor:
        mechanism = "TARGET_FORMULA_DOMINATED"
    elif formula_r2 < 0.50 and composition_r2 < 0.50:
        mechanism = "CONTEXT_DEPENDENCE_PLAUSIBLE"
    else:
        mechanism = "TARGET_MECHANISM_UNRESOLVED"

    independent_structure_ok = (
        integrity.get("verdict") != "FAIL"
        and integrity.get("atom_count_mismatch_rows", 1) == 0
        and integrity.get("bond_count_mismatch_rows", 1) == 0
    )
    repeated_ok = checks["min_rules_support_ge2_contexts"] and checks["min_rules_support_ge2_scaffolds"]
    if (
        coverage_verdict == "MMP_COVERAGE_SUFFICIENT"
        and independent_structure_ok
        and repeated_ok
        and mechanism == "CONTEXT_DEPENDENCE_PLAUSIBLE"
    ):
        next_step = "GO_TO_BASELINE_VALIDATION"
    elif mechanism == "TARGET_FORMULA_DOMINATED":
        next_step = "NO_GO_ON_ZINC"
    else:
        next_step = "INCONCLUSIVE"

    return {
        "A_coverage": coverage_verdict,
        "A_coverage_detail": {
            "threshold_checks": checks,
            "primary_regime": "P0",
            "primary_stats": {
                "n_pairs": primary["n_pairs"],
                "n_distinct_molecules": primary["n_distinct_molecules"],
                "n_distinct_scaffolds": primary["n_distinct_scaffolds"],
                "n_repeated_rules_exact": primary["n_repeated_rules_exact"],
                "rules_spanning_ge2_scaffold_pairs": primary["rules_spanning"]["ge2_scaffold_pairs"],
                "rules_spanning_ge2_contexts": primary["rules_spanning"]["ge2_contexts"],
            },
            "reason": (
                "coverage of *within-pair* local MMPs is broad, but exact cross-context "
                "rule repetition (the requirement for the core question) is absent"
                if not repeated_ok
                else "all frozen thresholds met"
            ),
        },
        "B_mechanism": mechanism,
        "B_mechanism_detail": {
            "formula_r2": formula_r2,
            "formula_residual_abs_median": median_residual,
            "per_molecule_reconstruction_abs_median": recon["abs_median"],
            "noise_floor_used": formula_noise_floor,
            "composition_ols_r2": composition_r2,
            "zero_baseline_r2": zero_r2,
        },
        "C_next": next_step,
        "explicit_claim": (
            "ZINC-12k official train contains no reusable exact local replacement: "
            "every distinct (R,R') rule occurs in exactly one context under the "
            "frozen single-cut definition, so cross-context effect transfer cannot "
            "be estimated at all; and the observed deltas are the known ZINC "
            "target formula. Do not use ZINC MMPs as evidence about real chemical "
            "substituent effects."
        ),
    }


# ---------------------------------------------------------------------------
# artifact hashing / summary
# ---------------------------------------------------------------------------


def build_summary() -> dict[str, Any]:
    integrity = read_json(RESULTS_DIR / "data_integrity.json")
    coverage = read_json(RESULTS_DIR / "phase0a_coverage.json")
    manifest = read_json(RESULTS_DIR / "phase0a_manifest.json")
    analysis = read_json(RESULTS_DIR / "phase0b_analysis.json")
    decisions = decide(coverage, analysis, integrity)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "frozen_date": FROZEN_DATE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_integrity": {
            "verdict": integrity["verdict"],
            "excluded_rows": integrity["excluded_rows"],
            "rdkit_parse_failures": len(integrity["rdkit_parse_failures"]),
            "rows_used": integrity["remaining_rows"],
            "tiered_check": integrity["tiered_check"],
        },
        "phase0a": {
            name: {
                key: value
                for key, value in coverage[name].items()
                if key
                in {
                    "n_pairs",
                    "n_molecule_pairs",
                    "n_distinct_molecules",
                    "n_distinct_cores",
                    "n_distinct_scaffolds",
                    "n_strict_rules",
                    "n_env_rules",
                    "n_uncapped_pair_incidences",
                    "n_capped_cores",
                    "rule_support_contexts_strict",
                    "rule_support_contexts_env",
                    "n_repeated_rules_exact",
                    "n_cut_realizations_uncapped",
                    "n_skipped_same_constitution",
                    "n_pairs_pre_dedup_variants",
                    "rules_spanning",
                    "coarse_class",
                    "cross_murcko_scaffold_pairs",
                    "same_murcko_scaffold_pairs",
                }
            }
            for name in ("P0", "P0s", "P1")
        },
        "phase0a_verdict": coverage["verdict"],
        "phase0b": {
            name: {
                "n_pairs": analysis["regimes"][name]["n_pairs"],
                "delta_y": analysis["regimes"][name]["delta_y"],
                "baselines": analysis["regimes"][name]["baselines"]["baselines"],
                "target_formula_community": analysis["regimes"][name]["target_formula"]["community"],
                "coarse_class_context": analysis["regimes"][name]["coarse_class_context"],
            }
            for name in ("P0", "P0s", "P1")
        },
        "target_reconstruction_per_molecule": analysis["target_reconstruction_per_molecule"],
        "decisions": decisions,
        "artifact_hashes": {
            "data_integrity.json": sha256_file(RESULTS_DIR / "data_integrity.json"),
            "phase0a_coverage.json": sha256_file(RESULTS_DIR / "phase0a_coverage.json"),
            "phase0a_manifest.json": sha256_file(RESULTS_DIR / "phase0a_manifest.json"),
            "phase0b_analysis.json": sha256_file(RESULTS_DIR / "phase0b_analysis.json"),
            "module": sha256_file(Path(__file__)),
            "smiles_table": sha256_file(SMILES_TABLE),
        },
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-data", action="store_true", help="run label-free SMILES/PyG alignment audit")
    parser.add_argument("--phase0a", action="store_true", help="build + freeze label-free MMP artifacts")
    parser.add_argument("--skip-integrity", action="store_true", help="with --phase0a: reuse frozen integrity JSON")
    parser.add_argument("--phase0b", action="store_true", help="read frozen pair tables + train labels")
    parser.add_argument("--summary", action="store_true", help="assemble summary.json")
    args = parser.parse_args(argv)
    if args.verify_data:
        write_json(RESULTS_DIR / "data_integrity.json", verify_data_integrity())
        return 0
    if args.phase0a:
        run_phase0a(skip_integrity=args.skip_integrity)
        return 0
    if args.phase0b:
        run_phase0b()
        return 0
    if args.summary:
        build_summary()
        return 0
    parser.error("no stage requested (use --verify-data / --phase0a / --phase0b / --summary)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
