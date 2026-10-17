"""Explicit-composition signal audit v1 — frozen structural objects (label-blind).

This module contains *only* structure-side definitions.  It never imports or
reads a target ``y``; the label stage lives in ``run_explicit_composition_signal_audit_v1.py``
and is executed only after ``FREEZE.json`` has been written and verified.

The three frozen information classes (see ``../PROTOCOL.md``):

B0  coarse structural marginals
    (n_heavy, n_bonds, atom-class counts, bond-class counts,
     bond line-graph degree distribution, per-bond junction-count
     distribution, number of junction atoms)

B1  multiset of exact rooted-bond local patches ``S`` (radius-1 bond-centred
    attributed patch with the two anchor endpoints marked), reused from
    ``structure_context_support_v1`` (T1 atom attributes).

C   true composition: the canonical coloured unit–atom incidence graph
    (unit = one ``S`` occurrence; atom = one heavy atom; incidence edge =
    membership of the atom in the unit patch, typed anchor/neighbor), plus an
    independent exact constitution key (non-isomeric canonical SMILES).

Everything is invariant to atom renumbering and bond order.  No labels, no
official valid/test, no model predictions are touched here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR.parent
REPO_ROOT = CODE_DIR.parents[4]
assert (REPO_ROOT / "AGENT.md").exists(), REPO_ROOT

# Reuse the frozen structure identity / canonicalisation / pairing code from
# the prior label-blind structural audit, rather than re-implementing it.
STRUCT_CONTEXT_CODE = (
    REPO_ROOT / "tracks/structure-contribution/results/structure_context_support_v1/code"
)
if str(STRUCT_CONTEXT_CODE) not in sys.path:
    sys.path.insert(0, str(STRUCT_CONTEXT_CODE))

from attributed_canon import canonical_form  # noqa: E402
from structure_context_support_v1 import (  # noqa: E402
    EXCLUDED_ZINC_ROWS,
    HETERO,
    bond_order_code,
    build_mol_rec,
    extract_instances,
    load_zinc_smiles,
    pairing_verification,
)

PROTOCOL_ID = "explicit-composition-signal-audit-v1"

# ---------------------------------------------------------------------------
# frozen inputs (hashes re-verified by the driver)
# ---------------------------------------------------------------------------

ZINC_SMILES_NPZ = (
    REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
)
ZINC_SMILES_SHA256 = "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff"
ZINC_TRAIN_PT = REPO_ROOT / "data/ZINC/subset/processed/train.pt"
ZINC_TRAIN_PT_SHA256 = "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b"

SASCORER_DIR = REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit/upstream"
FSCORES = SASCORER_DIR / "fpscores.pkl.gz"
SASCORER_SRC = SASCORER_DIR / "sascorer_gvae.py"

#: target-formula constants (identical to the bjoc-mmp-feasibility-v1 audit;
#: community JTNN-era values; the dataset constants are a sensitivity)
FORMULA_CONSTANTS_COMMUNITY: dict[str, float] = {
    "logP_mean": 2.4570953396190123,
    "logP_std": 1.4498737504491803,
    "SA_mean": -3.0536628609296634,
    "SA_std": 0.8315096218718826,
    "cycle_mean": -0.048569687328769765,
    "cycle_std": 0.2859726330705808,
}
FORMULA_CONSTANTS_DATASET: dict[str, float] = {
    "logP_mean": 2.4571235825155546,
    "logP_std": 1.4343304662156684,
    "SA_mean": -3.053303378915154,
    "SA_std": 0.8349004334000893,
    "cycle_mean": -0.049082804181899815,
    "cycle_std": 0.2902886264277354,
}

#: frozen exclusion list (constitutional isomers between the committed SMILES
#: table and the official PyG graph; established before any label read)
EXCLUDED = tuple(int(r) for r in EXCLUDED_ZINC_ROWS)

#: pre-registered matching tiers.  Definitions only; counts are Stage-A output.
TIER_DEFINITIONS: dict[str, str] = {
    "M01": "same B0 AND same B1(T1), different C  [strictest bag]",
    "M0": "same B0, different C  [strict marginal control; B1 may differ]",
    "M0_ring": "same B0 AND same n_rings, different C",
    "B1": "same B1(T1), different C  [local-patch bag only]",
    "A1": "same n_heavy+n_bonds+atom-class counts+bond-class counts+n_rings, different C "
          "[relaxes B0 line-graph-degree and junction-count distributions]",
    "A3": "same n_heavy+n_bonds+element counts+bond-class counts+n_rings+line-graph-degree "
          "distribution+junction-count distribution, different C "
          "[relaxes atom-class detail: H/charge/aromatic]",
    "A4": "same n_heavy+n_bonds+element counts+bond-class counts+n_rings, different C "
          "[large exploratory: relaxes atom-class detail and degree/junction distributions]",
    "L_old": "same (n_bonds, bond-class counts, line-graph-degree distribution, junction-count "
             "distribution), different C [the historical bond-RW coarse marginal signature; "
             "NOT composition-controlled]",
}


# ---------------------------------------------------------------------------
# serialisation / hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False, default=_json_default)
        handle.write("\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# exact canonical key for a coloured simple graph (pynauty)
# ---------------------------------------------------------------------------


def _pynauty():
    import pynauty

    return pynauty


def colored_graph_key(node_colors: Sequence[Any], edges: Sequence[tuple[int, int]]) -> bytes:
    """Exact canonical key of a simple graph with coloured vertices.

    The pynauty certificate alone encodes only adjacency; the canonical colour
    sequence is appended (same convention as the repo's other exact keys).
    """
    pynauty = _pynauty()
    nv = len(node_colors)
    adjacency: dict[int, list[int]] = {v: [] for v in range(nv)}
    for u, v in edges:
        u, v = int(u), int(v)
        adjacency[u].append(v)
        adjacency[v].append(u)
    groups: dict[Any, list[int]] = defaultdict(list)
    for v in range(nv):
        groups[node_colors[v]].append(v)
    cell_keys = sorted(groups, key=repr)
    coloring = [set(groups[k]) for k in cell_keys]
    pg = pynauty.Graph(
        number_of_vertices=nv,
        directed=False,
        adjacency_dict={k: list(v) for k, v in adjacency.items()},
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(pg))
    label = [int(v) for v in pynauty.canon_label(pg)]
    if sorted(label) != list(range(nv)):
        raise RuntimeError("pynauty.canon_label is not a permutation")
    colour_sequence = tuple(node_colors[v] for v in label)
    return certificate + b"|" + repr(colour_sequence).encode()


# ---------------------------------------------------------------------------
# structural signatures
# ---------------------------------------------------------------------------


def atom_class(atom) -> tuple[str, int, int, bool]:
    """T1 atom attribute reused from structure_context_support_v1."""
    return (
        atom.GetSymbol(),
        int(atom.GetFormalCharge()),
        int(atom.GetTotalNumHs()),
        bool(atom.GetIsAromatic()),
    )


def atom_class_counts(mol) -> tuple[tuple[tuple[str, int, int, bool], int], ...]:
    return tuple(sorted(Counter(atom_class(a) for a in mol.GetAtoms()).items()))


def element_counts(mol) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms()).items()))


def bond_class_counts(mol) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(Counter(bond_order_code(b) for b in mol.GetBonds()).items()))


def line_graph_degrees(mol) -> tuple[int, ...]:
    degrees = [a.GetDegree() for a in mol.GetAtoms()]
    return tuple(
        sorted(degrees[b.GetBeginAtomIdx()] + degrees[b.GetEndAtomIdx()] - 2 for b in mol.GetBonds())
    )


def junction_counts(mol) -> tuple[int, ...]:
    degrees = [a.GetDegree() for a in mol.GetAtoms()]
    return tuple(
        sorted(
            (1 if degrees[b.GetBeginAtomIdx()] >= 3 else 0)
            + (1 if degrees[b.GetEndAtomIdx()] >= 3 else 0)
            for b in mol.GetBonds()
        )
    )


def n_junctions(mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetDegree() >= 3)


def b0_key(mol) -> tuple:
    return (
        int(mol.GetNumHeavyAtoms()),
        int(mol.GetNumBonds()),
        atom_class_counts(mol),
        bond_class_counts(mol),
        line_graph_degrees(mol),
        junction_counts(mol),
        int(n_junctions(mol)),
    )


def b0_fields(mol) -> dict[str, Any]:
    return {
        "n_heavy": int(mol.GetNumHeavyAtoms()),
        "n_bonds": int(mol.GetNumBonds()),
        "atom_class_counts": [
            {"class": list(cls), "count": int(cnt)} for cls, cnt in atom_class_counts(mol)
        ],
        "bond_class_counts": [
            {"class": int(cls), "count": int(cnt)} for cls, cnt in bond_class_counts(mol)
        ],
        "line_graph_degree_distribution": list(line_graph_degrees(mol)),
        "junction_count_distribution": list(junction_counts(mol)),
        "n_junctions": int(n_junctions(mol)),
    }


def patch_atoms(mol, inst) -> set[int]:
    """The S patch: anchor bond endpoints plus all their immediate neighbours."""
    patch = {int(inst.u), int(inst.v)}
    patch |= {int(n.GetIdx()) for n in mol.GetAtomWithIdx(int(inst.u)).GetNeighbors()}
    patch |= {int(n.GetIdx()) for n in mol.GetAtomWithIdx(int(inst.v)).GetNeighbors()}
    return patch


def verify_patch_union(mol, instances: Sequence[Any]) -> bool:
    """The union of all S patches must contain every real bond (C is complete)."""
    all_bonds = {
        tuple(sorted((int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())))) for b in mol.GetBonds()
    }
    covered: set[tuple[int, int]] = set()
    for inst in instances:
        patch = patch_atoms(mol, inst)
        for u, v in all_bonds:
            if u in patch and v in patch:
                covered.add((u, v))
    return covered == all_bonds


def composition_key(mol, instances: Sequence[Any]) -> bytes:
    """Exact canonical key of the unit–atom incidence graph.

    Vertices are subdivided incidence nodes so the graph is a simple,
    vertex-coloured graph (pynauty has no edge-colour argument):
      * atom node        colour ("A", atom_class)
      * unit node        colour ("U", S_key)
      * incidence node   colour ("R", "anchor"|"neighbor")
    """
    atom_colors = [("A", atom_class(a)) for a in mol.GetAtoms()]
    unit_colors = [("U", inst.s_key) for inst in instances]
    node_colors: list[Any] = list(atom_colors) + list(unit_colors)
    edges: list[tuple[int, int]] = []
    n_atoms = int(mol.GetNumAtoms())
    for k, inst in enumerate(instances):
        unit_vertex = n_atoms + k
        anchors = {int(inst.u), int(inst.v)}
        for atom_idx in sorted(patch_atoms(mol, inst)):
            incidence_vertex = len(node_colors)
            node_colors.append(("R", "anchor" if atom_idx in anchors else "neighbor"))
            edges.append((unit_vertex, incidence_vertex))
            edges.append((atom_idx, incidence_vertex))
    return colored_graph_key(node_colors, edges)


def constitution_smiles(mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


# ---------------------------------------------------------------------------
# molecule record
# ---------------------------------------------------------------------------


@dataclass
class MoleculeRecord:
    row: int
    mol_uid: int
    smiles: str
    c_smiles: str
    c_key_hex: str
    scaffold: str
    topo: str
    formula: str
    n_heavy: int
    n_bonds: int
    n_rings: int
    atom_class_counts: tuple
    element_counts: tuple
    bond_class_counts: tuple
    line_graph_degrees: tuple
    junction_counts: tuple
    n_junctions: int
    b0_key: tuple
    b1_key: tuple
    b1_t2_key: tuple
    b1_t3_key: tuple
    b1_set: tuple[str, ...]
    reconstruction_ok: bool
    units: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def identity(self) -> dict[str, Any]:
        return {
            "row": int(self.row),
            "mol_uid": int(self.mol_uid),
            "smiles": self.smiles,
            "c_smiles": self.c_smiles,
            "c_key_sha256": self.c_key_hex,
            "scaffold": self.scaffold,
            "topo": self.topo,
            "formula": self.formula,
            "n_heavy": int(self.n_heavy),
            "n_bonds": int(self.n_bonds),
            "n_rings": int(self.n_rings),
        }

    def structure_json(self) -> dict[str, Any]:
        payload = self.identity()
        payload.update(
            {
                "atom_class_counts": [
                    {"class": list(cls), "count": int(cnt)} for cls, cnt in self.atom_class_counts
                ],
                "element_counts": [
                    {"element": el, "count": int(cnt)} for el, cnt in self.element_counts
                ],
                "bond_class_counts": [
                    {"class": int(cls), "count": int(cnt)} for cls, cnt in self.bond_class_counts
                ],
                "line_graph_degree_distribution": list(self.line_graph_degrees),
                "junction_count_distribution": list(self.junction_counts),
                "n_junctions": int(self.n_junctions),
                "b1_key_sha256": sha256_bytes(repr(self.b1_key).encode()),
                "b1_t2_key_sha256": sha256_bytes(repr(self.b1_t2_key).encode()),
                "b1_t3_key_sha256": sha256_bytes(repr(self.b1_t3_key).encode()),
                "b1_size": len(self.b1_key),
                "reconstruction_ok": bool(self.reconstruction_ok),
            }
        )
        return payload


def make_record(row: int, mol_uid: int, smiles: str, mol, rec) -> MoleculeRecord:
    from rdkit import Chem

    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    insts_t1 = extract_instances(rec, mol, "T1")
    insts_t2 = extract_instances(rec, mol, "T2")
    insts_t3 = extract_instances(rec, mol, "T3")
    c_key = composition_key(mol, insts_t1)
    return MoleculeRecord(
        row=int(row),
        mol_uid=int(mol_uid),
        smiles=str(smiles),
        c_smiles=constitution_smiles(mol),
        c_key_hex=sha256_bytes(c_key),
        scaffold=str(rec.scaffold),
        topo=str(rec.topo),
        formula=str(rec.heavy_formula),
        n_heavy=int(mol.GetNumHeavyAtoms()),
        n_bonds=int(mol.GetNumBonds()),
        n_rings=int(rec.n_rings),
        atom_class_counts=atom_class_counts(mol),
        element_counts=element_counts(mol),
        bond_class_counts=bond_class_counts(mol),
        line_graph_degrees=line_graph_degrees(mol),
        junction_counts=junction_counts(mol),
        n_junctions=int(n_junctions(mol)),
        b0_key=b0_key(mol),
        b1_key=tuple(sorted(inst.s_key for inst in insts_t1)),
        b1_t2_key=tuple(sorted(inst.s_key for inst in insts_t2)),
        b1_t3_key=tuple(sorted(inst.s_key for inst in insts_t3)),
        b1_set=tuple(sorted({inst.s_key for inst in insts_t1})),
        reconstruction_ok=bool(verify_patch_union(mol, insts_t1)),
        units=[
            {
                "u": int(inst.u),
                "v": int(inst.v),
                "s_key": inst.s_key,
                "patch_atoms": sorted(patch_atoms(mol, inst)),
            }
            for inst in insts_t1
        ],
    )


def build_records(limit: int | None = None) -> tuple[list[int], list[str], list[MoleculeRecord]]:
    """Load official-train structures and compute frozen signatures (no labels)."""
    from rdkit import Chem

    rows, smiles = load_zinc_smiles()
    records: list[MoleculeRecord] = []
    for uid, (row, smi) in enumerate(zip(rows, smiles)):
        if limit is not None and uid >= limit:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"RDKit parse failure at row {row}: {smi!r}")
        Chem.RemoveStereochemistry(mol)
        rec = build_mol_rec("zinc", int(row), int(uid), smi, mol)
        records.append(make_record(int(row), int(uid), str(smi), mol, rec))
    return rows[: len(records)], smiles[: len(records)], records


# ---------------------------------------------------------------------------
# grouping / pair statistics
# ---------------------------------------------------------------------------


def group_records(records: Sequence[MoleculeRecord], key_fn) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[key_fn(rec)].append(i)
    return dict(groups)


def _stable_group_id(tier: str, key: Any) -> str:
    return f"{tier}:{sha256_bytes(repr(key).encode())[:16]}"


def summarize_groups(
    records: Sequence[MoleculeRecord],
    groups: Mapping[Any, list[int]],
    tier: str,
    definition: str,
    *,
    max_pair_enumeration: int = 200_000,
) -> dict[str, Any]:
    """Counts at molecule/group/scaffold level; pairs are *not* independent samples."""
    diff_groups: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    n_pairs = 0
    n_scaffold_disjoint_pairs = 0
    n_pair_enum = 0
    pair_enum_capped = False
    n_molecules_diff = 0
    scaffolds_diff: set[str] = set()
    topo_diff: set[str] = set()
    max_group_size = max((len(v) for v in groups.values()), default=0)
    max_diff_group_size = 0

    for key, members in groups.items():
        c_counts = Counter(records[i].c_smiles for i in members)
        if len(c_counts) > 1:
            pairs: list[dict[str, Any]] = []
            group_pairs = (len(members) * len(members) - sum(c * c for c in c_counts.values())) // 2
            n_pairs += group_pairs
            max_diff_group_size = max(max_diff_group_size, len(members))
            n_molecules_diff += len(members)
            scaffolds_diff |= {records[i].scaffold for i in members}
            topo_diff |= {records[i].topo for i in members}
            if n_pair_enum + group_pairs <= max_pair_enumeration:
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        i, j = members[a], members[b]
                        if records[i].c_smiles == records[j].c_smiles:
                            continue
                        disjoint = records[i].scaffold != records[j].scaffold
                        n_scaffold_disjoint_pairs += int(disjoint)
                        pairs.append(
                            {
                                "row_a": int(records[i].row),
                                "row_b": int(records[j].row),
                                "mol_uid_a": int(records[i].mol_uid),
                                "mol_uid_b": int(records[j].mol_uid),
                                "scaffold_a": records[i].scaffold,
                                "scaffold_b": records[j].scaffold,
                                "scaffold_disjoint": bool(disjoint),
                                "C_a": records[i].c_smiles,
                                "C_b": records[j].c_smiles,
                            }
                        )
                n_pair_enum += group_pairs
            else:
                pair_enum_capped = True
            diff_groups.append(
                {
                    "group_id": _stable_group_id(tier, key),
                    "group_size": len(members),
                    "n_distinct_C": len(c_counts),
                    "n_pairs": int(group_pairs),
                    "members": [records[i].identity() for i in members],
                    "pairs": pairs,
                }
            )
        elif len(members) > 1:
            # Same constitution repeated in the table (duplicate rows).  These
            # are *not* "same bag, different C"; they measure label duplication.
            duplicate_groups.append(
                {
                    "group_id": _stable_group_id(tier, key),
                    "group_size": len(members),
                    "C": next(iter(c_counts)),
                    "members": [records[i].identity() for i in members],
                }
            )

    group_sizes = Counter(len(v) for v in groups.values())
    return {
        "tier": tier,
        "definition": definition,
        "n_molecules": len(records),
        "n_groups": len(groups),
        "n_singleton_groups": sum(1 for v in groups.values() if len(v) == 1),
        "n_multi_groups": sum(1 for v in groups.values() if len(v) > 1),
        "n_diffC_groups": len(diff_groups),
        "n_molecules_in_diffC_groups": int(n_molecules_diff),
        "n_scaffolds_in_diffC_groups": len(scaffolds_diff),
        "n_topologies_in_diffC_groups": len(topo_diff),
        "n_pairs_diffC": int(n_pairs),
        "n_scaffold_disjoint_pairs_diffC": int(n_scaffold_disjoint_pairs),
        "max_group_size": int(max_group_size),
        "max_diffC_group_size": int(max_diff_group_size),
        "group_size_histogram": {str(k): int(v) for k, v in sorted(group_sizes.items())},
        "pair_enumeration_capped": bool(pair_enum_capped),
        "diffC_groups": diff_groups,
        "sameC_duplicate_groups": duplicate_groups,
    }


# tier key functions (kept here so the proto-freeze and the run use one source)
def tier_key(tier: str):
    if tier == "M01":
        return lambda r: (r.b0_key, r.b1_key)
    if tier == "M0":
        return lambda r: r.b0_key
    if tier == "M0_ring":
        return lambda r: (r.b0_key, r.n_rings)
    if tier == "B1":
        return lambda r: r.b1_key
    if tier == "A1":
        return lambda r: (
            r.n_heavy,
            r.n_bonds,
            r.atom_class_counts,
            r.bond_class_counts,
            r.n_rings,
        )
    if tier == "A3":
        return lambda r: (
            r.n_heavy,
            r.n_bonds,
            r.element_counts,
            r.bond_class_counts,
            r.n_rings,
            r.line_graph_degrees,
            r.junction_counts,
            r.n_junctions,
        )
    if tier == "A4":
        return lambda r: (
            r.n_heavy,
            r.n_bonds,
            r.element_counts,
            r.bond_class_counts,
            r.n_rings,
        )
    if tier == "L_old":
        return lambda r: (
            r.n_bonds,
            r.bond_class_counts,
            r.line_graph_degrees,
            r.junction_counts,
        )
    raise KeyError(tier)


def build_tier(
    records: Sequence[MoleculeRecord], tier: str, *, max_pair_enumeration: int = 200_000
) -> dict[str, Any]:
    groups = group_records(records, tier_key(tier))
    return summarize_groups(
        records,
        groups,
        tier,
        TIER_DEFINITIONS[tier],
        max_pair_enumeration=max_pair_enumeration,
    )


def tier_pairs(tier_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in tier_payload["diffC_groups"]:
        for pair in group["pairs"]:
            item = dict(pair)
            item["group_id"] = group["group_id"]
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# bootstrap / descriptive statistics (used by Stage B)
# ---------------------------------------------------------------------------


def cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[Any],
    *,
    n_boot: int = 5000,
    seed: int = 20261016,
) -> dict[str, Any]:
    values_arr = np.asarray(values, dtype=np.float64)
    clusters_arr = np.asarray([str(c) for c in clusters])
    if values_arr.size == 0:
        return {"n": 0, "n_clusters": 0, "mean": None, "ci95": None}
    unique = list(dict.fromkeys(clusters_arr.tolist()))
    if len(unique) < 2:
        return {
            "n": int(values_arr.size),
            "n_clusters": len(unique),
            "mean": float(values_arr.mean()),
            "ci95": None,
            "note": "fewer than 2 clusters; no cluster bootstrap",
        }
    indices = {c: np.where(clusters_arr == c)[0] for c in unique}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        chosen = rng.choice(np.asarray(unique, dtype=object), size=len(unique), replace=True)
        idx = np.concatenate([indices[str(c)] for c in chosen])
        means[b] = float(values_arr[idx].mean())
    return {
        "n": int(values_arr.size),
        "n_clusters": len(unique),
        "mean": float(values_arr.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
        "cluster": "group",
    }


def summarize_numeric(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "mean_abs": float(abs_arr.mean()),
        "median_abs": float(np.median(abs_arr)),
        "p90_abs": float(np.percentile(abs_arr, 90)),
        "p95_abs": float(np.percentile(abs_arr, 95)),
        "max_abs": float(abs_arr.max()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "positive_fraction": float((arr > 0).mean()),
        "negative_fraction": float((arr < 0).mean()),
        "zero_fraction": float((arr == 0).mean()),
    }


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return None
    ss_res = float(((true - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def ols_with_intercept(y: Sequence[float], design: Sequence[Sequence[float]]) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    x = np.column_stack([np.ones(x.shape[0]), x])
    coef, *_ = np.linalg.lstsq(x, y_arr, rcond=None)
    pred = x @ coef
    return {
        "intercept": float(coef[0]),
        "coef": [float(v) for v in coef[1:]],
        "r2": r2_score(y_arr, pred),
        "n": int(y_arr.size),
        "rank": int(np.linalg.matrix_rank(x)),
        "n_columns": int(x.shape[1]),
    }
