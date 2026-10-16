"""structure-context-support-v1: structural data feasibility audit (label-free, CPU).

Round: ``structure-context-support-v1`` (frozen definitions in ``../PROTOCOL.md``).

Question: in the *official* ZINC-12k train structures (and, as an auxiliary
dataset, the frozen Lipophilicity train structures), does the same local
structure-attribute pattern (``S``) recur across molecules **and** does it
carry several independently supported, genuinely different structural relation
contexts (``C``)?

Hard boundaries (enforced by construction, asserted by tests):

* ZINC: only ``subset=True`` official **train** structures; the SMILES table is
  the committed, hash-checked, index-mapped table.  The official PyG train
  tensors are loaded **only** for the pairing spot-check and only
  ``x`` / ``edge_index`` / ``edge_attr`` are read -- ``y`` is never touched
  (the helper is unit-tested against a fake dataset whose ``y`` accessor
  raises).
* Lipophilicity: only the frozen label-free train objects
  (``objects_train.pt`` of ``bjoc_cross_task_lipophilicity_v1``), which carry
  SMILES + structure objects and no ``y`` field at all.
* No valid/test/locked split, no ZINC-Full, no model output, no embedding, no
  training, no GPU, no learned dictionary.

Run::

    uv run --group lipo python \\
        tracks/structure-contribution/results/structure_context_support_v1/code/structure_context_support_v1.py
"""

from __future__ import annotations

import argparse
import re
import hashlib
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR.parent
REPO_ROOT = CODE_DIR.parents[4]
assert (REPO_ROOT / "AGENT.md").exists(), REPO_ROOT
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from attributed_canon import canonical_form  # noqa: E402

# ---------------------------------------------------------------------------
# frozen inputs (hashes re-verified at run time; PROTOCOL.md §2)
# ---------------------------------------------------------------------------

ZINC_SMILES_NPZ = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ZINC_SMILES_SHA256 = "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff"
ZINC_TRAIN_PT = REPO_ROOT / "data/ZINC/subset/processed/train.pt"
ZINC_TRAIN_PT_SHA256 = "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b"
ZINC_ATOM_DICT = REPO_ROOT / "data/ZINC/raw/atom_dict.pickle"
MMP_MANIFEST = REPO_ROOT / "tracks/structure-contribution/results/bjoc_mmp_feasibility_v1/phase0a_manifest.json"
LIPO_OBJECTS_TRAIN = (
    REPO_ROOT / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/objects_train.pt"
)
LIPO_OBJECTS_TRAIN_SHA256 = "6d681cb0eaaab8c448564993c3740b24204e52e9e430f8c366189aa0061c7c86"

#: frozen exclusion list of the bjoc-mmp-feasibility-v1 audit: the committed
#: SMILES row is a constitutional isomer of the corresponding official PyG graph
#: (stereo-level only difference upstream).  Rows are dropped here as well, so
#: every analysis row is a verified (SMILES, official graph) pair.
EXCLUDED_ZINC_ROWS = (941, 2210, 6071, 9766)

HETERO = frozenset({"N", "O", "S", "F", "Cl", "Br", "I", "P"})

#: PyG ZINC subset atom vocabulary (raw ``atom_dict.pickle`` ``idx2word``);
#: used only by the label-free pairing spot-check.
PYG_ATOM_LABELS = (
    "C", "O", "N", "F", "C H1", "S", "Cl", "O -", "N H1 +", "Br", "N H3 +",
    "N H2 +", "N +", "N -", "S -", "I", "P", "O H1 +", "N H1 -", "O +", "S +",
    "P H1", "P H2", "C H2 -", "P +", "S H1 +", "C H1 -", "P H1 +",
)

#: ZINC bond type ids in the official tensors
PYG_BOND_ORDER = {1: 1, 2: 2, 3: 3}

# ---------------------------------------------------------------------------
# frozen definitions / decision rules (PROTOCOL.md §3, §6)
# ---------------------------------------------------------------------------

PRIMARY_TIER = "T1"
AUX_TIERS = ("T2", "T3")

DECISION_THRESHOLDS = {
    "C1_min_types_mol_support_5_scaffold_2": 10,
    "C2_min_types_two_contexts_each_2_molecules": 5,
    "C3_min_chemically_distinctive_types": 3,
    "C4_min_types_with_2_contexts_in_one_exact_stratum": 5,
    "support_curve_mol_thresholds": [1, 2, 3, 5, 10, 20, 50],
    "support_curve_scaffold_thresholds": [1, 2, 3, 5, 10],
    "context_diversity_min_mol_support": 5,
}

CT_MODES = ("primary", "coarse")

#: confound-control ladder (PROTOCOL.md §3.4): how much context diversity survives
#: progressively tighter molecule-level matching
STRATUM_LEVELS = ("size", "size_formula", "exact_stratum", "near_stratum")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def jbar(values: list[float], edges: tuple[float, ...]) -> dict:
    """Small histogram helper for reporting."""
    counts = Counter()
    for v in values:
        bucket = f">={edges[-1]}"
        for e in edges:
            if v < e:
                bucket = f"<{e}"
                break
        counts[bucket] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# data loading (label-free)
# ---------------------------------------------------------------------------


def load_zinc_smiles() -> tuple[list[int], list[str]]:
    with np.load(ZINC_SMILES_NPZ, allow_pickle=True) as blob:
        smiles = [str(s) for s in blob["smiles"]]
    assert len(smiles) == 10000, len(smiles)
    excluded = set(EXCLUDED_ZINC_ROWS)
    rows = [i for i in range(len(smiles)) if i not in excluded]
    return rows, [smiles[i] for i in rows]


def load_lipo_train_smiles() -> tuple[list[int], list[str]]:
    import torch

    track_code = REPO_ROOT / "tracks/structure-contribution/code"
    if str(track_code) not in sys.path:
        sys.path.insert(0, str(track_code))
    from structural_relation_objects_v1 import MolObjects  # noqa: F401  (unpickle class)

    objs = torch.load(LIPO_OBJECTS_TRAIN, map_location="cpu", weights_only=False)
    assert all(isinstance(o, MolObjects) for o in objs), "unexpected lipo object class"
    assert_label_free(objs)
    return [int(o.mol_index) for o in objs], [str(o.smiles) for o in objs]


def assert_label_free(objs) -> None:
    """The auxiliary dataset must not carry labels at all (PROTOCOL.md §1)."""
    labeled = [i for i, o in enumerate(objs) if hasattr(o, "y")]
    if labeled:
        raise AssertionError(f"objects carry a label attribute: {labeled[:5]}")


def load_pyg_structure_only(row_ids: list[int], blob=None) -> list[dict]:
    """Official PyG train structures: **``y`` is discarded before any access**.

    ``data/ZINC/subset/processed/train.pt`` is a PyG-collated
    ``({'x','edge_index','edge_attr','y'}, slices, Data)`` tuple.  The label
    tensor is popped from the structure dict immediately after load, before any
    row is touched, and only ``x``/``edge_index``/``edge_attr`` are read
    afterwards.  Tests inject a fake blob whose ``y`` raises on access, so the
    isolation is enforced rather than asserted.
    """
    import torch

    if blob is None:
        blob = torch.load(ZINC_TRAIN_PT, weights_only=False)
    data = dict(blob[0])
    slices = blob[1]
    data.pop("y", None)  # labels are discarded before any structure access
    assert "y" not in data, "label tensor must never be reachable from this loader"
    out = []
    for row in row_ids:
        a = int(slices["x"][row])
        b = int(slices["x"][row + 1])
        ea = int(slices["edge_index"][row])
        eb = int(slices["edge_index"][row + 1])
        out.append(
            {
                "row": row,
                "x": np.asarray(data["x"][a:b].reshape(-1).tolist(), dtype=np.int64),
                "edge_index": np.asarray(data["edge_index"][:, ea:eb].t().tolist(), dtype=np.int64),
                "edge_attr": np.asarray(data["edge_attr"][ea:eb].reshape(-1).tolist(), dtype=np.int64),
            }
        )
    return out


def parse_pyg_atom_label(label: str) -> tuple[str, int, int]:
    tokens = label.split()
    symbol = tokens[0]
    n_h = 0
    charge = 0
    for tok in tokens[1:]:
        if tok.startswith("H"):
            n_h = int(tok[1:])
        elif tok.startswith("+"):
            charge = int(tok[1:]) if len(tok) > 1 else 1
        elif tok.startswith("-"):
            charge = -(int(tok[1:]) if len(tok) > 1 else 1)
    return symbol, n_h, charge


def _graph_attrs_from_pyg(rec: dict):
    """Kekule (element, charge) graph of an official PyG structure record."""
    attrs = []
    for label in rec["x"]:
        symbol, _n_h, charge = parse_pyg_atom_label(PYG_ATOM_LABELS[int(label)])
        attrs.append((symbol, charge))
    seen: set[tuple[int, int]] = set()
    edges = []
    for (u, v), bt in zip(rec["edge_index"], rec["edge_attr"]):
        u, v = int(u), int(v)
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edges.append((key[0], key[1], (PYG_BOND_ORDER[int(bt)],)))
    return len(attrs), attrs, sorted(edges)


def _pyg_rebuilt_smiles(rec: dict) -> str:
    """RDKit molecule rebuilt from an official PyG record (no labels used).

    PyG ZINC stores a specific Kekule assignment and an atom vocabulary that only
    records explicitly written hydrogens, so the comparison is made after RDKit
    perceives aromaticity and infers implicit hydrogens from the valence.
    """
    from rdkit import Chem

    rw = Chem.RWMol()
    for label in rec["x"]:
        symbol, _n_h, charge = parse_pyg_atom_label(PYG_ATOM_LABELS[int(label)])
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        rw.AddAtom(atom)
    seen: set[tuple[int, int]] = set()
    for (u, v), bt in zip(rec["edge_index"], rec["edge_attr"]):
        u, v = int(u), int(v)
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        rw.AddBond(*key, {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[int(bt)])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def pairing_verification(n_rows: int | None = None, blob=None) -> dict:
    """Verify every kept ZINC row: table SMILES == official PyG train structure.

    ``n_rows=None`` checks all 10,000 rows (costs ~3 s on CPU).  Mismatches are
    reported; the frozen exclusion list must reproduce exactly.
    """
    from rdkit import Chem

    with np.load(ZINC_SMILES_NPZ, allow_pickle=True) as raw:
        all_smiles = [str(s) for s in raw["smiles"]]
    rows = list(range(len(all_smiles))) if n_rows is None else list(range(n_rows))
    structs = load_pyg_structure_only(rows, blob=blob)
    mismatches = []
    errors = []
    for rec in structs:
        row = rec["row"]
        try:
            rebuilt = _pyg_rebuilt_smiles(rec)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"row": row, "error": repr(exc)})
            continue
        table = Chem.CanonSmiles(all_smiles[row], useChiral=False)
        if rebuilt != table:
            mismatches.append(row)
    return {
        "n_rows_checked": len(rows),
        "n_mismatch": len(mismatches),
        "mismatch_rows": mismatches,
        "n_rdkit_errors": len(errors),
        "expected_mismatch_rows": list(EXCLUDED_ZINC_ROWS),
        "reproduces_frozen_exclusion_list": mismatches == list(EXCLUDED_ZINC_ROWS),
    }


# ---------------------------------------------------------------------------
# molecule records
# ---------------------------------------------------------------------------


@dataclass
class MolRec:
    dataset: str
    row: int
    mol_uid: int
    smiles: str
    n_heavy: int
    heavy_formula: str
    n_bonds: int
    n_rings: int
    n_ring_systems: int
    n_arom_rings: int
    scaffold: str
    topo: str


def _heavy_formula(mol) -> str:
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    return "".join(f"{sym}{counts[sym]}" for sym in sorted(counts))


def _generic_topology_smiles(mol) -> str:
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    generic = rw.GetMol()
    generic.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(generic)
    return Chem.MolToSmiles(generic, canonical=True, isomericSmiles=False)


def ring_systems(mol) -> tuple[list[list[int]], np.ndarray, Counter]:
    from rdkit import Chem

    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    parent = {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ring in rings:
        root = find(ring[0])
        for atom in ring[1:]:
            other = find(atom)
            if other != root:
                parent[max(root, other)] = min(root, other)
                root = find(root)
    sys_id = np.array([find(a.GetIdx()) for a in mol.GetAtoms()], dtype=np.int64)
    sizes = Counter(int(sys_id[i]) for ring in rings for i in ring)
    return rings, sys_id, sizes


def build_mol_rec(dataset: str, row: int, mol_uid: int, smiles: str, mol) -> MolRec:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    rings, _sys_id, _sizes = ring_systems(mol)
    bond_by_pair = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b for b in mol.GetBonds()
    }
    n_arom_rings = 0
    for ring in rings:
        pairs = [tuple(sorted((ring[i], ring[(i + 1) % len(ring)]))) for i in range(len(ring))]
        if all(bond_by_pair[p].GetIsAromatic() for p in pairs):
            n_arom_rings += 1
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "ACYCLIC"
    return MolRec(
        dataset=dataset,
        row=row,
        mol_uid=mol_uid,
        smiles=smiles,
        n_heavy=mol.GetNumHeavyAtoms(),
        heavy_formula=_heavy_formula(mol),
        n_bonds=mol.GetNumBonds(),
        n_rings=len(rings),
        n_ring_systems=len(set(int(v) for v in _sys_id)) if rings else 0,
        n_arom_rings=n_arom_rings,
        scaffold=scaffold,
        topo=_generic_topology_smiles(mol),
    )


# ---------------------------------------------------------------------------
# local structure identity S and contexts C
# ---------------------------------------------------------------------------

BOND_NAMES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 12: "AROMATIC"}


def bond_order_code(bond) -> int:
    bt = bond.GetBondTypeAsDouble()
    if bt == 1.5:
        return 12
    key = int(round(bt))
    assert key in BOND_NAMES, (bond.GetBondType(), bt)
    return key


def atom_attr(atom, tier: str):
    symbol = atom.GetSymbol()
    if tier == "T2":
        return (symbol,)
    if tier == "T3":
        return (symbol, atom.GetFormalCharge(), atom.GetIsAromatic())
    return (symbol, atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetIsAromatic())


def bond_attr(bond, tier: str):
    if tier == "T2":
        return ("SINGLE",) if bond_order_code(bond) == 1 else ("MULTIPLE",)
    return (BOND_NAMES[bond_order_code(bond)],)


@dataclass(slots=True)
class Instance:
    mol_uid: int
    row: int
    u: int
    v: int
    s_key: str
    ct_primary: str
    ct_coarse: str
    outer_key: str
    ring_key: tuple
    het_key: tuple
    ring_dist: int
    patch_has_hetero: bool
    anchor_in_ring: bool


def _bfs_dist(n: int, adj: list[list[tuple[int, tuple]]]) -> np.ndarray:
    dist = np.full((n, n), -1, dtype=np.int16)
    for src in range(n):
        dist[src, src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            base = dist[src, node] + 1
            for nxt, _ in adj[node]:
                if dist[src, nxt] < 0:
                    dist[src, nxt] = base
                    queue.append(nxt)
    return dist


def extract_instances(mol_rec: MolRec, mol, tier: str = PRIMARY_TIER) -> list[Instance]:
    """Every real bond of ``mol`` -> one local-structure instance with its contexts."""
    intern = sys.intern
    n = mol.GetNumAtoms()
    attrs = [atom_attr(a, tier) for a in mol.GetAtoms()]
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    adj: list[list[tuple[int, tuple]]] = [[] for _ in range(n)]
    bonds: list[tuple[int, int, tuple]] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u > v:
            u, v = v, u
        ba = bond_attr(bond, tier)
        bonds.append((u, v, ba))
        adj[u].append((v, ba))
        adj[v].append((u, ba))
    dist = _bfs_dist(n, adj)
    rings, sys_id, sys_sizes = ring_systems(mol)
    ring_atoms = {a for ring in rings for a in ring}
    bond_ring_sizes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ring in rings:
        for i in range(len(ring)):
            key = tuple(sorted((ring[i], ring[(i + 1) % len(ring)])))
            bond_ring_sizes[key].append(len(ring))

    out: list[Instance] = []
    for u, v, _ba in bonds:
        patch = {u, v}
        patch |= {j for j, _ in adj[u]}
        patch |= {j for j, _ in adj[v]}
        patch_list = sorted(patch)
        pos = {a: k for k, a in enumerate(patch_list)}
        s_key = canonical_form(
            len(patch_list),
            [attrs[a] for a in patch_list],
            sorted((pos[a], pos[b], ea) for a, b, ea in bonds if a in pos and b in pos),
            anchors=(pos[u], pos[v]),
        )

        ext = set(patch)
        for a in patch_list:
            ext |= {j for j, _ in adj[a]}
        ext_list = sorted(ext)
        epos = {a: k for k, a in enumerate(ext_list)}
        ext_attrs = []
        for a in ext_list:
            role = 0 if a in (u, v) else (1 if a in patch else 2)
            extra = (sum(1 for j, _ in adj[a] if j not in ext),) if role == 2 else ()
            ext_attrs.append((role,) + attrs[a] + extra)
        ext_key = canonical_form(
            len(ext_list),
            ext_attrs,
            sorted((epos[a], epos[b], ea) for a, b, ea in bonds if a in epos and b in epos),
            anchors=(epos[u], epos[v]),
        )

        outer_records = []
        for a in patch_list:
            for j, ea in adj[a]:
                if j in patch:
                    continue
                n_ext = sum(1 for k, _ in adj[j] if k not in ext)
                outer_records.append(f"{attrs[a]!r}<{attrs[j]!r}|{ea!r}|{n_ext}")
        outer_key = "|".join(sorted(outer_records))

        ring_sizes = bond_ring_sizes.get((u, v), [])
        anchor_in_ring = bool(ring_sizes)
        ring_key = (
            int(anchor_in_ring),
            min(ring_sizes) if ring_sizes else 0,
            len(ring_sizes),
            int(sys_sizes.get(int(sys_id[u]), 0)) if anchor_in_ring else 0,
        )

        hetero_out = [a for a in range(n) if elements[a] in HETERO and a not in patch]
        if hetero_out:
            d_het = min(int(min(dist[a, x] for a in patch_list)) for x in hetero_out)
            n_near = sum(1 for x in hetero_out if min(int(dist[a, x]) for a in patch_list) <= 2)
            het_key = (min(d_het, 4), min(n_near, 3))
        else:
            het_key = (-1, 0)

        ring_out = [a for a in ring_atoms if a not in patch]
        ring_dist = -1
        if ring_out:
            ring_dist = min(
                min(int(dist[a, x]) for x in ring_out) for a in patch_list
            )
            ring_dist = min(ring_dist, 4)

        ct_primary = intern(f"{ext_key}#{ring_key}#{het_key}#{ring_dist}")
        ct_coarse = intern(f"{outer_key}#{ring_key}#{het_key}#{ring_dist}")
        out.append(
            Instance(
                mol_uid=mol_rec.mol_uid,
                row=mol_rec.row,
                u=u,
                v=v,
                s_key=intern(s_key),
                ct_primary=ct_primary,
                ct_coarse=ct_coarse,
                outer_key=intern(outer_key),
                ring_key=ring_key,
                het_key=het_key,
                ring_dist=ring_dist,
                patch_has_hetero=any(elements[a] in HETERO for a in patch_list),
                anchor_in_ring=anchor_in_ring,
            )
        )
    return out


def instances_for_smiles(smiles: str, tier: str = PRIMARY_TIER) -> list[Instance]:
    """Single-molecule helper (used by the correctness tests; no data/ needed)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse {smiles!r}"
    rec = build_mol_rec("test", 0, 0, smiles, mol)
    return extract_instances(rec, mol, tier)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass
class TypeRow:
    s_key: str
    n_inst: int
    n_mols: int
    n_scaf: int
    n_topo: int
    n_ct: int
    n_ct_2mol: int
    n_ct_2scaf: int
    n_ct_2mol_2scaf: int
    n_ct_pair_disjoint_scaf: int
    n_outer: int
    n_ring_key: int
    n_het_key: int
    n_ring_dist: int
    max_scaf_share: float
    discordant_pairs: int
    matched_size: int
    matched_size_formula: int
    matched_size_formula_rings: int
    matched_near_stratum: int
    max_inst_per_molecule: int
    patch_has_hetero: bool
    anchor_in_ring: bool


def _strip_carbon(formula: str) -> str:
    """'C12Cl2N2O3S1' -> 'Cl2N2O3S1' (hetero-atom composition)."""
    match = re.match(r"^C(\d+)", formula)
    return formula[match.end():] if match else formula


def _matched_discordant_pairs(entries, key_fn) -> int:
    """Pairs of instances of one type that differ in context but share a stratum.

    ``entries`` = [(n_heavy, formula, n_rings, ct, count)]; exact closed form,
    no pair enumeration.  Counts only pairs whose two instances have different
    context types (same-context pairs are subtracted).
    """
    grouped: dict[tuple, dict[str, int]] = defaultdict(dict)
    for nh, formula, n_rings, ct, cnt in entries:
        key = key_fn(nh, formula, n_rings)
        per_ct = grouped[key]
        per_ct[ct] = per_ct.get(ct, 0) + cnt
    total = 0
    for per_ct in grouped.values():
        n_s = sum(per_ct.values())
        total += (n_s * n_s - sum(c * c for c in per_ct.values())) // 2
    return total


def aggregate(insts: list[Instance], mols: list[MolRec], tier: str, ct_mode: str = "primary") -> dict:
    """Per-structure-type aggregation of support, context diversity and confounds.

    Context types are the *primary* (exactly canonical radius-2 neighbourhood) or
    the *coarse* (outer-shell multiset) key, per ``PROTOCOL.md`` §3.3.  Two
    control strata are kept:

    * ``exact``: identical ``(n_heavy, heavy formula, n_rings)``;
    * ``near`` : ``n_heavy // 2``, identical hetero-atom composition and ring count
      (i.e. size differences of at most one heavy atom).
    """
    assert ct_mode in CT_MODES
    mol_by_uid = {m.mol_uid: m for m in mols}
    ct_attr = "ct_primary" if ct_mode == "primary" else "ct_coarse"

    type_inst: Counter[str] = Counter()
    type_mol: dict[str, set[int]] = defaultdict(set)
    type_scaf: dict[str, set[str]] = defaultdict(set)
    type_topo: dict[str, set[str]] = defaultdict(set)
    type_flags: dict[str, list[bool]] = defaultdict(lambda: [False, False])
    inst_per_mol: Counter[tuple[str, int]] = Counter()
    ct_mol: dict[tuple[str, str], set[int]] = defaultdict(set)
    ct_scaf: dict[tuple[str, str], set[str]] = defaultdict(set)
    ct_inst: Counter[tuple[str, str]] = Counter()
    scaf_inst: Counter[tuple[str, str]] = Counter()
    marginal: dict[str, dict[str, set]] = {
        "outer": defaultdict(set),
        "ring": defaultdict(set),
        "het": defaultdict(set),
        "ring_dist": defaultdict(set),
    }
    # (s_key, n_heavy, formula, n_rings, ct) -> instance count
    base: Counter[tuple[str, int, str, int, str]] = Counter()
    # confound-control ladder: (s_key, stratum, ct, molecule) memberships
    strata_members: dict[str, set[tuple[str, tuple, str, int]]] = {
        name: set() for name in STRATUM_LEVELS
    }

    for inst in insts:
        mol = mol_by_uid[inst.mol_uid]
        ct = getattr(inst, ct_attr)
        s_key = inst.s_key
        type_inst[s_key] += 1
        type_mol[s_key].add(inst.mol_uid)
        type_scaf[s_key].add(mol.scaffold)
        type_topo[s_key].add(mol.topo)
        flags = type_flags[s_key]
        flags[0] |= inst.patch_has_hetero
        flags[1] |= inst.anchor_in_ring
        ct_mol[(s_key, ct)].add(inst.mol_uid)
        ct_scaf[(s_key, ct)].add(mol.scaffold)
        ct_inst[(s_key, ct)] += 1
        scaf_inst[(s_key, mol.scaffold)] += 1
        inst_per_mol[(s_key, inst.mol_uid)] += 1
        marginal["outer"][s_key].add(inst.outer_key)
        marginal["ring"][s_key].add(inst.ring_key)
        marginal["het"][s_key].add(inst.het_key)
        marginal["ring_dist"][s_key].add(inst.ring_dist)
        base[(s_key, mol.n_heavy, mol.heavy_formula, mol.n_rings, ct)] += 1
        for level, key in zip(
            STRATUM_LEVELS,
            (
                (mol.n_heavy,),
                (mol.n_heavy, mol.heavy_formula),
                (mol.n_heavy, mol.heavy_formula, mol.n_rings),
                (mol.n_heavy // 2, _strip_carbon(mol.heavy_formula), mol.n_rings),
            ),
        ):
            strata_members[level].add((s_key, key, ct, inst.mol_uid))

    per_type_entries: dict[str, list[tuple[int, str, int, str, int]]] = defaultdict(list)
    for (s_key, n_heavy, formula, n_rings, ct), cnt in base.items():
        per_type_entries[s_key].append((n_heavy, formula, n_rings, ct, cnt))
    del base

    type_ct_inst: dict[str, dict[str, int]] = defaultdict(dict)
    for (s_key, ct), cnt in ct_inst.items():
        type_ct_inst[s_key][ct] = cnt

    rows: dict[str, TypeRow] = {}
    for s_key, n_inst in type_inst.items():
        ct_counts = type_ct_inst[s_key]
        cts = sorted(ct_counts)
        n_ct = len(cts)
        n_ct_2mol = sum(1 for ct in cts if len(ct_mol[(s_key, ct)]) >= 2)
        n_ct_2scaf = sum(1 for ct in cts if len(ct_scaf[(s_key, ct)]) >= 2)
        n_ct_2mol_2scaf = sum(
            1 for ct in cts if len(ct_mol[(s_key, ct)]) >= 2 and len(ct_scaf[(s_key, ct)]) >= 2
        )
        eligible = [ct for ct in cts if len(ct_mol[(s_key, ct)]) >= 2 and len(ct_scaf[(s_key, ct)]) >= 2]
        pair_disjoint = sum(
            1
            for i, cta in enumerate(eligible)
            for ctb in eligible[i + 1 :]
            if ct_scaf[(s_key, cta)].isdisjoint(ct_scaf[(s_key, ctb)])
        )
        entries = per_type_entries[s_key]
        discordant = (n_inst * n_inst - sum(c * c for c in ct_counts.values())) // 2
        matched = {
            "size": _matched_discordant_pairs(entries, lambda nh, f, r: (nh,)),
            "size_formula": _matched_discordant_pairs(entries, lambda nh, f, r: (nh, f)),
            "exact_stratum": _matched_discordant_pairs(entries, lambda nh, f, r: (nh, f, r)),
            "near_stratum": _matched_discordant_pairs(
                entries, lambda nh, f, r: (nh // 2, _strip_carbon(f), r)
            ),
        }
        scaf_max = max(v for (s, _sc), v in scaf_inst.items() if s == s_key)
        rows[s_key] = TypeRow(
            s_key=s_key,
            n_inst=n_inst,
            n_mols=len(type_mol[s_key]),
            n_scaf=len(type_scaf[s_key]),
            n_topo=len(type_topo[s_key]),
            n_ct=n_ct,
            n_ct_2mol=n_ct_2mol,
            n_ct_2scaf=n_ct_2scaf,
            n_ct_2mol_2scaf=n_ct_2mol_2scaf,
            n_ct_pair_disjoint_scaf=pair_disjoint,
            n_outer=len(marginal["outer"][s_key]),
            n_ring_key=len(marginal["ring"][s_key]),
            n_het_key=len(marginal["het"][s_key]),
            n_ring_dist=len(marginal["ring_dist"][s_key]),
            max_scaf_share=scaf_max / n_inst,
            discordant_pairs=discordant,
            matched_size=matched["size"],
            matched_size_formula=matched["size_formula"],
            matched_size_formula_rings=matched["exact_stratum"],
            matched_near_stratum=matched["near_stratum"],
            max_inst_per_molecule=max(v for (s, _m), v in inst_per_mol.items() if s == s_key),
            patch_has_hetero=type_flags[s_key][0],
            anchor_in_ring=type_flags[s_key][1],
        )

    strata: dict[str, dict[str, set[str]]] = {}
    for level in STRATUM_LEVELS:
        stratum_ct_mol: Counter[tuple[str, tuple, str]] = Counter()
        for s_key, stratum, ct, _mol_uid in strata_members[level]:
            stratum_ct_mol[(s_key, stratum, ct)] += 1
        strata_members[level] = set()  # free the raw memberships immediately
        bucket: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))
        for (s_key, stratum, ct), n_mol in stratum_ct_mol.items():
            if n_mol >= 2:
                bucket[s_key][stratum].add(ct)
        strata[level] = {
            s_key: sum(1 for cts in per.values() if len(cts) >= 2) for s_key, per in bucket.items()
        }
        del stratum_ct_mol, bucket

    return {
        "rows": rows,
        "strata": strata,
        "tier": tier,
        "ct_mode": ct_mode,
    }


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def support_curves(agg: dict) -> dict:
    rows: dict[str, TypeRow] = agg["rows"]
    mol_thresholds = DECISION_THRESHOLDS["support_curve_mol_thresholds"]
    scaf_thresholds = DECISION_THRESHOLDS["support_curve_scaffold_thresholds"]

    def curve(thresholds, key):
        out = []
        for t in thresholds:
            sel = [r for r in rows.values() if getattr(r, key) >= t]
            out.append(
                {
                    "threshold": t,
                    "n_types": len(sel),
                    "n_instances": sum(r.n_inst for r in sel),
                    "n_types_hetero_patch": sum(1 for r in sel if r.patch_has_hetero),
                    "n_types_ring_anchor": sum(1 for r in sel if r.anchor_in_ring),
                    "n_types_two_contexts_2mol": sum(1 for r in sel if r.n_ct_2mol >= 2),
                }
            )
        return out

    min_mol = DECISION_THRESHOLDS["context_diversity_min_mol_support"]
    hi = [r for r in rows.values() if r.n_mols >= min_mol]
    inst_hist = Counter(r.n_inst for r in rows.values())
    return {
        "n_structure_types": len(rows),
        "n_instances": sum(r.n_inst for r in rows.values()),
        "instance_count_histogram_head": {str(k): inst_hist[k] for k in sorted(inst_hist, reverse=True)[:15]},
        "molecule_support_curve": curve(mol_thresholds, "n_mols"),
        "scaffold_support_curve": curve(scaf_thresholds, "n_scaf"),
        "repeat_inflation": {
            "mean_instances_per_type_molecule": round(
                sum(r.n_inst for r in rows.values())
                / max(1, sum(r.n_mols for r in rows.values())),
                4,
            ),
            "mean_max_instances_per_molecule_mol5": round(
                float(np.mean([r.max_inst_per_molecule for r in hi])), 4
            )
            if hi
            else 0.0,
            "max_instances_per_molecule_overall": max(
                (r.max_inst_per_molecule for r in rows.values()), default=0
            ),
            "n_types_mol5_with_max_inst_per_mol_ge3": sum(
                1 for r in hi if r.max_inst_per_molecule >= 3
            ),
        },
        "context_diversity_at_mol5": {
            "n_types": len(hi),
            "n_types_n_ct_ge2": sum(1 for r in hi if r.n_ct >= 2),
            "n_types_n_ct_2mol_ge2": sum(1 for r in hi if r.n_ct_2mol >= 2),
            "n_types_n_ct_2scaf_ge2": sum(1 for r in hi if r.n_ct_2scaf >= 2),
            "n_types_n_ct_2mol_2scaf_ge2": sum(1 for r in hi if r.n_ct_2mol_2scaf >= 2),
            "n_types_pair_disjoint_scaffolds": sum(1 for r in hi if r.n_ct_pair_disjoint_scaf >= 1),
            "mean_n_ct": round(float(np.mean([r.n_ct for r in hi])), 3) if hi else 0.0,
            "mean_n_outer": round(float(np.mean([r.n_outer for r in hi])), 3) if hi else 0.0,
            "n_ct_histogram": dict(sorted(Counter(r.n_ct for r in hi).items())[:12]),
        },
    }


def decision_stats(agg: dict) -> dict:
    rows: dict[str, TypeRow] = agg["rows"]
    selected = [r for r in rows.values() if r.n_mols >= 5 and r.n_scaf >= 2]
    c2_rows = [r for r in selected if r.n_ct_2mol >= 2]
    c3 = sum(1 for r in c2_rows if r.patch_has_hetero)
    disc = sum(r.discordant_pairs for r in c2_rows)
    matched = {
        level: sum(getattr(r, f"matched_{level}") for r in c2_rows)
        for level in ("size", "size_formula", "size_formula_rings", "near_stratum")
    }
    ladder_types = {
        level: [r for r in c2_rows if agg["strata"][level].get(r.s_key, 0) >= 1]
        for level in STRATUM_LEVELS
    }
    return {
        "n_types_mol5_scaf2": len(selected),
        "n_types_mol5_scaf2_two_contexts_2mol": len(c2_rows),
        "n_types_mol5_scaf2_two_contexts_2mol_2scaf": sum(1 for r in c2_rows if r.n_ct_2scaf >= 2),
        "n_types_mol5_scaf2_two_contexts_2mol_2scaf_disjoint": sum(
            1 for r in c2_rows if r.n_ct_pair_disjoint_scaf >= 1
        ),
        "n_types_with_two_contexts_in_one_stratum": {
            level: len(rows_) for level, rows_ in ladder_types.items()
        },
        "stratum_types_by_level": {
            level: sum(agg["strata"][level].values()) for level in STRATUM_LEVELS
        },
        "n_chemically_distinctive_types": c3,
        "discordant_pairs": disc,
        "matched_pairs": matched,
        "matched_share": {
            level: (value / disc if disc else 0.0) for level, value in matched.items()
        },
        "c2_type_keys": sorted(r.s_key for r in c2_rows),
        "near_stratum_type_keys": sorted(r.s_key for r in ladder_types["near_stratum"]),
        "near_stratum_type_table": [
            _type_summary(r) for r in sorted(ladder_types["near_stratum"], key=lambda r: (-r.n_mols, r.s_key))
        ],
        "exact_stratum_type_keys": sorted(r.s_key for r in ladder_types["exact_stratum"]),
        "c2_type_table": [
            {
                "s_key": r.s_key,
                "n_inst": r.n_inst,
                "n_mols": r.n_mols,
                "n_scaf": r.n_scaf,
                "n_ct": r.n_ct,
                "n_ct_2mol": r.n_ct_2mol,
                "n_ct_2scaf": r.n_ct_2scaf,
                "max_scaf_share": round(r.max_scaf_share, 4),
                "patch_has_hetero": r.patch_has_hetero,
                "anchor_in_ring": r.anchor_in_ring,
            }
            for r in sorted(c2_rows, key=lambda r: (-r.n_mols, r.s_key))
        ],
    }


def triviality_tables(agg: dict) -> dict:
    """Confound diagnostics: composition of the high-support set + size/formula/ring strata."""
    rows = agg["rows"]
    hi = [r for r in rows.values() if r.n_mols >= 5]
    shares = [r.max_scaf_share for r in hi]
    return {
        "n_types_mol5": len(hi),
        "max_scaffold_share_histogram": jbar(shares, (0.5, 0.8, 0.95, 0.999)),
        "n_types_single_scaffold_family": sum(1 for r in hi if r.max_scaf_share >= 0.999),
        "n_types_mol5_n_scaf_ge2": sum(1 for r in hi if r.n_scaf >= 2),
        "n_types_mol5_n_topo_ge2": sum(1 for r in hi if r.n_topo >= 2),
        "n_types_mol5_n_scaf_ge5": sum(1 for r in hi if r.n_scaf >= 5),
    }


def chem_composition(agg: dict) -> dict:
    rows = agg["rows"]
    hi = [r for r in rows.values() if r.n_mols >= 5]
    return {
        "n_types_mol5": len(hi),
        "n_types_hetero_patch": sum(1 for r in hi if r.patch_has_hetero),
        "n_types_ring_anchor": sum(1 for r in hi if r.anchor_in_ring),
        "n_types_hetero_and_ring": sum(1 for r in hi if r.patch_has_hetero and r.anchor_in_ring),
        "n_types_c_c_chain_only": sum(1 for r in hi if not r.patch_has_hetero and not r.anchor_in_ring),
        "n_types_default_ring_only": sum(1 for r in hi if not r.patch_has_hetero and r.anchor_in_ring),
    }


def evaluate_criteria(decision: dict) -> dict:
    """Frozen criteria C1-C4 (PROTOCOL.md §6); no threshold is chosen post hoc."""
    t = DECISION_THRESHOLDS
    return {
        "C1_support_breadth": decision["n_types_mol5_scaf2"] >= t["C1_min_types_mol_support_5_scaffold_2"],
        "C2_independent_context_diversity": decision["n_types_mol5_scaf2_two_contexts_2mol"]
        >= t["C2_min_types_two_contexts_each_2_molecules"],
        "C3_chemical_distinctiveness": decision["n_chemically_distinctive_types"]
        >= t["C3_min_chemically_distinctive_types"],
        "C4_non_trivial_context_variation": decision["n_types_with_two_contexts_in_one_stratum"][
            "exact_stratum"
        ]
        >= t["C4_min_types_with_2_contexts_in_one_exact_stratum"],
    }


def decide(per_dataset: dict[str, dict], data_ok: bool) -> dict:
    """Combine the pre-registered criteria across the two datasets.

    ``CONTEXT_SUPPORT_PRESENT`` requires *one* dataset to satisfy all four
    criteria (the round brief: "ZINC 或 Lipophilicity 中至少一个数据集具备足够
    实例支持").  Otherwise the failure mode decides the verdict, and the verdict
    is always attributed to a single dataset: ``INSUFFICIENT_CROSS_GRAPH_SUPPORT``
    if no dataset has support breadth *and* independently supported contexts
    (C1&C2&C3); otherwise ``ONLY_TRIVIAL_CONTEXT_VARIATION`` (the dataset that has
    breadth and context diversity loses it under the composition control C4).
    """
    criteria = {name: evaluate_criteria(decision) for name, decision in per_dataset.items()}
    has_core = {
        name: c["C1_support_breadth"] and c["C2_independent_context_diversity"] and c["C3_chemical_distinctiveness"]
        for name, c in criteria.items()
    }
    if not data_ok:
        verdict = "INCONCLUSIVE"
    elif any(all(c.values()) for c in criteria.values()):
        verdict = "CONTEXT_SUPPORT_PRESENT"
    elif not any(has_core.values()):
        verdict = "INSUFFICIENT_CROSS_GRAPH_SUPPORT"
    else:
        verdict = "ONLY_TRIVIAL_CONTEXT_VARIATION"
    return {
        "verdict": verdict,
        "criteria": criteria,
        "thresholds": DECISION_THRESHOLDS,
        "combination_rule": (
            "CONTEXT_SUPPORT_PRESENT iff >=1 dataset satisfies C1&C2&C3&C4; "
            "else INSUFFICIENT_CROSS_GRAPH_SUPPORT if no dataset satisfies C1&C2&C3; "
            "else ONLY_TRIVIAL_CONTEXT_VARIATION."
        ),
    }


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------


def _patch_display(mol_kekule, patch_atoms: list[int], anchors: tuple[int, int]) -> str:
    from rdkit import Chem

    rw = Chem.RWMol()
    local = {a: k for k, a in enumerate(sorted(patch_atoms))}
    for a in sorted(patch_atoms):
        atom = mol_kekule.GetAtomWithIdx(a)
        new = Chem.Atom(atom.GetAtomicNum())
        new.SetFormalCharge(atom.GetFormalCharge())
        new.SetNoImplicit(True)
        new.SetNumExplicitHs(atom.GetTotalNumHs())
        rw.AddAtom(new)
    for bond in mol_kekule.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in local and v in local:
            rw.AddBond(local[u], local[v], bond.GetBondType())
    for a in sorted(patch_atoms):
        missing = [nb for nb in mol_kekule.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() not in local]
        for _ in missing:
            dummy = Chem.Atom(0)
            dummy.SetNoImplicit(True)
            dummy.SetNumExplicitHs(0)
            rw.AddBond(local[a], rw.AddAtom(dummy), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for k, a in enumerate(anchors):
        mol.GetAtomWithIdx(local[a]).SetAtomMapNum(k + 1)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def _mapped_smiles(mol, atoms: list[int]) -> str:
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for a in atoms:
        rw.GetAtomWithIdx(a).SetAtomMapNum(a + 1)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(out, canonical=True)


def build_examples(agg: dict, insts: list[Instance], mols: list[MolRec], smiles: list[str]) -> list[dict]:
    from rdkit import Chem

    rows: dict[str, TypeRow] = agg["rows"]
    mol_by_uid = {m.mol_uid: m for m in mols}
    by_type: dict[str, list[Instance]] = defaultdict(list)
    for inst in insts:
        by_type[inst.s_key].append(inst)

    def describe(inst: Instance) -> dict:
        mol = Chem.MolFromSmiles(smiles[inst.mol_uid])
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        patch = {inst.u, inst.v}
        patch |= {n.GetIdx() for n in kek.GetAtomWithIdx(inst.u).GetNeighbors()}
        patch |= {n.GetIdx() for n in kek.GetAtomWithIdx(inst.v).GetNeighbors()}
        rec = mol_by_uid[inst.mol_uid]
        bond = kek.GetBondBetweenAtoms(inst.u, inst.v)
        return {
            "row": inst.row,
            "mol_uid": inst.mol_uid,
            "smiles": smiles[inst.mol_uid],
            "mapped_smiles": _mapped_smiles(mol, [inst.u, inst.v]),
            "anchor_bond": [int(inst.u), int(inst.v)],
            "anchor_bond_type": BOND_NAMES[bond_order_code(bond)],
            "local_structure_smiles": _patch_display(kek, sorted(patch), (inst.u, inst.v)),
            "outer_records": inst.outer_key.split("|") if inst.outer_key else [],
            "ring_key": list(inst.ring_key),
            "het_key": list(inst.het_key),
            "ring_dist": int(inst.ring_dist),
            "n_heavy": rec.n_heavy,
            "heavy_formula": rec.heavy_formula,
            "n_rings": rec.n_rings,
            "scaffold": rec.scaffold,
            "topology_family": rec.topo,
            "primary_context_key": inst.ct_primary,
            "coarse_context_key": inst.ct_coarse,
        }

    examples: list[dict] = []
    ordered = sorted(rows.values(), key=lambda r: (-r.n_mols, r.s_key))

    # 1. same S in two different scaffolds
    cand = [r for r in ordered if r.n_scaf >= 2 and r.n_mols >= 5 and r.patch_has_hetero and r.n_ct_2mol >= 2]
    cand = cand or [r for r in ordered if r.n_scaf >= 2 and r.n_mols >= 5]
    if cand:
        row = cand[0]
        groups: dict[str, list[Instance]] = defaultdict(list)
        for inst in by_type[row.s_key]:
            groups[mol_by_uid[inst.mol_uid].scaffold].append(inst)
        best = sorted(groups, key=lambda s: (-len(groups[s]), s))[:2]
        examples.append(
            {
                "case": "same_S_two_different_scaffolds",
                "why": "同一局部结构在两个不同 Murcko 骨架中重复出现（跨骨架复用）",
                "type": _type_summary(row),
                "instances": [describe(sorted(groups[s], key=lambda i: i.row)[0]) for s in best],
                "scaffold_instance_counts": {s: len(groups[s]) for s in best},
            }
        )

    # 2. same S, two independently supported contexts
    cand2 = [r for r in ordered if r.n_ct_2mol >= 2 and r.n_mols >= 5 and r.patch_has_hetero]
    cand2 = cand2 or [r for r in ordered if r.n_ct_2mol >= 2]
    if cand2:
        row = cand2[0]
        per_ct: dict[str, list[Instance]] = defaultdict(list)
        for inst in by_type[row.s_key]:
            per_ct[inst.ct_primary].append(inst)
        order = sorted(per_ct, key=lambda c: (-len({i.mol_uid for i in per_ct[c]}), c))[:2]
        examples.append(
            {
                "case": "same_S_two_independent_contexts",
                "why": "同一局部结构上两个各自有 >=2 个独立分子支持的上下文类型",
                "type": _type_summary(row),
                "contexts": [
                    {
                        "n_instances": len(per_ct[c]),
                        "n_molecules": len({i.mol_uid for i in per_ct[c]}),
                        "n_scaffolds": len({mol_by_uid[i.mol_uid].scaffold for i in per_ct[c]}),
                        "example": describe(sorted(per_ct[c], key=lambda i: i.row)[0]),
                    }
                    for c in order
                ],
            }
        )

    # 3. same S, different *relation* axis (ring / hetero distance) at fixed local structure
    cand3 = [r for r in ordered if r.n_mols >= 5 and r.n_ring_dist >= 2 and r.n_het_key >= 2 and r.patch_has_hetero]
    if cand3:
        row = cand3[0]
        groups3: dict[int, list[Instance]] = defaultdict(list)
        for inst in by_type[row.s_key]:
            groups3[inst.ring_dist].append(inst)
        keys = sorted(groups3, key=lambda k: (-len(groups3[k]), k))[:2]
        examples.append(
            {
                "case": "same_S_different_relation_axis",
                "why": "同一局部结构，外部环/杂原子距离关系不同（radius-1 patch 自身不可见）",
                "type": _type_summary(row),
                "instances": [describe(sorted(groups3[k], key=lambda i: i.row)[0]) for k in keys],
                "ring_dist_values": {str(k): len(groups3[k]) for k in keys},
            }
        )

    # 4. high frequency, single context (counterexample)
    cand4 = sorted([r for r in rows.values() if r.n_ct_2mol <= 1], key=lambda r: (-r.n_inst, r.s_key))
    if cand4:
        row = cand4[0]
        first = sorted(by_type[row.s_key], key=lambda i: i.row)[0]
        examples.append(
            {
                "case": "high_frequency_single_context_counterexample",
                "why": "出现次数很多但只有 <=1 个有独立分子支持的上下文类型 -> 高频 != 上下文多样",
                "type": _type_summary(row),
                "instances": [describe(first)],
            }
        )

    # 5. high frequency, single scaffold family (counterexample)
    cand5 = sorted(
        [r for r in rows.values() if r.n_inst >= 50 and r.max_scaf_share >= 0.99],
        key=lambda r: (-r.n_inst, r.s_key),
    )
    if cand5:
        row = cand5[0]
        groups5: dict[str, list[Instance]] = defaultdict(list)
        for inst in by_type[row.s_key]:
            groups5[mol_by_uid[inst.mol_uid].scaffold].append(inst)
        top_scaf = max(groups5, key=lambda s: len(groups5[s]))
        picks = sorted(groups5[top_scaf], key=lambda i: i.row)[:2]
        examples.append(
            {
                "case": "high_frequency_single_family_counterexample",
                "why": "高频重复几乎全部来自同一骨架家族 -> 独立支持不足",
                "type": _type_summary(row),
                "scaffold": top_scaf,
                "scaffold_share": round(row.max_scaf_share, 4),
                "instances": [describe(i) for i in picks],
            }
        )
    return examples


def _type_summary(row: TypeRow) -> dict:
    return {
        "s_key": row.s_key,
        "n_instances": row.n_inst,
        "n_molecules": row.n_mols,
        "n_scaffolds": row.n_scaf,
        "n_topologies": row.n_topo,
        "n_context_types": row.n_ct,
        "n_context_types_with_2_molecules": row.n_ct_2mol,
        "n_context_types_with_2_scaffolds": row.n_ct_2scaf,
        "max_scaffold_share": round(row.max_scaf_share, 4),
        "patch_has_hetero": row.patch_has_hetero,
        "anchor_in_ring": row.anchor_in_ring,
        "marginal_context_counts": {
            "outer_shell": row.n_outer,
            "ring": row.n_ring_key,
            "hetero_relation": row.n_het_key,
            "ring_distance": row.n_ring_dist,
        },
    }


def cross_dataset(zinc_agg: dict, lipo_agg: dict) -> dict:
    z_rows = zinc_agg["rows"]
    l_rows = lipo_agg["rows"]
    shared = sorted(set(z_rows) & set(l_rows))
    return {
        "n_zinc_types": len(z_rows),
        "n_lipo_types": len(l_rows),
        "n_shared_types": len(shared),
        "n_shared_types_mol2_both": sum(1 for k in shared if z_rows[k].n_mols >= 2 and l_rows[k].n_mols >= 2),
        "n_shared_types_mol5_zinc_mol2_lipo": sum(
            1 for k in shared if z_rows[k].n_mols >= 5 and l_rows[k].n_mols >= 2
        ),
        "n_shared_types_two_independent_contexts_each": sum(
            1 for k in shared if z_rows[k].n_ct_2mol >= 2 and l_rows[k].n_ct_2mol >= 2
        ),
        "n_lipo_types_with_two_contexts_2mol": sum(1 for r in l_rows.values() if r.n_ct_2mol >= 2),
        "shared_type_examples": [
            {
                "s_key": k,
                "zinc": {"n_mols": z_rows[k].n_mols, "n_scaf": z_rows[k].n_scaf, "n_ct_2mol": z_rows[k].n_ct_2mol},
                "lipo": {"n_mols": l_rows[k].n_mols, "n_scaf": l_rows[k].n_scaf, "n_ct_2mol": l_rows[k].n_ct_2mol},
            }
            for k in shared
            if z_rows[k].n_mols >= 2 and l_rows[k].n_mols >= 2
        ][:20],
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def extract_dataset(name: str, rows: list[int], smiles: list[str], tiers: tuple[str, ...], limit: int | None):
    from rdkit import Chem

    mols: list[MolRec] = []
    insts: dict[str, list[Instance]] = {t: [] for t in tiers}
    bad_smiles: list[dict] = []
    n_multi_fragment = 0
    t0 = time.time()
    for uid, (row, smi) in enumerate(zip(rows, smiles)):
        if limit is not None and uid >= limit:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad_smiles.append({"row": row, "smiles": smi})
            continue
        if "." in smi:
            n_multi_fragment += 1
        rec = build_mol_rec(name, row, uid, smi, mol)
        mols.append(rec)
        for tier in tiers:
            insts[tier].extend(extract_instances(rec, mol, tier))
        if uid and uid % 1000 == 0:
            print(
                f"  [{name}] {uid} mols, {len(insts[tiers[0]])} instances, "
                f"rss={rss_mb():.0f}MB, {time.time() - t0:.0f}s",
                flush=True,
            )
    print(
        f"  [{name}] extracted {len(mols)} molecules / {len(insts[tiers[0]])} instances "
        f"in {time.time() - t0:.0f}s, rss={rss_mb():.0f}MB",
        flush=True,
    )
    return {
        "mols": mols,
        "insts": insts,
        "meta": {
            "dataset": name,
            "n_molecules": len(mols),
            "n_distinct_smiles": len({m.smiles for m in mols}),
            "n_instances_primary_tier": len(insts[tiers[0]]),
            "n_multi_fragment_smiles": n_multi_fragment,
            "bad_smiles": bad_smiles,
            "seconds_extract": time.time() - t0,
        },
    }


def pack_dataset(name: str, mols: list[MolRec], insts: list[Instance], z: dict, ct_mode: str = "primary") -> None:
    ct_attr = "ct_primary" if ct_mode == "primary" else "ct_coarse"
    s_ids: dict[str, int] = {}
    ct_ids: dict[str, int] = {}
    outer_ids: dict[str, int] = {}
    ring_ids: dict[tuple, int] = {}
    het_ids: dict[tuple, int] = {}
    arr = {k: np.empty(len(insts), dtype=dt) for k, dt in (
        ("mol_uid", np.int64), ("row", np.int32), ("u", np.int16), ("v", np.int16),
        ("s_id", np.int32), ("ct_id", np.int32), ("outer_id", np.int32), ("ring_id", np.int16),
        ("het_id", np.int16), ("ring_dist", np.int8),
    )}
    for k, inst in enumerate(insts):
        arr["mol_uid"][k] = inst.mol_uid
        arr["row"][k] = inst.row
        arr["u"][k] = inst.u
        arr["v"][k] = inst.v
        arr["s_id"][k] = s_ids.setdefault(inst.s_key, len(s_ids))
        arr["ct_id"][k] = ct_ids.setdefault(getattr(inst, ct_attr), len(ct_ids))
        arr["outer_id"][k] = outer_ids.setdefault(inst.outer_key, len(outer_ids))
        arr["ring_id"][k] = ring_ids.setdefault(inst.ring_key, len(ring_ids))
        arr["het_id"][k] = het_ids.setdefault(inst.het_key, len(het_ids))
        arr["ring_dist"][k] = inst.ring_dist
    for key, values in arr.items():
        z[f"{name}__{key}"] = values
    z[f"{name}__s_keys"] = np.array(list(s_ids), dtype=object)
    z[f"{name}__ct_keys"] = np.array(list(ct_ids), dtype=object)
    z[f"{name}__outer_keys"] = np.array(list(outer_ids), dtype=object)
    z[f"{name}__ring_keys"] = np.array([repr(k) for k in ring_ids], dtype=object)
    z[f"{name}__het_keys"] = np.array([repr(k) for k in het_ids], dtype=object)
    z[f"{name}__mol_row"] = np.array([m.row for m in mols], dtype=np.int32)
    z[f"{name}__mol_smiles"] = np.array([m.smiles for m in mols], dtype=object)
    z[f"{name}__mol_scaffold"] = np.array([m.scaffold for m in mols], dtype=object)
    z[f"{name}__mol_topo"] = np.array([m.topo for m in mols], dtype=object)
    z[f"{name}__mol_stats"] = np.array(
        [[m.n_heavy, m.n_bonds, m.n_rings, m.n_ring_systems, m.n_arom_rings] for m in mols],
        dtype=np.int32,
    )


def write_type_tables(analysis_dir: Path, datasets: dict) -> None:
    for name, res in datasets.items():
        data = res["agg"]
        payload = []
        for r in sorted(data["rows"].values(), key=lambda r: (-r.n_mols, r.s_key)):
            if r.n_mols < 2:
                continue
            payload.append(
                [
                    r.s_key, r.n_inst, r.n_mols, r.n_scaf, r.n_topo, r.n_ct, r.n_ct_2mol,
                    r.n_ct_2scaf, r.n_ct_2mol_2scaf, r.n_ct_pair_disjoint_scaf, r.n_outer,
                    r.n_ring_key, r.n_het_key, r.n_ring_dist, round(r.max_scaf_share, 4),
                    r.discordant_pairs, r.matched_size, r.matched_size_formula,
                    r.matched_size_formula_rings, r.matched_near_stratum, int(r.patch_has_hetero),
                    int(r.anchor_in_ring),
                ]
            )
        with (analysis_dir / f"types_{name}.json").open("w") as fh:
            json.dump(
                {
                    "columns": [
                        "s_key", "n_inst", "n_mols", "n_scaf", "n_topo", "n_ct", "n_ct_2mol",
                        "n_ct_2scaf", "n_ct_2mol_2scaf", "n_ct_pair_disjoint_scaf", "n_outer",
                        "n_ring_key", "n_het_key", "n_ring_dist", "max_scaf_share",
                        "discordant_pairs", "matched_size", "matched_size_formula",
                        "matched_size_formula_rings", "matched_near_stratum", "patch_has_hetero",
                        "anchor_in_ring",
                    ],
                    "rows": payload,
                },
                fh,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test molecule cap per dataset")
    parser.add_argument("--skip-spotcheck", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-instances", action="store_true", help="skip instances.npz (smoke tests)")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    integrity = {
        "frozen_inputs": {
            "zinc_smiles_npz": {
                "path": str(ZINC_SMILES_NPZ.relative_to(REPO_ROOT)),
                "sha256": sha256_file(ZINC_SMILES_NPZ),
                "expected": ZINC_SMILES_SHA256,
            },
            "zinc_train_pt": {
                "path": str(ZINC_TRAIN_PT.relative_to(REPO_ROOT)),
                "sha256": sha256_file(ZINC_TRAIN_PT),
                "expected": ZINC_TRAIN_PT_SHA256,
            },
            "lipo_objects_train": {
                "path": str(LIPO_OBJECTS_TRAIN.relative_to(REPO_ROOT)),
                "sha256": sha256_file(LIPO_OBJECTS_TRAIN),
                "expected": LIPO_OBJECTS_TRAIN_SHA256,
            },
        },
        "excluded_zinc_rows": list(EXCLUDED_ZINC_ROWS),
        "labels_loaded": False,
        "official_valid_or_test_loaded": False,
        "pyg_atom_dict_present": ZINC_ATOM_DICT.exists(),
    }
    data_ok = all(v["sha256"] == v["expected"] for v in integrity["frozen_inputs"].values())
    integrity["hash_check_passed"] = data_ok
    if MMP_MANIFEST.exists():
        with MMP_MANIFEST.open() as fh:
            mmp = json.load(fh)
        integrity["mmp_manifest_cross_check"] = {
            "excluded_rows": mmp.get("excluded_rows"),
            "labels_read": mmp.get("labels_read"),
            "pair_table_sha256": mmp.get("pair_table_sha256"),
        }
    if args.skip_spotcheck or not data_ok:
        integrity["pairing_verification"] = {"skipped": True}
    else:
        integrity["pairing_verification"] = pairing_verification()

    print("== ZINC-12k official train (frozen exclusions applied) ==", flush=True)
    zinc_rows, zinc_smiles = load_zinc_smiles()
    zinc = extract_dataset("zinc", zinc_rows, zinc_smiles, (PRIMARY_TIER, *AUX_TIERS), args.limit)
    print("== Lipophilicity frozen train (label-free objects) ==", flush=True)
    lipo_rows, lipo_smiles = load_lipo_train_smiles()
    lipo = extract_dataset("lipo", lipo_rows, lipo_smiles, (PRIMARY_TIER,), args.limit)

    print("== aggregation ==", flush=True)
    datasets: dict[str, dict] = {}
    for tier in (PRIMARY_TIER, *AUX_TIERS):
        for mode in CT_MODES:
            name = f"zinc__{tier}__{mode}"
            datasets[name] = {
                "meta": dict(zinc["meta"], tier=tier, ct_mode=mode),
                "agg": aggregate(zinc["insts"][tier], zinc["mols"], tier, mode),
            }
    for mode in CT_MODES:
        datasets[f"lipo__{PRIMARY_TIER}__{mode}"] = {
            "meta": dict(lipo["meta"], tier=PRIMARY_TIER, ct_mode=mode),
            "agg": aggregate(lipo["insts"][PRIMARY_TIER], lipo["mols"], PRIMARY_TIER, mode),
        }

    dataset_summary = {}
    for name, res in datasets.items():
        agg = res["agg"]
        dataset_summary[name] = {
            "meta": res["meta"],
            "support": support_curves(agg),
            "triviality": triviality_tables(agg),
            "chem_composition": chem_composition(agg),
            "decision_stats": decision_stats(agg),
        }

    primary = datasets["zinc__T1__primary"]
    examples = build_examples(primary["agg"], zinc["insts"][PRIMARY_TIER], zinc["mols"], zinc_smiles)
    cross = cross_dataset(primary["agg"], datasets["lipo__T1__primary"]["agg"])
    decision = decide(
        {
            "zinc_train10k": dataset_summary["zinc__T1__primary"]["decision_stats"],
            "lipo_train": dataset_summary["lipo__T1__primary"]["decision_stats"],
        },
        data_ok,
    )
    summary = {
        "protocol": "structure-context-support-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definitions": {
            "primary_tier": PRIMARY_TIER,
            "auxiliary_tiers": {
                "T2": "coarse: atom attribute = element only, bond attribute = {SINGLE, MULTIPLE}",
                "T3": "primary minus the atom H count",
            },
            "structure_identity": (
                "exact canonical form of the induced subgraph on N[u] u N[v] of a real bond (u,v); "
                "anchors unordered; atom attr = (element, formal charge, total H, aromatic); "
                "bond attr = single/double/triple/aromatic"
            ),
            "context_primary": (
                "exact canonical form of the induced radius-2 neighbourhood N[N[u] u N[v]] with roles "
                "(anchor / patch / outer) and outer-node external degree + ring axis + hetero-distance "
                "axis + ring-distance axis"
            ),
            "context_coarse": (
                "multiset of outer-shell records (patch atom attr < outer atom attr | bond | external "
                "degree) + ring/hetero/ring-distance axes"
            ),
            "independence_unit": "distinct molecule (SMILES) and distinct Murcko scaffold",
            "triviality_stratum": "(n_heavy, heavy-atom formula, n_rings)",
            "labels_loaded": False,
        },
        "data_integrity": integrity,
        "datasets": dataset_summary,
        "cross_dataset": cross,
        "examples": examples,
        "decision": decision,
        "runtime_seconds": time.time() - started,
    }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with (analysis_dir / "data_integrity.json").open("w") as fh:
        json.dump(integrity, fh, indent=2)
    with (analysis_dir / "examples.json").open("w") as fh:
        json.dump(examples, fh, indent=2, ensure_ascii=False)
    write_type_tables(analysis_dir, datasets)
    if not args.no_instances and args.limit is None:
        z: dict[str, np.ndarray] = {
            "meta": np.array([json.dumps({"zinc": zinc["meta"], "lipo": lipo["meta"]})], dtype=object)
        }
        for tier in (PRIMARY_TIER, *AUX_TIERS):
            pack_dataset(f"zinc_{tier}", zinc["mols"], zinc["insts"][tier], z)
        pack_dataset("lipo", lipo["mols"], lipo["insts"][PRIMARY_TIER], z)
        np.savez_compressed(analysis_dir / "instances.npz", **z)
    print(json.dumps({"decision": decision, "runtime_s": summary["runtime_seconds"]}, indent=2))
    print("rss_mb", round(rss_mb(), 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
