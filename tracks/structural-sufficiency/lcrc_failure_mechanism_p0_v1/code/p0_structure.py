"""Label-free structural support: rooted radius-2 signatures and descriptors.

Signature definition
--------------------
For a heavy atom ``v`` of a molecule we take the *rooted radius-2 induced ball*:
all atoms with graph distance <= 2 from ``v``, every bond whose two endpoints are
inside the ball, and one "boundary stub" for every bond that leaves the ball at a
distance-2 atom.  Each atom carries the real label

    (atomic number, formal charge, total H count, aromatic flag, in-ring flag)

and each stub also carries the label of the external atom it represents, so the
degree and identity of the second shell are not lost.  The root is marked with a
separate flag.  Bonds carry their bond order (single/double/triple/aromatic).

The labelled rooted ball is serialised through RDKit's canonical atom ranking on
a graph whose nodes are dummy atoms carrying the label in the isotope field.  The
result is a deterministic string, invariant to atom renumbering, and it is a
complete encoding of the labelled rooted ball (not a ``path_code``-style coarse
identity).

Training support
----------------
``df(s)`` is the **training-molecule document frequency** of signature ``s``:
each training molecule contributes at most one count for a signature no matter
how many of its atoms carry it.  Query molecules only read that dictionary.
Training-set self-diagnostics use leave-one-molecule-out frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from p0_common import (
    MORGAN_BITS,
    MORGAN_RADIUS,
    RARE_DF_MAX,
    SIGNATURE_RADIUS,
    sha256_text,
)

RDLogger.DisableLog("rdApp.*")

SIGNATURE_VERSION = "rooted-radius2-labelled-ball-v1"

#: stable integer code offsets used inside the canonicalisation graph
_BALL_OFFSET = 10_000
_ROOT_OFFSET = 1_000_000
_STUB_OFFSET = 2_000_000

_BOND_CODES = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
    Chem.BondType.DATIVE: 5,
}

SUPPORT_COLUMNS = (
    "unseen_atom_frac",
    "rare_atom_frac",
    "median_log1p_df",
    "p10_log1p_df",
)

DESCRIPTOR_COLUMNS = (
    "heavy_atoms",
    "tpsa",
    "mol_logp",
    "hbd",
    "hba",
    "formal_charge",
    "ring_count",
)


# ---------------------------------------------------------------------------
# molecule records
# ---------------------------------------------------------------------------


@dataclass
class MoleculeRecord:
    row_id: int
    smiles: str
    parse_ok: bool
    parse_error: str | None
    atom_signatures: list[str]
    n_atoms: int
    n_explicit_h: int
    heavy_atoms: int
    tpsa: float
    mol_logp: float
    hbd: int
    hba: int
    formal_charge: int
    ring_count: int
    n_components: int
    multi_component: bool
    scaffold: str
    scaffold_empty: bool
    scaffold_fallback_used: bool

    def descriptor_dict(self) -> dict[str, Any]:
        return {
            "heavy_atoms": self.heavy_atoms,
            "tpsa": self.tpsa,
            "mol_logp": self.mol_logp,
            "hbd": self.hbd,
            "hba": self.hba,
            "formal_charge": self.formal_charge,
            "ring_count": self.ring_count,
            "n_components": self.n_components,
            "multi_component": self.multi_component,
            "n_explicit_h": self.n_explicit_h,
        }


# ---------------------------------------------------------------------------
# rooted-ball construction
# ---------------------------------------------------------------------------


def _atom_label_code(mol: Chem.Mol, idx: int, ring_info) -> int:
    atom = mol.GetAtomWithIdx(int(idx))
    z = int(atom.GetAtomicNum())
    charge = int(atom.GetFormalCharge())
    hs = int(atom.GetTotalNumHs(includeNeighbors=False))
    aromatic = int(bool(atom.GetIsAromatic()))
    in_ring = int(ring_info.NumAtomRings(int(idx)) > 0)
    if not (0 <= z <= 118):
        raise ValueError(f"atomic number out of range: {z}")
    if not (-8 <= charge <= 8):
        raise ValueError(f"formal charge out of range: {charge}")
    if not (0 <= hs <= 7):
        raise ValueError(f"H count out of range: {hs}")
    return (((z * 32 + (charge + 8)) * 8 + hs) * 2 + aromatic) * 2 + in_ring


def _bond_label_code(bond: Chem.Bond) -> int:
    return int(_BOND_CODES.get(bond.GetBondType(), 9))


def rooted_ball_graph(mol: Chem.Mol, root: int, dist: np.ndarray, ring_info) -> dict[str, Any]:
    """Explicit labelled rooted ball (root at local node 0).

    ``node_labels`` entries are ``(kind, atom_code)`` with kind ``root``/``atom``/
    ``stub``.  ``edges`` are ``(i, j, bond_code)`` with ``i < j``.
    """
    n = mol.GetNumAtoms()
    mask = np.asarray(dist[int(root)] <= SIGNATURE_RADIUS + 1e-9, dtype=bool)
    ball = [int(root)] + [i for i in range(n) if mask[i] and i != int(root)]
    position = {a: k for k, a in enumerate(ball)}
    node_labels: list[tuple[str, int]] = []
    for k, atom_idx in enumerate(ball):
        node_labels.append(("root" if k == 0 else "atom", _atom_label_code(mol, atom_idx, ring_info)))
    edges: list[tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        if mask[i] and mask[j]:
            a, b = position[i], position[j]
            edges.append((min(a, b), max(a, b), _bond_label_code(bond)))
    for atom_idx in ball:
        if int(dist[int(root)][atom_idx]) < SIGNATURE_RADIUS:
            continue
        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            other = int(bond.GetOtherAtomIdx(atom_idx))
            if mask[other]:
                continue
            node_labels.append(("stub", _atom_label_code(mol, other, ring_info)))
            stub_index = len(node_labels) - 1
            a, b = position[atom_idx], stub_index
            edges.append((min(a, b), max(a, b), _bond_label_code(bond)))
    edges.sort()
    return {"node_labels": node_labels, "edges": edges, "n_nodes": len(node_labels)}


def canonicalise_ball(graph: Mapping[str, Any]) -> str:
    """Deterministic string for a labelled rooted ball.

    The canonical node order comes from RDKit's canonical atom ranking on a
    dummy-atom graph whose isotope encodes the node label.  The serialisation
    then lists node labels in canonical order and all edges as
    ``(rank_i, rank_j, bond_code)`` triples.
    """
    node_labels = list(graph["node_labels"])
    edges = list(graph["edges"])
    rw = Chem.RWMol()
    for kind, code in node_labels:
        atom = Chem.Atom(0)
        offset = {"root": _ROOT_OFFSET, "atom": _BALL_OFFSET, "stub": _STUB_OFFSET}[kind]
        atom.SetIsotope(int(offset + code))
        rw.AddAtom(atom)
    for i, j, bond_code in edges:
        bond_type = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
            5: Chem.BondType.DATIVE,
        }.get(int(bond_code), Chem.BondType.SINGLE)
        rw.AddBond(int(i), int(j), bond_type)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=False))
    order = sorted(range(len(node_labels)), key=lambda i: ranks[i])
    remap = {old: new for new, old in enumerate(order)}
    node_text = ",".join(f"{node_labels[i][0]}:{node_labels[i][1]}" for i in order)
    edge_text = ";".join(
        sorted(
            f"{min(remap[i], remap[j])}-{max(remap[i], remap[j])}:{bond_code}"
            for i, j, bond_code in edges
        )
    )
    return f"V{SIGNATURE_VERSION}|A[{node_text}]E[{edge_text}]"


def molecule_signatures(mol: Chem.Mol) -> tuple[list[str], list[dict[str, Any]]]:
    """All rooted atom signatures and their explicit ball graphs."""
    dist = Chem.GetDistanceMatrix(mol)
    ring_info = mol.GetRingInfo()
    ring_info.AtomRings()
    signatures: list[str] = []
    graphs: list[dict[str, Any]] = []
    for atom_idx in range(mol.GetNumAtoms()):
        graph = rooted_ball_graph(mol, atom_idx, dist, ring_info)
        graphs.append(graph)
        signatures.append(canonicalise_ball(graph))
    return signatures, graphs


# ---------------------------------------------------------------------------
# descriptors / scaffold
# ---------------------------------------------------------------------------


def _safe_scaffold(mol: Chem.Mol) -> tuple[str, bool, bool]:
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
    empty = scaffold == ""
    if empty:
        fallback = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return fallback, True, True
    return scaffold, False, False


def build_molecule_table(smiles: Sequence[str], *, verbose: bool = False) -> list[MoleculeRecord]:
    """Compute label-free atom signatures and descriptors once per molecule.

    Invalid SMILES are **not dropped**: they produce ``parse_ok=False`` and NaN
    descriptors, and the caller must report them.
    """
    records: list[MoleculeRecord] = []
    for row_id, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            records.append(
                MoleculeRecord(
                    row_id=row_id,
                    smiles=str(smi),
                    parse_ok=False,
                    parse_error="rdkit_parse_failed",
                    atom_signatures=[],
                    n_atoms=0,
                    n_explicit_h=0,
                    heavy_atoms=0,
                    tpsa=float("nan"),
                    mol_logp=float("nan"),
                    hbd=0,
                    hba=0,
                    formal_charge=0,
                    ring_count=0,
                    n_components=0,
                    multi_component=False,
                    scaffold="",
                    scaffold_empty=True,
                    scaffold_fallback_used=True,
                )
            )
            continue
        signatures, _ = molecule_signatures(mol)
        scaffold, scaffold_empty, fallback = _safe_scaffold(mol)
        try:
            mol_logp = float(Descriptors.MolLogP(mol))
        except Exception:
            mol_logp = float("nan")
        try:
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        except Exception:
            tpsa = float("nan")
        n_components = int(len(Chem.GetMolFrags(mol)))
        n_explicit_h = int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1))
        records.append(
            MoleculeRecord(
                row_id=row_id,
                smiles=str(smi),
                parse_ok=True,
                parse_error=None,
                atom_signatures=signatures,
                n_atoms=int(mol.GetNumAtoms()),
                n_explicit_h=n_explicit_h,
                heavy_atoms=int(mol.GetNumHeavyAtoms()),
                tpsa=tpsa,
                mol_logp=mol_logp,
                hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
                hba=int(rdMolDescriptors.CalcNumHBA(mol)),
                formal_charge=int(Chem.GetFormalCharge(mol)),
                ring_count=int(rdMolDescriptors.CalcNumRings(mol)),
                n_components=n_components,
                multi_component=bool(n_components > 1),
                scaffold=scaffold,
                scaffold_empty=scaffold_empty,
                scaffold_fallback_used=fallback,
            )
        )
        if verbose and (row_id + 1) % 1000 == 0:
            print(f"  molecule table: {row_id + 1}/{len(smiles)}", flush=True)
    return records


def fingerprints(records: Sequence[MoleculeRecord]) -> list[Any]:
    """Morgan(radius=2, 2048 bits) fingerprints, ``None`` for parse failures."""
    out: list[Any] = []
    for rec in records:
        if not rec.parse_ok:
            out.append(None)
            continue
        mol = Chem.MolFromSmiles(rec.smiles)
        out.append(AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS))
    return out


# ---------------------------------------------------------------------------
# support features
# ---------------------------------------------------------------------------


def document_frequency(atom_signature_lists: Iterable[Sequence[str]]) -> dict[str, int]:
    """Training-molecule document frequency: one count per molecule per signature."""
    df: dict[str, int] = {}
    for signatures in atom_signature_lists:
        for signature in set(signatures):
            df[signature] = df.get(signature, 0) + 1
    return df


def support_row(atom_signatures: Sequence[str], df: Mapping[str, int]) -> dict[str, Any]:
    """Support features for one query molecule against a training dictionary."""
    n = len(atom_signatures)
    if n == 0:
        return {
            "n_atoms": 0,
            "n_unseen_atoms": 0,
            "n_rare_atoms": 0,
            "n_atoms_df_1_4": 0,
            "n_atoms_df_ge_5": 0,
            "mean_log1p_df": float("nan"),
            "unseen_atom_frac": float("nan"),
            "rare_atom_frac": float("nan"),
            "median_log1p_df": float("nan"),
            "p10_log1p_df": float("nan"),
        }
    counts = np.asarray([int(df.get(s, 0)) for s in atom_signatures], dtype=np.int64)
    log_counts = np.log1p(counts.astype(np.float64))
    return {
        "n_atoms": int(n),
        "n_unseen_atoms": int(np.sum(counts == 0)),
        "n_rare_atoms": int(np.sum((counts >= 1) & (counts <= RARE_DF_MAX))),
        "n_atoms_df_1_4": int(np.sum((counts >= 1) & (counts <= RARE_DF_MAX))),
        "n_atoms_df_ge_5": int(np.sum(counts >= RARE_DF_MAX + 1)),
        "mean_log1p_df": float(np.mean(log_counts)),
        "unseen_atom_frac": float(np.mean(counts == 0)),
        "rare_atom_frac": float(np.mean((counts >= 1) & (counts <= RARE_DF_MAX))),
        "median_log1p_df": float(np.median(log_counts)),
        "p10_log1p_df": float(np.percentile(log_counts, 10.0)),
    }


def leave_one_out_support_row(atom_signatures: Sequence[str], df: Mapping[str, int]) -> dict[str, Any]:
    """Same features with the molecule itself removed from the dictionary."""
    counts = np.asarray([int(df.get(s, 0)) - 1 for s in atom_signatures], dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("leave-one-out df went negative; dictionary does not contain this molecule")
    n = len(atom_signatures)
    if n == 0:
        return support_row([], {})
    log_counts = np.log1p(counts.astype(np.float64))
    return {
        "n_atoms": int(n),
        "n_unseen_atoms": int(np.sum(counts == 0)),
        "n_rare_atoms": int(np.sum((counts >= 1) & (counts <= RARE_DF_MAX))),
        "n_atoms_df_1_4": int(np.sum((counts >= 1) & (counts <= RARE_DF_MAX))),
        "n_atoms_df_ge_5": int(np.sum(counts >= RARE_DF_MAX + 1)),
        "mean_log1p_df": float(np.mean(log_counts)),
        "unseen_atom_frac": float(np.mean(counts == 0)),
        "rare_atom_frac": float(np.mean((counts >= 1) & (counts <= RARE_DF_MAX))),
        "median_log1p_df": float(np.median(log_counts)),
        "p10_log1p_df": float(np.percentile(log_counts, 10.0)),
    }


def max_train_tanimoto(query_fps: Sequence[Any], train_fps: Sequence[Any]) -> np.ndarray:
    out = np.full(len(query_fps), np.nan, dtype=np.float64)
    valid_train = [fp for fp in train_fps if fp is not None]
    for i, fp in enumerate(query_fps):
        if fp is None or not valid_train:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, valid_train)
        out[i] = float(max(sims)) if sims else float("nan")
    return out


def max_train_tanimoto_loo(train_fps: Sequence[Any]) -> np.ndarray:
    """Leave-one-out maximum Tanimoto for every training molecule."""
    out = np.full(len(train_fps), np.nan, dtype=np.float64)
    for i, fp in enumerate(train_fps):
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        sims = list(sims)
        if i < len(sims):
            sims[i] = -1.0
        finite = [s for s in sims if s is not None and np.isfinite(s)]
        out[i] = float(max(finite)) if finite else float("nan")
    return out


@dataclass
class SupportData:
    group_key: str
    train_rows: np.ndarray
    df: dict[str, int]
    query: pd.DataFrame
    train_loo: pd.DataFrame
    summary: dict[str, Any]
    molecules: list[MoleculeRecord] = field(repr=False, default_factory=list)


def _descriptor_frame(records: Sequence[MoleculeRecord], rows: Sequence[int]) -> pd.DataFrame:
    out = []
    for row in rows:
        rec = records[int(row)]
        entry = {
            "row_id": int(row),
            "smiles": rec.smiles,
            "parse_ok": rec.parse_ok,
            "parse_error": rec.parse_error,
            "n_atoms_total": rec.n_atoms,
            "scaffold": rec.scaffold,
            "scaffold_empty": rec.scaffold_empty,
            "scaffold_fallback_used": rec.scaffold_fallback_used,
        }
        entry.update(rec.descriptor_dict())
        out.append(entry)
    return pd.DataFrame(out)


def build_support(
    *,
    group_key: str,
    train_rows: Sequence[int],
    query_splits: Mapping[str, Sequence[int]],
    molecules: Sequence[MoleculeRecord],
    include_loo: bool = True,
) -> SupportData:
    """Build the Train-only support dictionary and per-molecule diagnostics."""
    train_rows = np.asarray(train_rows, dtype=np.int64)
    train_records = [molecules[int(r)] for r in train_rows]
    df = document_frequency(rec.atom_signatures for rec in train_records)
    train_fps = fingerprints(train_records)
    train_scaffold_counts: dict[str, int] = {}
    for rec in train_records:
        train_scaffold_counts[rec.scaffold] = train_scaffold_counts.get(rec.scaffold, 0) + 1

    query_frames: list[pd.DataFrame] = []
    for split, rows in query_splits.items():
        rows = np.asarray(rows, dtype=np.int64)
        records = [molecules[int(r)] for r in rows]
        frame = _descriptor_frame(molecules, rows)
        support = [support_row(rec.atom_signatures, df) for rec in records]
        support_frame = pd.DataFrame(support)
        tanimoto = max_train_tanimoto(fingerprints(records), train_fps)
        frame = pd.concat([frame.reset_index(drop=True), support_frame], axis=1)
        frame["max_train_tanimoto"] = tanimoto
        frame["scaffold_seen"] = [bool(train_scaffold_counts.get(rec.scaffold, 0) > 0) for rec in records]
        frame["support_mode"] = "query_train_dictionary"
        frame["group"] = group_key
        frame["split"] = split
        frame["exposure"] = (
            "POST_HOC_EXPOSED_DISCOVERY" if split == "test" else "FROZEN_VALIDATION_ONLY"
        )
        query_frames.append(frame)

    loo_frame = pd.DataFrame()
    if include_loo:
        loo = [leave_one_out_support_row(rec.atom_signatures, df) for rec in train_records]
        loo_frame = _descriptor_frame(molecules, train_rows)
        loo_frame = pd.concat([loo_frame.reset_index(drop=True), pd.DataFrame(loo)], axis=1)
        loo_frame["max_train_tanimoto"] = max_train_tanimoto_loo(train_fps)
        loo_frame["scaffold_seen"] = [
            bool(train_scaffold_counts.get(rec.scaffold, 0) > 1) for rec in train_records
        ]
        loo_frame["support_mode"] = "leave_one_molecule_out"
        loo_frame["group"] = group_key
        loo_frame["split"] = "train_loo"
        loo_frame["exposure"] = "TRAIN_ONLY_SELF_DIAGNOSTIC"

    query = pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame()
    summary = {
        "group": group_key,
        "signature_version": SIGNATURE_VERSION,
        "signature_radius": SIGNATURE_RADIUS,
        "n_train_molecules": int(len(train_rows)),
        "n_train_atom_tokens": int(sum(len(rec.atom_signatures) for rec in train_records)),
        "n_signatures": int(len(df)),
        "df_histogram": {
            "df_1": int(sum(1 for v in df.values() if v == 1)),
            "df_2_4": int(sum(1 for v in df.values() if 2 <= v <= 4)),
            "df_5_9": int(sum(1 for v in df.values() if 5 <= v <= 9)),
            "df_10plus": int(sum(1 for v in df.values() if v >= 10)),
            "max_df": int(max(df.values()) if df else 0),
        },
        "signature_dictionary_sha256": sha256_text(sorted(df.keys())),
        "train_only": True,
        "valid_or_test_used_to_build_dictionary": False,
        "query_rows": {
            split: int(len(rows)) for split, rows in query_splits.items()
        },
        "molecule_handling": {
            "n_parse_failures": int(sum(1 for rec in molecules if not rec.parse_ok)),
            "n_multi_component": int(sum(1 for rec in molecules if rec.multi_component)),
            "n_charged": int(sum(1 for rec in molecules if rec.formal_charge != 0)),
            "n_empty_scaffold_fallback": int(sum(1 for rec in molecules if rec.scaffold_fallback_used)),
            "n_explicit_h_atoms": int(sum(rec.n_explicit_h for rec in molecules)),
            "invalid_smiles_policy": (
                "recorded with parse_ok=False and NaN descriptors; never dropped silently, "
                "and never included in a pairwise prediction comparison"
            ),
            "multi_component_policy": (
                "kept as one disconnected molecular graph (no salt stripping, no fragment selection); "
                "rooted balls stay inside one component; descriptors sum over the whole record"
            ),
            "empty_scaffold_policy": (
                "acyclic molecules have an empty Bemis-Murcko scaffold; the canonical isomeric-free "
                "SMILES is used as the scaffold key and scaffold_fallback_used=True is recorded"
            ),
        },
        "query_summary": {},
        "train_loo_summary": {},
    }
    for split, sub in (query.groupby("split", sort=False) if len(query) else []):
        summary["query_summary"][str(split)] = {
            "n": int(len(sub)),
            "mean_unseen_atom_frac": float(sub["unseen_atom_frac"].mean()),
            "median_unseen_atom_frac": float(sub["unseen_atom_frac"].median()),
            "mean_rare_atom_frac": float(sub["rare_atom_frac"].mean()),
            "frac_molecules_with_any_unseen": float(np.mean(sub["unseen_atom_frac"] > 0)),
            "mean_median_log1p_df": float(sub["median_log1p_df"].mean()),
        }
    if len(loo_frame):
        summary["train_loo_summary"] = {
            "n": int(len(loo_frame)),
            "mean_unseen_atom_frac": float(loo_frame["unseen_atom_frac"].mean()),
            "mean_rare_atom_frac": float(loo_frame["rare_atom_frac"].mean()),
            "frac_molecules_with_any_unseen": float(np.mean(loo_frame["unseen_atom_frac"] > 0)),
        }
    return SupportData(
        group_key=group_key,
        train_rows=train_rows,
        df=df,
        query=query,
        train_loo=loo_frame,
        summary=summary,
        molecules=list(molecules),
    )


# ---------------------------------------------------------------------------
# signature correctness audits
# ---------------------------------------------------------------------------


def _to_networkx(graph: Mapping[str, Any]):
    import networkx as nx

    g = nx.Graph()
    for i, (kind, code) in enumerate(graph["node_labels"]):
        g.add_node(i, label=f"{kind}:{code}")
    for i, j, bond_code in graph["edges"]:
        g.add_edge(int(i), int(j), bond=int(bond_code))
    return g


def balls_isomorphic(graph_a: Mapping[str, Any], graph_b: Mapping[str, Any]) -> bool:
    import networkx as nx

    if graph_a["n_nodes"] != graph_b["n_nodes"] or len(graph_a["edges"]) != len(graph_b["edges"]):
        return False
    return bool(
        nx.is_isomorphic(
            _to_networkx(graph_a),
            _to_networkx(graph_b),
            node_match=lambda a, b: a["label"] == b["label"],
            edge_match=lambda a, b: a["bond"] == b["bond"],
        )
    )


def signature_collision_audit(
    records: Sequence[MoleculeRecord],
    *,
    max_molecules: int = 60,
    max_pairs: int = 1500,
    seed: int = 0,
) -> dict[str, Any]:
    """Check that equal signatures are isomorphic and different ones are not.

    A deterministic sample of molecules is re-parsed and its rooted balls are
    compared with an independent exact VF2 isomorphism test (networkx), which
    does not share the canonicalisation code.
    """
    rng = np.random.default_rng(int(seed))
    usable = [rec for rec in records if rec.parse_ok and rec.atom_signatures]
    if not usable:
        return {"checked": False, "reason": "no parsed molecules"}
    take = min(int(max_molecules), len(usable))
    indices = np.sort(rng.choice(len(usable), size=take, replace=False))
    balls: list[dict[str, Any]] = []
    signatures: list[str] = []
    for idx in indices:
        rec = usable[int(idx)]
        mol = Chem.MolFromSmiles(rec.smiles)
        dist = Chem.GetDistanceMatrix(mol)
        ring_info = mol.GetRingInfo()
        ring_info.AtomRings()
        for atom_idx in range(mol.GetNumAtoms()):
            graph = rooted_ball_graph(mol, atom_idx, dist, ring_info)
            balls.append(graph)
            signatures.append(canonicalise_ball(graph))

    same_pairs = 0
    same_failures = 0
    sig_to_indices: dict[str, list[int]] = {}
    for i, sig in enumerate(signatures):
        sig_to_indices.setdefault(sig, []).append(i)
    for sig, members in sig_to_indices.items():
        if len(members) < 2:
            continue
        for _ in range(min(len(members) * (len(members) - 1) // 2, 40)):
            a, b = rng.choice(len(members), size=2, replace=False)
            i, j = members[int(a)], members[int(b)]
            same_pairs += 1
            if not balls_isomorphic(balls[i], balls[j]):
                same_failures += 1
            if same_pairs >= max_pairs:
                break
        if same_pairs >= max_pairs:
            break

    # Different-signature pairs with the same node count and same root label are
    # the only place a canonicalisation bug could hide; check a deterministic
    # sample using the exact VF2 test.
    buckets: dict[tuple[int, str], list[int]] = {}
    for i, graph in enumerate(balls):
        buckets.setdefault((graph["n_nodes"], graph["node_labels"][0][1]), []).append(i)
    diff_pairs = 0
    diff_failures = 0
    bucket_lists = [members for members in buckets.values() if len(members) >= 2]
    if bucket_lists:
        while diff_pairs < max_pairs:
            bucket = bucket_lists[int(rng.integers(0, len(bucket_lists)))]
            i, j = rng.choice(len(bucket), size=2, replace=False)
            i, j = bucket[int(i)], bucket[int(j)]
            if signatures[i] == signatures[j]:
                continue
            diff_pairs += 1
            if balls_isomorphic(balls[i], balls[j]):
                diff_failures += 1
            if diff_pairs >= max_pairs and diff_failures > 0:
                break
    return {
        "checked": True,
        "n_molecules_sampled": int(take),
        "n_balls": int(len(balls)),
        "n_distinct_signatures": int(len(set(signatures))),
        "same_signature_pairs_checked": int(same_pairs),
        "same_signature_isomorphism_failures": int(same_failures),
        "different_signature_pairs_checked": int(diff_pairs),
        "different_signature_isomorphic_pairs": int(diff_failures),
        "passed": bool(same_failures == 0 and diff_failures == 0),
        "isomorphism_test": "networkx VF2 with node label and bond-type matching",
    }


def permutation_invariance_audit(
    records: Sequence[MoleculeRecord], *, n_molecules: int = 25, seed: int = 0
) -> dict[str, Any]:
    """Check signatures under random atom renumbering of real molecules."""
    rng = np.random.default_rng(int(seed))
    usable = [rec for rec in records if rec.parse_ok and rec.atom_signatures]
    take = min(int(n_molecules), len(usable))
    indices = np.sort(rng.choice(len(usable), size=take, replace=False))
    mismatches = 0
    atoms_checked = 0
    for idx in indices:
        rec = usable[int(idx)]
        mol = Chem.MolFromSmiles(rec.smiles)
        base_sigs = rec.atom_signatures
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        permuted = Chem.RenumberAtoms(mol, [int(x) for x in perm])
        new_sigs, _ = molecule_signatures(permuted)
        for new_idx in range(permuted.GetNumAtoms()):
            old_idx = int(perm[new_idx])
            atoms_checked += 1
            if new_sigs[new_idx] != base_sigs[old_idx]:
                mismatches += 1
    return {
        "n_molecules_checked": int(take),
        "n_atoms_checked": int(atoms_checked),
        "renumbering_mismatches": int(mismatches),
        "passed": bool(mismatches == 0),
    }
