"""Label-free MolHIV structural features: rooted WL2 support, Morgan similarity, scaffolds.

Only SMILES from the frozen OGB mapping file are read.  No label column is
selected, and ``raw/graph-label.csv.gz`` is never opened here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import ph_common as C

RDLogger.DisableLog("rdApp.*")


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_atom(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "Z": int(atom.GetAtomicNum()),
        "charge": int(atom.GetFormalCharge()),
        "arom": bool(atom.GetIsAromatic()),
        "nH": int(atom.GetTotalNumHs()),
        "degree": int(atom.GetTotalDegree()),
        "ring": bool(atom.IsInRing()),
    }


def _json_bond(bond: Chem.Bond) -> dict[str, Any]:
    try:
        conj = bool(bond.GetIsConjugated())
    except Exception:
        conj = False
    return {
        "type": str(bond.GetBondType()),
        "arom": bool(bond.GetIsAromatic()),
        "conj": conj,
        "ring": bool(bond.IsInRing()),
    }


def parse_smiles(smiles: str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "sanitized"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES even without sanitisation")
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return mol, "unsanitized_fallback"


def wl2_root_colors(mol: Chem.Mol, rounds: int = 2) -> list[str]:
    """Root-centred radius-``rounds`` WL colours, one per atom, stable hashes."""
    atoms = list(mol.GetAtoms())
    colors = [_sha(C.stable_json(_json_atom(a))) for a in atoms]
    adj: list[list[tuple[dict[str, Any], int]]] = [[] for _ in atoms]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        desc = _json_bond(bond)
        adj[i].append((desc, j))
        adj[j].append((desc, i))
    for _ in range(rounds):
        new = []
        for i, a in enumerate(atoms):
            neigh = sorted(
                [[C.stable_json(desc), colors[j]] for desc, j in adj[i]],
                key=lambda x: (x[0], x[1]),
            )
            new.append(_sha(C.stable_json({"self": colors[i], "neigh": neigh})))
        colors = new
    return colors


def train_support(train_smiles: Sequence[str]) -> tuple[Counter, dict[str, Any]]:
    support: Counter = Counter()
    parse_modes = Counter()
    n_atoms = 0
    for smi in train_smiles:
        mol, mode = parse_smiles(smi)
        parse_modes[mode] += 1
        colors = wl2_root_colors(mol)
        n_atoms += len(colors)
        for fp in set(colors):
            support[fp] += 1
    return support, {
        "n_train_molecules": len(train_smiles),
        "n_train_atoms": int(n_atoms),
        "parse_modes": dict(parse_modes),
        "n_distinct_wl2": len(support),
    }


def test_features(
    test_smiles: Sequence[str],
    train_smiles: Sequence[str],
    support: Counter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_canon = set()
    train_scaffolds = set()
    train_modes = Counter()
    for smi in train_smiles:
        mol, mode = parse_smiles(smi)
        train_modes[mode] += 1
        train_canon.add(Chem.MolToSmiles(mol, canonical=True))
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        if scaf:
            train_scaffolds.add(scaf)

    train_fps = []
    for smi in train_smiles:
        mol, _ = parse_smiles(smi)
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    rows: list[dict[str, Any]] = []
    test_modes = Counter()
    test_canon = set()
    test_scaffolds = set()
    n_parse_fail = 0
    for idx, smi in enumerate(test_smiles):
        mol, mode = parse_smiles(smi)
        test_modes[mode] += 1
        colors = wl2_root_colors(mol)
        supports = np.asarray([support.get(c, 0) for c in colors], dtype=np.int64)
        n_atoms = int(len(colors))
        n_bonds = int(mol.GetNumBonds())
        n_heavy = int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1))
        unseen = float((supports == 0).mean()) if n_atoms else float("nan")
        low = float((supports <= 4).mean()) if n_atoms else float("nan")
        med = float(np.median(supports)) if n_atoms else float("nan")
        mn = int(supports.min()) if n_atoms else -1
        canon = Chem.MolToSmiles(mol, canonical=True)
        test_canon.add(canon)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        if scaf:
            test_scaffolds.add(scaf)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            max_sim = float(max(sims)) if sims else float("nan")
        except Exception:
            n_parse_fail += 1
            max_sim = float("nan")
        rows.append(
            {
                "test_index": int(idx),
                "n_heavy_atoms": n_heavy,
                "n_bonds": n_bonds,
                "n_distinct_wl2": int(len(set(colors))),
                "fraction_wl2_unseen": unseen,
                "fraction_wl2_low_support": low,
                "median_wl2_train_support": med,
                "min_wl2_train_support": mn,
                "max_train_tanimoto": max_sim,
                "canonical_smiles": canon,
                "murcko_scaffold": scaf,
                "parse_mode": mode,
            }
        )

    seen_scaffolds = train_scaffolds & test_scaffolds
    meta = {
        "n_train_molecules": len(train_smiles),
        "n_test_molecules": len(test_smiles),
        "train_parse_modes": dict(train_modes),
        "test_parse_modes": dict(test_modes),
        "n_morgan_failures": n_parse_fail,
        "n_train_distinct_canonical_smiles": len(train_canon),
        "n_test_distinct_canonical_smiles": len(test_canon),
        "n_test_canonical_seen_in_train": int(len(train_canon & test_canon)),
        "n_test_scaffolds": int(len(test_scaffolds)),
        "n_train_scaffolds": int(len(train_scaffolds)),
        "n_test_scaffolds_seen_in_train": int(len(seen_scaffolds)),
        "test_scaffolds_seen": sorted(seen_scaffolds),
        "seen_scaffold_group_available": bool(len(seen_scaffolds) > 0),
        "rdkit_version": __import__("rdkit").__version__,
        "smiles_source": C.rel(C.OGB_SOURCE / "mapping/mol.csv.gz"),
        "smiles_sha256": C.sha256_file(C.OGB_SOURCE / "mapping/mol.csv.gz"),
        "labels_read": False,
    }
    return rows, meta


def load_smiles_and_splits() -> tuple[list[str], dict[str, list[int]], dict[str, Any]]:
    mapping_path = C.OGB_SOURCE / "mapping/mol.csv.gz"
    frame = pd.read_csv(mapping_path, usecols=["smiles"])
    smiles = [str(x) for x in frame["smiles"].tolist()]
    splits: dict[str, list[int]] = {}
    files = {}
    for name in ("train", "valid", "test"):
        path = C.OGB_SOURCE / f"split/scaffold/{name}.csv.gz"
        arr = pd.read_csv(path, header=None).to_numpy().reshape(-1)
        splits[name] = [int(x) for x in arr]
        files[name] = C.file_record(path, f"OGB official scaffold {name} row ids")
    files["mapping"] = C.file_record(mapping_path, "OGB mapping; only the smiles column is read")
    return smiles, splits, files


def build_all() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    smiles, splits, files = load_smiles_and_splits()
    train_smi = [smiles[i] for i in splits["train"]]
    test_smi = [smiles[i] for i in splits["test"]]
    support, support_meta = train_support(train_smi)
    rows, meta = test_features(test_smi, train_smi, support)
    for i, row in enumerate(rows):
        row["ogb_row"] = int(splits["test"][i])
    meta["split_files"] = files
    meta["support"] = support_meta
    return rows, meta
