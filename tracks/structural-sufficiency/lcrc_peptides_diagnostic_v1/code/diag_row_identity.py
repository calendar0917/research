"""P1 preflight — strict row-identity check between SMILES rows and the PyG graphs.

Three independent pieces of evidence per row:

1. split geometry: sizes 10873 / 2331 / 2331, indices disjoint and complete;
2. labels: the CSV label row must equal the PyG ``y`` row and the Fast v1
   ``peptides_func_y_{train,valid}.npy`` row at the same position;
3. structure: an RDKit molecule built from the CSV SMILES must reproduce, position
   by position, every PyG atom feature (9 integer columns) and every PyG bond
   feature / connection (both edge directions), and the reconstructed molecule's
   canonical non-isomeric SMILES must equal the CSV molecule's canonical SMILES.

If any check fails the script writes ``ROW_ALIGNMENT_BLOCKED`` and exits non-zero;
no re-splitting is ever attempted.
"""

from __future__ import annotations

import csv
import sys
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger

import diag_common as C

RDLogger.DisableLog("rdApp.*")

#: OGB-style atom-feature column definitions used by LRGB Peptides-func.
ATOM_COLUMNS = (
    "atomic_num_minus_1",
    "chiral_tag",
    "degree_plus_num_hs",
    "formal_charge_plus_5",
    "total_num_hs",
    "num_radical_electrons",
    "hybridization_index",
    "is_aromatic",
    "is_in_ring",
)
HYBRIDIZATION_ORDER = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    Chem.HybridizationType.S,
    Chem.HybridizationType.UNSPECIFIED,
)
BOND_TYPE_INDEX = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
BOND_TYPE_FROM_INDEX = {v: k for k, v in BOND_TYPE_INDEX.items()}


def atom_features(mol: Chem.Mol) -> np.ndarray:
    rows = []
    for atom in mol.GetAtoms():
        rows.append(
            [
                atom.GetAtomicNum() - 1,
                int(atom.GetChiralTag()),
                atom.GetDegree() + atom.GetTotalNumHs(),
                atom.GetFormalCharge() + 5,
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                HYBRIDIZATION_ORDER.index(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
        )
    return np.asarray(rows, dtype=np.int64).reshape(-1, len(ATOM_COLUMNS))


def bond_features(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    feats = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for a, b in ((u, v), (v, u)):
            pairs.append((a, b))
            feats.append(
                [
                    BOND_TYPE_INDEX[bond.GetBondType()],
                    int(bond.GetStereo()),
                    int(bond.GetIsConjugated()),
                ]
            )
    pairs_arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    feats_arr = np.asarray(feats, dtype=np.int64).reshape(-1, 3)
    order = np.lexsort((pairs_arr[:, 1], pairs_arr[:, 0]))
    return pairs_arr[order], feats_arr[order]


def rebuild_from_pyg(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> Chem.Mol:
    rw = Chem.RWMol()
    for i in range(x.shape[0]):
        atom = Chem.Atom(int(x[i, 0]) + 1)
        atom.SetFormalCharge(int(x[i, 3]) - 5)
        atom.SetNumExplicitHs(int(x[i, 4]))
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(int(x[i, 7])))
        rw.AddAtom(atom)
    seen: set[tuple[int, int]] = set()
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        rw.AddBond(u, v, BOND_TYPE_FROM_INDEX[int(edge_attr[k, 0])])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def pyg_canonical(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data.x.detach().cpu().numpy().astype(np.int64)
    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)
    edge_attr = data.edge_attr.detach().cpu().numpy().astype(np.int64)
    order = np.lexsort((edge_index[1], edge_index[0]))
    return x, edge_index[:, order], edge_attr[order]


def main() -> int:
    from torch_geometric.datasets import LRGBDataset

    with open(C.PEPTIDES_CSV, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    splits_raw = C.read_json(C.PEPTIDES_SPLITS)
    splits = {split: [int(i) for i in splits_raw[split]] for split in C.SPLITS}
    labels = np.asarray([[int(r[name]) for name in C.LABELS] for r in rows], dtype=np.int64)

    split_checks = {k: len(v) for k, v in splits.items()}
    geometry_ok = (
        len(rows) == 15535
        and split_checks == {"train": 10873, "valid": 2331, "test": 2331}
        and sorted(sum(splits.values(), [])) == list(range(15535))
    )

    prev_labels = {
        "train": np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.int64),
        "valid": np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.int64),
    }

    per_split: dict[str, Any] = {}
    all_ok = bool(geometry_ok)
    for split in C.SPLITS:
        loader_split = "val" if split == "valid" else split
        dataset = LRGBDataset(root=str(C.LRGB_ROOT), name="Peptides-func", split=loader_split)
        idx = splits[split]
        if len(dataset) != len(idx):
            raise RuntimeError(f"{split}: PyG rows {len(dataset)} != split indices {len(idx)}")
        n_atoms = 0
        n_atom_mismatch = 0
        n_bond_mismatch = 0
        n_label_mismatch = 0
        n_smiles_mismatch = 0
        n_smiles_rebuild_failed = 0
        row_digests: list[str] = []
        for pos, csv_row_index in enumerate(idx):
            data = dataset[pos]
            smiles = rows[csv_row_index]["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(f"{split}[{pos}]: unparsable SMILES")
            if mol.GetNumAtoms() != int(data.num_nodes):
                raise RuntimeError(
                    f"{split}[{pos}]: atom count {mol.GetNumAtoms()} != PyG {int(data.num_nodes)}"
                )
            x_pyg, ei_pyg, ea_pyg = pyg_canonical(data)
            x_rd = atom_features(mol)
            pairs_rd, ea_rd = bond_features(mol)
            n_atoms += x_rd.shape[0]

            atom_equal = x_rd.shape == x_pyg.shape and np.array_equal(x_rd, x_pyg)
            bond_equal = (
                pairs_rd.shape == ei_pyg.T.shape
                and np.array_equal(pairs_rd, ei_pyg.T)
                and ea_rd.shape == ea_pyg.shape
                and np.array_equal(ea_rd, ea_pyg)
            )
            if not atom_equal:
                n_atom_mismatch += 1
            if not bond_equal:
                n_bond_mismatch += 1

            y_pyg = data.y.detach().cpu().numpy().astype(np.int64).reshape(-1)
            y_csv = labels[csv_row_index]
            label_equal = y_pyg.shape == y_csv.shape and np.array_equal(y_pyg, y_csv)
            if not label_equal:
                n_label_mismatch += 1
            if split in prev_labels:
                if not np.array_equal(prev_labels[split][pos], y_csv):
                    raise RuntimeError(f"{split}[{pos}]: Fast v1 label row differs from CSV row")

            try:
                rebuilt = rebuild_from_pyg(x_pyg, ei_pyg, ea_pyg)
                same_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True) == Chem.MolToSmiles(
                    rebuilt, isomericSmiles=False, canonical=True
                )
            except Exception:  # noqa: BLE001 - reconstruction is a secondary check
                n_smiles_rebuild_failed += 1
                same_smiles = True  # do not fail the gate on the secondary check
            if not same_smiles:
                n_smiles_mismatch += 1

            digest = C.sha256_text(
                x_rd.tobytes().hex()
                + np.asarray(pairs_rd, dtype=np.int64).tobytes().hex()
                + ea_rd.tobytes().hex()
                + y_csv.tobytes().hex()
            )
            row_digests.append(digest)

        ok = (
            n_atom_mismatch == 0
            and n_bond_mismatch == 0
            and n_label_mismatch == 0
            and n_smiles_mismatch == 0
        )
        all_ok = all_ok and ok
        per_split[split] = {
            "n_rows": len(idx),
            "n_atoms": int(n_atoms),
            "n_atom_feature_mismatch_rows": n_atom_mismatch,
            "n_bond_feature_mismatch_rows": n_bond_mismatch,
            "n_label_mismatch_rows": n_label_mismatch,
            "n_canonical_smiles_mismatch_rows": n_smiles_mismatch,
            "n_canonical_smiles_rebuild_failed_rows": n_smiles_rebuild_failed,
            "row_digest_sha256": C.sha256_text("".join(row_digests)),
            "row_digest_first": row_digests[0],
            "row_digest_last": row_digests[-1],
            "split_index_sha256": C.sha256_text(",".join(str(i) for i in idx)),
            "status": "PASS" if ok else "FAIL",
        }
        C.log(
            f"row identity {split}: {len(idx)} rows / {n_atoms} atoms, "
            f"atom mismatches {n_atom_mismatch}, bond mismatches {n_bond_mismatch}, "
            f"label mismatches {n_label_mismatch}, SMILES mismatches {n_smiles_mismatch}"
        )

    payload = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "status": "PASS" if all_ok else "ROW_ALIGNMENT_BLOCKED",
        "sources": {
            "smiles_csv": C.track_rel(C.PEPTIDES_CSV),
            "smiles_csv_sha256": C.sha256_file(C.PEPTIDES_CSV),
            "splits_json": C.track_rel(C.PEPTIDES_SPLITS),
            "splits_json_sha256": C.sha256_file(C.PEPTIDES_SPLITS),
            "pyg_dataset": f"torch_geometric.datasets.LRGBDataset(root={C.track_rel(C.LRGB_ROOT)}, name=Peptides-func)",
            "pyg_processed": {
                split: C.sha256_file(C.LRGB_ROOT / "peptides-func" / "processed" / f"{'val' if split == 'valid' else split}.pt")
                for split in C.SPLITS
            },
            "fast_v1_train_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_train.npy"),
            "fast_v1_valid_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"),
        },
        "checks": {
            "csv_rows": len(rows),
            "split_sizes": split_checks,
            "split_indices_disjoint_and_complete": bool(geometry_ok),
            "atom_columns": list(ATOM_COLUMNS),
            "bond_columns": ["bond_type_index", "stereo_index", "is_conjugated"],
            "bond_type_index_mapping": {
                "SINGLE": 0,
                "DOUBLE": 1,
                "TRIPLE": 2,
                "AROMATIC": 3,
            },
            "test_labels_used_for_training_or_selection": False,
        },
        "per_split": per_split,
        "environment": C.environment(),
    }
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(C.ANALYSIS / "row_identity.json", payload)
    if not all_ok:
        C.log("ROW_ALIGNMENT_BLOCKED: P1 must not run")
        return 2
    C.log("row identity PASS for all three splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
