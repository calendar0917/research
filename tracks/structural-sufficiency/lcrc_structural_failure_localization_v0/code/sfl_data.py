"""Data layer: official split, raw OGB records, frozen paired predictions.

Read-only.  Three independent artefacts are cross-checked before any analysis:

1. ``mapping/mol.csv.gz``   (official row order -> SMILES + experimental label)
2. ``split/scaffold/*.csv`` (official scaffold split of the same row order)
3. ``raw/node-feat.csv.gz`` + ``raw/num-node-list.csv.gz`` (the actual OGB graph
   input, i.e. what the two models consumed)

plus the frozen per-molecule predictions of the reference round
``lcrc_encoder_boundary_reference_v0/analysis/paired_errors.csv``.

The verification answers one question: does ``row_id`` in the paired-error table
denote the same molecule, in the same split, with the same label that the OGB
record for that row denotes?  Nothing is re-fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sfl_common as C

#: OGB ``possible_atomic_num_list`` = 1..118 + misc, so column value = Z - 1
OGB_ATOMIC_INDEX_OFFSET = 1
#: OGB ``possible_charge_list`` = [-5..5], so column value 5 == neutral
OGB_CHARGE_INDEX_NEUTRAL = 5


@dataclass
class Base:
    """one row per official molecule, Train+Valid only (Test never loaded)"""

    mol: pd.DataFrame          # row_id, smiles, exp, split, desc cols
    errors: pd.DataFrame       # row_id, split, y_raw, pred_A, pred_B, error_A, error_B
    checks: dict[str, Any]


def _load_split(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None)[0].to_numpy(dtype=np.int64)


def load_official_rows() -> dict[str, np.ndarray]:
    root = C.OGB_DIR / "split/scaffold"
    out = {
        "train": _load_split(root / "train.csv.gz"),
        "valid": _load_split(root / "valid.csv.gz"),
    }
    sizes = {k: int(v.size) for k, v in out.items()}
    if sizes != C.EXPECTED_SPLIT_SIZES:
        raise RuntimeError(f"official split sizes {sizes} != {C.EXPECTED_SPLIT_SIZES}")
    union = np.sort(np.concatenate(list(out.values())))
    if union.size != len(np.unique(union)):
        raise RuntimeError("Train/Valid row ids overlap")
    # the official Test file exists but is never opened by this round
    return out


def _raw_atom_records() -> pd.DataFrame:
    node = pd.read_csv(C.OGB_DIR / "raw/node-feat.csv.gz", header=None)
    n_nodes = pd.read_csv(C.OGB_DIR / "raw/num-node-list.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
    if node.shape[0] != int(n_nodes.sum()):
        raise RuntimeError("raw node-feat rows != sum(num-node-list)")
    graph = np.repeat(np.arange(n_nodes.size, dtype=np.int64), n_nodes)
    return pd.DataFrame(
        {
            "graph": graph,
            "Z": node[0].to_numpy(dtype=np.int64) + OGB_ATOMIC_INDEX_OFFSET,
            "charge": node[3].to_numpy(dtype=np.int64) - OGB_CHARGE_INDEX_NEUTRAL,
            "num_h": node[4].to_numpy(dtype=np.int64),
        }
    )


def verify_identity(mol: pd.DataFrame, rows_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    """Row id -> SMILES -> raw graph identity, using only original artefacts."""
    from rdkit import Chem

    atom = _raw_atom_records()
    used = np.sort(np.concatenate(list(rows_by_split.values())))
    sub = atom[atom["graph"].isin(set(int(x) for x in used))]
    grouped = sub.groupby("graph")

    by_row = mol.set_index("row_id")
    mismatched_Z: list[int] = []
    mismatched_charge: list[int] = []
    mismatched_n: list[int] = []
    charge_present_in_smiles = 0
    charge_present_in_raw = 0
    for row in used:
        rec = by_row.loc[int(row)]
        m = C.mol_from_smiles(rec["smiles"])
        zs = sorted(a.GetAtomicNum() for a in m.GetAtoms())
        qs = sorted(a.GetFormalCharge() for a in m.GetAtoms())
        g = grouped.get_group(int(row))
        z_raw = sorted(int(x) for x in g["Z"].to_numpy())
        q_raw = sorted(int(x) for x in g["charge"].to_numpy())
        if zs != z_raw:
            mismatched_Z.append(int(row))
        if qs != q_raw:
            mismatched_charge.append(int(row))
        if int(m.GetNumAtoms()) != int(g.shape[0]):
            mismatched_n.append(int(row))
        if any(q != 0 for q in qs):
            charge_present_in_smiles += 1
        if any(q != 0 for q in q_raw):
            charge_present_in_raw += 1

    smiles_dupes = int(mol["smiles"].duplicated().sum())
    canon_dupes = int(mol["canon_smiles"].duplicated().sum()) if "canon_smiles" in mol else None
    return {
        "rows_checked": int(used.size),
        "atom_multiset_mismatch_rows": mismatched_Z[:20],
        "n_atom_multiset_mismatch": len(mismatched_Z),
        "charge_multiset_mismatch_rows": mismatched_charge[:20],
        "n_charge_multiset_mismatch": len(mismatched_charge),
        "atom_count_mismatch_rows": mismatched_n[:20],
        "n_atom_count_mismatch": len(mismatched_n),
        "rows_with_nonzero_formal_charge_smiles": charge_present_in_smiles,
        "rows_with_nonzero_formal_charge_raw": charge_present_in_raw,
        "duplicate_smiles_strings": smiles_dupes,
        "duplicate_canonical_smiles": canon_dupes,
        "identity_ok": bool(not mismatched_Z and not mismatched_charge and not mismatched_n),
    }


def build_base() -> Base:
    rows_by_split = load_official_rows()
    mol = pd.read_csv(C.OGB_DIR / "mapping/mol.csv.gz")
    if len(mol) != C.N_MOLECULES:
        raise RuntimeError(f"mol.csv rows {len(mol)} != {C.N_MOLECULES}")
    mol = mol.reset_index().rename(columns={"index": "row_id"})
    split_of = {int(r): s for s, arr in rows_by_split.items() for r in arr}
    mol["split"] = mol["row_id"].map(split_of)
    mol = mol[mol["split"].notna()].copy()
    mol["split"] = mol["split"].astype(str)
    mol["canon_smiles"] = [C.canonical_smiles(s) for s in mol["smiles"]]

    ident = verify_identity(mol.reset_index(drop=True), rows_by_split)

    err = pd.read_csv(C.EBR_PAIRED)
    err = err[err["split"].isin(["train", "valid"])].copy()
    err["row_id"] = err["row_id"].astype(int)
    if len(err) != 3360 + 420:
        raise RuntimeError(f"paired-error rows {len(err)} != 3780")

    # row id <-> split agreement between the frozen table and the official file
    part = mol.set_index("row_id")["split"]
    split_match = bool((err["split"].to_numpy() == err["row_id"].map(part).to_numpy()).all())

    # label agreement between the frozen table and the official label column
    lab = mol.set_index("row_id")["exp"]
    label_gap = float(np.max(np.abs(err["y_raw"].to_numpy() - err["row_id"].map(lab).to_numpy())))

    # independent SMILES->graph check also implies the y column belongs to that molecule
    base = mol.merge(err, on=["row_id", "split"], how="inner", validate="one_to_one")
    if len(base) != len(err):
        raise RuntimeError("join lost molecules")

    checks = {
        "round_id": C.ROUND_ID,
        "paired_errors": C.rel(C.EBR_PAIRED),
        "paired_errors_sha256": C.sha256_file(C.EBR_PAIRED),
        "mol_csv_sha256": C.sha256_file(C.OGB_DIR / "mapping/mol.csv.gz"),
        "split_files": {
            s: C.sha256_file(C.OGB_DIR / "split/scaffold" / f"{s}.csv.gz") for s in ("train", "valid")
        },
        "test_split_file_opened": False,
        "n_rows": {"train": int((err.split == "train").sum()), "valid": int((err.split == "valid").sum())},
        "split_agreement_with_official_file": split_match,
        "max_label_gap_vs_mol_csv": label_gap,
        "identity_vs_raw_ogb_graph": ident,
        "rmse": {},
    }
    for split in ("train", "valid"):
        d = err[err.split == split]
        checks["rmse"][split] = {
            "A": C.rmse(d["error_A"]),
            "B": C.rmse(d["error_B"]),
            "MAE_A": float(d["error_A"].abs().mean()),
            "MAE_B": float(d["error_B"].abs().mean()),
        }
    ref_ok = {}
    for split, refs in C.REF_RMSE.items():
        for arm, ref in refs.items():
            if ref is None:
                continue
            got = checks["rmse"][split][arm]
            ref_ok[f"{split}_{arm}"] = {
                "reference": ref,
                "recomputed": got,
                "abs_diff": abs(got - ref),
                "within_1e-5": bool(abs(got - ref) <= C.REFATION_TOL),
            }
    checks["rmse_reference_alignment"] = ref_ok
    if not all(v["within_1e-5"] for v in ref_ok.values()):
        raise RuntimeError(f"RMSE does not align with the reference round: {ref_ok}")
    if not (split_match and label_gap < 1e-6 and ident["identity_ok"]):
        raise RuntimeError(f"identity verification failed: split={split_match} label={label_gap} {ident}")

    base["e_A2"] = base["error_A"] ** 2
    base["e_B2"] = base["error_B"] ** 2
    base["gain"] = base["e_A2"] - base["e_B2"]
    base["common2"] = base[["e_A2", "e_B2"]].min(axis=1)
    err = base[["split", "row_id", "y_raw", "pred_A", "pred_B", "error_A", "error_B",
                "e_A2", "e_B2", "gain", "common2"]].copy()
    return Base(mol=mol.reset_index(drop=True), errors=err.reset_index(drop=True), checks=checks)


def check_historical_1425(base: Base) -> dict[str, Any]:
    """Read-only cross-check of the historical display case ogbg-mollipo_1425."""
    path = C.TRACK_DIR / "lcrc_readout_context_pilot_small_v1/analysis/cases/ogbg-mollipo_1425.json"
    payload = C.read_json(path)
    rec = base.mol[base.mol.row_id == 1425].iloc[0]
    err = base.errors[base.errors.row_id == 1425].iloc[0]
    same_smiles = str(payload.get("smiles")) == str(rec["smiles"])
    same_canon = C.canonical_smiles(str(payload.get("smiles"))) == str(rec["canon_smiles"])
    return {
        "path": C.rel(path),
        "sha256": C.sha256_file(path),
        "official_row_id_in_json": payload.get("official_row_id"),
        "smiles_string_identical": bool(same_smiles),
        "canonical_smiles_identical": bool(same_canon),
        "json_y_raw": payload.get("y_raw"),
        "official_y": float(rec["exp"]),
        "y_abs_diff": abs(float(payload.get("y_raw")) - float(rec["exp"])),
        "json_pred_raw": payload.get("pred_raw"),
        "frozen_pred_A": float(err["pred_A"]),
        "frozen_pred_B": float(err["pred_B"]),
        "identity_ok": bool(same_canon and abs(float(payload.get("y_raw")) - float(rec["exp"])) < 1e-6),
    }
