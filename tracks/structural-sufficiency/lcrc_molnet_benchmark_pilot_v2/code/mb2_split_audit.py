"""Post-freeze diagnostic: how much *scaffold* structure leaks across the splits.

This module changes nothing.  It is computed **after** the P0 audit and the
protocol freeze purely so the report can explain the numbers: a dataset whose
Validation/Test scaffolds are entirely absent from Train measures
out-of-distribution generalisation, while a dataset whose scaffolds are largely
shared measures interpolation.  ``mb2_p0`` already proved there are no duplicate
SMILES across splits (except the four Train/Validation ESOL groups); scaffold
sharing is a different, softer quantity.

Reading Test *SMILES* to compute a Murcko scaffold is not reading a Test *label*;
no label of the Test split is loaded here (``labels="none"``).

Usage
-----
    uv run python .../code/mb2_split_audit.py
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

import mb2_common as C
import mb2_source as S


def murcko(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)  # Murcko needs initialised ring information
        except Exception:  # noqa: BLE001
            return None
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:  # noqa: BLE001 - RDKit valence pre-condition on the Al-adjuvant salts
        return None


def audit(dataset: str) -> dict[str, Any]:
    data = S.load_dataset(dataset, labels="none")
    scaffolds = np.array([murcko(s) for s in data["smiles"]], dtype=object)
    failed = int(sum(1 for s in scaffolds if s is None))
    rows = {k: np.asarray(v, dtype=np.int64) for k, v in data["rows"].items()}
    sets = {k: set(scaffolds[v]) - {None} for k, v in rows.items()}
    out: dict[str, Any] = {
        "dataset": dataset,
        "split_method": data["meta"]["split_method"],
        "n_scaffold_parse_failures": failed,
        "n_unique_scaffolds": {k: len(sets[k]) for k in sets},
        "sizes": {k: int(len(v)) for k, v in rows.items()},
    }
    for split in ("valid", "test"):
        values = scaffolds[rows[split]]
        out[f"{split}_scaffold_seen_in_train_fraction"] = float(np.mean([v in sets["train"] for v in values]))
        out[f"{split}_unique_scaffolds_absent_from_train"] = int(len({v for v in values} - sets["train"]))
        out[f"{split}_row_duplicate_of_train_fraction"] = None
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scaffold-overlap diagnostic across the frozen splits")
    parser.add_argument("--out", default=str(C.ANALYSIS / "split_scaffold_overlap.json"))
    args = parser.parse_args(argv)
    rows = [audit(d) for d in C.DATASET_ORDER]
    payload = {
        "schema": "lcrc-molnet-v2-split-scaffold-overlap-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "computed_after": "P0 PASS and protocol_freeze.json; diagnostic only, no protocol change",
        "definition": (
            "Murcko scaffold (includeChirality=False) of each published SMILES; a split molecule counts as "
            "'seen' if its scaffold string occurs anywhere in Train"
        ),
        "test_labels_read": False,
        "datasets": {r["dataset"]: r for r in rows},
    }
    C.write_json(args.out, payload)
    for r in rows:
        C.log(
            f"{r['dataset']:14s} {r['split_method']:22s} valid-seen {r['valid_scaffold_seen_in_train_fraction']:.3f} "
            f"test-seen {r['test_scaffold_seen_in_train_fraction']:.3f} "
            f"(unseen valid scaffolds {r['valid_unique_scaffolds_absent_from_train']})"
        )
    C.log(f"wrote {C.track_rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
