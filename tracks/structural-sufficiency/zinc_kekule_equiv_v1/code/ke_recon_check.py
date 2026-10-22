"""Audit the existing BJOC RDKit reconstruction against the strict frozen-H builder.

The existing function is usable for constitution/sanitisation, but it derives
implicit H from the dictionary word and cannot represent neutral ``[nH]``
explicitly.  This check compares it to :func:`ke_chem.build_mol_from_graph` on
the official train/valid graphs and records where the H state is inferred.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[3] / "tracks/structure-contribution/code"))

import bjoc_mmp_feasibility_v1 as BJ  # noqa: E402
import ke_chem as KC  # noqa: E402
import ke_graph as KG  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"


def check_split(split: str) -> dict:
    ds = KG.load_pyg(split)
    out = Counter()
    examples = []
    for row in range(len(ds)):
        cats, edges, orders, ei, ea = KG.graph_arrays(ds[row])
        out["n"] += 1
        try:
            old = BJ.build_rdkit_mol_from_graph(cats, ei, ea)
        except Exception as exc:
            out["old_fail"] += 1
            if len(examples) < 8:
                examples.append({"row": row, "error": f"old:{type(exc).__name__}"})
            continue
        try:
            new = KC.build_mol_from_graph(cats, edges, orders)
        except Exception as exc:
            out["new_fail"] += 1
            if len(examples) < 8:
                examples.append({"row": row, "error": f"new:{type(exc).__name__}"})
            continue
        if Chem.MolToSmiles(old, isomericSmiles=False) != Chem.MolToSmiles(
                new, isomericSmiles=False):
            out["smiles_mismatch"] += 1
            if len(examples) < 8:
                examples.append({"row": row, "error": "smiles_mismatch"})
            continue
        old_h = [int(a.GetTotalNumHs()) for a in old.GetAtoms()]
        new_h = [int(a.GetTotalNumHs()) for a in new.GetAtoms()]
        if old_h != new_h:
            out["total_h_mismatch"] += 1
            if len(examples) < 8:
                examples.append({"row": row, "error": "total_h_mismatch"})
        dict_h = KC.dictionary_h(cats)
        if any(a.GetIsAromatic() and int(a.GetTotalNumHs()) != int(dict_h[i])
               for i, a in enumerate(new.GetAtoms())):
            out["molecules_with_inferred_aromatic_h"] += 1
        if any((not a.GetIsAromatic()) and int(a.GetTotalNumHs()) != int(dict_h[i])
               for i, a in enumerate(new.GetAtoms())):
            out["molecules_with_inferred_nonaromatic_h"] += 1
    return {"split": split, "counts": dict(out), "examples": examples}


def main() -> int:
    result = {
        "rdkit_version": KC.rdkit_version(),
        "old_function": "tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py::build_rdkit_mol_from_graph",
        "new_function": "tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_chem.py::build_mol_from_graph",
        "note": ("Both reconstruct from the committed atom/bond words and sanitize. "
                 "The old function is applicable for constitution, but it cannot "
                 "make neutral [nH] explicit; H is inferred. The new path freezes "
                 "the inferred total H in candidate verification."),
        "valid": check_split("valid"),
        "train": check_split("train"),
        "official_test_loaded": False,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "reconstruction_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
