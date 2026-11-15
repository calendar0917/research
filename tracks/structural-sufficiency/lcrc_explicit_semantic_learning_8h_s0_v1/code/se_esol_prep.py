"""Local-only ESOL preparation: build the committed label-free artifacts.

Needs RDKit (the ``lipo`` dependency group); run once locally, never on the GPU
hosts.  Writes ``data/esol_{raw,labels,split}_v1.*`` under the round directory.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import se_common as C
import se_data as SED

import lp_common as LPC
import lp_data as LPD


def main() -> int:
    import sys

    for p in (str(C.PREP_CODE),):
        if p not in sys.path:
            sys.path.insert(0, p)
    import adapters as PREP_ADAPTERS

    root = SED._esol_download_root()
    if not root.is_dir():
        raise SystemExit(f"missing official {root}")
    raw_rows = _official_raw_rows(root)
    y_all, smiles = _official_labels(root)
    idx = SED._esol_split(root)

    spec = LPD.frozen_spec()
    # label-free artifact
    payload = {
        "schema": "lcrc-ogb-molesol-rawgraph-v1",
        "dataset": SED.OGB_MOLESOL,
        "label_free": True,
        "n_molecules": len(raw_rows),
        "row_order": "OGB mol.csv.gz row order (0-based)",
        "rows": raw_rows,
    }
    C.DATA.mkdir(parents=True, exist_ok=True)
    with gzip.open(SED.ESOL_RAW, "wb", compresslevel=9) as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    # labels + split
    np.savez(SED.ESOL_LABELS, y=y_all)
    C.write_json(SED.ESOL_SPLIT, {"rows": idx, "sizes": {k: len(v) for k, v in idx.items()},
                                  "source": f"official {SED.OGB_MOLESOL} scaffold split"})
    # validate round trip: build graphs and check the schema/codes
    view = SED.build_esol_view()
    print("ESOL view sizes", {k: int(view.caches[k]["n_molecules"]) for k in LPC.SPLITS})
    print("target norm", view.target_norm["mean"], view.target_norm["scale"])
    return 0


def _official_raw_rows(root: Path) -> list[dict]:
    import sys

    for p in (str(C.PREP_CODE),):
        if p not in sys.path:
            sys.path.insert(0, p)
    import mb_rdkit as R

    smiles = _read_smiles(root)
    rows = []
    for i, smi in enumerate(smiles):
        raw = R.smiles_to_raw_row(str(smi), i)
        raw["mol_id"] = str(i)
        rows.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in raw.items()})
    return rows


def _read_smiles(root: Path) -> list[str]:
    with gzip.open(root / "mapping" / "mol.csv.gz", "rt", encoding="utf-8") as fh:
        fh.readline()
        return [line.rstrip("\n").split(",")[1] for line in fh if line.strip()]


def _official_labels(root: Path) -> tuple[np.ndarray, list[str]]:
    with gzip.open(root / "mapping" / "mol.csv.gz", "rt", encoding="utf-8") as fh:
        fh.readline()
        rows = [line.rstrip("\n").split(",") for line in fh if line.strip()]
    y = np.asarray([float(r[0]) for r in rows], dtype=np.float64)
    smiles = [r[1] for r in rows]
    return y, smiles


if __name__ == "__main__":
    raise SystemExit(main())
