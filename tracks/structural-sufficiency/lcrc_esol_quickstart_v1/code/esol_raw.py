"""ESOL / Delaney source preparation (RDKit, CPU, one-off, label-blind).

Three artifacts are produced from the hash-pinned official CSV:

``esol_raw_v1.json.gz``
    A label-free raw molecular graph per row, encoded with the **same**
    RDKit function the Lipophilicity round used
    (``bjoc_lipophilicity_adapter_v1.raw_from_smiles``): atom class from the
    frozen 15-class element x aromatic vocabulary with an explicit ``UNK``
    slot, formal-charge bucket, total-H bucket, directed physical bonds with
    bond class (SINGLE/DOUBLE/TRIPLE/AROMATIC/UNK) and bond-stereo bucket.
    Reusing that function is what keeps the ESOL atom/bond vocabulary
    bit-identical to ``lipo_v1``; no new structural encoding is introduced.

``esol_physchem_v1.json``
    Six RDKit descriptors **recomputed from SMILES** (MolWt, MolLogP, TPSA,
    HBD, HBA, RotatableBonds).  These are the only non-topological numbers
    this round may ever see, and only the P2 ``ESOL-PHYSCHEM6`` arm uses them,
    on the global ``rho`` branch.  The CSV's own physchem columns are read
    exclusively by :func:`physchem_csv_crosscheck` (an audit that our
    recomputation is sane); they never enter a model input.

``esol_scaffolds_v1.json``
    Bemis--Murcko scaffold string per row (RDKit, ``includeChirality=False``),
    used for the label-blind split.

The ``ESOL predicted log solubility`` column is never parsed.  The measured
label column is not copied into any of these artifacts.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import esol_common as C

import bjoc_lipophilicity_adapter_v1 as LIP  # noqa: E402  (read-only frozen encoder)

RDGUNK_REPORT_LIMIT = 20


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def read_csv() -> dict[str, Any]:
    """Read and hash-verify the official CSV.

    Only ``Compound ID``, ``smiles`` and the measured label column are parsed
    into memory.  The ``ESOL predicted ...`` column is never touched, and the
    pre-computed physchem columns are only reachable through
    :func:`physchem_csv_crosscheck`.
    """
    actual = C.sha256_file(C.CSV_PATH)
    if actual != C.CSV_SHA256:
        raise PermissionError(f"delaney-processed.csv sha256 {actual} != pinned {C.CSV_SHA256}")
    with open(C.CSV_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    if header != C.CSV_COLUMNS:
        raise ValueError(f"unexpected CSV header {header!r}")
    if len(rows) != C.N_MOLECULES:
        raise ValueError(f"expected {C.N_MOLECULES} rows, parsed {len(rows)}")
    ids = [str(row[C.ID_COLUMN]) for row in rows]
    smiles = [str(row[C.SMILES_COLUMN]).strip() for row in rows]
    targets: list[float] = []
    for i, row in enumerate(rows):
        raw = row[C.TARGET_COLUMN]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {i}: non-numeric target {raw!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"row {i}: non-finite target {value!r}")
        targets.append(value)
    return {
        "ids": ids,
        "smiles": smiles,
        "y": np.asarray(targets, dtype=np.float64),
        "header": list(header),
        "sha256": actual,
        "n": len(rows),
        "parsed_columns": [C.ID_COLUMN, C.SMILES_COLUMN, C.TARGET_COLUMN],
        "forbidden_columns_parsed": [],
    }


# ---------------------------------------------------------------------------
# raw graphs (RDKit encoder shared with the Lipophilicity round)
# ---------------------------------------------------------------------------


def build_raw_graphs(csv_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import rdkit

    payload = read_csv() if csv_payload is None else dict(csv_payload)
    raws = []
    for i, smiles in enumerate(payload["smiles"]):
        raws.append(LIP.raw_from_smiles(smiles, i, payload["ids"][i]))
    artifact = {
        "schema": "bjoc-lipo-rawgraph-v1",
        "vocab": LIP.vocab_manifest(),
        "meta": {
            "dataset": "ESOL / Delaney",
            "label_free": True,
            "schema_source": "bjoc-lipo-rawgraph-v1 (read-only reuse)",
            "encoder": "bjoc_lipophilicity_adapter_v1.raw_from_smiles",
            "encoder_sha256": C.sha256_file(C.LIPOPHILICITY_CODE / "bjoc_lipophilicity_adapter_v1.py"),
            "source_url": C.SOURCE_URL,
            "loader_definition": C.LOADER_URL,
            "csv_sha256": payload["sha256"],
            "csv_rows": int(payload["n"]),
            "target_column": C.TARGET_COLUMN,
            "target_values_stored": False,
            "forbidden_columns_never_parsed": list(C.FORBIDDEN_COLUMNS),
            "download_utc": C.utcnow(),
            "rdkit": rdkit.__version__,
            "python": C.env_info()["python"],
            "numpy": np.__version__,
        },
        "rows": [LIP.raw_to_record(raw) for raw in raws],
    }
    for i, row in enumerate(artifact["rows"]):
        if int(row["mol_index"]) != i:
            raise ValueError(f"raw row {i} has mol_index {row['mol_index']}")
        if "exp" in row or "y" in row:
            raise PermissionError("raw row unexpectedly carries a label field")
    C.RAW_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(C.RAW_GRAPH, "wt", encoding="utf-8") as handle:
        json.dump(artifact, handle, separators=(",", ":"))
    return artifact


def load_raw_graphs(path: Path | str = C.RAW_GRAPH) -> dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "bjoc-lipo-rawgraph-v1":
        raise ValueError(f"unexpected raw graph schema {payload.get('schema')!r}")
    meta = payload.get("meta", {})
    if meta.get("label_free") is not True:
        raise PermissionError("ESOL raw artifact does not certify label_free=true")
    if meta.get("target_values_stored") is not False:
        raise PermissionError("ESOL raw artifact is not certified target-free")
    if str(meta.get("csv_sha256")) != C.CSV_SHA256:
        raise PermissionError("ESOL raw artifact was built from a different CSV")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != C.N_MOLECULES:
        raise ValueError("ESOL raw artifact has the wrong row count")
    for i, row in enumerate(rows):
        if int(row.get("mol_index", -1)) != i:
            raise ValueError(f"raw row {i} has mol_index {row.get('mol_index')!r}")
        if "exp" in row:
            raise PermissionError(f"raw row {i} unexpectedly contains a label field")
    return payload


# ---------------------------------------------------------------------------
# six recomputed physicochemical descriptors
# ---------------------------------------------------------------------------


def compute_physchem(smiles: str) -> list[float]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r} for descriptor recomputation")
    values = [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(Descriptors.TPSA(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Lipinski.NumRotatableBonds(mol)),
    ]
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite descriptor for {smiles!r}")
    return values


def build_physchem(csv_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import rdkit

    payload = read_csv() if csv_payload is None else dict(csv_payload)
    matrix = np.asarray([compute_physchem(s) for s in payload["smiles"]], dtype=np.float64)
    if matrix.shape != (C.N_MOLECULES, len(C.PHYSCHEM_NAMES)):
        raise ValueError(f"physchem matrix shape {matrix.shape}")
    artifact = {
        "schema": "esol-physchem-v1",
        "label_free": True,
        "recomputed_from_smiles": True,
        "csv_sha256": payload["sha256"],
        "names": list(C.PHYSCHEM_NAMES),
        "definition": dict(C.PHYSCHEM_DEFINITION),
        "rdkit": rdkit.__version__,
        "python": C.env_info()["python"],
        "created_utc": C.utcnow(),
        "n": int(matrix.shape[0]),
        "matrix": matrix.astype(np.float64).tolist(),
        "csv_columns_used": False,
        "note": (
            "recomputed with RDKit from the SMILES column only; the CSV's own "
            "Molecular Weight / H-Bond Donors / Rotatable Bonds / Polar Surface Area "
            "columns are not read here and never enter a model input"
        ),
    }
    C.PHYSCHEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(C.PHYSCHEM_FILE, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, separators=(",", ":"))
    return artifact


def load_physchem(path: Path | str = C.PHYSCHEM_FILE) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "esol-physchem-v1":
        raise ValueError("unexpected physchem schema")
    if payload.get("recomputed_from_smiles") is not True:
        raise PermissionError("physchem artifact is not certified as recomputed from SMILES")
    if payload.get("csv_columns_used") is not False:
        raise PermissionError("physchem artifact claims CSV descriptor columns were used")
    if str(payload.get("csv_sha256")) != C.CSV_SHA256:
        raise PermissionError("physchem artifact was built from a different CSV")
    if list(payload.get("names", [])) != list(C.PHYSCHEM_NAMES):
        raise ValueError("physchem descriptor names drifted")
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    if matrix.shape != (C.N_MOLECULES, len(C.PHYSCHEM_NAMES)):
        raise ValueError(f"physchem matrix shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("physchem matrix has non-finite entries")
    return {"matrix": matrix, "payload": payload}


def physchem_csv_crosscheck(csv_payload: Mapping[str, Any], physchem: np.ndarray) -> dict[str, Any]:
    """Audit only: our recomputation vs the CSV's own descriptive columns.

    The CSV columns are read *here and only here*, to show that the six
    recomputed descriptors are sane.  They never enter a model input; HBA and
    MolLogP have no CSV counterpart and are therefore not cross-checked.
    """
    with open(C.CSV_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != C.N_MOLECULES:
        raise ValueError("CSV row count drifted during the crosscheck")
    pairs = {
        "MolWt": "Molecular Weight",
        "TPSA": "Polar Surface Area",
        "HBD": "Number of H-Bond Donors",
        "RotatableBonds": "Number of Rotatable Bonds",
    }
    out: dict[str, Any] = {
        "note": "audit only; the CSV descriptor columns are never a model input",
        "csv_columns_read_here_only": list(pairs.values()),
        "columns_without_csv_counterpart": ["MolLogP", "HBA"],
        "comparisons": {},
    }
    names = list(C.PHYSCHEM_NAMES)
    for ours, theirs in pairs.items():
        col = np.asarray([float(row[theirs]) for row in rows], dtype=np.float64)
        mine = physchem[:, names.index(ours)]
        diff = np.abs(col - mine)
        out["comparisons"][ours] = {
            "csv_column": theirs,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "fraction_within_1e-6": float(np.mean(diff <= 1e-6)),
            "fraction_within_0.01": float(np.mean(diff <= 0.01)),
            "n_csv_missing_or_zero_but_ours_nonzero": int(np.sum((col == 0) & (mine != 0))),
        }
    return out


# ---------------------------------------------------------------------------
# Bemis--Murcko scaffolds
# ---------------------------------------------------------------------------


def bemis_murcko_scaffolds(smiles: list[str]) -> list[str]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    RDLogger.DisableLog("rdApp.*")
    out: list[str] = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"invalid SMILES at row {index}: {smi!r}")
        out.append(MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return out


def build_scaffolds(csv_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import rdkit

    payload = read_csv() if csv_payload is None else dict(csv_payload)
    scaffolds = bemis_murcko_scaffolds(payload["smiles"])
    artifact = {
        "schema": "esol-scaffolds-v1",
        "label_free": True,
        "csv_sha256": payload["sha256"],
        "definition": "RDKit MurckoScaffoldSmiles(mol, includeChirality=False)",
        "rdkit": rdkit.__version__,
        "n": len(scaffolds),
        "n_empty": int(sum(1 for s in scaffolds if s == "")),
        "scaffold_by_row": scaffolds,
        "created_utc": C.utcnow(),
    }
    C.SCAFFOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(C.SCAFFOLD_FILE, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, separators=(",", ":"))
    return artifact


def load_scaffolds(path: Path | str = C.SCAFFOLD_FILE) -> dict[str, Any]:
    payload = C.read_json(path)
    if payload.get("schema") != "esol-scaffolds-v1":
        raise ValueError("unexpected scaffold schema")
    if payload.get("label_free") is not True:
        raise PermissionError("scaffold artifact is not label-free")
    if str(payload.get("csv_sha256")) != C.CSV_SHA256:
        raise PermissionError("scaffold artifact was built from a different CSV")
    if len(payload["scaffold_by_row"]) != C.N_MOLECULES:
        raise ValueError("scaffold vector has the wrong length")
    return payload


# ---------------------------------------------------------------------------
# source manifest
# ---------------------------------------------------------------------------


def source_manifest() -> dict[str, Any]:
    csv_payload = read_csv()
    raw = load_raw_graphs()
    physchem = load_physchem()
    scaffolds = load_scaffolds()
    return {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "environment": C.env_info(),
        "dataset": {
            "name": "ESOL / Delaney",
            "source_url": C.SOURCE_URL,
            "loader_definition": C.LOADER_URL,
            "n_rows": int(csv_payload["n"]),
            "columns": list(csv_payload["header"]),
            "id_column": C.ID_COLUMN,
            "smiles_column": C.SMILES_COLUMN,
            "target_column": C.TARGET_COLUMN,
            "target_unit": "log(mol/L)",
            "forbidden_columns": list(C.FORBIDDEN_COLUMNS),
            "forbidden_columns_parsed": [],
        },
        "files": {
            "csv": {"path": C.track_rel(C.CSV_PATH), "sha256": C.sha256_file(C.CSV_PATH), "bytes": C.CSV_PATH.stat().st_size},
            "raw_graph": {
                "path": C.track_rel(C.RAW_GRAPH),
                "sha256": C.sha256_file(C.RAW_GRAPH),
                "schema": raw["schema"],
                "label_free": True,
                "rdkit": raw["meta"]["rdkit"],
                "encoder_sha256": raw["meta"]["encoder_sha256"],
            },
            "physchem": {
                "path": C.track_rel(C.PHYSCHEM_FILE),
                "sha256": C.sha256_file(C.PHYSCHEM_FILE),
                "names": list(C.PHYSCHEM_NAMES),
                "definition": dict(C.PHYSCHEM_DEFINITION),
                "recomputed_from_smiles": True,
                "rdkit": physchem["payload"]["rdkit"],
            },
            "scaffolds": {
                "path": C.track_rel(C.SCAFFOLD_FILE),
                "sha256": C.sha256_file(C.SCAFFOLD_FILE),
                "definition": scaffolds["definition"],
                "rdkit": scaffolds["rdkit"],
                "n_empty_scaffolds": int(scaffolds["n_empty"]),
            },
        },
        "hashes": {
            "row_ids_sha256": C.rows_sha256(list(range(C.N_MOLECULES))),
            "smiles_vector_sha256": C.sha256_bytes("\n".join(csv_payload["smiles"]).encode("utf-8")),
            "target_vector_sha256": C.vector_sha256_float64(csv_payload["y"]),
        },
    }


__all__ = [
    "read_csv",
    "build_raw_graphs",
    "load_raw_graphs",
    "compute_physchem",
    "build_physchem",
    "load_physchem",
    "physchem_csv_crosscheck",
    "bemis_murcko_scaffolds",
    "build_scaffolds",
    "load_scaffolds",
    "source_manifest",
]
