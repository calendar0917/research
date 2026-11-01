"""Build and verify the committed, label-free raw-attribute artifact.

RDKit is a *local* dependency (`lipo` group) and is not part of the remote
execution environment.  The GPU hosts must therefore consume a committed,
label-free artifact, exactly as the earlier Lipophilicity rounds did
(`bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz`).

``analysis/source/raw_attributes_v1.json.gz`` holds, for every source row of every
dataset, only the frozen raw attributes that the ``lipo_v1`` schema is defined
on:

    atom_class, atom_charge, atom_h,
    edge_index (both directions), edge_class, edge_stereo,
    n_components, smiles, canonical_smiles

There is no target, no descriptor and no derived label.  ``meta.label_free`` is
asserted by the loader, and ``mb_p0`` re-derives all of it from the vendored
SMILES and requires byte-equality.

Usage
-----
    uv run python .../code/mb_raw_artifact.py            # build + manifest
    uv run python .../code/mb_raw_artifact.py --check    # verify only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

import mb_common as C
import mb_rdkit as R
import mb_source as S

ARTIFACT = C.DATA_DIR / "raw_attributes_v1.json.gz"
MANIFEST = C.DATA_DIR / "raw_artifact_manifest.json"
SCHEMA = "lcrc-molnet-rawgraph-v1"
FIELDS = (
    "mol_index",
    "smiles",
    "canonical_smiles",
    "atom_class",
    "atom_charge",
    "atom_h",
    "edge_index",
    "edge_class",
    "edge_stereo",
    "n_components",
)


def _row_to_json(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in FIELDS:
        value = row[key]
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent-of-JSON-layout digest over the raw attribute arrays."""
    h = hashlib.sha256()
    for row in rows:
        h.update(int(row["mol_index"]).to_bytes(8, "little", signed=True))
        for key in ("atom_class", "atom_charge", "atom_h", "edge_class", "edge_stereo"):
            arr = np.ascontiguousarray(np.asarray(row[key], dtype=np.int64))
            h.update(key.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        h.update(np.ascontiguousarray(np.asarray(row["edge_index"], dtype=np.int64)).tobytes())
        h.update(str(int(row["n_components"])).encode())
        h.update(str(row["smiles"]).encode())
    return h.hexdigest()


def build() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "vocab": {
            "atom_classes": [f"{name}" + ("_aromatic" if arom else "") for name, arom in R.ATOM_CLASSES]
            + ["UNK"],
            "bond_classes": list(R.BOND_CLASSES) + ["UNK"],
            "charge_buckets": list(R.CHARGE_BUCKETS) + ["UNK"],
            "h_buckets": list(R.H_BUCKETS) + ["UNK"],
            "stereo_buckets": ["NONE", "E", "Z", "OTHER"],
            "atom_unk": R.ATOM_UNK,
            "bond_unk": R.BOND_UNK,
            "charge_unk": R.CHARGE_UNK,
            "h_unk": R.H_UNK,
        },
        "meta": {
            "label_free": True,
            "labels_present": False,
            "created_utc": C.utcnow(),
            "rdkit": C.rdkit_version(),
            "source_repository": S.CHEMBENCH_URL,
            "source_commit": S.CHEMBENCH_COMMIT,
            "provenance": (
                "raw atom/bond attributes reproduced from vendored ChemBench SMILES with the frozen "
                "lipo_v1 vocabulary of tracks/structure-contribution/code/bjoc_lipophilicity_adapter_v1.py"
            ),
            "datasets": {},
        },
        "rows": {},
    }
    for key in C.DATASET_ORDER:
        src = S.load_dataset(key, labels="none")
        rows = []
        failures = []
        for i, smiles in enumerate(src["smiles"]):
            try:
                rows.append(_row_to_json(R.smiles_to_raw_row(smiles, i)))
            except Exception as exc:  # recorded; not dropped
                failures.append({"row": i, "smiles": smiles, "error": f"{type(exc).__name__}: {exc}"})
        if failures:
            raise SystemExit(f"{key}: {len(failures)} unparseable SMILES, refusing to write the artifact")
        payload["rows"][key] = rows
        payload["meta"]["datasets"][key] = {
            "csv_sha256": src["csv_sha256"],
            "n_rows": len(rows),
            "n_atoms": int(sum(len(r["atom_class"]) for r in rows)),
            "n_directed_bond_records": int(sum(len(r["edge_class"]) for r in rows)),
            "rows_digest_sha256": _row_digest(rows),
        }
        C.log(f"[artifact] {key}: {len(rows)} rows built")

    C.ensure_dir(ARTIFACT.parent)
    with gzip.open(ARTIFACT, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    manifest = {
        "schema": "lcrc-molnet-rawgraph-manifest-v1",
        "created_utc": C.utcnow(),
        "artifact": C.track_rel(ARTIFACT),
        "artifact_sha256": C.sha256_file(ARTIFACT),
        "artifact_size_bytes": int(ARTIFACT.stat().st_size),
        "artifact_schema": SCHEMA,
        "label_free": True,
        "source_commit": S.CHEMBENCH_COMMIT,
        "datasets": payload["meta"]["datasets"],
        "note": (
            "consumed by the GPU path; mb_p0 re-derives every array from the vendored SMILES "
            "and requires equality, so the artifact is auditable, not trusted"
        ),
    }
    C.write_json(MANIFEST, manifest)
    C.log(f"wrote {C.track_rel(ARTIFACT)} ({manifest['artifact_size_bytes']/1e6:.2f} MB)")
    return manifest


def load(*, verify: bool = True) -> dict[str, Any]:
    if not ARTIFACT.exists() or not MANIFEST.exists():
        raise FileNotFoundError(f"raw artifact missing; run {Path(__file__).name}")
    manifest = C.read_json(MANIFEST)
    if verify:
        actual = C.sha256_file(ARTIFACT)
        if actual != manifest["artifact_sha256"]:
            raise ValueError(f"raw artifact sha256 {actual} != manifest {manifest['artifact_sha256']}")
    with gzip.open(ARTIFACT, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unexpected raw artifact schema {payload.get('schema')!r}")
    if payload.get("meta", {}).get("label_free") is not True:
        raise PermissionError("raw artifact does not certify label_free=true")
    for key, rows in payload["rows"].items():
        for i, row in enumerate(rows):
            if "exp" in row or "label" in row or "target" in row:
                raise PermissionError(f"{key} row {i} contains a label-like field")
            if int(row["mol_index"]) != i:
                raise ValueError(f"{key} row {i} has mol_index {row['mol_index']}")
    return payload


def check() -> dict[str, Any]:
    payload = load(verify=True)
    out = {}
    for key in C.DATASET_ORDER:
        rows = payload["rows"][key]
        digest = _row_digest(rows)
        recorded = payload["meta"]["datasets"][key]["rows_digest_sha256"]
        out[key] = {"n_rows": len(rows), "digest_matches": digest == recorded}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="raw attribute artifact builder")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        report = check()
        for key, item in report.items():
            C.log(f"[artifact] {key}: {item}")
        if not all(item["digest_matches"] for item in report.values()):
            raise SystemExit("artifact digest mismatch")
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
