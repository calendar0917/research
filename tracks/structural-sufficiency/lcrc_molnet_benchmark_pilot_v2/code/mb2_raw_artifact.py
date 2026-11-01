"""Build and verify the committed, label-free raw-attribute artifact (v2).

RDKit is a *local* dependency.  The GPU/CPU hosts must therefore consume a
committed, label-free artifact, exactly as the earlier rounds did.  v2 reuses the
v1 raw-attribute convention unchanged and extends it to the three new datasets.

Format
------
One ``<dataset>.npz`` per dataset (int arrays, no pickle) under
``analysis/source/raw/lcrc_raw_v2/`` plus ``raw_artifact_manifest.json``:

    atom_ptr      int64  [n_mol+1]        offsets into the per-atom arrays
    atom_class    int8   [n_atom]         frozen lipo_v1 (element x aromatic) code, 15 = UNK
    atom_charge   int8   [n_atom]         charge bucket, 5 = UNK
    atom_h        int8   [n_atom]         total-H bucket, 5 = UNK
    bond_ptr      int64  [n_mol+1]        offsets into the directed bond records
    edge_src      int32  [n_dir_bond]     directed bond begin atom (0-based, per molecule)
    edge_dst      int32  [n_dir_bond]
    edge_class    int8   [n_dir_bond]     SINGLE/DOUBLE/TRIPLE/AROMATIC -> 0..3, 4 = UNK
    edge_stereo   int8   [n_dir_bond]     NONE/E/Z -> 0..2, 3 = OTHER
    n_components  int16  [n_mol]
    mol_index     int32  [n_mol]          == positional row id (asserted)
    smiles        <U    [n_mol]           verbatim source SMILES
    canonical_smiles <U [n_mol]           RDKit canonical SMILES

There is no target, no descriptor and no derived label anywhere in the artifact;
:func:`load_rows` additionally refuses any object that would carry a label.

Usage
-----
    uv run python .../code/mb2_raw_artifact.py            # build + manifest
    uv run python .../code/mb2_raw_artifact.py --check    # verify only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch  # noqa: F401  (kept so a missing torch fails here, not mid-run)

import mb2_common as C
import mb2_rdkit as R
import mb2_source as S

ARTIFACT_DIR = C.SOURCE_DIR / "raw/lcrc_raw_v2"
MANIFEST = C.RAW_ARTIFACT_MANIFEST
SCHEMA = "lcrc-molnet-rawgraph-v2"

INT_FIELDS = ("atom_class", "atom_charge", "atom_h", "edge_class", "edge_stereo")
DTYPES = {
    "atom_ptr": np.int64,
    "atom_class": np.int8,
    "atom_charge": np.int8,
    "atom_h": np.int8,
    "bond_ptr": np.int64,
    "edge_src": np.int32,
    "edge_dst": np.int32,
    "edge_class": np.int8,
    "edge_stereo": np.int8,
    "n_components": np.int16,
    "mol_index": np.int32,
}


def artifact_path(dataset: str) -> Path:
    return ARTIFACT_DIR / f"{dataset}.npz"


def _encode_dataset(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    smiles = S.all_smiles(dataset)
    atom_class: list[np.ndarray] = []
    atom_charge: list[np.ndarray] = []
    atom_h: list[np.ndarray] = []
    edge_src: list[np.ndarray] = []
    edge_dst: list[np.ndarray] = []
    edge_class: list[np.ndarray] = []
    edge_stereo: list[np.ndarray] = []
    n_components: list[int] = []
    atom_ptr = [0]
    bond_ptr = [0]
    canonical: list[str] = []
    failures: list[dict[str, Any]] = []
    methods: dict[str, str] = {}
    fallback_rows: list[dict[str, Any]] = []
    for i, s in enumerate(smiles):
        try:
            row, method = R.smiles_to_raw_row_relaxed(s, i)
        except Exception as exc:  # recorded, never dropped
            failures.append({"row": i, "smiles": s, "error": f"{type(exc).__name__}: {exc}"})
            continue
        methods[method] = methods.get(method, 0) + 1
        if method != "strict":
            fallback_rows.append(
                {
                    "row": i,
                    "smiles": s,
                    "method": method,
                    "n_atoms": int(np.asarray(row["atom_class"]).size),
                    "n_components": int(row["n_components"]),
                    "atom_class_counts": {
                        str(int(k)): int(v)
                        for k, v in zip(*np.unique(np.asarray(row["atom_class"]), return_counts=True))
                    },
                    "why": row.get("strict_error", ""),
                }
            )
        atom_class.append(np.asarray(row["atom_class"], dtype=np.int8))
        atom_charge.append(np.asarray(row["atom_charge"], dtype=np.int8))
        atom_h.append(np.asarray(row["atom_h"], dtype=np.int8))
        edge_index = np.asarray(row["edge_index"], dtype=np.int64)
        edge_src.append(edge_index[0].astype(np.int32))
        edge_dst.append(edge_index[1].astype(np.int32))
        edge_class.append(np.asarray(row["edge_class"], dtype=np.int8))
        edge_stereo.append(np.asarray(row["edge_stereo"], dtype=np.int8))
        n_components.append(int(row["n_components"]))
        canonical.append(str(row["canonical_smiles"]))
        atom_ptr.append(atom_ptr[-1] + int(atom_class[-1].size))
        bond_ptr.append(bond_ptr[-1] + int(edge_class[-1].size))
    if failures:
        raise SystemExit(f"{dataset}: {len(failures)} unparseable SMILES, refusing to write the artifact")
    arrays = {
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "atom_class": np.concatenate(atom_class) if atom_class else np.zeros(0, dtype=np.int8),
        "atom_charge": np.concatenate(atom_charge) if atom_charge else np.zeros(0, dtype=np.int8),
        "atom_h": np.concatenate(atom_h) if atom_h else np.zeros(0, dtype=np.int8),
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "edge_src": np.concatenate(edge_src) if edge_src else np.zeros(0, dtype=np.int32),
        "edge_dst": np.concatenate(edge_dst) if edge_dst else np.zeros(0, dtype=np.int32),
        "edge_class": np.concatenate(edge_class) if edge_class else np.zeros(0, dtype=np.int8),
        "edge_stereo": np.concatenate(edge_stereo) if edge_stereo else np.zeros(0, dtype=np.int8),
        "n_components": np.asarray(n_components, dtype=np.int16),
        "mol_index": np.arange(len(smiles), dtype=np.int32),
        "smiles": np.asarray(smiles, dtype=str),
        "canonical_smiles": np.asarray(canonical, dtype=str),
    }
    audit = {
        "n_rows": int(len(smiles)),
        "n_atoms": int(arrays["atom_class"].size),
        "n_directed_bond_records": int(arrays["edge_class"].size),
        "n_multi_component_molecules": int((arrays["n_components"] > 1).sum()),
        "max_components": int(arrays["n_components"].max()) if arrays["n_components"].size else 0,
        "atom_unk_count": int((arrays["atom_class"] == R.ATOM_UNK).sum()),
        "bond_unk_count": int((arrays["edge_class"] == R.BOND_UNK).sum()),
        "charge_unk_count": int((arrays["atom_charge"] == R.CHARGE_UNK).sum()),
        "h_unk_count": int((arrays["atom_h"] == R.H_UNK).sum()),
        "parse_failures": 0,
        "parse_methods": methods,
        "unsanitized_fallback_rows": fallback_rows,
        "rdkit": C.public_env_info()["rdkit"],
    }
    return arrays, audit


def row_digest(dataset: str, arrays: Mapping[str, np.ndarray] | None = None) -> str:
    """Order-stable digest over every raw attribute of one dataset."""
    import hashlib

    arrs = load_arrays(dataset, verify=False) if arrays is None else arrays
    h = hashlib.sha256()
    for key in (
        "atom_ptr",
        "atom_class",
        "atom_charge",
        "atom_h",
        "bond_ptr",
        "edge_src",
        "edge_dst",
        "edge_class",
        "edge_stereo",
        "n_components",
        "mol_index",
    ):
        a = np.ascontiguousarray(arrs[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    for s in arrs["smiles"].tolist():
        h.update(str(s).encode())
    return h.hexdigest()


def build() -> dict[str, Any]:
    C.ensure_dir(ARTIFACT_DIR)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "label_free": True,
        "labels_present": False,
        "vocab": {
            "atom_classes": [f"{name}" + ("_aromatic" if arom else "") for name, arom in R.ATOM_CLASSES] + ["UNK"],
            "bond_classes": list(R.BOND_CLASSES) + ["UNK"],
            "charge_buckets": list(R.CHARGE_BUCKETS) + ["UNK"],
            "h_buckets": list(R.H_BUCKETS) + ["UNK"],
            "stereo_buckets": ["NONE", "E", "Z", "OTHER"],
            "atom_unk": R.ATOM_UNK,
            "bond_unk": R.BOND_UNK,
            "charge_unk": R.CHARGE_UNK,
            "h_unk": R.H_UNK,
        },
        "provenance": (
            "raw atom/bond attributes reproduced from the frozen source SMILES with the lipo_v1 "
            "vocabulary of tracks/structure-contribution/code/bjoc_lipophilicity_adapter_v1.py; "
            "identical convention to lcrc_molnet_benchmark_pilot_v1/analysis/source/raw_attributes_v1.json.gz"
        ),
        "datasets": {},
    }
    for key in C.DATASET_ORDER:
        arrays, audit = _encode_dataset(key)
        path = artifact_path(key)
        np.savez_compressed(path, **arrays)
        manifest["datasets"][key] = {
            **audit,
            "path": C.track_rel(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": C.sha256_file(path),
            "rows_digest_sha256": row_digest(key, arrays),
        }
        C.log(f"[artifact] {key}: {audit['n_rows']} rows, {path.stat().st_size/1e6:.2f} MB")
    C.write_json(MANIFEST, manifest)
    total_mb = sum(v["size_bytes"] for v in manifest["datasets"].values()) / 1e6
    C.log(f"wrote {C.track_rel(MANIFEST)} ({total_mb:.2f} MB total)")
    return manifest


_CACHE: dict[str, dict[str, np.ndarray]] = {}


def load_arrays(dataset: str, *, verify: bool = True) -> dict[str, np.ndarray]:
    if dataset in _CACHE:
        return _CACHE[dataset]
    path = artifact_path(dataset)
    if not path.exists() or not MANIFEST.exists():
        raise FileNotFoundError(f"raw artifact missing for {dataset}; run {Path(__file__).name}")
    manifest = C.read_json(MANIFEST)
    if verify:
        actual = C.sha256_file(path)
        recorded = manifest["datasets"][dataset]["sha256"]
        if actual != recorded:
            raise ValueError(f"raw artifact sha256 {actual} != manifest {recorded} for {dataset}")
    with np.load(path, allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    if manifest["datasets"][dataset]["n_rows"] != int(arrays["mol_index"].size):
        raise ValueError(f"{dataset}: artifact row count differs from the manifest")
    if not np.array_equal(arrays["mol_index"], np.arange(arrays["mol_index"].size, dtype=np.int32)):
        raise ValueError(f"{dataset}: mol_index is not the positional row id")
    for key in ("atom_ptr", "bond_ptr"):
        if arrays[key].shape != (arrays["mol_index"].size + 1,):
            raise ValueError(f"{dataset}: {key} has shape {arrays[key].shape}")
    _CACHE[dataset] = arrays
    return arrays


def load_rows(dataset: str, rows: Sequence[int], *, verify: bool = True) -> list[dict[str, Any]]:
    """Raw attribute rows for the requested global row ids, in the v1 field names."""
    arrs = load_arrays(dataset, verify=verify)
    out: list[dict[str, Any]] = []
    for row in rows:
        r = int(row)
        if r < 0 or r >= int(arrs["mol_index"].size):
            raise IndexError(f"{dataset}: row {r} outside the raw artifact")
        a0, a1 = int(arrs["atom_ptr"][r]), int(arrs["atom_ptr"][r + 1])
        b0, b1 = int(arrs["bond_ptr"][r]), int(arrs["bond_ptr"][r + 1])
        out.append(
            {
                "mol_index": r,
                "smiles": str(arrs["smiles"][r]),
                "canonical_smiles": str(arrs["canonical_smiles"][r]),
                "atom_class": arrs["atom_class"][a0:a1],
                "atom_charge": arrs["atom_charge"][a0:a1],
                "atom_h": arrs["atom_h"][a0:a1],
                "edge_index": np.stack([arrs["edge_src"][b0:b1], arrs["edge_dst"][b0:b1]]).astype(np.int64)
                if b1 > b0
                else np.zeros((2, 0), dtype=np.int64),
                "edge_class": arrs["edge_class"][b0:b1],
                "edge_stereo": arrs["edge_stereo"][b0:b1],
                "n_components": int(arrs["n_components"][r]),
            }
        )
    return out


def check() -> dict[str, Any]:
    manifest = C.read_json(MANIFEST)
    out: dict[str, Any] = {}
    for key in C.DATASET_ORDER:
        _CACHE.pop(key, None)
        actual_size = artifact_path(key).stat().st_size
        actual_sha = C.sha256_file(artifact_path(key))
        digest = row_digest(key)
        recorded = manifest["datasets"][key]
        out[key] = {
            "n_rows": recorded["n_rows"],
            "sha256_matches": actual_sha == recorded["sha256"],
            "size_matches": actual_size == recorded["size_bytes"],
            "rows_digest_matches": digest == recorded["rows_digest_sha256"],
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="raw attribute artifact builder (v2)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        report = check()
        for key, item in report.items():
            C.log(f"[artifact] {key}: {item}")
        if not all(all(v for k, v in item.items() if k.endswith("matches")) for item in report.values()):
            raise SystemExit("artifact verification failed")
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
