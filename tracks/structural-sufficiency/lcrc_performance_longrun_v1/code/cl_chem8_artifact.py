"""Build the committed label-free CHEMLOCAL8 channel artefact.

Usage
-----
    uv run python code/cl_chem8_artifact.py                 # all seven datasets
    uv run python code/cl_chem8_artifact.py --dataset ESOL  # one dataset
    uv run python code/cl_chem8_artifact.py --check         # verify hashes only

The artefact is one ``<dataset>.npz`` per dataset plus
``analysis/chem8_manifest.json``.  Nothing is fitted and no label is read: the
SMILES come from the frozen source loader, and the eight channels are
deterministic RDKit functions of the molecular graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import cl_common as C  # noqa: I001
import cl_chem8 as CH


def _write_manifest(entries: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "lcrc-chem8-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "label_free": True,
        "channel_names": list(CH.CHANNEL_NAMES),
        "definition": {
            "CIP_R": "FindMolChiralCenters -> 'R'",
            "CIP_S": "FindMolChiralCenters -> 'S'",
            "CIP_UNASSIGNED_TETRA": "FindMolChiralCenters(includeUnassigned=True) -> '?'",
            "HBD": "rdkit.Chem.Lipinski._HDonors SMARTS, first matched atom",
            "HBA": "rdkit.Chem.Lipinski._HAcceptors SMARTS, first matched atom",
            "HYB_SP": "RDKit GetHybridization() == SP",
            "HYB_SP2": "RDKit GetHybridization() == SP2",
            "HYB_SP3": "RDKit GetHybridization() == SP3",
        },
        "datasets": entries,
    }
    C.write_json(C.CHEM8_MANIFEST, payload)
    return payload


def build(datasets: Sequence[str]) -> dict[str, Any]:
    import rdkit

    C.SOURCE_CHEM8.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if C.CHEM8_MANIFEST.exists():
        existing = C.read_json(C.CHEM8_MANIFEST).get("datasets", {})
    entries: dict[str, Any] = dict(existing)
    for dataset in datasets:
        C.log(f"[chem8] building {dataset}")
        arrays, meta = CH.build_artifact(dataset)
        path = CH.artifact_path(dataset)
        np.savez_compressed(path, **arrays)
        meta = {
            **meta,
            "path": C.track_rel(path),
            "sha256": C.sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "rdkit_version": str(rdkit.__version__),
            "array_shapes": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
            "array_dtypes": {k: str(np.asarray(v).dtype) for k, v in arrays.items()},
        }
        entries[dataset] = meta
        C.log(
            f"[chem8] {dataset}: {meta['n_molecules']} molecules / {meta['n_atoms']} atoms, "
            f"{meta['size_bytes'] / 1e6:.2f} MB, fallback={meta['n_fallback']}"
        )
    payload = _write_manifest(entries)
    C.log(f"[chem8] wrote {C.track_rel(C.CHEM8_MANIFEST)}")
    return payload


def check() -> dict[str, Any]:
    manifest = C.read_json(C.CHEM8_MANIFEST)
    out: dict[str, Any] = {}
    for dataset, entry in manifest["datasets"].items():
        path = CH.artifact_path(dataset)
        actual = C.sha256_file(path)
        out[dataset] = {
            "path": C.track_rel(path),
            "recorded_sha256": entry["sha256"],
            "actual_sha256": actual,
            "consistent": bool(actual == entry["sha256"]),
        }
        if not out[dataset]["consistent"]:
            raise SystemExit(f"{dataset}: chem8 artefact changed after the manifest was written")
    print(out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build / verify the label-free chem8 artefact")
    parser.add_argument("--dataset", action="append", choices=list(C.V2C.DATASET_ORDER), default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        return 0
    datasets = args.dataset or list(C.V2C.DATASET_ORDER)
    build(datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
