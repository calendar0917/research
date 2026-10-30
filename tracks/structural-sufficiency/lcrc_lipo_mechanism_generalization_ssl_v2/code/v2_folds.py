"""Deterministic three-fold scaffold split of Fit2352 for internal CV.

The scaffold definition intentionally reuses the Phase 1a rule:

* RDKit canonical SMILES;
* Bemis--Murcko scaffold with ``includeChirality=False``;
* an empty scaffold is replaced by the canonical SMILES as its own group key.

The greedy assignment is fixed before any model is trained:

1. sort groups by molecule count descending;
2. break equal-size ties by ``sha256("lcrc-lipo-v2|" + group_key)`` ascending;
3. assign each whole group to the fold with the fewest molecules so far;
4. break equal-size fold ties by the smaller fold id.

The fold file stores global row ids, Fit cache positions, group keys, canonical
SMILES hashes and SHA-256 fingerprints.  It contains no labels.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import v2_common as C

import adapters as prep_adapters  # noqa: E402  (provided by the frozen prep sys.path)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_and_scaffold(smiles: str) -> tuple[str, str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse frozen SMILES {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    scaffold = str(MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return canonical, scaffold


def _fit_rows_and_positions() -> tuple[list[int], dict[int, int]]:
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    if len(fit_rows) != C.SPLIT_SIZES["fit"] or len(set(fit_rows)) != len(fit_rows):
        raise ValueError("frozen Fit split is not 2352 unique rows")
    return fit_rows, {row: pos for pos, row in enumerate(fit_rows)}


def _load_structures(fit_rows: Sequence[int]) -> tuple[dict[int, str], dict[int, str], dict[str, Any]]:
    """Return canonical SMILES and scaffolds for the requested rows."""
    C.verify_sha256(C.v1c.LIPO_RAW, C.v1c.EXPECTED_SHA256["lipo_raw"], "lipo_raw")
    C.verify_sha256(C.v1c.LIPO_ROW_MANIFEST, C.v1c.EXPECTED_SHA256["row_manifest"], "row_manifest")
    raw = prep_adapters.load_lipo_raw_artifact(C.v1c.LIPO_RAW)
    manifest = prep_adapters.load_lipo_row_manifest(C.v1c.LIPO_ROW_MANIFEST)
    canonical: dict[int, str] = {}
    scaffolds: dict[int, str] = {}
    for row in fit_rows:
        record = raw["rows"][int(row)]
        if "exp" in record:
            raise PermissionError(f"raw row {row} unexpectedly contains a label")
        can, scaffold = _canonical_and_scaffold(str(record["smiles"]))
        expected = manifest["canonical_smiles_sha256"][int(row)]
        actual = _sha256(can)
        if actual != expected:
            raise ValueError(f"row {row}: canonical SMILES hash {actual} != frozen {expected}")
        canonical[int(row)] = can
        scaffolds[int(row)] = scaffold
    if len(set(canonical.values())) != len(canonical):
        raise ValueError("derived canonical SMILES are not unique across Fit2352")
    return canonical, scaffolds, {
        "raw_sha256": C.sha256_file(C.v1c.LIPO_RAW),
        "row_manifest_sha256": C.sha256_file(C.v1c.LIPO_ROW_MANIFEST),
        "canonical_smiles_verified": True,
        "n_rows": len(fit_rows),
    }


def _group_key(row: int, canonical: Mapping[int, str], scaffolds: Mapping[int, str]) -> tuple[str, str]:
    scaffold = str(scaffolds[int(row)])
    if scaffold.strip():
        return scaffold, "bemis_murcko"
    return str(canonical[int(row)]), "canonical_smiles_fallback"


def build_folds() -> dict[str, Any]:
    fit_rows, cache_pos = _fit_rows_and_positions()
    canonical, scaffolds, source_audit = _load_structures(fit_rows)
    groups: dict[str, list[int]] = defaultdict(list)
    types: dict[str, str] = {}
    for row in fit_rows:
        key, kind = _group_key(row, canonical, scaffolds)
        groups[key].append(int(row))
        if key in types and types[key] != kind:
            raise ValueError(f"group key type conflict for {key!r}")
        types[key] = kind

    ordered = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), _sha256("lcrc-lipo-v2|" + kv[0])),
    )
    fold_rows: dict[int, list[int]] = {f: [] for f in C.FOLDS}
    fold_groups: dict[int, list[str]] = {f: [] for f in C.FOLDS}
    for key, rows in ordered:
        sizes = {f: len(fold_rows[f]) for f in C.FOLDS}
        target = min(C.FOLDS, key=lambda f: (sizes[f], f))
        fold_rows[target].extend(int(r) for r in rows)
        fold_groups[target].append(key)
    for f in C.FOLDS:
        fold_rows[f] = sorted(fold_rows[f])
        fold_groups[f] = sorted(fold_groups[f])

    if sorted(sum(fold_rows.values(), [])) != fit_rows:
        raise ValueError("fold assignment does not partition Fit2352")
    seen_groups: dict[str, int] = {}
    for f in C.FOLDS:
        for key in fold_groups[f]:
            if key in seen_groups:
                raise ValueError(f"group {key!r} appears in folds {seen_groups[key]} and {f}")
            seen_groups[key] = f

    per_row: dict[str, Any] = {}
    fold_payload = []
    for f in C.FOLDS:
        frows = fold_rows[f]
        per_row_groups = {}
        for row in frows:
            key, kind = _group_key(row, canonical, scaffolds)
            per_row_groups[str(row)] = {
                "fold": f,
                "global_row_id": int(row),
                "fit_cache_position": int(cache_pos[int(row)]),
                "canonical_smiles_sha256": _sha256(canonical[int(row)]),
                "canonical_smiles": canonical[int(row)],
                "scaffold": scaffolds[int(row)],
                "group_key_sha256": _sha256("lcrc-lipo-v2|" + key),
                "group_key_type": kind,
            }
        fold_payload.append(
            {
                "fold": f,
                "n_molecules": len(frows),
                "n_scaffold_groups": len(fold_groups[f]),
                "rows_global": frows,
                "rows_cache_positions": [int(cache_pos[r]) for r in frows],
                "row_sha256_int64": C.row_sha256_bytes(frows),
                "row_sha256_text": C.row_sha256_text(frows),
                "canonical_smiles_sha256_list": hashlib.sha256(
                    "\n".join(per_row_groups[str(r)]["canonical_smiles_sha256"] for r in frows).encode()
                ).hexdigest(),
                "groups": [
                    {
                        "group_key": key,
                        "group_key_type": types[key],
                        "group_key_sha256": _sha256("lcrc-lipo-v2|" + key),
                        "n_molecules": len(groups[key]),
                        "rows_global": sorted(int(r) for r in groups[key]),
                    }
                    for key in fold_groups[f]
                ],
            }
        )
        per_row.update(per_row_groups)

    if len(per_row) != C.SPLIT_SIZES["fit"]:
        raise ValueError("per-row fold map does not cover Fit2352")
    # Canonical SMILES must not cross folds.
    smiles_folds: dict[str, set[int]] = defaultdict(set)
    for row, info in per_row.items():
        smiles_folds[info["canonical_smiles_sha256"]].add(int(info["fold"]))
    if any(len(v) > 1 for v in smiles_folds.values()):
        raise ValueError("a canonical SMILES appears in more than one fold")

    return {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "FROZEN_FIT_ONLY_SCAFFOLD_CV",
        "scope": {
            "labels_read": False,
            "monitor_structures_used": False,
            "reserved_structures_used": False,
            "official_valid_test_used": False,
            "fit_only": True,
        },
        "algorithm": {
            "scaffold": "RDKit MurckoScaffoldSmiles(includeChirality=False)",
            "canonical_smiles": "RDKit MolToSmiles(canonical=True)",
            "fallback": "canonical SMILES as group key when the Murcko scaffold is empty",
            "group_order": "size desc, sha256('lcrc-lipo-v2|' + group_key) asc",
            "assignment": "whole group -> fold with fewest molecules; ties smaller fold id",
        },
        "source": source_audit,
        "fit_rows_global": fit_rows,
        "fit_rows_cache_positions": list(range(C.SPLIT_SIZES["fit"])),
        "fit_row_sha256_int64": C.row_sha256_bytes(fit_rows),
        "fit_row_sha256_text": C.row_sha256_text(fit_rows),
        "n_groups_total": len(groups),
        "folds": fold_payload,
        "rows": per_row,
    }


def write_folds() -> dict[str, Any]:
    payload = build_folds()
    C.write_json(C.FOLD_FILE, payload)
    return payload


def load_folds(*, verify_hash: bool = True) -> dict[str, Any]:
    if not C.FOLD_FILE.exists():
        raise FileNotFoundError(f"missing {C.track_rel(C.FOLD_FILE)}; run `run_v2.py folds`")
    payload = C.read_json(C.FOLD_FILE)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise ValueError("fold file protocol id mismatch")
    if payload.get("status") != "FROZEN_FIT_ONLY_SCAFFOLD_CV":
        raise ValueError("fold file is not the frozen fit-only scaffold CV")
    scope = payload.get("scope", {})
    if scope.get("labels_read") is not False or scope.get("monitor_structures_used") is not False:
        raise PermissionError("fold file is not certified label/Monitor-free")
    fit_rows, cache_pos = _fit_rows_and_positions()
    if payload.get("fit_rows_global") != fit_rows:
        raise ValueError("fold file Fit row order does not match the frozen split")
    seen: set[int] = set()
    for f in budget_folds(payload):
        rows = [int(r) for r in f["rows_global"]]
        if len(rows) != f["n_molecules"]:
            raise ValueError("fold molecule count mismatch")
        if set(rows) & seen:
            raise ValueError("fold row overlap")
        seen.update(rows)
        if rows != sorted(rows):
            raise ValueError("fold rows are not sorted")
        if f["rows_cache_positions"] != [cache_pos[r] for r in rows]:
            raise ValueError("fold cache positions do not match the frozen Fit order")
        if C.row_sha256_bytes(rows) != f["row_sha256_int64"]:
            raise ValueError("fold row hash mismatch")
    if seen != set(fit_rows):
        raise ValueError("folds do not partition Fit2352")
    if verify_hash:
        # The exact file hash is recorded by the protocol freeze; here we only
        # re-derive the structural invariants.
        pass
    return payload


def budget_folds(payload: Mapping[str, Any]):
    folds = list(payload.get("folds", []))
    if [int(f["fold"]) for f in folds] != list(C.FOLDS):
        raise ValueError("fold ids are not 0,1,2 in order")
    return folds


def fold_rows(payload: Mapping[str, Any], fold: int) -> list[int]:
    for f in payload["folds"]:
        if int(f["fold"]) == int(fold):
            return [int(r) for r in f["rows_global"]]
    raise KeyError(fold)


def fold_positions(payload: Mapping[str, Any], fold: int) -> list[int]:
    for f in payload["folds"]:
        if int(f["fold"]) == int(fold):
            return [int(p) for p in f["rows_cache_positions"]]
    raise KeyError(fold)


def train_positions(payload: Mapping[str, Any], fold: int) -> list[int]:
    dev = set(fold_positions(payload, fold))
    return [p for p in range(C.SPLIT_SIZES["fit"]) if p not in dev]


def fold_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_groups_total": int(payload["n_groups_total"]),
        "folds": [
            {
                "fold": int(f["fold"]),
                "n_molecules": int(f["n_molecules"]),
                "n_scaffold_groups": int(f["n_scaffold_groups"]),
                "row_sha256_int64": f["row_sha256_int64"],
                "row_sha256_text": f["row_sha256_text"],
                "rows_global": f["rows_global"],
                "rows_cache_positions": f["rows_cache_positions"],
                "canonical_smiles_sha256_list": f["canonical_smiles_sha256_list"],
            }
            for f in payload["folds"]
        ],
    }


__all__ = [
    "build_folds",
    "write_folds",
    "load_folds",
    "fold_rows",
    "fold_positions",
    "train_positions",
    "fold_summary",
]
