"""Structure caches, Fit-only normalisation and *auditable* label access.

Four label-free structure caches are built from the frozen raw artifact for the
four partitions of :mod:`edit_split`.  Every cache keeps the global ChemBench row
id of each molecule, so all later joins are on ``row_id`` and never on position.

Label scope is enforced here, not by convention:

* ``fit`` / ``select``  - available for training and checkpoint selection.
* ``diagnose``          - available only for probe fitting (``purpose="probe_fit"``).
* ``audit``             - available only after ``analysis/probe_freeze.json``
  exists and only once (``purpose="audit_score"``); every attempt is appended to
  ``analysis/label_access_ledger.json``.

Normalisation (both ``g``/``atom_scalar`` statistics and the target mean/std) is
fitted on **Fit only**.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import edit_common as C
import dissect_data as D
import dissect_source as S

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
import mb_raw_artifact as RAW
import mb_rdkit as R
from feature_spec import FeatureSpec

CACHE_VERSION = "lcrc-local-edit-diagnosis-v1"
LEDGER = C.ANALYSIS / "label_access_ledger.json"
PROBE_FREEZE = C.ANALYSIS / "probe_freeze.json"

#: purpose -> partitions whose labels that purpose may read
PURPOSE_SCOPE = {
    "train": ("fit",),
    "select": ("fit", "select"),
    "probe_fit": ("fit", "select", "diagnose"),
    "audit_score": ("fit", "select", "diagnose", "audit"),
}


# ---------------------------------------------------------------------------
# frozen spec
# ---------------------------------------------------------------------------


def frozen_spec() -> FeatureSpec:
    spec = D.frozen_spec()
    if spec.hash() != C.FEATURE_SPEC_HASH:
        raise ValueError(f"lipo_v1 spec hash {spec.hash()} != frozen {C.FEATURE_SPEC_HASH}")
    return spec


# ---------------------------------------------------------------------------
# partition rows
# ---------------------------------------------------------------------------


def split_payload() -> dict[str, Any]:
    payload = C.read_json(C.ANALYSIS / "component_split.json")
    if payload.get("status") != "OK":
        raise SystemExit(f"component split status is {payload.get('status')!r}; refusing to build caches")
    return payload


def partition_rows(partition: str) -> list[int]:
    payload = split_payload()
    rows = [int(r) for r in payload["rows_by_partition"][partition]]
    if len(set(rows)) != len(rows):
        raise ValueError(f"partition {partition} has duplicate rows")
    return rows


def all_partition_rows() -> dict[str, list[int]]:
    payload = split_payload()
    return {p: [int(r) for r in payload["rows_by_partition"][p]] for p in C.PARTITIONS}


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


def cache_path(partition: str) -> Path:
    return C.CACHE / f"Lipophilicity_edit_{partition}.pt"


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_partition_cache(partition: str, spec: FeatureSpec, *, save: bool = True) -> dict[str, Any]:
    """Same structure pipeline as the dissection round, on a partition row list.

    Re-implements the (upstream) ``build_split_cache`` body because that helper
    hard-codes the three public split names; the structure semantics, the raw
    artifact verification and the cache validation are the frozen ones.
    """
    if partition not in C.PARTITIONS:
        raise ValueError(partition)
    rows = partition_rows(partition)
    src = S.load_lipop(C.DATA_REPEAT, labels="none")
    payload = RAW.load(verify=True)
    all_rows = payload["rows"]["Lipophilicity"]
    raw_rows = []
    for row in rows:
        item = all_rows[int(row)]
        if int(item["mol_index"]) != int(row):
            raise ValueError(f"raw artifact row {row} has mol_index {item['mol_index']}")
        if str(item["smiles"]) != str(src["smiles"][int(row)]):
            raise ValueError(f"partition {partition}: artifact SMILES for row {row} differs from the CSV")
        raw_rows.append(item)
    graphs = [R.raw_row_to_graph(raw, int(raw["mol_index"]), spec) for raw in raw_rows]
    cache = PREP_DATA.build_cache(graphs, spec, split=f"Lipophilicity_edit_{partition}")
    cache["rows"] = np.asarray(rows, dtype=np.int64)
    cache["dataset"] = "Lipophilicity"
    cache["split"] = f"Lipophilicity_edit_{partition}"
    cache["split_name"] = partition
    cache["repeat"] = int(C.DATA_REPEAT)
    cache["cache_version"] = CACHE_VERSION
    cache["source_commit"] = C.DC.CHEMBENCH_COMMIT
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "global 0-based ChemBench CSV row index"
    PREP_DATA.validate_cache(cache, spec)
    if cache["molecule_ids"] != [str(r) for r in rows]:
        raise ValueError(f"partition {partition}: molecule_ids do not match source row ids")
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    if save:
        _atomic_save(cache, cache_path(partition))
    return cache


def load_cache(partition: str, *, rebuild: bool = False) -> dict[str, Any]:
    spec = frozen_spec()
    path = cache_path(partition)
    rows = partition_rows(partition)
    if path.exists() and not rebuild:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{path}: unexpected cache version {cache.get('cache_version')!r}")
        if [int(x) for x in np.asarray(cache["rows"]).tolist()] != rows:
            raise ValueError(f"{path}: cached row ids differ from the frozen partition")
        if cache["meta"].get("feature_spec_hash") != spec.hash():
            raise ValueError(f"{path}: cached feature spec hash differs")
        if cache["meta"].get("labels_present"):
            raise ValueError(f"{path}: structure cache must be label-free")
        return cache
    return build_partition_cache(partition, spec)


def prepare_all(rebuild: bool = False) -> dict[str, Any]:
    """Build/verify the four label-free caches and the Fit-only normalisation."""
    spec = frozen_spec()
    entries: dict[str, Any] = {}
    for partition in C.PARTITIONS:
        cache = load_cache(partition, rebuild=rebuild)
        entries[partition] = {
            "n_molecules": int(cache["n_molecules"]),
            "rows_sha256": C.row_sha256(np.asarray(cache["rows"]).tolist()),
            "content_fingerprint": cache["meta"]["content_fingerprint"],
            "cache_path": C.track_rel(cache_path(partition)),
            "cache_sha256": C.sha256_file(cache_path(partition)) if cache_path(partition).exists() else None,
            "labels_present": bool(cache["meta"]["labels_present"]),
        }
        C.log(
            f"[cache] {partition}: n={cache['n_molecules']} "
            f"fp={cache['meta']['content_fingerprint'][:16]}"
        )
    payload = {
        "schema": "lcrc-local-edit-cache-manifest-v1",
        "created_utc": C.utcnow(),
        "feature_spec_hash": spec.hash(),
        "label_free": True,
        "partitions": entries,
        "fit_norm": norm_payload(),
    }
    C.write_json(C.CACHE / "manifest.json", payload)
    return payload


# ---------------------------------------------------------------------------
# normalisation (Fit only)
# ---------------------------------------------------------------------------


def train_norm() -> dict[str, Any]:
    cache = load_cache("fit")
    spec = frozen_spec()
    norm = PREP_DATA.fit_norm(cache, spec, source="Lipophilicity_edit_fit_structures_only")
    if norm.get("valid_or_test_used") is not False:
        raise ValueError("structure normalisation must be Fit-only")
    return norm


def _target_normaliser(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    scale = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate Fit target distribution")
    return {
        "mean": mean,
        "scale": scale,
        "n": int(y.size),
        "source": "fit_partition_only",
        "diagnose_or_audit_used": False,
        "formula": "z = (y - mean) / scale; pred_raw = mean + scale * z_hat",
    }


def target_norm() -> dict[str, Any]:
    return _target_normaliser(labels("fit")["y"])


def norm_payload() -> dict[str, Any]:
    return {
        "structure": train_norm(),
        "target": target_norm(),
        "fingerprint": PREP_DATA.norm_fingerprint(train_norm()),
    }


# ---------------------------------------------------------------------------
# label access
# ---------------------------------------------------------------------------


def _read_ledger() -> dict[str, Any]:
    if LEDGER.exists():
        return C.read_json(LEDGER)
    return {"schema": "lcrc-local-edit-label-access-v1", "events": []}


def _append_ledger(event: Mapping[str, Any]) -> None:
    payload = _read_ledger()
    payload.setdefault("events", []).append(dict(event))
    C.write_json(LEDGER, payload)


def audit_labels_accessed() -> bool:
    payload = _read_ledger()
    return any(e.get("partition") == "audit" and e.get("purpose") == "audit_score" for e in payload["events"])


def labels(partition: str, purpose: str = "train") -> dict[str, Any]:
    """Targets for one partition, under the frozen purpose scope.

    Refuses (a) a purpose that does not cover the partition, (b) any Audit read
    before ``analysis/probe_freeze.json`` exists, and (c) a second Audit read.
    """
    if purpose not in PURPOSE_SCOPE:
        raise PermissionError(f"unknown label-access purpose {purpose!r}")
    if partition not in PURPOSE_SCOPE[purpose]:
        raise PermissionError(f"purpose {purpose!r} may not read {partition!r} labels")
    if partition == "audit":
        if not PROBE_FREEZE.exists():
            raise PermissionError("Audit labels may only be read after analysis/probe_freeze.json exists")
        if audit_labels_accessed():
            raise PermissionError("Audit labels were already read once; the round does not allow a second read")
    src = S.load_lipop(C.DATA_REPEAT, labels="train_valid")
    rows = partition_rows(partition)
    train_rows = {int(r): i for i, r in enumerate(src["rows"]["train"])}
    if any(int(r) not in train_rows for r in rows):
        raise ValueError(f"partition {partition} contains rows outside the published Train part")
    raw = np.asarray(src["y"]["train"], dtype=np.float64)
    y = np.asarray([raw[train_rows[int(r)]] for r in rows], dtype=np.float64)
    event = {
        "utc": C.utcnow(),
        "partition": partition,
        "purpose": purpose,
        "rows_sha256": C.row_sha256(rows),
        "n": len(rows),
        "git_commit": C.git_commit(),
        "member_of_train_part": True,
        "source_label_scope": "train_valid",
        "test_targets_decoded": False,
        "random122_test_or_reserved_used": False,
    }
    _append_ledger(event)
    return {"rows": rows, "y": y, "purpose": purpose, "event": event}


def y_aligned(partition: str, rows: Sequence[int], purpose: str) -> np.ndarray:
    """Targets for an explicit row order; refuses any row outside the partition."""
    data = labels(partition, purpose)
    index = {int(r): float(v) for r, v in zip(data["rows"], data["y"])}
    missing = [int(r) for r in rows if int(r) not in index]
    if missing:
        raise KeyError(f"rows {missing[:5]} are not in partition {partition}; refusing to substitute")
    return np.asarray([index[int(r)] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# prediction helpers (delegate to the frozen collate/model path)
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return D.to_device(batch, device)


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return PREP_DATA.collate(cache, list(rows))


def predict_scores(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    return D.predict_scores(model, cache, device, batch_size=batch_size)


__all__ = [
    "CACHE_VERSION",
    "LEDGER",
    "PROBE_FREEZE",
    "PURPOSE_SCOPE",
    "frozen_spec",
    "split_payload",
    "partition_rows",
    "all_partition_rows",
    "cache_path",
    "build_partition_cache",
    "load_cache",
    "prepare_all",
    "train_norm",
    "target_norm",
    "norm_payload",
    "labels",
    "y_aligned",
    "audit_labels_accessed",
    "to_device",
    "collate",
    "predict_scores",
]
