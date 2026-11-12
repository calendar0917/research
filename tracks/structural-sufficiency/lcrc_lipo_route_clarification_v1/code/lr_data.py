"""Split payloads, structure caches, Fit-only normalisation and label access.

The round has five *split systems* over the same 3360 published Train rows:

``s0``  the frozen upstream Fit2016/Select336/Diagnose504/Audit504 split
        (read-only copy of ``lcrc_local_edit_error_diagnosis_v1``);
``s1``  ``lipo-route-v1-split-20261112-1``;
``s2``  ``lipo-route-v1-split-20261112-2``;
``sg``  the Bemis-Murcko joint-scaffold pressure split;
``ft``  the full-Train development stage (Fit = all 3360 Train rows,
        Select = the published 420-row Validation; the published Test rows are
        never read, never cached and never scored).

Label discipline is enforced here:

* every access declares ``purpose`` and is checked against
  :data:`lr_common.PURPOSE_SCOPE`;
* a partition's labels are only decoded for rows inside that partition;
* Audit labels are refused until ``analysis/AUDIT_FREEZE.json`` exists and are
  then readable exactly once;
* every access appends to ``analysis/label_access_ledger.json``.

Normalisation (structure statistics and target mean/std) is fitted on the
*system's own Fit partition only*.
"""

from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import lr_common as C

import dissect_data as D
import dissect_source as S
import mb_raw_artifact as RAW
import mb_rdkit as R

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA

CACHE_VERSION = "lcrc-lipo-route-clarification-v1"
CHEMBENCH_COMMIT = "0b10d4c1ac72b6929503694a5f60446824736391"
LEDGER = C.ANALYSIS / "label_access_ledger.json"
AUDIT_FREEZE = C.ANALYSIS / "AUDIT_FREEZE.json"

#: split -> source of the row allocation
SPLIT_FILES = {
    "s0": C.ANALYSIS / "split_s0.json",
    "s1": C.ANALYSIS / "split_s1.json",
    "s2": C.ANALYSIS / "split_s2.json",
    "sg": C.ANALYSIS / "split_sg.json",
    "ft": C.ANALYSIS / "split_ft.json",
}


# ---------------------------------------------------------------------------
# frozen spec and raw source
# ---------------------------------------------------------------------------


def frozen_spec():
    return D.frozen_spec()


def source() -> dict[str, Any]:
    """SMILES + published split row ids (no target byte is decoded here)."""
    src = S.load_lipop(C.DATA_REPEAT, labels="none")
    if C.row_sha256(src["rows"]["train"]) != C.TRAIN_ROWS_SHA256:
        raise ValueError("random122 Train row sequence hash differs from the frozen value")
    return src


def read_targets(rows: Sequence[int]) -> np.ndarray:
    """Decode ``exp`` for exactly ``rows`` (and no other row) from the frozen CSV.

    The CSV *bytes* are read (the file is the only source of both ``smiles`` and
    ``exp``), but a target value is float-converted only for a permitted row.
    """
    wanted = {int(r): i for i, r in enumerate(rows)}
    if len(wanted) != len(rows):
        raise ValueError("duplicate row in target request")
    out = np.full(len(rows), np.nan, dtype=np.float64)
    path = C.PILOT / "analysis/source/chembench/Lipop/Lipop.csv.gz"
    if C.sha256_file(path) != C.LIPOP_CSV_SHA256:
        raise ValueError("Lipop.csv.gz sha256 differs from the frozen value")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header[:2] != ["smiles", "exp"]:
            raise ValueError(f"unexpected CSV header {header!r}")
        seen = 0
        for rec in reader:
            pos = wanted.get(seen)
            if pos is not None:
                out[pos] = float(rec[1])
            seen += 1
    if seen != 4200:
        raise ValueError(f"CSV has {seen} data rows != 4200")
    if not np.isfinite(out).all():
        raise ValueError("some requested rows were not found in the CSV")
    return out


# ---------------------------------------------------------------------------
# split payloads
# ---------------------------------------------------------------------------


def split_payload(split: str) -> dict[str, Any]:
    path = SPLIT_FILES[split]
    if not path.exists():
        raise SystemExit(f"split payload {path} is missing; run lr_split.py first")
    payload = C.read_json(path)
    if payload.get("split") != split:
        raise ValueError(f"{path}: split id {payload.get('split')!r} != {split!r}")
    if payload.get("status") not in ("OK", "LOW_SUPPORT", "SG_SCAFFOLD_ONLY", "SG_JOINT_GROUPS", "FEASIBLE"):
        raise ValueError(f"{path}: unexpected status {payload.get('status')!r}")
    return payload


def rows_by_partition(split: str) -> dict[str, list[int]]:
    payload = split_payload(split)
    out = {p: [int(r) for r in payload["rows_by_partition"].get(p, [])] for p in C.PARTITIONS}
    if split != "ft":
        all_rows = sorted(r for p in C.PARTITIONS for r in out[p])
        if len(all_rows) != C.TRAIN_ROWS or len(set(all_rows)) != C.TRAIN_ROWS:
            raise ValueError(f"split {split}: partitions do not cover the 3360 Train rows exactly once")
    return out


def global_components() -> dict[int, int]:
    """Split-independent global 分子-匹配核心 component id for every Train row."""
    payload = split_payload("s0")
    mapping = {int(k): int(v) for k, v in payload["row_to_component"].items()}
    if len(mapping) != C.TRAIN_ROWS:
        raise ValueError("global component map does not cover the 3360 Train rows")
    return mapping


def component_key(split: str, partition: str) -> dict[int, str]:
    """Component label used for component-equal-weight pair metrics."""
    payload = split_payload(split)
    units = payload.get("units_per_partition", {}).get(partition)
    if not units:
        return {}
    out: dict[int, str] = {}
    for unit_rows in units:
        key = C.stable_id(unit_rows)
        for r in unit_rows:
            out[int(r)] = key
    return out


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


def cache_path(split: str, partition: str) -> Path:
    return C.CACHE / f"lipo_{split}_{partition}.pt"


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _graph_for(raw: Mapping[str, Any], row: int, spec, smiles: Sequence[str]):
    if int(raw["mol_index"]) != int(row):
        raise ValueError(f"raw artifact row {row} has mol_index {raw['mol_index']}")
    if str(raw["smiles"]) != str(smiles[int(row)]):
        raise ValueError(f"artifact SMILES for row {row} differs from the CSV")
    return R.raw_row_to_graph(raw, int(row), spec)


def _cache_from_graphs(
    graphs: Sequence[Any], rows: Sequence[int], spec, split_name: str
) -> dict[str, Any]:
    cache = PREP_DATA.build_cache(list(graphs), spec, split=split_name)
    cache["rows"] = np.asarray([int(r) for r in rows], dtype=np.int64)
    cache["dataset"] = "Lipophilicity"
    cache["split"] = split_name
    cache["repeat"] = int(C.DATA_REPEAT)
    cache["cache_version"] = CACHE_VERSION
    cache["source_commit"] = CHEMBENCH_COMMIT
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "global 0-based ChemBench CSV row index"
    PREP_DATA.validate_cache(cache, spec)
    if cache["molecule_ids"] != [str(int(r)) for r in rows]:
        raise ValueError(f"{split_name}: molecule_ids do not match source row ids")
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    return cache


def build_all(rebuild: bool = False, splits: Sequence[str] = C.SPLITS + ("ft",)) -> dict[str, Any]:
    """Build every (split, partition) structure cache from the frozen artifact.

    All graphs of the published Train + Validation rows are constructed once and
    then sliced per partition, so the structure pipeline is applied exactly once
    per molecule even though the rows are re-used by five split systems.
    """
    spec = frozen_spec()
    src = source()
    missing = [s for s in splits for p in C.PARTITIONS
               if (rows_by_partition(s).get(p) or []) and (rebuild or not cache_path(s, p).exists())]
    graphs: dict[int, Any] = {}
    if missing:
        valid_rows = [int(r) for r in src["rows"]["valid"]]
        train_rows = [int(r) for r in src["rows"]["train"]]
        payload = RAW.load(verify=True)
        all_rows = payload["rows"]["Lipophilicity"]
        t0 = __import__("time").perf_counter()
        for row in train_rows + valid_rows:
            graphs[int(row)] = _graph_for(all_rows[int(row)], int(row), spec, src["smiles"])
        C.log(f"[cache] rebuilt {len(graphs)} molecular graphs in {__import__('time').perf_counter() - t0:.1f}s")
    else:
        C.log("[cache] all split caches present; no graph rebuild needed")

    manifest: dict[str, Any] = {"schema": "lcrc-lipo-route-cache-manifest-v1", "created_utc": C.utcnow(),
                                "label_free": True, "feature_spec_hash": spec.hash(), "partitions": {}}
    for split in splits:
        rows = rows_by_partition(split)
        manifest["partitions"][split] = {}
        for partition in C.PARTITIONS:
            part_rows = rows.get(partition) or []
            if not part_rows:
                continue
            path = cache_path(split, partition)
            if path.exists() and not rebuild:
                cache = torch.load(path, map_location="cpu", weights_only=False)
                if cache.get("cache_version") != CACHE_VERSION:
                    raise ValueError(f"{path}: unexpected cache version")
                if [int(x) for x in np.asarray(cache["rows"]).tolist()] != part_rows:
                    raise ValueError(f"{path}: cached rows differ from the frozen split")
                if cache["meta"].get("feature_spec_hash") != spec.hash():
                    raise ValueError(f"{path}: feature spec hash differs")
                if cache["meta"].get("labels_present"):
                    raise ValueError(f"{path}: structure cache must be label-free")
            else:
                cache = _cache_from_graphs([graphs[r] for r in part_rows], part_rows, spec,
                                           f"lipo_{split}_{partition}")
                _atomic_save(cache, path)
            manifest["partitions"][split][partition] = {
                "n_molecules": int(cache["n_molecules"]),
                "rows_sha256": C.row_sha256(part_rows),
                "content_fingerprint": cache["meta"]["content_fingerprint"],
                "path": C.track_rel(path),
                "sha256": C.sha256_file(path),
            }
            C.log(f"[cache] {split}/{partition}: n={cache['n_molecules']} "
                  f"fp={cache['meta']['content_fingerprint'][:12]}")
        manifest["partitions"][split]["norm"] = {
            "structure_fingerprint": PREP_DATA.norm_fingerprint(train_norm(split)),
            "target": target_norm(split),
        }
    C.write_json(C.CACHE / "manifest.json", manifest)
    return manifest


def load_cache(split: str, partition: str) -> dict[str, Any]:
    path = cache_path(split, partition)
    if not path.exists():
        raise SystemExit(f"cache {path} is missing; run lr_prepare.py")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"{path}: unexpected cache version")
    expected = rows_by_partition(split).get(partition) or []
    if [int(x) for x in np.asarray(cache["rows"]).tolist()] != expected:
        raise ValueError(f"{path}: cached rows differ from the frozen split")
    if cache["meta"].get("labels_present"):
        raise ValueError(f"{path}: structure cache must be label-free")
    return cache


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def train_norm(split: str) -> dict[str, Any]:
    cache = load_cache(split, "fit")
    norm = PREP_DATA.fit_norm(cache, frozen_spec(), source=f"lipo_{split}_fit_structures_only")
    if norm.get("valid_or_test_used") is not False:
        raise ValueError("structure normalisation must be Fit-only")
    return norm


def _target_normaliser(y: np.ndarray, source: str) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean, scale = float(y.mean()), float(y.std(ddof=0))
    if not (np.isfinite(mean) and np.isfinite(scale) and scale > 0):
        raise ValueError("degenerate Fit target distribution")
    return {"mean": mean, "scale": scale, "n": int(y.size), "source": source,
            "formula": "z = (y - mean) / scale; pred_raw = mean + scale * z_hat"}


def target_norm(split: str) -> dict[str, Any]:
    rows = rows_by_partition(split)["fit"]
    return _target_normaliser(read_targets(rows), f"{split}_fit_only")


# ---------------------------------------------------------------------------
# label access
# ---------------------------------------------------------------------------


def _read_ledger() -> dict[str, Any]:
    if LEDGER.exists():
        return C.read_json(LEDGER)
    return {"schema": "lcrc-lipo-route-label-access-v1", "events": []}


def _append_ledger(event: Mapping[str, Any]) -> None:
    payload = _read_ledger()
    payload.setdefault("events", []).append(dict(event))
    C.write_json(LEDGER, payload)


def audit_labels_accessed() -> bool:
    payload = _read_ledger()
    return any(e.get("partition") == "audit" and e.get("purpose") == "audit_score" for e in payload["events"])


def labels(split: str, partition: str, purpose: str) -> dict[str, Any]:
    if purpose not in C.PURPOSE_SCOPE:
        raise PermissionError(f"unknown label-access purpose {purpose!r}")
    if partition not in C.PURPOSE_SCOPE[purpose]:
        raise PermissionError(f"purpose {purpose!r} may not read {partition!r} labels")
    if partition == "audit":
        if not AUDIT_FREEZE.exists():
            raise PermissionError("Audit labels may only be read after analysis/AUDIT_FREEZE.json exists")
        if audit_labels_accessed():
            raise PermissionError("Audit labels were already read once")
    rows = rows_by_partition(split)[partition]
    y = read_targets(rows)
    event = {
        "utc": C.utcnow(), "split": split, "partition": partition, "purpose": purpose,
        "rows_sha256": C.row_sha256(rows), "n": len(rows), "git_commit": C.git_commit(),
        "member_of_train_part": split != "ft" or partition != "select",
        "csv_label_bytes_read": True,
        "labels_numerically_converted_only_for_permitted_rows": True,
        "published_test_rows_decoded": False,
    }
    _append_ledger(event)
    return {"rows": rows, "y": y, "purpose": purpose, "event": event}


def y_aligned(split: str, partition: str, rows: Sequence[int], purpose: str) -> np.ndarray:
    data = labels(split, partition, purpose)
    index = {int(r): float(v) for r, v in zip(data["rows"], data["y"])}
    missing = [int(r) for r in rows if int(r) not in index]
    if missing:
        raise KeyError(f"rows {missing[:5]} are not in {split}/{partition}; refusing to substitute")
    return np.asarray([index[int(r)] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# batching / prediction
# ---------------------------------------------------------------------------


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return PREP_DATA.collate(cache, list(rows))


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_rows(model, cache: Mapping[str, Any], rows: Sequence[int], device) -> np.ndarray:
    batch = to_device(collate(cache, list(rows)), device)
    model.eval()
    with torch.no_grad():
        return model(batch).detach().float().cpu().numpy().reshape(-1)


def predict_scores(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = to_device(collate(cache, rows), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# Morgan fingerprints (F architecture and the Tanimoto diagnostics)
# ---------------------------------------------------------------------------

FP_BITS = 2048
FP_RADIUS = 2


def morgan_bits(smiles: Sequence[str], rows: Sequence[int]) -> dict[int, np.ndarray]:
    """Exact binary Morgan(r=2, 2048) fingerprint, ``useChirality=False``."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out: dict[int, np.ndarray] = {}
    for row in rows:
        mol = Chem.MolFromSmiles(str(smiles[int(row)]))
        if mol is None:
            raise SystemExit(f"row {row}: unparseable SMILES")
        vec = np.zeros(FP_BITS, dtype=np.uint8)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_BITS, useChirality=False, useBondTypes=True, useFeatures=False
        )
        for bit in fp.GetOnBits():
            vec[int(bit)] = 1
        out[int(row)] = vec
    return out


def build_fingerprint_caches(splits: Sequence[str] = C.SPLITS + ("ft",), *, rebuild: bool = False) -> dict[str, Any]:
    src = source()
    need = sorted({int(r) for s in splits for p in C.PARTITIONS for r in (rows_by_partition(s).get(p) or [])})
    todo = [p for p in splits if rebuild or not (C.CACHE / f"fp_{p}.npz").exists()]
    if todo:
        try:
            import rdkit  # noqa: F401
        except Exception as exc:  # pragma: no cover - remote has no rdkit
            raise SystemExit(
                "Morgan fingerprints must be built on a host with rdkit and copied into cache/; "
                f"missing: {todo}"
            ) from exc
    bits = morgan_bits(src["smiles"], need) if todo else {}
    manifest = {}
    for split in splits:
        path = C.CACHE / f"fp_{split}.npz"
        rows = rows_by_partition(split)
        fit = np.stack([bits[r] for r in rows["fit"]]).astype(np.float32)
        mean = fit.mean(0)
        scale = fit.std(0)
        scale[scale <= 0] = 1.0
        payload = {"fit": fit, "mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}
        for partition in ("select", "diagnose", "audit"):
            part = rows.get(partition) or []
            if part:
                payload[partition] = np.stack([bits[r] for r in part]).astype(np.float32)
        if todo:
            tmp = path.parent / (path.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, path)
        manifest[split] = {"path": C.track_rel(path), "sha256": C.sha256_file(path),
                           "n_fit": int(fit.shape[0]), "n_bits": int(fit.shape[1]),
                           "n_constant_fit_bits": int((scale <= 1e-12).sum())}
        C.log(f"[fp] {split}: {fit.shape} constant bits={manifest[split]['n_constant_fit_bits']}")
    C.write_json(C.CACHE / "fingerprint_manifest.json", manifest)
    return manifest


__all__ = [n for n in dir() if not n.startswith("_")]
