"""Data scope, frozen-molecule access and WC_LOGP contribution identity.

The main experiment uses the complete official Valid/Test splits plus a fixed
128-molecule descriptive Train subset (the inherited sha256 selection rule).
The auxiliary experiment reuses the 823 frozen real-Lipophilicity pairs and
their 1646 endpoints.

No label is ever read while building candidates.  Targets appear only in
(a) the frozen target scaler used to express effect sizes and (b) the
auxiliary prediction-error report.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

import numpy as np

import lfa_common as C


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------

_BUNDLE: dict[str, Any] = {}
_OCC: dict[str, Any] = {}
_ORACLE: dict[str, dict[str, np.ndarray]] = {}


def bundle():
    if "base" not in _BUNDLE:
        import lfa_models as M

        _BUNDLE["base"] = M.raw_bundle()
    return _BUNDLE["base"]


def oracle_bundle():
    if "wc" not in _BUNDLE:
        mods = C.source_modules()
        _BUNDLE["wc"] = mods["gr_data"].oracle_bundle(C.DATASET, "WC_LOGP")
    return _BUNDLE["wc"]


def split_occ(split: str):
    if split not in _OCC:
        import lfa_models as M

        b = bundle()
        _OCC[split] = M.build_occ(b, split)
    return _OCC[split]


def oracle_payload() -> dict[str, np.ndarray]:
    if "wc" not in _ORACLE:
        path = C.XR_DIR / "analysis" / "ogbg-mollipo.npz"
        actual = C.sha256_file(path)
        expected = C.EXPECTED_ASSET_SHA256["ogbg-mollipo.npz"]
        if actual != expected:
            raise RuntimeError(f"WC_LOGP oracle identity: sha256 {actual} != {expected}")
        _ORACLE["wc"] = C.read_npz(path)
    return _ORACLE["wc"]


def graph_contribution(global_row_id: int) -> dict[str, Any]:
    """The WC_LOGP atom contributions of one official row and their closure."""
    payload = oracle_payload()
    ptr = np.asarray(payload["atom_ptr"], dtype=np.int64)
    atom = np.asarray(payload["WC_LOGP_atom"], dtype=np.float64)
    value = float(np.asarray(payload["WC_LOGP"], dtype=np.float64)[int(global_row_id)])
    a0, a1 = int(ptr[int(global_row_id)]), int(ptr[int(global_row_id) + 1])
    contrib = np.array(atom[a0:a1], dtype=np.float64)
    total = float(contrib.sum())
    err = abs(total - value)
    return {
        "contrib": contrib,
        "value": value,
        "sum": total,
        "closure_error": float(err),
        "closure_tolerance": float(C.oracle_tolerance(value)),
        "closed": bool(err <= float(C.oracle_tolerance(value))),
    }


# ---------------------------------------------------------------------------
# scope enumeration
# ---------------------------------------------------------------------------


def main_scope() -> list[dict[str, Any]]:
    """Official Valid/Test (all rows) + the fixed descriptive Train subset."""
    b = bundle()
    rows: list[dict[str, Any]] = []
    for split in C.MAIN_SPLITS:
        ids = np.asarray(b.global_row_ids[split], dtype=np.int64)
        if ids.size != C.EXPECTED_SPLIT_SIZES[split]:
            raise RuntimeError(f"{split}: {ids.size} rows != {C.EXPECTED_SPLIT_SIZES[split]}")
        for cache_row, row_id in enumerate(ids.tolist()):
            rows.append({"split": split, "cache_row": int(cache_row), "row_id": int(row_id),
                         "scope": C.SCOPE_MAIN, "stratum": split})
    mods = C.source_modules()
    from as_common import select_samples

    ids = np.asarray(b.global_row_ids[C.DESCRIPTION_SPLIT], dtype=np.int64)
    chosen = select_samples(C.DATASET, C.DESCRIPTION_SPLIT, ids, n=C.N_TRAIN_DESCRIPTION)["row_ids"]
    for row_id in chosen:
        hits = np.flatnonzero(ids == int(row_id))
        if hits.size != 1:
            raise RuntimeError(f"train descriptive row {row_id}: {hits.size} cache hits")
        rows.append({"split": C.DESCRIPTION_SPLIT, "cache_row": int(hits[0]), "row_id": int(row_id),
                     "scope": C.SCOPE_MAIN, "stratum": "train_description"})
    return rows


def aux_endpoints() -> list[dict[str, Any]]:
    """The frozen 823-pair endpoint catalogue (1646 endpoints)."""
    path = C.XR_DIR / "analysis" / "pair_endpoints.jsonl.gz"
    actual = C.sha256_file(path)
    expected = C.EXPECTED_ASSET_SHA256["pair_endpoints.jsonl.gz"]
    if actual != expected:
        raise RuntimeError(f"aux endpoint catalogue identity: sha256 {actual} != {expected}")
    rows = C.read_jsonl_gz(path)
    if len(rows) != 1646:
        raise RuntimeError(f"aux endpoints: {len(rows)} != 1646")
    return rows


def aux_scope(endpoints: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    endpoints = list(aux_endpoints() if endpoints is None else endpoints)
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in endpoints:
        key = (str(rec["split"]), int(rec["cache_row"]))
        if key not in seen:
            seen[key] = {"split": key[0], "cache_row": key[1], "row_id": int(rec["row_id"]),
                         "scope": C.SCOPE_AUX, "stratum": "aux"}
    return sorted(seen.values(), key=lambda r: (r["split"], r["cache_row"]))


def field_scope(scope: str) -> list[dict[str, Any]]:
    return main_scope() if scope == C.SCOPE_MAIN else aux_scope()


def iter_molecules(scope_rows: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for rec in scope_rows:
        yield dict(rec)


# ---------------------------------------------------------------------------
# frozen molecule / field access
# ---------------------------------------------------------------------------


def target_std(task: str) -> float:
    """Train-only target standard deviation of the task's frozen model."""
    if task == "wc":
        return float(oracle_bundle().scaler["std"])
    return float(bundle().scaler["std"])


def target_mean(task: str) -> float:
    if task == "wc":
        return float(oracle_bundle().scaler["mean"])
    return float(bundle().scaler["mean"])


def target_values(task: str, split: str) -> np.ndarray:
    b = oracle_bundle() if task == "wc" else bundle()
    return np.asarray(b.targets_raw[split], dtype=np.float64)


def build_npz_rows(rows: Sequence[Mapping[str, Any]], **arrays: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate per-molecule arrays with int64 pointers."""
    out: dict[str, Any] = {}
    for name, values in arrays.items():
        vals = [np.asarray(v).reshape(-1) for v in values]
        dtype = np.float64 if name in ("grad", "contrib", "pred_raw", "pred_std") else np.int64
        out[name] = np.concatenate([v.astype(dtype) for v in vals]) if vals else np.zeros(0, dtype=dtype)
        out[f"{name}_ptr"] = np.concatenate([[0], np.cumsum([v.size for v in vals])]).astype(np.int64)
    out["row_id"] = np.asarray([int(r["row_id"]) for r in rows], dtype=np.int64)
    out["cache_row"] = np.asarray([int(r["cache_row"]) for r in rows], dtype=np.int64)
    out["split_code"] = np.asarray([C.SPLIT_CODE[str(r["split"])] for r in rows], dtype=np.int64)
    out["n_atoms"] = np.asarray([int(r["n_atoms"]) for r in rows], dtype=np.int64)
    out["n_bonds"] = np.asarray([int(r["n_bonds"]) for r in rows], dtype=np.int64)
    return out


__all__ = [
    "bundle", "oracle_bundle", "split_occ", "oracle_payload", "graph_contribution",
    "main_scope", "aux_endpoints", "aux_scope", "field_scope", "iter_molecules",
    "target_std", "target_mean", "target_values", "build_npz_rows",
]
