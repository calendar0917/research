"""Phase B data: official ``ogbg-molhiv`` **Train split only** and the frozen CV folds.

Compliance rules implemented here (and asserted by ``tests/``):

* Only ``split/scaffold/train.csv.gz`` is opened.  The official Valid/Test index
  files are never read.
* The mapping CSV is read with ``usecols=['smiles','mol_id']``; the ``HIV_active``
  column is never decoded.
* ``raw/graph-label.csv.gz`` is streamed line by line and only the positions in
  the official Train index set are kept, so no Valid/Test label is ever
  materialised in memory.
* No feature is derived from a label.

Folds are exactly the pre-registered ones: outer
``StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261103)`` and,
inside each outer Train, ``StratifiedGroupKFold(n_splits=2, shuffle=True,
random_state=20261103 + outer_fold)``, grouped by Bemis-Murcko scaffold with the
canonical-SMILES fallback.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import nb_common as C
import nb_features as F

#: Recorded by the frozen molhiv benchmark for the official Train split.
FROZEN_TRAIN_ROWS_SHA256 = "4e77289653e41be5d2267f54d9f33fda4b70fd622c19b93a282db710a9318b0a"
FROZEN_TRAIN_CACHE_FINGERPRINT = "111d91752a6d451c6f94a6700b4df889a9d9762c90e734ef4f16e4b1a4258596"


# ---------------------------------------------------------------------------
# official Train-only load
# ---------------------------------------------------------------------------


def _stream_train_labels(train_rows: Sequence[int], path: Path | None = None) -> np.ndarray:
    """Read only the official Train label positions from the one-column label file.

    The file is consumed incrementally; a line whose position is not in the
    Train index set is discarded immediately and never stored.
    """
    wanted = {int(r): i for i, r in enumerate(train_rows)}
    out = np.full(len(train_rows), -1, dtype=np.int64)
    label_path = Path(path) if path is not None else C.MC.OGB_SOURCE / C.OGB_GRAPH_LABEL
    with gzip.open(label_path, "rt") as fh:
        for position, line in enumerate(fh):
            hit = wanted.get(position)
            if hit is not None:
                out[hit] = int(line.strip())
    if np.any(out < 0):
        raise ValueError("some official Train label positions were not found in graph-label.csv.gz")
    return out


def load_train_only() -> dict[str, Any]:
    """Official Train indices, Train SMILES (mapping) and Train labels."""
    root = C.MC.OGB_SOURCE
    train_path = root / C.OGB_TRAIN_SPLIT
    split = pd.read_csv(train_path, header=None).to_numpy().reshape(-1)
    train_rows = [int(x) for x in split]  # published order preserved
    if len(train_rows) != C.OGB_TRAIN_ROWS:
        raise ValueError(f"official Train size {len(train_rows)} != frozen {C.OGB_TRAIN_ROWS}")
    if len(set(train_rows)) != len(train_rows):
        raise ValueError("duplicate official Train row ids")
    if min(train_rows) < 0 or max(train_rows) >= C.OGB_TOTAL_ROWS:
        raise ValueError("official Train row id outside the dataset")

    mapping = pd.read_csv(root / C.OGB_MAPPING, usecols=["smiles", "mol_id"])
    if len(mapping) != C.OGB_TOTAL_ROWS:
        raise ValueError(f"mapping rows {len(mapping)} != {C.OGB_TOTAL_ROWS}")
    smiles_all = [str(s) for s in mapping["smiles"].tolist()]
    mol_ids_all = [str(s) for s in mapping["mol_id"].tolist()]
    smiles = [smiles_all[r] for r in train_rows]
    mol_ids = [mol_ids_all[r] for r in train_rows]

    y = _stream_train_labels(train_rows)
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError(f"non-binary official Train labels: {sorted(set(np.unique(y).tolist()))}")

    files = {}
    for rel in (C.OGB_TRAIN_SPLIT, C.OGB_MAPPING, C.OGB_GRAPH_LABEL):
        p = root / rel
        files[rel] = {"sha256": C.sha256_file(p), "size_bytes": int(p.stat().st_size)}

    rows_hash = C.rows_sha256(train_rows)
    return {
        "dataset": "ogbg-molhiv",
        "rows": train_rows,
        "smiles": smiles,
        "mol_ids": mol_ids,
        "labels": y,
        "files": files,
        "audit": {
            "n_train": int(len(train_rows)),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "positive_fraction": float(y.mean()),
            "rows_sha256": rows_hash,
            "frozen_rows_sha256": FROZEN_TRAIN_ROWS_SHA256,
            "rows_match_frozen_benchmark": bool(rows_hash == FROZEN_TRAIN_ROWS_SHA256),
            "valid_test_index_files_opened": False,
            "valid_test_labels_read": False,
            "mapping_hiv_active_column_read": False,
            "label_read_strategy": (
                "streamed from raw/graph-label.csv.gz; only official Train positions are retained in memory"
            ),
            "rows_used": "official split/scaffold/train.csv.gz only",
        },
    }


def load_train_cache(smiles: Sequence[str], rows: Sequence[int]) -> dict[str, Any]:
    """The frozen ``lipo_v1`` structure cache for exactly these Train rows."""
    spec = C.frozen_spec()
    cache = C.MD.load_or_build_split_cache("ogbg-molhiv", "train", list(rows), list(smiles), spec, rebuild=False)
    if [int(x) for x in np.asarray(cache["rows"]).tolist()] != [int(r) for r in rows]:
        raise ValueError("structure cache rows differ from the official Train rows")
    if cache["meta"].get("labels_present"):
        raise ValueError("structure cache must be label-free")
    if cache["meta"].get("feature_spec_hash") != spec.hash():
        raise ValueError("structure cache spec hash mismatch")
    return cache


def g42_matrix(cache: Mapping[str, Any]) -> np.ndarray:
    """The frozen 42-d global block (raw, pre-normalisation) per molecule."""
    g = np.asarray(cache["global"], dtype=np.float32)
    if g.shape[1] != C.GLOBAL_DIM:
        raise ValueError(f"global block width {g.shape[1]} != {C.GLOBAL_DIM}")
    if not np.isfinite(g).all():
        raise ValueError("non-finite global block")
    return g


# ---------------------------------------------------------------------------
# scaffolds
# ---------------------------------------------------------------------------


def scaffolds(smiles: Sequence[str]) -> dict[str, Any]:
    groups: list[str] = []
    kinds: list[str] = []
    modes: list[str] = []
    canonical: list[str] = []
    for i, s in enumerate(smiles):
        rec = F.scaffold_key(s)
        groups.append(str(rec["group"]))
        kinds.append(str(rec["kind"]))
        modes.append(str(rec["mode"]))
        canonical.append(str(rec["canonical_smiles"]))
        if (i + 1) % 10000 == 0:
            C.log(f"[scaffolds] {i + 1}/{len(smiles)}")
    uniq = sorted(set(groups))
    index = {g: i for i, g in enumerate(uniq)}
    codes = np.asarray([index[g] for g in groups], dtype=np.int64)
    return {
        "groups": groups,
        "codes": codes,
        "n_groups": int(len(uniq)),
        "kinds": kinds,
        "modes": modes,
        "canonical_smiles": canonical,
        "kind_counts": {k: int(kinds.count(k)) for k in sorted(set(kinds))},
        "mode_counts": {m: int(modes.count(m)) for m in sorted(set(modes))},
        "largest_group_size": int(np.bincount(codes).max()) if codes.size else 0,
        "singleton_groups": int((np.bincount(codes) == 1).sum()) if codes.size else 0,
    }


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------


def _fold_audit(idx: Sequence[int], y: np.ndarray, codes: np.ndarray) -> dict[str, Any]:
    idx = np.asarray(sorted(int(i) for i in idx), dtype=np.int64)
    labels = y[idx]
    return {
        "n": int(idx.size),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "n_classes": int(len(np.unique(labels))),
        "n_scaffolds": int(len(np.unique(codes[idx]))),
        "index_sha256": C.rows_sha256(idx.tolist()),
        "index_min": int(idx.min()) if idx.size else None,
        "index_max": int(idx.max()) if idx.size else None,
    }


def build_folds(y: np.ndarray, codes: np.ndarray) -> dict[str, Any]:
    """Outer 3-fold + inner 2-fold grouped splits with the frozen seeds."""
    n = int(y.size)
    X_placeholder = np.zeros((n, 1), dtype=np.float32)
    outer_splitter = StratifiedGroupKFold(
        n_splits=C.OUTER_SPLITS, shuffle=True, random_state=C.SPLIT_SEED
    )
    outer = list(outer_splitter.split(X_placeholder, y, codes))
    if len(outer) != C.OUTER_SPLITS:
        raise ValueError("outer splitter did not return the requested number of folds")

    folds: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(outer):
        tr = np.asarray(sorted(int(i) for i in tr), dtype=np.int64)
        va = np.asarray(sorted(int(i) for i in va), dtype=np.int64)
        overlap = set(np.unique(codes[tr])).intersection(set(np.unique(codes[va])))
        if overlap:
            raise ValueError(f"outer fold {fold_id}: {len(overlap)} shared scaffolds")
        if len(set(tr.tolist()).intersection(set(va.tolist()))):
            raise ValueError(f"outer fold {fold_id}: shared rows")
        inner_splitter = StratifiedGroupKFold(
            n_splits=C.INNER_SPLITS, shuffle=True, random_state=C.SPLIT_SEED + fold_id
        )
        inner = list(
            inner_splitter.split(
                np.zeros((tr.size, 1), dtype=np.float32), y[tr], codes[tr]
            )
        )
        if len(inner) != C.INNER_SPLITS:
            raise ValueError(f"inner splitter returned {len(inner)} folds")
        inner_folds = []
        for inner_id, (itr, iva) in enumerate(inner):
            itr_global = np.asarray(sorted(int(tr[i]) for i in itr), dtype=np.int64)
            iva_global = np.asarray(sorted(int(tr[i]) for i in iva), dtype=np.int64)
            inner_overlap = set(np.unique(codes[itr_global])).intersection(
                set(np.unique(codes[iva_global]))
            )
            if inner_overlap:
                raise ValueError(f"fold {fold_id}/{inner_id}: shared inner scaffolds")
            inner_folds.append(
                {
                    "inner_fold": int(inner_id),
                    "train": _fold_audit(itr_global, y, codes),
                    "valid": _fold_audit(iva_global, y, codes),
                    "train_rows": itr_global.tolist(),
                    "valid_rows": iva_global.tolist(),
                }
            )
        folds.append(
            {
                "outer_fold": int(fold_id),
                "train": _fold_audit(tr, y, codes),
                "valid": _fold_audit(va, y, codes),
                "train_rows": tr.tolist(),
                "valid_rows": va.tolist(),
                "inner": inner_folds,
                "inner_seed": int(C.SPLIT_SEED + fold_id),
                "inner_seed_rule": "SPLIT_SEED + outer_fold",
            }
        )

    feasible = True
    reasons: list[str] = []
    for fold in folds:
        for part in ("train", "valid"):
            if fold[part]["n_classes"] < 2:
                feasible = False
                reasons.append(f"outer fold {fold['outer_fold']} {part} has a single class")
        for inner in fold["inner"]:
            for part in ("train", "valid"):
                if inner[part]["n_classes"] < 2:
                    feasible = False
                    reasons.append(
                        f"outer fold {fold['outer_fold']} inner {inner['inner_fold']} {part} has a single class"
                    )

    # coverage: the outer validation folds must partition the Train set exactly
    valid_union: list[int] = []
    for fold in folds:
        valid_union.extend(fold["valid_rows"])
    covered = sorted(set(valid_union))
    partition_ok = bool(covered == list(range(n)) and len(valid_union) == n)

    return {
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "outer": {
            "n_splits": C.OUTER_SPLITS,
            "shuffle": True,
            "random_state": C.SPLIT_SEED,
        },
        "inner": {
            "n_splits": C.INNER_SPLITS,
            "shuffle": True,
            "random_state_rule": "SPLIT_SEED + outer_fold",
        },
        "group_key": "Bemis-Murcko scaffold; empty/uncomputable scaffold -> canonical SMILES",
        "folds": folds,
        "n_train_total": n,
        "n_positive_total": int((y == 1).sum()),
        "n_negative_total": int((y == 0).sum()),
        "n_scaffolds_total": int(len(np.unique(codes))),
        "outer_valid_partition_covers_all_rows": partition_ok,
        "feasible": bool(feasible),
        "infeasible_reasons": reasons,
        "status": "OK" if feasible and partition_ok else "SPLIT_INFEASIBLE",
    }


__all__ = [
    "FROZEN_TRAIN_ROWS_SHA256",
    "FROZEN_TRAIN_CACHE_FINGERPRINT",
    "load_train_only",
    "load_train_cache",
    "g42_matrix",
    "scaffolds",
    "build_folds",
]
