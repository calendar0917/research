"""OGB molecular datasets: official splits, SMILES -> frozen lipo_v1 LCRC cache."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import bp_common as C
import bp_data as D
import lcrc_data as PREP_DATA
from feature_spec import save_spec


class InputAdapterBlocked(RuntimeError):
    pass


SHORT = {name: name.replace("ogbg-", "") for name in C.MOL_DATASETS}


def load_official(name: str) -> dict[str, Any]:
    from ogb.graphproppred import GraphPropPredDataset

    if name not in C.MOL_DATASETS:
        raise KeyError(name)
    ds = GraphPropPredDataset(name=name, root=str(C.OGB_ROOT))
    split = ds.get_idx_split()
    split = {k: np.asarray(split[k], dtype=np.int64) for k in ("train", "valid", "test")}
    n = len(ds)
    for k, rows in split.items():
        if rows.ndim != 1 or rows.size == 0:
            raise InputAdapterBlocked(f"{name}: empty {k} split")
        if np.any(rows < 0) or np.any(rows >= n):
            raise InputAdapterBlocked(f"{name}: {k} row outside [0,{n})")
        if np.unique(rows).size != rows.size:
            raise InputAdapterBlocked(f"{name}: duplicate {k} rows")
    union = np.concatenate([split["train"], split["valid"], split["test"]])
    if np.unique(union).size != n or not np.array_equal(np.sort(union), np.arange(n)):
        raise InputAdapterBlocked(f"{name}: official splits do not partition the dataset")
    mapping = Path(ds.root) / "mapping" / "mol.csv.gz"
    if not mapping.is_file():
        raise InputAdapterBlocked(f"{name}: official mapping/mol.csv.gz is missing")
    import pandas as pd

    df = pd.read_csv(mapping)
    if "smiles" not in df.columns:
        raise InputAdapterBlocked(f"{name}: mapping has no smiles column: {list(df.columns)}")
    if len(df) != n:
        raise InputAdapterBlocked(f"{name}: mapping rows {len(df)} != graphs {n}")
    smiles = [str(x) for x in df["smiles"].tolist()]
    labels = np.asarray(ds.labels, dtype=np.float64)
    if labels.shape[0] != n:
        raise InputAdapterBlocked(f"{name}: label rows {labels.shape[0]} != graphs {n}")
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if name == "ogbg-moltox21":
        if labels.shape[1] != 12:
            raise InputAdapterBlocked(f"{name}: expected 12 tasks, got {labels.shape[1]}")
        observed = np.isfinite(labels)
        if not np.isin(labels[observed], (0.0, 1.0)).all():
            raise InputAdapterBlocked(f"{name}: non-binary observed target")
    else:
        if labels.shape[1] != 1:
            raise InputAdapterBlocked(f"{name}: expected 1 task, got {labels.shape[1]}")
        if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise InputAdapterBlocked(f"{name}: non-binary/non-finite target")
    return {
        "name": name,
        "dataset": ds,
        "n": n,
        "split": split,
        "smiles": smiles,
        "labels": labels,
        "mapping_path": mapping,
        "mapping_sha256": C.sha256_file(mapping),
        "root": Path(ds.root),
        "num_tasks": int(labels.shape[1]),
    }


def build(name: str, *, force: bool = False) -> dict[str, Any]:
    from mb2_data import frozen_spec
    import mb2_rdkit as R

    t0 = time.perf_counter()
    info = load_official(name)
    spec = frozen_spec()
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    short = SHORT[name]
    save_spec(spec, C.ANALYSIS / f"{short}_spec.json")
    mols = []
    modes = {"strict": 0, "unsanitized_fallback": 0}
    failures = []
    per_graph = []
    for i, smi in enumerate(info["smiles"]):
        try:
            row, mode = R.smiles_to_raw_row_relaxed(smi, i)
            graph = R.raw_row_to_graph(row, i, spec)
            mol = PREP_DATA.build_molecule(graph, spec, validate=True)
        except Exception as exc:  # noqa: BLE001 - recorded, task is blocked rather than silently dropping
            failures.append({"row": i, "smiles": smi, "error": f"{type(exc).__name__}: {exc}"})
            continue
        modes[mode] += 1
        mols.append(mol)
        per_graph.append(
            {
                "row": i,
                "n_atoms": int(mol["n_atoms"]),
                "n_bonds": int(mol["n_bonds"]),
                "s2_paths": int(mol["path_ptr"][-1]),
            }
        )
        if (i + 1) % 5000 == 0:
            C.log(f"[cache {name}] {i + 1}/{info['n']} graphs {time.perf_counter() - t0:.1f}s")
    if failures:
        C.write_json(C.ANALYSIS / f"{short}_adapter_failures.json", failures)
        raise InputAdapterBlocked(f"{name}: {len(failures)} molecules failed the frozen adapter")
    if len(mols) != info["n"]:
        raise InputAdapterBlocked(f"{name}: built {len(mols)} molecules != {info['n']}")
    cache = PREP_DATA.concat_molecules(mols, spec, split="all")
    cache["gids"] = np.arange(len(mols), dtype=np.int64)
    cache["meta"]["build_seconds"] = float(time.perf_counter() - t0)
    cache["meta"]["dataset"] = name
    PREP_DATA.validate_cache(cache, spec)
    cache_path = C.CACHE / f"{short}.pt"
    D.save_cache(cache, cache_path)
    np.save(C.ANALYSIS / f"{short}_labels.npy", info["labels"])
    split_payload = {
        "dataset": name,
        "source": "OGB GraphPropPredDataset.get_idx_split()",
        "num_tasks": info["num_tasks"],
        "sizes": {k: int(v.size) for k, v in info["split"].items()},
        "index_sha256": {k: C.index_hash(v) for k, v in info["split"].items()},
        "rows": {k: v.tolist() for k, v in info["split"].items()},
        "partition_sha256": C.sha256_bytes(
            b"".join(info["split"][k].tobytes() for k in ("train", "valid", "test"))
        ),
    }
    C.write_json(C.ANALYSIS / f"{short}_split.json", split_payload)
    arr_n = np.asarray([r["n_atoms"] for r in per_graph], dtype=np.int64)
    arr_m = np.asarray([r["n_bonds"] for r in per_graph], dtype=np.int64)
    arr_s = np.asarray([r["s2_paths"] for r in per_graph], dtype=np.int64)
    np.savez(
        C.ANALYSIS / f"{short}_graph_distribution.npz",
        rows=np.asarray([r["row"] for r in per_graph], dtype=np.int64),
        n_atoms=arr_n,
        n_bonds=arr_m,
        s2_paths=arr_s,
    )
    stats = {
        "dataset": name,
        "cache": C.track_rel(cache_path),
        "cache_sha256": C.sha256_file(cache_path),
        "cache_bytes": int(cache_path.stat().st_size),
        "cache_build_seconds": float(cache["meta"]["build_seconds"]),
        "feature_spec_hash": spec.hash(),
        "mapping": C.track_rel(info["mapping_path"]),
        "mapping_sha256": info["mapping_sha256"],
        "num_tasks": info["num_tasks"],
        "splits": split_payload,
        "parse_modes": modes,
        "n_atoms": _dist(arr_n),
        "n_bonds": _dist(arr_m),
        "s2_paths": _dist(arr_s),
        "s2_paths_total": int(arr_s.sum()),
        "cache_validation": PREP_DATA.validate_cache(cache, spec),
        "labels_sha256": C.sha256_file(C.ANALYSIS / f"{short}_labels.npy"),
    }
    C.write_json(C.ANALYSIS / f"{short}_cache_stats.json", stats)
    return stats


def _dist(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": int(a.min()) if a.size else 0,
        "max": int(a.max()) if a.size else 0,
        "mean": float(a.mean()) if a.size else 0.0,
        "p50": float(np.percentile(a, 50)) if a.size else 0.0,
        "p95": float(np.percentile(a, 95)) if a.size else 0.0,
        "p99": float(np.percentile(a, 99)) if a.size else 0.0,
        "total": int(a.sum()) if a.size else 0,
    }


__all__ = ["InputAdapterBlocked", "SHORT", "load_official", "build"]
