"""Official ``ogbg-molpcba`` access, 128-task label artifact, exact feature cache.

Layout
------
* :func:`load_official` - OGB ``GraphPropPredDataset`` + ``get_idx_split()`` and
  the official ``mapping/mol.csv.gz`` SMILES, with hard load-time assertions.
* :func:`build_label_artifact` - writes ``[437929, 128]`` float32 targets in which
  **every official Test row is NaN**.  Only Train/Valid targets are copied and
  validated; no Test target is inspected, written or scored.
* :func:`build_shards` - chunked, multi-process construction of the frozen
  lipo_v1 LCRC arrays.  Every molecule goes through exactly the historical
  adapter chain (``mb2_rdkit.smiles_to_raw_row_relaxed`` -> ``raw_row_to_graph``
  -> ``lcrc_data.build_molecule``), so the feature definition is inherited, not
  reimplemented.  A parse/feature failure is recorded per row and blocks the
  cache instead of silently dropping the row.
* :func:`merge_shards` - deterministic merge into one cache, staged through
  memory-mapped ``.npy`` files so the peak RSS stays well below the cache size.
* :func:`feature_cache_audit` - label-free checks on the merged cache, including
  an *exact, all-atom* S2 count identity and an exact S2 content comparison with
  the frozen upstream enumeration on a sample of molecules.
* :func:`historical_s2_crosscheck` - re-derives the S2 counts of the historical
  MolHIV cache from its own bond table and rebuilds a few MolHIV molecules from
  the official SMILES to prove the S2 definition is unchanged.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import pc_common as C


class InputAdapterBlocked(RuntimeError):
    """The official SMILES/feature adapter could not process some rows."""


# ---------------------------------------------------------------------------
# official dataset
# ---------------------------------------------------------------------------


def load_official(*, root: Path | str | None = None) -> dict[str, Any]:
    from ogb.graphproppred import GraphPropPredDataset

    root = Path(root) if root is not None else C.OGB_ROOT
    ds = GraphPropPredDataset(name=C.DATASET, root=str(root))
    split = ds.get_idx_split()
    split = {k: np.asarray(split[k], dtype=np.int64) for k in ("train", "valid", "test")}
    n = len(ds)
    if n != C.N_MOLECULES:
        raise InputAdapterBlocked(f"{C.DATASET}: {n} graphs != frozen official count {C.N_MOLECULES}")
    for name, rows in split.items():
        if rows.ndim != 1 or rows.size != C.SPLIT_SIZES[name]:
            raise InputAdapterBlocked(
                f"{C.DATASET}: {name} size {rows.size if rows.ndim == 1 else rows.shape} != {C.SPLIT_SIZES[name]}"
            )
        if np.any(rows < 0) or np.any(rows >= n):
            raise InputAdapterBlocked(f"{C.DATASET}: {name} row outside [0,{n})")
        if np.unique(rows).size != rows.size:
            raise InputAdapterBlocked(f"{C.DATASET}: duplicate {name} rows")
    union = np.concatenate([split["train"], split["valid"], split["test"]])
    if np.unique(union).size != n or not np.array_equal(np.sort(union), np.arange(n)):
        raise InputAdapterBlocked(f"{C.DATASET}: official splits do not partition the dataset")
    mapping = Path(ds.root) / "mapping" / "mol.csv.gz"
    if not mapping.is_file():
        raise InputAdapterBlocked(f"{C.DATASET}: official mapping/mol.csv.gz is missing")
    import pandas as pd

    df = pd.read_csv(mapping)
    if "smiles" not in df.columns:
        raise InputAdapterBlocked(f"{C.DATASET}: mapping has no smiles column: {list(df.columns)}")
    if len(df) != n:
        raise InputAdapterBlocked(f"{C.DATASET}: mapping rows {len(df)} != graphs {n}")
    smiles = [str(x) for x in df["smiles"].tolist()]
    labels = np.asarray(ds.labels)
    if labels.ndim != 2 or labels.shape != (n, C.N_TASKS):
        raise InputAdapterBlocked(
            f"{C.DATASET}: label matrix {labels.shape} != {(n, C.N_TASKS)}"
        )
    return {
        "name": C.DATASET,
        "dataset": ds,
        "n": int(n),
        "split": split,
        "smiles": smiles,
        "labels": labels,
        "mapping_path": mapping,
        "mapping_sha256": C.sha256_file(mapping),
        "root": Path(ds.root),
        "num_tasks": int(labels.shape[1]),
        "raw_files": sorted(str(p.relative_to(Path(ds.root))) for p in Path(ds.root).rglob("*") if p.is_file()),
    }


def split_payload(info: Mapping[str, Any]) -> dict[str, Any]:
    split = info["split"]
    return {
        "dataset": info["name"],
        "source": "OGB GraphPropPredDataset.get_idx_split()",
        "num_tasks": int(info["num_tasks"]),
        "n_molecules": int(info["n"]),
        "sizes": {k: int(v.size) for k, v in split.items()},
        "index_sha256": {k: C.index_hash(v) for k, v in split.items()},
        "partition_sha256": C.sha256_bytes(
            b"".join(np.asarray(split[k], dtype=np.int64).tobytes() for k in ("train", "valid", "test"))
        ),
        "rows": {k: np.asarray(v, dtype=np.int64).tolist() for k, v in split.items()},
    }


# ---------------------------------------------------------------------------
# label artifact: Train/Valid only, Test rows NaN
# ---------------------------------------------------------------------------


def build_label_artifact(info: Mapping[str, Any], out_path: Path | str) -> dict[str, Any]:
    """Copy Train/Valid targets; leave every Test row NaN (never inspected)."""
    split = info["split"]
    labels = np.asarray(info["labels"])
    tv = np.concatenate([np.asarray(split["train"], dtype=np.int64), np.asarray(split["valid"], dtype=np.int64)])
    y_tv = labels[tv]
    observed = np.isfinite(y_tv)
    if not np.isin(y_tv[observed], (0.0, 1.0)).all():
        bad = np.unique(y_tv[observed][~np.isin(y_tv[observed], (0.0, 1.0))])[:5]
        raise InputAdapterBlocked(f"{C.DATASET}: non-binary observed target values {bad}")
    masked = np.full(labels.shape, np.nan, dtype=np.float32)
    masked[tv] = np.asarray(labels[tv], dtype=np.float32)
    test_rows = np.asarray(split["test"], dtype=np.int64)
    if np.isfinite(masked[test_rows]).any():  # pragma: no cover - defensive
        raise RuntimeError("test rows must be NaN in the label artifact")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, masked)
    event = {
        "utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "n_rows": int(masked.shape[0]),
        "n_tasks": int(masked.shape[1]),
        "train_valid_rows_copied": int(tv.size),
        "test_rows_masked_nan": int(test_rows.size),
        "test_row_values_inspected": False,
        "test_row_values_written": False,
        "test_row_values_used_in_any_metric": False,
        "artifact": C.track_rel(p),
        "artifact_sha256": C.sha256_file(p),
    }
    ledger = C.ANALYSIS / "label_access_ledger.json"
    payload = C.read_json(ledger) if ledger.exists() else {"schema": "lcrc-pcba-label-access-v1", "events": []}
    payload["events"].append(event)
    C.write_json(ledger, payload)
    return event


def task_label_counts(y: np.ndarray) -> dict[str, Any]:
    """Per-task observed / positive / negative / missing counts for one split."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("label matrix must be 2-D")
    observed = np.isfinite(y)
    pos = np.sum((y == 1.0) & observed, axis=0)
    neg = np.sum((y == 0.0) & observed, axis=0)
    obs = observed.sum(axis=0)
    miss = y.shape[0] - obs
    return {
        "observed": obs.astype(np.int64).tolist(),
        "positive": pos.astype(np.int64).tolist(),
        "negative": neg.astype(np.int64).tolist(),
        "missing": miss.astype(np.int64).tolist(),
        "observed_total": int(obs.sum()),
        "positive_total": int(pos.sum()),
        "negative_total": int(neg.sum()),
        "missing_total": int(miss.sum()),
        "tasks_with_no_observed": int((obs == 0).sum()),
        "tasks_with_no_positive": int((pos == 0).sum()),
        "tasks_with_no_negative": int((neg == 0).sum()),
        "tasks_scorable_ap": int(((pos > 0) & (neg > 0)).sum()),
        "molecules_with_all_tasks_missing": int((~observed.any(axis=1)).sum()),
        "observed_per_molecule": {
            "min": int(observed.sum(axis=1).min()),
            "mean": float(observed.sum(axis=1).mean()),
            "max": int(observed.sum(axis=1).max()),
        },
    }


def data_audit(
    info: Mapping[str, Any], label_path: Path | str, label_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    split = info["split"]
    masked = np.load(Path(label_path))
    train = np.asarray(split["train"], dtype=np.int64)
    valid = np.asarray(split["valid"], dtype=np.int64)
    train_counts = task_label_counts(masked[train])
    valid_counts = task_label_counts(masked[valid])
    smiles = info["smiles"]
    uniq = len(set(smiles))
    payload = {
        "schema": "lcrc-pcba-data-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": {
            "name": info["name"],
            "n_molecules": int(info["n"]),
            "n_tasks": int(info["num_tasks"]),
            "ogb_root": C.track_rel(info["root"]),
            "mapping": C.track_rel(info["mapping_path"]),
            "mapping_sha256": info["mapping_sha256"],
            "num_raw_files": len(info["raw_files"]),
            "raw_files": info["raw_files"][:40],
        },
        "splits": {
            k: v
            for k, v in split_payload(info).items()
            if k != "rows"
        }
        | {"rows_reference": "analysis/pcba_split.json (full row lists, compact JSON)"},
        "smiles": {
            "n": len(smiles),
            "n_unique": uniq,
            "n_duplicate_strings": len(smiles) - uniq,
            "n_empty": int(sum(1 for s in smiles if not s.strip())),
        },
        "labels": {
            "artifact": label_artifact.get("artifact"),
            "artifact_sha256": label_artifact.get("artifact_sha256"),
            "shape": [int(masked.shape[0]), int(masked.shape[1])],
            "dtype": str(masked.dtype),
            "nan_policy": "unobserved label; masked out of the loss and out of the AP",
            "test_policy": "official Test rows are NaN in this artifact and are never indexed",
            "train": train_counts,
            "valid": valid_counts,
        },
        "gates": {
            "n_molecules_is_official": int(info["n"]) == C.N_MOLECULES,
            "n_tasks_is_128": int(info["num_tasks"]) == C.N_TASKS,
            "splits_are_official_and_disjoint": True,
            "split_sizes_official": {k: int(v.size) == C.SPLIT_SIZES[k] for k, v in split.items()},
            "test_rows_all_nan": bool(np.isnan(masked[np.asarray(split["test"])]).all()),
            "train_rows_have_observed_labels": train_counts["observed_total"] > 0,
            "valid_rows_have_observed_labels": valid_counts["observed_total"] > 0,
            "observed_values_binary": True,
        },
    }
    payload["status"] = "OK" if all(
        v for k, v in payload["gates"].items() if isinstance(v, bool)
    ) else "FAILED"
    return payload


# ---------------------------------------------------------------------------
# chunked feature cache build
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(spec_path: str, shard_dir: str) -> None:
    from feature_spec import load_spec

    _WORKER["spec"] = load_spec(Path(spec_path))
    _WORKER["dir"] = Path(shard_dir)


def _build_chunk(task: tuple[int, list[int], list[str]]) -> dict[str, Any]:
    """Worker: build one chunk and write a shard file (no labels involved)."""
    import lcrc_data as PREP_DATA
    import mb2_rdkit as R

    idx, rows, smiles = task
    spec = _WORKER["spec"]
    shard_dir = _WORKER["dir"]
    t0 = time.perf_counter()
    mols: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    modes = {"strict": 0, "unsanitized_fallback": 0}
    n_atoms: list[int] = []
    n_bonds: list[int] = []
    s2_paths: list[int] = []
    for row, smi in zip(rows, smiles):
        try:
            raw, mode = R.smiles_to_raw_row_relaxed(smi, int(row))
            graph = R.raw_row_to_graph(raw, int(row), spec)
            mol = PREP_DATA.build_molecule(graph, spec, validate=True)
        except Exception as exc:  # noqa: BLE001 - recorded, then the build is blocked
            failures.append({"row": int(row), "smiles": smi, "error": f"{type(exc).__name__}: {exc}"})
            continue
        modes[mode] += 1
        mols.append(mol)
        n_atoms.append(int(mol["n_atoms"]))
        n_bonds.append(int(mol["n_bonds"]))
        s2_paths.append(int(mol["path_ptr"][-1]))
    out = shard_dir / f"shard_{int(idx):05d}.pt"
    payload: dict[str, Any] = {
        "shard": int(idx),
        "rows": list(map(int, rows)),
        "failures": failures,
        "n_built": len(mols),
        "modes": modes,
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "s2_paths": s2_paths,
        "path": str(out),
        "seconds": float(time.perf_counter() - t0),
        "sha256": "",
    }
    if mols:
        cache = PREP_DATA.concat_molecules(mols, spec, split="pcba_shard")
        cache["gids"] = np.asarray(rows, dtype=np.int64)
        import torch

        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, out)
        payload["sha256"] = C.sha256_file(out)
    return payload


def build_shards(
    info: Mapping[str, Any],
    *,
    rows: Sequence[int] | None = None,
    shard_dir: Path | str,
    workers: int = 8,
    chunk_size: int = 4000,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    """Build the frozen lipo_v1 LCRC arrays for the selected official rows."""
    from mb2_data import frozen_spec
    from feature_spec import save_spec

    spec = frozen_spec()
    spec_path = Path(spec_path) if spec_path else C.ANALYSIS / "pcba_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    save_spec(spec, spec_path)
    smiles_all = info["smiles"]
    rows = list(range(len(smiles_all))) if rows is None else [int(r) for r in rows]
    if not rows:
        raise ValueError("no rows selected")
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate rows in build selection")
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(shard_dir.glob("shard_*.pt")):
        stale.unlink()
    tasks = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        tasks.append((len(tasks), chunk, [smiles_all[r] for r in chunk]))
    C.log(f"[cache] {len(rows)} rows in {len(tasks)} chunks, workers={workers}")
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    if workers <= 1:
        _init_worker(str(spec_path), str(shard_dir))
        for task in tasks:
            results.append(_build_chunk(task))
            C.log(f"[cache] chunk {results[-1]['shard']} done ({results[-1]['seconds']:.1f}s)")
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(spec_path), str(shard_dir)),
        ) as pool:
            for res in pool.map(_build_chunk, tasks):
                results.append(res)
                C.log(
                    f"[cache] chunk {res['shard']}/{len(tasks) - 1} rows={res['n_built']} "
                    f"{res['seconds']:.1f}s"
                )
    failures = [f for res in results for f in res["failures"]]
    summary = {
        "schema": "lcrc-pcba-cache-build-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "feature_spec_hash": spec.hash(),
        "feature_spec": C.track_rel(spec_path),
        "n_rows_requested": len(rows),
        "n_rows_built": int(sum(res["n_built"] for res in results)),
        "rows_sha256": C.index_hash(rows),
        "shard_size": int(chunk_size),
        "workers": int(workers),
        "wall_seconds": float(time.perf_counter() - t0),
        "parse_modes": {
            key: int(sum(res["modes"][key] for res in results))
            for key in ("strict", "unsanitized_fallback")
        },
        "s2_paths_total": int(sum(sum(res["s2_paths"]) for res in results)),
        "n_atoms_total": int(sum(sum(res["n_atoms"]) for res in results)),
        "n_bonds_total": int(sum(sum(res["n_bonds"]) for res in results)),
        "s2_paths": _dist(np.concatenate([np.asarray(res["s2_paths"], dtype=np.int64) for res in results]) if results else np.zeros(0)),
        "n_atoms": _dist(np.concatenate([np.asarray(res["n_atoms"], dtype=np.int64) for res in results]) if results else np.zeros(0)),
        "n_bonds": _dist(np.concatenate([np.asarray(res["n_bonds"], dtype=np.int64) for res in results]) if results else np.zeros(0)),
        "shards": [
            {
                "shard": int(res["shard"]),
                "path": C.track_rel(res["path"]) if res["path"] else None,
                "sha256": res["sha256"],
                "n_built": int(res["n_built"]),
                "first_row": int(res["rows"][0]),
                "last_row": int(res["rows"][-1]),
                "seconds": float(res["seconds"]),
            }
            for res in results
        ],
        "n_failures": len(failures),
        "failures_file": None,
        "status": "OK" if not failures else "BLOCKED_ADAPTER_FAILURES",
    }
    if failures:
        path = C.ANALYSIS / "pcba_adapter_failures.json"
        C.write_json(path, {"schema": "lcrc-pcba-adapter-failures-v1", "failures": failures})
        summary["failures_file"] = C.track_rel(path)
        C.write_json(C.ANALYSIS / "pcba_cache_build.json", summary)
        raise InputAdapterBlocked(
            f"{len(failures)} official SMILES rows failed the frozen LCRC adapter; "
            f"see {C.track_rel(path)} (rows are not dropped or substituted)"
        )
    C.write_json(C.ANALYSIS / "pcba_cache_build.json", summary)
    return summary


def _dist(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    if not a.size:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "total": 0}
    return {
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "total": int(a.sum()),
    }


# ---------------------------------------------------------------------------
# deterministic merge (memory-staged)
# ---------------------------------------------------------------------------

_POINTER_SPECS = (
    # name, per-molecule or per-atom
    ("atom_ptr", "molecule"),
    ("bond_ptr", "molecule"),
    ("nbr_ptr", "atom"),
    ("path_ptr", "atom"),
)
# global base offset for each concatenated pointer array
_POINTER_BASE = {"atom_ptr": "atoms", "bond_ptr": "bonds", "nbr_ptr": "s1", "path_ptr": "s2"}

_VALUE_FIELDS = (
    ("atom_feat", np.float32),
    ("atom_type", np.int64),
    ("atom_scalar", np.float32),
    ("bond_feat", np.float32),
    ("bond_type", np.int64),
    ("bond_index", np.int64),
    ("nbr_code", np.int32),
    ("path_cat", np.int32),
    ("path_end63", np.int32),
    ("path_end", np.int64),
    ("path_closure", np.int8),
    ("global", np.float32),
    ("n_components", np.int64),
    ("n_junctions", np.int64),
    ("n_ring_systems", np.int64),
)


def _field_shape(name: str, spec, counts: Mapping[str, int]) -> tuple[int, ...]:
    if name == "atom_type":
        return (int(counts["atoms"]),)
    if name == "bond_type":
        return (int(counts["bonds"]),)
    if name == "atom_feat":
        return (int(counts["atoms"]), int(spec.atom_feat_dim))
    if name == "atom_scalar":
        return (int(counts["atoms"]), int(spec.n_scalars))
    if name == "bond_feat":
        return (int(counts["bonds"]), int(spec.bond_feat_dim))
    if name == "bond_index":
        return (int(counts["bonds"]), 2)
    if name == "nbr_code":
        return (int(counts["s1"]),)
    if name in ("path_cat", "path_end63", "path_end", "path_closure"):
        return (int(counts["s2"]),)
    if name == "global":
        return (int(counts["molecules"]), int(spec.global_dim))
    if name in ("n_components", "n_junctions", "n_ring_systems"):
        return (int(counts["molecules"]),)
    raise KeyError(name)


def _field_offset(name: str, off: Mapping[str, int]) -> int:
    if name in ("atom_type", "atom_feat", "atom_scalar"):
        return int(off["atoms"])
    if name in ("bond_type", "bond_feat", "bond_index"):
        return int(off["bonds"])
    if name == "nbr_code":
        return int(off["s1"])
    if name in ("path_cat", "path_end63", "path_end", "path_closure"):
        return int(off["s2"])
    return int(off["molecules"])


def merge_shards(
    shard_paths: Sequence[Path | str], spec, *, out_dir: Path | str, split: str = "all"
) -> dict[str, Any]:
    """Merge shards into a memory-mapped **directory** cache.

    Every field is written straight into an ``.npy`` file that is never loaded
    back into RAM, so peak RSS stays at roughly one shard (~45 MB) plus dirty
    page cache that the kernel can reclaim.  Training reads it with
    ``np.load(..., mmap_mode="r")`` (see :func:`load_cache`).
    """
    import shutil

    t0 = time.perf_counter()
    paths = [Path(p) for p in shard_paths]
    if not paths:
        raise ValueError("no shards to merge")
    out_dir = Path(out_dir)
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    counts = {"molecules": 0, "atoms": 0, "bonds": 0, "s1": 0, "s2": 0}
    for p in paths:
        shard = torch_load(p)
        _check_shard(shard)
        counts["molecules"] += int(shard["n_molecules"])
        counts["atoms"] += int(shard["atom_ptr"][-1])
        counts["bonds"] += int(shard["bond_ptr"][-1])
        counts["s1"] += int(shard["nbr_ptr"][-1])
        counts["s2"] += int(shard["path_ptr"][-1])
        del shard
    C.log(
        f"[merge] totals molecules={counts['molecules']} atoms={counts['atoms']} "
        f"s1={counts['s1']} s2={counts['s2']}"
    )

    alloc: dict[str, np.ndarray] = {}
    for name, kind in _POINTER_SPECS:
        length = counts["molecules"] + 1 if kind == "molecule" else counts["atoms"] + 1
        alloc[name] = np.lib.format.open_memmap(
            tmp / f"{name}.npy", mode="w+", dtype=np.int64, shape=(length,)
        )
    for name, dtype in _VALUE_FIELDS:
        alloc[name] = np.lib.format.open_memmap(
            tmp / f"{name}.npy", mode="w+", dtype=dtype, shape=_field_shape(name, spec, counts)
        )
    gids = np.lib.format.open_memmap(
        tmp / "gids.npy", mode="w+", dtype=np.int64, shape=(counts["molecules"],)
    )

    off = {"molecules": 0, "atoms": 0, "bonds": 0, "s1": 0, "s2": 0}
    with open(tmp / "molecule_ids.jsonl", "w", encoding="utf-8") as fh_ids, open(
        tmp / "smiles.jsonl", "w", encoding="utf-8"
    ) as fh_smi:
        for i, p in enumerate(paths):
            shard = torch_load(p)
            nm = int(shard["n_molecules"])
            for name, kind in _POINTER_SPECS:
                # write position is per-owner (molecule/atom), the addend is the
                # concatenated value offset (atoms/bonds/s1/s2)
                pos = off["molecules"] if kind == "molecule" else off["atoms"]
                add = off[_POINTER_BASE[name]]
                src = np.asarray(shard[name], dtype=np.int64)
                alloc[name][pos : pos + src.size - 1] = src[:-1] + add
            for name, _dtype in _VALUE_FIELDS:
                src = np.asarray(shard[name])
                o = _field_offset(name, off)
                alloc[name][o : o + src.shape[0]] = src
            for mid, smi in zip(shard["molecule_ids"], shard["smiles"]):
                fh_ids.write(json.dumps(mid) + "\n")
                fh_smi.write(json.dumps(smi) + "\n")
            gids[off["molecules"] : off["molecules"] + nm] = np.asarray(shard["gids"], dtype=np.int64)
            off["molecules"] += nm
            off["atoms"] += int(shard["atom_ptr"][-1])
            off["bonds"] += int(shard["bond_ptr"][-1])
            off["s1"] += int(shard["nbr_ptr"][-1])
            off["s2"] += int(shard["path_ptr"][-1])
            del shard
            if i % 8 == 7:  # keep dirty page cache bounded
                for arr in list(alloc.values()) + [gids]:
                    arr.flush()
    alloc["atom_ptr"][counts["molecules"]] = counts["atoms"]
    alloc["bond_ptr"][counts["molecules"]] = counts["bonds"]
    alloc["nbr_ptr"][counts["atoms"]] = counts["s1"]
    alloc["path_ptr"][counts["atoms"]] = counts["s2"]
    for arr in list(alloc.values()) + [gids]:
        arr.flush()
    del alloc, gids

    meta = {
        "feature_spec_name": spec.name,
        "feature_spec_hash": spec.hash(),
        "n_molecules": int(counts["molecules"]),
        "n_atoms_total": int(counts["atoms"]),
        "n_bonds_total": int(counts["bonds"]),
        "n_s1_codes": int(counts["s1"]),
        "n_s2_paths": int(counts["s2"]),
        "split": str(split),
        "labels_present": False,
        "dataset": C.DATASET,
        "created_utc": C.utcnow(),
        "source": "official OGB mapping/mol.csv.gz -> frozen lipo_v1 adapter",
        "n_shards": int(len(paths)),
        "fields": sorted([n for n, _ in _POINTER_SPECS] + [n for n, _ in _VALUE_FIELDS]),
        "format": "npy-directory-v1",
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    files = sorted(out_dir.glob("*"))
    tree_sha = C.sha256_tree(out_dir)
    return {
        "cache": C.track_rel(out_dir),
        "cache_format": "npy-directory-v1",
        "cache_sha256": tree_sha,
        "cache_files": len(files),
        "cache_bytes": int(sum(f.stat().st_size for f in files)),
        "n_molecules": int(counts["molecules"]),
        "n_atoms_total": int(counts["atoms"]),
        "n_bonds_total": int(counts["bonds"]),
        "n_s1_codes": int(counts["s1"]),
        "n_s2_paths": int(counts["s2"]),
        "merge_seconds": float(time.perf_counter() - t0),
        "feature_spec_hash": spec.hash(),
    }


def _check_shard(shard: Mapping[str, Any]) -> None:
    need = {"n_molecules", "atom_ptr", "bond_ptr", "nbr_ptr", "path_ptr", "gids", "molecule_ids", "smiles"}
    missing = need - set(shard)
    if missing:
        raise RuntimeError(f"shard missing keys: {sorted(missing)}")


# ---------------------------------------------------------------------------
# exact S2 count identity (full coverage, no sampling)
# ---------------------------------------------------------------------------


def s2_count_identity(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Check ``#S2(root) == sum_{w in N(root)} (deg(w) - 1)`` for every atom.

    The right-hand side is the frozen historical counting rule
    (``bp_graph.s2_counts_from_formula`` / ``MG.enumerate_s2`` row lengths);
    the left-hand side is what the cache actually stores.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    n_atoms = int(atom_ptr[-1])
    n_bonds = int(bond_ptr[-1])
    if n_bonds:
        bond_batch = np.repeat(np.arange(len(atom_ptr) - 1), np.diff(bond_ptr))
        atom_off = atom_ptr[bond_batch]
        bu = bond_index[:, 0] + atom_off
        bv = bond_index[:, 1] + atom_off
    else:
        bu = np.zeros(0, dtype=np.int64)
        bv = np.zeros(0, dtype=np.int64)
    deg = np.zeros(n_atoms, dtype=np.int64)
    np.add.at(deg, bu, 1)
    np.add.at(deg, bv, 1)
    expected = np.zeros(n_atoms, dtype=np.int64)
    np.add.at(expected, bu, np.maximum(deg[bv] - 1, 0))
    np.add.at(expected, bv, np.maximum(deg[bu] - 1, 0))
    observed = np.diff(path_ptr)
    mismatch = np.flatnonzero(expected != observed)
    return {
        "n_atoms_checked": int(n_atoms),
        "n_bonds_checked": int(n_bonds),
        "n_mismatched_atoms": int(mismatch.size),
        "first_mismatches": mismatch[:10].tolist(),
        "s2_total_observed": int(observed.sum()),
        "s2_total_expected": int(expected.sum()),
        "s1_total_equals_2_bonds": bool(
            int(np.asarray(cache["nbr_ptr"], dtype=np.int64)[-1]) == 2 * n_bonds
        ),
        "ok": bool(mismatch.size == 0) and bool(np.diff(path_ptr).sum() == expected.sum()),
    }


# ---------------------------------------------------------------------------
# feature cache audit
# ---------------------------------------------------------------------------


def cache_label_free(cache: Mapping[str, Any]) -> dict[str, Any]:
    keys = [str(k) for k in cache.keys()]
    bad = sorted(k for k in keys if k.lower() in C.LABEL_KEYS)
    meta = dict(cache.get("meta", {}))
    meta_bad = sorted(k for k in meta if str(k).lower() in C.LABEL_KEYS)
    return {
        "top_level_keys": keys,
        "label_like_keys": bad,
        "label_like_meta_keys": meta_bad,
        "meta_labels_present": meta.get("labels_present", None),
        "ok": not bad and not meta_bad and meta.get("labels_present") is False,
    }


def cache_structure_audit(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    from lcrc_data import validate_cache

    n = int(cache["n_molecules"])
    ptr_ok = all(
        np.asarray(cache[name]).shape == ((n + 1) if kind == "molecule" else (int(cache["atom_ptr"][-1]) + 1),)
        for name, kind in _POINTER_SPECS
    )
    finite_ok = all(
        np.isfinite(np.asarray(cache[name])).all()
        for name in ("atom_feat", "atom_scalar", "bond_feat", "global")
    )
    ranges = {
        "atom_type_in_vocabulary": bool(
            (np.asarray(cache["atom_type"]) >= 0).all() and (np.asarray(cache["atom_type"]) < spec.n_atom_types).all()
        ),
        "bond_type_in_vocabulary": bool(
            (np.asarray(cache["bond_type"]) >= 1).all() and (np.asarray(cache["bond_type"]) <= spec.n_bond_types).all()
        ),
        "nbr_code_in_vocabulary": bool(
            (np.asarray(cache["nbr_code"]) >= 0).all() and (np.asarray(cache["nbr_code"]) < spec.n_nbr_codes).all()
        ),
        "path_cat_in_vocabulary": bool(
            (np.asarray(cache["path_cat"]) >= 0).all() and (np.asarray(cache["path_cat"]) < spec.n_path_codes).all()
        ),
        "path_end63_in_vocabulary": bool(
            (np.asarray(cache["path_end63"]) >= 0).all() and (np.asarray(cache["path_end63"]) < spec.n_nbr_codes).all()
        ),
        "path_closure_in_0_1": bool(
            (np.asarray(cache["path_closure"]) >= 0).all() and (np.asarray(cache["path_closure"]) <= 1).all()
        ),
    }
    shapes = {
        "atom_feat": list(np.asarray(cache["atom_feat"]).shape) == [int(cache["atom_ptr"][-1]), spec.atom_feat_dim],
        "atom_scalar": list(np.asarray(cache["atom_scalar"]).shape) == [int(cache["atom_ptr"][-1]), spec.n_scalars],
        "bond_feat": list(np.asarray(cache["bond_feat"]).shape) == [int(cache["bond_ptr"][-1]), spec.bond_feat_dim],
        "global": list(np.asarray(cache["global"]).shape) == [n, spec.global_dim],
        "bond_index": list(np.asarray(cache["bond_index"]).shape) == [int(cache["bond_ptr"][-1]), 2],
    }
    validate = validate_cache(cache, spec)
    return {
        "n_molecules": n,
        "n_atoms_total": int(cache["atom_ptr"][-1]),
        "n_bonds_total": int(cache["bond_ptr"][-1]),
        "n_s1_codes": int(cache["nbr_ptr"][-1]),
        "n_s2_paths": int(cache["path_ptr"][-1]),
        "pointer_shapes_ok": bool(ptr_ok),
        "finite_values": bool(finite_ok),
        "shapes_ok": shapes,
        "value_ranges": ranges,
        "validate_cache": validate,
        "gids_is_identity": bool(np.array_equal(np.asarray(cache["gids"]), np.arange(n, dtype=np.int64))),
        "feature_spec_hash": spec.hash(),
        "formula_s2": s2_count_identity(cache),
        "ok": bool(
            ptr_ok
            and finite_ok
            and all(shapes.values())
            and all(ranges.values())
            and validate["n_molecules_checked"] == n
        ),
    }


def feature_cache_audit(
    cache: Mapping[str, Any],
    spec,
    *,
    sample_rows: Sequence[int],
    build_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Label-free cache checks plus exact S2 content comparison on a sample."""
    import lcrc_data as PREP_DATA
    import mb2_rdkit as R
    from molecular_graph import analyse_graph, enumerate_s1, enumerate_s2
    import bp_graph

    structure = cache_structure_audit(cache, spec)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    rows = [int(r) for r in sample_rows]
    content_ok = True
    checked = 0
    mismatches: list[dict[str, Any]] = []
    modes = {"strict": 0, "unsanitized_fallback": 0}
    formula_ok = True
    for row in rows:
        smi = cache["smiles"][row]
        try:
            raw, mode = R.smiles_to_raw_row_relaxed(smi, row)
            graph = R.raw_row_to_graph(raw, row, spec)
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            mismatches.append({"row": row, "error": f"{type(exc).__name__}: {exc}"})
            content_ok = False
            continue
        modes[mode] += 1
        analysis = analyse_graph(graph)
        s2 = enumerate_s2(graph.atom_type, analysis.adj, spec.n_bond_types, analysis.dist)
        counts = np.asarray([len(r) for r in s2], dtype=np.int64)
        stored_counts = np.diff(path_ptr[atom_ptr[row] : atom_ptr[row + 1] + 1])
        if not np.array_equal(counts, stored_counts):
            content_ok = False
            mismatches.append({"row": row, "kind": "count", "enumerated": int(counts.sum()), "stored": int(stored_counts.sum())})
            continue
        formula = bp_graph.s2_counts_from_formula(analysis.adj)
        if not np.array_equal(formula, stored_counts):
            formula_ok = False
            mismatches.append({"row": row, "kind": "formula_count"})
        flat = [rec for root in s2 for rec in root]
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        lo, hi = int(path_ptr[a0]), int(path_ptr[a1])
        stored = {
            "path_cat": np.asarray(cache["path_cat"])[lo:hi],
            "path_end63": np.asarray(cache["path_end63"])[lo:hi],
            "path_end": np.asarray(cache["path_end"])[lo:hi],
            "path_closure": np.asarray(cache["path_closure"])[lo:hi],
        }
        want = {
            "path_cat": np.asarray([rec["path_cat"] for rec in flat], dtype=np.int32),
            "path_end63": np.asarray([rec["end63"] for rec in flat], dtype=np.int32),
            "path_end": np.asarray([rec["end"] for rec in flat], dtype=np.int64),
            "path_closure": np.asarray([rec["closure"] for rec in flat], dtype=np.int8),
        }
        for key in want:
            if not np.array_equal(stored[key], want[key]):
                content_ok = False
                mismatches.append({"row": row, "kind": f"content:{key}"})
        # S1 identity on the sample
        s1 = enumerate_s1(graph.atom_type, analysis.adj, spec.n_bond_types)
        flat_s1 = [code for root in s1 for code in root]
        s1_lo, s1_hi = int(nbr_ptr[a0]), int(nbr_ptr[a1])
        if not np.array_equal(np.asarray(cache["nbr_code"])[s1_lo:s1_hi], np.asarray(flat_s1, dtype=np.int32)):
            content_ok = False
            mismatches.append({"row": row, "kind": "content:nbr_code"})
        checked += 1
    payload = {
        "schema": "lcrc-pcba-feature-cache-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "atom_feat_dim": spec.atom_feat_dim,
            "bond_feat_dim": spec.bond_feat_dim,
            "global_dim": spec.global_dim,
            "n_scalars": spec.n_scalars,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
        },
        "label_free": cache_label_free(cache),
        "structure": structure,
        "s2_exactness_sample": {
            "n_requested": len(rows),
            "n_checked": checked,
            "rows": rows,
            "parse_modes": modes,
            "upstream_enumeration_matches_cache": bool(content_ok),
            "formula_count_matches_cache": bool(formula_ok),
            "s2_records_compared": ["path_cat", "path_end63", "path_end", "path_closure"],
            "s1_records_compared": ["nbr_code"],
            "mismatches": mismatches[:20],
        },
        "build": {
            "summary": C.track_rel(C.ANALYSIS / "pcba_cache_build.json"),
            "n_rows_requested": None if build_summary is None else build_summary.get("n_rows_requested"),
            "wall_seconds": None if build_summary is None else build_summary.get("wall_seconds"),
            "parse_modes": None if build_summary is None else build_summary.get("parse_modes"),
        },
        "no_truncation": {
            "policy": "every S2 record of every molecule is materialised; no sampling, cap or approximation",
            "s2_total": int(path_ptr[-1]),
            "formula_identity_all_atoms": structure["formula_s2"]["ok"],
        },
    }
    payload["gates"] = {
        "cache_is_label_free": payload["label_free"]["ok"],
        "structure_ok": structure["ok"],
        "s2_all_atom_formula_identity": structure["formula_s2"]["ok"],
        "s2_upstream_content_equality_on_sample": bool(content_ok and formula_ok and checked == len(rows)),
        "no_truncation": True,
    }
    payload["status"] = "OK" if all(payload["gates"].values()) else "FAILED"
    return payload


# ---------------------------------------------------------------------------
# historical (MolHIV) S2 cross-check
# ---------------------------------------------------------------------------


def historical_s2_crosscheck(cache: Mapping[str, Any], spec, *, n_molecules: int = 8) -> dict[str, Any]:
    """Re-derive the *historical* MolHIV cache's S2 counts and records."""
    from feature_spec import load_spec
    from molecular_graph import analyse_graph, enumerate_s2
    import lcrc_data as PREP_DATA
    import mb2_rdkit as R

    hist_cache = C.BP_CACHE / "molhiv.pt"
    payload: dict[str, Any] = {
        "schema": "lcrc-pcba-historical-s2-crosscheck-v1",
        "historical_cache": C.track_rel(hist_cache),
        "exists": hist_cache.is_file(),
    }
    if not hist_cache.is_file():
        payload["status"] = "UNAVAILABLE"
        payload["reason"] = "historical MolHIV cache is absent on this host"
        return payload
    payload["historical_cache_sha256"] = C.sha256_file(hist_cache)
    hist = torch_load(hist_cache)
    payload["historical_formula_identity"] = s2_count_identity(hist)
    hist_spec_path = C.BP_ANALYSIS / "molhiv_spec.json"
    if not hist_spec_path.is_file():
        payload["status"] = "UNAVAILABLE"
        payload["reason"] = f"missing historical spec {C.track_rel(hist_spec_path)}"
        return payload
    hist_spec = load_spec(hist_spec_path)
    payload["historical_spec_hash"] = hist_spec.hash()
    payload["spec_hash_matches_this_round"] = hist_spec.hash() == spec.hash()
    mapping = C.OGB_ROOT / "ogbg_molhiv" / "mapping" / "mol.csv.gz"
    payload["historical_mapping"] = C.track_rel(mapping)
    if not mapping.is_file():
        payload["status"] = "UNAVAILABLE"
        payload["reason"] = "historical MolHIV mapping/mol.csv.gz is absent"
        return payload
    import pandas as pd

    df = pd.read_csv(mapping)
    smiles = [str(x) for x in df["smiles"].tolist()]
    atom_ptr = np.asarray(hist["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(hist["path_ptr"], dtype=np.int64)
    step = max(len(hist["molecule_ids"]) // max(n_molecules, 1), 1)
    rows = list(range(0, len(hist["molecule_ids"]), step))[:n_molecules]
    comparisons: list[dict[str, Any]] = []
    all_ok = True
    for row in rows:
        smi = smiles[int(hist["gids"][row])] if "gids" in hist else smiles[row]
        raw, _mode = R.smiles_to_raw_row_relaxed(smi, int(hist["gids"][row]) if "gids" in hist else row)
        graph = R.raw_row_to_graph(raw, int(hist["gids"][row]) if "gids" in hist else row, hist_spec)
        analysis = analyse_graph(graph)
        s2 = enumerate_s2(graph.atom_type, analysis.adj, hist_spec.n_bond_types, analysis.dist)
        flat = [rec for root in s2 for rec in root]
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        lo, hi = int(path_ptr[a0]), int(path_ptr[a1])
        ok = (
            np.array_equal(np.asarray(hist["path_cat"])[lo:hi], np.asarray([r["path_cat"] for r in flat], dtype=np.int32))
            and np.array_equal(
                np.asarray(hist["path_end63"])[lo:hi], np.asarray([r["end63"] for r in flat], dtype=np.int32)
            )
            and np.array_equal(np.asarray(hist["path_end"])[lo:hi], np.asarray([r["end"] for r in flat], dtype=np.int64))
            and np.array_equal(
                np.asarray(hist["path_closure"])[lo:hi], np.asarray([r["closure"] for r in flat], dtype=np.int8)
            )
        )
        all_ok = all_ok and bool(ok)
        comparisons.append({"cache_row": int(row), "n_s2": int(len(flat)), "exact_match": bool(ok)})
    payload["sample_comparisons"] = comparisons
    payload["sample_exact_match"] = bool(all_ok)
    payload["status"] = (
        "OK"
        if payload["historical_formula_identity"]["ok"] and all_ok and payload.get("spec_hash_matches_this_round")
        else "FAILED"
    )
    return payload


def torch_load(path: Path | str) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def load_cache(path: Path | str) -> dict[str, Any]:
    """Load a merged cache: memory-mapped arrays, no full materialisation."""
    path = Path(path)
    if path.is_file():  # legacy single-file pickle
        return torch_load(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    cache: dict[str, Any] = {
        "meta": meta,
        "n_molecules": int(meta["n_molecules"]),
        "split": str(meta["split"]),
    }
    for name in meta["fields"]:
        cache[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    cache["gids"] = np.load(path / "gids.npy", mmap_mode="r")
    with open(path / "molecule_ids.jsonl", encoding="utf-8") as fh:
        cache["molecule_ids"] = [json.loads(line) for line in fh]
    with open(path / "smiles.jsonl", encoding="utf-8") as fh:
        cache["smiles"] = [json.loads(line) for line in fh]
    return cache


__all__ = [
    "InputAdapterBlocked",
    "build_label_artifact",
    "build_shards",
    "cache_label_free",
    "cache_structure_audit",
    "data_audit",
    "feature_cache_audit",
    "historical_s2_crosscheck",
    "load_cache",
    "load_official",
    "merge_shards",
    "s2_count_identity",
    "split_payload",
    "task_label_counts",
]
