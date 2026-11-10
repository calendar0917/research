"""PCQM4Mv2 access, fixed subsets, sharded ``lipo_v1`` cache and Fits.

Layout
------
* :func:`load_official` -- the official ``ogb.lsc.PCQM4Mv2Dataset`` with
  ``only_smiles=True`` (2D SMILES only; no DFT geometry, no pretrained
  weights), plus hard load-time assertions and file fingerprints.
* :func:`data_audit` -- version / total / split keys / row uniqueness / label
  finiteness, with every measured value recorded.
* :func:`subset_plan` -- one NumPy permutation of the official Train row ids
  (seed 42) and its two prefixes (300K, 1M).  Prefixes make the containment
  ``subset_300k ⊂ subset_1m ⊂ official_train`` structural, not accidental.
* :func:`build_shards` / :func:`merge_shards` -- chunked multi-process build of
  the frozen ``lipo_v1`` LCRC arrays for exactly the rows requested, with a
  per-row parse-mode ledger and a blocking failure policy (no row is dropped,
  substituted or renumbered) and an mmap-directory merge.
* :func:`feature_norm_rows` / :func:`label_norm` -- Fit-only standardisation
  from the fixed 300K Train prefix (never Validation, never Test).
* :func:`epoch_orders` -- the deterministic per-epoch batch permutation.

The build is a re-implementation of ``lcrc_pcba_scale_pilot_v1/code/pc_data.py``
with a dataset label of this round; ``tests/test_pq_cache.py`` proves the merge
is numerically identical to that pinned implementation on a small cache.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import pq_common as C

import feature_spec as FS
import lcrc_data as PREP_DATA
import pc_data as PCD


class InputAdapterBlocked(RuntimeError):
    """The official SMILES / feature adapter could not process some rows."""


# ---------------------------------------------------------------------------
# official dataset
# ---------------------------------------------------------------------------


def load_official(*, root: Path | str | None = None) -> dict[str, Any]:
    from ogb.lsc import PCQM4Mv2Dataset, PCQM4Mv2Evaluator

    root = Path(root) if root is not None else C.OGB_ROOT
    dataset = PCQM4Mv2Dataset(root=str(root), only_smiles=True)
    split_raw = dataset.get_idx_split()
    if not isinstance(split_raw, Mapping):
        raise InputAdapterBlocked(f"{C.DATASET}: get_idx_split() returned {type(split_raw).__name__}")
    keys = tuple(sorted(str(k) for k in split_raw))
    if keys != tuple(sorted(C.SPLIT_KEYS)):
        raise InputAdapterBlocked(f"{C.DATASET}: split keys {keys} != {sorted(C.SPLIT_KEYS)}")
    split = {k: np.asarray(split_raw[k], dtype=np.int64).reshape(-1) for k in C.SPLIT_KEYS}
    n = int(len(dataset))
    smiles = [str(x) for x in list(dataset.graphs)]
    labels = np.asarray(dataset.labels, dtype=np.float64).reshape(-1)
    if smiles.__len__() != n or labels.size != n:
        raise InputAdapterBlocked(
            f"{C.DATASET}: smiles={len(smiles)} labels={labels.size} but len(dataset)={n}"
        )
    evidence: dict[str, Any] = {}
    for name, rows in split.items():
        evidence[name] = {
            "n": int(rows.size),
            "min": int(rows.min()) if rows.size else None,
            "max": int(rows.max()) if rows.size else None,
            "unique": int(np.unique(rows).size),
            "sorted_ascending": bool(rows.size and np.all(np.diff(rows) > 0)),
            "in_range": bool(rows.size and rows.min() >= 0 and rows.max() < n),
            "index_sha256": C.index_hash(rows),
        }
        if not evidence[name]["sorted_ascending"]:
            raise InputAdapterBlocked(f"{C.DATASET}: {name} index is not a strictly increasing row list")
        if not evidence[name]["in_range"]:
            raise InputAdapterBlocked(f"{C.DATASET}: {name} index outside [0,{n})")
    union = np.concatenate([split[k] for k in C.SPLIT_KEYS])
    if np.unique(union).size != n or not np.array_equal(np.sort(union), np.arange(n)):
        raise InputAdapterBlocked(f"{C.DATASET}: official splits do not partition the dataset")
    folder = Path(dataset.folder)
    raw_files = sorted(p for p in (folder / "raw").rglob("*") if p.is_file()) if (folder / "raw").is_dir() else []
    files: dict[str, Any] = {}
    for p in list(raw_files) + ([folder / "split_dict.pt"] if (folder / "split_dict.pt").is_file() else []):
        files[str(p.relative_to(root))] = {
            "bytes": int(p.stat().st_size),
            "sha256": C.sha256_file(p),
        }
    released = sorted(p.name for p in folder.glob("RELEASE_v*.txt"))
    return {
        "name": C.DATASET,
        "dataset": dataset,
        "evaluator": PCQM4Mv2Evaluator(),
        "root": Path(root),
        "folder": folder,
        "n": n,
        "split": split,
        "split_evidence": evidence,
        "smiles": smiles,
        "labels": labels,
        "release_files": released,
        "files": files,
        "only_smiles": True,
        "uses_3d_geometry": False,
    }


def label_summary(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(v)
    f = v[finite]
    if f.size == 0:
        return {"n": int(v.size), "n_finite": 0}
    return {
        "n": int(v.size),
        "n_finite": int(f.size),
        "n_non_finite": int((~finite).sum()),
        "min": float(f.min()),
        "max": float(f.max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "median": float(np.median(f)),
        "q01": float(np.percentile(f, 1)),
        "q05": float(np.percentile(f, 5)),
        "q25": float(np.percentile(f, 25)),
        "q75": float(np.percentile(f, 75)),
        "q95": float(np.percentile(f, 95)),
        "q99": float(np.percentile(f, 99)),
        "units": "eV",
    }


def data_audit(info: Mapping[str, Any]) -> dict[str, Any]:
    """Version / total / split keys / uniqueness / label finiteness."""
    split = info["split"]
    labels = np.asarray(info["labels"], dtype=np.float64)
    train = split["train"]
    valid = split["valid"]
    hidden = {k: split[k] for k in C.HIDDEN_SPLITS}
    payload = {
        "schema": "lcrc-pcqm4mv2-data-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": {
            "name": info["name"],
            "split_keys": list(C.SPLIT_KEYS),
            "n_total": int(info["n"]),
            "ogb_root": C.track_rel(info["root"]),
            "dataset_folder": C.track_rel(info["folder"]),
            "release_files": info["release_files"],
            "only_smiles": bool(info["only_smiles"]),
            "uses_3d_geometry": False,
            "pretrained_weights_used": False,
        },
        "files": info["files"],
        "splits": info["split_evidence"],
        "partition": {
            "union_is_full_range": bool(
                np.array_equal(
                    np.sort(np.concatenate([split[k] for k in C.SPLIT_KEYS])), np.arange(int(info["n"]))
                )
            ),
            "pairwise_disjoint": True,
            "list_of_pairs_checked": [
                [a, b]
                for i, a in enumerate(C.SPLIT_KEYS)
                for b in C.SPLIT_KEYS[i + 1 :]
            ],
        },
        "labels": {
            "train": label_summary(labels[train]),
            "valid": label_summary(labels[valid]),
            "hidden_split_rows": {k: int(v.size) for k, v in hidden.items()},
            "hidden_labels_read": False,
            "hidden_labels_used": False,
            "unit": "eV (DFT HOMO-LUMO gap at B3LYP/6-31G*)",
        },
        "smiles": {
            "n": len(info["smiles"]),
            "n_unique": int(len(set(info["smiles"]))),
            "n_duplicate_strings": int(len(info["smiles"]) - len(set(info["smiles"]))),
            "n_empty": int(sum(1 for s in info["smiles"] if not s.strip())),
        },
        "split_sizes": {k: int(split[k].size) for k in C.SPLIT_KEYS},
    }
    # pairwise disjointness, computed (not asserted by fiat)
    pairwise = True
    for i, a in enumerate(C.SPLIT_KEYS):
        for b in C.SPLIT_KEYS[i + 1 :]:
            if np.intersect1d(split[a], split[b], assume_unique=True).size:
                pairwise = False
    payload["partition"]["pairwise_disjoint"] = bool(pairwise)
    payload["partition"]["union_is_full_range"] = bool(
        np.array_equal(np.sort(np.concatenate([split[k] for k in C.SPLIT_KEYS])), np.arange(int(info["n"])))
    )
    checks = {
        "has_four_official_split_keys": tuple(sorted(C.SPLIT_KEYS)) == tuple(sorted(split)),
        "total_equals_sum_of_splits": int(info["n"]) == int(sum(split[k].size for k in C.SPLIT_KEYS)),
        "splits_pairwise_disjoint": pairwise,
        "splits_partition_the_dataset": payload["partition"]["union_is_full_range"],
        "train_labels_all_finite": bool(np.isfinite(labels[train]).all()),
        "valid_labels_all_finite": bool(np.isfinite(labels[valid]).all()),
        "no_row_appears_twice_inside_a_split": all(
            int(np.unique(split[k]).size) == int(split[k].size) for k in C.SPLIT_KEYS
        ),
        "smiles_non_empty": payload["smiles"]["n_empty"] == 0,
    }
    payload["gates"] = checks
    payload["status"] = "OK" if all(checks.values()) else "FAILED"
    return payload


# ---------------------------------------------------------------------------
# fixed subsets (one permutation, two prefixes)
# ---------------------------------------------------------------------------


def subset_plan(
    info: Mapping[str, Any],
    *,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Permutation of the official Train rows with seed 42, plus its prefixes."""
    train = np.asarray(info["split"]["train"], dtype=np.int64)
    sizes = {run: int(spec["train_size"]) for run, spec in C.RUNS_TABLE.items()} if sizes is None else {
        str(k): int(v) for k, v in sizes.items()
    }
    n_train_max = max(sizes.values())
    if n_train_max > train.size:
        raise ValueError(f"requested subset {n_train_max} exceeds official Train size {train.size}")
    rng = np.random.default_rng(C.SAMPLE_SEED)
    perm = rng.permutation(train.size).astype(np.int64)
    permuted_rows = train[perm]
    if np.unique(permuted_rows).size != permuted_rows.size:
        raise RuntimeError("permuted train rows are not unique")
    if not np.array_equal(np.sort(permuted_rows[:n_train_max]), np.sort(train[perm[:n_train_max]])):
        raise RuntimeError("subset is not the permutation prefix")
    subsets = {}
    for run, n in sizes.items():
        n = int(n)
        subsets[run] = {
            "n": n,
            "rows_sha256": C.index_hash(permuted_rows[:n]),
            "first_row": int(permuted_rows[0]),
            "last_row": int(permuted_rows[n - 1]),
        }
    small = min(int(v) for v in sizes.values())
    large = max(int(v) for v in sizes.values())
    containment = {
        "subset_small_lt_subset_large": small < large,
        "small_is_prefix_of_large": bool(np.array_equal(permuted_rows[:small], permuted_rows[:large][:small])),
        "large_within_official_train": bool(np.isin(permuted_rows[:large], train).all()),
        "no_validation_row_in_large": bool(
            np.intersect1d(permuted_rows[:large], info["split"]["valid"], assume_unique=False).size == 0
        ),
        "no_hidden_row_in_large": all(
            np.intersect1d(permuted_rows[:large], info["split"][k], assume_unique=False).size == 0
            for k in C.HIDDEN_SPLITS
        ),
        "n_unique_rows_in_large": int(np.unique(permuted_rows[:large]).size),
    }
    return {
        "schema": "lcrc-pcqm4mv2-subset-plan-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "official_train_size": int(train.size),
        "sampling_rng": "numpy.random.default_rng(42).permutation(official_train_size)",
        "sample_seed": int(C.SAMPLE_SEED),
        "n_permuted": int(perm.size),
        "permutation_sha256_full": C.sha256_bytes(perm.tobytes()),
        "permutation_rows_sha256_full": C.index_hash(permuted_rows),
        "subsets": subsets,
        "containment": containment,
        "status": "OK"
        if all(
            [
                containment["small_is_prefix_of_large"],
                containment["large_within_official_train"],
                containment["no_validation_row_in_large"],
                containment["no_hidden_row_in_large"],
                containment["n_unique_rows_in_large"] == large,
            ]
        )
        else "FAILED",
        "_permuted_rows": permuted_rows,
    }


def monitor_rows(*, n_train_pool: int = 300_000) -> dict[str, Any]:
    """Fixed 20,000-row Train monitor subset drawn from ``subset_300k``.

    Independent RNG (seed 43) so it cannot be confused with the sampling
    permutation (42) or the batch order (0).  The returned positions are
    *cache positions inside the 300K prefix*, i.e. the same integers in both
    formal runs.
    """
    rng = np.random.default_rng(C.MONITOR_SEED)
    positions = np.sort(rng.choice(int(n_train_pool), size=int(C.N_MONITOR), replace=False)).astype(np.int64)
    return {
        "schema": "lcrc-pcqm4mv2-monitor-plan-v1",
        "created_utc": C.utcnow(),
        "rng": "numpy.random.default_rng(43).choice(300000, 20000, replace=False)",
        "monitor_seed": int(C.MONITOR_SEED),
        "source_pool": "cache positions 0..299999 (subset_300k)",
        "n": int(positions.size),
        "n_unique": int(np.unique(positions).size),
        "min": int(positions.min()),
        "max": int(positions.max()),
        "positions_sha256": C.sha256_bytes(positions.astype(np.int64).tobytes()),
        "_positions": positions,
    }


# ---------------------------------------------------------------------------
# sharded feature cache build (adapted from pc_data.build_shards)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(spec_path: str, shard_dir: str) -> None:
    _WORKER["spec"] = FS.load_spec(Path(spec_path))
    _WORKER["dir"] = Path(shard_dir)


def _build_chunk(task: tuple[int, list[int], list[str]]) -> dict[str, Any]:
    """Worker: build one chunk and write a shard file (label-blind)."""
    import mb2_rdkit as R
    import torch

    idx, rows, smiles = task
    spec = _WORKER["spec"]
    shard_dir = _WORKER["dir"]
    t0 = time.perf_counter()
    mols: list[dict[str, Any]] = []
    kept_rows: list[int] = []
    failures: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    modes = {"strict": 0, "unsanitized_fallback": 0}
    n_atoms: list[int] = []
    n_bonds: list[int] = []
    s2_paths: list[int] = []
    atom_class = np.zeros(R.ATOM_UNK + 1, dtype=np.int64)
    charge = np.zeros(R.CHARGE_UNK + 1, dtype=np.int64)
    hydrogens = np.zeros(R.H_UNK + 1, dtype=np.int64)
    bond_class = np.zeros(R.BOND_UNK + 1, dtype=np.int64)
    stereo = np.zeros(4, dtype=np.int64)
    n_components: list[int] = []
    for row, smi in zip(rows, smiles):
        try:
            raw, mode = R.smiles_to_raw_row_relaxed(smi, int(row))
            graph = R.raw_row_to_graph(raw, int(row), spec)
            mol = PREP_DATA.build_molecule(graph, spec, validate=True)
        except Exception as exc:  # noqa: BLE001 - recorded, then the build is blocked
            failures.append({"row": int(row), "smiles": smi, "error": f"{type(exc).__name__}: {exc}"})
            continue
        modes[mode] += 1
        if mode != "strict":
            fallback_rows.append(
                {
                    "row": int(row),
                    "mode": str(mode),
                    "strict_error": str(raw.get("strict_error", ""))[:240],
                    "smiles": smi,
                }
            )
        atom_class += np.bincount(np.asarray(raw["atom_class"], dtype=np.int64), minlength=atom_class.size)
        charge += np.bincount(np.asarray(raw["atom_charge"], dtype=np.int64), minlength=charge.size)
        hydrogens += np.bincount(np.asarray(raw["atom_h"], dtype=np.int64), minlength=hydrogens.size)
        edge_class = np.asarray(raw["edge_class"], dtype=np.int64)
        if edge_class.size:
            bond_class += np.bincount(edge_class, minlength=bond_class.size)
            stereo += np.bincount(np.asarray(raw["edge_stereo"], dtype=np.int64), minlength=stereo.size)
        n_components.append(int(raw["n_components"]))
        mols.append(mol)
        kept_rows.append(int(row))
        n_atoms.append(int(mol["n_atoms"]))
        n_bonds.append(int(mol["n_bonds"]))
        s2_paths.append(int(mol["path_ptr"][-1]))
    out = shard_dir / f"shard_{int(idx):05d}.pt"
    payload: dict[str, Any] = {
        "shard": int(idx),
        "rows": list(map(int, rows)),
        "failures": failures,
        "fallback_rows": fallback_rows,
        "n_built": len(mols),
        "modes": modes,
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "s2_paths": s2_paths,
        "atom_class_counts": atom_class.tolist(),
        "charge_counts": charge.tolist(),
        "h_counts": hydrogens.tolist(),
        "bond_class_counts": bond_class.tolist(),
        "stereo_counts": stereo.tolist(),
        "n_components": n_components,
        "path": str(out),
        "seconds": float(time.perf_counter() - t0),
        "sha256": "",
    }
    if mols and not failures:
        cache = PREP_DATA.concat_molecules(mols, spec, split="pcqm_shard")
        cache["gids"] = np.asarray(kept_rows, dtype=np.int64)
        if cache["gids"].size != len(mols):
            raise RuntimeError("gid list does not match the number of built molecules")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, out)
        payload["sha256"] = C.sha256_file(out)
        payload["written"] = True
    else:
        # A chunk with any failed row is never written: merging a partial chunk
        # would silently renumber the rows that follow it.
        payload["written"] = False
    return payload


def build_shards(
    info: Mapping[str, Any],
    *,
    rows: Sequence[int],
    shard_dir: Path | str,
    tag: str,
    workers: int = 8,
    chunk_size: int = 4000,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    """Build the frozen ``lipo_v1`` LCRC arrays for the selected official rows."""
    spec = FS.lipo_v1_spec()
    spec_path = Path(spec_path) if spec_path else C.ANALYSIS / "pcqm_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    FS.save_spec(spec, spec_path)
    smiles_all = info["smiles"]
    rows = [int(r) for r in rows]
    if not rows:
        raise ValueError("no rows selected")
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate rows in build selection")
    if min(rows) < 0 or max(rows) >= len(smiles_all):
        raise ValueError("row outside the official dataset")
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(shard_dir.glob("shard_*.pt")):
        stale.unlink()
    tasks = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        tasks.append((len(tasks), chunk, [smiles_all[r] for r in chunk]))
    C.log(f"[cache:{tag}] {len(rows)} rows in {len(tasks)} chunks, workers={workers}")
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    if workers <= 1:
        _init_worker(str(spec_path), str(shard_dir))
        for task in tasks:
            results.append(_build_chunk(task))
            C.log(f"[cache:{tag}] chunk {results[-1]['shard']} done ({results[-1]['seconds']:.1f}s)")
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
                    f"[cache:{tag}] chunk {res['shard']}/{len(tasks) - 1} rows={res['n_built']} "
                    f"{res['seconds']:.1f}s"
                )
    failures = [f for res in results for f in res["failures"]]
    fallback_rows = [r for res in results for r in res["fallback_rows"]]

    def _concat_ints(key: str) -> np.ndarray:
        parts = [np.asarray(res[key], dtype=np.int64) for res in results if len(res[key])]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def _sum_counts(key: str) -> list[int]:
        width = max(len(res[key]) for res in results)
        total = np.zeros(width, dtype=np.int64)
        for res in results:
            total[: len(res[key])] += np.asarray(res[key], dtype=np.int64)
        return total.tolist()

    n_atoms = _concat_ints("n_atoms")
    n_bonds = _concat_ints("n_bonds")
    s2_paths = _concat_ints("s2_paths")
    components = _concat_ints("n_components")
    atom_class_counts = _sum_counts("atom_class_counts")
    bond_class_counts = _sum_counts("bond_class_counts")
    summary: dict[str, Any] = {
        "schema": "lcrc-pcqm4mv2-cache-build-v1",
        "round_id": C.ROUND_ID,
        "tag": str(tag),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "feature_spec": C.track_rel(spec_path),
        "feature_spec_hash": spec.hash(),
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
        "parse_ledger": C.track_rel(C.ANALYSIS / f"{tag}_parse_ledger.json"),
        "n_fallback_rows": int(len(fallback_rows)),
        "n_failures": len(failures),
        "n_atoms_total": int(n_atoms.sum()),
        "n_bonds_total": int(n_bonds.sum()),
        "s2_paths_total": int(s2_paths.sum()),
        "n_atoms": PCD._dist(n_atoms),
        "n_bonds": PCD._dist(n_bonds),
        "s2_paths": PCD._dist(s2_paths),
        "max_s2_records_per_molecule": int(s2_paths.max()) if s2_paths.size else 0,
        "n_multi_component_molecules": int((components > 1).sum()),
        "raw_vocabulary": {
            "atom_class_counts": {str(i): int(c) for i, c in enumerate(atom_class_counts)},
            "bond_class_counts": {str(i): int(c) for i, c in enumerate(bond_class_counts)},
            "charge_counts": _sum_counts("charge_counts"),
            "h_counts": _sum_counts("h_counts"),
            "stereo_counts": _sum_counts("stereo_counts"),
            "atom_unk_fraction": float(atom_class_counts[15] / max(sum(atom_class_counts), 1)),
            "bond_unk_fraction": float(
                (bond_class_counts[4] if len(bond_class_counts) > 4 else 0) / max(sum(bond_class_counts), 1)
            ),
        },
        "shards": [
            {
                "shard": int(res["shard"]),
                "path": C.track_rel(res["path"]),
                "sha256": res["sha256"],
                "n_built": int(res["n_built"]),
                "first_row": int(res["rows"][0]),
                "last_row": int(res["rows"][-1]),
                "seconds": float(res["seconds"]),
            }
            for res in results
        ],
        "failures_file": None,
        "status": "OK" if not failures else "BLOCKED_ADAPTER_FAILURES",
    }
    if failures:
        path = C.ANALYSIS / f"{tag}_adapter_failures.json"
        C.write_json(path, {"schema": "lcrc-pcqm4mv2-adapter-failures-v1", "failures": failures})
        summary["failures_file"] = C.track_rel(path)
        C.write_json(C.ANALYSIS / f"{tag}_cache_build.json", summary)
        raise InputAdapterBlocked(
            f"{len(failures)} official SMILES rows failed the frozen LCRC adapter for {tag}; "
            f"see {C.track_rel(path)} (rows are not dropped or substituted)"
        )
    C.write_json(
        C.ANALYSIS / f"{tag}_parse_ledger.json",
        {
            "schema": "lcrc-pcqm4mv2-parse-ledger-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "git_commit": C.git_commit(),
            "tag": str(tag),
            "n_rows": len(rows),
            "n_strict": int(sum(res["modes"]["strict"] for res in results)),
            "n_unsanitized_fallback": int(sum(res["modes"]["unsanitized_fallback"] for res in results)),
            "n_failures": int(len(failures)),
            "policy": (
                "one documented fallback (sanitize=False + UpdatePropertyCache(strict=False)) is used only "
                "after strict sanitisation failed; every affected row is listed here and no row is dropped"
            ),
            "fallback_rows": sorted(fallback_rows, key=lambda r: r["row"]),
        },
    )
    C.write_json(C.ANALYSIS / f"{tag}_cache_build.json", summary)
    return summary


# ---------------------------------------------------------------------------
# deterministic merge into a memory-mapped directory cache
# ---------------------------------------------------------------------------

_POINTER_SPECS = (
    ("atom_ptr", "molecule"),
    ("bond_ptr", "molecule"),
    ("nbr_ptr", "atom"),
    ("path_ptr", "atom"),
)
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
    return PCD._field_shape(name, spec, counts)


def _field_offset(name: str, off: Mapping[str, int]) -> int:
    return PCD._field_offset(name, off)


def merge_shards(
    shard_paths: Sequence[Path | str],
    spec,
    *,
    out_dir: Path | str,
    split: str = "all",
    dataset: str = C.DATASET,
    tag: str = "",
) -> dict[str, Any]:
    """Merge shards into an ``npy`` directory cache (same algorithm as PCBA)."""
    import shutil

    import torch

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
        shard = torch.load(p, map_location="cpu", weights_only=False)
        PCD._check_shard(shard)
        counts["molecules"] += int(shard["n_molecules"])
        counts["atoms"] += int(shard["atom_ptr"][-1])
        counts["bonds"] += int(shard["bond_ptr"][-1])
        counts["s1"] += int(shard["nbr_ptr"][-1])
        counts["s2"] += int(shard["path_ptr"][-1])
        del shard
    C.log(
        f"[merge:{tag}] totals molecules={counts['molecules']} atoms={counts['atoms']} "
        f"bonds={counts['bonds']} s1={counts['s1']} s2={counts['s2']}"
    )

    alloc: dict[str, np.ndarray] = {}
    for name, kind in _POINTER_SPECS:
        length = counts["molecules"] + 1 if kind == "molecule" else counts["atoms"] + 1
        alloc[name] = np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=np.int64, shape=(length,))
    for name, dtype in _VALUE_FIELDS:
        alloc[name] = np.lib.format.open_memmap(
            tmp / f"{name}.npy", mode="w+", dtype=dtype, shape=_field_shape(name, spec, counts)
        )
    gids = np.lib.format.open_memmap(tmp / "gids.npy", mode="w+", dtype=np.int64, shape=(counts["molecules"],))

    off = {"molecules": 0, "atoms": 0, "bonds": 0, "s1": 0, "s2": 0}
    with open(tmp / "molecule_ids.jsonl", "w", encoding="utf-8") as fh_ids, open(
        tmp / "smiles.jsonl", "w", encoding="utf-8"
    ) as fh_smi:
        for i, p in enumerate(paths):
            shard = torch.load(p, map_location="cpu", weights_only=False)
            nm = int(shard["n_molecules"])
            for name, kind in _POINTER_SPECS:
                pos = off["molecules"] if kind == "molecule" else off["atoms"]
                add = off[_POINTER_BASE[name]]
                src = np.asarray(shard[name], dtype=np.int64)
                alloc[name][pos : pos + src.size - 1] = src[:-1] + add
            for name, _dtype in _VALUE_FIELDS:
                src = np.asarray(shard[name])
                o = _field_offset(name, off)
                alloc[name][o : o + src.shape[0]] = src
            for mid, smi in zip(shard["molecule_ids"], shard["smiles"]):
                fh_ids.write(json.dumps(int(mid)) + "\n")
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
        "dataset": str(dataset),
        "round_id": C.ROUND_ID,
        "tag": str(tag),
        "created_utc": C.utcnow(),
        "source": "official OGB-LSC pcqm4m-v2 raw data.csv.gz -> frozen lipo_v1 adapter",
        "n_shards": int(len(paths)),
        "fields": sorted([n for n, _ in _POINTER_SPECS] + [n for n, _ in _VALUE_FIELDS]),
        "format": "npy-directory-v1",
        "labels_in_cache": False,
        "uses_3d_geometry": False,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    files = sorted(p for p in out_dir.glob("*") if p.is_file())
    return {
        "cache": C.track_rel(out_dir),
        "cache_format": "npy-directory-v1",
        "cache_sha256": C.sha256_tree(out_dir),
        "cache_files": len(files),
        "cache_bytes": int(sum(f.stat().st_size for f in files)),
        "file_bytes": {p.name: int(p.stat().st_size) for p in files},
        "n_molecules": int(counts["molecules"]),
        "n_atoms_total": int(counts["atoms"]),
        "n_bonds_total": int(counts["bonds"]),
        "n_s1_codes": int(counts["s1"]),
        "n_s2_paths": int(counts["s2"]),
        "merge_seconds": float(time.perf_counter() - t0),
        "feature_spec_hash": spec.hash(),
        "peak_rss_mb": C.peak_rss_mb(),
    }


load_cache = PCD.load_cache
s2_count_identity = PCD.s2_count_identity
collate = PREP_DATA.collate


def cache_structure_audit(cache: Mapping[str, Any], spec) -> dict[str, Any]:
    """Structure audit of a merged cache (adapted: dataset label of this round)."""
    out = PCD.cache_structure_audit(cache, spec)
    out["gids_is_identity"] = bool(np.array_equal(np.asarray(cache["gids"]), np.arange(int(cache["n_molecules"]))))
    return out


# ---------------------------------------------------------------------------
# Fit-only standardisation
# ---------------------------------------------------------------------------


def feature_norm_rows(cache: Mapping[str, Any], spec, rows: Sequence[int], *, source: str) -> dict[str, Any]:
    """Standardisation of ``atom_scalar`` / ``global`` from the given cache rows only."""
    rows = np.asarray(sorted(int(r) for r in rows), dtype=np.int64)
    if rows.size == 0:
        raise ValueError("empty fit row set")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    a0, a1 = int(atom_ptr[rows[0]]), int(atom_ptr[rows[-1] + 1])
    scalars = np.asarray(cache["atom_scalar"][a0:a1], dtype=np.float64)
    globals_ = np.asarray(cache["global"][rows], dtype=np.float64)
    if scalars.shape[1] != spec.n_scalars or globals_.shape[1] != spec.global_dim:
        raise ValueError("cache dimensions do not match the feature spec")
    scalar_mean = scalars.mean(0).astype(np.float32)
    scalar_scale, scalar_const = PREP_DATA._scale_from_std(scalars.std(0))
    global_mean = globals_.mean(0).astype(np.float32)
    global_scale, global_const = PREP_DATA._scale_from_std(globals_.std(0))
    norm = {
        "source": str(source),
        "fit_split": "official Train, subset_300k prefix of the cache",
        "fit_rows": int(rows.size),
        "fit_rows_first": int(rows[0]),
        "fit_rows_last": int(rows[-1]),
        "fit_rows_are_contiguous_prefix": bool(np.array_equal(rows, np.arange(rows[0], rows[0] + rows.size))),
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
        "scalar_constant_indices": scalar_const,
        "global_constant_indices": global_const,
        "scalar_mean": scalar_mean,
        "scalar_scale": scalar_scale,
        "global_mean": global_mean,
        "global_scale": global_scale,
        "scalar_mean_list": [float(x) for x in scalar_mean],
        "scalar_scale_list": [float(x) for x in scalar_scale],
        "global_mean_list": [float(x) for x in global_mean],
        "global_scale_list": [float(x) for x in global_scale],
        "n_atoms_used": int(a1 - a0),
    }
    norm["norm_fingerprint"] = PREP_DATA.norm_fingerprint(norm)
    return norm


def label_norm(values: np.ndarray) -> dict[str, Any]:
    """Train-300K mean/std of the gap (ddof=0) used for ``z``."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        raise ValueError("non-finite label in the fit set")
    mean = float(v.mean())
    std = float(v.std())
    if not (std > 0):
        raise ValueError("degenerate label standard deviation")
    return {
        "source": "official Train, subset_300k (identical constants for both runs)",
        "both_runs_shared": True,
        "valid_or_test_used": False,
        "n": int(v.size),
        "mean_eV": mean,
        "std_eV": std,
        "ddof": 0,
        "min_eV": float(v.min()),
        "max_eV": float(v.max()),
        "median_eV": float(np.median(v)),
    }


def to_z(gap_eV: np.ndarray, label_norm_payload: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(gap_eV, dtype=np.float64) - float(label_norm_payload["mean_eV"])) / float(
        label_norm_payload["std_eV"]
    )


def from_z(z: np.ndarray, label_norm_payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) * float(label_norm_payload["std_eV"]) + float(
        label_norm_payload["mean_eV"]
    )


# ---------------------------------------------------------------------------
# deterministic batch order
# ---------------------------------------------------------------------------


def epoch_orders(n_rows: int, *, seed: int, epochs: int) -> dict[str, Any]:
    """Per-epoch permutations of ``range(n_rows)``.

    One ``numpy.random.default_rng(seed)`` drawn epoch by epoch, so epoch ``e``
    can be reproduced without materialising the earlier ones only if the same
    generator stream is replayed; the per-epoch hashes are stored so a resume
    can verify the exact order it continues with.
    """
    rng = np.random.default_rng(int(seed))
    hashes: list[str] = []
    per_epoch: list[int] = []
    for _ in range(int(epochs)):
        order = rng.permutation(int(n_rows)).astype(np.int64)
        hashes.append(C.sha256_bytes(order.tobytes()))
        per_epoch.append(int(order.size))
    return {
        "rng": f"numpy.random.default_rng({int(seed)}) drawn once per epoch",
        "order_seed": int(seed),
        "n_rows": int(n_rows),
        "epochs": int(epochs),
        "epoch_permutation_sha256": hashes,
        "rows_per_epoch": per_epoch,
        "combined_sha256": C.sha256_bytes("|".join(hashes).encode("utf-8")),
    }


def batches_for_epoch(n_rows: int, epoch: int, *, seed: int, batch_size: int = C.BATCH_SIZE) -> list[np.ndarray]:
    """The batches of one epoch, reproducible from ``(n_rows, epoch, seed)`` alone."""
    rng = np.random.default_rng(int(seed))
    order = None
    for _ in range(int(epoch)):
        order = rng.permutation(int(n_rows)).astype(np.int64)
    if order is None:
        raise ValueError("epoch must be >= 1")
    return [order[i : i + int(batch_size)] for i in range(0, order.size, int(batch_size))]


# ---------------------------------------------------------------------------
# label artifact: Train + Valid only, hidden rows NaN
# ---------------------------------------------------------------------------


def build_label_artifact(info: Mapping[str, Any], out_path: Path | str) -> dict[str, Any]:
    """Write one float64 vector of gap labels with every hidden Test row NaN."""
    split = info["split"]
    labels = np.asarray(info["labels"], dtype=np.float64)
    keep = np.concatenate([np.asarray(split[k], dtype=np.int64) for k in C.LABELLED_SPLITS])
    masked = np.full(labels.shape, np.nan, dtype=np.float64)
    masked[keep] = labels[keep]
    if not np.isfinite(masked[keep]).all():
        raise InputAdapterBlocked("non-finite Train/Valid label in the official file")
    hidden_rows = np.concatenate([np.asarray(split[k], dtype=np.int64) for k in C.HIDDEN_SPLITS])
    if np.isfinite(masked[hidden_rows]).any():  # pragma: no cover - defensive
        raise RuntimeError("hidden rows must be NaN in the label artifact")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, masked)
    return {
        "utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "dataset": C.DATASET,
        "n_rows": int(masked.size),
        "labelled_rows_copied": int(keep.size),
        "hidden_rows_masked_nan": int(hidden_rows.size),
        "hidden_row_values_read": False,
        "hidden_row_values_written": False,
        "hidden_row_values_used_in_any_metric": False,
        "artifact": C.track_rel(p),
        "artifact_sha256": C.sha256_file(p),
        "dtype": "float64",
        "units": "eV",
    }


# ---------------------------------------------------------------------------
# task assembly for one formal run
# ---------------------------------------------------------------------------


def load_task(run: str, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Load the frozen caches, normalisers, labels and plan for one run."""
    if run not in C.RUNS_TABLE:
        raise KeyError(run)
    spec = FS.load_spec(C.ANALYSIS / "pcqm_spec.json")
    feature_norm = C.read_json(C.ANALYSIS / "pcqm_feature_norm.json")
    label_norm_payload = C.read_json(C.ANALYSIS / "pcqm_label_norm.json")
    manifest = C.read_json(C.ANALYSIS / "pcqm_cache_manifest.json")
    subset_plan_payload = C.read_json(C.ANALYSIS / "pcqm_subset_plan.json")
    monitor_payload = C.read_json(C.ANALYSIS / "pcqm_monitor_plan.json")
    label_artifact = C.read_json(C.ANALYSIS / "pcqm_label_artifact.json")

    train_dir = C.CACHE / "pcqm_train1m"
    valid_dir = C.CACHE / "pcqm_valid"
    train_cache = load_cache(train_dir)
    valid_cache = load_cache(valid_dir)
    if int(train_cache["n_molecules"]) != int(subset_plan_payload["subsets"]["PCQM_W2_1M"]["n"]):
        raise RuntimeError("train cache size does not match the frozen 1M subset")
    if str(train_cache["meta"]["feature_spec_hash"]) != spec.hash():
        raise RuntimeError("train cache spec hash does not match the round spec")
    if str(valid_cache["meta"]["feature_spec_hash"]) != spec.hash():
        raise RuntimeError("valid cache spec hash does not match the round spec")

    labels = np.load(C.CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    train_rows_all = np.load(C.CACHE / "pcqm_subset_rows.npy").astype(np.int64)
    if train_rows_all.size != int(train_cache["n_molecules"]):
        raise RuntimeError("subset row list does not match the train cache")
    if not np.array_equal(np.asarray(train_cache["gids"], dtype=np.int64), train_rows_all):
        raise RuntimeError("train cache gids are not the frozen permutation prefix")
    if verify_hashes:
        want = int(subset_plan_payload["subsets"]["PCQM_W2_1M"]["n"])
        got = C.index_hash(train_rows_all[:want])
        if got != subset_plan_payload["subsets"]["PCQM_W2_1M"]["rows_sha256"]:
            raise RuntimeError("train subset rows hash does not match the frozen plan")
        if C.sha256_file(C.CACHE / "pcqm_subset_rows.npy") != manifest["subset_rows_sha256"]:
            raise RuntimeError("subset rows file hash does not match the cache manifest")
        if C.sha256_file(C.CACHE / "pcqm_labels_trainvalid.npy") != label_artifact["artifact_sha256"]:
            raise RuntimeError("label artifact hash does not match its ledger entry")
        for key, path in (("train1m", train_dir), ("valid", valid_dir)):
            if C.sha256_tree(path) != manifest[key]["cache_sha256"]:
                raise RuntimeError(f"{key} cache tree hash does not match the cache manifest")

    n_rows = int(C.RUNS_TABLE[run]["cache_rows"])
    if n_rows > int(train_cache["n_molecules"]):
        raise RuntimeError(f"run needs {n_rows} rows but the cache holds {train_cache['n_molecules']}")
    train_gids = train_rows_all[:n_rows]
    valid_gids = np.asarray(valid_cache["gids"], dtype=np.int64)
    if int(np.intersect1d(train_gids, valid_gids, assume_unique=False).size):
        raise RuntimeError("a Validation row is present in the Train rows")

    train_y_eV = labels[train_gids]
    if not np.isfinite(train_y_eV).all():
        raise RuntimeError("non-finite Train label in the run's rows")
    valid_y_eV = labels[valid_gids]
    if not np.isfinite(valid_y_eV).all():
        raise RuntimeError("non-finite Validation label")
    train_y_z = to_z(train_y_eV, label_norm_payload).astype(np.float32)

    monitor_rows = np.asarray(monitor_payload["positions"], dtype=np.int64)
    if monitor_rows.max() >= n_rows:
        raise RuntimeError("monitor rows are outside the run's Train rows")
    monitor_y_eV = train_y_eV[monitor_rows]

    norm = {
        "global_mean": np.asarray(feature_norm["global_mean_list"], dtype=np.float32),
        "global_scale": np.asarray(feature_norm["global_scale_list"], dtype=np.float32),
        "scalar_mean": np.asarray(feature_norm["scalar_mean_list"], dtype=np.float32),
        "scalar_scale": np.asarray(feature_norm["scalar_scale_list"], dtype=np.float32),
    }
    if PREP_DATA.norm_fingerprint({**feature_norm, **norm}) != feature_norm["norm_fingerprint"]:
        raise RuntimeError("feature normaliser arrays do not match their fingerprint")

    orders = epoch_orders(n_rows, seed=C.ORDER_SEED, epochs=int(C.RUNS_TABLE[run]["epochs"]))
    return {
        "run": run,
        "spec": spec,
        "norm": norm,
        "feature_norm": feature_norm,
        "label_norm": label_norm_payload,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "valid_rows": np.arange(int(valid_cache["n_molecules"]), dtype=np.int64),
        "valid_y_eV": valid_y_eV,
        "train_y_eV": train_y_eV,
        "train_y_z": train_y_z,
        "monitor_rows": monitor_rows,
        "monitor_y_eV": monitor_y_eV,
        "monitor_plan": {k: v for k, v in monitor_payload.items() if not k.startswith("_")},
        "n_rows": int(n_rows),
        "steps_epoch": int(C.steps_per_epoch(n_rows)),
        "steps_total": int(C.total_steps(n_rows)),
        "seed": int(C.RUNS_TABLE[run]["seed"]),
        "orders": orders,
        "evaluator": _evaluator(),
        "cache_manifest": manifest,
        "data_isolation": {
            "train_rows": int(n_rows),
            "train_rows_last_gid": int(train_gids[-1]),
            "valid_rows": int(valid_gids.size),
            "train_valid_overlap": int(np.intersect1d(train_gids, valid_gids).size),
            "train_is_prefix_of_frozen_1m_subset": bool(np.array_equal(train_gids, train_rows_all[:n_rows])),
            "subset_plan_containment": subset_plan_payload["containment"],
            "subset_plan_status": subset_plan_payload["status"],
            "hidden_split_rows_touched": 0,
            "monitor_rows_are_train_rows": bool(monitor_rows.max() < n_rows),
        },
    }


def _evaluator():
    from ogb.lsc import PCQM4Mv2Evaluator

    return PCQM4Mv2Evaluator()


__all__ = [
    "InputAdapterBlocked",
    "batches_for_epoch",
    "build_label_artifact",
    "build_shards",
    "cache_structure_audit",
    "collate",
    "data_audit",
    "epoch_orders",
    "feature_norm_rows",
    "from_z",
    "label_norm",
    "label_summary",
    "load_cache",
    "load_official",
    "load_task",
    "merge_shards",
    "monitor_rows",
    "s2_count_identity",
    "subset_plan",
    "to_z",
]
