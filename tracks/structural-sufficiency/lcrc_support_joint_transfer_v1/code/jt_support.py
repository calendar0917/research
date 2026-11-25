"""Train-only Omega construction for the official Lipo and PCBA splits.

The *definition* is exactly ``lcrc_support_joint_function_v0`` (read-only):

* ``Omega`` = distinct ``(path_cat, path_end63)`` pairs in the official Train
  split,
* bipartite support graph and float64 reduced SVD of its incidence matrix,
* ``P = U U^T`` and the two-level unseen-pair fallback.

This module only adds two data adapters:

* the official Lipo caches of ``lcrc_readout_context_pilot_small_v1``;
* a memory-bounded reader for the 163-GB PCBA cache directory, which extracts
  only the four S2 path arrays of one split.  No atom/bond/feature array is
  loaded, so support construction cannot exhaust local RAM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import jt_common as C

_PATHS_READY = False


def _prepare() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (
        str(C.SJ_CODE),
        str(C.READOUT_CODE),
        str(C.SS / "lcrc_cross_task_preparation_v1/code"),
        str(C.LR_CODE),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def _modules() -> dict[str, Any]:
    _prepare()
    import sj_support

    return {"sj_support": sj_support}


# ---------------------------------------------------------------------------
# compact split caches
# ---------------------------------------------------------------------------


def _compact_cache(
    atom_ptr: np.ndarray,
    path_ptr: np.ndarray,
    path_cat: np.ndarray,
    path_end63: np.ndarray,
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract one split's S2 records into a self-contained compact cache.

    The produced arrays satisfy exactly the structural contract used by
    ``sj_support.split_records``: ``atom_ptr`` (molecules+1), ``path_ptr``
    (atoms+1), one record per path.  Every value is copied from the read-only
    source; nothing is recomputed from a model or a label.
    """
    rows = np.asarray(rows, dtype=np.int64)
    ap_lo = np.asarray(atom_ptr[rows], dtype=np.int64)
    ap_hi = np.asarray(atom_ptr[rows + 1], dtype=np.int64)
    n_atoms_per_mol = ap_hi - ap_lo
    n_atoms = int(n_atoms_per_mol.sum())
    new_atom_ptr = np.zeros(rows.size + 1, dtype=np.int64)
    new_atom_ptr[1:] = np.cumsum(n_atoms_per_mol)

    # selected atom -> source atom index
    atom_cum = np.cumsum(n_atoms_per_mol)
    atom_off = np.repeat(ap_lo - (atom_cum - n_atoms_per_mol), n_atoms_per_mol)
    atom_idx = np.arange(n_atoms, dtype=np.int64) + atom_off
    del atom_off

    path_counts_all = np.diff(path_ptr).astype(np.int64)
    per_atom_paths = path_counts_all[atom_idx]
    n_paths = int(per_atom_paths.sum())
    new_path_ptr = np.zeros(n_atoms + 1, dtype=np.int64)
    new_path_ptr[1:] = np.cumsum(per_atom_paths)

    path_cum = np.cumsum(per_atom_paths)
    path_off = np.repeat(
        np.asarray(path_ptr[atom_idx], dtype=np.int64) - (path_cum - per_atom_paths),
        per_atom_paths,
    )
    rec_idx = np.arange(n_paths, dtype=np.int64) + path_off
    del path_off, path_cum, per_atom_paths, atom_idx

    out = {
        "atom_ptr": new_atom_ptr,
        "path_ptr": new_path_ptr,
        "path_cat": np.asarray(path_cat[rec_idx], dtype=np.int64),
        "path_end63": np.asarray(path_end63[rec_idx], dtype=np.int64),
        "n_molecules": np.int64(rows.size),
        "n_atoms": np.int64(n_atoms),
    }
    return out


def pcba_split_caches(
    split_payload: Mapping[str, Any] | None = None,
    cache_dir: Path | str = C.PC_CACHE,
) -> dict[str, dict[str, np.ndarray]]:
    """The three official PCBA split caches, S2 records only (memory bounded)."""
    if split_payload is None:
        split_payload = C.read_json(C.LR_ROUND / "analysis/pcba_split.json")
    cache_dir = Path(cache_dir)
    atom_ptr = np.load(cache_dir / "atom_ptr.npy", mmap_mode="r")
    path_ptr = np.load(cache_dir / "path_ptr.npy", mmap_mode="r")
    path_cat = np.load(cache_dir / "path_cat.npy", mmap_mode="r")
    path_end63 = np.load(cache_dir / "path_end63.npy", mmap_mode="r")
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in C.SPLITS:
        out[split] = _compact_cache(
            atom_ptr,
            path_ptr,
            path_cat,
            path_end63,
            np.asarray(split_payload["rows"][split], dtype=np.int64),
        )
    return out


def lipo_split_caches(bundle: Any) -> dict[str, dict[str, np.ndarray]]:
    caches = bundle.caches if hasattr(bundle, "caches") else bundle["caches"]
    return {split: caches[split] for split in C.SPLITS}


# ---------------------------------------------------------------------------
# support construction
# ---------------------------------------------------------------------------


def build_lipo_support(bundle: Any) -> Any:
    mods = _modules()
    caches = lipo_split_caches(bundle)
    spec = bundle.spec if hasattr(bundle, "spec") else bundle["spec"]
    return mods["sj_support"].build(
        "lipo",
        caches["train"],
        valid_cache=caches["valid"],
        test_cache=caches["test"],
        spec=spec,
    )


def pcba_spec() -> Any:
    _prepare()
    from feature_spec import load_spec

    return load_spec(C.PCBA_SPEC_PATH)


def build_pcba_support(spec: Any, caches: Mapping[str, Mapping[str, np.ndarray]]) -> Any:
    _register_pcba_task()
    mods = _modules()
    return mods["sj_support"].build(
        "pcba",
        caches["train"],
        valid_cache=caches["valid"],
        test_cache=caches["test"],
        spec=spec,
    )


# ``sj_support.build`` looks up ``C.TASKS[task_key]`` only for the dataset label.
def _register_pcba_task() -> None:
    _prepare()
    import sj_common as SC

    if "pcba" not in SC.TASKS:
        # Read-only, module-local registration; no file is modified.  The task
        # table entry supplies only the descriptive dataset field and the
        # primary/secondary metric names used by the analysis helpers.
        SC.TASKS["pcba"] = {
            "key": "pcba",
            "dataset": "ogbg-molpcba",
            "family": "pcba",
            "variant": "C1",
            "task": "binary",
            "epochs": 25,
            "primary": "ap",
            "secondary": "bce",
            "direction": "higher",
            "units": "128 binary assays",
            "split_sizes": {"train": 350343, "valid": 43793, "test": 43793},
            "test_role": C.PCBA_TEST_ROLE,
        }


def build_pcba_support_registered(spec: Any, caches: Mapping[str, Mapping[str, np.ndarray]]) -> Any:
    _register_pcba_task()
    return build_pcba_support(spec, caches)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

_ARRAYS = (
    "omega_p",
    "omega_e",
    "omega_code",
    "omega_count",
    "omega_docfreq",
    "p_types",
    "e_types",
    "X",
    "U",
    "s",
    "P",
    "pinvX",
    "comp_p",
    "comp_e",
    "index_table",
    "pair_mode",
    "p_col",
    "e_col",
)


def save_support(support: Any, path: Path | str) -> tuple[Path, Path]:
    path = Path(path)
    arrays = {name: np.asarray(getattr(support, name)) for name in _ARRAYS}
    C.atomic_save_npz(path, arrays)
    meta = {
        "schema": "lcrc-support-joint-transfer-support-v1",
        "round_id": C.ROUND_ID,
        "task_key": support.task_key,
        "dataset": support.dataset,
        "n_path_codes": int(support.n_path_codes),
        "n_nbr_codes": int(support.n_nbr_codes),
        "n_bond_types": int(support.n_bond_types),
        "rank": int(support.rank),
        "n_components": int(support.n_components),
        "n_rectangles": int(support.n_rectangles),
        "n_omega": int(support.n_omega),
        "n_p": int(support.n_p),
        "n_e": int(support.n_e),
        "summary": support.summary(),
        "coverage": support.coverage,
        "legality": support.legality,
        "build": support.build_meta,
        "arrays": {name: C.sha256_array(arrays[name]) for name in sorted(arrays)},
    }
    meta_path = path.with_suffix(".json")
    C.write_json(meta_path, meta)
    return path, meta_path


def load_support(path: Path | str) -> Any:
    _prepare()
    import sj_support

    path = Path(path)
    arrays = C.load_npz(path)
    meta = C.read_json(path.with_suffix(".json"))
    missing = [name for name in _ARRAYS if name not in arrays]
    if missing:
        raise RuntimeError(f"support artifact is missing {missing}")
    return sj_support.Support(
        task_key=str(meta["task_key"]),
        dataset=str(meta["dataset"]),
        n_path_codes=int(meta["n_path_codes"]),
        n_nbr_codes=int(meta["n_nbr_codes"]),
        n_bond_types=int(meta["n_bond_types"]),
        omega_p=arrays["omega_p"],
        omega_e=arrays["omega_e"],
        omega_code=arrays["omega_code"],
        omega_count=arrays["omega_count"],
        omega_docfreq=arrays["omega_docfreq"],
        p_types=arrays["p_types"],
        e_types=arrays["e_types"],
        X=arrays["X"],
        U=arrays["U"],
        s=arrays["s"],
        P=arrays["P"],
        pinvX=arrays["pinvX"],
        rank=int(meta["rank"]),
        n_components=int(meta["n_components"]),
        comp_p=arrays["comp_p"],
        comp_e=arrays["comp_e"],
        index_table=arrays["index_table"],
        pair_mode=arrays["pair_mode"],
        p_col=arrays["p_col"],
        e_col=arrays["e_col"],
        n_rectangles=int(meta["n_rectangles"]),
        coverage=meta.get("coverage") or {},
        molecule_coverage={},
        legality=meta.get("legality") or {},
        build_meta=meta.get("build") or {},
    )


def omega_identity(support: Any) -> dict[str, Any]:
    return {
        "n_omega": int(support.n_omega),
        "omega_code_sha256": C.sha256_array(np.asarray(support.omega_code, dtype=np.int64)),
        "omega_p_sha256": C.sha256_array(np.asarray(support.omega_p, dtype=np.int64)),
        "omega_e_sha256": C.sha256_array(np.asarray(support.omega_e, dtype=np.int64)),
        "P_sha256": C.sha256_array(np.asarray(support.P, dtype=np.float64)),
        "rank": int(support.rank),
        "n_components": int(support.n_components),
    }


__all__ = [name for name in dir() if not name.startswith("_")]
