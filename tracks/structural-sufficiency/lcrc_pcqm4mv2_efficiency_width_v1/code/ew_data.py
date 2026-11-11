"""Data/task assembly for the efficiency + width round.

The cache, the fixed 1M Train subset, the full 73,545-row Validation split,
the Train-only normalisation constants, the monitor rows and the per-epoch
batch order are **reused read-only** from ``lcrc_pcqm4mv2_transfer_pilot_v1``.

``pq_data.load_task('PCQM_W2_1M')`` is the frozen loader: it re-verifies the
subset hash, the label-artifact hash and the SHA256 tree of both mmap caches
against ``pcqm_cache_manifest.json`` before handing back the task.  This module
calls it unchanged, asserts the inherited identities, and annotates the task
with the width scale of the requested run.  Nothing about the data, the
normalisation or the order is re-derived here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ew_common as C


def load_task(run: str, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Frozen 1M task + this round's run identity (no re-sampling)."""
    if run not in C.RUNS_TABLE:
        raise KeyError(run)
    spec = C.RUNS_TABLE[run]
    t0 = time.perf_counter()
    task = C.PQ_DATA.load_task("PCQM_W2_1M", verify_hashes=bool(verify_hashes))
    inherited = {
        "n_rows": int(task["n_rows"]),
        "steps_epoch": int(task["steps_epoch"]),
        "steps_total": int(task["steps_total"]),
        "train_cache_rows": int(task["train_cache"]["n_molecules"]),
        "valid_rows": int(len(task["valid_rows"])),
        "monitor_rows": int(len(task["monitor_rows"])),
        "seed": int(task["seed"]),
        "order_seed": int(task["orders"]["order_seed"]),
        "epochs_in_plan": int(task["orders"]["epochs"]),
        "label_norm_mean_eV": float(task["label_norm"]["mean_eV"]),
        "label_norm_std_eV": float(task["label_norm"]["std_eV"]),
        "feature_norm_fingerprint": str(task["feature_norm"]["norm_fingerprint"]),
        "spec_hash": task["spec"].hash(),
        "train_orders_combined_sha256": task["orders"]["combined_sha256"],
    }
    checks = {
        "train_rows_is_1m": inherited["n_rows"] == C.TRAIN_ROWS,
        "cache_rows_is_1m": inherited["train_cache_rows"] == C.TRAIN_ROWS,
        "steps_epoch_7813": inherited["steps_epoch"] == C.STEPS_PER_EPOCH,
        "steps_total_312520": inherited["steps_total"] == C.TOTAL_STEPS,
        "epochs_40": inherited["epochs_in_plan"] == C.TOTAL_EPOCHS,
        "valid_rows_73545": inherited["valid_rows"] == 73_545,
        "monitor_rows_20000": inherited["monitor_rows"] == C.N_MONITOR,
        "seed_0": inherited["seed"] == 0,
        "order_seed_0": inherited["order_seed"] == 0,
        "batch_size_128": int(C.BATCH_SIZE) == 128,
        "no_train_valid_overlap": int(task["data_isolation"]["train_valid_overlap"]) == 0,
        "train_is_frozen_prefix": bool(task["data_isolation"]["train_is_prefix_of_frozen_1m_subset"]),
        "hidden_rows_touched_0": int(task["data_isolation"]["hidden_split_rows_touched"]) == 0,
    }
    failed = [k for k, v in checks.items() if not bool(v)]
    if failed:
        raise RuntimeError(f"inherited task identity failed: {failed}")

    task = dict(task)
    task["run"] = run
    task["scale"] = int(spec["scale"])
    task["model"] = str(spec["model"])
    task["model_name"] = str(spec["model_name"])
    task["expected_parameters"] = int(spec["expected_parameters"])
    task["widths"] = dict(C.WIDTHS[int(spec["scale"])])
    task["inherited_task_identity"] = inherited
    task["inherited_task_checks"] = checks
    task["load_seconds"] = float(time.perf_counter() - t0)
    task["source_round"] = C.OLD_ROUND.name
    task["cache_paths"] = {
        "train": C.track_rel(C.OLD_CACHE / "pcqm_train1m"),
        "valid": C.track_rel(C.OLD_CACHE / "pcqm_valid"),
    }
    task["source_files"] = {
        "label_norm": C.track_rel(C.OLD_ANALYSIS / "pcqm_label_norm.json"),
        "feature_norm": C.track_rel(C.OLD_ANALYSIS / "pcqm_feature_norm.json"),
        "subset_plan": C.track_rel(C.OLD_ANALYSIS / "pcqm_subset_plan.json"),
        "monitor_plan": C.track_rel(C.OLD_ANALYSIS / "pcqm_monitor_plan.json"),
        "cache_manifest": C.track_rel(C.OLD_ANALYSIS / "pcqm_cache_manifest.json"),
        "label_artifact": C.track_rel(C.OLD_ANALYSIS / "pcqm_label_artifact.json"),
        "spec": C.track_rel(C.OLD_ANALYSIS / "pcqm_spec.json"),
    }
    return task


# ---------------------------------------------------------------------------
# read-only helpers used by the diagnostics
# ---------------------------------------------------------------------------


def smiles_of(cache: Mapping[str, Any], rows: Sequence[int]) -> list[str]:
    """Exact SMILES strings of cache rows (read-only; cache order preserved)."""
    rows = [int(r) for r in rows]
    smiles = cache["smiles"]
    if len(smiles) != int(cache["n_molecules"]):
        raise RuntimeError("cache SMILES list does not match the molecule count")
    return [str(smiles[r]) for r in rows]


def official_train_smiles() -> list[str]:
    """The full official Train SMILES list (read-only dataset access).

    Only ``train`` rows are returned; Validation/Test strings are not indexed.
    Requires the local ``data/ogb`` copy (rdkit is present on the analysis host
    but is not needed to read ``only_smiles=True`` graphs).
    """
    from ogb.lsc import PCQM4Mv2Dataset

    dataset = PCQM4Mv2Dataset(root=str(C.PQ.OGB_ROOT), only_smiles=True)
    split = dataset.get_idx_split()
    train_rows = np.asarray(split["train"], dtype=np.int64)
    graphs = list(dataset.graphs)
    if len(graphs) != len(dataset):
        raise RuntimeError("official dataset length mismatch")
    return [str(graphs[int(r)]) for r in train_rows]


__all__ = ["load_task", "official_train_smiles", "smiles_of"]
