"""ESOL views with **per-split, cache-ordered** physicochemical descriptors.

What was wrong in ``lcrc_esol_quickstart_v1``
----------------------------------------------
``esol_data.build_views`` exposed ``views["physchem"]`` as the full 1,128-row
matrix in *original CSV row order*, and ``esol_models.collate`` indexed it with
the ``rows`` argument.  During training/validation ``rows`` are **local
positions inside the Train/Validation cache** (0..901 / 0..112), so position
``i`` was paired with the descriptors of CSV row ``i`` — a different molecule.
The model was therefore trained on misaligned descriptor/label pairs.

This module fixes the data contract:

* ``views["physchem_train"]`` has shape ``(902, 6)`` and row ``i`` is the raw
  RDKit descriptor vector of ``views["train_cache"]["molecule_ids"][i]``;
* ``views["physchem_valid"]`` is built the same way for the 113 Validation
  molecules;
* the shape guard in ``esol2_models.collate`` makes the old failure mode
  impossible: the matrix length must equal ``cache["n_molecules"]``;
* ``views["physchem"]`` (the full CSV-row-order matrix) is still exposed for
  audits but is **never** passed to a model; the audit records that fact.

Standardisation is **not** applied here.  The raw descriptors travel to
``LCRCPhyschem6.forward``, which standardises exactly once with the Train-only
``physchem_norm`` buffers (see ``esol2_models``).  The old round standardised in
``esol_train._physchem_matrix`` *and* again inside ``forward``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import numpy as np

import esol2_common as C
import esol_common as V1C
import esol_data as V1D
import esol_raw as V1R


@contextmanager
def _redirect_cache():
    """Point v1's derived cache at this round's directory (v1 stays read-only)."""
    original = V1D.CACHE_FILE
    V1D.CACHE_FILE = C.CACHE_FILE
    try:
        yield
    finally:
        V1D.CACHE_FILE = original


def _matrix_sha256(matrix: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(np.asarray(matrix, dtype="<f8")).tobytes())


def build_views() -> dict[str, Any]:
    """Train/Validation views with cache-aligned descriptor matrices (no Test)."""
    with _redirect_cache():
        views = V1D.build_views(include_test=False)

    split_rows = {name: [int(x) for x in views["split"]["rows"][name]] for name in ("train", "valid")}
    if [len(split_rows["train"]), len(split_rows["valid"])] != [C.N_TRAIN, C.N_VALID]:
        raise ValueError(f"unexpected split sizes {[len(split_rows['train']), len(split_rows['valid'])]}")
    if views.get("test_cache") is not None or "y_test" in views["labels"]:
        raise PermissionError("training views unexpectedly contain Test data")

    physchem_all = np.asarray(V1R.load_physchem()["matrix"], dtype=np.float64)
    if physchem_all.shape != (V1C.N_MOLECULES, C.N_PHYSCHEM):
        raise ValueError(f"descriptor artifact shape {physchem_all.shape}")
    views["physchem"] = physchem_all
    views["physchem_usage"] = {
        "full_matrix_order": "original CSV row order (1,128 rows)",
        "training_path": "views['physchem_train'] / views['physchem_valid'] only",
        "full_matrix_is_never_passed_to_a_model": True,
    }

    alignment: dict[str, Any] = {}
    for name, cache_key in (("train", "train_cache"), ("valid", "valid_cache")):
        cache = views[cache_key]
        rows = split_rows[name]
        ids = [int(x) for x in cache["molecule_ids"]]
        if ids != rows:
            raise PermissionError(
                f"{name}: cache molecule_ids are not the frozen split rows "
                f"(first difference at {next(i for i, (a, b) in enumerate(zip(ids, rows)) if a != b)})"
            )
        aligned = np.ascontiguousarray(physchem_all[np.asarray(ids, dtype=np.int64)], dtype=np.float64)
        if not np.array_equal(aligned, physchem_all[np.asarray(rows, dtype=np.int64)]):
            raise PermissionError(f"{name}: cache-ordered descriptors differ from CSV-row indexing")
        if aligned.shape != (len(rows), C.N_PHYSCHEM):
            raise ValueError(f"{name}: aligned descriptor matrix shape {aligned.shape}")
        if not np.isfinite(aligned).all():
            raise ValueError(f"{name}: non-finite descriptors")
        if [str(x) for x in cache["smiles"]] != [views["split"]["smiles"][row] for row in rows]:
            raise PermissionError(f"{name}: cache SMILES identity drifted from the CSV")
        views[f"physchem_{name}"] = aligned
        alignment[name] = {
            "n": int(len(rows)),
            "cache_positions_equal_split_rows": True,
            "molecule_ids_sha256": C.rows_sha256(ids),
            "matrix_sha256": _matrix_sha256(aligned),
            "matrix_shape": list(aligned.shape),
            "first_molecule_id": int(ids[0]),
            "last_molecule_id": int(ids[-1]),
        }

    # Train-only scaler, refit on the aligned Train matrix and compared with the
    # frozen v1 fit (which used the same 902 CSV rows).
    norm = V1D.fit_physchem_norm(views["physchem_train"], range(int(views["physchem_train"].shape[0])))
    recorded = views["physchem_norm"]
    if list(recorded["names"]) != list(C.PHYSCHEM_NAMES):
        raise PermissionError("physchem normalisation names drifted")
    if recorded.get("valid_or_test_used") is not False:
        raise PermissionError("physchem normalisation was not Train-only")
    mean_diff = float(np.max(np.abs(np.asarray(norm["mean"]) - np.asarray(recorded["mean"]))))
    scale_diff = float(np.max(np.abs(np.asarray(norm["scale"]) - np.asarray(recorded["scale"]))))
    if max(mean_diff, scale_diff) > 1.0e-9:
        raise PermissionError(f"refit Train-only physchem scaler drifted (mean {mean_diff}, scale {scale_diff})")

    views["physchem_alignment"] = {
        "splits": alignment,
        "norm_refit_matches_v1": True,
        "norm_mean_max_abs_diff": mean_diff,
        "norm_scale_max_abs_diff": scale_diff,
        "norm_fit_on": recorded["fit_on"],
        "norm_valid_or_test_used": bool(recorded["valid_or_test_used"]),
        "standardisation_location": "LCRCPhyschem6.forward (exactly once, Train-only buffers)",
    }

    # Diagnostic: how wrong was the v1 local-row indexing?  (recorded, not fixed)
    views["v1_alignment_bug"] = _v1_indexing_diagnostic(views, physchem_all, split_rows)
    return views


def _v1_indexing_diagnostic(
    views: Mapping[str, Any], physchem_all: np.ndarray, split_rows: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "description": (
            "v1 collate indexed the full CSV-row-order descriptor matrix with cache-local rows; "
            "this records how many position pairs actually disagreed"
        )
    }
    for name in ("train", "valid"):
        rows = np.asarray(split_rows[name], dtype=np.int64)
        local = np.arange(rows.size, dtype=np.int64)
        correct = physchem_all[rows]
        v1_lookup = physchem_all[np.clip(local, 0, physchem_all.shape[0] - 1)]
        different = (rows != local)
        diff = np.abs(correct - v1_lookup)
        out[name] = {
            "n": int(rows.size),
            "n_positions_with_wrong_csv_row": int(different.sum()),
            "fraction_wrong": float(different.mean()),
            "max_abs_MolWt_error": float(diff[:, 0].max()),
            "median_abs_MolWt_error": float(np.median(diff[:, 0])),
            "max_abs_TPSA_error": float(diff[:, 2].max()),
            "max_abs_MolLogP_error": float(diff[:, 1].max()),
        }
    # double standardisation drift (the second v1 defect), quantified on Train
    norm = views["physchem_norm"]
    mean = np.asarray(norm["mean"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    raw = np.asarray(views["physchem_train"], dtype=np.float64)
    single = (raw - mean) / scale
    double = (single - mean) / scale
    out["double_standardisation"] = {
        "max_abs_single_vs_double": float(np.max(np.abs(single - double))),
        "mean_abs_single_vs_double": float(np.mean(np.abs(single - double))),
        "max_abs_standardised_value_single": float(np.max(np.abs(single))),
        "max_abs_standardised_value_double": float(np.max(np.abs(double))),
    }
    return out


def physchem_for(model_kind: str, views: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Raw descriptors for (train, valid), or ``(None, None)`` for non-physchem arms."""
    if model_kind in ("base", "s2blind_e"):
        return None, None
    if model_kind != "physchem6":
        raise ValueError(f"unknown model kind {model_kind!r}")
    return views["physchem_train"], views["physchem_valid"]


__all__ = ["build_views", "physchem_for"]
