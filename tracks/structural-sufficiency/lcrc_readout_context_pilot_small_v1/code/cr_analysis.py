"""Analysis: main table, contrasts, traceback ledgers, interventions, verdicts.

Everything here is CPU-only and runs **after** the formal trajectories are pulled
back.  No quantity in this module can change a trained weight, a checkpoint or a
hyper-parameter: it reads the selected checkpoints the training stage wrote and
re-derives predictions, contributions and responses from them.

Frozen read-out scales (declared before the runs):

* traceback tolerance ``tol_raw(x) = 1e-5 + 1e-6 * max(1, |x|)``
* bootstrap 2000 paired molecule resamples, RNG 1703, one resample shared by all
  models of a split
* retention tolerance 5 %, context-gain threshold 2 %, interaction read-out
  resolution 1 % of the Train target standard deviation, root-specificity
  threshold 0.05
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cr_common as C
import cr_data as D
import cr_models as M
import cr_train as T

OUT = C.ANALYSIS_DIR

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = C.BOOTSTRAP_SEED
RETENTION_TOL = 1.05
CONTEXT_GAIN_MIN = 0.02
RESOLVE_RMS_MIN = 0.01
ROOT_SPECIFICITY_MAX = 0.05
DIAG_N_MOLECULES = 32
CONTEXT_CURVE_POINTS = 101

POSITIVE_COLOUR = "#c0392b"
NEUTRAL_COLOUR = "#7f8c8d"
NEGATIVE_COLOUR = "#2471a3"
CONTEXT_COLOURS = ("#8e44ad", "#e67e22")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    for path in (OUT, OUT / "figures", OUT / "ledgers", OUT / "cases", OUT / "tables"):
        C.ensure_dir(path)


def traj_path(dataset: str, arm: str):
    return C.RESULTS_DIR / dataset / arm / "trajectory.json"


def load_trajectory(dataset: str, arm: str) -> dict[str, Any]:
    path = traj_path(dataset, arm)
    if not path.is_file():
        raise FileNotFoundError(f"missing trajectory for {dataset}/{arm}: {path}")
    return C.read_json(path)


def load_predictions(dataset: str, arm: str, tag: str = "best") -> dict[str, dict[str, np.ndarray]]:
    path = C.RESULTS_DIR / dataset / arm / f"{tag}_predictions.npz"
    with np.load(path, allow_pickle=False) as handle:
        data = {k: handle[k] for k in handle.files}
    return {
        split: {
            "pred_raw": data[f"{split}_pred_raw"].astype(np.float64),
            "pred_std": data[f"{split}_pred_std"].astype(np.float64),
            "y_raw": data[f"{split}_y_raw"].astype(np.float64),
            "row_id": data[f"{split}_row_id"].astype(np.int64),
        }
        for split in C.SPLITS
    }

def align_to(
    payload: Mapping[str, np.ndarray], reference_row_id: np.ndarray, name: str
) -> dict[str, np.ndarray]:
    """Put one prediction block in the reference row order and verify alignment.

    Row ids are identical sets (official OGB indices); an ordering difference or
    a float32-stored target must not silently corrupt a paired contrast.
    """
    row_id = np.asarray(payload["row_id"], dtype=np.int64)
    reference = np.asarray(reference_row_id, dtype=np.int64)
    if row_id.size != reference.size or set(row_id.tolist()) != set(reference.tolist()):
        raise RuntimeError(f"{name}: official row ids differ from the reference split")
    order = np.argsort(row_id, kind="stable")
    sorted_ids = row_id[order]
    positions = np.searchsorted(sorted_ids, reference)
    out = {}
    for key in ("pred_raw", "pred_std", "y_raw"):
        values = np.asarray(payload[key], dtype=np.float64)[order][positions]
        out[key] = values
    out["row_id"] = reference.copy()
    return out


def load_reference_l(dataset: str) -> dict[str, Any] | None:
    """The frozen ``L`` (C1 / NO_BOND_MEAN, R3) trajectory of sq-pilot-small-v1.

    Arm A is that same head; the only intentional difference is the frozen batch
    order (this round uses ``np.random.default_rng(0)`` per epoch, the historical
    run used ``shuffle_offset=917000``).  Comparing A to L is therefore a
    reproduction check of the loop, not a new experiment.
    """
    path = C.SQL_RESULTS / dataset / "method_L.json"
    if not path.is_file():
        return None
    payload = C.read_json(path)
    return {
        "source": str(path.relative_to(C.REPO_ROOT)),
        "variant": payload.get("variant"),
        "n_parameters": payload.get("n_parameters"),
        "recipe": payload.get("recipe"),
        "selected": payload.get("selected"),
        "final": payload.get("final"),
        "last20": payload.get("last20"),
    }


def load_gine(dataset: str) -> dict[str, Any] | None:
    path = C.SQL_RESULTS / dataset / "method_G.json"
    if not path.is_file():
        return None
    payload = C.read_json(path)
    predictions = {}
    for split in C.SPLITS:
        block = payload["selected_predictions"][split]
        predictions[split] = {
            "pred_raw": np.asarray(block["pred_raw"], dtype=np.float64),
            "pred_std": np.asarray(block["pred_std"], dtype=np.float64),
            "y_raw": np.asarray(block["target_raw"], dtype=np.float64),
            "row_id": np.asarray(block["row_ids"], dtype=np.int64),
        }
    curves = payload["per_lr"][str(payload["selected_peak_lr"])]["curves"]
    return {
        "raw": payload,
        "predictions": predictions,
        "curves": curves,
        "n_parameters": int(payload["n_parameters"]),
        "selected": payload["selected"],
        "final": payload["final_round_selected"],
        "last20": payload["last20_selected"],
        "seconds_total": float(payload["seconds_total"]),
        "selected_peak_lr": float(payload["selected_peak_lr"]),
        "source": str(path.relative_to(C.REPO_ROOT)),
    }


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("inf")


def _arm_row(dataset: str, arm: str, traj: Mapping[str, Any]) -> dict[str, Any]:
    sel, fin, last20 = traj["selected"], traj["final"], traj["last20"]
    return {
        "dataset": dataset,
        "arm": arm,
        "origin": "THIS_ROUND",
        "reused": False,
        "n_parameters": traj["n_parameters"],
        "readout_parameters": traj["readout_parameters"],
        "selected_epoch": sel["epoch"],
        "train_rmse_raw": sel["train_rmse_raw"],
        "valid_rmse_raw": sel["valid_rmse_raw"],
        "test_rmse_raw": sel["test_rmse_raw"],
        "train_mae_raw": sel["train_mae_raw"],
        "valid_mae_raw": sel["valid_mae_raw"],
        "test_mae_raw": sel["test_mae_raw"],
        "final_epoch": fin["epoch"],
        "final_train_rmse_raw": fin["train_rmse_raw"],
        "final_valid_rmse_raw": fin["valid_rmse_raw"],
        "final_test_rmse_raw": fin["test_rmse_raw"],
        "last20_train_rmse_raw": last20["mean_last20_train_rmse_raw"],
        "last20_valid_rmse_raw": last20["mean_last20_valid_rmse_raw"],
        "last20_test_rmse_raw": last20["mean_last20_test_rmse_raw"],
        "total_updates": traj["total_updates"],
        "train_seconds": traj["train_seconds"],
        "eval_seconds_total": traj["eval_seconds_total"],
        "inference_seconds_three_splits": traj["inference_seconds_full_three_splits"],
        "gpu_hours": traj.get("gpu_hours"),
        "checkpoint_best_val_sha256": traj["checkpoint_hashes"]["checkpoint_best_val.pt"],
        "test_eval_count": traj["test_eval_count"],
        "test_policy": traj["test_policy"],
        "device": traj.get("device"),
    }


def _gine_row(dataset: str, gine: Mapping[str, Any]) -> dict[str, Any]:
    sel, fin, last20 = gine["selected"], gine["final"], gine["last20"]
    return {
        "dataset": dataset,
        "arm": "G",
        "origin": "REUSED_FROZEN_REFERENCE",
        "reused": True,
        "n_parameters": gine["n_parameters"],
        "readout_parameters": None,
        "selected_epoch": sel["epoch"],
        "train_rmse_raw": sel["train_rmse_raw"],
        "valid_rmse_raw": sel["valid_rmse_raw"],
        "test_rmse_raw": sel["test_rmse_raw"],
        "train_mae_raw": sel["train_mae_raw"],
        "valid_mae_raw": sel["valid_mae_raw"],
        "test_mae_raw": sel["test_mae_raw"],
        "final_epoch": fin["epoch"],
        "final_train_rmse_raw": fin["train_rmse_raw"],
        "final_valid_rmse_raw": fin["valid_rmse_raw"],
        "final_test_rmse_raw": fin["test_rmse_raw"],
        "last20_train_rmse_raw": last20["mean_last20_train_rmse_raw"],
        "last20_valid_rmse_raw": last20["mean_last20_valid_rmse_raw"],
        "last20_test_rmse_raw": last20["mean_last20_test_rmse_raw"],
        "total_updates": None,
        "train_seconds": gine["seconds_total"],
        "eval_seconds_total": None,
        "inference_seconds_three_splits": None,
        "gpu_hours": 0.0,
        "checkpoint_best_val_sha256": None,
        "test_eval_count": 360,
        "test_policy": "always (inherited from the frozen sq round)",
        "device": "res GPU, previous round",
    }


def contrast_entries(
    dataset: str, trajectory: Mapping[str, Mapping[str, Any]], gine: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    rows = []
    for name, (left, right) in {
        "B-A": ("B", "A"),
        "C-A": ("C", "A"),
        "C-B": ("C", "B"),
        "A-G": ("A", "G"),
        "C-G": ("C", "G"),
    }.items():
        if "G" in (left, right):
            if gine is None:
                continue
            lsel = gine["selected"] if left == "G" else trajectory[left]["selected"]
            rsel = gine["selected"] if right == "G" else trajectory[right]["selected"]
            ll20 = gine["last20"] if left == "G" else trajectory[left]["last20"]
            rl20 = gine["last20"] if right == "G" else trajectory[right]["last20"]
            lfin = gine["final"] if left == "G" else trajectory[left]["final"]
            rfin = gine["final"] if right == "G" else trajectory[right]["final"]
        else:
            lsel, rsel = trajectory[left]["selected"], trajectory[right]["selected"]
            ll20, rl20 = trajectory[left]["last20"], trajectory[right]["last20"]
            lfin, rfin = trajectory[left]["final"], trajectory[right]["final"]
        row: dict[str, Any] = {"dataset": dataset, "contrast": name, "left": left, "right": right}
        for split in C.SPLITS:
            key = f"{split}_rmse_raw"
            row[f"selected_delta_rmse_raw_{split}"] = float(lsel[key] - rsel[key])
            row[f"selected_ratio_rmse_{split}"] = _ratio(float(lsel[key]), float(rsel[key]))
            row[f"final_delta_rmse_raw_{split}"] = float(lfin[key] - rfin[key])
            row[f"last20_delta_rmse_raw_{split}"] = float(
                ll20[f"mean_last20_{split}_rmse_raw"] - rl20[f"mean_last20_{split}_rmse_raw"]
            )
            row[f"last20_ratio_rmse_{split}"] = _ratio(
                float(ll20[f"mean_last20_{split}_rmse_raw"]), float(rl20[f"mean_last20_{split}_rmse_raw"])
            )
        for split in ("valid", "test"):
            key = f"{split}_mae_raw"
            row[f"selected_delta_mae_raw_{split}"] = float(lsel[key] - rsel[key])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# bootstrap / drop-one
# ---------------------------------------------------------------------------


def paired_bootstrap(
    predictions: Mapping[str, np.ndarray],
    y_raw: np.ndarray,
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """One shared molecule resample per replicate; every model recomputed each time."""
    rng = np.random.default_rng(int(seed))
    n = int(y_raw.size)
    names = sorted(predictions)
    rmse_draws = {name: np.empty(n_replicates) for name in names}
    mae_draws = {name: np.empty(n_replicates) for name in names}
    for r in range(int(n_replicates)):
        idx = rng.integers(0, n, n)
        y = y_raw[idx]
        for name in names:
            pred = predictions[name][idx]
            rmse_draws[name][r] = C.rmse(pred, y)
            mae_draws[name][r] = C.mae(pred, y)
    point = {
        name: {"rmse_raw": C.rmse(predictions[name], y_raw), "mae_raw": C.mae(predictions[name], y_raw)}
        for name in names
    }
    contrasts: dict[str, Any] = {}
    for left, right in (("B", "A"), ("C", "A"), ("C", "B"), ("A", "G"), ("C", "G")):
        if left not in names or right not in names:
            continue
        contrasts[f"{left}-{right}"] = {}
        for metric, draws in (("rmse_raw", rmse_draws), ("mae_raw", mae_draws)):
            diff_draws = draws[left] - draws[right]
            low, high = float(np.percentile(diff_draws, 2.5)), float(np.percentile(diff_draws, 97.5))
            contrasts[f"{left}-{right}"][metric] = {
                "point": float(point[left][metric] - point[right][metric]),
                "ci95_low": low,
                "ci95_high": high,
                "contains_zero": bool(low <= 0.0 <= high),
                "n_replicates": int(n_replicates),
            }
    return {
        "n": n,
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "point": point,
        "contrasts": contrasts,
        "note": (
            "conditional on the single-seed selected state; this is not training-seed variance and "
            "does not repair Val selection or Test development exposure"
        ),
    }


def drop_one_sensitivity(pred_left: np.ndarray, pred_right: np.ndarray, y_raw: np.ndarray) -> dict[str, Any]:
    """Per-row leave-one-out range of the paired RMSE/MAE difference."""
    n = int(y_raw.size)
    base_rmse = C.rmse(pred_left, y_raw) - C.rmse(pred_right, y_raw)
    base_mae = C.mae(pred_left, y_raw) - C.mae(pred_right, y_raw)
    rmse_deltas = np.empty(n)
    mae_deltas = np.empty(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        rmse_deltas[i] = C.rmse(pred_left[keep], y_raw[keep]) - C.rmse(pred_right[keep], y_raw[keep])
        mae_deltas[i] = C.mae(pred_left[keep], y_raw[keep]) - C.mae(pred_right[keep], y_raw[keep])
    flips = np.sign(rmse_deltas) != np.sign(base_rmse)
    shift = np.abs(rmse_deltas - base_rmse)
    return {
        "n_rows": n,
        "delta_rmse_full": float(base_rmse),
        "delta_rmse_range": [float(rmse_deltas.min()), float(rmse_deltas.max())],
        "delta_mae_full": float(base_mae),
        "delta_mae_range": [float(mae_deltas.min()), float(mae_deltas.max())],
        "direction_flips_if_one_row_dropped": bool(flips.any()),
        "n_rows_that_flip_direction": int(flips.sum()),
        "max_single_row_shift_rmse": float(shift.max()),
        "max_single_row_share_of_delta": float(shift.max() / abs(base_rmse)) if base_rmse else None,
        "worst_row_index": int(np.argmax(shift)),
        "note": "dropped numbers never replace the official full-split score",
    }


# ---------------------------------------------------------------------------
# per-molecule cache slices (also used by the intervention diagnostic)
# ---------------------------------------------------------------------------


def molecule_slice(cache: Mapping[str, Any], row: int, *, drop_path: int | None = None) -> dict[str, Any]:
    """One molecule out of a split cache, optionally without one S2 occurrence."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    nbr_ptr_all = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr_all = np.asarray(cache["path_ptr"], dtype=np.int64)
    a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    per_atom_path: list[int] = []
    keep: list[int] = []
    for g in range(a0, a1):
        lo, hi = int(path_ptr_all[g]), int(path_ptr_all[g + 1])
        kept = [j for j in range(lo, hi) if drop_path is None or j != drop_path]
        per_atom_path.append(len(kept))
        keep.extend(kept)
    keep_arr = np.asarray(keep, dtype=np.int64)
    per_atom_nbr = [int(nbr_ptr_all[g + 1] - nbr_ptr_all[g]) for g in range(a0, a1)]
    sub = {
        "atom_ptr": np.asarray([0, a1 - a0], dtype=np.int64),
        "bond_ptr": np.asarray([0, b1 - b0], dtype=np.int64),
        "nbr_ptr": np.concatenate([[0], np.cumsum(per_atom_nbr)]).astype(np.int64),
        "path_ptr": np.concatenate([[0], np.cumsum(per_atom_path)]).astype(np.int64),
        "atom_feat": np.asarray(cache["atom_feat"][a0:a1]),
        "atom_type": np.asarray(cache["atom_type"][a0:a1]),
        "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1]),
        "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
        "bond_type": np.asarray(cache["bond_type"][b0:b1]),
        "bond_index": np.asarray(cache["bond_index"][b0:b1]),
        "nbr_code": np.asarray(cache["nbr_code"])[int(nbr_ptr_all[a0]) : int(nbr_ptr_all[a1])],
        "path_cat": np.asarray(cache["path_cat"])[keep_arr],
        "path_end63": np.asarray(cache["path_end63"])[keep_arr],
        "path_end": np.asarray(cache["path_end"])[keep_arr],
        "path_closure": np.asarray(cache["path_closure"])[keep_arr],
        "global": np.asarray(cache["global"][row : row + 1]),
        "n_components": np.asarray(cache["n_components"][row : row + 1]),
        "n_junctions": np.asarray(cache["n_junctions"][row : row + 1]),
        "n_ring_systems": np.asarray(cache["n_ring_systems"][row : row + 1]),
        "molecule_ids": [str(cache["molecule_ids"][row])],
        "smiles": [str(cache["smiles"][row])],
        "n_molecules": 1,
        "split": cache.get("split", ""),
        "meta": dict(cache.get("meta", {})),
    }
    sub["meta"]["n_atoms_total"] = int(a1 - a0)
    sub["meta"]["n_bonds_total"] = int(b1 - b0)
    sub["meta"]["n_s2_paths"] = int(keep_arr.size)
    sub["meta"]["n_s1_codes"] = int(sub["nbr_code"].size)
    return sub


def atom_symbols(smiles: str) -> list[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [str(atom.GetSymbol()) for atom in mol.GetAtoms()]


def path_identity(smiles: str, cache: Mapping[str, Any], path_index: int, molecule_row: int) -> dict[str, Any]:
    """Root / end / bond types of one S2 occurrence; the middle is a candidate set.

    ``path_ptr`` is indexed by the *split-cache* atom number, so the root is
    converted back to a molecule-local index; ``path_end`` is already local.
    """
    from rdkit import Chem

    n_bond_types = int(M.MOD["dissect_common"].N_BOND_TYPES)
    ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    atom_offset = int(atom_ptr[molecule_row])
    root_global = int(np.searchsorted(ptr, path_index, side="right") - 1)
    root = root_global - atom_offset
    cat = int(cache["path_cat"][path_index])
    end = int(cache["path_end"][path_index])
    bt1 = (cat // n_bond_types) % n_bond_types + 1
    bt2 = cat % n_bond_types + 1
    symbols = atom_symbols(smiles)
    middles: list[int] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(mol.GetNumAtoms())}
        for bond in mol.GetBonds():
            i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
            bt_raw = bond.GetBondTypeAsDouble()
            bt = 4 if bt_raw == 1.5 else int(bt_raw)
            adj[i].append((j, bt))
            adj[j].append((i, bt))
        middles = sorted(
            w for w, b1 in adj.get(root, []) if b1 == bt1 and any(u == end and b2 == bt2 for u, b2 in adj[w])
        )
    return {
        "path_index_in_split_cache": int(path_index),
        "root_atom_index": int(root),
        "root_atom_index_in_split_cache": int(root_global),
        "root_symbol": symbols[root] if 0 <= root < len(symbols) else "?",
        "end_atom_index": int(end),
        "end_symbol": symbols[end] if 0 <= end < len(symbols) else "?",
        "bt1": int(bt1),
        "bt2": int(bt2),
        "path_cat": cat,
        "closure": int(cache["path_closure"][path_index]),
        "middle_candidates": middles,
        "middle_symbol": (
            symbols[middles[0]] if len(middles) == 1 and 0 <= middles[0] < len(symbols) else "?"
        ),
        "middle_mapping_unique": bool(len(middles) == 1),
        "n_atoms": int(len(symbols)),
    }


# ---------------------------------------------------------------------------
# component ledger
# ---------------------------------------------------------------------------


def root_input_digest(cache: Mapping[str, Any], row: int, atom_in_molecule: int) -> str:
    """SHA-256 of one root's frozen cache window (a stable lookup key, not a model value).

    Covers the root atom row, its S1 code row, its S2 occurrence rows and the
    molecule's global vector: everything the frozen cache says about that root
    except the *identity* of its neighbours' rows, which depend on the rest of
    the graph.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    g = int(atom_ptr[row]) + int(atom_in_molecule)
    pieces = [
        np.asarray(cache["atom_feat"][g], dtype=np.float32).tobytes(),
        np.asarray([cache["atom_type"][g]], dtype=np.int32).tobytes(),
        np.asarray(cache["atom_scalar"][g], dtype=np.float32).tobytes(),
        np.asarray(cache["nbr_code"][int(nbr_ptr[g]) : int(nbr_ptr[g + 1])], dtype=np.int32).tobytes(),
        np.asarray(cache["path_cat"][int(path_ptr[g]) : int(path_ptr[g + 1])], dtype=np.int32).tobytes(),
        np.asarray(cache["path_end63"][int(path_ptr[g]) : int(path_ptr[g + 1])], dtype=np.int32).tobytes(),
        np.asarray(cache["path_end"][int(path_ptr[g]) : int(path_ptr[g + 1])], dtype=np.int32).tobytes(),
        np.asarray(cache["path_closure"][int(path_ptr[g]) : int(path_ptr[g + 1])], dtype=np.int32).tobytes(),
        np.asarray(cache["global"][row], dtype=np.float32).tobytes(),
    ]
    return C.sha256_bytes(b"".join(pieces))


def _atom_rows(
    bundle: D.Bundle,
    split: str,
    blocks: Sequence[Mapping[str, np.ndarray]],
    *,
    arm: str,
    sigma: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache = bundle.caches[split]
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"])
    symbols = {row: atom_symbols(str(cache["smiles"][row])) for row in range(int(cache["n_molecules"]))}
    for block in blocks:
        atom_batch = block["atom_batch"]
        counts = block["atom_counts"]
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        local_index = np.arange(atom_batch.size) - offsets[atom_batch]
        for j in range(atom_batch.size):
            g = int(atom_batch[j])
            cache_row = int(block["cache_rows"][g])
            row_id = int(bundle.global_row_ids[split][cache_row])
            atom_global = int(atom_ptr[cache_row]) + int(local_index[j])
            idx = int(local_index[j])
            symbols_local = symbols[cache_row]
            entry: dict[str, Any] = {
                "split": split,
                "official_row_id": row_id,
                "split_cache_row": cache_row,
                "root_index_in_molecule": idx,
                "atom_type": int(atom_type[atom_global]),
                "atom_symbol": symbols_local[idx] if idx < len(symbols_local) else "?",
                "s1_occurrence_range": [int(nbr_ptr[atom_global]), int(nbr_ptr[atom_global + 1])],
                "s2_occurrence_range": [int(path_ptr[atom_global]), int(path_ptr[atom_global + 1])],
                "root_frozen_input_sha256": root_input_digest(cache, cache_row, idx),
            }
            if arm == "A":
                entry["f_of_h_std"] = float(block["contrib_atom"][j])
                entry["f_of_h_raw"] = float(sigma * block["contrib_atom"][j])
            else:
                entry["c0_std"] = float(block["base_root"][j])
                entry["base_root_raw"] = float(sigma * block["base_root"][j])
                if arm == "C":
                    for k, name in enumerate(C.CONTEXT_FIELDS):
                        entry[f"c_{name}_std"] = float(block["context_root"][j, k])
                        entry[f"interaction_root_raw_{name}"] = float(sigma * block["interaction_root"][j, k])
            rows.append(entry)
    return rows


def write_csv_gz(path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Same content as :func:`cr_common.write_csv`, gzip-compressed on disk."""
    import gzip
    import io

    buffer = io.StringIO()
    if not rows:
        text = ""
    else:
        fieldnames = list(columns) if columns else sorted({k for row in rows for k in row})
        buffer.write(",".join(fieldnames) + "\n")
        for row in rows:
            cells = []
            for name in fieldnames:
                value = row.get(name)
                if value is None:
                    cells.append("")
                elif isinstance(value, (dict, list, tuple)):
                    cells.append(json.dumps(C.jsonable(value), separators=(",", ":")))
                else:
                    cells.append(str(value))
            buffer.write(",".join(cells) + "\n")
        text = buffer.getvalue()
    target = Path(path)
    C.ensure_dir(target.parent)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as handle:
        handle.write(text)
    tmp.replace(target)
    return str(target)


def _ledger_csv_rows(molecules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Molecule rows without the nested dicts (those go to the npz + field table)."""
    out = []
    for entry in molecules:
        out.append({k: v for k, v in entry.items() if not isinstance(v, dict)})
    return out


def component_ledger(
    bundle: D.Bundle,
    dataset: str,
    arm: str,
    *,
    splits: Sequence[str] = ("valid", "test"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Machine-readable raw-unit account of every Val/Test molecule and root."""
    checkpoint = C.RESULTS_DIR / dataset / arm / "checkpoint_best_val.pt"
    model, payload = T.load_arm_from_checkpoint(bundle, checkpoint, device=device)
    mu, sigma = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    active = list(bundle.active_global["active_indices"])
    active_names = list(bundle.active_global["active_names"])
    molecules: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []
    closure: dict[str, Any] = {}
    for split in splits:
        out = T.evaluate(model, bundle, split, device, with_ledger=True)
        blocks = out["ledger"]
        cat = lambda key: np.concatenate([b[key] for b in blocks])  # noqa: E731
        pred = cat("pred_std")
        row_ids = out["row_ids"]
        y_raw = out["y_raw"]
        if arm == "A":
            local_sum = cat("local_sum")
            rho = cat("rho")
            total = local_sum + rho
            rho_input = np.concatenate([b["rho_input"] for b in blocks], axis=0)
        else:
            intercept = cat("intercept")
            sum_base = cat("sum_base")
            u_sum = cat("u_sum")
            u_terms = cat("u_terms")
            z_active = cat("z_active")
            total = intercept + sum_base + u_sum
            if arm == "C":
                a_values = cat("a_values")
                sums = cat("context_sums")
                interaction = cat("interaction")
                z_context = cat("z_context")
                total = total + interaction.sum(axis=1)
        raw_total = total * sigma + mu
        err = np.abs(raw_total - (pred * sigma + mu))
        tol = C.trace_tolerance(raw_total)
        closure[split] = {
            "n_rows": int(row_ids.size),
            "max_abs_error_raw": float(err.max()),
            "median_abs_error_raw": float(np.median(err)),
            "tolerance_max": float(np.max(tol)),
            "failed_row_ids": [int(r) for r in row_ids[err > tol].tolist()],
            "pass": bool(np.all(err <= tol)),
        }
        for i, row_id in enumerate(row_ids.tolist()):
            cache_row = int(np.searchsorted(bundle.global_row_ids[split], row_id))
            if int(bundle.global_row_ids[split][cache_row]) != int(row_id):
                raise RuntimeError(f"{dataset}/{split}: row id {row_id} not found in the official index")
            g_raw = np.asarray(bundle.caches[split]["global"], dtype=np.float64)[cache_row]
            entry: dict[str, Any] = {
                "split": split,
                "official_row_id": int(row_id),
                "split_cache_row": cache_row,
                "smiles": bundle.smiles[split][cache_row],
                "y_raw": float(y_raw[i]),
                "pred_raw": float(pred[i] * sigma + mu),
                "n_atoms": int(bundle.n_atoms[split][cache_row]),
                "full_reconstructed_raw": float(raw_total[i]),
                "reconstruction_error": float(err[i]),
                "reconstruction_tolerance": float(tol[i]),
                "reconstruction_pass": bool(err[i] <= tol[i]),
                "global_raw": {name: float(g_raw[idx]) for name, idx in zip(active_names, active)},
                "inactive_global_raw": {
                    bundle.active_global["field_table"][idx]["name"]: float(g_raw[idx])
                    for idx in bundle.active_global["inactive_indices"]
                },
            }
            if arm == "A":
                entry.update(
                    {
                        "intercept_raw": float(mu),
                        "local_sum_raw": float(sigma * local_sum[i]),
                        "rho_raw": float(sigma * rho[i]),
                        "rho_input_standardised": [float(v) for v in rho_input[i]],
                    }
                )
            else:
                entry.update(
                    {
                        "intercept_raw": float(mu + sigma * intercept[i]),
                        "sum_base_raw": float(sigma * sum_base[i]),
                        "u_sum_raw": float(sigma * u_sum[i]),
                        "global_standardised": {
                            name: float(z_active[i, j]) for j, name in enumerate(active_names)
                        },
                    }
                )
                for j, name in enumerate(active_names):
                    entry[f"global_term_raw_{name}"] = float(sigma * u_terms[i, j])
                if arm == "C":
                    for k, name in enumerate(C.CONTEXT_FIELDS):
                        entry[f"a_{name}"] = float(a_values[i, k])
                        entry[f"S_{name}_std"] = float(sums[i, k])
                        entry[f"z_{name}"] = float(z_context[i, k])
                        entry[f"sum_interaction_raw_{name}"] = float(sigma * interaction[i, k])
            molecules.append(entry)
        atoms.extend(_atom_rows(bundle, split, blocks, arm=arm, sigma=sigma))
    meta = {
        "dataset": dataset,
        "arm": arm,
        "units": (
            "raw target units; only the intercept carries mu, roots / interactions / global terms carry sigma only"
        ),
        "n_active_global": len(active),
        "active_global_names": active_names,
        "inactive_global_names": list(bundle.active_global["inactive_names"]),
        "context_fields": list(C.CONTEXT_FIELDS),
        "target_scaler": {"mean": mu, "std": sigma},
        "checkpoint": str(checkpoint.relative_to(C.REPO_ROOT)),
        "checkpoint_sha256": C.sha256_file(checkpoint),
        "checkpoint_state_hash": payload.get("reloaded_state_hash"),
        "closure": closure,
        "explanation_scope": (
            "root-local input environment plus explicitly named global context terms; internal paths "
            "inside a root are not uniquely attributed and overlapping root environments are not "
            "independent chemical fragments"
        ),
        "root_lookup": (
            "every root row carries split_cache_row, root_index_in_molecule, atom_type, its S1/S2 "
            "occurrence ranges into the frozen cache and a sha256 of its 433-d local input row"
        ),
    }
    return {"meta": meta, "molecules": molecules, "atoms": atoms}


# ---------------------------------------------------------------------------
# checkpoint replay
# ---------------------------------------------------------------------------


def verify_checkpoint_replay(bundle: D.Bundle, dataset: str, arm: str, *, device: str = "cpu") -> dict[str, Any]:
    """Reload the selected checkpoint and replay predictions and every ledger."""
    model, payload = T.load_arm_from_checkpoint(
        bundle, C.RESULTS_DIR / dataset / arm / "checkpoint_best_val.pt", device=device
    )
    stored = load_predictions(dataset, arm, "best")
    out: dict[str, Any] = {"dataset": dataset, "arm": arm, "per_split": {}}
    ok = True
    for split in C.SPLITS:
        replayed = T.evaluate(model, bundle, split, device, with_ledger=arm in ("B", "C"))
        pred_close = np.abs(replayed["pred_raw"] - stored[split]["pred_raw"])
        entry = {
            "n_rows": int(replayed["row_ids"].size),
            "prediction_max_abs_diff_raw": float(pred_close.max()),
            "prediction_tolerance": float(np.max(C.trace_tolerance(stored[split]["pred_raw"]))),
            "row_ids_match": bool(np.array_equal(replayed["row_ids"], stored[split]["row_id"])),
        }
        entry["pass"] = bool(
            entry["row_ids_match"] and entry["prediction_max_abs_diff_raw"] <= entry["prediction_tolerance"]
        )
        if arm in ("B", "C"):
            total = np.concatenate([b["intercept"] for b in replayed["ledger"]]) + np.concatenate(
                [b["sum_base"] for b in replayed["ledger"]]
            ) + np.concatenate([b["u_sum"] for b in replayed["ledger"]])
            if arm == "C":
                total = total + np.concatenate([b["interaction"] for b in replayed["ledger"]]).sum(axis=1)
            raw_total = total * float(bundle.scaler["std"]) + float(bundle.scaler["mean"])
            err = np.abs(raw_total - (replayed["pred_std"] * float(bundle.scaler["std"]) + float(bundle.scaler["mean"])))
            entry["ledger_max_abs_error_raw"] = float(err.max())
            entry["ledger_pass"] = bool(np.all(err <= C.trace_tolerance(raw_total)))
            entry["pass"] = bool(entry["pass"] and entry["ledger_pass"])
        ok = ok and entry["pass"]
        out["per_split"][split] = entry
    out["reloaded_state_hash"] = payload.get("reloaded_state_hash")
    out["pass"] = bool(ok and payload.get("reloaded_matches_stored"))
    return out


# ---------------------------------------------------------------------------
# diagnostic 2: fixed local-input intervention
# ---------------------------------------------------------------------------


def _diag_rows(bundle: D.Bundle, split: str = "valid", limit: int = DIAG_N_MOLECULES) -> list[int]:
    path_ptr = np.asarray(bundle.caches[split]["path_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(bundle.caches[split]["atom_ptr"], dtype=np.int64)
    counts = path_ptr[atom_ptr[1:]] - path_ptr[atom_ptr[:-1]]
    order = np.argsort(bundle.global_row_ids[split], kind="stable")
    return [int(r) for r in order if counts[r] > 0][: int(limit)]


def intervention_diagnostic(bundle: D.Bundle, dataset: str, *, device: str = "cpu") -> dict[str, Any]:
    """Remove the molecule's first typed-S2 occurrence; rebuild the delta from the ledger."""
    import torch

    split = "valid"
    rows = _diag_rows(bundle, split)
    mu, sigma = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    models = {
        arm: T.load_arm_from_checkpoint(bundle, C.RESULTS_DIR / dataset / arm / "checkpoint_best_val.pt", device=device)[0]
        for arm in C.ARMS
    }
    cache = bundle.caches[split]
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    records: list[dict[str, Any]] = []
    for row in rows:
        a0 = int(atom_ptr[row])
        drop = int(path_ptr[a0])
        smiles = str(cache["smiles"][row])
        n_atoms = int(atom_ptr[row + 1] - a0)
        identity = path_identity(smiles, cache, drop, row)
        full_cache = molecule_slice(cache, row)
        drop_cache = molecule_slice(cache, row, drop_path=drop)
        record: dict[str, Any] = {
            "official_row_id": int(bundle.global_row_ids[split][row]),
            "split_cache_row": int(row),
            "smiles": smiles,
            "n_atoms": n_atoms,
            "removed_occurrence": identity,
            "intervention_kind": "input-channel intervention: one S2 record removed; raw graph/S1/scalars/g unchanged",
        }
        for arm, model in models.items():
            model.eval()
            with torch.no_grad():
                pred_base, parts_base = model(bundle.collate(full_cache, np.asarray([0], dtype=np.int64)), return_aux=True)
                pred_drop, parts_drop = model(bundle.collate(drop_cache, np.asarray([0], dtype=np.int64)), return_aux=True)
            base_pred = float(pred_base.reshape(-1)[0])
            drop_pred = float(pred_drop.reshape(-1)[0])
            entry: dict[str, Any] = {
                "pred_raw_before": float(base_pred * sigma + mu),
                "pred_raw_after": float(drop_pred * sigma + mu),
                "delta_pred_raw": float((drop_pred - base_pred) * sigma),
            }
            if arm == "A":
                entry["delta_local_sum_raw"] = float((parts_drop["local_sum"] - parts_base["local_sum"]).item() * sigma)
                entry["delta_rho_raw"] = float((parts_drop["rho"] - parts_base["rho"]).item() * sigma)
                entry["rho_responds"] = bool(abs(entry["delta_rho_raw"]) > 0.0)
                entry["note"] = "A keeps a rho response; recorded, not scored as an implementation error"
            else:
                root = identity["root_atom_index"]
                d_base = float(parts_drop["base_root"][root] - parts_base["base_root"][root]) * sigma
                other_base = (
                    float(
                        np.abs(
                            np.delete(parts_drop["base_root"].numpy(), root)
                            - np.delete(parts_base["base_root"].numpy(), root)
                        ).max()
                    )
                    if n_atoms > 1
                    else 0.0
                )
                entry.update(
                    {
                        "delta_base_root_raw": d_base,
                        "other_roots_base_max_abs_change": other_base,
                        "delta_global_terms_max_abs": float(
                            np.abs(parts_drop["u_terms"].numpy() - parts_base["u_terms"].numpy()).max()
                        ),
                    }
                )
                reconstructed = d_base
                if arm == "C":
                    for k, name in enumerate(C.CONTEXT_FIELDS):
                        d_inter = float(
                            parts_drop["interaction_root"][root, k] - parts_base["interaction_root"][root, k]
                        ) * sigma
                        other_inter = (
                            float(
                                np.abs(
                                    np.delete(parts_drop["interaction_root"].numpy()[:, k], root)
                                    - np.delete(parts_base["interaction_root"].numpy()[:, k], root)
                                ).max()
                            )
                            if n_atoms > 1
                            else 0.0
                        )
                        entry[f"delta_interaction_root_raw_{name}"] = d_inter
                        entry[f"other_roots_interaction_max_abs_change_{name}"] = other_inter
                        entry[f"delta_a_{name}"] = float(
                            parts_drop["a_values"][0, k].item() - parts_base["a_values"][0, k].item()
                        )
                        reconstructed += d_inter
                entry["delta_reconstructed_raw"] = float(reconstructed)
                entry["reconstruction_error"] = float(abs(reconstructed - entry["delta_pred_raw"]))
                entry["reconstruction_tolerance"] = float(C.trace_tolerance(entry["delta_pred_raw"]))
                entry["pass"] = bool(
                    entry["reconstruction_error"] <= entry["reconstruction_tolerance"]
                    and entry["other_roots_base_max_abs_change"] <= 1e-6
                    and entry["delta_global_terms_max_abs"] <= 1e-6
                )
                if arm == "C":
                    entry["pass"] = bool(
                        entry["pass"]
                        and all(
                            entry[f"other_roots_interaction_max_abs_change_{name}"] <= 1e-6
                            and abs(entry[f"delta_a_{name}"]) <= 1e-9
                            for name in C.CONTEXT_FIELDS
                        )
                    )
            record[arm] = entry
        records.append(record)
    failures = [
        {"official_row_id": r["official_row_id"], "arm": arm}
        for r in records
        for arm in ("B", "C")
        if not r[arm].get("pass", False)
    ]
    return {
        "dataset": dataset,
        "split": split,
        "n_molecules": len(records),
        "selection_rule": "Val molecules with at least one typed-S2 occurrence, ascending official row_id, first 32",
        "removed_occurrence_rule": "the molecule's first typed-S2 record (cache path index of its atom 0)",
        "records": records,
        "n_failures": len(failures),
        "failures": failures,
        "all_pass": bool(not failures),
    }


# ---------------------------------------------------------------------------
# diagnostic 3: context-conditioned response
# ---------------------------------------------------------------------------


def _single_function_value(fns, index: int, z: float) -> float:
    import torch

    w1, b1, w2 = fns.w1[index], fns.b1[index], fns.w2[index]
    hidden = torch.nn.functional.silu(w1 * float(z) + b1) - torch.nn.functional.silu(b1)
    return float((hidden * w2).sum())


def context_response_diagnostic(bundle: D.Bundle, dataset: str, *, device: str = "cpu") -> dict[str, Any]:
    """Move one context to its Train Q25/Q75 with h fixed; verify the closed form."""
    import torch

    split = "valid"
    rows = _diag_rows(bundle, split)
    mu, sigma = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    model, _payload = T.load_arm_from_checkpoint(
        bundle, C.RESULTS_DIR / dataset / "C" / "checkpoint_best_val.pt", device=device
    )
    model.eval()
    active = list(bundle.active_global["active_indices"])
    column_of = {name: D.field_index(bundle.spec, name) for name in C.CONTEXT_FIELDS}
    slot_of = {name: active.index(column_of[name]) for name in C.CONTEXT_FIELDS}
    q = {
        name: (bundle.contexts["per_field"][name]["z_q25"], bundle.contexts["per_field"][name]["z_q75"])
        for name in C.CONTEXT_FIELDS
    }
    cache = bundle.caches[split]
    records: list[dict[str, Any]] = []
    for row in rows:
        batch = bundle.collate(molecule_slice(cache, row), np.asarray([0], dtype=np.int64))
        with torch.no_grad():
            _pred, parts = model(batch, return_aux=True)
        base_pred = float(parts["pred_std"].reshape(-1)[0])
        sums = parts["context_sums"].numpy().reshape(-1).copy()
        record: dict[str, Any] = {
            "official_row_id": int(bundle.global_row_ids[split][row]),
            "split_cache_row": int(row),
            "smiles": str(cache["smiles"][row]),
            "n_atoms": int(bundle.n_atoms[split][row]),
            "pred_raw_base": float(base_pred * sigma + mu),
            "contexts": {},
        }
        ok = True
        for k, name in enumerate(C.CONTEXT_FIELDS):
            q25, q75 = q[name]
            slot = int(slot_of[name])
            z_obs = float(parts["z_context"][0, k])
            u_obs = _single_function_value(model.global_functions, slot, z_obs)
            u25 = _single_function_value(model.global_functions, slot, q25)
            u75 = _single_function_value(model.global_functions, slot, q75)
            a_obs = _single_function_value(model.context_functions, k, z_obs)
            a25 = _single_function_value(model.context_functions, k, q25)
            a75 = _single_function_value(model.context_functions, k, q75)
            # the closed form of *this* intervention: observed state -> Q75
            analytical = (u75 - u_obs) + (a75 - a_obs) * float(sums[k])
            # independent recomputation: a full forward with that raw global column replaced
            patched = dict(batch)
            global_patched = batch["global"].clone()
            column = int(column_of[name])
            gmean = float(bundle.norm["global_mean"][column])
            gscale = float(bundle.norm["global_scale"][column])
            global_patched[0, column] = float(q75 * gscale + gmean)
            patched["global"] = global_patched
            with torch.no_grad():
                pred75 = float(model(patched).reshape(-1)[0])
            forward_delta = pred75 - base_pred
            entry = {
                "column_index": column,
                "z_observed": float(parts["z_context"][0, k]),
                "z_q25": float(q25),
                "z_q75": float(q75),
                "a_at_observed": a_obs,
                "a_at_q25": a25,
                "a_at_q75": a75,
                "S_k_std": float(sums[k]),
                "u_at_observed_std": u_obs,
                "u_at_q25_std": u25,
                "u_at_q75_std": u75,
                "observed_to_q75_delta_u_std": float(u75 - u_obs),
                "observed_to_q75_delta_a": float(a75 - a_obs),
                "observed_to_q75_delta_interaction_std": float((a75 - a_obs) * float(sums[k])),
                "q25_to_q75_delta_u_std": float(u75 - u25),
                "q25_to_q75_delta_a": float(a75 - a25),
                "q25_to_q75_delta_interaction_std": float((a75 - a25) * float(sums[k])),
                "delta_u_std": float(u75 - u_obs),
                "delta_a": float(a75 - a_obs),
                "delta_interaction_std": float((a75 - a_obs) * float(sums[k])),
                "delta_pred_std_analytical": float(analytical),
                "delta_pred_raw_analytical": float(analytical * sigma),
                "delta_pred_raw_forward_q75": float(forward_delta * sigma),
                "forward_vs_analytical_abs_diff_raw": float(abs(forward_delta - analytical) * sigma),
                "forward_vs_analytical_tolerance": float(C.trace_tolerance(analytical * sigma)),
            }
            entry["pass"] = bool(entry["forward_vs_analytical_abs_diff_raw"] <= entry["forward_vs_analytical_tolerance"])
            ok = ok and entry["pass"]
            record["contexts"][name] = entry
        record["pass"] = bool(ok)
        record["local_terms_unchanged"] = True
        records.append(record)
    return {
        "dataset": dataset,
        "split": split,
        "n_molecules": len(records),
        "context_fields": list(C.CONTEXT_FIELDS),
        "quantiles": list(C.CONTEXT_QUANTILES),
        "quantile_source": "Train raw values of the same norm, converted with the frozen Train mean/scale",
        "records": records,
        "interaction_response_rms_std": {
            name: float(
                np.sqrt(
                    np.mean(
                        [r["contexts"][name]["delta_interaction_std"] ** 2 for r in records]
                    )
                )
            )
            if records
            else 0.0
            for name in C.CONTEXT_FIELDS
        },
        "global_main_effect_rms_std": {
            name: float(
                np.sqrt(np.mean([r["contexts"][name]["delta_u_std"] ** 2 for r in records]))
            )
            if records
            else 0.0
            for name in C.CONTEXT_FIELDS
        },
        "all_pass": bool(all(r["pass"] for r in records)),
        "n_failures": int(sum(0 if r["pass"] else 1 for r in records)),
        "note": (
            "conditional model interventions on inputs that need not be real molecules; global main "
            "effect and interaction effect are separate columns and must not be merged"
        ),
    }


# ---------------------------------------------------------------------------
# diagnostic 4: constant / averaged interaction?
# ---------------------------------------------------------------------------


def interaction_diagnostics(bundle: D.Bundle, dataset: str, *, device: str = "cpu") -> dict[str, Any]:
    model, _payload = T.load_arm_from_checkpoint(
        bundle, C.RESULTS_DIR / dataset / "C" / "checkpoint_best_val.pt", device=device
    )
    sigma = float(bundle.scaler["std"])
    train_out = T.evaluate(model, bundle, "train", device, with_ledger=True)
    train_root = np.concatenate([b["context_root"] for b in train_out["ledger"]], axis=0)
    cbar = train_root.mean(axis=0)
    out: dict[str, Any] = {
        "dataset": dataset,
        "train_root_count": int(train_root.shape[0]),
        "cbar_train_node_equal_weight_mean": {name: float(cbar[k]) for k, name in enumerate(C.CONTEXT_FIELDS)},
        "per_split": {},
        "labels": {},
        "note": (
            "I_k = a_k * [n_atoms * cbar_k + sum_v (c_k - cbar_k)] with cbar_k from Train roots; this "
            "is a reference decomposition, not a refit. RMS ratios can exceed 1 through cancellation "
            "and are not exclusive error-explanation percentages"
        ),
    }
    for split in C.SPLITS:
        res = train_out if split == "train" else T.evaluate(model, bundle, split, device, with_ledger=True)
        blocks = res["ledger"]
        interaction = np.concatenate([b["interaction"] for b in blocks])
        a_values = np.concatenate([b["a_values"] for b in blocks])
        counts = np.concatenate([b["atom_counts"] for b in blocks])
        split_entry: dict[str, Any] = {}
        for k, name in enumerate(C.CONTEXT_FIELDS):
            i_k = interaction[:, k]
            # per-block: ``atom_batch`` indexes the graphs *of that block* only
            deviation = np.empty(counts.size)
            per_mol_std = np.empty(counts.size)
            per_mol_rms = np.empty(counts.size)
            cursor = 0
            for block in blocks:
                ck = block["context_root"][:, k]
                atom_batch = block["atom_batch"]
                n_here = int(block["atom_counts"].size)
                for g in range(n_here):
                    values = ck[atom_batch == g]
                    index = cursor + g
                    deviation[index] = float((values - float(cbar[k])).sum()) if values.size else 0.0
                    per_mol_std[index] = float(values.std(ddof=0)) if values.size > 1 else 0.0
                    per_mol_rms[index] = float(np.sqrt(np.mean(values * values))) if values.size else 0.0
                cursor += n_here
            if cursor != counts.size:
                raise RuntimeError(f"{dataset}/{split}: graph count mismatch {cursor} != {counts.size}")
            ratio = per_mol_std / np.maximum(per_mol_rms, 1e-12)
            multi = counts >= 2
            constant_part = a_values[:, k] * counts.astype(np.float64) * float(cbar[k])
            deviation_part = a_values[:, k] * deviation
            split_entry[name] = {
                "rms_interaction_std": float(np.sqrt(np.mean(i_k * i_k))),
                "rms_interaction_raw": float(np.sqrt(np.mean(i_k * i_k)) * sigma),
                "rms_a": float(np.sqrt(np.mean(a_values[:, k] ** 2))),
                "root_std_over_rms_median": float(np.median(ratio[multi])) if multi.any() else None,
                "n_molecules_with_one_root": int((~multi).sum()),
                "n_molecules_multi_root": int(multi.sum()),
                "constant_part_rms_std": float(np.sqrt(np.mean(constant_part ** 2))),
                "deviation_part_rms_std": float(np.sqrt(np.mean(deviation_part ** 2))),
                "constant_part_rms_raw": float(np.sqrt(np.mean(constant_part ** 2)) * sigma),
                "deviation_part_rms_raw": float(np.sqrt(np.mean(deviation_part ** 2)) * sigma),
                "decomposition_max_abs_error_std": float(np.abs(constant_part + deviation_part - i_k).max()),
            }
        out["per_split"][split] = split_entry
    return out


# ---------------------------------------------------------------------------
# diagnostic 5: exact a_k / c_k scale equivalence
# ---------------------------------------------------------------------------


def equivalence_scale(bundle: D.Bundle, dataset: str, *, device: str = "cpu") -> dict[str, Any]:
    import torch

    model, _payload = T.load_arm_from_checkpoint(
        bundle, C.RESULTS_DIR / dataset / "C" / "checkpoint_best_val.pt", device=device
    )
    model.eval()
    split = "valid"
    order, row_ids, _y = T.eval_rows(bundle, split)
    batch = bundle.collate(bundle.caches[split], order)
    mu, sigma = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    with torch.no_grad():
        pred, parts = model(batch, return_aux=True)
        base = {
            "pred_raw": pred.numpy() * sigma + mu,
            "per_root_interaction": parts["interaction_root"].numpy(),
            "a_values": parts["a_values"].numpy(),
        }
        model.context_functions.w2.data.mul_(2.0)
        model.w_context.data.mul_(0.5)
        pred2, parts2 = model(batch, return_aux=True)
        scaled = {
            "pred_raw": pred2.numpy() * sigma + mu,
            "per_root_interaction": parts2["interaction_root"].numpy(),
            "a_values": parts2["a_values"].numpy(),
        }
    pred_close = {
        "max_abs_diff": float(np.abs(base["pred_raw"] - scaled["pred_raw"]).max()),
        "tolerance": float(np.max(C.trace_tolerance(base["pred_raw"]))),
    }
    pred_close["pass"] = bool(pred_close["max_abs_diff"] <= pred_close["tolerance"])
    root_close = {
        "max_abs_diff": float(np.abs(base["per_root_interaction"] - scaled["per_root_interaction"]).max()),
        "tolerance": 1e-5,
    }
    root_close["pass"] = bool(root_close["max_abs_diff"] <= root_close["tolerance"])
    nonzero = np.abs(base["a_values"]) > 1e-12
    ratio = scaled["a_values"][nonzero] / base["a_values"][nonzero]
    a_ratio = {"min": float(ratio.min()), "max": float(ratio.max()), "n": int(ratio.size)}
    return {
        "dataset": dataset,
        "split": split,
        "n_rows": int(row_ids.size),
        "transform": "a_k output x2 (context_functions.w2 x2) and c_k output x1/2 (w_context x1/2)",
        "prediction": pred_close,
        "per_root_interaction": root_close,
        "a_ratio": a_ratio,
        "a_ratio_is_two": bool(abs(a_ratio["min"] - 2.0) < 1e-5 and abs(a_ratio["max"] - 2.0) < 1e-5),
        "pass": bool(pred_close["pass"] and root_close["pass"] and abs(a_ratio["min"] - 2.0) < 1e-5),
        "note": (
            "so individual a_k / c_k parameter norms are not importance measures; the transform does "
            "not resolve every attribution non-uniqueness"
        ),
    }


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def verdict_performance(dataset: str, traj: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    a, c = traj["A"], traj["C"]
    sel_ratio = {s: _ratio(c["selected"][f"{s}_rmse_raw"], a["selected"][f"{s}_rmse_raw"]) for s in ("valid", "test")}
    l20_ratio = {
        s: _ratio(c["last20"][f"mean_last20_{s}_rmse_raw"], a["last20"][f"mean_last20_{s}_rmse_raw"])
        for s in ("valid", "test")
    }
    both_point = all(v <= RETENTION_TOL for v in sel_ratio.values())
    both_l20 = all(v <= RETENTION_TOL for v in l20_ratio.values())
    straddles = (sel_ratio["valid"] <= RETENTION_TOL) != (sel_ratio["test"] <= RETENTION_TOL)
    if both_point:
        label = "PRESERVED_POINT_ESTIMATE" + (" / HORIZON_CONSISTENT" if both_l20 else " / CHECKPOINT_SENSITIVE")
    elif straddles:
        label = "SPLIT_SENSITIVE"
    else:
        label = "PERFORMANCE_LOSS_THIS_RECIPE"
    return {
        "dataset": dataset,
        "axis": "performance_retention_C_vs_A",
        "tolerance": RETENTION_TOL,
        "selected_ratio_C_over_A": sel_ratio,
        "last20_ratio_C_over_A": l20_ratio,
        "label": label,
        "note": "single-seed development judgement at a 5 % practical tolerance, not an equivalence test",
    }


def verdict_context(
    dataset: str,
    traj: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    diag4: Mapping[str, Any],
) -> dict[str, Any]:
    """C vs B on four pre-declared surfaces; no sign is hidden by an average."""
    b, c = traj["B"], traj["C"]
    gain = {s: 1.0 - _ratio(c["selected"][f"{s}_rmse_raw"], b["selected"][f"{s}_rmse_raw"]) for s in ("valid", "test")}
    l20_gain = {
        s: 1.0 - _ratio(c["last20"][f"mean_last20_{s}_rmse_raw"], b["last20"][f"mean_last20_{s}_rmse_raw"])
        for s in ("valid", "test")
    }
    surfaces = {
        "selected_valid": gain["valid"],
        "selected_test": gain["test"],
        "last20_valid": l20_gain["valid"],
        "last20_test": l20_gain["test"],
    }
    positive = [name for name, value in surfaces.items() if value >= CONTEXT_GAIN_MIN]
    negative = [name for name, value in surfaces.items() if value < 0.0]
    strong = all(value >= CONTEXT_GAIN_MIN for value in gain.values()) and all(
        value >= 0.0 for value in l20_gain.values()
    )
    no_signal = all(value < CONTEXT_GAIN_MIN for value in surfaces.values())
    intervals = {s: bootstrap[s]["contrasts"].get("C-B", {}).get("rmse_raw") for s in ("valid", "test")}
    uncertain = any(v is not None and v["contains_zero"] for v in intervals.values())
    if strong:
        label = "CONTEXT_PERFORMANCE_POINT_SIGNAL" + (" / UNCERTAIN_FIXED_STATE" if uncertain else "")
    elif no_signal:
        label = "NO_MATERIAL_CONTEXT_GAIN_THIS_RECIPE"
    else:
        label = "SPLIT_OR_SELECTION_SENSITIVE"
    return {
        "dataset": dataset,
        "axis": "context_increment_C_vs_B",
        "threshold": CONTEXT_GAIN_MIN,
        "relative_gain_by_surface": surfaces,
        "relative_gain_C_over_B": gain,
        "last20_relative_gain_C_over_B": l20_gain,
        "surfaces_above_threshold": positive,
        "surfaces_below_zero": negative,
        "sign_disagrees_between_splits": bool(gain["valid"] * gain["test"] < 0.0),
        "paired_intervals": intervals,
        "label": label,
        "interaction_readout": diag4.get("labels", {}),
        "note": (
            "a performance label without a measurable interaction response cannot attribute the change "
            "to the interaction mechanism; matched parameter budgets are not unique causal attribution. "
            "A label of SPLIT_OR_SELECTION_SENSITIVE means the sign is not stable across Val/Test or "
            "across the selected state and the last-20 horizon, so no context gain is claimed."
        ),
    }


def verdict_gine(dataset: str, traj: Mapping[str, Mapping[str, Any]], gine: Mapping[str, Any]) -> dict[str, Any]:
    """Every arm against the frozen GINE coordinate, never only the best one."""
    arms: dict[str, Any] = {}
    for arm in C.ARMS:
        entry: dict[str, Any] = {}
        for split in ("valid", "test"):
            own = float(traj[arm]["selected"][f"{split}_rmse_raw"])
            ref = float(gine["selected"][f"{split}_rmse_raw"])
            entry[split] = {
                "rmse_raw": own,
                "gine_rmse_raw": ref,
                "ratio_over_gine": _ratio(own, ref),
                "abs_gap": own - ref,
            }
        entry["within_5pct_both_splits"] = bool(
            entry["valid"]["ratio_over_gine"] <= RETENTION_TOL and entry["test"]["ratio_over_gine"] <= RETENTION_TOL
        )
        arms[arm] = entry
    qualified = [arm for arm, entry in arms.items() if entry["within_5pct_both_splits"]]
    return {
        "dataset": dataset,
        "axis": "competitiveness_vs_G",
        "reference": "REUSED_FROZEN_REFERENCE",
        "arms": arms,
        "arms_within_5pct_both_splits": qualified,
        "label": (
            "WITHIN_5PCT_OF_THIS_GINE (" + ", ".join(qualified) + ")" if qualified else "BEHIND_THIS_GINE"
        ),
        "caveat": (
            "G is a strong performance coordinate with a different input path and a stronger learning-rate "
            "preset (1e-3 vs the frozen 3e-4); no gap is attributable to a single operator and no SOTA claim "
            "is made in either direction"
        ),
    }


def verdict_trace(
    dataset: str,
    ledger: Mapping[str, Mapping[str, Any]],
    intervention: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    replay: Mapping[str, Any],
    pretrain_ok: bool,
) -> dict[str, Any]:
    closure_ok = all(
        ledger[arm]["meta"]["closure"][split]["pass"] for arm in ("B", "C") for split in ("valid", "test")
    )
    ok = bool(
        pretrain_ok
        and closure_ok
        and intervention["all_pass"]
        and equivalence["pass"]
        and all(replay[arm]["pass"] for arm in C.ARMS)
    )
    return {
        "dataset": dataset,
        "axis": "trace",
        "label": "TRACE_PASS" if ok else "TRACE_FAIL",
        "prediction_reconstruction": {arm: ledger[arm]["meta"]["closure"] for arm in ("B", "C")},
        "intervention_reconstruction": {
            "n_molecules": intervention["n_molecules"],
            "n_failures": intervention["n_failures"],
        },
        "scale_equivalence": equivalence["pass"],
        "checkpoint_replay": {arm: replay[arm]["pass"] for arm in C.ARMS},
        "pretrain_checks": bool(pretrain_ok),
        "note": "TRACE_PASS is bookkeeping closure, not proof of unique or chemically real contributions",
    }


def interaction_labels(diag4: Mapping[str, Any], diag3: Mapping[str, Any]) -> dict[str, Any]:
    labels = {}
    for name in C.CONTEXT_FIELDS:
        rms_val = diag4["per_split"]["valid"][name]["rms_interaction_std"]
        rms_test = diag4["per_split"]["test"][name]["rms_interaction_std"]
        rms_diag3 = diag3["interaction_response_rms_std"][name]
        resolved = bool(rms_val >= RESOLVE_RMS_MIN and rms_diag3 >= RESOLVE_RMS_MIN)
        ratio_median = diag4["per_split"]["valid"][name]["root_std_over_rms_median"]
        if ratio_median is None:
            specificity = "UNDEFINED_SINGLE_ROOT_ONLY"
        elif ratio_median < ROOT_SPECIFICITY_MAX:
            specificity = "LOW_ROOT_SPECIFICITY_AT_5PCT"
        else:
            specificity = "NONUNIFORM_ROOT_RESPONSE"
        labels[name] = {
            "rms_interaction_std_valid": rms_val,
            "rms_interaction_std_test": rms_test,
            "rms_interaction_diagnostic3_std": rms_diag3,
            "global_main_effect_rms_std": diag3["global_main_effect_rms_std"][name],
            "resolution_threshold_std": RESOLVE_RMS_MIN,
            "response_label": "RESOLVED_FUNCTIONAL_RESPONSE" if resolved else "WEAK_AT_1PCT_TARGET_SCALE",
            "root_specificity_label": specificity,
            "root_std_over_rms_median_valid": ratio_median,
        }
    return labels


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_curves(dataset: str, traj: Mapping[str, Any], gine: Mapping[str, Any] | None, path) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, split in zip(axes, C.SPLITS):
        for arm, colour in zip(("A", "B", "C"), ("#1f77b4", "#2ca02c", "#d62728")):
            rows = traj[arm]["curves"]
            ax.plot([r["epoch"] for r in rows], [r[f"{split}_rmse_raw"] for r in rows], color=colour, lw=1.2, label=f"{arm} (this round)")
            best = traj[arm]["selected"]["epoch"]
            value = traj[arm]["selected"][f"{split}_rmse_raw"]
            ax.scatter([best], [value], color=colour, s=34, zorder=5)
            ax.annotate(f"{arm}@{best}", (best, value), textcoords="offset points", xytext=(4, 4), fontsize=7, color=colour)
        if gine is not None:
            ax.plot([r["epoch"] for r in gine["curves"]], [r[f"{split}_rmse_raw"] for r in gine["curves"]],
                    color="#7f7f7f", lw=1.0, ls="--", label=f"G reused, lr={gine['selected_peak_lr']}")
            ax.scatter([gine["selected"]["epoch"]], [gine["selected"][f"{split}_rmse_raw"]],
                       color="#7f7f7f", marker="s", s=30, zorder=5)
            ax.annotate(f"G@{gine['selected']['epoch']}", (gine["selected"]["epoch"], gine["selected"][f"{split}_rmse_raw"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color="#555555")
        ax.set_title(f"{dataset} — {split}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE (raw units)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_context_functions(bundle: D.Bundle, dataset: str, path) -> dict[str, Any]:
    import torch

    plt = _plt()
    model, _payload = T.load_arm_from_checkpoint(bundle, C.RESULTS_DIR / dataset / "C" / "checkpoint_best_val.pt")
    model.eval()
    fig, axes = plt.subplots(1, len(C.CONTEXT_FIELDS), figsize=(11, 3.6))
    if len(C.CONTEXT_FIELDS) == 1:
        axes = [axes]
    exported: dict[str, list[dict[str, float]]] = {}
    for k, name in enumerate(C.CONTEXT_FIELDS):
        column = int(bundle.contexts["per_field"][name]["index"])
        raw = np.asarray(bundle.caches["train"]["global"], dtype=np.float64)[:, column]
        lo, hi = np.percentile(raw, [1.0, 99.0])
        grid = np.linspace(lo, hi, CONTEXT_CURVE_POINTS)
        z = (grid - float(bundle.norm["global_mean"][column])) / float(bundle.norm["global_scale"][column])
        # every a_k depends on its own context only; evaluate the whole family on
        # a matrix whose column k holds the grid of field k
        z_path = np.repeat(z.reshape(-1, 1), len(C.CONTEXT_FIELDS), axis=1)
        with torch.no_grad():
            values = model.context_functions(torch.tensor(z_path, dtype=torch.float32))
        a_vals = [float(v) for v in values[:, k].numpy()]
        exported[name] = [
            {"raw_value": float(r), "standardised_value": float(zz), "a_k": float(av)}
            for r, zz, av in zip(grid, z, a_vals)
        ]
        axes[k].plot(grid, a_vals, color=CONTEXT_COLOURS[k % len(CONTEXT_COLOURS)], lw=1.6)
        axes[k].axhline(0.0, color="#bdc3c7", lw=0.8)
        for q in C.CONTEXT_QUANTILES:
            axes[k].axvline(float(np.percentile(raw, q)), color=NEUTRAL_COLOUR, ls=":", lw=0.8)
        axes[k].set_title(f"a({name}) over Train Q1–Q99")
        axes[k].set_xlabel(name)
        axes[k].set_ylabel("a_k (standardised target)")
        axes[k].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": str(path), "grid": exported}


def _draw_molecule(ax, smiles: str, base_root_raw: np.ndarray, interaction_total: np.ndarray, title: str) -> None:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, "SMILES parse failed", ha="center")
        ax.axis("off")
        return
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {a.GetIdx(): conf.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms()}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot([pos[i].x, pos[j].x], [pos[i].y, pos[j].y], color="#95a5a6", lw=1.4, zorder=1)
    limit = max(1e-9, float(np.abs(base_root_raw).max()) if base_root_raw.size else 1.0)
    cmap = _plt().get_cmap("coolwarm")
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        value = float(base_root_raw[i]) if i < base_root_raw.size else 0.0
        colour = cmap(float(0.5 + 0.5 * np.clip(value / limit, -1, 1)))
        strong = abs(float(interaction_total[i])) if i < interaction_total.size else 0.0
        ax.scatter([pos[i].x], [pos[i].y], s=620, color=[colour], edgecolor="#2c3e50",
                   linewidth=2.2 if strong > 0.01 else 0.4, zorder=2)
        ax.text(pos[i].x, pos[i].y, f"{i}", ha="center", va="center", fontsize=7, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def figure_cases(bundle: D.Bundle, dataset: str, ledger: Mapping[str, Any], *, limit: int = 3) -> dict[str, Any]:
    """Three frozen display cases for arm C on Val."""
    plt = _plt()
    molecules = [m for m in ledger["molecules"] if m["split"] == "valid"]
    atoms = [a for a in ledger["atoms"] if a["split"] == "valid"]
    path_ptr = np.asarray(bundle.caches["valid"]["path_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(bundle.caches["valid"]["atom_ptr"], dtype=np.int64)
    has_path = (path_ptr[atom_ptr[1:]] - path_ptr[atom_ptr[:-1]]) > 0
    with_path = sorted([m for m in molecules if has_path[m["split_cache_row"]]], key=lambda m: m["official_row_id"])
    candidates: list[tuple[str, dict[str, Any]]] = []
    if with_path:
        candidates.append(("smallest_row_id_with_path", with_path[0]))
        sizes = np.asarray([m["n_atoms"] for m in with_path], dtype=np.float64)
        median_size = float(np.median(sizes))
        candidates.append(
            (
                "median_n_atoms_with_path",
                min(with_path, key=lambda m: (abs(m["n_atoms"] - median_size), m["official_row_id"])),
            )
        )
    if molecules:
        candidates.append(
            ("largest_C_abs_error_vs_y", max(molecules, key=lambda m: (abs(m["pred_raw"] - m["y_raw"]), m["official_row_id"])))
        )
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rule, entry in candidates:
        if entry["official_row_id"] in seen:
            continue
        seen.add(entry["official_row_id"])
        chosen.append({"rule": rule, **entry})
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for entry in molecules:
            if entry["official_row_id"] in seen:
                continue
            seen.add(entry["official_row_id"])
            chosen.append({"rule": "row_id_fill", **entry})
            if len(chosen) >= limit:
                break
    figures = []
    for n, case in enumerate(chosen):
        root_rows = [a for a in atoms if a["official_row_id"] == case["official_row_id"]]
        root_rows.sort(key=lambda a: a["root_index_in_molecule"])
        base_root_raw = np.asarray([a["base_root_raw"] for a in root_rows], dtype=np.float64)
        inter = (
            np.stack(
                [
                    np.asarray([a[f"interaction_root_raw_{name}"] for a in root_rows], dtype=np.float64)
                    for name in C.CONTEXT_FIELDS
                ],
                axis=1,
            )
            if root_rows
            else np.zeros((0, len(C.CONTEXT_FIELDS)))
        )
        inter_total = inter.sum(axis=1) if inter.size else np.zeros(0)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1.5, 1.4]})
        _draw_molecule(
            axes[0],
            case["smiles"],
            base_root_raw,
            inter_total,
            f"row {case['official_row_id']} — fill = base_root_raw\n(thick outline = |interaction| > 0.01)",
        )
        labels = [("intercept (mu + sigma b)", case["intercept_raw"]), ("sum base over roots", case["sum_base_raw"])]
        for name in C.CONTEXT_FIELDS:
            labels.append((f"interaction {name}", case[f"sum_interaction_raw_{name}"]))
        labels.append(("sum single-variable global", case["u_sum_raw"]))
        labels.append(("reconstructed total", case["full_reconstructed_raw"]))
        values = [v for _, v in labels]
        colours = [POSITIVE_COLOUR if v > 0 else NEGATIVE_COLOUR for v in values]
        axes[1].barh([l for l, _ in labels], values, color=colours, alpha=0.85)
        axes[1].axvline(0.0, color="#2c3e50", lw=0.8)
        axes[1].set_title(
            f"y={case['y_raw']:.4f}   pred={case['pred_raw']:.4f}   n_atoms={case['n_atoms']}\n"
            f"a_size={case['a_log1p_n_atoms']:.4f}   a_ring={case['a_ring_atom_fraction']:.4f}",
            fontsize=9,
        )
        axes[1].grid(axis="x", alpha=0.25)
        if root_rows:
            idx = np.arange(len(root_rows))
            axes[2].bar(idx, base_root_raw, color="#5d6d7e", label="base_root_raw")
            bottom_pos = np.zeros(len(root_rows))
            bottom_neg = np.zeros(len(root_rows))
            for k, name in enumerate(C.CONTEXT_FIELDS):
                values_k = inter[:, k]
                pos = np.where(values_k >= 0, values_k, 0.0)
                neg = np.where(values_k < 0, values_k, 0.0)
                axes[2].bar(idx, pos, bottom=bottom_pos, color=CONTEXT_COLOURS[k % len(CONTEXT_COLOURS)], alpha=0.85, label=f"interaction {name}")
                axes[2].bar(idx, neg, bottom=bottom_neg, color=CONTEXT_COLOURS[k % len(CONTEXT_COLOURS)], alpha=0.85)
                bottom_pos += pos
                bottom_neg += neg
            axes[2].axhline(0.0, color="#2c3e50", lw=0.8)
            axes[2].set_xticks(idx)
            axes[2].set_xlabel("root index in molecule")
            axes[2].set_ylabel("raw contribution")
            axes[2].set_title("per-root account (raw units)", fontsize=9)
            axes[2].legend(fontsize=7)
            axes[2].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = OUT / "figures" / f"case_{dataset}_{n}_{case['official_row_id']}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        case_copy = dict(case)
        case_copy["global_raw_subset"] = {
            k: v for k, v in case["global_raw"].items() if abs(v) > 1e-12
        }
        case_copy["figure"] = str(out.relative_to(C.REPO_ROOT))
        case_copy["roots"] = root_rows
        case_copy["display_case_note"] = (
            "the largest-error case is chosen by looking at labels and is a display case only; it does "
            "not support a population-level improvement claim"
            if case["rule"].startswith("largest")
            else "fixed display rule, not a favourable selection"
        )
        C.atomic_write_json(OUT / "cases" / f"{dataset}_{case['official_row_id']}.json", case_copy)
        figures.append(case_copy)
    return {"dataset": dataset, "cases": figures}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def aggregate(
    *,
    datasets: Sequence[str] = C.DATASETS,
    ledger: C.Ledger | None = None,
    with_figures: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    _ensure_dirs()
    checks_path = OUT / "checks.json"
    checks = C.read_json(checks_path) if checks_path.is_file() else {"all_pass": False, "checks": []}
    # the two workers run in parallel, so the merged checks.json is not a reliable
    # medium: prefer the per-dataset files each worker wrote under its own name
    per_dataset_checks = {}
    for dataset in datasets:
        path = OUT / f"checks_{dataset}.json"
        if path.is_file():
            payload = C.read_json(path)
            per_dataset_checks[dataset] = payload
    pretrain_ok = {
        dataset: bool(
            per_dataset_checks[dataset].get("all_pass")
            if dataset in per_dataset_checks
            else checks.get("all_pass")
        )
        for dataset in datasets
    }
    checks = {
        "all_pass": bool(all(pretrain_ok.values())),
        "per_dataset": {
            dataset: {
                "path": str((OUT / f"checks_{dataset}.json").relative_to(C.REPO_ROOT)),
                "n_checks": payload.get("n_checks"),
                "all_pass": payload.get("all_pass"),
                "failed": payload.get("failed"),
            }
            for dataset, payload in per_dataset_checks.items()
        },
        "merged_path": str(checks_path.relative_to(C.REPO_ROOT)),
    }

    main_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    bootstraps: dict[str, Any] = {}
    drop_one: dict[str, Any] = {}
    ledgers: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    diag4: dict[str, Any] = {}
    equiv: dict[str, Any] = {}
    reference_l: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}
    case_figures: dict[str, Any] = {}

    for dataset in datasets:
        bundle = D.build_bundle(dataset)
        traj = {arm: load_trajectory(dataset, arm) for arm in C.ARMS}
        gine = load_gine(dataset)
        for arm in C.ARMS:
            main_rows.append(_arm_row(dataset, arm, traj[arm]))
        if gine is not None:
            main_rows.append(_gine_row(dataset, gine))
        ref_l = load_reference_l(dataset)
        if ref_l is None:
            reference_l[dataset] = None
        else:
            a_sel, l_sel = traj["A"]["selected"], ref_l["selected"]
            a_l20, l_l20 = traj["A"]["last20"], ref_l["last20"]
            reference_l[dataset] = {
                "arm": "A_vs_REUSED_L",
                "source": ref_l["source"],
                "variant": ref_l["variant"],
                "n_parameters_match": bool(ref_l["n_parameters"] == traj["A"]["n_parameters"]),
                "recipe_peak_lr_match": bool(
                    abs(float(ref_l["recipe"]["peak_lr"]) - float(T.RECIPE["peak_lr"])) < 1e-12
                ),
                "recipe_epochs_match": bool(int(ref_l["recipe"]["epochs"]) == int(T.RECIPE["epochs"])),
                "intentional_difference": "batch order (rng(0) permutation here, shuffle_offset=917000 there)",
                "reference_selected": l_sel,
                "this_round_selected": a_sel,
                "selected_epoch_delta": int(a_sel["epoch"]) - int(l_sel["epoch"]),
                "selected_valid_relative_delta": float(
                    a_sel["valid_rmse_raw"] / l_sel["valid_rmse_raw"] - 1.0
                ),
                "selected_test_relative_delta": float(a_sel["test_rmse_raw"] / l_sel["test_rmse_raw"] - 1.0),
                "last20_valid_relative_delta": float(
                    a_l20["mean_last20_valid_rmse_raw"] / l_l20["mean_last20_valid_rmse_raw"] - 1.0
                ),
                "last20_test_relative_delta": float(
                    a_l20["mean_last20_test_rmse_raw"] / l_l20["mean_last20_test_rmse_raw"] - 1.0
                ),
                "note": (
                    "identity check of the frozen loop; the prompt froze the rng(0) batch-order convention, "
                    "so a small difference is expected and is not a model change"
                ),
            }
        contrast_rows.extend(contrast_entries(dataset, traj, gine))

        predictions = {arm: load_predictions(dataset, arm) for arm in C.ARMS}
        if gine is not None:
            predictions["G"] = gine["predictions"]
        for split in C.SPLITS:
            reference = predictions["A"][split]
            for name in list(predictions):
                predictions[name][split] = align_to(
                    predictions[name][split], reference["row_id"], f"{dataset}/{split}/{name}"
                )
        bootstraps[dataset] = {}
        drop_one[dataset] = {}
        for split in C.SPLITS:
            reference = predictions["A"][split]
            for name, payload in predictions.items():
                if not np.allclose(payload[split]["y_raw"], reference["y_raw"], rtol=0.0, atol=1e-6):
                    raise RuntimeError(f"{dataset}/{split}: {name} targets are not aligned with A")
            preds = {name: payload[split]["pred_raw"] for name, payload in predictions.items()}
            bootstraps[dataset][split] = paired_bootstrap(preds, reference["y_raw"])
            drop_one[dataset][split] = {
                f"{left}-{right}": drop_one_sensitivity(preds[left], preds[right], reference["y_raw"])
                for left, right in (("B", "A"), ("C", "A"), ("C", "B"))
            }

        replay[dataset] = {arm: verify_checkpoint_replay(bundle, dataset, arm, device=device) for arm in C.ARMS}
        for arm in C.ARMS:
            ledgers[f"{dataset}/{arm}"] = component_ledger(bundle, dataset, arm, device=device)
        interventions[dataset] = intervention_diagnostic(bundle, dataset, device=device)
        contexts[dataset] = context_response_diagnostic(bundle, dataset, device=device)
        diag4[dataset] = interaction_diagnostics(bundle, dataset, device=device)
        diag4[dataset]["labels"] = interaction_labels(diag4[dataset], contexts[dataset])
        equiv[dataset] = equivalence_scale(bundle, dataset, device=device)
        verdicts[dataset] = {
            "trace": verdict_trace(
                dataset,
                {arm: ledgers[f"{dataset}/{arm}"] for arm in ("B", "C")},
                interventions[dataset],
                equiv[dataset],
                replay[dataset],
                pretrain_ok[dataset],
            ),
            "retention": verdict_performance(dataset, traj),
            "context": verdict_context(dataset, traj, bootstraps[dataset], diag4[dataset]),
            "gine": verdict_gine(dataset, traj, gine) if gine is not None else None,
        }
        if with_figures:
            figure_curves(dataset, traj, gine, OUT / "figures" / f"curves_{dataset}.png")
            context_figure = figure_context_functions(bundle, dataset, OUT / "figures" / f"context_functions_{dataset}.png")
            rows = []
            for name, values in context_figure["grid"].items():
                for entry in values:
                    rows.append({"context_field": name, **entry})
            C.write_csv(OUT / "tables" / f"context_functions_{dataset}.csv", rows)
            case_figures[dataset] = figure_cases(bundle, dataset, ledgers[f"{dataset}/C"])

    C.write_csv(OUT / "tables" / "main_table.csv", main_rows)
    C.write_csv(OUT / "tables" / "contrasts.csv", contrast_rows)
    boot_rows = []
    for dataset, per_split in bootstraps.items():
        for split, payload in per_split.items():
            for contrast, metrics in payload["contrasts"].items():
                for metric, values in metrics.items():
                    boot_rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "contrast": contrast,
                            "metric": metric,
                            "point": values["point"],
                            "ci95_low": values["ci95_low"],
                            "ci95_high": values["ci95_high"],
                            "contains_zero": values["contains_zero"],
                            "n_replicates": values["n_replicates"],
                            "n_rows": payload["n"],
                        }
                    )
    C.write_csv(OUT / "tables" / "bootstrap.csv", boot_rows)
    drop_rows = []
    for dataset, per_split in drop_one.items():
        for split, contrasts in per_split.items():
            for contrast, values in contrasts.items():
                drop_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "contrast": contrast,
                        "delta_rmse_full": values["delta_rmse_full"],
                        "delta_rmse_min": values["delta_rmse_range"][0],
                        "delta_rmse_max": values["delta_rmse_range"][1],
                        "delta_mae_full": values["delta_mae_full"],
                        "delta_mae_min": values["delta_mae_range"][0],
                        "delta_mae_max": values["delta_mae_range"][1],
                        "direction_flips_if_one_row_dropped": values["direction_flips_if_one_row_dropped"],
                        "n_rows_that_flip_direction": values["n_rows_that_flip_direction"],
                        "max_single_row_shift_rmse": values["max_single_row_shift_rmse"],
                    }
                )
    C.write_csv(OUT / "tables" / "drop_one.csv", drop_rows)
    C.write_csv(
        OUT / "tables" / "verdicts.csv",
        [
            {"dataset": dataset, "axis": axis, "label": (value or {}).get("label")}
            for dataset, axes_ in verdicts.items()
            for axis, value in axes_.items()
            if value is not None
        ],
    )

    for key, payload in ledgers.items():
        dataset, arm = key.split("/")
        base = OUT / "ledgers" / f"{dataset}_{arm}"
        C.atomic_write_json(base.with_suffix(".json"), payload["meta"])
        write_csv_gz(f"{base}_molecules.csv.gz", _ledger_csv_rows(payload["molecules"]))
        write_csv_gz(f"{base}_atoms.csv.gz", payload["atoms"])
        # the per-molecule global vectors are numeric evidence: keep them as arrays
        active_names = payload["meta"]["active_global_names"]
        C.atomic_write_npz(
            f"{base}_globals.npz",
            {
                "official_row_id": np.asarray(
                    [m["official_row_id"] for m in payload["molecules"]], dtype=np.int64
                ),
                "split": np.asarray([m["split"] for m in payload["molecules"]]),
                **{
                    name: np.asarray([m["global_raw"][name] for m in payload["molecules"]], dtype=np.float64)
                    for name in active_names
                },
            },
        )
        C.atomic_write_npz(
            f"{base}_molecules.npz",
            {
                "official_row_id": np.asarray([m["official_row_id"] for m in payload["molecules"]], dtype=np.int64),
                "split": np.asarray([m["split"] for m in payload["molecules"]]),
                "y_raw": np.asarray([m["y_raw"] for m in payload["molecules"]], dtype=np.float64),
                "pred_raw": np.asarray([m["pred_raw"] for m in payload["molecules"]], dtype=np.float64),
                "intercept_raw": np.asarray([m["intercept_raw"] for m in payload["molecules"]], dtype=np.float64),
                "reconstructed_raw": np.asarray(
                    [m["full_reconstructed_raw"] for m in payload["molecules"]], dtype=np.float64
                ),
                "reconstruction_error": np.asarray(
                    [m["reconstruction_error"] for m in payload["molecules"]], dtype=np.float64
                ),
            },
        )
    C.atomic_write_json(OUT / "checkpoint_replay.json", replay)
    C.atomic_write_json(OUT / "interventions.json", interventions)
    C.atomic_write_json(OUT / "context_response.json", contexts)
    C.atomic_write_json(OUT / "interaction_diagnostics.json", diag4)
    C.atomic_write_json(OUT / "equivalence_scale.json", equiv)
    C.atomic_write_json(OUT / "reference_arm_A_vs_L.json", reference_l)
    C.atomic_write_json(OUT / "verdicts.json", verdicts)
    C.atomic_write_json(OUT / "bootstrap.json", bootstraps)
    C.atomic_write_json(OUT / "drop_one.json", drop_one)
    C.atomic_write_json(OUT / "cases.json", case_figures)

    summary = {
        "round_id": C.ROUND_ID,
        "datasets": list(datasets),
        "n_main_rows": len(main_rows),
        "verdicts": verdicts,
        "trace_all_pass": all(v["trace"]["label"] == "TRACE_PASS" for v in verdicts.values()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.atomic_write_json(OUT / "analysis_summary.json", summary)
    return summary
