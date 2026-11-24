"""Fixed-hash molecule sample, its exact states and its root-record multiset.

Selection (label-free, prediction-free):

* only rectangles whose four cells are *clean* (one molecule identity per cell,
  no explicitly encoded stereo, no duplicate identity row) are candidates --
  a multi-molecule cell is never treated as one molecule;
* canonical replacement types are visited in fixed hash order, and inside each
  type the rectangles are visited in fixed hash order, round-robin, so the
  sample spreads over many replacement types;
* a rectangle is taken only if it keeps the unique-molecule count at or below
  ``MAX_STATE_MOLECULES``.

For those molecules the frozen W3 forward is run once (batch 512).  ``h``,
the four head scores and the root-record digests are stored per atom; the whole
graph state ``U`` is then assembled from ``h`` and the scores.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

import ssa_common as C
import ssa_records as REC


def select_state_sample(obs: Mapping[str, Any], sel: Mapping[str, Any],
                        *, max_molecules: int = C.MAX_STATE_MOLECULES) -> dict[str, Any]:
    rect = sel["rect"]
    pair_types = np.asarray(sel["pair_types"], np.int64)
    clean = np.asarray(rect["clean"], bool)
    cand = np.flatnonzero(clean)
    # fixed hash order of pair types and of rectangles inside a pair type
    pt_hash = np.asarray([C.hash64("pair", int(k)) for k in pair_types.tolist()], np.uint64)
    pt_order = np.argsort(pt_hash, kind="stable")
    pair_index = np.searchsorted(pair_types, rect["pair_key"])
    by_pair: dict[int, np.ndarray] = {}
    for pos, k in enumerate(pair_types.tolist()):
        m = cand[pair_index[cand] == pos]
        if m.size == 0:
            continue
        by_pair[pos] = m[np.argsort(rect["hash"][m], kind="stable")]

    mol_of = {  # cell position -> molecule identity
        "m1": obs["cell_mol"][rect["pos1_i"]],
        "m2": obs["cell_mol"][rect["pos1_j"]],
        "m3": obs["cell_mol"][rect["pos2_i"]],
        "m4": obs["cell_mol"][rect["pos2_j"]],
    }
    used: set[int] = set()
    chosen: list[int] = []
    rounds = 0
    max_len = max((len(v) for v in by_pair.values()), default=0)
    while rounds < max_len and len(used) < max_molecules:
        added = 0
        for pos in pt_order.tolist():
            arr = by_pair.get(pos)
            if arr is None or rounds >= len(arr):
                continue
            i = int(arr[rounds])
            mols = {int(mol_of[c][i]) for c in ("m1", "m2", "m3", "m4")}
            new = mols - used
            if len(used) + len(new) > max_molecules:
                continue
            used |= new
            chosen.append(i)
            added += 1
        if added == 0:
            break
        rounds += 1

    chosen_idx = np.asarray(sorted(chosen), np.int64)
    mol_ids = np.asarray(sorted(used), np.int64)
    rows = obs["mol_cache_row"][mol_ids].astype(np.int64)
    covered_pairs = np.unique(rect["pair_key"][chosen_idx])
    return {
        "rect_index": chosen_idx,
        "mol_ids": mol_ids,
        "rows": rows,
        "rounds": int(rounds),
        "n_clean_candidates": int(cand.size),
        "n_pair_types_clean": int(len(by_pair)),
        "n_covered_pair_types": int(covered_pairs.size),
        "molecule_budget": int(max_molecules),
        "molecule_cols": mol_of,
        "selection": (
            "clean-cell rectangles only; canonical replacement types in fixed "
            "hash order; inside a type rectangles in fixed hash order; "
            "round-robin; stop at the unique-molecule budget. No label, W3 "
            "prediction or error is read."
        ),
    }


def compute_states(w3: Mapping[str, Any], obs: Mapping[str, Any],
                   sample: Mapping[str, Any], machine: Any,
                   *, batch_size: int = 512) -> dict[str, Any]:
    """One frozen forward over the sample + per-atom root records."""
    import ew_collate as EC

    cache = w3["cache"]
    # ``sample["rows"]`` are Valid-cache row indices (not official gids)
    cache_rows = np.asarray(sample["rows"], np.int64)
    rows = cache_rows
    n_mol = int(rows.size)
    if n_mol == 0 or cache_rows.min() < 0 or cache_rows.max() >= int(cache["n_molecules"]):
        raise RuntimeError("sample molecule missing from the Valid cache")

    h_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    lo_parts: list[np.ndarray] = []
    rbytes_parts: list[np.ndarray] = []
    aux_parts: dict[str, list[np.ndarray]] = {"mean_h": [], "mean_h2": [],
                                              "attention_pooled": [], "rho": [],
                                              "residual_logits": [], "contrib": []}
    atom_off = [0]
    t0 = time.perf_counter()
    n_batches = 0
    for lo in range(0, n_mol, int(batch_size)):
        batch_rows = rows[lo:lo + int(batch_size)]
        batch = EC.collate_v2(cache, batch_rows.tolist())
        import ssa_state as S

        fw = S.forward_atoms(w3["model"], batch)
        rec = REC.extract_batch_records(batch)
        h_parts.append(fw["h"])
        s_parts.append(fw["scores"])
        z_parts.append(fw["logits"].astype(np.float64))
        hi_parts.append(rec["hi"])
        lo_parts.append(rec["lo"])
        rbytes_parts.append(rec["record_bytes"])
        for key in aux_parts:
            aux_parts[key].append(np.asarray(fw["aux"][key]))
        base = int(atom_off[-1])
        atom_off.extend((fw["atom_start"] + base).tolist()[1:])
        atom_off.append(base + int(fw["h"].shape[0]))
        n_batches += 1
        if n_batches % 8 == 0:
            print(f"    states batch {n_batches} ({lo + len(batch_rows)}/{n_mol}) "
                  f"{time.perf_counter() - t0:.1f}s rss={C.peak_rss_gb():.2f}GB", flush=True)

    h = np.concatenate(h_parts, axis=0)
    scores = np.concatenate(s_parts, axis=0)
    z = np.concatenate(z_parts, axis=0)
    hi = np.concatenate(hi_parts)
    lo = np.concatenate(lo_parts)
    rbytes = np.concatenate(rbytes_parts)
    aux = {k: np.concatenate(v, axis=0) for k, v in aux_parts.items()}
    atom_start = np.asarray(atom_off[:-1], np.int64)
    if atom_start.size != n_mol:
        raise RuntimeError(f"atom offsets {atom_start.size} != molecules {n_mol}")
    g = np.asarray(cache["global"])[cache_rows].astype(np.float32)

    return {
        "rows": rows,
        "cache_rows": cache_rows,
        "n_mol": n_mol,
        "h": h,
        "scores": scores,
        "z_model": z,
        "pred_model_eV": z * machine.label_std_eV + machine.label_mean_eV,
        "atom_start": atom_start,
        "atom_stop": np.r_[atom_start[1:], h.shape[0]].astype(np.int64),
        "atom_off": np.r_[atom_start, h.shape[0]].astype(np.int64),
        "rec_hi": hi,
        "rec_lo": lo,
        "record_bytes": rbytes,
        "aux": aux,
        "g": g,
        "forward_seconds": float(time.perf_counter() - t0),
        "n_atoms": int(h.shape[0]),
        "mean_record_bytes": float(rbytes.mean()),
        "max_record_bytes": int(rbytes.max()),
    }


__all__ = ["compute_states", "select_state_sample"]


def build_U(machine: Any, states: Mapping[str, Any], *, block_molecules: int = 256) -> np.ndarray:
    """Whole-graph states ``U(G)`` for the sample, from the stored ``h``/scores."""
    n_mol = int(states["n_mol"])
    U = np.zeros((n_mol, machine.phi_dim), np.float64)
    atom_start = np.asarray(states["atom_start"], np.int64)
    h = states["h"]
    scores = states["scores"]
    g0 = 0
    while g0 < n_mol:
        g1 = min(n_mol, g0 + int(block_molecules))
        a0 = int(atom_start[g0])
        a1 = int(atom_start[g1]) if g1 < n_mol else int(h.shape[0])
        U[g0:g1] = machine.graph_state(
            h[a0:a1], scores[a0:a1], atom_start[g0:g1] - a0
        )
        g0 = g1
    return U
