"""Train-sub support counts for the Support Gate (Arm E).

Definition (frozen in PROTOCOL.md): for a directed event, ``n`` is the number
of **distinct Train-sub molecules** that contain the same exact
``(S identity, C identity, cut-bond attributes)`` combination.

* Train-sub events use the leave-one-molecule-out count ``count - 1`` (the
  molecule itself is always one of the contributors, even if the same
  combination appears several times inside it).
* Dev events use the full Train-sub count.
* No Dev or official Valid / Test structure contributes to any count.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def rho_from_support(n: np.ndarray, floor: float = 0.5, span: float = 0.5,
                     offset: float = 5.0) -> np.ndarray:
    x = np.asarray(n, np.float64)
    return float(floor) + float(span) * x / (x + float(offset))


def positions_from_rows(pack: Mapping[str, Any], rows) -> np.ndarray:
    order = np.asarray(pack["row_ids"], np.int64)
    lookup = {int(r): i for i, r in enumerate(order.tolist())}
    missing = [int(r) for r in np.asarray(rows, np.int64).tolist() if int(r) not in lookup]
    if missing:
        raise KeyError(f"rows not present in the pack: {missing[:5]}")
    return np.asarray([lookup[int(r)] for r in np.asarray(rows, np.int64)], np.int64)


def build(pack: Mapping[str, Any], train_sub_rows, dev_rows,
          floor: float = 0.5, span: float = 0.5, offset: float = 5.0) -> dict[str, Any]:
    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    ev_combo = np.asarray(pack["ev_combo"], np.int64)
    n_combo = int(np.asarray(pack["combo_s"]).size)
    train_pos = positions_from_rows(pack, train_sub_rows)
    dev_pos = positions_from_rows(pack, dev_rows)
    if set(train_pos.tolist()) & set(dev_pos.tolist()):
        raise AssertionError("Train-sub / Dev pack positions overlap")

    counts = np.zeros(n_combo, np.int64)
    for pos in train_pos.tolist():
        ids = np.unique(ev_combo[ev_ptr[pos]:ev_ptr[pos + 1]])
        counts[ids] += 1
    full = counts[ev_combo]
    loo = full - 1
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    train_mask = np.isin(ev_mol, train_pos)
    dev_mask = np.isin(ev_mol, dev_pos)
    if np.any(loo[train_mask] < 0):
        raise AssertionError("negative leave-one-molecule-out support")

    support = full.copy()
    support[train_mask] = loo[train_mask]
    support[dev_mask] = full[dev_mask]
    rho = rho_from_support(support, floor=floor, span=span, offset=offset)

    train_combo_values = counts[np.unique(ev_combo[train_mask])]
    return {
        "support_by_event": support,
        "rho_by_event": rho,
        "combo_molecule_counts": counts,
        "train_sub_positions": train_pos,
        "dev_positions": dev_pos,
        "definition": {
            "n": "distinct Train-sub molecules containing the exact (S, C, cut-bond) combo",
            "train_sub": "leave-one-molecule-out: combo molecule count - 1",
            "dev": "full Train-sub combo molecule count",
            "rho": f"{floor} + {span} * n / (n + {offset})",
        },
        "summary": {
            "n_combo_identities": n_combo,
            "train_sub_events": int(train_mask.sum()),
            "dev_events": int(dev_mask.sum()),
            "train_sub_support_min": int(support[train_mask].min()) if train_mask.any() else 0,
            "train_sub_support_max": int(support[train_mask].max()) if train_mask.any() else 0,
            "dev_support_min": int(support[dev_mask].min()) if dev_mask.any() else 0,
            "dev_support_max": int(support[dev_mask].max()) if dev_mask.any() else 0,
            "dev_events_unseen_combo": int(np.sum(support[dev_mask] == 0)) if dev_mask.any() else 0,
            "train_sub_events_loo_zero": int(np.sum(support[train_mask] == 0)) if train_mask.any() else 0,
            "train_combo_count_nonzero": int(np.count_nonzero(counts)),
            "train_combo_count_max": int(counts.max(initial=0)),
            "train_combo_count_mean_nonzero": float(train_combo_values.mean())
            if train_combo_values.size else 0.0,
        },
    }