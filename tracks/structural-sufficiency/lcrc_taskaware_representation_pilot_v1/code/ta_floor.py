"""How much error can a *proven* representation collision actually explain?

For a pair of molecules with identical model input but different Train labels the
best possible constant prediction is the midpoint, so no model working on the
frozen input can do better than ``|delta| / 2`` on each member of the pair.  If
such pairs are the only thing the representation destroys, the Train RMSE floor
that the collision alone imposes is::

    floor_collision = sqrt( sum_pairs 2*(delta/2)^2 / n_train )
                    = sqrt( sum_pairs delta^2/2 / n_train )

This bound is a *floor from collisions only*: it is small when the collision is
rare, and it is an upper bound on the collision's contribution to the error, not
a decomposition of the observed error.  Reported for the two regression datasets
whose collisions are numerous enough to matter (Lipophilicity, FreeSolv) and for
the binary ones as a label-disagreement rate.

Usage
-----
    uv run python code/ta_floor.py
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

import mb2_source as S
import ta_collision as COL

OUT = C.ANALYSIS / "collision_label_floor.json"


def _pairs(dataset: str) -> list[tuple[int, int, bool]]:
    return COL._all_collision_pairs(dataset)


def floor_for(dataset: str) -> dict[str, Any]:
    task = C.dataset_entry(dataset)["task"]
    source = S.load_dataset(dataset, labels="train_valid")
    y_train = np.asarray(source["y"]["train"], dtype=np.float64)
    train_rows = [int(r) for r in source["rows"]["train"]]
    position = {row: i for i, row in enumerate(train_rows)}
    n_train = len(train_rows)

    per_pair = []
    for a, b, same_input in _pairs(dataset):
        if a not in position or b not in position:
            continue
        ya = y_train[position[a]].reshape(-1)
        yb = y_train[position[b]].reshape(-1)
        if task == "multitask":
            mask = np.isfinite(ya) & np.isfinite(yb)
            delta = float(np.abs(ya[mask] - yb[mask]).mean()) if mask.any() else 0.0
        else:
            delta = float(abs(ya[0] - yb[0]))
        per_pair.append(
            {
                "train_rows": [int(a), int(b)],
                "identical_model_input": bool(same_input),
                "label_delta": delta,
            }
        )

    identical = [p for p in per_pair if p["identical_model_input"]]
    deltas = np.asarray([p["label_delta"] for p in identical], dtype=np.float64)
    payload: dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "n_train": n_train,
        "n_collision_pairs_in_train": len(per_pair),
        "n_identical_input_pairs_in_train": len(identical),
        "n_pairs_with_nonzero_label_delta": int((deltas > 0).sum()) if deltas.size else 0,
        "label_scope": "published Train split only; no Test label decoded",
        "examples": identical[:8],
    }
    if deltas.size:
        payload["delta_stats"] = {
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "p90": float(np.percentile(deltas, 90)),
            "max": float(deltas.max()),
            "sum": float(deltas.sum()),
        }
        if task == "regression":
            floor = float(np.sqrt((deltas**2).sum() / 2.0 / n_train))
            payload["train_rmse_floor_from_collisions"] = floor
            payload["train_rmse_floor_formula"] = "sqrt(sum(delta^2)/2/n_train)"
            payload["fraction_of_train_rows_in_a_collision"] = float(2 * len(identical) / n_train)
        else:
            payload["label_disagreement_rate"] = float((deltas > 0).mean())
            payload["n_conflicting_label_pairs"] = int((deltas > 0).sum())
            payload["fraction_of_train_rows_in_a_collision"] = float(2 * len(identical) / n_train)
    else:
        payload["train_rmse_floor_from_collisions"] = 0.0
        payload["note"] = "no Train pair collides; the representation loses nothing on this dataset"
    return payload


def run(datasets: Sequence[str] | None = None) -> dict[str, Any]:
    keys = list(datasets) if datasets else [k for k in C.DATASETS]
    payload = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "scope": (
            "Upper bound on the Train error attributable to *proven* input collisions. "
            "Small floors mean the collision cannot be the main source of the observed error; "
            "they never prove that a representation gap is absent."
        ),
        "code_hashes": C.code_hashes(),
        "datasets": {k: floor_for(k) for k in keys},
    }
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    run(None if not argv else list(argv))
    data = C.read_json(OUT)["datasets"]
    for key, entry in data.items():
        floor = entry.get("train_rmse_floor_from_collisions")
        extra = f" floor {floor:.4f}" if floor is not None else f" disagreements {entry.get('label_disagreement_rate', 0):.3f}"
        print(
            f"{key:14s} pairs_in_train {entry['n_collision_pairs_in_train']:4d} "
            f"identical {entry['n_identical_input_pairs_in_train']:4d}{extra}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
