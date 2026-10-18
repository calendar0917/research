"""Train-only information diagnostic for exact rooted-radius-r encodings.

This is *not* a model and *not* a confirmatory experiment.  It asks a narrow
question on the official TRAIN rows only:

    How much Lipophilicity label information does the exact multiset E_r carry,
    and does increasing r from 2 to 3 add information?

It uses a leave-one-out group-mean readout: molecules with the same E_r key are
one group; a held-out molecule is predicted by the mean label of the *other*
molecules in its group (global train mean for singletons).  A label-permutation
null with the same group sizes calibrates whether any apparent signal is just
the result of many small groups.

No validation/test label is loaded.  ``labels.npy`` is never opened.  The
official split JSON is read only for ``train_rows`` and ``train_y``; all other
label fields are ignored.

Run:

    uv run python tracks/structural-sufficiency/results/phase0/code/train_only_er_information_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE0 = HERE.parent
ANALYSIS = PHASE0 / "analysis"
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))

from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # noqa: E402

from lipo_feasibility_audit import rooted_encoding_from_mol  # noqa: E402

LIPO = REPO / "data/lipophilicity/Lipophilicity.csv"
SPLIT = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json"


def group_ids_for(encodings: Sequence[tuple[bytes, ...]]) -> np.ndarray:
    mapping: dict[tuple[bytes, ...], int] = {}
    ids = np.empty(len(encodings), dtype=np.int64)
    for i, key in enumerate(encodings):
        gid = mapping.get(key)
        if gid is None:
            gid = len(mapping)
            mapping[key] = gid
        ids[i] = gid
    return ids


def loo_group_mean_mae(y: np.ndarray, gids: np.ndarray) -> float:
    n_groups = int(gids.max()) + 1
    sums = np.bincount(gids, weights=y, minlength=n_groups)
    counts = np.bincount(gids, minlength=n_groups)
    denom = np.maximum(counts[gids] - 1, 1)
    pred = (sums[gids] - y) / denom
    singleton = counts[gids] == 1
    pred[singleton] = float(y.mean())
    return float(np.mean(np.abs(pred - y)))


def group_r2(y: np.ndarray, gids: np.ndarray) -> float:
    n_groups = int(gids.max()) + 1
    sums = np.bincount(gids, weights=y, minlength=n_groups)
    counts = np.bincount(gids, minlength=n_groups)
    means = sums / counts
    within = float(np.sum((y - means[gids]) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - within / total if total > 0 else 0.0


def group_label_conflict(y: np.ndarray, gids: np.ndarray) -> dict[str, Any]:
    """Label disagreement among molecules sharing the same structural key."""
    n_groups = int(gids.max()) + 1
    groups = [np.flatnonzero(gids == g) for g in range(n_groups)]
    pair_diffs: list[float] = []
    group_ranges: list[float] = []
    for idx in groups:
        if len(idx) < 2:
            continue
        vals = y[idx]
        group_ranges.append(float(vals.max() - vals.min()))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pair_diffs.append(abs(float(vals[i] - vals[j])))
    diffs = np.asarray(pair_diffs, dtype=np.float64)
    return {
        "n_nonsingleton_groups": len(group_ranges),
        "n_pairs": int(len(diffs)),
        "mean_abs_pair_label_diff": float(diffs.mean()) if len(diffs) else None,
        "median_abs_pair_label_diff": float(np.median(diffs)) if len(diffs) else None,
        "fraction_pairs_abs_diff_gt_0_5": float(np.mean(diffs > 0.5)) if len(diffs) else None,
        "fraction_pairs_abs_diff_gt_1_0": float(np.mean(diffs > 1.0)) if len(diffs) else None,
        "max_group_label_range": float(max(group_ranges)) if group_ranges else None,
        "n_groups_with_range_gt_1_0": int(sum(r > 1.0 for r in group_ranges)),
    }


def permutation_test(
    y: np.ndarray, gids: np.ndarray, observed_mae: float, observed_r2: float, n_perm: int, rng: np.random.Generator
) -> dict[str, Any]:
    null_mae = np.empty(n_perm)
    null_r2 = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)
        null_mae[b] = loo_group_mean_mae(yp, gids)
        null_r2[b] = group_r2(yp, gids)
    p_mae = (1 + int(np.sum(null_mae <= observed_mae))) / (n_perm + 1)
    p_r2 = (1 + int(np.sum(null_r2 >= observed_r2))) / (n_perm + 1)
    return {
        "n_perm": n_perm,
        "null_mae_mean": float(null_mae.mean()),
        "null_mae_sd": float(null_mae.std(ddof=1)),
        "null_mae_q05": float(np.quantile(null_mae, 0.05)),
        "null_r2_mean": float(null_r2.mean()),
        "null_r2_sd": float(null_r2.std(ddof=1)),
        "null_r2_q95": float(np.quantile(null_r2, 0.95)),
        "p_value_mae_lower": p_mae,
        "p_value_r2_higher": p_r2,
    }


def main() -> int:
    t0 = time.perf_counter()
    split = json.loads(SPLIT.read_text())
    train_rows = [int(x) for x in split["train_rows"]]
    y = np.asarray([float(v) for v in split["train_y"]], dtype=np.float64)
    assert len(train_rows) == len(y) == 3360
    df = pd.read_csv(LIPO, usecols=["smiles"])
    mols = [Chem.MolFromSmiles(df["smiles"].iloc[r]) for r in train_rows]
    assert all(m is not None for m in mols)

    global_mae = float(np.mean(np.abs(y - y.mean())))
    out: dict[str, Any] = {
        "protocol": "ssrc-phase0-train-only-er-information-v1",
        "scope": {
            "train_rows_only": True,
            "validation_labels_read": False,
            "test_labels_read": False,
            "label_vault_opened": False,
            "model_trained": False,
        },
        "n_train": len(y),
        "label": {"mean": float(y.mean()), "sd": float(y.std(ddof=1)), "min": float(y.min()), "max": float(y.max())},
        "global_mean_mae": global_mae,
        "by_radius": {},
        "baselines": {},
    }
    rng = np.random.default_rng(20261018)
    # Scaffold-group baseline: a chemistry-informed grouping with larger groups.
    scaffold_strings = [MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in mols]
    scaffold_ids = group_ids_for([(s.encode(),) for s in scaffold_strings])
    scaffold_counts = np.bincount(scaffold_ids)
    scaffold_loo = loo_group_mean_mae(y, scaffold_ids)
    scaffold_r2 = group_r2(y, scaffold_ids)
    out["baselines"]["murcko_scaffold"] = {
        "n_groups": int(len(scaffold_counts)),
        "n_singleton_groups": int(np.sum(scaffold_counts == 1)),
        "singleton_molecule_fraction": float(np.mean(scaffold_counts[scaffold_ids] == 1)),
        "max_group_size": int(scaffold_counts.max()),
        "mean_group_size": float(scaffold_counts.mean()),
        "loo_group_mean_mae": scaffold_loo,
        "loo_mae_improvement_vs_global": global_mae - scaffold_loo,
        "in_sample_group_r2": scaffold_r2,
        "label_conflict": group_label_conflict(y, scaffold_ids),
        "permutation": permutation_test(y, scaffold_ids, scaffold_loo, scaffold_r2, n_perm=500, rng=rng),
    }
    print(f"  scaffold: groups={len(scaffold_counts)} singleton_mol_frac={np.mean(scaffold_counts[scaffold_ids] == 1):.3f} "
          f"LOO_MAE={scaffold_loo:.5f} R2={scaffold_r2:.4f}")

    for r in (2, 3):
        t = time.perf_counter()
        encodings = [rooted_encoding_from_mol(m, r) for m in mols]
        gids = group_ids_for(encodings)
        counts = np.bincount(gids)
        loo_mae = loo_group_mean_mae(y, gids)
        r2 = group_r2(y, gids)
        perm = permutation_test(y, gids, loo_mae, r2, n_perm=500, rng=rng)
        out["by_radius"][f"r{r}"] = {
            "seconds": time.perf_counter() - t,
            "n_groups": int(len(counts)),
            "n_singleton_groups": int(np.sum(counts == 1)),
            "n_molecules_in_nonsingleton_groups": int(np.sum(counts[gids] > 1)),
            "singleton_molecule_fraction": float(np.mean(counts[gids] == 1)),
            "max_group_size": int(counts.max()),
            "mean_group_size": float(counts.mean()),
            "group_size_histogram": {str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())},
            "loo_group_mean_mae": loo_mae,
            "global_mean_mae": global_mae,
            "loo_mae_improvement_vs_global": global_mae - loo_mae,
            "in_sample_group_r2": r2,
            "label_conflict": group_label_conflict(y, gids),
            "permutation": perm,
        }
        print(f"  r={r}: groups={len(counts)} singleton_mol_frac={np.mean(counts[gids] == 1):.3f} "
              f"LOO_MAE={loo_mae:.5f} (global {global_mae:.5f}) R2={r2:.4f} "
              f"p(MAE)={perm['p_value_mae_lower']:.4f} p(R2)={perm['p_value_r2_higher']:.4f}")

    out["runtime_seconds"] = time.perf_counter() - t0
    path = ANALYSIS / "train_only_er_information_audit.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"SSRC Phase 0 train-only E_r information audit")
    print(f"  output: {path}")
    print(f"  runtime: {out['runtime_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
