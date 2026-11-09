"""P3 — does Train-only structural support explain the model gains?

For each pre-registered pair (new vs W2), split and WL2 novelty group, the
per-task AP is recomputed on the molecules of that group only (both classes
required), and ``delta_AP = AP(new) - AP(W2)`` is reported with its support.

Also reported: the normalized rank percentile change of true positives as a
function of novelty, and a descriptive heavy-atom-size control (Validation
tertiles, fixed before looking at any result).

Outputs: ``analysis/novelty_vs_gain_by_task.csv``,
``analysis/novelty_vs_gain_summary.json``, ``analysis/novelty_rank_change_by_task.csv``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import lc_common as C
import lc_data as D
import lc_metrics as M

BY_TASK_PATH = C.ANALYSIS / "novelty_vs_gain_by_task.csv"
RANK_BY_TASK_PATH = C.ANALYSIS / "novelty_rank_change_by_task.csv"
SUMMARY_PATH = C.ANALYSIS / "novelty_vs_gain_summary.json"

SIZE_TERTILE_LABELS = ("size_low", "size_mid", "size_high")


def _load_groups(split: str) -> pd.DataFrame:
    path = C.ANALYSIS / f"structure_support_{split}.csv"
    df = pd.read_csv(path)
    rows = D.split_rows(split)
    if df.shape[0] != rows.size:
        raise RuntimeError(f"{split} structure support rows {df.shape[0]} != {rows.size}")
    if not np.array_equal(df["ogb_row"].to_numpy().astype(np.int64), rows):
        raise RuntimeError(f"{split} structure support rows are not aligned to the official split")
    if not np.array_equal(df["split_index"].to_numpy().astype(np.int64), np.arange(rows.size)):
        raise RuntimeError(f"{split} structure support row order changed")
    return df


def _size_tertiles(valid_df: pd.DataFrame) -> tuple[float, float]:
    sizes = valid_df["n_heavy_atoms"].to_numpy(dtype=np.float64)
    return float(np.nanpercentile(sizes, 100.0 / 3.0)), float(np.nanpercentile(sizes, 200.0 / 3.0))


def _size_bucket(sizes: np.ndarray, q1: float, q2: float) -> np.ndarray:
    out = np.full(sizes.shape, "", dtype=object)
    out[sizes <= q1] = "size_low"
    out[(sizes > q1) & (sizes <= q2)] = "size_mid"
    out[sizes > q2] = "size_high"
    return out


def _group_cell_ap(
    y_col: np.ndarray, new_col: np.ndarray, ref_col: np.ndarray, mask: np.ndarray
) -> dict[str, Any] | None:
    yy = np.asarray(y_col, dtype=np.float64)[mask]
    obs = np.isfinite(yy)
    yt = yy[obs]
    n_pos = int((yt == 1.0).sum())
    n_neg = int((yt == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    from sklearn.metrics import average_precision_score

    probs_new = D.sigmoid(np.asarray(new_col, dtype=np.float32))[mask][obs].astype(np.float64)
    probs_ref = D.sigmoid(np.asarray(ref_col, dtype=np.float32))[mask][obs].astype(np.float64)
    ap_new = float(average_precision_score(yt, probs_new))
    ap_ref = float(average_precision_score(yt, probs_ref))
    return {
        "n_molecules": int(mask.sum()),
        "n_observed": int(obs.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "support_stratum": C.support_stratum(n_pos, n_neg),
        "ap_new": ap_new,
        "ap_reference": ap_ref,
        "delta_ap": ap_new - ap_ref,
    }


def _aggregate_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in ("ge20", "5_19", "1_4"):
        sub = [r for r in rows if r["support_stratum"] == stratum]
        deltas = [float(r["delta_ap"]) for r in sub]
        out[stratum] = {
            "n_tasks": len(sub),
            "mean_delta_ap": float(np.mean(deltas)) if deltas else None,
            "median_delta_ap": float(np.median(deltas)) if deltas else None,
            "sum_delta_ap": float(np.sum(deltas)) if deltas else None,
            "n_improved": int(sum(1 for d in deltas if d > 0)),
            "n_degraded": int(sum(1 for d in deltas if d < 0)),
            "total_positive": int(sum(r["n_positive"] for r in sub)),
            "total_negative": int(sum(r["n_negative"] for r in sub)),
        }
    main = [float(r["delta_ap"]) for r in rows if r["support_stratum"] == "ge20"]
    out["main_eligible"] = len(main) >= C.HIGH_SUPPORT_POS
    return out


def _trend_label(per_group: dict[str, dict[str, Any]], split: str, pair_key: str) -> dict[str, Any]:
    """Pre-registered novelty-trend label for one split (PROTOCOL section 7)."""
    seen = per_group.get("SEEN", {}).get("ge20", {})
    partial = per_group.get("PARTIAL_NOVEL", {}).get("ge20", {})
    high = per_group.get("HIGH_NOVEL", {}).get("ge20", {})
    cells = {"SEEN": seen.get("n_tasks", 0), "PARTIAL_NOVEL": partial.get("n_tasks", 0), "HIGH_NOVEL": high.get("n_tasks", 0)}
    if min(cells.values()) < C.HIGH_SUPPORT_POS:
        return {
            "label": "NOVELTY_EFFECT_UNRESOLVED",
            "reason": "fewer than 20 main (>=20 positive) cells in at least one novelty group",
            "main_cells": cells,
        }
    d_seen = float(seen["mean_delta_ap"])
    d_partial = float(partial["mean_delta_ap"])
    d_high = float(high["mean_delta_ap"])
    monotone = d_seen >= d_partial >= d_high
    margin = d_seen - d_high
    if d_seen > 0 and d_high < 0:
        label = "GAIN_REVERSED_ON_NOVEL"
    elif margin > C.NOVELTY_GAIN_MARGIN and monotone:
        label = "GAIN_CONCENTRATED_ON_SEEN"
    elif d_seen > 0 and d_high >= 0 and margin > 0:
        label = "GAIN_LARGER_ON_SEEN_NOT_REVERSED"
    else:
        label = "NO_CLEAR_NOVELTY_TREND"
    return {
        "label": label,
        "main_cells": cells,
        "mean_delta_seen": d_seen,
        "mean_delta_partial": d_partial,
        "mean_delta_high": d_high,
        "seen_minus_high": margin,
        "monotone_decrease": bool(monotone),
        "margin_threshold": C.NOVELTY_GAIN_MARGIN,
    }


def _shared_task_trend(cells: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Novelty trend on tasks that are main-eligible (>=20 positive) in *both* splits.

    Comparing the two splits on all eligible cells mixes different task subsets
    (37 vs 50 tasks in HIGH_NOVEL), so the direction-consistency gate also uses
    this task-matched version.  Both versions are written to the summary.
    """
    per_group: dict[str, dict[str, Any]] = {}
    for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
        v = {r["task"]: r["delta_ap"] for r in cells["valid"][g] if r["support_stratum"] == "ge20"}
        t = {r["task"]: r["delta_ap"] for r in cells["test"][g] if r["support_stratum"] == "ge20"}
        shared = sorted(set(v) & set(t))
        if not shared:
            per_group[g] = {"n_shared_tasks": 0}
            continue
        dv = np.array([v[x] for x in shared], dtype=np.float64)
        dt = np.array([t[x] for x in shared], dtype=np.float64)
        per_group[g] = {
            "n_shared_tasks": len(shared),
            "mean_delta_valid": float(dv.mean()),
            "mean_delta_test": float(dt.mean()),
            "median_delta_valid": float(np.median(dv)),
            "median_delta_test": float(np.median(dt)),
            "n_improved_valid": int((dv > 0).sum()),
            "n_improved_test": int((dt > 0).sum()),
            "decay_test_minus_valid": float(dt.mean() - dv.mean()),
        }
    cells_n = {g: per_group[g].get("n_shared_tasks", 0) for g in per_group}
    if min(cells_n.values()) < C.HIGH_SUPPORT_POS:
        return {
            "label": "NOVELTY_EFFECT_UNRESOLVED",
            "reason": "fewer than 20 shared main cells in at least one novelty group",
            "per_group": per_group,
        }
    d_seen = per_group["SEEN"]["mean_delta_test"]
    d_partial = per_group["PARTIAL_NOVEL"]["mean_delta_test"]
    d_high = per_group["HIGH_NOVEL"]["mean_delta_test"]
    # Test-side novelty trend, gated by the same direction on Validation.
    monotone_test = d_seen >= d_partial >= d_high
    margin = d_seen - d_high
    v_seen = per_group["SEEN"]["mean_delta_valid"]
    v_high = per_group["HIGH_NOVEL"]["mean_delta_valid"]
    same_direction = (v_seen - v_high) * margin > 0
    if d_seen > 0 and d_high < 0 and same_direction:
        label = "GAIN_REVERSED_ON_NOVEL"
    elif margin > C.NOVELTY_GAIN_MARGIN and monotone_test and same_direction:
        label = "GAIN_CONCENTRATED_ON_SEEN"
    elif not same_direction:
        label = "NOVELTY_EFFECT_UNRESOLVED"
    else:
        label = "NO_CLEAR_NOVELTY_TREND"
    return {
        "label": label,
        "reason": "task-matched (>=20 positive in both splits) novelty trend on Test, gated by Validation direction",
        "per_group": per_group,
        "seen_minus_high_test": margin,
        "monotone_decrease_test": bool(monotone_test),
        "same_direction_across_splits": bool(same_direction),
    }


def run() -> dict[str, Any]:
    task_sets = C.read_json(C.ANALYSIS / "task_sets.json")
    high_tasks = set(int(t) for t in task_sets["HIGH_SUPPORT"])
    labels = {s: D.split_labels(s) for s in ("valid", "test")}
    rows = {s: D.split_rows(s) for s in ("valid", "test")}
    groups_df = {s: _load_groups(s) for s in ("valid", "test")}
    logits = {(label, s): D.load_logits(label, s) for label in C.MODEL_ORDER for s in ("valid", "test")}

    q1, q2 = _size_tertiles(groups_df["valid"])
    sizes = {s: groups_df[s]["n_heavy_atoms"].to_numpy(dtype=np.float64) for s in ("valid", "test")}
    novelty = {s: groups_df[s]["novelty_group"].to_numpy() for s in ("valid", "test")}
    size_bucket = {s: _size_bucket(sizes[s], q1, q2) for s in ("valid", "test")}
    frac_unseen = {s: groups_df[s]["fraction_unseen"].to_numpy(dtype=np.float64) for s in ("valid", "test")}

    gain_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": "lcrc-pcba-bottleneck-diagnosis-novelty-gain-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "definitions": {
            "delta_ap": "AP(new model) - AP(W2) recomputed inside the same split x task x novelty group cell (both classes required)",
            "reliability": {"ge20": "main diagnostic", "5_19": "descriptive only", "1_4": "record only", "na": "unscorable"},
            "rank_change": "within-task normalized percentile of a true positive in the new model minus W2 (positive = new model ranks it higher)",
            "novelty_thresholds": {
                "SEEN": "fraction_unseen == 0",
                "PARTIAL_NOVEL": f"0 < fraction_unseen <= {C.NOVELTY_PARTIAL_MAX}",
                "HIGH_NOVEL": f"fraction_unseen > {C.NOVELTY_PARTIAL_MAX}",
            },
            "size_control": {
                "tertiles_from": "Validation n_heavy_atoms only",
                "q1_heavy_atoms": q1,
                "q2_heavy_atoms": q2,
                "labels": list(SIZE_TERTILE_LABELS),
            },
        },
        "pairs": {},
        "size_confound": {},
        "caveats": [
            "AP depends on the positive rate: only the same-cell model difference is interpreted, never absolute AP across groups.",
            "WL2 novelty is a structural-support proxy, not proof about what the network learned.",
            "Test is an exposed comparison set; all numbers are descriptive.",
        ],
    }

    for pair in C.PRIMARY_PAIRS:
        a, b = pair
        pkey = C.pair_key(a, b)
        pair_out: dict[str, Any] = {"new_model": a, "reference_model": b, "splits": {}}
        pair_cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for split in ("valid", "test"):
            y = labels[split]
            grp = novelty[split]
            la = logits[(a, split)]
            lb = logits[(b, split)]
            split_cells: dict[str, list[dict[str, Any]]] = {g: [] for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL")}
            for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
                mask_g = grp == g
                for t in range(C.N_TASKS):
                    cell = _group_cell_ap(y[:, t], la[:, t], lb[:, t], mask_g)
                    if cell is None:
                        continue
                    row = {
                        "pair": pkey,
                        "new_model": a,
                        "reference_model": b,
                        "split": split,
                        "task": t,
                        "novelty_group": g,
                        "high_support_task": t in high_tasks,
                        **cell,
                    }
                    gain_rows.append(row)
                    split_cells[g].append(row)

            per_group = {g: _aggregate_cells(split_cells[g]) for g in split_cells}
            trends = {g: _trend_label(per_group, split, pkey) for g in ("SEEN",)}
            # direction consistency: per-task delta sign in SEEN vs HIGH_NOVEL
            seen_by_task = {r["task"]: r["delta_ap"] for r in split_cells["SEEN"] if r["support_stratum"] == "ge20"}
            high_by_task = {r["task"]: r["delta_ap"] for r in split_cells["HIGH_NOVEL"] if r["support_stratum"] == "ge20"}
            shared = sorted(set(seen_by_task) & set(high_by_task))
            sign_pairs = {
                "n_shared_tasks": len(shared),
                "both_positive": int(sum(1 for t in shared if seen_by_task[t] > 0 and high_by_task[t] > 0)),
                "both_negative": int(sum(1 for t in shared if seen_by_task[t] < 0 and high_by_task[t] < 0)),
                "seen_pos_high_neg": int(sum(1 for t in shared if seen_by_task[t] > 0 and high_by_task[t] < 0)),
                "seen_neg_high_pos": int(sum(1 for t in shared if seen_by_task[t] < 0 and high_by_task[t] > 0)),
            }
            pair_cells[split] = split_cells
            pair_out["splits"][split] = {
                "groups": per_group,
                "trend_all_cells": trends["SEEN"],
                "seen_vs_high_sign": sign_pairs,
            }

            # Rank change of true positives vs novelty (high-support tasks only).
            for t in range(C.N_TASKS):
                if t not in high_tasks:
                    continue
                yy = np.asarray(y[:, t], dtype=np.float64)
                obs = np.isfinite(yy)
                pos = obs & (yy == 1.0)
                if pos.sum() == 0:
                    continue
                pct_a = M.normalized_percentile(la[obs, t])
                pct_b = M.normalized_percentile(lb[obs, t])
                idx = np.flatnonzero(obs)
                pos_pos = np.flatnonzero(pos)
                pos_map = {int(p): k for k, p in enumerate(idx)}
                pos_within = np.array([pos_map[int(p)] for p in pos_pos], dtype=np.int64)
                delta_pct = (pct_a[pos_within] - pct_b[pos_within])  # percentile units: new - ref
                grp_pos = grp[pos]
                for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
                    sel = grp_pos == g
                    if not np.any(sel):
                        continue
                    rank_rows.append(
                        {
                            "pair": pkey,
                            "split": split,
                            "task": t,
                            "novelty_group": g,
                            "n_positive": int(sel.sum()),
                            "median_delta_pct": float(np.median(delta_pct[sel])),
                            "mean_delta_pct": float(np.mean(delta_pct[sel])),
                            "frac_improved": float(np.mean(delta_pct[sel] > 0)),
                            "median_frac_unseen_of_positive": float(np.nanmedian(frac_unseen[split][pos][sel])),
                        }
                    )

        # Pre-registered pair-level verdict: task-matched direction consistency.
        shared = _shared_task_trend(pair_cells)
        pair_out["trend_shared_tasks"] = shared
        all_cells_labels = [
            pair_out["splits"]["valid"]["trend_all_cells"]["label"],
            pair_out["splits"]["test"]["trend_all_cells"]["label"],
        ]
        pair_out["agree_all_cells_vs_shared"] = bool(shared["label"] == "NOVELTY_EFFECT_UNRESOLVED" and
                                                     "NOVELTY_EFFECT_UNRESOLVED" in all_cells_labels) or (
            shared["label"] != "NOVELTY_EFFECT_UNRESOLVED"
            and pair_out["splits"]["valid"]["trend_all_cells"]["label"]
            == pair_out["splits"]["test"]["trend_all_cells"]["label"]
        )
        pair_out["verdict"] = {
            "label": shared["label"],
            "reason": shared.get("reason", ""),
            "all_cells_labels": all_cells_labels,
        }
        summary["pairs"][pkey] = pair_out

    # ---- size confound (descriptive) ----------------------------------------
    for split in ("valid", "test"):
        conf: dict[str, Any] = {"n_heavy_atoms_by_group": {}}
        for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
            vals = sizes[split][novelty[split] == g]
            conf["n_heavy_atoms_by_group"][g] = C.quantiles(vals)
        conf["median_heavy_atoms_by_group_and_size_bucket"] = {
            g: {
                b: C.quantiles(sizes[split][(novelty[split] == g) & (size_bucket[split] == b)])["median"]
                for b in SIZE_TERTILE_LABELS
            }
            for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL")
        }
        conf["size_bucket_counts_by_group"] = {
            g: {b: int(np.sum((novelty[split] == g) & (size_bucket[split] == b))) for b in SIZE_TERTILE_LABELS}
            for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL")
        }
        summary["size_confound"][split] = conf

    # Descriptive stratified check: rank change of positives by novelty x size.
    rank_df = pd.DataFrame(rank_rows)
    size_strat: dict[str, Any] = {}
    if not rank_df.empty:
        for pair in C.PRIMARY_PAIRS:
            pkey = C.pair_key(*pair)
            size_strat[pkey] = {}
            for split in ("valid", "test"):
                sub = rank_df[(rank_df.pair == pkey) & (rank_df.split == split)]
                cells = {}
                for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
                    for b in SIZE_TERTILE_LABELS:
                        mask = np.zeros(rows[split].size, dtype=bool)
                        mask[(novelty[split] == g) & (size_bucket[split] == b)] = True
                        c = _size_cell_rank_change(
                            labels[split], logits[(pair[0], split)], logits[(pair[1], split)], mask, high_tasks
                        )
                        cells[f"{g}|{b}"] = c
                size_strat[pkey][split] = cells
    summary["rank_change"] = {
        "by_pair_split_group": _rank_aggregate(rank_rows),
        "size_stratified": size_strat,
        "note": "percentile units; positive = new model ranks that positive molecule higher than W2",
    }

    C.save_csv(BY_TASK_PATH, gain_rows)
    C.save_csv(RANK_BY_TASK_PATH, rank_rows)
    C.write_json(SUMMARY_PATH, summary)
    return summary


def _size_cell_rank_change(
    y: np.ndarray, la: np.ndarray, lb: np.ndarray, mask_rows: np.ndarray, high_tasks: set[int]
) -> dict[str, Any]:
    medians = []
    pooled = []
    n_pos = 0
    for t in sorted(high_tasks):
        yy = np.asarray(y[:, t], dtype=np.float64)
        obs = np.isfinite(yy) & mask_rows
        if obs.sum() < 3:
            continue
        pos = obs & (yy == 1.0)
        if pos.sum() == 0:
            continue
        pct_a = M.normalized_percentile(la[obs, t])
        pct_b = M.normalized_percentile(lb[obs, t])
        idx = np.flatnonzero(obs)
        pos_map = {int(p): k for k, p in enumerate(idx)}
        sel = np.array([pos_map[int(p)] for p in np.flatnonzero(pos)], dtype=np.int64)
        delta = pct_a[sel] - pct_b[sel]
        medians.append(float(np.median(delta)))
        pooled.append(delta)
        n_pos += int(sel.size)
    if not medians:
        return {"n_tasks": 0, "n_positive": 0, "median_delta_pct": None, "pooled_median_delta_pct": None}
    pooled_arr = np.concatenate(pooled)
    return {
        "n_tasks": len(medians),
        "n_positive": int(n_pos),
        "median_delta_pct": float(np.median(medians)),
        "pooled_median_delta_pct": float(np.median(pooled_arr)),
    }


def _rank_aggregate(rank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not rank_rows:
        return out
    df = pd.DataFrame(rank_rows)
    for pkey in df.pair.unique():
        out[pkey] = {}
        for split in ("valid", "test"):
            sub = df[(df.pair == pkey) & (df.split == split)]
            out[pkey][split] = {}
            for g in ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL"):
                cell = sub[sub.novelty_group == g]
                if cell.empty:
                    out[pkey][split][g] = {"n_tasks": 0}
                    continue
                out[pkey][split][g] = {
                    "n_tasks": int(cell.shape[0]),
                    "n_positive": int(cell.n_positive.sum()),
                    "median_of_task_medians": float(np.median(cell.median_delta_pct)),
                    "q1_of_task_medians": float(np.percentile(cell.median_delta_pct, 25)),
                    "q3_of_task_medians": float(np.percentile(cell.median_delta_pct, 75)),
                    "n_tasks_new_better": int((cell.median_delta_pct > 0).sum()),
                    "mean_frac_improved": float(cell.frac_improved.mean()),
                }
    return out


def main() -> int:
    s = run()
    for pkey, p in s["pairs"].items():
        print(f"== {pkey}: {p['verdict']['label']} ({p['verdict'].get('reason', '')})")
        for split in ("valid", "test"):
            g = p["splits"][split]["groups"]
            print(
                f"  {split:5s} main-cell mean delta AP: "
                f"SEEN {g['SEEN']['ge20']['mean_delta_ap']:+.5f} (n={g['SEEN']['ge20']['n_tasks']}) | "
                f"PARTIAL {g['PARTIAL_NOVEL']['ge20']['mean_delta_ap']:+.5f} (n={g['PARTIAL_NOVEL']['ge20']['n_tasks']}) | "
                f"HIGH {g['HIGH_NOVEL']['ge20']['mean_delta_ap']:+.5f} (n={g['HIGH_NOVEL']['ge20']['n_tasks']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
