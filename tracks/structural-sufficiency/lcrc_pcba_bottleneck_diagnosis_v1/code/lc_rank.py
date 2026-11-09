"""P1 — prediction-function convergence diagnostics on frozen logits.

For the pre-registered model pairs, on the fixed task sets, for each task:

* A: Spearman rho of the two logit vectors on the commonly observed molecules,
* B: deterministic Top-K (K1 = ceil(1% n), K5 = ceil(5% n)) Jaccard overlap and the
  number of true positives inside each model's Top-K,
* C: mutual positive discovery (both / only reference / only new / neither) and
  the normalized rank percentile of every true positive in both models.

No cross-task score mixing, no internal-representation analysis.

Outputs: ``analysis/ranking_by_task.csv``, ``analysis/topk_by_task.csv``,
``analysis/prediction_similarity_summary.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import lc_common as C
import lc_data as D
import lc_metrics as M

RANKING_PATH = C.ANALYSIS / "ranking_by_task.csv"
TOPK_PATH = C.ANALYSIS / "topk_by_task.csv"
SUMMARY_PATH = C.ANALYSIS / "prediction_similarity_summary.json"


def _task_metrics(
    *,
    pair: tuple[str, str],
    split: str,
    t: int,
    y_col: np.ndarray,
    rows: np.ndarray,
    la: np.ndarray,
    lb: np.ndarray,
    high_support: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    a, b = pair
    yy = np.asarray(y_col, dtype=np.float64)
    obs = np.isfinite(yy)
    n = int(obs.sum())
    n_pos = int((yy[obs] == 1.0).sum())
    n_neg = int((yy[obs] == 0.0).sum())
    stratum = C.support_stratum(n_pos, n_neg)
    if stratum == "na":
        return {}, {}

    o_rows = rows[obs]
    ya = np.asarray(la[obs], dtype=np.float64)
    yb = np.asarray(lb[obs], dtype=np.float64)

    rho = M.spearman(ya, yb)
    k1 = M.topk_size(n, 0.01)
    k5 = M.topk_size(n, 0.05)
    idx_a1 = M.top_k_indices(ya, o_rows, k1)
    idx_b1 = M.top_k_indices(yb, o_rows, k1)
    idx_a5 = M.top_k_indices(ya, o_rows, k5)
    idx_b5 = M.top_k_indices(yb, o_rows, k5)
    j1 = M.jaccard(idx_a1, idx_b1)
    j5 = M.jaccard(idx_a5, idx_b5)
    mt1 = M.mutual_positive_table(idx_a1, idx_b1, yy[obs])
    mt5 = M.mutual_positive_table(idx_a5, idx_b5, yy[obs])
    ranks = M.positive_rank_table(yb, ya, yy[obs])  # reference=b, new=a; delta = pct_new - pct_reference

    ranking = {
        "pair": C.pair_key(a, b),
        "new_model": a,
        "reference_model": b,
        "split": split,
        "task": t,
        "support_stratum": stratum,
        "high_support": t in high_support,
        "n_observed": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "spearman_rho": rho,
    }
    topk = {
        "pair": C.pair_key(a, b),
        "split": split,
        "task": t,
        "support_stratum": stratum,
        "high_support": t in high_support,
        "n_observed": n,
        "n_positive": n_pos,
        "k1": k1,
        "k5": k5,
        "jaccard_k1": j1,
        "jaccard_k5": j5,
        "pos_in_topk1_new": mt1["pos_in_a"],
        "pos_in_topk1_reference": mt1["pos_in_b"],
        "pos_in_topk5_new": mt5["pos_in_a"],
        "pos_in_topk5_reference": mt5["pos_in_b"],
        "both_pos_k1": mt1["both"],
        "only_new_pos_k1": mt1["only_a"],
        "only_reference_pos_k1": mt1["only_b"],
        "neither_pos_k1": mt1["neither"],
        "both_pos_k5": mt5["both"],
        "only_new_pos_k5": mt5["only_a"],
        "only_reference_pos_k5": mt5["only_b"],
        "neither_pos_k5": mt5["neither"],
        "median_pct_new": ranks["median_pct_new"],
        "median_pct_reference": ranks["median_pct_reference"],
        "median_delta_pct": ranks["median_delta_pct"],
        "frac_positive_rank_improved": ranks["frac_improved"],
    }
    return ranking, topk


def _aggregate(rows: list[dict[str, Any]], high: set[int], low: set[int]) -> dict[str, Any]:
    def vals(key: str, subset: list[dict[str, Any]]) -> list[float]:
        out = []
        for r in subset:
            v = r.get(key)
            if v is None or v == "":
                continue
            if isinstance(v, float) and not np.isfinite(v):
                continue
            out.append(float(v))
        return out

    scorable = [r for r in rows if r["support_stratum"] != "na"]
    hi = [r for r in scorable if r["task"] in high]
    lo = [r for r in scorable if r["task"] in low]
    rest = [r for r in scorable if r["task"] not in high and r["task"] not in low]
    rho_all = vals("spearman_rho", scorable)
    rho_high = vals("spearman_rho", hi)
    out = {
        "n_tasks_scorable": len(scorable),
        "n_tasks_high_support": len(hi),
        "n_tasks_low_support": len(lo),
        "n_tasks_neither_high_nor_low": len(rest),
        "spearman_all": C.quantiles(rho_all),
        "spearman_high_support": C.quantiles(rho_high),
        "spearman_high_support_lt_0.5": int(sum(1 for v in rho_high if v < 0.5)),
        "spearman_high_support_lt_0.8": int(sum(1 for v in rho_high if v < 0.8)),
        "spearman_high_support_negative": int(sum(1 for v in rho_high if v < 0.0)),
        # LOW_SUPPORT is the frozen Test-defined set (<20 Test positives), 19 tasks.
        "spearman_low_support": C.quantiles(vals("spearman_rho", lo)),
        # Remaining scorable tasks not in either frozen set (Test-high/Valid-low), n=2.
        "spearman_not_in_frozen_sets": C.quantiles(vals("spearman_rho", rest)),
    }
    return out


def _aggregate_topk(rows: list[dict[str, Any]], high: set[int], low: set[int]) -> dict[str, Any]:
    scorable = [r for r in rows if r["support_stratum"] != "na"]
    hi = [r for r in scorable if r["task"] in high]
    lo = [r for r in scorable if r["task"] in low]
    rest = [r for r in scorable if r["task"] not in high and r["task"] not in low]

    def col(sub: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in sub if r.get(key) not in (None, "")]

    def s(sub: list[dict[str, Any]], key: str) -> int:
        return int(sum(int(r[key]) for r in sub))

    return {
        "n_tasks_high_support": len(hi),
        "n_tasks_low_support": len(lo),
        "jaccard_k1_high_support": C.quantiles(col(hi, "jaccard_k1")),
        "jaccard_k5_high_support": C.quantiles(col(hi, "jaccard_k5")),
        "jaccard_k1_all": C.quantiles(col(scorable, "jaccard_k1")),
        "jaccard_k5_all": C.quantiles(col(scorable, "jaccard_k5")),
        "jaccard_k1_low_support": C.quantiles(col(lo, "jaccard_k1")),
        "jaccard_k5_low_support": C.quantiles(col(lo, "jaccard_k5")),
        "n_tasks_neither_high_nor_low": len(rest),
        "jaccard_k1_not_in_frozen_sets": C.quantiles(col(rest, "jaccard_k1")),
        "jaccard_k5_not_in_frozen_sets": C.quantiles(col(rest, "jaccard_k5")),
        "high_support_sums": {
            "positives_total": s(hi, "n_positive"),
            "pos_in_topk1_new": s(hi, "pos_in_topk1_new"),
            "pos_in_topk1_reference": s(hi, "pos_in_topk1_reference"),
            "pos_in_topk5_new": s(hi, "pos_in_topk5_new"),
            "pos_in_topk5_reference": s(hi, "pos_in_topk5_reference"),
            "both_pos_k1": s(hi, "both_pos_k1"),
            "only_new_pos_k1": s(hi, "only_new_pos_k1"),
            "only_reference_pos_k1": s(hi, "only_reference_pos_k1"),
            "neither_pos_k1": s(hi, "neither_pos_k1"),
            "both_pos_k5": s(hi, "both_pos_k5"),
            "only_new_pos_k5": s(hi, "only_new_pos_k5"),
            "only_reference_pos_k5": s(hi, "only_reference_pos_k5"),
            "neither_pos_k5": s(hi, "neither_pos_k5"),
        },
        "high_support_median_delta_pct": C.quantiles(col(hi, "median_delta_pct")),
        "high_support_frac_positive_rank_improved": C.quantiles(col(hi, "frac_positive_rank_improved")),
        "high_support_tasks_new_better_rank": int(sum(1 for r in hi if (r["median_delta_pct"] or 0.0) > 0.0)),
    }


def _verdict(pair_summary: dict[str, Any]) -> dict[str, Any]:
    hi_valid = pair_summary["valid"]["spearman"]["spearman_high_support"]["median"]
    hi_test = pair_summary["test"]["spearman"]["spearman_high_support"]["median"]
    j1_valid = pair_summary["valid"]["topk"]["jaccard_k1_high_support"]["median"]
    j1_test = pair_summary["test"]["topk"]["jaccard_k1_high_support"]["median"]
    j5_valid = pair_summary["valid"]["topk"]["jaccard_k5_high_support"]["median"]
    j5_test = pair_summary["test"]["topk"]["jaccard_k5_high_support"]["median"]
    vals = [hi_valid, hi_test, j1_valid, j1_test, j5_valid, j5_test]
    if any(v is None for v in vals):
        return {"label": "H1_UNRESOLVED", "reason": "missing aggregate statistic"}
    ranking_similar = hi_valid >= C.RHO_SIMILAR and hi_test >= C.RHO_SIMILAR
    decision_similar = (
        j1_valid >= C.TOPK1_JACCARD_SIMILAR
        and j1_test >= C.TOPK1_JACCARD_SIMILAR
        and j5_valid >= C.TOPK5_JACCARD_SIMILAR
        and j5_test >= C.TOPK5_JACCARD_SIMILAR
    )
    if ranking_similar and decision_similar:
        label = "H1_SUPPORTED_SIMILAR_RANKING_AND_DECISIONS"
    elif ranking_similar:
        label = "GLOBAL_RANKING_SIMILAR_BUT_DECISION_REGION_DIFFERS"
    elif hi_valid >= C.RHO_SIMILAR or hi_test >= C.RHO_SIMILAR:
        label = "H1_MIXED"
    else:
        label = "H1_NOT_SUPPORTED"
    return {
        "label": label,
        "criteria": {
            "ranking_similar_both_splits": ranking_similar,
            "decision_region_similar_both_splits": decision_similar,
            "rho_threshold": C.RHO_SIMILAR,
            "jaccard_k1_threshold": C.TOPK1_JACCARD_SIMILAR,
            "jaccard_k5_threshold": C.TOPK5_JACCARD_SIMILAR,
            "rho_median_valid": hi_valid,
            "rho_median_test": hi_test,
            "jaccard_k1_median_valid": j1_valid,
            "jaccard_k1_median_test": j1_test,
            "jaccard_k5_median_valid": j5_valid,
            "jaccard_k5_median_test": j5_test,
        },
    }


def _representative(ranking: list[dict[str, Any]], topk: list[dict[str, Any]], high: set[int]) -> dict[str, Any]:
    """Extremal tasks chosen by fixed metrics only (no cherry-picking)."""

    def pick(rows: list[dict[str, Any]], key: str, most: bool) -> dict[str, Any] | None:
        cand = [r for r in rows if r["task"] in high and r.get(key) not in (None, "")]
        if not cand:
            return None
        r = max(cand, key=lambda x: x[key]) if most else min(cand, key=lambda x: x[key])
        return {"task": r["task"], "value": float(r[key]), "n_observed": r["n_observed"], "n_positive": r["n_positive"]}

    return {
        "highest_rho": pick(ranking, "spearman_rho", True),
        "lowest_rho": pick(ranking, "spearman_rho", False),
        "highest_jaccard_k1": pick(topk, "jaccard_k1", True),
        "lowest_jaccard_k1": pick(topk, "jaccard_k1", False),
    }


def run() -> dict[str, Any]:
    task_sets = C.read_json(C.ANALYSIS / "task_sets.json")
    high = set(int(t) for t in task_sets["HIGH_SUPPORT"])
    low = set(int(t) for t in task_sets["LOW_SUPPORT_test"])
    labels = {s: D.split_labels(s) for s in ("valid", "test")}
    rows = {s: D.split_rows(s) for s in ("valid", "test")}
    logits = {(label, s): D.load_logits(label, s) for label in C.MODEL_ORDER for s in ("valid", "test")}

    ranking_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": "lcrc-pcba-bottleneck-diagnosis-similarity-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "task_sets": {"task_sets_path": C.rel(C.ANALYSIS / "task_sets.json"), "n": task_sets["n"]},
        "definitions": {
            "spearman": "per task, Pearson correlation of average ranks of raw logits on observed molecules",
            "topk": "K1=ceil(0.01*n_observed), K5=ceil(0.05*n_observed), ties broken by ascending official row id",
            "delta_pct": "normalized percentile (1=best) of a true positive in the new model minus the reference model; positive = new model ranks it higher",
            "aggregation": "medians over tasks; no cross-task score mixing",
        },
        "pairs": {},
        "warnings": [
            "High Spearman only indicates similar global ordering; negatives dominate most tasks and can drive it.",
            "Exclusive high-ranked positives are not a net gain: reference-only positives are reported alongside.",
            "Test is an exposed comparison set; all numbers are descriptive.",
        ],
    }

    for pair in C.ALL_PAIRS:
        a, b = pair
        pkey = C.pair_key(a, b)
        pair_out: dict[str, Any] = {
            "new_model": a,
            "reference_model": b,
            "orientation": f"{a} - {b}",
        }
        for split in ("valid", "test"):
            y = labels[split]
            r = rows[split]
            la = logits[(a, split)]
            lb = logits[(b, split)]
            split_ranking: list[dict[str, Any]] = []
            split_topk: list[dict[str, Any]] = []
            for t in range(C.N_TASKS):
                rk, tk = _task_metrics(
                    pair=pair, split=split, t=t, y_col=y[:, t], rows=r, la=la[:, t], lb=lb[:, t], high_support=high
                )
                if rk:
                    split_ranking.append(rk)
                    split_topk.append(tk)
            ranking_rows.extend(split_ranking)
            topk_rows.extend(split_topk)
            pair_out[split] = {
                "spearman": _aggregate(split_ranking, high, low),
                "topk": _aggregate_topk(split_topk, high, low),
                "representative_tasks": _representative(split_ranking, split_topk, high),
            }
        pair_out["verdict"] = _verdict(pair_out)
        summary["pairs"][pkey] = pair_out

    summary["signature"] = {
        "ranking_by_task": C.rel(RANKING_PATH),
        "topk_by_task": C.rel(TOPK_PATH),
        "n_ranking_rows": len(ranking_rows),
        "n_topk_rows": len(topk_rows),
    }
    C.save_csv(RANKING_PATH, ranking_rows)
    C.save_csv(TOPK_PATH, topk_rows)
    C.write_json(SUMMARY_PATH, summary)
    return summary


def main() -> int:
    s = run()
    for pkey, p in s["pairs"].items():
        print(f"{pkey:32s} {p['verdict']['label']}")
        for split in ("valid", "test"):
            rho = p[split]["spearman"]["spearman_high_support"]
            j1 = p[split]["topk"]["jaccard_k1_high_support"]
            j5 = p[split]["topk"]["jaccard_k5_high_support"]
            print(
                f"  {split:5s} rho_hi med {rho['median']:.3f} q1 {rho['q1']:.3f} | "
                f"j1 med {j1['median']:.3f} | j5 med {j5['median']:.3f} | "
                f"rank-improved tasks {p[split]['topk']['high_support_tasks_new_better_rank']}/"
                f"{p[split]['topk']['n_tasks_high_support']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
