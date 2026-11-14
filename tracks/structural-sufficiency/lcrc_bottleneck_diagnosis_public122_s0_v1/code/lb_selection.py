"""Selection rules, state-set selection, oracle gap, windows, bootstrap, jackknife.

All rules are frozen in ``lb_common`` and evaluated on the *same* candidate
state-set (the 50 grid points per phase, or the shared probe grid).  The three
pre-registered rules are:

* ``R_SEL`` primary: regression min Valid MSE, classification max Valid ROC-AUC;
* ``A_SEL``: regression min Valid MAE, classification min Valid BCE;
* ``W_SEL``: 5-point rolling mean of the primary criterion, model = window end.

``observed_oracle_gap`` is named exactly that: the *observed, finite-state-set*
post-hoc gap, containing Test-selection optimism.  It is not an achievable gain,
a population risk bound or an official score.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

import lb_common as C


def _criterion(rule: str, task: str, m: Mapping[str, Any]) -> float:
    if rule == C.R_SEL:
        return float(m["rmse"] if task == "regression" else m["roc_auc"])
    if rule == C.A_SEL:
        return float(m["mae"] if task == "regression" else m["bce"])
    raise KeyError(rule)


def _is_primary_rule(rule: str) -> bool:
    return rule == C.R_SEL


def select_state(task: str, rule: str, records: Sequence[Mapping[str, Any]],
                 *, include_step0: bool = False) -> dict[str, Any]:
    """Pick one state from a candidate set by a frozen rule.

    ``records`` are dicts with ``update`` and per-split metric dicts under the
    keys ``valid`` / ``train_fit`` / ``test``.  Ties keep the earliest update.
    """
    cand = [r for r in records if (include_step0 or int(r["update"]) > 0)]
    if not cand:
        raise ValueError("empty candidate set")
    if rule in (C.R_SEL, C.A_SEL):
        key = lambda r: (_criterion(rule, task, r["valid"]), int(r["update"]))  # noqa: E731
        best = min(cand, key=key) if not (rule == C.R_SEL and task == "binary") else \
            min(cand, key=lambda r: (-_criterion(rule, task, r["valid"]), int(r["update"])))
        return {"rule": rule, "update": int(best["update"]),
                "valid_criterion": _criterion(rule, task, best["valid"]),
                "selected": best}
    # W_SEL: 5-point window on the primary criterion
    primary = [_criterion(C.R_SEL, task, r["valid"]) for r in cand]
    best_val = None
    best_idx = None
    best_win = None
    for i in range(C.WINDOW_MIN_INDEX, len(cand)):
        win = primary[i - C.WINDOW + 1: i + 1]
        val = float(np.mean(win))
        better = (best_val is None or
                  (val > best_val if task == "binary" else val < best_val))
        if better:
            best_val = val
            best_idx = i
            best_win = list(range(i - C.WINDOW + 1, i + 1))
    # fall back to the raw primary criterion when there is no full window
    if best_idx is None:
        idx = int(np.argmax(primary)) if task == "binary" else int(np.argmin(primary))
        return {"rule": rule, "update": int(cand[idx]["update"]),
                "valid_criterion": primary[idx], "window": None, "selected": cand[idx],
                "window_fallback": True}
    return {"rule": rule, "update": int(cand[best_idx]["update"]), "valid_criterion": best_val,
            "window": [int(cand[j]["update"]) for j in best_win], "selected": cand[best_idx],
            "window_fallback": False}


def observed_oracle_gap(task: str, valid_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """R_Test(theta_Valid) - min_E R_Test(theta_E) over the *observed* state set."""
    if not valid_records:
        return {"observed_oracle_gap": None}
    if task == "regression":
        chosen = min(valid_records, key=lambda r: (r["valid"]["rmse"], int(r["update"])))
        rt_valid = float(chosen["test"]["rmse"])
        rt_min = float(min(r["test"]["rmse"] for r in valid_records))
        gap = rt_valid - rt_min
        arg_min = int(min(valid_records, key=lambda r: (r["test"]["rmse"], int(r["update"])))["update"])
        assert gap >= -1e-12, gap
    else:
        chosen = min(valid_records, key=lambda r: (-r["valid"]["roc_auc"], int(r["update"])))
        rt_valid = float(chosen["test"]["roc_auc"])
        rt_max = float(max(r["test"]["roc_auc"] for r in valid_records))
        gap = rt_max - rt_valid
        arg_min = int(max(valid_records, key=lambda r: (r["test"]["roc_auc"], -int(r["update"])))["update"])
    return {
        "observed_oracle_gap": float(gap),
        "test_at_valid_selected": float(rt_valid),
        "test_min_or_max_over_states": float(rt_min if task == "regression" else rt_max),
        "valid_selected_update": int(chosen["update"]),
        "oracle_update": arg_min,
        "definition": "R_Test(theta_valid)-min_E R_Test(theta_e); contains Test-selection optimism",
    }


# ---------------------------------------------------------------------------
# windows / persistence
# ---------------------------------------------------------------------------


def good_state_persistence(task: str, records: Sequence[Mapping[str, Any]], tol: float = 0.005) -> dict[str, Any]:
    """Length of the longest contiguous run of near-best Valid states."""
    if not records:
        return {"longest_run_updates": 0}
    if task == "regression":
        best = min(r["valid"]["rmse"] for r in records)
        ok = [r["valid"]["rmse"] <= best + tol for r in records]
    else:
        best = max(r["valid"]["roc_auc"] for r in records)
        ok = [r["valid"]["roc_auc"] >= best - tol for r in records]
    run = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0
        run = max(run, cur)
    return {"longest_run_points": int(run), "best_valid": float(best), "tol": tol}


# ---------------------------------------------------------------------------
# bootstrap / jackknife  (over the *evaluation rows*, not training seeds)
# ---------------------------------------------------------------------------


def bootstrap_selection(task: str, rule: str, states: Mapping[str, Any],
                        valid_rows: np.ndarray, test_rows: np.ndarray,
                        valid_pred: np.ndarray, valid_y: np.ndarray,
                        test_pred: np.ndarray, test_y: np.ndarray,
                        clusters: np.ndarray | None = None,
                        replicates: int = C.BOOTSTRAP_REPLICATES,
                        seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    """Validation-composition -> selection sensitivity (fixed training states).

    Resample the *validation rows* (or their clusters), re-select a state with the
    frozen rule on each replicate, then record that state's metric on the
    *original* Test set.  ``clusters`` (e.g. an ``eval_component`` label) is
    resampled as a whole; rows otherwise.
    """
    rng = np.random.default_rng(int(seed))
    n_states = int(np.asarray(valid_pred).shape[0])
    n_valid = len(valid_y)
    out_metrics: list[float] = []
    out_updates: list[int] = []
    out_train_order: list[list[int]] = []

    def pair_metric(pred, y):
        if task == "regression":
            return float(math.sqrt(np.mean((pred - y) ** 2)))
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan")
        p = 1.0 / (1.0 + np.exp(-pred))
        return float(roc_auc_score(y, p))

    if clusters is not None:
        uniq = np.unique(clusters)
        members = {c: np.where(clusters == c)[0] for c in uniq}
    else:
        members = None

    for _ in range(int(replicates)):
        if clusters is not None:
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([members[c] for c in draw]) if len(draw) else np.zeros(0, np.int64)
        else:
            idx = rng.integers(0, n_valid, size=n_valid)
        yb = valid_y[idx]
        vals = []
        for s in range(n_states):
            pred = np.asarray(valid_pred)[s][idx]
            vals.append(pair_metric(pred, yb))
        vals = np.asarray(vals, np.float64)
        if task == "regression":
            order = np.lexsort((np.arange(n_states), vals))
        else:
            order = np.lexsort((np.arange(n_states), -vals))
        best = int(order[0])
        out_updates.append(best)
        out_train_order.append(list(order[:3]))
        tp = np.asarray(test_pred)[best]
        out_metrics.append(pair_metric(tp, test_y))

    arr = np.asarray(out_metrics, np.float64)
    finite = arr[np.isfinite(arr)]
    updates = np.asarray(out_updates, np.int64)
    return {
        "rule": rule, "task": task, "replicates": int(replicates),
        "n_states": n_states, "cluster": (clusters is not None),
        "test_metric_p10": float(np.percentile(finite, 10)) if finite.size else None,
        "test_metric_p50": float(np.percentile(finite, 50)) if finite.size else None,
        "test_metric_p90": float(np.percentile(finite, 90)) if finite.size else None,
        "test_metric_mean": float(np.mean(finite)) if finite.size else None,
        "selected_update_counts": {int(u): int((updates == u).sum()) for u in np.unique(updates)},
        "selected_update_spread": int(len(np.unique(updates))),
    }


def jackknife_selection(task: str, rule: str, valid_rows: np.ndarray, test_rows: np.ndarray,
                        valid_pred: np.ndarray, valid_y: np.ndarray,
                        test_pred: np.ndarray, test_y: np.ndarray,
                        clusters: np.ndarray | None = None,
                        drop_rows: Sequence[int] = ()) -> dict[str, Any]:
    """Leave-one-cluster-out (or row-out) re-selection on the original Test."""
    n_states = int(np.asarray(valid_pred).shape[0])

    def pair_metric(pred, y):
        if task == "regression":
            return float(math.sqrt(np.mean((pred - y) ** 2)))
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan")
        p = 1.0 / (1.0 + np.exp(-pred))
        return float(roc_auc_score(y, p))

    if clusters is not None:
        uniq = np.unique(clusters)
        drops = [[c] for c in uniq]
    else:
        drops = [[int(r)] for r in np.asarray(valid_rows).tolist()]
    rows_index = np.asarray(valid_rows, np.int64)

    changes = []
    for drop in drops:
        mask = np.ones(len(valid_y), bool)
        if clusters is not None:
            mask &= ~np.isin(clusters, drop)
        else:
            mask &= ~np.isin(rows_index, drop)
        if mask.sum() < 10:
            continue
        yb = valid_y[mask]
        vals = np.array([pair_metric(np.asarray(valid_pred)[s][mask], yb) for s in range(n_states)])
        if task == "regression":
            order = np.lexsort((np.arange(n_states), vals))
        else:
            order = np.lexsort((np.arange(n_states), -vals))
        best = int(order[0])
        changes.append({"dropped": [int(x) for x in drop],
                        "selected_update": best,
                        "test_metric": pair_metric(np.asarray(test_pred)[best], test_y)})
    sel = np.asarray([c["selected_update"] for c in changes], np.int64)
    tst = np.asarray([c["test_metric"] for c in changes], np.float64)
    return {
        "rule": rule, "n_drops": len(changes),
        "selected_update_unique": int(len(np.unique(sel))),
        "selected_update_modal_fraction": float(np.bincount(sel).max() / max(len(sel), 1)) if len(sel) else None,
        "test_metric_min": float(np.nanmin(tst)) if len(tst) else None,
        "test_metric_max": float(np.nanmax(tst)) if len(tst) else None,
        "test_metric_spread": float(np.nanmax(tst) - np.nanmin(tst)) if len(tst) else None,
        "targeted_drop": [
            {"dropped": c["dropped"], "selected_update": c["selected_update"], "test_metric": c["test_metric"]}
            for c in changes if intersect(c["dropped"], drop_rows)
        ],
    }


def intersect(a, b) -> bool:
    b = set(int(x) for x in b)
    return any(int(x) in b for x in a)


# ---------------------------------------------------------------------------
# probe selector (mirrors lb_train.StreamSelector but lives here for reuse)
# ---------------------------------------------------------------------------


class ProbeSelector:
    """Selection that allows the shared step-0 state as a candidate.

    Unlike the main-arm selector, step 0 *is* a legitimate candidate.  ``W_SEL``
    falls back to the raw primary criterion over all considered states (which
    includes step 0) until a full 5-point window exists, and reports whichever of
    the two is better.
    """

    def __init__(self, task: str):
        self.task = task
        self.records: list[dict[str, Any]] = []
        self.best = {r: {"update": None, "criterion": None, "state": None, "valid": None}
                     for r in C.SELECTION_RULES}
        self._window: list[dict[str, Any]] = []
        self._window_rec: dict[str, Any] | None = None
        self._raw_rec: dict[str, Any] | None = None

    def _better(self, rule, val, best):
        if best is None:
            return math.isfinite(val)
        if rule == C.R_SEL and self.task == "binary":
            return val > best
        return val < best

    def consider(self, update: int, valid_metrics: Mapping[str, Any], state) -> None:
        keep = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.records.append({"update": int(update), "valid": dict(valid_metrics)})
        for rule in (C.R_SEL, C.A_SEL):
            val = _criterion(rule, self.task, valid_metrics)
            if self._better(rule, val, self.best[rule]["criterion"]):
                self.best[rule] = {"update": int(update), "criterion": val, "state": keep,
                                   "valid": dict(valid_metrics)}
        prim = _criterion(C.R_SEL, self.task, valid_metrics)
        rec = {"update": int(update), "primary": prim, "valid": dict(valid_metrics), "state": keep}
        self._window.append(rec)
        if len(self._window) > C.WINDOW:      # never retain more than one window (memory)
            self._window = self._window[-C.WINDOW:]
        if self._raw_rec is None or self._better(C.R_SEL, prim, self._raw_rec["primary"]):
            self._raw_rec = rec
        if len(self._window) >= C.WINDOW:
            win = self._window[-C.WINDOW:]
            mean_prim = float(np.mean([p["primary"] for p in win]))
            if self._window_rec is None or self._better(C.R_SEL, mean_prim, self._window_rec["mean"]):
                self._window_rec = {"mean": mean_prim, "end": win[-1],
                                    "window_updates": [p["update"] for p in win]}

    def freeze(self) -> dict[str, Any]:
        wsel: dict[str, Any]
        if self._window_rec is not None:
            end = self._window_rec["end"]
            wsel = {"update": end["update"], "criterion": self._window_rec["mean"],
                    "state": end["state"], "valid": end["valid"],
                    "window_updates": self._window_rec["window_updates"], "fallback": False}
            if self._raw_rec is not None and self._better(C.R_SEL, self._raw_rec["primary"],
                                                          self._window_rec["mean"]):
                r = self._raw_rec
                wsel = {"update": r["update"], "criterion": r["primary"], "state": r["state"],
                        "valid": r["valid"], "window_updates": None, "fallback": True}
        elif self._raw_rec is not None:
            r = self._raw_rec
            wsel = {"update": r["update"], "criterion": r["primary"], "state": r["state"],
                    "valid": r["valid"], "window_updates": None, "fallback": True}
        else:
            wsel = {"update": None, "criterion": None, "state": None, "valid": None,
                    "window_updates": None, "fallback": True}
        merged = {C.R_SEL: self.best[C.R_SEL], C.A_SEL: self.best[C.A_SEL], C.W_SEL: wsel}
        return {r: {"update": rec["update"], "criterion": rec["criterion"],
                    "valid_metrics": rec["valid"], "has_state": rec["state"] is not None,
                    "window_updates": rec.get("window_updates"), "state": rec["state"],
                    "fallback": rec.get("fallback")}
                for r, rec in merged.items()}


__all__ = [n for n in dir() if not n.startswith("_")]
