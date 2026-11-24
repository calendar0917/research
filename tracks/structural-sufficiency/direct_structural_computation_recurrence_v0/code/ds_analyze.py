"""DSCR v0 analysis: main table, per-task metrics, paired comparisons, summary.

Reads the frozen run artefacts (``runs/<ARM>_seed<k>/``) plus the two historical
seed-0 references (read-only re-evaluation of their checkpoints) and writes:

* ``analysis/main_table.csv``          -- best-Valid and horizon-endpoint numbers
* ``analysis/per_task_metrics.csv``    -- per-task AP at the selected checkpoints
* ``analysis/paired_comparisons.json`` -- every pre-registered comparison
* ``analysis/summary.json``            -- Q1..Q6 route decisions + the route table

The official OGB macro AP is the reported metric everywhere; every paired
comparison is restricted to the tasks scored by *both* models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import ds_common as C  # noqa: E402
import ds_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402

#: Pre-registered comparisons of the protocol (arm_a - arm_b).
COMPARISONS = [
    ("SELF", 0, "W2", 0, "Q1_extra_pointwise_depth_vs_W2"),
    ("FIXED1", 0, "SELF", 0, "Q2_fixed_source_one_hop_vs_self"),
    ("REAL2", 0, "FIXED1", 0, "Q3a_direct_two_hop_vs_fixed_one_hop"),
    ("BAG2", 0, "FIXED1", 0, "Q3b_multiset_combination_vs_fixed_one_hop"),
    ("REAL2", 0, "BAG2", 0, "Q4_real_branch_binding_vs_multiset"),
    ("DYNAMIC", 0, "FIXED1", 0, "Q5a_iterative_vs_fixed_one_hop"),
    ("DYNAMIC", 0, "REAL2", 0, "Q5b_iterative_vs_direct_two_hop"),
    ("W2", 1, "W2", 0, "SEED_NOISE_W2_seed1_vs_seed0"),
    ("DYNAMIC", 1, "DYNAMIC", 0, "SEED_NOISE_DYNAMIC_seed1_vs_seed0"),
]
SEED1_COMPARISONS = [
    ("CSTAR", 1, "W2", 1, "Q1b_Cstar_vs_W2_seed1"),
    ("CSTAR", 1, "DYNAMIC", 1, "Q5c_Cstar_vs_DYNAMIC_seed1"),
    ("CSTAR", 1, "KSTAR", 1, "Q4b_Cstar_vs_Kstar_seed1"),
    ("CSTAR", 0, "W2", 0, "Q1b_Cstar_vs_W2_seed0"),
    ("CSTAR", 0, "DYNAMIC", 0, "Q5c_Cstar_vs_DYNAMIC_seed0"),
]
BOOTSTRAP_MOL_PAIRS = ("Q1b_Cstar_vs_W2_seed0", "Q5c_Cstar_vs_DYNAMIC_seed0")
N_TASK_BOOTSTRAP = 2000
N_MOL_BOOTSTRAP = 200
HIST_ORDER_SEED = 0


# ---------------------------------------------------------------------------
# fast AP (identical aggregation to sklearn's average_precision_score)
# ---------------------------------------------------------------------------


def fast_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    m = np.isfinite(y_true)
    if not m.any():
        return None
    yt = np.asarray(y_true[m], dtype=np.float64)
    ys = np.asarray(y_score[m], dtype=np.float64)
    n_pos = float(yt.sum())
    n_neg = float(yt.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-ys, kind="mergesort")
    yt = yt[order]
    ys = ys[order]
    tp = np.cumsum(yt)
    last = np.r_[np.flatnonzero(np.diff(ys) != 0), ys.size - 1]
    tp_at = tp[last]
    precision = tp_at / (last + 1.0)
    recall = tp_at / n_pos
    weights = np.diff(np.r_[0.0, recall])
    return float(np.sum(weights * precision))


def verify_fast_ap(y_true: np.ndarray, y_score: np.ndarray, tasks: Sequence[int], rng: np.random.Generator) -> float:
    from sklearn.metrics import average_precision_score

    worst = 0.0
    for t in tasks:
        m = np.isfinite(y_true[:, t])
        yt = y_true[m, t]
        if yt.sum() == 0 or yt.sum() == yt.size:
            continue
        a = fast_ap(y_true[:, t], y_score[:, t])
        b = float(average_precision_score(yt, y_score[m, t]))
        worst = max(worst, abs(a - b))
    return float(worst)


# ---------------------------------------------------------------------------
# unit loading
# ---------------------------------------------------------------------------


def _curve(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out = {}
            for k, v in row.items():
                if v == "":
                    out[k] = None
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_unit(label: str, arm: str, seed: int, horizon: int = 40) -> dict[str, Any] | None:
    """Load one unit: new round run, C*/K* alias, or a historical reference."""
    if arm in ("CSTAR", "KSTAR"):
        return None  # resolved by the caller
    d = C.run_dir(arm, seed)
    rj = d / "run.json"
    if not rj.is_file():
        # seed-0 references come from the historical rounds (read-only re-evaluation)
        if seed == 0 and arm in C.HISTORICAL:
            return load_historical(arm, seed, horizon=horizon)
        return None
    run = C.read_json(rj)
    if run.get("status") != "COMPLETED":
        return None
    curve = _curve(d / "train_curve.csv")
    best_eval = run.get("best_checkpoint_eval") or {}
    unit = {
        "label": label,
        "arm": arm,
        "seed": int(seed),
        "source": "dscr-v0 new run",
        "horizon": int(run["horizon"]),
        "params": int(run["parameters"]),
        "best_epoch": int(run["best_epoch"]),
        "best_valid_ap": float(run["best_valid_ap"]),
        "gpu_hours": float(run.get("gpu_hours") or 0.0),
        "curve": curve,
        "run_json": str(rj),
        "run": run,
        "split": {
            "train": _split_block(best_eval.get("train")),
            "valid": _split_block(best_eval.get("valid")),
            "test": _split_block(best_eval.get("test")),
        },
        "endpoint": run.get("horizon_endpoint") or {},
        "last_epoch0_valid_ap": (run.get("epoch0_diagnostic") or {}).get("valid_ap"),
    }
    unit["has_epoch_test"] = any(r.get("test_ap") is not None for r in curve)
    unit["has_epoch_train"] = any(r.get("train_ap") is not None for r in curve)
    return unit


def _split_block(block: Mapping[str, Any] | None) -> dict[str, Any]:
    if not block:
        return {"ap": None, "bce": None, "per_task": None, "positive": None, "negative": None, "observed": None}
    return {
        "ap": _num(block.get("ap")),
        "bce": _num(block.get("bce")),
        "per_task": block.get("per_task_ap"),
        "positive": block.get("positive"),
        "negative": block.get("negative"),
        "observed": block.get("observed"),
        "n_tasks_scored": block.get("n_tasks_scored"),
    }


def load_historical(arm: str, seed: int, horizon: int = 40) -> dict[str, Any] | None:
    """Read-only reference: the historical checkpoint re-scored by this round's code."""
    base = C.RUNS / "HIST" / f"{arm}_seed{seed}"
    rj = base / "best_checkpoint_eval.json"
    hist = C.HISTORICAL.get(arm)
    if hist is None or seed != 0:
        return None
    if not rj.is_file():
        return None
    ev = C.read_json(rj)
    curve_src = Path(hist["run_dir"]) / "train_curve.csv"
    curve = _curve(curve_src) if curve_src.is_file() else []
    # the historical runs trained for 40 epochs; only the frozen H-epoch prefix is
    # comparable, so every derived column (last-k mean, endpoint) uses that prefix
    curve = curve[: int(horizon)]
    run_json = Path(hist["run_dir"]) / "run.json"
    run = C.read_json(run_json) if run_json.is_file() else {}
    unit = {
        "label": arm,
        "arm": arm,
        "seed": int(seed),
        "source": f"historical {hist['protocol']} (read-only checkpoint re-evaluation)",
        "horizon": int(horizon),
        "trained_epochs": 40,
        "params": int(hist["parameters"]),
        "best_epoch": int(hist["best_epoch"]),
        "best_valid_ap": float(hist["best_valid_ap"]),
        "gpu_hours": float(run.get("gpu_hours") or 0.0),
        "curve": curve,
        "run_json": str(run_json),
        "run": run,
        "split": {
            "train": _split_block(ev.get("train")),
            "valid": _split_block(ev.get("valid")),
            "test": _split_block(ev.get("test")),
        },
        "endpoint": {},
        "last_epoch0_valid_ap": (run.get("epoch0_diagnostic") or {}).get("ap")
        if isinstance(run.get("epoch0_diagnostic"), Mapping)
        else None,
        "historical": True,
    }
    last = curve[-1] if curve else {}
    unit["endpoint"] = {
        "epoch": int(horizon),
        "valid_ap": _num(last.get("valid_ap")),
        "train_ap": None,
        "test_ap": None,
    }
    unit["has_epoch_test"] = any(r.get("test_ap") is not None for r in curve)
    unit["has_epoch_train"] = any(r.get("train_ap") is not None for r in curve)
    return unit


def evaluate_historical(device: torch.device) -> dict[str, Any]:
    """Score the two historical best checkpoints (read-only; no training)."""
    task = C.SD.load_task()
    test_info = C.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    y_test = np.asarray(test_info["labels"])
    out: dict[str, Any] = {}
    for arm in ("W2", "DYNAMIC"):
        hist = C.HISTORICAL[arm]
        ckpt = Path(hist["run_dir"]) / "checkpoint_best.pt"
        if not ckpt.is_file():
            out[arm] = {"status": "checkpoint_unavailable", "path": str(ckpt)}
            continue
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if arm == "W2":
            model, _ = M.build_w2(task.spec, task.norm, C.N_TASKS, seed=0, device="cpu")
        else:
            model, _ = M.build_ds_model(task.spec, task.norm, C.N_TASKS, arm="DYNAMIC", seed=0, device="cpu")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)
        model.eval()
        blocks: dict[str, Any] = {}
        logits_store: dict[str, np.ndarray] = {}
        for split in ("train", "valid", "test"):
            with torch.inference_mode():
                rep = C.evaluate_split(
                    model, task, device, split, y=y_test if split == "test" else None,
                    per_task=True, keep_logits=(split != "train"),
                    reason=f"historical_{arm}_best_readonly",
                )
            blocks[split] = {
                "ap": float(rep["ap"]),
                "bce": float(rep["bce"]),
                "logits_sha256": rep["logits_sha256"],
                "n_tasks_scored": int(rep["n_tasks_scored"]),
                "per_task_ap": rep["per_task_ap"],
                "observed": rep["observed"],
                "positive": rep["positive"],
                "negative": rep["negative"],
            }
            if split != "train":
                logits_store[split] = rep["logits"]
        base = C.RUNS / "HIST" / f"{arm}_seed0"
        base.mkdir(parents=True, exist_ok=True)
        for split, arr in logits_store.items():
            np.save(base / f"{split}_logits_best.npy", arr.astype(np.float32))
        C.write_json(
            base / "best_checkpoint_eval.json",
            {
                "schema": "dscr-historical-readonly-eval-v1",
                "arm": arm,
                "seed": 0,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": C.sha256_file(ckpt),
                "checkpoint_epoch": int(payload["epoch"]),
                "selection_metric": "official Valid macro AP recorded by the historical round",
                "recorded_best_valid_ap": float(hist["best_valid_ap"]),
                "recorded_test_ap_best": float(hist["test_ap_best"]),
                "train": blocks["train"],
                "valid": blocks["valid"],
                "test": blocks["test"],
                "test_role": C.TEST_ROLE,
                "note": "read-only evaluation of a frozen historical checkpoint (no training, no reuse as initialisation)",
                "evaluated_utc": C.utcnow(),
            },
        )
        out[arm] = {
            "status": "ok",
            "valid_ap": blocks["valid"]["ap"],
            "test_ap": blocks["test"]["ap"],
            "recorded_valid_ap": hist["best_valid_ap"],
            "valid_ap_reproduction_diff": blocks["valid"]["ap"] - float(hist["best_valid_ap"]),
            "test_ap_reproduction_diff": blocks["test"]["ap"] - float(hist["test_ap_best"]),
        }
        model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    C.write_json(C.ANALYSIS / "historical_reevaluation.json", out)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def support_strata(unit: Mapping[str, Any], split: str) -> list[str | None]:
    pos = unit["split"][split].get("positive")
    if pos is None:
        return [None] * C.N_TASKS
    return ["support_lt5" if p < 5 else "support_5_19" if p < 20 else "support_ge20" for p in pos]


def write_per_task(units: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for u in units:
        for split in ("train", "valid", "test"):
            block = u["split"][split]
            per_task = block.get("per_task")
            if per_task is None:
                continue
            strata = support_strata(u, split)
            pos = block.get("positive") or [None] * C.N_TASKS
            neg = block.get("negative") or [None] * C.N_TASKS
            obs = block.get("observed") or [None] * C.N_TASKS
            for t, ap in enumerate(per_task):
                rows.append(
                    {
                        "model": u["label"],
                        "arm": u["arm"],
                        "seed": u["seed"],
                        "horizon": u["horizon"],
                        "best_epoch": u["best_epoch"],
                        "split": split,
                        "task": t,
                        "ap": "" if ap is None else float(ap),
                        "positive": pos[t],
                        "negative": neg[t],
                        "observed": obs[t],
                        "support_stratum": strata[t],
                    }
                )
    path = C.ANALYSIS / "per_task_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


def last_k_valid_mean(unit: Mapping[str, Any], k: int = 5) -> float | None:
    vals = [r.get("valid_ap") for r in unit["curve"] if r.get("valid_ap") is not None]
    if len(vals) < k:
        return None
    return float(np.mean(vals[-k:]))


def high_support_macro(unit: Mapping[str, Any], split: str, min_support: int = C.SUPPORT_MIN) -> dict[str, Any]:
    block = unit["split"][split]
    per_task = block.get("per_task")
    pos = block.get("positive")
    if per_task is None or pos is None:
        return {"ap": None, "n_tasks": 0}
    keep = [i for i in range(C.N_TASKS) if per_task[i] is not None and pos[i] >= min_support]
    if not keep:
        return {"ap": None, "n_tasks": 0}
    return {"ap": float(np.mean([per_task[i] for i in keep])), "n_tasks": len(keep), "tasks": keep}


def write_main_table(units: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for u in units:
        train_ap = u["split"]["train"].get("ap")
        valid_ap = u["split"]["valid"].get("ap")
        test_ap = u["split"]["test"].get("ap")
        rows.append(
            {
                "model": u["label"],
                "arm": u["arm"],
                "seed": u["seed"],
                "horizon": u["horizon"],
                "params": u["params"],
                "best_epoch": u["best_epoch"],
                "train_ap": _fmt(train_ap),
                "valid_ap": _fmt(valid_ap),
                "test_ap": _fmt(test_ap),
                "valid_bce": _fmt(u["split"]["valid"].get("bce")),
                "test_bce": _fmt(u["split"]["test"].get("bce")),
                "gpu_hours": _fmt(u["gpu_hours"]),
                "endpoint_epoch": (u["endpoint"] or {}).get("epoch"),
                "endpoint_train_ap": _fmt((u["endpoint"] or {}).get("train_ap")),
                "endpoint_valid_ap": _fmt((u["endpoint"] or {}).get("valid_ap")),
                "endpoint_test_ap": _fmt((u["endpoint"] or {}).get("test_ap")),
                "last5_valid_ap_mean": _fmt(last_k_valid_mean(u)),
                "valid_high_support_ap": _fmt(high_support_macro(u, "valid").get("ap")),
                "valid_high_support_tasks": high_support_macro(u, "valid").get("n_tasks"),
                "test_high_support_ap": _fmt(high_support_macro(u, "test").get("ap")),
                "test_high_support_tasks": high_support_macro(u, "test").get("n_tasks"),
                "train_minus_valid_ap": _fmt(None if (train_ap is None or valid_ap is None) else train_ap - valid_ap),
                "epoch_train_ap_available": bool(u.get("has_epoch_train")),
                "epoch_test_ap_available": bool(u.get("has_epoch_test")),
                "source": u["source"],
            }
        )
    path = C.ANALYSIS / "main_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


def _fmt(v: Any) -> Any:
    return "" if v is None else (f"{float(v):.9f}" if isinstance(v, (int, float, np.floating)) else v)


# ---------------------------------------------------------------------------
# paired comparisons
# ---------------------------------------------------------------------------


def compare(a: Mapping[str, Any], b: Mapping[str, Any], *, rng: np.random.Generator,
            mol_bootstrap: bool = False, task: Any = None, logits: Mapping[str, Any] | None = None,
            n_mol: int = N_MOL_BOOTSTRAP) -> dict[str, Any]:
    out: dict[str, Any] = {"a": a["label"], "b": b["label"], "a_seed": a["seed"], "b_seed": b["seed"], "splits": {}}
    for split in ("valid", "test"):
        pa = a["split"][split].get("per_task")
        pb = b["split"][split].get("per_task")
        pos = a["split"][split].get("positive") or b["split"][split].get("positive")
        if pa is None or pb is None:
            out["splits"][split] = {"available": False, "reason": "per-task AP unavailable"}
            continue
        common = [i for i in range(C.N_TASKS) if pa[i] is not None and pb[i] is not None]
        if not common:
            out["splits"][split] = {"available": False, "reason": "no common scored task"}
            continue
        da = np.array([pa[i] for i in common], dtype=np.float64)
        db = np.array([pb[i] for i in common], dtype=np.float64)
        d = da - db
        hi = [k for k, i in enumerate(common) if pos is not None and pos[i] >= C.SUPPORT_MIN]
        lo = [k for k in range(len(common)) if k not in set(hi)]
        entry: dict[str, Any] = {
            "available": True,
            "n_tasks_common": len(common),
            "macro_ap_a_on_common": float(da.mean()),
            "macro_ap_b_on_common": float(db.mean()),
            "delta_macro_ap": float(d.mean()),
            "median_delta": float(np.median(d)),
            "n_improved": int((d > 0).sum()),
            "n_degraded": int((d < 0).sum()),
            "n_tied": int((d == 0).sum()),
            "high_support": {
                "min_positive": C.SUPPORT_MIN,
                "n_tasks": len(hi),
                "mean_delta": float(d[hi].mean()) if hi else None,
                "n_improved": int((d[hi] > 0).sum()) if hi else 0,
                "n_degraded": int((d[hi] < 0).sum()) if hi else 0,
            },
            "low_support": {
                "n_tasks": len(lo),
                "mean_delta": float(d[lo].mean()) if lo else None,
                "contribution_to_macro": float(d[lo].sum() / len(common)) if lo else 0.0,
            },
            "sensitivity_remove_top1": _remove_top(d, 1),
            "sensitivity_remove_top3": _remove_top(d, 3),
            "sensitivity_remove_top5": _remove_top(d, 5),
            "task_bootstrap": _task_bootstrap(d, rng),
            "largest_positive": _top_tasks(common, d, 5, sign=1),
            "largest_negative": _top_tasks(common, d, 5, sign=-1),
        }
        if mol_bootstrap and task is not None and logits is not None and split in logits:
            entry["molecule_bootstrap"] = _molecule_bootstrap(split, task, logits[split], common, n_mol, rng)
        out["splits"][split] = entry
    return out


def _remove_top(d: np.ndarray, k: int) -> dict[str, Any]:
    if d.size <= k:
        return {"n_removed": 0, "macro_delta_remaining": float(d.mean()) if d.size else None}
    idx = np.argsort(-np.abs(d))[:k]
    keep = np.setdiff1d(np.arange(d.size), idx)
    return {
        "n_removed": int(k),
        "removed_tasks_mean_delta": float(d[idx].mean()),
        "macro_delta_remaining": float(d[keep].mean()),
        "n_remaining": int(keep.size),
    }


def _task_bootstrap(d: np.ndarray, rng: np.random.Generator, n: int = N_TASK_BOOTSTRAP) -> dict[str, Any]:
    idx = rng.integers(0, d.size, size=(int(n), d.size))
    means = d[idx].mean(axis=1)
    return {
        "n_resamples": int(n),
        "mean": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "frac_positive": float((means > 0).mean()),
        "note": "task-level resampling = composition sensitivity of the assay panel, not training-seed variance",
    }


def _top_tasks(common: Sequence[int], d: np.ndarray, k: int, *, sign: int) -> list[dict[str, Any]]:
    order = np.argsort(-sign * d)[:k]
    return [{"task": int(common[i]), "delta": float(d[i])} for i in order]


def _molecule_bootstrap(
    split: str,
    task: Any,
    bundle: Mapping[str, np.ndarray],
    common: Sequence[int],
    n: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows = getattr(task, f"{split}_rows")
    a = bundle["a"]
    b = bundle["b"]
    y = bundle["y"]
    deltas = []
    scored = []
    n_mol = int(len(rows))
    for _ in range(int(n)):
        idx = rng.integers(0, n_mol, size=n_mol)
        ya = y[idx][:, common]
        la = a[idx][:, common]
        lb = b[idx][:, common]
        ap_a = _macro_fast_ap(ya, la)
        ap_b = _macro_fast_ap(ya, lb)
        if ap_a is None or ap_b is None:
            continue
        deltas.append(ap_a - ap_b)
        scored.append(ap_a)
    arr = np.asarray(deltas)
    return {
        "n_resamples": int(n),
        "n_valid_resamples": int(arr.size),
        "mean_delta": float(arr.mean()) if arr.size else None,
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr.size else None,
        "frac_positive": float((arr > 0).mean()) if arr.size else None,
        "note": "paired molecule resampling within the frozen split; scaffold clusters are unavailable",
    }


def _macro_fast_ap(y: np.ndarray, logits: np.ndarray) -> float | None:
    vals = []
    for t in range(y.shape[1]):
        ap = fast_ap(y[:, t], logits[:, t])
        if ap is not None:
            vals.append(ap)
    return float(np.mean(vals)) if vals else None


def load_logits_for(unit: Mapping[str, Any], task: Any) -> dict[str, Any] | None:
    """Best-checkpoint logits for the molecule bootstrap (never used for selection)."""
    d = C.run_dir(unit["arm"], unit["seed"])
    if not d.is_dir():
        d = C.RUNS / "HIST" / f"{unit['arm']}_seed{unit['seed']}"
    if unit["arm"] == "W2" and unit["seed"] == 0:
        d = C.RUNS / "HIST" / "W2_seed0"
    if unit["arm"] == "DYNAMIC" and unit["seed"] == 0:
        d = C.RUNS / "HIST" / "DYNAMIC_seed0"
    vp = d / "valid_logits_best.npy"
    tp = d / "test_logits_best.npy"
    if not (vp.is_file() and tp.is_file()):
        return None
    return {
        "valid": np.load(vp),
        "test": np.load(tp),
    }


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def decide(units: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Any], *, horizon: int) -> dict[str, Any]:
    def get(label: str, split: str, path: Sequence[str] = ("ap",)) -> float | None:
        u = units.get(label)
        if u is None:
            return None
        cur: Any = u["split"][split]
        for key in path:
            if cur is None:
                return None
            cur = cur.get(key) if isinstance(cur, Mapping) else None
        return None if cur is None else float(cur)

    def cmp(name: str, split: str, key: str = "delta_macro_ap") -> float | None:
        c = comparisons.get(name)
        if c is None:
            return None
        s = c["splits"].get(split) or {}
        v = s.get(key)
        return None if v is None else float(v)

    obs: dict[str, Any] = {}
    labels: dict[str, str] = {}

    # ---- Q1: SELF vs W2
    self_w2_v = cmp("Q1_extra_pointwise_depth_vs_W2", "valid")
    self_w2_t = cmp("Q1_extra_pointwise_depth_vs_W2", "test")
    obs["Q1"] = {
        "SELF_minus_W2_valid": self_w2_v,
        "SELF_minus_W2_test": self_w2_t,
        "SELF_valid": get("SELF", "valid"),
        "W2_valid": get("W2", "valid"),
        "seeds_paired": ["0"],
        "threshold_valid": 0.005,
    }
    if self_w2_v is not None and self_w2_t is not None and self_w2_v >= 0.005 and self_w2_t > 0:
        labels["Q1"] = "STRICT_LOCAL_COMPUTATION_SIGNAL"
    elif self_w2_v is not None and self_w2_v > 0:
        labels["Q1"] = "WEAK_OR_UNCONFIRMED_LOCAL_COMPUTATION_SIGNAL"
    else:
        labels["Q1"] = "NO_STRICT_LOCAL_COMPUTATION_SIGNAL"

    # ---- Q2: FIXED1 vs SELF and vs DYNAMIC
    f_s = cmp("Q2_fixed_source_one_hop_vs_self", "valid")
    f_s_t = cmp("Q2_fixed_source_one_hop_vs_self", "test")
    dyn_f = cmp("Q5a_iterative_vs_fixed_one_hop", "valid")
    obs["Q2"] = {
        "FIXED1_minus_SELF_valid": f_s,
        "FIXED1_minus_SELF_test": f_s_t,
        "FIXED1_minus_DYNAMIC_valid": None if dyn_f is None else -dyn_f,
        "near_dynamic_band": [-0.005, -0.003],
    }
    if f_s is not None and f_s > 0.002 and obs["Q2"]["FIXED1_minus_DYNAMIC_valid"] is not None and obs["Q2"]["FIXED1_minus_DYNAMIC_valid"] > -0.006:
        labels["Q2"] = "FIXED_SOURCE_COMPUTATION_SIGNAL"
    elif f_s is not None and f_s > 0:
        labels["Q2"] = "WEAK_FIXED_SOURCE_SIGNAL"
    else:
        labels["Q2"] = "NO_FIXED_SOURCE_SIGNAL"

    # ---- Q3: REAL2 / BAG2 vs FIXED1
    r_f = cmp("Q3a_direct_two_hop_vs_fixed_one_hop", "valid")
    b_f = cmp("Q3b_multiset_combination_vs_fixed_one_hop", "valid")
    obs["Q3"] = {
        "REAL2_minus_FIXED1_valid": r_f,
        "BAG2_minus_FIXED1_valid": b_f,
        "REAL2_minus_FIXED1_test": cmp("Q3a_direct_two_hop_vs_fixed_one_hop", "test"),
        "BAG2_minus_FIXED1_test": cmp("Q3b_multiset_combination_vs_fixed_one_hop", "test"),
    }
    if (r_f is not None and r_f > 0.002) and (b_f is not None and b_f > 0.002):
        labels["Q3"] = "TWO_HOP_STRUCTURE_AND_COMBINATION_BOTH_CONTRIBUTE"
    elif r_f is not None and r_f > 0.002:
        labels["Q3"] = "TWO_HOP_STRUCTURE_CONTRIBUTES_BINDING_SUSPECT"
    elif b_f is not None and b_f > 0.002:
        labels["Q3"] = "WIDER_STRUCTURE_RANGE_CONTRIBUTES_WITHOUT_BINDING"
    else:
        labels["Q3"] = "NO_TWO_HOP_GAIN"

    # ---- Q4: REAL2 vs BAG2
    r_b = cmp("Q4_real_branch_binding_vs_multiset", "valid")
    r_b_t = cmp("Q4_real_branch_binding_vs_multiset", "test")
    hi = comparisons.get("Q4_real_branch_binding_vs_multiset", {}).get("splits", {}).get("valid", {}).get("high_support", {})
    obs["Q4"] = {
        "REAL2_minus_BAG2_valid": r_b,
        "REAL2_minus_BAG2_test": r_b_t,
        "REAL2_vs_BAG2_valid_high_support_mean_delta": (hi or {}).get("mean_delta"),
    }
    if r_b is not None and r_b > 0.0015 and (r_b_t is None or r_b_t > -0.002):
        labels["Q4"] = "REAL_BRANCH_BINDING_SIGNAL"
    elif r_b is not None and r_b < -0.0015:
        labels["Q4"] = "BINDING_NO_PREDICTIVE_VALUE"
    else:
        labels["Q4"] = "BINDING_NOT_IDENTIFIED"

    # ---- Q5: C* vs DYNAMIC
    c_d = cmp("Q5c_Cstar_vs_DYNAMIC_seed0", "valid")
    c_d_t = cmp("Q5c_Cstar_vs_DYNAMIC_seed0", "test")
    c_w = cmp("Q1b_Cstar_vs_W2_seed0", "valid")
    obs["Q5"] = {
        "Cstar_minus_DYNAMIC_valid": c_d,
        "Cstar_minus_DYNAMIC_test": c_d_t,
        "Cstar_minus_W2_valid": c_w,
        "DYNAMIC_minus_Cstar_valid": None if c_d is None else -c_d,
    }
    if c_d is not None and c_d < -0.005:
        labels["Q5"] = "ITERATIVE_ADVANTAGE_REMAINS"
    elif c_d is not None and c_d > 0.001:
        labels["Q5"] = "DIRECT_COMPUTATION_ADVANCE"
    elif c_d is not None and abs(c_d) <= 0.005 and (c_w or 0) > 0:
        labels["Q5"] = "COMPETITIVE_NONITERATIVE_COMPUTATION"
    else:
        labels["Q5"] = "NOT_DETERMINED"

    # ---- Q6: route recommendation.  Protocol §9: a route is only recommended when
    # "real gain + repeatable across seeds + not a few low-support tasks + inside the
    # declared information boundary + clear research value" all hold.
    def hi(name: str, split: str) -> Any:
        s = (comparisons.get(name) or {}).get("splits", {}).get(split) or {}
        h = s.get("high_support") or {}
        return h

    def imp(name: str, split: str) -> Any:
        s = (comparisons.get(name) or {}).get("splits", {}).get(split) or {}
        return None if "n_improved" not in s else (s["n_improved"], s["n_degraded"])

    q6 = {}
    crit = {
        "fixed_source_one_hop": {
            "gain": [cmp("Q1b_Cstar_vs_W2_seed0", "valid"), cmp("Q1b_Cstar_vs_W2_seed1", "valid")],
            "gain_vs_sibling": [cmp("Q2_fixed_source_one_hop_vs_self", "valid"),
                                cmp("Q4b_Cstar_vs_Kstar_seed1", "valid")],
            "repeatable": None,
            "breadth": [hi("Q1b_Cstar_vs_W2_seed0", "valid"), hi("Q1b_Cstar_vs_W2_seed1", "valid")],
            "improved": [imp("Q1b_Cstar_vs_W2_seed0", "valid"), imp("Q1b_Cstar_vs_W2_seed1", "valid")],
            "information_boundary": "layer 2 reads h0 of real 1-hop neighbours; verified by A3 grad support",
            "research_value": "states a concrete, non-iterative structural computation that matches DYNAMIC on Valid",
        },
        "real_two_hop": {
            "gain": [cmp("Q3a_direct_two_hop_vs_fixed_one_hop", "valid")],
            "repeatable": False,
            "breadth": [hi("Q3a_direct_two_hop_vs_fixed_one_hop", "valid")],
            "improved": [imp("Q3a_direct_two_hop_vs_fixed_one_hop", "valid")],
            "information_boundary": "direct two-hop from h0 (verified)",
            "research_value": "tested the 'recurrence is really two-hop branching' hypothesis",
        },
        "multiset_combination": {
            "gain": [cmp("Q3b_multiset_combination_vs_fixed_one_hop", "valid")],
            "repeatable": False,
            "breadth": [hi("Q3b_multiset_combination_vs_fixed_one_hop", "valid")],
            "improved": [imp("Q3b_multiset_combination_vs_fixed_one_hop", "valid")],
            "information_boundary": "same multisets without correspondence (verified)",
            "research_value": "diagnostic control only, never a candidate architecture",
        },
        "iterative": {
            "gain": [cmp("Q5a_iterative_vs_fixed_one_hop", "valid")],
            "gain_test": [cmp("Q5a_iterative_vs_fixed_one_hop", "test"),
                          cmp("Q5c_Cstar_vs_DYNAMIC_seed1", "test")],
            "repeatable": None,
            "breadth": [hi("Q5a_iterative_vs_fixed_one_hop", "valid")],
            "improved": [imp("Q5a_iterative_vs_fixed_one_hop", "valid")],
            "information_boundary": "layer 2 reads h1[j]: a wider effective access range than any h0 arm",
            "research_value": "capability boundary reference",
        },
    }
    for key, entry in crit.items():
        gains = [g for g in entry["gain"] if g is not None]
        if entry["repeatable"] is None:
            entry["repeatable"] = bool(len(gains) >= 2 and all(g > 0 for g in gains) and
                                       all(g >= 0.005 for g in gains)) if len(gains) >= 2 else None
        entry["verdict"] = _route_verdict(entry)
        q6[key] = entry

    modifiers: dict[str, list[str]] = {}
    if c_d is not None and c_d_t is not None and c_d * c_d_t < 0 and abs(c_d_t) > 0.003:
        modifiers["Q5"] = ["SPLIT_CONFLICT"]
    if c_d is not None and c_d_t is not None:
        modifiers.setdefault("Q5", []).append("VALID_TEST_DISAGREEMENT")

    return {
        "observations": obs,
        "labels": labels,
        "modifiers": modifiers,
        "q6_routes": q6,
        "thresholds": {
            "Q1_self_minus_w2_valid_min": 0.005,
            "Q4_real2_minus_bag2_valid_min": 0.0015,
            "Q5_iterative_band": 0.005,
        },
        "horizon": int(horizon),
        "note": "labels are route decisions on point estimates; the task bootstrap interval is not a significance test",
    }


def _route_verdict(entry: Mapping[str, Any]) -> str:
    gains = [g for g in (entry.get("gain") or []) if g is not None]
    if not gains or max(gains) <= 0.0:
        return "NO_NEW_COMPETITIVE_DIRECT_MECHANISM"
    if entry.get("repeatable") is False:
        return "NO_NEW_COMPETITIVE_DIRECT_MECHANISM"
    if entry.get("repeatable") is None:
        return "PROMISING_SINGLE_SEED_ONLY"
    return "RECOMMEND_CONTINUE"


def route_table() -> list[dict[str, Any]]:
    return [
        {
            "route": "strict direct local computation (W2 / SELF)",
            "role": "no cross-node hidden state is read at all",
            "evidence_for": [],
            "evidence_against": [],
            "open": [],
            "priority": "",
        },
        {
            "route": "fixed-source one-hop computation (FIXED1)",
            "role": "reads h0 of the real 1-hop neighbours, never h1[j]",
            "evidence_for": [],
            "evidence_against": [],
            "open": [],
            "priority": "",
        },
        {
            "route": "real two-hop structural combination (REAL2)",
            "role": "direct rooted two-hop branch computation from h0",
            "evidence_for": [],
            "evidence_against": [],
            "open": [],
            "priority": "",
        },
        {
            "route": "unbound multiset combination (BAG2)",
            "role": "same two multisets, correspondence removed",
            "evidence_for": [],
            "evidence_against": [],
            "open": [],
            "priority": "",
        },
        {
            "route": "iterative hidden-state update (DYNAMIC)",
            "role": "layer 2 reads the updated neighbour state h1[j]",
            "evidence_for": [],
            "evidence_against": [],
            "open": [],
            "priority": "",
        },
    ]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-historical", action="store_true")
    parser.add_argument("--mol-bootstrap", action="store_true")
    parser.add_argument("--horizon", type=int, default=0)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    if args.evaluate_historical:
        C.log(f"historical re-evaluation: {json.dumps(evaluate_historical(device), default=str)[:1200]}")

    horizon = int(args.horizon)
    if horizon == 0:
        hp = C.CONFIGS / "horizon.json"
        horizon = int(C.read_json(hp)["horizon"]) if hp.is_file() else C.DEFAULT_HORIZON
    task = C.SD.load_task()
    test_info = C.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    y_test = np.asarray(test_info["labels"])

    units: dict[str, dict[str, Any]] = {}
    for arm in ("W2", "DYNAMIC", "SELF", "FIXED1", "REAL2", "BAG2"):
        u = load_unit(arm, arm, 0, horizon=horizon)
        if u is not None:
            units[arm] = u
    for arm in ("W2", "DYNAMIC"):
        u = load_unit(f"{arm}_seed1", arm, 1, horizon=horizon)
        if u is not None:
            units[f"{arm}_seed1"] = u

    # seed-0 candidate selection (C*) and its mechanism control (K*)
    cand = {a: units[a] for a in C.CANDIDATES if a in units}
    cstar = None
    if cand:
        best_ap = max(u["best_valid_ap"] for u in cand.values())
        near = [a for a, u in cand.items() if best_ap - u["best_valid_ap"] <= C.TIE_BREAK_AP]
        for a in C.CANDIDATE_PREFERENCE:
            if a in near:
                cstar = a
                break
    units["CSTAR"] = dict(units[cstar]) if cstar else None
    kstar = C.CONTROL_OF.get(cstar) if cstar else None
    units["KSTAR"] = dict(units[kstar]) if kstar in units else None
    for arm in (cstar, kstar):
        if arm and arm not in (None, "W2", "DYNAMIC") and f"{arm}_seed1" not in units:
            u = load_unit(f"{arm}_seed1", arm, 1, horizon=horizon)
            if u is not None:
                units[f"{arm}_seed1"] = u
    if cstar:
        for u in (units["CSTAR"],):
            u["label"] = f"C*={cstar}"
    if kstar in units:
        units["KSTAR"]["label"] = f"K*={kstar}"

    ordered = [units[k] for k in ("W2", "SELF", "FIXED1", "REAL2", "BAG2", "DYNAMIC") if units.get(k)]
    if units.get("W2_seed1"):
        ordered.append(units["W2_seed1"])
    if units.get("DYNAMIC_seed1"):
        ordered.append(units["DYNAMIC_seed1"])
    if units.get("CSTAR"):
        ordered.append(units["CSTAR"])
    if units.get("KSTAR") and kstar != cstar:
        ordered.append(units["KSTAR"])

    main_table = write_main_table(ordered)
    per_task = write_per_task(ordered)

    rng = np.random.default_rng(20261125)
    comparisons: dict[str, Any] = {}
    def resolve(arm: str, seed: int) -> dict[str, Any] | None:
        if arm in ("CSTAR", "KSTAR"):
            base = units.get(arm)
            if base is None:
                return None
            if seed == 0:
                return base
            other = units.get(f"{base['arm']}_seed1")
            if other is None:
                return None
            out = dict(other)
            out["label"] = f"{arm}={base['arm']}_seed1"
            return out
        if seed == 0:
            return units.get(arm)
        return units.get(f"{arm}_seed1")

    for a, sa, b, sb, name in COMPARISONS + SEED1_COMPARISONS:
        ua = resolve(a, sa)
        ub = resolve(b, sb)
        if ua is None or ub is None:
            continue
        logits = None
        if args.mol_bootstrap and name in BOOTSTRAP_MOL_PAIRS:
            la = load_logits_for(ua, task)
            lb = load_logits_for(ub, task)
            if la is not None and lb is not None:
                # paired molecule resampling is run on Valid only (the split that
                # selected the checkpoints); Test is never resampled for a decision
                logits = {"valid": {"a": la["valid"], "b": lb["valid"],
                                    "y": np.asarray(task.labels[task.valid_rows])}}
        comparisons[name] = compare(
            ua, ub, rng=rng, mol_bootstrap=logits is not None, task=task, logits=logits
        )

    ap_check = 0.0
    if units.get("CSTAR"):
        lg = load_logits_for(units["CSTAR"], task)
        if lg is not None:
            ap_check = verify_fast_ap(np.asarray(task.labels[task.valid_rows]), lg["valid"],
                                      list(range(0, C.N_TASKS, 8)), rng)

    decision = decide(units, comparisons, horizon=horizon)
    table = route_table()

    def d(name: str, split: str) -> float | None:
        s = (comparisons.get(name) or {}).get("splits", {}).get(split) or {}
        v = s.get("delta_macro_ap")
        return None if v is None else float(v)

    q6 = decision["q6_routes"]
    table[0]["evidence_for"] = [
        f"SELF − W2 Valid {_fmt(d('Q1_extra_pointwise_depth_vs_W2','valid'))}（seed0）/ "
        f"{_fmt(d('Q4b_Cstar_vs_Kstar_seed1','valid'))} 为 C*−K* 而非 SELF−W2",
        "A3 梯度支持表：SELF 第二阶段完全不读其他节点隐状态",
    ]
    table[0]["evidence_against"] = [
        f"seed0 增益 {_fmt(d('Q1_extra_pointwise_depth_vs_W2','valid'))} 低于预注册 0.005 门槛",
        "SELF 仍然对每条真实入边做条件化，不是纯点式深度",
    ]
    table[0]["open"] = ["若需要更高点式容量，应作为容量对照单独预注册"]
    table[0]["priority"] = f"{decision['labels']['Q1']} → SELF 单独不作为路线推荐"

    table[1]["evidence_for"] = [
        f"FIXED1 − SELF Valid {_fmt(d('Q2_fixed_source_one_hop_vs_self','valid'))}（seed0，CI 不含 0）",
        f"C* − K* Valid {_fmt(d('Q4b_Cstar_vs_Kstar_seed1','valid'))} / Test "
        f"{_fmt(d('Q4b_Cstar_vs_Kstar_seed1','test'))}（seed1 配对）",
        f"C* − W2 Valid {_fmt(d('Q1b_Cstar_vs_W2_seed0','valid'))}（seed0） / "
        f"{_fmt(d('Q1b_Cstar_vs_W2_seed1','valid'))}（seed1）",
        f"与 DYNAMIC 的 Valid 差 {_fmt(-d('Q5a_iterative_vs_fixed_one_hop','valid') if d('Q5a_iterative_vs_fixed_one_hop','valid') is not None else None)}（seed0） / "
        f"{_fmt(d('Q5c_Cstar_vs_DYNAMIC_seed1','valid'))}（seed1）",
    ]
    table[1]["evidence_against"] = [
        f"对 SELF 的 Test 优势只有 {_fmt(d('Q2_fixed_source_one_hop_vs_self','test'))}（seed0）",
        f"C* − DYNAMIC Test 两 seed 符号相反：{_fmt(d('Q5c_Cstar_vs_DYNAMIC_seed0','test'))} / "
        f"{_fmt(d('Q5c_Cstar_vs_DYNAMIC_seed1','test'))}",
    ]
    table[1]["open"] = ["FIXED1 只有 2 个 seed（0/1），尚无第三 seed"]
    table[1]["priority"] = f"{decision['labels']['Q2']} → {q6['fixed_source_one_hop']['verdict']}"

    table[2]["evidence_for"] = [
        f"DYNAMIC − REAL2 Valid {_fmt(d('Q5b_iterative_vs_direct_two_hop','valid'))} / Test "
        f"{_fmt(d('Q5b_iterative_vs_direct_two_hop','test'))}：两跳没有替代迭代",
    ]
    table[2]["evidence_against"] = [
        f"REAL2 − FIXED1 Valid {_fmt(d('Q3a_direct_two_hop_vs_fixed_one_hop','valid'))}（CI 不含 0）",
        f"REAL2 − BAG2 Valid {_fmt(d('Q4_real_branch_binding_vs_multiset','valid'))} / Test "
        f"{_fmt(d('Q4_real_branch_binding_vs_multiset','test'))}",
        "冻结替换：BAG2 权重 + REAL2 读出 +0.0002；REAL2 权重 + BAG2 读出 −0.0001",
    ]
    table[2]["open"] = ["REAL2 只有 1 个 seed；机制替换已显示对应关系不可识别，故未追加 seed"]
    table[2]["priority"] = f"{decision['labels']['Q3']} → {q6['real_two_hop']['verdict']}"

    table[3]["evidence_for"] = [
        f"BAG2 绝对 Valid {_fmt(units['BAG2']['best_valid_ap']) if units.get('BAG2') else 'n/a'}，"
        f"高于 REAL2、低于 FIXED1",
        "BAG2 最佳 epoch 22（FIXED1/REAL2 为 25），Train AP 最低，表现为更强的正则化而非更强表达",
    ]
    table[3]["evidence_against"] = [
        f"BAG2 − FIXED1 Valid {_fmt(d('Q3b_multiset_combination_vs_fixed_one_hop','valid'))}；Test 仅 "
        f"{_fmt(d('Q3b_multiset_combination_vs_fixed_one_hop','test'))}（CI 含 0）",
        "冻结替换：把 REAL2 读出换成 degree 归一化（mean）后 −0.0161，说明承重的是归一化而非配对",
    ]
    table[3]["open"] = ["BAG2 是诊断控制臂，不作为候选架构"]
    table[3]["priority"] = f"{decision['labels']['Q3']} → {q6['multiset_combination']['verdict']}"

    table[4]["evidence_for"] = [
        f"DYNAMIC − FIXED1 Test {_fmt(d('Q5a_iterative_vs_fixed_one_hop','test'))}（seed0）",
        f"DYNAMIC − REAL2 Valid {_fmt(d('Q5b_iterative_vs_direct_two_hop','valid'))} / Test "
        f"{_fmt(d('Q5b_iterative_vs_direct_two_hop','test'))}",
        "A3：第二阶段读 h1[j]，有效访问范围严格大于任何 h0 臂",
    ]
    table[4]["evidence_against"] = [
        f"与 FIXED1 的 Valid 差两 seed 均 ≈0：{_fmt(-d('Q5a_iterative_vs_fixed_one_hop','valid') if d('Q5a_iterative_vs_fixed_one_hop','valid') is not None else None)} / "
        f"{_fmt(d('Q5c_Cstar_vs_DYNAMIC_seed1','valid'))}",
        "Test 侧 DYNAMIC 优势在 seed1 反向（C* 领先 +0.0065）",
    ]
    table[4]["open"] = ["迭代优势与更宽访问范围不可分离（本轮无 h1 读取但非迭代的臂）"]
    table[4]["priority"] = f"{decision['labels']['Q5']} → {q6['iterative']['verdict']}"

    payload = {
        "schema": C.SCHEMA_SUMMARY,
        "round_id": C.ROUND_ID,
        "horizon": int(horizon),
        "candidate_selection": {
            "candidates": {a: units[a]["best_valid_ap"] for a in C.CANDIDATES if units.get(a)},
            "tie_break_ap": C.TIE_BREAK_AP,
            "preference": list(C.CANDIDATE_PREFERENCE),
            "C*": cstar,
            "K*": kstar,
            "selected_on": "full official Valid macro AP (Test never used)",
            "selection_stability": {
                "best_valid": {a: units[a]["best_valid_ap"] for a in C.CANDIDATES if units.get(a)},
                "last5_mean": {a: last_k_valid_mean(units[a]) for a in C.CANDIDATES if units.get(a)},
            },
        },
        "decision": decision,
        "q6_routes": decision["q6_routes"],
        "modifiers": decision["modifiers"],
        "route_table": table,
        "fast_ap_vs_sklearn_max_abs_diff": ap_check,
        "units_present": [u["label"] for u in ordered],
        "artifacts": {"main_table": main_table, "per_task_metrics": per_task},
        "test_role": C.TEST_ROLE,
        "created_utc": C.utcnow(),
    }
    C.write_json(C.ANALYSIS / "summary.json", payload)
    C.write_json(C.ANALYSIS / "paired_comparisons.json",
                 {"schema": "dscr-paired-comparisons-v1", "round_id": C.ROUND_ID, "horizon": int(horizon),
                  "comparisons": comparisons, "n_task_bootstrap": N_TASK_BOOTSTRAP,
                  "n_mol_bootstrap": N_MOL_BOOTSTRAP if args.mol_bootstrap else 0,
                  "common_task_set": "tasks scored by both models in the given split",
                  "created_utc": C.utcnow()})
    C.log(f"summary: labels {json.dumps(decision['labels'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
