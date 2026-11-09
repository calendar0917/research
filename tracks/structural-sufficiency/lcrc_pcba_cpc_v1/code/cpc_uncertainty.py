"""Post-freeze uncertainty summaries for the one sealed CPC comparison.

This script reads only already-frozen Test logits and the official Test labels
opened through ``cc_data.read_official_test_labels``.  It never changes the
model, checkpoint, selection, threshold or decision label.

Two complementary uncertainties are reported:

* molecule bootstrap: resample the 43,793 official Test molecules with
  replacement and recompute the macro AP of each frozen prediction;
* task bootstrap: resample the scored tasks with replacement and recompute the
  mean per-task AP or the paired CPC-minus-reference difference.

Both are descriptive engineering intervals on an exposed Test set, not blind
generalisation estimates and not significance tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import cpc_common as C  # noqa: E402
import cc_data as CD  # noqa: E402


def _abs_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else C.REPO / path


def _macro_ap(y: np.ndarray, logits: np.ndarray) -> float | None:
    from sklearn.metrics import average_precision_score

    values: list[float] = []
    for t in range(y.shape[1]):
        mask = np.isfinite(y[:, t])
        yt = y[mask, t]
        if not (np.any(yt == 1.0) and np.any(yt == 0.0)):
            continue
        values.append(float(average_precision_score(yt, logits[mask, t])))
    return float(np.mean(values)) if values else None


def _task_arrays(decision: Mapping[str, Any]) -> dict[str, list[Any]]:
    rows = decision["per_task"]
    return {
        "cpc_valid": [r["cpc_best_valid_ap"] for r in rows],
        "w2_valid": [r["w2_best_valid_ap"] for r in rows],
        "dynamic_valid": [r["dynamic_best_valid_ap"] for r in rows],
        "cpc_test": [r["cpc_best_test_ap"] for r in rows],
        "w2_test": [r["w2_best_test_ap"] for r in rows],
        "dynamic_test": [r["dynamic_best_test_ap"] for r in rows],
    }


def _mean(values: Sequence[Any]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _task_bootstrap(
    arrays: Mapping[str, Sequence[Any]],
    ids: Sequence[int],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    rows = {}
    for name, values in arrays.items():
        v = np.asarray([values[i] if values[i] is not None else np.nan for i in ids], dtype=np.float64)
        rows[name] = v
    out: dict[str, Any] = {"n_tasks": int(len(ids)), "reps": int(reps), "intervals": {}}
    for a, b in (
        ("cpc_valid", "w2_valid"),
        ("cpc_valid", "dynamic_valid"),
        ("cpc_test", "w2_test"),
        ("cpc_test", "dynamic_test"),
    ):
        va, vb = rows[a], rows[b]
        common = np.isfinite(va) & np.isfinite(vb)
        if not common.any():
            continue
        va, vb = va[common], vb[common]
        idx = rng.integers(0, len(va), size=(int(reps), len(va)))
        delta = va[idx].mean(axis=1) - vb[idx].mean(axis=1)
        out["intervals"][f"{a}_minus_{b}"] = {
            "mean_ap_a": float(va.mean()),
            "mean_ap_b": float(vb.mean()),
            "delta": float(va.mean() - vb.mean()),
            "ci95_low": float(np.percentile(delta, 2.5)),
            "ci95_high": float(np.percentile(delta, 97.5)),
            "fraction_positive": float((delta > 0.0).mean()),
            "n_tasks": int(common.sum()),
        }
    return out


def _row_bootstrap(
    y: np.ndarray,
    logits: Mapping[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(y.shape[0])
    names = sorted(logits)
    draws = {name: np.empty(int(reps), dtype=np.float64) for name in names}
    usable = np.zeros(int(reps), dtype=bool)
    for b in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        base = _macro_ap(y[idx], logits["W2_best_valid"][idx])
        if base is None:
            continue
        usable[b] = True
        for name in names:
            value = _macro_ap(y[idx], logits[name][idx])
            draws[name][b] = np.nan if value is None else value
    out: dict[str, Any] = {
        "n_rows": n,
        "reps": int(reps),
        "usable_reps": int(usable.sum()),
        "intervals": {},
    }
    for name in names:
        if name == "W2_best_valid":
            continue
        d = draws[name][usable] - draws["W2_best_valid"][usable]
        d = d[np.isfinite(d)]
        out["intervals"][f"{name}_minus_W2_best_valid"] = {
            "mean_ap_a": float(draws[name][usable].mean()),
            "mean_ap_w2": float(draws["W2_best_valid"][usable].mean()),
            "delta": float(draws[name][usable].mean() - draws["W2_best_valid"][usable].mean()),
            "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "fraction_positive": float((d > 0.0).mean()),
            "usable_reps": int(d.size),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default=str(C.ANALYSIS / "test_prediction_manifest.json"))
    parser.add_argument("--decision", default=str(C.ANALYSIS / "decision.json"))
    parser.add_argument("--scores", default=str(C.ANALYSIS / "test_scores.json"))
    parser.add_argument("--out", default=str(C.ANALYSIS / "uncertainty.json"))
    parser.add_argument("--row-reps", type=int, default=300)
    parser.add_argument("--task-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    freeze = C.read_json(args.freeze)
    if freeze.get("schema") != C.SCHEMA_TEST_FREEZE:
        raise RuntimeError("freeze has the wrong schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("freeze does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != C.freeze_seal(freeze):
        raise RuntimeError("freeze seal does not verify")
    scores = C.read_json(args.scores)
    if scores.get("freeze_seal_sha256") != freeze["seal_sha256"]:
        raise RuntimeError("scores do not reference this freeze")
    decision = C.read_json(args.decision)

    task = C.SD.load_task()
    test_ref = CD.read_official_test_labels(task, freeze_path=args.freeze)
    y = np.asarray(test_ref["labels"], dtype=np.float64)

    logits: dict[str, np.ndarray] = {}
    for label, record in freeze["predictions"].items():
        path = _abs_path(record["test_logits_path"])
        if C.sha256_file(path) != record["test_logits_sha256"]:
            raise RuntimeError(f"prediction hash changed: {path}")
        logits[label] = np.load(path)
    for label in ("W2_best_valid", "W2_epoch40", "W2_DYNAMIC_best_valid", "W2_DYNAMIC_epoch40"):
        record = C.verify_frozen_reference_logits()[label]
        logits[label] = np.load(_abs_path(record["path"]))

    arrays = _task_arrays(decision)
    high = [int(x) for x in decision["high_support_task_ids"]]
    low = [int(x) for x in decision["low_support_task_ids"]]
    all_ids = [int(x) for x in range(C.N_TASKS)]

    out = {
        "schema": "lcrc-pcba-cpc-uncertainty-v1",
        "round_id": C.ROUND_ID,
        "test_role": C.TEST_ROLE,
        "test_labels_read_after_freeze": True,
        "freeze_seal_sha256": freeze["seal_sha256"],
        "decision": decision["label"],
        "methods": {
            "molecule_bootstrap": "resample official Test molecules with replacement; recompute macro AP per replicate",
            "task_bootstrap": "resample scored tasks with replacement; recompute mean AP or paired difference",
            "interval": "2.5th and 97.5th percentiles; descriptive only",
            "not_significance_test": True,
        },
        "task_bootstrap": {
            "all_tasks": _task_bootstrap(arrays, all_ids, reps=int(args.task_reps), seed=int(args.seed)),
            "high_support": _task_bootstrap(arrays, high, reps=int(args.task_reps), seed=int(args.seed) + 1),
            "low_support": _task_bootstrap(arrays, low, reps=int(args.task_reps), seed=int(args.seed) + 2),
        },
        "molecule_bootstrap": _row_bootstrap(
            y,
            {
                "LCRC_CPC_best_valid": logits["LCRC_CPC_best_valid"],
                "LCRC_CPC_epoch40": logits["LCRC_CPC_epoch40"],
                "W2_best_valid": logits["W2_best_valid"],
                "W2_epoch40": logits["W2_epoch40"],
                "W2_DYNAMIC_best_valid": logits["W2_DYNAMIC_best_valid"],
                "W2_DYNAMIC_epoch40": logits["W2_DYNAMIC_epoch40"],
            },
            reps=int(args.row_reps),
            seed=int(args.seed) + 3,
        ),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
    }
    C.write_json(Path(args.out), out)
    C.log(
        f"[uncertainty] wrote {args.out} "
        f"row_reps={args.row_reps} task_reps={args.task_reps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
