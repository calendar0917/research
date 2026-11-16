"""Aggregate stage: main table, controls, paired uncertainty, decision axes, REPORT.

Everything here is computed from the per-dataset artefacts written by the two
GPU workers; no model is retrained.  The three axes of the round brief are
evaluated separately and never collapsed into a single success/failure score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_common as C
from sq_common import Ledger, mae, rmse

BOOTSTRAP_ROUNDS = 2000
METHOD_ORDER = ("F", "R", "A", "E", "G", "L")
METHOD_TITLES = {
    "F": "F fixed base bank B + XGB",
    "R": "R random complexity-matched queries + XGB",
    "A": "A discovered queries + XGB",
    "E": "E ECFP4 + descriptors + XGB",
    "G": "G GINE (neural reference)",
    "L": "L C1 / NO_BOND_MEAN (repo reference)",
}
SPLITS = ("train", "valid", "test")
COMPETITIVE_TOL = 0.05
SIGNAL_GAIN = 0.03


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return C.read_json(path)


def _pred_path(directory: Path, method: str, preset: str | None) -> Path:
    if method in ("F", "R", "A", "E"):
        return directory / f"method_{method}_{preset}_pred.npz"
    return directory / f"method_{method}_pred.npz"


def _load_pred(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    return {
        split: {
            "pred_raw": data[f"{split}_pred_raw"],
            "y_raw": data[f"{split}_y_raw"],
            "row_id": data[f"{split}_row_id"],
        }
        for split in SPLITS
    }


def collect_dataset(dataset: str) -> dict[str, Any]:
    directory = C.RESULTS_DIR / dataset
    if not (directory / "methods_fra.json").is_file():
        raise FileNotFoundError(f"{dataset}: no methods_fra.json under {directory}; worker not finished")
    fra = _read_json(directory / "methods_fra.json")
    gine = _read_json(directory / "method_G.json")
    lcrc = _read_json(directory / "method_L.json")
    identity = _read_json(directory / "split_identity.json")
    audit = _read_json(directory / "data_audit.json")
    discovery = _read_json(directory / "discovery.json")
    random_payload = _read_json(directory / "random_library.json")
    base_bank = _read_json(directory / "base_bank.json")
    replay = _read_json(directory / "replay_audit.json")
    worker = _read_json(directory / "worker_summary.json")

    scorers: dict[str, dict[str, Any]] = {}
    for method in ("F", "R", "A", "E"):
        entry = fra[method]
        preset = entry["selected_preset"]
        predictions = _load_pred(_pred_path(directory, method, preset))
        scorers[method] = {
            "method": method,
            "selected_label": f"{preset}@{entry['selected']['iteration']}",
            "selected_preset": preset,
            "selected_iteration": int(entry["selected"]["iteration"]),
            "declared": entry["selected"],
            "predictions": predictions,
            "n_features": int(entry["per_preset"][preset]["n_features"]),
            "n_trees": int(entry["per_preset"][preset]["n_trees"]),
            "fit_seconds": float(entry["fit_seconds_total"]),
            "complete": True,
            "last20": entry["last20_selected"],
            "final_round": entry["final_round_selected"],
        }
    # G
    g_predictions = _load_pred(_pred_path(directory, "G", None))
    scorers["G"] = {
        "method": "G",
        "selected_label": f"peak_lr={gine['selected_peak_lr']}@{gine['selected']['epoch']}",
        "selected_preset": f"lr={gine['selected_peak_lr']}",
        "selected_iteration": int(gine["selected"]["epoch"]),
        "declared": gine["selected"],
        "predictions": g_predictions,
        "n_parameters": int(gine["n_parameters"]),
        "fit_seconds": float(gine["seconds_total"]),
        "complete": True,
        "last20": gine["last20_selected"],
        "final_round": gine["final_round_selected"],
    }
    # L
    if lcrc.get("status", "complete") == "complete" and (directory / "method_L_pred.npz").is_file():
        l_predictions = _load_pred(_pred_path(directory, "L", None))
        selected = lcrc["selected"]
        scorers["L"] = {
            "method": "L",
            "selected_label": f"R3@{selected['epoch']}",
            "selected_preset": "R3",
            "selected_iteration": int(selected["epoch"]),
            "declared": {
                "epoch": int(selected["epoch"]),
                "train_rmse_raw": selected["train_rmse_raw"],
                "valid_rmse_raw": selected["valid_rmse_raw"],
                "test_rmse_raw": selected["test_rmse_raw"],
                "train_mae_raw": selected["train_mae_raw"],
                "valid_mae_raw": selected["valid_mae_raw"],
                "test_mae_raw": selected["test_mae_raw"],
            },
            "predictions": l_predictions,
            "n_parameters": int(lcrc["n_parameters"]),
            "fit_seconds": float(lcrc["seconds"]),
            "complete": True,
            "last20": lcrc["last20"],
            "final_round": lcrc["final"],
        }
    else:
        scorers["L"] = {
            "method": "L",
            "selected_label": "UNAVAILABLE",
            "complete": False,
            "status": lcrc.get("status"),
            "error": lcrc.get("error"),
            "predictions": None,
        }

    # recomputed metrics (raw units) for every complete method
    for method, entry in scorers.items():
        if not entry["complete"]:
            continue
        metrics: dict[str, Any] = {}
        for split in SPLITS:
            pred = entry["predictions"][split]["pred_raw"]
            target = entry["predictions"][split]["y_raw"]
            official = D_metric(dataset, pred, target)
            metrics[split] = {
                "rmse": official[f"raw_rmse"],
                "mae": official["raw_mae"],
                "n": int(pred.size),
            }
        entry["metrics"] = metrics
    # B* reference: best of E/G/L by Val RMSE
    candidates = [scorers[m] for m in ("E", "G", "L") if scorers[m]["complete"]]
    best_ref = min(candidates, key=lambda item: (item["metrics"]["valid"]["rmse"], METHOD_ORDER.index(item["method"])))
    return {
        "dataset": dataset,
        "identity": identity,
        "audit": audit,
        "discovery": discovery,
        "random_library": random_payload,
        "base_bank": base_bank,
        "replay": replay,
        "worker": worker,
        "scorers": scorers,
        "best_reference": best_ref["method"],
    }


def D_metric(dataset: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    import sq_data as D

    return D.official_metric(dataset, pred, target)


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


def paired_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired molecule bootstrap of the RMSE difference ``RMSE(a) - RMSE(b)``."""
    rng = np.random.default_rng(seed)
    n = int(target.size)
    err_a = (np.asarray(pred_a, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    err_b = (np.asarray(pred_b, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    point = math.sqrt(err_a.mean()) - math.sqrt(err_b.mean())
    deltas = np.empty(rounds, dtype=np.float64)
    for index in range(rounds):
        sample = rng.integers(0, n, size=n)
        deltas[index] = math.sqrt(err_a[sample].mean()) - math.sqrt(err_b[sample].mean())
    return {
        "delta_rmse_a_minus_b": float(point),
        "bootstrap_mean": float(deltas.mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "p_delta_ge_0": float((deltas >= 0).mean()),
        "rounds": int(rounds),
        "seed": int(seed),
        "n_molecules": n,
    }


def drop_one_sensitivity(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Remove the single worst reference error (fixed by the reference model)."""
    err_b = np.abs(np.asarray(pred_b, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    worst = int(np.argmax(err_b))
    keep = np.ones(target.size, dtype=bool)
    keep[worst] = False
    full = rmse(pred_a, target) - rmse(pred_b, target)
    dropped = rmse(pred_a[keep], target[keep]) - rmse(pred_b[keep], target[keep])
    return {
        "dropped_index": worst,
        "delta_rmse_full": float(full),
        "delta_rmse_drop1": float(dropped),
        "sign_flip": bool(np.sign(full) != np.sign(dropped) and full != 0.0),
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def aggregate(config: Mapping[str, Any], ledger: Ledger, *, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        dataset for dataset in C.DATASETS if (output_dir / dataset / "methods_fra.json").is_file()
    ]
    if not datasets:
        raise FileNotFoundError(f"no completed dataset worker under {output_dir}")
    collected = {dataset: collect_dataset(dataset) for dataset in datasets}

    main_rows: list[dict[str, Any]] = []
    for dataset, payload in collected.items():
        worker = payload.get("worker") or {}
        n_base = int(worker.get("n_queries_B") or 0)
        n_extra = {
            "F": 0,
            "R": int((payload.get("random_library") or {}).get("n_filled") or 0),
            "A": int(worker.get("n_discovered") or 0),
            "E": None,
            "G": 0,
            "L": 0,
        }
        for method in METHOD_ORDER:
            entry = payload["scorers"][method]
            if not entry["complete"]:
                main_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "title": METHOD_TITLES[method],
                        "complete": False,
                        "status": entry.get("status"),
                        "error": entry.get("error"),
                    }
                )
                continue
            metrics = entry["metrics"]
            main_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "title": METHOD_TITLES[method],
                    "selected_state": entry["selected_label"],
                    "n_features": entry.get("n_features"),
                    "n_queries": (
                        None if n_extra.get(method) is None else n_base + int(n_extra[method])
                    ) if method in ("F", "R", "A") else None,
                    "n_trees": entry.get("n_trees"),
                    "n_parameters": entry.get("n_parameters"),
                    "train_rmse": metrics["train"]["rmse"],
                    "valid_rmse": metrics["valid"]["rmse"],
                    "test_rmse": metrics["test"]["rmse"],
                    "train_mae": metrics["train"]["mae"],
                    "valid_mae": metrics["valid"]["mae"],
                    "test_mae": metrics["test"]["mae"],
                    "last20_valid_rmse": entry.get("last20", {}).get("mean_last20_valid_rmse_raw"),
                    "final_round_valid_rmse": (entry.get("final_round") or {}).get("valid_rmse_raw"),
                    "fit_seconds": entry.get("fit_seconds"),
                    "complete": True,
                }
            )
    C.write_csv(output_dir / "tables" / "main_table.csv", main_rows)

    # controls + references with paired uncertainty
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_payload: dict[str, Any] = {}
    for dataset, payload in collected.items():
        scorers = payload["scorers"]
        if not scorers["A"]["complete"]:
            continue
        refs = ["F", "R", payload["best_reference"]]
        for ref in dict.fromkeys(refs):
            if not scorers[ref]["complete"]:
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"A - {ref}",
                        "complete": False,
                        "reason": "reference incomplete",
                    }
                )
                continue
            for split in ("valid", "test"):
                pred_a = scorers["A"]["predictions"][split]["pred_raw"]
                pred_b = scorers[ref]["predictions"][split]["pred_raw"]
                target = scorers["A"]["predictions"][split]["y_raw"]
                boot = paired_bootstrap(pred_a, pred_b, target)
                drop = drop_one_sensitivity(pred_a, pred_b, target)
                rel = boot["delta_rmse_a_minus_b"] / rmse(pred_b, target)
                bootstrap_payload[f"{dataset}|A-{ref}|{split}"] = {
                    "bootstrap": boot,
                    "drop1": drop,
                    "relative_delta": float(rel),
                    "reference_rmse": float(rmse(pred_b, target)),
                }
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"A - {ref}",
                        "split": split,
                        "rmse_A": rmse(pred_a, target),
                        "rmse_ref": rmse(pred_b, target),
                        "delta_rmse": boot["delta_rmse_a_minus_b"],
                        "relative_delta": float(rel),
                        "ci95_low": boot["ci95"][0],
                        "ci95_high": boot["ci95"][1],
                        "p_delta_ge_0": boot["p_delta_ge_0"],
                        "drop1_delta": drop["delta_rmse_drop1"],
                        "drop1_sign_flip": drop["sign_flip"],
                        "complete": True,
                    }
                )
    C.write_csv(output_dir / "tables" / "control_table.csv", comparison_rows)
    C.write_json(output_dir / "tables" / "bootstrap.json", bootstrap_payload)

    # curves
    curve_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for path in sorted((output_dir / dataset).glob("curves_*.csv")):
            import csv

            with path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    curve_rows.append({"dataset": dataset, **row})
    C.write_csv(output_dir / "tables" / "curves_all.csv", curve_rows)

    decision = _decision(collected, bootstrap_payload)
    C.write_json(output_dir / "decision.json", decision)
    ledger_payload = _merge_ledgers(ledger, collected)
    C.write_json(output_dir / "ledger_all.json", ledger_payload)
    _write_markdown(
        output_dir, collected, main_rows, comparison_rows, bootstrap_payload, decision,
        ledger_payload=ledger_payload,
    )
    return {
        "datasets": datasets,
        "n_main_rows": len(main_rows),
        "n_comparison_rows": len(comparison_rows),
        "decision_axes": {key: value["axis"] for key, value in decision.items() if isinstance(value, dict)},
    }


# ---------------------------------------------------------------------------
# decision axes
# ---------------------------------------------------------------------------


def _merge_ledgers(ledger, collected: Mapping[str, Any]) -> dict[str, Any]:
    """Round-level resource account, merged from the per-worker ledgers.

    The aggregate stage runs in its own process, so its own ledger is empty; the
    comparable numbers are the ones each worker recorded next to its
    ``worker_summary.json``.
    """
    merged: dict[str, Any] = {"aggregate_run": ledger.summary()}
    per_dataset: dict[str, Any] = {}
    stages: dict[str, float] = {}
    total_wall = 0.0
    total_cpu = 0.0
    for dataset, payload in collected.items():
        record = (payload.get("worker") or {}).get("ledger")
        if not record:
            continue
        per_dataset[dataset] = record
        total_wall += float(record.get("wall_seconds_total", 0.0))
        total_cpu += float(record.get("cpu_seconds_total", 0.0))
        for stage, seconds in (record.get("wall_seconds_by_stage") or {}).items():
            stages[stage] = stages.get(stage, 0.0) + float(seconds)
    accounting: dict[str, Any] = {}
    for dataset, payload in collected.items():
        worker = payload.get("worker") or {}
        scorer_seconds = sum(
            float(scorer.get("fit_seconds", 0.0))
            for scorer in (payload.get("scorers") or {}).values()
            if scorer.get("method") in ("F", "R", "A", "E")
        )
        stage_seconds = {
            "discovery": float((worker.get("discovery") or {}).get("elapsed_seconds", 0.0)),
            "random_library": float((worker.get("random_library") or {}).get("elapsed_seconds", 0.0)),
            "xgboost_heads": scorer_seconds,
            "G": float((worker.get("G") or {}).get("seconds_total", 0.0) or 0.0),
            "L": float((worker.get("L") or {}).get("seconds", 0.0) or 0.0),
        }
        accounting[dataset] = {
            "stage_seconds": stage_seconds,
            "total_seconds": float(sum(stage_seconds.values())),
            "xgboost_device": worker.get("xgboost_device_used"),
            "torch_device": worker.get("torch_device_used"),
            "note": (
                "one dedicated GPU per worker: GPU time is the wall time of these "
                "stages (discovery runs the proxy XGBoost on the GPU as well)"
            ),
        }
    merged.update(
        {
            "per_dataset_worker_ledgers": per_dataset,
            "workers_wall_seconds_total": total_wall,
            "workers_cpu_seconds_total": total_cpu,
            "workers_wall_seconds_by_stage": dict(sorted(stages.items())),
            "gpu_accounting": accounting,
            "gpu_seconds_total": float(sum(item["total_seconds"] for item in accounting.values())),
            "gpu_hours_total": float(sum(item["total_seconds"] for item in accounting.values()) / 3600.0),
        }
    )
    return merged


def _pct_delta(value: float, reference: float) -> float:
    return float(value / reference)


def _decision(collected: Mapping[str, Any], bootstrap_payload: Mapping[str, Any]) -> dict[str, Any]:
    performance: dict[str, Any] = {"axis": "INCOMPLETE", "per_dataset": {}}
    signal: dict[str, Any] = {"axis": "CONTROL_INCOMPLETE", "per_dataset": {}}
    export: dict[str, Any] = {"axis": "EXPORT_PASS" if collected else "EXPORT_FAIL"}

    perf_flags = []
    signal_flags = []
    for dataset, payload in collected.items():
        scorers = payload["scorers"]
        entry: dict[str, Any] = {}
        if not scorers["A"]["complete"] or not scorers[payload["best_reference"]]["complete"]:
            perf_flags.append(False)
            entry["performance"] = "INCOMPLETE"
        else:
            ref = payload["best_reference"]
            ratios = {
                split: scorers["A"]["metrics"][split]["rmse"] / scorers[ref]["metrics"][split]["rmse"]
                for split in ("valid", "test")
            }
            ok = all(ratio <= 1.0 + COMPETITIVE_TOL for ratio in ratios.values())
            perf_flags.append(ok)
            entry["performance"] = {
                "reference": ref,
                "ratio_A_over_ref": {split: float(value) for split, value in ratios.items()},
                "competitive": bool(ok),
            }
        # query-learning signal
        if (
            not scorers["A"]["complete"]
            or not scorers["F"]["complete"]
            or not scorers["R"]["complete"]
            or not payload["random_library"]["n_missing"] == 0
        ):
            entry["signal"] = "CONTROL_INCOMPLETE"
            signal_flags.append(None)
        else:
            gains = {}
            for ref in ("F", "R"):
                for split in ("valid", "test"):
                    a = scorers["A"]["metrics"][split]["rmse"]
                    b = scorers[ref]["metrics"][split]["rmse"]
                    gains[f"{ref}_{split}"] = float((b - a) / b)
            beats_f = all(gains[f"F_{split}"] >= SIGNAL_GAIN for split in ("valid", "test"))
            beats_r = all(gains[f"R_{split}"] >= SIGNAL_GAIN for split in ("valid", "test"))
            r_not_worse = all(gains[f"R_{split}"] >= -COMPETITIVE_TOL for split in ("valid", "test"))
            entry["signal"] = {
                "relative_gain_vs_reference": gains,
                "beats_F_by_3pct_both_splits": bool(beats_f),
                "beats_R_by_3pct_both_splits": bool(beats_r),
                "R_not_worse_than_5pct": bool(r_not_worse),
            }
            signal_flags.append(
                {"gain": bool((beats_f and beats_r)), "r_not_worse": bool(r_not_worse)}
            )
        if not payload["replay"]["pass"]:
            export["axis"] = "EXPORT_FAIL"
        entry["replay_max_abs_error"] = payload["replay"]["max_abs_error_overall"]
        performance["per_dataset"][dataset] = entry.get("performance")
        signal["per_dataset"][dataset] = entry.get("signal")

    if len(perf_flags) == len(collected) and all(perf_flags):
        performance["axis"] = "PERFORMANCE_COMPETITIVE"
    elif collected:
        performance["axis"] = "PERFORMANCE_BEHIND"

    present = [flag for flag in signal_flags if flag is not None]
    if len(present) != len(collected) or not collected:
        signal["axis"] = "CONTROL_INCOMPLETE"
    else:
        gain_tasks = sum(1 for flag in present if flag["gain"])
        weak_tasks = sum(1 for flag in present if not flag["gain"] and not flag["r_not_worse"])
        if gain_tasks >= 1 and weak_tasks == 0:
            signal["axis"] = "QUERY_LEARNING_SIGNAL"
        elif gain_tasks == 0:
            signal["axis"] = "NO_INCREMENT_THIS_SEARCH"
        else:
            signal["axis"] = "MIXED_OR_SMALL"

    verdict = (
        performance["axis"] == "PERFORMANCE_COMPETITIVE"
        and signal["axis"] == "QUERY_LEARNING_SIGNAL"
        and export["axis"] == "EXPORT_PASS"
    )
    return {
        "performance": performance,
        "query_learning": signal,
        "export": export,
        "promising_small_data_pilot": bool(verdict),
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(
    output_dir: Path,
    collected: Mapping[str, Any],
    main_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    bootstrap_payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    ledger_payload: Mapping[str, Any] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Structural query learning pilot — sq-pilot-small-v1\n")
    lines.append(
        "官方 OGB `ogbg-molesol` / `ogbg-mollipo`，官方 Train/Validation/Test 划分，"
        "单训练 seed=0，`test_policy: always`（Test 作为本轮开发评估集）。"
        "本文件由 `stage=aggregate` 依据两个 GPU worker 的产物自动生成。\n"
    )
    lines.append("## 主表（每任务每方法：Val 选中状态，原始单位）\n")
    lines.append("| dataset | method | selected state | #queries | #feat | #trees | Train RMSE | Val RMSE | Test RMSE | Test MAE | fit s | complete |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in main_rows:
        if not row.get("complete"):
            lines.append(
                f"| {row['dataset']} | {row['method']} | {row.get('status')} | - | - | - | - | - | - | - | - | no |"
            )
            continue
        lines.append(
            "| {dataset} | {method} | {state} | {queries} | {feat} | {trees} | {tr} | {va} | {te} | {mae} | {sec} | yes |".format(
                dataset=row["dataset"],
                method=row["method"],
                state=row["selected_state"],
                queries=_fmt(row.get("n_queries")),
                feat=_fmt(row.get("n_features")),
                trees=_fmt(row.get("n_trees")),
                tr=_fmt(row.get("train_rmse")),
                va=_fmt(row.get("valid_rmse")),
                te=_fmt(row.get("test_rmse")),
                mae=_fmt(row.get("test_mae")),
                sec=_fmt(row.get("fit_seconds"), 1),
            )
        )
    lines.append("")
    lines.append("## F / R / A 对照与配对不确定性\n")
    lines.append("| dataset | comparison | split | RMSE A | RMSE ref | ΔRMSE (A−ref) | rel Δ | CI95 low | CI95 high | drop-1 Δ | flip |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in comparison_rows:
        if not row.get("complete"):
            lines.append(f"| {row['dataset']} | {row['comparison']} | - | - | - | - | - | - | - | - | incomplete |")
            continue
        lines.append(
            "| {d} | {c} | {s} | {a} | {b} | {delta} | {rel} | {lo} | {hi} | {d1} | {flip} |".format(
                d=row["dataset"],
                c=row["comparison"],
                s=row["split"],
                a=_fmt(row["rmse_A"]),
                b=_fmt(row["rmse_ref"]),
                delta=_fmt(row["delta_rmse"]),
                rel=_fmt(row["relative_delta"]),
                lo=_fmt(row["ci95_low"]),
                hi=_fmt(row["ci95_high"]),
                d1=_fmt(row["drop1_delta"]),
                flip="SINGLE_CASE_SENSITIVE" if row["drop1_sign_flip"] else "stable",
            )
        )
    lines.append("")
    lines.append("## 协议轴结论\n")
    lines.append(f"- PERFORMANCE 轴：**{decision['performance']['axis']}**")
    lines.append(f"- QUERY-LEARNING 轴：**{decision['query_learning']['axis']}**")
    lines.append(f"- EXECUTABLE-EXPORT 轴：**{decision['export']['axis']}**")
    lines.append(
        f"- PROMISING_SMALL_DATA_PILOT：**{decision['promising_small_data_pilot']}**"
    )
    lines.append("")
    lines.append("## 每任务查询规模与导出\n")
    lines.append("| dataset | B | discovered | R filled | trees | nodes | queries.json bytes | trees bytes | replay max abs err |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for dataset, payload in collected.items():
        worker = payload["worker"]
        costs = worker.get("costs", {})
        lines.append(
            "| {d} | {b} | {n} | {r} | {t} | {nodes} | {qb} | {tb} | {err} |".format(
                d=dataset,
                b=worker.get("n_queries_B"),
                n=worker.get("n_discovered"),
                r=f"{payload['random_library']['n_filled']}/{payload['random_library']['n_requested']}",
                t=costs.get("n_trees"),
                nodes=costs.get("n_nodes"),
                qb=costs.get("queries_file_bytes"),
                tb=costs.get("trees_file_bytes"),
                err=_fmt(payload["replay"]["max_abs_error_overall"], 8),
            )
        )
    accounting = (ledger_payload or {}).get("gpu_accounting") or {}
    if accounting:
        lines.append("")
        lines.append("## 资源账本（每个 worker 独占一块 GPU）\n")
        lines.append("| dataset | discovery s | R s | XGB heads s | G s | L s | 合计 s | XGB device | torch device |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for dataset, item in accounting.items():
            st = item["stage_seconds"]
            lines.append(
                "| {d} | {disc:.1f} | {rand:.1f} | {xgb:.1f} | {g:.1f} | {l:.1f} | {tot:.1f} | {xd} | {td} |".format(
                    d=dataset,
                    disc=st["discovery"],
                    rand=st["random_library"],
                    xgb=st["xgboost_heads"],
                    g=st["G"],
                    l=st["L"],
                    tot=item["total_seconds"],
                    xd=item.get("xgboost_device"),
                    td=item.get("torch_device"),
                )
            )
        lines.append("")
        lines.append(
            f"- 合计 GPU·s = {(ledger_payload or {}).get('gpu_seconds_total', 0.0):.1f}，"
            f"GPU·h = {(ledger_payload or {}).get('gpu_hours_total', 0.0):.2f}；"
            f"两 worker 的 stage 时间之和 = {(ledger_payload or {}).get('workers_wall_seconds_total', 0.0) / 3600.0:.2f} h"
            "（两个 worker 并行执行，实际墙钟取较大者）；本轮预算 8 h 墙钟 / 6 GPU·h。"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_lines = [
        "# DECISION — sq-pilot-small-v1\n",
        f"- PERFORMANCE axis: {decision['performance']['axis']}",
        f"- QUERY_LEARNING axis: {decision['query_learning']['axis']}",
        f"- EXECUTABLE_EXPORT axis: {decision['export']['axis']}",
        f"- PROMISING_SMALL_DATA_PILOT: {decision['promising_small_data_pilot']}",
        "",
        "详见 REPORT.md 与 decision.json（逐任务细目）。",
    ]
    (output_dir / "DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
