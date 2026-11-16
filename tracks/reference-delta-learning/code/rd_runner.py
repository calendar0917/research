"""Runner ``rd-pilot-small-v1``: prepare / smoke / dataset_worker / aggregate.

One run = one stage.  The two GPU workers are

    --set stage=dataset_worker --set dataset=ogbg-molesol   (pool res-gpu0)
    --set stage=dataset_worker --set dataset=ogbg-mollipo   (pool res-gpu1)

and each executes the frozen per-dataset sequence of the round brief:

    1 data identity + frozen Train references + frozen pair list
    2 D-H1, D-H2 (recording R at every round)
    3 N (constant, no training)
    4 C-H1, C-H2
    5 selected states, full-prefix recomputation, XGBoost cross-checks, reload

Everything is persisted per step, so an interrupted worker reloads finished
trajectories instead of retraining them, and the aggregate stage never needs a
GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rd_arms as A
import rd_common as C
import rd_data as D
import rd_pairs as P
import rd_refs as R
import rd_tree as T

DEFAULT_CONFIG = C.TRACK_ROOT / "configs" / "rd-pilot-small-v1.yaml"
SPLITS = C.SPLITS


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name=C.ROUND_ID,
        default_config=DEFAULT_CONFIG,
        default_study=C.STUDY_ID,
        description=(
            "Reference-delta pilot: joint pairwise delta learner vs direct absolute regression, "
            "Train-neighbour labels and direct-model neighbour residual correction on official "
            "ogbg-molesol / ogbg-mollipo (test_policy=always, seed 0)"
        ),
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "cache_files": dict(C.EXPECTED_CACHE_SHA256),
        "datasets": {},
    }
    for dataset in C.DATASETS:
        entry: dict[str, Any] = {
            "graph_cache_sha256": C.sha256_file(D.SQD.cache_path(dataset, base=C.DATA_DIR)),
            "ecfp_sha256": C.sha256_file(D.SQD.ecfp_path(dataset, base=C.DATA_DIR)),
        }
        graph = D.SQD.load_cache(dataset, base=C.DATA_DIR)
        entry["split_sizes"] = {split: int(D.SQD.split_indices(graph, split).size) for split in SPLITS}
        entry["split_hash"] = {
            split: C.sha256_array(D.SQD.split_indices(graph, split)) for split in SPLITS
        }
        entry["train_target_scaler"] = C.train_target_scaler(C.label_array(graph, "train"))
        payload["datasets"][dataset] = entry
    return payload, C.sha256_json(payload)


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------


def _write(path: Path, payload: Any) -> None:
    C.write_json(path, payload)


def _read(path: Path) -> Any:
    return C.read_json(path)


def _exists(path: Path) -> bool:
    return Path(path).is_file()


def _save_predictions(path: Path, payload: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for split, values in payload.items():
        for key, array in values.items():
            arrays[f"{split}__{key}"] = np.asarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": C.sha256_file(path)}


def _load_predictions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    payload: dict[str, dict[str, np.ndarray]] = {}
    for key, array in arrays.items():
        split, name = key.split("__", 1)
        payload.setdefault(split, {})[name] = array
    return payload


class GpuSampler:
    """Sample this process's GPU memory through ``nvidia-smi`` (best effort)."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb: float | None = None

    def _sample(self) -> float | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode != 0:
                return None
            total = 0.0
            found = False
            for line in out.stdout.splitlines():
                parts = [piece.strip() for piece in line.split(",")]
                if len(parts) != 2:
                    continue
                if int(parts[0]) == os.getpid():
                    total += float(parts[1])
                    found = True
            return total if found else None
        except Exception:  # noqa: BLE001 - provenance only
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            value = self._sample()
            if value is not None:
                self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        value = self._sample()
        if value is not None:
            self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)
        return False


# ---------------------------------------------------------------------------
# resume bookkeeping
# ---------------------------------------------------------------------------


def _state_path(dataset: str) -> Path:
    return C.results_dir(dataset) / "state.json"


def _load_state(dataset: str) -> dict[str, Any]:
    path = _state_path(dataset)
    if _exists(path):
        return _read(path)
    return {"dataset": dataset, "parts": {}}


def _save_state(state: dict[str, Any]) -> None:
    _write(_state_path(state["dataset"]), state)


def _mark(state: dict[str, Any], name: str, **info: Any) -> None:
    state["parts"][name] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    _save_state(state)


# ---------------------------------------------------------------------------
# prepare: data identity, frozen references, frozen pairs
# ---------------------------------------------------------------------------


def _prepare_dataset(dataset: str, ledger: C.Ledger) -> dict[str, Any]:
    out_dir = C.ensure_dir(C.results_dir(dataset))
    with C.stage_timer(ledger, "load_data", dataset=dataset):
        task = D.load_task(dataset)
        audit = D.data_audit(task)
    with C.stage_timer(ledger, "references", dataset=dataset):
        table = R.reference_table(task)
        ref_hash = R.reference_hash(table)
        summary = R.config_summary(table)
    with C.stage_timer(ledger, "eligible_order", dataset=dataset):
        eligible = P.eligible_order_train(task)
    # the frozen table and the pair pass must agree on the first 64 slots
    for query in range(task.n("train")):
        top = eligible[query][: C.MAX_REFERENCES]
        stored = table["splits"]["train"]["ref_local"][query, : table["splits"]["train"]["n_kept"][query]]
        if not np.array_equal(top, stored):
            raise AssertionError(f"{dataset}: Train Top64 table disagrees with the pair pass at query {query}")
    with C.stage_timer(ledger, "pairs", dataset=dataset):
        pairs = P.build_pairs(task, eligible)
    stats = pairs["stats"]
    if not stats["matches_expected_count"]:
        raise AssertionError(
            f"{dataset}: {stats['n_records']} pair records, expected {stats['n_records_expected']}"
        )
    if any(
        int(table["splits"][split]["eligible_count"].min()) <= 0 for split in SPLITS
    ):
        raise AssertionError(f"{dataset}: a query has no legal Train reference (REFERENCE_UNAVAILABLE)")
    # ---- cross-run freeze: the committed hashes must reproduce exactly ----
    for name, actual, kind in (
        ("references.json", ref_hash, "reference_hash"),
        ("pairs.json", pairs["hash"], "pair_hash"),
    ):
        frozen_path = out_dir / name
        if not frozen_path.is_file():
            continue
        frozen = C.read_json(frozen_path)
        if str(frozen.get("hash")) != str(actual):
            raise AssertionError(
                f"{dataset}: {kind} drifted: recomputed {actual} != frozen {frozen.get('hash')}"
            )
    ref_files = R.save_reference_table(table, out_dir / "references.npz")
    pair_files = P.save_pairs(pairs, out_dir / "pairs.npz")
    _write(out_dir / "data_audit.json", audit)
    _write(
        out_dir / "references.json",
        {
            "dataset": dataset,
            "hash": ref_hash,
            "configurations": list(C.CONFIGS),
            "summary": summary,
            "files": ref_files,
            "aggregate": {
                split: {
                    "n_queries": int(table["splits"][split]["eligible_count"].size),
                    "eligible_min": int(table["splits"][split]["eligible_count"].min()),
                    "eligible_max": int(table["splits"][split]["eligible_count"].max()),
                    "same_smiles_excluded_total": int(
                        table["splits"][split]["same_smiles_excluded"].sum()
                    ),
                    "max_sim_mean": float(table["splits"][split]["max_sim"].mean()),
                }
                for split in SPLITS
            },
        },
    )
    pair_payload = {
        "dataset": dataset,
        "hash": pairs["hash"],
        "seed": int(pairs["seed"]),
        "stats": stats,
        "files": pair_files,
        "contract": {
            "rng": "numpy.random.default_rng(1701)",
            "n_top": C.N_TOP,
            "n_tail": C.N_TAIL,
            "max_pairs": C.MAX_PAIRS,
            "top_pool": C.TOP_POOL,
            "weight": "1/(2*q_i) on both directions",
        },
    }
    _write(out_dir / "pairs.json", pair_payload)
    return {
        "dataset": dataset,
        "audit": audit,
        "reference_hash": ref_hash,
        "pair_hash": pairs["hash"],
        "pair_stats": stats,
        "reference_files": ref_files,
        "pair_files": pair_files,
    }


def _stage_prepare(config: Mapping[str, Any], ledger: C.Ledger) -> dict[str, Any]:
    metrics: dict[str, Any] = {"stage": "prepare", "datasets": {}}
    for dataset in C.DATASETS:
        info = _prepare_dataset(dataset, ledger)
        metrics["datasets"][dataset] = {
            "reference_hash": info["reference_hash"],
            "pair_hash": info["pair_hash"],
            "n_records": info["pair_stats"]["n_records"],
            "n_records_expected": info["pair_stats"]["n_records_expected"],
            "duplicate_directed_rate": info["pair_stats"]["duplicate_directed_rate"],
        }
    metrics["environment"] = C.environment_snapshot()
    return metrics


# ---------------------------------------------------------------------------
# trajectory persistence
# ---------------------------------------------------------------------------


def _curve_rows_to_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    C.write_csv(
        path,
        rows,
        columns=["dataset", "method", "preset", "round", "split", "config", "rmse_raw", "mae_raw"],
    )


def _read_curves(path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "preset": row["preset"],
                    "round": int(row["round"]),
                    "split": row["split"],
                    "config": row["config"],
                    "rmse_raw": float(row["rmse_raw"]),
                    "mae_raw": float(row["mae_raw"]),
                }
            )
    return rows


def _write_pair_curves(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    C.write_csv(
        path,
        rows,
        columns=[
            "dataset",
            "method",
            "preset",
            "round",
            "pair_rmse_std",
            "pair_rmse_std_antisym",
        ],
    )


def _read_pair_curves(path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "preset": row["preset"],
                    "round": int(row["round"]),
                    "pair_rmse_std": float(row["pair_rmse_std"]),
                    "pair_rmse_std_antisym": float(row["pair_rmse_std_antisym"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# dataset worker
# ---------------------------------------------------------------------------


def _run_or_load_d(
    ctx: A.ArmContext,
    dataset: str,
    preset: str,
    device: str,
    rounds: int,
    state: dict[str, Any],
    ledger: C.Ledger,
) -> tuple[A.DRun, dict[str, Any]]:
    import xgboost as xgb

    out_dir = C.ensure_dir(C.results_dir(dataset))
    model_path = out_dir / "models" / f"D_{preset}.ubj"
    state_path = out_dir / f"state_D_{preset}.npz"
    curves_path = out_dir / f"curves_D_{preset}.csv"
    part = f"D_{preset}"
    if part in state["parts"] and _exists(model_path) and _exists(state_path) and _exists(curves_path):
        booster = xgb.Booster(model_file=str(model_path))
        with np.load(state_path, allow_pickle=False) as handle:
            stored = {key: handle[key] for key in handle.files}
        trees = T.extract_trees(booster, list(ctx.task.feature_names), rounds=int(stored["trained_rounds"]))
        curves = _read_curves(curves_path)
        checkpoints: dict[int, np.ndarray] = {}
        run = A.DRun(
            preset=preset,
            booster=booster,
            trees=trees,
            curves=curves,
            final_accumulator=stored["final_accumulator"],
            final_predictions={
                split: stored[f"final_pred_{split}"] for split in SPLITS
            },
            seconds=float(stored["seconds"]),
            trained_rounds=int(stored["trained_rounds"]),
            eval_seconds=float(stored["eval_seconds"]),
            checkpoints=checkpoints,
        )
        return run, state["parts"][part]
    with C.stage_timer(ledger, f"train_D_{preset}", dataset=dataset, preset=preset):
        with GpuSampler() as sampler:
            run = A.run_d(ctx, preset=preset, device=device, rounds=rounds)
        gpu_peak = sampler.peak_mb
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run.booster.save_model(str(model_path))
    arrays = {
        "final_accumulator": run.final_accumulator,
        "trained_rounds": np.asarray(run.trained_rounds),
        "seconds": np.asarray(run.seconds),
        "eval_seconds": np.asarray(run.eval_seconds),
    }
    for split in SPLITS:
        arrays[f"final_pred_{split}"] = run.final_predictions[split]
    np.savez_compressed(state_path, **arrays)
    _curve_rows_to_csv(curves_path, run.curves)
    info = {
        "preset": preset,
        "trained_rounds": run.trained_rounds,
        "seconds": run.seconds,
        "eval_seconds": run.eval_seconds,
        "gpu_peak_mb": gpu_peak,
        "model_sha256": C.sha256_file(model_path),
        "curves_sha256": C.sha256_file(curves_path),
    }
    _mark(state, part, **info)
    return run, info


def _run_or_load_c(
    ctx: A.ArmContext,
    dataset: str,
    preset: str,
    device: str,
    rounds: int,
    pair_spec: dict[str, Any],
    state: dict[str, Any],
    ledger: C.Ledger,
) -> tuple[A.CRun, dict[str, Any]]:
    import xgboost as xgb

    out_dir = C.ensure_dir(C.results_dir(dataset))
    model_path = out_dir / "models" / f"C_{preset}.ubj"
    state_path = out_dir / f"state_C_{preset}.npz"
    curves_path = out_dir / f"curves_C_{preset}.csv"
    pair_curves_path = out_dir / f"pair_curves_C_{preset}.csv"
    part = f"C_{preset}"
    if (
        part in state["parts"]
        and _exists(model_path)
        and _exists(state_path)
        and _exists(curves_path)
        and _exists(pair_curves_path)
    ):
        booster = xgb.Booster(model_file=str(model_path))
        names = A._pair_feature_names(ctx.task)
        with np.load(state_path, allow_pickle=False) as handle:
            stored = {key: handle[key] for key in handle.files}
        trees = T.extract_trees(booster, names, rounds=int(stored["trained_rounds"]))
        final_margins: dict[str, dict[str, np.ndarray]] = {}
        for split in SPLITS:
            final_margins[split] = {
                "fwd": stored[f"margin_{split}_fwd"],
                "rev": stored[f"margin_{split}_rev"],
            }
        final_margins["pairs"] = {
            "fwd": stored["margin_pairs_fwd"],
            "rev": stored["margin_pairs_rev"],
        }
        run = A.CRun(
            preset=preset,
            booster=booster,
            trees=trees,
            curves=_read_curves(curves_path),
            pair_curves=_read_pair_curves(pair_curves_path),
            seconds=float(stored["seconds"]),
            trained_rounds=int(stored["trained_rounds"]),
            matrix_build_seconds=float(stored["matrix_build_seconds"]),
            eval_seconds=float(stored["eval_seconds"]),
            final_margins=final_margins,
        )
        return run, state["parts"][part]
    with C.stage_timer(ledger, f"train_C_{preset}", dataset=dataset, preset=preset):
        with GpuSampler() as sampler:
            run = A.run_c(
                ctx,
                pair_spec,
                preset=preset,
                device=device,
                rounds=rounds,
            )
        gpu_peak = sampler.peak_mb
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run.booster.save_model(str(model_path))
    arrays: dict[str, np.ndarray] = {
        "trained_rounds": np.asarray(run.trained_rounds),
        "seconds": np.asarray(run.seconds),
        "matrix_build_seconds": np.asarray(run.matrix_build_seconds),
        "eval_seconds": np.asarray(run.eval_seconds),
    }
    for split in SPLITS + ("pairs",):
        arrays[f"margin_{split}_fwd"] = run.final_margins[split]["fwd"]
        arrays[f"margin_{split}_rev"] = run.final_margins[split]["rev"]
    np.savez_compressed(state_path, **arrays)
    _curve_rows_to_csv(curves_path, run.curves)
    _write_pair_curves(pair_curves_path, run.pair_curves)
    info = {
        "preset": preset,
        "trained_rounds": run.trained_rounds,
        "seconds": run.seconds,
        "matrix_build_seconds": run.matrix_build_seconds,
        "eval_seconds": run.eval_seconds,
        "gpu_peak_mb": gpu_peak,
        "model_sha256": C.sha256_file(model_path),
        "curves_sha256": C.sha256_file(curves_path),
        "pair_curves_sha256": C.sha256_file(pair_curves_path),
    }
    _mark(state, part, **info)
    return run, info


def _selected_state_payload(
    dataset: str,
    arm: str,
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    preset: str,
    round_index: int,
    config: str | None,
    trained_rounds: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Selected row plus the fixed final-round / last-20 summaries (never selection)."""

    def _rows_at(round_value: int) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for row in curve_rows:
            if (
                row["preset"] == preset
                and int(row["round"]) == int(round_value)
                and (config is None or row["config"] == config)
            ):
                found.setdefault(row["split"], row)
        return found

    payload: dict[str, Any] = {"arm": arm, "preset": preset, "round": int(round_index)}
    if config is not None:
        payload["config"] = config
    payload["trained_rounds"] = int(trained_rounds)
    payload["used_rounds"] = int(round_index)
    for split in SPLITS:
        selected = None
        for row in curve_rows:
            if (
                row["split"] == split
                and row["preset"] == preset
                and int(row["round"]) == int(round_index)
                and (config is None or row["config"] == config)
            ):
                selected = row
                break
        if selected is None:
            raise KeyError(f"{dataset} {arm}: no selected row for {split}")
        payload[f"{split}_rmse_raw"] = float(selected["rmse_raw"])
        payload[f"{split}_mae_raw"] = float(selected["mae_raw"])
        final = _rows_at(int(trained_rounds)).get(split)
        if final is not None:
            payload[f"final_round_{split}_rmse_raw"] = float(final["rmse_raw"])
            payload[f"final_round_{split}_mae_raw"] = float(final["mae_raw"])
        tail = [
            row
            for row in curve_rows
            if row["split"] == split
            and row["preset"] == preset
            and (config is None or row["config"] == config)
        ]
        tail = tail[-20:]
        payload[f"mean_last20_{split}_rmse_raw"] = float(np.mean([row["rmse_raw"] for row in tail]))
        payload[f"mean_last20_{split}_mae_raw"] = float(np.mean([row["mae_raw"] for row in tail]))
    if extra:
        payload.update(dict(extra))
    return payload


def _reload_check_d(
    ctx: A.ArmContext,
    model_path: Path,
    predictions: Mapping[str, np.ndarray],
    final_predictions: Mapping[str, np.ndarray],
    *,
    rounds: int,
    final_round: int,
) -> dict[str, Any]:
    """Reloaded booster vs (a) the selected prefix and (b) the recorded final state."""
    import xgboost as xgb

    booster = xgb.Booster(model_file=str(model_path))
    names = list(ctx.task.feature_names)
    matrix = xgb.DMatrix(ctx.task.x["valid"], feature_names=names)
    selected = np.asarray(booster.predict(matrix, iteration_range=(0, int(rounds))), dtype=np.float64)
    raw = ctx.mu + ctx.sigma * selected
    payload: dict[str, Any] = {
        "model": str(model_path),
        "split": "valid",
        "selected_prefix_round": int(rounds),
        "final_round": int(final_round),
        "max_abs_diff_raw": float(np.max(np.abs(raw - predictions["valid"]))),
    }
    if int(final_round) != int(rounds):
        final = np.asarray(booster.predict(matrix), dtype=np.float64)
        raw_final = ctx.mu + ctx.sigma * final
        payload["max_abs_diff_final_raw"] = float(
            np.max(np.abs(raw_final - final_predictions["valid"]))
        )
    return payload


def _reload_check_c(
    ctx: A.ArmContext,
    model_path: Path,
    state: dict[str, Any],
    final_margins: Mapping[str, Mapping[str, np.ndarray]],
    sample_rows: Mapping[str, np.ndarray],
    *,
    rounds: int,
    final_round: int,
) -> dict[str, Any]:
    """Reloaded booster vs (a) the selected prefix and (b) the recorded final state."""
    import xgboost as xgb

    booster = xgb.Booster(model_file=str(model_path))
    names = A._pair_feature_names(ctx.task)
    prefix: dict[str, float] = {}
    final: dict[str, float] = {}
    for split in SPLITS:
        entry = ctx.ref_table["splits"][split]
        rows = np.asarray(sample_rows[split], dtype=np.int64)
        fwd_matrix, rev_matrix = A._sample_pair_matrices(ctx.task, entry, rows)
        fwd = xgb.DMatrix(fwd_matrix, feature_names=names)
        rev = xgb.DMatrix(rev_matrix, feature_names=names)
        del fwd_matrix, rev_matrix
        for label, matrix, key in (("fwd", fwd, "fwd"), ("rev", rev, "rev")):
            selected = np.asarray(
                booster.predict(matrix, iteration_range=(0, int(rounds))), dtype=np.float64
            )
            prefix[f"{split}_{label}"] = float(
                np.max(
                    np.abs(
                        selected - state["margins"][split][key][rows].astype(np.float64)
                    )
                )
            )
            if int(final_round) != int(rounds):
                full = np.asarray(booster.predict(matrix), dtype=np.float64)
                final[f"{split}_{label}"] = float(
                    np.max(
                        np.abs(
                            full
                            - np.asarray(final_margins[split][key])[rows].astype(np.float64)
                        )
                    )
                )
    return {
        "model": str(model_path),
        "selected_prefix_round": int(rounds),
        "final_round": int(final_round),
        "max_abs_margin_diff": prefix,
        "max_abs_margin_diff_final": final,
    }


def _stage_dataset_worker(
    config: Mapping[str, Any], ledger: C.Ledger, artifact_dir: Path
) -> dict[str, Any]:
    dataset = str(config["dataset"])
    if dataset not in C.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    device_request = str(config.get("xgboost_device", "cuda:0"))
    out_dir = C.ensure_dir(C.results_dir(dataset))
    tasks = int(config.get("num_tasks", 1))
    stage_t0 = time.perf_counter()
    state = _load_state(dataset)

    prepared = _prepare_dataset(dataset, ledger)
    task = D.load_task(dataset)
    ref_table = R.load_reference_table(out_dir / "references.npz", dataset)
    pair_spec = P.load_pairs(out_dir / "pairs.npz", dataset)
    if pair_spec["hash"] != prepared["pair_hash"]:
        raise AssertionError("pair hash changed between prepare and worker")
    ctx = A.ArmContext.build(task, ref_table)

    device, device_status = C.SQX.resolve_backend(device_request)
    _write(out_dir / "device.json", {"requested": device_request, "used": device, "status": device_status})

    # ---- D and R -------------------------------------------------------
    d_runs: dict[str, A.DRun] = {}
    d_info: dict[str, Any] = {}
    for preset in C.PRESET_ORDER:
        run, info = _run_or_load_d(ctx, dataset, preset, device, C.ROUNDS_D, state, ledger)
        d_runs[preset] = run
        d_info[preset] = info
    d_selection = A.select_d(list(d_runs.values()))
    r_selection = A.select_r(list(d_runs.values()))
    d_selected_run = d_runs[d_selection["preset"]]
    d_selected_pred = A.predict_d_full(
        ctx, d_selected_run.booster, d_selected_run.trees, int(d_selection["round"])
    )
    # R selects its own (preset, round, config): the residual anchor is that D state
    r_run = d_runs[r_selection["preset"]]
    r_prefix = A.predict_d_full(ctx, r_run.booster, r_run.trees, int(r_selection["round"]))
    r_selected_pred = {
        split: A.d_r_predictions_at(ctx, r_prefix, r_selection["config"], split) for split in SPLITS
    }

    # ---- N -------------------------------------------------------------
    with C.stage_timer(ledger, "N", dataset=dataset):
        n_run = A.run_n(ctx)

    # ---- C -------------------------------------------------------------
    c_runs: dict[str, A.CRun] = {}
    c_info: dict[str, Any] = {}
    for preset in C.PRESET_ORDER:
        run, info = _run_or_load_c(ctx, dataset, preset, device, C.ROUNDS_C, pair_spec, state, ledger)
        c_runs[preset] = run
        c_info[preset] = info
    c_selection = A.select_c(list(c_runs.values()))
    c_selected_run = c_runs[c_selection["preset"]]

    # ---- selected states and selected predictions ----------------------
    with C.stage_timer(ledger, "selected_state_C", dataset=dataset):
        sample_rows = {
            split: A.sample_flat_rows(
                int(task.n(split) * C.MAX_REFERENCES), 512, seed=C.BOOTSTRAP_SEED
            )
            for split in SPLITS
        }
        c_state = A.c_state(
            ctx,
            booster=c_selected_run.booster,
            trees=c_selected_run.trees,
            rounds=int(c_selection["round"]),
            config_name=str(c_selection["config"]),
            sample_rows=sample_rows,
        )

    # every selected state must exist in the curve file of *its own* preset
    for arm, chosen, curves, config in (
        ("D", d_selection, d_selected_run.curves, "none"),
        ("R", r_selection, d_runs[r_selection["preset"]].curves, str(r_selection["config"])),
        ("C", c_selection, c_selected_run.curves, str(c_selection["config"])),
    ):
        wanted_config = config
        for split in SPLITS:
            found = any(
                row["split"] == split
                and row["preset"] == chosen["preset"]
                and int(row["round"]) == int(chosen["round"])
                and row["config"] == wanted_config
                for row in curves
            )
            if not found:
                raise AssertionError(
                    f"{dataset} {arm}: selected {chosen['preset']}/round {chosen['round']}/"
                    f"config {wanted_config} has no {split} curve row"
                )

    selection = {
        "dataset": dataset,
        "D": _selected_state_payload(
            dataset,
            "D",
            d_selected_run.curves,
            preset=d_selection["preset"],
            round_index=int(d_selection["round"]),
            config=None,
            trained_rounds=d_selected_run.trained_rounds,
        ),
        "R": _selected_state_payload(
            dataset,
            "R",
            # R selects its own preset, so its curve rows live in that preset's file
            d_runs[r_selection["preset"]].curves,
            preset=r_selection["preset"],
            round_index=int(r_selection["round"]),
            config=str(r_selection["config"]),
            trained_rounds=d_runs[r_selection["preset"]].trained_rounds,
            extra={
                "d_preset": r_selection["preset"],
                "d_round": int(r_selection["round"]),
                "d_source_trajectory": f"R reuses D-{r_selection['preset']} at round {r_selection['round']}",
            },
        ),
        "N": {
            "arm": "N",
            **{key: value for key, value in n_run["selected"].items()},
            "preset": "constant",
            "round": 0,
            "used_rounds": 0,
            "trained_rounds": 0,
        },
        "C": _selected_state_payload(
            dataset,
            "C",
            c_selected_run.curves,
            preset=c_selection["preset"],
            round_index=int(c_selection["round"]),
            config=str(c_selection["config"]),
            trained_rounds=c_selected_run.trained_rounds,
            extra={
                "pair_rmse_std_at_selected": _pair_curve_value(
                    c_selected_run.pair_curves, int(c_selection["round"]), "pair_rmse_std"
                ),
                "pair_rmse_std_antisym_at_selected": _pair_curve_value(
                    c_selected_run.pair_curves, int(c_selection["round"]), "pair_rmse_std_antisym"
                ),
                "mean_last20_pair_rmse_std": float(
                    np.mean([row["pair_rmse_std"] for row in c_selected_run.pair_curves[-20:]])
                ),
            },
        ),
        "configs": [
            {"arm": "N", **row} for row in n_run["states"]
        ],
        "D_candidates": [
            {"preset": preset, **run.best} for preset, run in d_runs.items()
        ],
        "R_candidates_best_per_preset": {
            preset: A.select_r([run]) for preset, run in d_runs.items()
        },
        "C_candidates_best_per_preset": {
            preset: A.select_c([run]) for preset, run in c_runs.items()
        },
        "trajectories": {"D": d_info, "C": c_info},
    }
    _write(out_dir / "selection.json", selection)

    # ---- the persisted selected predictions must reproduce the curves ----
    for arm, payload_pred, chosen in (
        ("D", d_selected_pred, selection["D"]),
        ("R", r_selected_pred, selection["R"]),
        ("N", n_run["selected_predictions"], selection["N"]),
        ("C", c_state["pred"], selection["C"]),
    ):
        for split in C.SPLITS:
            recomputed = C.rmse(payload_pred[split], task.y[split])
            reported = float(chosen[f"{split}_rmse_raw"])
            if abs(recomputed - reported) > 1e-9:
                raise AssertionError(
                    f"{dataset} {arm}: persisted {split} RMSE {recomputed} != "
                    f"selected {reported}"
                )

    # ---- persistence of selected predictions ---------------------------
    d_payload = {
        split: {
            "pred_raw": d_selected_pred[split],
            "y_raw": task.y[split],
            "row_id": task.row_id[split],
        }
        for split in SPLITS
    }
    r_payload = {
        split: {
            "pred_raw": r_selected_pred[split],
            "y_raw": task.y[split],
            "row_id": task.row_id[split],
        }
        for split in SPLITS
    }
    n_payload = {
        split: {
            "pred_raw": n_run["selected_predictions"][split],
            "y_raw": task.y[split],
            "row_id": task.row_id[split],
        }
        for split in SPLITS
    }
    c_payload = {
        split: {
            "pred_raw": c_state["pred"][split],
            "y_raw": task.y[split],
            "row_id": task.row_id[split],
        }
        for split in SPLITS
    }
    files = {
        "D": _save_predictions(out_dir / "pred_D.npz", d_payload),
        "R": _save_predictions(out_dir / "pred_R.npz", r_payload),
        "N": _save_predictions(out_dir / "pred_N.npz", n_payload),
        "C": _save_predictions(out_dir / "pred_C.npz", c_payload),
    }
    delta_arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        entry = ref_table["splits"][split]
        delta_arrays[f"{split}__delta_raw"] = c_state["delta"][split]
        delta_arrays[f"{split}__weights"] = ctx.weights[split][str(c_selection["config"])]
        delta_arrays[f"{split}__y_ref"] = ctx.y_ref[split]
        delta_arrays[f"{split}__ref_row_id"] = entry["ref_row_id"]
        delta_arrays[f"{split}__sim"] = entry["sim"]
        delta_arrays[f"{split}__row_id"] = task.row_id[split]
    np.savez_compressed(out_dir / "pred_C_delta.npz", **delta_arrays)
    files["C_delta"] = {
        "path": str(out_dir / "pred_C_delta.npz"),
        "sha256": C.sha256_file(out_dir / "pred_C_delta.npz"),
    }

    # ---- cross-checks --------------------------------------------------
    checks: dict[str, Any] = {"dataset": dataset, "D": {}, "R": {}, "C": {}}
    d_index = d_selected_run.curve_index()
    with C.stage_timer(ledger, "checks_D", dataset=dataset):
        for round_index in (1, 2, 3):
            if round_index in d_selected_run.checkpoints:
                preds = {
                    split: ctx.mu + ctx.sigma * d_selected_run.checkpoints[round_index][
                        _split_slice(task, split)
                    ].astype(np.float64)
                    for split in SPLITS
                }
            else:
                preds = A.predict_d_full(ctx, d_selected_run.booster, d_selected_run.trees, round_index)
            checks["D"][f"round_{round_index}"] = A.verify_d(
                ctx,
                booster=d_selected_run.booster,
                trees=d_selected_run.trees,
                round_index=round_index,
                recorded_predictions=preds,
                curves_index=d_index,
            )
        selected_round = int(d_selection["round"])
        if selected_round not in (1, 2, 3, d_selected_run.trained_rounds):
            fresh = A.predict_d_full(ctx, d_selected_run.booster, d_selected_run.trees, selected_round)
        else:
            fresh = d_selected_pred
        checks["D"][f"selected_round_{selected_round}"] = A.verify_d(
            ctx,
            booster=d_selected_run.booster,
            trees=d_selected_run.trees,
            round_index=selected_round,
            recorded_predictions=fresh,
            curves_index=d_index,
        )
        checks["D"][f"final_round_{d_selected_run.trained_rounds}"] = A.verify_d(
            ctx,
            booster=d_selected_run.booster,
            trees=d_selected_run.trees,
            round_index=d_selected_run.trained_rounds,
            recorded_predictions=d_selected_run.final_predictions,
            curves_index=d_index,
        )
        checks["D"]["reload"] = _reload_check_d(
            ctx,
            out_dir / "models" / f"D_{d_selection['preset']}.ubj",
            d_selected_pred,
            d_selected_run.final_predictions,
            rounds=int(d_selection["round"]),
            final_round=int(d_selected_run.trained_rounds),
        )

    c_index = c_selected_run.curve_index() if hasattr(c_selected_run, "curve_index") else None
    if c_index is None:
        c_index = {
            (row["round"], row["split"], row["config"]): row for row in c_selected_run.curves
        }
    with C.stage_timer(ledger, "checks_C", dataset=dataset):
        for round_index in (1, 2, 3):
            state_r = A.c_state(
                ctx,
                booster=c_selected_run.booster,
                trees=c_selected_run.trees,
                rounds=round_index,
                config_name=str(c_selection["config"]),
                sample_rows={
                    split: A.sample_flat_rows(
                        int(task.n(split) * C.MAX_REFERENCES), 256, seed=C.BOOTSTRAP_SEED + round_index
                    )
                    for split in SPLITS
                },
            )
            checks["C"][f"round_{round_index}"] = A.verify_c_prefix(
                ctx,
                state=state_r,
                round_index=round_index,
                config_name=str(c_selection["config"]),
                curves_index=c_index,
            )
        checks["C"][f"selected_round_{c_selection['round']}"] = A.verify_c(
            ctx,
            booster=c_selected_run.booster,
            pair_spec=pair_spec,
            final_margins=c_selected_run.final_margins,
            state=c_state,
            round_index=int(c_selection["round"]),
            final_round=c_selected_run.trained_rounds,
            curves_index=c_index,
            config_name=str(c_selection["config"]),
            sample_count=512,
        )
        checks["C"]["reload"] = _reload_check_c(
            ctx,
            out_dir / "models" / f"C_{c_selection['preset']}.ubj",
            c_state,
            c_selected_run.final_margins,
            sample_rows,
            rounds=int(c_selection["round"]),
            final_round=int(c_selected_run.trained_rounds),
        )
    _write(out_dir / "checks.json", checks)

    summary = {
        "dataset": dataset,
        "device_requested": device_request,
        "device_used": device,
        "device_status": device_status,
        "n_train": task.n("train"),
        "n_valid": task.n("valid"),
        "n_test": task.n("test"),
        "scaler": dict(task.scaler),
        "reference_hash": prepared["reference_hash"],
        "pair_hash": prepared["pair_hash"],
        "pair_stats": prepared["pair_stats"],
        "selection": {arm: selection[arm] for arm in ("D", "R", "N", "C")},
        "trajectories": {"D": d_info, "C": c_info},
        "prediction_files": files,
        "checks_ok": _checks_ok(checks),
        "tasks_per_worker": tasks,
        "wall_seconds_total": time.perf_counter() - stage_t0,
        "gpu_hours_total": (time.perf_counter() - stage_t0) / 3600.0,
        "environment": C.environment_snapshot(),
        **C.peak_memory_mb(),
    }
    _write(out_dir / "worker_summary.json", summary)
    _mark(state, "dataset_worker", summary={k: summary[k] for k in ("checks_ok", "dataset")})
    return {
        "dataset": dataset,
        "selected": {
            arm: {
                "valid_rmse_raw": selection[arm]["valid_rmse_raw"],
                "test_rmse_raw": selection[arm]["test_rmse_raw"],
                "preset": selection[arm]["preset"],
                "round": selection[arm]["round"],
            }
            for arm in ("D", "R", "N", "C")
        },
        "checks_ok": summary["checks_ok"],
        "device_used": device,
        "pair_hash": prepared["pair_hash"],
        "reference_hash": prepared["reference_hash"],
    }


def _split_slice(task: D.TaskData, split: str) -> slice:
    cursor = 0
    for name in SPLITS:
        n = task.n(name)
        if name == split:
            return slice(cursor, cursor + n)
        cursor += n
    raise KeyError(split)


def _pair_curve_value(rows: Sequence[Mapping[str, Any]], round_index: int, key: str) -> float:
    for row in rows:
        if int(row["round"]) == int(round_index):
            return float(row[key])
    raise KeyError((round_index, key))


def _checks_ok(checks: Mapping[str, Any]) -> dict[str, Any]:
    tol = 1e-5
    worst_d = 0.0
    worst_c = 0.0
    worst_metric = 0.0
    for key, report in checks["D"].items():
        if key == "reload":
            worst_d = max(worst_d, float(report["max_abs_diff_raw"]))
            if "max_abs_diff_final_raw" in report:
                worst_d = max(worst_d, float(report["max_abs_diff_final_raw"]))
            continue
        for split, entry in report["splits"].items():
            worst_d = max(worst_d, float(entry["max_abs_sample_diff_raw"]))
            worst_metric = max(worst_metric, float(entry["rmse_abs_diff"]))
        for key_r, entry_r in report.get("r_axis", {}).items():
            worst_metric = max(worst_metric, float(entry_r["rmse_abs_diff"]))
    for key, report in checks["C"].items():
        if key == "reload":
            worst_c = max(worst_c, max(float(v) for v in report["max_abs_margin_diff"].values()))
            if report.get("max_abs_margin_diff_final"):
                worst_c = max(
                    worst_c, max(float(v) for v in report["max_abs_margin_diff_final"].values())
                )
            continue
        for split, entry in report["splits"].items():
            for name in (
                "max_abs_margin_diff_sel_fwd",
                "max_abs_margin_diff_sel_rev",
                "max_abs_margin_diff_fwd",
                "max_abs_margin_diff_rev",
            ):
                if name in entry:
                    worst_c = max(worst_c, float(entry[name]))
            worst_metric = max(worst_metric, float(entry["rmse_abs_diff"]))
        if "pair_sample" in report:
            worst_c = max(
                worst_c,
                float(report["pair_sample"]["max_abs_margin_diff_final_fwd"]),
                float(report["pair_sample"]["max_abs_margin_diff_final_rev"]),
            )
        if "final_state" in report:
            for split, entry in report["final_state"]["splits"].items():
                worst_c = max(
                    worst_c,
                    float(entry["max_abs_margin_diff_fwd"]),
                    float(entry["max_abs_margin_diff_rev"]),
                )
    return {
        "per_sample_tolerance_raw": tol,
        "max_abs_diff_D_raw": worst_d,
        "max_abs_margin_diff_C": worst_c,
        "max_abs_metric_diff": worst_metric,
        "pass": bool(worst_d <= tol and worst_c <= tol and worst_metric <= 1e-6),
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _stage_smoke(config: Mapping[str, Any], ledger: C.Ledger) -> dict[str, Any]:
    dataset = str(config["dataset"])
    rounds = int(config.get("smoke_rounds", 3))
    out_dir = C.ensure_dir(C.CACHE_DIR / "smoke" / dataset)
    device, device_status = C.SQX.resolve_backend(str(config.get("xgboost_device", "cuda:0")))
    with C.stage_timer(ledger, "smoke_prepare", dataset=dataset):
        task = D.load_task(dataset)
        _ = D.data_audit(task)
        ref_table = R.reference_table(task)
        eligible = P.eligible_order_train(task)
        pairs = P.build_pairs(task, eligible)
        ctx = A.ArmContext.build(task, ref_table)
    report: dict[str, Any] = {
        "dataset": dataset,
        "device_requested": config.get("xgboost_device", "cuda:0"),
        "device_used": device,
        "device_status": device_status,
        "pair_stats": pairs["stats"],
        "rounds": rounds,
    }
    with C.stage_timer(ledger, "smoke_D", dataset=dataset):
        with GpuSampler() as sampler:
            d_run = A.run_d(ctx, preset="H1", device=device, rounds=rounds)
        report["D"] = {
            "train_seconds": d_run.seconds,
            "eval_seconds": d_run.eval_seconds,
            "gpu_peak_mb": sampler.peak_mb,
            "best_valid_rmse_raw": d_run.best["rmse_raw"],
        }
    with C.stage_timer(ledger, "smoke_C", dataset=dataset):
        with GpuSampler() as sampler:
            c_run = A.run_c(ctx, pairs, preset="H1", device=device, rounds=rounds)
        report["C"] = {
            "matrix_build_seconds": c_run.matrix_build_seconds,
            "train_seconds": c_run.seconds,
            "eval_seconds": c_run.eval_seconds,
            "gpu_peak_mb": sampler.peak_mb,
        }
    with C.stage_timer(ledger, "smoke_checks", dataset=dataset):
        sample_rows = {
            split: A.sample_flat_rows(int(task.n(split) * C.MAX_REFERENCES), 128) for split in SPLITS
        }
        c_state = A.c_state(
            ctx,
            booster=c_run.booster,
            trees=c_run.trees,
            rounds=rounds,
            config_name=C.CONFIGS[0]["name"],
            sample_rows=sample_rows,
        )
        index = {(row["round"], row["split"], row["config"]): row for row in c_run.curves}
        c_check = A.verify_c_prefix(
            ctx,
            state=c_state,
            round_index=rounds,
            config_name=C.CONFIGS[0]["name"],
            curves_index=index,
        )
        d_check = A.verify_d(
            ctx,
            booster=d_run.booster,
            trees=d_run.trees,
            round_index=rounds,
            recorded_predictions=d_run.final_predictions,
            curves_index=d_run.curve_index(),
        )
    report["checks"] = {"D": d_check, "C": c_check}
    per_round_D = d_run.seconds / max(rounds, 1)
    per_round_C = c_run.seconds / max(rounds, 1)
    report["estimate"] = {
        "D_train_seconds_full": per_round_D * C.ROUNDS_D * 2,
        "C_train_seconds_full": per_round_C * C.ROUNDS_C * 2,
        "C_matrix_build_seconds": c_run.matrix_build_seconds,
        "per_round_D_seconds": per_round_D,
        "per_round_C_seconds": per_round_C,
        "note": "smoke extrapolation from the first rounds; init effects make it an upper bound",
    }
    _write(out_dir / "smoke.json", report)
    return report


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _stage_aggregate(config: Mapping[str, Any], ledger: C.Ledger) -> dict[str, Any]:
    import rd_report as REP

    metrics = REP.aggregate(ledger=ledger)
    return metrics


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(config: Mapping[str, Any], context) -> Any:
    from ksvd_research.runtime.manifest import RunResult

    ledger_dir = Path(context.artifact_dir)
    ledger = C.Ledger(ledger_dir / "ledger.jsonl")
    stage = str(config.get("stage", "dataset_worker"))
    t0 = time.perf_counter()
    if stage == "prepare":
        metrics = _stage_prepare(config, ledger)
    elif stage == "smoke":
        metrics = _stage_smoke(config, ledger)
    elif stage == "dataset_worker":
        metrics = _stage_dataset_worker(config, ledger, ledger_dir)
    elif stage == "aggregate":
        metrics = _stage_aggregate(config, ledger)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    metrics = {**metrics, "stage": stage, "wall_seconds": time.perf_counter() - t0}
    _write(ledger_dir / f"{stage}_summary.json", {**metrics, "ledger": ledger.summary()})
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[f"artifacts/{stage}_summary.json", "artifacts/ledger.jsonl"],
    )
