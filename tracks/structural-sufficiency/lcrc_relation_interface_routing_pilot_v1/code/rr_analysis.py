"""Aggregate stage: predictions, block gating, oracle localisation, decision.

Everything here is CPU-only and reads only frozen artifacts:

* the four formal trajectories (``results/<task>/<model>/``),
* their per-molecule native ledgers (``ledger_<split>.npz``),
* the frozen routing caches (block membership of atoms and interfaces),
* the verified per-atom ``WC_LOGP`` contribution cache of the oracle round,
* the historical baselines (read-only numbers, never retrained).

Two oracle quantities are kept strictly separate:

1. **unallocated candidate atom coverage**: ``sum_{v in atoms(block)} |c_v|``
   divided by the molecule total — used for the fair comparison against random
   connected subgraphs with the same bond and atom count;
2. **weight-allocated oracle mass bookkeeping**: ``w(v,i)`` splits every atom's
   ``|c_v|`` across the blocks that share it, and the allocation is checked to
   close on all blocks of every molecule.
"""

from __future__ import annotations

import csv
import time
from typing import Any, Mapping, Sequence

import numpy as np

import rr_common as C
import rr_data as D
import rr_train as T
from rr_checks import select_blocks

MOD = C.modules()


# ---------------------------------------------------------------------------
# frozen historical baselines (read-only artifacts)
# ---------------------------------------------------------------------------


def _read_stage1() -> list[dict[str, str]]:
    path = C.ROUTE_ANALYSIS / "tables/stage1.csv"
    if not path.is_file():
        return []
    return C.read_csv(path)


def historical_baselines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage1 = _read_stage1()
    for row in stage1:
        if row.get("dataset") != C.DATASET:
            continue
        if row.get("recipe") != "Rf":
            continue
        arch = row.get("model")
        name = {"L": "C1", "G0": "GINE-G0", "Gx": "GINE-Gx"}.get(arch, arch)
        rows.append(
            {
                "task": "lipo",
                "model": f"{name} (frozen, {row['recipe']})",
                "split": "valid",
                "rmse": float(row["valid_rmse_raw"]),
                "mae": float(row["valid_mae_raw"]),
                "epoch": int(float(row["selected_epoch"])),
                "n_parameters": int(float(row["n_parameters"])),
                "source": "tracks/structural-sufficiency/gnn_route_clarification_8h_v1/analysis/tables/stage1.csv",
                "comparability": "DIRECTLY_COMPARABLE",
                "comparability_note": "same dataset/split/target/scaler/seed 0/recipe Rf",
            }
        )
        rows.append(
            {
                "task": "lipo",
                "model": f"{name} (frozen, {row['recipe']})",
                "split": "test",
                "rmse": float(row["test_rmse_raw"]),
                "mae": float(row["test_mae_raw"]),
                "epoch": int(float(row["selected_epoch"])),
                "n_parameters": int(float(row["n_parameters"])),
                "source": "tracks/structural-sufficiency/gnn_route_clarification_8h_v1/analysis/tables/stage1.csv",
                "comparability": "DIRECTLY_COMPARABLE",
                "comparability_note": "same dataset/split/target/scaler/seed 0/recipe Rf; test is DEVELOPMENT_TEST",
            }
        )
    # the readout-context round's re-trained C1 (recipe R3)
    c1_traj = C.C1_LIPO_TRAJECTORY
    if c1_traj.is_file():
        payload = C.read_json(c1_traj)
        for split in ("valid", "test"):
            rows.append(
                {
                    "task": "lipo",
                    "model": "C1 (frozen, R3)",
                    "split": split,
                    "rmse": float(payload["selected"][f"{split}_rmse_raw"]),
                    "mae": float(payload["selected"][f"{split}_mae_raw"]),
                    "epoch": int(payload["selected"]["epoch"]),
                    "n_parameters": int(payload["n_parameters"]),
                    "source": "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results/ogbg-mollipo/A/trajectory.json",
                    "comparability": "RECIPE_DIFFERS",
                    "comparability_note": "same split/target/scaler/seed, recipe R3 (peak 3e-4, wd 1e-3, no warmup) vs Rf here",
                }
            )
    # the WC_LOGP LCRC-C1 run of the oracle round (recipe R3)
    if C.WC_LOG_TRAIN_JSON.is_file():
        payload = C.read_json(C.WC_LOG_TRAIN_JSON)
        for split in ("valid", "test"):
            m = payload["metrics"]["best"][split]
            rows.append(
                {
                    "task": "wc_logp",
                    "model": "C1 (frozen, R3)",
                    "split": split,
                    "rmse": float(m["rmse"]),
                    "mae": float(m["mae"]),
                    "epoch": int(payload["checkpoint"]["best_epoch"]),
                    "n_parameters": int(payload["model"]["parameters"]),
                    "source": "tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs/wc_logp/train.json",
                    "comparability": "RECIPE_DIFFERS",
                    "comparability_note": "same split/target/scaler/seed; recipe R3 (peak 3e-4, wd 1e-3) vs Rf here",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# loading the four formal runs
# ---------------------------------------------------------------------------


def load_run(task: str, model: str, split: str) -> dict[str, Any] | None:
    out_dir = C.task_results_dir(task, model)
    traj_path = out_dir / "trajectory.json"
    ledger_path = out_dir / f"ledger_{split}.npz"
    if not traj_path.is_file() or not ledger_path.is_file():
        return None
    traj = C.read_json(traj_path)
    with np.load(ledger_path, allow_pickle=False) as mesh:
        ledger = {key: np.asarray(mesh[key]) for key in mesh.files}
    with np.load(out_dir / f"best_predictions.npz", allow_pickle=False) as mesh:
        preds = {f"{split}_pred_raw": np.asarray(mesh[f"{split}_pred_raw"])}
        preds[f"{split}_y_raw"] = np.asarray(mesh[f"{split}_y_raw"])
        preds[f"{split}_row_id"] = np.asarray(mesh[f"{split}_row_id"])
    return {"trajectory": traj, "ledger": ledger, "predictions": preds, "dir": out_dir}


def _ledger_core(ledger: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    block_ptr = np.asarray(ledger["ldg_block_ptr"], dtype=np.int64)
    iface_ptr = np.asarray(ledger["ldg_iface_ptr"], dtype=np.int64)
    c_block = np.asarray(ledger["ldg_c_block"], dtype=np.float64)
    c_iface = np.asarray(ledger["ldg_c_iface"], dtype=np.float64)
    ia = np.asarray(ledger["ldg_iface_a_local"], dtype=np.int64)
    ib = np.asarray(ledger["ldg_iface_b_local"], dtype=np.int64)
    n_mol = block_ptr.size - 1
    return {
        "block_ptr": block_ptr,
        "iface_ptr": iface_ptr,
        "c_block": c_block,
        "c_iface": c_iface,
        "iface_a": ia,
        "iface_b": ib,
        "n_molecules": np.asarray([n_mol]),
        "block_n_bonds": np.asarray(ledger["ldg_blk_n_bonds"], dtype=np.int64),
        "block_n_atoms": np.asarray(ledger["ldg_blk_n_atoms"], dtype=np.int64),
        "bias": np.asarray(ledger["ldg_bias"], dtype=np.float64),
        "rho": np.asarray(ledger["ldg_rho"], dtype=np.float64),
        "block_sum": np.asarray(ledger["ldg_block_sum"], dtype=np.float64),
        "iface_sum": np.asarray(ledger["ldg_iface_sum"], dtype=np.float64),
        "pred_std": np.asarray(ledger["ldg_pred_std"], dtype=np.float64),
        "y_raw": np.asarray(ledger["y_raw"], dtype=np.float64),
        "pred_raw": np.asarray(ledger["pred_raw"], dtype=np.float64),
        "row_id": np.asarray(ledger["row_id"], dtype=np.int64),
        "cache_rows": np.asarray(ledger["ldg_cache_rows"], dtype=np.int64),
    }


def gating_table(core: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-block ``delta_i = c_i + sum_j c_ij`` plus the frozen selection score."""
    block_ptr = core["block_ptr"]
    iface_ptr = core["iface_ptr"]
    c_block = core["c_block"]
    c_iface = core["c_iface"]
    ia = core["iface_a"]
    ib = core["iface_b"]
    n_bonds = core["block_n_bonds"]
    n_mol = int(block_ptr.size - 1)
    delta = c_block.astype(np.float64, copy=True)
    for k in range(ia.size):
        g = int(np.searchsorted(iface_ptr, k, side="right") - 1)
        base = int(block_ptr[g])
        delta[base + int(ia[k])] += float(c_iface[k])
        delta[base + int(ib[k])] += float(c_iface[k])
    score = np.abs(delta) / np.sqrt(np.maximum(1.0, n_bonds.astype(np.float64)))
    selected = select_blocks(
        c_block,
        c_iface,
        ia,
        ib,
        n_bonds,
        np.diff(block_ptr),
        np.diff(iface_ptr),
    )
    return {
        "delta": delta,
        "score": score,
        "selected_local": selected["selected_local_index"],
        "selected_score": selected["score"],
    }


# ---------------------------------------------------------------------------
# random connected subgraph controls (WC_LOGP oracle localisation)
# ---------------------------------------------------------------------------


def _connected_subgraph(
    n_atoms: int,
    edges: Sequence[tuple[int, int]],
    target_bonds: int,
    target_atoms: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 400,
) -> frozenset | None:
    """One random connected bond set with ``target_bonds`` bonds and ``target_atoms`` atoms."""
    m = len(edges)
    if target_bonds <= 0 or target_bonds > m:
        return None
    for _ in range(max_attempts):
        chosen = {int(rng.integers(0, m))}
        atoms = set(edges[next(iter(chosen))])
        guard = 0
        while len(chosen) < target_bonds and guard < 4 * m:
            guard += 1
            candidates = []
            for e in range(m):
                if e in chosen:
                    continue
                u, v = edges[e]
                if u in atoms or v in atoms:
                    candidates.append(e)
            if not candidates:
                break
            e = int(candidates[int(rng.integers(0, len(candidates)))])
            chosen.add(e)
            atoms.update(edges[e])
        if len(chosen) == target_bonds and len(atoms) == target_atoms:
            return frozenset(chosen)
    return None


def oracle_localisation(task: str, model: str, split: str, max_controls: int = C.MAX_ORACLE_CONTROLS) -> dict[str, Any]:
    """Unallocated candidate atom coverage of the selected block vs random controls."""
    run = load_run(task, model, split)
    if run is None:
        return {"available": False}
    core = _ledger_core(run["ledger"])
    gating = gating_table(core)
    bundle = D.build_bundle()
    cache = bundle.caches[split]
    routing = bundle.routing[split]
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    with np.load(C.ORACLE_CONTRIB_NPZ, allow_pickle=False) as mesh:
        atoms_value = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr_oracle = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    global_ids = bundle.global_row_ids[split]
    blk_bond_ptr = np.asarray(routing["blk_bond_ptr"], dtype=np.int64)
    blk_bond = np.asarray(routing["blk_bond"], dtype=np.int64)
    belong_ptr = np.asarray(routing["belong_ptr"], dtype=np.int64)
    belong_block = np.asarray(routing["belong_block"], dtype=np.int64)
    belong_weight = np.asarray(routing["belong_weight"], dtype=np.float64)
    block_ptr = core["block_ptr"]
    n_mol = int(block_ptr.size - 1)

    rows: list[dict[str, Any]] = []
    allocated_gap = 0.0
    allocated_total = 0.0
    for g in range(n_mol):
        row_id = int(global_ids[int(core["cache_rows"][g])])
        a0, a1 = int(atom_ptr[int(core["cache_rows"][g])]), int(atom_ptr[int(core["cache_rows"][g]) + 1])
        b0, b1 = int(bond_ptr[int(core["cache_rows"][g])]), int(bond_ptr[int(core["cache_rows"][g]) + 1])
        o0 = int(atom_ptr_oracle[row_id])
        atom_abs = np.abs(atoms_value[o0 : o0 + (a1 - a0)])
        total_mass = float(atom_abs.sum())
        edges = [(int(bond_index[j, 0]), int(bond_index[j, 1])) for j in range(b0, b1)]
        # unallocated candidate coverage of the selected block
        base = int(block_ptr[g])
        sel = int(gating["selected_local"][g])
        if sel < 0:
            continue
        block_global = base + sel
        bb0, bb1 = int(blk_bond_ptr[block_global]), int(blk_bond_ptr[block_global + 1])
        block_bonds = [int(x) - b0 for x in blk_bond[bb0:bb1]]
        block_atoms = sorted({int(v) for e in block_bonds for v in edges[e]})
        capture = float(atom_abs[block_atoms].sum() / total_mass) if total_mass > 0 else 0.0
        # matched random connected subgraphs
        rng = np.random.default_rng([C.ORACLE_CONTROL_SEED, row_id])
        controls: list[float] = []
        seen: set[frozenset] = {frozenset(block_bonds)}
        attempts = 0
        while len(controls) < max_controls and attempts < 40 * max_controls:
            attempts += 1
            candidate = _connected_subgraph(
                a1 - a0, edges, len(block_bonds), len(block_atoms), rng
            )
            if candidate is None:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            ctrl_atoms = sorted({int(v) for e in candidate for v in edges[e]})
            controls.append(float(atom_abs[ctrl_atoms].sum() / total_mass) if total_mass > 0 else 0.0)
        matched = len(controls) >= C.MIN_ORACLE_CONTROLS
        control_mean = float(np.mean(controls)) if controls else float("nan")
        # weight-allocated oracle mass over all blocks (bookkeeping only)
        alloc = np.zeros(int(block_ptr[g + 1] - block_ptr[g]), dtype=np.float64)
        for v in range(a1 - a0):
            lo, hi = int(belong_ptr[a0 + v]), int(belong_ptr[a0 + v + 1])
            for k in range(lo, hi):
                alloc[int(belong_block[k])] += belong_weight[k] * atom_abs[v]
        allocated_total += float(alloc.sum())
        allocated_gap = max(allocated_gap, abs(float(alloc.sum()) - total_mass))
        rows.append(
            {
                "task": task,
                "model": model,
                "split": split,
                "row_id": row_id,
                "n_blocks": int(block_ptr[g + 1] - block_ptr[g]),
                "selected_block": sel,
                "selected_n_bonds": int(len(block_bonds)),
                "selected_n_atoms": int(len(block_atoms)),
                "delta": float(gating["delta"][block_global]),
                "score": float(gating["score"][block_global]),
                "capture": capture,
                "control_mean": control_mean,
                "enrichment": capture - control_mean,
                "enrichment_ratio": (capture / control_mean) if control_mean > 0 else float("nan"),
                "n_controls": len(controls),
                "matched": bool(matched),
                "oracle_mass": total_mass,
                "allocated_mass": float(alloc.sum()),
                "allocated_max_block": float(alloc.max()) if alloc.size else 0.0,
                "selected_allocated_mass": float(alloc[sel]),
            }
        )
    return {
        "available": True,
        "rows": rows,
        "allocation_closes": bool(allocated_gap < 1e-8),
        "allocation_max_abs_gap": float(allocated_gap),
        "oracle_mass_total": float(allocated_total),
    }


# ---------------------------------------------------------------------------
# paired bootstrap (R2 - R1)
# ---------------------------------------------------------------------------


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, *, draws: int = C.BOOTSTRAP_DRAWS, seed: int = C.BOOTSTRAP_SEED):
    """Molecule-level paired bootstrap of the RMSE difference ``a - b``.

    ``err_*`` are signed residuals in raw units; the same row resampling is used
    for both models, which is what makes the comparison paired.
    """
    n = int(err_a.size)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        idx = rng.integers(0, n, size=n)
        rmse_a = float(np.sqrt(np.mean(err_a[idx] ** 2)))
        rmse_b = float(np.sqrt(np.mean(err_b[idx] ** 2)))
        diffs[d] = rmse_a - rmse_b
    point_a = float(np.sqrt(np.mean(err_a ** 2)))
    point_b = float(np.sqrt(np.mean(err_b ** 2)))
    return {
        "n": n,
        "rmse_a": point_a,
        "rmse_b": point_b,
        "diff": point_a - point_b,
        "ratio": point_a / point_b if point_b else float("nan"),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "prob_a_better": float(np.mean(diffs < 0.0)),
        "draws": int(draws),
        "seed": int(seed),
    }


def paired_bootstrap_ratio(err_a: np.ndarray, err_b: np.ndarray, *, draws: int = C.BOOTSTRAP_DRAWS, seed: int = C.BOOTSTRAP_SEED):
    n = int(err_a.size)
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        idx = rng.integers(0, n, size=n)
        rmse_a = float(np.sqrt(np.mean(err_a[idx] ** 2)))
        rmse_b = float(np.sqrt(np.mean(err_b[idx] ** 2)))
        ratios[d] = rmse_a / rmse_b if rmse_b else float("nan")
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _rows_for_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def aggregate() -> dict[str, Any]:
    C.ensure_dir(C.ANALYSIS_DIR)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    support = C.read_json(C.ANALYSIS_DIR / "partition_support.json")
    budget = C.read_json(C.ANALYSIS_DIR / "parameter_budget.json")
    freeze = C.read_json(C.ANALYSIS_DIR / "protocol_freeze.json")

    prediction_rows: list[dict[str, Any]] = []
    prediction_rows.extend(historical_baselines())
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    rho_rows: list[dict[str, Any]] = []
    for task in C.TASKS:
        for model in C.MODELS:
            for split in ("valid", "test"):
                run = load_run(task, model, split)
                if run is None:
                    continue
                traj = run["trajectory"]
                sel = traj["selected"]
                prediction_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "split": split,
                        "rmse": float(sel[f"{split}_rmse_raw"]),
                        "mae": float(sel[f"{split}_mae_raw"]),
                        "epoch": int(sel["epoch"]),
                        "n_parameters": int(traj["n_parameters"]),
                        "source": str(C.task_results_dir(task, model) / "trajectory.json"),
                        "comparability": "THIS_ROUND",
                        "comparability_note": "seed 0, recipe Rf, official split, DEVELOPMENT_TEST",
                    }
                )
                runs[(task, model, split)] = run
                core = _ledger_core(run["ledger"])
                scaler = D.build_bundle().scaler(task)
                rho_raw = (core["bias"] + core["rho"]) * scaler["std"] + scaler["mean"]
                y_raw = core["y_raw"]
                rho_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "split": split,
                        "rho_only_rmse": float(np.sqrt(np.mean((rho_raw - y_raw) ** 2))),
                        "rho_only_mae": float(np.mean(np.abs(rho_raw - y_raw))),
                        "full_rmse": float(np.sqrt(np.mean((core["pred_raw"] - y_raw) ** 2))),
                        "train_mean_constant_rmse": float(np.sqrt(np.mean((y_raw - y_raw.mean()) ** 2))),
                    }
                )
    # dedupe repeated historical rows (valid/test appended in the loop)
    seen = set()
    deduped = []
    for row in prediction_rows:
        key = (row["task"], row["model"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    prediction_rows = deduped
    C.write_csv(
        C.ANALYSIS_DIR / "prediction_comparison.csv",
        prediction_rows,
        columns=[
            "task", "model", "split", "rmse", "mae", "epoch", "n_parameters",
            "source", "comparability", "comparability_note",
        ],
    )
    C.write_csv(
        C.ANALYSIS_DIR / "rho_only_baseline.csv",
        rho_rows,
        columns=["task", "model", "split", "rho_only_rmse", "rho_only_mae", "full_rmse", "train_mean_constant_rmse"],
    )

    # ---- block gating / ledger scale ------------------------------------
    ledger_rows: list[dict[str, Any]] = []
    block_scale_rows: list[dict[str, Any]] = []
    for (task, model, split), run in runs.items():
        core = _ledger_core(run["ledger"])
        gating = gating_table(core)
        block_ptr = core["block_ptr"]
        iface_ptr = core["iface_ptr"]
        n_mol = int(block_ptr.size - 1)
        abs_block = np.abs(gating["delta"])
        for g in range(n_mol):
            b0, b1 = int(block_ptr[g]), int(block_ptr[g + 1])
            i0, i1 = int(iface_ptr[g]), int(iface_ptr[g + 1])
            sel = int(gating["selected_local"][g])
            ledger_rows.append(
                {
                    "task": task,
                    "model": model,
                    "split": split,
                    "row_id": int(core["row_id"][g]),
                    "n_blocks": int(b1 - b0),
                    "n_interfaces": int(i1 - i0),
                    "bias": float(core["bias"][g]),
                    "rho": float(core["rho"][g]),
                    "block_sum": float(core["block_sum"][g]),
                    "iface_sum": float(core["iface_sum"][g]),
                    "pred_raw": float(core["pred_raw"][g]),
                    "y_raw": float(core["y_raw"][g]),
                    "residual": float(core["pred_raw"][g] - core["y_raw"][g]),
                    "ledger_closure_error": float(
                        abs(
                            core["pred_raw"][g]
                            - (
                                (core["bias"][g] + core["block_sum"][g] + core["iface_sum"][g] + core["rho"][g])
                            )
                        )
                    ),
                    "max_abs_delta": float(abs_block[b0:b1].max()) if b1 > b0 else 0.0,
                    "sum_abs_delta": float(abs_block[b0:b1].sum()) if b1 > b0 else 0.0,
                    "selected_block": sel,
                    "selected_delta": float(gating["delta"][b0 + sel]) if sel >= 0 else float("nan"),
                    "selected_score": float(gating["selected_score"][g]),
                    "selected_n_bonds": int(core["block_n_bonds"][b0 + sel]) if sel >= 0 else -1,
                }
            )
            block_scale_rows.append(
                {
                    "task": task,
                    "model": model,
                    "split": split,
                    "row_id": int(core["row_id"][g]),
                    "rms_block": float(np.sqrt(np.mean(core["c_block"][b0:b1] ** 2))) if b1 > b0 else 0.0,
                    "rms_iface": float(np.sqrt(np.mean(core["c_iface"][i0:i1] ** 2))) if i1 > i0 else 0.0,
                    "abs_block_sum": float(abs(core["block_sum"][g])),
                    "abs_iface_sum": float(abs(core["iface_sum"][g])),
                    "abs_rho": float(abs(core["rho"][g])),
                    "n_blocks": int(b1 - b0),
                    "n_interfaces": int(i1 - i0),
                }
            )
    C.write_csv(
        C.ANALYSIS_DIR / "block_ledger.csv",
        ledger_rows,
        columns=[
            "task", "model", "split", "row_id", "n_blocks", "n_interfaces", "bias", "rho",
            "block_sum", "iface_sum", "pred_raw", "y_raw", "residual", "ledger_closure_error",
            "max_abs_delta", "sum_abs_delta", "selected_block", "selected_delta",
            "selected_score", "selected_n_bonds",
        ],
    )
    block_scale = {}
    for task in C.TASKS:
        for model in C.MODELS:
            for split in ("valid", "test"):
                subset = [r for r in block_scale_rows if (r["task"], r["model"], r["split"]) == (task, model, split)]
                if not subset:
                    continue
                block_scale[f"{task}/{model}/{split}"] = {
                    "mean_abs_block_sum": float(np.mean([r["abs_block_sum"] for r in subset])),
                    "mean_abs_iface_sum": float(np.mean([r["abs_iface_sum"] for r in subset])),
                    "mean_abs_rho": float(np.mean([r["abs_rho"] for r in subset])),
                    "median_rms_block": float(np.median([r["rms_block"] for r in subset])),
                    "median_rms_iface": float(np.median([r["rms_iface"] for r in subset])),
                    "median_n_blocks": float(np.median([r["n_blocks"] for r in subset])),
                    "median_n_interfaces": float(np.median([r["n_interfaces"] for r in subset])),
                    "mean_abs_delta_selected": float(np.mean([abs(r["selected_delta"]) for r in subset])),
                    "max_ledger_closure_error": float(max(r["ledger_closure_error"] for r in subset)),
                }
    C.atomic_write_json(C.ANALYSIS_DIR / "native_ledger_summary.json", block_scale)

    # ---- interface ledger -----------------------------------------------
    interface_rows: list[dict[str, Any]] = []
    for (task, model, split), run in runs.items():
        core = _ledger_core(run["ledger"])
        block_ptr = core["block_ptr"]
        iface_ptr = core["iface_ptr"]
        n_mol = int(block_ptr.size - 1)
        for g in range(n_mol):
            b0 = int(block_ptr[g])
            for k in range(int(iface_ptr[g]), int(iface_ptr[g + 1])):
                a_local = int(core["iface_a"][k])
                b_local = int(core["iface_b"][k])
                interface_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "split": split,
                        "row_id": int(core["row_id"][g]),
                        "block_a": a_local,
                        "block_b": b_local,
                        "n_bonds_a": int(core["block_n_bonds"][b0 + a_local]),
                        "n_bonds_b": int(core["block_n_bonds"][b0 + b_local]),
                        "c_iface": float(core["c_iface"][k]),
                    }
                )
    C.write_csv(
        C.ANALYSIS_DIR / "interface_ledger.csv",
        interface_rows,
        columns=["task", "model", "split", "row_id", "block_a", "block_b", "n_bonds_a", "n_bonds_b", "c_iface"],
    )

    # ---- oracle localisation (WC_LOGP only) ------------------------------
    oracle_rows: list[dict[str, Any]] = []
    oracle_meta: dict[str, Any] = {}
    for model in C.MODELS:
        for split in ("valid", "test"):
            res = oracle_localisation("wc_logp", model, split)
            if not res.get("available"):
                continue
            oracle_rows.extend(res["rows"])
            matched = [r for r in res["rows"] if r["matched"]]
            oracle_meta[f"{model}/{split}"] = {
                "n_molecules": len(res["rows"]),
                "n_matched": len(matched),
                "matched_fraction": len(matched) / max(1, len(res["rows"])),
                "allocation_closes": res["allocation_closes"],
                "allocation_max_abs_gap": res["allocation_max_abs_gap"],
                "mean_controls": float(np.mean([r["n_controls"] for r in res["rows"]])) if res["rows"] else 0.0,
                "mean_enrichment_matched": float(np.mean([r["enrichment"] for r in matched])) if matched else float("nan"),
                "median_enrichment_matched": float(np.median([r["enrichment"] for r in matched])) if matched else float("nan"),
                "fraction_enrichment_positive": float(np.mean([r["enrichment"] > 0 for r in matched])) if matched else float("nan"),
                "fraction_enrichment_zero": float(np.mean([r["enrichment"] == 0 for r in matched])) if matched else float("nan"),
                "mean_capture": float(np.mean([r["capture"] for r in matched])) if matched else float("nan"),
                "mean_control_capture": float(np.mean([r["control_mean"] for r in matched])) if matched else float("nan"),
                "selected_block_bonds": _quantiles([r["selected_n_bonds"] for r in res["rows"]]),
                "selected_block_atoms": _quantiles([r["selected_n_atoms"] for r in res["rows"]]),
                "spearman_delta_vs_allocated": _spearman(
                    [abs(r["delta"]) for r in res["rows"]], [r["selected_allocated_mass"] for r in res["rows"]]
                ),
                "spearman_delta_vs_capture": _spearman(
                    [abs(r["delta"]) for r in res["rows"]], [r["capture"] for r in res["rows"]]
                ),
            }
    C.write_csv(
        C.ANALYSIS_DIR / "oracle_localization.csv",
        oracle_rows,
        columns=[
            "task", "model", "split", "row_id", "n_blocks", "selected_block",
            "selected_n_bonds", "selected_n_atoms", "delta", "score", "capture",
            "control_mean", "enrichment", "enrichment_ratio", "n_controls", "matched",
            "oracle_mass", "allocated_mass", "allocated_max_block", "selected_allocated_mass",
        ],
    )
    C.atomic_write_json(C.ANALYSIS_DIR / "oracle_localization_summary.json", oracle_meta)

    # ---- paired bootstrap R2 - R1 ---------------------------------------
    bootstrap: dict[str, Any] = {}
    bundle = D.build_bundle()
    for task in C.TASKS:
        for split in ("valid", "test"):
            ra = load_run(task, "R1", split)
            rb = load_run(task, "R2", split)
            if ra is None or rb is None:
                continue
            ca = _ledger_core(ra["ledger"])
            cb = _ledger_core(rb["ledger"])
            if not np.array_equal(ca["row_id"], cb["row_id"]):
                raise RuntimeError(f"{task}/{split}: R1 and R2 row order differs")
            ea = ca["pred_raw"] - ca["y_raw"]
            eb = cb["pred_raw"] - cb["y_raw"]
            entry = paired_bootstrap(ea, eb)
            entry["ci95_ratio"] = paired_bootstrap_ratio(ea, eb)
            entry["mean_abs_residual_R1"] = float(np.mean(np.abs(ea)))
            entry["mean_abs_residual_R2"] = float(np.mean(np.abs(eb)))
            # per-molecule paired error difference (R1^2 - R2^2); positive = R2 better
            sq = ea ** 2 - eb ** 2
            entry["prob_R2_better"] = float(np.mean(sq > 0))
            entry["n_positive"] = int((sq > 0).sum())
            bootstrap[f"{task}/{split}"] = entry
    C.atomic_write_json(C.ANALYSIS_DIR / "bootstrap_r2_vs_r1.json", bootstrap)

    # ---- decision --------------------------------------------------------
    decision = build_decision(prediction_rows, bootstrap, oracle_meta, support, budget, freeze, block_scale)
    C.atomic_write_json(C.ANALYSIS_DIR / "decision.json", decision)
    manifest = run_manifest(started)
    C.atomic_write_json(C.ANALYSIS_DIR / "run_manifest.json", manifest)
    out = {
        "stage": "aggregate",
        "n_runs": len(runs),
        "decision": decision["flags"],
        "oracle_matched_fraction": {k: v["matched_fraction"] for k, v in oracle_meta.items()},
    }
    return out


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    # average ties
    unique, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for u in np.flatnonzero(counts > 1):
            mask = inverse == u
            ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr, yr = _rank(x[mask]), _rank(y[mask])
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


# ---------------------------------------------------------------------------
# decision flags
# ---------------------------------------------------------------------------


def build_decision(
    prediction_rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    oracle_meta: Mapping[str, Any],
    support: Mapping[str, Any],
    budget: Mapping[str, Any],
    freeze: Mapping[str, Any],
    block_scale: Mapping[str, Any],
) -> dict[str, Any]:
    def rmse(task: str, model: str, split: str) -> float | None:
        for row in prediction_rows:
            if (row["task"], row["model"], row["split"]) == (task, model, split):
                return float(row["rmse"])
        return None

    c1 = {split: rmse("lipo", "C1 (frozen, Rf)", split) for split in ("valid", "test")}
    gine = {split: rmse("lipo", "GINE-G0 (frozen, Rf)", split) for split in ("valid", "test")}
    c1_wc = {split: rmse("wc_logp", "C1 (frozen, R3)", split) for split in ("valid", "test")}
    new = {(t, m, s): rmse(t, m, s) for t in C.TASKS for m in C.MODELS for s in ("valid", "test")}

    retention = {}
    for model in C.MODELS:
        lipo_v = new[("lipo", model, "valid")]
        wc_v = new[("wc_logp", model, "valid")]
        retention[model] = {
            "lipo_valid": lipo_v,
            "lipo_valid_ratio_vs_c1_rf": (lipo_v / c1["valid"]) if lipo_v and c1["valid"] else None,
            "lipo_test": new[("lipo", model, "test")],
            "lipo_test_ratio_vs_c1_rf": (
                new[("lipo", model, "test")] / c1["test"] if new[("lipo", model, "test")] and c1["test"] else None
            ),
            "wc_logp_valid": wc_v,
            "wc_logp_valid_ratio_vs_c1_r3": (wc_v / c1_wc["valid"]) if wc_v and c1_wc["valid"] else None,
            "wc_logp_test": new[("wc_logp", model, "test")],
        }
    retention_ok = {
        model: bool(
            retention[model]["lipo_valid_ratio_vs_c1_rf"] is not None
            and retention[model]["lipo_valid_ratio_vs_c1_rf"] <= C.RETENTION_FACTOR
            and retention[model]["wc_logp_valid_ratio_vs_c1_r3"] is not None
            and retention[model]["wc_logp_valid_ratio_vs_c1_r3"] <= C.RETENTION_FACTOR
        )
        for model in C.MODELS
    }
    any_retention = any(retention_ok.values())

    rho_dominance = {}
    for model in C.MODELS:
        for task in C.TASKS:
            key = f"{task}/{model}/valid"
            if key in block_scale:
                bs = block_scale[key]
                rho_dominance[f"{task}/{model}"] = {
                    "mean_abs_rho": bs["mean_abs_rho"],
                    "mean_abs_block_sum": bs["mean_abs_block_sum"],
                    "mean_abs_iface_sum": bs["mean_abs_iface_sum"],
                    "rho_over_structural": bs["mean_abs_rho"]
                    / max(1e-12, bs["mean_abs_block_sum"] + bs["mean_abs_iface_sum"]),
                }
    rho_only = {}
    rho_path = C.ANALYSIS_DIR / "rho_only_baseline.csv"
    if rho_path.is_file():
        for row in C.read_csv(rho_path):
            rho_only[f"{row['task']}/{row['model']}/{row['split']}"] = {
                "rho_only_rmse": float(row["rho_only_rmse"]),
                "full_rmse": float(row["full_rmse"]),
            }

    inc = {}
    for task in C.TASKS:
        for split in ("valid", "test"):
            key = f"{task}/{split}"
            if key in bootstrap:
                b = bootstrap[key]
                inc[key] = {
                    "diff_rmse_R2_minus_R1": b["diff"],
                    "ci95": b["ci95"],
                    "ratio": b["ratio"],
                    "prob_R2_better": b["prob_R2_better"],
                    "excludes_zero": bool(b["ci95"][0] > 0 or b["ci95"][1] < 0),
                    "favours_R2": bool(b["diff"] < 0),
                }
    incremental_signal = any(
        v["favours_R2"] and v["excludes_zero"] and v["prob_R2_better"] >= 0.95 for v in inc.values()
    )
    no_regression = all((not v["favours_R2"]) or (v["prob_R2_better"] >= 0.05) for v in inc.values())
    oracle_signal = {}
    for model in C.MODELS:
        for split in ("valid", "test"):
            meta = oracle_meta.get(f"{model}/{split}")
            if meta is None:
                continue
            oracle_signal[f"{model}/{split}"] = {
                "matched_fraction": meta["matched_fraction"],
                "mean_enrichment": meta["mean_enrichment_matched"],
                "fraction_positive": meta["fraction_enrichment_positive"],
                "n_matched": meta["n_matched"],
            }
    oracle_positive = any(
        (v["n_matched"] >= C.MIN_ORACLE_CONTROLS)
        and (v["fraction_positive"] is not None)
        and np.isfinite(v["fraction_positive"])
        and v["fraction_positive"] >= 0.60
        and v["mean_enrichment"] > 0
        for v in oracle_signal.values()
    )

    flags: dict[str, Any] = {}
    flags["partition_gate"] = support["gate"]["status"]
    capacity_ratio = float(budget["ratio"]["r2_over_r1"])
    flags["CAPACITY_CONFOUNDED"] = bool(capacity_ratio > C.CAPACITY_CONFOUNDED_RATIO)
    if not any_retention:
        flags["PREDICTION_RETENTION_FAIL"] = True
    if not incremental_signal:
        flags["INTERFACE_CONTEXT_INCREMENT_NOT_SHOWN"] = True
    if incremental_signal and not flags["CAPACITY_CONFOUNDED"]:
        flags["INTERFACE_CONTEXT_PROMISING"] = True
    flags["ORACLE_LOCALISATION_SIGNAL"] = bool(oracle_positive)
    flags["RHO_DOMINATES_STRUCTURE"] = bool(
        any(v["rho_over_structural"] > 4.0 for v in rho_dominance.values())
    )
    flags["TEST_USED_FOR_SELECTION"] = False
    flags["OLD_MODELS_RETRAINED"] = False

    next_stage = "STOP"
    if any_retention and (oracle_positive or incremental_signal):
        next_stage = "CONTINUE_WITH_STRUCTURE_ROUTING" if not incremental_signal else "CONTINUE_WITH_INTERFACE_CONTEXT"
    elif not any_retention:
        next_stage = "PAUSE_FIXED_TOPOLOGY_PARTITION"

    return {
        "flags": {k: v for k, v in flags.items() if isinstance(v, (bool, str))},
        "retention": retention,
        "retention_threshold": C.RETENTION_FACTOR,
        "retention_ok": retention_ok,
        "baselines": {"c1_lipo_rf": c1, "gine_g0_lipo_rf": gine, "c1_wc_logp_r3": c1_wc},
        "interface_increment": inc,
        "oracle_signal": oracle_signal,
        "rho_dominance": rho_dominance,
        "rho_only": rho_only,
        "capacity": {
            "params_R1": budget["models"]["R1"]["total"],
            "params_R2": budget["models"]["R2"]["total"],
            "params_C1": budget["c1_reference_total"],
            "r2_over_r1": capacity_ratio,
            "threshold": C.CAPACITY_CONFOUNDED_RATIO,
        },
        "next_stage_recommendation": next_stage,
        "propositions": {
            "A_structure_routing_viable": {
                "retention_ok": retention_ok,
                "any_retention": bool(any_retention),
                "rho_dominance": rho_dominance,
            },
            "B_interface_context_increment": {
                "incremental_signal": bool(incremental_signal),
                "per_task_split": inc,
                "capacity_confounded": bool(flags["CAPACITY_CONFOUNDED"]),
            },
            "C_meaningful_structural_explanation": {
                "oracle_positive": bool(oracle_positive),
                "oracle_signal": oracle_signal,
                "native_ledger_closes": all(
                    v["max_ledger_closure_error"] < C.TRACE_TOL_BASE for v in block_scale.values()
                ),
                "note": "ledger closure is a property of the architecture, not evidence of chemical correctness",
            },
            "D_next_stage": {"recommendation": next_stage},
        },
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def run_manifest(started: str) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(C.RESULTS_DIR.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(C.ROUND_DIR))] = {
                "bytes": int(path.stat().st_size),
                "sha256": C.sha256_file(path),
            }
    frozen_inputs = {}
    for path in (
        C.WC_LOG_PRED_CSV,
        C.ORACLE_CONTRIB_NPZ,
        C.ORACLE_CONTRIB_SUMMARY,
        C.ROUTE_ANALYSIS / "tables/stage1.csv",
        C.C1_LIPO_TRAJECTORY,
        C.WC_LOG_TRAIN_JSON,
    ):
        frozen_inputs[str(path.relative_to(C.REPO_ROOT))] = C.sha256_file(path) if path.is_file() else None
    routing = {
        split: {
            "path": str(C.routing_cache_path(split).relative_to(C.ROUND_DIR)),
            "sha256": C.sha256_file(C.routing_cache_path(split)),
        }
        for split in C.SPLITS
        if C.routing_cache_path(split).is_file()
    }
    return {
        "round_id": C.ROUND_ID,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": {
            "commit": C.git_commit(),
            "branch": C.git_branch(),
            "dirty_tracked": C.git_dirty_tracked(),
            "lockfile_sha256": C.lockfile_sha256(),
        },
        "code_sha256": T.code_hashes(),
        "routing_cache": routing,
        "frozen_inputs": frozen_inputs,
        "results_files": files,
        "environment": C.environment_snapshot(),
        "training_units": {
            f"{task}/{model}": {
                "trajectory": str((C.task_results_dir(task, model) / "trajectory.json").relative_to(C.ROUND_DIR)),
                "exists": (C.task_results_dir(task, model) / "trajectory.json").is_file(),
            }
            for task in C.TASKS
            for model in C.MODELS
        },
    }


__all__ = [
    "historical_baselines",
    "load_run",
    "gating_table",
    "oracle_localisation",
    "paired_bootstrap",
    "paired_bootstrap_ratio",
    "aggregate",
    "build_decision",
    "run_manifest",
]
