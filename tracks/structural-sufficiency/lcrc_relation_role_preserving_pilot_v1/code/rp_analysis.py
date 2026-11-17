"""Aggregate stage: predictions, bootstrap, gamma ablation, oracle localisation, decision.

Everything here is computed from the four frozen training units (best-Valid
checkpoints), the frozen data/routing caches and the read-only historical
numbers.  No model is retrained, no epoch is re-selected and the official Test
split is only ever read as ``DEVELOPMENT_TEST``.

Outputs (all under ``analysis/``):

* ``prediction_comparison.csv`` — every unit's Train/Valid/Test RMSE/MAE plus the
  read-only historical baselines with an explicit comparability label;
* ``retention.json`` — the PREDICTION_PRESERVED arithmetic against C1(Rf);
* ``bootstrap_real_vs_control.json`` — paired molecule-level bootstrap of
  RMSE(REAL) - RMSE(CONTROL) on identical rows;
* ``gamma_analysis.json`` — the learned gamma vectors and the frozen-model
  ablation response (learned / both one / each one / swap);
* ``oracle_localisation.csv`` + ``oracle_localisation_summary.json`` — WC_LOGP
  candidate capture versus the identical matched random-connected-subgraph
  controls of the previous round;
* ``mask_response_wc_logp.csv`` + ``mask_response_summary.json`` — both models'
  native block-mask response on the *same* candidate set;
* ``predictions_<task>_<model>.csv`` — molecule-level predictions (Valid/Test);
* ``decision.json`` — the pre-declared decision flags and verdict;
* ``run_manifest.json`` — environment, commit, hashes.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rp_common as C
import rp_data as D
import rp_models as M
import rp_train as T

MOD = C.modules()


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


def load_unit(task: str, variant: str) -> dict[str, Any] | None:
    paths = C.prediction_paths(task, variant)
    if not paths["trajectory"].is_file():
        return None
    traj = C.read_json(paths["trajectory"])
    preds: dict[str, dict[str, np.ndarray]] = {}
    for split in C.SPLITS:
        path = paths[f"best_{split}"]
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as mesh:
            preds[split] = {key: np.asarray(mesh[key]) for key in mesh.files}
    return {
        "task": task,
        "variant": variant,
        "trajectory": traj,
        "predictions": preds,
        "checkpoint": paths["best_checkpoint"],
        "gamma": _load_npz(C.task_results_dir(task, variant) / "gamma_best.npz"),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as mesh:
        return {key: np.asarray(mesh[key]) for key in mesh.files}


# ---------------------------------------------------------------------------
# historical baselines (read only; never retrained)
# ---------------------------------------------------------------------------


def historical_baselines() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparable: list[dict[str, Any]] = []
    appendix: list[dict[str, Any]] = []
    if C.ROUTE_STAGE1.is_file():
        for row in C.read_csv(C.ROUTE_STAGE1):
            if row.get("dataset") != C.DATASET or row.get("recipe") != "Rf":
                continue
            name = {"L": "C1", "G0": "GINE-G0", "Gx": "GINE-Gx"}.get(row.get("model"), row.get("model"))
            for split in C.SPLITS:
                comparable.append(
                    {
                        "task": "lipo",
                        "model": f"{name} (frozen, Rf)",
                        "variant": "",
                        "split": split,
                        "rmse": float(row[f"{split}_rmse_raw"]),
                        "mae": (
                            float(row[f"{split}_mae_raw"])
                            if row.get(f"{split}_mae_raw") not in (None, "")
                            else None
                        ),
                        "epoch": int(float(row["selected_epoch"])),
                        "n_parameters": int(float(row["n_parameters"])),
                        "source": "tracks/structural-sufficiency/gnn_route_clarification_8h_v1/analysis/tables/stage1.csv",
                        "comparability": "DIRECTLY_COMPARABLE",
                        "note": "same dataset/split/target/scaler/seed 0/recipe Rf/canonical C1 init",
                    }
                )
    if C.WC_LOG_TRAIN_JSON.is_file():
        payload = C.read_json(C.WC_LOG_TRAIN_JSON)
        for split in ("valid", "test"):
            m = payload["metrics"]["best"][split]
            comparable.append(
                {
                    "task": "wc_logp",
                    "model": "C1 (frozen, R3)",
                    "variant": "",
                    "split": split,
                    "rmse": float(m["rmse"]),
                    "mae": float(m["mae"]),
                    "epoch": int(payload["checkpoint"]["best_epoch"]),
                    "n_parameters": int(payload["model"]["parameters"]),
                    "source": "tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs/wc_logp/train.json",
                    "comparability": "RECIPE_DIFFERS",
                    "note": "same split/target/scaler/seed; recipe R3 (peak 3e-4, wd 1e-3) vs Rf here",
                }
            )
    # the previous architecture round is *not* comparable: different architecture
    # and a different pre-registered protocol.  Kept in an appendix, never in the
    # main comparison and never in a decision rule.
    for task in ("lipo", "wc_logp"):
        for variant in ("R1", "R2"):
            path = C.PRED_ANALYSIS / f"trajectory_{task}_{variant}.json"
            if not path.is_file():
                continue
            payload = C.read_json(path)
            for split in ("valid", "test"):
                m = payload["selected"] if split == "valid" else payload["selected"]
                appendix.append(
                    {
                        "task": task,
                        "model": f"{variant} (previous round, frozen)",
                        "variant": variant,
                        "split": split,
                        "rmse": float(m[f"{split}_rmse_raw"]),
                        "mae": float(m[f"{split}_mae_raw"]),
                        "epoch": int(m["epoch"]),
                        "n_parameters": int(payload["n_parameters"]),
                        "source": f"tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/analysis/trajectory_{task}_{variant}.json",
                        "comparability": "NOT_DIRECTLY_COMPARABLE",
                        "note": "different architecture (block/interface tokens, different readout) and a different protocol round; appendix only",
                    }
                )
    return comparable, appendix


def prediction_comparison(units: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (task, variant), unit in sorted(units.items()):
        traj = unit["trajectory"]
        for split in C.SPLITS:
            pred = unit["predictions"][split]
            rows.append(
                {
                    "task": task,
                    "model": variant,
                    "variant": variant,
                    "split": split,
                    "rmse": C.rmse(pred["pred_raw"], pred["y_raw"]),
                    "mae": C.mae(pred["pred_raw"], pred["y_raw"]),
                    "epoch": int(traj["selected"]["epoch"]),
                    "n_parameters": int(traj["n_parameters"]),
                    "source": f"current round: results/{task}/{variant}",
                    "comparability": "CURRENT_ROUND",
                    "note": "best-Valid checkpoint, Test = DEVELOPMENT_TEST",
                }
            )
    comparable, appendix = historical_baselines()
    C.ensure_dir(C.ANALYSIS_DIR)
    C.write_csv(C.ANALYSIS_DIR / "prediction_comparison.csv", rows + comparable)
    C.write_csv(C.ANALYSIS_DIR / "appendix_not_comparable.csv", appendix)
    return {"rows": rows, "historical": comparable, "appendix": appendix}


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------


def retention(units: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rule": f"RR_REAL Lipophilicity Valid RMSE <= {C.RETENTION_FACTOR} x C1(Rf) Valid RMSE",
        "factor": C.RETENTION_FACTOR,
        "baseline": None,
        "models": {},
    }
    base_row = None
    if C.ROUTE_STAGE1.is_file():
        for row in C.read_csv(C.ROUTE_STAGE1):
            if row.get("dataset") == C.DATASET and row.get("recipe") == "Rf" and row.get("model") == "L":
                base_row = row
                break
    if base_row is None:
        return out
    baseline = {
        "model": "C1 (frozen, Rf)",
        "valid_rmse": float(base_row["valid_rmse_raw"]),
        "test_rmse": float(base_row["test_rmse_raw"]),
        "train_rmse": float(base_row["train_rmse_raw"]),
        "valid_mae": float(base_row["valid_mae_raw"]),
        "test_mae": float(base_row["test_mae_raw"]),
        "epoch": int(float(base_row["selected_epoch"])),
        "n_parameters": int(float(base_row["n_parameters"])),
        "source": "tracks/structural-sufficiency/gnn_route_clarification_8h_v1/analysis/tables/stage1.csv",
        "comparability": "DIRECTLY_COMPARABLE",
    }
    out["baseline"] = baseline
    for variant in C.MODELS:
        unit = units.get(("lipo", variant))
        if unit is None:
            out["models"][variant] = {"available": False}
            continue
        valid = C.rmse(unit["predictions"]["valid"]["pred_raw"], unit["predictions"]["valid"]["y_raw"])
        test = C.rmse(unit["predictions"]["test"]["pred_raw"], unit["predictions"]["test"]["y_raw"])
        ratio_valid = valid / baseline["valid_rmse"]
        out["models"][variant] = {
            "available": True,
            "valid_rmse": valid,
            "test_rmse": test,
            "valid_ratio_over_c1": ratio_valid,
            "test_ratio_over_c1": test / baseline["test_rmse"],
            "valid_abs_gap": valid - baseline["valid_rmse"],
            "test_abs_gap": test - baseline["test_rmse"],
            "selected_epoch": int(unit["trajectory"]["selected"]["epoch"]),
            "passes_retention": bool(ratio_valid <= C.RETENTION_FACTOR),
        }
    flags = {v: bool(out["models"].get(v, {}).get("passes_retention")) for v in C.MODELS}
    out["prediction_preserved"] = flags["RR_REAL"]
    out["control_also_preserved"] = flags["RR_CONTROL"]
    out["both_preserved"] = bool(flags["RR_REAL"] and flags["RR_CONTROL"])
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(real: Mapping[str, np.ndarray], control: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Paired molecule-level bootstrap of ``RMSE(REAL) - RMSE(CONTROL)``.

    Both models are evaluated on exactly the same molecules (asserted by the
    caller).  A negative difference means REAL is better.
    """
    y = np.asarray(real["y_raw"], dtype=np.float64)
    pr = np.asarray(real["pred_raw"], dtype=np.float64)
    pc = np.asarray(control["pred_raw"], dtype=np.float64)
    if not np.array_equal(np.asarray(real["row_id"]), np.asarray(control["row_id"])):
        raise RuntimeError("paired bootstrap requires identical row ids")
    if y.shape != control["y_raw"].shape or not np.allclose(y, np.asarray(control["y_raw"], dtype=np.float64)):
        raise RuntimeError("paired bootstrap requires identical targets")
    er = pr - y
    ec = pc - y
    n = int(er.size)
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    diffs = np.empty(C.BOOTSTRAP_DRAWS)
    ratios = np.empty(C.BOOTSTRAP_DRAWS)
    for d in range(C.BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, size=n)
        r_real = float(np.sqrt(np.mean(er[idx] ** 2)))
        r_ctrl = float(np.sqrt(np.mean(ec[idx] ** 2)))
        diffs[d] = r_real - r_ctrl
        ratios[d] = r_real / r_ctrl if r_ctrl else float("nan")
    point_real = float(np.sqrt(np.mean(er ** 2)))
    point_ctrl = float(np.sqrt(np.mean(ec ** 2)))
    return {
        "n": n,
        "rmse_real": point_real,
        "rmse_control": point_ctrl,
        "diff_real_minus_control": point_real - point_ctrl,
        "ratio_real_over_control": point_real / point_ctrl if point_ctrl else float("nan"),
        "ci95_diff": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "ci95_ratio": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "prob_real_better": float(np.mean(diffs < 0.0) + 0.5 * np.mean(diffs == 0.0)),
        "draws": int(C.BOOTSTRAP_DRAWS),
        "seed": int(C.BOOTSTRAP_SEED),
        "positive_ci_excludes_zero_in_favour_of_real": bool(np.percentile(diffs, 97.5) < 0.0),
    }


def bootstrap_all(units: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "draws": int(C.BOOTSTRAP_DRAWS),
        "seed": int(C.BOOTSTRAP_SEED),
        "convention": "diff = RMSE(REAL) - RMSE(CONTROL); negative favours RR_REAL",
        "comparisons": {},
    }
    for task in C.TASKS:
        if (task, "RR_REAL") not in units or (task, "RR_CONTROL") not in units:
            continue
        real = units[(task, "RR_REAL")]["predictions"]
        control = units[(task, "RR_CONTROL")]["predictions"]
        for split in C.SPLITS:
            out["comparisons"][f"{task}/{split}"] = paired_bootstrap(real[split], control[split])
    return out


# ---------------------------------------------------------------------------
# gamma: learned values and frozen-model ablation response
# ---------------------------------------------------------------------------


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def gamma_ablation(unit: dict[str, Any], bundle, device: str = "cpu") -> dict[str, Any]:
    import torch

    task, variant = unit["task"], unit["variant"]
    model, payload = T.load_model_from_checkpoint(bundle, task, variant, unit["checkpoint"], device="cpu")
    model = model.to(device).eval()
    gi = model.gamma_internal.detach().clone()
    gc = model.gamma_cross.detach().clone()
    ones = torch.ones_like(gi)
    settings = {
        "learned": (gi, gc),
        "both_one": (ones, ones),
        "internal_one": (ones, gc),
        "cross_one": (gi, ones),
        "swap": (gc, gi),
    }
    out: dict[str, Any] = {"task": task, "variant": variant, "epoch": int(payload["epoch"]), "ablations": {}}
    for split in ("valid", "test"):
        out["ablations"][split] = {}
        for name, (a, b) in settings.items():
            res = T.evaluate(model, bundle, task, split, device, gammas=(a, b))
            learned = out["ablations"][split].get("learned")
            out["ablations"][split][name] = {
                "rmse": float(res["rmse_raw"]),
                "mae": float(res["mae_raw"]),
                "delta_rmse_vs_learned": None if learned is None else float(res["rmse_raw"] - learned["rmse"]),
            }
    g = {k: np.asarray(v, dtype=np.float64) for k, v in model.role_state().items()}
    out["gamma"] = {}
    for name, value in g.items():
        dev = np.abs(value - 1.0)
        order = np.argsort(-dev)[:8]
        out["gamma"][name] = {
            "mean": float(value.mean()),
            "std": float(value.std()),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean_abs_dev_from_one": float(dev.mean()),
            "max_abs_dev_from_one": float(dev.max()),
            "fraction_abs_dev_gt_0.05": float((dev > 0.05).mean()),
            "top_channels": [
                {"channel": int(i), "gamma": float(value[i]), "abs_dev": float(dev[i])} for i in order
            ],
        }
    out["gamma_pair_correlation"] = _safe_corr(g["gamma_internal"], g["gamma_cross"])
    out["gamma_abs_dev_ratio_cross_over_internal"] = float(
        np.abs(g["gamma_cross"] - 1.0).mean() / max(1e-12, np.abs(g["gamma_internal"] - 1.0).mean())
    )
    out["role_response"] = {}
    for split in ("valid", "test"):
        res = T.evaluate(model, bundle, task, split, device, gammas=(gi, gc))
        base = T.evaluate(model, bundle, task, split, device, gammas=(ones, ones))
        d = np.abs(np.asarray(res["pred_raw"]) - np.asarray(base["pred_raw"]))
        out["role_response"][split] = {
            "mean_abs_delta_pred_both_one_vs_learned": float(d.mean()),
            "median_abs_delta_pred_both_one_vs_learned": float(np.median(d)),
            "max_abs_delta_pred_both_one_vs_learned": float(d.max()),
            "rmse_change_both_one_minus_learned": float(base["rmse_raw"] - res["rmse_raw"]),
        }
    return out


# ---------------------------------------------------------------------------
# WC_LOGP closure
# ---------------------------------------------------------------------------


def wc_logp_closure(units: Mapping[tuple[str, str], dict[str, Any]], bundle) -> dict[str, Any]:
    out: dict[str, Any] = {"label_source": str(C.ORACLE_CONTRIB_NPZ), "splits": {}, "models": {}}
    with np.load(C.ORACLE_CONTRIB_NPZ, allow_pickle=False) as mesh:
        graph_values = np.asarray(mesh["WC_LOGP"], dtype=np.float64)
        atom_values = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    for split in C.SPLITS:
        ids = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
        labels = np.asarray(bundle.targets("wc_logp", split), dtype=np.float64)
        atom_gap = 0.0
        for row, gid in enumerate(ids.tolist()):
            a0, a1 = int(atom_ptr[gid]), int(atom_ptr[gid + 1])
            atom_gap = max(atom_gap, abs(float(atom_values[a0:a1].sum()) - float(labels[row])))
        out["splits"][split] = {
            "n_rows": int(ids.size),
            "max_abs_graph_minus_label": float(np.abs(graph_values[ids] - labels).max()),
            "max_abs_atom_sum_minus_label": float(atom_gap),
        }
    for variant in C.MODELS:
        unit = units.get(("wc_logp", variant))
        if unit is None:
            continue
        entry: dict[str, Any] = {}
        for split in ("valid", "test"):
            ledger_path = C.task_results_dir("wc_logp", variant) / f"best_ledger_{split}.npz"
            if not ledger_path.is_file():
                entry[split] = {"available": False}
                continue
            led = _load_npz(ledger_path)
            # the readout closes in standardised units (pred_std = sum_v f + rho);
            # in raw units the frozen target scaler adds the constant mean(y_train),
            # which is stored explicitly so the identity is auditable row by row
            bias = float(led["bias_raw"][0]) if "bias_raw" in led else float(bundle.scaler("wc_logp")["mean"])
            gap = led["pred_raw"] - (led["local_sum_raw"] + led["rho_raw"] + bias)
            entry[split] = {
                "available": True,
                "n_rows": int(led["pred_raw"].size),
                "bias_raw": bias,
                "max_abs_readout_closure_gap": float(np.abs(gap).max()),
                "max_abs_gap_without_bias": float(
                    np.abs(led["pred_raw"] - (led["local_sum_raw"] + led["rho_raw"])).max()
                ),
                "max_abs_pred_minus_saved_prediction": float(
                    np.abs(led["pred_raw"] - unit["predictions"][split]["pred_raw"]).max()
                ),
                "rmse_recomputed": C.rmse(led["pred_raw"], led["y_raw"]),
            }
        out["models"][variant] = entry
    return out


# ---------------------------------------------------------------------------
# random connected subgraph controls (identical to the previous round)
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


def _block_deltas(model, bundle, split: str, device: str = "cpu", chunk_size: int = 8) -> list[dict[str, Any]]:
    """Native block-mask response of one frozen model on one split.

    For every frozen block ``i`` of every molecule: ``delta_i = y(all) - y(mask i)``
    where the mask drops exactly the S2 occurrences owned (fully or partly) by
    block ``i``.  The prediction is recomputed through the whole model, so the
    response includes the local input, the readout and ``rho``.
    """
    import torch

    order, row_ids, targets = T.eval_rows(bundle, "wc_logp", split)
    cache = bundle.caches[split]
    routing = bundle.routing[split]
    blk_n_bonds = np.asarray(routing["blk_n_bonds"], dtype=np.int64)
    block_ptr = np.asarray(routing["block_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    out: list[dict[str, Any]] = []
    role_key = C.MODEL_ROLE_KEY[model.variant]
    with torch.no_grad():
        for start in range(0, order.size, chunk_size):
            chunk = order[start : start + chunk_size]
            batch = bundle.collate(split, chunk)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            prep = model.prepare(batch)
            pred_all = model.predict(prep, batch, role_key=role_key)
            counts = np.asarray(batch["blocks_per_graph"].detach().cpu().numpy(), dtype=np.int64)
            offs = np.asarray(batch["block_off"].detach().cpu().numpy(), dtype=np.int64)
            owner1 = batch["path_owner1_local"]
            owner2 = batch["path_owner2_local"]
            for g in range(int(chunk.size)):
                row = int(chunk[g])
                deltas = np.zeros(int(counts[g]), dtype=np.float64)
                for i in range(int(counts[g])):
                    block = int(offs[g] + i)
                    keep = M.block_keep(owner1, owner2, block)
                    pred = model.predict(prep, batch, role_key=role_key, keep_occ=keep)
                    deltas[i] = float(pred_all[g].item() - pred[g].item())
                gi = int(block_ptr[row])
                nb = blk_n_bonds[gi : gi + int(counts[g])]
                score = np.abs(deltas) / np.sqrt(np.maximum(1, nb).astype(np.float64))
                selected = int(np.argmax(score)) if score.size else -1
                out.append(
                    {
                        "cache_row": row,
                        "row_id": int(row_ids[start + g]),
                        "y_raw": float(targets[start + g]),
                        "pred_raw": float(pred_all[g].item()),
                        "n_blocks": int(counts[g]),
                        "block_ptr": gi,
                        "n_bonds": nb.astype(np.int64),
                        "delta": deltas,
                        "score": score,
                        "selected_local": selected,
                        "selected_score": float(score[selected]) if selected >= 0 else 0.0,
                        "n_atoms": int(atom_ptr[row + 1] - atom_ptr[row]),
                        "bond_range": (int(bond_ptr[row]), int(bond_ptr[row + 1])),
                    }
                )
    return out


def _oracle_capture(
    bundle, split: str, row: int, block_local: int
) -> dict[str, Any] | None:
    """Oracle mass capture of one frozen block and its matched random controls."""
    cache = bundle.caches[split]
    routing = bundle.routing[split]
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    blk_bond_ptr = np.asarray(routing["blk_bond_ptr"], dtype=np.int64)
    blk_bond = np.asarray(routing["blk_bond"], dtype=np.int64)
    belong_ptr = np.asarray(routing["belong_ptr"], dtype=np.int64)
    belong_block = np.asarray(routing["belong_block"], dtype=np.int64)
    belong_weight = np.asarray(routing["belong_weight"], dtype=np.float64)
    block_ptr = np.asarray(routing["block_ptr"], dtype=np.int64)
    row_id = int(bundle.global_row_ids[split][row])
    with np.load(C.ORACLE_CONTRIB_NPZ, allow_pickle=False) as mesh:
        atoms_value = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr_oracle = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    o0 = int(atom_ptr_oracle[row_id])
    atom_abs = np.abs(atoms_value[o0 : o0 + (a1 - a0)])
    total_mass = float(atom_abs.sum())
    edges = [(int(bond_index[j, 0]), int(bond_index[j, 1])) for j in range(b0, b1)]
    base = int(block_ptr[row])
    if block_local < 0:
        return None
    block_global = base + int(block_local)
    bb0, bb1 = int(blk_bond_ptr[block_global]), int(blk_bond_ptr[block_global + 1])
    block_bonds = [int(x) - b0 for x in blk_bond[bb0:bb1]]
    block_atoms = sorted({int(v) for e in block_bonds for v in edges[e]})
    capture = float(atom_abs[block_atoms].sum() / total_mass) if total_mass > 0 else 0.0
    rng = np.random.default_rng([C.ORACLE_CONTROL_SEED, row_id])
    controls: list[float] = []
    seen: set[frozenset] = {frozenset(block_bonds)}
    attempts = 0
    while len(controls) < C.MAX_ORACLE_CONTROLS and attempts < 40 * C.MAX_ORACLE_CONTROLS:
        attempts += 1
        candidate = _connected_subgraph(a1 - a0, edges, len(block_bonds), len(block_atoms), rng)
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        ctrl_atoms = sorted({int(v) for e in candidate for v in edges[e]})
        controls.append(float(atom_abs[ctrl_atoms].sum() / total_mass) if total_mass > 0 else 0.0)
    matched = len(controls) >= C.MIN_ORACLE_CONTROLS
    control_mean = float(np.mean(controls)) if controls else float("nan")
    alloc = np.zeros(int(block_ptr[row + 1] - block_ptr[row]), dtype=np.float64)
    for v in range(a1 - a0):
        lo, hi = int(belong_ptr[a0 + v]), int(belong_ptr[a0 + v + 1])
        for kk in range(lo, hi):
            alloc[int(belong_block[kk])] += belong_weight[kk] * atom_abs[v]
    return {
        "row_id": row_id,
        "cache_row": int(row),
        "selected_block": int(block_local),
        "selected_n_bonds": int(len(block_bonds)),
        "selected_n_atoms": int(len(block_atoms)),
        "capture": capture,
        "control_mean": control_mean,
        "enrichment": capture - control_mean,
        "enrichment_ratio": (capture / control_mean) if control_mean > 0 else float("nan"),
        "n_controls": len(controls),
        "matched": bool(matched),
        "oracle_mass": total_mass,
        "allocated_mass": float(alloc.sum()),
        "allocated_gap": float(abs(alloc.sum() - total_mass)),
        "selected_allocated_mass": float(alloc[int(block_local)]),
    }


def _bootstrap_mean(values: Sequence[float], *, draws: int = C.BOOTSTRAP_DRAWS, seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    medians = np.empty(draws)
    for d in range(draws):
        idx = rng.integers(0, arr.size, size=arr.size)
        means[d] = float(arr[idx].mean())
        medians[d] = float(np.median(arr[idx]))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "positive_fraction": float((arr > 0).mean()),
        "ci95_mean": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "ci95_median": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "draws": int(draws),
        "seed": int(seed),
    }


def _paired_mean(values_a: Sequence[float], values_b: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return {"n": int(a.size)}
    diff = a - b
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    means = np.empty(C.BOOTSTRAP_DRAWS)
    for d in range(C.BOOTSTRAP_DRAWS):
        idx = rng.integers(0, diff.size, size=diff.size)
        means[d] = float(diff[idx].mean())
    return {
        "n": int(diff.size),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "ci95_mean_diff": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "prob_a_better": float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0)),
        "fraction_a_positive": float((a > 0).mean()),
        "fraction_b_positive": float((b > 0).mean()),
        "draws": int(C.BOOTSTRAP_DRAWS),
        "seed": int(C.BOOTSTRAP_SEED),
    }


def _rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = _rank(a)
    rb = _rank(b)
    if ra.size < 2 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def oracle_localisation(units: Mapping[tuple[str, str], dict[str, Any]], bundle, device: str = "cpu") -> dict[str, Any]:
    """Both models' native candidates on the frozen block pool, identical controls.

    The candidate pool is exactly the frozen topology block set, the selection
    score is the model's own native S2 block-mask response, and the random
    connected-subgraph controls are generated by the identical function, seed
    and matching rule as the previous round.
    """
    if not any(("wc_logp", v) in units for v in C.MODELS):
        return {"available": False}
    deltas: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in C.MODELS:
        unit = units.get(("wc_logp", variant))
        if unit is None:
            continue
        model, _ = T.load_model_from_checkpoint(bundle, "wc_logp", variant, unit["checkpoint"], device="cpu")
        model = model.to(device).eval()
        deltas[variant] = {}
        for split in ("valid", "test"):
            deltas[variant][split] = _block_deltas(model, bundle, split, device)

    rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "available": True,
        "candidate_pool": "the frozen topology blocks of lcrc_relation_interface_routing_pilot_v1",
        "selection_score": "argmax_i |delta_i| / sqrt(max(1, n_bonds_i)); ties -> smallest block index",
        "mask_semantics": "an S2 occurrence survives iff neither of its two owning blocks is masked",
        "controls": {
            "generator": "identical _connected_subgraph as the previous round",
            "seed": int(C.ORACLE_CONTROL_SEED),
            "min": int(C.MIN_ORACLE_CONTROLS),
            "max": int(C.MAX_ORACLE_CONTROLS),
            "matched_on": "bond count and atom count of the candidate",
        },
        "models": {},
    }
    for variant in C.MODELS:
        if variant not in deltas:
            continue
        entry: dict[str, Any] = {}
        for split in ("valid", "test"):
            recs: list[dict[str, Any]] = []
            for rec in deltas[variant][split]:
                cand = int(rec["selected_local"])
                cap = _oracle_capture(bundle, split, int(rec["cache_row"]), cand)
                if cap is None:
                    continue
                cap.update(
                    {
                        "task": "wc_logp",
                        "model": variant,
                        "split": split,
                        "selected_score": float(rec["selected_score"]),
                        "n_blocks": int(rec["n_blocks"]),
                        "pred_raw": float(rec["pred_raw"]),
                        "y_raw": float(rec["y_raw"]),
                    }
                )
                recs.append(cap)
                rows.append(dict(cap))
            matched = [r for r in recs if r["matched"]]
            enrich = [r["enrichment"] for r in matched]
            entry[split] = {
                "n_molecules": len(recs),
                "n_matched": len(matched),
                "matched_fraction": (len(matched) / len(recs)) if recs else 0.0,
                "enrichment": _bootstrap_mean(enrich),
                "capture": _bootstrap_mean([r["capture"] for r in matched]),
                "control_mean_of_means": float(np.mean([r["control_mean"] for r in matched])) if matched else float("nan"),
                "mean_selected_n_bonds": float(np.mean([r["selected_n_bonds"] for r in matched])) if matched else float("nan"),
                "mean_oracle_mass": float(np.mean([r["oracle_mass"] for r in matched])) if matched else float("nan"),
                "allocation_max_abs_gap": float(max((r["allocated_gap"] for r in recs), default=0.0)),
                "n_bonds_nonzero": int(sum(1 for r in matched if r["selected_n_bonds"] > 0)),
            }
            signal = (
                entry[split]["enrichment"].get("n", 0) >= C.MIN_ORACLE_CONTROLS
                and entry[split]["enrichment"]["mean"] > 0
                and entry[split]["enrichment"]["positive_fraction"] >= C.ORACLE_POSITIVE_FRACTION
            )
            entry[split]["oracle_signal"] = bool(signal)
        summary["models"][variant] = entry

    # same-candidate mask response: every candidate discovered by either model is
    # evaluated by BOTH models (same molecule, same block, same mask).
    for split in ("valid", "test"):
        by_row = {v: {int(r["row_id"]): r for r in deltas.get(v, {}).get(split, [])} for v in C.MODELS}
        for source in C.MODELS:
            for rec in deltas.get(source, {}).get(split, []):
                row_id = int(rec["row_id"])
                cand = int(rec["selected_local"])
                if cand < 0:
                    continue
                entry = {
                    "split": split,
                    "row_id": row_id,
                    "candidate_source": source,
                    "candidate_block": cand,
                    "n_bonds": int(rec["n_bonds"][cand]),
                }
                for target in C.MODELS:
                    target_rec = by_row.get(target, {}).get(row_id)
                    label = "real" if target == "RR_REAL" else "control"
                    entry[f"delta_{label}"] = (
                        float(target_rec["delta"][cand])
                        if target_rec is not None and cand < int(target_rec["delta"].size)
                        else float("nan")
                    )
                mask_rows.append(entry)

    # paired REAL - CONTROL enrichment on identical molecules
    paired: dict[str, Any] = {}
    for split in ("valid", "test"):
        a = {int(r["row_id"]): r for r in rows if r["model"] == "RR_REAL" and r["split"] == split and r["matched"]}
        b = {int(r["row_id"]): r for r in rows if r["model"] == "RR_CONTROL" and r["split"] == split and r["matched"]}
        shared = sorted(set(a) & set(b))
        paired[split] = _paired_mean(
            [a[i]["enrichment"] for i in shared], [b[i]["enrichment"] for i in shared]
        )
        paired[split]["n_shared"] = len(shared)
        paired[split]["real_oracle_signal"] = bool(
            summary["models"].get("RR_REAL", {}).get(split, {}).get("oracle_signal", False)
        )
        if len(shared) >= C.MIN_ORACLE_CONTROLS:
            lo, _hi = paired[split]["ci95_mean_diff"]
            paired[split]["positive_in_favour_of_real"] = bool(lo > 0.0)
            paired[split]["prob_real_enrichment_better"] = float(paired[split]["prob_a_better"])
        else:
            paired[split]["positive_in_favour_of_real"] = False
            paired[split]["prob_real_enrichment_better"] = float("nan")
    summary["paired_real_minus_control"] = paired

    # same-candidate mask response summary
    mask_summary: dict[str, Any] = {"pairs": {}, "rank_agreement": {}}
    for split in ("valid", "test"):
        sub = [r for r in mask_rows if r["split"] == split]
        for source in C.MODELS:
            src = [r for r in sub if r["candidate_source"] == source]
            if not src:
                continue
            ar = np.asarray([r["delta_real"] for r in src], dtype=np.float64)
            ac = np.asarray([r["delta_control"] for r in src], dtype=np.float64)
            mask_summary["pairs"][f"{split}/{source}"] = {
                "n": len(src),
                "mean_abs_delta_real": float(np.abs(ar).mean()),
                "mean_abs_delta_control": float(np.abs(ac).mean()),
                "median_abs_delta_real": float(np.median(np.abs(ar))),
                "median_abs_delta_control": float(np.median(np.abs(ac))),
                "paired_abs_delta_real_minus_control": _paired_mean(
                    np.abs(ar).tolist(), np.abs(ac).tolist()
                ),
            }
        # per-molecule rank agreement of the full |delta| vectors
        real_map = {r["row_id"]: r for r in deltas.get("RR_REAL", {}).get(split, [])}
        ctrl_map = {r["row_id"]: r for r in deltas.get("RR_CONTROL", {}).get(split, [])}
        shared = sorted(set(real_map) & set(ctrl_map))
        rhos = []
        same_argmax = 0
        for row_id in shared:
            dr = np.abs(real_map[row_id]["delta"])
            dc = np.abs(ctrl_map[row_id]["delta"])
            if dr.size >= 3 and dr.size == dc.size:
                rho = _spearman(dr, dc)
                if np.isfinite(rho):
                    rhos.append(rho)
            if int(real_map[row_id]["selected_local"]) == int(ctrl_map[row_id]["selected_local"]):
                same_argmax += 1
        mask_summary["rank_agreement"][split] = {
            "n_molecules": len(shared),
            "mean_spearman_abs_delta": float(np.mean(rhos)) if rhos else float("nan"),
            "median_spearman_abs_delta": float(np.median(rhos)) if rhos else float("nan"),
            "fraction_same_argmax": (same_argmax / len(shared)) if shared else float("nan"),
        }
    summary["mask_response"] = mask_summary

    C.write_csv(C.ANALYSIS_DIR / "oracle_localisation.csv", rows)
    C.write_csv(C.ANALYSIS_DIR / "mask_response_wc_logp.csv", mask_rows)
    C.atomic_write_json(C.ANALYSIS_DIR / "oracle_localisation_summary.json", summary)
    C.atomic_write_json(C.ANALYSIS_DIR / "mask_response_summary.json", mask_summary)
    return summary


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def build_decision(
    units: Mapping[tuple[str, str], dict[str, Any]],
    retention_info: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    gamma: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    flags["TEST_USED_FOR_SELECTION"] = False
    flags["OLD_MODELS_RETRAINED"] = False
    flags["PREDICTION_PRESERVED"] = bool(retention_info.get("prediction_preserved", False))
    flags["CONTROL_ALSO_PRESERVED"] = bool(retention_info.get("control_also_preserved", False))
    flags["PREDICTION_RECOVERED_BOTH"] = bool(retention_info.get("both_preserved", False))

    real = retention_info.get("models", {}).get("RR_REAL", {})
    control = retention_info.get("models", {}).get("RR_CONTROL", {})
    comps = bootstrap.get("comparisons", {})
    predictive = {}
    for split in C.SPLITS:
        key = f"lipo/{split}"
        if key not in comps:
            continue
        c = comps[key]
        ok = bool(c["ci95_diff"][1] < 0.0 and c["prob_real_better"] >= C.INCREMENT_PROB)
        predictive[split] = {
            "diff": c["diff_real_minus_control"],
            "ci95": c["ci95_diff"],
            "prob_real_better": c["prob_real_better"],
            "credit": ok,
            "n": c["n"],
        }
    flags["ROLE_INCREMENT_PREDICTIVE"] = bool(any(v["credit"] for v in predictive.values()))

    oracle_entry = oracle.get("paired_real_minus_control", {}) if oracle.get("available") else {}
    oracle_credit = {}
    for split in ("valid", "test"):
        e = oracle_entry.get(split, {})
        signal = bool(e.get("real_oracle_signal", False))
        ci = e.get("ci95_mean_diff", [None, None])
        ok = bool(
            signal
            and e.get("n_shared", 0) >= C.MIN_ORACLE_CONTROLS
            and ci[0] is not None
            and ci[0] > 0.0
            and e.get("prob_real_enrichment_better", 0.0) >= C.INCREMENT_PROB
        )
        oracle_credit[split] = {
            "real_oracle_signal": signal,
            "n_shared": int(e.get("n_shared", 0)),
            "diff_mean": e.get("mean_diff"),
            "ci95": ci,
            "prob_real_better": e.get("prob_real_enrichment_better"),
            "credit": ok,
        }
    flags["ROLE_INCREMENT_ORACLE"] = bool(any(v["credit"] for v in oracle_credit.values()))
    flags["ROLE_INCREMENT_SHOWN"] = bool(
        flags["PREDICTION_PRESERVED"]
        and (flags["ROLE_INCREMENT_PREDICTIVE"] or flags["ROLE_INCREMENT_ORACLE"])
    )
    flags["ONLY_C1_READOUT_RECOVERY"] = bool(
        flags["PREDICTION_RECOVERED_BOTH"] and not flags["ROLE_INCREMENT_SHOWN"]
    )
    flags["ROLE_NOT_SUPPORTED"] = bool(
        not flags["PREDICTION_PRESERVED"] and not flags["ROLE_INCREMENT_SHOWN"]
    )
    flags["ORACLE_LOCALISATION_SIGNAL"] = bool(
        any(
            oracle.get("models", {}).get(v, {}).get(split, {}).get("oracle_signal", False)
            for v in C.MODELS
            for split in ("valid", "test")
        )
    )

    complete = all((task, v) in units for task in C.TASKS for v in C.MODELS)
    if not complete:
        verdict = "UNRESOLVED"
    elif flags["ROLE_INCREMENT_SHOWN"]:
        verdict = "ROLE_INCREMENT_SHOWN"
    elif flags["ONLY_C1_READOUT_RECOVERY"]:
        verdict = "ONLY_C1_READOUT_RECOVERY"
    elif not flags["PREDICTION_PRESERVED"]:
        verdict = "ROLE_NOT_SUPPORTED"
    else:
        # RR_REAL passes retention, RR_CONTROL does not, and no pre-declared
        # increment statistic clears its threshold -> the numbers are suggestive
        # but not statistically sufficient.
        verdict = "UNRESOLVED"

    gamma_response = {
        variant: {
            "mean_abs_dev_from_one": {
                "internal": gamma.get(variant, {}).get("gamma", {}).get("gamma_internal", {}).get("mean_abs_dev_from_one"),
                "cross": gamma.get(variant, {}).get("gamma", {}).get("gamma_cross", {}).get("mean_abs_dev_from_one"),
            },
            "role_response": gamma.get(variant, {}).get("role_response"),
            "ablations": gamma.get(variant, {}).get("ablations"),
        }
        for variant in C.MODELS
    }
    return {
        "round_id": C.ROUND_ID,
        "verdict": verdict,
        "flags": flags,
        "retention": retention_info,
        "prediction_increment": predictive,
        "oracle_increment": oracle_credit,
        "gamma_response": gamma_response,
        "caveats": [
            "single training seed per unit; no multi-seed stability claim",
            "official Test is DEVELOPMENT_TEST: evaluated every epoch, never used for selection",
            "the historical C1(Rf) baseline is read-only and was not re-trained; the Lipophilicity comparison is DIRECTLY_COMPARABLE (same split/target/scaler/seed/recipe/canonical init)",
            "the WC_LOGP C1 baseline is recipe R3 and is flagged RECIPE_DIFFERS",
            "no historical architecture result of the previous round enters any decision rule",
            "oracle controls are matched on bond and atom count only, not on composition",
        ],
        "not_done": [
            "no historical model re-trained, re-selected or re-evaluated",
            "no new seed, no hyper-parameter search, no extra dataset",
            "no partition extension / learnable grouping",
            "no Test-based selection or tuning",
        ],
    }


def run_manifest(started: str) -> dict[str, Any]:
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
        "environment": C.environment_snapshot(),
        "inputs_sha256": {
            "routing_caches": dict(C.EXPECTED_ROUTING_CACHE_SHA256),
            "control_table": C.read_json(C.control_manifest_path())["sha256"]
            if C.control_manifest_path().is_file()
            else None,
            "stage1_table": C.sha256_file(C.ROUTE_STAGE1) if C.ROUTE_STAGE1.is_file() else None,
            "oracle_contrib": C.sha256_file(C.ORACLE_CONTRIB_NPZ) if C.ORACLE_CONTRIB_NPZ.is_file() else None,
            "c1_trajectory": C.sha256_file(C.C1_LIPO_TRAJECTORY) if C.C1_LIPO_TRAJECTORY.is_file() else None,
        },
        "checkpoints": {f"{t}/{v}": str(C.prediction_paths(t, v)["best_checkpoint"]) for t in C.TASKS for v in C.MODELS},
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def write_prediction_csv(units: Mapping[tuple[str, str], dict[str, Any]]) -> list[str]:
    paths = []
    for (task, variant), unit in sorted(units.items()):
        rows = []
        for split in ("valid", "test"):
            pred = unit["predictions"][split]
            for i in range(int(pred["row_id"].size)):
                rows.append(
                    {
                        "row_id": int(pred["row_id"][i]),
                        "split": split,
                        "task": task,
                        "model": variant,
                        "y_raw": float(pred["y_raw"][i]),
                        "pred_raw": float(pred["pred_raw"][i]),
                        "residual": float(pred["pred_raw"][i] - pred["y_raw"][i]),
                    }
                )
        path = C.ANALYSIS_DIR / f"predictions_{task}_{variant}.csv"
        C.write_csv(path, rows, columns=["row_id", "split", "task", "model", "y_raw", "pred_raw", "residual"])
        paths.append(str(path.relative_to(C.ROUND_DIR)))
    return paths


def write_molecule_records(oracle_rows: list[dict[str, Any]]) -> str | None:
    if not oracle_rows:
        return None
    path = C.ANALYSIS_DIR / "molecule_records_wc_logp.csv"
    C.write_csv(path, oracle_rows)
    return str(path.relative_to(C.ROUND_DIR))


def aggregate(device: str | None = None) -> dict[str, Any]:
    import torch

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.ensure_dir(C.ANALYSIS_DIR)
    bundle = D.build_bundle(verify=True)
    units = {}
    for task in C.TASKS:
        for variant in C.MODELS:
            unit = load_unit(task, variant)
            if unit is None:
                raise FileNotFoundError(f"missing unit {task}/{variant}; cannot aggregate")
            units[(task, variant)] = unit
            C.atomic_write_json(
                C.ANALYSIS_DIR / f"trajectory_{task}_{variant}.json",
                {
                    "task": task,
                    "variant": variant,
                    "purpose": "formal",
                    "selected": unit["trajectory"].get("selected"),
                    "final": unit["trajectory"].get("final"),
                    "last20": unit["trajectory"].get("last20"),
                    "n_parameters": unit["trajectory"].get("n_parameters"),
                    "signature": unit["trajectory"].get("signature"),
                    "status": unit["trajectory"].get("status"),
                    "device": unit["trajectory"].get("device"),
                    "environment": unit["trajectory"].get("environment"),
                },
            )
            curve_src = C.task_results_dir(task, variant) / "curve.csv"
            if curve_src.is_file():
                (C.ANALYSIS_DIR / f"curve_{task}_{variant}.csv").write_text(curve_src.read_text())

    comparison = prediction_comparison(units)
    retention_info = retention(units)
    bootstrap = bootstrap_all(units)
    closure = wc_logp_closure(units, bundle)
    gamma = {}
    for variant in C.MODELS:
        unit = units.get(("lipo", variant))
        if unit is not None:
            gamma[variant] = gamma_ablation(unit, bundle, device=device)
    oracle = oracle_localisation(units, bundle, device=device)
    pred_csv = write_prediction_csv(units)
    molecule_csv = "analysis/oracle_localisation.csv" if (C.ANALYSIS_DIR / "oracle_localisation.csv").is_file() else None
    decision = build_decision(units, retention_info, bootstrap, gamma, oracle)
    manifest = run_manifest(started)

    C.atomic_write_json(C.ANALYSIS_DIR / "retention.json", retention_info)
    C.atomic_write_json(C.ANALYSIS_DIR / "bootstrap_real_vs_control.json", bootstrap)
    C.atomic_write_json(C.ANALYSIS_DIR / "gamma_analysis.json", gamma)
    C.atomic_write_json(C.ANALYSIS_DIR / "wc_logp_closure.json", closure)
    C.atomic_write_json(C.ANALYSIS_DIR / "decision.json", decision)
    C.atomic_write_json(C.ANALYSIS_DIR / "run_manifest.json", manifest)
    return {
        "stage": "aggregate",
        "verdict": decision["verdict"],
        "flags": decision["flags"],
        "n_comparisons": len(comparison["rows"]),
        "prediction_csv": pred_csv,
        "molecule_records_csv": molecule_csv,
        "device": device,
    }


__all__ = [
    "load_unit",
    "historical_baselines",
    "prediction_comparison",
    "retention",
    "paired_bootstrap",
    "bootstrap_all",
    "gamma_ablation",
    "wc_logp_closure",
    "oracle_localisation",
    "build_decision",
    "aggregate",
    "run_manifest",
]
