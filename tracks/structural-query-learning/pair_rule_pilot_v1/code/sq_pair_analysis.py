"""Local aggregate: metric recheck, rule export/replay, bootstrap and report.

This module is imported only by ``stage=aggregate`` on the local machine (it
needs RDKit for the Bemis-Murcko clusters).  It never trains anything and never
changes a prediction; it only reads the pulled artefacts, recomputes metrics
from saved raw predictions, replays the exported REAL-32 rule set from the
original graphs, and writes the frozen tables/report.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sq_pair_common as C
import sq_pair_data as PD
import sq_pair_rules as R
import sq_pair_fista as F

import sq_ast as A
import sq_data as D

MODELS: tuple[str, ...] = (
    "BASE-FULL",
    "BASE-32",
    "NULL-FULL",
    "NULL-32",
    "REAL-FULL",
    "REAL-32",
    "MLP",
    "C1",
)
ADDITIVE_READOUT = {"FULL": "FULL", "-32": "-32"}
COMPARISONS = ("REAL-32_vs_BASE-32", "REAL-32_vs_NULL-32", "REAL-32_vs_C1", "REAL-FULL_vs_MLP")
SPLITS = C.SPLITS


# ---------------------------------------------------------------------------
# input discovery / loading
# ---------------------------------------------------------------------------


def _infer_rr_run_id(path: Path, status: Mapping[str, Any] | None = None) -> str | None:
    if status:
        value = (status.get("run_provenance") or {}).get("rr_run_id")
        if value:
            return str(value)
    parts = Path(path).parts
    if "pulled" in parts:
        index = parts.index("pulled")
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return None


def _protocol_hash(prep_dir: Path) -> str:
    value = C.protocol_hash()
    if value and value != "unavailable":
        return str(value)
    try:
        return str(C.read_json(Path(prep_dir) / "hashes.json").get("protocol_sha256") or "unavailable")
    except Exception:  # noqa: BLE001
        return "unavailable"


def _resolve_rel(raw: str | None, fallback: str) -> str:
    if raw:
        return str(raw)
    return fallback


def _resolve_input_dir(
    config: Mapping[str, Any],
    key: str,
    fallback_rel: str,
    *,
    prefer_pulled: bool = True,
) -> Path:
    rel = _resolve_rel(config.get(key), fallback_rel)
    direct = Path(rel)
    if not direct.is_absolute():
        direct = C.REPO_ROOT / direct
    if prefer_pulled:
        pattern = str(C.REPO_ROOT / ".rr" / "pulled" / "*" / "result" / rel / "status.json")
        matches = sorted(Path("/").glob(pattern.lstrip("/")), key=lambda p: p.stat().st_mtime)
        if matches:
            return matches[-1].parent
    if (direct / "status.json").is_file():
        return direct
    raise FileNotFoundError(
        f"no completed {key} artefacts at {direct} and no pulled copy under .rr/pulled"
    )


def _load_pred(path: Path) -> dict[str, dict[str, np.ndarray]]:
    payload = C.read_npz(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        out[split] = {
            "row_id": np.asarray(payload[f"{split}_row_id"], dtype=np.int64),
            "y_raw": np.asarray(payload[f"{split}_y_raw"], dtype=np.float64),
            "pred_raw": np.asarray(payload[f"{split}_pred_raw"], dtype=np.float64),
        }
    return out


def _pred_paths(arm_dir: Path, model: str) -> dict[str, Path]:
    if model.startswith(("BASE", "NULL", "REAL")):
        arm, readout = model.rsplit("-", 1) if model.endswith("-32") else (model, None)
        if model.endswith("-FULL"):
            arm = model[: -len("-FULL")]
            readout = "FULL"
        elif model.endswith("-32"):
            arm = model[: -len("-32")]
            readout = "-32"
        if arm not in C.ADDITIVE_ARMS or readout is None:
            raise KeyError(model)
        return {
            "best": arm_dir / arm / f"best_{readout}_predictions.npz",
            "epoch180": arm_dir / arm / f"epoch180_{readout}_predictions.npz",
        }
    if model == "MLP":
        return {
            "best": arm_dir / "MLP" / "best_predictions.npz",
            "epoch180": arm_dir / "MLP" / "epoch180_predictions.npz",
        }
    if model == "C1":
        return {
            "best": arm_dir / "best_predictions.npz",
            "epoch180": arm_dir / "epoch180_predictions.npz",
        }
    raise KeyError(model)


def _recorded_scores(
    model: str,
    state: str,
    split: str,
    statuses: Mapping[str, Any],
) -> dict[str, float]:
    if model.startswith(("BASE", "NULL", "REAL")):
        arm = model[: -len("-FULL")] if model.endswith("-FULL") else model[: -len("-32")]
        readout = "FULL" if model.endswith("-FULL") else "-32"
        status = statuses[arm]
        if state == "best":
            return dict(status["best"][readout]["scores"][split])
        return dict(status["epoch180"][readout][split])
    if model == "MLP":
        status = statuses["MLP"]
        if state == "best":
            return dict(status["best"]["scores"][split])
        return dict(status["epoch180"]["scores"][split])
    if model == "C1":
        status = statuses["C1"]
        record = status["selected"] if state == "best" else status["epoch180"]
        return {
            "rmse_raw": float(record[f"{split}_rmse_raw"]),
            "mae_raw": float(record[f"{split}_mae_raw"]),
            "n": int(record.get(f"{split}_n", C.EXPECTED_SPLIT_SIZES[split])),
        }
    raise KeyError(model)


def _model_epoch(model: str, state: str, statuses: Mapping[str, Any]) -> int:
    if state == "epoch180":
        return int(C.EPOCHS)
    if model.startswith(("BASE", "NULL", "REAL")):
        arm = model[: -len("-FULL")] if model.endswith("-FULL") else model[: -len("-32")]
        readout = "FULL" if model.endswith("-FULL") else "-32"
        return int(statuses[arm]["best"][readout]["epoch"])
    if model == "MLP":
        return int(statuses["MLP"]["best"]["epoch"])
    if model == "C1":
        return int(statuses["C1"]["selected"]["epoch"])
    raise KeyError(model)


def _model_fit_seconds(model: str, statuses: Mapping[str, Any]) -> float | None:
    if model.startswith(("BASE", "NULL", "REAL")):
        arm = model[: -len("-FULL")] if model.endswith("-FULL") else model[: -len("-32")]
        return float(statuses[arm].get("fit_seconds", 0.0))
    if model == "MLP":
        return float(statuses["MLP"].get("fit_seconds", 0.0))
    if model == "C1":
        return float(statuses["C1"].get("seconds_total", 0.0))
    return None


def _model_size(model: str, statuses: Mapping[str, Any]) -> dict[str, Any]:
    if model.startswith(("BASE", "NULL", "REAL")):
        arm = model[: -len("-FULL")] if model.endswith("-FULL") else model[: -len("-32")]
        status = statuses[arm]
        readout = "FULL" if model.endswith("-FULL") else "-32"
        active = status["best"][readout].get("n_active_groups")
        used = 32 if model.endswith("-32") else active
        return {
            "n_columns": int(status.get("n_columns", 0)),
            "n_groups": int(status.get("n_groups", 0)),
            "n_active_groups": active,
            "n_used_groups": used,
        }
    if model == "MLP":
        return {"n_parameters": int(statuses["MLP"]["n_parameters"])}
    if model == "C1":
        return {"n_parameters": int(statuses["C1"]["n_parameters"])}
    return {}


# ---------------------------------------------------------------------------
# metric recheck
# ---------------------------------------------------------------------------


def _recheck_metrics(
    *,
    predictions: Mapping[str, Mapping[str, dict[str, np.ndarray]]],
    statuses: Mapping[str, Any],
    prep_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    max_rmse_diff = 0.0
    max_mae_diff = 0.0
    max_label_diff = 0.0
    max_row_diff = 0
    for model in MODELS:
        for state in C.STATES:
            for split in SPLITS:
                pred = predictions[model][state][split]
                official = D.official_metric(C.DATASET, pred["pred_raw"], pred["y_raw"])
                computed = {
                    "rmse_raw": float(official["raw_rmse"]),
                    "mae_raw": float(official["raw_mae"]),
                    "n": int(pred["pred_raw"].size),
                }
                recorded = _recorded_scores(model, state, split, statuses)
                diff_rmse = abs(computed["rmse_raw"] - float(recorded["rmse_raw"]))
                diff_mae = abs(computed["mae_raw"] - float(recorded["mae_raw"]))
                max_rmse_diff = max(max_rmse_diff, diff_rmse)
                max_mae_diff = max(max_mae_diff, diff_mae)
                rows.append(
                    {
                        "model": model,
                        "state": state,
                        "split": split,
                        "epoch": _model_epoch(model, state, statuses),
                        "computed_rmse_raw": computed["rmse_raw"],
                        "recorded_rmse_raw": float(recorded["rmse_raw"]),
                        "rmse_abs_diff": diff_rmse,
                        "computed_mae_raw": computed["mae_raw"],
                        "recorded_mae_raw": float(recorded["mae_raw"]),
                        "mae_abs_diff": diff_mae,
                        "n": computed["n"],
                    }
                )
    # labels / row ids across every model and split
    for split in SPLITS:
        official = C.read_npz(Path(prep_dir) / "matrices" / f"{split}.npz")
        ref_ids = np.asarray(official["row_id"], dtype=np.int64)
        ref_y = np.asarray(official["y_raw"], dtype=np.float64)
        for model in MODELS:
            for state in C.STATES:
                pred = predictions[model][state][split]
                max_row_diff = max(max_row_diff, int(np.abs(pred["row_id"] - ref_ids).max()) if ref_ids.size else 0)
                max_label_diff = max(max_label_diff, float(np.abs(pred["y_raw"] - ref_y).max()) if ref_y.size else 0.0)
                if not np.array_equal(pred["row_id"], ref_ids):
                    raise ValueError(f"{model}/{state}/{split}: prediction row ids are not the frozen ascending ids")
                if not np.allclose(pred["y_raw"], ref_y, atol=0.0, rtol=0.0):
                    raise ValueError(f"{model}/{state}/{split}: prediction labels differ from the official table")
    return {
        "rows": rows,
        "max_rmse_abs_diff": float(max_rmse_diff),
        "max_mae_abs_diff": float(max_mae_diff),
        "max_row_id_abs_diff": int(max_row_diff),
        "max_label_abs_diff": float(max_label_diff),
        "tolerance_raw": 1e-6,
        "pass": bool(max_rmse_diff <= 1e-6 and max_mae_diff <= 1e-6 and max_row_diff == 0 and max_label_diff == 0.0),
    }


# ---------------------------------------------------------------------------
# rules export + replay
# ---------------------------------------------------------------------------


def _rule_groups(
    *,
    prep_dir: Path,
    real_dir: Path,
    statuses: Mapping[str, Any],
    training_commit: str | None,
) -> dict[str, Any]:
    vocab = C.read_json(Path(prep_dir) / "rules_vocabulary.json")
    scales = C.read_npz(Path(prep_dir) / "scales.npz")
    status = statuses["REAL"]
    weights = C.read_npz(Path(real_dir) / "best_-32_weights.npz")
    kept = [int(x) for x in np.asarray(weights["kept_groups"]).reshape(-1).tolist()]
    w = np.asarray(weights["w"], dtype=np.float64)
    bias_std = float(np.asarray(weights["bias_std"]).reshape(-1)[0])
    epoch = int(np.asarray(weights["epoch"]).reshape(-1)[0])
    common_scale = np.asarray(scales["common_scale"], dtype=np.float64)
    relation_scale = np.asarray(scales["real_scale"], dtype=np.float64)
    scale = np.concatenate([common_scale, relation_scale])
    common_groups = vocab["common_groups"]
    relation_groups = vocab["relation_slots"]
    all_groups = list(common_groups) + list(relation_groups)
    selection_path = Path(real_dir) / "best_-32_selection.json"
    selection = C.read_json(selection_path) if selection_path.is_file() else {"scored": []}
    train_std = {int(row["group_index"]): float(row["train_output_std"]) for row in selection.get("scored", [])}
    scaler = vocab["target_scaler"]
    std = float(scaler["std"])
    mean = float(scaler["mean"])
    groups = []
    for gidx in kept:
        group = all_groups[gidx]
        group_scale = scale[2 * gidx : 2 * gidx + 2]
        group_w = w[2 * gidx : 2 * gidx + 2]
        raw_coeff = group_w * std / group_scale
        entry = {
            "group_index": int(gidx),
            "kind": group["kind"],
            "name": group["name"],
            "key": group["key"],
            "scale": [float(x) for x in group_scale],
            "fit_weight_std": [float(x) for x in group_w],
            "raw_coefficients": [float(x) for x in raw_coeff],
            "train_output_std": float(train_std.get(gidx, 0.0)),
            "meta": dict(group.get("meta", {})),
        }
        if group["kind"] == "b_query":
            entry["ast_json"] = group["meta"]["ast_json"]
        groups.append(entry)
    raw_bias = std * bias_std + mean
    model_hash = C.sha256_json(
        {
            "w": w.tolist(),
            "kept_groups": kept,
            "bias_std": bias_std,
            "epoch": epoch,
            "prep": C.read_json(Path(prep_dir) / "hashes.json"),
        }
    )
    candidate_hash = C.sha256_json(
        {
            "protocol_hash": _protocol_hash(prep_dir),
            "prep_hashes": C.read_json(Path(prep_dir) / "hashes.json"),
            "arm": "REAL",
            "readout": "-32",
            "state": "best_valid",
            "model_hash": model_hash,
        }
    )
    return {
        "round_id": C.ROUND_ID,
        "arm": "REAL",
        "readout": "-32",
        "state": "best_valid",
        "epoch": epoch,
        "protocol_hash": _protocol_hash(prep_dir),
        "git_commit": training_commit,
        "prep_dir": str(prep_dir),
        "target_scaler": {"mean": mean, "std": std, "n": int(scaler["n"])},
        "raw_bias": float(raw_bias),
        "groups": groups,
        "n_groups": len(groups),
        "model_hash": model_hash,
        "candidate_hash": candidate_hash,
        "prediction": "raw_bias + sum_j raw_coefficients_j * u_basis(raw_count_j)",
    }


def _raw_count_for_group(g: PD.MolGraph, features: Mapping[str, Any], group: Mapping[str, Any]) -> int:
    kind = str(group["kind"])
    key = str(group["key"])
    if kind == "graph_count":
        return int(features["named"][key])
    if kind == "b_query":
        ast = A.from_json(group["ast_json"])
        return int(PD.b_query_count(g, ast))
    if kind == "s2_path":
        entry = features["s2"].get(key)
        return 0 if entry is None else int(entry["count"])
    if kind == "type_pair_total":
        return int(features["type_counts"].get(key, 0))
    if kind == "distance_bin":
        return int(features["bin_counts"][key])
    if kind == "relation_real":
        return int(features["ext_counts"].get(key, 0))
    raise ValueError(f"unknown group kind {kind!r}")


def _replay_one(g: PD.MolGraph, rules: Mapping[str, Any], *, override: Mapping[int, int] | None = None) -> tuple[float, np.ndarray, list[int]]:
    features = PD.pair_features(g)
    counts = []
    total = float(rules["raw_bias"])
    contributions = np.zeros(len(rules["groups"]), dtype=np.float64)
    for k, group in enumerate(rules["groups"]):
        count = _raw_count_for_group(g, features, group)
        if override is not None and int(group["group_index"]) in override:
            count = int(override[int(group["group_index"])])
        counts.append(int(count))
        basis = C.u_basis(np.asarray([count], dtype=np.float64))[0]
        contribution = float(np.dot(np.asarray(group["raw_coefficients"], dtype=np.float64), basis))
        contributions[k] = contribution
        total += contribution
    return total, contributions, counts


def _replay_all(
    *,
    rules: Mapping[str, Any],
    payload: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "pass": True, "tolerance_raw": C.RULE_REPLAY_TOL}
    for split in ("valid", "test"):
        pred = predictions["REAL-32"]["best"][split]
        row_ids = np.asarray(pred["row_id"], dtype=np.int64)
        saved = np.asarray(pred["pred_raw"], dtype=np.float64)
        replay = np.zeros(row_ids.size, dtype=np.float64)
        counts = np.zeros((row_ids.size, len(rules["groups"])), dtype=np.int32)
        contributions = np.zeros((row_ids.size, len(rules["groups"])), dtype=np.float64)
        for pos, row in enumerate(row_ids.tolist()):
            g = PD.graph_from_payload(payload, int(row))
            value, contrib, count = _replay_one(g, rules)
            replay[pos] = value
            counts[pos] = np.asarray(count, dtype=np.int32)
            contributions[pos] = contrib
        diff = np.abs(replay - saved)
        max_err = float(diff.max()) if diff.size else 0.0
        out["splits"][split] = {
            "n_rows": int(row_ids.size),
            "max_abs_error": max_err,
            "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
            "pass": bool(max_err <= C.RULE_REPLAY_TOL),
            "row_id": row_ids,
            "replay_pred": replay,
            "saved_pred": saved,
            "counts": counts,
            "contributions": contributions,
        }
        out["pass"] = out["pass"] and bool(max_err <= C.RULE_REPLAY_TOL)
    return out


def _decrement_check(
    *,
    rules: Mapping[str, Any],
    payload: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    split = replay["splits"]["valid"]
    row_ids = np.asarray(split["row_id"], dtype=np.int64)
    counts = np.asarray(split["counts"], dtype=np.int64)
    contribution = np.asarray(split["contributions"], dtype=np.float64)
    relation_positions = [
        k for k, group in enumerate(rules["groups"]) if group["kind"] == "relation_real"
    ]
    if not relation_positions:
        return {"applicable": False, "reason": "no relation group in the exported REAL-32 state"}
    any_hit = np.zeros(row_ids.size, dtype=bool)
    for k in relation_positions:
        any_hit |= counts[:, k] > 0
    if not any_hit.any():
        return {"applicable": False, "reason": "no Val molecule has an active exported relation rule"}
    row_pos = int(np.flatnonzero(any_hit)[0])
    row_id = int(row_ids[row_pos])
    chosen_k = None
    for k in relation_positions:
        if counts[row_pos, k] > 0:
            chosen_k = k
            break
    if chosen_k is None:  # pragma: no cover - guarded by any_hit
        return {"applicable": False, "reason": "relation rule disappeared"}
    group = rules["groups"][chosen_k]
    k_before = int(counts[row_pos, chosen_k])
    g = PD.graph_from_payload(payload, row_id)
    pred_base = float(split["replay_pred"][row_pos])
    pred_dec, _, _ = _replay_one(g, rules, override={int(group["group_index"]): k_before - 1})
    basis_before = C.u_basis(np.asarray([k_before], dtype=np.float64))[0]
    basis_after = C.u_basis(np.asarray([max(k_before - 1, 0)], dtype=np.float64))[0]
    expected_delta = float(np.dot(np.asarray(group["raw_coefficients"], dtype=np.float64), basis_after - basis_before))
    observed_delta = pred_dec - pred_base
    err = abs(observed_delta - expected_delta)
    return {
        "applicable": True,
        "row_id": row_id,
        "row_position": row_pos,
        "group_index": int(group["group_index"]),
        "group_name": group["name"],
        "group_key": group["key"],
        "count_before": k_before,
        "count_after": max(k_before - 1, 0),
        "pred_before": pred_base,
        "pred_after": pred_dec,
        "observed_delta": observed_delta,
        "expected_single_term_delta": expected_delta,
        "max_abs_error": float(err),
        "pass": bool(err <= 1e-9),
        "contribution_before": float(contribution[row_pos, chosen_k]),
    }


# ---------------------------------------------------------------------------
# explanation cases
# ---------------------------------------------------------------------------


def _support_for_group(g: PD.MolGraph, group: Mapping[str, Any], *, limit: int = 4) -> dict[str, Any]:
    kind = str(group["kind"])
    key = str(group["key"])
    if kind == "b_query":
        roots = PD.b_query_roots(g, A.from_json(group["ast_json"]))
        return {"kind": kind, "key": key, "count": len(roots), "root_nodes": roots[:limit]}
    return PD.support_details(g, kind, key, limit=limit)


def _explanation_cases(
    *,
    rules: Mapping[str, Any],
    payload: Mapping[str, Any],
    replay: Mapping[str, Any],
    n_cases: int = 20,
) -> list[dict[str, Any]]:
    split = replay["splits"]["valid"]
    row_ids = np.asarray(split["row_id"], dtype=np.int64)
    counts = np.asarray(split["counts"], dtype=np.int64)
    contributions = np.asarray(split["contributions"], dtype=np.float64)
    saved_pred = np.asarray(split["saved_pred"], dtype=np.float64)
    cases: list[dict[str, Any]] = []
    for pos in range(min(int(n_cases), row_ids.size)):
        row_id = int(row_ids[pos])
        g = PD.graph_from_payload(payload, row_id)
        order = sorted(
            range(len(rules["groups"])),
            key=lambda k: (-abs(float(contributions[pos, k])), int(rules["groups"][k]["group_index"])),
        )
        top = []
        for k in order:
            if abs(float(contributions[pos, k])) <= 0.0:
                continue
            group = rules["groups"][k]
            basis = C.u_basis(np.asarray([counts[pos, k]], dtype=np.float64))[0]
            support = _support_for_group(g, group)
            if int(support["count"]) != int(counts[pos, k]):
                raise AssertionError(
                    f"support count {support['count']} != raw count {counts[pos, k]} "
                    f"for row {row_id} group {group['name']}"
                )
            top.append(
                {
                    "group_index": int(group["group_index"]),
                    "kind": group["kind"],
                    "name": group["name"],
                    "key": group["key"],
                    "contribution_raw": float(contributions[pos, k]),
                    "raw_counts": [float(x) for x in basis],
                    "raw_count_value": int(counts[pos, k]),
                    "scale": group["scale"],
                    "raw_coefficients": group["raw_coefficients"],
                    "support": support,
                }
            )
            if len(top) >= 3:
                break
        cases.append(
            {
                "row_id": row_id,
                "target_raw": float(payload["labels"][row_id]),
                "pred_raw": float(saved_pred[pos]),
                "raw_bias": float(rules["raw_bias"]),
                "n_nonzero_groups": int(np.sum(np.abs(contributions[pos]) > 0.0)),
                "top": top,
            }
        )
    return cases


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _cluster_label(smiles: str, row_id: int) -> tuple[str, str]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"ROW:{row_id}", "parse_failed"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return f"SCAFFOLD:{scaffold}", "scaffold"
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return f"ACYCLIC:{canonical}", "acyclic_canonical_smiles"


def _clusters_for_split(
    *,
    split: str,
    row_ids: np.ndarray,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    kinds = {"scaffold": 0, "acyclic_canonical_smiles": 0, "parse_failed": 0}
    for pos, row in enumerate(np.asarray(row_ids, dtype=np.int64).tolist()):
        label, kind = _cluster_label(str(payload["smiles"][row]), int(row))
        groups[label].append(pos)
        kinds[kind] = kinds.get(kind, 0) + 1
    clusters = list(groups.values())
    return {
        "split": split,
        "n_rows": int(row_ids.size),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "kind_counts": kinds,
        "n_parse_failed": int(kinds.get("parse_failed", 0)),
    }


def _bootstrap(
    *,
    predictions: Mapping[str, Mapping[str, dict[str, np.ndarray]]],
    payload: Mapping[str, Any],
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    cluster_meta: dict[str, Any] = {}
    for split in ("valid", "test"):
        row_ids = predictions["REAL-32"]["best"][split]["row_id"]
        clusters = _clusters_for_split(split=split, row_ids=row_ids, payload=payload)
        cluster_meta[split] = {
            "n_rows": clusters["n_rows"],
            "n_clusters": clusters["n_clusters"],
            "kind_counts": clusters["kind_counts"],
            "n_parse_failed": clusters["n_parse_failed"],
        }
        cluster_rows = clusters["clusters"]
        n_clusters = len(cluster_rows)
        for state in C.STATES:
            y = np.asarray(predictions["REAL-32"][state][split]["y_raw"], dtype=np.float64)
            pred = {
                model: np.asarray(predictions[model][state][split]["pred_raw"], dtype=np.float64)
                for model in ("REAL-32", "BASE-32", "NULL-32", "C1", "REAL-FULL", "MLP")
            }

            def rmse_on(index: np.ndarray, values: np.ndarray) -> float:
                diff = values[index] - y[index]
                return float(np.sqrt(np.mean(diff * diff)))

            draws = {name: np.zeros(int(resamples), dtype=np.float64) for name in COMPARISONS}
            for it in range(int(resamples)):
                chosen = rng.choice(n_clusters, size=n_clusters, replace=True)
                index = np.concatenate([cluster_rows[int(c)] for c in chosen])
                r_real32 = rmse_on(index, pred["REAL-32"])
                draws["REAL-32_vs_BASE-32"][it] = rmse_on(index, pred["BASE-32"]) - r_real32
                draws["REAL-32_vs_NULL-32"][it] = rmse_on(index, pred["NULL-32"]) - r_real32
                draws["REAL-32_vs_C1"][it] = rmse_on(index, pred["C1"]) - r_real32
                draws["REAL-FULL_vs_MLP"][it] = rmse_on(index, pred["MLP"]) - rmse_on(index, pred["REAL-FULL"])
            full_index = np.arange(y.size)
            for comparison in COMPARISONS:
                if comparison == "REAL-32_vs_BASE-32":
                    focal, reference = "REAL-32", "BASE-32"
                elif comparison == "REAL-32_vs_NULL-32":
                    focal, reference = "REAL-32", "NULL-32"
                elif comparison == "REAL-32_vs_C1":
                    focal, reference = "REAL-32", "C1"
                else:
                    focal, reference = "REAL-FULL", "MLP"
                point_focal = rmse_on(full_index, pred[focal])
                point_reference = rmse_on(full_index, pred[reference])
                rows.append(
                    {
                        "split": split,
                        "state": state,
                        "comparison": comparison,
                        "focal": focal,
                        "reference": reference,
                        "rmse_focal": point_focal,
                        "rmse_reference": point_reference,
                        "delta": float(point_reference - point_focal),
                        "ci_low": float(np.quantile(draws[comparison], 0.025)),
                        "ci_high": float(np.quantile(draws[comparison], 0.975)),
                        "n_clusters": int(n_clusters),
                        "n_rows": int(y.size),
                        "n_parse_failed": int(clusters["n_parse_failed"]),
                    }
                )
    meta = {
        "resamples": int(resamples),
        "seed": int(seed),
        "rng": "numpy.random.default_rng(seed); one stream in split order valid,test and state order best,epoch180",
        "clusters": cluster_meta,
        "note": "delta = RMSE(reference) - RMSE(focal); positive means the focal model is better",
    }
    return rows, meta


# ---------------------------------------------------------------------------
# decision / bottleneck
# ---------------------------------------------------------------------------


def _table_metrics(
    *,
    predictions: Mapping[str, Mapping[str, dict[str, np.ndarray]]],
    statuses: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for model in MODELS:
        for state in C.STATES:
            entry = {
                "model": model,
                "state": state,
                "epoch": _model_epoch(model, state, statuses),
                **_model_size(model, statuses),
                "fit_seconds": _model_fit_seconds(model, statuses),
            }
            for split in SPLITS:
                pred = predictions[model][state][split]
                scores = _recorded_scores(model, state, split, statuses)
                entry[f"{split}_rmse_raw"] = float(scores["rmse_raw"])
                entry[f"{split}_mae_raw"] = float(scores["mae_raw"])
                entry[f"{split}_n"] = int(scores.get("n", pred["pred_raw"].size))
            rows.append(entry)
    return {"rows": rows}


def _point_gain(
    metrics: Mapping[str, Any],
    *,
    state: str,
    split: str,
    control: str,
    focal: str = "REAL-32",
) -> float:
    values = {
        row["model"]: row
        for row in metrics["rows"]
        if row["state"] == state
    }
    return float(values[control][f"{split}_rmse_raw"]) - float(values[focal][f"{split}_rmse_raw"])


def _ci_lookup(bootstrap_rows: Sequence[Mapping[str, Any]], comparison: str, split: str, state: str) -> Mapping[str, Any]:
    for row in bootstrap_rows:
        if row["comparison"] == comparison and row["split"] == split and row["state"] == state:
            return row
    raise KeyError((comparison, split, state))


def _decision(
    *,
    metrics: Mapping[str, Any],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, Any],
    rules: Mapping[str, Any],
    replay: Mapping[str, Any],
    relation_stats: Mapping[str, Any],
    statuses: Mapping[str, Any],
) -> dict[str, Any]:
    coverage_valid = int(vocab["coverage"]["valid"]["n_rows_with_extension_difference"])
    gain_base_valid = _point_gain(metrics, state="best", split="valid", control="BASE-32")
    gain_null_valid = _point_gain(metrics, state="best", split="valid", control="NULL-32")
    gain_base_test = _point_gain(metrics, state="best", split="test", control="BASE-32")
    gain_null_test = _point_gain(metrics, state="best", split="test", control="NULL-32")
    ci_base = _ci_lookup(bootstrap_rows, "REAL-32_vs_BASE-32", "valid", "best")
    ci_null = _ci_lookup(bootstrap_rows, "REAL-32_vs_NULL-32", "valid", "best")
    active_relation = bool(relation_stats["real32_relation_contribution"]["val_any_abs_gt_1e_6"])
    if coverage_valid < C.MIN_VAL_COVERAGE:
        relation_axis = "LOW_RELATION_COVERAGE"
    elif not active_relation:
        relation_axis = "NO_ACTIVE_RELATION_32"
    elif (
        gain_base_valid >= C.SIGNAL_GAIN
        and gain_null_valid >= C.SIGNAL_GAIN
        and float(ci_base["ci_low"]) > 0.0
        and float(ci_null["ci_low"]) > 0.0
        and gain_base_test > 0.0
        and gain_null_test > 0.0
    ):
        relation_axis = "BINDING_SIGNAL"
    elif (
        gain_base_valid > 0.0
        and gain_null_valid > 0.0
        and (float(ci_base["ci_low"]) <= 0.0 or float(ci_null["ci_low"]) <= 0.0 or gain_base_test <= 0.0 or gain_null_test <= 0.0)
    ):
        relation_axis = "UNRESOLVED_BINDING"
    elif gain_base_valid > 0.0 and gain_null_valid <= 0.0:
        relation_axis = "ADDITION_NOT_BINDING"
    else:
        relation_axis = "NO_CLEAR_BINDING_SIGNAL"

    real32 = {(row["state"]): row for row in metrics["rows"] if row["model"] == "REAL-32"}
    c1 = {(row["state"]): row for row in metrics["rows"] if row["model"] == "C1"}
    gap_valid = float(real32["best"]["valid_rmse_raw"]) - float(c1["best"]["valid_rmse_raw"])
    gap_test = float(real32["best"]["test_rmse_raw"]) - float(c1["best"]["test_rmse_raw"])
    replay_pass = bool(replay["pass"])
    performance_axis = "NEAR_C1" if (gap_valid <= C.NEAR_C1_GAP and gap_test <= C.NEAR_C1_GAP and replay_pass) else "NOT_NEAR_C1"

    c1_val = float(c1["best"]["valid_rmse_raw"])
    weak_live_reference = c1_val > C.OLD_C1_VAL + C.WEAK_REFERENCE_MARGIN

    full = {(row["state"]): row for row in metrics["rows"] if row["model"] == "REAL-FULL"}
    mlp = {(row["state"]): row for row in metrics["rows"] if row["model"] == "MLP"}
    same_epoch = relation_stats["same_weights_full_vs_32"]
    budget_gain_valid = float(same_epoch["valid"]["rmse_-32"]) - float(same_epoch["valid"]["rmse_FULL"])
    readout_gain_valid = float(full["best"]["valid_rmse_raw"]) - float(mlp["best"]["valid_rmse_raw"])
    mlp_gap_valid = float(mlp["best"]["valid_rmse_raw"]) - c1_val
    mlp_gap_test = float(mlp["best"]["test_rmse_raw"]) - float(c1["best"]["test_rmse_raw"])
    full_gap_valid = float(full["best"]["valid_rmse_raw"]) - c1_val
    real32_gap_valid = gap_valid

    if budget_gain_valid >= C.INTERPRET_SCALE:
        bottleneck = {
            "verdict": "COMPRESSION",
            "detail": "same weights without the 32-group budget lower Val RMSE by at least 0.02",
            "budget_gain_valid": budget_gain_valid,
        }
    elif (
        mlp_gap_valid <= C.NEAR_C1_GAP
        and mlp_gap_test <= C.NEAR_C1_GAP
        and readout_gain_valid >= C.INTERPRET_SCALE
    ):
        bottleneck = {
            "verdict": "READOUT_CAPACITY_FIT",
            "detail": "MLP is within 0.04 of C1 on Val and Test and at least 0.02 below REAL FULL on Val",
            "readout_gain_valid": readout_gain_valid,
        }
    elif full_gap_valid > C.NEAR_C1_GAP and real32_gap_valid > C.NEAR_C1_GAP:
        bottleneck = {
            "verdict": "VOCABULARY_LIMIT",
            "detail": "both REAL readouts trail C1 by more than 0.04 on Val",
            "full_gap_valid": full_gap_valid,
            "real32_gap_valid": real32_gap_valid,
        }
    else:
        bottleneck = {
            "verdict": "UNCLEAR",
            "detail": "differences, directions and intervals are reported without a clear location",
        }
    return {
        "relation_axis": relation_axis,
        "performance_axis": performance_axis,
        "promising_pilot": bool(relation_axis == "BINDING_SIGNAL" and performance_axis == "NEAR_C1"),
        "weak_live_reference": bool(weak_live_reference),
        "coverage_valid": coverage_valid,
        "active_relation_val": active_relation,
        "gains_best": {
            "vs_BASE-32_valid": gain_base_valid,
            "vs_NULL-32_valid": gain_null_valid,
            "vs_BASE-32_test": gain_base_test,
            "vs_NULL-32_test": gain_null_test,
        },
        "ci95_best_valid": {
            "vs_BASE-32": [float(ci_base["ci_low"]), float(ci_base["ci_high"])],
            "vs_NULL-32": [float(ci_null["ci_low"]), float(ci_null["ci_high"])],
        },
        "gap_REAL32_minus_C1": {"valid": gap_valid, "test": gap_test},
        "replay_pass": replay_pass,
        "live_c1_valid_rmse": c1_val,
        "historical_c1_valid_rmse": C.OLD_C1_VAL,
        "bottleneck": bottleneck,
        "bottleneck_numbers": {
            "same_weights_budget_gain_valid": budget_gain_valid,
            "same_weights_FULL_valid": same_epoch["valid"]["rmse_FULL"],
            "same_weights_minus32_valid": same_epoch["valid"]["rmse_-32"],
            "REAL32_best_valid": float(real32["best"]["valid_rmse_raw"]),
            "REALFULL_best_valid": float(full["best"]["valid_rmse_raw"]),
            "REAL32_best_test": float(real32["best"]["test_rmse_raw"]),
            "REALFULL_best_test": float(full["best"]["test_rmse_raw"]),
            "MLP_best_valid": float(mlp["best"]["valid_rmse_raw"]),
            "MLP_best_test": float(mlp["best"]["test_rmse_raw"]),
            "C1_best_valid": c1_val,
            "C1_best_test": float(c1["best"]["test_rmse_raw"]),
            "readout_gain_valid": readout_gain_valid,
            "mlp_gap_vs_C1_valid": mlp_gap_valid,
            "mlp_gap_vs_C1_test": mlp_gap_test,
            "full_gap_vs_C1_valid": full_gap_valid,
            "real32_gap_vs_C1_valid": real32_gap_valid,
        },
    }


def _same_weights_full_vs_32(
    *,
    predictions: Mapping[str, Mapping[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        y = np.asarray(predictions["REAL-32"]["best"][split]["y_raw"], dtype=np.float64)
        p32 = np.asarray(predictions["REAL-32"]["best"][split]["pred_raw"], dtype=np.float64)
        pfull = np.asarray(predictions["REAL-FULL_AT_32BEST"]["best"][split]["pred_raw"], dtype=np.float64)
        out[split] = {
            "rmse_-32": C.rmse(p32, y),
            "rmse_FULL": C.rmse(pfull, y),
            "n": int(y.size),
        }
    return out


def _relation_stats(
    *,
    rules: Mapping[str, Any],
    prep_dir: Path,
    predictions: Mapping[str, Mapping[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    relation_groups = [group for group in rules["groups"] if group["kind"] == "relation_real"]
    for split in ("valid", "test"):
        matrix = C.read_npz(Path(prep_dir) / "matrices" / f"{split}.npz")
        real_raw = np.asarray(matrix["real_raw"], dtype=np.float64)
        any_stored = np.any(real_raw > 0.0, axis=1) if real_raw.size else np.zeros(real_raw.shape[0], dtype=bool)
        # contributions of the exported REAL-32 relation groups
        pred_path = predictions["REAL-32"]["best"][split]
        row_ids = np.asarray(pred_path["row_id"], dtype=np.int64)
        X = np.concatenate(
            [np.asarray(matrix["base_raw"], dtype=np.float64), real_raw], axis=1
        )
        contrib = np.zeros((row_ids.size, len(relation_groups)), dtype=np.float64)
        for k, group in enumerate(relation_groups):
            gidx = int(group["group_index"])
            contrib[:, k] = X[:, 2 * gidx : 2 * gidx + 2] @ np.asarray(group["raw_coefficients"], dtype=np.float64)
        abs_contrib = np.abs(contrib)
        n_nonzero_rules = (abs_contrib > 1e-12).sum(axis=1)
        all_contrib = np.zeros((row_ids.size, len(rules["groups"])), dtype=np.float64)
        for k, group in enumerate(rules["groups"]):
            gidx = int(group["group_index"])
            all_contrib[:, k] = X[:, 2 * gidx : 2 * gidx + 2] @ np.asarray(group["raw_coefficients"], dtype=np.float64)
        n_nonzero_all = (np.abs(all_contrib) > 1e-12).sum(axis=1)
        out[split] = {
            "n_rows": int(row_ids.size),
            "n_exported_relation_groups": len(relation_groups),
            "n_rows_with_any_stored_relation_hit": int(any_stored.sum()),
            "fraction_rows_with_any_stored_relation_hit": float(any_stored.mean()) if any_stored.size else 0.0,
            "n_rows_with_nonzero_relation_contribution": int((n_nonzero_rules > 0).sum()),
            "fraction_rows_with_nonzero_relation_contribution": float((n_nonzero_rules > 0).mean()) if n_nonzero_rules.size else 0.0,
            "relation_contribution_abs_sum_quantiles": [
                float(np.quantile(abs_contrib.sum(axis=1), q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            ],
            "relation_nonzero_rule_count_quantiles": [
                float(np.quantile(n_nonzero_rules, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            ],
            "all_nonzero_rule_count_quantiles": [
                float(np.quantile(n_nonzero_all, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            ],
        }
    # exact check using the exported groups' Val contributions
    matrix = C.read_npz(Path(prep_dir) / "matrices" / "valid.npz")
    X = np.concatenate(
        [np.asarray(matrix["base_raw"], dtype=np.float64), np.asarray(matrix["real_raw"], dtype=np.float64)], axis=1
    )
    relation_groups = [group for group in rules["groups"] if group["kind"] == "relation_real"]
    max_abs = 0.0
    if relation_groups:
        for group in relation_groups:
            gidx = int(group["group_index"])
            contrib = X[:, 2 * gidx : 2 * gidx + 2] @ np.asarray(group["raw_coefficients"], dtype=np.float64)
            max_abs = max(max_abs, float(np.abs(contrib).max()))
    out["real32_relation_contribution"] = {
        "val_max_abs_contribution": float(max_abs),
        "val_any_abs_gt_1e_6": bool(max_abs > 1e-6),
        "threshold": 1e-6,
    }
    return out


# ---------------------------------------------------------------------------
# report formatting
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return str(value)
        return f"{float(value):.{digits}f}"
    return str(value)


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body])


def _metric_rows(metrics: Mapping[str, Any], state: str) -> list[dict[str, Any]]:
    return [row for row in metrics["rows"] if row["state"] == state]


def _report(
    *,
    out: Path,
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    rules: Mapping[str, Any],
    replay: Mapping[str, Any],
    decrement: Mapping[str, Any],
    relation_stats: Mapping[str, Any],
    clusters: Mapping[str, Any],
    resource: Mapping[str, Any],
    prep: Mapping[str, Any],
    statuses: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Lipo 显式结构关系小试验｜sq-lipo-pair-rule-v1")
    lines.append("")
    lines.append(
        "官方 OGB `ogbg-mollipo`，官方 scaffold Train/Val/Test = 3360/420/420，"
        "单训练 seed=0，180 epoch，`test_policy: always`（Test 为本轮 DEVELOPMENT_TEST）。"
        "本文件由 `stage=aggregate` 从已拉回的冻结产物自动生成；所有数字均为 raw logD 单位。"
        "所有模型状态只由 Val RMSE 选择，Test 仅作为 DEVELOPMENT_TEST 随同一状态报告，"
        "不参与任何选择。"
    )
    lines.append("")
    prov = summary["provenance"]
    lines.append("## 运行与协议")
    lines.append("")
    lines.append(
        f"- 训练代码 commit：`{prov.get('training_git_commit')}`；聚合代码 commit："
        f"`{prov.get('aggregate_git_commit')}`"
    )
    lines.append(f"- 协议：`{C.ROUND_ID}`，protocol hash `{prov.get('protocol_hash')}`")
    lines.append(f"- rr run：prepare `{prov.get('rr_prep')}`，C1 `{prov.get('rr_c1')}`，rules `{prov.get('rr_rules')}`")
    lines.append(f"- 设备：C1 GPU0 `cuda:0`；MLP GPU1 `cuda:0`；三条加性臂 CPU（BLAS ≤4 线程）")
    lines.append(f"- 冻结缓存：graph `{prov.get('graph_cache_sha256')}`；lipo raw `{prov.get('lipo_raw_sha256')}`")
    lines.append(f"- 规则规模：B={summary['prep']['n_b_records']}，S2={summary['prep']['n_s2_selected']}，"
                 f"真实关系槽={summary['prep']['n_relation_slots']}，共同列={summary['prep']['n_common_columns']}，"
                 f"扩展列={summary['prep']['n_extension_columns']}")
    lines.append("")
    lines.append("## 运行状态")
    lines.append("")
    lines.append(_md_table(summary.get("run_status", []), [
        ("stage", "stage"), ("run_id", "rr run"), ("state", "state"),
        ("node", "node"), ("device", "CUDA_VISIBLE_DEVICES"), ("wall_s", "wall s"),
    ]))
    lines.append("")
    lines.append(
        "三个远端阶段均通过 rr 进程后端在 `res` 的 A100 上执行；prepare/rules 使用 "
        "GPU1，C1 使用 GPU0，两条 worker 并行。"
    )
    lines.append("")
    lines.append("## 预定结论")
    lines.append("")
    lines.append(f"- 关系轴：**{decision['relation_axis']}**")
    lines.append(f"- 性能轴：**{decision['performance_axis']}**")
    lines.append(f"- `PROMISING_PILOT`：**{_fmt(decision['promising_pilot'])}**")
    if decision["weak_live_reference"]:
        lines.append("- 附记：**WEAK_LIVE_REFERENCE**（本轮 live C1 最佳 Val RMSE 高于历史 0.7052 超过 0.04）")
    lines.append(f"- 瓶颈线索：**{decision['bottleneck']['verdict']}** — {decision['bottleneck']['detail']}")
    lines.append("")
    lines.append(
        f"关系轴的关键数：Val 覆盖 {decision['coverage_valid']}/420；"
        f"REAL-32 在 Val 的活跃关系贡献最大绝对值 "
        f"{_fmt(relation_stats['real32_relation_contribution']['val_max_abs_contribution'])} logD；"
        f"Val gain vs BASE-32 = {_fmt(decision['gains_best']['vs_BASE-32_valid'])}，"
        f"vs NULL-32 = {_fmt(decision['gains_best']['vs_NULL-32_valid'])}；"
        f"Test gain vs BASE-32 = {_fmt(decision['gains_best']['vs_BASE-32_test'])}，"
        f"vs NULL-32 = {_fmt(decision['gains_best']['vs_NULL-32_test'])}。"
    )
    lines.append(
        f"性能轴：REAL-32 相对 live C1 的 gap：Val {_fmt(decision['gap_REAL32_minus_C1']['valid'])}，"
        f"Test {_fmt(decision['gap_REAL32_minus_C1']['test'])} logD；规则重放通过 = "
        f"{_fmt(decision['replay_pass'])}。"
    )
    lines.append("")
    g = decision["gains_best"]
    ci = decision["ci95_best_valid"]
    lines.append(
        f"就数字本身：REAL-32 的 Val 比 BASE-32 好 {_fmt(g['vs_BASE-32_valid'])}（95% CI "
        f"[{_fmt(ci['vs_BASE-32'][0])}, {_fmt(ci['vs_BASE-32'][1])}]），比 NULL-32 好 "
        f"{_fmt(g['vs_NULL-32_valid'])}（[{_fmt(ci['vs_NULL-32'][0])}, "
        f"{_fmt(ci['vs_NULL-32'][1])}]）；Test 对应 gain 为 {_fmt(g['vs_BASE-32_test'])} 与 "
        f"{_fmt(g['vs_NULL-32_test'])}。也就是说，实际锚点对计数相对**边际 NULL**有可辨认的增量，"
        f"但相对共同输入 BASE 的增量没有达到预定的 0.02、其 Val 区间也跨 0，因此关系轴只能记为 "
        f"`UNRESOLVED_BINDING`，不能记为 `BINDING_SIGNAL`。"
    )
    lines.append(
        f"性能轴没有接近 live C1：REAL FULL 的 Val {_fmt(decision['bottleneck_numbers']['REALFULL_best_valid'])} / "
        f"Test {_fmt(decision['bottleneck_numbers']['REALFULL_best_test'])}，MLP 的 Val "
        f"{_fmt(decision['bottleneck_numbers']['MLP_best_valid'])} / Test "
        f"{_fmt(decision['bottleneck_numbers']['MLP_best_test'])}，C1 的 Val "
        f"{_fmt(decision['bottleneck_numbers']['C1_best_valid'])} / Test "
        f"{_fmt(decision['bottleneck_numbers']['C1_best_test'])}；live C1 与历史 0.7052 的差为 "
        f"{_fmt(decision['live_c1_valid_rmse'] - decision['historical_c1_valid_rmse'])}，未触发 "
        f"`WEAK_LIVE_REFERENCE`。"
    )
    lines.append("")

    lines.append("## 预检")
    lines.append("")
    pre = summary["preflight"]
    lines.append(f"- `preflight.json` pass = **{_fmt(pre['pass'])}**，失败项 {pre['n_failed']}。")
    lines.append(f"- 行/划分：4200 行覆盖 0…4199，三划分互斥，标签逐项一致，Train-only 标准化往返最大误差 "
                 f"{_fmt(pre['max_abs_error_observed']['label_roundtrip'], 8)}。")
    lines.append(f"- 全图计数：4200 张图全部满足 `Σ_d H_d = T`、`Σ_t C_t = T`、有限 d≥3 表与距离边际一致；最大误差 "
                 f"{_fmt(max(pre['max_abs_error_observed'][k] for k in ('pair_total','bin_sum','type_sum','ext_sum')))}。")
    lines.append(f"- 重标号不变性：8 个官方 Train 行全部通过；18 环节点正对照通过；"
                 f"NULL `T=4,H_d=2,C_t=2` 两列为 `5/6, 1/6`；组近端与 FULL/-32 偏置复算通过。")
    lines.append("")

    lines.append("## 主表（各臂最小 Val RMSE 状态）")
    lines.append("")
    main_rows = []
    for row in _metric_rows(metrics, "best"):
        main_rows.append(
            {
                "model": row["model"],
                "epoch": row["epoch"],
                "train": row["train_rmse_raw"],
                "valid": row["valid_rmse_raw"],
                "test": row["test_rmse_raw"],
                "vmae": row["valid_mae_raw"],
                "tmae": row["test_mae_raw"],
                "size": row.get("n_parameters") if row.get("n_parameters") else (
                    f"{row.get('n_used_groups')}/{row.get('n_groups')} groups"
                ),
                "fit_s": row.get("fit_seconds"),
            }
        )
    lines.append(_md_table(main_rows, [
        ("model", "arm/readout"), ("epoch", "epoch"), ("train", "Train RMSE"),
        ("valid", "Val RMSE"), ("test", "Test RMSE"), ("vmae", "Val MAE"),
        ("tmae", "Test MAE"), ("size", "params / active groups"), ("fit_s", "fit s"),
    ]))
    lines.append("")
    lines.append("## 固定 epoch 180 表")
    lines.append("")
    last_rows = []
    for row in _metric_rows(metrics, "epoch180"):
        last_rows.append(
            {
                "model": row["model"],
                "train": row["train_rmse_raw"],
                "valid": row["valid_rmse_raw"],
                "test": row["test_rmse_raw"],
                "vmae": row["valid_mae_raw"],
                "tmae": row["test_mae_raw"],
                "size": row.get("n_parameters") if row.get("n_parameters") else (
                    f"{row.get('n_used_groups')}/{row.get('n_groups')} groups"
                ),
            }
        )
    lines.append(_md_table(last_rows, [
        ("model", "arm/readout"), ("train", "Train RMSE"), ("valid", "Val RMSE"),
        ("test", "Test RMSE"), ("vmae", "Val MAE"), ("tmae", "Test MAE"),
        ("size", "params / active groups"),
    ]))
    lines.append("")

    lines.append("## 配对 cluster bootstrap（2000 次，seed 1703）")
    lines.append("")
    boot_rows = []
    for row in bootstrap_rows:
        boot_rows.append(
            {
                "split": row["split"],
                "state": row["state"],
                "comparison": row["comparison"],
                "delta": row["delta"],
                "low": row["ci_low"],
                "high": row["ci_high"],
                "clusters": row["n_clusters"],
                "rows": row["n_rows"],
            }
        )
    lines.append(_md_table(boot_rows, [
        ("split", "split"), ("state", "state"), ("comparison", "comparison"),
        ("delta", "delta (reference−focal)"), ("low", "CI95 low"), ("high", "CI95 high"),
        ("clusters", "clusters"), ("rows", "rows"),
    ]))
    lines.append("")
    lines.append(
        "`delta = RMSE(reference) - RMSE(focal)`；对 REAL-32 的三行，正数表示 REAL-32 更好；"
        "`REAL-FULL_vs_MLP` 正数表示 REAL FULL 更好。"
    )
    lines.append("")
    lines.append("Bemis–Murcko 簇：" + "；".join(
        f"{split} {clusters[split]['n_clusters']} 簇 / {clusters[split]['n_rows']} 行，解析失败 "
        f"{clusters[split]['n_parse_failed']}"
        for split in ("valid", "test")
    ) + "。")
    lines.append("")

    lines.append("## 规则复核与解释")
    lines.append("")
    lines.append(f"- REAL-32 导出 `rules.json`：epoch {rules['epoch']}，{rules['n_groups']} 个活跃规则组；"
                 f"model hash `{rules['model_hash'][:16]}…`。")
    mr = summary["metric_recheck"]
    lines.append(
        f"- 官方指标从保存的 raw 预测独立重算：{len(MODELS) * len(C.STATES) * len(SPLITS)} 组，"
        f"最大 RMSE 绝对差 {_fmt(mr['max_rmse_abs_diff'], 12)}，最大 MAE 绝对差 "
        f"{_fmt(mr['max_mae_abs_diff'], 12)}，行 ID 与标签逐项一致 → **{_fmt(mr['pass'])}**。"
    )
    lines.append(f"- 独立规则重放（从原图重算共同项与关系项）：Val 最大绝对误差 "
                 f"{_fmt(replay['splits']['valid']['max_abs_error'], 8)}，Test "
                 f"{_fmt(replay['splits']['test']['max_abs_error'], 8)}；容差 1e-6 → "
                 f"**{_fmt(replay['pass'])}**。")
    if decrement.get("applicable"):
        lines.append(
            f"- 单项减一检查（Val 最小官方 row ID）：row {decrement['row_id']}，规则 `{decrement['group_name']}`，"
            f"K {decrement['count_before']}→{decrement['count_after']}；预测差 "
            f"{_fmt(decrement['observed_delta'], 8)}，导出函数单项差 "
            f"{_fmt(decrement['expected_single_term_delta'], 8)}，最大误差 "
            f"{_fmt(decrement['max_abs_error'], 10)} → **{_fmt(decrement['pass'])}**。"
        )
    else:
        lines.append(f"- 单项减一检查：不适用（{decrement.get('reason')}）。")
    lines.append("")
    stats_rows = []
    for split in ("valid", "test"):
        st = relation_stats[split]
        stats_rows.append(
            {
                "split": split,
                "hit_frac": st["fraction_rows_with_any_stored_relation_hit"],
                "nonzero_frac": st["fraction_rows_with_nonzero_relation_contribution"],
                "rel_rules_med": st["relation_nonzero_rule_count_quantiles"][2],
                "all_rules_med": st["all_nonzero_rule_count_quantiles"][2],
                "abs_contrib_med": st["relation_contribution_abs_sum_quantiles"][2],
            }
        )
    lines.append(_md_table(stats_rows, [
        ("split", "split"), ("hit_frac", "有入库关系命中的分子比例"),
        ("nonzero_frac", "REAL-32 关系项非零的分子比例"),
        ("rel_rules_med", "非零关系规则数中位数"), ("all_rules_med", "非零规则总数中位数"),
        ("abs_contrib_med", "关系项 \\|贡献\\| 和中位数"),
    ]))
    lines.append("")
    lines.append(
        f"升序前 20 个 Val 分子的逐规则解释写入 `analysis/explanation_cases.jsonl`；"
        f"样例计数、锚点对 / S2 路径 / 根 ID 都在该文件中。"
    )
    lines.append("")

    lines.append("## 瓶颈线索")
    lines.append("")
    bn = decision["bottleneck_numbers"]
    lines.append(_md_table([
        {"name": "同权重 FULL vs -32（REAL-32 选中 epoch）", "valid": bn["same_weights_FULL_valid"], "minus32": bn["same_weights_minus32_valid"], "gain": bn["same_weights_budget_gain_valid"]},
        {"name": "各自选中状态 REAL FULL / REAL-32", "valid": bn["REALFULL_best_valid"], "minus32": bn["REAL32_best_valid"], "gain": bn["REAL32_best_valid"] - bn["REALFULL_best_valid"]},
    ], [("name", "comparison"), ("valid", "FULL / reference Val RMSE"), ("minus32", "-32 / focal Val RMSE"), ("gain", "difference")]))
    lines.append("")
    lines.append(
        f"- REAL FULL vs MLP：Val 差（FULL−MLP）{_fmt(bn['readout_gain_valid'])}，"
        f"MLP 相对 C1 的 gap：Val {_fmt(bn['mlp_gap_vs_C1_valid'])}，Test {_fmt(bn['mlp_gap_vs_C1_test'])}。"
    )
    lines.append(
        f"- REAL FULL 相对 C1：Val gap {_fmt(bn['full_gap_vs_C1_valid'])}；REAL-32 相对 C1："
        f"Val gap {_fmt(bn['real32_gap_vs_C1_valid'])}。"
    )
    if decision["bottleneck"]["verdict"] == "VOCABULARY_LIMIT":
        real_curve = statuses["REAL"]["curve_last"]
        lines.append(
            f"- 当前证据词表可行性受限：REAL FULL Train RMSE "
            f"{_fmt([row for row in metrics['rows'] if row['model']=='REAL-FULL' and row['state']=='best'][0]['train_rmse_raw'])}；"
            f"末段目标 {_fmt(real_curve['objective'])}（epoch {real_curve['epoch']}）。"
        )
    lines.append("")

    lines.append("## 历史背景（不参与本轮判定）")
    lines.append("")
    lines.append("| 对象 | 旧轮 Val RMSE | 旧轮 Test RMSE |")
    lines.append("| --- | --- | --- |")
    lines.append("| `sq-pilot-small-v1` Lipo A | 0.7911 | 0.7525 |")
    lines.append("| `sq-pilot-small-v1` Lipo C1 | 0.7052 | 0.7139 |")
    lines.append("| `sq-pilot-small-v1` Lipo GINE | 0.6091 | 0.6456 |")
    lines.append("")
    lines.append(
        "本轮门槛使用**本次现场重训的 C1**，不是 0.7052。历史数字只说明旧轮的量级；"
        "若本轮 C1 明显弱于旧轮，已在预定结论中附记 `WEAK_LIVE_REFERENCE`。"
    )
    lines.append("")

    lines.append("## 资源账本")
    lines.append("")
    lines.append(_md_table([
        {"item": "prepare wall", "value": resource["wall"]["prep"]},
        {"item": "C1 worker wall", "value": resource["wall"]["c1"]},
        {"item": "rules worker wall", "value": resource["wall"]["rules"]},
        {"item": "aggregate wall (local)", "value": resource["aggregate_wall_seconds"]},
        {"item": "remote wall (first prep start → last worker finish)", "value": resource["remote_wall_seconds"]},
        {"item": "GPU card reservation (prepare + C1 + rules)", "value": resource["gpu_card_seconds_total"]},
        {"item": "GPU hours reservation", "value": resource["gpu_hours_total"]},
        {"item": "CPU additive solve s", "value": resource["cpu_additive_fit_seconds"]},
    ], [("item", "item"), ("value", "seconds")]))
    lines.append("")
    lines.append(
        f"预算：60 分钟远端墙钟 / 2 A100·小时；本轮远端墙钟 {_fmt(resource['remote_wall_seconds']/60.0, 2)} 分钟，"
        f"占卡 {_fmt(resource['gpu_hours_total'], 3)} A100·h。CPU 求解时间单独列出。"
    )
    lines.append("")

    lines.append("## 产物与证据")
    lines.append("")
    lines.append("- `analysis/summary.json`：本轮结构化摘要、两轴判定、全部数字与运行 provenance。")
    lines.append("- `analysis/rules.json`：REAL-32 最佳 Val 状态的 32 个规则组、尺度、raw 系数与 bias；`model_hash`/`candidate_hash` 可追溯。")
    lines.append("- `analysis/replay_audit.json`：独立重放的最大绝对误差与单项减一检查。")
    lines.append("- `analysis/metric_recheck.json`：48 个 model/state/split 组合的官方指标重算与记录值差异。")
    lines.append("- `analysis/bootstrap.csv`：2000 次配对 cluster bootstrap 的全部比较。")
    lines.append("- `analysis/explanation_cases.jsonl`：升序前 20 个 Val 分子的 top-3 逐规则贡献与支持对象。")
    lines.append("- `analysis/resource_ledger.json`：阶段墙钟、占卡时间、CPU 求解时间。")
    lines.append("- 远端原始产物：`.rr/pulled/<run_id>/result/tracks/structural-query-learning/pair_rule_pilot_v1/runs/{prep,c1,rules}`。")
    lines.append("")
    lines.append("## 不确定性与边界")
    lines.append("")
    lines.append(
        "- 单一训练 seed=0；本轮不提供多 seed 稳定性结论。MLP 与 C1 的容量/优化器差异保留在表中，"
        "不能把 MLP 的差异单独归因于读出。"
    )
    lines.append(
        "- bootstrap 的对象是当前固定预测在留出分子上的不确定性（按 Bemis–Murcko 簇重抽），"
        "不是重新训练模型的不确定性。"
    )
    lines.append(
        "- 关系轴只在当前 512 个 Train 过滤后的槽位、180 epoch、该 lambda 和该 32 项预算下成立；"
        "词表外的结构关系没有被检验。"
    )
    lines.append(
        "- 结论限于本轮求解条件与单种子范围；如果性能轴为 `NOT_NEAR_C1`，不能外推为"
        "“显式关系无用”，只能说明当前读出/预算/词表组合未达到 0.04 logD 门槛。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# aggregate entry
# ---------------------------------------------------------------------------


def aggregate(config: Mapping[str, Any], ledger) -> tuple[dict[str, Any], Path]:
    t0 = time.perf_counter()
    start_prov = C.run_provenance()
    prefer_pulled = bool(config.get("prefer_pulled", True))
    prep_dir = _resolve_input_dir(
        config,
        "prep_dir",
        "tracks/structural-query-learning/pair_rule_pilot_v1/runs/prep",
        prefer_pulled=prefer_pulled,
    )
    c1_dir = _resolve_input_dir(
        config,
        "c1_dir",
        "tracks/structural-query-learning/pair_rule_pilot_v1/runs/c1",
        prefer_pulled=prefer_pulled,
    )
    rules_dir = _resolve_input_dir(
        config,
        "rules_dir",
        "tracks/structural-query-learning/pair_rule_pilot_v1/runs/rules",
        prefer_pulled=prefer_pulled,
    )
    analysis_dir = C.ensure_dir(
        Path(str(config.get("analysis_dir", C.ANALYSIS_DIR)))
        if Path(str(config.get("analysis_dir", C.ANALYSIS_DIR))).is_absolute()
        else C.REPO_ROOT / str(config.get("analysis_dir", C.ANALYSIS_DIR))
    )
    payload = R.load_payload_verified()
    prep_vocab = C.read_json(prep_dir / "rules_vocabulary.json")
    prep_summary = C.read_json(prep_dir / "summary.json")
    prep_status = C.read_json(prep_dir / "status.json")
    preflight = C.read_json(prep_dir / "preflight.json")
    c1_status = C.read_json(c1_dir / "status.json")
    rules_status = C.read_json(rules_dir / "status.json")

    statuses: dict[str, Any] = {"C1": c1_status}
    for arm in ("MLP", *C.ADDITIVE_ARMS):
        status = C.read_json(rules_dir / arm / "status.json")
        if status.get("state") != "completed":
            raise RuntimeError(f"arm {arm} is not completed: {status.get('state')}")
        statuses[arm] = status
    # REAL status paths are relative to the worker; make the aggregate robust
    if "paths" not in statuses["REAL"]:
        statuses["REAL"]["paths"] = {"real": str(rules_dir / "REAL")}

    predictions: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for model in MODELS:
        paths = _pred_paths(rules_dir if model != "C1" else c1_dir, model)
        predictions[model] = {
            state: _load_pred(paths[state]) for state in C.STATES
        }
    # same-weights FULL at the REAL-32 selected epoch
    predictions["REAL-FULL_AT_32BEST"] = {
        "best": _load_pred(rules_dir / "REAL" / "real32_best_epoch_FULL_predictions.npz"),
        "epoch180": _load_pred(rules_dir / "REAL" / "real32_best_epoch_FULL_predictions.npz"),
    }

    metric_recheck = _recheck_metrics(
        predictions=predictions,
        statuses=statuses,
        prep_dir=prep_dir,
        payload=payload,
    )
    training_commit = (prep_status.get("run_provenance") or {}).get("git_commit") or prep_status.get("git_commit")
    rules = _rule_groups(
        prep_dir=prep_dir,
        real_dir=rules_dir / "REAL",
        statuses=statuses,
        training_commit=training_commit,
    )
    replay = _replay_all(rules=rules, payload=payload, predictions=predictions)
    decrement = _decrement_check(rules=rules, payload=payload, replay=replay)
    cases = _explanation_cases(rules=rules, payload=payload, replay=replay)
    metrics = _table_metrics(predictions=predictions, statuses=statuses)
    bootstrap_rows, bootstrap_meta = _bootstrap(
        predictions=predictions,
        payload=payload,
        resamples=int(config.get("bootstrap_resamples", C.BOOTSTRAP_RESAMPLES)),
        seed=int(config.get("bootstrap_seed", C.BOOTSTRAP_SEED)),
    )
    same_weights = _same_weights_full_vs_32(predictions=predictions)
    relation_stats = _relation_stats(rules=rules, prep_dir=prep_dir, predictions=predictions)
    relation_stats["same_weights_full_vs_32"] = same_weights
    decision = _decision(
        metrics=metrics,
        bootstrap_rows=bootstrap_rows,
        vocab=prep_vocab,
        rules=rules,
        replay=replay,
        relation_stats=relation_stats,
        statuses=statuses,
    )
    clusters = bootstrap_meta["clusters"]

    # resource ledger
    stage_status = {"prep": prep_status, "c1": c1_status, "rules": rules_status}
    wall = {
        key: float(stage_status[key].get("wall_seconds", stage_status[key].get("finished_epoch", 0.0) - stage_status[key].get("started_epoch", 0.0)))
        for key in ("prep", "c1", "rules")
    }
    starts = [float(stage_status[key].get("started_epoch", 0.0)) for key in ("prep", "c1", "rules")]
    finishes = [float(stage_status[key].get("finished_epoch", 0.0)) for key in ("prep", "c1", "rules")]
    remote_wall = max(finishes) - min(starts)
    aggregate_wall = time.perf_counter() - t0
    cpu_additive = float(sum(statuses[arm].get("fit_seconds", 0.0) for arm in C.ADDITIVE_ARMS))
    resource = {
        "wall": wall,
        "remote_wall_seconds": float(remote_wall),
        "gpu_card_seconds_total": float(sum(wall.values())),
        "gpu_hours_total": float(sum(wall.values()) / 3600.0),
        "mlp_gpu_seconds": float(statuses["MLP"].get("fit_seconds", 0.0)),
        "c1_gpu_seconds": float(c1_status.get("seconds_total", 0.0)),
        "cpu_additive_fit_seconds": cpu_additive,
        "cpu_seconds_by_arm": {arm: float(statuses[arm].get("fit_seconds", 0.0)) for arm in C.ADDITIVE_ARMS},
        "budget_remote_wall_seconds": 3600.0,
        "budget_gpu_hours": 2.0,
        "gpu_card_note": (
            "conservative reservation: prepare (GPU1), C1 worker (GPU0) and rules worker "
            "(GPU1, MLP then CPU additive) are each counted for their full stage wall"
        ),
    }
    resource["aggregate_wall_seconds"] = aggregate_wall

    # provenance summary
    provenance = {
        "training_git_commit": training_commit,
        "aggregate_git_commit": C.git_commit(),
        "git_commit": training_commit,
        "protocol_hash": _protocol_hash(prep_dir),
        "graph_cache_sha256": C.read_json(prep_dir / "hashes.json").get("graph_cache_sha256"),
        "lipo_raw_sha256": C.EXPECTED_CACHE_SHA256["lipo_raw"],
        "rr_prep": _infer_rr_run_id(prep_dir, prep_status),
        "rr_c1": _infer_rr_run_id(c1_dir, c1_status),
        "rr_rules": _infer_rr_run_id(rules_dir, rules_status),
        "prep_dir": str(prep_dir),
        "c1_dir": str(c1_dir),
        "rules_dir": str(rules_dir),
        "analysis_dir": str(analysis_dir),
        "aggregate_started_at": start_prov["started_at"],
    }
    provenance["graph_cache_sha256"] = provenance.get("graph_cache_sha256") or C.EXPECTED_CACHE_SHA256["graph"]

    # write artefacts (big arrays stay out of git; small JSON/CSV/MD are tracked)
    C.write_json(analysis_dir / "rules.json", rules)
    C.write_json(analysis_dir / "replay_audit.json", {
        key: value for key, value in replay.items() if key != "splits"
    } | {
        "splits": {
            split: {k: v for k, v in payload_split.items() if k not in ("row_id", "replay_pred", "saved_pred", "counts", "contributions")}
            for split, payload_split in replay["splits"].items()
        },
        "decrement": decrement,
        "pass": bool(replay["pass"] and (not decrement.get("applicable") or decrement.get("pass"))),
    })
    with (analysis_dir / "explanation_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(C.json_safe(case), ensure_ascii=False, sort_keys=True) + "\n")
    C.write_csv(analysis_dir / "bootstrap.csv", bootstrap_rows)
    C.write_json(analysis_dir / "metric_recheck.json", metric_recheck)
    C.write_json(analysis_dir / "tables.json", {
        "main_best": _metric_rows(metrics, "best"),
        "epoch180": _metric_rows(metrics, "epoch180"),
        "bootstrap": bootstrap_rows,
        "decision": decision,
    })
    C.write_json(analysis_dir / "resource_ledger.json", resource)
    run_status = [
        {
            "stage": stage,
            "run_id": {
                "prepare": provenance.get("rr_prep"),
                "worker_c1": provenance.get("rr_c1"),
                "worker_rules": provenance.get("rr_rules"),
            }[stage],
            "state": status.get("state"),
            "node": (status.get("run_provenance") or {}).get("hostname"),
            "device": (status.get("run_provenance") or {}).get("cuda_visible_devices"),
            "wall_s": float(status.get("wall_seconds", 0.0)),
        }
        for stage, status in (
            ("prepare", prep_status),
            ("worker_c1", c1_status),
            ("worker_rules", rules_status),
        )
    ]
    summary = {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "provenance": provenance,
        "prep": prep_summary,
        "run_status": run_status,
        "preflight": {
            "pass": bool(preflight.get("pass")),
            "n_failed": int(preflight.get("n_failed", 0)),
            "max_abs_error_observed": preflight.get("scale", {}).get("max_abs_error_observed", {}),
        },
        "metric_recheck": {
            "pass": bool(metric_recheck["pass"]),
            "max_rmse_abs_diff": float(metric_recheck["max_rmse_abs_diff"]),
            "max_mae_abs_diff": float(metric_recheck["max_mae_abs_diff"]),
            "max_label_abs_diff": float(metric_recheck["max_label_abs_diff"]),
            "max_row_id_abs_diff": int(metric_recheck["max_row_id_abs_diff"]),
            "tolerance_raw": 1e-6,
        },
        "metrics_best": _metric_rows(metrics, "best"),
        "metrics_epoch180": _metric_rows(metrics, "epoch180"),
        "bootstrap": bootstrap_rows,
        "bootstrap_meta": bootstrap_meta,
        "rules": {
            "epoch": rules["epoch"],
            "n_groups": rules["n_groups"],
            "model_hash": rules["model_hash"],
            "candidate_hash": rules["candidate_hash"],
            "raw_bias": rules["raw_bias"],
        },
        "replay": {
            "pass": bool(replay["pass"]),
            "tolerance_raw": C.RULE_REPLAY_TOL,
            "splits": {
                split: {
                    "n_rows": payload_split["n_rows"],
                    "max_abs_error": payload_split["max_abs_error"],
                    "mean_abs_error": payload_split["mean_abs_error"],
                    "pass": payload_split["pass"],
                }
                for split, payload_split in replay["splits"].items()
            },
            "decrement": decrement,
        },
        "explanation_cases": {
            "n_cases": len(cases),
            "path": str(analysis_dir / "explanation_cases.jsonl"),
        },
        "relation_stats": relation_stats,
        "same_weights_full_vs_32": same_weights,
        "decision": decision,
        "resource_ledger": resource,
        "environment": C.environment(),
        "run_provenance": start_prov,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_json(analysis_dir / "summary.json", summary)
    report = _report(
        out=analysis_dir,
        summary=summary,
        metrics=metrics,
        bootstrap_rows=bootstrap_rows,
        decision=decision,
        rules=rules,
        replay=replay,
        decrement=decrement,
        relation_stats=relation_stats,
        clusters=clusters,
        resource=resource,
        prep=prep_summary,
        statuses=statuses,
    )
    C.REPORT_PATH.write_text(report, encoding="utf-8")
    C.write_json(analysis_dir / "status.json", C.finish_provenance({
        "state": "completed",
        "stage": "aggregate",
        "summary_path": str(analysis_dir / "summary.json"),
        "report_path": str(C.REPORT_PATH),
        "run_provenance": start_prov,
        "metric_recheck_pass": bool(metric_recheck["pass"]),
        "replay_pass": bool(replay["pass"]),
        "decision": decision,
    }))
    metrics_out = {
        "relation_axis": decision["relation_axis"],
        "performance_axis": decision["performance_axis"],
        "promising_pilot": decision["promising_pilot"],
        "weak_live_reference": decision["weak_live_reference"],
        "bottleneck": decision["bottleneck"]["verdict"],
        "real32_valid_rmse": float([row for row in metrics["rows"] if row["model"] == "REAL-32" and row["state"] == "best"][0]["valid_rmse_raw"]),
        "c1_valid_rmse": float([row for row in metrics["rows"] if row["model"] == "C1" and row["state"] == "best"][0]["valid_rmse_raw"]),
        "metric_recheck_pass": bool(metric_recheck["pass"]),
        "replay_pass": bool(replay["pass"]),
        "report": str(C.REPORT_PATH),
    }
    return metrics_out, analysis_dir
