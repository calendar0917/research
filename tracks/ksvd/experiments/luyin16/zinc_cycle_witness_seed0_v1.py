"""ZINC cycle-conflict witnesses on the old CPU head input (seed 0).

CPU companion of ``zinc_task_dictionary_and_cycle_witness_seed0_v1``.  It does
**not** train any model.  It answers, with reproducible tables:

1. what the old ``exact_conflict_analysis`` numbers actually meant
   (``repeated`` vs ``conflicting`` classes, and the difference between the
   global input-optimal output, that same output evaluated per severity group,
   per-group optimal outputs, and the severity-weighted bound);
2. which molecules/graphs/targets actually conflict on the old CPU head's
   float32 ``T25`` input, including the five old-fit ``k<=-3`` rows and the old
   internal-dev extreme tail;
3. whether the repo's cycle helper is node-order dependent, and whether the
   conflicting graphs are non-isomorphic (input compression) rather than
   ordering artefacts.

Everything is train-only: the old 8000 fit / 2000 internal dev rows are a
*model-internal* split of the same 10000 official-train rows.  No
official-valid / official-test row is loaded.  This module intentionally does
not touch the new main fold.

Run:
    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_cycle_witness_seed0_v1 --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

PROTOCOL_VERSION = "zinc-cycle-witness-seed0-v1"

TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_task_dictionary_and_cycle_witness_seed0_v1"
T25_PATH = TRACK_ROOT / "results/zinc_frozen_chemistry_learned_cycle_v1/T25_all.npz"
O_PRED_PATH = TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1/O_seed0_predictions.npz"
TARGET_PATH = TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1/target_decomposition.npz"
AUDIT_CSV = TRACK_ROOT / "results/zinc_long_cycle_audit/train_cycle_audit_label.csv"
REFINE_JSON = TRACK_ROOT / "results/zinc_long_cycle_audit/stage_refine.json"
OVERNIGHT_DIR = TRACK_ROOT / "results/zinc_overnight_interface_and_tail_seed0_v1"
ZINC_ROOT = TRACK_ROOT.parents[1] / "data/ZINC"

RENUMBER_SEED = 20261004
RENUMBER_N = 16
GROUP_NAMES = ("k=0", "k=-1", "k=-2", "k<=-3")
COMMUNITY_CYCLE_MEAN = -0.048569687328769765
COMMUNITY_CYCLE_STD = 0.2859726330705808


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stable_id(index: int) -> str:
    return f"train:{int(index):04d}"


def severity_group(k: int) -> str:
    if int(k) == 0:
        return "k=0"
    if int(k) == -1:
        return "k=-1"
    if int(k) == -2:
        return "k=-2"
    return "k<=-3"


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k=0": k == 0,
        "k=-1": k == -1,
        "k=-2": k == -2,
        "k<=-3": k <= -3,
        "k<=-2": k <= -2,
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * float(weights.sum())
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(values[min(index, values.size - 1)])


# ---------------------------------------------------------------------------
# exact input classes (bitwise-comparable float32 semantics)
# ---------------------------------------------------------------------------


def exact_classes(T: np.ndarray) -> dict[str, Any]:
    """Group rows by exact ``T25`` equality.

    Primary grouping is numpy ``==`` on the float32 rows (the old semantics:
    ``-0.0 == 0.0``); the byte-key class count is recorded as a check.  NaNs
    would break both, so their absence is asserted.
    """
    T = np.ascontiguousarray(T, np.float32)
    if np.isnan(T).any():
        raise RuntimeError("T25 contains NaN; exact-class semantics undefined")
    negative_zero = int(np.sum(np.signbit(T) & (T == 0)))
    order = np.lexsort(T.T)
    Ts = T[order]
    new_class = np.ones(Ts.shape[0], dtype=bool)
    if Ts.shape[0] > 1:
        new_class[1:] = np.any(Ts[1:] != Ts[:-1], axis=1)
    class_id_sorted = np.cumsum(new_class) - 1
    # map the sorted class ids back to the original row order
    class_id = np.empty(Ts.shape[0], np.int64)
    class_id[order] = class_id_sorted
    counts = np.bincount(class_id)
    # byte-key consistency check (would differ only with signed zeros)
    keys = [row.tobytes() for row in Ts]
    byte_classes = len(set(keys))
    return {
        "class_id": class_id,
        "order": order,
        "counts": counts,
        "n_classes": int(counts.size),
        "byte_class_count": int(byte_classes),
        "negative_zero_count": negative_zero,
        "signed_zero_consistent": bool(byte_classes == counts.size),
    }


# ---------------------------------------------------------------------------
# stage 1: corrected bound semantics
# ---------------------------------------------------------------------------


def stage_bounds(out_dir: Path, log: Any = print) -> dict[str, Any]:
    t25 = np.load(T25_PATH)
    T_all = np.asarray(t25["T"], np.float32)
    with np.load(O_PRED_PATH) as o:
        fit_idx = np.asarray(o["fit_idx"], np.int64)
        dev_idx = np.asarray(o["dev_idx"], np.int64)
    with np.load(TARGET_PATH) as z:
        k_all = np.asarray(z["k"], np.int64)
        c_all = np.asarray(z["c"], np.float64)
        y_all = np.asarray(z["y"], np.float64)

    T = T_all[fit_idx]
    c = c_all[fit_idx]
    k = k_all[fit_idx]
    n = int(T.shape[0])
    classes = exact_classes(T)
    class_id = classes["class_id"]
    counts = classes["counts"]
    size = counts[class_id]
    multi = size >= 2

    # per-class over all fit rows
    class_c: dict[int, np.ndarray] = {}
    class_median: dict[int, float] = {}
    for cid in range(int(counts.size)):
        rows = np.where(class_id == cid)[0]
        class_c[cid] = c[rows]
        class_median[cid] = float(np.median(c[rows]))
    pred_global = np.array([class_median[int(cid)] for cid in class_id], np.float64)

    conflict_ids: list[int] = []
    for cid in range(int(counts.size)):
        if counts[cid] >= 2 and np.unique(class_c[cid]).size > 1:
            conflict_ids.append(cid)
    conflicting_rows = np.isin(class_id, conflict_ids)

    global_error = np.abs(c - pred_global)
    global_min_l1 = float(global_error.mean())
    global_opt_group_cost = {
        name: float(global_error[mask].mean()) for name, mask in group_masks(k).items()
    }
    global_opt_group_total = {
        name: float(global_error[mask].sum()) for name, mask in group_masks(k).items()
    }

    # per-group class medians (other groups are allowed to get worse)
    pred_group = np.zeros(n, np.float64)
    group_only_per_group: dict[str, float] = {}
    for name in GROUP_NAMES:
        mask = group_masks(k)[name]
        medians: dict[int, float] = {}
        for cid in np.unique(class_id[mask]):
            rows = np.where(mask & (class_id == cid))[0]
            medians[int(cid)] = float(np.median(c[rows]))
        rows = np.where(mask)[0]
        pred_group[rows] = [medians[int(class_id[i])] for i in rows]
        group_only_per_group[name] = float(np.abs(c[mask] - pred_group[mask]).mean())
    group_only_min_l1 = float(np.abs(c - pred_group).mean())

    # old fixed severity weights
    group_counts = {name: int(group_masks(k)[name].sum()) for name in GROUP_NAMES}
    present = {name: count for name, count in group_counts.items() if count > 0}
    g = len(present)
    weights = np.zeros(n, np.float64)
    for name, count in present.items():
        weights[group_masks(k)[name]] = n / (g * count)
    pred_balanced = np.zeros(n, np.float64)
    balanced_group_cost: dict[str, float] = {}
    balanced_class_count: dict[int, int] = {}
    for cid in range(int(counts.size)):
        rows = np.where(class_id == cid)[0]
        pred_balanced[rows] = weighted_median(c[rows], weights[rows])
        balanced_class_count[cid] = int(rows.size)
    balanced_error = np.abs(c - pred_balanced)
    balanced_weighted_l1 = float(np.sum(weights * balanced_error) / np.sum(weights))
    for name in GROUP_NAMES:
        mask = group_masks(k)[name]
        balanced_group_cost[name] = float(balanced_error[mask].mean())

    old_json = json.loads((OVERNIGHT_DIR / "cpu_conflict_8k.json").read_text())
    out = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "bounds",
        "source": {
            "T25": str(T25_PATH.relative_to(TRACK_ROOT)),
            "T25_sha256_fit": sha256_array(T),
            "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
            "fit_rows": int(n),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
        "class_semantics": {
            "n_exact_classes": int(classes["n_classes"]),
            "byte_class_count": int(classes["byte_class_count"]),
            "negative_zero_count": int(classes["negative_zero_count"]),
            "signed_zero_consistent": bool(classes["signed_zero_consistent"]),
            "n_repeated_classes": int((counts >= 2).sum()),
            "repeated_row_fraction": float(multi.mean()),
            "n_conflict_classes": int(len(conflict_ids)),
            "conflicting_row_fraction": float(conflicting_rows.mean()),
            "old_conflict_row_fraction_reported": float(old_json["conflict_row_fraction"]),
            "old_value_meaning": "multi.mean() = repeated-input row fraction, not conflicting-class row fraction",
        },
        "global_min_l1_per_row": global_min_l1,
        "global_min_l1_total": float(global_error.sum()),
        "global_opt_group_cost": global_opt_group_cost,
        "global_opt_group_total": global_opt_group_total,
        "group_only_min_l1": group_only_min_l1,
        "group_only_per_group": group_only_per_group,
        "balanced_opt": {
            "weights": {name: float(n / (g * count)) for name, count in present.items()},
            "group_counts": group_counts,
            "weighted_l1_per_row": balanced_weighted_l1,
            "group_cost": balanced_group_cost,
            "class_count_with_weights": int(sum(1 for v in balanced_class_count.values() if v >= 2)),
        },
        "old_reported": {
            "irreducible_l1_per_row": float(old_json["irreducible_l1_per_row"]),
            "irreducible_per_group": old_json["per_group"],
            "note": "old per-group k<=-3 value is the all-fit class-median output evaluated on that group, not the group's own optimum",
        },
    }
    out["checks"] = {
        "global_min_l1_matches_old_overall": bool(
            abs(out["global_min_l1_per_row"] - float(old_json["irreducible_l1_per_row"])) < 1e-9
        ),
        "group_kle3_matches_old_5_549": bool(
            abs(out["global_opt_group_cost"]["k<=-3"] - 5.5494614659849475) < 1e-6
        ),
    }
    write_json(out_dir / "cycle_bounds.json", out)

    rows = ["quantity,group,n,value"]
    rows.append(f"global_min_l1_per_row,overall,{n},{global_min_l1:.10f}")
    for name in ("k=0", "k=-1", "k=-2", "k<=-3", "k<=-2"):
        rows.append(f"global_opt_group_cost,{name},{int(group_masks(k)[name].sum())},{global_opt_group_cost[name]:.10f}")
    for name in GROUP_NAMES:
        rows.append(f"group_only_min_l1,{name},{int(group_masks(k)[name].sum())},{group_only_per_group[name]:.10f}")
    for name in GROUP_NAMES:
        rows.append(f"balanced_opt_group_cost,{name},{int(group_masks(k)[name].sum())},{balanced_group_cost[name]:.10f}")
    (out_dir / "cycle_bounds_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(
        f"[bounds] global={global_min_l1:.6f} k<=-3(global-opt)={global_opt_group_cost['k<=-3']:.6f} "
        f"k<=-3(group-only)={group_only_per_group['k<=-3']:.6f} conflict_rows={conflicting_rows.mean():.4f}"
    )
    return out


# ---------------------------------------------------------------------------
# stage 2: witnesses
# ---------------------------------------------------------------------------


def _pair_extremes(values: np.ndarray) -> tuple[int, int, float]:
    """Index pair with the largest absolute value difference (smallest ids tie)."""
    order = np.argsort(values, kind="stable")
    best_i, best_j, best_diff = 0, 0, -1.0
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = int(order[a]), int(order[b])
            diff = float(abs(values[i] - values[j]))
            if diff > best_diff + 1e-15:
                best_i, best_j, best_diff = i, j, diff
    return best_i, best_j, best_diff


def _graph_from_data(data: Any) -> Any:
    import networkx as nx

    edge_index = data.edge_index.detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from((int(edge_index[0, k]), int(edge_index[1, k])) for k in range(edge_index.shape[1]))
    return graph


def _max_basis_stats(graph: Any) -> dict[str, float]:
    """Exact copy of the repo diagnostic helper semantics (``_cycle_basis_stats``)."""
    import networkx as nx

    if len(graph) == 0:
        cycles: list[list[int]] = []
    else:
        try:
            cycles = nx.cycle_basis(graph)
        except nx.NetworkXException:
            cycles = []
    max_len = max((len(cycle) for cycle in cycles), default=0)
    excess = max(max_len - 6, 0)
    return {
        "basis_cycle_count": float(len(cycles)),
        "max_basis_cycle_length": float(max_len),
        "excess_basis_cycle_length": float(excess),
        "cycle_score": float(-excess),
    }


def stage_witnesses(out_dir: Path, log: Any = print) -> dict[str, Any]:
    import networkx as nx
    from networkx.algorithms import isomorphism as nx_iso
    import pandas as pd

    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    t25 = np.load(T25_PATH)
    T_all = np.asarray(t25["T"], np.float32)
    with np.load(O_PRED_PATH) as o:
        fit_idx = np.asarray(o["fit_idx"], np.int64)
        dev_idx = np.asarray(o["dev_idx"], np.int64)
        h_fit = np.asarray(o["fit_raw"], np.float64)
        h_dev = np.asarray(o["dev_raw"], np.float64)
        y_fit = np.asarray(o["fit_y"], np.float64)
        y_dev = np.asarray(o["dev_y"], np.float64)
        c_fit = np.asarray(o["fit_c"], np.float64)
        c_dev = np.asarray(o["dev_c"], np.float64)
    with np.load(TARGET_PATH) as z:
        k_all = np.asarray(z["k"], np.int64)
        c_all = np.asarray(z["c"], np.float64)
        y_all = np.asarray(z["y"], np.float64)
        g_all = np.asarray(z["g"], np.float64)
    with np.load(OVERNIGHT_DIR / "cpu_P_U_predictions.npz") as z:
        q_u_fit, q_u_dev = np.asarray(z["q_fit"], np.float64), np.asarray(z["q_dev"], np.float64)
    with np.load(OVERNIGHT_DIR / "cpu_P_B_predictions.npz") as z:
        q_b_fit, q_b_dev = np.asarray(z["q_fit"], np.float64), np.asarray(z["q_dev"], np.float64)
    cpu_analysis = json.loads((OVERNIGHT_DIR / "cpu_analysis_8k.json").read_text())
    b_u = float(cpu_analysis["arms"]["P_U"]["b_P"])
    b_b = float(cpu_analysis["arms"]["P_B"]["b_P"])
    audit = pd.read_csv(AUDIT_CSV)
    audit = audit.set_index("subset_index", drop=False)
    refine = json.loads(REFINE_JSON.read_text())
    mu_cycle = float(refine["constants_fitted"]["mu_cycle"])
    sigma_cycle = float(refine["constants_fitted"]["sigma_cycle"])

    T = T_all[fit_idx]
    n = int(T.shape[0])
    classes = exact_classes(T)
    class_id = classes["class_id"]
    counts = classes["counts"]
    size = counts[class_id]
    multi_rows = size >= 2

    class_members: dict[int, list[int]] = {}
    for position in range(n):
        class_members.setdefault(int(class_id[position]), []).append(position)

    conflict_ids: list[int] = []
    for cid, members in class_members.items():
        if len(members) >= 2 and np.unique(c_fit[members]).size > 1:
            conflict_ids.append(cid)
    conflict_ids.sort()

    class_rows: list[dict[str, Any]] = []
    witness_pairs: list[dict[str, Any]] = []
    audit_class_fraction: list[float] = []
    for cid in conflict_ids:
        members = class_members[cid]
        cs = c_fit[members]
        ks = k_all[fit_idx[members]]
        i, j, diff = _pair_extremes(cs)
        member_ids = [stable_id(int(fit_idx[m])) for m in members]
        t_value = [float(v) for v in T[members[0]]]
        record = {
            "class_id": int(cid),
            "class_size": int(len(members)),
            "member_stable_ids": member_ids,
            "member_positions": [int(m) for m in members],
            "k_values": [int(v) for v in ks],
            "c_values": [float(v) for v in cs],
            "c_min": float(cs.min()),
            "c_max": float(cs.max()),
            "c_median": float(np.median(cs)),
            "T25_values": t_value,
            "T25_sha256": sha256_array(T[members[0]]),
            "contains_k_le_minus2": bool((ks <= -2).any()),
            "contains_k_le_minus3": bool((ks <= -3).any()),
            "audit_cycle_score_stored_order": [float(audit.loc[int(fit_idx[m]), "cycle_score_stored_order"]) for m in members],
            "audit_cycle_score_gvae_order": [float(audit.loc[int(fit_idx[m]), "cycle_score_gvae_order"]) for m in members],
            "audit_label_effective_cycle_snapped": [float(audit.loc[int(fit_idx[m]), "label_effective_cycle_snapped"]) for m in members],
            "audit_effective_cycle_norm": [
                float((audit.loc[int(fit_idx[m]), "label_effective_cycle_snapped"] - mu_cycle) / sigma_cycle) for m in members
            ],
            "audit_y_without_cycle_label": [float(audit.loc[int(fit_idx[m]), "y_without_cycle_label"]) for m in members],
            "witness_pair_stable_ids": [member_ids[i], member_ids[j]],
            "witness_pair_positions": [int(members[i]), int(members[j])],
            "witness_pair_c_diff": float(diff),
        }
        class_rows.append(record)
        witness_pairs.append(
            {
                "class_id": int(cid),
                "left_position": int(members[i]),
                "right_position": int(members[j]),
                "left_stable_id": member_ids[i],
                "right_stable_id": member_ids[j],
                "c_diff": float(diff),
                "left_c": float(cs[i]),
                "right_c": float(cs[j]),
                "left_k": int(ks[i]),
                "right_k": int(ks[j]),
            }
        )
        audit_class_fraction.append(float(len(members)) / 10000.0)

    # extreme tails: old fit k<=-3 and old internal dev k<=-3 / k<=-2
    def tail_record(position: int, split: str) -> dict[str, Any]:
        global_id = int(fit_idx[position]) if split == "fit" else int(dev_idx[position])
        cid = int(class_id[position]) if split == "fit" else None
        if split == "fit":
            cls = class_members[cid]
            class_size = len(cls)
            class_c = c_fit[cls]
        else:
            matches = np.where(np.all(T_all[fit_idx] == T_all[dev_idx[position]], axis=1))[0]
            class_size = int(matches.size)
            class_c = c_fit[matches] if matches.size else np.array([c_dev[position]])
        return {
            "split": split,
            "position": int(position),
            "stable_id": stable_id(global_id),
            "k": int(k_all[global_id]),
            "c": float(c_all[(fit_idx if split == "fit" else dev_idx)[position]]),
            "y": float(y_all[(fit_idx if split == "fit" else dev_idx)[position]]),
            "exact_class_size_on_fit": int(class_size),
            "exact_class_c_min": float(np.min(class_c)),
            "exact_class_c_max": float(np.max(class_c)),
            "exact_class_c_spread": float(np.max(class_c) - np.min(class_c)),
            "T25_sha256": sha256_array(T_all[global_id]),
        }

    extremes_fit_positions = [int(i) for i in np.where(k_all[fit_idx] <= -3)[0]]
    extremes_dev_positions = [int(i) for i in np.where(k_all[dev_idx] <= -3)[0]]
    medium_dev_positions = [int(i) for i in np.where(k_all[dev_idx] == -2)[0]]

    tail_rows: list[dict[str, Any]] = []
    for position in extremes_fit_positions:
        record = tail_record(position, "fit")
        record["q_U_raw_error"] = float(abs(y_fit[position] - (h_fit[position] + q_u_fit[position])))
        record["q_U_cal_error"] = float(abs(y_fit[position] - (h_fit[position] + q_u_fit[position] + b_u)))
        record["q_B_raw_error"] = float(abs(y_fit[position] - (h_fit[position] + q_b_fit[position])))
        record["q_B_cal_error"] = float(abs(y_fit[position] - (h_fit[position] + q_b_fit[position] + b_b)))
        record["q_U_pred_fit"] = float(q_u_fit[position])
        record["q_B_pred_fit"] = float(q_b_fit[position])
        record["in_conflict_class"] = bool(record["exact_class_c_spread"] > 0 and record["exact_class_size_on_fit"] >= 2)
        tail_rows.append(record)
    tail_rows_dev: list[dict[str, Any]] = []
    for position in extremes_dev_positions + medium_dev_positions:
        record = tail_record(position, "dev")
        record["q_U_raw_error"] = float(abs(y_dev[position] - (h_dev[position] + q_u_dev[position])))
        record["q_U_cal_error"] = float(abs(y_dev[position] - (h_dev[position] + q_u_dev[position] + b_u)))
        record["q_B_raw_error"] = float(abs(y_dev[position] - (h_dev[position] + q_b_dev[position])))
        record["q_B_cal_error"] = float(abs(y_dev[position] - (h_dev[position] + q_b_dev[position] + b_b)))
        record["q_U_pred_dev"] = float(q_u_dev[position])
        record["q_B_pred_dev"] = float(q_b_dev[position])
        tail_rows_dev.append(record)

    # graph checks for witness pairs
    dataset = _load_zinc(ZINC_ROOT, "train")
    graph_rows: list[dict[str, Any]] = []
    graph_payload: dict[str, Any] = {}
    for pair in witness_pairs:
        pair_records = []
        for side in ("left", "right"):
            position = int(pair[f"{side}_position"])
            global_id = int(fit_idx[position])
            data = dataset[global_id]
            y_value = float(np.asarray(data.y).reshape(-1)[0])
            if abs(y_value - float(y_all[global_id])) > 1e-5:
                raise RuntimeError(f"graph/y mismatch at {stable_id(global_id)}: {y_value} vs {y_all[global_id]}")
            graph = _graph_from_data(data)
            atom_types = np.asarray(data.x.detach().cpu().numpy()).reshape(-1).astype(int).tolist()
            edge_attr = getattr(data, "edge_attr", None)
            bond_types = (
                np.asarray(edge_attr.detach().cpu().numpy()).reshape(int(data.num_edges), -1)[:, 0].astype(int).tolist()
                if edge_attr is not None
                else [0] * int(data.num_edges)
            )
            edges = [(int(u), int(v)) for u, v in graph.edges()]
            degrees = [int(graph.degree(node)) for node in range(int(data.num_nodes))]
            stats = _max_basis_stats(graph)
            record = {
                "stable_id": stable_id(global_id),
                "n_nodes": int(data.num_nodes),
                "n_edges": int(data.num_edges),
                "components": int(nx.number_connected_components(graph)),
                "degree_sequence": sorted(degrees),
                "atom_type_counts": {str(t): int(atom_types.count(t)) for t in sorted(set(atom_types))},
                "bond_type_counts": {str(t): int(bond_types.count(t)) for t in sorted(set(bond_types))},
                "helper_stored_order": stats,
                "helper_matches_audit_stored_order": bool(
                    abs(stats["cycle_score"] - float(audit.loc[global_id, "cycle_score_stored_order"])) < 1e-9
                ),
                "audit_cycle_score_stored_order": float(audit.loc[global_id, "cycle_score_stored_order"]),
                "audit_cycle_score_gvae_order": float(audit.loc[global_id, "cycle_score_gvae_order"]),
                "label_effective_cycle_snapped": float(audit.loc[global_id, "label_effective_cycle_snapped"]),
                "smi_line": int(audit.loc[global_id, "smi_line"]),
                "endpoints_in_range": bool(
                    all(0 <= u < int(data.num_nodes) and 0 <= v < int(data.num_nodes) for u, v in edges)
                ),
                "adjacency_symmetric": bool(all(u != v for u, v in edges)),
                "graph_sha256": sha256_array(np.asarray(edges, np.int64).reshape(-1)),
            }
            pair_records.append(record)
            graph_payload[stable_id(global_id)] = {
                "adjacency": edges,
                "atom_types": atom_types,
                "bond_types": bond_types,
            }
        topology_iso = bool(
            nx.is_isomorphic(
                _dataset_graph(dataset, int(fit_idx[pair["left_position"]])),
                _dataset_graph(dataset, int(fit_idx[pair["right_position"]])),
            )
        )
        typed_iso = bool(
            nx.is_isomorphic(
                _dataset_graph(dataset, int(fit_idx[pair["left_position"]])),
                _dataset_graph(dataset, int(fit_idx[pair["right_position"]])),
                node_match=nx_iso.categorical_node_match("t", -1),
                edge_match=nx_iso.categorical_edge_match("b", -1),
            )
        )
        graph_rows.append(
            {
                "class_id": int(pair["class_id"]),
                "left": pair_records[0],
                "right": pair_records[1],
                "topology_only_isomorphic": topology_iso,
                "typed_isomorphic": typed_iso,
                "T25_equal": True,
                "c_diff": float(pair["c_diff"]),
            }
        )

    # static figures
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for pair, graph_row in zip(witness_pairs, graph_rows):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for axis, side, record in zip(axes, ("left", "right"), (graph_row["left"], graph_row["right"])):
            data = dataset[int(fit_idx[pair[f"{side}_position"]])]
            graph = _graph_from_data(data)
            atom_types = np.asarray(data.x.detach().cpu().numpy()).reshape(-1).astype(int)
            color_values = atom_types / max(1, int(atom_types.max()))
            positions = nx.spring_layout(graph, seed=0)
            nx.draw_networkx(
                graph,
                pos=positions,
                ax=axis,
                node_color=color_values,
                cmap="tab20",
                node_size=120,
                with_labels=False,
                width=1.0,
            )
            axis.set_title(
                f"{record['stable_id']}  c={pair[f'{side}_c']:.3f}  k={pair[f'{side}_k']}\n"
                f"max basis={int(record['helper_stored_order']['max_basis_cycle_length'])}  smi_line={record['smi_line']}"
            )
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"witness_class_{pair['class_id']}.png", bbox_inches="tight")
        plt.close(fig)

    witnesses = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "witnesses",
        "source": {
            "T25": str(T25_PATH.relative_to(TRACK_ROOT)),
            "old_fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
            "old_dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
            "n_fit": int(fit_idx.size),
            "n_dev": int(dev_idx.size),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
        "n_conflict_classes": int(len(conflict_ids)),
        "conflict_classes": class_rows,
        "witness_pairs": witness_pairs,
        "fit_extreme_k_le_minus3": tail_rows,
        "dev_tail_rows": tail_rows_dev,
        "graph_checks": graph_rows,
        "classification": _classify(graph_rows, tail_rows, witness_pairs),
    }
    write_json(out_dir / "cycle_class_witnesses.json", witnesses)
    with open(out_dir / "cycle_class_witnesses.csv", "w", encoding="utf-8") as handle:
        handle.write("class_id,member_stable_id,k,c,c_min,c_max,c_median,T25_sha256,contains_k<=-2,contains_k<=-3\n")
        for record in class_rows:
            for member, kid, cval in zip(record["member_stable_ids"], record["k_values"], record["c_values"]):
                handle.write(
                    f"{record['class_id']},{member},{kid},{cval:.10f},{record['c_min']:.10f},"
                    f"{record['c_max']:.10f},{record['c_median']:.10f},{record['T25_sha256']},"
                    f"{record['contains_k_le_minus2']},{record['contains_k_le_minus3']}\n"
                )
    write_json(out_dir / "witness_graphs.json", {"graphs": graph_payload, "checks": graph_rows})
    with open(out_dir / "witness_adjs.csv", "w", encoding="utf-8") as handle:
        handle.write("stable_id,edge_u,edge_v,atom_u,atom_v\n")
        for sid, payload in graph_payload.items():
            for (u, v) in payload["adjacency"]:
                handle.write(f"{sid},{u},{v},{payload['atom_types'][u]},{payload['atom_types'][v]}\n")
    log(
        f"[witnesses] conflict_classes={len(conflict_ids)} fit_k<=-3={len(tail_rows)} "
        f"dev_k<=-3={len(extremes_dev_positions)}"
    )
    return witnesses


def _dataset_graph(dataset: Sequence[Any], index: int) -> Any:
    import networkx as nx

    data = dataset[int(index)]
    edge_index = data.edge_index.detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    atom_types = np.asarray(data.x.detach().cpu().numpy()).reshape(-1).astype(int)
    for node in range(int(data.num_nodes)):
        graph.nodes[node]["t"] = int(atom_types[node])
    edge_attr = getattr(data, "edge_attr", None)
    bond_types = (
        np.asarray(edge_attr.detach().cpu().numpy()).reshape(int(data.num_edges), -1)[:, 0].astype(int)
        if edge_attr is not None
        else np.zeros(int(data.num_edges), np.int64)
    )
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == v:
            continue
        graph.add_edge(u, v, b=int(bond_types[k]))
    return graph


def _classify(graph_rows: Sequence[Mapping[str, Any]], tail_rows: Sequence[Mapping[str, Any]], witness_pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flags = {
        "INPUT_ALIASING_WITNESS": [],
        "ORDER_DEPENDENT_HELPER": [],
        "INPUT_EQUIVALENT_TARGET_CONFLICT": [],
        "UNRESOLVED_PROVENANCE": [],
    }
    for row in graph_rows:
        left, right = row["left"], row["right"]
        if not row["topology_only_isomorphic"] and row["c_diff"] > 0:
            flags["INPUT_ALIASING_WITNESS"].append(row["class_id"])
        if left["audit_cycle_score_stored_order"] != left["audit_cycle_score_gvae_order"] or right[
            "audit_cycle_score_stored_order"
        ] != right["audit_cycle_score_gvae_order"]:
            flags["ORDER_DEPENDENT_HELPER"].append(row["class_id"])
        if row["T25_equal"] and row["c_diff"] > 0:
            flags["INPUT_EQUIVALENT_TARGET_CONFLICT"].append(row["class_id"])
    if not flags["INPUT_ALIASING_WITNESS"] and not flags["ORDER_DEPENDENT_HELPER"]:
        flags["UNRESOLVED_PROVENANCE"].append("no witness pair classified")
    return {
        "flags": {key: sorted(set(value)) for key, value in flags.items()},
        "note": (
            "INPUT_ALIASING_WITNESS: identical T25, non-isomorphic graphs, different trusted c. "
            "ORDER_DEPENDENT_HELPER: the repo stored-order cycle helper differs from the canonical-order "
            "helper on the same molecule; the official label relation needs separate evidence. "
            "INPUT_EQUIVALENT_TARGET_CONFLICT: strictly identical model-visible T25 with different label c."
        ),
    }


# ---------------------------------------------------------------------------
# stage 3: node renumbering of the helper
# ---------------------------------------------------------------------------


def stage_renumber(out_dir: Path, log: Any = print) -> dict[str, Any]:
    import networkx as nx
    import pandas as pd

    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    witnesses = json.loads((out_dir / "cycle_class_witnesses.json").read_text())
    positions = sorted(
        {
            int(position)
            for pair in witnesses["witness_pairs"]
            for position in (pair["left_position"], pair["right_position"])
        }
    )
    if not positions:
        raise RuntimeError("no witness positions available; run --stage witnesses first")
    t25 = np.load(T25_PATH)
    with np.load(O_PRED_PATH) as o:
        fit_idx = np.asarray(o["fit_idx"], np.int64)
    dataset = _load_zinc(ZINC_ROOT, "train")
    audit = pd.read_csv(AUDIT_CSV).set_index("subset_index", drop=False)
    rng = np.random.default_rng(RENUMBER_SEED)
    rows: list[dict[str, Any]] = []
    for position in positions:
        global_id = int(fit_idx[position])
        data = dataset[global_id]
        graph = _graph_from_data(data)
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        base = _max_basis_stats(graph)
        values = {"max_basis_cycle_length": [], "basis_cycle_count": [], "cycle_score": []}
        score_list: list[float] = []
        max_len_list: list[float] = []
        basis_list: list[list[int]] = []
        for _ in range(RENUMBER_N):
            permutation = rng.permutation(len(nodes))
            shuffled = nx.Graph()
            shuffled.add_nodes_from([nodes[int(i)] for i in permutation])
            shuffled.add_edges_from(edges)
            stats = _max_basis_stats(shuffled)
            for key in values:
                values[key].append(stats[key])
            score_list.append(stats["cycle_score"])
            max_len_list.append(stats["max_basis_cycle_length"])
            basis_list.append(sorted(len(cycle) for cycle in nx.cycle_basis(shuffled)))
        rows.append(
            {
                "position": int(position),
                "stable_id": stable_id(global_id),
                "original_max_basis_cycle_length": float(base["max_basis_cycle_length"]),
                "original_cycle_score": float(base["cycle_score"]),
                "original_basis_cycle_count": float(base["basis_cycle_count"]),
                "perm_cycle_score_min": float(min(score_list)),
                "perm_cycle_score_max": float(max(score_list)),
                "perm_n_distinct_cycle_score": int(len(set(score_list))),
                "perm_max_basis_min": float(min(max_len_list)),
                "perm_max_basis_max": float(max(max_len_list)),
                "perm_n_distinct_max_basis": int(len(set(max_len_list))),
                "perm_basis_length_multisets": basis_list,
                "helper_cycle_score_normalized": [
                    float((value - COMMUNITY_CYCLE_MEAN) / COMMUNITY_CYCLE_STD) for value in score_list
                ],
                "audit_cycle_score_stored_order": float(audit.loc[global_id, "cycle_score_stored_order"]),
                "audit_cycle_score_gvae_order": float(audit.loc[global_id, "cycle_score_gvae_order"]),
                "label_effective_cycle_snapped": float(audit.loc[global_id, "label_effective_cycle_snapped"]),
                "order_sensitive": bool(len(set(score_list)) > 1 or len(set(max_len_list)) > 1),
            }
        )
    out = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "renumber",
        "seed": int(RENUMBER_SEED),
        "n_permutations": int(RENUMBER_N),
        "n_molecules": int(len(rows)),
        "constants_used_for_helper_normalization": {
            "cycle_mean": float(COMMUNITY_CYCLE_MEAN),
            "cycle_std": float(COMMUNITY_CYCLE_STD),
            "note": "helper normalization only; official label c is the label-derived column and is not changed here",
        },
        "molecules": rows,
        "n_order_sensitive": int(sum(1 for row in rows if row["order_sensitive"])),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "renumber_check.json", out)
    with open(out_dir / "renumber_check.csv", "w", encoding="utf-8") as handle:
        handle.write(
            "stable_id,perm_cycle_score_min,perm_cycle_score_max,n_distinct,perm_max_basis_min,perm_max_basis_max,"
            "n_distinct_max_basis,order_sensitive,audit_stored_order,audit_gvae_order,label_snapped\n"
        )
        for row in rows:
            handle.write(
                f"{row['stable_id']},{row['perm_cycle_score_min']},{row['perm_cycle_score_max']},"
                f"{row['perm_n_distinct_cycle_score']},{row['perm_max_basis_min']},{row['perm_max_basis_max']},"
                f"{row['perm_n_distinct_max_basis']},{row['order_sensitive']},"
                f"{row['audit_cycle_score_stored_order']},{row['audit_cycle_score_gvae_order']},"
                f"{row['label_effective_cycle_snapped']}\n"
            )
    log(f"[renumber] molecules={len(rows)} order_sensitive={out['n_order_sensitive']}")
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC cycle-conflict witnesses (train-only, no training)")
    parser.add_argument("--stage", default="all", choices=("bounds", "witnesses", "renumber", "all"))
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(int(args.threads))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.stage in ("bounds", "all"):
        stage_bounds(out_dir)
    if args.stage in ("witnesses", "all"):
        stage_witnesses(out_dir)
    if args.stage in ("renumber", "all"):
        stage_renumber(out_dir)
    log_line = f"[done] stage={args.stage} in {time.perf_counter() - started:.1f}s"
    print(log_line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
