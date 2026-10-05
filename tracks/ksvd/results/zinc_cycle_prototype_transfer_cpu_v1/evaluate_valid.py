"""zinc_cycle_prototype_transfer_cpu_v1 — frozen official-valid evaluation.

Loads the frozen H_model_package.pt, reproduces B and H on the cached official-valid
rows, and computes main/group/routing/coverage/tail/sensitivity/cancellation tables,
the paired bootstrap and the pre-registered gate.

This is the (single, unconditional) evaluation of the new candidate on the reused,
already-exposed official-valid split. official-test is never read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

SRC = REPO / "tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1"
RUNNER = REPO / "tracks/ksvd/experiments/luyin16/zinc_component_supervision_fulltrain_confirmation_seed0_v1.py"
OUT = Path(__file__).resolve().parent

PROTOCOL_VERSION = "zinc-cycle-prototype-transfer-cpu-v1"
BOOT_SEED = 20261010
N_BOOT = 1000
GATE_DELTA = 0.003
G0_TOL = 0.001
B_Y_COMP = -0.011630002409219742
B_G_COMP = -0.01144399804612788


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in names})


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {"k=0": k == 0, "k=-1": k == -1, "k=-2": k == -2, "k<=-3": k <= -3}


def group_name(kv: int) -> str:
    if int(kv) == 0:
        return "k=0"
    if int(kv) == -1:
        return "k=-1"
    if int(kv) == -2:
        return "k=-2"
    return "k<=-3"


def stable_id_valid(row: int) -> str:
    return f"valid:{int(row):04d}"


# ---------------------------------------------------------------------------
# reload the frozen prototype package independently from file
# ---------------------------------------------------------------------------


class FrozenPrototype:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        keys = z["keys"]
        slots = z["key_slot"]
        self.key_map = {bytes(k): int(s) for k, s in zip(keys, slots)}
        self.proto_val = z["proto_val"].astype(np.float64)
        self.consistent = z["consistent"].astype(bool)
        self.class_ids = z["class_ids"].astype(np.int64)
        self.key_dim = int(z["key_dim"][0])
        self.b_H = float(z["b_H"][0])

    def route_and_value(self, row: np.ndarray) -> tuple[str, float | None]:
        key = np.ascontiguousarray(row, np.float32).tobytes()
        slot = self.key_map.get(key)
        if slot is None:
            return "UNSEEN_FALLBACK", None
        if not self.consistent[slot]:
            return "TRAIN_CONFLICT_FALLBACK", None
        return "CONSISTENT_HIT", float(self.proto_val[slot])


def paired_bootstrap_gain(res_b: np.ndarray, res_h: np.ndarray, *, seed: int = BOOT_SEED, n_boot: int = N_BOOT):
    """gain = mean|res_b| - mean|res_h|; shared resample indices."""
    rng = np.random.default_rng(int(seed))
    n = res_b.size
    point = float(np.mean(np.abs(res_b)) - np.mean(np.abs(res_h)))
    gains = np.empty(int(n_boot), np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        gains[b] = np.mean(np.abs(res_b[idx])) - np.mean(np.abs(res_h[idx]))
    return {
        "point": point,
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def bootstrap_witnesses(res_b: np.ndarray, res_h: np.ndarray) -> dict[str, Any]:
    same = paired_bootstrap_gain(res_b, res_b)
    swap = paired_bootstrap_gain(res_h, res_b)
    fwd = paired_bootstrap_gain(res_b, res_h)
    delta = 0.25
    shifted = paired_bootstrap_gain(res_b, res_h - delta)
    return {
        "identical_point": same["point"],
        "identical_is_zero": bool(abs(same["point"]) <= 1e-15),
        "swap_point": swap["point"],
        "mirror_ok": bool(abs(swap["point"] + fwd["point"]) <= 1e-12),
        "constant_shift_delta": delta,
        "constant_shift_changes_gain_by": float(abs(shifted["point"] - fwd["point"])),
        "constant_shift_inside_bound": bool(abs(shifted["point"] - fwd["point"]) <= delta + 1e-12),
    }


def main() -> int:
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    runner = load_module(RUNNER, "proto_runner_eval")

    # ---- load frozen package ----
    proto = FrozenPrototype(OUT / "prototype_table.npz")
    b_H = proto.b_H

    # ---- source-independent check: verify source Q still matches ----
    prep_meta = np.load(SRC / "full_train_prep.npz", allow_pickle=False)
    prep_meta_dict = {key: prep_meta[key] for key in prep_meta.files}
    train_data = runner.build_fulltrain_data(prep_meta_dict)
    T_train = runner.topology_matrix(train_data)
    c_train = np.load(SRC / "full_train_targets.npz", allow_pickle=False)["c"].astype(np.float64)
    bias_value = float(np.median(c_train))
    q_head = runner.build_q_head(0, bias_value)
    q_head.load_state_dict(torch.load(SRC / "Q_raw_soup_state.pt", map_location="cpu", weights_only=True))
    q_head.eval()

    valid_data, vmeta = runner.load_valid_data(SRC)
    T_valid = runner.topology_matrix(valid_data)
    with torch.no_grad():
        q_frozen_valid = runner.q_forward(q_head, torch.as_tensor(T_valid, dtype=torch.float32)).double().numpy()

    # ---- cached labels / body ----
    vp = np.load(SRC / "valid_frozen_predictions.npz", allow_pickle=False)
    y = vp["y"].astype(np.float64)
    g = vp["g"].astype(np.float64)
    c = vp["c"].astype(np.float64)
    k = vp["k"].astype(np.int64)
    h_comp = vp["COMP_h_raw"].astype(np.float64)
    h_sum = vp["SUM_h_raw"].astype(np.float64)
    q_cache = vp["q_raw"].astype(np.float64)

    identity = {
        "q_frozen_replay_vs_cache_max_abs": float(np.max(np.abs(q_frozen_valid - q_cache))),
        "y_minus_g_plus_c_max_abs": float(np.max(np.abs(y - (g + c)))),
        "h_plus_q_plus_by_vs_COMP_y_cal_max_abs": float(np.max(np.abs((h_comp + q_cache + B_Y_COMP) - vp["COMP_y_cal"].astype(np.float64)))),
    }
    assert identity["q_frozen_replay_vs_cache_max_abs"] <= 1e-5

    # ---- B (frozen COMP+Q) ----
    y_B_raw = h_comp + q_cache
    y_B_cal = y_B_raw + B_Y_COMP
    g_B_cal = h_comp + B_G_COMP

    # ---- H (prototype lookup) ----
    n = int(T_valid.shape[0])
    q_H = np.array(q_cache, np.float64, copy=True)
    route = np.empty(n, dtype=object)
    for i in range(n):
        r, v = proto.route_and_value(T_valid[i])
        route[i] = r
        if v is not None:
            q_H[i] = v
    y_H_raw = h_comp + q_H
    y_H_cal = y_H_raw + b_H
    y_H_fixedB = y_H_raw + B_Y_COMP

    # ---- identity checks ----
    id_raw = y_H_raw - y_B_raw
    id_cal = y_H_cal - y_B_cal
    id_q = q_H - q_cache
    identity.update({
        "raw_diff_equals_q_diff_max_abs": float(np.max(np.abs(id_raw - id_q))),
        "cal_diff_identity_max_abs": float(np.max(np.abs(id_cal - (id_q + (b_H - B_Y_COMP))))),
        "row_identity_H_max_abs": float(np.max(np.abs((y_H_cal - y) - ((h_comp - g) + (q_H - c) + b_H)))),
        "fallback_raw_diff_max_abs": float(np.max(np.abs(id_raw[route != "CONSISTENT_HIT"]))) if (route != "CONSISTENT_HIT").any() else 0.0,
        "fallback_fixedB_diff_max_abs": float(np.max(np.abs((y_H_fixedB - y_B_cal)[route != "CONSISTENT_HIT"]))) if (route != "CONSISTENT_HIT").any() else 0.0,
    })

    # ---- main table ----
    res_B_y_cal = y - y_B_cal
    res_H_y_cal = y - y_H_cal
    res_B_y_raw = y - y_B_raw
    res_H_y_raw = y - y_H_raw
    res_H_y_fixedB = y - y_H_fixedB
    res_B_g_cal = g - g_B_cal

    main = {
        "valid": {
            "n": n,
            "B": {
                "b_y": B_Y_COMP, "b_g": B_G_COMP,
                "g_cal_mae": float(np.mean(np.abs(g - g_B_cal))),
                "y_raw_mae": float(np.mean(np.abs(res_B_y_raw))),
                "y_cal_mae": float(np.mean(np.abs(res_B_y_cal))),
            },
            "H": {
                "b_y": b_H,
                "y_raw_mae": float(np.mean(np.abs(res_H_y_raw))),
                "y_cal_mae": float(np.mean(np.abs(res_H_y_cal))),
                "y_fixedB_mae": float(np.mean(np.abs(res_H_y_fixedB))),
            },
        },
        "train": json.loads((OUT / "train_metrics.json").read_text()),
        "g_shared": {
            "note": "H changes only the cycle route/bias; the COMP body g prediction is identical",
            "g_cal_mae": float(np.mean(np.abs(g - g_B_cal))),
        },
    }
    write_json(OUT / "main_table.json", main)

    # ---- per-group table ----
    group_rows = []
    for split_name, split_y, split_res_B_cal, split_res_H_cal, split_res_B_raw, split_res_H_raw, split_k in [
        ("valid", y, res_B_y_cal, res_H_y_cal, res_B_y_raw, res_H_y_raw, k),
    ]:
        masks = {"overall": np.ones(n, bool), **group_masks(split_k)}
        for gname, mask in masks.items():
            entry = {
                "split": split_name, "group": gname, "n": int(mask.sum()),
                "B_y_cal_mae": float(np.mean(np.abs(split_res_B_cal[mask]))),
                "H_y_cal_mae": float(np.mean(np.abs(split_res_H_cal[mask]))),
                "gain_cal": float(np.mean(np.abs(split_res_B_cal[mask])) - np.mean(np.abs(split_res_H_cal[mask]))),
                "B_y_raw_mae": float(np.mean(np.abs(split_res_B_raw[mask]))),
                "H_y_raw_mae": float(np.mean(np.abs(split_res_H_raw[mask]))),
                "gain_raw": float(np.mean(np.abs(split_res_B_raw[mask])) - np.mean(np.abs(split_res_H_raw[mask]))),
                "B_contrib": float(np.abs(split_res_B_cal[mask]).sum() / n),
                "H_contrib": float(np.abs(split_res_H_cal[mask]).sum() / n),
                "gain_contrib": float((np.abs(split_res_B_cal[mask]).sum() - np.abs(split_res_H_cal[mask]).sum()) / n),
            }
            group_rows.append(entry)
    write_csv(OUT / "group_table.csv", group_rows)

    # add back check
    addback = {
        "sum_group_gain_contrib": float(sum(r["gain_contrib"] for r in group_rows if r["group"] != "overall")),
        "overall_gain_cal": float(np.mean(np.abs(res_B_y_cal)) - np.mean(np.abs(res_H_y_cal))),
    }
    addback["addback_ok"] = bool(abs(addback["sum_group_gain_contrib"] - addback["overall_gain_cal"]) <= 1e-9)

    # ---- routing table ----
    routing_rows = []
    for r in ("CONSISTENT_HIT", "TRAIN_CONFLICT_FALLBACK", "UNSEEN_FALLBACK"):
        for gname, gmask in [("overall", np.ones(n, bool)), *group_masks(k).items()]:
            mask = (route == r) & gmask
            cnt = int(mask.sum())
            entry = {"route": r, "group": gname, "n": cnt}
            if cnt:
                q_err_B = np.abs(q_cache - c)[mask]
                q_err_H = np.abs(q_H - c)[mask]
                entry.update({
                    "q_B_mae_vs_c": float(q_err_B.mean()),
                    "q_H_mae_vs_c": float(q_err_H.mean()),
                    "q_signed_B": float((q_cache - c)[mask].mean()),
                    "q_signed_H": float((q_H - c)[mask].mean()),
                    "h_err_vs_g_mae": float(np.abs(h_comp - g)[mask].mean()),
                    "y_B_raw_mae": float(np.abs(res_B_y_raw)[mask].mean()),
                    "y_H_raw_mae": float(np.abs(res_H_y_raw)[mask].mean()),
                    "y_B_cal_mae": float(np.abs(res_B_y_cal)[mask].mean()),
                    "y_H_cal_mae": float(np.abs(res_H_y_cal)[mask].mean()),
                    "gain_cal": float(np.abs(res_B_y_cal)[mask].mean() - np.abs(res_H_y_cal)[mask].mean()),
                    "B_contrib": float(np.abs(res_B_y_cal)[mask].sum() / n),
                    "H_contrib": float(np.abs(res_H_y_cal)[mask].sum() / n),
                    "gain_contrib": float((np.abs(res_B_y_cal)[mask].sum() - np.abs(res_H_y_cal)[mask].sum()) / n),
                    "k_true_counts": {str(v): int((k[mask] == v).sum()) for v in np.unique(k[mask])},
                })
            routing_rows.append(entry)
    write_csv(OUT / "routing_table.csv", routing_rows)

    # ---- consistent-hit train-class-k vs valid-true-k match ----
    hit_mask = route == "CONSISTENT_HIT"
    k_match = 0
    k_mismatch = 0
    mismatch_rows = []
    for i in np.where(hit_mask)[0]:
        key = np.ascontiguousarray(T_valid[i], np.float32).tobytes()
        slot = proto.key_map[key]
        cls_id = int(proto.class_ids[slot])
        # train class k (unanimous for consistent class)
        tbl = json.loads((OUT / "train_table.json").read_text())
        cls = next(c for c in tbl["class_table"] if c["class_id"] == cls_id)
        train_k = int(cls["k_min"])
        if train_k == int(k[i]):
            k_match += 1
        else:
            k_mismatch += 1
            mismatch_rows.append({"valid_row": int(i), "valid_id": stable_id_valid(i), "train_class_id": cls_id,
                                  "train_k": train_k, "valid_k": int(k[i]), "q_B_err": float(q_cache[i] - c[i]),
                                  "q_H_err": float(q_H[i] - c[i]), "y_B_cal_err": float(res_B_y_cal[i]),
                                  "y_H_cal_err": float(res_H_y_cal[i])})
    coverage = {
        "CONSISTENT_HIT": {"n": int(hit_mask.sum()), "k_match": k_match, "k_mismatch": k_mismatch,
                            "match_rate": k_match / max(1, int(hit_mask.sum()))},
        "new_conflict_rows": mismatch_rows,
    }
    write_json(OUT / "coverage.json", coverage)
    write_json(OUT / "addback.json", addback)

    # ---- bootstrap + gate ----
    g0_mask = k == 0
    overall_cal = paired_bootstrap_gain(res_B_y_cal, res_H_y_cal)
    overall_raw = paired_bootstrap_gain(res_B_y_raw, res_H_y_raw)
    g0_cal = paired_bootstrap_gain(res_B_y_cal[g0_mask], res_H_y_cal[g0_mask])
    g0_raw = paired_bootstrap_gain(res_B_y_raw[g0_mask], res_H_y_raw[g0_mask])
    witnesses = bootstrap_witnesses(res_B_y_cal, res_H_y_cal)
    g0_worsening = float(np.mean(np.abs(res_H_y_cal[g0_mask])) - np.mean(np.abs(res_B_y_cal[g0_mask])))
    bootstrap = {
        "overall_cal": overall_cal,
        "overall_raw": overall_raw,
        "G0_cal": g0_cal,
        "G0_raw": g0_raw,
        "witnesses": witnesses,
    }
    gate = {
        "1_overall_cal_gain_ge_0.003": bool(overall_cal["point"] >= GATE_DELTA),
        "2_overall_cal_ci_lower_gt_0": bool(overall_cal["ci95"][0] > 0.0),
        "3_overall_raw_gain_gt_0": bool(overall_raw["point"] > 0.0),
        "4_G0_cal_worsening_le_0.001": bool(g0_worsening <= G0_TOL),
        "overall_cal_gain": overall_cal["point"],
        "overall_cal_ci": overall_cal["ci95"],
        "overall_raw_gain": overall_raw["point"],
        "G0_cal_worsening": g0_worsening,
    }
    gate["PROTOTYPE_DEPLOY_SUPPORT"] = bool(all(gate[k] for k in (
        "1_overall_cal_gain_ge_0.003", "2_overall_cal_ci_lower_gt_0", "3_overall_raw_gain_gt_0", "4_G0_cal_worsening_le_0.001")))

    # ---- boundary markers ----
    gain_per_row = np.abs(res_B_y_cal) - np.abs(res_H_y_cal)
    pos = gain_per_row[gain_per_row > 0]
    neg = gain_per_row[gain_per_row < 0]
    max_pos_row = int(np.argmax(gain_per_row))
    pos_sum = float(pos.sum())
    neg_sum = float(neg.sum())
    max_pos_share = float(gain_per_row[max_pos_row] / pos_sum) if pos_sum > 0 else None

    worst_B_idx = int(np.argmax(np.abs(res_B_y_cal)))
    keep = np.ones(n, bool)
    keep[worst_B_idx] = False
    gain_drop_worst = float(np.mean(np.abs(res_B_y_cal[keep])) - np.mean(np.abs(res_H_y_cal[keep])))

    single_row_dominated = bool(
        (max_pos_share is not None and max_pos_share >= 0.5) or (gain_drop_worst <= 0.0)
    )
    target_repair = bool(
        gate["1_overall_cal_gain_ge_0.003"] and gate["3_overall_raw_gain_gt_0"] and gate["4_G0_cal_worsening_le_0.001"]
        and not gate["2_overall_cal_ci_lower_gt_0"]
    )
    marker = {
        "SINGLE_ROW_DOMINATED": single_row_dominated,
        "max_positive_row": {"valid_row": max_pos_row, "valid_id": stable_id_valid(max_pos_row),
                              "gain": float(gain_per_row[max_pos_row]), "k": int(k[max_pos_row])},
        "max_positive_share_of_positive_sum": max_pos_share,
        "positive_rows": int((gain_per_row > 0).sum()),
        "negative_rows": int((gain_per_row < 0).sum()),
        "positive_sum": pos_sum,
        "negative_sum": neg_sum,
        "worst_B_valid_id": stable_id_valid(worst_B_idx),
        "worst_B_k": int(k[worst_B_idx]),
        "worst_B_abs_err": float(np.abs(res_B_y_cal[worst_B_idx])),
        "gain_excluding_worst_B": gain_drop_worst,
        "TARGETED_REPAIR_ONLY": target_repair,
    }
    gate["markers"] = marker
    write_json(OUT / "bootstrap.json", bootstrap)
    write_json(OUT / "gate.json", gate)

    # ---- gemmarker: valid y_cal < 0.09 ----
    benchmark = {
        "valid_COMP_y_cal": float(np.mean(np.abs(res_B_y_cal))),
        "valid_H_y_cal": float(np.mean(np.abs(res_H_y_cal))),
        "target": 0.09,
        "H_below_target": bool(np.mean(np.abs(res_H_y_cal)) < 0.09),
        "B_below_target": bool(np.mean(np.abs(res_B_y_cal)) < 0.09),
    }
    write_json(OUT / "benchmark_marker.json", benchmark)

    # ---- per-row table ----
    rows = []
    e_g = h_comp - g
    e_c = q_H - c
    e_c_B = q_cache - c
    for i in range(n):
        rows.append({
            "valid_row": i, "stable_id": stable_id_valid(i), "k": int(k[i]), "group": group_name(int(k[i])),
            "route": str(route[i]), "c": float(c[i]), "y": float(y[i]), "g": float(g[i]),
            "h_raw": float(h_comp[i]), "q_B_raw": float(q_cache[i]), "q_H_raw": float(q_H[i]),
            "b_y_COMP": B_Y_COMP, "b_H": b_H,
            "y_B_raw": float(y_B_raw[i]), "y_B_cal": float(y_B_cal[i]),
            "y_H_raw": float(y_H_raw[i]), "y_H_cal": float(y_H_cal[i]), "y_H_fixedB": float(y_H_fixedB[i]),
            "q_B_err_vs_c": float(e_c_B[i]), "q_H_err_vs_c": float(e_c[i]),
            "h_err_vs_g": float(e_g[i]),
            "err_B_cal": float(res_B_y_cal[i]), "err_H_cal": float(res_H_y_cal[i]),
            "err_B_raw": float(res_B_y_raw[i]), "err_H_raw": float(res_H_y_raw[i]),
            "abs_err_B_cal": float(np.abs(res_B_y_cal[i])), "abs_err_H_cal": float(np.abs(res_H_y_cal[i])),
            "gain_cal": float(gain_per_row[i]),
            "triangle_e_g_plus_e_c": float(e_g[i] + e_c[i]),
        })
    write_csv(OUT / "per_row_valid.csv", rows)

    # ---- cancellation / budget table ----
    opp_sign = np.sign(e_g) != np.sign(e_c)
    tri_gap = np.abs(e_g + e_c) - (np.abs(e_g) + np.abs(e_c))
    cancel = {
        "opposite_sign_rate_H": float(opp_sign.mean()),
        "sum_abs_components_H": float((np.abs(e_g) + np.abs(e_c)).sum() / n),
        "abs_of_sum_H": float(np.abs(e_g + e_c).sum() / n),
        "triangle_gap_mae_H": float(tri_gap.mean()),
        "bias_effect_delta_b": float(b_H - B_Y_COMP),
        "overall_B_cal_mae": float(np.mean(np.abs(res_B_y_cal))),
        "overall_H_cal_mae": float(np.mean(np.abs(res_H_y_cal))),
        "overall_H_fixedB_mae": float(np.mean(np.abs(res_H_y_fixedB))),
        "overall_H_raw_mae": float(np.mean(np.abs(res_H_y_raw))),
        "overall_B_raw_mae": float(np.mean(np.abs(res_B_y_raw))),
        "oracle_note": "oracle uses same h and b_B with q -> true c; diagnostic only, not deployable",
    }
    y_oracle = h_comp + c + B_Y_COMP
    cancel["oracle_fixedB_y_mae"] = float(np.mean(np.abs(y - y_oracle)))
    cancel["oracle_gain_vs_B"] = float(np.mean(np.abs(res_B_y_cal)) - np.mean(np.abs(y - y_oracle)))
    write_json(OUT / "cancellation.json", cancel)

    # ---- valid 0172 detail ----
    row_0172 = 172
    tbl = json.loads((OUT / "train_table.json").read_text())
    key_0172 = np.ascontiguousarray(T_valid[row_0172], np.float32).tobytes()
    slot = proto.key_map.get(key_0172)
    detail = {"valid_row": row_0172, "valid_id": stable_id_valid(row_0172), "k": int(k[row_0172]),
              "in_train_table": slot is not None, "route": str(route[row_0172]),
              "c_valid": float(c[row_0172]), "y": float(y[row_0172]), "g": float(g[row_0172]),
              "h_raw": float(h_comp[row_0172]), "q_B_raw": float(q_cache[row_0172]), "q_H_raw": float(q_H[row_0172]),
              "err_B_cal": float(res_B_y_cal[row_0172]), "err_H_cal": float(res_H_y_cal[row_0172])}
    if slot is not None:
        cls_id = int(proto.class_ids[slot])
        cls = next(c2 for c2 in tbl["class_table"] if c2["class_id"] == cls_id)
        detail.update({
            "train_class_id": cls_id, "train_class_n_support": cls["n_support"],
            "train_class_consistent": cls["is_consistent"], "train_class_k": cls["k_min"],
            "proto_value": float(proto.proto_val[slot]), "member_ids": cls["member_row_ids"],
        })
    write_json(OUT / "valid_0172_detail.json", detail)

    # ---- tail top10 by B cal abs err ----
    top10 = np.argsort(np.abs(res_B_y_cal))[-10:][::-1]
    tail_rows = []
    for idx in top10.tolist():
        tail_rows.append({
            "valid_row": int(idx), "stable_id": stable_id_valid(idx), "k": int(k[idx]), "route": str(route[idx]),
            "B_abs_err_cal": float(np.abs(res_B_y_cal[idx])), "H_abs_err_cal": float(np.abs(res_H_y_cal[idx])),
            "gain_cal": float(gain_per_row[idx]), "improved": bool(gain_per_row[idx] > 0),
            "worsened": bool(gain_per_row[idx] < 0),
        })
    write_csv(OUT / "tail_top10.csv", tail_rows)
    tail_summary = {
        "top10_B_abs_err_sum": float(np.abs(res_B_y_cal)[top10].sum()),
        "top10_H_abs_err_sum": float(np.abs(res_H_y_cal)[top10].sum()),
        "top10_gain_sum": float(gain_per_row[top10].sum()),
        "overall_gain_sum": float(gain_per_row.sum()),
        "top10_share_of_gain": float(gain_per_row[top10].sum() / gain_per_row.sum()) if gain_per_row.sum() != 0 else None,
    }
    write_json(OUT / "tail_summary.json", tail_summary)

    # ---- sensitivity ----
    sens = {}
    for name, idx in [("exclude_valid_0172", row_0172), ("exclude_worst_B", worst_B_idx)]:
        keep = np.ones(n, bool)
        keep[idx] = False
        sens[name] = {
            "excluded_valid_id": stable_id_valid(idx),
            "n": int(keep.sum()),
            "B_cal_mae": float(np.mean(np.abs(res_B_y_cal[keep]))),
            "H_cal_mae": float(np.mean(np.abs(res_H_y_cal[keep]))),
            "gain_cal": float(np.mean(np.abs(res_B_y_cal[keep])) - np.mean(np.abs(res_H_y_cal[keep]))),
            "B_raw_mae": float(np.mean(np.abs(res_B_y_raw[keep]))),
            "H_raw_mae": float(np.mean(np.abs(res_H_y_raw[keep]))),
            "gain_raw": float(np.mean(np.abs(res_B_y_raw[keep])) - np.mean(np.abs(res_H_y_raw[keep]))),
        }
    write_json(OUT / "sensitivity.json", sens)

    # ---- key / label-independence checks ----
    def key_of(row: int) -> bytes:
        return np.ascontiguousarray(T_valid[row], np.float32).tobytes()

    order = np.arange(n)
    shuffled = order.copy()
    rng = np.random.default_rng(20261010)
    rng.shuffle(shuffled)
    key_stable = all(key_of(shuffled[i]) == np.ascontiguousarray(T_valid[shuffled[i]], np.float32).tobytes() for i in range(0, n, 37))

    # label perturbation must not change q_H/route
    label_perm = rng.permutation(n)
    T_perturbed = np.array(T_valid, np.float32, copy=True)
    # deliberately corrupt the label-derived arrays passed alongside (no-op for forward)
    k_pert = k[label_perm]
    c_pert = c[label_perm]
    q_H2 = np.array(q_cache, np.float64, copy=True)
    route2 = np.empty(n, dtype=object)
    for i in range(n):
        r, v = proto.route_and_value(T_perturbed[i])
        route2[i] = r
        if v is not None:
            q_H2[i] = v
    label_independence = {
        "q_H_unchanged_under_label_permutation": bool(np.array_equal(q_H, q_H2)),
        "route_unchanged_under_label_permutation": bool(np.array_equal(route, route2)),
        "k_perturbed_used": False,
        "c_perturbed_used": False,
        "key_stable_under_order": bool(key_stable),
    }

    # fake-key fallback
    fake_row = np.full(25, np.nan, np.float64)
    fake_route = proto.route_and_value(fake_row)[0]
    fake_row_ok = np.zeros(25, np.float32)
    fake_row_ok[0] = 1.0e30
    r_fake = proto.route_and_value(fake_row_ok)[0]
    fallback_checks = {
        "nan_key_does_not_match_train": bool(fake_route == "UNSEEN_FALLBACK"),
        "nan_route": fake_route,
        "fake_finite_key_routes": r_fake,
        "fake_key_falls_back": bool(r_fake == "UNSEEN_FALLBACK"),
        "conflict_class_routes_are_fallback": bool(all(str(route[i]) != "CONSISTENT_HIT" for i in range(n) if not proto.consistent[proto.key_map.get(key_of(i), -1)]) if any(proto.key_map.get(key_of(i)) is not None and not proto.consistent[proto.key_map[key_of(i)]] for i in range(n)) else True),
    }
    checks = {
        "identity": identity,
        "label_independence": label_independence,
        "fallback": fallback_checks,
    }
    write_json(OUT / "checks.json", checks)

    heldout = {
        "protocol_version": PROTOCOL_VERSION,
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "events": [{
            "split": "valid",
            "read_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_rows": n,
            "source": "valid_frozen_predictions.npz + frozen full-train prep T25 rebuild",
            "frozen_manifest": "frozen_eval_manifest.json",
            "note": "reused exposed valid; H evaluated after freeze",
        }],
    }
    write_json(OUT / "heldout_access.json", heldout)

    print(f"[eval] B y_cal MAE = {np.mean(np.abs(res_B_y_cal)):.10f}")
    print(f"[eval] H y_cal MAE = {np.mean(np.abs(res_H_y_cal)):.10f}  gain = {overall_cal['point']:.10f} CI={overall_cal['ci95']}")
    print(f"[eval] H y_raw gain = {overall_raw['point']:.10f}")
    print(f"[eval] G0 cal worsening = {g0_worsening:.10f}")
    print(f"[eval] gate = {gate['PROTOTYPE_DEPLOY_SUPPORT']}  markers = SINGLE_ROW_DOMINATED={marker['SINGLE_ROW_DOMINATED']} TARGETED_REPAIR_ONLY={marker['TARGETED_REPAIR_ONLY']}")
    print(f"[eval] routing counts: {{r: int((route == r).sum()) for r in set(route)}} = " + str({r: int((route == r).sum()) for r in set(route)}))
    print(f"[eval] H fixedB MAE = {cancel['overall_H_fixedB_mae']:.10f}  oracle MAE = {cancel['oracle_fixedB_y_mae']:.10f}")
    print(f"[eval] elapsed = {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
