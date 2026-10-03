"""Stage A — cycle-input ambiguity audit for the ZINC joint-dictionary round.

Read-only, CPU.  Never instantiates the official test split.

Deliverables (in this directory):
  cycle_input_classes.csv        one row per queried valid graph (35 ring + severe)
  cycle_class_members.csv        one row per (query, class-member) pair
  cycle_probe_predictions.csv    OOF + valid probe predictions
  cycle_input_decision.json      machine-readable summary

Run:
    uv run python -m tracks.ksvd.results.zinc_joint_dictionary_decision_v1.cycle_input_audit
"""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

OUT = REPO_ROOT / "tracks/ksvd/results/zinc_joint_dictionary_decision_v1"
HANDOFF = REPO_ROOT / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"
CYCLE = REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit"
OVERNIGHT = REPO_ROOT / "tracks/ksvd/results/zinc_overnight_bottleneck_v1"

TOPO_SLICE = slice(806, 814)
# The two historical grouping schemes (kept strictly separate).
SEVERE_MAX = -2          # <= -2  -> the 5 severe valid rows
PROBE_TREES = 256
PROBE_LEAF = 2
PROBE_MAX_FEATURES = 1.0
PROBE_JOBS = 8
PROBE_SEED = 0
FOLD_SEED = 20261003


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _sha256_bytes(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def _keys(matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Row keys from the exact float32 byte pattern (strict equality)."""
    assert matrix.dtype == np.float32
    assert matrix.flags["C_CONTIGUOUS"]
    keys = [matrix[i].tobytes() for i in range(matrix.shape[0])]
    return keys, matrix


def _median_of(vals: np.ndarray) -> float:
    return float(np.median(vals)) if len(vals) else float("nan")


# ---------------------------------------------------------------------------
# A1 — identity and accounting
# ---------------------------------------------------------------------------


def stage_a1() -> dict[str, Any]:
    pair = pd.read_csv(OVERNIGHT / "pair_valid_predictions.csv")
    valid = np.load(HANDOFF / "valid.npz", allow_pickle=True)
    lab = pd.read_csv(CYCLE / "valid_cycle_audit_label.csv")
    sev = pd.read_csv(OVERNIGHT / "severity_table.csv")
    out: dict[str, Any] = {"n_rows": int(len(pair)), "official_test_loaded": False}

    # identity
    out["ids_are_row_order"] = bool(np.array_equal(pair["graph_id"].to_numpy(), np.arange(len(pair))))
    out["y_pair_vs_handoff_max_abs"] = float(np.max(np.abs(pair["y"].to_numpy() - valid["y"])))
    out["y_handoff_vs_label_max_abs"] = float(np.max(np.abs(valid["y"] - lab["target"].to_numpy())))
    out["ids_label_vs_handoff"] = bool(np.array_equal(lab["subset_index"].to_numpy(), valid["ids"]))
    pen = np.round(lab["label_effective_cycle_snapped"].to_numpy()).astype(int)
    out["valid_penalty_counts"] = {str(k): int(v) for k, v in sorted(collections.Counter(pen).items())}
    out["severe_indices"] = [int(i) for i in np.where(pen <= SEVERE_MAX)[0]]
    out["ring_indices_all"] = [int(i) for i in np.where(pen < 0)[0]]
    out["ring_indices_ex172"] = [int(i) for i in np.where((pen < 0) & (np.arange(len(pen)) != 172))[0]]

    # bias consistency on every arm (cal - raw should be the arm's constant bias)
    for col in pair.columns:
        if col.endswith("_cal") and col.replace("_cal", "_raw") in pair.columns:
            arm = col[:-4]
            diff = pair[col].to_numpy() - pair[col.replace("_cal", "_raw")].to_numpy()
            out.setdefault("bias_consistency", {})[arm] = {
                "min": float(diff.min()), "max": float(diff.max()),
                "spread": float(diff.max() - diff.min()),
            }

    # recompute group accounting for N0_s0 under both schemes
    def account(mask: np.ndarray, cal: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        err = np.abs(cal - y)
        return {"n": int(mask.sum()), "mae": float(err[mask].mean()) if mask.sum() else float("nan"),
                "contribution": float(err[mask].sum() / len(y))}

    cal = pair["N0_s0_cal"].to_numpy()
    y = pair["y"].to_numpy()
    scheme_sev = {
        "penalty_0": np.array(pen == 0),
        "penalty_-1": np.array(pen == -1),
        "penalty_le_-2": np.array(pen <= SEVERE_MAX),
    }
    scheme_grp = {
        "G0": np.array(pen == 0),
        "G1_ex172": np.array((pen < 0) & (np.arange(len(pen)) != 172)),
        "G172": np.array(np.arange(len(pen)) == 172),
    }
    out["n0_s0_priority_severity"] = {k: account(m, cal, y) for k, m in scheme_sev.items()}
    out["n0_s0_priority_severity"]["overall"] = account(np.ones(len(pen), bool), cal, y)
    out["n0_s0_group_partition"] = {k: account(m, cal, y) for k, m in scheme_grp.items()}
    out["n0_s0_group_partition"]["overall"] = account(np.ones(len(pen), bool), cal, y)
    out["n0_s0_sum_severity_contrib"] = float(
        out["n0_s0_priority_severity"]["penalty_0"]["contribution"]
        + out["n0_s0_priority_severity"]["penalty_-1"]["contribution"]
        + out["n0_s0_priority_severity"]["penalty_le_-2"]["contribution"])
    out["n0_s0_sum_group_contrib"] = float(
        out["n0_s0_group_partition"]["G0"]["contribution"]
        + out["n0_s0_group_partition"]["G1_ex172"]["contribution"]
        + out["n0_s0_group_partition"]["G172"]["contribution"])

    # published severity rows for cross-check
    pub = sev[(sev["arm"] == "N0_s0")]
    out["severity_table_n0_s0"] = {r["stratum"]: {"n": int(r["n"]), "mae": float(r["mae"]),
                                                  "contribution": float(r["contribution"])}
                                   for _, r in pub.iterrows()}

    # ERRATA cross-check: REPORT section 5 labels
    out["errata_report_section5_labels"] = {
        "report_section5_G1_value": 0.0046438747644424435,
        "report_section5_G172_value": 0.02248182761669159,
        "report_G1_value_equals_penalty_-1": bool(abs(
            0.0046438747644424435
            - out["n0_s0_priority_severity"]["penalty_-1"]["contribution"]) < 1e-12),
        "report_G172_value_equals_penalty_le_-2": bool(abs(
            0.02248182761669159
            - out["n0_s0_priority_severity"]["penalty_le_-2"]["contribution"]) < 1e-12),
        "correct_G1_ex172_contribution": out["n0_s0_group_partition"]["G1_ex172"]["contribution"],
        "correct_G172_contribution": out["n0_s0_group_partition"]["G172"]["contribution"],
        "verdict": (
            "REPORT section 5 prints the 30/5 severity-stratum contributions under the "
            "G1(34)/G172(1) labels; the row-set labels and the printed numbers disagree."),
    }
    return out


def replay_n0_s0() -> dict[str, Any]:
    """Optional replay of the real N0 seed0 soup (no training)."""
    soup_path = OVERNIGHT / "N0_s0/soup_state.pt"
    if not soup_path.exists():
        return {"status": "MISSING_CHECKPOINT", "path": str(soup_path)}
    import torch
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1lib
    from tracks.ksvd.experiments.luyin16 import zinc_overnight_bottleneck_v1 as onb
    from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun

    torch.set_num_threads(8)
    device = torch.device("cpu")
    dictionary = uprun._dictionary_tensor()
    subspace = uprun._load_parent_subspace()
    model = onb.build_readout_model(dictionary, subspace)
    state = torch.load(soup_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()

    def collect(data):
        loader = p1lib.make_env_loader(data, 128, False, onb.SEED + int(uprun.EVAL_SHUFFLE_OFFSET))
        preds, ys = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
                ys.append(batch.y.view(-1).detach().cpu())
        return torch.cat(preds).numpy(), torch.cat(ys).numpy()

    train = uprun.load_split("control", "train")
    valid = uprun.load_split("control", "valid")
    raw_t, y_t = collect(train)
    raw_v, y_v = collect(valid)
    bias = float(np.median(y_t - raw_t))
    cal_v = raw_v + bias
    return {
        "status": "REPLAYED",
        "raw_valid_mae": float(np.mean(np.abs(raw_v - y_v))),
        "calibrated_valid_mae": float(np.mean(np.abs(cal_v - y_v))),
        "train_fitted_bias": bias,
        "raw_train_mae": float(np.mean(np.abs(raw_t - y_t))),
        "published_cal_valid_mae": 0.1112061332334415,
        "published_raw_valid_mae": 0.1107364371418953,
        "n_valid": int(len(y_v)),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# A2 — exact topology25 equivalence classes
# ---------------------------------------------------------------------------


def stage_a2() -> dict[str, Any]:
    train = np.load(HANDOFF / "train.npz", allow_pickle=True)
    valid = np.load(HANDOFF / "valid.npz", allow_pickle=True)
    labt = pd.read_csv(CYCLE / "train_cycle_audit_label.csv")
    labv = pd.read_csv(CYCLE / "valid_cycle_audit_label.csv")
    pen_t = np.round(labt["label_effective_cycle_snapped"].to_numpy()).astype(int)
    pen_v = np.round(labv["label_effective_cycle_snapped"].to_numpy()).astype(int)

    raw_t = np.ascontiguousarray(train["topo_raw"], dtype=np.float32)
    raw_v = np.ascontiguousarray(valid["topo_raw"], dtype=np.float32)
    nrm_t = np.ascontiguousarray(train["topo_model_input"], dtype=np.float32)
    nrm_v = np.ascontiguousarray(valid["topo_model_input"], dtype=np.float32)

    out: dict[str, Any] = {
        "shapes": {"raw_t": list(raw_t.shape), "raw_v": list(raw_v.shape),
                   "nrm_t": list(nrm_t.shape), "nrm_v": list(nrm_v.shape)},
        "dtypes": {"raw": str(raw_t.dtype), "normalized": str(nrm_t.dtype)},
        "sha256": {"raw_t": _sha256_bytes(raw_t), "raw_v": _sha256_bytes(raw_v),
                   "nrm_t": _sha256_bytes(nrm_t), "nrm_v": _sha256_bytes(nrm_v)},
        "official_test_loaded": False,
    }

    # strict-equality classes
    raw_keys_t, _ = _keys(raw_t)
    nrm_keys_t, _ = _keys(nrm_t)
    classes_raw = collections.defaultdict(list)
    classes_nrm = collections.defaultdict(list)
    for i, k in enumerate(raw_keys_t):
        classes_raw[k].append(i)
    for i, k in enumerate(nrm_keys_t):
        classes_nrm[k].append(i)
    out["n_classes_raw_train"] = int(len(classes_raw))
    out["n_classes_nrm_train"] = int(len(classes_nrm))
    out["raw_nrm_class_partition_identical"] = bool(
        sorted(map(tuple, classes_raw.values())) == sorted(map(tuple, classes_nrm.values())))
    # normalized is a per-column affine of raw: verify the class maps coincide
    out["max_class_size_raw"] = int(max(len(v) for v in classes_raw.values()))

    # raw-vs-normalized transform is fixed (train-only standardizer). Record it
    # exactly as the affine that maps *raw* train to normalized train.
    A = np.column_stack([raw_t, np.ones(len(raw_t))])
    coef, *_ = np.linalg.lstsq(A, nrm_t, rcond=None)
    out["standardizer_affine_residual_max"] = float(np.max(np.abs(A @ coef - nrm_t)))
    out["standardizer_slope"] = coef[:25].tolist()
    out["standardizer_intercept"] = coef[25].tolist()

    # query sets
    severe5 = [int(i) for i in np.where(pen_v <= SEVERE_MAX)[0]]
    ring35 = [int(i) for i in np.where(pen_v < 0)[0]]
    queries = sorted(set(severe5) | set(ring35))
    out["queries_severe5"] = severe5
    out["queries_ring35"] = ring35

    # class rows
    class_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for qi in queries:
        for label, keys_t, pen_t_, in (
            ("raw", raw_keys_t, pen_t),
            ("normalized", nrm_keys_t, pen_t),
        ):
            kv = (raw_v if label == "raw" else nrm_v)[qi].tobytes()
            members = [i for i, k in enumerate(keys_t) if k == kv]
            p = int(pen_v[qi])
            dist = collections.Counter(int(pen_t_[m]) for m in members)
            med = _median_of(pen_t_[members]) if members else float("nan")
            class_rows.append({
                "valid_index": int(qi),
                "molecule_id": str(labv["molecule_id"].iloc[qi]),
                "valid_penalty": int(p),
                "key_space": label,
                "n_exact_members": int(len(members)),
                "status": "MATCHED" if members else "NO_EXACT_MATCH",
                "class_penalty_counts": json.dumps({str(k): int(v) for k, v in sorted(dist.items())}),
                "class_n_penalty_0": int(dist.get(0, 0)),
                "class_n_penalty_-1": int(dist.get(-1, 0)),
                "class_n_penalty_le_-2": int(sum(v for k, v in dist.items() if k <= SEVERE_MAX)),
                "class_conditional_median": med,
                "valid_matches_median": bool(members and float(med) == float(p)),
                "class_penalty_mixed": bool(len(dist) > 1),
            })
            for m in members:
                member_rows.append({
                    "valid_index": int(qi),
                    "molecule_id": str(labv["molecule_id"].iloc[qi]),
                    "valid_penalty": int(p),
                    "key_space": label,
                    "train_index": int(m),
                    "train_molecule_id": str(labt["molecule_id"].iloc[m]),
                    "train_penalty": int(pen_t[m]),
                    "train_y_without_cycle": float(labt["y_without_cycle_label"].iloc[m]),
                    "valid_y_without_cycle": float(labv["y_without_cycle_label"].iloc[qi]),
                    "same_penalty_as_valid": bool(int(pen_t[m]) == int(p)),
                })

    out["n_query_pairs"] = len(class_rows)
    out["n_matched"] = int(sum(1 for r in class_rows if r["status"] == "MATCHED"))
    out["n_no_exact_match"] = int(sum(1 for r in class_rows if r["status"] == "NO_EXACT_MATCH"))
    out["n_class_penalty_mixed"] = int(sum(1 for r in class_rows if r["class_penalty_mixed"]))

    # global picture: every train class, not only the query classes
    all_mixed = []
    for members in classes_raw.values():
        pens = sorted(set(int(pen_t[m]) for m in members))
        if len(pens) > 1:
            all_mixed.append({
                "class_size": int(len(members)),
                "penalties": pens,
                "penalty_counts": {str(k): int(sum(1 for m in members if int(pen_t[m]) == k)) for k in pens},
                "member_train_ids": [int(m) for m in sorted(members)],
                "member_penalties": [int(pen_t[m]) for m in sorted(members)],
            })
    out["all_mixed_train_classes"] = sorted(all_mixed, key=lambda r: -r["class_size"])
    out["n_mixed_train_classes_global"] = int(len(all_mixed))
    out["n_mixed_train_rows_global"] = int(sum(r["class_size"] for r in all_mixed))
    out["n_pure_train_classes_global"] = int(len(classes_raw) - len(all_mixed))

    # the decisive pair train:3776 / valid:0172, explicitly
    def class_of_train(idx: int) -> list[int]:
        k = raw_keys_t[idx]
        return classes_raw.get(k, [])

    t3776 = 3776
    pair_analysis = {
        "train_3776_penalty": int(pen_t[t3776]),
        "train_3776_class_size": int(len(class_of_train(t3776))),
        "train_3776_class_members": [int(x) for x in class_of_train(t3776)],
        "valid_172_penalty": int(pen_v[172]),
        "valid_172_class_size": int(len(keys_t_lookup(raw_keys_t, raw_v[172].tobytes()))),
        "valid_172_class_members": [int(x) for x in keys_t_lookup(raw_keys_t, raw_v[172].tobytes())],
        "same_key_3776_172": bool(raw_t[t3776].tobytes() == raw_v[172].tobytes()),
        "same_norm_key_3776_172": bool(nrm_t[t3776].tobytes() == nrm_v[172].tobytes()),
        "raw_bitwise_equal_3776_172": bool(np.array_equal(raw_t[t3776], raw_v[172])),
        "norm_bitwise_equal_3776_172": bool(np.array_equal(nrm_t[t3776], nrm_v[172])),
    }
    out["decisive_pair"] = pair_analysis

    # approximate (not exact) neighbour check, kept strictly separate
    approx_rows = []
    for qi in ring35:
        d = np.linalg.norm(raw_t.astype(np.float64) - raw_v[qi].astype(np.float64), axis=1)
        order = np.argsort(d, kind="stable")[:5]
        approx_rows.append({
            "valid_index": int(qi),
            "molecule_id": str(labv["molecule_id"].iloc[qi]),
            "valid_penalty": int(pen_v[qi]),
            "approx_nn_ids": [int(x) for x in order],
            "approx_nn_dist": [float(d[x]) for x in order],
            "approx_nn_penalty": [int(pen_t[x]) for x in order],
            "approx_exact_dist_zero_count": int(np.sum(d < 1e-12)),
        })
    out["approximate_neighbours"] = approx_rows

    # chemical remainder as diagnostic only
    ywc_v = labv["y_without_cycle_label"].to_numpy()
    ywc_t = labt["y_without_cycle_label"].to_numpy()
    out["y_without_cycle_diagnostic"] = {
        "valid_severe5": [float(ywc_v[i]) for i in severe5],
        "train_global_mean": float(np.mean(ywc_t)), "train_global_std": float(np.std(ywc_t)),
        "valid_ring35_mean": float(np.mean(ywc_v[ring35])),
    }

    pd.DataFrame(class_rows).to_csv(OUT / "cycle_input_classes.csv", index=False)
    pd.DataFrame(member_rows).to_csv(OUT / "cycle_class_members.csv", index=False)
    return out, class_rows, member_rows, pen_t, pen_v, raw_t, raw_v, nrm_t, nrm_v


def keys_t_lookup(keys_t: list[str], key: bytes) -> list[int]:
    return [i for i, k in enumerate(keys_t) if k == key]


# ---------------------------------------------------------------------------
# A3 — train-only composition probe
# ---------------------------------------------------------------------------


def _fold_assignment(train: np.ndarray, labt: pd.DataFrame) -> np.ndarray:
    """Two molecule-level folds from the verified canonical group key.

    Groups (not rows) are assigned so that duplicate molecules never straddle
    the fold boundary.  Deterministic: stable hash of the group key + fold seed.
    """
    gid = train["canonical_group_id"].astype(np.int64)
    fold = np.array([
        int.from_bytes(hashlib.sha256(f"{FOLD_SEED}:{int(g)}".encode()).digest()[:8], "big") % 2
        for g in gid.tolist()
    ], dtype=np.int64)
    return fold


def stage_a3(train: np.ndarray, raw_t: np.ndarray, raw_v: np.ndarray,
             nrm_t: np.ndarray, pen_t: np.ndarray, pen_v: np.ndarray,
             labv: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.ensemble import ExtraTreesRegressor

    out: dict[str, Any] = {"official_test_loaded": False}
    fold = _fold_assignment(train, None)
    out["fold_sizes"] = {str(k): int(v) for k, v in sorted(collections.Counter(fold).items())}
    pairs = list(zip(train["canonical_group_id"].tolist(), fold.tolist()))
    out["fold_group_sizes"] = {
        "n_groups_fold0": int(len(set(g for g, f in pairs if f == 0))),
        "n_groups_fold1": int(len(set(g for g, f in pairs if f == 1))),
    }

    # exact-class median lookup, cross-fold
    lookup_pred = np.full(len(pen_t), np.nan)
    lookup_hit = np.zeros(len(pen_t), bool)
    for f in (0, 1):
        fit_idx = np.where(fold != f)[0]
        hold_idx = np.where(fold == f)[0]
        table: dict[bytes, list[int]] = collections.defaultdict(list)
        for i in fit_idx:
            table[raw_t[i].tobytes()].append(int(pen_t[i]))
        for i in hold_idx:
            m = table.get(raw_t[i].tobytes())
            if m:
                lookup_pred[i] = float(np.median(m))
                lookup_hit[i] = True
    out["lookup_oof_covered"] = int(lookup_hit.sum())
    out["lookup_oof_coverage"] = float(lookup_hit.mean())
    out["lookup_oof_mae_covered"] = float(np.mean(np.abs(lookup_pred[lookup_hit] - pen_t[lookup_hit]))) if lookup_hit.any() else float("nan")
    out["lookup_oof_exact_frac_covered"] = float(np.mean(lookup_pred[lookup_hit] == pen_t[lookup_hit])) if lookup_hit.any() else float("nan")

    # ExtraTrees, two folds
    et_pred = np.full(len(pen_t), np.nan)
    et = ExtraTreesRegressor(n_estimators=PROBE_TREES, min_samples_leaf=PROBE_LEAF,
                             max_features=PROBE_MAX_FEATURES, random_state=PROBE_SEED,
                             n_jobs=PROBE_JOBS)
    for f in (0, 1):
        fit_idx = np.where(fold != f)[0]
        hold_idx = np.where(fold == f)[0]
        et.fit(raw_t[fit_idx].astype(np.float64), pen_t[fit_idx])
        et_pred[hold_idx] = et.predict(raw_t[hold_idx].astype(np.float64))
    out["et_oof_mae"] = float(np.mean(np.abs(et_pred - pen_t)))
    out["et_oof_medae"] = float(np.median(np.abs(et_pred - pen_t)))
    out["et_oof_r2"] = float(1.0 - np.sum((et_pred - pen_t) ** 2) / np.sum((pen_t - pen_t.mean()) ** 2))

    # full-train refit -> valid (evaluate once)
    et_full = ExtraTreesRegressor(n_estimators=PROBE_TREES, min_samples_leaf=PROBE_LEAF,
                                  max_features=PROBE_MAX_FEATURES, random_state=PROBE_SEED,
                                  n_jobs=PROBE_JOBS)
    et_full.fit(raw_t.astype(np.float64), pen_t)
    et_valid = et_full.predict(raw_v.astype(np.float64))
    et_train = et_full.predict(raw_t.astype(np.float64))
    out["et_train_mae"] = float(np.mean(np.abs(et_train - pen_t)))
    out["et_valid_mae"] = float(np.mean(np.abs(et_valid - pen_v)))
    out["et_valid_medae"] = float(np.median(np.abs(et_valid - pen_v)))

    # exact-class lookup refit on all train -> valid
    table_all: dict[bytes, list[int]] = collections.defaultdict(list)
    for i in range(len(pen_t)):
        table_all[raw_t[i].tobytes()].append(int(pen_t[i]))
    lookup_valid = np.full(len(pen_v), np.nan)
    for i in range(len(pen_v)):
        m = table_all.get(raw_v[i].tobytes())
        if m:
            lookup_valid[i] = float(np.median(m))
    out["lookup_valid_coverage"] = float(np.mean(~np.isnan(lookup_valid)))
    cov = ~np.isnan(lookup_valid)
    out["lookup_valid_mae_covered"] = float(np.mean(np.abs(lookup_valid[cov] - pen_v[cov]))) if cov.any() else float("nan")

    # group-wise breakdown (0 / -1 / <=-2) for OOF and valid
    def group_breakdown(pred: np.ndarray, pen: np.ndarray, mask_valid: np.ndarray | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for name, m in (("0", pen == 0), ("-1", pen == -1), ("le_-2", pen <= SEVERE_MAX)):
            p, t = pred[m], pen[m]
            if len(p):
                res[name] = {"n": int(len(p)), "mae": float(np.mean(np.abs(p - t))),
                             "exact_frac": float(np.mean(p == t))}
        return res

    out["et_oof_by_group"] = group_breakdown(et_pred, pen_t)
    out["et_valid_by_group"] = group_breakdown(et_valid, pen_v)
    out["lookup_valid_by_group"] = group_breakdown(lookup_valid, pen_v)

    # normalized cycle-component error with the fixed (data-derived) affine map
    comp_by_pen = {}
    for _, r in labv.iterrows():
        comp_by_pen[int(round(r["label_effective_cycle_snapped"]))] = float(r["label_cycle_component"])
    # fit affine from the fixed lattice points only (no valid fitting of the probe)
    pts = np.array(sorted(comp_by_pen.items()), dtype=np.float64)
    slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
    out["cycle_component_affine"] = {"slope": float(slope), "intercept": float(intercept),
                                     "points": {str(int(k)): float(v) for k, v in comp_by_pen.items()}}
    comp_t = slope * pen_t + intercept
    out["et_oof_cycle_component_mae"] = float(np.mean(np.abs(slope * et_pred + intercept - comp_t)))
    out["et_valid_cycle_component_mae"] = float(np.mean(np.abs(slope * et_valid + intercept - (slope * pen_v + intercept))))

    # rows for the CSV
    rows = []
    for i in range(len(pen_t)):
        rows.append({"split": "train_oof", "index": i, "penalty_true": int(pen_t[i]),
                     "penalty_lookup": float(lookup_pred[i]), "penalty_et": float(et_pred[i]),
                     "lookup_covered": bool(lookup_hit[i])})
    for i in range(len(pen_v)):
        rows.append({"split": "valid", "index": i, "penalty_true": int(pen_v[i]),
                     "penalty_lookup": float(lookup_valid[i]), "penalty_et": float(et_valid[i]),
                     "lookup_covered": bool(~np.isnan(lookup_valid[i]))})
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"official_test_loaded": False}
    payload["A1"] = stage_a1()
    payload["A1_replay"] = replay_n0_s0()
    a2, class_rows, member_rows, pen_t, pen_v, raw_t, raw_v, nrm_t, _ = stage_a2()
    payload["A2"] = {k: v for k, v in a2.items()}
    a3, probe_df = stage_a3(np.load(HANDOFF / "train.npz", allow_pickle=True),
                            raw_t, raw_v, nrm_t, pen_t, pen_v,
                            pd.read_csv(CYCLE / "valid_cycle_audit_label.csv"))
    payload["A3"] = a3
    probe_df.to_csv(OUT / "cycle_probe_predictions.csv", index=False)
    payload["probe_config"] = {
        "input": "raw topo_raw 25-D only",
        "target": "train raw cycle penalty (label_effective_cycle_snapped)",
        "folds": 2, "fold_seed": FOLD_SEED, "fold_unit": "canonical_group_id",
        "extratrees": {"n_estimators": PROBE_TREES, "min_samples_leaf": PROBE_LEAF,
                       "max_features": PROBE_MAX_FEATURES, "random_state": PROBE_SEED,
                       "n_jobs": PROBE_JOBS},
        "sklearn_version": __import__("sklearn").__version__,
    }
    payload["seconds"] = float(time.perf_counter() - started)
    (OUT / "cycle_input_decision.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    # console summary
    print(json.dumps(_jsonable({
        "A1": payload["A1"],
        "A1_replay": payload["A1_replay"],
        "A2_summary": {k: payload["A2"][k] for k in (
            "n_matched", "n_no_exact_match", "n_class_penalty_mixed", "raw_nrm_class_partition_identical",
            "decisive_pair", "n_classes_raw_train", "max_class_size_raw", "y_without_cycle_diagnostic")},
        "A3": payload["A3"],
    }), indent=1))
    print(f"[stageA] seconds={payload['seconds']:.1f}")


if __name__ == "__main__":
    main()