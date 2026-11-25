"""Main analysis for lcrc-support-joint-transfer-v1.

Reads the formal run artifacts and writes the tables/curves/decomposition used by
REPORT.md.  No training and no checkpoint selection happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROUND = Path(__file__).resolve().parents[1]
CODE = ROUND / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import jt_common as C  # noqa: E402
import jt_models as M  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()
M.prepare_imports()

import cr_data as D  # noqa: E402
import cr_train as T  # noqa: E402
import ds_common as DC  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
SIG = {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        return
    fields = list(fields or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Lipo
# ---------------------------------------------------------------------------


def _lipo_arm(arm: str) -> dict[str, Any]:
    if arm == "C1":
        payload = np.load(C.LIPO_A_DIR / "best_predictions.npz")
        traj = read_json(C.LIPO_A_TRAJECTORY)
        curve = traj["curves"]
        last20 = traj["last20"]
        run = None
    else:
        payload = np.load(RUNS / arm / "best_predictions.npz")
        run = read_json(RUNS / arm / "trajectory.json")
        curve = run["curves"]
        last20 = run["last20"]
    out = {"arm": arm, "payload": payload, "curve": curve, "last20": last20, "run": run}
    for split in C.SPLITS:
        out[f"{split}_pred"] = np.asarray(payload[f"{split}_pred_raw"], dtype=np.float64)
        out[f"{split}_y"] = np.asarray(payload[f"{split}_y_raw"], dtype=np.float64)
        out[f"{split}_row"] = np.asarray(payload[f"{split}_row_id"], dtype=np.int64)
    return out


def _align(a: Mapping[str, Any], b: Mapping[str, Any], split: str) -> tuple[np.ndarray, ...]:
    ra, rb = a[f"{split}_row"], b[f"{split}_row"]
    ia, ib = np.argsort(ra), np.argsort(rb)
    if not np.array_equal(ra[ia], rb[ib]):
        raise RuntimeError(f"row ids differ for {split}")
    return (
        a[f"{split}_pred"][ia],
        a[f"{split}_y"][ia],
        b[f"{split}_pred"][ib],
        b[f"{split}_y"][ib],
    )


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def _bootstrap_rmse(e0: np.ndarray, e1: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(e0.size)
    d = np.empty(n_boot, dtype=np.float64)
    rel = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        r0 = _rmse(e0[idx])
        r1 = _rmse(e1[idx])
        d[i] = r1 - r0
        rel[i] = d[i] / r0 if r0 else math.nan
    base = _rmse(e0)
    point = _rmse(e1) - base
    return {
        "n": n,
        "n_boot": int(n_boot),
        "rmse_a": base,
        "rmse_b": _rmse(e1),
        "delta": float(point),
        "relative_delta": float(point / base),
        "ci95_delta": [float(x) for x in np.quantile(d, [0.025, 0.975])],
        "ci95_relative": [float(x) for x in np.quantile(rel, [0.025, 0.975])],
        "p_delta_lt_zero": float(np.mean(d < 0.0)),
        "bootstrap_delta_mean": float(np.mean(d)),
        "bootstrap_delta_sd": float(np.std(d, ddof=1)),
    }


def _lipo_support_features(bundle: Any, support: Any, split: str) -> dict[str, np.ndarray]:
    c = bundle.caches[split]
    n = int(c["n_molecules"])
    idx = np.asarray(support.index_table, dtype=np.int64)
    cnt = np.asarray(support.omega_count, dtype=np.float64)
    n_nbr = int(support.n_nbr_codes)
    rows = []
    for m in range(n):
        a0, a1 = int(c["atom_ptr"][m]), int(c["atom_ptr"][m + 1])
        p0, p1 = int(c["path_ptr"][a0]), int(c["path_ptr"][a1])
        codes = np.asarray(c["path_cat"][p0:p1], dtype=np.int64) * n_nbr + np.asarray(
            c["path_end63"][p0:p1], dtype=np.int64
        )
        ii = idx[codes]
        seen = ii >= 0
        doc = np.zeros(codes.size, dtype=np.float64)
        doc[seen] = cnt[ii[seen]]
        rows.append(
            {
                "n_paths": int(codes.size),
                "n_unseen": int((~seen).sum()),
                "mean_docfreq": float(doc.mean()) if codes.size else 0.0,
                "min_docfreq": float(doc.min()) if codes.size else 0.0,
                "max_docfreq": float(doc.max()) if codes.size else 0.0,
            }
        )
    out = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    out["row_id"] = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
    return out


def _lipo_strata(e0: np.ndarray, e1: np.ndarray, feat: Mapping[str, np.ndarray], prefix: str) -> list[dict[str, Any]]:
    masks = {
        "all": np.ones(e0.size, dtype=bool),
        "seen_all_pairs": feat["n_unseen"] == 0,
        "any_unseen_pair": feat["n_unseen"] > 0,
        "min_docfreq_le_1": feat["min_docfreq"] <= 1,
        "min_docfreq_gt_1": feat["min_docfreq"] > 1,
        "n_paths_q1_low": feat["n_paths"] <= np.quantile(feat["n_paths"], 0.25),
        "n_paths_q4_high": feat["n_paths"] >= np.quantile(feat["n_paths"], 0.75),
    }
    out = []
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        out.append(
            {
                "comparison": prefix,
                "stratum": name,
                "n": int(mask.sum()),
                "rmse_a": _rmse(e0[mask]),
                "rmse_b": _rmse(e1[mask]),
                "delta_rmse": _rmse(e1[mask]) - _rmse(e0[mask]),
                "mae_a": float(np.mean(np.abs(e0[mask]))),
                "mae_b": float(np.mean(np.abs(e1[mask]))),
            }
        )
    return out


def analyze_lipo(n_boot: int) -> dict[str, Any]:
    bundle = D.build_bundle("ogbg-mollipo")
    support = S.load_support(ANALYSIS / "support/lipo_support.npz")
    arms = {a: _lipo_arm(a) for a in ("C1", "C1_JOINT", "C1_ADD")}
    result: dict[str, Any] = {"schema": "lcrc-support-joint-transfer-lipo-analysis-v1", "arms": {}}
    table_rows = []
    support_rows = []
    extreme_rows = []
    for arm, obj in arms.items():
        entry: dict[str, Any] = {"arm": arm, "parameters": 228674 if arm == "C1" else obj["run"]["n_parameters"]}
        for split in C.SPLITS:
            y = obj[f"{split}_y"]
            p = obj[f"{split}_pred"]
            e = p - y
            entry[split] = {
                "n": int(y.size),
                "rmse": _rmse(e),
                "mae": float(np.mean(np.abs(e))),
                "max_abs_error": float(np.max(np.abs(e))),
                "q90_abs_error": float(np.quantile(np.abs(e), 0.90)),
                "q99_abs_error": float(np.quantile(np.abs(e), 0.99)),
            }
        if arm == "C1":
            entry["best_epoch"] = int(read_json(C.LIPO_A_TRAJECTORY)["selected"]["epoch"])
            entry["final"] = read_json(C.LIPO_A_TRAJECTORY)["final"]
        else:
            entry["best_epoch"] = int(obj["run"]["selected"]["epoch"])
            entry["final"] = obj["run"]["final"]
        entry["last20"] = obj["last20"]
        result["arms"][arm] = entry
        table_rows.append(
            {
                "arm": arm,
                "parameters": entry["parameters"],
                "best_epoch": entry["best_epoch"],
                "valid_rmse_best": entry["valid"]["rmse"],
                "test_rmse_best": entry["test"]["rmse"],
                "valid_mae_best": entry["valid"]["mae"],
                "test_mae_best": entry["test"]["mae"],
                "valid_rmse_final": entry["final"]["valid_rmse_raw"],
                "test_rmse_final": entry["final"]["test_rmse_raw"],
                "valid_rmse_last20": entry["last20"]["mean_last20_valid_rmse_raw"],
                "test_rmse_last20": entry["last20"]["mean_last20_test_rmse_raw"],
            }
        )

    for split in C.SPLITS:
        feat = _lipo_support_features(bundle, support, split)
        for arm in ("C1_JOINT", "C1_ADD"):
            p0, y, p1, y1 = _align(arms["C1"], arms[arm], split)
            if not np.array_equal(y, y1):
                raise RuntimeError("targets differ")
            # support features are in cache order; align to row-id sorted
            order = np.argsort(arms["C1"][f"{split}_row"])
            feat_sorted = {k: v[order] for k, v in feat.items()}
            if not np.array_equal(feat_sorted["row_id"], arms["C1"][f"{split}_row"][order]):
                raise RuntimeError("support feature row ids differ")
            e0, e1 = p0 - y, p1 - y1
            comp = f"{arm}_minus_C1"
            res = _bootstrap_rmse(e0, e1, n_boot, C.SEED + 11)
            res.update({"split": split, "comparison": comp})
            result.setdefault("paired", {})[f"{split}:{comp}"] = res
            support_rows.extend(_lipo_strata(e0, e1, feat_sorted, comp))
            # extremes
            order_abs = np.argsort(-np.abs(e1))
            for k in (1, 5, 10):
                idx = order_abs[:k]
                extreme_rows.append(
                    {
                        "split": split,
                        "comparison": comp,
                        "k": k,
                        "new_mean_abs_error_topk": float(np.mean(np.abs(e1[idx]))),
                        "base_mean_abs_error_same_rows": float(np.mean(np.abs(e0[idx]))),
                        "new_max_abs_error": float(np.max(np.abs(e1))),
                        "base_max_abs_error": float(np.max(np.abs(e0))),
                        "top10_overlap_with_base": int(len(set(np.argsort(-np.abs(e0))[:10]) & set(np.argsort(-np.abs(e1))[:10]))),
                    }
                )
    # JOINT vs ADD
    for split in C.SPLITS:
        p_add, y_add, p_j, y_j = _align(arms["C1_ADD"], arms["C1_JOINT"], split)
        res = _bootstrap_rmse(p_add - y_add, p_j - y_j, n_boot, C.SEED + 13)
        res.update({"split": split, "comparison": "C1_JOINT_minus_C1_ADD"})
        result.setdefault("paired", {})[f"{split}:C1_JOINT_minus_C1_ADD"] = res

    # decompositions of the two new Lipo models
    dec = {}
    for arm in ("C1_JOINT", "C1_ADD"):
        model, info = M.build_lipo_model(bundle, support, arm)
        ck = __import__("torch").load(RUNS / arm / "checkpoint_best_val.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        dec[arm] = decomposition(model, support, bundle, arm=arm)
    result["decomposition"] = dec
    result["main_table"] = table_rows
    result["support_strata"] = support_rows
    result["extreme"] = extreme_rows
    result["curves_path"] = "analysis/curves_lipo.csv"
    return result


# ---------------------------------------------------------------------------
# Decomposition / local projection
# ---------------------------------------------------------------------------


def decomposition(model: Any, support: Any, family_data: Any, *, arm: str) -> dict[str, Any]:
    table = M.table_of(model)
    p = np.asarray(support.P, dtype=np.float64)
    add = p @ table
    inter = table - add
    closure = table - add - inter
    comp = np.asarray(support.comp_p, dtype=np.int64)[
        np.asarray(support.p_col, dtype=np.int64)[np.asarray(support.omega_p, dtype=np.int64)]
    ]
    comp_rows = []
    for c in sorted(set(int(x) for x in comp)):
        mask = comp == c
        comp_rows.append(
            {
                "component": c,
                "n_omega": int(mask.sum()),
                "add_rms": float(np.sqrt(np.mean(add[mask] ** 2))),
                "interaction_rms": float(np.sqrt(np.mean(inter[mask] ** 2))),
                "total_rms": float(np.sqrt(np.mean(table[mask] ** 2))),
                "interaction_fraction": float(
                    np.sqrt(np.mean(inter[mask] ** 2)) / max(1e-30, np.sqrt(np.mean(table[mask] ** 2)))
                ),
            }
        )
    local = model.trunk.local if hasattr(model, "trunk") else model.local
    cols = M.joint_input_columns(model)
    w = local.mlp.lins[0].weight.detach().to(torch.float64).cpu().numpy()
    wj = w[:, cols]
    col_norms = np.linalg.norm(wj, axis=0)
    return {
        "arm": arm,
        "n_omega": int(support.n_omega),
        "rank": int(support.rank),
        "table_rms": float(np.sqrt(np.mean(table**2))),
        "add_rms": float(np.sqrt(np.mean(add**2))),
        "interaction_rms": float(np.sqrt(np.mean(inter**2))),
        "interaction_fraction": float(np.sqrt(np.mean(inter**2)) / max(1e-30, np.sqrt(np.mean(table**2)))),
        "closure_max_abs": float(np.max(np.abs(closure))),
        "component_rows": comp_rows,
        "joint_input_columns": [int(x) for x in cols],
        "first_layer_joint_weight_fraction": float(np.linalg.norm(wj) / max(1e-30, np.linalg.norm(w))),
        "joint_column_norm_min": float(col_norms.min()),
        "joint_column_norm_mean": float(col_norms.mean()),
        "joint_column_norm_max": float(col_norms.max()),
    }


# ---------------------------------------------------------------------------
# PCBA
# ---------------------------------------------------------------------------


def _pcba_new(arm: str) -> dict[str, Any]:
    run = read_json(RUNS / arm / "run.json")
    best = read_json(RUNS / arm / "best_checkpoint_eval.json")
    return {
        "run": run,
        "best": best,
        "valid_logits": np.load(RUNS / arm / "valid_logits_best.npy"),
        "test_logits": np.load(RUNS / arm / "test_logits_best.npy"),
    }


def _pcba_original(arm: str) -> dict[str, Any]:
    path = ANALYSIS / "replay_originals" / f"{arm}.npz"
    meta = ANALYSIS / "replay_originals" / f"{arm}.json"
    arr = np.load(path)
    return {
        "npz": arr,
        "meta": read_json(meta),
        "valid_logits": np.asarray(arr["valid_logits"]),
        "test_logits": np.asarray(arr["test_logits"]),
        "valid_y": np.asarray(arr["valid_y"]),
        "test_y": np.asarray(arr["test_y"]),
    }


def _macro_ap(y: np.ndarray, logits: np.ndarray) -> float:
    probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
    return float(DC.MET.official_ap(y, probs)["value"])


def _bootstrap_macro(y: np.ndarray, cols: Sequence[np.ndarray], n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(y.shape[0])
    vals = {i: [] for i in range(len(cols))}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        for i, col in enumerate(cols):
            vals[i].append(_macro_ap(yy, col[idx]))
    out = {}
    for i, v in vals.items():
        out[i] = np.asarray(v, dtype=np.float64)
    return out


def analyze_pcba(n_boot: int) -> dict[str, Any]:
    originals = {"W2": _pcba_original("W2"), "FIXED1": _pcba_original("FIXED1")}
    new = {"W2_JOINT": _pcba_new("W2_JOINT"), "FIXED1_JOINT": _pcba_new("FIXED1_JOINT")}
    valid_y = originals["W2"]["valid_y"]
    test_y = originals["W2"]["test_y"]
    result: dict[str, Any] = {"schema": "lcrc-support-joint-transfer-pcba-analysis-v1", "models": {}, "main_table": []}

    for arm, src in (("W2_ORIGINAL", originals["W2"]), ("FIXED1_ORIGINAL", originals["FIXED1"])):
        meta = src["meta"]
        result["models"][arm] = {
            "parameters": 994324 if arm.startswith("W2") else 1128468,
            "best_epoch": int(meta["epoch"]),
            "valid_ap": float(meta["valid_ap"]),
            "test_ap": float(meta["test_ap"]),
            "valid_bce": float(meta.get("valid_bce", math.nan)),
            "test_bce": float(meta.get("test_bce", math.nan)),
            "valid_per_task": [None if x is None else float(x) for x in meta["valid_per_task_ap"]],
            "test_per_task": [None if x is None else float(x) for x in meta["test_per_task_ap"]],
            "valid_positive": [int(x) for x in meta["valid_positive"]],
            "test_positive": [int(x) for x in meta["test_positive"]],
            "valid_observed": [int(x) for x in meta["valid_observed"]],
            "test_observed": [int(x) for x in meta["test_observed"]],
        }
    for arm, src in new.items():
        run, best = src["run"], src["best"]
        result["models"][arm] = {
            "parameters": int(run["parameters"]),
            "best_epoch": int(run["best_epoch"]),
            "valid_ap": float(run["best_checkpoint_eval"]["valid_ap"]),
            "test_ap": float(run["best_checkpoint_eval"]["test_ap"]),
            "valid_bce": float(run["best_checkpoint_eval"]["valid_bce"]),
            "test_bce": float(run["best_checkpoint_eval"]["test_bce"]),
            "valid_per_task": [None if x is None else float(x) for x in best["valid"]["per_task_ap"]],
            "test_per_task": [None if x is None else float(x) for x in best["test"]["per_task_ap"]],
            "valid_positive": [int(x) for x in best["valid"]["positive"]],
            "test_positive": [int(x) for x in best["test"]["positive"]],
            "valid_observed": [int(x) for x in best["valid"]["observed"]],
            "test_observed": [int(x) for x in best["test"]["observed"]],
            "final": run["final"],
            "last20": run["last20"],
        }
    # delete non-serialisable references before writing
    for arm in result["models"]:
        pass

    # main table
    for arm in ("W2_ORIGINAL", "W2_JOINT", "FIXED1_ORIGINAL", "FIXED1_JOINT"):
        m = result["models"][arm]
        result["main_table"].append(
            {
                "arm": arm,
                "parameters": m["parameters"],
                "best_epoch": m["best_epoch"],
                "valid_ap": m["valid_ap"],
                "test_ap": m["test_ap"],
                "valid_bce": m["valid_bce"],
                "test_bce": m["test_bce"],
            }
        )
    # paired per-task and macro
    configs = {
        "W2_JOINT_minus_W2_ORIGINAL": ("W2_JOINT", "W2_ORIGINAL"),
        "FIXED1_JOINT_minus_FIXED1_ORIGINAL": ("FIXED1_JOINT", "FIXED1_ORIGINAL"),
    }
    for split, y, key in (("valid", valid_y, "valid"), ("test", test_y, "test")):
        for name, (a, b) in configs.items():
            ma, mb = result["models"][a], result["models"][b]
            pa = np.asarray([x if x is not None else np.nan for x in ma[f"{key}_per_task"]], dtype=np.float64)
            pb = np.asarray([x if x is not None else np.nan for x in mb[f"{key}_per_task"]], dtype=np.float64)
            pos = np.asarray(ma[f"{key}_positive"], dtype=np.int64)
            common = np.isfinite(pa) & np.isfinite(pb)
            delta = pa[common] - pb[common]
            high = common & (pos >= 20)
            result.setdefault("per_task", {})[f"{split}:{name}"] = {
                "n_common_tasks": int(common.sum()),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "improved": int((delta > 0).sum()),
                "degraded": int((delta < 0).sum()),
                "unchanged": int((delta == 0).sum()),
                "high_support_n": int(high.sum()),
                "high_support_mean_delta": float(np.mean(pa[high] - pb[high])) if high.any() else None,
                "high_support_improved": int((pa[high] - pb[high] > 0).sum()) if high.any() else 0,
                "high_support_degraded": int((pa[high] - pb[high] < 0).sum()) if high.any() else 0,
                "top3_abs_contribution_fraction": float(
                    np.sort(np.abs(delta))[-3:].sum() / max(1e-30, np.abs(delta).sum())
                ),
            }
            # task bootstrap
            rng = np.random.default_rng(C.SEED + 17)
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, delta.size, delta.size)
                boots.append(float(np.mean(delta[idx])))
            result["per_task"][f"{split}:{name}"]["ci95_mean_delta"] = [
                float(x) for x in np.quantile(boots, [0.025, 0.975])
            ]
        # macro bootstrap with all four models on the same molecule resample
        cols = [
            originals["W2"][f"{split}_logits"],
            new["W2_JOINT"][f"{split}_logits"],
            originals["FIXED1"][f"{split}_logits"],
            new["FIXED1_JOINT"][f"{split}_logits"],
        ]
        boots = _bootstrap_macro(y, cols, n_boot, C.SEED + 19)
        w2d = boots[1] - boots[0]
        f1d = boots[3] - boots[2]
        inter = f1d - w2d
        result.setdefault("macro_bootstrap", {})[split] = {
            "n_boot": int(n_boot),
            "W2": {"delta": float(result["models"]["W2_JOINT"]["valid_ap"] if split == "valid" else result["models"]["W2_JOINT"]["test_ap"]) - float(result["models"]["W2_ORIGINAL"]["valid_ap"] if split == "valid" else result["models"]["W2_ORIGINAL"]["test_ap"]), "ci95": [float(x) for x in np.quantile(w2d, [0.025, 0.975])], "p_lt0": float(np.mean(w2d < 0))},
            "FIXED1": {"delta": float(result["models"]["FIXED1_JOINT"]["valid_ap"] if split == "valid" else result["models"]["FIXED1_JOINT"]["test_ap"]) - float(result["models"]["FIXED1_ORIGINAL"]["valid_ap"] if split == "valid" else result["models"]["FIXED1_ORIGINAL"]["test_ap"]), "ci95": [float(x) for x in np.quantile(f1d, [0.025, 0.975])], "p_lt0": float(np.mean(f1d < 0))},
            "interaction": {"delta": float(np.mean(inter)), "ci95": [float(x) for x in np.quantile(inter, [0.025, 0.975])], "p_lt0": float(np.mean(inter < 0))},
        }
    # interactions point estimates
    for split in ("valid", "test"):
        key = "valid_ap" if split == "valid" else "test_ap"
        m = result["models"]
        w2d = m["W2_JOINT"][key] - m["W2_ORIGINAL"][key]
        f1d = m["FIXED1_JOINT"][key] - m["FIXED1_ORIGINAL"][key]
        result.setdefault("interaction", {})[split] = {"delta_W2": float(w2d), "delta_FIXED1": float(f1d), "interaction": float(f1d - w2d)}
        result["interaction"][split]["gap_to_0.28"] = {
            "W2_JOINT": float(m["W2_JOINT"][key] - 0.28),
            "FIXED1_JOINT": float(m["FIXED1_JOINT"][key] - 0.28),
        }
    # curves
    curve_rows = []
    for arm, path in (
        ("W2_ORIGINAL", C.W2_RUN.parent / "train_curve.csv"),
        ("FIXED1_ORIGINAL", C.FIXED1_RECORD.parent / "train_curve.csv"),
    ):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r.get("epoch", 0)) > 25:
                    continue
                curve_rows.append({"arm": arm, "epoch": int(r["epoch"]), "train_ap": "", "valid_ap": r.get("valid_ap", ""), "test_ap": ""})
    for arm in ("W2_JOINT", "FIXED1_JOINT"):
        with (RUNS / arm / "train_curve.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                curve_rows.append({"arm": arm, "epoch": int(r["epoch"]), "train_ap": r["train_ap"], "valid_ap": r["valid_ap"], "test_ap": r["test_ap"]})
    result["curves"] = curve_rows
    # decomposition on both JOINT backbones
    task = None
    dec = {}
    for arm in ("W2_JOINT", "FIXED1_JOINT"):
        model, info = M.build_pcba_model_cached(arm) if hasattr(M, "build_pcba_model_cached") else (None, None)
        if model is None:
            if task is None:
                import lr_data as LD

                task = LD.load_task()
            model, info = M.build_pcba_model(task, S.load_support(ANALYSIS / "support/pcba_support.npz"), arm)
        import torch

        ck = torch.load(RUNS / arm / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        dec[arm] = decomposition(model, S.load_support(ANALYSIS / "support/pcba_support.npz"), task, arm=arm)
    result["decomposition"] = dec
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lipo", action="store_true")
    parser.add_argument("--pcba", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    do_lipo = args.lipo or not (args.lipo or args.pcba)
    do_pcba = args.pcba or not (args.lipo or args.pcba)
    if do_lipo:
        result = analyze_lipo(args.n_boot)
        write_json(ANALYSIS / "lipo_analysis.json", result)
        write_csv(ANALYSIS / "main_table_lipo.csv", result["main_table"])
        rows = []
        for split in C.SPLITS:
            for arm in ("C1_JOINT", "C1_ADD"):
                obj = result["paired"][f"{split}:{arm}_minus_C1"]
                rows.append({"split": split, "comparison": f"{arm}_minus_C1", **{k: v for k, v in obj.items() if k not in ("ci95_delta", "ci95_relative")}, "ci95_delta_low": obj["ci95_delta"][0], "ci95_delta_high": obj["ci95_delta"][1], "ci95_relative_low": obj["ci95_relative"][0], "ci95_relative_high": obj["ci95_relative"][1]})
            obj = result["paired"][f"{split}:C1_JOINT_minus_C1_ADD"]
            rows.append({"split": split, "comparison": "C1_JOINT_minus_C1_ADD", **{k: v for k, v in obj.items() if k not in ("ci95_delta", "ci95_relative")}, "ci95_delta_low": obj["ci95_delta"][0], "ci95_delta_high": obj["ci95_delta"][1], "ci95_relative_low": obj["ci95_relative"][0], "ci95_relative_high": obj["ci95_relative"][1]})
        write_csv(ANALYSIS / "lipo_paired.csv", rows)
        write_csv(ANALYSIS / "lipo_support_strata.csv", result["support_strata"])
        write_csv(ANALYSIS / "lipo_extremes.csv", result["extreme"])
        # curves
        curve_rows = []
        for arm, path in (
            ("C1", C.LIPO_A_TRAJECTORY),
            ("C1_JOINT", RUNS / "C1_JOINT" / "trajectory.json"),
            ("C1_ADD", RUNS / "C1_ADD" / "trajectory.json"),
        ):
            for r in read_json(path)["curves"]:
                curve_rows.append({"arm": arm, "epoch": int(r["epoch"]), "train_rmse": r.get("train_rmse_raw", ""), "valid_rmse": r.get("valid_rmse_raw", ""), "test_rmse": r.get("test_rmse_raw", "")})
        write_csv(ANALYSIS / "curves_lipo.csv", curve_rows)
        print("lipo done", result["paired"]["test:C1_ADD_minus_C1"]["delta"], result["paired"]["test:C1_JOINT_minus_C1"]["delta"])
    if do_pcba:
        result = analyze_pcba(args.n_boot)
        write_json(ANALYSIS / "pcba_analysis.json", result)
        write_csv(ANALYSIS / "main_table_pcba.csv", result["main_table"])
        write_csv(ANALYSIS / "curves_pcba.csv", result["curves"])
        print("pcba done", result["interaction"])


if __name__ == "__main__":
    main()
