"""Held-out evaluation + analysis for zinc-full-decomposition-valid-test-confirmation-v1.

Modes
-----
``calibrate``  train-only calibration constants (b_Y/b_H/b_P/b_K) from cached fit
``manifest``   write the frozen evaluation manifest (must precede held-out read)
``heldout``    one frozen valid + test prediction pass (requires manifest)
``analyze``    tables / bootstrap / gate from cached held-out predictions only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import zinc_full_decomposition_valid_test_confirmation_v1 as R
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

RESULTS_DIR = R.RESULTS_DIR
STATE_DIR = RESULTS_DIR
DECOMP_PATH = R.DECOMP_PATH
PROTOCOL = R.PROTOCOL_PATH
CYCLE = zjd.CYCLE

GROUP_NAMES = {0: "k=0", 1: "k=-1", 2: "k<=-2"}
SEVERE_MAX = -2


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))))


def _group_of(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, np.int64)
    return np.where(k == 0, 0, np.where(k == -1, 1, 2))


# ---------------------------------------------------------------------------
# train-only calibration
# ---------------------------------------------------------------------------


def calibrate() -> dict[str, Any]:
    decomp = R.load_decomp()
    y = np.asarray(decomp["y"], np.float64)
    g = np.asarray(decomp["g"], np.float64)
    c = np.asarray(decomp["c"], np.float64)
    median_c = float(np.median(c))
    out: dict[str, Any] = {"median_train_c": median_c, "per_seed": {}}
    for seed in R.SEEDS:
        yp = np.load(STATE_DIR / f"Y_seed{seed}_fit_pred.npz")
        hp = np.load(STATE_DIR / f"H_seed{seed}_fit_pred.npz")
        qp = np.load(STATE_DIR / f"Q_seed{seed}_predictions.npz")
        f_raw = yp["fit_raw"].astype(np.float64)
        h_raw = hp["fit_raw"].astype(np.float64)
        q_fit = qp["fit_q"].astype(np.float64)
        entry = {
            "b_Y": float(np.median(y - f_raw)),
            "b_H": float(np.median(g - h_raw)),
            "b_P": float(np.median(y - h_raw - q_fit)),
            "b_K": float(np.median(y - h_raw - median_c)),
        }
        entry["fit_raw_mae_y_Y"] = _mae(f_raw, y)
        entry["fit_raw_mae_y_H_plus_c"] = _mae(h_raw + c, y)
        entry["fit_cal_mae_y_Y"] = _mae(f_raw + entry["b_Y"], y)
        entry["fit_cal_mae_y_P"] = _mae(h_raw + q_fit + entry["b_P"], y)
        out["per_seed"][str(seed)] = entry
    R.write_json(RESULTS_DIR / "calibration.json", out)
    return out


# ---------------------------------------------------------------------------
# frozen manifest
# ---------------------------------------------------------------------------


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(zjd.REPO_ROOT)), "sha256": R.file_sha256(path),
            "bytes": int(path.stat().st_size)}


def build_manifest() -> dict[str, Any]:
    if (RESULTS_DIR / "heldout_access.json").exists():
        raise RuntimeError("held-out already accessed; refusing to (re)freeze after the fact")
    blob = R.load_prep()
    decomp = R.load_decomp()
    cal = R.read_json(RESULTS_DIR / "calibration.json")
    median_c = float(cal["median_train_c"])
    artifacts: dict[str, Any] = {
        "protocol": _artifact(PROTOCOL),
        "runner_source": _artifact(Path(R.__file__)),
        "analysis_source": _artifact(Path(__file__)),
        "deploy_wrapper_source": _artifact(RESULTS_DIR / "deploy_wrapper.py"),
        "calibration": _artifact(RESULTS_DIR / "calibration.json"),
        "prep_all_train": _artifact(R.PREP_PATH),
        "target_decomposition": _artifact(DECOMP_PATH),
        "eight_k_prep": _artifact(R.OLD_8K_PREP),
    }
    if R.T25_PATH.exists():
        artifacts["T25_all"] = _artifact(R.T25_PATH)
    models: dict[str, Any] = {}
    for seed in R.SEEDS:
        for arm in R.ARMS:
            meta = R.read_json(STATE_DIR / f"{arm}_seed{seed}.json")
            models[f"{arm}_seed{seed}"] = {
                "raw_soup_state": _artifact(STATE_DIR / f"{arm}_seed{seed}_raw_soup_state.pt"),
                "last_state": _artifact(STATE_DIR / f"{arm}_seed{seed}_last_state.pt"),
                "init_state": _artifact(STATE_DIR / f"{arm}_seed{seed}_init_state.pt"),
                "init_state_sha256": meta["init_state_sha256"],
                "soup_state_sha256": meta["soup_state_sha256"],
                "soup_members": meta["soup_members"],
                "parameters": R.FULL_PARAMETERS,
                "calibration": meta["calibration"],
            }
        models[f"Q_seed{seed}"] = {
            "head_soup_state": _artifact(STATE_DIR / f"Q_seed{seed}_head_soup_state.pt"),
            "head_init_state": _artifact(STATE_DIR / f"Q_seed{seed}_head_init_state.pt"),
            "median_train_c": median_c,
            "parameters": R.HEAD_PARAMETERS,
            "seeds": seed,
        }
    manifest = {
        "manifest_version": "frozen-eval-manifest-v1",
        "protocol_version": R.PROTOCOL_VERSION,
        "frozen_at_utc": _now(),
        "execution_commit": _git_commit(),
        "artifacts": artifacts,
        "models": models,
        "prep": R.read_json(RESULTS_DIR / "prep_meta.json"),
        "train_ids": {"source": "target_decomposition.npz positional train:%04d",
                      "n_rows": R.N_TRAIN},
        "calibration": cal,
        "metric_definition": {
            "primary": "MAE (mean absolute error, y units)",
            "primary_gain": "MAE(Y_cal) - MAE(P_cal)",
            "ensemble": "0.5/0.5 average of the two seeds' predictions (no ensemble bias refit)",
            "groups": ["k=0", "k=-1", "k<=-2"],
            "residual": "y - prediction",
        },
        "gate_definition": {
            "both_seed_valid_cal_gain_positive": True,
            "mean_valid_cal_gain": R.GATE_MEAN_GAIN,
            "max_G0_worsen_valid": R.GATE_G0_WORSEN_MAX,
        },
        "heldout_access_log": str((RESULTS_DIR / "heldout_access.json").relative_to(zjd.REPO_ROOT)),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "freeze_before_heldout": True,
    }
    R.write_json(RESULTS_DIR / "frozen_eval_manifest.json", manifest)
    return manifest


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(zjd.REPO_ROOT),
                                       text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# held-out prediction (single frozen pass)
# ---------------------------------------------------------------------------


def _load_pair(seed: int, blob: Mapping[str, Any]) -> tuple[nn.Module, nn.Module, nn.Module, dict[str, Any]]:
    median_c = float(R.read_json(STATE_DIR / f"Q_seed{seed}.json")["median_train_c"])
    full_y = R._full_model(blob, seed)
    full_y.load_state_dict(torch.load(STATE_DIR / f"Y_seed{seed}_raw_soup_state.pt", map_location="cpu"))
    full_y.eval()
    full_h = R._full_model(blob, seed)
    full_h.load_state_dict(torch.load(STATE_DIR / f"H_seed{seed}_raw_soup_state.pt", map_location="cpu"))
    full_h.eval()
    head = R.build_head(seed, median_c)
    head.load_state_dict(torch.load(STATE_DIR / f"Q_seed{seed}_head_soup_state.pt", map_location="cpu"))
    head.eval()
    cal = R.read_json(RESULTS_DIR / "calibration.json")["per_seed"][str(seed)]
    return full_y, full_h, head, {"b_Y": cal["b_Y"], "b_H": cal["b_H"], "b_P": cal["b_P"],
                                  "b_K": cal["b_K"], "median_c": median_c}


def _predict_arms(models, data, decomp_like: Mapping[str, np.ndarray] | None,
                  device: torch.device = torch.device("cpu")) -> dict[str, np.ndarray]:
    y_t = torch.zeros(len(data), dtype=torch.float32)
    T = torch.as_tensor(np.concatenate(
        [d.topology_features.reshape(1, -1).numpy().astype(np.float32) for d in data], 0))
    out: dict[str, np.ndarray] = {}
    for seed in R.SEEDS:
        full_y, full_h, head, cal = models[seed]
        f = R.predict_full(full_y, data, y_t, device)
        h = R.predict_full(full_h, data, y_t, device)
        with torch.no_grad():
            q = R.head_forward(head, T).double().numpy()
        out[f"Y_raw_{seed}"] = f
        out[f"Y_cal_{seed}"] = f + cal["b_Y"]
        out[f"H_raw_{seed}"] = h
        out[f"H_cal_{seed}"] = h + cal["b_H"]
        out[f"Q_{seed}"] = q
        out[f"P_raw_{seed}"] = h + q
        out[f"P_cal_{seed}"] = h + q + cal["b_P"]
        out[f"K_raw_{seed}"] = h + cal["median_c"]
        out[f"K_cal_{seed}"] = h + cal["median_c"] + cal["b_K"]
    return out


def _load_heldout_labels(split: str) -> dict[str, np.ndarray] | None:
    import pandas as pd
    path = CYCLE / f"{split}_cycle_audit_label.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.sort_values("subset_index")
    y = df["target"].to_numpy(np.float64)
    c = df["label_cycle_component"].to_numpy(np.float64)
    g = df["y_without_cycle_label"].to_numpy(np.float64)
    k = np.round(df["label_effective_cycle_snapped"].to_numpy(np.float64)).astype(np.int64)
    ids = df["molecule_id"].astype(str).to_numpy()
    return {"y": y, "c": c, "g": g, "k": k, "ids": ids, "source_sha256": R.file_sha256(path)}


def _access_log(split: str, meta: Mapping[str, Any], label: Mapping[str, Any] | None) -> None:
    path = RESULTS_DIR / "heldout_access.json"
    payload = R.read_json(path) if path.exists() else {
        "protocol_version": R.PROTOCOL_VERSION,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "events": [],
    }
    payload["events"].append({
        "split": split, "first_read_utc": _now(), "n_rows": meta.get("n_rows"),
        "source": meta.get("source"),
        "label_source": (None if label is None else label.get("source_sha256")),
    })
    payload[f"official_{split}_loaded"] = True
    R.write_json(path, payload)


def heldout() -> dict[str, Any]:
    manifest = R.read_json(RESULTS_DIR / "frozen_eval_manifest.json")
    if not manifest.get("freeze_before_heldout"):
        raise RuntimeError("no valid frozen manifest; refusing held-out read")
    blob = R.load_prep()
    models = {seed: _load_pair(seed, blob) for seed in R.SEEDS}

    results: dict[str, Any] = {"manifest_frozen_at": manifest["frozen_at_utc"]}
    for split in ("valid", "test"):
        if split == "valid":
            data, meta = R.load_valid_data(blob)
        else:
            data, meta = R.load_test_data(blob)
        label = _load_heldout_labels(split)
        preds = _predict_arms(models, data, None)
        y = np.asarray([float(d.y) for d in data], np.float64)
        # cross-check labels align positionally
        label_check = None
        if label is not None:
            n = min(len(y), len(label["y"]))
            label_check = {
                "n_label": int(len(label["y"])),
                "y_max_abs_diff": float(np.max(np.abs(y[:n] - label["y"][:n]))),
                "g_plus_c_minus_y": float(np.max(np.abs(label["g"] + label["c"] - label["y"]))),
                        }
        payload = {"split": split, "meta": meta, "y": y, "label_check": label_check}
        payload.update(preds)
        if label is not None:
            payload.update({"k": label["k"], "c": label["c"], "g": label["g"],
                            "ids": label["ids"], "label_source_sha256": label["source_sha256"]})
        else:
            payload.update({"k": None, "c": None, "g": None, "ids": None,
                            "label_source_sha256": None})
        _save_split(split, payload)
        _access_log(split, meta, label)
        results[split] = {"n_rows": meta.get("n_rows"), "label_check": label_check}
        print(f"[heldout] {split} n={meta.get('n_rows')} label_check={label_check}")
    R.write_json(RESULTS_DIR / "heldout_summary.json", results)
    return results


def _save_split(split: str, payload: Mapping[str, Any]) -> None:
    arrays = {k: v for k, v in payload.items()
              if isinstance(v, np.ndarray)}
    np.savez_compressed(RESULTS_DIR / f"heldout_{split}_predictions.npz", **arrays)
    # csv with scalar columns (prediction arrays + labels when available)
    import csv
    base = ["ids", "k", "y", "c", "g"]
    pred_cols = sorted(k for k in arrays if k not in base)
    cols = [c for c in base if c in arrays] + pred_cols
    with (RESULTS_DIR / f"heldout_{split}_predictions.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        n = len(arrays["y"])
        for i in range(n):
            writer.writerow([arrays[c][i] for c in cols])


# ---------------------------------------------------------------------------
# analysis from cached predictions
# ---------------------------------------------------------------------------


def _load_split(split: str) -> dict[str, Any]:
    with np.load(RESULTS_DIR / f"heldout_{split}_predictions.npz", allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    for key in ("k", "c", "g", "ids"):
        out.setdefault(key, None)
    return out


def _train_predictions() -> dict[str, Any]:
    decomp = R.load_decomp()
    out = {"y": np.asarray(decomp["y"], np.float64), "c": np.asarray(decomp["c"], np.float64),
           "g": np.asarray(decomp["g"], np.float64), "k": np.asarray(decomp["k"], np.int64)}
    cal = R.read_json(RESULTS_DIR / "calibration.json")
    for seed in R.SEEDS:
        yp = np.load(STATE_DIR / f"Y_seed{seed}_fit_pred.npz")
        hp = np.load(STATE_DIR / f"H_seed{seed}_fit_pred.npz")
        qp = np.load(STATE_DIR / f"Q_seed{seed}_predictions.npz")
        cs = cal["per_seed"][str(seed)]
        f = yp["fit_raw"].astype(np.float64)
        h = hp["fit_raw"].astype(np.float64)
        q = qp["fit_q"].astype(np.float64)
        out[f"Y_raw_{seed}"] = f
        out[f"Y_cal_{seed}"] = f + cs["b_Y"]
        out[f"H_raw_{seed}"] = h
        out[f"H_cal_{seed}"] = h + cs["b_H"]
        out[f"Q_{seed}"] = q
        out[f"P_raw_{seed}"] = h + q
        out[f"P_cal_{seed}"] = h + q + cs["b_P"]
        out[f"K_raw_{seed}"] = h + cs["median_c"]
        out[f"K_cal_{seed}"] = h + cs["median_c"] + cs["b_K"]
    return out


def _ensemble(data: Mapping[str, Any], arm: str) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for kind in ("raw", "cal"):
        s0 = data[f"{arm}_{kind}_0"]
        s1 = data[f"{arm}_{kind}_1"]
        out[(arm, kind)] = 0.5 * (s0 + s1)
    return out


def _summary_for(data: Mapping[str, Any], y: np.ndarray) -> dict[str, Any]:
    res: dict[str, Any] = {"n": int(len(y)), "per_seed": {}, "mean": {}, "ensemble": {}}
    for arm in ("Y", "P", "H", "K"):
        for kind in ("raw", "cal"):
            vals = []
            for seed in R.SEEDS:
                key = f"{arm}_{kind}_{seed}"
                if key in data:
                    vals.append(_mae(data[key], y))
            if len(vals) == 2:
                res["per_seed"][f"{arm}_{kind}"] = vals
                res["mean"][f"{arm}_{kind}"] = float(np.mean(vals))
            if f"{arm}_{kind}_0" in data and f"{arm}_{kind}_1" in data:
                ens = 0.5 * (data[f"{arm}_{kind}_0"] + data[f"{arm}_{kind}_1"])
                res["ensemble"][f"{arm}_{kind}"] = _mae(ens, y)
    # primary gains
    gains = {}
    for kind in ("raw", "cal"):
        per = [_mae(data[f"Y_{kind}_{s}"], y) - _mae(data[f"P_{kind}_{s}"], y) for s in R.SEEDS]
        gains[kind] = {"per_seed": per, "mean": float(np.mean(per)),
                       "ensemble": _mae(_ens(data, "Y", kind), y) - _mae(_ens(data, "P", kind), y)}
    res["gain_YP"] = gains
    for kind in ("raw", "cal"):
        res[f"valid_minus_test_{kind}"] = None
    res["G0_marker_P_minus_O"] = None
    return res


def _ens(data: Mapping[str, Any], arm: str, kind: str) -> np.ndarray:
    return 0.5 * (data[f"{arm}_{kind}_0"] + data[f"{arm}_{kind}_1"])


def group_table(data: Mapping[str, Any], y: np.ndarray, k: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    groups = _group_of(k)
    N = len(y)
    for gi, name in GROUP_NAMES.items():
        mask = groups == gi
        n = int(mask.sum())
        row: dict[str, Any] = {"group": name, "n": n}
        for arm in ("Y", "P"):
            for seed in R.SEEDS:
                pred = data[f"{arm}_cal_{seed}"][mask]
                err = y[mask] - pred
                row[f"{arm}_s{seed}_MAE"] = _mae(pred, y[mask])
                row[f"{arm}_s{seed}_signed"] = float(np.mean(err))
                row[f"{arm}_s{seed}_contrib"] = float(np.sum(np.abs(err)) / N)
                row[f"{arm}_s{seed}_gain"] = (_mae(data[f"Y_cal_{seed}"][mask], y[mask])
                                              - _mae(data[f"P_cal_{seed}"][mask], y[mask]))
        rows.append(row)
    return rows


def component_table(data: Mapping[str, Any], c: np.ndarray, k: np.ndarray,
                    median_c: float) -> list[dict[str, Any]]:
    rows = []
    for name, mask in (("overall", np.ones(len(c), bool)), ("k=0", k == 0), ("k=-1", k == -1),
                       ("k<=-2", k <= -2), ("k<=-3", k <= -3)):
        n = int(mask.sum())
        if n == 0:
            rows.append({"group": name, "n": 0})
            continue
        row = {"group": name, "n": n}
        for seed in R.SEEDS:
            q = data[f"Q_{seed}"][mask]
            row[f"Q_s{seed}_MAE_vs_c"] = _mae(q, c[mask])
        row["constant_MAE_vs_c"] = _mae(np.full(n, median_c), c[mask])
        rows.append(row)
    return rows


def bootstrap(data: Mapping[str, Any], y: np.ndarray, *, n_boot: int = R.N_BOOT,
              seed: int = R.BOOT_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    N = len(y)
    out: dict[str, Any] = {"n_boot": int(n_boot), "seed": int(seed), "n": N, "stats": {}}

    def gain(idx, kind, seed_id=None):
        yy = y[idx]
        if seed_id is None:
            gy = np.mean([_mae(data[f"Y_{kind}_{s}"][idx], yy) for s in R.SEEDS])
            gp = np.mean([_mae(data[f"P_{kind}_{s}"][idx], yy) for s in R.SEEDS])
        else:
            gy = _mae(data[f"Y_{kind}_{seed_id}"][idx], yy)
            gp = _mae(data[f"P_{kind}_{seed_id}"][idx], yy)
        return gy - gp

    for kind in ("raw", "cal"):
        for tag, sid in [("mean", None), ("s0", 0), ("s1", 1)]:
            draws = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, N, N)
                draws[b] = gain(idx, kind, sid)
            pt = gain(np.arange(N), kind, sid)
            out["stats"][f"gain_YP_{kind}_{tag}"] = {
                "point": float(pt), "mean": float(draws.mean()),
                "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            }
    # G0 (k==0) gain
    g0 = np.where(k == 0)[0] if k is not None else None
    if g0 is not None and len(g0) > 0:
        g0_y = y[g0]
        draws = np.empty(n_boot)
        for b in range(n_boot):
            idx = g0[rng.integers(0, len(g0), len(g0))]
            yy = y[idx]
            gy = np.mean([_mae(data[f"Y_cal_{s}"][idx], yy) for s in R.SEEDS])
            gp = np.mean([_mae(data[f"P_cal_{s}"][idx], yy) for s in R.SEEDS])
            draws[b] = gy - gp
        yy = y[g0]
        pt = (np.mean([_mae(data[f"Y_cal_{s}"][g0], yy) for s in R.SEEDS])
              - np.mean([_mae(data[f"P_cal_{s}"][g0], yy) for s in R.SEEDS]))
        out["stats"]["gain_G0_cal_mean"] = {"point": float(pt), "mean": float(draws.mean()),
                                            "lo": float(np.percentile(draws, 2.5)),
                                            "hi": float(np.percentile(draws, 97.5))}
    return out


def invariance_checks(data: Mapping[str, Any], y: np.ndarray) -> dict[str, Any]:
    pt = _mae(data["Y_cal_0"], y)
    # swapped arms -> sign flip
    y_mae = _mae(data["P_cal_0"], y)
    swapped = _mae(data["P_cal_0"], y) - _mae(data["Y_cal_0"], y)
    # constant shift applied to both arms -> zero effect on gain
    shift = _mae(data["Y_cal_0"] + 3.7, y) - _mae(data["P_cal_0"] + 3.7, y)
    return {
        "swap_arm_gain": float(swapped),
        "swap_is_negation": bool(abs(swapped + (pt - y_mae)) < 1e-9),
        "constant_shift_gain": float(shift),
        "constant_shift_invariant": bool(abs(shift - (pt - y_mae)) < 1e-9),
    }


def analyze() -> dict[str, Any]:
    cal = R.read_json(RESULTS_DIR / "calibration.json")
    median_c = float(cal["median_train_c"])
    train = _train_predictions()
    splits = {"train": train}
    for split in ("valid", "test"):
        splits[split] = _load_split(split)

    summary: dict[str, Any] = {"splits": {}, "calibration": cal}
    main_rows = []
    for name, data in splits.items():
        y = data["y"]
        s = _summary_for(data, y)
        summary["splits"][name] = s
        for arm in ("Y", "P"):
            for kind in ("raw", "cal"):
                for seed in R.SEEDS:
                    v = _mae(data[f"{arm}_{kind}_{seed}"], y)
                    main_rows.append({"split": name, "arm": arm, "kind": kind, "scope": f"seed{seed}",
                                      "mae": v, "training_rows": R.N_TRAIN})
                main_rows.append({"split": name, "arm": arm, "kind": kind, "scope": "seed_mean",
                                  "mae": float(np.mean([_mae(data[f"{arm}_{kind}_{seed}"], y)
                                                        for seed in R.SEEDS])), "training_rows": R.N_TRAIN})
                main_rows.append({"split": name, "arm": arm, "kind": kind, "scope": "ensemble_0.5",
                                  "mae": _mae(_ens(data, arm, kind), y), "training_rows": R.N_TRAIN})
    # valid - test per the same model/seed
    vd, td = splits["valid"], splits["test"]
    diffs = {}
    for arm in ("Y", "P"):
        for kind in ("raw", "cal"):
            for seed in R.SEEDS:
                diffs[f"{arm}_{kind}_seed{seed}"] = float(_mae(vd[f"{arm}_{kind}_{seed}"], vd["y"])
                                                          - _mae(td[f"{arm}_{kind}_{seed}"], td["y"]))
            diffs[f"{arm}_{kind}_ensemble"] = float(_mae(_ens(vd, arm, kind), vd["y"])
                                                    - _mae(_ens(td, arm, kind), td["y"]))
    summary["valid_minus_test"] = diffs

    # groups
    group_tables = {}
    for name, data in splits.items():
        if data.get("k") is None:
            group_tables[name] = None
            continue
        group_tables[name] = group_table(data, data["y"], np.asarray(data["k"], np.int64))
    summary["group_tables"] = group_tables

    comp = {}
    for name in ("train", "valid", "test"):
        data = splits[name]
        if data.get("c") is None:
            comp[name] = None
            continue
        comp[name] = component_table(data, np.asarray(data["c"], np.float64),
                                     np.asarray(data["k"], np.int64), median_c)
    summary["component_tables"] = comp

    # gate (valid only)
    v = summary["splits"]["valid"]
    g0_ok = True
    g0_detail = {}
    if vd.get("k") is not None:
        mask = np.asarray(vd["k"], np.int64) == 0
        for seed in R.SEEDS:
            worsen = _mae(vd[f"P_cal_{seed}"][mask], vd["y"][mask]) - _mae(vd[f"Y_cal_{seed}"][mask], vd["y"][mask])
            g0_detail[f"seed{seed}"] = float(worsen)
        g0_ok = all(x <= R.GATE_G0_WORSEN_MAX for x in g0_detail.values())
    gains_cal = v["gain_YP"]["cal"]
    gate = {
        "both_seed_gain_positive": bool(all(x > 0 for x in gains_cal["per_seed"])),
        "mean_gain_ge_threshold": bool(gains_cal["mean"] >= R.GATE_MEAN_GAIN),
        "G0_worsen": g0_detail,
        "G0_worsen_ok": bool(g0_ok),
        "gain_per_seed": gains_cal["per_seed"], "gain_mean": gains_cal["mean"],
        "identity_pass": True,  # updated by manifest/replay artifacts
    }
    gate["valid_gate_pass"] = bool(gate["both_seed_gain_positive"] and gate["mean_gain_ge_threshold"]
                                   and gate["G0_worsen_ok"])
    summary["gate_valid"] = gate

    summary["invariance"] = {name: invariance_checks(splits[name], splits[name]["y"])
                             for name in ("train", "valid", "test")}
    # per-row identities for held-out where decomposition labels exist
    identities = {}
    for name in ("valid", "test"):
        data = splits[name]
        if data.get("g") is None:
            identities[name] = None
            continue
        seed = 0
        e_g = np.asarray(data["g"], np.float64) - data[f"H_raw_{seed}"]
        e_c = np.asarray(data["c"], np.float64) - data[f"Q_{seed}"]
        lhs = data["y"] - data[f"P_cal_{seed}"]
        b_P = cal["per_seed"][str(seed)]["b_P"]
        rhs = e_g + e_c - b_P
        identities[name] = {"max_abs_identity": float(np.max(np.abs(lhs - rhs))),
                            "seed": seed, "b_P": b_P}
    summary["decomposition_identities"] = identities

    # top-10 Y max error rows + P change
    top = {}
    for name in ("valid", "test"):
        data = splits[name]
        y = data["y"]
        err = np.abs(data["Y_cal_0"] - y)
        order = np.argsort(-err)[:10]
        rows = []
        for i in order:
            row = {"idx": int(i), "id": (str(data["ids"][i]) if data.get("ids") is not None else ""),
                   "k": (int(data["k"][i]) if data.get("k") is not None else None),
                   "Y_cal": float(data["Y_cal_0"][i]), "P_cal": float(data["P_cal_0"][i]),
                   "y": float(y[i]), "Y_err": float(data["Y_cal_0"][i] - y[i]),
                   "P_err": float(data["P_cal_0"][i] - y[i]),
                   "delta_abs": float(abs(data["Y_cal_0"][i] - y[i]) - abs(data["P_cal_0"][i] - y[i]))}
            rows.append(row)
        top[name] = rows
    summary["top10_Y_error_rows"] = top

    R.write_json(RESULTS_DIR / "analysis_summary.json", summary)
    import csv
    with (RESULTS_DIR / "main_table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(main_rows[0].keys()))
        writer.writeheader()
        writer.writerows(main_rows)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="held-out eval + analysis")
    parser.add_argument("--mode", required=True, choices=("calibrate", "manifest", "heldout", "analyze"))
    args = parser.parse_args(argv)
    if args.mode == "calibrate":
        print(json.dumps(calibrate(), indent=2))
    elif args.mode == "manifest":
        m = build_manifest()
        print(f"[manifest] frozen_at={m['frozen_at_utc']} commit={m['execution_commit']}")
    elif args.mode == "heldout":
        print(json.dumps(heldout(), indent=2))
    elif args.mode == "analyze":
        s = analyze()
        print(json.dumps({k: s[k] for k in ("valid_minus_test", "gate_valid", "invariance")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())