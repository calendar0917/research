#!/usr/bin/env python3
"""Post-hoc CPU error analysis of frozen C/G context modes.

Round: ``zinc-bond-conditioned-context-attention-v1``
Post-hoc id: ``posthoc-frozen-mode-error-v1``

Question
--------
Under the *same already-trained weights*, does restoring the uniform
(P3-Null-equivalent) context mapping reduce raw-y MAE?  This is a post-hoc
intervention diagnostic, not a new model and not an independent confirmation.

What this script does
---------------------
* Reads only frozen arrays/records: the mechanism predictions, the official
  frozen valid predictions (P3/C/G/F) and the cached valid labels.
* Verifies every file/array SHA256 against the frozen records, checks the
  ``row`` arrays, finiteness and shape, and re-derives the original valid
  metrics to prove that the metric primitives below match the frozen ones.
* Scores the six fixed endpoints A--F and the six fixed comparisons.  No
  endpoint search, threshold search, lambda/checkpoint/seed search.
* Writes ``metrics.json``, ``input_manifest.json`` and
  ``valid_label_access.json`` only inside this post-hoc directory.

What this script deliberately does **not** do
---------------------------------------------
No model construction, no forward pass, no training, no GPU, no official-test
access, no call to the original ``cmd_score``, no modification of
``valid_read.lock`` or any frozen artefact.

The five metric primitives below are verbatim copies of
``tracks/structural-sufficiency/code/phase2b/phase2b_common.py`` (the frozen
implementation used by the original score).  The script checks them by
reproducing all six original contrasts and every original per-seed/seed-mean
MAE from ``valid_metrics.json``.  Keeping the copies local also means the
analysis never imports torch or any model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = HERE.parents[4]
if not (REPO / "tracks/structural-sufficiency").is_dir():
    raise SystemExit(f"cannot locate repository root from {HERE}")

PROTOCOL_ID = "zinc-bond-conditioned-context-attention-v1"
POSTHOC_ID = "posthoc-frozen-mode-error-v1"
RESULT_DIR_REL = "tracks/structural-sufficiency/results/zinc_bond_conditioned_context_attention_v1"

SEEDS = (0, 1)
N_VALID = 1000
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261019
TAIL_THRESHOLD = -3.0
EXTREME_VALID_ROW = 172

MECH_PATH = ROUND / "mechanism_this_round.json"
METRICS_PATH = ROUND / "valid_metrics.json"
DECISION_PATH = ROUND / "valid_decision.json"
PRED_FREEZE_PATH = ROUND / "valid_prediction_freeze.json"
READ_LOCK_PATH = ROUND / "valid_read.lock"
P3_AUDIT_PATH = ROUND / "p3_reuse_audit.json"
CONFIG_PATH = REPO / "tracks/structural-sufficiency/configs/zinc_bond_conditioned_context_attention_v1.yaml"
Y_PATH = REPO / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/valid_y_read_once.npy"

# Frozen arrays are not committed in the result directory (round convention).
# These are the verified `rr pull` copies of the frozen compute-host outputs.
PULLED_SCORE = REPO / (
    ".rr/pulled/bcca-v1-score-20261019-194048-7e8c60e9/result/"
    + RESULT_DIR_REL
)
PULLED_MECH = REPO / (
    ".rr/pulled/bcca-v1-mech-20261019-193749-d580d4cc/result/"
    + RESULT_DIR_REL
)

ENDPOINT_DESCRIPTIONS = {
    "A": {
        "short": "P3*Null",
        "description": "P3-frozen weights + P3 Null context (reference)",
        "weights": "P3",
        "mode": "Null",
        "source": "historical frozen P3 valid prediction",
    },
    "B": {
        "short": "Cw*C",
        "description": "C-trained weights + C mode (original formal C)",
        "weights": "C",
        "mode": "C",
        "source": "official frozen valid prediction (mechanism C-mode cross-check)",
    },
    "C": {
        "short": "Cw*G",
        "description": "C-trained weights + G mode (same-weight marginalisation)",
        "weights": "C",
        "mode": "G",
        "source": "frozen mechanism prediction",
    },
    "D": {
        "short": "Cw*U",
        "description": "C-trained weights + uniform mode (same-weight Null mapping)",
        "weights": "C",
        "mode": "uniform",
        "source": "frozen mechanism prediction",
    },
    "E": {
        "short": "Gw*G",
        "description": "G-trained weights + G mode (original formal G)",
        "weights": "G",
        "mode": "G",
        "source": "official frozen valid prediction (mechanism G-mode cross-check)",
    },
    "F": {
        "short": "Gw*U",
        "description": "G-trained weights + uniform mode (same-weight Null mapping)",
        "weights": "G",
        "mode": "uniform",
        "source": "frozen mechanism prediction",
    },
}

COMPARISONS = (
    ("D", "B"),
    ("D", "A"),
    ("F", "E"),
    ("F", "A"),
    ("C", "B"),
    ("C", "A"),
)

ENDPOINT_ARRAY_KEYS = {
    "A": {0: "P3_s0", 1: "P3_s1"},
    "B": {0: "official_C_s0", 1: "official_C_s1"},
    "C": {0: "mech_C_s0_G", 1: "mech_C_s1_G"},
    "D": {0: "mech_C_s0_uniform", 1: "mech_C_s1_uniform"},
    "E": {0: "official_G_s0", 1: "official_G_s1"},
    "F": {0: "mech_G_s0_uniform", 1: "mech_G_s1_uniform"},
}


# ---------------------------------------------------------------------------
# hashing / io
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray, dtype: str = "float32") -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_json(path: Path | str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.resolve().parent != HERE.resolve():
        raise SystemExit(f"refusing to write outside the post-hoc directory: {path}")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# frozen metric primitives (verbatim from phase2b_common.py)
# ---------------------------------------------------------------------------


def paired_bootstrap(
    delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n), d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "better_fraction": float((d > 0).mean()),
        "n_resamples": int(n),
        "bootstrap_seed": int(seed),
        "per_molecule_unit": True,
    }


def seed_averaged_delta(
    pred_cand: Mapping[int, np.ndarray],
    pred_ref: Mapping[int, np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    """d_i = mean_s (|e_ref,s,i| - |e_cand,s,i|); mean(d) = seed-mean MAE gain."""
    y = np.asarray(y, dtype=np.float64)
    parts = [
        np.abs(np.asarray(pred_ref[s], dtype=np.float64) - y)
        - np.abs(np.asarray(pred_cand[s], dtype=np.float64) - y)
        for s in SEEDS
    ]
    return np.mean(np.stack(parts, 0), axis=0)


def error_rank(pred: Mapping[int, np.ndarray], y: np.ndarray) -> np.ndarray:
    """Mean absolute error over seeds, descending order of magnitude."""
    y = np.asarray(y, dtype=np.float64)
    e = np.mean(
        [np.abs(np.asarray(pred[s], dtype=np.float64) - y) for s in SEEDS], axis=0
    )
    return np.argsort(-e)


def top_removed_gain(d: np.ndarray, indices: Iterable[int]) -> float:
    drop = np.zeros(d.size, dtype=bool)
    drop[list(indices)] = True
    keep = ~drop
    return float(d[keep].mean())


# ---------------------------------------------------------------------------
# loading / verification
# ---------------------------------------------------------------------------


def load_npz_pred(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        row = np.asarray(z["row"], dtype=np.int64)
        pred = np.asarray(z["pred"], dtype=np.float64)
    return row, pred


def resolve_array(
    name: str, expected_file_sha256: str, array_root: Path | None = None
) -> Path:
    roots: list[Path] = []
    if array_root is not None:
        roots.append(Path(array_root))
    roots.extend([ROUND, PULLED_SCORE, PULLED_MECH])
    seen: list[Path] = []
    for root in roots:
        path = root / name
        if not path.exists():
            continue
        seen.append(path)
        if sha256_file(path) == expected_file_sha256:
            return path
    if seen:
        raise SystemExit(
            f"frozen array hash mismatch for {name}; candidates: "
            + ", ".join(str(p) for p in seen)
        )
    raise SystemExit(
        f"missing frozen array {name}; looked in: "
        + ", ".join(str(p) for p in roots)
        + ". Do not regenerate it by retraining or re-inference."
    )


def verify_npz_array(
    name: str,
    path: Path,
    expected_file_sha256: str,
    expected_array_sha256: str,
    role: str,
) -> tuple[dict[str, Any], np.ndarray]:
    if sha256_file(path) != expected_file_sha256:
        raise SystemExit(f"file SHA256 mismatch for {name}: {path}")
    row, pred = load_npz_pred(path)
    if pred.shape != (N_VALID,):
        raise SystemExit(f"unexpected prediction shape for {name}: {pred.shape}")
    if row.shape != (N_VALID,):
        raise SystemExit(f"unexpected row shape for {name}: {row.shape}")
    if not np.array_equal(row, np.arange(N_VALID, dtype=np.int64)):
        raise SystemExit(f"row order mismatch for {name}")
    if not np.isfinite(pred).all():
        raise SystemExit(f"non-finite prediction in {name}")
    got_array = sha256_array(pred)
    if got_array != expected_array_sha256:
        raise SystemExit(
            f"array SHA256 mismatch for {name}: {got_array} != {expected_array_sha256}"
        )
    record = {
        "role": role,
        "path": rel(path),
        "file_sha256": expected_file_sha256,
        "array_sha256_float32": got_array,
        "expected_array_sha256_float32": expected_array_sha256,
        "shape": [int(pred.shape[0])],
        "row_equals_arange_0_999": True,
        "finite": True,
        "verified": True,
    }
    return record, pred


def mae(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.abs(pred - y).mean())
    return float(np.abs(pred[mask] - y[mask]).mean())


def per_seed_gain(
    cand: Mapping[int, np.ndarray],
    ref: Mapping[int, np.ndarray],
    y: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is None:
        return {
            str(s): float(
                np.abs(ref[s] - y).mean() - np.abs(cand[s] - y).mean()
            )
            for s in SEEDS
        }
    return {
        str(s): float(
            np.abs(ref[s][mask] - y[mask]).mean()
            - np.abs(cand[s][mask] - y[mask]).mean()
        )
        for s in SEEDS
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--array-root",
        type=Path,
        default=None,
        help="optional directory containing the frozen .npz arrays by basename",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run all verifications and print a summary without writing outputs",
    )
    args = parser.parse_args(argv)

    import yaml  # local import; PyYAML is already a repository dependency

    created_utc = utc_now()

    # -- records ---------------------------------------------------------
    mech = read_json(MECH_PATH)
    valid_metrics = read_json(METRICS_PATH)
    valid_decision = read_json(DECISION_PATH)
    pred_freeze = read_json(PRED_FREEZE_PATH)
    read_lock = read_json(READ_LOCK_PATH)
    p3_audit = read_json(P3_AUDIT_PATH)
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    if mech.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit("mechanism record protocol_id mismatch")
    if valid_metrics.get("n") != N_VALID:
        raise SystemExit("valid_metrics n mismatch")
    if int(read_lock.get("read_count", -1)) != 1 or read_lock.get("reread") is not False:
        raise SystemExit("valid_read.lock is not the frozen read_count=1 record")
    if read_lock.get("prediction_freeze_sha256") != sha256_file(PRED_FREEZE_PATH):
        raise SystemExit("prediction freeze hash in valid_read.lock does not match")
    if read_lock.get("mechanism_freeze_sha256") != sha256_file(MECH_PATH):
        raise SystemExit("mechanism freeze hash in valid_read.lock does not match")

    records = {
        "mechanism_this_round.json": MECH_PATH,
        "valid_metrics.json": METRICS_PATH,
        "valid_decision.json": DECISION_PATH,
        "valid_prediction_freeze.json": PRED_FREEZE_PATH,
        "valid_read.lock": READ_LOCK_PATH,
        "p3_reuse_audit.json": P3_AUDIT_PATH,
        "config": CONFIG_PATH,
    }
    records_manifest = {
        name: {"path": rel(path), "sha256": sha256_file(path)}
        for name, path in records.items()
    }

    # -- labels (cached historical array; no dataset rebuild) ------------
    y = np.asarray(np.load(Y_PATH, allow_pickle=False), dtype=np.float64)
    if y.shape != (N_VALID,):
        raise SystemExit(f"unexpected valid label shape {y.shape}")
    if not np.isfinite(y).all():
        raise SystemExit("non-finite valid label")
    y_f32 = sha256_array(y)
    y_f64 = sha256_array(y, "float64")
    expected_y_f32 = valid_metrics["labels"]["array_sha256_float32"]
    expected_y_f64 = valid_metrics["labels"]["array_sha256_float64"]
    if y_f32 != expected_y_f32 or y_f64 != expected_y_f64:
        raise SystemExit("valid label array hash does not match valid_metrics")
    if y_f32 != read_lock["y_sha256_float32"] or y_f64 != read_lock["y_sha256_float64"]:
        raise SystemExit("valid label array hash does not match valid_read.lock")
    if y_f32 != cfg["label_provenance"]["valid_y"]["array_sha256_float32"]:
        raise SystemExit("valid label array hash does not match frozen config")
    if sha256_file(Y_PATH) != cfg["label_provenance"]["valid_y"]["historical_file_sha256"]:
        raise SystemExit("valid label .npy file hash does not match frozen config")

    labels_manifest = {
        "path": rel(Y_PATH),
        "file_sha256": sha256_file(Y_PATH),
        "array_sha256_float32": y_f32,
        "array_sha256_float64": y_f64,
        "shape": [int(y.shape[0])],
        "finite": True,
        "matches_valid_metrics": True,
        "matches_valid_read_lock": True,
        "source_policy": "cached historical array; no dataset rebuild, no split change",
    }

    # -- arrays ----------------------------------------------------------
    arrays_manifest: dict[str, dict[str, Any]] = {}
    p3: dict[int, np.ndarray] = {}
    official: dict[tuple[str, int], np.ndarray] = {}
    mech_preds: dict[tuple[str, int, str], np.ndarray] = {}

    # P3 historical frozen predictions
    for seed in SEEDS:
        prov = cfg["historical_prediction_provenance"][f"valid_P3{seed}"]
        path = REPO / prov["historical_path"]
        if not path.exists():
            raise SystemExit(f"missing historical P3 prediction: {path}")
        record, pred = verify_npz_array(
            f"P3_s{seed}",
            path,
            prov["historical_file_sha256"],
            prov["array_sha256_float32"],
            "endpoint A: historical frozen P3 valid prediction",
        )
        expected_freeze = pred_freeze["entries"][f"P3{seed}"]["array_sha256"]
        if record["array_sha256_float32"] != expected_freeze:
            raise SystemExit(f"P3 s{seed} array hash does not match prediction freeze")
        expected_audit = p3_audit["p3"][str(seed)]["frozen_array_sha256"]
        if record["array_sha256_float32"] != expected_audit:
            raise SystemExit(f"P3 s{seed} array hash does not match p3_reuse_audit")
        arrays_manifest[f"P3_s{seed}"] = record
        p3[seed] = pred

    # Official frozen valid predictions for C/G (endpoints B/E) and F
    # (verification only, used to reproduce the original contrasts).
    for arm in ("C", "G", "F"):
        for seed in SEEDS:
            freeze = read_json(ROUND / f"valid_freeze_{arm}_s{seed}.json")
            name = Path(freeze["prediction_file"]).name
            path = resolve_array(name, freeze["prediction_sha256"], args.array_root)
            record, pred = verify_npz_array(
                f"official_{arm}_s{seed}",
                path,
                freeze["prediction_sha256"],
                freeze["prediction_array_sha256"],
                f"official frozen valid prediction, arm {arm}, seed {seed}",
            )
            expected_freeze = pred_freeze["entries"][f"{arm}{seed}"]["array_sha256"]
            if record["array_sha256_float32"] != expected_freeze:
                raise SystemExit(
                    f"{arm} s{seed} array hash does not match prediction freeze"
                )
            arrays_manifest[f"official_{arm}_s{seed}"] = record
            official[(arm, seed)] = pred

    # Frozen mechanism predictions (all C/G x C/G/uniform cells).
    for seed in SEEDS:
        for arm in ("C", "G"):
            for mode in ("C", "G", "uniform"):
                entry = mech["per_seed"][str(seed)][arm]["modes"][mode]
                name = Path(entry["prediction_file"]).name
                path = resolve_array(
                    name, entry["prediction_sha256"], args.array_root
                )
                record, pred = verify_npz_array(
                    f"mech_{arm}_s{seed}_{mode}",
                    path,
                    entry["prediction_sha256"],
                    entry["prediction_array_sha256"],
                    f"frozen mechanism prediction: arm {arm}, seed {seed}, mode {mode}",
                )
                arrays_manifest[f"mech_{arm}_s{seed}_{mode}"] = record
                mech_preds[(arm, seed, mode)] = pred

    # -- original metric reproduction ------------------------------------
    orig = {
        "P3": {s: p3[s] for s in SEEDS},
        "F": {s: official[("F", s)] for s in SEEDS},
        "G": {s: official[("G", s)] for s in SEEDS},
        "C": {s: official[("C", s)] for s in SEEDS},
    }

    mae_reproduction: dict[str, Any] = {}
    max_mae_diff = 0.0
    for name in ("P3", "F", "G", "C"):
        for seed in SEEDS:
            got = mae(orig[name][seed], y)
            expected = float(valid_metrics["models"]["per_seed"][name][str(seed)]["mae"])
            max_mae_diff = max(max_mae_diff, abs(got - expected))
            mae_reproduction[f"{name}_s{seed}"] = {
                "got": got,
                "expected": expected,
                "abs_diff": abs(got - expected),
            }
        got_mean = float(np.mean([mae(orig[name][s], y) for s in SEEDS]))
        expected_mean = float(valid_metrics["models"]["seed_mean_mae"][name])
        max_mae_diff = max(max_mae_diff, abs(got_mean - expected_mean))
        mae_reproduction[f"{name}_seed_mean"] = {
            "got": got_mean,
            "expected": expected_mean,
            "abs_diff": abs(got_mean - expected_mean),
        }
        if abs(
            float(valid_decision["seed_mean_mae"][name]) - expected_mean
        ) > 1e-15:
            raise SystemExit(f"valid_decision seed mean mismatch for {name}")

    masks = {
        "all": np.ones(N_VALID, dtype=bool),
        "ordinary_y_gt_minus3": y > TAIL_THRESHOLD,
        "without_extreme_valid172": np.arange(N_VALID) != EXTREME_VALID_ROW,
    }
    mask_checks: dict[str, Any] = {}
    for key, mask in masks.items():
        got_rows = sha256_array(np.flatnonzero(mask), "int64")
        expected_rows = valid_metrics["masks"][key]["rows_sha256"]
        if got_rows != expected_rows:
            raise SystemExit(f"mask rows hash mismatch for {key}")
        mask_checks[key] = {
            "n": int(mask.sum()),
            "rows_sha256": got_rows,
            "matches_valid_metrics": True,
        }

    drop10_order = error_rank(orig["P3"], y)[:10]
    drop10_mask = np.ones(N_VALID, dtype=bool)
    drop10_mask[drop10_order] = False
    mask_checks["drop_p3_worst10_original_mask"] = {
        "n": int(drop10_mask.sum()),
        "rows_sha256": sha256_array(np.flatnonzero(drop10_mask), "int64"),
        "rows": [int(i) for i in drop10_order],
        "selection": "original error_rank(P3, y)[:10]; fixed before this analysis",
        "matches_valid_metrics": None,  # no mask hash was stored; see contrast checks
    }

    contrast_reproduction: dict[str, Any] = {}
    max_contrast_diff = 0.0
    for key, expected in valid_metrics["contrasts"].items():
        cand_name, ref_name = key.split("_minus_")
        cand, ref = orig[cand_name], orig[ref_name]
        gains = per_seed_gain(cand, ref, y)
        d = seed_averaged_delta(cand, ref, y)
        boot = paired_bootstrap(d)
        got = {
            "per_seed_gain": gains,
            "mean_gain": float(np.mean(list(gains.values()))),
            "bootstrap_mae_delta": boot,
            "ordinary_gain": float(d[masks["ordinary_y_gt_minus3"]].mean()),
            "drop_top10_p3_error_gain": top_removed_gain(d, drop10_order),
            "without_extreme_row_gain": float(
                d[masks["without_extreme_valid172"]].mean()
            ),
            "better_fraction": float((d > 0).mean()),
        }
        entry: dict[str, Any] = {"got": got}
        diffs: dict[str, float] = {}
        for field in (
            "mean_gain",
            "ordinary_gain",
            "drop_top10_p3_error_gain",
            "without_extreme_row_gain",
            "better_fraction",
        ):
            diffs[field] = abs(float(got[field]) - float(expected[field]))
            max_contrast_diff = max(max_contrast_diff, diffs[field])
        for seed_key in ("0", "1"):
            diffs[f"per_seed_gain_{seed_key}"] = abs(
                float(got["per_seed_gain"][seed_key])
                - float(expected["per_seed_gain"][seed_key])
            )
            max_contrast_diff = max(max_contrast_diff, diffs[f"per_seed_gain_{seed_key}"])
        for field in ("mean_diff", "ci95_low", "ci95_high", "better_fraction"):
            diffs[f"bootstrap_{field}"] = abs(
                float(got["bootstrap_mae_delta"][field])
                - float(expected["bootstrap_mae_delta"][field])
            )
            max_contrast_diff = max(
                max_contrast_diff, diffs[f"bootstrap_{field}"]
            )
        if max(diffs.values()) > 1e-12:
            raise SystemExit(
                f"original contrast reproduction failed for {key}: {diffs}"
            )
        entry["max_abs_diff_vs_valid_metrics"] = max(diffs.values())
        contrast_reproduction[key] = entry

    p3_mae_reproduction: dict[str, Any] = {}
    for seed in SEEDS:
        got = mae(orig["P3"][seed], y)
        expected_p3 = float(valid_metrics["p3_replay_check"]["per_seed_mae"][str(seed)])
        expected_hist = float(
            valid_metrics["p3_replay_check"]["recorded_historical_mae"][str(seed)]
        )
        expected_audit = float(
            p3_audit["p3"][str(seed)]["recorded_historical_valid_mae"]
        )
        if not (abs(got - expected_p3) <= 1e-15 and abs(got - expected_hist) <= 1e-15):
            raise SystemExit(f"P3 s{seed} MAE does not match original score record")
        if abs(got - expected_audit) > 1e-15:
            raise SystemExit(f"P3 s{seed} MAE does not match p3_reuse_audit")
        p3_mae_reproduction[str(seed)] = {
            "got": got,
            "valid_metrics_per_seed_mae": expected_p3,
            "recorded_historical_mae": expected_hist,
            "p3_reuse_audit_recorded_mae": expected_audit,
            "abs_diff": abs(got - expected_p3),
        }
    p3_seed_mean = float(np.mean([mae(orig["P3"][s], y) for s in SEEDS]))
    if abs(p3_seed_mean - float(valid_decision["seed_mean_mae"]["P3"])) > 1e-15:
        raise SystemExit("P3 seed-mean MAE does not match valid_decision")

    # -- fixed endpoints -------------------------------------------------
    endpoint_preds: dict[str, dict[int, np.ndarray]] = {
        "A": {s: p3[s] for s in SEEDS},
        "B": {s: official[("C", s)] for s in SEEDS},
        "C": {s: mech_preds[("C", s, "G")] for s in SEEDS},
        "D": {s: mech_preds[("C", s, "uniform")] for s in SEEDS},
        "E": {s: official[("G", s)] for s in SEEDS},
        "F": {s: mech_preds[("G", s, "uniform")] for s in SEEDS},
    }

    endpoint_mae: dict[str, Any] = {}
    for endpoint, by_seed in endpoint_preds.items():
        endpoint_mae[endpoint] = {
            str(s): mae(by_seed[s], y) for s in SEEDS
        }
        endpoint_mae[endpoint]["seed_mean"] = float(
            np.mean([endpoint_mae[endpoint][str(s)] for s in SEEDS])
        )

    # Cross-check the mechanism C/G mode arrays against the official arrays.
    mode_cross_checks: dict[str, Any] = {}
    for arm, mode in (("C", "C"), ("G", "G")):
        mode_cross_checks[arm] = {}
        for seed in SEEDS:
            diff = float(
                np.abs(mech_preds[(arm, seed, mode)] - official[(arm, seed)]).max()
            )
            mode_cross_checks[arm][str(seed)] = {
                "max_abs_diff": diff,
                "within_1e-5": bool(diff <= 1e-5),
                "mechanism_record_primary_max_abs_diff": float(
                    mech["per_seed"][str(seed)][arm]["modes"][
                        "primary_matches_frozen_valid"
                    ]["max_abs_diff"]
                ),
                "mechanism_record_within_1e-5": bool(
                    mech["per_seed"][str(seed)][arm]["modes"][
                        "primary_matches_frozen_valid"
                    ]["within_1e-5"]
                ),
            }
            if diff > 1e-5:
                raise SystemExit(
                    f"mechanism {arm}-mode does not reproduce official {arm} s{seed}"
                )

    # -- fixed comparisons ----------------------------------------------
    comparisons: dict[str, Any] = {}
    for cand_name, ref_name in COMPARISONS:
        cand, ref = endpoint_preds[cand_name], endpoint_preds[ref_name]
        gains = per_seed_gain(cand, ref, y)
        d = seed_averaged_delta(cand, ref, y)
        boot = paired_bootstrap(d)
        comparisons[f"{cand_name}_vs_{ref_name}"] = {
            "candidate_endpoint": cand_name,
            "reference_endpoint": ref_name,
            "candidate": ENDPOINT_DESCRIPTIONS[cand_name]["description"],
            "reference": ENDPOINT_DESCRIPTIONS[ref_name]["description"],
            "definition": f"gain = MAE({ref_name}) - MAE({cand_name}); positive = {cand_name} better",
            "per_seed_gain": gains,
            "mean_gain": float(np.mean(list(gains.values()))),
            "bootstrap_mae_delta": boot,
            "n": N_VALID,
        }

    # -- fixed robustness groups ----------------------------------------
    robustness_groups = {
        "all": masks["all"],
        "ordinary_y_gt_minus3": masks["ordinary_y_gt_minus3"],
        "drop_p3_worst10_original_mask": drop10_mask,
        "without_row_172": masks["without_extreme_valid172"],
    }
    robustness: dict[str, Any] = {}
    for group, mask in robustness_groups.items():
        entry: dict[str, Any] = {
            "n": int(mask.sum()),
            "rows_sha256": sha256_array(np.flatnonzero(mask), "int64"),
            "definition": {
                "all": "all 1,000 valid molecules",
                "ordinary_y_gt_minus3": "y > -3 (original definition)",
                "drop_p3_worst10_original_mask": (
                    "drop the 10 rows with the largest mean P3 absolute error; "
                    "mask fixed from the original P3 prediction and y"
                ),
                "without_row_172": "drop the original extreme row 172",
            }[group],
            "mae": {},
            "gains": {},
        }
        for endpoint, by_seed in endpoint_preds.items():
            entry["mae"][endpoint] = {
                str(s): mae(by_seed[s], y, mask) for s in SEEDS
            }
            entry["mae"][endpoint]["seed_mean"] = float(
                np.mean([entry["mae"][endpoint][str(s)] for s in SEEDS])
            )
        for cand_name, ref_name in COMPARISONS:
            gains = per_seed_gain(
                endpoint_preds[cand_name], endpoint_preds[ref_name], y, mask
            )
            entry["gains"][f"{cand_name}_vs_{ref_name}"] = {
                "per_seed_gain": gains,
                "mean_gain": float(np.mean(list(gains.values()))),
            }
        robustness[group] = entry

    # The full-set robustness block must equal the main table.
    for endpoint in "ABCDEF":
        got = float(robustness["all"]["mae"][endpoint]["seed_mean"])
        if abs(got - float(endpoint_mae[endpoint]["seed_mean"])) > 1e-15:
            raise SystemExit(f"all-group MAE mismatch for endpoint {endpoint}")
    for cand_name, ref_name in COMPARISONS:
        key = f"{cand_name}_vs_{ref_name}"
        got = float(robustness["all"]["gains"][key]["mean_gain"])
        expected = float(comparisons[key]["mean_gain"])
        if abs(got - expected) > 1e-15:
            raise SystemExit(f"all-group gain mismatch for {key}")

    # -- checks / manifest ----------------------------------------------
    checks = {
        "all_arrays_present": True,
        "all_file_hashes_match": True,
        "all_array_hashes_match": True,
        "all_rows_are_original_order_0_999": True,
        "all_predictions_finite": True,
        "label_hashes_match_valid_metrics_and_read_lock": True,
        "p3_prediction_matches_original_score_record": True,
        "p3_mae_matches_original_score_record": p3_mae_reproduction,
        "c_mode_reproduces_official_valid": mode_cross_checks["C"],
        "g_mode_reproduces_official_valid": mode_cross_checks["G"],
        "original_contrasts_reproduced": {
            "max_abs_diff_vs_valid_metrics": max_contrast_diff,
            "tolerance": 1e-12,
            "all_six_contrasts": True,
        },
        "original_model_mae_reproduced": {
            "max_abs_diff_vs_valid_metrics": max_mae_diff,
            "tolerance": 1e-12,
        },
        "masks": mask_checks,
        "drop_p3_worst10_gain_reproduced": {
            "contrasts": {
                f"{name}_minus_P3": contrast_reproduction[f"{name}_minus_P3"][
                    "got"
                ]["drop_top10_p3_error_gain"]
                for name in ("C", "G", "F")
            },
            "matches_valid_metrics": True,
        },
        "valid_read_lock_untouched": True,
        "valid_read_lock_read_count": int(read_lock["read_count"]),
        "official_test_loaded": False,
        "gpu_used": False,
        "training_or_inference_run": False,
        "posthoc_labels_access_recorded": True,
    }

    # Sanity: every endpoint's declared source key exists and points at the
    # same array object that was scored.
    for endpoint, by_seed in ENDPOINT_ARRAY_KEYS.items():
        for seed, key in by_seed.items():
            if key not in arrays_manifest:
                raise SystemExit(f"endpoint {endpoint} s{seed} source key missing")
            if endpoint_preds[endpoint][seed].shape != (N_VALID,):
                raise SystemExit(f"endpoint {endpoint} s{seed} shape invalid")

    metrics = {
        "protocol_id": PROTOCOL_ID,
        "posthoc_id": POSTHOC_ID,
        "stage": "posthoc_frozen_mode_error",
        "status": "POST_HOC_EXPLORATORY",
        "created_utc": created_utc,
        "git_commit": git_commit(),
        "question": (
            "Under the same frozen trained weights, does restoring the uniform "
            "(P3-Null-equivalent) context mapping reduce raw-y MAE, and does it "
            "recover P3?"
        ),
        "scope": {
            "new_training": False,
            "inference_run": False,
            "gpu_used": False,
            "official_test_loaded": False,
            "hyperparameter_search": False,
            "endpoints_fixed": True,
            "seed_count": len(SEEDS),
            "n_valid": N_VALID,
            "metric": "raw-y MAE",
            "seed_aggregation": (
                "per-seed MAE first, then arithmetic mean of the two seed MAEs; "
                "no prediction ensembling"
            ),
            "bootstrap": {
                "n_resamples": BOOTSTRAP_N,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "unit": "molecule",
                "shared_resample_index_across_seeds": True,
                "note": (
                    "d_i = mean_s (|e_ref,s,i| - |e_cand,s,i|), then molecule-level "
                    "bootstrap; the same molecule index is therefore used for both seeds"
                ),
            },
            "posthoc_note": (
                "Proposed after the original official-valid scores were seen. "
                "Not an independent confirmation and not a blind test."
            ),
        },
        "labels": {
            "path": labels_manifest["path"],
            "file_sha256": labels_manifest["file_sha256"],
            "array_sha256_float32": y_f32,
            "array_sha256_float64": y_f64,
            "matches_valid_metrics": True,
            "matches_valid_read_lock": True,
            "historical_read_count": int(read_lock["read_count"]),
            "historical_reread": bool(read_lock["reread"]),
            "posthoc_access_to_existing_labels": True,
            "note": (
                "The frozen read_count=1 record is unchanged. This post-hoc "
                "analysis additionally read the same cached valid labels; the "
                "track must not be described as having only one valid-label use."
            ),
        },
        "endpoints": {
            endpoint: {
                **ENDPOINT_DESCRIPTIONS[endpoint],
                "mae": endpoint_mae[endpoint],
                "source_arrays": {
                    str(seed): ENDPOINT_ARRAY_KEYS[endpoint][seed] for seed in SEEDS
                },
            }
            for endpoint in "ABCDEF"
        },
        "comparisons": comparisons,
        "robustness": robustness,
        "checks": checks,
        "interpretation": {
            "uniform_vs_own_mode": {
                "D_vs_B_mean_gain": comparisons["D_vs_B"]["mean_gain"],
                "F_vs_E_mean_gain": comparisons["F_vs_E"]["mean_gain"],
                "reading": (
                    "Restoring uniform is much worse than the model's own mode on "
                    "both seeds; this is not an MAE improvement."
                ),
            },
            "uniform_vs_P3": {
                "D_vs_A_mean_gain": comparisons["D_vs_A"]["mean_gain"],
                "F_vs_A_mean_gain": comparisons["F_vs_A"]["mean_gain"],
                "reading": (
                    "Uniform does not recover P3; it is substantially worse. "
                    "Inference-time restoration is not a rescue path."
                ),
            },
            "coadaptation": (
                "The large same-weight degradation supports co-adaptation between "
                "the attention context and the rest of the trained network. It does "
                "not show that the learned reweighting is beneficial, and it means "
                "the original valid gap should not be read as a clean causal effect "
                "of the reweighting operation alone."
            ),
            "posthoc_caveat": (
                "Same-weight mode switching can be off the training distribution. "
                "This is an intervention diagnostic, not a chemistry claim and not "
                "an estimate of training-randomness uncertainty."
            ),
        },
        "recommendation": {
            "continue_attention_branch": False,
            "reason": (
                "No same-weight mode restores P3, and uniform is much worse than "
                "both the original modes and P3. The original negative result is "
                "strengthened, not rescued."
            ),
            "next_step": (
                "Do not start new training from this analysis. If the track later "
                "tests a zero-initialised gated correction, treat it as a separate "
                "pre-registered hypothesis with a fresh confirmation endpoint; this "
                "post-hoc result does not justify it."
            ),
        },
    }

    input_manifest = {
        "protocol_id": PROTOCOL_ID,
        "posthoc_id": POSTHOC_ID,
        "created_utc": created_utc,
        "git_commit": git_commit(),
        "source_note": (
            "Read-only inputs. The C/G/F official and mechanism .npz arrays are not "
            "committed in the result directory (round convention); they were read "
            "from the verified .rr/pulled copies of the frozen compute-host outputs. "
            "The P3 arrays came from their historical frozen paths. Every file and "
            "array hash was checked against the frozen records before scoring."
        ),
        "records": records_manifest,
        "labels": labels_manifest,
        "arrays": arrays_manifest,
        "endpoint_sources": {
            endpoint: {
                str(seed): ENDPOINT_ARRAY_KEYS[endpoint][seed] for seed in SEEDS
            }
            for endpoint in "ABCDEF"
        },
        "checks": {
            "all_arrays_present": True,
            "all_file_hashes_match": True,
            "all_array_hashes_match": True,
            "all_rows_are_original_order_0_999": True,
            "all_predictions_finite": True,
            "labels_match_valid_metrics_and_read_lock": True,
            "p3_matches_original_score_record": True,
            "c_mode_matches_official_c": mode_cross_checks["C"],
            "g_mode_matches_official_g": mode_cross_checks["G"],
            "original_contrasts_reproduced": {
                "max_abs_diff_vs_valid_metrics": max_contrast_diff,
                "tolerance": 1e-12,
            },
            "original_model_mae_reproduced": {
                "max_abs_diff_vs_valid_metrics": max_mae_diff,
                "tolerance": 1e-12,
            },
            "masks_reproduced": mask_checks,
            "official_test_loaded": False,
            "gpu_used": False,
            "training_or_inference_run": False,
        },
    }

    label_access = {
        "protocol_id": PROTOCOL_ID,
        "posthoc_id": POSTHOC_ID,
        "created_utc": created_utc,
        "original_round": {
            "valid_read_lock_path": rel(READ_LOCK_PATH),
            "valid_read_lock_sha256": sha256_file(READ_LOCK_PATH),
            "read_count": int(read_lock["read_count"]),
            "reread": bool(read_lock["reread"]),
            "read_utc": read_lock["read_utc"],
            "untouched_by_this_analysis": True,
        },
        "this_posthoc_analysis": {
            "accessed_existing_valid_labels": True,
            "label_path": rel(Y_PATH),
            "label_sha256_float32": y_f32,
            "label_sha256_float64": y_f64,
            "purpose": "post-hoc error scoring of already-frozen valid predictions",
            "new_independent_label_read": False,
            "additional_access_to_already_read_cache": True,
            "modifies_valid_read_lock": False,
            "official_test_loaded": False,
            "gpu_used": False,
            "training_or_inference_run": False,
        },
        "statement": (
            "The frozen valid_read.lock remains read_count=1, reread=false. "
            "This post-hoc analysis additionally accessed the same cached valid "
            "labels after the original scores were known. Do not describe the "
            "whole track as having only one valid-label use; this access is an "
            "extra, explicitly post-hoc use."
        ),
    }

    if args.check_only:
        print("all checks passed")
        print(
            "endpoint seed-mean MAE: "
            + ", ".join(
                f"{ep}={endpoint_mae[ep]['seed_mean']:.9f}" for ep in "ABCDEF"
            )
        )
        for cand_name, ref_name in COMPARISONS:
            c = comparisons[f"{cand_name}_vs_{ref_name}"]
            print(
                f"{cand_name} vs {ref_name}: gain={c['mean_gain']:+.9f} "
                f"CI=[{c['bootstrap_mae_delta']['ci95_low']:+.9f},"
                f"{c['bootstrap_mae_delta']['ci95_high']:+.9f}]"
            )
        return 0

    write_json(HERE / "metrics.json", metrics)
    write_json(HERE / "input_manifest.json", input_manifest)
    write_json(HERE / "valid_label_access.json", label_access)
    print(f"wrote {rel(HERE / 'metrics.json')}")
    print(f"wrote {rel(HERE / 'input_manifest.json')}")
    print(f"wrote {rel(HERE / 'valid_label_access.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
