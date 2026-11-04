"""Post-hoc terminal Test evaluation of frozen Peptides-func checkpoints.

This entry point is deliberately separate from the Validation-only training
path: it is only valid after an explicit user authorization file exists.  It
does not train, select, tune or modify anything.  It reconstructs each frozen
checkpoint, evaluates Train/Validation (sanity) and the official Test split, and
records the Test metrics for comparison.

Usage (repository root, CPU or GPU):

    uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_test_eval.py \
        --device cpu --out-dir tracks/.../runs/test_eval
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[2]
DIAG = REPO / "tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1"
DIAG_CODE = DIAG / "code"
sys.path.insert(0, str(DIAG_CODE))

import diag_common as H  # noqa: E402

import bp_data as D  # noqa: E402
import bp_model as BM  # noqa: E402
import bp_train as T  # noqa: E402

AMENDMENT_ID = "posthoc-terminal-test-eval-v1"
RESULT_LABEL = "POSTHOC_AUTHORIZED_TERMINAL_TEST_EVALUATION"
AUTHORIZATION = ROUND / "analysis/test_access_authorization.json"
AMENDMENT = ROUND / "configs/test_access_amendment.json"
TEST_CACHE = H.PREV_CACHE / "peptides_func_test.pt"
TEST_LABELS = H.PREV_ANALYSIS / "peptides_func_y_test.npy"
ROW_IDENTITY = DIAG / "analysis/row_identity.json"
POS_WEIGHT = ROUND / "analysis/pos_weight.json"
RESULTS = ROUND / "results"
ANALYSIS = ROUND / "analysis"
FORMAL_RUN_DIR = ROUND / "runs/weighted_bce_seed0"
HISTORY_RUN_DIR = DIAG / "runs/peptide_c1_seed0_240ep"
FASTV1_CKPT = H.history_checkpoint_path(0)
FASTV1_RUN = H.history_run(0)
HISTORY_RUN = H.read_json(HISTORY_RUN_DIR / "run.json")
FORMAL_RUN = H.read_json(FORMAL_RUN_DIR / "run.json")
LABELS = tuple(H.LABELS)
N_LABELS = int(H.N_LABELS)


def sha256_file(path: Path | str) -> str:
    return H.sha256_file(path)


def read_json(path: Path | str) -> Any:
    return H.read_json(path)


def write_json(path: Path | str, payload: Any) -> None:
    H.write_json(path, payload)


def track_rel(path: Path | str) -> str:
    return H.track_rel(path)


def utcnow() -> str:
    return H.utcnow()


def log(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


def array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def build_model(data, seed: int, device: torch.device):
    norm_payload = {
        **data.norm,
        "continuous_atom": {
            "dim": int(data.cont_norm.get("dim", 0)),
            "mean": data.cont_norm.get("mean", []),
            "scale": data.cont_norm.get("scale", []),
            "fingerprint": data.cont_norm.get("fingerprint", ""),
        },
    }
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return BM.build_c1(
        data.spec,
        norm_payload,
        data.n_out,
        int(seed),
        device,
        cont_slice=data.cont_slice,
        cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
    )


def checkpoint_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "fastv1_seed0_120ep",
            "role": "historical Test-pipeline sanity check",
            "path": FASTV1_CKPT,
            "run": FASTV1_RUN,
            "run_json": H.history_dir(0) / "run.json",
            "stored_valid": float(FASTV1_RUN["best_metric"]),
            "stored_test": float(FASTV1_RUN["test_metrics"]["macro_ap"]),
            "stored_test_per_label": [float(x) for x in FASTV1_RUN["test_metrics"]["per_label"]],
            "frozen_at": str(FASTV1_RUN["test_evaluated_once_utc"]),
            "loss": "ordinary BCE",
            "selection_metric": "Validation macro AP",
        },
        {
            "id": "diagnostic_ordinary_bce_240ep",
            "role": "single-variable control",
            "path": HISTORY_RUN_DIR / "checkpoint.pt",
            "run": HISTORY_RUN,
            "run_json": HISTORY_RUN_DIR / "run.json",
            "stored_valid": float(HISTORY_RUN["best_valid_macro_ap"]),
            "stored_test": None,
            "stored_test_per_label": None,
            "frozen_at": str(HISTORY_RUN["finished_utc"]),
            "loss": "ordinary BCE",
            "selection_metric": "Validation macro AP",
        },
        {
            "id": "weighted_bce_240ep",
            "role": "single-variable intervention",
            "path": FORMAL_RUN_DIR / "checkpoint.pt",
            "run": FORMAL_RUN,
            "run_json": FORMAL_RUN_DIR / "run.json",
            "stored_valid": float(FORMAL_RUN["best_valid_macro_ap"]),
            "stored_test": None,
            "stored_test_per_label": None,
            "frozen_at": str(FORMAL_RUN["finished_utc"]),
            "loss": "weighted BCE",
            "selection_metric": "Validation macro AP",
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(ROUND / "runs/test_eval"))
    parser.add_argument("--authorization", default=str(AUTHORIZATION))
    parser.add_argument("--amendment", default=str(AMENDMENT))
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    auth = read_json(Path(args.authorization))
    amendment = read_json(Path(args.amendment))
    if auth.get("authorized") is not True:
        raise SystemExit("TEST_EVALUATION_NOT_AUTHORIZED")
    if amendment.get("status") != "AUTHORIZED_POSTHOC_TERMINAL_TEST_EVALUATION":
        raise SystemExit(f"TEST_AMENDMENT_STATUS_INVALID: {amendment.get('status')!r}")
    auth_time = parse_utc(str(auth["authorization_received_utc"]))
    log(f"authorization present: {auth['authorization_id']} at {auth['authorization_received_utc']}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA_UNAVAILABLE")
    data = T._make_task("peptide", H.DATASET, 0)
    test_cache = D.load_cache(TEST_CACHE)
    y_test = np.load(TEST_LABELS).astype(np.float64)
    test_rows = np.arange(int(test_cache["n_molecules"]), dtype=np.int64)
    if y_test.shape != (int(test_cache["n_molecules"]), N_LABELS):
        raise SystemExit(f"TEST_LABEL_SHAPE_MISMATCH: {y_test.shape}")
    if not np.isfinite(y_test).all() or not np.isin(np.unique(y_test), [0.0, 1.0]).all():
        raise SystemExit("TEST_LABELS_NOT_FINITE_BINARY")
    if y_test.shape[0] != int(H.SPLIT_SIZES["test"]):
        raise SystemExit("TEST_SPLIT_SIZE_MISMATCH")
    test_cache_sha = sha256_file(TEST_CACHE)
    test_cache_spec_hash = str(test_cache["meta"]["feature_spec_hash"])
    test_cache_split = str(test_cache["meta"]["split"])
    if test_cache_spec_hash != data.spec.hash():
        raise SystemExit(
            f"TEST_FEATURE_SPEC_MISMATCH: {test_cache_spec_hash} != {data.spec.hash()}"
        )
    if test_cache_split != "test":
        raise SystemExit(f"TEST_CACHE_SPLIT_INVALID: {test_cache_split!r}")
    fastv1_test_cache_sha = str(FASTV1_RUN["test_cache_identity"]["cache_test_sha256"])
    row_identity = read_json(ROW_IDENTITY)
    row_identity_ok = bool(
        row_identity.get("status") == "PASS"
        and int(row_identity["per_split"]["test"]["n_rows"]) == int(H.SPLIT_SIZES["test"])
    )
    if not row_identity_ok:
        raise SystemExit("TEST_ROW_IDENTITY_NOT_PASSED")
    if test_cache_sha != fastv1_test_cache_sha:
        raise SystemExit(f"TEST_CACHE_SHA_MISMATCH: {test_cache_sha} != {fastv1_test_cache_sha}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_payload = read_json(POS_WEIGHT)
    pos_by_label = {row["label"]: float(row["pos_weight"]) for row in pos_payload["labels"]}
    pos_rate_by_label = {row["label"]: float(row["train_pos_rate"]) for row in pos_payload["labels"]}

    results: list[dict[str, Any]] = []
    for spec in checkpoint_specs():
        ckpt_path = Path(spec["path"])
        expected_sha = str(spec["run"]["checkpoint_sha256"])
        actual_sha = sha256_file(ckpt_path)
        if actual_sha != expected_sha:
            raise SystemExit(f"CHECKPOINT_SHA_MISMATCH {spec['id']}: {actual_sha} != {expected_sha}")
        frozen_at = parse_utc(str(spec["frozen_at"]))
        if frozen_at >= auth_time:
            raise SystemExit(
                f"CHECKPOINT_NOT_FROZEN_BEFORE_AUTHORIZATION {spec['id']}: {spec['frozen_at']} >= "
                f"{auth['authorization_received_utc']}"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, model_info = build_model(data, 0, device)
        model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()}, strict=True)
        model.eval()
        valid_metrics, _, _ = T.evaluate(
            data.kind, data.dataset, model, data.valid_cache, data.valid_rows, data.y_valid, device,
            n_classes=data.n_out,
        )
        test_metrics, test_logits, test_probs = T.evaluate(
            data.kind, data.dataset, model, test_cache, test_rows, y_test, device,
            n_classes=data.n_out,
        )
        valid_ap = [float(x) for x in valid_metrics["per_label"]]
        test_ap = [float(x) for x in test_metrics["per_label"]]
        recheck = [
            float(average_precision_score(y_test[:, j], test_probs[:, j])) for j in range(N_LABELS)
        ]
        recheck_delta = max(abs(test_ap[j] - recheck[j]) for j in range(N_LABELS))
        if recheck_delta > 1e-12:
            raise SystemExit(f"TEST_AP_RECHECK_MISMATCH {spec['id']}: {recheck_delta}")
        if not np.isfinite(test_probs).all() or test_probs.min() < 0.0 or test_probs.max() > 1.0:
            raise SystemExit(f"TEST_PROBS_RANGE_INVALID {spec['id']}")
        np.savez_compressed(
            out_dir / f"{spec['id']}_test_predictions.npz",
            test_probs=test_probs,
            y_test=y_test,
            labels=np.asarray(LABELS),
            test_rows=test_rows,
        )
        stored_test = spec["stored_test"]
        stored_test_delta = None if stored_test is None else abs(float(test_metrics["value"]) - stored_test)
        stored_test_per_label_delta = None
        if spec["stored_test_per_label"] is not None:
            stored_test_per_label_delta = max(
                abs(test_ap[j] - float(spec["stored_test_per_label"][j])) for j in range(N_LABELS)
            )
        valid_delta = abs(float(valid_metrics["value"]) - float(spec["stored_valid"]))
        row = {
            "id": spec["id"],
            "role": spec["role"],
            "loss": spec["loss"],
            "checkpoint_path": track_rel(ckpt_path),
            "checkpoint_sha256": actual_sha,
            "best_epoch": int(spec["run"]["best_epoch"]),
            "selection_metric": spec["selection_metric"],
            "frozen_at": spec["frozen_at"],
            "stored_validation_macro_ap": float(spec["stored_valid"]),
            "recomputed_validation_macro_ap": float(valid_metrics["value"]),
            "validation_delta": float(valid_delta),
            "stored_test_macro_ap": stored_test,
            "test_macro_ap": float(test_metrics["value"]),
            "test_macro_ap_pct": 100.0 * float(test_metrics["value"]),
            "test_minus_validation_pp": 100.0 * (float(test_metrics["value"]) - float(valid_metrics["value"])),
            "stored_test_delta": stored_test_delta,
            "stored_test_per_label_delta": stored_test_per_label_delta,
            "test_ap_recheck_delta": float(recheck_delta),
            "test_probs_min": float(test_probs.min()),
            "test_probs_max": float(test_probs.max()),
            "per_label_valid_ap": {name: valid_ap[j] for j, name in enumerate(LABELS)},
            "per_label_test_ap": {name: test_ap[j] for j, name in enumerate(LABELS)},
        }
        results.append(row)
        log(
            f"{spec['id']}: valid {100.0 * float(valid_metrics['value']):.4f}% "
            f"(stored {100.0 * float(spec['stored_valid']):.4f}%), "
            f"test {100.0 * float(test_metrics['value']):.4f}%"
        )

    by_id = {row["id"]: row for row in results}
    ordinary = by_id["diagnostic_ordinary_bce_240ep"]
    weighted = by_id["weighted_bce_240ep"]
    fastv1 = by_id["fastv1_seed0_120ep"]
    comparison = {
        "ordinary_bce_240ep": {
            "validation_macro_ap": ordinary["recomputed_validation_macro_ap"],
            "test_macro_ap": ordinary["test_macro_ap"],
        },
        "weighted_bce_240ep": {
            "validation_macro_ap": weighted["recomputed_validation_macro_ap"],
            "test_macro_ap": weighted["test_macro_ap"],
        },
        "weighted_minus_ordinary_validation_pp": 100.0
        * (weighted["recomputed_validation_macro_ap"] - ordinary["recomputed_validation_macro_ap"]),
        "weighted_minus_ordinary_test_pp": 100.0
        * (weighted["test_macro_ap"] - ordinary["test_macro_ap"]),
        "fastv1_test_pipeline_sanity": {
            "recomputed_test_macro_ap": fastv1["test_macro_ap"],
            "stored_test_macro_ap": fastv1["stored_test_macro_ap"],
            "delta": fastv1["stored_test_delta"],
            "per_label_delta": fastv1["stored_test_per_label_delta"],
        },
    }

    # ---- comparison tables ------------------------------------------------
    comparison_rows = []
    for row in results:
        comparison_rows.append(
            {
                "method": {
                    "fastv1_seed0_120ep": "LCRC-C1 Fast v1 120ep (historical Test sanity)",
                    "diagnostic_ordinary_bce_240ep": "LCRC-C1 ordinary BCE 240ep",
                    "weighted_bce_240ep": "LCRC-C1 weighted BCE 240ep",
                }[row["id"]],
                "role": row["role"],
                "seed": 0,
                "loss": row["loss"],
                "best_epoch": row["best_epoch"],
                "validation_macro_ap": row["recomputed_validation_macro_ap"],
                "validation_macro_ap_pct": 100.0 * row["recomputed_validation_macro_ap"],
                "test_macro_ap": row["test_macro_ap"],
                "test_macro_ap_pct": row["test_macro_ap_pct"],
                "test_minus_validation_pp": row["test_minus_validation_pp"],
                "stored_test_macro_ap": "" if row["stored_test_macro_ap"] is None else row["stored_test_macro_ap"],
                "stored_test_delta": "" if row["stored_test_delta"] is None else row["stored_test_delta"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "source": row["checkpoint_path"],
            }
        )
    comparison_fields = [
        "method",
        "role",
        "seed",
        "loss",
        "best_epoch",
        "validation_macro_ap",
        "validation_macro_ap_pct",
        "test_macro_ap",
        "test_macro_ap_pct",
        "test_minus_validation_pp",
        "stored_test_macro_ap",
        "stored_test_delta",
        "checkpoint_sha256",
        "source",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "test_comparison.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    per_label_rows = []
    for j, name in enumerate(LABELS):
        per_label_rows.append(
            {
                "label_index": j,
                "label": name,
                "train_pos_rate": pos_rate_by_label[name],
                "pos_weight": pos_by_label[name],
                "fastv1_test_ap": fastv1["per_label_test_ap"][name],
                "ordinary_test_ap": ordinary["per_label_test_ap"][name],
                "weighted_test_ap": weighted["per_label_test_ap"][name],
                "delta_test_ap_pp": 100.0
                * (weighted["per_label_test_ap"][name] - ordinary["per_label_test_ap"][name]),
                "ordinary_valid_ap": ordinary["per_label_valid_ap"][name],
                "weighted_valid_ap": weighted["per_label_valid_ap"][name],
                "delta_valid_ap_pp": 100.0
                * (weighted["per_label_valid_ap"][name] - ordinary["per_label_valid_ap"][name]),
            }
        )
    per_label_fields = [
        "label_index",
        "label",
        "train_pos_rate",
        "pos_weight",
        "fastv1_test_ap",
        "ordinary_test_ap",
        "weighted_test_ap",
        "delta_test_ap_pp",
        "ordinary_valid_ap",
        "weighted_valid_ap",
        "delta_valid_ap_pp",
    ]
    with open(RESULTS / "test_per_label_comparison.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_label_fields)
        writer.writeheader()
        writer.writerows(per_label_rows)

    # ---- gates ------------------------------------------------------------
    checks = {
        "authorization_present": auth.get("authorized") is True,
        "amendment_status_valid": amendment.get("status") == "AUTHORIZED_POSTHOC_TERMINAL_TEST_EVALUATION",
        "all_checkpoints_frozen_before_authorization": all(
            parse_utc(str(spec["frozen_at"])) < auth_time for spec in checkpoint_specs()
        ),
        "all_checkpoint_hashes_match": all(
            row["checkpoint_sha256"] == str(spec["run"]["checkpoint_sha256"])
            for row, spec in zip(results, checkpoint_specs(), strict=True)
        ),
        "test_cache_sha_matches_historical": test_cache_sha == fastv1_test_cache_sha,
        "test_labels_shape_2331x10": y_test.shape == (int(H.SPLIT_SIZES["test"]), N_LABELS),
        "test_labels_finite_binary": bool(np.isfinite(y_test).all() and np.isin(np.unique(y_test), [0.0, 1.0]).all()),
        "test_row_identity_pass": row_identity_ok,
        "fastv1_test_ap_reproduces_stored_within_1e-3": float(fastv1["stored_test_delta"]) < 1e-3,
        "fastv1_test_per_label_reproduces_stored_within_1e-3": float(fastv1["stored_test_per_label_delta"]) < 1e-3,
        "fastv1_valid_ap_reproduces_stored_within_1e-3": float(fastv1["validation_delta"]) < 1e-3,
        "ordinary_valid_ap_reproduces_stored_within_1e-3": float(ordinary["validation_delta"]) < 1e-3,
        "weighted_valid_ap_reproduces_stored_within_1e-3": float(weighted["validation_delta"]) < 1e-3,
        "test_probabilities_finite_and_in_0_1": all(
            np.isfinite(np.load(out_dir / f"{row['id']}_test_predictions.npz")["test_probs"]).all()
            and float(np.load(out_dir / f"{row['id']}_test_predictions.npz")["test_probs"].min()) >= 0.0
            and float(np.load(out_dir / f"{row['id']}_test_predictions.npz")["test_probs"].max()) <= 1.0
            for row in results
        ),
        "test_ap_recheck_delta_zero": all(float(row["test_ap_recheck_delta"]) <= 1e-12 for row in results),
        "test_cache_feature_spec_matches": test_cache_spec_hash == data.spec.hash(),
        "test_cache_split_is_test": test_cache_split == "test",
        "no_training_performed": True,
        "no_checkpoint_selection_used_test": True,
        "ecfp_not_retrained": True,
        "formal_training_run_kept_test_evaluated_false": FORMAL_RUN.get("test_evaluated") is False,
    }
    gates = {
        "amendment_id": AMENDMENT_ID,
        "round_id": "lcrc-peptides-weighted-bce-v1",
        "result_label": RESULT_LABEL,
        "generated_utc": utcnow(),
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "authorization": auth,
        "test_split": {
            "n": int(H.SPLIT_SIZES["test"]),
            "cache": track_rel(TEST_CACHE),
            "cache_sha256": test_cache_sha,
            "labels": track_rel(TEST_LABELS),
            "labels_file_sha256": sha256_file(TEST_LABELS),
            "labels_array_sha256": array_sha256(y_test),
            "row_identity_status": row_identity.get("status"),
            "feature_spec_hash": test_cache_spec_hash,
            "cache_split": test_cache_split,
        },
    }
    write_json(ANALYSIS / "test_gates.json", gates)

    payload = {
        "amendment_id": AMENDMENT_ID,
        "round_id": "lcrc-peptides-weighted-bce-v1",
        "result_label": RESULT_LABEL,
        "authorization": auth,
        "amendment": amendment,
        "test_split": gates["test_split"],
        "checkpoints": results,
        "comparison": comparison,
        "checks": checks,
        "all_passed": gates["all_passed"],
        "test_evaluated": True,
        "training_performed": False,
        "checkpoint_selection_used_test": False,
        "test_used_for_any_decision": False,
        "formal_training_test_evaluated": FORMAL_RUN.get("test_evaluated"),
        "ecfp_test_status": amendment.get("ecfp_test_status"),
        "environment": H.environment(),
        "duration_seconds": float(time.perf_counter() - t0),
        "finished_utc": utcnow(),
        "artifacts": {
            "test_comparison": "results/test_comparison.csv",
            "test_per_label_comparison": "results/test_per_label_comparison.csv",
            "test_gates": "analysis/test_gates.json",
            "predictions_dir": track_rel(out_dir),
        },
    }
    write_json(ANALYSIS / "test_eval.json", payload)
    log(
        f"ordinary Test {100.0 * ordinary['test_macro_ap']:.4f}% | "
        f"weighted Test {100.0 * weighted['test_macro_ap']:.4f}% | "
        f"delta {comparison['weighted_minus_ordinary_test_pp']:+.4f} pp"
    )
    log(f"test gates all_passed={gates['all_passed']} -> {track_rel(ANALYSIS / 'test_gates.json')}")
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
