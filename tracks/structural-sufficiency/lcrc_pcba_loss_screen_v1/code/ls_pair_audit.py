"""Combine the two GPU preflight records into the paired-initialization audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import ls_common as C  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return C.read_json(path)


def _tensor_max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def audit(
    *,
    preflight_m: Path,
    preflight_t: Path,
    out_path: Path,
) -> dict[str, Any]:
    m = _load(preflight_m)
    t = _load(preflight_t)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["both_records_present"] = True
    checks["group_M"] = m.get("group") == "M"
    checks["group_T"] = t.get("group") == "T"
    checks["preflight_M_all_passed"] = bool(m.get("all_passed"))
    checks["preflight_T_all_passed"] = bool(t.get("all_passed"))

    def get(record: dict[str, Any], *keys: str, default=None):
        cur: Any = record
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    details["run_ids"] = {
        "M": get(m, "run_id"),
        "T": get(t, "run_id"),
    }
    details["environment"] = {"M": m.get("environment"), "T": t.get("environment")}
    details["gpu_occupancy"] = {
        "M_start": m.get("gpu_occupancy_start"),
        "T_start": t.get("gpu_occupancy_start"),
        "M_end": m.get("gpu_occupancy_end"),
        "T_end": t.get("gpu_occupancy_end"),
    }
    details["gpu_hours"] = {
        "M": m.get("gpu_hours"),
        "T": t.get("gpu_hours"),
        "sum": float((m.get("gpu_hours") or 0.0) + (t.get("gpu_hours") or 0.0)),
    }

    # Checkpoint and model identity.
    sha_m = m.get("source_checkpoint_sha256")
    sha_t = t.get("source_checkpoint_sha256")
    details["source_checkpoint_sha256"] = {"M": sha_m, "T": sha_t, "expected": C.SOURCE["checkpoint_sha256"]}
    checks["checkpoint_sha_equal"] = bool(
        sha_m == sha_t == C.SOURCE["checkpoint_sha256"]
    )
    fp_m = m.get("model_parameter_fingerprint")
    fp_t = t.get("model_parameter_fingerprint")
    source_fp_m = get(m, "initialization", "source_parameter_fingerprint")
    source_fp_t = get(t, "initialization", "source_parameter_fingerprint")
    details["model_parameter_fingerprint"] = {
        "M": fp_m,
        "T": fp_t,
        "source_M": source_fp_m,
        "source_T": source_fp_t,
    }
    checks["model_fingerprints_equal"] = bool(fp_m == fp_t == source_fp_m == source_fp_t)
    checks["model_state_bitwise_equal"] = bool(
        get(m, "initialization", "state_dict_bitwise_equal")
        and get(t, "initialization", "state_dict_bitwise_equal")
    )

    # AdamW state identity.
    opt_m = m.get("optimizer_state_fingerprint")
    opt_t = t.get("optimizer_state_fingerprint")
    opt_src_m = m.get("source_optimizer_state_fingerprint")
    opt_src_t = t.get("source_optimizer_state_fingerprint")
    details["optimizer_state_fingerprint"] = {
        "M": opt_m,
        "T": opt_t,
        "source_M": opt_src_m,
        "source_T": opt_src_t,
    }
    checks["optimizer_fingerprints_equal"] = bool(
        opt_m == opt_t == opt_src_m == opt_src_t
    )
    checks["optimizer_state_identical_to_source"] = bool(
        get(m, "optimizer_state", "state_identical_to_source")
        and get(t, "optimizer_state", "state_identical_to_source")
    )
    details["optimizer_state_audit"] = {
        "M": m.get("optimizer_state"),
        "T": t.get("optimizer_state"),
    }

    # Initial Validation AP.
    ap0_m = m.get("epoch0_ap")
    ap0_t = t.get("epoch0_ap")
    delta_hist_m = abs(float(ap0_m) - float(C.SOURCE["best_valid_ap"])) if ap0_m is not None else None
    delta_hist_t = abs(float(ap0_t) - float(C.SOURCE["best_valid_ap"])) if ap0_t is not None else None
    details["epoch0_ap"] = {
        "M": ap0_m,
        "T": ap0_t,
        "historical": C.SOURCE["best_valid_ap"],
        "abs_delta_historical_M": delta_hist_m,
        "abs_delta_historical_T": delta_hist_t,
        "abs_delta_pair": None
        if ap0_m is None or ap0_t is None
        else abs(float(ap0_m) - float(ap0_t)),
    }
    checks["epoch0_ap_reproduces_historical"] = bool(
        delta_hist_m is not None
        and delta_hist_t is not None
        and delta_hist_m <= 1e-4
        and delta_hist_t <= 1e-4
    )
    checks["epoch0_ap_pair_equal_within_1e_5"] = bool(
        ap0_m is not None
        and ap0_t is not None
        and abs(float(ap0_m) - float(ap0_t)) <= 1e-5
    )
    checks["epoch0_official_independent_agree"] = bool(
        get(m, "gates", "epoch0_official_independent_agree")
        and get(t, "gates", "epoch0_official_independent_agree")
    )

    # Batch order / inputs / labels.
    order_m = get(m, "training_order", "global_plan_sha256")
    order_t = get(t, "training_order", "global_plan_sha256")
    epoch1_m = get(m, "training_order", "epochs_detail", default=[{}])[0]
    epoch1_t = get(t, "training_order", "epochs_detail", default=[{}])[0]
    details["training_order"] = {
        "global_plan_sha256_M": order_m,
        "global_plan_sha256_T": order_t,
        "epoch1_M": epoch1_m,
        "epoch1_T": epoch1_t,
    }
    checks["training_order_identical"] = bool(
        order_m == order_t
        and epoch1_m.get("row_order_sha256") == epoch1_t.get("row_order_sha256")
        and epoch1_m.get("batch_plan_sha256") == epoch1_t.get("batch_plan_sha256")
    )
    checks["no_valid_or_test_rows_in_training"] = bool(
        get(m, "training_order_isolation", "passed")
        and get(t, "training_order_isolation", "passed")
    )

    # RNG start.
    rng_m = get(m, "rng", "start_hashes") or {}
    rng_t = get(t, "rng", "start_hashes") or {}
    details["rng_start_hashes"] = {"M": rng_m, "T": rng_t}
    checks["rng_start_identical"] = bool(
        rng_m.get("combined_sha256") is not None
        and rng_m.get("combined_sha256") == rng_t.get("combined_sha256")
    )
    checks["rng_cpu_numpy_python_restored"] = bool(
        get(m, "gates", "rng_restored_from_checkpoint")
        and get(t, "gates", "rng_restored_from_checkpoint")
    )

    # First batch logits and labels.
    logits_m_path = preflight_m.parent / "first_batch_logits.pt"
    logits_t_path = preflight_t.parent / "first_batch_logits.pt"
    details["first_batch_logits_paths"] = {
        "M": C.track_rel(logits_m_path),
        "T": C.track_rel(logits_t_path),
    }
    if logits_m_path.exists() and logits_t_path.exists():
        pm = torch.load(logits_m_path, map_location="cpu", weights_only=False)
        pt = torch.load(logits_t_path, map_location="cpu", weights_only=False)
        diff_m = _tensor_max_diff(pm["logits_M"], pt["logits_M"])
        diff_t = _tensor_max_diff(pm["logits_T"], pt["logits_T"])
        details["first_batch_logits"] = {
            "max_abs_diff_M": diff_m,
            "max_abs_diff_T": diff_t,
            "rows_sha256_M": pm.get("rows_sha256"),
            "rows_sha256_T": pt.get("rows_sha256"),
            "labels_sha256_M": pm.get("labels_sha256"),
            "labels_sha256_T": pt.get("labels_sha256"),
        }
        checks["first_batch_logits_pair_equal_within_1e_5"] = bool(
            diff_m <= 1e-5 and diff_t <= 1e-5
        )
        checks["first_batch_rows_labels_identical"] = bool(
            pm.get("rows_sha256") == pt.get("rows_sha256")
            and pm.get("labels_sha256") == pt.get("labels_sha256")
        )
    else:
        checks["first_batch_logits_pair_equal_within_1e_5"] = False
        checks["first_batch_rows_labels_identical"] = False

    fb_m = m.get("first_batch") or {}
    fb_t = t.get("first_batch") or {}
    details["first_batch_reports"] = {"M": fb_m, "T": fb_t}
    checks["first_batch_rows_labels_same_in_json"] = bool(
        fb_m.get("rows_sha256") == fb_t.get("rows_sha256")
        and fb_m.get("labels_sha256") == fb_t.get("labels_sha256")
    )
    checks["M_loss_matches_historical"] = bool(
        get(m, "gates", "M_loss_exact_historical")
        and get(t, "gates", "M_loss_exact_historical")
    )
    checks["T_loss_matches_independent_numpy"] = bool(
        get(m, "gates", "T_loss_matches_independent_numpy")
        and get(t, "gates", "T_loss_matches_independent_numpy")
    )
    checks["T_loss_and_gradients_finite"] = bool(
        get(m, "gates", "T_loss_and_gradients_finite")
        and get(t, "gates", "T_loss_and_gradients_finite")
    )
    checks["task_weighting_really_changes_gradients"] = bool(
        get(m, "gates", "task_weighting_really_changes_gradients")
        and get(t, "gates", "task_weighting_really_changes_gradients")
    )
    checks["pinned_stager_equivalence"] = bool(
        get(m, "gates", "pinned_stager_equivalence")
        and get(t, "gates", "pinned_stager_equivalence")
    )
    checks["test_isolation"] = bool(
        get(m, "gates", "test_isolation") and get(t, "gates", "test_isolation")
    )
    checks["data_contract"] = bool(
        get(m, "gates", "data_contract") and get(t, "gates", "data_contract")
    )
    checks["fixed_recipe"] = bool(
        get(m, "gates", "fixed_recipe") and get(t, "gates", "fixed_recipe")
    )
    checks["paired_models_identical"] = bool(
        get(m, "gates", "paired_models_identical")
        and get(t, "gates", "paired_models_identical")
    )
    checks["paired_optimizers_identical"] = bool(
        get(m, "gates", "paired_optimizers_identical")
        and get(t, "gates", "paired_optimizers_identical")
    )
    checks["paired_dropout_logits_match"] = bool(
        get(m, "gates", "paired_dropout_logits_match")
        and get(t, "gates", "paired_dropout_logits_match")
    )
    checks["test_evaluated_false"] = bool(
        m.get("test_evaluated") is False and t.get("test_evaluated") is False
    )

    all_passed = bool(all(checks.values()))
    out = {
        "schema": "lcrc-pcba-loss-screen-paired-initialization-v1",
        "round_id": C.ROUND_ID,
        "phase": "B_paired_initialization_audit",
        "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "historical_valid_ap": C.SOURCE["best_valid_ap"],
        "preflight_M": C.track_rel(preflight_m),
        "preflight_T": C.track_rel(preflight_t),
        "checks": {k: bool(v) for k, v in checks.items()},
        "all_passed": all_passed,
        "details": details,
        "gates": {
            "paired_initialization_passed": all_passed,
            "formal_training_allowed": all_passed,
        },
        "created_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_path, out)
    C.log(f"paired initialization audit all_passed={all_passed} -> {C.track_rel(out_path)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-M", required=True)
    parser.add_argument("--preflight-T", required=True)
    parser.add_argument("--out", default=str(C.ANALYSIS / "paired_initialization.json"))
    args = parser.parse_args(argv)
    out = audit(
        preflight_m=Path(args.preflight_M),
        preflight_t=Path(args.preflight_T),
        out_path=Path(args.out),
    )
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
