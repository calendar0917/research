"""Local paired audit: refuse formal training unless both preflights agree.

Writes ``analysis/initialization_audit.json`` and
``analysis/paired_training_order.json``.  This is intentionally a *local* step:
the two GPU jobs cannot see each other's fingerprint/order record.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import lr_common as C  # noqa: E402


def _order_detail(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    order = run.get("training_order")
    if isinstance(order, Mapping) and "epochs_detail" in order:
        return order
    return None


def build_initialization_audit(
    h: Mapping[str, Any],
    low: Mapping[str, Any],
    *,
    run_h: Mapping[str, Any] | None = None,
    run_l: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hi = h.get("initialization", {})
    li = low.get("initialization", {})
    he0 = h.get("epoch0_validation", {})
    le0 = low.get("epoch0_validation", {})
    hc = h.get("historical_checkpoint", {})
    lc = low.get("historical_checkpoint", {})
    h_order = h.get("training_order", {})
    l_order = low.get("training_order", {})
    h_iso = h.get("training_order_isolation", {})
    l_iso = low.get("training_order_isolation", {})

    checks = {
        "both_preflights_passed": bool(h.get("all_passed")) and bool(low.get("all_passed")),
        "same_source_checkpoint_sha256": hc.get("sha256") == lc.get("sha256") == C.HIST["checkpoint_sha256"],
        "same_parameter_fingerprint": bool(
            hi.get("loaded_parameter_fingerprint")
            and hi.get("loaded_parameter_fingerprint") == li.get("loaded_parameter_fingerprint")
        ),
        "both_loaded_state_bitwise_equal_to_history": bool(
            hi.get("state_dict_bitwise_equal") and li.get("state_dict_bitwise_equal")
        ),
        "same_model_config": bool(
            h.get("inheritance", {}).get("model_config")
            == low.get("inheritance", {}).get("model_config")
            == C.PC.MODEL_CONFIG
        ),
        "same_feature_spec_hash": h.get("data_contract", {}).get("feature_spec_hash")
        == low.get("data_contract", {}).get("feature_spec_hash")
        == C.HIST["feature_spec_hash"],
        "same_train_only_norm_fingerprint": h.get("data_contract", {}).get("norm_fingerprint")
        == low.get("data_contract", {}).get("norm_fingerprint")
        == C.HIST["norm_fingerprint"],
        "both_epoch0_reproduce_historical_ap": bool(
            he0.get("reproduces_historical") and le0.get("reproduces_historical")
        ),
        "epoch0_ap_pair_delta_within_1e_minus_6": bool(
            he0.get("ap") is not None
            and le0.get("ap") is not None
            and abs(float(he0["ap"]) - float(le0["ap"])) <= 1e-6
        ),
        "both_orders_deterministic": bool(
            h.get("training_order_repeat_equal") and low.get("training_order_repeat_equal")
        ),
        "same_order_plan": h_order.get("global_plan_sha256") == l_order.get("global_plan_sha256"),
        "both_order_isolation_passed": bool(h_iso.get("passed") and l_iso.get("passed")),
        "both_pinned_stager_smoke_passed": bool(
            h.get("pinned_stager_smoke", {}).get("passed")
            and low.get("pinned_stager_smoke", {}).get("passed")
        ),
        "both_test_isolation_passed": bool(
            h.get("test_isolation", {}).get("test_rows_all_nan")
            and low.get("test_isolation", {}).get("test_rows_all_nan")
        ),
        "both_fixed_recipe": bool(
            all(h.get("recipe_checks", {}).values()) and all(low.get("recipe_checks", {}).values())
        ),
    }
    if run_h is not None and run_l is not None:
        checks["formal_order_matches_preflight"] = bool(
            run_h.get("training_order", {}).get("global_plan_sha256")
            == h_order.get("global_plan_sha256")
            and run_l.get("training_order", {}).get("global_plan_sha256")
            == l_order.get("global_plan_sha256")
        )
        checks["formal_initialization_fingerprint_matches_preflight"] = bool(
            run_h.get("initialization", {}).get("loaded_parameter_fingerprint")
            == hi.get("loaded_parameter_fingerprint")
            and run_l.get("initialization", {}).get("loaded_parameter_fingerprint")
            == li.get("loaded_parameter_fingerprint")
        )
    return {
        "schema": "lcrc-pcba-lr-probe-initialization-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "historical": {
            "round": C.HIST["round"],
            "checkpoint": C.track_rel(C.HIST["checkpoint"]),
            "checkpoint_sha256": C.HIST["checkpoint_sha256"],
            "best_epoch": C.HIST["best_epoch"],
            "best_valid_ap": C.HIST["best_valid_ap"],
            "feature_spec_hash": C.HIST["feature_spec_hash"],
            "norm_fingerprint": C.HIST["norm_fingerprint"],
            "split_partition_sha256": C.HIST["split_partition_sha256"],
            "label_artifact_sha256": C.HIST["label_artifact_sha256"],
        },
        "groups": {
            "H": {
                "high_lr": C.GROUPS["H"]["learning_rate"],
                "source_checkpoint_sha256": hc.get("sha256"),
                "loaded_parameter_fingerprint": hi.get("loaded_parameter_fingerprint"),
                "epoch0_valid_ap": he0.get("ap"),
                "epoch0_abs_delta_vs_historical": he0.get("abs_delta_vs_historical"),
                "order_plan_sha256": h_order.get("global_plan_sha256"),
                "device": h.get("device"),
                "assigned_pool": h.get("assigned_pool"),
                "gpu_seconds": h.get("gpu_seconds"),
            },
            "L": {
                "low_lr": C.GROUPS["L"]["learning_rate"],
                "source_checkpoint_sha256": lc.get("sha256"),
                "loaded_parameter_fingerprint": li.get("loaded_parameter_fingerprint"),
                "epoch0_valid_ap": le0.get("ap"),
                "epoch0_abs_delta_vs_historical": le0.get("abs_delta_vs_historical"),
                "order_plan_sha256": l_order.get("global_plan_sha256"),
                "device": low.get("device"),
                "assigned_pool": low.get("assigned_pool"),
                "gpu_seconds": low.get("gpu_seconds"),
            },
        },
        "epoch0_pair_delta": (
            None
            if he0.get("ap") is None or le0.get("ap") is None
            else float(he0["ap"]) - float(le0["ap"])
        ),
        "checks": checks,
        "passed": bool(all(bool(v) for v in checks.values())),
        "limitations": [
            "AdamW moments and step count are absent from the historical checkpoint; both arms re-initialise AdamW from zero.",
            "This audit proves identical starting weights, not identical historical optimizer state.",
            "Official Test targets were NaN in the label artifact and were never evaluated.",
        ],
    }


def build_paired_order(
    h: Mapping[str, Any],
    low: Mapping[str, Any],
    *,
    run_h: Mapping[str, Any] | None = None,
    run_l: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    h_order = _order_detail(h)
    l_order = _order_detail(low)
    if h_order is None or l_order is None:
        raise RuntimeError("preflight JSON is missing training_order.epochs_detail")
    hd = h_order["epochs_detail"]
    ld = l_order["epochs_detail"]
    per_epoch = []
    for a, b in zip(hd, ld):
        per_epoch.append(
            {
                "epoch": int(a["epoch"]),
                "H_row_order_sha256": a["row_order_sha256"],
                "L_row_order_sha256": b["row_order_sha256"],
                "row_order_equal": a["row_order_sha256"] == b["row_order_sha256"],
                "H_batch_plan_sha256": a["batch_plan_sha256"],
                "L_batch_plan_sha256": b["batch_plan_sha256"],
                "batch_plan_equal": a["batch_plan_sha256"] == b["batch_plan_sha256"],
                "first_batch_equal": a["first_batch_rows"] == b["first_batch_rows"],
                "n_batches_equal": int(a["n_batches"]) == int(b["n_batches"]),
                "first_row_equal": int(a["first_row"]) == int(b["first_row"]),
                "last_row_equal": int(a["last_row"]) == int(b["last_row"]),
            }
        )
    checks = {
        "both_preflights_passed": bool(h.get("all_passed")) and bool(low.get("all_passed")),
        "same_global_plan_sha256": h_order.get("global_plan_sha256")
        == l_order.get("global_plan_sha256"),
        "same_generator_state_sha256": h_order.get("generator_state_sha256")
        == l_order.get("generator_state_sha256"),
        "same_epochs": int(h_order.get("epochs", -1)) == int(l_order.get("epochs", -2)) == C.TOTAL_EPOCHS,
        "same_steps_per_epoch": int(h_order.get("steps_per_epoch", -1))
        == int(l_order.get("steps_per_epoch", -2))
        == C.STEPS_PER_EPOCH,
        "every_epoch_row_order_equal": all(x["row_order_equal"] for x in per_epoch),
        "every_epoch_batch_plan_equal": all(x["batch_plan_equal"] for x in per_epoch),
        "every_epoch_first_batch_equal": all(x["first_batch_equal"] for x in per_epoch),
        "H_no_valid_or_test_rows": bool(h.get("training_order_isolation", {}).get("passed")),
        "L_no_valid_or_test_rows": bool(low.get("training_order_isolation", {}).get("passed")),
    }
    if run_h is not None and run_l is not None:
        checks["formal_run_H_matches_preflight"] = (
            run_h.get("training_order", {}).get("global_plan_sha256")
            == h_order.get("global_plan_sha256")
        )
        checks["formal_run_L_matches_preflight"] = (
            run_l.get("training_order", {}).get("global_plan_sha256")
            == l_order.get("global_plan_sha256")
        )
        checks["formal_run_orders_equal"] = (
            run_h.get("training_order", {}).get("global_plan_sha256")
            == run_l.get("training_order", {}).get("global_plan_sha256")
        )
    return {
        "schema": "lcrc-pcba-lr-probe-paired-order-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "seed": C.SEED,
        "batch_size": C.BATCH_SIZE,
        "n_train_rows": int(h_order.get("n_train_rows", -1)),
        "steps_per_epoch": int(h_order.get("steps_per_epoch", -1)),
        "epochs": int(h_order.get("epochs", -1)),
        "source": h_order.get("source"),
        "H_global_plan_sha256": h_order.get("global_plan_sha256"),
        "L_global_plan_sha256": l_order.get("global_plan_sha256"),
        "generator_state_sha256": h_order.get("generator_state_sha256"),
        "per_epoch": per_epoch,
        "checks": checks,
        "passed": bool(all(bool(v) for v in checks.values())),
        "limitations": [
            "The order plan is generated from the official train rows and seed 0 only; it is not a re-shuffle per arm.",
            "No Validation or Test row is ever in a training batch.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-h", default=str(C.ANALYSIS / "preflight_H.json"))
    parser.add_argument("--preflight-l", default=str(C.ANALYSIS / "preflight_L.json"))
    parser.add_argument("--run-h", default="")
    parser.add_argument("--run-l", default="")
    parser.add_argument("--init-out", default=str(C.ANALYSIS / "initialization_audit.json"))
    parser.add_argument("--order-out", default=str(C.ANALYSIS / "paired_training_order.json"))
    args = parser.parse_args(argv)
    h = C.read_json(args.preflight_h)
    low = C.read_json(args.preflight_l)
    run_h = C.read_json(args.run_h) if args.run_h else None
    run_l = C.read_json(args.run_l) if args.run_l else None
    init = build_initialization_audit(h, low, run_h=run_h, run_l=run_l)
    order = build_paired_order(h, low, run_h=run_h, run_l=run_l)
    C.write_json(args.init_out, init)
    C.write_json(args.order_out, order)
    C.log(f"initialization passed={init['passed']} checks={init['checks']}")
    C.log(f"paired order passed={order['passed']} checks={order['checks']}")
    return 0 if init["passed"] and order["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
