"""Terminal, reporting-only official-test evaluation for LCRC-v1.

By default this is ONLY run after the valid-side decision is frozen and the
pre-registered test gate in PROTOCOL.md is met.  ``--allow-any-arm`` exists only
for a documented, user-requested post-hoc descriptive read; it records that the
gate was not met and that the read is not confirmatory.  It:

1. writes ``analysis/test_freeze.json`` (selected run/checkpoint/config/code
   hashes, recomputed valid MAE) *before* touching test;
2. builds the label-free test feature cache from the official PyG test graphs;
3. reads the official test labels exactly once and evaluates the frozen
   best-valid checkpoint;
4. writes descriptive predictions/metrics.

The official test split is already spent in this repository; these numbers are
a descriptive performance record, not a new blind test, and must not drive any
model/seed/epoch choice.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

import lc_common as L
import lc_data as D
import lc_models as M
import lc_train as T

HISTORICAL_DCC_L_TEST_4SEED = 0.067319
HISTORICAL_DCC_L_S0_TEST = 0.066267


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="e.g. lcrcR_s0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="only for a documented rerun")
    parser.add_argument("--allow-any-arm", action="store_true",
                        help="documented post-hoc read of a non-R checkpoint; not confirmatory")
    parser.add_argument("--rationale", default="",
                        help="why this checkpoint was selected for a post-hoc read")
    args = parser.parse_args(argv)

    freeze_path = L.ANALYSIS / "test_freeze.json"
    report_path = L.ANALYSIS / "test_report.json"
    if report_path.exists() and not args.force:
        raise SystemExit(f"{report_path} exists; the terminal test was already read once")
    run_dir = L.RUNS / args.run
    info = L.read_json(run_dir / "train.json")
    arm = info["arm"]
    seed = int(info["seed"])
    if arm != "R" and not args.allow_any_arm:
        raise SystemExit("the pre-registered terminal candidate must be an R run; "
                         "use --allow-any-arm only for a documented post-hoc read")

    norm = D.load_norm()
    valid = D.load_cache("valid")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    state = best["state_dict"]
    model = M.load_model_from_state(arm, norm, state, args.device)
    rows_valid = np.arange(len(valid["labels"]))
    _, valid_mae = T.evaluate(model, valid, rows_valid, device=args.device, batch_size=256)
    if abs(valid_mae - float(info["best"]["valid_mae"])) > 1e-6:
        raise RuntimeError(f"valid replay mismatch {valid_mae} vs {info['best']['valid_mae']}")
    gate = {}
    gate_path = L.ANALYSIS / "seed_summary.json"
    if gate_path.exists():
        gate = L.read_json(gate_path).get("test_gate", {})
    protocol_compliant = bool(arm == "R" and gate.get("pass", False))
    freeze = {
        "round": L.ROUND_ID,
        "protocol_gate": gate,
        "protocol_compliant": protocol_compliant,
        "user_override_post_hoc": not protocol_compliant,
        "selection_rationale": args.rationale,
        "selected_run": args.run,
        "arm": arm,
        "seed": seed,
        "best_epoch": int(info["best"]["epoch"]),
        "valid_mae_recomputed": float(valid_mae),
        "valid_mae_recorded": float(info["best"]["valid_mae"]),
        "checkpoint_path": L.track_rel(run_dir / "best.pt"),
        "checkpoint_sha256": L.C.sha256_file(run_dir / "best.pt"),
        "config_path": L.track_rel(L.CONFIGS / "lcrc_v1.json"),
        "config_sha256": L.C.sha256_file(L.CONFIGS / "lcrc_v1.json"),
        "git_commit": L.C.git_commit(),
        "official_test_loaded_before_freeze": False,
        "created": L.now(),
    }
    L.write_json(freeze_path, freeze)
    L.log(f"freeze written before test access: {L.track_rel(freeze_path)}")

    # Label-free test graph features (P3 objects rebuilt from official PyG).
    test = D.build_split("test", with_labels=False)
    D.save_cache(test, "test")
    test = D.load_cache("test")
    rows = np.arange(test["n_molecules"])
    preds = []
    model.eval()
    with torch.no_grad(), L.C.isolated_rng():
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                     for k, v in D.collate(test, chunk).items()}
            preds.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds, 0)
    # single terminal label read
    labels = L.D.official_labels("test")
    if len(labels) != len(pred):
        raise RuntimeError("test label/prediction length mismatch")
    ae = np.abs(pred.astype(np.float64) - labels.astype(np.float64))
    test_mae = float(ae.mean())
    ordinary = labels > -3.0
    tail = labels <= -3.0
    order = np.argsort(ae)[::-1]
    report = {
        "round": L.ROUND_ID,
        "descriptive_only": True,
        "test_is_spent": True,
        "selected_run": args.run,
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "valid_mae": float(valid_mae),
        "test_mae_raw_y": test_mae,
        "test_rmse": float(np.sqrt(np.mean((pred - labels) ** 2))),
        "test_p90_ae": float(np.percentile(ae, 90)),
        "test_max_ae": float(ae.max()),
        "test_mae_ordinary_y_gt_-3": float(ae[ordinary].mean()),
        "test_n_ordinary_y_gt_-3": int(ordinary.sum()),
        "test_mae_tail_y_le_-3": float(ae[tail].mean()),
        "test_n_tail_y_le_-3": int(tail.sum()),
        "test_top10_ae_rows": [int(i) for i in order[:10]],
        "test_top10_ae_values": [float(ae[i]) for i in order[:10]],
        "historical_anchors_not_retrained": {
            "DCC_L_test_4seed_mean": HISTORICAL_DCC_L_TEST_4SEED,
            "DCC_L_s0_test": HISTORICAL_DCC_L_S0_TEST,
        },
        "read_count": 1,
        "git_commit": L.C.git_commit(),
        "created": L.now(),
    }
    L.write_json(report_path, report)
    L.ensure_dir(L.PREDICTIONS)
    np.savez(L.PREDICTIONS / f"test_{args.run}.npz", row=rows, pred=pred)
    with open(L.ANALYSIS / f"test_per_molecule_{args.run}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "y_raw", "pred", "signed_error", "abs_error"])
        for i in range(len(rows)):
            w.writerow([i, labels[i], pred[i], pred[i] - labels[i], ae[i]])
    L.log(f"terminal test MAE (descriptive): {test_mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
