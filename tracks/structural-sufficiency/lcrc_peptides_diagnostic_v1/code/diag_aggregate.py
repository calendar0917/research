"""Aggregate the diagnostic round into report tables and a resource ledger.

Facts only: every number is read from an artifact produced by the P0/P1/P2
scripts.  Nothing is trained or selected here, and no Test value is used.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

import diag_common as C

P2_RUN = C.ROUND / "runs/peptide_c1_seed0_240ep"


def read_csv(path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(label for label, _, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for row in rows:
        cells = []
        for _, key, fmt in columns:
            value = row.get(key)
            if value is None:
                cells.append("—")
            elif fmt == "":
                cells.append(str(value))
            else:
                cells.append(format(float(value), fmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    curve_summary = read_csv(C.ANALYSIS / "learning_curve_summary.csv")
    snapshots = read_csv(C.ANALYSIS / "learning_curves.csv")
    labels = read_csv(C.ANALYSIS / "label_statistics.csv")
    frozen = read_csv(C.RESULTS / "p0_frozen_checkpoints_summary.csv")
    frozen_labels = read_csv(C.RESULTS / "p0_frozen_checkpoints_per_label.csv")
    ecfp = read_csv(C.RESULTS / "ecfp_validation.csv")[0]
    ecfp_labels = read_csv(C.RESULTS / "ecfp_per_label.csv")
    manifest = C.read_json(C.ANALYSIS / "ecfp_manifest.json")
    row_identity = C.read_json(C.ANALYSIS / "row_identity.json")
    label_audit = C.read_json(C.ANALYSIS / "label_statistics.json")
    curve_audit = C.read_json(C.ANALYSIS / "learning_curve_audit.json")
    frozen_audit = C.read_json(C.ANALYSIS / "p0_frozen_checkpoints_audit.json")
    cpu_steps = C.read_json(C.ANALYSIS / "cpu_steps.json") if (C.ANALYSIS / "cpu_steps.json").exists() else None
    p2 = C.read_json(P2_RUN / "run.json") if (P2_RUN / "run.json").exists() else None

    # ---------------------------------------------------------------- comparison
    fast = {int(r["seed"]): float(r["best_valid_mean_ap"]) for r in curve_summary}
    fast_pct = {k: 100.0 * v for k, v in fast.items()}
    fast_vals = np.asarray([fast_pct[k] for k in sorted(fast_pct)], dtype=np.float64)
    comp: list[dict[str, Any]] = []
    for seed in sorted(fast_pct):
        row = next(r for r in curve_summary if int(r["seed"]) == seed)
        comp.append(
            {
                "method": "LCRC-C1 (Fast v1)",
                "seed": seed,
                "selection": "max Validation mean AP",
                "epochs_configured": int(next(r for r in read_csv(C.ANALYSIS / "history_runs.csv") if int(r["seed"]) == seed)["max_epochs"]),
                "cosine_horizon_epochs": 120,
                "best_epoch": int(row["best_epoch"]),
                "epochs_run": int(row["epochs_run"]),
                "valid_macro_ap_pct": fast_pct[seed],
                "note": "historical frozen run; copied curve + checkpoint hash verified",
            }
        )
    comp.append(
        {
            "method": "LCRC-C1 (Fast v1) mean±std",
            "seed": "0,1,2",
            "selection": "max Validation mean AP",
            "cosine_horizon_epochs": 120,
            "best_epoch": None,
            "epochs_run": 120,
            "valid_macro_ap_pct": float(fast_vals.mean()),
            "note": f"sample std {fast_vals.std(ddof=1):.2f} pp over 3 seeds",
        }
    )
    comp.append(
        {
            "method": "Count ECFP4 + LightGBM",
            "seed": 0,
            "selection": "fixed n_estimators=500, no early stopping",
            "cosine_horizon_epochs": None,
            "best_epoch": None,
            "epochs_run": 500,
            "valid_macro_ap_pct": 100.0 * float(ecfp["valid_macro_ap"]),
            "note": "Train-only fit, Validation evaluation; no Test access",
        }
    )
    comp.append(
        {
            "method": "constant-score baseline",
            "seed": None,
            "selection": "none",
            "cosine_horizon_epochs": None,
            "best_epoch": None,
            "epochs_run": None,
            "valid_macro_ap_pct": 100.0 * float(ecfp["valid_macro_constant_baseline_ap"]),
            "note": "mean Validation positive rate over the 10 labels",
        }
    )
    if p2:
        for name, payload, note in (
            ("LCRC-C1 long run (phase 1–120)", p2.get("phase_first_120"), "cosine horizon 240 (not 120)"),
            ("LCRC-C1 long run (phase 121–240)", p2.get("phase_121_240"), "cosine horizon 240"),
            ("LCRC-C1 long run (overall)", p2.get("overall_best"), "cosine horizon 240; best Validation checkpoint"),
        ):
            if payload is None:
                comp.append(
                    {
                        "method": name,
                        "seed": 0,
                        "selection": "max Validation mean AP",
                        "cosine_horizon_epochs": 240,
                        "best_epoch": None,
                        "epochs_run": p2.get("epochs_run"),
                        "valid_macro_ap_pct": None,
                        "note": f"MISSING_PHASE; {note}",
                    }
                )
                continue
            comp.append(
                {
                    "method": name,
                    "seed": 0,
                    "selection": "max Validation mean AP",
                    "cosine_horizon_epochs": 240,
                    "best_epoch": int(payload["best_epoch"]),
                    "epochs_run": int(p2.get("epochs_run")),
                    "valid_macro_ap_pct": 100.0 * float(payload["best_valid_macro_ap"]),
                    "note": note,
                }
            )
    write_csv(C.RESULTS / "validation_comparison.csv", comp)

    # ------------------------------------------------------- per-label comparison
    frozen_by_seed = {}
    for row in frozen_labels:
        frozen_by_seed.setdefault(int(row["seed"]), {})[int(row["label_index"])] = row
    ecfp_by_label = {int(row["label_index"]): row for row in ecfp_labels}
    label_comp = []
    for j, name in enumerate(C.LABELS):
        fast_aps = [float(frozen_by_seed[s][j]["valid_ap"]) for s in sorted(frozen_by_seed)]
        row = {
            "label_index": j,
            "label": name,
            "train_pos_rate": float(labels[j]["train_pos_rate"]),
            "valid_pos_rate": float(labels[j]["valid_pos_rate"]),
            "valid_constant_baseline_ap": float(labels[j]["valid_constant_ap_baseline"]),
            "fast_v1_valid_ap_seed0": fast_aps[0] if len(fast_aps) > 0 else None,
            "fast_v1_valid_ap_seed1": fast_aps[1] if len(fast_aps) > 1 else None,
            "fast_v1_valid_ap_seed2": fast_aps[2] if len(fast_aps) > 2 else None,
            "fast_v1_valid_ap_mean": float(np.mean(fast_aps)),
            "fast_v1_valid_ap_std": float(np.std(fast_aps, ddof=1)) if len(fast_aps) > 1 else None,
            "fast_v1_train_ap_seed0": float(frozen_by_seed[0][j]["train_ap"]),
            "ecfp_valid_ap": float(ecfp_by_label[j]["valid_ap"]),
            "ecfp_valid_ap_minus_valid_pos_rate": float(ecfp_by_label[j]["valid_ap_minus_valid_pos_rate"]),
        }
        if p2:
            row["long_run_valid_ap"] = p2["per_label_valid_ap"][name]
            row["long_run_train_ap"] = p2["per_label_train_ap"][name]
            row["long_run_valid_ap_minus_fast_v1_seed0"] = p2["per_label_valid_ap"][name] - row["fast_v1_valid_ap_seed0"]
        label_comp.append(row)
    write_csv(C.RESULTS / "label_ap_comparison.csv", label_comp)

    # ------------------------------------------------------------------- gates
    gates = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "result_label": C.RESULT_LABEL,
        "gates": {
            "history_curve_hashes_match_run_json": bool(curve_audit["checks"]["curve_sha256_matches_run_json"]),
            "history_checkpoint_hashes_match_run_json": bool(curve_audit["checks"]["checkpoint_sha256_matches_run_json"]),
            "history_best_epoch_reproduced_from_curve": bool(
                curve_audit["checks"]["argmax_curve_epoch_equals_recorded_best_epoch"]
            ),
            "split_sizes_10873_2331_2331": bool(label_audit["checks"]["split_sizes_10873_2331_2331"]),
            "split_indices_disjoint_and_complete": bool(label_audit["checks"]["split_indices_disjoint_and_complete"]),
            "labels_binary": bool(label_audit["checks"]["labels_binary"]),
            "train_labels_equal_fast_v1_npy": bool(label_audit["checks"]["train_labels_equal_fast_v1_npy"]),
            "valid_labels_equal_fast_v1_npy": bool(label_audit["checks"]["valid_labels_equal_fast_v1_npy"]),
            "test_labels_read_in_p0_2": bool(label_audit["checks"]["test_labels_read"]),
            "row_identity_status_pass": row_identity["status"] == "PASS",
            "row_identity_atom_feature_mismatches": sum(
                row_identity["per_split"][s]["n_atom_feature_mismatch_rows"] for s in C.SPLITS
            ),
            "row_identity_bond_feature_mismatches": sum(
                row_identity["per_split"][s]["n_bond_feature_mismatch_rows"] for s in C.SPLITS
            ),
            "row_identity_label_mismatches": sum(
                row_identity["per_split"][s]["n_label_mismatch_rows"] for s in C.SPLITS
            ),
            "row_identity_canonical_smiles_mismatches": sum(
                row_identity["per_split"][s]["n_canonical_smiles_mismatch_rows"] for s in C.SPLITS
            ),
            "frozen_checkpoint_eval_available_for_all_seeds": len(frozen) == 3
            and all(r["status"] == "CHECKPOINT_EVALUATED" for r in frozen),
            "frozen_checkpoint_valid_macro_ap_reproduces_stored": all(
                abs(float(r["valid_macro_ap"]) - float(r["stored_best_valid_macro_ap"])) < 1e-3 for r in frozen
            ),
            "frozen_checkpoint_helper_recheck_delta_zero": all(float(r["valid_macro_ap_helper_delta"]) < 1e-9 for r in frozen),
            "frozen_checkpoint_eval_read_no_test": frozen_audit["test_artifacts_opened"] == 0,
            "ecfp_features_shape_10873x2048_and_2331x2048": manifest["fingerprint"]["features"]["shape_ok"],
            "ecfp_fingerprint_purity_checks": all(manifest["fingerprint"]["purity"].values()),
            "ecfp_labels_equal_fast_v1_arrays": bool(
                manifest["checks"]["train_labels_equal_fast_v1"] and manifest["checks"]["valid_labels_equal_fast_v1"]
            ),
            "ecfp_test_rows_read": manifest["test_rows_read"],
            "ecfp_macro_ap_recheck_delta": manifest["metrics"]["macro_ap_recheck_delta"],
            "ecfp_as_configured_no_tuning": manifest["classifier"]["tuning"] == "none"
            and manifest["classifier"]["kwargs"]["n_estimators"] == 500,
        },
    }
    if p2:
        gates["gates"].update(
            {
                "p2_status": p2["status"],
                "p2_parameters_match_fast_v1": p2["structural_gate"]["parameters_match_history"],
                "p2_init_hash_match_fast_v1_seed0": p2["structural_gate"]["init_state_hash_match_history_seed0"],
                "p2_spec_and_norm_match_fast_v1": bool(
                    p2["structural_gate"]["spec_hash_match"] and p2["structural_gate"]["norm_fingerprint_match"]
                ),
                "p2_test_evaluated": p2["test_evaluated"],
                "p2_test_artifacts_opened": p2["guard"]["attempts"],
                "p2_batch_size": p2["batch_size"],
                "p2_reload_valid_delta": p2["reload_valid_delta"],
            }
        )
    C.write_json(C.ANALYSIS / "gates.json", gates)

    # --------------------------------------------------------------- resources
    ledger = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "cpu": {
            "pipeline": cpu_steps,
            "note": "local CPU workstation; P0 and P1 run without any GPU",
        },
        "gpu": None,
        "versions": {
            "repo_gpu_env": manifest.get("environment", {}),
            "ecfp_env": manifest["interpreter"]["versions"],
        },
        "hashes": {
            "smiles_csv_sha256": C.sha256_file(C.PEPTIDES_CSV),
            "splits_json_sha256": C.sha256_file(C.PEPTIDES_SPLITS),
            "train_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_train.npy"),
            "valid_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"),
            "cache_train_sha256": C.sha256_file(C.PREV_CACHE / "peptides_func_train.pt"),
            "cache_valid_sha256": C.sha256_file(C.PREV_CACHE / "peptides_func_valid.pt"),
            "history_checkpoints": {
                str(seed): C.sha256_file(C.history_checkpoint_path(seed)) for seed in C.LEGACY_SEEDS
            },
            "pyg_processed": row_identity["sources"]["pyg_processed"],
            "row_identity_split_index_sha256": {
                split: row_identity["per_split"][split]["split_index_sha256"] for split in C.SPLITS
            },
            "row_identity_digest_sha256": {
                split: row_identity["per_split"][split]["row_digest_sha256"] for split in C.SPLITS
            },
        },
    }
    if p2:
        ledger["gpu"] = {
            "run_id_note": "lcrc-pepdiag-p2-seed0 (see analysis/rr_run.json)",
            "host": p2["environment"].get("hostname"),
            "gpu_names": p2["environment"].get("gpu_names"),
            "torch": p2["environment"].get("torch"),
            "torch_cuda": p2["environment"].get("torch_cuda"),
            "seconds": p2["seconds"],
            "gpu_hours_estimate": p2["seconds"] / 3600.0,
            "epochs_run": p2["epochs_run"],
            "seconds_per_epoch": p2["seconds"] / max(p2["epochs_run"], 1),
            "gpu_peak_allocated_bytes": p2["gpu_peak_allocated_bytes"],
            "gpu_peak_reserved_bytes": p2["gpu_peak_reserved_bytes"],
            "checkpoint_sha256": p2["checkpoint_sha256"],
            "curve_sha256": p2["curve_sha256"],
            "git_commit": p2["environment"].get("git_commit"),
        }
    C.write_json(C.ANALYSIS / "resource_ledger.json", ledger)

    # ------------------------------------------------------------ report tables
    parts: list[str] = []
    parts.append("### T1 — label distribution (official Train/Validation)\n")
    parts.append(
        md_table(
            [
                {
                    "label": r["label"],
                    "train_pos": r["train_pos"],
                    "train_n": r["train_n"],
                    "train_rate": 100 * float(r["train_pos_rate"]),
                    "valid_pos": r["valid_pos"],
                    "valid_n": r["valid_n"],
                    "valid_rate": 100 * float(r["valid_pos_rate"]),
                }
                for r in labels
            ],
            [
                ("label", "label", ""),
                ("Train pos", "train_pos", ""),
                ("Train n", "train_n", ""),
                ("Train pos %", "train_rate", ".2f"),
                ("Valid pos", "valid_pos", ""),
                ("Valid n", "valid_n", ""),
                ("Valid pos % (= constant-AP baseline)", "valid_rate", ".2f"),
            ],
        )
    )
    parts.append("\n### T2 — learning-curve snapshots (Fast v1)\n")
    wide = {}
    for row in snapshots:
        key = int(row["epoch"])
        entry = wide.setdefault(key, {"epoch": key})
        entry[f"loss_s{row['seed']}"] = row["train_loss"]
        entry[f"ap_s{row['seed']}"] = 100 * float(row["valid_mean_ap"]) if row["valid_mean_ap"] else None
    parts.append(
        md_table(
            [wide[k] for k in sorted(wide)],
            [("epoch", "epoch", "")]
            + [(f"seed{s} loss", f"loss_s{s}", ".5f") for s in C.LEGACY_SEEDS]
            + [(f"seed{s} Valid AP %", f"ap_s{s}", ".2f") for s in C.LEGACY_SEEDS],
        )
    )
    parts.append("\n### T3 — learning-curve summary (Fast v1)\n")
    parts.append(
        md_table(
            [
                {
                    "seed": r["seed"],
                    "best_epoch": r["best_epoch"],
                    "epochs_run": r["epochs_run"],
                    "best_ap": 100 * float(r["best_valid_mean_ap"]),
                    "ap_100": 100 * float(r["valid_mean_ap_epoch_100"]),
                    "ap_last": 100 * float(r["valid_mean_ap_last"]),
                    "delta_tail": 100 * float(r["valid_mean_ap_delta_last20"]),
                    "new_best": r["new_best_in_last20"],
                    "loss_100": r["train_loss_epoch_100"],
                    "loss_last": r["train_loss_last"],
                    "delta_loss_tail": r["train_loss_delta_last20"],
                }
                for r in curve_summary
            ],
            [
                ("seed", "seed", ""),
                ("best epoch", "best_epoch", ""),
                ("epochs run", "epochs_run", ""),
                ("best Valid AP %", "best_ap", ".2f"),
                ("Valid AP @100 %", "ap_100", ".2f"),
                ("Valid AP @last %", "ap_last", ".2f"),
                ("Δ Valid AP last 20 (pp)", "delta_tail", "+.2f"),
                ("new best in last 20", "new_best", ""),
                ("train loss @100", "loss_100", ".5f"),
                ("train loss @last", "loss_last", ".5f"),
                ("Δ train loss last 20", "delta_loss_tail", "+.5f"),
            ],
        )
    )
    parts.append("\n### T4 — per-label Train/Validation AP of the frozen Fast v1 checkpoints\n")
    t4 = []
    for j, name in enumerate(C.LABELS):
        entry = {"label": name}
        for seed in sorted(frozen_by_seed):
            entry[f"train_{seed}"] = 100 * float(frozen_by_seed[seed][j]["train_ap"])
            entry[f"valid_{seed}"] = 100 * float(frozen_by_seed[seed][j]["valid_ap"])
        entry["valid_mean"] = float(
            np.mean([100 * float(frozen_by_seed[seed][j]["valid_ap"]) for seed in sorted(frozen_by_seed)])
        )
        entry["valid_pos_rate"] = 100 * float(frozen_by_seed[0][j]["valid_pos_rate"])
        entry["valid_minus_rate"] = float(
            np.mean(
                [
                    100 * float(frozen_by_seed[seed][j]["valid_ap_minus_valid_pos_rate"])
                    for seed in sorted(frozen_by_seed)
                ]
            )
        )
        t4.append(entry)
    parts.append(
        md_table(
            t4,
            [("label", "label", ""), ("Valid pos % (baseline)", "valid_pos_rate", ".2f")]
            + [(f"seed{seed} Train AP %", f"train_{seed}", ".2f") for seed in sorted(frozen_by_seed)]
            + [(f"seed{seed} Valid AP %", f"valid_{seed}", ".2f") for seed in sorted(frozen_by_seed)]
            + [("Valid AP mean %", "valid_mean", ".2f"), ("mean Valid AP − pos rate (pp)", "valid_minus_rate", "+.2f")],
        )
    )
    parts.append("\n### T5 — ECFP4 + LightGBM per-label Validation AP\n")
    parts.append(
        md_table(
            [
                {
                    "label": r["label"],
                    "train_rate": 100 * float(r["train_pos_rate"]),
                    "valid_rate": 100 * float(r["valid_pos_rate"]),
                    "ap": 100 * float(r["valid_ap"]),
                    "delta": 100 * float(r["valid_ap_minus_valid_pos_rate"]),
                }
                for r in ecfp_labels
            ],
            [
                ("label", "label", ""),
                ("Train pos %", "train_rate", ".2f"),
                ("Valid pos % (baseline)", "valid_rate", ".2f"),
                ("Valid AP %", "ap", ".2f"),
                ("AP − baseline (pp)", "delta", "+.2f"),
            ],
        )
    )
    parts.append("\n### T6 — Validation comparison (this round)\n")
    parts.append(
        md_table(
            comp,
            [
                ("method", "method", ""),
                ("seed", "seed", ""),
                ("cosine horizon", "cosine_horizon_epochs", ""),
                ("best epoch", "best_epoch", ""),
                ("epochs run", "epochs_run", ""),
                ("Valid macro AP %", "valid_macro_ap_pct", ".2f"),
                ("note", "note", ""),
            ],
        )
    )
    parts.append("\n### T7 — per-label Validation AP comparison\n")
    label_comp_pct = []
    for row in label_comp:
        out = dict(row)
        for key, value in row.items():
            if isinstance(value, float) and key != "label_index":
                out[key] = 100.0 * value
        label_comp_pct.append(out)
    parts.append(
        md_table(
            label_comp_pct,
            [
                ("label", "label", ""),
                ("Valid pos % (baseline)", "valid_pos_rate", ".2f"),
                ("Fast v1 seed0 %", "fast_v1_valid_ap_seed0", ".2f"),
                ("Fast v1 seed1 %", "fast_v1_valid_ap_seed1", ".2f"),
                ("Fast v1 seed2 %", "fast_v1_valid_ap_seed2", ".2f"),
                ("ECFP4+LGBM %", "ecfp_valid_ap", ".2f"),
                ("long run %", "long_run_valid_ap", ".2f"),
            ],
        )
    )
    (C.ANALYSIS / "report_tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    if p2:
        C.log(
            f"P2 overall best {100*p2['best_valid_macro_ap']:.2f}% at epoch {p2['best_epoch']}; "
            f"delta vs Fast v1 seed0 {p2['delta_vs_fast_v1_seed0_pp']:+.2f} pp"
        )
    else:
        C.log("P2 artifacts missing: comparison table marks the long run as MISSING")
    C.log(f"wrote {C.track_rel(C.RESULTS / 'validation_comparison.csv')} and {C.track_rel(C.ANALYSIS / 'report_tables.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
