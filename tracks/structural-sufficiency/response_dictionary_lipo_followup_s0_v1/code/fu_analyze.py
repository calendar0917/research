"""Aggregate the JOINT unit, the frozen dictionary stress test and the edit replay.

    uv run python code/fu_analyze.py --out <round dir>

Produces ``main_table.csv``, ``joint_gain.csv``, ``bootstrap_joint.csv``,
``corrections.csv``, ``rmse_curves.png`` and ``summary.json``.  Nothing here
retrains, re-selects or adds a seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fu_common as FC  # noqa: E402
import rd_analyze as RA  # noqa: E402
import rd_common as RC  # noqa: E402

TASK = "lipo"
ARMS = ("RESP", "ADD", "JOINT")
STATES = ("best", "final", "last10")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(FC.ROUND_DIR))
    parser.add_argument("--joint-results", default=str(FC.run_dir(TASK, "JOINT")))
    parser.add_argument("--stress-dir", default=str(FC.run_dir(TASK, "RESP") / "dictionary_stress_k16"))
    parser.add_argument("--edit-support", default=str(FC.ROUND_DIR / "analysis" / "edit_support.json"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    follow = FC.load_config()

    units: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        root = Path(args.joint_results).parent if arm == "JOINT" else FC.S0_RESULTS
        unit = RA.load_unit(root, TASK, arm)
        if unit is None:
            raise SystemExit(f"missing unit {arm}")
        units[arm] = unit

    main_rows = _main_table(units)
    _write_csv(out / "main_table.csv", main_rows)

    gain_rows, bootstrap, influence = _joint_comparison(units, follow)
    _write_csv(out / "joint_gain.csv", gain_rows)
    _write_csv(out / "bootstrap_joint.csv", [bootstrap])
    RC.write_json(out / "influence_joint.json", influence)

    corrections = _corrections(units, Path(args.stress_dir))
    _write_csv(out / "corrections.csv", corrections["same_epoch_train_rmse"])
    RC.write_json(out / "corrections.json", corrections)
    matched = _matched_epochs(units)
    _write_csv(out / "matched_epochs.csv", matched["rows"])
    RC.write_json(out / "matched_epochs.json", matched)

    stress = _read_json(Path(args.stress_dir) / "stress_summary.json") if (Path(args.stress_dir) / "stress_summary.json").is_file() else {}
    edits = _read_json(Path(args.edit_support)) if Path(args.edit_support).is_file() else {}
    labels = _labels(gain_rows, bootstrap, influence, stress, edits, follow)
    plot = _plot(units, out)

    summary = {
        "round_id": follow["round_id"],
        "parent_round": follow["parent_round"],
        "protocol_hash": FC.protocol_hash(),
        "commit": RC.git_commit(),
        "test_use": follow["test_use"],
        "main_table": main_rows,
        "joint_gain": gain_rows,
        "bootstrap_joint": bootstrap,
        "influence_joint": influence,
        "corrections": corrections,
        "matched_epochs": matched,
        "dictionary_stress": {
            "state": stress.get("decision", {}).get("label", "NOT_RUN"),
            "table": stress.get("table", []),
            "subset_table": stress.get("subset_table", []),
            "view_groups": stress.get("view_groups", []),
            "counts_note": "see results/lipo_RESP/dictionary_stress_k16/stress_counts.csv",
            "decision": stress.get("decision", {}),
            "alpha": stress.get("alpha", {}),
            "response_dimensions": stress.get("response_dimensions", {}),
            "panel_intervention": stress.get("panel_intervention", {}),
            "replay": stress.get("replay", {}),
        },
        "edits": edits,
        "labels": labels,
        "curves_png": str(plot) if plot else None,
    }
    RC.write_json(out / "summary.json", summary)
    print(json.dumps(RC.jsonable({"labels": labels, "gain": gain_rows,
                                 "bootstrap": bootstrap, "influence": influence}), indent=2))
    return 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _main_table(units) -> list[dict[str, Any]]:
    rows = []
    for arm, unit in units.items():
        curves = unit["curves"]
        last10 = curves[-10:]
        for state in STATES:
            if state == "best":
                epoch = int(unit["best"]["epoch"])
                metrics = unit["best_metrics"]
            elif state == "final":
                epoch = len(curves)
                metrics = unit["final_metrics"]
            else:
                epoch = -1
                metrics = {f"{split}_{key}": float(np.mean([r[f"{split}_{key}"] for r in last10]))
                           for split in ("train", "valid", "test") for key in ("rmse_raw", "mae_raw")}
            rows.append({
                "task": TASK, "arm": arm, "state": state, "selected_epoch": epoch,
                "n_parameters": int(unit["n_parameters"]),
                "train_rmse_raw": float(metrics["train_rmse_raw"]),
                "valid_rmse_raw": float(metrics["valid_rmse_raw"]),
                "test_rmse_raw": float(metrics["test_rmse_raw"]),
                "train_mae_raw": float(metrics["train_mae_raw"]),
                "valid_mae_raw": float(metrics["valid_mae_raw"]),
                "test_mae_raw": float(metrics["test_mae_raw"]),
                "seconds": float(unit["seconds"]),
                "init_state_hash": unit["init_state_hash"],
                "view_dispersion_valid_raw": float(unit["diagnostics"]["valid"]["view_dispersion_raw"]),
                "fallback_valid_fraction": float(unit["diagnostics"]["valid"]["fallback_molecule_fraction"]),
                "horizon_limited": bool(unit["horizon"]["horizon_limited"]),
            })
    return rows


def _state_of(unit, state: str) -> dict[str, float]:
    curves = unit["curves"]
    if state == "best":
        return unit["best_metrics"]
    if state == "final":
        return unit["final_metrics"]
    last10 = curves[-10:]
    return {f"{split}_{key}": float(np.mean([r[f"{split}_{key}"] for r in last10]))
            for split in ("train", "valid", "test") for key in ("rmse_raw", "mae_raw")}


def _joint_comparison(units, follow) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = []
    for state in STATES:
        resp = _state_of(units["RESP"], state)
        joint = _state_of(units["JOINT"], state)
        add = _state_of(units["ADD"], state)
        rows.append({
            "task": TASK, "state": state,
            "valid_rmse_RESP": resp["valid_rmse_raw"], "valid_rmse_JOINT": joint["valid_rmse_raw"],
            "valid_rmse_ADD": add["valid_rmse_raw"],
            "gain_val_JOINT_minus_RESP": joint["valid_rmse_raw"] - resp["valid_rmse_raw"],
            "gain_test_JOINT_minus_RESP": joint["test_rmse_raw"] - resp["test_rmse_raw"],
            "gain_val_JOINT_minus_ADD": joint["valid_rmse_raw"] - add["valid_rmse_raw"],
            "gain_test_JOINT_minus_ADD": joint["test_rmse_raw"] - add["test_rmse_raw"],
            "gain_val_ADD_minus_RESP": add["valid_rmse_raw"] - resp["valid_rmse_raw"],
            "gain_test_ADD_minus_RESP": add["test_rmse_raw"] - resp["test_rmse_raw"],
        })
    best = next(r for r in rows if r["state"] == "best")
    bootstrap = _paired_bootstrap(units["RESP"]["predictions_best"], units["JOINT"]["predictions_best"],
                                  n=int(follow["bootstrap"]["n"]), seed=int(follow["bootstrap"]["seed"]))
    influence = _influence(units["RESP"]["predictions_best"], units["JOINT"]["predictions_best"])
    bootstrap["best_val_point"] = best["gain_val_JOINT_minus_RESP"]
    return rows, bootstrap, influence


def _paired_bootstrap(resp, joint, n: int, seed: int) -> dict[str, Any]:
    target = np.asarray(resp["valid_target_raw"], dtype=np.float64)
    r = np.asarray(resp["valid_pred_raw"], dtype=np.float64)
    j = np.asarray(joint["valid_pred_raw"], dtype=np.float64)
    if not (np.array_equal(resp["valid_row_ids"], joint["valid_row_ids"])
            and np.array_equal(target, joint["valid_target_raw"])):
        raise AssertionError("paired Val rows are not aligned")
    se_r = (r - target) ** 2
    se_j = (j - target) ** 2
    point = math.sqrt(float(se_j.mean())) - math.sqrt(float(se_r.mean()))
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n))
    for i in range(int(n)):
        idx = rng.integers(0, target.size, size=target.size)
        diffs[i] = math.sqrt(float(se_j[idx].mean())) - math.sqrt(float(se_r[idx].mean()))
    return {
        "n_bootstrap": int(n), "seed": int(seed),
        "metric": "RMSE_JOINT - RMSE_RESP (raw units, Val molecules resampled with replacement)",
        "point": point, "mean": float(diffs.mean()),
        "interval_2.5": float(np.quantile(diffs, 0.025)),
        "interval_97.5": float(np.quantile(diffs, 0.975)),
        "share_positive_RESP_better": float((diffs > 0).mean()),
        "note": ("Molecule-row resampling at the two fixed best-Val training states; it describes sample "
                 "uncertainty, not training-seed variability."),
    }


def _influence(resp, joint) -> dict[str, Any]:
    target = np.asarray(resp["valid_target_raw"], dtype=np.float64)
    se_r = (np.asarray(resp["valid_pred_raw"], dtype=np.float64) - target) ** 2
    se_j = (np.asarray(joint["valid_pred_raw"], dtype=np.float64) - target) ** 2
    gain = se_j - se_r
    row_ids = np.asarray(resp["valid_row_ids"])
    out: dict[str, Any] = {}
    for k in (1, 5):
        keep = np.ones(target.size, dtype=bool)
        order = np.argsort(-gain)
        keep[order[:k]] = False
        out[f"gain_after_dropping_top{k}_val"] = math.sqrt(float(se_j[keep].mean())) - math.sqrt(float(se_r[keep].mean()))
        out[f"dropped_rows_top{k}"] = row_ids[order[:k]].tolist()
        out[f"dropped_contributions_top{k}"] = [float(x) for x in gain[order[:k]]]
    out["gain_val"] = math.sqrt(float(se_j.mean())) - math.sqrt(float(se_r.mean()))
    out["largest_single_molecule_contribution"] = float(gain.max())
    out["share_of_total_val_gain_top1"] = float(gain.max() / gain.sum()) if gain.sum() != 0 else None
    # Test side is descriptive only
    tgt = np.asarray(resp["test_target_raw"], dtype=np.float64)
    tr = (np.asarray(resp["test_pred_raw"], dtype=np.float64) - tgt) ** 2
    tj = (np.asarray(joint["test_pred_raw"], dtype=np.float64) - tgt) ** 2
    tgain = tj - tr
    keep = np.ones(tgt.size, dtype=bool)
    keep[np.argsort(-tgain)[:1]] = False
    out["gain_test"] = math.sqrt(float(tj.mean())) - math.sqrt(float(tr.mean()))
    out["gain_after_dropping_top1_test"] = math.sqrt(float(tj[keep].mean())) - math.sqrt(float(tr[keep].mean()))
    return out


def _corrections(units, stress_dir: Path) -> dict[str, Any]:
    """Two interpretation fixes + one text/implementation mismatch, on S0 numbers."""
    rows = []
    for epoch in (63, 91, 360):
        resp = units["RESP"]["curves"][epoch - 1]
        add = units["ADD"]["curves"][epoch - 1]
        rows.append({
            "epoch": epoch,
            "RESP_train_rmse_raw": float(resp["train_rmse_raw"]),
            "ADD_train_rmse_raw": float(add["train_rmse_raw"]),
            "train_diff_ADD_minus_RESP": float(add["train_rmse_raw"] - resp["train_rmse_raw"]),
            "RESP_valid_rmse_raw": float(resp["valid_rmse_raw"]),
            "ADD_valid_rmse_raw": float(add["valid_rmse_raw"]),
        })
    counts_path = stress_dir / "stress_counts.csv"
    compressed = []
    total_candidates = 0
    view_share = {"train": 0.0, "valid": 0.0, "test": 0.0}
    if counts_path.is_file():
        with counts_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total_candidates += int(row["n_candidates"])
                if row["compressed"] == "True":
                    compressed.append({
                        "tau": row["tau"], "n_candidates": int(row["n_candidates"]),
                        "n_words_k16": int(row["n_words"]), "panel_size": int(row["panel_size"]),
                        "n_views_train": int(row["n_train_views"]),
                        "n_views_valid": int(row["n_valid_views"]),
                        "n_views_test": int(row["n_test_views"]),
                    })
        # view shares per split from the same table
        with counts_path.open(newline="", encoding="utf-8") as handle:
            rows_all = list(csv.DictReader(handle))
        for split, key in (("train", "n_train_views"), ("valid", "n_valid_views"), ("test", "n_test_views")):
            total = sum(int(r[key]) for r in rows_all)
            comp = sum(int(r[key]) for r in rows_all if r["compressed"] == "True")
            view_share[split] = comp / max(1, total)
    return {
        "same_epoch_train_rmse": rows,
        "note_epoch": ("Comparing RESP epoch 63 Train RMSE with ADD epoch 91 Train RMSE is not a fit "
                       "diagnostic: the two selected states are different points on different curves. At the "
                       "same epoch RESP also has the lower Train RMSE (63: 0.3475 vs 0.3788; 91: 0.2797 vs "
                       "0.3018; 360: 0.0835 vs 0.1002), so a fitting/optimisation component cannot be "
                       "excluded from the S0 Lipo RESP-vs-ADD difference."),
        "compressed_candidate_mass": {
            "classes": compressed,
            "n_candidates_compressed": int(sum(c["n_candidates"] for c in compressed)),
            "n_candidates_total": int(total_candidates),
            "candidate_share": float(sum(c["n_candidates"] for c in compressed) / max(1, total_candidates)),
            "view_share_of_compressed_classes": view_share,
            "note": ("The three Lipo tau classes hold 442+81+91=614 of 702 Train candidates (~87.5%), so "
                     "'13/16 classes stored exactly' is a class-count statement, not evidence that the "
                     "compressed structure is small. Class count, candidate count, actual view frequency and "
                     "held-out structure coverage are different quantities."),
        },
        "latent_text_mismatch": {
            "parent_report_text": "D_LATENT described as farthest-first in the latent k(S) space",
            "parent_code": "response_dictionary_small_s0_v1/code/rd_dict_run.py builds [a(S), q(S)]",
            "resolution": ("This round uses the implementation as fact ([a(S), q(S)]). The parent experiment "
                           "is not re-run for a wording difference."),
        },
    }


def _matched_epochs(units) -> dict[str, Any]:
    """Three-arm numbers at matched epochs (not only at each arm's selected state)."""
    rows = []
    checkpoints = (63, 85, 91, 120, 200, 360)
    for epoch in checkpoints:
        row: dict[str, Any] = {"epoch": epoch}
        for arm in ARMS:
            curve = units[arm]["curves"][epoch - 1]
            for split, key in (("train", "train_rmse_raw"), ("valid", "valid_rmse_raw"), ("test", "test_rmse_raw")):
                row[f"{arm}_{split}_rmse_raw"] = float(curve[key])
        rows.append(row)
    resp_val = np.asarray([float(r["valid_rmse_raw"]) for r in units["RESP"]["curves"]])
    joint_val = np.asarray([float(r["valid_rmse_raw"]) for r in units["JOINT"]["curves"]])
    resp_train = np.asarray([float(r["train_rmse_raw"]) for r in units["RESP"]["curves"]])
    joint_train = np.asarray([float(r["train_rmse_raw"]) for r in units["JOINT"]["curves"]])
    add_val = np.asarray([float(r["valid_rmse_raw"]) for r in units["ADD"]["curves"]])
    return {
        "rows": rows,
        "n_epochs": int(resp_val.size),
        "n_epochs_RESP_valid_below_JOINT": int((resp_val < joint_val).sum()),
        "n_epochs_RESP_train_below_JOINT": int((resp_train < joint_train).sum()),
        "n_epochs_RESP_valid_below_ADD": int((resp_val < add_val).sum()),
        "note": ("Matched-epoch numbers from the committed per-epoch curves. They show that the "
                 "selected-state comparison is not only an epoch-choice effect on Val, but they also "
                 "show it is not epoch-invariant on Test (at epoch 85, JOINT's best-Val epoch, Test is "
                 "0.6999 for JOINT vs 0.7095 for RESP while Val is 0.6555 vs 0.6642)."),
    }


def _labels(gain_rows, bootstrap, influence, stress, edits, follow) -> dict[str, Any]:
    best = next(r for r in gain_rows if r["state"] == "best")
    val_gain = best["gain_val_JOINT_minus_RESP"]
    test_gain = best["gain_test_JOINT_minus_RESP"]
    ref = float(follow["reference_gain"]["val_gain_min"])
    top1 = influence["gain_after_dropping_top1_val"]
    crosses = bool(bootstrap["interval_2.5"] <= 0 <= bootstrap["interval_97.5"])
    if val_gain >= ref and test_gain > 0 and top1 > 0:
        product = "PRODUCT_SPECIFIC_SIGNAL_THIS_CONFIG"
    else:
        product = "NO_PRODUCT_SPECIFIC_ADVANTAGE_SHOWN"
    if crosses:
        sample = "SAMPLE_UNCERTAIN"
    elif val_gain > 0:
        sample = "SAMPLE_CONSISTENT_AT_THIS_STATE"
    else:
        sample = "NO_VAL_ADVANTAGE"
    dictionary = stress.get("decision", {}).get("label", "BUDGET_NOT_RUN")
    edit_label = "EDIT_REPLAY_BLOCKED" if not edits else "HELDOUT_EDIT_SUPPORT_LOW"
    text = {
        "PRODUCT_SPECIFIC_SIGNAL_THIS_CONFIG": (
            "RESP beats the matched joint MLP by at least the pre-declared 0.02 Val margin, Test follows and "
            "the sign survives removing the single most favourable Val molecule."),
        "NO_PRODUCT_SPECIFIC_ADVANTAGE_SHOWN": (
            "The product form does not beat the parameter-matched joint MLP by the pre-declared margin; a "
            "small difference is reported as is and the two arms are not called equivalent. If JOINT is "
            "better, the current evidence supports a general joint computation, not the product operator."),
    }
    return {
        "product_form": product, "product_form_note": text[product],
        "sample_uncertainty": sample,
        "dictionary_stress": dictionary,
        "edits": edit_label,
        "val_gain_JOINT_minus_RESP": float(val_gain),
        "val_gain_reference": ref,
        "test_gain_JOINT_minus_RESP": float(test_gain),
        "gain_after_dropping_top1_val": float(top1),
        "bootstrap_interval_crosses_zero": crosses,
    }


def _plot(units, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is not required
        return None
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, arm in enumerate(ARMS):
        ax = axes[0, col]
        curves = units[arm]["curves"]
        epochs = [r["epoch"] for r in curves]
        for split, style in (("train", "-"), ("valid", "--"), ("test", ":")):
            ax.plot(epochs, [r[f"{split}_rmse_raw"] for r in curves], style, label=split)
        ax.axvline(units[arm]["best"]["epoch"], color="grey", lw=0.8, alpha=0.7)
        ax.set_title(f"Lipo {arm} (best epoch {units[arm]['best']['epoch']})")
        ax.set_ylabel("raw RMSE")
        ax.legend(fontsize=8)
    for col, split in enumerate(("train", "valid", "test")):
        ax = axes[1, col]
        for arm in ARMS:
            curves = units[arm]["curves"]
            ax.plot([r["epoch"] for r in curves], [r[f"{split}_rmse_raw"] for r in curves], label=arm)
        ax.set_title(f"{split} RMSE, three arms")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "rmse_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
