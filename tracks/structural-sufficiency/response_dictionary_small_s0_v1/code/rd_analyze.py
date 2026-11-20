"""Aggregate the four training units + dictionary stage into the round report facts.

    uv run python .../rd_analyze.py --results <results root> --out <round dir>

Produces ``summary.json``, ``main_table.csv``, ``bootstrap.csv``,
``rmse_curves.png`` and the markdown fragment used by ``REPORT.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rd_common as RC  # noqa: E402

UNITS = (("lipo", "RESP"), ("lipo", "ADD"), ("esol", "RESP"), ("esol", "ADD"))
HISTORICAL = {
    # lower Val RMSE of the two pre-declared historical sources (see REPORT)
    "lipo": {"val": 0.7052, "test": 0.7139, "source": "sq-pilot-small-v1 C1/L; gnn_route_clarification_8h_v1 C1/L=0.7069"},
    "esol": {"val": 0.7958, "test": 0.9534, "source": "gnn_route_clarification_8h_v1 C1/L; sq-pilot-small-v1 C1/L=0.8686"},
}


def execution_commit(summary: dict[str, Any]) -> str | None:
    """Commit the process actually started from (``status.json`` is written first)."""
    status = summary.get("status") or {}
    return status.get("commit") or summary.get("commit")


def load_unit(results: Path, task: str, arm: str) -> dict[str, Any] | None:
    base = results / f"{task}_{arm}"
    if not (base / "summary.json").is_file():
        return None
    summary = RC.read_json(base / "summary.json")
    summary["dir"] = str(base)
    summary["predictions_best"] = dict(np.load(base / "predictions_best.npz"))
    summary["predictions_final"] = dict(np.load(base / "predictions_final.npz"))
    status = RC.read_json(base / "status.json") if (base / "status.json").is_file() else {}
    summary["status"] = status
    summary["execution_commit"] = execution_commit(summary)
    return summary


def paired_bootstrap(resp: dict[str, np.ndarray], add: dict[str, np.ndarray], n: int = 2000, seed: int = 1703) -> dict[str, Any]:
    target = np.asarray(resp["valid_target_raw"], dtype=np.float64)
    r = np.asarray(resp["valid_pred_raw"], dtype=np.float64)
    a = np.asarray(add["valid_pred_raw"], dtype=np.float64)
    if not (np.array_equal(resp["valid_row_ids"], add["valid_row_ids"]) and np.array_equal(target, add["valid_target_raw"])):
        raise AssertionError("paired rows are not aligned")
    se_r = (r - target) ** 2
    se_a = (a - target) ** 2
    point = math.sqrt(float(se_a.mean())) - math.sqrt(float(se_r.mean()))
    rng = np.random.default_rng(seed)
    rows = np.arange(target.size)
    diffs = np.empty(int(n))
    for i in range(int(n)):
        idx = rng.integers(0, target.size, size=target.size)
        diffs[i] = math.sqrt(float(se_a[idx].mean())) - math.sqrt(float(se_r[idx].mean()))
    return {
        "n_bootstrap": int(n),
        "seed": int(seed),
        "metric": "RMSE_ADD - RMSE_RESP (raw units, Val molecules resampled with replacement)",
        "point": point,
        "mean": float(diffs.mean()),
        "interval_2.5": float(np.quantile(diffs, 0.025)),
        "interval_97.5": float(np.quantile(diffs, 0.975)),
        "share_positive": float((diffs > 0).mean()),
        "note": "Molecule-row resampling at one fixed training state; not a training-seed interval.",
    }


def influence(resp: dict[str, np.ndarray], add: dict[str, np.ndarray]) -> dict[str, Any]:
    target = np.asarray(resp["valid_target_raw"], dtype=np.float64)
    se_r = (np.asarray(resp["valid_pred_raw"], dtype=np.float64) - target) ** 2
    se_a = (np.asarray(add["valid_pred_raw"], dtype=np.float64) - target) ** 2
    gain = se_a - se_r
    out = {}
    for k in (1, 5):
        keep = np.ones(target.size, dtype=bool)
        if k:
            keep[np.argsort(-gain)[:k]] = False
        out[f"gain_after_dropping_top{k}"] = math.sqrt(float(se_a[keep].mean())) - math.sqrt(float(se_r[keep].mean()))
        out[f"dropped_rows_top{k}"] = np.asarray(resp["valid_row_ids"])[np.argsort(-gain)[:k]].tolist()
    out["gain_full"] = math.sqrt(float(se_a.mean())) - math.sqrt(float(se_r.mean()))
    out["largest_single_molecule_contribution"] = float(gain.max())
    out["share_of_total_gain_top1"] = float(gain.max() / gain.sum()) if gain.sum() != 0 else None
    return out


def selection_labels(task: str, val_gain: float, test_gain: float, interval: dict[str, Any], influence_out: dict[str, Any], val_rmse: float) -> list[str]:
    ref = 0.02 if task == "lipo" else 0.03
    labels: list[str] = []
    if val_gain >= ref and test_gain > 0 and influence_out["gain_after_dropping_top1"] > 0:
        labels.append("CONDITIONAL_SIGNAL_S0")
        if interval["interval_2.5"] <= 0 <= interval["interval_97.5"]:
            labels.append("SAMPLE_UNCERTAIN")
    if val_gain * test_gain < 0 and max(abs(val_gain), abs(test_gain)) >= ref:
        labels.append("SPLIT_CONFLICT")
    if not labels:
        labels.append("NO_CONDITIONAL_ADVANCE_THIS_CONFIG")
    hist = HISTORICAL[task]["val"]
    labels.append("NEAR_HISTORICAL_C1" if val_rmse <= 1.05 * hist else "PERFORMANCE_BEHIND_C1")
    return labels


def w0_probe(summary: dict[str, Any], config: dict[str, Any], task: str) -> dict[str, Any]:
    import torch

    import rd_model as RM
    import rd_train as RT

    dataset_name = "ogbg-mollipo" if task == "lipo" else "ogbg-molesol"
    dataset = RT.build_dataset(dataset_name, task)
    state = torch.load(Path(summary["dir"]) / "best.pt", map_location="cpu", weights_only=True)
    model = RM.build_model(
        dataset.dims,
        RM.ModelConfig(
            hidden=int(config["hidden"]), latent=int(config["latent"]), n_layers=int(config["n_layers"]),
            dropout=float(config["dropout"]), mode="RESP"),
        state,
    ).to("cpu")
    model.eval()
    original = {split: RT.evaluate_split(model, dataset, split, "cpu", 256) for split in ("train", "valid", "test")}
    zeroed = RM.zero_fusion_w(model, state)
    model0 = RM.build_model(
        dataset.dims,
        RM.ModelConfig(
            hidden=int(config["hidden"]), latent=int(config["latent"]), n_layers=int(config["n_layers"]),
            dropout=float(config["dropout"]), mode="RESP"),
        zeroed,
    ).to("cpu")
    model0.eval()
    out = {}
    for split in ("train", "valid", "test"):
        pred = RT.evaluate_split(model0, dataset, split, "cpu", 256)
        std = float(dataset.scaler["std"])
        out[split] = {
            "rmse_raw_w0": RC.rmse(pred["pred_raw"], pred["target_raw"]),
            "rmse_raw_trained": RC.rmse(original[split]["pred_raw"], original[split]["target_raw"]),
            "prediction_change_rmse_raw": RC.rmse(pred["pred_raw"], original[split]["pred_raw"]),
            "max_abs_prediction_change_raw": float(np.max(np.abs(pred["pred_raw"] - original[split]["pred_raw"]))),
            "prediction_change_std_units": float(RC.rmse(pred["pred_raw"], original[split]["pred_raw"]) / std),
        }
    out["note"] = "Fixed-weight dependence probe on the best-Val RESP state; not a post-hoc necessity proof."
    return out


def paired_edit_diagnostic(units: dict[tuple[str, str], dict | None], pairs_path: Path | None) -> dict[str, Any]:
    """Real paired property differences on full F(G), if a catalogue maps exactly.

    The catalogue's own split labels are ignored; official split membership is
    recomputed from the OGB row ids and checked against the frozen labels.
    """
    if pairs_path is None or not pairs_path.is_file():
        return {"state": "NOT_COVERED", "reason": f"catalogue not found: {pairs_path}"}
    import csv

    resp = units.get(("lipo", "RESP"))
    add = units.get(("lipo", "ADD"))
    if resp is None and add is None:
        return {"state": "NOT_COVERED", "reason": "no Lipo unit"}
    pred_maps = {}
    target_maps = {}
    for arm, unit in (("RESP", resp), ("ADD", add)):
        if unit is None:
            continue
        preds = unit["predictions_best"]
        pm: dict[int, float] = {}
        tm: dict[int, float] = {}
        for split in ("train", "valid", "test"):
            for row, pred, target in zip(
                np.asarray(preds[f"{split}_row_ids"]).tolist(),
                np.asarray(preds[f"{split}_pred_raw"]).tolist(),
                np.asarray(preds[f"{split}_target_raw"]).tolist(),
            ):
                pm[int(row)] = float(pred)
                tm[int(row)] = float(target)
        pred_maps[arm] = pm
        target_maps[arm] = tm
    rows = list(csv.DictReader(pairs_path.open(encoding="utf-8")))
    import sys as _sys

    _sys.path.insert(0, str(RC.SQ_CODE))
    import sq_common as SQ

    payload = RC.dataset_cache("ogbg-mollipo")
    official = np.empty(len(payload["row_ids"]), dtype=object)
    for split in ("train", "valid", "test"):
        official[np.asarray(payload[f"{split}_idx"], dtype=np.int64)] = split
    labels = np.asarray(payload["labels"], dtype=np.float64)
    by_split: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        try:
            a = int(row["row_a"])
            b = int(row["row_b"])
            ya = float(row["y_a"])
            yb = float(row["y_b"])
        except (KeyError, ValueError):
            return {"state": "NOT_COVERED", "reason": "catalogue lacks row_a/row_b/y_a/y_b"}
        if not (np.isfinite(ya) and np.isfinite(yb)):
            continue
        if abs(ya - labels[a]) > 1e-6 or abs(yb - labels[b]) > 1e-6:
            return {"state": "NOT_COVERED", "reason": "catalogue labels do not match official rows"}
        split = str(official[a])
        if split != str(official[b]):
            continue
        by_split.setdefault(split, []).append({"a": a, "b": b, "dy": ya - yb})
    out = {"state": "COVERED", "catalogue": str(pairs_path.relative_to(RC.REPO_ROOT)),
           "catalogue_rows": len(rows), "official_split_recomputed": True,
           "support": {split: len(values) for split, values in by_split.items()}, "per_split": {}}
    for split, values in by_split.items():
        entry = {"n_pairs": len(values)}
        for arm, pm in pred_maps.items():
            errors = [pm[v["a"]] - pm[v["b"]] - v["dy"] for v in values]
            entry[arm] = {
                "paired_difference_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "paired_difference_mae": float(np.mean(np.abs(errors))),
                "mean_signed_error": float(np.mean(errors)),
            }
        out["per_split"][split] = entry
    if not by_split:
        out["state"] = "NOT_COVERED"
        out["reason"] = "no pair with both rows in one official split"
    out["note"] = ("Catalogue is an existing amide-difference edit set; its own split column is not used. "
                   "Support is small and restricted to the splits listed.")
    return out


def plot_curves(units: dict[tuple[str, str], dict], out: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (task, arm) in zip(axes.flat, UNITS):
        unit = units.get((task, arm))
        ax.set_title(f"{task.upper()} {arm}")
        if unit is None:
            ax.text(0.5, 0.5, "BUDGET_NOT_RUN", ha="center", va="center")
            continue
        curves = unit["curves"]
        epochs = [r["epoch"] for r in curves]
        for split, style in (("train", "-"), ("valid", "--"), ("test", ":")):
            ax.plot(epochs, [r[f"{split}_rmse_raw"] for r in curves], style, label=split)
        best = unit["best"]["epoch"]
        ax.axvline(best, color="grey", lw=0.8, alpha=0.7)
        ax.set_ylabel("raw RMSE")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    path = out / "rmse_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| task | arm | best epoch | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | fallback Val |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (task, arm), unit in summary["units"].items():
        if unit is None:
            lines.append(f"| {task} | {arm} | - | - | - | - | - | - | BUDGET_NOT_RUN |")
            continue
        b = unit["best_metrics"]
        fb = unit["diagnostics"]["valid"]["fallback_molecule_fraction"]
        lines.append(
            f"| {task} | {arm} | {b['epoch']} | {b['train_rmse_raw']:.4f} | {b['valid_rmse_raw']:.4f} | "
            f"{b['test_rmse_raw']:.4f} | {b['valid_mae_raw']:.4f} | {b['test_mae_raw']:.4f} | {fb:.3f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    parser.add_argument("--out", default=str(RC.ROUND_DIR))
    parser.add_argument("--no-w0", action="store_true")
    parser.add_argument("--edit-pairs", default="tracks/structural-sufficiency/lcrc_lipo_amide_dependency_public122_v1/analysis/natural_pairs.csv")
    args = parser.parse_args(argv)
    results = Path(args.results) if args.results else RC.results_root()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = RC.load_config()

    units: dict[tuple[str, str], dict[str, Any] | None] = {}
    for task, arm in UNITS:
        units[(task, arm)] = load_unit(results, task, arm)
    loaded = {k: v for k, v in units.items() if v is not None}
    if not loaded:
        raise SystemExit(f"no completed units under {results}")

    rows = []
    diagnostics: dict[str, Any] = {"units": {}, "gains": {}, "labels": {}, "historical": HISTORICAL}
    for (task, arm), unit in loaded.items():
        key = f"{task}_{arm}"
        rows.append({
            "task": task,
            "arm": arm,
            "best_epoch": unit["best"]["epoch"],
            "train_rmse_raw": unit["best_metrics"]["train_rmse_raw"],
            "valid_rmse_raw": unit["best_metrics"]["valid_rmse_raw"],
            "test_rmse_raw": unit["best_metrics"]["test_rmse_raw"],
            "train_mae_raw": unit["best_metrics"]["train_mae_raw"],
            "valid_mae_raw": unit["best_metrics"]["valid_mae_raw"],
            "test_mae_raw": unit["best_metrics"]["test_mae_raw"],
            "final_valid_rmse_raw": unit["final_metrics"]["valid_rmse_raw"],
            "final_test_rmse_raw": unit["final_metrics"]["test_rmse_raw"],
            "last10_valid_rmse_raw": unit["last10_mean"]["valid_rmse_raw"],
            "last10_test_rmse_raw": unit["last10_mean"]["test_rmse_raw"],
            "n_parameters": unit["n_parameters"],
            "seconds": unit["seconds"],
            "horizon_limited": unit["horizon"]["horizon_limited"],
            "init_state_hash": unit["init_state_hash"],
            "data_cache_sha256": unit["data_cache_sha256"],
            "commit": unit["execution_commit"],
            "checkout_commit_at_finish": unit["commit"],
            "status": unit["status"].get("state"),
        })
    RC.write_csv(out / "main_table.csv", rows)

    for task in ("lipo", "esol"):
        resp = units.get((task, "RESP"))
        add = units.get((task, "ADD"))
        if resp is None or add is None:
            diagnostics["gains"][task] = {"state": "MISSING_ARM", "have": [k for k in (resp, add) if k]}
            diagnostics["labels"][task] = ["BUDGET_NOT_RUN"]
            continue
        r = resp["best_metrics"]
        a = add["best_metrics"]
        val_gain = a["valid_rmse_raw"] - r["valid_rmse_raw"]
        test_gain = a["test_rmse_raw"] - r["test_rmse_raw"]
        final_gain_val = add["final_metrics"]["valid_rmse_raw"] - resp["final_metrics"]["valid_rmse_raw"]
        final_gain_test = add["final_metrics"]["test_rmse_raw"] - resp["final_metrics"]["test_rmse_raw"]
        last10_gain_val = add["last10_mean"]["valid_rmse_raw"] - resp["last10_mean"]["valid_rmse_raw"]
        last10_gain_test = add["last10_mean"]["test_rmse_raw"] - resp["last10_mean"]["test_rmse_raw"]
        boot = paired_bootstrap(resp["predictions_best"], add["predictions_best"])
        infl = influence(resp["predictions_best"], add["predictions_best"])
        labels = selection_labels(task, val_gain, test_gain, boot, infl, r["valid_rmse_raw"])
        if resp["horizon"]["horizon_limited"] or add["horizon"]["horizon_limited"]:
            labels.append("HORIZON_LIMITED")
        w0 = None if args.no_w0 else w0_probe(resp, config, task)
        diagnostics["gains"][task] = {
            "val_best": {"val_gain": val_gain, "test_gain": test_gain, "resp_val": r["valid_rmse_raw"], "add_val": a["valid_rmse_raw"]},
            "final": {"val_gain": final_gain_val, "test_gain": final_gain_test},
            "last10": {"val_gain": last10_gain_val, "test_gain": last10_gain_test},
            "reference_amplitude": config["datasets"]["ogbg-mollipo" if task == "lipo" else "ogbg-molesol"]["train_std_gain_reference"],
        }
        diagnostics["bootstrap"] = diagnostics.get("bootstrap", {})
        diagnostics["bootstrap"][task] = boot
        diagnostics["influence"] = diagnostics.get("influence", {})
        diagnostics["influence"][task] = infl
        diagnostics["w0_probe"] = diagnostics.get("w0_probe", {})
        diagnostics["w0_probe"][task] = w0
        diagnostics["labels"][task] = labels
        diagnostics["units"][task] = {
            "resp_best_epoch": resp["best"]["epoch"],
            "add_best_epoch": add["best"]["epoch"],
            "resp_diagnostics": resp["diagnostics"],
            "add_diagnostics": add["diagnostics"],
            "resp_horizon": resp["horizon"],
            "add_horizon": add["horizon"],
            "init_hashes_identical": resp["init_state_hash"] == add["init_state_hash"],
        }

    # dictionary stage
    dictionaries = {}
    for task in ("lipo", "esol"):
        path = results / f"{task}_RESP" / "dictionary" / "dictionary_summary.json"
        if path.is_file():
            dictionaries[task] = RC.read_json(path)
        else:
            dictionaries[task] = {"state": "NOT_RUN"}
    diagnostics["dictionaries"] = dictionaries
    for task, payload in dictionaries.items():
        if payload.get("state") == "NOT_RUN":
            continue
        unit = units.get((task, "RESP"))
        std = float(unit["target_std"]) if unit else float("nan")
        resp_val = float(unit["best_metrics"]["valid_rmse_raw"]) if unit else float("nan")
        z = np.load(results / f"{task}_RESP" / "dictionary" / "dictionary_predictions.npz")
        ds_name = "ogbg-mollipo" if task == "lipo" else "ogbg-molesol"
        cache = RC.fragment_cache(ds_name)
        for name in ("D_RESPONSE", "D_LATENT", "D_FREQ"):
            per = payload["dictionaries"][name]["per_split"]
            diff = np.abs(z[f"{name}_valid_pred_raw"] - z[f"{name}_valid_resp_pred_raw"])
            worst = int(np.argmax(diff))
            keep = np.ones(diff.size, bool)
            keep[worst] = False
            pred = z[f"{name}_valid_pred_raw"]
            resp = z[f"{name}_valid_resp_pred_raw"]
            target = z[f"{name}_valid_target_raw"]
            per["dev_criterion"] = {
                "delta_val_rmse": float(per["valid"]["rmse_raw"] - resp_val),
                "threshold_train_std": 0.01 * std,
                "support_view_fraction": per["valid"]["supported_view_fraction"],
                "passes": bool(per["valid"]["rmse_raw"] - resp_val <= 0.01 * std and per["valid"]["supported_view_fraction"] >= 0.8),
                "worst_val_fidelity": {
                    "split_index": worst,
                    "official_row_id": int(cache["valid_mol_row_id"][worst]),
                    "abs_prediction_change_raw": float(diff[worst]),
                    "val_rmse_dictionary_excluding_worst": float(np.sqrt(np.mean((pred[keep] - target[keep]) ** 2))),
                    "val_rmse_resp_excluding_worst": float(np.sqrt(np.mean((resp[keep] - target[keep]) ** 2))),
                },
            }
    diagnostics["paired_edit_diagnostic"] = paired_edit_diagnostic(
        units, (RC.REPO_ROOT / args.edit_pairs) if args.edit_pairs else None
    )
    diagnostics["test_use"] = "DEVELOPMENT_TEST"
    diagnostics["test_policy"] = config["test_policy"]
    diagnostics["protocol_hash"] = RC.protocol_hash(config)
    diagnostics["commit"] = RC.git_commit()
    code_names = (
        "rd_common.py", "rd_fragment.py", "rd_model.py", "rd_train.py", "rd_run.py",
        "rd_dictionary.py", "rd_dict_run.py", "rd_prepare.py", "rd_checks.py",
    )
    diagnostics["analysis_working_tree_code_sha256"] = RC.source_hashes([RC.CODE_DIR / n for n in code_names])
    exec_commits = sorted({u["execution_commit"] for u in loaded.values() if u.get("execution_commit")})
    diagnostics["executed_code_sha256"] = {
        commit: {
            name: RC.git_blob_sha256(commit, f"tracks/structural-sufficiency/response_dictionary_small_s0_v1/code/{name}")
            for name in code_names
        }
        for commit in exec_commits
    }
    diagnostics["task_relevance"] = {
        "label": "TASK_RELEVANCE_NOT_ESTABLISHED",
        "reason": "The only real paired-edit set available has 73 pairs, all in the official Train split, and on those pairs ADD has the lower paired-difference RMSE; there is no Val/Test edit support.",
    }
    diagnostics["config_sha256"] = RC.sha256_file(RC.CONFIG_PATH)
    diagnostics["data_identity"] = {
        "official_cache_sha256": RC.SQ.EXPECTED_CACHE_SHA256,
        "fragment_cache_sha256": RC.EXPECTED_FRAGMENT_CACHE_SHA256,
        "canonical_init_state_sha256": RC.sha256_file(RC.DATA_DIR / "canonical_init_state.pt"),
    }
    # Post-hoc subgroup read: molecules that actually use the fusion head vs the
    # no-cut fallback head.  The formal endpoint remains the whole split; this
    # decomposition only shows where an arm difference is generated.
    fusion = {}
    for task, cache_name in (("lipo", "ogbg-mollipo"), ("esol", "ogbg-molesol")):
        cache = RC.fragment_cache(cache_name)
        fusion[task] = {}
        for split in ("valid", "test"):
            fb = np.asarray(cache[f"{split}_mol_fallback"]) >= 0
            entry = {"n_all": int(fb.size), "n_fallback": int(fb.sum()), "n_views": int((~fb).sum()), "subsets": {}}
            for subset, mask in (("all", np.ones(fb.size, bool)), ("views", ~fb), ("fallback", fb)):
                preds = {}
                for arm in ("RESP", "ADD"):
                    z = np.load(results / f"{task}_{arm}" / "predictions_best.npz")
                    preds[arm] = (z[f"{split}_pred_raw"], z[f"{split}_target_raw"])
                target = preds["RESP"][1]
                row = {}
                for arm in ("RESP", "ADD"):
                    row[f"{arm}_rmse_raw"] = float(np.sqrt(np.mean((preds[arm][0][mask] - target[mask]) ** 2)))
                row["gain_add_minus_resp"] = row["ADD_rmse_raw"] - row["RESP_rmse_raw"]
                entry["subsets"][subset] = row
            fusion[task][split] = entry
    diagnostics["fusion_subgroups"] = fusion

    diagnostics["resources"] = {
        "sum_unit_seconds": float(sum(u["seconds"] for u in loaded.values())),
        "sum_unit_gpu_hours_estimate": float(sum(u["seconds"] for u in loaded.values()) / 3600.0),
        "soft_a100_hours": 4.0,
        "hard_a100_hours": 6.0,
        "note": "Each unit ran alone on one physical A100; wall time includes per-epoch three-split evaluation.",
    }
    runs = RC.read_json(RC.ROUND_DIR / "runs.json") if (RC.ROUND_DIR / "runs.json").is_file() else {}
    manifest = {
        "round_id": config["round_id"],
        "protocol_id": config["protocol_id"],
        "protocol_hash": diagnostics["protocol_hash"],
        "config_path": str(RC.CONFIG_PATH.relative_to(RC.REPO_ROOT)),
        "config_sha256": diagnostics["config_sha256"],
        "analysis_commit": RC.git_commit(),
        "analysis_working_tree_code_sha256": diagnostics["analysis_working_tree_code_sha256"],
        "executed_code_sha256": diagnostics["executed_code_sha256"],
        "data_identity": diagnostics["data_identity"],
        "resources": diagnostics["resources"],
        "test_policy": "always",
        "test_use": "DEVELOPMENT_TEST",
        "execution_registry": runs,
        "units": {},
    }
    for (task, arm), unit in loaded.items():
        st = unit["status"]
        reg = (runs.get("formal_units") or {}).get(f"{task}_{arm}", {})
        manifest["units"][f"{task}_{arm}"] = {
            "run_id": reg.get("run_id") or st.get("run_id"),
            "experiment": st.get("experiment"),
            "execution_commit": unit["execution_commit"],
            "checkout_commit_at_finish": unit["commit"],
            "status": st.get("state"),
            "started_at": st.get("started_at"),
            "finished_at": st.get("finished_at"),
            "seconds": unit["seconds"],
            "best_epoch": unit["best"]["epoch"],
            "best_metrics": unit["best_metrics"],
            "final_metrics": unit["final_metrics"],
            "last10_mean": unit["last10_mean"],
            "init_state_hash": unit["init_state_hash"],
            "train_order_sha256": st.get("train_order_sha256"),
            "n_parameters": unit["n_parameters"],
            "eval_batch_reductions": {k: unit["diagnostics"][k]["eval_batch_reductions"] for k in ("train", "valid", "test")},
            "environment": unit.get("environment"),
            "horizon": unit["horizon"],
        }
    RC.write_json(out / "run_manifest.json", manifest)
    gain_rows = []
    for task in ("lipo", "esol"):
        g = diagnostics["gains"].get(task, {})
        if "val_best" not in g:
            continue
        for state in ("val_best", "final", "last10"):
            gain_rows.append({"task": task, "state": state, **g[state]})
    RC.write_csv(out / "gains.csv", gain_rows)
    boot_rows = []
    for task, b in diagnostics.get("bootstrap", {}).items():
        boot_rows.append({"task": task, **{k: v for k, v in b.items() if not isinstance(v, str) or k in ("metric", "note")}})
    RC.write_csv(out / "bootstrap.csv", boot_rows)
    dict_rows = []
    for task, payload in diagnostics.get("dictionaries", {}).items():
        if payload.get("state") == "NOT_RUN":
            continue
        for name, block in payload["dictionaries"].items():
            for split in ("train", "valid", "test"):
                q = block["per_split"][split]
                dict_rows.append({
                    "task": task,
                    "dictionary": name,
                    "split": split,
                    "rmse_raw": q["rmse_raw"],
                    "mae_raw": q["mae_raw"],
                    "resp_rmse_raw": q["resp_rmse_raw"],
                    "fidelity_rmse_raw": q["fidelity_rmse_raw"],
                    "supported_view_fraction": q["supported_view_fraction"],
                    "exact_word_view_fraction": q["exact_word_view_fraction"],
                    "fallback_molecule_fraction": q["fallback_molecule_fraction"],
                    "n_words_total": block["n_words_total"],
                    "n_candidate_words_total": block["n_candidate_words_total"],
                    "n_classes": block["n_classes"],
                    "n_classes_exact_storage": block["n_classes_exact_storage"],
                    "compression_ratio_words_over_candidates": block["compression_ratio_words_over_candidates"],
                    "alpha_seconds": block["alpha_seconds"],
                    "dev_pass_val": block["per_split"]["dev_criterion"]["passes"],
                })
    RC.write_csv(out / "dictionary_table.csv", dict_rows)
    fusion_rows = []
    for task, splits in diagnostics.get("fusion_subgroups", {}).items():
        for split, entry in splits.items():
            for subset, row in entry["subsets"].items():
                fusion_rows.append({
                    "task": task,
                    "split": split,
                    "subset": subset,
                    "n_molecules": entry["n_all"] if subset == "all" else entry["n_fallback"] if subset == "fallback" else entry["n_views"],
                    **row,
                })
    RC.write_csv(out / "fusion_subgroups.csv", fusion_rows)
    diagnostics["main_table_markdown"] = markdown_table({"units": units})
    path = plot_curves(units, out)
    diagnostics["plot"] = str(path) if path else None
    RC.write_json(out / "summary.json", diagnostics)
    (out / "main_table.md").write_text(diagnostics["main_table_markdown"] + "\n", encoding="utf-8")
    print(json.dumps(RC.jsonable(diagnostics), indent=2, sort_keys=True)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
