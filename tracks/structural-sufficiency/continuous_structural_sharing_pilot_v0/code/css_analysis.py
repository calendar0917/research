"""CSS-v0 analysis: four-arm table, paired molecule bootstrap, size sensitivity,
reference usage and near/mid/far structure groups.

Reads only frozen run outputs and frozen preparation artefacts; no refitting and
no selection.  Run after all four formal runs have been pulled into
``results/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import css_common as C
import css_fit as F

ARMS = ("DIRECT", "SHARE")
SIZES = (100, 50)
N_BOOT = 1000
BOOT_SEED = 0
METRIC_KEYS = ("fit", "heldout", "train_full", "valid", "test")


# --------------------------------------------------------------------- loading
def _load_runs(results_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/manifest.json")):
        payload = C.read_json(path)
        if payload.get("format") != "css-v0-run-v1" or payload.get("smoke"):
            continue
        key = (payload["arm"], int(payload["size"]))
        if key in runs:
            raise ValueError(f"duplicate run for {key}: {path}")
        payload["_dir"] = str(path.parent)
        runs[key] = payload
    missing = [(a, s) for s in SIZES for a in ARMS if (a, s) not in runs]
    if missing:
        raise ValueError(f"missing runs: {missing}")
    return runs


def _load_curves(payload: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(payload["_dir"]) / "curves.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_predictions(payload: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for tag, recorded in payload["predictions"].items():
        path = Path(recorded)
        if not path.is_file():
            path = Path(payload["_dir"]) / path.name
        with np.load(path) as data:
            out[tag] = {key: np.asarray(data[key]) for key in data.files}
    return out


def _metrics_from_predictions(predictions: Mapping[str, Mapping[str, np.ndarray]],
                              split: str) -> dict[str, float]:
    block = predictions[split]
    return {"rmse": float(C.rmse(block["label"], block["pred"])),
            "mae": float(C.mae(block["label"], block["pred"]))}


# --------------------------------------------------------------------- bootstrap
def paired_bootstrap(labels: np.ndarray, pred_direct: np.ndarray,
                     pred_share: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> dict[str, Any]:
    """Molecule-level paired bootstrap of RMSE(DIRECT) - RMSE(SHARE)."""
    labels = np.asarray(labels, np.float64)
    pred_direct = np.asarray(pred_direct, np.float64)
    pred_share = np.asarray(pred_share, np.float64)
    n = labels.size
    point = float(C.rmse(labels, pred_direct) - C.rmse(labels, pred_share))
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot)
    rmse_d = np.empty(n_boot)
    rmse_s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        rd = float(C.rmse(y, pred_direct[idx]))
        rs = float(C.rmse(y, pred_share[idx]))
        rmse_d[i] = rd
        rmse_s[i] = rs
        gains[i] = rd - rs
    low, high = np.percentile(gains, [2.5, 97.5])
    return {
        "n_molecules": int(n),
        "n_replicates": int(n_boot),
        "seed": int(seed),
        "gain_point": point,
        "gain_mean": float(gains.mean()),
        "gain_std": float(gains.std(ddof=1)),
        "gain_ci95": [float(low), float(high)],
        "fraction_positive": float((gains > 0).mean()),
        "rmse_direct_point": float(C.rmse(labels, pred_direct)),
        "rmse_share_point": float(C.rmse(labels, pred_share)),
        "rmse_direct_boot_mean": float(rmse_d.mean()),
        "rmse_share_boot_mean": float(rmse_s.mean()),
    }


# --------------------------------------------------------------------- tables
def main_table(runs: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for size in SIZES:
        for arm in ARMS:
            payload = runs[(arm, size)]
            best = payload["best"]
            predictions = _load_predictions(payload)
            check = {split: _metrics_from_predictions(predictions, split)
                     for split in ("fit", "train_full", "valid", "test")}
            if size == 50:
                check["heldout"] = _metrics_from_predictions(predictions, "heldout")
            for split in ("fit", "valid", "test"):
                recorded = best.get(f"{split}_rmse")
                if recorded is not None and abs(check[split]["rmse"] - float(recorded)) > 1e-6:
                    raise ValueError(f"prediction/manifest mismatch {arm}_{size} {split}")
            rows.append({
                "unit": f"{arm}_{size}",
                "arm": arm,
                "fit_fraction": size / 100.0,
                "fit_molecules": int(payload["n_fit"]),
                "heldout_molecules": int(payload.get("n_heldout", 0)),
                "parameters_total": int(payload["params"]["total"]),
                "best_epoch": int(best["epoch"]),
                "epochs": int(payload["epochs"]),
                "fit_rmse": check["fit"]["rmse"],
                "fit_mae": check["fit"]["mae"],
                "heldout_rmse": check.get("heldout", {}).get("rmse"),
                "heldout_mae": check.get("heldout", {}).get("mae"),
                "valid_rmse": check["valid"]["rmse"],
                "valid_mae": check["valid"]["mae"],
                "test_rmse": check["test"]["rmse"],
                "test_mae": check["test"]["mae"],
                "train_full_rmse": check["train_full"]["rmse"],
                "generalisation_gap_valid": check["valid"]["rmse"] - check["fit"]["rmse"],
                "train_wall_seconds": float(payload["train_seconds"]),
                "train_gpu_hours": float(payload.get("gpu_hours", 0.0)),
                "peak_vram_mb": payload.get("peak_vram_mb"),
            })
    return rows


def curves_table(runs: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (arm, size), payload in sorted(runs.items(), key=lambda item: (item[0][1], item[0][0])):
        for row in _load_curves(payload):
            out = {"unit": f"{arm}_{size}", "arm": arm, "size": size,
                   "epoch": int(row["epoch"]), "lr": float(row["lr"]),
                   "epoch_seconds": float(row["epoch_seconds"])}
            for key in METRIC_KEYS:
                value = row.get(f"{key}_rmse", "")
                out[f"{key}_rmse"] = float(value) if value not in ("", "nan") else None
            rows.append(out)
    return rows


# --------------------------------------------------------------------- analysis
def reference_analysis(runs: dict[tuple[str, int], dict[str, Any]],
                       prep_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "format": "css-v0-reference-analysis-v1",
        "caveat": ("Reference weights are soft similarities to real training "
                   "structures. They are not atom/bond contributions, not "
                   "causal attributions, and not chemical interpretations."),
        "sizes": {},
    }
    for size in SIZES:
        share = runs[("SHARE", size)]
        direct = runs[("DIRECT", size)]
        pred_share = _load_predictions(share)
        pred_direct = _load_predictions(direct)
        valid_share = pred_share["valid"]
        valid_direct = pred_direct["valid"]
        if not np.array_equal(valid_share["row_id"], valid_direct["row_id"]):
            raise ValueError("valid row order mismatch between arms")
        with np.load(prep_dir / f"valid_{size}.npz") as data:
            q = np.asarray(data["q"], np.float32)
            nearest = np.asarray(data["nearest_d2"], np.float32)
            root_row = np.asarray(data["root_row"], np.int64)
            canonical_rank = np.asarray(data["canonical_rank"], np.int64)
        refs = C.read_json(prep_dir / f"refs_{size}.json")
        examples = C.read_json(prep_dir / f"examples_{size}.json")

        # --- reference usage over official Valid roots -------------------
        usage = q.mean(axis=0)
        order = np.argsort(-usage)
        ref_rows = []
        for rank in order:
            entry = dict(refs["references"][int(rank)])
            entry["mean_weight_valid"] = float(usage[rank])
            entry["rank_by_mean_weight"] = int(np.where(order == rank)[0][0])
            ref_rows.append(entry)

        entropy = F.entropy_stats(q.astype(np.float64))
        entropy.pop("mean_column_weight", None)

        # --- near / mid / far molecule groups ----------------------------
        mol_rows = np.unique(root_row)
        mol_distance = np.array([nearest[root_row == row].mean() for row in mol_rows])
        order_mol = np.argsort(mol_distance, kind="stable")
        groups = {"near": order_mol[:140], "mid": order_mol[140:280],
                  "far": order_mol[280:]}
        assert groups["far"].size == 140
        valid_row = valid_share["row_id"]
        label = valid_share["label"]
        group_stats: dict[str, Any] = {}
        for name, idx in groups.items():
            rows = mol_rows[idx]
            mask = np.isin(valid_row, rows)
            stats = {
                "n_molecules": int(rows.size),
                "mean_nearest_d2_range": [float(mol_distance[idx].min()),
                                          float(mol_distance[idx].max())],
            }
            for arm, block in (("DIRECT", valid_direct), ("SHARE", valid_share)):
                stats[arm] = {
                    "rmse": float(C.rmse(label[mask], block["pred"][mask])),
                    "mae": float(C.mae(label[mask], block["pred"][mask])),
                }
            stats["gain_direct_minus_share_rmse"] = (
                stats["DIRECT"]["rmse"] - stats["SHARE"]["rmse"])
            group_stats[name] = stats

        # --- the eight fixed Valid examples ------------------------------
        example_rows = []
        for example in examples["examples"]:
            row = int(example["row"])
            entry = dict(example)
            mask = valid_row == row
            entry["valid_prediction"] = {
                "label": float(label[mask][0]),
                "SHARE": float(valid_share["pred"][mask][0]),
                "DIRECT": float(valid_direct["pred"][mask][0]),
            }
            entry["valid_abs_error"] = {
                "SHARE": abs(entry["valid_prediction"]["SHARE"] - entry["valid_prediction"]["label"]),
                "DIRECT": abs(entry["valid_prediction"]["DIRECT"] - entry["valid_prediction"]["label"]),
            }
            example_rows.append(entry)

        # --- reference sharing across different molecules ----------------
        # how many distinct roots fall in the top-8 references of the Valid set
        top8 = np.argsort(-q, axis=1)[:, :8]
        out["sizes"][str(size)] = {
            "n_valid_roots": int(q.shape[0]),
            "n_references": int(q.shape[1]),
            "tau": float(refs["temperature"]["tau"]),
            "mean_nearest_d2_valid": float(nearest.mean()),
            "entropy": entropy,
            "reference_usage_mean": usage.tolist(),
            "reference_usage_max": float(usage.max()),
            "reference_usage_min": float(usage.min()),
            "reference_usage_top10": ref_rows[:10],
            "reference_usage_bottom10": ref_rows[-10:],
            "groups": group_stats,
            "examples": example_rows,
            "mean_top8_cooccurrence": float(
                np.mean([len(set(np.asarray(top8[m]).tolist())) for m in range(top8.shape[0])])),
            "best_epoch": int(share["best"]["epoch"]),
            "checkpoint": f"{Path(share['_dir']).name}/state_best.pt",
        }
    return out


# --------------------------------------------------------------------- writing
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(results_dir: str = str(C.RESULTS_DIR),
         prep_dir: str = str(C.PREP_DIR),
         out_dir: str | None = None) -> dict[str, Any]:
    results = Path(results_dir)
    prep = Path(prep_dir)
    out = Path(out_dir) if out_dir else results
    out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(results)

    table_rows = main_table(runs)
    _write_csv(out / "main_table.csv", table_rows)

    curve_rows = curves_table(runs)
    _write_csv(out / "curves_all.csv", curve_rows)
    # compatibility name used in the task brief
    _write_csv(out / "curve.csv", curve_rows)

    gains: dict[str, Any] = {"format": "css-v0-gain-bootstrap-v1",
                             "metric": "RMSE(DIRECT) - RMSE(SHARE), positive favours SHARE",
                             "n_boot": N_BOOT, "seed": BOOT_SEED}
    for size in SIZES:
        pred_direct = _load_predictions(runs[("DIRECT", size)])
        pred_share = _load_predictions(runs[("SHARE", size)])
        for split in ("valid", "test"):
            if not np.array_equal(pred_direct[split]["row_id"], pred_share[split]["row_id"]):
                raise ValueError(f"row order mismatch {split} size {size}")
            gains[f"{split}_{size}"] = paired_bootstrap(
                pred_share[split]["label"],
                pred_direct[split]["pred"],
                pred_share[split]["pred"])
    gains["size_delta"] = {
        "gain_50_minus_gain_100_valid": (gains["valid_50"]["gain_point"]
                                         - gains["valid_100"]["gain_point"]),
        "gain_50_minus_gain_100_test": (gains["test_50"]["gain_point"]
                                        - gains["test_100"]["gain_point"]),
    }
    C.write_json(out / "gain_bootstrap.json", gains)

    sensitivity_rows = []
    for arm in ARMS:
        row: dict[str, Any] = {"arm": arm}
        for split in ("fit", "heldout", "valid", "test"):
            values = {}
            for size in SIZES:
                values[size] = runs[(arm, size)]["metrics"].get(f"{split}_rmse")
            row[f"{split}_rmse_100"] = values[100]
            row[f"{split}_rmse_50"] = values[50]
            row[f"{split}_delta_100_minus_50"] = (
                values[100] - values[50]
                if values[100] is not None and values[50] is not None else None)
        sensitivity_rows.append(row)
    # one shared row so the table reads as a single comparison
    shared = {key: None for key in sensitivity_rows[0]}
    shared["arm"] = "SHARED"
    shared["valid_gain_100"] = gains["valid_100"]["gain_point"]
    shared["valid_gain_50"] = gains["valid_50"]["gain_point"]
    shared["test_gain_100"] = gains["test_100"]["gain_point"]
    shared["test_gain_50"] = gains["test_50"]["gain_point"]
    shared["delta_gain_valid_50_minus_100"] = gains["size_delta"]["gain_50_minus_gain_100_valid"]
    shared["delta_gain_test_50_minus_100"] = gains["size_delta"]["gain_50_minus_gain_100_test"]
    sensitivity_rows.append(shared)
    _write_csv(out / "size_sensitivity.csv", sensitivity_rows)

    reference = reference_analysis(runs, prep)
    C.write_json(out / "reference_analysis.json", reference)
    C.write_json(Path(__file__).resolve().parents[1] / "analysis" / "reference_analysis.json",
                 reference)

    summary = {
        "format": "css-v0-summary-v1",
        "runs": {f"{arm}_{size}": {
            "run_id": runs[(arm, size)]["run_id"],
            "best_epoch": runs[(arm, size)]["best"]["epoch"],
            "epochs": runs[(arm, size)]["epochs"],
            "wall_seconds": runs[(arm, size)]["wall_seconds"],
            "gpu_hours": runs[(arm, size)].get("gpu_hours"),
            "git_commit": runs[(arm, size)].get("git_commit"),
        } for size in SIZES for arm in ARMS},
        "main_table": table_rows,
        "gains": {key: gains[key] for key in gains if key not in ("format",)},
        "totals": {
            "formal_gpu_hours": float(sum(runs[key].get("gpu_hours", 0.0)
                                          for key in runs)),
            "formal_wall_seconds_sum": float(sum(runs[key]["wall_seconds"]
                                                 for key in runs)),
        },
    }
    C.write_json(out / "summary.json", summary)
    print(f"wrote {out}/main_table.csv, curve.csv, curves_all.csv, "
          f"gain_bootstrap.json, size_sensitivity.csv; analysis/reference_analysis.json",
          flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    parser.add_argument("--prep-dir", default=str(C.PREP_DIR))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    main(args.results_dir, args.prep_dir, args.out_dir)
