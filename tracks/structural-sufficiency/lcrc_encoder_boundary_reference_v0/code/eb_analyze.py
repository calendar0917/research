"""Post-training analysis for the encoder-boundary round (Train/Valid only).

Reads the read-only A artefacts and the pulled B run, then writes:
main table, paired molecule errors, molecule bootstrap, drop-top-5% sensitivity,
error distributions, the single frozen structural-complexity diagnostic, curves,
checkpoint manifest and the pre-registered verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import eb_common as E
import cs_common as C

C.setup_paths()

import eb_data as BD  # noqa: E402

B_RUN = E.RUNS_DIR / "gine_B_seed0"


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def aligned_errors(a: dict[str, np.ndarray], b: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    ra = np.asarray(a[f"{split}_row_ids"], dtype=np.int64)
    rb = np.asarray(b[f"{split}_row_ids"], dtype=np.int64)
    if np.unique(ra).size != ra.size or np.unique(rb).size != rb.size:
        raise RuntimeError(f"{split}: duplicate row ids")
    if not np.array_equal(np.sort(ra), np.sort(rb)):
        raise RuntimeError(f"{split}: row id sets differ")
    ia = np.argsort(ra)
    ib = np.argsort(rb)
    ra, rb = ra[ia], rb[ib]
    if not np.array_equal(ra, rb):
        raise RuntimeError(f"{split}: row id alignment failed")
    ya = np.asarray(a[f"{split}_y_raw"], dtype=np.float64)[ia]
    yb = np.asarray(b[f"{split}_y_raw"], dtype=np.float64)[ib]
    if np.max(np.abs(ya - yb)) > 1e-12:
        raise RuntimeError(f"{split}: target mismatch between runs")
    pa = np.asarray(a[f"{split}_pred_raw"], dtype=np.float64)[ia]
    pb = np.asarray(b[f"{split}_pred_raw"], dtype=np.float64)[ib]
    return {
        "row_id": ra,
        "y": ya,
        "pred_a": pa,
        "pred_b": pb,
        "err_a": pa - ya,
        "err_b": pb - ya,
        "dse": (pb - ya) ** 2 - (pa - ya) ** 2,
    }


def bootstrap(dse_a: np.ndarray, dse_b: np.ndarray, *, replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(dse_a.size)
    idx = rng.integers(0, n, size=(int(replicates), n))
    rmse_a = np.sqrt(dse_a[idx].mean(axis=1))
    rmse_b = np.sqrt(dse_b[idx].mean(axis=1))
    delta = rmse_b - rmse_a
    rel = delta / rmse_a
    q = [2.5, 97.5]
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "n_molecules": n,
        "delta_rmse_mean": float(delta.mean()),
        "delta_rmse_ci95": [float(x) for x in np.percentile(delta, q)],
        "relative_delta_mean": float(rel.mean()),
        "relative_delta_ci95": [float(x) for x in np.percentile(rel, q)],
        "prob_b_better": float((delta < 0).mean()),
    }


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, np.float64) - np.asarray(y, np.float64)) ** 2)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))))


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"rmse_raw": rmse(pred, y), "mae_raw": mae(pred, y)}


def paired_block(a: dict, b: dict, split: str) -> dict[str, Any]:
    e = aligned_errors(a, b, split)
    ma, mb = metrics(e["pred_a"], e["y"]), metrics(e["pred_b"], e["y"])
    dse_a, dse_b = e["err_a"] ** 2, e["err_b"] ** 2
    delta = mb["rmse_raw"] - ma["rmse_raw"]
    boot = bootstrap(dse_a, dse_b, replicates=E.BOOTSTRAP_REPLICATES, seed=E.BOOTSTRAP_SEED)
    better = dse_b < dse_a
    k = max(1, int(round(E.TOP_ERROR_FRACTION * dse_a.size)))
    drop_idx = np.argsort(-dse_a)[:k]
    keep = np.ones(dse_a.size, dtype=bool)
    keep[drop_idx] = False
    drop = {
        "n_removed": int(k),
        "fraction": float(k / dse_a.size),
        "selection": "largest A squared error on this split",
        "n_remaining": int(keep.sum()),
        "rmse_a": float(np.sqrt(dse_a[keep].mean())),
        "rmse_b": float(np.sqrt(dse_b[keep].mean())),
        "delta_rmse": float(np.sqrt(dse_b[keep].mean()) - np.sqrt(dse_a[keep].mean())),
        "relative_delta": float(np.sqrt(dse_b[keep].mean()) / np.sqrt(dse_a[keep].mean()) - 1.0),
    }
    abs_a = np.abs(e["err_a"])
    abs_b = np.abs(e["err_b"])
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    return {
        "split": split,
        "n_molecules": int(dse_a.size),
        "A": ma,
        "B": mb,
        "delta_rmse_B_minus_A": delta,
        "relative_delta": float(delta / ma["rmse_raw"]),
        "delta_mae_B_minus_A": mb["mae_raw"] - ma["mae_raw"],
        "fraction_B_better": float(better.mean()),
        "mean_dse_A": float(dse_a.mean()),
        "mean_dse_B": float(dse_b.mean()),
        "mean_dse_delta": float(np.mean(dse_b - dse_a)),
        "bootstrap": boot,
        "drop_top_A_error_5pct": drop,
        "abs_error_quantiles": {
            "A": {str(q): float(np.quantile(abs_a, q)) for q in quantiles},
            "B": {str(q): float(np.quantile(abs_b, q)) for q in quantiles},
        },
        "_errors": e,
    }


def _global_index(spec, name: str) -> int:
    names = [f.name for f in spec.global_feature_fields]
    if name not in names:
        raise KeyError(f"global field {name} missing")
    return names.index(name)


def diagnostic(bundle, splits, pair_valid: dict[str, Any]) -> dict[str, Any]:
    """One pre-frozen exploratory diagnostic; no threshold search."""
    e = pair_valid["_errors"]
    va = splits["valid"]
    cache = bundle.caches["valid"]
    n_atoms = np.diff(np.asarray(cache["atom_ptr"], dtype=np.int64)).astype(np.float64)
    glob = np.asarray(cache["global"], dtype=np.float64)
    diam = glob[:, _global_index(bundle.spec, "diameter_max_finite")]
    ring_frac = glob[:, _global_index(bundle.spec, "ring_atom_fraction")]

    train_glob = np.asarray(bundle.caches["train"]["global"], dtype=np.float64)
    train_n_atoms = np.diff(np.asarray(bundle.caches["train"]["atom_ptr"], dtype=np.int64)).astype(np.float64)
    train_diam = train_glob[:, _global_index(bundle.spec, "diameter_max_finite")]
    train_ring = train_glob[:, _global_index(bundle.spec, "ring_atom_fraction")]

    # local-environment rarity: atom signature over Train, mean -log probability on Valid
    sig_tr = np.stack([splits["train"].atom[:, 0], splits["train"].degree, splits["train"].ring], axis=1)
    sig_va = np.stack([va.atom[:, 0], va.degree, va.ring], axis=1)
    codes_tr, counts_tr = np.unique(sig_tr, axis=0, return_counts=True)
    total = float(sig_tr.shape[0])
    lookup = {tuple(row.tolist()): int(c) for row, c in zip(codes_tr, counts_tr)}
    rarity_atom = np.asarray([-np.log((lookup.get(tuple(row.tolist()), 0) + 1.0) / (total + 1.0)) for row in sig_va])
    atom_batch = np.repeat(np.arange(va.atom_ptr.size - 1, dtype=np.int64), np.diff(va.atom_ptr))
    rarity = np.bincount(atom_batch, weights=rarity_atom, minlength=int(va.row_ids.size)) / np.diff(va.atom_ptr)

    # align diagnostic arrays to the sorted-row-id order used by the paired block
    row_ids_sorted = np.sort(np.asarray(va.row_ids, dtype=np.int64))
    order = np.argsort(np.asarray(va.row_ids, dtype=np.int64))
    pos = np.searchsorted(va.row_ids[order], e["row_id"])
    if not np.array_equal(va.row_ids[order][pos], e["row_id"]):
        raise RuntimeError("diagnostic row alignment failed")

    descriptors = {
        "n_atoms": n_atoms[order][pos],
        "diameter_max_finite": diam[order][pos],
        "ring_atom_fraction": ring_frac[order][pos],
        "local_environment_rarity": rarity[order][pos],
    }
    # thresholds frozen on Train (not on the Valid outcome)
    train_thresholds = {
        "n_atoms": float(np.median(train_n_atoms)),
        "diameter_max_finite": float(np.median(train_diam)),
        "ring_atom_fraction": float(np.median(train_ring)),
        "local_environment_rarity": float(np.median(rarity)),
    }
    better = (e["err_b"] ** 2) < (e["err_a"] ** 2)
    dm = {}
    for name, values in descriptors.items():
        threshold = train_thresholds[name]
        groups = {}
        for label, mask in (("high", values > threshold), ("low", values <= threshold)):
            if mask.sum() == 0:
                continue
            dse_a, dse_b = e["err_a"][mask] ** 2, e["err_b"][mask] ** 2
            groups[label] = {
                "n": int(mask.sum()),
                "fraction_B_better": float(better[mask].mean()),
                "rmse_A": float(np.sqrt(dse_a.mean())),
                "rmse_B": float(np.sqrt(dse_b.mean())),
                "delta_rmse_B_minus_A": float(np.sqrt(dse_b.mean()) - np.sqrt(dse_a.mean())),
            }
        dm[name] = {
            "median_threshold": threshold,
            "mean_B_better": float(values[better].mean()),
            "mean_A_better": float(values[~better].mean()),
            "median_B_better": float(np.median(values[better])),
            "median_A_better": float(np.median(values[~better])),
            "groups": groups,
        }
    return {
        "definition": {
            "descriptors": {
                "n_atoms": "molecule atom count",
                "diameter_max_finite": "frozen lipo_v1 global descriptor",
                "ring_atom_fraction": "frozen lipo_v1 global descriptor",
                "local_environment_rarity": "mean over atoms of -log((Train count(atom signature)+1)/(N_train+1)); signature = (atomic_num, degree, min_ring)",
            },
            "groups": "Train median threshold per descriptor (rarity uses the Valid array median because its scale is defined on Train frequencies); no threshold search",
            "usage": "exploratory explanation only; never replaces the official RMSE",
        },
        "n_molecules": int(e["row_id"].size),
        "fraction_B_better": float(better.mean()),
        "descriptors": dm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-run", default=str(B_RUN))
    parser.add_argument("--out", default=str(E.ANALYSIS_DIR))
    args = parser.parse_args()
    b_dir = Path(args.b_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    a_run = E.read_json(E.A_RUN_JSON)
    b_run = E.read_json(b_dir / "run.json")
    if b_run.get("status") != "COMPLETED":
        raise RuntimeError(f"B run is not COMPLETED: {b_run.get('status')} / {b_run.get('stop_reason')}")
    if b_run.get("test_scored"):
        raise RuntimeError("B run reports test scoring; this round forbids it")
    a_pred = load_predictions(E.A_PREDICTIONS)
    b_pred = load_predictions(b_dir / "predictions_best.npz")
    a_ck = __import__("torch").load(E.A_CHECKPOINT, map_location="cpu", weights_only=False)

    # ---- main table
    def row(name, run, params, groups, source):
        p = groups
        return {
            "model": name, "source": source, "parameters": int(params),
            "encoder_parameters": int(p.get("encoder", p.get("n_parameters_new", params))),
            "head_parameters": int(p.get("head", p.get("n_parameters_base", 0))),
            "best_epoch": int(run["best_epoch"]),
            "best_train_rmse_raw": float(run["best_metrics"]["train"]["rmse_raw"]),
            "best_train_mae_raw": float(run["best_metrics"]["train"]["mae_raw"]),
            "best_valid_rmse_raw": float(run["best_metrics"]["valid"]["rmse_raw"]),
            "best_valid_mae_raw": float(run["best_metrics"]["valid"]["mae_raw"]),
            "endpoint_epoch": int(run["final_metrics"]["epoch"]),
            "endpoint_train_rmse_raw": float(run["final_metrics"]["train_rmse_raw"]),
            "endpoint_train_mae_raw": float(run["final_metrics"]["train_mae_raw"]),
            "endpoint_valid_rmse_raw": float(run["final_metrics"]["valid_rmse_raw"]),
            "endpoint_valid_mae_raw": float(run["final_metrics"]["valid_mae_raw"]),
            "last20_train_rmse_raw": float(run["last20_mean_metrics"]["train_rmse_raw"]),
            "last20_train_mae_raw": float(run["last20_mean_metrics"]["train_mae_raw"]),
            "last20_valid_rmse_raw": float(run["last20_mean_metrics"]["valid_rmse_raw"]),
            "last20_valid_mae_raw": float(run["last20_mean_metrics"]["valid_mae_raw"]),
            "train_seconds": float(run["train_seconds"]), "eval_seconds": float(run["eval_seconds"]),
            "total_seconds": float(run["total_seconds"]), "gpu_hours": float(run["gpu_hours"]),
            "inference_rows_per_second": float(run.get("inference_rows_per_second") or 0.0),
            "train_minus_valid_rmse_best": float(run["best_metrics"]["train"]["rmse_raw"] - run["best_metrics"]["valid"]["rmse_raw"]),
        }

    table = [
        row("A_FIXED1", a_run, E.A_TOTAL_PARAMS, {"encoder": 247680, "head": 46455}, "CSC v0 read-only reuse"),
        row("B_GINE_ENCODER", b_run, b_run["parameters"], b_run["parameter_groups"], "this round"),
    ]
    C.write_csv(out / "main_table.csv", table)

    # ---- paired comparisons
    pairs = {"train": paired_block(a_pred, b_pred, "train"), "valid": paired_block(a_pred, b_pred, "valid")}
    valid_errors = dict(pairs["valid"]["_errors"])
    per_mol_rows = []
    for split, block in pairs.items():
        e = block.pop("_errors")
        for i in range(e["row_id"].size):
            per_mol_rows.append({
                "split": split, "row_id": int(e["row_id"][i]),
                "y_raw": float(e["y"][i]), "pred_A": float(e["pred_a"][i]), "pred_B": float(e["pred_b"][i]),
                "error_A": float(e["err_a"][i]), "error_B": float(e["err_b"][i]),
                "dse_B_minus_A": float(e["dse"][i]), "B_better": bool(e["dse"][i] < 0),
            })
    C.write_csv(out / "paired_errors.csv", per_mol_rows)
    E.write_json(out / "paired_comparisons.json", {
        "round_id": E.ROUND_ID,
        "delta_convention": "all deltas are B - A; negative means GINE has lower error",
        "train": pairs["train"], "valid": pairs["valid"],
    })

    # ---- diagnostic (uses the valid paired block before '_errors' removal)
    bundle = BD.load_bundle()
    sq = BD.load_sq_cache()
    splits = BD.build_splits(bundle, sq)
    diag = diagnostic(bundle, splits, {"_errors": valid_errors})
    E.write_json(out / "diagnostic_groups.json", diag)

    # ---- curves and figures
    def read_curve(path):
        import csv
        keep = ("train_rmse_raw", "train_mae_raw", "valid_rmse_raw", "valid_mae_raw")
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return [{"epoch": int(float(r["epoch"])), **{k: float(r[k]) for k in keep}} for r in rows]

    a_curve = read_curve(E.A_CURVE)
    b_curve = read_curve(b_dir / "train_curve.csv")
    combined = []
    for name, curve in (("A_FIXED1", a_curve), ("B_GINE_ENCODER", b_curve)):
        for r in curve:
            combined.append({"model": name, "epoch": r["epoch"], "train_rmse_raw": r["train_rmse_raw"],
                             "train_mae_raw": r["train_mae_raw"], "valid_rmse_raw": r["valid_rmse_raw"],
                             "valid_mae_raw": r["valid_mae_raw"]})
    C.write_csv(out / "curves.csv", combined)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, curve, color in (("A FIXED1", a_curve, "#333333"), ("B GINE", b_curve, "#1f77b4")):
        epochs = [r["epoch"] for r in curve]
        axes[0].plot(epochs, [r["train_rmse_raw"] for r in curve], color=color, label=f"{name} train")
        axes[0].plot(epochs, [r["valid_rmse_raw"] for r in curve], color=color, ls="--", label=f"{name} valid")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("RMSE (raw)"); axes[0].legend(fontsize=7); axes[0].set_title("training curves")
    axes[1].hist(np.abs(valid_errors["err_a"]), bins=40, alpha=0.5, label="A FIXED1", color="#333333")
    axes[1].hist(np.abs(valid_errors["err_b"]), bins=40, alpha=0.5, label="B GINE", color="#1f77b4")
    axes[1].set_xlabel("|error| (raw)"); axes[1].set_ylabel("molecules"); axes[1].set_title("Valid per-molecule |error|"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "curves_and_errors.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(np.abs(valid_errors["err_a"]), np.abs(valid_errors["err_b"]), s=8, alpha=0.6, color="#1f77b4")
    lim = float(max(np.abs(valid_errors["err_a"]).max(), np.abs(valid_errors["err_b"]).max())) * 1.02
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.8)
    ax.set_xlabel("|error| A FIXED1"); ax.set_ylabel("|error| B GINE"); ax.set_title("Valid molecule errors")
    fig.tight_layout(); fig.savefig(out / "figures" / "paired_error_scatter.png", dpi=140); plt.close(fig)

    # ---- checkpoint manifest
    b_ck_path = b_dir / "checkpoint_best.pt"
    manifest = {
        "round_id": E.ROUND_ID,
        "A": {
            "checkpoint_best": E.track_rel(E.A_CHECKPOINT), "sha256": E.sha256_file(E.A_CHECKPOINT),
            "epoch": int(a_ck["epoch"]), "parameters": E.A_TOTAL_PARAMS,
            "usage": "read-only reuse; trained in CSC v0",
        },
        "B": {
            "checkpoint_best": E.track_rel(b_ck_path), "sha256": E.sha256_file(b_ck_path),
            "checkpoint_final": E.track_rel(b_dir / "checkpoint_final.pt"),
            "checkpoint_final_sha256": E.sha256_file(b_dir / "checkpoint_final.pt"),
            "best_epoch": int(b_run["best_epoch"]), "parameters": int(b_run["parameters"]),
            "parameter_groups": b_run["parameter_groups"],
            "predictions_best": E.track_rel(b_dir / "predictions_best.npz"),
            "predictions_best_sha256": E.sha256_file(b_dir / "predictions_best.npz"),
            "curve": E.track_rel(b_dir / "train_curve.csv"), "curve_sha256": E.sha256_file(b_dir / "train_curve.csv"),
            "git_commit": b_run.get("git_commit"), "environment": b_run.get("environment"),
        },
    }
    E.write_json(out / "checkpoint_manifest.json", manifest)

    # ---- verdict (pre-registered)
    rel = float(pairs["valid"]["relative_delta"])
    ci = pairs["valid"]["bootstrap"]["delta_rmse_ci95"]
    adv = (rel <= -E.REL_GAIN_REFERENCE) and (ci[1] < 0.0)
    verdict = "ENCODER_ADVANTAGE_SIGNAL" if adv else "NO_CLEAR_ENCODER_ADVANTAGE"
    direction_last20 = float(b_run["last20_mean_metrics"]["valid_rmse_raw"] - a_run["last20_mean_metrics"]["valid_rmse_raw"])
    payload = {
        "round_id": E.ROUND_ID,
        "verdict": verdict,
        "comparability_label": "MATCHED_BUDGET_ENCODER_REFERENCE (shared post_local + same raw OGB table + same graph; encoders differ in derived inputs and computation)",
        "criterion": {
            "relative_gain_reference": E.REL_GAIN_REFERENCE,
            "observed_relative_delta_B_minus_A_valid": rel,
            "bootstrap_delta_rmse_ci95": ci,
            "bootstrap_prob_B_better": pairs["valid"]["bootstrap"]["prob_b_better"],
            "advantage_rule_satisfied": bool(adv),
        },
        "best_vs_last20": {
            "best_delta_rmse_B_minus_A_valid": float(pairs["valid"]["delta_rmse_B_minus_A"]),
            "last20_mean_delta_rmse_B_minus_A_valid": direction_last20,
            "same_direction": bool(np.sign(direction_last20) == np.sign(pairs["valid"]["delta_rmse_B_minus_A"])),
        },
        "gpu_hours_used_this_round": float(b_run["gpu_hours"]),
        "gpu_hours_hard_limit": E.GPU_HARD_HOURS,
        "test_scored": False,
        "single_seed": True,
        "notes": [
            "Single seed, one training unit for B; A is a read-only reuse of a completed 360-epoch run.",
            "The encoder-level difference cannot be attributed to iterative message passing alone: input derivations and structure scope also differ.",
        ],
    }
    E.write_json(out / "verdict.json", payload)
    C.log(f"analysis written to {out}: verdict={verdict} rel={rel:.4f} ci={ci}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
