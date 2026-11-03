"""Analysis of the frozen Validation and Test results (no model is touched).

Reads only artefacts written earlier:

* ``analysis/checkpoint_manifest.json``   (frozen pre-Test objects, Validation metrics)
* ``analysis/test_scores.json``          (the single one-shot Test pass)
* ``analysis/*_checks.json``             (gate evidence)
* ``runs/*/run.json``                    (timings, curves, provenance)

Writes

    analysis/validation_metrics.json
    analysis/test_metrics.json
    analysis/paired_bootstrap.json
    analysis/gpu_ledger.json
    analysis/provenance.json
    analysis/scaffolds.json
    analysis/report_tables.md
    analysis/decision.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

import numpy as np

import si_common as C


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "sd_ddof1": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "sd_ddof0": float(v.std(ddof=0)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
        "values": [float(x) for x in v],
    }


def _units_by(manifest: Mapping[str, Any], key: str = "arm") -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for unit in manifest["units"]:
        out[(str(unit[key]), int(unit["seed"]))] = unit
    return out


def _rocauc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y, dtype=np.float64), np.asarray(score, dtype=np.float64)))


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y, dtype=np.float64), np.asarray(score, dtype=np.float64)))


# ---------------------------------------------------------------------------
# scaffolds (Bemis-Murcko) for the paired bootstrap
# ---------------------------------------------------------------------------


def scaffolds_for(smiles: Sequence[str], *, key: str) -> dict[str, Any]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    groups: list[str] = []
    failures = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            failures += 1
            groups.append(f"__PARSE_FAILED__{failures}")
            continue
        try:
            groups.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        except Exception:  # noqa: BLE001
            failures += 1
            groups.append(f"__SCAFFOLD_FAILED__{failures}")
    uniq = sorted(set(groups))
    index = {g: i for i, g in enumerate(uniq)}
    codes = np.asarray([index[g] for g in groups], dtype=np.int64)
    sizes = np.bincount(codes)
    return {
        "split": key,
        "n_molecules": len(groups),
        "n_groups": int(len(uniq)),
        "n_parse_or_scaffold_failures": int(failures),
        "group_sizes": {
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "n_singletons": int((sizes == 1).sum()),
        },
        "codes_sha256": C.sha256_bytes(codes.tobytes()),
        "groups": groups,
        "source": "RDKit MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)",
    }


# ---------------------------------------------------------------------------
# paired scaffold bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(
    preds: Mapping[str, Mapping[int, np.ndarray]],
    y: np.ndarray,
    codes: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    codes = np.asarray(codes, dtype=np.int64)
    n_groups = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    row_groups = [order[bounds[g] : bounds[g + 1]] for g in range(n_groups)]
    arms = sorted(preds)
    point: dict[str, float] = {}
    per_arm_point = {
        arm: float(np.mean([_rocauc(y, preds[arm][s]) for s in sorted(preds[arm])])) for arm in arms
    }
    for a, b in C.CONTRASTS:
        point[f"{a}-{b}"] = per_arm_point[a] - per_arm_point[b]
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in C.CONTRASTS}
    skipped = 0
    for _ in range(int(replicates)):
        picked = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([row_groups[g] for g in picked])
        yy = y[rows]
        if yy.min() == yy.max():
            skipped += 1
            continue
        per_arm = {arm: float(np.mean([_rocauc(yy, preds[arm][s][rows]) for s in sorted(preds[arm])])) for arm in arms}
        for a, b in C.CONTRASTS:
            draws[f"{a}-{b}"].append(per_arm[a] - per_arm[b])
    out: dict[str, Any] = {
        "seed": int(seed),
        "replicates_requested": int(replicates),
        "replicates_used": int(replicates - skipped),
        "replicates_skipped_single_class": int(skipped),
        "n_molecules": int(y.size),
        "n_scaffold_groups": int(n_groups),
        "resampling_unit": "Bemis-Murcko scaffold group (same resampled rows for every arm and seed)",
        "statistic": "3-seed mean ROC-AUC difference",
        "seed_arm_roc_auc_point": per_arm_point,
        "point_differences": point,
        "contrasts": {},
    }
    for name, values in draws.items():
        v = np.asarray(values, dtype=np.float64)
        out["contrasts"][name] = {
            "point": float(point[name]),
            "bootstrap_mean": float(v.mean()) if v.size else None,
            "ci95_percentile": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None,
            "n_used": int(v.size),
            "n_positive": int((v > 0).sum()),
            "fraction_positive": float((v > 0).mean()) if v.size else None,
        }
    return out


# ---------------------------------------------------------------------------
# ledger / provenance
# ---------------------------------------------------------------------------


def gpu_ledger() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(C.RUNS.rglob("run.json")):
        record = C.read_json(path)
        rows.append(
            {
                "unit": C.track_rel(path.parent.relative_to(C.RUNS)).replace("\\", "/"),
                "arm": record.get("arm"),
                "seed": int(record.get("seed", -1)),
                "smoke": bool(record.get("smoke")),
                "tag": record.get("tag", ""),
                "epochs_run": int(record.get("epochs_run", 0)),
                "gpu_hours": float(record.get("timing", {}).get("gpu_hours_estimate", 0.0)),
                "train_wall_seconds": float(record.get("timing", {}).get("train_wall_seconds", 0.0)),
                "mean_epoch_seconds": float(record.get("timing", {}).get("mean_epoch_seconds", 0.0)),
                "gpu_peak_allocated_bytes": int(record.get("timing", {}).get("gpu_peak_allocated_bytes", 0)),
                "hostname": record.get("env", {}).get("hostname"),
                "git_commit": record.get("git_commit"),
            }
        )
    total = float(sum(r["gpu_hours"] for r in rows))
    formal = float(sum(r["gpu_hours"] for r in rows if not r["smoke"]))
    preflight = total - formal
    return {
        "schema": "lcrc-sparse-identity-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "gpu_soft_cap_hours": float(C.GPU_SOFT_CAP_HOURS),
        "gpu_hard_cap_hours": float(C.GPU_HARD_CAP_HOURS),
        "units": rows,
        "n_units": len(rows),
        "formal_gpu_hours": formal,
        "preflight_gpu_hours": preflight,
        "total_gpu_hours": total,
        "within_soft_cap": bool(total <= C.GPU_SOFT_CAP_HOURS),
        "within_hard_cap": bool(total <= C.GPU_HARD_CAP_HOURS),
        "billing": "two A100 cards billed together; gpu_hours = train wall clock of each process",
        "note": "Test forward passes ran on CPU and are not billed; feature construction is CPU (local).",
    }


TRAINING_PATH_FILES = (
    "code/si_common.py",
    "code/si_features.py",
    "code/si_model.py",
    "code/si_train.py",
    "code/si_sweep.py",
    "code/si_gates.py",
)


def training_code_identity(commits: Sequence[str]) -> dict[str, Any]:
    """Prove the files that trained are byte-identical to the committed ones.

    The training jobs ran at one or more recorded ``git_commit`` values.  For every
    training-path file this compares the bytes in that commit with the bytes on disk
    now, so any post-hoc edit to the training path would be visible.
    """
    import subprocess

    out: dict[str, Any] = {"training_commits": sorted(set(commits)), "files": {}, "all_identical": True}
    for rel in TRAINING_PATH_FILES:
        path = C.TRACK / rel
        current = C.sha256_file(path) if path.exists() else None
        per_commit: dict[str, Any] = {}
        for commit in sorted(set(commits)):
            try:
                blob = subprocess.check_output(
                    ["git", "show", f"{commit}:{C.track_rel(path)}"], cwd=C.REPO, text=False
                )
                digest = C.sha256_bytes(blob)
            except subprocess.CalledProcessError:
                digest = None
            per_commit[commit] = {"sha256": digest, "matches_current": bool(digest == current)}
            out["all_identical"] = out["all_identical"] and bool(digest == current)
        out["files"][rel] = {"current_sha256": current, "per_commit": per_commit}
    return out


def historical_reference() -> dict[str, Any]:
    base = (
        C.REPO
        / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/baseline/ogbg-molhiv.json"
    )
    payload = C.read_json(base)
    test_path = base.parent / "ogbg-molhiv_test.json"
    test_payload = C.read_json(test_path) if test_path.exists() else {}
    return {
        "ecfp4_xgboost_baseline": {
            "source": C.track_rel(base),
            "sha256": C.sha256_file(base),
            "features": payload["frozen_config"]["features"],
            "valid_roc_auc": float(payload["metrics"]["valid"]["roc_auc"]),
            "valid_pr_auc": float(payload["metrics"]["valid"]["pr_auc"]),
            "test_roc_auc": float(test_payload["metrics"]["test"]["roc_auc"]) if test_payload else None,
            "test_pr_auc": float(test_payload["metrics"]["test"]["pr_auc"]) if test_payload else None,
            "n_seeds": 1,
            "comparability": "descriptive only: single seed, different model family and features, "
            "not a paired run in this round",
        },
        "ptb_molhiv_a0_history": {
            "source": C.track_rel(C.PTB / "analysis/molecular_results.csv"),
            "sha256": C.sha256_file(C.PTB / "analysis/molecular_results.csv"),
            "valid_roc_auc": [0.8314089138741917, 0.8356741745051929, 0.8376414609053496],
            "test_roc_auc": [0.7653488866142646, 0.7736128546321868, 0.775857007667201],
            "valid_pr_auc": [0.3131555159188919, 0.40047098860814556, 0.4038194003418263],
            "test_pr_auc": [0.24693590534319748, 0.3122057019660126, 0.2984051614523935],
            "best_epoch": [53, 32, 29],
            "note": "same architecture and recipe, 100 epochs, earlier round; used only as a replication "
            "sanity check for the A0 arm, never as the paired baseline",
        },
    }


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def build_tables(valid: Mapping[str, Any], test: Mapping[str, Any]) -> str:
    lines: list[str] = []
    arms = [C.MODEL_A0, C.MODEL_TRUE, C.MODEL_NULL, C.MODEL_FOLD]
    lines.append("### Validation (official OGB scaffold Validation split)\n")
    lines.append("| arm | seed | ROC-AUC | AP | BCE | best epoch | train ROC-AUC | train−valid |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for seed in C.SEEDS:
            u = valid["units"][f"{arm}_s{seed}"]
            lines.append(
                f"| {arm} | {seed} | {_fmt(u['roc_auc'])} | {_fmt(u['pr_auc'])} | {_fmt(u['bce'])} | "
                f"{u['best_epoch']} | {_fmt(u['train_roc_auc'])} | {_fmt(u['train_minus_valid_roc_auc'])} |"
            )
    sp = valid["sparse_only"]
    lines.append(
        f"| SPARSE_ONLY | – | {_fmt(sp['roc_auc'])} | {_fmt(sp['pr_auc'])} | {_fmt(sp['bce'])} | – | "
        f"{_fmt(sp['train_roc_auc'])} | {_fmt(sp['train_roc_auc'] - sp['roc_auc'])} |"
    )
    lines.append("\n### Test (one-shot, EXPOSED_TEST_EXPLORATORY)\n")
    lines.append("| arm | seed | ROC-AUC | OGB ROC-AUC | AP | best epoch |")
    lines.append("|---|---|---|---|---|---|")
    for arm in arms:
        for seed in C.SEEDS:
            u = test["units"][f"{arm}_s{seed}"]
            lines.append(
                f"| {arm} | {seed} | {_fmt(u['roc_auc'])} | {_fmt(u['official_ogb_roc_auc'])} | "
                f"{_fmt(u['pr_auc'])} | {u['best_epoch']} |"
            )
    spt = test["sparse_only"]
    lines.append(f"| SPARSE_ONLY | – | {_fmt(spt['roc_auc'])} | {_fmt(spt['official_ogb_roc_auc'])} | {_fmt(spt['pr_auc'])} | – |")
    lines.append("\n### Three-seed means\n")
    lines.append("| arm | valid ROC-AUC | valid AP | test ROC-AUC | test AP |")
    lines.append("|---|---|---|---|---|")
    for arm in arms + [C.MODEL_SPARSE]:
        v = valid["summary"][arm]
        t = test["summary"][arm]
        lines.append(
            f"| {arm} | {_fmt(v['roc_auc']['mean'])} ± {_fmt(v['roc_auc']['sd_ddof1'])} | "
            f"{_fmt(v['pr_auc']['mean'])} ± {_fmt(v['pr_auc']['sd_ddof1'])} | "
            f"{_fmt(t['roc_auc']['mean'])} ± {_fmt(t['roc_auc']['sd_ddof1'])} | "
            f"{_fmt(t['pr_auc']['mean'])} ± {_fmt(t['pr_auc']['sd_ddof1'])} |"
        )
    lines.append("\n### Paired differences (ROC-AUC)\n")
    lines.append("| contrast | valid per seed | valid mean | test per seed | test mean |")
    lines.append("|---|---|---|---|---|")
    for a, b in C.CONTRASTS:
        name = f"{a} − {b}"
        v = valid["contrasts"][f"{a}-{b}"]
        t = test["contrasts"][f"{a}-{b}"]
        lines.append(
            f"| {name} | " + ", ".join(_fmt(x) for x in v["values"]) + f" | {_fmt(v['mean'])} | "
            + ", ".join(_fmt(x) for x in t["values"]) + f" | {_fmt(t['mean'])} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validation/Test analysis of the frozen results")
    ap.parse_args(argv)
    manifest = C.read_json(C.ANALYSIS / "checkpoint_manifest.json")
    scores = C.read_json(C.ANALYSIS / "test_scores.json")
    gates = C.read_json(C.ANALYSIS / "correctness_gates.json")

    units = _units_by(manifest)

    # ---- Validation --------------------------------------------------------
    valid_units: dict[str, Any] = {}
    for (arm, seed), unit in units.items():
        valid_units[f"{arm}_s{seed}"] = {
            "arm": arm,
            "seed": seed,
            "roc_auc": float(unit["valid_metrics"]["roc_auc"]),
            "pr_auc": float(unit["valid_metrics"]["pr_auc"]),
            "bce": float(unit["valid_metrics"]["bce"]),
            "accuracy_at_0_5": float(unit["valid_metrics"]["accuracy_at_0.5"]),
            "train_roc_auc": float(unit["train_metrics"]["roc_auc"]),
            "train_pr_auc": float(unit["train_metrics"]["pr_auc"]),
            "train_minus_valid_roc_auc": float(unit["gap"]["train_minus_valid_roc_auc"]),
            "best_epoch": int(unit["best_epoch"]),
            "parameters": int(unit["parameters"]),
            "gpu_hours": float(unit["gpu_hours"]),
            "gpu_peak_allocated_bytes": int(unit["gpu_peak_allocated_bytes"]),
            "mean_epoch_seconds": float(unit["mean_epoch_seconds"]),
            "branch_norms": unit["branch"]["norms"],
            "branch_contribution_valid": unit["branch"]["contribution"]["valid"],
            "branch_contribution_train": unit["branch"]["contribution"]["train"],
            "feature_meta": unit["feature_meta"],
        }
    sparse = manifest["sparse_only"]
    valid_sparse = {
        "arm": C.MODEL_SPARSE,
        "best_epoch": 0,
        "roc_auc": float(sparse["valid_metrics"]["roc_auc"]),
        "pr_auc": float(sparse["valid_metrics"]["pr_auc"]),
        "bce": float(sparse["valid_metrics"]["bce"]),
        "train_roc_auc": float(sparse["train_metrics"]["roc_auc"]),
        "train_pr_auc": float(sparse["train_metrics"]["pr_auc"]),
        "coef_l1": float(sparse["coef_l1"]),
        "coef_l2": float(sparse["coef_l2"]),
        "coef_n_nonzero": int(sparse["coef_n_nonzero"]),
        "n_iter": int(sparse["n_iter"]),
        "hyperparameters": sparse["hyperparameters"],
        "note": "single CPU fit on official Train only; no training seeds",
    }

    def summary_from(getter) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm in (C.MODEL_A0, C.MODEL_TRUE, C.MODEL_NULL, C.MODEL_FOLD):
            out[arm] = {
                "roc_auc": _mean_sd([getter(arm, s, "roc_auc") for s in C.SEEDS]),
                "pr_auc": _mean_sd([getter(arm, s, "pr_auc") for s in C.SEEDS]),
                "best_epoch": _mean_sd([float(getter(arm, s, "best_epoch")) for s in C.SEEDS]),
            }
        out[C.MODEL_SPARSE] = {
            "roc_auc": _mean_sd([getter(C.MODEL_SPARSE, 0, "roc_auc")]),
            "pr_auc": _mean_sd([getter(C.MODEL_SPARSE, 0, "pr_auc")]),
            "best_epoch": _mean_sd([0.0]),
        }
        return out

    def valid_get(arm: str, seed: int, key: str) -> Any:
        return valid_sparse[key] if arm == C.MODEL_SPARSE else valid_units[f"{arm}_s{seed}"][key]

    valid_summary = summary_from(valid_get)
    hist_a0 = historical_reference()["ptb_molhiv_a0_history"]
    replication = {}
    for seed in C.SEEDS:
        mine = valid_units[f"{C.MODEL_A0}_s{seed}"]
        replication[str(seed)] = {
            "valid_roc_auc_recorded": mine["roc_auc"],
            "valid_roc_auc_history": float(hist_a0["valid_roc_auc"][seed]),
            "valid_roc_auc_diff": float(mine["roc_auc"] - float(hist_a0["valid_roc_auc"][seed])),
            "valid_pr_auc_recorded": mine["pr_auc"],
            "valid_pr_auc_history": float(hist_a0["valid_pr_auc"][seed]),
            "valid_pr_auc_diff": float(mine["pr_auc"] - float(hist_a0["valid_pr_auc"][seed])),
            "best_epoch_recorded": int(mine["best_epoch"]),
            "best_epoch_history": int(hist_a0["best_epoch"][seed]),
        }
    replication["max_abs_valid_roc_auc_diff"] = max(
        abs(v["valid_roc_auc_diff"]) for k, v in replication.items() if k.isdigit()
    )
    replication["identical_best_epochs"] = all(
        v["best_epoch_recorded"] == v["best_epoch_history"] for k, v in replication.items() if k.isdigit()
    )
    replication["interpretation"] = (
        "informational replication check of the frozen A0 recipe on the same host/software: a bitwise "
        "reproduction of the historical PTB A0 numbers means the paired A0 baseline is the same recipe, "
        "not an older checkpoint; any difference is reported, not corrected"
    )

    def contrast_block(getter) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for a, b in C.CONTRASTS:
            diffs = [float(getter(a, s) - getter(b, s)) for s in C.SEEDS]
            stats = _mean_sd(diffs)
            out[f"{a}-{b}"] = {
                "values": stats["values"],
                "mean": stats["mean"],
                "sd_ddof1": stats["sd_ddof1"],
                "min": stats["min"],
                "max": stats["max"],
                "n_seeds_positive": int(sum(1 for d in diffs if d > 0)),
            }
        return out

    valid_contrasts = contrast_block(lambda arm, seed: valid_units[f"{arm}_s{seed}"]["roc_auc"])
    valid_ap_contrasts = contrast_block(lambda arm, seed: valid_units[f"{arm}_s{seed}"]["pr_auc"])
    valid = {
        "schema": "lcrc-sparse-identity-validation-metrics-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "split": "official OGB scaffold Validation (4,113 molecules)",
        "selection_used_test": False,
        "units": valid_units,
        "sparse_only": valid_sparse,
        "summary": valid_summary,
        "contrasts": valid_contrasts,
        "ap_contrasts": valid_ap_contrasts,
        "a0_history_replication": replication,
        "train_valid_gap": {
            arm: {
                "train_roc_auc_mean": float(
                    np.mean([valid_units[f"{arm}_s{s}"]["train_roc_auc"] for s in C.SEEDS])
                ),
                "valid_roc_auc_mean": float(np.mean([valid_units[f"{arm}_s{s}"]["roc_auc"] for s in C.SEEDS])),
                "gap_mean": float(
                    np.mean([valid_units[f"{arm}_s{s}"]["train_minus_valid_roc_auc"] for s in C.SEEDS])
                ),
            }
            for arm in (C.MODEL_A0, C.MODEL_TRUE, C.MODEL_NULL, C.MODEL_FOLD)
        },
    }
    C.write_json(C.ANALYSIS / "validation_metrics.json", valid)

    # ---- Test -------------------------------------------------------------
    test_units: dict[str, Any] = {}
    for name, entry in scores["scores"].items():
        if entry["arm"] == C.MODEL_SPARSE:
            continue
        test_units[f"{entry['arm']}_s{int(entry['seed'])}"] = {
            "arm": entry["arm"],
            "seed": int(entry["seed"]),
            "roc_auc": float(entry["metrics"]["roc_auc"]),
            "pr_auc": float(entry["metrics"]["pr_auc"]),
            "bce": float(entry["metrics"]["bce"]),
            "official_ogb_roc_auc": float(entry["official_ogb"]["ogb_rocauc"]),
            "sklearn_roc_auc": float(entry["official_ogb"]["sklearn_roc_auc"]),
            "evaluator_consistent": bool(entry["official_ogb"]["consistent"]),
            "best_epoch": int(entry["best_epoch"]),
            "branch_contribution_test": entry["branch_contribution"],
            "branch_norms": entry["branch_norms"],
        }
    sp = scores["scores"][C.MODEL_SPARSE]
    test_sparse = {
        "arm": C.MODEL_SPARSE,
        "best_epoch": 0,
        "roc_auc": float(sp["metrics"]["roc_auc"]),
        "pr_auc": float(sp["metrics"]["pr_auc"]),
        "bce": float(sp["metrics"]["bce"]),
        "official_ogb_roc_auc": float(sp["official_ogb"]["ogb_rocauc"]),
        "sklearn_roc_auc": float(sp["official_ogb"]["sklearn_roc_auc"]),
        "evaluator_consistent": bool(sp["official_ogb"]["consistent"]),
    }
    def test_get(arm: str, seed: int, key: str) -> Any:
        return test_sparse[key] if arm == C.MODEL_SPARSE else test_units[f"{arm}_s{seed}"][key]

    test_summary = summary_from(test_get)
    test_contrasts = contrast_block(lambda arm, seed: test_units[f"{arm}_s{seed}"]["roc_auc"])
    test_ap_contrasts = contrast_block(lambda arm, seed: test_units[f"{arm}_s{seed}"]["pr_auc"])
    test = {
        "schema": "lcrc-sparse-identity-test-metrics-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "split": "official OGB scaffold Test (4,113 molecules)",
        "exposure": "EXPOSED_TEST_EXPLORATORY",
        "n_models_scored": int(len(scores["scores"])),
        "manifest_sha256": scores["manifest_sha256"],
        "test_rows_sha256": scores["test_rows_sha256"],
        "test_null_seed": int(scores["test_null_seed"]),
        "test_null_stats": scores["test_null_stats"],
        "execution_regime": scores["execution_regime"],
        "units": test_units,
        "sparse_only": test_sparse,
        "summary": test_summary,
        "contrasts": test_contrasts,
        "ap_contrasts": test_ap_contrasts,
        "evaluator_cross_check": {
            "all_consistent": bool(all(v["evaluator_consistent"] for v in test_units.values()))
            and bool(test_sparse["evaluator_consistent"]),
            "max_abs_diff": float(
                max([abs(v["official_ogb_roc_auc"] - v["sklearn_roc_auc"]) for v in test_units.values()]
                    + [abs(test_sparse["official_ogb_roc_auc"] - test_sparse["sklearn_roc_auc"])])
            ),
            "note": "official OGB Evaluator(name='ogbg-molhiv') vs sklearn roc_auc_score on the same logits",
        },
        "no_seed_selection": True,
        "no_post_test_retraining": True,
        "result_label": C.RESULT_LABEL,
    }
    C.write_json(C.ANALYSIS / "test_metrics.json", test)

    # ---- paired scaffold bootstrap ---------------------------------------
    import ptb_data as PD

    view = PD.molecule_view(C.DATASET)
    bootstrap_out: dict[str, Any] = {
        "schema": "lcrc-sparse-identity-paired-bootstrap-v1",
        "round_id": C.ROUND_ID,
        "seed": int(C.BOOTSTRAP_SEED),
        "replicates": int(C.BOOTSTRAP_REPLICATES),
        "created_utc": C.utcnow(),
        "splits": {},
        "interpretation_limit": "describes sample uncertainty for fixed checkpoints and a fixed split; "
        "does not replace training-seed uncertainty; the Test interval is not confirmatory blind-test evidence",
    }
    scaffolds_all: dict[str, Any] = {"schema": "lcrc-sparse-identity-scaffolds-v1", "splits": {}}
    for split in ("valid", "test"):
        if split == "valid":
            smiles = view.valid_cache["smiles"]
            rows = [int(x) for x in np.asarray(view.valid_cache["rows"]).tolist()]
            y = np.asarray(view.targets.y_valid, dtype=np.float64).reshape(-1)
            preds = {
                arm: {
                    seed: np.load(C.PREDICTIONS / f"{arm}_s{seed}_valid.npz")["pred"].reshape(-1)
                    for seed in C.SEEDS
                }
                for arm in (C.MODEL_A0, C.MODEL_TRUE, C.MODEL_NULL, C.MODEL_FOLD)
            }
        else:
            test_cache_path = (
                C.REPO
                / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/cache/ogbg-molhiv_test.pt"
            )
            import torch as _torch

            test_cache = _torch.load(test_cache_path, map_location="cpu", weights_only=False)
            smiles = test_cache["smiles"]
            rows = [int(x) for x in np.asarray(test_cache["rows"]).tolist()]
            y = np.load(C.PREDICTIONS / f"{C.MODEL_A0}_s0_test.npz")["y_true"].reshape(-1)
            preds = {
                arm: {
                    seed: np.load(C.PREDICTIONS / f"{arm}_s{seed}_test.npz")["pred"].reshape(-1)
                    for seed in C.SEEDS
                }
                for arm in (C.MODEL_A0, C.MODEL_TRUE, C.MODEL_NULL, C.MODEL_FOLD)
            }
            if C.index_sha256(rows) != scores["test_rows_sha256"]:
                raise RuntimeError("Test row ids differ from the frozen scoring pass")
        scaffold = scaffolds_for(smiles, key=split)
        scaffold["rows_sha256"] = C.index_sha256(rows)
        scaffolds_all["splits"][split] = {k: v for k, v in scaffold.items() if k != "groups"}
        uniq = sorted(set(scaffold["groups"]))
        index = {g: i for i, g in enumerate(uniq)}
        code_arr = np.asarray([index[g] for g in scaffold["groups"]], dtype=np.int64)
        bootstrap_out["splits"][split] = paired_bootstrap(
            preds, y, code_arr, seed=C.BOOTSTRAP_SEED, replicates=C.BOOTSTRAP_REPLICATES
        )
        bootstrap_out["splits"][split]["scaffold_summary"] = {k: v for k, v in scaffold.items() if k != "groups"}
    C.write_json(C.ANALYSIS / "paired_bootstrap.json", bootstrap_out)
    C.write_json(C.ANALYSIS / "scaffolds.json", scaffolds_all)

    # ---- golden decision --------------------------------------------------
    def _mean_diff(block: Mapping[str, Any], a: str, b: str) -> float:
        return float(block[f"{a}-{b}"]["mean"])

    cond: dict[str, Any] = {}
    cond["1_true_minus_a0_valid"] = {
        "value": _mean_diff(valid_contrasts, C.MODEL_TRUE, C.MODEL_A0),
        "threshold": C.DECISION_MIN_TRUE_MINUS_A0,
        "pass": _mean_diff(valid_contrasts, C.MODEL_TRUE, C.MODEL_A0) >= C.DECISION_MIN_TRUE_MINUS_A0,
    }
    cond["1_true_minus_a0_test"] = {
        "value": _mean_diff(test_contrasts, C.MODEL_TRUE, C.MODEL_A0),
        "threshold": C.DECISION_MIN_TRUE_MINUS_A0,
        "pass": _mean_diff(test_contrasts, C.MODEL_TRUE, C.MODEL_A0) >= C.DECISION_MIN_TRUE_MINUS_A0,
    }
    seeds_valid = int(valid_contrasts[f"{C.MODEL_TRUE}-{C.MODEL_A0}"]["n_seeds_positive"])
    seeds_test = int(test_contrasts[f"{C.MODEL_TRUE}-{C.MODEL_A0}"]["n_seeds_positive"])
    seeds_both = int(
        sum(
            1
            for s in C.SEEDS
            if valid_units[f"{C.MODEL_TRUE}_s{s}"]["roc_auc"] > valid_units[f"{C.MODEL_A0}_s{s}"]["roc_auc"]
            and test_units[f"{C.MODEL_TRUE}_s{s}"]["roc_auc"] > test_units[f"{C.MODEL_A0}_s{s}"]["roc_auc"]
        )
    )
    cond["2_seed_consistency"] = {
        "seeds_true_above_a0_valid": seeds_valid,
        "seeds_true_above_a0_test": seeds_test,
        "seeds_true_above_a0_on_both": seeds_both,
        "required": int(C.DECISION_MIN_SEEDS_BEATING_A0),
        "pass": bool(
            seeds_valid >= C.DECISION_MIN_SEEDS_BEATING_A0 and seeds_test >= C.DECISION_MIN_SEEDS_BEATING_A0
        ),
    }
    cond["3_true_minus_null_valid"] = {
        "value": _mean_diff(valid_contrasts, C.MODEL_TRUE, C.MODEL_NULL),
        "threshold": C.DECISION_MIN_TRUE_MINUS_NULL,
        "pass": _mean_diff(valid_contrasts, C.MODEL_TRUE, C.MODEL_NULL) >= C.DECISION_MIN_TRUE_MINUS_NULL,
    }
    cond["3_true_minus_null_test"] = {
        "value": _mean_diff(test_contrasts, C.MODEL_TRUE, C.MODEL_NULL),
        "threshold": C.DECISION_MIN_TRUE_MINUS_NULL,
        "pass": _mean_diff(test_contrasts, C.MODEL_TRUE, C.MODEL_NULL) >= C.DECISION_MIN_TRUE_MINUS_NULL,
    }
    cond["4_test_ap_not_worse"] = {
        "true_mean_ap": float(test_summary[C.MODEL_TRUE]["pr_auc"]["mean"]),
        "a0_mean_ap": float(test_summary[C.MODEL_A0]["pr_auc"]["mean"]),
        "pass": float(test_summary[C.MODEL_TRUE]["pr_auc"]["mean"])
        >= float(test_summary[C.MODEL_A0]["pr_auc"]["mean"]),
    }
    cond["5_all_gates_pass"] = {
        "status": gates["status"],
        "pass": bool(gates["status"] == "PASS" and all(bool(g["pass"]) for g in gates["gates"])),
        "failed_gates": [g["id"] for g in gates["gates"] if not bool(g["pass"])],
    }

    perf_positive = bool(
        cond["1_true_minus_a0_valid"]["value"] > 0.0 and cond["1_true_minus_a0_test"]["value"] > 0.0
    )
    if len(manifest["units"]) != 12 or len(test_units) != 12:
        verdict = "INCOMPLETE"
        reason = "fewer than 12 neural units were trained and/or scored"
    elif not perf_positive:
        verdict = "NO_PERFORMANCE_SIGNAL"
        reason = "WIDE_TRUE minus A0 3-seed mean ROC-AUC is <= 0 on Validation or on Test"
    elif all(v.get("pass") for v in cond.values()):
        verdict = "PREDICTIVE_IDENTITY_SIGNAL"
        reason = "all five pre-registered conditions are met"
    elif not (cond["3_true_minus_null_valid"]["pass"] and cond["3_true_minus_null_test"]["pass"]):
        verdict = "NO_IDENTITY_SPECIFIC_SIGNAL"
        reason = "WIDE_TRUE has a positive increment over A0 but does not reach the +0.005 identity margin "
        "over WIDE_NULL on both splits"
    else:
        verdict = "MIXED_OR_WEAK_SIGNAL"
        reason = "partial positive evidence: not every pre-registered condition is met"
    decision = {
        "schema": "lcrc-sparse-identity-decision-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "verdict": verdict,
        "reason": reason,
        "conditions": cond,
        "thresholds": {
            "true_minus_a0": C.DECISION_MIN_TRUE_MINUS_A0,
            "true_minus_null": C.DECISION_MIN_TRUE_MINUS_NULL,
            "min_seeds_beating_a0": int(C.DECISION_MIN_SEEDS_BEATING_A0),
        },
        "additional_context_not_used_for_the_verdict": {
            "true_minus_fold64_valid": _mean_diff(valid_contrasts, C.MODEL_TRUE, C.MODEL_FOLD),
            "true_minus_fold64_test": _mean_diff(test_contrasts, C.MODEL_TRUE, C.MODEL_FOLD),
            "fold64_minus_a0_valid": _mean_diff(valid_contrasts, C.MODEL_FOLD, C.MODEL_A0),
            "fold64_minus_a0_test": _mean_diff(test_contrasts, C.MODEL_FOLD, C.MODEL_A0),
            "null_minus_a0_valid": _mean_diff(valid_contrasts, C.MODEL_NULL, C.MODEL_A0),
            "null_minus_a0_test": _mean_diff(test_contrasts, C.MODEL_NULL, C.MODEL_A0),
            "sparse_valid_roc_auc": float(valid_sparse["roc_auc"]),
            "sparse_test_roc_auc": float(test_sparse["roc_auc"]),
        },
        "result_label": C.RESULT_LABEL,
    }
    C.write_json(C.ANALYSIS / "decision.json", decision)

    # ---- ledgers / provenance --------------------------------------------
    ledger = gpu_ledger()
    C.write_json(C.ANALYSIS / "gpu_ledger.json", ledger)

    gates_payload = C.read_json(C.ANALYSIS / "correctness_gates.json")
    provenance = {
        "schema": "lcrc-sparse-identity-provenance-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "protocol": {
            "path": "PROTOCOL.md",
            "sha256": C.sha256_file(C.TRACK / "PROTOCOL.md"),
            "k": int(C.K),
            "n_buckets": int(C.N_BUCKETS),
            "fold_salt": C.FOLD_SALT,
            "null_seeds": dict(C.NULL_SEED),
            "null_swap_multiplier": int(C.NULL_SWAP_MULTIPLIER),
            "seeds": list(C.SEEDS),
            "epochs": int(C.EPOCHS),
            "recipe": dict(C.PC.RECIPE),
            "decision_thresholds": {
                "true_minus_a0": C.DECISION_MIN_TRUE_MINUS_A0,
                "true_minus_null": C.DECISION_MIN_TRUE_MINUS_NULL,
                "min_seeds_beating_a0": int(C.DECISION_MIN_SEEDS_BEATING_A0),
            },
            "budget": {
                "soft_hours": float(C.GPU_SOFT_CAP_HOURS),
                "hard_hours": float(C.GPU_HARD_CAP_HOURS),
            },
        },
        "code_hashes": C.code_hashes(),
        "data": {
            "dataset": C.DATASET,
            "split_method": "official OGB scaffold split",
            "sizes": dict(C.EXPECTED_SPLIT_SIZES),
            "split_file_sha256": PD.molecule_view(C.DATASET).source_meta["split_file_sha256"],
            "rows_sha256": {
                "train": C.index_sha256(PD.molecule_view(C.DATASET).rows["train"]),
                "valid": C.index_sha256(PD.molecule_view(C.DATASET).rows["valid"]),
                "test": scores["test_rows_sha256"],
            },
            "frozen_split_manifest": C.track_rel(
                C.REPO
                / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/split_manifest_ogbg-molhiv.json"
            ),
            "frozen_split_manifest_sha256": C.sha256_file(
                C.REPO
                / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/split_manifest_ogbg-molhiv.json"
            ),
            "feature_spec_hash": C.FROZEN_SPEC_HASH,
            "test_labels_read_before_freeze": False,
        },
        "features": {
            "vocabulary_sha256": manifest["features"]["vocabulary_sha256"],
            "fold_assignment_sha256": manifest["features"]["fold_assignment_sha256"],
            "split_stats": manifest["features"]["splits"],
            "null_checks_sha256": manifest["null_checks_sha256"],
        },
        "checkpoints": {
            "manifest_sha256": manifest["manifest_sha256"],
            "per_unit_sha256": {u["unit"]: u["checkpoint_sha256"] for u in manifest["units"]},
            "per_unit_init_state_sha256": {u["unit"]: u["init_state_hash"] for u in manifest["units"]},
            "per_unit_best_state_sha256": {u["unit"]: u["best_state_hash"] for u in manifest["units"]},
        },
        "execution_regimes": sorted(
            {
                json.dumps(
                    {
                        "hostname": u["env"].get("hostname"),
                        "torch": u["env"].get("torch"),
                        "torch_cuda": u["env"].get("torch_cuda"),
                        "cuda_devices": u["env"].get("cuda_device_names"),
                        "git_commit": u["git_commit"],
                    },
                    sort_keys=True,
                )
                for u in manifest["units"]
            }
        ),
        "test_scoring_regime": scores["execution_regime"],
        "gates": {
            "status": gates_payload["status"],
            "sha256": C.sha256_file(C.ANALYSIS / "correctness_gates.json"),
            "failed_gates": [g["id"] for g in gates_payload["gates"] if not bool(g["pass"])],
        },
        "training_code_identity": training_code_identity(
            [u["git_commit"] for u in manifest["units"]]
        ),
        "gpu_ledger_sha256": C.sha256_file(C.ANALYSIS / "gpu_ledger.json"),
        "gpu_hours_total": ledger["total_gpu_hours"],
        "historical_reference": historical_reference(),
        "result_label": C.RESULT_LABEL,
    }
    C.write_json(C.ANALYSIS / "provenance.json", provenance)

    tables = build_tables(valid, test)
    (C.ANALYSIS / "report_tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    print(json.dumps({"verdict": verdict, "reason": reason, "conditions": cond}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
