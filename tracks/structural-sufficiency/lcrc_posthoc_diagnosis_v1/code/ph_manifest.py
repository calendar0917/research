"""Generate the source manifest and the data-access ledger for this post-hoc round."""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path
from typing import Any

import ph_common as C


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _prediction_inputs() -> list[dict[str, Any]]:
    out = []
    for dataset in C.MOLECULE_ORDER:
        for seed in C.MOLECULE_SEEDS:
            for model in C.MODELS:
                d = C.mol_run_dir(dataset, model, seed)
                for name, role in (
                    ("test_predictions.npz", "saved frozen Test predictions; rows/pred/y_true"),
                    ("test_eval.json", "saved Test metrics and checkpoint SHA record"),
                    ("run.json", "saved Train-only target normaliser and split metadata"),
                ):
                    out.append(
                        {
                            **C.file_record(d / name, role),
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "kind": "molecule",
                            "label_content": "saved y_true only (already exposed Test split)"
                            if name == "test_predictions.npz"
                            else "aggregated metrics / normaliser; no raw labels"
                            if name == "test_eval.json"
                            else "target normaliser fitted on Train only",
                        }
                    )
    for fold in range(5):
        for seed in C.TU_SEEDS:
            for model in C.MODELS:
                d = C.tu_run_dir("NCI1", fold, seed, model)
                out.append(
                    {
                        **C.file_record(d / "test_predictions.npz", "saved frozen NCI1 Test predictions"),
                        "dataset": "NCI1",
                        "fold": fold,
                        "model": model,
                        "seed": seed,
                        "kind": "tu",
                        "label_content": "saved y_true only (already exposed outer fold)",
                    }
                )
    return out


def _frozen_inputs() -> list[dict[str, Any]]:
    specs = [
        (C.PTB / "analysis/molecular_results.csv", "previous round aggregated molecule metrics"),
        (C.PTB / "analysis/paired_differences.csv", "previous round per-seed paired differences"),
        (C.PTB / "analysis/statistics.json", "previous round per-seed bootstrap values"),
        (C.PTB / "analysis/mechanism_audit.json", "fixed-weight typed-S2 input intervention rows"),
        (C.PTB / "analysis/split_manifest.json", "frozen split hashes and NCI1 fold definitions"),
        (C.PTB / "analysis/checkpoint_manifest.json", "frozen checkpoint SHA records"),
        (C.PTB / "analysis/tu_results.csv", "previous round NCI1 per-fold/seed accuracy"),
        (C.PTB / "code/ptb_report.py", "source of the old seed-wise CI envelope"),
        (C.EDIT / "analysis/probe_audit_scores.json", "saved Audit d_pred/e_delta/pair metadata; raw labels not read"),
        (C.EDIT / "analysis/probe_report.json", "old TRACE verdict and residual MAE table"),
        (C.EDIT / "analysis/core_classes.json", "frozen mapping_unique per core class"),
        (C.OGB_SOURCE / "mapping/mol.csv.gz", "OGB SMILES column only"),
        (C.OGB_SOURCE / "split/scaffold/train.csv.gz", "OGB official scaffold train row ids"),
        (C.OGB_SOURCE / "split/scaffold/valid.csv.gz", "OGB official scaffold valid row ids"),
        (C.OGB_SOURCE / "split/scaffold/test.csv.gz", "OGB official scaffold test row ids"),
        (
            C.REPO
            / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/ESOL/ESOL.csv.gz",
            "ESOL SMILES column only",
        ),
        (
            C.REPO
            / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/ESOL/ESOL.random122.ind.pkl",
            "ESOL frozen random122 split indices only",
        ),
        (C.REPO / "data/TUD/NCI1/NCI1/raw/NCI1_graph_indicator.txt", "NCI1 node->graph ids; node counts"),
        (C.REPO / "data/TUD/NCI1/NCI1/raw/NCI1_graph_labels.txt", "NCI1 labels used only to reconstruct frozen folds"),
    ]
    out = []
    for path, role in specs:
        rec = C.file_record(path, role)
        rec["label_content"] = (
            "smiles only; no label column selected"
            if "mol.csv.gz" in str(path) or "ESOL.csv.gz" in str(path)
            else "already-exposed graph labels; fold reconstruction only"
            if "NCI1_graph_labels" in str(path)
            else "derived audit quantities; raw Audit labels not read"
            if "probe_audit_scores" in str(path)
            else "aggregated metrics / hashes; no raw labels"
        )
        out.append(rec)
    return out


def _code_files() -> list[dict[str, Any]]:
    out = []
    for path in sorted((C.ROUND / "code").glob("*.py")) + sorted((C.ROUND / "tests").glob("*.py")):
        out.append(C.file_record(path, "new analysis code/test"))
    out.append(C.file_record(C.ROUND / "PROTOCOL.md", "fixed post-hoc analysis protocol"))
    return out


def _outputs() -> list[dict[str, Any]]:
    out = []
    for path in sorted(C.ANALYSIS.glob("*")):
        if path.is_file():
            out.append(C.file_record(path, "generated analysis artifact"))
    for name in ("ERRATA.md", "REPORT.md", "DECISION.md", "README.md"):
        path = C.ROUND / name
        if path.exists():
            out.append(C.file_record(path, "generated report"))
    return out


def main() -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=C.REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=C.REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    preds = _prediction_inputs()
    frozen = _frozen_inputs()
    code = _code_files()
    manifest = {
        "schema": "lcrc-posthoc-source-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": _now(),
        "baseline_commit": C.BASELINE_COMMIT,
        "head_at_generation": head,
        "branch": branch,
        "worktree_entries_at_generation": len(status),
        "environment": C.environment(),
        "prediction_inputs": preds,
        "frozen_inputs": frozen,
        "code": code,
        "outputs": _outputs(),
        "missing_artifacts": [],
    }
    C.write_json(C.ANALYSIS / "source_manifest.json", manifest)

    ledger = {
        "schema": "lcrc-posthoc-data-access-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": _now(),
        "baseline_commit": C.BASELINE_COMMIT,
        "head_at_generation": head,
        "policy": {
            "gpu_used": False,
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "training_or_finetuning_calls": [],
            "test_scoring_entrypoint_calls": [],
            "new_test_inference": False,
            "new_test_label_source_read": False,
            "raw_audit_labels_read": False,
            "new_split_generated": False,
            "frozen_protocols_modified": False,
            "historical_checkpoints_modified": False,
        },
        "inputs": {
            "molecule_saved_predictions": {
                "files": [p for p in preds if p["kind"] == "molecule"],
                "note": "Test rows/pred/y_true are copied from the frozen runs; no inference rerun.",
            },
            "nci1_saved_predictions": {
                "files": [p for p in preds if p["kind"] == "tu"],
                "note": "outer-fold gids/logits/y_true are copied from the frozen runs.",
            },
            "frozen_reports": [p for p in frozen if p["role"] is not None and "analysis" in p["path"]],
            "structure_sources": [
                p
                for p in frozen
                if any(
                    key in p["path"]
                    for key in (
                        "ogb/ogbg_molhiv",
                        "chembench/ESOL",
                        "TUD/NCI1",
                    )
                )
            ],
            "audit_derived_scores": [p for p in frozen if "probe_audit_scores" in p["path"]],
        },
        "label_access": [
            {
                "source": C.rel(C.EDIT / "analysis/probe_audit_scores.json"),
                "content": "d_pred, e_delta, pair_id, component, class_id, same_var_heavy",
                "raw_label_read": False,
                "note": "Δy reconstructed as d_pred - e_delta; no Audit label file is opened.",
            },
            {
                "source": C.rel(C.OGB_SOURCE / "mapping/mol.csv.gz"),
                "content": "smiles column only",
                "raw_label_read": False,
                "note": "HIV_active / raw/graph-label.csv.gz not read by this round.",
            },
            {
                "source": C.rel(C.REPO / "data/TUD/NCI1/NCI1/raw/NCI1_graph_labels.txt"),
                "content": "TUD graph labels",
                "raw_label_read": True,
                "note": "Already exposed; used only to reproduce the frozen 5-fold split exactly, not to score a new prediction.",
            },
            {
                "source": "saved test_predictions.npz (36 molecule + 20 NCI1)",
                "content": "y_true arrays",
                "raw_label_read": True,
                "note": "Already exposed Test labels saved by the previous round; no new label source is decoded.",
            },
        ],
        "products": [
            {
                "product": "analysis/p0_errata.json",
                "inputs": ["molecular_results.csv", "tu_results.csv", "mechanism_audit.json", "statistics.json"],
            },
            {
                "product": "analysis/mean_seed_bootstrap_ci.json|csv",
                "inputs": ["36 molecule test_predictions.npz", "run.json target_norm", "statistics.json"],
            },
            {
                "product": "analysis/molhiv_*.csv|molhiv_analysis.json",
                "inputs": [
                    "6 MolHIV test_predictions.npz",
                    "OGB mapping smiles",
                    "OGB scaffold split ids",
                ],
            },
            {
                "product": "analysis/edit_response_*",
                "inputs": ["probe_audit_scores.json", "probe_report.json", "core_classes.json"],
            },
            {
                "product": "analysis/esol_subgroups.csv|esol_aux.json",
                "inputs": ["6 ESOL test_predictions.npz", "ESOL smiles", "ESOL random122 split"],
            },
            {
                "product": "analysis/nci1_subgroups.csv|nci1_aux.json",
                "inputs": ["20 NCI1 test_predictions.npz", "NCI1 raw graph structure", "split_manifest.json"],
            },
        ],
        "missing_artifacts": [],
    }
    C.write_json(C.ANALYSIS / "data_access_ledger.json", ledger)
    print(f"predictions: {len(preds)}; frozen: {len(frozen)}; code: {len(code)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
