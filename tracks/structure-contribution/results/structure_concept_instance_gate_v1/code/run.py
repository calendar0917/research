"""Orchestration CLI for ``structure_concept_instance_gate_v1``.

Usage (from the repository root, CPU only):

    uv run python tracks/structure-contribution/results/structure_concept_instance_gate_v1/code/run.py preflight
    uv run python .../run.py freeze
    uv run python .../run.py train --seed 0
    uv run python .../run.py analyze --seed 0
    uv run python .../run.py finalize

The round is deliberately small: <= 4 architectures, first one seed, then at
most one confirmation seed, no hyper-parameter search, no real molecular data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
for p in (str(HERE),):
    if p not in sys.path:
        sys.path.insert(0, p)

import analysis as A  # noqa: E402
import pipeline as P  # noqa: E402
import synthetic as S  # noqa: E402
from models import ARMS  # noqa: E402

CODE_FILES = ("synthetic.py", "models.py", "pipeline.py", "analysis.py", "run.py")
TRAIN_ARMS = ("A", "B", "C", "D")
SCRAMBLE_ARMS = ("B", "D")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def cmd_preflight(_args) -> None:
    rep = S.run_preflight()
    rep["passed"] = True
    _write_json(RESULTS / "preflight.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "checks"}, indent=1))
    print("preflight: PASS")


def cmd_freeze(_args) -> None:
    existing = list(P.TRAIN_LOGS.glob("*.pt")) if P.TRAIN_LOGS.exists() else []
    protocol = {
        "round": "structure_concept_instance_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_training": not existing,
        "code_sha256": {name: _sha256(HERE / name) for name in CODE_FILES},
        "split_manifest": S.frozen_split_manifest(),
        "generator_inventory": {
            "binding_bases": [b.name for b in S.BINDING_BASES],
            "composition_bases": [b.name for b in S.COMPOSITION_BASES],
            "cycle_sizes": {k: list(v) for k, v in S.CYCLE_SIZES.items()},
            "relation_pair_types": [[S.ATTR_NAMES[a], S.ATTR_NAMES[b]] for a, b in S.REL_PAIR_TYPES],
            "short_d": S.SHORT_D,
            "long_d": S.LONG_D,
        },
        "train_config": {
            "max_epochs": P.TrainConfig().max_epochs,
            "patience": P.TrainConfig().patience,
            "lr": P.TrainConfig().lr,
            "batch_size": P.TrainConfig().batch_size,
            "optimizer": "Adam",
            "clip_grad_norm": 5.0,
            "loss": "BCE + 0.5*MSE(recon) + 0.1*commitment + 0.1*codebook (VQ arms only)",
            "model": {
                "patch_feat_dim": 35,
                "relation_feat_dim": 11,
                "d": 32,
                "hidden": 64,
                "n_concepts": 16,
                "tau": 0.1,
            },
            "seeds": [0, 1],
            "primary_seed": 0,
        },
        "evaluation": {
            "split_policy": "train/dev/test by base graph; no base graph crosses splits",
            "primary_task": "relation matched pairs (identical patch bag, opposite label)",
            "scrambled_relation_control": "B/D retrained with i.i.d. relation features drawn from the train empirical distance distribution",
        },
        "gate_thresholds": {
            "G1_information_limit": [0.45, 0.55],
            "G2_relation_used": {"true_min": 0.80, "scrambled_max": 0.60, "margin_min": 0.15},
            "G3_hard_concept": {"hard_min": 0.80, "hard_soft_gap_max": 0.05},
            "G4_concept_semantics": {"test_purity_min": 0.80, "key_sap_purity_min": 0.80, "cross_base_min": 0.80, "cross_seed_min": 0.80},
            "G5_explanation_recovery": {"binding_rank_max": 3, "binding_spearman_min": 0.5, "relation_pair_rank_max": 5},
            "G6_composition": {"min": 0.65},
            "soft_leakage": {"hard_below": 0.70, "soft_at_least": 0.85, "gap_above": 0.10},
        },
        "final_label_vocabulary": [
            "GROUNDING_AND_RELATIONS_SUPPORTED",
            "PREDICTION_ONLY_NO_GROUNDING",
            "RELATION_NOT_USED",
            "SOFT_CONCEPT_LEAKAGE",
            "INCONCLUSIVE",
        ],
        "budget": {
            "cpu_only": True,
            "max_architectures": 4,
            "seeds": [0, 1],
            "no_hyperparameter_search": True,
            "no_real_molecular_data": True,
        },
    }
    if existing:
        raise SystemExit(f"refusing to freeze: training checkpoints already exist ({len(existing)} files)")
    _write_json(RESULTS / "protocol_freeze.json", protocol)
    print("protocol frozen:", protocol["code_sha256"])


def _environment() -> dict:
    import subprocess

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RESULTS, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=RESULTS, text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        git_commit, git_status = f"unavailable: {exc}", ""
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "git_commit": git_commit,
        "git_status_short": git_status,
    }


def cmd_train(args) -> None:
    seed = args.seed
    freeze = json.loads((RESULTS / "protocol_freeze.json").read_text(encoding="utf-8"))
    current = {name: _sha256(HERE / name) for name in CODE_FILES}
    # Science-critical frozen code: generator, models, training pipeline.  The
    # orchestration (run.py) and analysis (analysis.py) files are recorded but
    # may receive post-freeze bug fixes that cannot change a training result.
    for name in ("synthetic.py", "models.py", "pipeline.py"):
        if current.get(name) != freeze["code_sha256"].get(name):
            raise SystemExit(f"code hash mismatch after freeze for {name}: {current.get(name)} != {freeze['code_sha256'].get(name)}")
    cfg = P.TrainConfig()
    if args.max_epochs or args.patience:
        cfg = P.TrainConfig(
            max_epochs=args.max_epochs or cfg.max_epochs,
            patience=args.patience or cfg.patience,
        )
        if args.amendment is None:
            raise SystemExit("a training-budget override requires --amendment <id>")
        amendments = freeze.get("training_budget_amendments", {})
        if args.amendment not in amendments:
            raise SystemExit(f"unknown amendment id {args.amendment!r}; recorded: {sorted(amendments)}")
        recorded = amendments[args.amendment]
        if (recorded["max_epochs"], recorded["patience"]) != (cfg.max_epochs, cfg.patience):
            raise SystemExit(
                f"amendment {args.amendment!r} records {(recorded['max_epochs'], recorded['patience'])}, "
                f"requested {(cfg.max_epochs, cfg.patience)}"
            )
    data, manifest = P.prepare_data()
    _write_json(RESULTS / "analysis" / f"data_manifest_seed{seed}.json", manifest)
    summary: dict = {
        "seed": seed,
        "runs": {},
        "environment": _environment(),
        "train_config": cfg.__dict__,
        "amendment": args.amendment,
    }
    if seed == 0:
        _write_json(RESULTS / "environment.json", summary["environment"])
    for arm in TRAIN_ARMS:
        run = P.train_arm(arm, seed, data, relation_source="true", cfg=cfg, verbose=True)
        tag = f"{arm}_seed{seed}"
        P.save_run(run, tag)
        summary["runs"][tag] = {k: v for k, v in run["report"].items() if k != "history"}
        print(f"trained {tag}: params={run['report']['n_parameters']} best_dev_acc={run['report']['best_dev_balanced_accuracy']:.3f}@{run['report']['best_epoch']}")
    for arm in SCRAMBLE_ARMS:
        run = P.train_arm(arm, seed, data, relation_source="scrambled", cfg=cfg, verbose=True)
        tag = f"{arm}_scrambled_seed{seed}"
        P.save_run(run, tag)
        summary["runs"][tag] = {k: v for k, v in run["report"].items() if k != "history"}
        print(f"trained {tag}: best_dev_acc={run['report']['best_dev_balanced_accuracy']:.3f}@{run['report']['best_epoch']}")
    _write_json(RESULTS / "train_logs" / f"train_summary_seed{seed}.json", summary)


def _load_models(seed: int) -> dict:
    models = {}
    for arm in TRAIN_ARMS:
        models[arm] = P.load_model(f"{arm}_seed{seed}")
    for arm in SCRAMBLE_ARMS:
        models[f"{arm}_scrambled"] = P.load_model(f"{arm}_scrambled_seed{seed}")
    return models


def cmd_analyze(args) -> None:
    seed = args.seed
    data, manifest = P.prepare_data()
    models = _load_models(seed)
    m: dict = {
        "seed": seed,
        "preflight": json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8")),
        "data_manifest": {k: v for k, v in manifest.items() if k != "split_manifest"},
        "training": {},
        "eval_dev": {},
        "eval_test": {},
        "scrambled_control": {},
        "concepts": {},
        "explanations": {"binding": {}, "relation": {}},
        "audits": {},
    }
    for arm in TRAIN_ARMS:
        with open(P.TRAIN_LOGS / f"{arm}_seed{seed}.json", encoding="utf-8") as fh:
            m["training"][arm] = json.load(fh)
    for arm in SCRAMBLE_ARMS:
        with open(P.TRAIN_LOGS / f"{arm}_scrambled_seed{seed}.json", encoding="utf-8") as fh:
            m["training"][f"{arm}_scrambled"] = json.load(fh)

    # -- predictions -------------------------------------------------------
    concept_records: dict[str, list[dict]] = {}
    for arm in TRAIN_ARMS:
        model = models[arm]
        dev = P.evaluate_split(model, data, "dev", "true", "hard")
        test = P.evaluate_split(model, data, "test", "true", "hard")
        m["eval_dev"][arm] = {
            "metrics": A.eval_metrics(dev),
            "matched_pairs": {
                "relation": A.matched_pair_metrics(dev, "relation"),
                "composition": A.matched_pair_metrics(dev, "composition"),
            },
        }
        m["eval_test"][arm] = {
            "metrics": A.eval_metrics(test),
            "by_family": A.eval_metrics(test)["by_family"],
            "matched_pairs": {
                "relation": A.matched_pair_metrics(test, "relation"),
                "composition": A.matched_pair_metrics(test, "composition"),
            },
            "hard_soft": A.hard_soft_gap(test),
            "bag_limit": A.bag_limit_check(test, "relation") if arm in ("A", "C") else None,
        }
        concept_records[arm] = A.concept_records(data, model) if arm in ("C", "D") else []
        (RESULTS / "analysis").mkdir(parents=True, exist_ok=True)
        if arm in ("C", "D"):
            with open(RESULTS / "analysis" / f"concepts_{arm}_seed{seed}.jsonl", "w", encoding="utf-8") as fh:
                for r in concept_records[arm]:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")

    for arm in SCRAMBLE_ARMS:
        model = models[f"{arm}_scrambled"]
        test = P.evaluate_split(model, data, "test", "scrambled", "hard")
        m["scrambled_control"][arm] = {
            "metrics": A.eval_metrics(test),
            "matched_pairs": {
                "relation": A.matched_pair_metrics(test, "relation"),
                "composition": A.matched_pair_metrics(test, "composition"),
            },
        }

    # -- audits ------------------------------------------------------------
    m["audits"]["head_input_C"] = A.head_input_audit(models["C"], data, "test")
    m["audits"]["head_input_D"] = A.head_input_audit(models["D"], data, "test")

    # -- concept semantics --------------------------------------------------
    for arm in ("C", "D"):
        recs = concept_records[arm]
        key_saps: set[str] = set()
        for gid, d in data.items():
            if d.family == "binding":
                nodes = [d.meta.get("p_in"), d.meta.get("p_out")]
                for p in d.patches:
                    nodeset = set(p.nodes)
                    if any((n is not None and n in nodeset) for n in nodes):
                        key_saps.add(p.sap)
        semantics = {
            "all": A.contingency_stats(recs),
            "train_purity": A.contingency_stats(recs, "train"),
            "test_purity": A.contingency_stats(recs, "test"),
            "test_purity_support5": A.contingency_stats(recs, "test", min_sap_support=5),
            "all_support5": A.contingency_stats(recs, min_sap_support=5),
            "key_sap_purity": A.key_sap_purity(recs, key_saps) if key_saps else {"n_patches": 0, "dominant_share": float("nan")},
            "key_saps": sorted(key_saps),
            "cross_base": A.cross_base_consistency(recs),
            "degeneracy": A.degeneracy_probes(recs),
            "conditional_mi": A.conditional_mi(recs),
            "examples": A.concept_examples(recs),
        }
        m["concepts"][arm] = semantics

    # -- explanations -------------------------------------------------------
    for arm in ("C", "D"):
        m["explanations"]["binding"][arm] = A.binding_recovery(models[arm], data, concept_records[arm], "test")
        m["explanations"]["relation"][arm] = A.relation_recovery(models[arm], data, concept_records[arm], "test")

    _write_json(RESULTS / "analysis" / f"metrics_seed{seed}.json", m)
    print("wrote", RESULTS / "analysis" / f"metrics_seed{seed}.json")


def cmd_finalize(_args) -> None:
    m0 = json.loads((RESULTS / "analysis" / "metrics_seed0.json").read_text(encoding="utf-8"))
    m1 = None
    p1 = RESULTS / "analysis" / "metrics_seed1.json"
    if p1.exists():
        m1 = json.loads(p1.read_text(encoding="utf-8"))
        for arm in ("C", "D"):
            rec0 = _read_jsonl(RESULTS / "analysis" / f"concepts_{arm}_seed0.jsonl")
            rec1 = _read_jsonl(RESULTS / "analysis" / f"concepts_{arm}_seed1.jsonl")
            m0["concepts"][arm]["cross_seed"] = A.cross_seed_agreement(rec0, rec1)
    merged = {
        "round": "structure_concept_instance_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": m0["preflight"],
        "seed0": m0,
        "seed1": m1,
    }
    gates = A.decide_gates(merged)
    merged["gates"] = gates
    merged["verdict"] = gates["verdict"]["label"]
    merged["recommendation"] = _recommendation(gates)
    _write_json(RESULTS / "metrics.json", merged)
    print("verdict:", merged["verdict"])
    print("recommendation:", merged["recommendation"])


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _recommendation(gates) -> str:
    label = gates["verdict"]["label"]
    if label == "GROUNDING_AND_RELATIONS_SUPPORTED":
        return (
            "The synthetic gate supports proceeding to the next research stage: discrete shared concepts and "
            "true instance relations both enter prediction and the concepts recover the generator-level structure. "
            "This does NOT yet support any molecular or paper-novelty claim; a real-task preregistration with a "
            "matched baseline is required first."
        )
    if label == "PREDICTION_ONLY_NO_GROUNDING":
        return (
            "Prediction works but the learned concept ids do not align with the synthetic structural-attribute "
            "patterns; do not proceed with interpretability claims. Re-scope the concept objective/regularisation "
            "in a NEW preregistration before any real-task work."
        )
    if label == "RELATION_NOT_USED":
        return (
            "The relation channel is not used beyond the label-blind scramble; stop the instance-relation route "
            "and re-examine the relation feature design in a new preregistration."
        )
    if label == "SOFT_CONCEPT_LEAKAGE":
        return (
            "Hard concepts are insufficient while soft concept scores succeed; treat the interpretable-concept "
            "claim as NOT established and audit the bottleneck implementation before any further use."
        )
    return (
        "Evidence is mixed or gates could not be evaluated; do not proceed and do not claim concept grounding. "
        "A new preregistration with a precise failure analysis is required."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").set_defaults(func=cmd_preflight)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    tr = sub.add_parser("train")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--max-epochs", type=int, default=0)
    tr.add_argument("--patience", type=int, default=0)
    tr.add_argument("--amendment", type=str, default=None)
    tr.set_defaults(func=cmd_train)
    an = sub.add_parser("analyze")
    an.add_argument("--seed", type=int, default=0)
    an.set_defaults(func=cmd_analyze)
    sub.add_parser("finalize").set_defaults(func=cmd_finalize)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
