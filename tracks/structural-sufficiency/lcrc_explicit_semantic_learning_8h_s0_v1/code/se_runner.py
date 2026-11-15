"""Control-plane runner for ``lcrc-explicit-semantic-learning-8h-s0-v1``.

Stages (``--set stage=...``):

``prepare``    verify the reused caches / spec / codec and freeze the nested
               label-budget subsets + the ESOL view; record hashes
``preflight``  run the targeted construction checks (CPU or GPU) and write
               ``analysis/preflight.json``
``train``      one formal run (``--set family=LS|FS|LP|FP|B --set recipe=R|W0|Rwd
               --set dataset=lipo|esol --set budget=N100|N50|N25``)
``analyze``    assemble the tables/DiD/bootstrap/support/kernel/MMP outputs (CPU)

The official Test split is a declared development-evaluation target of this
study; selection stays on Validation (grid for budget runs, best-epoch
otherwise).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import se_common as C

from ksvd_research.runner_api import Runner, RunnerError
from ksvd_research.runtime.manifest import RunResult

NAME = "lcrc_explicit_semantic_learning_v1"
CONFIG_PATH = C.CONFIGS / f"{C.PROTOCOL_ID}.yaml"
STUDY = C.STUDY

PREFLIGHT_PATH = C.ANALYSIS / "preflight.json"
OPERATIONAL_KEYS = ("stage", "family", "recipe", "dataset", "budget", "sections", "out_dir", "device_pref")


def _frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    import lp_common as LPC

    required = {
        "protocol_id": C.PROTOCOL_ID,
        "seed": int(0),
        "dataset_name": "ogbg-mollipo",
        "split_method": "scaffold",
        "families": list(C.SEM_FAMILIES),
        "recipes": {k: dict(v) for k, v in C.RECIPES.items()},
        "c1_recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in _c1_recipe().items()},
        "codec": {
            "n_atom_types": C.N_ATOM_TYPES, "n_bond_types": C.N_BOND_TYPES,
            "n_path_codes": C.N_PATH_CODES, "n_nbr_codes": C.N_NBR_CODES,
            "rank": C.RANK, "atom_emb_dim": C.ATOM_EMB_DIM, "bond_emb_dim": C.BOND_EMB_DIM,
            "path_hidden": C.PATH_MLP_HIDDEN, "binary_hidden": C.BINARY_MLP_HIDDEN,
            "gate_hidden": C.GATE_MLP_HIDDEN,
        },
        "params": {"free_table": C.FREE_TABLE_PARAMS, "generator": C.GENERATOR_PARAMS, "FS": C.FS_PARAMS},
        "seeds": {"generator": C.GENERATOR_SEED, "perm": C.PERM_SEED},
        "gpu_budget": {"gpu_soft_hours": C.GPU_SOFT_HOURS, "gpu_hard_hours": C.GPU_HARD_HOURS},
    }
    for key, expected in required.items():
        if key not in config:
            raise RunnerError(f"config missing frozen key {key!r}")
        if config[key] != expected:
            raise RunnerError(f"config key {key!r} drifted:\n  config={config[key]}\n  frozen={expected}")
    return config


def _c1_recipe() -> dict[str, Any]:
    import lp_common as LPC

    return dict(LPC.C1_RECIPE)


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in OPERATIONAL_KEYS}


def _env_info():
    import lp_common as LPC
    return LPC.env_info()


def _setup_determinism():
    import lp_common as LPC
    return LPC.setup_determinism()


def _git_commit():
    import lp_common as LPC
    return LPC.git_commit()


def _sha_json(obj):
    import lp_common as LPC
    return LPC.sha256_json(obj)


def code_hashes() -> dict[str, str]:
    return C.addr2line()


def environment() -> dict[str, Any]:
    return _env_info()


def fingerprints(config):
    import lp_common as LPC

    manifest = C.read_json(C.HOME / "data/manifest.json")
    split = C.read_json(C.HOME / "data/split_v1.json")
    payload = {
        "dataset": LPC.DATASET, "split_method": LPC.OGB_SPLIT_METHOD,
        "mol_csv_sha256": manifest["files"]["mol_csv"]["sha256"],
        "raw_artifact_sha256": manifest["artifacts"]["raw_attributes"]["sha256"],
        "labels_sha256": manifest["artifacts"]["labels"]["sha256"],
        "split_file_sha256": split["source_file_sha256"], "split_sizes": split["sizes"],
        "n_molecules": LPC.N_MOLECULES,
    }
    split_fingerprint = _sha_json({"rows_sha256": split["rows_sha256"], "method": LPC.OGB_SPLIT_METHOD})
    return payload, split_fingerprint


def stage_prepare(config, context) -> RunResult:
    import se_data as SED
    import lp_data as LPD

    _setup_determinism()
    started = time.perf_counter()
    base = LPD.build_view()
    budgets = {b: SED.lipo_budget_view(b, base=base).audit()["budget"] for b in ("N100", "N50", "N25")}
    payload = {
        "schema": "lcrc-explicit-semantic-learning-prepare-v1",
        "protocol_id": C.PROTOCOL_ID, "created_utc": C.utcnow(),
        "git_commit": _git_commit(), "data": base.audit(), "budgets": budgets,
        "codec": SED_codec(),
        "seconds": float(time.perf_counter() - started), "environment": environment(),
    }
    C.write_json(C.ANALYSIS / "prepare.json", payload)
    return RunResult(metrics={"n_train": len(base.rows["train"]), "budgets": {k: v["n_fit"] for k, v in budgets.items()}},
                     artifacts=[C.track_rel(C.ANALYSIS / "prepare.json")])


def SED_codec():
    import se_tables as T

    return T.codec_roundtrip()


def stage_preflight(config, context) -> RunResult:
    import torch

    import se_checks as CH

    device = torch.device(str(config.get("device_pref", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    _setup_determinism()
    sections = list(config.get("sections") or CH.SECTIONS)
    payload = CH.run_sections(device, sections)
    payload["environment"] = environment()
    payload["git_commit"] = _git_commit()
    checks = CH.collect_checks(payload)
    payload["checks"] = checks
    payload["all_passed"] = all(checks.values())
    C.write_json(PREFLIGHT_PATH, payload)
    return RunResult(status="completed" if payload["all_passed"] else "failed",
                     metrics={"all_passed": payload["all_passed"], "failed": [k for k, v in checks.items() if not v]},
                     artifacts=[C.track_rel(PREFLIGHT_PATH)],
                     error=None if payload["all_passed"] else "preflight failed")


def stage_train(config, context) -> RunResult:
    import torch

    import se_train as TR

    device = torch.device(str(config.get("device_pref", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA requested but unavailable")
    cfg = {"family": config["family"], "recipe": config.get("recipe", "R"),
           "dataset": config.get("dataset", "lipo"), "budget": config.get("budget", "N100")}
    name = C.arm_name(cfg["family"], cfg["recipe"], cfg["dataset"], cfg["budget"])
    out_dir = Path(config.get("out_dir") or (C.RUNS / f"arm_{name}"))
    payload = TR.train_run(cfg, device=device, out_dir=out_dir)
    s = payload["summary"]
    return RunResult(metrics={
        "family": cfg["family"], "recipe": cfg["recipe"], "dataset": cfg["dataset"], "budget": cfg["budget"],
        "best_epoch": s["best_valid"]["epoch"],
        "best_valid_rmse": s["best_valid"]["valid"]["rmse"], "best_test_rmse": s["best_valid"]["test"]["rmse"],
        "grid_valid_rmse": (s.get("grid_best") or {}).get("valid_rmse"),
        "grid_test_rmse": (s.get("grid_best") or {}).get("test_rmse"),
        "parameters": payload["model"]["parameters"], "gpu_hours": payload["timing"]["gpu_hours_charged"],
    }, artifacts=[C.track_rel(out_dir / "metrics.json"), C.track_rel(out_dir / "curve.csv")])


def stage_analyze(config, context) -> RunResult:
    import se_analysis as A

    started = time.perf_counter()
    payload = A.run_all()
    return RunResult(metrics={**payload.get("metrics", {}), "seconds": float(time.perf_counter() - started)},
                     artifacts=payload.get("artifacts", []))


STAGES = {"prepare": stage_prepare, "preflight": stage_preflight, "train": stage_train, "analyze": stage_analyze}


def run(config, context) -> RunResult:
    config = _frozen_config(config)
    stage = str(config.get("stage") or "")
    if stage not in STAGES:
        raise RunnerError(f"stage must be one of {sorted(STAGES)}")
    return STAGES[stage](config, context)


def build_runner() -> Runner:
    import sys

    return Runner(name=NAME, default_config=CONFIG_PATH, default_study=STUDY,
                  description="LCRC explicit semantic parameter sharing on ogbg-mollipo; LS/FS/LP/FP + controls.",
                  run_module=sys.modules[__name__])
