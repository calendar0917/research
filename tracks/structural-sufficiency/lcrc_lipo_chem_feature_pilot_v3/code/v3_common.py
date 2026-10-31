"""Shared constants, paths and integrity guards for the v3 chemistry pilot.

Scope of this round
-------------------
A single exploratory question: does adding label-free global physicochemical
information to the LCRC-A / LCRC-C global channel (``rho`` input) improve
Fold-1 seed-0 R3 dev performance on Lipophilicity?

The round is a strictly downstream consumer of the frozen v2 round.  It does
not change v1/v2 code, checkpoints, predictions or reports.  Every v3 write
goes to this directory, and every write-behaviour test uses a temporary
directory.

Frozen historical comparators (v2, Fold 1, seed 0, R3)
------------------------------------------------------
* ``F1_A_sel_s0``: A, dev RMSE 0.7403427006133139, dev MAE 0.546694119695902
* ``F1_C_sel_s0``: C, dev RMSE 0.7264913055478032, dev MAE 0.5387064120516623

Both were executed on ``res`` / ``res-gpu0`` with an A100-SXM4-40GB and
torch 2.5.1+cu124.  The v3 GPU pilot must record its own execution regime and
must not mix regimes silently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

V1 = REPO / "tracks/structural-sufficiency/lcrc_lipo_screen_v1"
V2 = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
V2_CODE = V2 / "code"
V2_ANALYSIS = V2 / "analysis"
V2_RUNS = V2_ANALYSIS / "runs"

if str(V2_CODE) not in sys.path:
    sys.path.insert(0, str(V2_CODE))

import v2_common as C  # noqa: E402
import v2_baselines as B  # noqa: E402
import v2_data as D  # noqa: E402
import v2_evaluate as E  # noqa: E402
import v2_folds as F  # noqa: E402
import v2_models as M  # noqa: E402
import v2_training as T  # noqa: E402

#: Explicit re-exports so downstream v3 modules do not reach through ``C.C``.
DESCRIPTOR_13 = C.DESCRIPTOR_13
SPLIT_SIZES = C.SPLIT_SIZES
BATCH_SIZE = C.BATCH_SIZE
EPOCHS = C.EPOCHS
SOUP_LAST = C.SOUP_LAST
RECIPES = C.RECIPES
prep_data = C.prep_data
load_frozen_spec = C.load_frozen_spec
load_frozen_norm = C.load_frozen_norm
load_frozen_split = C.load_frozen_split
load_fit_cache = C.load_fit_cache
load_fit_labels = C.load_fit_labels

ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
RUNS = ANALYSIS / "runs"
CONFIG_PATH = CONFIGS / "lcrc_lipo_chem_feature_pilot_v3.yaml"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze_v3.json"
GPU_AUTHORIZATION = ANALYSIS / "gpu_authorization_v3.json"
RUN_REGISTRY = ANALYSIS / "run_registry_v3.json"
V2_MANIFEST = ANALYSIS / "v2_immutable_manifest_v3.json"

ROUND_ID = "lcrc-lipo-chem-feature-pilot-v3"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen design constants
# ---------------------------------------------------------------------------

FOLD = 1
SEED = 0
RECIPE = "R3"

#: Exactly the two chemistry feature versions requested by the task book.
CHEM12_NAMES = (
    "TPSA",
    "MolWt",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "MolMR",
)
CHEM13_NAMES = CHEM12_NAMES + ("MolLogP",)

GLOBAL_DIM_BASE = 42
MATERIALITY_RMSE = 0.03
TAIL_FRACTION = 0.02
TAIL_MIN_RETAINED_FRACTION = 0.5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261101

#: This pilot only.  The historical v1/v2 GPU hours are reference-only and are
#: not part of this cap.
GPU_HARD_CAP_HOURS = 0.30
HISTORICAL_V1_V2_HOURS_REFERENCE_ONLY = 0.5539 + 2.4647

ARMS = ("A_Chem12", "A_Chem13", "C_Chem12")
UNCONDITIONAL_ARMS = ("A_Chem12", "A_Chem13")
CONDITIONAL_ARM = "C_Chem12"

ARM_VARIANT = {
    "A_Chem12": ("A", "chem12"),
    "A_Chem13": ("A", "chem13"),
    "C_Chem12": ("C", "chem12"),
}

#: v2 Fold-1 seed-0 R3 historical anchors.  The exact values are re-read from
#: the immutable v2 train.json; these constants are the report-level checks.
HISTORICAL = {
    "A": {
        "key": "F1_A_sel_s0",
        "arm": "A",
        "recipe": "R3",
        "fold": 1,
        "seed": 0,
        "dev_rmse": 0.7403427006133139,
        "dev_mae": 0.546694119695902,
        "train_rmse": 0.2696302152948522,
        "train_mae": 0.20082336499368744,
        "report_rmse_4dp": 0.74034,
    },
    "C": {
        "key": "F1_C_sel_s0",
        "arm": "C",
        "recipe": "R3",
        "fold": 1,
        "seed": 0,
        "dev_rmse": 0.7264913055478032,
        "dev_mae": 0.5387064120516623,
        "train_rmse": 0.299891945560453,
        "train_mae": 0.21867214653479058,
        "report_rmse_4dp": 0.72649,
    },
}

#: Files that pin the historical comparators and the v2 freeze.  These hashes
#: were computed before any v3 work started.
V2_HISTORICAL_FILES = {
    "analysis/protocol_freeze_v2.json": "76524f0e9dfc3b6ca65850d2e162d650b9195455fa5f33d2f0068de49b49c354",
    "analysis/predictions/MANIFEST.json": "1bc705295b4edb117818f00fa26d02e93da0b72f55e88c9f9afb8aad6baaf7f2",
    "analysis/gpu_ledger_v2.json": "e6f21bb7bddba695cd30c28a62fee589146af04f003716f718fec6eb02bddc2d",
    "analysis/runs/F1_A_sel_s0/train.json": "74f87fe8b071e9d7a17d8230d64d5aa81ddaa3a79b4f629fcce619b9a903dea2",
    "analysis/runs/F1_A_sel_s0/soup.pt": "8ee2c8324955586f0120121119313c73333720786a11a42b4540a093a3f0ca8d",
    "analysis/runs/F1_A_sel_s0/predictions.npz": "5c76b929763c79c3845deabd12b17a5864bfb8426a7ec6190c2844a270860756",
    "analysis/runs/F1_C_sel_s0/train.json": "8b6ab8ae71f4b50b92d1988a5ea5c4086b737a2f45720d8aeb00d8e04a5a81fa",
    "analysis/runs/F1_C_sel_s0/soup.pt": "7b1fdab904a29b644ff22aadc1eb54c8e937b8f4a3dbbb28a0fd2506b46de691",
    "analysis/runs/F1_C_sel_s0/predictions.npz": "a9fa4f427141c61ac11b8b697be979d3e4ee2b513279bd923de8957960816125",
    "cache/features_v2.npz": "a0baf161fe1a49e1f263acb2055612e5564d3d292e39419a1685e1168b2130bc",
    "cache/features_v2.json": "d6cf8ec1058cf334708907ec6b22207e8a16127e2cef62d92c3f05181621d9a7",
}

# ---------------------------------------------------------------------------
# hashing / JSON helpers (v1/v2 implementations are reused verbatim)
# ---------------------------------------------------------------------------

sha256_file = C.sha256_file
sha256_text = C.sha256_text
read_json = C.read_json
write_json = C.write_json
stable_json = C.stable_json
track_rel = C.track_rel
utcnow = C.utcnow
git_commit = C.git_commit
git_dirty = C.git_dirty
ensure_dir = C.ensure_dir
log = C.log
file_record = C.file_record
verify_sha256 = C.verify_sha256
env_info = C.env_info
row_sha256_bytes = C.row_sha256_bytes
row_sha256_text = C.row_sha256_text


def source_dirty() -> bool:
    """True iff a tracked v3 source/config/test file is modified."""
    rel = track_rel(EXP_DIR)
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                f"{rel}/code",
                f"{rel}/tests",
                f"{rel}/configs",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_text(stable_json(cfg))


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    def need(actual, expected, name):
        if actual != expected:
            raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")

    need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    need(cfg.get("round_id"), ROUND_ID, "round_id")
    need(int(cfg["fold"]), FOLD, "fold")
    need(int(cfg["seed"]), SEED, "seed")
    need(str(cfg["recipe"]), RECIPE, "recipe")
    need(tuple(cfg["features"]["chem12"]), CHEM12_NAMES, "features.chem12")
    need(tuple(cfg["features"]["chem13"]), CHEM13_NAMES, "features.chem13")
    need(int(cfg["features"]["global_dim_base"]), GLOBAL_DIM_BASE, "features.global_dim_base")
    need(str(cfg["features"]["standardisation"]), "fold1_train_only", "features.standardisation")
    need(float(cfg["decision"]["materiality_rmse"]), MATERIALITY_RMSE, "decision.materiality_rmse")
    need(float(cfg["decision"]["tail_fraction"]), TAIL_FRACTION, "decision.tail_fraction")
    need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")
    need(tuple(cfg["arms"]["unconditional"]), UNCONDITIONAL_ARMS, "arms.unconditional")
    need(str(cfg["arms"]["conditional"]), CONDITIONAL_ARM, "arms.conditional")
    for name, row in C.RECIPES.items():
        need(float(cfg["training"][name]["peak_lr"]), row["peak_lr"], f"training.{name}.peak_lr")
        need(float(cfg["training"][name]["weight_decay"]), row["weight_decay"], f"training.{name}.weight_decay")
        need(float(cfg["training"][name]["dropout"]), row["dropout"], f"training.{name}.dropout")
    need(int(cfg["training"]["batch_size"]), C.BATCH_SIZE, "training.batch_size")
    need(int(cfg["training"]["epochs"]), C.EPOCHS, "training.epochs")
    need(int(cfg["training"]["soup_last"]), C.SOUP_LAST, "training.soup_last")
    need(str(cfg["comparators"]["A"]), HISTORICAL["A"]["key"], "comparators.A")
    need(str(cfg["comparators"]["C"]), HISTORICAL["C"]["key"], "comparators.C")


# ---------------------------------------------------------------------------
# immutability
# ---------------------------------------------------------------------------


def _v2_manifest_paths() -> list[Path]:
    paths: list[Path] = []
    paths += sorted(V2.glob("code/*.py"))
    paths += sorted(V2.glob("tests/*.py"))
    paths += sorted(V2.glob("configs/*.yaml"))
    paths += [V2 / "REPORT.md", V2 / "DECISION.md"]
    paths += sorted(V2_ANALYSIS.glob("*.json"))
    paths += sorted(V2_ANALYSIS.glob("predictions/*.npz"))
    paths += sorted(V2_ANALYSIS.glob("predictions/*.json"))
    paths += sorted(V2_ANALYSIS.glob("runs/*/train.json"))
    paths += sorted(V2_ANALYSIS.glob("runs/*/predictions.npz"))
    paths += sorted(V2_ANALYSIS.glob("runs/*/soup.pt"))
    paths += sorted(V2.glob("cache/*"))
    return [p for p in paths if p.exists()]


def build_v2_manifest() -> dict[str, Any]:
    entries = {track_rel(p): sha256_file(p) for p in _v2_manifest_paths()}
    return {
        "protocol_id": PROTOCOL_ID,
        "created_utc": utcnow(),
        "scope": "v2 source, config, analysis JSON, prediction arrays, run train.json and cache",
        "n_files": len(entries),
        "files": entries,
        "v2_protocol_freeze_sha256": sha256_file(V2_ANALYSIS / "protocol_freeze_v2.json"),
        "v2_prediction_manifest_sha256": sha256_file(V2_ANALYSIS / "predictions/MANIFEST.json"),
    }


def assert_v1_unchanged() -> dict[str, Any]:
    return C.assert_v1_unchanged()


def assert_v2_unchanged(*, use_manifest: bool = True) -> dict[str, Any]:
    """Verify the v2 freeze, its code hashes, and (optionally) every v2 file."""
    # ``v2_common.verify_protocol_freeze`` re-hashes the v2 code/config against
    # the frozen v2 protocol and re-runs the v1 guard.
    frozen = C.verify_protocol_freeze(require_clean=False)
    checks = []
    for rel, expected in V2_HISTORICAL_FILES.items():
        path = V2 / rel
        actual = sha256_file(path)
        ok = actual == expected
        checks.append({"path": f"v2:{rel}", "expected": expected, "actual": actual, "ok": ok})
    if use_manifest and V2_MANIFEST.exists():
        manifest = read_json(V2_MANIFEST)
        for rel, expected in sorted(manifest.get("files", {}).items()):
            path = REPO / rel
            actual = sha256_file(path) if path.exists() else None
            ok = actual == expected
            checks.append({"path": f"v2_manifest:{rel}", "expected": expected, "actual": actual, "ok": ok})
    ok = all(c["ok"] for c in checks)
    return {
        "ok": ok,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "checks": checks,
        "v2_freeze_protocol_id": frozen.get("protocol_id"),
        "manifest_used": bool(use_manifest and V2_MANIFEST.exists()),
    }


# ---------------------------------------------------------------------------
# code hashes / freeze
# ---------------------------------------------------------------------------


def code_file_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(list((EXP_DIR / "code").glob("*.py")) + list((EXP_DIR / "tests").glob("*.py"))):
        out[track_rel(path)] = sha256_file(path)
    out[track_rel(CONFIG_PATH)] = sha256_file(CONFIG_PATH)
    return out


def build_protocol_freeze(cfg: Mapping[str, Any]) -> dict[str, Any]:
    v1 = assert_v1_unchanged()
    if not v1["ok"]:
        raise PermissionError(f"v1 frozen artifacts changed: {[c for c in v1['checks'] if not c['ok']]}")
    v2 = assert_v2_unchanged(use_manifest=True)
    if not v2["ok"]:
        raise PermissionError(f"v2 frozen artifacts changed: {[c for c in v2['checks'] if not c['ok']]}")
    artifacts = {}
    for name in (
        "chem_feature_audit_v3.json",
        "fold_alignment_v3.json",
        "norm_audit_v3.json",
        "forward_equivalence_v3.json",
        "denorm_check_v3.json",
        "descriptive_groups_v3.json",
        "temp_training_v3.json",
        "frozen_artifacts_v3.json",
        "frozen_extras_v3.json",
        "cpu_gates_v3.json",
        "run_registry_v3.json",
        "budget_v3.json",
        "v2_immutable_manifest_v3.json",
    ):
        path = ANALYSIS / name
        artifacts[name] = file_record(path) if path.exists() else None
    extras_npz = ANALYSIS / "frozen_extras_v3.npz"
    artifacts["frozen_extras_v3.npz"] = file_record(extras_npz) if extras_npz.exists() else None
    return {
        "protocol_id": PROTOCOL_ID,
        "round_id": ROUND_ID,
        "generated_utc": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": source_dirty(),
        "git_dirty_worktree": git_dirty(),
        "config_hash": config_hash(cfg),
        "config_path": track_rel(CONFIG_PATH),
        "code_hashes": code_file_hashes(),
        "artifacts": artifacts,
        "v1_immutability": v1,
        "v2_immutability": v2,
        "historical_comparators": {
            "A": {**HISTORICAL["A"], "train_json": file_record(V2_RUNS / HISTORICAL["A"]["key"] / "train.json")},
            "C": {**HISTORICAL["C"], "train_json": file_record(V2_RUNS / HISTORICAL["C"]["key"] / "train.json")},
        },
        "frozen_constants": {
            "fold": FOLD,
            "seed": SEED,
            "recipe": RECIPE,
            "recipe_config": dict(C.RECIPES[RECIPE]),
            "chem12": list(CHEM12_NAMES),
            "chem13": list(CHEM13_NAMES),
            "global_dim_base": GLOBAL_DIM_BASE,
            "descriptor_source": "rdkit.Chem.Descriptors.CalcMolDescriptors",
            "rdkit_version": "2026.09.1",
            "canonical_smiles": "RDKit MolToSmiles(canonical=True), sha256 per row",
            "nonfinite_policy": "train-fold median; zero training variance -> scale 1.0",
            "standardisation": "fold1 train only (mean/std); dev never used",
            "training": dict(cfg["training"]),
            "decision": dict(cfg["decision"]),
            "budget": dict(cfg["budget"]),
            "arms": dict(cfg["arms"]),
            "training_loop": {
                "implementation": "v2_training.train_job reused verbatim",
                "extension": "v3 builder dispatch + global_extra_* + spec_override + norm_override",
                "message_passing_changed": False,
                "s2_semantics_changed": False,
                "atom51_bond13_scalar9_changed": False,
            },
        },
        "gpu_authorized": False,
        "gpu_hours_used": 0.0,
        "gpu_hard_cap_hours": GPU_HARD_CAP_HOURS,
    }


def verify_protocol_freeze(*, require_clean: bool = False) -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError(f"missing protocol freeze {track_rel(PROTOCOL_FREEZE)}")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("v3 protocol freeze has the wrong protocol id")
    cfg = load_config()
    if frozen["config_hash"] != config_hash(cfg):
        raise PermissionError("v3 config changed after the protocol freeze")
    current = code_file_hashes()
    if current != frozen["code_hashes"]:
        changed = sorted(k for k in set(current) | set(frozen["code_hashes"]) if current.get(k) != frozen["code_hashes"].get(k))
        raise PermissionError(f"v3 code changed after the protocol freeze: {changed}")
    if require_clean and git_dirty():
        raise PermissionError("formal run requires a clean committed worktree")
    v1 = assert_v1_unchanged()
    if not v1["ok"]:
        raise PermissionError("v1 frozen artifacts changed")
    v2 = assert_v2_unchanged()
    if not v2["ok"]:
        raise PermissionError("v2 frozen artifacts changed")
    return frozen


def freeze_sha256() -> str:
    return sha256_file(PROTOCOL_FREEZE)


__all__ = [name for name in dir() if not name.startswith("_")]
