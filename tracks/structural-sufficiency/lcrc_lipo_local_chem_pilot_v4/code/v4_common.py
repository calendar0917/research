"""Shared constants, paths and integrity guards for the v4 local-chemistry pilot.

Research question (fixed by the researcher, not by this code)
-------------------------------------------------------------
Is injecting hydrophobicity / polarity / hydrogen-bond information into the
*atom-level* representation of the frozen LCRC-A more effective than the v3
attempt, which appended molecule-level descriptors to the ``rho`` (global)
input?

This module is a strictly downstream consumer of the frozen v1/v2/v3 rounds.
Nothing in v1, v2 or v3 is written.  Every v4 write goes to this directory and
every write-behaviour test uses a temporary directory.

Frozen historical comparator (v2, Fold 1, seed 0, R3, arm A)
------------------------------------------------------------
* ``F1_A_sel_s0``: dev RMSE 0.7403427006133139, dev MAE 0.546694119695902
* train RMSE 0.2696302152948522, train MAE 0.20082336499368744
Executed on ``res`` / ``res-gpu0`` with an A100-SXM4-40GB and torch 2.5.1+cu124.
The v4 GPU pilot records its own execution regime and must not mix regimes
silently.
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
V3 = REPO / "tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3"
V3_ANALYSIS = V3 / "analysis"
V3_RUNS = V3_ANALYSIS / "runs"

if str(V2_CODE) not in sys.path:
    sys.path.insert(0, str(V2_CODE))

import v2_common as C  # noqa: E402
import v2_data as D  # noqa: E402
import v2_folds as F  # noqa: E402
import v2_models as M  # noqa: E402
import v2_training as T  # noqa: E402
import v2_baselines as B  # noqa: E402

ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
RUNS = ANALYSIS / "runs"
CACHE = EXP_DIR / "cache"
CONFIG_PATH = CONFIGS / "lcrc_lipo_local_chem_pilot_v4.yaml"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze_v4.json"
GPU_AUTHORIZATION = ANALYSIS / "gpu_authorization_v4.json"
RUN_REGISTRY = ANALYSIS / "run_registry_v4.json"
V3_MANIFEST = ANALYSIS / "v3_immutable_manifest_v4.json"

ROUND_ID = "lcrc-lipo-local-chem-pilot-v4"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen design constants
# ---------------------------------------------------------------------------

FOLD = 1
SEED = 0
RECIPE = "R3"

ATOM_DIM_BASE = 51
ATOM_DIM_AUGMENTED = 55
N_ATOM_EXTRA = ATOM_DIM_AUGMENTED - ATOM_DIM_BASE  # 4
ATOM_HIDDEN = 128
EXPECTED_PARAMETER_DELTA = ATOM_HIDDEN * N_ATOM_EXTRA  # 512

ATOMCHEM_NAMES = (
    "crippen_logp_atom",
    "tpsa_atom",
    "is_hbd",
    "is_hba",
)
ATOMCHEM_RAW_NAMES = (
    "crippen_atom_logp_heavy_incl_folded_H",
    "tpsa_atom_contribution",
    "donor_atom_flag",
    "acceptor_atom_flag",
)

ATOMCHEM_DESCRIPTIONS = {
    "crippen_logp_atom": (
        "RDKit Crippen atom logP contribution; implicit/explicit H contributions folded onto the parent heavy atom "
        "so that the heavy-atom sum equals RDKit Crippen.MolLogP"
    ),
    "tpsa_atom": (
        "RDKit rdMolDescriptors._CalcTPSAContribs per heavy atom; sum equals RDKit Descriptors.TPSA "
        "(S/P excluded convention)"
    ),
    "is_hbd": "1 if the heavy atom is the single atom of any RDKit BaseFeatures.fdef Donor feature",
    "is_hba": "1 if the heavy atom is the single atom of any RDKit BaseFeatures.fdef Acceptor feature",
}

GLOBAL_DIM_BASE = 42
MATERIALITY_RMSE = 0.03
TAIL_FRACTION = 0.02
TAIL_MIN_RETAINED_FRACTION = 0.5
ALIGNMENT_GAP_MATERIAL = 0.015
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261102

#: This pilot only.  Historical v1/v2/v3 GPU hours are reference-only.
GPU_HARD_CAP_HOURS = 0.20
HISTORICAL_V1_V2_V3_HOURS_REFERENCE_ONLY = 0.5539 + 2.4647 + 0.076225

ARMS = ("A_AtomChem4", "A_BroadcastChem4")
ARM_VARIANT = {
    "A_AtomChem4": "atomchem4",
    "A_BroadcastChem4": "broadcastchem4",
}
ARM_DESCRIPTION = {
    "A_AtomChem4": "P1: true per-atom AtomChem4 correspondence",
    "A_BroadcastChem4": "P2: molecule-mean AtomChem4 broadcast to every atom (same features, no atom correspondence)",
}

HISTORICAL = {
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
}

#: v3 froze a manifest of every v2 file; v4 re-verifies that manifest itself.
V3_FREEZE_SHA256 = "72f2e2142be7eedfe056bcbb948118d0d32ce73f3167fbb356847cf9acbfa14c"
V3_V2_MANIFEST_SHA256 = "2de5d180a0e32b6f4ee4f7b1495c3108b278a32477e5cbd1c6a27382c7787d5b"
V3_FROZEN_EXTRAS_NPZ_SHA256 = "fd73ccd7b4b430e64e6e98148490c319a4be0e8521ae4aa47a890c57c6fd767b"
V2_FREEZE_SHA256 = "76524f0e9dfc3b6ca65850d2e162d650b9195455fa5f33d2f0068de49b49c354"

#: Files that pin the historical comparator and the v2 freeze (same list v3 used).
V2_HISTORICAL_FILES = {
    "analysis/protocol_freeze_v2.json": V2_FREEZE_SHA256,
    "analysis/predictions/MANIFEST.json": "1bc705295b4edb117818f00fa26d02e93da0b72f55e88c9f9afb8aad6baaf7f2",
    "analysis/gpu_ledger_v2.json": "e6f21bb7bddba695cd30c28a62fee589146af04f003716f718fec6eb02bddc2d",
    "analysis/runs/F1_A_sel_s0/train.json": "74f87fe8b071e9d7a17d8230d64d5aa81ddaa3a79b4f629fcce619b9a903dea2",
    "analysis/runs/F1_A_sel_s0/soup.pt": "8ee2c8324955586f0120121119313c73333720786a11a42b4540a093a3f0ca8d",
    "analysis/runs/F1_A_sel_s0/predictions.npz": "5c76b929763c79c3845deabd12b17a5864bfb8426a7ec6190c2844a270860756",
    "cache/features_v2.npz": "a0baf161fe1a49e1f263acb2055612e5564d3d292e39419a1685e1168b2130bc",
    "cache/features_v2.json": "d6cf8ec1058cf334708907ec6b22207e8a16127e2cef62d92c3f05181621d9a7",
}

# ---------------------------------------------------------------------------
# hashing / JSON helpers (v1/v2 implementations reused verbatim)
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

prep_data = C.prep_data
prep_model = C.prep_model
load_frozen_spec = C.load_frozen_spec
load_frozen_norm = C.load_frozen_norm
load_frozen_split = C.load_frozen_split
load_fit_cache = C.load_fit_cache
load_fit_labels = C.load_fit_labels
SPLIT_SIZES = C.SPLIT_SIZES
BATCH_SIZE = C.BATCH_SIZE
EPOCHS = C.EPOCHS
SOUP_LAST = C.SOUP_LAST
RECIPES = C.RECIPES


def source_dirty() -> bool:
    """True iff a tracked v4 source/config/test file is modified."""
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
    need(tuple(cfg["atomchem"]["names"]), ATOMCHEM_NAMES, "atomchem.names")
    need(int(cfg["atomchem"]["atom_dim_base"]), ATOM_DIM_BASE, "atomchem.atom_dim_base")
    need(int(cfg["atomchem"]["atom_dim_augmented"]), ATOM_DIM_AUGMENTED, "atomchem.atom_dim_augmented")
    need(str(cfg["atomchem"]["standardisation"]), "fold1_train_only", "atomchem.standardisation")
    need(float(cfg["decision"]["materiality_rmse"]), MATERIALITY_RMSE, "decision.materiality_rmse")
    need(float(cfg["decision"]["alignment_gap_material"]), ALIGNMENT_GAP_MATERIAL, "decision.alignment_gap_material")
    need(float(cfg["decision"]["tail_fraction"]), TAIL_FRACTION, "decision.tail_fraction")
    need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")
    need(tuple(cfg["arms"]["unconditional"]), (ARMS[0],), "arms.unconditional")
    need(str(cfg["arms"]["conditional"]), ARMS[1], "arms.conditional")
    for name, row in C.RECIPES.items():
        need(float(cfg["training"][name]["peak_lr"]), row["peak_lr"], f"training.{name}.peak_lr")
        need(float(cfg["training"][name]["weight_decay"]), row["weight_decay"], f"training.{name}.weight_decay")
        need(float(cfg["training"][name]["dropout"]), row["dropout"], f"training.{name}.dropout")
    need(int(cfg["training"]["batch_size"]), C.BATCH_SIZE, "training.batch_size")
    need(int(cfg["training"]["epochs"]), C.EPOCHS, "training.epochs")
    need(int(cfg["training"]["soup_last"]), C.SOUP_LAST, "training.soup_last")
    need(str(cfg["comparator"]["key"]), HISTORICAL["key"], "comparator.key")


# ---------------------------------------------------------------------------
# immutability: v1, v2, v3
# ---------------------------------------------------------------------------


def assert_v1_unchanged() -> dict[str, Any]:
    return C.assert_v1_unchanged()


def assert_v2_unchanged() -> dict[str, Any]:
    """Verify the v2 freeze, the pinned files, and the frozen v3-built v2 manifest."""
    frozen = C.verify_protocol_freeze(require_clean=False)
    checks = []
    for rel, expected in V2_HISTORICAL_FILES.items():
        path = V2 / rel
        actual = sha256_file(path)
        checks.append({"path": f"v2:{rel}", "expected": expected, "actual": actual, "ok": actual == expected})
    manifest_path = V3_ANALYSIS / "v2_immutable_manifest_v3.json"
    manifest_sha = sha256_file(manifest_path)
    checks.append(
        {
            "path": "v3:v2_immutable_manifest_v3.json",
            "expected": V3_V2_MANIFEST_SHA256,
            "actual": manifest_sha,
            "ok": manifest_sha == V3_V2_MANIFEST_SHA256,
        }
    )
    manifest = read_json(manifest_path)
    for rel, expected in sorted(manifest.get("files", {}).items()):
        path = REPO / rel
        actual = sha256_file(path) if path.exists() else None
        checks.append({"path": f"v2_manifest:{rel}", "expected": expected, "actual": actual, "ok": actual == expected})
    ok = all(c["ok"] for c in checks)
    return {
        "ok": ok,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "checks": checks,
        "v2_freeze_protocol_id": frozen.get("protocol_id"),
        "v2_freeze_sha256": sha256_file(V2_ANALYSIS / "protocol_freeze_v2.json"),
        "manifest_n_files": len(manifest.get("files", {})),
    }


def _v3_manifest_paths() -> list[Path]:
    paths: list[Path] = []
    paths += sorted(V3.glob("code/*.py"))
    paths += sorted(V3.glob("tests/*.py"))
    paths += sorted(V3.glob("configs/*.yaml"))
    paths += [V3 / "REPORT.md", V3 / "DECISION.md"]
    paths += sorted(V3_ANALYSIS.glob("*.json"))
    paths += sorted(V3_ANALYSIS.glob("*.npz"))
    paths += sorted(V3_ANALYSIS.glob("*.txt"))
    paths += sorted(V3_ANALYSIS.glob("*.csv"))
    paths += sorted(V3_RUNS.glob("*/*.json"))
    paths += sorted(V3_RUNS.glob("*/*.npz"))
    paths += sorted(V3_RUNS.glob("*/*.pt"))
    paths += sorted(V3.glob("cache/*"))
    return [p for p in paths if p.exists()]


def build_v3_manifest() -> dict[str, Any]:
    entries = {track_rel(p): sha256_file(p) for p in _v3_manifest_paths()}
    return {
        "protocol_id": PROTOCOL_ID,
        "created_utc": utcnow(),
        "scope": "v3 source, config, reports, analysis JSON/NPZ/CSV, run artifacts and cache",
        "n_files": len(entries),
        "files": entries,
        "v3_protocol_freeze_sha256": sha256_file(V3_ANALYSIS / "protocol_freeze_v3.json"),
        "v3_frozen_extras_npz_sha256": sha256_file(V3_ANALYSIS / "frozen_extras_v3.npz"),
    }


def assert_v3_unchanged(*, use_manifest: bool = True) -> dict[str, Any]:
    checks = []
    freeze_path = V3_ANALYSIS / "protocol_freeze_v3.json"
    actual = sha256_file(freeze_path)
    checks.append(
        {"path": "v3:protocol_freeze_v3.json", "expected": V3_FREEZE_SHA256, "actual": actual, "ok": actual == V3_FREEZE_SHA256}
    )
    if actual == V3_FREEZE_SHA256:
        frozen = read_json(freeze_path)
        for rel, expected in sorted(frozen.get("code_hashes", {}).items()):
            path = REPO / rel
            got = sha256_file(path) if path.exists() else None
            checks.append({"path": f"v3_code:{rel}", "expected": expected, "actual": got, "ok": got == expected})
        for name, record in sorted(frozen.get("artifacts", {}).items()):
            if not record:
                continue
            path = V3_ANALYSIS / name
            got = sha256_file(path) if path.exists() else None
            checks.append(
                {"path": f"v3_artifact:{name}", "expected": record["sha256"], "actual": got, "ok": got == record["sha256"]}
            )
    extras = sha256_file(V3_ANALYSIS / "frozen_extras_v3.npz")
    checks.append(
        {
            "path": "v3:frozen_extras_v3.npz",
            "expected": V3_FROZEN_EXTRAS_NPZ_SHA256,
            "actual": extras,
            "ok": extras == V3_FROZEN_EXTRAS_NPZ_SHA256,
        }
    )
    if use_manifest and V3_MANIFEST.exists():
        manifest = read_json(V3_MANIFEST)
        for rel, expected in sorted(manifest.get("files", {}).items()):
            path = REPO / rel
            got = sha256_file(path) if path.exists() else None
            checks.append({"path": f"v3_manifest:{rel}", "expected": expected, "actual": got, "ok": got == expected})
    ok = all(c["ok"] for c in checks)
    return {
        "ok": ok,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "checks": checks,
        "manifest_used": bool(use_manifest and V3_MANIFEST.exists()),
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
    v2 = assert_v2_unchanged()
    if not v2["ok"]:
        raise PermissionError(f"v2 frozen artifacts changed: {[c for c in v2['checks'] if not c['ok']]}")
    v3 = assert_v3_unchanged(use_manifest=True)
    if not v3["ok"]:
        raise PermissionError(f"v3 frozen artifacts changed: {[c for c in v3['checks'] if not c['ok']]}")

    artifacts = {}
    for name in (
        "atomchem_audit_v4.json",
        "fold_alignment_v4.json",
        "atom_norm_v4.json",
        "forward_equivalence_v4.json",
        "broadcast_control_v4.json",
        "denorm_check_v4.json",
        "descriptive_v4.json",
        "temp_training_v4.json",
        "frozen_extras_v4.json",
        "cpu_gates_v4.json",
        "run_registry_v4.json",
        "budget_v4.json",
        "v3_immutable_manifest_v4.json",
    ):
        path = ANALYSIS / name
        artifacts[name] = file_record(path) if path.exists() else None
    extras_npz = ANALYSIS / "frozen_extras_v4.npz"
    artifacts["frozen_extras_v4.npz"] = file_record(extras_npz) if extras_npz.exists() else None
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
        "v3_immutability": v3,
        "historical_comparator": {
            **HISTORICAL,
            "train_json": file_record(V2_RUNS / HISTORICAL["key"] / "train.json"),
            "predictions": file_record(V2_RUNS / HISTORICAL["key"] / "predictions.npz"),
            "soup": file_record(V2_RUNS / HISTORICAL["key"] / "soup.pt"),
        },
        "frozen_constants": {
            "fold": FOLD,
            "seed": SEED,
            "recipe": RECIPE,
            "recipe_config": dict(C.RECIPES[RECIPE]),
            "atomchem_names": list(ATOMCHEM_NAMES),
            "atomchem_raw_names": list(ATOMCHEM_RAW_NAMES),
            "atom_dim_base": ATOM_DIM_BASE,
            "atom_dim_augmented": ATOM_DIM_AUGMENTED,
            "n_atom_extra": N_ATOM_EXTRA,
            "atom_hidden": ATOM_HIDDEN,
            "expected_parameter_delta": EXPECTED_PARAMETER_DELTA,
            "both_arms_identical_except_atom_feature_assignment": True,
            "standardisation": "fold1 train atoms only (mean/std); zero-variance columns -> scale 1",
            "training": dict(cfg["training"]),
            "decision": dict(cfg["decision"]),
            "budget": dict(cfg["budget"]),
            "arms": dict(cfg["arms"]),
            "training_loop": {
                "implementation": "v2_training.train_job reused verbatim",
                "extension": "v4 builder dispatch + patched v2_data.collate appending the 4 atom columns",
                "message_passing_changed": False,
                "s2_semantics_changed": False,
                "atom51_bond13_scalar9_global42_changed": False,
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
        raise PermissionError("v4 protocol freeze has the wrong protocol id")
    cfg = load_config()
    if frozen["config_hash"] != config_hash(cfg):
        raise PermissionError("v4 config changed after the protocol freeze")
    current = code_file_hashes()
    if current != frozen["code_hashes"]:
        changed = sorted(k for k in set(current) | set(frozen["code_hashes"]) if current.get(k) != frozen["code_hashes"].get(k))
        raise PermissionError(f"v4 code changed after the protocol freeze: {changed}")
    if require_clean and git_dirty():
        raise PermissionError("formal run requires a clean committed worktree")
    for name, fn in (("v1", assert_v1_unchanged), ("v2", assert_v2_unchanged), ("v3", assert_v3_unchanged)):
        report = fn()
        if not report["ok"]:
            raise PermissionError(f"{name} frozen artifacts changed")
    return frozen


def freeze_sha256() -> str:
    return sha256_file(PROTOCOL_FREEZE)


def npz_sha256(path: Path | str) -> str:
    return sha256_file(Path(path))


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


__all__ = [name for name in dir() if not name.startswith("_")]
