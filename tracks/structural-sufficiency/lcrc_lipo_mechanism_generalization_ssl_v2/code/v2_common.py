"""Shared paths, frozen constants and integrity helpers for the v2 round.

This round is a *downstream* consumer of:

* the frozen Phase 1a Lipophilicity split and Fit/Monitor label artifact;
* the frozen ``lcrc_cross_task_preparation_v1`` ``lipo_v1`` FeatureSpec/caches;
* the completed ``lcrc_lipo_screen_v1`` round, which is treated as an immutable
  historical baseline.

Nothing in the v1 directory is written by this module.  ``assert_v1_unchanged``
re-verifies the v1 freeze, the v1 prediction manifest and every file referenced
by that manifest.
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
V1_CODE = V1 / "code"
V1_ANALYSIS = V1 / "analysis"
V1_PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"

ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
CACHE = EXP_DIR / "cache"
RUNS = ANALYSIS / "runs"
PREDICTIONS = ANALYSIS / "predictions"
CONFIG_PATH = CONFIGS / "lcrc_lipo_mechanism_generalization_ssl_v2.yaml"

FOLD_FILE = ANALYSIS / "folds_v2.json"
NORM_TARGET_V2 = ANALYSIS / "norm_target_fit2352.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze_v2.json"
GPU_AUTHORIZATION = ANALYSIS / "gpu_authorization_v2.json"
RUN_REGISTRY = ANALYSIS / "run_registry_v2.json"
RECIPE_LOCK = ANALYSIS / "recipe_lock_v2.json"
SSL_GATE = ANALYSIS / "ssl_gate_v2.json"

ROUND_ID = "lcrc-lipo-mechanism-generalization-ssl-v2"
PROTOCOL_ID = ROUND_ID

#: v1 immutable artifact hashes.  These are copied from v1 REPORT/PROTOCOL.
V1_FREEZE_SHA256 = "7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8"
V1_MANIFEST_SHA256 = "ba8ad0be8c1e01395aa03e619f9fc4d0552d33ea8dabfc8370cc43c4a79ee161"
V1_MANIFEST_LOCK_SHA256 = "f083600e7b69b29a41208e9e3f9a18f117fe45a05ec36d0c582e62dd8ae757fd"
V1_MONITOR_METRICS_SHA256 = "f9df66abe7c93cdb01b65d9601f2362f01a37763f0f0bf1aa515ec600fa65779"
V1_MONITOR_PAIRED_SHA256 = "9c0e4631df52b3d133654678ed41b7edfbf15eff5d118fd2dd78bfe4157ff528"
V1_MONITOR_DIAGNOSTICS_SHA256 = "41d6b860aa64f853fd9c2ffd23d7399da910c95214aeb373896141c42765e90f"
V1_GPU_LEDGER_SHA256 = "2eca7b148e90795e74432116742abd05618e93ed5f6191aac0495348caa57268"
V1_AUTHORIZATION_SHA256 = "a2183bdf62f61041b844791f97f1be3d848e0d3c88ad9d27c0c99b820c291dcc"
V1_CPU_GATES_SHA256 = "98d8e27cd547f0a91c0a83e82854506a69c8e2162df7cd4d81c7de29da3a176c"

V1_IMMUTABLE = {
    "analysis/protocol_freeze.json": V1_FREEZE_SHA256,
    "analysis/predictions/MANIFEST.json": V1_MANIFEST_SHA256,
    "analysis/predictions/MANIFEST.sha256": V1_MANIFEST_LOCK_SHA256,
    "analysis/monitor_metrics.json": V1_MONITOR_METRICS_SHA256,
    "analysis/monitor_paired.json": V1_MONITOR_PAIRED_SHA256,
    "analysis/monitor_diagnostics.json": V1_MONITOR_DIAGNOSTICS_SHA256,
    "analysis/gpu_ledger.json": V1_GPU_LEDGER_SHA256,
    "analysis/gpu_authorization.json": V1_AUTHORIZATION_SHA256,
    "analysis/cpu_gates.json": V1_CPU_GATES_SHA256,
    # Formal v1 checkpoints and their training manifests.  The v1 freeze did
    # not include these bytes, so the v2 guard records them explicitly.
    "analysis/runs/A_s0/soup.pt": "221b20d364aea9099a070192cdb14afff179065836e4fa357982aec783e27d41",
    "analysis/runs/A_s1/soup.pt": "33383966de0bed1347df6412ebb48e50375c7827e8d27798374beb75213fd75f",
    "analysis/runs/A_s2/soup.pt": "a2da6207f7ad62a7c001178e737530ea4c7e26511c274b0fc595234ea9b40b9f",
    "analysis/runs/B_s0/soup.pt": "4b53a1cac7501cc3e8177915d93d6b06a15e26e6052a4c7c30f452fa35dd5f6f",
    "analysis/runs/B_s1/soup.pt": "35a278f720d93ce2087686c64d2c1d8d442c7b57123ef87db3405cce5466b64c",
    "analysis/runs/B_s2/soup.pt": "67143acc7b05bf8162d4719c61793b098285ca7e2b40507be2bef38e774ae926",
    "analysis/runs/C_s0/soup.pt": "fc8ea1ed4fee46ea0f591f88a94b8ca9a0e365fdd3c22e614482b47d3a6d2071",
    "analysis/runs/C_s1/soup.pt": "96343583cc434b8e2969076f9a8e117c4136ae730cfa34aac49aa568a4b0ab04",
    "analysis/runs/C_s2/soup.pt": "18d4e78627150b295051999992d42866fa56e0eb9f6e7e737c5cb2e724346933",
    "analysis/runs/A_s0/train.json": "c7047d6e77fc031df0f294a1443f6b10b3c0de22115e10b9568433342a56599f",
    "analysis/runs/A_s1/train.json": "763d6c304074f7272ae8a68c21723ed99507a2e7e3a514e3d58ba31043e9cb05",
    "analysis/runs/A_s2/train.json": "d5c062f884425f569b496433ba942a082ea2816f3cc3deabcf600b7ea4fe4918",
    "analysis/runs/B_s0/train.json": "9effa5177e3de5fc7859e3141de96ef799039cab761a03072f72fe7315b940a6",
    "analysis/runs/B_s1/train.json": "1cd1a4d1a3f0cfb4e5f58d1631c5e60745efdd9d5f0b252beb3e70cc3d84179b",
    "analysis/runs/B_s2/train.json": "e89100097b522184a1d8016684158c2c25867b10449c68ea71fd77b904507dd5",
    "analysis/runs/C_s0/train.json": "2d04e1eee1202599f76ddccc7870a30bb9f9663498d90add28d9472b7715d42c",
    "analysis/runs/C_s1/train.json": "c236698e065721bfcd9af87fe3f3b6afee2bdaed698062c80de142d3b56da536",
    "analysis/runs/C_s2/train.json": "6f0eb5628aff13afe683759a4836f2b922427f80e0698e648adee75928b51889",
}

# ---------------------------------------------------------------------------
# v2 design constants (frozen before GPU)
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2)
ARMS = ("A", "E", "C")
BASELINE_ARMS = ("P1", "P2", "P3")

BATCH_SIZE = 128
EPOCHS = 360
SOUP_LAST = 5
GRAD_CLIP = 5.0
LR_MIN = 1e-5
SHUFFLE_OFFSET = 917_000
SPLIT_SIZES = {"fit": 2352, "monitor": 504, "reserved": 504}
STEPS_PER_EPOCH_FULL = (SPLIT_SIZES["fit"] + BATCH_SIZE - 1) // BATCH_SIZE  # 19
TOTAL_UPDATES_FULL = STEPS_PER_EPOCH_FULL * EPOCHS

#: Fixed recipe grid.  ``R0`` is the v1 recipe.
RECIPES = {
    "R0": {"peak_lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.0},
    "R1": {"peak_lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.10},
    "R2": {"peak_lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.20},
    "R3": {"peak_lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.10},
}
RECIPE_ORDER = ("R0", "R1", "R2", "R3")
RECIPE_IMPROVEMENT_MIN = 0.02

#: Fold-0 recipe selection uses seeds 0/1 only.
RECIPE_SELECT_SEEDS = (0, 1)

#: SSL fixed recipe.
SSL_EPOCHS = 120
SSL_BATCH_SIZE = 128
SSL_LR = 1e-3
SSL_MIN_LR = 1e-5
SSL_WEIGHT_DECAY = 1e-4
SSL_GRAD_CLIP = 5.0
SSL_MASK_FRACTION = 0.15
SSL_VAL_FRACTION = 0.10
SSL_VAL_MASK_SEED = 20261031
SSL_HEAD_DIMS = (64, 64, 16)
SSL_GATE_MIN_NAT = 0.05

#: Decision constants.
MATERIALITY_RMSE = 0.03
H1_PROMISING_GAIN = 0.03
H2_PROMISING_GAIN = 0.02
H3_PROMISING_GAIN = 0.02
CPU_DOMINANCE_GAP = 0.07
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

#: Readable aliases used by the analysis layer.
H1_MATERIALITY = H1_PROMISING_GAIN
H2_MATERIALITY = H2_PROMISING_GAIN
H3_MATERIALITY = H3_PROMISING_GAIN
BOOTSTRAP_N = BOOTSTRAP_RESAMPLES

#: Budget.
GPU_HARD_CAP_HOURS = 3.5
V1_HISTORICAL_GPU_HOURS = 0.5539  # recorded separately, not part of this cap

DESCRIPTOR_13 = (
    "MolLogP",
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

XGB_GRID = tuple(
    {"n_estimators": n, "max_depth": d, "min_child_weight": w}
    for n in (400, 800)
    for d in (3, 5)
    for w in (1, 5)
)
XGB_FIXED = {
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 4,
}
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
RIDGE_FIXED = {
    "solver": "lsqr",
    "fit_intercept": True,
    "tol": 1e-6,
    "max_iter": None,
    "positive": False,
}

#: Feature-group complexity order used only to break a <0.01 RMSE tie.
FEATURE_COMPLEXITY = {"P2": 0, "P1": 1, "P3": 2}
FAMILY_COMPLEXITY = {"ridge": 0, "xgboost": 1}


# ---------------------------------------------------------------------------
# v1 imports (read-only)
# ---------------------------------------------------------------------------

if str(V1_CODE) not in sys.path:
    sys.path.insert(0, str(V1_CODE))

import access_guard as v1_access_guard  # noqa: E402
import fit_labels as v1_fit_labels  # noqa: E402
import screen_common as v1c  # noqa: E402
import screen_data as v1d  # noqa: E402
import screen_models as v1m  # noqa: E402

prep_model = v1c.prep_model
prep_data = v1c.prep_data
feature_spec = v1c.feature_spec


# ---------------------------------------------------------------------------
# hashing / json / paths
# ---------------------------------------------------------------------------

sha256_file = v1c.sha256_file
sha256_text = v1c.sha256_text
read_json = v1c.read_json
write_json = v1c.write_json
stable_json = v1c.stable_json
track_rel = v1c.track_rel
utcnow = v1c.utcnow
git_commit = v1c.git_commit
git_dirty = v1c.git_dirty
ensure_dir = v1c.ensure_dir
log = v1c.log
file_record = v1c.file_record
verify_sha256 = v1c.verify_sha256
env_info = v1c.env_info


def source_dirty() -> bool:
    """True iff a tracked v2 source file is modified."""
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


def row_sha256_text(rows) -> str:
    return hashlib.sha256(",".join(str(int(r)) for r in rows).encode("utf-8")).hexdigest()


def row_sha256_bytes(rows) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# frozen v1 access
# ---------------------------------------------------------------------------


def load_frozen_spec():
    return v1c.load_frozen_spec()


def load_frozen_norm() -> dict[str, Any]:
    return v1c.load_frozen_norm()


def load_frozen_split() -> dict[str, Any]:
    return v1c.load_frozen_split()


def load_fit_cache() -> dict[str, Any]:
    return v1c.load_frozen_cache("fit")


def load_fit_labels() -> dict[str, Any]:
    return v1_fit_labels.load_fit_labels()


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_text(stable_json(cfg))


def validate_config(cfg: Mapping[str, Any]) -> None:
    def need(actual, expected, name):
        if actual != expected:
            raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")

    need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    need(cfg.get("round_id"), ROUND_ID, "round_id")
    need(tuple(int(x) for x in cfg["seeds"]), SEEDS, "seeds")
    need(tuple(int(x) for x in cfg["folds"]), FOLDS, "folds")
    need(int(cfg["training"]["batch_size"]), BATCH_SIZE, "training.batch_size")
    need(int(cfg["training"]["epochs"]), EPOCHS, "training.epochs")
    need(int(cfg["training"]["soup_last"]), SOUP_LAST, "training.soup_last")
    need(float(cfg["training"]["grad_clip"]), GRAD_CLIP, "training.grad_clip")
    need(float(cfg["training"]["min_lr"]), LR_MIN, "training.min_lr")
    need(int(cfg["training"]["shuffle_offset"]), SHUFFLE_OFFSET, "training.shuffle_offset")
    need(tuple(sorted(cfg["recipes"])), tuple(sorted(RECIPES)), "recipes")
    for name, row in RECIPES.items():
        need(float(cfg["recipes"][name]["peak_lr"]), row["peak_lr"], f"recipes.{name}.peak_lr")
        need(float(cfg["recipes"][name]["weight_decay"]), row["weight_decay"], f"recipes.{name}.weight_decay")
        need(float(cfg["recipes"][name]["dropout"]), row["dropout"], f"recipes.{name}.dropout")
    need(int(cfg["ssl"]["epochs"]), SSL_EPOCHS, "ssl.epochs")
    need(int(cfg["ssl"]["batch_size"]), SSL_BATCH_SIZE, "ssl.batch_size")
    need(float(cfg["ssl"]["lr"]), SSL_LR, "ssl.lr")
    need(float(cfg["ssl"]["min_lr"]), SSL_MIN_LR, "ssl.min_lr")
    need(float(cfg["ssl"]["weight_decay"]), SSL_WEIGHT_DECAY, "ssl.weight_decay")
    need(float(cfg["ssl"]["grad_clip"]), SSL_GRAD_CLIP, "ssl.grad_clip")
    need(float(cfg["ssl"]["mask_fraction"]), SSL_MASK_FRACTION, "ssl.mask_fraction")
    need(float(cfg["ssl"]["val_fraction"]), SSL_VAL_FRACTION, "ssl.val_fraction")
    need(int(cfg["ssl"]["val_mask_seed"]), SSL_VAL_MASK_SEED, "ssl.val_mask_seed")
    need(float(cfg["ssl"]["gate_min_nat"]), SSL_GATE_MIN_NAT, "ssl.gate_min_nat")
    need(float(cfg["decision"]["materiality_rmse"]), MATERIALITY_RMSE, "decision.materiality_rmse")
    need(float(cfg["decision"]["h1_promising_gain"]), H1_PROMISING_GAIN, "decision.h1_promising_gain")
    need(float(cfg["decision"]["h2_promising_gain"]), H2_PROMISING_GAIN, "decision.h2_promising_gain")
    need(float(cfg["decision"]["h3_promising_gain"]), H3_PROMISING_GAIN, "decision.h3_promising_gain")
    need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")
    need(tuple(cfg["descriptors"]["thirteen"]), DESCRIPTOR_13, "descriptors.thirteen")
    need(tuple(float(x) for x in cfg["baselines"]["ridge_alphas"]), RIDGE_ALPHAS, "baselines.ridge_alphas")
    need(tuple(tuple(int(v) for v in (r["n_estimators"], r["max_depth"], r["min_child_weight"])) for r in cfg["baselines"]["xgb_grid"]), tuple(tuple(int(v) for v in (r["n_estimators"], r["max_depth"], r["min_child_weight"])) for r in XGB_GRID), "baselines.xgb_grid")


# ---------------------------------------------------------------------------
# v1 immutability
# ---------------------------------------------------------------------------


def assert_v1_unchanged() -> dict[str, Any]:
    """Verify the v1 freeze, its manifest, and every manifest-referenced file."""
    out: dict[str, Any] = {"checks": [], "ok": True}
    for rel, expected in V1_IMMUTABLE.items():
        path = V1 / rel
        actual = sha256_file(path)
        ok = actual == expected
        out["checks"].append({"path": rel, "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            out["ok"] = False
    freeze = read_json(V1_ANALYSIS / "protocol_freeze.json")
    freeze_ok = freeze.get("protocol_id") == "lcrc-lipo-screen-v1"
    out["checks"].append({"path": "protocol_id", "expected": "lcrc-lipo-screen-v1", "actual": freeze.get("protocol_id"), "ok": freeze_ok})
    out["ok"] = out["ok"] and freeze_ok
    # Every v1 code/config file recorded in the v1 freeze must still hash to it.
    for rel, expected in sorted(freeze.get("code_hashes", {}).items()):
        path = V1 / rel.split("lcrc_lipo_screen_v1/", 1)[-1] if rel.startswith("tracks/") else V1 / rel
        if not path.exists():
            path = REPO / rel
        actual = sha256_file(path)
        ok = actual == expected
        out["checks"].append({"path": f"v1_code:{rel}", "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            out["ok"] = False
    # Every prediction file referenced by the v1 manifest.
    manifest = read_json(V1_ANALYSIS / "predictions/MANIFEST.json")
    for key, row in sorted(manifest.get("predictions", {}).items()):
        for field in ("path", "sidecar_path"):
            rel = row.get(field)
            if not rel:
                continue
            path = REPO / rel
            actual = sha256_file(path)
            expected = row.get("sha256") if field == "path" else row.get("sidecar_sha256")
            if expected is None:
                out["checks"].append({"path": f"v1_pred:{key}:{field}", "expected": None, "actual": actual, "ok": path.exists()})
                continue
            ok = actual == expected
            out["checks"].append({"path": f"v1_pred:{key}:{field}", "expected": expected, "actual": actual, "ok": ok})
            if not ok:
                out["ok"] = False
    out["n_checks"] = len(out["checks"])
    out["n_failed"] = sum(1 for c in out["checks"] if not c["ok"])
    return out


# ---------------------------------------------------------------------------
# v2 code hashes / freeze
# ---------------------------------------------------------------------------


def code_file_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(list((EXP_DIR / "code").glob("*.py")) + list((EXP_DIR / "tests").glob("*.py"))):
        out[track_rel(path)] = sha256_file(path)
    out[track_rel(CONFIG_PATH)] = sha256_file(CONFIG_PATH)
    return out


def build_protocol_freeze(cfg: Mapping[str, Any]) -> dict[str, Any]:
    v1_report = assert_v1_unchanged()
    if not v1_report["ok"]:
        raise PermissionError(f"v1 frozen artifacts changed: {[c for c in v1_report['checks'] if not c['ok']]}")
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
        "artifacts": {
            "v1_protocol_freeze": file_record(V1_ANALYSIS / "protocol_freeze.json"),
            "v1_prediction_manifest": file_record(V1_ANALYSIS / "predictions/MANIFEST.json"),
            "v1_monitor_metrics": file_record(V1_ANALYSIS / "monitor_metrics.json"),
            "folds_v2": file_record(FOLD_FILE),
            "prep_feature_spec": file_record(V1_PREP / "analysis/feature_spec_lipo.json"),
            "prep_norm": file_record(V1_PREP / "analysis/norm_lipo.json"),
            "phase1a_split": file_record(REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"),
            # The label artifact is gated: hash it from the Fit-only scanner
            # audit instead of opening it directly with the guard disarmed.
            "labels_fit_monitor": {
                "path": track_rel(REPO / "tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json"),
                "exists": True,
                "sha256": load_fit_labels()["labels_sha256"],
                "scope": "fit_values_constructed_monitor_values_skipped",
            },
            "norm_target_fit2352": file_record(NORM_TARGET_V2) if NORM_TARGET_V2.exists() else None,
            "cpu_baselines_v2": file_record(ANALYSIS / "cpu_baselines_v2.json") if (ANALYSIS / "cpu_baselines_v2.json").exists() else None,
            "lipo_raw": file_record(REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"),
            "row_manifest": file_record(REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/row_manifest.json"),
        },
        "v1_immutability": v1_report,
        "frozen_constants": {
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "arms": list(ARMS),
            "recipes": {k: dict(v) for k, v in RECIPES.items()},
            "recipe_order": list(RECIPE_ORDER),
            "recipe_improvement_min": RECIPE_IMPROVEMENT_MIN,
            "recipe_select_seeds": list(RECIPE_SELECT_SEEDS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "soup_last": SOUP_LAST,
            "grad_clip": GRAD_CLIP,
            "lr_min": LR_MIN,
            "shuffle_offset": SHUFFLE_OFFSET,
            "ssl": {
                "epochs": SSL_EPOCHS,
                "batch_size": SSL_BATCH_SIZE,
                "lr": SSL_LR,
                "min_lr": SSL_MIN_LR,
                "weight_decay": SSL_WEIGHT_DECAY,
                "grad_clip": SSL_GRAD_CLIP,
                "mask_fraction": SSL_MASK_FRACTION,
                "val_fraction": SSL_VAL_FRACTION,
                "val_mask_seed": SSL_VAL_MASK_SEED,
                "head_dims": list(SSL_HEAD_DIMS),
                "gate_min_nat": SSL_GATE_MIN_NAT,
            },
            "decision": {
                "materiality_rmse": MATERIALITY_RMSE,
                "h1_promising_gain": H1_PROMISING_GAIN,
                "h2_promising_gain": H2_PROMISING_GAIN,
                "h3_promising_gain": H3_PROMISING_GAIN,
                "cpu_dominance_gap": CPU_DOMINANCE_GAP,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "baselines": {
                "descriptor_13": list(DESCRIPTOR_13),
                "ridge_alphas": list(RIDGE_ALPHAS),
                "ridge_fixed": dict(RIDGE_FIXED),
                "xgb_grid": [dict(r) for r in XGB_GRID],
                "xgb_fixed": dict(XGB_FIXED),
                "feature_complexity": dict(FEATURE_COMPLEXITY),
                "family_complexity": dict(FAMILY_COMPLEXITY),
            },
        },
        "gpu_authorized": False,
        "gpu_hours_used": 0.0,
        "gpu_hard_cap_hours": GPU_HARD_CAP_HOURS,
    }


def verify_protocol_freeze(require_clean: bool = False) -> dict[str, Any]:
    if not PROTOCOL_FREEZE.exists():
        raise PermissionError(f"missing protocol freeze {track_rel(PROTOCOL_FREEZE)}")
    frozen = read_json(PROTOCOL_FREEZE)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("v2 protocol freeze has the wrong protocol id")
    cfg = load_config()
    if frozen["config_hash"] != config_hash(cfg):
        raise PermissionError("v2 config changed after the protocol freeze")
    current = code_file_hashes()
    if current != frozen["code_hashes"]:
        changed = sorted(k for k in set(current) | set(frozen["code_hashes"]) if current.get(k) != frozen["code_hashes"].get(k))
        raise PermissionError(f"v2 code changed after the protocol freeze: {changed}")
    if require_clean and git_dirty():
        raise PermissionError("formal run requires a clean committed worktree")
    v1 = assert_v1_unchanged()
    if not v1["ok"]:
        raise PermissionError("v1 frozen artifacts changed")
    return frozen


__all__ = [name for name in dir() if not name.startswith("_")]
