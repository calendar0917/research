"""zinc-full-y-budget-control-v1 — matched training-budget control for Full-Y.

Single question (protocol: ``configs/zinc_full_y_budget_control_v1.yaml``):
under the *same* Full-Y non-GNN skeleton, raw-y-only task supervision, hyper
parameters, 240 epochs and **last-5 soup** rule, how much does expanding the
supervised training set from the internal fit_inner 7200 rows to the full
official-train 10000 rows change the official-validation raw-y MAE?

Two arms (identical code path, identical evaluation):

* ``F7200``  — fit_inner 7200 rows; every fitted object (patch/ctx/topo/anchor
  standardisers, K-SVD dictionary ``D_fit``, common subspace ``U``) refit on
  fit_inner only (reuses the already audited ``prep_fit7200.npz``).
* ``F10000`` — official train 10000 rows; the **historical** seed-0 Full-Y
  raw-y checkpoint (``zinc_full_decomposition_valid_test_confirmation_v1``)
  is reused if and only if the read-only reuse audit passes
  (``REUSE_10K``), otherwise the protocol requires ``RETRAIN_BOTH``; an
  unresolved provenance problem is ``AUDIT_BLOCK`` (no silent substitution).

Both arms use the same primary checkpoint rule: the parameter mean of the
states at epochs 236,237,238,239,240 (``last-5``).  The historical F7200
monitor-top-5 soup is background only.

Discipline
----------
* seed 0 only (argparse enforces ``choices=[0]``); at most two formal training
  trajectories, and when the audit says ``REUSE_10K`` exactly **one** new run.
* task supervision = raw ``y`` L1 only; the auxiliary loss is the model's own
  label-free structural reconstruction with the frozen ``H1_LAMBDA``.
  ``g/ell/s/c/k`` never enter the driver; the historical Y-arm loss path is
  audited by AST/source inspection before anything is reused.
* official valid is read **once**, after both prediction files are written and
  hashed; official test is never touched anywhere in this driver.

Stages
------
``audit``    Phase-A historical-10k reuse audit + data/prep audit (CPU).
``replay``   frozen-checkpoint forward replay vs the historical train export.
``train``    the single new F7200 seed-0 run (240 epochs, last-5 soup).
``predict``  official-valid predictions for both arms (no labels read).
``score``    ONE formal scoring: read official-valid y once, metrics,
             paired bootstrap, subgroups, extreme-row analysis, decision.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    e2e_dictenv_clean_mechanism_v1 as cm,
    e2e_dictenv_scale_v1 as sc,
    zinc_e2e_dictenv_p1 as p1run,
    zinc_full_cycle_target_decomposition_v1 as zftd,
    zinc_joint_dictionary_decision_v1 as zjd,
    zinc_static_dictionary_pair as sdp,
)

PROTOCOL_ID = "zinc-full-y-budget-control-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "zinc_full_y_budget_control_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/zinc_full_y_budget_control_v1"

# ---- frozen training constants (identical for both arms) ------------------
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
H1_LAMBDA = float(cm.H1_LAMBDA)
FULL_PARAMETERS = 408_651
SCALE_SEED = 0

# ---- historical artifacts (read-only) -------------------------------------
KSVD_CONF = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1"
HIST_RUNNER = (
    REPO_ROOT
    / "tracks/ksvd/experiments/luyin16/zinc_full_decomposition_valid_test_confirmation_v1.py"
)
HIST_MANIFEST = KSVD_CONF / "frozen_eval_manifest.json"
HIST_PREP = KSVD_CONF / "all_train_prep.npz"
HIST_PREP_META = KSVD_CONF / "prep_meta.json"
HIST_Y_SOUP = KSVD_CONF / "Y_seed0_raw_soup_state.pt"
HIST_Y_LAST = KSVD_CONF / "Y_seed0_last_state.pt"
HIST_Y_INIT = KSVD_CONF / "Y_seed0_init_state.pt"
HIST_Y_JSON = KSVD_CONF / "Y_seed0.json"
HIST_Y_FIT_PRED = KSVD_CONF / "Y_seed0_fit_pred.npz"
HIST_Y_VALID_CSV = KSVD_CONF / "heldout_valid_predictions.csv"
DECOMP_PATH = (
    REPO_ROOT / "tracks/ksvd/results/zinc_full_cycle_target_decomposition_v1/target_decomposition.npz"
)

# ---- CSCL fit7200 artifacts ----------------------------------------------
PREP7200 = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/prep_fit7200.npz"
PREP7200_META = PREP7200.with_name("prep_fit7200_meta.json")
F7200_HIST_SOUP = (
    REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_soup_state.pt"
)
F7200_HIST_JSON = (
    REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/train_Full-Y-fit7200-seed0.json"
)
V3D_EVAL_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d_official_eval"
V3D_EVAL_F_PREDS = V3D_EVAL_DIR / "valid_F_preds.npz"
ENCODED_TRAIN = rpv2.ENCODED_TRAIN
ENCODED_VALID = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_valid.pt"

# ---- new artifacts from this round ---------------------------------------
F7200_SOUP = OUT / "F7200_last5_soup_state.pt"
F7200_META = OUT / "F7200_train_meta.json"
F7200_VALID_PREDS = OUT / "F7200_valid_preds.npz"
F10000_VALID_PREDS = OUT / "F10000_valid_preds.npz"
VALID_PREDS = OUT / "valid_predictions.npz"
FREEZE_MANIFEST = OUT / "freeze_manifest.json"

# ---- evaluation constants -------------------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
REPLAY_ATOL = 5.0e-5          # CPU float32 replay vs saved GPU export
REPLAY_MAE_ATOL = 1.0e-4
EXTREME_VALID_ID = "valid:0172"   # historically observed extreme row (index 172)
DECISION_BANDS = {"A": -0.010, "B": -0.003}  # delta <= A : clear; (A,B] : mid; > B : none

#: expected SHA256 of the frozen historical artifacts (from frozen_eval_manifest.json)
EXPECTED_ARTIFACT_SHA = {
    "all_train_prep.npz": "a7dbcc43fe3949c529db3b92dce3e3e0600a6c43c7277746552df5f05273bc03",
    "Y_seed0_raw_soup_state.pt": "7792c56045459a37c569424501352afb4f3fd8ebc06643522192710bfe85ecfe",
    "Y_seed0_last_state.pt": "427eb84d134929bfd817b4eb0296196ced505fef7c5b2623f23429dd66c6e77b",
    "Y_seed0_init_state.pt": "6736eecb8b764dd5f99cfe4ec037282b718cb256868b93c256737b45368700c6",
}
EXPECTED_RUNNER_SHA = "7a86a1358ab2298e728c2901ceb69d5db690f9288758089b33beaf2b063fc723"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=1), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def sha256_rows(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_split_indices() -> dict[str, np.ndarray]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    return cf.build_split_indices(smiles)


def load_blob(path: Path) -> dict[str, np.ndarray]:
    return {k: z for k, z in np.load(path, allow_pickle=False).items()}


def load_encoded_train() -> list[Any]:
    rows = list(torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False))
    p1run.attach_env(rows, "train")
    if len(rows) != 10000:
        raise RuntimeError(f"expected 10000 encoded train rows, got {len(rows)}")
    return rows


def build_model(blob: Mapping[str, Any], seed: int = 0) -> nn.Module:
    model = sc.build_scale_model(
        np.asarray(blob["D_fit"], np.float32),
        int(seed),
        zjd.fold_subspace(blob),
        sc.FULL,
        scale_seed=SCALE_SEED,
    )
    audit = sc.scale_parameter_audit(model)
    if int(audit["actual_parameters"]) != FULL_PARAMETERS or not audit["parameter_exact"]:
        raise RuntimeError(f"parameter audit failed: {audit}")
    return model


def torch_load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location=device, weights_only=True)


def read_csv_col(path: Path, column: str) -> np.ndarray:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([float(r[column]) for r in rows], dtype=np.float64)


def background_v3d_f_valid() -> dict[str, Any]:
    """Frozen background number (historical F7200 monitor-top-5 soup).

    The historical evaluation directory is git-ignored and may not exist on a
    compute checkout; the protocol-frozen constant is used as fallback.
    """
    path = V3D_EVAL_DIR / "valid_metrics.json"
    if path.exists():
        return {"value": float(read_json(path)["mae"]["F"]),
                "source": str(path.relative_to(REPO_ROOT)), "file_available": True}
    return {"value": 0.1329529655483202,
            "source": "protocol config background reference (cscl_v3d_official_eval absent on this checkout)",
            "file_available": False}


def _batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    if shuffle:
        order = torch.randperm(int(n), generator=generator).tolist()
    else:
        order = list(range(int(n)))
    return [order[s:s + batch_size] for s in range(0, int(n), batch_size)]


def _batch(data_list: Sequence[Any], indices: Sequence[int], targets: torch.Tensor, device: torch.device):
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    batch = p1.env_collate([data_list[i] for i in indices])
    batch.y = targets[torch.as_tensor(list(indices), dtype=torch.long)].view(-1).clone()
    return batch.to(device)


# ---------------------------------------------------------------------------
# static source audits (no data needed)
# ---------------------------------------------------------------------------


def audit_runner_source() -> dict[str, Any]:
    """Verify the historical Y branch cannot receive component supervision."""
    text = HIST_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_full_arm"
    )
    src = ast.unparse(fn)
    # decomp reads inside the post-training export call are provenance exports,
    # never supervision; every other read must be the y/g target selection.
    export_reads: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "savez_compressed"
        ):
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Subscript)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "decomp"
                ):
                    export_reads.add(ast.unparse(inner).replace("'", '"'))
    all_reads = {
        ast.unparse(node).replace("'", '"')
        for node in ast.walk(fn)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "decomp"
    }
    training_reads = sorted(all_reads - export_reads)
    checks = {
        "target_select_raw_y_for_Y": (
            "target_fit_np = y_all if arm == 'Y' else g_all" in src
            or 'target_fit_np = y_all if arm == "Y" else g_all' in src
        ),
        "loss_is_l1_batch_y_plus_lambda_rec": (
            "F.l1_loss(prediction.view(-1), batch.y.view(-1))" in src
            and "loss = task + float(cm.H1_LAMBDA) * rec" in src
        ),
        "rec_is_structural_only": ("model.reconstruction_loss(aux['phi'], aux['coord'])" in src),
        "mask_is_C6": ("model(batch, mask=cm.C6_MASK, return_aux=True)" in src),
        "soup_epochs_last_5": (
            "soup_epochs = set(range(max(1, int(epochs) - SOUP_LAST + 1), int(epochs) + 1))" in src
        ),
        "train_gen_seed_offset_101": (
            "torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)" in src
        ),
        "decomp_reads_in_train_fn": sorted(all_reads),
        "decomp_reads_in_export_only": sorted(export_reads),
        "training_decomp_reads": training_reads,
        "training_decomp_reads_are_y_and_g_targets_only": training_reads == ['decomp["g"]', 'decomp["y"]'],
    }

    # reconstruction_loss depends on phi/coord only
    cssd_text = (
        REPO_ROOT / "tracks/ksvd/experiments/luyin16/e2e_dictenv_common_subspace_dictionary_v1.py"
    ).read_text(encoding="utf-8")
    cssd_tree = ast.parse(cssd_text)
    rec = next(
        node for node in ast.walk(cssd_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reconstruction_loss"
    )
    rec_src = ast.unparse(rec)
    used_names = sorted({n.id for n in ast.walk(rec) if isinstance(n, ast.Name)})
    checks["reconstruction_loss_names"] = used_names
    checks["reconstruction_loss_no_label_names"] = not [
        n for n in used_names if n in ("y", "targets", "label", "labels", "batch")
    ]

    # current inference path uses the same mask
    checks["predict_raw_uses_C6"] = (
        "model(batch, mask=cm.C6_MASK)" in (Path(rpv2.__file__).read_text(encoding="utf-8"))
    )

    checks["runner_source_sha256"] = file_sha256(HIST_RUNNER)
    checks["runner_source_matches_frozen_manifest"] = bool(checks["runner_source_sha256"] == EXPECTED_RUNNER_SHA)
    checks["frozen_manifest_file_sha256"] = file_sha256(HIST_MANIFEST)
    checks["ok"] = bool(
        checks["target_select_raw_y_for_Y"]
        and checks["loss_is_l1_batch_y_plus_lambda_rec"]
        and checks["rec_is_structural_only"]
        and checks["mask_is_C6"]
        and checks["soup_epochs_last_5"]
        and checks["train_gen_seed_offset_101"]
        and checks["training_decomp_reads_are_y_and_g_targets_only"]
        and checks["reconstruction_loss_no_label_names"]
        and checks["predict_raw_uses_C6"]
        and checks["runner_source_matches_frozen_manifest"]
    )
    return checks


def audit_artifact_hashes() -> dict[str, Any]:
    manifest = read_json(HIST_MANIFEST)
    files = {}
    for name, expected in EXPECTED_ARTIFACT_SHA.items():
        actual = file_sha256(KSVD_CONF / name)
        files[name] = {"sha256": actual, "expected_sha256": expected, "match": bool(actual == expected)}
    prep_meta = read_json(HIST_PREP_META)
    out = {
        "frozen_manifest_sha256": file_sha256(HIST_MANIFEST),
        "files": files,
        "prep_meta_sha256": file_sha256(HIST_PREP_META),
        "prep_meta_prep_sha256": prep_meta["prep_file_sha256"],
        "prep_meta_matches_file": bool(prep_meta["prep_file_sha256"] == files["all_train_prep.npz"]["sha256"]),
        "y_valid_csv_sha256": file_sha256(HIST_Y_VALID_CSV),
        "y_fit_pred_sha256": file_sha256(HIST_Y_FIT_PRED),
        "y_json_sha256": file_sha256(HIST_Y_JSON),
        "manifest_artifact_count": len(manifest["artifacts"]),
    }
    out["ok"] = bool(all(v["match"] for v in files.values()) and out["prep_meta_matches_file"])
    return out


# ---------------------------------------------------------------------------
# stage: audit (Phase A reuse decision + data/prep audit)
# ---------------------------------------------------------------------------


def stage_audit(device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    source = audit_runner_source()
    hashes = audit_artifact_hashes()

    y_json = read_json(HIST_Y_JSON)
    recipe_keys = (
        "arm", "seed", "scale_seed", "epochs", "steps_done", "lr", "weight_decay",
        "grad_clip", "batch_size", "h1_lambda", "soup_members", "soup_epochs",
        "train_generator_seed", "fit_target_is_y", "official_valid_loaded",
        "official_test_loaded", "fit_raw_mae_y", "fit_cal_mae_y",
    )
    recipe = {k: y_json[k] for k in recipe_keys}
    recipe_expected = {
        "arm": "Y", "seed": 0, "scale_seed": 0, "epochs": 240, "steps_done": 18960,
        "lr": 1e-3, "weight_decay": 1e-5, "grad_clip": 5.0, "batch_size": 128,
        "h1_lambda": H1_LAMBDA, "soup_members": [236, 237, 238, 239, 240],
        "soup_epochs": [236, 237, 238, 239, 240], "train_generator_seed": 101,
        "fit_target_is_y": True, "official_valid_loaded": False, "official_test_loaded": False,
    }
    recipe_match = {k: (recipe[k] == v) for k, v in recipe_expected.items()}

    # ---- split + data provenance -----------------------------------------
    idx = load_split_indices()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    fit_rows = np.asarray(idx["fit_inner"], dtype=np.int64)
    mon_rows = np.asarray(idx["monitor"], dtype=np.int64)
    dev_rows = np.asarray(idx["dev"], dtype=np.int64)
    all_rows = np.arange(10000, dtype=np.int64)
    subset = bool(
        np.isin(fit_rows, all_rows).all()
        and np.isin(mon_rows, all_rows).all()
        and np.isin(dev_rows, all_rows).all()
    )
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(s, []).append(i)
    fit_set = set(fit_rows.tolist())
    group_crossing = sum(
        1 for rows in groups.values() if len({0 if r in fit_set else 1 for r in rows}) > 1
    )
    data_checks = {
        "n_encoded_train": 10000,
        "n_split_fit_inner": int(len(fit_rows)),
        "n_split_monitor": int(len(mon_rows)),
        "n_split_dev": int(len(dev_rows)),
        "fit_inner_is_subset_of_train_rows": subset,
        "fit_inner_7200_unique": bool(len(fit_rows) == 7200 and len(fit_set) == 7200),
        "split_sides_disjoint": bool(
            len(fit_set & set(mon_rows.tolist())) == 0
            and len(fit_set & set(dev_rows.tolist())) == 0
            and len(set(mon_rows.tolist()) & set(dev_rows.tolist())) == 0
        ),
        "smiles_groups_crossing_fit_dev": int(group_crossing),
        "smiles_groups_crossing_zero": bool(group_crossing == 0),
    }

    # ---- prep provenance --------------------------------------------------
    prep7200 = load_blob(PREP7200)
    prep7200_meta = read_json(PREP7200_META)
    blob10 = load_blob(HIST_PREP)
    prep_checks = {
        "prep7200_file_sha256": file_sha256(PREP7200),
        "prep7200_meta_D_fit_sha256": prep7200_meta["sha256"]["D_fit"],
        "prep7200_fit_idx_equals_fit_inner": bool(
            sha256_rows(np.asarray(prep7200["fit_idx"], np.int64)) == sha256_rows(fit_rows)
        ),
        "prep7200_monitor_idx_equals_monitor": bool(
            sha256_rows(np.asarray(prep7200["monitor_idx"], np.int64)) == sha256_rows(mon_rows)
        ),
        "prep7200_dev_idx_equals_dev": bool(
            sha256_rows(np.asarray(prep7200["dev_idx"], np.int64)) == sha256_rows(dev_rows)
        ),
        "prep7200_n_fit": int(len(prep7200["fit_idx"])),
        "all_train_prep_fit_idx_is_all_10000": bool(
            np.array_equal(np.asarray(blob10["fit_idx"], np.int64), all_rows)
        ),
        "all_train_prep_dev_idx_is_all_10000_placeholder": bool(
            np.array_equal(np.asarray(blob10["dev_idx"], np.int64), all_rows)
        ),
        "D_fit_7200_sha256": sha256_array(np.asarray(prep7200["D_fit"], np.float32)),
        "D_fit_10000_sha256": sha256_array(np.asarray(blob10["D_fit"], np.float32)),
        "D_fit_differs": bool(
            sha256_array(np.asarray(prep7200["D_fit"], np.float32))
            != sha256_array(np.asarray(blob10["D_fit"], np.float32))
        ),
        "U_7200_sha256": sha256_array(np.asarray(prep7200["U_components"], np.float32)),
        "U_10000_sha256": sha256_array(np.asarray(blob10["U_components"], np.float32)),
        "U_differs": bool(
            sha256_array(np.asarray(prep7200["U_components"], np.float32))
            != sha256_array(np.asarray(blob10["U_components"], np.float32))
        ),
    }

    # ---- labels and row alignment (official train only) -------------------
    train = load_encoded_train()
    y_encoded = np.array([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    y_fit_pred = np.load(HIST_Y_FIT_PRED, allow_pickle=False)
    y_hist = np.asarray(y_fit_pred["y"], dtype=np.float64)
    gid = np.asarray(y_fit_pred["gid"])
    decomp = np.load(DECOMP_PATH, allow_pickle=False)
    y_decomp = np.asarray(decomp["y"], dtype=np.float64)
    g_decomp = np.asarray(decomp["g"], dtype=np.float64)
    c_decomp = np.asarray(decomp["c"], dtype=np.float64)
    data_checks.update({
        "max_abs_y_encoded_minus_hist_export": float(np.max(np.abs(y_encoded - y_hist))),
        "max_abs_y_encoded_minus_decomp": float(np.max(np.abs(y_encoded - y_decomp))),
        "max_abs_g_plus_c_minus_y": float(np.max(np.abs(g_decomp + c_decomp - y_decomp))),
        "gid_n_unique": int(len(np.unique(gid))),
        "gid_role": "canonical group id, not row index (historical export convention)",
    })
    data_checks["ok"] = bool(
        data_checks["fit_inner_is_subset_of_train_rows"]
        and data_checks["fit_inner_7200_unique"]
        and data_checks["split_sides_disjoint"]
        and data_checks["smiles_groups_crossing_zero"]
        and data_checks["max_abs_y_encoded_minus_hist_export"] == 0.0
        and data_checks["max_abs_y_encoded_minus_decomp"] == 0.0
        and data_checks["max_abs_g_plus_c_minus_y"] < 1e-9
    )
    del decomp, train  # free memory before the model build

    # ---- model identity / initialisation ----------------------------------
    seed_everything(0)
    m10 = build_model(blob10, 0)
    init10 = {k: v.detach().cpu().clone() for k, v in m10.state_dict().items()}
    seed_everything(0)
    m72 = build_model(prep7200, 0)
    init72 = {k: v.detach().cpu().clone() for k, v in m72.state_dict().items()}
    differing = [key for key in init10 if not torch.equal(init10[key], init72[key])]
    soup_state = torch_load_state(HIST_Y_SOUP, torch.device("cpu"))
    last_state = torch_load_state(HIST_Y_LAST, torch.device("cpu"))
    init_state = torch_load_state(HIST_Y_INIT, torch.device("cpu"))
    model_checks = {
        "params_F10000": int(sum(p.numel() for p in m10.parameters())),
        "params_F7200": int(sum(p.numel() for p in m72.parameters())),
        "state_dict_keys_equal": bool(set(init10) == set(init72)),
        "init_differing_keys": differing,
        "init_diff_only_data_buffers": bool(set(differing) <= {"D", "U", "common_rms"}),
        "soup_state_hash": state_hash(soup_state),
        "soup_state_hash_recorded": y_json["soup_state_sha256"],
        "soup_hash_matches_recorded": bool(state_hash(soup_state) == y_json["soup_state_sha256"]),
        "last_state_hash": state_hash(last_state),
        "last_state_hash_recorded": y_json["last_state_sha256"],
        "last_hash_matches_recorded": bool(state_hash(last_state) == y_json["last_state_sha256"]),
        "init_state_hash": state_hash(init_state),
        "init_hash_recorded": y_json["init_state_sha256"],
        "init_hash_matches_recorded": bool(state_hash(init_state) == y_json["init_state_sha256"]),
        "soup_differs_from_last": bool(state_hash(soup_state) != state_hash(last_state)),
        "strict_load_ok": False,
    }
    try:
        m10.load_state_dict(soup_state, strict=True)
        model_checks["strict_load_ok"] = True
    except RuntimeError as exc:  # pragma: no cover
        model_checks["strict_load_error"] = str(exc)
    forbidden_tokens = ("GNN", "MessagePassing", "Transformer", "Attention", "GAT", "GCN", "SAGE")
    bad_modules = sorted({
        type(module).__name__
        for module in m10.modules()
        if any(token in type(module).__name__ for token in forbidden_tokens)
    })
    model_checks["forbidden_module_classes"] = bad_modules
    model_checks["non_gnn_scan_ok"] = len(bad_modules) == 0
    model_checks["trainable_parameters"] = int(sum(p.numel() for p in m10.parameters() if p.requires_grad))

    steps = {
        "F7200_batches_per_epoch": int(np.ceil(7200 / BATCH_SIZE)),
        "F10000_batches_per_epoch": int(np.ceil(10000 / BATCH_SIZE)),
        "F7200_steps_240": int(np.ceil(7200 / BATCH_SIZE)) * EPOCHS,
        "F10000_steps_240": int(np.ceil(10000 / BATCH_SIZE)) * EPOCHS,
        "historical_steps_done": y_json["steps_done"],
        "row_increase_fraction": 2800 / 7200,
    }
    steps["F10000_steps_match_historical"] = bool(steps["F10000_steps_240"] == y_json["steps_done"])
    steps["step_increase_fraction"] = (
        (steps["F10000_steps_240"] - steps["F7200_steps_240"]) / steps["F7200_steps_240"]
    )

    background = {
        "historical_F7200_monitor_top5": {
            "soup_state_sha256": file_sha256(F7200_HIST_SOUP),
            "prep_fit7200_sha256": file_sha256(PREP7200),
            "recorded_dev_mae": read_json(F7200_HIST_JSON)["primary_soup"]["dev_mae"],
            "recorded_valid_raw_mae": background_v3d_f_valid(),
            "role": "background reference only (monitor top-5 soup, different soup rule)",
        },
        "historical_F10000_seed0": {
            "valid_raw_mae": 0.11861754740116885,
            "valid_cal_mae": 0.11839167641935637,
            "test_raw_mae": 0.09458897127117961,
            "fit_raw_mae_y": y_json["fit_raw_mae_y"],
            "source": "notes/strong_prediction_route_audit_v1.md + heldout CSVs recomputed read-only",
        },
    }

    audit = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_audit": source,
        "artifact_hashes": hashes,
        "historical_recipe": recipe,
        "historical_recipe_match": recipe_match,
        "historical_recipe_all_match": bool(all(recipe_match.values())),
        "data_audit": data_checks,
        "prep_audit": prep_checks,
        "model_audit": model_checks,
        "budget": steps,
        "historical_reference": background,
        "seconds": float(time.perf_counter() - started),
    }
    hard_ok = bool(
        source["ok"]
        and hashes["ok"]
        and audit["historical_recipe_all_match"]
        and data_checks["ok"]
        and model_checks["strict_load_ok"]
        and model_checks["soup_hash_matches_recorded"]
        and model_checks["last_hash_matches_recorded"]
        and model_checks["init_hash_matches_recorded"]
        and model_checks["init_diff_only_data_buffers"]
        and model_checks["non_gnn_scan_ok"]
        and prep_checks["prep7200_fit_idx_equals_fit_inner"]
        and steps["F10000_steps_match_historical"]
    )
    audit["hard_checks_pass"] = hard_ok
    audit["decision"] = "REUSE_10K" if hard_ok else "PENDING_REPLAY_OR_INVESTIGATION"
    write_json(OUT / "historical_reuse_audit.json", audit)
    write_json(OUT / "data_prep_audit.json", {
        "protocol_id": PROTOCOL_ID,
        "data_audit": data_checks,
        "prep_audit": prep_checks,
        "budget": steps,
        "decision_stage": audit["decision"],
    })
    log(f"[audit] hard_checks_pass={hard_ok} decision={audit['decision']} "
        f"init_differing_keys={differing} seconds={audit['seconds']:.1f}")
    return audit


# ---------------------------------------------------------------------------
# stage: replay (frozen 10k checkpoint forward vs historical export)
# ---------------------------------------------------------------------------


def stage_replay(device: torch.device, limit: int | None) -> dict[str, Any]:
    started = time.perf_counter()
    y_json = read_json(HIST_Y_JSON)
    blob10 = load_blob(HIST_PREP)
    train = load_encoded_train()
    n = len(train) if limit is None else min(int(limit), len(train))
    y_all = np.array([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    zftd.apply_prep_train_only(train, blob10)
    data = train[:n]

    model = build_model(blob10, 0)
    state = torch_load_state(HIST_Y_SOUP, torch.device(device))
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.eval()
    pred = rpv2.predict_raw(model, data, device)  # y never passed (zeros)

    stored = np.load(HIST_Y_FIT_PRED, allow_pickle=False)
    pred_hist = np.asarray(stored["fit_raw"], dtype=np.float64)[:n]
    y_hist = np.asarray(stored["y"], dtype=np.float64)[:n]
    diff = np.abs(pred - pred_hist)
    mae_replay = float(np.mean(np.abs(pred - y_all[:n])))
    mae_hist = float(np.mean(np.abs(pred_hist - y_hist)))
    out = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(device),
        "n_rows": n,
        "subset": "all official-train rows" if limit is None else f"first {n} official-train rows",
        "checkpoint": {"path": str(HIST_Y_SOUP.relative_to(REPO_ROOT)), "sha256": file_sha256(HIST_Y_SOUP)},
        "checkpoint_state_hash": state_hash(state),
        "checkpoint_state_hash_recorded": y_json["soup_state_sha256"],
        "soup_members_recorded": y_json["soup_members"],
        "soup_rule_verified_from_json": bool(y_json["soup_members"] == [236, 237, 238, 239, 240]),
        "max_abs_pred_diff": float(diff.max()),
        "mean_abs_pred_diff": float(diff.mean()),
        "replay_mae": mae_replay,
        "historical_export_mae": mae_hist,
        "mae_abs_diff": float(abs(mae_replay - mae_hist)),
        "historical_recorded_fit_raw_mae_y": y_json["fit_raw_mae_y"],
        "replay_atol": REPLAY_ATOL,
        "mae_atol": REPLAY_MAE_ATOL,
        "seconds": float(time.perf_counter() - started),
    }
    out["ok"] = bool(out["max_abs_pred_diff"] <= REPLAY_ATOL and out["mae_abs_diff"] <= REPLAY_MAE_ATOL)
    write_json(OUT / "protocol_replay.json", out)
    # tighten the reuse decision with the replay result
    audit = read_json(OUT / "historical_reuse_audit.json")
    if audit.get("hard_checks_pass") and out["ok"]:
        audit["decision"] = "REUSE_10K"
    elif audit.get("hard_checks_pass") and not out["ok"]:
        audit["decision"] = "AUDIT_BLOCK_REPLAY_FAILED"
    audit["replay_gate"] = {"ok": out["ok"], "max_abs_pred_diff": out["max_abs_pred_diff"]}
    write_json(OUT / "historical_reuse_audit.json", audit)
    log(f"[replay] n={n} max|dpred|={out['max_abs_pred_diff']:.3e} "
        f"mae_diff={out['mae_abs_diff']:.3e} ok={out['ok']} {out['seconds']:.1f}s")
    return out


# ---------------------------------------------------------------------------
# stage: train (the single new F7200 seed-0 trajectory)
# ---------------------------------------------------------------------------


def train_one_arm(
    *,
    arm: str,
    blob: Mapping[str, Any],
    train_rows: Sequence[Any],
    fit_rows: np.ndarray,
    y_all: np.ndarray,
    device: torch.device,
    epochs: int,
    log_every: int = 20,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    fit_data = [train_rows[i] for i in fit_rows]
    y_fit = torch.as_tensor(y_all[fit_rows], dtype=torch.float32)

    seed_everything(0)
    model = build_model(blob, 0).to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)

    soup_epochs = set(range(max(1, epochs - SOUP_LAST + 1), epochs + 1))
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    steps_done = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps = 0.0, 0.0, 0, 0
        for indices in _batches(len(fit_data), BATCH_SIZE, train_gen, True):
            batch = _batch(fit_data, indices, y_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_seconds.append(float(time.perf_counter() - t0))
        curve.append({
            "epoch": int(epoch),
            "train_task_mae": float(task_sum / max(n_mol, 1)),
            "train_rec": float(rec_sum / max(n_steps, 1)),
            "seconds": epoch_seconds[-1],
        })
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            log(f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"rec={curve[-1]['train_rec']:.4g} {epoch_seconds[-1]:.1f}s")

    members = sorted(soup)
    expected_members = list(range(max(1, epochs - SOUP_LAST + 1), epochs + 1))
    if members != expected_members:
        raise RuntimeError(f"{arm}: soup members {members} != last {SOUP_LAST} epochs {expected_members}")
    soup_state = {
        k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]
    }
    meta = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "git_commit": git_commit(),
        "seed": 0,
        "scale_seed": SCALE_SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "batches_per_epoch": int(np.ceil(len(fit_data) / BATCH_SIZE)),
        "soup_members": members,
        "soup_rule": "mean(state_dict at epochs 236..240)",
        "n_train_rows": int(len(fit_data)),
        "train_rows_sha256": sha256_rows(np.asarray(fit_rows, dtype=np.int64)),
        "hyper": {
            "optimizer": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "h1_lambda": H1_LAMBDA,
            "train_shuffle_seed": 0 + TRAIN_SHUFFLE_OFFSET, "amp": False, "scheduler": None,
            "task_supervision": "raw y L1 only",
            "auxiliary": "label-free structural reconstruction (disclosed)",
        },
        "init_state_sha256": state_hash(init_state),
        "soup_state_sha256": state_hash(soup_state),
        "curve": curve,
        "wall_seconds": float(time.perf_counter() - started),
        "seconds_per_epoch_mean": float(np.mean(epoch_seconds)),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    return soup_state, meta


def stage_train(device: torch.device, epochs: int, smoke: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "historical_reuse_audit.json"
    if not smoke:
        if not audit_path.exists():
            raise SystemExit("run `audit` (+ `replay`) before training")
        decision = read_json(audit_path)["decision"]
        if decision == "AUDIT_BLOCK":
            raise SystemExit("reuse audit returned AUDIT_BLOCK; refusing to train")
        if decision not in ("REUSE_10K", "RETRAIN_BOTH"):
            raise SystemExit(f"unexpected audit decision {decision!r}; refusing to train")
    idx = load_split_indices()
    fit_rows = np.asarray(idx["fit_inner"], dtype=np.int64)
    train = load_encoded_train()
    y_all = np.array([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    prep7200 = load_blob(PREP7200)
    blob10 = load_blob(HIST_PREP)

    rpv2.apply_prep_fit7200(train, prep7200, blob10)
    use_rows = fit_rows[:256] if smoke else fit_rows
    use_epochs = min(int(epochs), 3) if smoke else int(epochs)
    soup_state, meta = train_one_arm(
        arm="F7200" if not smoke else "F7200-smoke",
        blob=prep7200, train_rows=train, fit_rows=use_rows,
        y_all=y_all, device=device, epochs=use_epochs, log_every=1 if smoke else 20,
    )
    if smoke:
        write_json(OUT / "F7200_smoke_meta.json", meta)
        log("[train] smoke finished (no checkpoint written)")
        return meta

    torch.save(soup_state, F7200_SOUP)
    meta["soup_state_file"] = str(F7200_SOUP.relative_to(REPO_ROOT))
    meta["soup_state_file_sha256"] = file_sha256(F7200_SOUP)
    meta["prep_path"] = str(PREP7200.relative_to(REPO_ROOT))
    meta["prep_sha256"] = file_sha256(PREP7200)
    meta["prep_provenance"] = {
        "fit_idx_sha256_rows": sha256_rows(np.asarray(prep7200["fit_idx"], np.int64)),
        "D_fit_sha256": sha256_array(np.asarray(prep7200["D_fit"], np.float32)),
        "U_components_sha256": sha256_array(np.asarray(prep7200["U_components"], np.float32)),
    }

    # auxiliary (post-training, never used for selection): internal fit/monitor/dev
    y_t = torch.as_tensor(y_all, dtype=torch.float32)
    model = build_model(prep7200, 0).to(device)
    model.load_state_dict({k: v.to(device) for k, v in soup_state.items()}, strict=True)
    model.eval()
    aux = {}
    for name, rows in (
        ("fit", fit_rows),
        ("monitor", np.asarray(idx["monitor"])),
        ("dev", np.asarray(idx["dev"])),
    ):
        pred = rpv2.predict_raw(model, [train[i] for i in rows], device)
        aux[f"{name}_mae"] = float(np.mean(np.abs(pred - y_all[rows])))
    meta["auxiliary_internal_mae"] = aux
    fit_pred = rpv2.predict_raw(model, [train[i] for i in fit_rows], device)
    meta["fit_only_calibration_b"] = float(np.median(y_all[fit_rows] - fit_pred))
    np.savez_compressed(
        OUT / "F7200_fit_pred.npz",
        fit_raw=fit_pred.astype(np.float32),
        y=y_all[fit_rows].astype(np.float32),
        row=fit_rows.astype(np.int64),
    )
    # historical F7200 trajectory comparison (background; training must be identical)
    hist = read_json(F7200_HIST_JSON)
    hist_curve = {int(c["epoch"]): float(c["train_task_mae"]) for c in hist["curve"]}
    meta["historical_F7200_curve_comparison"] = {
        "max_abs_train_mae_diff": max(
            abs(c["train_task_mae"] - hist_curve[int(c["epoch"])])
            for c in meta["curve"] if int(c["epoch"]) in hist_curve
        ),
        "historical_soup_members": hist["soup_members"],
        "note": "same code path + same prep; only the soup rule differs (last-5 vs monitor top-5)",
    }
    write_json(F7200_META, meta)
    log(f"[train] F7200 done: fit={aux['fit_mae']:.5f} monitor={aux['monitor_mae']:.5f} "
        f"dev={aux['dev_mae']:.5f} soup_sha={meta['soup_state_sha256'][:12]} "
        f"curve_gap={meta['historical_F7200_curve_comparison']['max_abs_train_mae_diff']:.3e}")

    # F10000 metadata follows from the audited historical artifact
    y_json = read_json(HIST_Y_JSON)
    f10000_meta = {
        "protocol_id": PROTOCOL_ID,
        "arm": "F10000",
        "reused_historical_checkpoint": True,
        "historical_round": "zinc_full_decomposition_valid_test_confirmation_v1",
        "checkpoint_path": str(HIST_Y_SOUP.relative_to(REPO_ROOT)),
        "checkpoint_file_sha256": file_sha256(HIST_Y_SOUP),
        "checkpoint_state_hash": y_json["soup_state_sha256"],
        "n_train_rows": 10000,
        "epochs": y_json["epochs"],
        "steps_done": y_json["steps_done"],
        "soup_members": y_json["soup_members"],
        "soup_rule": "mean(state_dict at epochs 236..240)",
        "hyper": {
            "optimizer": "Adam", "lr": y_json["lr"], "weight_decay": y_json["weight_decay"],
            "grad_clip": y_json["grad_clip"], "batch_size": y_json["batch_size"],
            "h1_lambda": y_json["h1_lambda"], "train_shuffle_seed": y_json["train_generator_seed"],
            "amp": False, "scheduler": None, "task_supervision": "raw y L1 only",
        },
        "fit_raw_mae_y": y_json["fit_raw_mae_y"],
        "prep_path": str(HIST_PREP.relative_to(REPO_ROOT)),
        "prep_sha256": file_sha256(HIST_PREP),
        "audit_receipt": "historical_reuse_audit.json (hard_checks_pass + replay gate)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "F10000_train_or_reuse_meta.json", f10000_meta)
    return meta


# ---------------------------------------------------------------------------
# stage: predict (official valid, labels never read)
# ---------------------------------------------------------------------------


def _load_valid_rows() -> tuple[list[Any], Any, dict[str, Any]]:
    """Committed encoded-valid cache + env, order == official ZINC val split."""
    from torch_geometric.datasets import ZINC

    _train, rows, _audit = sdp.load_encoded(valid_subset=None)
    p1run.attach_env(rows, "valid")
    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    if not (len(rows) == len(ds) == 1000):
        raise RuntimeError(f"valid length mismatch: {len(rows)} vs {len(ds)}")
    n_atom_mismatch = n_type_mismatch = 0
    for i in range(1000):
        if int(rows[i].dict_phi.shape[0]) != int(ds[i].x.shape[0]):
            n_atom_mismatch += 1
        elif not torch.equal(rows[i].dict_atom.long().reshape(-1), ds[i].x.long().reshape(-1)):
            n_type_mismatch += 1
    if n_atom_mismatch or n_type_mismatch:
        raise RuntimeError(
            f"valid row alignment failed: atom={n_atom_mismatch} type={n_type_mismatch}"
        )
    return rows, ds, {"atom_count_mismatches": n_atom_mismatch, "atom_type_mismatches": n_type_mismatch}


def stage_predict(device: torch.device, smoke: int | None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    n_mol = int(smoke) if smoke else 1000
    suffix = "_smoke" if smoke else ""
    prep7200 = load_blob(PREP7200)
    blob10 = load_blob(HIST_PREP)

    rows72, _ds, align = _load_valid_rows()
    rows10, _ds10, align10 = _load_valid_rows()
    if smoke:
        rows72, rows10 = rows72[:n_mol], rows10[:n_mol]
    if align != align10:
        raise RuntimeError("valid alignment audit mismatch between loads")

    rpv2.apply_prep_fit7200(rows72, prep7200, blob10)
    zftd.apply_prep_train_only(rows10, blob10)

    model72 = build_model(prep7200, 0).to(device)
    state72 = torch_load_state(F7200_SOUP, device)
    model72.load_state_dict({k: v.to(device) for k, v in state72.items()}, strict=True)
    model72.eval()
    pred72 = rpv2.predict_raw(model72, rows72, device)

    model10 = build_model(blob10, 0).to(device)
    state10 = torch_load_state(HIST_Y_SOUP, device)
    model10.load_state_dict({k: v.to(device) for k, v in state10.items()}, strict=True)
    model10.eval()
    pred10 = rpv2.predict_raw(model10, rows10, device)

    if smoke:
        np.savez_compressed(OUT / f"valid_predictions{suffix}.npz",
                            pred_F7200=pred72.astype(np.float32),
                            pred_F10000=pred10.astype(np.float32),
                            row=np.arange(n_mol, dtype=np.int64))
        log("[predict] smoke predictions written")
        return {"smoke": True, "n": n_mol}

    np.savez_compressed(F7200_VALID_PREDS, pred=pred72.astype(np.float32))
    np.savez_compressed(F10000_VALID_PREDS, pred=pred10.astype(np.float32))
    np.savez_compressed(VALID_PREDS,
                        pred_F7200=pred72.astype(np.float32),
                        pred_F10000=pred10.astype(np.float32),
                        row=np.arange(n_mol, dtype=np.int64))

    preds_hist = read_csv_col(HIST_Y_VALID_CSV, "Y_raw_0")
    diff_hist = float(np.max(np.abs(pred10 - preds_hist)))
    diff_v3d = None
    if V3D_EVAL_F_PREDS.exists():
        v3d = np.load(V3D_EVAL_F_PREDS, allow_pickle=False)
        pred_v3d = np.asarray(v3d["pred"], dtype=np.float64)
        diff_v3d = float(np.max(np.abs(pred72 - pred_v3d))) if pred_v3d.shape[0] == 1000 else None
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_molecules": n_mol,
        "device": str(device),
        "official_test_loaded": False,
        "y_in_file": False,
        "valid_alignment": align,
        "checkpoints": {
            "F7200": {"path": str(F7200_SOUP.relative_to(REPO_ROOT)), "sha256": file_sha256(F7200_SOUP),
                      "state_sha256": state_hash(state72)},
            "F10000": {"path": str(HIST_Y_SOUP.relative_to(REPO_ROOT)), "sha256": file_sha256(HIST_Y_SOUP),
                       "state_sha256": state_hash(state10)},
        },
        "preps": {
            "F7200": {"path": str(PREP7200.relative_to(REPO_ROOT)), "sha256": file_sha256(PREP7200)},
            "F10000": {"path": str(HIST_PREP.relative_to(REPO_ROOT)), "sha256": file_sha256(HIST_PREP)},
        },
        "data_sources": {
            "encoded_valid": str(ENCODED_VALID.relative_to(REPO_ROOT)),
            "encoded_valid_sha256": file_sha256(ENCODED_VALID),
            "extraction": "sdp.load_encoded(valid) + attach_env + arm-specific prep; rpv2.predict_raw",
        },
        "prediction_files": {
            "F7200": {"path": str(F7200_VALID_PREDS.relative_to(REPO_ROOT)),
                      "sha256": file_sha256(F7200_VALID_PREDS)},
            "F10000": {"path": str(F10000_VALID_PREDS.relative_to(REPO_ROOT)),
                       "sha256": file_sha256(F10000_VALID_PREDS)},
            "combined": {"path": str(VALID_PREDS.relative_to(REPO_ROOT)),
                         "sha256": file_sha256(VALID_PREDS)},
        },
        "engineering_crosschecks": {
            "F10000_cpu_replay_vs_historical_valid_csv_max_abs": diff_hist,
            "F10000_cpu_replay_vs_historical_valid_csv_ok": bool(diff_hist <= REPLAY_ATOL),
            "F7200_top5_background_replay_vs_v3d_eval_max_abs": diff_v3d,
            "note": "historical valid predictions were produced on GPU; CPU float32 replay is not bit-exact",
        },
    }
    write_json(FREEZE_MANIFEST, manifest)
    log(f"[predict] written; F10000 vs historical CSV max|d|={diff_hist:.3e}; "
        f"F7200 top5 background vs v3d max|d|={diff_v3d}")
    return manifest


# ---------------------------------------------------------------------------
# stage: score (read official-valid y exactly once)
# ---------------------------------------------------------------------------


def bootstrap_paired(d: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n))
    for b in range(int(n)):
        smp = rng.integers(0, len(d), len(d))
        boots[b] = d[smp].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "resamples": int(n),
        "seed": int(seed),
        "statistic": (
            "paired signed per-molecule diff d_i=|e_F10000_i|-|e_F7200_i|; "
            "percentile bootstrap of the mean"
        ),
        "frac_F10000_better": float((d < 0).mean()),
        "frac_tied": float((d == 0).mean()),
        "n_F10000_better": int((d < 0).sum()),
        "n_F7200_better": int((d > 0).sum()),
    }


def decide_band(delta: float) -> str:
    """Pre-frozen three-region mapping for the matched-budget delta."""
    if delta <= DECISION_BANDS["A"]:
        return "A_clear_budget_gain"
    if delta <= DECISION_BANDS["B"]:
        return "B_moderate_gain"
    return "C_no_useful_gain"


def stage_score(device: torch.device) -> dict[str, Any]:
    manifest = read_json(FREEZE_MANIFEST)
    z = np.load(VALID_PREDS, allow_pickle=False)
    pred72 = np.asarray(z["pred_F7200"], dtype=np.float64)
    pred10 = np.asarray(z["pred_F10000"], dtype=np.float64)
    for key, path in (("F7200", F7200_VALID_PREDS), ("F10000", F10000_VALID_PREDS), ("combined", VALID_PREDS)):
        recorded = manifest["prediction_files"][key]["sha256"]
        actual = file_sha256(path)
        if recorded != actual:
            raise RuntimeError(f"{key}: prediction file changed after freeze ({recorded} != {actual})")

    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    y = np.array([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float64)
    n = len(y)
    if len(pred72) != n or len(pred10) != n:
        raise RuntimeError("prediction/label length mismatch")

    e72 = np.abs(pred72 - y)
    e10 = np.abs(pred10 - y)
    d = e10 - e72
    metrics = {
        "protocol_id": PROTOCOL_ID,
        "split": "official ZINC validation (1000)",
        "official_test_loaded": False,
        "n_molecules": n,
        "mae": {"F7200": float(e72.mean()), "F10000": float(e10.mean())},
        "delta_F10000_minus_F7200": float(d.mean()),
        "rmse": {"F7200": float(np.sqrt(np.mean((pred72 - y) ** 2))),
                 "F10000": float(np.sqrt(np.mean((pred10 - y) ** 2)))},
        "median_abs_error": {"F7200": float(np.median(e72)), "F10000": float(np.median(e10))},
        "p90_abs_error": {"F7200": float(np.percentile(e72, 90)), "F10000": float(np.percentile(e10, 90))},
        "p95_abs_error": {"F7200": float(np.percentile(e72, 95)), "F10000": float(np.percentile(e10, 95))},
        "max_abs_error": {"F7200": float(e72.max()), "F10000": float(e10.max())},
        "fit_mae_y": {
            "F7200": read_json(F7200_META)["auxiliary_internal_mae"]["fit"],
            "F10000": read_json(HIST_Y_JSON)["fit_raw_mae_y"],
        },
        "background": {
            "historical_F7200_monitor_top5_valid_raw": background_v3d_f_valid(),
            "historical_F10000_seed0_valid_raw": 0.11861754740116885,
            "note": "background only; the matched comparison uses the shared last-5 soup rule",
        },
        "gap_to_valid_thresholds": {
            "F7200": {"to_0.09": float(e72.mean() - 0.09), "to_0.08": float(e72.mean() - 0.08)},
            "F10000": {"to_0.09": float(e10.mean() - 0.09), "to_0.08": float(e10.mean() - 0.08)},
        },
    }
    write_json(OUT / "valid_metrics.json", metrics)

    paired = bootstrap_paired(d)
    write_json(OUT / "paired_bootstrap.json", paired)

    # ---- structural covariates (label-free) + subgroups -------------------
    import networkx as nx

    n_atoms = np.array([int(ds[i].x.shape[0]) for i in range(n)])
    cyc = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ei = ds[i].edge_index.numpy()
        edges = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(ei[0].tolist(), ei[1].tolist())}
        g = nx.Graph()
        g.add_nodes_from(range(n_atoms[i]))
        g.add_edges_from(edges)
        cyc[i] = int(g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g))

    def grp(mask: np.ndarray) -> dict[str, Any]:
        out = {"n": int(mask.sum())}
        if mask.any():
            out.update({
                "mae_F7200": float(e72[mask].mean()),
                "mae_F10000": float(e10[mask].mean()),
                "delta": float(d[mask].mean()),
                "frac_F10000_better": float((d[mask] < 0).mean()),
            })
        return out

    order72 = np.argsort(-e72)
    order10 = np.argsort(-e10)
    extreme_id = 172  # valid:0172 -> positional id in the official split
    keep = np.ones(n, dtype=bool)
    keep[extreme_id] = False
    quartile_edges = np.percentile(y, [0, 25, 50, 75, 100])
    subgroup = {
        "y_le_minus3": grp(y <= -3.0),
        "y_gt_minus3": grp(y > -3.0),
        "atoms_le_20": grp(n_atoms <= 20),
        "atoms_gt_20": grp(n_atoms > 20),
        "cyclomatic_0_1": grp(cyc <= 1),
        "cyclomatic_2_3": grp((cyc >= 2) & (cyc <= 3)),
        "cyclomatic_ge_4": grp(cyc >= 4),
        "y_quartiles": {
            f"q{k+1}": grp((y > quartile_edges[k]) & (y <= quartile_edges[k + 1])) if k > 0
            else grp(y <= quartile_edges[1])
            for k in range(4)
        },
        "extreme_rows": {
            "description": "descriptive only; the main metric always uses all 1000 rows",
            "top10_error_mass_share": {
                "F7200": float(np.sort(e72)[-10:].sum() / e72.sum()),
                "F10000": float(np.sort(e10)[-10:].sum() / e10.sum()),
            },
            "top1_error_mass_share": {
                "F7200": float(e72.max() / e72.sum()),
                "F10000": float(e10.max() / e10.sum()),
            },
            "max_error_rows": {
                "F7200": {"row": int(order72[0]), "y": float(y[order72[0]]), "e": float(e72[order72[0]])},
                "F10000": {"row": int(order10[0]), "y": float(y[order10[0]]), "e": float(e10[order10[0]])},
            },
            "delta_without_valid_0172": float(d[keep].mean()),
            "delta_without_F7200_max_error_row": float(np.delete(d, order72[0]).mean()),
            "delta_without_F10000_max_error_row": float(np.delete(d, order10[0]).mean()),
            "delta_without_both_max_error_rows": float(
                np.delete(d, sorted({int(order72[0]), int(order10[0])})).mean()
            ),
            "historical_row_note": (
                f"{EXTREME_VALID_ID} was already observed in earlier historical research "
                "(prototype cycle-routing round); per-row diagnostics on it are exposed-data diagnostics"
            ),
            "row_172": {
                "y": float(y[extreme_id]),
                "e_F7200": float(e72[extreme_id]),
                "e_F10000": float(e10[extreme_id]),
                "d": float(d[extreme_id]),
                "share_of_positive_delta_sum": float(d[extreme_id] / d[d > 0].sum()) if (d > 0).any() else None,
            },
        },
    }
    write_json(OUT / "subgroup_analysis.json", subgroup)

    # ---- pre-frozen decision rule ----------------------------------------
    delta = float(d.mean())
    ci = paired["ci95"]
    band = decide_band(delta)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "delta_F10000_minus_F7200": delta,
        "paired_ci95": ci,
        "ci_supports_negative_improvement": bool(ci[1] < 0.0),
        "band": band,
        "thresholds": DECISION_BANDS,
        "not_dominated_by_extreme_rows": bool(
            abs(subgroup["extreme_rows"]["delta_without_valid_0172"]) <= 1.5 * abs(delta) + 1e-9
            and abs(subgroup["extreme_rows"]["delta_without_both_max_error_rows"]) <= 1.5 * abs(delta) + 1e-9
        ),
        "absolute_target": {
            "F10000_valid_mae": float(e10.mean()),
            "gap_to_0.09": float(e10.mean() - 0.09),
            "gap_to_0.08": float(e10.mean() - 0.08),
            "reached_0.09": bool(e10.mean() <= 0.09),
            "reached_0.08": bool(e10.mean() <= 0.08),
        },
        "test_status": (
            "official test NOT re-evaluated in this round; the historical F10000 test number is an "
            "artifact of that frozen version and cannot certify this configuration"
        ),
        "notes": [
            "delta measures the whole training-budget regime change (rows + optimizer steps + "
            "fit-only preprocessing refit), not the isolated causal effect of 2800 extra labels",
            "official validation has been read in earlier rounds; this is a controlled comparison on "
            "an already-exposed instrument, not a fresh blind test",
        ],
    }
    write_json(OUT / "decision.json", decision)
    log(json.dumps(_j({"mae": metrics["mae"], "delta": delta, "ci": ci, "band": band}), indent=1))
    return {"metrics": metrics, "paired": paired, "subgroup": subgroup, "decision": decision}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("audit", "replay", "train", "predict", "score"))
    parser.add_argument("--seed", type=int, default=0, choices=[0], help="seed 0 only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=None, help="replay row limit (default all rows)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-molecules", type=int, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode == "audit":
        stage_audit(device)
    elif args.mode == "replay":
        stage_replay(device, args.limit)
    elif args.mode == "train":
        stage_train(device, int(args.epochs), bool(args.smoke))
    elif args.mode == "predict":
        stage_predict(device, args.smoke_molecules)
    elif args.mode == "score":
        stage_score(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
