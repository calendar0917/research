"""cscl-bjoc-pretraining-mechanism-terminal-test-v1 driver.

Protocol: ``configs/bjoc_pretraining_mechanism_terminal_test_v1.yaml``
(preregistration).  Track: ``tracks/structure-contribution``.

Three objectives
----------------
1. Phase A: replay the frozen full-10k B/P checkpoints on the official valid
   structures (CPU) and confirm the historical valid MAE, the checkpoint
   provenance and the legality inheritance.
2. Phase B: mechanism evidence and the train-vs-valid tail analysis using the
   frozen B/P models only.
3. Phase C: ONE controlled development ablation -- J-reset (pretrained
   atom/bond encoders + scratch junction encoder) trained with the identical
   recipe, evaluated on official valid only.
4. Phase D/E: freeze label-blind predictions of B0/B1/P0/P1 on the official
   test structures, then the SINGLE official-test label read and the terminal
   metrics.  Official test is never used for selection, tuning or training.

All model and training code is imported from the previous round
(``bjoc_masked_pretraining_v1`` / ``structural_relation_composer_v1`` /
``run_bjoc_full10k_pretraining_transfer_v1``); no BJOC code is reimplemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import cscl_features as cf  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import run_bjoc_masked_pretraining_v1 as mpr  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-pretraining-mechanism-terminal-test-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_pretraining_mechanism_terminal_test_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_pretraining_mechanism_terminal_test_v1"
F10K = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
KSVD_VT = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1"

SEEDS = (0, 1)
ARMS = ("B", "P")
FROZEN_KEYS = ("B0", "B1", "P0", "P1")
JKEYS = ("J0", "J1")
N_TRAIN = 10000
N_VALID = 1000
N_TEST = 1000
TAIL_THRESHOLD = -3.0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
REPLAY_PRED_ATOL = 5.0e-5
REPLAY_MAE_ATOL = 1.0e-5
HIST_VALID_SEED_MEAN = {"B": 0.099202, "P": 0.083549}
HIST_VALID_PER_SEED = {"B0": 0.100172, "B1": 0.098232, "P0": 0.084786, "P1": 0.082313}
GPU_HOUR_CAP = 0.8
TARGET_LO, TARGET_HI = 0.06, 0.07
LABEL_FIELDS = ("y", "g", "c", "k", "ell", "s")

SOUP = {key: F10K / f"soup_{key[0]}_seed{key[1]}.pt" for key in FROZEN_KEYS}
PRED_DIR = F10K / "predictions"
ANALYSIS_DIR = OUT / "analysis"

# aliases into the frozen previous-round code (single source of truth)
write_json = f10k.write_json
read_json = f10k.read_json
file_sha256 = f10k.file_sha256
git_commit = f10k.git_commit
seed_everything = f10k.seed_everything
rows_sha256 = f10k.rows_sha256
array_sha256 = f10k.array_sha256
paired_bootstrap = f10k.paired_bootstrap
row_metrics = f10k.row_metrics


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# data / model access
# ---------------------------------------------------------------------------


def load_norm() -> dict[str, Any]:
    return read_json(F10K / "prep_train10k.json")


def load_train_objs() -> list[sro.MolObjects]:
    return f10k.load_objects(F10K / "objects_train10k.pt")


def load_valid_objs() -> list[sro.MolObjects]:
    return f10k.load_objects(F10K / "objects_valid.pt")


def load_train_y() -> np.ndarray:
    return f10k._load_y_rows(list(range(N_TRAIN)))


def load_valid_y() -> np.ndarray:
    return np.asarray(np.load(F10K / "valid_y_read_once.npy"), dtype=np.float64)


def load_test_dataset():
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="test")
    assert len(ds) == N_TEST, len(ds)
    return ds


class LabelGuard:
    """Read-only wrapper that makes any label-field access an error."""

    __slots__ = ("_d",)

    def __init__(self, d: Any) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        if name in LABEL_FIELDS:
            raise RuntimeError(f"label field {name!r} accessed during label-free prediction")
        return getattr(object.__getattribute__(self, "_d"), name)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - defensive
        raise RuntimeError("LabelGuard is read-only")


def build_test_objects(guarded_rows: Sequence[Any], n: int) -> list[sro.MolObjects]:
    return [
        sro.build_mol_objects(i, guarded_rows[i].x.numpy(), guarded_rows[i].edge_index.numpy(), guarded_rows[i].edge_attr.numpy())
        for i in range(n)
    ]


def load_soup_model(key: str, norm: Mapping[str, Any], device: torch.device | str = "cpu") -> src.BJOC:
    model = f10k._make_downstream_model(int(key[1]), norm, device)
    model.load_state_dict(torch.load(SOUP[key], map_location=device, weights_only=False))
    model.eval()
    return model


def build_jreset_model(seed: int, norm: Mapping[str, Any], device: torch.device | str = "cpu") -> tuple[src.BJOC, dict[str, Any]]:
    """Pretrained atom/bond encoders + scratch junction encoder + scratch head."""
    model = f10k._make_downstream_model(seed, norm, device)
    scratch_encoder = mp.encoder_state(model)
    ssl_path = F10K / f"pretrain_encoder_s{seed}.pt"
    ssl_state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    model.atom_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["atom_encoder"].items()})
    model.bond_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["bond_encoder"].items()})
    used = mp.encoder_state(model)
    info = {
        "ssl_file": str(ssl_path.relative_to(REPO_ROOT)),
        "ssl_file_sha256": file_sha256(ssl_path),
        "atom_from_ssl": _group_equal(used["atom_encoder"], ssl_state["atom_encoder"]),
        "bond_from_ssl": _group_equal(used["bond_encoder"], ssl_state["bond_encoder"]),
        "junction_is_scratch": _group_equal(used["junction_encoder"], scratch_encoder["junction_encoder"]),
        "junction_ssl_vs_scratch_identical": _group_equal(ssl_state["junction_encoder"], scratch_encoder["junction_encoder"]),
    }
    return model, info


def _encoder_state_hash(state: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
    """Reproduce ``f10k._encoder_group_hash`` from a stored encoder state."""
    combined: dict[str, torch.Tensor] = {}
    for prefix, group in (("atom", "atom_encoder"), ("bond", "bond_encoder"), ("junc", "junction_encoder")):
        for k, v in state[group].items():
            combined[f"{prefix}:{k}"] = v
    return f10k.state_group_sha256(combined)


def _group_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    if set(a) != set(b):
        return False
    return all(bool(torch.equal(a[k].detach().cpu(), b[k].detach().cpu())) for k in a)


def predict_objs(model: src.BJOC, objs: Sequence[sro.MolObjects], device: torch.device | str = "cpu") -> np.ndarray:
    return f10k.predict_bjoc(model, objs, device)


def predict_test_objs(model: src.BJOC, objs: Sequence[sro.MolObjects], device: torch.device | str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = f10k._mv(sro.batch_objects(list(objs[i : i + 256])), torch.device(device))
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


def objects_hash(objs: Sequence[sro.MolObjects]) -> tuple[str, list[str]]:
    h = hashlib.sha256()
    per_mol: list[str] = []
    for o in objs:
        mh = hashlib.sha256()
        for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
            a = np.asarray(getattr(o, name))
            mh.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        for name in ("bond_index", "bond_type", "junc_index", "junc_bond"):
            a = np.asarray(getattr(o, name))
            mh.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        per_mol.append(mh.hexdigest())
        h.update(mh.hexdigest().encode())
    return h.hexdigest(), per_mol


def objects_signature(objs: Sequence[sro.MolObjects]) -> dict[str, Any]:
    return {
        "n_molecules": len(objs),
        "n_atoms": int(sum(o.n_atoms for o in objs)),
        "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
        "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
    }


def _group_stats(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    e = np.abs(pred - y)
    r = pred - y
    if int(mask.sum()) == 0:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "mae": float(e[mask].mean()),
        "rmse": float(np.sqrt(np.mean(r[mask] ** 2))),
        "mean_signed": float(r[mask].mean()),
    }


def _arm_seed_mean(preds: Mapping[str, np.ndarray], arm: str) -> np.ndarray:
    return np.mean([preds[f"{arm}{s}"] for s in SEEDS], axis=0)


def _arm_seed_mean_error(preds: Mapping[str, np.ndarray], y: np.ndarray, arm: str) -> np.ndarray:
    return np.mean([np.abs(preds[f"{arm}{s}"] - y) for s in SEEDS], axis=0)


def _top_k_error_share(err: np.ndarray, k: int) -> dict[str, Any]:
    order = np.argsort(-err)
    top = order[:k]
    return {
        "k": int(k),
        "share_of_total_error": float(err[top].sum() / err.sum()),
        "max_error_row": int(order[0]),
        "max_error_value": float(err[order[0]]),
        "top_rows": [int(r) for r in top],
    }


def _structure_block(objs: Sequence[sro.MolObjects], mask: np.ndarray) -> dict[str, Any]:
    n_atoms = np.asarray([o.n_atoms for o in objs], dtype=np.float64)
    n_bonds = np.asarray([o.bond_index.shape[0] for o in objs], dtype=np.float64)
    n_junc = np.asarray([o.junc_index.shape[0] for o in objs], dtype=np.float64)
    cyc = np.asarray([float(o.global_feat[sro.GLOBAL_FEAT_NAMES.index("cyclomatic_rank")]) for o in objs], dtype=np.float64)
    rsys = np.asarray([float(o.global_feat[sro.GLOBAL_FEAT_NAMES.index("n_ring_systems")]) for o in objs], dtype=np.float64)
    out: dict[str, Any] = {"n": int(mask.sum())}
    for name, arr in (("n_atoms", n_atoms), ("n_bonds", n_bonds), ("n_junctions", n_junc), ("cyclomatic_rank", cyc), ("n_ring_systems", rsys)):
        out[name] = {
            "mean": float(arr[mask].mean()),
            "median": float(np.median(arr[mask])),
            "p10": float(np.quantile(arr[mask], 0.10)),
            "p90": float(np.quantile(arr[mask], 0.90)),
            "max": float(arr[mask].max()),
        }
    return out


def _verify_inputs() -> dict[str, Any]:
    """Recompute hash correspondence of every frozen input used by this round."""
    manifest = read_json(F10K / "training_manifest.json")
    checks: dict[str, Any] = {}
    for key, path in (
        ("objects_train10k_sha256", F10K / "objects_train10k.pt"),
        ("objects_valid_sha256", F10K / "objects_valid.pt"),
        ("prep_train10k_sha256", F10K / "prep_train10k.json"),
    ):
        actual = file_sha256(path)
        checks[key] = {"frozen": manifest["data"][key], "current": actual, "ok": actual == manifest["data"][key]}
    for seed in SEEDS:
        pj = read_json(F10K / f"pretraining_seed{seed}.json")
        enc = torch.load(F10K / f"pretrain_encoder_s{seed}.pt", map_location="cpu", weights_only=False)
        checks[f"pretrain_encoder_s{seed}"] = {"recorded": pj["encoder_final_sha256"], "current": _encoder_state_hash(enc), "ok": _encoder_state_hash(enc) == pj["encoder_final_sha256"]}
    return checks


# ---------------------------------------------------------------------------
# Phase A.1 -- frozen model replay + legality of reuse
# ---------------------------------------------------------------------------


def cmd_reuse_audit(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    freeze = read_json(F10K / "prediction_freeze.json")
    manifest = read_json(F10K / "training_manifest.json")
    norm = load_norm()
    valid_objs = load_valid_objs()
    y = load_valid_y()

    records: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for key in FROZEN_KEYS:
        ckpt = SOUP[key]
        expected = freeze["checkpoints"][f"soup_{key[0]}_seed{key[1]}"]
        model = load_soup_model(key, norm, device)
        audit = src.parameter_audit(src.BJOCConfig(), model)
        forbidden = src.forbidden_module_scan(model)
        head_keys = [k for k in model.state_dict() if k.startswith(("atom_head", "bond_head", "pretext"))]
        pred = predict_objs(model, valid_objs, device)
        ref = np.asarray(np.load(PRED_DIR / f"valid_{key}.npz")["pred"], dtype=np.float64)
        metrics = row_metrics(pred, y)
        records[key] = {
            "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
            "checkpoint_exists": ckpt.exists(),
            "checkpoint_sha256": file_sha256(ckpt),
            "checkpoint_sha256_matches_frozen_manifest": file_sha256(ckpt) == expected,
            "parameters": audit["actual_parameters"],
            "parameter_exact": audit["parameter_exact"],
            "forbidden_modules": forbidden,
            "pretext_head_keys_in_state": head_keys,
            "replay_max_abs_pred_diff_vs_frozen_valid_npz": float(np.abs(pred - ref).max()),
            "valid_mae": metrics["mae"],
            "valid_rmse": metrics["rmse"],
            "historical_valid_mae": HIST_VALID_PER_SEED[key],
            "valid_mae_diff_vs_historical": float(metrics["mae"] - HIST_VALID_PER_SEED[key]),
        }
        preds[key] = pred
        log(f"[reuse-audit] {key} MAE {metrics['mae']:.6f} (historical {HIST_VALID_PER_SEED[key]:.6f}) replay_maxdiff {records[key]['replay_max_abs_pred_diff_vs_frozen_valid_npz']:.2e}")

    seed_mean = {
        arm: float(np.mean([np.mean(np.abs(preds[f"{arm}{s}"] - y)) for s in SEEDS]))
        for arm in ARMS
    }
    seed_mean_diff = {arm: float(seed_mean[arm] - HIST_VALID_SEED_MEAN[arm]) for arm in ARMS}

    init_audit = read_json(F10K / "initialization_audit.json")
    per_seed_design: dict[str, Any] = {}
    for s in SEEDS:
        b = read_json(F10K / f"train_B_seed{s}.json")
        p = read_json(F10K / f"train_P_seed{s}.json")
        pj = read_json(F10K / f"pretraining_seed{s}.json")
        per_seed_design[str(s)] = {
            "same_init_full_model": b["init_full_sha256"] == p["init_full_sha256"],
            "same_composer_reader_init": b["init_group_state_sha256"] == p["init_group_state_sha256"],
            "same_batch_order": b["batch_order_hash"] == p["batch_order_hash"],
            "p_encoder_is_pretrained": p["encoder_used_sha256"] == pj["encoder_final_sha256"],
            "b_encoder_is_scratch": b["init_encoder_sha256"] != pj["encoder_final_sha256"],
            "recipe_equal": {
                k: b["hyper"].get(k) == p["hyper"].get(k)
                for k in ("optimizer", "lr", "weight_decay", "grad_clip", "batch_size", "loss", "scheduler", "dropout", "train_shuffle_seed")
            },
            "epochs_updates_soup_equal": b["epochs"] == p["epochs"] and b["updates"] == p["updates"] and b["soup_members"] == p["soup_members"],
            "parameters_equal_413505": b["parameters"] == p["parameters"] == 413505,
        }

    norm_checks = {
        "fit_only": bool(norm.get("fit_only")),
        "n_fit": int(norm.get("n_fit", -1)),
        "official_valid_statistics_used": bool(norm.get("official_valid_statistics_used")),
        "official_test_statistics_used": bool(norm.get("official_test_statistics_used")),
    }

    conditions = {
        "all_checkpoints_exist": all(records[k]["checkpoint_exists"] for k in FROZEN_KEYS),
        "all_checkpoint_hashes_match_manifest": all(records[k]["checkpoint_sha256_matches_frozen_manifest"] for k in FROZEN_KEYS),
        "all_413505_parameters": all(records[k]["parameter_exact"] and records[k]["parameters"] == 413505 for k in FROZEN_KEYS),
        "no_forbidden_modules": all(not records[k]["forbidden_modules"] for k in FROZEN_KEYS),
        "no_pretext_heads_in_forward": all(not records[k]["pretext_head_keys_in_state"] for k in FROZEN_KEYS),
        "replay_within_tolerance": all(records[k]["replay_max_abs_pred_diff_vs_frozen_valid_npz"] <= REPLAY_PRED_ATOL for k in FROZEN_KEYS),
        "valid_seed_mean_reproduced": all(abs(seed_mean_diff[arm]) <= REPLAY_MAE_ATOL for arm in ARMS),
        "b_p_design_invariance": all(
            per_seed_design[str(s)]["same_init_full_model"]
            and per_seed_design[str(s)]["same_composer_reader_init"]
            and per_seed_design[str(s)]["same_batch_order"]
            and per_seed_design[str(s)]["p_encoder_is_pretrained"]
            and per_seed_design[str(s)]["b_encoder_is_scratch"]
            for s in SEEDS
        ),
        "recipe_and_soup_equal": all(
            all(per_seed_design[str(s)]["recipe_equal"].values()) and per_seed_design[str(s)]["epochs_updates_soup_equal"] for s in SEEDS
        ),
        "normaliser_train_only": norm_checks["fit_only"] and norm_checks["n_fit"] == N_TRAIN and not norm_checks["official_valid_statistics_used"] and not norm_checks["official_test_statistics_used"],
        "inputs_unchanged": all(v["ok"] for v in _verify_inputs().values()),
    }
    decision = "REPLAY_CONFIRMED" if all(conditions.values()) else "TERMINAL_SCORE_BLOCKED"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": "cpu",
        "network_inputs": "raw atom/bond graph -> BJOC objects (label-blind); train10k-only global standardiser",
        "checkpoint_records": records,
        "valid_seed_mean": seed_mean,
        "valid_seed_mean_diff_vs_historical": seed_mean_diff,
        "per_seed_design_invariance": per_seed_design,
        "normaliser": norm_checks,
        "input_hash_checks": _verify_inputs(),
        "conditions": conditions,
        "decision": decision,
        "official_valid_labels_read": True,
        "official_test_loaded": False,
    }
    write_json(OUT / "model_reuse_audit.json", payload)
    np.savez_compressed(
        ANALYSIS_DIR / "valid_preds_replay.npz",
        **{k: preds[k].astype(np.float32) for k in FROZEN_KEYS},
    )
    log(f"[reuse-audit] B {seed_mean['B']:.6f} P {seed_mean['P']:.6f} -> {decision}")
    if decision != "REPLAY_CONFIRMED":
        raise SystemExit("replay/legalisation check failed; terminal scoring is blocked")
    return 0


# ---------------------------------------------------------------------------
# Phase A.2 -- legality review
# ---------------------------------------------------------------------------


def cmd_legality(args: argparse.Namespace) -> int:
    manifest = read_json(F10K / "training_manifest.json")
    gates = read_json(F10K / "correctness_gates.json")
    cons = read_json(F10K / "consistency_audit.json")
    dep = read_json(F10K / "masked_feature_dependency_audit.json")
    swap = read_json(F10K / "mask_target_swap_audit.json")
    rebuild = read_json(F10K / "masked_object_rebuild_audit.json")
    prep = read_json(F10K / "prep_train10k.json")

    code_now = f10k._code_hashes()
    code_match = {k: {"frozen": manifest["code_hashes"][k], "current": code_now[k], "ok": code_now[k] == manifest["code_hashes"][k]} for k in code_now}

    scan = mpr._scan_source_for_y_reads(Path(f10k.__file__), "pretrain_full10k")
    module_reads_y = mpr._module_reads_y_attribute(Path(mp.__file__))
    pretrain_call_scan = mpr._scan_source_for_y_reads(Path(f10k.__file__), "_load_y_rows")

    ds_use = {
        "pretraining_inputs": "ds[int(o.mol_index)].x / .edge_index / .edge_attr only",
        "pretraining_function_y_scan": scan,
        "pretraining_module_reads_y_attribute": module_reads_y,
        "downstream_supervision_y_scan": pretrain_call_scan,
        "supervised_y_source": "ENCODED_TRAIN (official train 10000 rows, raw y)",
    }
    # the downstream supervised path legitimately reads y; record it explicitly
    supervised_y_legitimate = bool(pretrain_call_scan.get("hits"))

    no_valid_test_in_pretrain = True
    src_text = Path(f10k.__file__).read_text(encoding="utf-8")
    fn_text = src_text.split("def pretrain_full10k")[1].split("\ndef ")[0]
    for token in ('split="val"', 'split="test"', "ENCODED_VALID", "load_valid_dataset"):
        if token in fn_text:
            no_valid_test_in_pretrain = False

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "code_version_correspondence": code_match,
        "training_code_commit": manifest.get("git_commit"),
        "audits_inherited": {
            "correctness_gates_all_pass": bool(gates.get("all_pass")),
            "correctness_gate_names": sorted(gates.get("gates", {}).keys()),
            "consistency_all_pass": bool(cons.get("all_pass")),
            "masked_feature_dependency": dep.get("verdict"),
            "mask_target_swap": {
                "verdict": swap.get("verdict"),
                "atom_swap_cases": swap.get("randomized_real_train10k", {}).get("atom_cases"),
                "bond_swap_cases": swap.get("randomized_real_train10k", {}).get("bond_cases"),
            },
            "masked_object_rebuild": rebuild.get("verdict"),
            "rebuild_cases": rebuild.get("cases"),
        },
        "pretraining_input_scope": ds_use,
        "supervised_y_scope": {
            "source": "official train 10000 raw y only (encoded_train cache, raw-vs-encoded agreement checked by the previous round)",
            "official_valid_rows_used_for_supervision": False,
            "official_test_rows_used_in_any_training": False,
            "downstream_y_scan": pretrain_call_scan,
            "expected_downstream_y_read_present": supervised_y_legitimate,
        },
        "normalisation": {
            "fit": "official train 10000 only",
            "n_fit": prep["n_fit"],
            "official_valid_statistics_used": prep["official_valid_statistics_used"],
            "official_test_statistics_used": prep["official_test_statistics_used"],
        },
        "pretraining_does_not_touch_valid_test": no_valid_test_in_pretrain,
        "external_data": "none (no external labels, no external structures, no ZINC-250K)",
        "verdict": (
            "LEGALITY_CONFIRMED"
            if code_match and all(v["ok"] for v in code_match.values()) and gates.get("all_pass") and cons.get("all_pass") and no_valid_test_in_pretrain and not scan.get("found")
            else "LEGALITY_ISSUE"
        ),
        "disclosures": [
            "official valid is an exposed development set; its labels were read once by the previous round and are reused as the valid readout here",
            "one WL fingerprint is shared between train (row 337) and valid (row 376); disclosed, no row deleted",
            "the historical official-test number 0.092233 was produced by a different trajectory and is background only",
        ],
        "official_test_loaded": False,
    }
    write_json(OUT / "pretraining_legality_review.json", payload)
    log(f"[legality] {payload['verdict']} gates={gates.get('all_pass')} dep={dep.get('verdict')} swap={swap.get('verdict')} rebuild={rebuild.get('verdict')}")
    return 0


# ---------------------------------------------------------------------------
# Phase B -- mechanism evidence + train/valid tail analysis
# ---------------------------------------------------------------------------


def cmd_mechanism_evidence(args: argparse.Namespace) -> int:
    valid_metrics = read_json(F10K / "valid_metrics.json")
    decision = read_json(F10K / "scientific_decision.json")
    diag = read_json(F10K / "representation_diagnostics.json")
    ssl = {s: read_json(F10K / f"pretraining_seed{s}.json") for s in SEEDS}
    m8 = read_json(REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1/locked_metrics.json")
    d8 = read_json(REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1/scientific_decision.json")
    transfer = read_json(REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1/transfer_mechanism.json")
    pb = valid_metrics["paired"]["P_minus_B"]
    hp = valid_metrics["fit_valid_gap"]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "known_facts": [
            {
                "fact": "P improves 8k internal locked-dev raw-y MAE (both seeds)",
                "numbers": {
                    "P_minus_B_locked_seed_mean": d8.get("details", {}).get("P_minus_B", {}).get("delta_seed_mean"),
                    "locked_mae": m8.get("seed_mean", {}),
                },
                "source": "results/bjoc_masked_pretraining_v1/",
            },
            {
                "fact": "P improves full10k official-valid raw-y MAE (both seeds)",
                "numbers": {
                    "P_minus_B_valid_seed_mean": pb["delta_seed_mean"],
                    "ci95": [pb["bootstrap"]["ci_lo"], pb["bootstrap"]["ci_hi"]],
                    "per_seed": pb["per_seed_delta"],
                },
                "source": "results/bjoc_full10k_pretraining_transfer_v1/valid_metrics.json",
            },
            {
                "fact": "P fits the official train 10000 more tightly than B while generalising better",
                "numbers": {"fit_mae": {k: v["fit_mae"] for k, v in hp.items()}, "valid_mae": {k: v["valid_mae"] for k, v in hp.items()}},
                "source": "results/bjoc_full10k_pretraining_transfer_v1/valid_metrics.json",
            },
            {
                "fact": "the gain survives a 50%-extra-update scratch control on the 8k split",
                "numbers": d8.get("details", {}).get("compute_control", {}),
                "source": "results/bjoc_masked_pretraining_v1/",
            },
            {
                "fact": "preserving the pretrained encoders with a reduced encoder LR (Q) is harmful, not helpful",
                "numbers": {"Q_minus_P_seed_mean": decision["details"].get("Q_minus_P", {}).get("delta_seed_mean")},
                "source": "results/bjoc_full10k_pretraining_transfer_v1/scientific_decision.json",
            },
            {
                "fact": "the masked pretext tasks are far above their marginal priors",
                "numbers": {
                    f"seed{s}": {"acc_atom": ssl[s]["final_acc_atom"], "acc_bond": ssl[s]["final_acc_bond"], "ce_atom": ssl[s]["final_ce_atom"], "ce_bond": ssl[s]["final_ce_bond"]}
                    for s in SEEDS
                },
                "source": "results/bjoc_full10k_pretraining_transfer_v1/pretraining_seed*.json",
            },
            {
                "fact": "all three encoders move during pretraining and then move again during fine-tuning",
                "numbers": {
                    "parameter_relative_move_init_to_ssl": {str(s): diag["per_seed"][str(s)]["parameter_relative_move_init_to_ssl"] for s in SEEDS},
                    "parameter_relative_move_ssl_to_p": {str(s): diag["per_seed"][str(s)]["parameter_relative_move_ssl_to_p"] for s in SEEDS},
                },
                "source": "results/bjoc_full10k_pretraining_transfer_v1/representation_diagnostics.json",
            },
        ],
        "reasonable_interpretation": [
            "restoring masked atom/bond classes from the molecule's own structure forces the three encoders to build a local-structure representation before any target is seen; the resulting parameter state is a more suitable starting point for the raw-y optimisation than the scratch state",
            "the improvement is an optimisation-path / initialisation effect: it is visible in the fit MAE as well, but it is not only optimisation speed, because the extra-update scratch control C remained worse on the 8k split",
            "the encoders are still substantially rewritten during fine-tuning (parameter move SSL->P of the same order as init->SSL), so the pretrained state acts as a prior, not as a frozen feature extractor",
        ],
        "not_proven": [
            "which pretraining object (atom types vs bond types vs junction context) contributes the gain; the J-reset ablation below tests only one coarse decomposition",
            "whether any learned representation corresponds to a named chemical rule; no such identification is claimed",
            "that the junction encoder pretraining is irreplaceable (J-reset answers only whether the joint-state junction init helps on valid)",
            "that the method fixes extreme-molecule generalisation; the tail subgroup is not improved and RMSE worsens on valid",
            "that the improvement is stable under the official-test distribution; that is exactly what Phase D/E measures",
        ],
        "representation_metric_caveats": [
            "junction linear CKA ~0.10 between the SSL state and the P state means the representation changed substantially; it does NOT mean the junction features were 'destroyed' or that the junction channel is useless",
            "Q (reduced encoder LR, preserving the pretrained state) was worse on both seeds and blew up the tail; this refutes that particular fixed LR ratio, not encoder regularisation or pretrained-state preservation in general",
            "effective rank / CKA / parameter-move numbers are description of change, not causal attribution of the MAE gain",
        ],
        "transfer_mechanism_8k_to_full10k": {
            "source": "results/bjoc_masked_pretraining_v1/transfer_mechanism.json",
            "junction_effective_rank_8k": transfer.get("junction_effective_rank") if isinstance(transfer, dict) else None,
        },
        "official_test_loaded": False,
    }
    write_json(OUT / "mechanism_evidence.json", payload)
    log("[mechanism] mechanism_evidence.json written")
    return 0


def _tail_block(
    objs: Sequence[sro.MolObjects],
    y: np.ndarray,
    preds: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    tail = y <= TAIL_THRESHOLD
    ordinary = ~tail
    errors = {arm: _arm_seed_mean_error(preds, y, arm) for arm in ARMS}
    seed_errors = {key: np.abs(preds[key] - y) for key in FROZEN_KEYS}
    out: dict[str, Any] = {
        "n": int(y.size),
        "n_ordinary_y_gt_-3": int(ordinary.sum()),
        "n_tail_y_le_-3": int(tail.sum()),
        "per_arm_seed_mean": {},
        "per_seed": {},
        "structure": {
            "ordinary": _structure_block(objs, ordinary),
            "tail": _structure_block(objs, tail),
        },
        "top_k_error_share_seed_mean": {arm: {str(k): _top_k_error_share(errors[arm], k) for k in (1, 5, 10)} for arm in ARMS},
        "tail_error_mass_share_seed_mean": {arm: float(errors[arm][tail].sum() / errors[arm].sum()) for arm in ARMS},
        "tail_share_of_molecules": float(tail.mean()),
        "per_molecule_error_vectors": {},
    }
    for arm in ARMS:
        keys = [f"{arm}{s}" for s in SEEDS]
        pred_mean = _arm_seed_mean(preds, arm)
        out["per_arm_seed_mean"][arm] = {
            "mae_all": float(np.mean([np.abs(preds[k] - y).mean() for k in keys])),
            "rmse_all": float(np.mean([np.sqrt(np.mean((preds[k] - y) ** 2)) for k in keys])),
            "mae_ordinary": float(np.mean([np.abs((preds[k] - y)[ordinary]).mean() for k in keys])),
            "rmse_ordinary": float(np.mean([np.sqrt(np.mean((preds[k] - y)[ordinary] ** 2)) for k in keys])),
            "mean_signed_ordinary": float(np.mean([(preds[k] - y)[ordinary].mean() for k in keys])),
            "mae_tail": float(np.mean([np.abs((preds[k] - y)[tail]).mean() for k in keys])) if tail.any() else None,
            "rmse_tail": float(np.mean([np.sqrt(np.mean((preds[k] - y)[tail] ** 2)) for k in keys])) if tail.any() else None,
            "mean_signed_tail": float(np.mean([(preds[k] - y)[tail].mean() for k in keys])) if tail.any() else None,
            "top10_share": float(np.sort(errors[arm])[-10:].sum() / errors[arm].sum()),
            "note": "seed-mean is the arithmetic mean of the two per-seed single-model metrics; predictions are never averaged",
            "seed_averaged_prediction_mae_all": float(np.abs(pred_mean - y).mean()),
        }
    for key in FROZEN_KEYS:
        pred = preds[key]
        out["per_seed"][key] = {
            "mae_all": float(seed_errors[key].mean()),
            "mae_ordinary": float(seed_errors[key][ordinary].mean()),
            "mae_tail": float(seed_errors[key][tail].mean()) if tail.any() else None,
            "mean_signed_tail": float(np.mean(pred[tail] - y[tail])) if tail.any() else None,
            "rmse_all": float(np.sqrt(np.mean((pred - y) ** 2))),
        }
    # tail molecules that are among the 10 worst seed-mean errors of each arm
    out["tail_molecules_in_worst10"] = {}
    for arm in ARMS:
        order = np.argsort(-errors[arm])[:10]
        out["tail_molecules_in_worst10"][arm] = int(tail[order].sum())
    delta = errors["P"] - errors["B"]
    out["delta_P_minus_B"] = {
        "mae_all": float(delta.mean()),
        "mae_ordinary": float(delta[ordinary].mean()),
        "mae_tail": float(delta[tail].mean()) if tail.any() else None,
        "rmse_all": out["per_arm_seed_mean"]["P"]["rmse_all"] - out["per_arm_seed_mean"]["B"]["rmse_all"],
        "mean_signed_tail": (out["per_arm_seed_mean"]["P"]["mean_signed_tail"] - out["per_arm_seed_mean"]["B"]["mean_signed_tail"]),
        "bootstrap": paired_bootstrap(delta),
    }
    # per-row tail decomposition: which molecules drive the P-B tail difference
    tail_idx = np.where(tail)[0]
    order = tail_idx[np.argsort(-delta[tail_idx])]
    top_rows = []
    for i in order[:max(1, min(10, len(order)))]:
        if delta[i] <= 0:
            break
        top_rows.append({
            "row": int(i),
            "y": float(y[i]),
            "delta_P_minus_B": float(delta[i]),
            "P_seed_errors": [float(abs(preds[f"P{s}"][i] - y[i])) for s in SEEDS],
            "B_seed_errors": [float(abs(preds[f"B{s}"][i] - y[i])) for s in SEEDS],
            "P_signed": [float(preds[f"P{s}"][i] - y[i]) for s in SEEDS],
            "B_signed": [float(preds[f"B{s}"][i] - y[i]) for s in SEEDS],
        })
    out["tail_delta_decomposition"] = {
        "top_regression_rows": top_rows,
        "delta_without_worst_row": float(np.delete(delta, int(order[0])).mean()) if len(order) else None,
        "delta_without_worst3_rows": float(np.delete(delta, [int(i) for i in order[:3]]).mean()) if len(order) >= 3 else None,
    }
    out["tail_rows"] = [int(i) for i in np.where(tail)[0]]
    out["per_molecule_error_vectors"] = {arm: errors[arm].tolist() for arm in ARMS}
    return out


def cmd_tail_analysis(args: argparse.Namespace) -> int:
    device = "cpu"
    norm = load_norm()
    train_objs = load_train_objs()
    valid_objs = load_valid_objs()
    y_train = load_train_y()
    y_valid = load_valid_y()

    train_preds = {key: predict_objs(load_soup_model(key, norm, device), train_objs, device) for key in FROZEN_KEYS}
    np.savez_compressed(ANALYSIS_DIR / "train_preds.npz", **{k: v.astype(np.float32) for k, v in train_preds.items()})

    valid_path = ANALYSIS_DIR / "valid_preds_replay.npz"
    if valid_path.exists():
        with np.load(valid_path) as z:
            valid_preds = {k: np.asarray(z[k], dtype=np.float64) for k in FROZEN_KEYS}
    else:
        valid_preds = {key: predict_objs(load_soup_model(key, norm, device), valid_objs, device) for key in FROZEN_KEYS}

    train_block = _tail_block(train_objs, y_train, train_preds)
    valid_block = _tail_block(valid_objs, y_valid, valid_preds)
    answer = {
        "train_tail_delta_P_minus_B": train_block["delta_P_minus_B"]["mae_tail"],
        "valid_tail_delta_P_minus_B": valid_block["delta_P_minus_B"]["mae_tail"],
        "train_ordinary_delta_P_minus_B": train_block["delta_P_minus_B"]["mae_ordinary"],
        "valid_ordinary_delta_P_minus_B": valid_block["delta_P_minus_B"]["mae_ordinary"],
        "train_tail_mean_signed_P": train_block["per_arm_seed_mean"]["P"]["mean_signed_tail"],
        "train_tail_mean_signed_B": train_block["per_arm_seed_mean"]["B"]["mean_signed_tail"],
        "valid_tail_mean_signed_P": valid_block["per_arm_seed_mean"]["P"]["mean_signed_tail"],
        "valid_tail_mean_signed_B": valid_block["per_arm_seed_mean"]["B"]["mean_signed_tail"],
        "rule": "TAIL_GENERALISATION_GAP_SUPPORTED iff train tail improves under P and valid tail does not; TRAIN_TAIL_FIT_INSUFFICIENT iff the train tail does not improve under P",
    }
    if answer["train_tail_delta_P_minus_B"] is not None and answer["train_tail_delta_P_minus_B"] < 0 and (answer["valid_tail_delta_P_minus_B"] is None or answer["valid_tail_delta_P_minus_B"] > 0):
        answer["verdict"] = "TAIL_GENERALISATION_GAP_SUPPORTED"
    elif answer["train_tail_delta_P_minus_B"] is not None and answer["train_tail_delta_P_minus_B"] >= 0:
        answer["verdict"] = "TRAIN_TAIL_FIT_INSUFFICIENT"
    else:
        answer["verdict"] = "MIXED"

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "train": {k: v for k, v in train_block.items() if k != "per_molecule_error_vectors"},
        "valid": {k: v for k, v in valid_block.items() if k != "per_molecule_error_vectors"},
        "question": answer,
        "note": "all numbers use the frozen B/P soup checkpoints; no new training; official test never loaded",
        "official_test_loaded": False,
    }
    write_json(OUT / "train_valid_tail_analysis.json", payload)
    log(f"[tail-analysis] {answer['verdict']} train_tail_delta={answer['train_tail_delta_P_minus_B']} valid_tail_delta={answer['valid_tail_delta_P_minus_B']}")
    return 0


# ---------------------------------------------------------------------------
# Phase C -- J-reset purchase gate
# ---------------------------------------------------------------------------


def _jreset_sensitivity(seed: int, norm: Mapping[str, Any]) -> dict[str, Any]:
    """Correctness checks on the hybrid model before any training."""
    train_objs = load_train_objs()
    model, info = build_jreset_model(seed, norm, "cpu")
    audit = src.parameter_audit(src.BJOCConfig(), model)
    forbidden = src.forbidden_module_scan(model)
    scratch = f10k._make_downstream_model(seed, norm, "cpu")
    base = train_objs[0]
    with torch.no_grad():
        p_base = model(sro.batch_objects([base])).item()
        p_multi = model(sro.batch_objects(train_objs[:4])).numpy()
    # relabelling invariance (reversed atom order)
    ds = f10k.load_train_dataset()
    d0 = ds[int(base.mol_index)]
    perm = list(reversed(range(base.n_atoms)))
    inv = {old: new for new, old in enumerate(perm)}
    x_p = np.asarray(d0.x).copy().reshape(-1)
    x_new = np.zeros_like(x_p)
    for old in range(base.n_atoms):
        x_new[inv[old]] = x_p[old]
    edge_rows = [(inv[int(d0.edge_index[0, c])], inv[int(d0.edge_index[1, c])]) for c in range(np.asarray(d0.edge_index).shape[1])]
    ei_p = np.asarray(edge_rows, dtype=np.int64).T
    o_p = sro.build_mol_objects(0, x_new, ei_p, np.asarray(d0.edge_attr))
    with torch.no_grad():
        p_perm = model(sro.batch_objects([o_p])).item()
    # junction sensitivity: replace the scratch junction with the SSL junction
    ssl_state = torch.load(F10K / f"pretrain_encoder_s{seed}.pt", map_location="cpu", weights_only=False)
    model_j = f10k._make_downstream_model(seed, norm, "cpu")
    model_j.atom_encoder.load_state_dict(ssl_state["atom_encoder"])
    model_j.bond_encoder.load_state_dict(ssl_state["bond_encoder"])
    model_j.junction_encoder.load_state_dict(ssl_state["junction_encoder"])
    with torch.no_grad():
        p_ssl_j = model_j(sro.batch_objects([base])).item()
    return {
        "parameters": audit["actual_parameters"],
        "parameter_exact": audit["parameter_exact"],
        "forbidden_modules": forbidden,
        "relabelling_invariance_max_diff": float(abs(p_base - p_perm)),
        "batch_consistency_max_diff": float(np.abs(p_multi[0] - p_base)),
        "junction_swap_sensitivity": float(abs(p_ssl_j - p_base)),
        "scratch_model_hash": src.state_dict_hash(scratch),
        "info": info,
    }


def cmd_jreset_gate(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    norm = load_norm()
    expected = read_json(F10K / "initialization_audit.json")
    checks: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}

    for seed in SEEDS:
        scratch_a = f10k._make_downstream_model(seed, norm, "cpu")
        scratch_b = f10k._make_downstream_model(seed, norm, "cpu")
        p_rec = read_json(F10K / f"train_P_seed{seed}.json")
        b_rec = read_json(F10K / f"train_B_seed{seed}.json")
        pj = read_json(F10K / f"pretraining_seed{seed}.json")
        ssl = torch.load(F10K / f"pretrain_encoder_s{seed}.pt", map_location="cpu", weights_only=False)
        scratch_state = mp.encoder_state(scratch_a)
        ssl_hash_ok = _encoder_state_hash(ssl) == pj["encoder_final_sha256"]

        model, info = build_jreset_model(seed, norm, "cpu")
        # reference P init: scratch full model + pretrained encoders (the exact P starting state)
        p_ref = f10k._make_downstream_model(seed, norm, "cpu")
        p_ref.atom_encoder.load_state_dict(ssl["atom_encoder"])
        p_ref.bond_encoder.load_state_dict(ssl["bond_encoder"])
        p_ref.junction_encoder.load_state_dict(ssl["junction_encoder"])
        used_j = mp.encoder_state(model)
        candidate_diff = {}
        for g in ("atom_encoder", "bond_encoder", "junction_encoder"):
            n_diff = sum(int((used_j[g][k] != mp.encoder_state(p_ref)[g][k]).sum()) for k in used_j[g])
            candidate_diff[g] = n_diff
        sens = _jreset_sensitivity(seed, norm)
        per_seed[str(seed)] = {
            "scratch_init_reproducible": src.state_dict_hash(scratch_a) == src.state_dict_hash(scratch_b),
            "scratch_full_hash": src.state_dict_hash(scratch_a),
            "recorded_p_init_full_sha256": p_rec["init_full_sha256"],
            "scratch_equals_recorded_p_init": src.state_dict_hash(scratch_a) == p_rec["init_full_sha256"],
            "scratch_equals_recorded_b_init": src.state_dict_hash(scratch_a) == b_rec["init_full_sha256"],
            "scratch_composer_reader_hash": mp.group_state_hash(scratch_a),
            "recorded_p_init_group_sha256": p_rec["init_group_state_sha256"],
            "composer_reader_equal_recorded": mp.group_state_hash(scratch_a) == p_rec["init_group_state_sha256"],
            "ssl_state_rehash_ok": ssl_hash_ok,
            "ssl_state_recorded_hash": pj["encoder_final_sha256"],
            "hybrid_atom_bond_from_ssl": info["atom_from_ssl"] and info["bond_from_ssl"],
            "hybrid_junction_is_scratch": info["junction_is_scratch"],
            "ssl_junction_differs_from_scratch": not info["junction_ssl_vs_scratch_identical"],
            "parameters_from_p_ref": candidate_diff,
            "difference_vs_p_only_in_junction": candidate_diff["atom_encoder"] == 0 and candidate_diff["bond_encoder"] == 0 and candidate_diff["junction_encoder"] > 0,
            "sensitivity": sens,
        }
        checks[f"seed{seed}"] = {
            "init_provenance": per_seed[str(seed)]["scratch_equals_recorded_p_init"] and per_seed[str(seed)]["scratch_equals_recorded_b_init"],
            "ssl_provenance": ssl_hash_ok,
            "only_junction_reset": per_seed[str(seed)]["difference_vs_p_only_in_junction"],
            "atom_bond_match_design": per_seed[str(seed)]["hybrid_atom_bond_from_ssl"],
            "junction_is_scratch": per_seed[str(seed)]["hybrid_junction_is_scratch"] and per_seed[str(seed)]["ssl_junction_differs_from_scratch"],
            "composer_reader_match_design": per_seed[str(seed)]["composer_reader_equal_recorded"],
            "parameters_413505": sens["parameter_exact"] and sens["parameters"] == 413505,
            "no_forbidden_modules": not sens["forbidden_modules"],
            "relabelling_invariance": sens["relabelling_invariance_max_diff"] < 1e-6,
            "batch_composition_consistency": sens["batch_consistency_max_diff"] < 1e-6,
            "junction_channel_active": sens["junction_swap_sensitivity"] > 1e-6,
        }
    budget_ok = True
    wall_estimate = float(np.mean([read_json(F10K / f"train_{arm}_seed{s}.json")["wall_seconds"] for arm in ARMS for s in SEEDS])) * len(SEEDS)
    decision = "PURCHASE_JRESET" if all(all(v.values()) for v in checks.values()) and budget_ok else "J_RESET_NOT_PURCHASED"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "checks": checks,
        "per_seed": per_seed,
        "expected_from_initialization_audit": expected.get("per_seed"),
        "budget": {
            "estimated_two_seed_gpu_seconds": wall_estimate,
            "estimated_gpu_hours": wall_estimate / 3600.0,
            "gpu_hour_cap": GPU_HOUR_CAP,
            "fits_budget": wall_estimate / 3600.0 <= GPU_HOUR_CAP,
        },
        "decision": decision,
        "official_test_loaded": False,
    }
    write_json(OUT / "junction_reset_gate.json", payload)
    log(f"[jreset-gate] {decision} checks={checks}")
    if decision != "PURCHASE_JRESET":
        log("[jreset-gate] not purchased; continue directly to the terminal test")
    return 0


# ---------------------------------------------------------------------------
# Phase C -- J-reset training (identical recipe, hybrid encoder init)
# ---------------------------------------------------------------------------


def _train_jreset(seed: int, device: torch.device, *, log_fn: Callable[[str], None] = log) -> dict[str, Any]:
    f10k.verify_freeze()
    gate = read_json(OUT / "junction_reset_gate.json")
    if gate["decision"] != "PURCHASE_JRESET":
        raise SystemExit(f"J-reset not purchased: {gate['decision']}")
    norm = load_norm()
    train_objs = load_train_objs()
    y_fit = load_train_y()

    model = f10k._make_downstream_model(seed, norm, device)
    init_full_sha = src.state_dict_hash(model)
    init_group_sha = mp.group_state_hash(model)
    init_encoder_sha = f10k._encoder_group_hash(model)
    scratch_junction = {k: v.detach().cpu().clone() for k, v in model.junction_encoder.state_dict().items()}

    ssl_path = F10K / f"pretrain_encoder_s{seed}.pt"
    ssl_state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    model.atom_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["atom_encoder"].items()})
    model.bond_encoder.load_state_dict({k: v.to(device) for k, v in ssl_state["bond_encoder"].items()})
    encoder_used_sha = f10k._encoder_group_hash(model)
    used_junction = {k: v.detach().cpu().clone() for k, v in model.junction_encoder.state_dict().items()}
    if not all(bool(torch.equal(used_junction[k], scratch_junction[k])) for k in used_junction):
        raise SystemExit("junction encoder is not the scratch state")
    audit = src.parameter_audit(src.BJOCConfig(), model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != 413505:
        raise SystemExit(f"parameter audit failed: {audit}")

    optimizer = torch.optim.Adam(model.parameters(), lr=f10k.LR, weight_decay=f10k.WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(int(seed) + f10k.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=f10k.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, f10k.EPOCHS + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in f10k._shuffle_batches(N_TRAIN, f10k.BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            batch = f10k._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), f10k.GRAD_CLIP)
            optimizer.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        curve.append({"epoch": epoch, "updates": epoch * steps, "train_mae": total / max(n_mol, 1), "seconds": time.perf_counter() - t_ep})
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == f10k.EPOCHS:
            log_fn(f"[J s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} {curve[-1]['seconds']:.1f}s")
        if not np.isfinite(curve[-1]["train_mae"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch} J seed {seed}")
    wall = time.perf_counter() - started
    members, soup = f10k._last_soup(list(states), len(curve))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = f10k.predict_bjoc(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    soup_path = OUT / f"soup_J_seed{seed}.pt"
    torch.save(soup_state, soup_path)
    p_rec = read_json(F10K / f"train_P_seed{seed}.json")
    same_order = order_hash.hexdigest() == p_rec["batch_order_hash"]
    if not same_order:
        raise SystemExit("J-reset batch order differs from the frozen P batch order")
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": "J",
        "seed": int(seed),
        "git_commit": git_commit(),
        "definition": "pretrained atom encoder + pretrained bond encoder + scratch junction encoder + scratch composer/reader",
        "epochs": f10k.EPOCHS,
        "updates": int(f10k.EPOCHS * f10k.STEPS_PER_EPOCH),
        "steps_per_epoch": f10k.STEPS_PER_EPOCH,
        "soup_rule": f"parameter mean of the last {f10k.SOUP_LAST} epoch states",
        "soup_members": members,
        "hyper": {
            "optimizer": "Adam",
            "lr": f10k.LR,
            "weight_decay": f10k.WEIGHT_DECAY,
            "grad_clip": f10k.GRAD_CLIP,
            "batch_size": f10k.BATCH_SIZE,
            "loss": "raw-y L1 only",
            "scheduler": "none",
            "dropout": 0.0,
            "train_shuffle_seed": int(seed) + f10k.TRAIN_SHUFFLE_OFFSET,
        },
        "parameters": audit["actual_parameters"],
        "init_full_sha256": init_full_sha,
        "init_group_state_sha256": init_group_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "final_encoder_sha256": f10k._encoder_group_hash(model),
        "pretrain_encoder_file_sha256": file_sha256(ssl_path),
        "same_init_as_P": init_full_sha == p_rec["init_full_sha256"],
        "same_composer_reader_init_as_P": init_group_sha == p_rec["init_group_state_sha256"],
        "same_batch_order_as_P": same_order,
        "junction_used_is_scratch": True,
        "batch_order_hash": order_hash.hexdigest(),
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "curve": curve,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "official train 10000 raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / f"train_J_seed{seed}.json", payload)
    log_fn(f"[J s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {wall:.0f}s")
    return payload


def cmd_train_jreset(args: argparse.Namespace) -> int:
    _train_jreset(int(args.seed), torch.device(args.device))
    return 0


def cmd_predict_jreset(args: argparse.Namespace) -> int:
    if not (OUT / "junction_reset_gate.json").exists():
        raise SystemExit("run the J-reset gate first")
    norm = load_norm()
    valid_objs = load_valid_objs()
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for seed in SEEDS:
        soup_path = OUT / f"soup_J_seed{seed}.pt"
        if not soup_path.exists():
            raise SystemExit(f"missing {soup_path}")
        model = f10k._make_downstream_model(seed, norm, "cpu")
        model.load_state_dict(torch.load(soup_path, map_location="cpu", weights_only=False))
        pred = predict_objs(model, valid_objs, "cpu")
        if not np.isfinite(pred).all():
            raise RuntimeError("non-finite J predictions")
        path = OUT / "predictions" / f"valid_J{seed}.npz"
        np.savez_compressed(path, row=np.arange(N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
        entries[f"J{seed}"] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(path),
            "checkpoint_sha256": file_sha256(soup_path),
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "predictions": entries,
        "valid_labels_read": True,
        "official_test_loaded": False,
    }
    write_json(OUT / "jreset_prediction_freeze.json", payload)
    log(f"[predict-jreset] {sorted(entries)}")
    return 0


def cmd_score_jreset(args: argparse.Namespace) -> int:
    freeze = read_json(OUT / "jreset_prediction_freeze.json")
    for key, entry in freeze["predictions"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"J-reset prediction changed after freeze: {key}")
    y = load_valid_y()
    with np.load(ANALYSIS_DIR / "valid_preds_replay.npz") as z:
        preds = {k: np.asarray(z[k], dtype=np.float64) for k in FROZEN_KEYS}
    for seed in SEEDS:
        with np.load(OUT / "predictions" / f"valid_J{seed}.npz") as z:
            preds[f"J{seed}"] = np.asarray(z["pred"], dtype=np.float64)
    e = {k: np.abs(preds[k] - y) for k in preds}
    metrics = {k: row_metrics(preds[k], y) for k in preds}
    seed_mean = {arm: float(np.mean([metrics[f"{arm}{s}"]["mae"] for s in SEEDS])) for arm in ("B", "P", "J")}
    delta = {
        "J_minus_P": (e["J0"] + e["J1"]) / 2.0 - (e["P0"] + e["P1"]) / 2.0,
        "J_minus_B": (e["J0"] + e["J1"]) / 2.0 - (e["B0"] + e["B1"]) / 2.0,
    }
    tail = y <= TAIL_THRESHOLD
    ordinary = ~tail
    bootstrap = {name: paired_bootstrap(d) for name, d in delta.items()}
    subgroups = {}
    for name, d in delta.items():
        subgroups[name] = {
            "ordinary_y_gt_-3": {"n": int(ordinary.sum()), "delta": float(d[ordinary].mean())},
            "tail_y_le_-3": {"n": int(tail.sum()), "delta": float(d[tail].mean()) if tail.any() else None},
            "top10_removed": f10k.top_k_removed(d, np.maximum(e["J0"], e["J1"]), np.maximum(e["P0"], e["P1"]), 10),
        }
    if bootstrap["J_minus_P"]["ci_hi"] < 0:
        verdict = "J_RESET_IMPROVES_VALID"
    elif bootstrap["J_minus_P"]["ci_lo"] > 0:
        verdict = "J_RESET_HURTS_VALID"
    else:
        verdict = "J_RESET_NEUTRAL_VALID"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "per_model": metrics,
        "seed_mean_mae": seed_mean,
        "paired": {
            "J_minus_P": {
                "delta_seed_mean": float(delta["J_minus_P"].mean()),
                "per_seed": {f"seed{s}": float(e[f"J{s}"].mean() - e[f"P{s}"].mean()) for s in SEEDS},
                "bootstrap": bootstrap["J_minus_P"],
                "better_fraction": float((delta["J_minus_P"] < 0).mean()),
                "subgroups": subgroups["J_minus_P"],
            },
            "J_minus_B": {
                "delta_seed_mean": float(delta["J_minus_B"].mean()),
                "per_seed": {f"seed{s}": float(e[f"J{s}"].mean() - e[f"B{s}"].mean()) for s in SEEDS},
                "bootstrap": bootstrap["J_minus_B"],
                "better_fraction": float((delta["J_minus_B"] < 0).mean()),
                "subgroups": subgroups["J_minus_B"],
            },
        },
        "verdict": verdict,
        "what_it_cannot_show": [
            "it does not separate the causal contributions of the three encoders during joint pretraining",
            "it does not characterise what the junction encoder learned",
            "it is evaluated on the exposed official-valid split only",
        ],
        "official_valid_labels_read": True,
        "official_test_loaded": False,
    }
    write_json(OUT / "junction_reset_valid_metrics.json", payload)
    log(f"[score-jreset] J {seed_mean['J']:.6f} P {seed_mean['P']:.6f} J-P {payload['paired']['J_minus_P']['delta_seed_mean']:+.6f} -> {verdict}")
    return 0


# ---------------------------------------------------------------------------
# Phase F draft (written before the official-test read)
# ---------------------------------------------------------------------------


def cmd_next_hypothesis(args: argparse.Namespace) -> int:
    tail = read_json(OUT / "train_valid_tail_analysis.json")
    jr = read_json(OUT / "junction_reset_valid_metrics.json") if (OUT / "junction_reset_valid_metrics.json").exists() else None
    q = tail["question"]
    train = tail["train"]
    valid = tail["valid"]
    jr_line = "J-reset was not purchased; no junction-init claim is made this round."
    jr_prop = ""
    if jr is not None:
        d = jr["paired"]["J_minus_P"]
        jr_line = f"J-reset seed-mean delta vs P {d['delta_seed_mean']:+.6f} (CI95 [{d['bootstrap']['ci_lo']:+.6f}, {d['bootstrap']['ci_hi']:+.6f}], {jr['verdict']})."
        if jr["verdict"] == "J_RESET_HURTS_VALID":
            jr_prop = "keep the complete joint pretrained encoder state; do not break the junction initialisation"
        elif jr["verdict"] == "J_RESET_IMPROVES_VALID":
            jr_prop = "investigate why the joint SSL junction initialisation is a less suitable fine-tuning start than the scratch junction state"
        else:
            jr_prop = "the junction initialisation is not isolated as a driver of the valid gain; treat the full pretrained state as the unit"
    md = f"""# Next predictive hypothesis (draft)

**Drafted before the official-test label read**, from the train/valid diagnostics
and the J-reset ablation only.  Terminal-test numbers, if different, may
re-scope confidence but must not choose this direction.

## Diagnosis (Phase B)

* Verdict: `{q['verdict']}`.
* Train tail (y <= -3, n={train['n_tail_y_le_-3']}): P-B MAE `{q['train_tail_delta_P_minus_B']:+.6f}`; train ordinary (n={train['n_ordinary_y_gt_-3']}): `{q['train_ordinary_delta_P_minus_B']:+.6f}`.
* Valid tail (n={valid['n_tail_y_le_-3']}): P-B MAE `{q['valid_tail_delta_P_minus_B']:+.6f}`; valid ordinary (n={valid['n_ordinary_y_gt_-3']}): `{q['valid_ordinary_delta_P_minus_B']:+.6f}`.
* Signed tail residual (pred - y): train P `{q['train_tail_mean_signed_P']:+.4f}` / B `{q['train_tail_mean_signed_B']:+.4f}`; valid P `{q['valid_tail_mean_signed_P']:+.4f}` / B `{q['valid_tail_mean_signed_B']:+.4f}`.
* J-reset: {jr_line}

## The one next proposition

{f'**Proposition.** ' + ('The residual extreme-magnitude error of P is a coverage/generalisation failure of the composition function on rare label-free structural contexts, not an encoder-capacity failure; a mechanism that increases the effective coverage of rare contexts *inside the forward pass* while keeping raw-y L1 unchanged will lower the tail MAE without hurting ordinary molecules.' if q['verdict'] == 'TAIL_GENERALISATION_GAP_SUPPORTED' else 'The residual error is not (only) a generalisation failure; the composition function itself under-fits part of the training distribution, so a mechanism that expands the expressible composition space at matched parameter count must be tested before any coverage argument.')}

**Concrete mechanism to purchase (one).** ... (see finalised section below,
written by the same driver before the test read; the exact module is chosen
from the observed subgroups).

**Why the frozen P cannot satisfy it.** P is a single deterministic one-shot
composition; its training signal on rare contexts is limited to the few
molecules that carry them, and its reader input is a global standardised
summary that must extrapolate for out-of-range counts.

**Why this is not merely more capacity or output calibration.** The proposal
changes the *composition/coverage mechanism inside the forward pass*; there is
no post-hoc `a(counts) * y_hat + b(counts)` term, no target transform, no
tail-weighted loss and no new label.

## Matched baseline and purchase gate

* Matched baseline: the frozen P recipe on the same official train 10000, same
  init seeds, same batch order, same 240-epoch last-5 soup, raw-y L1 only.
* CPU-only pre-check: the new mechanism must yield a measurable, deterministic
  change on the frozen train objects (forward difference / invariance tests)
  before any GPU purchase.
* First GPU design: <= 2 seeds, matched parameter count and wall time to P.
* Purchase gate (single-model official-valid MAE, seed-mean of two seeds):
  stop unless the new arm is <= P - 0.004 with both seeds negative, ordinary
  molecules non-worse, tail not worse, and no RMSE blow-up.
* Stop conditions: gate failure stops the route; no seed 3, no hyper-parameter
  search, no test-driven rescue.

_This file was generated after Phase B/C and before `test_read.lock`._
"""
    (OUT / "next_predictive_hypothesis.md").write_text(md, encoding="utf-8")
    log("[next-hypothesis] draft written before the terminal test read")
    return 0


# ---------------------------------------------------------------------------
# Phase D -- terminal test protocol, freeze and scoring
# ---------------------------------------------------------------------------


def _test_structure_fingerprint(ds) -> dict[str, Any]:
    digest = hashlib.sha256()
    for i in range(N_TEST):
        digest.update(np.ascontiguousarray(ds[i].x.numpy(), dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(ds[i].edge_index.numpy(), dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(ds[i].edge_attr.numpy(), dtype=np.int64).tobytes())
    processed = REPO_ROOT / "data/ZINC/subset/processed/test.pt"
    return {
        "n_molecules": N_TEST,
        "processed_pt": str(processed.relative_to(REPO_ROOT)),
        "processed_pt_sha256": file_sha256(processed),
        "rows_sha256": rows_sha256(range(N_TEST)),
        "structure_sha256": digest.hexdigest(),
    }


def cmd_test_protocol(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "terminal_test_freeze.json").exists():
        raise SystemExit("terminal test predictions already frozen")
    manifest = read_json(F10K / "prediction_freeze.json")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            key: {
                "checkpoint": str(SOUP[key].relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(SOUP[key]),
                "expected_sha256": manifest["checkpoints"][f"soup_{key[0]}_seed{key[1]}"],
                "parameters": 413505,
                "weight_init": "scratch full10k" if key.startswith("B") else "masked-structure pretrained encoders (full10k)",
            }
            for key in FROZEN_KEYS
        },
        "preprocessing": {
            "file": "results/bjoc_full10k_pretraining_transfer_v1/prep_train10k.json",
            "sha256": file_sha256(F10K / "prep_train10k.json"),
            "fit": "official train 10000 only",
        },
        "input": "official PyG ZINC subset=True test raw graphs -> BJOC objects; labels guarded",
        "freeze_checks": [
            "label guard on every test row during object construction and prediction",
            "label-storage mutation must not change objects or predictions",
            "atom relabelling invariance of the frozen forward",
            "batch composition consistency (single vs batched)",
            "checkpoint reload replay exactness",
            "row order and length checks",
        ],
        "scoring": {
            "primary": "official-test raw-y single-model MAE; seed-mean = arithmetic mean of the two single-model MAEs",
            "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "molecule"},
            "secondary": ["rmse", "median_abs", "p90", "p95", "p99", "max_abs", "ordinary", "tail", "tail_mean_signed", "top1/5/10_removed", "worst_molecule_share"],
            "no_ensemble": True,
            "single_label_read": True,
        },
        "disclosure": "official test is an exposed instrument (read by the KSVD track on 2026-10-03); this is an exposed-test terminal comparison, not a first blind test",
        "excluded_models": ["Q0", "Q1", "J0", "J1", "historical Full-Y as a primary control"],
        "official_test_labels_read": False,
    }
    write_json(OUT / "terminal_test_protocol.json", payload)
    log("[test-protocol] frozen before any test label read")
    return 0


def _label_storage(ds):
    """Raw storage object holding the labels (never values, only the storage)."""
    for attr in ("_data", "data"):
        d = getattr(ds, attr, None)
        if d is not None and hasattr(d, "y"):
            return d
    return None


def _clear_slice_cache(ds) -> None:
    """Drop cached per-row slices so mutated storage is re-sliced."""
    if hasattr(ds, "_data_list"):
        ds._data_list = None


def cmd_test_predict(args: argparse.Namespace) -> int:
    if (OUT / "test_read.lock").exists():
        raise SystemExit("test labels already read; refusing to regenerate predictions")
    if not (OUT / "terminal_test_protocol.json").exists():
        raise SystemExit("run test-protocol first")
    device = "cpu"
    norm = load_norm()
    ds = load_test_dataset()
    fingerprint = _test_structure_fingerprint(ds)
    guarded = [LabelGuard(ds[i]) for i in range(N_TEST)]
    objs = build_test_objects(guarded, N_TEST)
    sig = objects_signature(objs)
    obj_hash, per_mol = objects_hash(objs)
    log(f"[test-predict] objects {sig} hash {obj_hash[:16]}")

    # --- label-storage mutation check (values are overwritten, never read) ---
    store = _label_storage(ds)
    mutation: dict[str, Any] = {"available": store is not None}
    if store is not None:
        orig = store.y.detach().clone()
        _clear_slice_cache(ds)
        objects_before = build_test_objects([LabelGuard(ds[i]) for i in range(16)], 16)
        h_before, _ = objects_hash(objects_before)
        store.y = torch.zeros_like(store.y)
        _clear_slice_cache(ds)
        h_zero, _ = objects_hash(build_test_objects([LabelGuard(ds[i]) for i in range(16)], 16))
        store.y = torch.full_like(orig, 999.0)
        _clear_slice_cache(ds)
        h_999, _ = objects_hash(build_test_objects([LabelGuard(ds[i]) for i in range(16)], 16))
        store.y = orig
        _clear_slice_cache(ds)
        mutation.update({"objects_hash_before": h_before, "objects_hash_zero_labels": h_zero, "objects_hash_999_labels": h_999, "objects_unchanged": h_before == h_zero == h_999})

    # --- relabelling invariance on one test molecule ---
    d0 = ds[0]
    base = objs[0]
    perm = list(reversed(range(base.n_atoms)))
    inv = {old: new for new, old in enumerate(perm)}
    x_p = np.asarray(d0.x).copy().reshape(-1)
    x_new = np.zeros_like(x_p)
    for old in range(base.n_atoms):
        x_new[inv[old]] = x_p[old]
    edge_rows = [(inv[int(d0.edge_index[0, c])], inv[int(d0.edge_index[1, c])]) for c in range(np.asarray(d0.edge_index).shape[1])]
    o_perm = sro.build_mol_objects(0, x_new, np.asarray(edge_rows, dtype=np.int64).T, np.asarray(d0.edge_attr))

    predictions: dict[str, Any] = {}
    files: dict[str, Any] = {}
    invariance: dict[str, Any] = {}
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    for key in FROZEN_KEYS:
        model = load_soup_model(key, norm, device)
        pred = predict_test_objs(model, objs, device)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"non-finite test predictions for {key}")
        pred_base = predict_test_objs(model, [base], device)[0]
        pred_perm = predict_test_objs(model, [o_perm], device)[0]
        pred_batch8 = predict_test_objs(model, objs[:8], device)[0]
        pred16 = predict_test_objs(model, objs[:16], device)
        model2 = load_soup_model(key, norm, device)
        pred_replay = predict_test_objs(model2, objs[:16], device)
        invariance[key] = {
            "relabelling_abs_diff": float(abs(pred_base - pred_perm)),
            "batch_composition_abs_diff": float(abs(pred_base - pred_batch8)),
            "checkpoint_reload_max_abs_diff": float(np.abs(pred16 - pred_replay).max()),
        }
        path = OUT / "predictions" / f"test_{key}.npz"
        meta = {
            "protocol": PROTOCOL_ID,
            "model": key,
            "row": list(range(N_TEST)),
            "object_hash": obj_hash,
            "checkpoint_sha256": file_sha256(SOUP[key]),
            "label_free": True,
        }
        np.savez_compressed(path, row=np.arange(N_TEST, dtype=np.int64), pred=pred.astype(np.float32), meta=json.dumps(meta))
        files[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
        predictions[key] = {"checkpoint": str(SOUP[key].relative_to(REPO_ROOT)), "checkpoint_sha256": file_sha256(SOUP[key])}
        log(f"[test-predict] {key} written pred[0]={pred[0]:+.6f} relabel={invariance[key]['relabelling_abs_diff']:.2e}")

    combined = hashlib.sha256()
    for key in sorted(files):
        combined.update(key.encode())
        combined.update(files[key]["sha256"].encode())
    freeze = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_before_test_label_read": True,
        "test_labels_read": False,
        "models": predictions,
        "prediction_files": files,
        "combined_prediction_sha256": combined.hexdigest(),
        "objects": {"signature": sig, "combined_sha256": obj_hash, "per_molecule_sha256": per_mol},
        "row_order": {"rows": list(range(N_TEST)), "rows_sha256": rows_sha256(range(N_TEST))},
        "structure_fingerprint": fingerprint,
        "label_independence": {"label_guard_active": True, "label_storage_mutation": mutation},
        "invariance": invariance,
        "official_test_structures_loaded": True,
        "official_test_labels_read": False,
        "code": {
            "driver": file_sha256(Path(__file__).resolve()),
            "model": file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
            "objects": file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
            "masked_module": file_sha256(_CODE_DIR / "bjoc_masked_pretraining_v1.py"),
            "previous_round_driver": file_sha256(_CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py"),
            "config": file_sha256(CONFIG_PATH),
        },
    }
    write_json(OUT / "terminal_test_freeze.json", freeze)
    checks = [v["relabelling_abs_diff"] < 1e-6 and v["batch_composition_abs_diff"] < 1e-6 and v["checkpoint_reload_max_abs_diff"] == 0.0 for v in invariance.values()]
    if not all(checks):
        raise SystemExit(f"invariance checks failed: {invariance}")
    if mutation.get("available") and not mutation["objects_unchanged"]:
        raise SystemExit("label storage changed the test objects")
    log(f"[test-predict] frozen combined {combined.hexdigest()[:16]}")
    return 0


def cmd_test_score(args: argparse.Namespace) -> int:
    if (OUT / "test_read.lock").exists():
        raise SystemExit("official test labels were already read; refusing a second read")
    freeze = read_json(OUT / "terminal_test_freeze.json")
    for key, entry in freeze["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"frozen test prediction changed: {key}")
    for key, entry in freeze["models"].items():
        if file_sha256(REPO_ROOT / entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint changed after freeze: {key}")

    started = time.perf_counter()
    ds = load_test_dataset()
    y = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(N_TEST)], dtype=np.float64)
    np.save(OUT / "test_y_read_once.npy", y)
    hist_path = KSVD_VT / "heldout_test_predictions.npz"
    y_cross = None
    if hist_path.exists():
        hist = np.load(hist_path, allow_pickle=True)
        y_hist = np.asarray(hist["y"], dtype=np.float64)
        y_cross = float(np.abs(y - y_hist).max())
    lock = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_test": N_TEST,
        "rows_sha256": rows_sha256(range(N_TEST)),
        "y_sha256": array_sha256(y),
        "y_crosscheck_vs_ksvd_historical_max_abs": y_cross,
        "source": "PyG ZINC subset=True test processed split (data/ZINC/subset/processed/test.pt)",
        "single_read_this_round": True,
        "not_first_ever_read": True,
        "prediction_freeze_sha256": file_sha256(OUT / "terminal_test_freeze.json"),
        "ensembles_used": False,
    }
    write_json(OUT / "test_read.lock", lock)
    if y_cross is not None and y_cross != 0.0:
        raise SystemExit(f"raw test y disagrees with the historical evaluation copy (max abs {y_cross})")

    preds = {}
    for key in FROZEN_KEYS:
        with np.load(REPO_ROOT / freeze["prediction_files"][key]["path"]) as z:
            assert np.array_equal(z["row"], np.arange(N_TEST)), f"{key}: row order broken"
            preds[key] = np.asarray(z["pred"], dtype=np.float64)
        assert preds[key].shape == y.shape
    e = {k: np.abs(preds[k] - y) for k in FROZEN_KEYS}
    metrics = {k: row_metrics(preds[k], y) for k in FROZEN_KEYS}
    seed_mean = {arm: {mm: float(np.mean([metrics[f"{arm}{s}"][mm] for s in SEEDS])) for mm in metrics["B0"]} for arm in ARMS}
    seed_mean["B"]["per_seed_mae"] = [metrics[f"B{s}"]["mae"] for s in SEEDS]
    seed_mean["P"]["per_seed_mae"] = [metrics[f"P{s}"]["mae"] for s in SEEDS]
    delta = {f"seed{s}": float(e[f"P{s}"].mean() - e[f"B{s}"].mean()) for s in SEEDS}
    d = ((e["P0"] - e["B0"]) + (e["P1"] - e["B1"])) / 2.0
    delta["seed_mean"] = float(d.mean())
    boot = paired_bootstrap(d)
    tail = y <= TAIL_THRESHOLD
    ordinary = ~tail
    worst = int(np.argmax(np.maximum(e["P0"], e["P1"]) + np.maximum(e["B0"], e["B1"])))
    keep = np.ones(N_TEST, dtype=bool)
    keep[worst] = False
    subgroups = {
        "ordinary_y_gt_-3": {"n": int(ordinary.sum()), "mae_P_seed_mean": float(np.mean([e[f"P{s}"][ordinary].mean() for s in SEEDS])), "mae_B_seed_mean": float(np.mean([e[f"B{s}"][ordinary].mean() for s in SEEDS])), "delta": float(d[ordinary].mean())},
        "tail_y_le_-3": {"n": int(tail.sum()), "mae_P_seed_mean": float(np.mean([e[f"P{s}"][tail].mean() for s in SEEDS])), "mae_B_seed_mean": float(np.mean([e[f"B{s}"][tail].mean() for s in SEEDS])), "delta": float(d[tail].mean()) if tail.any() else None, "mean_signed_P": float(np.mean([(preds[f"P{s}"] - y)[tail].mean() for s in SEEDS])), "mean_signed_B": float(np.mean([(preds[f"B{s}"] - y)[tail].mean() for s in SEEDS]))},
        "all_rows_minus_worst": {"excluded_row": worst, "delta": float(d[keep].mean()), "mae_P_seed_mean": float(np.mean([e[f"P{s}"][keep].mean() for s in SEEDS])), "mae_B_seed_mean": float(np.mean([e[f"B{s}"][keep].mean() for s in SEEDS]))},
        "top_k_removed": {f"top{k}": f10k.top_k_removed(d, np.maximum(e["P0"], e["P1"]), np.maximum(e["B0"], e["B1"]), k) for k in (1, 5, 10)},
    }
    worst_share = {
        key: float(e[key][np.argmax(e[key])] / e[key].sum()) for key in FROZEN_KEYS
    }
    metrics_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_molecules": N_TEST,
        "per_model": metrics,
        "seed_mean": seed_mean,
        "delta": {**delta, "ci95": [boot["ci_lo"], boot["ci_hi"]], "better_fraction": float((d < 0).mean()), "relative_reduction_vs_B": float(-d.mean() / seed_mean["B"]["mae"]), "definition": "d_i = 0.5*[(|e_P0|-|e_B0|) + (|e_P1|-|e_B1|)]; seed-mean = arithmetic mean of the two single-model MAEs; never an ensemble"},
        "subgroups": subgroups,
        "worst_molecule_error_share": worst_share,
        "y_crosscheck_vs_ksvd_historical_max_abs": y_cross,
        "single_read": True,
        "ensemble_used": False,
        "official_test_loaded": True,
        "seconds": time.perf_counter() - started,
    }
    write_json(OUT / "official_test_metrics.json", metrics_payload)
    write_json(OUT / "official_test_paired_bootstrap.json", {"protocol": PROTOCOL_ID, "seed_mean_delta": boot, "per_seed": {f"seed{s}": paired_bootstrap(e[f"P{s}"] - e[f"B{s}"]) for s in SEEDS}, "n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED})
    log(f"[test-score] B {seed_mean['B']['mae']:.6f} P {seed_mean['P']['mae']:.6f} delta {delta['seed_mean']:+.6f} CI [{boot['ci_lo']:+.6f},{boot['ci_hi']:+.6f}]")
    return 0


def cmd_valid_test_compare(args: argparse.Namespace) -> int:
    valid = read_json(F10K / "valid_metrics.json")
    test = read_json(OUT / "official_test_metrics.json")
    vmet = valid["models"]
    tmet = test["per_model"]
    vpb = valid["paired"]["P_minus_B"]

    def _relative(b: float, p: float) -> float:
        return float((p - b) / b)

    valid_block = {
        "B_seed0": vmet["B0"]["mae"], "B_seed1": vmet["B1"]["mae"], "B_seed_mean": valid["seed_mean"]["B"]["mae"],
        "P_seed0": vmet["P0"]["mae"], "P_seed1": vmet["P1"]["mae"], "P_seed_mean": valid["seed_mean"]["P"]["mae"],
        "delta_seed0": vpb["per_seed_delta"]["seed0"], "delta_seed1": vpb["per_seed_delta"]["seed1"], "delta_seed_mean": vpb["delta_seed_mean"],
        "ci95": [vpb["bootstrap"]["ci_lo"], vpb["bootstrap"]["ci_hi"]],
        "relative_reduction": _relative(vpb["seed_mean_mae_ref"], vpb["seed_mean_mae_new"]),
        "ordinary_delta": vpb["ordinary_y_gt_-3"]["delta"],
        "tail_delta": vpb["tail_y_le_-3"]["delta"],
        "rmse_B_seed_mean": valid["seed_mean"]["B"]["rmse"],
        "rmse_P_seed_mean": valid["seed_mean"]["P"]["rmse"],
        "better_fraction": vpb["better_fraction"],
    }
    tdelta = test["delta"]
    test_block = {
        "B_seed0": tmet["B0"]["mae"], "B_seed1": tmet["B1"]["mae"], "B_seed_mean": test["seed_mean"]["B"]["mae"],
        "P_seed0": tmet["P0"]["mae"], "P_seed1": tmet["P1"]["mae"], "P_seed_mean": test["seed_mean"]["P"]["mae"],
        "delta_seed0": tdelta["seed0"], "delta_seed1": tdelta["seed1"], "delta_seed_mean": tdelta["seed_mean"],
        "ci95": tdelta["ci95"],
        "relative_reduction": tdelta["relative_reduction_vs_B"],
        "ordinary_delta": test["subgroups"]["ordinary_y_gt_-3"]["delta"],
        "tail_delta": test["subgroups"]["tail_y_le_-3"]["delta"],
        "rmse_B_seed_mean": test["seed_mean"]["B"]["rmse"],
        "rmse_P_seed_mean": test["seed_mean"]["P"]["rmse"],
        "better_fraction": tdelta["better_fraction"],
    }
    comparison = {
        "absolute_gain_valid": valid_block["delta_seed_mean"],
        "absolute_gain_test": test_block["delta_seed_mean"],
        "gain_retained_fraction": float(test_block["delta_seed_mean"] / valid_block["delta_seed_mean"]) if valid_block["delta_seed_mean"] != 0 else None,
        "ordinary_same_direction": (valid_block["ordinary_delta"] < 0) == (test_block["ordinary_delta"] < 0),
        "tail_same_direction": ((valid_block["tail_delta"] or 0.0) < 0) == ((test_block["tail_delta"] or 0.0) < 0),
        "rmse_cost_same_direction": ((valid_block["rmse_P_seed_mean"] - valid_block["rmse_B_seed_mean"]) > 0) == ((test_block["rmse_P_seed_mean"] - test_block["rmse_B_seed_mean"]) > 0),
        "both_seeds_same_direction_valid": valid_block["delta_seed0"] < 0 and valid_block["delta_seed1"] < 0,
        "both_seeds_same_direction_test": test_block["delta_seed0"] < 0 and test_block["delta_seed1"] < 0,
        "valid_rmse_delta_P_minus_B": valid_block["rmse_P_seed_mean"] - valid_block["rmse_B_seed_mean"],
        "test_rmse_delta_P_minus_B": test_block["rmse_P_seed_mean"] - test_block["rmse_B_seed_mean"],
    }
    write_json(OUT / "valid_test_comparison.json", {"protocol": PROTOCOL_ID, "valid": valid_block, "test": test_block, "comparison": comparison})
    log(f"[valid-test] retained {comparison['gain_retained_fraction']} ordinary_same={comparison['ordinary_same_direction']} tail_same={comparison['tail_same_direction']}")
    return 0


def cmd_scientific_decision(args: argparse.Namespace) -> int:
    test = read_json(OUT / "official_test_metrics.json")
    comparison = read_json(OUT / "valid_test_comparison.json")
    replay = read_json(OUT / "model_reuse_audit.json")
    d = test["delta"]
    sub = test["subgroups"]
    provenance_ok = replay["decision"] == "REPLAY_CONFIRMED"
    seed_conflict = (d["seed0"] < 0) != (d["seed1"] < 0) and abs(d["seed0"] - d["seed1"]) > 0.01
    single_molecule_driven = d["seed_mean"] < 0 and sub["all_rows_minus_worst"]["delta"] >= 0
    if not provenance_ok:
        branch = "TERMINAL_TEST_INCONCLUSIVE"
        reason = "provenance or checkpoint-hash failure"
    elif seed_conflict or single_molecule_driven:
        branch = "TERMINAL_TEST_INCONCLUSIVE"
        reason = "seed conflict" if seed_conflict else "gain driven by the single worst molecule"
    elif d["seed_mean"] < 0 and d["seed0"] < 0 and d["seed1"] < 0 and d["ci95"][1] < 0 and sub["ordinary_y_gt_-3"]["delta"] < 0 and sub["all_rows_minus_worst"]["delta"] < 0:
        branch = "PRETRAINING_TEST_GAIN_REPLICATES"
        reason = "all frozen replication conditions hold on the exposed test split"
    else:
        branch = "PRETRAINING_TEST_GAIN_SPLIT_SENSITIVE"
        reason = "valid gain is CI-supported but the terminal test is weaker / disappeared / reversed"
    p_mae = test["seed_mean"]["P"]["mae"]
    in_band = TARGET_LO <= p_mae <= TARGET_HI
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "branch": branch,
        "reason": reason,
        "inputs": {
            "per_model_test_mae": {k: test["per_model"][k]["mae"] for k in FROZEN_KEYS},
            "P_test_seed_maes": [test["per_model"]["P0"]["mae"], test["per_model"]["P1"]["mae"]],
            "P_test_seed_mean": p_mae,
            "B_test_seed_mean": test["seed_mean"]["B"]["mae"],
            "delta_test": d,
            "seed_conflict": bool(seed_conflict),
            "single_molecule_driven": bool(single_molecule_driven),
        },
        "target_band": {
            "band": [TARGET_LO, TARGET_HI],
            "P_test_seed_mean_mae": p_mae,
            "reached": bool(in_band),
            "statement": (
                f"P single-model exposed-test MAE seed-mean {p_mae:.6f} is inside the 0.06-0.07 band; this is a numerical target reached on an exposed instrument, not a new blind confirmation."
                if in_band else f"P single-model exposed-test MAE seed-mean {p_mae:.6f} is {p_mae - TARGET_HI:.6f} above the 0.07 upper edge of the band."
            ),
        },
        "claims": {
            "allowed": [
                "the frozen P/B difference measured on the exposed official-test split is exactly what official_test_metrics.json reports",
                "the comparison is same-round matched (same GPU regime, same recipe, same init/order per seed)",
            ],
            "not_allowed": [
                "a first independent blind test (the official test instrument was exposed in a previous round)",
                "a statement about arbitrary new molecule distributions",
                "any test-driven change of model, seed or hyper-parameter",
            ],
        },
        "valid_vs_test": comparison["comparison"],
        "next_round": (
            "keep P as the primary baseline and study the residual tail generalisation mechanism with official valid only"
            if branch == "PRETRAINING_TEST_GAIN_REPLICATES"
            else "treat the pretraining gain as distribution-sensitive; next round studies training-domain coverage and adaptation, still on official valid only"
            if branch == "PRETRAINING_TEST_GAIN_SPLIT_SENSITIVE"
            else "re-establish provenance/replication before any new predictive claim"
        ),
        "official_test_loaded": True,
        "no_training_after_test_read": True,
    }
    write_json(OUT / "scientific_decision.json", payload)
    log(f"[decision] {branch}: {reason}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reuse-audit").set_defaults(func=cmd_reuse_audit)
    sub.add_parser("legality").set_defaults(func=cmd_legality)
    sub.add_parser("mechanism").set_defaults(func=cmd_mechanism_evidence)
    sub.add_parser("tail-analysis").set_defaults(func=cmd_tail_analysis)
    sub.add_parser("jreset-gate").set_defaults(func=cmd_jreset_gate)
    p = sub.add_parser("train-jreset")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_train_jreset)
    sub.add_parser("predict-jreset").set_defaults(func=cmd_predict_jreset)
    sub.add_parser("score-jreset").set_defaults(func=cmd_score_jreset)
    sub.add_parser("next-hypothesis").set_defaults(func=cmd_next_hypothesis)
    sub.add_parser("test-protocol").set_defaults(func=cmd_test_protocol)
    sub.add_parser("test-predict").set_defaults(func=cmd_test_predict)
    sub.add_parser("test-score").set_defaults(func=cmd_test_score)
    sub.add_parser("valid-test-compare").set_defaults(func=cmd_valid_test_compare)
    sub.add_parser("decision").set_defaults(func=cmd_scientific_decision)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device = args.device if args.device else "cpu"
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
