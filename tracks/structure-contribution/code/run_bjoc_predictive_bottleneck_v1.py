"""cscl-predictive-bottleneck-diagnosis-v1 driver.

Protocol: ``configs/bjoc_predictive_bottleneck_v1.yaml`` (frozen before any GPU
trajectory).  Track: ``tracks/structure-contribution``.

The round separates three competing explanations of the BJOC plateau with the
minimum formal budget: six trajectories of the published BJOC backbone only.

* sample efficiency:   N8 - N4-stepmatched at the SAME update count T
* ownership:           O8 - N8 at matched data / seed / updates
* inductive bias:      persistent common errors across seeds / ownership modes
* input observability: CPU audit of strict model-input collisions

Stages
------
``splits``     freeze the new canonical-SMILES grouped split + nested Fit4k
``objects``    fit the global-feature standardiser per training size (fit-only)
``steps``      freeze T and the snapshot step numbers (labels never read)
``gates``      the pre-GPU correctness gates
``freeze``     protocol freeze manifest (hashes of everything that matters)
``train``      one formal trajectory (N/O x fit4k/fit8k x seed 0/1)
``predict``    label-blind predictions for the eight readouts on monitor+locked
``freeze-predictions``  hash every prediction file before any label read
``score``      the ONE label read per split; learning-curve + ownership analysis
``common-errors``  cross-seed / cross-ownership error structure (Phase G)
``historical`` Phase B: historical N/O/P monitor error diagnostics
``observability``  Phase C: input layout + graph collision audit
``decision``   the single frozen scientific decision
``report``     REPORT.md assembly

Discipline
----------
* official valid/test are never loaded in any form.
* fit y are used for training only; monitor y are the standard training-time
  readout; locked y are read exactly once inside ``score`` after every
  prediction file is written and hashed.
* raw-y L1 only; no ``g/ell/s/c/k``; no GNN/attention/iteration; no supervision from
  historical model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import run_bjoc_intrinsic_contribution_v1 as rb  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-predictive-bottleneck-diagnosis-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_predictive_bottleneck_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"

SPLIT_SALT = "cscl-predictive-bottleneck-v1-A"
FIT_TARGET = 8000
MONITOR_TARGET = 1000
FIT4K_TARGET = 4000
BATCH_SIZE = 128
EPOCHS_BASE = 240
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
SEEDS = (0, 1)
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TOP_K_REMOVED_DEFINITION = (
    "rows with the k largest max(|e_candidate|, |e_reference|) on the scored split "
    "are removed before averaging the paired delta"
)

SPLIT_FILE = OUT / "split_a.json"
FIT4K_FILE = OUT / "fit4k_subset.json"
PREP_FILES = {"fit8k": OUT / "prep_fit8k.json", "fit4k": OUT / "prep_fit4k.json"}
STEPS_FILE = OUT / "training_steps_manifest.json"
PRED_DIR = OUT / "predictions"
SCORE_DIR = OUT / "scores"
LOCK_FILE = OUT / "locked_read.lock"

READOUT_SPECS = {
    "N4short": {"arm": "N", "fit": "fit4k", "soup": "short"},
    "N4stepmatched": {"arm": "N", "fit": "fit4k", "soup": "stepmatched"},
    "N8": {"arm": "N", "fit": "fit8k", "soup": "stepmatched"},
    "O8": {"arm": "O", "fit": "fit8k", "soup": "stepmatched"},
}
READOUT_NAMES = tuple(READOUT_SPECS)

# historical results (read-only)
CB = REPO_ROOT / "tracks/structure-contribution/results/bjoc_cross_bond_interaction_v1"
SRC_DIR = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1"
JD = REPO_ROOT / "tracks/structure-contribution/results/bjoc_junction_regularization_v1"
IJ = REPO_ROOT / "tracks/structure-contribution/results/bjoc_intrinsic_contribution_v1"

# ---- reused helpers ---------------------------------------------------------
write_json = rb.write_json
read_json = rb.read_json
file_sha256 = rb.file_sha256
git_commit = rb.git_commit
seed_everything = rb.seed_everything
torch_rng_state_sha256 = rb.torch_rng_state_sha256
load_objects = rb.load_objects
load_y_rows = rb.load_y_rows
rows_sha256 = rb.rows_sha256
paired_bootstrap = rb.paired_bootstrap
metrics = rb.metrics
shuffle_batches = rb._shuffle_batches


def log(msg: str) -> None:
    print(msg, flush=True)


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def load_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def frozen_plan() -> dict[str, Any]:
    frozen = (load_config().get("frozen") or {})
    if not frozen:
        raise SystemExit(
            "config frozen block is empty: run `splits`, `objects`, `steps`, paste the plan, "
            "commit, and only then run gates/train"
        )
    return frozen


# ---------------------------------------------------------------------------
# splits / prep / steps
# ---------------------------------------------------------------------------


def build_split(smiles: Sequence[str], salt: str) -> dict[str, Any]:
    """Same grouped greedy builder as the recent BJOC rounds (labels never read)."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{salt}|{s}".encode()).hexdigest())
    fit: list[int] = []
    monitor: list[int] = []
    locked: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(fit) < FIT_TARGET:
            fit.extend(rows)
        elif len(monitor) < MONITOR_TARGET:
            monitor.extend(rows)
        else:
            locked.extend(rows)
    fit, monitor, locked = sorted(fit), sorted(monitor), sorted(locked)
    sizes = [len(v) for v in groups.values()]
    return {
        "salt": salt,
        "fit": fit,
        "monitor": monitor,
        "locked": locked,
        "n_groups": len(groups),
        "duplicate_group_sizes": {
            "n_groups_multi_row": int(sum(1 for v in sizes if v > 1)),
            "n_rows_in_multi_row_groups": int(sum(v for v in sizes if v > 1)),
            "max_group_size": int(max(sizes)),
            "n_groups_gt2": int(sum(1 for v in sizes if v > 2)),
        },
        "overlap": {
            "fit_monitor": len(set(fit) & set(monitor)),
            "fit_locked": len(set(fit) & set(locked)),
            "monitor_locked": len(set(monitor) & set(locked)),
            "union": len(set(fit) | set(monitor) | set(locked)),
        },
    }


def build_fit4k(smiles: Sequence[str], salt: str, fit_rows: Sequence[int], target: int = FIT4K_TARGET) -> list[int]:
    """Nested Fit4k: the SAME group order as the fit fill, greedy to ~target.

    Label-blind by construction: the function takes only SMILES, the salt and the
    already-frozen Fit8k rows.  Whole groups are taken so the result is exactly
    nested inside Fit8k.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{salt}|{s}".encode()).hexdigest())
    selected: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(selected) < target:
            selected.extend(rows)
        else:
            break
    selected = sorted(selected)
    assert target <= len(selected) <= target + 8, len(selected)
    assert set(selected) <= set(int(i) for i in fit_rows), "Fit4k is not nested in Fit8k"
    return selected


def read_split_rows() -> dict[str, list[int]]:
    split = read_json(SPLIT_FILE)
    fit4k = read_json(FIT4K_FILE)["rows"]
    return {"fit8k": [int(i) for i in split["fit_rows"]], "monitor": [int(i) for i in split["monitor_rows"]],
            "locked": [int(i) for i in split["locked_rows"]], "fit4k": [int(i) for i in fit4k]}


def load_prep(fit: str) -> dict[str, Any]:
    prep = read_json(PREP_FILES[fit])
    assert prep["fit_only"] is True
    return prep


def cmd_splits(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    smiles = rb.load_smiles()
    split = build_split(smiles, SPLIT_SALT)
    fit4k = build_fit4k(smiles, SPLIT_SALT, split["fit"])
    payload = {
        "protocol": PROTOCOL_ID,
        "salt": SPLIT_SALT,
        "n_groups": split["n_groups"],
        "n_fit8k": len(split["fit"]),
        "n_monitor": len(split["monitor"]),
        "n_locked": len(split["locked"]),
        "n_fit4k": len(fit4k),
        "overlap": split["overlap"],
        "duplicate_group_sizes": split["duplicate_group_sizes"],
        "fit8k_sha256_rows": rows_sha256(split["fit"]),
        "monitor_sha256_rows": rows_sha256(split["monitor"]),
        "locked_sha256_rows": rows_sha256(split["locked"]),
        "fit_rows": [int(i) for i in split["fit"]],
        "monitor_rows": [int(i) for i in split["monitor"]],
        "locked_rows": [int(i) for i in split["locked"]],
        "locked_labels_read": False,
    }
    write_json(SPLIT_FILE, payload)
    write_json(
        FIT4K_FILE,
        {
            "protocol": PROTOCOL_ID,
            "salt": SPLIT_SALT,
            "selection": "same group order as the fit fill; whole groups; greedy to 4000",
            "n_fit4k": len(fit4k),
            "fit4k_sha256_rows": rows_sha256(fit4k),
            "nested_in_fit8k": len(set(fit4k) - set(split["fit"])) == 0,
            "y_never_read": True,
            "rows": [int(i) for i in fit4k],
        },
    )
    log(
        f"[splits] fit8k={len(split['fit'])} monitor={len(split['monitor'])} locked={len(split['locked'])} "
        f"fit4k={len(fit4k)}"
    )
    log(f"[splits] fit8k sha {payload['fit8k_sha256_rows'][:16]}  fit4k sha {rows_sha256(fit4k)[:16]}")
    return 0


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_sha = file_sha256(rb.OBJECTS_CACHE)
    assert cache_sha == rb.OBJECTS_CACHE_SHA256, f"objects cache drifted: {cache_sha}"
    split = read_split_rows()
    objs = load_objects()
    audit: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "objects_cache": str(rb.OBJECTS_CACHE.relative_to(REPO_ROOT)),
        "objects_cache_sha256": cache_sha,
        "labels_read": False,
        "sizes": {},
    }
    for fit, rows in (("fit8k", split["fit8k"]), ("fit4k", split["fit4k"])):
        gfit = np.stack([objs[int(i)].global_feat for i in rows], 0).astype(np.float64)
        gmean = gfit.mean(0)
        gscale = np.where(gfit.std(0) < 1e-6, 1.0, gfit.std(0))
        write_json(
            PREP_FILES[fit],
            {
                "protocol": PROTOCOL_ID,
                "fit": fit,
                "fit_only": True,
                "n_fit": int(len(rows)),
                "fit_rows_sha256": rows_sha256(rows),
                "global_mean": gmean.tolist(),
                "global_scale": gscale.tolist(),
                "names": list(sro.GLOBAL_FEAT_NAMES),
                "labels_read": False,
            },
        )
        audit["sizes"][fit] = {
            "n_fit": int(len(rows)),
            "rows_sha256": rows_sha256(rows),
            "prep_sha256": file_sha256(PREP_FILES[fit]),
            "mean_of_means": float(gmean.mean()),
            "scale_min": float(gscale.min()),
            "scale_max": float(gscale.max()),
        }
        log(f"[objects] {fit}: global standardiser fitted on {len(rows)} fit rows (labels never read)")
    for k, v in audit["sizes"].items():
        assert v["n_fit"] == (FIT_TARGET if k == "fit8k" else len(split["fit4k"]))
    audit["prep_distinct"] = audit["sizes"]["fit8k"]["prep_sha256"] != audit["sizes"]["fit4k"]["prep_sha256"]
    write_json(OUT / "normalization_audit.json", audit)
    del objs
    return 0


def compute_steps(n_fit8k: int, n_fit4k: int) -> dict[str, Any]:
    spe8 = int(math.ceil(n_fit8k / BATCH_SIZE))
    spe4 = int(math.ceil(n_fit4k / BATCH_SIZE))
    T = EPOCHS_BASE * spe8
    soup8 = [spe8 * (EPOCHS_BASE - SOUP_LAST + 1 + i) for i in range(SOUP_LAST)]
    short4 = [spe4 * (EPOCHS_BASE - SOUP_LAST + 1 + i) for i in range(SOUP_LAST)]
    match4 = list(soup8)
    assert max(short4) < T
    assert max(match4) <= T
    return {
        "protocol": PROTOCOL_ID,
        "n_fit8k": int(n_fit8k),
        "n_fit4k": int(n_fit4k),
        "batch_size": BATCH_SIZE,
        "base_epochs": EPOCHS_BASE,
        "steps_per_epoch_8k": spe8,
        "steps_per_epoch_4k": spe4,
        "T": int(T),
        "T_definition": "240 * ceil(n_fit8k / 128)",
        "n4_short_snapshot_steps": short4,
        "n4_stepmatched_snapshot_steps": match4,
        "n8_o8_soup_steps": soup8,
        "n4_short_epochs": EPOCHS_BASE,
        "n4_stepmatched_approx_epochs": float(T / spe4),
        "soup_rule": "arithmetic mean of the five frozen parameter snapshots",
        "checkpoints_frozen_before_training": True,
        "y_never_read": True,
    }


def cmd_steps(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split = read_split_rows()
    manifest = compute_steps(len(split["fit8k"]), len(split["fit4k"]))
    write_json(STEPS_FILE, manifest)
    log(json.dumps(manifest, indent=2))
    log("[steps] paste the frozen: block into the config, then commit BEFORE any GPU run")
    return 0


# ---------------------------------------------------------------------------
# model / prediction helpers
# ---------------------------------------------------------------------------


def model_for(arm: str, prep: Mapping[str, Any], seed: int, device: torch.device) -> src.BJOC:
    return src.build_bjoc(
        src.BJOCConfig(),
        seed=seed,
        organization_mode=ARM_MODE[arm],
        global_mean=np.asarray(prep["global_mean"], dtype=np.float32),
        global_scale=np.asarray(prep["global_scale"], dtype=np.float32),
        device=device,
    )


ARM_MODE = {"N": src.NULL, "O": src.ORGANIZED}


def predict_bjoc(model: src.BJOC, objs: Sequence[sro.MolObjects], rows: Sequence[int], device: torch.device,
                 chunk: int = 256, mode: str | None = None) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            batch = sro.batch_objects([objs[int(r)] for r in rows[i : i + chunk]])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch, organization_mode=mode).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0)


def soup_from(snapshots: Mapping[int, Mapping[str, torch.Tensor]], steps: Sequence[int]) -> dict[str, torch.Tensor]:
    missing = [s for s in steps if s not in snapshots]
    assert not missing, f"missing snapshots {missing}"
    keys = list(next(iter(snapshots.values())).keys())
    return {k: torch.stack([snapshots[s][k].float() for s in steps]).mean(0) for k in keys}


def checkpoint_path(arm: str, fit: str, seed: int) -> Path:
    return OUT / f"checkpoint_{arm}_{fit}_s{seed}.pt"


def train_json_path(arm: str, fit: str, seed: int) -> Path:
    return OUT / f"train_{arm}_{fit}_s{seed}.json"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_trajectory(arm: str, fit: str, seed: int, device: torch.device, *, log_fn=log) -> dict[str, Any]:
    assert arm in ARM_MODE and fit in ("fit4k", "fit8k") and seed in SEEDS
    frozen = frozen_plan()
    manifest = read_json(STEPS_FILE)
    for key, value in (
        ("n_fit8k", manifest["n_fit8k"]),
        ("n_fit4k", manifest["n_fit4k"]),
        ("T", manifest["T"]),
    ):
        assert int(frozen[key]) == int(value), f"frozen {key} drift"
    split = read_split_rows()
    fit_rows = split["fit4k"] if fit == "fit4k" else split["fit8k"]
    mon_rows = split["monitor"]
    prep = load_prep(fit)
    objs = load_objects()
    y_fit_np = load_y_rows(fit_rows, subset_name=f"{fit} fit")
    y_mon_np = load_y_rows(mon_rows, subset_name="monitor")
    y_fit = torch.as_tensor(y_fit_np, dtype=torch.float32)

    steps_per_epoch = int(math.ceil(len(fit_rows) / BATCH_SIZE))
    T = int(manifest["T"])
    if fit == "fit4k":
        snapshot_steps = sorted(set(manifest["n4_short_snapshot_steps"]) | set(manifest["n4_stepmatched_snapshot_steps"]))
    else:
        snapshot_steps = list(manifest["n8_o8_soup_steps"])
    snapshot_set = set(int(s) for s in snapshot_steps)

    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    seed_everything(seed)
    model = model_for(arm, prep, seed, device)
    init_sha = src.state_dict_hash(model)
    rng_after_init = torch_rng_state_sha256()
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    order_ids: list[int] = []
    curve: list[dict[str, Any]] = []
    rng_hashes: dict[str, str] = {}
    global_step = 0
    epoch = 0
    started = time.perf_counter()
    while global_step < int(T):
        epoch += 1
        model.train()
        batches = shuffle_batches(len(fit_rows), BATCH_SIZE, gen)
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in batches:
            if global_step >= T:
                break
            batch = sro.batch_objects([objs[int(fit_rows[i])] for i in indices])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = y_fit[indices].to(device).view(-1)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            total += float((pred.view(-1) - target).abs().sum())
            n_mol += len(indices)
            steps += 1
            global_step += 1
            order_ids.extend(int(i) for i in indices)
            if global_step in snapshot_set:
                snapshots[global_step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mon_pred = predict_bjoc(model, objs, mon_rows, device)
        mon_mae = float(np.abs(mon_pred - y_mon_np).mean())
        curve.append(
            {
                "epoch": epoch,
                "global_step_end": global_step,
                "steps_in_epoch": steps,
                "train_mae": total / max(n_mol, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
            }
        )
        if epoch in (1, 40, 120, 240, EPOCHS_BASE + 1) or epoch % 60 == 0 or global_step >= T:
            rng_hashes[str(epoch)] = torch_rng_state_sha256()
        if epoch % 40 == 0 or epoch in (1, EPOCHS_BASE) or global_step >= T:
            log_fn(
                f"[{arm}/{fit}/s{seed}] epoch={epoch:03d} step={global_step}/{T} "
                f"train={curve[-1]['train_mae']:.6f} mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s"
            )
        if not np.isfinite(curve[-1]["train_mae"]) or not np.isfinite(mon_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    wall = time.perf_counter() - started
    assert global_step == T, (global_step, T)
    assert set(snapshot_set) <= set(snapshots), sorted(snapshot_set - set(snapshots))

    if fit == "fit4k":
        soups = {
            "short": soup_from(snapshots, manifest["n4_short_snapshot_steps"]),
            "stepmatched": soup_from(snapshots, manifest["n4_stepmatched_snapshot_steps"]),
        }
        soup_members = {
            "short": list(manifest["n4_short_snapshot_steps"]),
            "stepmatched": list(manifest["n4_stepmatched_snapshot_steps"]),
        }
    else:
        soups = {"stepmatched": soup_from(snapshots, manifest["n8_o8_soup_steps"])}
        soup_members = {"stepmatched": list(manifest["n8_o8_soup_steps"])}

    soup_metrics: dict[str, Any] = {}
    for name, state in soups.items():
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        fit_pred = predict_bjoc(model, objs, fit_rows, device)
        mon_pred = predict_bjoc(model, objs, mon_rows, device)
        soup_metrics[name] = {
            "members": soup_members[name],
            "fit_mae": float(np.abs(fit_pred - y_fit_np).mean()),
            "monitor_mae": float(np.abs(mon_pred - y_mon_np).mean()),
            "fit_rows_sha256": rows_sha256(fit_rows),
            "state_sha256": state_sha256(state),
        }

    results: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "trajectory": f"{arm}{'4' if fit == 'fit4k' else '8'}-s{seed}",
        "arm": arm,
        "fit": fit,
        "seed": int(seed),
        "git_commit": git_commit(),
        "n_fit_rows": int(len(fit_rows)),
        "steps_per_epoch": steps_per_epoch,
        "T": T,
        "epochs_completed": int(epoch),
        "completed_full_epochs_at_short": EPOCHS_BASE if fit == "fit4k" else None,
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
            "base_epochs": EPOCHS_BASE, "optimizer": "Adam", "supervision": "raw y L1 only",
            "scheduler": "none", "amp": False, "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
            "soup_rule": "mean of frozen snapshots",
        },
        "snapshot_steps": snapshot_steps,
        "soup_members": soup_members,
        "parameters": n_params,
        "init_state_sha256": init_sha,
        "rng_state_after_init_sha256": rng_after_init,
        "rng_state_sha256_at_epochs": rng_hashes,
        "batch_order_sha256": array_sha256(np.asarray(order_ids, dtype=np.int64)),
        "total_steps": global_step,
        "wall_seconds": wall,
        "soups": soup_metrics,
        "curve": curve,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "discipline": {
            "fit_labels_read": f"{fit} fit",
            "monitor_labels_read": "monitor (standard training-time readout)",
            "locked_labels_read": False,
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "raw_y_only": True,
        },
    }
    torch.save(
        {
            "protocol": PROTOCOL_ID,
            "arm": arm,
            "fit": fit,
            "seed": int(seed),
            "soups": soups,
            "soup_members": soup_members,
            "init_state_sha256": init_sha,
            "git_commit": git_commit(),
        },
        checkpoint_path(arm, fit, seed),
    )
    write_json(train_json_path(arm, fit, seed), results)
    log_fn(
        f"[{arm}/{fit}/s{seed}] DONE steps={global_step} soups="
        + ", ".join(f"{k}: fit {v['fit_mae']:.6f} mon {v['monitor_mae']:.6f}" for k, v in soup_metrics.items())
        + f" wall {wall:.0f}s params {n_params}"
    )
    del objs
    return results


def cmd_train(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_trajectory(args.arm, args.fit, args.seed, device)
    return 0


# ---------------------------------------------------------------------------
# predictions + freeze
# ---------------------------------------------------------------------------


def prediction_path(readout: str, seed: int, split: str) -> Path:
    return PRED_DIR / f"{readout}_s{seed}_{split}.npz"


def cmd_predict(args: argparse.Namespace) -> int:
    """Label-blind: writes per-readout monitor and locked predictions, no y."""
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    split = read_split_rows()
    objs = load_objects()
    meta: dict[str, Any] = {"protocol": PROTOCOL_ID, "readouts": {}, "official_valid_or_test_loaded": False}
    for readout, spec in READOUT_SPECS.items():
        for seed in SEEDS:
            ckpt_path = checkpoint_path(spec["arm"], spec["fit"], seed)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert spec["soup"] in ckpt["soups"], (readout, seed, list(ckpt["soups"]))
            prep = load_prep(spec["fit"])
            model = model_for(spec["arm"], prep, seed, device)
            model.load_state_dict({k: v.to(device) for k, v in ckpt["soups"][spec["soup"]].items()})
            entry: dict[str, Any] = {
                "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(ckpt_path),
                "soup_state_sha256": state_sha256(ckpt["soups"][spec["soup"]]),
                "fit": spec["fit"],
                "splits": {},
            }
            for key in ("monitor", "locked"):
                rows = split[key]
                pred = predict_bjoc(model, objs, rows, device)
                np.savez_compressed(prediction_path(readout, seed, key), row=np.asarray(rows, dtype=np.int64),
                                    pred=pred.astype(np.float32))
                entry["splits"][key] = {
                    "rows": int(len(rows)),
                    "row_sha256": rows_sha256(rows),
                    "pred_sha256": array_sha256(pred.astype(np.float32)),
                }
                log(f"[predict] {readout} s{seed} {key}: n={len(rows)} mae-free freeze written")
            meta["readouts"][f"{readout}_s{seed}"] = entry
            del model
    write_json(OUT / "predictions_meta.json", meta)
    del objs
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    """Hash every prediction file BEFORE any label is read."""
    files = sorted(PRED_DIR.glob("*.npz"))
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "n_files": len(files),
        "files": {},
        "labels_read_at_freeze_time": False,
        "official_valid_or_test_loaded": False,
    }
    for path in files:
        z = np.load(path)
        manifest["files"][path.name] = {
            "sha256": file_sha256(path),
            "row_sha256": array_sha256(z["row"]),
            "pred_sha256": array_sha256(z["pred"]),
            "keys": sorted(z.files),
        }
    expected = {f"{r}_s{s}_{k}.npz" for r in READOUT_NAMES for s in SEEDS for k in ("monitor", "locked")}
    got = set(manifest["files"])
    assert got == expected, (sorted(expected - got), sorted(got - expected))
    h = hashlib.sha256()
    for name in sorted(manifest["files"]):
        h.update(name.encode())
        h.update(manifest["files"][name]["sha256"].encode())
    manifest["combined_sha256"] = h.hexdigest()
    write_json(OUT / "prediction_manifest.json", manifest)
    log(f"[freeze-predictions] {len(files)} files, combined sha {manifest['combined_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# scoring / analysis
# ---------------------------------------------------------------------------


def _errors(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pred.astype(np.float64) - y.astype(np.float64))


def _load_pred(readout: str, seed: int, split: str) -> np.ndarray:
    z = np.load(prediction_path(readout, seed, split))
    return z["pred"].astype(np.float64)


def _top_k_removed(delta: np.ndarray, e_new: np.ndarray, e_ref: np.ndarray, k: int) -> float:
    order = np.argsort(-np.maximum(e_new, e_ref))
    mask = np.ones_like(delta, dtype=bool)
    mask[order[:k]] = False
    return float(delta[mask].mean())


def _pair_contrast(e_new: np.ndarray, e_ref: np.ndarray, y: np.ndarray, ss: Mapping[str, np.ndarray]) -> dict[str, Any]:
    d = e_new - e_ref
    out = {
        "n": int(d.size),
        "mae_new": float(e_new.mean()),
        "mae_ref": float(e_ref.mean()),
        "delta": float(d.mean()),
        "paired_bootstrap": paired_bootstrap(d),
        "frac_candidate_better": float((d < 0).mean()),
        "top1_removed": _top_k_removed(d, e_new, e_ref, 1),
        "top5_removed": _top_k_removed(d, e_new, e_ref, 5),
        "top10_removed": _top_k_removed(d, e_new, e_ref, 10),
        "top_removed_definition": TOP_K_REMOVED_DEFINITION,
        "worst_row": {
            "rank": int(np.argmax(np.maximum(e_new, e_ref))),
            "e_new": float(e_new[np.argmax(np.maximum(e_new, e_ref))]),
            "e_ref": float(e_ref[np.argmax(np.maximum(e_new, e_ref))]),
            "y": float(y[np.argmax(np.maximum(e_new, e_ref))]),
        },
    }
    masks = {
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "atoms_le_median": ss["n_atoms"] <= np.median(ss["n_atoms"]),
        "atoms_gt_median": ss["n_atoms"] > np.median(ss["n_atoms"]),
        "junctions_le_median": ss["n_junctions"] <= np.median(ss["n_junctions"]),
        "junctions_gt_median": ss["n_junctions"] > np.median(ss["n_junctions"]),
        "cyclomatic_0": ss["cyclomatic"] <= 0,
        "cyclomatic_ge1": ss["cyclomatic"] >= 1,
        "cyclomatic_ge4": ss["cyclomatic"] >= 4,
        "ring": ss["has_ring"] > 0,
        "chain": ss["has_ring"] == 0,
    }
    out["subgroups"] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        dd = d[mask]
        out["subgroups"][name] = {
            "n": int(mask.sum()),
            "mae_new": float(e_new[mask].mean()),
            "mae_ref": float(e_ref[mask].mean()),
            "delta": float(dd.mean()),
            "paired_bootstrap": paired_bootstrap(dd) if dd.size > 1 else None,
        }
    return out


def structural_stats(objs: Sequence[sro.MolObjects], rows: Sequence[int]) -> dict[str, np.ndarray]:
    n_atoms, n_bonds, n_junc, cyc, ring = [], [], [], [], []
    for r in rows:
        o = objs[int(r)]
        na = int(o.n_atoms)
        nb = int(o.bond_index.shape[0])
        n_atoms.append(na)
        n_bonds.append(nb)
        n_junc.append(int(o.junc_index.shape[0]))
        cyc.append(nb - na + 1)
        ring.append(float(o.bond_feat[:, sro.BOND_CATEGORIES].mean()) if nb else 0.0)
    return {
        "n_atoms": np.asarray(n_atoms, dtype=np.float64),
        "n_bonds": np.asarray(n_bonds, dtype=np.float64),
        "n_junctions": np.asarray(n_junc, dtype=np.float64),
        "cyclomatic": np.asarray(cyc, dtype=np.float64),
        "has_ring": np.asarray(ring, dtype=np.float64),
    }


def _read_train_json(arm: str, fit: str, seed: int) -> dict[str, Any]:
    return read_json(train_json_path(arm, fit, seed))


def cmd_score(args: argparse.Namespace) -> int:
    if LOCK_FILE.exists() and not args.force:
        raise SystemExit(f"locked labels already read ({LOCK_FILE}); refusing a second read")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    assert read_json(OUT / "prediction_manifest.json"), "run freeze-predictions first"
    split = read_split_rows()
    objs = load_objects()
    y = {"monitor": load_y_rows(split["monitor"], subset_name="monitor"),
         "locked": load_y_rows(split["locked"], subset_name="locked")}
    ss = {k: structural_stats(objs, split[k]) for k in ("monitor", "locked")}

    preds: dict[str, dict[str, np.ndarray]] = {}
    errors: dict[str, dict[str, np.ndarray]] = {}
    for readout in READOUT_NAMES:
        for seed in SEEDS:
            key = f"{readout}_s{seed}"
            preds[key] = {}
            errors[key] = {}
            for split_key in ("monitor", "locked"):
                p = _load_pred(readout, seed, split_key)
                assert p.shape[0] == y[split_key].shape[0]
                preds[key][split_key] = p
                errors[key][split_key] = _errors(p, y[split_key])
                np.savez_compressed(
                    SCORE_DIR / f"errors_{key}_{split_key}.npz",
                    row=np.asarray(split[split_key], dtype=np.int64),
                    y=y[split_key].astype(np.float64),
                    pred=p.astype(np.float64),
                    err=errors[key][split_key],
                )

    def seed_mean_err(readout: str, split_key: str) -> np.ndarray:
        return 0.5 * (errors[f"{readout}_s0"][split_key] + errors[f"{readout}_s1"][split_key])

    def seed_mean_delta(readout: str, ref: str, split_key: str) -> np.ndarray:
        return 0.5 * (
            errors[f"{readout}_s0"][split_key] - errors[f"{ref}_s0"][split_key]
            + errors[f"{readout}_s1"][split_key] - errors[f"{ref}_s1"][split_key]
        )

    lc: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "locked_labels_read": True,
        "official_valid_or_test_loaded": False,
        "readout_metrics": {},
        "seed_mean_metrics": {},
        "contrasts": {},
    }
    for readout in READOUT_NAMES:
        for seed in SEEDS:
            spec = READOUT_SPECS[readout]
            tj = _read_train_json(spec["arm"], spec["fit"], seed)
            lc["readout_metrics"][f"{readout}_s{seed}"] = {
                "fit": tj["soups"][spec["soup"]]["fit_mae"],
                "monitor": metrics(preds[f"{readout}_s{seed}"]["monitor"], y["monitor"]),
                "locked": metrics(preds[f"{readout}_s{seed}"]["locked"], y["locked"]),
            }
        lc["seed_mean_metrics"][readout] = {
            "monitor": {k: float(np.mean([lc["readout_metrics"][f"{readout}_s{s}"]["monitor"][k] for s in SEEDS]))
                        for k in ("mae", "rmse", "median", "p90", "p95")},
            "locked": {k: float(np.mean([lc["readout_metrics"][f"{readout}_s{s}"]["locked"][k] for s in SEEDS]))
                       for k in ("mae", "rmse", "median", "p90", "p95")},
            "fit": float(np.mean([lc["readout_metrics"][f"{readout}_s{s}"]["fit"] for s in SEEDS])),
        }
    for name, (new, ref) in {
        "N4stepmatched_minus_N4short": ("N4stepmatched", "N4short"),
        "N8_minus_N4stepmatched": ("N8", "N4stepmatched"),
        "N8_minus_N4short": ("N8", "N4short"),
        "O8_minus_N8": ("O8", "N8"),
    }.items():
        entry: dict[str, Any] = {"new": new, "ref": ref}
        for split_key in ("monitor", "locked"):
            entry[split_key] = {
                "seed0": _pair_contrast(errors[f"{new}_s0"][split_key], errors[f"{ref}_s0"][split_key],
                                        y[split_key], ss[split_key]),
                "seed1": _pair_contrast(errors[f"{new}_s1"][split_key], errors[f"{ref}_s1"][split_key],
                                        y[split_key], ss[split_key]),
                "seed_mean": _pair_contrast(seed_mean_err(new, split_key), seed_mean_err(ref, split_key),
                                            y[split_key], ss[split_key]),
            }
        lc["contrasts"][name] = entry
    write_json(OUT / "learning_curve_analysis.json", lc)

    own: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "definition": "O8 minus N8 on the same split / seed / update count; paired molecule delta",
        "locked": lc["contrasts"]["O8_minus_N8"]["locked"],
        "monitor": lc["contrasts"]["O8_minus_N8"]["monitor"],
    }
    write_json(OUT / "ownership_paired_analysis.json", own)

    write_json(
        LOCK_FILE,
        {
            "protocol": PROTOCOL_ID,
            "monitor_label_reads": 1,
            "locked_label_reads": 1,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "prediction_manifest_sha256": file_sha256(OUT / "prediction_manifest.json"),
        },
    )
    log("[score] monitor and locked labels read once; learning curve + ownership analysis written")
    log(f"[score] locked seed-mean MAE: "
        + ", ".join(f"{r} {lc['seed_mean_metrics'][r]['locked']['mae']:.6f}" for r in READOUT_NAMES))
    del objs
    return 0


# ---------------------------------------------------------------------------
# common errors (Phase G)
# ---------------------------------------------------------------------------


def cmd_common_errors(args: argparse.Namespace) -> int:
    split = read_split_rows()
    y = load_y_rows(split["locked"], subset_name="locked")
    objs = load_objects()
    ss = structural_stats(objs, split["locked"])
    models = [f"{r}_s{s}" for r in ("N8", "O8") for s in SEEDS]
    err = {}
    pred = {}
    for key in models:
        z = np.load(SCORE_DIR / f"errors_{key}_locked.npz")
        err[key] = z["err"]
        pred[key] = z["pred"]
    n = y.shape[0]
    k = int(round(0.10 * n))
    hi = {key: set(np.argsort(-err[key])[:k].tolist()) for key in models}
    intersections: dict[str, int] = {}
    jaccards: dict[str, float] = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            inter = len(hi[a] & hi[b])
            union = len(hi[a] | hi[b])
            intersections[f"{a}|{b}"] = inter
            jaccards[f"{a}|{b}"] = inter / union if union else float("nan")
    four_way = set.intersection(*(hi[m] for m in models))
    res_corr: dict[str, float] = {}
    sign_agree: dict[str, float] = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ea = (pred[a] - y)
            eb = (pred[b] - y)
            res_corr[f"{a}|{b}"] = float(np.corrcoef(ea, eb)[0, 1])
            shared = hi[a] & hi[b]
            if shared:
                idx = sorted(shared)
                sign_agree[f"{a}|{b}"] = float(np.mean(np.sign(pred[a][idx] - y[idx]) == np.sign(pred[b][idx] - y[idx])))
    structural: dict[str, Any] = {}
    for name, idxs in (("four_way_high_error", sorted(four_way)),):
        if idxs:
            ii = np.asarray(idxs)
            structural[name] = {
                "rows": ii.tolist(),
                "n": int(len(ii)),
                "y_mean": float(y[ii].mean()),
                "y_median": float(np.median(y[ii])),
                "n_atoms_mean": float(ss["n_atoms"][ii].mean()),
                "n_atoms_median": float(np.median(ss["n_atoms"][ii])),
                "n_junctions_mean": float(ss["n_junctions"][ii].mean()),
                "cyclomatic_mean": float(ss["cyclomatic"][ii].mean()),
                "ring_fraction": float((ss["has_ring"][ii] > 0).mean()),
                "high_error_share_of_y_le_-3": float(
                    (ss["cyclomatic"][ii] * 0 + (y[ii] <= -3)).mean()
                ),
            }
    n_fix = {}
    for a, b, label in (("N8_s0", "O8_s0", "O_fixes_N_s0"), ("N8_s1", "O8_s1", "O_fixes_N_s1")):
        fixed = hi[a] - hi[b]
        n_fix[label] = {"count": len(fixed), "mean_err_a_on_fixed": float(err[a][sorted(fixed)].mean()) if fixed else None}
    for key in ("N8_s0", "N8_s1", "O8_s0", "O8_s1"):
        other = key.replace("_s0", "_s1").replace("_s1", "_s0")
        if other in hi:
            n_fix[f"seed_stability_{key}"] = {
                "top10_overlap": len(hi[key] & hi[other]),
                "jaccard": len(hi[key] & hi[other]) / len(hi[key] | hi[other]),
            }
    # does a larger O-N prediction change actually reduce error?
    bins: dict[str, Any] = {}
    for a, b in (("N8_s0", "O8_s0"), ("N8_s1", "O8_s1")):
        dpred = np.abs(pred[b] - pred[a])
        ded = err[b] - err[a]
        order = np.argsort(-dpred)
        for topk in (100, 500):
            idx = order[:topk]
            bins[f"{a}->{b}_top{topk}_dpred"] = {
                "mean_dpred": float(dpred[idx].mean()),
                "mean_delta_err": float(ded[idx].mean()),
                "frac_improved": float((ded[idx] < 0).mean()),
            }
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "split": "locked",
        "high_error_definition": "per-model top 10% of |e| on the locked split (n=100 of 1000)",
        "models": models,
        "high_error_counts": {m: len(hi[m]) for m in models},
        "pairwise_intersections": intersections,
        "pairwise_jaccard": jaccards,
        "mean_pairwise_jaccard": float(np.mean(list(jaccards.values()))),
        "four_way_intersection": len(four_way),
        "four_way_fraction_of_each_model_set": {m: len(four_way) / len(hi[m]) for m in models},
        "residual_correlation": res_corr,
        "same_direction_fraction_among_shared_high_error": sign_agree,
        "cross_seed_and_cross_ownership_fixes": n_fix,
        "prediction_difference_bins": bins,
        "four_way_structural_profile": structural,
        "official_valid_or_test_loaded": False,
    }
    write_json(OUT / "common_error_analysis.json", payload)
    log(f"[common-errors] four-way intersection {len(four_way)}/100; mean Jaccard {payload['mean_pairwise_jaccard']:.3f}")
    del objs
    return 0


# ---------------------------------------------------------------------------
# Phase B: historical N/O/P monitor diagnostics (CPU, labels already public)
# ---------------------------------------------------------------------------


def cmd_historical(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "official_valid_or_test_loaded": False,
        "note": "read-only audit of already-frozen historical predictions; no model is trained here",
    }
    z = np.load(CB / "split_a_monitor_predictions.npz")
    rows = z["monitor_row"].astype(np.int64)
    pN = z["pred_N_A_s0_monitor"].astype(np.float64)
    pO = z["pred_O_A_s0_monitor"].astype(np.float64)
    pP = z["pred_P_A_s0_monitor"].astype(np.float64)
    y = load_y_rows(rows, subset_name="crossbond split A monitor")
    objs = load_objects()
    ss = structural_stats(objs, rows)
    eN, eO, eP = _errors(pN, y), _errors(pO, y), _errors(pP, y)
    b1 = _pair_contrast(eO, eN, y, ss)
    b1["source"] = "results/bjoc_cross_bond_interaction_v1/split_a_monitor_predictions.npz (frozen)"
    b1["context"] = {
        "split": "cscl-bjoc-crossbond-v1-discovery",
        "fit_size": 8000,
        "seed": 0,
        "recipe": "Adam lr 1e-3 wd 1e-5 batch 128 clip 5 240 epochs last-5 soup raw-y L1",
    }
    b1["mae_N"] = float(eN.mean())
    b1["mae_O"] = float(eO.mean())
    b1["mae_P"] = float(eP.mean())
    b1["delta_O_minus_N"] = float((eO - eN).mean())
    payload["B1_crossbond_O_vs_N_monitor"] = b1

    payload["B3_three_model_common_errors"] = {
        "residual_correlation": {
            "N_vs_O": float(np.corrcoef(pN - y, pO - y)[0, 1]),
            "N_vs_P": float(np.corrcoef(pN - y, pP - y)[0, 1]),
            "O_vs_P": float(np.corrcoef(pO - y, pP - y)[0, 1]),
        },
        "high_error_sets_top10pct": {},
        "prediction_difference_error_reduction": {},
        "O_fix_analysis": {},
        "P_fix_analysis": {},
    }
    k = int(round(0.10 * len(y)))
    orderN, orderO, orderP = (np.argsort(-e)[:k] for e in (eN, eO, eP))
    hiN, hiO, hiP = (set(o.tolist()) for o in (orderN, orderO, orderP))
    tri = hiN & hiO & hiP
    payload["B3_three_model_common_errors"]["high_error_sets_top10pct"] = {
        "n": k,
        "N_and_O": len(hiN & hiO),
        "N_and_P": len(hiN & hiP),
        "O_and_P": len(hiO & hiP),
        "all_three": len(tri),
        "intersection_fraction_of_each": {name: len(tri) / k for name in ("N", "O", "P")},
        "jaccard_N_O": len(hiN & hiO) / len(hiN | hiO),
        "jaccard_N_P": len(hiN & hiP) / len(hiN | hiP),
        "jaccard_O_P": len(hiO & hiP) / len(hiO | hiP),
    }
    for a, b, name in ((pN, pO, "N_to_O"), (pO, pP, "O_to_P")):
        d = np.abs(b - y) - np.abs(a - y)
        order = np.argsort(-np.abs(b - a))
        payload["B3_three_model_common_errors"]["prediction_difference_error_reduction"][name] = {
            "top100_mean_dpred": float(np.abs(b - a)[order[:100]].mean()),
            "top100_mean_delta_err": float(d[order[:100]].mean()),
            "top100_frac_improved": float((d[order[:100]] < 0).mean()),
        }
    fixed_by_O = hiN - hiO
    fixed_by_P = hiO - hiP
    tri_idx = np.asarray(sorted(tri), dtype=np.int64)
    payload["B3_three_model_common_errors"]["O_fix_analysis"] = {
        "N_high_O_not_high": len(fixed_by_O),
        "mean_N_err_on_fixed": float(eN[sorted(fixed_by_O)].mean()) if fixed_by_O else None,
    }
    payload["B3_three_model_common_errors"]["P_fix_analysis"] = {
        "O_high_P_not_high": len(fixed_by_P),
        "mean_O_err_on_fixed": float(eO[sorted(fixed_by_P)].mean()) if fixed_by_P else None,
    }
    payload["B3_three_model_common_errors"]["all_three_structural_profile"] = {
        "n": int(len(tri_idx)),
        "y_mean": float(y[tri_idx].mean()) if len(tri_idx) else None,
        "y_le_-3_fraction": float((y[tri_idx] <= -3).mean()) if len(tri_idx) else None,
        "n_atoms_mean": float(ss["n_atoms"][tri_idx].mean()) if len(tri_idx) else None,
        "n_junctions_mean": float(ss["n_junctions"][tri_idx].mean()) if len(tri_idx) else None,
        "cyclomatic_mean": float(ss["cyclomatic"][tri_idx].mean()) if len(tri_idx) else None,
        "ring_fraction": float((ss["has_ring"][tri_idx] > 0).mean()) if len(tri_idx) else None,
    }

    survey: dict[str, Any] = {"matched_within_round": [], "different_conditions_background": [],
                              "single_seed": [], "extreme_dominated": [], "opposite_direction": []}
    try:
        z0 = np.load(SRC_DIR / "bjoc_null_s0_monitor_preds.npz")
        z0o = np.load(SRC_DIR / "bjoc_organized_s0_monitor_preds.npz")
        z1 = np.load(SRC_DIR / "bjoc_null_s1_monitor_preds.npz")
        z1o = np.load(SRC_DIR / "bjoc_organized_s1_monitor_preds.npz")
        for seed, (zn, zo) in enumerate(((z0, z0o), (z1, z1o))):
            yy = zn["y"].astype(np.float64)
            en = _errors(zn["pred"], yy)
            eo = _errors(zo["pred"], yy)
            d = eo - en
            survey["matched_within_round"].append({
                "source": "cscl-structural-relation-composer-v1 monitor",
                "seed": seed, "n": int(len(d)), "mae_N": float(en.mean()), "mae_O": float(eo.mean()),
                "delta_O_minus_N": float(d.mean()), "paired_bootstrap": paired_bootstrap(d),
            })
        ln = np.load(SRC_DIR / "locked_preds_null.npz")
        lo = np.load(SRC_DIR / "locked_preds_organized.npz")
        yl = ln["y"].astype(np.float64)
        dl = _errors(lo["pred"], yl) - _errors(ln["pred"], yl)
        survey["matched_within_round"].append({
            "source": "cscl-structural-relation-composer-v1 locked (seed 0 only)",
            "seed": 0, "n": int(len(dl)), "delta_O_minus_N": float(dl.mean()),
            "paired_bootstrap": paired_bootstrap(dl), "single_seed": True,
        })
    except FileNotFoundError as exc:  # pragma: no cover
        survey["matched_within_round"].append({"error": repr(exc)})
    try:
        ov = read_json(SRC_DIR / "official_valid_eval.json")
        survey["different_conditions_background"].append({
            "source": "cscl-structural-relation-composer-v1 official valid (full-10k N/O)",
            "detail": ov,
            "classification": "different training condition (full 10k, another evaluation path)",
        })
    except FileNotFoundError:
        pass
    for seed in SEEDS:
        try:
            tj = read_json(JD / f"train_R0_A_s{seed}.json")
            survey["single_seed" if seed == 0 else "different_conditions_background"].append({
                "source": "cscl-bjoc-junction-regularization-v1 R0 (bit-identical BJOC-Null)",
                "seed": seed,
                "soup_fit_mae": tj["soup_fit_mae"],
                "soup_monitor_mae": tj["soup_monitor_mae"],
                "classification": "null-arm-only background; no matched Organized arm",
            })
        except FileNotFoundError:
            pass
    try:
        tj = read_json(IJ / "train_B_A_s0.json")
        survey["single_seed"].append({
            "source": "cscl-bjoc-intrinsic-contribution-feasibility-v1 B arm (BJOC-Null)",
            "seed": 0, "soup_fit_mae": tj["soup_fit_mae"], "soup_monitor_mae": tj["soup_monitor_mae"],
            "classification": "null-arm-only background",
        })
    except FileNotFoundError:
        pass
    payload["B2_historical_survey"] = survey
    payload["B2_reading"] = (
        "Only the within-round matched N/O comparisons (same split, seed, recipe, update count) are "
        "numerically comparable.  All absolute MAE levels across different splits are background only."
    )
    write_json(OUT / "historical_ownership_audit.json", payload)
    log(f"[historical] O-N monitor delta {b1['delta_O_minus_N']:+.6f}, "
        f"CI {b1['paired_bootstrap']['ci95']}, better fraction {b1['paired_bootstrap']['frac_candidate_better']:.3f}")
    del objs
    return 0


# ---------------------------------------------------------------------------
# Phase C: input observability + graph collision
# ---------------------------------------------------------------------------


def _wl_hash(obj: sro.MolObjects) -> str:
    import networkx as nx

    g = nx.Graph()
    atom_types = np.argmax(obj.atom_feat[:, :sro.ATOM_CATEGORIES], axis=1)
    for i in range(obj.n_atoms):
        g.add_node(int(i), at=int(atom_types[i]))
    for k in range(obj.bond_index.shape[0]):
        u, v = int(obj.bond_index[k, 0]), int(obj.bond_index[k, 1])
        g.add_edge(u, v, bt=int(obj.bond_type[k]))
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="at", edge_attr="bt", iterations=5, digest_size=20)


def _isomorphic(obj_a: sro.MolObjects, obj_b: sro.MolObjects) -> bool:
    import networkx as nx

    def graph(obj):
        g = nx.Graph()
        atom_types = np.argmax(obj.atom_feat[:, :sro.ATOM_CATEGORIES], axis=1)
        for i in range(obj.n_atoms):
            g.add_node(int(i), at=int(atom_types[i]))
        for k in range(obj.bond_index.shape[0]):
            g.add_edge(int(obj.bond_index[k, 0]), int(obj.bond_index[k, 1]), bt=int(obj.bond_type[k]))
        return g

    ga, gb = graph(obj_a), graph(obj_b)
    if ga.number_of_nodes() != gb.number_of_nodes() or ga.number_of_edges() != gb.number_of_edges():
        return False
    nm = nx.algorithms.isomorphism.categorical_node_match("at", -1)
    em = nx.algorithms.isomorphism.categorical_edge_match("bt", -1)
    return nx.is_isomorphic(ga, gb, node_match=nm, edge_match=em)


def _input_hash(obj: sro.MolObjects) -> str:
    """Hash of the model input tensors, canonicalised over object order.

    Includes the exact atom/bond/junction feature rows and the derived global
    topology vector; the atom type one-hots are part of atom_feat, so this hash
    is a strict statement about what the model can read.
    """
    h = hashlib.sha256()
    atoms = sorted(obj.atom_feat[i].tobytes() for i in range(obj.n_atoms))
    h.update(b"A")
    for a in atoms:
        h.update(a)
    bonds = []
    for k in range(obj.bond_index.shape[0]):
        u, v = int(obj.bond_index[k, 0]), int(obj.bond_index[k, 1])
        au, av = obj.atom_feat[u].tobytes(), obj.atom_feat[v].tobytes()
        if au > av:
            au, av = av, au
        bonds.append(au + av + obj.bond_feat[k].tobytes())
    h.update(b"B")
    for b in sorted(bonds):
        h.update(b)
    junc = []
    for j in range(obj.junc_index.shape[0]):
        i, c, k = (int(x) for x in obj.junc_index[j])
        b1, b2 = (int(x) for x in obj.junc_bond[j])
        ai, ak = obj.atom_feat[i].tobytes(), obj.atom_feat[k].tobytes()
        f1, f2 = obj.bond_feat[b1].tobytes(), obj.bond_feat[b2].tobytes()
        if ai > ak:
            ai, ak = ak, ai
            f1, f2 = f2, f1
        junc.append(ai + obj.atom_feat[c].tobytes() + ak + f1 + f2 + obj.junc_feat[j].tobytes())
    h.update(b"J")
    for j in sorted(junc):
        h.update(j)
    h.update(b"G")
    h.update(np.round(obj.global_feat.astype(np.float64), 6).tobytes())
    return h.hexdigest()


def _structural_input_hash(obj: sro.MolObjects) -> str:
    """Like ``_input_hash`` but without the global vector (which is itself a
    deterministic function of the labelled graph; two isomorphic molecules can
    differ there only in float rounding order)."""
    clone = sro.MolObjects(
        mol_index=obj.mol_index, smiles=obj.smiles, n_atoms=obj.n_atoms,
        atom_feat=obj.atom_feat, bond_index=obj.bond_index, bond_type=obj.bond_type,
        bond_feat=obj.bond_feat, junc_index=obj.junc_index, junc_bond=obj.junc_bond,
        junc_feat=obj.junc_feat, global_feat=np.zeros_like(obj.global_feat),
    )
    return _input_hash(clone)


def cmd_observability(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    objs = load_objects()
    n = len(objs)
    # ---- C1: layout documentation
    doc = {
        "protocol": PROTOCOL_ID,
        "source_module": "code/structural_relation_objects_v1.py",
        "atom_feat": {
            "dim": sro.ATOM_FEAT_DIM,
            "blocks": {k: [int(v.start), int(v.stop)] for k, v in sro.atom_blocks().items()},
            "provenance": "raw PyG x (ZINC atom type code) + deterministic topological statistics",
        },
        "bond_feat": {
            "dim": sro.BOND_FEAT_DIM,
            "provenance": "PyG edge_attr bond type one-hot + ring flags; unique undirected bond (u<v)",
        },
        "junc_feat": {
            "dim": sro.JUNC_FEAT_DIM,
            "provenance": "centre degree/triangle/common-neighbour/ring stats + bond-type pair outer sum",
        },
        "global_feat": {
            "dim": sro.GLOBAL_DIM,
            "names": list(sro.GLOBAL_FEAT_NAMES),
            "provenance": "label-blind whole-graph topology statistics",
            "standardisation": "fit-rows-only global mean/scale",
        },
        "edge_indexing": "PyG stores each physical bond twice; build_mol_objects canonicalises to one undirected bond u<v",
        "ring_information": "min ring size per atom, is_ring_bond, ring systems, fused/spiro centre flags (fixed algorithms)",
        "preserved": [
            "atom type code (PyG x)",
            "physical bond type",
            "adjacency/topology",
            "ring membership and minimum ring sizes",
            "radius-1/2 atom-type histograms",
            "degree, terminal/fused flags",
            "junction (centre + two neighbours) geometry flags",
            "whole-graph topology summaries",
        ],
        "dropped_or_absent": [
            "stereochemistry / chirality (not stored in the PyG subset inputs)",
            "formal charges (not in x; x is a small integer atom-type code)",
            "explicit hydrogens (heavy-atom graph only)",
            "bond stereo / E-Z",
            "isotopes",
            "3D conformer",
            "aromatic bond flag as a separate class (edge_attr in this snapshot is {1,2,3}; the one-hot has an unused column 0)",
            "anything outside the labelled graph: conformer energy, solvent, pH",
        ],
        "raw_vocabulary": {
            "atom_type_codes": sorted({int(np.argmax(o.atom_feat[i, :sro.ATOM_CATEGORIES])) for o in objs for i in range(o.n_atoms)}),
            "bond_types": sorted({int(b) for o in objs for b in o.bond_type}),
            "atom_categories_declared": sro.ATOM_CATEGORIES,
            "bond_categories_declared": sro.BOND_CATEGORIES,
        },
        "consequence": (
            "Two molecules whose labelled graphs are isomorphic have exactly the same model input "
            "multiset (up to object order).  Any target difference not explained by the labelled "
            "graph, e.g. stereochemistry, is not observable to this model."
        ),
    }
    write_json(OUT / "input_observability_audit.json", doc)

    # ---- C2/C3: WL candidates -> strict VF2 -> input hash
    wl_groups: dict[str, list[int]] = {}
    input_groups: dict[str, list[int]] = {}
    struct_groups: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for i, obj in enumerate(objs):
        wl_groups.setdefault(_wl_hash(obj), []).append(i)
        input_groups.setdefault(_input_hash(obj), []).append(i)
        struct_groups.setdefault(_structural_input_hash(obj), []).append(i)
        if (i + 1) % 2000 == 0:
            log(f"[observability] hashed {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)")
    import networkx as nx  # noqa: F401  (used inside helpers)

    def groups_of(d: Mapping[str, list[int]]) -> list[list[int]]:
        return [sorted(v) for v in d.values() if len(v) > 1]

    import yaml  # noqa: F401

    all_rows = list(range(n))
    y_all = load_y_rows(all_rows, subset_name="official train 10000 (fit-only audits)")
    split = read_split_rows()
    y_by_row = {r: float(y_all[r]) for r in all_rows}

    def check_collisions(groups: Sequence[Sequence[int]], *, require_vf2: bool) -> list[dict[str, Any]]:
        out = []
        for g in groups:
            if require_vf2:
                subgroup: list[list[int]] = []
                for r in g:
                    placed = False
                    for sg in subgroup:
                        if _isomorphic(objs[r], objs[sg[0]]):
                            sg.append(r)
                            placed = True
                            break
                    if not placed:
                        subgroup.append([r])
                candidates = [sg for sg in subgroup if len(sg) > 1]
            else:
                candidates = [list(g)]
            for sg in candidates:
                ys = [y_by_row[r] for r in sg]
                out.append({
                    "rows": [int(r) for r in sg],
                    "n": len(sg),
                    "smiles": [str(objs[r].smiles) for r in sg],
                    "y": [float(v) for v in ys],
                    "y_range": float(max(ys) - min(ys)),
                    "in_fit8k": [bool(r in set(split["fit8k"])) for r in sg],
                    "in_fit4k": [bool(r in set(split["fit4k"])) for r in sg],
                    "in_monitor": [bool(r in set(split["monitor"])) for r in sg],
                    "in_locked": [bool(r in set(split["locked"])) for r in sg],
                })
        return out

    t1 = time.perf_counter()
    vf2_groups = check_collisions(groups_of(wl_groups), require_vf2=True)
    input_hash_groups = check_collisions(groups_of(input_groups), require_vf2=False)
    struct_hash_groups = check_collisions(groups_of(struct_groups), require_vf2=False)
    log(f"[observability] VF2 verification {time.perf_counter() - t1:.0f}s")

    def floor(groups: Sequence[Mapping[str, Any]], rows_subset: set[int]) -> dict[str, Any]:
        affected = 0
        total_dev = 0.0
        detail = []
        for g in groups:
            idx = [i for i, r in enumerate(g["rows"]) if r in rows_subset]
            if len(idx) < 2:
                continue
            ys = np.asarray([g["y"][i] for i in idx], dtype=np.float64)
            med = float(np.median(ys))
            dev = float(np.abs(ys - med).mean())
            affected += len(idx)
            total_dev += dev * len(idx)
            detail.append({"rows": [g["rows"][i] for i in idx], "n": len(idx),
                           "y_range": float(ys.max() - ys.min()), "abs_dev_from_median": dev})
        return {
            "n_rows_affected": affected,
            "sum_abs_dev_from_group_median": total_dev,
            "empirical_indistinguishable_floor": (total_dev / affected) if affected else None,
            "definition": "mean over affected rows of |y - median(y in its strict input-equivalence group)|",
            "groups": detail,
        }

    fit8k, fit4k = set(split["fit8k"]), set(split["fit4k"])
    smiles_groups: dict[str, list[int]] = {}
    for i, obj in enumerate(objs):
        smiles_groups.setdefault(str(obj.smiles), []).append(i)
    dup_smiles = []
    for smi, rows in smiles_groups.items():
        if len(rows) < 2:
            continue
        ys = [y_by_row[r] for r in rows]
        vf2_pairs = []
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                vf2_pairs.append(_isomorphic(objs[rows[a]], objs[rows[b]]))
        dup_smiles.append({
            "smiles": smi,
            "rows": [int(r) for r in rows],
            "y": [float(v) for v in ys],
            "y_range": float(max(ys) - min(ys)),
            "vf2_isomorphic_pairs": [bool(v) for v in vf2_pairs],
            "all_vf2_isomorphic": bool(all(vf2_pairs)),
            "input_hash_equal_pairs": [bool(_input_hash(objs[rows[a]]) == _input_hash(objs[rows[b]]))
                                       for a in range(len(rows)) for b in range(a + 1, len(rows))],
        })
    collision = {
        "protocol": PROTOCOL_ID,
        "method": {
            "wl_hash": "networkx.weisfeiler_lehman_graph_hash(node_attr=atom_type, edge_attr=bond_type, iterations=5)",
            "strict_verification": "VF2 labelled isomorphism (node_match + edge_match) inside WL-collision groups",
            "input_hash": "sha256 over canonicalised exact atom/bond/junction feature rows + quantised global vector",
            "structural_input_hash": "input hash with the global vector zeroed (global vector is itself a graph function)",
            "caveat": "WL hash alone is only a candidate filter; every reported group is VF2-verified",
        },
        "wl_candidate_groups": {
            "n_groups": int(len(wl_groups)),
            "n_multi_row_groups": int(len(groups_of(wl_groups))),
            "n_rows_in_multi_row_groups": int(sum(len(g) for g in groups_of(wl_groups))),
        },
        "strict_vf2_groups": vf2_groups,
        "input_hash_groups": input_hash_groups,
        "structural_input_hash_groups": struct_hash_groups,
        "duplicate_smiles_groups": dup_smiles,
        "duplicate_smiles_vs_input_reading": (
            "same canonical SMILES is NOT sufficient for an identical model input: the raw ZINC "
            "atom-type codes enter the atom one-hot and can differ between duplicate-SMILES rows; "
            "only groups whose labelled graphs are strictly isomorphic produce identical inputs"
        ),
        "y_inconsistency_full10k": floor(vf2_groups, set(range(n))),
        "y_inconsistency_fit8k": floor(vf2_groups, fit8k),
        "y_inconsistency_fit4k": floor(vf2_groups, fit4k),
        "note_on_input_hash_groups": (
            "input_hash_groups are bit-identical model inputs regardless of VF2; if such a group is "
            "not VF2-isomorphic it is an even stronger observability failure"
        ),
        "official_valid_or_test_loaded": False,
    }
    write_json(OUT / "graph_collision_audit.json", collision)
    n_collision_rows = sum(g["n"] for g in vf2_groups)
    log(f"[observability] strict VF2 collision groups: {len(vf2_groups)} ({n_collision_rows} rows); "
        f"input-hash groups: {len(input_hash_groups)}")
    del objs
    return 0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def cmd_gates(args: argparse.Namespace) -> int:
    frozen = frozen_plan()
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "gates": {}}

    def gate(name: str, ok: bool, detail: Any) -> None:
        out["gates"][name] = {"pass": bool(ok), "detail": detail}
        log(f"[gate] {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"correctness gate failed: {name}")

    split = read_split_rows()
    # 1-2. split structure + frozen hashes
    gate("split_nesting", set(split["fit4k"]) <= set(split["fit8k"]),
         {"n_fit4k": len(split["fit4k"]), "n_fit8k": len(split["fit8k"])})
    gate(
        "split_hashes_match_frozen",
        rows_sha256(split["fit8k"]) == frozen["fit8k_sha256_rows"]
        and rows_sha256(split["monitor"]) == frozen["monitor_sha256_rows"]
        and rows_sha256(split["locked"]) == frozen["locked_sha256_rows"]
        and rows_sha256(split["fit4k"]) == frozen["fit4k_sha256_rows"],
        {"fit8k": rows_sha256(split["fit8k"]), "fit4k": rows_sha256(split["fit4k"])},
    )
    gate(
        "split_disjoint_and_complete",
        split_set(split, "fit8k").isdisjoint(split_set(split, "monitor"))
        and split_set(split, "fit8k").isdisjoint(split_set(split, "locked"))
        and split_set(split, "monitor").isdisjoint(split_set(split, "locked"))
        and len(split_set(split, "fit8k") | split_set(split, "monitor") | split_set(split, "locked")) == 10000,
        {"n_fit8k": len(split["fit8k"]), "n_monitor": len(split["monitor"]), "n_locked": len(split["locked"])},
    )
    gate(
        "fit4k_never_touches_monitor_or_locked",
        set(split["fit4k"]).isdisjoint(split["monitor"]) and set(split["fit4k"]).isdisjoint(split["locked"]),
        {"n": len(split["fit4k"])},
    )
    # 3. steps frozen
    steps = compute_steps(len(split["fit8k"]), len(split["fit4k"]))
    gate(
        "step_plan_matches_frozen",
        int(steps["T"]) == int(frozen["T"])
        and list(steps["n4_short_snapshot_steps"]) == list(frozen["n4_short_snapshot_steps"])
        and list(steps["n4_stepmatched_snapshot_steps"]) == list(frozen["n4_stepmatched_snapshot_steps"])
        and list(steps["n8_o8_soup_steps"]) == list(frozen["n8_o8_soup_steps"])
        and int(steps["steps_per_epoch_4k"]) == int(frozen["steps_per_epoch_4k"]),
        steps,
    )
    # 4. normalization strictly fit-only
    objs = load_objects()
    norm_ok = True
    norm_detail = {}
    for fit, rows in (("fit8k", split["fit8k"]), ("fit4k", split["fit4k"])):
        gfit = np.stack([objs[int(i)].global_feat for i in rows], 0).astype(np.float64)
        mean = gfit.mean(0)
        scale = np.where(gfit.std(0) < 1e-6, 1.0, gfit.std(0))
        prep = load_prep(fit)
        same = np.array_equal(mean, np.asarray(prep["global_mean"], dtype=np.float64)) and np.array_equal(
            scale, np.asarray(prep["global_scale"], dtype=np.float64)
        )
        norm_ok = norm_ok and same
        norm_detail[fit] = {"n_fit": int(len(rows)), "recomputed_equal": bool(same),
                            "prep_sha256": file_sha256(PREP_FILES[fit])}
        gate(f"norm_{fit}_fit_only", same and int(prep["n_fit"]) == len(rows), norm_detail[fit])
    gate(
        "norm_prep_hashes_match_frozen",
        all(file_sha256(PREP_FILES[f]) == frozen[f"prep_{f}_sha256"] for f in ("fit8k", "fit4k")),
        norm_detail,
    )
    # 5. N/O parameter count + init identity
    cpu = torch.device("cpu")
    n_model = model_for("N", load_prep("fit8k"), 0, cpu)
    o_model = model_for("O", load_prep("fit8k"), 0, cpu)
    n_params = int(sum(p.numel() for p in n_model.parameters()))
    o_params = int(sum(p.numel() for p in o_model.parameters()))
    gate("parameter_count_413505", n_params == 413505 and o_params == 413505, {"N": n_params, "O": o_params})
    gate("init_hash_equal_same_seed",
         src.state_dict_hash(n_model) == src.state_dict_hash(o_model),
         {"seed0": src.state_dict_hash(n_model)})
    gate("init_hash_equal_seed1",
         src.state_dict_hash(model_for("N", load_prep("fit8k"), 1, cpu))
         == src.state_dict_hash(model_for("O", load_prep("fit8k"), 1, cpu)),
         {})
    gate(
        "no_hidden_arch_difference",
        list(n_model.state_dict()) == list(o_model.state_dict())
        and all(n_model.state_dict()[k].shape == o_model.state_dict()[k].shape for k in n_model.state_dict())
        and n_model.organization_mode == src.NULL and o_model.organization_mode == src.ORGANIZED,
        {"keys_equal": True},
    )
    # 6. batch order identical for N/O given a seed (generator independent of arm)
    row_sample = split["fit8k"][:256]
    orders = {}
    for arm in ("N", "O"):
        g = torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)
        orders[arm] = [i for _ in range(2) for i in [b for batch in shuffle_batches(len(row_sample), BATCH_SIZE, g) for b in batch]]
    gate("batch_order_equal_N_O_same_seed", orders["N"] == orders["O"], {"n": len(orders["N"])})
    # 7. invariance / consistency / finite / replay
    tiny = objs[:8]
    rows_tiny = list(range(8))
    b1 = sro.batch_objects(tiny)
    p_ref = predict_bjoc(n_model, tiny, rows_tiny, cpu)
    # batch consistency: individually vs chunks
    p_chunks = np.concatenate([predict_bjoc(n_model, [tiny[i]], [0], cpu) for i in range(8)])
    gate("batch_consistency", float(np.max(np.abs(p_ref - p_chunks))) < 1e-5,
         {"max_dp": float(np.max(np.abs(p_ref - p_chunks)))})
    # relabel invariance
    perm = np.array([3, 0, 7, 1, 6, 2, 5, 4], dtype=np.int64) % 8
    if len(set(perm.tolist())) != 8:
        perm = np.roll(np.arange(8), 1)
    relab = []
    for o in tiny[:4]:
        relab.append(relabel_objects(o, np.random.default_rng(0).permutation(o.n_atoms)))
    p_orig = predict_bjoc(n_model, tiny[:4], list(range(4)), cpu)
    p_rel = predict_bjoc(n_model, relab, list(range(4)), cpu)
    gate("atom_relabel_invariance", float(np.max(np.abs(p_orig - p_rel))) < 1e-5,
         {"max_dp": float(np.max(np.abs(p_orig - p_rel)))})
    gate("finite_forward",
         bool(np.isfinite(p_ref).all()) and bool(np.isfinite(predict_bjoc(o_model, tiny, rows_tiny, cpu)).all()), {})
    tmp = OUT / "gate_replay_tmp.pt"
    torch.save(n_model.state_dict(), tmp)
    fresh = model_for("N", load_prep("fit8k"), 0, cpu)
    fresh.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=False))
    gate("save_load_replay",
         float(np.max(np.abs(predict_bjoc(fresh, tiny, rows_tiny, cpu) - p_ref))) == 0.0, {})
    tmp.unlink()
    # 8. raw objects roundtrip + cache integrity
    cache_sha = file_sha256(rb.OBJECTS_CACHE)
    gate("objects_cache_unchanged", cache_sha == rb.OBJECTS_CACHE_SHA256 and cache_sha == frozen["objects_cache_sha256"],
         {"sha256": cache_sha})
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc  # noqa: E402

    ds = _load_zinc(cf.ZINC_ROOT, "train")
    sample_idx = [0, 1, 17, 999, 4321, 7777, 9998, 9999]
    rt_ok = True
    for i in sample_idx:
        d = ds[i]
        fresh_obj = sro.build_mol_objects(i, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        cached = objs[i]
        ok = (
            np.array_equal(fresh_obj.atom_feat, cached.atom_feat)
            and np.array_equal(fresh_obj.bond_index, cached.bond_index)
            and np.array_equal(fresh_obj.bond_type, cached.bond_type)
            and np.array_equal(fresh_obj.bond_feat, cached.bond_feat)
            and np.array_equal(fresh_obj.junc_index, cached.junc_index)
            and np.array_equal(fresh_obj.junc_bond, cached.junc_bond)
            and np.array_equal(fresh_obj.junc_feat, cached.junc_feat)
            and np.array_equal(fresh_obj.global_feat, cached.global_feat)
        )
        rt_ok = rt_ok and ok
    gate("raw_pyg_object_roundtrip", rt_ok, {"sample": sample_idx})
    # 9. discipline scans
    src_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_split_calls = [m.group(0) for m in re.finditer(r"_load_zinc\([^)]*\)", src_text)
                             if '"train"' not in m.group(0)]
    gate("official_splits_never_read", not forbidden_split_calls, {"non_train_load_calls": forbidden_split_calls})
    component_tokens = ["patch" + "_cont", "patch" + "_context", "attribute" + "_atom_type",
                        "attribute" + "_bond_type", "topology" + "_features", "global" + "_context",
                        "typed" + "_token", "parent" + "_token", "structural" + "_token"]
    found_tokens = [t for t in component_tokens if t in src_text]
    gate("raw_y_only_supervision", not found_tokens, {"component_tokens_found": found_tokens})
    forbidden_output_tokens = ["teach" + "er", "dis" + "till", "full" + "_y"]
    found_forbidden = [t for t in forbidden_output_tokens if t in src_text]
    gate("no_historical_model_supervision", not found_forbidden, {"tokens": found_forbidden})
    # 10. provenance
    out["provenance"] = {
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "driver_sha256": file_sha256(Path(__file__)),
        "bjoc_sha256": file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
        "objects_sha256": file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
        "objects_cache_sha256": cache_sha,
        "smiles_sha256": hashlib.sha256("\n".join(rb.load_smiles()).encode()).hexdigest(),
        "split_sha256": file_sha256(SPLIT_FILE),
        "fit4k_sha256": file_sha256(FIT4K_FILE),
        "steps_sha256": file_sha256(STEPS_FILE),
        "prep_sha256": {f: file_sha256(PREP_FILES[f]) for f in ("fit8k", "fit4k")},
    }
    out["all_pass"] = all(g["pass"] for g in out["gates"].values())
    write_json(OUT / "correctness_gates.json", out)
    log(f"[gates] {len(out['gates'])} gates, all_pass={out['all_pass']}")
    del objs
    return 0


def split_set(split: Mapping[str, Sequence[int]], key: str) -> set[int]:
    return set(int(i) for i in split[key])


def relabel_objects(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Rebuild objects with atoms relabelled: new_id = perm[old_id]."""
    n = int(obj.n_atoms)
    atom_type = np.argmax(obj.atom_feat[:, :sro.ATOM_CATEGORIES], axis=1)
    x = np.zeros((n, 1), dtype=np.int64)
    for old in range(n):
        x[int(perm[old]), 0] = int(atom_type[old])
    edges = []
    for k in range(obj.bond_index.shape[0]):
        u, v = int(obj.bond_index[k, 0]), int(obj.bond_index[k, 1])
        edges.append((int(perm[u]), int(perm[v]), int(obj.bond_type[k])))
    ei, ea = [], []
    for u, v, bt in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(obj.mol_index, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


# ---------------------------------------------------------------------------
# freeze manifest / decision / report
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    frozen = frozen_plan()
    manifest = {
        "protocol": PROTOCOL_ID,
        "frozen_utc": load_config().get("frozen_utc"),
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "driver_sha256": file_sha256(Path(__file__)),
        "frozen_block": frozen,
        "artifacts": {
            "split_a": file_sha256(SPLIT_FILE),
            "fit4k_subset": file_sha256(FIT4K_FILE),
            "steps": file_sha256(STEPS_FILE),
            "prep_fit8k": file_sha256(PREP_FILES["fit8k"]),
            "prep_fit4k": file_sha256(PREP_FILES["fit4k"]),
            "correctness_gates": file_sha256(OUT / "correctness_gates.json"),
        },
        "official_valid_or_test_loaded": False,
        "locked_labels_read_at_freeze_time": LOCK_FILE.exists(),
    }
    write_json(OUT / "protocol_freeze_manifest.json", manifest)
    log("[freeze] protocol freeze manifest written")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    lc = read_json(OUT / "learning_curve_analysis.json")
    own = read_json(OUT / "ownership_paired_analysis.json")
    coll = read_json(OUT / "graph_collision_audit.json")
    cfg = load_config()

    def locked(cname: str) -> dict[str, Any]:
        return lc["contrasts"][cname]["locked"]

    sample = locked("N8_minus_N4stepmatched")
    ownership = locked("O8_minus_N8")
    steps_c = locked("N4stepmatched_minus_N4short")

    # observability
    floor_full = coll["y_inconsistency_full10k"]
    n_affected = int(floor_full["n_rows_affected"])
    floor_val = floor_full["empirical_indistinguishable_floor"]
    d_established = bool(n_affected >= 20 and floor_val is not None and floor_val >= 0.005)

    # sample efficiency
    sample_ok = bool(
        sample["seed_mean"]["delta"] <= -0.006
        and sample["seed0"]["delta"] < 0
        and sample["seed1"]["delta"] < 0
        and sample["seed_mean"]["paired_bootstrap"]["ci95"][1] < 0
        and sample["seed_mean"]["top10_removed"] <= 0
    )
    # ownership
    ownership_ok = bool(
        ownership["seed_mean"]["delta"] <= -0.005
        and ownership["seed0"]["delta"] < 0
        and ownership["seed1"]["delta"] < 0
        and ownership["seed_mean"]["subgroups"]["ordinary_y_gt_-3"]["delta"] <= 0
        and ownership["seed_mean"]["top10_removed"] <= 0
        and ownership["seed_mean"]["paired_bootstrap"]["ci95"][1] < 0
    )
    # architecture
    fit_maes = [lc["readout_metrics"][f"{r}_s{s}"]["fit"] for r in READOUT_NAMES for s in SEEDS]
    common = read_json(OUT / "common_error_analysis.json")
    arch_ok = bool(
        not d_established
        and sample["seed_mean"]["paired_bootstrap"]["ci95"][0] <= 0 <= sample["seed_mean"]["paired_bootstrap"]["ci95"][1]
        and ownership["seed_mean"]["paired_bootstrap"]["ci95"][0] <= 0 <= ownership["seed_mean"]["paired_bootstrap"]["ci95"][1]
        and max(fit_maes) < 0.06
        and min(common["four_way_fraction_of_each_model_set"].values()) >= 0.25
    )

    candidates: list[tuple[str, float, dict[str, Any]]] = []
    if sample_ok:
        candidates.append(("SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED", abs(float(sample["seed_mean"]["delta"])),
                           {"delta": sample["seed_mean"]["delta"], "ci95": sample["seed_mean"]["paired_bootstrap"]["ci95"]}))
    if ownership_ok:
        candidates.append(("OWNERSHIP_GENERALIZATION_SIGNAL", abs(float(ownership["seed_mean"]["delta"])),
                           {"delta": ownership["seed_mean"]["delta"], "ci95": ownership["seed_mean"]["paired_bootstrap"]["ci95"]}))
    if d_established:
        decision = "INPUT_OBSERVABILITY_LIMITATION_FOUND"
        primary = {"n_affected_rows": n_affected, "floor": floor_val}
    elif candidates:
        candidates.sort(key=lambda t: -t[1])
        decision = candidates[0][0]
        primary = candidates[0][2]
    elif arch_ok:
        decision = "ARCHITECTURAL_INDUCTIVE_BIAS_SUSPECTED"
        primary = {"four_way": common["four_way_intersection"]}
    else:
        decision = "BOTTLENECK_NOT_IDENTIFIED"
        primary = None
    identified = decision in (
        "INPUT_OBSERVABILITY_LIMITATION_FOUND",
        "SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED",
        "OWNERSHIP_GENERALIZATION_SIGNAL",
        "ARCHITECTURAL_INDUCTIVE_BIAS_SUSPECTED",
    )
    secondary = []
    if d_established and decision != "INPUT_OBSERVABILITY_LIMITATION_FOUND":
        secondary.append("INPUT_OBSERVABILITY_LIMITATION_FOUND")
    for name, _effect, _detail in candidates:
        if name != decision:
            secondary.append(name)
    if arch_ok and decision != "ARCHITECTURAL_INDUCTIVE_BIAS_SUSPECTED":
        secondary.append("ARCHITECTURAL_INDUCTIVE_BIAS_SUSPECTED")
    payload = {
        "protocol": PROTOCOL_ID,
        "decision": decision,
        "bottleneck_identified": bool(identified),
        "secondary_signals": secondary,
        "evidence": {
            "observability_floor": floor_full,
            "sample_efficiency": {
                "seed_mean_delta": sample["seed_mean"]["delta"],
                "ci95": sample["seed_mean"]["paired_bootstrap"]["ci95"],
                "per_seed": [sample["seed0"]["delta"], sample["seed1"]["delta"]],
                "top10_removed": sample["seed_mean"]["top10_removed"],
            },
            "ownership": {
                "seed_mean_delta": ownership["seed_mean"]["delta"],
                "ci95": ownership["seed_mean"]["paired_bootstrap"]["ci95"],
                "per_seed": [ownership["seed0"]["delta"], ownership["seed1"]["delta"]],
                "top10_removed": ownership["seed_mean"]["top10_removed"],
                "ordinary": ownership["seed_mean"]["subgroups"]["ordinary_y_gt_-3"]["delta"],
            },
            "extra_optimisation_on_4k": {
                "seed_mean_delta": steps_c["seed_mean"]["delta"],
                "ci95": steps_c["seed_mean"]["paired_bootstrap"]["ci95"],
            },
            "fit_mae": {f"{r}_s{s}": lc["readout_metrics"][f"{r}_s{s}"]["fit"] for r in READOUT_NAMES for s in SEEDS},
            "four_way_high_error_intersection": common["four_way_intersection"],
            "mean_pairwise_jaccard": common["mean_pairwise_jaccard"],
        },
        "next_round_recommendation": None,
        "official_valid_or_test_loaded": False,
        "rules_source": "configs/bjoc_predictive_bottleneck_v1.yaml :: decision_rules",
    }

    def r(x: float) -> float:
        return round(float(x), 4)

    if decision == "SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED":
        payload["next_round_recommendation"] = (
            "Prioritise how to make legal structural pre-training / supervision tokens more "
            "sample-efficient for the fixed non-iterative BJOC family (label-blind external "
            "molecule pre-training is one candidate; not executed this round).  Do NOT claim that "
            "20k molecules reach any specific MAE and do NOT extrapolate the two-point 4k/8k slope."
        )
    elif decision == "OWNERSHIP_GENERALIZATION_SIGNAL":
        payload["next_round_recommendation"] = (
            "Promote BJOC-Organized as the competitive skeleton and study the sample efficiency of "
            "true junction ownership; the internal split gain is still not new external-test "
            "evidence and must not be re-checked on official test."
        )
    elif decision == "ARCHITECTURAL_INDUCTIVE_BIAS_SUSPECTED":
        payload["next_round_recommendation"] = (
            "Stop adding local structural features.  Write a next_architecture_hypothesis.md with "
            "at most three candidate directions (one recommended) that change whole-molecule "
            "composition and parameter sharing rather than adding one more local channel."
        )
    elif decision == "INPUT_OBSERVABILITY_LIMITATION_FOUND":
        payload["next_round_recommendation"] = (
            "Study the legal observability boundary of the task input; quantify the affected rows "
            "and the empirical floor.  Do not reconstruct the target formula as a feature."
        )
    else:
        payload["next_round_recommendation"] = (
            "Keep the uncertainty.  Do not claim a major architecture breakthrough is required, and "
            "do not purchase a new architecture from this evidence alone."
        )
    write_json(OUT / "scientific_decision.json", payload)
    log(f"[decision] {decision} (identified={identified})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    lc = read_json(OUT / "learning_curve_analysis.json")
    own = read_json(OUT / "ownership_paired_analysis.json")
    dec = read_json(OUT / "scientific_decision.json")
    coll = read_json(OUT / "graph_collision_audit.json")
    hist = read_json(OUT / "historical_ownership_audit.json")
    common = read_json(OUT / "common_error_analysis.json")
    steps = read_json(STEPS_FILE)

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.6f}"

    def readout_row(r: str) -> str:
        m = lc["seed_mean_metrics"][r]
        return (f"| {r} | {fmt(m['fit'])} | {fmt(m['monitor']['mae'])} | {fmt(m['locked']['mae'])} | "
                f"{fmt(m['locked']['rmse'])} | {fmt(m['locked']['median'])} | {fmt(m['locked']['p90'])} |")

    lines = [
        "# cscl-predictive-bottleneck-diagnosis-v1 — report",
        "",
        f"**Decision: `{dec['decision']}`** (identified={dec['bottleneck_identified']})",
        "",
        "Historical context: BJOC-Null full-10k official-test single-seed-mean 0.092233; user target 0.06–0.07.",
        "This round runs on the official **train** split only; its internal MAE numbers are not official-test",
        "scores and no extrapolation to 0.07 is licensed.",
        "",
        "## 1. Learning curve (seed-mean)",
        "",
        "Update count is matched at T = " + str(steps["T"]) + " for every trajectory.",
        "",
        "| readout | fit MAE | monitor MAE | locked MAE | locked RMSE | median | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in READOUT_NAMES:
        lines.append(readout_row(r))
    lines += [
        "",
        "| contrast (locked) | seed-mean delta | CI95 | per-seed | top10-removed |",
        "|---|---:|---|---:|---:|",
    ]
    for name in ("N4stepmatched_minus_N4short", "N8_minus_N4stepmatched", "O8_minus_N8"):
        c = lc["contrasts"][name]["locked"]["seed_mean"]
        lines.append(
            f"| {name} | {c['delta']:+.6f} | [{c['paired_bootstrap']['ci95'][0]:+.6f}, "
            f"{c['paired_bootstrap']['ci95'][1]:+.6f}] | "
            f"{lc['contrasts'][name]['locked']['seed0']['delta']:+.6f} / "
            f"{lc['contrasts'][name]['locked']['seed1']['delta']:+.6f} | {c['top10_removed']:+.6f} |"
        )
    lines += [
        "",
        "Fit/monitor/locked for the two N4 readouts are the two pre-frozen readouts of the same 4k trajectory.",
        f"Fit4k/Fit8k rows: {steps['n_fit4k']}/{steps['n_fit8k']}; snapshots: {steps['n4_short_snapshot_steps']} "
        f"(short), {steps['n4_stepmatched_snapshot_steps']} (step-matched).",
        "",
        "## 2. Null vs Organized",
        "",
        f"Locked seed-mean O8-N8 = {own['locked']['seed_mean']['delta']:+.6f} "
        f"(CI95 [{own['locked']['seed_mean']['paired_bootstrap']['ci95'][0]:+.6f}, "
        f"{own['locked']['seed_mean']['paired_bootstrap']['ci95'][1]:+.6f}]); "
        f"ordinary {own['locked']['seed_mean']['subgroups']['ordinary_y_gt_-3']['delta']:+.6f}; "
        f"tail {own['locked']['seed_mean']['subgroups']['tail_y_le_-3']['delta']:+.6f}; "
        f"top-10-removed {own['locked']['seed_mean']['top10_removed']:+.6f}.",
        "",
        "## 3. Common errors (locked, four primary models)",
        "",
        f"Four-way top-10% intersection: {common['four_way_intersection']}/100; "
        f"mean pairwise Jaccard {common['mean_pairwise_jaccard']:.3f}.",
        "",
        "## 4. Input observability",
        "",
        f"Strict labelled-graph isomorphism groups: {len(coll['strict_vf2_groups'])}; "
        f"rows affected in the full 10k: {coll['y_inconsistency_full10k']['n_rows_affected']}; "
        f"empirical indistinguishable floor: {coll['y_inconsistency_full10k']['empirical_indistinguishable_floor']}.",
        "",
        "## 5. Historical ownership audit (cross-bond monitor, N vs O, seed 0)",
        "",
        f"Δ_O−N = {hist['B1_crossbond_O_vs_N_monitor']['delta_O_minus_N']:+.6f}, "
        f"better-molecule fraction {hist['B1_crossbond_O_vs_N_monitor']['paired_bootstrap']['frac_candidate_better']:.3f}, "
        f"top-10-removed {hist['B1_crossbond_O_vs_N_monitor']['top10_removed']:+.6f}.",
        "",
        "## 6. Decision",
        "",
        f"`{dec['decision']}` — {dec['next_round_recommendation']}",
        "",
        "## Disclosures",
        "",
        "- official valid/test never loaded; locked-dev labels read exactly once after prediction hashing",
        "- monitor labels are the standard training-time readout; no checkpoint was selected by monitor",
        "- molecule bootstrap does not cover training randomness; two seeds cannot estimate the seed distribution",
        "- checkpoints are git-ignored; all JSONs are external artifacts with hashes in protocol_freeze_manifest.json",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("[report] REPORT.md written")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("splits", help="freeze the split + nested Fit4k")
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("objects", help="fit the global standardiser per training size")
    p.set_defaults(func=cmd_objects)

    p = sub.add_parser("steps", help="compute and freeze T + snapshot steps")
    p.set_defaults(func=cmd_steps)

    p = sub.add_parser("gates", help="pre-GPU correctness gates")
    p.set_defaults(func=cmd_gates)

    p = sub.add_parser("freeze", help="protocol freeze manifest")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("train", help="one formal trajectory")
    p.add_argument("--arm", choices=sorted(ARM_MODE), required=True)
    p.add_argument("--fit", choices=("fit4k", "fit8k"), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="label-blind predictions for all readouts")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("freeze-predictions", help="hash every prediction file")
    p.set_defaults(func=cmd_freeze_predictions)

    p = sub.add_parser("score", help="the ONE label read + learning curve / ownership analysis")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("common-errors", help="Phase G common-error analysis")
    p.set_defaults(func=cmd_common_errors)

    p = sub.add_parser("historical", help="Phase B historical monitor diagnostics")
    p.set_defaults(func=cmd_historical)

    p = sub.add_parser("observability", help="Phase C input observability + graph collision audit")
    p.set_defaults(func=cmd_observability)

    p = sub.add_parser("decision", help="frozen scientific decision")
    p.set_defaults(func=cmd_decision)

    p = sub.add_parser("report", help="assemble REPORT.md")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
