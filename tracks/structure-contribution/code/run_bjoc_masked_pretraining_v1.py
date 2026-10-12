"""cscl-bjoc-masked-pretraining-v1 driver.

Protocol: ``configs/bjoc_masked_pretraining_v1.yaml`` (frozen before any GPU
trajectory).  Track: ``tracks/structure-contribution``.

Single scientific question
--------------------------
Without extra y labels, without extra molecules and without changing the
downstream BJOC-Null architecture/recipe, does strictly leak-free masked
atom/bond structural pretraining of the three BJOC encoders improve the way
the model uses the existing 8k labelled molecules?

Design (all within official train 10000)
----------------------------------------
* new canonical-SMILES grouped split ``cscl-bjoc-maskpretrain-v1-A``
  (Fit8k 8000 / Monitor 1000 / Locked-dev 1000);
* one pretext task only: 15% raw atom-type + 15% raw physical bond-type
  masking, mask tokens 27/0, full rebuild of every BJOC feature from the
  masked raw graph, ``L_SSL = CE_atom + CE_bond`` (1.0/1.0), 7560 updates;
* downstream arms, all 413,505 params, raw-y L1, Adam 1e-3/1e-5, batch 128,
  clip 5, last-5 soup:
    - B: scratch, 15120 supervised updates (240 epochs);
    - P: same init, three encoders overwritten by the frozen pretrained
      encoder state, 15120 supervised updates;
    - C (only if the frozen monitor purchase gate fires): scratch, 22680
      supervised updates (360 epochs) - the extra-supervision control.

Stages
------
``splits``             freeze the grouped split (labels never read)
``objects``            structure-only object caches + Fit8k global standardiser
``audit-dependency``   masked feature dependency map + mask-token scan
``audit-swap``         target-swap non-leakage test (witness + random)
``rebuild-audit``      full-rebuild consistency vs the dependency map
``feasibility``        CPU throughput / memory / class-prior audit
``bench``              measured CPU data-path timing
``gates``              the pre-GPU correctness gates (20) + leakage gates
``freeze``             training manifest (hashes of code/config/split/caches)
``pretrain``           masked pretraining, seed 0/1 (no y anywhere)
``train``              B / P / C trajectory, seed 0/1
``predict``            label-blind monitor + locked predictions (all arms)
``freeze-predictions`` hash every prediction file before the locked read
``gate-monitor``       B vs P monitor readout + compute-control purchase gate
``score``              the ONE locked-dev label read + metrics + bootstrap
``mechanism``          frozen-checkpoint transfer-mechanism diagnostics
``decision``           the single frozen scientific decision
``phase-f``            remote chain: objects + pretrain x2 + B x2 + P x2 + predict
``phase-g``            remote chain: C x2 + predict (only after the gate fires)

Discipline
----------
* official valid / official test are never loaded.
* fit y are used for training only; monitor y are the standard readout and the
  purchase-gate read; locked y are read exactly once inside ``score`` after
  every prediction file is hashed.
* pretraining never loads the encoded-y cache and never touches ``.y``; only
  ``x`` / ``edge_index`` / ``edge_attr`` of Fit8k rows are used.
* no GNN / Transformer / attention / iterative message passing; no component
  supervision; no teacher; no external data; no full-10k final training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import platform
import random
import re
import resource
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
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-masked-pretraining-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_masked_pretraining_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1"

# ---- frozen split ----------------------------------------------------------
SPLIT_SALT = "cscl-bjoc-maskpretrain-v1-A"
FIT_TARGET = 8000
MONITOR_TARGET = 1000
SEEDS = (0, 1)

# ---- frozen downstream recipe ---------------------------------------------
BATCH_SIZE = 128
EPOCHS_BASE = 240
EPOCHS_CONTROL = 360
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
SUPERVISED_UPDATES_B = 240 * 63
SUPERVISED_UPDATES_C = 360 * 63

# ---- frozen analysis constants --------------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TOP_K_REMOVED = 10
TAIL_THRESHOLD = -3.0

ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"

# ---- artifacts -------------------------------------------------------------
SPLIT_JSON = OUT / "split_a.json"
OBJECTS_FIT8K = OUT / "objects_fit8k.pt"
OBJECTS_MONITOR = OUT / "objects_monitor.pt"
OBJECTS_LOCKED = OUT / "objects_locked.pt"
PREP_FIT8K = OUT / "prep_fit8k.json"
OBJECTS_MANIFEST = OUT / "objects_manifest.json"
DEPENDENCY_AUDIT = OUT / "masked_feature_dependency_audit.json"
SWAP_AUDIT = OUT / "mask_target_swap_audit.json"
REBUILD_AUDIT = OUT / "masked_object_rebuild_audit.json"
FEASIBILITY_AUDIT = OUT / "pretraining_feasibility_audit.json"
GATES_JSON = OUT / "pretext_leakage_gates.json"
TRAINING_MANIFEST = OUT / "training_manifest.json"
INIT_AUDIT = OUT / "initialization_audit.json"
MONITOR_GATE = OUT / "compute_control_gate.json"
SCRATCH_VS_PRETRAINED = OUT / "scratch_vs_pretrained_monitor.json"
PRED_DIR = OUT / "predictions"
PRED_MANIFEST = OUT / "locked_prediction_manifest.json"
LOCKED_METRICS = OUT / "locked_metrics.json"
PAIRED_BOOTSTRAP = OUT / "paired_bootstrap.json"
MONITOR_METRICS = OUT / "monitor_metrics.json"
TRANSFER_MECHANISM = OUT / "transfer_mechanism.json"
SCIENTIFIC_DECISION = OUT / "scientific_decision.json"
LOCK_FILE = OUT / "locked_read.lock"
COMPUTE_CONTROL_RESULT = OUT / "compute_control_result.json"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# generic io / hashing helpers
# ---------------------------------------------------------------------------


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_short(path: Path) -> str:
    return file_sha256(path)[:16]


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def state_group_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n_rows = len(delta)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        means[i] = delta[idx].mean()
    return {
        "mean": float(delta.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "frac_better": float((delta < 0).mean()),
        "n_resamples": int(n),
        "seed": int(seed),
        "unit": "molecule",
    }


def metrics(prefix: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - y)
    return {
        f"{prefix}_mae": float(err.mean()),
        f"{prefix}_median_abs": float(np.median(err)),
        f"{prefix}_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        f"{prefix}_p90_abs": float(np.quantile(err, 0.90)),
        f"{prefix}_p95_abs": float(np.quantile(err, 0.95)),
    }


def top_k_removed(delta: np.ndarray, e_new: np.ndarray, e_ref: np.ndarray, k: int = TOP_K_REMOVED) -> float:
    worst = np.argsort(-np.maximum(e_new, e_ref))[:k]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    return float(delta[keep].mean())


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def build_split(smiles: Sequence[str], salt: str) -> dict[str, Any]:
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
    out = {
        "salt": salt,
        "n_groups": len(groups),
        "fit": np.asarray(sorted(fit), dtype=np.int64),
        "monitor": np.asarray(sorted(monitor), dtype=np.int64),
        "locked": np.asarray(sorted(locked), dtype=np.int64),
        "duplicate_groups": {s: sorted(r) for s, r in groups.items() if len(r) > 1},
    }
    return out


def read_split() -> dict[str, Any]:
    payload = read_json(SPLIT_JSON)
    return {
        "salt": payload["split_salt"],
        "fit8k": np.asarray(payload["fit_rows"], dtype=np.int64),
        "monitor": np.asarray(payload["monitor_rows"], dtype=np.int64),
        "locked": np.asarray(payload["locked_rows"], dtype=np.int64),
    }


def verify_split_rows(split: Mapping[str, Any]) -> None:
    assert int(len(split["fit8k"])) == 8000, len(split["fit8k"])
    assert int(len(split["monitor"])) == 1000, len(split["monitor"])
    assert int(len(split["locked"])) == 1000, len(split["locked"])
    union = set(int(i) for i in split["fit8k"]) | set(int(i) for i in split["monitor"]) | set(int(i) for i in split["locked"])
    assert len(union) == 10000, len(union)


def cmd_splits(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    assert len(smiles) == 10000
    split = build_split(smiles, SPLIT_SALT)
    fit, monitor, locked = split["fit"], split["monitor"], split["locked"]
    dup = split["duplicate_groups"]
    dup_rows = sum(len(v) for v in dup.values())
    # duplicates never cross sides by construction (grouped split)
    side_of: dict[int, str] = {}
    for name, rows in (("fit", fit), ("monitor", monitor), ("locked", locked)):
        for r in rows:
            side_of[int(r)] = name
    cross = 0
    for rows in dup.values():
        if len({side_of[int(r)] for r in rows}) > 1:
            cross += 1
    assert cross == 0, cross
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_salt": SPLIT_SALT,
        "n_groups": split["n_groups"],
        "n_fit8k": int(len(fit)),
        "n_monitor": int(len(monitor)),
        "n_locked": int(len(locked)),
        "fit_rows": [int(x) for x in fit],
        "monitor_rows": [int(x) for x in monitor],
        "locked_rows": [int(x) for x in locked],
        "fit_sha256_rows": rows_sha256(fit),
        "monitor_sha256_rows": rows_sha256(monitor),
        "locked_sha256_rows": rows_sha256(locked),
        "overlap": {
            "fit_monitor": int(len(set(map(int, fit)) & set(map(int, monitor)))),
            "fit_locked": int(len(set(map(int, fit)) & set(map(int, locked)))),
            "monitor_locked": int(len(set(map(int, monitor)) & set(map(int, locked)))),
            "union": int(len(set(map(int, fit)) | set(map(int, monitor)) | set(map(int, locked)))),
        },
        "duplicate_canonical_smiles_groups": len(dup),
        "rows_in_duplicate_groups": int(dup_rows),
        "duplicate_groups_cross_sides": int(cross),
        "duplicate_group_examples": [{"smiles": s, "rows": r} for s, r in list(dup.items())[:5]],
        "label_information_used": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(SPLIT_JSON, payload)
    log(f"[splits] fit={len(fit)} monitor={len(monitor)} locked={len(locked)} dup_groups={len(dup)} cross={cross}")
    log(f"[splits] fit rows sha {payload['fit_sha256_rows'][:16]} monitor {payload['monitor_sha256_rows'][:16]} locked {payload['locked_sha256_rows'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# objects + Fit8k standardiser
# ---------------------------------------------------------------------------


def load_raw_dataset():
    return cf.load_official_train()


def build_objects_for_rows(rows: Sequence[int], ds) -> list[sro.MolObjects]:
    objs = []
    for i in rows:
        d = ds[int(i)]
        objs.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
    return objs


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split = build_split([str(s) for s in cf.load_canonical_smiles()], SPLIT_SALT)
    verify_split_rows({"fit8k": split["fit"], "monitor": split["monitor"], "locked": split["locked"]})
    ds = load_raw_dataset()
    assert len(ds) == 10000

    t0 = time.perf_counter()
    fit_objs = build_objects_for_rows(split["fit"], ds)
    mon_objs = build_objects_for_rows(split["monitor"], ds)
    lock_objs = build_objects_for_rows(split["locked"], ds)
    build_seconds = time.perf_counter() - t0
    assert [o.mol_index for o in fit_objs] == [int(i) for i in split["fit"]]
    assert [o.mol_index for o in mon_objs] == [int(i) for i in split["monitor"]]
    assert [o.mol_index for o in lock_objs] == [int(i) for i in split["locked"]]
    torch.save(fit_objs, OBJECTS_FIT8K)
    torch.save(mon_objs, OBJECTS_MONITOR)
    torch.save(lock_objs, OBJECTS_LOCKED)

    gfit = np.stack([o.global_feat for o in fit_objs], 0).astype(np.float64)
    mean = gfit.mean(0)
    scale = gfit.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    write_json(
        PREP_FIT8K,
        {
            "protocol": PROTOCOL_ID,
            "fit_only": True,
            "n_fit": int(len(split["fit"])),
            "global_mean": mean.tolist(),
            "global_scale": scale.tolist(),
            "names": list(sro.GLOBAL_FEAT_NAMES),
        },
    )
    if not SPLIT_JSON.exists():
        cmd_splits(args)

    # row alignment between the raw PyG dataset and the encoded-y cache is
    # verified on fit + monitor rows ONLY; locked y is not read before ``score``.
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    assert len(enc) == 10000, len(enc)
    check_rows = list(map(int, split["fit"][:200])) + list(map(int, split["monitor"]))
    max_diff = 0.0
    for i in check_rows:
        yr = float(ds[i].y.reshape(-1)[0])
        ye = float(enc[i].y.reshape(-1)[0])
        max_diff = max(max_diff, abs(yr - ye))
    assert max_diff < 1e-9, max_diff
    del enc

    counts = {
        "n_atoms": int(sum(o.n_atoms for o in fit_objs)),
        "n_bonds": int(sum(o.bond_index.shape[0] for o in fit_objs)),
        "n_junctions": int(sum(o.junc_index.shape[0] for o in fit_objs)),
    }
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "build_seconds": build_seconds,
        "fit_rows_sha256": rows_sha256(split["fit"]),
        "objects_fit8k_sha256": file_sha256(OBJECTS_FIT8K),
        "objects_monitor_sha256": file_sha256(OBJECTS_MONITOR),
        "objects_locked_sha256": file_sha256(OBJECTS_LOCKED),
        "prep_fit8k_sha256": file_sha256(PREP_FIT8K),
        "fit_counts": counts,
        "alignment_rows_checked": len(check_rows),
        "alignment_max_abs_diff": max_diff,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OBJECTS_MANIFEST, manifest)
    log(f"[objects] fit8k {counts} build {build_seconds:.1f}s; alignment max|dy|={max_diff:.2e}")
    log(f"[objects] fit8k sha {manifest['objects_fit8k_sha256'][:16]} prep sha {manifest['prep_fit8k_sha256'][:16]}")
    return 0


def load_objects(path: Path) -> list[sro.MolObjects]:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Phase A audits
# ---------------------------------------------------------------------------


def _small_molecule_objects() -> list[tuple[sro.MolObjects, np.ndarray, np.ndarray, np.ndarray]]:
    """A deterministic minimal molecule set for the leakage audits."""
    specs = [
        ([0, 1], [(0, 1, 2)]),
        ([0, 1, 2], [(0, 1, 1), (1, 2, 2), (2, 0, 1)]),
        ([3, 4, 5, 6], [(0, 1, 1), (1, 2, 3), (2, 3, 1), (1, 3, 2)]),
        ([7, 8, 9, 10, 11], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 0, 1)]),
    ]
    out = []
    for mol_id, (x, edges) in enumerate(specs):
        xa = np.asarray(x, dtype=np.int64).reshape(-1, 1)
        ei = []
        ea = []
        for u, v, bt in edges:
            ei.extend([(u, v), (v, u)])
            ea.extend([bt, bt])
        eia = np.asarray(ei, dtype=np.int64).T
        eaa = np.asarray(ea, dtype=np.int64)
        obj = sro.build_mol_objects(mol_id, xa, eia, eaa)
        out.append((obj, xa, eia, eaa))
    return out


def _diff_cells(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.asarray(a) != np.asarray(b)).sum())


def cmd_audit_dependency(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # mask-token scan on the frozen Fit8k rows
    split = build_split([str(s) for s in cf.load_canonical_smiles()], SPLIT_SALT)
    ds = load_raw_dataset()
    atom_codes: set[int] = set()
    bond_codes: set[int] = set()
    for i in split["fit"]:
        atom_codes |= {int(v) for v in ds[int(i)].x.numpy().reshape(-1)}
        bond_codes |= {int(v) for v in ds[int(i)].edge_attr.numpy().reshape(-1)}
    atom_conflict = mp.ATOM_MASK_TOKEN in atom_codes
    bond_conflict = mp.BOND_MASK_TOKEN in bond_codes
    blocks = sro.atom_blocks()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "feature_layout": {
            "atom_dim": sro.ATOM_FEAT_DIM,
            "atom_blocks": {k: [v.start, v.stop] for k, v in blocks.items()},
            "bond_dim": sro.BOND_FEAT_DIM,
            "bond_blocks": {
                "own_type": [0, 4],
                "is_ring": [4, 5],
                "exactly_one_in_ring": [5, 6],
                "both_in_ring": [6, 7],
            },
            "junction_dim": sro.JUNC_FEAT_DIM,
            "junction_type_sum": [12, 16],
            "junction_type_outer": [16, 32],
            "global_dim": sro.GLOBAL_DIM,
            "global_bond_type_fractions": [19, 23],
            "global_distinct_bond_types": 31,
            "global_distinct_atom_types": 32,
        },
        "known_leak_paths_before_masking": {
            "atom_type": [
                "own atom-type one-hot (28)",
                "radius-1 neighbour-type histogram (28)",
                "radius-2 atom-type histogram (28)",
                "global distinct atom-type count (1)",
            ],
            "bond_type": [
                "own bond-type one-hot (4)",
                "endpoint atom incident bond-type histograms (4 each)",
                "junction symmetric type sum (4)",
                "junction symmetric type outer sum (16)",
                "global bond-type fractions (4)",
                "global distinct bond-type count (1)",
            ],
            "consequence": (
                "zeroing only the masked position's own one-hot is ILLEGAL: the hidden type "
                "would remain readable from neighbour histograms, radius-2 histograms, "
                "junction type channels and global distinct/ratio features"
            ),
        },
        "mask_token_scan_fit8k": {
            "atom_mask_token": mp.ATOM_MASK_TOKEN,
            "bond_mask_token": mp.BOND_MASK_TOKEN,
            "atom_codes_present": sorted(atom_codes),
            "bond_codes_present": sorted(bond_codes),
            "atom_token_conflict": bool(atom_conflict),
            "bond_token_conflict": bool(bond_conflict),
            "n_rows_scanned": int(len(split["fit"])),
        },
        "construction_rule": (
            "replace the raw atom/bond codes by the mask tokens on a copy of the raw PyG "
            "arrays, then rebuild every BJOC object field with "
            "structural_relation_objects_v1.build_mol_objects (full rebuild, no incremental "
            "feature patch); each physical bond replaces BOTH directed PyG records"
        ),
        "verdict": "MASK_TARGET_LEAKAGE_BLOCKED" if (atom_conflict or bond_conflict) else "MASK_TOKEN_SLOTS_FREE",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(DEPENDENCY_AUDIT, payload)
    log(f"[audit-dependency] atom codes {sorted(atom_codes)}; bond codes {sorted(bond_codes)}")
    log(f"[audit-dependency] verdict {payload['verdict']}")
    if atom_conflict or bond_conflict:
        raise SystemExit("MASK_TARGET_LEAKAGE_BLOCKED: mask token collides with a real training target")
    return 0


def _swap_case(sample_a: mp.PretextSample, sample_b: mp.PretextSample) -> dict[str, Any]:
    fields = {}
    for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
        fields[name] = bool(np.array_equal(getattr(sample_a.masked, name), getattr(sample_b.masked, name)))
    fields["bond_index"] = bool(np.array_equal(sample_a.masked.bond_index, sample_b.masked.bond_index))
    fields["junc_index"] = bool(np.array_equal(sample_a.masked.junc_index, sample_b.masked.junc_index))
    fields["junc_bond"] = bool(np.array_equal(sample_a.masked.junc_bond, sample_b.masked.junc_bond))
    fields["mask_ids"] = bool(
        np.array_equal(sample_a.atom_ids, sample_b.atom_ids) and np.array_equal(sample_a.bond_ids, sample_b.bond_ids)
    )
    return fields


def _pretext_logits_equal(sample_a: mp.PretextSample, sample_b: mp.PretextSample) -> dict[str, Any]:
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=123,
        organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    pretext = mp.PretextModel(model)
    ba, at_a, bt_a, _, _ = mp.batch_pretext_samples([sample_a])
    bb, at_b, bt_b, _, _ = mp.batch_pretext_samples([sample_b])
    with torch.no_grad():
        oa = pretext(ba, ba["masked_atom_ids"], ba["masked_bond_ids"])
        ob = pretext(bb, bb["masked_atom_ids"], bb["masked_bond_ids"])
    return {
        "atom_logits_equal": bool(torch.equal(oa["atom_logits"], ob["atom_logits"])),
        "bond_logits_equal": bool(torch.equal(oa["bond_logits"], ob["bond_logits"])),
        "atom_ctx_equal": bool(torch.equal(oa["atom_ctx"], ob["atom_ctx"])),
        "bond_ctx_equal": bool(torch.equal(oa["bond_ctx"], ob["bond_ctx"])),
    }


def cmd_audit_swap(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # -- deterministic minimal witness -------------------------------------
    witness: dict[str, Any] = {}
    obj, x, ei, ea = _small_molecule_objects()[0]
    a0 = mp.build_masked_sample(obj, x, ei, ea, seed=0, visit_index=0)
    # hidden atom 0: change truth 0 -> 5
    x_b = x.copy()
    x_b[a0.atom_ids[0], 0] = (int(x[a0.atom_ids[0], 0]) + 5) % 27
    a1 = mp.build_masked_sample(obj, x_b, ei, ea, seed=0, visit_index=0)
    witness["atom_swap"] = {
        "molecule": "2 atoms, single bond type 2",
        "mask": {"atoms": a0.atom_ids.tolist(), "bonds": a0.bond_ids.tolist()},
        "target_before": a0.atom_types.tolist(),
        "target_after": a1.atom_types.tolist(),
        "features_identical": _swap_case(a0, a1),
        "pretext_inputs_identical": _pretext_logits_equal(a0, a1),
    }
    # hidden bond: change truth 2 -> 3
    base_obj = sro.build_mol_objects(0, x, ei, ea)
    ea_b = ea.copy()
    for col in range(ei.shape[1]):
        u, v = int(ei[0, col]), int(ei[1, col])
        if (min(u, v), max(u, v)) == (0, 1):
            ea_b[col] = 3
    b0 = mp.build_masked_sample(base_obj, x, ei, ea, seed=0, visit_index=0)
    base_variant = sro.build_mol_objects(0, x, ei, ea_b)
    b1 = mp.build_masked_sample(base_variant, x, ei, ea_b, seed=0, visit_index=0)
    witness["bond_swap"] = {
        "target_before": b0.bond_types.tolist(),
        "target_after": b1.bond_types.tolist(),
        "features_identical": _swap_case(b0, b1),
        "pretext_inputs_identical": _pretext_logits_equal(b0, b1),
    }
    # -- randomized swap test on real Fit8k molecules -----------------------
    split = read_split()
    ds = load_raw_dataset()
    rng = np.random.default_rng(20261013)
    atom_pass = bond_pass = 0
    atom_cases = bond_cases = 0
    visible_leak_checked = 0
    visible_leak_ok = True
    fit = list(map(int, split["fit8k"]))
    for trial in range(args.random_cases):
        i = int(rng.choice(fit))
        d = ds[i]
        x_r = d.x.numpy()
        ei_r = d.edge_index.numpy()
        ea_r = d.edge_attr.numpy()
        obj_r = sro.build_mol_objects(i, x_r, ei_r, ea_r)
        if obj_r.n_atoms < 2 or obj_r.bond_index.shape[0] < 1:
            continue
        s0 = mp.build_masked_sample(obj_r, x_r, ei_r, ea_r, seed=7, visit_index=trial)
        # atom swap
        if len(s0.atom_ids):
            x_b2 = x_r.copy().reshape(-1)
            aid = int(s0.atom_ids[0])
            old = int(x_b2[aid])
            new = (old + 1) % 27
            if new == mp.ATOM_MASK_TOKEN:
                new = (old + 2) % 27
            x_b2[aid] = new
            s1 = mp.build_masked_sample(obj_r, x_b2.reshape(x_r.shape), ei_r, ea_r, seed=7, visit_index=trial)
            atom_cases += 1
            if all(_swap_case(s0, s1).values()):
                atom_pass += 1
        # bond swap
        if len(s0.bond_ids):
            vid = int(s0.bond_ids[0])
            old = int(obj_r.bond_type[vid])
            new = 3 if old != 3 else 1
            ea_b2 = ea_r.copy()
            u, v = int(obj_r.bond_index[vid][0]), int(obj_r.bond_index[vid][1])
            for col in range(ei_r.shape[1]):
                a, b = int(ei_r[0, col]), int(ei_r[1, col])
                if (min(a, b), max(a, b)) == (min(u, v), max(u, v)):
                    ea_b2[col] = new
            obj_variant = sro.build_mol_objects(i, x_r, ei_r, ea_b2)
            s2 = mp.build_masked_sample(obj_variant, x_r, ei_r, ea_b2, seed=7, visit_index=trial)
            bond_cases += 1
            if all(_swap_case(s0, s2).values()):
                bond_pass += 1
        # positive control: changing a VISIBLE atom must change features inside
        # exactly its dependency cells (the auditor is not blind)
        if trial < args.positive_cases:
            n_visible = [a for a in range(obj_r.n_atoms) if a not in set(map(int, s0.atom_ids))]
            if n_visible and obj_r.n_atoms >= 2:
                v = int(rng.choice(n_visible))
                x_v = x_r.copy().reshape(-1)
                x_v[v] = (int(x_v[v]) + 1) % 27
                if int(x_v[v]) == mp.ATOM_MASK_TOKEN:
                    x_v[v] = (int(x_v[v]) + 1) % 27
                s3 = mp.build_masked_sample(obj_r, x_v.reshape(x_r.shape), ei_r, ea_r, seed=7, visit_index=trial)
                dep = mp.dependent_cell_mask(s0.masked, [v], [])
                diff_atom = np.asarray(s0.masked.atom_feat) != np.asarray(s3.masked.atom_feat)
                changed = diff_atom.any(axis=1)
                unexpected = int((changed & ~dep["atom"].any(axis=1)).sum())
                if unexpected or int(changed.sum()) == 0:
                    visible_leak_ok = False
                visible_leak_checked += 1
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "question": "does changing the HIDDEN truth at a masked position change any model input?",
        "minimal_deterministic_witness": witness,
        "randomized_real_fit8k": {
            "seed": 7,
            "n_trials": int(args.random_cases),
            "atom_swap_cases": atom_cases,
            "atom_swap_passed": atom_pass,
            "bond_swap_cases": bond_cases,
            "bond_swap_passed": bond_pass,
            "features_identical_all": atom_pass == atom_cases and bond_pass == bond_cases and atom_cases > 0 and bond_cases > 0,
        },
        "positive_control_visible_change": {
            "cases": visible_leak_checked,
            "features_change_confined_to_dependency_cells": bool(visible_leak_ok),
        },
        "verdict": (
            "MASK_SWAP_INVARIANT"
            if (atom_pass == atom_cases and bond_pass == bond_cases and atom_cases > 0 and bond_cases > 0 and visible_leak_ok)
            else "MASK_TARGET_LEAKAGE_BLOCKED"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(SWAP_AUDIT, payload)
    log(f"[audit-swap] atom {atom_pass}/{atom_cases} bond {bond_pass}/{bond_cases} verdict {payload['verdict']}")
    if payload["verdict"] != "MASK_SWAP_INVARIANT":
        raise SystemExit("MASK_TARGET_LEAKAGE_BLOCKED")
    return 0


def cmd_rebuild_audit(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split = read_split()
    ds = load_raw_dataset()
    rng = np.random.default_rng(11)
    worst_subset_ok = True
    own_slot_ok = True
    determinism_ok = True
    hidden_absent_ok = True
    bidi_ok = True
    checked = 0
    max_ratio_gap = 0.0
    fit = list(map(int, split["fit8k"]))
    for trial in range(args.cases):
        i = int(rng.choice(fit))
        d = ds[i]
        x_r = d.x.numpy()
        ei_r = d.edge_index.numpy()
        ea_r = d.edge_attr.numpy()
        base = sro.build_mol_objects(i, x_r, ei_r, ea_r)
        s = mp.build_masked_sample(base, x_r, ei_r, ea_r, seed=3, visit_index=trial)
        # 1. deterministic reproduction
        s_again = mp.build_masked_sample(base, x_r, ei_r, ea_r, seed=3, visit_index=trial)
        for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
            if not np.array_equal(getattr(s.masked, name), getattr(s_again.masked, name)):
                determinism_ok = False
        # 2. differences vs the unmasked object stay inside the dependency map
        dep = mp.dependent_cell_mask(base, s.atom_ids, s.bond_ids)
        checks = [
            ("atom", base.atom_feat, s.masked.atom_feat, dep["atom"]),
            ("bond", base.bond_feat, s.masked.bond_feat, dep["bond"]),
            ("junc", base.junc_feat, s.masked.junc_feat, dep["junc"]),
            ("global", base.global_feat, s.masked.global_feat, dep["global"]),
        ]
        for name, before, after, allowed in checks:
            diff = np.asarray(before) != np.asarray(after)
            if not np.all(diff <= allowed):
                worst_subset_ok = False
                log(f"[rebuild] unexpected change in {name} molecule {i}")
        # 3. masked own slots hold the mask token, never the hidden truth
        for a, t in zip(s.atom_ids, s.atom_types):
            if int(s.masked.atom_feat[int(a), : sro.ATOM_CATEGORIES].argmax()) != mp.ATOM_MASK_TOKEN:
                own_slot_ok = False
            if s.masked.atom_feat[int(a), int(t)] != 0.0:
                hidden_absent_ok = False
        for b, t in zip(s.bond_ids, s.bond_types):
            if int(s.masked.bond_feat[int(b), : sro.BOND_CATEGORIES].argmax()) != mp.BOND_MASK_TOKEN:
                own_slot_ok = False
            if s.masked.bond_feat[int(b), int(t)] != 0.0:
                hidden_absent_ok = False
        # 4. both directed records of every masked physical bond replaced
        _, ea_masked = mp.mask_raw_arrays(x_r, ei_r, ea_r, base.bond_index, s.atom_ids, s.bond_ids)
        x_masked = np.asarray(x_r).copy().reshape(-1).astype(np.int64)
        x_masked[s.atom_ids] = mp.ATOM_MASK_TOKEN
        mapping = mp.directed_bond_index(ei_r)
        masked_bonds = set(int(z) for z in s.bond_ids)
        for k in range(base.bond_index.shape[0]):
            u, v = int(base.bond_index[k][0]), int(base.bond_index[k][1])
            cols = mapping[(min(u, v), max(u, v))]
            for col in cols:
                expected = mp.BOND_MASK_TOKEN if k in masked_bonds else int(ea_r.reshape(-1)[col])
                if int(ea_masked.reshape(-1)[col]) != expected:
                    bidi_ok = False
        # 5. observed mask ratios
        if base.n_atoms:
            max_ratio_gap = max(max_ratio_gap, abs(len(s.atom_ids) / base.n_atoms - mp.ATOM_MASK_RATIO))
        checked += 1
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "construction": "full_rebuild_only",
        "n_cases": checked,
        "random_seed": 3,
        "deterministic_reproduction": bool(determinism_ok),
        "differences_confined_to_dependency_map": bool(worst_subset_ok),
        "masked_own_slots_hold_token": bool(own_slot_ok),
        "hidden_truth_absent_from_own_slot": bool(hidden_absent_ok),
        "both_directed_records_masked": bool(bidi_ok),
        "max_abs_observed_atom_mask_ratio_gap": float(max_ratio_gap),
        "note": (
            "no incremental feature update exists in this round; the masked object is always "
            "rebuilt end-to-end from the masked raw graph, so no per-field incremental patch "
            "needs a separate equivalence proof"
        ),
        "verdict": (
            "MASKED_REBUILD_LEAK_FREE"
            if (determinism_ok and worst_subset_ok and own_slot_ok and hidden_absent_ok and bidi_ok)
            else "MASKED_REBUILD_INCONSISTENT"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(REBUILD_AUDIT, payload)
    log(f"[rebuild-audit] cases={checked} verdict {payload['verdict']}")
    if payload["verdict"] != "MASKED_REBUILD_LEAK_FREE":
        raise SystemExit(payload["verdict"])
    return 0


# ---------------------------------------------------------------------------
# feasibility / timing
# ---------------------------------------------------------------------------


def cmd_feasibility(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split = read_split()
    ds = load_raw_dataset()
    bases = load_objects(OBJECTS_FIT8K)
    fit = list(map(int, split["fit8k"]))
    assert [o.mol_index for o in bases] == fit

    # per-molecule raw structures (labels never touched)
    atom_codes = []
    bond_codes = []
    for i in fit:
        d = ds[i]
        atom_codes.append(d.x.numpy().reshape(-1))
        bond_codes.append(d.edge_attr.numpy().reshape(-1))
    atom_all = np.concatenate(atom_codes)
    bond_all = np.concatenate(bond_codes)
    priors = mp.class_prior_stats(atom_all, bond_all)

    # masked-position priors (visit-0 masks over a fixed sample)
    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(len(fit), size=min(500, len(fit)), replace=False))
    masked_atoms, masked_bonds = [], []
    n_masked_atoms = n_masked_bonds = 0
    for j in sample_idx:
        o = bases[int(j)]
        a, b = mp.sample_mask(0, o.mol_index, 0, o.n_atoms, o.bond_index.shape[0])
        masked_atoms.append(np.asarray([ds[o.mol_index].x.numpy().reshape(-1)[int(z)] for z in a], dtype=np.int64))
        masked_bonds.append(np.asarray([int(o.bond_type[int(z)]) for z in b], dtype=np.int64))
        n_masked_atoms += len(a)
        n_masked_bonds += len(b)
    masked_priors = mp.class_prior_stats(np.concatenate(masked_atoms), np.concatenate(masked_bonds))

    # throughput
    n_timing = min(256, len(fit))
    t0 = time.perf_counter()
    samples = [
        mp.build_masked_sample(bases[j], ds[bases[j].mol_index].x.numpy(), ds[bases[j].mol_index].edge_index.numpy(), ds[bases[j].mol_index].edge_attr.numpy(), seed=0, visit_index=0)
        for j in range(n_timing)
    ]
    t_mask = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch, at, bt, _, _ = mp.batch_pretext_samples(samples)
    t_batch = time.perf_counter() - t0

    per_mol_mask = t_mask / n_timing
    per_batch_mask = per_mol_mask * BATCH_SIZE
    per_batch_batch = t_batch * (BATCH_SIZE / n_timing)
    projected = (per_batch_mask + per_batch_batch) * mp.PRETRAIN_UPDATES
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    cfg = src.BJOCConfig()
    backbone = src.build_bjoc(
        cfg, seed=0, organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    pretext = mp.PretextModel(backbone)
    backbone_params = int(sum(p.numel() for p in backbone.parameters()))
    pretext_params = int(sum(p.numel() for p in pretext.parameters()))
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "class_priors_fit8k": priors,
        "masked_position_priors_visit0_500mol": masked_priors,
        "class_imbalance": {
            "atom_majority_fraction": priors["atom"]["majority_fraction"],
            "bond_majority_fraction": priors["bond"]["majority_fraction"],
            "severe_atom_imbalance": bool(priors["atom"]["majority_fraction"] > 0.5),
            "severe_bond_imbalance": bool(priors["bond"]["majority_fraction"] > 0.5),
        },
        "throughput": {
            "timed_molecules": n_timing,
            "masked_build_seconds_per_molecule": per_mol_mask,
            "projected_masked_build_seconds_per_batch": per_batch_mask,
            "projected_batch_assembly_seconds_per_batch": per_batch_batch,
            "projected_data_prep_seconds_for_7560_updates": projected,
            "projected_data_prep_minutes_for_7560_updates": projected / 60.0,
            "n_masked_atoms_in_sample": n_masked_atoms,
            "n_masked_bonds_in_sample": n_masked_bonds,
        },
        "memory": {
            "peak_rss_kib": int(peak_rss),
            "objects_fit8k_bytes": int(OBJECTS_FIT8K.stat().st_size),
            "raw_structures_bytes": int(atom_all.nbytes + bond_all.nbytes),
            "masked_type_cache_bytes_if_persisted": int((atom_all.size + bond_all.size) * 1),
        },
        "model": {
            "downstream_parameters": backbone_params,
            "pretext_total_parameters": pretext_params,
            "pretext_head_parameters": pretext_params - backbone_params,
            "atom_head_in": 4 * cfg.atom_out,
            "bond_head_in": 4 * cfg.bond_out + cfg.junc_out,
        },
        "budget_projection": {
            "pretrain_updates": mp.PRETRAIN_UPDATES,
            "pretrain_batch": mp.PRETRAIN_BATCH,
            "supervised_updates_B_P": SUPERVISED_UPDATES_B,
            "supervised_updates_C": SUPERVISED_UPDATES_C,
            "projected_total_data_prep_minutes_two_ssl_seeds": 2 * projected / 60.0,
        },
        "verdict": "MASKED_DATA_PATH_FEASIBLE" if projected < 3600 else "MASKED_DATA_PATH_TOO_SLOW",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(FEASIBILITY_AUDIT, payload)
    log(
        f"[feasibility] mask build {per_mol_mask*1000:.2f} ms/mol; projected SSL data prep "
        f"{projected/60:.1f} min/seed; verdict {payload['verdict']}"
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    return cmd_feasibility(args)


# ---------------------------------------------------------------------------
# correctness gates
# ---------------------------------------------------------------------------


def _scan_forbidden_source(text: str) -> list[str]:
    found = []
    tokens = (
        "Message" + "Passing",
        "Transformer" + "Encoder",
        "Multihead" + "Attention",
        "nn." + "Transformer",
        "self_" + "attention",
        "LS" + "TM",
        "G" + "RU",
        "nn." + "G" + "RU",
        "nn." + "LS" + "TM",
    )
    for token in tokens:
        if token in text:
            found.append(token)
    return found


def _scan_source_for_y_reads(path: Path, function_name: str) -> dict[str, Any]:
    """AST scan of one function: does it read a ``.y`` attribute or load the
    encoded labelled cache?"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target = node
            break
    if target is None:
        return {"found": False, "error": f"function {function_name} not found"}
    hits = []
    for node in ast.walk(target):
        if isinstance(node, ast.Attribute) and node.attr == "y":
            hits.append("attribute .y")
        if isinstance(node, ast.Name) and node.id == "ENCODED_TRAIN":
            hits.append("ENCODED_TRAIN")
    return {"found": bool(hits), "hits": sorted(set(hits))}


def _module_reads_y_attribute(path: Path) -> bool:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "y":
            return True
    return False


def cmd_gates(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gates: dict[str, Any] = {}

    def gate(name: str, ok: bool, detail: Any) -> None:
        gates[name] = {"pass": bool(ok), "detail": detail}
        log(f"[gate] {name}: {'PASS' if ok else 'FAIL'}")

    # frozen inputs
    split = read_split()
    split_cfg = read_json(SPLIT_JSON)
    verify_split_rows(split)
    objects = load_objects(OBJECTS_FIT8K)
    mon_objects = load_objects(OBJECTS_MONITOR)
    lock_objects = load_objects(OBJECTS_LOCKED)
    ds = load_raw_dataset()
    fit = list(map(int, split["fit8k"]))

    # 0. split integrity (label-blind)
    gate(
        "split_integrity",
        split_cfg["overlap"] == {"fit_monitor": 0, "fit_locked": 0, "monitor_locked": 0, "union": 10000}
        and split_cfg["duplicate_groups_cross_sides"] == 0
        and rows_sha256(split["fit8k"]) == split_cfg["fit_sha256_rows"],
        {"n_fit8k": len(fit), "n_monitor": len(split["monitor"]), "n_locked": len(split["locked"])},
    )
    # 1. mask target-swap invariance
    swap = read_json(SWAP_AUDIT)
    gate("mask_target_swap_invariance", swap["verdict"] == "MASK_SWAP_INVARIANT", {"verdict": swap["verdict"], "random": swap["randomized_real_fit8k"]})
    # 2/3. raw truth absent from model inputs / full rebuild dependency
    rebuild = read_json(REBUILD_AUDIT)
    gate(
        "hidden_truth_absent_from_features",
        rebuild["hidden_truth_absent_from_own_slot"] and rebuild["masked_own_slots_hold_token"],
        {k: rebuild[k] for k in ("hidden_truth_absent_from_own_slot", "masked_own_slots_hold_token")},
    )
    gate(
        "masked_rebuild_matches_dependency_map",
        rebuild["differences_confined_to_dependency_map"] and rebuild["deterministic_reproduction"],
        {k: rebuild[k] for k in ("differences_confined_to_dependency_map", "deterministic_reproduction", "construction")},
    )
    # 4. PyG bidirectional masking
    gate("pyg_bidirectional_masking", rebuild["both_directed_records_masked"], {"detail": rebuild["both_directed_records_masked"]})
    # 5. masked label correspondence + 7-8. finite CE / encoder gradients
    probe = _small_molecule_objects()
    samples = [mp.build_masked_sample(o, x, ei, ea, seed=5, visit_index=0) for (o, x, ei, ea) in probe]
    label_ok = True
    for s, (o, x, ei, ea) in zip(samples, probe):
        for a, t in zip(s.atom_ids, s.atom_types):
            label_ok &= int(x.reshape(-1)[int(a)]) == int(t)
        for b, t in zip(s.bond_ids, s.bond_types):
            label_ok &= int(o.bond_type[int(b)]) == int(t)
    gate("masked_label_correspondence", label_ok, {"cases": len(samples)})
    cfg = src.BJOCConfig()
    backbone = src.build_bjoc(
        cfg, seed=0, organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    pretext = mp.PretextModel(backbone)
    batch, at, bt, _, _ = mp.batch_pretext_samples(samples)
    loss, parts = pretext.loss(batch, at, bt)
    gate("pretext_ce_finite", bool(np.isfinite(float(loss)) and float(parts["ce_atom"]) > 0 and float(parts["ce_bond"]) > 0), {"loss": float(loss), "ce_atom": float(parts["ce_atom"]), "ce_bond": float(parts["ce_bond"])})
    pretext.zero_grad()
    loss.backward()
    enc_grads = {}
    for group in ("atom_encoder", "bond_encoder", "junction_encoder"):
        module = getattr(backbone, group)
        enc_grads[group] = float(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None))
    gate("encoder_gradients_nonzero", all(v > 0 for v in enc_grads.values()), enc_grads)
    # 6. pretext loss target isolation (AST)
    y_scan = _scan_source_for_y_reads(_CODE_DIR / "run_bjoc_masked_pretraining_v1.py", "pretrain_one_seed")
    mod_reads_y = _module_reads_y_attribute(_CODE_DIR / "bjoc_masked_pretraining_v1.py")
    gate("pretext_loss_targets_isolated", (not y_scan.get("found", True)) and (not mod_reads_y), {"driver_scan": y_scan, "module_reads_y_attribute": mod_reads_y})
    # 9/10/11. downstream model isolation
    downstream_params = int(sum(p.numel() for p in backbone.parameters()))
    gate("pretext_heads_not_in_downstream", not hasattr(backbone, "atom_head") and not hasattr(backbone, "bond_head"), {"backbone_attrs": [k for k in vars(backbone) if "head" in k]})
    gate("downstream_parameters_413505", downstream_params == 413505, {"actual": downstream_params})
    with torch.no_grad():
        p1 = backbone(batch)
        p2 = backbone(batch)
    gate("downstream_forward_deterministic", bool(torch.equal(p1, p2)), {"shape": list(p1.shape)})
    # 12. atom permutation invariance (real molecule, raw relabel)
    perm = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)
    o = probe[3][0]
    x, ei, ea = probe[3][1], probe[3][2], probe[3][3]
    x2 = x.copy().reshape(-1)[np.argsort(perm)].reshape(-1, 1)
    ei2 = perm[ei]
    o2 = sro.build_mol_objects(3, x2, ei2, ea.copy())
    b1 = sro.batch_objects([o])
    b2 = sro.batch_objects([o2])
    with torch.no_grad():
        r1 = backbone(b1)
        r2 = backbone(b2)
    gate("atom_permutation_invariance", bool(torch.allclose(r1, r2, atol=1e-6)), {"max_abs_diff": float((r1 - r2).abs().max())})
    # 13. bond endpoint symmetry of the bond pretext head
    s0 = samples[0]
    b0, t0a, t0b, _, _ = mp.batch_pretext_samples([s0])
    b0_swap = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b0.items()}
    b0_swap["bond_index"] = b0_swap["bond_index"][:, [1, 0]]
    with torch.no_grad():
        o1 = pretext(b0, b0["masked_atom_ids"], b0["masked_bond_ids"])
        o2 = pretext(b0_swap, b0_swap["masked_atom_ids"], b0_swap["masked_bond_ids"])
    gate("bond_endpoint_symmetry", bool(torch.allclose(o1["bond_logits"], o2["bond_logits"], atol=1e-6)), {"max_abs_diff": float((o1["bond_logits"] - o2["bond_logits"]).abs().max())})
    # 14. batch composition consistency
    with torch.no_grad():
        solo = backbone(sro.batch_objects([o]))
        mixed = backbone(sro.batch_objects([s0.masked, o]))
    gate("batch_composition_consistency", bool(torch.allclose(solo[0], mixed[1], atol=1e-6)), {"max_abs_diff": float((solo[0] - mixed[1]).abs().max())})
    # 15. pretraining sees Fit8k structures only (source + data manifest)
    pretrain_scan = _scan_source_for_y_reads(_CODE_DIR / "run_bjoc_masked_pretraining_v1.py", "pretrain_one_seed")
    gate("pretrain_uses_fit8k_only", [m.mol_index for m in objects] == fit and not pretrain_scan.get("found", True), {"n_fit": len(fit), "scan": pretrain_scan})
    # 16. pretraining does not load labelled tensors
    gate("pretrain_process_does_not_load_y", (not y_scan.get("found", True)) and "ENCODED_TRAIN" not in inspect.getsource(pretrain_one_seed), {"scan": y_scan})
    # 17. save/load replay of the encoder state
    enc_state = mp.encoder_state(backbone)
    fresh = src.build_bjoc(cfg, seed=999, organization_mode=src.NULL, global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32), global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32))
    mp.load_encoder_state(fresh, enc_state)
    with torch.no_grad():
        r = backbone.encode_atoms(batch)
        r3 = fresh.encode_atoms(batch)
    gate("save_load_replay", bool(torch.equal(r, r3)), {"max_abs_diff": float((r - r3).abs().max())})
    # 18. mask RNG reproducibility
    m_a = mp.sample_mask(0, 17, 3, 23, 24)
    m_b = mp.sample_mask(0, 17, 3, 23, 24)
    m_c = mp.sample_mask(1, 17, 3, 23, 24)
    gate(
        "mask_rng_reproducible",
        np.array_equal(m_a[0], m_b[0]) and np.array_equal(m_a[1], m_b[1]) and not (np.array_equal(m_a[0], m_c[0]) and np.array_equal(m_a[1], m_c[1])),
        {"atoms": m_a[0].tolist(), "bonds": m_a[1].tolist()},
    )
    # 19. official valid/test never loaded
    src_text = (_CODE_DIR / "run_bjoc_masked_pretraining_v1.py").read_text(encoding="utf-8")
    mod_text = (_CODE_DIR / "bjoc_masked_pretraining_v1.py").read_text(encoding="utf-8")
    official_tokens = [
        "_load_official_" + 'split("val")',
        "_load_official_" + 'split("test")',
        "official_valid" + "_preds",
    ]
    hits = [t for t in official_tokens if t in src_text or t in mod_text]
    gate("official_valid_test_never_loaded", not hits, {"hits": hits})
    # 20. no GNN / attention / iteration
    forbidden = src.forbidden_module_scan(backbone) + _scan_forbidden_source(src_text) + _scan_forbidden_source(mod_text)
    gate("no_gnn_attention_iteration", not forbidden, {"found": forbidden})
    # extra: mask ratios and token slots
    ratios = {"atom": len(m_a[0]) / 23.0, "bond": len(m_a[1]) / 24.0}
    gate("mask_ratio_15_percent", abs(ratios["atom"] - 0.15) < 0.035 and abs(ratios["bond"] - 0.15) < 0.035, ratios)

    passed = all(v["pass"] for v in gates.values())
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_gates": len(gates),
        "n_passed": int(sum(v["pass"] for v in gates.values())),
        "all_pass": bool(passed),
        "gates": gates,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(GATES_JSON, payload)
    log(f"[gates] {payload['n_passed']}/{payload['n_gates']} pass")
    if not passed:
        raise SystemExit("correctness gates failed")
    return 0


# ---------------------------------------------------------------------------
# freeze manifest
# ---------------------------------------------------------------------------


def _code_hashes() -> dict[str, str]:
    return {
        "driver": file_sha256(_CODE_DIR / "run_bjoc_masked_pretraining_v1.py"),
        "module": file_sha256(_CODE_DIR / "bjoc_masked_pretraining_v1.py"),
        "objects": file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
        "composer": file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
        "features": file_sha256(_CODE_DIR / "cscl_features.py"),
        "config": file_sha256(CONFIG_PATH),
    }


def cmd_freeze(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split_cfg = read_json(SPLIT_JSON)
    objs_manifest = read_json(OBJECTS_MANIFEST)
    config = read_json(CONFIG_PATH) if CONFIG_PATH.suffix == ".json" else None
    steps_per_epoch = math.ceil(split_cfg["n_fit8k"] / BATCH_SIZE)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hashes": _code_hashes(),
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "split": {
            "salt": SPLIT_SALT,
            "n_fit8k": split_cfg["n_fit8k"],
            "n_monitor": split_cfg["n_monitor"],
            "n_locked": split_cfg["n_locked"],
            "fit_sha256_rows": split_cfg["fit_sha256_rows"],
            "monitor_sha256_rows": split_cfg["monitor_sha256_rows"],
            "locked_sha256_rows": split_cfg["locked_sha256_rows"],
            "duplicate_groups_cross_sides": split_cfg["duplicate_groups_cross_sides"],
        },
        "caches": {
            "objects_fit8k_sha256": objs_manifest["objects_fit8k_sha256"],
            "objects_monitor_sha256": objs_manifest["objects_monitor_sha256"],
            "objects_locked_sha256": objs_manifest["objects_locked_sha256"],
            "prep_fit8k_sha256": objs_manifest["prep_fit8k_sha256"],
        },
        "mask": {
            "atom_ratio": mp.ATOM_MASK_RATIO,
            "bond_ratio": mp.BOND_MASK_RATIO,
            "atom_token": mp.ATOM_MASK_TOKEN,
            "bond_token": mp.BOND_MASK_TOKEN,
            "rng": "numpy.random.SeedSequence([seed, molecule_id, visit_index])",
            "min_one_per_molecule": True,
            "physical_bonds_double_edge_sync": True,
        },
        "pretext": {
            "task": "masked raw atom-type + physical bond-type reconstruction",
            "loss": "cross_entropy(masked atoms) + cross_entropy(masked physical bonds), weights 1.0/1.0",
            "shared_backbone": ["atom_encoder", "bond_encoder", "junction_encoder"],
            "discarded_heads": ["atom_head", "bond_head"],
            "updates": mp.PRETRAIN_UPDATES,
            "batch_size": mp.PRETRAIN_BATCH,
            "optimizer": "Adam",
            "lr": mp.PRETRAIN_LR,
            "weight_decay": mp.PRETRAIN_WD,
            "grad_clip": mp.PRETRAIN_CLIP,
            "checkpoint_rule": "last step encoder state only; no selection on reconstruction accuracy",
        },
        "downstream": {
            "arms": {
                "B": {"init": "scratch", "updates": SUPERVISED_UPDATES_B, "epochs": EPOCHS_BASE},
                "P": {"init": "frozen pretrained encoders + matching scratch composer/reader", "updates": SUPERVISED_UPDATES_B, "epochs": EPOCHS_BASE},
                "C": {"init": "scratch", "updates": SUPERVISED_UPDATES_C, "epochs": EPOCHS_CONTROL, "condition": "only if the monitor purchase gate fires"},
            },
            "recipe": {
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "grad_clip": GRAD_CLIP,
                "loss": "raw-y L1 only",
                "scheduler": "none",
                "dropout": 0.0,
                "soup": "parameter mean of the last 5 epoch states",
                "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            },
            "parameters_expected": 413505,
            "steps_per_epoch": steps_per_epoch,
            "seeds": list(SEEDS),
        },
        "purchase_gate": {
            "seed_mean_monitor_delta_max": -0.005,
            "both_seed_deltas_negative": True,
            "ordinary_delta_max": 0.0,
            "top10_removed_delta_max": 0.0,
            "bootstrap_ci_upper_max": 0.0,
            "no_leakage_or_protocol_deviation": True,
            "encoder_actually_changed_and_used": True,
        },
        "budget": {
            "wall_clock_minutes_max": 465,
            "gpu_hours_max": 2.0,
            "gpus_max": 1,
            "max_formal_stages": 8,
            "no_full10k_final_training": True,
        },
        "decision_vocabulary": [
            "SELF_SUPERVISED_PRETRAINING_USEFUL",
            "PRETRAINING_ONLY_ACCELERATES_OPTIMIZATION",
            "PRETRAINING_NEGATIVE_TRANSFER",
            "PRETEXT_TASK_TRIVIAL_OR_LEAKING",
            "NO_USEFUL_TRANSFER",
            "INCONCLUSIVE_OR_BLOCKED",
        ],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if config is None:
        import yaml

        cfg_yaml = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg_frozen = cfg_yaml.get("frozen", {})
        mismatches = []
        for key, value in (
            ("fit8k_sha256_rows", payload["split"]["fit_sha256_rows"]),
            ("monitor_sha256_rows", payload["split"]["monitor_sha256_rows"]),
            ("locked_sha256_rows", payload["split"]["locked_sha256_rows"]),
            ("objects_fit8k_sha256", payload["caches"]["objects_fit8k_sha256"]),
            ("objects_monitor_sha256", payload["caches"]["objects_monitor_sha256"]),
            ("objects_locked_sha256", payload["caches"]["objects_locked_sha256"]),
            ("prep_fit8k_sha256", payload["caches"]["prep_fit8k_sha256"]),
        ):
            if key in cfg_frozen and cfg_frozen[key] != value:
                mismatches.append({"key": key, "config": cfg_frozen[key], "actual": value})
        payload["config_frozen_block_mismatches"] = mismatches
        if mismatches:
            write_json(TRAINING_MANIFEST, payload)
            raise SystemExit(f"config frozen block mismatches: {mismatches}")
    write_json(TRAINING_MANIFEST, payload)
    log(f"[freeze] training manifest written; fit sha {payload['split']['fit_sha256_rows'][:16]}")
    return 0


def _config_frozen_reference() -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg.get("frozen", {})


def verify_freeze() -> dict[str, Any]:
    """Refuse to train unless the deployed revision matches the frozen code,
    split and (when present) object-cache hashes."""
    split_cfg = read_json(SPLIT_JSON)
    fit = read_split()["fit8k"]
    if rows_sha256(fit) != split_cfg["fit_sha256_rows"]:
        raise SystemExit("fit split hash mismatch vs split_a.json")
    if TRAINING_MANIFEST.exists():
        manifest = read_json(TRAINING_MANIFEST)
        current = _code_hashes()
        mismatches = {k: {"frozen": manifest["code_hashes"][k], "current": v} for k, v in current.items() if manifest["code_hashes"].get(k) != v}
        if mismatches:
            raise SystemExit(f"code hash mismatch vs frozen manifest: {mismatches}")
        if rows_sha256(fit) != manifest["split"]["fit_sha256_rows"]:
            raise SystemExit("fit split hash mismatch vs manifest")
        return manifest
    # remote fallback: the committed config carries the frozen hashes
    frozen = _config_frozen_reference()
    checks = {
        "fit8k_sha256_rows": split_cfg["fit_sha256_rows"],
        "monitor_sha256_rows": split_cfg["monitor_sha256_rows"],
        "locked_sha256_rows": split_cfg["locked_sha256_rows"],
    }
    for key, value in checks.items():
        if key in frozen and frozen[key] != value:
            raise SystemExit(f"config frozen {key} mismatch: {frozen[key]} != {value}")
    if OBJECTS_FIT8K.exists() and "objects_fit8k_sha256" in frozen:
        actual = file_sha256(OBJECTS_FIT8K)
        if actual != frozen["objects_fit8k_sha256"]:
            raise SystemExit(f"objects_fit8k hash mismatch vs config frozen block: {actual}")
    if PREP_FIT8K.exists() and "prep_fit8k_sha256" in frozen:
        actual = file_sha256(PREP_FIT8K)
        if actual != frozen["prep_fit8k_sha256"]:
            raise SystemExit(f"prep_fit8k hash mismatch vs config frozen block: {actual}")
    return {"source": "config_frozen_block", "frozen": frozen}


# ---------------------------------------------------------------------------
# training: masked pretraining
# ---------------------------------------------------------------------------


def _mv(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(int(n), generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def batch_order_hash(seed: int, n: int, batch_size: int, epochs: int) -> str:
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    h = hashlib.sha256()
    for _ in range(int(epochs)):
        order = torch.randperm(int(n), generator=gen)
        h.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
    return h.hexdigest()


def pretrain_one_seed(seed: int, device: torch.device, *, log_fn: Callable[[str], None] = log) -> dict[str, Any]:
    """Masked atom/bond pretraining.  Reads **no labels at all**: only the raw
    ``x`` / ``edge_index`` / ``edge_attr`` of the frozen Fit8k rows."""
    verify_freeze()
    split = read_split()
    fit = list(map(int, split["fit8k"]))
    bases = load_objects(OBJECTS_FIT8K)
    assert [o.mol_index for o in bases] == fit
    ds = load_raw_dataset()
    train_rows = set(fit)
    assert len(train_rows) == 8000

    cfg = src.BJOCConfig()
    seed_everything(seed)
    backbone = src.build_bjoc(
        cfg, seed=seed, organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        device=device,
    )
    init_encoder_sha = _encoder_group_hash(backbone)
    pretext = mp.PretextModel(backbone).to(device)
    optimizer = torch.optim.Adam(pretext.parameters(), lr=mp.PRETRAIN_LR, weight_decay=mp.PRETRAIN_WD)
    gen = torch.Generator().manual_seed(int(seed) + mp.PRETRAIN_SHUFFLE_OFFSET)

    curve: list[dict[str, Any]] = []
    steps = 0
    visit = 0
    loss_at_steps: dict[int, float] = {}
    started = time.perf_counter()
    while steps < mp.PRETRAIN_UPDATES:
        pretext.train()
        ce_a_sum = ce_b_sum = acc_a_sum = acc_b_sum = 0.0
        n_a = n_b = n_steps = 0
        t_ep = time.perf_counter()
        stop = False
        for chunk in _shuffle_batches(len(fit), BATCH_SIZE, gen):
            pre = []
            for j in chunk:
                o = bases[j]
                assert int(o.mol_index) in train_rows
                d = ds[int(o.mol_index)]
                pre.append(
                    mp.build_masked_sample(
                        o, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy(),
                        seed=seed, visit_index=visit,
                    )
                )
            batch, targets_a, targets_b, _, _ = mp.batch_pretext_samples(pre)
            batch = _mv(batch, device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
            loss, parts = pretext.loss(batch, targets_a, targets_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretext.parameters(), mp.PRETRAIN_CLIP)
            optimizer.step()
            steps += 1
            n_steps += 1
            with torch.no_grad():
                logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                if targets_a.numel():
                    ce_a_sum += float(parts["ce_atom"])
                    acc_a_sum += float((logits["atom_logits"].argmax(1) == targets_a).float().mean())
                    n_a += 1
                if targets_b.numel():
                    ce_b_sum += float(parts["ce_bond"])
                    acc_b_sum += float((logits["bond_logits"].argmax(1) == targets_b).float().mean())
                    n_b += 1
            loss_at_steps[steps] = float(loss)
            if steps >= mp.PRETRAIN_UPDATES:
                stop = True
                break
        curve.append(
            {
                "visit": visit,
                "steps_total": steps,
                "ce_atom": ce_a_sum / max(n_a, 1),
                "ce_bond": ce_b_sum / max(n_b, 1),
                "acc_atom": acc_a_sum / max(n_a, 1),
                "acc_bond": acc_b_sum / max(n_b, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        if visit % 10 == 0 or stop:
            log_fn(
                f"[ssl s{seed}] visit={visit} steps={steps}/{mp.PRETRAIN_UPDATES} "
                f"ce_a={curve[-1]['ce_atom']:.4f} acc_a={curve[-1]['acc_atom']:.4f} "
                f"ce_b={curve[-1]['ce_bond']:.4f} acc_b={curve[-1]['acc_bond']:.4f}"
            )
        visit += 1
    wall = time.perf_counter() - started

    enc_state = mp.encoder_state(backbone)
    final_encoder_sha = _encoder_group_hash(backbone)
    torch.save(enc_state, OUT / f"pretrain_encoder_s{seed}.pt")
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": "SSL",
        "seed": int(seed),
        "git_commit": git_commit(),
        "updates": int(steps),
        "visits": visit,
        "batch_size": mp.PRETRAIN_BATCH,
        "hyper": {
            "optimizer": "Adam",
            "lr": mp.PRETRAIN_LR,
            "weight_decay": mp.PRETRAIN_WD,
            "grad_clip": mp.PRETRAIN_CLIP,
            "loss": "CE(masked atoms) + CE(masked physical bonds), weights 1.0/1.0",
            "checkpoint_rule": "last step only",
        },
        "mask": {
            "atom_ratio": mp.ATOM_MASK_RATIO,
            "bond_ratio": mp.BOND_MASK_RATIO,
            "atom_token": mp.ATOM_MASK_TOKEN,
            "bond_token": mp.BOND_MASK_TOKEN,
            "rng": "SeedSequence([seed, molecule_id, visit_index])",
        },
        "fit_rows_only": True,
        "n_fit_rows": len(fit),
        "wall_seconds": wall,
        "encoder_init_sha256": init_encoder_sha,
        "encoder_final_sha256": final_encoder_sha,
        "encoder_parameters_changed": bool(init_encoder_sha != final_encoder_sha),
        "curve": curve,
        "final_ce_atom": curve[-1]["ce_atom"],
        "final_ce_bond": curve[-1]["ce_bond"],
        "final_acc_atom": curve[-1]["acc_atom"],
        "final_acc_bond": curve[-1]["acc_bond"],
        "loss_at_steps": {str(k): v for k, v in sorted(loss_at_steps.items())},
        "params": int(sum(p.numel() for p in pretext.parameters())),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / f"pretraining_seed{seed}.json", payload)
    log_fn(f"[ssl s{seed}] final acc atom {payload['final_acc_atom']:.4f} bond {payload['final_acc_bond']:.4f} | wall {wall:.0f}s")
    return payload


# ---------------------------------------------------------------------------
# training: downstream arms
# ---------------------------------------------------------------------------


def _make_downstream_model(seed: int, norm: Mapping[str, Any], device: torch.device) -> src.BJOC:
    cfg = src.BJOCConfig()
    seed_everything(seed)
    return src.build_bjoc(
        cfg,
        seed=seed,
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )


def _encoder_group_hash(model: src.BJOC) -> str:
    st = mp.encoder_state(model)
    combined = (
        {f"atom:{k}": v for k, v in st["atom_encoder"].items()}
        | {f"bond:{k}": v for k, v in st["bond_encoder"].items()}
        | {f"junc:{k}": v for k, v in st["junction_encoder"].items()}
    )
    return state_group_sha256(combined)


def train_downstream(
    arm: str,
    seed: int,
    device: torch.device,
    *,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    """One formal downstream trajectory (B / P / C).  Raw-y L1, frozen recipe."""
    verify_freeze()
    assert arm in ("B", "P", "C"), arm
    if arm == "C":
        epochs = EPOCHS_CONTROL
    else:
        epochs = EPOCHS_BASE
    split = read_split()
    fit = list(map(int, split["fit8k"]))
    monitor = list(map(int, split["monitor"]))
    fit_objs = load_objects(OBJECTS_FIT8K)
    mon_objs = load_objects(OBJECTS_MONITOR)
    norm = read_json(PREP_FIT8K)
    y_fit = _load_y_rows(fit)
    y_mon = _load_y_rows(monitor)

    model = _make_downstream_model(seed, norm, device)
    init_full_sha = src.state_dict_hash(model)
    init_group_sha = mp.group_state_hash(model)
    init_encoder_sha = _encoder_group_hash(model)
    pretrain_encoder_sha = None
    encoder_used_sha = init_encoder_sha
    if arm == "P":
        enc_path = OUT / f"pretrain_encoder_s{seed}.pt"
        if not enc_path.exists():
            raise SystemExit(f"missing pretrained encoder state {enc_path}; run pretrain first")
        pretrain_encoder_sha = file_sha256(enc_path)
        mp.load_encoder_state(model, torch.load(enc_path, map_location="cpu", weights_only=False), device)
        encoder_used_sha = _encoder_group_hash(model)
        if encoder_used_sha != read_json(OUT / f"pretraining_seed{seed}.json")["encoder_final_sha256"]:
            raise SystemExit("loaded encoder state does not match the recorded pretraining state")
    audit = src.parameter_audit(src.BJOCConfig(), model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != 413505:
        raise SystemExit(f"parameter audit failed: {audit}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in _shuffle_batches(len(fit), BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            batch = _mv(sro.batch_objects([fit_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        mon_pred = predict_bjoc(model, mon_objs, list(range(len(mon_objs))), device)
        mon_mae = float(np.abs(mon_pred - y_mon).mean())
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": total / max(n_mol, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s")
        if not np.isfinite(curve[-1]["train_mae"]) or not np.isfinite(mon_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} arm {arm} seed {seed}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_bjoc(model, fit_objs, list(range(len(fit_objs))), device)
    mon_pred = predict_bjoc(model, mon_objs, list(range(len(mon_objs))), device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(soup_state, OUT / f"soup_{arm}_seed{seed}.pt")
    final_encoder_sha = _encoder_group_hash(model)
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "git_commit": git_commit(),
        "epochs": epochs,
        "updates": int(epochs * math.ceil(len(fit) / BATCH_SIZE)),
        "steps_per_epoch": math.ceil(len(fit) / BATCH_SIZE),
        "soup_rule": f"parameter mean of the last {SOUP_LAST} epoch states",
        "soup_members": members,
        "hyper": {
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "loss": "raw-y L1 only",
            "scheduler": "none",
            "dropout": 0.0,
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        "parameters": audit["actual_parameters"],
        "init_full_sha256": init_full_sha,
        "init_group_state_sha256": init_group_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "final_encoder_sha256": final_encoder_sha,
        "pretrain_encoder_file_sha256": pretrain_encoder_sha,
        "batch_order_hash": order_hash.hexdigest(),
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "soup_monitor_mae": float(np.abs(mon_pred - y_mon).mean()),
        "curve": curve,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "fit rows only",
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / f"train_{arm}_seed{seed}.json", payload)
    log_fn(f"[{arm} s{seed}] soup fit {payload['soup_fit_mae']:.5f} monitor {payload['soup_monitor_mae']:.5f} wall {wall:.0f}s")
    return payload


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    assert 0 < len(states) <= SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    keys = states[0].keys()
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}
    return members, soup


def predict_bjoc(model: src.BJOC, objs: Sequence[sro.MolObjects], rows: Sequence[int], device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            chunk = [objs[int(r)] for r in rows[i : i + 256]]
            batch = _mv(sro.batch_objects(chunk), device)
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# y access (fit / monitor only outside score)
# ---------------------------------------------------------------------------


def _load_y_rows(rows: Sequence[int]) -> np.ndarray:
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    values = np.asarray([float(enc[int(i)].y.reshape(-1)[0]) for i in rows], dtype=np.float64)
    del enc
    return values


def load_locked_y(locked: Sequence[int]) -> np.ndarray:
    """The ONE locked-dev label read; only ``cmd_score`` may call this."""
    return _load_y_rows(locked)


# ---------------------------------------------------------------------------
# prediction / freeze
# ---------------------------------------------------------------------------


def _model_keys(arms: Sequence[str], seeds: Sequence[int] = SEEDS) -> list[str]:
    return [f"{arm}{seed}" for arm in arms for seed in seeds]


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze()
    device = torch.device(args.device)
    norm = read_json(PREP_FIT8K)
    split = read_split()
    mon_objs = load_objects(OBJECTS_MONITOR)
    lock_objs = load_objects(OBJECTS_LOCKED)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models]
    for key in models:
        arm, seed = key[0], int(key[1:])
        soup_path = OUT / f"soup_{arm}_seed{seed}.pt"
        if not soup_path.exists():
            raise SystemExit(f"missing checkpoint {soup_path}")
        model = _make_downstream_model(seed, norm, device)
        model.load_state_dict(torch.load(soup_path, map_location=device, weights_only=False))
        for split_name, objs in (("monitor", mon_objs), ("locked", lock_objs)):
            pred = predict_bjoc(model, objs, list(range(len(objs))), device)
            if not np.isfinite(pred).all():
                raise RuntimeError("non-finite predictions")
            path = PRED_DIR / f"pred_{split_name}_{key}.npz"
            np.savez_compressed(path, row=np.arange(len(pred), dtype=np.int64), pred=pred.astype(np.float32))
        log(f"[predict] {key} monitor+locked written (checkpoint sha {file_sha256(soup_path)[:12]})")
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    verify_freeze()
    files = {}
    for path in sorted(PRED_DIR.glob("pred_*.npz")):
        zone = path.stem.split("_")[1]
        key = path.stem.split("_")[2]
        with np.load(path) as z:
            if not np.isfinite(z["pred"]).all():
                raise SystemExit(f"non-finite prediction in {path}")
        files[f"{zone}:{key}"] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    ckpts = {}
    for path in sorted(OUT.glob("soup_*.pt")):
        ckpts[path.stem] = file_sha256(path)
    combined = hashlib.sha256()
    for key in sorted(files):
        combined.update(key.encode())
        combined.update(files[key]["sha256"].encode())
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_prediction_files": len(files),
        "prediction_files": files,
        "checkpoints": ckpts,
        "combined_prediction_sha256": combined.hexdigest(),
        "labels_read_for_freezing": False,
        "locked_labels_read": False,
        "frozen_before_locked_read": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(PRED_MANIFEST, payload)
    log(f"[freeze-predictions] {len(files)} files combined {payload['combined_prediction_sha256'][:16]}")
    return 0


def _load_pred(zone: str, key: str) -> np.ndarray:
    path = PRED_DIR / f"pred_{zone}_{key}.npz"
    with np.load(path) as z:
        return z["pred"].astype(np.float64)


# ---------------------------------------------------------------------------
# monitor gate
# ---------------------------------------------------------------------------


def _seed_errors(keys: Sequence[str], zone: str, y: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.abs(_load_pred(zone, k) - y) for k in keys}


def _delta_seedmean(keys_new: Sequence[str], keys_ref: Sequence[str], zone: str, y: np.ndarray) -> np.ndarray:
    e_new = np.mean([_seed_errors([k], zone, y)[k] for k in keys_new], axis=0)
    e_ref = np.mean([_seed_errors([k], zone, y)[k] for k in keys_ref], axis=0)
    return e_new - e_ref


def _contrast_report(zone: str, keys_new: Sequence[str], keys_ref: Sequence[str], y: np.ndarray) -> dict[str, Any]:
    e_new = {k: _seed_errors([k], zone, y)[k] for k in keys_new}
    e_ref = {k: _seed_errors([k], zone, y)[k] for k in keys_ref}
    delta = np.mean([e_new[k] for k in keys_new], axis=0) - np.mean([e_ref[k] for k in keys_ref], axis=0)
    out = {
        "zone": zone,
        "new": list(keys_new),
        "ref": list(keys_ref),
        "delta_seed_mean": float(delta.mean()),
        "per_seed_delta": {
            f"seed{k[1:]}": float(e_new[k].mean() - e_ref[r].mean()) for k, r in zip(keys_new, keys_ref)
        },
        "bootstrap": paired_bootstrap(delta),
        "ordinary_y_gt_-3": {
            "n": int((y > TAIL_THRESHOLD).sum()),
            "delta": float(delta[y > TAIL_THRESHOLD].mean()),
        },
        "tail_y_le_-3": {
            "n": int((y <= TAIL_THRESHOLD).sum()),
            "delta": float(delta[y <= TAIL_THRESHOLD].mean()) if (y <= TAIL_THRESHOLD).any() else None,
        },
        "top10_removed_delta": top_k_removed(
            delta,
            np.mean([e_new[k] for k in keys_new], axis=0),
            np.mean([e_ref[k] for k in keys_ref], axis=0),
        ),
        "rmse_new": float(np.sqrt(np.mean((np.mean([_load_pred(zone, k) for k in keys_new], axis=0) - y) ** 2))),
        "rmse_ref": float(np.sqrt(np.mean((np.mean([_load_pred(zone, k) for k in keys_ref], axis=0) - y) ** 2))),
        "mae_new_seed_mean": float(np.mean([e_new[k].mean() for k in keys_new])),
        "mae_ref_seed_mean": float(np.mean([e_ref[k].mean() for k in keys_ref])),
        "better_fraction": float((delta < 0).mean()),
    }
    return out


def cmd_gate_monitor(args: argparse.Namespace) -> int:
    verify_freeze()
    if LOCK_FILE.exists():
        raise SystemExit("locked read already happened; refusing to touch the monitor gate")
    split = read_split()
    monitor = list(map(int, split["monitor"]))
    y = _load_y_rows(monitor)
    comparison = _contrast_report("monitor", [f"P{s}" for s in SEEDS], [f"B{s}" for s in SEEDS], y)
    pretrain = [read_json(OUT / f"pretraining_seed{s}.json") for s in SEEDS]
    init_audit = _collect_init_audit()
    checks = {
        "seed_mean_delta_le_-0.005": comparison["delta_seed_mean"] <= -0.005,
        "both_seed_deltas_negative": all(v < 0 for v in comparison["per_seed_delta"].values()),
        "ordinary_delta_le_0": comparison["ordinary_y_gt_-3"]["delta"] <= 0.0,
        "top10_removed_delta_lt_0": comparison["top10_removed_delta"] < 0.0,
        "bootstrap_ci_upper_lt_0": comparison["bootstrap"]["ci_hi"] < 0.0,
        "no_leakage_or_protocol_deviation": bool(read_json(GATES_JSON)["all_pass"]),
        "encoder_changed_and_used": all(p["encoder_parameters_changed"] for p in pretrain)
        and all(init_audit["per_seed"][str(s)]["encoder_used_is_pretrained"] for s in SEEDS),
    }
    purchase = all(checks.values())
    gate_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "stage": "monitor_purchase_gate",
        "comparison": comparison,
        "checks": checks,
        "purchase_compute_control": bool(purchase),
        "status": "PURCHASED" if purchase else "NOT_PURCHASED",
        "label": "COMPUTE_CONTROL_PURCHASED" if purchase else "MASK_PRETRAIN_NO_COMPUTE_PURCHASE",
        "note": (
            "C is purchased only when every frozen check fires; C is an extra-supervision "
            "control (22680 updates), not a FLOP-matched causal control"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(MONITOR_GATE, gate_payload)
    write_json(SCRATCH_VS_PRETRAINED, {"protocol": PROTOCOL_ID, "stage": "scratch_vs_pretrained_monitor", "comparison": comparison, "pretraining": pretrain})
    log(f"[gate-monitor] delta {comparison['delta_seed_mean']:+.5f} per-seed {comparison['per_seed_delta']} -> {gate_payload['status']}")
    return 0


# ---------------------------------------------------------------------------
# initialization audit
# ---------------------------------------------------------------------------


def _collect_init_audit() -> dict[str, Any]:
    per_seed = {}
    for s in SEEDS:
        b = read_json(OUT / f"train_B_seed{s}.json")
        entry: dict[str, Any] = {
            "B_init_full_sha256": b["init_full_sha256"],
            "B_init_group_state_sha256": b["init_group_state_sha256"],
            "B_init_encoder_sha256": b["init_encoder_sha256"],
            "B_batch_order_hash": b["batch_order_hash"],
        }
        p_path = OUT / f"train_P_seed{s}.json"
        if p_path.exists():
            p = read_json(p_path)
            entry.update(
                {
                    "P_init_full_sha256": p["init_full_sha256"],
                    "P_init_group_state_sha256": p["init_group_state_sha256"],
                    "P_init_encoder_sha256": p["init_encoder_sha256"],
                    "P_encoder_used_sha256": p["encoder_used_sha256"],
                    "P_batch_order_hash": p["batch_order_hash"],
                    "same_init_full_model": b["init_full_sha256"] == p["init_full_sha256"],
                    "same_composer_reader_init": b["init_group_state_sha256"] == p["init_group_state_sha256"],
                    "same_batch_order": b["batch_order_hash"] == p["batch_order_hash"],
                    "encoder_used_is_pretrained": p["encoder_used_sha256"] == read_json(OUT / f"pretraining_seed{s}.json")["encoder_final_sha256"],
                    "encoder_moved_by_pretraining": p["encoder_used_sha256"] != p["init_encoder_sha256"],
                }
            )
        c_path = OUT / f"train_C_seed{s}.json"
        if c_path.exists():
            c = read_json(c_path)
            fit_n = len(read_split()["fit8k"])
            entry.update(
                {
                    "C_init_full_sha256": c["init_full_sha256"],
                    "C_same_init_as_B": c["init_full_sha256"] == b["init_full_sha256"],
                    "C_updates": c["updates"],
                    "C_batch_order_prefix_matches_B": (
                        int(c["epochs"]) >= int(b["epochs"])
                        and batch_order_hash(s, fit_n, BATCH_SIZE, int(b["epochs"])) == b["batch_order_hash"]
                        and batch_order_hash(s, fit_n, BATCH_SIZE, int(c["epochs"])) == c["batch_order_hash"]
                    ),
                }
            )
        per_seed[str(s)] = entry
    return {"protocol": PROTOCOL_ID, "per_seed": per_seed}


def cmd_init_audit(args: argparse.Namespace) -> int:
    payload = _collect_init_audit()
    payload["git_commit"] = git_commit()
    write_json(INIT_AUDIT, payload)
    log(f"[init-audit] written {INIT_AUDIT}")
    return 0


# ---------------------------------------------------------------------------
# locked scoring (the ONE label read)
# ---------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    if LOCK_FILE.exists():
        raise SystemExit("locked-dev labels already read once; refusing a second read")
    manifest = read_json(PRED_MANIFEST)
    for key, entry in manifest["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"frozen prediction hash mismatch: {key}")
    split = read_split()
    monitor = list(map(int, split["monitor"]))
    locked = list(map(int, split["locked"]))
    y_mon = _load_y_rows(monitor)
    y_lock = load_locked_y(locked)
    np.save(OUT / "locked_y_read_once.npy", y_lock.astype(np.float64))
    # write the lock immediately after the single read
    LOCK_FILE.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_ID,
                "git_commit": git_commit(),
                "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_locked": len(locked),
                "locked_rows_sha256": rows_sha256(locked),
                "locked_y_copy_sha256": file_sha256(OUT / "locked_y_read_once.npy"),
                "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    available = sorted({key.split(":")[1] for key in manifest["prediction_files"] if key.startswith("locked:")})
    arms_present = sorted({k[0] for k in available})
    seeds_present = sorted({int(k[1:]) for k in available})

    model_metrics = {}
    for key in available:
        pred = _load_pred("locked", key)
        model_metrics[key] = metrics("locked", pred, y_lock)
    monitor_metrics = {}
    for key in sorted({key.split(":")[1] for key in manifest["prediction_files"] if key.startswith("monitor:")}):
        pred = _load_pred("monitor", key)
        monitor_metrics[key] = metrics("monitor", pred, y_mon)

    paired: dict[str, Any] = {}
    if "B" in arms_present and "P" in arms_present:
        paired["P_minus_B"] = _contrast_report("locked", [f"P{s}" for s in seeds_present], [f"B{s}" for s in seeds_present], y_lock)
    if "B" in arms_present and "C" in arms_present:
        paired["C_minus_B"] = _contrast_report("locked", [f"C{s}" for s in seeds_present], [f"B{s}" for s in seeds_present], y_lock)
    if "P" in arms_present and "C" in arms_present:
        paired["P_minus_C"] = _contrast_report("locked", [f"P{s}" for s in seeds_present], [f"C{s}" for s in seeds_present], y_lock)
    if "B" in arms_present and "P" in arms_present:
        paired["P_minus_B_monitor"] = _contrast_report("monitor", [f"P{s}" for s in seeds_present], [f"B{s}" for s in seeds_present], y_mon)

    fit_locked_gap = {}
    for key in available:
        arm, seed = key[0], key[1:]
        train = read_json(OUT / f"train_{arm}_seed{seed}.json")
        fit_locked_gap[key] = {
            "fit_mae": train["soup_fit_mae"],
            "monitor_mae": train["soup_monitor_mae"],
            "locked_mae": model_metrics[key]["locked_mae"],
            "fit_minus_locked": train["soup_fit_mae"] - model_metrics[key]["locked_mae"],
        }
    locked_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_locked": len(locked),
        "models": model_metrics,
        "arms_present": arms_present,
        "seeds_present": seeds_present,
        "paired": paired,
        "fit_locked_gap": fit_locked_gap,
        "top_k_removed_definition": "rows with the k largest max(|e_candidate|, |e_reference|) are removed",
        "bootstrap": {"n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "molecule"},
        "locked_labels_read": True,
        "single_read": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(LOCKED_METRICS, locked_payload)
    write_json(PAIRED_BOOTSTRAP, {"protocol": PROTOCOL_ID, "paired": paired, "n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED})
    write_json(MONITOR_METRICS, {"protocol": PROTOCOL_ID, "models": monitor_metrics})
    for key, m in model_metrics.items():
        log(f"[score] {key}: locked MAE {m['locked_mae']:.6f}")
    return 0


# ---------------------------------------------------------------------------
# transfer mechanism
# ---------------------------------------------------------------------------


def _effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = s**2 / max((s**2).sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def cmd_mechanism(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    norm = read_json(PREP_FIT8K)
    lock_objects = load_objects(OBJECTS_LOCKED)
    split = read_split()
    locked = list(map(int, split["locked"]))
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zone": "locked (label-blind forward)"}

    # M1: encoder movement + representation geometry (no labels needed)
    for seed in SEEDS:
        b_state = torch.load(OUT / f"soup_B_seed{seed}.pt", map_location="cpu", weights_only=False)
        p_state = torch.load(OUT / f"soup_P_seed{seed}.pt", map_location="cpu", weights_only=False)
        move = {}
        for group in ("atom_encoder", "bond_encoder", "junction_encoder"):
            num = 0.0
            den = 0.0
            for k, v in b_state.items():
                if k.startswith(group + "."):
                    d = (p_state[k].float() - v.float())
                    num += float(d.norm() ** 2)
                    den += float(v.float().norm() ** 2)
            move[group] = {"relative_move_b_to_p": math.sqrt(num / max(den, 1e-12)), "abs_move": math.sqrt(num)}
        payload.setdefault("M1_parameters", {})[f"seed{seed}"] = move
        # activations / effective rank of the atom encoder on locked structures
        model = _make_downstream_model(seed, norm, device)
        model.load_state_dict(b_state)
        model_p = _make_downstream_model(seed, norm, device)
        model_p.load_state_dict(p_state)
        batch = sro.batch_objects(lock_objects)
        with torch.no_grad():
            a_b = model.encode_atoms(batch).numpy()
            a_p = model_p.encode_atoms(batch).numpy()
            w_b = model.encode_junctions(model.encode_atoms(batch), model.encode_bonds(model.encode_atoms(batch), batch), batch).numpy()
            a_ = model_p.encode_atoms(batch)
            w_p = model_p.encode_junctions(a_, model_p.encode_bonds(a_, batch), batch).numpy()
        corr = float(np.corrcoef(a_b.reshape(-1), a_p.reshape(-1))[0, 1])
        payload.setdefault("M1_representation", {})[f"seed{seed}"] = {
            "atom_activation_scale_B": float(np.abs(a_b).mean()),
            "atom_activation_scale_P": float(np.abs(a_p).mean()),
            "atom_effective_rank_B": _effective_rank(a_b),
            "atom_effective_rank_P": _effective_rank(a_p),
            "junction_effective_rank_B": _effective_rank(w_b),
            "junction_effective_rank_P": _effective_rank(w_p),
            "atom_activation_correlation_B_P": corr,
        }
    # M2: optimization trajectories from the frozen logs
    m2 = {}
    for seed in SEEDS:
        b = read_json(OUT / f"train_B_seed{seed}.json")
        p_path = OUT / f"train_P_seed{seed}.json"
        if not p_path.exists():
            continue
        p = read_json(p_path)
        c_path = OUT / f"train_C_seed{seed}.json"
        entry = {
            "fit_mae_at": {},
            "monitor_mae_at": {},
        }
        for epoch in (1, 20, 60, 120, 180, 240):
            idx = epoch - 1
            if idx < len(b["curve"]):
                entry["fit_mae_at"][f"ep{epoch}"] = {"B": b["curve"][idx]["train_mae"], "P": p["curve"][idx]["train_mae"]}
                entry["monitor_mae_at"][f"ep{epoch}"] = {"B": b["curve"][idx]["monitor_mae"], "P": p["curve"][idx]["monitor_mae"]}
        entry["final_fit"] = {"B": b["soup_fit_mae"], "P": p["soup_fit_mae"]}
        entry["final_monitor"] = {"B": b["soup_monitor_mae"], "P": p["soup_monitor_mae"]}
        if c_path.exists():
            c = read_json(c_path)
            for epoch in (1, 20, 60, 120, 180, 240, 300, 360):
                idx = epoch - 1
                if idx < len(c["curve"]):
                    entry["fit_mae_at"].setdefault(f"ep{epoch}", {})["C"] = c["curve"][idx]["train_mae"]
                    entry["monitor_mae_at"].setdefault(f"ep{epoch}", {})["C"] = c["curve"][idx]["monitor_mae"]
            entry["final_fit"]["C"] = c["soup_fit_mae"]
            entry["final_monitor"]["C"] = c["soup_monitor_mae"]
        m2[f"seed{seed}"] = entry
    payload["M2_trajectories"] = m2
    # M3: structural subgroups (locked)
    if LOCKED_METRICS.exists():
        lm = read_json(LOCKED_METRICS)
        y = np.load(OUT / "locked_y_read_once.npy")
        assert file_sha256(OUT / "locked_y_read_once.npy") == read_json(LOCK_FILE)["locked_y_copy_sha256"]
        stats = {
            "ordinary": (y > TAIL_THRESHOLD),
            "tail": (y <= TAIL_THRESHOLD),
            "junction_rich": None,
            "ring_rich": None,
            "rare_atom_env": None,
            "rare_bond_env": None,
        }
        junc = np.asarray([o.junc_index.shape[0] for o in lock_objects])
        stats["junction_rich"] = junc > np.median(junc)
        ring = np.asarray([bool(o.atom_feat[:, sro.atom_blocks()["in_ring"].start].any()) for o in lock_objects])
        stats["ring_rich"] = ring
        # rare atom environment: contains an atom type that is rare in Fit8k
        fit_objs = load_objects(OBJECTS_FIT8K)
        counts = np.zeros(sro.ATOM_CATEGORIES, dtype=np.int64)
        for o in fit_objs:
            codes = o.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1)
            counts += np.bincount(codes, minlength=sro.ATOM_CATEGORIES)
        rare = set(np.where(counts < 200)[0].tolist())
        lock_codes = [set(o.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1).tolist()) for o in lock_objects]
        stats["rare_atom_env"] = np.asarray([len(c & rare) > 0 for c in lock_codes])
        m3 = {}
        for name, mask in stats.items():
            entry = {"n": int(np.asarray(mask).sum())}
            if entry["n"]:
                for key in ("P0", "P1", "B0", "B1"):
                    if key in lm["models"]:
                        entry[key] = float(np.abs(_load_pred("locked", key)[np.asarray(mask)] - y[np.asarray(mask)]).mean())
                if "B0" in entry and "P0" in entry:
                    entry["delta_seed0"] = entry["P0"] - entry["B0"]
                if "B1" in entry and "P1" in entry:
                    entry["delta_seed1"] = entry["P1"] - entry["B1"]
            m3[name] = entry
        payload["M3_subgroups"] = m3
        # persistent-error similarity: overlap of top-10% errors
        n = len(y)
        sets = {}
        for key in lm["models"]:
            err = np.abs(_load_pred("locked", key) - y)
            sets[key] = set(np.argsort(-err)[: max(1, n // 10)].tolist())
        payload["M3_error_overlap"] = {
            "B0_B1": len(sets["B0"] & sets["B1"]),
            "P0_P1": len(sets["P0"] & sets["P1"]),
            "B0_P0": len(sets["B0"] & sets["P0"]),
            "B1_P1": len(sets["B1"] & sets["P1"]),
        }
    # M4: negative-transfer checks
    pretext = [read_json(OUT / f"pretraining_seed{s}.json") for s in SEEDS]
    feats = read_json(FEASIBILITY_AUDIT)
    payload["M4_negative_transfer"] = {
        "pretext_learnable": all(p["final_ce_atom"] < feats["masked_position_priors_visit0_500mol"]["atom"]["marginal_ce_nats"] and p["final_ce_bond"] < feats["masked_position_priors_visit0_500mol"]["bond"]["marginal_ce_nats"] for p in pretext),
        "pretext_final_ce": {f"seed{p['seed']}": {"atom": p["final_ce_atom"], "bond": p["final_ce_bond"]} for p in pretext},
        "prior_ce": {
            "atom": feats["masked_position_priors_visit0_500mol"]["atom"]["marginal_ce_nats"],
            "bond": feats["masked_position_priors_visit0_500mol"]["bond"]["marginal_ce_nats"],
        },
        "trivial_shortcut_check": "own masked slots hold only the mask token; hidden truths live outside the model input (see the swap/rebuild audits)",
    }
    write_json(TRANSFER_MECHANISM, payload)
    log("[mechanism] transfer_mechanism.json written")
    return 0


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def cmd_decision(args: argparse.Namespace) -> int:
    if not LOCKED_METRICS.exists():
        raise SystemExit("no locked metrics; run score first")
    lm = read_json(LOCKED_METRICS)
    gate = read_json(MONITOR_GATE) if MONITOR_GATE.exists() else {"status": "NOT_PURCHASED", "purchase_compute_control": False}
    pretrain = [read_json(OUT / f"pretraining_seed{s}.json") for s in SEEDS]
    gates = read_json(GATES_JSON)
    swap = read_json(SWAP_AUDIT)
    arm_keys = lm["arms_present"]
    seeds = lm["seeds_present"]
    branch = "INCONCLUSIVE_OR_BLOCKED"
    details: dict[str, Any] = {}
    if not gates["all_pass"] or swap["verdict"] != "MASK_SWAP_INVARIANT":
        branch = "PRETEXT_TASK_TRIVIAL_OR_LEAKING"
    elif "P" in arm_keys and "B" in arm_keys and len(seeds) == 2:
        pb = lm["paired"]["P_minus_B"]
        per_seed = list(pb["per_seed_delta"].values())
        ordinary = pb["ordinary_y_gt_-3"]["delta"]
        top10 = pb["top10_removed_delta"]
        ci_hi = pb["bootstrap"]["ci_hi"]
        feats = read_json(FEASIBILITY_AUDIT)
        prior_ce = {
            "atom": feats["masked_position_priors_visit0_500mol"]["atom"]["marginal_ce_nats"],
            "bond": feats["masked_position_priors_visit0_500mol"]["bond"]["marginal_ce_nats"],
        }
        pretext_beats_prior = all(
            p["final_ce_atom"] < prior_ce["atom"] and p["final_ce_bond"] < prior_ce["bond"] for p in pretrain
        )
        details = {
            "delta_locked_seed_mean": pb["delta_seed_mean"],
            "per_seed": pb["per_seed_delta"],
            "ordinary_delta": ordinary,
            "tail_delta": pb["tail_y_le_-3"]["delta"],
            "top10_removed_delta": top10,
            "bootstrap_ci95": [pb["bootstrap"]["ci_lo"], ci_hi],
            "pc": lm["paired"].get("P_minus_C"),
            "cb": lm["paired"].get("C_minus_B"),
            "pretext_beats_prior": bool(pretext_beats_prior),
            "prior_ce": prior_ce,
        }
        useful = (
            pretext_beats_prior
            and all(v < 0 for v in per_seed)
            and pb["delta_seed_mean"] <= -0.005
            and ordinary <= 0.0
            and top10 < 0.0
            and ci_hi < 0.0
        )
        if gate.get("purchase_compute_control") and "P_minus_C" in lm["paired"]:
            useful = useful and lm["paired"]["P_minus_C"]["delta_seed_mean"] <= 0.0
        if useful:
            branch = "SELF_SUPERVISED_PRETRAINING_USEFUL"
        elif not pretext_beats_prior:
            branch = "PRETEXT_TASK_TRIVIAL_OR_LEAKING" if not gates["all_pass"] else "NO_USEFUL_TRANSFER"
        elif any(v >= 0.0 for v in per_seed) and pb["delta_seed_mean"] > 0.05:
            branch = "PRETRAINING_NEGATIVE_TRANSFER"
        elif all(v >= -0.005 for v in per_seed) and pb["delta_seed_mean"] > -0.005:
            branch = "NO_USEFUL_TRANSFER"
        else:
            branch = "PRETRAINING_ONLY_ACCELERATES_OPTIMIZATION"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "branch": branch,
        "details": details,
        "compute_control": gate.get("status", "NOT_PURCHASED"),
        "compute_control_gate": gate,
        "hard_constraints": {
            "no_extra_y_labels": True,
            "no_extra_molecules": True,
            "downstream_architecture_unchanged": True,
            "strict_masked_target_isolation": bool(gates["all_pass"] and swap["verdict"] == "MASK_SWAP_INVARIANT"),
            "official_valid_test_never_loaded": True,
        },
        "answers": {
            "1_mask_leak_free": bool(gates["all_pass"]),
            "2_pretext_beats_prior": details.get("pretext_beats_prior", False),
            "3_improves_raw_y_generalization": details.get("delta_locked_seed_mean"),
            "4_exceeds_compute_control": (details.get("pc") or {}).get("delta_seed_mean") if details.get("pc") else "NOT_PURCHASED",
            "5_cross_seed": details.get("per_seed"),
            "6_ordinary_molecules": details.get("ordinary_delta"),
            "7_tail": details.get("tail_delta"),
            "8_few_outliers": details.get("top10_removed_delta"),
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(SCIENTIFIC_DECISION, payload)
    log(f"[decision] {branch}")
    return 0


# ---------------------------------------------------------------------------
# remote phases
# ---------------------------------------------------------------------------


def _ns(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def cmd_phase_f(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if not OBJECTS_MANIFEST.exists():
        cmd_objects(_ns())
    if not DEPENDENCY_AUDIT.exists():
        cmd_audit_dependency(_ns())
    if not SWAP_AUDIT.exists():
        cmd_audit_swap(_ns(random_cases=200, positive_cases=50))
    if not REBUILD_AUDIT.exists():
        cmd_rebuild_audit(_ns(cases=100))
    if not FEASIBILITY_AUDIT.exists():
        cmd_feasibility(_ns())
    gates_ok = GATES_JSON.exists() and bool(read_json(GATES_JSON).get("all_pass"))
    if not gates_ok:
        cmd_gates(_ns())
    for seed in SEEDS:
        if not (OUT / f"pretraining_seed{seed}.json").exists():
            pretrain_one_seed(seed, device)
    for seed in SEEDS:
        if not (OUT / f"train_B_seed{seed}.json").exists():
            train_downstream("B", seed, device)
    for seed in SEEDS:
        if not (OUT / f"train_P_seed{seed}.json").exists():
            train_downstream("P", seed, device)
    models = [f"{arm}{s}" for arm in ("B", "P") for s in SEEDS]
    if not all((PRED_DIR / f"pred_locked_{k}.npz").exists() for k in models):
        cmd_predict(_ns(device="cpu", models=models))
    cmd_init_audit(_ns())
    return 0


def cmd_phase_g(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    gate = read_json(MONITOR_GATE)
    if not gate.get("purchase_compute_control"):
        raise SystemExit("compute control not purchased by the frozen monitor gate")
    for seed in SEEDS:
        if not (OUT / f"train_C_seed{seed}.json").exists():
            train_downstream("C", seed, device)
    models = [f"C{s}" for s in SEEDS]
    if not all((PRED_DIR / f"pred_locked_{k}.npz").exists() for k in models):
        cmd_predict(_ns(device="cpu", models=models))
    cmd_init_audit(_ns())
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("splits").set_defaults(func=cmd_splits)
    sub.add_parser("objects").set_defaults(func=cmd_objects)
    sub.add_parser("audit-dependency").set_defaults(func=cmd_audit_dependency)
    p = sub.add_parser("audit-swap")
    p.add_argument("--random-cases", type=int, default=200)
    p.add_argument("--positive-cases", type=int, default=50)
    p.set_defaults(func=cmd_audit_swap)
    p = sub.add_parser("rebuild-audit")
    p.add_argument("--cases", type=int, default=100)
    p.set_defaults(func=cmd_rebuild_audit)
    sub.add_parser("feasibility").set_defaults(func=cmd_feasibility)
    sub.add_parser("bench").set_defaults(func=cmd_bench)
    sub.add_parser("gates").set_defaults(func=cmd_gates)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    p = sub.add_parser("pretrain")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda a: (pretrain_one_seed(a.seed, torch.device(a.device)), 0)[1])
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=["B", "P", "C"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda a: (train_downstream(a.arm, a.seed, torch.device(a.device)), 0)[1])
    p = sub.add_parser("predict")
    p.add_argument("--models", nargs="+", required=True)
    p.set_defaults(func=cmd_predict)
    sub.add_parser("freeze-predictions").set_defaults(func=cmd_freeze_predictions)
    sub.add_parser("gate-monitor").set_defaults(func=cmd_gate_monitor)
    sub.add_parser("init-audit").set_defaults(func=cmd_init_audit)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("mechanism").set_defaults(func=cmd_mechanism)
    sub.add_parser("decision").set_defaults(func=cmd_decision)
    p = sub.add_parser("phase-f")
    p.set_defaults(func=cmd_phase_f)
    p = sub.add_parser("phase-g")
    p.set_defaults(func=cmd_phase_g)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
