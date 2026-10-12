"""cscl-bjoc-full10k-pretraining-transfer-v1 driver.

Protocol: ``configs/bjoc_full10k_pretraining_transfer_v1.yaml`` (frozen before
any GPU trajectory).  Track: ``tracks/structure-contribution``.

Single scientific question
--------------------------
The validated leak-free masked atom/bond pretraining (``cscl-bjoc-masked-
pretraining-v1``) improved locked-dev raw-y MAE on an 8k internal fit split.
Does the *same frozen* pretraining still improve single-model raw-y MAE when
the downstream model is trained on the **complete official train 10000** and
scored once on the **official valid 1000**?

Design
------
* No new internal split: official-train 10000 rows in their original order for
  both SSL pretraining (structures only) and raw-y supervised training; the
  official valid 1000 is the exposed development set.
* Arms (all BJOC-Null, 413,505 params, raw-y L1, Adam 1e-3 / wd 1e-5 /
  batch 128 / clip 5, 240 epochs = 18,960 updates, last-5 soup):
    - B: scratch full-10k (historical full-10k checkpoints reused iff the
      frozen reuse audit proves recipe equivalence);
    - P: three encoders initialised from the frozen SSL state;
    - Q (conditional): same as P but the three encoders fine-tune at 3e-4
      while composer/reader stay at 1e-3  (Q is purchased only if the
      label-blind representation gate fires).
* SSL: the frozen ``bjoc_masked_pretraining_v1`` pretext (15% atoms + 15%
  physical bonds, mask tokens 27/0, full rebuild from the masked raw graph,
  CE + CE 1.0/1.0, 7560 updates at batch 128 = 967,680 molecule-visits).
* Phase F discipline: every prediction file is written and hashed first; the
  official-valid labels are read exactly once afterwards, in ``score``.
  Official test is never loaded in any form.

Stages
------
``objects``          official-train-10k + official-valid structure caches,
                     train-only global standardiser (labels never touched)
``consistency``      Phase-A data-consistency + mask-token audit
``gates``            swap / rebuild / isolation / model correctness gates
``freeze``           training manifest (all hashes frozen before GPU)
``reuse-audit``      Phase-B audit of the historical full-10k BJOC-Null
``stage-b``          copy the reused B checkpoints (or retrain if audit fails)
``pretrain``         masked SSL seed 0/1 on official train 10k (no y)
``train``            downstream arm B/P/Q seed 0/1
``predict``          label-blind official-valid forward for frozen soups
``freeze-predictions`` hash every prediction file before any valid y read
``diagnostics``      Phase-D label-blind representation transfer analysis
``q-gate``           Phase-E conditional purchase gate (no training)
``score``            the ONE official-valid label read + metrics + bootstrap
``error-analysis``   Phase-G/H error structure and common errors
``decision``         the frozen scientific decision
``stage-remote-prep``  CPU remote chain: objects + audits + freeze + reuse +
                       stage-b + predict B
``stage-ssl-p``        GPU chain per seed: pretrain s, train P s, predict P s
``stage-q``            GPU chain per seed: train Q s, predict Q s
"""

from __future__ import annotations

import argparse
import ast
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
import run_bjoc_masked_pretraining_v1 as mpr  # noqa: E402  (audit helpers only)
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-full10k-pretraining-transfer-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_full10k_pretraining_transfer_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
HISTORICAL_BJOC_DIR = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1"
HISTORICAL_TERMINAL_DIR = REPO_ROOT / "tracks/structure-contribution/results/bjoc_predictor_terminal_v1"

# ---- official data (no internal split) ------------------------------------
N_TRAIN = 10000
N_VALID = 1000
SEEDS = (0, 1)

# ---- frozen downstream recipe (identical to bjoc-masked-pretraining-v1) ---
BATCH_SIZE = 128
EPOCHS = 240
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
LR = 1.0e-3
ENCODER_LR_Q = 3.0e-4
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)  # 79
UPDATES = EPOCHS * STEPS_PER_EPOCH                    # 18960

# ---- frozen analysis constants --------------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TOP_K_REMOVED = (1, 5, 10)
TAIL_THRESHOLD = -3.0
EXTREME_VALID_ROW = 172      # valid:0172 (historically exposed extreme row)
Q_CKA_THRESHOLD = 0.75
Q_RANK_DROP_THRESHOLD = 0.25
Q_RANK_REPLICATIONS = 5
Q_RANK_REPLICATION_SIZE = 1000

# ---- artifacts -------------------------------------------------------------
OBJECTS_TRAIN = OUT / "objects_train10k.pt"
OBJECTS_VALID = OUT / "objects_valid.pt"
PREP_TRAIN = OUT / "prep_train10k.json"
OBJECTS_MANIFEST = OUT / "objects_manifest.json"
CONSISTENCY_AUDIT = OUT / "consistency_audit.json"
GATES_JSON = OUT / "correctness_gates.json"
SWAP_AUDIT = OUT / "mask_target_swap_audit.json"
REBUILD_AUDIT = OUT / "masked_object_rebuild_audit.json"
DEPENDENCY_AUDIT = OUT / "masked_feature_dependency_audit.json"
TRAINING_MANIFEST = OUT / "training_manifest.json"
REUSE_AUDIT = OUT / "reuse_audit.json"
INIT_AUDIT = OUT / "initialization_audit.json"
PRED_DIR = OUT / "predictions"
PRED_FREEZE = OUT / "prediction_freeze.json"
PRED_ACCESS = OUT / "prediction_access_log.json"
REPRESENTATION = OUT / "representation_diagnostics.json"
Q_GATE = OUT / "q_purchase_gate.json"
VALID_METRICS = OUT / "valid_metrics.json"
PAIRED_BOOTSTRAP = OUT / "paired_bootstrap.json"
ERROR_ANALYSIS = OUT / "error_analysis.json"
SCIENTIFIC_DECISION = OUT / "scientific_decision.json"
VALID_LOCK = OUT / "valid_read.lock"
VALID_Y_COPY = OUT / "valid_y_read_once.npy"

ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
ENCODED_VALID = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_valid.pt"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# io / hashing helpers
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
    k = len(delta)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, k, k)
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


def row_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - y)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "median_abs": float(np.median(err)),
        "p90_abs": float(np.quantile(err, 0.90)),
        "p95_abs": float(np.quantile(err, 0.95)),
        "p99_abs": float(np.quantile(err, 0.99)),
        "max_abs": float(err.max()),
    }


def top_k_removed(delta: np.ndarray, e_new: np.ndarray, e_ref: np.ndarray, k: int) -> dict[str, Any]:
    worst = np.argsort(-np.maximum(e_new, e_ref))[: int(k)]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    return {
        f"delta_top{k}_removed": float(delta[keep].mean()),
        f"worst_rows_top{k}": [int(i) for i in worst.tolist()],
    }


# ---------------------------------------------------------------------------
# data loading (official splits only; test never referenced)
# ---------------------------------------------------------------------------


def load_train_dataset():
    ds = cf.load_official_train()
    assert len(ds) == N_TRAIN, len(ds)
    return ds


def load_valid_dataset():
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    assert len(ds) == N_VALID, len(ds)
    return ds


def load_objects(path: Path) -> list[sro.MolObjects]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_objects(rows: Sequence[int], ds) -> list[sro.MolObjects]:
    return [
        sro.build_mol_objects(int(i), ds[int(i)].x.numpy(), ds[int(i)].edge_index.numpy(), ds[int(i)].edge_attr.numpy())
        for i in rows
    ]


# ---------------------------------------------------------------------------
# objects + train-only standardiser
# ---------------------------------------------------------------------------


def _wl_fingerprint(obj: sro.MolObjects) -> str:
    import networkx as nx
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

    graph = nx.Graph()
    graph.add_nodes_from(range(obj.n_atoms))
    for i in range(obj.n_atoms):
        graph.nodes[i]["label"] = int(obj.atom_feat[i, : sro.ATOM_CATEGORIES].argmax())
    for k, (u, v) in enumerate(np.asarray(obj.bond_index)):
        graph.add_edge(int(u), int(v), label=int(obj.bond_type[k]))
    return weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="label", iterations=3)


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_train_dataset()
    val_ds = load_valid_dataset()
    t0 = time.perf_counter()
    train_objs = build_objects(range(N_TRAIN), ds)
    build_seconds = time.perf_counter() - t0
    valid_objs = build_objects(range(N_VALID), val_ds)
    assert [o.mol_index for o in train_objs] == list(range(N_TRAIN))
    assert [o.mol_index for o in valid_objs] == list(range(N_VALID))
    torch.save(train_objs, OBJECTS_TRAIN)
    torch.save(valid_objs, OBJECTS_VALID)

    gfit = np.stack([o.global_feat for o in train_objs], 0).astype(np.float64)
    mean = gfit.mean(0)
    scale = gfit.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    write_json(
        PREP_TRAIN,
        {
            "protocol": PROTOCOL_ID,
            "fit_only": True,
            "n_fit": N_TRAIN,
            "global_mean": mean.tolist(),
            "global_scale": scale.tolist(),
            "names": list(sro.GLOBAL_FEAT_NAMES),
            "official_valid_statistics_used": False,
            "official_test_statistics_used": False,
        },
    )

    # label-blind duplicate-structure check (WL fingerprints, no y touched)
    train_fp = [_wl_fingerprint(o) for o in train_objs]
    valid_fp = [_wl_fingerprint(o) for o in valid_objs]
    train_map: dict[str, list[int]] = {}
    for i, fp in enumerate(train_fp):
        train_map.setdefault(fp, []).append(i)
    shared = sorted(set(train_map) & set(valid_fp))
    shared_examples = [
        {"wl": fp, "train_rows": train_map[fp][:5], "valid_rows": [i for i, f in enumerate(valid_fp) if f == fp][:5]}
        for fp in shared[:10]
    ]

    counts = {
        "n_atoms": int(sum(o.n_atoms for o in train_objs)),
        "n_bonds": int(sum(o.bond_index.shape[0] for o in train_objs)),
        "n_junctions": int(sum(o.junc_index.shape[0] for o in train_objs)),
    }
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "build_seconds": build_seconds,
        "train_rows_sha256": rows_sha256(range(N_TRAIN)),
        "valid_rows_sha256": rows_sha256(range(N_VALID)),
        "objects_train10k_sha256": file_sha256(OBJECTS_TRAIN),
        "objects_valid_sha256": file_sha256(OBJECTS_VALID),
        "prep_train10k_sha256": file_sha256(PREP_TRAIN),
        "train_counts": counts,
        "global_mean_sha256": array_sha256(mean),
        "global_scale_sha256": array_sha256(scale),
        "train_valid_wl_shared_fingerprints": len(shared),
        "train_valid_wl_shared_examples": shared_examples,
        "labels_read_during_object_build": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OBJECTS_MANIFEST, manifest)
    log(f"[objects] train10k {counts} build {build_seconds:.1f}s; WL train/valid shared={len(shared)}")
    log(f"[objects] objects {manifest['objects_train10k_sha256'][:16]} prep {manifest['prep_train10k_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# Phase-A consistency audit
# ---------------------------------------------------------------------------


def cmd_consistency(args: argparse.Namespace) -> int:
    ds = load_train_dataset()
    val_ds = load_valid_dataset()
    train_objs = load_objects(OBJECTS_TRAIN)
    valid_objs = load_objects(OBJECTS_VALID)
    manifest = read_json(OBJECTS_MANIFEST)
    checks: dict[str, Any] = {}
    ok = True

    def gate(name: str, passed: bool, detail: Any) -> None:
        nonlocal ok
        checks[name] = {"pass": bool(passed), "detail": detail}
        ok = ok and bool(passed)

    # 1. full rebuild equality on a deterministic row sample + all valid rows
    rng = np.random.default_rng(20261013)
    sample = sorted(int(x) for x in rng.choice(N_TRAIN, 300, replace=False))
    max_diff: dict[str, float] = {}
    rebuild_mismatch = 0
    for i in sample:
        d = ds[i]
        o = sro.build_mol_objects(i, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        h = train_objs[i]
        for f in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
            a, b = np.asarray(getattr(o, f)), np.asarray(getattr(h, f))
            if a.shape != b.shape:
                rebuild_mismatch += 1
                continue
            max_diff[f] = max(max_diff.get(f, 0.0), float(np.abs(a - b).max()) if a.size else 0.0)
        for f in ("bond_index", "bond_type", "junc_index", "junc_bond"):
            if not np.array_equal(getattr(o, f), getattr(h, f)):
                rebuild_mismatch += 1
    gate(
        "object_rebuild_deterministic",
        rebuild_mismatch == 0,
        {"rows": len(sample), "mismatches": rebuild_mismatch, "max_abs_field_diff": max_diff},
    )

    # 2. directed-bond structure and unique physical bonds
    bond_ok = True
    junction_ok = True
    n_undirected = 0
    n_junc = 0
    for i in sample:
        d = ds[i]
        o = train_objs[i]
        ei = np.asarray(d.edge_index)
        ea = np.asarray(d.edge_attr).reshape(-1)
        pairs: dict[tuple[int, int], list[int]] = {}
        for col in range(ei.shape[1]):
            a, b = int(ei[0, col]), int(ei[1, col])
            pairs.setdefault((min(a, b), max(a, b)), []).append(int(ea[col]))
        if any(len(v) != 2 or v[0] != v[1] for v in pairs.values()):
            bond_ok = False
        if len(pairs) != o.bond_index.shape[0]:
            bond_ok = False
        for k, (u, v) in enumerate(np.asarray(o.bond_index)):
            if not (int(u) < int(v)) or pairs.get((int(u), int(v))) is None:
                bond_ok = False
            if pairs[(int(u), int(v))][0] != int(o.bond_type[k]):
                bond_ok = False
        n_undirected += o.bond_index.shape[0]
        # junction uniqueness / count
        junc = np.asarray(o.junc_index)
        if junc.shape[0]:
            triples = {tuple(int(z) for z in row) for row in junc}
            if len(triples) != junc.shape[0]:
                junction_ok = False
            degree = np.zeros(o.n_atoms, dtype=np.int64)
            for u, v in np.asarray(o.bond_index):
                degree[int(u)] += 1
                degree[int(v)] += 1
            if junc.shape[0] != int(sum(dg * (dg - 1) // 2 for dg in degree)):
                junction_ok = False
            for row in junc:
                a, j, b = int(row[0]), int(row[1]), int(row[2])
                if not (a < b) or j == a or j == b:
                    junction_ok = False
        n_junc += o.junc_index.shape[0]
    gate("directed_bond_structure", bond_ok, {"rows": len(sample), "n_undirected_bonds": n_undirected})
    gate("junction_uniqueness_and_count", junction_ok, {"rows": len(sample), "n_junctions": n_junc})

    # 3. vocabulary + mask-token slots on the FULL official train 10k
    atom_codes: set[int] = set()
    bond_codes: set[int] = set()
    for i in range(N_TRAIN):
        atom_codes |= {int(v) for v in np.asarray(ds[i].x).reshape(-1)}
        bond_codes |= {int(v) for v in np.asarray(ds[i].edge_attr).reshape(-1)}
    atom_conflict = mp.ATOM_MASK_TOKEN in atom_codes
    bond_conflict = mp.BOND_MASK_TOKEN in bond_codes
    vocab_ok = all(0 <= c < sro.ATOM_CATEGORIES for c in atom_codes) and all(0 <= c < sro.BOND_CATEGORIES for c in bond_codes)
    gate(
        "vocabulary_and_mask_tokens_free",
        vocab_ok and not atom_conflict and not bond_conflict,
        {
            "atom_codes_present": sorted(atom_codes),
            "bond_codes_present": sorted(bond_codes),
            "atom_mask_token": mp.ATOM_MASK_TOKEN,
            "bond_mask_token": mp.BOND_MASK_TOKEN,
            "atom_token_conflict": atom_conflict,
            "bond_token_conflict": bond_conflict,
            "n_train_rows_scanned": N_TRAIN,
        },
    )

    # 4. normalisation fit on train 10k only and reproducible
    recomputed = np.stack([o.global_feat for o in train_objs], 0).astype(np.float64)
    mean = recomputed.mean(0)
    scale = np.where(recomputed.std(0) < 1e-6, 1.0, recomputed.std(0))
    prep = read_json(PREP_TRAIN)
    norm_ok = (
        float(np.abs(mean - np.asarray(prep["global_mean"])).max()) == 0.0
        and float(np.abs(scale - np.asarray(prep["global_scale"])).max()) == 0.0
        and prep["n_fit"] == N_TRAIN
        and prep["official_valid_statistics_used"] is False
        and prep["official_test_statistics_used"] is False
    )
    gate("global_norm_train_only_reproducible", norm_ok, {"n_fit": prep["n_fit"], "fit_only": prep["fit_only"]})

    # 5. no label can live inside the object caches
    fields = set(sro.MolObjects.__dataclass_fields__)
    gate(
        "object_cache_has_no_label_field",
        "y" not in fields and not hasattr(train_objs[0], "y"),
        {"fields": sorted(fields)},
    )

    # 6. rebuildable / row alignment of every training object
    aligned = all(int(o.mol_index) == i for i, o in enumerate(train_objs)) and all(
        int(o.mol_index) == i for i, o in enumerate(valid_objs)
    )
    gate("row_alignment_10000_1000", aligned, {"n_train": len(train_objs), "n_valid": len(valid_objs)})

    # 7. duplicate structures inside train (canonical SMILES) and train/valid WL overlap
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(s, []).append(i)
    dup_groups = {s: r for s, r in groups.items() if len(r) > 1}
    gate(
        "duplicate_structure_disclosure",
        True,
        {
            "train_duplicate_smiles_groups": len(dup_groups),
            "train_duplicate_rows": int(sum(len(v) for v in dup_groups.values())),
            "train_duplicate_examples": [{"smiles": s, "rows": r} for s, r in list(dup_groups.items())[:5]],
            "train_valid_wl_shared_fingerprints": manifest["train_valid_wl_shared_fingerprints"],
            "train_valid_wl_shared_examples": manifest["train_valid_wl_shared_examples"],
            "action": "disclosed only; no benchmark row deleted and no official metric changed",
        },
    )

    # 8. official test never referenced
    source = Path(__file__).read_text(encoding="utf-8")
    # assemble the forbidden snippets at runtime so this scanner does not match itself
    quoted_test = "test"
    bad = ["split=" + '"' + quoted_test + '"', "split=" + "'" + quoted_test + "'"]
    bad = [t for t in bad if t in source]
    gate("official_test_not_referenced", not bad, {"forbidden_snippets_found": bad})

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "all_pass": bool(ok),
        "checks": checks,
        "official_valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(CONSISTENCY_AUDIT, payload)
    log(f"[consistency] all_pass={ok}")
    if not ok:
        raise SystemExit("Phase-A consistency gate failed")
    return 0


# ---------------------------------------------------------------------------
# dependency / swap / rebuild audits (reuse the frozen primitives)
# ---------------------------------------------------------------------------


def cmd_audit_dependency(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_train_dataset()
    atom_codes: set[int] = set()
    bond_codes: set[int] = set()
    for i in range(N_TRAIN):
        atom_codes |= {int(v) for v in np.asarray(ds[i].x).reshape(-1)}
        bond_codes |= {int(v) for v in np.asarray(ds[i].edge_attr).reshape(-1)}
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
            "junction_dim": sro.JUNC_FEAT_DIM,
            "global_dim": sro.GLOBAL_DIM,
        },
        "mask_token_scan_train10k": {
            "atom_mask_token": mp.ATOM_MASK_TOKEN,
            "bond_mask_token": mp.BOND_MASK_TOKEN,
            "atom_codes_present": sorted(atom_codes),
            "bond_codes_present": sorted(bond_codes),
            "atom_token_conflict": bool(atom_conflict),
            "bond_token_conflict": bool(bond_conflict),
            "n_rows_scanned": N_TRAIN,
        },
        "construction_rule": (
            "replace the raw atom/bond codes by the mask tokens on a copy of the raw PyG arrays, then "
            "rebuild every BJOC object field with build_mol_objects (full rebuild); each physical bond "
            "replaces BOTH directed PyG records"
        ),
        "verdict": "MASK_TARGET_LEAKAGE_BLOCKED" if (atom_conflict or bond_conflict) else "MASK_TOKEN_SLOTS_FREE",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(DEPENDENCY_AUDIT, payload)
    log(f"[audit-dependency] verdict {payload['verdict']} (atom codes {sorted(atom_codes)}, bond codes {sorted(bond_codes)})")
    if atom_conflict or bond_conflict:
        raise SystemExit("MASK_TARGET_LEAKAGE_BLOCKED")
    return 0


def cmd_audit_swap(args: argparse.Namespace) -> int:
    ds = load_train_dataset()
    rng = np.random.default_rng(20261013)
    atom_pass = bond_pass = atom_cases = bond_cases = 0
    visible_ok = True
    visible_checked = 0
    for trial in range(int(args.random_cases)):
        i = int(rng.integers(0, N_TRAIN))
        d = ds[i]
        x_r = d.x.numpy()
        ei_r = d.edge_index.numpy()
        ea_r = d.edge_attr.numpy()
        obj_r = sro.build_mol_objects(i, x_r, ei_r, ea_r)
        if obj_r.n_atoms < 2 or obj_r.bond_index.shape[0] < 1:
            continue
        s0 = mp.build_masked_sample(obj_r, x_r, ei_r, ea_r, seed=7, visit_index=trial)
        if len(s0.atom_ids):
            x_b = x_r.copy().reshape(-1)
            aid = int(s0.atom_ids[0])
            old = int(x_b[aid])
            new = (old + 1) % 27
            if new == mp.ATOM_MASK_TOKEN:
                new = (old + 2) % 27
            x_b[aid] = new
            s1 = mp.build_masked_sample(obj_r, x_b.reshape(x_r.shape), ei_r, ea_r, seed=7, visit_index=trial)
            atom_cases += 1
            if all(mpr._swap_case(s0, s1).values()):
                atom_pass += 1
        if len(s0.bond_ids):
            vid = int(s0.bond_ids[0])
            old = int(obj_r.bond_type[vid])
            new = 3 if old != 3 else 1
            ea_b = ea_r.copy()
            u, v = int(obj_r.bond_index[vid][0]), int(obj_r.bond_index[vid][1])
            for col in range(ei_r.shape[1]):
                a, b = int(ei_r[0, col]), int(ei_r[1, col])
                if (min(a, b), max(a, b)) == (min(u, v), max(u, v)):
                    ea_b[col] = new
            obj_variant = sro.build_mol_objects(i, x_r, ei_r, ea_b)
            s2 = mp.build_masked_sample(obj_variant, x_r, ei_r, ea_b, seed=7, visit_index=trial)
            bond_cases += 1
            if all(mpr._swap_case(s0, s2).values()):
                bond_pass += 1
        if trial < int(args.positive_cases):
            n_visible = [a for a in range(obj_r.n_atoms) if a not in set(map(int, s0.atom_ids))]
            if n_visible:
                v = int(rng.choice(n_visible))
                x_v = x_r.copy().reshape(-1)
                x_v[v] = (int(x_v[v]) + 1) % 27
                if int(x_v[v]) == mp.ATOM_MASK_TOKEN:
                    x_v[v] = (int(x_v[v]) + 1) % 27
                s3 = mp.build_masked_sample(obj_r, x_v.reshape(x_r.shape), ei_r, ea_r, seed=7, visit_index=trial)
                dep = mp.dependent_cell_mask(s0.masked, [v], [])
                diff = np.asarray(s0.masked.atom_feat) != np.asarray(s3.masked.atom_feat)
                changed = diff.any(axis=1)
                if int((changed & ~dep["atom"].any(axis=1)).sum()) or int(changed.sum()) == 0:
                    visible_ok = False
                visible_checked += 1
    witness = {}
    tiny = mpr._small_molecule_objects()[0]
    a0 = mp.build_masked_sample(tiny[0], tiny[1], tiny[2], tiny[3], seed=0, visit_index=0)
    x_b = tiny[1].copy()
    x_b[a0.atom_ids[0], 0] = (int(tiny[1][a0.atom_ids[0], 0]) + 5) % 27
    a1 = mp.build_masked_sample(tiny[0], x_b, tiny[2], tiny[3], seed=0, visit_index=0)
    witness["atom_swap"] = {
        "target_before": a0.atom_types.tolist(),
        "target_after": a1.atom_types.tolist(),
        "features_identical": mpr._swap_case(a0, a1),
        "pretext_inputs_identical": mpr._pretext_logits_equal(a0, a1),
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "randomized_real_train10k": {
            "seed": 7,
            "n_trials": int(args.random_cases),
            "atom_swap_cases": atom_cases,
            "atom_swap_passed": atom_pass,
            "bond_swap_cases": bond_cases,
            "bond_swap_passed": bond_pass,
        },
        "positive_control_visible_change": {"cases": visible_checked, "confined_to_dependency_cells": bool(visible_ok)},
        "minimal_witness": witness,
        "verdict": (
            "MASK_SWAP_INVARIANT"
            if (atom_pass == atom_cases and bond_pass == bond_cases and atom_cases > 0 and bond_cases > 0 and visible_ok)
            else "MASK_TARGET_LEAKAGE_BLOCKED"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(SWAP_AUDIT, payload)
    log(f"[audit-swap] atom {atom_pass}/{atom_cases} bond {bond_pass}/{bond_cases} -> {payload['verdict']}")
    if payload["verdict"] != "MASK_SWAP_INVARIANT":
        raise SystemExit("MASK_TARGET_LEAKAGE_BLOCKED")
    return 0


def cmd_rebuild_audit(args: argparse.Namespace) -> int:
    ds = load_train_dataset()
    rng = np.random.default_rng(11)
    determinism = diff_ok = own_ok = hidden_ok = bidi_ok = True
    checked = 0
    for trial in range(int(args.cases)):
        i = int(rng.integers(0, N_TRAIN))
        d = ds[i]
        x_r = d.x.numpy()
        ei_r = d.edge_index.numpy()
        ea_r = d.edge_attr.numpy()
        base = sro.build_mol_objects(i, x_r, ei_r, ea_r)
        s = mp.build_masked_sample(base, x_r, ei_r, ea_r, seed=3, visit_index=trial)
        s_again = mp.build_masked_sample(base, x_r, ei_r, ea_r, seed=3, visit_index=trial)
        for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
            if not np.array_equal(getattr(s.masked, name), getattr(s_again.masked, name)):
                determinism = False
        dep = mp.dependent_cell_mask(base, s.atom_ids, s.bond_ids)
        for name, before, after, allowed in (
            ("atom", base.atom_feat, s.masked.atom_feat, dep["atom"]),
            ("bond", base.bond_feat, s.masked.bond_feat, dep["bond"]),
            ("junc", base.junc_feat, s.masked.junc_feat, dep["junc"]),
            ("global", base.global_feat, s.masked.global_feat, dep["global"]),
        ):
            diff = np.asarray(before) != np.asarray(after)
            if not np.all(diff <= allowed):
                diff_ok = False
        for a, t in zip(s.atom_ids, s.atom_types):
            if int(s.masked.atom_feat[int(a), : sro.ATOM_CATEGORIES].argmax()) != mp.ATOM_MASK_TOKEN:
                own_ok = False
            if s.masked.atom_feat[int(a), int(t)] != 0.0:
                hidden_ok = False
        for b, t in zip(s.bond_ids, s.bond_types):
            if int(s.masked.bond_feat[int(b), : sro.BOND_CATEGORIES].argmax()) != mp.BOND_MASK_TOKEN:
                own_ok = False
            if s.masked.bond_feat[int(b), int(t)] != 0.0:
                hidden_ok = False
        _, ea_masked = mp.mask_raw_arrays(x_r, ei_r, ea_r, base.bond_index, s.atom_ids, s.bond_ids)
        mapping = mp.directed_bond_index(ei_r)
        masked_bonds = set(int(z) for z in s.bond_ids)
        for k in range(base.bond_index.shape[0]):
            u, v = int(base.bond_index[k][0]), int(base.bond_index[k][1])
            for col in mapping[(min(u, v), max(u, v))]:
                expected = mp.BOND_MASK_TOKEN if k in masked_bonds else int(ea_r.reshape(-1)[col])
                if int(ea_masked.reshape(-1)[col]) != expected:
                    bidi_ok = False
        checked += 1
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_cases": checked,
        "deterministic_reproduction": bool(determinism),
        "differences_confined_to_dependency_map": bool(diff_ok),
        "masked_own_slots_hold_token": bool(own_ok),
        "hidden_truth_absent_from_own_slot": bool(hidden_ok),
        "both_directed_records_masked": bool(bidi_ok),
        "verdict": "MASKED_REBUILD_LEAK_FREE"
        if (determinism and diff_ok and own_ok and hidden_ok and bidi_ok)
        else "MASKED_REBUILD_INCONSISTENT",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(REBUILD_AUDIT, payload)
    log(f"[rebuild-audit] cases={checked} -> {payload['verdict']}")
    if payload["verdict"] != "MASKED_REBUILD_LEAK_FREE":
        raise SystemExit(payload["verdict"])
    return 0


# ---------------------------------------------------------------------------
# correctness gates
# ---------------------------------------------------------------------------


def _encoder_group_hash(model: src.BJOC) -> str:
    st = mp.encoder_state(model)
    combined = (
        {f"atom:{k}": v for k, v in st["atom_encoder"].items()}
        | {f"bond:{k}": v for k, v in st["bond_encoder"].items()}
        | {f"junc:{k}": v for k, v in st["junction_encoder"].items()}
    )
    return state_group_sha256(combined)


def cmd_gates(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_train_dataset()
    train_objs = load_objects(OBJECTS_TRAIN)
    gates: dict[str, Any] = {}

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates[name] = {"pass": bool(passed), "detail": detail}
        log(f"[gate] {name}: {'PASS' if passed else 'FAIL'}")

    dep = read_json(DEPENDENCY_AUDIT)
    swap = read_json(SWAP_AUDIT)
    rebuild = read_json(REBUILD_AUDIT)
    gate("mask_token_slots_free", dep["verdict"] == "MASK_TOKEN_SLOTS_FREE", dep["mask_token_scan_train10k"])
    gate("mask_target_swap_invariance", swap["verdict"] == "MASK_SWAP_INVARIANT", swap["randomized_real_train10k"])
    gate(
        "hidden_truth_absent_and_rebuild",
        rebuild["verdict"] == "MASKED_REBUILD_LEAK_FREE",
        {k: rebuild[k] for k in ("hidden_truth_absent_from_own_slot", "both_directed_records_masked", "differences_confined_to_dependency_map")},
    )

    # pretraining path reads no y (AST scan of the function source)
    scan = mpr._scan_source_for_y_reads(Path(__file__), "pretrain_full10k")
    module_reads_y = mpr._module_reads_y_attribute(Path(_CODE_DIR / "bjoc_masked_pretraining_v1.py"))
    gate("pretraining_no_label_read", (not scan["found"]) and (not module_reads_y), {"driver_scan": scan, "module_reads_y_attribute": module_reads_y})

    # architecture: no GNN/attention/iteration; parameter count
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg, seed=0, organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    forbidden = src.forbidden_module_scan(model)
    audit = src.parameter_audit(cfg, model)
    gate("no_forbidden_modules", not forbidden, {"found": forbidden})
    gate("parameter_count_413505", audit["parameter_exact"] and audit["actual_parameters"] == 413505, audit)

    # mask RNG determinism + visit dependence
    a1, b1 = mp.sample_mask(0, 7, 0, 20, 20)
    a2, b2 = mp.sample_mask(0, 7, 0, 20, 20)
    a3, b3 = mp.sample_mask(0, 7, 1, 20, 20)
    gate(
        "mask_rng_deterministic_and_visit_dependent",
        np.array_equal(a1, a2) and np.array_equal(b1, b2) and (not np.array_equal(a1, a3)),
        {"repeat_equal": True, "visit0_vs_visit1_equal": bool(np.array_equal(a1, a3))},
    )

    # batch order determinism
    h1 = batch_order_hash(0, N_TRAIN, BATCH_SIZE, EPOCHS)
    h2 = batch_order_hash(0, N_TRAIN, BATCH_SIZE, EPOCHS)
    gate("batch_order_deterministic", h1 == h2, {"hash": h1})

    # save / load replay of the encoder state
    seed_everything(3)
    back = src.build_bjoc(cfg, seed=3, organization_mode=src.NULL, global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32), global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32))
    pre = mp.PretextModel(back)
    opt = torch.optim.Adam(pre.parameters(), lr=1e-3)
    sample = mp.build_masked_sample(
        train_objs[0],
        ds[0].x.numpy(), ds[0].edge_index.numpy(), ds[0].edge_attr.numpy(),
        seed=0, visit_index=0,
    )
    batch, ta, tb, _, _ = mp.batch_pretext_samples([sample])
    loss, _ = pre.loss(batch, ta, tb)
    opt.zero_grad(); loss.backward(); opt.step()
    state = mp.encoder_state(back)
    fresh = src.build_bjoc(cfg, seed=99, organization_mode=src.NULL, global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32), global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32))
    mp.load_encoder_state(fresh, state)
    replay_ok = _encoder_group_hash(fresh) == state_group_sha256(
        {f"{g}:{k}": v for g in ("atom", "bond", "junc") for k, v in state[{"atom": "atom_encoder", "bond": "bond_encoder", "junc": "junction_encoder"}[g]].items()}
    )
    gate("encoder_save_load_replay", replay_ok, {"exact": bool(replay_ok)})

    # atomic relabelling invariance of the BJOC forward
    base = train_objs[0]
    d0 = ds[0]
    perm = list(reversed(range(base.n_atoms)))
    inv = {old: new for new, old in enumerate(perm)}
    x_p = np.asarray(d0.x).copy().reshape(-1)
    x_new = np.zeros_like(x_p)
    for old in range(base.n_atoms):
        x_new[inv[old]] = x_p[old]
    edge_rows = []
    for col in range(np.asarray(d0.edge_index).shape[1]):
        u, v = int(d0.edge_index[0, col]), int(d0.edge_index[1, col])
        edge_rows.append((inv[u], inv[v]))
    ei_p = np.asarray(edge_rows, dtype=np.int64).T
    o_p = sro.build_mol_objects(0, x_new, ei_p, np.asarray(d0.edge_attr))
    with torch.no_grad():
        p_base = model(sro.batch_objects([base])).item()
        p_perm = model(sro.batch_objects([o_p])).item()
    gate("atom_relabelling_invariance", abs(p_base - p_perm) < 1e-6, {"base": p_base, "relabelled": p_perm})

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "all_pass": all(v["pass"] for v in gates.values()),
        "gates": gates,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(GATES_JSON, payload)
    log(f"[gates] all_pass={payload['all_pass']}")
    if not payload["all_pass"]:
        raise SystemExit("correctness gates failed")
    return 0


# ---------------------------------------------------------------------------
# freeze manifest
# ---------------------------------------------------------------------------


def _code_hashes() -> dict[str, str]:
    return {
        "driver": file_sha256(_CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py"),
        "module": file_sha256(_CODE_DIR / "bjoc_masked_pretraining_v1.py"),
        "objects": file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
        "composer": file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
        "features": file_sha256(_CODE_DIR / "cscl_features.py"),
        "config": file_sha256(CONFIG_PATH),
    }


def cmd_freeze(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(OBJECTS_MANIFEST)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hashes": _code_hashes(),
        "data": {
            "train_rows": [0, N_TRAIN - 1],
            "valid_rows": [0, N_VALID - 1],
            "train_rows_sha256": manifest["train_rows_sha256"],
            "valid_rows_sha256": manifest["valid_rows_sha256"],
            "objects_train10k_sha256": manifest["objects_train10k_sha256"],
            "objects_valid_sha256": manifest["objects_valid_sha256"],
            "prep_train10k_sha256": manifest["prep_train10k_sha256"],
            "no_internal_split": True,
            "pretraining_structure_source": "official train 10000 only (x/edge_index/edge_attr)",
            "normalisation_fit": "official train 10000 only",
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
            "loss": "CE(masked atoms) + CE(masked physical bonds), weights 1.0/1.0",
            "updates": mp.PRETRAIN_UPDATES,
            "batch_size": mp.PRETRAIN_BATCH,
            "optimizer": "Adam",
            "lr": mp.PRETRAIN_LR,
            "weight_decay": mp.PRETRAIN_WD,
            "grad_clip": mp.PRETRAIN_CLIP,
            "checkpoint_rule": "last-step encoder state only",
            "molecule_visits": int(mp.PRETRAIN_UPDATES * mp.PRETRAIN_BATCH),
            "mean_molecule_exposures_per_molecule": float(mp.PRETRAIN_UPDATES * mp.PRETRAIN_BATCH / N_TRAIN),
            "old_fit8k_mean_molecule_exposures_per_molecule": float(mp.PRETRAIN_UPDATES * mp.PRETRAIN_BATCH / 8000),
        },
        "downstream": {
            "recipe": {
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "grad_clip": GRAD_CLIP,
                "loss": "raw-y L1 only",
                "scheduler": "none",
                "dropout": 0.0,
                "epochs": EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "updates": UPDATES,
                "soup": f"parameter mean of the last {SOUP_LAST} epoch states",
                "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            },
            "arms": {
                "B": "scratch full-10k; reuse of the historical full-10k checkpoints allowed only if reuse_audit passes",
                "P": "three encoders from the frozen SSL state; composer/reader from the same scratch init as B",
                "Q": f"same as P but encoder lr {ENCODER_LR_Q}; composer/reader lr {LR}; conditional on the Q purchase gate",
            },
            "parameters_expected": 413505,
            "seeds": list(SEEDS),
        },
        "q_purchase_gate": {
            "junction_linear_cka_below": Q_CKA_THRESHOLD,
            "junction_effective_rank_drop_above": Q_RANK_DROP_THRESHOLD,
            "rank_replications": Q_RANK_REPLICATIONS,
            "rank_replication_size": Q_RANK_REPLICATION_SIZE,
            "criterion": "purchase iff (min-seed junction CKA < 0.75) or (a seed has median rank drop > 25% with >= 4/5 draws above 25%)",
            "if_not": "NO_ENCODER_PRESERVATION_PURCHASE; Q is never trained; no other candidate",
        },
        "evaluation": {
            "primary_metric": "official-valid raw-y MAE, seed-mean of the two per-seed MAEs",
            "freeze_order": "all prediction files hashed before the single valid label read",
            "bootstrap": {"n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "molecule"},
            "top_k_removed": list(TOP_K_REMOVED),
            "tail_threshold": TAIL_THRESHOLD,
            "extreme_row": f"valid:{EXTREME_VALID_ROW:04d}",
            "no_ensemble_headline": True,
            "seed_mean_definition": "arithmetic mean of the two independent single-model MAEs",
        },
        "budget": {
            "wall_clock_minutes_total": 405,
            "gpu_hours_max": 3.0,
            "gpus_max": 1,
            "no_test_access": True,
        },
        "bans": [
            "ensemble as headline; prediction calibration; label clipping; target transforms; tail-weighted loss",
            "extra internal split search; mask-ratio / pretraining-length / encoder-lr multiplier search",
            "new junction / three-body / four-body modules; GNN/Transformer/attention/iteration",
            "external labelled or unlabelled data; full ZINC-250K",
            "official test access in any form; historical official-test 0.092233 for any decision",
            "re-tuning after the valid read; deleting rows; replacing the worst row",
        ],
        "decision_vocabulary": [
            "FULL10K_PRETRAINING_GAIN_CONFIRMED_ON_DEV",
            "FULL10K_PRETRAINING_GAIN_WEAK_OR_UNSTABLE",
            "FULL10K_PRETRAINING_NO_GAIN",
            "ENCODER_LR_ADAPTATION_USEFUL",
            "ENCODER_LR_ADAPTATION_NOT_SUPPORTED",
            "BULK_GAIN_WITH_TAIL_COST",
            "NO_ENCODER_PRESERVATION_PURCHASE",
            "INCONCLUSIVE_OR_BLOCKED",
        ],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = cfg.get("frozen", {}) or {}
    mismatches = []
    for key, value in (
        ("objects_train10k_sha256", payload["data"]["objects_train10k_sha256"]),
        ("objects_valid_sha256", payload["data"]["objects_valid_sha256"]),
        ("prep_train10k_sha256", payload["data"]["prep_train10k_sha256"]),
    ):
        if key in frozen and frozen[key] != value:
            mismatches.append({"key": key, "config": frozen[key], "actual": value})
    for key, value in (
        ("pretrain_updates", payload["pretext"]["updates"]),
        ("updates", payload["downstream"]["recipe"]["updates"]),
        ("train_shuffle_offset", payload["downstream"]["recipe"]["train_shuffle_offset"]),
        ("encoder_lr_q", ENCODER_LR_Q),
        ("q_cka_threshold", Q_CKA_THRESHOLD),
        ("q_rank_drop_threshold", Q_RANK_DROP_THRESHOLD),
    ):
        if key in frozen and frozen[key] != value:
            mismatches.append({"key": key, "config": frozen[key], "actual": value})
    payload["config_frozen_block_mismatches"] = mismatches
    write_json(TRAINING_MANIFEST, payload)
    if mismatches:
        raise SystemExit(f"config frozen block mismatches: {mismatches}")
    log(f"[freeze] training manifest written; driver {payload['code_hashes']['driver'][:16]}")
    return 0


def verify_freeze() -> dict[str, Any]:
    if not TRAINING_MANIFEST.exists():
        raise SystemExit("training_manifest.json missing; run freeze first")
    manifest = read_json(TRAINING_MANIFEST)
    current = _code_hashes()
    mismatches = {k: {"frozen": manifest["code_hashes"].get(k), "current": v} for k, v in current.items() if manifest["code_hashes"].get(k) != v}
    if mismatches:
        raise SystemExit(f"code hash mismatch vs frozen manifest: {mismatches}")
    for key, path in (("objects_train10k_sha256", OBJECTS_TRAIN), ("objects_valid_sha256", OBJECTS_VALID), ("prep_train10k_sha256", PREP_TRAIN)):
        if not path.exists():
            raise SystemExit(f"missing frozen input {path}")
        actual = file_sha256(path)
        if actual != manifest["data"][key]:
            raise SystemExit(f"{key} mismatch: {actual} != {manifest['data'][key]}")
    if not read_json(GATES_JSON)["all_pass"]:
        raise SystemExit("correctness gates did not pass")
    return manifest


# ---------------------------------------------------------------------------
# Phase B: reuse audit of the historical full-10k BJOC-Null
# ---------------------------------------------------------------------------


def _compare_objects(a: Sequence[sro.MolObjects], b: Sequence[sro.MolObjects]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_a": len(a), "n_b": len(b), "max_abs_diff": {}, "index_equal_all": True, "mol_index_equal_all": True}
    fields = ("atom_feat", "bond_feat", "junc_feat", "global_feat")
    idx_fields = ("bond_index", "bond_type", "junc_index", "junc_bond")
    for oa, ob in zip(a, b):
        if int(oa.mol_index) != int(ob.mol_index):
            out["mol_index_equal_all"] = False
        for f in fields:
            x, y = np.asarray(getattr(oa, f)), np.asarray(getattr(ob, f))
            if x.shape != y.shape:
                out["max_abs_diff"][f] = float("inf")
                continue
            out["max_abs_diff"][f] = max(out["max_abs_diff"].get(f, 0.0), float(np.abs(x - y).max()) if x.size else 0.0)
        for f in idx_fields:
            if not np.array_equal(getattr(oa, f), getattr(ob, f)):
                out["index_equal_all"] = False
    out["exact"] = out["index_equal_all"] and out["mol_index_equal_all"] and all(v == 0.0 for v in out["max_abs_diff"].values())
    return out


def batch_order_hash(seed: int, n: int, batch_size: int, epochs: int) -> str:
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    h = hashlib.sha256()
    for _ in range(int(epochs)):
        order = torch.randperm(int(n), generator=gen)
        h.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
    return h.hexdigest()


def _make_downstream_model(seed: int, norm: Mapping[str, Any], device: torch.device | str) -> src.BJOC:
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


def _shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(int(n), generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def _mv(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_bjoc(model: src.BJOC, objs: Sequence[sro.MolObjects], device: torch.device | str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = _mv(sro.batch_objects(list(objs[i : i + 256])), torch.device(device))
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


def cmd_reuse_audit(args: argparse.Namespace) -> int:
    norm = read_json(PREP_TRAIN)
    train_objs = load_objects(OBJECTS_TRAIN)
    valid_objs = load_objects(OBJECTS_VALID)

    hist_objects = torch.load(HISTORICAL_BJOC_DIR / "bjoc_objects_v1.pt", map_location="cpu", weights_only=False)
    obj_cmp = _compare_objects(train_objs, hist_objects)

    hist_norm = read_json(HISTORICAL_BJOC_DIR / "bjoc_global_norm_full10k.json")
    norm_cmp = {
        "mean_max_abs_diff": float(np.abs(np.asarray(norm["global_mean"]) - np.asarray(hist_norm["global_mean"])).max()),
        "scale_max_abs_diff": float(np.abs(np.asarray(norm["global_scale"]) - np.asarray(hist_norm["global_scale"])).max()),
        "n_fit_new": norm["n_fit"],
        "n_fit_historical": hist_norm["n_fit"],
    }

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        hist_json = read_json(HISTORICAL_BJOC_DIR / f"train_bjoc_null_full10k_s{seed}.json")
        model = _make_downstream_model(seed, norm, "cpu")
        init_sha = src.state_dict_hash(model)
        init_encoder = _encoder_group_hash(model)
        order_hash = batch_order_hash(seed, N_TRAIN, BATCH_SIZE, EPOCHS)
        ckpt_path = HISTORICAL_BJOC_DIR / f"bjoc_null_full10k_s{seed}_soup_state.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        pred = predict_bjoc(model, valid_objs, "cpu")
        ref_path = HISTORICAL_TERMINAL_DIR / f"valid_N{seed}_preds.npz"
        replay = None
        if ref_path.exists():
            ref = np.asarray(np.load(ref_path, allow_pickle=True)["pred"], dtype=np.float64)
            replay = {"max_abs_pred_diff": float(np.abs(pred - ref).max()), "mean_abs_pred_diff": float(np.abs(pred - ref).mean())}
        per_seed[str(seed)] = {
            "recorded_init_state_sha256": hist_json["init_state_sha256"],
            "recomputed_init_state_sha256": init_sha,
            "init_state_match": init_sha == hist_json["init_state_sha256"],
            "recomputed_init_encoder_sha256": init_encoder,
            "recorded_parameters": hist_json["parameters"],
            "recorded_epochs": hist_json["epochs"],
            "recorded_soup_members": hist_json["soup_members"],
            "recorded_train_shuffle_seed": hist_json["hyper"]["train_shuffle_seed"],
            "recorded_recipe": hist_json["hyper"],
            "checkpoint_sha256": file_sha256(ckpt_path),
            "batch_order_hash_recomputed": order_hash,
            "valid_replay": replay,
            "checkpoint_exists": ckpt_path.exists(),
        }

    conditions = {
        "objects_bit_identical": bool(obj_cmp["exact"]),
        "norm_bit_identical": norm_cmp["mean_max_abs_diff"] == 0.0 and norm_cmp["scale_max_abs_diff"] == 0.0,
        "init_state_hash_match_both_seeds": all(per_seed[str(s)]["init_state_match"] for s in SEEDS),
        "recipe_match": all(
            per_seed[str(s)]["recorded_recipe"]["lr"] == LR
            and per_seed[str(s)]["recorded_recipe"]["weight_decay"] == WEIGHT_DECAY
            and per_seed[str(s)]["recorded_recipe"]["grad_clip"] == GRAD_CLIP
            and per_seed[str(s)]["recorded_recipe"]["batch_size"] == BATCH_SIZE
            and per_seed[str(s)]["recorded_epochs"] == EPOCHS
            and per_seed[str(s)]["recorded_soup_members"] == list(range(EPOCHS - SOUP_LAST + 1, EPOCHS + 1))
            and per_seed[str(s)]["recorded_train_shuffle_seed"] == s + TRAIN_SHUFFLE_OFFSET
            and per_seed[str(s)]["recorded_parameters"] == 413505
            for s in SEEDS
        ),
        "valid_replay_identical": all(
            per_seed[str(s)]["valid_replay"] is not None and per_seed[str(s)]["valid_replay"]["max_abs_pred_diff"] < 1e-4 for s in SEEDS
        ),
    }
    decision = "REUSE_HISTORICAL_FULL10K_B" if all(conditions.values()) else "RETRAIN_B"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "objects_comparison": obj_cmp,
        "norm_comparison": norm_cmp,
        "per_seed": per_seed,
        "conditions": conditions,
        "decision": decision,
        "note": (
            "reuse is accepted only when the historical checkpoints are the same recipe, the same "
            "init state, the same train-only normalisation and bit-identical on the frozen object "
            "cache; the valid replay uses structure-only forwards and reads no labels"
        ),
        "official_valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(REUSE_AUDIT, payload)
    log(f"[reuse-audit] decision {decision} conditions {conditions}")
    return 0


def cmd_stage_b(args: argparse.Namespace) -> int:
    verify_freeze()
    audit = read_json(REUSE_AUDIT)
    norm = read_json(PREP_TRAIN)
    device = torch.device(args.device)
    if audit["decision"] != "REUSE_HISTORICAL_FULL10K_B":
        log("[stage-b] reuse audit decision RETRAIN_B; B is trained on the GPU chains")
        return 0
    for seed in SEEDS:
        if (OUT / f"train_B_seed{seed}.json").exists():
            log(f"[stage-b] seed {seed} already staged")
            continue
        src_path = HISTORICAL_BJOC_DIR / f"bjoc_null_full10k_s{seed}_soup_state.pt"
        state = torch.load(src_path, map_location="cpu", weights_only=False)
        torch.save(state, OUT / f"soup_B_seed{seed}.pt")
        model = _make_downstream_model(seed, norm, device)
        hist = read_json(HISTORICAL_BJOC_DIR / f"train_bjoc_null_full10k_s{seed}.json")
        audit_seed = audit["per_seed"][str(seed)]
        payload = {
            "protocol": PROTOCOL_ID,
            "arm": "B",
            "seed": int(seed),
            "git_commit": git_commit(),
            "reused_historical_checkpoint": True,
            "source_path": str(src_path.relative_to(REPO_ROOT)),
            "source_sha256": file_sha256(src_path),
            "soup_sha256": file_sha256(OUT / f"soup_B_seed{seed}.pt"),
            "epochs": EPOCHS,
            "updates": UPDATES,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "soup_rule": f"parameter mean of the last {SOUP_LAST} epoch states",
            "soup_members": list(range(EPOCHS - SOUP_LAST + 1, EPOCHS + 1)),
            "hyper": {
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "batch_size": BATCH_SIZE,
                "loss": "raw-y L1 only",
                "scheduler": "none",
                "dropout": 0.0,
                "train_shuffle_seed": seed + TRAIN_SHUFFLE_OFFSET,
            },
            "parameters": 413505,
            "init_full_sha256": src.state_dict_hash(model),
            "init_group_state_sha256": mp.group_state_hash(model),
            "init_encoder_sha256": _encoder_group_hash(model),
            "batch_order_hash": audit_seed["batch_order_hash_recomputed"],
            "soup_fit_mae": hist["soup_fit_mae"],
            "soup_monitor_mae": None,
            "wall_seconds_historical": hist["wall_seconds"],
            "reuse_audit_conditions": audit["conditions"],
            "reuse_valid_replay": audit_seed["valid_replay"],
            "labels_used": "official train 10000 raw y (historical full-10k trajectory)",
            "device": "historical res A100 (reused; no new training)",
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        write_json(OUT / f"train_B_seed{seed}.json", payload)
        log(f"[stage-b] seed {seed} reused (fit MAE {hist['soup_fit_mae']:.5f})")
    return 0


# ---------------------------------------------------------------------------
# SSL pretraining on the full official train 10k
# ---------------------------------------------------------------------------


def pretrain_full10k(seed: int, device: torch.device, *, log_fn: Callable[[str], None] = log) -> dict[str, Any]:
    """Masked atom/bond pretraining on all 10,000 official-train structures.

    Reads only ``x`` / ``edge_index`` / ``edge_attr``; no label, no monitor and
    no valid/test structure is used.
    """
    verify_freeze()
    ds = load_train_dataset()
    bases = load_objects(OBJECTS_TRAIN)
    assert [o.mol_index for o in bases] == list(range(N_TRAIN))

    cfg = src.BJOCConfig()
    seed_everything(seed)
    backbone = src.build_bjoc(
        cfg,
        seed=seed,
        organization_mode=src.NULL,
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
    started = time.perf_counter()
    while steps < mp.PRETRAIN_UPDATES:
        pretext.train()
        ce_a_sum = ce_b_sum = acc_a_sum = acc_b_sum = 0.0
        n_a = n_b = n_steps = 0
        t_ep = time.perf_counter()
        stop = False
        for chunk in _shuffle_batches(N_TRAIN, BATCH_SIZE, gen):
            pre = []
            for j in chunk:
                o = bases[j]
                d = ds[int(o.mol_index)]
                pre.append(mp.build_masked_sample(o, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy(), seed=seed, visit_index=visit))
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
    torch.save(enc_state, OUT / f"pretrain_encoder_s{seed}.pt")
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": "SSL",
        "seed": int(seed),
        "git_commit": git_commit(),
        "updates": int(steps),
        "visits": visit,
        "n_train_rows": N_TRAIN,
        "batch_size": mp.PRETRAIN_BATCH,
        "molecule_visits": int(steps * mp.PRETRAIN_BATCH),
        "mean_molecule_exposures_per_molecule": float(steps * mp.PRETRAIN_BATCH / N_TRAIN),
        "hyper": {
            "optimizer": "Adam",
            "lr": mp.PRETRAIN_LR,
            "weight_decay": mp.PRETRAIN_WD,
            "grad_clip": mp.PRETRAIN_CLIP,
            "loss": "CE(masked atoms) + CE(masked physical bonds), weights 1.0/1.0",
            "checkpoint_rule": "last step only",
        },
        "mask": {"atom_ratio": mp.ATOM_MASK_RATIO, "bond_ratio": mp.BOND_MASK_RATIO, "atom_token": mp.ATOM_MASK_TOKEN, "bond_token": mp.BOND_MASK_TOKEN},
        "wall_seconds": wall,
        "encoder_init_sha256": init_encoder_sha,
        "encoder_final_sha256": _encoder_group_hash(backbone),
        "encoder_parameters_changed": bool(init_encoder_sha != _encoder_group_hash(backbone)),
        "curve": curve,
        "final_ce_atom": curve[-1]["ce_atom"],
        "final_ce_bond": curve[-1]["ce_bond"],
        "final_acc_atom": curve[-1]["acc_atom"],
        "final_acc_bond": curve[-1]["acc_bond"],
        "params": int(sum(p.numel() for p in pretext.parameters())),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / f"pretraining_seed{seed}.json", payload)
    log_fn(f"[ssl s{seed}] acc atom {payload['final_acc_atom']:.4f} bond {payload['final_acc_bond']:.4f} | wall {wall:.0f}s")
    return payload


# ---------------------------------------------------------------------------
# downstream training (B / P / Q)
# ---------------------------------------------------------------------------


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    assert 0 < len(states) <= SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    keys = states[0].keys()
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}
    return members, soup


def _load_y_rows(rows: Sequence[int]) -> np.ndarray:
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    values = np.asarray([float(enc[int(i)].y.reshape(-1)[0]) for i in rows], dtype=np.float64)
    del enc
    return values


def _train_downstream(arm: str, seed: int, device: torch.device, *, encoder_lr: float | None = None, log_fn: Callable[[str], None] = log) -> dict[str, Any]:
    verify_freeze()
    assert arm in ("B", "P", "Q"), arm
    norm = read_json(PREP_TRAIN)
    train_objs = load_objects(OBJECTS_TRAIN)
    y_fit = _load_y_rows(list(range(N_TRAIN)))
    model = _make_downstream_model(seed, norm, device)
    init_full_sha = src.state_dict_hash(model)
    init_group_sha = mp.group_state_hash(model)
    init_encoder_sha = _encoder_group_hash(model)
    pretrain_encoder_sha = None
    encoder_used_sha = init_encoder_sha
    if arm in ("P", "Q"):
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

    if encoder_lr is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        groups = {"all": sum(p.numel() for p in model.parameters())}
    else:
        enc_params = list(model.atom_encoder.parameters()) + list(model.bond_encoder.parameters()) + list(model.junction_encoder.parameters())
        head_params = list(model.group_composer.parameters()) + list(model.reader.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": enc_params, "lr": float(encoder_lr)},
                {"params": head_params, "lr": LR},
            ],
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        groups = {
            "encoder_lr": float(encoder_lr),
            "composer_reader_lr": LR,
            "encoder_parameters": int(sum(p.numel() for p in enc_params)),
            "composer_reader_parameters": int(sum(p.numel() for p in head_params)),
        }
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in _shuffle_batches(N_TRAIN, BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            batch = _mv(sro.batch_objects([train_objs[i] for i in indices]), device)
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
        curve.append({"epoch": epoch, "updates": epoch * steps, "train_mae": total / max(n_mol, 1), "seconds": time.perf_counter() - t_ep})
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} {curve[-1]['seconds']:.1f}s")
        if not np.isfinite(curve[-1]["train_mae"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch} arm {arm} seed {seed}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_bjoc(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(soup_state, OUT / f"soup_{arm}_seed{seed}.pt")
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "git_commit": git_commit(),
        "epochs": EPOCHS,
        "updates": int(EPOCHS * STEPS_PER_EPOCH),
        "steps_per_epoch": STEPS_PER_EPOCH,
        "soup_rule": f"parameter mean of the last {SOUP_LAST} epoch states",
        "soup_members": members,
        "hyper": {
            "optimizer": "Adam",
            "lr": LR,
            "encoder_lr": encoder_lr,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "loss": "raw-y L1 only",
            "scheduler": "none",
            "dropout": 0.0,
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
            "parameter_groups": groups,
        },
        "parameters": audit["actual_parameters"],
        "init_full_sha256": init_full_sha,
        "init_group_state_sha256": init_group_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "final_encoder_sha256": _encoder_group_hash(model),
        "pretrain_encoder_file_sha256": pretrain_encoder_sha,
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
    write_json(OUT / f"train_{arm}_seed{seed}.json", payload)
    log_fn(f"[{arm} s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {wall:.0f}s")
    return payload


# ---------------------------------------------------------------------------
# prediction + freeze
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze()
    device = torch.device(args.device)
    norm = read_json(PREP_TRAIN)
    valid_objs = load_objects(OBJECTS_VALID)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for key in args.models:
        arm, seed = key[0], int(key[1:])
        soup_path = OUT / f"soup_{arm}_seed{seed}.pt"
        if not soup_path.exists():
            raise SystemExit(f"missing checkpoint {soup_path}")
        model = _make_downstream_model(seed, norm, device)
        model.load_state_dict(torch.load(soup_path, map_location=device, weights_only=False))
        pred = predict_bjoc(model, valid_objs, device)
        if not np.isfinite(pred).all():
            raise RuntimeError("non-finite predictions")
        path = PRED_DIR / f"valid_{key}.npz"
        np.savez_compressed(path, row=np.arange(N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
        entries[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path), "checkpoint_sha256": file_sha256(soup_path)}
        log(f"[predict] {key} written (checkpoint {entries[key]['checkpoint_sha256'][:12]})")
    log_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(device),
        "predictions": entries,
        "valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(PRED_ACCESS, log_payload)
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    verify_freeze()
    if VALID_LOCK.exists():
        raise SystemExit("valid labels were already read; refusing to re-freeze predictions")
    required = [f"{a}{s}" for a in ("B", "P") for s in SEEDS]
    files: dict[str, Any] = {}
    for key in required + (["Q0", "Q1"] if (OUT / "q_purchase_gate.json").exists() and read_json(Q_GATE).get("purchase_q") else []):
        path = PRED_DIR / f"valid_{key}.npz"
        if not path.exists():
            raise SystemExit(f"missing prediction file {path}")
        with np.load(path) as z:
            if not np.isfinite(z["pred"]).all():
                raise SystemExit(f"non-finite prediction in {path}")
        files[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
    ckpts = {p.stem: file_sha256(p) for p in sorted(OUT.glob("soup_*.pt"))}
    q_gate = read_json(Q_GATE) if Q_GATE.exists() else {"purchase_q": False, "status": "NOT_PURCHASED"}
    combined = hashlib.sha256()
    for key in sorted(files):
        combined.update(key.encode()); combined.update(files[key]["sha256"].encode())
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_prediction_files": len(files),
        "prediction_files": files,
        "checkpoints": ckpts,
        "combined_prediction_sha256": combined.hexdigest(),
        "q_purchase": bool(q_gate.get("purchase_q")),
        "q_status": q_gate.get("status"),
        "frozen_before_valid_read": True,
        "valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(PRED_FREEZE, payload)
    log(f"[freeze-predictions] {len(files)} files combined {payload['combined_prediction_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# Phase D: representation transfer diagnostics + Q purchase gate
# ---------------------------------------------------------------------------


def _effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = s**2 / max((s**2).sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    return float(xy / max(xx * yy, 1e-300))


def _encode_states(model: src.BJOC, batch: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        a = model.encode_atoms(batch)
        b = model.encode_bonds(a, batch)
        w = model.encode_junctions(a, b, batch)
    return {"atom": a.numpy(), "bond": b.numpy(), "junction": w.numpy()}


def cmd_diagnostics(args: argparse.Namespace) -> int:
    norm = read_json(PREP_TRAIN)
    train_objs = load_objects(OBJECTS_TRAIN)
    rng = np.random.default_rng(2026)
    sample_rows = np.sort(rng.choice(N_TRAIN, 2000, replace=False))
    sample_objs = [train_objs[int(i)] for i in sample_rows]
    batch = sro.batch_objects(sample_objs)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "sample": {"n": len(sample_rows), "rows_sha256": rows_sha256(sample_rows.tolist())},
        "per_seed": {},
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    junction_cka: dict[str, float] = {}
    rank_replication: dict[str, Any] = {}
    for seed in SEEDS:
        model_init = _make_downstream_model(seed, norm, "cpu")
        reps_init = _encode_states(model_init, batch)
        model_ssl = _make_downstream_model(seed, norm, "cpu")
        mp.load_encoder_state(model_ssl, torch.load(OUT / f"pretrain_encoder_s{seed}.pt", map_location="cpu", weights_only=False))
        reps_ssl = _encode_states(model_ssl, batch)
        model_p = _make_downstream_model(seed, norm, "cpu")
        model_p.load_state_dict(torch.load(OUT / f"soup_P_seed{seed}.pt", map_location="cpu", weights_only=False))
        reps_p = _encode_states(model_p, batch)

        def move(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, float]:
            out = {}
            for g in ("atom_encoder", "bond_encoder", "junction_encoder"):
                num = den = 0.0
                for k, v in a.items():
                    if k.startswith(g + "."):
                        d = (b[k].float() - v.float())
                        num += float(d.norm() ** 2); den += float(v.float().norm() ** 2)
                out[g] = float(math.sqrt(num / max(den, 1e-12)))
            return out

        init_state = {k: v.cpu() for k, v in model_init.state_dict().items()}
        ssl_state = {k: v.cpu() for k, v in model_ssl.state_dict().items()}
        p_state = {k: v.cpu() for k, v in model_p.state_dict().items()}
        seed_entry: dict[str, Any] = {
            "parameter_relative_move_init_to_ssl": move(init_state, ssl_state),
            "parameter_relative_move_ssl_to_p": move(ssl_state, p_state),
            "parameter_relative_move_init_to_p": move(init_state, p_state),
            "representation": {},
            "linear_cka_ssl_vs_p": {},
        }
        for name in ("atom", "bond", "junction"):
            seed_entry["representation"][name] = {
                "init_scale_mean_abs": float(np.abs(reps_init[name]).mean()),
                "ssl_scale_mean_abs": float(np.abs(reps_ssl[name]).mean()),
                "p_scale_mean_abs": float(np.abs(reps_p[name]).mean()),
                "init_effective_rank": _effective_rank(reps_init[name]),
                "ssl_effective_rank": _effective_rank(reps_ssl[name]),
                "p_effective_rank": _effective_rank(reps_p[name]),
            }
            seed_entry["linear_cka_ssl_vs_p"][name] = _linear_cka(reps_ssl[name], reps_p[name])
        junction_cka[str(seed)] = seed_entry["linear_cka_ssl_vs_p"]["junction"]

        # repeated-subsample rank drop
        drops = []
        for rep in range(Q_RANK_REPLICATIONS):
            sub_rng = np.random.default_rng(rep)
            sub = np.sort(sub_rng.choice(len(sample_rows), Q_RANK_REPLICATION_SIZE, replace=False))
            sub_objs = [sample_objs[int(i)] for i in sub]
            sub_batch = sro.batch_objects(sub_objs)
            w_ssl = _encode_states(model_ssl, sub_batch)["junction"]
            w_p = _encode_states(model_p, sub_batch)["junction"]
            r_ssl = _effective_rank(w_ssl)
            r_p = _effective_rank(w_p)
            drops.append({"replication": rep, "rank_ssl": r_ssl, "rank_p": r_p, "relative_drop": float((r_ssl - r_p) / max(r_ssl, 1e-12))})
        arr = np.asarray([d["relative_drop"] for d in drops])
        rank_replication[str(seed)] = {
            "drops": drops,
            "median_relative_drop": float(np.median(arr)),
            "n_draws_above_threshold": int((arr > Q_RANK_DROP_THRESHOLD).sum()),
            "n_draws": len(drops),
        }
        seed_entry["rank_replication"] = rank_replication[str(seed)]
        payload["per_seed"][str(seed)] = seed_entry
    payload["junction_linear_cka_ssl_vs_p"] = junction_cka
    payload["rank_replication"] = rank_replication
    write_json(REPRESENTATION, payload)

    cka_fires = any(v < Q_CKA_THRESHOLD for v in junction_cka.values())
    rank_fires = any(
        rank_replication[str(s)]["median_relative_drop"] > Q_RANK_DROP_THRESHOLD
        and rank_replication[str(s)]["n_draws_above_threshold"] >= max(1, Q_RANK_REPLICATIONS - 1)
        for s in SEEDS
    )
    purchase = bool(cka_fires or rank_fires)
    gate = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "junction_linear_cka": junction_cka,
        "cka_threshold": Q_CKA_THRESHOLD,
        "cka_fires": bool(cka_fires),
        "rank_replication": rank_replication,
        "rank_drop_threshold": Q_RANK_DROP_THRESHOLD,
        "rank_fires": bool(rank_fires),
        "purchase_q": purchase,
        "status": "PURCHASE_Q" if purchase else "NO_ENCODER_PRESERVATION_PURCHASE",
        "note": (
            "purchase is a resource gate for a pre-registered intervention only; it is not evidence of "
            "harmful forgetting and a negative CKA / rank drop alone is not a performance claim"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(Q_GATE, gate)
    log(f"[diagnostics] junction CKA {junction_cka} rank_fires={rank_fires} -> {gate['status']}")
    return 0


# ---------------------------------------------------------------------------
# scoring (the ONE official-valid label read)
# ---------------------------------------------------------------------------


def _load_pred(key: str) -> np.ndarray:
    with np.load(PRED_DIR / f"valid_{key}.npz") as z:
        return z["pred"].astype(np.float64)


def _contrast(keys_new: Sequence[str], keys_ref: Sequence[str], y: np.ndarray, preds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    e_new = np.mean([np.abs(preds[k] - y) for k in keys_new], axis=0)
    e_ref = np.mean([np.abs(preds[k] - y) for k in keys_ref], axis=0)
    delta = e_new - e_ref
    out = {
        "new": list(keys_new),
        "ref": list(keys_ref),
        "delta_seed_mean": float(delta.mean()),
        "per_seed_delta": {f"seed{k[1:]}": float(np.abs(preds[k] - y).mean() - np.abs(preds[r] - y).mean()) for k, r in zip(keys_new, keys_ref)},
        "seed_mean_mae_new": float(np.mean([np.abs(preds[k] - y).mean() for k in keys_new])),
        "seed_mean_mae_ref": float(np.mean([np.abs(preds[k] - y).mean() for k in keys_ref])),
        "seed_mean_rmse_new": float(np.mean([np.sqrt(np.mean((preds[k] - y) ** 2)) for k in keys_new])),
        "seed_mean_rmse_ref": float(np.mean([np.sqrt(np.mean((preds[k] - y) ** 2)) for k in keys_ref])),
        "bootstrap": paired_bootstrap(delta),
        "ordinary_y_gt_-3": {"n": int((y > TAIL_THRESHOLD).sum()), "delta": float(delta[y > TAIL_THRESHOLD].mean()), "delta_seed_mean": float(delta[y > TAIL_THRESHOLD].mean())},
        "tail_y_le_-3": {"n": int((y <= TAIL_THRESHOLD).sum()), "delta": float(delta[y <= TAIL_THRESHOLD].mean()) if (y <= TAIL_THRESHOLD).any() else None},
        "better_fraction": float((delta < 0).mean()),
    }
    for k in TOP_K_REMOVED:
        out.update(top_k_removed(delta, e_new, e_ref, k))
    extreme = EXTREME_VALID_ROW
    out["extreme_row"] = {
        "row": f"valid:{extreme:04d}",
        "y": float(y[extreme]),
        "delta_row": float(delta[extreme]),
        "delta_without_row": float(np.delete(delta, extreme).mean()),
        "share_of_total_delta": float(delta[extreme] / 1000.0 / delta.mean()) if delta.mean() != 0 else None,
    }
    return out


def cmd_score(args: argparse.Namespace) -> int:
    if VALID_LOCK.exists():
        raise SystemExit("official-valid labels already read once; refusing a second read")
    freeze = read_json(PRED_FREEZE)
    for key, entry in freeze["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"frozen prediction hash mismatch: {key}")
    if not freeze["frozen_before_valid_read"]:
        raise SystemExit("prediction freeze is not marked pre-label")

    val_ds = load_valid_dataset()
    y = np.asarray([float(val_ds[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
    np.save(VALID_Y_COPY, y)
    enc = torch.load(ENCODED_VALID, map_location="cpu", weights_only=False)
    y_enc = np.asarray([float(enc[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
    del enc
    y_crosscheck = float(np.abs(y - y_enc).max())
    if y_crosscheck != 0.0:
        raise SystemExit("raw official-valid y and the encoded-valid cache disagree")
    VALID_LOCK.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_ID,
                "git_commit": git_commit(),
                "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_valid": N_VALID,
                "valid_rows_sha256": rows_sha256(range(N_VALID)),
                "y_sha256": array_sha256(y),
                "y_raw_vs_encoded_max_abs": y_crosscheck,
                "prediction_freeze_sha256": file_sha256(PRED_FREEZE),
                "extreme_row": f"valid:{EXTREME_VALID_ROW:04d}",
                "extreme_row_y": float(y[EXTREME_VALID_ROW]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model_keys = [f"{a}{s}" for a in ("B", "P", "Q") for s in SEEDS if (PRED_DIR / f"valid_{a}{s}.npz").exists()]
    preds = {k: _load_pred(k) for k in model_keys}
    models = {k: row_metrics(preds[k], y) for k in model_keys}
    seed_mean = {}
    for arm in ("B", "P", "Q"):
        keys = [f"{arm}{s}" for s in SEEDS]
        if all(k in models for k in keys):
            seed_mean[arm] = {
                "mae": float(np.mean([models[k]["mae"] for k in keys])),
                "rmse": float(np.mean([models[k]["rmse"] for k in keys])),
                "median_abs": float(np.mean([models[k]["median_abs"] for k in keys])),
                "p90_abs": float(np.mean([models[k]["p90_abs"] for k in keys])),
                "p95_abs": float(np.mean([models[k]["p95_abs"] for k in keys])),
                "p99_abs": float(np.mean([models[k]["p99_abs"] for k in keys])),
                "per_seed_mae": [models[k]["mae"] for k in keys],
            }
    paired = {}
    if all(f"B{s}" in preds and f"P{s}" in preds for s in SEEDS):
        paired["P_minus_B"] = _contrast([f"P{s}" for s in SEEDS], [f"B{s}" for s in SEEDS], y, preds)
    if all(f"P{s}" in preds and f"Q{s}" in preds for s in SEEDS):
        paired["Q_minus_P"] = _contrast([f"Q{s}" for s in SEEDS], [f"P{s}" for s in SEEDS], y, preds)
        paired["Q_minus_B"] = _contrast([f"Q{s}" for s in SEEDS], [f"B{s}" for s in SEEDS], y, preds)
    fit_gaps = {}
    for k in model_keys:
        train = read_json(OUT / f"train_{k[0]}_seed{k[1:]}.json")
        fit_gaps[k] = {"fit_mae": train["soup_fit_mae"], "valid_mae": models[k]["mae"], "fit_minus_valid": train["soup_fit_mae"] - models[k]["mae"]}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_valid": N_VALID,
        "models": models,
        "seed_mean": seed_mean,
        "paired": paired,
        "fit_valid_gap": fit_gaps,
        "y_raw_vs_encoded_max_abs": y_crosscheck,
        "single_read": True,
        "official_test_loaded": False,
    }
    write_json(VALID_METRICS, payload)
    write_json(PAIRED_BOOTSTRAP, {"protocol": PROTOCOL_ID, "paired": paired, "n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED})
    for k, m in models.items():
        log(f"[score] {k}: valid MAE {m['mae']:.6f} RMSE {m['rmse']:.6f}")
    for arm, m in seed_mean.items():
        log(f"[score] {arm}: seed-mean MAE {m['mae']:.6f}")
    return 0


# ---------------------------------------------------------------------------
# Phase G / H: error structure and common errors
# ---------------------------------------------------------------------------


def cmd_error_analysis(args: argparse.Namespace) -> int:
    if not VALID_LOCK.exists():
        raise SystemExit("run score first")
    metrics = read_json(VALID_METRICS)
    y = np.load(VALID_Y_COPY)
    assert array_sha256(y) == read_json(VALID_LOCK)["y_sha256"]
    train_objs = load_objects(OBJECTS_TRAIN)
    valid_objs = load_objects(OBJECTS_VALID)
    preds = {k: _load_pred(k) for k in metrics["models"]}

    # --- G: tail / ordinary / extreme-row decomposition --------------------
    g: dict[str, Any] = {}
    for arm in ("B", "P", "Q"):
        keys = [f"{arm}{s}" for s in SEEDS if f"{arm}{s}" in preds]
        if not keys:
            continue
        e = np.mean([np.abs(preds[k] - y) for k in keys], axis=0)
        tail = y <= TAIL_THRESHOLD
        g[arm] = {
            "mae": float(e.mean()),
            "mae_ordinary": float(e[~tail].mean()),
            "mae_tail": float(e[tail].mean()) if tail.any() else None,
            "rmse": float(np.sqrt(np.mean((np.mean([preds[k] for k in keys], axis=0) - y) ** 2))),
            "mean_signed_error_tail": float(np.mean(np.mean([preds[k] for k in keys], axis=0)[tail] - y[tail])) if tail.any() else None,
            "mean_signed_error_ordinary": float(np.mean(np.mean([preds[k] for k in keys], axis=0)[~tail] - y[~tail])),
            "extreme_row_pred": [float(preds[k][EXTREME_VALID_ROW]) for k in keys],
            "extreme_row_abs_error": [float(abs(preds[k][EXTREME_VALID_ROW] - y[EXTREME_VALID_ROW])) for k in keys],
        }
    # --- H1: common errors --------------------------------------------------
    top10 = {}
    for k in preds:
        err = np.abs(preds[k] - y)
        top10[k] = set(np.argsort(-err)[: max(1, N_VALID // 10)].tolist())
    stable = {}
    for arm_keys in (["B0", "B1"], ["P0", "P1"], ["B0", "B1", "P0", "P1"] + (["Q0", "Q1"] if "Q0" in preds else [])):
        tag = "_".join(arm_keys)
        inter = set.intersection(*[top10[k] for k in arm_keys])
        stable[tag] = {"n": len(inter), "rows": sorted(int(x) for x in inter)}
    b_high = np.mean([np.abs(preds[f"B{s}"] - y) for s in SEEDS], axis=0) > np.quantile(np.mean([np.abs(preds[f"B{s}"] - y) for s in SEEDS], axis=0), 0.9)
    p_high = np.mean([np.abs(preds[f"P{s}"] - y) for s in SEEDS], axis=0) > np.quantile(np.mean([np.abs(preds[f"P{s}"] - y) for s in SEEDS], axis=0), 0.9)
    fixed = sorted(set(np.where(b_high)[0]) - set(np.where(p_high)[0]))
    persistent = sorted(set(np.where(b_high)[0]) & set(np.where(p_high)[0]))
    h1 = {
        "top10pct_overlap": {k: sorted(int(x) for x in v) for k, v in top10.items()},
        "stable_sets": stable,
        "B_high_error_rows": int(b_high.sum()),
        "P_high_error_rows": int(p_high.sum()),
        "fixed_by_pretraining": len(fixed),
        "persistent_high_error_rows": persistent,
        "persistent_n": len(persistent),
        "persistent_fraction_of_B_high": float(len(persistent) / max(int(b_high.sum()), 1)),
    }
    # --- subgroups by structure (label-free features) -----------------------
    subgroups: dict[str, Any] = {}
    junc = np.asarray([o.junc_index.shape[0] for o in valid_objs])
    n_atoms = np.asarray([o.n_atoms for o in valid_objs])
    in_ring = np.asarray([bool(o.atom_feat[:, sro.atom_blocks()["in_ring"].start].any()) for o in valid_objs])
    diameter = np.asarray([float(o.global_feat[sro.GLOBAL_FEAT_NAMES.index("diameter")]) for o in valid_objs])
    fields = {
        "junction_rich": junc > np.median(junc),
        "atom_rich": n_atoms > np.median(n_atoms),
        "ring_containing": in_ring,
        "large_diameter": diameter > np.median(diameter),
        "tail": y <= TAIL_THRESHOLD,
    }
    for name, mask in fields.items():
        entry: dict[str, Any] = {"n": int(mask.sum())}
        for arm, ref in (("P", "B"), ("Q", "P")):
            if all(f"{a}{s}" in preds for a in (arm, ref) for s in SEEDS):
                e_new = np.mean([np.abs(preds[f"{arm}{s}"] - y) for s in SEEDS], axis=0)
                e_ref = np.mean([np.abs(preds[f"{ref}{s}"] - y) for s in SEEDS], axis=0)
                entry[f"{arm}_minus_{ref}"] = float((e_new - e_ref)[mask].mean())
        subgroups[name] = entry
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "phase_g_tail": g,
        "phase_h_common_errors": h1,
        "subgroups": subgroups,
        "official_test_loaded": False,
    }
    write_json(ERROR_ANALYSIS, payload)
    log(f"[error-analysis] extreme row {g.get('B', {}).get('extreme_row_abs_error')} -> {g.get('P', {}).get('extreme_row_abs_error')}")
    return 0


# ---------------------------------------------------------------------------
# frozen decision
# ---------------------------------------------------------------------------


def cmd_decision(args: argparse.Namespace) -> int:
    if not VALID_METRICS.exists():
        raise SystemExit("run score first")
    metrics = read_json(VALID_METRICS)
    gates = read_json(GATES_JSON)
    consistency = read_json(CONSISTENCY_AUDIT)
    q_gate = read_json(Q_GATE) if Q_GATE.exists() else {"purchase_q": False, "status": "NO_ENCODER_PRESERVATION_PURCHASE"}
    blocked = not (gates["all_pass"] and consistency["all_pass"])
    paired = metrics["paired"]
    pb = paired.get("P_minus_B")
    details: dict[str, Any] = {"q_gate_status": q_gate["status"], "blocked_precondition": blocked}

    branch = "INCONCLUSIVE_OR_BLOCKED"
    flags: list[str] = []
    qp = paired.get("Q_minus_P")
    if not blocked and pb is not None:
        per_seed = list(pb["per_seed_delta"].values())
        ordinary = pb["ordinary_y_gt_-3"]["delta"]
        top10 = pb["delta_top10_removed"]
        without_extreme = pb["extreme_row"]["delta_without_row"]
        gain_confirmed = (
            pb["delta_seed_mean"] <= -0.005
            and all(v < 0 for v in per_seed)
            and ordinary < 0
            and top10 < 0
            and without_extreme < 0
            and pb["bootstrap"]["ci_hi"] < 0
        )
        if gain_confirmed:
            branch = "FULL10K_PRETRAINING_GAIN_CONFIRMED_ON_DEV"
        elif pb["delta_seed_mean"] < 0 and (any(v < 0 for v in per_seed)) and (ordinary < 0 or top10 < 0):
            branch = "FULL10K_PRETRAINING_GAIN_WEAK_OR_UNSTABLE"
        elif pb["delta_seed_mean"] <= -0.005 and any(v >= 0 for v in per_seed):
            branch = "FULL10K_PRETRAINING_GAIN_WEAK_OR_UNSTABLE"
        else:
            branch = "FULL10K_PRETRAINING_NO_GAIN"
        # bulk-vs-tail trade-off flag
        m_p = metrics["seed_mean"].get("P")
        m_b = metrics["seed_mean"].get("B")
        if m_p is not None and m_b is not None:
            bulk_better = m_p["median_abs"] < m_b["median_abs"] and m_p["p90_abs"] < m_b["p90_abs"] and pb["ordinary_y_gt_-3"]["delta"] < 0
            tail_worse = (pb["tail_y_le_-3"]["delta"] or 0.0) > 0 or m_p["rmse"] > m_b["rmse"]
            if bulk_better and tail_worse:
                flags.append("BULK_GAIN_WITH_TAIL_COST")
        # Q
        qp = paired.get("Q_minus_P")
        if q_gate.get("purchase_q") and qp is not None:
            q_ok = (
                all(v < 0 for v in qp["per_seed_delta"].values())
                and qp["delta_seed_mean"] <= -0.003
                and qp["ordinary_y_gt_-3"]["delta"] < 0
                and qp["extreme_row"]["delta_without_row"] < 0
                and qp["seed_mean_rmse_new"] <= qp["seed_mean_rmse_ref"] + 0.01
            )
            flags.append("ENCODER_LR_ADAPTATION_USEFUL" if q_ok else "ENCODER_LR_ADAPTATION_NOT_SUPPORTED")
        else:
            flags.append("ENCODER_LR_ADAPTATION_NOT_SUPPORTED")
        details.update(
            {
                "P_minus_B": pb,
                "Q_minus_P": qp,
                "seed_mean": metrics["seed_mean"],
            }
        )
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "branch": branch,
        "flags": sorted(set(flags)),
        "details": details,
        "answers": {
            "1_full10k_pretraining_lowers_single_model_mae": (
                None if pb is None else pb["delta_seed_mean"] < 0
            ),
            "2_pretraining_transfer_vs_adaptation": {
                "P_minus_B_seed_mean": None if pb is None else pb["delta_seed_mean"],
                "Q_minus_P_seed_mean": None if qp is None else qp["delta_seed_mean"],
                "q_gate_status": q_gate["status"],
            },
            "3_distance_to_0.06_0.07": {
                "BJOC_full10k_seed1_official_test_context_only": 0.089903,
                "this_round_valid_B_seed_mean": metrics["seed_mean"].get("B", {}).get("mae"),
                "this_round_valid_P_seed_mean": metrics["seed_mean"].get("P", {}).get("mae"),
            },
            "4_tail_vs_bulk": {
                "ordinary_delta": None if pb is None else pb["ordinary_y_gt_-3"]["delta"],
                "tail_delta": None if pb is None else pb["tail_y_le_-3"]["delta"],
                "flags": sorted(set(flags)),
            },
        },
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    write_json(SCIENTIFIC_DECISION, payload)
    log(f"[decision] {branch} flags={payload['flags']}")
    return 0


def cmd_init_audit(args: argparse.Namespace) -> int:
    per_seed: dict[str, Any] = {}
    for s in SEEDS:
        b_path = OUT / f"train_B_seed{s}.json"
        if not b_path.exists():
            continue
        b = read_json(b_path)
        p_path = OUT / f"train_P_seed{s}.json"
        entry: dict[str, Any] = {
            "B_init_full_sha256": b["init_full_sha256"],
            "B_init_group_state_sha256": b["init_group_state_sha256"],
            "B_init_encoder_sha256": b["init_encoder_sha256"],
            "B_batch_order_hash": b["batch_order_hash"],
        }
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
        q_path = OUT / f"train_Q_seed{s}.json"
        if q_path.exists():
            q = read_json(q_path)
            entry.update(
                {
                    "Q_init_full_sha256": q["init_full_sha256"],
                    "Q_same_init_as_P": b["init_full_sha256"] == q["init_full_sha256"],
                    "Q_same_encoder_state_as_P": (OUT / f"train_P_seed{s}.json").exists()
                    and q.get("encoder_used_sha256") == read_json(OUT / f"train_P_seed{s}.json").get("encoder_used_sha256"),
                    "Q_batch_order_hash": q["batch_order_hash"],
                    "Q_same_batch_order_as_P": (OUT / f"train_P_seed{s}.json").exists()
                    and q["batch_order_hash"] == read_json(OUT / f"train_P_seed{s}.json")["batch_order_hash"],
                }
            )
        per_seed[str(s)] = entry
    payload = {"protocol": PROTOCOL_ID, "per_seed": per_seed, "git_commit": git_commit()}
    write_json(INIT_AUDIT, payload)
    log(f"[init-audit] written {INIT_AUDIT}")
    return 0


# ---------------------------------------------------------------------------
# remote chains
# ---------------------------------------------------------------------------


def cmd_stage_remote_prep(args: argparse.Namespace) -> int:
    if not OBJECTS_MANIFEST.exists():
        cmd_objects(args)
    if not CONSISTENCY_AUDIT.exists():
        cmd_consistency(args)
    if not DEPENDENCY_AUDIT.exists():
        cmd_audit_dependency(args)
    if not SWAP_AUDIT.exists():
        cmd_audit_swap(argparse.Namespace(random_cases=200, positive_cases=50))
    if not REBUILD_AUDIT.exists():
        cmd_rebuild_audit(argparse.Namespace(cases=100))
    if not GATES_JSON.exists():
        cmd_gates(args)
    cmd_freeze(args)
    if not REUSE_AUDIT.exists():
        cmd_reuse_audit(args)
    cmd_stage_b(args)
    cmd_init_audit(args)
    return 0


def cmd_stage_ssl_p(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    device = torch.device(args.device)
    if not (OUT / f"train_B_seed{seed}.json").exists():
        _train_downstream("B", seed, device)
    if not (OUT / f"pretraining_seed{seed}.json").exists():
        pretrain_full10k(seed, device)
    if not (OUT / f"train_P_seed{seed}.json").exists():
        _train_downstream("P", seed, device)
    cmd_init_audit(args)
    return 0


def cmd_stage_q(args: argparse.Namespace) -> int:
    gate = read_json(Q_GATE)
    if not gate.get("purchase_q"):
        raise SystemExit(f"Q not purchased: {gate['status']}")
    seed = int(args.seed)
    device = torch.device(args.device)
    if not (OUT / f"train_Q_seed{seed}.json").exists():
        _train_downstream("Q", seed, device, encoder_lr=ENCODER_LR_Q)
    cmd_init_audit(args)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("objects").set_defaults(func=cmd_objects)
    sub.add_parser("consistency").set_defaults(func=cmd_consistency)
    sub.add_parser("audit-dependency").set_defaults(func=cmd_audit_dependency)
    p = sub.add_parser("audit-swap")
    p.add_argument("--random-cases", type=int, default=200)
    p.add_argument("--positive-cases", type=int, default=50)
    p.set_defaults(func=cmd_audit_swap)
    p = sub.add_parser("rebuild-audit")
    p.add_argument("--cases", type=int, default=100)
    p.set_defaults(func=cmd_rebuild_audit)
    sub.add_parser("gates").set_defaults(func=cmd_gates)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    sub.add_parser("reuse-audit").set_defaults(func=cmd_reuse_audit)
    sub.add_parser("stage-b").set_defaults(func=cmd_stage_b)
    p = sub.add_parser("pretrain")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda a: (pretrain_full10k(a.seed, torch.device(a.device)), 0)[1])
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=["B", "P", "Q"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=lambda a: (_train_downstream(a.arm, a.seed, torch.device(a.device), encoder_lr=(ENCODER_LR_Q if a.arm == "Q" else None)), 0)[1])
    p = sub.add_parser("predict")
    p.add_argument("--models", nargs="+", required=True)
    p.set_defaults(func=cmd_predict)
    sub.add_parser("freeze-predictions").set_defaults(func=cmd_freeze_predictions)
    sub.add_parser("diagnostics").set_defaults(func=cmd_diagnostics)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("error-analysis").set_defaults(func=cmd_error_analysis)
    sub.add_parser("decision").set_defaults(func=cmd_decision)
    sub.add_parser("init-audit").set_defaults(func=cmd_init_audit)
    sub.add_parser("stage-remote-prep").set_defaults(func=cmd_stage_remote_prep)
    p = sub.add_parser("stage-ssl-p")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_stage_ssl_p)
    p = sub.add_parser("stage-q")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_stage_q)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
