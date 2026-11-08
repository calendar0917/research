"""Frozen definitions for the LCRC-S3-SP round v1 (selective three-hop paths).

Round: ``lcrc-pcba-s3-selective-v1``

Only one new model is trained::

    SP = W2 + S3 + root-conditioned selective path residual   (1,036,420 params)

The W2 trunk, the S3 path encoder, the S3 rooted pooling and the S3 residual
are inherited **unchanged** from ``lcrc_pcba_s3_v1`` (read-only).  The only new
computation is a two-head content attention over the *already enumerated* S3
paths of each root, plus ``attn - mean`` re-projected into the residual stream.

Nothing historical is edited.  The S3 training cache (Train ∪ Valid, label-free)
is read, never rewritten.  Official Test targets are NaN in the label artifact
and are only readable through ``sp_test.py`` **after** the prediction freeze;
the Test *structure* cache is a separate label-free artifact that the training
DataLoader never sees.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

S3_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1"
S3_CODE = S3_ROUND / "code"
BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
import sys as _sys  # noqa: E402

for _p in (str(CODE_DIR), str(S3_CODE)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import s3_common as C3  # noqa: E402  (adds bs/s3 code dirs; frozen S3 identities)
import s3_paths as S3P  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-s3-selective-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-s3-selective-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-s3-selective-checkpoint-v1"
SCHEMA_TEST_CACHE = "lcrc-pcba-s3-selective-test-structure-cache-v1"
SCHEMA_SELECTIVE_AUDIT = "lcrc-pcba-s3-selective-init-audit-v1"
SCHEMA_ARCH_AUDIT = "lcrc-pcba-s3-selective-architecture-audit-v1"
SCHEMA_TEST_FREEZE = "lcrc-pcba-s3-selective-test-freeze-v1"

# ---------------------------------------------------------------------------
# inherited, read-only identities
# ---------------------------------------------------------------------------

BC = C3.BC
DATASET = C3.DATASET
N_TASKS = C3.N_TASKS
N_MOLECULES = C3.N_MOLECULES
N_TRAIN = C3.N_TRAIN
N_VALID = C3.N_VALID
N_TEST = C3.N_TEST
BATCH_SIZE = C3.BATCH_SIZE
TOTAL_EPOCHS = C3.TOTAL_EPOCHS
STEPS_PER_EPOCH = C3.STEPS_PER_EPOCH
TOTAL_STEPS_40 = C3.TOTAL_STEPS_40

SOURCE = C3.SOURCE
RECIPE = C3.RECIPE
SEED = C3.SEED
WIDTHS = C3.WIDTHS
scheduled_lr = C3.scheduled_lr
lr_audit = C3.lr_audit


def lr_at(step: int) -> float:
    """The inherited from-scratch schedule at a global step (phase 3, all groups)."""
    return float(C3.scheduled_lr(int(step), phase=3, kind="all"))

BASE_MODEL_KEY = C3.BASE_MODEL_KEY
BASE_PARAMETERS = int(C3.BASE_PARAMETERS)          # 994,324
S3_ADDED = int(C3.S3_ADDED)                        # 24,208
S3_PARAMETERS = int(C3.EXPECTED_PARAMETERS["S3"])  # 1,018,532

S3_DIM = int(C3.S3_DIM)     # 48
SP_HEADS = 2
SP_HEAD_DIM = S3_DIM // SP_HEADS  # 24
SP_DIM = SP_HEADS * SP_HEAD_DIM   # 48
SP_PRIVATE_SEED = 1201

#: New-parameter arithmetic of the selective branch (exact, checked by sum()).
SP_BRANCH = {
    "q": {"in": int(WIDTHS["atom_out"]), "out": SP_DIM, "parameters": int(WIDTHS["atom_out"]) * SP_DIM + SP_DIM},
    "k": {"in": SP_DIM, "out": SP_DIM, "parameters": SP_DIM * SP_DIM + SP_DIM},
    "out": {"in": SP_DIM, "out": int(WIDTHS["width"]), "parameters": SP_DIM * int(WIDTHS["width"]) + int(WIDTHS["width"])},
}
SP_ADDED = int(sum(int(v["parameters"]) for v in SP_BRANCH.values()))  # 17,888
SP_PARAMETERS = S3_PARAMETERS + SP_ADDED                               # 1,036,420
SP_ADDED_FRACTION_OF_S3 = float(SP_ADDED) / float(S3_PARAMETERS)

INIT_RULE = (
    "torch.manual_seed(0) -> historical W2 seed-0 build (bs_model.build_model) -> "
    "historical S3 seed-0 arm (s3_model.build_model, private generator 901) -> "
    "base+S3 state copied bitwise into the SP model -> selective q/k drawn from a "
    "private CPU generator with seed 1201 (q first, then k, kaiming-uniform + "
    "fan_in bias, the same rule the S3 path encoder uses for its Linear) -> the "
    "selective output Linear(48,128) is zeroed, so epoch-0 logits == W2 == S3. "
    "The global torch RNG is never advanced by any of it."
)

# ---------------------------------------------------------------------------
# historical read-only references
# ---------------------------------------------------------------------------

W2 = C3.W2
S3 = {
    "round": "lcrc-pcba-s3-v1",
    "arm": "S3",
    "seed": 0,
    "epochs": 40,
    "parameters": S3_PARAMETERS,
    "new_parameters": S3_ADDED,
    "run_dir": REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/S3",
    "checkpoint_best": REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/S3/checkpoint_best.pt",
    "checkpoint_epoch40": REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/S3/checkpoint_epoch40.pt",
    "best_epoch": 29,
    "best_valid_ap": 0.271489039695166,
    "ap40": 0.2677211,
    "best_checkpoint_sha256": "ad665168da562120321171f6042a97f746ae4ae7377bd8364fc8d6264af007d0",
    "epoch40_checkpoint_sha256": "5aa8a31782cddda1c1bfe6cd8450714252151f34d0973f47aaae67a9174b6c9b",
    "model_state_sha256_at_init": "a850ac855cc38858cfc0bfa90d2e4d81c27be11ee3af373a37099d9277a90722",
    "base_state_sha256_at_init": "6a3df38f00a5a4754c310a82d4ad5f71d7f89e3392f1c914aacc4113a3d6f53b",
    "gpu_hours": 1.8479,
    "epoch_seconds_mean": 165.6,
}
W2_CHECKPOINT_BEST = C3.W2_CHECKPOINT_BEST
S3_CHECKPOINT_BEST = S3["checkpoint_best"]

#: The frozen Train+Valid S3 structural cache (read-only in this round).
S3_TRAIN_CACHE_DIR = C3.S3_CACHE_DIR
#: The label-free Test *structure* cache built for inference only.
S3_TEST_CACHE_DIR = RUNS / "test_s3_cache"

SP_RUN_DIR = RUNS / "SP"
SP_RUN_JSON = SP_RUN_DIR / "run.json"
SP_CURVE_CSV = SP_RUN_DIR / "train_curve.csv"
SP_CHECKPOINT_BEST = SP_RUN_DIR / "checkpoint_best.pt"
SP_CHECKPOINT_EPOCH40 = SP_RUN_DIR / "checkpoint_epoch40.pt"
SP_CHECKPOINT_LAST = SP_RUN_DIR / "checkpoint_last.pt"

MODELS: dict[str, dict[str, Any]] = {
    "SP": {
        "name": "LCRC-S3-SP",
        "role": "root_conditioned_selective_three_hop_residual",
        "new_parameters": SP_ADDED,
        "parameters": SP_PARAMETERS,
        "assigned_pool": "res-gpu1",
    }
}
MODEL_ORDER = ("SP",)
EXPECTED_PARAMETERS = {k: int(v["parameters"]) for k, v in MODELS.items()}

#: Private generator seed for the selective q/k (round brief §5), not an offset.
PRIVATE_GENERATOR_SEEDS = {"SP_selective": int(SP_PRIVATE_SEED)}

BUDGET = {"gpu_target_hours": 3.0, "gpu_hard_hours": 5.0, "max_concurrent_gpus": 1}

# ---------------------------------------------------------------------------
# decision rules (pre-registered in PROTOCOL.md §10)
# ---------------------------------------------------------------------------

#: A dimension counts as "improved" only above this practical threshold; the
#: same 0.002 scale the S3 round used for its NO_CLEAR_GAIN rule.
GAIN_THRESHOLD = 0.002
WORK_TARGET = {"best_valid_ap_min": 0.2745, "test_gain_min": 0.003}
DECISION_RULES = {
    "valid_gain_threshold": GAIN_THRESHOLD,
    "test_gain_threshold": GAIN_THRESHOLD,
    "labels": {
        "valid_and_test_gain": "VALID_AND_TEST_GAIN",
        "valid_only_gain": "VALID_ONLY_GAIN",
        "test_only_gain": "TEST_ONLY_GAIN",
        "no_clear_gain": "NO_CLEAR_GAIN",
    },
}

# ---------------------------------------------------------------------------
# IO helpers (inherited)
# ---------------------------------------------------------------------------

utcnow = C3.utcnow
write_json = C3.write_json
read_json = C3.read_json
sha256_file = C3.sha256_file
sha256_bytes = C3.sha256_bytes
log = C3.log
environment = C3.environment
seed_everything = C3.seed_everything
capture_rng_state = C3.capture_rng_state
rng_state_hashes = C3.rng_state_hashes
git_commit = C3.git_commit
git_state = C3.git_state
track_rel = C3.track_rel
state_fingerprint = C3.state_fingerprint
parameter_count = C3.parameter_count
better_candidate = C3.better_candidate


def recipe_for_model(model_key: str = "SP") -> dict[str, Any]:
    m = MODELS[model_key]
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "model": model_key,
            "model_name": m["name"],
            "role": m["role"],
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_every": 1,
            "eval_epochs": list(range(1, TOTAL_EPOCHS + 1)),
            "widths": dict(WIDTHS),
            "base_parameters": BASE_PARAMETERS,
            "s3_parameters": S3_PARAMETERS,
            "new_parameters": int(m["new_parameters"]),
            "parameters": int(m["parameters"]),
            "selective_branch": {
                "heads": SP_HEADS,
                "head_dim": SP_HEAD_DIM,
                "query_source": "static atom encoder output a (192) -> Linear(192,48)",
                "key_source": "s3 path embedding z (48) -> Linear(48,48)",
                "value_source": "s3 path embedding z (48), no extra projection",
                "score": "q[root] . k / sqrt(24) per head",
                "normalisation": "segment softmax inside each root, per head",
                "delta": "attn - uniform_mean_z",
                "output": "Linear(48,128), zero-initialised",
                "empty_root_rule": "r_selective[root] = 0 when counts[root] == 0",
            },
            "lr_schedule": lr_audit(),
            "init_rule": INIT_RULE,
            "model_selection": "argmax official Validation macro AP over epochs 1..40 (earliest epoch wins ties)",
            "test_policy": (
                "train/valid only until the 40-epoch run is complete and the best-Valid checkpoint is "
                "fixed; Test logits are then frozen and hashed before the official Test labels are opened"
            ),
        }
    )
    return recipe


# ---------------------------------------------------------------------------
# S3 caches
# ---------------------------------------------------------------------------


def load_s3_train_cache(path: Path | str = S3_TRAIN_CACHE_DIR) -> dict[str, Any]:
    """The frozen Train+Valid structural cache (read-only)."""
    cache = C3.load_s3_cache(path)
    return cache


def load_task_gated(*, train_cache: Path | str = S3_TRAIN_CACHE_DIR) -> Any:
    """Official split + frozen identities + the read-only Train+Valid S3 cache."""
    task = C3.load_task_with_s3(train_cache)
    checks = {
        "feature_spec_hash": task.spec.hash() == SOURCE["feature_spec_hash"],
        "norm_fingerprint": task.norm["norm_fingerprint"] == SOURCE["norm_fingerprint"],
        "split_partition_sha256": task.split_partition_sha256 == SOURCE["split_partition_sha256"],
        "label_artifact_sha256": task.label_artifact_sha256 == SOURCE["label_artifact_sha256"],
        "test_labels_masked": not bool(np.isfinite(task.labels[task.test_rows]).any()),
        "s3_scope_is_train_valid_only": not bool(
            np.intersect1d(np.asarray(task.s3["rows"], np.int64), np.asarray(task.test_rows, np.int64)).size
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"frozen data identity drifted: {failed}")
    return task


# ---------------------------------------------------------------------------
# label-free Test S3 structure cache (inference only, never in a DataLoader)
# ---------------------------------------------------------------------------


def build_s3_test_cache(
    task: Any,
    out_dir: Path | str = S3_TEST_CACHE_DIR,
    *,
    verify_per_molecule: bool = True,
    log_every: int = 20_000,
) -> dict[str, Any]:
    """Enumerate S3 for every official Test row.  Structure only, no labels.

    The enumeration, FeatureSpec, field layout and per-molecule closed-form
    verification are **identical** to ``s3_common.build_s3_cache``; the only
    difference is the row scope (official Test instead of Train ∪ Valid) and the
    cache's own schema/invariants.  It never touches the Train+Valid cache.
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = task.cache
    scope = np.sort(np.asarray(task.test_rows, dtype=np.int64))
    if scope.size != int(N_TEST):
        raise RuntimeError(f"Test scope {scope.size} != frozen N_TEST {N_TEST}")
    if np.intersect1d(scope, np.asarray(task.train_rows, np.int64)).size or np.intersect1d(
        scope, np.asarray(task.valid_rows, np.int64)
    ).size:
        raise RuntimeError("Test scope overlaps Train/Valid rows")
    if str(cache["meta"].get("feature_spec_hash")) != SOURCE["feature_spec_hash"]:
        raise RuntimeError("S3 test cache build: feature spec drift")

    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    bond_type = np.asarray(cache["bond_type"])
    atom_type = np.asarray(cache["atom_type"])

    atom_counts = (atom_ptr[scope + 1] - atom_ptr[scope]).astype(np.int64)
    n_scope_atoms = int(atom_counts.sum())

    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("end", "m1", "m2", "b1", "b2", "b3", "d")}
    counts_parts: list[np.ndarray] = []
    row_paths = np.zeros(scope.size, dtype=np.int64)
    dist_totals = np.zeros(3, dtype=np.int64)

    for i, row in enumerate(scope.tolist()):
        a0 = int(atom_ptr[row])
        a1 = int(atom_ptr[row + 1])
        n = a1 - a0
        b0 = int(bond_ptr[row])
        b1 = int(bond_ptr[row + 1])
        paths = None
        if n < 4 or b1 == b0:
            counts = np.zeros(n, dtype=np.int64)
        else:
            bi = bond_index[b0:b1].astype(np.int64)
            bt = bond_type[b0:b1].astype(np.int64)
            adj = S3P.molecule_adjacency(n, bi[:, 0], bi[:, 1], bt)
            at = atom_type[a0:a1]
            dist = S3P.all_pairs_shortest_paths(n, adj)
            paths = S3P.enumerate_s3(n, adj, at, dist, keep_mid_atoms=True)
            counts = paths.per_root_counts()
            if verify_per_molecule:
                if not np.array_equal(counts, S3P.count_s3_closed_form(n, adj)):
                    raise RuntimeError(f"test row {row}: S3 closed form != enumeration")
                if paths.n_paths and not np.all(
                    paths.end != np.repeat(np.arange(n), counts)
                ):
                    raise RuntimeError(f"test row {row}: endpoint equals its own root")
            if paths.n_paths:
                dist_totals += np.bincount(paths.dist.astype(np.int64) - 1, minlength=3)
        counts_parts.append(counts)
        row_paths[i] = int(counts.sum())
        if paths is not None and paths.n_paths:
            parts["end"].append(paths.end.astype(np.int32))
            parts["m1"].append(paths.atom_type_mid1.astype(np.int8))
            parts["m2"].append(paths.atom_type_mid2.astype(np.int8))
            parts["b1"].append(paths.bond_type_1.astype(np.int8))
            parts["b2"].append(paths.bond_type_2.astype(np.int8))
            parts["b3"].append(paths.bond_type_3.astype(np.int8))
            parts["d"].append(paths.dist.astype(np.int8))
        if log_every and (i + 1) % int(log_every) == 0:
            log(f"[sp-test-cache] {i + 1}/{scope.size} rows, {int(row_paths.sum())} paths")

    per_atom_counts = np.concatenate(counts_parts) if counts_parts else np.zeros(0, dtype=np.int64)
    if per_atom_counts.size != n_scope_atoms:
        raise RuntimeError("per-atom S3 count length != Test atom count")
    s3_atom_ptr = np.concatenate([[0], np.cumsum(per_atom_counts)]).astype(np.int64)

    def _stack(key: str, dtype) -> np.ndarray:
        return np.concatenate(parts[key]).astype(dtype) if parts[key] else np.zeros(0, dtype=dtype)

    arrays = {
        "s3_atom_ptr": s3_atom_ptr,
        "s3_end": _stack("end", np.int32),
        "s3_mid1_type": _stack("m1", np.int8),
        "s3_mid2_type": _stack("m2", np.int8),
        "s3_bond1": _stack("b1", np.int8),
        "s3_bond2": _stack("b2", np.int8),
        "s3_bond3": _stack("b3", np.int8),
        "s3_dist": _stack("d", np.int8),
        "s3_row_paths": row_paths.astype(np.int64),
        "s3_row_atoms": atom_counts.astype(np.int64),
        "rows": scope.astype(np.int64),
    }
    n_paths = int(s3_atom_ptr[-1])
    if n_paths != int(row_paths.sum()) or n_paths != int(arrays["s3_end"].size):
        raise RuntimeError("Test S3 cache path count mismatch")

    hashes: dict[str, str] = {}
    for name, arr in arrays.items():
        path = out_dir / f"{name}.npy"
        np.save(path, arr)
        hashes[name] = sha256_file(path)
    seconds = float(time.perf_counter() - t0)
    meta = {
        "schema": SCHEMA_TEST_CACHE,
        "round_id": ROUND_ID,
        "created_utc": utcnow(),
        "scope": "official Test rows (sorted); structure only; no labels",
        "roles": "inference only; must never enter a training DataLoader",
        "n_rows": int(scope.size),
        "n_atoms": int(n_scope_atoms),
        "n_paths": int(n_paths),
        "dist_counts": {"1": int(dist_totals[0]), "2": int(dist_totals[1]), "3": int(dist_totals[2])},
        "bytes": int(sum((out_dir / f"{name}.npy").stat().st_size for name in arrays)),
        "build_seconds": seconds,
        "rows_sha256": sha256_bytes(np.ascontiguousarray(scope, dtype=np.int64).tobytes()),
        "field_sha256": hashes,
        "feature_spec_hash": SOURCE["feature_spec_hash"],
        "split_partition_sha256": SOURCE["split_partition_sha256"],
        "labels_present": False,
        "test_rows_present": True,
        "training_scope": False,
        "definition": C3.S3_DEFINITION,
    }
    write_json(out_dir / "meta.json", meta)
    log(f"[sp-test-cache] rows={meta['n_rows']} paths={meta['n_paths']} seconds={seconds:.1f}")
    return meta


def load_s3_test_cache(path: Path | str = S3_TEST_CACHE_DIR) -> dict[str, Any]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA_TEST_CACHE:
        raise RuntimeError(f"Test S3 cache schema {meta.get('schema')} != {SCHEMA_TEST_CACHE}")
    if bool(meta.get("labels_present")) or not bool(meta.get("training_scope") is False):
        raise RuntimeError("Test S3 cache is not label-free / not inference-only")
    out: dict[str, Any] = {"meta": meta, "path": path}
    for name in C3.S3_CACHE_FIELDS:
        out[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    rows = np.asarray(out["rows"], dtype=np.int64)
    row_index = np.full(N_MOLECULES, -1, dtype=np.int64)
    row_index[rows] = np.arange(rows.size, dtype=np.int64)
    out["row_index"] = row_index
    out["row_atom_ptr"] = np.concatenate([[0], np.cumsum(np.asarray(out["s3_row_atoms"], dtype=np.int64))])
    if int(out["row_atom_ptr"][-1]) != int(out["s3_atom_ptr"].shape[0]) - 1:
        raise RuntimeError("Test S3 cache atom bookkeeping mismatch")
    return out


def test_cache_identity(path: Path | str | None = None) -> dict[str, Any]:
    base = Path(path) if path is not None else S3_TEST_CACHE_DIR
    meta_path = base / "meta.json"
    if not meta_path.is_file():
        return {"present": False, "path": track_rel(base)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "path": track_rel(base),
        "meta_sha256": sha256_file(meta_path),
        "n_rows": int(meta["n_rows"]),
        "n_paths": int(meta["n_paths"]),
        "bytes": int(meta["bytes"]),
        "build_seconds": float(meta["build_seconds"]),
        "rows_sha256": meta["rows_sha256"],
        "field_sha256": meta["field_sha256"],
        "labels_present": bool(meta.get("labels_present", True)),
        "training_scope": meta.get("training_scope"),
        "definition": meta.get("definition"),
    }


def s3_train_cache_identity(path: Path | str | None = None) -> dict[str, Any]:
    return C3.s3_cache_identity(S3_TRAIN_CACHE_DIR if path is None else path)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

collate_with_s3 = C3.collate_with_s3
collate_s3 = C3.collate_s3
check_s3_batch_isolation = C3.check_s3_batch_isolation


def batch_for_rows(task: Any, s3: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Frozen ``pc_data.collate`` + S3 tensors, for any in-scope S3 cache."""
    return C3.collate_with_s3(task, s3, rows)


__all__ = [
    "ANALYSIS",
    "BASE_MODEL_KEY",
    "BASE_PARAMETERS",
    "BATCH_SIZE",
    "BC",
    "BUDGET",
    "CODE_DIR",
    "DATASET",
    "DECISION_RULES",
    "EXPECTED_PARAMETERS",
    "GAIN_THRESHOLD",
    "INIT_RULE",
    "MODELS",
    "MODEL_ORDER",
    "N_MOLECULES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PRIVATE_GENERATOR_SEEDS",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "S3",
    "S3_ADDED",
    "S3_BRANCH",
    "S3_CHECKPOINT_BEST",
    "S3_CODE",
    "S3_DIM",
    "S3_PARAMETERS",
    "S3_ROUND",
    "S3_TEST_CACHE_DIR",
    "SP_RUN_DIR",
    "SP_RUN_JSON",
    "SP_CURVE_CSV",
    "SP_CHECKPOINT_BEST",
    "SP_CHECKPOINT_EPOCH40",
    "SP_CHECKPOINT_LAST",
    "S3_TRAIN_CACHE_DIR",
    "SCHEMA_ARCH_AUDIT",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_RUN",
    "SCHEMA_SELECTIVE_AUDIT",
    "SCHEMA_TEST_CACHE",
    "SCHEMA_TEST_FREEZE",
    "SEED",
    "SOURCE",
    "SP_ADDED",
    "SP_ADDED_FRACTION_OF_S3",
    "SP_BRANCH",
    "SP_DIM",
    "SP_HEAD_DIM",
    "SP_HEADS",
    "SP_PARAMETERS",
    "SP_PRIVATE_SEED",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "W2",
    "W2_CHECKPOINT_BEST",
    "WIDTHS",
    "WORK_TARGET",
    "batch_for_rows",
    "better_candidate",
    "build_s3_test_cache",
    "capture_rng_state",
    "check_s3_batch_isolation",
    "collate_s3",
    "collate_with_s3",
    "environment",
    "git_commit",
    "git_state",
    "load_s3_test_cache",
    "lr_at",
    "load_s3_train_cache",
    "load_task_gated",
    "log",
    "lr_audit",
    "parameter_count",
    "read_json",
    "recipe_for_model",
    "rng_state_hashes",
    "s3_train_cache_identity",
    "scheduled_lr",
    "seed_everything",
    "sha256_bytes",
    "sha256_file",
    "state_fingerprint",
    "test_cache_identity",
    "track_rel",
    "utcnow",
    "write_json",
]
