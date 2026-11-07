"""Frozen definitions for the LCRC PCBA explicit three-hop (S3) round v1.

Round: ``lcrc-pcba-s3-v1``

The base model is the historical **W2** (2x LCRC-A, 994,324 parameters, one
atom encode + one bond encode + one-shot S1/S2 statistics + per-atom ``f`` +
global ``rho`` + four-head attention pooling, no iterative message passing).

This round asks one question: does an explicit, **one-shot** third-hop
structure encoding -- every simple bond path ``r-u-v-x`` of length exactly three
per root atom -- add validation information beyond W2's radius-2 statistics, or
is it just capacity?  Two and only two arms are trained, both from scratch:
``S3`` (W2 + rooted S3 residual, new 24,208 params) and ``F`` (W2 +
near-parameter-matched per-atom FFN residual, new 24,285 params).

Inheritance is read-only: ``bs_common``/``bs_model``/``bs_train`` (W2 identity,
frozen recipe, order plan, official evaluator), ``tu_model``/``nmp_model`` (the
trunk and the readout), ``sg_data``/``pc_data``/``pc_metrics`` (data + metric).
Nothing historical is edited.  No official Test row is indexed, cached or
scored -- the label artifact keeps every Test target at NaN and the S3 cache is
built from the Train+Valid row set only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_ROUND / "code"
if str(BS_CODE) not in sys.path:
    sys.path.insert(0, str(BS_CODE))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import bs_common as BC  # noqa: E402  (frozen recipe, W2 identity, IO helpers)
import s3_paths as S3P  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-s3-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-s3-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-s3-checkpoint-v1"
SCHEMA_CACHE = "lcrc-pcba-s3-structural-cache-v1"

# ---------------------------------------------------------------------------
# inherited, read-only identities
# ---------------------------------------------------------------------------

DATASET = BC.DATASET
N_TASKS = BC.N_TASKS
N_MOLECULES = BC.N_MOLECULES
N_TRAIN = BC.N_TRAIN
N_VALID = BC.N_VALID
N_TEST = BC.N_TEST
BATCH_SIZE = BC.BATCH_SIZE
TOTAL_EPOCHS = BC.TOTAL_EPOCHS
STEPS_PER_EPOCH = BC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = BC.TOTAL_STEPS_40

SOURCE = BC.SOURCE
RECIPE = BC.RECIPE
SEED = BC.SEED
MODEL_SEED = BC.MODEL_SEED
scheduled_lr = BC.scheduled_lr
lr_audit = BC.lr_audit

#: Frozen W2 width dict (scale 2) passed explicitly to every constructor.
WIDTHS = BC.config_for_scale(2)
BASE_MODEL_KEY = "W2"
BASE_PARAMETERS = 994_324

# ---------------------------------------------------------------------------
# the historical W2 reference (read-only; never retrained in this round)
# ---------------------------------------------------------------------------

W2_RUN_DIR = BS_ROUND / "runs" / BASE_MODEL_KEY
W2_RUN_JSON = W2_RUN_DIR / "run.json"
W2_CURVE = W2_RUN_DIR / "train_curve.csv"
W2_PER_TASK = W2_RUN_DIR / "per_task_ap.csv"
W2_TRAIN_EVAL = W2_RUN_DIR / "train_eval_epoch40.json"
W2_CHECKPOINT_BEST = W2_RUN_DIR / "checkpoint_best.pt"
W2_CHECKPOINT_EPOCH40 = W2_RUN_DIR / "checkpoint_epoch40.pt"

W2 = {
    "round": "lcrc-pcba-backbone-scaling-v1",
    "arm": "W2",
    "seed": 0,
    "epochs": 40,
    "parameters": BASE_PARAMETERS,
    "ap40": 0.26186691813350904,
    "train_ap40": 0.5818745570283522,
    "train_bce40": 0.02628940533386738,
    "valid_bce40": 0.04653792303564479,
    "best_epoch": 24,
    "best_valid_ap": 0.26527331610028143,
    "best_checkpoint_sha256": "d86fe5e56e898d382acf66eb0533beb915f8ca39144c6c1c74e25e147667d2bd",
    "epoch40_checkpoint_sha256": "7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a",
    "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
    "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
    "trunk_init_fingerprint": "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d",
    "full_trunk_init_fingerprint": "109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db",
    "gpu_hours": 1.3949014246219305,
    "epoch_seconds_mean": 124.86,
}

BASE_AP40 = float(W2["ap40"])
BASE_BEST_AP = float(W2["best_valid_ap"])
BASE_BEST_EPOCH = int(W2["best_epoch"])

# ---------------------------------------------------------------------------
# the S3 definition (frozen)
# ---------------------------------------------------------------------------

#: Exactly three consecutive real bonds, four distinct atoms, no walk.
S3_DEFINITION = {
    "path": "p = (r, u, v, x)",
    "bonds": ["r-u", "u-v", "v-x"],
    "distinct_atoms": 4,
    "length": 3,
    "shortest_distance_classes": [1, 2, 3],
    "deduplication": "none (path multiplicity to the same endpoint is preserved)",
    "truncation": "none",
    "random_sampling": "none",
    "iterative_message_passing": False,
    "rooted": True,
    "bond_type_slots": "three position-independent bond-type embeddings",
    "atom_type_slots": "two intermediate atom-type embeddings (mid1, mid2)",
}

S3_DIM = 48
S3_OUT_WIDTH = 128  # == W2 local width
S3_INPUT_DIM = 2 * S3_DIM + 1  # [mu_r, q_r, log1p(m_r)]

#: New-parameter arithmetic for the S3 branch.
S3_BRANCH = {
    "we": {"in": int(WIDTHS["atom_out"]), "out": S3_DIM, "parameters": int(WIDTHS["atom_out"]) * S3_DIM + S3_DIM},
    "e_mid1": {"vocab": 16, "dim": S3_DIM, "parameters": 16 * S3_DIM},
    "e_mid2": {"vocab": 16, "dim": S3_DIM, "parameters": 16 * S3_DIM},
    "e_bond1": {"vocab": 5, "dim": S3_DIM, "parameters": 5 * S3_DIM},
    "e_bond2": {"vocab": 5, "dim": S3_DIM, "parameters": 5 * S3_DIM},
    "e_bond3": {"vocab": 5, "dim": S3_DIM, "parameters": 5 * S3_DIM},
    "e_dist": {"vocab": 3, "dim": S3_DIM, "parameters": 3 * S3_DIM},
    "out": {"in": S3_INPUT_DIM, "out": S3_OUT_WIDTH, "parameters": S3_INPUT_DIM * S3_OUT_WIDTH + S3_OUT_WIDTH},
}
S3_ADDED = int(sum(int(v["parameters"]) for v in S3_BRANCH.values()))

#: FFN control: LayerNorm(128) -> Linear(128,H) -> SiLU -> Linear(H,128),
#: last Linear zero-initialised.  Delta(H) = 257*H + 384.
FFN_FORMULA = {"slope": 257, "intercept": 384}


def ffn_parameters(hidden: int) -> int:
    w = int(WIDTHS["width"])
    return 2 * w + (w * int(hidden) + int(hidden)) + (int(hidden) * w + w)


def choose_ffn_hidden(target_new_parameters: int) -> dict[str, Any]:
    """Positive integer ``H`` minimising ``|params_S3 - params_F(H)|``."""
    best: tuple[int, int, int] | None = None
    for h in range(1, 4096):
        delta = abs(int(target_new_parameters) - ffn_parameters(h))
        if best is None or delta < best[0]:
            best = (delta, h, ffn_parameters(h))
    assert best is not None
    return {"hidden": int(best[1]), "parameters": int(best[2]), "absolute_difference": int(best[0])}


_CHOSEN = choose_ffn_hidden(S3_ADDED)
FFN_HIDDEN = int(_CHOSEN["hidden"])
F_ADDED = int(_CHOSEN["parameters"])
PARAMETER_GAP = int(S3_ADDED - F_ADDED)
PARAMETER_GAP_FRACTION = float(abs(PARAMETER_GAP) / S3_ADDED)

MODELS: dict[str, dict[str, Any]] = {
    "S3": {
        "name": "LCRC-S3",
        "role": "explicit_one_shot_three_hop_residual",
        "new_parameters": int(S3_ADDED),
        "parameters": int(BASE_PARAMETERS + S3_ADDED),
        "assigned_pool": "res-gpu1",
    },
    "F": {
        "name": "LCRC-W2-LocalResidualFFN",
        "role": "near_parameter_matched_capacity_control",
        "new_parameters": int(F_ADDED),
        "parameters": int(BASE_PARAMETERS + F_ADDED),
        "assigned_pool": "res-gpu0",
    },
}
MODEL_ORDER = ("S3", "F")
EXPECTED_PARAMETERS = {k: int(v["parameters"]) for k, v in MODELS.items()}
PARAMETER_GAP_MAX_FRACTION = 0.01

#: Private-generator offsets for the new modules (never touch the global RNG).
PRIVATE_GENERATOR_OFFSETS = {"S3": 901, "F": 902}

INIT_RULE = (
    "torch.manual_seed(0) -> historical W2 seed-0 build (bs_model.build_model) -> "
    "new modules constructed under torch.random.fork_rng -> base state copied bitwise -> "
    "new tensors re-initialised from a private generator seed*1000003+offset; the S3 "
    "output layer (and the FFN's last Linear) are zero, so epoch-0 logits == W2 logits"
)

# ---------------------------------------------------------------------------
# stage-0 structural audit (frozen before it is run)
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 4096
#: Pre-fixed size strata (atoms per molecule) and the minimum draw per stratum.
SAMPLE_STRATA = [
    ("le9", 1, 9, 256),
    ("10_14", 10, 14, 256),
    ("15_19", 15, 19, 512),
    ("20_24", 20, 24, 512),
    ("25_29", 25, 29, 512),
    ("30_39", 30, 39, 512),
    ("ge40", 40, 10**9, 256),
]
SAMPLE_SELECTION = (
    "stratum = pre-fixed atom-count interval; allocation = max(minimum, proportional) then "
    "largest-remainder to 4096; within a stratum the train rows are sorted by row id and "
    "sampled at floor((i+0.5)*n/K) — deterministic, label-free, no RNG"
)

GATES = {
    "min_molecule_fraction_with_dist3": 0.30,
    "max_cache_gib": 20.0,
    "max_preprocess_minutes": 90.0,
    "max_epoch_ratio_vs_w2": 2.5,
    "require_witness_pair": True,
    "require_no_truncation": True,
}

# ---------------------------------------------------------------------------
# decision rules (pre-registered; §10 of the round brief)
# ---------------------------------------------------------------------------

DECISION_RULES = {
    "promising": {
        "d_best_s3_min": 0.005,
        "d_mech_min": 0.003,
        "mean_delta_support_ge20_min": 0.003,
        "improved_support_ge20_min": 60,
        "support_ge20_total": 107,
        "label": "PROMISING_S3_SIGNAL",
    },
    "no_clear_gain": {"d_best_s3_max": 0.002, "label": "NO_CLEAR_GAIN"},
    "regression": {"d_best_s3_max": -0.002, "label": "REGRESSION"},
    "support_strata": {"ge20": 20, "ge5_lt20": 5},
    "capacity_only": {"label": "CAPACITY_ONLY"},
    "weak_or_mixed": {"label": "WEAK_OR_MIXED"},
}

BUDGET = {"gpu_target_hours": 6.0, "gpu_hard_hours": 10.0, "max_concurrent_gpus": 2}

# ---------------------------------------------------------------------------
# IO helpers (inherited)
# ---------------------------------------------------------------------------

utcnow = BC.utcnow
write_json = BC.write_json
read_json = BC.read_json
sha256_file = BC.sha256_file
sha256_bytes = BC.sha256_bytes
log = BC.log
environment = BC.environment
seed_everything = BC.seed_everything
capture_rng_state = BC.capture_rng_state
rng_state_hashes = BC.rng_state_hashes
git_commit = BC.git_commit
git_state = BC.git_state
track_rel = BC.track_rel
state_fingerprint = BC.state_fingerprint
parameter_count = BC.parameter_count


def recipe_for_model(model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise KeyError(model_key)
    m = MODELS[model_key]
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "model": model_key,
            "model_name": m["name"],
            "role": m["role"],
            "seed": int(BC.SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_every": 1,
            "eval_epochs": list(range(1, TOTAL_EPOCHS + 1)),
            "widths": dict(WIDTHS),
            "base_parameters": int(BASE_PARAMETERS),
            "new_parameters": int(m["new_parameters"]),
            "parameters": int(m["parameters"]),
            "lr_schedule": lr_audit(),
            "init_rule": INIT_RULE,
            "model_selection": "argmax official Validation macro AP over epochs 1..40 (earliest epoch wins ties)",
            "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        }
    )
    return recipe


# ---------------------------------------------------------------------------
# the S3 structural cache (Train+Valid only, label-free)
# ---------------------------------------------------------------------------

S3_CACHE_DIR = RUNS / "s3_cache"
S3_CACHE_FIELDS = (
    "s3_atom_ptr",
    "s3_end",
    "s3_mid1_type",
    "s3_mid2_type",
    "s3_bond1",
    "s3_bond2",
    "s3_bond3",
    "s3_dist",
    "s3_row_paths",
    "s3_row_atoms",
    "rows",
)


def better_candidate(candidate_ap: float, best_ap: float | None, candidate_epoch: int, best_epoch: int | None) -> bool:
    """The frozen selection rule: strictly higher AP wins; the earliest epoch wins ties."""
    if best_ap is None or best_epoch is None:
        return True
    if float(candidate_ap) > float(best_ap):
        return True
    if float(candidate_ap) == float(best_ap) and int(candidate_epoch) < int(best_epoch):
        return True
    return False


def s3_scope_rows(task: Any) -> np.ndarray:
    """Train ∪ Valid row ids (sorted).  Official Test rows are never included."""
    rows = np.sort(np.concatenate([np.asarray(task.train_rows, np.int64), np.asarray(task.valid_rows, np.int64)]))
    if np.intersect1d(rows, np.asarray(task.test_rows, np.int64)).size:
        raise RuntimeError("S3 scope overlaps official Test rows")
    return rows


def build_s3_cache(
    task: Any,
    out_dir: Path | str = S3_CACHE_DIR,
    *,
    rows: Sequence[int] | np.ndarray | None = None,
    verify_per_molecule: bool = True,
    log_every: int = 25_000,
) -> dict[str, Any]:
    """Enumerate S3 for every in-scope molecule and persist the structural cache.

    Only structure (bond list, bond types, atom types) and the in-scope row ids
    are written.  No label array is read anywhere in this function.
    """

    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = task.cache
    scope = np.asarray(rows if rows is not None else s3_scope_rows(task), dtype=np.int64)
    scope = np.sort(scope)
    if np.intersect1d(scope, np.asarray(task.test_rows, np.int64)).size:
        raise RuntimeError("refusing to build an S3 cache that contains official Test rows")
    if str(cache["meta"].get("feature_spec_hash")) != SOURCE["feature_spec_hash"]:
        raise RuntimeError("S3 cache build: feature spec drift")

    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    bond_type = np.asarray(cache["bond_type"])
    atom_type = np.asarray(cache["atom_type"])

    atom_counts = (atom_ptr[scope + 1] - atom_ptr[scope]).astype(np.int64)
    n_scope_atoms = int(atom_counts.sum())

    counts_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    m1_parts: list[np.ndarray] = []
    m2_parts: list[np.ndarray] = []
    b1_parts: list[np.ndarray] = []
    b2_parts: list[np.ndarray] = []
    b3_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    row_paths = np.zeros(scope.size, dtype=np.int64)
    row_mol_atoms = atom_counts
    dist_totals = np.zeros(3, dtype=np.int64)
    n_bonds_total = 0

    for i, row in enumerate(scope.tolist()):
        a0 = int(atom_ptr[row])
        a1 = int(atom_ptr[row + 1])
        n = a1 - a0
        b0 = int(bond_ptr[row])
        b1 = int(bond_ptr[row + 1])
        n_bonds_total += b1 - b0
        if n < 4 or b1 == b0:
            counts = np.zeros(n, dtype=np.int64)
            paths = None
        else:
            bi = bond_index[b0:b1].astype(np.int64)  # molecule-local ids
            bt = bond_type[b0:b1].astype(np.int64)
            adj = S3P.molecule_adjacency(n, bi[:, 0], bi[:, 1], bt)
            at = atom_type[a0:a1]
            dist = S3P.all_pairs_shortest_paths(n, adj)
            paths = S3P.enumerate_s3(n, adj, at, dist, keep_mid_atoms=True)
            counts = paths.per_root_counts()
            if verify_per_molecule:
                closed = S3P.count_s3_closed_form(n, adj)
                if not np.array_equal(counts, closed):
                    raise RuntimeError(f"row {row}: S3 closed form {int(closed.sum())} != enumeration {int(counts.sum())}")
                if int(counts.sum()) != int(paths.n_paths):
                    raise RuntimeError(f"row {row}: path count bookkeeping mismatch")
                if paths.n_paths:
                    same = paths.end >= 0
                    if not np.all(same):
                        raise RuntimeError(f"row {row}: negative endpoint")
                    if not np.all((paths.dist >= 1) & (paths.dist <= 3)):
                        raise RuntimeError(f"row {row}: S3 distance class outside {{1,2,3}}")
                    if not np.all(paths.end != np.repeat(np.arange(n), counts)):
                        raise RuntimeError(f"row {row}: S3 path endpoint equals its own root")
            if paths is not None and paths.n_paths:
                dist_totals += np.bincount(paths.dist.astype(np.int64) - 1, minlength=3)
        counts_parts.append(counts)
        row_paths[i] = int(counts.sum())
        if paths is not None and paths.n_paths:
            end_parts.append(paths.end.astype(np.int32))
            m1_parts.append(paths.atom_type_mid1.astype(np.int8))
            m2_parts.append(paths.atom_type_mid2.astype(np.int8))
            b1_parts.append(paths.bond_type_1.astype(np.int8))
            b2_parts.append(paths.bond_type_2.astype(np.int8))
            b3_parts.append(paths.bond_type_3.astype(np.int8))
            d_parts.append(paths.dist.astype(np.int8))
        if log_every and (i + 1) % int(log_every) == 0:
            log(
                f"[s3-cache] {i + 1}/{scope.size} rows, {int(row_paths.sum())} paths, "
                f"{time.perf_counter() - t0:.1f}s"
            )

    per_atom_counts = (
        np.concatenate(counts_parts) if counts_parts else np.zeros(0, dtype=np.int64)
    )
    if per_atom_counts.size != n_scope_atoms:
        raise RuntimeError("per-atom S3 count length != scope atom count")
    s3_atom_ptr = np.concatenate([[0], np.cumsum(per_atom_counts)]).astype(np.int64)

    def _stack(parts: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    arrays = {
        "s3_atom_ptr": s3_atom_ptr,
        "s3_end": _stack(end_parts, np.int32),
        "s3_mid1_type": _stack(m1_parts, np.int8),
        "s3_mid2_type": _stack(m2_parts, np.int8),
        "s3_bond1": _stack(b1_parts, np.int8),
        "s3_bond2": _stack(b2_parts, np.int8),
        "s3_bond3": _stack(b3_parts, np.int8),
        "s3_dist": _stack(d_parts, np.int8),
        "s3_row_paths": row_paths.astype(np.int64),
        "s3_row_atoms": row_mol_atoms.astype(np.int64),
        "rows": scope.astype(np.int64),
    }
    n_paths = int(s3_atom_ptr[-1])
    if n_paths != int(row_paths.sum()) or n_paths != int(arrays["s3_end"].size):
        raise RuntimeError("S3 cache total path count mismatch")
    if not np.all((arrays["s3_bond1"] >= 1) & (arrays["s3_bond1"] <= 5)):
        raise RuntimeError("S3 cache bond types outside the 1..5 vocabulary")
    if not np.all((arrays["s3_mid1_type"] >= 0) & (arrays["s3_mid1_type"] < 16)):
        raise RuntimeError("S3 cache mid atom types outside the 0..15 vocabulary")

    hashes: dict[str, str] = {}
    for name, arr in arrays.items():
        path = out_dir / f"{name}.npy"
        np.save(path, arr)
        hashes[name] = sha256_file(path)

    seconds = float(time.perf_counter() - t0)
    nbytes = int(sum((out_dir / f"{name}.npy").stat().st_size for name in arrays))
    meta = {
        "schema": SCHEMA_CACHE,
        "round_id": ROUND_ID,
        "created_utc": utcnow(),
        "scope": "official Train + Valid rows (sorted); official Test rows excluded",
        "n_rows": int(scope.size),
        "n_atoms": int(n_scope_atoms),
        "n_bonds": int(n_bonds_total),
        "n_paths": int(n_paths),
        "paths_per_row_mean": float(row_paths.mean()) if row_paths.size else 0.0,
        "dist_counts": {"1": int(dist_totals[0]), "2": int(dist_totals[1]), "3": int(dist_totals[2])},
        "bytes": nbytes,
        "build_seconds": seconds,
        "paths_per_second": float(n_paths / seconds) if seconds > 0 else 0.0,
        "rows_sha256": sha256_bytes(np.ascontiguousarray(scope, dtype=np.int64).tobytes()),
        "field_sha256": hashes,
        "feature_spec_hash": SOURCE["feature_spec_hash"],
        "split_partition_sha256": SOURCE["split_partition_sha256"],
        "cache_sha256": SOURCE["cache_sha256"],
        "labels_present": False,
        "test_rows_present": False,
        "definition": S3_DEFINITION,
    }
    write_json(out_dir / "meta.json", meta)
    log(
        f"[s3-cache] done rows={meta['n_rows']} atoms={meta['n_atoms']} paths={meta['n_paths']} "
        f"bytes={nbytes / 1e9:.3f}GB seconds={seconds:.1f}"
    )
    return meta


def load_s3_cache(path: Path | str = S3_CACHE_DIR) -> dict[str, Any]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA_CACHE:
        raise RuntimeError(f"S3 cache schema {meta.get('schema')} != {SCHEMA_CACHE}")
    if bool(meta.get("test_rows_present")) or bool(meta.get("labels_present")):
        raise RuntimeError("S3 cache is not label-free / Test-free")
    out: dict[str, Any] = {"meta": meta, "path": path}
    for name in S3_CACHE_FIELDS:
        out[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    row_index = np.full(N_MOLECULES, -1, dtype=np.int64)
    row_index[np.asarray(out["rows"], dtype=np.int64)] = np.arange(len(out["rows"]), dtype=np.int64)
    out["row_index"] = row_index
    out["row_atom_ptr"] = np.concatenate([[0], np.cumsum(np.asarray(out["s3_row_atoms"], dtype=np.int64))])
    if int(out["row_atom_ptr"][-1]) != int(out["s3_atom_ptr"].shape[0]) - 1:
        raise RuntimeError("S3 cache atom bookkeeping mismatch")
    return out


def collate_s3(s3: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    """Batch-local S3 tensors for the given cache rows (same order as ``rows``)."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    idx = np.asarray(s3["row_index"])[rows]
    if np.any(idx < 0):
        raise ValueError("row outside the S3 scope (official Test rows are never in scope)")
    atom_ptr = np.asarray(s3["row_atom_ptr"])
    a0 = atom_ptr[idx]
    a1 = atom_ptr[idx + 1]
    atom_counts = (a1 - a0).astype(np.int64)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if rows.size else np.zeros(0, np.int64)
    atom_idx = np.concatenate([np.arange(x, y, dtype=np.int64) for x, y in zip(a0.tolist(), a1.tolist())]) if rows.size else np.zeros(0, np.int64)
    s3_ptr = np.asarray(s3["s3_atom_ptr"])
    counts = (s3_ptr[atom_idx + 1] - s3_ptr[atom_idx]).astype(np.int64)
    atom_local = np.arange(atom_idx.size, dtype=np.int64)
    path_atom = np.repeat(atom_local, counts) if atom_idx.size else np.zeros(0, np.int64)
    atom_graph = np.repeat(np.arange(rows.size, dtype=np.int64), atom_counts) if rows.size else np.zeros(0, np.int64)
    path_graph = np.repeat(atom_graph, counts) if atom_idx.size else np.zeros(0, np.int64)
    path_idx = (
        np.concatenate([np.arange(s3_ptr[a], s3_ptr[a] + c, dtype=np.int64) for a, c in zip(atom_idx.tolist(), counts.tolist())])
        if atom_idx.size
        else np.zeros(0, np.int64)
    )
    end = np.asarray(s3["s3_end"])[path_idx].astype(np.int64) + atom_off[path_graph] if path_idx.size else np.zeros(0, np.int64)
    return {
        "s3_path_atom": path_atom,
        "s3_path_end": end,
        "s3_mid1_type": np.asarray(s3["s3_mid1_type"])[path_idx].astype(np.int64) if path_idx.size else np.zeros(0, np.int64),
        "s3_mid2_type": np.asarray(s3["s3_mid2_type"])[path_idx].astype(np.int64) if path_idx.size else np.zeros(0, np.int64),
        "s3_bond1": np.asarray(s3["s3_bond1"])[path_idx].astype(np.int64) if path_idx.size else np.zeros(0, np.int64),
        "s3_bond2": np.asarray(s3["s3_bond2"])[path_idx].astype(np.int64) if path_idx.size else np.zeros(0, np.int64),
        "s3_bond3": np.asarray(s3["s3_bond3"])[path_idx].astype(np.int64) if path_idx.size else np.zeros(0, np.int64),
        "s3_dist": np.asarray(s3["s3_dist"])[path_idx].astype(np.int64) if path_idx.size else np.zeros(0, np.int64),
        "atom_counts": atom_counts,
    }


def s3_torch_batch(s3_np: Mapping[str, np.ndarray]) -> dict[str, Any]:
    import torch

    out = {
        "s3_path_atom": torch.from_numpy(np.ascontiguousarray(s3_np["s3_path_atom"], dtype=np.int64)),
        "s3_path_end": torch.from_numpy(np.ascontiguousarray(s3_np["s3_path_end"], dtype=np.int64)),
        "s3_mid1_type": torch.from_numpy(np.ascontiguousarray(s3_np["s3_mid1_type"], dtype=np.int64)),
        "s3_mid2_type": torch.from_numpy(np.ascontiguousarray(s3_np["s3_mid2_type"], dtype=np.int64)),
        "s3_bond1": torch.from_numpy(np.ascontiguousarray(s3_np["s3_bond1"], dtype=np.int64)),
        "s3_bond2": torch.from_numpy(np.ascontiguousarray(s3_np["s3_bond2"], dtype=np.int64)),
        "s3_bond3": torch.from_numpy(np.ascontiguousarray(s3_np["s3_bond3"], dtype=np.int64)),
        "s3_dist": torch.from_numpy(np.ascontiguousarray(s3_np["s3_dist"], dtype=np.int64)),
    }
    return out


def collate_with_s3(task: Any, s3: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """The frozen ``pc_data.collate`` batch plus the S3 tensors (same rows)."""
    batch = dict(BC.PC.collate(task.cache, list(rows)))
    s3_np = collate_s3(s3, rows)
    if not np.array_equal(batch["atom_counts"], s3_np["atom_counts"]):
        raise RuntimeError("S3 atom counts disagree with the frozen collate")
    batch.update(s3_torch_batch(s3_np))
    return batch


def s3_cache_identity(path: Path | str | None = None) -> dict[str, Any]:
    """Provenance record of the S3 structural cache (never a label identity)."""
    base = Path(path) if path is not None else S3_CACHE_DIR
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
        "definition": meta.get("definition"),
        "labels_present": bool(meta.get("labels_present", True)),
        "test_rows_present": bool(meta.get("test_rows_present", True)),
    }


def load_task_with_s3(cache_path: Path | str | None = None) -> Any:
    """The frozen PCBA task plus its S3 cache, with the scope checked."""
    task = BC.SD.load_task()
    s3 = load_s3_cache(Path(cache_path) if cache_path else S3_CACHE_DIR)
    scope_rows = np.asarray(s3["rows"], dtype=np.int64)
    missing_train = np.setdiff1d(np.asarray(task.train_rows, np.int64), scope_rows)
    missing_valid = np.setdiff1d(np.asarray(task.valid_rows, np.int64), scope_rows)
    if missing_train.size or missing_valid.size:
        raise RuntimeError(
            f"S3 cache is missing split rows: train {missing_train.size}, valid {missing_valid.size}"
        )
    if np.intersect1d(scope_rows, np.asarray(task.test_rows, np.int64)).size:
        raise RuntimeError("S3 cache contains official Test rows")
    if int(np.sum(np.asarray(s3["s3_row_atoms"], dtype=np.int64))) != int(s3["meta"]["n_atoms"]):
        raise RuntimeError("S3 cache atom accounting drift")
    task.s3 = s3
    return task


#: Batch-composition helper: every S3 path must sit inside exactly one molecule.
def check_s3_batch_isolation(batch: Mapping[str, Any]) -> dict[str, Any]:

    path_atom = batch["s3_path_atom"]
    end = batch["s3_path_end"]
    atom_batch = batch["atom_batch"]
    if path_atom.numel() == 0:
        return {"n_paths": 0, "cross_molecule": 0, "isolated": True}
    cross = int((atom_batch[path_atom] != atom_batch[end]).sum().item())
    return {
        "n_paths": int(path_atom.numel()),
        "cross_molecule": cross,
        "isolated": cross == 0,
        "max_path_atom": int(path_atom.max().item()),
        "max_end": int(end.max().item()),
        "n_atoms": int(atom_batch.numel()),
    }


__all__ = [
    "ANALYSIS",
    "BASE_AP40",
    "BASE_BEST_AP",
    "BASE_BEST_EPOCH",
    "BASE_MODEL_KEY",
    "BASE_PARAMETERS",
    "BATCH_SIZE",
    "BC",
    "BS_ROUND",
    "BUDGET",
    "CODE_DIR",
    "DATASET",
    "DECISION_RULES",
    "EXPECTED_PARAMETERS",
    "F_ADDED",
    "FFN_FORMULA",
    "FFN_HIDDEN",
    "GATES",
    "INIT_RULE",
    "MODELS",
    "MODEL_ORDER",
    "N_MOLECULES",
    "N_TASKS",
    "N_TEST",
    "N_TRAIN",
    "N_VALID",
    "PARAMETER_GAP",
    "PARAMETER_GAP_FRACTION",
    "PARAMETER_GAP_MAX_FRACTION",
    "PRIVATE_GENERATOR_OFFSETS",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "S3_ADDED",
    "S3_BRANCH",
    "S3_CACHE_DIR",
    "S3_DIM",
    "S3_INPUT_DIM",
    "S3_OUT_WIDTH",
    "SAMPLE_SIZE",
    "SAMPLE_STRATA",
    "SCHEMA_CACHE",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_RUN",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "W2",
    "W2_CHECKPOINT_BEST",
    "W2_CHECKPOINT_EPOCH40",
    "W2_CURVE",
    "W2_PER_TASK",
    "W2_RUN_DIR",
    "W2_RUN_JSON",
    "W2_TRAIN_EVAL",
    "WIDTHS",
    "better_candidate",
    "build_s3_cache",
    "capture_rng_state",
    "check_s3_batch_isolation",
    "choose_ffn_hidden",
    "collate_s3",
    "collate_with_s3",
    "environment",
    "ffn_parameters",
    "git_commit",
    "git_state",
    "load_s3_cache",
    "load_task_with_s3",
    "log",
    "lr_audit",
    "parameter_count",
    "read_json",
    "recipe_for_model",
    "rng_state_hashes",
    "s3_cache_identity",
    "s3_scope_rows",
    "s3_torch_batch",
    "scheduled_lr",
    "seed_everything",
    "sha256_bytes",
    "sha256_file",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "write_json",
]
