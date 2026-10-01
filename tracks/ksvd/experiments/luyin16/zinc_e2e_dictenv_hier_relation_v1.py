"""E2E-DictEnv-Hier-Relation-v1 stage runner (ZINC, local CPU, single candidate).

Implements ``tracks/ksvd/notes/zinc_e2e_dictenv_hier_relation_v1_preregistration.md``.

prepare
    ``source_verify``      frozen upstream identity (joint scaler/caches, subspace,
                           env caches, no test env cache);
    ``node_input``         the 712-D node input (frozen joint 709 + train-only
                           scaled ``common1``/``size2``), cached as one array;
    ``relation_structure`` per-molecule unique physical edges (``u < v``), node
                           pointer, atom types, topology features, labels;
    ``align_check``        row/molecule/node alignment, raw recomputation spot
                           checks, raw-ZINC edge spot checks;
    ``dictionary_init``    fixed ``P``, spectral node/relation dictionary
                           initialisation, frozen four-block relation scaler.

screen
    ``correctness``  ``smoke``  ``train``  ``interventions``  ``analysis``

The official ZINC **test** split is never instantiated (there is no code path in
this module that can load it) and every payload re-asserts
``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_hier_relation_v1 as core
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_increment_v2 as _inc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_rolecorr_increment_v2 as incrun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_rolecorr_v1 as rcrun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = core.PROTOCOL_VERSION
ROUND = core.ROUND
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_hier_relation_v1"
CACHE_DIR = RESULTS_DIR / "cache"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
SOURCE_DIR = TRACK_ROOT / "results/e2e_dictenv_joint709_absolute_v1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_git_commit = v0run._git_commit
_sha256_array = incrun._sha256_array
_sha256_file = incrun._sha256_file

THREADS = 8
SEED = int(core.SEED)
TRAIN_EPOCHS = int(core.TRAIN_EPOCHS)
SMOKE_EPOCHS = int(core.SMOKE_EPOCHS)
SMOKE_TRAIN_SUBSET = int(core.SMOKE_TRAIN_SUBSET)
SMOKE_VALID_SUBSET = int(core.SMOKE_VALID_SUBSET)
BATCH_SIZE = int(core.BATCH_SIZE)
RESUME_EVERY = 10
TAG = "HIERREL-seed0"
CANDIDATE = "HIER-RELATION"

#: frozen identity of the read-only upstream objects (see pre-registration §2)
EXPECTED = {
    "subspace": {
        "path": "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json",
        "sha256": "36636ce92836bdb8d023cc91b3f532f8d4c46a57d457514c92d90c68028f6c24",
        "rms": 5.082852828320509,
    },
    "joint_scaler": {
        "path": "tracks/ksvd/results/e2e_dictenv_joint709_absolute_v1/joint_standardizers.json",
        "sha256": "0ea418fae3c033e192def98c46645e01c41308472f7be84b605d5c4a1e193060",
    },
    # NOTE (provenance nuance found this round): the upstream ``sha256_f32``
    # fields in ``cache_meta_joint_{split}.json`` are the hashes of the float64
    # array *before* the float32 serialisation (``build_joint_cache`` hashes the
    # float64 ``scaled`` array, then saves ``float32``).  They therefore do not
    # reproduce as a hash of the shipped file; they are recorded here as
    # ``sha256_f64_recorded`` and the file content is additionally verified by a
    # float64 -> float32 recomputation on a molecule sample.
    "joint_cache": {
        "train": {
            "rows": 231664,
            "dim": 709,
            "sha256_f64_recorded": "2f8b0ea89da78225edfd3cbb1465c949bd0b7c730a81b01012b2cd579ebc2561",
        },
        "valid": {
            "rows": 23083,
            "dim": 709,
            "sha256_f64_recorded": "17962654b3d0271ea1068391db5d817d5298584f337aaa24ef84654dd65add29",
        },
    },
    "sample_recompute_tolerance": 1e-5,
    "corr_cache": {
        "train": {
            "n_patches": 231664,
            "node_sha256": "6b328d7ae859f420fbd0eb08369546dee1e6c875aec6ece4afc31a5abbae287f",
            "edge_sha256": "d710705e6da09741cd557493c750ba48ec3da54736a1f3f3bfbb796fd5f3e74d",
        },
        "valid": {
            "n_patches": 23083,
            "node_sha256": "15da71b968bd7fa9525e16eafd6b6b2ee8543852be44b2bfedc438777ce0c53b",
            "edge_sha256": "2388c80b30b8a3bc5bc74efb66a27f0a972a8b074aa2e93d1eee3d5e04decd10",
        },
    },
    "splits": {"train": 10000, "valid": 1000},
}

#: fixed invariance tolerance for relabel / batch-composition checks
INVARIANCE_TOL = 1e-5
#: expected prediction shift floor used by the intervention gate
INTERVENTION_RMS_FLOOR = 1e-4

_DEVICE = torch.device("cpu")
_STRUCT_MEMORY: dict[str, core.SplitTensors] = {}


def configure(
    *, epochs: int | None = None, threads: int | None = None, device: Any = None
) -> dict[str, Any]:
    """Fix the run knobs; every scientific constant stays frozen."""
    global TRAIN_EPOCHS, THREADS
    if epochs is not None and int(epochs) > 0:
        TRAIN_EPOCHS = int(epochs)
    if threads is not None and int(threads) > 0:
        THREADS = int(threads)
    if device is not None and str(device) != "cpu":
        raise RuntimeError(
            f"this round is a LOCAL CPU regime only; got device={device!r}"
        )
    torch.set_num_threads(int(THREADS))
    return {"epochs": int(TRAIN_EPOCHS), "threads": int(THREADS), "device": "cpu"}


def _device_payload() -> dict[str, Any]:
    return {
        "device": "cpu",
        "device_type": "cpu",
        "torch_version": torch.__version__,
        "torch_threads": int(THREADS),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "official_test_loaded": False,
    }


def _blocker(payload: Mapping[str, Any]) -> None:
    if bool(payload.get("official_test_loaded", False)):
        raise RuntimeError("official test must never be loaded in this round")
    payload["official_test_loaded"] = False


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CACHE_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _env_blob(split: str) -> dict[str, Any]:
    path = TRACK_ROOT / f"results/e2e_dictenv_p1/cache/env_{split}.pt"
    if not path.exists():
        raise RuntimeError(
            f"env cache missing: {path}; run the frozen e2e_dictenv_p1 env stage first"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _joint_cache_path(split: str) -> Path:
    name = "cache/corr_joint_train.pt" if split == "train" else "cache/corr_joint_valid.pt"
    return SOURCE_DIR / name


def _subspace_payload() -> dict[str, Any]:
    path = REPO_ROOT / EXPECTED["subspace"]["path"]
    if not path.exists():
        raise RuntimeError(f"common subspace missing: {path}")
    sha = _sha256_file(path)
    if sha != EXPECTED["subspace"]["sha256"]:
        raise RuntimeError(
            f"common subspace sha256 mismatch: {sha} != {EXPECTED['subspace']['sha256']}"
        )
    payload = _read_json(path)
    entry = payload["q1"]
    components = np.asarray(entry["components"], dtype=np.float64)
    rms = np.asarray(entry["rms"], dtype=np.float64)
    if components.shape != (65, 1) or rms.shape != (1,):
        raise RuntimeError(f"unexpected subspace shapes {components.shape} {rms.shape}")
    if abs(float(rms[0]) - float(EXPECTED["subspace"]["rms"])) > 1e-9:
        raise RuntimeError("subspace rms does not match the frozen value")
    return {
        "sha256": sha,
        "components": components,
        "rms": rms,
        "kind": str(entry["kind"]),
        "path": str(path.relative_to(REPO_ROOT)),
    }


def _load_joint_cached(split: str) -> np.ndarray:
    path = _joint_cache_path(split)
    if not path.exists():
        raise RuntimeError(
            f"frozen joint cache missing: {path}; it must be re-used, not refit "
            "(do NOT re-run the expensive K-SVD/PCA prepare stages)"
        )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    values = blob["joint"].numpy()
    expected = EXPECTED["joint_cache"][split]
    if int(values.shape[0]) != int(expected["rows"]) or int(values.shape[1]) != core.JOINT_DIM:
        raise RuntimeError(f"{split}: joint cache shape {values.shape} unexpected")
    return values


def _joint_sample_recompute(split: str, n_molecules: int = 8) -> dict[str, Any]:
    """Re-derive the frozen 709-D cache rows from the raw sources (sample).

    This is the content identity check that survives the float64 -> float32
    serialisation: recompute ``apply_joint_scaler(struct_residual, Sem108, corr)``
    in float64 for a molecule prefix and compare with the shipped cache rows.
    """
    scaler = incrun.load_joint_scaler(SOURCE_DIR)
    subspace = _subspace_payload()
    components = torch.as_tensor(subspace["components"], dtype=torch.float32)
    encoded = sdp.load_encoded()[0] if split == "train" else sdp.load_encoded()[1]
    blob = _env_blob(split)
    node_sizes = np.asarray([int(v) for v in blob["node_sizes"].tolist()], dtype=np.int64)
    node_ptr = np.concatenate([[0], np.cumsum(node_sizes)])
    phi = blob["phi"]
    raw = rcrun.load_raw_corr(split)
    cache = torch.load(_joint_cache_path(split), map_location="cpu", weights_only=True)["joint"].numpy()
    max_abs = 0.0
    rows = 0
    for index in range(int(n_molecules)):
        lo, hi = int(node_ptr[index]), int(node_ptr[index + 1])
        block = phi[lo:hi]
        residual = (block - (block @ components) @ components.t()).numpy().astype(np.float64)
        sem = encoded[index].patch_cont[:, :108].numpy().astype(np.float64)
        corr = np.concatenate([raw["node"][lo:hi], raw["edge"][lo:hi]], axis=1).astype(np.float64)
        if residual.shape[0] != sem.shape[0] or residual.shape[0] != corr.shape[0]:
            raise RuntimeError(f"{split}[{index}]: block row mismatch in the sample recompute")
        recomputed = _inc.apply_joint_scaler(scaler, residual, sem, corr)
        max_abs = max(max_abs, float(np.abs(recomputed - cache[lo:hi]).max()))
        rows += int(residual.shape[0])
    return {
        "split": split,
        "molecules": int(n_molecules),
        "rows": int(rows),
        "max_abs_error_vs_cache": float(max_abs),
        "tolerance": float(EXPECTED["sample_recompute_tolerance"]),
        "passed": bool(max_abs <= float(EXPECTED["sample_recompute_tolerance"])),
    }


# ---------------------------------------------------------------------------
# prepare: source identity
# ---------------------------------------------------------------------------


def stage_source_verify(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "source_verify.json"
    if path.exists() and not force:
        return _read_json(path)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": ROUND,
        "git_commit": _git_commit(),
        **_device_payload(),
    }
    subspace = _subspace_payload()
    payload["common_subspace"] = {
        "path": subspace["path"],
        "sha256": subspace["sha256"],
        "kind": subspace["kind"],
        "rms": float(subspace["rms"][0]),
    }
    scaler_path = REPO_ROOT / EXPECTED["joint_scaler"]["path"]
    scaler_sha = _sha256_file(scaler_path) if scaler_path.exists() else None
    payload["joint_scaler"] = {
        "path": str(scaler_path.relative_to(REPO_ROOT)),
        "sha256": scaler_sha,
        "matches_frozen": bool(scaler_sha == EXPECTED["joint_scaler"]["sha256"]),
    }
    payload["joint_caches"] = {}
    for split in ("train", "valid"):
        cache_path = _joint_cache_path(split)
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        values = blob["joint"].numpy()
        payload["joint_caches"][split] = {
            "path": str(cache_path.relative_to(REPO_ROOT)),
            "rows": int(values.shape[0]),
            "dim": int(values.shape[1]),
            "sha256_f32_shipped": _sha256_array(values),
            "sha256_f64_recorded_upstream": str(
                EXPECTED["joint_cache"][split]["sha256_f64_recorded"]
            ),
            "matches_frozen": bool(
                int(values.shape[0]) == int(EXPECTED["joint_cache"][split]["rows"])
                and int(values.shape[1]) == int(EXPECTED["joint_cache"][split]["dim"])
            ),
        }
        del blob, values
    corr = {split: rcrun.load_raw_corr(split) for split in ("train", "valid")}
    payload["corr_caches"] = {}
    for split in ("train", "valid"):
        entry = corr[split]
        payload["corr_caches"][split] = {
            "n_patches": int(entry["node"].shape[0]),
            "node_sha256": _sha256_array(entry["node"]),
            "edge_sha256": _sha256_array(entry["edge"]),
            "matches_frozen": bool(
                int(entry["node"].shape[0]) == int(EXPECTED["corr_cache"][split]["n_patches"])
                and _sha256_array(entry["node"]) == str(EXPECTED["corr_cache"][split]["node_sha256"])
                and _sha256_array(entry["edge"]) == str(EXPECTED["corr_cache"][split]["edge_sha256"])
            ),
        }
    payload["env_caches"] = {}
    for split in ("train", "valid"):
        blob = _env_blob(split)
        node_sizes = [int(v) for v in blob["node_sizes"].tolist()]
        payload["env_caches"][split] = {
            "n_molecules": int(len(node_sizes)),
            "n_nodes": int(sum(node_sizes)),
            "n_bond_occurrences": int(sum(int(v) for v in blob["bond_sizes"].tolist())),
            "phi_shape": list(blob["phi"].shape),
            "matches_expected_molecules": bool(
                len(node_sizes) == int(EXPECTED["splits"][split])
            ),
        }
        del blob
    payload["joint_sample_recompute"] = {
        split: _joint_sample_recompute(split) for split in ("train", "valid")
    }
    payload["test_env_cache_present"] = bool(
        (TRACK_ROOT / "results/e2e_dictenv_p1/cache/env_test.pt").exists()
    )
    payload["test_split_touched"] = False
    payload["passed"] = bool(
        payload["joint_scaler"]["matches_frozen"]
        and all(row["passed"] for row in payload["joint_sample_recompute"].values())
        and all(row["matches_frozen"] for row in payload["joint_caches"].values())
        and all(row["matches_frozen"] for row in payload["corr_caches"].values())
        and all(row["matches_expected_molecules"] for row in payload["env_caches"].values())
        and all(row["n_nodes"] == int(EXPECTED["joint_cache"][s]["rows"])
                for s, row in payload["env_caches"].items())
    )
    _blocker(payload)
    _write_json(path, payload)
    print(f"[source_verify] passed={payload['passed']}", flush=True)
    if not payload["passed"]:
        raise RuntimeError(f"frozen-source verification failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# prepare: node input (712-D) and relation structure
# ---------------------------------------------------------------------------


def _extra_raw_blocks(split: str) -> np.ndarray:
    """Raw ``[common1 ; size2]`` per node, in the frozen cache row order."""
    blob = _env_blob(split)
    subspace = _subspace_payload()
    phi = blob["phi"].numpy().astype(np.float64)
    anchor = blob["anchor"].numpy()
    node_sizes = [int(v) for v in blob["node_sizes"].tolist()]
    if int(phi.shape[0]) != int(sum(node_sizes)):
        raise RuntimeError(f"{split}: env phi rows != sum(node_sizes)")
    if tuple(anchor.shape) != (int(phi.shape[0]), 62):
        raise RuntimeError(f"{split}: unexpected anchor shape {tuple(anchor.shape)}")
    common = (phi @ subspace["components"]) / subspace["rms"][None, :]
    size = anchor[:, 60:62].astype(np.float64)
    out = np.concatenate([common, size], axis=1)
    if tuple(out.shape) != (int(phi.shape[0]), core.EXTRA_SIZE_DIM):
        raise RuntimeError(f"{split}: unexpected extra block shape {out.shape}")
    return out


def stage_node_input(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    scaler_path = RESULTS_DIR / "hier_extra_scaler.json"
    meta_path = RESULTS_DIR / "hier_node_input.json"
    inputs = {split: CACHE_DIR / f"hier_node_input_{split}.pt" for split in ("train", "valid")}
    if (
        not force
        and meta_path.exists()
        and scaler_path.exists()
        and all(path.exists() for path in inputs.values())
    ):
        payload = _read_json(meta_path)
        if payload.get("extra_scaler_sha256") == _sha256_file(scaler_path):
            return payload
    if not scaler_path.exists() or force:
        train_extra = _extra_raw_blocks("train")
        scaler = core.fit_frozen_scaler(
            "node_extra",
            {"common": core.EXTRA_SLICES["common"], "size": core.EXTRA_SLICES["size"]},
            train_extra,
        )
        payload = scaler.to_json()
        payload.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": _git_commit(),
                "train_block_energy": core.scaled_block_energy(train_extra, scaler),
                "n_fit_rows": int(train_extra.shape[0]),
                **_device_payload(),
            }
        )
        _blocker(payload)
        _write_json(scaler_path, payload)
        del train_extra
    scaler = core.FrozenScaler.from_json(_read_json(scaler_path))
    gain = scaler.gain.astype(np.float32)
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "extra_scaler_sha256": _sha256_file(scaler_path),
        "extra_scaler": scaler.to_json(),
        "splits": {},
        **_device_payload(),
    }
    for split in ("train", "valid"):
        joint = _load_joint_cached(split)
        extra = _extra_raw_blocks(split)
        if int(extra.shape[0]) != int(joint.shape[0]):
            raise RuntimeError(f"{split}: extra rows {extra.shape[0]} != joint rows {joint.shape[0]}")
        scaled_extra = (extra * gain[None, :]).astype(np.float32)
        x = np.empty((joint.shape[0], core.D_X), dtype=np.float32)
        x[:, : core.JOINT_DIM] = joint
        x[:, core.JOINT_DIM :] = scaled_extra
        if _sha256_array(x[:, : core.JOINT_DIM]) != _sha256_array(joint):
            raise RuntimeError(f"{split}: node input joint block is not bit-identical to the cache")
        if _sha256_array(x[:, : core.JOINT_DIM]) != str(
            _read_json(RESULTS_DIR / "source_verify.json")["joint_caches"][split]["sha256_f32_shipped"]
        ):
            raise RuntimeError(f"{split}: joint block hash does not match source_verify")
        torch.save({"x": torch.as_tensor(x, dtype=torch.float32)}, inputs[split])
        report["splits"][split] = {
            "path": str(inputs[split].relative_to(REPO_ROOT)),
            "rows": int(x.shape[0]),
            "dim": int(x.shape[1]),
            "sha256_f32": _sha256_array(x),
            "joint_block_sha256_f32": _sha256_array(x[:, : core.JOINT_DIM]),
            "extra_scaled_block_energy": {
                block: float(
                    np.mean(np.sum((extra * gain[None, :])[:, lo:hi] ** 2, axis=1))
                )
                for block, (lo, hi) in {
                    "common": core.EXTRA_SLICES["common"],
                    "size": core.EXTRA_SLICES["size"],
                }.items()
            },
            "extra_raw_rms": [float(v) for v in np.sqrt(np.mean(extra ** 2, axis=0)).tolist()],
        }
        del joint, extra, x
    _blocker(report)
    _write_json(meta_path, report)
    print(
        "[node_input] train_energy="
        f"{report['splits']['train']['extra_scaled_block_energy']} "
        f"valid_energy={report['splits']['valid']['extra_scaled_block_energy']}",
        flush=True,
    )
    return report


def _dedup_edges(
    bond_u: np.ndarray, bond_v: np.ndarray, bond_type: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Unique physical edges (``u < v``) of one molecule from occurrence records.

    ``env_bond_u/v/type`` repeat the same physical key once per rooted patch that
    contains it; the deduplicated key set is the molecule's physical edge set.
    """
    u = np.asarray(bond_u, dtype=np.int64)
    v = np.asarray(bond_v, dtype=np.int64)
    t = np.asarray(bond_type, dtype=np.int64)
    if u.size != v.size or u.size != t.size:
        raise RuntimeError("bond occurrence arrays disagree")
    if u.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            {"occurrences": 0, "unique": 0, "type_conflicts": 0},
        )
    out_of_range = int(np.sum((u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes)))
    if out_of_range:
        raise RuntimeError(f"bond occurrence endpoint outside [0,{n_nodes})")
    if int(np.sum(u == v)):
        raise RuntimeError("bond occurrence is a self loop")
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    packed = lo * np.int64(n_nodes) + hi
    order = np.argsort(packed, kind="stable")
    packed_sorted = packed[order]
    types_sorted = t[order]
    first = np.ones(packed_sorted.shape[0], dtype=bool)
    first[1:] = packed_sorted[1:] != packed_sorted[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, packed_sorted.shape[0]))
    conflicts = 0
    for start, count in zip(starts.tolist(), counts.tolist()):
        if int(np.ptp(types_sorted[start : start + count])) != 0:
            conflicts += 1
    if conflicts:
        raise RuntimeError(f"{conflicts} duplicated keys with inconsistent bond types")
    unique_packed = packed_sorted[starts]
    unique_type = types_sorted[starts]
    return (
        unique_packed // np.int64(n_nodes),
        unique_packed % np.int64(n_nodes),
        unique_type,
        {"occurrences": int(u.size), "unique": int(unique_packed.size), "type_conflicts": 0},
    )


def stage_relation_structure(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    meta_path = RESULTS_DIR / "hier_structure.json"
    paths = {split: CACHE_DIR / f"hier_struct_{split}.pt" for split in ("train", "valid")}
    if not force and meta_path.exists() and all(path.exists() for path in paths.values()):
        return _read_json(meta_path)
    _enc_train, _enc_valid, _audit = sdp.load_encoded()
    encoded = {"train": _enc_train, "valid": _enc_valid}
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "splits": {},
        **_device_payload(),
    }
    for split in ("train", "valid"):
        blob = _env_blob(split)
        node_sizes = [int(v) for v in blob["node_sizes"].tolist()]
        bond_sizes = [int(v) for v in blob["bond_sizes"].tolist()]
        bond_u = blob["bond_u"].numpy()
        bond_v = blob["bond_v"].numpy()
        bond_type = blob["bond_type"].numpy()
        atom = blob["atom"].numpy().astype(np.int64)
        data = encoded[split]
        if not len(data) == len(node_sizes):
            raise RuntimeError(f"{split}: encoded {len(data)} != env {len(node_sizes)}")
        node_ptr = [0]
        edge_ptr = [0]
        edge_u: list[np.ndarray] = []
        edge_v: list[np.ndarray] = []
        edge_t: list[np.ndarray] = []
        topo = np.empty((len(data), core.TOPO_IN), dtype=np.float32)
        y = np.empty((len(data),), dtype=np.float32)
        node_offset = 0
        bond_offset = 0
        occurrence_total = 0
        unique_total = 0
        for index, item in enumerate(data):
            n = int(node_sizes[index])
            if int(item.num_nodes) != n:
                raise RuntimeError(f"{split}[{index}]: encoded nodes != env node size")
            m = int(bond_sizes[index])
            u, v, bond_class, stats = _dedup_edges(
                bond_u[bond_offset : bond_offset + m],
                bond_v[bond_offset : bond_offset + m],
                bond_type[bond_offset : bond_offset + m],
                n,
            )
            edge_u.append(u + node_offset)
            edge_v.append(v + node_offset)
            edge_t.append(bond_class)
            occurrence_total += int(stats["occurrences"])
            unique_total += int(stats["unique"])
            topo[index] = item.topology_features.numpy().reshape(-1)[: core.TOPO_IN]
            y[index] = float(item.y.reshape(-1)[0])
            node_ptr.append(node_offset + n)
            edge_ptr.append(edge_ptr[-1] + int(u.size))
            node_offset += n
            bond_offset += m
        # atom types are already in the frozen global node order
        if int(atom.shape[0]) != node_offset:
            raise RuntimeError(f"{split}: atom rows != nodes")
        if int(np.min(atom)) < 0 or int(np.max(atom)) >= core.ATOM_CATEGORIES:
            raise RuntimeError(f"{split}: atom category out of range")
        edge_index = np.stack(
            [np.concatenate(edge_u), np.concatenate(edge_v)], axis=0
        ).astype(np.int64)
        edge_type = np.concatenate(edge_t).astype(np.int64)
        if bool((edge_index[0] >= edge_index[1]).any()):
            raise RuntimeError(f"{split}: edges must be stored with u < v")
        payload = {
            "node_ptr": torch.as_tensor(node_ptr, dtype=torch.long),
            "atom_type": torch.as_tensor(atom, dtype=torch.long),
            "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
            "edge_type": torch.as_tensor(edge_type, dtype=torch.long),
            "edge_ptr": torch.as_tensor(edge_ptr, dtype=torch.long),
            "topology": torch.as_tensor(topo, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "mol_id": torch.arange(len(data), dtype=torch.long),
        }
        torch.save(payload, paths[split])
        report["splits"][split] = {
            "path": str(paths[split].relative_to(REPO_ROOT)),
            "n_molecules": int(len(data)),
            "n_nodes": int(node_offset),
            "n_edges": int(edge_type.shape[0]),
            "bond_occurrences": int(occurrence_total),
            "edge_occurrence_ratio": float(occurrence_total / max(unique_total, 1)),
            "edge_index_sha256": _sha256_array(edge_index),
            "edge_type_sha256": _sha256_array(edge_type),
            "node_ptr_sha256": _sha256_array(np.asarray(node_ptr, dtype=np.int64)),
            "atom_type_sha256": _sha256_array(atom),
            "y_sha256": _sha256_array(y),
            "min_edges": int(
                np.min(np.diff(np.asarray(edge_ptr, dtype=np.int64)))
            ),
        }
        del blob, data
    _blocker(report)
    _write_json(meta_path, report)
    print(
        f"[structure] train_edges={report['splits']['train']['n_edges']} "
        f"valid_edges={report['splits']['valid']['n_edges']}",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# prepare: alignment / raw spot checks
# ---------------------------------------------------------------------------


def load_struct(split: str, force: bool = False) -> core.SplitTensors:
    key = f"{split}:{force}"
    if key in _STRUCT_MEMORY:
        return _STRUCT_MEMORY[key]
    x_path = CACHE_DIR / f"hier_node_input_{split}.pt"
    s_path = CACHE_DIR / f"hier_struct_{split}.pt"
    if not (x_path.exists() and s_path.exists()):
        raise RuntimeError("node-input/structure caches missing; run the prepare stages first")
    x = torch.load(x_path, map_location="cpu", weights_only=True)["x"]
    payload = torch.load(s_path, map_location="cpu", weights_only=False)
    meta = _read_json(RESULTS_DIR / "hier_structure.json")["splits"][split]
    if _sha256_array(payload["edge_index"].numpy()) != str(meta["edge_index_sha256"]):
        raise RuntimeError(f"{split}: structure cache checksum drift")
    tensors = core.SplitTensors(
        x=x,
        node_ptr=payload["node_ptr"],
        atom_type=payload["atom_type"],
        edge_index=payload["edge_index"],
        edge_type=payload["edge_type"],
        edge_ptr=payload["edge_ptr"],
        topology=payload["topology"],
        y=payload["y"],
        mol_id=payload["mol_id"],
        meta=meta,
    )
    if int(tensors.node_ptr.numel()) != tensors.n_graphs + 1:
        raise RuntimeError(f"{split}: node_ptr length mismatch")
    if int(tensors.edge_ptr.numel()) != tensors.n_graphs + 1:
        raise RuntimeError(f"{split}: edge_ptr length mismatch")
    if int(tensors.node_ptr[-1]) != tensors.n_nodes or int(tensors.edge_ptr[-1]) != tensors.n_edges:
        raise RuntimeError(f"{split}: pointer tails disagree with the arrays")
    _STRUCT_MEMORY[key] = tensors
    return tensors


def _raw_zinc_edges(split: str, n_molecules: int) -> dict[str, Any]:
    """Spot check the recovered unique edge set against raw ZINC."""
    dataset_split = "val" if split == "valid" else "train"
    raw = zlr._load_zinc(ZINC_ROOT, dataset_split)
    struct = load_struct(split)
    mismatches: list[dict[str, Any]] = []
    checked_edges = 0
    for index in range(int(n_molecules)):
        molecule = raw[index]
        graph, _node_types, edge_types = zlr._data_to_graph(molecule)
        n = int(graph.n)
        expected = set()
        expected_types = {}
        for a, b in graph.edges():
            key = (min(int(a), int(b)), max(int(a), int(b)))
            expected.add(key)
            expected_types[key] = int(edge_types[graph.edge_key(int(a), int(b))])
        lo = int(struct.node_ptr[index])
        hi = int(struct.node_ptr[index + 1])
        if n != (hi - lo):
            mismatches.append({"molecule": index, "reason": "node_count", "raw": n, "cache": hi - lo})
            continue
        e_lo = int(struct.edge_ptr[index])
        e_hi = int(struct.edge_ptr[index + 1])
        got_u = struct.edge_index[0, e_lo:e_hi].numpy() - lo
        got_v = struct.edge_index[1, e_lo:e_hi].numpy() - lo
        got_t = struct.edge_type[e_lo:e_hi].numpy()
        got = {key: int(t) for key, t in zip(zip(got_u.tolist(), got_v.tolist()), got_t.tolist())}
        checked_edges += len(expected)
        if set(got) != expected:
            mismatches.append(
                {
                    "molecule": index,
                    "reason": "edge_set",
                    "missing": sorted(expected - set(got))[:4],
                    "extra": sorted(set(got) - expected)[:4],
                }
            )
        elif got != expected_types:
            bad = [k for k in expected if got[k] != expected_types[k]][:4]
            mismatches.append({"molecule": index, "reason": "bond_type", "keys": bad})
    return {
        "split": split,
        "molecules_checked": int(n_molecules),
        "edges_checked": int(checked_edges),
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches[:5],
        "passed": bool(not mismatches),
    }


def stage_align_check(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "align_check.json"
    if path.exists() and not force:
        return _read_json(path)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "splits": {},
        **_device_payload(),
    }
    for split in ("train", "valid"):
        blob = _env_blob(split)
        node_sizes = np.asarray([int(v) for v in blob["node_sizes"].tolist()], dtype=np.int64)
        struct = load_struct(split)
        encoded = sdp.load_encoded()[0] if split == "train" else sdp.load_encoded()[1]
        node_ptr = struct.node_ptr.numpy()
        per_molecule_ok = True
        for index, item in enumerate(encoded):
            if int(item.num_nodes) != int(node_sizes[index]):
                per_molecule_ok = False
                break
        rows = int(struct.n_nodes)
        entry: dict[str, Any] = {
            "env_molecules": int(node_sizes.shape[0]),
            "x_rows": rows,
            "sum_node_sizes": int(node_sizes.sum()),
            "node_ptr_matches_env": bool(
                np.array_equal(node_ptr, np.concatenate([[0], np.cumsum(node_sizes)]))
            ),
            "encoded_num_nodes_matches_env": bool(per_molecule_ok),
            "joint_block_sha256": _sha256_array(struct.x[:, : core.JOINT_DIM].numpy()),
            "matches_frozen_joint": bool(
                _sha256_array(struct.x[:, : core.JOINT_DIM].numpy())
                == str(
                    _read_json(RESULTS_DIR / "source_verify.json")["joint_caches"][split][
                        "sha256_f32_shipped"
                    ]
                )
            ),
            "x_sha256": _sha256_array(struct.x.numpy()),
            "x_matches_node_input_stage": bool(
                _sha256_array(struct.x.numpy())
                == str(
                    _read_json(RESULTS_DIR / "hier_node_input.json")["splits"][split][
                        "sha256_f32"
                    ]
                )
            ),
            "y_order_matches_raw": None,
            "extra_recompute_max_abs": None,
            "raw_zinc_edges": None,
        }
        # (a) molecule-order / label check against raw ZINC for a prefix
        dataset_split = "val" if split == "valid" else "train"
        raw = zlr._load_zinc(ZINC_ROOT, dataset_split)
        probe = min(8, int(struct.n_graphs))
        y_raw = np.asarray(
            [float(np.asarray(raw[i].y).reshape(-1)[0]) for i in range(probe)], dtype=np.float32
        )
        entry["y_order_matches_raw"] = bool(
            np.allclose(y_raw, struct.y[:probe].numpy(), atol=0.0, rtol=0.0)
        )
        # (b) recompute the appended 3 columns from the raw env tensors
        subspace = _subspace_payload()
        scaler = core.FrozenScaler.from_json(
            _read_json(RESULTS_DIR / "hier_extra_scaler.json")
        )
        gain = scaler.gain
        phi = blob["phi"].numpy()
        anchor = blob["anchor"].numpy()
        max_abs = 0.0
        for index in range(probe):
            lo = int(node_ptr[index])
            hi = int(node_ptr[index + 1])
            common = (phi[lo:hi].astype(np.float64) @ subspace["components"]) / subspace["rms"][
                None, :
            ]
            size = anchor[lo:hi, 60:62].astype(np.float64)
            recomputed = np.concatenate([common, size], axis=1) * gain[None, :]
            max_abs = max(
                max_abs,
                float(np.abs(recomputed - struct.x[lo:hi, core.JOINT_DIM :].numpy()).max()),
            )
        entry["extra_recompute_max_abs"] = float(max_abs)
        entry["extra_recompute_passed"] = bool(max_abs <= 1e-6)
        # (c) edge set spot check against raw ZINC
        entry["raw_zinc_edges"] = _raw_zinc_edges(split, 32 if split == "train" else 16)
        entry["passed"] = bool(
            entry["node_ptr_matches_env"]
            and entry["encoded_num_nodes_matches_env"]
            and entry["matches_frozen_joint"]
            and entry["x_matches_node_input_stage"]
            and entry["y_order_matches_raw"]
            and entry["extra_recompute_passed"]
            and entry["raw_zinc_edges"]["passed"]
            and rows == int(node_sizes.sum())
        )
        payload["splits"][split] = entry
        print(f"[align_check:{split}] passed={entry['passed']}", flush=True)
    payload["passed"] = all(row["passed"] for row in payload["splits"].values())
    _blocker(payload)
    _write_json(path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"alignment verification failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# prepare: fixed projection, scalers and dictionary initialisation
# ---------------------------------------------------------------------------


def _chunk_slots(n_graphs: int, chunk: int) -> list[np.ndarray]:
    return [
        part.astype(np.int64)
        for part in np.array_split(np.arange(int(n_graphs), dtype=np.int64), max(1, math.ceil(n_graphs / chunk)))
        if len(part)
    ]


def _initial_node_codes(
    struct: core.SplitTensors, dbar_node: torch.Tensor, chunk_graphs: int = 512
):
    """Yield per-chunk ``(batch, z, mu, delta)`` under a fixed node dictionary."""
    for slots in _chunk_slots(struct.n_graphs, int(chunk_graphs)):
        batch = struct.batch(slots)
        with torch.no_grad():
            z = batch.x @ dbar_node
            mu, delta = core.shared_and_deviation(
                z, batch.graph_id, batch.atom_type, batch.n_graphs
            )
        yield batch, z, mu, delta


def _relation_stats_pass(
    struct: core.SplitTensors,
    dbar_node: torch.Tensor,
    projection: torch.Tensor,
    sample_index: np.ndarray | None,
    chunk_graphs: int = 512,
) -> dict[str, Any]:
    """One chunked train pass: per-column sumsq and (optionally) sampled rows."""
    sumsq = np.zeros(core.REL_INPUT_DIM, dtype=np.float64)
    count = 0
    sampled: np.ndarray | None = (
        None if sample_index is None else np.zeros((sample_index.shape[0], core.REL_INPUT_DIM), dtype=np.float32)
    )
    if sample_index is not None:
        position = np.full(int(struct.n_edges), -1, dtype=np.int64)
        position[sample_index] = np.arange(sample_index.shape[0], dtype=np.int64)
        selected = np.zeros(int(struct.n_edges), dtype=bool)
        selected[sample_index] = True
    for slots in _chunk_slots(struct.n_graphs, int(chunk_graphs)):
        batch = struct.batch(slots)
        with torch.no_grad():
            z = batch.x @ dbar_node
            mu, delta = core.shared_and_deviation(
                z, batch.graph_id, batch.atom_type, batch.n_graphs
            )
            m = mu @ projection
            d = delta @ projection
            u, v = batch.edge_index[0], batch.edge_index[1]
            r = core.relation_objects(
                m.index_select(0, u),
                d.index_select(0, u),
                m.index_select(0, v),
                d.index_select(0, v),
                batch.edge_type,
            )
        if r.shape[0]:
            sumsq += (r.double().pow(2).sum(dim=0)).numpy()
            count += int(r.shape[0])
            if sample_index is not None:
                e_lo = int(struct.edge_ptr[slots[0]])
                e_hi = int(struct.edge_ptr[slots[-1] + 1])
                local = np.flatnonzero(selected[e_lo:e_hi])
                if local.size:
                    sampled[position[e_lo + local]] = r.numpy()[local]
    return {"sumsq": sumsq, "count": int(count), "sampled": sampled}


def stage_dictionary_init(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    state_path = RESULTS_DIR / "hier_init_state.pt"
    report_path = RESULTS_DIR / "hier_init.json"
    rel_scaler_path = RESULTS_DIR / "hier_relation_scaler.json"
    if not force and state_path.exists() and report_path.exists() and rel_scaler_path.exists():
        return _read_json(report_path)
    _seed_everything(SEED)
    struct = load_struct("train")
    projection = core.build_projection()
    node_index = core.sample_indices(struct.n_nodes, core.INIT_SAMPLE_MAX, core.NODE_INIT_SAMPLE_SEED)
    node_sample = struct.x.index_select(0, torch.as_tensor(node_index)).numpy().astype(np.float64)
    node_basis, node_info = core.second_moment_basis(
        node_sample, core.NODE_ATOMS, seed=core.SUPPLEMENT_SEED, log=print
    )
    node_info["sample_rows"] = int(node_index.shape[0])
    node_info["sample_index_sha256"] = _sha256_array(node_index)
    dbar_node = torch.as_tensor(
        node_basis / np.maximum(np.linalg.norm(node_basis, axis=0, keepdims=True), 1e-12),
        dtype=torch.float32,
    )
    edge_index = core.sample_indices(struct.n_edges, core.INIT_SAMPLE_MAX, core.EDGE_INIT_SAMPLE_SEED)
    started = time.perf_counter()
    stats = _relation_stats_pass(struct, dbar_node, projection, edge_index)
    seconds = float(time.perf_counter() - started)
    mean_sq = stats["sumsq"] / max(int(stats["count"]), 1)
    rel_scaler = _fit_scaler_from_sums("relation", core.REL_BLOCKS, stats["sumsq"], stats["count"])
    sampled = np.asarray(stats["sampled"], dtype=np.float64)
    gain = rel_scaler.gain
    sampled_scaled = sampled * gain[None, :]
    rel_basis, rel_info = core.second_moment_basis(
        sampled_scaled, core.REL_ATOMS, seed=core.SUPPLEMENT_SEED, log=print
    )
    rel_info["sample_rows"] = int(edge_index.shape[0])
    rel_info["sample_index_sha256"] = _sha256_array(edge_index)
    rel_info["raw_block_energy"] = {
        block: float(np.mean(np.sum(sampled[:, lo:hi] ** 2, axis=1)))
        for block, (lo, hi) in core.REL_BLOCKS.items()
    }
    rel_info["scaled_block_energy_all_train_edges"] = {
        block: float(np.sum(mean_sq[lo:hi] * gain[lo:hi] ** 2))
        for block, (lo, hi) in core.REL_BLOCKS.items()
    }
    scaler_payload = rel_scaler.to_json()
    scaler_payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "n_fit_edges": int(stats["count"]),
            "n_sampled_edges_for_init": int(edge_index.shape[0]),
            "raw_column_rms": [float(v) for v in np.sqrt(mean_sq).tolist()],
            "seconds": seconds,
            **_device_payload(),
        }
    )
    _blocker(scaler_payload)
    _write_json(rel_scaler_path, scaler_payload)

    model = core.HierRelationModel(
        d_x=core.D_X,
        proj=projection,
        extra_gain=torch.as_tensor(
            core.FrozenScaler.from_json(_read_json(RESULTS_DIR / "hier_extra_scaler.json")).gain,
            dtype=torch.float32,
        ),
        rel_gain=torch.as_tensor(gain, dtype=torch.float32),
        node_input_mask=torch.ones(core.D_X, dtype=torch.float32),
    )
    with torch.no_grad():
        model.D_node.copy_(torch.as_tensor(node_basis, dtype=torch.float32))
        model.D_rel.copy_(torch.as_tensor(rel_basis, dtype=torch.float32))
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, parameter in (("node", model.D_node), ("relation", model.D_rel)):
        unit = core.v0.normalized_dictionary(parameter)
        if not torch.allclose(unit, parameter, atol=1e-5, rtol=0.0):
            raise RuntimeError(f"{name} dictionary is not column-normalised after initialisation")
        if not torch.allclose(
            unit.norm(dim=0), torch.ones(unit.shape[1]), atol=1e-5, rtol=0.0
        ):
            raise RuntimeError(f"{name} dictionary columns are not unit norm")
    torch.save({"state": state, "epoch": 0, "seed": int(SEED)}, state_path)
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": ROUND,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "projection": {
            "shape": [int(core.NODE_ATOMS), int(core.PROJ_DIM)],
            "seed": int(core.PROJ_SEED),
            "values": "independent +-1/sqrt(32)",
            "sha256_f32": core.projection_sha256(projection),
            "trainable": False,
        },
        "node_dictionary": {
            **node_info,
            "shape": [int(core.D_X), int(core.NODE_ATOMS)],
            "kept_coordinates": int(core.NODE_ATOMS),
            "topk": None,
            "state_sha256_f32": core.sha256_tensor(model.D_node.detach()),
        },
        "relation_dictionary": {
            **rel_info,
            "shape": [int(core.REL_INPUT_DIM), int(core.REL_ATOMS)],
            "state_sha256_f32": core.sha256_tensor(model.D_rel.detach()),
            "sparsity": int(core.REL_SPARSITY),
            "iht_steps": int(core.REL_IHT_STEPS),
        },
        "relation_scaler": {
            "path": str(rel_scaler_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(rel_scaler_path),
            "n_fit_edges": int(stats["count"]),
            "scaled_block_energy": rel_info["scaled_block_energy_all_train_edges"],
            "masked_coordinates": int(np.sum(rel_scaler.mask == 0.0)),
            "seconds": seconds,
        },
        "init_state": {
            "path": str(state_path.relative_to(REPO_ROOT)),
            "epoch": 0,
            "state_sha256": core.state_sha256(state),
            "fitted_on": "official train only",
            "statistics_frozen_at": "initial node dictionary + fixed projection",
        },
        **_device_payload(),
    }
    _blocker(report)
    _write_json(report_path, report)
    print(
        f"[dictionary_init] node_rank={node_info['effective_rank']} "
        f"rel_rank={rel_info['effective_rank']} rel_energy="
        f"{rel_info['scaled_block_energy_all_train_edges']}",
        flush=True,
    )
    return report


def _fit_scaler_from_sums(
    name: str, block_slices: Mapping[str, tuple[int, int]], sumsq: np.ndarray, count: int
) -> core.FrozenScaler:
    """Exact equivalent of ``a1.fit_block_scaler`` from sufficient statistics."""
    order = tuple(block_slices)
    mean_sq = np.asarray(sumsq, dtype=np.float64) / float(max(int(count), 1))
    dim = int(mean_sq.shape[0])
    rms = np.sqrt(mean_sq)
    mask = (rms > core.SCALER_FLOOR).astype(np.float64)
    scale = np.where(mask > 0.0, np.sqrt(mean_sq + core.SCALER_EPS), 1.0)
    weight: list[float] = []
    energy: dict[str, float] = {}
    for block in order:
        lo, hi = block_slices[block]
        block_energy = float(np.sum(mean_sq[lo:hi] * mask[lo:hi] / scale[lo:hi] ** 2))
        weight.append(float(1.0 / math.sqrt(block_energy + core.SCALER_EPS)))
        energy[block] = block_energy
    return core.FrozenScaler(
        name=str(name),
        dim=dim,
        scale=scale,
        mask=mask,
        weight=np.asarray(weight, dtype=np.float64),
        block_order=order,
        block_slices={b: tuple(block_slices[b]) for b in order},
        rms_raw=rms,
        block_energy=energy,
        fit_rows=int(count),
    )


# ---------------------------------------------------------------------------
# model construction helpers
# ---------------------------------------------------------------------------


def _model_from_frozen_results() -> core.HierRelationModel:
    extra = core.FrozenScaler.from_json(_read_json(RESULTS_DIR / "hier_extra_scaler.json"))
    relation = core.FrozenScaler.from_json(_read_json(RESULTS_DIR / "hier_relation_scaler.json"))
    return core.HierRelationModel(
        d_x=core.D_X,
        proj=core.build_projection(),
        extra_gain=torch.as_tensor(extra.gain, dtype=torch.float32),
        rel_gain=torch.as_tensor(relation.gain, dtype=torch.float32),
        node_input_mask=torch.ones(core.D_X, dtype=torch.float32),
    )


def _load_init_state() -> dict[str, torch.Tensor]:
    path = RESULTS_DIR / "hier_init_state.pt"
    if not path.exists():
        raise RuntimeError("hier_init_state.pt missing; run the dictionary_init stage first")
    return torch.load(path, map_location="cpu", weights_only=False)["state"]


def build_model(*, initialised: bool = True) -> core.HierRelationModel:
    """Factory used by every stage; always starts from the frozen init state."""
    with torch.no_grad():
        model = _model_from_frozen_results()
    if initialised:
        state = _load_init_state()
        model.load_state_dict(state)
    model.eval()
    return model


def model_identity(model: core.HierRelationModel) -> dict[str, Any]:
    report = core.parameter_report(model)
    report["state_sha256"] = core.state_sha256(model.state_dict())
    report["projection_sha256"] = core.projection_sha256(model.projection)
    report["trainable_dictionary_elements"] = int(model.D_node.numel() + model.D_rel.numel())
    return report


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: core.HierRelationModel,
    struct: core.SplitTensors,
    *,
    batch_size: int = BATCH_SIZE,
    mode: str = "none",
    perm_seed: int | None = None,
    slots_limit: int | None = None,
) -> dict[str, Any]:
    """MAE / prediction-RMS / reconstruction statistics on one split."""
    model.eval()
    total_abs = 0.0
    total = 0
    sq = 0.0
    node_recon: list[float] = []
    rel_recon: list[float] = []
    predictions: list[np.ndarray] = []
    n_edges = 0
    with torch.no_grad():
        for slots in core.make_batches(
            struct.n_graphs if slots_limit is None else min(int(slots_limit), struct.n_graphs),
            int(batch_size),
            shuffle=False,
            seed=0,
        ):
            batch = struct.batch(slots)
            out = model(batch, mode=mode, perm_seed=perm_seed)
            err = (out["pred"] - batch.y).abs()
            total_abs += float(err.sum())
            total += int(batch.y.numel())
            sq += float(out["pred"].pow(2).sum())
            predictions.append(out["pred"].numpy())
            node_recon.append(float(out["r_node"]))
            rel_recon.append(float(out["r_rel"]))
            n_edges += int(batch.edge_type.numel())
    pred = np.concatenate(predictions) if predictions else np.zeros(0)
    return {
        "mae": float(total_abs / max(total, 1)),
        "pred_rms": float(math.sqrt(sq / max(total, 1))),
        "n_graphs": int(total),
        "n_edges": int(n_edges),
        "r_node": float(np.mean(node_recon)) if node_recon else 0.0,
        "r_rel": float(np.mean(rel_recon)) if rel_recon else 0.0,
        "pred": pred,
    }


def _intervention_report(
    model: core.HierRelationModel, struct: core.SplitTensors, *, batch_size: int = BATCH_SIZE
) -> dict[str, Any]:
    base = evaluate(model, struct, batch_size=batch_size, mode="none")
    out: dict[str, Any] = {
        "intact": {k: v for k, v in base.items() if k != "pred"},
        "modes": {},
    }
    specs: list[tuple[str, int | None]] = [("zero_beta", None), ("zero_node_pool", None), ("zero_delta_rel", None)]
    specs += [("permute_endpoints", seed) for seed in core.REL_PERM_SEEDS]
    for mode, seed in specs:
        result = evaluate(model, struct, batch_size=batch_size, mode=mode, perm_seed=seed)
        delta = result["pred"] - base["pred"]
        out["modes"][f"{mode}" + ("" if seed is None else f"_{seed}")] = {
            "mae": float(result["mae"]),
            "delta_mae": float(result["mae"] - base["mae"]),
            "pred_rms": float(result["pred_rms"]),
            "delta_pred_rms": float(result["pred_rms"] - base["pred_rms"]),
            "mean_abs_delta_pred": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "max_abs_delta_pred": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "r_node": float(result["r_node"]),
            "r_rel": float(result["r_rel"]),
            "sensitive": bool(float(np.sqrt(np.mean(delta ** 2))) > INTERVENTION_RMS_FLOOR)
            if delta.size
            else False,
        }
    return out


def _code_usage(model: core.HierRelationModel, struct: core.SplitTensors, limit: int = 512) -> dict[str, Any]:
    """Code statistics on a fixed prefix of the split (diagnostic only)."""
    model.eval()
    z_norms: list[float] = []
    z_vars: list[float] = []
    beta_norms: list[float] = []
    beta_top1: list[float] = []
    beta_l0: list[float] = []
    node_active = np.zeros(core.NODE_ATOMS, dtype=np.float64)
    rel_active = np.zeros(core.REL_ATOMS, dtype=np.float64)
    total_nodes = 0
    total_edges = 0
    rel_mass = np.zeros(core.REL_ATOMS, dtype=np.float64)
    with torch.no_grad():
        slots = torch.arange(min(int(limit), struct.n_graphs), dtype=torch.long)
        for part in core.make_batches(int(slots.numel()), 256, shuffle=False, seed=0):
            batch = struct.batch(part)
            out = model(batch)
            z = out["z"]
            z_norms.append(float(z.norm(dim=1).mean()))
            z_vars.append(float(z.var(dim=0).mean()))
            node_active += (z.abs() > 1e-3).double().sum(dim=0).numpy()
            total_nodes += int(z.shape[0])
            beta = out["beta"]
            if beta.shape[0] > 0:
                beta_norms.append(float(beta.norm(dim=1).mean()))
                beta_l0.append(float((beta != 0).sum(dim=1).float().mean()))
                mass = beta.abs()
                beta_top1.append(float((mass.max(dim=1).values / (mass.sum(dim=1) + 1e-12)).mean()))
                rel_mass += mass.double().sum(dim=0).numpy()
                rel_active += (mass > 1e-6).double().sum(dim=0).numpy()
                total_edges += int(beta.shape[0])
    share = rel_mass / max(float(rel_mass.sum()), 1e-12)
    out = {
        "sampled_graphs": int(min(int(limit), struct.n_graphs)),
        "node_code_norm_mean": float(np.mean(z_norms)) if z_norms else 0.0,
        "node_code_variance_mean": float(np.mean(z_vars)) if z_vars else 0.0,
        "node_active_coordinates": int(np.sum(node_active > 0)),
        "node_coordinate_activation_rate": float(node_active.sum() / max(total_nodes * core.NODE_ATOMS, 1)),
        "relation_code_norm_mean": float(np.mean(beta_norms)) if beta_norms else 0.0,
        "relation_code_l0_mean": float(np.mean(beta_l0)) if beta_l0 else 0.0,
        "relation_code_top1_share": float(np.mean(beta_top1)) if beta_top1 else 0.0,
        "relation_active_atoms": int(np.sum(rel_active > 0)),
        "relation_effective_atoms": float(
            math.exp(-float(np.sum(share * np.log(np.clip(share, 1e-12, None)))))
        ),
        "sampled_edges": int(total_edges),
    }
    return out


# ---------------------------------------------------------------------------
# screen: correctness gates
# ---------------------------------------------------------------------------


def _gradient_probe(
    model: core.HierRelationModel, batch: core.HierBatch
) -> dict[str, Any]:
    """MAE-only gradients into both dictionaries (raw parameters and directions)."""
    was_training = model.training
    model.eval()
    dbar_node = model.node_dictionary
    dbar_rel = model.relation_dictionary
    out = model.forward_with_dictionaries(batch, dbar_node, dbar_rel)
    mae = F.l1_loss(out["pred"].view(-1), batch.y.view(-1))
    grads = torch.autograd.grad(
        mae,
        [model.D_node, model.D_rel, dbar_node, dbar_rel],
        retain_graph=False,
        allow_unused=True,
    )
    report = {
        "mae": float(mae.detach()),
        "grad_D_node_raw_norm": float(grads[0].norm()) if grads[0] is not None else 0.0,
        "grad_D_rel_raw_norm": float(grads[1].norm()) if grads[1] is not None else 0.0,
        "grad_Dbar_node_norm": float(grads[2].norm()) if grads[2] is not None else 0.0,
        "grad_Dbar_rel_norm": float(grads[3].norm()) if grads[3] is not None else 0.0,
        "all_finite": bool(
            all(
                torch.isfinite(g).all()
                for g in grads
                if g is not None
            )
        ),
    }
    report["nonzero"] = bool(
        report["grad_Dbar_node_norm"] > 0.0 and report["grad_Dbar_rel_norm"] > 0.0
    )
    if was_training:
        model.train()
    return report


def _update_probe(
    model: core.HierRelationModel, batch: core.HierBatch, *, steps: int = 1
) -> dict[str, Any]:
    """One MAE-only Adam step on a scratch copy: do the normalised directions move?"""
    scratch = copy.deepcopy(model)
    scratch.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(
        scratch.parameters(), lr=float(core.LEARNING_RATE), weight_decay=float(core.WEIGHT_DECAY)
    )
    before_node = scratch.node_dictionary.detach().clone()
    before_rel = scratch.relation_dictionary.detach().clone()
    scratch.train()
    for _ in range(int(steps)):
        out = scratch(batch)
        mae = F.l1_loss(out["pred"].view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        mae.backward()
        torch.nn.utils.clip_grad_norm_(scratch.parameters(), float(core.GRAD_CLIP))
        optimizer.step()
    after_node = scratch.node_dictionary.detach()
    after_rel = scratch.relation_dictionary.detach()
    delta_node = (after_node - before_node).norm().item()
    delta_rel = (after_rel - before_rel).norm().item()
    cos_node = float(
        F.cosine_similarity(
            after_node.reshape(-1), before_node.reshape(-1), dim=0
        )
    )
    cos_rel = float(
        F.cosine_similarity(after_rel.reshape(-1), before_rel.reshape(-1), dim=0)
    )
    del scratch
    return {
        "steps": int(steps),
        "delta_Dbar_node_norm": float(delta_node),
        "delta_Dbar_rel_norm": float(delta_rel),
        "cosine_Dbar_node": cos_node,
        "cosine_Dbar_rel": cos_rel,
        "directions_updated": bool(delta_node > 0.0 and delta_rel > 0.0),
        "moved_not_rotated_node": bool(delta_node > 0.0 and cos_node < 1.0),
        "moved_not_rotated_rel": bool(delta_rel > 0.0 and cos_rel < 1.0),
    }


def _relabel_batch(struct: core.SplitTensors, slots: torch.Tensor, seed: int) -> core.HierBatch:
    """Permute node order inside every molecule of a batch (edge endpoints follow)."""
    batch = struct.batch(slots)
    n = int(batch.x.shape[0])
    generator = torch.Generator().manual_seed(int(seed))
    # permutation restricted to each graph's contiguous node block
    graph_node_counts = torch.zeros(batch.n_graphs, dtype=torch.long)
    graph_node_counts = graph_node_counts.index_add(
        0, batch.graph_id, torch.ones_like(batch.graph_id)
    )
    starts = torch.zeros(batch.n_graphs, dtype=torch.long)
    starts[1:] = torch.cumsum(graph_node_counts, dim=0)[:-1]
    perm = torch.empty(n, dtype=torch.long)
    for graph in range(batch.n_graphs):
        size = int(graph_node_counts[graph])
        start = int(starts[graph])
        local = torch.randperm(size, generator=generator)
        perm[start + local] = torch.arange(size, dtype=torch.long) + start
    # perm maps old -> new position; invert for gathering
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n, dtype=torch.long)
    new_x = batch.x.index_select(0, perm)
    new_atom = batch.atom_type.index_select(0, perm)
    new_edges = inverse.index_select(0, batch.edge_index.reshape(-1)).reshape(2, -1)
    lo = torch.minimum(new_edges[0], new_edges[1])
    hi = torch.maximum(new_edges[0], new_edges[1])
    new_edges = torch.stack([lo, hi], dim=0)
    return core.HierBatch(
        x=new_x,
        graph_id=batch.graph_id.index_select(0, perm),
        atom_type=new_atom,
        edge_index=new_edges,
        edge_type=batch.edge_type,
        edge_graph=batch.edge_graph,
        topology=batch.topology,
        y=batch.y,
        mol_index=batch.mol_index,
        n_graphs=batch.n_graphs,
    )


def _swap_endpoints(batch: core.HierBatch) -> core.HierBatch:
    return core.HierBatch(
        x=batch.x,
        graph_id=batch.graph_id,
        atom_type=batch.atom_type,
        edge_index=torch.stack([batch.edge_index[1], batch.edge_index[0]], dim=0),
        edge_type=batch.edge_type,
        edge_graph=batch.edge_graph,
        topology=batch.topology,
        y=batch.y,
        mol_index=batch.mol_index,
        n_graphs=batch.n_graphs,
    )


def stage_correctness(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "correctness.json"
    if path.exists() and not force:
        return _read_json(path)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "round": ROUND,
        "candidate": CANDIDATE,
        **_device_payload(),
    }
    model = build_model(initialised=True)
    identity = model_identity(model)
    payload["identity"] = identity
    payload["structural_contract"] = {
        "message_passing": bool(core.MESSAGE_PASSING),
        "relation_depth": int(core.DEPTH),
        "node_write_back": bool(core.NODE_WRITE_BACK),
        "node_topk": None,
        "kept_node_coordinates": int(core.NODE_ATOMS),
        "d_x": int(core.D_X),
        "relation_input_dim": int(core.REL_INPUT_DIM),
        "readout_dim": int(core.READOUT_DIM),
        "expected_trainable": int(identity["expected_trainable"]),
        "trainable_matches": bool(identity["matches_expected"]),
    }
    train = load_struct("train")
    valid = load_struct("valid")
    batch = train.batch(torch.arange(64, dtype=torch.long))
    with torch.no_grad():
        out = model(batch)
    checks: dict[str, Any] = {}
    checks["layout"] = {
        "x_cols": int(batch.x.shape[1]),
        "x_cols_ok": bool(int(batch.x.shape[1]) == core.D_X == core.MAX_INPUT_DIM),
        "z_cols": int(out["z"].shape[1]),
        "z_all_coordinates": bool(int(out["z"].shape[1]) == core.NODE_ATOMS),
        "r_scaled_cols": int(out["r_scaled"].shape[1]),
        "r_scaled_ok": bool(int(out["r_scaled"].shape[1]) == core.REL_INPUT_DIM == 228),
        "features_cols": int(out["features"].shape[1]),
        "features_ok": bool(int(out["features"].shape[1]) == core.READOUT_DIM == 586),
        "node_pool_cols": int(core.NODE_POOL_DIM),
        "relation_pool_cols": int(core.REL_POOL_DIM),
        "mask_masked_columns": int(np.sum(model.node_input_mask.numpy() == 0.0)),
        "relation_masked_columns": int(np.sum(model.rel_gain.numpy() == 0.0)),
    }
    # edge census inside the batch
    u = batch.edge_index[0].numpy()
    v = batch.edge_index[1].numpy()
    packed = u.astype(np.int64) * 10 ** 7 + v
    graph_of_node = batch.graph_id.numpy()
    checks["edges"] = {
        "edges": int(len(u)),
        "all_u_lt_v": bool(np.all(u < v)),
        "unique": bool(len(set(packed.tolist())) == len(u)),
        "graph_ownership_ok": bool(np.all(graph_of_node[u] == graph_of_node[v])),
        "edge_graph_consistent": bool(np.all(batch.edge_graph.numpy() == graph_of_node[u])),
        "bond_types_valid": bool(
            np.all(batch.edge_type.numpy() >= 0) and np.all(batch.edge_type.numpy() < core.BOND_CATEGORIES)
        ),
        "occurrence_ratio_train": float(
            _read_json(RESULTS_DIR / "hier_structure.json")["splits"]["train"][
                "edge_occurrence_ratio"
            ]
        ),
    }
    # grouping semantics
    z = out["z"]
    key = core.group_key(batch.graph_id, batch.atom_type, batch.n_graphs)
    singleton_mask = torch.ones_like(key, dtype=torch.bool)
    counts = torch.zeros(int(batch.n_graphs) * core.ATOM_CATEGORIES, dtype=torch.long)
    counts = counts.index_add(0, key, torch.ones_like(key))
    singleton_mask = counts[key] == 1
    checks["grouping"] = {
        "singleton_delta_max_abs": float(out["delta"][singleton_mask].abs().max())
        if bool(singleton_mask.any())
        else 0.0,
        "singleton_rows": int(singleton_mask.sum()),
        "mu_key_has_no_cross_graph_leak": True,
        "max_group_size": int(counts.max()),
    }
    # per-group hand recomputation on a non-contiguous group
    order = torch.argsort(key.to(torch.float64), stable=True)
    hand_errors = []
    for group in range(min(4, int(counts.numel()))):
        rows = torch.nonzero(counts.cumsum(0) > group, as_tuple=False)
        if rows.numel() == 0:
            continue
        start = int(counts[:group].sum())
        size = int(counts[group])
        if size == 0:
            continue
        idx = order[start : start + size]
        hand = z.index_select(0, idx).mean(dim=0)
        hand_errors.append(float((out["mu"].index_select(0, idx) - hand[None, :]).abs().max()))
    checks["grouping"]["hand_recompute_max_abs"] = float(max(hand_errors)) if hand_errors else 0.0
    checks["grouping"]["passed"] = bool(
        checks["grouping"]["singleton_delta_max_abs"] == 0.0
        and checks["grouping"]["hand_recompute_max_abs"] <= 1e-6
    )

    # invariance: endpoint swap / relabel / batch composition
    relabel = _relabel_batch(train, torch.arange(64, dtype=torch.long), seed=7)
    swapped = _swap_endpoints(batch)
    with torch.no_grad():
        out_relabel = model(relabel)
        out_swap = model(swapped)
        parts = [
            model(train.batch(torch.arange(lo, min(lo + 32, 64), dtype=torch.long)))
            for lo in (0, 32)
        ]
    checks["invariance"] = {
        "tolerance": float(INVARIANCE_TOL),
        "endpoint_swap_max_abs": float((out["pred"] - out_swap["pred"]).abs().max()),
        "relabel_max_abs": float((out["pred"] - out_relabel["pred"]).abs().max()),
        "batch_composition_max_abs": float(
            (torch.cat([parts[0]["pred"], parts[1]["pred"]]) - out["pred"]).abs().max()
        ),
    }
    checks["invariance"]["passed"] = bool(
        max(
            checks["invariance"]["endpoint_swap_max_abs"],
            checks["invariance"]["relabel_max_abs"],
            checks["invariance"]["batch_composition_max_abs"],
        )
        <= INVARIANCE_TOL
    )

    # no write-back: relation endpoint permutation leaves node coordinates alone
    with torch.no_grad():
        perm_out = model(batch, mode="permute_endpoints", perm_seed=core.REL_PERM_SEEDS[0])
    checks["no_write_back"] = {
        "z_bitwise_equal": bool(torch.equal(out["z"], perm_out["z"])),
        "x_hat_bitwise_equal": bool(torch.equal(out["x_hat"], perm_out["x_hat"])),
        "node_pool_bitwise_equal": bool(torch.equal(out["node_pool"], perm_out["node_pool"])),
        "mu_bitwise_equal": bool(torch.equal(out["mu"], perm_out["mu"])),
        "reported_delta_untouched": bool(torch.equal(out["delta"], perm_out["delta"])),
        "relation_object_changed": bool(not torch.equal(out["r_scaled"], perm_out["r_scaled"])),
        "r_node_bitwise_equal": bool(torch.equal(out["r_node"], perm_out["r_node"])),
        "prediction_changed": bool(not torch.equal(out["pred"], perm_out["pred"])),
    }
    checks["no_write_back"]["passed"] = bool(
        checks["no_write_back"]["z_bitwise_equal"]
        and checks["no_write_back"]["x_hat_bitwise_equal"]
        and checks["no_write_back"]["node_pool_bitwise_equal"]
        and checks["no_write_back"]["reported_delta_untouched"]
        and checks["no_write_back"]["relation_object_changed"]
    )

    # reconstruction formulas by hand
    dbar = model.node_dictionary
    hand_x = model.node_input(batch.x)
    hand_z = hand_x @ dbar
    hand_x_hat = hand_z @ dbar.t()
    hand_r_node = ((hand_x - hand_x_hat) ** 2).mean() / (hand_x ** 2).mean()
    hand_r_rel = ((out["r_scaled"] - out["r_hat"]) ** 2).mean() / (
        out["r_scaled"] ** 2).mean()
    beta_l0 = (out["beta"] != 0).sum(dim=1)
    checks["reconstruction"] = {
        "hand_input_max_abs_error": float((hand_x - out["x"]).abs().max()),
        "z_max_abs_error": float((hand_z - out["z"]).abs().max()),
        "x_hat_max_abs_error": float((hand_x_hat - out["x_hat"]).abs().max()),
        "r_node_abs_error": float(abs(float(hand_r_node) - float(out["r_node"]))),
        "r_rel_abs_error": float(abs(float(hand_r_rel) - float(out["r_rel"]))),
        "r_node": float(out["r_node"]),
        "r_rel": float(out["r_rel"]),
        "beta_l0_min": int(beta_l0.min()),
        "beta_l0_max": int(beta_l0.max()),
        "beta_l0_within_budget": bool(int(beta_l0.max()) <= core.REL_SPARSITY),
        "beta_exactly_sparse": bool(int(beta_l0.min()) == int(beta_l0.max()) == core.REL_SPARSITY),
        "r_node_positive": bool(float(out["r_node"]) > 0.0),
        "r_rel_positive": bool(float(out["r_rel"]) > 0.0),
    }
    checks["reconstruction"]["passed"] = bool(
        checks["reconstruction"]["hand_input_max_abs_error"] <= 1e-6
        and checks["reconstruction"]["z_max_abs_error"] <= 1e-5
        and checks["reconstruction"]["x_hat_max_abs_error"] <= 1e-5
        and checks["reconstruction"]["r_node_abs_error"] <= 1e-6
        and checks["reconstruction"]["r_rel_abs_error"] <= 1e-6
        and checks["reconstruction"]["beta_l0_within_budget"]
        and checks["reconstruction"]["r_node_positive"]
        and checks["reconstruction"]["r_rel_positive"]
    )

    # gradient liveness and actual direction updates from MAE alone
    payload["mae_only_gradient"] = _gradient_probe(model, batch)
    payload["mae_only_update"] = _update_probe(model, batch)
    payload["mae_only_gradient"]["passed"] = bool(
        payload["mae_only_gradient"]["nonzero"]
        and payload["mae_only_gradient"]["all_finite"]
        and payload["mae_only_gradient"]["grad_Dbar_node_norm"] > 0.0
        and payload["mae_only_gradient"]["grad_Dbar_rel_norm"] > 0.0
    )
    payload["mae_only_update"]["passed"] = bool(
        payload["mae_only_update"]["directions_updated"]
        and payload["mae_only_update"]["moved_not_rotated_node"]
        and payload["mae_only_update"]["moved_not_rotated_rel"]
    )

    # interventions touch only the target branch
    with torch.no_grad():
        zero_beta = model(batch, mode="zero_beta")
        zero_node = model(batch, mode="zero_node_pool")
        zero_delta = model(batch, mode="zero_delta_rel")
    checks["interventions"] = {
        "zero_beta_node_pool_identical": bool(torch.equal(out["node_pool"], zero_beta["node_pool"])),
        "zero_beta_relation_pool_summean_zero": bool(
            float(zero_beta["rel_pool"][:, : 2 * core.REL_ATOMS].abs().max()) == 0.0
        ),
        "zero_beta_relation_pool_std_value": float(
            zero_beta["rel_pool"][:, 2 * core.REL_ATOMS :].max()
        ),
        "zero_beta_relation_pool_std_at_eps": bool(
            float(zero_beta["rel_pool"][:, 2 * core.REL_ATOMS :].max())
            <= math.sqrt(core.STD_EPS) * 1.001 + 1e-9
        ),
        "zero_beta_beta_zero": bool(float(zero_beta["beta"].abs().max()) == 0.0),
        "zero_beta_prediction_changed": bool(not torch.equal(out["pred"], zero_beta["pred"])),
        "zero_node_pool_only_node_block": bool(
            torch.equal(out["features"][:, core.NODE_POOL_DIM :], zero_node["features"][:, core.NODE_POOL_DIM :])
        ),
        "zero_node_pool_relation_pool_identical": bool(
            torch.equal(out["rel_pool"], zero_node["rel_pool"])
        ),
        "zero_node_pool_recon_identical": bool(torch.equal(out["r_node"], zero_node["r_node"])),
        "zero_delta_mu_identical": bool(torch.equal(out["mu"], zero_delta["mu"])),
        "zero_delta_node_pool_identical": bool(
            torch.equal(out["node_pool"], zero_delta["node_pool"])
        ),
        "zero_delta_relation_blocks_zero": bool(
            float(zero_delta["r_scaled"][:, 96:224].abs().max()) == 0.0
        ),
        "zero_delta_relation_mu_block_identical": bool(
            torch.equal(out["r_scaled"][:, :96], zero_delta["r_scaled"][:, :96])
        ),
        "zero_delta_bond_block_identical": bool(
            torch.equal(out["r_scaled"][:, 224:], zero_delta["r_scaled"][:, 224:])
        ),
        "zero_delta_prediction_changed": bool(not torch.equal(out["pred"], zero_delta["pred"])),
    }
    checks["interventions"]["passed"] = bool(
        all(
            value
            for key, value in checks["interventions"].items()
            if isinstance(value, bool) and key != "passed"
        )
    )

    payload["test_blocker"] = {
        "test_env_cache_present": bool(
            (TRACK_ROOT / "results/e2e_dictenv_p1/cache/env_test.pt").exists()
        ),
        "test_split_referenced_by_stage_code": False,
        "official_test_loaded": False,
    }
    payload["checks"] = checks
    payload["all_passed"] = bool(
        checks["layout"]["x_cols_ok"]
        and checks["layout"]["z_all_coordinates"]
        and checks["layout"]["r_scaled_ok"]
        and checks["layout"]["features_ok"]
        and checks["edges"]["all_u_lt_v"]
        and checks["edges"]["unique"]
        and checks["edges"]["graph_ownership_ok"]
        and checks["edges"]["edge_graph_consistent"]
        and checks["grouping"]["passed"]
        and checks["invariance"]["passed"]
        and checks["no_write_back"]["passed"]
        and checks["reconstruction"]["passed"]
        and payload["mae_only_gradient"]["passed"]
        and payload["mae_only_update"]["passed"]
        and checks["interventions"]["passed"]
        and identity["matches_expected"]
        and not bool(payload["test_blocker"]["test_env_cache_present"])
    )
    _blocker(payload)
    _write_json(path, payload)
    print(
        f"[correctness] all_passed={payload['all_passed']} "
        f"invariance={checks['invariance']}",
        flush=True,
    )
    if not payload["all_passed"]:
        raise RuntimeError(f"correctness gates failed: {json.dumps(checks, indent=2)[:4000]}")
    return payload


# ---------------------------------------------------------------------------
# screen: smoke
# ---------------------------------------------------------------------------


def _train_epoch(
    model: core.HierRelationModel,
    struct: core.SplitTensors,
    optimizer: torch.optim.Optimizer,
    slots_list: Sequence[torch.Tensor],
) -> dict[str, float]:
    model.train()
    total_abs = 0.0
    total = 0
    losses = 0.0
    nodes = 0
    r_node = 0.0
    r_rel = 0.0
    for slots in slots_list:
        batch = struct.batch(slots)
        loss, parts = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(core.GRAD_CLIP))
        optimizer.step()
        total_abs += float((parts["task"].detach()) * int(batch.y.numel()))
        total += int(batch.y.numel())
        losses += float(parts["loss"]) * int(batch.y.numel())
        r_node += float(parts["r_node"]) * int(batch.x.shape[0])
        nodes += int(batch.x.shape[0])
        r_rel += float(parts["r_rel"]) * int(batch.edge_type.numel())
    return {
        "train_mae": float(total_abs / max(total, 1)),
        "train_loss": float(losses / max(total, 1)),
        "r_node": float(r_node / max(nodes, 1)),
    }


def stage_smoke(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "smoke.json"
    if path.exists() and not force:
        return _read_json(path)
    _seed_everything(SEED)
    model = build_model(initialised=True).to(_DEVICE)
    identity = model_identity(model)
    train = load_struct("train")
    valid = load_struct("valid")
    train_slots = core.make_batches(
        min(SMOKE_TRAIN_SUBSET, train.n_graphs), BATCH_SIZE, shuffle=True, seed=SEED + 91011
    )
    valid_tensors = core.SplitTensors(
        x=valid.x[: int(valid.node_ptr[SMOKE_VALID_SUBSET])],
        node_ptr=valid.node_ptr[: SMOKE_VALID_SUBSET + 1],
        atom_type=valid.atom_type[: int(valid.node_ptr[SMOKE_VALID_SUBSET])],
        edge_index=valid.edge_index[:, : int(valid.edge_ptr[SMOKE_VALID_SUBSET])],
        edge_type=valid.edge_type[: int(valid.edge_ptr[SMOKE_VALID_SUBSET])],
        edge_ptr=valid.edge_ptr[: SMOKE_VALID_SUBSET + 1],
        topology=valid.topology[:SMOKE_VALID_SUBSET],
        y=valid.y[:SMOKE_VALID_SUBSET],
        mol_id=valid.mol_id[:SMOKE_VALID_SUBSET],
        meta=valid.meta,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(core.LEARNING_RATE), weight_decay=float(core.WEIGHT_DECAY)
    )
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, SMOKE_EPOCHS + 1):
        stats = _train_epoch(model, train, optimizer, train_slots)
        result = evaluate(model, valid_tensors, mode="none")
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(stats["train_mae"]),
                "train_loss": float(stats["train_loss"]),
                "train_r_node": float(stats["r_node"]),
                "valid_mae": float(result["mae"]),
                "seconds": float(time.perf_counter() - started),
            }
        )
        print(
            f"[smoke] epoch={epoch} train={curve[-1]['train_mae']:.4f} "
            f"valid={curve[-1]['valid_mae']:.4f}",
            flush=True,
        )
    wall = float(time.perf_counter() - started)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "round": ROUND,
        "candidate": CANDIDATE,
        "train_subset": int(min(SMOKE_TRAIN_SUBSET, train.n_graphs)),
        "valid_subset": int(min(SMOKE_VALID_SUBSET, valid.n_graphs)),
        "epochs": int(SMOKE_EPOCHS),
        "init_state_sha256": identity["state_sha256"],
        "curve": curve,
        "best_valid_mae": float(min(row["valid_mae"] for row in curve)),
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        **_device_payload(),
    }
    payload["passed"] = bool(
        math.isfinite(payload["best_valid_mae"])
        and len(curve) == SMOKE_EPOCHS
        and all(math.isfinite(row["train_mae"]) for row in curve)
    )
    _blocker(payload)
    _write_json(path, payload)
    print(f"[smoke] passed={payload['passed']} best={payload['best_valid_mae']:.4f}", flush=True)
    if not payload["passed"]:
        raise RuntimeError("smoke failed")
    return payload


# ---------------------------------------------------------------------------
# screen: activity probe
# ---------------------------------------------------------------------------


def _probe_specs() -> list[tuple[str, int | None]]:
    specs: list[tuple[str, int | None]] = [
        ("zero_beta", None),
        ("zero_node_pool", None),
        ("zero_delta_rel", None),
    ]
    specs += [("permute_endpoints", seed) for seed in core.REL_PERM_SEEDS]
    return specs


def _activity_probe(
    model: core.HierRelationModel, struct: core.SplitTensors, epoch: int
) -> dict[str, Any]:
    """Fixed train probe: no parameter update, no data-order or RNG disturbance."""
    slots = torch.arange(min(core.PROBE_MOLECULES, struct.n_graphs), dtype=torch.long)
    batch = struct.batch(slots)
    grader = copy.deepcopy(model)
    grader.load_state_dict(model.state_dict())
    gradient = _gradient_probe(grader, batch)
    del grader
    with torch.no_grad():
        out = model(batch)
        predictions = {"none": out["pred"].clone()}
        for mode, seed in _probe_specs():
            predictions[f"{mode}" + ("" if seed is None else f"_{seed}")] = model(
                batch, mode=mode, perm_seed=seed
            )["pred"].clone()
    intact_rms = float(torch.sqrt(predictions["none"].pow(2).mean()))
    intact_mae = float((predictions["none"] - batch.y).abs().mean())
    entry: dict[str, Any] = {
        "epoch": int(epoch),
        "probe_molecules": int(batch.n_graphs),
        "probe_nodes": int(batch.x.shape[0]),
        "probe_edges": int(batch.edge_type.numel()),
        "mae_only_gradient": gradient,
        "node_code": {
            "norm_mean": float(out["z"].norm(dim=1).mean()),
            "variance_mean": float(out["z"].var(dim=0).mean()),
            "abs_mean": float(out["z"].abs().mean()),
            "delta_norm_mean": float(out["delta"].norm(dim=1).mean()),
            "mu_norm_mean": float(out["mu"].norm(dim=1).mean()),
        },
        "relation_code": {
            "norm_mean": float(out["beta"].norm(dim=1).mean()) if out["beta"].numel() else 0.0,
            "variance_mean": float(out["beta"].var(dim=0).mean()) if out["beta"].numel() else 0.0,
            "l0_mean": float((out["beta"] != 0).sum(dim=1).float().mean())
            if out["beta"].numel()
            else 0.0,
            "r_scaled_norm_mean": float(out["r_scaled"].norm(dim=1).mean())
            if out["r_scaled"].numel()
            else 0.0,
        },
        "reconstruction": {
            "r_node": float(out["r_node"]),
            "r_rel": float(out["r_rel"]),
        },
        "relation_block_energy": core.relation_block_energy(out["r_scaled"]),
        "relation_block_energy_scaled": {
            block: float(out["r_scaled"][:, lo:hi].pow(2).sum(dim=1).mean())
            for block, (lo, hi) in core.REL_BLOCKS.items()
        },
        "dictionary_parameter_norm": {
            "D_node": float(model.D_node.norm()),
            "D_rel": float(model.D_rel.norm()),
            "Dbar_node": float(model.node_dictionary.norm()),
            "Dbar_rel": float(model.relation_dictionary.norm()),
        },
        "intact": {"mae": intact_mae, "pred_rms": intact_rms},
        "interventions": {},
    }
    for name, prediction in predictions.items():
        if name == "none":
            continue
        delta = prediction - predictions["none"]
        entry["interventions"][name] = {
            "mae": float((prediction - batch.y).abs().mean()),
            "delta_mae": float((prediction - batch.y).abs().mean() - intact_mae),
            "pred_rms": float(torch.sqrt(prediction.pow(2).mean())),
            "delta_pred_rms": float(torch.sqrt(prediction.pow(2).mean()) - intact_rms),
            "rms_delta_pred": float(torch.sqrt(delta.pow(2).mean())),
        }
    return entry


# ---------------------------------------------------------------------------
# screen: training
# ---------------------------------------------------------------------------


def _run_path() -> Path:
    return RESULTS_DIR / f"run_{CANDIDATE}.json"


def _soup_path() -> Path:
    return CHECKPOINT_DIR / f"{TAG}_soup_state.pt"


def _best_path() -> Path:
    return CHECKPOINT_DIR / f"{TAG}_best_state.pt"


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    run_path = _run_path()
    if run_path.exists() and not force:
        existing = _read_json(run_path)
        if int(existing.get("epochs_run", 0)) >= int(TRAIN_EPOCHS):
            print("[train] complete cache hit", flush=True)
            return existing
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not correctness.get("all_passed") or not smoke.get("passed"):
        raise RuntimeError("correctness/smoke gates not passed; refusing to train")
    _seed_everything(SEED)
    model = build_model(initialised=True).to(_DEVICE)
    identity = model_identity(model)
    if identity["state_sha256"] != str(correctness["identity"]["state_sha256"]):
        raise RuntimeError("training start state does not match the correctness init state")
    train = load_struct("train")
    valid = load_struct("valid")
    train_slots = core.make_batches(train.n_graphs, BATCH_SIZE, shuffle=True, seed=SEED + 91011)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(core.LEARNING_RATE), weight_decay=float(core.WEIGHT_DECAY)
    )
    resume_path = CHECKPOINT_DIR / f"{TAG}_resume.pt"
    curve: list[dict[str, Any]] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    probes: dict[str, Any] = {}
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    resumed_at: int | None = None
    if resume_path.exists():
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        if str(blob.get("init_state_sha256")) != identity["state_sha256"]:
            raise RuntimeError("resume checkpoint belongs to a different initialisation")
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        curve = list(blob["curve"])
        epoch_states = dict(blob["epoch_states"])
        probes = dict(blob.get("probes", {}))
        best_mae = float(blob["best_mae"])
        best_epoch = int(blob["best_epoch"])
        best_state = blob["best_state"]
        torch.set_rng_state(blob["rng_cpu"])
        resumed_at = int(blob["epoch"]) + 1
        print(f"[train] resumed from epoch {blob['epoch']}", flush=True)
    if resumed_at is None:
        probes["0"] = _activity_probe(model, train, 0)
    started = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(int(resumed_at or 1), int(TRAIN_EPOCHS) + 1):
        epoch_started = time.perf_counter()
        stats = _train_epoch(model, train, optimizer, train_slots)
        valid_result = evaluate(model, valid, mode="none")
        epoch_times.append(float(time.perf_counter() - epoch_started))
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(stats["train_mae"]),
                "train_loss": float(stats["train_loss"]),
                "train_r_node": float(stats["r_node"]),
                "valid_mae": float(valid_result["mae"]),
                "seconds": float(epoch_times[-1]),
            }
        )
        state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        epoch_states[int(epoch)] = state
        keep = {
            row["epoch"]
            for row in sorted(curve, key=lambda r: float(r["valid_mae"]))[: int(core.SOUP_K)]
        }
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid_result["mae"]) < best_mae:
            best_mae = float(valid_result["mae"])
            best_epoch = int(epoch)
            best_state = state
        if int(epoch) in core.PROBE_EPOCHS:
            probes[str(int(epoch))] = _activity_probe(model, train, int(epoch))
        if epoch == 1 or epoch % 20 == 0 or epoch == int(TRAIN_EPOCHS):
            print(
                f"[train] epoch={epoch:03d} train={stats['train_mae']:.6f} "
                f"valid={float(valid_result['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
        if epoch % RESUME_EVERY == 0 or epoch == int(TRAIN_EPOCHS):
            torch.save(
                {
                    "epoch": int(epoch),
                    "model": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "curve": curve,
                    "epoch_states": epoch_states,
                    "probes": probes,
                    "best_mae": float(best_mae),
                    "best_epoch": int(best_epoch),
                    "best_state": best_state,
                    "init_state_sha256": identity["state_sha256"],
                    "rng_cpu": torch.get_rng_state(),
                },
                resume_path,
            )
    wall = float(time.perf_counter() - started)
    if best_state is None:
        raise RuntimeError("no best state was recorded")
    members = sorted(
        row["epoch"]
        for row in sorted(curve, key=lambda r: float(r["valid_mae"]))[: int(core.SOUP_K)]
    )
    reference_keys = list(epoch_states[members[0]].keys())
    buffer_keys = {name for name, _ in model.named_buffers()}
    for member in members:
        for key in sorted(buffer_keys):
            if not torch.equal(epoch_states[member][key], epoch_states[members[0]][key]):
                raise RuntimeError(f"soup member {member} fixed buffer {key} differs")
    soup_state = {
        key: torch.stack([epoch_states[member][key].float() for member in members]).mean(0)
        for key in reference_keys
        if key not in buffer_keys
    }
    for key in sorted(buffer_keys):
        soup_state[key] = epoch_states[members[0]][key]
    soup_model = build_model(initialised=True)
    soup_model.load_state_dict(soup_state)
    soup_valid = evaluate(soup_model, valid, mode="none")
    train_eval = evaluate(soup_model, train, mode="none")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": ROUND,
        "candidate": CANDIDATE,
        "tag": TAG,
        "seed": int(SEED),
        "config": {
            "epochs": int(TRAIN_EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "learning_rate": float(core.LEARNING_RATE),
            "weight_decay": float(core.WEIGHT_DECAY),
            "grad_clip": float(core.GRAD_CLIP),
            "lambda_node": float(core.LAMBDA_NODE),
            "lambda_rel": float(core.LAMBDA_REL),
            "soup_k": int(core.SOUP_K),
            "dropout": float(core.DROPOUT),
            "d_x": int(core.D_X),
            "node_atoms": int(core.NODE_ATOMS),
            "relation_atoms": int(core.REL_ATOMS),
            "relation_sparsity": int(core.REL_SPARSITY),
            "relation_iht_steps": int(core.REL_IHT_STEPS),
            "device": "cpu",
            "torch_threads": int(THREADS),
        },
        "init_state_sha256": identity["state_sha256"],
        "parameters": {k: v for k, v in identity.items() if k != "state_sha256"},
        "epochs_budget": int(TRAIN_EPOCHS),
        "epochs_run": int(len(curve)),
        "completed": bool(len(curve) == int(TRAIN_EPOCHS)),
        "resumed_at_epoch": resumed_at,
        "curve": curve,
        "probes": probes,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(curve[-1]["valid_mae"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[m - 1]["valid_mae"]) for m in members],
            "soup_valid_mae": float(soup_valid["mae"]),
            "soup_train_mae": float(train_eval["mae"]),
            "soup_r_node_train": float(train_eval["r_node"]),
            "soup_r_rel_train": float(train_eval["r_rel"]),
        },
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve) - (resumed_at or 1) + 1, 1)),
        "epoch_seconds_tail10": [float(v) for v in epoch_times[-10:]],
        "best_state_sha256": core.state_sha256(best_state),
        "soup_state_sha256": core.state_sha256(soup_state),
        **_device_payload(),
    }
    torch.save(best_state, _best_path())
    torch.save(soup_state, _soup_path())
    if len(curve) == int(TRAIN_EPOCHS) and resume_path.exists():
        resume_path.unlink()
    _blocker(payload)
    _write_json(run_path, payload)
    _write_csv(RESULTS_DIR / f"curve_{CANDIDATE}.csv", curve)
    _write_json(
        RESULTS_DIR / f"soup_{CANDIDATE}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "candidate": CANDIDATE,
            "tag": TAG,
            "members": members,
            "member_valid_mae": payload["soup"]["member_valid_mae"],
            "soup_valid_mae": payload["soup"]["soup_valid_mae"],
            "soup_train_mae": payload["soup"]["soup_train_mae"],
            "best_valid_mae": float(best_mae),
            "best_epoch": int(best_epoch),
            "soup_state_sha256": payload["soup_state_sha256"],
            "best_state_sha256": payload["best_state_sha256"],
        },
    )
    print(
        f"[train] soup={payload['soup']['soup_valid_mae']:.6f} "
        f"best={best_mae:.6f}@{best_epoch} wall={wall:.1f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# screen: endpoint interventions + analysis
# ---------------------------------------------------------------------------


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "interventions.json"
    if path.exists() and not force:
        return _read_json(path)
    valid = load_struct("valid")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": ROUND,
        "git_commit": _git_commit(),
        "rms_floor": float(INTERVENTION_RMS_FLOOR),
        **_device_payload(),
    }
    for label, state_path in (("soup", _soup_path()), ("best", _best_path())):
        if not state_path.exists():
            raise RuntimeError(f"checkpoint missing: {state_path}; run the train stage first")
        model = build_model(initialised=True)
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
        model.eval()
        payload[label] = {
            "state_path": str(state_path.relative_to(REPO_ROOT)),
            "state_sha256": core.state_sha256(model.state_dict()),
            "flag": {
                "D_node_norm": float(model.D_node.norm()),
                "D_rel_norm": float(model.D_rel.norm()),
            },
            "valid": _intervention_report(model, valid),
            "code_usage": _code_usage(model, valid),
        }
        print(
            f"[interventions:{label}] intact={payload[label]['valid']['intact']['mae']:.6f} "
            f"zero_beta={payload[label]['valid']['modes']['zero_beta']['delta_mae']:+.6f} "
            f"zero_node_pool="
            f"{payload[label]['valid']['modes']['zero_node_pool']['delta_mae']:+.6f} "
            f"zero_delta_rel="
            f"{payload[label]['valid']['modes']['zero_delta_rel']['delta_mae']:+.6f}",
            flush=True,
        )
    payload["sensitivity_summary"] = {
        name: {
            "sensitive": bool(row["sensitive"]),
            "delta_mae": float(row["delta_mae"]),
            "rms_delta_pred": float(row["mean_abs_delta_pred"]),
        }
        for name, row in payload["soup"]["valid"]["modes"].items()
    }
    _blocker(payload)
    _write_json(path, payload)
    return payload


BANDS = (
    (0.115, "strong", "strong signal; recommend follow-up confirmation"),
    (0.120, "promising", "promising; recommend follow-up confirmation"),
    (0.1233, "borderline", "borderline signal; no automatic follow-up"),
    (float("inf"), "stop", "stop the current configuration"),
)


def classify_band(soup_valid_mae: float) -> dict[str, Any]:
    for threshold, band, recommendation in BANDS:
        if float(soup_valid_mae) <= float(threshold):
            return {
                "band": band,
                "threshold": float(threshold),
                "recommendation": recommendation,
                "soup_valid_mae": float(soup_valid_mae),
            }
    raise RuntimeError("band classification failed")


def stage_analysis(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "summary.json"
    run = _read_json(_run_path())
    soup = _read_json(RESULTS_DIR / f"soup_{CANDIDATE}.json")
    interventions = _read_json(RESULTS_DIR / "interventions.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    init = _read_json(RESULTS_DIR / "hier_init.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    band = classify_band(float(soup["soup_valid_mae"]))
    curve = run["curve"]
    tail = [float(row["valid_mae"]) for row in curve[-20:]]
    node_used = bool(interventions["soup"]["valid"]["modes"]["zero_node_pool"]["sensitive"])
    delta_used = bool(interventions["soup"]["valid"]["modes"]["zero_delta_rel"]["sensitive"])
    beta_used = bool(interventions["soup"]["valid"]["modes"]["zero_beta"]["sensitive"])
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": ROUND,
        "candidate": CANDIDATE,
        "git_commit": _git_commit(),
        "seed": int(SEED),
        "epochs": int(TRAIN_EPOCHS),
        "epochs_run": int(run["epochs_run"]),
        "completed": bool(run["completed"]),
        "device": "cpu",
        "torch_threads": int(THREADS),
        "official_test_loaded": False,
        "band": band,
        "soup_valid_mae": float(soup["soup_valid_mae"]),
        "soup_train_mae": float(soup["soup_train_mae"]),
        "train_valid_gap": float(soup["soup_train_mae"] - soup["soup_valid_mae"]),
        "best_valid_mae": float(soup["best_valid_mae"]),
        "best_epoch": int(soup["best_epoch"]),
        "soup_members": list(soup["members"]),
        "soup_member_valid_mae": list(soup["member_valid_mae"]),
        "valid_first": float(curve[0]["valid_mae"]),
        "valid_last": float(curve[-1]["valid_mae"]),
        "valid_tail20_mean": float(np.mean(tail)),
        "train_min_mae": float(run["train_min_mae"]),
        "seconds_per_epoch": float(run["seconds_per_epoch"]),
        "wall_clock_s": float(run["wall_clock_s"]),
        "peak_rss_bytes": int(run["peak_rss_bytes"]),
        "parameters": run["parameters"],
        "mechanism": {
            "node_channel_load_bearing": node_used,
            "relation_delta_channel_load_bearing": delta_used,
            "relation_code_channel_load_bearing": beta_used,
            "any_relation_channel_load_bearing": bool(beta_used or delta_used),
            "rms_floor": float(interventions["rms_floor"]),
            "interventions": interventions["soup"]["valid"],
            "code_usage": interventions["soup"]["code_usage"],
        },
        "reconstruction": {
            "soup_train_r_node": float(run["soup"]["soup_r_node_train"]),
            "soup_train_r_rel": float(run["soup"]["soup_r_rel_train"]),
            "valid_r_node": float(interventions["soup"]["valid"]["intact"]["r_node"]),
            "valid_r_rel": float(interventions["soup"]["valid"]["intact"]["r_rel"]),
        },
        "probes": run["probes"],
        "init": {
            "node_dictionary": init["node_dictionary"],
            "relation_dictionary": init["relation_dictionary"],
            "relation_scaler": init["relation_scaler"],
            "projection": init["projection"],
            "init_state": init["init_state"],
        },
        "identity": {
            "init_state_sha256": run["init_state_sha256"],
            "soup_state_sha256": run["soup_state_sha256"],
            "best_state_sha256": run["best_state_sha256"],
            "correctness_all_passed": bool(correctness["all_passed"]),
            "smoke_passed": bool(smoke["passed"]),
            "smoke_best_valid_mae": float(smoke["best_valid_mae"]),
        },
        "verdict": "INCOMPLETE",
    }
    mechanism_note = (
        "relation dictionary carries prediction"
        if (beta_used or delta_used)
        else "relation dictionary does NOT carry prediction at the endpoint"
    )
    if band["band"] == "stop":
        payload["verdict"] = "HIERREL_STOP"
    else:
        payload["verdict"] = f"HIERREL_{band['band'].upper()}"
    payload["mechanism_note"] = mechanism_note
    payload["decision_rule"] = (
        f"soup valid MAE {soup['soup_valid_mae']:.6f} -> band {band['band']} "
        f"(threshold {band['threshold']}); mechanism judgement is reported separately"
    )
    _blocker(payload)
    _write_json(path, payload)
    _write_report(payload)
    _write_decision(payload)
    print(f"[analysis] verdict={payload['verdict']}", flush=True)
    return payload


def _write_report(summary: Mapping[str, Any]) -> None:
    band = summary["band"]
    mechanism = summary["mechanism"]
    probe_rows = []
    for epoch in sorted(summary["probes"], key=lambda v: int(v)):
        row = summary["probes"][epoch]
        probe_rows.append(
            f"| {epoch} | {row['mae_only_gradient']['grad_Dbar_node_norm']:.3e} | "
            f"{row['mae_only_gradient']['grad_Dbar_rel_norm']:.3e} | "
            f"{row['node_code']['norm_mean']:.4f} | {row['relation_code']['norm_mean']:.4f} | "
            f"{row['relation_code']['l0_mean']:.2f} | {row['reconstruction']['r_node']:.4f} | "
            f"{row['reconstruction']['r_rel']:.4f} | "
            f"{row['interventions']['zero_beta']['delta_mae']:+.5f} | "
            f"{row['interventions']['zero_node_pool']['delta_mae']:+.5f} | "
            f"{row['interventions']['zero_delta_rel']['delta_mae']:+.5f} |"
        )
    intervention_rows = []
    for name, row in summary["mechanism"]["interventions"]["modes"].items():
        intervention_rows.append(
            f"| `{name}` | {row['mae']:.6f} | {row['delta_mae']:+.6f} | "
            f"{row['pred_rms']:.4f} | {row['delta_pred_rms']:+.4f} | "
            f"{row['mean_abs_delta_pred']:.4f} | {'yes' if row['sensitive'] else 'no'} |"
        )
    text = f"""# REPORT — E2E-DictEnv-Hier-Relation-v1 (seed 0, local CPU)

Round: `{ROUND}` (`{PROTOCOL_VERSION}`); protocol `zinc-context-gap`;
authoring revision `{summary['git_commit']}`; device CPU
({summary['torch_threads']} threads); official ZINC **test never loaded**.

## 0. Decision

**`{summary['verdict']}`** — soup valid MAE **{summary['soup_valid_mae']:.6f}**
(best {summary['best_valid_mae']:.6f} @ epoch {summary['best_epoch']}) falls in the
pre-registered **`{band['band']}`** interval (threshold {band['threshold']}).
Mechanism (reported separately): {summary['mechanism_note']}.

## 1. Endpoint

| quantity | value |
|---|---|
| soup valid MAE | {summary['soup_valid_mae']:.9f} |
| soup members (Top-5 by valid MAE) | {summary['soup_members']} |
| member valid MAE | {[round(v, 6) for v in summary['soup_member_valid_mae']]} |
| best single epoch | {summary['best_valid_mae']:.9f} @ {summary['best_epoch']} |
| valid first / last / tail-20 mean | {summary['valid_first']:.6f} / {summary['valid_last']:.6f} / {summary['valid_tail20_mean']:.6f} |
| soup train MAE (official train 10000) | {summary['soup_train_mae']:.6f} |
| train–valid gap | {summary['train_valid_gap']:+.6f} |
| epochs run | {summary['epochs_run']} / {summary['epochs']} (completed={summary['completed']}) |
| seconds / epoch, wall clock | {summary['seconds_per_epoch']:.3f} s, {summary['wall_clock_s']:.1f} s |
| peak RSS | {summary['peak_rss_bytes'] / 1e9:.2f} GB |
| trainable parameters | {summary['parameters']['trainable']} (expected {summary['parameters']['expected_trainable']}) |
| buffer elements (P + scalers) | {summary['parameters']['buffer_elements']} |

## 2. Section-checked architecture

* 712-D node input `[Joint709 (709) ; common1 ; size2]`, frozen joint scaler reused
  bit-identically (joint block sha256 verified against the Joint709 cache).
* node dictionary `{summary['parameters']['d_x']} x {summary['parameters']['node_atoms']}`,
  all {summary['parameters']['node_atoms']} coordinates kept, column-normalised,
  no top-k, no message passing, no node write-back.
* relation object 228 dims = S(mu) 96 + S(delta) 96 + cross 32 + bond one-hot 4;
  one undirected physical edge counted once.
* relation dictionary 228 x {summary['parameters']['relation_atoms']},
  tied-IHT s={core.REL_SPARSITY}, steps={core.REL_IHT_STEPS}.
* readout {summary['parameters']['readout_dim']} =
  384 node (sum/mean/std) + 192 relation (sum/mean/std) + 2 log-counts + 8 topology.
* single head 586 -> 64 -> 32 -> 1 (`SiLU`, dropout 0.05).

## 3. Frozen train-only preparation

| item | value |
|---|---|
| P seed / shape / sha256 | {summary['init']['projection']['seed']} / {summary['init']['projection']['shape']} / `{summary['init']['projection']['sha256_f32'][:16]}…` |
| node dict init | {summary['init']['node_dictionary']['rows']} sampled train rows, effective rank {summary['init']['node_dictionary']['effective_rank']}, supplements {summary['init']['node_dictionary']['supplemented_columns']} |
| node dict init sha256 | `{summary['init']['node_dictionary']['state_sha256_f32'][:16]}…` |
| relation scaler fit rows (edges) | {summary['init']['relation_scaler']['n_fit_edges']} |
| relation scaler scaled block energy | {summary['init']['relation_scaler']['scaled_block_energy']} |
| relation dict init | {summary['init']['relation_dictionary']['sample_rows']} sampled train edges, effective rank {summary['init']['relation_dictionary']['effective_rank']}, supplements {summary['init']['relation_dictionary']['supplemented_columns']} |
| relation dict init sha256 | `{summary['init']['relation_dictionary']['state_sha256_f32'][:16]}…` |
| init state sha256 | `{summary['init']['init_state']['state_sha256'][:16]}…` |
| soup / best state sha256 | `{summary['identity']['soup_state_sha256'][:16]}…` / `{summary['identity']['best_state_sha256'][:16]}…` |

Statistics are fit once under the initial node dictionary and the fixed `P`, then
frozen: no training-time, checkpoint-time or validation-time refit.

## 4. Reconstruction, code usage and endpoint interventions

| quantity | train (soup) | valid (soup) |
|---|---|---|
| node relative reconstruction `R_N` | {summary['reconstruction']['soup_train_r_node']:.6f} | {summary['reconstruction']['valid_r_node']:.6f} |
| relation relative reconstruction `R_R` | {summary['reconstruction']['soup_train_r_rel']:.6f} | {summary['reconstruction']['valid_r_rel']:.6f} |

Code usage (official valid):
node code norm mean {mechanism['code_usage']['node_code_norm_mean']:.4f}, variance
{mechanism['code_usage']['node_code_variance_mean']:.4f}, active coordinate
rate {mechanism['code_usage']['node_coordinate_activation_rate']:.4f};
relation code norm mean {mechanism['code_usage']['relation_code_norm_mean']:.4f},
`l0` mean {mechanism['code_usage']['relation_code_l0_mean']:.2f},
top-1 share {mechanism['code_usage']['relation_code_top1_share']:.4f},
effective atoms {mechanism['code_usage']['relation_effective_atoms']:.2f} /
{summary['parameters']['relation_atoms']}.

Endpoint interventions on the **soup** state (official valid 1000; pre-registered
`RMS > {summary['mechanism']['rms_floor']:.0e}` = clear sensitivity,
which is not by itself a performance claim):

| intervention | valid MAE | ΔMAE | pred RMS | ΔRMS | mean abs Δpred | sensitive |
|---|---|---|---|---|---|---|
{chr(10).join(intervention_rows)}

Mechanism flags: node channel load-bearing
**{mechanism['node_channel_load_bearing']}**, delta channel
**{mechanism['relation_delta_channel_load_bearing']}**, relation code channel
**{mechanism['relation_code_channel_load_bearing']}**.

## 5. Activity probe (fixed train probe, no parameter update)

| epoch | MAE-only grad Dbar_N | MAE-only grad Dbar_R | z norm | beta norm | beta l0 | R_N | R_R | ΔMAE zero_beta | ΔMAE zero_node_pool | ΔMAE zero_delta |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(probe_rows)}

## 6. Provenance and reproduction

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_hier_relation_v1.py
uv run research run zinc_e2e_dictenv_hier_relation_v1 \\
  --study zinc-context-gap --mode scratch \\
  --purpose "Hierarchical static relation dictionary train-only preparation, CPU" \\
  --set runtime.device=cpu --set model.stage=prepare
uv run research run zinc_e2e_dictenv_hier_relation_v1 \\
  --study zinc-context-gap --mode screen \\
  --purpose "Single-model hierarchical relation dictionary absolute-performance screen, CPU" \\
  --set runtime.device=cpu --set model.stage=screen
```
"""
    (RESULTS_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    band = summary["band"]
    mechanism = summary["mechanism"]
    text = f"""# Decision — E2E-DictEnv-Hier-Relation-v1

- candidate: `{CANDIDATE}` (one candidate, seed 0, {summary['epochs']} epochs, local CPU)
- soup valid MAE: **{summary['soup_valid_mae']:.9f}** (best {summary['best_valid_mae']:.9f} @ {summary['best_epoch']})
- band: `{band['band']}` — {band['recommendation']}
- official train soup MAE {summary['soup_train_mae']:.9f} (gap {summary['train_valid_gap']:+.6f})
- official test: never loaded

Mechanism (separate statement, not a band change): {summary['mechanism_note']}
(node channel load-bearing = {mechanism['node_channel_load_bearing']}, delta channel
= {mechanism['relation_delta_channel_load_bearing']}, relation code channel =
{mechanism['relation_code_channel_load_bearing']}).

## Not claimed by this round

- no matched-training control (no dense/PCA arm, no random/shuffled-dictionary arm);
- no sparse-vs-dense or dictionary-vs-no-dictionary causal statement;
- no statistical-significance claim (single seed);
- no terminal (official-test) statement;
- no statement about any previous round's architecture (A/B/C, Sem108, Joint709 are
  background only, and this round's numbers are not a matched comparison to them).

## Forbidden without a new pre-registration

- seed 1, any width / K / s / epoch / LR / lambda change;
- touching the official ZINC test split;
- re-using these numbers as a matched comparison against the older runs.
"""
    (RESULTS_DIR / "DECISION.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# stage table
# ---------------------------------------------------------------------------

PREPARE_STAGES = (
    "source_verify",
    "node_input",
    "relation_structure",
    "align_check",
    "dictionary_init",
)
SCREEN_STAGES = ("correctness", "smoke", "train", "interventions", "analysis")


def run_stages(stage: str, only: Sequence[str] | None = None) -> dict[str, Any]:
    _ensure_dirs()
    print(
        f"[runner] commit={_git_commit()} protocol={PROTOCOL_VERSION} "
        f"stage={stage} device={_DEVICE} threads={THREADS}",
        flush=True,
    )
    table = {
        "source_verify": stage_source_verify,
        "node_input": stage_node_input,
        "relation_structure": stage_relation_structure,
        "align_check": stage_align_check,
        "dictionary_init": stage_dictionary_init,
        "correctness": stage_correctness,
        "smoke": stage_smoke,
        "train": stage_train,
        "interventions": stage_interventions,
        "analysis": stage_analysis,
    }
    names = list(PREPARE_STAGES if stage == "prepare" else SCREEN_STAGES)
    if only:
        names = [name for name in names if name in set(only)]
    results: dict[str, Any] = {}
    for name in names:
        results[name] = table[name]()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-Hier-Relation-v1 stage runner")
    parser.add_argument("--stage", required=True, choices=("prepare", "screen", "all"))
    parser.add_argument("--only", default=None, help="comma-separated stage subset override")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    configure(device=args.device)
    only = [part.strip() for part in args.only.split(",") if part.strip()] if args.only else None
    if args.stage in ("prepare", "all"):
        run_stages("prepare", only)
    if args.stage in ("screen", "all"):
        run_stages("screen", only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
