"""E2E-DictEnv-A2 runner — Attributed Code Value & Compression Control (ZINC).

Round ``e2e_dictenv_a2``.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md`` (frozen).
Prior-artifact audit: ``tracks/ksvd/notes/e2e_dictenv_a2_prior_artifact_audit.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_a2.py``.

Stage order (each stage is resumable and refuses to run out of order):

``identity qualify continuity-v2 omp-screen omp-decision [compression
  compression-decision] [coder] [formal mechanism liveness] [specificity]
  decision report``

``identity`` and ``qualify`` re-verify and re-audit the **A1** artifacts
(bit-identical reuse), ``omp-screen``/``formal``/``specificity`` drive the frozen
A1 trainer unchanged (only the tag names and the output directory differ), and
``compression`` adds the single new scientific object of the round: a train-only
dense rank-32 compression reference.  Continuity-v2 is a *diagnostic* — it can
neither stop nor select anything.

Official ZINC **test is never loaded**; the shared P1 loader refuses
``split == "test"`` and every artifact records ``official_test_loaded = false``.
All CUDA work runs on physical **GPU1** only (``CUDA_VISIBLE_DEVICES == "1"``,
enforced); GPU0 is foreign-owned and is never touched.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2 as a2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as a1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = a2.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_a2"
STATE_DIR = RESULTS_DIR / "states"
CURVE_DIR = RESULTS_DIR / "curves"
FORMAL_DIR = RESULTS_DIR / "formal_runs"
A1_RESULTS_DIR = a1run.RESULTS_DIR
A1_CACHE_DIR = a1run.CACHE_DIR

#: durable digests pinned by tracked files (A1 decision record + STATE.yaml +
#: the A1 runner's own TOPO constant); see the audit §4.
EXPECTED_DICT_SHA: dict[str, str] = {
    "TOPO": a1run.SDB_DICT_SHA256_F32,
    "INDEP": "400821ee5105050eb34600a0fb4cb8040f983daa1e773738dc9e2774036ff32d",
    "REAL": "c1cafb086662fb0753d52987fc321b1369d4f586164dde7960a3bed775d4b809",
}
#: tracked prefixes quoted in notes/e2e_dictenv_a1_analysis.md §2
TRACKED_CACHE_PREFIXES: dict[str, dict[str, str]] = {
    "train": {"phi": "f8641dab", "joint_v": "206f4958", "joint_e": "f3559ca8",
              "marginal_v": "40651b98", "marginal_e": "67dee4ca"},
    "valid": {"phi": "33db50b8", "joint_v": "1599e147", "joint_e": "dd77a9ec",
              "marginal_v": "95469216", "marginal_e": "616372a9"},
}
#: cross-checks quoted in the same tracked note / A1 cache metadata
TRACKED_CACHE_SHAPES = {
    "train": {"n_molecules": 10000, "n_nodes": 231664, "sum_n_patch": 1418500},
    "valid": {"n_molecules": 1000, "n_nodes": 23083},
}

#: frozen trainer configuration (only identifiers, no new hyper-parameters)
SEED = 0
BATCH_SIZE = int(p1run.BATCH_SIZE)
LIVENESS_TRAIN_ROWS = 8192
LIVENESS_VALID_ROWS = 2048
LIVENESS_PROBE_MOLECULES = 256
LIVENESS_SEED = 20260930

STAGE_STATUS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mark(stage: str, status: str, *, reason: str = "", artifact: str | None = None,
          seconds: float | None = None, device: str | None = None) -> None:
    STAGE_STATUS[stage] = {
        "status": str(status),
        "reason": str(reason),
        "artifact": None if artifact is None else str(artifact),
        "seconds": None if seconds is None else float(seconds),
        "device": None if device is None else str(device),
    }


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _prereg_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%h", "--",
             "tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def provenance(device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    payload: dict[str, Any] = {
        "round": a2.ROUND,
        "protocol_version": PROTOCOL_VERSION,
        "subtitle": a2.SUBTITLE,
        "study": a2.STUDY,
        "git_commit": v0run._git_commit(),
        "preregistration_commit": _prereg_commit(),
        "branch": _git_branch(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "physical_gpu_requested": 1,
        "device": str(device_obj),
        "seed": SEED,
        "official_test_loaded": False,
    }
    if device_obj.type == "cuda":
        payload["gpu_model"] = torch.cuda.get_device_name(device_obj)
        payload["cuda"] = str(torch.version.cuda)
        payload["cuda_visible_devices"] = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        payload["peak_gpu_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)
        )
    return payload


def _set_device_policy(device: str) -> torch.device:
    """Hard GPU policy: CUDA work happens on physical GPU1 and nowhere else."""
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
        if visible == "":
            raise RuntimeError(
                "E2E-DictEnv-A2 requires CUDA_VISIBLE_DEVICES=1 (GPU0 is foreign-occupied)"
            )
        if visible != "1":
            raise RuntimeError(
                f"E2E-DictEnv-A2 forbids CUDA_VISIBLE_DEVICES={visible!r}; expected '1'"
            )
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device_obj)
    return device_obj


_write_json = v0run._write_json
_read_json = v0run._read_json


@contextlib.contextmanager
def a1_emits_into(root: Path) -> Iterator[None]:
    """Make the frozen A1 runner write its artifacts under ``root``.

    Only the *output* directories are redirected (``RESULTS_DIR``,
    ``STATE_DIR``, ``CURVE_DIR``, ``FORMAL_DIR``); the input paths
    (``CACHE_DIR`` / ``RAW_CACHE`` / ``SCALER_JSON`` / ``DICT_STORE``) stay
    pointed at the identity-verified A1 artifacts, so the reused round is read
    but never rewritten.
    """
    root = Path(root)
    mapping = {
        "RESULTS_DIR": root,
        "STATE_DIR": root / "states",
        "CURVE_DIR": root / "curves",
        "FORMAL_DIR": root / "formal_runs",
    }
    saved = {name: getattr(a1run, name) for name in mapping}
    for name, value in mapping.items():
        setattr(a1run, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(a1run, name, value)


@contextlib.contextmanager
def dictionary_override(arm: str, dictionary: np.ndarray) -> Iterator[None]:
    """Serve one frozen dictionary to the A1 trainer for one arm (Stage 2)."""
    original = a1run.load_arm_dictionary
    frozen = np.asarray(dictionary, dtype=np.float32)

    def patched(requested: str):
        if str(requested) == str(arm):
            return frozen, a2.a1_hash(frozen)
        return original(requested)

    a1run.load_arm_dictionary = patched
    try:
        yield
    finally:
        a1run.load_arm_dictionary = original


@contextlib.contextmanager
def matched_init_enforced(arm: str = "REAL") -> Iterator[dict[str, Any]]:
    """Enforce pre-registration §11: one reference state for every arm.

    ``a1.build_model`` reseeds deterministically per arm, so tensors whose shapes
    coincide across arms already start bit-identical; this wrapper *enforces* that
    by handing the frozen ``build_model``'s own ``reference_state`` copy the
    reference arm's state, instead of assuming it.  The reference arm's ``D`` is
    never shared (each arm keeps its own frozen dictionary), and tensors whose
    shapes differ keep their own initialisation.

    ``identity_stage`` verifies that this enforcement is a *no-op* on the frozen
    A1 initialisation and that the shared-shape tensors really are bit-identical
    across arms, so the enforcement can never silently move the numbers.
    """
    info: dict[str, Any] = {"reference_arm": str(arm)}
    reference = reference_state(arm)
    info["reference_keys"] = len(reference)
    original = a1.build_model

    def patched(model_arm: str, dictionary: np.ndarray, **kwargs: Any):
        kwargs.setdefault("reference_state", reference)
        return original(model_arm, dictionary, **kwargs)

    a1.build_model = patched  # type: ignore[assignment]
    try:
        yield info
    finally:
        a1.build_model = original  # type: ignore[assignment]


def reference_state(arm: str = "REAL") -> dict[str, torch.Tensor]:
    """Detached CPU state dict of one arm's frozen-init model (seed 0).

    ``D`` is removed: the dictionary is each arm's own frozen K-SVD input object
    (the REAL and INDEP dictionaries have the same shape and would otherwise
    overwrite each other), whereas every other tensor is an initialisation that
    must be identical across arms.
    """
    D, _sha = a1run.load_arm_dictionary(arm)
    model = a1.build_model(arm, D, seed=SEED)
    return {
        key: value.detach().to("cpu", copy=True)
        for key, value in model.state_dict().items()
        if key != "D"
    }


def matched_init_report(reference_arm: str = "REAL") -> dict[str, Any]:
    """Prove §11's matched initialisation on the reused objects (identity gate)."""
    reference = reference_state(reference_arm)
    per_arm: dict[str, Any] = {}
    for arm in a1.ARMS:
        D, _sha = a1run.load_arm_dictionary(arm)
        plain = a1.build_model(arm, D, seed=SEED)
        enforced = a1.build_model(arm, D, seed=SEED, reference_state=reference)
        plain_state, enforced_state = plain.state_dict(), enforced.state_dict()
        shared = [
            key
            for key, value in reference.items()
            if key in plain_state and plain_state[key].shape == value.shape
        ]
        noop = all(
            torch.equal(plain_state[key], enforced_state[key]) for key in enforced_state
        )
        matches_reference = all(torch.equal(enforced_state[key], reference[key]) for key in shared)
        dictionary_kept = torch.equal(
            enforced_state["D"], torch.as_tensor(a1run.load_arm_dictionary(arm)[0])
        )
        per_arm[arm] = {
            "passed": bool(noop and matches_reference and dictionary_kept),
            "shared_shape_tensors": len(shared),
            "enforcement_is_a_noop": bool(noop),
            "matches_reference_state": bool(matches_reference),
            "D_kept_per_arm": bool(dictionary_kept),
            "total_tensors": len(enforced_state),
        }
    return {
        "reference_arm": str(reference_arm),
        "arms": per_arm,
        "passed": bool(all(entry["passed"] for entry in per_arm.values())),
        "policy": (
            "same seed-0 initial values for every tensor whose shape coincides across "
            "arms (all except D); D is each arm's own frozen K-SVD object"
        ),
    }


def _reemit(payload: Mapping[str, Any], *, source: str, **extra: Any) -> dict[str, Any]:
    """Add A2 provenance to a payload produced by the frozen A1 machinery."""
    out = dict(payload)
    out["source_protocol_version"] = out.get("protocol_version")
    out["reused_from"] = str(source)
    out["round"] = a2.ROUND
    out["protocol_version"] = PROTOCOL_VERSION
    out["subtitle"] = a2.SUBTITLE
    out["preregistration_commit"] = _prereg_commit()
    out["official_test_loaded"] = False
    out.update(extra)
    return out


def _curve_is_finite(path: Path) -> bool:
    """Fail-closed: a missing, empty or non-finite training curve is *not* finite.

    ``train_arm`` writes the state and the curve before the run JSON, so a
    present run JSON always implies a present curve; a missing curve therefore
    means the artifacts were cleaned or tampered with.
    """
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if not header or header == [""]:
            return False
        seen = 0
        for line in handle:
            if not line.strip():
                continue
            fields = line.strip().split(",")
            for name, value in zip(header, fields):
                if name == "epoch":
                    continue
                try:
                    if not np.isfinite(float(value)):
                        return False
                except ValueError:
                    return False
            seen += 1
    return seen > 0


def _run_is_finite(tag: str) -> bool:
    return _curve_is_finite(CURVE_DIR / f"{tag}_curve.csv")


# ---------------------------------------------------------------------------
# stage: identity (blocking)
# ---------------------------------------------------------------------------


def identity_stage(rows: int = 0, write: bool = True) -> dict[str, Any]:
    """Verify every reused A1 artifact against a durable or recomputed identity."""
    started = time.perf_counter()
    entries: dict[str, Any] = {}
    failures: list[str] = []

    def record(name: str, *, passed: bool, **fields: Any) -> None:
        entries[name] = {"passed": bool(passed), **fields}
        if not passed:
            failures.append(name)

    # --- dictionaries -------------------------------------------------------
    for arm in a1.ARMS:
        D, sha = a1run.load_arm_dictionary(arm)
        expected = EXPECTED_DICT_SHA[arm]
        record(
            f"dictionary_{arm}",
            passed=bool(sha == expected),
            source="a1run.load_arm_dictionary",
            expected_sha256_f32=expected,
            observed_sha256_f32=sha,
            shape=[int(D.shape[0]), int(D.shape[1])],
            pinned_in=(
                "tracked code constant + A1 decision record"
                if arm == "TOPO"
                else "tracked A1 decision record + STATE.yaml"
            ),
        )

    # --- raw block caches ---------------------------------------------------
    for split in ("train", "valid"):
        meta = _read_json(A1_RESULTS_DIR / f"cache_meta_{split}.json")
        raw = a1run.load_raw(split)
        for key in ("phi", "joint_v", "joint_e", "marginal_v", "marginal_e"):
            observed = a1run._sha256_array(raw[key])
            expected = meta["cache_sha256"][key]
            prefix_ok = observed.startswith(TRACKED_CACHE_PREFIXES[split][key])
            record(
                f"cache_{split}_{key}",
                passed=bool(observed == expected and prefix_ok),
                source=str(A1_RESULTS_DIR / f"cache/a1_raw_{split}.pt"),
                expected_sha256=expected,
                observed_sha256=observed,
                tracked_prefix=TRACKED_CACHE_PREFIXES[split][key],
                tracked_prefix_matches=bool(prefix_ok),
                shape=[int(raw[key].shape[0]), int(raw[key].shape[1])],
                pinned_in="A1 cache metadata (gitignored) + tracked prefixes in notes/e2e_dictenv_a1_analysis.md",
            )
        checks = {
            "n_molecules": int(raw["node_sizes"].shape[0]),
            "n_nodes": int(raw["phi"].shape[0]),
            "sum_n_patch": int(np.sum(raw["n_patch"])),
        }
        expected_shapes = TRACKED_CACHE_SHAPES[split]
        shape_ok = all(checks[key] == value for key, value in expected_shapes.items())
        record(
            f"cache_{split}_shape_crosscheck",
            passed=bool(shape_ok),
            observed=checks,
            expected=expected_shapes,
            pinned_in="tracked notes/e2e_dictenv_a1_analysis.md §2 / cache metadata",
        )

    # --- scaler: exact deterministic recomputation from the raw train cache --
    raw_train = a1run.load_raw("train")
    stored_scalers = a1run.load_scalers()
    for coordinate in ("real", "indep"):
        recomputed = a1.fit_object_scaler(coordinate, raw_train)
        stored = stored_scalers[coordinate]
        scale_diffs = []
        mask_ok = True
        weight_ok = True
        for block in a1.BLOCKS:
            left, right = recomputed.blocks[block], stored.blocks[block]
            if left.scale.shape != right.scale.shape:
                scale_diffs.append(float("inf"))
                continue
            scale_diffs.append(float(np.max(np.abs(left.scale - right.scale))))
            mask_ok = mask_ok and bool(np.array_equal(left.mask, right.mask))
            weight_ok = weight_ok and bool(float(left.weight) == float(right.weight))
        max_diff = max(scale_diffs) if scale_diffs else float("inf")
        record(
            f"scaler_{coordinate}",
            passed=bool(max_diff == 0.0 and mask_ok and weight_ok),
            source=str(A1_RESULTS_DIR / "scaler_meta.json"),
            max_abs_scale_difference=float(max_diff),
            mask_identical=bool(mask_ok),
            block_weight_identical=bool(weight_ok),
            masked_coordinates={
                block: int(stored.blocks[block].masked_coordinates) for block in a1.BLOCKS
            },
            verification="exact recomputation with a1.fit_object_scaler on the raw train cache",
        )
    del raw_train

    # --- exact-OMP codes: deterministic recomputation ------------------------
    for arm in a1.ARMS:
        D, _sha = a1run.load_arm_dictionary(arm)
        Dbar = sdb.normalize_columns(np.asarray(D, dtype=np.float64))
        for split in ("train", "valid"):
            path = A1_CACHE_DIR / f"omp_{arm}_{split}.pt"
            stored = torch.load(path, map_location="cpu", weights_only=True).numpy()
            X = np.asarray(a1run.arm_coordinate(arm, split))
            subset = X if int(rows) <= 0 else X[: int(rows)]
            recomputed = sdb.omp_codes(Dbar, subset, s=a1.DICT_S).astype(np.float32)[
                : (stored.shape[0] if int(rows) <= 0 else min(int(rows), stored.shape[0]))
            ]
            reference = stored if int(rows) <= 0 else stored[: recomputed.shape[0]]
            identical = bool(np.array_equal(reference, recomputed))
            record(
                f"omp_{arm}_{split}",
                passed=bool(identical),
                source=str(path),
                rows_verified=int(reference.shape[0]),
                rows_total=int(stored.shape[0]),
                full_split=bool(int(rows) <= 0),
                verification="exact recomputation sdb.omp_codes(normalize_columns(D), X, s=8)",
            )

    # --- matched initialisation (pre-registration §11), enforced and verified --
    init = matched_init_report()
    for arm, entry in init["arms"].items():
        record(
            f"matched_init_{arm}",
            passed=bool(entry["passed"]),
            source="a1.build_model(seed=0) with and without reference_state",
            shared_shape_tensors=int(entry["shared_shape_tensors"]),
            enforcement_is_a_noop=bool(entry["enforcement_is_a_noop"]),
            matches_reference_state=bool(entry["matches_reference_state"]),
            reference_arm=init["reference_arm"],
        )

    payload = {
        **provenance("cpu"),
        "kind": "artifact identity (blocking gate)",
        "identity_rows_verified": None if int(rows) <= 0 else int(rows),
        "matched_init": init,
        "entries": entries,
        "failures": failures,
        "all_passed": bool(not failures),
        "seconds": float(time.perf_counter() - started),
    }
    if write:
        _write_json(RESULTS_DIR / "artifact_identity.json", payload)
        _mark("identity", "RUN", artifact="artifact_identity.json", seconds=payload["seconds"], device="cpu",
              reason="all reused artifacts verified" if payload["all_passed"] else "identity failures")
    return payload


# ---------------------------------------------------------------------------
# stage: qualification (correctness / assignment / health / accounting)
# ---------------------------------------------------------------------------


def qualify_stage() -> dict[str, Any]:
    """Run the frozen A1 Gate-0 audits against the reused artifacts."""
    started = time.perf_counter()
    with a1_emits_into(RESULTS_DIR):
        correctness = a1run.correctness_stage()
        assignment = a1run.assignment_stage()
        health = a1run.health_stage()
        accounting = a1run.accounting_stage()
    for name, payload, source in (
        ("correctness.json", correctness, "zinc_e2e_dictenv_a1.correctness_stage"),
        ("assignment_semantics.json", assignment, "zinc_e2e_dictenv_a1.assignment_stage"),
        ("dictionary_health.json", health, "zinc_e2e_dictenv_a1.health_stage"),
        ("parameter_accounting.json", accounting, "zinc_e2e_dictenv_a1.accounting_stage"),
    ):
        _write_json(RESULTS_DIR / name, _reemit(payload, source=source))
    for arm in a1.ARMS:
        arm_path = RESULTS_DIR / f"dictionary_health_{arm.lower()}.json"
        if arm_path.exists():
            _write_json(
                arm_path,
                _reemit(_read_json(arm_path), source="zinc_e2e_dictenv_a1.health_stage"),
            )
    payload = {
        "correctness_all_passed": bool(correctness.get("all_passed")),
        "assignment_passed": bool(assignment.get("passed")),
        "health_arms": {
            arm: {
                "omp_normalized_err_holdout": float(health["arms"][arm]["omp_normalized_err_holdout"]),
                "used_atoms": int(health["arms"][arm]["usage_fit"]["used_atoms"]),
                "effective_atoms": float(health["arms"][arm]["usage_fit"]["effective_atom_count"]),
                "train_valid_usage_spearman": float(
                    health["arms"][arm]["train_valid_usage_spearman"]
                ),
            }
            for arm in a1.ARMS
        },
        "official_test_blocker_passed": bool(
            correctness["gates"]["G0k_official_test_blocker"]["passed"]
        ),
        "parameter_accounting_passed": bool(accounting.get("all_passed")),
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    payload["passed"] = bool(
        payload["correctness_all_passed"]
        and payload["assignment_passed"]
        and payload["official_test_blocker_passed"]
        and payload["parameter_accounting_passed"]
    )
    _write_json(RESULTS_DIR / "qualification.json", payload)
    _mark(
        "qualify",
        "RUN",
        artifact="qualification.json",
        seconds=payload["seconds"],
        device="cpu",
        reason="gate-0 audits pass" if payload["passed"] else "gate-0 audit failure",
    )
    return payload


# ---------------------------------------------------------------------------
# stage: continuity-v2 (diagnostic only)
# ---------------------------------------------------------------------------


def _sample_pool(n_molecules: int, node_sizes: np.ndarray, n_pool: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    pool: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pool) < int(n_pool):
        molecule_index = int(rng.integers(int(n_molecules)))
        size = int(node_sizes[molecule_index])
        root = int(rng.integers(size))
        if (molecule_index, root) in seen:
            continue
        seen.add((molecule_index, root))
        pool.append((molecule_index, root))
    return pool


def _pool_frames(pool: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Attributed-WL fingerprints, canonical keys, sizes and per-arm rows."""
    from tracks.ksvd.code import tccd_v0 as tccd
    from tracks.ksvd.code.run_tccd_v0 import _patch_graph_for_wl
    from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (
        _histogram_matrix,
        attributed_wl_fingerprint,
    )

    raw = a1run.load_raw("train")
    node_sizes = raw["node_sizes"]
    offsets = np.concatenate([[0], np.cumsum(node_sizes)])
    raw_molecules = a1run._raw_molecules("train")
    widths = {arm: a1.ARM_INPUT_DIM[arm] for arm in a1.ARMS}
    x_by_arm = {arm: np.asarray(a1run.arm_coordinate(arm, "train")) for arm in a1.ARMS}
    codes_by_arm = {arm: np.asarray(a1run.load_or_build_codes(arm, "train")) for arm in a1.ARMS}

    molecules: dict[int, Any] = {}
    hists, sizes, rootcats, key_list, key_ids, mol_ids = [], [], [], [], [], []
    rows_by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in a1.ARMS}
    size_mismatch = 0
    for molecule_index, root in pool:
        if molecule_index not in molecules:
            molecules[molecule_index] = a1run._mol_like(raw_molecules[molecule_index])
        mol = molecules[molecule_index]
        adjacency = tccd._adjacency(mol)
        graph, node_types, edge_types = _patch_graph_for_wl(mol, root, adjacency)
        hists.append(
            attributed_wl_fingerprint(graph, node_types, edge_types, rounds=a2.CONTINUITY_V2_ROUNDS)
        )
        sizes.append(int(graph.n))
        rootcats.append(int(mol.node_types[root]))
        _slot_nodes, _slot_shells, key = tccd.patch_slot_order(mol, root, adj=adjacency)
        key_list.append(key)
        mol_ids.append(int(molecule_index))
        row = int(offsets[molecule_index]) + int(root)
        if int(raw["n_patch"][row]) != int(graph.n):
            size_mismatch += 1
        for arm in a1.ARMS:
            rows_by_arm[arm].append(
                np.concatenate([x_by_arm[arm][row], codes_by_arm[arm][row]], axis=0)
            )
    vocab = sorted({k for hist in hists for k in hist}, key=repr)
    features = _histogram_matrix(hists, vocab)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -1.0)
    key_index = {key: index for index, key in enumerate(sorted(set(key_list), key=repr))}
    key_ids = [key_index[key] for key in key_list]
    return {
        "pool": list(pool),
        "similarity": similarity,
        "sizes": np.asarray(sizes, dtype=np.int64),
        "rootcats": np.asarray(rootcats, dtype=np.int64),
        "molecules": np.asarray(mol_ids, dtype=np.int64),
        "key_ids": np.asarray(key_ids, dtype=np.int64),
        "key_list": key_list,
        "widths": widths,
        "rows": {arm: np.stack(rows_by_arm[arm], 0) for arm in a1.ARMS},
        "n_zero_size_mismatch": int(size_mismatch),
        "wl_vocab": len(vocab),
    }


def _pool_geometry(frames: Mapping[str, Any], pairs: np.ndarray) -> dict[str, Any]:
    similarity = np.asarray(frames["similarity"], dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    similarities = (
        np.asarray([similarity[i, j] for i, j in pairs], dtype=np.float64)
        if pairs.shape[0]
        else np.zeros(0, dtype=np.float64)
    )
    out: dict[str, Any] = {}
    for arm in a1.ARMS:
        width = int(frames["widths"][arm])
        rows = np.asarray(frames["rows"][arm], dtype=np.float64)
        x_space, code_space = rows[:, :width], rows[:, width:]
        out[arm] = {
            "x": a2.graded_geometry(x_space, pairs, similarities),
            "code": a2.graded_geometry(code_space, pairs, similarities),
        }
    return out


def _same_partition(left: np.ndarray, right: np.ndarray) -> bool:
    """True when two integer labelings induce the same partition of the pool."""
    left = np.asarray(left, dtype=np.int64).reshape(-1)
    right = np.asarray(right, dtype=np.int64).reshape(-1)
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left[:, None] == left[None, :], right[:, None] == right[None, :]))


def _contrasts(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Paired rho contrasts per space and stratum (geometry is stratum-major)."""
    out: dict[str, Any] = {}
    for space in ("x", "code"):
        out[space] = {}
        for stratum in geometry:
            def rho(arm: str) -> float:
                return float(geometry[stratum][arm][space]["rho_angular"])

            out[space][stratum] = {
                "REAL_minus_INDEP": rho("REAL") - rho("INDEP"),
                "REAL_minus_TOPO": rho("REAL") - rho("TOPO"),
                "INDEP_minus_TOPO": rho("INDEP") - rho("TOPO"),
            }
    return out


def continuity_v2_stage() -> dict[str, Any]:
    """Corrected graded continuity diagnostic (never a gate, never a selector)."""
    started = time.perf_counter()
    raw = a1run.load_raw("train")
    node_sizes = raw["node_sizes"]
    n_molecules = int(node_sizes.shape[0])

    pool = _sample_pool(n_molecules, node_sizes, a2.CONTINUITY_V2_POOL, a2.CONTINUITY_V2_SEED)
    frames = _pool_frames(pool)
    pairs = a2.equal_size_molecule_disjoint_pairs(
        frames["sizes"],
        frames["molecules"],
        frames["key_ids"],
        cap_per_stratum=a2.CONTINUITY_V2_PAIRS_PER_STRATUM,
        seed=a2.CONTINUITY_V2_SEED,
    )
    geometry = {
        name: _pool_geometry(frames, subset) for name, subset in pairs["strata"].items()
    }
    geometry["pooled"] = _pool_geometry(frames, pairs["pooled"])
    payload: dict[str, Any] = {
        **provenance("cpu"),
        "kind": "diagnostic (not a gate; cannot stop or select anything)",
        "pool": {
            "n_pool": int(len(pool)),
            "seed": int(a2.CONTINUITY_V2_SEED),
            "rounds": int(a2.CONTINUITY_V2_ROUNDS),
            "source": "official train",
            "wl_vocab": int(frames["wl_vocab"]),
            "n_patch_size_mismatch": int(frames["n_zero_size_mismatch"]),
        },
        "strata_definition": a2.stratum_table(),
        "pair_population": {
            key: value for key, value in pairs.items() if key not in ("strata", "pooled", "stratum_counts")
        },
        "stratum_counts": pairs["stratum_counts"],
        "geometry": geometry,
        "contrasts": _contrasts(geometry),
        "primary_statistic": "Spearman(attributed_WL_similarity, -angular_distance)",
        "official_test_loaded": False,
    }

    # --- reconciliation with the A1 post-hoc populations (descriptive only) --
    try:
        a1_data = a1run.continuity_pool_data()
        a1_pool = list(a1_data["pool"])
        a1_key_ids = {key: index for index, key in enumerate(a1_data["keys"])}
        a1_sizes = np.asarray(a1_data["sizes"], dtype=np.int64)
        a1_molecules = np.asarray([mol for mol, _root in a1_pool], dtype=np.int64)
        a1_key_ids_arr = np.asarray([a1_key_ids[key] for key in a1_data["keys"]], dtype=np.int64)
        a1_similarity = np.asarray(a1_data["similarity"], dtype=np.float64)
        a1_widths = {"TOPO": a1.PHI_DIM, "INDEP": a1.A1_DIM, "REAL": a1.A1_DIM}
        sample = a2.reconstruct_a1_posthoc_pairs(
            a1_sizes,
            a1_molecules,
            a1_key_ids_arr,
            samples=a2.CONTINUITY_A1_SAMPLED_PAIRS,
            seed=a2.CONTINUITY_A1_SEED + a2.CONTINUITY_A1_POSTHOC_SEED_OFFSET,
        )
        a1_pairs = sample["pairs"]
        a1_sims = np.asarray([a1_similarity[i, j] for i, j in a1_pairs], dtype=np.float64)
        strict = sample["equal_size_molecule_disjoint"]
        reconciliation: dict[str, Any] = {
            "pool": int(a2.CONTINUITY_A1_POOL),
            "seed": int(a2.CONTINUITY_A1_SEED),
            "n_sampled_pairs": int(a1_pairs.shape[0]),
            "n_equal_size": int(sample["equal_size"].sum()),
            "n_equal_size_molecule_disjoint": int(strict.sum()),
            "note": "reproduces the A1 post-hoc populations from the audit §3; descriptive only",
            "arms": {},
        }
        for arm in a1.ARMS:
            rows = np.asarray(a1_data["rows"][arm], dtype=np.float64)
            width = int(a1_widths[arm])
            x_space, code_space = rows[:, :width], rows[:, width:]
            reconciliation["arms"][arm] = {
                "all_pairs": {
                    "x": a2.graded_geometry(x_space, a1_pairs, a1_sims),
                    "code": a2.graded_geometry(code_space, a1_pairs, a1_sims),
                },
                "equal_size_molecule_disjoint": {
                    "x": a2.graded_geometry(x_space, a1_pairs[strict], a1_sims[strict]),
                    "code": a2.graded_geometry(code_space, a1_pairs[strict], a1_sims[strict]),
                },
            }
        payload["reconciliation_a1_pool"] = reconciliation
        # --- the *pipeline* check: our attributed-WL / slot-key reconstruction of
        # the A1 pool must reproduce A1's own similarity matrix and key partition
        a1_frames = _pool_frames(a1_pool)
        upper = np.triu_indices(int(a1_similarity.shape[0]), k=1)
        reconciliation["pipeline"] = {
            "n_patches": int(a1_frames["similarity"].shape[0]),
            "sizes_identical": bool(np.array_equal(a1_frames["sizes"], a1_sizes)),
            "molecules_identical": bool(np.array_equal(a1_frames["molecules"], a1_molecules)),
            "key_partition_identical": bool(
                _same_partition(a1_frames["key_ids"], a1_key_ids_arr)
            ),
            "max_abs_similarity_difference": float(
                np.abs(
                    np.asarray(a1_frames["similarity"], dtype=np.float64)[upper]
                    - a1_similarity[upper]
                ).max()
            ),
            "n_patch_size_mismatch": int(a1_frames["n_zero_size_mismatch"]),
            "wl_vocab_our": int(a1_frames["wl_vocab"]),
            "note": (
                "A2's attributed-WL + patch_slot_order pipeline re-derives A1's own "
                "pool similarity matrix and key partition"
            ),
        }
        del a1_frames
        del a1_data
    except Exception as error:  # pragma: no cover - defensive, reported not hidden
        payload["reconciliation_a1_pool"] = {"error": f"{type(error).__name__}: {error}"}

    payload["seconds"] = float(time.perf_counter() - started)
    _write_json(RESULTS_DIR / "continuity_v2.json", payload)
    _mark("continuity-v2", "RUN", artifact="continuity_v2.json", seconds=payload["seconds"], device="cpu",
          reason="diagnostic only (no gate)")
    print(
        "[continuity-v2] "
        + json.dumps(
            {
                stratum: {
                    "REAL_x": round(geometry[stratum]["REAL"]["x"]["rho_angular"], 4),
                    "INDEP_x": round(geometry[stratum]["INDEP"]["x"]["rho_angular"], 4),
                    "TOPO_x": round(geometry[stratum]["TOPO"]["x"]["rho_angular"], 4),
                    "REAL_code": round(geometry[stratum]["REAL"]["code"]["rho_angular"], 4),
                }
                for stratum in geometry
            }
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: Stage 1 — frozen exact-OMP screen
# ---------------------------------------------------------------------------


def omp_screen_stage(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    values: dict[str, float] = {}
    for arm in a2.SCREEN_ARMS:
        tag = f"omp_screen_{a2.SCREEN_LABEL[arm]}"
        with a1_emits_into(RESULTS_DIR), matched_init_enforced():
            payload = a1run.train_arm(
                arm, stage="omp", device=device, tag=tag, out_dir=RESULTS_DIR
            )
        payload = _reemit(
            payload,
            source="zinc_e2e_dictenv_a1.train_arm(stage='omp')",
            stage1_arm=a2.SCREEN_LABEL[arm],
            dictionary_kind="frozen_ksvd_k32_s8_omp",
        )
        _write_json(RESULTS_DIR / f"omp_screen_{arm.lower()}.json", payload)
        values[arm] = float(payload["soup"]["soup_valid_mae"])
    seconds = float(time.perf_counter() - started)
    _mark("omp-screen", "RUN", artifact="omp_screen_{topo,indep,real}.json", seconds=seconds, device=device,
          reason="frozen exact-OMP screen complete")
    return values


def omp_decision_stage() -> dict[str, Any]:
    values = {
        arm: float(_read_json(RESULTS_DIR / f"omp_screen_{arm.lower()}.json")["soup"]["soup_valid_mae"])
        for arm in a2.SCREEN_ARMS
    }
    g_pair = float(values["INDEP"] - values["REAL"])
    g_topo = float(values["TOPO"] - values["REAL"])
    finite = all(
        _run_is_finite(f"omp_screen_{a2.SCREEN_LABEL[arm]}") for arm in a2.SCREEN_ARMS
    )
    payload = {
        **provenance(_device_of_stage("omp-screen")),
        "stage": "stage1_frozen_omp_screen",
        "run": True,
        "soup_valid_mae": values,
        "G_pair_OMP": g_pair,
        "G_topo_OMP": g_topo,
        "threshold": float(a2.MATERIAL),
        "primary_pass": bool(g_pair >= a2.MATERIAL and finite),
        "secondary_pass": bool(g_topo >= a2.MATERIAL),
        "curves_finite": bool(finite),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "omp_decision.json", payload)
    _mark("omp-decision", "RUN", artifact="omp_decision.json", device="cpu",
          reason=f"G_pair_OMP={g_pair:.6f}")
    print(f"[omp-decision] {json.dumps({k: payload[k] for k in ('soup_valid_mae','G_pair_OMP','G_topo_OMP','primary_pass')})}", flush=True)
    return payload


def _device_of_stage(stage: str) -> str:
    entry = STAGE_STATUS.get(stage, {})
    return "cuda" if entry.get("status") == "RUN" else "cpu"


# ---------------------------------------------------------------------------
# stage: Stage 2 — dense rank-32 compression control
# ---------------------------------------------------------------------------


def _pca_artifacts(arm: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit (or reload) the frozen train-only rank-32 projection of one object.

    Returns ``(projection [433, 32] float32, mean [433] float64, meta)``.  The
    fit is ``sdb_v0.fit_pca_rank`` on official train only; valid rows are only
    ever *transformed*.  All four artifacts are content-addressed by a sha256 in
    the metadata, and a reload is verified against a recomputation on the first
    256 train rows.
    """
    meta_path = RESULTS_DIR / f"pca_meta_{arm.lower()}.json"
    projection_path = RESULTS_DIR / f"pca_projection_{arm.lower()}.pt"
    mean_path = RESULTS_DIR / f"pca_mean_{arm.lower()}.pt"
    codes_path = RESULTS_DIR / f"pca_codes_{arm.lower()}.pt"
    if projection_path.exists() and mean_path.exists() and codes_path.exists() and meta_path.exists():
        blob = torch.load(projection_path, map_location="cpu", weights_only=True)
        projection = blob["components"].numpy()
        components = blob.get("components_f64", None)
        components = projection.astype(np.float64) if components is None else components.numpy()
        mean = torch.load(mean_path, map_location="cpu", weights_only=True)["mean"].numpy()
        meta = _read_json(meta_path)
        if tuple(projection.shape) != (int(a1.A1_DIM), int(a2.PCA_RANK)):
            raise RuntimeError(f"stored PCA projection has shape {projection.shape}")
        X_head = np.asarray(a1run.arm_coordinate(arm, "train")[:256], dtype=np.float64)
        stored_head = torch.load(codes_path, map_location="cpu", weights_only=True)["train"].numpy()[:256]
        recomputed_head = np.asarray(
            (X_head - np.asarray(mean, dtype=np.float64)[None, :])
            @ np.asarray(components, dtype=np.float64).T,
            dtype=np.float32,
        )
        if not np.array_equal(stored_head, recomputed_head):
            raise RuntimeError(f"stored PCA32 codes for {arm} do not reproduce from the stored projection")
        del X_head
        return np.asarray(projection, dtype=np.float32), np.asarray(mean, dtype=np.float64), meta

    X_train = np.asarray(a1run.arm_coordinate(arm, "train"), dtype=np.float64)
    pca, meta = a2.fit_dense_rank(X_train, rank=a2.PCA_RANK)
    projection = a2.pca_projection_matrix(pca)
    mean = np.asarray(pca.mean, dtype=np.float64)
    codes_train = a2.dense_codes_f32(pca, X_train)
    del X_train
    X_valid = np.asarray(a1run.arm_coordinate(arm, "valid"), dtype=np.float64)
    codes_valid = a2.dense_codes_f32(pca, X_valid)
    valid_denom = max(float((X_valid ** 2).sum()), 1e-30)
    affine_valid = mean[None, :] + np.asarray(codes_valid, dtype=np.float64) @ np.asarray(
        pca.components, dtype=np.float64
    )
    tied_valid = np.asarray(codes_valid, dtype=np.float64) @ np.asarray(pca.components, dtype=np.float64)
    meta["valid_affine_normalized_rec"] = float(((X_valid - affine_valid) ** 2).sum() / valid_denom)
    meta["valid_tied_normalized_rec"] = float(((X_valid - tied_valid) ** 2).sum() / valid_denom)
    meta["n_transform_rows_valid"] = int(X_valid.shape[0])
    meta["valid_never_enters_fit"] = True
    del X_valid, affine_valid, tied_valid
    torch.save(
        {
            "components": torch.as_tensor(projection),
            "components_f64": torch.as_tensor(np.asarray(pca.components, dtype=np.float64)),
        },
        projection_path,
    )
    torch.save({"mean": torch.as_tensor(mean)}, mean_path)
    torch.save(
        {"train": torch.as_tensor(codes_train), "valid": torch.as_tensor(codes_valid)}, codes_path
    )
    meta = _reemit(
        meta,
        source="sdb_v0.fit_pca_rank (shared implementation)",
        arm=arm,
        coordinate=arm.lower(),
        codes_train_sha256_f32=a2.a1_hash(codes_train),
        codes_valid_sha256_f32=a2.a1_hash(codes_valid),
    )
    _write_json(meta_path, meta)
    del codes_train, codes_valid
    return projection, mean, meta


def compression_stage(device: str = "cuda") -> dict[str, Any]:
    """Stage 2 — frozen dense rank-32 compression control (train-only PCA)."""
    started = time.perf_counter()
    for arm in a2.PCA_ARMS:
        projection, _mean, meta = _pca_artifacts(arm)
        blob = torch.load(
            RESULTS_DIR / f"pca_codes_{arm.lower()}.pt", map_location="cpu", weights_only=True
        )
        codes_train = blob["train"].numpy()
        codes_valid = blob["valid"].numpy()
        X_train = np.asarray(a1run.arm_coordinate(arm, "train"), dtype=np.float32)
        X_valid = np.asarray(a1run.arm_coordinate(arm, "valid"), dtype=np.float32)
        if codes_train.shape[0] != X_train.shape[0] or codes_valid.shape[0] != X_valid.shape[0]:
            raise RuntimeError("dense code rows do not match the coordinate rows")
        tag = f"pca_screen_{a2.SCREEN_LABEL[arm]}"
        with a1_emits_into(RESULTS_DIR), dictionary_override(arm, projection), matched_init_enforced():
            payload = a1run.train_arm(
                arm,
                stage="omp",
                device=device,
                tag=tag,
                out_dir=RESULTS_DIR,
                codes_train=np.asarray(codes_train, dtype=np.float32),
                codes_valid=np.asarray(codes_valid, dtype=np.float32),
                x_train=X_train,
                x_valid=X_valid,
            )
        payload = _reemit(
            payload,
            source="zinc_e2e_dictenv_a1.train_arm(stage='omp') with a frozen PCA32 code",
            stage2_arm=f"ATTR-{arm}-PCA32",
            dictionary_kind="frozen_pca32_components",
            pca_meta=meta,
        )
        payload.pop("dictionary_sha256_f32", None)
        payload["pca_components_sha256_f32"] = a2.a1_hash(projection)
        _write_json(RESULTS_DIR / f"pca_screen_{arm.lower()}.json", payload)
        del codes_train, codes_valid, X_train, X_valid, blob
    seconds = float(time.perf_counter() - started)
    _mark("compression", "RUN", artifact="pca_screen_{indep,real}.json", seconds=seconds, device=device,
          reason="train-only PCA32 control complete")
    return {"seconds": seconds}


def compression_decision_stage() -> dict[str, Any]:
    values = {
        arm: float(_read_json(RESULTS_DIR / f"pca_screen_{arm.lower()}.json")["soup"]["soup_valid_mae"])
        for arm in a2.PCA_ARMS
    }
    g_pair = float(values["INDEP"] - values["REAL"])
    finite = all(_run_is_finite(f"pca_screen_{a2.SCREEN_LABEL[arm]}") for arm in a2.PCA_ARMS)
    payload = {
        **provenance(_device_of_stage("compression")),
        "stage": "stage2_dense_pca32_compression_control",
        "run": True,
        "soup_valid_mae": values,
        "G_pair_PCA": g_pair,
        "threshold": float(a2.MATERIAL),
        "primary_pass": bool(g_pair >= a2.MATERIAL and finite),
        "curves_finite": bool(finite),
        "rank": int(a2.PCA_RANK),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "compression_decision.json", payload)
    _mark("compression-decision", "RUN", artifact="compression_decision.json", device="cpu",
          reason=f"G_pair_PCA={g_pair:.6f}")
    print(f"[compression-decision] {json.dumps({k: payload[k] for k in ('soup_valid_mae','G_pair_PCA','primary_pass')})}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: Stage 3 — IHT coder qualification
# ---------------------------------------------------------------------------


def coder_stage(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    with a1_emits_into(RESULTS_DIR):
        payload = a1run.iht_diag(device=device)
    payload = _reemit(payload, source="zinc_e2e_dictenv_a1.iht_diag")
    _write_json(RESULTS_DIR / "iht_diagnostic.json", payload)
    recomputed = a2.select_common_iht_steps(
        {
            arm: {
                step: float(payload["arms"][arm]["steps"][step]["normalized_err"])
                for step in payload["arms"][arm]["steps"]
            }
            for arm in a1.ARMS
        }
    )
    if list(payload.get("qualified_steps", [])) != recomputed["qualified_steps"] or payload.get(
        "selected_steps"
    ) != recomputed["selected_steps"]:
        raise RuntimeError("A1 coder qualification disagrees with the frozen A2 shared-step rule")
    decision = {
        **provenance(device),
        "stage": "stage3_iht_coder_qualification",
        "run": True,
        "candidate_steps": list(a1.IHT_CANDIDATE_STEPS),
        "qualification_bar": float(a1.IHT_QUALIFY_MAX_REC),
        "qualified_steps": list(payload.get("qualified_steps", [])),
        "selected_steps": payload.get("selected_steps"),
        "coder_qualified": bool(payload.get("coder_qualified")),
        "shared_step_rule_recomputed": recomputed,
        "per_arm": {
            arm: {
                steps: {
                    "normalized_err": float(payload["arms"][arm]["steps"][steps]["normalized_err"]),
                    "support_jaccard_vs_omp": float(
                        payload["arms"][arm]["steps"][steps]["support_jaccard_vs_omp"]
                    ),
                    "mean_l0": float(payload["arms"][arm]["steps"][steps]["mean_l0"]),
                }
                for steps in payload["arms"][arm]["steps"]
            }
            for arm in a1.ARMS
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "coder_decision.json", decision)
    _mark("coder", "RUN", artifact="coder_decision.json", device=device,
          seconds=float(time.perf_counter() - started),
          reason=f"selected_steps={decision['selected_steps']}")
    return decision


# ---------------------------------------------------------------------------
# stage: Stage 4 — matched E2E sparse dictionary
# ---------------------------------------------------------------------------


def formal_stage(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    with a1_emits_into(RESULTS_DIR), matched_init_enforced():
        selection = a1run.formal_runs(device=device)
    payload = _reemit(
        selection,
        source="zinc_e2e_dictenv_a1.formal_runs",
        stage="stage4_matched_e2e_sparse_dictionary",
        run=True,
        G_pair_E2E=float(selection["G_attr"]),
        G_topo_E2E=float(selection["G_topo"]),
        finite=bool(all(_run_is_finite(f"formal_{a2.E2E_LABEL[arm]}") for arm in a1.ARMS)),
    )
    payload["primary_pass"] = bool(
        float(selection["G_attr"]) >= a2.MATERIAL and payload["finite"]
    )
    _write_json(RESULTS_DIR / "formal_decision.json", payload)
    # A1-compatible alias, consumed by the frozen mechanism/specificity stages;
    # its ``primary_pass`` is re-stated under the A2 rule (material threshold plus
    # the fail-closed finiteness check) so the frozen downstream guards cannot
    # proceed past a Stage-4 the A2 decision table has already rejected.
    _write_json(
        RESULTS_DIR / "formal_selection.json",
        {**dict(selection), "primary_pass": bool(payload["primary_pass"])},
    )
    _mark("formal", "RUN", artifact="formal_decision.json", seconds=float(time.perf_counter() - started), device=device,
          reason=f"G_pair_E2E={payload['G_pair_E2E']:.6f}")
    print(f"[formal] {json.dumps({k: payload[k] for k in ('soup_valid_mae','G_pair_E2E','G_topo_E2E','primary_pass')})}", flush=True)
    return payload


def mechanism_stage(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    with a1_emits_into(RESULTS_DIR):
        payload = a1run.mechanism_stage(device=device)
    payload = _reemit(
        payload,
        source="zinc_e2e_dictenv_a1.mechanism_stage",
        stage="mechanism_code_pairing_removal",
        G_code_pairing=float(payload["primary"]["value"]),
    )
    payload["primary"]["pass"] = bool(float(payload["primary"]["value"]) >= a2.MECHANISM_THRESHOLD)
    _write_json(RESULTS_DIR / "mechanism_code_pairing.json", payload)
    interventions = payload["coordinate_interventions"]
    for name, key in (("node_only", "node_pairing_removed"), ("edge_only", "edge_pairing_removed")):
        entry = interventions[key]
        _write_json(
            RESULTS_DIR / f"mechanism_{name}.json",
            {
                **provenance(device),
                "stage": f"mechanism_{name}",
                "kind": "descriptive decomposition (cannot change the architecture)",
                "variant": key,
                "valid_mae": float(entry["valid_mae"]),
                "delta_vs_real": float(entry["delta_vs_real"]),
                "mean_abs_prediction_shift": float(entry["mean_abs_prediction_shift"]),
                "shift_quantiles": entry.get("shift_quantiles"),
                "soup_valid_mae_real": float(payload["soup_valid_mae"]),
                "official_test_loaded": False,
            },
        )
    for extra in ("zero", "node_shuffle", "edge_shuffle", "all_shuffle"):
        path = RESULTS_DIR / f"mechanism_{extra}.json"
        if path.exists():
            _write_json(path, _reemit(_read_json(path), source="zinc_e2e_dictenv_a1.mechanism_stage"))
    _mark("mechanism", "RUN", artifact="mechanism_code_pairing.json", device=device,
          seconds=float(time.perf_counter() - started),
          reason=f"G_code_pairing={payload['G_code_pairing']:.6f}")
    print(f"[mechanism] G_code_pairing={payload['G_code_pairing']:.6f} pass={payload['primary']['pass']}", flush=True)
    return payload


def _load_soup_model(arm: str, stage: str, iht_steps: int | None, device_obj: torch.device) -> Any:
    return a1run._load_soup_model(arm, stage, iht_steps, device_obj)


def _slot_gradient_norms(model: Any, capture: Any) -> dict[str, Any]:
    """Per-slot liveness instrument: parameter gradient norm + A1's input probe.

    ``param_grad_norm`` is the positive/gate instrument (a branch is alive iff its
    own parameters receive task gradient); ``slot_grad_norm`` is A1's frozen
    input-tensor probe, reported for continuity.  The anchor encoder's input
    tensor is grad-free by construction, so the input probe alone would report a
    live branch as dead.
    """
    probe = capture.stats()
    out: dict[str, Any] = {}
    for slot in a2.LIVENESS_SLOTS:
        module = getattr(model, slot, None)
        total = 0.0
        count = 0
        if module is not None:
            for parameter in module.parameters():
                count += 1
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().norm()) ** 2
        out[slot] = {
            "param_grad_norm": float(np.sqrt(total)),
            "param_tensors": int(count),
            "slot_grad_norm": (probe.get(slot) or {}).get("slot_grad_norm"),
            "slot_value_std": (probe.get(slot) or {}).get("slot_value_std"),
        }
    return out


def liveness_stage(device: str = "cuda") -> dict[str, Any]:
    """Per-arm liveness audit of the formally trained E0/E1/E2 dictionaries."""
    started = time.perf_counter()
    decision = _read_json(RESULTS_DIR / "formal_decision.json")
    steps = int(decision["selected_iht_steps"])
    device_obj = _set_device_policy(device)
    rng = np.random.default_rng(LIVENESS_SEED)
    arms: dict[str, Any] = {}
    for arm in a1.ARMS:
        D_init, _sha = a1run.load_arm_dictionary(arm)
        with a1_emits_into(RESULTS_DIR):
            model = _load_soup_model(arm, "formal", steps, device_obj)
        state = torch.load(
            RESULTS_DIR / "states" / f"formal_{a2.E2E_LABEL[arm]}_soup_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        capture = a1run._SlotCapture(model)
        data = p1run.load_split("train", subset=int(LIVENESS_PROBE_MOLECULES))
        a1run.attach_a1(data, "train", arm)
        loader = p1.make_env_loader(data, 128, False, SEED)
        model.train()
        probe: dict[str, Any] = {}
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(a1.LAMBDA_REC) * rec
            model.zero_grad()
            loss.backward()
            probe = {
                "dictionary_grad_norm": float(model.D.grad.norm()),
                "slot_gradients": _slot_gradient_norms(model, capture),
                "loss": float(loss.detach()),
                "n_nodes": int(aux["phi"].shape[0]),
            }
            break
        capture.close()
        del data, loader

        D_soup = state["D"].numpy().astype(np.float64)
        dbar_init = sdb.normalize_columns(np.asarray(D_init, dtype=np.float64))
        dbar_soup = sdb.normalize_columns(D_soup)
        X_train = np.asarray(a1run.arm_coordinate(arm, "train"))
        X_valid = np.asarray(a1run.arm_coordinate(arm, "valid"))
        train_rows = np.sort(rng.choice(X_train.shape[0], size=min(LIVENESS_TRAIN_ROWS, X_train.shape[0]), replace=False))
        valid_rows = np.sort(rng.choice(X_valid.shape[0], size=min(LIVENESS_VALID_ROWS, X_valid.shape[0]), replace=False))
        Dbar_t = torch.as_tensor(dbar_soup, dtype=torch.float32, device=device_obj)
        with torch.no_grad():
            codes_train = v0.tied_iht_codes(
                Dbar_t,
                torch.as_tensor(X_train[train_rows], dtype=torch.float32, device=device_obj),
                s=a1.DICT_S,
                steps=int(steps),
            ).cpu().numpy()
            codes_valid = v0.tied_iht_codes(
                Dbar_t,
                torch.as_tensor(X_valid[valid_rows], dtype=torch.float32, device=device_obj),
                s=a1.DICT_S,
                steps=int(steps),
            ).cpu().numpy()
        usage_train = (np.abs(codes_train) > 0).sum(0).astype(np.float64)
        usage_valid = (np.abs(codes_valid) > 0).sum(0).astype(np.float64)
        usage_spearman = (
            float(np.corrcoef(np.argsort(np.argsort(usage_train)), np.argsort(np.argsort(usage_valid)))[0, 1])
            if usage_train.size > 2
            else 0.0
        )
        singular = np.linalg.svd(dbar_soup, compute_uv=False)
        variance = singular ** 2
        entry = {
            "arm": arm,
            "e2e_arm": a2.E2E_LABEL[arm],
            "selected_iht_steps": int(steps),
            "dictionary_grad_norm": probe.get("dictionary_grad_norm"),
            "slot_gradients": probe.get("slot_gradients"),
            "probe_loss": probe.get("loss"),
            "probe_nodes": probe.get("n_nodes"),
            "soup_D_vs_ksvd_init": {
                "fro": float(np.linalg.norm(dbar_soup - dbar_init)),
                "relative": float(
                    np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + a1.EPS)
                ),
            },
            "effective_rank_D_soup": float((variance.sum() ** 2) / max((variance ** 2).sum(), 1e-30)),
            "active_atoms_train": int((usage_train >= 1).sum()),
            "effective_atoms_train": float(a1run._usage_stats(codes_train)["effective_atom_count"]),
            "effective_atoms_valid": float(a1run._usage_stats(codes_valid)["effective_atom_count"]),
            "train_valid_usage_spearman": usage_spearman,
            "code_variance_mean_train": float(codes_train.var(0).mean()),
            "code_space_std": float(codes_train.std()),
            "usage_entropy_train": float(a1run._usage_stats(codes_train)["support_entropy"]),
            "train_rows_used": int(codes_train.shape[0]),
            "valid_rows_used": int(codes_valid.shape[0]),
            "code_source": "tied IHT on the trained soup dictionary",
        }
        arms[arm] = entry
        print(
            f"[liveness:{arm}] grad={entry['dictionary_grad_norm']:.4e} "
            f"move={entry['soup_D_vs_ksvd_init']['relative']:.4f} "
            f"active={entry['active_atoms_train']} eff={entry['effective_atoms_train']:.2f} "
            f"rho={usage_spearman:.4f}",
            flush=True,
        )
    gate = a2.liveness_all_pass(arms)
    payload = {
        **provenance(device),
        "stage": "liveness",
        "arms": arms,
        "gate": gate,
        "all_pass": bool(gate["all_pass"]),
        "failing_arms": gate["failing_arms"],
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "liveness.json", payload)
    _mark("liveness", "RUN", artifact="liveness.json", seconds=payload["seconds"], device=device,
          reason="all formal arms live" if payload["all_pass"] else f"dead: {payload['failing_arms']}")
    return payload


# ---------------------------------------------------------------------------
# stage: Stage 5 — sparse specificity
# ---------------------------------------------------------------------------


def specificity_stage(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    with a1_emits_into(RESULTS_DIR), matched_init_enforced():
        payload = a1run.specificity_stage(device=device)
    payload = _reemit(
        payload,
        source="zinc_e2e_dictenv_a1.specificity_stage",
        stage="stage5_sparse_vs_dense_tied_specificity",
        run=True,
        G_sparse=float(payload["G_sparse"]),
        dense_tied_tag="specificity_E2_vs_dense",
    )
    payload["pass"] = bool(float(payload["G_sparse"]) >= a2.MATERIAL and _run_is_finite("specificity_E2_vs_dense"))
    _write_json(RESULTS_DIR / "specificity.json", payload)
    _mark("specificity", "RUN", artifact="specificity.json", seconds=float(time.perf_counter() - started), device=device,
          reason=f"G_sparse={payload['G_sparse']:.6f}")
    return payload


# ---------------------------------------------------------------------------
# stage: smoke (plumbing only — cannot influence any decision)
# ---------------------------------------------------------------------------


SMOKE_MOLECULES = 216
SMOKE_HORIZON = 2


def smoke_stage(device: str = "cuda") -> dict[str, Any]:
    """Short GPU1 plumbing smoke for the frozen trainer on the reused objects.

    Runs the real artifacts through the real training loop (task loss, task
    gradient, coding, soup bookkeeping, curve/state writing) on a small molecule
    subset for two epochs.  It writes only under ``results/e2e_dictenv_a2/smoke/``
    and is never read by any gate, decision or report number.
    """
    started = time.perf_counter()
    smoke_dir = RESULTS_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _set_device_policy(device)
    original_load_split = p1run.load_split

    def patched_load_split(split: str, subset: int | None = None):
        data = original_load_split(split)
        limit = int(subset) if subset is not None else SMOKE_MOLECULES
        return data[:limit]

    p1run.load_split = patched_load_split  # type: ignore[assignment]
    results: dict[str, Any] = {}
    try:
        for arm in a2.SCREEN_ARMS:
            with a1_emits_into(smoke_dir):
                payload = a1run.train_arm(
                    arm,
                    stage="omp",
                    device=device,
                    tag=f"smoke_omp_{a2.SCREEN_LABEL[arm]}",
                    horizon=SMOKE_HORIZON,
                    out_dir=smoke_dir,
                    capture_grads=True,
                )
            results[f"omp_{arm}"] = {
                "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
                "train_mae_last": float(payload["train_mae_at_best"]),
                "curve_finite": bool(
                    _curve_is_finite(
                        smoke_dir / "curves" / f"smoke_omp_{a2.SCREEN_LABEL[arm]}_curve.csv"
                    )
                ),
                "gradient_probe": payload.get("gradient_probe"),
            }
        with a1_emits_into(smoke_dir):
            payload = a1run.train_arm(
                "REAL",
                stage="formal",
                device=device,
                iht_steps=int(v0.IHT_STEPS if hasattr(v0, "IHT_STEPS") else 10),
                tag="smoke_iht_E2",
                horizon=SMOKE_HORIZON,
                out_dir=smoke_dir,
                capture_grads=True,
            )
        results["iht_REAL"] = {
            "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
            "dictionary_grad_norm": (payload.get("gradient_probe") or {}).get("dictionary_grad_norm"),
            "dictionary_movement": payload.get("dictionary_movement"),
            "curve_finite": bool(
                _curve_is_finite(smoke_dir / "curves" / "smoke_iht_E2_curve.csv")
            ),
        }
    finally:
        p1run.load_split = original_load_split  # type: ignore[assignment]
    gate = {
        "all_finite": bool(
            all(entry["curve_finite"] for entry in results.values())
        ),
        "omp_dictionary_frozen": bool(
            all((results[f"omp_{arm}"].get("gradient_probe") or {}).get("dictionary_grad_norm") in (None, 0.0)
                for arm in a2.SCREEN_ARMS)
        ),
        "iht_dictionary_gradient_nonzero": bool(
            (results["iht_REAL"].get("dictionary_grad_norm") or 0.0) > 0.0
        ),
    }
    payload = {
        **provenance(device),
        "stage": "smoke",
        "kind": "plumbing smoke only; cannot influence any gate, decision or number",
        "molecules": int(SMOKE_MOLECULES),
        "horizon": int(SMOKE_HORIZON),
        "results": results,
        "gate": gate,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(smoke_dir / "smoke.json", payload)
    _mark("smoke", "RUN (plumbing only)", artifact="smoke/smoke.json", device=device,
          seconds=payload["seconds"], reason="GPU1 trainer smoke")
    print(f"[smoke] {json.dumps(results, default=str)}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: decision / report
# ---------------------------------------------------------------------------


def _maybe(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / name
    return _read_json(path) if path.exists() else None


def decision_stage() -> dict[str, Any]:
    identity = _maybe("artifact_identity.json")
    qualification = _maybe("qualification.json")
    continuity = _maybe("continuity_v2.json")
    stage1 = _maybe("omp_decision.json")
    stage2 = _maybe("compression_decision.json")
    stage3 = _maybe("coder_decision.json")
    stage4 = _maybe("formal_decision.json")
    liveness = _maybe("liveness.json")
    mechanism = _maybe("mechanism_code_pairing.json")
    specificity = _maybe("specificity.json")

    evidence = {
        "identity_passed": bool(identity and identity.get("all_passed")),
        "gate0_passed": bool(qualification and qualification.get("passed")),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "liveness": liveness,
        "mechanism": mechanism,
        "specificity": specificity,
    }
    result = a2.verdict_from_evidence(**evidence)

    _mark_stage_status_defaults(evidence)
    cuda_stages = [
        name for name, entry in STAGE_STATUS.items() if entry.get("device") == "cuda"
    ]
    cpu_stages = [
        name for name, entry in STAGE_STATUS.items() if entry.get("device") == "cpu"
    ]
    payload = {
        **provenance("cpu"),
        "cuda_stages_run": cuda_stages,
        "cpu_stages_run": cpu_stages,
        "device_policy": "physical GPU1 only (CUDA_VISIBLE_DEVICES=1); GPU0 never touched",
        "verdict": result["verdict"],
        "reason": result["reason"],
        "branch": result["branch"],
        "identity": {
            "all_passed": None if identity is None else bool(identity.get("all_passed")),
            "failures": [] if identity is None else identity.get("failures", []),
        },
        "qualification": qualification,
        "continuity_v2": None if continuity is None else {
            "kind": continuity.get("kind"),
            "pool": continuity.get("pool"),
            "stratum_counts": continuity.get("stratum_counts"),
            "geometry": continuity.get("geometry"),
            "contrasts": continuity.get("contrasts"),
        },
        "stage1_omp": None if stage1 is None else {
            "soup_valid_mae": stage1["soup_valid_mae"],
            "G_pair_OMP": stage1["G_pair_OMP"],
            "G_topo_OMP": stage1["G_topo_OMP"],
            "primary_pass": stage1["primary_pass"],
            "secondary_pass": stage1["secondary_pass"],
        },
        "stage2_compression": None if stage2 is None else {
            "soup_valid_mae": stage2["soup_valid_mae"],
            "G_pair_PCA": stage2["G_pair_PCA"],
            "primary_pass": stage2["primary_pass"],
        },
        "stage3_coder": None if stage3 is None else {
            "qualified_steps": stage3["qualified_steps"],
            "selected_steps": stage3["selected_steps"],
            "coder_qualified": stage3["coder_qualified"],
        },
        "stage4_e2e": None if stage4 is None else {
            "soup_valid_mae": stage4["soup_valid_mae"],
            "G_pair_E2E": stage4["G_pair_E2E"],
            "G_topo_E2E": stage4["G_topo_E2E"],
            "primary_pass": stage4["primary_pass"],
        },
        "mechanism": None if mechanism is None else {
            "G_code_pairing": mechanism["G_code_pairing"],
            "primary_pass": mechanism["primary"]["pass"],
            "coordinate_interventions": mechanism["coordinate_interventions"],
            "inherited_interventions": mechanism["inherited_interventions"],
        },
        "liveness": None if liveness is None else {
            "all_pass": liveness["all_pass"],
            "failing_arms": liveness["failing_arms"],
            "arms": {
                arm: {
                    "dictionary_grad_norm": entry["dictionary_grad_norm"],
                    "soup_D_vs_ksvd_init": entry["soup_D_vs_ksvd_init"],
                    "active_atoms_train": entry["active_atoms_train"],
                    "effective_atoms_train": entry["effective_atoms_train"],
                    "train_valid_usage_spearman": entry["train_valid_usage_spearman"],
                }
                for arm, entry in (liveness.get("arms") or {}).items()
            },
        },
        "specificity": None if specificity is None else {
            "G_sparse": specificity["G_sparse"],
            "pass": specificity["pass"],
        },
        "stage_status": STAGE_STATUS,
        "seed": SEED,
        "seed1_authorised": bool(
            stage4 is not None and float(stage4.get("G_pair_E2E", 0.0)) >= a2.MATERIAL
        ),
        "seed1_run": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_json(RESULTS_DIR / "stage_status.json", {
        **provenance("cpu"),
        "kind": "explicit RUN / NOT RUN record for every stage (no ambiguous gaps)",
        "stages": STAGE_STATUS,
        "official_test_loaded": False,
    })
    print(f"[decision] {payload['verdict']} ({payload['reason']})", flush=True)
    return payload


def _mark_stage_status_defaults(evidence: Mapping[str, Any]) -> None:
    stage1 = evidence.get("stage1")
    if stage1 is not None and stage1.get("primary_pass"):
        _mark("compression", "NOT RUN", reason="stage1_primary_pass (PCA32 not triggered)")
        _mark("compression-decision", "NOT RUN", reason="stage1_primary_pass (PCA32 not triggered)")
    elif stage1 is not None:
        for name in ("coder", "formal", "mechanism", "liveness", "specificity"):
            _mark(name, "NOT RUN", reason="stage1_primary_fail (sparse route not entered)")
    if evidence.get("stage3") is not None and not evidence["stage3"].get("coder_qualified"):
        for name in ("formal", "mechanism", "liveness", "specificity"):
            _mark(name, "NOT RUN", reason="coder_not_qualified")
    stage4 = evidence.get("stage4")
    if stage4 is not None:
        if not stage4.get("primary_pass"):
            _mark("specificity", "NOT RUN", reason="stage4_G_pair_E2E_below_threshold")
        elif evidence.get("mechanism") is not None and not evidence["mechanism"]["primary"]["pass"]:
            _mark("specificity", "NOT RUN", reason="code_pairing_below_threshold")
        elif evidence.get("liveness") is not None and not evidence["liveness"].get("all_pass"):
            _mark("specificity", "NOT RUN", reason="liveness_failed")


def _seven_questions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    def not_run(stage: str) -> str:
        entry = (payload.get("stage_status") or {}).get(stage) or {}
        reason = entry.get("reason") or "not reached by the frozen order"
        return f"NOT RUN — {reason}"

    identity = payload.get("identity") or {}
    continuity = payload.get("continuity_v2") or {}
    stage1 = payload.get("stage1_omp")
    stage2 = payload.get("stage2_compression")
    stage4 = payload.get("stage4_e2e")
    mechanism = payload.get("mechanism")
    specificity = payload.get("specificity")
    geometry = continuity.get("geometry") or {}
    strata = [name for name, _low, _high in a2.STRATA] + ["pooled"]
    continuity_summary = {
        stratum: {
            arm: {
                space: round(float(geometry[stratum][arm][space]["rho_angular"]), 4)
                for space in ("x", "code")
            }
            for arm in ("REAL", "INDEP", "TOPO")
        }
        for stratum in strata
        if stratum in geometry
    }
    questions = [
        {
            "id": "Q1",
            "question": "Were the A1 REAL/INDEP 433-D objects reused bit-identically, and are the dictionaries/codes identity-verified?",
            "answer": "YES" if identity.get("all_passed") else "NO / NOT VERIFIED",
            "status": "RUN",
            "evidence": "artifact_identity.json",
            "detail": identity.get("failures", []),
        },
        {
            "id": "Q2",
            "question": "How does continuity-v2 behave per patch-size stratum, in x-space and OMP-code space, for TOPO/INDEP/REAL?",
            "answer": "reported (diagnostic, not a verdict)",
            "status": "RUN" if geometry else "NOT RUN",
            "evidence": "continuity_v2.json",
            "detail": {
                "stratum_counts": continuity.get("stratum_counts"),
                "rho_angular": continuity_summary,
                "contrasts": continuity.get("contrasts"),
            },
        },
        {
            "id": "Q3",
            "question": "Frozen exact-OMP: who wins, REAL or INDEP, and what is G_pair_OMP?",
            "answer": "NOT RUN" if stage1 is None else (
                f"G_pair_OMP={stage1['G_pair_OMP']:.6f} "
                + ("(REAL better)" if stage1["G_pair_OMP"] > 0 else "(INDEP better)")
            ),
            "status": "NOT RUN" if stage1 is None else "RUN",
            "evidence": "omp_decision.json",
            "detail": stage1,
        },
        {
            "id": "Q4",
            "question": "If OMP did not support REAL: what does the train-only PCA32 control say?",
            "answer": not_run("compression")
            if stage2 is None
            else (
                f"G_pair_PCA={stage2['G_pair_PCA']:.6f} "
                + ("(pairing useful but sparse-compressed away)"
                   if stage2["primary_pass"]
                   else "(no compression-recoverable signal)")
            ),
            "status": "NOT RUN" if stage2 is None else "RUN",
            "evidence": "compression_decision.json",
            "detail": stage2,
        },
        {
            "id": "Q5",
            "question": "If the sparse route entered E2E: is there a material REAL vs INDEP gain (G_pair_E2E)?",
            "answer": not_run("formal")
            if stage4 is None
            else (
                f"G_pair_E2E={stage4['G_pair_E2E']:.6f}, G_topo_E2E={stage4['G_topo_E2E']:.6f} "
                + ("(material)" if stage4["primary_pass"] else "(below threshold)")
            ),
            "status": "NOT RUN" if stage4 is None else "RUN",
            "evidence": "formal_decision.json",
            "detail": stage4,
        },
        {
            "id": "Q6",
            "question": "Does removing only the attribute->dictionary-code pairing materially degrade the trained REAL model?",
            "answer": not_run("mechanism")
            if mechanism is None
            else (
                f"G_code_pairing={mechanism['G_code_pairing']:.6f} "
                + ("(material)" if mechanism["primary_pass"] else "(below threshold)")
            ),
            "status": "NOT RUN" if mechanism is None else "RUN",
            "evidence": "mechanism_code_pairing.json",
            "detail": mechanism,
        },
        {
            "id": "Q7",
            "question": "If authorised: does sparsity have independent value over the matched dense tied coordinate (G_sparse)?",
            "answer": not_run("specificity")
            if specificity is None
            else (
                f"G_sparse={specificity['G_sparse']:.6f} "
                + ("(sparse wins)" if specificity["pass"] else "(not specific)")
            ),
            "status": "NOT RUN" if specificity is None else "RUN",
            "evidence": "specificity.json",
            "detail": specificity,
        },
    ]
    return questions


def report_stage() -> dict[str, Any]:
    decision = _read_json(RESULTS_DIR / "decision.json")
    questions = _seven_questions(decision)
    lines = [
        f"# {a2.ROUND} — report",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        f"* reason: {decision['reason']}",
        f"* branch: {decision['branch']}",
        f"* protocol `{PROTOCOL_VERSION}` · preregistration `{decision.get('preregistration_commit')}` · "
        f"git `{decision.get('git_commit')}` · branch `{decision.get('branch')}`",
        f"* device: {decision.get('device')} · seed {decision.get('seed')} · "
        f"official test loaded: {decision.get('official_test_loaded')}",
        "",
        "## Stage status",
        "",
        "| stage | status | reason |",
        "|---|---|---|",
    ]
    for stage, entry in (decision.get("stage_status") or {}).items():
        lines.append(f"| {stage} | {entry['status']} | {entry.get('reason', '')} |")
    lines += ["", "## Seven questions", ""]
    for item in questions:
        lines += [f"### {item['id']} — {item['question']}", "", f"* status: {item['status']}", f"* answer: {item['answer']}", ""]
    lines += ["## Evidence", ""]
    stage_keys = [
        ("stage1_omp", "Stage 1 — frozen exact-OMP screen"),
        ("stage2_compression", "Stage 2 — dense rank-32 compression control"),
        ("stage3_coder", "Stage 3 — IHT coder qualification"),
        ("stage4_e2e", "Stage 4 — matched E2E sparse dictionary"),
        ("mechanism", "Mechanism — code-pairing removal"),
        ("liveness", "Liveness"),
        ("specificity", "Stage 5 — sparse specificity"),
    ]
    for key, title in stage_keys:
        entry = decision.get(key)
        lines += [f"### {title}", ""]
        lines += ["```json", json.dumps(entry, indent=1, sort_keys=True, default=str), "```", ""]
    decision_lines = [
        f"# {a2.ROUND} — decision",
        "",
        f"**Verdict: `{decision['verdict']}`**",
        "",
        f"Reason: {decision['reason']} (branch `{decision['branch']}`).",
        "",
        "## Frozen evidence",
        "",
        f"* protocol `{PROTOCOL_VERSION}`, preregistration `{decision.get('preregistration_commit')}`, "
        f"git `{decision.get('git_commit')}` on `{decision.get('branch')}`",
        f"* device: {decision.get('device')} (physical GPU1 only; GPU0 never touched)",
        f"* `official_test_loaded = {decision.get('official_test_loaded')}`",
        f"* seed {decision.get('seed')}; seed 1 authorised: {decision.get('seed1_authorised')} "
        f"(run: {decision.get('seed1_run')})",
        "",
        "## Stage status",
        "",
        "| stage | status | reason |",
        "|---|---|---|",
    ]
    for stage, entry in (decision.get("stage_status") or {}).items():
        decision_lines.append(f"| {stage} | {entry['status']} | {entry.get('reason', '')} |")
    decision_lines += [
        "",
        "## Seven questions",
        "",
    ]
    for item in questions:
        decision_lines += [f"* **{item['id']}** ({item['status']}): {item['answer']}"]
    decision_lines += [
        "",
        "## Claim boundary",
        "",
        "* Valid-only: the official ZINC test split was never loaded, so every number is an "
        "official-valid Top-5 soup MAE under the frozen split.",
        "* No capacity, sparsity, decoder, optimizer or λ axis was searched; the verdicts follow the "
        "pre-registered decision table exactly.",
        "* A `NOT RUN` stage is reported as `NOT RUN`, never by inference.",
        "",
        "## Evidence files",
        "",
    ]
    for name in sorted(p.name for p in RESULTS_DIR.glob("*.json")):
        decision_lines.append(f"* `{name}`")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    _mark("report", "RUN", artifact="REPORT.md, DECISION.md", reason="report written", device="cpu")
    return {"report": str(RESULTS_DIR / "REPORT.md"), "decision": str(RESULTS_DIR / "DECISION.md")}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _not_run(names: Sequence[str], reason: str) -> None:
    """Explicitly record every stage the frozen order leaves unexecuted."""
    for name in names:
        if name not in STAGE_STATUS or STAGE_STATUS[name].get("status") != "RUN":
            _mark(name, "NOT RUN", reason=reason)


SPARSE_ROUTE = ("coder", "formal", "mechanism", "liveness", "specificity")


def run_all(device: str = "cuda") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    identity = identity_stage()
    if not identity["all_passed"]:
        _not_run(("qualify", "continuity-v2", *SPARSE_ROUTE, "compression", "compression-decision"),
                 "artifact_identity_failure")
        decision_stage()
        report_stage()
        print("[stop] ARTIFACT_IDENTITY_FAILURE", flush=True)
        return
    qualification = qualify_stage()
    continuity_v2_stage()
    if not qualification["passed"]:
        _not_run(("omp-screen", "omp-decision", *SPARSE_ROUTE, "compression", "compression-decision"),
                 "gate0_not_qualified")
        decision_stage()
        report_stage()
        print("[stop] GATE0_NOT_QUALIFIED", flush=True)
        return
    _set_device_policy(device)
    omp_screen_stage(device=device)
    stage1 = omp_decision_stage()
    if stage1["primary_pass"]:
        _not_run(("compression", "compression-decision"),
                 "stage1_primary_pass (Stage 2 is the fallback route only)")
        coder = coder_stage(device=device)
        if not coder["coder_qualified"]:
            _not_run(("formal", "mechanism", "liveness", "specificity"), "coder_not_qualified")
            decision_stage()
            report_stage()
            print("[stop] CODER_NOT_QUALIFIED", flush=True)
            return
        # frozen order: §7 Stage 4 (E2E) → §8 mechanism (required whenever the
        # E2 run completed) → §9 liveness → §10 conditional Stage 5
        formal_stage(device=device)
        mechanism_stage(device=device)
        liveness_stage(device=device)
        decision = decision_stage()
        if (
            decision["stage4_e2e"]["primary_pass"]
            and decision["mechanism"]["primary_pass"]
            and decision["liveness"]["all_pass"]
        ):
            specificity_stage(device=device)
        else:
            _not_run(("specificity",), "stage4 / mechanism / liveness condition not met")
        decision = decision_stage()
        report_stage()
        print(f"[done] {decision['verdict']}", flush=True)
        return
    _not_run(SPARSE_ROUTE, "stage1_primary_fail (sparse route not entered)")
    compression_stage(device=device)
    compression_decision_stage()
    decision = decision_stage()
    report_stage()
    print(f"[done] {decision['verdict']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-A2 runner")
    parser.add_argument(
        "stage",
        choices=(
            "identity", "qualify", "continuity-v2", "omp-screen", "omp-decision",
            "compression", "compression-decision", "coder", "formal", "mechanism",
            "liveness", "specificity", "smoke", "decision", "report", "all",
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--identity-rows", type=int, default=0,
                        help="verify only the first N rows of each reused OMP cache (0 = all)")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "identity":
        identity_stage(rows=int(args.identity_rows))
    elif args.stage == "qualify":
        qualify_stage()
    elif args.stage == "continuity-v2":
        continuity_v2_stage()
    elif args.stage == "omp-screen":
        omp_screen_stage(device=args.device)
    elif args.stage == "omp-decision":
        omp_decision_stage()
    elif args.stage == "compression":
        compression_stage(device=args.device)
    elif args.stage == "compression-decision":
        compression_decision_stage()
    elif args.stage == "coder":
        coder_stage(device=args.device)
    elif args.stage == "formal":
        formal_stage(device=args.device)
    elif args.stage == "mechanism":
        mechanism_stage(device=args.device)
    elif args.stage == "liveness":
        liveness_stage(device=args.device)
    elif args.stage == "specificity":
        specificity_stage(device=args.device)
    elif args.stage == "smoke":
        smoke_stage(device=args.device)
    elif args.stage == "decision":
        decision_stage()
    elif args.stage == "report":
        report_stage()
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
