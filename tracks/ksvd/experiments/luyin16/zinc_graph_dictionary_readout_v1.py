"""Zinc graph-dictionary readout v1 — frozen Full representation screen.

Round ``zinc_graph_dictionary_readout_v1`` (Workstream Z, ZINC).

Question: given the **already trained** ``Full m=3`` seed-0 parameter soup
(408,651 parameters, original soup valid MAE ``0.1191540920053958``), can a
*fixed* graph-level prototype dictionary plus an L2-regularised MAE readout
placed directly on the 814-D reader input ``R`` beat the same frozen Full
model's own MLP readout?

Protocol (frozen before any real fit, see the preregistration):

* one backbone only — the existing Full soup; every parameter frozen;
* capture the exact tensor that reaches ``model.reader`` (814 dims) with a
  forward pre-hook, once per batch;
* the vendored ``prototype_dictionary`` / ``run_cached_head`` package does all
  preprocessing, prototype selection, kernel bandwidth, Nyström whitening and
  the certified convex MAE head; this round never rewrites the kernel family;
* head dev (every 10th canonical group) only selects ``lambda`` among three
  fixed values; the final head is refit once on all official-train rows;
* the official valid split is opened exactly once for the paired screen;
* the official test split is never instantiated.

This module is CPU-only.  ``official_test_loaded`` is False in every payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_scale_v1 as scale
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    extract_full_features as eff,
)
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    prototype_dictionary as pd,
)
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    run_cached_head as rch,
)
from tracks.ksvd.experiments.luyin16.graph_dictionary_readout.v1_20261002 import (
    torch_readout_scaffold as trs,
)

PROTOCOL_VERSION = "zinc_graph_dictionary_readout_v1"

TRACK_ROOT = scale.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_graph_dictionary_readout_v1"
CACHE_DIR = RESULTS_DIR / "cache"
FIT_DIR = RESULTS_DIR / "fitted"
DEPLOY_DIR = RESULTS_DIR / "deploy"

TRAIN_CACHE = CACHE_DIR / "full_train_R.npz"
VALID_CACHE = CACHE_DIR / "full_valid_R.npz"
MODEL_PATH = FIT_DIR / "model.npz"
FIT_JSON = FIT_DIR / "FIT.json"
EVAL_JSON = RESULTS_DIR / "evaluated.json"
DEPLOY_JSON = DEPLOY_DIR / "deploy.json"

#: verified canonical train group keys from the immediately previous round.
GROUP_CACHE = (
    TRACK_ROOT
    / "results/zinc_e2e_dictenv_typed_cycle_v1/probe/train_ring_probe.npz"
)

#: frozen Full soup checkpoint and its published in-protocol valid MAE.
CHECKPOINT = scale.CHECKPOINT_DIR / "SCALE-FULL-seed0_soup_state.pt"
EXPECTED_VALID_MAE = 0.1191540920053958
EXPECTED_FULL_PARAMETERS = 408651
REPLAY_MAE_TOL = 2e-6
READER_REPLAY_TOL = 2e-6
DEPLOY_MAX_TOL = 1e-5
DEPLOY_MAE_TOL = 2e-6

THREADS = 8
SEED = scale.SEED

_read_json = base._read_json
_write_json = base._write_json
_sha256_file = eff.sha256_file


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CACHE_DIR, FIT_DIR, DEPLOY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _git_commit() -> str:
    return base._git_commit()


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError(
            "official-test blocker violated (official_test_loaded must be False)"
        )


def configure(*, threads: int | None = None, device: str = "cpu") -> dict[str, Any]:
    if torch.device(device).type != "cpu":
        raise RuntimeError(f"this round is CPU-only, got device={device!r}")
    global THREADS
    if threads is not None:
        THREADS = int(threads)
    torch.set_num_threads(int(THREADS))
    return {"device": "cpu", "threads": int(THREADS)}


# ---------------------------------------------------------------------------
# backbone state / hashing
# ---------------------------------------------------------------------------


def _backbone_state_dict() -> dict[str, torch.Tensor]:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return {str(key): value for key, value in state.items()}


def _exclude_reader(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if not key.startswith("reader.")}


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Order-stable float64 byte hash over the named tensors."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(
            state[name].detach().cpu().numpy().astype(np.float64)
        )
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _soup_model() -> sc.LatentScaleSEM108:
    """Build the frozen Full and load the existing parameter soup."""
    model = scale._soup_model()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    configure()
    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"existing Full soup absent at {CHECKPOINT}; do not retrain it for this screen"
        )
    model = _soup_model()
    n_parameters = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    from ksvd_research.runtime.fingerprints import zinc_fingerprints

    fingerprints = zinc_fingerprints(
        TRACK_ROOT.parents[1] / "data/ZINC",
        expected_sizes={"train": 10000, "val": 1000, "test": 1000},
    )
    state = _backbone_state_dict()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "device": "cpu",
        "threads": THREADS,
        "seed": SEED,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": _sha256_file(CHECKPOINT),
        "checkpoint_bytes": int(CHECKPOINT.stat().st_size),
        "full_parameters": n_parameters,
        "expected_full_parameters": EXPECTED_FULL_PARAMETERS,
        "trainable_parameters": trainable,
        "backbone_frozen": bool(trainable == 0),
        "backbone_state_sha256": _state_hash(state),
        "backbone_state_sha256_no_reader": _state_hash(_exclude_reader(state)),
        "expected_valid_mae": EXPECTED_VALID_MAE,
        "published_soup_valid_mae": EXPECTED_VALID_MAE,
        "split_fingerprint": fingerprints["split_fingerprint"],
        "fingerprints": fingerprints,
        "reader_input_width": int(sc.readout_layout(sc.FULL)["reader_input"]),
        "group_cache": str(GROUP_CACHE),
        "group_cache_present": bool(GROUP_CACHE.is_file()),
        "group_cache_sha256": (
            _sha256_file(GROUP_CACHE) if GROUP_CACHE.is_file() else None
        ),
        "passed": bool(n_parameters == EXPECTED_FULL_PARAMETERS and trainable == 0),
    }
    official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] params={n_parameters} frozen={trainable == 0} "
        f"checkpoint_sha={payload['checkpoint_sha256'][:12]}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError("Full checkpoint / freeze preflight failed")
    return payload


# ---------------------------------------------------------------------------
# feature export (train and valid both go through the vendored scaffold)
# ---------------------------------------------------------------------------


def _attach_group_ids(cache_path: Path, split: str) -> dict[str, Any]:
    """Attach canonical train groups in official row order, verified by labels."""
    meta: dict[str, Any] = {"group_source": "official_row_id", "canonical": False}
    with np.load(cache_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    if split != "train":
        return meta
    if not GROUP_CACHE.is_file():
        meta["group_note"] = "canonical group cache absent; per-row IDs used"
        return meta
    with np.load(GROUP_CACHE, allow_pickle=False) as gz:
        groups = gz["group_ids"].astype(np.int64).copy()
        probe_y = gz["y"].astype(np.float64)
    cache_y = np.asarray(arrays["y"], dtype=np.float64)
    if groups.shape != (len(cache_y),) or not np.allclose(
        probe_y, cache_y, atol=1e-5, rtol=0.0
    ):
        meta["group_note"] = (
            "canonical group cache split/order mismatch; per-row IDs used"
        )
        return meta
    arrays["group_ids"] = groups
    meta.update(
        {
            "group_source": "typed_cycle_probe canonical certificate",
            "canonical": True,
            "group_cache": str(GROUP_CACHE),
            "group_cache_sha256": _sha256_file(GROUP_CACHE),
            "n_unique_groups": int(len(np.unique(groups))),
            "order_verified_by": "10,000-row label vector allclose(atol=1e-5)",
        }
    )
    np.savez(cache_path, **arrays)
    return meta


def _verify_export(cache_path: Path, split: str, report: Mapping[str, Any]) -> dict[str, Any]:
    """Independent replay / alignment / freeze acceptance on the saved cache."""
    with np.load(cache_path, allow_pickle=False) as z:
        r = z["R"].astype(np.float64)
        y = z["y"].astype(np.float64)
        p_base = z["p_base"].astype(np.float64)
        ids = z["ids"].copy()
    model = _soup_model()
    r32 = torch.from_numpy(r.astype(np.float32))
    with torch.no_grad():
        replay = model.reader(r32).cpu().double().numpy().reshape(-1)
    reader_max = float(np.max(np.abs(replay - p_base))) if len(p_base) else 0.0
    # Batch-size / order alignment on the actual reader.
    with torch.no_grad():
        single = torch.cat(
            [model.reader(r32[index : index + 1]) for index in range(min(8, len(r32)))]
        ).double().numpy().reshape(-1)
        permutation = torch.arange(len(r32) - 1, -1, -1)
        reversed_full = model.reader(r32[permutation]).cpu().double().numpy().reshape(-1)
    batch_delta = float(np.max(np.abs(single - replay[: min(8, len(r32))])))
    order_delta = float(
        np.max(np.abs(reversed_full[::-1] - replay)) if len(replay) else 0.0
    )
    owned = {
        "reader_replay_max_abs": reader_max,
        "reader_replay_tolerance": READER_REPLAY_TOL,
        "single_vs_batch_max_abs": batch_delta,
        "reversed_order_max_abs": order_delta,
        "row_ids_are_positional": bool(np.array_equal(ids, np.arange(len(ids)))),
        "n_rows": int(len(y)),
        "r_shape": list(r.shape),
        "original_MAE_scaffold": float(report["original_MAE"]),
        "original_MAE_recomputed": float(np.mean(np.abs(p_base - y))),
        "replay_MAE": float(np.mean(np.abs(replay - y))),
    }
    owned["replay_gate_passed"] = bool(reader_max <= READER_REPLAY_TOL)
    owned = {key: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for key, v in owned.items()}
    if split == "valid":
        owned["published_valid_MAE"] = EXPECTED_VALID_MAE
        owned["valid_MAE_delta"] = float(owned["replay_MAE"] - EXPECTED_VALID_MAE)
        owned["valid_replay_gate_passed"] = bool(
            abs(owned["valid_MAE_delta"]) <= REPLAY_MAE_TOL
        )
    return owned


def export_split(split: str) -> Path:
    """Export one split through the vendored scaffold, then add provenance."""
    _ensure_dirs()
    if split not in ("train", "valid"):
        raise RuntimeError("official test is forbidden in this round")
    cache_path = TRAIN_CACHE if split == "train" else VALID_CACHE
    provenance_path = cache_path.with_suffix(".provenance.json")
    if cache_path.is_file() and provenance_path.is_file():
        print(f"[export:{split}] cache hit", flush=True)
        return cache_path
    started = time.perf_counter()
    report = eff.export_split(split, cache_path)
    group_meta = _attach_group_ids(cache_path, split)
    acceptance = _verify_export(cache_path, split, report)
    with np.load(cache_path, allow_pickle=False) as z:
        frozen = bool(z["frozen_backbone"].item())
        official = bool(z["official_test_loaded"].item())
        checkpoint_sha = str(z["checkpoint_sha"].item())
    state = _backbone_state_dict()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "split": split,
        "cache_path": str(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "cache_bytes": int(cache_path.stat().st_size),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": _sha256_file(CHECKPOINT),
        "cache_checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha_match": bool(checkpoint_sha == _sha256_file(CHECKPOINT)),
        "backbone_state_sha256_no_reader": _state_hash(_exclude_reader(state)),
        "frozen_backbone_flag": frozen,
        "official_test_loaded_flag": official,
        "full_parameters": EXPECTED_FULL_PARAMETERS,
        "device": "cpu",
        "threads": THREADS,
        "scaffold_report": report,
        "group_attach": group_meta,
        "acceptance": acceptance,
        "wall_seconds": float(time.perf_counter() - started),
    }
    payload["passed"] = bool(
        acceptance["replay_gate_passed"]
        and acceptance["row_ids_are_positional"]
        and payload["checkpoint_sha_match"]
        and frozen
        and not official
        and acceptance["single_vs_batch_max_abs"] <= READER_REPLAY_TOL
        and acceptance["reversed_order_max_abs"] <= READER_REPLAY_TOL
        and (
            split != "valid"
            or (
                acceptance.get("valid_replay_gate_passed", False)
                and abs(acceptance["original_MAE_recomputed"] - EXPECTED_VALID_MAE)
                <= REPLAY_MAE_TOL
            )
        )
    )
    official_test_blocker(payload)
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[export:{split}] rows={acceptance['n_rows']} replay_mae={_fmt(acceptance['replay_MAE'])} "
        f"reader_delta={acceptance['reader_replay_max_abs']:.2e} passed={payload['passed']} "
        f"wall={payload['wall_seconds']:.1f}s",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(f"export acceptance failed for split={split}")
    return cache_path


# ---------------------------------------------------------------------------
# stage: fit (train only; the valid split is not touched here)
# ---------------------------------------------------------------------------


def _maybe_export_train() -> Path:
    if not TRAIN_CACHE.is_file():
        export_split("train")
    return TRAIN_CACHE


def stage_fit() -> dict[str, Any]:
    _ensure_dirs()
    configure()
    if FIT_JSON.is_file() and MODEL_PATH.is_file():
        print("[fit] cache hit", flush=True)
        return _read_json(FIT_JSON)
    _maybe_export_train()
    args = argparse.Namespace(train=TRAIN_CACHE, out=FIT_DIR)
    try:
        rch.fit(args)
    except SystemExit as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"cached head fit failed: {exc}") from exc
    fit_record = _read_json(FIT_JSON)
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    state = _backbone_state_dict()
    fit_record.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "backbone_state_sha256_no_reader": _state_hash(_exclude_reader(state)),
            "preflight_backbone_state_sha256_no_reader": preflight[
                "backbone_state_sha256_no_reader"
            ],
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256": _sha256_file(CHECKPOINT),
            "valid_opened_during_fit": False,
            "official_test_loaded": False,
        }
    )
    official_test_blocker(fit_record)
    _write_json(FIT_JSON, fit_record)
    print(
        f"[fit] selected_lambda={fit_record['selected_lambda']} "
        f"train_MAE={_fmt(fit_record['train_MAE'])}",
        flush=True,
    )
    return fit_record


# ---------------------------------------------------------------------------
# stage: evaluate (single paired screen on the frozen valid split)
# ---------------------------------------------------------------------------


def stage_evaluate() -> dict[str, Any]:
    _ensure_dirs()
    configure()
    if not (FIT_JSON.is_file() and MODEL_PATH.is_file()):
        stage_fit()
    # The valid cache is a transform-only replay of the same frozen backbone.
    if not VALID_CACHE.is_file():
        export_split("valid")
    args = argparse.Namespace(model=MODEL_PATH, valid=VALID_CACHE, out=EVAL_JSON)
    rch.evaluate(args)
    result = _read_json(EVAL_JSON)
    valid_provenance = _read_json(VALID_CACHE.with_suffix(".provenance.json"))
    result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "checkpoint_sha256": _sha256_file(CHECKPOINT),
            "train_cache_sha256": _sha256_file(TRAIN_CACHE),
            "valid_cache_sha256": _sha256_file(VALID_CACHE),
            "head_model_sha256": _sha256_file(MODEL_PATH),
            "train_fit_json": str(FIT_JSON),
            "valid_cache_provenance": valid_provenance,
            "backbone_state_sha256_no_reader": valid_provenance[
                "backbone_state_sha256_no_reader"
            ],
            "original_MAE_replay_check": float(
                valid_provenance["acceptance"]["replay_MAE"]
            ),
            "published_valid_MAE": EXPECTED_VALID_MAE,
            "valid_replay_delta": float(
                valid_provenance["acceptance"]["replay_MAE"] - EXPECTED_VALID_MAE
            ),
            "valid_replay_gate_passed": bool(
                abs(
                    valid_provenance["acceptance"]["replay_MAE"] - EXPECTED_VALID_MAE
                )
                <= REPLAY_MAE_TOL
            ),
            "official_valid_loaded": True,
            "official_test_loaded": False,
        }
    )
    result["passed"] = bool(
        result["verdict"] == "GRAPH_DICTIONARY_READOUT_PROMISING"
        and result["valid_replay_gate_passed"]
    )
    official_test_blocker(result)
    _write_json(EVAL_JSON, result)
    print(
        f"[evaluate] original={_fmt(result['original_MAE'])} new={_fmt(result['graph_dictionary_MAE'])} "
        f"gain={_fmt(result['gain'])} verdict={result['verdict']}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# stage: scaffold smoke (real mini-batch gate for the vendored Torch readout)
# ---------------------------------------------------------------------------


SCAFFOLD_SMOKE_JSON = DEPLOY_DIR / "scaffold_smoke.json"


def stage_scaffold_smoke() -> dict[str, Any]:
    """Run the vendored ``torch_readout_scaffold`` on a real train mini-batch.

    This is **not** deployment acceptance and never touches the official valid
    split: it only proves that the provider's PyTorch scaffold executes in this
    repository and reproduces the cached NumPy folded head on a real batch.
    """
    _ensure_dirs()
    configure()
    dictionary, coef, median, checkpoint_sha = rch.load_model(MODEL_PATH)
    with np.load(TRAIN_CACHE, allow_pickle=False) as z:
        r = z["R"].astype(np.float64)
        y = z["y"].astype(np.float64)
    limit = 128
    reference = pd.folded_predict(dictionary, r[:limit], coef, median)
    train = p1run.load_split("train", subset=limit)
    loader = p1.make_env_loader(train, limit, False, 0)
    batch = next(iter(loader))
    mask = cm.C6_MASK
    model, readout = _deployed_model()
    model.eval()
    calls = {"count": 0}
    handle = model.local_dictionary_bridge.register_forward_hook(
        lambda *_a: calls.__setitem__("count", calls["count"] + 1)
    )
    try:
        with torch.no_grad():
            prediction = model(batch, mask=mask).view(-1).double().numpy()
    finally:
        handle.remove()
    max_delta = float(np.max(np.abs(prediction - reference)))
    mae_delta = float(
        abs(np.mean(np.abs(prediction - y[:limit])) - np.mean(np.abs(reference - y[:limit])))
    )
    dtype_ok = bool(model.forward(batch, mask=mask).dtype == torch.float32)
    buffer_names = {name for name, _ in readout.named_buffers()}
    buffers_ok = {"mean", "scale", "weights", "centers", "values", "offset"} <= buffer_names
    state = model.state_dict()
    rebuilt, _ = _deployed_model()
    rebuilt.load_state_dict(state)
    rebuilt.eval()
    with torch.no_grad():
        replay = rebuilt(batch, mask=mask).view(-1).double().numpy()
    roundtrip = float(np.max(np.abs(replay - prediction)))
    blocker_raised = False
    try:
        sc.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocker_raised = True
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "scope": "train-only scaffold execution gate; NOT deployment acceptance",
        "n_graphs": limit,
        "checkpoint_sha": checkpoint_sha,
        "max_abs_vs_cached_numpy": max_delta,
        "max_abs_tolerance": DEPLOY_MAX_TOL,
        "MAE_delta_vs_cached_numpy": mae_delta,
        "MAE_delta_tolerance": DEPLOY_MAE_TOL,
        "backbone_forward_calls": int(calls["count"]),
        "returns_original_dtype": dtype_ok,
        "prototypes_and_scaler_are_buffers": bool(buffers_ok),
        "state_dict_roundtrip_max_abs": roundtrip,
        "official_test_blocker_raised": bool(blocker_raised),
        "deployment_acceptance": "NOT_RUN_NO_POSITIVE_SCREEN",
        "scaffold_smoke": None,
    }
    payload["scaffold_smoke"] = (
        "PASS"
        if (
            max_delta <= DEPLOY_MAX_TOL
            and mae_delta <= DEPLOY_MAE_TOL
            and calls["count"] == 1
            and dtype_ok
            and buffers_ok
            and roundtrip <= DEPLOY_MAX_TOL
            and blocker_raised
        )
        else "FAIL"
    )
    official_test_blocker(payload)
    _write_json(SCAFFOLD_SMOKE_JSON, payload)
    print(
        f"[scaffold-smoke] max_delta={max_delta:.2e} mae_delta={mae_delta:.2e} "
        f"calls={calls['count']} result={payload['scaffold_smoke']}",
        flush=True,
    )
    if payload["scaffold_smoke"] != "PASS":
        raise RuntimeError("vendored PyTorch scaffold smoke gate failed")
    return payload


# ---------------------------------------------------------------------------
# stage: cached-readout diagnostic (read-only; no fit, no new configuration)
# ---------------------------------------------------------------------------


DIAG_JSON = RESULTS_DIR / "readout_diagnostic.json"
VENDOR_DIR = TRACK_ROOT / "experiments/luyin16/graph_dictionary_readout/v1_20261002"
VENDOR_SOURCES = (
    VENDOR_DIR / "prototype_dictionary.py",
    VENDOR_DIR / "run_cached_head.py",
    VENDOR_DIR / "torch_readout_scaffold.py",
    VENDOR_DIR / "extract_full_features.py",
)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _mae_rmse(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = prediction - target
    return {
        "MAE": float(np.mean(np.abs(delta))),
        "RMSE": float(np.sqrt(np.mean(delta * delta))),
    }


def stage_diagnose() -> dict[str, Any]:
    """Read-only cache diagnostic: replay the frozen head and one fixed SVD
    projection onto the frozen train ``p_base``; no new configuration is chosen.
    """
    _ensure_dirs()
    configure()
    watched = {
        "model.npz": MODEL_PATH,
        "train_cache": TRAIN_CACHE,
        "valid_cache": VALID_CACHE,
        "checkpoint": CHECKPOINT,
    }
    watched.update({f"source::{path.name}": path for path in VENDOR_SOURCES})
    hashes_before = {name: _sha256_file(path) for name, path in watched.items()}

    with np.load(TRAIN_CACHE, allow_pickle=False) as z:
        r_train = z["R"].astype(np.float64)
        y_train = z["y"].astype(np.float64)
        p_base_train = z["p_base"].astype(np.float64)
        ids_train = z["ids"].copy()
        train_split = str(z["split"].item())
        train_frozen = bool(z["frozen_backbone"].item())
        train_test = bool(z["official_test_loaded"].item())
        train_chk = str(z["checkpoint_sha"].item())
        train_groups = z["group_ids"].copy() if "group_ids" in z.files else None
    with np.load(VALID_CACHE, allow_pickle=False) as z:
        r_valid = z["R"].astype(np.float64)
        y_valid = z["y"].astype(np.float64)
        p_base_valid = z["p_base"].astype(np.float64)
        ids_valid = z["ids"].copy()
        valid_split = str(z["split"].item())
        valid_frozen = bool(z["frozen_backbone"].item())
        valid_test = bool(z["official_test_loaded"].item())
        valid_chk = str(z["checkpoint_sha"].item())

    dictionary, coef, median, checkpoint_sha = rch.load_model(MODEL_PATH)
    checkpoint_now = _sha256_file(CHECKPOINT)
    state_now = _state_hash(_exclude_reader(_backbone_state_dict()))
    preflight = _read_json(RESULTS_DIR / "preflight.json")

    # -- 1. replay the existing prototype head ---------------------------------
    kernel_train = dictionary.kernel(r_train)
    kernel_valid = dictionary.kernel(r_valid)
    pred_train = median + np.column_stack(
        [np.ones(len(r_train)), kernel_train @ dictionary.inverse_root]
    ) @ coef
    pred_valid = median + np.column_stack(
        [np.ones(len(r_valid)), kernel_valid @ dictionary.inverse_root]
    ) @ coef
    design_train = np.column_stack([np.ones(len(r_train)), kernel_train @ dictionary.inverse_root])
    design_valid = np.column_stack([np.ones(len(r_valid)), kernel_valid @ dictionary.inverse_root])

    # -- 2. one fixed float64 SVD least-squares projection onto train p_base ----
    projection_coef, _res, rank, singular = np.linalg.lstsq(
        design_train, p_base_train, rcond=1e-12
    )
    proj_train = design_train @ projection_coef
    proj_valid = design_valid @ projection_coef
    rank_floor = 1e-12 * float(singular[0]) if singular.size else 0.0

    # -- 3. nearest-prototype similarity (train stats exclude prototype rows) --
    selected = dictionary.selected_rows.astype(np.int64)
    kernel_train_self = kernel_train.copy()
    position = {int(row): int(col) for col, row in enumerate(selected)}
    for row in selected:
        kernel_train_self[int(row), position[int(row)]] = -np.inf
    nearest_train_all = kernel_train_self.max(axis=1)
    is_prototype = np.zeros(len(r_train), dtype=bool)
    is_prototype[selected] = True
    nearest_train_nonproto = nearest_train_all[~is_prototype]
    nearest_valid = kernel_valid.max(axis=1)

    # -- 4. hash / provenance checks -------------------------------------------
    hashes_after = {name: _sha256_file(path) for name, path in watched.items()}
    unchanged = {name: hashes_before[name] == hashes_after[name] for name in watched}
    diagnostics = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "scope": "read-only cached readout diagnostic; empirical L2 approximation only",
        "official_test_loaded": False,
        "device": "cpu",
        "threads": THREADS,
        "checks": {
            "checkpoint_sha_match": bool(
                checkpoint_now == train_chk == valid_chk == checkpoint_sha
            ),
            "backbone_state_sha256_no_reader": state_now,
            "backbone_state_matches_preflight": bool(
                state_now == preflight["backbone_state_sha256_no_reader"]
            ),
            "train_ids_positional": bool(np.array_equal(ids_train, np.arange(len(ids_train)))),
            "valid_ids_positional": bool(np.array_equal(ids_valid, np.arange(len(ids_valid)))),
            "train_split": train_split,
            "valid_split": valid_split,
            "frozen_backbone_flags": bool(train_frozen and valid_frozen),
            "official_test_flags": bool(train_test or valid_test),
            "train_groups_canonical": bool(
                train_groups is not None
                and len(np.unique(train_groups)) < len(train_groups)
            ),
            "n_train": int(len(y_train)),
            "n_valid": int(len(y_valid)),
            "feature_width": int(r_train.shape[1]),
        },
        "fit_json": {
            "train_MAE": float(_read_json(FIT_JSON)["train_MAE"]),
            "selected_lambda": float(_read_json(FIT_JSON)["selected_lambda"]),
            "head_dev_MAE_by_lambda": {
                str(row["lambda_value"]): float(row["head_dev_MAE"])
                for row in _read_json(FIT_JSON)["candidates"]
            },
            "n_prototypes": int(dictionary.centers.shape[0]),
            "active_dimensions": int(dictionary.centers.shape[1]),
            "seed": int(pd.SEED),
        },
        "replay": {
            "prototype_head": {
                "train": _mae_rmse(pred_train, y_train),
                "valid": _mae_rmse(pred_valid, y_valid),
            },
            "old_p_base_same_metric": {
                "train": _mae_rmse(p_base_train, y_train),
                "valid": _mae_rmse(p_base_valid, y_valid),
            },
        },
        "projection": {
            "target": "frozen train p_base only",
            "rcond": 1e-12,
            "design_shape_train": list(design_train.shape),
            "design_shape_valid": list(design_valid.shape),
            "numerical_rank": int(rank),
            "rank_floor": rank_floor,
            "singular_max": float(singular[0]),
            "singular_min": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]),
            "n_singular_above_floor": int(np.sum(singular > rank_floor)),
            "coef_l2_norm": float(np.linalg.norm(projection_coef)),
            "coef_max_abs": float(np.max(np.abs(projection_coef))),
            "vs_old_p_base": {
                "train": _mae_rmse(proj_train, p_base_train),
                "valid": _mae_rmse(proj_valid, p_base_valid),
            },
            "vs_target_y": {
                "train": _mae_rmse(proj_train, y_train),
                "valid": _mae_rmse(proj_valid, y_valid),
            },
        },
        "nearest_prototype_similarity": {
            "train_all_rows_self_excluded": _percentiles(nearest_train_all),
            "train_non_prototype_rows": _percentiles(nearest_train_nonproto),
            "valid_rows": _percentiles(nearest_valid),
            "n_train_prototype_rows": int(is_prototype.sum()),
            "note": "self-match set to -inf for the 256 train rows used as prototypes",
        },
        "immutability": {
            "hashes_before": hashes_before,
            "unchanged": unchanged,
            "all_unchanged": bool(all(unchanged.values())),
            "projection_coefficients_saved": False,
        },
    }
    diagnostics["passed"] = bool(
        diagnostics["checks"]["checkpoint_sha_match"]
        and diagnostics["checks"]["backbone_state_matches_preflight"]
        and diagnostics["checks"]["train_ids_positional"]
        and diagnostics["checks"]["valid_ids_positional"]
        and diagnostics["checks"]["frozen_backbone_flags"]
        and not diagnostics["checks"]["official_test_flags"]
        and diagnostics["immutability"]["all_unchanged"]
    )
    official_test_blocker(diagnostics)
    _write_json(DIAG_JSON, diagnostics)
    print(json.dumps(diagnostics["replay"], indent=2), flush=True)
    print(json.dumps(diagnostics["projection"], indent=2), flush=True)
    print(f"[diagnose] passed={diagnostics['passed']}", flush=True)
    return diagnostics


# ---------------------------------------------------------------------------
# stage: deploy acceptance (positive signal only)
# ---------------------------------------------------------------------------


def _deployed_model() -> tuple[sc.LatentScaleSEM108, trs.GraphPrototypeReadout]:
    model = _soup_model()
    readout = trs.GraphPrototypeReadout(MODEL_PATH)
    model.reader = readout
    return model, readout


def stage_deploy() -> dict[str, Any]:
    _ensure_dirs()
    configure()
    if EVAL_JSON.is_file():
        evaluated = _read_json(EVAL_JSON)
        if evaluated.get("verdict") != "GRAPH_DICTIONARY_READOUT_PROMISING":
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": _git_commit(),
                "official_test_loaded": False,
                "status": "SKIPPED_NO_POSITIVE_SCREEN",
                "verdict": evaluated.get("verdict"),
                "note": "deployment acceptance is purchased only after a promising screen",
                "deployment_acceptance": "NOT_RUN",
            }
            official_test_blocker(payload)
            _write_json(DEPLOY_JSON, payload)
            print("[deploy] skipped: screen not promising", flush=True)
            return payload
    if not EVAL_JSON.is_file():
        raise RuntimeError("deploy requires a completed evaluate stage")
    evaluated = _read_json(EVAL_JSON)
    valid = p1run.load_split("valid")
    loader = p1.make_env_loader(valid, 128, False, 0)
    mask = cm.C6_MASK
    model, readout = _deployed_model()
    model.eval()

    backbone_calls = {"count": 0}
    handle = model.local_dictionary_bridge.register_forward_hook(
        lambda *_a: backbone_calls.__setitem__("count", backbone_calls["count"] + 1)
    )
    predictions: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for batch in loader:
                predictions.append(
                    model(batch, mask=mask).view(-1).cpu().double().numpy()
                )
    finally:
        handle.remove()
    prediction = np.concatenate(predictions)

    with np.load(EVAL_JSON.with_suffix(".predictions.npz"), allow_pickle=False) as z:
        cached = z["p_graph_dictionary"].astype(np.float64)
        y = z["y"].astype(np.float64)
        p_base = z["p_base"].astype(np.float64)
    max_delta = float(np.max(np.abs(prediction - cached)))
    mae_delta = float(abs(np.mean(np.abs(prediction - y)) - np.mean(np.abs(cached - y))))

    # Forward-pass shape/order invariance on a real mini-batch.
    batch = next(iter(p1.make_env_loader(valid[:32], 32, False, 0)))
    baseline = model(batch, mask=mask).view(-1).double()
    with torch.no_grad():
        pieces = torch.cat(
            [
                model(p1.env_collate([valid[i]]), mask=mask).view(-1).double()
                for i in range(min(8, len(valid)))
            ]
        )
    batch_invariance = float((baseline[: len(pieces)] - pieces).abs().max())
    # float64 kernel path returns the old output dtype.
    dtype_ok = bool(model.forward(batch, mask=mask).dtype == torch.float32)

    # Online buffers are constant and are not parameters.
    buffer_names = {name for name, _ in readout.named_buffers()}
    parameter_names = {name for name, _ in readout.named_parameters()}
    prototype_buffers = bool(
        {"mean", "scale", "weights", "centers", "values", "offset"} <= buffer_names
    )
    no_online_fit = bool(model.training is False)
    with torch.no_grad():
        first = readout.centers.clone()
        _ = readout(torch.zeros((2, 814), dtype=torch.float32))
        constant = bool(torch.equal(first, readout.centers))

    # State-dict round-trip replay (never a pickled module instance).
    state = model.state_dict()
    rebuilt, _ = _deployed_model()
    rebuilt.load_state_dict(state)
    rebuilt.eval()
    with torch.no_grad():
        replay = rebuilt(batch, mask=mask).view(-1).double()
    roundtrip = float((baseline - replay).abs().max())

    blocker_raised = False
    try:
        sc.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocker_raised = True
    test_path = scale.TRACK_ROOT / "data"  # never touched; documented only
    del test_path

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "device": "cpu",
        "threads": THREADS,
        "status": "RUN",
        "n_valid": int(len(prediction)),
        "backbone_forward_calls": int(backbone_calls["count"]),
        "max_abs_vs_cached_numpy": max_delta,
        "max_abs_tolerance": DEPLOY_MAX_TOL,
        "MAE_delta_vs_cached_numpy": mae_delta,
        "MAE_delta_tolerance": DEPLOY_MAE_TOL,
        "deployed_MAE": float(np.mean(np.abs(prediction - y))),
        "cached_numpy_MAE": float(np.mean(np.abs(cached - y))),
        "original_Full_MAE": float(np.mean(np.abs(p_base - y))),
        "batch_size_invariance_max_abs": batch_invariance,
        "returns_original_dtype": dtype_ok,
        "prototypes_and_scaler_are_buffers": prototype_buffers,
        "readout_parameters": sorted(parameter_names),
        "readout_buffer_names": sorted(buffer_names),
        "online_does_not_fit": bool(no_online_fit and constant),
        "state_dict_roundtrip_max_abs": roundtrip,
        "official_test_blocker_raised": bool(blocker_raised),
        "one_backbone_path_calls": int(backbone_calls["count"]),
    }
    payload["passed"] = bool(
        max_delta <= DEPLOY_MAX_TOL
        and mae_delta <= DEPLOY_MAE_TOL
        and backbone_calls["count"] == 1
        and batch_invariance <= DEPLOY_MAX_TOL
        and dtype_ok
        and prototype_buffers
        and no_online_fit
        and constant
        and roundtrip <= DEPLOY_MAX_TOL
        and blocker_raised
    )
    payload["deployment_acceptance"] = (
        "PASS_ONE_UNIT" if payload["passed"] else "NOT_COMPLETED"
    )
    official_test_blocker(payload)
    _write_json(DEPLOY_JSON, payload)
    torch.save(state, DEPLOY_DIR / "deployed_state_dict.pt")
    print(
        f"[deploy] max_delta={max_delta:.2e} mae_delta={mae_delta:.2e} "
        f"calls={backbone_calls['count']} passed={payload['passed']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    fit_record = _read_json(FIT_JSON)
    evaluated = _read_json(EVAL_JSON)
    deploy = _read_json(DEPLOY_JSON) if DEPLOY_JSON.is_file() else None
    scaffold_smoke = (
        _read_json(SCAFFOLD_SMOKE_JSON) if SCAFFOLD_SMOKE_JSON.is_file() else None
    )
    train_prov = _read_json(TRAIN_CACHE.with_suffix(".provenance.json"))
    valid_prov = _read_json(VALID_CACHE.with_suffix(".provenance.json"))
    promising = evaluated["verdict"] == "GRAPH_DICTIONARY_READOUT_PROMISING"
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "seed": SEED,
        "official_test_loaded": False,
        "official_valid_loaded": True,
        "parent": "Full m=3 seed-0 parameter soup (408,651 parameters)",
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "backbone_state_sha256_no_reader": preflight[
            "backbone_state_sha256_no_reader"
        ],
        "split_fingerprint": preflight["split_fingerprint"],
        "feature_width": 814,
        "n_prototypes": pd.N_PROTOTYPES,
        "lambda_grid": list(pd.LAMBDA_GRID),
        "selected_lambda": fit_record["selected_lambda"],
        "head_dev_MAE": {
            str(row["lambda_value"]): row["head_dev_MAE"]
            for row in fit_record["candidates"]
        },
        "train_cache_sha256": train_prov["cache_sha256"],
        "valid_cache_sha256": valid_prov["cache_sha256"],
        "model_sha256": evaluated["head_model_sha256"],
        "group_attach": train_prov["group_attach"],
        "original_MAE": evaluated["original_MAE"],
        "published_valid_MAE": EXPECTED_VALID_MAE,
        "valid_replay_delta": evaluated["valid_replay_delta"],
        "valid_replay_gate_passed": evaluated["valid_replay_gate_passed"],
        "graph_dictionary_MAE": evaluated["graph_dictionary_MAE"],
        "gain": evaluated["gain"],
        "better_fraction": evaluated["better_fraction"],
        "gain_median": evaluated["gain_median"],
        "five_ID_bin_gains": evaluated["five_ID_bin_gains"],
        "five_ID_bins_improved": int(sum(x > 0 for x in evaluated["five_ID_bin_gains"])),
        "gates": {
            "gain_ge_0.006": bool(evaluated["gain"] >= 0.006),
            "MAE_le_0.113": bool(evaluated["graph_dictionary_MAE"] <= 0.113),
            "four_of_five_bins": bool(
                sum(x > 0 for x in evaluated["five_ID_bin_gains"]) >= 4
            ),
        },
        "verdict": evaluated["verdict"],
        "deployment_acceptance": (
            "NOT_RUN_NO_POSITIVE_SCREEN"
            if deploy is None or deploy.get("status") == "SKIPPED_NO_POSITIVE_SCREEN"
            else deploy.get("deployment_acceptance")
        ),
        "scaffold_smoke": None if scaffold_smoke is None else scaffold_smoke["scaffold_smoke"],
        "scaffold_smoke_scope": (
            None if scaffold_smoke is None else scaffold_smoke["scope"]
        ),
        "scaffold_smoke_max_abs_vs_numpy": (
            None if scaffold_smoke is None else scaffold_smoke["max_abs_vs_cached_numpy"]
        ),
        "limitation": (
            "One backbone checkpoint; the official valid split has been reused across many "
            "historical rounds and selected the Full soup, so this is an exploratory screen, "
            "not an independent confirmation."
        ),
    }
    summary["passed"] = bool(
        promising
        and evaluated["valid_replay_gate_passed"]
        and all(summary["gates"].values())
    )
    official_test_blocker(summary)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, fit_record, evaluated, deploy, valid_prov)
    _write_decision(summary, deploy)
    print(f"[analysis] verdict={summary['verdict']}", flush=True)
    return summary


def _write_report(
    summary: Mapping[str, Any],
    fit_record: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    deploy: Mapping[str, Any] | None,
    valid_prov: Mapping[str, Any],
) -> None:
    lines = [
        "# zinc_graph_dictionary_readout_v1 — frozen Full graph-dictionary readout",
        "",
        "CPU only · official test never loaded · single frozen Full seed-0 soup.",
        "",
        "## A. Provenance",
        "",
        f"- commit `{summary['git_commit']}` · device `cpu` · threads `{summary['threads']}` · seed `{summary['seed']}`",
        f"- parent checkpoint `{summary['checkpoint_sha256']}` (408,651 parameters, soup valid MAE `{_fmt(EXPECTED_VALID_MAE)}`)",
        f"- backbone state (reader excluded) `{summary['backbone_state_sha256_no_reader']}`",
        f"- split fingerprint `{summary['split_fingerprint']}`",
        f"- train cache `{summary['train_cache_sha256']}` · valid cache `{summary['valid_cache_sha256']}` · frozen head `{summary['model_sha256']}`",
        "- `official_test_loaded = false`",
        "",
        "## B. Frozen representation replay",
        "",
        f"- scaffold valid replay MAE `{_fmt(valid_prov['acceptance']['replay_MAE'])}` against published `{_fmt(EXPECTED_VALID_MAE)}` (delta `{valid_prov['acceptance']['valid_MAE_delta']:+.2e}`)",
        f"- reader replay of the captured 814-D tensor: max per-molecule `{valid_prov['acceptance']['reader_replay_max_abs']:.2e}`",
        f"- train groups: {summary['group_attach']['group_source']} (canonical `{summary['group_attach']['canonical']}`)",
        "",
        "## C. Fixed dictionary fit (train only)",
        "",
        f"- K=`{summary['n_prototypes']}` seed `{pd.SEED}` · asinh / train-fit standardiser / four-block balance",
        f"- head dev MAE by lambda: {json.dumps(summary['head_dev_MAE'])}",
        f"- selected lambda `{summary['selected_lambda']}` (equal dev MAE breaks to the larger lambda)",
        f"- final head train MAE `{_fmt(fit_record['train_MAE'])}` · solver iterations `{fit_record['final_solver']['iterations']}` · gap `{fit_record['final_solver']['certificate']['gap']:.2e}`",
        "- the valid split is not accepted by the fit stage and is opened only for the paired screen",
        "",
        "## D. One paired screen on the frozen official valid",
        "",
        f"- original Full H0 readout MAE `{_fmt(summary['original_MAE'])}`",
        f"- graph-dictionary readout MAE `{_fmt(summary['graph_dictionary_MAE'])}`",
        f"- absolute gain `{summary['gain']:+.6f}` (gate >= 0.006) · new MAE <= 0.113 gate `{summary['gates']['MAE_le_0.113']}`",
        f"- per-molecule improvement fraction `{summary['better_fraction']:.4f}` · median gain `{summary['gain_median']:+.6f}`",
        f"- five fixed-ID bins: {[round(x, 6) for x in summary['five_ID_bin_gains']]} ({summary['five_ID_bins_improved']}/5 improved)",
        "",
        "## E. Deployment acceptance",
        "",
        (
            "Not purchased — the screen did not meet all frozen gates."
            if summary["deployment_acceptance"]
            in ("NOT_RUN_NO_POSITIVE_SCREEN", None)
            else f"`{summary['deployment_acceptance']}` · max vs NumPy `{deploy['max_abs_vs_cached_numpy']:.2e}`, "
            f"MAE delta `{deploy['MAE_delta_vs_cached_numpy']:.2e}`, backbone calls `{deploy['backbone_forward_calls']}`"
        ),
        "",
        "## F. Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        f"> {summary['limitation']}",
        "",
        "## G. Storage and capacity accounting",
        "",
        "| item | count | storage | nature |",
        "|---|---|---|---|",
        "| frozen Full backbone parameters | 408,651 | 1,634,604 B (float32) | trained, frozen this round |",
        "| prototype / normaliser / scaler buffers | mean 814, scale 814, keep 814, weights 814, centers 256x598, inverse_root 256x256, bandwidth 1, selected_rows 256, spectrum 256 | 1,773,478 B (float64) | fixed transforms, not trainable capacity |",
        "| fitted readout coefficients | 257 | 2,056 B (float64) | the only fitted head values |",
        "| head container `model.npz` | - | 1,779,346 B | prototype buffers + 257 coefficients + metadata |",
        "",
        "Only the 257 coefficients are fit; the backbone and the prototype/scaler buffers are fixed.",
        "",
        "## H. Evidence discipline",
        "",
        "- One backbone checkpoint, one reused official-valid screen; no significance claim.",
        "- A negative result excludes only this fixed dictionary / kernel / lambda family; it does not prove the 814-D representation is sufficient or insufficient.",
        "- No old weak-ridge proxy, no unmatched historical delta, no inactive-ablation gain.",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(summary: Mapping[str, Any], deploy: Mapping[str, Any] | None) -> None:
    if summary["verdict"] == "GRAPH_DICTIONARY_READOUT_PROMISING":
        body = [
            "# DECISION — zinc_graph_dictionary_readout_v1",
            "",
            "**GRAPH_DICTIONARY_READOUT_PROMISING**",
            "",
            f"The frozen Full 814-D reader input supports a fixed graph-level prototype dictionary "
            f"readout with absolute gain `{summary['gain']:+.6f}` and MAE `{_fmt(summary['graph_dictionary_MAE'])}`, "
            "against the replayed same-checkpoint Full readout.",
            "",
            "One complete staged single-model result; a fresh full neural training run is not purchased in this round.",
            "Independent seed / matched-control confirmation requires a new task.",
            "",
        ]
    else:
        body = [
            "# DECISION — zinc_graph_dictionary_readout_v1",
            "",
            "**GRAPH_DICTIONARY_READOUT_STOP**",
            "",
            f"Fixed graph-dictionary readout MAE `{_fmt(summary['graph_dictionary_MAE'])}` vs replayed Full "
            f"`{_fmt(summary['original_MAE'])}` (gain `{summary['gain']:+.6f}`); no frozen gate set was met.",
            "",
            "This does not purchase a new full neural training run. It excludes only this fixed "
            "prototype dictionary / Gaussian kernel / MAE+L2 head configuration; it does not prove the "
            "814-D representation is theoretically sufficient or insufficient.",
            "",
            "No K increase, no bandwidth scan, no normaliser or object swap, no backbone training.",
            "",
        ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(body), encoding="utf-8")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


STAGES = (
    "preflight",
    "export",
    "fit",
    "evaluate",
    "smoke",
    "diagnose",
    "deploy",
    "analysis",
    "chain",
)


def run_stage(stage: str) -> dict[str, Any]:
    configure()
    if stage == "preflight":
        return stage_preflight()
    if stage == "export":
        stage_preflight()
        export_split("train")
        export_split("valid")
        return _read_json(RESULTS_DIR / "summary.json") if (RESULTS_DIR / "summary.json").exists() else {}
    if stage == "fit":
        stage_preflight()
        return stage_fit()
    if stage == "evaluate":
        stage_preflight()
        stage_evaluate()
        stage_scaffold_smoke()
        stage_deploy()
        return stage_analysis()
    if stage == "smoke":
        stage_scaffold_smoke()
        return _read_json(SCAFFOLD_SMOKE_JSON)
    if stage == "diagnose":
        return stage_diagnose()
    if stage == "deploy":
        return stage_deploy()
    if stage == "analysis":
        return stage_analysis()
    if stage == "chain":
        stage_preflight()
        stage_fit()
        stage_evaluate()
        stage_scaffold_smoke()
        stage_deploy()
        return stage_analysis()
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


__all__ = [
    "PROTOCOL_VERSION",
    "RESULTS_DIR",
    "CACHE_DIR",
    "FIT_DIR",
    "DEPLOY_DIR",
    "TRAIN_CACHE",
    "VALID_CACHE",
    "MODEL_PATH",
    "FIT_JSON",
    "EVAL_JSON",
    "DEPLOY_JSON",
    "DIAG_JSON",
    "SCAFFOLD_SMOKE_JSON",
    "CHECKPOINT",
    "EXPECTED_VALID_MAE",
    "EXPECTED_FULL_PARAMETERS",
    "GROUP_CACHE",
    "configure",
    "stage_preflight",
    "export_split",
    "stage_fit",
    "stage_evaluate",
    "stage_scaffold_smoke",
    "stage_diagnose",
    "stage_deploy",
    "stage_analysis",
    "run_stage",
    "official_test_blocker",
]
