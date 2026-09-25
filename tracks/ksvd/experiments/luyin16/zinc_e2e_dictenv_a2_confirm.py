"""E2E-DictEnv-A2-Confirm runner — frozen exact-OMP 320-epoch pairing confirmation.

Preregistration: ``tracks/ksvd/notes/e2e_dictenv_a2_confirm_preregistration.md``
(frozen and committed before any confirmation training).  Core module:
``tracks/ksvd/experiments/luyin16/e2e_dictenv_a2_confirm.py``.

Stage order (each stage refuses to run out of order):

``verify continuation screen decision report``  (+ ``smoke``, plumbing only)
``arm --arm {REAL,INDEP}`` trains exactly one arm (the user-authorised parallel
schedule entry point; the frozen protocol is identical either way).

``verify`` re-reads the completed parent A2 artifact identity *by reference*
(no cache, scaler, dictionary or OMP recomputation), pins the live dictionary
shas, records the completed ``TOPO-OMP`` 320-epoch result and the completed lite
160-epoch screen as fixed context, and stops with ``ARTIFACT_IDENTITY_FAILURE``
if anything moved.  ``continuation`` checks — and records — whether the lite
160-epoch segments could be continued exactly; only if *both* arms are provably
resumable would exact continuation be used, and mixing regimes is prohibited, so
the round otherwise restarts both arms as fresh matched 320-epoch runs.
``screen`` drives the unchanged frozen A1 trainer (seed 0, matched init, frozen
exact-OMP codes, parent horizon 320) for ``REAL`` then ``INDEP`` sequentially,
one CUDA process at a time, on physical **GPU1** only.  ``decision`` applies the
frozen decision matrix (``G_pair_320 >= 0.003``; ``Delta_vs_TOPO <= 0.003``) and
writes the paired late-window and 160 -> 320 diagnostics.  Then the round stops.

Never auto-started here: seed 1, PCA32, continuity-v2, IHT qualification,
task-coupled E2E, mechanism interventions, DenseTied specificity, TOPO
retraining, K64, s12, the official test and every architecture/hyper-parameter
sweep — all recorded as ``DEFERRED_PENDING_USER_AUTHORIZATION``.

Official ZINC **test is never loaded**; all CUDA work runs on physical **GPU1**
only (``CUDA_VISIBLE_DEVICES == "1"``, enforced by the shared frozen policy).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_confirm as a2conf
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_lite as a2lite
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as a1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2 as a2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = a2conf.PROTOCOL_VERSION
ROUND = a2conf.ROUND
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_a2_confirm"
STATE_DIR = RESULTS_DIR / "states"
CURVE_DIR = RESULTS_DIR / "curves"
#: read-only references: the truncated parent round and the lite screen
A2_RESULTS_DIR = a2run.RESULTS_DIR
A2_IDENTITY_PATH = A2_RESULTS_DIR / "artifact_identity.json"
A2_TOPO_PATH = A2_RESULTS_DIR / "omp_screen_topo.json"
A2_TOPO_CURVE = A2_RESULTS_DIR / "curves" / "omp_screen_T0_curve.csv"
LITE_RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_a2_lite"
A1_CACHE_DIR = a1run.CACHE_DIR

SEED = int(a2run.SEED)
HORIZON = int(a2conf.CONFIRM_HORIZON)
SMOKE_MOLECULES = 24
SMOKE_HORIZON = 1

CONFIRM_STAGES = ("verify", "continuation", "screen", "decision", "report")

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


def _set_device_policy(device: str) -> torch.device:
    """Frozen GPU policy (GPU1 only): delegated, never re-implemented."""
    return a2run._set_device_policy(device)


def _git_commit_for(path: str, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%h", *args, "--", path],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _prereg_commit() -> str:
    """Commit that last touched this round's preregistration."""
    return _git_commit_for(a2conf.PREREGISTRATION_NOTE)


def _prereg_rule_commit() -> str:
    """Commit that first introduced the preregistration (rules were frozen there)."""
    return _git_commit_for(a2conf.PREREGISTRATION_NOTE, "--diff-filter=A", "--follow")


def provenance(device: str = "cpu") -> dict[str, Any]:
    """Frozen provenance with the confirmation round's identity."""
    payload = a2run.provenance(device)
    payload.update(
        {
            "round": ROUND,
            "protocol_version": PROTOCOL_VERSION,
            "subtitle": a2conf.SUBTITLE,
            "parent_round": a2conf.PARENT_ROUND,
            "parent_protocol_version": a2conf.PARENT_PROTOCOL_VERSION,
            "parent_a2_commit": a2conf.PARENT_A2_COMMIT,
            "parent_a2_preregistration_commit": a2conf.PARENT_A2_PREREG_COMMIT,
            "predecessor_round": a2conf.PREDECESSOR_ROUND,
            "predecessor_protocol_version": a2conf.PREDECESSOR_PROTOCOL_VERSION,
            "preregistration_note": a2conf.PREREGISTRATION_NOTE,
            "preregistration_commit": _prereg_commit(),
            "preregistration_rule_commit": _prereg_rule_commit(),
            "confirm_horizon": int(HORIZON),
            "confirm_arms": list(a2conf.CONFIRM_ARMS),
            "material_threshold": float(a2conf.MATERIAL),
            "topo_tolerance": float(a2conf.TOPO_TOLERANCE),
            "topo_320_soup": float(a2conf.TOPO_320_SOUP),
            "deferred_stages": a2conf.deferred_stage_record(),
        }
    )
    return payload


_write_json = a2run._write_json
_read_json = a2run._read_json
a1_emits_into = a2run.a1_emits_into
matched_init_enforced = a2run.matched_init_enforced


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reemit(payload: Any, *, source: str, **extra: Any) -> dict[str, Any]:
    """Add this round's provenance to a payload produced by the frozen machinery."""
    out = dict(payload)
    out["source_protocol_version"] = out.get("protocol_version")
    out["reused_from"] = str(source)
    out["round"] = ROUND
    out["protocol_version"] = PROTOCOL_VERSION
    out["subtitle"] = a2conf.SUBTITLE
    out["parent_round"] = a2conf.PARENT_ROUND
    out["parent_protocol_version"] = a2conf.PARENT_PROTOCOL_VERSION
    out["predecessor_round"] = a2conf.PREDECESSOR_ROUND
    out["preregistration_note"] = a2conf.PREREGISTRATION_NOTE
    out["official_test_loaded"] = False
    out.update(extra)
    return out


def _require_verified() -> dict[str, Any]:
    path = RESULTS_DIR / "artifact_identity.json"
    if not path.exists():
        raise RuntimeError("verify stage must run first (artifact_identity.json missing)")
    payload = _read_json(path)
    if not payload.get("passed"):
        raise RuntimeError(
            f"{a2conf.VERDICT_IDENTITY_FAILURE}: frozen artifact provenance did not verify"
        )
    return payload


def _curve_path(tag: str) -> Path:
    return CURVE_DIR / f"{tag}_curve.csv"


def _arm_tag(arm: str) -> str:
    return f"{arm.lower()}_{int(HORIZON)}"


def _complete_curve(path: Path, horizon: int) -> bool:
    """Fail-closed: the curve must be finite *and* carry every epoch 1..horizon."""
    curve = a2lite.curve_or_none(path)
    if curve is None:
        return False
    return set(curve) == set(range(1, int(horizon) + 1))


# ---------------------------------------------------------------------------
# stage: verify (provenance by reference — nothing is recomputed)
# ---------------------------------------------------------------------------


def _topo_reference() -> dict[str, Any]:
    """Read the completed parent TOPO-OMP 320-epoch arm and check it is intact."""
    if not A2_TOPO_PATH.exists():
        raise RuntimeError(f"completed TOPO-OMP reference missing: {A2_TOPO_PATH}")
    payload = _read_json(A2_TOPO_PATH)
    soup = float(payload["soup"]["soup_valid_mae"])
    best = float(payload["best_valid_mae"])
    horizon = int(payload["horizon"])
    finite = _complete_curve(A2_TOPO_CURVE, a2conf.TOPO_320_HORIZON)
    return {
        "source": str(A2_TOPO_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(A2_TOPO_PATH),
        "arm": payload.get("arm"),
        "horizon": horizon,
        "soup_valid_mae": soup,
        "best_valid_mae": best,
        "best_epoch": int(payload["best_epoch"]),
        "soup_members": payload["soup"]["members"],
        "dictionary_sha256_f32": payload.get("dictionary_sha256_f32"),
        "frozen_dictionary": bool(payload.get("frozen_dictionary")),
        "curve_path": str(A2_TOPO_CURVE.relative_to(REPO_ROOT)),
        "curve_sha256": _sha256_file(A2_TOPO_CURVE) if A2_TOPO_CURVE.exists() else None,
        "curve_complete": bool(finite),
        "expected_soup": float(a2conf.TOPO_320_SOUP),
        "expected_best": float(a2conf.TOPO_320_BEST),
        "passed": bool(
            payload.get("arm") == a2conf.TOPO_320_ARM
            and horizon == a2conf.TOPO_320_HORIZON
            and soup == float(a2conf.TOPO_320_SOUP)
            and best == float(a2conf.TOPO_320_BEST)
            and bool(payload.get("frozen_dictionary"))
            and payload.get("dictionary_sha256_f32") == a2run.EXPECTED_DICT_SHA["TOPO"]
            and finite
        ),
        "kind": "completed parent result; never retrained in this round",
    }


def _lite_160_references() -> dict[str, Any]:
    """Read the completed lite 160-epoch screen as fixed context (160 -> 320)."""
    entries: dict[str, Any] = {}
    for arm in a2conf.CONFIRM_ARMS:
        tag = a2conf.LITE_160_TAG[arm]
        path = LITE_RESULTS_DIR / f"{tag}.json"
        payload = _read_json(path) if path.exists() else None
        curve = LITE_RESULTS_DIR / "curves" / f"{tag}_curve.csv"
        soup = None if payload is None else float(payload["soup"]["soup_valid_mae"])
        entries[arm] = {
            "source": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(path) if path.exists() else None,
            "horizon": None if payload is None else int(payload["horizon"]),
            "soup_valid_mae": soup,
            "best_valid_mae": None if payload is None else float(payload["best_valid_mae"]),
            "best_epoch": None if payload is None else int(payload["best_epoch"]),
            "expected_soup": float(a2conf.LITE_160_SOUP[arm]),
            "curve_complete": bool(_complete_curve(curve, a2conf.LITE_160_HORIZON)),
            "passed": bool(
                payload is not None
                and int(payload["horizon"]) == a2conf.LITE_160_HORIZON
                and soup == float(a2conf.LITE_160_SOUP[arm])
                and _complete_curve(curve, a2conf.LITE_160_HORIZON)
            ),
            "kind": "completed lite screen; diagnostic context only",
        }
    return entries


def verify_stage() -> dict[str, Any]:
    """Attach this round to the identity-verified frozen artifacts, by reference."""
    started = time.perf_counter()
    entries: dict[str, Any] = {}
    if not A2_IDENTITY_PATH.exists():
        raise RuntimeError(f"A2 artifact identity record missing: {A2_IDENTITY_PATH}")
    identity = _read_json(A2_IDENTITY_PATH)
    entries["a2_identity_file"] = {
        "path": str(A2_IDENTITY_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(A2_IDENTITY_PATH),
        "protocol_version": identity.get("protocol_version"),
        "git_commit": identity.get("git_commit"),
        "preregistration_commit": identity.get("preregistration_commit"),
        "all_passed": bool(identity.get("all_passed")),
        "n_entries": int(len(identity.get("entries", {}))),
        "seconds": identity.get("seconds"),
        "passed": bool(
            identity.get("protocol_version") == a2conf.PARENT_PROTOCOL_VERSION
            and identity.get("all_passed") is True
            and identity.get("git_commit") == a2conf.PARENT_A2_COMMIT
            and identity.get("preregistration_commit") == a2conf.PARENT_A2_PREREG_COMMIT
        ),
    }
    prereg_path = REPO_ROOT / a2conf.PREREGISTRATION_NOTE
    entries["confirm_preregistration"] = {
        "path": a2conf.PREREGISTRATION_NOTE,
        "sha256": _sha256_file(prereg_path) if prereg_path.exists() else None,
        "prereg_commit": _prereg_commit(),
        "prereg_rule_commit": _prereg_rule_commit(),
        "passed": bool(prereg_path.exists()),
    }
    required = list(a2conf.REQUIRED_A2_IDENTITY_ENTRIES)
    referenced = identity.get("entries", {})
    referenced_ok = {name: bool(referenced.get(name, {}).get("passed")) for name in required}
    entries["a2_identity_references"] = {
        "required": required,
        "passed": referenced_ok,
        "all_required_passed": bool(all(referenced_ok.values())),
    }
    producer: dict[str, Any] = {}
    for relative, expected in a2conf.A2_PRODUCER_SHA256.items():
        path = REPO_ROOT / relative
        observed = _sha256_file(path) if path.exists() else None
        producer[relative] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passed": observed == expected,
        }
    entries["producer_freeze"] = {
        "files": producer,
        "passed": bool(all(item["passed"] for item in producer.values())),
    }
    dictionaries: dict[str, Any] = {}
    for arm in a2conf.CONFIRM_ARMS:
        _D, sha = a1run.load_arm_dictionary(arm)
        recorded = referenced.get(f"dictionary_{arm}", {}).get("observed_sha256_f32")
        dictionaries[arm] = {
            "expected_sha256_f32": a2run.EXPECTED_DICT_SHA[arm],
            "observed_sha256_f32": sha,
            "a2_identity_sha256_f32": recorded,
            "passed": bool(sha == a2run.EXPECTED_DICT_SHA[arm] and recorded in (None, sha)),
        }
    entries["dictionary_live"] = dictionaries
    codes: dict[str, Any] = {}
    for arm in a2conf.CONFIRM_ARMS:
        for split in ("train", "valid"):
            path = A1_CACHE_DIR / f"omp_{arm}_{split}.pt"
            blob = torch.load(path, map_location="cpu", weights_only=True)
            expected_shape = (
                int(a2run.TRACKED_CACHE_SHAPES[split]["n_nodes"]), int(a1.DICT_K)
            )
            codes[f"{arm}_{split}"] = {
                "path": str(path),
                "sha256_file": _sha256_file(path),
                "shape": list(blob.shape),
                "expected_shape": list(expected_shape),
                "dtype": str(blob.dtype),
                "a2_identity_passed": bool(
                    referenced.get(f"omp_{arm}_{split}", {}).get("passed")
                ),
                "rows": int(blob.shape[0]),
                "passed": bool(
                    referenced.get(f"omp_{arm}_{split}", {}).get("passed")
                    and tuple(blob.shape) == expected_shape
                ),
            }
            del blob
    entries["omp_cache_bytes"] = codes
    topo = _topo_reference()
    entries["topo_320_reference"] = topo
    lite = _lite_160_references()
    entries["lite_160_reference"] = lite
    passed = bool(
        entries["a2_identity_file"]["passed"]
        and entries["confirm_preregistration"]["passed"]
        and entries["a2_identity_references"]["all_required_passed"]
        and entries["producer_freeze"]["passed"]
        and all(item["passed"] for item in dictionaries.values())
        and all(item["passed"] for item in codes.values())
        and topo["passed"]
        and all(item["passed"] for item in lite.values())
    )
    payload = {
        **provenance("cpu"),
        "stage": "confirm_verify",
        "kind": "provenance by reference; no cache, scaler, dictionary or OMP recomputation",
        "entries": entries,
        "passed": passed,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "artifact_identity.json", payload)
    _mark(
        "verify",
        "RUN" if passed else "FAIL",
        artifact="artifact_identity.json",
        seconds=payload["seconds"],
        device="cpu",
        reason="frozen references verified" if passed else a2conf.VERDICT_IDENTITY_FAILURE,
    )
    print(f"[confirm-verify] passed={passed}", flush=True)
    if not passed:
        raise RuntimeError(
            f"{a2conf.VERDICT_IDENTITY_FAILURE}: confirmation is not authorised"
        )
    return payload


# ---------------------------------------------------------------------------
# stage: continuation (fresh matched 320 vs exact resume — checked, not assumed)
# ---------------------------------------------------------------------------


def _state_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # pragma: no cover - corrupt checkpoint
        return []
    return sorted(str(key) for key in state)


def _trainer_capability() -> dict[str, Any]:
    """Static probe of the frozen trainer: can it continue an interrupted run?"""
    signature = inspect.signature(a1run.train_arm)
    source = inspect.getsource(a1run.train_arm)
    return {
        "signature_parameters": sorted(signature.parameters),
        "accepts_resume_argument": bool(
            {"resume", "start_epoch", "resume_from"} & set(signature.parameters)
        ),
        "writes_optimizer_state": bool("optimizer.state_dict" in source),
        "writes_rng_state": bool("get_rng_state" in source or "get_state()" in source),
        "loop_starts_at_epoch_one": bool("range(1, int(horizon) + 1)" in source),
        "kind": "static inspection of the frozen trainer; no training",
    }


def continuation_stage() -> dict[str, Any]:
    """Record whether the completed 160-epoch segments can be continued exactly."""
    _require_verified()
    started = time.perf_counter()
    per_arm: dict[str, Any] = {}
    for arm in a2conf.CONFIRM_ARMS:
        tag = a2conf.LITE_160_TAG[arm]
        raw = LITE_RESULTS_DIR / "states" / f"{tag}_raw_state.pt"
        soup = LITE_RESULTS_DIR / "states" / f"{tag}_soup_state.pt"
        curve = LITE_RESULTS_DIR / "curves" / f"{tag}_curve.csv"
        raw_keys = _state_keys(raw)
        soup_keys = _state_keys(soup)
        has_optimizer = bool(
            any("optim" in key.lower() for key in raw_keys + soup_keys)
        )
        curve_map = a2lite.curve_or_none(curve)
        final_epoch = None if curve_map is None else max(curve_map)
        per_arm[arm] = {
            "lite_tag": tag,
            "lite_state_dir": str(raw.parent.relative_to(REPO_ROOT)),
            "raw_state": {"path": str(raw.relative_to(REPO_ROOT)), "exists": raw.exists(),
                          "n_tensors": len(raw_keys), "tensor_keys": raw_keys[:12]},
            "soup_state": {"path": str(soup.relative_to(REPO_ROOT)), "exists": soup.exists(),
                           "n_tensors": len(soup_keys), "tensor_keys": soup_keys[:12]},
            "curve": {"path": str(curve.relative_to(REPO_ROOT)), "exists": curve.exists(),
                      "final_epoch": final_epoch},
        }
        evidence = {
            "model_state": {
                "present_and_provable": bool(raw_keys and soup_keys),
                "detail": "best-state and Top-5 soup *average* are stored",
            },
            "optimizer_state": {
                "present_and_provable": has_optimizer,
                "detail": "no optimizer state is written by the frozen trainer",
            },
            "scheduler_state": {
                "present_and_provable": False,
                "detail": "the frozen trainer constructs no scheduler, so no state exists",
            },
            "current_epoch": {
                "present_and_provable": bool(final_epoch == a2conf.LITE_160_HORIZON),
                "detail": f"curve reaches epoch {final_epoch!r}; the trainer always restarts at epoch 1",
            },
            "rng_states": {
                "present_and_provable": False,
                "detail": "no python/torch/CUDA RNG state is persisted",
            },
            "loader_order_state": {
                "present_and_provable": False,
                "detail": "the loader is rebuilt from SEED + offset per call; no sampler state is persisted",
            },
            "soup_member_states_for_full_trajectory": {
                "present_and_provable": False,
                "detail": (
                    "only the 5-member *average* is stored, member states cannot be "
                    "recovered, and Top-5 selection happens inside a single trainer call"
                ),
            },
        }
        per_arm[arm]["evidence"] = evidence
    decision = a2conf.continuation_decision(
        {arm: entry["evidence"] for arm, entry in per_arm.items()}
    )
    trainer = _trainer_capability()
    trainer["consistent_with_missing_state"] = bool(
        not trainer["writes_optimizer_state"] and not trainer["accepts_resume_argument"]
    )
    payload = {
        **provenance("cpu"),
        "stage": "confirm_continuation_check",
        "kind": "resume feasibility inspection; no training, no pseudo-resume",
        "horizon": int(HORIZON),
        "lite_horizon": int(a2conf.LITE_160_HORIZON),
        "required_state": list(a2conf.REQUIRED_RESUME_STATE),
        "per_arm": per_arm,
        "trainer_capability": trainer,
        **decision,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "continuation_mode.json", payload)
    _mark(
        "continuation",
        "RUN",
        artifact="continuation_mode.json",
        device="cpu",
        seconds=payload["seconds"],
        reason=f"mode={decision['mode']}; mixing_prohibited={decision['mixing_prohibited']}",
    )
    print(
        f"[confirm-continuation] mode={decision['mode']} "
        f"missing={ {a: m for a, m in decision['missing'].items()} }",
        flush=True,
    )
    return payload


def _require_continuation() -> dict[str, Any]:
    path = RESULTS_DIR / "continuation_mode.json"
    if not path.exists():
        raise RuntimeError("continuation stage must run before training")
    return _read_json(path)


# ---------------------------------------------------------------------------
# stage: screen (the two matched frozen-OMP arms at the parent horizon)
# ---------------------------------------------------------------------------


def _train_one_arm(arm: str, device: str, *, mode: str, schedule: str) -> dict[str, Any]:
    """Train exactly one matched frozen-OMP arm at the frozen confirmation horizon."""
    tag = _arm_tag(arm)
    with a1_emits_into(RESULTS_DIR), matched_init_enforced():
        payload = a1run.train_arm(
            arm,
            stage="omp",
            device=device,
            tag=tag,
            horizon=int(HORIZON),
            out_dir=RESULTS_DIR,
        )
    if int(payload.get("horizon", -1)) != int(HORIZON):
        raise RuntimeError(
            f"{arm}: trainer horizon {payload.get('horizon')!r} != frozen {HORIZON}"
        )
    if not bool(payload.get("frozen_dictionary")):
        raise RuntimeError(f"{arm}: dictionary was not frozen during the confirmation run")
    if payload.get("dictionary_sha256_f32") != a2run.EXPECTED_DICT_SHA[arm]:
        raise RuntimeError(f"{arm}: trained against an unexpected dictionary")
    if not _complete_curve(_curve_path(tag), int(HORIZON)):
        raise RuntimeError(f"{arm}: {HORIZON}-epoch curve missing or incomplete")
    payload = _reemit(
        payload,
        source="zinc_e2e_dictenv_a1.train_arm(stage='omp')",
        confirm_arm=f"ATTR-{arm}-OMP-{int(HORIZON)}",
        dictionary_kind="frozen_ksvd_k32_s8_omp",
        confirm_horizon=int(HORIZON),
        continuation_mode=mode,
        schedule=schedule,
    )
    payload["source_preregistration_commit"] = payload.get("preregistration_commit")
    payload["preregistration_commit"] = _prereg_commit()
    payload["parent_a2_preregistration_commit"] = a2conf.PARENT_A2_PREREG_COMMIT
    payload["parent_a2_commit"] = a2conf.PARENT_A2_COMMIT
    payload["predecessor_round"] = a2conf.PREDECESSOR_ROUND
    _write_json(RESULTS_DIR / f"{arm.lower()}_{int(HORIZON)}.json", payload)
    return payload


def _authorised_mode() -> str:
    """Shared gate for every training entry point (verify + continuation first)."""
    _require_verified()
    continuation = _require_continuation()
    mode = str(continuation.get("mode"))
    if mode not in a2conf.CONTINUATION_MODES:
        raise RuntimeError(f"unknown continuation mode {mode!r}")
    if mode == a2conf.CONTINUATION_RESUME:
        raise RuntimeError(
            "exact continuation was authorised but this runner implements only the "
            "fresh matched 320 regime; refusing to mix regimes"
        )
    return mode


def screen_stage(device: str = "cuda") -> dict[str, Any]:
    """Train ``ATTR-REAL-OMP`` and ``ATTR-INDEP-OMP`` at the frozen 320 horizon."""
    started = time.perf_counter()
    mode = _authorised_mode()
    _set_device_policy(device)
    values: dict[str, float] = {}
    for arm in a2conf.confirm_arm_list():
        payload = _train_one_arm(
            arm, device, mode=mode, schedule="sequential within one job"
        )
        values[arm] = float(payload["soup"]["soup_valid_mae"])
    seconds = float(time.perf_counter() - started)
    _mark(
        "screen",
        "RUN",
        artifact=f"{{real,indep}}_{int(HORIZON)}.json",
        seconds=seconds,
        device=device,
        reason=f"frozen-OMP confirmation at the parent horizon {HORIZON}",
    )
    print(f"[confirm-screen] {json.dumps(values)}", flush=True)
    return values


def arm_stage(arm: str, device: str = "cuda") -> dict[str, Any]:
    """Train exactly one arm: the entry point for a user-authorised parallel schedule.

    The two arms are independent single-process CUDA jobs that share nothing but
    read-only frozen artifacts (seed, init, batch order, frozen dictionary and
    exact-OMP codes are fixed per arm), so running them as two concurrent
    processes on the same physical GPU1 changes scheduling only.  Both arms are
    written to the same result directory with distinct tags.
    """
    started = time.perf_counter()
    mode = _authorised_mode()
    if arm not in a2conf.confirm_arm_list():
        raise RuntimeError(f"arm {arm!r} is not authorised by the frozen preregistration")
    _set_device_policy(device)
    payload = _train_one_arm(
        arm, device, mode=mode, schedule="single-arm job (user-authorised parallel GPU1 schedule)"
    )
    _mark(
        f"arm-{arm.lower()}",
        "RUN",
        artifact=f"{arm.lower()}_{int(HORIZON)}.json",
        seconds=float(time.perf_counter() - started),
        device=device,
        reason="single-arm job (user-authorised parallel schedule on GPU1)",
    )
    value = float(payload["soup"]["soup_valid_mae"])
    print(f"[confirm-arm] {arm} {json.dumps({arm: value})}", flush=True)
    return {arm: value}


# ---------------------------------------------------------------------------
# stage: decision (frozen matrix + diagnostics; no new gates)
# ---------------------------------------------------------------------------


def decision_stage() -> dict[str, Any]:
    identity = _require_verified()
    continuation = _require_continuation()
    arms: dict[str, Any] = {}
    curves: dict[str, dict[int, float]] = {}
    for arm in a2conf.CONFIRM_ARMS:
        tag = _arm_tag(arm)
        path = RESULTS_DIR / f"{tag}.json"
        if not path.exists():
            raise RuntimeError(f"{arm} arm result missing: {path.name}")
        payload = _read_json(path)
        if int(payload.get("horizon", -1)) != int(HORIZON):
            raise RuntimeError(f"{arm}: recorded horizon != {HORIZON}")
        if not bool(payload.get("frozen_dictionary")) or (
            payload.get("dictionary_sha256_f32") != a2run.EXPECTED_DICT_SHA[arm]
        ):
            raise RuntimeError(
                f"{a2conf.VERDICT_IDENTITY_FAILURE}: {arm} arm did not train against "
                "its frozen dictionary"
            )
        curve = a2lite.curve_or_none(_curve_path(tag))
        if curve is not None:
            curves[arm] = curve
        arms[arm] = {
            "tag": tag,
            "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
            "best_valid_mae": float(payload["best_valid_mae"]),
            "best_epoch": int(payload["best_epoch"]),
            "soup_members": payload["soup"]["members"],
            "horizon": int(payload["horizon"]),
            "dictionary_sha256_f32": payload.get("dictionary_sha256_f32"),
            "frozen_dictionary": bool(payload.get("frozen_dictionary")),
            "seconds": payload.get("wall_clock_s"),
            "schedule": payload.get("schedule"),
        }
    curves_finite = bool(
        len(curves) == len(a2conf.CONFIRM_ARMS)
        and all(_complete_curve(_curve_path(_arm_tag(arm)), int(HORIZON)) for arm in a2conf.CONFIRM_ARMS)
    )
    topo = identity["entries"]["topo_320_reference"]
    g_pair_320 = float(arms["INDEP"]["soup_valid_mae"]) - float(arms["REAL"]["soup_valid_mae"])
    delta_vs_topo = a2conf.topo_delta(arms["REAL"]["soup_valid_mae"], topo["soup_valid_mae"])
    decision = a2conf.confirm_decision(
        g_pair_320,
        delta_vs_topo,
        identity_ok=True,
        curves_finite=curves_finite,
    )
    late = None
    if curves_finite:
        late = a2conf.late_window_stats(curves["INDEP"], curves["REAL"])
    g_pair_160 = float(a2conf.LITE_160_SOUP["INDEP"]) - float(a2conf.LITE_160_SOUP["REAL"])
    change = a2conf.gap_change(g_pair_160, g_pair_320)
    paired = {
        **provenance("cpu"),
        "stage": "confirm_paired_analysis",
        "horizon": int(HORIZON),
        "arms": arms,
        "curves_finite": curves_finite,
        "paired_epochs": int(len(set(curves.get("REAL", {})) & set(curves.get("INDEP", {}))))
        if curves_finite else 0,
        "G_pair_320": g_pair_320,
        "material": float(a2conf.MATERIAL),
        "topo_reference": {
            "source": topo["source"],
            "sha256": topo["sha256"],
            "soup_valid_mae": topo["soup_valid_mae"],
            "best_valid_mae": topo["best_valid_mae"],
            "best_epoch": topo["best_epoch"],
            "horizon": topo["horizon"],
            "never_retrained": True,
        },
        "Delta_vs_TOPO": delta_vs_topo,
        "topo_tolerance": float(a2conf.TOPO_TOLERANCE),
        "late_window": late,
        "from_160_to_320": {
            "REAL_soup": {"at_160": float(a2conf.LITE_160_SOUP["REAL"]),
                          "at_320": arms["REAL"]["soup_valid_mae"],
                          "change": arms["REAL"]["soup_valid_mae"] - float(a2conf.LITE_160_SOUP["REAL"])},
            "INDEP_soup": {"at_160": float(a2conf.LITE_160_SOUP["INDEP"]),
                           "at_320": arms["INDEP"]["soup_valid_mae"],
                           "change": arms["INDEP"]["soup_valid_mae"] - float(a2conf.LITE_160_SOUP["INDEP"])},
            "G_pair": change,
        },
        "continuation_mode": continuation.get("mode"),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "paired_analysis.json", paired)
    if decision["verdict"] == a2conf.VERDICT_UNUSABLE:
        _mark("decision", "FAIL", artifact="paired_analysis.json", device="cpu",
              reason=str(decision["reason"]))
        raise RuntimeError(f"{decision['verdict']}: {decision['reason']}")
    payload = {
        **provenance("cpu"),
        "stage": "confirm_decision",
        "run": True,
        "soup_valid_mae": {arm: arms[arm]["soup_valid_mae"] for arm in a2conf.CONFIRM_ARMS},
        "G_pair_320": g_pair_320,
        "Delta_vs_TOPO": delta_vs_topo,
        "topo_soup": float(topo["soup_valid_mae"]),
        "curves_finite": curves_finite,
        "continuation_mode": continuation.get("mode"),
        **{key: decision[key] for key in
           ("case", "verdict", "primary_pass", "competitive", "material", "tolerance", "reason")},
        "late_window": late,
        "from_160_to_320": paired["from_160_to_320"],
        "deferred_stages": a2conf.deferred_stage_record(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _mark("decision", "RUN", artifact="decision.json", device="cpu",
          reason=f"G_pair_320={g_pair_320:.6f} Delta_vs_TOPO={delta_vs_topo:.6f} "
                 f"case={decision['case']} verdict={decision['verdict']}")
    print(
        f"[confirm-decision] G_pair_320={g_pair_320:.6f} "
        f"Delta_vs_TOPO={delta_vs_topo:.6f} verdict={decision['verdict']}",
        flush=True,
    )
    return payload


def _detect_completed_stages() -> None:
    """Record stages whose artifacts already exist.

    The confirmation arms ran as separate processes (user-authorised parallel
    schedule), so ``report`` may be the first place that sees them; an accurate
    stage record must reflect artifacts on disk, not only this process' memory.
    Existing marks are never overwritten.
    """
    def _mark_if_absent(name: str, status: str, **kwargs: Any) -> None:
        if STAGE_STATUS.get(name, {}).get("status") != status:
            _mark(name, status, **kwargs)

    if (RESULTS_DIR / "artifact_identity.json").exists():
        _mark_if_absent("verify", "RUN", artifact="artifact_identity.json", device="cpu",
                        reason="frozen references verified (artifact present)")
    if (RESULTS_DIR / "continuation_mode.json").exists():
        _mark_if_absent("continuation", "RUN", artifact="continuation_mode.json",
                        device="cpu", reason="continuation regime recorded (artifact present)")
    schedules: dict[str, Any] = {}
    arm_seconds = 0.0
    for arm in a2conf.CONFIRM_ARMS:
        path = RESULTS_DIR / f"{_arm_tag(arm)}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        schedule = str(payload.get("schedule") or "unknown")
        schedules[arm] = schedule
        seconds = payload.get("wall_clock_s")
        if isinstance(seconds, (int, float)):
            arm_seconds += float(seconds)
        _mark_if_absent(
            f"arm-{arm.lower()}", "RUN", artifact=path.name, device="cuda",
            seconds=seconds if isinstance(seconds, (int, float)) else None,
            reason=f"320-epoch frozen-OMP arm completed; schedule={schedule}",
        )
    if len(schedules) == len(a2conf.CONFIRM_ARMS):
        _mark_if_absent("screen", "RUN", artifact=f"{{real,indep}}_{int(HORIZON)}.json",
                        device="cuda", seconds=arm_seconds or None,
                        reason="both matched arms complete (see arm_schedules)")
    if (RESULTS_DIR / "decision.json").exists():
        _mark_if_absent("decision", "RUN", artifact="decision.json", device="cpu",
                        reason="frozen decision matrix applied (artifact present)")
    smoke_path = RESULTS_DIR / "smoke" / "smoke.json"
    if smoke_path.exists():
        smoke = _read_json(smoke_path)
        _mark_if_absent(
            "smoke", "RUN (plumbing only)", artifact="smoke/smoke.json",
            device=str(smoke.get("device") or "cuda"),
            seconds=smoke.get("seconds"),
            reason="confirmation trainer smoke (separate process)",
        )


def report_stage() -> dict[str, Any]:
    _detect_completed_stages()
    decision = _read_json(RESULTS_DIR / "decision.json")
    paired = _read_json(RESULTS_DIR / "paired_analysis.json")
    continuation = _require_continuation()
    arms = paired["arms"]
    late = decision.get("late_window")
    change = decision["from_160_to_320"]["G_pair"]
    schedules = {
        arm: _read_json(RESULTS_DIR / f"{_arm_tag(arm)}.json").get("schedule")
        for arm in a2conf.CONFIRM_ARMS
        if (RESULTS_DIR / f"{_arm_tag(arm)}.json").exists()
    }
    parallel_schedule = any(
        str(schedule).startswith("single-arm") for schedule in schedules.values()
    )
    gpu_line = (
        "* GPU: physical GPU1 only; user-authorised parallel schedule "
        f"(arm schedules: {schedules})"
        if parallel_schedule
        else "* GPU: physical GPU1 only (`CUDA_VISIBLE_DEVICES=1`), one CUDA process at a time"
    )
    lines = [
        f"# {ROUND} — DECISION",
        "",
        f"* protocol: `{PROTOCOL_VERSION}`",
        f"* parent round: `{a2conf.PARENT_ROUND}` (frozen protocol "
        f"`{a2conf.PARENT_PROTOCOL_VERSION}`, prereg `{a2conf.PARENT_A2_PREREG_COMMIT}`)",
        f"* predecessor: `{a2conf.PREDECESSOR_ROUND}` / `{a2conf.PREDECESSOR_PROTOCOL_VERSION}`",
        gpu_line,
        f"* seed: {SEED}; horizon: {HORIZON}; arms: `REAL`, `INDEP` (TOPO not retrained)",
        f"* continuation regime: `{continuation['mode']}` (mixing prohibited)",
        "",
        f"## Verdict: `{decision['verdict']}` (case {decision['case']})",
        "",
        f"* reason: {decision['reason']}",
        f"* G_pair_320 = MAE(INDEP-OMP-320) - MAE(REAL-OMP-320) = {decision['G_pair_320']!r}",
        f"* material bar = {decision['material']!r} (parent A2 frozen bar)",
        f"* Delta_vs_TOPO = MAE(REAL-OMP-320) - MAE(TOPO-OMP-320) = {decision['Delta_vs_TOPO']!r}",
        f"* practical tolerance = {decision['tolerance']!r}; TOPO soup = {decision['topo_soup']!r}",
        "",
        "## Frozen-arm results (320 epochs, matched protocol)",
        "",
        "| arm | soup valid MAE | best valid MAE | best epoch | soup members |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in a2conf.CONFIRM_ARMS:
        item = arms[arm]
        lines.append(
            f"| `ATTR-{arm}-OMP-{HORIZON}` | {item['soup_valid_mae']!r} | "
            f"{item['best_valid_mae']!r} | {item['best_epoch']} | {item['soup_members']} |"
        )
    lines += [
        f"| `ATTR-TOPO-OMP-{HORIZON}` (parent, not retrained) | "
        f"{decision['topo_soup']!r} | {paired['topo_reference']['best_valid_mae']!r} | "
        f"{paired['topo_reference']['best_epoch']} | — |",
        "",
        "## Diagnostics (never gates)",
        "",
    ]
    if late:
        lines.append(
            f"* paired late window {late['window']}: mean {late['mean_delta']!r}, "
            f"median {late['median_delta']!r}, positive fraction "
            f"{late['positive_fraction']!r}, first {late['delta_first']!r}, "
            f"last {late['delta_last']!r}"
        )
    else:
        lines.append("* paired late window: unavailable (curves unusable)")
    lines += [
        f"* 160 -> 320: REAL {paired['from_160_to_320']['REAL_soup']['at_160']!r} -> "
        f"{paired['from_160_to_320']['REAL_soup']['at_320']!r}; INDEP "
        f"{paired['from_160_to_320']['INDEP_soup']['at_160']!r} -> "
        f"{paired['from_160_to_320']['INDEP_soup']['at_320']!r}; "
        f"G_pair {change['G_pair_160']!r} -> {change['G_pair_320']!r} "
        f"({change['classification']}, sign_reversed={change['sign_reversed']})",
        "",
        "## Not run (deferred; not cancelled)",
        "",
    ]
    lines += [f"* {name} — DEFERRED_PENDING_USER_AUTHORIZATION" for name in a2conf.DEFERRED_STAGES]
    lines += [
        "",
        "## Stop",
        "",
        "This round stops here in every case.  No IHT qualification, task-coupled E2E,",
        "mechanism intervention, DenseTied specificity, second seed, capacity study or",
        "official-test access follows without explicit user authorisation.",
        "",
        "Official test loaded: `false`.",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_lines = [
        f"# {ROUND} — REPORT",
        "",
        f"* protocol `{PROTOCOL_VERSION}`; commit `{a2run.provenance('cpu')['git_commit']}`",
        f"* preregistration `{a2conf.PREREGISTRATION_NOTE}` "
        f"(commit `{_prereg_commit()}`, rule commit `{_prereg_rule_commit()}`)",
        f"* parent round `{a2conf.PARENT_ROUND}` at `{a2conf.PARENT_A2_COMMIT}` "
        f"(prereg `{a2conf.PARENT_A2_PREREG_COMMIT}`); predecessor `{a2conf.PREDECESSOR_ROUND}`",
        f"* seed {SEED}; horizon {HORIZON}; device policy: physical GPU1 only",
        f"* continuation regime: `{continuation['mode']}`",
        "",
        "## Q1 — exact resume or fresh matched 320?",
        "",
        f"`{continuation['mode']}`.  " + str(continuation["reason"]),
        "",
        "Evidence (recorded in `continuation_mode.json`): the completed 160-epoch lite",
        "segments persist only the best model state and the Top-5 soup *average*; no",
        "optimizer state, no scheduler state, no RNG state, no loader/sampler state, no",
        "individual soup-member states, and the frozen trainer always restarts at epoch 1",
        "(`accepts_resume_argument=false`, `writes_optimizer_state=false`).  Exact",
        "continuation therefore cannot be proven for either arm, so both arms restart as",
        "fresh matched 320-epoch runs (no pseudo-resume, no mixing).",
        "",
        "## Q2 — 320-epoch paired result",
        "",
        f"* REAL soup = {arms['REAL']['soup_valid_mae']!r}",
        f"* INDEP soup = {arms['INDEP']['soup_valid_mae']!r}",
        f"* G_pair_320 = {decision['G_pair_320']!r}  (material bar {decision['material']!r})",
        f"* reaches the bar: `{decision['primary_pass']}`",
        "",
        "## Q3 — practical comparison with the completed TOPO arm",
        "",
        f"* TOPO soup (320, parent, never retrained) = {decision['topo_soup']!r}",
        f"* Delta_vs_TOPO = {decision['Delta_vs_TOPO']!r} (positive = REAL worse)",
        f"* within the +{decision['tolerance']!r} practical tolerance: `{decision['competitive']}`",
        "",
        "## Q4 — final label",
        "",
        f"`{decision['verdict']}`",
        "",
        "## Diagnostics",
        "",
        f"* late window {late['window'] if late else 'n/a'}: "
        f"mean {late['mean_delta'] if late else None!r}, "
        f"median {late['median_delta'] if late else None!r}, "
        f"positive fraction {late['positive_fraction'] if late else None!r}, "
        f"first {late['delta_first'] if late else None!r}, "
        f"last {late['delta_last'] if late else None!r}",
        f"* 160 -> 320 gap: {change['G_pair_160']!r} -> {change['G_pair_320']!r} "
        f"({change['classification']}, sign_reversed={change['sign_reversed']})",
        f"* REAL soup change {paired['from_160_to_320']['REAL_soup']['change']!r}; "
        f"INDEP soup change {paired['from_160_to_320']['INDEP_soup']['change']!r}",
        "",
        "## Deferred",
        "",
    ]
    report_lines += [
        f"* {name} — DEFERRED_PENDING_USER_AUTHORIZATION" for name in a2conf.DEFERRED_STAGES
    ]
    report_lines += [
        "",
        "Official test loaded: `false`.  Stage status: see `stage_status.json`.",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    _not_run_stages = [
        name for name in ("PCA32 dense control", "continuity-v2 diagnostic",
                          "IHT-10/30/100/200 coder qualification",
                          "task-coupled E2E sparse dictionary (E0/E1/E2)",
                          "REAL->INDEP code-pairing-removal mechanism",
                          "node-only / edge-only pairing interventions",
                          "DenseTied specificity control", "seed 1 replication",
                          "TOPO-OMP retraining", "K64 dictionary", "s12 dictionary",
                          "official test split", "any architecture / hyper-parameter sweep")
    ]
    for name in _not_run_stages:
        if name not in STAGE_STATUS:
            _mark(name, "NOT RUN", reason="deferred by the frozen confirmation preregistration")
    _mark("report", "RUN", artifact="REPORT.md, DECISION.md, stage_status.json", device="cpu",
          reason=f"final label {decision['verdict']}")
    payload = {
        **provenance("cpu"),
        "stage": "confirm_report",
        "verdict": decision["verdict"],
        "case": decision["case"],
        "G_pair_320": decision["G_pair_320"],
        "Delta_vs_TOPO": decision["Delta_vs_TOPO"],
        "continuation_mode": continuation["mode"],
        "arm_schedules": schedules,
        "schedule_note": (
            "user-authorised parallel GPU1 schedule: the two 320-epoch arms ran as two "
            "concurrent single-process CUDA jobs with identical frozen protocol "
            "(seed, matched init, batch order, frozen dictionary and exact-OMP codes); "
            "GPU0 was never used and no DDP/multi-GPU was involved"
        ),
        "stages": STAGE_STATUS,
        "cuda_stages_run": sorted(
            name for name, item in STAGE_STATUS.items() if item.get("device") == "cuda"
        ),
        "cpu_stages_run": sorted(
            name for name, item in STAGE_STATUS.items() if item.get("device") == "cpu"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "stage_status.json", payload)
    return payload


# ---------------------------------------------------------------------------
# smoke (plumbing only; cannot influence any gate, decision or number)
# ---------------------------------------------------------------------------


def smoke_stage(device: str = "cuda", *, molecules: int = SMOKE_MOLECULES,
                horizon: int = SMOKE_HORIZON) -> dict[str, Any]:
    """Tiny plumbing check of the confirmation training path (GPU1 only)."""
    started = time.perf_counter()
    smoke_dir = RESULTS_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    _set_device_policy(device)
    original_load_split = p1run.load_split

    def patched_load_split(split: str, subset: int | None = None):
        data = original_load_split(split)
        limit = int(subset) if subset is not None else int(molecules)
        return data[:limit]

    p1run.load_split = patched_load_split  # type: ignore[assignment]
    result: dict[str, Any] = {}
    try:
        arm = "REAL"
        tag = f"smoke_confirm_{arm.lower()}"
        with a1_emits_into(smoke_dir), matched_init_enforced():
            payload = a1run.train_arm(
                arm, stage="omp", device=device, tag=tag,
                horizon=int(horizon), out_dir=smoke_dir, capture_grads=True,
            )
        result = {
            "arm": arm,
            "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
            "curve_finite": a2lite.curve_or_none(smoke_dir / "curves" / f"{tag}_curve.csv") is not None,
            "dictionary_grad_norm": (payload.get("gradient_probe") or {}).get(
                "dictionary_grad_norm"
            ),
        }
    finally:
        p1run.load_split = original_load_split  # type: ignore[assignment]
    gate = {
        "all_finite": bool(result.get("curve_finite")),
        "omp_dictionary_frozen": bool(result.get("dictionary_grad_norm") in (None, 0.0)),
    }
    payload = {
        **provenance(device),
        "stage": "smoke",
        "kind": "plumbing smoke only; cannot influence any gate, decision or number",
        "molecules": int(molecules),
        "horizon": int(horizon),
        "results": result,
        "gate": gate,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(smoke_dir / "smoke.json", payload)
    _mark("smoke", "RUN (plumbing only)", artifact="smoke/smoke.json", device=device,
          seconds=payload["seconds"], reason="confirmation trainer smoke")
    print(f"[confirm-smoke] {json.dumps(gate)}", flush=True)
    if not all(gate.values()):
        raise RuntimeError(f"confirmation smoke gate failed: {gate}")
    return payload


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _not_run(names: Sequence[str], reason: str) -> None:
    for name in names:
        if name not in STAGE_STATUS or STAGE_STATUS[name].get("status") != "RUN":
            _mark(name, "NOT RUN", reason=reason)


def run_all(device: str = "cuda") -> None:
    """verify -> continuation -> screen -> decision -> report, then STOP."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    verify_stage()
    continuation_stage()
    screen_stage(device=device)
    decision_stage()
    payload = report_stage()
    print(
        f"[done] {payload['verdict']} (case {payload['case']}) — stopping; every further "
        f"stage is DEFERRED_PENDING_USER_AUTHORIZATION",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-A2-Confirm runner")
    parser.add_argument(
        "stage",
        choices=("verify", "continuation", "screen", "arm", "decision", "report",
                 "smoke", "all"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", default="REAL", choices=tuple(a2conf.CONFIRM_ARMS))
    parser.add_argument("--smoke-molecules", type=int, default=SMOKE_MOLECULES)
    parser.add_argument("--smoke-horizon", type=int, default=SMOKE_HORIZON)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "verify":
        verify_stage()
    elif args.stage == "continuation":
        continuation_stage()
    elif args.stage == "screen":
        screen_stage(device=args.device)
    elif args.stage == "arm":
        arm_stage(args.arm, device=args.device)
    elif args.stage == "decision":
        decision_stage()
    elif args.stage == "report":
        report_stage()
    elif args.stage == "smoke":
        smoke_stage(device=args.device, molecules=int(args.smoke_molecules),
                    horizon=int(args.smoke_horizon))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
