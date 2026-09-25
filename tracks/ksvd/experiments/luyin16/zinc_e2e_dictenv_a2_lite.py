"""E2E-DictEnv-A2-Lite runner — minimum paired frozen-OMP screen (GPU1 only).

Compute-budget amendment round.  Amendment record:
``tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_a2_lite.py``.

Stage order (each stage is resumable and refuses to run out of order):

``verify screen decision [pca-screen pca-decision] report``

``verify`` re-reads the completed A2 artifact identity *by reference* (no
recomputation of the frozen caches, scaler, dictionaries or OMP codes),
``screen`` drives the frozen A1 trainer unchanged on the two minimum arms
(``REAL`` / ``INDEP``) with the amendment's 160-epoch screen horizon, and
``decision`` applies the frozen lite rule.  The dense rank-32 diagnostic runs
only when the sparse screen does not clear the strong bar, and it stops there.

Deferred by the amendment (never auto-started): TOPO-OMP baseline, IHT coder
qualification, task-coupled E2E sparse dictionary, mechanism interventions,
DenseTied specificity, seed 1, official test and every capacity/sweep variant.

Frozen objects are reused and never re-derived: 433-D coordinates, train-only
scaler, K-SVD ``K=32``/``s=8`` dictionaries, exact-OMP codes, the frozen trainer
and ``sdb_v0.fit_pca_rank`` (through ``e2e_dictenv_a2.fit_dense_rank``).

Official ZINC **test is never loaded**; the shared P1 loader refuses
``split == "test"`` and every artifact records ``official_test_loaded = false``.
All CUDA work runs on physical **GPU1** only (``CUDA_VISIBLE_DEVICES == "1"``,
enforced); GPU0 is foreign-owned and is never touched.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2 as a2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a2_lite as a2lite
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as a1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a2 as a2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = a2lite.PROTOCOL_VERSION
ROUND = a2lite.ROUND
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_a2_lite"
STATE_DIR = RESULTS_DIR / "states"
CURVE_DIR = RESULTS_DIR / "curves"
#: the round this lite continuation hangs off (read-only reference)
A2_RESULTS_DIR = a2run.RESULTS_DIR
A2_IDENTITY_PATH = A2_RESULTS_DIR / "artifact_identity.json"
A1_CACHE_DIR = a1run.CACHE_DIR

SEED = int(a2run.SEED)
HORIZON = int(a2lite.SCREEN_HORIZON)
SMOKE_MOLECULES = 24
SMOKE_HORIZON = 1

SCREEN_STAGES = ("verify", "screen", "decision", "pca-screen", "pca-decision", "report")

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
    """Commit that last touched this amendment (``e2e_dictenv_a2_lite``)."""
    return _git_commit_for(a2lite.AMENDMENT_NOTE)


def _prereg_rule_commit() -> str:
    """Commit that first introduced the amendment (its rules were frozen there)."""
    return _git_commit_for(a2lite.AMENDMENT_NOTE, "--diff-filter=A", "--follow")


def provenance(device: str = "cpu") -> dict[str, Any]:
    """Frozen provenance with the lite round's identity (parent round recorded)."""
    payload = a2run.provenance(device)
    payload.update(
        {
            "round": ROUND,
            "protocol_version": PROTOCOL_VERSION,
            "subtitle": a2lite.SUBTITLE,
            "parent_round": a2lite.PARENT_ROUND,
            "parent_protocol_version": a2lite.PARENT_PROTOCOL_VERSION,
            "amendment": "user_requested_compute_budget_reduction",
            "amendment_note": a2lite.AMENDMENT_NOTE,
            "amendment_commit": _prereg_commit(),
            "amendment_rule_commit": _prereg_rule_commit(),
            "screen_horizon": int(HORIZON),
            "screen_arms": list(a2lite.SCREEN_ARMS),
            "strong_positive_threshold": float(a2lite.STRONG_POSITIVE),
            "pca_threshold": float(a2lite.PCA_MATERIAL),
            "deferred_stages": a2lite.deferred_stage_record(),
        }
    )
    return payload


_write_json = a2run._write_json
_read_json = a2run._read_json
_reemit_a2 = a2run._reemit
a1_emits_into = a2run.a1_emits_into
dictionary_override = a2run.dictionary_override
matched_init_enforced = a2run.matched_init_enforced


def _reemit(payload: Mapping[str, Any], *, source: str, **extra: Any) -> dict[str, Any]:
    """Add lite provenance to a payload produced by the frozen machinery."""
    out = dict(payload)
    out["source_protocol_version"] = out.get("protocol_version")
    out["reused_from"] = str(source)
    out["round"] = ROUND
    out["protocol_version"] = PROTOCOL_VERSION
    out["parent_round"] = a2lite.PARENT_ROUND
    out["subtitle"] = a2lite.SUBTITLE
    out["amendment"] = "user_requested_compute_budget_reduction"
    out["official_test_loaded"] = False
    out.update(extra)
    return out


@contextlib.contextmanager
def a2_emits_into(root: Path) -> Iterator[None]:
    """Point the *A2* core's dense-control artifact writer at ``root``.

    Only ``RESULTS_DIR`` of the A2 runner module is redirected, so the audited
    ``_pca_artifacts`` implementation is reused verbatim while every file it
    writes lands in the lite round's own directory.
    """
    root = Path(root)
    saved = a2run.RESULTS_DIR
    a2run.RESULTS_DIR = root
    try:
        yield
    finally:
        a2run.RESULTS_DIR = saved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curve_path(tag: str) -> Path:
    return CURVE_DIR / f"{tag}_curve.csv"


def _screen_tag(arm: str) -> str:
    return f"lite_omp_{a2lite.SCREEN_LABEL[arm]}"


def _pca_tag(arm: str) -> str:
    return f"lite_pca_{a2lite.SCREEN_LABEL[arm]}"


# ---------------------------------------------------------------------------
# stage: verify (provenance by reference — nothing is recomputed)
# ---------------------------------------------------------------------------


def verify_stage() -> dict[str, Any]:
    """Attach the lite round to the identity-verified frozen artifacts.

    The amendment forbids recomputing the A1 caches, scaler, dictionaries or
    exact-OMP codes.  This stage therefore *reads* the completed A2 identity
    record and re-checks the cheap irreversible digests (module pins, dictionary
    array shas, OMP cache file bytes); it never re-derives a code matrix.
    """
    started = time.perf_counter()
    entries: dict[str, Any] = {}
    if not A2_IDENTITY_PATH.exists():
        raise RuntimeError(f"A2 artifact identity record missing: {A2_IDENTITY_PATH}")
    identity = _read_json(A2_IDENTITY_PATH)
    entries["a2_identity_file"] = {
        "path": str(A2_IDENTITY_PATH),
        "sha256": _sha256_file(A2_IDENTITY_PATH),
        "protocol_version": identity.get("protocol_version"),
        "git_commit": identity.get("git_commit"),
        "preregistration_commit": identity.get("preregistration_commit"),
        "all_passed": bool(identity.get("all_passed")),
        "n_entries": int(len(identity.get("entries", {}))),
        "seconds": identity.get("seconds"),
        "passed": bool(
            identity.get("protocol_version") == a2lite.PARENT_PROTOCOL_VERSION
            and identity.get("all_passed") is True
            and identity.get("git_commit") == a2lite.PARENT_A2_COMMIT
        ),
    }
    # the amendment itself is recorded by digest (its own text defines the rule)
    amendment_path = REPO_ROOT / a2lite.AMENDMENT_NOTE
    entries["amendment_note"] = {
        "path": a2lite.AMENDMENT_NOTE,
        "sha256": _sha256_file(amendment_path) if amendment_path.exists() else None,
        "prereg_commit": _prereg_commit(),
        "rule_commit": _prereg_rule_commit(),
        "passed": bool(amendment_path.exists()),
    }
    # the referenced identity entries that this round depends on must all pass
    required = list(a2lite.REQUIRED_A2_IDENTITY_ENTRIES)
    referenced = identity.get("entries", {})
    referenced_ok = {
        name: bool(referenced.get(name, {}).get("passed"))
        for name in required
    }
    entries["a2_identity_references"] = {
        "required": required,
        "passed": referenced_ok,
        "all_required_passed": bool(all(referenced_ok.values())),
    }
    # the producing code and the frozen preregistration must be byte-identical
    producer = {}
    for relative, expected in a2lite.A2_PRODUCER_SHA256.items():
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
    # live dictionary shas against the tracked pins (and against the A2 record)
    dictionaries = {}
    for arm in a2lite.SCREEN_ARMS:
        _D, sha = a1run.load_arm_dictionary(arm)
        recorded = referenced.get(f"dictionary_{arm}", {}).get("observed_sha256_f32")
        dictionaries[arm] = {
            "expected_sha256_f32": a2run.EXPECTED_DICT_SHA[arm],
            "observed_sha256_f32": sha,
            "a2_identity_sha256_f32": recorded,
            "passed": bool(
                sha == a2run.EXPECTED_DICT_SHA[arm] and (recorded in (None, sha))
            ),
        }
    entries["dictionary_live"] = dictionaries
    # frozen OMP caches: recorded by file digest (bytes preserved), never rebuilt
    codes = {}
    for arm in a2lite.SCREEN_ARMS:
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
    passed = bool(
        entries["a2_identity_file"]["passed"]
        and entries["amendment_note"]["passed"]
        and entries["a2_identity_references"]["all_required_passed"]
        and entries["producer_freeze"]["passed"]
        and all(item["passed"] for item in dictionaries.values())
        and all(item["passed"] for item in codes.values())
    )
    payload = {
        **provenance("cpu"),
        "stage": "lite_verify",
        "kind": "provenance by reference; no cache, scaler, dictionary or OMP recomputation",
        "entries": entries,
        "passed": passed,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "lite_verify.json", payload)
    _mark("verify", "RUN" if passed else "FAIL", artifact="lite_verify.json",
          seconds=payload["seconds"], device="cpu",
          reason="frozen-artifact references verified" if passed else "provenance failure")
    print(f"[lite-verify] passed={passed}", flush=True)
    if not passed:
        raise RuntimeError("lite provenance verification failed; screening is not authorised")
    return payload


def _require_verified() -> dict[str, Any]:
    path = RESULTS_DIR / "lite_verify.json"
    if not path.exists():
        raise RuntimeError("verify stage must run first (lite_verify.json missing)")
    payload = _read_json(path)
    if not payload.get("passed"):
        raise RuntimeError("lite provenance verification did not pass")
    return payload


# ---------------------------------------------------------------------------
# stage: screen (the minimum paired comparison)
# ---------------------------------------------------------------------------


def screen_stage(device: str = "cuda") -> dict[str, Any]:
    """Train the two matched frozen-OMP arms (REAL / INDEP) at the screen horizon."""
    started = time.perf_counter()
    _require_verified()
    _set_device_policy(device)
    values: dict[str, float] = {}
    for arm in a2lite.screen_arm_list():
        tag = _screen_tag(arm)
        with a1_emits_into(RESULTS_DIR), matched_init_enforced():
            payload = a1run.train_arm(
                arm,
                stage="omp",
                device=device,
                tag=tag,
                horizon=int(HORIZON),
                out_dir=RESULTS_DIR,
            )
        payload = _reemit(
            payload,
            source="zinc_e2e_dictenv_a1.train_arm(stage='omp')",
            lite_arm=f"ATTR-{arm}-OMP",
            dictionary_kind="frozen_ksvd_k32_s8_omp",
            screen_horizon=int(HORIZON),
        )
        payload["source_preregistration_commit"] = payload.get("preregistration_commit")
        payload["preregistration_commit"] = _prereg_commit()
        payload["parent_a2_preregistration_commit"] = (
            a2lite.PARENT_A2_PREREG_COMMIT
        )
        payload["parent_a2_commit"] = a2lite.PARENT_A2_COMMIT
        _write_json(RESULTS_DIR / f"lite_omp_{arm.lower()}.json", payload)
        values[arm] = float(payload["soup"]["soup_valid_mae"])
    seconds = float(time.perf_counter() - started)
    _mark("screen", "RUN", artifact="lite_omp_{real,indep}.json", seconds=seconds,
          device=device, reason=f"frozen-OMP screen at horizon {HORIZON} complete")
    print(f"[lite-screen] {json.dumps(values)}", flush=True)
    return values


def decision_stage() -> dict[str, Any]:
    values = {
        arm: float(
            _read_json(RESULTS_DIR / f"lite_omp_{arm.lower()}.json")["soup"]["soup_valid_mae"]
        )
        for arm in a2lite.SCREEN_ARMS
    }
    curve_indep = a2lite.curve_or_none(_curve_path(_screen_tag("INDEP")))
    curve_real = a2lite.curve_or_none(_curve_path(_screen_tag("REAL")))
    curves_finite = curve_indep is not None and curve_real is not None
    direction = None
    paired = 0
    if curves_finite:
        assert curve_indep is not None and curve_real is not None
        paired = a2lite.paired_epochs(curve_indep, curve_real)
        try:
            direction = a2lite.late_direction(curve_indep, curve_real)
        except a2lite.CurveError:
            curves_finite = False
    decision = a2lite.screen_decision(
        values, direction=direction, curves_finite=curves_finite, horizon=int(HORIZON)
    )
    payload = {
        **provenance(_device_of_stage("screen")),
        "stage": "lite_screen_decision",
        "run": True,
        "soup_valid_mae": values,
        "G_pair_screen": float(decision["G_pair_screen"]),
        "direction": direction,
        "paired_epochs": int(paired),
        "screen_horizon": int(HORIZON),
        "curves_finite": bool(curves_finite),
        "strong_positive": bool(decision["strong_positive"]),
        "pca_authorised": bool(decision["pca_authorised"]),
        "verdict": decision["verdict"],
        "reason": decision["reason"],
        "threshold": float(a2lite.STRONG_POSITIVE),
        "pca_threshold": float(a2lite.PCA_MATERIAL),
        "deferred_stages": a2lite.deferred_stage_record(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "lite_screen_decision.json", payload)
    _mark("decision", "RUN", artifact="lite_screen_decision.json", device="cpu",
          reason=f"G_pair_screen={payload['G_pair_screen']:.6f} verdict={payload['verdict']}")
    print(
        f"[lite-decision] G_pair_screen={payload['G_pair_screen']:.6f} "
        f"direction_stable={payload['direction'] and payload['direction']['stable']} "
        f"verdict={payload['verdict']}",
        flush=True,
    )
    return payload


def _device_of_stage(stage: str) -> str:
    entry = STAGE_STATUS.get(stage, {})
    return "cuda" if entry.get("status") == "RUN" else "cpu"


# ---------------------------------------------------------------------------
# stage: cheap dense rank-32 diagnostic (only when the screen is not strong)
# ---------------------------------------------------------------------------


def pca_screen_stage(device: str = "cuda") -> dict[str, Any]:
    """Train the two matched train-only rank-32 dense arms (ambiguous screen only)."""
    started = time.perf_counter()
    _require_verified()
    screen = _read_json(RESULTS_DIR / "lite_screen_decision.json")
    if not screen.get("pca_authorised"):
        raise RuntimeError("the dense diagnostic is not authorised by the frozen screen rule")
    _set_device_policy(device)
    values: dict[str, float] = {}
    for arm in a2lite.screen_arm_list():
        with a2_emits_into(RESULTS_DIR):
            projection, _mean, meta = a2run._pca_artifacts(arm)
        meta = _reemit(
            meta,
            source="e2e_dictenv_a2.runner._pca_artifacts (frozen train-only rank-32 control)",
            lite_arm=f"ATTR-{arm}-PCA32",
            screen_horizon=int(HORIZON),
        )
        _write_json(RESULTS_DIR / f"pca_meta_{arm.lower()}.json", meta)
        blob = torch.load(
            RESULTS_DIR / f"pca_codes_{arm.lower()}.pt", map_location="cpu", weights_only=True
        )
        codes_train = blob["train"].numpy()
        codes_valid = blob["valid"].numpy()
        X_train = np.asarray(a1run.arm_coordinate(arm, "train"), dtype=np.float32)
        X_valid = np.asarray(a1run.arm_coordinate(arm, "valid"), dtype=np.float32)
        if codes_train.shape[0] != X_train.shape[0] or codes_valid.shape[0] != X_valid.shape[0]:
            raise RuntimeError("dense code rows do not match the coordinate rows")
        tag = _pca_tag(arm)
        with a1_emits_into(RESULTS_DIR), dictionary_override(arm, projection), matched_init_enforced():
            payload = a1run.train_arm(
                arm,
                stage="omp",
                device=device,
                tag=tag,
                horizon=int(HORIZON),
                out_dir=RESULTS_DIR,
                codes_train=np.asarray(codes_train, dtype=np.float32),
                codes_valid=np.asarray(codes_valid, dtype=np.float32),
                x_train=X_train,
                x_valid=X_valid,
            )
        payload = _reemit(
            payload,
            source="zinc_e2e_dictenv_a1.train_arm(stage='omp') with a frozen PCA32 code",
            lite_arm=f"ATTR-{arm}-PCA32",
            dictionary_kind="frozen_pca32_components",
            screen_horizon=int(HORIZON),
            pca_meta={key: meta[key] for key in ("rank", "fit_split", "n_fit_rows", "explained_variance_ratio") if key in meta},
        )
        payload["source_preregistration_commit"] = payload.get("preregistration_commit")
        payload["preregistration_commit"] = _prereg_commit()
        payload.pop("dictionary_sha256_f32", None)
        payload["pca_components_sha256_f32"] = a2.a1_hash(projection)
        _write_json(RESULTS_DIR / f"lite_pca_{arm.lower()}.json", payload)
        values[arm] = float(payload["soup"]["soup_valid_mae"])
        del codes_train, codes_valid, X_train, X_valid, blob
    seconds = float(time.perf_counter() - started)
    _mark("pca-screen", "RUN", artifact="lite_pca_{real,indep}.json", seconds=seconds,
          device=device, reason=f"train-only PCA32 dense control at horizon {HORIZON}")
    print(f"[lite-pca-screen] {json.dumps(values)}", flush=True)
    return values


def pca_decision_stage() -> dict[str, Any]:
    values = {
        arm: float(
            _read_json(RESULTS_DIR / f"lite_pca_{arm.lower()}.json")["soup"]["soup_valid_mae"]
        )
        for arm in a2lite.SCREEN_ARMS
    }
    curves_finite = all(
        a2lite.curve_or_none(_curve_path(_pca_tag(arm))) is not None
        for arm in a2lite.SCREEN_ARMS
    )
    g_pair = float(values["INDEP"] - values["REAL"])
    decision = a2lite.pca_label(g_pair, curves_finite=curves_finite)
    payload = {
        **provenance(_device_of_stage("pca-screen")),
        "stage": "lite_pca32_decision",
        "run": True,
        "soup_valid_mae": values,
        "G_pair_PCA": g_pair,
        "rank": int(a2lite.PCA_RANK),
        "screen_horizon": int(HORIZON),
        "curves_finite": bool(curves_finite),
        "primary_pass": bool(decision["primary_pass"]),
        "verdict": decision["verdict"],
        "reason": decision["reason"],
        "threshold": float(a2lite.PCA_MATERIAL),
        "deferred_stages": a2lite.deferred_stage_record(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "lite_pca_decision.json", payload)
    _mark("pca-decision", "RUN", artifact="lite_pca_decision.json", device="cpu",
          reason=f"G_pair_PCA={g_pair:.6f} verdict={payload['verdict']}")
    print(f"[lite-pca-decision] G_pair_PCA={g_pair:.6f} verdict={payload['verdict']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------


def _final_label() -> tuple[str, dict[str, Any]]:
    screen = _read_json(RESULTS_DIR / "lite_screen_decision.json")
    if screen.get("strong_positive") or screen.get("verdict") == a2lite.VERDICT_UNRESOLVED:
        return str(screen["verdict"]), {"screen": screen}
    pca_path = RESULTS_DIR / "lite_pca_decision.json"
    pca = _read_json(pca_path) if pca_path.exists() else None
    if pca is None:
        return a2lite.STATE_PCA_AUTHORISED, {"screen": screen}
    return str(pca["verdict"]), {"screen": screen, "pca": pca}


def report_stage() -> dict[str, Any]:
    started = time.perf_counter()
    label, evidence = _final_label()
    for stage in ("pca-screen", "pca-decision"):
        if stage not in STAGE_STATUS:
            _mark(stage, "NOT RUN",
                  reason="sparse screen cleared the strong bar or was unusable")
    payload = {
        **provenance("cpu"),
        "stage": "lite_report",
        "run": True,
        "final_label": label,
        "evidence": {
            "screen": {
                key: evidence["screen"].get(key)
                for key in ("soup_valid_mae", "G_pair_screen", "direction", "curves_finite",
                            "strong_positive", "pca_authorised", "verdict", "reason",
                            "screen_horizon", "threshold")
            },
            "pca": None
            if evidence.get("pca") is None
            else {
                key: evidence["pca"].get(key)
                for key in ("soup_valid_mae", "G_pair_PCA", "rank", "curves_finite",
                            "primary_pass", "verdict", "reason", "threshold")
            },
        },
        "stage_status": STAGE_STATUS,
        "deferred_stages": a2lite.deferred_stage_record(),
        "not_claims": [
            "this screen is not the frozen A2 320-epoch Stage 1 and cannot be reported as it",
            f"the screen horizon is {HORIZON} epochs; it is a continue/stop screen only",
            "no IHT coder, no task-coupled E2E dictionary, no mechanism and no specificity "
            "control were run; every one is DEFERRED_PENDING_USER_AUTHORIZATION",
        ],
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "lite_report.json", payload)
    _write_json(RESULTS_DIR / "stage_status.json", {"stages": STAGE_STATUS,
                                                    "deferred": a2lite.deferred_stage_record(),
                                                    "official_test_loaded": False})
    lines = [
        f"# {ROUND} — DECISION",
        "",
        f"* protocol: `{PROTOCOL_VERSION}` (compute-budget amendment)",
        f"* parent round: `{a2lite.PARENT_ROUND}` — `COMPUTE-BUDGET-TRUNCATED`",
        f"* git commit: `{payload['git_commit']}`",
        "* GPU: physical GPU1 only (`CUDA_VISIBLE_DEVICES=1`)",
        f"* screen: horizon {HORIZON}, arms `{', '.join(a2lite.SCREEN_ARMS)}`, seed {SEED}",
        "",
        f"## Verdict: `{label}`",
        "",
        f"* reason: {evidence['screen'].get('reason')}",
        f"* G_pair_screen = MAE(INDEP-OMP) - MAE(REAL-OMP) = "
        f"{evidence['screen'].get('G_pair_screen')}",
    ]
    if evidence.get("pca") is not None:
        lines += [
            f"* G_pair_PCA = MAE(INDEP-PCA32) - MAE(REAL-PCA32) = "
            f"{evidence['pca'].get('G_pair_PCA')} (rank {evidence['pca'].get('rank')})",
            f"* dense reason: {evidence['pca'].get('reason')}",
        ]
    lines += [
        "",
        "## Not run (deferred by the amendment; not cancelled)",
        "",
        *[f"* {name} — DEFERRED_PENDING_USER_AUTHORIZATION" for name in a2lite.DEFERRED_STAGES],
        "",
        "No official test data was loaded.",
        "",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = [
        f"# {ROUND} — REPORT",
        "",
        "* amendment: `tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md`",
        f"* amendment commit: `{_prereg_commit()}`",
        f"* git commit: `{payload['git_commit']}`; branch `{payload.get('branch')}`",
        f"* frozen artifacts reused by reference: `{A2_IDENTITY_PATH.name}` "
        f"(all_passed={_read_json(A2_IDENTITY_PATH).get('all_passed')})",
        "",
        f"## Screen (frozen exact-OMP, horizon {HORIZON}, seed {SEED}, matched init/batch order)",
        "",
        f"* MAE(REAL-OMP) = {evidence['screen']['soup_valid_mae'].get('REAL')}",
        f"* MAE(INDEP-OMP) = {evidence['screen']['soup_valid_mae'].get('INDEP')}",
        f"* G_pair_screen = {evidence['screen']['G_pair_screen']} "
        f"(strong bar {a2lite.STRONG_POSITIVE})",
        f"* late direction: {json.dumps(evidence['screen'].get('direction'))}",
        f"* curves finite: {evidence['screen']['curves_finite']}",
    ]
    if evidence.get("pca") is not None:
        report += [
            "",
            f"## Dense control (train-only PCA32, horizon {HORIZON})",
            "",
            f"* MAE(REAL-PCA32) = {evidence['pca']['soup_valid_mae'].get('REAL')}",
            f"* MAE(INDEP-PCA32) = {evidence['pca']['soup_valid_mae'].get('INDEP')}",
            f"* G_pair_PCA = {evidence['pca']['G_pair_PCA']} (bar {a2lite.PCA_MATERIAL})",
        ]
    report += [
        "",
        f"**Verdict: `{label}`**",
        "",
        "Deferred (not cancelled): "
        + "; ".join(f"{name} = DEFERRED_PENDING_USER_AUTHORIZATION"
                    for name in a2lite.DEFERRED_STAGES)
        + ". Official test never loaded.",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _mark("report", "RUN", artifact="DECISION.md, REPORT.md", device="cpu",
          seconds=float(time.perf_counter() - started), reason=f"final_label={label}")
    print(f"[lite-report] final_label={label}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# plumbing smoke (local CPU validation / GPU1 plumbing)
# ---------------------------------------------------------------------------


def smoke_stage(device: str = "cuda", *, molecules: int = SMOKE_MOLECULES,
                horizon: int = SMOKE_HORIZON) -> dict[str, Any]:
    """Tiny end-to-end plumbing check of the lite training path."""
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
    results: dict[str, Any] = {}
    try:
        for arm in a2lite.screen_arm_list():
            tag = f"smoke_{_screen_tag(arm)}"
            with a1_emits_into(smoke_dir), matched_init_enforced():
                payload = a1run.train_arm(
                    arm, stage="omp", device=device, tag=tag,
                    horizon=int(horizon), out_dir=smoke_dir, capture_grads=True,
                )
            results[arm] = {
                "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
                "curve_finite": a2lite.curve_or_none(_curve_path_in(smoke_dir, tag)) is not None,
                "dictionary_grad_norm": (payload.get("gradient_probe") or {}).get(
                    "dictionary_grad_norm"
                ),
            }
    finally:
        p1run.load_split = original_load_split  # type: ignore[assignment]
    gate = {
        "all_finite": bool(all(item["curve_finite"] for item in results.values())),
        "omp_dictionary_frozen": bool(
            all(item["dictionary_grad_norm"] in (None, 0.0) for item in results.values())
        ),
    }
    payload = {
        **provenance(device),
        "stage": "smoke",
        "kind": "plumbing smoke only; cannot influence any gate, decision or number",
        "molecules": int(molecules),
        "horizon": int(horizon),
        "results": results,
        "gate": gate,
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(smoke_dir / "smoke.json", payload)
    _mark("smoke", "RUN (plumbing only)", artifact="smoke/smoke.json", device=device,
          seconds=payload["seconds"], reason="lite trainer smoke")
    print(f"[lite-smoke] {json.dumps(gate)}", flush=True)
    return payload


def _curve_path_in(root: Path, tag: str) -> Path:
    return Path(root) / "curves" / f"{tag}_curve.csv"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _not_run(names: Sequence[str], reason: str) -> None:
    for name in names:
        if name not in STAGE_STATUS or STAGE_STATUS[name].get("status") != "RUN":
            _mark(name, "NOT RUN", reason=reason)


def run_all(device: str = "cuda") -> None:
    """verify -> screen -> decision -> [pca-screen pca-decision] -> report, then STOP."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    verify_stage()
    screen_stage(device=device)
    screen = decision_stage()
    if screen["verdict"] == a2lite.VERDICT_UNRESOLVED:
        _not_run(("pca-screen", "pca-decision"), "screen unusable; diagnosis not authorised")
    elif screen["pca_authorised"]:
        pca_screen_stage(device=device)
        pca_decision_stage()
    else:
        _not_run(("pca-screen", "pca-decision"),
                 "screen cleared the strong bar; dense diagnosis not authorised")
    payload = report_stage()
    print(f"[done] {payload['final_label']} — stopping; all further stages are "
          f"DEFERRED_PENDING_USER_AUTHORIZATION", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-A2-Lite runner")
    parser.add_argument(
        "stage",
        choices=(
            "verify", "screen", "decision", "pca-screen", "pca-decision",
            "report", "smoke", "all",
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-molecules", type=int, default=SMOKE_MOLECULES)
    parser.add_argument("--smoke-horizon", type=int, default=SMOKE_HORIZON)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "verify":
        verify_stage()
    elif args.stage == "screen":
        screen_stage(device=args.device)
    elif args.stage == "decision":
        decision_stage()
    elif args.stage == "pca-screen":
        pca_screen_stage(device=args.device)
    elif args.stage == "pca-decision":
        pca_decision_stage()
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
