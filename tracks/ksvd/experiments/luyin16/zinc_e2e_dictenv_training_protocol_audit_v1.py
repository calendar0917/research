"""TPA round runner — shared-prefix exact-fork learning-rate audit.

Round ``e2e_dictenv_training_protocol_audit_v1`` (Workstream Z, ZINC).
Core module:
``tracks/ksvd/experiments/luyin16/e2e_dictenv_training_protocol_audit_v1.py``.
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md``.

Stages
------
``preflight  prefix  fork  control  lowlr  compare  chain``

Exactly one architecture (frozen CSSD-q1, 97727 parameters) and one seed (0).
The shared prefix (epochs 1-280, Adam lr = 1e-3) is trained once; the exact
model/optimizer/RNG/loader state is then cloned and only the learning rate of
epochs 281-320 changes (CONTROL 1e-3, LOW-LR 1e-4).  CPU only; the official
ZINC test split is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_training_protocol_audit_v1 as tpa
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_dictionary_coder_audit_v1 as dcarun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = tpa.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_training_protocol_audit_v1"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_training_protocol_audit_v1_preregistration.md"
CSSD_RESULTS = TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1"
CSSD_SUBSPACE_PATH = CSSD_RESULTS / "common_subspace.json"
CSSD_FINAL_PATH = CSSD_RESULTS / "training/final.json"
CSSD_CURVE_PATH = CSSD_RESULTS / "training/curve.csv"

PREFIX_DIR = RESULTS_DIR / "shared_prefix"
FORK_DIR = RESULTS_DIR / "fork"
CONTROL_DIR = RESULTS_DIR / "control_lr1e3"
LOW_LR_DIR = RESULTS_DIR / "low_lr1e4"

PREFIX_TAG = "TPA-PREFIX"
CONTROL_TAG = "TPA-CONTROL-LR1E3"
LOW_LR_TAG = "TPA-LOW-LR1E4"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

THREADS = 4
CSSD_SEED = tpa.SEED
FORK_PROBE_SIZE = 256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, PREFIX_DIR, FORK_DIR, CONTROL_DIR, LOW_LR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_subspace() -> tuple[cssd.CommonSubspace, str, str]:
    """Load the frozen CSSD-q1 subspace plus file and canonical SHA-256."""
    payload = _read_json(CSSD_SUBSPACE_PATH)
    subspace = cssd.CommonSubspace(
        components=np.asarray(payload["q1"]["components"], dtype=np.float64),
        rms=np.asarray(payload["q1"]["rms"], dtype=np.float64),
        kind=str(payload["q1"]["kind"]),
    )
    canonical = json.dumps(
        {
            "kind": subspace.kind,
            "q": subspace.q,
            "components": subspace.components.tolist(),
            "rms": subspace.rms.tolist(),
        },
        sort_keys=True,
    )
    return subspace, _sha256_file(CSSD_SUBSPACE_PATH), hashlib.sha256(canonical.encode()).hexdigest()


def _light(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop tensor-carrying keys before JSON serialization."""
    return {key: value for key, value in payload.items() if key not in ("keeper", "tail_keeper")}


def _tagged_data(split: str, subset: int | None = None) -> list[Any]:
    data = p1run.load_split(split, subset=subset)
    tpa.tag_graph_ids(data)
    return data


def _data_fingerprints() -> dict[str, Any]:
    return {
        "train_cache": {
            "path": str(p1run._env_cache_path("train").relative_to(REPO_ROOT)),
            "sha256": _sha256_file(p1run._env_cache_path("train")),
        },
        "valid_cache": {
            "path": str(p1run._env_cache_path("valid").relative_to(REPO_ROOT)),
            "sha256": _sha256_file(p1run._env_cache_path("valid")),
        },
    }


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    tpa.cpu_only_guard(torch.device("cpu"))
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    subspace, subspace_file_sha, subspace_canonical_sha = _load_subspace()
    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    rebuilt = cssd.build_common_subspace(train_phi, 1)
    subspace_checks = {
        "components_bit_identical": bool(np.array_equal(subspace.components, rebuilt.components)),
        "rms_bit_identical": bool(np.array_equal(subspace.rms, rebuilt.rms)),
        "shape": [int(value) for value in subspace.components.shape],
    }
    if not all(value for key, value in subspace_checks.items() if key != "shape"):
        raise RuntimeError(f"stored CSSD-q1 subspace is not reproducible: {subspace_checks}")
    dictionary, dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    model = cssd.build_cssd_model(dictionary, int(CSSD_SEED), subspace)
    actual_params = int(sum(parameter.numel() for parameter in model.parameters()))
    if actual_params != tpa.EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"CSSD parameter count mismatch: expected {tpa.EXPECTED_PARAMETERS}, got {actual_params}"
        )
    historical: dict[str, Any] = {"final_json_available": CSSD_FINAL_PATH.exists()}
    if CSSD_FINAL_PATH.exists():
        final = _read_json(CSSD_FINAL_PATH)
        historical.update(
            {
                "soup_valid_mae": float(final["soup"]["soup_valid_mae"]),
                "soup_members": [int(value) for value in final["soup"]["members"]],
                "best_valid_mae": float(final["best_valid_mae"]),
                "best_epoch": int(final["best_epoch"]),
                "actual_params": int(final["actual_params"]),
                "epochs_run": int(final["epochs_run"]),
                "soup_reference_match": bool(
                    abs(float(final["soup"]["soup_valid_mae"]) - tpa.HISTORICAL_SOUP_MAE) <= 1e-9
                ),
                "soup_members_match": bool(
                    tuple(int(value) for value in final["soup"]["members"])
                    == tuple(tpa.HISTORICAL_SOUP_MEMBERS)
                ),
            }
        )
    curve_available = CSSD_CURVE_PATH.exists()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(PREREG_PATH),
        },
        "frozen_design": {
            "prefix_epochs": tpa.PREFIX_EPOCHS,
            "tail_epochs": tpa.TAIL_EPOCHS,
            "total_epochs": tpa.TOTAL_EPOCHS,
            "fork_epoch": tpa.FORK_EPOCH,
            "control_lr": tpa.CONTROL_LR,
            "low_lr": tpa.LOW_LR,
            "seed": tpa.SEED,
            "soup_k": tpa.SOUP_K,
            "batch_size": int(p2run.BATCH_SIZE),
            "weight_decay": float(p2run.WEIGHT_DECAY),
            "grad_clip": float(p2run.GRAD_CLIP),
            "lambda_rec": float(cm.H1_LAMBDA),
            "train_shuffle_offset": int(p2run.TRAIN_SHUFFLE_OFFSET),
            "eval_shuffle_offset": int(p2run.EVAL_SHUFFLE_OFFSET),
            "module_groups": list(tpa.MODULE_GROUPS),
        },
        "architecture": {
            "tag": "CSSD-q1",
            "expected_parameters": tpa.EXPECTED_PARAMETERS,
            "actual_parameters": actual_params,
            "mask": tpa.AUDIT_MASK.as_dict(),
            "config": cm.H1_CONFIG.as_dict(),
        },
        "subspace": {
            "q": subspace.q,
            "kind": subspace.kind,
            "rms": subspace.rms.tolist(),
            "file_sha256": subspace_file_sha,
            "canonical_sha256": subspace_canonical_sha,
            "rebuild_checks": subspace_checks,
        },
        "dictionary": {
            "kind": cm.H1_CONFIG.dict_kind,
            "shape": list(dictionary.shape),
            "sha256": dict_sha,
            "frozen_cssd_sha256": "b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd",
        },
        "data": {
            "phi_train": {
                "shape": list(train_phi.shape),
                "dtype": str(train_phi.dtype),
                "cache": str((p1run.CACHE_DIR / "env_train.pt").relative_to(REPO_ROOT)),
            },
            "phi_valid": {
                "shape": list(valid_phi.shape),
                "dtype": str(valid_phi.dtype),
                "cache": str((p1run.CACHE_DIR / "env_valid.pt").relative_to(REPO_ROOT)),
            },
            "fingerprints": _data_fingerprints(),
        },
        "historical": historical,
        "historical_curve_available": bool(curve_available),
        "expected_budget_epochs": tpa.TOTAL_EPOCHS,
    }
    if dict_sha != payload["dictionary"]["frozen_cssd_sha256"]:
        raise RuntimeError(f"dictionary sha mismatch: {dict_sha}")
    if historical.get("final_json_available") and not (
        historical["soup_reference_match"] and historical["soup_members_match"]
    ):
        raise RuntimeError("historical CSSD reference values are not reproducible from stored results")
    _write_json(
        RESULTS_DIR / "preregistration_snapshot.json",
        payload["preregistration"] | {"frozen_design": payload["frozen_design"]},
    )
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {payload['preregistration']['sha256'][:12]} "
        f"params={actual_params} dict={dict_sha[:12]} subspace_ok "
        f"historical_curve={curve_available}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: shared prefix (epochs 1-280, lr = 1e-3)
# ---------------------------------------------------------------------------


def _model_from_state(
    state: Mapping[str, torch.Tensor], dictionary: np.ndarray, subspace: cssd.CommonSubspace
) -> cssd.CSSDModel:
    model = cssd.build_cssd_model(dictionary, int(CSSD_SEED), subspace)
    model.load_state_dict({key: value for key, value in state.items()})
    return model


def stage_prefix(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    tpa.cpu_only_guard(torch.device("cpu"))
    summary_path = PREFIX_DIR / "prefix_summary.json"
    if summary_path.exists() and not force:
        print("[prefix] cache hit", flush=True)
        return _read_json(summary_path)
    subspace, subspace_file_sha, subspace_canonical_sha = _load_subspace()
    train_data = _tagged_data("train")
    valid_data = _tagged_data("valid")
    started = time.perf_counter()
    payload = tpa.run_epochs(
        tag=PREFIX_TAG,
        start_epoch=1,
        end_epoch=tpa.PREFIX_EPOCHS,
        threads=int(threads),
        out_dir=PREFIX_DIR,
        subspace=subspace,
        train_data=train_data,
        valid_data=valid_data,
        seed=CSSD_SEED,
        lr=tpa.CONTROL_LR,
        log=True,
    )
    round_seconds = float(time.perf_counter() - started)
    checkpoint = torch.load(
        PREFIX_DIR / f"{PREFIX_TAG}_resume_checkpoint.pt", map_location="cpu", weights_only=False
    )
    torch.save(checkpoint["model_state"], PREFIX_DIR / "epoch280_state.pt")
    torch.save(checkpoint["optimizer_state"], PREFIX_DIR / "optimizer_epoch280.pt")
    torch.save(checkpoint, PREFIX_DIR / "resume_checkpoint.pt")
    dictionary, dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    model = _model_from_state(checkpoint["model_state"], dictionary, subspace)
    valid_phi = dcarun.load_phi("valid")
    health = tpa.dictionary_health(
        model=model, phi_valid=valid_phi, reference_D=dictionary, fork_D=None
    )
    gate = tpa.prefix_gate(
        curve=payload["curve"],
        initial_state_sha256=payload["initial_state_sha256"],
        final_state_sha256=payload["final_state_sha256"],
        dictionary_sha256=payload["dictionary_sha256"],
        actual_parameters=payload["actual_params"],
        model_finite=tpa.dictionary_finite(model),
        dictionary_moved=tpa.dictionary_moved(model, dictionary),
    )
    historical_curve = (
        tpa.load_historical_curve(CSSD_CURVE_PATH) if CSSD_CURVE_PATH.exists() else None
    )
    historical = tpa.historical_comparison(payload["curve"], historical_curve)
    _write_csv(PREFIX_DIR / "curve.csv", payload["curve"])
    summary = {
        **_light(payload),
        "git_commit": _git_commit(),
        "round_seconds": round_seconds,
        "subspace": {
            "file_sha256": subspace_file_sha,
            "canonical_sha256": subspace_canonical_sha,
            "q": subspace.q,
            "kind": subspace.kind,
        },
        "contract": {
            "seed": CSSD_SEED,
            "optimizer": f"Adam(lr={tpa.CONTROL_LR}, weight_decay={float(p2run.WEIGHT_DECAY)})",
            "batch_size": int(p2run.BATCH_SIZE),
            "grad_clip": float(p2run.GRAD_CLIP),
            "lambda_rec": float(cm.H1_LAMBDA),
            "loader": (
                f"make_env_loader(train, {int(p2run.BATCH_SIZE)}, shuffle=True, "
                f"seed={CSSD_SEED + int(p2run.TRAIN_SHUFFLE_OFFSET)})"
            ),
            "eval_loader": (
                f"make_env_loader(valid, {int(p2run.BATCH_SIZE)}, shuffle=False, "
                f"seed={CSSD_SEED + int(p2run.EVAL_SHUFFLE_OFFSET)})"
            ),
            "dictionary": {"kind": cm.H1_CONFIG.dict_kind, "sha256": dict_sha},
            "data_fingerprints": _data_fingerprints(),
        },
        "prefix_gate": gate,
        "historical_comparison": historical,
        "dictionary_health_epoch280": health,
        "official_test_loaded": False,
    }
    tpa.official_test_blocker(summary)
    _write_json(summary_path, summary)
    _write_json(PREFIX_DIR / "prefix_gate.json", gate)
    print(
        f"[prefix] epoch280 valid={gate['epoch280_valid_mae']:.6f} "
        f"gate={'PASS' if gate['passed'] else 'FAIL'} "
        f"hist_drift_mean={historical.get('mean_abs_delta')} soup={payload['soup']['soup_valid_mae']:.6f}",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# stage: exact-fork integrity
# ---------------------------------------------------------------------------


def stage_fork(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    tpa.cpu_only_guard(torch.device("cpu"))
    out_path = FORK_DIR / "fork_integrity.json"
    if out_path.exists() and not force:
        print("[fork] cache hit", flush=True)
        return _read_json(out_path)
    checkpoint_path = PREFIX_DIR / "resume_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"prefix checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    subspace, _file_sha, _canonical_sha = _load_subspace()
    dictionary, _dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    train_data = _tagged_data("train")
    valid_data = _tagged_data("valid", subset=FORK_PROBE_SIZE)
    payload = tpa.fork_integrity(
        checkpoint=checkpoint,
        dictionary=dictionary,
        subspace=subspace,
        train_data=train_data,
        valid_data=valid_data,
        threads=int(threads),
        seed=CSSD_SEED,
        probe_size=FORK_PROBE_SIZE,
    )
    payload["git_commit"] = _git_commit()
    _write_json(out_path, payload)
    print(
        f"[fork] verdict={payload['verdict']} pred_diff={payload['prediction_max_abs_diff']} "
        f"order={payload['next_epoch_batch_order_hash']['sha256'][:12]}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: continuations (epochs 281-320)
# ---------------------------------------------------------------------------


def _run_continuation(
    *, tag: str, lr: float, out_dir: Path, threads: int, force: bool
) -> dict[str, Any]:
    out_path = out_dir / "result.json"
    if out_path.exists() and not force:
        print(f"[{tag}] cache hit", flush=True)
        return _read_json(out_path)
    tpa.cpu_only_guard(torch.device("cpu"))
    checkpoint = torch.load(
        PREFIX_DIR / "resume_checkpoint.pt", map_location="cpu", weights_only=False
    )
    subspace, _file_sha, _canonical_sha = _load_subspace()
    train_data = _tagged_data("train")
    valid_data = _tagged_data("valid")
    started = time.perf_counter()
    payload = tpa.run_epochs(
        tag=tag,
        start_epoch=tpa.FORK_EPOCH + 1,
        end_epoch=tpa.TOTAL_EPOCHS,
        threads=int(threads),
        out_dir=out_dir,
        subspace=subspace,
        train_data=train_data,
        valid_data=valid_data,
        seed=CSSD_SEED,
        lr=float(lr),
        model_state=checkpoint["model_state"],
        optimizer_state=checkpoint["optimizer_state"],
        rng_state=checkpoint["rng_state"],
        loader_state=checkpoint["train_loader_state"],
        soup_seed=checkpoint["keeper"],
        best_seed=(int(checkpoint["best"][0]), float(checkpoint["best"][1])),
        track_tail=True,
        log=True,
    )
    payload["git_commit"] = _git_commit()
    payload["round_seconds"] = float(time.perf_counter() - started)
    final_state = torch.load(
        out_dir / f"{tag}_final_state.pt", map_location="cpu", weights_only=False
    )
    soup_state = torch.load(out_dir / f"{tag}_soup_state.pt", map_location="cpu", weights_only=False)
    torch.save(final_state, out_dir / "final_state.pt")
    torch.save(soup_state, out_dir / "soup_state.pt")
    _write_csv(out_dir / "curve.csv", payload["curve"])
    _write_json(
        out_dir / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "tag": tag,
            "lr": float(lr),
            "soup": payload["soup"],
            "tail_soup": payload["tail_soup"],
            "batch_order_hashes": payload["batch_order_hashes"],
            "official_test_loaded": False,
        },
    )
    _write_json(out_path, _light(payload))
    print(
        f"[{tag}] lr={lr:g} soup={payload['soup']['soup_valid_mae']:.6f} "
        f"members={payload['soup']['members']} tail={payload['tail_soup']['soup_valid_mae']:.6f} "
        f"best={payload['best_valid_mae']:.6f}@{payload['best_epoch']}",
        flush=True,
    )
    return payload


def stage_control(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    return _run_continuation(
        tag=CONTROL_TAG, lr=tpa.CONTROL_LR, out_dir=CONTROL_DIR, threads=threads, force=force
    )


def stage_lowlr(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    return _run_continuation(
        tag=LOW_LR_TAG, lr=tpa.LOW_LR, out_dir=LOW_LR_DIR, threads=threads, force=force
    )


# ---------------------------------------------------------------------------
# stage: comparison, analysis tables, report, decision
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.{digits}f}"


def _batch_order_check(control: Mapping[str, Any], low_lr: Mapping[str, Any]) -> dict[str, Any]:
    control_hashes = control["batch_order_hashes"]
    low_hashes = low_lr["batch_order_hashes"]
    epochs = sorted(int(value) for value in control_hashes)
    mismatched = [
        epoch
        for epoch in epochs
        if control_hashes.get(str(epoch)) != low_hashes.get(str(epoch))
    ]
    return {
        "n_epochs": len(epochs),
        "epochs": epochs,
        "mismatched_epochs": mismatched,
        "all_equal": bool(not mismatched and epochs),
    }


def _merged_best_epoch(
    prefix_curve: Sequence[Mapping[str, Any]], continuation_curve: Sequence[Mapping[str, Any]]
) -> tuple[int, float]:
    """Frozen full-run best: min valid MAE over epochs 1-320, ties by earlier epoch."""
    rows = list(prefix_curve) + list(continuation_curve)
    best = min(rows, key=lambda row: (float(row["valid_mae"]), int(row["epoch"])))
    return int(best["epoch"]), float(best["valid_mae"])


def _repair_continuation_best(
    prefix: Mapping[str, Any], payload: dict[str, Any], out_dir: Path, tag: str
) -> dict[str, Any]:
    """Harness fix: recompute the full-run best seed from the stored curves.

    The first continuation pass received an integer-cast prefix best-MAE seed
    (``best=0.000000`` in the log).  Training and every other recorded metric
    were untouched; this function recomputes the two derived fields from the
    stored per-epoch journals and records the fix in the artifact.
    """
    epoch, mae = _merged_best_epoch(prefix["curve"], payload["curve"])
    needs_fix = (
        abs(float(payload.get("best_valid_mae", float("inf"))) - mae) > 1e-15
        or int(payload.get("best_epoch", -1)) != epoch
    )
    if not needs_fix:
        return payload
    payload["best_valid_mae"] = mae
    payload["best_epoch"] = epoch
    payload["harness_fix"] = {
        "reason": (
            "runner-side integer cast on the prefix best-MAE seed (bookkeeping only; "
            "training math, loss, optimizer states and batch order untouched)"
        ),
        "recomputed_from": "prefix curve + continuation curve, min valid MAE, ties by earlier epoch",
        "best_valid_mae": float(mae),
        "best_epoch": int(epoch),
    }
    _write_json(out_dir / "result.json", payload)
    checkpoint_path = out_dir / f"{tag}_resume_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if [int(checkpoint["best"][0]), float(checkpoint["best"][1])] != [epoch, mae]:
            checkpoint["best"] = [int(epoch), float(mae)]
            torch.save(checkpoint, checkpoint_path)
    print(f"[repair] {tag}: best_valid_mae={mae:.6f} @ {epoch} (harness fix)", flush=True)
    return payload


def stage_compare(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    tpa.cpu_only_guard(torch.device("cpu"))
    out_path = RESULTS_DIR / "comparison.json"
    if out_path.exists() and not force:
        print("[compare] cache hit", flush=True)
        return _read_json(out_path)
    prefix = _read_json(PREFIX_DIR / "prefix_summary.json")
    control = _read_json(CONTROL_DIR / "result.json")
    low_lr = _read_json(LOW_LR_DIR / "result.json")
    control = _repair_continuation_best(prefix, control, CONTROL_DIR, CONTROL_TAG)
    low_lr = _repair_continuation_best(prefix, low_lr, LOW_LR_DIR, LOW_LR_TAG)
    fork = _read_json(FORK_DIR / "fork_integrity.json")
    if not fork["passed"]:
        raise RuntimeError("fork integrity failed; comparison is invalid")
    order_check = _batch_order_check(control, low_lr)
    if not order_check["all_equal"]:
        raise RuntimeError(f"batch-order mismatch between arms: {order_check['mismatched_epochs']}")
    control_arm = tpa.arm_summary(prefix_payload=prefix, continuation_payload=control)
    low_arm = tpa.arm_summary(prefix_payload=prefix, continuation_payload=low_lr)
    control_curve = list(prefix["curve"]) + list(control["curve"])
    low_curve = list(prefix["curve"]) + list(low_lr["curve"])
    decision = tpa.training_protocol_decision(
        m_control=float(control_arm["soup_valid_mae"]),
        m_low_lr=float(low_arm["soup_valid_mae"]),
        low_lr_best_epoch=int(low_arm["best_epoch"]),
        low_lr_late_slope=float(low_arm["late_slope_311_320"]),
    )
    subspace, _file_sha, _canonical_sha = _load_subspace()
    dictionary, _dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    valid_phi = dcarun.load_phi("valid")
    prefix_state = torch.load(PREFIX_DIR / "epoch280_state.pt", map_location="cpu", weights_only=False)
    control_state = torch.load(CONTROL_DIR / "final_state.pt", map_location="cpu", weights_only=False)
    low_state = torch.load(LOW_LR_DIR / "final_state.pt", map_location="cpu", weights_only=False)
    prefix_model = _model_from_state(prefix_state, dictionary, subspace)
    control_model = _model_from_state(control_state, dictionary, subspace)
    low_model = _model_from_state(low_state, dictionary, subspace)
    fork_D = np.asarray(prefix_model.D.detach().double().numpy())
    health = {
        "epoch280_prefix": prefix["dictionary_health_epoch280"],
        "control_epoch320": tpa.dictionary_health(
            model=control_model, phi_valid=valid_phi, reference_D=dictionary, fork_D=fork_D
        ),
        "low_lr_epoch320": tpa.dictionary_health(
            model=low_model, phi_valid=valid_phi, reference_D=dictionary, fork_D=fork_D
        ),
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "design": {
            "shared_prefix_epochs": tpa.PREFIX_EPOCHS,
            "fork_epoch": tpa.FORK_EPOCH,
            "tail_epochs": tpa.TAIL_EPOCHS,
            "control_lr": tpa.CONTROL_LR,
            "low_lr": tpa.LOW_LR,
            "seed": CSSD_SEED,
            "noise_floor": tpa.NOISE_FLOOR,
        },
        "prefix": {
            "epoch280_valid_mae": prefix["prefix_gate"]["epoch280_valid_mae"],
            "epoch280_train_mae": prefix["prefix_gate"]["epoch280_train_mae"],
            "prefix_gate": prefix["prefix_gate"],
            "historical_comparison": prefix["historical_comparison"],
            "initial_state_sha256": prefix["initial_state_sha256"],
            "final_state_sha256": prefix["final_state_sha256"],
            "wall_clock_s": prefix["wall_clock_s"],
        },
        "fork": {
            "verdict": fork["verdict"],
            "prediction_max_abs_diff": fork["prediction_max_abs_diff"],
            "train_mode_prediction_max_abs_diff": fork["train_mode_prediction_max_abs_diff"],
            "model_state_identical": fork["model_state_identical"],
            "optimizer_state_identical_before_lr_edit": fork[
                "optimizer_state_identical_before_lr_edit"
            ],
            "exp_avg_identical": fork["optimizer_field_comparison_ignoring_lr"]["exp_avg_identical"],
            "exp_avg_sq_identical": fork["optimizer_field_comparison_ignoring_lr"][
                "exp_avg_sq_identical"
            ],
            "step_identical": fork["optimizer_field_comparison_ignoring_lr"]["step_identical"],
            "only_lr_differs": fork["checks"]["only_lr_differs_after_edit"],
            "next_epoch_batch_order_hash": fork["next_epoch_batch_order_hash"],
        },
        "arms": {
            "control_lr1e3": {
                **control_arm,
                "lr_tail": float(tpa.CONTROL_LR),
                "soup_member_valid_mae": control["soup"]["member_valid_mae"],
                "tail_soup_member_valid_mae": control["tail_soup"]["member_valid_mae"],
            },
            "low_lr1e4": {
                **low_arm,
                "lr_tail": float(tpa.LOW_LR),
                "soup_member_valid_mae": low_lr["soup"]["member_valid_mae"],
                "tail_soup_member_valid_mae": low_lr["tail_soup"]["member_valid_mae"],
            },
        },
        "batch_order": {
            **order_check,
            "control": control["batch_order_hashes"],
            "low_lr": low_lr["batch_order_hashes"],
        },
        "dynamics": tpa.dynamics_rows(control_curve, low_curve),
        "task_vs_valid": {
            "control": tpa.task_vs_valid_delta(prefix, control_arm),
            "low_lr": tpa.task_vs_valid_delta(prefix, low_arm),
        },
        "reconstruction_balance": {
            "control": tpa.reconstruction_balance(prefix, control_arm),
            "low_lr": tpa.reconstruction_balance(prefix, low_arm),
        },
        "dictionary_health": health,
        "decision": decision,
        "budget": {
            "prefix_epochs_run": int(prefix["epochs_run"]),
            "control_epochs_run": int(control["epochs_run"]),
            "low_lr_epochs_run": int(low_lr["epochs_run"]),
            "epoch_equivalents": int(
                prefix["epochs_run"] + control["epochs_run"] + low_lr["epochs_run"]
            ),
            "wall_clock_s": float(
                prefix["wall_clock_s"] + control["wall_clock_s"] + low_lr["wall_clock_s"]
            ),
        },
    }
    tpa.official_test_blocker(payload)
    _write_json(out_path, payload)
    _write_analysis_tables(payload)
    _write_report(payload)
    _write_decision(payload)
    print(
        f"[compare] G_schedule={decision['g_schedule']:+.6f} "
        f"({decision['g_over_noise_floor']:+.1f}x floor) verdict={decision['verdict']}",
        flush=True,
    )
    return payload


def _write_analysis_tables(comparison: Mapping[str, Any]) -> None:
    arms = comparison["arms"]
    decision = comparison["decision"]
    health = comparison["dictionary_health"]
    lines: list[str] = [
        "# e2e_dictenv_training_protocol_audit_v1 — generated analysis tables",
        "",
        "CPU only; official ZINC test never loaded. Generated by the round runner.",
        "",
        "## 0. Design and integrity",
        "",
        f"- shared prefix: epochs 1–{tpa.PREFIX_EPOCHS} at `Adam lr = {tpa.CONTROL_LR:g}` (one trajectory)",
        f"- fork at epoch {tpa.FORK_EPOCH}; only the learning rate of epochs 281–320 changes",
        f"- CONTROL `lr = {tpa.CONTROL_LR:g}`, LOW-LR `lr = {tpa.LOW_LR:g}`",
        f"- fork verdict `{comparison['fork']['verdict']}`, "
        f"prediction max |Δ| `{comparison['fork']['prediction_max_abs_diff']}`, "
        f"train-mode max |Δ| `{comparison['fork']['train_mode_prediction_max_abs_diff']}`",
        f"- fork optimizer: exp_avg identical `{comparison['fork']['exp_avg_identical']}`, "
        f"exp_avg_sq identical `{comparison['fork']['exp_avg_sq_identical']}`, "
        f"step identical `{comparison['fork']['step_identical']}`, "
        f"only lr differs `{comparison['fork']['only_lr_differs']}`",
        f"- batch order: `{comparison['batch_order']['n_epochs']}` tail epochs, "
        f"all hashes equal `{comparison['batch_order']['all_equal']}`",
        "",
        "## 1. Primary comparison (frozen)",
        "",
        "| arm | lr 1–280 | lr 281–320 | best valid | best epoch | Top-5 soup 1–320 | Top-5 members | members > 280 | tail soup 281–320 | last-10 mean | epoch 320 | train MAE 320 | gen. gap 320 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, arm in (("CONTROL", arms["control_lr1e3"]), ("LOW-LR", arms["low_lr1e4"])):
        lines.append(
            f"| {label} | {tpa.CONTROL_LR:g} | {arm['lr_tail']:g} | "
            f"{_fmt(arm['best_valid_mae'], 6)} | {arm['best_epoch']} | "
            f"{_fmt(arm['soup_valid_mae'], 6)} | {arm['soup_members']} | "
            f"{arm['n_soup_members_after_280']} | {_fmt(arm['tail_soup_valid_mae'], 6)} | "
            f"{_fmt(arm['last10_mean_valid_mae'], 6)} | {_fmt(arm['epoch320_valid_mae'], 6)} | "
            f"{_fmt(arm['epoch320_train_mae'], 6)} | {_fmt(arm['generalization_gap_epoch320'], 6)} |"
        )
    lines += [
        "",
        f"`G_schedule = M_control - M_low_lr = {decision['g_schedule']:+.6f}` "
        f"({decision['g_over_noise_floor']:+.1f}x the {tpa.NOISE_FLOOR:g} CPU soup floor).",
        "",
        "## 2. Training-dynamics table",
        "",
        "| epoch | CONTROL valid | LOW-LR valid | difference | CONTROL update norm | LOW-LR update norm | CONTROL grad norm | LOW-LR grad norm |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison["dynamics"]:
        lines.append(
            f"| {row['epoch']} | {_fmt(row['control_valid_mae'], 6)} | "
            f"{_fmt(row['low_lr_valid_mae'], 6)} | {row['difference']:+.6f} | "
            f"{_fmt(row['control_update_norm'], 5)} | {_fmt(row['low_lr_update_norm'], 5)} | "
            f"{_fmt(row['control_grad_norm'], 4)} | {_fmt(row['low_lr_grad_norm'], 4)} |"
        )
    lines += [
        "",
        "## 3. Train vs valid and reconstruction balance (280 → 320)",
        "",
        "| arm | train Δ MAE | valid Δ MAE | reading | task Δ loss | rec Δ term |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (("CONTROL", "control"), ("LOW-LR", "low_lr")):
        tv = comparison["task_vs_valid"][key]
        rb = comparison["reconstruction_balance"][key]
        lines.append(
            f"| {label} | {tv['train_mae_delta_280_to_320']:+.6f} | "
            f"{tv['valid_mae_delta_280_to_320']:+.6f} | {tv['reading']} | "
            f"{rb['task_loss_delta']:+.6f} | {rb['rec_term_delta']:+.3e} |"
        )
    lines += [
        "",
        "## 4. Dictionary health (official valid, minimal frozen set)",
        "",
        "| checkpoint | active atoms | N_eff | top-5 usage | max activation rate | recon (frobenius) | recon (mean row sq) | movement vs init | movement vs fork |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, key in (
        ("epoch 280 (prefix)", "epoch280_prefix"),
        ("CONTROL epoch 320", "control_epoch320"),
        ("LOW-LR epoch 320", "low_lr_epoch320"),
    ):
        entry = health[key]
        lines.append(
            f"| {label} | {entry['active_atoms']} | {_fmt(entry['effective_atoms'], 2)} | "
            f"{_fmt(entry['top5_usage'], 3)} | {_fmt(entry['max_activation_rate'], 3)} | "
            f"{_fmt(entry['reconstruction_frobenius'], 5)} | "
            f"{_fmt(entry['reconstruction_mean_row_squared'], 5)} | "
            f"{_fmt(entry['dictionary_movement_vs_init'], 4)} | "
            f"{_fmt(entry['dictionary_movement_vs_fork'], 4)} |"
        )
    lines += [
        "",
        "## 5. Historical context (not a primary gate)",
        "",
        f"- historical CSSD-q1 seed-0 soup `{tpa.HISTORICAL_SOUP_MAE:.6f}` (members {list(tpa.HISTORICAL_SOUP_MEMBERS)})",
        f"- v2 low-rate warm soup `{tpa.V2_LOW_RATE_WARM_SOUP:.6f}` (trained soup → lr=1e-4 adaptation, not a matched fork)",
        f"- FINAL-CLEAN seed-0 soup `{tpa.FINAL_CLEAN_SOUP:.6f}`",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    historical = comparison["prefix"]["historical_comparison"]
    if historical.get("available"):
        lines += [
            f"| prefix vs historical, mean \\|Δ\\| over 1–280 | {_fmt(historical['mean_abs_delta'], 6)} |",
            f"| median \\|Δ\\| | {_fmt(historical['median_abs_delta'], 6)} |",
            f"| p90 \\|Δ\\| | {_fmt(historical['p90_abs_delta'], 6)} |",
        ]
    for epoch, entry in sorted(historical.get("checkpoints", {}).items(), key=lambda item: int(item[0])):
        lines.append(
            f"| epoch {epoch} valid (new / historical) | {_fmt(entry['shared_prefix'], 6)} / "
            f"{_fmt(entry['historical'], 6)} (Δ {entry['delta']:+.6f}) |"
        )
    lines += ["", "## 6. Decision", "", f"- verdict **{decision['verdict']}**", ""]
    (RESULTS_DIR / "analysis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(comparison: Mapping[str, Any]) -> None:
    arms = comparison["arms"]
    decision = comparison["decision"]
    control = arms["control_lr1e3"]
    low = arms["low_lr1e4"]
    health = comparison["dictionary_health"]
    prefix = comparison["prefix"]
    fork = comparison["fork"]
    g = decision["g_schedule"]
    q3_gap = low["soup_valid_mae"] - tpa.V2_LOW_RATE_WARM_SOUP
    q3_historical_gap = low["soup_valid_mae"] - tpa.HISTORICAL_SOUP_MAE
    lines: list[str] = [
        "# REPORT — `e2e_dictenv_training_protocol_audit_v1`",
        "",
        "CPU only; the official ZINC test split was never loaded. One architecture",
        "(frozen CSSD-q1, 97727 parameters), one seed (0), one shared 280-epoch prefix",
        "and one exact fork that changes only the learning rate of epochs 281–320.",
        "",
        "## 0. Setup",
        "",
        "- pre-registration: `notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md`",
        f"- shared prefix: epochs 1–{tpa.PREFIX_EPOCHS}, `Adam lr = {tpa.CONTROL_LR:g}`, wd "
        f"{float(p2run.WEIGHT_DECAY):g}, batch {int(p2run.BATCH_SIZE)}, clip {float(p2run.GRAD_CLIP):g}",
        f"- CONTROL continuation: epochs 281–320 at `lr = {tpa.CONTROL_LR:g}`",
        f"- LOW-LR continuation: epochs 281–320 at `lr = {tpa.LOW_LR:g}`",
        f"- prefix epoch-280 valid MAE `{prefix['prefix_gate']['epoch280_valid_mae']:.6f}` "
        f"(frozen historical `{prefix['prefix_gate']['historical_epoch280_valid_mae']:.6f}`, "
        f"tolerance +{tpa.PREFIX_MAE_TOLERANCE:g}) → `{prefix['prefix_gate']['verdict']}`",
        f"- fork integrity `{fork['verdict']}`: prediction max |Δ| `{fork['prediction_max_abs_diff']}`, "
        f"optimizer exp_avg/exp_avg_sq/step identical `{fork['exp_avg_identical']}/"
        f"{fork['exp_avg_sq_identical']}/{fork['step_identical']}`, only lr differs "
        f"`{fork['only_lr_differs']}`, batch order equal over "
        f"{comparison['batch_order']['n_epochs']} epochs `{comparison['batch_order']['all_equal']}`",
        "",
        "## 1. Primary result (frozen Top-5 soup over epochs 1–320)",
        "",
        "| arm | best valid | best epoch | Top-5 soup 1–320 | members | tail soup 281–320 | last-10 mean | epoch 320 | train MAE 320 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| CONTROL `lr={tpa.CONTROL_LR:g}` | {control['best_valid_mae']:.6f} | {control['best_epoch']} | "
        f"{control['soup_valid_mae']:.6f} | {control['soup_members']} | {control['tail_soup_valid_mae']:.6f} | "
        f"{control['last10_mean_valid_mae']:.6f} | {control['epoch320_valid_mae']:.6f} | {control['epoch320_train_mae']:.6f} |",
        f"| LOW-LR `lr={tpa.LOW_LR:g}` | {low['best_valid_mae']:.6f} | {low['best_epoch']} | "
        f"{low['soup_valid_mae']:.6f} | {low['soup_members']} | {low['tail_soup_valid_mae']:.6f} | "
        f"{low['last10_mean_valid_mae']:.6f} | {low['epoch320_valid_mae']:.6f} | {low['epoch320_train_mae']:.6f} |",
        "",
        f"```text\nG_schedule = M_control - M_low_lr = {g:+.6f}"
        f"   ({decision['g_over_noise_floor']:+.1f}x the ~{tpa.NOISE_FLOOR:g} CPU soup floor)\n```",
        "",
        "## 2. Dictionary health",
        "",
        "| checkpoint | active | N_eff | top-5 | max rate | recon (mean row sq) |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (
        ("epoch 280", "epoch280_prefix"),
        ("CONTROL 320", "control_epoch320"),
        ("LOW-LR 320", "low_lr_epoch320"),
    ):
        entry = health[key]
        lines.append(
            f"| {label} | {entry['active_atoms']} | {entry['effective_atoms']:.2f} | "
            f"{entry['top5_usage']:.3f} | {entry['max_activation_rate']:.3f} | "
            f"{entry['reconstruction_mean_row_squared']:.5f} |"
        )
    lines += [
        "",
        "## 3. Answers to the frozen questions",
        "",
        "### Q1 — Which late phase is better after the same 280-epoch prefix?",
        "",
    ]
    if g > tpa.NOISE_FLOOR:
        lines.append(
            f"LOW-LR (`lr=1e-4`) is better by `{g:+.6f}` soup MAE "
            f"({decision['g_over_noise_floor']:+.1f}x the measured floor)."
        )
    elif g < -tpa.NOISE_FLOOR:
        lines.append(
            f"CONTROL (`lr=1e-3`) is better by `{-g:+.6f}` soup MAE; the low LR tail is worse."
        )
    else:
        lines.append(
            f"The two tail rates are indistinguishable at the measured floor "
            f"(`G_schedule = {g:+.6f}`)."
        )
    lines += [
        "",
        "### Q2 — Is the low-LR improvement above the ~6e-4 CPU soup floor?",
        "",
        f"`G_schedule / floor = {decision['g_over_noise_floor']:+.2f}`; "
        f"frozen case A threshold is `G_schedule < {tpa.CASE_A_MAX}`.",
        "",
        "### Q3 — Is the historical `0.130028 -> 0.128723` warm improvement reproduced inside the formal schedule?",
        "",
        f"- LOW-LR soup `{low['soup_valid_mae']:.6f}` vs historical CSSD soup "
        f"`{tpa.HISTORICAL_SOUP_MAE:.6f}` (Δ `{q3_historical_gap:+.6f}`)",
        f"- LOW-LR soup vs v2 warm soup `{tpa.V2_LOW_RATE_WARM_SOUP:.6f}` (Δ `{q3_gap:+.6f}`)",
        f"- CONTROL soup vs historical CSSD soup "
        f"`{control['soup_valid_mae'] - tpa.HISTORICAL_SOUP_MAE:+.6f}`",
        "",
        "### Q4 — Is fixed `lr=1e-3` a material contributor to the plateau?",
        "",
        f"Frozen verdict: **{decision['verdict']}** "
        f"(case A `{decision['case_a_no_material']}`, directional `{decision['directional_small']}`, "
        f"case B `{decision['case_b_material']}`, case C `{decision['case_c_major']}`, "
        f"case D `{decision['case_d_new_band']}`).",
        "",
        "### Q5 — Minor polish, material factor, or major factor?",
        "",
        (
            "minor polish (not the plateau's main cause)"
            if not (decision["case_b_material"] or decision["case_c_major"] or decision["case_d_new_band"])
            else (
                "major factor"
                if (decision["case_c_major"] or decision["case_d_new_band"])
                else "material factor"
            )
        ),
        "",
        "### Q6 — Should the canonical training protocol be updated?",
        "",
        f"`canonical_protocol = {decision['canonical_protocol']}` "
        f"(adopt low-LR tail: `{decision['adopt_low_lr_tail']}`).",
        "",
        "### Q7 — Is the 320-epoch horizon still possibly binding?",
        "",
        f"`horizon_may_still_be_binding = {decision['horizon_may_still_be_binding']}` "
        f"(LOW-LR best epoch {low['best_epoch']}, late slope 311–320 "
        f"`{low['late_slope_311_320']:+.3e}` per epoch). Reported only; no longer run was executed.",
        "",
        "## 4. Budget",
        "",
        f"- epochs: prefix {comparison['budget']['prefix_epochs_run']} + control "
        f"{comparison['budget']['control_epochs_run']} + low-LR {comparison['budget']['low_lr_epochs_run']} "
        f"= {comparison['budget']['epoch_equivalents']} epoch-equivalents",
        f"- CPU wall clock {comparison['budget']['wall_clock_s'] / 60.0:.1f} min",
        "- no seed 1, no LR sweep, no extra epochs, no scheduler, no official test",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(comparison: Mapping[str, Any]) -> None:
    decision = comparison["decision"]
    control = comparison["arms"]["control_lr1e3"]
    low = comparison["arms"]["low_lr1e4"]
    lines: list[str] = [
        "# DECISION — `e2e_dictenv_training_protocol_audit_v1`",
        "",
        "Date: 2026-09-29 (CPU only; official ZINC test never loaded)",
        "Preregistration: `notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md`",
        "Evidence: `comparison.json`, `REPORT.md`, `analysis_tables.md`,",
        "`shared_prefix/prefix_summary.json`, `fork/fork_integrity.json`,",
        "`control_lr1e3/result.json`, `low_lr1e4/result.json`.",
        "",
        "## Decision",
        "",
        "```text",
        f"{decision['verdict']}",
        "",
        f"G_schedule = {decision['g_schedule']:+.6f} "
        f"({decision['g_over_noise_floor']:+.1f}x the ~{tpa.NOISE_FLOOR:g} CPU soup floor)",
        f"M_control(lr=1e-3 tail) = {control['soup_valid_mae']:.6f}",
        f"M_low_lr(lr=1e-4 tail)  = {low['soup_valid_mae']:.6f}",
        f"canonical_protocol      = {decision['canonical_protocol']}",
        f"horizon_may_still_be_binding = {decision['horizon_may_still_be_binding']}",
        "```",
        "",
        "## Because",
        "",
        f"- the shared prefix and the fork are exact: epoch-280 valid MAE "
        f"`{comparison['prefix']['epoch280_valid_mae']:.6f}` "
        f"(`{comparison['prefix']['prefix_gate']['verdict']}`), prediction max |Δ| "
        f"`{comparison['fork']['prediction_max_abs_diff']}`, only the learning rate differs "
        f"after the fork, and all {comparison['batch_order']['n_epochs']} tail batch-order "
        f"hashes match;",
        "- the primary comparison is therefore a matched, single-variable LR experiment;",
        f"- the dictionary stays healthy on both arms "
        f"(CONTROL N_eff {comparison['dictionary_health']['control_epoch320']['effective_atoms']:.2f}, "
        f"LOW-LR N_eff {comparison['dictionary_health']['low_lr_epoch320']['effective_atoms']:.2f}, "
        f"epoch-280 N_eff {comparison['dictionary_health']['epoch280_prefix']['effective_atoms']:.2f});",
        "- no official test was read; the decision is based on official valid only.",
        "",
        "## Next",
        "",
    ]
    verdict = decision["verdict"]
    if verdict in (
        "TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK",
        "DIRECTIONAL_SMALL_TRAINING_EFFECT",
        "LOW_LR_TAIL_HARMFUL",
    ):
        lines.append(
            "Close the training-protocol-as-primary-bottleneck hypothesis. The next round is a\n"
            "new architecture study — a Dictionary-Conditioned Semantic Operator\n"
            "(`e_v = sum_k alpha_vk f_k(q_v)`, shared dictionary roles driving a transferable\n"
            "structure→semantics map) — still with **no message passing and no environment\n"
            "update**; no further optimizer/scheduler search."
        )
    elif verdict in ("LOW_LR_TAIL_MATERIALLY_SUPPORTED", "TRAINING_PROTOCOL_MAJOR_FACTOR"):
        lines.append(
            "Adopt `280@1e-3 + 40@1e-4` as the canonical training protocol for all\n"
            "subsequent architecture experiments. No more small deltas against the old\n"
            "`320@1e-3` baseline; the next priority remains the Dictionary-Conditioned\n"
            "Semantic Operator."
        )
    else:
        lines.append(
            "Freeze the new canonical baseline first and re-run minimal mechanism probes\n"
            "(dictionary still load-bearing; node/edge correspondence still used; relation\n"
            "still used) before designing a new architecture."
        )
    lines.append("")
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# chain / CLI
# ---------------------------------------------------------------------------


def chain(threads: int = THREADS, force: bool = False) -> None:
    preflight = stage_preflight()
    prefix = stage_prefix(threads=int(threads), force=bool(force))
    if not prefix["prefix_gate"]["passed"]:
        _write_json(
            RESULTS_DIR / "PREFIX_REGIME_MISMATCH.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "official_test_loaded": False,
                "reason": prefix["prefix_gate"]["verdict"],
                "gate": prefix["prefix_gate"],
                "continuations_run": False,
            },
        )
        print("[chain] prefix gate failed -> no fork, no continuation", flush=True)
        return
    fork = stage_fork(threads=int(threads), force=bool(force))
    if not fork["passed"]:
        _write_json(
            RESULTS_DIR / "FORK_INTEGRITY_FAILURE.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "official_test_loaded": False,
                "reason": fork["verdict"],
                "fork": fork,
                "continuations_run": False,
            },
        )
        print("[chain] fork integrity failed -> no continuation", flush=True)
        return
    stage_control(threads=int(threads), force=bool(force))
    stage_lowlr(threads=int(threads), force=bool(force))
    stage_compare(threads=int(threads), force=bool(force))
    print(f"[chain] done; preflight budget epochs = {preflight['expected_budget_epochs']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["preflight", "prefix", "fork", "control", "lowlr", "compare", "chain"],
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        stage_preflight()
    elif args.stage == "prefix":
        stage_prefix(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "fork":
        stage_fork(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "control":
        stage_control(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "lowlr":
        stage_lowlr(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "compare":
        stage_compare(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "chain":
        chain(threads=int(args.threads), force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
