"""Optimization-regime audit for the frozen compact-v4 T=2 recurrent pair--centre.

Single question: is the current Q16 reference (82,115 params, seed0 valid
0.138376, seed1 valid 0.133440) limited mainly by the *training horizon* /
*early-stopping patience* or by the *learning-rate schedule*, rather than by
capacity?  And was the earlier Q24 negative result merely an under-optimised
wider model?

Phases
------
* Phase A -- zero-cost 2-seed ensemble diagnostic on the existing official-valid
  predictions (``recurrent_seed0/1.json``), no training.
* Phase B -- Q16 long-horizon control (identical architecture / optimizer, only
  ``max_epochs 240 -> 500`` and ``patience 40 -> 80``), seed0, validation only.
* Phase C -- Q16 plateau schedule (same, plus
  ``ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-5)`` driven by the
  existing validation MAE), seed0, validation only.
* Phase D -- decision gate on the pre-registered ``+0.002`` / ``+0.003`` bands.
* Phase E -- Q24 retry with the *identical* best new protocol, seed0 only,
  executed **only if** Phase B or C improves on the current Q16 seed0 by at
  least ``+0.002``.

Nothing else changes: same tokenizer / patch construction / relation descriptor
/ readout / small head / weight decay / batch size / activation / architecture /
T / output split.  Official test is NEVER loaded by this module.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_optimization_audit <stage>

Stages: ``phase_a fairness long plateau diagnose_q16 diagnose_q24 decision
q24_retry report all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as capacity,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_optimization_audit"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_optimization_audit_v1"

# Frozen T=2 recurrent reference (never retrained / overwritten).
REFERENCE_DIR = rec.RESULTS_DIR
Q16_SEED0_VALID = 0.13837560486892472
Q16_SEED1_VALID = 0.13343997858563672
Q16_SEED0_BEST_EPOCH = 199
Q16_EXPECTED_TOTAL = 82115
Q24_EXPECTED_TOTAL = 91211

# Pre-registered signal bands (from the task brief).
CONTINUE_GATE = 0.002  # worth continuing
STRONG_GATE = 0.003    # strong seed / optimization variance signal
PLATEAU_OVER_LONG = 0.001  # "plateau clearly better than long" margin

TAGS = {"Q16_long": "q16_long", "Q16_plateau": "q16_plateau", "Q24_retry": "q24_retry"}

# Only the optimization protocol changes -- architecture / optimizer / data are
# inherited verbatim from the canonical small-head loop.
LONG_PROTOCOL: dict[str, Any] = {
    "max_epochs": 500,
    "patience": 80,
    "scheduler": "none",
}
PLATEAU_PROTOCOL: dict[str, Any] = {
    "max_epochs": 500,
    "patience": 80,
    "scheduler": "reduce_on_plateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 20,
    "scheduler_min_lr": 1.0e-5,
}


# ---------------------------------------------------------------------------
# io / helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _n_params(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _rss_peak_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        for line in handle:
            parts = line.strip().split(",")
            if not parts or not parts[0]:
                continue
            rows.append({key: float(value) for key, value in zip(header, parts)})
    return rows


def _first_batch(valid_data: Sequence[Any], device: torch.device) -> Any:
    return next(iter(zpp._make_loader(list(valid_data)[:128], 128, False, 0))).to(device)


# ---------------------------------------------------------------------------
# Phase A -- zero-cost 2-seed ensemble diagnostic
# ---------------------------------------------------------------------------


def phase_a() -> dict[str, Any]:
    seed0 = _read_json(REFERENCE_DIR / "recurrent_seed0.json")
    seed1 = _read_json(REFERENCE_DIR / "recurrent_seed1.json")
    p0 = np.asarray(seed0["valid_predictions"], dtype=np.float64)
    p1 = np.asarray(seed1["valid_predictions"], dtype=np.float64)
    t0 = np.asarray(seed0["valid_targets"], dtype=np.float64)
    t1 = np.asarray(seed1["valid_targets"], dtype=np.float64)
    targets_identical = bool(np.array_equal(t0, t1)) and bool(np.array_equal(p0.shape, p1.shape))
    mae0 = float(np.mean(np.abs(p0 - t0)))
    mae1 = float(np.mean(np.abs(p1 - t1)))
    mean_single = 0.5 * (mae0 + mae1)
    ensemble = 0.5 * (p0 + p1)
    mae_ensemble = float(np.mean(np.abs(ensemble - t0)))
    improvement = float(mean_single - mae_ensemble)
    disagreement = float(np.mean(np.abs(p0 - p1)))
    r0 = p0 - t0
    r1 = p1 - t1
    residual_corr = float(np.corrcoef(r0, r1)[0, 1])
    pred_corr = float(np.corrcoef(p0, p1)[0, 1])
    # Ensemble error decomposition: E|r_ens| vs mean E|r_i| (descriptive).
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "description": "2-seed equal-weight validation ensemble (no training).",
        "seed0_valid_mae": mae0,
        "seed1_valid_mae": mae1,
        "mean_single_model_mae": mean_single,
        "two_seed_ensemble_mae": mae_ensemble,
        "ensemble_improvement": improvement,
        "ensemble_improvement_ge_0003": bool(improvement >= 0.003),
        "mean_abs_seed_disagreement": disagreement,
        "prediction_correlation": pred_corr,
        "residual_correlation": residual_corr,
        "n_valid": int(p0.shape[0]),
        "targets_identical": targets_identical,
        "seed0_best_epoch": int(seed0["best_epoch"]),
        "seed1_best_epoch": int(seed1["best_epoch"]),
        "seed0_source": str(REFERENCE_DIR / "recurrent_seed0.json"),
        "seed1_source": str(REFERENCE_DIR / "recurrent_seed1.json"),
        "read": (
            "Zero-cost diagnostic only; no ensemble-weight search.  A large "
            "ensemble gain relative to the mean single model is evidence of "
            "seed / optimization variance rather than a capacity bottleneck."
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "phase_a_seed_ensemble.json", payload)
    return payload


# ---------------------------------------------------------------------------
# fairness / protocol lock
# ---------------------------------------------------------------------------


def fairness() -> dict[str, Any]:
    train_data, valid_data, _ = rec.load_encoded()
    seed0 = _read_json(REFERENCE_DIR / "recurrent_seed0.json")
    seed1 = _read_json(REFERENCE_DIR / "recurrent_seed1.json")
    t0 = np.asarray(seed0["valid_targets"], dtype=np.float64)
    t1 = np.asarray(seed1["valid_targets"], dtype=np.float64)

    ref_q16 = rec.build_recurrent(0)
    cap_q16 = capacity.build_q16(0)
    q24 = capacity.build_q24(0)
    q16_state = {k: v.detach().clone() for k, v in ref_q16.state_dict().items()}
    cap_state = {k: v.detach().clone() for k, v in cap_q16.state_dict().items()}
    digest = hashlib.sha256()
    for key in sorted(q16_state):
        digest.update(key.encode())
        digest.update(q16_state[key].detach().cpu().contiguous().numpy().tobytes())
    cap_digest = hashlib.sha256()
    for key in sorted(cap_state):
        cap_digest.update(key.encode())
        cap_digest.update(cap_state[key].detach().cpu().contiguous().numpy().tobytes())

    checks = {
        "valid_targets_identical_seed0_seed1": bool(np.array_equal(t0, t1)),
        "train_size_10000": int(len(train_data)) == 10000,
        "valid_size_1000": int(len(valid_data)) == 1000,
        "q16_reference_params_82115": _n_params(ref_q16) == Q16_EXPECTED_TOTAL,
        "q16_capacity_params_82115": _n_params(cap_q16) == Q16_EXPECTED_TOTAL,
        "q16_capacity_builder_bit_identical": bool(
            digest.hexdigest() == cap_digest.hexdigest()
        ),
        "q24_params_91211": _n_params(q24) == Q24_EXPECTED_TOTAL,
        "same_optimizer_adam_lr_wd_clip": True,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "reference_protocol": dict(shead.OPTIMIZED_PROTOCOL),
        "long_protocol_override": dict(LONG_PROTOCOL),
        "plateau_protocol_override": dict(PLATEAU_PROTOCOL),
        "only_changed_keys": ["max_epochs", "patience", "scheduler", "scheduler_*"],
        "split_source": "rec.load_encoded -> shead.build_encoded_records (official train/valid only)",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "fairness.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"fairness checks failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _train(
    build_fn: Callable[[int], torch.nn.Module],
    tag: str,
    *,
    seed: int,
    protocol_override: Mapping[str, Any],
    expected_total: int,
    real_batch_identity: bool,
) -> dict[str, Any]:
    train_data, valid_data, _ = rec.load_encoded()
    original_dirs = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    original_protocol = shead.OPTIMIZED_PROTOCOL
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    protocol = dict(original_protocol)
    protocol.update(protocol_override)
    shead.OPTIMIZED_PROTOCOL = protocol
    rss_before = _rss_peak_kb()
    started = time.perf_counter()
    try:
        summary = shead.train_model(
            build_fn=build_fn,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=str(tag),
            save_state=True,
            real_batch_identity=bool(real_batch_identity),
            expected_total=int(expected_total),
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original_dirs
        shead.OPTIMIZED_PROTOCOL = original_protocol
    summary = dict(summary)
    summary["outer_wall_clock_s"] = float(time.perf_counter() - started)
    summary["peak_rss_kb"] = _rss_peak_kb()
    summary["peak_rss_delta_kb"] = _rss_peak_kb() - int(rss_before)
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    _write_json(RESULTS_DIR / f"{tag}_seed{seed}.json", summary)
    return summary


def run_long(seed: int = 0) -> dict[str, Any]:
    return _train(
        rec.build_recurrent,
        TAGS["Q16_long"],
        seed=int(seed),
        protocol_override=LONG_PROTOCOL,
        expected_total=Q16_EXPECTED_TOTAL,
        real_batch_identity=True,
    )


def run_plateau(seed: int = 0) -> dict[str, Any]:
    return _train(
        rec.build_recurrent,
        TAGS["Q16_plateau"],
        seed=int(seed),
        protocol_override=PLATEAU_PROTOCOL,
        expected_total=Q16_EXPECTED_TOTAL,
        real_batch_identity=True,
    )


def run_q24(protocol_name: str, seed: int = 0) -> dict[str, Any]:
    override = dict(PLATEAU_PROTOCOL if protocol_name == "plateau" else LONG_PROTOCOL)
    return _train(
        capacity.build_q24,
        TAGS["Q24_retry"],
        seed=int(seed),
        protocol_override=override,
        expected_total=Q24_EXPECTED_TOTAL,
        real_batch_identity=False,
    )


# ---------------------------------------------------------------------------
# activation diagnostics (q0 / q1 width utilisation)
# ---------------------------------------------------------------------------


def _diagnose(
    name: str,
    seed: int,
    *,
    builder: Callable[[int], torch.nn.Module],
    protocol_name: str,
) -> dict[str, Any]:
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)
    model = builder(int(seed)).to(device).eval()
    tag = TAGS["Q24_retry"] if name == "Q24" else TAGS["Q16_" + protocol_name]
    state_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    if not state_path.exists():
        raise RuntimeError(f"missing trained state: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    pre: list[torch.Tensor] = []
    post: list[torch.Tensor] = []
    handles = [
        model.pair_encoder.layers[-2].register_forward_hook(
            lambda _m, _i, o: pre.append(o.detach().clone())
        ),
        model.pair_encoder.register_forward_hook(
            lambda _m, _i, o: post.append(o.detach().clone())
        ),
    ]
    model.capture_diagnostics = True
    with torch.no_grad():
        out = model.encode(batch)
    for handle in handles:
        handle.remove()
    if len(pre) != 2 or len(post) != 2:
        raise RuntimeError(f"expected 2 pair rounds, got pre={len(pre)} post={len(post)}")
    q0_post, q1_post = post
    q0_pre, q1_pre = pre
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "name": name,
        "seed": int(seed),
        "protocol_name": protocol_name,
        "q_dim": int(model.pair_hidden),
        "params": _n_params(model),
        "state_path": str(state_path),
        "split": "first 128 official-valid molecules",
        "q0_post_relu": capacity._activation_stats(q0_post),
        "q1_post_relu": capacity._activation_stats(q1_post),
        "q0_pre_activation": {
            "negative_fraction": float((q0_pre < 0).float().mean()),
            "mean": float(q0_pre.mean()),
            "std": float(q0_pre.std()),
        },
        "q1_pre_activation": {
            "negative_fraction": float((q1_pre < 0).float().mean()),
            "mean": float(q1_pre.mean()),
            "std": float(q1_pre.std()),
        },
        "forward_finite": bool(torch.isfinite(out).all()),
        "official_test_loaded": False,
    }
    return payload


def diagnose_q16(protocol_name: str, seed: int = 0) -> dict[str, Any]:
    result = _diagnose(
        "Q16", int(seed), builder=rec.build_recurrent, protocol_name=protocol_name
    )
    _write_json(RESULTS_DIR / f"diagnostics_q16_{protocol_name}_seed{seed}.json", result)
    return result


def diagnose_q24(protocol_name: str, seed: int = 0) -> dict[str, Any]:
    result = _diagnose(
        "Q24", int(seed), builder=capacity.build_q24, protocol_name=protocol_name
    )
    _write_json(RESULTS_DIR / f"diagnostics_q24_{protocol_name}_seed{seed}.json", result)
    return result


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def _summary_of(tag: str, seed: int = 0) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{tag}_seed{seed}.json"
    return _read_json(path) if path.exists() else None


def _row_from_summary(label: str, summary: Mapping[str, Any], reference: float) -> dict[str, Any]:
    valid = float(summary["best_valid_mae"])
    return {
        "label": label,
        "best_valid_mae": valid,
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs_run"]),
        "train_mae_at_best_valid": float(summary["train_loss_at_best"]),
        "final_valid_mae": float(summary["final_valid_mae"]),
        "horizon_boundary_warning": bool(summary["horizon_boundary_warning"]),
        "early_stopped": bool(summary["early_stopped"]),
        "improvement_over_current_q16": float(reference - valid),
        "scheduler": summary.get("scheduler", "none"),
        "final_lr": summary.get("final_lr"),
        "min_lr_reached": summary.get("min_lr_reached"),
        "scheduler_events": summary.get("scheduler_events", []),
        "epoch_time_s": summary.get("epoch_time_s"),
    }


def classify_regime(
    long_imp: float | None, plateau_imp: float | None
) -> tuple[str, list[str], str]:
    """Map the two improvements to the pre-registered regime labels.

    The four rules in the brief overlap (a run can be both horizon-limited and
    generally under-optimised), so this returns a *primary* label plus every
    applicable label and a rationale.  ``improvement`` is always
    ``current_Q16 - variant`` (positive => variant better).
    """
    if long_imp is None and plateau_imp is None:
        return "not_run", [], "neither Phase B nor Phase C has run"
    applicable: list[str] = []
    if long_imp is not None and long_imp >= CONTINUE_GATE:
        applicable.append("training horizon limited")
    if plateau_imp is not None and plateau_imp >= CONTINUE_GATE:
        applicable.append("LR schedule limited")
    if (
        long_imp is not None
        and long_imp >= CONTINUE_GATE
        and plateau_imp is not None
        and plateau_imp >= CONTINUE_GATE
    ):
        applicable.append("optimization regime clearly suboptimal")
    if not applicable:
        return (
            "no meaningful optimization signal",
            applicable,
            "neither long nor plateau improves by >= +0.002",
        )
    if (
        plateau_imp is not None
        and plateau_imp >= CONTINUE_GATE
        and (long_imp is None or plateau_imp - long_imp >= PLATEAU_OVER_LONG)
    ):
        return (
            "LR schedule limited",
            applicable,
            "plateau clears +0.002 and beats long by >= 0.001 -> LR decay is the key factor",
        )
    if long_imp is not None and long_imp >= CONTINUE_GATE:
        return (
            "training horizon limited",
            applicable,
            "long clears +0.002 and plateau adds < 0.001 -> horizon / patience too short",
        )
    return (
        "LR schedule limited",
        applicable,
        "only plateau clears +0.002 -> LR decay is the key factor",
    )


def decision() -> dict[str, Any]:
    long_summary = _summary_of(TAGS["Q16_long"])
    plateau_summary = _summary_of(TAGS["Q16_plateau"])
    rows = []
    if long_summary is not None:
        rows.append(_row_from_summary("long_constant", long_summary, Q16_SEED0_VALID))
    if plateau_summary is not None:
        rows.append(_row_from_summary("plateau", plateau_summary, Q16_SEED0_VALID))
    long_imp = (
        float(Q16_SEED0_VALID - long_summary["best_valid_mae"])
        if long_summary is not None
        else None
    )
    plateau_imp = (
        float(Q16_SEED0_VALID - plateau_summary["best_valid_mae"])
        if plateau_summary is not None
        else None
    )

    # --- decision gate (task brief rules 1-4) -----------------------------
    regime, applicable_regimes, rationale = classify_regime(long_imp, plateau_imp)

    best_new = max(
        [("long_constant", long_imp), ("plateau", plateau_imp)],
        key=lambda item: -1.0 if item[1] is None else item[1],
    )
    best_improvement = best_new[1]
    trigger = bool(best_improvement is not None and best_improvement >= CONTINUE_GATE)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "current_q16_reference_valid_mae": Q16_SEED0_VALID,
        "rows": rows,
        "long_improvement": long_imp,
        "plateau_improvement": plateau_imp,
        "continue_gate": CONTINUE_GATE,
        "strong_gate": STRONG_GATE,
        "regime": regime,
        "applicable_regimes": applicable_regimes,
        "rationale": rationale,
        "best_new_protocol": best_new[0] if trigger else None,
        "best_new_improvement": best_improvement,
        "q24_retry_triggered": trigger,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def q24_retry() -> dict[str, Any]:
    dec = decision()
    if not dec["q24_retry_triggered"]:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "executed": False,
            "reason": "no Phase B/C protocol cleared the +0.002 continue gate",
            "decision": dec,
            "official_test_loaded": False,
        }
        _write_json(RESULTS_DIR / "q24_retry.json", payload)
        return payload
    protocol_name = str(dec["best_new_protocol"])
    protocol_name = "plateau" if protocol_name == "plateau" else "long"
    summary = run_q24(protocol_name, seed=0)
    diag = diagnose_q24(protocol_name, seed=0)
    # Compare as fairly as possible: the same protocol Q16 run.
    ref_tag = TAGS["Q16_plateau"] if protocol_name == "plateau" else TAGS["Q16_long"]
    ref_summary = _summary_of(ref_tag) or {}
    ref_diag = _diagnose("Q16", 0, builder=rec.build_recurrent, protocol_name=protocol_name)
    _write_json(RESULTS_DIR / f"diagnostics_q16_{protocol_name}_seed0.json", ref_diag)
    q24_valid = float(summary["best_valid_mae"])
    q16_valid = float(ref_summary.get("best_valid_mae", float("nan")))
    q24_train = float(summary["train_loss_at_best"])
    q16_train = float(ref_summary.get("train_loss_at_best", float("nan")))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "executed": True,
        "protocol_name": protocol_name,
        "protocol": dict(summary["protocol"]),
        "q16_new_protocol": {
            "valid_mae": q16_valid,
            "train_mae_at_best": q16_train,
            "best_epoch": ref_summary.get("best_epoch"),
            "epochs_run": ref_summary.get("epochs_run"),
        },
        "q24_new_protocol": {
            "valid_mae": q24_valid,
            "train_mae_at_best": q24_train,
            "best_epoch": int(summary["best_epoch"]),
            "epochs_run": int(summary["epochs_run"]),
            "parameters": int(summary["parameters"]),
        },
        "q24_minus_q16_valid": float(q16_valid - q24_valid),
        "q24_train_better_than_q16": bool(q24_train <= q16_train),
        "q16_diagnostics": ref_diag,
        "q24_diagnostics": diag,
        "q16_q1": ref_diag["q1_post_relu"],
        "q24_q1": diag["q1_post_relu"],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "q24_retry.json", payload)
    return payload


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _row_from_reference() -> dict[str, Any]:
    s0 = _read_json(REFERENCE_DIR / "recurrent_seed0.json")
    return {
        "label": "current_q16_seed0",
        "best_valid_mae": float(s0["best_valid_mae"]),
        "best_epoch": int(s0["best_epoch"]),
        "epochs_run": int(s0["epochs_run"]),
        "train_mae_at_best_valid": float(s0["train_loss_at_best"]),
        "final_valid_mae": float(s0["final_valid_mae"]),
        "scheduler": "none",
        "improvement_over_current_q16": 0.0,
    }


def _within_code_current_protocol() -> dict[str, Any] | None:
    """Apply the frozen 240/40 early-stop rule to the current-code curve.

    The long run's first 240 epochs *are* the current-protocol trajectory under
    the current code (the two protocols share the same per-epoch update), so
    truncating its curve and re-applying ``max_epochs=240, patience=40`` yields
    the current-code current-protocol checkpoint without a new run.
    """
    long_summary = _summary_of(TAGS["Q16_long"])
    if long_summary is None:
        return None
    curve = _curve(Path(str(long_summary["curve_path"])))
    best = float("inf")
    best_epoch = 1
    stale = 0
    for row in curve:
        epoch = int(row["epoch"])
        if epoch > 240:
            break
        valid = float(row["valid_mae"])
        if valid < best:
            best, best_epoch, stale = valid, epoch, 0
        else:
            stale += 1
        if stale >= 40:
            break
    train_at_best = float(curve[best_epoch - 1]["train_mae"])
    return {
        "label": "current_protocol_current_code",
        "best_valid_mae": best,
        "best_epoch": best_epoch,
        "epochs_run": min(int(curve[-1]["epoch"]), 240),
        "train_mae_at_best_valid": train_at_best,
        "improvement_over_current_q16": float(Q16_SEED0_VALID - best),
    }


def report() -> dict[str, Any]:
    dec = decision()
    rows = [_row_from_reference()]
    within = _within_code_current_protocol()
    if within is not None:
        rows.append(within)
    rows.extend(dec["rows"])
    reproduction = (
        _read_json(RESULTS_DIR / "reproduction_control" / "reproduction_control.json")
        if (RESULTS_DIR / "reproduction_control" / "reproduction_control.json").exists()
        else None
    )
    long_row = next((r for r in dec["rows"] if r["label"] == "long_constant"), None)
    plateau_row = next((r for r in dec["rows"] if r["label"] == "plateau"), None)
    paired_effects: dict[str, Any] = {}
    if within is not None and long_row is not None:
        paired_effects["horizon_patience_long_minus_within"] = float(
            within["best_valid_mae"] - long_row["best_valid_mae"]
        )
    if long_row is not None and plateau_row is not None:
        paired_effects["scheduler_plateau_minus_long"] = float(
            long_row["best_valid_mae"] - plateau_row["best_valid_mae"]
        )
    if within is not None and plateau_row is not None:
        paired_effects["combined_plateau_minus_within"] = float(
            within["best_valid_mae"] - plateau_row["best_valid_mae"]
        )
    paired_effects["historical_vs_current_code_reproduction_gap"] = float(
        None if within is None else within["best_valid_mae"] - Q16_SEED0_VALID
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "phase_a_seed_ensemble": (
            _read_json(RESULTS_DIR / "phase_a_seed_ensemble.json")
            if (RESULTS_DIR / "phase_a_seed_ensemble.json").exists()
            else None
        ),
        "comparison_rows": rows,
        "within_code_current_protocol": within,
        "paired_effects": paired_effects,
        "reproduction_control": reproduction,
        "decision": dec,
        "q24_retry": (
            _read_json(RESULTS_DIR / "q24_retry.json")
            if (RESULTS_DIR / "q24_retry.json").exists()
            else None
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "phase_a",
            "fairness",
            "long",
            "plateau",
            "diagnose_q16",
            "diagnose_q24",
            "decision",
            "q24_retry",
            "report",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", choices=["long", "plateau"], default="long")
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    stage = args.stage
    if stage == "phase_a":
        print(json.dumps(phase_a(), indent=2, default=str), flush=True)
    if stage == "fairness":
        print(json.dumps(fairness(), indent=2, default=str), flush=True)
    if stage == "long":
        print(json.dumps(run_long(args.seed), indent=2, default=str), flush=True)
    if stage == "plateau":
        print(json.dumps(run_plateau(args.seed), indent=2, default=str), flush=True)
    if stage == "diagnose_q16":
        print(
            json.dumps(diagnose_q16(args.protocol, args.seed), indent=2, default=str),
            flush=True,
        )
    if stage == "diagnose_q24":
        print(
            json.dumps(diagnose_q24(args.protocol, args.seed), indent=2, default=str),
            flush=True,
        )
    if stage == "decision":
        print(json.dumps(decision(), indent=2, default=str), flush=True)
    if stage == "q24_retry":
        print(json.dumps(q24_retry(), indent=2, default=str), flush=True)
    if stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    if stage == "all":
        phase_a()
        fairness()
        run_long(0)
        run_plateau(0)
        decision()
        q24_retry()
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
