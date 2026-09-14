"""Capacity-scaling audit for the frozen compact-v4 T=2 recurrent pair--centre.

Goal (single question): is the current best model limited mainly by the
*relation* width ``q_dim`` (16) and/or by the *centre* width ``h_dim`` (48)?

The frozen reference is the T=2 recurrent pair--centre model
(``zinc_compact_v4_recurrent_pair_centre``): 82,115 params, ``h=48``,
``q=16``, seed0 valid 0.138376, seed1 valid 0.133440.  This module changes
**only** the capacity allocation:

* Phase A -- read the existing Q16 seed0/seed1 logs + curves and report
  train/valid fit, best epoch, best/final train MAE and the train--valid gap.
* Phase B -- ``Q16`` / ``Q24`` / ``Q32`` at fixed ``h=48`` (q scalar width).
* Phase C -- one balanced ~100k--110k variant (``h=64, q=32``, 104,211 params)
  that raises the centre state *and* the relation width.
* Phase D -- replication gate: buy seed1 only if a new config beats the Q16
  reference by >= 0.003 valid MAE.

Everything else is inherited verbatim: tokenizer / patch construction /
relation descriptor / ReLU / readout / small head / optimizer / scheduler /
batch size / protocol / seeds / official split.  Official test is NEVER
loaded by this module.

Mechanism note
--------------
``PatchPathSmallHeadModel`` hard-codes the reference ``R_DIM=302`` and the
recurrent model hard-codes ``Q_DIM=16``; both are pure construction guards.
This audit temporarily relaxes those two guards *only while instantiating a
width variant* and then restores them, so the frozen reference code path is
untouched.  The small raw head (13,13) and every other module are unchanged;
the head *input* width grows because the pair-moment block of ``R`` grows with
``q`` -- that is the unavoidable consequence of widening the relation channel.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_capacity <stage> [--seed 0]

Stages: ``phase_a params sanity train_q24 train_q32 train_balanced
diagnose_q24 diagnose_q32 diagnose_balanced decide replicate all``.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_capacity"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_capacity_v1"

# Frozen T=2 recurrent reference (do not overwrite / retrain).
REFERENCE_DIR = rec.RESULTS_DIR
Q16_SEED0_VALID = 0.13837560486892472
Q16_SEED1_VALID = 0.13343997858563672

EXPECTED_Q16_TOTAL = 82115
EXPECTED_Q16_HEAD = 4135
DISTANCE_BUCKETS = zpp.DISTANCE_BUCKETS
TOPOLOGY_OUT_WIDTH = 8  # hinge topology branch width (fixed by canonical config)

# Pre-registered Phase B/C continuation gates (from the task brief).
STRONG_GATE = 0.002   # q-width signal is clear -> continue to Q32
WEAK_GATE = 0.001     # weak signal -> still worth a Q32 trend read
REPLICATION_GATE = 0.003  # Phase D: buy seed1 only above this improvement

EPS = 1.0e-8

# Exact capacity variants.  ``h`` = patch/centre width, ``q`` = relation width.
CONFIGS: dict[str, dict[str, int]] = {
    "Q16": {"h": 48, "q": 16},
    "Q24": {"h": 48, "q": 24},
    "Q32": {"h": 48, "q": 32},
    "BALANCED": {"h": 64, "q": 32},
}

# Exact parameter counts computed by ``params`` (recorded to disk as well).
EXPECTED_PARAMS: dict[str, int] = {
    "Q16": 82115,
    "Q24": 91211,
    "Q32": 100307,
    "BALANCED": 104211,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _rss_peak_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _first_batch(valid_data: Sequence[Any], device: torch.device) -> Any:
    return next(iter(zpp._make_loader(list(valid_data)[:128], 128, False, 0))).to(device)


# ---------------------------------------------------------------------------
# width-variant construction
# ---------------------------------------------------------------------------


def relation_readout_width(h: int, q: int) -> int:
    """Exact ``unified_graph_width`` for (h, q) under the frozen readout.

    unary moments = 2h+1; pair moments = 5 * (2q+1); global = 32;
    topology hinge branch = 8.
    """
    return (2 * int(h) + 1) + DISTANCE_BUCKETS * (2 * int(q) + 1) + 32 + TOPOLOGY_OUT_WIDTH


@contextmanager
def _construction_guards(h: int, q: int):
    """Temporarily relax the two frozen construction guards (documented above)."""
    saved_q = rec.Q_DIM
    saved_r = shead.R_DIM
    rec.Q_DIM = int(q)
    shead.R_DIM = relation_readout_width(h, q)
    try:
        yield
    finally:
        rec.Q_DIM = saved_q
        shead.R_DIM = saved_r


def build_capacity(seed: int, *, h: int, q: int) -> nn.Module:
    """Build a T=2 recurrent width-variant with the frozen init convention.

    Exactly mirrors ``rec.build_recurrent``: shared tensors that keep their
    shape are copied bit-exactly from the matched Q16 baseline initialisation;
    the head is re-initialised from the historical ``head_seed=0`` stream.
    """
    baseline = shead.build_baseline(int(seed))
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(int(seed))
    kwargs = shead._base_kwargs()
    kwargs["patch_hidden"] = int(h)
    kwargs.pop("pair_hidden", None)
    with _construction_guards(int(h), int(q)):
        model = rec.PatchPathRecurrentPairCentreModel(
            6785,
            32,
            pair_hidden=int(q),
            recurrence_rounds=rec.RECURRENCE_ROUNDS,
            recurrence_enabled=True,
            recurrence_mode="refresh",
            **kwargs,
        )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    torch.manual_seed(int(shead.SMALL_HEAD_SEED))
    model.head = shead.GenericReader(int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN)
    return model


def build_q16(seed: int = 0) -> nn.Module:
    return build_capacity(seed, h=48, q=16)


def build_q24(seed: int = 0) -> nn.Module:
    return build_capacity(seed, h=48, q=24)


def build_q32(seed: int = 0) -> nn.Module:
    return build_capacity(seed, h=48, q=32)


def build_balanced(seed: int = 0) -> nn.Module:
    return build_capacity(seed, h=64, q=32)


BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "Q16": build_q16,
    "Q24": build_q24,
    "Q32": build_q32,
    "BALANCED": build_balanced,
}

TAGS = {"Q16": "q16", "Q24": "q24", "Q32": "q32", "BALANCED": "balanced"}


# ---------------------------------------------------------------------------
# Phase A -- current Q16 capacity audit from existing logs
# ---------------------------------------------------------------------------


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


def _audit_one(tag: str, seed: int) -> dict[str, Any]:
    summary = _read_json(REFERENCE_DIR / f"{tag}_seed{seed}.json")
    curve = _curve(Path(summary["curve_path"]))
    best_epoch = int(summary["best_epoch"])
    best_valid = float(summary["best_valid_mae"])
    train_at_best = float(summary["train_loss_at_best"])
    train_curve = np.asarray([row["train_mae"] for row in curve], dtype=np.float64)
    valid_curve = np.asarray([row["valid_mae"] for row in curve], dtype=np.float64)
    n = len(curve)
    tail = max(1, n // 10)
    best_train = float(train_curve.min())
    final_train = float(train_curve[-1])
    gap_at_best = best_valid - train_at_best
    # Descriptive trend over the last 10% of epochs.
    train_slope = float(train_curve[-1] - train_curve[-tail])
    valid_slope = float(valid_curve[-1] - valid_curve[-tail])
    return {
        "tag": tag,
        "seed": int(seed),
        "parameters": int(summary["parameters"]),
        "best_valid_mae": best_valid,
        "best_epoch": best_epoch,
        "epochs_run": int(summary["epochs_run"]),
        "final_valid_mae": float(summary["final_valid_mae"]),
        "train_mae_at_best_valid": train_at_best,
        "best_train_mae": best_train,
        "final_train_mae": final_train,
        "train_valid_gap_at_best": gap_at_best,
        "best_valid_epoch_min_train_gap": float(best_valid - train_at_best),
        "first_epoch_train_mae": float(train_curve[0]),
        "first_epoch_valid_mae": float(valid_curve[0]),
        "tail_train_delta": train_slope,
        "tail_valid_delta": valid_slope,
        "min_valid_mae": float(valid_curve.min()),
        "min_valid_epoch": int(valid_curve.argmin() + 1),
        "tail_epochs": tail,
        "early_stopped": bool(summary["early_stopped"]),
        "horizon_boundary_warning": bool(summary["horizon_boundary_warning"]),
        "wall_clock_s": float(summary["wall_clock_s"]),
        "epoch_time_s": float(summary["wall_clock_s"]) / max(int(summary["epochs_run"]), 1),
        "curve_path": str(summary["curve_path"]),
    }


def phase_a() -> dict[str, Any]:
    rows = [_audit_one(tag, seed) for tag in ("recurrent",) for seed in (0, 1)]
    baseline_rows = [_audit_one("baseline", seed) for seed in (0, 1)]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "description": "Q16 T=2 recurrent pair-centre capacity audit (existing logs).",
        "recurrent": rows,
        "matched_baseline_1round": baseline_rows,
        "seed0_summary": {
            "best_valid_mae": rows[0]["best_valid_mae"],
            "train_mae_at_best_valid": rows[0]["train_mae_at_best_valid"],
            "train_valid_gap_at_best": rows[0]["train_valid_gap_at_best"],
            "best_train_mae": rows[0]["best_train_mae"],
            "final_train_mae": rows[0]["final_train_mae"],
        },
        "read": (
            "Train MAE is well below valid MAE and the gap is large; both curves "
            "are still moving at the end of the run.  This supports a "
            "generalization / finite-data bottleneck over pure under-capacity, "
            "but the train error itself is not at an interpolation floor either."
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "phase_a_capacity_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# exact parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    rows = []
    for name, cfg in CONFIGS.items():
        model = BUILDERS[name](0)
        total = _n_params(model)
        head = _n_params(model.head)
        rows.append(
            {
                "name": name,
                "h_dim": int(cfg["h"]),
                "q_dim": int(cfg["q"]),
                "total_params": total,
                "head_params": head,
                "backbone_params": total - head,
                "unified_graph_width": int(model.unified_graph_width),
                "pooled_unary_width": int(model.pooled_unary_width),
                "pooled_pair_width": int(model.pooled_pair_width),
                "center_context_width": int(model.center_context_width),
                "delta_vs_Q16": total - EXPECTED_PARAMS["Q16"],
                "expected": EXPECTED_PARAMS.get(name),
                "matches_expected": total == EXPECTED_PARAMS.get(name),
                "recurrence_rounds": int(model.recurrence_rounds),
                "recurrence_mode": str(model.recurrence_mode),
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_Q16": EXPECTED_PARAMS["Q16"],
        "rows": rows,
        "all_match_expected": bool(all(row["matches_expected"] for row in rows)),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# sanity -- identity of the Q16 builder + finite forward/backward for variants
# ---------------------------------------------------------------------------


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)

    frozen = rec.build_recurrent(0).to(device).eval()
    capacity_q16 = build_q16(0).to(device).eval()
    frozen_state = {k: v.detach().clone() for k, v in frozen.state_dict().items()}
    capacity_state = {k: v.detach().clone() for k, v in capacity_q16.state_dict().items()}
    state_hash_equal = _state_hash(frozen_state) == _state_hash(capacity_state)
    with torch.no_grad():
        out_frozen = frozen.encode(batch)
        out_q16 = capacity_q16.encode(batch)
    output_max_diff = float((out_frozen - out_q16).abs().max())

    per_variant: dict[str, Any] = {}
    all_finite = True
    for name, builder in BUILDERS.items():
        model = builder(0).to(device)
        model.train()
        out = model(batch)
        loss = F.l1_loss(out.view(-1), batch.y.view(-1))
        loss.backward()
        grad_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        grad_nonzero = any(
            parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0
            for parameter in model.parameters()
        )
        counts = model.module_call_counts(batch)
        finite = bool(torch.isfinite(out).all()) and bool(grad_finite)
        all_finite = all_finite and finite
        per_variant[name] = {
            "parameters": _n_params(model),
            "head_parameters": _n_params(model.head),
            "unified_graph_width": int(model.unified_graph_width),
            "module_call_counts": counts,
            "forward_finite": bool(torch.isfinite(out).all()),
            "backward_grad_finite": bool(grad_finite),
            "backward_grad_nonzero": bool(grad_nonzero),
            "loss": float(loss.detach()),
        }
        model.zero_grad(set_to_none=True)

    checks = {
        "q16_capacity_builder_matches_frozen_builder": state_hash_equal,
        "q16_capacity_output_matches_frozen": output_max_diff == 0.0,
        "all_variants_forward_backward_finite": bool(all_finite),
        "q16_total_expected": per_variant["Q16"]["parameters"] == EXPECTED_Q16_TOTAL,
        "q16_head_expected": per_variant["Q16"]["head_parameters"] == EXPECTED_Q16_HEAD,
        "weight_tying_2x_every_variant": all(
            row["module_call_counts"]
            == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
            for row in per_variant.values()
        ),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "q16_state_hash_equal": state_hash_equal,
        "q16_output_max_diff": output_max_diff,
        "variants": per_variant,
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"capacity sanity checks failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_config(
    name: str,
    seed: int = 0,
    *,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = CONFIGS[name]
    tag = TAGS[name]
    train_data, valid_data, _ = rec.load_encoded()
    expected_total = _n_params(BUILDERS[name](int(seed)))

    original_dirs = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    original_protocol = shead.OPTIMIZED_PROTOCOL
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    protocol = dict(original_protocol)
    if protocol_override:
        protocol.update(protocol_override)
    shead.OPTIMIZED_PROTOCOL = protocol
    rss_before = _rss_peak_kb()
    started = time.perf_counter()
    try:
        summary = shead.train_model(
            build_fn=BUILDERS[name],
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=tag,
            save_state=True,
            # The R-dim identity guard compares against the Q16 baseline; it is
            # only meaningful for the matched width and is skipped here.
            real_batch_identity=False,
            expected_total=int(expected_total),
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original_dirs
        shead.OPTIMIZED_PROTOCOL = original_protocol
    elapsed = float(time.perf_counter() - started)

    summary = dict(summary)
    summary["config_name"] = name
    summary["h_dim"] = int(cfg["h"])
    summary["q_dim"] = int(cfg["q"])
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_rss_kb"] = _rss_peak_kb()
    summary["peak_rss_delta_kb"] = _rss_peak_kb() - int(rss_before)
    summary["outer_wall_clock_s"] = elapsed
    _write_json(RESULTS_DIR / f"{tag}_seed{seed}.json", summary)

    diagnostics = activate_diagnostics(name, int(seed), summary=summary)
    _write_json(RESULTS_DIR / f"diagnostics_{tag}_seed{seed}.json", diagnostics)
    return summary


def train_q24(seed: int = 0) -> dict[str, Any]:
    return train_config("Q24", int(seed))


def train_q32(seed: int = 0) -> dict[str, Any]:
    return train_config("Q32", int(seed))


def train_balanced(seed: int = 0) -> dict[str, Any]:
    return train_config("BALANCED", int(seed))


# ---------------------------------------------------------------------------
# post-training activation diagnostics (q0 / q1 width utilisation)
# ---------------------------------------------------------------------------


def _activation_stats(value: torch.Tensor) -> dict[str, float]:
    """Global + per-dimension activity of a (n_pairs, q_dim) relation state."""
    array = value.detach().double().cpu().numpy()
    per_dim_std = array.std(axis=0)
    per_dim_max = array.max(axis=0)
    per_dim_pos = (array > 0).mean(axis=0)
    variance = per_dim_std**2
    participation = float((variance.sum() ** 2) / (np.square(variance).sum() + EPS))
    return {
        "zero_fraction": float((array == 0).mean()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "abs_mean": float(np.abs(array).mean()),
        "n_pairs": int(array.shape[0]),
        "q_dim": int(array.shape[1]),
        "dead_dim_fraction": float((per_dim_std < 1.0e-6).mean()),
        "never_active_dim_fraction": float((per_dim_max <= 0).mean()),
        "mean_per_dim_positive_fraction": float(per_dim_pos.mean()),
        "min_per_dim_positive_fraction": float(per_dim_pos.min()),
        "effective_rank_participation": participation,
    }


def activate_diagnostics(
    name: str, seed: int, *, summary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Capture q0/q1 (and their pre-activations) on a small valid batch.

    Returns zero-training statistics used to check whether widening ``q`` adds
    mostly inactive dimensions.  The trained selection state is loaded; no
    gradient is taken.
    """
    device = torch.device("cpu")
    tag = TAGS[name]
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)
    model = BUILDERS[name](int(seed)).to(device).eval()
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
        "h_dim": int(CONFIGS[name]["h"]),
        "q_dim": int(CONFIGS[name]["q"]),
        "state_path": str(state_path),
        "split": "first 128 official-valid molecules",
        "q0_post_relu": _activation_stats(q0_post),
        "q1_post_relu": _activation_stats(q1_post),
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
        "summary_best_epoch": None if summary is None else int(summary["best_epoch"]),
        "summary_best_valid_mae": (
            None if summary is None else float(summary["best_valid_mae"])
        ),
        "official_test_loaded": False,
    }
    return payload


def diagnose_q24(seed: int = 0) -> dict[str, Any]:
    result = activate_diagnostics("Q24", int(seed))
    _write_json(RESULTS_DIR / f"diagnostics_q24_seed{seed}.json", result)
    return result


def diagnose_q32(seed: int = 0) -> dict[str, Any]:
    result = activate_diagnostics("Q32", int(seed))
    _write_json(RESULTS_DIR / f"diagnostics_q32_seed{seed}.json", result)
    return result


def diagnose_balanced(seed: int = 0) -> dict[str, Any]:
    result = activate_diagnostics("BALANCED", int(seed))
    _write_json(RESULTS_DIR / f"diagnostics_balanced_seed{seed}.json", result)
    return result


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def decision_config(name: str, seed: int = 0, reference: float = Q16_SEED0_VALID) -> dict[str, Any]:
    path = RESULTS_DIR / f"{TAGS[name]}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing run for {name} seed{seed}: {path}")
    summary = _read_json(path)
    value = float(summary["best_valid_mae"])
    improvement = float(reference) - value  # positive => variant better
    if improvement >= STRONG_GATE:
        verdict, next_step = "STRONG_POSITIVE", "continue_trend"
    elif improvement >= WEAK_GATE:
        verdict, next_step = "WEAK_POSITIVE", "read_trend"
    elif improvement > -WEAK_GATE:
        verdict, next_step = "NO_MEANINGFUL_SIGNAL", "stop"
    else:
        verdict, next_step = "DEGRADED", "stop"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "name": name,
        "seed": int(seed),
        "h_dim": int(summary["h_dim"]),
        "q_dim": int(summary["q_dim"]),
        "parameters": int(summary["parameters"]),
        "best_valid_mae": value,
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs_run"]),
        "horizon_boundary_warning": bool(summary["horizon_boundary_warning"]),
        "train_mae_at_best_valid": float(summary["train_loss_at_best"]),
        "reference_Q16_valid_mae": float(reference),
        "improvement_over_Q16": improvement,
        "strong_gate": STRONG_GATE,
        "weak_gate": WEAK_GATE,
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_{TAGS[name]}_seed{seed}.json", payload)
    return payload


def decide(seed: int = 0) -> dict[str, Any]:
    reference = Q16_SEED0_VALID if int(seed) == 0 else Q16_SEED1_VALID
    rows = []
    for name in ("Q24", "Q32", "BALANCED"):
        path = RESULTS_DIR / f"{TAGS[name]}_seed{seed}.json"
        if path.exists():
            rows.append(decision_config(name, int(seed), reference=reference))
    q24 = next((row for row in rows if row["name"] == "Q24"), None)
    q32 = next((row for row in rows if row["name"] == "Q32"), None)
    balanced = next((row for row in rows if row["name"] == "BALANCED"), None)
    # Phase C is authorised only when q-width shows a positive signal.
    phase_c_authorised = bool(
        (q24 is not None and q24["improvement_over_Q16"] >= WEAK_GATE)
        or (q32 is not None and q32["improvement_over_Q16"] >= WEAK_GATE)
    )
    if q24 is not None and q24["improvement_over_Q16"] <= WEAK_GATE:
        phase_c_authorised = False
    # Phase D replication gate against the best new config.
    best = min(rows, key=lambda row: row["best_valid_mae"]) if rows else None
    replicate = bool(best is not None and best["improvement_over_Q16"] >= REPLICATION_GATE)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "reference_Q16_valid_mae": float(reference),
        "rows": rows,
        "phase_c_authorised_by_q_width": phase_c_authorised,
        "best_new_config": None if best is None else best["name"],
        "best_improvement_over_Q16": (
            None if best is None else best["improvement_over_Q16"]
        ),
        "replication_gate": REPLICATION_GATE,
        "replicate_seed1": replicate,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_summary_seed{seed}.json", payload)
    return payload


def replicate() -> dict[str, Any]:
    rows = []
    for name in ("Q24", "Q32", "BALANCED"):
        if (RESULTS_DIR / f"decision_{TAGS[name]}_seed0.json").exists():
            improvement = float(
                _read_json(RESULTS_DIR / f"decision_{TAGS[name]}_seed0.json")[
                    "improvement_over_Q16"
                ]
            )
            if improvement >= REPLICATION_GATE:
                rows.append((name, improvement))
    if not rows:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "replicated": False,
            "reason": "no seed0 config cleared the +0.003 replication gate",
            "official_test_loaded": False,
        }
        _write_json(RESULTS_DIR / "replication_decision.json", payload)
        return payload
    name = max(rows, key=lambda item: item[1])[0]
    summary = train_config(name, seed=1)
    decision = decision_config(name, seed=1, reference=Q16_SEED1_VALID)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "name": name,
        "seed1_valid_mae": float(summary["best_valid_mae"]),
        "seed1_best_epoch": int(summary["best_epoch"]),
        "seed1_improvement_over_Q16": float(decision["improvement_over_Q16"]),
        "replicated": bool(decision["improvement_over_Q16"] > 0.0),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "replication_decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# reporting helper
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    """Compact console + JSON report of whatever stages have been run."""
    rows = []
    for name in ("Q16", "Q24", "Q32", "BALANCED"):
        tag = TAGS[name]
        seed0 = RESULTS_DIR / f"{tag}_seed0.json"
        if seed0.exists():
            summary = _read_json(seed0)
        elif name == "Q16":
            summary = _read_json(REFERENCE_DIR / "recurrent_seed0.json")
        else:
            summary = None
        diag_path = RESULTS_DIR / f"diagnostics_{tag}_seed0.json"
        diag = _read_json(diag_path) if diag_path.exists() else None
        rows.append(
            {
                "name": name,
                "h": CONFIGS[name]["h"],
                "q": CONFIGS[name]["q"],
                "params": None if summary is None else int(summary.get("parameters", EXPECTED_PARAMS[name])),
                "best_valid_mae": None if summary is None else float(summary["best_valid_mae"]),
                "best_epoch": None if summary is None else int(summary["best_epoch"]),
                "train_mae_at_best": None if summary is None else float(summary["train_loss_at_best"]),
                "epoch_time_s": None if summary is None else summary.get("epoch_time_s"),
                "peak_rss_kb": None if summary is None else summary.get("peak_rss_kb"),
                "q1_zero_fraction": None if diag is None else diag["q1_post_relu"]["zero_fraction"],
                "q1_dead_dim_fraction": None if diag is None else diag["q1_post_relu"]["dead_dim_fraction"],
                "q1_effective_rank": None if diag is None else diag["q1_post_relu"]["effective_rank_participation"],
            }
        )
    payload = {"rows": rows, "official_test_loaded": False}
    _write_json(RESULTS_DIR / "report_seed0.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def all_seed0() -> None:
    phase_a()
    params()
    sanity()
    train_q24(0)
    decision = decide(0)
    if decision["phase_c_authorised_by_q_width"]:
        train_q32(0)
        decision = decide(0)
    if decision["replicate_seed1"]:
        replicate()
    report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "phase_a",
            "params",
            "sanity",
            "train_q24",
            "train_q32",
            "train_balanced",
            "diagnose_q24",
            "diagnose_q32",
            "diagnose_balanced",
            "decide",
            "replicate",
            "report",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    stage = args.stage
    if stage == "phase_a":
        print(json.dumps(phase_a(), indent=2, default=str), flush=True)
    if stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    if stage == "train_q24":
        train_q24(args.seed)
    if stage == "train_q32":
        train_q32(args.seed)
    if stage == "train_balanced":
        train_balanced(args.seed)
    if stage == "diagnose_q24":
        print(json.dumps(diagnose_q24(args.seed), indent=2, default=str), flush=True)
    if stage == "diagnose_q32":
        print(json.dumps(diagnose_q32(args.seed), indent=2, default=str), flush=True)
    if stage == "diagnose_balanced":
        print(json.dumps(diagnose_balanced(args.seed), indent=2, default=str), flush=True)
    if stage == "decide":
        print(json.dumps(decide(args.seed), indent=2, default=str), flush=True)
    if stage == "replicate":
        print(json.dumps(replicate(), indent=2, default=str), flush=True)
    if stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    if stage == "all":
        all_seed0()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
