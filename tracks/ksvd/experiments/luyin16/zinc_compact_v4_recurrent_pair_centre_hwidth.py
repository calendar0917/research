"""Centre / patch-state width scaling for the frozen compact-v4 T=2 recurrent
pair--centre model.

Single question
---------------
The canonical T=2 recurrent pair--centre model is ``h_dim=48, q_dim=16``
(82,115 params).  Widening the *relation* channel ``q_dim: 16 -> 24`` gave no
signal.  The remaining candidate capacity bottleneck is the persistent
*centre* state ``h``.  This module changes exactly one number::

    h_dim: 48 -> 64        (q_dim stays 16, T stays 2)

Everything else is inherited verbatim from the frozen canonical code path:
tokenizer / patch construction / relation descriptor / pair projection /
pair encoder / ReLU / distance buckets / recurrent refresh logic / readout /
small raw head / optimizer / lr / wd / batch / max epochs / patience / no
scheduler / sequential execution / seeds.  Only the tensors that *depend on*
``patch_hidden`` grow; no new module is introduced.

Pre-registered 2-seed Top-5 soup gate against the canonical H48 reference
``0.134643558``::

    improvement >= 0.002, both seeds same direction -> centre-state capacity
    improvement 0.001..0.002, both seeds same direction -> weak signal
    improvement <= 0.001 -> no centre-width scaling
    one seed clearly up, the other clearly down -> inconclusive

The official **test** split is never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_hwidth <stage> [--seed 0]

Stages: ``params sanity train soup diagnostics decide report all``.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_hwidth"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SOUP_DIR = RESULTS_DIR / "soup_states"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_hwidth_v1"

H48 = {"h": 48, "q": 16}
H64 = {"h": 64, "q": 16}
WIDE_CANDIDATES = {"H80": 80, "H96": 96}

EXPECTED_H48_TOTAL = 82115
EXPECTED_H48_HEAD = 4135
EXPECTED_H64_TOTAL = 85763
EXPECTED_H64_HEAD = 4551
EXPECTED_H64_WIDTH = 334

TAG = "h64"
SEEDS = (0, 1)

# Canonical H48 reference (frozen compact_v4_recurrent_variance_diagnosis runs;
# selection state / Top-5 soup on the official-valid split only).
H48_REF_RAW = {0: 0.1406094916117727, 1: 0.13343997858563672}
H48_REF_SOUP = {0: 0.13707755148684372, 1: 0.13220956423232566}
H48_REF_SOUP_MEAN = 0.13464355785958469

# Pre-registered 2-seed Top-5 soup decision gates.
STRONG_GATE = 0.002
WEAK_GATE = 0.001
SIGN_FLIP_MAG = 0.0005

# Exact set of state_dict tensors whose shape must be the only thing that
# changes when ``patch_hidden`` 48 -> 64 at fixed ``q_dim=16``.
H_DEPENDENT_KEYS = frozenset(
    {
        "patch_encoder.layers.4.weight",
        "patch_encoder.layers.4.bias",
        "pair_projection.weight",
        "center_update.0.weight",
        "center_update.4.weight",
        "center_update.4.bias",
        "head.net.0.weight",
    }
)

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# io helpers
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


def _first_batch(valid_data: Sequence[Any], device: torch.device) -> Any:
    return next(iter(zpp._make_loader(list(valid_data)[:128], 128, False, 0))).to(device)


def build_h64(seed: int = 0) -> torch.nn.Module:
    """H64 builder: the frozen capacity builder with ``h=64, q=16``."""
    return cap.build_capacity(int(seed), h=int(H64["h"]), q=int(H64["q"]))


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    h48 = cap.build_q16(0)
    h64 = build_h64(0)
    s48 = {k: v.detach().clone() for k, v in h48.state_dict().items()}
    s64 = {k: v.detach().clone() for k, v in h64.state_dict().items()}

    changed = []
    for key in sorted(s48):
        a, b = s48[key], s64[key]
        if a.shape != b.shape or a.numel() != b.numel():
            changed.append(
                {
                    "key": key,
                    "h48_shape": list(a.shape),
                    "h64_shape": list(b.shape),
                    "h48_numel": int(a.numel()),
                    "h64_numel": int(b.numel()),
                    "delta_numel": int(b.numel()) - int(a.numel()),
                }
            )

    total48 = _n_params(h48)
    total64 = _n_params(h64)
    head48 = _n_params(h48.head)
    head64 = _n_params(h64.head)

    wide = {}
    for name, h in WIDE_CANDIDATES.items():
        model = cap.build_capacity(0, h=int(h), q=int(H64["q"]))
        wide[name] = {
            "h_dim": int(h),
            "q_dim": int(H64["q"]),
            "total_params": _n_params(model),
            "head_params": _n_params(model.head),
            "backbone_params": _n_params(model) - _n_params(model.head),
            "unified_graph_width": int(model.unified_graph_width),
        }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": [
            {
                "name": "H48",
                "h_dim": 48,
                "q_dim": 16,
                "total_params": total48,
                "head_params": head48,
                "backbone_params": total48 - head48,
                "unified_graph_width": int(h48.unified_graph_width),
                "pooled_unary_width": int(h48.pooled_unary_width),
                "pooled_pair_width": int(h48.pooled_pair_width),
                "center_context_width": int(h48.center_context_width),
            },
            {
                "name": "H64",
                "h_dim": 64,
                "q_dim": 16,
                "total_params": total64,
                "head_params": head64,
                "backbone_params": total64 - head64,
                "unified_graph_width": int(h64.unified_graph_width),
                "pooled_unary_width": int(h64.pooled_unary_width),
                "pooled_pair_width": int(h64.pooled_pair_width),
                "center_context_width": int(h64.center_context_width),
            },
        ],
        "delta_total_params": int(total64) - int(total48),
        "delta_head_params": int(head64) - int(head48),
        "changed_tensors": changed,
        "changed_key_set": sorted({row["key"] for row in changed}),
        "expected_h_dependent_keys": sorted(H_DEPENDENT_KEYS),
        "matches_expected_h64_total": int(total64) == EXPECTED_H64_TOTAL,
        "matches_expected_h64_head": int(head64) == EXPECTED_H64_HEAD,
        "matches_expected_h64_width": int(h64.unified_graph_width) == EXPECTED_H64_WIDTH,
        "wide_candidates": wide,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------


def _refresh_probe(model: torch.nn.Module, batch: Any) -> dict[str, Any]:
    """Perturb the (zero-init) centre update and prove q^(1) is recomputed."""
    model.eval()
    with torch.no_grad():
        # centre_update[-1] is zero-initialised at init; add a fixed perturbation
        # so h^(1) != h^(0) and the shared pair encoder must be re-run.
        model.center_update[-1].weight.add_(0.1)
        model.center_update[-1].bias.add_(0.05)
    model.capture_diagnostics = True
    with torch.no_grad():
        model.encode(batch)
    q0, q1 = model.last_q0, model.last_q1
    h0, h1, h2 = model.last_h0, model.last_h1, model.last_h2
    out = {
        "q1_minus_q0_max_abs": float((q1 - q0).abs().max()),
        "q1_q0_cosine": float(
            F.cosine_similarity(q0.reshape(1, -1), q1.reshape(1, -1)).item()
        ),
        "h1_minus_h0_max_abs": float((h1 - h0).abs().max()),
        "h2_minus_h1_max_abs": float((h2 - h1).abs().max()),
        "h1_minus_h0_positive": bool(float((h1 - h0).abs().max()) > 0.0),
        "h2_minus_h1_positive": bool(float((h2 - h1).abs().max()) > 0.0),
        "q1_minus_q0_positive": bool(float((q1 - q0).abs().max()) > 0.0),
    }
    model.capture_diagnostics = False
    return out


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)

    # 1) the H48 capacity builder is still bit-identical to the frozen builder.
    frozen = rec.build_recurrent(0).to(device).eval()
    construction_q16 = cap.build_q16(0).to(device).eval()
    frozen_hash = cap._state_hash(frozen.state_dict())
    q16_hash = cap._state_hash(construction_q16.state_dict())
    with torch.no_grad():
        out_frozen = frozen.encode(batch)
        out_q16 = construction_q16.encode(batch)
    q16_max_diff = float((out_frozen - out_q16).abs().max())

    # 2) only h-dependent tensor shapes change.
    h48 = cap.build_q16(0)
    h64 = build_h64(0)
    s48, s64 = h48.state_dict(), h64.state_dict()
    changed_keys = {k for k in s48 if s48[k].shape != s64[k].shape}
    # The small raw head is re-drawn under ``manual_seed(head_seed)`` after the
    # backbone; its first Linear consumes a width-dependent amount of RNG, so
    # same-shape *head* tensors legitimately differ.  Every same-shape
    # *backbone* tensor must stay bit-identical to H48.
    backbone_same_shape_equal = all(
        torch.equal(s48[k], s64[k])
        for k in s48
        if k not in changed_keys and not k.startswith("head.")
    )
    head_same_shape_diffs = [
        k
        for k in s48
        if k not in changed_keys
        and k.startswith("head.")
        and not torch.equal(s48[k], s64[k])
    ]

    # 3) architecture facts that must not move.
    model = build_h64(0).to(device)
    head_in_features = int(model.head.net[0].in_features)
    counts = model.module_call_counts(batch)
    refresh_probe = _refresh_probe(build_h64(0).to(device), batch)

    # 4) deterministic rebuild.
    rebuild_hash = cap._state_hash(build_h64(0).state_dict())
    build_hash = cap._state_hash(h64.state_dict())

    # 5) finite forward / backward on the first 128 valid molecules.
    model.train()
    out = model(batch)
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    fwd_finite = bool(torch.isfinite(out).all())
    grad_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    grad_nonzero = any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parameters()
    )
    model.zero_grad(set_to_none=True)

    checks = {
        "h48_builder_bit_identical_to_frozen": bool(frozen_hash == q16_hash)
        and q16_max_diff == 0.0,
        "h64_only_h_dependent_shapes_change": changed_keys == set(H_DEPENDENT_KEYS),
        "h64_backbone_same_shape_tensors_bit_identical_to_h48": bool(
            backbone_same_shape_equal
        ),
        "q_dim_still_16": int(model.pair_hidden) == 16
        and int(model.pair_encoder.layers[-2].out_features) == 16,
        "two_round_weight_tying": counts
        == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2},
        "recurrence_flags": int(model.recurrence_rounds) == 2
        and bool(model.recurrence_enabled)
        and str(model.recurrence_mode) == "refresh",
        "relation_refresh_recomputes_q1": refresh_probe["q1_minus_q0_positive"]
        and refresh_probe["h1_minus_h0_positive"]
        and refresh_probe["h2_minus_h1_positive"],
        "readout_width_updated": int(model.unified_graph_width) == EXPECTED_H64_WIDTH
        and int(h64.pooled_unary_width) == 129
        and int(h64.pooled_pair_width) == 33
        and head_in_features == EXPECTED_H64_WIDTH,
        "no_other_hidden_dim_widened": int(model.patch_hidden) == 64
        and int(h64.center_context_width) == 165
        and int(model.center_update[0].out_features) == 60,
        "param_count_exact": _n_params(model) == EXPECTED_H64_TOTAL
        and _n_params(model.head) == EXPECTED_H64_HEAD,
        "deterministic_rebuild": build_hash == rebuild_hash,
        "forward_finite": fwd_finite,
        "backward_finite_nonzero": bool(grad_finite) and bool(grad_nonzero),
        "h48_checkpoints_untouched": H48_REF_SOUP[0] == 0.13707755148684372
        and H48_REF_SOUP[1] == 0.13220956423232566,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "q16_state_hash_equal": bool(frozen_hash == q16_hash),
        "q16_output_max_diff": q16_max_diff,
        "changed_keys": sorted(changed_keys),
        "expected_changed_keys": sorted(H_DEPENDENT_KEYS),
        "head_same_shape_differing_keys": sorted(head_same_shape_diffs),
        "head_reinit_note": (
            "head.net.0 changes input width 302->334; the head is re-drawn from "
            "the same head_seed stream, so later same-shape head tensors differ."
        ),
        "refresh_probe": refresh_probe,
        "module_call_counts": counts,
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"hwidth sanity checks failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training (canonical protocol, snapshots for the fixed Top-5 soup)
# ---------------------------------------------------------------------------


def train(seed: int) -> dict[str, Any]:
    train_data, valid_data, _ = rec.load_encoded()
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    original_protocol = shead.OPTIMIZED_PROTOCOL
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    snapshot_dir = SNAPSHOT_DIR / f"{TAG}_seed{seed}"
    rss_before = _rss_peak_kb()
    started = time.perf_counter()
    try:
        summary = shead.train_model(
            build_fn=build_h64,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=TAG,
            save_state=True,
            real_batch_identity=False,
            expected_total=int(EXPECTED_H64_TOTAL),
            snapshot_dir=snapshot_dir,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
        shead.OPTIMIZED_PROTOCOL = original_protocol
    elapsed = float(time.perf_counter() - started)

    summary = dict(summary)
    summary["config_name"] = "H64"
    summary["h_dim"] = 64
    summary["q_dim"] = 16
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_rss_kb"] = _rss_peak_kb()
    summary["peak_rss_delta_kb"] = _rss_peak_kb() - int(rss_before)
    summary["outer_wall_clock_s"] = elapsed
    _write_json(RUNS_DIR / f"{TAG}_seed{seed}.json", summary)
    return summary


# ---------------------------------------------------------------------------
# fixed Top-5 checkpoint soup (repo-locked rule, no k / weight search)
# ---------------------------------------------------------------------------


def _run(seed: int) -> dict[str, Any]:
    return _read_json(RUNS_DIR / f"{TAG}_seed{seed}.json")


def _valid_loader():
    _train_data, valid_data, _ = rec.load_encoded()
    return zpp._make_loader(list(valid_data), 128, False, 0)


def soup(seed: int) -> dict[str, Any]:
    summary = _run(seed)
    top = vd._top5_epochs(summary)
    soup_state = vd.build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{TAG}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    loader = _valid_loader()
    model = build_h64(seed)
    selection_path = STATE_DIR / f"{TAG}_seed{seed}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model, torch.load(selection_path, map_location="cpu", weights_only=True), loader
    )
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    best_mae = vd._mae(targets, best_preds)
    soup_mae = vd._mae(targets, soup_preds)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": TAG,
        "seed": int(seed),
        "soup_rule": {
            "K": vd.SOUP_K,
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "arithmetic mean",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": best_mae,
        "top5_soup_valid_mae": soup_mae,
        "soup_improvement_over_best": best_mae - soup_mae,
        "soup_state_path": str(soup_path),
        "parameters": int(sum(p.numel() for p in soup_state.values())),
        "official_test_loaded": False,
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
        "valid_targets": targets.tolist(),
    }
    _write_json(RESULTS_DIR / f"soup_{TAG}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# additional diagnostics
# ---------------------------------------------------------------------------


def _centre_state_stats(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().double().cpu().numpy()
    per_dim_std = array.std(axis=0)
    variance = per_dim_std**2
    denominator = float(np.square(variance).sum())
    participation = (
        float((variance.sum() ** 2) / (denominator + EPS)) if denominator > 0 else 0.0
    )
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "abs_mean": float(np.abs(array).mean()),
        "dead_dim_fraction": float((per_dim_std < 1.0e-6).mean()),
        "effective_rank_participation": participation,
    }


def _centre_diagnostics(model: torch.nn.Module, batch: Any) -> dict[str, Any]:
    model.eval()
    model.capture_diagnostics = True
    with torch.no_grad():
        model.encode(batch)
    payload = {
        "h0": _centre_state_stats(model.last_h0),
        "h1": _centre_state_stats(model.last_h1),
        "h2": _centre_state_stats(model.last_h2),
        "q1": _centre_state_stats(model.last_q1),
    }
    model.capture_diagnostics = False
    return payload


def diagnostics() -> dict[str, Any]:
    batch = _first_batch(list(rec.load_encoded()[1]), torch.device("cpu"))

    h64_best: dict[int, np.ndarray] = {}
    h64_soup: dict[int, np.ndarray] = {}
    centre: dict[int, Any] = {}
    raw: dict[int, float] = {}
    for seed in SEEDS:
        soup_payload = soup(seed)
        h64_best[seed] = np.asarray(soup_payload["best_predictions"])
        h64_soup[seed] = np.asarray(soup_payload["soup_predictions"])
        raw[seed] = float(soup_payload["best_checkpoint_valid_mae"])
        model = build_h64(seed)
        model.load_state_dict(
            torch.load(
                STATE_DIR / f"{TAG}_seed{seed}_selection_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        centre[seed] = _centre_diagnostics(model, batch)

    h48_best = {
        s: np.asarray(
            _read_json(vd.RESULTS_DIR / f"soup_canonical_seed{s}.json")[
                "best_predictions"
            ]
        )
        for s in SEEDS
    }
    h48_soup = {
        s: np.asarray(
            _read_json(vd.RESULTS_DIR / f"soup_canonical_seed{s}.json")[
                "soup_predictions"
            ]
        )
        for s in SEEDS
    }
    targets = np.asarray(
        _read_json(RESULTS_DIR / f"soup_{TAG}_seed0.json")["valid_targets"]
    )

    def disagreement(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    h64_raw_vals = np.array([raw[s] for s in SEEDS])
    h64_soup_vals = np.array(
        [
            float(_read_json(RESULTS_DIR / f"soup_{TAG}_seed{s}.json")["top5_soup_valid_mae"])
            for s in SEEDS
        ]
    )
    h48_raw_vals = np.array([H48_REF_RAW[s] for s in SEEDS])
    h48_soup_vals = np.array([H48_REF_SOUP[s] for s in SEEDS])
    best_epochs_h64 = np.array([int(_run(s)["best_epoch"]) for s in SEEDS])
    best_epochs_h48 = np.array(
        [
            int(_read_json(vd.RESULTS_DIR / f"soup_canonical_seed{s}.json")["best_epoch"])
            for s in SEEDS
        ]
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "h64": {
            "raw_best_valid_mae": {str(s): float(raw[s]) for s in SEEDS},
            "raw_2seed_mean": float(h64_raw_vals.mean()),
            "raw_spread": float(h64_raw_vals.max() - h64_raw_vals.min()),
            "best_epoch": {str(s): int(best_epochs_h64[i]) for i, s in enumerate(SEEDS)},
            "soup_valid_mae": {str(s): float(h64_soup_vals[i]) for i, s in enumerate(SEEDS)},
            "soup_2seed_mean": float(h64_soup_vals.mean()),
            "soup_spread": float(h64_soup_vals.max() - h64_soup_vals.min()),
            "seed_prediction_disagreement_raw": disagreement(h64_best[0], h64_best[1]),
            "seed_prediction_disagreement_soup": disagreement(h64_soup[0], h64_soup[1]),
            "centre_state": {str(s): centre[s] for s in SEEDS},
        },
        "h48_reference": {
            "raw_best_valid_mae": {str(s): float(H48_REF_RAW[s]) for s in SEEDS},
            "raw_2seed_mean": float(h48_raw_vals.mean()),
            "raw_spread": float(h48_raw_vals.max() - h48_raw_vals.min()),
            "best_epoch": {str(s): int(best_epochs_h48[i]) for i, s in enumerate(SEEDS)},
            "soup_valid_mae": {str(s): float(H48_REF_SOUP[s]) for s in SEEDS},
            "soup_2seed_mean": float(h48_soup_vals.mean()),
            "soup_spread": float(h48_soup_vals.max() - h48_soup_vals.min()),
            "seed_prediction_disagreement_raw": disagreement(h48_best[0], h48_best[1]),
            "seed_prediction_disagreement_soup": disagreement(h48_soup[0], h48_soup[1]),
        },
        "cross_arch_same_seed_disagreement_raw": {
            str(s): disagreement(h48_best[s], h64_best[s]) for s in SEEDS
        },
        "cross_arch_same_seed_disagreement_soup": {
            str(s): disagreement(h48_soup[s], h64_soup[s]) for s in SEEDS
        },
        "valid_targets_length": int(targets.shape[0]),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide() -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        payload = _read_json(RESULTS_DIR / f"soup_{TAG}_seed{seed}.json")
        soup_mae = float(payload["top5_soup_valid_mae"])
        raw_mae = float(payload["best_checkpoint_valid_mae"])
        rows.append(
            {
                "seed": int(seed),
                "raw_best_valid_mae": raw_mae,
                "raw_improvement_vs_h48": float(H48_REF_RAW[seed] - raw_mae),
                "soup_valid_mae": soup_mae,
                "soup_improvement_vs_h48": float(H48_REF_SOUP[seed] - soup_mae),
                "best_epoch": int(payload["best_epoch"]),
            }
        )
    soup_mean = float(np.mean([row["soup_valid_mae"] for row in rows]))
    raw_mean = float(np.mean([row["raw_best_valid_mae"] for row in rows]))
    soup_improvement = float(H48_REF_SOUP_MEAN - soup_mean)
    raw_improvement = float(np.mean(list(H48_REF_RAW.values())) - raw_mean)
    per_seed_soup = [row["soup_improvement_vs_h48"] for row in rows]
    same_direction = all(x > 0.0 for x in per_seed_soup) or all(
        x < 0.0 for x in per_seed_soup
    )
    sign_flip = bool(
        (per_seed_soup[0] > SIGN_FLIP_MAG and per_seed_soup[1] < -SIGN_FLIP_MAG)
        or (per_seed_soup[1] > SIGN_FLIP_MAG and per_seed_soup[0] < -SIGN_FLIP_MAG)
    )
    if sign_flip:
        verdict = "SEED_DEPENDENT_INCONCLUSIVE"
        decision = "seed-dependent / inconclusive"
    elif soup_improvement >= STRONG_GATE and same_direction and min(per_seed_soup) > 0.0:
        verdict = "STRONG_POSITIVE"
        decision = "centre-state capacity-limited"
    elif soup_improvement >= WEAK_GATE and same_direction and min(per_seed_soup) > 0.0:
        verdict = "WEAK_POSITIVE"
        decision = "weak centre-width signal"
    elif soup_improvement <= WEAK_GATE:
        verdict = "NO_SIGNAL"
        decision = "no centre-width scaling"
    else:
        verdict = "INCONCLUSIVE"
        decision = "no centre-width scaling"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_h48_soup_2seed_mean": H48_REF_SOUP_MEAN,
        "h64_raw_2seed_mean": raw_mean,
        "h64_soup_2seed_mean": soup_mean,
        "soup_improvement_vs_h48": soup_improvement,
        "raw_improvement_vs_h48": raw_improvement,
        "per_seed": rows,
        "strong_gate": STRONG_GATE,
        "weak_gate": WEAK_GATE,
        "both_seeds_same_direction": bool(same_direction),
        "sign_flip": sign_flip,
        "verdict": verdict,
        "decision": decision,
        "next_step": (
            "test one wider centre width near ~100k-110k params (h=96 => 103,219)"
            if verdict == "STRONG_POSITIVE"
            else (
                "buy seed2 replication before expanding h"
                if verdict == "WEAK_POSITIVE"
                else "stop; close plain h-width scaling"
            )
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_accounting": _read_json(RESULTS_DIR / "parameter_accounting.json"),
        "sanity": _read_json(RESULTS_DIR / "sanity.json")["checks"],
        "decision": (
            _read_json(RESULTS_DIR / "decision.json")
            if (RESULTS_DIR / "decision.json").exists()
            else None
        ),
        "runs": {},
        "soups": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        run_path = RUNS_DIR / f"{TAG}_seed{seed}.json"
        if run_path.exists():
            summary = _read_json(run_path)
            payload["runs"][str(seed)] = {
                "raw_best_valid_mae": float(summary["best_valid_mae"]),
                "best_epoch": int(summary["best_epoch"]),
                "epochs_run": int(summary["epochs_run"]),
                "train_mae_at_best": float(summary["train_loss_at_best"]),
                "train_valid_gap": float(summary["best_valid_mae"])
                - float(summary["train_loss_at_best"]),
                "epoch_time_s": summary.get("epoch_time_s"),
                "wall_clock_s": float(summary["wall_clock_s"]),
                "peak_rss_kb": summary.get("peak_rss_kb"),
                "early_stopped": bool(summary["early_stopped"]),
                "horizon_boundary_warning": bool(summary["horizon_boundary_warning"]),
            }
        soup_path = RESULTS_DIR / f"soup_{TAG}_seed{seed}.json"
        if soup_path.exists():
            soup_payload = _read_json(soup_path)
            payload["soups"][str(seed)] = {
                "top5_soup_valid_mae": float(soup_payload["top5_soup_valid_mae"]),
                "best_checkpoint_valid_mae": float(
                    soup_payload["best_checkpoint_valid_mae"]
                ),
                "top5_epochs": soup_payload["top5_epochs"],
                "parameters": int(soup_payload["parameters"]),
            }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


def all_stages() -> None:
    params()
    sanity()
    for seed in SEEDS:
        train(seed)
        soup(seed)
    diagnostics()
    decide()
    report()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "params",
            "sanity",
            "train",
            "soup",
            "diagnostics",
            "decide",
            "report",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if args.stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    if args.stage == "train":
        train(args.seed)
    if args.stage == "soup":
        print(json.dumps(soup(args.seed), indent=2, default=str), flush=True)
    if args.stage == "diagnostics":
        print(json.dumps(diagnostics(), indent=2, default=str), flush=True)
    if args.stage == "decide":
        print(json.dumps(decide(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    if args.stage == "all":
        all_stages()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
