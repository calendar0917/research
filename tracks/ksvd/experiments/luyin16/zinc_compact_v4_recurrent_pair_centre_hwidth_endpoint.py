"""Final centre-width endpoint: H64 -> H96 at fixed ``q_dim=16``, ``T=2``.

Two strict phases:

* **Phase A--D (validation only).**  Train the H96 candidate (``h=96, q=16``)
  under the canonical protocol, compute the fixed Top-5 checkpoint soup, and
  run the pre-registered architecture-freeze gate against H64.
* **Phase E (official test, exactly once).**  Only after an architecture is
  frozen and written to ``architecture_freeze.json`` may the official test
  split be loaded.  The frozen architecture's pre-existing selection
  checkpoints / soups are then evaluated once; test labels never influence the
  H64-vs-H96 choice, checkpoint selection, or soup construction.

The frozen H64 reference lives in
``results/compact_v4_recurrent_pair_centre_hwidth`` (read-only here).

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_hwidth_endpoint <stage> [--seed 0]

Stages: ``params sanity train soup diagnostics decide freeze test report all_validation``.
"""

from __future__ import annotations

import argparse
import platform
import time
from typing import Any, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_hwidth as hw,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_training_sufficiency as ztraining,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_hwidth_endpoint"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SOUP_DIR = RESULTS_DIR / "soup_states"
H64_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_hwidth"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_hwidth_endpoint_v1"

H96 = {"h": 96, "q": 16}
TAG = "h96"
SEEDS = (0, 1)

EXPECTED_H96_TOTAL = 103219
EXPECTED_H96_HEAD = 5383
EXPECTED_H96_WIDTH = 398

# Frozen H64 validation reference (from the H64 endpoint run).
H64_TOTAL = 85763
H64_SOUP = {0: 0.12984323308611057, 1: 0.131813230478263}
H64_SOUP_MEAN = 0.13082823178218678
H64_RAW = {0: 0.13423950145492564, 1: 0.1352006213227869}
H64_RAW_MEAN = 0.13472006138885628

# Historical H48 official-test raw mean (selection checkpoints, 2 seeds), from
# results/compact_v4_recurrent_pair_centre/terminal_test.json.  Reference only.
H48_TEST_RAW_MEAN = 0.11495107560744508

STRONG_GATE = 0.002
WEAK_GATE = 0.001
SIGN_FLIP_MAG = 0.0005

# Tensor shapes that change H64 -> H96.  Beyond the direct ``patch_hidden``
# consumers (patch encoder output, pair projection, centre update, head input),
# H96 crosses the existing derived-width floors and therefore also grows the
# patch-encoder hidden ``max(patch_hidden, 64)`` (64 -> 96) and the
# global-encoder hidden ``max(patch_hidden // 2, 32)`` (32 -> 48).  These are
# the frozen architecture's own ``patch_hidden`` dependencies -- no new module.
H96_DEPENDENT_KEYS = frozenset(
    {
        "patch_encoder.layers.0.weight",
        "patch_encoder.layers.0.bias",
        "patch_encoder.layers.1.weight",
        "patch_encoder.layers.1.bias",
        "patch_encoder.layers.4.weight",
        "patch_encoder.layers.4.bias",
        "global_encoder.layers.0.weight",
        "global_encoder.layers.0.bias",
        "global_encoder.layers.1.weight",
        "global_encoder.layers.1.bias",
        "global_encoder.layers.4.weight",
        "pair_projection.weight",
        "center_update.0.weight",
        "center_update.4.weight",
        "center_update.4.bias",
        "head.net.0.weight",
    }
)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_h96(seed: int = 0) -> torch.nn.Module:
    return cap.build_capacity(int(seed), h=int(H96["h"]), q=int(H96["q"]))


BUILDERS = {"H64": hw.build_h64, "H96": build_h96}
TAGS = {"H64": "h64", "H96": "h96"}


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    h64 = hw.build_h64(0)
    h96 = build_h96(0)
    s64 = {k: v.detach().clone() for k, v in h64.state_dict().items()}
    s96 = {k: v.detach().clone() for k, v in h96.state_dict().items()}

    changed = []
    for key in sorted(s64):
        a, b = s64[key], s96[key]
        if a.shape != b.shape or a.numel() != b.numel():
            changed.append(
                {
                    "key": key,
                    "h64_shape": list(a.shape),
                    "h96_shape": list(b.shape),
                    "h64_numel": int(a.numel()),
                    "h96_numel": int(b.numel()),
                    "delta_numel": int(b.numel()) - int(a.numel()),
                }
            )

    total64 = hw._n_params(h64)
    total96 = hw._n_params(h96)
    head64 = hw._n_params(h64.head)
    head96 = hw._n_params(h96.head)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": [
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
            {
                "name": "H96",
                "h_dim": 96,
                "q_dim": 16,
                "total_params": total96,
                "head_params": head96,
                "backbone_params": total96 - head96,
                "unified_graph_width": int(h96.unified_graph_width),
                "pooled_unary_width": int(h96.pooled_unary_width),
                "pooled_pair_width": int(h96.pooled_pair_width),
                "center_context_width": int(h96.center_context_width),
            },
        ],
        "delta_total_params": int(total96) - int(total64),
        "delta_head_params": int(head96) - int(head64),
        "changed_tensors": changed,
        "changed_key_set": sorted({row["key"] for row in changed}),
        "expected_h_dependent_keys": sorted(H96_DEPENDENT_KEYS),
        "derived_widths": {
            "patch_encoder_hidden": max(96, 64),
            "global_encoder_hidden": max(96 // 2, 32),
            "center_update_hidden": int(h96.center_context_hidden),
            "relation_encoder_hidden": max(16, 32),
            "pair_encoder_hidden": max(2 * 16, 64),
        },
        "note": (
            "H96 crosses the existing max(h,64) / max(h//2,32) floors, so the "
            "patch/global encoder hidden widths grow naturally with patch_hidden."
        ),
        "matches_expected_h96_total": int(total96) == EXPECTED_H96_TOTAL,
        "matches_expected_h96_head": int(head96) == EXPECTED_H96_HEAD,
        "matches_expected_h96_width": int(h96.unified_graph_width) == EXPECTED_H96_WIDTH,
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# sanity
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = hw._first_batch(valid_data, device)

    # H48 code path still frozen / bit-identical.
    frozen = rec.build_recurrent(0).to(device).eval()
    q16 = cap.build_q16(0).to(device).eval()
    frozen_hash = cap._state_hash(frozen.state_dict())
    q16_hash = cap._state_hash(q16.state_dict())
    with torch.no_grad():
        q16_max_diff = float((frozen.encode(batch) - q16.encode(batch)).abs().max())

    # H64 builder / assets unchanged, and H96 only grows h-dependent tensors.
    h64 = hw.build_h64(0)
    h64_assets_match = hw._n_params(h64) == H64_TOTAL and all(
        abs(
            float(
                hw._read_json(H64_DIR / f"soup_h64_seed{s}.json")[
                    "top5_soup_valid_mae"
                ]
            )
            - H64_SOUP[s]
        )
        < 1.0e-12
        for s in SEEDS
    )
    h96 = build_h96(0)
    s64, s96 = h64.state_dict(), h96.state_dict()
    changed_keys = {k for k in s64 if s64[k].shape != s96[k].shape}
    backbone_same_shape_equal = all(
        torch.equal(s64[k], s96[k])
        for k in s64
        if k not in changed_keys and not k.startswith("head.")
    )

    model = build_h96(0).to(device)
    counts = model.module_call_counts(batch)
    refresh_probe = hw._refresh_probe(build_h96(0).to(device), batch)

    model.train()
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
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
        "h64_builder_and_reference_assets_unchanged": bool(h64_assets_match),
        "h96_only_h_dependent_shapes_change": changed_keys == set(H96_DEPENDENT_KEYS),
        "h96_backbone_same_shape_bit_identical_to_h64": bool(backbone_same_shape_equal),
        "q_dim_still_16": int(model.pair_hidden) == 16
        and int(model.pair_encoder.layers[-2].out_features) == 16,
        "two_round_weight_tying": counts
        == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2},
        "recurrence_flags_refresh": int(model.recurrence_rounds) == 2
        and bool(model.recurrence_enabled)
        and str(model.recurrence_mode) == "refresh",
        "relation_refresh_recomputes_q1": refresh_probe["q1_minus_q0_positive"]
        and refresh_probe["h1_minus_h0_positive"]
        and refresh_probe["h2_minus_h1_positive"],
        "readout_width_updated": int(model.unified_graph_width) == EXPECTED_H96_WIDTH
        and int(h96.pooled_unary_width) == 193
        and int(h96.pooled_pair_width) == 33
        and int(model.head.net[0].in_features) == EXPECTED_H96_WIDTH,
        "no_other_hidden_dim_widened": int(model.patch_hidden) == 96
        and int(h96.center_context_width) == 165
        and int(model.center_update[0].out_features) == 60,
        "param_count_exact": hw._n_params(model) == EXPECTED_H96_TOTAL
        and hw._n_params(model.head) == EXPECTED_H96_HEAD,
        "forward_finite": bool(torch.isfinite(out).all()),
        "backward_finite_nonzero": bool(grad_finite) and bool(grad_nonzero),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "q16_state_hash_equal": bool(frozen_hash == q16_hash),
        "q16_output_max_diff": q16_max_diff,
        "h64_total_params": hw._n_params(h64),
        "changed_keys": sorted(changed_keys),
        "refresh_probe": refresh_probe,
        "module_call_counts": counts,
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"hwidth endpoint sanity checks failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training / soup (validation only)
# ---------------------------------------------------------------------------


def train(seed: int) -> dict[str, Any]:
    train_data, valid_data, _ = rec.load_encoded()
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    original_protocol = shead.OPTIMIZED_PROTOCOL
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    snapshot_dir = SNAPSHOT_DIR / f"{TAG}_seed{seed}"
    rss_before = hw._rss_peak_kb()
    started = time.perf_counter()
    try:
        summary = shead.train_model(
            build_fn=build_h96,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=TAG,
            save_state=True,
            real_batch_identity=False,
            expected_total=int(EXPECTED_H96_TOTAL),
            snapshot_dir=snapshot_dir,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
        shead.OPTIMIZED_PROTOCOL = original_protocol
    elapsed = float(time.perf_counter() - started)

    summary = dict(summary)
    summary["config_name"] = "H96"
    summary["h_dim"] = 96
    summary["q_dim"] = 16
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_rss_kb"] = hw._rss_peak_kb()
    summary["peak_rss_delta_kb"] = hw._rss_peak_kb() - int(rss_before)
    summary["outer_wall_clock_s"] = elapsed
    hw._write_json(RUNS_DIR / f"{TAG}_seed{seed}.json", summary)
    return summary


def _valid_loader():
    _train_data, valid_data, _ = rec.load_encoded()
    return zpp._make_loader(list(valid_data), 128, False, 0)


def soup(seed: int) -> dict[str, Any]:
    summary = hw._read_json(RUNS_DIR / f"{TAG}_seed{seed}.json")
    top = vd._top5_epochs(summary)
    soup_state = vd.build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{TAG}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    loader = _valid_loader()
    model = build_h96(seed)
    selection_path = STATE_DIR / f"{TAG}_seed{seed}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model, torch.load(selection_path, map_location="cpu", weights_only=True), loader
    )
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": TAG,
        "seed": int(seed),
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": vd._mae(targets, best_preds),
        "top5_soup_valid_mae": vd._mae(targets, soup_preds),
        "soup_improvement_over_best": vd._mae(targets, best_preds)
        - vd._mae(targets, soup_preds),
        "soup_state_path": str(soup_path),
        "parameters": int(sum(p.numel() for p in soup_state.values())),
        "official_test_loaded": False,
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
        "valid_targets": targets.tolist(),
    }
    hw._write_json(RESULTS_DIR / f"soup_{TAG}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics() -> dict[str, Any]:
    batch = hw._first_batch(list(rec.load_encoded()[1]), torch.device("cpu"))
    h96_best, h96_soup, centre = {}, {}, {}
    for seed in SEEDS:
        payload = soup(seed)
        h96_best[seed] = np.asarray(payload["best_predictions"])
        h96_soup[seed] = np.asarray(payload["soup_predictions"])
        model = build_h96(seed)
        model.load_state_dict(
            torch.load(
                STATE_DIR / f"{TAG}_seed{seed}_selection_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        centre[seed] = hw._centre_diagnostics(model, batch)

    h64_best = {
        s: np.asarray(
            hw._read_json(H64_DIR / f"soup_h64_seed{s}.json")["best_predictions"]
        )
        for s in SEEDS
    }
    h64_soup = {
        s: np.asarray(
            hw._read_json(H64_DIR / f"soup_h64_seed{s}.json")["soup_predictions"]
        )
        for s in SEEDS
    }

    def disagree(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    h96_raw = np.array([float(soup(s)["best_checkpoint_valid_mae"]) for s in SEEDS])
    h96_soup_vals = np.array(
        [float(soup(s)["top5_soup_valid_mae"]) for s in SEEDS]
    )
    h64_raw = np.array([H64_RAW[s] for s in SEEDS])
    h64_soup_vals = np.array([H64_SOUP[s] for s in SEEDS])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "h96": {
            "raw_best_valid_mae": {str(s): float(h96_raw[i]) for i, s in enumerate(SEEDS)},
            "raw_2seed_mean": float(h96_raw.mean()),
            "raw_spread": float(h96_raw.max() - h96_raw.min()),
            "best_epoch": {
                str(s): int(hw._read_json(RUNS_DIR / f"{TAG}_seed{s}.json")["best_epoch"])
                for s in SEEDS
            },
            "soup_valid_mae": {
                str(s): float(h96_soup_vals[i]) for i, s in enumerate(SEEDS)
            },
            "soup_2seed_mean": float(h96_soup_vals.mean()),
            "soup_spread": float(h96_soup_vals.max() - h96_soup_vals.min()),
            "seed_prediction_disagreement_raw": disagree(h96_best[0], h96_best[1]),
            "seed_prediction_disagreement_soup": disagree(h96_soup[0], h96_soup[1]),
            "centre_state": {str(s): centre[s] for s in SEEDS},
        },
        "h64_reference": {
            "raw_best_valid_mae": {str(s): float(H64_RAW[s]) for s in SEEDS},
            "raw_2seed_mean": float(h64_raw.mean()),
            "soup_valid_mae": {str(s): float(H64_SOUP[s]) for s in SEEDS},
            "soup_2seed_mean": float(h64_soup_vals.mean()),
            "seed_prediction_disagreement_raw": disagree(h64_best[0], h64_best[1]),
            "seed_prediction_disagreement_soup": disagree(h64_soup[0], h64_soup[1]),
        },
        "cross_arch_same_seed_disagreement_raw": {
            str(s): disagree(h64_best[s], h96_best[s]) for s in SEEDS
        },
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase D -- architecture-freeze gate
# ---------------------------------------------------------------------------


def decide() -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        payload = soup(seed)
        soup_mae = float(payload["top5_soup_valid_mae"])
        raw_mae = float(payload["best_checkpoint_valid_mae"])
        rows.append(
            {
                "seed": int(seed),
                "raw_best_valid_mae": raw_mae,
                "raw_improvement_vs_h64": float(H64_RAW[seed] - raw_mae),
                "soup_valid_mae": soup_mae,
                "soup_improvement_vs_h64": float(H64_SOUP[seed] - soup_mae),
                "best_epoch": int(payload["best_epoch"]),
            }
        )
    soup_mean = float(np.mean([row["soup_valid_mae"] for row in rows]))
    raw_mean = float(np.mean([row["raw_best_valid_mae"] for row in rows]))
    soup_improvement = float(H64_SOUP_MEAN - soup_mean)
    raw_improvement = float(H64_RAW_MEAN - raw_mean)
    per_seed_soup = [row["soup_improvement_vs_h64"] for row in rows]
    same_direction = all(x > 0.0 for x in per_seed_soup) or all(
        x < 0.0 for x in per_seed_soup
    )
    sign_flip = bool(
        (per_seed_soup[0] > SIGN_FLIP_MAG and per_seed_soup[1] < -SIGN_FLIP_MAG)
        or (per_seed_soup[1] > SIGN_FLIP_MAG and per_seed_soup[0] < -SIGN_FLIP_MAG)
    )
    if sign_flip:
        verdict, selected, reason = (
            "SIGN_FLIP",
            "H64",
            "H96 seeds move in opposite directions -> keep H64",
        )
    elif soup_improvement >= STRONG_GATE and same_direction and min(per_seed_soup) > 0:
        verdict, selected, reason = (
            "CLEAR_CONTINUED_SCALING",
            "H96",
            "H96 soup mean improvement >= +0.002 with both seeds same direction",
        )
    elif soup_improvement >= WEAK_GATE and same_direction and min(per_seed_soup) > 0:
        verdict, selected, reason = (
            "WEAK_SCALING",
            "UNDECIDED",
            "H96 improvement in [0.001, 0.002); seed2 replication required",
        )
    elif abs(soup_improvement) < WEAK_GATE:
        verdict, selected, reason = (
            "SATURATION",
            "H64",
            "|H96 - H64| < 0.001 -> prefer the smaller/efficient H64",
        )
    else:
        verdict, selected, reason = (
            "DEGRADATION",
            "H64",
            "H96 does not beat H64 -> keep H64",
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_h64_soup_2seed_mean": H64_SOUP_MEAN,
        "h96_soup_2seed_mean": soup_mean,
        "h96_raw_2seed_mean": raw_mean,
        "soup_improvement_vs_h64": soup_improvement,
        "raw_improvement_vs_h64": raw_improvement,
        "per_seed": rows,
        "strong_gate": STRONG_GATE,
        "weak_gate": WEAK_GATE,
        "both_seeds_same_direction": bool(same_direction),
        "sign_flip": sign_flip,
        "verdict": verdict,
        "selected_architecture": selected,
        "selection_reason": reason,
        "test_unlock_authorised": bool(selected in {"H64", "H96"}),
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def freeze(selected: str | None = None) -> dict[str, Any]:
    """Write the architecture-freeze record.  Refuses to overwrite a test lock."""
    if (RESULTS_DIR / "official_test_unlock.json").exists():
        raise RuntimeError("refusing to change architecture after test unlock")
    decision = hw._read_json(RESULTS_DIR / "decision.json")
    selected = str(selected or decision["selected_architecture"])
    if selected not in {"H64", "H96"}:
        raise RuntimeError(f"cannot freeze unresolved architecture: {selected!r}")
    if selected == "H96":
        params_total = EXPECTED_H96_TOTAL
        validation = {
            "soup_valid_mae": {str(s): decision["per_seed"][i]["soup_valid_mae"] for i, s in enumerate(SEEDS)},
            "soup_2seed_mean": decision["h96_soup_2seed_mean"],
            "raw_valid_mae": {str(s): decision["per_seed"][i]["raw_best_valid_mae"] for i, s in enumerate(SEEDS)},
            "raw_2seed_mean": decision["h96_raw_2seed_mean"],
        }
    else:
        params_total = H64_TOTAL
        validation = {
            "soup_valid_mae": {str(s): H64_SOUP[s] for s in SEEDS},
            "soup_2seed_mean": H64_SOUP_MEAN,
            "raw_valid_mae": {str(s): H64_RAW[s] for s in SEEDS},
            "raw_2seed_mean": H64_RAW_MEAN,
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": selected,
        "selection_reason": decision["selection_reason"],
        "decision_verdict": decision["verdict"],
        "validation_numbers": validation,
        "parameter_count": int(params_total),
        "chosen_seeds": [int(s) for s in SEEDS],
        "checkpoint_rule": "best official-valid MAE selection state",
        "soup_rule": {
            "K": vd.SOUP_K,
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "arithmetic mean",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "h64_reference_soup_2seed_mean": H64_SOUP_MEAN,
        "test_status": "not yet loaded",
        "platform": platform.platform(),
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase E -- one-shot official test
# ---------------------------------------------------------------------------


def _frozen_assets(selected: str, seed: int):
    if selected == "H96":
        builder = build_h96
        state_path = STATE_DIR / f"{TAG}_seed{seed}_selection_state.pt"
        soup_path = SOUP_DIR / f"{TAG}_seed{seed}_top5_soup.pt"
    else:
        builder = hw.build_h64
        state_path = H64_DIR / "states" / f"h64_seed{seed}_selection_state.pt"
        soup_path = H64_DIR / "soup_states" / f"h64_seed{seed}_top5_soup.pt"
    return builder, state_path, soup_path


def test_eval() -> dict[str, Any]:
    """Load the official test split exactly once for the frozen architecture."""
    freeze_path = RESULTS_DIR / "architecture_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("refusing test: architecture_freeze.json missing")
    freeze_record = hw._read_json(freeze_path)
    selected = str(freeze_record["selected_architecture"])
    lock_path = RESULTS_DIR / "official_test_unlock.json"
    if lock_path.exists():
        raise RuntimeError("refusing test: official test already unlocked once")

    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": selected,
        "frozen_before_test": True,
        "encoding": "transforms fit on official train only (identical to selection)",
        "checkpoints": "pre-existing selection states and Top-5 soups",
        "official_test_loaded": True,
        "git_note": "test labels are used for reporting only",
    }
    hw._write_json(lock_path, lock)

    # --- first and only official-test load ---------------------------------
    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_data, test_data, audit = ztraining.build_encoded(
        train_records, test_records, config
    )
    del _train_data
    loader = zpp._make_loader(test_data, 128, False, 0)
    targets = np.asarray(
        [float(graph.y.view(-1)[0]) for graph in test_data], dtype=np.float64
    )

    rows = []
    raw_preds, soup_preds = {}, {}
    for seed in SEEDS:
        builder, state_path, soup_path = _frozen_assets(selected, seed)
        if not state_path.exists() or not soup_path.exists():
            raise RuntimeError(f"missing frozen asset for {selected} seed{seed}")
        model = builder(int(seed)).eval()
        t_raw, raw_pred = vd._predict_state(
            model, torch.load(state_path, map_location="cpu", weights_only=True), loader
        )
        t_soup, soup_pred = vd._predict_state(
            model, torch.load(soup_path, map_location="cpu", weights_only=True), loader
        )
        if not np.array_equal(t_raw, targets) or not np.array_equal(t_soup, targets):
            raise RuntimeError("official-test target order mismatch")
        raw_preds[seed] = raw_pred
        soup_preds[seed] = soup_pred
        rows.append(
            {
                "seed": int(seed),
                "raw_selection_test_mae": float(np.mean(np.abs(targets - raw_pred))),
                "soup_test_mae": float(np.mean(np.abs(targets - soup_pred))),
            }
        )

    raw_values = np.array([row["raw_selection_test_mae"] for row in rows])
    soup_values = np.array([row["soup_test_mae"] for row in rows])

    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        return mean, std

    raw_mean, raw_std = _mean_std(raw_values)
    soup_mean, soup_std = _mean_std(soup_values)
    raw_ensemble = np.mean(np.stack([raw_preds[s] for s in SEEDS], axis=0), axis=0)
    soup_ensemble = np.mean(np.stack([soup_preds[s] for s in SEEDS], axis=0), axis=0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_architecture": selected,
        "parameter_count": int(freeze_record["parameter_count"]),
        "n_test": int(targets.shape[0]),
        "rows": rows,
        "raw_test_mean": raw_mean,
        "raw_test_std": raw_std,
        "soup_test_mean": soup_mean,
        "soup_test_std": soup_std,
        "diagnostic_raw_seed_ensemble_test_mae": float(
            np.mean(np.abs(targets - raw_ensemble))
        ),
        "diagnostic_soup_seed_ensemble_test_mae": float(
            np.mean(np.abs(targets - soup_ensemble))
        ),
        "historical_h48_test_raw_mean": H48_TEST_RAW_MEAN,
        "official_test_loaded": True,
        "audit": {key: audit[key] for key in sorted(audit) if key != "topology"},
    }
    hw._write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# report / orchestration
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return hw._read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "decision": _maybe("decision.json"),
        "architecture_freeze": _maybe("architecture_freeze.json"),
        "official_test_results": _maybe("official_test_results.json"),
        "runs": {
            str(s): {
                key: hw._read_json(RUNS_DIR / f"{TAG}_seed{s}.json")[key]
                for key in (
                    "best_valid_mae",
                    "best_epoch",
                    "epochs_run",
                    "train_loss_at_best",
                    "wall_clock_s",
                    "epoch_time_s",
                    "peak_rss_kb",
                    "early_stopped",
                )
            }
            for s in SEEDS
            if (RUNS_DIR / f"{TAG}_seed{s}.json").exists()
        },
    }
    hw._write_json(RESULTS_DIR / "report.json", payload)
    return payload


def all_validation() -> None:
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
    import json

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
            "freeze",
            "test",
            "report",
            "all_validation",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select", type=str, default=None)
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
    if args.stage == "freeze":
        print(json.dumps(freeze(args.select), indent=2, default=str), flush=True)
    if args.stage == "test":
        print(json.dumps(test_eval(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    if args.stage == "all_validation":
        all_validation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
