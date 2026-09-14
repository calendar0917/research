"""Low-cost diagnostics on the frozen compact-v4-smallhead 2-round recurrent
pair--centre model (``zinc_compact_v4_recurrent_pair_centre``).

Two independent, pre-registered single-change diagnostics:

Task A -- signed relation state
    Remove **only** the final ``ReLU`` of ``pair_encoder.layers`` (the last
    activation applied to ``q``), replacing it with ``nn.Identity``.  Every
    other module, hidden width, relation descriptor, centre update, readout,
    head, optimizer and training protocol is inherited verbatim.  The
    parameter count must stay exactly 82,115.  Before training we audit the
    pre-activation statistics of the *original ReLU* model on a small batch
    (negative fraction, mean/std, post-ReLU zero fraction, q0->q1 sign flips,
    cosine similarity, relative delta norm).

Task B -- T=3 recurrent depth
    Reuse the already parameterised ``recurrence_rounds`` switch with three
    weight-tied rounds:

        h0 -> q0 -> A0 -> h1 -> q1 -> A1 -> h2 -> q2 -> A2 -> h3
        readout = unary_moments(h3) + pair_moments(q2) + global + topology

    ``P = pair_projection``, ``Q = pair_encoder``, ``U = center_update`` are the
    same tensor objects in all three rounds.  Zero new parameters (82,115).

Design constraints (deliberately *not* here)
--------------------------------------------
* No activation sweep (only the single signed-q change), no T-depth sweep
  (only T=3), no pair-to-pair composition, no attention, no path features.
* No change to hidden dims, relation descriptor, centre pooling, readout,
  global/topology branch, small head, optimizer, scheduler or loss.
* Validation only; the official test split is never loaded.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_diagnostics \
        {audit,sanity_signed,sanity_t3,train_signed,train_t3,decision_signed,decision_t3,all}
"""

from __future__ import annotations

import argparse
import json
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

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_diagnostics"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_diagnostics_v1"

EXPECTED_SMALL_TOTAL = 82115
EXPECTED_HEAD = 4135
Q_DIM = rec.Q_DIM
DISTANCE_BUCKETS = zpp.DISTANCE_BUCKETS

# Frozen T=2 recurrent references (must never be overwritten by this module).
T2_RESULTS_DIR = rec.RESULTS_DIR
T2_SEED0_VALID = 0.13837560486892472
T2_SEED1_VALID = 0.1334398  # only used if the reference run file is missing

# Task A continuation gates (per the task brief).
A_STRONG = 0.002
A_WEAK = 0.001
A_NOISE = 0.001

# Task B depth gates (per the task brief).
B_STRONG = 0.003
B_WEAK = 0.001

# T=3 horizon: same protocol, but a longer max-epoch cap so a late-converging
# depth-3 model cannot be a false negative.  Patience is unchanged.
T3_MAX_EPOCHS = 320
T3_PATIENCE = 40

EPS = 1.0e-8


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


def _relu_count(module: nn.Module) -> int:
    return int(sum(1 for item in module.modules() if isinstance(item, nn.ReLU)))


def _first_batch(valid_data: Sequence[Any], device: torch.device) -> Any:
    return next(iter(zpp._make_loader(list(valid_data)[:128], 128, False, 0))).to(device)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def make_signed(model: nn.Module) -> nn.Module:
    """Replace ONLY ``model.pair_encoder.layers[-1]`` (ReLU) with Identity."""
    layers = model.pair_encoder.layers
    if not isinstance(layers[-1], nn.ReLU):
        raise RuntimeError(
            f"pair_encoder.layers[-1] is {type(layers[-1]).__name__}, not ReLU"
        )
    layers[-1] = nn.Identity()
    return model


def build_recurrent_signed(seed: int = 0) -> nn.Module:
    """T=2 recurrent pair--centre, signed q (final pair_encoder ReLU removed)."""
    return make_signed(rec.build_recurrent(int(seed)))


def build_recurrent_t3(seed: int = 0) -> nn.Module:
    """T=3 weight-tied recurrent pair--centre (same tensors, 3 rounds)."""
    return rec.build_recurrent(int(seed), recurrence_rounds=3)


# ---------------------------------------------------------------------------
# Task A -- pre-activation audit on the original ReLU T=2 model
# ---------------------------------------------------------------------------


def _pre_stats(value: torch.Tensor) -> dict[str, float]:
    return {
        "negative_fraction": float((value < 0).float().mean()),
        "zero_fraction_after_relu": float((value <= 0).float().mean()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "abs_mean": float(value.abs().mean()),
    }


def _pair_channel_stats(q0: torch.Tensor, q1: torch.Tensor) -> dict[str, float]:
    sign_flip = float((torch.sign(q0) != torch.sign(q1)).float().mean())
    on_off_flip = float(((q0 > 0) != (q1 > 0)).float().mean())
    cosine = F.cosine_similarity(q0, q1, dim=1, eps=EPS)
    rel_delta = (q1 - q0).norm(dim=1) / (q0.norm(dim=1) + EPS)
    return {
        "sign_flip_fraction": sign_flip,
        "activation_on_off_flip_fraction": on_off_flip,
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
        "relative_delta_norm_mean": float(rel_delta.mean()),
        "relative_delta_norm_median": float(rel_delta.median()),
    }


def audit() -> dict[str, Any]:
    """Record q0/q1 pre-activation statistics of the *original ReLU* T=2 model.

    Uses the frozen trained seed0 selection checkpoint and the first 128
    official-valid molecules.  Captures the input to the final pair ReLU (the
    output of ``pair_encoder.layers[-2]``) for both rounds.
    """
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)

    model = rec.build_recurrent(0).to(device).eval()
    state_path = rec.STATE_DIR / "recurrent_seed0_selection_state.pt"
    if not state_path.exists():
        raise RuntimeError(f"missing reference T=2 checkpoint: {state_path}")
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
        raise RuntimeError(
            f"expected 2 pair-encoder rounds, captured {len(pre)} pre / {len(post)} post"
        )
    q0_pre, q1_pre = pre
    q0_post, q1_post = post
    if int(q0_pre.shape[1]) != Q_DIM:
        raise RuntimeError(f"pre-activation width {q0_pre.shape[1]} != Q_DIM {Q_DIM}")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_checkpoint": str(state_path),
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
        "q0_pre_activation": _pre_stats(q0_pre),
        "q1_pre_activation": _pre_stats(q1_pre),
        "q0_post_relu": _pre_stats(q0_post),
        "q1_post_relu": _pre_stats(q1_post),
        "q0_to_q1": _pair_channel_stats(q0_pre, q1_pre),
        "n_pairs": int(q0_pre.shape[0]),
        "finite": bool(torch.isfinite(out).all() and torch.isfinite(q1_pre).all()),
    }
    _write_json(RESULTS_DIR / "taskA_pre_activation_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Task A -- sanity checks
# ---------------------------------------------------------------------------


def sanity_signed() -> dict[str, Any]:
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)

    baseline = rec.build_baseline(0).to(device).eval()
    original = rec.build_recurrent(0).to(device).eval()
    signed = build_recurrent_signed(0).to(device).eval()

    params = {
        "baseline": _n_params(baseline),
        "recurrent": _n_params(original),
        "signed": _n_params(signed),
    }
    shapes_identical = (
        {k: tuple(v.shape) for k, v in original.state_dict().items()}
        == {k: tuple(v.shape) for k, v in signed.state_dict().items()}
    )
    relu_original = _relu_count(original)
    relu_signed = _relu_count(signed)
    only_final_pair_relu_removed = (
        relu_original - relu_signed == 1
        and _relu_count(original.pair_encoder) == 2
        and _relu_count(signed.pair_encoder) == 1
        and isinstance(signed.pair_encoder.layers[-1], nn.Identity)
        and isinstance(signed.pair_encoder.layers[-2], nn.Linear)
        and int(signed.pair_encoder.layers[-2].out_features) == Q_DIM
        and isinstance(original.pair_encoder.layers[-1], nn.ReLU)
        and isinstance(signed.patch_encoder.layers[-1], nn.ReLU)
        and isinstance(signed.relation_encoder.layers[-1], nn.ReLU)
        and isinstance(signed.global_encoder.layers[-1], nn.ReLU)
        and isinstance(signed.center_update[-1], nn.Linear)
    )

    counts_signed = signed.module_call_counts(batch)
    counts_original = original.module_call_counts(batch)

    # forward / backward must be finite and produce non-zero gradients.
    signed.train()
    out = signed(batch)
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    grad_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in signed.parameters()
    )
    nonzero_grad = any(
        parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0
        for parameter in signed.parameters()
    )
    signed.eval()
    signed.zero_grad(set_to_none=True)
    with torch.no_grad():
        out_eval = signed.encode(batch)
        original_eval = original.encode(batch)
    # The recurrent pathway is structurally intact: the signed model differs
    # from the ReLU model only through the single activation.
    pathway_intact = (
        counts_signed == counts_original == {
            "pair_projection": 4,
            "pair_encoder": 2,
            "center_update": 2,
        }
    )

    checks = {
        "params_signed_equals_recurrent": params["signed"] == params["recurrent"],
        "params_signed_expected_82115": params["signed"] == EXPECTED_SMALL_TOTAL,
        "head_expected_4135": _n_params(signed.head) == EXPECTED_HEAD,
        "state_dict_shapes_identical": bool(shapes_identical),
        "only_final_pair_relu_removed": bool(only_final_pair_relu_removed),
        "recurrent_pathway_call_counts_intact": bool(pathway_intact),
        "forward_finite": bool(torch.isfinite(out_eval).all()),
        "backward_grad_finite": bool(grad_finite),
        "backward_grad_nonzero": bool(nonzero_grad),
        "output_differs_from_relu_recurrent": (
            float((out_eval - original_eval).abs().max()) > 0.0
        ),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parameters": params,
        "relu_counts": {
            "original_total": relu_original,
            "signed_total": relu_signed,
            "original_pair_encoder": _relu_count(original.pair_encoder),
            "signed_pair_encoder": _relu_count(signed.pair_encoder),
        },
        "module_call_counts": {
            "recurrent": counts_original,
            "signed": counts_signed,
        },
        "forward_backward": {
            "loss": float(loss.detach()),
            "max_abs_output_diff_vs_relu": float(
                (out_eval - original_eval).abs().max()
            ),
        },
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity_signed.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"signed-q sanity checks failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# Task B -- sanity checks
# ---------------------------------------------------------------------------


def sanity_t3() -> dict[str, Any]:
    device = torch.device("cpu")
    _train_data, valid_data, _ = rec.load_encoded()
    batch = _first_batch(valid_data, device)

    reference = rec.build_recurrent(0).to(device).eval()
    model = build_recurrent_t3(0).to(device).eval()
    # The inherited centre update is zero-initialised, so at init every round
    # collapses.  Apply a deterministic shared perturbation (same convention as
    # the reference sanity) so the three-round mechanism is genuinely exercised.
    torch.manual_seed(20260913)
    with torch.no_grad():
        for parameter in model.center_update.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))

    params = {"t2_recurrent": _n_params(reference), "t3_recurrent": _n_params(model)}
    keys_identical = set(reference.state_dict().keys()) == set(model.state_dict().keys())
    shapes_identical = (
        {k: tuple(v.shape) for k, v in reference.state_dict().items()}
        == {k: tuple(v.shape) for k, v in model.state_dict().items()}
    )

    # --- capture the full three-round flow ---------------------------------
    qs: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    captured: dict[str, torch.Tensor] = {}
    ctxs: list[torch.Tensor] = []
    centre_pool_calls = [0]

    handles = [
        model.patch_encoder.register_forward_hook(
            lambda _m, _i, o: captured.__setitem__("h0", o)
        ),
        model.pair_encoder.register_forward_hook(lambda _m, _i, o: qs.append(o)),
        model.center_update.register_forward_hook(lambda _m, _i, o: deltas.append(o)),
    ]
    orig_pool_nodes = model._pool_nodes
    orig_pool_pairs = model._pool_pairs
    orig_pool_centres = model._pool_pairs_to_centres

    def wrap_pool_nodes(value: torch.Tensor, batch_t: torch.Tensor, n: int) -> torch.Tensor:
        captured["unary_input"] = value
        return orig_pool_nodes(value, batch_t, n)

    def wrap_pool_pairs(
        value: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_bucket: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        captured["pair_input"] = value
        return orig_pool_pairs(value, pair_batch, pair_bucket, n)

    def wrap_pool_centres(*args: Any, **kwargs: Any) -> torch.Tensor:
        centre_pool_calls[0] += 1
        result = orig_pool_centres(*args, **kwargs)
        ctxs.append(result)
        return result

    model._pool_nodes = wrap_pool_nodes  # type: ignore[method-assign]
    model._pool_pairs = wrap_pool_pairs  # type: ignore[method-assign]
    model._pool_pairs_to_centres = wrap_pool_centres  # type: ignore[method-assign]

    model.capture_diagnostics = True
    out = model(batch)
    num_rounds = len(qs)
    if num_rounds != 3 or len(deltas) != 3 or len(ctxs) != 3:
        raise RuntimeError(
            f"unexpected T=3 flow: q={num_rounds} delta={len(deltas)} ctx={len(ctxs)}"
        )
    q0, q1, q2 = qs
    # Use the *actual* tensors that participated in the forward graph.
    h0 = model.last_h0
    h1 = model.last_h1
    h2 = model.last_h2
    h3 = captured["unary_input"]

    # q2 depends on h2 (autograd, not just numerical drift); q1 on h1.
    grad_q2_h2 = torch.autograd.grad(q2.sum(), h2, retain_graph=True)[0]
    grad_q1_h1 = torch.autograd.grad(q1.sum(), h1, retain_graph=True)[0]
    # The readout must consume the third centre update (h3 = h2 + delta2) and
    # the final relation q2.
    unary_input_diff = float(
        (captured["unary_input"] - (h2 + deltas[2])).abs().max()
    )
    unary_differs_from_h2 = float(
        (captured["unary_input"] - h2).abs().max()
    )
    pair_input_diff = float((captured["pair_input"] - q2).abs().max())

    # --- mask / bucket / batch indexing across all three rounds ------------
    width = 2 * Q_DIM + 1
    count_columns = [bucket * width + 2 * Q_DIM for bucket in range(DISTANCE_BUCKETS)]
    count_diff = max(
        float((ctxs[0][:, count_columns] - ctxs[i][:, count_columns]).abs().max())
        for i in (1, 2)
    )
    ctx_width_ok = all(int(ctx.shape[1]) == DISTANCE_BUCKETS * width for ctx in ctxs)
    ctx_finite = all(bool(torch.isfinite(ctx).all()) for ctx in ctxs)

    source = batch.pair_index[0]
    target = batch.pair_index[1]
    bucket = batch.pair_bucket
    n_centres = int(ctxs[0].shape[0])
    counts_ok = True
    for b in range(DISTANCE_BUCKETS):
        mask = bucket == b
        manual = torch.zeros(n_centres)
        if int(mask.sum()) > 0:
            endpoints = torch.cat([source[mask], target[mask]], dim=0)
            manual.index_add_(0, endpoints, torch.ones(endpoints.shape[0]))
        logged = torch.expm1(ctxs[0][:, count_columns[b]])
        if float((manual - logged).abs().max()) > 1.0e-4:
            counts_ok = False

    index_range_ok = bool(
        int(source.min()) >= 0
        and int(target.min()) >= 0
        and int(source.max()) < int(batch.num_nodes)
        and int(target.max()) < int(batch.num_nodes)
    )
    bucket_range_ok = bool(
        int(bucket.min()) >= 0 and int(bucket.max()) < DISTANCE_BUCKETS
    )

    # backward pass finite
    loss = F.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    grad_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    for handle in handles:
        handle.remove()
    del model._pool_nodes  # type: ignore[attr-defined]
    del model._pool_pairs  # type: ignore[attr-defined]
    del model._pool_pairs_to_centres  # type: ignore[attr-defined]

    checks = {
        "params_t3_equals_t2_recurrent": params["t3_recurrent"] == params["t2_recurrent"],
        "params_t3_expected_82115": params["t3_recurrent"] == EXPECTED_SMALL_TOTAL,
        "head_expected_4135": _n_params(model.head) == EXPECTED_HEAD,
        "state_dict_keys_identical": bool(keys_identical),
        "state_dict_shapes_identical": bool(shapes_identical),
        "three_pair_encoder_calls": num_rounds == 3,
        "three_center_update_calls": len(deltas) == 3,
        "three_centre_aggregations": centre_pool_calls[0] == 3,
        "q1_differs_from_q0": float((q1 - q0).abs().max()) > 1.0e-6,
        "q2_differs_from_q1": float((q2 - q1).abs().max()) > 1.0e-6,
        "h3_differs_from_h2": unary_differs_from_h2 > 1.0e-6,
        "q1_depends_on_h1": float(grad_q1_h1.abs().max()) > 0.0,
        "q2_depends_on_h2": float(grad_q2_h2.abs().max()) > 0.0,
        "readout_unary_uses_third_update": unary_input_diff == 0.0,
        "readout_pair_uses_q2": pair_input_diff == 0.0,
        "round_count_block_identical": count_diff == 0.0,
        "ctx_width_expected": bool(ctx_width_ok),
        "ctx_finite": bool(ctx_finite),
        "manual_incidence_counts_match": bool(counts_ok),
        "pair_index_range_ok": bool(index_range_ok),
        "bucket_range_ok": bool(bucket_range_ok),
        "forward_finite": bool(torch.isfinite(out).all()),
        "backward_grad_finite": bool(grad_finite),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parameters": params,
        "flow": {
            "n_pair_encoder_calls": num_rounds,
            "n_center_update_calls": len(deltas),
            "n_centre_aggregations": centre_pool_calls[0],
            "q1_q0_max_abs_drift": float((q1 - q0).abs().max()),
            "q2_q1_max_abs_drift": float((q2 - q1).abs().max()),
            "h1_h0_max_abs_drift": float((h1 - h0).detach().abs().max()),
            "h2_h1_max_abs_drift": float((h2 - h1).detach().abs().max()),
            "h3_h2_max_abs_drift": unary_differs_from_h2,
            "grad_q1_wrt_h1_max": float(grad_q1_h1.abs().max()),
            "grad_q2_wrt_h2_max": float(grad_q2_h2.abs().max()),
            "unary_input_vs_h3_max_diff": unary_input_diff,
            "pair_input_vs_q2_max_diff": pair_input_diff,
            "round_count_block_max_diff": count_diff,
        },
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "sanity_t3.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"T=3 sanity checks failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_variant(
    build_fn: Callable[[int], nn.Module],
    *,
    seed: int,
    tag: str,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> dict[str, Any]:
    """Run the inherited training loop, redirecting outputs to this result dir."""
    train_data, valid_data, _ = rec.load_encoded()
    original_dirs = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    original_protocol = shead.OPTIMIZED_PROTOCOL
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    protocol = dict(original_protocol)
    if max_epochs is not None:
        protocol["max_epochs"] = int(max_epochs)
    if patience is not None:
        protocol["patience"] = int(patience)
    shead.OPTIMIZED_PROTOCOL = protocol
    try:
        return shead.train_model(
            build_fn=build_fn,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=str(tag),
            save_state=True,
            real_batch_identity=True,
            expected_total=EXPECTED_SMALL_TOTAL,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original_dirs
        shead.OPTIMIZED_PROTOCOL = original_protocol


def train_signed(seed: int = 0) -> dict[str, Any]:
    summary = train_variant(build_recurrent_signed, seed=int(seed), tag="signed")
    _write_json(RESULTS_DIR / f"signed_seed{seed}.json", summary)
    return summary


def train_t3(seed: int = 0) -> dict[str, Any]:
    summary = train_variant(
        build_recurrent_t3,
        seed=int(seed),
        tag="t3",
        max_epochs=T3_MAX_EPOCHS,
        patience=T3_PATIENCE,
    )
    _write_json(RESULTS_DIR / f"t3_seed{seed}.json", summary)
    return summary


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def _t2_reference(seed: int) -> dict[str, Any]:
    path = T2_RESULTS_DIR / f"recurrent_seed{seed}.json"
    if path.exists():
        data = _read_json(path)
        return {
            "best_valid_mae": float(data["best_valid_mae"]),
            "best_epoch": int(data["best_epoch"]),
            "parameters": int(data["parameters"]),
            "source": str(path),
        }
    fallback = T2_SEED0_VALID if seed == 0 else T2_SEED1_VALID
    return {
        "best_valid_mae": float(fallback),
        "best_epoch": None,
        "parameters": EXPECTED_SMALL_TOTAL,
        "source": "frozen constant",
    }


def decision_signed(seed: int = 0) -> dict[str, Any]:
    reference = _t2_reference(seed)
    candidate = _read_json(RESULTS_DIR / f"signed_seed{seed}.json")
    ref = float(reference["best_valid_mae"])
    value = float(candidate["best_valid_mae"])
    improvement = ref - value  # positive => signed-q improves over ReLU T=2
    if improvement >= A_STRONG:
        verdict, next_step = "STRONG_POSITIVE", "run_seed1_replication"
    elif improvement >= A_WEAK:
        verdict, next_step = "WEAK_POSITIVE", "optional_seed1"
    elif improvement > -A_NOISE:
        verdict, next_step = "NO_MEANINGFUL_SIGNAL", "stop_no_sweep"
    else:
        verdict, next_step = "DEGRADED", "stop"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "A_signed_q",
        "seed": int(seed),
        "t2_recurrent_valid_mae": ref,
        "t2_recurrent_best_epoch": reference["best_epoch"],
        "signed_valid_mae": value,
        "signed_best_epoch": int(candidate["best_epoch"]),
        "signed_epochs_run": int(candidate["epochs_run"]),
        "signed_horizon_boundary_warning": bool(
            candidate["horizon_boundary_warning"]
        ),
        "signed_params": int(candidate["parameters"]),
        "improvement_over_t2": improvement,
        "strong_gate": A_STRONG,
        "weak_gate": A_WEAK,
        "noise_band": A_NOISE,
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_signed_seed{seed}.json", payload)
    return payload


def decision_t3(seed: int = 0) -> dict[str, Any]:
    reference = _t2_reference(seed)
    candidate = _read_json(RESULTS_DIR / f"t3_seed{seed}.json")
    ref = float(reference["best_valid_mae"])
    value = float(candidate["best_valid_mae"])
    improvement = ref - value  # positive => T=3 improves over T=2
    horizon = bool(candidate["horizon_boundary_warning"])
    if improvement >= B_STRONG:
        verdict, next_step = "STRONG_DEPTH_SIGNAL", "stop_no_T4"
    elif improvement >= B_WEAK:
        verdict, next_step = "WEAK_DEPTH_SCALING", "stop_no_T4"
    elif improvement > -B_WEAK:
        verdict, next_step = "SATURATED", "stop"
    else:
        verdict, next_step = "DEGRADED", "T2_is_sweet_spot"
    if horizon and improvement > 0.0:
        next_step = "extend_horizon_before_concluding"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "B_t3_depth",
        "seed": int(seed),
        "t2_recurrent_valid_mae": ref,
        "t2_recurrent_best_epoch": reference["best_epoch"],
        "t3_valid_mae": value,
        "t3_best_epoch": int(candidate["best_epoch"]),
        "t3_epochs_run": int(candidate["epochs_run"]),
        "t3_max_epochs": int(candidate["protocol"]["max_epochs"]),
        "t3_patience": int(candidate["protocol"]["patience"]),
        "t3_horizon_boundary_warning": horizon,
        "t3_params": int(candidate["parameters"]),
        "improvement_over_t2": improvement,
        "strong_gate": B_STRONG,
        "weak_gate": B_WEAK,
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_t3_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "sanity_signed",
            "sanity_t3",
            "train_signed",
            "train_t3",
            "decision_signed",
            "decision_t3",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    stage = args.stage
    if stage in {"audit", "all"}:
        print(json.dumps(audit(), indent=2, default=str), flush=True)
    if stage in {"sanity_signed", "all"}:
        print(json.dumps(sanity_signed()["checks"], indent=2), flush=True)
    if stage in {"sanity_t3", "all"}:
        print(json.dumps(sanity_t3()["checks"], indent=2), flush=True)
    if stage in {"train_signed", "all"}:
        train_signed(args.seed)
    if stage in {"train_t3", "all"}:
        train_t3(args.seed)
    if stage in {"decision_signed", "all"}:
        print(json.dumps(decision_signed(args.seed), indent=2), flush=True)
    if stage in {"decision_t3", "all"}:
        print(json.dumps(decision_t3(args.seed), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
