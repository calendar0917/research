"""Seed-1 confirmation + zero-training common-horizon audit for the strict-static
dictionary-pair line.

Round: **ZINC-strict-static-dictionary-pair-v1-confirmation**.

This module does **not** change the architecture contract of
``zinc_static_dictionary_pair`` (v0).  It imports that module as a library and
adds exactly two things:

Phase A -- a zero-training *common-horizon* audit of the existing seed-0
    curves.  ``H_common = min(epochs_run)`` over the three seed-0 arms is read
    from the durable curve CSVs (never hardcoded).  For each arm it recomputes
    ``best_valid_within_[1, H_common]``, the best epoch and the mean of the five
    best valid MAEs within the window, plus the running-best curve.  A fixed
    Top-5 *weight soup* restricted to ``epoch <= H_common`` is only reported if
    the v0 run saved enough checkpoint states; the v0 runner keeps only the best
    checkpoint on disk, so this diagnostic is reported as **unavailable from
    saved states** instead of being approximated by a best-5 MAE mean.

Phase B -- exactly three seed-1 full-training runs (``S0``, ``S-Dict`` and,
    conditionally, the parameter-matched ``S-Dense``), architecture unchanged.
    Every arm is forced to run the full 240 epochs (no early termination) so the
    equal-horizon view has identical checkpoint opportunities.  A *shadow* early
    stopping tracker simulates the v0 protocol (patience 40, best
    official-valid checkpoint) without ever stopping the optimizer, so each
    single run yields two legitimate views:

    * ``equal_horizon`` -- best valid / best epoch / Top-5 soup over epochs
      1..240;
    * ``shadow``        -- best valid / best epoch / Top-5 soup over the prefix
      ``[1, shadow_stop_epoch]`` the inherited protocol would have used.

Stages: ``phase_a param_audit init_match integrity smoke s0 dense dict train
provisional analyze all``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = sdp.REPO_ROOT
TRACK_ROOT = sdp.TRACK_ROOT
EXPERIMENT_NAME = "zinc_static_dictionary_pair_v1_confirmation"
PROTOCOL_VERSION = "zinc_strict_static_dict_pair_v1_confirmation"

RESULTS_DIR = TRACK_ROOT / "results/zinc_static_dictionary_pair_v1_confirmation"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"

# The v0 encoded cache is reused read-only: the tokenizer / data split are
# identical, and re-encoding would only burn GPU wall clock.
V0_RESULTS_DIR = sdp.RESULTS_DIR
V0_RUNS_DIR = sdp.RUNS_DIR
V0_CURVE_DIR = sdp.CURVE_DIR

SEED1 = 1
FORCED_MAX_EPOCHS = int(sdp.OPTIMIZED_PROTOCOL["max_epochs"])  # 240
SHADOW_PATIENCE = int(sdp.OPTIMIZED_PROTOCOL["patience"])  # 40

# Pre-registered gates.  Phase A / seed-1 share the v0 gate magnitudes.
GATE_BASE_GAIN = 0.004
GATE_DICT_GAIN = 0.002
# "Clearly positive direction" threshold used only to decide whether the
# conditional S-Dense seed-1 control is purchased.
PROVISIONAL_DIRECTION = 0.002

FIXED_EPOCH_REPORTS = (160, 180, 200)
ARMS = ("s0", "dense", "dict")
ARM_LABEL = {"s0": "S0", "dense": "S-Dense", "dict": "S-Dict"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_curve(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


# ---------------------------------------------------------------------------
# Phase A -- zero-training common-horizon audit
# ---------------------------------------------------------------------------


def _running_best(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    best = float("inf")
    out: list[float] = []
    for row in rows:
        best = min(best, float(row["valid_mae"]))
        out.append(best)
    return out


def _window_stats(rows: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    window = [row for row in rows if int(row["epoch"]) <= int(horizon)]
    if not window:
        raise ValueError("empty horizon window")
    valid = [float(row["valid_mae"]) for row in window]
    best_index = int(np.argmin(valid))
    best5 = float(np.mean(sorted(valid)[:5]))
    return {
        "horizon": int(horizon),
        "best_valid": float(valid[best_index]),
        "best_epoch": int(window[best_index]["epoch"]),
        "mean_best5_valid": best5,
        "n_epochs_in_window": int(len(window)),
    }


def _saved_epoch_states_available() -> dict[str, Any]:
    """Does the v0 round have enough on-disk states for a restricted soup?

    The v0 runner keeps only the *best* checkpoint in ``states/``; per-epoch
    Top-5 snapshots are an in-memory cache that is never written.  A restricted
    Top-5 weight soup therefore cannot be formed without retraining, which this
    round forbids.
    """
    state_dir = sdp.STATE_DIR
    present = sorted(p.name for p in state_dir.glob("*_selection_state.pt")) if state_dir.exists() else []
    return {
        "state_dir": str(state_dir),
        "selection_states": present,
        "per_epoch_states_saved": False,
        "restricted_top5_soup_available": False,
        "reason": (
            "v0 saved only the single best-checkpoint selection_state per arm; "
            "per-epoch Top-5 snapshots were an in-memory cache and are not on disk"
        ),
    }


def _classify_common_horizon(m0c: float, mdc: float, mkc: float) -> str:
    base_gain = float(m0c) - float(mkc)
    dict_gain = float(mdc) - float(mkc)
    if base_gain >= GATE_BASE_GAIN and dict_gain >= GATE_DICT_GAIN:
        return "COMMON_HORIZON_SIGNAL_SURVIVES"
    if float(mkc) < float(m0c) and float(mkc) < float(mdc):
        return "COMMON_HORIZON_DIRECTION_ONLY"
    return "COMMON_HORIZON_SIGNAL_FRAGILE"


def phase_a_stage() -> dict[str, Any]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        path = V0_CURVE_DIR / f"{arm}_seed0_curve.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing seed-0 curve: {path}")
        curves[arm] = _read_curve(path)

    epochs_run = {arm: int(len(curves[arm])) for arm in ARMS}
    h_common = int(min(epochs_run.values()))

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = curves[arm]
        full = _window_stats(rows, int(epochs_run[arm]))
        common = _window_stats(rows, h_common)
        running = _running_best(rows)
        fixed = {
            int(epoch): float(rows[int(epoch) - 1]["valid_mae"])
            for epoch in (*FIXED_EPOCH_REPORTS, h_common)
            if 1 <= int(epoch) <= int(epochs_run[arm])
        }
        per_arm[arm] = {
            "epochs_run": int(epochs_run[arm]),
            "best_valid_full": full["best_valid"],
            "best_epoch_full": full["best_epoch"],
            "mean_best5_valid_full": full["mean_best5_valid"],
            "best_valid_common": common["best_valid"],
            "best_epoch_common": common["best_epoch"],
            "mean_best5_valid_common": common["mean_best5_valid"],
            "running_best_valid_at_epochs": {
                str(epoch): float(running[int(epoch) - 1])
                for epoch in (*FIXED_EPOCH_REPORTS, h_common)
                if 1 <= int(epoch) <= int(epochs_run[arm])
            },
            "valid_mae_at_epochs": {str(k): v for k, v in fixed.items()},
            "late_window_gain_full_minus_common": float(full["best_valid"] - common["best_valid"]),
        }

    m0c = per_arm["s0"]["best_valid_common"]
    mdc = per_arm["dense"]["best_valid_common"]
    mkc = per_arm["dict"]["best_valid_common"]
    s0c = per_arm["s0"]["mean_best5_valid_common"]
    sdc = per_arm["dense"]["mean_best5_valid_common"]
    skc = per_arm["dict"]["mean_best5_valid_common"]

    base_gain_common = float(m0c) - float(mkc)
    dict_gain_common = float(mdc) - float(mkc)
    base_gain_common_best5 = float(s0c) - float(skc)
    dict_gain_common_best5 = float(sdc) - float(skc)

    verdict = _classify_common_horizon(m0c, mdc, mkc)
    saved = _saved_epoch_states_available()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "phase_a_common_horizon_audit",
        "source": "zinc_static_dictionary_pair (v0) seed-0 curve CSVs",
        "epochs_run": epochs_run,
        "H_common": h_common,
        "H_common_read_from_artifacts": True,
        "fixed_epoch_reports": list((*FIXED_EPOCH_REPORTS, h_common)),
        "per_arm": per_arm,
        "common_horizon": {
            "M0c_s0": float(m0c),
            "MDc_dense": float(mdc),
            "MKc_dict": float(mkc),
            "base_gain_common": base_gain_common,
            "dict_gain_common": dict_gain_common,
            "S0c_mean_best5": float(s0c),
            "SDc_mean_best5": float(sdc),
            "SKc_mean_best5": float(skc),
            "base_gain_common_best5": base_gain_common_best5,
            "dict_gain_common_best5": dict_gain_common_best5,
            "gate_base_gain": GATE_BASE_GAIN,
            "gate_dict_gain": GATE_DICT_GAIN,
            "verdict": verdict,
        },
        "restricted_top5_soup": saved,
        "late_horizon_dependence": {
            "note": (
                "full-run best minus common-horizon best; a positive value means the "
                "arm's headline number depends on epochs after H_common"
            ),
            **{
                arm: per_arm[arm]["late_window_gain_full_minus_common"] for arm in ARMS
            },
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "common_horizon_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training with a shadow early-stopping tracker (single 240-epoch run)
# ---------------------------------------------------------------------------


def _topk_epochs(rows: Sequence[Mapping[str, Any]], k: int = 5) -> list[int]:
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["valid_mae"]))
    return [int(i) + 1 for i in order[: int(k)]]


def _build_soup(
    arm: str,
    cache: Mapping[int, Mapping[str, torch.Tensor]],
    members: Sequence[int],
    seed: int,
    device: torch.device,
    curve: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not cache or not members:
        return {"available": False}
    keys = list(next(iter(cache.values())).keys())
    soup_state = {
        key: torch.stack([cache[epoch][key].float() for epoch in members]).mean(0) for key in keys
    }
    soup_model = sdp.ARMS[str(arm)](seed=seed)
    soup_model.load_state_dict(soup_state)
    soup_model.to(device)
    soup_mae, _, soup_predictions = sdp._evaluate_mae(soup_model, _SOUP_EVAL_LOADER["loader"], device)
    payload = {
        "available": True,
        "members": [int(epoch) for epoch in members],
        "member_valid_mae": [float(curve[int(epoch) - 1]["valid_mae"]) for epoch in members],
        "soup_valid_mae": float(soup_mae),
        "soup_predictions": soup_predictions.tolist(),
    }
    del soup_model
    return payload


# The soup evaluation loader is set per-run; keeping it in a module-level slot
# avoids threading it through the soup helper while staying single-process.
_SOUP_EVAL_LOADER: dict[str, Any] = {}


def train_arm_v1(
    arm: str,
    *,
    seed: int = SEED1,
    device: str = "cpu",
    max_epochs: int | None = None,
    patience: int | None = None,
    train_subset: int | None = None,
    valid_subset: int | None = None,
    tag: str | None = None,
    save_state: bool = True,
) -> dict[str, Any]:
    """One full-horizon run that simultaneously produces the two protocol views.

    Training is **never** terminated early: ``max_epochs`` is always run.  The
    ``patience`` argument only configures the shadow early-stopping tracker that
    records where the inherited v0 protocol would have stopped.
    """
    device_obj = torch.device(str(device))
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    max_epochs = int(FORCED_MAX_EPOCHS if max_epochs is None else max_epochs)
    patience = int(SHADOW_PATIENCE if patience is None else patience)

    train_data, valid_data, audit = sdp.load_encoded(train_subset, valid_subset)
    if int(audit["typed_vocabulary_size_with_oov"]) != sdp.TYPED_VOCAB_SIZE:
        raise RuntimeError("typed vocabulary drift")
    if int(audit["parent_vocabulary_size_with_oov"]) != sdp.PARENT_VOCAB_SIZE:
        raise RuntimeError("parent vocabulary drift")

    protocol = dict(sdp.OPTIMIZED_PROTOCOL)
    protocol["max_epochs"] = int(max_epochs)
    protocol["patience"] = int(patience)
    protocol["forced_full_horizon"] = True
    protocol["shadow_early_stopping"] = True

    model = sdp.ARMS[str(arm)](seed=int(seed)).to(device_obj)
    total_params = sdp._n_params(model)
    branch_params = sdp._n_params(model.residual_branch)

    first_batch = sdp._first_batch(valid_data, device_obj, limit=min(128, len(valid_data)))
    contract = sdp.static_contract_checks(model, first_batch)
    if not contract["passed"]:
        raise RuntimeError(f"strict-static integrity gates failed for arm {arm}: {contract}")
    if not model.patch_encoder.residual_enabled:
        raise RuntimeError("residual unexpectedly disabled")

    viability = None
    if arm in ("dense", "dict"):
        viability = sdp.gradient_viability_checks(model, first_batch)
        if not viability["branch_receives_gradient"]:
            torch.manual_seed(int(seed) + 17)
            model.residual_branch.reset_parameters()
            viability = sdp.gradient_viability_checks(model, first_batch)
            if not viability["branch_receives_gradient"]:
                raise RuntimeError(f"branch receives no step-0 gradient for arm {arm}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    loader = zpp._make_loader(
        train_data,
        int(protocol["batch_size"]),
        True,
        int(seed) + int(protocol["train_shuffle_seed_offset"]),
    )
    eval_loader = zpp._make_loader(
        valid_data,
        int(protocol["batch_size"]),
        False,
        int(seed) + int(protocol["eval_shuffle_seed_offset"]),
    )
    _SOUP_EVAL_LOADER["loader"] = eval_loader
    steps_per_epoch = int(math.ceil(len(train_data) / int(protocol["batch_size"])))
    clip = float(protocol["gradient_clip_norm"])

    # --- equal-horizon view -------------------------------------------------
    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    # --- shadow early-stopping view (never stops the optimizer) -------------
    shadow_best_mae = float("inf")
    shadow_best_epoch = 1
    shadow_stale = 0
    shadow_stop_epoch: int | None = None
    shadow_states: dict[int, dict[str, torch.Tensor]] = {}
    shadow_members_frozen: list[int] | None = None

    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    branch_params_list = [] if model.residual_branch is None else list(model.residual_branch.parameters())
    branch_ids = {id(p) for p in branch_params_list}

    def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return math.sqrt(total)

    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        head_grad = 0.0
        branch_grad = 0.0
        backbone_grad = 0.0
        steps = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            head_grad += _grad_norm(head_params)
            branch_grad += _grad_norm(branch_params_list)
            backbone_grad += _grad_norm(
                [p for p in model.parameters() if id(p) not in head_ids and id(p) not in branch_ids]
            )
            steps += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_mae = epoch_loss / max(seen, 1)
        losses.append(float(train_mae))
        valid_mae, _, _ = sdp._evaluate_mae(model, eval_loader, device_obj)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_mae": float(train_mae),
            "valid_mae": float(valid_mae),
            "head_grad_norm": float(head_grad / max(steps, 1)),
            "branch_grad_norm": float(branch_grad / max(steps, 1)),
            "backbone_grad_norm": float(backbone_grad / max(steps, 1)),
            "gamma": float(model.patch_encoder.gamma.detach()),
        }
        if isinstance(model.residual_branch, sdp.ResidualDictionaryBranch):
            row["tau"] = float(model.residual_branch.current_tau().detach())
        curve.append(row)

        snapshot = {
            key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
        }

        # Equal-horizon cache: Top-5 over all epochs seen (final: 1..max_epochs).
        epoch_states[int(epoch)] = snapshot
        keep = set(_topk_epochs(curve, 5))
        for cached_epoch in list(epoch_states):
            if cached_epoch not in keep:
                del epoch_states[cached_epoch]

        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in model.state_dict().items()
            }

        # Shadow tracker: identical rule to the v0 protocol, but it never stops.
        if shadow_stop_epoch is None:
            if valid_mae < shadow_best_mae:
                shadow_best_mae = float(valid_mae)
                shadow_best_epoch = int(epoch)
                shadow_stale = 0
            else:
                shadow_stale += 1
            shadow_states[int(epoch)] = snapshot
            keep_shadow = set(_topk_epochs(curve, 5))
            for cached_epoch in list(shadow_states):
                if cached_epoch not in keep_shadow:
                    del shadow_states[cached_epoch]
            if shadow_stale >= patience:
                shadow_stop_epoch = int(epoch)
                shadow_members_frozen = sorted(int(e) for e in shadow_states)

        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{tag or arm} seed{seed}] epoch={epoch:03d} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"shadow_best={shadow_best_mae:.6f}@{shadow_best_epoch} "
                f"shadow_stop={shadow_stop_epoch} gamma={row['gamma']:.4f}",
                flush=True,
            )
    wall_clock = float(time.perf_counter() - started)

    for row in curve:
        row["checkpoint_selected"] = int(row["epoch"] == best_epoch)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device_obj)

    best_valid, valid_targets, best_predictions = sdp._evaluate_mae(model, eval_loader, device_obj)

    equal_members = sorted(_topk_epochs(curve, 5))
    soup_payload = _build_soup(arm, epoch_states, equal_members, int(seed), device_obj, curve)
    soup_payload["view"] = "equal_horizon_1_to_max_epochs"

    if shadow_members_frozen is not None:
        shadow_members = shadow_members_frozen
    else:
        shadow_members = sorted(_topk_epochs(curve, 5))
    shadow_soup = _build_soup(arm, shadow_states, shadow_members, int(seed), device_obj, curve)
    shadow_soup["view"] = "shadow_prefix_1_to_stop_epoch"
    shadow_soup["shadow_stop_epoch"] = None if shadow_stop_epoch is None else int(shadow_stop_epoch)

    diagnostics: dict[str, Any] = {}
    if arm == "dict":
        diagnostics = sdp.dictionary_diagnostics(model, eval_loader, device_obj)
    residual_ablation = sdp.residual_ablation_shift(model, eval_loader, device_obj, best_predictions)

    curve_path = CURVE_DIR / f"{tag or arm}_seed{seed}_curve.csv"
    sdp._write_csv(curve_path, curve)
    state_path = None
    if save_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path = STATE_DIR / f"{tag or arm}_seed{seed}_selection_state.pt"
        torch.save(model.state_dict(), state_path)

    peak_memory_mb = None
    if device_obj.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))

    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "arm": str(arm),
        "seed": int(seed),
        "tag": str(tag or arm),
        "device": str(device_obj),
        "protocol": protocol,
        "max_epochs": int(max_epochs),
        "shadow_patience": int(patience),
        "forced_full_horizon": True,
        "epochs_run": int(len(losses)),
        "early_stopped": False,
        "steps_per_epoch": int(steps_per_epoch),
        "equal_horizon": {
            "best_valid_mae": float(best_mae),
            "best_epoch": int(best_epoch),
            "mean_best5_valid_mae": float(np.mean(sorted(r["valid_mae"] for r in curve)[:5])),
            "soup": soup_payload,
        },
        "shadow": {
            "stop_epoch": None if shadow_stop_epoch is None else int(shadow_stop_epoch),
            "epochs_run": int(shadow_stop_epoch if shadow_stop_epoch is not None else len(losses)),
            "best_valid_mae": float(shadow_best_mae),
            "best_epoch": int(shadow_best_epoch),
            "soup": shadow_soup,
        },
        # Backwards-compatible aliases (equal-horizon == this round's primary).
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(best_valid),
        "wall_clock_s": float(wall_clock),
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameters": int(total_params),
        "branch_parameters": int(branch_params),
        "gamma_final": float(model.patch_encoder.gamma.detach()),
        "tau_final": (
            float(model.residual_branch.current_tau().detach())
            if isinstance(model.residual_branch, sdp.ResidualDictionaryBranch)
            else None
        ),
        "curve_path": str(curve_path),
        "state_path": None if state_path is None else str(state_path),
        "strict_static_contract": contract,
        "gradient_viability": viability,
        "dictionary_diagnostics": diagnostics,
        "residual_ablation": residual_ablation,
        "soup": soup_payload,
        "valid_predictions": best_predictions.tolist(),
        "valid_targets": valid_targets.tolist(),
        "official_test_loaded": False,
        "git_commit": sdp._git_commit(),
        "environment": sdp._environment_fingerprint(str(device_obj)),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{tag or arm}_seed{seed}.json", summary)
    print(
        f"[{tag or arm} seed{seed}] equal_best={best_mae:.6f}@{best_epoch} "
        f"shadow_best={shadow_best_mae:.6f}@{shadow_best_epoch} stop={shadow_stop_epoch} "
        f"params={total_params} branch_params={branch_params} wall={wall_clock:.1f}s",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# audit / integrity / smoke stages
# ---------------------------------------------------------------------------


def parameter_audit(seed: int = SEED1) -> dict[str, Any]:
    s0 = sdp.build_s0(seed)
    dense = sdp.build_dense(seed)
    dict_model = sdp.build_dict(seed)
    r_dim = int(s0.unified_graph_width)
    match = sdp.matched_dense_hidden(48)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "unified_graph_width": r_dim,
        "unified_graph_width_is_302": bool(r_dim == 302),
        "center_context_false": bool(s0.center_context is False),
        "center_update_is_none": bool(s0.center_update is None),
        "head_hidden": list(sdp.HEAD_HIDDEN),
        "params": {
            "s0_total": sdp._n_params(s0),
            "s0_head": sdp._n_params(s0.head),
            "dense_total": sdp._n_params(dense),
            "dict_total": sdp._n_params(dict_model),
            "dense_branch": sdp._n_params(dense.residual_branch),
            "dict_branch": sdp._n_params(dict_model.residual_branch),
        },
        "dict_spec": {"rank": sdp.DICT_RANK, "atoms": sdp.DICT_ATOMS, "tau_init": sdp.DICT_TAU_INIT},
        "dense_hidden_matched": int(sdp.DENSE_HIDDEN),
        "parameter_match": {
            "dict_branch_params": int(sdp._n_params(dict_model.residual_branch)),
            "dense_branch_params": int(sdp._n_params(dense.residual_branch)),
            "abs_diff": int(sdp._n_params(dict_model.residual_branch)) - int(sdp._n_params(dense.residual_branch)),
            "relative_error": abs(
                int(sdp._n_params(dense.residual_branch)) - int(sdp._n_params(dict_model.residual_branch))
            )
            / float(int(sdp._n_params(dict_model.residual_branch))),
            "within_one_percent": bool(
                abs(
                    int(sdp._n_params(dense.residual_branch)) - int(sdp._n_params(dict_model.residual_branch))
                )
                / float(int(sdp._n_params(dict_model.residual_branch)))
                < 0.01
            ),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


def initialization_match(seed: int = SEED1) -> dict[str, Any]:
    s0 = sdp.build_s0(seed)
    dense = sdp.build_dense(seed)
    dict_model = sdp.build_dict(seed)
    s0_state = s0.state_dict()
    dense_state = dense.state_dict()
    dict_state = dict_model.state_dict()
    branch_keys = {
        key for key in dict_state if key.startswith("patch_encoder.branch.") or key == "patch_encoder.gamma"
    }
    shared_keys = sorted(
        key
        for key in s0_state
        if key in dense_state
        and key in dict_state
        and s0_state[key].shape == dense_state[key].shape == dict_state[key].shape
        and key not in branch_keys
    )
    max_s0_dense = max(float((s0_state[key] - dense_state[key]).abs().max().item()) for key in shared_keys)
    max_s0_dict = max(float((s0_state[key] - dict_state[key]).abs().max().item()) for key in shared_keys)
    max_dense_dict = max(float((dense_state[key] - dict_state[key]).abs().max().item()) for key in shared_keys)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "n_shared_tensors": int(len(shared_keys)),
        "n_head_tensors": int(len([k for k in shared_keys if k.startswith("head.")])),
        "shared_keys": shared_keys,
        "max_abs_diff_s0_vs_dense": max_s0_dense,
        "max_abs_diff_s0_vs_dict": max_s0_dict,
        "max_abs_diff_dense_vs_dict": max_dense_dict,
        "bit_identical": bool(max_s0_dense == 0.0 and max_s0_dict == 0.0 and max_dense_dict == 0.0),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "initialization_match.json", payload)
    return payload


def integrity_stage() -> dict[str, Any]:
    train_data, valid_data, _audit = sdp.load_encoded(valid_subset=64)
    batch = sdp._first_batch(valid_data, torch.device("cpu"), limit=64)
    payload: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "seed": SEED1}
    for arm in ARMS:
        model = sdp.ARMS[arm](seed=SEED1)
        payload[f"{arm}_contract"] = sdp.static_contract_checks(model, batch)
        if arm in ("dense", "dict"):
            payload[f"{arm}_gradient_viability"] = sdp.gradient_viability_checks(model, batch)
    payload["passed"] = bool(all(payload[f"{arm}_contract"]["passed"] for arm in ARMS))
    payload["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "integrity_gates.json", payload)
    return payload


def smoke_stage(device: str = "cuda") -> dict[str, Any]:
    payload: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "arms": {}}
    for arm in ARMS:
        summary = train_arm_v1(
            arm,
            seed=SEED1,
            device=device,
            max_epochs=2,
            patience=2,
            train_subset=256,
            valid_subset=128,
            tag=f"smoke_{arm}",
            save_state=False,
        )
        payload["arms"][arm] = {
            "equal_horizon_best_valid_mae": summary["equal_horizon"]["best_valid_mae"],
            "shadow_best_valid_mae": summary["shadow"]["best_valid_mae"],
            "parameters": summary["parameters"],
            "branch_parameters": summary["branch_parameters"],
            "contract_passed": summary["strict_static_contract"]["passed"],
            "gradient_viability": summary["gradient_viability"],
            "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
            "device": summary["device"],
        }
    payload["official_test_loaded"] = False
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


# ---------------------------------------------------------------------------
# provisional (buy S-Dense seed1?) and final analysis
# ---------------------------------------------------------------------------


def provisional_stage() -> dict[str, Any]:
    paths = {arm: RUNS_DIR / f"{arm}_seed{SEED1}.json" for arm in ("s0", "dict")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"provisional check requires completed runs: {missing}")
    runs = {arm: _read_json(p) for arm, p in paths.items()}
    eq_s0 = float(runs["s0"]["equal_horizon"]["best_valid_mae"])
    eq_k = float(runs["dict"]["equal_horizon"]["best_valid_mae"])
    eq_s0_soup = float(runs["s0"]["equal_horizon"]["soup"]["soup_valid_mae"])
    eq_k_soup = float(runs["dict"]["equal_horizon"]["soup"]["soup_valid_mae"])
    gains = {
        "equal_horizon_best_s0_minus_dict": eq_s0 - eq_k,
        "equal_horizon_soup_s0_minus_dict": eq_s0_soup - eq_k_soup,
    }
    dict_better_both = bool(eq_k < eq_s0 and eq_k_soup < eq_s0_soup)
    clearly_positive = bool(max(gains.values()) >= PROVISIONAL_DIRECTION)
    if clearly_positive:
        decision = "BUY_DENSE_SEED1"
    elif dict_better_both:
        decision = "BUY_DENSE_SEED1_SUBTHRESHOLD"
    else:
        decision = "SEED1_DICT_NOT_REPLICATED"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "provisional",
        "gains_s0_minus_dict": gains,
        "dict_better_on_both_views": dict_better_both,
        "clearly_positive_direction": clearly_positive,
        "provisional_direction_threshold": PROVISIONAL_DIRECTION,
        "decision": decision,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "provisional.json", payload)
    return payload


def _seed1_view(runs: Mapping[str, Mapping[str, Any]], view: str, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for arm in ARMS:
        node = runs[arm][view]
        out[arm] = float(node["soup"]["soup_valid_mae"] if metric == "soup" else node["best_valid_mae"])
    return out


def _classify_seed1(m0: float, md: float, mk: float, m0_soup: float, md_soup: float, mk_soup: float) -> dict[str, Any]:
    gbase = float(m0) - float(mk)
    gdict = float(md) - float(mk)
    gbase_soup = float(m0_soup) - float(mk_soup)
    gdict_soup = float(md_soup) - float(mk_soup)
    if gbase >= GATE_BASE_GAIN and gdict >= GATE_DICT_GAIN and gbase_soup > 0 and gdict_soup > 0:
        case = "B1_SEED1_DICT_SPECIFIC_REPLICATION"
    elif gbase > 0 and gdict > 0:
        case = "B2_SEED1_DIRECTIONAL_REPLICATION"
    elif gbase > 0 and gdict <= 0:
        case = "B3_SEED1_GENERIC_CAPACITY_OR_AMBIGUOUS"
    else:
        case = "B4_SEED1_NO_REPLICATION"
    return {
        "M0_1_s0": float(m0),
        "MD_1_dense": float(md),
        "MK_1_dict": float(mk),
        "Gbase_1": gbase,
        "Gdict_1": gdict,
        "Gbase_1_soup": gbase_soup,
        "Gdict_1_soup": gdict_soup,
        "gate_base": GATE_BASE_GAIN,
        "gate_dict": GATE_DICT_GAIN,
        "case": case,
    }


def analyze_stage() -> dict[str, Any]:
    paths = {arm: RUNS_DIR / f"{arm}_seed{SEED1}.json" for arm in ARMS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing seed-1 run outputs: {missing}")
    runs = {arm: _read_json(p) for arm, p in paths.items()}

    equal_best = {arm: float(runs[arm]["equal_horizon"]["best_valid_mae"]) for arm in ARMS}
    equal_soup = {arm: float(runs[arm]["equal_horizon"]["soup"]["soup_valid_mae"]) for arm in ARMS}
    shadow_best = {arm: float(runs[arm]["shadow"]["best_valid_mae"]) for arm in ARMS}
    shadow_soup = {arm: float(runs[arm]["shadow"]["soup"]["soup_valid_mae"]) for arm in ARMS}

    equal_case = _classify_seed1(
        equal_best["s0"],
        equal_best["dense"],
        equal_best["dict"],
        equal_soup["s0"],
        equal_soup["dense"],
        equal_soup["dict"],
    )
    shadow_case = _classify_seed1(
        shadow_best["s0"],
        shadow_best["dense"],
        shadow_best["dict"],
        shadow_soup["s0"],
        shadow_soup["dense"],
        shadow_soup["dict"],
    )

    # Original-protocol paired summary: seed0 original + seed1 shadow view.
    v0 = {arm: _read_json(V0_RUNS_DIR / f"{arm}_seed0.json") for arm in ARMS}
    paired = {
        "seed0_best_s0_minus_dict": float(v0["s0"]["best_valid_mae"]) - float(v0["dict"]["best_valid_mae"]),
        "seed0_best_dense_minus_dict": float(v0["dense"]["best_valid_mae"]) - float(v0["dict"]["best_valid_mae"]),
        "seed1_shadow_best_s0_minus_dict": float(shadow_best["s0"]) - float(shadow_best["dict"]),
        "seed1_shadow_best_dense_minus_dict": float(shadow_best["dense"]) - float(shadow_best["dict"]),
        "seed0_soup_s0_minus_dict": float(v0["s0"]["soup"]["soup_valid_mae"]) - float(v0["dict"]["soup"]["soup_valid_mae"]),
        "seed0_soup_dense_minus_dict": float(v0["dense"]["soup"]["soup_valid_mae"]) - float(v0["dict"]["soup"]["soup_valid_mae"]),
        "seed1_shadow_soup_s0_minus_dict": float(shadow_soup["s0"]) - float(shadow_soup["dict"]),
        "seed1_shadow_soup_dense_minus_dict": float(shadow_soup["dense"]) - float(shadow_soup["dict"]),
    }
    paired["mean_best_s0_minus_dict"] = 0.5 * (
        paired["seed0_best_s0_minus_dict"] + paired["seed1_shadow_best_s0_minus_dict"]
    )
    paired["mean_best_dense_minus_dict"] = 0.5 * (
        paired["seed0_best_dense_minus_dict"] + paired["seed1_shadow_best_dense_minus_dict"]
    )
    paired["mean_soup_s0_minus_dict"] = 0.5 * (
        paired["seed0_soup_s0_minus_dict"] + paired["seed1_shadow_soup_s0_minus_dict"]
    )
    paired["mean_soup_dense_minus_dict"] = 0.5 * (
        paired["seed0_soup_dense_minus_dict"] + paired["seed1_shadow_soup_dense_minus_dict"]
    )
    paired["sign_consistent_best_s0_minus_dict"] = bool(
        (paired["seed0_best_s0_minus_dict"] > 0) == (paired["seed1_shadow_best_s0_minus_dict"] > 0)
    )
    paired["sign_consistent_best_dense_minus_dict"] = bool(
        (paired["seed0_best_dense_minus_dict"] > 0) == (paired["seed1_shadow_best_dense_minus_dict"] > 0)
    )

    common_horizon = phase_a_stage()

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "primary_metric": "equal-horizon 240-epoch best official-valid MAE",
        "secondary_metric": "equal-horizon 240-epoch fixed Top-5 weight soup",
        "seed1_equal_horizon": {
            "best_valid_mae": equal_best,
            "soup_valid_mae": equal_soup,
            "best_epoch": {arm: int(runs[arm]["equal_horizon"]["best_epoch"]) for arm in ARMS},
            "soup_members": {arm: runs[arm]["equal_horizon"]["soup"]["members"] for arm in ARMS},
            "classification": equal_case,
        },
        "seed1_shadow_protocol": {
            "best_valid_mae": shadow_best,
            "soup_valid_mae": shadow_soup,
            "best_epoch": {arm: int(runs[arm]["shadow"]["best_epoch"]) for arm in ARMS},
            "stop_epoch": {arm: runs[arm]["shadow"]["stop_epoch"] for arm in ARMS},
            "soup_members": {arm: runs[arm]["shadow"]["soup"]["members"] for arm in ARMS},
            "classification": shadow_case,
        },
        "seed1_equal_horizon_vs_shadow_same_direction": bool(
            (equal_case["Gbase_1"] > 0) == (shadow_case["Gbase_1"] > 0)
            and (equal_case["Gdict_1"] > 0) == (shadow_case["Gdict_1"] > 0)
        ),
        "original_protocol_paired_summary": paired,
        "seed0_common_horizon_diagnostic": common_horizon["common_horizon"],
        "seed0_common_horizon_H": common_horizon["H_common"],
        "dictionary_diagnostics": runs["dict"]["dictionary_diagnostics"],
        "dictionary_gradient_viability": runs["dict"]["gradient_viability"],
        "dense_gradient_viability": runs["dense"]["gradient_viability"],
        "residual_ablation": {arm: runs[arm]["residual_ablation"] for arm in ARMS},
        "strict_static_contract_passed": {arm: bool(runs[arm]["strict_static_contract"]["passed"]) for arm in ARMS},
        "parameter_audit": parameter_audit(),
        "initialization_match": initialization_match(),
        "budget": {
            "seed1_full_training_runs": int(len(ARMS)),
            "arms": [f"{ARM_LABEL[a]} seed1" for a in ARMS],
            "forced_full_horizon_epochs": FORCED_MAX_EPOCHS,
            "total_wall_clock_s": float(sum(float(runs[arm]["wall_clock_s"]) for arm in ARMS)),
            "shadow_patience": SHADOW_PATIENCE,
            "seed0_retrained": False,
            "hpo_run": False,
            "dictionary_pair_kernel_implemented": False,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "analysis.json", payload)
    _write_summary_markdown(payload)
    return payload


def _write_summary_markdown(payload: Mapping[str, Any]) -> None:
    equal = payload["seed1_equal_horizon"]
    shadow = payload["seed1_shadow_protocol"]
    lines: list[str] = []
    lines.append("# ZINC strict-static dictionary-pair v1 confirmation — results summary")
    lines.append("")
    lines.append(
        "One paired seed-1 confirmation of the v0 seed-0 dictionary-specific signal, "
        "with every arm forced to 240 epochs and a shadow early-stopping tracker."
    )
    lines.append("")
    lines.append("Official ZINC **test was never loaded** in any stage.")
    lines.append("")
    lines.append("## Seed-1 equal-horizon view (primary) — 240 epochs, best official-valid MAE")
    lines.append("")
    lines.append("| arm | best MAE | best epoch | Top-5 soup | params | branch params |")
    lines.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        params = payload["parameter_audit"]["params"]
        lines.append(
            f"| {ARM_LABEL[arm]} | {equal['best_valid_mae'][arm]:.6f} | {equal['best_epoch'][arm]} | "
            f"{equal['soup_valid_mae'][arm]:.6f} | {params[arm + '_total']} | "
            f"{params.get(arm + '_branch', 0)} |"
        )
    cls = equal["classification"]
    lines.append("")
    lines.append(f"* `Gbase_1 = M0 - MK` = {cls['Gbase_1']:+.6f} (gate {cls['gate_base']})")
    lines.append(f"* `Gdict_1 = MD - MK` = {cls['Gdict_1']:+.6f} (gate {cls['gate_dict']})")
    lines.append(f"* equal-horizon soup: `S0 - Dict` {cls['Gbase_1_soup']:+.6f}, `Dense - Dict` {cls['Gdict_1_soup']:+.6f}")
    lines.append(f"* equal-horizon case: **{cls['case']}**")
    lines.append("")
    lines.append("## Seed-1 shadow-protocol view (simulated v0 early stopping)")
    lines.append("")
    lines.append("| arm | best MAE | best epoch | stop epoch | Top-5 soup |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        lines.append(
            f"| {ARM_LABEL[arm]} | {shadow['best_valid_mae'][arm]:.6f} | {shadow['best_epoch'][arm]} | "
            f"{shadow['stop_epoch'][arm]} | {shadow['soup_valid_mae'][arm]:.6f} |"
        )
    lines.append("")
    scls = shadow["classification"]
    lines.append(f"* shadow `Gbase_1` {scls['Gbase_1']:+.6f}, `Gdict_1` {scls['Gdict_1']:+.6f} -> **{scls['case']}**")
    lines.append(f"* equal-horizon and shadow views same direction: {payload['seed1_equal_horizon_vs_shadow_same_direction']}")
    lines.append("")
    lines.append("## Seed-0 common-horizon diagnostic")
    lines.append("")
    ch = payload["seed0_common_horizon_diagnostic"]
    lines.append(f"* `H_common` = {payload['seed0_common_horizon_H']}")
    lines.append(f"* M0c {ch['M0c_s0']:.6f} / MDc {ch['MDc_dense']:.6f} / MKc {ch['MKc_dict']:.6f}")
    lines.append(f"* base_gain_common {ch['base_gain_common']:+.6f}, dict_gain_common {ch['dict_gain_common']:+.6f}")
    lines.append(f"* verdict: **{ch['verdict']}**")
    lines.append("")
    lines.append("## Two-seed original-protocol paired summary")
    lines.append("")
    paired = payload["original_protocol_paired_summary"]
    lines.append(f"* mean `S0 - Dict` (best) = {paired['mean_best_s0_minus_dict']:+.6f}")
    lines.append(f"* mean `Dense - Dict` (best) = {paired['mean_best_dense_minus_dict']:+.6f}")
    lines.append(f"* mean `S0 - Dict` (soup) = {paired['mean_soup_s0_minus_dict']:+.6f}")
    lines.append(f"* mean `Dense - Dict` (soup) = {paired['mean_soup_dense_minus_dict']:+.6f}")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    budget = payload["budget"]
    lines.append(f"* seed-1 full training runs: {budget['seed1_full_training_runs']}")
    lines.append(f"* forced full horizon: {budget['forced_full_horizon_epochs']} epochs per arm")
    lines.append(f"* total wall clock: {budget['total_wall_clock_s'] / 60.0:.1f} min")
    lines.append("* seed-0 retrained: false; HPO: none; dictionary-pair kernel: not implemented")
    lines.append("* official test loaded: false")
    lines.append("")
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "phase_a",
            "param_audit",
            "init_match",
            "integrity",
            "smoke",
            "s0",
            "dense",
            "dict",
            "train",
            "provisional",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seed", type=int, default=SEED1)
    args = parser.parse_args(argv)
    sdp._configure_determinism()
    if args.stage == "phase_a":
        print(json.dumps(phase_a_stage(), indent=2, default=float))
    elif args.stage == "param_audit":
        print(json.dumps(parameter_audit(args.seed), indent=2, default=float))
    elif args.stage == "init_match":
        print(json.dumps(initialization_match(args.seed), indent=2, default=float))
    elif args.stage == "integrity":
        print(json.dumps(integrity_stage(), indent=2, default=float))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(args.device), indent=2, default=float))
    elif args.stage in ("s0", "dense", "dict"):
        summary = train_arm_v1(args.stage, seed=args.seed, device=args.device)
        print(json.dumps({k: v for k, v in summary.items() if "predictions" not in k}, indent=2, default=float))
    elif args.stage == "train":
        if args.arm is None:
            raise SystemExit("`train` requires --arm {s0,dense,dict}")
        summary = train_arm_v1(args.arm, seed=args.seed, device=args.device)
        print(json.dumps({k: v for k, v in summary.items() if "predictions" not in k}, indent=2, default=float))
    elif args.stage == "provisional":
        print(json.dumps(provisional_stage(), indent=2, default=float))
    elif args.stage == "analyze":
        print(json.dumps({k: v for k, v in analyze_stage().items() if "predictions" not in k}, indent=2, default=float))
    else:
        phase_a_stage()
        parameter_audit(args.seed)
        initialization_match(args.seed)
        integrity_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
