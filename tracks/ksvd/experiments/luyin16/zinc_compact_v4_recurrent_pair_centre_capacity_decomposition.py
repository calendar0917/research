"""ZINC 2x2 capacity decomposition for the frozen T=2 recurrent pair--centre.

Question
--------
The frozen H48 -> H64 -> H96 centre-width scaling forced three widths to grow
together (``patch_hidden`` drives the patch state *and* the derived encoder
hidden widths ``max(patch_hidden, 64)`` / ``max(patch_hidden // 2, 32)``).  This
module separates the two capacity axes that were confounded:

* **centre-width**: the persistent patch/centre state width ``h`` (which also
  sets the readout unary width and ``center_context_width``);
* **encoder-capacity**: the patch-encoder hidden width and the global-encoder
  hidden width.

Four matched 2x2 cells (``q_dim=16``, ``T=2``, weight-tied refresh throughout):

=====  =====  =====  =====  ==========
cell   h      patch  global params
=====  =====  =====  =====  ==========
A      64     64     32     85,763  (== frozen H64)
B      96     64     32     93,059
C      64     96     48     94,899
D      96     96     48     103,219 (== frozen H96)
=====  =====  =====  =====  ==========

Everything else is inherited verbatim: tokenizer / patch construction /
relation descriptor / pair projection / relation encoder / distance gate /
pair encoder / centre update (hidden 60) / readout semantics / small raw head
(13, 13) / optimizer / schedule / seeds.  No module is added.  The only code
change is an explicit hidden-width override for the two encoder MLPs.

Official ZINC test is **never** loaded by this module.  The workstream answers
a parameter-allocation question; the frozen H96 test number (0.104990 soup
mean) is a historical reference, not a target.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_pair_centre_capacity_decomposition <stage>

Stages: ``params sanity train train_queue soup diagnostics decide report``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any, Mapping, Sequence

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
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = (
    TRACK_ROOT / "results/compact_v4_recurrent_pair_centre_capacity_decomposition"
)
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SOUP_DIR = RESULTS_DIR / "soup_states"

PROTOCOL_VERSION = "compact_v4_recurrent_pair_centre_capacity_decomposition_v1"

Q_DIM = 16
SEEDS = (0, 1)

# cell -> (centre/state width h, patch-encoder hidden, global-encoder hidden)
CELLS: dict[str, dict[str, int]] = {
    "A": {"h": 64, "patch_encoder_hidden": 64, "global_encoder_hidden": 32},
    "B": {"h": 96, "patch_encoder_hidden": 64, "global_encoder_hidden": 32},
    "C": {"h": 64, "patch_encoder_hidden": 96, "global_encoder_hidden": 48},
    "D": {"h": 96, "patch_encoder_hidden": 96, "global_encoder_hidden": 48},
}
CELL_ORDER = ("A", "B", "C", "D")

# Exact parameter counts (verified by ``params`` / ``sanity``).
EXPECTED_PARAMS: dict[str, int] = {"A": 85763, "B": 93059, "C": 94899, "D": 103219}
EXPECTED_HEAD: dict[str, int] = {"A": 4551, "B": 5383, "C": 4551, "D": 5383}
EXPECTED_WIDTH: dict[str, int] = {"A": 334, "B": 398, "C": 334, "D": 398}

# Frozen historical H64/H96 references (validation, CPU regime).
H64_TOTAL = 85763
H96_TOTAL = 103219

STRONG_GATE = 0.002
WEAK_GATE = 0.001


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_cell(cell: str, seed: int = 0) -> torch.nn.Module:
    """Build one 2x2 cell with the frozen initialisation convention.

    Mirrors ``cap.build_capacity`` exactly, but additionally passes explicit
    ``patch_encoder_hidden`` / ``global_encoder_hidden`` so the two encoder
    hidden widths can be varied independently of the centre state width.
    Shared tensors that keep their shape are copied bit-exactly from the
    matched baseline initialisation; the head is re-drawn from the historical
    ``head_seed=0`` stream.
    """
    spec = CELLS[cell]
    h = int(spec["h"])
    baseline = shead.build_baseline(int(seed))
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(int(seed))
    kwargs = shead._base_kwargs()
    kwargs["patch_hidden"] = h
    kwargs["patch_encoder_hidden"] = int(spec["patch_encoder_hidden"])
    kwargs["global_encoder_hidden"] = int(spec["global_encoder_hidden"])
    kwargs.pop("pair_hidden", None)
    with cap._construction_guards(h, Q_DIM):
        model = rec.PatchPathRecurrentPairCentreModel(
            6785,
            32,
            pair_hidden=Q_DIM,
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
    model.head = shead.GenericReader(
        int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN
    )
    return model


def tag_for(cell: str) -> str:
    return f"cell_{cell}"


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    rows = []
    states = {cell: build_cell(cell, 0).state_dict() for cell in CELL_ORDER}
    for cell in CELL_ORDER:
        model = build_cell(cell, 0)
        total = hw._n_params(model)
        head = hw._n_params(model.head)
        rows.append(
            {
                "cell": cell,
                "centre_width_h": int(CELLS[cell]["h"]),
                "patch_encoder_hidden": int(CELLS[cell]["patch_encoder_hidden"]),
                "global_encoder_hidden": int(CELLS[cell]["global_encoder_hidden"]),
                "total_params": int(total),
                "head_params": int(head),
                "backbone_params": int(total - head),
                "unified_graph_width": int(model.unified_graph_width),
                "pooled_unary_width": int(model.pooled_unary_width),
                "pooled_pair_width": int(model.pooled_pair_width),
                "center_context_width": int(model.center_context_width),
                "center_context_hidden": int(model.center_context_hidden),
                "matches_expected": int(total) == EXPECTED_PARAMS[cell]
                and int(head) == EXPECTED_HEAD[cell]
                and int(model.unified_graph_width) == EXPECTED_WIDTH[cell],
            }
        )
    a, d = states["A"], states["D"]
    ad_changed = sorted(k for k in a if a[k].shape != d[k].shape)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "q_dim": Q_DIM,
        "recurrence_rounds": int(rec.RECURRENCE_ROUNDS),
        "rows": rows,
        "a_vs_d_changed_tensors": ad_changed,
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

    a = build_cell("A", 0)
    d = build_cell("D", 0)
    h64 = hw.build_h64(0)
    h96 = cap.build_capacity(0, h=96, q=16)

    def max_state_diff(x: torch.nn.Module, y: torch.nn.Module) -> float:
        xs, ys = x.state_dict(), y.state_dict()
        return max(float((xs[k] - ys[k]).abs().max()) for k in xs)

    a_matches_h64 = max_state_diff(a, h64) == 0.0
    d_matches_h96 = max_state_diff(d, h96) == 0.0

    # B and C must differ from A/D only in the encoder hidden tensors.
    def changed_keys(x: torch.nn.Module, y: torch.nn.Module) -> list[str]:
        xs, ys = x.state_dict(), y.state_dict()
        return sorted(k for k in xs if xs[k].shape != ys[k].shape)

    enc_keys_prefix = ("patch_encoder.", "global_encoder.")
    b = build_cell("B", 0)
    c = build_cell("C", 0)
    b_changed = set(changed_keys(a, b))
    c_changed = set(changed_keys(a, c))
    # B (h 64->96 at fixed narrow encoders) changes only state-dependent
    # tensors; C (fixed h=64, wide encoder hidden layers) changes only
    # patch/global encoder tensors.
    state_keys = set(hw.H_DEPENDENT_KEYS)
    b_only_state = b_changed == state_keys
    c_only_encoders = all(k.startswith(enc_keys_prefix) for k in c_changed) and bool(
        c_changed
    )

    model = build_cell("B", 0).to(device)
    counts = model.module_call_counts(batch)
    refresh_probe = hw._refresh_probe(build_cell("B", 0).to(device), batch)

    model.train()
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    grad_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    grad_nonzero = any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in model.parameters()
    )
    model.zero_grad(set_to_none=True)

    checks = {
        "cell_A_bit_identical_to_frozen_H64": bool(a_matches_h64),
        "cell_D_bit_identical_to_frozen_H96": bool(d_matches_h96),
        "cell_C_changes_only_encoder_hidden_tensors": bool(c_only_encoders),
        "cell_B_changes_only_state_tensors": bool(b_only_state),
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
        "param_counts_exact": all(row["matches_expected"] for row in params()["rows"]),
        "forward_finite": bool(torch.isfinite(out).all()),
        "backward_finite_nonzero": bool(grad_finite) and bool(grad_nonzero),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "a_vs_h64_max_diff": float(max_state_diff(a, h64)),
        "d_vs_h96_max_diff": float(max_state_diff(d, h96)),
        "cell_B_changed_keys": sorted(b_changed),
        "cell_C_changed_keys": sorted(c_changed),
        "refresh_probe": refresh_probe,
        "module_call_counts": counts,
        "split": "first 128 official-valid molecules",
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "sanity.json", payload)
    if not payload["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"capacity decomposition sanity failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# training / soup
# ---------------------------------------------------------------------------


def _valid_loader():
    _train_data, valid_data, _ = rec.load_encoded()
    return zpp._make_loader(list(valid_data), 128, False, 0)


def train(cell: str, seed: int, device: str = "cpu") -> dict[str, Any]:
    spec = CELLS[cell]
    tag = tag_for(cell)
    train_data, valid_data, _ = rec.load_encoded()
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    snapshot_dir = SNAPSHOT_DIR / f"{tag}_seed{seed}"
    rss_before = hw._rss_peak_kb()
    started = time.perf_counter()
    expected_total = int(EXPECTED_PARAMS[cell])
    build_fn = lambda s: build_cell(cell, s)  # noqa: E731
    try:
        summary = shead.train_model(
            build_fn=build_fn,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=tag,
            save_state=True,
            real_batch_identity=False,
            expected_total=expected_total,
            snapshot_dir=snapshot_dir,
            device=device,
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original
    elapsed = float(time.perf_counter() - started)

    summary = dict(summary)
    summary["config_name"] = f"cell_{cell}"
    summary["cell"] = cell
    summary["h_dim"] = int(spec["h"])
    summary["patch_encoder_hidden"] = int(spec["patch_encoder_hidden"])
    summary["global_encoder_hidden"] = int(spec["global_encoder_hidden"])
    summary["q_dim"] = Q_DIM
    summary["epoch_time_s"] = float(summary["wall_clock_s"]) / max(
        int(summary["epochs_run"]), 1
    )
    summary["peak_rss_kb"] = hw._rss_peak_kb()
    summary["peak_rss_delta_kb"] = hw._rss_peak_kb() - int(rss_before)
    summary["outer_wall_clock_s"] = elapsed
    summary["platform"] = platform.platform()
    hw._write_json(RUNS_DIR / f"{tag}_seed{seed}.json", summary)

    soup(cell, seed)
    return summary


def soup(cell: str, seed: int) -> dict[str, Any]:
    tag = tag_for(cell)
    summary = hw._read_json(RUNS_DIR / f"{tag}_seed{seed}.json")
    top = vd._top5_epochs(summary)
    soup_state = vd.build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    loader = _valid_loader()
    model = build_cell(cell, seed)
    selection_path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    targets, best_preds = vd._predict_state(
        model, torch.load(selection_path, map_location="cpu", weights_only=True), loader
    )
    _t2, soup_preds = vd._predict_state(model, soup_state, loader)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cell": cell,
        "tag": tag,
        "seed": int(seed),
        "h_dim": int(CELLS[cell]["h"]),
        "patch_encoder_hidden": int(CELLS[cell]["patch_encoder_hidden"]),
        "global_encoder_hidden": int(CELLS[cell]["global_encoder_hidden"]),
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
    hw._write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


def train_queue(cells: Sequence[str], seeds: Sequence[int], device: str) -> None:
    for cell in cells:
        for seed in seeds:
            if (RUNS_DIR / f"{tag_for(cell)}_seed{seed}.json").exists():
                print(f"skip existing {cell} seed{seed}", flush=True)
                continue
            print(f"=== train {cell} seed{seed} device={device} ===", flush=True)
            train(str(cell), int(seed), device=device)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(torch.float32).cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def repro(cell: str, seed: int, epochs: int, device: str, run_tag: str = "") -> dict[str, Any]:
    """Short GPU reproducibility sanity: same seed, same short protocol.

    Runs a fixed small number of epochs in an isolated ``repro`` result
    subtree (never the formal run paths) and records the validation curve and
    a hash of the selection state so two GPUs / two repeats can be compared.
    ``run_tag`` keeps two concurrent workers from sharing files.
    """
    global CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR
    tag = str(run_tag or "default").replace("/", "_")
    saved = (CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR)
    root = RESULTS_DIR / "repro" / tag
    CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR = (
        root / "curves",
        root / "states",
        root / "runs",
        root / "snapshots",
        root / "soup_states",
    )
    original = dict(shead.OPTIMIZED_PROTOCOL)
    shead.OPTIMIZED_PROTOCOL = {
        **original,
        "max_epochs": int(epochs),
        "patience": int(epochs),
    }
    try:
        summary = train(cell, seed, device=device)
        state = torch.load(
            STATE_DIR / f"{tag_for(cell)}_seed{seed}_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        state_hash = _state_sha256(state)
        curve_path = CURVE_DIR / f"{tag_for(cell)}_seed{seed}_curve.csv"
        valid_curve: list[float] = []
        if curve_path.exists():
            for line in curve_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split(",")
                if len(parts) > 2:
                    valid_curve.append(float(parts[2]))
    finally:
        shead.OPTIMIZED_PROTOCOL = original
        CURVE_DIR, STATE_DIR, RUNS_DIR, SNAPSHOT_DIR, SOUP_DIR = saved
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "gpu_reproducibility_sanity",
        "cell": cell,
        "seed": int(seed),
        "epochs": int(epochs),
        "run_tag": tag,
        "device": str(device),
        "best_valid_mae": float(summary["best_valid_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_run": int(summary["epochs_run"]),
        "wall_clock_s": float(summary["wall_clock_s"]),
        "valid_mae_curve": valid_curve,
        "selection_state_sha256": state_hash,
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / f"repro_{tag}_{cell}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics() -> dict[str, Any]:
    batch = hw._first_batch(list(rec.load_encoded()[1]), torch.device("cpu"))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "centre_state": {},
        "prediction_disagreement": {},
        "official_test_loaded": False,
    }
    raw: dict[str, dict[int, float]] = {}
    soupv: dict[str, dict[int, float]] = {}
    best_preds: dict[str, dict[int, np.ndarray]] = {}
    soup_preds: dict[str, dict[int, np.ndarray]] = {}
    for cell in CELL_ORDER:
        tag = tag_for(cell)
        raw[cell], soupv[cell] = {}, {}
        best_preds[cell], soup_preds[cell] = {}, {}
        payload["centre_state"][cell] = {}
        for seed in SEEDS:
            sp = soup(cell, seed)
            raw[cell][seed] = float(sp["best_checkpoint_valid_mae"])
            soupv[cell][seed] = float(sp["top5_soup_valid_mae"])
            best_preds[cell][seed] = np.asarray(sp["best_predictions"])
            soup_preds[cell][seed] = np.asarray(sp["soup_predictions"])
            model = build_cell(cell, seed)
            model.load_state_dict(
                torch.load(
                    STATE_DIR / f"{tag}_seed{seed}_selection_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            payload["centre_state"][cell][str(seed)] = hw._centre_diagnostics(model, batch)
        payload["prediction_disagreement"][cell] = {
            "raw_seed0_seed1": float(
                np.mean(np.abs(best_preds[cell][0] - best_preds[cell][1]))
            ),
            "soup_seed0_seed1": float(
                np.mean(np.abs(soup_preds[cell][0] - soup_preds[cell][1]))
            ),
        }
    payload["raw_best_valid"] = {
        cell: {str(s): raw[cell][s] for s in SEEDS} for cell in CELL_ORDER
    }
    payload["soup_valid"] = {
        cell: {str(s): soupv[cell][s] for s in SEEDS} for cell in CELL_ORDER
    }
    payload["raw_2seed_mean"] = {
        cell: float(np.mean([raw[cell][s] for s in SEEDS])) for cell in CELL_ORDER
    }
    payload["soup_2seed_mean"] = {
        cell: float(np.mean([soupv[cell][s] for s in SEEDS])) for cell in CELL_ORDER
    }
    hw._write_json(RESULTS_DIR / "diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision -- main effects / interaction
# ---------------------------------------------------------------------------


def decide() -> dict[str, Any]:
    soup_mean: dict[str, float] = {}
    raw_mean: dict[str, float] = {}
    runs: dict[str, Any] = {}
    for cell in CELL_ORDER:
        sp = [soup(cell, seed) for seed in SEEDS]
        soup_vals = [float(x["top5_soup_valid_mae"]) for x in sp]
        raw_vals = [float(x["best_checkpoint_valid_mae"]) for x in sp]
        soup_mean[cell] = float(np.mean(soup_vals))
        raw_mean[cell] = float(np.mean(raw_vals))
        summary = hw._read_json(RUNS_DIR / f"{tag_for(cell)}_seed0.json")
        runs[cell] = {
            "params": EXPECTED_PARAMS[cell],
            "soup_per_seed": soup_vals,
            "raw_per_seed": raw_vals,
            "best_epochs": [int(x["best_epoch"]) for x in sp],
            "epoch_time_s": float(summary.get("epoch_time_s", float("nan"))),
        }

    # Main effects, defined so that a POSITIVE value means the larger-capacity
    # cell is better (lower MAE): effect = MAE(smaller) - MAE(larger).
    centre_at_narrow = soup_mean["A"] - soup_mean["B"]  # A -> B
    centre_at_wide = soup_mean["C"] - soup_mean["D"]  # C -> D
    centre_main = 0.5 * (centre_at_narrow + centre_at_wide)
    encoder_at_narrow_centre = soup_mean["A"] - soup_mean["C"]  # A -> C
    encoder_at_wide_centre = soup_mean["B"] - soup_mean["D"]  # B -> D
    encoder_main = 0.5 * (encoder_at_narrow_centre + encoder_at_wide_centre)
    interaction = (soup_mean["D"] - soup_mean["C"]) - (
        soup_mean["B"] - soup_mean["A"]
    )
    interaction_alt = (soup_mean["D"] - soup_mean["B"]) - (
        soup_mean["C"] - soup_mean["A"]
    )

    best_cell = min(CELL_ORDER, key=lambda c: soup_mean[c])
    # Efficiency: best validation per parameter, tie-break by lower MAE.
    efficiency = min(CELL_ORDER, key=lambda c: soup_mean[c] / EXPECTED_PARAMS[c])

    if centre_main > 0.002 and encoder_main < 0.001:
        verdict = "CENTRE_WIDTH_DOMINANT"
    elif encoder_main > 0.002 and centre_main < 0.001:
        verdict = "ENCODER_CAPACITY_DOMINANT"
    elif centre_main > 0.001 and encoder_main > 0.001:
        verdict = "BOTH_CONTRIBUTE"
    elif abs(centre_main) < 0.001 and abs(encoder_main) < 0.001:
        verdict = "NEITHER_AXIS"
    else:
        verdict = "INCONCLUSIVE"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "soup_2seed_mean": soup_mean,
        "raw_2seed_mean": raw_mean,
        "runs": runs,
        "effects": {
            "centre_width_at_narrow_encoders_A_to_B": float(centre_at_narrow),
            "centre_width_at_wide_encoders_C_to_D": float(centre_at_wide),
            "centre_width_main_effect": float(centre_main),
            "encoder_capacity_at_narrow_centre_A_to_C": float(encoder_at_narrow_centre),
            "encoder_capacity_at_wide_centre_B_to_D": float(encoder_at_wide_centre),
            "encoder_capacity_main_effect": float(encoder_main),
            "interaction_D_minus_C_minus_B_plus_A": float(interaction),
            "interaction_alt_D_minus_B_minus_C_plus_A": float(interaction_alt),
        },
        "best_cell": best_cell,
        "best_cell_soup_2seed_mean": float(soup_mean[best_cell]),
        "best_validation_per_param_cell": efficiency,
        "verdict": verdict,
        "official_test_loaded": False,
    }
    hw._write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return hw._read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_accounting": _maybe("parameter_accounting.json"),
        "sanity": (lambda d: None if d is None else d["checks"])(_maybe("sanity.json")),
        "diagnostics": _maybe("diagnostics.json"),
        "decision": _maybe("decision.json"),
    }
    hw._write_json(RESULTS_DIR / "report.json", payload)
    return payload


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
            "train_queue",
            "repro",
            "soup",
            "diagnostics",
            "decide",
            "report",
        ],
    )
    parser.add_argument("--cell", type=str, default="A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells", type=str, default="A,B")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-tag", type=str, default="")
    args = parser.parse_args(argv)

    torch.set_num_threads(4)
    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if args.stage == "sanity":
        print(json.dumps(sanity()["checks"], indent=2, default=str), flush=True)
    if args.stage == "train":
        print(json.dumps(train(args.cell, args.seed, args.device), indent=2, default=str))
    if args.stage == "train_queue":
        train_queue(args.cells.split(","), [int(s) for s in args.seeds.split(",")], args.device)
    if args.stage == "repro":
        print(
            json.dumps(
                repro(args.cell, args.seed, args.epochs, args.device, args.run_tag),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "soup":
        print(json.dumps(soup(args.cell, args.seed), indent=2, default=str), flush=True)
    if args.stage == "diagnostics":
        print(json.dumps(diagnostics(), indent=2, default=str), flush=True)
    if args.stage == "decide":
        print(json.dumps(decide(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
