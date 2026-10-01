"""E2E-DictEnv-Scale-v1 runner — unified Small/Full task-dictionary scaling.

Round ``e2e_dictenv_scale_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_scale_v1.py``.
Pre-registration:
``tracks/ksvd/notes/zinc_e2e_dictenv_scale_v1_preregistration.md``.
Delivered handoff package (vendored):
``tracks/ksvd/experiments/luyin16/dictionary_scaling/v1_20261001/``.

Stages
------
``references audit preflight correctness timing smoke train interventions analysis chain``

Exactly one from-scratch **Full** (m=3, 408,651 parameters) seed-0 trajectory on
CPU, and only when the real forward/backward timing predicts the frozen
320-epoch budget fits in 4 hours.  Small (m=1) is never retrained: the
correctness stage builds it only to prove parameter / function identity with
the frozen ``LatentBridgeSEM108`` candidate.  The official ZINC **test** split
is never instantiated; every payload records ``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = sc.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_scale_v1"
AUDIT_DIR = RESULTS_DIR / "audit"
MECHANISM_DIR = RESULTS_DIR / "mechanism"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "zinc_e2e_dictenv_scale_v1_preregistration.md"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
REFERENCE_DIR = TRACK_ROOT / "experiments/luyin16/dictionary_scaling/v1_20261001"
#: read-only audit artifacts from the immediately previous (frozen) interface.
PREVIOUS_ROUND_DIR = TRACK_ROOT / "results/e2e_dictenv_latent_bridge_v1"
PREVIOUS_AUDIT_DIR = PREVIOUS_ROUND_DIR / "audit"

TAG = "SCALE-FULL"
SMALL_TAG = "SCALE-SMALL"
THREADS = 8
TRAIN_EPOCHS = 320
SEED = 0
SCALE_SEED = 0
SMOKE_EPOCHS = 3  # 3 x 8 = 24 optimizer steps on 1024 train molecules
SMOKE_SUBSET = 1024
SHUFFLE_SEEDS = audit.SHUFFLE_SEEDS  # (101, 202, 303, 404, 505)
DIAG_EPOCHS: tuple[int, ...] = (1, 80, 160, 240, 320)

#: pre-registered Full absolute-MAE resource bands (frozen before training).
BANDS: tuple[tuple[float, str], ...] = (
    (0.110, "SCALE_STRONG_SINGLE_SEED_SIGNAL"),
    (0.115, "SCALE_PROMISING_SINGLE_SEED"),
    (0.120, "SCALE_LIMITED_SIGNAL_CLOSE_CAPACITY_ROUTE"),
    (float("inf"), "SCALE_STOP_CAPACITY_ROUTE"),
)
#: read-only background references (never rerun, unmatched).
SEM108_BACKGROUND_SOUP = 0.123704927947314
BRIDGE_BACKGROUND_SOUP = 0.12105831989174476

#: the formal run budget (training + routine evaluation) and timing safety.
FORMAL_BUDGET_S = 4.0 * 3600.0
TIMING_SAFETY = 1.10
TIMING_TRAIN_BATCHES = 6
TIMING_EVAL_BATCHES = 3
TIMING_EIGEN_REPEATS = 20

#: expected Full task-path widths (pre-registered).
EXPECTED_WIDTHS_FULL = {
    "h": 144,
    "E": 144,
    "alpha": 288,
    "u": 48,
    "pair": 48,
    "reader_input": 814,
}

#: acceptance tolerances (pre-registration section 5).
GATE_SMALL_FORWARD_MAX = 1.0e-5
GATE_CONTAINMENT_MAX = 1.0e-4
GATE_IDENTITY_RELATIVE_L2 = 1.0e-6
GATE_FORMAL_RELATIVE_SQ = 0.05
GATE_RELABEL_TOL = 1.0e-4
GATE_BATCH_OFFSET_TOL = 1.0e-5
GATE_ENDPOINT_SWAP_TOL = 1.0e-6
GATE_BRIDGE_ZERO = 0.005

_write_json = base._write_json
_read_json = base._read_json
_write_csv = base._write_csv
_git_commit = base._git_commit
_sha256_file = base._sha256_file
_clone_batch = base._clone_batch
_synthetic_batch = base._synthetic_batch
_load_parent_subspace = base._load_parent_subspace
_dictionary_tensor = base._dictionary_tensor
_real_batch = base._real_batch


def _model_factory(
    dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace
) -> sc.LatentScaleSEM108:
    return sc.build_scale_model(dictionary, int(seed), subspace, sc.FULL, scale_seed=SCALE_SEED)


def _small_factory(
    dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace
) -> sc.LatentScaleSEM108:
    return sc.build_small_scale_model(dictionary, int(seed), subspace)


def _band(mae: float) -> str:
    for threshold, label in BANDS:
        if float(mae) <= threshold:
            return label
    return BANDS[-1][1]


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, AUDIT_DIR, MECHANISM_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.{digits}f}"


# ---------------------------------------------------------------------------
# shared model / data helpers
# ---------------------------------------------------------------------------


def _soup_state() -> dict[str, torch.Tensor]:
    state = torch.load(
        CHECKPOINT_DIR / f"{TAG}-seed0_soup_state.pt", map_location="cpu", weights_only=False
    )
    return {key: value.float() for key, value in state.items()}


def _soup_model() -> sc.LatentScaleSEM108:
    model = _model_factory(_dictionary_tensor(), SEED, _load_parent_subspace())
    model.load_state_dict(_soup_state())
    return model


def _gradient_diagnostics(
    model: sc.LatentScaleSEM108, batch: Any, mask: Any, lam: float
) -> dict[str, Any]:
    """Full parent-loss gradients (structural reconstruction + MAE)."""
    was_training = bool(model.training)
    model.eval()
    prediction, aux = model(batch, mask=mask, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(lam) * rec
    model.zero_grad(set_to_none=True)
    loss.backward()

    def _grad(parameter: torch.nn.Parameter) -> float:
        return float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0

    alpha = aux["coord"][:, model.common_dim :].detach()
    out = {
        "loss": float(loss.detach()),
        "reconstruction": float(rec.detach()),
        "grad_D": _grad(model.D),
        "grad_fusion_W1": _grad(model.fusion[0].weight),
        "grad_fusion_W2": _grad(model.fusion[2].weight),
        "grad_W_A_S": _grad(model.W_A_S),
        "grad_W_E_S": _grad(model.W_E_S),
        "grad_pair_projection": _grad(model.pair_projection.weight),
        "grad_pair_encoder_W1": _grad(model.pair_encoder.layers[0].weight),
        "grad_pair_encoder_W2": _grad(model.pair_encoder.layers[4].weight),
        "grad_reader_W1": _grad(model.reader.net[0].weight),
        "grad_bridge_D_L": _grad(model.local_dictionary_bridge.D_L),
        "grad_bridge_V_L": _grad(model.local_dictionary_bridge.V_L),
        "coord_nnz_per_row_mean": float((alpha != 0).double().sum(dim=1).mean()),
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def _task_gradients(
    model: sc.LatentScaleSEM108, batch: Any, mask: Any
) -> dict[str, Any]:
    """MAE-only gradients (no structural reconstruction term)."""
    was_training = bool(model.training)
    model.eval()
    prediction, _aux = model(batch, mask=mask, return_aux=True)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()

    def _grad(parameter: torch.nn.Parameter) -> float:
        return float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0

    out = {
        "task_loss": float(loss.detach()),
        "task_grad_bridge_D_L": _grad(model.local_dictionary_bridge.D_L),
        "task_grad_bridge_V_L": _grad(model.local_dictionary_bridge.V_L),
        "task_grad_fusion_W1": _grad(model.fusion[0].weight),
        "task_grad_fusion_W2": _grad(model.fusion[2].weight),
        "task_grad_pair_projection": _grad(model.pair_projection.weight),
        "task_grad_pair_encoder_W1": _grad(model.pair_encoder.layers[0].weight),
        "task_grad_pair_encoder_W2": _grad(model.pair_encoder.layers[4].weight),
        "task_grad_reader_W1": _grad(model.reader.net[0].weight),
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def _evaluate_detailed(
    model: sc.LatentScaleSEM108, data_list: Sequence[Any], mask: Any
) -> dict[str, Any]:
    """MAE plus the full prediction vector (fixed order, no shuffling)."""
    device = audit.attach_cpu(THREADS)
    loader = p1.make_env_loader(
        data_list, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch, mask=mask).view(-1).cpu())
            targets.append(batch.y.view(-1).cpu())
    prediction = torch.cat(predictions) if predictions else torch.zeros(0)
    target = torch.cat(targets) if targets else torch.zeros(0)
    return {
        "mae": float((prediction - target).abs().mean()) if target.numel() else float("nan"),
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# stage: references
# ---------------------------------------------------------------------------


def stage_references() -> dict[str, Any]:
    _ensure_dirs()

    def _read(relative: str) -> dict[str, Any] | None:
        path = REPO_ROOT / relative
        return _read_json(path) if path.exists() else None

    sem_soup = _read("tracks/ksvd/results/e2e_dictenv_sem108_v1/soup.json")
    bridge_soup = _read("tracks/ksvd/results/e2e_dictenv_latent_bridge_v1/soup.json")
    bridge_run = _read("tracks/ksvd/results/e2e_dictenv_latent_bridge_v1/run_seed0.json")
    cssd_final = _read("tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json")
    clean = _read(
        "tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json"
    )
    t1 = _read("tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json")
    p1_run = _read("tracks/ksvd/results/e2e_dictenv_p1/sparse_seed0.json")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "policy": (
            "read-only durable artifacts; no historical model is retrained; every comparison "
            "is historical / unmatched / contextual only"
        ),
        "vendored_reference_package": str(REFERENCE_DIR.relative_to(REPO_ROOT)),
        "references": {
            "latent-bridge-seed0-background": (
                None
                if bridge_soup is None
                else {
                    "value": float(bridge_soup["soup_valid_mae"]),
                    "members": list(bridge_soup.get("members", [])),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_latent_bridge_v1/soup.json",
                    "status": (
                        "immediately previous round on the same split and the same local object; "
                        "background only, not a matched control (Full adds width and parameters)"
                    ),
                }
            ),
            "Sem108-seed0-background": (
                None
                if sem_soup is None
                else {
                    "value": float(sem_soup["soup_valid_mae"]),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_sem108_v1/soup.json",
                    "status": "unmatched background only",
                }
            ),
            "CSSD-q1-seed0": (
                None
                if cssd_final is None
                else {
                    "value": float(cssd_final["soup"]["soup_valid_mae"]),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json",
                    "status": "not_rerun_unmatched_context",
                }
            ),
            "FINAL-CLEAN-C6-seed0": (
                None
                if clean is None
                else {
                    "value": float(clean["baseline_valid_mae"]),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json",
                    "status": "not_rerun_unmatched_context",
                }
            ),
            "T1-tuned-seed0": (
                None
                if t1 is None
                else {
                    "value": float(t1["best_tuned_soup_valid_mae"]),
                    "candidate_id": str(t1.get("candidate_id", "")),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json",
                    "status": "not_rerun_unmatched_context",
                }
            ),
            "P1-sparse-seed0": (
                None
                if p1_run is None
                else {
                    "value": float(p1_run["soup"]["soup_valid_mae"]),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_p1/sparse_seed0.json",
                    "status": "not_rerun_unmatched_context",
                }
            ),
        },
        "previous_round": (
            None
            if bridge_run is None
            else {
                "soup_valid_mae": float(bridge_run["soup"]["soup_valid_mae"]),
                "soup_train_mae_at_reference_eval": 0.055133,
                "artifact": "tracks/ksvd/results/e2e_dictenv_latent_bridge_v1/run_seed0.json",
                "status": "background only",
            }
        ),
    }
    sc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "historical_references.json", payload)
    bridge_value = payload["references"]["latent-bridge-seed0-background"]
    print(
        "[references] latent-bridge background "
        + ("n/a" if bridge_value is None else f"{bridge_value['value']:.6f}"),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: audit (reuse of the frozen-interface audits; no retraining)
# ---------------------------------------------------------------------------


def stage_audit(force: bool = False) -> dict[str, Any]:
    """Reuse the frozen Sem108 interface audits read-only (interface unchanged).

    The identity / anchor-relation / T1 audits concern the unchanged
    ``[Sem108 ; size2]`` interface and the unchanged parent body.  They are
    copied byte-identically from the previous round and recorded with source
    paths and SHA-256s; nothing is recomputed and no historical model is
    retrained.
    """
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    names = ("sem108_identity.json", "anchor_relation.json", "t1_block_ablation.json", "audit_decision.json")
    sources: dict[str, Any] = {}
    for name in names:
        source = PREVIOUS_AUDIT_DIR / name
        if not source.exists():
            raise FileNotFoundError(
                f"frozen-interface audit artifact missing: {source}; this round reuses the "
                "previous round's audits and never recomputes them"
            )
        target = AUDIT_DIR / name
        if force or not target.exists() or target.read_bytes() != source.read_bytes():
            target.write_bytes(source.read_bytes())
        sources[name] = {
            "source": str(source.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(target),
        }
    reused = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "policy": (
            "the [Sem108 ; size2] interface, the structural dictionary, the bindings and the "
            "mask are unchanged by this round; the previous round's audits are reused "
            "byte-identically and are not recomputed"
        ),
        "artifacts": sources,
        "previous_round": str(PREVIOUS_ROUND_DIR.relative_to(REPO_ROOT)),
    }
    _write_json(AUDIT_DIR / "audit_reuse.json", reused)
    decision = _read_json(AUDIT_DIR / "audit_decision.json")
    print(
        f"[audit] reused=4 decision={decision['decision']} "
        f"t1={_read_json(AUDIT_DIR / 't1_block_ablation.json')['status']}",
        flush=True,
    )
    if decision.get("decision") != "PROCEED":
        raise RuntimeError(f"reused interface audit did not authorise the round: {decision}")
    return reused


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    if not (AUDIT_DIR / "audit_reuse.json").exists():
        raise RuntimeError("audit reuse record missing; run the audit stage first")
    decision = _read_json(AUDIT_DIR / "audit_decision.json")
    if decision.get("decision") != "PROCEED":
        raise RuntimeError("Phase-A audit did not authorise the formal run")
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    prereg_sha = _sha256_file(PREREG_PATH)

    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    reference_model = lb.build_latent_bridge_model(dictionary, SEED, subspace)
    small = _small_factory(dictionary, SEED, subspace)
    full = _model_factory(dictionary, SEED, subspace)
    parent = cssd.build_cssd_model(dictionary, SEED, subspace)
    parent_params = int(sum(parameter.numel() for parameter in parent.parameters()))

    state_reference = reference_model.state_dict()
    state_small = small.state_dict()
    if sorted(state_reference) != sorted(state_small):
        raise RuntimeError("Small scale model state dict keys differ from LatentBridgeSEM108")
    shared_mismatch = [
        key for key in state_reference if not torch.equal(state_reference[key], state_small[key])
    ]
    if shared_mismatch:
        raise RuntimeError(f"Small scale model differs from LatentBridgeSEM108: {shared_mismatch[:8]}")

    small_audit = sc.scale_parameter_audit(small)
    full_audit = sc.scale_parameter_audit(full)
    fixed = sc.compare_fixed_modules(small, full)
    if not small_audit["parameter_exact"] or small_audit["actual_parameters"] != 106925:
        raise RuntimeError(f"Small parameter contract violated: {small_audit}")
    if not full_audit["parameter_exact"] or full_audit["actual_parameters"] != 408651:
        raise RuntimeError(f"Full parameter contract violated: {full_audit}")
    if not full_audit["within_budget"]:
        raise RuntimeError(f"Full model exceeds the task-path budget: {full_audit}")
    if not fixed["identical"]:
        raise RuntimeError(f"fixed modules changed with the multiplier: {fixed}")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": prereg_sha,
        },
        "vendored_reference_package": str(REFERENCE_DIR.relative_to(REPO_ROOT)),
        "specs": {
            "small": sc.SMALL.as_dict(),
            "full": sc.FULL.as_dict(),
            "audit_only_m2": sc.AUDIT_ONLY_M2.as_dict(),
        },
        "parameter_audit": {
            "small": small_audit,
            "full": full_audit,
            "fixed_modules": fixed,
        },
        "small_identity_with_latent_bridge": {
            "state_dict_keys_equal": True,
            "value_mismatch": shared_mismatch,
            "latent_bridge_parameters": int(
                sum(parameter.numel() for parameter in reference_model.parameters())
            ),
        },
        "parent": {
            "config": cm.H1_CONFIG.as_dict(),
            "lambda_rec": float(cm.H1_LAMBDA),
            "c6_mask_identity": bool(cssd.CSSD_MASK is cm.C6_MASK),
            "c6_equivalence_check": bool(cm.c6_equivalence_check()),
            "subspace_kind": subspace.kind,
            "common_dim": int(subspace.q),
            "parent_params": parent_params,
        },
        "interface": {
            "input": "Sem108 fusion output h in R^d",
            "output": "E in R^d consumed by unary pooling and static pair computation",
            "graph_indices_accepted": False,
            "state_writeback": False,
            "residual_bypass": False,
            "new_raw_features": False,
        },
        "epochs": TRAIN_EPOCHS,
        "seed": SEED,
        "scale_seed": SCALE_SEED,
    }
    sc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration"])
    _write_json(RESULTS_DIR / "parameter_audit.json", payload["parameter_audit"])
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {prereg_sha[:12]} small={small_audit['actual_parameters']} "
        f"full={full_audit['actual_parameters']} fixed_identical={fixed['identical']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: correctness / acceptance audit (real train batch, zero training)
# ---------------------------------------------------------------------------


def acceptance_audit(
    *, subset: int = 32, threads: int = THREADS, seed: int = SEED
) -> dict[str, Any]:
    """Pre-registered zero-training acceptance audit on a real official-train batch.

    S1 Small identity, S2 Full parameter / width / fixed-module contract,
    S3 eval-mode function containment, S4 bridge identity mode + formal-init
    reconstruction, S5 task gradients and one update, S6 wiring / invariance /
    zero-code, S7 official-test blocker.
    """
    audit.attach_cpu(int(threads))
    torch.set_num_threads(int(threads))
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    reference = lb.build_latent_bridge_model(dictionary, seed, subspace)
    small = _small_factory(dictionary, seed, subspace)
    full = _model_factory(dictionary, seed, subspace)
    embedded_full = _model_factory(dictionary, seed, subspace)
    for model in (reference, small, full, embedded_full):
        model.eval()
    batch = _real_batch(subset=int(subset), split="train")
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)

    # -- S1: Small identity with the frozen LatentBridgeSEM108 ----------------
    state_reference = reference.state_dict()
    state_small = small.state_dict()
    key_mismatch = sorted(set(state_reference) ^ set(state_small))
    small_value_mismatch = [
        key
        for key in sorted(set(state_reference) & set(state_small))
        if not torch.equal(state_reference[key], state_small[key])
    ]
    with torch.no_grad():
        pred_reference_plain = reference(batch).view(-1).double()
        pred_small_plain = small(batch).view(-1).double()
        pred_reference_masked = reference(batch, mask=mask).view(-1).double()
        pred_small_masked = small(batch, mask=mask).view(-1).double()
    small_audit = sc.scale_parameter_audit(small)
    s1_plain = float((pred_reference_plain - pred_small_plain).abs().max())
    s1_masked = float((pred_reference_masked - pred_small_masked).abs().max())

    # -- S2: Full parameter / width / fixed-module contract -------------------
    full_audit = sc.scale_parameter_audit(full)
    widths = sc.capture_task_path_widths(full, batch, mask=mask)
    fixed = sc.compare_fixed_modules(small, full)

    # -- S3: eval-mode function containment (Small embedded into Full) --------
    containment = sc.containment_witness_delta(small, embedded_full, batch, mask=mask)

    # -- S4: identity mode + formal initialisation ----------------------------
    identity_full = _model_factory(dictionary, seed, subspace)
    identity_full.local_dictionary_bridge = lb.LatentDictionaryBridge(
        dim=sc.FULL.d,
        n_atoms=sc.FULL.k,
        lambda1=0.0,
        lambda2=0.0,
        steps=lb.BRIDGE_STEPS,
    )
    identity_full.eval()
    h = sc._capture_bridge_input(full, batch, mask)
    with torch.no_grad():
        e_formal = full.local_dictionary_bridge(h)
        e_identity = identity_full.local_dictionary_bridge(h)
        e_identity_repeat = identity_full.local_dictionary_bridge(h)
    identity_relative_l2 = float(torch.norm(e_identity - h) / torch.norm(h).clamp_min(lb.BRIDGE_EPS))
    formal_relative_sq = float(
        ((e_formal - h) ** 2).sum() / (h * h).sum().clamp_min(lb.BRIDGE_EPS)
    )
    per_row = ((e_formal - h) ** 2).sum(dim=1) / (h * h).sum(dim=1).clamp_min(lb.BRIDGE_EPS)
    formal_row_p95 = float(torch.quantile(per_row, 0.95))
    repeat_exact = bool(torch.equal(e_identity, e_identity_repeat))
    finite_codes = bool(torch.isfinite(e_formal).all() and torch.isfinite(e_identity).all())

    # -- S5: task gradients and one update ------------------------------------
    task_init = _task_gradients(full, batch, mask)
    full_init = _gradient_diagnostics(full, batch, mask, lam)
    update_model = _model_factory(dictionary, seed, subspace)
    update_model.load_state_dict(full.state_dict())
    d_before = update_model.local_dictionary_bridge.D_L.detach().clone()
    v_before = update_model.local_dictionary_bridge.V_L.detach().clone()
    update_model.eval()
    prediction_u, aux_u = update_model(batch, mask=mask, return_aux=True)
    loss_before = F.l1_loss(prediction_u.view(-1), batch.y.view(-1))
    optimizer = torch.optim.Adam(
        update_model.parameters(),
        lr=float(p2run.LEARNING_RATE),
        weight_decay=float(p2run.WEIGHT_DECAY),
    )
    optimizer.zero_grad()
    loss_before.backward()
    torch.nn.utils.clip_grad_norm_(update_model.parameters(), float(p2run.GRAD_CLIP))
    optimizer.step()
    with torch.no_grad():
        prediction_after, _aux_after = update_model(batch, mask=mask, return_aux=True)
        loss_after = F.l1_loss(prediction_after.view(-1), batch.y.view(-1))
    d_delta = float((update_model.local_dictionary_bridge.D_L.detach() - d_before).norm())
    v_delta = float((update_model.local_dictionary_bridge.V_L.detach() - v_before).norm())
    update_finite = bool(
        math.isfinite(float(loss_before))
        and math.isfinite(float(loss_after))
        and math.isfinite(d_delta)
        and math.isfinite(v_delta)
    )
    sgd_model = _model_factory(dictionary, seed, subspace)
    sgd_model.load_state_dict(full.state_dict())
    sgd_model.eval()
    prediction_s, _aux_s = sgd_model(batch, mask=mask, return_aux=True)
    sgd_loss = F.l1_loss(prediction_s.view(-1), batch.y.view(-1))
    sgd_model.zero_grad()
    sgd_loss.backward()
    with torch.no_grad():
        for parameter in sgd_model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1.0e-3)
        prediction_sgd, _aux_sgd = sgd_model(batch, mask=mask, return_aux=True)
        sgd_loss_after = F.l1_loss(prediction_sgd.view(-1), batch.y.view(-1))
    sgd_decreased = bool(float(sgd_loss_after) < float(sgd_loss))

    # -- S6: wiring, no bypass, invariance ------------------------------------
    call_counts: list[int] = []

    def _count_hook(_module: Any, _args: Any, _output: Any) -> None:
        call_counts.append(1)

    handle = full.local_dictionary_bridge.register_forward_hook(_count_hook)
    with torch.no_grad():
        full(batch, mask=mask)
    masked_calls = len(call_counts)
    call_counts.clear()
    with torch.no_grad():
        full(batch)
    plain_calls = len(call_counts)
    handle.remove()

    pair_inputs: dict[str, torch.Tensor] = {}
    pair_handle = full.pair_projection.register_forward_pre_hook(
        lambda _module, inputs: pair_inputs.__setitem__("value", inputs[0].detach())
    )
    with torch.no_grad():
        _pred_w, aux_w = full(batch, mask=mask, return_aux=True)
    pair_handle.remove()
    unary_reference = audit.pool_moments_masked(
        aux_w["E"], batch.batch, int(batch.global_context.shape[0]), mask.unary_zero_blocks, None
    )
    pair_reads_bridge = bool(
        "value" in pair_inputs and torch.equal(pair_inputs["value"], aux_w["E"].detach())
    )
    unary_reads_bridge = bool(torch.equal(unary_reference, aux_w["unary"].detach()))
    no_h_bypass = bool(float((e_formal - h).abs().max()) > 0.0)

    relabel = base._relabel_invariance(full, n_molecules=2)
    data_pair = p1run.load_split("valid", subset=2)
    batch_pair = p1.env_collate(data_pair)
    with torch.no_grad():
        env_pair = full.environments(full.code(batch_pair.dict_phi), batch_pair)
        env_single = torch.cat(
            [
                full.environments(full.code(item.dict_phi), p1.env_collate([item]))
                for item in data_pair
            ],
            dim=0,
        )
    batch_offset_max = float((env_pair - env_single).abs().max())

    swapped = _clone_batch(batch)
    swapped.env_bond_u = batch.env_bond_v.clone()
    swapped.env_bond_v = batch.env_bond_u.clone()
    with torch.no_grad():
        env_formal = full.environments(full.code(batch.dict_phi), batch)
        env_swap = full.environments(full.code(swapped.dict_phi), swapped)
    endpoint_swap_max = float((env_swap - env_formal).abs().max())

    relation_probe = _clone_batch(batch)
    relation_probe.pair_relation = batch.pair_relation + 7.0
    relation_probe.pair_bucket = (batch.pair_bucket + 1) % int(p2.DISTANCE_BUCKETS)
    with torch.no_grad():
        env_relation = full.environments(full.code(relation_probe.dict_phi), relation_probe)
    relation_no_writeback = float((env_relation - env_formal).abs().max())

    baseline = _evaluate_detailed(full, p1run.load_split("valid", subset=128), mask)
    full.set_bridge_intervention(zero_code=True)
    with torch.no_grad():
        zero_env = full.environments(full.code(batch.dict_phi), batch)
    zero_detailed = _evaluate_detailed(full, p1run.load_split("valid", subset=128), mask)
    full.clear_bridge_intervention()
    zero_env_abs = float(zero_env.abs().max())
    zero_delta_mae = float(zero_detailed["mae"] - baseline["mae"])
    zero_delta_pred_rms = float(
        torch.sqrt(((zero_detailed["prediction"] - baseline["prediction"]) ** 2).mean())
    )
    with torch.no_grad():
        prediction_finite = bool(torch.isfinite(full(batch, mask=mask)).all())

    # -- S7: official-test blocker --------------------------------------------
    blocker_raised = False
    try:
        sc.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocker_raised = True

    gates = {
        "S1_small_identity_with_latent_bridge": {
            "state_dict_key_mismatch": key_mismatch,
            "state_dict_value_mismatch": small_value_mismatch,
            "plain_prediction_max_abs_diff": s1_plain,
            "masked_prediction_max_abs_diff": s1_masked,
            "tolerance": GATE_SMALL_FORWARD_MAX,
            "small_parameters": int(small_audit["actual_parameters"]),
            "expected_small_parameters": 106925,
            "passed": bool(
                not key_mismatch
                and not small_value_mismatch
                and s1_plain <= GATE_SMALL_FORWARD_MAX
                and s1_masked <= GATE_SMALL_FORWARD_MAX
                and int(small_audit["actual_parameters"]) == 106925
            ),
        },
        "S2_full_parameter_width_contract": {
            **full_audit,
            "widths": widths,
            "expected_widths": dict(EXPECTED_WIDTHS_FULL),
            "fixed_modules": fixed,
            "passed": bool(
                full_audit["parameter_exact"]
                and int(full_audit["actual_parameters"]) == 408651
                and full_audit["within_budget"]
                and widths == EXPECTED_WIDTHS_FULL
                and fixed["identical"]
            ),
        },
        "S3_eval_mode_function_containment": {
            **containment,
            "tolerance": GATE_CONTAINMENT_MAX,
            "note": "eval-mode block-replication witness only; never a training initialiser",
            "passed": bool(
                containment["plain_prediction_max_delta"] <= GATE_CONTAINMENT_MAX
                and containment["masked_prediction_max_delta"] <= GATE_CONTAINMENT_MAX
            ),
        },
        "S4_identity_mode_and_formal_init": {
            "identity_relative_l2": identity_relative_l2,
            "identity_tolerance": GATE_IDENTITY_RELATIVE_L2,
            "formal_relative_squared_error": formal_relative_sq,
            "formal_row_error_p95": formal_row_p95,
            "formal_threshold": GATE_FORMAL_RELATIVE_SQ,
            "repeat_code_exact": repeat_exact,
            "finite": finite_codes,
            "passed": bool(
                identity_relative_l2 <= GATE_IDENTITY_RELATIVE_L2
                and formal_relative_sq <= GATE_FORMAL_RELATIVE_SQ
                and repeat_exact
                and finite_codes
            ),
        },
        "S5_task_gradients_and_update": {
            "task_init": task_init,
            "full_init": full_init,
            "adam_parameter_delta": {"D_L": d_delta, "V_L": v_delta},
            "sgd_loss_before": float(sgd_loss),
            "sgd_loss_after": float(sgd_loss_after),
            "sgd_step_decreased": sgd_decreased,
            "finite": update_finite,
            "passed": bool(
                task_init["task_grad_bridge_D_L"] > 0.0
                and task_init["task_grad_bridge_V_L"] > 0.0
                and task_init["task_grad_fusion_W1"] > 0.0
                and task_init["task_grad_pair_projection"] > 0.0
                and task_init["task_grad_pair_encoder_W1"] > 0.0
                and d_delta > 0.0
                and v_delta > 0.0
                and update_finite
                and sgd_decreased
            ),
        },
        "S6_wiring_no_bypass_invariance": {
            "masked_bridge_calls": int(masked_calls),
            "plain_bridge_calls": int(plain_calls),
            "unary_matches_bridge_output": unary_reads_bridge,
            "pair_projection_input_matches_bridge_output": pair_reads_bridge,
            "no_h_bypass_max_abs": float((e_formal - h).abs().max()),
            "no_h_bypass": no_h_bypass,
            "relabel": relabel,
            "relabel_tolerance": GATE_RELABEL_TOL,
            "batch_offset_max_abs": batch_offset_max,
            "batch_offset_tolerance": GATE_BATCH_OFFSET_TOL,
            "endpoint_swap_max_abs": endpoint_swap_max,
            "endpoint_swap_tolerance": GATE_ENDPOINT_SWAP_TOL,
            "relation_no_writeback_max_abs": relation_no_writeback,
            "zero_code_env_max_abs": zero_env_abs,
            "zero_code_delta_mae": zero_delta_mae,
            "zero_code_delta_pred_rms": zero_delta_pred_rms,
            "predictions_finite": prediction_finite,
            "passed": bool(
                masked_calls == 1
                and plain_calls == 1
                and unary_reads_bridge
                and pair_reads_bridge
                and no_h_bypass
                and float(relabel["max_abs_pred_diff"]) <= GATE_RELABEL_TOL
                and batch_offset_max <= GATE_BATCH_OFFSET_TOL
                and endpoint_swap_max <= GATE_ENDPOINT_SWAP_TOL
                and relation_no_writeback == 0.0
                and zero_env_abs == 0.0
                and prediction_finite
                and abs(zero_delta_mae) > 0.0
                and zero_delta_pred_rms > 0.0
            ),
        },
        "S7_official_test_blocker": {"raised": blocker_raised, "passed": blocker_raised},
    }
    all_passed = bool(all(bool(gate["passed"]) for gate in gates.values()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "scope": (
            "implementation-level acceptance audit on one real official-train batch; "
            "no training, no official valid selection, official test never loaded"
        ),
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "scale_seed": int(SCALE_SEED),
        "train_batch_molecules": int(subset),
        "official_test_loaded": False,
        "identity_mode_note": (
            "lambda1 = lambda2 = 0 with the initial V_L is an audit-only mode and is never a "
            "training or candidate configuration"
        ),
        "readout_layout": sc.readout_layout(sc.FULL),
        "gates": gates,
        "all_passed": all_passed,
    }
    sc.official_test_blocker(payload)
    return payload


def stage_correctness() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    payload = acceptance_audit()
    _write_json(RESULTS_DIR / "correctness.json", payload)
    print(
        f"[correctness] all_passed={payload['all_passed']} "
        f"S1_plain={payload['gates']['S1_small_identity_with_latent_bridge']['plain_prediction_max_abs_diff']:.2e} "
        f"S3_plain={payload['gates']['S3_eval_mode_function_containment']['plain_prediction_max_delta']:.2e} "
        f"S4_formal_sq={payload['gates']['S4_identity_mode_and_formal_init']['formal_relative_squared_error']:.5f}",
        flush=True,
    )
    if not payload["all_passed"]:
        failed = [name for name, gate in payload["gates"].items() if not gate["passed"]]
        raise RuntimeError(f"acceptance gates failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# stage: CPU timing (authorises or blocks the formal 320-epoch run)
# ---------------------------------------------------------------------------


def stage_timing(force: bool = False) -> dict[str, Any]:
    """Real train-batch forward/backward timing and a full-run budget forecast.

    The formal 320-epoch run is authorised only when the measured steady-state
    cost over the exact official batch counts (train 79 + valid 8 per epoch)
    times ``TIMING_SAFETY`` fits ``FORMAL_BUDGET_S``.  No inferred cost model is
    used for the decision.
    """
    _ensure_dirs()
    out = RESULTS_DIR / "timing.json"
    if out.exists() and not force:
        print("[timing] cache hit", flush=True)
        return _read_json(out)
    sem.cpu_only_guard(torch.device("cpu"))
    audit.attach_cpu(THREADS)
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()

    started = time.perf_counter()
    train_data = p1run.load_split("train")
    train_load_s = float(time.perf_counter() - started)
    started = time.perf_counter()
    valid_data = p1run.load_split("valid")
    valid_load_s = float(time.perf_counter() - started)

    model = _model_factory(dictionary, SEED, subspace)
    device = audit.attach_cpu(THREADS)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    train_loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, SEED + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)

    train_steps: list[float] = []
    model.train()
    iterator = iter(train_loader)
    for index in range(TIMING_TRAIN_BATCHES):
        batch = next(iterator).to(device)
        step_started = time.perf_counter()
        prediction, aux = model(batch, mask=mask, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
        optimizer.step()
        train_steps.append(float(time.perf_counter() - step_started))
    del iterator

    eval_loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    eval_steps: list[float] = []
    model.eval()
    iterator = iter(eval_loader)
    with torch.no_grad():
        for _ in range(TIMING_EVAL_BATCHES):
            batch = next(iterator).to(device)
            step_started = time.perf_counter()
            model(batch, mask=mask)
            eval_steps.append(float(time.perf_counter() - step_started))
    del iterator

    bridge = model.local_dictionary_bridge
    eigen_steps: list[float] = []
    dictionary_normalized = bridge.normalized_dictionary()
    for _ in range(TIMING_EIGEN_REPEATS):
        step_started = time.perf_counter()
        bridge.solver_step(dictionary_normalized)
        eigen_steps.append(float(time.perf_counter() - step_started))

    def _median(values: Sequence[float]) -> float:
        sorted_values = sorted(float(value) for value in values)
        middle = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return sorted_values[middle]
        return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])

    cold_train = float(train_steps[0])
    steady_train = float(_median(train_steps[2:] if len(train_steps) > 2 else train_steps))
    steady_eval = float(_median(eval_steps))
    eigen_median = float(_median(eigen_steps))
    n_train_batches = int(math.ceil(len(train_data) / int(p2run.BATCH_SIZE)))
    n_valid_batches = int(math.ceil(len(valid_data) / int(p2run.BATCH_SIZE)))
    predicted_epoch = n_train_batches * steady_train + n_valid_batches * steady_eval
    predicted_total = TRAIN_EPOCHS * predicted_epoch
    predicted_total_with_margin = predicted_total * float(TIMING_SAFETY)
    authorized = bool(predicted_total_with_margin <= FORMAL_BUDGET_S)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "torch_num_threads": int(torch.get_num_threads()),
        "official_test_loaded": False,
        "method": (
            "real official-train batches through the exact parent loss / optimizer / C6 mask; "
            "steady-state medians exclude the first two batches; the decision uses the exact "
            "official batch counts and none of the smoke numbers"
        ),
        "data": {
            "train_molecules": int(len(train_data)),
            "valid_molecules": int(len(valid_data)),
            "train_load_s": train_load_s,
            "valid_load_s": valid_load_s,
            "batch_size": int(p2run.BATCH_SIZE),
            "n_train_batches": n_train_batches,
            "n_valid_batches": n_valid_batches,
        },
        "train_step_seconds": {
            "cold": cold_train,
            "steady_median": steady_train,
            "samples": train_steps,
        },
        "eval_step_seconds": {"steady_median": steady_eval, "samples": eval_steps},
        "bridge_eigen_seconds": {"median": eigen_median, "samples": eigen_steps},
        "predicted": {
            "epoch_seconds": predicted_epoch,
            "total_320_epochs_seconds": predicted_total,
            "safety_factor": float(TIMING_SAFETY),
            "total_with_margin_seconds": predicted_total_with_margin,
            "budget_seconds": float(FORMAL_BUDGET_S),
        },
        "formal_run_authorized": authorized,
        "decision": (
            "AUTHORISED" if authorized else "NOT_RUN_RESOURCE_BUDGET"
        ),
        "note": (
            "CPU/8 threads confirmed by torch.get_num_threads(); no GPU is used and no remote "
            "host is contacted"
        ),
    }
    sc.official_test_blocker(payload)
    _write_json(out, payload)
    print(
        f"[timing] steady_train={steady_train:.4f}s steady_eval={steady_eval:.4f}s "
        f"predicted_total={predicted_total:.0f}s margin={predicted_total_with_margin:.0f}s "
        f"authorized={authorized}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: smoke (3 epochs, 1024 molecules, no valid)
# ---------------------------------------------------------------------------


def _smoke_loop(
    model: sc.LatentScaleSEM108,
    train_data: Sequence[Any],
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    device = audit.attach_cpu(THREADS)
    p2run._seed_everything(int(seed))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    curve: list[dict[str, Any]] = []
    epoch_times: list[float] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        task_sum = 0.0
        n_molecules = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_molecules += int(batch.y.numel())
        epoch_times.append(float(time.perf_counter() - epoch_started))
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": float(task_sum / max(n_molecules, 1)),
                "seconds": epoch_times[-1],
            }
        )
    return {
        "train_mae_curve": curve,
        "first_train_mae": float(curve[0]["train_mae"]),
        "final_train_mae": float(curve[-1]["train_mae"]),
        "epoch_seconds": epoch_times,
        "seconds_per_epoch": float(sum(epoch_times) / max(len(epoch_times), 1)),
        "n_epochs": int(epochs),
        "n_molecules": int(len(train_data)),
    }


def stage_smoke() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = p1run.load_split("train", subset=SMOKE_SUBSET)
    model = _model_factory(dictionary, SEED, subspace)
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    init_grads = _gradient_diagnostics(model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    init_task = _task_gradients(model, gradient_batch, cm.C6_MASK)
    started = time.perf_counter()
    result = _smoke_loop(model, train_data, epochs=SMOKE_EPOCHS, seed=SEED)
    wall = float(time.perf_counter() - started)
    post_grads = _gradient_diagnostics(model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    post_task = _task_gradients(model, gradient_batch, cm.C6_MASK)
    finite = bool(
        math.isfinite(result["final_train_mae"])
        and all(math.isfinite(row["train_mae"]) for row in result["train_mae_curve"])
    )
    task_keys = (
        "task_grad_bridge_D_L",
        "task_grad_bridge_V_L",
        "task_grad_fusion_W1",
        "task_grad_fusion_W2",
        "task_grad_pair_projection",
        "task_grad_pair_encoder_W1",
        "task_grad_pair_encoder_W2",
        "task_grad_reader_W1",
    )
    gradient_ok = bool(
        all(init_task[key] > 0.0 for key in task_keys)
        and all(post_task[key] > 0.0 for key in task_keys)
        and init_grads["grad_bridge_D_L"] > 0.0
        and init_grads["grad_bridge_V_L"] > 0.0
        and post_grads["grad_bridge_D_L"] > 0.0
        and post_grads["grad_bridge_V_L"] > 0.0
    )
    timing = _read_json(RESULTS_DIR / "timing.json") if (RESULTS_DIR / "timing.json").exists() else None
    predicted_smoke_epoch = None
    if timing is not None:
        n_batches = int(math.ceil(SMOKE_SUBSET / int(p2run.BATCH_SIZE)))
        predicted_smoke_epoch = n_batches * float(timing["train_step_seconds"]["steady_median"])
    steps = int(math.ceil(SMOKE_SUBSET / int(p2run.BATCH_SIZE))) * SMOKE_EPOCHS
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "epochs": SMOKE_EPOCHS,
        "subset": SMOKE_SUBSET,
        "optimizer_steps": int(steps),
        "optimizer_step_budget": 100,
        "valid_accessed": False,
        "finite": finite,
        "gradient_ok": gradient_ok,
        "init_gradients": init_grads,
        "post_gradients": post_grads,
        "init_task_gradients": init_task,
        "post_task_gradients": post_task,
        "first_train_mae": result["first_train_mae"],
        "final_train_mae": result["final_train_mae"],
        "train_mae_curve": result["train_mae_curve"],
        "wall_clock_s": wall,
        "seconds_per_epoch": result["seconds_per_epoch"],
        "predicted_smoke_epoch_seconds": predicted_smoke_epoch,
        "passed": bool(finite and gradient_ok),
    }
    sc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "smoke.json", payload)
    print(
        f"[smoke] steps={steps} finite={finite} gradient_ok={gradient_ok} "
        f"final_train={result['final_train_mae']:.6f} wall={wall:.1f}s",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError("smoke failed (non-finite loss or dead task-path gradients)")
    return payload


# ---------------------------------------------------------------------------
# stage: train (exactly one Full seed-0 trajectory, 320 epochs)
# ---------------------------------------------------------------------------


def _scale_diag_row(
    model: sc.LatentScaleSEM108, batch: Any, initial_dictionary: torch.Tensor
) -> dict[str, Any]:
    grads = _gradient_diagnostics(model, batch, cm.C6_MASK, cm.H1_LAMBDA)
    task = _task_gradients(model, batch, cm.C6_MASK)
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        usage = base._usage_from_coord(coord, model.common_dim)
        h = sc._capture_bridge_input(model, batch, cm.C6_MASK)
        _e, bridge_aux = model.local_dictionary_bridge(h, return_aux=True)
    bridge_parameter = model.local_dictionary_bridge
    return {
        **grads,
        **task,
        "dictionary": usage,
        "dictionary_movement_frobenius": float(
            (model.D.detach().cpu() - initial_dictionary).norm()
        ),
        "bridge": {
            "code": lb.bridge_code_stats(bridge_aux["alpha"]),
            "reconstruction": lb.bridge_reconstruction_diagnostic(model, bridge_aux),
            "D_L_movement_frobenius": float(
                (bridge_parameter.D_L.detach().cpu() - bridge_parameter.D_init.cpu()).norm()
            ),
            "V_L_movement_frobenius": float(
                (bridge_parameter.V_L.detach().cpu() - bridge_parameter.V_init.cpu()).norm()
            ),
        },
        "widths": sc.capture_task_path_widths(model, batch, mask=cm.C6_MASK),
    }


def _write_train_artifacts(
    result: Mapping[str, Any], dictionary: np.ndarray, valid_data: Sequence[Any]
) -> None:
    _write_csv(
        RESULTS_DIR / "curve_seed0.csv",
        result["curve"],
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    model = _soup_model()
    _write_json(
        MECHANISM_DIR / "dictionary_health.json",
        base._dictionary_health(model, dictionary, valid_data),
    )
    _write_json(
        RESULTS_DIR / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "members": result["soup"]["members"],
            "member_valid_mae": result["soup"]["member_valid_mae"],
            "soup_valid_mae": result["soup"]["soup_valid_mae"],
            "soup_state_sha256": result.get("soup_state_sha256"),
        },
    )


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = RESULTS_DIR / "run_seed0.json"
    if final_path.exists() and not force:
        print("[train] cache hit", flush=True)
        result = _read_json(final_path)
        if not (RESULTS_DIR / "soup.json").exists():
            _write_train_artifacts(result, _dictionary_tensor(), p1run.load_split("valid"))
        return result
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    if not correctness.get("all_passed"):
        raise RuntimeError("acceptance gates not passed; refusing to train")
    timing = _read_json(RESULTS_DIR / "timing.json")
    if not timing.get("formal_run_authorized"):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "status": "NOT_RUN_RESOURCE_BUDGET",
            "reason": (
                "the measured CPU forward/backward timing predicts the frozen 320-epoch budget "
                "exceeds 4 hours; the implementation is delivered without the performance screen"
            ),
            "timing": timing,
        }
        _write_json(RESULTS_DIR / "train_skipped.json", payload)
        sc.official_test_blocker(payload)
        print("[train] skipped: NOT_RUN_RESOURCE_BUDGET", flush=True)
        return payload
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not smoke.get("passed"):
        raise RuntimeError("smoke not passed; refusing to train")
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    if preflight["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("preregistration changed after preflight; refusing to train")
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    initial_dictionary = torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    diagnostics: dict[str, Any] = {}
    started = time.perf_counter()

    def callback(epoch: int, model: sc.LatentScaleSEM108, optimizer: Any, curve: list[dict[str, Any]]):
        if epoch in DIAG_EPOCHS:
            row = _scale_diag_row(model, gradient_batch, initial_dictionary)
            row["epoch"] = int(epoch)
            row["valid_mae"] = float(curve[epoch - 1]["valid_mae"])
            row["train_mae"] = float(curve[epoch - 1]["train_mae"])
            diagnostics[str(epoch)] = row
            print(
                f"[diag@{epoch}] valid={row['valid_mae']:.6f} task_grad_D_L={row['task_grad_bridge_D_L']:.3e} "
                f"l0={row['bridge']['code']['mean_nonzero']:.1f}",
                flush=True,
            )
        return True, None

    result = cssd.train_cssd(
        tag=f"{TAG}-seed0",
        epochs=TRAIN_EPOCHS,
        threads=THREADS,
        out_dir=CHECKPOINT_DIR,
        subspace=subspace,
        train_data=train_data,
        valid_data=valid_data,
        seed=SEED,
        callback=callback,
        model_factory=_model_factory,
        save_states=True,
        log=True,
    )
    result.update(
        {
            "git_commit": _git_commit(),
            "device": "cpu",
            "threads": THREADS,
            "seed": SEED,
            "scale_seed": SCALE_SEED,
            "official_test_loaded": False,
            "diagnostics": diagnostics,
            "round_seconds": float(time.perf_counter() - started),
            "parameter_audit": sc.scale_parameter_audit(_soup_model()),
        }
    )
    sc.official_test_blocker(result)
    _write_json(final_path, result)
    _write_train_artifacts(result, dictionary, valid_data)
    print(
        f"[train] epochs={result['epochs_run']} best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# stage: frozen inference probes
# ---------------------------------------------------------------------------


def _intervention_row(
    model: sc.LatentScaleSEM108,
    valid_data: Sequence[Any],
    baseline: Mapping[str, Any],
    mask: Any,
) -> dict[str, float]:
    detailed = _evaluate_detailed(model, valid_data, mask)
    prediction = detailed["prediction"]
    base_prediction = baseline["prediction"]
    return {
        "mae": float(detailed["mae"]),
        "delta_mae": float(detailed["mae"] - baseline["mae"]),
        "delta_pred_rms": float(torch.sqrt(((prediction - base_prediction) ** 2).mean())),
    }


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = MECHANISM_DIR / "bridge_probes.json"
    if out.exists() and not force:
        print("[interventions] cache hit", flush=True)
        return _read_json(out)
    if not (RESULTS_DIR / "run_seed0.json").exists():
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "status": "NOT_RUN_RESOURCE_BUDGET",
        }
        _write_json(out, payload)
        print("[interventions] skipped: no formal run", flush=True)
        return payload
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    if int(run.get("epochs_run", 0)) < TRAIN_EPOCHS:
        raise RuntimeError("formal run incomplete; refusing to run the frozen probes")
    mask = cm.C6_MASK
    model = _soup_model()
    model.eval()
    valid_data = p1run.load_split("valid")
    baseline = _evaluate_detailed(model, valid_data, mask)
    m_s = float(baseline["mae"])

    model.set_bridge_intervention(zero_code=True)
    zero_row = _intervention_row(model, valid_data, baseline, mask)
    model.clear_bridge_intervention()

    permutation_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        model.set_bridge_intervention(permute_seed=int(shuffle_seed))
        row = _intervention_row(model, valid_data, baseline, mask)
        row["seed"] = int(shuffle_seed)
        permutation_rows.append(row)
        model.clear_bridge_intervention()
    permutation_deltas = [row["delta_mae"] for row in permutation_rows]
    permutation_rms = [row["delta_pred_rms"] for row in permutation_rows]

    trained_d = model.local_dictionary_bridge.D_L.detach().clone()
    trained_v = model.local_dictionary_bridge.V_L.detach().clone()
    model.reset_bridge_to_initialization()
    reset_row = _intervention_row(model, valid_data, baseline, mask)
    with torch.no_grad():
        model.local_dictionary_bridge.D_L.copy_(trained_d)
        model.local_dictionary_bridge.V_L.copy_(trained_v)

    valid_batch = next(iter(p1.make_env_loader(valid_data, int(p2run.BATCH_SIZE), False, 0)))
    with torch.no_grad():
        h = sc._capture_bridge_input(model, valid_batch, mask)
        _e, bridge_aux = model.local_dictionary_bridge(h, return_aux=True)
    code = lb.bridge_code_stats(bridge_aux["alpha"])
    reconstruction = lb.bridge_reconstruction_diagnostic(model, bridge_aux)
    task = _task_gradients(model, valid_batch, mask)
    bridge_parameter = model.local_dictionary_bridge
    movement = {
        "D_L_movement_frobenius": float(
            (bridge_parameter.D_L.detach().cpu() - bridge_parameter.D_init.cpu()).norm()
        ),
        "V_L_movement_frobenius": float(
            (bridge_parameter.V_L.detach().cpu() - bridge_parameter.V_init.cpu()).norm()
        ),
    }
    alpha = bridge_aux["alpha"].detach().double()
    support = alpha != 0.0
    per_object = support.sum(dim=1).to(dtype=torch.float64)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "soup_valid_mae": m_s,
        "modes": {
            "zero_bridge_code": zero_row,
            "within_graph_code_permutation": {
                "seeds": list(SHUFFLE_SEEDS),
                "rows": permutation_rows,
                "mean_delta_mae": float(np.mean(permutation_deltas)),
                "min_delta_mae": float(np.min(permutation_deltas)),
                "max_delta_mae": float(np.max(permutation_deltas)),
                "mean_delta_pred_rms": float(np.mean(permutation_rms)),
            },
            "dictionary_reset_to_init": reset_row,
        },
        "code_density": code,
        "code_effective_entries": {
            "mean_nonzero": float(per_object.mean()),
            "mean_nonzero_fraction": float(per_object.mean() / float(alpha.shape[1])),
        },
        "low_dimensional_reconstruction": reconstruction,
        "task_gradients": task,
        "parameter_movement": movement,
        "readout_layout": sc.readout_layout(sc.FULL),
        "interpretation": {
            "zero_code": (
                "the zero-code probe shows the task dictionary channel carries information; it "
                "does not show that dictionary learning is more valuable than a matched control"
            ),
            "density": (
                "soft-threshold coding density is variable; it is never reported as a fixed l0"
            ),
        },
    }
    sc.official_test_blocker(payload)
    _write_json(out, payload)
    print(
        f"[interventions] M_S={m_s:.6f} zero_delta={zero_row['delta_mae']:+.6f} "
        f"perm_mean_delta={float(np.mean(permutation_deltas)):+.6f} "
        f"reset_delta={reset_row['delta_mae']:+.6f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    references = _read_json(RESULTS_DIR / "historical_references.json")
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    timing = _read_json(RESULTS_DIR / "timing.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    run_path = RESULTS_DIR / "run_seed0.json"

    if not run_path.exists():
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "threads": THREADS,
            "official_test_loaded": False,
            "seed": SEED,
            "status": "NOT_RUN_RESOURCE_BUDGET",
            "verdict": "SCALE_TIMING_ONLY_NOT_RUN",
            "case": "Q",
            "parameters": {
                "small": int(preflight["parameter_audit"]["small"]["actual_parameters"]),
                "full": int(preflight["parameter_audit"]["full"]["actual_parameters"]),
                "full_task_dictionary": int(
                    preflight["parameter_audit"]["full"]["task_dictionary_parameters"]
                ),
                "full_body": int(preflight["parameter_audit"]["full"]["body_parameters"]),
            },
            "timing": timing,
            "smoke": {
                "epochs": int(smoke["epochs"]),
                "subset": int(smoke["subset"]),
                "final_train_mae": float(smoke["final_train_mae"]),
                "seconds_per_epoch": float(smoke["seconds_per_epoch"]),
                "passed": bool(smoke["passed"]),
            },
            "gates": {
                "correctness_all_passed": bool(correctness["all_passed"]),
                "smoke_passed": bool(smoke["passed"]),
                "timing_authorized": bool(timing["formal_run_authorized"]),
            },
            "note": (
                "the implementation and the zero-training acceptance are delivered; the formal "
                "320-epoch screen was not started because the measured CPU budget exceeds 4 h. "
                "This is a resource decision, not an algorithmic failure."
            ),
        }
        sc.official_test_blocker(summary)
        _write_json(RESULTS_DIR / "summary.json", summary)
        _write_not_run_report(summary, references)
        _write_decision(summary)
        print("[analysis] status=NOT_RUN_RESOURCE_BUDGET", flush=True)
        return summary

    run = _read_json(run_path)
    probes = _read_json(MECHANISM_DIR / "bridge_probes.json")
    health = _read_json(MECHANISM_DIR / "dictionary_health.json")

    soup_model = _soup_model()
    soup_model.eval()
    train_eval = _evaluate_detailed(soup_model, p1run.load_split("train"), cm.C6_MASK)

    soup_mae = float(run["soup"]["soup_valid_mae"])
    band = _band(soup_mae)
    zero_delta = float(probes["modes"]["zero_bridge_code"]["delta_mae"])
    permutation_mean = float(probes["modes"]["within_graph_code_permutation"]["mean_delta_mae"])
    reset_delta = float(probes["modes"]["dictionary_reset_to_init"]["delta_mae"])
    task_grad_d = float(probes["task_gradients"]["task_grad_bridge_D_L"])
    task_grad_v = float(probes["task_gradients"]["task_grad_bridge_V_L"])
    d_movement = float(probes["parameter_movement"]["D_L_movement_frobenius"])
    v_movement = float(probes["parameter_movement"]["V_L_movement_frobenius"])
    bridge_learned = bool(d_movement > 0.0 and v_movement > 0.0 and task_grad_d > 0.0 and task_grad_v > 0.0)
    bridge_load_bearing = bool(zero_delta >= GATE_BRIDGE_ZERO and permutation_mean > 0.0)
    if soup_mae <= 0.110:
        case = "A"
    elif soup_mae <= 0.115:
        case = "B"
    elif soup_mae <= 0.120:
        case = "C"
    else:
        case = "D"
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "official_test_loaded": False,
        "seed": SEED,
        "epochs": int(run["epochs_run"]),
        "wall_clock_s": float(run["wall_clock_s"]),
        "seconds_per_epoch": float(run.get("seconds_per_epoch", float("nan"))),
        "parameters": {
            "small": int(preflight["parameter_audit"]["small"]["actual_parameters"]),
            "full": int(preflight["parameter_audit"]["full"]["actual_parameters"]),
            "full_task_dictionary": int(
                preflight["parameter_audit"]["full"]["task_dictionary_parameters"]
            ),
            "full_body": int(preflight["parameter_audit"]["full"]["body_parameters"]),
        },
        "timing": {
            "predicted_total_seconds_with_margin": float(
                timing["predicted"]["total_with_margin_seconds"]
            ),
            "budget_seconds": float(FORMAL_BUDGET_S),
            "steady_train_step_seconds": float(timing["train_step_seconds"]["steady_median"]),
            "steady_eval_step_seconds": float(timing["eval_step_seconds"]["steady_median"]),
            "smoke_seconds_per_epoch": float(smoke["seconds_per_epoch"]),
        },
        "M_S": soup_mae,
        "soup_train_mae": float(train_eval["mae"]),
        "train_valid_gap": float(soup_mae - float(train_eval["mae"])),
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": list(run["soup"]["members"]),
        "performance_band": band,
        "backgrounds": {
            "latent_bridge_soup_valid_mae": BRIDGE_BACKGROUND_SOUP,
            "delta_vs_latent_bridge": float(BRIDGE_BACKGROUND_SOUP - soup_mae),
            "sem108_soup_valid_mae": SEM108_BACKGROUND_SOUP,
            "delta_vs_sem108": float(SEM108_BACKGROUND_SOUP - soup_mae),
            "note": "unmatched backgrounds only; no causal claim from a single seed",
        },
        "bridge": {
            "zero_code_delta_mae": zero_delta,
            "zero_code_delta_pred_rms": float(probes["modes"]["zero_bridge_code"]["delta_pred_rms"]),
            "permutation_mean_delta_mae": permutation_mean,
            "permutation_mean_delta_pred_rms": float(
                probes["modes"]["within_graph_code_permutation"]["mean_delta_pred_rms"]
            ),
            "reset_to_init_delta_mae": reset_delta,
            "reset_to_init_delta_pred_rms": float(
                probes["modes"]["dictionary_reset_to_init"]["delta_pred_rms"]
            ),
            "task_grad_D_L": task_grad_d,
            "task_grad_V_L": task_grad_v,
            "D_L_movement_frobenius": d_movement,
            "V_L_movement_frobenius": v_movement,
            "code_density": probes["code_density"],
            "low_dimensional_reconstruction": probes["low_dimensional_reconstruction"],
            "learned": bridge_learned,
            "load_bearing": bridge_load_bearing,
            "note": (
                "D_L / V_L moved from the frozen frame and receive MAE-only task gradient; the "
                "zero-code and permutation probes show the channel carries information, not that "
                "dictionary learning beats a matched trained control"
            ),
        },
        "diagnostics": run.get("diagnostics", {}),
        "gates": {
            "correctness_all_passed": bool(correctness["all_passed"]),
            "smoke_passed": bool(smoke["passed"]),
            "parameter_exact": bool(preflight["parameter_audit"]["full"]["parameter_exact"]),
            "timing_authorized": bool(timing["formal_run_authorized"]),
            "dictionary_health": health,
        },
        "historical_unmatched_only": True,
        "matched_baseline_rerun": False,
        "case": case,
        "verdict": band,
    }
    sc.official_test_blocker(summary)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, references, run, probes)
    _write_decision(summary)
    print(f"[analysis] M_S={soup_mae:.6f} band={band} case={case}", flush=True)
    return summary


def _write_not_run_report(summary: Mapping[str, Any], references: Mapping[str, Any]) -> None:
    timing = summary["timing"]
    lines = [
        "# e2e_dictenv_scale_v1 — unified Small/Full task-dictionary scaling",
        "",
        "CPU only · official test never loaded · **no performance screen** (resource decision).",
        "",
        "## Status",
        "",
        f"**{summary['verdict']}** — the implementation and the zero-training acceptance are "
        "delivered; the formal 320-epoch Full run was not started because the measured CPU "
        "forward/backward timing predicts the frozen 4-hour budget is exceeded.",
        "",
        "## Provenance",
        "",
        f"- commit `{summary['git_commit']}` · device `cpu` · threads `{summary['threads']}` · seed `{summary['seed']}`",
        "- `official_test_loaded = false`",
        "",
        "## Parameters",
        "",
        f"- Small `{summary['parameters']['small']}` · Full `{summary['parameters']['full']}` "
        f"(task dictionary `{summary['parameters']['full_task_dictionary']}`, body "
        f"`{summary['parameters']['full_body']}`)",
        "",
        "## Timing (measured on real official-train batches)",
        "",
        f"- steady train step `{_fmt(timing['train_step_seconds']['steady_median'], 4)} s` · "
        f"steady eval step `{_fmt(timing['eval_step_seconds']['steady_median'], 4)} s`",
        f"- predicted 320-epoch total with margin `{_fmt(timing['predicted']['total_with_margin_seconds'], 1)} s` "
        f"against a `{_fmt(FORMAL_BUDGET_S, 0)} s` budget",
        f"- smoke (3 epochs x 1024 molecules): `{_fmt(summary['smoke']['seconds_per_epoch'], 3)} s/epoch`, "
        f"final train MAE `{_fmt(summary['smoke']['final_train_mae'], 6)}`",
        "",
        "## Gates",
        "",
        f"- correctness all passed: `{summary['gates']['correctness_all_passed']}`",
        f"- smoke passed: `{summary['gates']['smoke_passed']}`",
        f"- timing authorised: `{summary['gates']['timing_authorized']}`",
        "",
        "No ZINC valid MAE is reported for this round.",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    summary: Mapping[str, Any],
    references: Mapping[str, Any],
    run: Mapping[str, Any],
    probes: Mapping[str, Any],
) -> None:
    s = summary
    lines: list[str] = []
    lines.append("# e2e_dictenv_scale_v1 — unified Small/Full task-dictionary scaling")
    lines.append("")
    lines.append("CPU only · official test never loaded · single Full seed-0 trajectory.")
    lines.append("")
    lines.append("## A. Provenance")
    lines.append("")
    lines.append(f"- formal-run commit `{run.get('git_commit', 'n/a')}`")
    lines.append(f"- analysis / report commit `{s['git_commit']}`")
    lines.append(f"- device `cpu` · threads `{s['threads']}` · seed `{s['seed']}`")
    lines.append("- `official_test_loaded = false`")
    lines.append(
        f"- vendored reference package `{references.get('vendored_reference_package', '')}`"
    )
    lines.append("")
    lines.append("## B. Historical references (read-only; never rerun; unmatched)")
    lines.append("")
    for name, ref in references["references"].items():
        value = ref.get("value") if isinstance(ref, Mapping) else ref
        line = f"- {name}: " + ("n/a" if value is None else f"`{_fmt(value, 6)}`")
        lines.append(line)
    lines.append("")
    lines.append("## C. Exact architecture")
    lines.append("")
    lines.append("```")
    lines.append("Sem108: static local object -> fusion (446 -> 342 -> 144) -> h (144)")
    lines.append("candidate: h -> shared task dictionary [D_L 144x288, V_L 288x144]")
    lines.append("           -> rho-normalised 16-step unrolled ISTA codes alpha (288)")
    lines.append("           -> E = rho * alpha @ V_L (144) -> unary + pair (16m -> 48) + reader (814 -> 39 -> 39 -> 1)")
    lines.append("no graph index, no message passing, no residual bypass, no new raw features")
    lines.append("```")
    lines.append("")
    lines.append("## D. Parameter budget")
    lines.append("")
    p = s["parameters"]
    lines.append(
        f"- Small `{p['small']}` (projection reference) · Full `{p['full']}` "
        f"(task dictionary `{p['full_task_dictionary']}`, body `{p['full_body']}`)"
    )
    lines.append("")
    lines.append("## E. Training")
    lines.append("")
    lines.append(
        f"- epochs `{s['epochs']}` · wall `{_fmt(s['wall_clock_s'], 1)} s` · seed `{s['seed']}`"
    )
    lines.append(
        f"- best valid `{_fmt(s['best_valid_mae'], 6)}` @ {s['best_epoch']} · Top-5 members "
        f"{s['soup_members']} · soup `{_fmt(s['M_S'], 6)}`"
    )
    lines.append(
        f"- same-protocol soup train MAE `{_fmt(s['soup_train_mae'], 6)}` · "
        f"train→valid gap `{s['train_valid_gap']:+.6f}`"
    )
    lines.append("")
    lines.append("## F. Performance interpretation")
    lines.append("")
    lines.append(f"- `M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**")
    for name in ("latent_bridge", "sem108"):
        background = s["backgrounds"]
        lines.append(
            f"- {name} background `{_fmt(background[name + '_soup_valid_mae'], 6)}` "
            f"(difference `{background['delta_vs_' + name]:+.6f}`) — unmatched background only"
        )
    lines.append("")
    lines.append("## G. Task-dictionary mechanism (frozen soup state, inference only)")
    lines.append("")
    b = s["bridge"]
    lines.append(
        f"- zero task-dictionary code: `delta_mae = {_fmt(b['zero_code_delta_mae'], 6)}`, "
        f"`delta_pred_rms = {_fmt(b['zero_code_delta_pred_rms'], 6)}`"
    )
    lines.append(
        f"- within-molecule code permutation (5 seeds): mean `delta_mae = "
        f"{_fmt(b['permutation_mean_delta_mae'], 6)}`, mean `delta_pred_rms = "
        f"{_fmt(b['permutation_mean_delta_pred_rms'], 6)}`"
    )
    lines.append(
        f"- reset D_L / V_L to init (encoder and head kept): `delta_mae = "
        f"{_fmt(b['reset_to_init_delta_mae'], 6)}`, `delta_pred_rms = "
        f"{_fmt(b['reset_to_init_delta_pred_rms'], 6)}`"
    )
    lines.append(
        f"- MAE-only task gradients: `D_L {b['task_grad_D_L']:.3e}`, `V_L {b['task_grad_V_L']:.3e}`; "
        f"movement Frobenius `D_L {b['D_L_movement_frobenius']:.4f}`, "
        f"`V_L {b['V_L_movement_frobenius']:.4f}`"
    )
    density = b["code_density"]
    lines.append(
        f"- code density: mean non-zero `{_fmt(density['mean_nonzero'], 2)}/288` "
        f"(p50 `{_fmt(density['l0_p50'], 1)}`, p95 `{_fmt(density['l0_p95'], 1)}`) — variable density"
    )
    lines.append(
        f"- diagnostic low-dimensional reconstruction: mean `{b['low_dimensional_reconstruction']['mean_relative']:.4f}` "
        "(diagnostic only, never an outer loss)"
    )
    lines.append(f"- learned `{b['learned']}` · load bearing `{b['load_bearing']}`")
    lines.append("")
    lines.append("## H. Verdict")
    lines.append("")
    lines.append(f"**Case {s['case']} — {s['verdict']}**")
    lines.append("")
    lines.append(f"> {b['note']}")
    lines.append("")
    lines.append("## I. Evidence discipline")
    lines.append("")
    lines.append(
        "- single seed, no second seed, no matched control arm, no official-test read; every "
        "historical comparison is unmatched context"
    )
    lines.append(
        "- the zero-code / permutation / reset probes establish channel information only; they "
        "do not establish that dictionary learning beats a matched trained control"
    )
    lines.append(
        "- the band is a resource-decision threshold, not a significance test"
    )
    lines.append("")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    s = summary
    if s.get("status") == "NOT_RUN_RESOURCE_BUDGET":
        lines = [
            "# DECISION — e2e_dictenv_scale_v1",
            "",
            "**SCALE_TIMING_ONLY_NOT_RUN**",
            "",
            f"Predicted 320-epoch CPU total with margin `{_fmt(s['timing']['predicted']['total_with_margin_seconds'], 1)} s` "
            f"exceeds the frozen `{_fmt(FORMAL_BUDGET_S, 0)} s` budget.",
            "",
            "The unified Small/Full model, its tests, acceptance gates and timing record are "
            "delivered; the formal Full performance screen was not started.",
            "This is a resource decision, not an algorithmic failure.",
            "",
        ]
    else:
        lines = [
            "# DECISION — e2e_dictenv_scale_v1",
            "",
            f"`M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**.",
            "",
            f"Task-dictionary zero-code `delta_mae = {_fmt(s['bridge']['zero_code_delta_mae'], 6)}`, "
            f"permutation mean `delta_mae = {_fmt(s['bridge']['permutation_mean_delta_mae'], 6)}`, "
            f"reset-to-init `delta_mae = {_fmt(s['bridge']['reset_to_init_delta_mae'], 6)}`.",
            "",
            "Single Full seed-0 trajectory, no matched baseline rerun, no official-test read.",
            "Next-step authorization is recorded in the analysis note; no rescue run was executed.",
            "",
        ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


STAGES = (
    "references",
    "audit",
    "preflight",
    "correctness",
    "timing",
    "smoke",
    "train",
    "interventions",
    "analysis",
    "chain",
)


def configure(*, epochs: int | None = None, threads: int | None = None, device: str = "cpu"):
    """Freeze-check the runtime settings (CPU only, 320-epoch budget fixed)."""
    global TRAIN_EPOCHS, THREADS
    if torch.device(device).type != "cpu":
        raise RuntimeError(f"this round is CPU-only, got device={device!r}")
    if epochs is not None and int(epochs) != TRAIN_EPOCHS:
        raise RuntimeError(
            f"the epoch budget is frozen at {TRAIN_EPOCHS}, got {epochs!r}; "
            "a new pre-registration is required to change it"
        )
    if threads is not None:
        THREADS = int(threads)
    return {"device": "cpu", "epochs": int(TRAIN_EPOCHS), "threads": int(THREADS)}


def run_stages(stage: str, *, force: bool = False) -> int:
    sem.cpu_only_guard(torch.device("cpu"))
    if stage in ("references", "chain"):
        stage_references()
    if stage in ("audit", "chain"):
        stage_audit(force=force)
    if stage in ("preflight", "chain"):
        stage_preflight()
    if stage in ("correctness", "chain"):
        stage_correctness()
    if stage in ("timing", "chain"):
        stage_timing(force=force)
    if stage in ("smoke", "chain"):
        stage_smoke()
    if stage in ("train", "chain"):
        stage_train(force=force)
    if stage in ("interventions", "chain"):
        stage_interventions(force=force)
    if stage in ("analysis", "chain"):
        stage_analysis()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run_stages(args.stage, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
