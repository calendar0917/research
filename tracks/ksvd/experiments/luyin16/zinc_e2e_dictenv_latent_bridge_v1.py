"""E2E-DictEnv-Latent-Bridge-v1 runner — SEM108 + local task dictionary bridge.

Round ``e2e_dictenv_latent_bridge_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_latent_bridge_v1.py``.
Pre-registration:
``tracks/ksvd/notes/zinc_e2e_dictenv_latent_bridge_v1_preregistration.md``.
Delivered reference package:
``tracks/ksvd/experiments/luyin16/latent_bridge_reference/v1_20261001/``.

Stages
------
``references audit preflight correctness smoke train interventions analysis dev chain``

Exactly one from-scratch ``SEM108 + latent bridge`` seed-0 trajectory (320
epochs) on CPU.  ``dev`` runs the two zero-training / short paired checks (A:
implementation acceptance; B: 40-epoch train-only paired trainability) and is
deliberately kept outside the formal chain: its evidence is scratch/dev only.
The official ZINC **test** split is never instantiated; every payload records
``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import math
import time
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
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = lb.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_latent_bridge_v1"
AUDIT_DIR = RESULTS_DIR / "audit"
MECHANISM_DIR = RESULTS_DIR / "mechanism"
DEV_DIR = RESULTS_DIR / "dev"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "zinc_e2e_dictenv_latent_bridge_v1_preregistration.md"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
REFERENCE_DIR = (
    TRACK_ROOT / "experiments/luyin16/latent_bridge_reference/v1_20261001"
)

TAG = "LATENT-BRIDGE"
THREADS = 8
TRAIN_EPOCHS = 320
SEED = 0
SMOKE_EPOCHS = 8
SMOKE_SUBSET = 1024  # 8 x 8 = 64 optimizer steps (<= 100)
SHORT_EPOCHS = 40
SHORT_SUBSET = 1024
SHUFFLE_SEEDS = audit.SHUFFLE_SEEDS  # (101, 202, 303, 404, 505)
DIAG_EPOCHS: tuple[int, ...] = (1, 20, 40, 80, 160, 240, 320)

#: pre-registered absolute-MAE performance bands (pre-registration section 8).
BANDS: tuple[tuple[float, str], ...] = (
    (0.115, "LATENT_BRIDGE_STRONG_SINGLE_SEED_SIGNAL"),
    (0.120, "LATENT_BRIDGE_PROMISING_SINGLE_SEED"),
    (0.1233, "LATENT_BRIDGE_BORDERLINE"),
    (float("inf"), "LATENT_BRIDGE_STOP_CURRENT_CANDIDATE"),
)
#: read-only background reference (the immediately previous Sem108 candidate).
SEM108_BACKGROUND_SOUP = 0.123704927947314
#: relabel invariance tolerance: relabelling permutes the index_add_ reduction
#: order, so in float32 the prediction may move by a few ulps.
RELABEL_TOL = 1.0e-4
#: bridge-channel liveness threshold for the zero-code probe (dev evidence).
GATE_BRIDGE_ZERO = 0.005
#: acceptance gap: total relative squared error of the formal-init bridge vs
#: the sem108 environment on the same real train batch.  Larger than this
#: blocks the formal run (normalisation / V transpose / step / wiring check).
GATE_FORMAL_RELATIVE_SQ = 0.05
#: identity-mode tolerances (pre-registration section 5A).
GATE_IDENTITY_RELATIVE_L2 = 1.0e-6
GATE_IDENTITY_PRED_MAX = 1.0e-5

_write_json = base._write_json
_read_json = base._read_json
_write_csv = base._write_csv
_git_commit = base._git_commit
_sha256_file = base._sha256_file
_clone_batch = base._clone_batch
_synthetic_batch = base._synthetic_batch
_load_parent_subspace = base._load_parent_subspace
_dictionary_tensor = base._dictionary_tensor


def _model_factory(
    dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace
) -> lb.LatentBridgeSEM108:
    return lb.build_latent_bridge_model(dictionary, int(seed), subspace)


def _band(mae: float) -> str:
    for threshold, label in BANDS:
        if float(mae) <= threshold:
            return label
    return BANDS[-1][1]


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, AUDIT_DIR, MECHANISM_DIR, DEV_DIR, CHECKPOINT_DIR):
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


def _soup_model() -> lb.LatentBridgeSEM108:
    model = _model_factory(_dictionary_tensor(), SEED, _load_parent_subspace())
    model.load_state_dict(_soup_state())
    return model


def _real_batch(subset: int = 32, split: str = "train") -> Any:
    data_list = p1run.load_split(split, subset=int(subset))
    return next(iter(p1.make_env_loader(data_list, int(subset), False, 0)))


def _gradient_diagnostics(
    model: lb.LatentBridgeSEM108, batch: Any, mask: Any, lam: float
) -> dict[str, Any]:
    """Full parent-loss gradients, with the two new bridge matrices added."""
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
        "grad_node_encoder_W1": _grad(model.node_encoder[0].weight),
        "grad_edge_encoder_W1": _grad(model.edge_encoder[0].weight),
        "grad_bridge_D_L": _grad(model.local_dictionary_bridge.D_L),
        "grad_bridge_V_L": _grad(model.local_dictionary_bridge.V_L),
        "coord_nnz_per_row_mean": float((alpha != 0).double().sum(dim=1).mean()),
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def _task_gradients(
    model: lb.LatentBridgeSEM108, batch: Any, mask: Any
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
        "task_grad_D": _grad(model.D),
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def _capture_bridge_inputs(
    model: lb.LatentBridgeSEM108, batch: Any, mask: Any | None
) -> torch.Tensor:
    """Run one forward pass and return the fusion output ``h`` feeding the bridge."""
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: Any, args: Sequence[Any]) -> None:
        captured["h"] = args[0].detach()

    handle = model.local_dictionary_bridge.register_forward_pre_hook(_hook)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            if mask is None:
                model(batch)
            else:
                model(batch, mask=mask)
    finally:
        handle.remove()
        model.train(was_training)
    if "h" not in captured:
        raise RuntimeError("bridge was not called during the forward pass")
    return captured["h"]


def _evaluate_detailed(
    model: lb.LatentBridgeSEM108, data_list: Sequence[Any], mask: Any
) -> dict[str, Any]:
    """Valid MAE plus the full prediction vector (fixed order, no shuffling)."""
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
        "reference_package": str(REFERENCE_DIR.relative_to(REPO_ROOT)),
        "references": {
            "Sem108-seed0-background": (
                None
                if sem_soup is None
                else {
                    "value": float(sem_soup["soup_valid_mae"]),
                    "members": list(sem_soup.get("members", [])),
                    "artifact": "tracks/ksvd/results/e2e_dictenv_sem108_v1/soup.json",
                    "status": "immediately previous round on the same split; background only, "
                    "not a matched control (the candidate adds 9,216 parameters and changes "
                    "the architecture, so no causal increment can be claimed)",
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
    }
    lb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "historical_references.json", payload)
    sem_value = payload["references"]["Sem108-seed0-background"]
    print(
        "[references] Sem108 background "
        + ("n/a" if sem_value is None else f"{sem_value['value']:.6f}"),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: zero-training audits (identical Sem108 interface)
# ---------------------------------------------------------------------------


def stage_audit(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    identity_path = AUDIT_DIR / "sem108_identity.json"
    if identity_path.exists() and not force:
        identity = _read_json(identity_path)
    else:
        started = time.perf_counter()
        identity = sem.audit_sem108_identity()
        identity["wall_clock_s"] = float(time.perf_counter() - started)
        identity["protocol_version"] = PROTOCOL_VERSION
        lb.official_test_blocker(identity)
        _write_json(identity_path, identity)
    print(
        f"[audit] A1 identity_established={identity['identity_established']} "
        f"atom_shell={identity['atom_shell_cols']} bond_shell={identity['bond_shell_cols']}",
        flush=True,
    )

    relation_path = AUDIT_DIR / "anchor_relation.json"
    if relation_path.exists() and not force:
        relation = _read_json(relation_path)
    else:
        started = time.perf_counter()
        relation = sem.audit_anchor_relation_and_redundancy()
        relation["wall_clock_s"] = float(time.perf_counter() - started)
        relation["protocol_version"] = PROTOCOL_VERSION
        lb.official_test_blocker(relation)
        _write_json(relation_path, relation)
    print(
        f"[audit] A2 a2_pass={relation['a2_pass']} a4_pass={relation['a4_pass']}",
        flush=True,
    )

    t1_path = AUDIT_DIR / "t1_block_ablation.json"
    if t1_path.exists() and not force:
        t1 = _read_json(t1_path)
    else:
        t1 = sem.audit_t1_block_ablation()
        t1["protocol_version"] = PROTOCOL_VERSION
        lb.official_test_blocker(t1)
        _write_json(t1_path, t1)
    decision_path = AUDIT_DIR / "audit_decision.json"
    if decision_path.exists() and not force:
        decision = _read_json(decision_path)
    else:
        decision = sem.audit_decision(identity, relation, t1)
        decision["protocol_version"] = PROTOCOL_VERSION
        lb.official_test_blocker(decision)
        _write_json(decision_path, decision)
    print(f"[audit] decision={decision['decision']} t1={t1['status']}", flush=True)
    return {"identity": identity, "relation": relation, "t1": t1, "decision": decision}


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    decision = _read_json(AUDIT_DIR / "audit_decision.json")
    if decision.get("decision") != "PROCEED":
        raise RuntimeError("Phase-A audit did not authorise the formal run")
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    prereg_sha = _sha256_file(PREREG_PATH)

    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    sem_model = sem.build_sem108_model(dictionary, SEED, subspace)
    parent = cssd.build_cssd_model(dictionary, SEED, subspace)
    candidate = _model_factory(dictionary, SEED, subspace)
    parent_params = int(sum(parameter.numel() for parameter in parent.parameters()))
    sem_audit = sem.parameter_audit(sem_model, parent_params)
    bridge_audit = lb.bridge_parameter_audit(candidate)
    if not bridge_audit["sem108_body_unchanged"]:
        raise RuntimeError(f"Sem108 body parameters changed: {bridge_audit}")
    if not bridge_audit["candidate_exact"]:
        raise RuntimeError(f"candidate parameter count mismatch: {bridge_audit}")
    if not sem_audit["parameter_ratio_within_bound"]:
        raise RuntimeError(f"Sem108 body parameter ratio violated: {sem_audit}")

    # every shared state-dict entry must be bit-identical to the Sem108 model.
    state_sem = sem_model.state_dict()
    state_candidate = candidate.state_dict()
    shared_mismatch = [
        key
        for key, value in state_sem.items()
        if key in state_candidate
        and state_candidate[key].shape == value.shape
        and not torch.equal(value, state_candidate[key])
    ]
    if shared_mismatch:
        raise RuntimeError(f"shared parameters differ from Sem108: {shared_mismatch[:8]}")

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
        "reference_package": str(REFERENCE_DIR.relative_to(REPO_ROOT)),
        "sem108_body": {
            "parameter_audit": sem_audit,
            "shared_state_mismatch": shared_mismatch,
        },
        "bridge": {
            "dim": int(lb.BRIDGE_DIM),
            "atoms": int(lb.BRIDGE_ATOMS),
            "steps": int(lb.BRIDGE_STEPS),
            "lambda1": float(lb.BRIDGE_LAMBDA1),
            "lambda2": float(lb.BRIDGE_LAMBDA2),
            "step_safety": float(lb.BRIDGE_STEP_SAFETY),
            "seed": int(lb.BRIDGE_SEED),
            "initialization": "D_L = [I, Q] (unit columns), V_L = D_L.T, private NumPy RNG",
            "parameter_audit": bridge_audit,
        },
        "interface": {
            "input": "Sem108 fusion output h in R^48",
            "output": "E in R^48 consumed by unary pooling and static pair computation",
            "graph_indices_accepted": False,
            "state_writeback": False,
            "residual_bypass": False,
            "new_raw_features": False,
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
        "epochs": TRAIN_EPOCHS,
        "seed": SEED,
    }
    lb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration"])
    _write_json(RESULTS_DIR / "parameter_audit.json", bridge_audit)
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {prereg_sha[:12]} sem_body={bridge_audit['sem108_body_params']} "
        f"bridge={bridge_audit['bridge_params']} total={bridge_audit['candidate_params']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: correctness / acceptance audit (A)
# ---------------------------------------------------------------------------


def acceptance_audit(
    *,
    subset: int = 32,
    threads: int = THREADS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Zero-training acceptance audit on one real official-train batch.

    Implements pre-registration section 5A exactly: shared initial state, the
    identity mode (``lambda1 = lambda2 = 0`` with the initial ``V_L``), the
    formal-initialisation gap, MAE-only bridge gradients and an actual update,
    the single bridge call on both paths, wiring into unary + pair, relabel /
    batch-offset / endpoint-swap invariance, relation no-write-back, the clean
    zero-code intervention and the official-test blocker.
    """
    audit.attach_cpu(int(threads))
    torch.set_num_threads(int(threads))
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    sem_model = sem.build_sem108_model(dictionary, seed, subspace)
    candidate = _model_factory(dictionary, seed, subspace)
    identity = lb.build_latent_bridge_model(
        dictionary, seed, subspace, bridge_lambda1=0.0, bridge_lambda2=0.0
    )
    for model in (sem_model, candidate, identity):
        model.eval()

    # -- shared initial state --------------------------------------------------
    state_sem = sem_model.state_dict()
    state_candidate = candidate.state_dict()
    shared_keys = [
        key
        for key in state_sem
        if key in state_candidate and not key.startswith("local_dictionary_bridge.")
    ]
    shared_mismatch = [
        key for key in shared_keys if not torch.equal(state_sem[key], state_candidate[key])
    ]
    parameter_audit = lb.bridge_parameter_audit(candidate)

    batch = _real_batch(subset=int(subset), split="train")
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    n_graphs = int(batch.global_context.shape[0])

    # -- identity mode: bridge must reproduce the Sem108 environment ----------
    with torch.no_grad():
        pred_sem, aux_sem = sem_model(batch, mask=mask, return_aux=True)
        pred_identity, aux_identity = identity(batch, mask=mask, return_aux=True)
        coord_sem = sem_model.code(batch.dict_phi)
        coord_identity = identity.code(batch.dict_phi)
        env_sem = sem_model.environments(coord_sem, batch)
        env_identity = identity.environments(coord_identity, batch)
    identity_e_norm = float(torch.norm(env_sem))
    identity_e_abs = float((env_identity - env_sem).abs().max())
    identity_e_rel_l2 = float(torch.norm(env_identity - env_sem) / max(identity_e_norm, 1e-12))
    identity_pred_max = float((pred_identity - pred_sem).abs().max())

    # -- formal initialisation gap vs the Sem108 environment -------------------
    with torch.no_grad():
        env_formal = candidate.environments(candidate.code(batch.dict_phi), batch)
        _pred_formal, aux_formal = candidate(batch, mask=mask, return_aux=True)
    residual = env_formal - env_sem
    formal_relative_sq = float(
        (residual * residual).sum() / (env_sem * env_sem).sum().clamp_min(lb.BRIDGE_EPS)
    )
    per_row = (residual * residual).sum(dim=1) / (env_sem * env_sem).sum(dim=1).clamp_min(
        lb.BRIDGE_EPS
    )
    formal_row_p95 = float(torch.quantile(per_row, 0.95))

    # -- bridge gradients and one actual update --------------------------------
    task_init = _task_gradients(candidate, batch, mask)
    full_init = _gradient_diagnostics(candidate, batch, mask, lam)
    update_model = _model_factory(dictionary, seed, subspace)
    update_model.load_state_dict(candidate.state_dict())
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
    # small SGD step on a fresh clone: direction must decrease the same-batch MAE
    sgd_model = _model_factory(dictionary, seed, subspace)
    sgd_model.load_state_dict(candidate.state_dict())
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

    # -- exactly one bridge call per forward on both paths ---------------------
    call_counts: list[int] = []

    def _count_hook(_module: Any, _args: Any, _output: Any) -> None:
        call_counts.append(1)

    handle = candidate.local_dictionary_bridge.register_forward_hook(_count_hook)
    with torch.no_grad():
        candidate(batch, mask=mask)
    plain_calls = len(call_counts)
    call_counts.clear()
    with torch.no_grad():
        candidate(batch)
    unmasked_calls = len(call_counts)
    handle.remove()

    # -- E enters unary pooling and the static pair projection -----------------
    pair_inputs: dict[str, torch.Tensor] = {}
    pair_handle = candidate.pair_projection.register_forward_pre_hook(
        lambda _module, inputs: pair_inputs.__setitem__("value", inputs[0].detach())
    )
    with torch.no_grad():
        _pred_w, aux_w = candidate(batch, mask=mask, return_aux=True)
    pair_handle.remove()
    unary_reference = audit.pool_moments_masked(
        aux_w["E"], batch.batch, n_graphs, mask.unary_zero_blocks, None
    )
    pair_reads_bridge = bool(
        "value" in pair_inputs and torch.equal(pair_inputs["value"], aux_w["E"].detach())
    )
    unary_reads_bridge = bool(torch.equal(unary_reference, aux_w["unary"].detach()))

    # -- relabel / batch offset / endpoint swap --------------------------------
    relabel = base._relabel_invariance(candidate, n_molecules=2)
    data_pair = p1run.load_split("valid", subset=2)
    batch_pair = p1.env_collate(data_pair)
    with torch.no_grad():
        env_pair = candidate.environments(candidate.code(batch_pair.dict_phi), batch_pair)
        env_single = torch.cat(
            [
                candidate.environments(
                    candidate.code(item.dict_phi), p1.env_collate([item])
                )
                for item in data_pair
            ],
            dim=0,
        )
    batch_offset_max = float((env_pair - env_single).abs().max())

    swapped = _clone_batch(batch)
    swapped.env_bond_u = batch.env_bond_v.clone()
    swapped.env_bond_v = batch.env_bond_u.clone()
    with torch.no_grad():
        env_swap = candidate.environments(candidate.code(swapped.dict_phi), swapped)
    endpoint_swap_max = float((env_swap - env_formal).abs().max())

    # -- relation changes must not write back into the local code --------------
    relation_probe = _clone_batch(batch)
    relation_probe.pair_relation = batch.pair_relation + 7.0
    relation_probe.pair_bucket = (batch.pair_bucket + 1) % int(p2.DISTANCE_BUCKETS)
    with torch.no_grad():
        env_relation = candidate.environments(
            candidate.code(relation_probe.dict_phi), relation_probe
        )
    relation_no_writeback = float((env_relation - env_formal).abs().max())

    # -- clean zero-code intervention ------------------------------------------
    baseline = _evaluate_detailed(candidate, p1run.load_split("valid", subset=128), mask)
    candidate.set_bridge_intervention(zero_code=True)
    with torch.no_grad():
        zero_env = candidate.environments(candidate.code(batch.dict_phi), batch)
    zero_detailed = _evaluate_detailed(candidate, p1run.load_split("valid", subset=128), mask)
    candidate.clear_bridge_intervention()
    zero_env_abs = float(zero_env.abs().max())
    zero_delta_mae = float(zero_detailed["mae"] - baseline["mae"])
    zero_delta_pred_rms = float(
        torch.sqrt(((zero_detailed["prediction"] - baseline["prediction"]) ** 2).mean())
    )
    with torch.no_grad():
        _pred_f, _aux_f = candidate(batch, mask=mask, return_aux=True)
    finite_predictions = bool(torch.isfinite(_pred_f).all())

    # -- official-test blocker --------------------------------------------------
    blocker_raised = False
    try:
        lb.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocker_raised = True

    gates = {
        "A1_shared_init_mismatch": {
            "n_shared_keys": len(shared_keys),
            "mismatch": shared_mismatch,
            "passed": bool(not shared_mismatch),
        },
        "A2_parameter_contract": {
            **parameter_audit,
            "passed": bool(
                parameter_audit["sem108_body_unchanged"]
                and parameter_audit["bridge_exact"]
                and parameter_audit["candidate_exact"]
            ),
        },
        "A3_identity_mode": {
            "relative_l2": identity_e_rel_l2,
            "max_abs": identity_e_abs,
            "prediction_max_abs": identity_pred_max,
            "tolerance_relative_l2": GATE_IDENTITY_RELATIVE_L2,
            "tolerance_prediction_max_abs": GATE_IDENTITY_PRED_MAX,
            "passed": bool(
                identity_e_rel_l2 <= GATE_IDENTITY_RELATIVE_L2
                and identity_pred_max <= GATE_IDENTITY_PRED_MAX
            ),
        },
        "A4_formal_init_gap": {
            "relative_squared_error": formal_relative_sq,
            "row_error_p95": formal_row_p95,
            "block_threshold": GATE_FORMAL_RELATIVE_SQ,
            "passed": bool(formal_relative_sq <= GATE_FORMAL_RELATIVE_SQ),
        },
        "A5_bridge_gradients_and_update": {
            "task_init": task_init,
            "full_init": full_init,
            "adam_parameter_delta": {
                "D_L": d_delta,
                "V_L": v_delta,
            },
            "sgd_loss_before": float(sgd_loss),
            "sgd_loss_after": float(sgd_loss_after),
            "sgd_step_decreased": sgd_decreased,
            "finite": update_finite,
            "passed": bool(
                task_init["task_grad_bridge_D_L"] > 0.0
                and task_init["task_grad_bridge_V_L"] > 0.0
                and full_init["grad_bridge_D_L"] > 0.0
                and full_init["grad_bridge_V_L"] > 0.0
                and d_delta > 0.0
                and v_delta > 0.0
                and update_finite
                and sgd_decreased
            ),
        },
        "A6_single_bridge_call_both_paths": {
            "masked_calls": int(plain_calls),
            "unmasked_calls": int(unmasked_calls),
            "passed": bool(plain_calls == 1 and unmasked_calls == 1),
        },
        "A7_unary_and_pair_read_bridged_E": {
            "unary_matches_bridge_output": unary_reads_bridge,
            "pair_projection_input_matches_bridge_output": pair_reads_bridge,
            "passed": bool(unary_reads_bridge and pair_reads_bridge),
        },
        "A8_relabel_invariance": {
            **relabel,
            "passed": bool(relabel["max_abs_pred_diff"] <= RELABEL_TOL),
        },
        "A9_batch_offset_invariance": {
            "max_abs_env_diff": batch_offset_max,
            "tolerance": 1.0e-5,
            "passed": bool(batch_offset_max <= 1.0e-5),
        },
        "A10_endpoint_swap_invariance": {
            "max_abs_env_diff": endpoint_swap_max,
            "tolerance": 1.0e-6,
            "passed": bool(endpoint_swap_max <= 1.0e-6),
        },
        "A11_relation_no_writeback": {
            "max_abs_env_diff": relation_no_writeback,
            "tolerance": 0.0,
            "passed": bool(relation_no_writeback == 0.0),
        },
        "A12_zero_code_intervention": {
            "env_max_abs": zero_env_abs,
            "baseline_valid_mae": float(baseline["mae"]),
            "zero_valid_mae": float(zero_detailed["mae"]),
            "delta_mae": zero_delta_mae,
            "delta_pred_rms": zero_delta_pred_rms,
            "predictions_finite": finite_predictions,
            "passed": bool(
                zero_env_abs == 0.0
                and finite_predictions
                and abs(zero_delta_mae) > 0.0
                and zero_delta_pred_rms > 0.0
            ),
        },
        "A13_official_test_blocker": {
            "raised": blocker_raised,
            "passed": blocker_raised,
        },
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
        "train_batch_molecules": int(subset),
        "official_test_loaded": False,
        "identity_mode_note": (
            "lambda1 = lambda2 = 0 with the initial V_L is an audit-only mode and is never a "
            "training or candidate configuration"
        ),
        "identity_env_abs_max": identity_e_abs,
        "gates": gates,
        "all_passed": all_passed,
    }
    lb.official_test_blocker(payload)
    return payload


def stage_correctness() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    payload = acceptance_audit()
    _write_json(RESULTS_DIR / "correctness.json", payload)
    print(
        f"[correctness] all_passed={payload['all_passed']} "
        f"identity_rel_l2={payload['gates']['A3_identity_mode']['relative_l2']:.3e} "
        f"formal_rel_sq={payload['gates']['A4_formal_init_gap']['relative_squared_error']:.5f}",
        flush=True,
    )
    if not payload["all_passed"]:
        failed = [name for name, gate in payload["gates"].items() if not gate["passed"]]
        raise RuntimeError(f"acceptance gates failed: {failed}")
    return payload


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------


def stage_smoke() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = p1run.load_split("train", subset=SMOKE_SUBSET)
    valid_data = p1run.load_split("valid", subset=512)
    model = _model_factory(dictionary, SEED, subspace)
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    init_grads = _gradient_diagnostics(model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    init_task = _task_gradients(model, gradient_batch, cm.C6_MASK)
    result = cssd.train_cssd(
        tag=f"{TAG}-smoke",
        epochs=SMOKE_EPOCHS,
        threads=THREADS,
        out_dir=CHECKPOINT_DIR,
        subspace=subspace,
        train_data=train_data,
        valid_data=valid_data,
        seed=SEED,
        model_factory=_model_factory,
        save_states=True,
        log=True,
    )
    final_model = _model_factory(dictionary, SEED, subspace)
    final_model.load_state_dict(
        {
            key: value.float()
            for key, value in torch.load(
                CHECKPOINT_DIR / f"{TAG}-smoke_final_state.pt",
                map_location="cpu",
                weights_only=False,
            ).items()
        }
    )
    post_grads = _gradient_diagnostics(final_model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    post_task = _task_gradients(final_model, gradient_batch, cm.C6_MASK)
    activity = base._interface_activity(final_model, gradient_batch, cm.C6_MASK)
    finite = bool(
        math.isfinite(result["best_valid_mae"])
        and math.isfinite(result["final_valid_mae"])
        and all(math.isfinite(row["train_mae"]) for row in result["curve"])
    )
    gradient_keys = ("grad_D", "grad_fusion_W1", "grad_fusion_W2", "grad_W_A_S", "grad_W_E_S")
    gradient_ok = bool(
        all(init_grads[key] > 0.0 for key in gradient_keys)
        and all(post_grads[key] > 0.0 for key in gradient_keys)
        and init_grads["grad_bridge_D_L"] > 0.0
        and init_grads["grad_bridge_V_L"] > 0.0
        and post_grads["grad_bridge_D_L"] > 0.0
        and post_grads["grad_bridge_V_L"] > 0.0
    )
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
        "finite": finite,
        "gradient_ok": gradient_ok,
        "init_gradients": init_grads,
        "post_gradients": post_grads,
        "init_task_gradients": init_task,
        "post_task_gradients": post_task,
        "interface_activity": activity,
        "best_valid_mae": float(result["best_valid_mae"]),
        "final_valid_mae": float(result["final_valid_mae"]),
        "train_curve": result["curve"],
        "passed": bool(finite and gradient_ok and activity["passed"]),
    }
    lb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "smoke.json", payload)
    print(
        f"[smoke] steps={steps} finite={finite} gradient_ok={gradient_ok} "
        f"init_grad_D_L={init_grads['grad_bridge_D_L']:.3e} best={result['best_valid_mae']:.6f}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError("smoke failed (non-finite loss, dead gradients or inactive interface)")
    return payload


# ---------------------------------------------------------------------------
# stage: train (exactly one seed-0 trajectory)
# ---------------------------------------------------------------------------


def _bridge_diag_row(
    model: lb.LatentBridgeSEM108, batch: Any, initial_dictionary: torch.Tensor
) -> dict[str, Any]:
    grads = _gradient_diagnostics(model, batch, cm.C6_MASK, cm.H1_LAMBDA)
    task = _task_gradients(model, batch, cm.C6_MASK)
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        usage = base._usage_from_coord(coord, model.common_dim)
        h = _capture_bridge_inputs(model, batch, cm.C6_MASK)
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
    }


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = RESULTS_DIR / "run_seed0.json"
    if final_path.exists() and not force:
        print("[train] cache hit", flush=True)
        return _read_json(final_path)
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    if not correctness.get("all_passed"):
        raise RuntimeError("acceptance gates not passed; refusing to train")
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

    def callback(epoch: int, model: lb.LatentBridgeSEM108, optimizer: Any, curve: list[dict[str, Any]]):
        if epoch in DIAG_EPOCHS:
            row = _bridge_diag_row(model, gradient_batch, initial_dictionary)
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
            "official_test_loaded": False,
            "diagnostics": diagnostics,
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    lb.official_test_blocker(result)
    _write_json(final_path, result)
    _write_csv(
        RESULTS_DIR / "curve_seed0.csv",
        result["curve"],
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    health_model = _soup_model()
    _write_json(
        MECHANISM_DIR / "dictionary_health.json",
        base._dictionary_health(health_model, dictionary, valid_data),
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
    model: lb.LatentBridgeSEM108,
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
        "delta_pred_rms": float(
            torch.sqrt(((prediction - base_prediction) ** 2).mean())
        ),
    }


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = MECHANISM_DIR / "bridge_probes.json"
    if out.exists() and not force:
        print("[interventions] cache hit", flush=True)
        return _read_json(out)
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

    # S1 — zero the whole bridge code (alpha -> 0, hence E -> 0).
    model.set_bridge_intervention(zero_code=True)
    zero_row = _intervention_row(model, valid_data, baseline, mask)
    model.clear_bridge_intervention()

    # S2 — within-molecule code permutation (identical to a row permutation of
    # the bridge input, because the bridge is strictly row-local).
    permutation_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        model.set_bridge_intervention(permute_seed=int(shuffle_seed))
        row = _intervention_row(model, valid_data, baseline, mask)
        row["seed"] = int(shuffle_seed)
        permutation_rows.append(row)
        model.clear_bridge_intervention()
    permutation_deltas = [row["delta_mae"] for row in permutation_rows]
    permutation_rms = [row["delta_pred_rms"] for row in permutation_rows]

    # S3 — keep the trained encoder / head, reset only D_L / V_L to initial state.
    trained_d = model.local_dictionary_bridge.D_L.detach().clone()
    trained_v = model.local_dictionary_bridge.V_L.detach().clone()
    model.reset_bridge_to_initialization()
    reset_row = _intervention_row(model, valid_data, baseline, mask)
    with torch.no_grad():
        model.local_dictionary_bridge.D_L.copy_(trained_d)
        model.local_dictionary_bridge.V_L.copy_(trained_v)

    # code density + diagnostic reconstruction on a fixed valid batch (no grad)
    valid_batch = next(iter(p1.make_env_loader(valid_data, int(p2run.BATCH_SIZE), False, 0)))
    with torch.no_grad():
        h = _capture_bridge_inputs(model, valid_batch, mask)
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
        "low_dimensional_reconstruction": reconstruction,
        "task_gradients": task,
        "parameter_movement": movement,
        "interpretation": {
            "zero_code": (
                "the zero-code probe shows the bridge channel carries information; it does not "
                "show that dictionary learning is more valuable than a matched trained control"
            ),
            "delta_pred_rms": (
                "true sqrt(mean((intervened - baseline)^2)) over the same eval loader; distinct "
                "from RMS(intervened) - RMS(baseline), which can be negative"
            ),
        },
    }
    lb.official_test_blocker(payload)
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
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    audit_decision = _read_json(AUDIT_DIR / "audit_decision.json")
    probes = _read_json(MECHANISM_DIR / "bridge_probes.json")
    health = _read_json(MECHANISM_DIR / "dictionary_health.json")

    soup_mae = float(run["soup"]["soup_valid_mae"])
    band = _band(soup_mae)
    zero_delta = float(probes["modes"]["zero_bridge_code"]["delta_mae"])
    permutation_mean = float(
        probes["modes"]["within_graph_code_permutation"]["mean_delta_mae"]
    )
    reset_delta = float(probes["modes"]["dictionary_reset_to_init"]["delta_mae"])
    task_grad_d = float(probes["task_gradients"]["task_grad_bridge_D_L"])
    task_grad_v = float(probes["task_gradients"]["task_grad_bridge_V_L"])
    d_movement = float(probes["parameter_movement"]["D_L_movement_frobenius"])
    v_movement = float(probes["parameter_movement"]["V_L_movement_frobenius"])
    bridge_learned = bool(
        d_movement > 0.0 and v_movement > 0.0 and task_grad_d > 0.0 and task_grad_v > 0.0
    )
    bridge_load_bearing = bool(zero_delta >= GATE_BRIDGE_ZERO and permutation_mean > 0.0)
    if soup_mae <= 0.115 and bridge_learned and bridge_load_bearing:
        case = "A"
        verdict = "LATENT_BRIDGE_STRONG_SINGLE_SEED_SIGNAL"
    elif soup_mae <= 0.120 and bridge_learned and bridge_load_bearing:
        case = "B"
        verdict = "LATENT_BRIDGE_PROMISING_SINGLE_SEED"
    elif soup_mae <= 0.1233:
        case = "C"
        verdict = "LATENT_BRIDGE_BORDERLINE"
    else:
        case = "D"
        verdict = "LATENT_BRIDGE_STOP_CURRENT_CANDIDATE"
    bridge_note = (
        "D_L / V_L moved from the frozen initial frame and both receive a non-zero MAE-only "
        "task gradient. The zero-code and within-molecule permutation probes are live, but "
        "they only establish that the channel carries information; they do not establish "
        "that dictionary learning beats a matched trained control."
    )
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
            "sem108_body": int(preflight["bridge"]["parameter_audit"]["sem108_body_params"]),
            "bridge": int(preflight["bridge"]["parameter_audit"]["bridge_params"]),
            "candidate": int(preflight["bridge"]["parameter_audit"]["candidate_params"]),
        },
        "audit": {
            "decision": audit_decision["decision"],
            "t1_status": str(audit_decision["checks"].get("t1_block_audit", "")),
        },
        "M_S": soup_mae,
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": list(run["soup"]["members"]),
        "performance_band": band,
        "sem108_background": {
            "soup_valid_mae": SEM108_BACKGROUND_SOUP,
            "delta_vs_background": float(SEM108_BACKGROUND_SOUP - soup_mae),
            "note": "background only; the candidate adds 9,216 parameters, so no causal claim",
        },
        "bridge": {
            "zero_code_delta_mae": zero_delta,
            "zero_code_delta_pred_rms": float(
                probes["modes"]["zero_bridge_code"]["delta_pred_rms"]
            ),
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
            "note": bridge_note,
        },
        "diagnostics": run.get("diagnostics", {}),
        "gates": {
            "correctness_all_passed": bool(correctness["all_passed"]),
            "smoke_passed": bool(smoke["passed"]),
            "parameter_exact": bool(preflight["bridge"]["parameter_audit"]["candidate_exact"]),
            "dictionary_health": health,
        },
        "historical_unmatched_only": True,
        "matched_baseline_rerun": False,
        "scratch_dev_evidence": "results/e2e_dictenv_latent_bridge_v1/dev/",
        "case": case,
        "verdict": verdict,
    }
    lb.official_test_blocker(summary)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, references, run, probes)
    _write_decision(summary)
    print(f"[analysis] M_S={soup_mae:.6f} band={band} case={case} verdict={verdict}", flush=True)
    return summary


def _write_report(
    summary: Mapping[str, Any],
    references: Mapping[str, Any],
    run: Mapping[str, Any],
    probes: Mapping[str, Any],
) -> None:
    s = summary
    lines: list[str] = []
    lines.append("# e2e_dictenv_latent_bridge_v1 — SEM108 + local task dictionary bridge")
    lines.append("")
    lines.append("CPU only · official test never loaded · single seed-0 trajectory.")
    lines.append("")
    lines.append("## A. Provenance")
    lines.append("")
    lines.append(f"- formal-run commit `{run.get('git_commit', 'n/a')}`")
    lines.append(f"- analysis / report commit `{s['git_commit']}`")
    lines.append(f"- device `cpu` · threads `{s['threads']}` · seed `{s['seed']}`")
    lines.append("- `official_test_loaded = false`")
    lines.append(
        f"- vendored reference package `{references.get('reference_package', '')}`"
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
    lines.append("Sem108: static local object -> fusion -> h (48)")
    lines.append("candidate: h -> shared task dictionary [D_L 48x96, V_L 96x48]")
    lines.append("           -> rho-normalised 16-step unrolled ISTA codes alpha (96)")
    lines.append("           -> E = rho * alpha @ V_L (48) -> unchanged unary + pair + reader")
    lines.append("no graph index, no message passing, no residual bypass, no new raw features")
    lines.append("```")
    lines.append("")
    lines.append("## D. Parameter budget")
    lines.append("")
    p = s["parameters"]
    lines.append(
        f"- Sem108 body `{p['sem108_body']}`, bridge `{p['bridge']}`, candidate `{p['candidate']}` "
        "(D_L / V_L only)"
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
    lines.append("")
    lines.append("## F. Performance interpretation")
    lines.append("")
    lines.append(f"- `M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**")
    background = s["sem108_background"]
    lines.append(
        f"- background Sem108 soup `{_fmt(background['soup_valid_mae'], 6)}` "
        f"(difference `{background['delta_vs_background']:+.6f}`) — unmatched background only, "
        "not a causal increment"
    )
    lines.append("")
    lines.append("## G. Bridge mechanism (frozen soup state, inference only)")
    lines.append("")
    b = s["bridge"]
    lines.append(
        f"- zero bridge code: `delta_mae = {_fmt(b['zero_code_delta_mae'], 6)}`, "
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
    lines.append(
        f"- code density: mean non-zero `{_fmt(b['code_density']['mean_nonzero'], 2)}/96` "
        f"(p50 `{_fmt(b['code_density']['l0_p50'], 1)}`, p95 `{_fmt(b['code_density']['l0_p95'], 1)}`) "
        "— variable density, never a fixed l0"
    )
    lines.append(
        f"- diagnostic low-dimensional reconstruction: mean `{b['low_dimensional_reconstruction']['mean_relative']:.4f}` "
        "(diagnostic only, never an outer loss)"
    )
    lines.append(f"- learned `{b['learned']}` · load bearing `{b['load_bearing']}`")
    lines.append("")
    lines.append("## H. Verdict")
    lines.append("")
    lines.append("**Case {case} — {verdict}**".format(case=s["case"], verdict=s["verdict"]))
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
    lines.append("")
    lines.append("## J. Dev evidence (scratch only, outside the formal chain)")
    lines.append("")
    for name in ("acceptance.json", "short_trainability.json"):
        path = DEV_DIR / name
        lines.append(f"- `dev/{name}`" + (" (absent)" if not path.exists() else ""))
    lines.append("")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    s = summary
    lines = [
        "# DECISION — e2e_dictenv_latent_bridge_v1",
        "",
        f"`M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**.",
        "",
        f"Bridge zero-code `delta_mae = {_fmt(s['bridge']['zero_code_delta_mae'], 6)}`, "
        f"permutation mean `delta_mae = {_fmt(s['bridge']['permutation_mean_delta_mae'], 6)}`, "
        f"reset-to-init `delta_mae = {_fmt(s['bridge']['reset_to_init_delta_mae'], 6)}`.",
        "",
        f"**Case {s['case']} — {s['verdict']}**",
        "",
        "Single seed-0 trajectory, no matched baseline rerun, no official-test read.",
        "Next-step authorization is recorded in the analysis note; no rescue run was executed.",
        "",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# stage: dev (A + B, scratch evidence)
# ---------------------------------------------------------------------------


def _short_train_model(
    model: lb.LatentBridgeSEM108,
    train_data: Sequence[Any],
    *,
    epochs: int,
    seed: int,
    collect: bool,
) -> dict[str, Any]:
    """Matched short loop: parent loss, parent optimizer, no valid access."""
    device = audit.attach_cpu(THREADS)
    p2run._seed_everything(int(seed))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(p2run.LEARNING_RATE),
        weight_decay=float(p2run.WEIGHT_DECAY),
    )
    loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    curve: list[dict[str, float]] = []
    code_rows: list[dict[str, Any]] = []
    model.train()
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
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
        curve.append({"epoch": int(epoch), "train_mae": float(task_sum / max(n_molecules, 1))})
        if collect and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            with torch.no_grad():
                probe = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
                h = _capture_bridge_inputs(model, probe, mask)
                _e, aux_bridge = model.local_dictionary_bridge(h, return_aux=True)
            code_rows.append(
                {
                    "epoch": int(epoch),
                    "code": lb.bridge_code_stats(aux_bridge["alpha"]),
                    "reconstruction": lb.bridge_reconstruction_diagnostic(model, aux_bridge),
                }
            )
    wall = float(time.perf_counter() - started)
    return {
        "train_mae_curve": curve,
        "first_train_mae": float(curve[0]["train_mae"]),
        "final_train_mae": float(curve[-1]["train_mae"]),
        "min_train_mae": float(min(row["train_mae"] for row in curve)),
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "code_rows": code_rows,
        "n_epochs": int(epochs),
    }


def short_trainability(
    *, epochs: int = SHORT_EPOCHS, subset: int = SHORT_SUBSET, seed: int = SEED
) -> dict[str, Any]:
    """B: 40-epoch paired from-scratch trainability check on 1024 train molecules."""
    audit.attach_cpu(THREADS)
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = p1run.load_split("train", subset=int(subset))
    sem_model = sem.build_sem108_model(dictionary, seed, subspace)
    candidate = _model_factory(dictionary, seed, subspace)
    shared_mismatch = [
        key
        for key, value in sem_model.state_dict().items()
        if key in candidate.state_dict()
        and not key.startswith("local_dictionary_bridge.")
        and not torch.equal(value, candidate.state_dict()[key])
    ]
    probe_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    with torch.no_grad():
        candidate.eval()
        h_probe = _capture_bridge_inputs(candidate, probe_batch, cm.C6_MASK)
        _e_init, aux_init = candidate.local_dictionary_bridge(h_probe, return_aux=True)
        e_init = _e_init.detach().clone()
    init_task = _task_gradients(candidate, probe_batch, cm.C6_MASK)

    parent_result = _short_train_model(sem_model, train_data, epochs=epochs, seed=seed, collect=False)
    candidate_result = _short_train_model(
        candidate, train_data, epochs=epochs, seed=seed, collect=True
    )

    with torch.no_grad():
        candidate.eval()
        h_final = _capture_bridge_inputs(candidate, probe_batch, cm.C6_MASK)
        e_final_bridge, aux_final = candidate.local_dictionary_bridge(h_final, return_aux=True)
    e_drift = float(
        torch.norm(e_final_bridge - e_init) / torch.norm(e_init).clamp_min(lb.BRIDGE_EPS)
    )
    task_final = _task_gradients(candidate, probe_batch, cm.C6_MASK)
    movement = {
        "D_L": float(
            (candidate.local_dictionary_bridge.D_L.detach().cpu()
             - candidate.local_dictionary_bridge.D_init.cpu()).norm()
        ),
        "V_L": float(
            (candidate.local_dictionary_bridge.V_L.detach().cpu()
             - candidate.local_dictionary_bridge.V_init.cpu()).norm()
        ),
    }
    paired = [
        {
            "epoch": int(parent_row["epoch"]),
            "parent_train_mae": float(parent_row["train_mae"]),
            "candidate_train_mae": float(candidate_row["train_mae"]),
            "candidate_minus_parent": float(
                candidate_row["train_mae"] - parent_row["train_mae"]
            ),
        }
        for parent_row, candidate_row in zip(
            parent_result["train_mae_curve"], candidate_result["train_mae_curve"]
        )
    ]
    finite = bool(
        math.isfinite(parent_result["final_train_mae"])
        and math.isfinite(candidate_result["final_train_mae"])
        and all(math.isfinite(row["candidate_minus_parent"]) for row in paired)
    )
    learned = bool(
        candidate_result["final_train_mae"] < candidate_result["first_train_mae"]
        and task_final["task_grad_bridge_D_L"] > 0.0
        and task_final["task_grad_bridge_V_L"] > 0.0
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "scope": (
            "scratch/dev short trainability check only; fixed official-train subset; "
            "official valid and official test never loaded; not a formal result"
        ),
        "device": "cpu",
        "threads": THREADS,
        "seed": int(seed),
        "n_molecules": int(subset),
        "epochs": int(epochs),
        "batch_size": int(p2run.BATCH_SIZE),
        "optimizer": {
            "name": "Adam",
            "lr": float(p2run.LEARNING_RATE),
            "weight_decay": float(p2run.WEIGHT_DECAY),
            "grad_clip": float(p2run.GRAD_CLIP),
        },
        "shared_init_mismatch": shared_mismatch,
        "parent": parent_result,
        "candidate": candidate_result,
        "paired": paired,
        "bridge": {
            "init_task_gradients": init_task,
            "final_task_gradients": task_final,
            "parameter_movement": movement,
            "init_to_final_E_relative_l2": e_drift,
            "final_code": (
                candidate_result["code_rows"][-1] if candidate_result["code_rows"] else None
            ),
        },
        "checks": {
            "finite": finite,
            "candidate_learned": learned,
            "shared_init_identical": bool(not shared_mismatch),
            "passed": bool(finite and learned and not shared_mismatch),
        },
        "no_valid_access": True,
        "official_test_loaded": False,
    }
    lb.official_test_blocker(payload)
    return payload


def stage_dev(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    acceptance_path = DEV_DIR / "acceptance.json"
    if acceptance_path.exists() and not force:
        acceptance = _read_json(acceptance_path)
    else:
        acceptance = acceptance_audit()
        lb.official_test_blocker(acceptance)
        _write_json(acceptance_path, acceptance)
    print(f"[dev] acceptance all_passed={acceptance['all_passed']}", flush=True)
    short_path = DEV_DIR / "short_trainability.json"
    if short_path.exists() and not force:
        short = _read_json(short_path)
    else:
        short = short_trainability()
        _write_json(short_path, short)
    print(
        "[dev] short trainability passed={} parent_final={:.6f} candidate_final={:.6f}".format(
            short["checks"]["passed"],
            short["parent"]["final_train_mae"],
            short["candidate"]["final_train_mae"],
        ),
        flush=True,
    )
    return {"acceptance": acceptance, "short_trainability": short}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


STAGES = (
    "references",
    "audit",
    "preflight",
    "correctness",
    "smoke",
    "train",
    "interventions",
    "analysis",
    "dev",
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
    """Execute one named stage (``dev`` is never part of ``chain``)."""
    sem.cpu_only_guard(torch.device("cpu"))
    if stage in ("references", "chain"):
        stage_references()
    if stage in ("audit", "chain"):
        stage_audit(force=force)
    if stage in ("preflight", "chain"):
        stage_preflight()
    if stage in ("correctness", "chain"):
        stage_correctness()
    if stage in ("smoke", "chain"):
        stage_smoke()
    if stage in ("train", "chain"):
        stage_train(force=force)
    if stage in ("interventions", "chain"):
        stage_interventions(force=force)
    if stage in ("analysis", "chain"):
        stage_analysis()
    if stage == "dev":
        stage_dev(force=force)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run_stages(args.stage, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
