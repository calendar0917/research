"""E2E-DictEnv-JointBond-v1 runner — Key-level Joint Structure-Semantics Fusion.

Round ``e2e_dictenv_jointbond_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_jointbond_v1.py``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_jointbond_v1_preregistration.md``.

Stages
------
``references audit preflight correctness smoke train interventions analysis chain``

Exactly one from-scratch ``JointBond`` seed-0 trajectory (320 epochs) on CPU,
extending the frozen ``CSSD-Sem108`` parent.  The official ZINC test split is
never loaded; every payload records ``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_jointbond_v1 as jb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as semrun
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = jb.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_jointbond_v1"
AUDIT_DIR = RESULTS_DIR / "audit"
MECHANISM_DIR = RESULTS_DIR / "mechanism"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_jointbond_v1_preregistration.md"
PRIOR_AUDIT_PATH = NOTES_DIR / "e2e_dictenv_jointbond_v1_prior_artifact_audit.md"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
SEM108_RESULTS = TRACK_ROOT / "results/e2e_dictenv_sem108_v1"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

THREADS = 8
TRAIN_EPOCHS = 320
SEED = 0
SMOKE_EPOCHS = 8
SMOKE_SUBSET = 1024  # 8 x 8 = 64 optimizer steps (<= 64)
SHUFFLE_SEEDS = (101, 202, 303)  # joint branch probes (pre-registration section 6)
PARENT_SHUFFLE_SEEDS = audit.SHUFFLE_SEEDS  # frozen parent assignment shuffles
DIAG_EPOCHS: tuple[int, ...] = (1, 20, 40, 80, 160, 240, 320)

#: pre-registered absolute-MAE performance bands (round brief section 6).
BANDS: tuple[tuple[float, str], ...] = (
    (0.120, "JOINTBOND_FURTHER_CONFIRMATION_BAND"),
    (0.123, "JOINTBOND_CANDIDATE_SIGNAL_BAND"),
    (float("inf"), "JOINTBOND_NO_NEW_TASK_BAND"),
)
BRANCH_LOAD_GATE = 0.003
#: relabel invariance tolerance (float32 index_add_ reduction order).
RELABEL_TOL = 1.0e-4
#: routing reference tolerance: the reference is float64, the model float32.
ROUTING_TOL = 1.0e-4
#: a branch response is "used" if the branch-off degradation exceeds this.
BRANCH_INERT_TOL = 1.0e-6


def _band(mae: float) -> str:
    for threshold, label in BANDS:
        if float(mae) <= threshold:
            return label
    return BANDS[-1][1]


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, AUDIT_DIR, MECHANISM_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# shared model / data helpers
# ---------------------------------------------------------------------------


def _subspace_from_payload(payload: Mapping[str, Any], kind: str = "q1") -> cssd.CommonSubspace:
    entry = payload[kind]
    return cssd.CommonSubspace(
        components=np.asarray(entry["components"], dtype=np.float64),
        rms=np.asarray(entry["rms"], dtype=np.float64),
        kind=str(entry["kind"]),
    )


def _load_parent_subspace() -> cssd.CommonSubspace:
    path = TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
    return _subspace_from_payload(_read_json(path), "q1")


def _dictionary_tensor() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


def _model_factory(dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace) -> jb.JointBondModel:
    return jb.build_jointbond_model(dictionary, int(seed), subspace)


def _soup_state() -> dict[str, torch.Tensor]:
    state = torch.load(
        CHECKPOINT_DIR / "JOINTBOND-seed0_soup_state.pt", map_location="cpu", weights_only=False
    )
    return {key: value.float() for key, value in state.items()}


def _soup_model() -> jb.JointBondModel:
    model = jb.build_jointbond_model(_dictionary_tensor(), SEED, _load_parent_subspace())
    model.load_state_dict(_soup_state())
    return model


def _evaluate(model: jb.JointBondModel, valid_data: Sequence[Any], mask: Any) -> float:
    device = audit.attach_cpu(THREADS)
    loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    return float(audit.evaluate_mask(model, loader, device, mask)["mae"])


def _gradient_diagnostics(
    model: jb.JointBondModel, batch: Any, mask: Any, lam: float
) -> dict[str, Any]:
    """Task + reconstruction gradients; includes the new branch and D."""
    model.train(False)
    prediction, aux = model(batch, mask=mask, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(lam) * rec
    model.zero_grad(set_to_none=True)
    loss.backward()

    def _grad(parameter: torch.nn.Parameter) -> float:
        return float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0

    alpha = aux["coord"][:, model.common_dim :].detach()
    payload = {
        "loss": float(loss.detach()),
        "reconstruction": float(rec.detach()),
        "grad_D": _grad(model.D),
        "grad_fusion_W1": _grad(model.fusion[0].weight),
        "grad_fusion_W2": _grad(model.fusion[2].weight),
        "grad_W_A_S": _grad(model.W_A_S),
        "grad_W_E_S": _grad(model.W_E_S),
        "coord_nnz_per_row_mean": float((alpha != 0).double().sum(dim=1).mean()),
    }
    for name, parameter in model.joint_parameters().items():
        payload[f"grad_{name}"] = _grad(parameter)
    model.zero_grad(set_to_none=True)
    model.train(True)
    return payload


def _branch_activity(model: jb.JointBondModel, batch: Any, mask: Any) -> dict[str, Any]:
    """The branch must change the prediction; branch-off must reproduce the parent."""
    model.eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        delta = model.joint_edge_delta(coord, batch, mask=mask)
        delta_off = model.joint_edge_delta(
            coord, batch, mask=jb.merge_joint_mask(mask, jb.JointBondMask(joint_branch_off=True))
        )
        pred_on = model(batch, mask=mask)
        pred_off = model(
            batch,
            mask=jb.merge_joint_mask(mask, jb.JointBondMask(joint_branch_off=True)),
        )
    branch_norm = float(delta.double().norm()) if delta is not None else 0.0
    return {
        "branch_response_norm": branch_norm,
        "branch_response_absmax": float(delta.abs().max()) if delta is not None else 0.0,
        "branch_off_delta_is_none": bool(delta_off is None),
        "prediction_diff_on_vs_off": float((pred_on - pred_off).abs().max()),
        "passed": bool(branch_norm > 0.0 and float((pred_on - pred_off).abs().max()) > 0.0),
    }


# ---------------------------------------------------------------------------
# stage: references
# ---------------------------------------------------------------------------


def stage_references() -> dict[str, Any]:
    _ensure_dirs()

    def _read(relative: str) -> dict[str, Any] | None:
        path = REPO_ROOT / relative
        return _read_json(path) if path.exists() else None

    sem_summary = _read("tracks/ksvd/results/e2e_dictenv_sem108_v1/summary.json")
    sem_run = _read("tracks/ksvd/results/e2e_dictenv_sem108_v1/run_seed0.json")
    cssd_final = _read("tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json")
    rndb = _read("tracks/ksvd/results/e2e_dictenv_rndb_v1/summary.json")
    clean = _read("tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json")
    t1 = _read("tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json")
    tpa = _read("tracks/ksvd/results/e2e_dictenv_training_protocol_audit_v1/comparison.json")

    def _value(payload: Mapping[str, Any] | None, *keys: str) -> float | None:
        if payload is None:
            return None
        current: Any = payload
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        return float(current) if current is not None else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "policy": (
            "read-only durable artifacts; no historical model is retrained; comparisons are "
            "historical / unmatched / contextual only; there is no contemporaneous baseline"
        ),
        "references": {
            "SEM108-parent-seed0": {
                "value": _value(sem_summary, "M_S"),
                "best_valid_mae": _value(sem_run, "best_valid_mae"),
                "best_epoch": int(sem_run["best_epoch"]) if sem_run else None,
                "actual_params": int(sem_run["actual_params"]) if sem_run else None,
                "artifact": "tracks/ksvd/results/e2e_dictenv_sem108_v1/summary.json",
                "status": "not_rerun_unmatched_context",
            },
            "CSSD-q1-seed0": {
                "value": _value(cssd_final, "soup", "soup_valid_mae"),
                "artifact": "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json",
                "status": "not_rerun_unmatched_context",
            },
            "RNDB-seed0": {
                "value": _value(rndb, "M_R"),
                "artifact": "tracks/ksvd/results/e2e_dictenv_rndb_v1/summary.json",
                "status": "not_rerun_unmatched_context",
            },
            "FINAL-CLEAN-C6-seed0": {
                "value": _value(clean, "baseline_valid_mae"),
                "artifact": "tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json",
                "status": "not_rerun_unmatched_context",
            },
            "T1-tuned-seed0": {
                "value": _value(t1, "best_tuned_soup_valid_mae"),
                "artifact": "tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json",
                "status": "not_rerun_unmatched_context",
            },
            "training-protocol-audit-control-lr1e3": {
                "value": _value(tpa, "arms", "control_lr1e3", "soup_valid_mae"),
                "artifact": "tracks/ksvd/results/e2e_dictenv_training_protocol_audit_v1/comparison.json",
                "status": "not_rerun_unmatched_context",
            },
        },
    }
    jb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "historical_references.json", payload)
    print(
        "[references] SEM108-parent={:.6f} CSSD-q1={} RNDB={}".format(
            payload["references"]["SEM108-parent-seed0"]["value"],
            payload["references"]["CSSD-q1-seed0"]["value"],
            payload["references"]["RNDB-seed0"]["value"],
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: Phase-A audit (parent artifacts, read-only)
# ---------------------------------------------------------------------------


def stage_audit(force: bool = False) -> dict[str, Any]:
    """Phase A: reuse the frozen parent's durable audits (no re-run, no training).

    The JointBond round changes no input feature and no parent code path, so the
    Sem108 identity / relation audits remain valid; this stage only reads them
    and checks that the parent code the audits described is still the parent.
    """
    _ensure_dirs()
    jb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "reused_parent_artifacts": {},
    }
    checks: dict[str, Any] = {}
    for name, relative in (
        ("sem108_identity", "tracks/ksvd/results/e2e_dictenv_sem108_v1/audit/sem108_identity.json"),
        ("anchor_relation", "tracks/ksvd/results/e2e_dictenv_sem108_v1/audit/anchor_relation.json"),
        ("audit_decision", "tracks/ksvd/results/e2e_dictenv_sem108_v1/audit/audit_decision.json"),
    ):
        path = REPO_ROOT / relative
        entry: dict[str, Any] = {"path": relative, "exists": path.exists()}
        if path.exists():
            data = _read_json(path)
            entry["sha256"] = _sha256_file(path)
            entry["official_test_loaded"] = data.get("official_test_loaded")
        payload["reused_parent_artifacts"][name] = entry
        if not path.exists():
            raise FileNotFoundError(
                f"frozen parent audit artifact missing: {relative}; the JointBond round "
                "reuses the Sem108 Phase-A evidence instead of retraining it"
            )
    identity = _read_json(REPO_ROOT / payload["reused_parent_artifacts"]["sem108_identity"]["path"])
    relation = _read_json(REPO_ROOT / payload["reused_parent_artifacts"]["anchor_relation"]["path"])
    decision = _read_json(REPO_ROOT / payload["reused_parent_artifacts"]["audit_decision"]["path"])
    checks["identity_established"] = bool(identity.get("identity_established"))
    checks["a2_pass"] = bool(relation.get("a2_pass"))
    checks["parent_decision"] = str(decision.get("decision"))
    checks["scaler_fit_split"] = str(identity["standardization"]["fit_split"])
    checks["hook_is_default_none"] = bool(
        sem.SEM108Model._edge_response_delta(None, None, None, None, None) is None
    )
    checks["joint_model_overrides_hook"] = bool(
        jb.JointBondModel._edge_response_delta is not sem.SEM108Model._edge_response_delta
    )
    payload["checks"] = checks
    payload["decision"] = (
        "PROCEED"
        if (
            checks["identity_established"]
            and checks["a2_pass"]
            and checks["parent_decision"] == "PROCEED"
            and checks["scaler_fit_split"] == "official train"
            and checks["hook_is_default_none"]
            and checks["joint_model_overrides_hook"]
        )
        else "STOP"
    )
    payload["rationale"] = (
        "the Sem108 Phase-A identity / marginal-relation audits are reused unchanged (the "
        "JointBond round touches neither the input interface nor the parent code path); the "
        "parent hook default is verified to be None so the frozen forward is untouched."
    )
    jb.official_test_blocker(payload)
    _write_json(AUDIT_DIR / "audit_decision.json", payload)
    print(
        f"[audit] parent_decision={checks['parent_decision']} decision={payload['decision']}",
        flush=True,
    )
    if payload["decision"] != "PROCEED":
        raise RuntimeError(f"Phase-A stop rule fired: {payload}")
    return payload


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    jb.cpu_only_guard(torch.device("cpu"))
    decision = _read_json(AUDIT_DIR / "audit_decision.json")
    if decision.get("decision") != "PROCEED":
        raise RuntimeError("Phase-A audit did not authorise the formal run")
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    if not PRIOR_AUDIT_PATH.exists():
        raise FileNotFoundError(f"prior-artifact audit missing: {PRIOR_AUDIT_PATH}")
    prereg_sha = _sha256_file(PREREG_PATH)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    parent = jb.build_jointbond_model(dictionary, SEED, subspace)
    sem_parent = sem.build_sem108_model(dictionary, SEED, subspace)
    parent_params = int(sum(p.numel() for p in sem_parent.parameters()))
    audit_payload = jb.parameter_audit(parent, parent_params)
    for key in ("new_params_exact", "total_exact"):
        if not audit_payload[key]:
            raise RuntimeError(
                f"frozen JointBond parameter contract violated ({key}): {audit_payload}"
            )
    layout = p2.decoder_layers(cm.H1_CONFIG)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": prereg_sha,
            "prior_audit_path": str(PRIOR_AUDIT_PATH.relative_to(REPO_ROOT)),
            "prior_audit_sha256": _sha256_file(PRIOR_AUDIT_PATH),
        },
        "parent": {
            "model": jb.PARENT_MODEL,
            "commit": jb.PARENT_COMMIT,
            "config": cm.H1_CONFIG.as_dict(),
            "lambda_rec": float(cm.H1_LAMBDA),
            "c6_mask_identity": bool(cssd.CSSD_MASK is cm.C6_MASK),
            "c6_equivalence_check": bool(cm.c6_equivalence_check()),
            "subspace_kind": subspace.kind,
            "common_dim": int(subspace.q),
            "common_rms": [float(value) for value in subspace.rms],
            "dictionary_sha256": p2run.dictionary_sha256(cm.H1_CONFIG.dict_kind),
            "dictionary_shape": list(np.asarray(dictionary).shape),
            "interface_dim": int(sem.SEM_INTERFACE_DIM),
            "fusion_input_dim": int(sem.SEM_FUSION_IN),
            "fusion_hidden": int(sem.SEM_FUSION_HIDDEN),
            "node_slots": [int(p2.N_SHELLS), int(p2.D_A)],
            "edge_slots": [int(p2.SHELLPAIR_CLASSES), int(cm.D_E)],
            "parent_soup_valid_mae": float(jb.PARENT_SOUP_VALID_MAE),
            "decoder_layout": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in layout.items()
            },
        },
        "joint_branch": {
            "alpha_dim": int(jb.JOINT_ALPHA_DIM),
            "atom_classes": int(jb.JOINT_ATOM_DIM),
            "bond_classes": int(jb.JOINT_BOND_DIM),
            "h": int(jb.JOINT_H),
            "t": int(jb.JOINT_T_DIM),
            "f_hidden": int(jb.JOINT_F_HIDDEN),
            "init": {
                "A_C_B_F1": "kaiming_uniform_(a=sqrt(5))",
                "F2": f"normal_(0, {jb.JOINT_INIT_STD})",
            },
            "insertion_point": "per-bond edge response, before the shellpair index_add_",
            "reads_common_coordinate": False,
        },
        "parameter_audit": audit_payload,
        "epochs": TRAIN_EPOCHS,
        "seed": SEED,
    }
    jb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration"])
    _write_json(RESULTS_DIR / "parameter_audit.json", audit_payload)
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {prereg_sha[:12]} parent={parent_params} candidate="
        f"{audit_payload['candidate_params']} new={audit_payload['new_params']} "
        f"rel={audit_payload['relative_delta']:.5f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: correctness gates
# ---------------------------------------------------------------------------


def _real_batch(subset: int = 32, split: str = "train") -> Any:
    data_list = p1run.load_split(split, subset=subset)
    return next(iter(p1.make_env_loader(data_list, int(subset), False, 0)))


def stage_correctness() -> dict[str, Any]:
    _ensure_dirs()
    jb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    model = jb.build_jointbond_model(dictionary, SEED, subspace)
    parent = sem.build_sem108_model(dictionary, SEED, subspace)
    mask = cm.C6_MASK
    gates: dict[str, Any] = {}

    # -- J0/J1: frozen parent interface and Phase-A provenance ----------------
    geometry = sem.resolve_sem108_geometry()
    audit_decision = _read_json(AUDIT_DIR / "audit_decision.json")
    gates["J0_parent_interface_reused"] = {
        "sem_dim": int(geometry["sem_dim"]),
        "interface_dim": int(sem.SEM_INTERFACE_DIM),
        "fusion_in": int(sem.SEM_FUSION_IN),
        "fusion_hidden": int(sem.SEM_FUSION_HIDDEN),
        "c6_mask_identity": bool(cssd.CSSD_MASK is cm.C6_MASK),
        "c6_equivalence_check": bool(cm.c6_equivalence_check()),
        "passed": bool(
            geometry["sem_dim"] == 108
            and sem.SEM_INTERFACE_DIM == 110
            and cssd.CSSD_MASK is cm.C6_MASK
            and cm.c6_equivalence_check()
        ),
    }
    gates["J1_prereg_phase_a"] = {
        "phase_a_decision": str(audit_decision["decision"]),
        "parent_identity": bool(audit_decision["checks"]["identity_established"]),
        "parent_a2": bool(audit_decision["checks"]["a2_pass"]),
        "prereg_exists": bool(PREREG_PATH.exists()),
        "prereg_sha256": _sha256_file(PREREG_PATH) if PREREG_PATH.exists() else None,
        "passed": bool(
            audit_decision["decision"] == "PROCEED"
            and audit_decision["checks"]["identity_established"]
            and audit_decision["checks"]["a2_pass"]
            and PREREG_PATH.exists()
        ),
    }

    # -- J2: frozen widths / exact parameter contract -------------------------
    parent_params = int(sum(p.numel() for p in parent.parameters()))
    audit_payload = jb.parameter_audit(model, parent_params)
    shapes = {
        "joint_A": list(model.joint_A.weight.shape),
        "joint_C": list(model.joint_C.weight.shape),
        "joint_B": list(model.joint_B.weight.shape),
        "joint_F_0": list(model.joint_F[0].weight.shape),
        "joint_F_2": list(model.joint_F[2].weight.shape),
    }
    expected_shapes = {
        "joint_A": [jb.JOINT_H, jb.JOINT_ALPHA_DIM],
        "joint_C": [jb.JOINT_H, jb.JOINT_ATOM_DIM],
        "joint_B": [jb.JOINT_T_DIM, jb.JOINT_BOND_DIM],
        "joint_F_0": [jb.JOINT_F_HIDDEN, jb.JOINT_T_DIM],
        "joint_F_2": [jb.JOINT_T_DIM, jb.JOINT_F_HIDDEN],
    }
    gates["J2_parameter_contract"] = {
        **audit_payload,
        "shapes": shapes,
        "shapes_frozen": bool(shapes == expected_shapes),
        "passed": bool(
            audit_payload["new_params_exact"]
            and audit_payload["total_exact"]
            and shapes == expected_shapes
            and audit_payload["candidate_params"] == jb.JOINT_EXPECTED_TOTAL
        ),
    }

    # -- J3: parent shared parameters bit-identical ---------------------------
    state_joint = model.state_dict()
    state_parent = parent.state_dict()
    shared = {
        key: value
        for key, value in state_parent.items()
        if key in state_joint
        and state_joint[key].shape == value.shape
        and not key.startswith(("fusion.", "anchor_encoder."))
    }
    shared_mismatch = [
        key for key, value in shared.items() if not torch.equal(value, state_joint[key])
    ]
    new_keys = sorted(set(state_joint) - set(state_parent))
    removed_keys = sorted(set(state_parent) - set(state_joint))
    expected_new_keys = sorted(
        [
            "joint_A.weight",
            "joint_C.weight",
            "joint_B.weight",
            "joint_F.0.weight",
            "joint_F.2.weight",
        ]
    )
    gates["J3_parent_shared_bit_identical"] = {
        "shared_state_keys": int(len(shared)),
        "shared_state_mismatches": shared_mismatch,
        "new_keys": new_keys,
        "removed_keys": removed_keys,
        "passed": bool(not shared_mismatch and new_keys == expected_new_keys and not removed_keys),
    }

    # -- J4: hook default + branch-off full-prediction equivalence -------------
    equivalence = jb.build_jointbond_model(dictionary, SEED, subspace)
    equivalence.load_state_dict(parent.state_dict(), strict=False)
    model.eval()
    parent.eval()
    equivalence.eval()
    batch = _real_batch(subset=32, split="train")
    off_mask = jb.merge_joint_mask(mask, jb.JointBondMask(joint_branch_off=True))
    with torch.no_grad():
        pred_parent, aux_parent = parent(batch, mask=mask, return_aux=True)
        pred_equiv, aux_equiv = equivalence(batch, mask=mask, return_aux=True)
        pred_equiv_off, aux_equiv_off = equivalence(batch, mask=off_mask, return_aux=True)
        pred_joint_on, _aux_on = model(batch, mask=mask, return_aux=True)
        slot_ref_off = jb.reference_edge_slots(
            equivalence, equivalence.code(batch.dict_phi), batch, mask=off_mask
        )
        slots_off = equivalence.joint_edge_slots(equivalence.code(batch.dict_phi), batch, off_mask)
        slot_ref_on = jb.reference_edge_slots(model, model.code(batch.dict_phi), batch, mask=mask)
        slots_on = model.joint_edge_slots(model.code(batch.dict_phi), batch, mask)
    gates["J4_parent_equivalence_and_edges"] = {
        "hook_default_none": bool(
            sem.SEM108Model._edge_response_delta(None, None, None, None, None) is None
        ),
        "branch_off_prediction_max_abs_diff": float((pred_equiv_off - pred_parent).abs().max()),
        "branch_off_environment_max_abs_diff": float((aux_equiv_off["E"] - aux_parent["E"]).abs().max()),
        "branch_off_edge_slot_reference_max_abs_diff": float(
            np.abs(slot_ref_off - slots_off.double().numpy()).max()
        ),
        "branch_on_edge_slot_reference_max_abs_diff": float(
            np.abs(slot_ref_on - slots_on.double().numpy()).max()
        ),
        "branch_on_vs_off_prediction_max_abs_diff": float(
            (pred_joint_on - pred_equiv_off).abs().max()
        ),
        "passed": bool(
            torch.equal(pred_equiv_off, pred_parent)
            and torch.equal(aux_equiv_off["E"], aux_parent["E"])
            and float(np.abs(slot_ref_off - slots_off.double().numpy()).max()) <= ROUTING_TOL
            and float(np.abs(slot_ref_on - slots_on.double().numpy()).max()) <= ROUTING_TOL
            and float((pred_joint_on - pred_equiv_off).abs().max()) > 0.0
        ),
    }
    # bit-identity of the *equiv* (untrained-branch) forward must hold exactly
    gates["J4_parent_equivalence_and_edges"]["branch_off_bit_identical"] = bool(
        torch.equal(pred_equiv_off, pred_parent)
    )
    gates["J3_parent_shared_bit_identical"]["parent_equivalence_forward_diff"] = float(
        (pred_equiv - pred_parent).abs().max()
    )
    gates["J3_parent_shared_bit_identical"]["partial_load_shared_exact"] = bool(
        torch.equal(pred_equiv, pred_parent)
    )

    # -- J5: endpoint correspondence (joint swap invariant; single swaps move) --
    model.eval()
    parent.eval()
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        alpha = coord[:, model.common_dim :]
        q = torch.nn.functional.one_hot(batch.dict_atom, num_classes=jb.JOINT_ATOM_DIM).to(coord.dtype)
        bu = batch.env_bond_u
        bv = batch.env_bond_v
        au, qu, av, qv, bt = alpha[bu], q[bu], alpha[bv], q[bv], batch.env_bond_type
        j_here = jb.joint_branch_from_endpoints(model, au, qu, av, qv, bt)
        j_swap_both = jb.joint_branch_from_endpoints(model, av, qv, au, qu, bt)
        j_swap_atom = jb.joint_branch_from_endpoints(model, au, qv, av, qu, bt)
        j_swap_struct = jb.joint_branch_from_endpoints(model, av, qu, au, qv, bt)
        j_routed = model.joint_edge_response(coord, batch, bu, bv, mask)
    both_diff = float((j_here - j_swap_both).abs().max())
    atom_diff = float((j_here - j_swap_atom).abs().max())
    struct_diff = float((j_here - j_swap_struct).abs().max())
    atom_changed = float(((j_here - j_swap_atom).abs().sum(dim=1) > 0).double().mean())
    struct_changed = float(((j_here - j_swap_struct).abs().sum(dim=1) > 0).double().mean())
    both_endpoint_diff = float(((au - av).abs().sum(dim=1) > 0).double().mean())
    distinct_atoms = float((batch.dict_atom[bu] != batch.dict_atom[bv]).double().mean())
    gates["J5_endpoint_correspondence"] = {
        "routed_vs_endpoint_formulation_max_abs_diff": float((j_routed - j_here).abs().max()),
        "joint_swap_both_max_abs_diff": both_diff,
        "atom_only_swap_max_abs_diff": atom_diff,
        "structure_only_swap_max_abs_diff": struct_diff,
        "atom_only_swap_changed_bond_fraction": atom_changed,
        "structure_only_swap_changed_bond_fraction": struct_changed,
        "distinct_endpoint_atoms_fraction": distinct_atoms,
        "distinct_endpoint_codes_fraction": both_endpoint_diff,
        "passed": bool(
            float((j_routed - j_here).abs().max()) == 0.0
            and both_diff == 0.0
            and atom_diff > 0.0
            and struct_diff > 0.0
            and atom_changed > 0.0
            and struct_changed > 0.0
        ),
    }

    # -- J6: synthetic distinguishable endpoint pairs (no accidental symmetry) --
    with torch.no_grad():
        gen = torch.Generator().manual_seed(101)
        alpha_syn = torch.randn(4, jb.JOINT_ALPHA_DIM, generator=gen)
        q_syn = torch.nn.functional.one_hot(
            torch.tensor([0, 3, 7, 15]), num_classes=jb.JOINT_ATOM_DIM
        ).to(torch.float32)
        alpha_u, alpha_v = alpha_syn[0:1], alpha_syn[1:2]
        q_u, q_v = q_syn[0:1], q_syn[1:2]
        bt_syn = torch.tensor([1])
        j_syn = jb.joint_branch_from_endpoints(model, alpha_u, q_u, alpha_v, q_v, bt_syn)
        j_syn_swap_both = jb.joint_branch_from_endpoints(model, alpha_v, q_v, alpha_u, q_u, bt_syn)
        j_syn_swap_atom = jb.joint_branch_from_endpoints(model, alpha_u, q_v, alpha_v, q_u, bt_syn)
        j_syn_swap_struct = jb.joint_branch_from_endpoints(model, alpha_v, q_u, alpha_u, q_v, bt_syn)
        j_syn_zero = jb.joint_branch_from_endpoints(
            model, torch.zeros_like(alpha_u), q_u, torch.zeros_like(alpha_v), q_v, bt_syn
        )
    gates["J6_synthetic_endpoint_pairs"] = {
        "sample_structure_max_abs_diff": float((alpha_u - alpha_v).abs().max()),
        "sample_atom_distinct": bool(int(q_u.argmax()) != int(q_v.argmax())),
        "joint_swap_both_max_abs_diff": float((j_syn - j_syn_swap_both).abs().max()),
        "atom_only_swap_max_abs_diff": float((j_syn - j_syn_swap_atom).abs().max()),
        "structure_only_swap_max_abs_diff": float((j_syn - j_syn_swap_struct).abs().max()),
        "zero_structure_max_abs": float(j_syn_zero.abs().max()),
        "passed": bool(
            float((alpha_u - alpha_v).abs().max()) > 0.0
            and int(q_u.argmax()) != int(q_v.argmax())
            and float((j_syn - j_syn_swap_both).abs().max()) == 0.0
            and float((j_syn - j_syn_swap_atom).abs().max()) > 0.0
            and float((j_syn - j_syn_swap_struct).abs().max()) > 0.0
            and float(j_syn_zero.abs().max()) == 0.0
        ),
    }

    # -- J7: branch-local residual-code zero + no gate/normalisation ----------
    zero_mask = jb.merge_joint_mask(mask, jb.JointBondMask(joint_alpha_zero=True))
    model.eval()
    with torch.no_grad():
        coord_zero = model.code(batch.dict_phi)
        delta_zero = model.joint_edge_delta(coord_zero, batch, mask=zero_mask)
        pred_zero = model(batch, mask=zero_mask)
        pred_off = model(batch, mask=off_mask)
        pred_on = model(batch, mask=mask)
        common_zeroed = torch.cat(
            [
                torch.zeros_like(coord_zero[:, : model.common_dim]),
                coord_zero[:, model.common_dim :],
            ],
            dim=1,
        )
        branch_common_zeroed = model.joint_edge_response(
            common_zeroed, batch, batch.env_bond_u, batch.env_bond_v, mask
        )
        branch_common_plain = model.joint_edge_response(
            coord_zero, batch, batch.env_bond_u, batch.env_bond_v, mask
        )
    reads_common = bool(float((branch_common_zeroed - branch_common_plain).abs().max()) > 0.0)
    no_bias_or_norm = bool(
        all(
            layer.bias is None
            for layer in (
                model.joint_A,
                model.joint_C,
                model.joint_B,
                model.joint_F[0],
                model.joint_F[2],
            )
        )
        and [type(module).__name__ for module in model.joint_F] == ["Linear", "SiLU", "Linear"]
    )
    gates["J7_residual_code_zero"] = {
        "branch_output_absmax_with_alpha_zero": float(delta_zero.abs().max()),
        "branch_output_is_none_with_branch_off": bool(
            model.joint_edge_delta(coord_zero, batch, mask=off_mask) is None
        ),
        "prediction_diff_alpha_zero_vs_branch_off": float((pred_zero - pred_off).abs().max()),
        "prediction_diff_branch_on_vs_off": float((pred_on - pred_off).abs().max()),
        "zeroed_alpha_is_exactly_branch_off": bool(torch.equal(pred_zero, pred_off)),
        "reads_common_coordinate": reads_common,
        "branch_has_no_bias_or_norm": no_bias_or_norm,
        "extra_modules": [type(module).__name__ for module in model.joint_F],
        "passed": bool(
            float(delta_zero.abs().max()) == 0.0
            and torch.equal(pred_zero, pred_off)
            and float((pred_on - pred_off).abs().max()) > 0.0
            and not reads_common
            and no_bias_or_norm
        ),
    }

    # -- J8: gradients reach the branch and the dictionary --------------------
    gradients = _gradient_diagnostics(model, batch, mask, cm.H1_LAMBDA)
    gradient_keys = [f"grad_{name}" for name in model.joint_parameters()]
    gates["J8_gradients"] = {
        **gradients,
        "branch_gradient_keys": gradient_keys,
        "passed": bool(
            all(gradients[key] > 0.0 for key in gradient_keys)
            and gradients["grad_D"] > 0.0
            and gradients["grad_fusion_W1"] > 0.0
        ),
    }

    # -- J9: routing / batch offsets on a two-graph synthetic batch ------------
    model.eval()
    synthetic = jb.synthetic_batch(n_nodes=40, n_occ=120, n_bond=90, seed=7, n_graphs=2)
    generator = torch.Generator().manual_seed(11)
    rows_per_graph = [(synthetic.batch == graph).nonzero(as_tuple=False).view(-1) for graph in (0, 1)]
    bond_u: list[int] = []
    bond_v: list[int] = []
    bond_root: list[int] = []
    for rows in rows_per_graph:
        endpoints = rows[torch.randint(0, int(rows.numel()), (45, 2), generator=generator)]
        bond_u.extend(int(value) for value in endpoints[:, 0])
        bond_v.extend(int(value) for value in endpoints[:, 1])
        bond_root.extend(
            int(value)
            for value in rows[torch.randint(0, int(rows.numel()), (45,), generator=generator)]
        )
    synthetic.env_bond_u = torch.tensor(bond_u, dtype=torch.long)
    synthetic.env_bond_v = torch.tensor(bond_v, dtype=torch.long)
    synthetic.env_bond_root = torch.tensor(bond_root, dtype=torch.long)
    with torch.no_grad():
        coord_syn = model.code(synthetic.dict_phi)
        slots_syn = model.joint_edge_slots(coord_syn, synthetic, mask)
        slot_reference = jb.reference_edge_slots(model, coord_syn, synthetic, mask=mask)
        delta_syn = model.joint_edge_response(
            coord_syn, synthetic, synthetic.env_bond_u, synthetic.env_bond_v, mask
        )
        alpha_syn2 = coord_syn[:, model.common_dim :]
        q_syn2 = torch.nn.functional.one_hot(
            synthetic.dict_atom, num_classes=jb.JOINT_ATOM_DIM
        ).to(coord_syn.dtype)
        delta_manual = jb.joint_branch_from_endpoints(
            model,
            alpha_syn2[synthetic.env_bond_u],
            q_syn2[synthetic.env_bond_u],
            alpha_syn2[synthetic.env_bond_v],
            q_syn2[synthetic.env_bond_v],
            synthetic.env_bond_type,
        )
        shifted_u = (synthetic.env_bond_u + 1) % int(coord_syn.shape[0])
        delta_shifted = model.joint_edge_response(
            coord_syn, synthetic, shifted_u, synthetic.env_bond_v, mask
        )
        cross_graph = float(
            (
                (synthetic.batch[synthetic.env_bond_u] != synthetic.batch[synthetic.env_bond_v])
            )
            .double()
            .mean()
        )
    gates["J9_routing_and_batch_offsets"] = {
        "synthetic_edge_slot_reference_max_abs_diff": float(
            np.abs(slot_reference - slots_syn.double().numpy()).max()
        ),
        "routed_delta_vs_manual_max_abs_diff": float((delta_syn - delta_manual).abs().max()),
        "shifted_index_delta_max_abs_diff": float((delta_syn - delta_shifted).abs().max()),
        "n_graphs": int(synthetic.num_graphs),
        "cross_graph_bond_fraction": cross_graph,
        "passed": bool(
            float(np.abs(slot_reference - slots_syn.double().numpy()).max()) <= ROUTING_TOL
            and float((delta_syn - delta_manual).abs().max()) == 0.0
            and float((delta_syn - delta_shifted).abs().max()) > 0.0
            and cross_graph == 0.0
        ),
    }

    # -- J10: parent structure shuffle keeps endpoint semantics -----------------
    model.eval()
    parent.eval()
    valid_data = p1run.load_split("valid", subset=8)
    audit.prepare_shuffles(valid_data, "edge", int(SHUFFLE_SEEDS[0]))
    try:
        shuffled_batch = next(iter(p1.make_env_loader(valid_data, 8, False, 0)))
    finally:
        audit.clear_shuffles(valid_data)
    with torch.no_grad():
        coord_sh = model.code(shuffled_batch.dict_phi)
        su = shuffled_batch.env_bond_u_shuffled
        sv = shuffled_batch.env_bond_v_shuffled
        delta_sh = model.joint_edge_response(coord_sh, shuffled_batch, su, sv, mask)
        alpha_flat = coord_sh[:, model.common_dim :]
        q_flat = torch.nn.functional.one_hot(
            shuffled_batch.dict_atom, num_classes=jb.JOINT_ATOM_DIM
        ).to(coord_sh.dtype)
        delta_sh_manual = jb.joint_branch_from_endpoints(
            model,
            alpha_flat[su],
            q_flat[su],
            alpha_flat[sv],
            q_flat[sv],
            shuffled_batch.env_bond_type,
        )
        delta_sh_unshuffled = model.joint_edge_response(
            coord_sh, shuffled_batch, shuffled_batch.env_bond_u, shuffled_batch.env_bond_v, mask
        )
        pred_sh_off = model(
            shuffled_batch,
            mask=jb.merge_joint_mask(mask, jb.JointBondMask(joint_branch_off=True)),
        )
        parent_sh_off = parent(
            shuffled_batch,
            mask=jb.merge_joint_mask(mask, jb.JointBondMask(joint_branch_off=True)),
        )
    gates["J10_structure_shuffle_endpoint_semantics"] = {
        "endpoint_shuffle_defined": bool(su is not None and sv is not None),
        "shuffled_delta_vs_endpoint_formulation_max_abs_diff": float(
            (delta_sh - delta_sh_manual).abs().max()
        ),
        "shuffled_vs_unshuffled_delta_max_abs_diff": float(
            (delta_sh - delta_sh_unshuffled).abs().max()
        ),
        "branch_off_shuffle_prediction_max_abs_diff_vs_parent": float(
            (pred_sh_off - parent_sh_off).abs().max()
        ),
        "passed": bool(
            su is not None
            and sv is not None
            and float((delta_sh - delta_sh_manual).abs().max()) == 0.0
            and float((delta_sh - delta_sh_unshuffled).abs().max()) > 0.0
            and torch.equal(pred_sh_off, parent_sh_off)
        ),
    }

    # -- J11: frozen parent soup reproduction (no parent drift) ---------------
    parent_drift: dict[str, Any] = {"checked": False}
    sem_soup_path = SEM108_RESULTS / "checkpoints/SEM108-seed0_soup_state.pt"
    sem_summary_path = SEM108_RESULTS / "summary.json"
    if sem_soup_path.exists() and sem_summary_path.exists():
        frozen_soup = sem.build_sem108_model(dictionary, SEED, subspace)
        frozen_soup.load_state_dict(
            {
                key: value.float()
                for key, value in torch.load(sem_soup_path, map_location="cpu", weights_only=False).items()
            }
        )
        frozen_soup.eval()
        recorded = float(_read_json(sem_summary_path)["M_S"])
        recomputed = _evaluate(frozen_soup, p1run.load_split("valid"), mask)
        parent_drift = {
            "checked": True,
            "recorded_sem108_soup_valid_mae": recorded,
            "recomputed_sem108_soup_valid_mae": float(recomputed),
            "abs_diff": float(abs(recomputed - recorded)),
            "bit_identical": bool(recomputed == recorded),
        }
    gates["J11_frozen_parent_reproduction"] = {
        **parent_drift,
        "tolerance": 1.0e-9,
        "passed": bool(parent_drift["checked"] and float(parent_drift["abs_diff"]) <= 1.0e-9),
    }

    # -- J12: relabel invariance ----------------------------------------------
    model.eval()
    gates["J12_relabel_invariance"] = semrun._relabel_invariance(model, n_molecules=2)

    # -- J13: official-test blocker -------------------------------------------
    blocked = False
    try:
        jb.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocked = True
    gates["J13_official_test_blocker"] = {
        "blocker_raises_on_true": bool(blocked),
        "test_split_loaded": False,
        "passed": bool(blocked),
    }

    all_passed = all(bool(entry.get("passed")) for entry in gates.values() if "passed" in entry)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "gates": gates,
        "all_passed": bool(all_passed),
    }
    jb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "correctness.json", payload)
    for key in sorted(gates):
        print(f"[correctness] {key} passed={gates[key].get('passed')}", flush=True)
    if not all_passed:
        raise RuntimeError(
            f"correctness gates failed: {[k for k, v in gates.items() if not v.get('passed')]}"
        )
    return payload


# ---------------------------------------------------------------------------
# stage: smoke (<= 64 optimizer steps, trainability only)
# ---------------------------------------------------------------------------


def stage_smoke() -> dict[str, Any]:
    _ensure_dirs()
    jb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = p1run.load_split("train", subset=SMOKE_SUBSET)
    valid_data = p1run.load_split("valid", subset=512)
    model = _model_factory(dictionary, SEED, subspace)
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    init_grads = _gradient_diagnostics(model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    init_activity = _branch_activity(model, gradient_batch, cm.C6_MASK)
    init_scale = jb.response_scale_stats(model, model.code(gradient_batch.dict_phi), gradient_batch, cm.C6_MASK)
    result = cssd.train_cssd(
        tag="JOINTBOND-smoke",
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
                CHECKPOINT_DIR / "JOINTBOND-smoke_final_state.pt", map_location="cpu", weights_only=False
            ).items()
        }
    )
    post_grads = _gradient_diagnostics(final_model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    post_activity = _branch_activity(final_model, gradient_batch, cm.C6_MASK)
    post_scale = jb.response_scale_stats(
        final_model, final_model.code(gradient_batch.dict_phi), gradient_batch, cm.C6_MASK
    )
    finite = bool(
        math.isfinite(result["best_valid_mae"])
        and math.isfinite(result["final_valid_mae"])
        and all(math.isfinite(row["train_mae"]) for row in result["curve"])
    )
    gradient_keys = [f"grad_{name}" for name in final_model.joint_parameters()] + [
        "grad_D",
        "grad_fusion_W1",
        "grad_fusion_W2",
        "grad_W_A_S",
        "grad_W_E_S",
    ]
    gradient_ok = bool(
        all(init_grads[key] > 0.0 for key in gradient_keys)
        and all(post_grads[key] > 0.0 for key in gradient_keys)
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
        "optimizer_step_budget": 64,
        "finite": finite,
        "gradient_ok": gradient_ok,
        "init_gradients": init_grads,
        "post_gradients": post_grads,
        "init_branch_activity": init_activity,
        "post_branch_activity": post_activity,
        "init_response_scale": init_scale,
        "post_response_scale": post_scale,
        "best_valid_mae": float(result["best_valid_mae"]),
        "final_valid_mae": float(result["final_valid_mae"]),
        "train_curve": result["curve"],
        "passed": bool(
            finite
            and gradient_ok
            and init_activity["passed"]
            and post_activity["passed"]
            and steps <= 64
        ),
    }
    jb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "smoke.json", payload)
    print(
        f"[smoke] steps={steps} finite={finite} gradient_ok={gradient_ok} "
        f"branch_norm={post_activity['branch_response_norm']:.4f} best={result['best_valid_mae']:.6f}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError("smoke failed (non-finite loss, dead gradients or inert branch)")
    return payload


# ---------------------------------------------------------------------------
# stage: train (exactly one seed-0 trajectory)
# ---------------------------------------------------------------------------


def _usage_from_coord(coord: torch.Tensor, common_dim: int) -> dict[str, Any]:
    alpha = coord[:, common_dim:].detach().double()
    frequencies = (alpha != 0).double().mean(dim=0)
    active = int((frequencies > 0).sum())
    probability = frequencies / max(float(frequencies.sum()), 1e-12)
    nonzero = probability[probability > 0]
    entropy = float(-(nonzero * nonzero.log()).sum()) if int(nonzero.numel()) else 0.0
    sorted_share = torch.sort(frequencies, descending=True).values
    return {
        "active_atoms": active,
        "effective_atoms": float(math.exp(entropy)),
        "max_activation_rate": float(frequencies.max()),
        "top5_share": float(sorted_share[:5].sum()),
        "frequencies": [float(value) for value in frequencies],
    }


def model_diag_row(
    model: jb.JointBondModel, batch: Any, initial_dictionary: torch.Tensor
) -> dict[str, Any]:
    grads = _gradient_diagnostics(model, batch, cm.C6_MASK, cm.H1_LAMBDA)
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        usage = _usage_from_coord(coord, model.common_dim)
        movement = float((model.D.detach().cpu() - initial_dictionary).norm())
        scale = jb.response_scale_stats(model, coord, batch, cm.C6_MASK)
    return {
        **grads,
        "dictionary": usage,
        "dictionary_movement_frobenius": movement,
        "response_scale": scale,
    }


def _dictionary_health(
    model: jb.JointBondModel,
    dictionary: np.ndarray,
    valid_data: Sequence[Any],
    *,
    gradient_batch: Any | None = None,
) -> dict[str, Any]:
    device = audit.attach_cpu(THREADS)
    loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    health = cm.dictionary_diagnostics(
        semrun._ResidualDictionaryView(model), loader, device, dictionary, gradient_batch=gradient_batch
    )
    health.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
        }
    )
    return health


def _write_train_artifacts(result: Mapping[str, Any], dictionary: np.ndarray, valid_data: Sequence[Any]) -> None:
    _write_csv(
        RESULTS_DIR / "curve_seed0.csv",
        result["curve"],
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    model = _soup_model()
    _write_json(
        RESULTS_DIR / "mechanism/dictionary_health.json",
        _dictionary_health(model, dictionary, valid_data),
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
        raise RuntimeError("correctness gates not passed; refusing to train")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not smoke.get("passed"):
        raise RuntimeError("smoke not passed; refusing to train")
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    if preflight["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("preregistration changed after preflight; refusing to train")
    jb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    initial_dictionary = torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    diagnostics: dict[str, Any] = {}
    started = time.perf_counter()

    def callback(epoch: int, model: jb.JointBondModel, optimizer: Any, curve: list[dict[str, Any]]):
        if epoch in DIAG_EPOCHS:
            row = model_diag_row(model, gradient_batch, initial_dictionary)
            row["epoch"] = int(epoch)
            row["valid_mae"] = float(curve[epoch - 1]["valid_mae"])
            row["train_mae"] = float(curve[epoch - 1]["train_mae"])
            diagnostics[str(epoch)] = row
            print(
                f"[diag@{epoch}] valid={row['valid_mae']:.6f} Dgrad={row['grad_D']:.3e} "
                f"jointA={row['grad_joint_A.weight']:.2e} active={row['dictionary']['active_atoms']}",
                flush=True,
            )
        return True, None

    result = cssd.train_cssd(
        tag="JOINTBOND-seed0",
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
    jb.official_test_blocker(result)
    _write_json(final_path, result)
    _write_train_artifacts(result, dictionary, valid_data)
    print(
        f"[train] epochs={result['epochs_run']} best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# stage: frozen mechanism probes
# ---------------------------------------------------------------------------


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = MECHANISM_DIR / "branch_off.json"
    if out.exists() and not force:
        print("[interventions] cache hit", flush=True)
        return {"cached": True}
    jb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    dictionary = _dictionary_tensor()
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    if int(run.get("epochs_run", 0)) < TRAIN_EPOCHS:
        raise RuntimeError("formal run incomplete; refusing to run the frozen probes")
    model = _soup_model()
    model.eval()
    valid_data = p1run.load_split("valid")
    base_mask = cm.C6_MASK

    baseline = _evaluate(model, valid_data, base_mask)

    # 1. the new branch off (full parent path retained)
    m_branch_off = _evaluate(
        model, valid_data, jb.merge_joint_mask(base_mask, jb.JointBondMask(joint_branch_off=True))
    )
    _write_json(
        MECHANISM_DIR / "branch_off.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "M_S": baseline,
            "M_branch_off": float(m_branch_off),
            "G_branch_off": float(m_branch_off - baseline),
            "note": "only j_uv is removed; every parent path is unchanged",
        },
    )

    # 2/3. the branch-local structure-code and atom-semantics shuffles
    for kind, field, seed_field, filename in (
        ("joint_alpha_shuffle", "use_joint_alpha_shuffle", "joint_alpha_shuffle_seed", "joint_alpha_shuffle.json"),
        ("joint_atom_shuffle", "use_joint_atom_shuffle", "joint_atom_shuffle_seed", "joint_atom_shuffle.json"),
    ):
        rows: list[dict[str, Any]] = []
        for shuffle_seed in SHUFFLE_SEEDS:
            probe = jb.JointBondMask(**{field: True, seed_field: int(shuffle_seed)})
            mae = _evaluate(model, valid_data, jb.merge_joint_mask(base_mask, probe))
            rows.append(
                {"seed": int(shuffle_seed), "mae": float(mae), "delta_vs_M_S": float(mae - baseline)}
            )
        deltas = [row["delta_vs_M_S"] for row in rows]
        _write_json(
            MECHANISM_DIR / filename,
            {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": _git_commit(),
                "official_test_loaded": False,
                "kind": kind,
                "seeds": list(SHUFFLE_SEEDS),
                "M_S": baseline,
                "rows": rows,
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "note": (
                    "node rows are permuted within each molecule for the new branch only; "
                    "the frozen parent paths are untouched"
                ),
            },
        )

    # 4. branch-local residual code zero (diagnostic; the parent's dict-zero follows)
    m_joint_alpha0 = _evaluate(
        model, valid_data, jb.merge_joint_mask(base_mask, jb.JointBondMask(joint_alpha_zero=True))
    )
    _write_json(
        MECHANISM_DIR / "joint_alpha_zero.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_S": baseline,
            "M_joint_alpha0": float(m_joint_alpha0),
            "G_joint_alpha0": float(m_joint_alpha0 - baseline),
            "diagnostic_only": True,
            "note": "residual code zeroed inside the new branch only (common coordinate never read)",
        },
    )

    # 5. frozen parent diagnostics: residual dictionary zero, node/edge shuffles
    m_dict0 = _evaluate(model, valid_data, sem.merge_sem_mask(base_mask, sem.SEMMask(residual_zero=True)))
    _write_json(
        MECHANISM_DIR / "residual_dict_zero.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_S": baseline,
            "M_dict0": float(m_dict0),
            "G_dict0": float(m_dict0 - baseline),
            "diagnostic_only": True,
            "note": "frozen parent residual_zero (alpha -> 0 globally); the new branch reads the same zeroed alpha",
        },
    )
    for kind, filename in (
        ("node", "node_assignment_shuffle.json"),
        ("edge", "edge_assignment_shuffle.json"),
    ):
        rows = []
        probe_mask = sem.merge_sem_mask(
            base_mask,
            sem.SEMMask(use_node_shuffle=(kind == "node"), use_edge_shuffle=(kind == "edge")),
        )
        for shuffle_seed in PARENT_SHUFFLE_SEEDS:
            audit.prepare_shuffles(valid_data, kind, int(shuffle_seed))
            try:
                mae = _evaluate(model, valid_data, probe_mask)
            finally:
                audit.clear_shuffles(valid_data)
            rows.append(
                {"seed": int(shuffle_seed), "mae": float(mae), "delta_vs_M_S": float(mae - baseline)}
            )
        deltas = [row["delta_vs_M_S"] for row in rows]
        _write_json(
            MECHANISM_DIR / filename,
            {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": _git_commit(),
                "official_test_loaded": False,
                "kind": kind,
                "seeds": list(PARENT_SHUFFLE_SEEDS),
                "M_S": baseline,
                "rows": rows,
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "note": (
                    "frozen parent assignment shuffle; the new branch reads both the structural "
                    "code and the atom semantics at the shuffled (still real) endpoint"
                    if kind == "edge"
                    else "frozen parent node assignment shuffle (structure codes only; the new "
                    "branch is an edge path and is unaffected)"
                ),
            },
        )

    # 6. frozen parent Sem108 row shuffle
    rows = []
    for shuffle_seed in PARENT_SHUFFLE_SEEDS:
        probe_mask = sem.merge_sem_mask(
            base_mask, sem.SEMMask(use_sem_row_shuffle=True, sem_shuffle_seed=int(shuffle_seed))
        )
        mae = _evaluate(model, valid_data, probe_mask)
        rows.append({"seed": int(shuffle_seed), "mae": float(mae), "delta_vs_M_S": float(mae - baseline)})
    deltas = [row["delta_vs_M_S"] for row in rows]
    _write_json(
        MECHANISM_DIR / "sem_root_shuffle.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "kind": "sem_root_row_shuffle",
            "seeds": list(PARENT_SHUFFLE_SEEDS),
            "M_S": baseline,
            "rows": rows,
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
        },
    )

    # 7. branch health / response scale on the frozen soup state
    gradient_batch = next(iter(p1.make_env_loader(valid_data, int(p2run.BATCH_SIZE), False, 0)))
    model.eval()
    with torch.no_grad():
        coord = model.code(gradient_batch.dict_phi)
        scale_on = jb.response_scale_stats(model, coord, gradient_batch, base_mask)
        scale_off = jb.response_scale_stats(
            model,
            coord,
            gradient_batch,
            jb.merge_joint_mask(base_mask, jb.JointBondMask(joint_branch_off=True)),
        )
    health = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "parameter_norms": jb.parameter_norms(model),
        "response_scale_branch_on": scale_on,
        "response_scale_branch_off": scale_off,
    }
    _write_json(MECHANISM_DIR / "branch_health.json", health)
    _write_json(
        MECHANISM_DIR / "dictionary_health.json",
        _dictionary_health(model, dictionary, valid_data, gradient_batch=gradient_batch),
    )
    print(
        f"[interventions] M_S={baseline:.6f} M_off={m_branch_off:.6f} G_off={m_branch_off - baseline:+.6f} "
        f"G_dict0={m_dict0 - baseline:+.6f}",
        flush=True,
    )
    return {"M_S": float(baseline), "G_branch_off": float(m_branch_off - baseline)}


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.{digits}f}"


def _binding_health(model: jb.JointBondModel) -> dict[str, Any]:
    """Inference-only parameter-level health of the trained soup state."""
    stats: dict[str, Any] = {}
    for name in ("D", "W_A_S", "W_A_C", "W_E_S", "W_E_C"):
        parameter = getattr(model, name).detach()
        stats[name] = {
            "numel": int(parameter.numel()),
            "absmax": float(parameter.abs().max()),
            "norm": float(parameter.norm()),
        }
    for name in ("fusion.0.weight", "fusion.2.weight", "node_encoder.0.weight", "edge_encoder.0.weight"):
        parameter = model.get_parameter(name).detach()
        stats[name] = {
            "numel": int(parameter.numel()),
            "absmax": float(parameter.abs().max()),
            "norm": float(parameter.norm()),
        }
    stats["node_dictionary_binding_dead"] = bool(
        stats["W_A_S"]["absmax"] < 1.0e-30 or stats["W_A_C"]["absmax"] < 1.0e-30
    )
    stats["edge_dictionary_binding_dead"] = bool(
        stats["W_E_S"]["absmax"] < 1.0e-30 or stats["W_E_C"]["absmax"] < 1.0e-30
    )
    stats.update(jb.parameter_norms(model))
    stats["protocol_version"] = PROTOCOL_VERSION
    stats["git_commit"] = _git_commit()
    stats["official_test_loaded"] = False
    stats["note"] = "parameter-level diagnostic of the frozen soup state (no training, no intervention)"
    return stats


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    references = _read_json(RESULTS_DIR / "historical_references.json")
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    audit_decision = _read_json(AUDIT_DIR / "audit_decision.json")
    branch_off = _read_json(MECHANISM_DIR / "branch_off.json")
    joint_alpha_shuffle = _read_json(MECHANISM_DIR / "joint_alpha_shuffle.json")
    joint_atom_shuffle = _read_json(MECHANISM_DIR / "joint_atom_shuffle.json")
    joint_alpha_zero = _read_json(MECHANISM_DIR / "joint_alpha_zero.json")
    dict_zero = _read_json(MECHANISM_DIR / "residual_dict_zero.json")
    sem_shuffle = _read_json(MECHANISM_DIR / "sem_root_shuffle.json")
    node_shuffle = _read_json(MECHANISM_DIR / "node_assignment_shuffle.json")
    edge_shuffle = _read_json(MECHANISM_DIR / "edge_assignment_shuffle.json")
    branch_health = _read_json(MECHANISM_DIR / "branch_health.json")
    health = _read_json(MECHANISM_DIR / "dictionary_health.json")

    model = _soup_model()
    model.eval()
    binding = _binding_health(model)
    jb.official_test_blocker(binding)
    _write_json(MECHANISM_DIR / "binding_health.json", binding)

    m_s = float(run["soup"]["soup_valid_mae"])
    band = _band(m_s)
    g_off = float(branch_off["G_branch_off"])
    g_alpha_shuffle = float(joint_alpha_shuffle["mean_delta"])
    g_atom_shuffle = float(joint_atom_shuffle["mean_delta"])
    g_joint_alpha0 = float(joint_alpha_zero["G_joint_alpha0"])
    g_dict0 = float(dict_zero["G_dict0"])
    g_sem_shuffle = float(sem_shuffle["mean_delta"])
    g_node = float(node_shuffle["mean_delta"])
    g_edge = float(edge_shuffle["mean_delta"])
    scale_on = branch_health["response_scale_branch_on"]
    param_health = branch_health["parameter_norms"]
    branch_dead = bool(
        param_health["joint_branch_dead"]
        or float(scale_on["joint_branch_zero_fraction"]) >= 1.0
        or abs(g_off) < BRANCH_INERT_TOL
    )
    health_pass = bool(
        int(health["active_atoms"]) >= 24
        and float(health["effective_atoms"]) >= 8
        and float(health["usage_top1_share"]) <= 0.75
    )
    branch_used = bool(g_off >= BRANCH_LOAD_GATE)
    endpoint_read = str(
        "read"
        if (g_alpha_shuffle > 0.0 and g_atom_shuffle > 0.0)
        else ("partial" if (g_alpha_shuffle > 0.0 or g_atom_shuffle > 0.0) else "not_established")
    )
    buy_matched_control = bool(band != BANDS[-1][1] and branch_used)
    mechanism_note = (
        "Node-side dictionary correspondence is exactly zero and the frozen soup state confirms why: "
        f"W_A_S / W_A_C collapsed to float32 denormals (absmax "
        f"{binding['W_A_S']['absmax']:.3e} / {binding['W_A_C']['absmax']:.3e}), so node slots are "
        "identically zero and the node assignment shuffle is a no-op. This is the same learned "
        "redundancy collapse the Sem108 round recorded; it was NOT repaired this round."
        if binding["node_dictionary_binding_dead"]
        else "Node and edge dictionary correspondences are both measured on the frozen soup state."
    )
    sem_use = (
        "strong"
        if g_sem_shuffle >= 0.010
        else ("load_bearing" if g_sem_shuffle >= 0.005 else ("weak" if g_sem_shuffle < 0.003 else "marginal"))
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
            "parent": int(preflight["parameter_audit"]["parent_params"]),
            "candidate": int(preflight["parameter_audit"]["candidate_params"]),
            "new": int(preflight["parameter_audit"]["new_params"]),
            "delta": int(preflight["parameter_audit"]["delta"]),
            "relative_delta": float(preflight["parameter_audit"]["relative_delta"]),
            "matched": False,
        },
        "audit": {
            "decision": audit_decision["decision"],
            "parent_identity": bool(audit_decision["checks"]["identity_established"]),
            "parent_a2": bool(audit_decision["checks"]["a2_pass"]),
        },
        "M_S": m_s,
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": list(run["soup"]["members"]),
        "performance_band": band,
        "historical": {
            "SEM108_parent_soup": float(references["references"]["SEM108-parent-seed0"]["value"]),
            "G_vs_parent_unmatched": float(
                references["references"]["SEM108-parent-seed0"]["value"] - m_s
            ),
            "CSSD_q1_soup": references["references"]["CSSD-q1-seed0"]["value"],
            "RNDB_soup": references["references"]["RNDB-seed0"]["value"],
            "note": (
                "no contemporaneous baseline; the parent's 0.123704927947314 is a historical "
                "unmatched reference and the +4224 parameters are un-matched"
            ),
        },
        "gates": {
            "correctness_all_passed": bool(correctness["all_passed"]),
            "smoke_passed": bool(smoke["passed"]),
            "parameter_contract_exact": bool(
                preflight["parameter_audit"]["new_params_exact"]
                and preflight["parameter_audit"]["total_exact"]
            ),
            "dictionary_health_pass": health_pass,
            "branch_dead": branch_dead,
            "branch_used": branch_used,
        },
        "branch_mechanism": {
            "M_branch_off": float(branch_off["M_branch_off"]),
            "G_branch_off": g_off,
            "M_joint_alpha0": float(joint_alpha_zero["M_joint_alpha0"]),
            "G_joint_alpha0": g_joint_alpha0,
            "G_joint_alpha_shuffle": g_alpha_shuffle,
            "G_joint_alpha_shuffle_rows": joint_alpha_shuffle["rows"],
            "G_joint_atom_shuffle": g_atom_shuffle,
            "G_joint_atom_shuffle_rows": joint_atom_shuffle["rows"],
            "endpoint_correspondence": endpoint_read,
            "branch_load_gate": BRANCH_LOAD_GATE,
        },
        "parent_mechanism": {
            "M_sem0": None,
            "G_sem_shuffle": g_sem_shuffle,
            "sem_use": sem_use,
            "G_dict0": g_dict0,
            "G_node": g_node,
            "G_edge": g_edge,
            "node_dictionary_binding_dead": bool(binding["node_dictionary_binding_dead"]),
            "sem_shuffle_rows": sem_shuffle["rows"],
            "node_rows": node_shuffle["rows"],
            "edge_rows": edge_shuffle["rows"],
        },
        "branch_health": branch_health,
        "binding_health": binding,
        "dictionary_health": health,
        "decision": {
            "buy_matched_control": buy_matched_control,
            "rule": "band in {further_confirmation, candidate_signal} AND G_branch_off >= 0.003",
            "performance_interval": band,
            "branch_used": branch_used,
            "endpoint_correspondence_read": endpoint_read,
            "note": (
                "the branch-off / shuffle drops are interventions on one frozen soup state, not a "
                "retrained matched control"
            ),
        },
        "mechanism_note": mechanism_note,
        "historical_unmatched_only": True,
        "matched_baseline_rerun": False,
        "mechanism_probes_inference_only": True,
    }
    jb.official_test_blocker(summary)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, references, run, health)
    _write_decision(summary)
    print(
        f"[analysis] M_S={m_s:.6f} band={band} G_off={g_off:+.6f} "
        f"branch_used={branch_used} buy={buy_matched_control}",
        flush=True,
    )
    return summary


def _write_report(
    summary: Mapping[str, Any],
    references: Mapping[str, Any],
    run: Mapping[str, Any],
    health: Mapping[str, Any],
) -> None:
    s = summary
    lines: list[str] = []
    lines.append("# e2e_dictenv_jointbond_v1 — JointBond (Key-level Joint Structure-Semantics Fusion)")
    lines.append("")
    lines.append("CPU only · official test never loaded · single seed-0 trajectory · parent `CSSD-Sem108`.")
    lines.append("")
    lines.append("## A. Provenance")
    lines.append("")
    lines.append(f"- formal-run commit `{run.get('git_commit', 'n/a')}`")
    lines.append(f"- analysis / report commit `{s['git_commit']}`")
    lines.append(f"- parent `{jb.PARENT_MODEL}` @ `{jb.PARENT_COMMIT}`")
    lines.append(f"- device `cpu` · threads `{s['threads']}` · `official_test_loaded = false`")
    lines.append("")
    lines.append("## B. Historical references (read-only; never rerun; unmatched)")
    lines.append("")
    for name, ref in references["references"].items():
        value = ref.get("value") if isinstance(ref, Mapping) else ref
        lines.append(f"- {name}: " + ("n/a" if value is None else f"`{_fmt(value, 6)}`"))
    lines.append("")
    lines.append("## C. Exactly one architectural change")
    lines.append("")
    lines.append("```")
    lines.append("alpha_v = coord[v, common_dim:]            # 32-D residual code (common excluded)")
    lines.append("q_v     = one_hot(dict_atom[v])            # 28 atom categories")
    lines.append("b_uv    = one_hot(env_bond_type[uv])       # 4 bond categories")
    lines.append("h_v     = (alpha_v @ A) * (q_v @ C)        # 16-D")
    lines.append("t_uv    = [h_u + h_v ; |h_u - h_v| ; h_u * h_v]   # 48-D")
    lines.append("j_uv    = F(t_uv) * (b_uv @ B)             # 48-D, added BEFORE the shellpair index_add_")
    lines.append("```")
    lines.append("")
    lines.append("- `A: 32->16`, `C: 28->16`, `B: 4->48` bias-free; `F = 48->32->48`, SiLU, bias-free")
    lines.append("- parent paths, C6 mask, dictionary, IHT-10, fusion, backend, reader, lambda, soup: frozen")
    lines.append("")
    lines.append("## D. Parameter budget")
    lines.append("")
    p = s["parameters"]
    lines.append(
        f"- parent `{p['parent']}`, candidate `{p['candidate']}`, new `{p['new']}` "
        f"(relative `{p['relative_delta']:.5f}`) · **un-matched by design**"
    )
    lines.append("")
    lines.append("## E. Training")
    lines.append("")
    lines.append(f"- epochs `{s['epochs']}` · wall `{_fmt(s['wall_clock_s'], 1)} s` · seed `{s['seed']}`")
    lines.append(
        f"- best valid `{_fmt(s['best_valid_mae'], 6)}` @ {s['best_epoch']} · Top-5 members "
        f"{s['soup_members']} · soup `{_fmt(s['M_S'], 6)}`"
    )
    lines.append("")
    lines.append("## F. Performance interpretation")
    lines.append("")
    lines.append(f"- `M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**")
    lines.append(
        "- pre-registered intervals: `<= 0.120` further confirmation · `0.120–0.123` candidate "
        "signal · `> 0.123` no clear breakthrough"
    )
    lines.append(
        f"- historical parent reference `{_fmt(s['historical']['SEM108_parent_soup'], 6)}` "
        f"(`G = {_fmt(s['historical']['G_vs_parent_unmatched'], 6)}`, unmatched, not a baseline)"
    )
    lines.append("")
    lines.append("## G. Joint branch mechanism (frozen soup, inference only)")
    lines.append("")
    bm = s["branch_mechanism"]
    lines.append(
        f"- branch off: `M_branch_off={_fmt(bm['M_branch_off'], 6)}` "
        f"`G_branch_off={_fmt(bm['G_branch_off'], 6)}` (gate `{bm['branch_load_gate']}`)"
    )
    lines.append(
        f"- branch-local residual-code zero: `G_joint_alpha0={_fmt(bm['G_joint_alpha0'], 6)}`"
    )
    lines.append(
        f"- residual-code row shuffle: `G={_fmt(bm['G_joint_alpha_shuffle'], 6)}` "
        f"- atom-semantics row shuffle: `G={_fmt(bm['G_joint_atom_shuffle'], 6)}` "
        f"→ endpoint correspondence `{bm['endpoint_correspondence']}`"
    )
    lines.append("")
    lines.append("## H. Parent mechanism (unchanged probes)")
    lines.append("")
    pm = s["parent_mechanism"]
    lines.append(
        f"- Sem108 row shuffle `G_sem_shuffle={_fmt(pm['G_sem_shuffle'], 6)}` ({pm['sem_use']}) · "
        f"residual dict zero `G_dict0={_fmt(pm['G_dict0'], 6)}`"
    )
    lines.append(
        f"- parent node shuffle `G_node={_fmt(pm['G_node'], 6)}` · edge `G_edge={_fmt(pm['G_edge'], 6)}`"
    )
    lines.append(
        f"- dictionary health: active {health['active_atoms']}/32, effective {_fmt(health['effective_atoms'], 2)}, "
        f"top1 {_fmt(health['usage_top1_share'], 3)}, recon {_fmt(health['valid_reconstruction_relative'], 5)}"
    )
    lines.append("")
    lines.append("## I. Response scale (branch vs parent edge path)")
    lines.append("")
    scale = s["branch_health"]["response_scale_branch_on"]
    lines.append(
        f"- parent edge response norm (mean/median/p90) "
        f"`{_fmt(scale['parent_edge_response']['mean'], 4)}` / "
        f"`{_fmt(scale['parent_edge_response']['median'], 4)}` / "
        f"`{_fmt(scale['parent_edge_response']['p90'], 4)}`"
    )
    lines.append(
        f"- joint branch response norm (mean/median/p90) "
        f"`{_fmt(scale['joint_branch_response']['mean'], 4)}` / "
        f"`{_fmt(scale['joint_branch_response']['median'], 4)}` / "
        f"`{_fmt(scale['joint_branch_response']['p90'], 4)}`"
    )
    lines.append(
        f"- branch/parent norm ratio mean `{_fmt(scale['branch_to_parent_norm_ratio']['mean'], 4)}` · "
        f"branch-dead `{s['gates']['branch_dead']}`"
    )
    lines.append(
        f"- parameter norms: " + ", ".join(
            f"`{name}={_fmt(value['norm'], 4)}`" for name, value in s["branch_health"]["parameter_norms"].items()
            if isinstance(value, Mapping)
        )
    )
    lines.append("")
    lines.append("## J. Verdict")
    lines.append("")
    lines.append(
        f"**band {s['performance_band']} · branch_used={s['gates']['branch_used']} · "
        f"endpoint_correspondence={bm['endpoint_correspondence']} · "
        f"buy_matched_control={s['decision']['buy_matched_control']}**"
    )
    lines.append("")
    lines.append(f"> {s['mechanism_note']}")
    lines.append("")
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    s = summary
    lines = [
        "# DECISION — e2e_dictenv_jointbond_v1",
        "",
        f"`M_S = {_fmt(s['M_S'], 6)}` → interval **{s['performance_band']}**.",
        "",
        f"`G_branch_off = {_fmt(s['branch_mechanism']['G_branch_off'], 6)}` "
        f"(gate {s['branch_mechanism']['branch_load_gate']}), "
        f"`G_joint_alpha_shuffle = {_fmt(s['branch_mechanism']['G_joint_alpha_shuffle'], 6)}`, "
        f"`G_joint_atom_shuffle = {_fmt(s['branch_mechanism']['G_joint_atom_shuffle'], 6)}`.",
        "",
        f"**buy_matched_control = {s['decision']['buy_matched_control']}**",
        "",
        "Single seed-0 trajectory, no contemporaneous baseline, no official-test read.",
        "The +4224 branch parameters are un-matched; branch-off / shuffle drops are frozen-soup "
        "interventions, not a retrained control.",
        "",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


STAGES = ("references", "audit", "preflight", "correctness", "smoke", "train", "interventions", "analysis", "chain")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    jb.cpu_only_guard(torch.device("cpu"))

    if args.stage in ("references", "chain"):
        stage_references()
    if args.stage in ("audit", "chain"):
        stage_audit(force=args.force)
    if args.stage in ("preflight", "chain"):
        stage_preflight()
    if args.stage in ("correctness", "chain"):
        stage_correctness()
    if args.stage in ("smoke", "chain"):
        stage_smoke()
    if args.stage in ("train", "chain"):
        stage_train(force=args.force)
    if args.stage in ("interventions", "chain"):
        stage_interventions(force=args.force)
    if args.stage in ("analysis", "chain"):
        stage_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
