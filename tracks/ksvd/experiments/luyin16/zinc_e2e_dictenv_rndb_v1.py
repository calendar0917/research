"""E2E-DictEnv-RNDB-v1 runner — Rolewise Nonlinear Dictionary Binding.

Round ``e2e_dictenv_rndb_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_rndb_v1.py``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_rndb_v1_preregistration.md``.

Stages
------
``references  preflight  correctness  smoke  train  interventions  analysis  chain``

Exactly one from-scratch RNDB seed-0 trajectory (320 epochs) on CPU.  The
official ZINC test split is never loaded; every payload records
``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rndb_v1 as rndb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = rndb.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_rndb_v1"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_rndb_v1_preregistration.md"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

THREADS = 8
TRAIN_EPOCHS = 320
SEED = 0
DIAG_EPOCHS: tuple[int, ...] = (1, 20, 40, 80, 160, 240, 320)
SMOKE_EPOCHS = 8
SMOKE_SUBSET = 2048
SHUFFLE_SEEDS = audit.SHUFFLE_SEEDS  # (101, 202, 303, 404, 505)

#: pre-registered absolute-MAE performance bands.
BANDS: tuple[tuple[float, str], ...] = (
    (0.120, "RNDB_SINGLE_SEED_STRONG_SIGNAL"),
    (0.123, "RNDB_SINGLE_SEED_PROMISING"),
    (0.126, "RNDB_TASK_NEUTRAL_WITHIN_HISTORICAL_BAND"),
    (0.135, "RNDB_NO_USEFUL_TASK_GAIN"),
    (float("inf"), "RNDB_TASK_REGRESSION"),
)


def _band(mae: float) -> str:
    for threshold, label in BANDS:
        if float(mae) <= threshold:
            return label
    return BANDS[-1][1]


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _subspace_from_payload(payload: Mapping[str, Any], kind: str = "q1") -> cssd.CommonSubspace:
    entry = payload[kind]
    return cssd.CommonSubspace(
        components=np.asarray(entry["components"], dtype=np.float64),
        rms=np.asarray(entry["rms"], dtype=np.float64),
        kind=str(entry["kind"]),
    )


def _load_parent_subspace() -> cssd.CommonSubspace:
    path = (
        TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
    )
    return _subspace_from_payload(_read_json(path), "q1")


def _model_factory(dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace) -> rndb.RNDBModel:
    return rndb.build_rndb_model(dictionary, int(seed), subspace)


def _dictionary_tensor() -> np.ndarray:
    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


# ---------------------------------------------------------------------------
# stage: references
# ---------------------------------------------------------------------------


def stage_references() -> dict[str, Any]:
    _ensure_dirs()

    cssd_path = "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json"
    cssd = _read_json(REPO_ROOT / cssd_path)
    clean_path = "tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json"
    clean = _read_json(REPO_ROOT / clean_path)
    t1_path = "tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json"
    t1 = _read_json(REPO_ROOT / t1_path)
    selection = _read_json(
        REPO_ROOT
        / "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/zero_training/selection.json"
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "policy": (
            "read-only durable artifacts; no historical model is retrained; comparisons are "
            "unmatched descriptive context only"
        ),
        "prompt_approximations_note": (
            "The prompt's approximate values (CSSD-q1 ~= 0.1244, CSSD-q2 ~= 0.1236) do not match "
            "the local artifacts; the exact local artifacts below are the source of truth."
        ),
        "references": {
            "CSSD-Q1-seed0": {
                "value": float(cssd["soup"]["soup_valid_mae"]),
                "best_valid_mae": float(cssd["best_valid_mae"]),
                "best_epoch": int(cssd["best_epoch"]),
                "actual_params": int(cssd["actual_params"]),
                "artifact": cssd_path,
                "keys": ["soup", "soup_valid_mae"],
                "status": "not_rerun_unmatched_context",
            },
            "CSSD-Q2": {
                "value": None,
                "artifact": (
                    "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/zero_training/selection.json"
                ),
                "q1_passed_seeds": selection["q1_passed_seeds"],
                "q2_passed_seeds": selection["q2_passed_seeds"],
                "selected": selection["selected"],
                "status": "never_trained_context_only",
            },
            "FINAL-CLEAN-seed0": {
                "value": float(clean["baseline_valid_mae"]),
                "artifact": clean_path,
                "keys": ["baseline_valid_mae"],
                "status": "not_rerun_unmatched_context",
            },
            "T1-tuned-seed0": {
                "value": float(t1["best_tuned_soup_valid_mae"]),
                "artifact": t1_path,
                "keys": ["best_tuned_soup_valid_mae"],
                "status": "not_rerun_unmatched_context",
            },
        },
    }
    rndb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "historical_references.json", payload)
    print(
        "[references] CSSD-Q1={:.6f} FINAL-CLEAN={:.6f} T1={:.6f} (CSSD-Q2 not trained)".format(
            payload["references"]["CSSD-Q1-seed0"]["value"],
            payload["references"]["FINAL-CLEAN-seed0"]["value"],
            payload["references"]["T1-tuned-seed0"]["value"],
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    rndb.cpu_only_guard(torch.device("cpu"))
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    import hashlib

    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    model = rndb.build_rndb_model(dictionary, SEED, subspace)
    parent = cssd.build_cssd_model(dictionary, SEED, subspace)
    parent_params = int(sum(p.numel() for p in parent.parameters()))
    audit_payload = rndb.parameter_audit(model, parent_params)
    if not audit_payload["new_params_ok"] or not audit_payload["ratio_ok"]:
        raise RuntimeError(f"parameter budget violated: {audit_payload}")
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
        "parent": {
            "config": cm.H1_CONFIG.as_dict(),
            "lambda_rec": float(cm.H1_LAMBDA),
            "c6_mask_equals_parent": bool(rndb.RNDB_MASK is cm.C6_MASK),
            "c6_equivalence_check": bool(cm.c6_equivalence_check()),
            "subspace_kind": subspace.kind,
            "common_dim": int(subspace.q),
            "common_rms": [float(value) for value in subspace.rms],
            "d_struct": int(model.n_common + model.n_dict),
            "n_common": int(model.n_common),
            "n_dict": int(model.n_dict),
            "dictionary_shape": list(np.asarray(dictionary).shape),
        },
        "parameter_audit": audit_payload,
        "epochs": TRAIN_EPOCHS,
        "seed": SEED,
    }
    rndb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration"])
    _write_json(RESULTS_DIR / "parameter_audit.json", audit_payload)
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {prereg_sha[:12]} params {audit_payload['rndb_params']} "
        f"(+{audit_payload['new_params']}) d_struct={payload['parent']['d_struct']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: correctness gates G0-G6
# ---------------------------------------------------------------------------


def _synthetic_batch(n_nodes: int = 40, n_occ: int = 120, n_bond: int = 90, seed: int = 0) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(int(seed))
    data = SimpleNamespace()
    data.dict_phi = torch.randn(n_nodes, cssd.PHI_DIM, generator=generator)
    data.dict_atom = torch.randint(0, int(p2.ATOM_CATEGORIES), (n_nodes,), generator=generator)
    data.anchor = torch.randn(n_nodes, audit.ANCHOR_DIM_EXPECTED, generator=generator)
    data.env_occ_node = torch.randint(0, n_nodes, (n_occ,), generator=generator)
    data.env_occ_root = torch.randint(0, n_nodes, (n_occ,), generator=generator)
    data.env_occ_shell = torch.randint(0, int(p2.N_SHELLS), (n_occ,), generator=generator)
    data.env_bond_u = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_v = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_type = torch.randint(0, int(p2.BOND_CATEGORIES), (n_bond,), generator=generator)
    data.env_bond_root = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_shellpair = torch.randint(0, int(p2.SHELLPAIR_CLASSES), (n_bond,), generator=generator)
    data.batch = torch.zeros(n_nodes, dtype=torch.long)
    data.num_graphs = 1
    data.y = torch.zeros(1)
    data.global_context = torch.randn(1, p2.GLOBAL_WIDTH, generator=generator)
    data.topology_features = torch.randn(1, p2.TOPOLOGY_IN, generator=generator)
    data.pair_relation = torch.randn(5, int(p2.RELATION_WIDTH), generator=generator)
    data.pair_bucket = torch.randint(0, int(p2.DISTANCE_BUCKETS), (5,), generator=generator)
    data.pair_index = torch.stack([torch.randint(0, n_nodes, (5,), generator=generator) for _ in range(2)])
    return data


def stage_correctness() -> dict[str, Any]:
    _ensure_dirs()
    rndb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    model = rndb.build_rndb_model(dictionary, SEED, subspace)
    parent = cssd.build_cssd_model(dictionary, SEED, subspace)
    mask = cm.C6_MASK
    gates: dict[str, Any] = {}

    # -- G0a synthetic decomposition identity ---------------------------------
    syn = _synthetic_batch(seed=11)
    coord_syn = model.code(syn.dict_phi)
    p_all_n, parent_n = rndb.node_role_contributions(model, coord_syn, syn.env_occ_node, syn.dict_atom)
    p_all_e, parent_e = rndb.edge_role_contributions(
        model, coord_syn, syn.env_bond_u, syn.env_bond_v, syn.env_bond_type
    )
    gates["G0_synthetic"] = {
        "node_max_abs_diff": rndb.max_abs_diff(p_all_n.sum(dim=1), parent_n),
        "edge_max_abs_diff": rndb.max_abs_diff(p_all_e.sum(dim=1), parent_e),
    }

    # -- G0b real ZINC batch decomposition identity ---------------------------
    train_small = p1run.load_split("train", subset=64)
    loader = p1.make_env_loader(train_small, 64, False, 0)
    real = next(iter(loader))
    coord_real = model.code(real.dict_phi)
    p_all_n, parent_n = rndb.node_role_contributions(model, coord_real, real.env_occ_node, real.dict_atom)
    p_all_e, parent_e = rndb.edge_role_contributions(
        model, coord_real, real.env_bond_u, real.env_bond_v, real.env_bond_type
    )
    gates["G0_real"] = {
        "node_max_abs_diff": rndb.max_abs_diff(p_all_n.sum(dim=1), parent_n),
        "edge_max_abs_diff": rndb.max_abs_diff(p_all_e.sum(dim=1), parent_e),
        "n_nodes": int(real.dict_phi.shape[0]),
        "n_occ": int(real.env_occ_node.shape[0]),
        "n_bond": int(real.env_bond_u.shape[0]),
    }
    g0_ok = all(
        gates[key][name] <= rndb.DECOMPOSITION_TOL
        for key in ("G0_synthetic", "G0_real")
        for name in ("node_max_abs_diff", "edge_max_abs_diff")
    )
    gates["G0"] = {"passed": bool(g0_ok), "tolerance": rndb.DECOMPOSITION_TOL}

    # -- G1 psi-off exact parent equivalence ----------------------------------
    model.eval()
    parent.eval()
    with torch.no_grad():
        model.psi_node_on = False
        model.psi_edge_on = False
        pred_rndb, aux_rndb = model(real, mask=mask, return_aux=True)
        pred_parent, aux_parent = parent(real, mask=mask, return_aux=True)
        model.psi_node_on = True
        model.psi_edge_on = True
    g1 = {
        "prediction_max_abs_diff": float((pred_rndb - pred_parent).abs().max()),
        "environment_max_abs_diff": float((aux_rndb["E"] - aux_parent["E"]).abs().max()),
        "coord_max_abs_diff": float((aux_rndb["coord"] - aux_parent["coord"]).abs().max()),
    }
    g1["passed"] = bool(max(g1["prediction_max_abs_diff"], g1["environment_max_abs_diff"]) <= rndb.PSI_OFF_TOL)
    gates["G1"] = g1

    # -- G2 inactive-role strict zero -----------------------------------------
    model.eval()
    with torch.no_grad():
        coord_zero_dict = coord_real.clone()
        coord_zero_dict[:, model.n_common :] = 0.0
        u_zero = model.rndb_node_occurrence(coord_zero_dict, real)
        ue_zero = model.rndb_edge_occurrence(coord_zero_dict, real)
        model.dict_zero = True
        u_common = model.rndb_node_occurrence(coord_real, real)
        ue_common = model.rndb_edge_occurrence(coord_real, real)
        model.dict_zero = False
        psi_zero = float(model.psi_A(torch.zeros(3, 5, int(p2.D_A))).abs().max())
        psi_zero_e = float(model.psi_E(torch.zeros(3, 5, int(model.W_E_S.shape[1]))).abs().max())
    g2 = {
        "node_zero_dict_equals_common": float((u_zero - u_common).abs().max()),
        "edge_zero_dict_equals_common": float((ue_zero - ue_common).abs().max()),
        "psi_A_of_zero_max_abs": psi_zero,
        "psi_E_of_zero_max_abs": psi_zero_e,
    }
    g2["passed"] = bool(
        g2["node_zero_dict_equals_common"] == 0.0
        and g2["edge_zero_dict_equals_common"] == 0.0
        and g2["psi_A_of_zero_max_abs"] == 0.0
        and g2["psi_E_of_zero_max_abs"] == 0.0
    )
    gates["G2"] = g2

    # -- G3 role permutation equivariance -------------------------------------
    generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(model.n_dict, generator=generator)
    n = model.n_common
    d_struct = int(coord_real.shape[1])
    with torch.no_grad():
        coord_perm = coord_real.clone()
        coord_perm[:, n:] = coord_real[:, n:][:, permutation]
        w_a = model.W_A_S.detach().clone()
        w_e = model.W_E_S.detach().clone()
        model.W_A_S[n:] = w_a[n:][permutation]
        for block in range(3):
            start = block * d_struct + n
            model.W_E_S[start : start + model.n_dict] = w_e[start : start + model.n_dict][permutation]
        u_perm = model.rndb_node_occurrence(coord_perm, real)
        ue_perm = model.rndb_edge_occurrence(coord_perm, real)
        model.W_A_S.copy_(w_a)
        model.W_E_S.copy_(w_e)
        u_orig = model.rndb_node_occurrence(coord_real, real)
        ue_orig = model.rndb_edge_occurrence(coord_real, real)
    g3 = {
        "node_permutation_max_abs_diff": float((u_perm - u_orig).abs().max()),
        "edge_permutation_max_abs_diff": float((ue_perm - ue_orig).abs().max()),
        "common_coordinate_fixed": True,
    }
    g3["passed"] = bool(
        g3["node_permutation_max_abs_diff"] <= 1e-5 and g3["edge_permutation_max_abs_diff"] <= 1e-5
    )
    gates["G3"] = g3

    # -- G4 chemistry / topology purity ---------------------------------------
    mutated = real.clone()
    mutated.topology_features = torch.randn_like(real.topology_features)
    mutated.global_context = torch.randn_like(real.global_context)
    mutated.pair_relation = torch.randn_like(real.pair_relation)
    mutated.pair_bucket = torch.randint_like(real.pair_bucket, 0, int(p2.DISTANCE_BUCKETS))
    mutated.pair_index = torch.stack(
        [torch.randint(0, int(real.dict_phi.shape[0]), (real.pair_index.shape[1],)) for _ in range(2)]
    )
    model.eval()
    with torch.no_grad():
        E_orig = model.environments_masked(coord_real, real, mask)
        E_mut = model.environments_masked(coord_real, mutated, mask)
    new_param_names = sorted(set(model.state_dict()) - set(parent.state_dict()))
    g4 = {
        "environment_invariant_to_nonbinding_inputs_max_abs": float((E_orig - E_mut).abs().max()),
        "new_parameter_names": new_param_names,
    }
    g4["passed"] = bool(
        g4["environment_invariant_to_nonbinding_inputs_max_abs"] == 0.0
        and new_param_names
        == ["psi_A.0.weight", "psi_A.2.weight", "psi_E.0.weight", "psi_E.2.weight"]
    )
    gates["G4"] = g4

    # -- G5 C6 / relation / backend frozen ------------------------------------
    g5 = {
        "c6_mask_identity": bool(rndb.RNDB_MASK is cm.C6_MASK),
        "c6_equivalence_check": bool(cm.c6_equivalence_check()),
        "relation_width": int(p2.RELATION_WIDTH),
        "n_shells": int(p2.N_SHELLS),
        "shellpair_classes": int(p2.SHELLPAIR_CLASSES),
        "reader_class": type(model.reader).__name__,
        "reader_class_parent": type(parent.reader).__name__,
        "pair_projection_reused": bool(type(model.pair_projection) is type(parent.pair_projection)),
        "global_encoder_reused": bool(type(model.global_encoder) is type(parent.global_encoder)),
        "topology_encoder_reused": bool(type(model.topology_encoder) is type(parent.topology_encoder)),
    }
    g5["passed"] = bool(
        g5["c6_mask_identity"]
        and g5["c6_equivalence_check"]
        and g5["reader_class"] == g5["reader_class_parent"]
        and g5["pair_projection_reused"]
        and g5["global_encoder_reused"]
        and g5["topology_encoder_reused"]
    )
    gates["G5"] = g5

    # -- G6 official-test blocker ---------------------------------------------
    blocked = False
    try:
        rndb.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocked = True
    gates["G6"] = {"blocker_raises_on_true": bool(blocked), "passed": bool(blocked)}

    all_passed = all(bool(entry.get("passed")) for entry in gates.values() if "passed" in entry)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "gates": gates,
        "all_passed": bool(all_passed),
        "tolerances": {"decomposition": rndb.DECOMPOSITION_TOL, "psi_off": rndb.PSI_OFF_TOL},
    }
    rndb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "correctness.json", payload)
    for key in ("G0", "G1", "G2", "G3", "G4", "G5", "G6"):
        print(f"[correctness] {key} passed={gates[key]['passed']}", flush=True)
    if not all_passed:
        raise RuntimeError(f"correctness gates failed: {gates}")
    return payload


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------


def _gradient_diagnostics(model: rndb.RNDBModel, batch: Any, mask: Any) -> dict[str, Any]:
    model.train(False)
    prediction, aux = model(batch, mask=mask, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(cm.H1_LAMBDA) * rec
    model.zero_grad(set_to_none=True)
    loss.backward()
    out = {
        "loss": float(loss.detach()),
        "reconstruction": float(rec.detach()),
        "grad_D": float(model.D.grad.detach().norm()) if model.D.grad is not None else 0.0,
        "grad_psi_A_W1": float(model.psi_A[0].weight.grad.detach().norm())
        if model.psi_A[0].weight.grad is not None
        else 0.0,
        "grad_psi_A_W2": float(model.psi_A[2].weight.grad.detach().norm())
        if model.psi_A[2].weight.grad is not None
        else 0.0,
        "grad_psi_E_W1": float(model.psi_E[0].weight.grad.detach().norm())
        if model.psi_E[0].weight.grad is not None
        else 0.0,
        "grad_psi_E_W2": float(model.psi_E[2].weight.grad.detach().norm())
        if model.psi_E[2].weight.grad is not None
        else 0.0,
    }
    model.zero_grad(set_to_none=True)
    model.train(True)
    return out


def stage_smoke() -> dict[str, Any]:
    _ensure_dirs()
    rndb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    train_data = p1run.load_split("train", subset=SMOKE_SUBSET)
    valid_data = p1run.load_split("valid", subset=512)
    model = rndb.build_rndb_model(dictionary, SEED, subspace)
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    init_grads = _gradient_diagnostics(model, gradient_batch, cm.C6_MASK)
    init_response = rndb.response_stats(model, gradient_batch)
    result = cssd.train_cssd(
        tag="RNDB-smoke",
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
    final_model = rndb.build_rndb_model(dictionary, SEED, subspace)
    final_model.load_state_dict(
        {
            key: value.float()
            for key, value in torch.load(
                CHECKPOINT_DIR / "RNDB-smoke_final_state.pt", map_location="cpu", weights_only=False
            ).items()
        }
    )
    post_grads = _gradient_diagnostics(final_model, gradient_batch, cm.C6_MASK)
    post_response = rndb.response_stats(final_model, gradient_batch)
    finite = bool(
        math.isfinite(result["best_valid_mae"])
        and math.isfinite(result["final_valid_mae"])
        and all(math.isfinite(row["train_mae"]) for row in result["curve"])
    )
    gradient_ok = bool(
        all(
            init_grads[key] > 0.0
            for key in ("grad_D", "grad_psi_A_W1", "grad_psi_A_W2", "grad_psi_E_W1", "grad_psi_E_W2")
        )
        and all(
            post_grads[key] > 0.0
            for key in ("grad_D", "grad_psi_A_W1", "grad_psi_A_W2", "grad_psi_E_W1", "grad_psi_E_W2")
        )
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "epochs": SMOKE_EPOCHS,
        "subset": SMOKE_SUBSET,
        "finite": finite,
        "gradient_ok": gradient_ok,
        "init_gradients": init_grads,
        "post_gradients": post_grads,
        "init_response": init_response,
        "post_response": post_response,
        "psi_health": rndb.psi_health(final_model),
        "best_valid_mae": float(result["best_valid_mae"]),
        "final_valid_mae": float(result["final_valid_mae"]),
        "train_curve": result["curve"],
        "passed": bool(finite and gradient_ok),
    }
    rndb.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "smoke.json", payload)
    print(
        f"[smoke] finite={finite} gradient_ok={gradient_ok} "
        f"init_grad_D={init_grads['grad_D']:.3e} best={result['best_valid_mae']:.6f}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError("smoke failed (non-finite loss or dead gradients)")
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


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = RESULTS_DIR / "run_seed0.json"
    if final_path.exists() and not force:
        print("[train] cache hit", flush=True)
        return _read_json(final_path)
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    if not correctness.get("all_passed"):
        raise RuntimeError("correctness gates not passed; refusing to train")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not smoke.get("passed"):
        raise RuntimeError("smoke not passed; refusing to train")
    rndb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    initial_dictionary = torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    diagnostics: dict[str, Any] = {}
    started = time.perf_counter()

    def _diag(epoch: int, model: rndb.RNDBModel) -> dict[str, Any]:
        row = model_diag_row(model, gradient_batch, initial_dictionary)
        row["epoch"] = int(epoch)
        row["valid_mae"] = float(curve_ref[0][epoch - 1]["valid_mae"]) if curve_ref[0] else float("nan")
        row["train_mae"] = float(curve_ref[0][epoch - 1]["train_mae"]) if curve_ref[0] else float("nan")
        return row

    curve_ref: list[list[dict[str, Any]]] = [[]]

    def callback(epoch: int, model: rndb.RNDBModel, optimizer: Any, curve: list[dict[str, Any]]):
        curve_ref[0] = curve
        if epoch in DIAG_EPOCHS:
            payload = _diag(epoch, model)
            diagnostics[str(epoch)] = payload
            print(
                f"[diag@{epoch}] valid={payload['valid_mae']:.6f} Dgrad={payload['grad_D']:.3e} "
                f"psiA=({payload['grad_psi_A_W1']:.2e},{payload['grad_psi_A_W2']:.2e}) "
                f"node_ratio={payload['response']['node']['ratio']['mean']:.4f}",
                flush=True,
            )
        return True, None

    result = cssd.train_cssd(
        tag="RNDB-seed0",
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
    rndb.official_test_blocker(result)
    _write_json(final_path, result)
    _write_csv(
        RESULTS_DIR / "curve_seed0.csv",
        result["curve"],
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    _write_json(
        RESULTS_DIR / "dictionary_health.json",
        _dictionary_health(result, subspace, dictionary, valid_data),
    )
    _write_json(
        RESULTS_DIR / "rndb_response_stats.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "diagnostics": diagnostics,
            "psi_health_final": rndb.psi_health(_soup_model(result, subspace, dictionary)),
        },
    )
    print(
        f"[train] epochs={result['epochs_run']} best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


def model_diag_row(model: rndb.RNDBModel, batch: Any, initial_dictionary: torch.Tensor) -> dict[str, Any]:
    grads = _gradient_diagnostics(model, batch, cm.C6_MASK)
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        usage = _usage_from_coord(coord, model.common_dim)
        movement = float((model.D.detach().cpu() - initial_dictionary).norm())
    row = {
        **grads,
        "dictionary": usage,
        "dictionary_movement_frobenius": movement,
        "response": rndb.response_stats(model, batch),
        "psi_health": rndb.psi_health(model),
    }
    return row


def _soup_model(result: Mapping[str, Any], subspace: cssd.CommonSubspace, dictionary: np.ndarray) -> rndb.RNDBModel:
    state = torch.load(
        CHECKPOINT_DIR / "RNDB-seed0_soup_state.pt", map_location="cpu", weights_only=False
    )
    model = rndb.build_rndb_model(dictionary, SEED, subspace)
    model.load_state_dict({key: value.float() for key, value in state.items()})
    return model


def _dictionary_health(
    result: Mapping[str, Any],
    subspace: cssd.CommonSubspace,
    dictionary: np.ndarray,
    valid_data: Sequence[Any],
) -> dict[str, Any]:
    model = _soup_model(result, subspace, dictionary)
    device = audit.attach_cpu(THREADS)
    loader = p1.make_env_loader(valid_data, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET))
    health = cm.dictionary_diagnostics(model, loader, device, dictionary)
    with torch.no_grad():
        coord = model.code(valid_data[0].dict_phi)
    health.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
        }
    )
    return health


# ---------------------------------------------------------------------------
# stage: frozen interventions
# ---------------------------------------------------------------------------


def _evaluate(model: rndb.RNDBModel, data: Sequence[Any], mask: Any) -> float:
    device = audit.attach_cpu(THREADS)
    loader = p1.make_env_loader(data, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET))
    return float(audit.evaluate_mask(model, loader, device, mask)["mae"])


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = RESULTS_DIR / "psi_disable.json"
    if out.exists() and not force:
        print("[interventions] cache hit", flush=True)
        return _read_json(out)
    rndb.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    result = _read_json(RESULTS_DIR / "run_seed0.json")
    model = _soup_model(result, subspace, dictionary)
    model.eval()
    valid_data = p1run.load_split("valid")
    mask = cm.C6_MASK

    baseline = _evaluate(model, valid_data, mask)
    m_psi0 = None
    m_node_off = None
    m_edge_off = None
    model.psi_node_on = False
    model.psi_edge_on = False
    m_psi0 = _evaluate(model, valid_data, mask)
    model.psi_node_on = True
    model.psi_edge_on = False
    m_edge_off = _evaluate(model, valid_data, mask)
    model.psi_node_on = False
    model.psi_edge_on = True
    m_node_off = _evaluate(model, valid_data, mask)
    model.psi_node_on = True
    model.psi_edge_on = True

    _write_json(
        RESULTS_DIR / "psi_disable.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "M_R_recomputed": baseline,
            "M_clean": baseline,
            "M_psi0": m_psi0,
            "M_both_off": m_psi0,
            "G_psi": m_psi0 - baseline,
            "gate": 0.003,
            "mechanism_load_bearing": bool((m_psi0 - baseline) >= 0.003),
        },
    )
    _write_json(
        RESULTS_DIR / "psi_node_disable.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_R": baseline,
            "M_node_off": m_node_off,
            "G_psi_node": m_node_off - baseline,
        },
    )
    _write_json(
        RESULTS_DIR / "psi_edge_disable.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_R": baseline,
            "M_edge_off": m_edge_off,
            "G_psi_edge": m_edge_off - baseline,
        },
    )

    # dictionary alpha -> 0 (keep common coordinate / chemistry / backend).
    model.dict_zero = True
    m_dict0 = _evaluate(model, valid_data, mask)
    model.dict_zero = False
    _write_json(
        RESULTS_DIR / "dictionary_zero.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_R": baseline,
            "M_dict0": m_dict0,
            "G_dict": m_dict0 - baseline,
            "gate": 0.010,
            "dictionary_load_bearing": bool((m_dict0 - baseline) >= 0.010),
        },
    )

    # assignment shuffles (frozen parent semantics).
    for kind, filename in (("node", "assignment_shuffle_node.json"), ("edge", "assignment_shuffle_edge.json")):
        rows: list[dict[str, Any]] = []
        shuffle_mask = cm.merge_masks(mask, audit.AuditMask(use_node_shuffle=(kind == "node"), use_edge_shuffle=(kind == "edge")))
        for shuffle_seed in SHUFFLE_SEEDS:
            audit.prepare_shuffles(valid_data, kind, int(shuffle_seed))
            try:
                mae = _evaluate(model, valid_data, shuffle_mask)
            finally:
                audit.clear_shuffles(valid_data)
            rows.append({"seed": int(shuffle_seed), "mae": float(mae), "delta_vs_M_R": float(mae - baseline)})
        deltas = [row["delta_vs_M_R"] for row in rows]
        _write_json(
            RESULTS_DIR / filename,
            {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": _git_commit(),
                "official_test_loaded": False,
                "kind": kind,
                "seeds": list(SHUFFLE_SEEDS),
                "M_R": baseline,
                "rows": rows,
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
            },
        )

    payload = _read_json(RESULTS_DIR / "psi_disable.json")
    print(
        f"[interventions] M_R={baseline:.6f} M_psi0={m_psi0:.6f} G_psi={m_psi0 - baseline:+.6f} "
        f"G_dict={m_dict0 - baseline:+.6f}",
        flush=True,
    )
    return payload


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


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    references = _read_json(RESULTS_DIR / "historical_references.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    psi_disable = _read_json(RESULTS_DIR / "psi_disable.json")
    node_off = _read_json(RESULTS_DIR / "psi_node_disable.json")
    edge_off = _read_json(RESULTS_DIR / "psi_edge_disable.json")
    dict_zero = _read_json(RESULTS_DIR / "dictionary_zero.json")
    shuffle_node = _read_json(RESULTS_DIR / "assignment_shuffle_node.json")
    shuffle_edge = _read_json(RESULTS_DIR / "assignment_shuffle_edge.json")
    dictionary_health = _read_json(RESULTS_DIR / "dictionary_health.json")
    response_stats = _read_json(RESULTS_DIR / "rndb_response_stats.json")

    m_r = float(run["soup"]["soup_valid_mae"])
    g_psi = float(psi_disable["G_psi"])
    g_dict = float(dict_zero["G_dict"])
    band = _band(m_r)
    hist = references["references"]["CSSD-Q1-seed0"]["value"]
    g_hist = float(hist) - m_r
    health_pass = bool(
        int(dictionary_health["active_atoms"]) >= 24
        and float(dictionary_health["effective_atoms"]) >= 8
        and float(dictionary_health["usage_top1_share"]) <= 0.50
    )
    psi_health_final = response_stats["psi_health_final"]
    psi_alive = bool(
        float(psi_health_final["psi_A"]["param_norm"]) > 0.0
        and float(psi_health_final["psi_E"]["param_norm"]) > 0.0
        and float(psi_health_final["psi_A"]["frac_abs_lt_1e12"]) < 0.95
        and float(psi_health_final["psi_E"]["frac_abs_lt_1e12"]) < 0.95
    )
    response = response_stats["diagnostics"][str(DIAG_EPOCHS[-1])]["response"]
    node_ratio = float(response["node"]["ratio"]["mean"])
    edge_ratio = float(response["edge"]["ratio"]["mean"])
    branch_alive = bool(node_ratio > 0.0 and edge_ratio > 0.0)

    if m_r <= 0.120 and g_psi >= 0.003 and g_dict >= 0.010 and health_pass and psi_alive:
        case = "A"
        verdict = "RNDB_STRONG_SINGLE_SEED_SUPPORTED"
    elif 0.120 < m_r <= 0.123 and g_psi >= 0.003 and g_dict >= 0.010 and health_pass:
        case = "B"
        verdict = "RNDB_PROMISING_SINGLE_SEED"
    elif 0.123 < m_r <= 0.126 and g_psi >= 0.003 and g_dict >= 0.010:
        case = "C"
        verdict = "RNDB_MECHANISM_SUPPORTED_TASK_NEUTRAL"
    elif m_r <= 0.123 and g_psi < 0.003:
        case = "D"
        verdict = "RNDB_GOOD_TRAJECTORY_MECHANISM_NOT_ESTABLISHED"
    elif g_dict < 0.010:
        case = "E"
        verdict = "RNDB_DICTIONARY_MECHANISM_LOST"
    elif m_r > 0.126:
        case = "F"
        verdict = "RNDB_NO_GO"
    else:
        case = "F"
        verdict = "RNDB_NO_GO"
    if (not psi_alive or not branch_alive) and g_psi < 0.003:
        case = "G"
        verdict = "RNDB_BRANCH_COLLAPSE"

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": THREADS,
        "official_test_loaded": False,
        "seed": SEED,
        "epochs": int(run["epochs_run"]),
        "wall_clock_s": float(run["wall_clock_s"]),
        "parameters": {
            "parent": int(preflight["parameter_audit"]["parent_params"]),
            "rndb": int(preflight["parameter_audit"]["rndb_params"]),
            "new": int(preflight["parameter_audit"]["new_params"]),
        },
        "M_R": m_r,
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": list(run["soup"]["members"]),
        "historical": {
            "CSSD_Q1_seed0_soup": float(hist),
            "G_hist_unmatched": g_hist,
        },
        "performance_band": band,
        "gates": {
            "correctness_all_passed": bool(correctness["all_passed"]),
            "smoke_passed": bool(smoke["passed"]),
            "G_psi": g_psi,
            "G_psi_gate": 0.003,
            "G_dict": g_dict,
            "G_dict_gate": 0.010,
            "dictionary_health_pass": health_pass,
            "psi_branch_alive": psi_alive,
            "node_response_ratio": node_ratio,
            "edge_response_ratio": edge_ratio,
        },
        "psi_interventions": {
            "M_clean": float(psi_disable["M_clean"]),
            "M_psi0": float(psi_disable["M_psi0"]),
            "M_node_off": float(node_off["M_node_off"]),
            "M_edge_off": float(edge_off["M_edge_off"]),
            "G_psi_node": float(node_off["G_psi_node"]),
            "G_psi_edge": float(edge_off["G_psi_edge"]),
        },
        "dictionary_intervention": {"M_dict0": float(dict_zero["M_dict0"]), "G_dict": g_dict},
        "assignment_shuffle": {
            "node_mean_delta": float(shuffle_node["mean_delta"]),
            "edge_mean_delta": float(shuffle_edge["mean_delta"]),
            "node_rows": shuffle_node["rows"],
            "edge_rows": shuffle_edge["rows"],
        },
        "case": case,
        "verdict": verdict,
    }
    rndb.official_test_blocker(summary)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, run, dictionary_health, response_stats, node_off, edge_off, shuffle_node, shuffle_edge, references)
    print(f"[analysis] M_R={m_r:.6f} band={band} case={case} verdict={verdict}", flush=True)
    return summary


def _write_report(
    summary: Mapping[str, Any],
    run: Mapping[str, Any],
    dictionary_health: Mapping[str, Any],
    response_stats: Mapping[str, Any],
    node_off: Mapping[str, Any],
    edge_off: Mapping[str, Any],
    shuffle_node: Mapping[str, Any],
    shuffle_edge: Mapping[str, Any],
    references: Mapping[str, Any],
) -> None:
    s = summary
    lines: list[str] = []
    lines.append("# e2e_dictenv_rndb_v1 — Rolewise Nonlinear Dictionary Binding")
    lines.append("")
    lines.append("CPU only · official test never loaded · single seed-0 trajectory.")
    lines.append("")
    lines.append("## A. Provenance")
    lines.append("")
    lines.append(f"- starting HEAD `cfe0716f3c4ac3c72df9521c9e075092b5213ac3`")
    lines.append(f"- formal/analysis commit `{s['git_commit']}`")
    lines.append(f"- device `cpu` · threads `{s['threads']}`")
    lines.append(f"- `official_test_loaded = false`")
    lines.append("")
    lines.append("## B. Historical references (read-only; never rerun; unmatched)")
    lines.append("")
    for name, ref in references["references"].items():
        value = ref.get("value")
        line = f"- {name}: " + ("not trained (context only)" if value is None else f"`{_fmt(value, 6)}`")
        lines.append(line + f" — {ref.get('status', '')} `{ref.get('artifact', '')}`")
    lines.append("")
    lines.append(
        "The prompt's approximate CSSD-q1/q2 values do not match the local artifacts; the exact "
        "artifacts above are the source of truth. No matched baseline was rerun."
    )
    lines.append("")
    lines.append("## C. Exact RNDB architecture")
    lines.append("")
    lines.append("```")
    lines.append("p_{v,k}  = z_{v,k} (W_A_S[k,:] odot c_v),   c_v = q_v W_A_C / sqrt(D_A)")
    lines.append("u_v      = p_v^common + sum_{k in K_dict} [ p_{v,k} + psi_A(p_{v,k}) ]")
    lines.append("p^E_{uv,k} = r_{uv,k} odot c^E_{uv},  r from [+, |.|, odot] blocks")
    lines.append("u^E_uv   = p^E,common_uv + sum_{k in K_dict} [ p^E_{uv,k} + psi_E(p^E_{uv,k}) ]")
    lines.append("psi_A: 96->32->96 (SiLU), bias=False;  psi_E: 48->16->48 (SiLU), bias=False")
    lines.append("K_common = {0}; K_dict = {1..32}; W1 Kaiming, W2 ~ N(0, 0.01)")
    lines.append("```")
    lines.append("")
    lines.append("## D. Why RNDB is new")
    lines.append("")
    lines.append(
        "- not F (multi-rank bilinear): F is affine in the coordinate with new heads; RNDB applies "
        "a nonlinearity per role before the sum, with no new heads."
    )
    lines.append("- not FEC-D1: no 2688-D covariance/joint statistic is formed.")
    lines.append("- not FSAB/BCE: no new binding stream or residual MLP is added.")
    lines.append("- not T1: no coarse146 / handcrafted bypass; no raw phi inside psi.")
    lines.append("")
    lines.append("## E. Correctness gates")
    lines.append("")
    lines.append(f"- all passed: `{s['gates']['correctness_all_passed']}`")
    lines.append("")
    lines.append("## F. Parameter budget")
    lines.append("")
    p = s["parameters"]
    lines.append(f"- parent `{p['parent']}`, RNDB `{p['rndb']}`, new `{p['new']}`")
    lines.append("")
    lines.append("## G. Training")
    lines.append("")
    lines.append(
        f"- epochs `{s['epochs']}` · wall `{_fmt(s['wall_clock_s'], 1)} s` · seed `{s['seed']}`"
    )
    lines.append(
        f"- best valid `{_fmt(s['best_valid_mae'], 6)}` @ {s['best_epoch']} · Top-5 members "
        f"{s['soup_members']} · soup `{_fmt(s['M_R'], 6)}`"
    )
    lines.append("")
    lines.append("## H. Performance interpretation")
    lines.append("")
    lines.append(f"- `M_R = {_fmt(s['M_R'], 6)}` → band **{s['performance_band']}**")
    lines.append(
        f"- `G_hist = {_fmt(s['historical']['G_hist_unmatched'], 6)}` vs historical CSSD-q1 "
        f"`{_fmt(s['historical']['CSSD_Q1_seed0_soup'], 6)}` — **historical, unmatched comparison**"
    )
    lines.append("")
    lines.append("## I. psi mechanism")
    lines.append("")
    pi = s["psi_interventions"]
    lines.append(
        f"- `M_clean={_fmt(pi['M_clean'], 6)}` `M_psi0={_fmt(pi['M_psi0'], 6)}` "
        f"`G_psi={_fmt(pi['G_psi'], 6)}` (gate 0.003)"
    )
    lines.append(
        f"- `M_node_off={_fmt(pi['M_node_off'], 6)}` (`G_psi_node={_fmt(pi['G_psi_node'], 6)}`), "
        f"`M_edge_off={_fmt(pi['M_edge_off'], 6)}` (`G_psi_edge={_fmt(pi['G_psi_edge'], 6)}`)"
    )
    lines.append("")
    lines.append("## J. psi health (final soup)")
    lines.append("")
    lines.append("| operator | param norm | frac<1e-12 | W1 norm | W2 norm |")
    lines.append("|---|---|---|---|---|")
    for name in ("psi_A", "psi_E"):
        h = response_stats["psi_health_final"][name]
        lines.append(
            f"| {name} | {_fmt(h['param_norm'], 4)} | {_fmt(h['frac_abs_lt_1e12'], 4)} | "
            f"{_fmt(h['W1_norm'], 4)} | {_fmt(h['W2_norm'], 4)} |"
        )
    lines.append("")
    lines.append("| role | mean ratio ||psi||/||p|| | mean ||psi|| | eff. rank |")
    lines.append("|---|---|---|---|")
    last = response_stats["diagnostics"][str(max(int(k) for k in response_stats["diagnostics"]))]["response"]
    for role in ("node", "edge"):
        block = last[role]
        lines.append(
            f"| {role} | {_fmt(block['ratio']['mean'], 4)} | {_fmt(block['psi_norm']['mean'], 6)} | "
            f"{_fmt(block['psi_output_effective_rank'], 3)} |"
        )
    lines.append("")
    lines.append("## K. Dictionary mechanism")
    lines.append("")
    lines.append(
        f"- `M_dict0={_fmt(s['dictionary_intervention']['M_dict0'], 6)}` "
        f"`G_dict={_fmt(s['dictionary_intervention']['G_dict'], 6)}` (gate 0.010) → "
        f"`{'load_bearing' if s['dictionary_intervention']['G_dict'] >= 0.010 else 'not_load_bearing'}`"
    )
    lines.append(
        f"- health: active {dictionary_health['active_atoms']}/32, effective "
        f"{_fmt(dictionary_health['effective_atoms'], 2)}, top1 share "
        f"{_fmt(dictionary_health['usage_top1_share'], 3)}, recon "
        f"{_fmt(dictionary_health['valid_reconstruction_relative'], 5)}"
    )
    lines.append("")
    lines.append("## L. Assignment correspondence")
    lines.append("")
    lines.append(
        f"- node shuffle mean delta `{_fmt(shuffle_node['mean_delta'], 6)}` "
        f"(seeds {shuffle_node['seeds']})"
    )
    lines.append(
        f"- edge shuffle mean delta `{_fmt(shuffle_edge['mean_delta'], 6)}` "
        f"(seeds {shuffle_edge['seeds']})"
    )
    lines.append("")
    lines.append("## M. Final verdict")
    lines.append("")
    lines.append(f"**Case {s['case']} — {s['verdict']}**")
    lines.append("")
    lines.append("## N. Next step")
    lines.append("")
    if s["case"] in ("A", "B"):
        lines.append(
            "One next step only: matched confirmation next round (seed 1 and a same-round matched "
            "baseline) before any architecture claim. Not executed in this round."
        )
    else:
        lines.append(
            "Stop this direction. No rescue, no seed 1, no width/init/gate sweep. Record the "
            "negative mechanism result."
        )
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": s["git_commit"],
        "official_test_loaded": False,
        "M_R": s["M_R"],
        "performance_band": s["performance_band"],
        "G_psi": s["gates"]["G_psi"],
        "G_dict": s["gates"]["G_dict"],
        "case": s["case"],
        "verdict": s["verdict"],
        "seed1_authorized": bool(s["case"] in ("A", "B")),
        "seed1_executed": False,
        "post_hoc_rescue": False,
    }
    (RESULTS_DIR / "DECISION.md").write_text(
        "\n".join(
            [
                "# e2e_dictenv_rndb_v1 — decision",
                "",
                f"**Case {s['case']} — {s['verdict']}**",
                "",
                f"- `M_R = {_fmt(s['M_R'], 6)}` ({s['performance_band']})",
                f"- `G_psi = {_fmt(s['gates']['G_psi'], 6)}` (gate 0.003)",
                f"- `G_dict = {_fmt(s['gates']['G_dict'], 6)}` (gate 0.010)",
                f"- seed 1 authorized for a *future* round: `{decision['seed1_authorized']}` "
                f"(not executed this round)",
                "- no post-hoc rescue",
                "",
            ]
        ),
        encoding="utf-8",
    )


def chain(force: bool = False) -> None:
    stage_references()
    stage_preflight()
    stage_correctness()
    stage_smoke()
    stage_train(force=force)
    stage_interventions(force=force)
    stage_analysis()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "references",
            "preflight",
            "correctness",
            "smoke",
            "train",
            "interventions",
            "analysis",
            "chain",
        ],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "references":
        stage_references()
    elif args.stage == "preflight":
        stage_preflight()
    elif args.stage == "correctness":
        stage_correctness()
    elif args.stage == "smoke":
        stage_smoke()
    elif args.stage == "train":
        stage_train(force=bool(args.force))
    elif args.stage == "interventions":
        stage_interventions(force=bool(args.force))
    elif args.stage == "analysis":
        stage_analysis()
    elif args.stage == "chain":
        chain(force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
