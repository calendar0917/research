"""E2E-DictEnv-Sem108-v1 runner — Shell-Resolved Primitive Semantic Interface.

Round ``e2e_dictenv_sem108_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_sem108_v1.py``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_sem108_v1_preregistration.md``.

Stages
------
``references audit preflight correctness smoke train interventions analysis chain``

Exactly one from-scratch ``CSSD-Sem108`` seed-0 trajectory (320 epochs) on CPU.
The official ZINC test split is never loaded; every payload records
``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = sem.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_sem108_v1"
AUDIT_DIR = RESULTS_DIR / "audit"
MECHANISM_DIR = RESULTS_DIR / "mechanism"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_sem108_v1_preregistration.md"
PRIOR_AUDIT_PATH = NOTES_DIR / "e2e_dictenv_sem108_v1_prior_artifact_audit.md"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

THREADS = 8
TRAIN_EPOCHS = 320
SEED = 0
SMOKE_EPOCHS = 8
SMOKE_SUBSET = 1024  # 8 x 8 = 64 optimizer steps (<= 100)
SHUFFLE_SEEDS = audit.SHUFFLE_SEEDS  # (101, 202, 303, 404, 505)
DIAG_EPOCHS: tuple[int, ...] = (1, 20, 40, 80, 160, 240, 320)

#: pre-registered absolute-MAE performance bands (pre-registration section 8).
BANDS: tuple[tuple[float, str], ...] = (
    (0.120, "SEM108_STRONG_SINGLE_SEED_SIGNAL"),
    (0.123, "SEM108_PROMISING_SINGLE_SEED"),
    (0.127, "SEM108_WITHIN_EXISTING_PERFORMANCE_BAND"),
    (float("inf"), "SEM108_NO_GO_TASK_LEVEL"),
)
MATERIAL_REGRESSION = 0.135
#: relabel invariance tolerance: relabelling permutes the index_add_ reduction
#: order, so in float32 the prediction may move by a few ulps (observed ~1.7e-5).
RELABEL_TOL = 1.0e-4
GATE_SEM_SHUFFLE_LOAD_BEARING = 0.005
GATE_SEM_SHUFFLE_STRONG = 0.010
GATE_SEM_SHUFFLE_WEAK = 0.003
GATE_DICT_CORR_CLEAR = 0.010
GATE_DICT_CORR_DIRECTIONAL = 0.003


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


def _model_factory(dictionary: np.ndarray, seed: int, subspace: cssd.CommonSubspace) -> sem.SEM108Model:
    return sem.build_sem108_model(dictionary, int(seed), subspace)


def _soup_state() -> dict[str, torch.Tensor]:
    state = torch.load(
        CHECKPOINT_DIR / "SEM108-seed0_soup_state.pt", map_location="cpu", weights_only=False
    )
    return {key: value.float() for key, value in state.items()}


def _soup_model() -> sem.SEM108Model:
    model = sem.build_sem108_model(_dictionary_tensor(), SEED, _load_parent_subspace())
    model.load_state_dict(_soup_state())
    return model


# ---------------------------------------------------------------------------
# stage: references
# ---------------------------------------------------------------------------


def stage_references() -> dict[str, Any]:
    _ensure_dirs()

    def _read(relative: str) -> dict[str, Any]:
        return _read_json(REPO_ROOT / relative)

    cssd_path = "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/training/final.json"
    cssd_final = _read(cssd_path)
    cssd_selection = _read(
        "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/zero_training/selection.json"
    )
    clean_path = (
        "tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1/stage_c_independence/frozen_C6_seed0.json"
    )
    clean = _read(clean_path)
    t1_path = "tracks/ksvd/results/e2e_dictenv_t1/tuning_decision.json"
    t1 = _read(t1_path)
    rndb_path = "tracks/ksvd/results/e2e_dictenv_rndb_v1/summary.json"
    rndb = _read(rndb_path)
    tpa_path = "tracks/ksvd/results/e2e_dictenv_training_protocol_audit_v1/comparison.json"
    tpa = _read(tpa_path)
    p1_path = "tracks/ksvd/results/e2e_dictenv_p1/sparse_seed0.json"
    p1_run = _read(p1_path)
    fec_s1_path = "tracks/ksvd/results/fec_s1/final_sparse_seed0.json"
    fec_s1 = _read(fec_s1_path) if (REPO_ROOT / fec_s1_path).exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "policy": (
            "read-only durable artifacts; no historical model is retrained; comparisons are "
            "historical / unmatched / contextual only"
        ),
        "references": {
            "CSSD-q1-seed0": {
                "value": float(cssd_final["soup"]["soup_valid_mae"]),
                "best_valid_mae": float(cssd_final["best_valid_mae"]),
                "best_epoch": int(cssd_final["best_epoch"]),
                "actual_params": int(cssd_final["actual_params"]),
                "artifact": cssd_path,
                "status": "not_rerun_unmatched_context",
            },
            "CSSD-q1-zero-training-selection": {
                "q1_passed_seeds": cssd_selection["q1_passed_seeds"],
                "q2_passed_seeds": cssd_selection["q2_passed_seeds"],
                "selected": cssd_selection["selected"],
                "status": "never_trained_context_only",
            },
            "FINAL-CLEAN-C6-seed0": {
                "value": float(clean["baseline_valid_mae"]),
                "artifact": clean_path,
                "status": "not_rerun_unmatched_context",
            },
            "T1-tuned-seed0": {
                "value": float(t1["best_tuned_soup_valid_mae"]),
                "candidate_id": t1["candidate_id"],
                "artifact": t1_path,
                "status": "not_rerun_unmatched_context",
            },
            "RNDB-seed0": {
                "value": float(rndb["M_R"]),
                "artifact": rndb_path,
                "status": "not_rerun_unmatched_context",
            },
            "training-protocol-audit-control-lr1e3": {
                "value": float(tpa["arms"]["control_lr1e3"]["soup_valid_mae"]),
                "best_valid_mae": float(tpa["arms"]["control_lr1e3"]["best_valid_mae"]),
                "artifact": tpa_path,
                "status": "not_rerun_unmatched_context",
            },
            "training-protocol-audit-low-lr1e4": {
                "value": float(tpa["arms"]["low_lr1e4"]["soup_valid_mae"]),
                "best_valid_mae": float(tpa["arms"]["low_lr1e4"]["best_valid_mae"]),
                "artifact": tpa_path,
                "status": "not_rerun_unmatched_context",
            },
            "P1-sparse-seed0": {
                "value": float(p1_run["soup"]["soup_valid_mae"]),
                "artifact": p1_path,
                "status": "not_rerun_unmatched_context",
            },
            "FEC-S1-sparse-seed0": (
                None
                if fec_s1 is None
                else {
                    "value": float(fec_s1.get("soup", {}).get("soup_valid_mae", float("nan"))),
                    "artifact": fec_s1_path,
                    "status": "not_rerun_unmatched_context",
                }
            ),
        },
    }
    sem.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "historical_references.json", payload)
    print(
        "[references] CSSD-q1={:.6f} FINAL-CLEAN={:.6f} T1={:.6f} RNDB={:.6f}".format(
            payload["references"]["CSSD-q1-seed0"]["value"],
            payload["references"]["FINAL-CLEAN-C6-seed0"]["value"],
            payload["references"]["T1-tuned-seed0"]["value"],
            payload["references"]["RNDB-seed0"]["value"],
        ),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: zero-training prior audit (Phase A)
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
        sem.official_test_blocker(identity)
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
        sem.official_test_blocker(relation)
        _write_json(relation_path, relation)
    split = relation["splits"]["train"]
    print(
        "[audit] A2 root_agree={:.4f} atom_maxerr={:.2e} bond_maxerr={:.2e} "
        "A4 distinct_sem_per_anchor_median={:.1f}".format(
            split["root"]["argmax_agreement"],
            split["atom_mass"]["max_abs_error"],
            split["bond_mass"]["max_abs_error"],
            split["redundancy"]["distinct_sem108_per_anchor_exact"]["median"],
        ),
        flush=True,
    )

    t1_path = AUDIT_DIR / "t1_block_ablation.json"
    if t1_path.exists() and not force:
        t1 = _read_json(t1_path)
    else:
        t1 = sem.audit_t1_block_ablation()
        sem.official_test_blocker(t1)
        _write_json(t1_path, t1)
    print(f"[audit] A3 {t1['status']}", flush=True)

    decision = sem.audit_decision(identity, relation, t1)
    decision["git_commit"] = _git_commit()
    sem.official_test_blocker(decision)
    _write_json(AUDIT_DIR / "audit_decision.json", decision)
    print(f"[audit] decision={decision['decision']}", flush=True)
    if decision["decision"] != "PROCEED":
        raise RuntimeError(f"Phase-A stop rule fired: {decision}")
    return decision


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
    import hashlib

    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    parent = cssd.build_cssd_model(dictionary, SEED, subspace)
    model = _model_factory(dictionary, SEED, subspace)
    parent_params = int(sum(p.numel() for p in parent.parameters()))
    audit_payload = sem.parameter_audit(model, parent_params)
    if not audit_payload["parameter_ratio_within_bound"]:
        raise RuntimeError(f"parameter budget violated: {audit_payload}")
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
        },
        "parent": {
            "config": cm.H1_CONFIG.as_dict(),
            "lambda_rec": float(cm.H1_LAMBDA),
            "c6_mask_identity": bool(cm.CSSD_MASK is cm.C6_MASK),
            "c6_equivalence_check": bool(cm.c6_equivalence_check()),
            "subspace_kind": subspace.kind,
            "common_dim": int(subspace.q),
            "common_rms": [float(value) for value in subspace.rms],
            "d_struct": int(model.W_A_S.shape[0]),
            "n_common": int(model.common_dim),
            "n_dict": int(model.config.K),
            "dictionary_shape": list(np.asarray(dictionary).shape),
            "parent_local_interface": sem.parent_local_interface_parameters(),
            "decoder_layout": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in layout.items()
            },
            "node_slots": [int(p2.N_SHELLS), int(p2.D_A)],
            "edge_slots": [int(p2.SHELLPAIR_CLASSES), int(p2.D_E)],
            "env_dim": int(p2.ENV_DIM),
            "reader_input_dim": int(97 + p2.DISTANCE_BUCKETS * (2 * p2.PAIR_HIDDEN + 1) + 32 + p2.TOPOLOGY_OUT),
        },
        "semantic_interface": {
            "sem108_block": [int(sem.SEM_ATOM_BLOCK[0]), int(sem.SEM_BOND_BLOCK[1])],
            "atom_shell": list(sem.SEM_ATOM_BLOCK),
            "bond_shell": list(sem.SEM_BOND_BLOCK),
            "size2": list(sem.SIZE2_BLOCK),
            "interface_dim": int(sem.SEM_INTERFACE_DIM),
            "fusion_input_dim": int(sem.SEM_FUSION_IN),
            "fusion_hidden": int(sem.SEM_FUSION_HIDDEN),
            "forbidden_t1_blocks_exposed": False,
            "rndb_modules_present": False,
        },
        "parameter_audit": audit_payload,
        "epochs": TRAIN_EPOCHS,
        "seed": SEED,
    }
    sem.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration"])
    _write_json(RESULTS_DIR / "parameter_audit.json", audit_payload)
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {prereg_sha[:12]} parent={parent_params} candidate="
        f"{audit_payload['candidate_params']} ratio={audit_payload['relative_delta']:.5f} "
        f"H_sem={audit_payload['sem_fusion_hidden']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: correctness gates
# ---------------------------------------------------------------------------


def _clone_batch(batch: Any) -> Any:
    """Clone a PyG ``Batch`` or an equivalent ``SimpleNamespace`` payload."""
    if hasattr(batch, "clone"):
        return batch.clone()
    return SimpleNamespace(
        **{key: (value.clone() if torch.is_tensor(value) else value) for key, value in vars(batch).items()}
    )


def _synthetic_batch(n_nodes: int = 40, n_occ: int = 120, n_bond: int = 90, seed: int = 0) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(int(seed))
    data = SimpleNamespace()
    data.dict_phi = torch.randn(n_nodes, cssd.PHI_DIM, generator=generator)
    data.dict_atom = torch.randint(0, int(p2.ATOM_CATEGORIES), (n_nodes,), generator=generator)
    data.anchor = torch.randn(n_nodes, audit.ANCHOR_DIM_EXPECTED, generator=generator)
    data.patch_cont = torch.randn(n_nodes, sem.PATCH_CONT_DIM, generator=generator)
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
    data.pair_relation = torch.randn(5, 21, generator=generator)
    data.pair_bucket = torch.randint(0, int(p2.DISTANCE_BUCKETS), (5,), generator=generator)
    data.pair_index = torch.stack([torch.randint(0, n_nodes, (5,), generator=generator) for _ in range(2)])
    return data


def _real_batch(subset: int = 32, split: str = "train") -> Any:
    data_list = p1run.load_split(split, subset=subset)
    return next(iter(p1.make_env_loader(data_list, subset, False, 0)))


def _relabel_invariance(model: sem.SEM108Model, n_molecules: int = 2) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e_v0
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import _relabel_raw

    data_list = p1run.load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(REPO_ROOT / "data/ZINC", "val"))[: int(n_molecules)]
    max_diff = 0.0
    for data, raw_molecule in zip(data_list, raw):
        n = int(data.num_nodes)
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(20260930))
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(n)
        graph_new, _node_types, edge_types_new = zlr._data_to_graph(_relabel_raw(raw_molecule))
        incidence = e2e_v0.env_incidence(graph_new, edge_types_new)
        relabelled = data.clone()
        relabelled.dict_phi = data.dict_phi[permutation]
        relabelled.dict_atom = data.dict_atom[permutation]
        relabelled.patch_cont = data.patch_cont[permutation]
        relabelled.anchor = data.anchor[permutation]
        relabelled.env_occ_node = incidence["occ_node"]
        relabelled.env_occ_root = incidence["occ_root"]
        relabelled.env_occ_shell = incidence["occ_shell"]
        relabelled.env_bond_u = incidence["bond_u"]
        relabelled.env_bond_v = incidence["bond_v"]
        relabelled.env_bond_root = incidence["bond_root"]
        relabelled.env_bond_shellpair = incidence["bond_shellpair"]
        relabelled.env_bond_type = incidence["bond_type"]
        relabelled.pair_index = inverse[data.pair_index]
        with torch.no_grad():
            p0 = model(e2e_v0.env_collate([data])).view(-1)
            p1v = model(e2e_v0.env_collate([relabelled])).view(-1)
        max_diff = max(max_diff, float((p0 - p1v).abs().max()))
    return {
        "n_molecules": int(n_molecules),
        "max_abs_pred_diff": float(max_diff),
        "tolerance": RELABEL_TOL,
        "passed": bool(max_diff <= RELABEL_TOL),
    }


def stage_correctness() -> dict[str, Any]:
    _ensure_dirs()
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    parent = cssd.build_cssd_model(dictionary, SEED, subspace)
    model = _model_factory(dictionary, SEED, subspace)
    mask = cm.C6_MASK
    gates: dict[str, Any] = {}

    # -- G0/G1/G2: identity, scaler provenance, marginal relations ------------
    identity_path = AUDIT_DIR / "sem108_identity.json"
    relation_path = AUDIT_DIR / "anchor_relation.json"
    identity = (
        _read_json(identity_path)
        if identity_path.exists()
        else sem.audit_sem108_identity(n_train=4, n_valid=4)
    )
    relation = (
        _read_json(relation_path)
        if relation_path.exists()
        else sem.audit_anchor_relation_and_redundancy()
    )
    geometry = sem.resolve_sem108_geometry()
    gates["G0_sem108_identity"] = {
        "atom_shell_cols": int(geometry["atom_shell_cols"]),
        "bond_shell_cols": int(geometry["bond_shell_cols"]),
        "sem_dim": int(geometry["sem_dim"]),
        "cache_bit_identical_train": bool(identity["splits"]["train"]["cache_bit_identical"]),
        "cache_bit_identical_valid": bool(identity["splits"]["valid"]["cache_bit_identical"]),
        "semantics_max_abs_diff_train": float(identity["splits"]["train"]["semantics_max_abs_diff"]),
        "semantics_max_abs_diff_valid": float(identity["splits"]["valid"]["semantics_max_abs_diff"]),
        "passed": bool(
            geometry["sem_dim"] == 108
            and identity["splits"]["train"]["cache_bit_identical"]
            and identity["splits"]["valid"]["cache_bit_identical"]
            and identity["splits"]["train"]["semantics_max_abs_diff"] == 0.0
            and identity["splits"]["valid"]["semantics_max_abs_diff"] == 0.0
        ),
    }
    gates["G1_train_fit_scaler"] = {
        "fit_split": identity["standardization"]["fit_split"],
        "valid_in_fit": bool(identity["standardization"]["valid_in_fit"]),
        "mean_sha256": identity["standardization"]["mean_sha256"],
        "scale_sha256": identity["standardization"]["scale_sha256"],
        "passed": bool(
            identity["standardization"]["fit_split"] == "official train"
            and not identity["standardization"]["valid_in_fit"]
        ),
    }
    gates["G2_anchor_marginals"] = {
        "train_root_agreement": float(relation["splits"]["train"]["root"]["argmax_agreement"]),
        "train_atom_fraction_exact": float(relation["splits"]["train"]["atom_mass"]["fraction_exact"]),
        "train_bond_fraction_exact": float(relation["splits"]["train"]["bond_mass"]["fraction_exact"]),
        "valid_root_agreement": float(relation["splits"]["valid"]["root"]["argmax_agreement"]),
        "valid_atom_fraction_exact": float(relation["splits"]["valid"]["atom_mass"]["fraction_exact"]),
        "valid_bond_fraction_exact": float(relation["splits"]["valid"]["bond_mass"]["fraction_exact"]),
        "a2_pass": bool(relation["a2_pass"]),
        "passed": bool(relation["a2_pass"]),
    }

    # -- G3: size2 bit-identical retention ------------------------------------
    batch = _real_batch(subset=32, split="train")
    with torch.no_grad():
        interface = model.semantic_interface(model.code(batch.dict_phi), batch, sem.SEMMask())
    size2_from_anchor = batch.anchor[:, int(sem.SIZE2_BLOCK[0]) : int(sem.SIZE2_BLOCK[1])]
    size2_from_patch = batch.patch_cont[:, 140:142]
    gates["G3_size2_retained"] = {
        "anchor_vs_patch_bit_identical": bool(torch.equal(size2_from_anchor, size2_from_patch)),
        "interface_last2_bit_identical": bool(torch.equal(interface[:, 108:110], size2_from_anchor)),
        "passed": bool(
            torch.equal(size2_from_anchor, size2_from_patch)
            and torch.equal(interface[:, 108:110], size2_from_anchor)
        ),
    }

    # -- G4/G5/G6/G7: parent code-path identity --------------------------------
    state_model = model.state_dict()
    state_parent = parent.state_dict()
    replaced_prefixes = ("fusion.", "anchor_encoder.")
    shared = {
        key: value
        for key, value in state_parent.items()
        if key in state_model
        and state_model[key].shape == value.shape
        and not key.startswith(replaced_prefixes)
    }
    shared_mismatch = [key for key, value in shared.items() if not torch.equal(value, state_model[key])]
    equivalence = sem.build_parent_equivalence_model(dictionary, SEED, subspace)
    equivalence.load_state_dict(parent.state_dict())
    model.eval()
    parent.eval()
    equivalence.eval()
    with torch.no_grad():
        coord_model = model.code(batch.dict_phi)
        coord_parent = parent.code(batch.dict_phi)
        node_model = model.node_slots(coord_model, batch)
        node_parent = parent.node_slots(coord_parent, batch)
        edge_model = model.edge_slots(coord_model, batch)
        edge_parent = parent.edge_slots(coord_parent, batch)
        relation_input = batch.pair_relation[:, list(p1.P1_RELATION_INDICES)]
        rel_model = model.relation_encoder(relation_input)
        rel_parent = parent.relation_encoder(relation_input)
        global_model = model.global_encoder(batch.global_context)
        global_parent = parent.global_encoder(batch.global_context)
        topo_model = model.topology_encoder(batch.topology_features)
        topo_parent = parent.topology_encoder(batch.topology_features)
        pred_parent, aux_parent = parent(batch, mask=mask, return_aux=True)
        pred_equiv, aux_equiv = equivalence(batch, mask=mask, return_aux=True)
    deltas = {
        "coord": float((coord_model - coord_parent).abs().max()),
        "node_slots": float((node_model - node_parent).abs().max()),
        "edge_slots": float((edge_model - edge_parent).abs().max()),
        "relation": float((rel_model - rel_parent).abs().max()),
        "global": float((global_model - global_parent).abs().max()),
        "topology": float((topo_model - topo_parent).abs().max()),
        "environment": float((aux_equiv["E"] - aux_parent["E"]).abs().max()),
        "prediction": float((pred_equiv - pred_parent).abs().max()),
    }
    gates["G4_parent_code_path"] = {
        "shared_state_keys": int(len(shared)),
        "shared_state_mismatches": shared_mismatch,
        "parent_equivalence_max_abs": deltas,
        "passed": bool(not shared_mismatch and all(value == 0.0 for value in deltas.values())),
    }
    gates["G5_node_binding"] = {
        "max_abs_diff": deltas["node_slots"],
        "passed": bool(deltas["node_slots"] == 0.0),
    }
    gates["G6_edge_binding"] = {
        "max_abs_diff": deltas["edge_slots"],
        "passed": bool(deltas["edge_slots"] == 0.0),
    }
    gates["G7_relation_backend"] = {
        "relation_max_abs_diff": deltas["relation"],
        "global_max_abs_diff": deltas["global"],
        "topology_max_abs_diff": deltas["topology"],
        "environment_max_abs_diff": deltas["environment"],
        "prediction_max_abs_diff": deltas["prediction"],
        "reader_class": type(model.reader).__name__,
        "reader_class_parent": type(parent.reader).__name__,
        "pair_encoder_reused": bool(type(model.pair_encoder) is type(parent.pair_encoder)),
        "distance_gate_reused": bool(type(model.distance_gate) is type(parent.distance_gate)),
        "passed": bool(
            max(
                deltas["relation"],
                deltas["global"],
                deltas["topology"],
                deltas["environment"],
                deltas["prediction"],
            )
            == 0.0
            and type(model.reader) is type(parent.reader)
            and type(model.pair_encoder) is type(parent.pair_encoder)
            and type(model.distance_gate) is type(parent.distance_gate)
        ),
    }

    # -- G8: C6 masks ----------------------------------------------------------
    gates["G8_c6_masks"] = {
        "cssd_mask_is_c6": bool(cm.CSSD_MASK is cm.C6_MASK),
        "c6_equivalence_check": bool(cm.c6_equivalence_check()),
        "arm_mask_is_c6": bool(cm.arm_mask(cm.CSSD_SPEC) is cm.C6_MASK),
        "passed": bool(
            cm.CSSD_MASK is cm.C6_MASK
            and cm.c6_equivalence_check()
            and cm.arm_mask(cm.CSSD_SPEC) is cm.C6_MASK
        ),
    }

    # -- G9: RNDB absence ------------------------------------------------------
    parameter_names = [name for name, _ in model.named_parameters()]
    psi_names = [name for name in parameter_names if "psi" in name]
    gates["G9_rndb_absent"] = {
        "psi_parameter_names": psi_names,
        "has_psi_A": bool(hasattr(model, "psi_A")),
        "has_psi_E": bool(hasattr(model, "psi_E")),
        "passed": bool(
            not psi_names and not hasattr(model, "psi_A") and not hasattr(model, "psi_E")
        ),
    }

    # -- G10: forbidden T1 blocks not exposed ----------------------------------
    forbidden = sem.forbidden_block_columns(batch.patch_cont)
    batch_forbidden = _clone_batch(batch)
    batch_forbidden.patch_cont = batch.patch_cont.clone()
    batch_forbidden.patch_cont[:, 108:146] += 5.0
    batch_sem = _clone_batch(batch)
    batch_sem.patch_cont = batch.patch_cont.clone()
    batch_sem.patch_cont[:, 0:108] += 0.25
    with torch.no_grad():
        pred_clean = model(batch, mask=mask)
        pred_forbidden = model(batch_forbidden, mask=mask)
        pred_sem = model(batch_sem, mask=mask)
    gates["G10_forbidden_blocks_absent"] = {
        "interface_dim": int(sem.SEM_INTERFACE_DIM),
        "forbidden_perturbation_prediction_diff": float((pred_clean - pred_forbidden).abs().max()),
        "semantic_perturbation_prediction_diff": float((pred_clean - pred_sem).abs().max()),
        "forbidden_columns_read": int(sem.SEM_DIM),
        "forbidden_block_widths": {name: int(value.shape[1]) for name, value in forbidden.items()},
        "passed": bool(
            sem.SEM_INTERFACE_DIM == 110
            and float((pred_clean - pred_forbidden).abs().max()) == 0.0
            and float((pred_clean - pred_sem).abs().max()) > 0.0
        ),
    }

    # -- G11: parameter ratio --------------------------------------------------
    parameter_audit = sem.parameter_audit(model, int(sum(p.numel() for p in parent.parameters())))
    gates["G11_parameter_ratio"] = {**parameter_audit, "passed": bool(parameter_audit["parameter_ratio_within_bound"])}

    # -- G12: permutation invariance -------------------------------------------
    gates["G12_relabel_invariance"] = _relabel_invariance(model, n_molecules=2)

    # -- G13: official-test blocker -------------------------------------------
    blocked = False
    try:
        sem.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        blocked = True
    gates["G13_official_test_blocker"] = {"blocker_raises_on_true": bool(blocked), "passed": bool(blocked)}

    all_passed = all(bool(entry.get("passed")) for entry in gates.values() if "passed" in entry)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "gates": gates,
        "all_passed": bool(all_passed),
    }
    sem.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "correctness.json", payload)
    for key in sorted(gates):
        print(f"[correctness] {key} passed={gates[key]['passed']}", flush=True)
    if not all_passed:
        raise RuntimeError(f"correctness gates failed: {[k for k, v in gates.items() if not v.get('passed')]}")
    return payload


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------


def _gradient_diagnostics(model: sem.SEM108Model, batch: Any, mask: Any, lam: float) -> dict[str, Any]:
    model.train(False)
    prediction, aux = model(batch, mask=mask, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(lam) * rec
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
        "coord_nnz_per_row_mean": float((alpha != 0).double().sum(dim=1).mean()),
    }
    model.zero_grad(set_to_none=True)
    model.train(True)
    return out


def _interface_activity(model: sem.SEM108Model, batch: Any, mask: Any) -> dict[str, Any]:
    model.eval()
    batch_sem = _clone_batch(batch)
    batch_sem.patch_cont = batch.patch_cont.clone()
    batch_sem.patch_cont[:, 0:108] += 0.25
    with torch.no_grad():
        pred_clean = model(batch, mask=mask)
        pred_sem = model(batch_sem, mask=mask)
    return {
        "sem_perturbation_prediction_diff": float((pred_clean - pred_sem).abs().max()),
        "passed": bool(float((pred_clean - pred_sem).abs().max()) > 0.0),
    }


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
    init_activity = _interface_activity(model, gradient_batch, cm.C6_MASK)
    result = cssd.train_cssd(
        tag="SEM108-smoke",
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
                CHECKPOINT_DIR / "SEM108-smoke_final_state.pt", map_location="cpu", weights_only=False
            ).items()
        }
    )
    post_grads = _gradient_diagnostics(final_model, gradient_batch, cm.C6_MASK, cm.H1_LAMBDA)
    post_activity = _interface_activity(final_model, gradient_batch, cm.C6_MASK)
    finite = bool(
        math.isfinite(result["best_valid_mae"])
        and math.isfinite(result["final_valid_mae"])
        and all(math.isfinite(row["train_mae"]) for row in result["curve"])
    )
    gradient_keys = ("grad_D", "grad_fusion_W1", "grad_fusion_W2", "grad_W_A_S", "grad_W_E_S")
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
        "optimizer_step_budget": 100,
        "finite": finite,
        "gradient_ok": gradient_ok,
        "init_gradients": init_grads,
        "post_gradients": post_grads,
        "init_interface_activity": init_activity,
        "post_interface_activity": post_activity,
        "best_valid_mae": float(result["best_valid_mae"]),
        "final_valid_mae": float(result["final_valid_mae"]),
        "train_curve": result["curve"],
        "passed": bool(finite and gradient_ok and init_activity["passed"] and post_activity["passed"]),
    }
    sem.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "smoke.json", payload)
    print(
        f"[smoke] steps={steps} finite={finite} gradient_ok={gradient_ok} "
        f"init_grad_D={init_grads['grad_D']:.3e} best={result['best_valid_mae']:.6f}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError("smoke failed (non-finite loss, dead gradients or inactive interface)")
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
    model: sem.SEM108Model, batch: Any, initial_dictionary: torch.Tensor
) -> dict[str, Any]:
    grads = _gradient_diagnostics(model, batch, cm.C6_MASK, cm.H1_LAMBDA)
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        usage = _usage_from_coord(coord, model.common_dim)
        movement = float((model.D.detach().cpu() - initial_dictionary).norm())
    return {
        **grads,
        "dictionary": usage,
        "dictionary_movement_frobenius": movement,
    }


class _ResidualDictionaryView(nn.Module):
    """Adapter exposing only the 32 residual roles to ``cm.dictionary_diagnostics``."""

    def __init__(self, model: sem.SEM108Model) -> None:
        super().__init__()
        self.model = model

    @property
    def D(self) -> torch.nn.Parameter:
        return self.model.D

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        return self.model.code(phi)[:, self.model.common_dim :]

    def reconstruct(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        return self.model.reconstruct(phi, self.model.code(phi))

    def reconstruction_loss(self, phi: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        return self.model.reconstruction_loss(phi, self.model.code(phi))

    def forward(self, data: Any, **kwargs: Any):
        return self.model(data, **kwargs)


def _dictionary_health(
    model: sem.SEM108Model,
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
        _ResidualDictionaryView(model), loader, device, dictionary, gradient_batch=gradient_batch
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
            _write_train_artifacts(
                result, _dictionary_tensor(), p1run.load_split("valid")
            )
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
    sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    subspace = _load_parent_subspace()
    dictionary = _dictionary_tensor()
    initial_dictionary = torch.as_tensor(np.asarray(dictionary, dtype=np.float32)).clone()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    gradient_batch = next(iter(p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)))
    diagnostics: dict[str, Any] = {}
    curve_ref: list[list[dict[str, Any]]] = [[]]
    started = time.perf_counter()

    def callback(epoch: int, model: sem.SEM108Model, optimizer: Any, curve: list[dict[str, Any]]):
        curve_ref[0] = curve
        if epoch in DIAG_EPOCHS:
            row = model_diag_row(model, gradient_batch, initial_dictionary)
            row["epoch"] = int(epoch)
            row["valid_mae"] = float(curve[epoch - 1]["valid_mae"])
            row["train_mae"] = float(curve[epoch - 1]["train_mae"])
            diagnostics[str(epoch)] = row
            print(
                f"[diag@{epoch}] valid={row['valid_mae']:.6f} Dgrad={row['grad_D']:.3e} "
                f"fusion={row['grad_fusion_W1']:.2e} active={row['dictionary']['active_atoms']}",
                flush=True,
            )
        return True, None

    result = cssd.train_cssd(
        tag="SEM108-seed0",
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
    sem.official_test_blocker(result)
    _write_json(final_path, result)
    _write_train_artifacts(result, dictionary, valid_data)
    print(
        f"[train] epochs={result['epochs_run']} best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# stage: frozen mechanism probes
# ---------------------------------------------------------------------------


def _evaluate(model: sem.SEM108Model, valid_data: Sequence[Any], mask: Any) -> float:
    device = audit.attach_cpu(THREADS)
    loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    return float(audit.evaluate_mask(model, loader, device, mask)["mae"])


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = MECHANISM_DIR / "dictionary_health.json"
    if out.exists() and not force:
        print("[interventions] cache hit", flush=True)
        return {"cached": True}
    sem.cpu_only_guard(torch.device("cpu"))
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
    m_sem0 = _evaluate(model, valid_data, sem.merge_sem_mask(base_mask, sem.SEMMask(sem_block_zero=True)))
    m_atom0 = _evaluate(model, valid_data, sem.merge_sem_mask(base_mask, sem.SEMMask(sem_atom_zero=True)))
    m_bond0 = _evaluate(model, valid_data, sem.merge_sem_mask(base_mask, sem.SEMMask(sem_bond_zero=True)))
    model.eval()
    with torch.no_grad():
        residual_zero = sem.SEMMask(residual_zero=True)
        m_dict0 = _evaluate(model, valid_data, sem.merge_sem_mask(base_mask, residual_zero))

    _write_json(
        MECHANISM_DIR / "sem_meanfill.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "M_S_recomputed": baseline,
            "M_sem0": m_sem0,
            "G_sem0": float(m_sem0 - baseline),
            "note": "standardized zero == train-mean neutralization; size2 retained",
        },
    )
    _write_json(
        MECHANISM_DIR / "atom_block.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_S": baseline,
            "M_atom0": m_atom0,
            "G_atom": float(m_atom0 - baseline),
        },
    )
    _write_json(
        MECHANISM_DIR / "bond_block.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_S": baseline,
            "M_bond0": m_bond0,
            "G_bond": float(m_bond0 - baseline),
        },
    )
    _write_json(
        MECHANISM_DIR / "dict_zero.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "M_S": baseline,
            "M_dict0": m_dict0,
            "G_dict0": float(m_dict0 - baseline),
            "diagnostic_only": True,
        },
    )

    for kind, filename in (
        ("node", "node_assignment_shuffle.json"),
        ("edge", "edge_assignment_shuffle.json"),
    ):
        rows: list[dict[str, Any]] = []
        probe_mask = sem.merge_sem_mask(
            base_mask,
            sem.SEMMask(use_node_shuffle=(kind == "node"), use_edge_shuffle=(kind == "edge")),
        )
        for shuffle_seed in SHUFFLE_SEEDS:
            audit.prepare_shuffles(valid_data, kind, int(shuffle_seed))
            try:
                mae = _evaluate(model, valid_data, probe_mask)
            finally:
                audit.clear_shuffles(valid_data)
            rows.append({"seed": int(shuffle_seed), "mae": float(mae), "delta_vs_M_S": float(mae - baseline)})
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
            },
        )

    rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        probe_mask = sem.merge_sem_mask(
            base_mask,
            sem.SEMMask(use_sem_row_shuffle=True, sem_shuffle_seed=int(shuffle_seed)),
        )
        mae = _evaluate(model, valid_data, probe_mask)
        rows.append(
            {"seed": int(shuffle_seed), "mae": float(mae), "delta_vs_M_S": float(mae - baseline)}
        )
    deltas = [row["delta_vs_M_S"] for row in rows]
    _write_json(
        MECHANISM_DIR / "sem_root_shuffle.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "kind": "sem_root_row_shuffle",
            "seeds": list(SHUFFLE_SEEDS),
            "M_S": baseline,
            "rows": rows,
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "note": "whole Sem108 row permuted across roots within each molecule; size2 retained per root",
        },
    )

    model.eval()
    gradient_batch = next(iter(p1.make_env_loader(valid_data, int(p2run.BATCH_SIZE), False, 0)))
    _write_json(
        MECHANISM_DIR / "dictionary_health.json",
        _dictionary_health(model, dictionary, valid_data, gradient_batch=gradient_batch),
    )
    print(
        f"[interventions] M_S={baseline:.6f} M_sem0={m_sem0:.6f} G_sem0={m_sem0 - baseline:+.6f} "
        f"G_dict0={m_dict0 - baseline:+.6f}",
        flush=True,
    )
    return {"M_S": baseline, "G_sem0": float(m_sem0 - baseline), "G_dict0": float(m_dict0 - baseline)}


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
    references = _read_json(RESULTS_DIR / "historical_references.json")
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    audit_decision = _read_json(AUDIT_DIR / "audit_decision.json")
    t1_audit = _read_json(AUDIT_DIR / "t1_block_ablation.json")
    sem_meanfill = _read_json(MECHANISM_DIR / "sem_meanfill.json")
    atom_block = _read_json(MECHANISM_DIR / "atom_block.json")
    bond_block = _read_json(MECHANISM_DIR / "bond_block.json")
    dict_zero = _read_json(MECHANISM_DIR / "dict_zero.json")
    sem_shuffle = _read_json(MECHANISM_DIR / "sem_root_shuffle.json")
    node_shuffle = _read_json(MECHANISM_DIR / "node_assignment_shuffle.json")
    edge_shuffle = _read_json(MECHANISM_DIR / "edge_assignment_shuffle.json")
    health = _read_json(MECHANISM_DIR / "dictionary_health.json")

    m_s = float(run["soup"]["soup_valid_mae"])
    band = _band(m_s)
    g_sem0 = float(sem_meanfill["G_sem0"])
    g_atom = float(atom_block["G_atom"])
    g_bond = float(bond_block["G_bond"])
    g_dict0 = float(dict_zero["G_dict0"])
    g_sem_shuffle = float(sem_shuffle["mean_delta"])
    g_node = float(node_shuffle["mean_delta"])
    g_edge = float(edge_shuffle["mean_delta"])
    g_corr = max(g_node, g_edge)
    health_pass = bool(
        int(health["active_atoms"]) >= 24
        and float(health["effective_atoms"]) >= 8
        and float(health["usage_top1_share"]) <= 0.75
    )
    sem_use = (
        "strong"
        if g_sem_shuffle >= GATE_SEM_SHUFFLE_STRONG
        else ("load_bearing" if g_sem_shuffle >= GATE_SEM_SHUFFLE_LOAD_BEARING else ("weak" if g_sem_shuffle < GATE_SEM_SHUFFLE_WEAK else "marginal"))
    )
    dict_corr = (
        "clear_incremental"
        if g_corr >= GATE_DICT_CORR_CLEAR
        else ("directional" if g_corr >= GATE_DICT_CORR_DIRECTIONAL else "not_established")
    )
    hist = float(references["references"]["CSSD-q1-seed0"]["value"])
    g_hist = hist - m_s
    case_boundary_note: str | None = None

    if m_s <= 0.120 and g_sem_shuffle >= GATE_SEM_SHUFFLE_LOAD_BEARING and g_corr >= GATE_DICT_CORR_CLEAR and health_pass:
        case, verdict = "A", "CSSD_SEM108_STRONG_SINGLE_SEED_SUPPORTED"
    elif 0.120 < m_s <= 0.123 and g_sem_shuffle >= GATE_SEM_SHUFFLE_LOAD_BEARING and g_corr >= GATE_DICT_CORR_CLEAR:
        case, verdict = "B", "CSSD_SEM108_PROMISING_SINGLE_SEED"
    elif m_s <= 0.123 and g_sem_shuffle >= GATE_SEM_SHUFFLE_LOAD_BEARING and g_corr < GATE_DICT_CORR_DIRECTIONAL:
        case, verdict = "C", "SEM108_BACKBONE_SIGNAL_DICTIONARY_INCREMENT_NOT_ESTABLISHED"
    elif 0.123 < m_s <= 0.127 and g_corr >= GATE_DICT_CORR_CLEAR:
        case, verdict = "D", "DICTIONARY_INCREMENT_RETAINED_BUT_NO_NEW_TASK_BAND"
    elif m_s > 0.127:
        case, verdict = "E", "CSSD_SEM108_NO_GO"
    else:
        case, verdict = "BOUNDARY", "CSSD_SEM108_TASK_SIGNAL_WITH_DIRECTIONAL_DICTIONARY_INCREMENT"
        case_boundary_note = (
            "Boundary outcome not enumerated in the pre-registered decision table: "
            f"M_S={m_s:.6f}, G_sem_shuffle={g_sem_shuffle:.6f}, G_corr={g_corr:.6f}. "
            "The task band and the semantic-use gate pass; the dictionary-correspondence "
            "gate is directional (0.003 <= G_corr < 0.010), so no strong claim is made."
        )
    if m_s > MATERIAL_REGRESSION:
        case_boundary_note = (case_boundary_note or "") + " MATERIAL_REGRESSION (M_S > 0.135)."

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
            "delta": int(preflight["parameter_audit"]["delta"]),
            "relative_delta": float(preflight["parameter_audit"]["relative_delta"]),
            "sem_fusion_hidden": int(preflight["parameter_audit"]["sem_fusion_hidden"]),
        },
        "audit": {
            "decision": audit_decision["decision"],
            "t1_status": t1_audit["status"],
        },
        "M_S": m_s,
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": list(run["soup"]["members"]),
        "performance_band": band,
        "historical": {
            "CSSD_q1_seed0_soup": hist,
            "G_hist_unmatched": float(g_hist),
            "T1_tuned_soup": float(references["references"]["T1-tuned-seed0"]["value"]),
            "RNDB_soup": float(references["references"]["RNDB-seed0"]["value"]),
        },
        "gates": {
            "correctness_all_passed": bool(correctness["all_passed"]),
            "smoke_passed": bool(smoke["passed"]),
            "parameter_ratio_within_bound": bool(
                preflight["parameter_audit"]["parameter_ratio_within_bound"]
            ),
            "dictionary_health_pass": health_pass,
        },
        "semantic_mechanism": {
            "M_sem0": float(sem_meanfill["M_sem0"]),
            "G_sem0": g_sem0,
            "M_atom0": float(atom_block["M_atom0"]),
            "G_atom": g_atom,
            "M_bond0": float(bond_block["M_bond0"]),
            "G_bond": g_bond,
            "G_sem_shuffle": g_sem_shuffle,
            "G_sem_shuffle_min": float(sem_shuffle["min_delta"]),
            "G_sem_shuffle_max": float(sem_shuffle["max_delta"]),
            "sem_shuffle_rows": sem_shuffle["rows"],
            "sem_use": sem_use,
        },
        "dictionary_mechanism": {
            "M_dict0": float(dict_zero["M_dict0"]),
            "G_dict0": g_dict0,
            "G_node": g_node,
            "G_edge": g_edge,
            "G_corr": float(g_corr),
            "dict_corr": dict_corr,
            "node_rows": node_shuffle["rows"],
            "edge_rows": edge_shuffle["rows"],
        },
        "dictionary_health": health,
        "case": case,
        "verdict": verdict,
        "case_boundary_note": case_boundary_note,
        "historical_unmatched_only": True,
        "matched_baseline_rerun": False,
        "mechanism_probes_inference_only": True,
    }
    sem.official_test_blocker(summary)
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, references, run, health, t1_audit)
    _write_decision(summary)
    print(f"[analysis] M_S={m_s:.6f} band={band} case={case} verdict={verdict}", flush=True)
    return summary


def _write_report(
    summary: Mapping[str, Any],
    references: Mapping[str, Any],
    run: Mapping[str, Any],
    health: Mapping[str, Any],
    t1_audit: Mapping[str, Any],
) -> None:
    s = summary
    lines: list[str] = []
    lines.append("# e2e_dictenv_sem108_v1 — CSSD-Sem108 (Shell-Resolved Primitive Semantic Interface)")
    lines.append("")
    lines.append("CPU only · official test never loaded · single seed-0 trajectory.")
    lines.append("")
    lines.append("## A. Provenance")
    lines.append("")
    lines.append(f"- formal commit `{s['git_commit']}`")
    lines.append(f"- device `cpu` · threads `{s['threads']}`")
    lines.append("- `official_test_loaded = false`")
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
    lines.append("parent:  anchor62 -> anchor_encoder(62->32->32) -> concat with 3x48 node / 6x32 edge -> fusion(368->128->48)")
    lines.append("Sem108:  patch_cont[:, 0:108] = atom_shell(3x28=84) | bond_shell(6x4=24)")
    lines.append("candidate: [Sem108 ; size2](110) -> concat with 3x48 node / 6x32 edge -> fusion(446->114->48)")
    lines.append("no RNDB, no message passing, no new fine structural descriptor")
    lines.append("```")
    lines.append("")
    lines.append("## D. Parameter budget")
    lines.append("")
    p = s["parameters"]
    lines.append(
        f"- parent `{p['parent']}`, candidate `{p['candidate']}` (delta `{p['delta']}`), "
        f"relative `{p['relative_delta']:.5f}` · fusion hidden `{p['sem_fusion_hidden']}`"
    )
    lines.append("")
    lines.append("## E. Zero-training prior audit")
    lines.append("")
    lines.append(f"- decision `{s['audit']['decision']}`")
    lines.append(f"- T1 block audit: `{s['audit']['t1_status']}`")
    lines.append("")
    lines.append("## F. Training")
    lines.append("")
    lines.append(
        f"- epochs `{s['epochs']}` · wall `{_fmt(s['wall_clock_s'], 1)} s` · seed `{s['seed']}`"
    )
    lines.append(
        f"- best valid `{_fmt(s['best_valid_mae'], 6)}` @ {s['best_epoch']} · Top-5 members "
        f"{s['soup_members']} · soup `{_fmt(s['M_S'], 6)}`"
    )
    lines.append("")
    lines.append("## G. Performance interpretation")
    lines.append("")
    lines.append(f"- `M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**")
    lines.append(
        f"- `G_hist = {_fmt(s['historical']['G_hist_unmatched'], 6)}` vs historical CSSD-q1 "
        f"`{_fmt(s['historical']['CSSD_q1_seed0_soup'], 6)}` — historical, unmatched"
    )
    lines.append("")
    lines.append("## H. Semantic mechanism")
    lines.append("")
    sm = s["semantic_mechanism"]
    lines.append(
        f"- mean-fill Sem108: `M_sem0={_fmt(sm['M_sem0'], 6)}` `G_sem0={_fmt(sm['G_sem0'], 6)}`"
    )
    lines.append(
        f"- atom block: `G_atom={_fmt(sm['G_atom'], 6)}` · bond block: `G_bond={_fmt(sm['G_bond'], 6)}`"
    )
    lines.append(
        f"- Sem108 row shuffle: `G_sem_shuffle={_fmt(sm['G_sem_shuffle'], 6)}` "
        f"[{_fmt(sm['G_sem_shuffle_min'], 6)}, {_fmt(sm['G_sem_shuffle_max'], 6)}] → `{sm['sem_use']}`"
    )
    lines.append("")
    lines.append("## I. Dictionary mechanism")
    lines.append("")
    dm = s["dictionary_mechanism"]
    lines.append(f"- zero-alpha diagnostic: `M_dict0={_fmt(dm['M_dict0'], 6)}` `G_dict0={_fmt(dm['G_dict0'], 6)}`")
    lines.append(
        f"- node correspondence `G_node={_fmt(dm['G_node'], 6)}` · edge `G_edge={_fmt(dm['G_edge'], 6)}` "
        f"· `G_corr={_fmt(dm['G_corr'], 6)}` → `{dm['dict_corr']}`"
    )
    lines.append(
        f"- health: active {health['active_atoms']}/32, effective {_fmt(health['effective_atoms'], 2)}, "
        f"top1 share {_fmt(health['usage_top1_share'], 3)}, recon {_fmt(health['valid_reconstruction_relative'], 5)}"
    )
    lines.append("")
    lines.append("## J. Verdict")
    lines.append("")
    lines.append(f"**Case {s['case']} — {s['verdict']}**")
    lines.append("")
    if s.get("case_boundary_note"):
        lines.append(f"> {s['case_boundary_note']}")
        lines.append("")
    lines.append("## K. T1 attribution limitation")
    lines.append("")
    lines.append(f"- `{t1_audit['status']}`: {t1_audit['reason']}")
    lines.append("")
    REPORT_PATH = RESULTS_DIR / "REPORT.md"
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    s = summary
    lines = [
        "# DECISION — e2e_dictenv_sem108_v1",
        "",
        f"`M_S = {_fmt(s['M_S'], 6)}` → band **{s['performance_band']}**.",
        "",
        f"`G_sem_shuffle = {_fmt(s['semantic_mechanism']['G_sem_shuffle'], 6)}` "
        f"(`{s['semantic_mechanism']['sem_use']}`), "
        f"`G_corr = {_fmt(s['dictionary_mechanism']['G_corr'], 6)}` "
        f"(`{s['dictionary_mechanism']['dict_corr']}`).",
        "",
        f"**Case {s['case']} — {s['verdict']}**",
        "",
    ]
    if s.get("case_boundary_note"):
        lines.extend([s["case_boundary_note"], ""])
    lines.extend(
        [
            "Single seed-0 trajectory, no matched baseline rerun, no official-test read.",
            "Next-step authorization is recorded in the analysis note; no rescue run was executed.",
            "",
        ]
    )
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
    sem.cpu_only_guard(torch.device("cpu"))

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
