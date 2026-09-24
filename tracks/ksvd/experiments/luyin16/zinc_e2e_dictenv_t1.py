"""E2E-DictEnv-T1 runner — interface / lambda / horizon tuning.

Round ``e2e_dictenv_t1``.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_t1_preregistration.md`` (frozen).
Prior-artifact audit:
``tracks/ksvd/notes/e2e_dictenv_t1_prior_artifact_audit.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_t1.py``.

Stages
------
``identity correct stage_a select stage_b stage_c finalize
 freeze confirm health mechanism analyze unlock test all``

* ``stage_a``  — 3 registered interface candidates (Sparse, seed 0, lambda_0).
* ``select``   — Stage-A winner written to ``stage_a_selection.json``.
* ``stage_b``  — two lambda arms (0.5x, 0.25x of lambda_0) on the winner.
* ``stage_c``  — conditional 320-epoch run (only if the trigger fires).
* ``finalize`` — ``tuning_decision.json``.
* ``freeze``   — ``architecture_freeze.json`` (before any test read).
* ``confirm``  — seed-1 Sparse + DenseTied on the frozen config.
* ``health``   — dictionary health for the final Sparse soup states.
* ``mechanism``— zero-code + assignment shuffle on the final Sparse soup.
* ``analyze``  — frozen verdict / report / decision.
* ``unlock``   — write ``official_test_unlock.json`` and load the test once.

Official ZINC **test is never loaded** before ``unlock``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_t1 as t1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e_v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import (
    OPTIMIZED_PROTOCOL,
    REPO_ROOT,
)

PROTOCOL_VERSION = t1.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_t1"
STATE_DIR = RESULTS_DIR / "states"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

MODULE_PATH = Path(t1.__file__)

MAX_EPOCHS = int(OPTIMIZED_PROTOCOL["max_epochs"])  # 240
HORIZON_EXTENDED = 320
BATCH_SIZE = int(OPTIMIZED_PROTOCOL["batch_size"])  # 128
LEARNING_RATE = float(OPTIMIZED_PROTOCOL["learning_rate"])
WEIGHT_DECAY = float(OPTIMIZED_PROTOCOL["weight_decay"])
GRAD_CLIP = float(OPTIMIZED_PROTOCOL["gradient_clip_norm"])
TRAIN_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"])
EVAL_SHUFFLE_OFFSET = int(OPTIMIZED_PROTOCOL["eval_shuffle_seed_offset"])

SHUFFLE_SEEDS = (101, 202, 303, 404, 505)

V0_SOUP = 0.14550823224702616  # durable v0 SparseDictEnv soup (reference A0)
FEC_S1_SOUP = 0.13042183499777457
FEC_S1_BEST = 0.1367825070246472

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_seed_everything = v0run._seed_everything
_git_commit = v0run._git_commit


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _module_sha256() -> str:
    return _sha256_text(MODULE_PATH.read_text(encoding="utf-8"))


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        digest.update(key.encode("utf-8"))
        tensor = state[key].detach().cpu().contiguous()
        digest.update(np.asarray(tensor, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _n_params(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _cand(key: str) -> t1.Candidate:
    return t1.CANDIDATES[str(key)]


def _candidate_by_id(candidate_id: str) -> str:
    for key, candidate in t1.CANDIDATES.items():
        if candidate.candidate_id == candidate_id:
            return key
    raise KeyError(candidate_id)


def _v0_json(name: str) -> dict[str, Any] | None:
    path = TRACK_ROOT / "results/e2e_dictenv_v0" / name
    return _read_json(path) if path.exists() else None


# ---------------------------------------------------------------------------
# stage: identity / parameter audit
# ---------------------------------------------------------------------------


def identity_stage() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "module_sha256": _module_sha256(),
        "official_test_loaded": False,
        "lambda_v0": t1.LAMBDA_V0,
        "lambda_scales": list(t1.LAMBDA_SCALES),
        "candidates": {},
    }
    for key in t1.CANDIDATE_ORDER:
        candidate = _cand(key)
        sparse = t1.build_model(t1.SPARSE_ARM, candidate, seed=0)
        dense = t1.build_model(
            t1.DENSE_ARM, candidate, seed=0,
            reference_state={k: v.detach().clone() for k, v in sparse.state_dict().items()},
        )
        accounting = t1.total_parameter_count(candidate)
        local = t1.local_parameter_count(candidate)
        payload["candidates"][key] = {
            "candidate_id": candidate.candidate_id,
            "r_dict": int(candidate.r_dict),
            "mode": candidate.mode,
            "decoder_hidden": int(candidate.decoder_hidden),
            "decoder_input": int(candidate.decoder_input),
            "local": local,
            "accounting": accounting,
            "sparse_params": _n_params(sparse),
            "dense_params": _n_params(dense),
            "parameter_identity": bool(
                _n_params(sparse) == _n_params(dense) == int(accounting["whole_model"])
            ),
            "D_sha256": hashlib.sha256(
                np.asarray(sparse.D.detach().cpu(), dtype=np.float32).tobytes()
            ).hexdigest(),
        }
    _write_json(RESULTS_DIR / "parameter_audit.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage: correctness gates
# ---------------------------------------------------------------------------


def _g0b_coarse146_identity(n_molecules: int = 4) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import (
        FactorizedFeatureTransform,
        build_fits,
        factorized_record,
    )
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
    from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
        _extract_v4_records,
    )
    from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp

    train_records, valid_records = _extract_v4_records()
    fits = build_fits(train_records)
    transform = FactorizedFeatureTransform(fits)
    valid_raw = list(_load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    records = list(valid_records)[: int(n_molecules)]
    _train_encoded, valid_encoded, _audit = sdp.load_encoded(valid_subset=int(n_molecules))
    exact = 0
    max_abs = 0.0
    n_patches = 0
    for raw, record, encoded in zip(valid_raw, records, list(valid_encoded)):
        fact = factorized_record(raw, tokenize=True)
        y = float(raw.y.view(-1)[0])
        data = transform.build(raw, y, topology_raw=record.topology_features, raw=fact)
        reference = encoded.patch_cont
        candidate = data.patch_cont
        n_patches += int(reference.shape[0])
        if torch.equal(reference, candidate):
            exact += 1
        max_abs = max(max_abs, float((reference - candidate).abs().max().item()))
    return {
        "n_molecules": int(len(valid_raw)),
        "n_patches": int(n_patches),
        "molecules_bit_identical": int(exact),
        "max_abs_diff": float(max_abs),
        "width": int(t1.COARSE_DIM),
        "source": "fec_s0_factorization.FactorizedFeatureTransform (train-fit scaler)",
        "passed": bool(exact == len(valid_raw)),
    }


def _g1_chemistry_purity(n_molecules: int = 2) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e

    encoded = v0run.load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    sparse = t1.build_model(t1.SPARSE_ARM, _cand("a1"), seed=0).eval()
    dense = t1.build_model(t1.DENSE_ARM, _cand("a1"), seed=0).eval()
    max_phi = max_alpha = max_z = 0.0
    incidence_identical = 0
    for data, raw_molecule in zip(encoded, raw):
        graph, _nt, edge_types = zlr._data_to_graph(raw_molecule)
        incidence_a = e2e.env_incidence(graph, edge_types)
        relabelled = {key: (value + 1) % t1.ATOM_CATEGORIES for key, value in edge_types.items()}
        incidence_b = e2e.env_incidence(graph, relabelled)
        if all(torch.equal(incidence_a[k], incidence_b[k]) for k in ("occ_node", "occ_root", "occ_shell")):
            incidence_identical += 1
        phi_a = r2.build_phi(graph).astype(np.float32)
        phi_b = r2.build_phi(graph).astype(np.float32)
        with torch.no_grad():
            max_alpha = max(max_alpha, float((sparse.code(torch.as_tensor(phi_a)) - sparse.code(torch.as_tensor(phi_b))).abs().max()))
            max_z = max(max_z, float((dense.code(torch.as_tensor(phi_a)) - dense.code(torch.as_tensor(phi_b))).abs().max()))
        max_phi = max(max_phi, float(np.abs(phi_a - phi_b).max()))
    return {
        "n_molecules": int(n_molecules),
        "incidence_topology_only": int(incidence_identical),
        "phi_max_abs_diff": float(max_phi),
        "sparse_alpha_max_abs_diff": float(max_alpha),
        "dense_z_max_abs_diff": float(max_z),
        "passed": bool(incidence_identical == int(n_molecules) and max_phi == 0.0 and max_alpha == 0.0 and max_z == 0.0),
    }


def _g2_exact_sparsity(n_molecules: int = 32) -> dict[str, Any]:
    encoded = v0run.load_split("valid", subset=int(n_molecules))
    model = t1.build_model(t1.SPARSE_ARM, _cand("a1"), seed=0).eval()
    phi = torch.cat([data.dict_phi for data in encoded], dim=0)
    with torch.no_grad():
        alpha = model.code(phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    exact = float((l0 == int(t1.SPARSITY)).float().mean())
    return {
        "n_nodes": int(alpha.shape[0]),
        "max_l0": int(l0.max()),
        "exact_top_s_fraction": float(exact),
        "s": int(t1.SPARSITY),
        "passed": bool(int(l0.max()) <= int(t1.SPARSITY) and exact >= 0.99),
    }


def _g3_gradient_to_dictionary(candidate_key: str, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = v0run.load_split("train", subset=32)
    loader = v0run.e2e.make_env_loader(encoded, 32, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    _pred, aux = model(batch, return_aux=True)
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    rec.backward()
    rec_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    return {
        "grad_D_from_task_alone": float(task_grad),
        "grad_D_from_reconstruction": float(rec_grad),
        "passed": bool(math.isfinite(task_grad) and task_grad > 0.0 and rec_grad > 0.0),
    }


def _g4_tied_reconstruction(candidate_key: str, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = v0run.load_split("valid", subset=16)
    loader = v0run.e2e.make_env_loader(encoded, 16, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj).eval()
    phi = batch.dict_phi
    with torch.no_grad():
        a = model.code(phi)
        ra = model.reconstruct(phi, a)
        model.D.add_(0.01 * torch.randn_like(model.D))
        b = model.code(phi)
        rb = model.reconstruct(phi, b)
    keys = list(model.state_dict().keys())
    forbidden = [
        key for key in keys
        if ("encoder" in key and not any(tag in key for tag in ("relation", "pair", "global", "topology")))
    ]
    return {
        "code_changed": bool(not torch.equal(a, b)),
        "reconstruction_changed": bool(not torch.equal(ra, rb)),
        "forbidden_encoder_keys": forbidden,
        "passed": bool(not torch.equal(a, b) and not torch.equal(ra, rb) and not forbidden),
    }


def _g5_no_dense_bypass(candidate_key: str, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = v0run.load_split("valid", subset=16)
    loader = v0run.e2e.make_env_loader(encoded, 16, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj).eval()
    with torch.no_grad():
        _p, aux_a = model(batch, coord_zero=True, return_aux=True)
    mutated = batch.clone()
    mutated.dict_phi = torch.randn_like(mutated.dict_phi)
    with torch.no_grad():
        _p2, aux_b = model(mutated, coord_zero=True, return_aux=True)
        _p3, aux_c = model(mutated, coord_zero=False, return_aux=True)
    return {
        "zero_code_E_identical_under_phi_mutation": bool(torch.equal(aux_a["E"], aux_b["E"])),
        "active_code_changes_E": bool(not torch.equal(aux_b["E"], aux_c["E"])),
        "passed": bool(torch.equal(aux_a["E"], aux_b["E"]) and not torch.equal(aux_b["E"], aux_c["E"])),
    }


def _g6_environment_freeze(candidate_key: str, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = v0run.load_split("valid", subset=8)
    loader = v0run.e2e.make_env_loader(encoded, 8, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj).eval()
    captured: dict[str, torch.Tensor] = {}
    handle = model.env_mlp.register_forward_hook(lambda _m, _i, out: captured.__setitem__("E", out.detach().clone()))
    mutated = batch.clone()
    mutated.pair_relation = torch.randn_like(mutated.pair_relation)
    try:
        with torch.no_grad():
            model(batch)
            before = captured["E"].clone()
            model(mutated)
            after = captured["E"].clone()
    finally:
        handle.remove()
    return {
        "E_bit_identical_under_pair_mutation": bool(torch.equal(before, after)),
        "passed": bool(torch.equal(before, after)),
    }


def _g7_no_pair_to_centre(candidate_key: str, device: str = "cpu") -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    device_obj = torch.device(device)
    encoded = v0run.load_split("valid", subset=8)
    loader = v0run.e2e.make_env_loader(encoded, 8, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj).eval()
    original = zpp.PatchPathModel._pool_pairs_to_centres
    calls = {"count": 0}

    def _guard(*_a, **_k):
        calls["count"] += 1
        raise RuntimeError("pair->centre must not exist in E2E-DictEnv-T1")

    zpp.PatchPathModel._pool_pairs_to_centres = _guard
    try:
        with torch.no_grad():
            model(batch)
        forward_ok = True
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    return {"forward_ok": bool(forward_ok), "pair_to_centre_calls": int(calls["count"]), "passed": bool(forward_ok and calls["count"] == 0)}


def _g8_once_only(candidate_key: str, device: str = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    encoded = v0run.load_split("valid", subset=8)
    loader = v0run.e2e.make_env_loader(encoded, 8, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj).eval()
    counts = {"pair_encoder": 0, "relation_encoder": 0, "env_mlp": 0}

    def _mk(name):
        def _hook(_m, _i, _o):
            counts[name] += 1
        return _hook

    handles = [
        model.pair_encoder.register_forward_hook(_mk("pair_encoder")),
        model.relation_encoder.register_forward_hook(_mk("relation_encoder")),
        model.env_mlp.register_forward_hook(_mk("env_mlp")),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return {"calls": counts, "passed": bool(counts["pair_encoder"] == 1 and counts["relation_encoder"] == 1 and counts["env_mlp"] == 1)}


def _g9_relabel_invariance(candidate_key: str, n_molecules: int = 2, device: str = "cpu") -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import _relabel_raw

    device_obj = torch.device(device)
    encoded = v0run.load_split("valid", subset=int(n_molecules))
    raw = list(zlr._load_zinc(ZINC_ROOT, "val"))[: int(n_molecules)]
    model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0).to(device_obj).eval()
    max_diff = 0.0
    for data, raw_molecule in zip(encoded, raw):
        n = int(data.num_nodes)
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(20260924))
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(n)
        incidence_new = e2e.env_incidence(*_graph_and_types(_relabel_raw(raw_molecule)))
        relabelled = data.clone()
        relabelled.dict_phi = data.dict_phi[permutation]
        relabelled.dict_atom = data.dict_atom[permutation]
        relabelled.patch_cont = data.patch_cont[permutation]
        relabelled.env_occ_node = incidence_new["occ_node"]
        relabelled.env_occ_root = incidence_new["occ_root"]
        relabelled.env_occ_shell = incidence_new["occ_shell"]
        relabelled.pair_index = inverse[data.pair_index]
        with torch.no_grad():
            p0 = model(e2e.env_collate([data]).to(device_obj)).view(-1)
            p1 = model(e2e.env_collate([relabelled]).to(device_obj)).view(-1)
        max_diff = max(max_diff, float((p0 - p1).abs().max()))
    return {"n_molecules": int(n_molecules), "max_abs_pred_diff": float(max_diff), "tolerance": 1.0e-5, "passed": bool(max_diff <= 1.0e-5)}


def _graph_and_types(raw_molecule):
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr

    graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
    return graph, edge_types


def _g10_parameter_identity(candidate_key: str) -> dict[str, Any]:
    candidate = _cand(candidate_key)
    sparse = t1.build_model(t1.SPARSE_ARM, candidate, seed=0)
    dense = t1.build_model(
        t1.DENSE_ARM, candidate, seed=0,
        reference_state={k: v.detach().clone() for k, v in sparse.state_dict().items()},
    )
    accounting = t1.total_parameter_count(candidate)
    return {
        "candidate_id": candidate.candidate_id,
        "sparse_params": _n_params(sparse),
        "dense_params": _n_params(dense),
        "accounted_total": int(accounting["whole_model"]),
        "local": t1.local_parameter_count(candidate),
        "passed": bool(_n_params(sparse) == _n_params(dense) == int(accounting["whole_model"])),
    }


def _g11_init_matching(candidate_key: str) -> dict[str, Any]:
    candidate = _cand(candidate_key)
    sparse = t1.build_model(t1.SPARSE_ARM, candidate, seed=0)
    dense = t1.build_model(
        t1.DENSE_ARM, candidate, seed=0,
        reference_state={k: v.detach().clone() for k, v in sparse.state_dict().items()},
    )
    ss, ds = sparse.state_dict(), dense.state_dict()
    mismatched = [k for k in ss if not torch.equal(ss[k], ds[k])]
    return {
        "n_tensors": int(len(ss)),
        "mismatched_keys": mismatched,
        "D_bit_identical": bool(torch.equal(ss["D"], ds["D"])),
        "D_sha256": _state_hash({"D": ss["D"]}),
        "passed": bool(not mismatched and torch.equal(ss["D"], ds["D"])),
    }


def correctness_stage(device: str = "cpu") -> dict[str, Any]:
    gates: dict[str, Any] = {
        "G0a_phi65_identity": v0run._g0_phi_identity(3),
        "G0b_coarse146_identity": _g0b_coarse146_identity(4),
        "G1_chemistry_purity": _g1_chemistry_purity(2),
        "G2_exact_sparsity": _g2_exact_sparsity(32),
        "G12_official_test_blocker": v0run._g12_official_test_blocker(),
    }
    for key in t1.CANDIDATE_ORDER:
        gates[f"G3_gradient_to_dictionary_{key}"] = _g3_gradient_to_dictionary(key, device)
        gates[f"G4_tied_reconstruction_{key}"] = _g4_tied_reconstruction(key, device)
        gates[f"G5_no_dense_bypass_{key}"] = _g5_no_dense_bypass(key, device)
        gates[f"G6_environment_freeze_{key}"] = _g6_environment_freeze(key, device)
        gates[f"G7_no_pair_to_centre_{key}"] = _g7_no_pair_to_centre(key, device)
        gates[f"G8_once_only_{key}"] = _g8_once_only(key, device)
        gates[f"G9_relabel_invariance_{key}"] = _g9_relabel_invariance(key, 2, device)
        gates[f"G10_parameter_identity_{key}"] = _g10_parameter_identity(key)
        gates[f"G11_init_matching_{key}"] = _g11_init_matching(key)
    passed = {name: bool(gate.get("passed", False)) for name, gate in gates.items()}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device),
        "gates": gates,
        "gate_pass": passed,
        "all_passed": bool(all(passed.values())),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "correctness.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"E2E-DictEnv-T1 correctness gates failed: {[k for k, v in passed.items() if not v]}")
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_run(
    arm: str,
    candidate_key: str,
    seed: int,
    lambda_rec: float,
    max_epochs: int,
    tag: str,
    device: str = "cuda",
) -> dict[str, Any]:
    run_path = RESULTS_DIR / f"{tag}.json"
    if run_path.exists():
        return _read_json(run_path)
    device_obj = torch.device(device)
    candidate = _cand(candidate_key)
    train_data = v0run.load_split("train")
    valid_data = v0run.load_split("valid")
    reference_model = t1.build_model(t1.SPARSE_ARM, candidate, seed=int(seed))
    reference_state = {k: v.detach().clone() for k, v in reference_model.state_dict().items()}
    model = t1.build_model(arm, candidate, seed=int(seed), reference_state=reference_state).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = v0run.e2e.make_env_loader(train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    eval_loader = v0run.e2e.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_molecules = n_nodes = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lambda_rec * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_molecules += int(batch.y.view(-1).shape[0])
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + t1.EPS
            rec_sum += float((numerator / denominator).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_molecules, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = v0run.evaluate(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "valid_mae": float(valid["mae"]),
                "valid_rec": float(valid["rec"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec:.6f} "
                f"valid={float(valid['mae']):.6f} valid_rec={float(valid['rec']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    best_model = t1.build_model(arm, candidate, seed=int(seed))
    best_model.load_state_dict(best_state)
    best_valid = v0run.evaluate(best_model.to(device_obj), eval_loader, device_obj)

    members = sorted(int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:5])
    soup_state = {k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0) for k in epoch_states[members[0]]}
    soup_model = t1.build_model(arm, candidate, seed=int(seed))
    soup_model.load_state_dict(soup_state)
    soup_valid = v0run.evaluate(soup_model.to(device_obj), eval_loader, device_obj)

    dbar_init = np.asarray(e2e_v0.normalized_dictionary(reference_model.D.detach().float()))
    dbar_soup = np.asarray(e2e_v0.normalized_dictionary(soup_state["D"].detach().float()))

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STATE_DIR / f"{tag}_selection_state.pt")
    torch.save(soup_state, STATE_DIR / f"{tag}_soup_state.pt")
    torch.save({int(e): epoch_states[int(e)] for e in members}, STATE_DIR / f"{tag}_soup_members.pt")
    _write_csv(RESULTS_DIR / f"{tag}_curve.csv", curve)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "candidate_key": candidate_key,
        "candidate_id": candidate.candidate_id,
        "coding_operator": "tied-IHT(D), exact top-8" if arm == t1.SPARSE_ARM else "phi @ Dbar (dense tied)",
        "seed": int(seed),
        "device": str(device_obj),
        "lambda_rec": float(lambda_rec),
        "lambda_v0": float(t1.LAMBDA_V0),
        "lambda_scale": float(lambda_rec / t1.LAMBDA_V0),
        "max_epochs": int(max_epochs),
        "parameter_accounting": t1.total_parameter_count(candidate),
        "epochs_run": int(len(curve)),
        "best_valid_mae": float(best_valid["mae"]),
        "best_epoch": int(best_epoch),
        "best_valid_rec": float(best_valid["rec"]),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "soup": {
            "available": True,
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_valid["mae"]),
            "soup_valid_rec": float(soup_valid["rec"]),
        },
        "dictionary_movement": {
            "soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "soup_vs_init_relative": float(np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + t1.EPS)),
        },
        "soup_members": members,
        "member_state_sha256": {str(e): _state_hash(epoch_states[int(e)]) for e in members},
        "wall_clock_s": wall,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "selection_state_sha256": _state_hash(best_state),
        "soup_state_sha256": _state_hash(soup_state),
        "valid_targets": best_valid["targets"].tolist(),
        "valid_predictions_best": best_valid["predictions"].tolist(),
        "valid_predictions_soup": soup_valid["predictions"].tolist(),
        "official_test_loaded": False,
    }
    _write_json(run_path, payload)
    print(f"[{tag}] best={best_valid['mae']:.6f}@{best_epoch} soup={soup_valid['mae']:.6f} members={members} wall={wall:.1f}s", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stages A / B / C
# ---------------------------------------------------------------------------


def _stage_a_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in t1.CANDIDATE_ORDER:
        path = RESULTS_DIR / f"stage_a_{key}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        rows.append(
            {
                "candidate_key": key,
                "candidate_id": payload["candidate_id"],
                "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
                "best_valid_mae": float(payload["best_valid_mae"]),
                "best_epoch": int(payload["best_epoch"]),
                "train_mae_at_best": float(payload["train_mae_at_best"]),
                "soup_members": payload["soup"]["members"],
                "wall_clock_s": float(payload["wall_clock_s"]),
                "params": int(payload["parameter_accounting"]["whole_model"]),
            }
        )
    return rows


def stage_a_stage(device: str = "cuda", only: str | None = None) -> dict[str, Any]:
    keys = [only] if only else list(t1.CANDIDATE_ORDER)
    for key in keys:
        train_run(t1.SPARSE_ARM, key, 0, t1.LAMBDA_V0, MAX_EPOCHS, f"stage_a_{key}", device)
    rows = _stage_a_rows()
    if only is None:
        _write_json(RESULTS_DIR / "stage_a_summary.json", {"rows": rows, "official_test_loaded": False})
    return {"rows": rows}


def resolve_stage_a() -> dict[str, Any]:
    rows = _stage_a_rows()
    if not rows:
        raise RuntimeError("Stage A has not run yet")
    eligible = [row for row in rows if _health_gate_ok(row["candidate_key"], f"stage_a_{row['candidate_key']}")]
    pool = eligible if eligible else rows
    winner = min(pool, key=lambda r: (float(r["soup_valid_mae"]), float(r["best_valid_mae"]), int(r["best_epoch"])))
    return {"rows": rows, "eligible": [r["candidate_key"] for r in eligible], "winner": winner["candidate_key"], "winner_row": winner}


def stage_b_stage(device: str = "cuda", only: str | None = None) -> dict[str, Any]:
    selection = resolve_stage_a()
    winner = selection["winner"]
    targets: list[tuple[float, str]] = []
    if only in (None, "b1"):
        targets.append((0.5, "stage_b_lambda050"))
    if only in (None, "b2"):
        targets.append((0.25, "stage_b_lambda025"))
    rows: list[dict[str, Any]] = []
    for scale, tag in targets:
        payload = train_run(t1.SPARSE_ARM, winner, 0, t1.LAMBDA_V0 * scale, MAX_EPOCHS, tag, device)
        rows.append(
            {
                "tag": tag,
                "lambda_scale": scale,
                "lambda_rec": float(payload["lambda_rec"]),
                "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
                "best_valid_mae": float(payload["best_valid_mae"]),
                "best_epoch": int(payload["best_epoch"]),
                "train_mae_at_best": float(payload["train_mae_at_best"]),
                "soup_valid_rec": float(payload["soup"]["soup_valid_rec"]),
                "soup_members": payload["soup"]["members"],
                "wall_clock_s": float(payload["wall_clock_s"]),
            }
        )
    return {"winner": winner, "rows": rows}


def _stage_b_rows(winner: str) -> list[dict[str, Any]]:
    stage_a_payload = _read_json(RESULTS_DIR / f"stage_a_{winner}.json")
    rows: list[dict[str, Any]] = [
        {
            "tag": f"stage_a_{winner}",
            "lambda_scale": 1.0,
            "lambda_rec": float(stage_a_payload["lambda_rec"]),
            "soup_valid_mae": float(stage_a_payload["soup"]["soup_valid_mae"]),
            "best_valid_mae": float(stage_a_payload["best_valid_mae"]),
            "best_epoch": int(stage_a_payload["best_epoch"]),
            "soup_members": stage_a_payload["soup"]["members"],
            "soup_valid_rec": float(stage_a_payload["soup"]["soup_valid_rec"]),
        }
    ]
    for tag, scale in (("stage_b_lambda050", 0.5), ("stage_b_lambda025", 0.25)):
        path = RESULTS_DIR / f"{tag}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        rows.append(
            {
                "tag": tag,
                "lambda_scale": scale,
                "lambda_rec": float(payload["lambda_rec"]),
                "soup_valid_mae": float(payload["soup"]["soup_valid_mae"]),
                "best_valid_mae": float(payload["best_valid_mae"]),
                "best_epoch": int(payload["best_epoch"]),
                "soup_members": payload["soup"]["members"],
                "soup_valid_rec": float(payload["soup"]["soup_valid_rec"]),
            }
        )
    return rows


def resolve_stage_b() -> dict[str, Any]:
    selection = resolve_stage_a()
    winner = selection["winner"]
    rows = _stage_b_rows(winner)
    eligible = [row for row in rows if _health_gate_ok(winner, row["tag"])]
    pool = eligible if eligible else rows
    chosen = min(pool, key=lambda r: (float(r["soup_valid_mae"]), float(r["best_valid_mae"]), int(r["best_epoch"])))
    return {"winner": winner, "rows": rows, "eligible": [r["tag"] for r in eligible], "chosen": chosen}


def stage_c_stage(device: str = "cuda") -> dict[str, Any]:
    selection = resolve_stage_b()
    chosen = selection["chosen"]
    winner = selection["winner"]
    trigger = bool(int(chosen["best_epoch"]) >= 220 or any(int(e) >= 230 for e in chosen["soup_members"]))
    payload: dict[str, Any] = {
        "triggered": trigger,
        "reason": f"best_epoch={chosen['best_epoch']} soup_members={chosen['soup_members']}",
        "winner": winner,
        "chosen_lambda_scale": float(chosen["lambda_scale"]),
        "official_test_loaded": False,
    }
    if not trigger:
        _write_json(RESULTS_DIR / "stage_c_horizon320.json", payload)
        return payload
    run = train_run(t1.SPARSE_ARM, winner, 0, float(chosen["lambda_rec"]), HORIZON_EXTENDED, "stage_c_horizon320", device)
    delta = float(chosen["soup_valid_mae"]) - float(run["soup"]["soup_valid_mae"])
    payload.update(
        {
            "horizon320_soup_valid_mae": float(run["soup"]["soup_valid_mae"]),
            "horizon320_best_valid_mae": float(run["best_valid_mae"]),
            "horizon320_best_epoch": int(run["best_epoch"]),
            "horizon240_soup_valid_mae": float(chosen["soup_valid_mae"]),
            "delta_240_minus_320": float(delta),
            "adopted": bool(delta >= 0.001),
            "adopted_horizon": int(HORIZON_EXTENDED if delta >= 0.001 else MAX_EPOCHS),
        }
    )
    _write_json(RESULTS_DIR / "stage_c_horizon320.json", payload)
    return payload


def finalize_selection_stage() -> dict[str, Any]:
    a = resolve_stage_a()
    b = resolve_stage_b()
    c_path = RESULTS_DIR / "stage_c_horizon320.json"
    c = _read_json(c_path) if c_path.exists() else {"triggered": False, "adopted_horizon": MAX_EPOCHS}
    horizon = int(c.get("adopted_horizon", MAX_EPOCHS)) if c.get("triggered") else MAX_EPOCHS
    winner_tag = b["chosen"]["tag"]
    best_tuned = float(b["chosen"]["soup_valid_mae"])
    if c.get("triggered") and c.get("adopted"):
        winner_tag = "stage_c_horizon320"
        best_tuned = float(c["horizon320_soup_valid_mae"])
    improvement = float(V0_SOUP - best_tuned)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "stage_a": a,
        "stage_b": b,
        "stage_c": c,
        "candidate_key": b["winner"],
        "candidate_id": _cand(b["winner"]).candidate_id,
        "lambda_scale": float(b["chosen"]["lambda_scale"]),
        "lambda_rec": float(b["chosen"]["lambda_rec"]),
        "horizon": horizon,
        "winner_tag": winner_tag,
        "best_tuned_soup_valid_mae": best_tuned,
        "v0_reference_soup": float(V0_SOUP),
        "improvement_vs_v0": improvement,
        "material_improvement": bool(improvement >= t1.TUNING_MATERIAL_DELTA),
        "interface_material_gain": bool(float(min(r["soup_valid_mae"] for r in a["rows"])) <= float(V0_SOUP - t1.TUNING_MATERIAL_DELTA)),
        "official_test_loaded": False,
    }
    payload["band"] = _band(best_tuned)
    _write_json(RESULTS_DIR / "tuning_decision.json", payload)
    return payload


def _band(value: float) -> str:
    if value <= t1.BAND_STRONG_MAX:
        return "strong"
    if value <= t1.BAND_VIABLE_MAX:
        return "viable"
    return "weak"


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def _load_state_model(arm: str, candidate_key: str, state_path: Path, device: torch.device) -> t1.T1Model:
    model = t1.build_model(arm, _cand(candidate_key), seed=0)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def health_for(candidate_key: str, tag: str, device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    model = _load_state_model(t1.SPARSE_ARM, candidate_key, STATE_DIR / f"{tag}_soup_state.pt", device_obj)
    alpha_train = v0run._codes_for_split(model, "train", device_obj)
    alpha_valid = v0run._codes_for_split(model, "valid", device_obj)
    usage_train = v0run._atom_usage(alpha_train)
    usage_valid = v0run._atom_usage(alpha_valid)
    l0_train = (np.abs(alpha_train) > 0).sum(axis=1)
    l0_valid = (np.abs(alpha_valid) > 0).sum(axis=1)

    init_model = t1.build_model(t1.SPARSE_ARM, _cand(candidate_key), seed=0)
    init_model.to(device_obj)
    alpha_init = v0run._codes_for_split(init_model, "train", device_obj)
    init_active = set(np.nonzero((np.abs(alpha_init) > 0).sum(axis=0))[0].tolist())
    trained_active = set(np.nonzero((np.abs(alpha_train) > 0).sum(axis=0))[0].tolist())
    retained = (len(init_active & trained_active) / len(init_active)) if init_active else 0.0

    blob_train = torch.load(v0run._env_cache_path("train"), map_location="cpu", weights_only=False)
    blob_valid = torch.load(v0run._env_cache_path("valid"), map_location="cpu", weights_only=False)
    dbar = np.asarray(e2e_v0.normalized_dictionary(model.D.detach().float()).cpu())

    def _rec(alpha: np.ndarray, blob: dict[str, Any]) -> float:
        phi = blob["phi"].numpy().astype(np.float64)
        phi_hat = alpha.astype(np.float64) @ dbar.astype(np.float64).T
        numerator = ((phi - phi_hat) ** 2).sum(axis=1)
        denominator = (phi ** 2).sum(axis=1) + t1.EPS
        return float((numerator / denominator).mean())

    rec_train = _rec(alpha_train, blob_train)
    rec_valid = _rec(alpha_valid, blob_valid)

    valid_data = v0run.load_split("valid", subset=2000)
    loader = v0run.e2e.make_env_loader(valid_data, BATCH_SIZE, False, 0)
    e_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_obj)
            _pred, aux = model(batch, return_aux=True)
            e_rows.append(aux["E"].cpu().numpy())
            if sum(row.shape[0] for row in e_rows) >= 8000:
                break
    rank = v0run._environment_rank(np.concatenate(e_rows, axis=0))

    train_data = v0run.load_split("train", subset=64)
    loader = v0run.e2e.make_env_loader(train_data, 64, False, 0)
    batch = next(iter(loader)).to(device_obj)
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm()) if model.D.grad is not None else 0.0

    singular = np.linalg.svd(dbar, compute_uv=False)
    energy = singular ** 2
    effective_rank = float((energy.sum() ** 2) / (energy ** 2).sum())
    spearman = v0run._spearman(np.asarray(usage_train["usage"]), np.asarray(usage_valid["usage"]))

    gates = {
        "exact_top8": bool(int(l0_valid.max()) <= int(t1.SPARSITY)),
        "initial_active_retention": bool(retained >= t1.HEALTH_MIN_INITIAL_RETENTION),
        "active_train": bool(usage_train["active_atoms"] >= t1.HEALTH_MIN_ACTIVE),
        "active_valid": bool(usage_valid["active_atoms"] >= t1.HEALTH_MIN_ACTIVE),
        "effective_atom_count": bool(usage_valid["effective_atom_count"] >= t1.HEALTH_MIN_EFFECTIVE),
        "no_single_atom_dominance": bool(usage_valid["top1_share"] <= t1.HEALTH_MAX_ATOM_SHARE),
        "task_gradient_to_d": bool(math.isfinite(task_grad) and task_grad > 0.0),
        "environment_rank": bool(rank["participation_ratio"] > 1.0),
        "usage_stable": bool(spearman >= 0.5),
        "valid_reconstruction_ceiling": bool(rec_valid <= t1.HEALTH_MAX_VALID_RECONSTRUCTION),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "candidate_key": candidate_key,
        "candidate_id": _cand(candidate_key).candidate_id,
        "tag": tag,
        "state": "soup",
        "train": {**usage_train, "max_l0": int(l0_train.max()), "exact_top_s_fraction": float((l0_train == int(t1.SPARSITY)).mean())},
        "valid": {**usage_valid, "max_l0": int(l0_valid.max()), "exact_top_s_fraction": float((l0_valid == int(t1.SPARSITY)).mean())},
        "initial_active_atoms": sorted(init_active),
        "trained_active_atoms": sorted(trained_active),
        "initial_active_retention": float(retained),
        "usage_spearman_train_valid": float(spearman),
        "dictionary": {
            "effective_rank": effective_rank,
            "movement_fro_from_ksvd_init": float(np.linalg.norm(dbar - np.asarray(e2e_v0.normalized_dictionary(init_model.D.detach().float()).cpu()))),
        },
        "reconstruction": {"train": rec_train, "valid": rec_valid},
        "environment_rank": rank,
        "task_gradient_to_D": float(task_grad),
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "official_test_loaded": False,
    }


def _health_gate_ok(candidate_key: str, tag: str) -> bool:
    path = RESULTS_DIR / "health" / f"{tag}.json"
    if path.exists():
        return bool(_read_json(path)["all_passed"])
    state = STATE_DIR / f"{tag}_soup_state.pt"
    if not state.exists():
        return False
    payload = health_for(candidate_key, tag, device="cuda" if torch.cuda.is_available() else "cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return bool(payload["all_passed"])


def health_stage(device: str = "cuda") -> dict[str, Any]:
    selection = resolve_stage_b()
    candidate_key = selection["winner"]
    final = finalize_selection_stage()
    health_dir = RESULTS_DIR / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Any] = {}
    for tag in [f"stage_a_{candidate_key}", "stage_b_lambda050", "stage_b_lambda025", "stage_c_horizon320", "final_sparse_seed1"]:
        if not (STATE_DIR / f"{tag}_soup_state.pt").exists():
            continue
        payload = health_for(candidate_key, tag, device)
        _write_json(health_dir / f"{tag}.json", payload)
        payloads[tag] = payload
    _write_json(RESULTS_DIR / "dictionary_health.json", {"candidate_key": candidate_key, "final": final, "by_tag": payloads, "official_test_loaded": False})
    return {"candidate_key": candidate_key, "by_tag": payloads}


# ---------------------------------------------------------------------------
# mechanism interventions
# ---------------------------------------------------------------------------


def mechanism_stage(device: str = "cuda", tag: str = "final_sparse_seed0") -> dict[str, Any]:
    selection = resolve_stage_b()
    candidate_key = selection["winner"]
    final = finalize_selection_stage()
    state_path = STATE_DIR / f"{final['winner_tag']}_soup_state.pt"
    device_obj = torch.device(device)
    model = _load_state_model(t1.SPARSE_ARM, candidate_key, state_path, device_obj)
    valid_data = v0run.load_split("valid")
    loader = v0run.e2e.make_env_loader(valid_data, BATCH_SIZE, False, EVAL_SHUFFLE_OFFSET)
    clean = v0run.evaluate(model, loader, device_obj)
    m_soup = float(clean["mae"])

    zero = v0run.evaluate(model, loader, device_obj, coord_zero=True)
    zero_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "candidate_key": candidate_key,
        "tag": final["winner_tag"],
        "state": "soup",
        "M_S": m_soup,
        "M_zero": float(zero["mae"]),
        "G_dict_use": float(zero["mae"] - m_soup),
        "gate_threshold": float(t1.GATE_ZERO),
        "gate_pass": bool(float(zero["mae"] - m_soup) >= float(t1.GATE_ZERO)),
        "mean_abs_prediction_shift": float(np.mean(np.abs(zero["predictions"] - clean["predictions"]))),
        "max_abs_prediction_shift": float(np.max(np.abs(zero["predictions"] - clean["predictions"]))),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_zero.json", zero_payload)

    rows: list[dict[str, Any]] = []
    for shuffle_seed in SHUFFLE_SEEDS:
        for data in valid_data:
            data.env_occ_coord_node = e2e_v0.shuffled_occ_node_for_molecule(
                data.env_occ_node, data.env_occ_root, data.env_occ_shell, int(shuffle_seed)
            )
        with_node_loader = v0run.e2e.make_env_loader(valid_data, BATCH_SIZE, False, EVAL_SHUFFLE_OFFSET)
        shuffled = v0run.evaluate(model, with_node_loader, device_obj, use_coord_node=True)
        rows.append(
            {
                "seed": int(shuffle_seed),
                "valid_mae": float(shuffled["mae"]),
                "mean_abs_prediction_shift": float(np.mean(np.abs(shuffled["predictions"] - clean["predictions"]))),
                "max_abs_prediction_shift": float(np.max(np.abs(shuffled["predictions"] - clean["predictions"]))),
            }
        )
        for data in valid_data:
            data.env_occ_coord_node = None
    m_shuffle = float(np.mean([row["valid_mae"] for row in rows]))
    shuffle_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "candidate_key": candidate_key,
        "tag": final["winner_tag"],
        "state": "soup",
        "M_S": m_soup,
        "M_shuffle": m_shuffle,
        "G_assign": float(m_shuffle - m_soup),
        "gate_threshold": float(t1.GATE_SHUFFLE),
        "gate_pass": bool(float(m_shuffle - m_soup) >= float(t1.GATE_SHUFFLE)),
        "shuffle_seeds": [int(s) for s in SHUFFLE_SEEDS],
        "rows": rows,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_shuffle.json", shuffle_payload)
    return {"zero": zero_payload, "shuffle": shuffle_payload}


# ---------------------------------------------------------------------------
# freeze / confirmation
# ---------------------------------------------------------------------------


def _historical_comparators() -> dict[str, Any]:
    comparators: dict[str, Any] = {}
    v0_soup = TRACK_ROOT / "results/e2e_dictenv_v0/states/sparse_seed0_soup_state.pt"
    if v0_soup.exists():
        comparators["v0_sparse_seed0_soup"] = {
            "path": str(v0_soup.relative_to(REPO_ROOT)),
            "sha256": _file_sha256(v0_soup),
            "available": True,
            "note": "historical E2E-DictEnv-v0 Sparse soup state (persisted); read-only comparator",
        }
    else:
        comparators["v0_sparse_seed0_soup"] = {"available": False, "reason": "v0 soup state not persisted"}
    fec_s1_states = TRACK_ROOT / "results/fec_s1/states"
    member_files = sorted(fec_s1_states.glob("*soup*member*")) if fec_s1_states.exists() else []
    comparators["fec_s1_seed0_soup"] = (
        {"available": True, "files": [str(p.relative_to(REPO_ROOT)) for p in member_files]}
        if member_files
        else {"available": False, "reason": "FEC-S1 soup member states are not persisted (SKIP per preregistration)"}
    )
    return comparators


def freeze_stage() -> dict[str, Any]:
    final = finalize_selection_stage()
    candidate_key = final["candidate_key"]
    candidate = _cand(candidate_key)
    soup_state = torch.load(STATE_DIR / f"{final['winner_tag']}_soup_state.pt", map_location="cpu", weights_only=False)
    selection_payload = _read_json(RESULTS_DIR / f"{final['winner_tag']}.json")
    init_d = t1.build_model(t1.SPARSE_ARM, candidate, seed=0).D.detach()
    health_path = RESULTS_DIR / "health" / f"{final['winner_tag']}.json"
    if not health_path.exists():
        health_for(candidate_key, final["winner_tag"], device="cuda" if torch.cuda.is_available() else "cpu")
    health = _read_json(health_path)
    member_hashes = dict(selection_payload.get("member_state_sha256", {}))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "config_id": f"e2e_dictenv_t1_{candidate_key}_lam{final['lambda_scale']:.2f}_h{final['horizon']}",
        "git_commit": _git_commit(),
        "architecture_module": str(MODULE_PATH.relative_to(REPO_ROOT)),
        "architecture_source_sha256": _module_sha256(),
        "candidate_key": candidate_key,
        "candidate_id": candidate.candidate_id,
        "mode": candidate.mode,
        "r_dict": int(candidate.r_dict),
        "decoder_hidden": int(candidate.decoder_hidden),
        "decoder_input": int(candidate.decoder_input),
        "parameter_count": int(t1.total_parameter_count(candidate)["whole_model"]),
        "K": int(t1.K_ATOMS),
        "s": int(t1.SPARSITY),
        "iht_steps": int(t1.IHT_STEPS),
        "D_init_sha256": _state_hash({"D": init_d}),
        "lambda": float(final["lambda_rec"]),
        "lambda_scale": float(final["lambda_scale"]),
        "horizon": int(final["horizon"]),
        "seed0_tuning_winner_tag": final["winner_tag"],
        "seed0_selection_state_sha256": selection_payload["selection_state_sha256"],
        "seed0_soup_state_sha256": selection_payload["soup_state_sha256"],
        "seed0_soup_state_computed_sha256": _state_hash(soup_state),
        "soup_members": selection_payload["soup"]["members"],
        "soup_tensor_hashes": member_hashes,
        "official_valid": {
            "soup_valid_mae": float(selection_payload["soup"]["soup_valid_mae"]),
            "best_valid_mae": float(selection_payload["best_valid_mae"]),
            "best_epoch": int(selection_payload["best_epoch"]),
            "soup_valid_rec": float(selection_payload["soup"]["soup_valid_rec"]),
        },
        "health_all_passed": bool(health["all_passed"]),
        "historical_comparators": _historical_comparators(),
        "official_test_loaded_at_freeze_time": False,
        "official_test_loaded_scope": "FOR THIS ROUND: no test read occurred before this freeze; not a project-wide pristine claim",
    }
    _write_json(RESULTS_DIR / "architecture_freeze.json", payload)
    return payload


def confirm_stage(device: str = "cuda") -> dict[str, Any]:
    freeze = _read_json(RESULTS_DIR / "architecture_freeze.json")
    candidate_key = freeze["candidate_key"]
    lambda_rec = float(freeze["lambda"])
    horizon = int(freeze["horizon"])
    sparse = train_run(t1.SPARSE_ARM, candidate_key, 1, lambda_rec, horizon, "final_sparse_seed1", device)
    dense = train_run(t1.DENSE_ARM, candidate_key, 1, lambda_rec, horizon, "final_dense_seed1", device)
    freeze["seed1_confirmation"] = {
        "sparse_soup_valid_mae": float(sparse["soup"]["soup_valid_mae"]),
        "sparse_best_valid_mae": float(sparse["best_valid_mae"]),
        "sparse_best_epoch": int(sparse["best_epoch"]),
        "sparse_selection_state_sha256": sparse["selection_state_sha256"],
        "sparse_soup_state_sha256": sparse["soup_state_sha256"],
        "dense_soup_valid_mae": float(dense["soup"]["soup_valid_mae"]),
        "dense_best_valid_mae": float(dense["best_valid_mae"]),
        "dense_best_epoch": int(dense["best_epoch"]),
        "dense_selection_state_sha256": dense["selection_state_sha256"],
        "dense_soup_state_sha256": dense["soup_state_sha256"],
        "G_sparse_seed1": float(dense["soup"]["soup_valid_mae"] - sparse["soup"]["soup_valid_mae"]),
        "G_sparse_seed1_gate": float(t1.GATE_DICT_SPECIFIC),
        "G_sparse_seed1_pass": bool(dense["soup"]["soup_valid_mae"] - sparse["soup"]["soup_valid_mae"] >= t1.GATE_DICT_SPECIFIC),
    }
    freeze["official_test_loaded_at_freeze_time"] = False
    _write_json(RESULTS_DIR / "architecture_freeze.json", freeze)
    return freeze


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def analyze_stage() -> dict[str, Any]:
    final = _read_json(RESULTS_DIR / "tuning_decision.json") if (RESULTS_DIR / "tuning_decision.json").exists() else finalize_selection_stage()
    freeze = _read_json(RESULTS_DIR / "architecture_freeze.json")
    seed1 = freeze.get("seed1_confirmation", {})
    zero = _read_json(RESULTS_DIR / "mechanism_zero.json")
    shuffle = _read_json(RESULTS_DIR / "mechanism_shuffle.json")
    health = _read_json(RESULTS_DIR / "health" / f"{final['winner_tag']}.json")
    seed0 = float(final["best_tuned_soup_valid_mae"])
    m_s1 = float(seed1["sparse_soup_valid_mae"])
    m_d1 = float(seed1["dense_soup_valid_mae"])
    g_seed1 = float(m_d1 - m_s1)
    health_pass = bool(health["all_passed"])
    zero_pass = bool(zero["gate_pass"])
    shuffle_pass = bool(shuffle["gate_pass"])
    mechanism_pass = health_pass and zero_pass and shuffle_pass
    seed1_viable = bool(m_s1 <= t1.BAND_VIABLE_MAX)
    specific = bool(g_seed1 >= t1.GATE_DICT_SPECIFIC)
    improvement = float(V0_SOUP - seed0)
    if not mechanism_pass:
        verdict = t1.VERDICTS["mechanism_lost"]
    elif improvement < t1.TUNING_MATERIAL_DELTA:
        verdict = t1.VERDICTS["no_material_gain"]
    elif not (specific and seed1_viable):
        verdict = t1.VERDICTS["perf_gain_specific_unstable"]
    elif seed0 <= t1.BAND_STRONG_MAX:
        verdict = t1.VERDICTS["strong_specific"]
    else:
        verdict = t1.VERDICTS["viable_specific"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "primary_metric": "fixed Top-5 soup official-valid MAE",
        "development_result": {
            "seed0_tuned_soup_valid_mae": seed0,
            "band": _band(seed0),
            "v0_reference_soup": float(V0_SOUP),
            "improvement_vs_v0": improvement,
            "winner_tag": final["winner_tag"],
            "candidate_id": final["candidate_id"],
            "lambda_scale": float(final["lambda_scale"]),
            "horizon": int(final["horizon"]),
        },
        "confirmation_result": {
            "seed1_sparse_soup_valid_mae": m_s1,
            "seed1_dense_soup_valid_mae": m_d1,
            "G_sparse_seed1": g_seed1,
            "G_sparse_seed1_gate": float(t1.GATE_DICT_SPECIFIC),
            "dictionary_specific": specific,
            "seed1_sparse_viable": seed1_viable,
        },
        "mechanism": {
            "M_zero": float(zero["M_zero"]),
            "G_dict_use": float(zero["G_dict_use"]),
            "zero_pass": zero_pass,
            "M_shuffle": float(shuffle["M_shuffle"]),
            "G_assign": float(shuffle["G_assign"]),
            "shuffle_pass": shuffle_pass,
            "health_pass": health_pass,
        },
        "historical_anchors": {"FEC_S1_soup": FEC_S1_SOUP, "FEC_S1_best": FEC_S1_BEST, "v0_soup": V0_SOUP},
        "tuning_decision": final,
        "verdict": verdict,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_report(payload, health)
    _write_decision(payload)
    return payload


def _write_report(payload: Mapping[str, Any], health: Mapping[str, Any]) -> None:
    dev = payload["development_result"]
    conf = payload["confirmation_result"]
    mech = payload["mechanism"]
    lines = [
        "# E2E-DictEnv-T1 — report",
        "",
        "Round `e2e_dictenv_t1`; study `zinc-context-gap`.  Official ZINC test is",
        "loaded **only** at the terminal reporting stage, after the freeze.",
        "",
        "## Frozen verdict",
        "",
        f"```\n{payload['verdict']}\n```",
        "",
        "## Development result (valid-guided tuning, seed 0)",
        "",
        f"* tuned Sparse soup valid MAE = **{dev['seed0_tuned_soup_valid_mae']:.6f}** "
        f"(band **{dev['band']}**)",
        f"* candidate {dev['candidate_id']}, lambda scale {dev['lambda_scale']}, "
        f"horizon {dev['horizon']}, winner `{dev['winner_tag']}`",
        f"* improvement vs v0 `{dev['v0_reference_soup']:.6f}`: **{dev['improvement_vs_v0']:+.6f}**",
        "",
        "## Confirmation result (frozen config, seed 1)",
        "",
        f"* Sparse soup = **{conf['seed1_sparse_soup_valid_mae']:.6f}**",
        f"* DenseTied soup = **{conf['seed1_dense_soup_valid_mae']:.6f}**",
        f"* `G_sparse^seed1 = {conf['G_sparse_seed1']:+.6f}` "
        f"(gate {conf['G_sparse_seed1_gate']}: {conf['dictionary_specific']})",
        f"* seed-1 Sparse viable (<= {t1.BAND_VIABLE_MAX}): {conf['seed1_sparse_viable']}",
        "",
        "## Mechanism (frozen Sparse soup)",
        "",
        f"* zero-code `G_dict-use = {mech['G_dict_use']:+.6f}` ({mech['zero_pass']})",
        f"* assignment shuffle `G_assign = {mech['G_assign']:+.6f}` ({mech['shuffle_pass']})",
        f"* dictionary health: {mech['health_pass']} "
        f"(active train/valid {health['train']['active_atoms']}/{health['valid']['active_atoms']}, "
        f"initial retention {health['initial_active_retention']:.3f})",
        "",
        "## Historical anchors (context only)",
        "",
        f"* FEC-S1 seed-0 soup {FEC_S1_SOUP:.6f}; v0 Sparse soup {V0_SOUP:.6f}",
        f"* commit `{payload['git_commit']}`; official_test_loaded = "
        f"{payload['official_test_loaded']} (terminal read separated)",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(payload: Mapping[str, Any]) -> None:
    dev = payload["development_result"]
    conf = payload["confirmation_result"]
    mech = payload["mechanism"]
    lines = [
        "# E2E-DictEnv-T1 — decision",
        "",
        f"```\n{payload['verdict']}\n```",
        "",
        f"* tuned seed-0 Sparse soup = **{dev['seed0_tuned_soup_valid_mae']:.6f}** "
        f"({dev['candidate_id']}, lambda scale {dev['lambda_scale']}, horizon {dev['horizon']})",
        f"* improvement vs v0 = **{dev['improvement_vs_v0']:+.6f}**",
        f"* seed-1 Sparse `{conf['seed1_sparse_soup_valid_mae']:.6f}` vs DenseTied "
        f"`{conf['seed1_dense_soup_valid_mae']:.6f}` -> `G_sparse^seed1` = "
        f"**{conf['G_sparse_seed1']:+.6f}** ({'PASS' if conf['dictionary_specific'] else 'FAIL'})",
        f"* zero-code `{mech['G_dict_use']:+.6f}` / shuffle `{mech['G_assign']:+.6f}` / "
        f"health {mech['health_pass']}",
        "",
        "## Reading",
        "",
        _reading(payload["verdict"]),
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reading(verdict: str) -> str:
    if verdict == t1.VERDICTS["strong_specific"]:
        return ("The tuned sparse dictionary core reaches the strong absolute band while its value "
                "replicates specifically against the matched dense tied coordinate at seed 1.  This is "
                "the strongest evidence so far that a sparse structural dictionary can be the "
                "predictive core of a no-message-passing molecular environment.")
    if verdict == t1.VERDICTS["viable_specific"]:
        return ("The tuned sparse dictionary core is viable (absolute band 0.135-0.145) and its value "
                "replicates specifically at seed 1: dictionary-core architecture is supported, with a "
                "remaining absolute gap to the strong static anchor.")
    if verdict == t1.VERDICTS["perf_gain_specific_unstable"]:
        return ("Tuning improved the Sparse absolute performance but the dictionary-specific "
                "superiority over the matched dense tied control did not survive the frozen seed-1 "
                "confirmation.  Do not use the old seed-0 gap to claim specificity.")
    if verdict == t1.VERDICTS["no_material_gain"]:
        return ("Interface/lambda/horizon tuning produced no material absolute gain over v0 "
                "(< 0.002).  The dictionary mechanism is established but the current dictionary-first "
                "environment class has an absolute-performance ceiling under this budget.")
    return ("A core dictionary mechanism (health / zero-code / assignment shuffle) failed after "
            "tuning; no dictionary-core success may be claimed.")


# ---------------------------------------------------------------------------
# terminal test (one load, reporting only)
# ---------------------------------------------------------------------------


def unlock_and_test_stage(device: str = "cuda") -> dict[str, Any]:
    unlock_path = RESULTS_DIR / "official_test_unlock.json"
    if unlock_path.exists():
        raise RuntimeError("refusing test: official test already unlocked once this round")
    freeze = _read_json(RESULTS_DIR / "architecture_freeze.json")
    if "seed1_confirmation" not in freeze:
        raise RuntimeError("refusing test: seed-1 confirmation not frozen yet")
    if freeze.get("official_test_loaded_at_freeze_time") is not False:
        raise RuntimeError("refusing test: freeze does not assert a pre-test freeze")
    _write_json(
        unlock_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "user_authorised_test_read": True,
            "purpose": "terminal reporting only",
            "project_wide_pristine": False,
            "reason": "historical unrelated ZINC official-test reads already exist",
            "architecture_frozen_before_this_rounds_test_read": True,
            "test_will_not_affect_any_model_config_checkpoint_decision": True,
            "official_test_loaded": True,
        },
    )

    from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e

    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_enc, test_data, _audit = ztraining.build_encoded(train_records, test_records, config)
    del _train_enc
    raw_test = list(zlr._load_zinc(ZINC_ROOT, "test"))
    if len(test_data) != len(raw_test):
        raise RuntimeError("official-test length mismatch")
    for index, (data, raw_molecule) in enumerate(zip(test_data, raw_test)):
        graph, node_types, edge_types = zlr._data_to_graph(raw_molecule)
        if int(data.num_nodes) != int(raw_molecule.num_nodes):
            raise RuntimeError(f"test[{index}]: node mismatch")
        incidence = e2e.env_incidence(graph, edge_types)
        data.dict_phi = torch.as_tensor(r2.build_phi(graph).astype(np.float32))
        data.dict_atom = torch.as_tensor(np.asarray(node_types, dtype=np.int64))
        data.env_occ_node = incidence["occ_node"]
        data.env_occ_root = incidence["occ_root"]
        data.env_occ_shell = incidence["occ_shell"]
        # v0-compatible environment tensors (only used by the v0 comparator)
        data.env_scalars = data.patch_cont[:, -6:].clone()
        data.env_bond_root = incidence["bond_root"]
        data.env_bond_shellpair = incidence["bond_shellpair"]
        data.env_bond_type = incidence["bond_type"]

    device_obj = torch.device(device)
    loader = v0run.e2e.make_env_loader(list(test_data), BATCH_SIZE, False, 0)
    candidate_key = freeze["candidate_key"]

    objects = [
        ("final_sparse_seed0_soup", t1.SPARSE_ARM, freeze["seed0_tuning_winner_tag"]),
        ("final_sparse_seed1_soup", t1.SPARSE_ARM, "final_sparse_seed1"),
        ("final_dense_seed1_soup", t1.DENSE_ARM, "final_dense_seed1"),
    ]
    rows: list[dict[str, Any]] = []
    for name, arm, tag in objects:
        state_path = STATE_DIR / f"{tag}_soup_state.pt"
        if not state_path.exists():
            rows.append({"name": name, "available": False, "tag": tag})
            continue
        model = _load_state_model(arm, candidate_key, state_path, device_obj)
        result = v0run.evaluate(model, loader, device_obj)
        pred = np.asarray(result["predictions"], dtype=np.float64)
        rows.append(
            {
                "name": name,
                "available": True,
                "arm": arm,
                "tag": tag,
                "params": int(t1.total_parameter_count(_cand(candidate_key))["whole_model"]),
                "test_mae": float(result["mae"]),
                "test_mean_prediction": float(pred.mean()),
                "test_std_prediction": float(pred.std()),
            }
        )
    # optional frozen historical comparator: E2E-DictEnv-v0 Sparse soup state
    v0_state_path = TRACK_ROOT / "results/e2e_dictenv_v0/states/sparse_seed0_soup_state.pt"
    if v0_state_path.exists():
        v0_model = v0run.e2e.build_model(v0run.e2e.SPARSE_ARM, seed=0)
        v0_model.load_state_dict(torch.load(v0_state_path, map_location="cpu", weights_only=False))
        v0_model.to(device_obj).eval()
        result = v0run.evaluate(v0_model, loader, device_obj)
        pred = np.asarray(result["predictions"], dtype=np.float64)
        rows.append(
            {
                "name": "v0_sparse_seed0_soup",
                "available": True,
                "arm": "sparse",
                "source": "E2E-DictEnv-v0 frozen state (read-only comparator)",
                "params": 66158,
                "test_mae": float(result["mae"]),
                "test_mean_prediction": float(pred.mean()),
                "test_std_prediction": float(pred.std()),
            }
        )
    # mechanism generalization diagnostics on the final Sparse soup
    sparse_row = next((r for r in rows if r["name"] == "final_sparse_seed1_soup" and r.get("available")), None)
    mechanism: dict[str, Any] = {}
    if sparse_row is not None:
        model = _load_state_model(t1.SPARSE_ARM, candidate_key, STATE_DIR / "final_sparse_seed1_soup_state.pt", device_obj)
        zero = v0run.evaluate(model, loader, device_obj, coord_zero=True)
        mechanism["zero_code_test_mae"] = float(zero["mae"])
        shuffle_maes: list[float] = []
        for shuffle_seed in SHUFFLE_SEEDS:
            for data in test_data:
                data.env_occ_coord_node = e2e_v0.shuffled_occ_node_for_molecule(
                    data.env_occ_node, data.env_occ_root, data.env_occ_shell, int(shuffle_seed)
                )
            sh_loader = v0run.e2e.make_env_loader(list(test_data), BATCH_SIZE, False, 0)
            shuffled = v0run.evaluate(model, sh_loader, device_obj, use_coord_node=True)
            shuffle_maes.append(float(shuffled["mae"]))
            for data in test_data:
                data.env_occ_coord_node = None
        mechanism["assignment_shuffle_test_mae"] = float(np.mean(shuffle_maes))
        mechanism["assignment_shuffle_test_mae_per_seed"] = shuffle_maes

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "note": ("terminal reporting-only evaluation of a configuration frozen using train/valid "
                 "only within E2E-DictEnv-T1; the official test is not project-wide pristine"),
        "n_test": int(len(test_data)),
        "rows": rows,
        "mechanism_generalization": mechanism,
        "official_test_loaded": True,
    }
    _write_json(RESULTS_DIR / "official_test_results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    identity_stage()
    correctness_stage(device="cpu")
    stage_a_stage(device)
    finalize_selection_stage()
    stage_b_stage(device)
    stage_c_stage(device)
    finalize_selection_stage()
    freeze_stage()
    confirm_stage(device)
    health_stage(device)
    mechanism_stage(device)
    analyze_stage()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "identity", "correct", "stage_a", "select", "stage_b", "stage_c",
            "finalize", "freeze", "confirm", "health", "mechanism", "analyze",
            "unlock", "all",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default=None, choices=["a1", "a2", "a3", "b1", "b2"])
    args = parser.parse_args(argv)
    v0run.sdp._configure_determinism()
    if args.stage == "identity":
        print(json.dumps(identity_stage(), indent=2))
    elif args.stage == "correct":
        print(json.dumps(correctness_stage(device="cpu"), indent=2))
    elif args.stage == "stage_a":
        print(json.dumps(stage_a_stage(device=args.device, only=args.only), indent=2))
    elif args.stage == "select":
        print(json.dumps(resolve_stage_a(), indent=2))
    elif args.stage == "stage_b":
        print(json.dumps(stage_b_stage(device=args.device, only=args.only), indent=2))
    elif args.stage == "stage_c":
        print(json.dumps(stage_c_stage(device=args.device), indent=2))
    elif args.stage == "finalize":
        print(json.dumps(finalize_selection_stage(), indent=2))
    elif args.stage == "freeze":
        print(json.dumps(freeze_stage(), indent=2))
    elif args.stage == "confirm":
        print(json.dumps(confirm_stage(device=args.device), indent=2))
    elif args.stage == "health":
        print(json.dumps(health_stage(device=args.device), indent=2))
    elif args.stage == "mechanism":
        print(json.dumps(mechanism_stage(device=args.device), indent=2))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    elif args.stage == "unlock":
        print(json.dumps(unlock_and_test_stage(device=args.device), indent=2))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
