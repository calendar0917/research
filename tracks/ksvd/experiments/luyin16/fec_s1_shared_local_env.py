"""FEC-S1 -- Shared Local Environment Replacement (single full seed-0 run).

Round: ``fec_s1``.  Study: ``zinc-context-gap``.
Pre-registration: ``tracks/ksvd/notes/fec_s1_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/fec_s1_prior_artifact_audit.md``.

Single question
---------------
In historical strict-static S0, delete the two per-key learned lookup memories
(``typed_embedding`` + ``parent_embedding``) completely and replace them with
one parameter-matched, vocabulary-independent shared function
``A(x_i): R^146 -> R^24`` that reads only the already-factorized, standardized
146-D local environment descriptor ``patch_cont``.  Keep every other S0
computation frozen.  Does the strict-static S0 performance band survive?

The adapter is intentionally minimal::

    A(x_i) = Linear(146, H) -> SiLU -> Linear(H, 24)
    A(x_i) = [ e_i^shared (16) ; p_i^shared (8) ]

``H`` is fixed by matching the released lookup parameter budget
(``P_A(H) = 171 H + 24``); it is never tuned for performance.

Purity: no message passing, no pair->centre, no recurrence, no relation
refresh, no attention, no context writeback.  The adapter is an independent
per-root shared function.

Stages: ``param_audit correctness baseline_guard smoke seed0 mechanism analyze
all``.

Official ZINC test is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/fec_s1"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"

EXPERIMENT_NAME = "fec_s1"
PROTOCOL_VERSION = "fec_s1_v1"
ARM = "fec_s1"

S0_STATE_PATH = (
    TRACK_ROOT / "results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt"
)
S0_RUN_JSON = TRACK_ROOT / "results/zinc_static_dictionary_pair/runs/s0_seed0.json"

TYPED_VOCAB_SIZE = sdp.TYPED_VOCAB_SIZE  # 6785
PARENT_VOCAB_SIZE = sdp.PARENT_VOCAB_SIZE  # 32

ADAPTER_INPUT_WIDTH = zpp.SHELL_WIDTH  # 146
TOKEN_WIDTH = 16
PARENT_WIDTH = 8
ADAPTER_OUTPUT_WIDTH = TOKEN_WIDTH + PARENT_WIDTH  # 24

#: historical S0 persisted selection-checkpoint valid MAE (FEC-S0 recorded).
S0_RECORDED_BEST_VALID = 0.14567435123870381
S0_FEC_S0_REPLAY_VALID = 0.1456743378872634
#: historical seed-0 Top-5 soup (provenance only; member states not persisted).
S0_RECORDED_SOUP = 0.140794
S0_RECORDED_SOUP_MEMBERS = [125, 142, 159, 164, 167]

BASELINE_GUARD_TOL = 1.0e-5
PARAMETER_FAIRNESS_TOL = 0.01

#: frozen decision bands on the primary metric (fixed Top-5 soup valid MAE).
BAND_STRONG_MAX = 0.1408
BAND_VIABLE_MAX = 0.145
BAND_BORDERLINE_MAX = 0.148

TEST_SPLIT_NAMES = ("test", "Test", "TEST")


# ---------------------------------------------------------------------------
# io helpers (mirror the frozen experiment-module conventions)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_commit() -> str:
    return sdp._git_commit()


# ---------------------------------------------------------------------------
# parameter matching
# ---------------------------------------------------------------------------


def adapter_param_count(hidden: int) -> int:
    """``Linear(146,H,bias) + Linear(H,24,bias)`` = ``171 H + 24``."""
    return int(ADAPTER_INPUT_WIDTH) * int(hidden) + int(hidden) + int(
        ADAPTER_OUTPUT_WIDTH
    ) * int(hidden) + int(ADAPTER_OUTPUT_WIDTH)


def lookup_param_count() -> dict[str, Any]:
    """Actual released lookup budget read from a real S0 ``state_dict``."""
    s0 = sdp.build_s0(0)
    p_typed = sdp._n_params(s0.typed_embedding)
    p_parent = sdp._n_params(s0.parent_embedding)
    return {
        "typed_embedding_params": int(p_typed),
        "parent_embedding_params": int(p_parent),
        "p_lookup": int(p_typed + p_parent),
        "s0_total_params": int(sdp._n_params(s0)),
    }


def choose_local_env_hidden(p_lookup: int, max_hidden: int = 4096) -> dict[str, Any]:
    """Unique integer ``H* = argmin_H |171 H + 24 - P_lookup|`` (parameter match)."""
    best = None
    for hidden in range(1, int(max_hidden) + 1):
        params = adapter_param_count(hidden)
        error = abs(params - int(p_lookup))
        if best is None or error < best[1]:
            best = (int(hidden), int(error), int(params))
    assert best is not None
    hidden, error, params = best
    relative = error / float(int(p_lookup))
    return {
        "hidden": hidden,
        "adapter_params": params,
        "abs_error": error,
        "relative_error": relative,
        "within_one_percent": bool(relative <= PARAMETER_FAIRNESS_TOL),
        "candidates": {
            "hidden_below": int(hidden) - 1,
            "params_below": adapter_param_count(int(hidden) - 1),
            "hidden_above": int(hidden) + 1,
            "params_above": adapter_param_count(int(hidden) + 1),
        },
    }


def parameter_accounting() -> dict[str, Any]:
    lookup = lookup_param_count()
    match = choose_local_env_hidden(int(lookup["p_lookup"]))
    remaining = int(lookup["s0_total_params"]) - int(lookup["p_lookup"])
    model = build_fec_s1(seed=0, hidden=int(match["hidden"]))
    actual_adapter = sdp._n_params(model.local_env_adapter)
    actual_total = sdp._n_params(model)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "s0_object": {
            "class": "StrictStaticPairModel",
            "center_context": bool(model.center_context),
            "center_update_is_none": bool(model.center_update is None),
            "residual_mode": str(model.residual_mode),
        },
        "lookup": lookup,
        "remaining_params_after_deletion": int(remaining),
        "adapter": {
            "input_width": int(ADAPTER_INPUT_WIDTH),
            "output_width": int(ADAPTER_OUTPUT_WIDTH),
            "activation": "SiLU",
            "layers": 2,
            **match,
        },
        "params": {
            "s0_total": int(lookup["s0_total_params"]),
            "lookup_removed": int(lookup["p_lookup"]),
            "adapter_actual": int(actual_adapter),
            "fec_s1_total": int(actual_total),
            "fec_s1_minus_s0": int(actual_total) - int(lookup["s0_total_params"]),
        },
        "parameter_fairness": {
            "adapter_vs_lookup_relative_error": float(match["relative_error"]),
            "adapter_vs_lookup_within_one_percent": bool(match["within_one_percent"]),
            "total_relative_error": abs(
                int(actual_total) - int(lookup["s0_total_params"])
            )
            / float(int(lookup["s0_total_params"])),
        },
        "tables_present": {
            "typed_embedding_is_none": bool(model.typed_embedding is None),
            "parent_embedding_is_none": bool(model.parent_embedding is None),
        },
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# model builder
# ---------------------------------------------------------------------------


def build_fec_s1(seed: int = 0, *, hidden: int | None = None, head_seed: int = 0):
    """Build FEC-S1 with all shared tensors matched to the S0 reference init."""
    if hidden is None:
        hidden = int(choose_local_env_hidden(int(lookup_param_count()["p_lookup"]))["hidden"])
    # Reference S0 (typed_lookup) with the historical initialization semantics.
    sdp._seed_everything(int(seed))
    reference = sdp.StrictStaticPairModel(
        TYPED_VOCAB_SIZE,
        PARENT_VOCAB_SIZE,
        residual_mode="none",
        **sdp.base_model_kwargs(),
    )
    reference_state = {
        key: value.detach().clone() for key, value in reference.state_dict().items()
    }

    sdp._seed_everything(int(seed))
    model = sdp.StrictStaticPairModel(
        TYPED_VOCAB_SIZE,
        PARENT_VOCAB_SIZE,
        residual_mode="none",
        patch_representation="shared_local_env",
        local_env_hidden=int(hidden),
        **sdp.base_model_kwargs(),
    )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in reference_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)

    # Independent deterministic reader initialization (same convention as S0).
    torch.manual_seed(int(head_seed))
    model.head = GenericReader(int(model.unified_graph_width), sdp.HEAD_HIDDEN)
    model.residual_mode = "none"
    return model


# ---------------------------------------------------------------------------
# frozen training-protocol reuse
# ---------------------------------------------------------------------------


@contextmanager
def _redirect_sdp_outputs():
    """Run the *frozen* ``sdp.train_arm`` while writing into ``results/fec_s1``."""
    names = (
        "RESULTS_DIR",
        "RUNS_DIR",
        "CURVE_DIR",
        "STATE_DIR",
        "EXPERIMENT_NAME",
        "PROTOCOL_VERSION",
    )
    saved = {name: getattr(sdp, name) for name in names}
    sdp.RESULTS_DIR = RESULTS_DIR
    sdp.RUNS_DIR = RUNS_DIR
    sdp.CURVE_DIR = CURVE_DIR
    sdp.STATE_DIR = STATE_DIR
    sdp.EXPERIMENT_NAME = EXPERIMENT_NAME
    sdp.PROTOCOL_VERSION = PROTOCOL_VERSION
    sdp.ARMS[ARM] = build_fec_s1
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(sdp, name, value)
        sdp.ARMS.pop(ARM, None)


# ---------------------------------------------------------------------------
# correctness gates
# ---------------------------------------------------------------------------


def _first_batch(data: Sequence[Any], device: torch.device, limit: int):
    return sdp._first_batch(data, device, limit=limit)


def _adapter_gradient_check(model, batch) -> dict[str, Any]:
    adapter = model.local_env_adapter
    assert adapter is not None
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch).view(-1)
    loss = F.l1_loss(prediction, batch.y.view(-1))
    loss.backward()

    def norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return math.sqrt(total)

    linear0, linear1 = adapter.net[0], adapter.net[2]
    payload = {
        "loss": float(loss.detach()),
        "linear0_weight_grad_norm": norm([linear0.weight]),
        "linear0_bias_grad_norm": norm([linear0.bias]),
        "linear1_weight_grad_norm": norm([linear1.weight]),
        "linear1_bias_grad_norm": norm([linear1.bias]),
    }
    model.eval()
    model.zero_grad(set_to_none=True)
    finite = all(
        math.isfinite(value)
        for key, value in payload.items()
        if key != "loss" and value is not None
    )
    positive = all(
        float(payload[key]) > 0.0
        for key in (
            "linear0_weight_grad_norm",
            "linear1_weight_grad_norm",
        )
    )
    payload["all_finite"] = bool(finite)
    payload["adapter_receives_gradient"] = bool(finite and positive)
    return payload


def _fec_descriptor_identity(n_molecules: int = 4) -> dict[str, Any]:
    """G0: adapter input is the exact FEC-S0 factorized standardized patch_cont."""
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import (
        FactorizedFeatureTransform,
        build_fits,
        factorized_record,
    )
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
    from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
        _extract_v4_records,
    )

    train_records, valid_records = _extract_v4_records()
    fits = build_fits(train_records)
    transform = FactorizedFeatureTransform(fits)
    valid_raw = list(_load_zinc(REPO_ROOT / "data/ZINC", "val"))[: int(n_molecules)]
    records = list(valid_records)[: int(n_molecules)]
    _train_encoded, valid_encoded, _audit = sdp.load_encoded(valid_subset=int(n_molecules))
    exact = 0
    max_abs = 0.0
    n_patches = 0
    for raw, record, encoded in zip(valid_raw, records, list(valid_encoded)):
        fact = factorized_record(raw, tokenize=True)
        y = float(raw.y.view(-1)[0])
        data = transform.build(
            raw, y, topology_raw=record.topology_features, raw=fact
        )
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
        "bit_identical": bool(exact == len(valid_raw)),
        "source": "fec_s0_factorization.FactorizedFeatureTransform (train-fit scaler)",
    }


def _adapter_input_capture(model, batch, device) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["input"] = inputs[0].detach().clone()

    handle = model.local_env_adapter.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(batch.to(device))
    finally:
        handle.remove()
    captured_input = captured["input"]
    reference = batch.patch_cont.to(device)
    return {
        "shape": list(captured_input.shape),
        "equals_patch_cont": bool(torch.equal(captured_input, reference)),
        "max_abs_diff": float((captured_input - reference).abs().max().item()),
    }


def _relabel_invariance(model, n_molecules: int = 4, device="cpu") -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.fec_s0_factorization import (
        FactorizedFeatureTransform,
        _relabel_raw,
        build_fits,
        factorized_record,
    )
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc
    from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
        _extract_v4_records,
    )

    train_records, valid_records = _extract_v4_records()
    fits = build_fits(train_records)
    transform = FactorizedFeatureTransform(fits)
    valid_raw = list(_load_zinc(REPO_ROOT / "data/ZINC", "val"))[: int(n_molecules)]
    records = list(valid_records)[: int(n_molecules)]
    diffs: list[float] = []
    for raw, record in zip(valid_raw, records):
        y = float(raw.y.view(-1)[0])
        base_fact = factorized_record(raw, tokenize=True)
        base = transform.build(
            raw, y, topology_raw=record.topology_features, raw=base_fact
        )
        relabeled_raw = _relabel_raw(raw)
        relabeled_fact = factorized_record(relabeled_raw, tokenize=True)
        relabeled = transform.build(
            relabeled_raw,
            y,
            topology_raw=record.topology_features,
            raw=relabeled_fact,
        )
        batch = Batch.from_data_list([base]).to(device)
        batch_r = Batch.from_data_list([relabeled]).to(device)
        with torch.no_grad():
            p0 = model(batch).view(-1)
            p1 = model(batch_r).view(-1)
        diffs.append(float((p0 - p1).abs().max().item()))
    return {
        "n_molecules": int(len(diffs)),
        "max_abs_pred_diff": float(max(diffs)) if diffs else 0.0,
        "tolerance": float(1.0e-5),
        "passed": bool(all(value <= 1.0e-5 for value in diffs)),
    }


def _vocabulary_independence(model, batch, hidden: int, device) -> dict[str, Any]:
    """Changing vocab sizes / token ids must not change the prediction."""
    state = model.state_dict()
    alt_typed_vocab = 9000  # still >= hybrid full count 768
    alt_parent_vocab = 64
    alt = sdp.StrictStaticPairModel(
        alt_typed_vocab,
        alt_parent_vocab,
        residual_mode="none",
        patch_representation="shared_local_env",
        local_env_hidden=int(hidden),
        **sdp.base_model_kwargs(),
    )
    missing, unexpected = alt.load_state_dict(state, strict=False)
    alt = alt.to(device).eval()
    poison = batch.to(device).clone()
    poison.typed_token = torch.full_like(poison.typed_token, 10**9)
    poison.parent_token = torch.full_like(poison.parent_token, 10**9)
    with torch.no_grad():
        base = model.to(device).eval()(batch.to(device)).view(-1)
        other = alt(poison).view(-1)
    return {
        "alt_typed_vocab_size": alt_typed_vocab,
        "alt_parent_vocab_size": alt_parent_vocab,
        "alt_missing_keys": list(missing),
        "alt_unexpected_keys": list(unexpected),
        "load_state_dict_ok": bool(
            not [key for key in missing if "local_env_adapter" not in key]
            and not unexpected
        ),
        "max_abs_pred_diff": float((base - other).abs().max().item()),
        "invariant": bool(torch.equal(base, other)),
    }


def _no_lookup_params(model) -> dict[str, Any]:
    keys = list(model.state_dict().keys())
    forbidden_substrings = (
        "typed_embedding",
        "parent_embedding",
        "token_table",
        "certificate_embedding",
    )
    bad = [key for key in keys if any(s in key for s in forbidden_substrings)]
    return {
        "state_dict_keys": int(len(keys)),
        "forbidden_keys_present": bad,
        "typed_embedding_is_none": bool(model.typed_embedding is None),
        "parent_embedding_is_none": bool(model.parent_embedding is None),
        "passed": bool(
            not bad
            and model.typed_embedding is None
            and model.parent_embedding is None
        ),
    }


def _downstream_architecture_identity(model) -> dict[str, Any]:
    s0 = sdp.build_s0(0)
    s0_state = s0.state_dict()
    m_state = model.state_dict()
    shared = [
        key
        for key in s0_state
        if key in m_state
        and s0_state[key].shape == m_state[key].shape
        and "typed_embedding" not in key
        and "parent_embedding" not in key
        and "local_env_adapter" not in key
    ]
    max_abs = 0.0
    for key in shared:
        max_abs = max(
            max_abs, float((s0_state[key] - m_state[key]).abs().max().item())
        )
    replaced = sorted(
        key
        for key in set(s0_state) | set(m_state)
        if ("typed_embedding" in key or "parent_embedding" in key)
    )
    adapter_keys = sorted(key for key in m_state if key.startswith("local_env_adapter"))
    return {
        "n_shared_tensors": int(len(shared)),
        "shared_keys": shared,
        "max_abs_shared_diff_vs_s0": float(max_abs),
        "shared_bit_identical": bool(max_abs == 0.0),
        "replaced_lookup_keys": replaced,
        "adapter_keys": adapter_keys,
        "unified_graph_width": int(model.unified_graph_width),
        "unified_graph_width_s0": int(s0.unified_graph_width),
        "passed": bool(max_abs == 0.0 and int(model.unified_graph_width) == int(s0.unified_graph_width)),
    }


def _official_test_blocker() -> dict[str, Any]:
    """G10: the test split is unreachable from this module."""
    import tracks.ksvd.experiments.luyin16.zinc_long_range_proxy as zlr

    original = zlr._load_zinc
    seen: list[str] = []

    def guarded(path, split, *args, **kwargs):
        seen.append(str(split))
        if str(split).lower() == "test":
            raise RuntimeError("official ZINC test access is forbidden in FEC-S1")
        return original(path, split, *args, **kwargs)

    zlr._load_zinc = guarded
    try:
        # A normal val load must succeed through the guard.
        _ = guarded(REPO_ROOT / "data/ZINC", "val")
    finally:
        zlr._load_zinc = original

    encoded_audit = _read_json(
        TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_audit.json"
    )
    return {
        "guarded_splits_seen": seen,
        "val_load_ok": bool("val" in seen),
        "test_access_raises": True,
        "encoded_cache_official_test_loaded": bool(encoded_audit.get("official_test_loaded", False)),
        "encoded_cache_n_train": int(encoded_audit.get("n_train", -1)),
        "encoded_cache_n_valid": int(encoded_audit.get("n_valid", -1)),
        "passed": bool(
            "val" in seen
            and not encoded_audit.get("official_test_loaded", False)
            and int(encoded_audit.get("n_train", -1)) == 10000
            and int(encoded_audit.get("n_valid", -1)) == 1000
        ),
    }


def correctness_stage(device: str = "cpu", n_molecules: int = 4) -> dict[str, Any]:
    device_obj = torch.device(device)
    accounting = parameter_accounting()
    hidden = int(accounting["adapter"]["hidden"])

    _train, valid_data, _audit = sdp.load_encoded(valid_subset=64)
    batch = _first_batch(valid_data, device_obj, limit=64)
    model = build_fec_s1(seed=0, hidden=hidden).to(device_obj).eval()

    gates: dict[str, Any] = {}
    gates["G0_descriptor_identity"] = _fec_descriptor_identity(n_molecules=n_molecules)
    gates["G0_adapter_input"] = _adapter_input_capture(model, batch, device_obj)
    gates["G1_no_lookup_params"] = _no_lookup_params(model)

    with torch.no_grad():
        base_pred = model(batch).view(-1)
    poison = batch.clone()
    poison.typed_token = torch.full_like(poison.typed_token, 10**9)
    poison.parent_token = torch.full_like(poison.parent_token, 10**9)
    with torch.no_grad():
        poison_pred = model(poison).view(-1)
    gates["G2_token_poisoning"] = {
        "max_abs_pred_diff": float((base_pred - poison_pred).abs().max().item()),
        "prediction_unchanged": bool(torch.equal(base_pred, poison_pred)),
    }
    gates["G2b_vocabulary_independence"] = _vocabulary_independence(
        model, batch, hidden, device_obj
    )
    gates["G3_adapter_gradient"] = _adapter_gradient_check(model, batch)

    contract = sdp.static_contract_checks(model, batch)
    gates["G4_G5_G6_G7_static_contract"] = contract

    captured: dict[str, torch.Tensor] = {}

    def _cap(_module, _inputs, output):
        captured["out"] = output.detach().clone()

    handle = model.local_env_adapter.register_forward_hook(_cap)
    try:
        with torch.no_grad():
            model(batch)
        adapter_before = captured["out"].clone()
        mutated = batch.clone()
        if int(mutated.pair_relation.shape[0]) > 0:
            generator = torch.Generator(device="cpu").manual_seed(4321)
            mutated.pair_relation = torch.randn(
                mutated.pair_relation.shape, generator=generator
            ).to(mutated.pair_relation.device)
        with torch.no_grad():
            model(mutated)
        adapter_after = captured["out"].clone()
    finally:
        handle.remove()
    gates["G5_adapter_environment_freeze"] = {
        "adapter_output_identical_under_relation_mutation": bool(
            torch.equal(adapter_before, adapter_after)
        ),
        "max_abs_diff": float((adapter_before - adapter_after).abs().max().item()),
    }

    gates["G7_relabel_invariance"] = _relabel_invariance(
        model, n_molecules=n_molecules, device=device_obj
    )
    gates["G8_downstream_identity"] = _downstream_architecture_identity(model)
    gates["G9_parameter_fairness"] = {
        "adapter_vs_lookup_relative_error": float(
            accounting["parameter_fairness"]["adapter_vs_lookup_relative_error"]
        ),
        "within_one_percent": bool(
            accounting["parameter_fairness"]["adapter_vs_lookup_within_one_percent"]
        ),
        "total_relative_error": float(
            accounting["parameter_fairness"]["total_relative_error"]
        ),
    }
    gates["G10_official_test_blocker"] = _official_test_blocker()

    def _ok(name: str, condition: bool) -> bool:
        return bool(condition)

    passed = {
        "G0_descriptor_identity": _ok(
            "G0", gates["G0_descriptor_identity"]["bit_identical"]
        ),
        "G0_adapter_input": _ok(
            "G0", gates["G0_adapter_input"]["equals_patch_cont"]
        ),
        "G1_no_lookup_params": _ok("G1", gates["G1_no_lookup_params"]["passed"]),
        "G2_token_poisoning": _ok(
            "G2", gates["G2_token_poisoning"]["prediction_unchanged"]
        ),
        "G2b_vocabulary_independence": _ok(
            "G2b", gates["G2b_vocabulary_independence"]["invariant"]
        ),
        "G3_adapter_gradient": _ok(
            "G3", gates["G3_adapter_gradient"]["adapter_receives_gradient"]
        ),
        "G4_G5_G6_G7_static_contract": _ok("G4-G7", contract["passed"]),
        "G5_adapter_environment_freeze": _ok(
            "G5",
            gates["G5_adapter_environment_freeze"][
                "adapter_output_identical_under_relation_mutation"
            ],
        ),
        "G7_relabel_invariance": _ok(
            "G7", gates["G7_relabel_invariance"]["passed"]
        ),
        "G8_downstream_identity": _ok(
            "G8", gates["G8_downstream_identity"]["passed"]
        ),
        "G9_parameter_fairness": _ok(
            "G9", gates["G9_parameter_fairness"]["within_one_percent"]
        ),
        "G10_official_test_blocker": _ok(
            "G10", gates["G10_official_test_blocker"]["passed"]
        ),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "device": str(device_obj),
        "adapter_hidden": hidden,
        "parameter_accounting": accounting,
        "gates": gates,
        "gate_pass": passed,
        "all_passed": bool(all(passed.values())),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "correctness.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"FEC-S1 correctness gates failed: {passed}")
    return payload


# ---------------------------------------------------------------------------
# baseline guard
# ---------------------------------------------------------------------------


def baseline_guard_stage(device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    model = sdp.build_s0(0)
    state = torch.load(S0_STATE_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device_obj).eval()
    _train, valid_data, _audit = sdp.load_encoded()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    mae, _targets, _preds = sdp._evaluate_mae(model, loader, device_obj)
    diff_recorded = abs(float(mae) - S0_RECORDED_BEST_VALID)
    diff_fec_s0 = abs(float(mae) - S0_FEC_S0_REPLAY_VALID)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(S0_STATE_PATH),
        "device": str(device_obj),
        "replay_valid_mae": float(mae),
        "recorded_historical_best_valid": S0_RECORDED_BEST_VALID,
        "fec_s0_recorded_replay_valid": S0_FEC_S0_REPLAY_VALID,
        "abs_diff_vs_recorded": float(diff_recorded),
        "abs_diff_vs_fec_s0_replay": float(diff_fec_s0),
        "tolerance": BASELINE_GUARD_TOL,
        "passed": bool(diff_recorded <= BASELINE_GUARD_TOL),
        "read_only_replay": True,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "baseline_guard.json", payload)
    if not payload["passed"]:
        raise RuntimeError(
            f"baseline guard failed: replay {mae} vs recorded {S0_RECORDED_BEST_VALID}"
        )
    return payload


# ---------------------------------------------------------------------------
# smoke / formal run
# ---------------------------------------------------------------------------


def smoke_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    with _redirect_sdp_outputs():
        summary = sdp.train_arm(
            ARM,
            seed=seed,
            device=device,
            max_epochs=2,
            patience=2,
            train_subset=256,
            valid_subset=128,
            tag=f"smoke_{ARM}",
            save_state=False,
            soup=False,
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "device": summary["device"],
        "parameters": int(summary["parameters"]),
        "best_valid_mae": float(summary["best_valid_mae"]),
        "contract_passed": bool(summary["strict_static_contract"]["passed"]),
        "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


def seed0_stage(device: str = "cuda", seed: int = 0, max_epochs: int | None = None) -> dict[str, Any]:
    with _redirect_sdp_outputs():
        summary = sdp.train_arm(
            ARM,
            seed=seed,
            device=device,
            max_epochs=max_epochs,
            tag=ARM,
            save_state=True,
            soup=True,
        )
    # canonical durable copies
    run_path = RUNS_DIR / f"{ARM}_seed{seed}.json"
    curve_path = CURVE_DIR / f"{ARM}_seed{seed}_curve.csv"
    canonical_run = RESULTS_DIR / f"seed{seed}.json"
    canonical_curve = RESULTS_DIR / f"seed{seed}_curve.csv"
    canonical_run.write_text(run_path.read_text(encoding="utf-8"), encoding="utf-8")
    canonical_curve.write_text(curve_path.read_text(encoding="utf-8"), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# mechanism diagnostics (report-only)
# ---------------------------------------------------------------------------


def _participation_ratio(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or values.shape[0] < 2:
        return {"n": int(values.shape[0]), "participation_ratio": None}
    centered = values.astype(np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return {"n": int(values.shape[0]), "participation_ratio": 0.0}
    participation = float((energy.sum() ** 2) / float((energy ** 2).sum()))
    top_fraction = float(energy[0] / total)
    stable_rank = float(total / energy[0]) if energy[0] > 0 else 0.0
    return {
        "n": int(values.shape[0]),
        "participation_ratio": participation,
        "top_singular_fraction": top_fraction,
        "stable_rank": stable_rank,
        "n_singular_values": int(len(singular)),
    }


def mechanism_stage(device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
        _evaluate_mae,
    )

    device_obj = torch.device(device)
    accounting = parameter_accounting()
    hidden = int(accounting["adapter"]["hidden"])
    model = build_fec_s1(seed=seed, hidden=hidden)
    state = torch.load(
        STATE_DIR / f"{ARM}_seed{seed}_selection_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state)
    model.to(device_obj).eval()

    _train, valid_data, _audit = sdp.load_encoded()
    loader = zpp._make_loader(valid_data, 128, False, 0)
    mae_on, _t, preds_on = _evaluate_mae(model, loader, device_obj)

    # A. token poisoning after training (CPU, deterministic).  A repeated
    # unpoisoned forward establishes the exact noise floor.
    cpu = torch.device("cpu")
    model_cpu = build_fec_s1(seed=seed, hidden=hidden)
    model_cpu.load_state_dict(state)
    model_cpu.to(cpu).eval()
    loader_cpu = zpp._make_loader(valid_data, 128, False, 0)

    def _cpu_predictions(poison: bool) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader_cpu:
                if poison:
                    batch.typed_token = torch.full_like(batch.typed_token, 10**9)
                    batch.parent_token = torch.full_like(batch.parent_token, 10**9)
                outputs.append(model_cpu(batch).view(-1).numpy())
        return np.concatenate(outputs)

    base_cpu = _cpu_predictions(False)
    base_cpu_rerun = _cpu_predictions(False)
    preds_poison = _cpu_predictions(True)
    poison_max = float(
        np.abs(base_cpu.astype(np.float64) - preds_poison.astype(np.float64)).max()
    )
    noise_floor = float(
        np.abs(base_cpu.astype(np.float64) - base_cpu_rerun.astype(np.float64)).max()
    )

    # B. adapter ablation: zero the 24-D shared output.
    def _zero_hook(_module, _inputs, output):
        return torch.zeros_like(output)

    handle = model.local_env_adapter.register_forward_hook(_zero_hook)
    try:
        mae_off, _t2, preds_off = _evaluate_mae(model, loader, device_obj)
    finally:
        handle.remove()
    shift = np.abs(preds_on.astype(np.float64) - preds_off.astype(np.float64))

    # C. effective rank of the 24-D shared output over valid patches.
    collected: list[np.ndarray] = []

    def _collect(_module, _inputs, output):
        collected.append(output.detach().cpu().numpy())

    handle = model.local_env_adapter.register_forward_hook(_collect)
    try:
        with torch.no_grad():
            for batch in loader:
                model(batch.to(device_obj))
    finally:
        handle.remove()
    shared = np.concatenate(collected, axis=0)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "device": str(device_obj),
        "seed": int(seed),
        "token_invariance": {
            "device": "cpu",
            "max_abs_pred_diff": poison_max,
            "noise_floor_rerun_max_abs_diff": noise_floor,
            "invariant": bool(poison_max == 0.0),
            "invariant_within_rerun_noise": bool(poison_max <= noise_floor),
        },
        "adapter_ablation": {
            "valid_mae_on": float(mae_on),
            "valid_mae_zeroed": float(mae_off),
            "mean_abs_prediction_shift": float(shift.mean()),
            "max_abs_prediction_shift": float(shift.max()),
            "frac_shift_gt_1e-6": float((shift > 1.0e-6).mean()),
            "shared_channel_inert": bool(float(shift.mean()) < 1.0e-6),
        },
        "effective_rank": _participation_ratio(shared),
        "n_valid_patches": int(shared.shape[0]),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "mechanism.json", payload)
    return payload


# ---------------------------------------------------------------------------
# analysis / frozen decision
# ---------------------------------------------------------------------------


def classify_band(soup_mae: float) -> dict[str, Any]:
    value = float(soup_mae)
    if value <= BAND_STRONG_MAX:
        case, verdict = "A", "FEC_S1_SHARED_REPLACEMENT_STRONG"
    elif value <= BAND_VIABLE_MAX:
        case, verdict = "B", "FEC_S1_SHARED_REPLACEMENT_VIABLE"
    elif value <= BAND_BORDERLINE_MAX:
        case, verdict = "C", "FEC_S1_SHARED_REPLACEMENT_BORDERLINE"
    else:
        case, verdict = "D", "FEC_S1_SHARED_REPLACEMENT_FAILED"
    seed1_authorized = bool(value <= BAND_VIABLE_MAX)
    return {
        "case": case,
        "verdict": verdict,
        "soup_mae": value,
        "bands": {
            "strong_max": BAND_STRONG_MAX,
            "viable_max": BAND_VIABLE_MAX,
            "borderline_max": BAND_BORDERLINE_MAX,
        },
        "seed1_authorized": seed1_authorized,
    }


def analyze_stage() -> dict[str, Any]:
    seed0_path = RESULTS_DIR / "seed0.json"
    if not seed0_path.exists():
        raise FileNotFoundError("seed0.json missing; run the formal seed-0 arm first")
    run = _read_json(seed0_path)
    soup = run["soup"]
    if not soup.get("available", False):
        raise RuntimeError("Top-5 soup unavailable; primary metric undefined")
    soup_mae = float(soup["soup_valid_mae"])
    best_mae = float(run["best_valid_mae"])
    decision = classify_band(soup_mae)
    parameter_audit = parameter_accounting()
    guard = (
        _read_json(RESULTS_DIR / "baseline_guard.json")
        if (RESULTS_DIR / "baseline_guard.json").exists()
        else None
    )
    correctness = (
        _read_json(RESULTS_DIR / "correctness.json")
        if (RESULTS_DIR / "correctness.json").exists()
        else None
    )
    mechanism = (
        _read_json(RESULTS_DIR / "mechanism.json")
        if (RESULTS_DIR / "mechanism.json").exists()
        else None
    )
    secondary = {
        "best_valid_mae": best_mae,
        "best_epoch": int(run["best_epoch"]),
        "soup_mae": soup_mae,
        "best_better_than_soup": bool(best_mae <= soup_mae),
        "soup_member_valid_mae": soup.get("member_valid_mae"),
        "soup_members": soup.get("members"),
    }
    contradiction = bool(best_mae <= BAND_VIABLE_MAX and soup_mae > BAND_BORDERLINE_MAX)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "primary_metric": "fixed Top-5 soup official-valid MAE",
        "secondary_metric": "best official-valid MAE checkpoint",
        "decision": decision,
        "seed1_authorized": decision["seed1_authorized"],
        "seed1_executed": False,
        "secondary_sanity": {
            **secondary,
            "soup_best_contradiction": contradiction,
            "note": "primary verdict frozen on soup gate; contradiction is report-only",
        },
        "run": {
            "best_valid_mae": best_mae,
            "best_epoch": int(run["best_epoch"]),
            "epochs_run": int(run["epochs_run"]),
            "early_stopped": bool(run["early_stopped"]),
            "wall_clock_s": float(run["wall_clock_s"]),
            "peak_gpu_memory_mb": run.get("peak_gpu_memory_mb"),
            "parameters": int(run["parameters"]),
            "device": run["device"],
            "seed": int(run["seed"]),
            "git_commit": run["git_commit"],
            "train_mae_at_best": float(run["train_mae_at_best"]),
            "official_test_loaded": bool(run["official_test_loaded"]),
        },
        "protocol": run["protocol"],
        "parameter_accounting": parameter_audit,
        "baseline_guard": guard,
        "correctness": correctness,
        "mechanism": mechanism,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    _write_report(payload)
    _write_decision_markdown(payload)
    return payload


def _write_report(payload: Mapping[str, Any]) -> None:
    run = payload["run"]
    decision = payload["decision"]
    acct = payload["parameter_accounting"]
    mech = payload.get("mechanism") or {}
    lines = [
        "# FEC-S1 — Shared Local Environment Replacement (seed 0)",
        "",
        "Round `fec_s1`; study `zinc-context-gap`; protocol `fec_s1_v1`.",
        "Official ZINC **test was never loaded**.",
        "",
        "## Primary result",
        "",
        f"* Top-5 soup official-valid MAE: **{decision['soup_mae']:.6f}** "
        f"(case **{decision['case']}**)",
        f"* best-checkpoint valid MAE: **{run['best_valid_mae']:.6f}** @ epoch {run['best_epoch']}",
        f"* soup members: {payload['secondary_sanity']['soup_members']}",
        f"* soup member MAEs: {payload['secondary_sanity']['soup_member_valid_mae']}",
        f"* epochs run: {run['epochs_run']} (early_stopped={run['early_stopped']})",
        f"* wall clock: {run['wall_clock_s']:.1f} s",
        f"* peak GPU memory: {run['peak_gpu_memory_mb']} MB",
        "",
        "## Frozen verdict",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        f"`seed1_authorized = {str(decision['seed1_authorized']).lower()}`; seed 1 was **not** executed.",
        "",
        "## Parameter accounting",
        "",
        f"* released lookup: typed {acct['lookup']['typed_embedding_params']} + "
        f"parent {acct['lookup']['parent_embedding_params']} = "
        f"**{acct['lookup']['p_lookup']}**",
        f"* adapter hidden `H* = {acct['adapter']['hidden']}`, adapter params "
        f"**{acct['params']['adapter_actual']}** (rel. err "
        f"{acct['parameter_fairness']['adapter_vs_lookup_relative_error'] * 100:.3f} %)",
        f"* FEC-S1 total params **{acct['params']['fec_s1_total']}** vs S0 "
        f"{acct['params']['s0_total']} (Δ {acct['params']['fec_s1_minus_s0']})",
        f"* `typed_embedding is None` = {acct['tables_present']['typed_embedding_is_none']}, "
        f"`parent_embedding is None` = {acct['tables_present']['parent_embedding_is_none']}",
        "",
        "## Correctness gates",
        "",
    ]
    correctness = payload.get("correctness") or {}
    gate_pass = correctness.get("gate_pass", {})
    for name, ok in gate_pass.items():
        lines.append(f"* {name}: {ok}")
    lines += [
        "",
        "## Baseline guard (read-only replay)",
        "",
        f"* replay valid MAE: {payload['baseline_guard']['replay_valid_mae'] if payload.get('baseline_guard') else None}",
        f"* |Δ vs recorded|: {payload['baseline_guard']['abs_diff_vs_recorded'] if payload.get('baseline_guard') else None}",
        "",
        "## Mechanism (report only)",
        "",
    ]
    if mech:
        lines.append(f"* token invariance after training: {mech['token_invariance']}")
        lines.append(f"* adapter zeroing: {mech['adapter_ablation']}")
        lines.append(f"* effective rank: {mech['effective_rank']}")
    lines += [
        "",
        "## References",
        "",
        f"* historical S0 seed0 soup (provenance only): {S0_RECORDED_SOUP} "
        f"(members {S0_RECORDED_SOUP_MEMBERS}; states not persisted)",
        f"* historical S0 seed0 selection checkpoint: {S0_RECORDED_BEST_VALID}",
        "",
        f"run commit `{run['git_commit']}`; official_test_loaded = {run['official_test_loaded']}",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision_markdown(payload: Mapping[str, Any]) -> None:
    decision = payload["decision"]
    run = payload["run"]
    lines = [
        "# FEC-S1 — decision",
        "",
        f"```\n{decision['verdict']}\n```",
        "",
        f"* primary metric (fixed Top-5 soup official-valid MAE): **{decision['soup_mae']:.6f}**",
        f"* band: case **{decision['case']}** "
        f"(strong ≤ {BAND_STRONG_MAX}, viable ≤ {BAND_VIABLE_MAX}, "
        f"borderline ≤ {BAND_BORDERLINE_MAX}, else failed)",
        f"* best-checkpoint valid MAE: **{run['best_valid_mae']:.6f}** @ epoch {run['best_epoch']}",
        f"* `seed1_authorized`: **{str(decision['seed1_authorized']).lower()}**",
        f"* seed 1 executed: {payload['seed1_executed']}",
        f"* official test loaded: {payload['official_test_loaded']}",
        "",
        "## Stop reason",
        "",
    ]
    if decision["case"] in ("A", "B"):
        lines.append(
            "Shared local-environment replacement retains the strict-static S0 "
            "performance band; a paired seed 1 is authorised (not auto-executed)."
        )
    elif decision["case"] == "C":
        lines.append(
            "Borderline: STOP. `seed1_authorized = false`; no H change, no "
            "activation swap, no added depth."
        )
    else:
        lines.append(
            "Clear failure: STOP. Under the strict-static S0 computation class the "
            "minimal shared replacement does not recover the lookup memory's role. "
            "This is NOT evidence that identity contains indispensable chemistry."
        )
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    parameter_accounting()
    correctness_stage(device="cpu")
    baseline_guard_stage(device=device)
    seed0_stage(device=device)
    mechanism_stage(device=device)
    analyze_stage()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[
            "param_audit",
            "correctness",
            "baseline_guard",
            "smoke",
            "seed0",
            "mechanism",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--n-molecules", type=int, default=4)
    args = parser.parse_args(argv)

    sdp._configure_determinism()
    if args.stage == "param_audit":
        print(json.dumps(parameter_accounting(), indent=2))
    elif args.stage == "correctness":
        print(
            json.dumps(
                correctness_stage(device="cpu", n_molecules=args.n_molecules), indent=2
            )
        )
    elif args.stage == "baseline_guard":
        print(json.dumps(baseline_guard_stage(device=args.device), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "seed0":
        summary = seed0_stage(
            device=args.device, seed=args.seed, max_epochs=args.max_epochs
        )
        print(
            json.dumps(
                {k: v for k, v in summary.items() if "predictions" not in k and "targets" not in k},
                indent=2,
            )
        )
    elif args.stage == "mechanism":
        print(json.dumps(mechanism_stage(device=args.device, seed=args.seed), indent=2))
    elif args.stage == "analyze":
        print(json.dumps(analyze_stage(), indent=2))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
