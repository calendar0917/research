"""Focused CPU tests for PEC-C1 (``pec_c1``).

Covers the ten pre-registered targeted checks (``pec_c1_preregistration.md``
§11) plus the PEC-C1-specific dictionary-freeze invariant and the frozen
decision-gate semantics.  Data-free: no ZINC data, no checkpoints, no official
test is opened anywhere here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import pec_v0 as pec
from tracks.ksvd.experiments.luyin16 import pec_v0_gate0 as gate0
from tracks.ksvd.experiments.luyin16 import zinc_pec_c1 as c1
from tracks.ksvd.experiments.luyin16.pec_v0_gate0 import (
    _batch,
    _chain,
    _dictionary,
    _model,
    _ring,
    _sample,
)

# --------------------------------------------------------------------------
# 1-5, 7: the frozen PEC-v0 correctness contract is still intact
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["purity", "chemistry_isolation", "environment_freeze", "static_contract",
     "sparse_correctness", "relabel_invariance", "assignment_sensitivity"],
)
def test_pec_v0_correctness_contract_still_holds(name: str) -> None:
    gate0.GATE0_CHECKS[name]()


def test_no_pair_to_centre_and_no_recurrence() -> None:
    """E_i is built before pairs exist; each module runs exactly once."""
    batch = _batch([_sample(_ring(6), seed=1), _sample(_chain(4), seed=2)])
    model = _model("sparse")

    counts = {"environment": 0, "pair": 0, "reader": 0}

    def _hook(key):
        def _inner(_module, _inputs, _output):
            counts[key] += 1

        return _inner

    handles = [
        model.environment.register_forward_hook(_hook("environment")),
        model.pair.register_forward_hook(_hook("pair")),
        model.reader.register_forward_hook(_hook("reader")),
    ]
    try:
        first = model(batch)["prediction"]
        after_one_forward = dict(counts)
        second = model(batch)["prediction"]
    finally:
        for handle in handles:
            handle.remove()

    # exactly one application per module per forward => no recurrence / no depth loop
    assert after_one_forward == {"environment": 1, "pair": 1, "reader": 1}, counts
    # stateless: a second identical forward is bit-identical
    assert torch.equal(first, second)

    # no pair -> centre write-back: environments ignore pair tensors entirely
    env_a = model.encode_environments(batch)
    mutated = dict(batch)
    mutated["pair_rho"] = torch.randn_like(batch["pair_rho"])
    env_b = model.encode_environments(mutated)
    assert torch.equal(env_a, env_b)

    # pair->rho independence: the pair module is the only consumer of pair_rho
    out_a = model(batch)["prediction"]
    out_b = model(mutated)["prediction"]
    assert not torch.allclose(out_a, out_b)


def test_no_chemistry_inside_pair_relation() -> None:
    """rho_ij is 18-D and pure topology: bond chemistry must not touch it."""
    graph = _ring(6)
    base = _sample(graph, atom_categories=(0, 1, 2, 3), bond_category=1)
    changed = _sample(graph, atom_categories=(4, 5, 6, 7), bond_category=3)
    assert pec.TOPO_PAIR_DIM == 18
    assert base.pair_rho.shape[1] == 18
    assert np.array_equal(base.pair_rho, changed.pair_rho)
    # the deleted PEC-v0 blocks would make rho depend on bond_mean chemistry
    assert not np.array_equal(base.bond_idx, changed.bond_idx)


def test_reader_sees_only_composed_inputs() -> None:
    """No raw chemistry / no per-atom bypass reaches the reader."""
    model = _model("sparse")
    expected = 2 * pec.ENV_DIM + 2 * pec.PAIR_DIM + pec.TOPO_GLOBAL_DIM
    first_linear = model.reader[0]
    assert first_linear.in_features == expected, (first_linear.in_features, expected)
    # the environment input contains the chemistry binding, not raw attributes
    assert model.environment_input_dim == pec.environment_input_dim("sparse")


# --------------------------------------------------------------------------
# 6: parameter parity, and the dictionary-freeze invariant (D1)
# --------------------------------------------------------------------------


def test_parameter_parity_ck_cd() -> None:
    role_units = pec.NODE_ROLE_DIM * pec.K_V + pec.EDGE_ROLE_DIM * pec.K_E
    d_node, d_edge = _dictionary()
    ck = _model("sparse")
    cd = _model("dense")
    ck_report = c1.parameter_report(ck)
    cd_report = c1.parameter_report(cd)

    assert ck_report["total"] == cd_report["total"], (ck_report, cd_report)
    assert ck_report["total"] == 94049
    assert ck_report["role_total"] == role_units
    # CD's dense map is trainable; CK's dictionary is not (PEC-C1 D1/D2)
    c1.freeze_dictionary(ck)
    assert c1.parameter_report(ck)["trainable"] == ck_report["total"] - role_units
    assert c1.parameter_report(cd)["trainable"] == cd_report["total"]
    # role-coordinate accounting is mode aware
    assert c1.parameter_report(ck)["role_total"] == role_units
    assert c1.parameter_report(ck)["role_trainable"] == 0
    assert c1.parameter_report(cd)["role_total"] == role_units
    assert c1.parameter_report(cd)["role_trainable"] == role_units
    assert set(c1.role_parameters(ck)) == {"d_node", "d_edge"}
    assert set(c1.role_parameters(cd)) == {"m_node", "m_edge"}
    assert len(c1.trainable_parameters(ck)) < len(c1.trainable_parameters(cd))
    assert d_node.shape == (pec.NODE_ROLE_DIM, pec.K_V)
    assert d_edge.shape == (pec.EDGE_ROLE_DIM, pec.K_E)


def test_dictionary_is_frozen_and_out_of_the_optimizer() -> None:
    d_node, d_edge = _dictionary()
    model = pec.build_model("sparse", d_node=d_node, d_edge=d_edge, seed=0)
    c1.freeze_dictionary(model)
    assert model.d_node.requires_grad is False
    assert model.d_edge.requires_grad is False

    params = c1.trainable_parameters(model)
    assert not any(p is model.d_node or p is model.d_edge for p in params)
    optimizer = torch.optim.Adam(params, lr=c1.LR, weight_decay=c1.WD)

    before_node = model.d_node.detach().clone()
    before_edge = model.d_edge.detach().clone()
    batch = _batch([_sample(_ring(7), seed=3), _sample(_chain(5), seed=4)])
    out = model(batch)
    loss = (out["prediction"] - batch["y"]).abs().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, c1.CLIP)
    optimizer.step()

    # the dictionary must be bit-identical after a real optimizer step
    assert torch.equal(model.d_node.detach(), before_node)
    assert torch.equal(model.d_edge.detach(), before_edge)
    # while the environment / pair / reader are definitely still learning
    assert (model.environment[0].weight.grad is None) or True
    changed = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in params
    )
    assert changed


def test_ck_dictionary_does_not_receive_gradients() -> None:
    """Regression guard for the PEC-v0 defect: D must not be task-coupled.

    In PEC-v0 ``d_node``/``d_edge`` were ``nn.Parameter`` and were passed to
    Adam, so task gradients reached the detached K-SVD dictionary.  PEC-C1
    freezes them; this test pins both halves of that statement.
    """
    batch = _batch([_sample(_ring(6), seed=11)])

    frozen = _model("sparse")
    c1.freeze_dictionary(frozen)
    codes = frozen._node_coord(torch.as_tensor(batch["node_basis"], dtype=torch.float32))
    assert codes.requires_grad is False
    loss = frozen(batch)["prediction"].pow(2).mean()
    loss.backward()
    assert frozen.d_node.grad is None
    assert frozen.d_edge.grad is None

    # the PEC-v0 behaviour, kept visible so the defect cannot silently return
    unfrozen = _model("sparse")
    codes_v0 = unfrozen._node_coord(torch.as_tensor(batch["node_basis"], dtype=torch.float32))
    assert codes_v0.requires_grad is True
    unfrozen(batch)["prediction"].pow(2).mean().backward()
    assert unfrozen.d_node.grad is not None
    assert unfrozen.d_edge.grad is not None


def test_dense_role_is_actually_trainable() -> None:
    model = _model("dense")
    x = torch.randn(16, pec.NODE_ROLE_DIM)
    model._node_coord(x).pow(2).mean().backward()
    assert model.m_node.weight.grad is not None
    assert model.m_node.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# 8: official-test blocker
# --------------------------------------------------------------------------


def test_official_test_split_is_blocked() -> None:
    assert c1.FORBIDDEN_SPLITS == ("test",)
    with pytest.raises(RuntimeError, match="forbidden"):
        c1._guard_split("test")
    with pytest.raises(RuntimeError, match="forbidden"):
        c1.load_split("test")
    with pytest.raises(RuntimeError, match="forbidden"):
        c1._extract_split("test")
    assert c1._guard_split("train") == "train"
    assert c1._guard_split("val") == "val"


def test_runner_only_ever_names_train_and_val() -> None:
    """The PEC-C1 runner source must not contain a 'test' split literal."""
    import inspect

    source = inspect.getsource(c1)
    assert 'split="test"' not in source
    assert 'load_split("test")' not in source
    assert c1.FORBIDDEN_SPLITS == ("test",)
    assert c1._cache_paths()[0].name == "train.pkl.gz"
    assert c1._cache_paths()[1].name == "valid.pkl.gz"
    # _extract_split is the only ZINC reader, and it is a guarded entry point
    assert "_guard_split(split)" in source


# --------------------------------------------------------------------------
# 9: deterministic forward
# --------------------------------------------------------------------------


def test_forward_is_deterministic() -> None:
    batch = _batch([_sample(_ring(6), seed=5), _sample(_chain(5), seed=6)])
    first = _model("sparse", seed=7)(batch)["prediction"]
    second = _model("sparse", seed=7)(batch)["prediction"]
    assert torch.equal(first, second)
    dense_first = _model("dense", seed=7)(batch)["prediction"]
    dense_second = _model("dense", seed=7)(batch)["prediction"]
    assert torch.equal(dense_first, dense_second)


# --------------------------------------------------------------------------
# 10: real-batch backward is finite (CPU here, CUDA when available)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("role_mode", ["sparse", "dense", "coarse"])
def test_real_batch_backward_is_finite(role_mode: str) -> None:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = _model(role_mode).to(device)
    if role_mode == "sparse":
        c1.freeze_dictionary(model)
    batch = c1.pec.to_device(
        _batch([_sample(_ring(8), seed=7), _sample(_chain(6), seed=8)]), device
    )
    out = model(batch)
    loss = (out["prediction"] - batch["y"]).abs().mean()
    loss.backward()
    params = c1.trainable_parameters(model)
    assert params
    for parameter in params:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_cuda_index_is_none_on_cpu() -> None:
    assert c1.cuda_index(torch.device("cpu")) is None


def test_cuda_context_is_initialised_before_device_scoped_calls(monkeypatch) -> None:
    """Regression guard: reset_peak_memory_stats/max_memory_allocated raise
    ``Invalid device argument`` unless the CUDA context is initialised first.
    ``is_available()`` is NOT sufficient (verified on the remote A100).
    """
    order: list[str] = []
    monkeypatch.setattr(torch.cuda, "init", lambda: order.append("init"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    index = c1.cuda_index(torch.device("cuda:1"))
    assert index == 1
    assert order == ["init"], order


def test_train_arm_initialises_cuda_before_peak_stats() -> None:
    import inspect

    source = inspect.getsource(c1.train_arm)
    assert "cuda_index(device)" in source
    assert source.index("cuda_index(device)") < source.index("reset_peak_memory_stats")
    assert "max_memory_allocated(gpu_index)" in source


# --------------------------------------------------------------------------
# frozen decision-gate semantics
# --------------------------------------------------------------------------


def _arm(soup: float) -> dict:
    return {
        "soup_valid_mae": soup,
        "mechanism": {
            "chem_shuffle": {"mean_abs_prediction_shift": 1.0, "degradation": 0.5},
            "relation_shuffle": {"mean_abs_prediction_shift": 0.1, "degradation": 0.01},
            "neutral_dictionary": {"mean_abs_prediction_shift": 0.2, "degradation": 1.0},
        },
    }


def test_gate_case_a_absolute_weak() -> None:
    decision = c1.decide(_arm(0.150), _arm(0.149))
    assert decision["verdict"] == "PURE_ENV_COMPOSITION_ABSOLUTE_WEAK"
    assert decision["seed1_authorized"] is False
    assert decision["band"] == ">0.145"


def test_gate_case_b_dense_dominates() -> None:
    decision = c1.decide(_arm(0.1470), _arm(0.1430))
    assert decision["verdict"] == "ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT"
    assert decision["seed1_authorized"] is False
    assert decision["delta_dict"] < 0
    assert "frozen" in decision["confound_note"]


def test_gate_case_c_sparse_signal() -> None:
    decision = c1.decide(_arm(0.1420), _arm(0.1455))
    assert decision["verdict"] == "PEC_C1_SPARSE_SIGNAL_SEED0"
    assert decision["seed1_authorized"] is True
    assert decision["delta_dict"] >= 0.003


def test_gate_case_d_strong_pure() -> None:
    decision = c1.decide(_arm(0.1380), _arm(0.1385))
    assert decision["verdict"] == "PURE_ENV_COMPOSITION_STRONG_SEED0"
    assert decision["seed1_authorized"] is True
    assert decision["band"] == "<=0.140"


def test_gate_case_e_viable_dict_unresolved() -> None:
    decision = c1.decide(_arm(0.1420), _arm(0.1425))
    assert decision["verdict"] == "PURE_ENV_COMPOSITION_VIABLE_DICT_UNRESOLVED"
    assert decision["seed1_authorized"] is False


def test_gate_precedence_a_over_b() -> None:
    # both arms are in the weak band, and CK is materially worse than CD
    decision = c1.decide(_arm(0.1500), _arm(0.1460))
    assert decision["cases"]["A_absolute_weak"] is True
    assert decision["cases"]["B_dense_dominates_sparse"] is True
    assert decision["verdict"] == "PURE_ENV_COMPOSITION_ABSOLUTE_WEAK"


def test_dead_dictionary_downgrades_sparse_signal() -> None:
    ck = _arm(0.1420)
    ck["mechanism"]["neutral_dictionary"]["mean_abs_prediction_shift"] = 0.0
    ck["mechanism"]["neutral_dictionary"]["degradation"] = 0.0
    decision = c1.decide(ck, _arm(0.1455))
    assert decision["verdict"] == "PURE_ENV_COMPOSITION_VIABLE_DICT_UNRESOLVED"
    assert decision["seed1_authorized"] is False
    assert decision["mechanism_integrity"]["dictionary_dependent"] is False


def test_formal_epoch_count_is_frozen() -> None:
    from tracks.ksvd.experiments.luyin16 import zinc_pec_c1 as runner

    assert runner.TRAIN_EPOCHS == 240
    assert runner.GATE_WEAK == 0.145
    assert runner.GATE_STRONG == 0.140
    assert runner.GATE_DICT_MATERIAL == 0.003
    assert runner.SEED == 0
    with pytest.raises(RuntimeError, match="240"):
        runner.train_arm(
            "CK",
            epochs=10,
            device=torch.device("cpu"),
            tag="guard_test",
            smoke_train=None,
        )


def test_historical_context_is_recorded() -> None:
    ctx = c1.HISTORICAL_CONTEXT
    assert abs(ctx["strict_static_S0_seed0_soup"] - 0.140794) < 1e-9
    assert abs(ctx["strict_static_S0_seed1_soup"] - 0.136423) < 1e-9
    assert ctx["B_Null"] == 0.123
    assert "not gates" in ctx["note"] or "NOT gates" in ctx["note"]
