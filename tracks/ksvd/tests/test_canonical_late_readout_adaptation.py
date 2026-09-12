"""Tests for the canonical late-readout adaptation confirmation.

These pin the pre-registered protocol of
``experiments/luyin16/zinc_canonical_late_readout_adaptation``:

1.  B0 / C / E start from the same canonical selected checkpoint;
2.  C / E adaptation step 0 is function-equivalent to B0;
3.  standardisation statistics use official train only;
4.  the first-layer reparameterisation is algebraically exact;
5.  the E optimizer contains only graph-head parameters;
6.  E leaves every backbone tensor numerically unchanged;
7.  the C optimizer contains the expected full trainable set;
8.  C and E share batch order and step count;
9.  C and E use the same L1 objective;
10. the adaptation protocol lock is produced before any valid evaluation;
11. the adaptation loop accepts no valid loader;
12. official test is never loaded.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_canonical_late_readout_adaptation as clr,
)


# ---------------------------------------------------------------------------
# lightweight fake backbone so the mechanics can be tested without a GPU run
# ---------------------------------------------------------------------------


class _FakeBackbone(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 4) -> None:
        super().__init__()
        self.trunk = nn.Linear(in_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 3),
            nn.LayerNorm(3),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(3, 1),
        )

    def encode(self, data: Data) -> torch.Tensor:
        return torch.relu(self.trunk(data.x))

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data)).view(-1)


def _fake_graphs(n: int = 32, in_dim: int = 6, seed: int = 0) -> list[Data]:
    rng = np.random.default_rng(seed)
    graphs: list[Data] = []
    for _ in range(n):
        x = torch.tensor(rng.standard_normal((1, in_dim)), dtype=torch.float32)
        y = torch.tensor([rng.standard_normal()], dtype=torch.float32)
        graphs.append(Data(x=x, y=y))
    return graphs


def _fake_wrapper(seed: int = 0, in_dim: int = 6, hidden: int = 4) -> clr.StandardizedReadout:
    torch.manual_seed(seed)
    backbone = _FakeBackbone(in_dim=in_dim, hidden=hidden)
    mean = np.zeros(hidden)
    scale = np.ones(hidden)
    clr.reparameterize_head(backbone.head, mean, scale)
    return clr.StandardizedReadout(backbone, mean, scale)


def _checkpoint_available() -> bool:
    try:
        return all(clr.canonical_run(seed)["state_path"].exists() for seed in (0, 1))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1 -- shared starting checkpoint
# ---------------------------------------------------------------------------


def test_fresh_wrappers_share_starting_checkpoint():
    if not _checkpoint_available():
        pytest.skip("canonical checkpoints not present")
    info = clr.canonical_run(0)
    config, audit = info["config"], None
    train_data, valid_data, audit = clr.build_data(config)  # noqa: F841
    payload = np.load(clr.CACHE_DIR / "seed0_prepare.npz", allow_pickle=True)
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    wrapper_c = clr._fresh_wrapper(config, audit, info["run_dir"], mean, scale)
    wrapper_e = clr._fresh_wrapper(config, audit, info["run_dir"], mean, scale)
    state_c = {k: v.clone() for k, v in wrapper_c.state_dict().items()}
    state_e = {k: v.clone() for k, v in wrapper_e.state_dict().items()}
    assert set(state_c) == set(state_e)
    for name in state_c:
        assert torch.equal(state_c[name], state_e[name]), name


# ---------------------------------------------------------------------------
# 2 -- step 0 equivalence (fake backbone, real reparameterisation)
# ---------------------------------------------------------------------------


def test_reparameterised_step0_equivalent_to_original():
    graphs = _fake_graphs(8)
    torch.manual_seed(3)
    backbone = _FakeBackbone()
    raw = clr.predict_model(backbone, graphs)
    mean = np.asarray([0.5, -0.25, 0.1, 0.0])
    scale = np.asarray([2.0, 0.5, 3.0, 1.0])
    clr.reparameterize_head(backbone.head, mean, scale)
    standardized = clr.StandardizedReadout(backbone, mean, scale).eval()
    transformed = clr.predict_model(standardized, graphs)
    assert np.abs(raw - transformed).max() <= clr.REPARAM_ATOL


# ---------------------------------------------------------------------------
# 3 -- standardisation uses official train only
# ---------------------------------------------------------------------------


def test_standardizer_uses_train_only():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((200, 5))
    mean, scale = clr.fit_standardizer(train)
    reference_mean = train.mean(axis=0)
    reference_scale = np.maximum(train.std(axis=0), clr.STANDARDIZE_EPS)
    assert np.allclose(mean, reference_mean)
    assert np.allclose(scale, reference_scale)
    # a different "valid" matrix cannot influence the statistics
    valid = rng.standard_normal((50, 5)) * 100.0
    mean2, scale2 = clr.fit_standardizer(train)
    assert np.array_equal(mean, mean2) and np.array_equal(scale, scale2)
    assert valid.shape == (50, 5)  # used only to document that valid is ignored


def test_standardizer_fit_signature_only_takes_train():
    signature = inspect.signature(clr.fit_standardizer)
    assert list(signature.parameters) == ["R_train"]


# ---------------------------------------------------------------------------
# 4 -- exact first-layer reparameterisation
# ---------------------------------------------------------------------------


def test_first_layer_transform_formula_toy():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((4, 6))
    bias = rng.standard_normal(4)
    mean = rng.standard_normal(6)
    scale = np.abs(rng.standard_normal(6)) + 0.1
    new_weight, new_bias = clr.reparameterized_head_reference(weight, bias, mean, scale)
    z = rng.standard_normal((17, 6))
    R = mean + scale * z
    original = R @ weight.T + bias
    transformed = z @ new_weight.T + new_bias
    assert np.abs(original - transformed).max() < 1e-10


# ---------------------------------------------------------------------------
# 5 -- E optimizer contains only head parameters
# ---------------------------------------------------------------------------


def test_e_optimizer_only_head_parameters():
    wrapper = _fake_wrapper()
    trainable = clr.set_head_only_trainable(wrapper)
    assert trainable
    assert all(name.startswith(clr.HEAD_PREFIX) for name in trainable)
    for name, parameter in wrapper.named_parameters():
        expected = name.startswith(clr.HEAD_PREFIX)
        assert bool(parameter.requires_grad) is expected, name


def test_e_adaptation_trainable_parameter_list_is_head_only():
    graphs = _fake_graphs(16)
    wrapper = _fake_wrapper()
    result = clr.run_adaptation(wrapper, graphs, epochs=2, mode="E", batch_size=8)
    assert result["mode"] == "E"
    assert all(
        name.startswith(clr.HEAD_PREFIX) for name in result["trainable_parameters"]
    )
    assert result["trainable_parameter_count"] < result["total_parameter_count"]


# ---------------------------------------------------------------------------
# 6 -- E leaves backbone tensors unchanged
# ---------------------------------------------------------------------------


def test_e_adaptation_leaves_backbone_unchanged():
    graphs = _fake_graphs(24)
    wrapper = _fake_wrapper()
    before = {
        name: parameter.detach().clone()
        for name, parameter in wrapper.named_parameters()
        if not name.startswith(clr.HEAD_PREFIX)
    }
    before_R = clr.encode_representation(wrapper.backbone, graphs)
    clr.run_adaptation(wrapper, graphs, epochs=3, mode="E", batch_size=8)
    after = {
        name: parameter.detach().clone()
        for name, parameter in wrapper.named_parameters()
        if not name.startswith(clr.HEAD_PREFIX)
    }
    after_R = clr.encode_representation(wrapper.backbone, graphs)
    assert set(before) == set(after)
    for name in before:
        assert torch.equal(before[name], after[name]), name
    assert np.array_equal(before_R, after_R)


# ---------------------------------------------------------------------------
# 7 -- C optimizer contains the expected full trainable set
# ---------------------------------------------------------------------------


def test_c_adaptation_trains_non_head_parameters():
    graphs = _fake_graphs(24)
    wrapper = _fake_wrapper()
    before = {
        name: parameter.detach().clone()
        for name, parameter in wrapper.named_parameters()
        if not name.startswith(clr.HEAD_PREFIX)
    }
    result = clr.run_adaptation(wrapper, graphs, epochs=3, mode="C", batch_size=8)
    assert result["trainable_parameter_count"] == len(list(wrapper.named_parameters()))
    assert result["trainable_numel"] == result["total_parameter_count"]
    after = {
        name: parameter.detach().clone()
        for name, parameter in wrapper.named_parameters()
        if not name.startswith(clr.HEAD_PREFIX)
    }
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed, "full continuation must update some non-head parameter"


# ---------------------------------------------------------------------------
# 8 -- C and E share batch order and step count
# ---------------------------------------------------------------------------


def test_c_and_e_share_batch_order_and_steps():
    graphs = _fake_graphs(30)
    wrapper_c = _fake_wrapper(seed=5)
    wrapper_e = _fake_wrapper(seed=5)
    result_c = clr.run_adaptation(wrapper_c, graphs, epochs=4, mode="C", batch_size=8)
    result_e = clr.run_adaptation(wrapper_e, graphs, epochs=4, mode="E", batch_size=8)
    assert len(result_c["curve"]) == len(result_e["curve"]) == 4
    for row_c, row_e in zip(result_c["curve"], result_e["curve"]):
        assert row_c["epoch"] == row_e["epoch"]
        assert row_c["first_batch_indices"] == row_e["first_batch_indices"]


# ---------------------------------------------------------------------------
# 9 -- C and E use the same L1 objective
# ---------------------------------------------------------------------------


def test_adaptation_uses_l1_objective():
    source = inspect.getsource(clr.run_adaptation)
    assert "(prediction - target).abs().mean()" in source
    assert "mse" not in source.lower()
    assert "smooth_l1" not in source.lower()


# ---------------------------------------------------------------------------
# 10 -- protocol lock is produced before valid evaluation
# ---------------------------------------------------------------------------


def test_adaptation_is_deterministic_under_fixed_seed():
    graphs = _fake_graphs(24)
    first = _fake_wrapper(seed=1)
    second = _fake_wrapper(seed=1)
    clr.run_adaptation(first, graphs, epochs=3, mode="C", batch_size=8, order_seed=0)
    clr.run_adaptation(second, graphs, epochs=3, mode="C", batch_size=8, order_seed=0)
    state_a = first.state_dict()
    state_b = second.state_dict()
    for name in state_a:
        assert torch.equal(state_a[name], state_b[name]), name


def test_protocol_lock_precedes_results_and_forbids_valid():
    lock = clr.protocol_lock(write=False)
    assert lock["official_valid_used_during_adaptation"] is False
    assert lock["official_test_loaded"] is False
    assert lock["adaptation_budget"]["k_star"] > 0
    if clr.OOF_FOLD_RESULTS.exists():
        assert lock["adaptation_budget"]["k_star"] == 94
    source = inspect.getsource(clr.stage_all)
    assert source.index("protocol_lock()") < source.index("run_seed(")


# ---------------------------------------------------------------------------
# 11 -- adaptation loop takes no valid loader
# ---------------------------------------------------------------------------


def test_adaptation_loop_has_no_valid_argument():
    signature = inspect.signature(clr.run_adaptation)
    assert not any("valid" in name for name in signature.parameters)
    source = inspect.getsource(clr.run_seed)
    assert "run_adaptation(" in source
    assert "valid_data" in source  # only for the post-adaptation evaluation


# ---------------------------------------------------------------------------
# 12 -- official test is never loaded
# ---------------------------------------------------------------------------


def test_official_test_never_loaded():
    module_source = inspect.getsource(clr)
    assert "_load_zinc(ZINC_ROOT, \"test\")" not in module_source
    assert "matrices_for_split(\"test\"" not in module_source
    # the experiment builds records from the cached official train+valid only
    assert "_extract_v4_records" in inspect.getsource(clr.build_data)
    inventory = clr.checkpoint_inventory(write=False)
    assert all(entry["official_test_loaded"] is False for entry in inventory["entries"])


# ---------------------------------------------------------------------------
# extra: decision logic is deterministic and matches the preregistration
# ---------------------------------------------------------------------------


def test_seed0_decision_cases():
    strong = {"seed": 0, "delta_A_B0_minus_E": 0.004, "delta_F_C_minus_E": 0.002, "delta_C_B0_minus_C": 0.003}
    decision = clr.seed0_decision(strong)
    assert decision["status"] == "STRONG_ADVANCE" and decision["run_seed1"] is True

    clear = {"seed": 0, "delta_A_B0_minus_E": 0.001, "delta_F_C_minus_E": 0.0, "delta_C_B0_minus_C": 0.001}
    decision = clr.seed0_decision(clear)
    assert decision["status"] == "CLEAR_NO_GO" and decision["run_seed1"] is False

    weak = {"seed": 0, "delta_A_B0_minus_E": 0.004, "delta_F_C_minus_E": 0.0005, "delta_C_B0_minus_C": 0.0035}
    decision = clr.seed0_decision(weak)
    assert decision["status"] == "ADAPTATION_SIGNAL_MECHANISM_WEAK"


def test_final_decision_go():
    results = {
        0: {
            "b0_valid_mae": 0.17,
            "c_valid_mae": 0.168,
            "e_valid_mae": 0.166,
            "delta_A_B0_minus_E": 0.004,
            "delta_C_B0_minus_C": 0.002,
            "delta_F_C_minus_E": 0.002,
        },
        1: {
            "b0_valid_mae": 0.16,
            "c_valid_mae": 0.158,
            "e_valid_mae": 0.1565,
            "delta_A_B0_minus_E": 0.0035,
            "delta_C_B0_minus_C": 0.002,
            "delta_F_C_minus_E": 0.0015,
        },
    }
    decision = clr.final_decision(results, write=False)
    assert decision["verdict"] == "GO_CANONICAL_LATE_READOUT_ADAPTATION_CONFIRMED"
    assert decision["decision_case"] == "Case A"
