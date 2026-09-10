"""Viability-repair tests for the compact-v6 attribute branch.

These tests answer the pre-registered viability questions without touching
the real dataset:

1. every attribute-branch parameter is trainable and present in the optimizer;
2. an adversarial placement pair changes the factorized raw representation
   while leaving the count view identical;
3. a permutation-equivalent patch yields the identical representation;
4. one backward step produces the expected nonzero downstream gradient
   (and the original zero-init produces an exactly-zero upstream gradient);
5. the repaired init gives a nonzero attribute-branch variance at step 0;
6. 100 toy optimizer steps change the branch output;
7. once trained, zeroing e_attribute alters the prediction;
8. with the attribute branch disabled the model is the historical v4 model.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import yaml

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import compact_v6_attribute_roles as v6
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module
from tracks.ksvd.tests import test_compact_v6_attribute_roles as t6


def _model(attribute_mode: str = "factorized_role", *, fusion_init: str = "zero"):
    config = yaml.safe_load(
        (
            module.REPO_ROOT
            / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
        ).read_text(encoding="utf-8")
    )["model"]
    return module.PatchPathModel(
        t6.SELECTION_TYPED_VOCAB,
        t6.PARENT_VOCAB,
        patch_hidden=int(config["patch_hidden"]),
        pair_hidden=int(config["pair_hidden"]),
        token_width=int(config["token_width"]),
        dropout=float(config["dropout"]),
        embedding_mode=str(config["embedding_mode"]),
        embedding_rank=int(config["embedding_rank"]),
        hybrid_full_typed_tokens=int(config["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(config["hybrid_full_parent_tokens"]),
        center_context=bool(config["center_context"]),
        center_context_hidden=int(config["center_context_hidden"]),
        graph_head_hidden_0=int(config["graph_head_hidden_0"]),
        graph_head_hidden_1=int(config["graph_head_hidden_1"]),
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=int(config["topology_hidden_dim"]),
        topology_out_dim=int(config["topology_out_dim"]),
        attribute_mode=attribute_mode,
        attribute_fusion_init=fusion_init,
    )


def _e_attribute(model, data) -> torch.Tensor:
    holder: dict[str, torch.Tensor] = {}

    def hook(_m, _i, output):
        holder["value"] = output.detach()

    handle = model.attribute_encoder.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(data)
    handle.remove()
    return holder["value"]


# --------------------------------------------------------------------------
# Test 1 -- optimizer registration
# --------------------------------------------------------------------------
def test_attribute_branch_parameters_all_in_optimizer() -> None:
    model = _model("factorized_role", fusion_init="small_normal")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    registered = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    attribute = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("attribute_encoder")
    ]
    assert attribute, "attribute branch must expose parameters"
    assert len(attribute) == 14
    for name, parameter in attribute:
        assert parameter.requires_grad, name
        assert id(parameter) in registered, name


# --------------------------------------------------------------------------
# Test 2 -- adversarial placement pair
# --------------------------------------------------------------------------
def test_adversarial_placement_pair_factorized_differs() -> None:
    graph = from_edges(5, [(0, 1), (1, 2), (1, 3), (0, 4)])
    edge_types = {(0, 1): 0, (1, 2): 0, (1, 3): 0, (0, 4): 0}
    p1 = np.asarray([0, 0, 1, 0, 0], dtype=np.int64)  # heteroatom on leaf
    p2 = np.asarray([0, 1, 0, 0, 0], dtype=np.int64)  # heteroatom on branch
    count1 = t6._pooled(graph, 1, p1, edge_types, use_role=False)
    count2 = t6._pooled(graph, 1, p2, edge_types, use_role=False)
    factor1 = t6._pooled(graph, 1, p1, edge_types, use_role=True)
    factor2 = t6._pooled(graph, 1, p2, edge_types, use_role=True)
    assert count1 == count2
    assert factor1 != factor2


# --------------------------------------------------------------------------
# Test 3 -- permutation equivalence
# --------------------------------------------------------------------------
def test_permutation_equivalent_patch_same_representation() -> None:
    graph, node_types, edge_types = t6._branched()
    reference = t6._pooled(graph, 0, node_types, edge_types)
    rng = np.random.default_rng(7)
    for _ in range(40):
        permutation = rng.permutation(len(graph.nodes)).tolist()
        rg, rnt, ret, mapping = t6._relabel(graph, node_types, edge_types, permutation)
        assert t6._pooled(rg, mapping[0], rnt, ret) == reference


# --------------------------------------------------------------------------
# Test 4 -- step-0 gradient path
# --------------------------------------------------------------------------
def test_one_backward_step_nonzero_downstream_gradient() -> None:
    data = t6._attribute_data()

    # repaired init: the whole branch is connected from step 0
    repaired = _model("factorized_role", fusion_init="small_normal")
    repaired.train()
    repaired(data).sum().backward()
    first = repaired.patch_encoder.layers[0].weight
    attr_cols = slice(int(first.shape[1]) - 8, int(first.shape[1]))
    assert float(first.grad[:, attr_cols].abs().sum()) > 0
    assert float(repaired.attribute_encoder.atom_mlp[0].weight.grad.abs().sum()) > 0
    assert float(repaired.attribute_encoder.fusion[-1].weight.grad.abs().sum()) > 0

    # original zero-init: the upstream encoder is exactly blocked at step 0
    original = _model("factorized_role", fusion_init="zero")
    original.train()
    original(data).sum().backward()
    assert float(original.attribute_encoder.atom_mlp[0].weight.grad.abs().sum()) == 0.0
    assert float(original.attribute_encoder.bond_mlp[0].weight.grad.abs().sum()) == 0.0
    assert float(original.attribute_encoder.fusion[-1].weight.grad.abs().sum()) > 0


# --------------------------------------------------------------------------
# Test 5 -- repaired init gives nonzero branch variance
# --------------------------------------------------------------------------
def test_repaired_init_nonzero_branch_variance() -> None:
    data = t6._attribute_data()
    repaired = _model("factorized_role", fusion_init="small_normal")
    original = _model("factorized_role", fusion_init="zero")
    e_repaired = _e_attribute(repaired, data)
    e_original = _e_attribute(original, data)
    assert float(e_repaired.std(dim=0).mean()) > 0.0
    assert float(e_repaired.norm()) > 0.0
    # the original model is exactly v4 at init
    assert float(e_original.abs().sum()) == 0.0


# --------------------------------------------------------------------------
# Test 6 -- 100 toy steps change the branch output
# --------------------------------------------------------------------------
def test_100_step_toy_learning_changes_e_attribute() -> None:
    torch.manual_seed(0)
    model = _model("factorized_role", fusion_init="small_normal")
    data = t6._attribute_data()
    before = _e_attribute(model, data).clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        prediction = model(data)
        target = torch.linspace(-1.0, 1.0, prediction.numel())
        loss = torch.nn.functional.l1_loss(prediction.reshape(-1), target)
        loss.backward()
        optimizer.step()
    after = _e_attribute(model, data)
    assert not torch.allclose(before, after, atol=1e-6)
    assert float(after.norm()) > 0.0
    assert float(after.std(dim=0).mean()) > 0.0


# --------------------------------------------------------------------------
# Test 7 -- attr-zero alters the output once trained
# --------------------------------------------------------------------------
def test_attr_zero_alters_output_once_trained() -> None:
    torch.manual_seed(0)
    model = _model("factorized_role", fusion_init="small_normal")
    data = t6._attribute_data()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        prediction = model(data)
        target = torch.linspace(-1.0, 1.0, prediction.numel())
        torch.nn.functional.l1_loss(prediction.reshape(-1), target).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        clean = model(copy.deepcopy(data))
    handle = model.attribute_encoder.register_forward_hook(
        lambda _m, _i, output: torch.zeros_like(output)
    )
    with torch.no_grad():
        zeroed = model(copy.deepcopy(data))
    handle.remove()
    assert float((clean - zeroed).abs().max()) > 0.0


# --------------------------------------------------------------------------
# Test 8 -- attribute mode disabled == historical v4 path
# --------------------------------------------------------------------------
def test_attribute_mode_disabled_is_historical_v4() -> None:
    base = _model("none")
    assert base.attribute_encoder is None
    assert int(base.attribute_input_width) == 0
    assert (
        sum(p.numel() for p in base.parameters() if p.requires_grad)
        == t6.V4_HINGE_TOTAL
    )

    torch.manual_seed(0)
    base = _model("none")
    torch.manual_seed(0)
    factor = _model("factorized_role", fusion_init="small_normal")
    # zero the attribute fusion so the branch contributes nothing, then copy
    # the historical weights and check the two models agree exactly
    factor_state = factor.state_dict()
    base_state = base.state_dict()
    for key, value in base_state.items():
        if key == "patch_encoder.layers.0.weight":
            factor_state[key][:, : value.shape[1]] = value.clone()
        elif key in factor_state and factor_state[key].shape == value.shape:
            factor_state[key] = value.clone()
    factor.load_state_dict(factor_state)
    with torch.no_grad():
        factor.attribute_encoder.fusion[-1].weight.zero_()
        factor.attribute_encoder.fusion[-1].bias.zero_()
    data = t6._attribute_data()
    base.eval()
    factor.eval()
    with torch.no_grad():
        out_base = base(data)
        out_factor = factor(data)
    assert torch.allclose(out_base, out_factor, atol=1e-6)
