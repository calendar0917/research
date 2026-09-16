"""FSAR correctness tests (brief section 26).

Covers the architecture-level information-path exclusivity contract:

* no mixed bypass survives into the FSAR forward,
* S chemistry invariance (exactly unchanged),
* A context-assignment invariance,
* B assignment sensitivity (with A and S unchanged),
* topology-only relation purity,
* node-relabel invariance,
* batch invariance,
* gradient viability for A / S / B / relation core / head.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import fsar
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as sspe

REPO_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_FORWARD_NAMES = {
    "patch_cont",
    "patch_context",
    "parent_token",
    "parent_embedding",
    "global_context",
    "global_encoder",
    "typed_token",
    "typed_embedding",
    "attribute_encoder",
    "structural_encoder",
    "structural_token",
    "structural_coarse",
}


# ---------------------------------------------------------------------------
# synthetic molecule builders
# ---------------------------------------------------------------------------


def _molecule(
    node_types: list[int],
    edges: list[tuple[int, int]],
    bond_types: list[int],
    y: float = 0.0,
) -> Data:
    edge_index = []
    edge_attr = []
    for (left, right), bond in zip(edges, bond_types):
        edge_index.append([left, right])
        edge_index.append([right, left])
        edge_attr.append(bond)
        edge_attr.append(bond)
    data = Data(
        x=torch.tensor(node_types, dtype=torch.long).view(-1, 1),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long).view(-1, 1),
        y=torch.tensor([float(y)], dtype=torch.float32),
    )
    data.num_nodes = len(node_types)
    return data


def _asymmetric_molecule() -> Data:
    # A 6-node graph with distinct local roles, so a context-attribute
    # permutation is not an automorphism.
    node_types = [3, 7, 11, 5, 13, 2]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]
    bond_types = [1, 2, 1, 3, 2]
    return _molecule(node_types, edges, bond_types, y=1.25)


def _encode_one(data: Data) -> Data:
    patch_graphs = sspe._patch_graphs_from_dataset([data])
    topology = np.zeros((1, fsar.TOPOLOGY_WIDTH), dtype=np.float32)
    encoded, _ = fsar.build_fsar_dataset(
        [data],
        patch_graphs,
        topology,
        topology_mean=np.zeros(fsar.TOPOLOGY_WIDTH, dtype=np.float32),
        topology_scale=np.ones(fsar.TOPOLOGY_WIDTH, dtype=np.float32),
    )
    return encoded[0]


def _batch(data_list: list[Data]):
    return fsar.fsar_collate(data_list)


def _clone(data: Data) -> Data:
    return copy.deepcopy(data)


def _fresh_model(mode: str = "SAB", seed: int = 0) -> fsar.PatchPathFSARModel:
    torch.manual_seed(int(seed))
    model = fsar.PatchPathFSARModel(mode=mode)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# A. no mixed bypass
# ---------------------------------------------------------------------------


def test_no_mixed_bypass_attributes_and_forward_audit() -> None:
    model = _fresh_model("SAB")
    for attribute in (
        "typed_embedding",
        "patch_encoder",
        "global_encoder",
        "parent_embedding",
        "attribute_encoder",
        "structural_encoder",
    ):
        assert not hasattr(model, attribute), f"{attribute} should be deleted in FSAR"

    data = _encode_one(_asymmetric_molecule())
    for attribute in (
        "patch_cont",
        "patch_context",
        "parent_token",
        "global_context",
        "typed_token",
    ):
        assert not hasattr(data, attribute), f"FSAR data still carries {attribute}"

    source = (REPO_ROOT / "tracks/ksvd/experiments/luyin16/fsar.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
    leaked = FORBIDDEN_FORWARD_NAMES & used
    assert not leaked, f"FSAR reads forbidden mixed tensors: {sorted(leaked)}"


# ---------------------------------------------------------------------------
# B. S chemistry invariance
# ---------------------------------------------------------------------------


def test_s_chemistry_invariance_exact() -> None:
    model = _fresh_model("SAB")
    original = _encode_one(_asymmetric_molecule())

    perturbed_mol = _asymmetric_molecule()
    perturbed_mol.x = (perturbed_mol.x + 5) % fsar.ATOM_CATEGORIES
    perturbed_mol.edge_attr = (perturbed_mol.edge_attr + 1) % fsar.BOND_CATEGORIES
    perturbed = _encode_one(perturbed_mol)

    batch_a = _batch([original])
    batch_b = _batch([perturbed])
    with torch.no_grad():
        s_a = model.encoder.forward_channels(batch_a)["structure"]
        s_b = model.encoder.forward_channels(batch_b)["structure"]
    assert torch.equal(s_a, s_b), "S must be exactly invariant to chemistry"


# ---------------------------------------------------------------------------
# C. A context-assignment invariance
# ---------------------------------------------------------------------------


def test_a_context_assignment_invariance() -> None:
    model = _fresh_model("SAB")
    original = _encode_one(_asymmetric_molecule())
    permuted = _clone(original)

    patch = permuted.struct_patch.long()
    root = permuted.struct_root.long()
    # Permute atom categories among the non-root nodes inside each patch.
    for patch_id in torch.unique(patch):
        mask = (patch == patch_id) & (root == 0)
        index = torch.nonzero(mask, as_tuple=False).view(-1)
        if index.numel() >= 2:
            values = permuted.struct_atom[index].clone()
            permuted.struct_atom[index] = values.flip(0)
    # Permute bond categories among the edges inside each patch.
    edge_patch = permuted.struct_edge_patch.long()
    for patch_id in torch.unique(edge_patch):
        index = torch.nonzero(edge_patch == patch_id, as_tuple=False).view(-1)
        if index.numel() >= 2:
            values = permuted.struct_bond[index].clone()
            permuted.struct_bond[index] = values.flip(0)

    batch_a = _batch([original])
    batch_b = _batch([permuted])
    with torch.no_grad():
        a_a = model.encoder.forward_channels(batch_a)["attributes"]
        a_b = model.encoder.forward_channels(batch_b)["attributes"]
        s_a = model.encoder.forward_channels(batch_a)["structure"]
        s_b = model.encoder.forward_channels(batch_b)["structure"]
    assert torch.allclose(a_a, a_b, atol=1e-6, rtol=0.0), "A must ignore assignment"
    assert torch.equal(s_a, s_b)


# ---------------------------------------------------------------------------
# D. B assignment sensitivity
# ---------------------------------------------------------------------------


def test_b_assignment_sensitivity() -> None:
    model = _fresh_model("SAB", seed=3)
    original = _encode_one(_asymmetric_molecule())
    permuted = _clone(original)

    patch = permuted.struct_patch.long()
    root = permuted.struct_root.long()
    changed = 0
    for patch_id in torch.unique(patch):
        mask = (patch == patch_id) & (root == 0)
        index = torch.nonzero(mask, as_tuple=False).view(-1)
        if index.numel() >= 2:
            values = permuted.struct_atom[index].clone()
            if not torch.equal(values, values.flip(0)):
                changed += 1
            permuted.struct_atom[index] = values.flip(0)
    assert changed > 0, "test molecule must have an asymmetric context assignment"

    batch_a = _batch([original])
    batch_b = _batch([permuted])
    with torch.no_grad():
        ch_a = model.encoder.forward_channels(batch_a)
        ch_b = model.encoder.forward_channels(batch_b)
    assert torch.allclose(ch_a["attributes"], ch_b["attributes"], atol=1e-6, rtol=0.0)
    assert torch.equal(ch_a["structure"], ch_b["structure"])
    # Same centre attribute + multisets, different role assignment: B must move.
    delta = (ch_a["binding"] - ch_b["binding"]).norm()
    assert float(delta) > 1e-5, "B must react to the attribute<->role assignment"


# ---------------------------------------------------------------------------
# E. relation purity
# ---------------------------------------------------------------------------


def test_relation_purity_topology_only() -> None:
    molecule = _asymmetric_molecule()
    graph, _node_types, _edge_types = fsar.zpp._data_to_graph(molecule)
    centers = list(graph.nodes)
    patch_sets = [fsar._patch_node_sets(graph, int(center)) for center in centers]
    reference = fsar.topology_pair_relation(graph, centers, patch_sets)

    # Same untyped graph, different chemistry -> relation must be identical.
    other = _clone(molecule)
    other.x = (other.x + 9) % fsar.ATOM_CATEGORIES
    other.edge_attr = (other.edge_attr + 2) % fsar.BOND_CATEGORIES
    graph2, _, _ = fsar.zpp._data_to_graph(other)
    other_sets = [fsar._patch_node_sets(graph2, int(center)) for center in centers]
    perturbed = fsar.topology_pair_relation(graph2, centers, other_sets)

    for ref, other_block in zip(reference, perturbed):
        assert np.array_equal(ref, other_block), "relation must be chemistry-free"


# ---------------------------------------------------------------------------
# F. node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariance() -> None:
    model = _fresh_model("SAB", seed=5)
    molecule = _asymmetric_molecule()
    permutation = [3, 0, 5, 1, 4, 2]
    inverse = [0] * len(permutation)
    for new, old in enumerate(permutation):
        inverse[old] = new
    relabeled_edges: list[tuple[int, int]] = []
    relabeled_bonds: list[int] = []
    for (left, right), bond in zip(
        molecule.edge_index.t().tolist(), molecule.edge_attr.view(-1).tolist()
    ):
        if int(left) >= int(right):
            continue
        new_left, new_right = inverse[int(left)], inverse[int(right)]
        relabeled_edges.append(tuple(sorted((new_left, new_right))))
        relabeled_bonds.append(int(bond))
    order = sorted(range(len(relabeled_edges)), key=lambda index: relabeled_edges[index])
    relabeled = _molecule(
        [molecule.x.view(-1)[old].item() for old in permutation],
        [relabeled_edges[index] for index in order],
        [relabeled_bonds[index] for index in order],
        y=float(molecule.y.view(-1)[0]),
    )
    batch_a = _batch([_encode_one(molecule)])
    batch_b = _batch([_encode_one(relabeled)])
    with torch.no_grad():
        pred_a = model(batch_a)
        pred_b = model(batch_b)
    assert torch.allclose(pred_a, pred_b, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# G. batch invariance
# ---------------------------------------------------------------------------


def test_batch_invariance() -> None:
    model = _fresh_model("SAB", seed=7)
    first = _encode_one(_asymmetric_molecule())
    second_mol = _asymmetric_molecule()
    second_mol.y = torch.tensor([0.5])
    second = _encode_one(second_mol)

    with torch.no_grad():
        single_a = model(_batch([first]))
        single_b = model(_batch([second]))
        stacked = model(_batch([first, second]))
    assert torch.allclose(stacked[0], single_a[0], atol=1e-5, rtol=0.0)
    assert torch.allclose(stacked[1], single_b[0], atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# H. gradient viability
# ---------------------------------------------------------------------------


def test_gradient_viability() -> None:
    torch.manual_seed(11)
    model = fsar.PatchPathFSARModel(mode="SAB")
    model.train()
    batch = _batch([_encode_one(_asymmetric_molecule())])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    # The centre update is zero-initialised (proven cell-A design), so at step 0
    # the relation core cannot influence the output.  One optimizer step makes
    # the centre update non-degenerate; every path must then receive gradient.
    loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
    optimizer.zero_grad()
    loss.backward()

    checks = {
        "A": model.encoder.atom_embedding.weight,
        "S": model.encoder.root_embedding.weight,
        "B": model.encoder.node_role_projection.weight,
        "relation_core": model.relation_encoder[0].weight,
        "center_update": model.center_update[0].weight,
        "head": model.head[-1].weight,
    }
    for name, parameter in checks.items():
        grad = parameter.grad
        assert grad is not None, f"{name} received no gradient"
        assert bool(torch.isfinite(grad).all()), f"{name} gradient is not finite"
        assert float(grad.abs().sum()) > 0.0, f"{name} gradient is exactly zero"


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def test_parameter_accounting_and_nested_modes() -> None:
    totals = {}
    for mode in fsar.MODES:
        torch.manual_seed(0)
        model = fsar.PatchPathFSARModel(mode=mode)
        breakdown = fsar.parameter_breakdown(model)
        assert breakdown["total"] <= 200_000
        assert breakdown["dataset_dependent_vocabulary_params"] == 0
        totals[mode] = breakdown["total"]
    assert totals["A"] < totals["SA"] < totals["SAB"]


def test_modes_share_relation_core_and_head_geometry() -> None:
    models = {mode: _fresh_model(mode) for mode in fsar.MODES}
    widths = {mode: model.unified_graph_width for mode, model in models.items()}
    assert len(set(widths.values())) == 1
    for mode, model in models.items():
        assert tuple(model.head[-1].weight.shape) == (1, fsar.HEAD_HIDDEN[-1])
        assert model.center_update is not None
