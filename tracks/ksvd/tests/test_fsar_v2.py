"""FSAR-v2 correctness tests (brief sections 3, 6, 12, 19-23, 31).

Covers the two new scientific objects:

* ``A_exact`` -- lossless discrete categorical marginals of the rooted frame:
  assignment invariance, marginal collision-freedom, topology-field blindness,
  and removal of the FSAR-v1 information-resolution confound vs ``B``.
* ``S_explicit`` / ``B_explicit`` -- fixed readable rooted structural operator
  basis: chemistry purity, relabel equivariance, assignment sensitivity with
  unchanged ``A``/``S``, and *no* adjacency message passing in the explicit
  path.

Plus the inherited FSAR-v1 contract checks (no mixed bypass, batch invariance,
node-relabel invariance, gradient viability, parameter accounting).
"""

from __future__ import annotations

import ast
import copy

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import fsar
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)

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
    node_types = [3, 7, 11, 5, 13, 2]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]
    bond_types = [1, 2, 1, 3, 2]
    return _molecule(node_types, edges, bond_types, y=1.25)


def _asymmetric_molecule_two() -> Data:
    node_types = [1, 4, 9, 2, 6, 14, 7]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]
    bond_types = [2, 1, 3, 1, 2, 1]
    return _molecule(node_types, edges, bond_types, y=0.5)


def _encode_one(data: Data) -> Data:
    patch_graphs = sspe._patch_graphs_from_dataset([data])
    topology = np.zeros((1, fsar.TOPOLOGY_WIDTH), dtype=np.float32)
    encoded, _ = v2.build_fsar_v2_dataset(
        [data],
        patch_graphs,
        topology,
        topology_mean=np.zeros(fsar.TOPOLOGY_WIDTH, dtype=np.float32),
        topology_scale=np.ones(fsar.TOPOLOGY_WIDTH, dtype=np.float32),
    )
    return encoded[0]


def _batch(data_list: list[Data]):
    return v2.fsar_v2_collate(data_list)


def _clone(data: Data) -> Data:
    return copy.deepcopy(data)


def _fresh_model(mode: str = "SAB", seed: int = 0) -> v2.PatchPathFSARV2Model:
    torch.manual_seed(int(seed))
    model = v2.PatchPathFSARV2Model(mode=mode)
    model.eval()
    return model


def _permute_context_attributes(encoded: Data, *, change_marginal: bool) -> Data:
    """Permute context atom / bond categories inside each patch.

    With ``change_marginal=False`` this is a pure assignment change (multiset
    preserved).  With ``change_marginal=True`` the first context atom is moved
    to a category absent from that patch, so the marginal changes.
    """
    permuted = _clone(encoded)
    patch = permuted.struct_patch.long()
    root = permuted.struct_root.long()
    for patch_id in torch.unique(patch):
        mask = (patch == patch_id) & (root == 0)
        index = torch.nonzero(mask, as_tuple=False).view(-1)
        if index.numel() >= 2:
            values = permuted.struct_atom[index].clone()
            permuted.struct_atom[index] = values.flip(0)
    edge_patch = permuted.struct_edge_patch.long()
    for patch_id in torch.unique(edge_patch):
        index = torch.nonzero(edge_patch == patch_id, as_tuple=False).view(-1)
        if index.numel() >= 2:
            values = permuted.struct_bond[index].clone()
            permuted.struct_bond[index] = values.flip(0)
    if change_marginal:
        context = torch.nonzero((root == 0), as_tuple=False).view(-1)
        present = set(permuted.struct_atom[context].tolist())
        replacement = next(
            category
            for category in range(v2.ATOM_CATEGORIES)
            if category not in present
        )
        permuted.struct_atom[context[0]] = int(replacement)
    return permuted


# ---------------------------------------------------------------------------
# A_exact: losslessness / collision-freedom
# ---------------------------------------------------------------------------


def test_exact_marginal_assignment_invariance_and_collision_freedom() -> None:
    model = _fresh_model("A")
    original = _encode_one(_asymmetric_molecule())
    permuted = _permute_context_attributes(original, change_marginal=False)
    changed = _permute_context_attributes(original, change_marginal=True)

    batch_a = _batch([original])
    batch_b = _batch([permuted])
    batch_c = _batch([changed])
    with torch.no_grad():
        ch_a = model.encoder.forward_channels(batch_a)
        ch_b = model.encoder.forward_channels(batch_b)
        ch_c = model.encoder.forward_channels(batch_c)

    # Same root + same atom/bond multisets: raw exact marginal and A identical.
    assert torch.equal(ch_a["raw_marginal"], ch_b["raw_marginal"])
    assert torch.allclose(ch_a["attributes"], ch_b["attributes"], atol=0.0, rtol=0.0)

    # Different multiset: raw exact marginal vector must differ.
    differ = (ch_a["raw_marginal"] - ch_c["raw_marginal"]).abs().sum()
    assert float(differ) > 0.0, "distinct categorical marginals must not collide"

    # Every raw entry is an exact non-negative count / indicator.
    raw = ch_a["raw_marginal"]
    assert bool((raw >= 0).all())
    assert bool((raw == raw.round()).all())


def test_a_exact_reconstructs_full_attribute_multiset() -> None:
    """The raw marginal determines the complete categorical multiset B sees."""
    encoded = _encode_one(_asymmetric_molecule_two())
    batch = _batch([encoded])
    raw = v2.raw_marginal_vector(
        batch.struct_atom,
        batch.struct_root,
        batch.struct_patch,
        batch.struct_bond,
        batch.struct_edge_patch,
        int(batch.num_nodes),
    )
    # Reconstruct root / context atom / bond category counts from the raw
    # vector blocks and compare against a direct enumeration of the frame.
    root_block = raw[:, : v2.ATOM_CATEGORIES]
    ctx_block = raw[:, v2.ATOM_CATEGORIES : 2 * v2.ATOM_CATEGORIES]
    bond_block = raw[:, 2 * v2.ATOM_CATEGORIES :]
    patch = batch.struct_patch.long()
    root = batch.struct_root.long()
    atom = batch.struct_atom.long()
    bond = batch.struct_bond.long()
    edge_patch = batch.struct_edge_patch.long()
    for patch_id in range(int(batch.num_nodes)):
        direct_root = torch.bincount(
            atom[(patch == patch_id) & (root > 0)],
            minlength=v2.ATOM_CATEGORIES,
        ).float()
        direct_ctx = torch.bincount(
            atom[(patch == patch_id) & (root == 0)],
            minlength=v2.ATOM_CATEGORIES,
        ).float()
        direct_bond = torch.bincount(
            bond[edge_patch == patch_id], minlength=v2.BOND_CATEGORIES
        ).float()
        assert torch.equal(root_block[patch_id], direct_root)
        assert torch.equal(ctx_block[patch_id], direct_ctx)
        assert torch.equal(bond_block[patch_id], direct_bond)


def test_a_exact_is_blind_to_topology_fields() -> None:
    """A_exact must not read distance, shell or adjacency (src/dst) fields."""
    model = _fresh_model("A")
    original = _encode_one(_asymmetric_molecule())
    scrambled = _clone(original)
    generator = torch.Generator().manual_seed(0)
    scrambled.struct_dist = torch.randint(
        0,
        v2.N_SHELLS,
        scrambled.struct_dist.shape,
        generator=generator,
    )
    scrambled.struct_src = torch.randint(
        0,
        int(scrambled.struct_atom.numel()),
        scrambled.struct_src.shape,
        generator=generator,
    )
    scrambled.struct_dst = torch.randint(
        0,
        int(scrambled.struct_atom.numel()),
        scrambled.struct_dst.shape,
        generator=generator,
    )
    with torch.no_grad():
        a = model.encoder.forward_channels(_batch([original]))["attributes"]
        b = model.encoder.forward_channels(_batch([scrambled]))["attributes"]
    assert torch.equal(a, b), "A_exact must ignore all topology-only fields"


# ---------------------------------------------------------------------------
# explicit basis: chemistry purity / relabel equivariance
# ---------------------------------------------------------------------------


def test_explicit_basis_chemistry_purity() -> None:
    original = _asymmetric_molecule()
    perturbed = _asymmetric_molecule()
    perturbed.x = (perturbed.x + 5) % v2.ATOM_CATEGORIES
    perturbed.edge_attr = (perturbed.edge_attr + 1) % v2.BOND_CATEGORIES
    original_enc = _encode_one(original)
    perturbed_enc = _encode_one(perturbed)

    assert torch.equal(original_enc.node_basis, perturbed_enc.node_basis)
    assert torch.equal(original_enc.edge_basis, perturbed_enc.edge_basis)

    model = _fresh_model("SAE")
    with torch.no_grad():
        s_a = model.encoder.forward_channels(_batch([original_enc]))["structure"]
        s_b = model.encoder.forward_channels(_batch([perturbed_enc]))["structure"]
    assert torch.equal(s_a, s_b), "explicit S must be exactly chemistry-free"


def test_explicit_basis_relabel_equivariance() -> None:
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
        relabeled_edges.append(
            tuple(sorted((inverse[int(left)], inverse[int(right)])))
        )
        relabeled_bonds.append(int(bond))
    order = sorted(range(len(relabeled_edges)), key=lambda index: relabeled_edges[index])
    relabeled = _molecule(
        [molecule.x.view(-1)[old].item() for old in permutation],
        [relabeled_edges[index] for index in order],
        [relabeled_bonds[index] for index in order],
        y=float(molecule.y.view(-1)[0]),
    )
    original_enc = _encode_one(molecule)
    relabeled_enc = _encode_one(relabeled)

    # Node basis is a permutation-equivariant function of the rooted frame:
    # the multiset of rows is invariant under relabeling.
    original_rows = np.sort(original_enc.node_basis.numpy(), axis=0)
    relabeled_rows = np.sort(relabeled_enc.node_basis.numpy(), axis=0)
    assert np.allclose(original_rows, relabeled_rows, atol=1e-6)
    original_edges = np.sort(original_enc.edge_basis.numpy(), axis=0)
    relabeled_edges_sorted = np.sort(relabeled_enc.edge_basis.numpy(), axis=0)
    assert np.allclose(original_edges, relabeled_edges_sorted, atol=1e-6)


def test_explicit_path_has_no_message_passing() -> None:
    model = _fresh_model("SABE")
    encoder = model.encoder
    assert encoder.message is None and encoder.update is None
    assert encoder.root_embedding is None and encoder.distance_embedding is None

    import pathlib

    source = pathlib.Path(v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_explicit_structure":
            target = node
            break
    assert target is not None, "explicit structure routine not found"
    used: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
    forbidden = {"struct_src", "struct_dst", "message", "update"}
    assert not (forbidden & used), (
        f"explicit S must not touch adjacency/message passing: {sorted(forbidden & used)}"
    )


# ---------------------------------------------------------------------------
# assignment sensitivity across A / S / B
# ---------------------------------------------------------------------------


def test_assignment_moves_b_only() -> None:
    original = _encode_one(_asymmetric_molecule())
    permuted = _permute_context_attributes(original, change_marginal=False)
    batch_a = _batch([original])
    batch_b = _batch([permuted])
    for mode in ("SAB", "SABE"):
        model = _fresh_model(mode, seed=3)
        with torch.no_grad():
            ch_a = model.encoder.forward_channels(batch_a)
            ch_b = model.encoder.forward_channels(batch_b)
        assert torch.equal(ch_a["raw_marginal"], ch_b["raw_marginal"])
        assert torch.allclose(
            ch_a["attributes"], ch_b["attributes"], atol=0.0, rtol=0.0
        ), f"{mode}: A must not react to assignment"
        assert torch.equal(ch_a["structure"], ch_b["structure"]), (
            f"{mode}: S must not react to assignment"
        )
        delta = float((ch_a["binding"] - ch_b["binding"]).norm())
        assert delta > 1e-5, f"{mode}: B must react to attribute<->role assignment"


# ---------------------------------------------------------------------------
# inherited contract checks
# ---------------------------------------------------------------------------


def test_no_mixed_bypass_v2() -> None:
    model = _fresh_model("SABE")
    for attribute in (
        "typed_embedding",
        "patch_encoder",
        "global_encoder",
        "parent_embedding",
        "attribute_encoder",
        "structural_encoder",
    ):
        assert not hasattr(model, attribute)
    data = _encode_one(_asymmetric_molecule())
    for attribute in (
        "patch_cont",
        "patch_context",
        "parent_token",
        "global_context",
        "typed_token",
    ):
        assert not hasattr(data, attribute)
    source = (
        __import__("pathlib").Path(v2.__file__).read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    used = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not (FORBIDDEN_FORWARD_NAMES & used)


def test_node_relabel_invariance_v2() -> None:
    for mode in ("SAB", "SABE"):
        model = _fresh_model(mode, seed=5)
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
            relabeled_edges.append(
                tuple(sorted((inverse[int(left)], inverse[int(right)])))
            )
            relabeled_bonds.append(int(bond))
        order = sorted(
            range(len(relabeled_edges)), key=lambda index: relabeled_edges[index]
        )
        relabeled = _molecule(
            [molecule.x.view(-1)[old].item() for old in permutation],
            [relabeled_edges[index] for index in order],
            [relabeled_bonds[index] for index in order],
            y=float(molecule.y.view(-1)[0]),
        )
        with torch.no_grad():
            pred_a = model(_batch([_encode_one(molecule)]))
            pred_b = model(_batch([_encode_one(relabeled)]))
        assert torch.allclose(pred_a, pred_b, atol=1e-5, rtol=0.0), mode


def test_batch_invariance_v2() -> None:
    model = _fresh_model("SABE", seed=7)
    first = _encode_one(_asymmetric_molecule())
    second_mol = _asymmetric_molecule_two()
    second = _encode_one(second_mol)
    with torch.no_grad():
        single_a = model(_batch([first]))
        single_b = model(_batch([second]))
        stacked = model(_batch([first, second]))
    assert torch.allclose(stacked[0], single_a[0], atol=1e-5, rtol=0.0)
    assert torch.allclose(stacked[1], single_b[0], atol=1e-5, rtol=0.0)


def test_gradient_viability_v2() -> None:
    for mode in ("A", "SA", "SAB", "SAM", "SAE", "SABE", "SAME"):
        torch.manual_seed(11)
        model = v2.PatchPathFSARV2Model(mode=mode)
        model.train()
        batch = _batch([_encode_one(_asymmetric_molecule())])
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        loss = torch.nn.functional.l1_loss(
            model(batch).view(-1), batch.y.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = torch.nn.functional.l1_loss(
            model(batch).view(-1), batch.y.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        checks = {
            "A_exact": model.encoder.a_exact_mlp[0].weight,
            "relation_core": model.relation_encoder[0].weight,
            "center_update": model.center_update[0].weight,
            "head": model.head[-1].weight,
        }
        if model.encoder.include_structure:
            checks["S"] = model.encoder.structure_pool[0].weight
        if model.encoder.include_binding:
            checks["B"] = model.encoder.node_role_projection.weight
        if model.capacity_mlp is not None:
            checks["M"] = model.capacity_mlp[0].weight
        for name, parameter in checks.items():
            grad = parameter.grad
            assert grad is not None and bool(torch.isfinite(grad).all()), (
                f"{mode}:{name} gradient missing/non-finite"
            )
            assert float(grad.abs().sum()) > 0.0, f"{mode}:{name} zero gradient"


def test_parameter_accounting_v2() -> None:
    totals = {}
    breakdowns = {}
    for mode in v2.ALL_MODES:
        model = v2.PatchPathFSARV2Model(mode=mode)
        breakdown = v2.parameter_breakdown(model)
        totals[mode] = breakdown["total"]
        breakdowns[mode] = breakdown
        assert breakdown["total"] <= 200_000
        assert breakdown["dataset_dependent_vocabulary_params"] == 0
    assert totals["A"] < totals["SA"] < totals["SAB"]
    assert totals["SAE"] < totals["SABE"]
    # SAM / SAME marginal-only capacity control: the *total* parameter count
    # (node-init included) must match the aligned organ mode to < 1 %, so the
    # control tests aligned access rather than raw parameter gain.
    for control, organ in (("SAM", "SAB"), ("SAME", "SABE")):
        total_gap = abs(totals[control] - totals[organ])
        assert total_gap <= max(1, int(0.01 * totals[organ])), (
            f"{control} vs {organ} total parameter mismatch {total_gap}"
        )
        # The marginal-only MLP is present and reads only the pooled marginals.
        assert breakdowns[control]["capacity_mlp"] > 0
        assert breakdowns[control]["B_encoder"] == 0
