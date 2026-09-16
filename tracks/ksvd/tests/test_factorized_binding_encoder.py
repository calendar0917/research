"""Correctness tests for the Factorized Structure--Attribute Binding encoder.

Fast unit / static tests only: no training, no official valid, no official test.

Coverage (numbered as in the FSAB brief, section 19):

A. node relabel invariance (path / branch / triangle / 4-cycle)
B. structure purity: S exactly unchanged when all atom/bond attributes change,
   plus a source-region audit that the S path never reads attribute tensors
C. attribute marginal purity: A unchanged under node/edge attribute assignment
   changes and adjacency rearrangement, plus a source-region audit that the A
   path never reads root / distance / src / dst
D. binding assignment sensitivity: same topology / marginals, different
   attribute -> role assignment gives S same, A same, B different
E. attribute permutation null: max |S_true - S_shuffle| ~ 0 and
   max |A_true - A_shuffle| ~ 0 over random within-patch permutations
F. edge binding correctness: bond-type assignment shuffle keeps the bond
   marginal A unchanged and is allowed to change the edge binding
G. batch invariance: single patch vs batched patches
H. gradient viability for the S / A / B-node / B-edge / fusion paths
I. no prohibited losses / teacher / attention (AST audit)

plus a target-free parameter-accounting check (no vocabulary-sized lookup).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    structural_patch_encoder as spe,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_factorized_binding_encoder as fsab,
)

# ---------------------------------------------------------------------------
# graph fixtures (single patch, root = 0)
# ---------------------------------------------------------------------------

GRAPHS = {
    "path3": dict(
        atom=[0, 1, 2], dist=[0, 1, 2], edges=[(0, 1), (1, 2)], bonds=[0, 1]
    ),
    "branch": dict(
        atom=[3, 1, 4, 2, 0],
        dist=[0, 1, 1, 2, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
        bonds=[1, 2, 0, 3],
    ),
    "triangle": dict(
        atom=[0, 5, 2],
        dist=[0, 1, 1],
        edges=[(0, 1), (0, 2), (1, 2)],
        bonds=[0, 1, 2],
    ),
    "four_cycle": dict(
        atom=[1, 0, 4, 2],
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        bonds=[3, 2, 1, 0],
    ),
}


def _encoder(seed: int = 0) -> spe.FactorizedStructureAttributeBindingEncoder:
    torch.manual_seed(int(seed))
    return spe.FactorizedStructureAttributeBindingEncoder(
        atom_categories=28,
        bond_categories=4,
        n_distance_bins=3,
        role_dim=16,
        s_dim=8,
        a_dim=8,
        b_dim=8,
        attr_dim=8,
        s_hidden=16,
        a_hidden=16,
        b_hidden=16,
        fusion_hidden=16,
        rounds=2,
        output_dim=16,
        activation="silu",
    )


def _patch(
    name: str = "branch",
    *,
    atom=None,
    dist=None,
    edges=None,
    bonds=None,
    patch_ids=None,
) -> fsab._FSABBatch:
    spec = GRAPHS[name]
    atom = list(spec["atom"] if atom is None else atom)
    dist = list(spec["dist"] if dist is None else dist)
    edges = list(spec["edges"] if edges is None else edges)
    bonds = list(spec["bonds"] if bonds is None else bonds)
    n = len(atom)
    root = [0] * n
    root[0] = 1
    both = list(edges) + [(v, u) for u, v in edges]
    src = torch.tensor([u for u, _ in both], dtype=torch.long)
    dst = torch.tensor([v for _, v in both], dtype=torch.long)
    # every undirected bond keeps the same type in both directions
    edge_bond: list[int] = list(bonds) + list(bonds)
    patch = torch.zeros(n, dtype=torch.long) if patch_ids is None else torch.as_tensor(patch_ids)
    return fsab._FSABBatch(
        atom=torch.tensor(atom, dtype=torch.long),
        root=torch.tensor(root, dtype=torch.long),
        dist=torch.tensor(dist, dtype=torch.long),
        patch=patch,
        edge_patch=torch.zeros(len(both), dtype=torch.long),
        bond=torch.tensor(edge_bond, dtype=torch.long),
        src=src,
        dst=dst,
    )


def _channels(encoder, batch):
    encoder.eval()
    with torch.no_grad():
        return encoder.forward_channels(batch)


# ---------------------------------------------------------------------------
# A. node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariance_all_fixtures():
    encoder = _encoder(0)
    for name in GRAPHS:
        batch = _patch(name)
        n = int(batch.struct_atom.numel())
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(101 + n))
        relabelled = fsab._relabel(batch, perm)
        reference = _channels(encoder, batch)
        moved = _channels(encoder, relabelled)
        for key in ("structure", "attributes", "binding", "output"):
            delta = float((reference[key] - moved[key]).abs().max())
            assert delta < 1.0e-5, (name, key, delta)


def test_directed_edge_order_and_direction_invariance():
    encoder = _encoder(1)
    batch = _patch("four_cycle")
    n_edges = int(batch.struct_bond.numel())
    perm = torch.randperm(n_edges, generator=torch.Generator().manual_seed(7))
    reordered = fsab._reorder_directed_edges(batch, perm)
    reference = _channels(encoder, batch)
    moved = _channels(encoder, reordered)
    for key in ("structure", "attributes", "binding", "output"):
        delta = float((reference[key] - moved[key]).abs().max())
        assert delta < 1.0e-5, (key, delta)


# ---------------------------------------------------------------------------
# B. structure purity
# ---------------------------------------------------------------------------


def test_structure_channel_ignores_attributes():
    encoder = _encoder(2)
    batch = _patch("branch")
    changed = fsab._FSABBatch(
        atom=torch.full_like(batch.struct_atom, 7),
        root=batch.struct_root.clone(),
        dist=batch.struct_dist.clone(),
        patch=batch.struct_patch.clone(),
        edge_patch=batch.struct_edge_patch.clone(),
        bond=torch.full_like(batch.struct_bond, 3),
        src=batch.struct_src.clone(),
        dst=batch.struct_dst.clone(),
    )
    reference = _channels(encoder, batch)
    other = _channels(encoder, changed)
    assert float((reference["structure"] - other["structure"]).abs().max()) == 0.0
    # ... while the attribute channel does react.
    assert float((reference["attributes"] - other["attributes"]).abs().max()) > 1.0e-6


def test_structure_path_does_not_read_attribute_tensors():
    source = inspect.getsource(
        spe.FactorizedStructureAttributeBindingEncoder.forward_channels
    )
    marker_s = "# ---------------- structure stream S"
    marker_a = "# -------------- attribute stream A"
    assert marker_s in source and marker_a in source
    structure_region = source.split(marker_s)[1].split(marker_a)[0]
    for token in ("struct_atom", "struct_bond"):
        assert token not in structure_region, token


# ---------------------------------------------------------------------------
# C. attribute marginal purity
# ---------------------------------------------------------------------------


def test_attribute_channel_ignores_root_distance_and_adjacency():
    encoder = _encoder(3)
    batch = _patch("four_cycle")
    # same atom / bond multisets, different root and distance labelling
    moved = fsab._FSABBatch(
        atom=batch.struct_atom.clone(),
        root=torch.tensor([0, 1, 0, 0], dtype=torch.long),
        dist=torch.tensor([2, 0, 2, 1], dtype=torch.long),
        patch=batch.struct_patch.clone(),
        edge_patch=batch.struct_edge_patch.clone(),
        bond=batch.struct_bond.clone(),
        src=batch.struct_src.clone(),
        dst=batch.struct_dst.clone(),
    )
    reference = _channels(encoder, batch)
    other = _channels(encoder, moved)
    assert float((reference["attributes"] - other["attributes"]).abs().max()) < 1.0e-6
    # the structure channel must react to the topology change
    assert float((reference["structure"] - other["structure"]).abs().max()) > 1.0e-6


def test_attribute_channel_is_assignment_invariant():
    encoder = _encoder(4)
    batch = _patch("branch")
    # permute atom labels and bond labels within the patch: multisets preserved
    perm = torch.tensor([4, 3, 1, 2, 0])
    shuffled_atom = batch.struct_atom[perm]
    edge_perm = torch.tensor([6, 7, 4, 5, 0, 1, 2, 3])
    shuffled_bond = batch.struct_bond[edge_perm]
    moved = fsab._FSABBatch(
        atom=shuffled_atom,
        root=batch.struct_root.clone(),
        dist=batch.struct_dist.clone(),
        patch=batch.struct_patch.clone(),
        edge_patch=batch.struct_edge_patch.clone(),
        bond=shuffled_bond,
        src=batch.struct_src.clone(),
        dst=batch.struct_dst.clone(),
    )
    reference = _channels(encoder, batch)
    other = _channels(encoder, moved)
    assert float((reference["attributes"] - other["attributes"]).abs().max()) < 1.0e-5
    assert float((reference["binding"] - other["binding"]).abs().max()) > 1.0e-6


def test_attribute_path_does_not_read_topology_tensors():
    source = inspect.getsource(
        spe.FactorizedStructureAttributeBindingEncoder.forward_channels
    )
    marker_a = "# -------------- attribute stream A"
    marker_b = "# --------- binding stream B"
    assert marker_a in source and marker_b in source
    attribute_region = source.split(marker_a)[1].split(marker_b)[0]
    for token in ("struct_root", "struct_dist", "struct_src", "struct_dst"):
        assert token not in attribute_region, token


# ---------------------------------------------------------------------------
# D. binding assignment sensitivity (the architecture witness)
# ---------------------------------------------------------------------------


def test_binding_witness_same_topology_same_marginals():
    encoder = _encoder(5)
    reference, node_swap, bond_swap = fsab._binding_witness_pair()
    for candidate, label in ((node_swap, "node"), (bond_swap, "bond")):
        assert sorted(reference.struct_atom.tolist()) == sorted(
            candidate.struct_atom.tolist()
        )
        assert sorted(reference.struct_bond.tolist()) == sorted(
            candidate.struct_bond.tolist()
        )
        assert torch.equal(reference.struct_src, candidate.struct_src)
        assert torch.equal(reference.struct_dst, candidate.struct_dst)
        assert torch.equal(reference.struct_dist, candidate.struct_dist)
        assert torch.equal(reference.struct_root, candidate.struct_root)
        left = _channels(encoder, reference)
        right = _channels(encoder, candidate)
        assert float((left["structure"] - right["structure"]).abs().max()) == 0.0, label
        assert (
            float((left["attributes"] - right["attributes"]).abs().max()) < 1.0e-6
        ), label
        assert (
            float((left["binding"] - right["binding"]).abs().max()) > 1.0e-6
        ), label


# ---------------------------------------------------------------------------
# E. attribute permutation null (large-sample random audit)
# ---------------------------------------------------------------------------


def test_attribute_permutation_null_over_random_permutations():
    encoder = _encoder(6)
    # path3 has three structurally distinct roles (root / dist-1 / dist-2), so
    # every non-identity attribute permutation moves an attribute to a
    # different structural role.
    batch = _patch("path3")
    reference = _channels(encoder, batch)
    n = int(batch.struct_atom.numel())
    max_s = 0.0
    max_a = 0.0
    min_b = float("inf")
    for seed in range(32):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        if torch.equal(perm, torch.arange(n)):
            continue
        shuffled = fsab._FSABBatch(
            atom=batch.struct_atom[perm],
            root=batch.struct_root.clone(),
            dist=batch.struct_dist.clone(),
            patch=batch.struct_patch.clone(),
            edge_patch=batch.struct_edge_patch.clone(),
            bond=batch.struct_bond.clone(),
            src=batch.struct_src.clone(),
            dst=batch.struct_dst.clone(),
        )
        other = _channels(encoder, shuffled)
        max_s = max(max_s, float((reference["structure"] - other["structure"]).abs().max()))
        max_a = max(max_a, float((reference["attributes"] - other["attributes"]).abs().max()))
        min_b = min(min_b, float((reference["binding"] - other["binding"]).abs().max()))
    assert max_s == 0.0
    assert max_a < 1.0e-5
    assert min_b > 1.0e-6


def test_binding_is_orbit_symmetric():
    """Swapping attributes between structurally equivalent nodes is a no-op.

    Nodes 1 and 2 of the ``branch`` fixture are structurally equivalent (both
    at distance 1 from the root with an identical single child), so the
    role--attribute correspondence multiset is unchanged by exchanging their
    attributes.
    """
    encoder = _encoder(11)
    batch = _patch("branch")
    perm = torch.tensor([0, 2, 1, 3, 4])
    swapped = fsab._FSABBatch(
        atom=batch.struct_atom[perm],
        root=batch.struct_root.clone(),
        dist=batch.struct_dist.clone(),
        patch=batch.struct_patch.clone(),
        edge_patch=batch.struct_edge_patch.clone(),
        bond=batch.struct_bond.clone(),
        src=batch.struct_src.clone(),
        dst=batch.struct_dst.clone(),
    )
    reference = _channels(encoder, batch)
    other = _channels(encoder, swapped)
    assert float((reference["structure"] - other["structure"]).abs().max()) == 0.0
    assert float((reference["attributes"] - other["attributes"]).abs().max()) < 1.0e-6
    assert float((reference["binding"] - other["binding"]).abs().max()) < 1.0e-5


def test_within_group_rotation_preserves_every_multiset():
    group = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    values = np.arange(group.shape[0], dtype=np.int64)
    for seed in range(16):
        source = fsab._within_group_rotation(group, seed)
        assert sorted(source.tolist()) == list(range(group.shape[0]))
        moved = values[source]
        for group_id in np.unique(group):
            mask = group == group_id
            assert sorted(moved[mask].tolist()) == sorted(values[mask].tolist())


# ---------------------------------------------------------------------------
# F. edge binding correctness
# ---------------------------------------------------------------------------


def test_edge_binding_keeps_bond_marginal_and_reacts_to_assignment():
    encoder = _encoder(7)
    batch = _patch("branch")
    reference = _channels(encoder, batch)
    edge_patch = batch.struct_edge_patch.cpu().numpy().astype(np.int64)
    source = fsab._within_group_rotation(edge_patch, 12345)
    moved = fsab._FSABBatch(
        atom=batch.struct_atom.clone(),
        root=batch.struct_root.clone(),
        dist=batch.struct_dist.clone(),
        patch=batch.struct_patch.clone(),
        edge_patch=batch.struct_edge_patch.clone(),
        bond=batch.struct_bond[torch.from_numpy(source)],
        src=batch.struct_src.clone(),
        dst=batch.struct_dst.clone(),
    )
    other = _channels(encoder, moved)
    assert float((reference["attributes"] - other["attributes"]).abs().max()) < 1.0e-5
    assert float((reference["binding"] - other["binding"]).abs().max()) > 1.0e-7


# ---------------------------------------------------------------------------
# G. batch invariance
# ---------------------------------------------------------------------------


def test_batch_invariance_single_vs_batched():
    encoder = _encoder(8)
    first = _patch("path3")
    second = _patch("triangle")
    n_first = int(first.struct_atom.numel())
    n_edge_first = int(first.struct_bond.numel())

    def _offset(batch, node_offset, patch_id):
        return fsab._FSABBatch(
            atom=batch.struct_atom.clone(),
            root=batch.struct_root.clone(),
            dist=batch.struct_dist.clone(),
            patch=torch.full_like(batch.struct_patch, patch_id),
            edge_patch=torch.full_like(batch.struct_edge_patch, patch_id),
            bond=batch.struct_bond.clone(),
            src=batch.struct_src.clone() + node_offset,
            dst=batch.struct_dst.clone() + node_offset,
        )

    batched = fsab._FSABBatch(
        atom=torch.cat([first.struct_atom, second.struct_atom]),
        root=torch.cat([first.struct_root, second.struct_root]),
        dist=torch.cat([first.struct_dist, second.struct_dist]),
        patch=torch.cat(
            [
                torch.zeros(n_first, dtype=torch.long),
                torch.ones(int(second.struct_atom.numel()), dtype=torch.long),
            ]
        ),
        # edge_patch is a *global* patch index here, matching the collate
        # convention (patch ids are globally unique inside a batch).
        edge_patch=torch.cat(
            [
                torch.zeros(n_edge_first, dtype=torch.long),
                torch.ones(int(second.struct_bond.numel()), dtype=torch.long),
            ]
        ),
        bond=torch.cat([first.struct_bond, second.struct_bond]),
        src=torch.cat([first.struct_src, second.struct_src + n_first]),
        dst=torch.cat([first.struct_dst, second.struct_dst + n_first]),
    )
    single_first = _channels(encoder, first)["output"]
    single_second = _channels(encoder, second)["output"]
    batched_out = _channels(encoder, batched)["output"]
    assert int(batched_out.shape[0]) == 2
    assert float((batched_out[0] - single_first).abs().max()) < 1.0e-5
    assert float((batched_out[1] - single_second).abs().max()) < 1.0e-5


# ---------------------------------------------------------------------------
# H. gradient viability
# ---------------------------------------------------------------------------


def test_gradient_viability_all_paths():
    encoder = _encoder(9)
    batch = _patch("branch")
    channels = encoder.forward_channels(batch)
    loss = channels["output"].pow(2).mean() + channels["binding"].pow(2).mean()
    loss.backward()
    groups = {
        "structure": [
            encoder.message,
            encoder.update,
            encoder.structure_pool,
        ],
        "attribute": [encoder.atom_mlp, encoder.bond_mlp, encoder.attribute_fuse],
        "binding_node": [
            encoder.node_role_projection,
            encoder.node_attribute_projection,
        ],
        "binding_edge": [
            encoder.edge_role_mlp,
            encoder.edge_attribute_mlp,
            encoder.edge_role_projection,
            encoder.edge_attribute_projection,
        ],
        "fusion": [
            encoder.structure_weight,
            encoder.attribute_weight,
            encoder.binding_weight,
            encoder.fusion_mlp,
        ],
    }
    for name, modules in groups.items():
        total = 0.0
        for module in modules:
            for parameter in module.parameters():
                assert parameter.grad is not None, (name, "missing grad")
                assert bool(torch.isfinite(parameter.grad).all()), name
                total += float(parameter.grad.abs().sum())
        assert total > 0.0, name


# ---------------------------------------------------------------------------
# I. static audit: no attention / prohibited losses / teacher
# ---------------------------------------------------------------------------


def test_static_audit_no_attention_or_auxiliary_losses():
    encoder_source = inspect.getsource(
        spe.FactorizedStructureAttributeBindingEncoder
    )
    runner_source = Path(fsab.__file__).read_text(encoding="utf-8")
    encoder_calls = fsab._called_names(encoder_source)
    runner_calls = fsab._called_names(runner_source)
    runner_losses = runner_calls & fsab._LOSS_CALL_NAMES
    assert not (encoder_calls & fsab._LOSS_CALL_NAMES)
    assert not (encoder_calls & fsab._ATTENTION_CALL_NAMES)
    assert not (runner_calls & fsab._ATTENTION_CALL_NAMES)
    assert not (encoder_calls & fsab._DISTILLATION_CALL_NAMES)
    assert not (runner_calls & fsab._DISTILLATION_CALL_NAMES)
    assert runner_losses == {"l1_loss"}


def test_no_softmax_operator_in_encoder_code():
    """The encoder *code* (docstrings stripped) contains no attention tokens."""
    import ast
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(spe.FactorizedStructureAttributeBindingEncoder)
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ):
                node.body = body[1:] or [ast.Pass()]
    code_only = ast.unparse(tree)
    assert not re.search(r"softmax|attention|multihead", code_only, re.I)


# ---------------------------------------------------------------------------
# parameter accounting (target-free)
# ---------------------------------------------------------------------------


def test_parameter_accounting_no_vocabulary_lookup():
    payload = fsab.params()
    assert payload["dataset_dependent_vocabulary_params"] == 0
    assert payload["candidate_has_typed_embedding"] is False
    assert payload["candidate_typed_state_keys"] == []
    assert payload["params_in_range"] is True
    breakdown = payload["fsab_stream_breakdown"]
    assert set(breakdown) == {
        "structure_stream",
        "attribute_stream",
        "binding_stream",
        "fusion",
    }
    assert sum(breakdown.values()) == payload["fsab_encoder_params"]
